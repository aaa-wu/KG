"""FastAPI 后端：从 Neo4j 读取图数据并返回 3D 可视化所需的 JSON"""
import os
from typing import Optional
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from src.config import get_neo4j_driver
from src.models.schema import (
    LABEL_MAJOR, LABEL_COURSE, LABEL_KNOWLEDGE_POINT,
    REL_BELONGS_TO, REL_COVERS, REL_PREREQUISITE_OF, REL_PREREQUISITE_FOR,
)
from src.qa import answer_question
from src.recommendation.path_finder import find_prerequisites

app = FastAPI(title="KG 3D Visualizer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 节点类型 → 颜色映射（给前端用）
LABEL_COLORS = {
    LABEL_MAJOR: "#00e5ff",
    LABEL_COURSE: "#26c6da",
    LABEL_KNOWLEDGE_POINT: "#ffb74d",
}

# 节点类型 → 大小
LABEL_SIZES = {
    LABEL_MAJOR: 12,
    LABEL_COURSE: 8,
    LABEL_KNOWLEDGE_POINT: 5,
}

# 关系类型 → 颜色
REL_COLORS = {
    REL_BELONGS_TO: "#00e5ff",
    REL_COVERS: "#26c6da",
    REL_PREREQUISITE_OF: "#00e5ff",
    REL_PREREQUISITE_FOR: "#ffb74d",
    "RELATED_TO": "#dfe6e9",
}


class QARequest(BaseModel):
    question: str
    known: list[str] = Field(default_factory=list)
    goal: Optional[str] = None
    weekly_hours: Optional[float] = None
    level: Optional[str] = None


def _make_node_id(label: str, name: str, extra: str = "") -> str:
    """生成唯一节点 ID"""
    parts = [label, name]
    if extra:
        parts.append(extra)
    return "::".join(parts)


def _major_icon_key(name: str) -> str:
    if any(key in name for key in ("人工智能", "计算机", "软件", "数据")):
        return "compute"
    if any(key in name for key in ("金融", "经济", "工商", "管理", "人力资源")):
        return "market"
    if any(key in name for key in ("化学", "物理", "数学", "统计")):
        return "science"
    if any(key in name for key in ("法学", "政治", "行政")):
        return "civic"
    if any(key in name for key in ("文学", "英语", "汉语言", "新闻", "历史", "哲学")):
        return "humanities"
    if any(key in name for key in ("社会", "心理")):
        return "society"
    if "档案" in name:
        return "archive"
    return "general"


def _course_key(course: dict) -> str:
    return course.get("id") or _make_node_id(LABEL_COURSE, course.get("name", ""))


def _infer_course_stage(
    course: dict,
    knowledge_count: int,
    prereq_in: int,
    prereq_out: int,
    index: int,
    total: int,
) -> str:
    name = (course.get("name") or "").lower()
    foundation_terms = (
        "导论", "概论", "基础", "数学", "线性代数", "概率", "高等数学",
        "programming", "calculus", "introduction", "foundation", "basics",
    )
    practice_terms = (
        "实践", "实习", "实验", "项目", "设计", "研讨", "专题",
        "practice", "project", "lab", "seminar", "internship", "workshop",
    )
    advanced_terms = (
        "高级", "深度", "机器学习", "优化", "强化", "系统", "模型",
        "advanced", "deep", "machine learning", "optimization", "systems",
    )

    if prereq_out and not prereq_in:
        return "foundation"
    if prereq_in and prereq_out:
        return "core"
    if prereq_in and not prereq_out:
        return "practice" if any(term in name for term in practice_terms) else "advanced"
    if any(term in name for term in foundation_terms):
        return "foundation"
    if any(term in name for term in practice_terms):
        return "practice"
    if any(term in name for term in advanced_terms) or knowledge_count >= 6:
        return "core"
    if total:
        ratio = index / max(total - 1, 1)
        if ratio < 0.28:
            return "foundation"
        if ratio < 0.68:
            return "core"
        return "advanced"
    return "core"


def _stage_label(stage: str) -> str:
    return {
        "foundation": "基础",
        "core": "核心",
        "advanced": "进阶",
        "practice": "实践",
    }.get(stage, stage)


@app.get("/api/majors")
def get_majors():
    """获取专业入口列表及轻量统计"""
    driver = get_neo4j_driver()
    with driver.session() as session:
        rows = session.run(
            f"""
            MATCH (m:{LABEL_MAJOR})
            OPTIONAL MATCH (m)-[:{REL_BELONGS_TO}]->(c:{LABEL_COURSE})
            OPTIONAL MATCH (c)-[:{REL_COVERS}]->(k:{LABEL_KNOWLEDGE_POINT})
            RETURN properties(m) AS major,
                   count(DISTINCT c) AS course_count,
                   count(DISTINCT k) AS knowledge_count
            ORDER BY major.name
            """
        )
        majors = []
        for row in rows:
            major = dict(row["major"])
            name = major.get("name", "")
            majors.append({
                "id": major.get("id") or _make_node_id(LABEL_MAJOR, name, major.get("university", "")),
                "name": name,
                "course_count": row["course_count"],
                "knowledge_count": row["knowledge_count"],
                "icon_key": _major_icon_key(name),
                "properties": major,
            })
    return majors


@app.get("/api/majors/{major_name}/roadmap")
def get_major_roadmap(
    major_name: str,
    limit: int = Query(default=32, ge=4, le=260),
    include_all: bool = Query(default=False),
):
    """获取某个专业的课程路线图及课程覆盖知识点"""
    driver = get_neo4j_driver()
    with driver.session() as session:
        rows = list(session.run(
            f"""
            MATCH (m:{LABEL_MAJOR} {{name: $major_name}})-[:{REL_BELONGS_TO}]->(c:{LABEL_COURSE})
            OPTIONAL MATCH (c)-[:{REL_COVERS}]->(k:{LABEL_KNOWLEDGE_POINT})
            RETURN properties(m) AS major,
                   properties(c) AS course,
                   collect(DISTINCT properties(k)) AS knowledge
            ORDER BY course.name
            """,
            major_name=major_name,
        ))
        if not rows:
            raise HTTPException(status_code=404, detail=f"未找到专业：{major_name}")

        major = dict(rows[0]["major"])
        raw_courses = []
        for row in rows:
            course = dict(row["course"])
            knowledge = [dict(item) for item in row["knowledge"] if item]
            raw_courses.append({
                "id": _course_key(course),
                "name": course.get("name", ""),
                "properties": course,
                "knowledge_points": [
                    {
                        "id": item.get("id") or _make_node_id(LABEL_KNOWLEDGE_POINT, item.get("name", "")),
                        "name": item.get("name", ""),
                    }
                    for item in knowledge
                ],
                "knowledge_count": len(knowledge),
            })

        course_ids = [course["id"] for course in raw_courses]
        course_id_set = set(course_ids)
        prereq_rows = list(session.run(
            f"""
            MATCH (a:{LABEL_COURSE})-[r:{REL_PREREQUISITE_FOR}]->(b:{LABEL_COURSE})
            WHERE (a.id IN $course_ids OR a.name IN $course_names)
              AND (b.id IN $course_ids OR b.name IN $course_names)
            RETURN properties(a) AS source, properties(b) AS target, type(r) AS rel_type
            """,
            course_ids=[course["properties"].get("id") for course in raw_courses],
            course_names=[course["name"] for course in raw_courses],
        ))

    prereq_links = []
    prereq_in = {course_id: 0 for course_id in course_ids}
    prereq_out = {course_id: 0 for course_id in course_ids}
    for row in prereq_rows:
        source = _course_key(dict(row["source"]))
        target = _course_key(dict(row["target"]))
        if source not in course_id_set or target not in course_id_set:
            continue
        prereq_links.append({
            "source": source,
            "target": target,
            "type": row["rel_type"],
        })
        prereq_out[source] += 1
        prereq_in[target] += 1

    total_courses = len(raw_courses)
    sorted_courses = sorted(
        raw_courses,
        key=lambda item: (
            -int(prereq_in[item["id"]] + prereq_out[item["id"]] > 0),
            -item["knowledge_count"],
            item["name"],
        ),
    )
    display_courses = sorted_courses if include_all else sorted_courses[:limit]
    display_ids = {course["id"] for course in display_courses}
    display_links = [
        link for link in prereq_links
        if link["source"] in display_ids and link["target"] in display_ids
    ]

    staged_courses = []
    for index, course in enumerate(display_courses):
        stage = _infer_course_stage(
            course["properties"],
            course["knowledge_count"],
            prereq_in[course["id"]],
            prereq_out[course["id"]],
            index,
            len(display_courses),
        )
        staged_courses.append({**course, "stage": stage, "stage_label": _stage_label(stage)})

    stage_order = ["foundation", "core", "advanced", "practice"]
    stages = [
        {
            "key": stage,
            "label": _stage_label(stage),
            "courses": [course for course in staged_courses if course["stage"] == stage],
        }
        for stage in stage_order
    ]

    return {
        "major": {
            "id": major.get("id") or _make_node_id(LABEL_MAJOR, major.get("name", ""), major.get("university", "")),
            "name": major.get("name", major_name),
            "icon_key": _major_icon_key(major.get("name", major_name)),
            "properties": major,
        },
        "total_courses": total_courses,
        "displayed_courses": len(staged_courses),
        "total_prerequisite_links": len(prereq_links),
        "include_all": include_all,
        "stages": stages,
        "courses": staged_courses,
        "links": display_links,
    }


@app.get("/api/graph")
def get_full_graph(limit: int = Query(default=3000, description="最大节点数")):
    """获取全量图谱数据"""
    driver = get_neo4j_driver()
    with driver.session() as session:
        nodes_result = session.run(
            f"""
            MATCH (n)
            WHERE n:{LABEL_MAJOR} OR n:{LABEL_COURSE} OR n:{LABEL_KNOWLEDGE_POINT}
            RETURN labels(n)[0] AS label, properties(n) AS props
            LIMIT $limit
            """,
            limit=limit,
        )
        nodes = []
        for r in nodes_result:
            label = r["label"]
            props = r["props"]
            node_id = _make_node_id(label, props.get("name", ""), props.get("university", ""))
            nodes.append({
                "id": node_id,
                "name": props.get("name", ""),
                "label": label,
                "color": LABEL_COLORS.get(label, "#cccccc"),
                "size": LABEL_SIZES.get(label, 5),
                "properties": props,
            })

        # 获取所有关系
        link_limit = max(limit * 2, 500)
        links_result = session.run(
            f"""
            MATCH (a)-[r]->(b)
            WHERE (a:{LABEL_MAJOR} OR a:{LABEL_COURSE} OR a:{LABEL_KNOWLEDGE_POINT})
              AND (b:{LABEL_MAJOR} OR b:{LABEL_COURSE} OR b:{LABEL_KNOWLEDGE_POINT})
            RETURN labels(a)[0] AS label_a, properties(a) AS props_a,
                   labels(b)[0] AS label_b, properties(b) AS props_b,
                   type(r) AS rel_type
            LIMIT $link_limit
            """,
            link_limit=link_limit,
        )
        node_ids = {n["id"] for n in nodes}
        links = []
        for r in links_result:
            src_id = _make_node_id(
                r["label_a"], r["props_a"].get("name", ""),
                r["props_a"].get("university", "")
            )
            tgt_id = _make_node_id(
                r["label_b"], r["props_b"].get("name", ""),
                r["props_b"].get("university", "")
            )
            if src_id in node_ids and tgt_id in node_ids:
                links.append({
                    "source": src_id,
                    "target": tgt_id,
                    "type": r["rel_type"],
                    "color": REL_COLORS.get(r["rel_type"], "#cccccc"),
                })

    return {"nodes": nodes, "links": links}


@app.get("/api/graph/neighbors/{node_label}/{node_name}")
def get_neighbors(
    node_label: str,
    node_name: str,
    depth: int = Query(default=1, description="扩展深度"),
):
    """获取指定节点及其邻居"""
    driver = get_neo4j_driver()
    with driver.session() as session:
        result = session.run(
            f"""
            MATCH (center:{node_label} {{name: $name}})
            MATCH path = (center)-[*1..{depth}]-(neighbor)
            WHERE neighbor:{LABEL_MAJOR} OR neighbor:{LABEL_COURSE} OR neighbor:{LABEL_KNOWLEDGE_POINT}
            RETURN DISTINCT labels(neighbor)[0] AS label,
                   properties(neighbor) AS props,
                   CASE WHEN neighbor = center THEN true ELSE false END AS is_center
            UNION
            MATCH (center:{node_label} {{name: $name}})
            RETURN labels(center)[0] AS label,
                   properties(center) AS props,
                   true AS is_center
            """,
            name=node_name,
            depth=depth,
        )
        nodes = []
        for r in result:
            label = r["label"]
            props = r["props"]
            node_id = _make_node_id(label, props.get("name", ""), props.get("university", ""))
            nodes.append({
                "id": node_id,
                "name": props.get("name", ""),
                "label": label,
                "color": LABEL_COLORS.get(label, "#cccccc"),
                "size": LABEL_SIZES.get(label, 5) * (2 if r["is_center"] else 1),
                "is_center": r["is_center"],
                "properties": props,
            })

        node_ids = {n["id"] for n in nodes}
        # 获取这些节点之间的关系
        links_result = session.run(
            f"""
            MATCH (a)-[r]->(b)
            WHERE (a:{LABEL_MAJOR} OR a:{LABEL_COURSE} OR a:{LABEL_KNOWLEDGE_POINT})
              AND (b:{LABEL_MAJOR} OR b:{LABEL_COURSE} OR b:{LABEL_KNOWLEDGE_POINT})
            RETURN labels(a)[0] AS la, properties(a) AS pa,
                   labels(b)[0] AS lb, properties(b) AS pb,
                   type(r) AS rel_type
            LIMIT 1000
            """
        )
        links = []
        for r in links_result:
            src_id = _make_node_id(
                r["la"], r["pa"].get("name", ""), r["pa"].get("university", "")
            )
            tgt_id = _make_node_id(
                r["lb"], r["pb"].get("name", ""), r["pb"].get("university", "")
            )
            if src_id in node_ids and tgt_id in node_ids:
                links.append({
                    "source": src_id,
                    "target": tgt_id,
                    "type": r["rel_type"],
                    "color": REL_COLORS.get(r["rel_type"], "#cccccc"),
                })

    return {"nodes": nodes, "links": links}


@app.get("/api/graph/prerequisites/{kp_name}")
def get_prerequisite_chain(kp_name: str):
    """获取知识点的前驱学习路径"""
    driver = get_neo4j_driver()
    with driver.session() as session:
        path_data = find_prerequisites(session, kp_name)

    if "error" in path_data:
        return {"error": path_data["error"]}

    # 将阶段数据转为前端可用的图结构
    nodes = []
    links = []
    seen = set()

    for stage in path_data.get("stages", []):
        for kp_name_in_stage in stage["knowledge_points"]:
            node_id = _make_node_id(LABEL_KNOWLEDGE_POINT, kp_name_in_stage)
            if node_id not in seen:
                seen.add(node_id)
                nodes.append({
                    "id": node_id,
                    "name": kp_name_in_stage,
                    "label": LABEL_KNOWLEDGE_POINT,
                    "color": LABEL_COLORS[LABEL_KNOWLEDGE_POINT],
                    "size": 5 + stage["stage"],
                    "stage": stage["stage"],
                    "properties": {"stage": stage["stage"]},
                })

    # 先获取这些知识点之间的 PREREQUISITE_OF 关系
    kp_names = [n["name"] for n in nodes]
    if kp_names:
        driver2 = get_neo4j_driver()
        with driver2.session() as session:
            edge_result = session.run(
                f"""
                MATCH (a:{LABEL_KNOWLEDGE_POINT})-[r:{REL_PREREQUISITE_OF}]->(b:{LABEL_KNOWLEDGE_POINT})
                WHERE a.name IN $names AND b.name IN $names
                RETURN a.name AS src, b.name AS dst
                """,
                names=kp_names,
            )
            for r in edge_result:
                src_id = _make_node_id(LABEL_KNOWLEDGE_POINT, r["src"])
                tgt_id = _make_node_id(LABEL_KNOWLEDGE_POINT, r["dst"])
                links.append({
                    "source": src_id,
                    "target": tgt_id,
                    "type": REL_PREREQUISITE_OF,
                    "color": REL_COLORS[REL_PREREQUISITE_OF],
                })

    return {"nodes": nodes, "links": links, "stages": path_data.get("stages", [])}


@app.get("/api/search")
def search(q: str = Query(..., description="搜索关键词")):
    """模糊搜索节点"""
    driver = get_neo4j_driver()
    with driver.session() as session:
        result = session.run(
            f"""
            MATCH (n)
            WHERE (n:{LABEL_MAJOR} OR n:{LABEL_COURSE} OR n:{LABEL_KNOWLEDGE_POINT})
              AND toLower(n.name) CONTAINS toLower($q)
            RETURN labels(n)[0] AS label, properties(n) AS props
            LIMIT 20
            """,
            q=q,
        )
        nodes = []
        for r in result:
            label = r["label"]
            props = r["props"]
            nodes.append({
                "id": _make_node_id(label, props.get("name", ""), props.get("university", "")),
                "name": props.get("name", ""),
                "label": label,
                "color": LABEL_COLORS.get(label, "#cccccc"),
                "size": LABEL_SIZES.get(label, 5),
                "properties": props,
            })
    return nodes


@app.post("/api/qa")
def ask_question(payload: QARequest):
    """基于知识图谱的学习问答"""
    driver = get_neo4j_driver()
    with driver.session() as session:
        return answer_question(
            session,
            payload.question,
            known=payload.known,
            goal=payload.goal,
            weekly_hours=payload.weekly_hours,
            level=payload.level,
        )


@app.get("/api/stats")
def get_stats():
    """获取图谱统计信息"""
    driver = get_neo4j_driver()
    with driver.session() as session:
        r = session.run(
            f"""
            MATCH (n) WHERE n:{LABEL_MAJOR} OR n:{LABEL_COURSE} OR n:{LABEL_KNOWLEDGE_POINT}
            RETURN labels(n)[0] AS label, count(n) AS cnt
            """
        )
        counts = {rec["label"]: rec["cnt"] for rec in r}
        r = session.run(
            """
            MATCH ()-[r]->() RETURN type(r) AS rel, count(r) AS cnt
            """
        )
        rel_counts = {rec["rel"]: rec["cnt"] for rec in r}
    return {"node_counts": counts, "relationship_counts": rel_counts}


# 静态文件挂载（放在所有 API 路由之后）
static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
if os.path.exists(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
