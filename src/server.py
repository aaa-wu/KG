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
    LABEL_TOPIC, LABEL_SUBTOPIC, LABEL_DOMAIN,
    REL_BELONGS_TO, REL_COVERS, REL_PREREQUISITE_OF, REL_PREREQUISITE_FOR,
    REL_HAS_TOPIC, REL_HAS_SUBTOPIC, REL_COVERS_SUBTOPIC, REL_IN_DOMAIN,
    REL_SEMANTIC_SIMILARITY, REL_PREDICTED_PREREQ,
)
from src.qa import answer_question
from src.recommendation.path_finder import find_prerequisites
from src.recommendation.roadmap_classifier import build_module_roadmap

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
    REL_PREDICTED_PREREQ: "#ff9f43",
    "RELATED_TO": "#dfe6e9",
}


class QARequest(BaseModel):
    question: str
    known: list[str] = Field(default_factory=list)
    goal: Optional[str] = None
    weekly_hours: Optional[float] = None
    level: Optional[str] = None


class AgentChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str
    known: list[str] = Field(default_factory=list)
    target: Optional[str] = None
    goal: Optional[str] = None
    weekly_hours: Optional[float] = None
    level: Optional[str] = None


class ExtractTopicsRequest(BaseModel):
    major_name: str


class ValidationActionRequest(BaseModel):
    id: str
    notes: Optional[str] = ""


class PredictPrereqRequest(BaseModel):
    threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    dry_run: bool = False
    max_predictions: int = Field(default=500, ge=1, le=5000)
    entity_label: str = Field(default=LABEL_KNOWLEDGE_POINT)
    method: str = Field(default="llm")  # "llm" | "mlp"
    max_targets: Optional[int] = Field(default=None, ge=1, le=5000)
    top_k: int = Field(default=15, ge=1, le=50)


class ComputeSimilarityRequest(BaseModel):
    entity_label: str = Field(default=LABEL_KNOWLEDGE_POINT)
    threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    max_edges: int = Field(default=5000, ge=1, le=20000)


class RecommendRequest(BaseModel):
    known: list[str] = Field(default_factory=list)
    target: Optional[str] = None
    major: Optional[str] = None
    weekly_hours: Optional[float] = None
    level: Optional[str] = "intermediate"
    goal: Optional[str] = None
    use_predicted: bool = False


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
        knowledge_names = set()
        for row in rows:
            course = dict(row["course"])
            knowledge = [dict(item) for item in row["knowledge"] if item]
            knowledge_names.update(item.get("name", "") for item in knowledge if item.get("name"))
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

        relation_counts = {}
        if knowledge_names:
            count_rows = session.run(
                f"""
                MATCH (k:{LABEL_KNOWLEDGE_POINT})
                WHERE k.name IN $names
                OPTIONAL MATCH (pre:{LABEL_KNOWLEDGE_POINT})-[:{REL_PREREQUISITE_OF}|{REL_PREDICTED_PREREQ}]->(k)
                WITH k, count(DISTINCT pre) AS prerequisite_count
                OPTIONAL MATCH (k)-[:{REL_PREREQUISITE_OF}|{REL_PREDICTED_PREREQ}]->(dep:{LABEL_KNOWLEDGE_POINT})
                RETURN k.name AS name,
                       prerequisite_count,
                       count(DISTINCT dep) AS dependent_count
                """,
                names=list(knowledge_names),
            )
            relation_counts = {
                row["name"]: {
                    "prerequisite_count": row["prerequisite_count"],
                    "dependent_count": row["dependent_count"],
                }
                for row in count_rows
            }
            for course in raw_courses:
                for item in course["knowledge_points"]:
                    item.update(relation_counts.get(item["name"], {
                        "prerequisite_count": 0,
                        "dependent_count": 0,
                    }))

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

    total_courses = len(raw_courses)
    prereq_links = []
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

    # Use manually curated per-major module dependency graph.
    roadmap = build_module_roadmap(major_name, raw_courses, prereq_links)

    return {
        "major": {
            "id": major.get("id") or _make_node_id(LABEL_MAJOR, major.get("name", ""), major.get("university", "")),
            "name": major.get("name", major_name),
            "icon_key": _major_icon_key(major.get("name", major_name)),
            "properties": major,
        },
        "total_courses": roadmap["total_courses"],
        "total_prerequisite_links": roadmap["total_prerequisite_links"],
        "modules": roadmap["modules"],
        "module_links": roadmap["module_links"],
        "courses": roadmap["courses"],
        "links": roadmap["links"],
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


@app.get("/api/knowledge/{kp_name}/relations")
def get_knowledge_relations(kp_name: str):
    """获取知识点直接前置和直接后续关系。"""
    driver = get_neo4j_driver()

    def _knowledge_node(props: dict, *, is_target: bool = False, is_predicted: bool = False) -> dict:
        name = props.get("name", "")
        return {
            "id": _make_node_id(LABEL_KNOWLEDGE_POINT, name),
            "name": name,
            "label": LABEL_KNOWLEDGE_POINT,
            "color": LABEL_COLORS[LABEL_KNOWLEDGE_POINT],
            "size": LABEL_SIZES[LABEL_KNOWLEDGE_POINT] * (2 if is_target else 1),
            "is_predicted": is_predicted,
            "properties": props,
        }

    with driver.session() as session:
        target_row = session.run(
            f"""
            MATCH (target:{LABEL_KNOWLEDGE_POINT} {{name: $name}})
            RETURN properties(target) AS props
            """,
            name=kp_name,
        ).single()

        if not target_row:
            raise HTTPException(status_code=404, detail="Knowledge point not found")

        explicit_prereq_rows = session.run(
            f"""
            MATCH (pre:{LABEL_KNOWLEDGE_POINT})-[:{REL_PREREQUISITE_OF}]->(target:{LABEL_KNOWLEDGE_POINT} {{name: $name}})
            RETURN DISTINCT properties(pre) AS props, pre.name AS name
            ORDER BY name
            """,
            name=kp_name,
        )
        explicit_dependent_rows = session.run(
            f"""
            MATCH (target:{LABEL_KNOWLEDGE_POINT} {{name: $name}})-[:{REL_PREREQUISITE_OF}]->(dep:{LABEL_KNOWLEDGE_POINT})
            RETURN DISTINCT properties(dep) AS props, dep.name AS name
            ORDER BY name
            """,
            name=kp_name,
        )
        predicted_prereq_rows = session.run(
            f"""
            MATCH (pre:{LABEL_KNOWLEDGE_POINT})-[:{REL_PREDICTED_PREREQ}]->(target:{LABEL_KNOWLEDGE_POINT} {{name: $name}})
            RETURN DISTINCT properties(pre) AS props, pre.name AS name
            ORDER BY name
            """,
            name=kp_name,
        )
        predicted_dependent_rows = session.run(
            f"""
            MATCH (target:{LABEL_KNOWLEDGE_POINT} {{name: $name}})-[:{REL_PREDICTED_PREREQ}]->(dep:{LABEL_KNOWLEDGE_POINT})
            RETURN DISTINCT properties(dep) AS props, dep.name AS name
            ORDER BY name
            """,
            name=kp_name,
        )

        target = _knowledge_node(target_row["props"], is_target=True)

        def _merge_nodes(explicit_rows, predicted_rows):
            nodes = {}
            for row in explicit_rows:
                node = _knowledge_node(row["props"])
                nodes[node["name"]] = node
            for row in predicted_rows:
                name = row["name"]
                if name not in nodes:
                    nodes[name] = _knowledge_node(row["props"], is_predicted=True)
            return list(nodes.values())

        prerequisites = _merge_nodes(explicit_prereq_rows, predicted_prereq_rows)
        dependents = _merge_nodes(explicit_dependent_rows, predicted_dependent_rows)

    links = [
        {
            "source": node["id"],
            "target": target["id"],
            "type": REL_PREDICTED_PREREQ if node.get("is_predicted") else REL_PREREQUISITE_OF,
            "color": REL_COLORS[REL_PREDICTED_PREREQ] if node.get("is_predicted") else REL_COLORS[REL_PREREQUISITE_OF],
        }
        for node in prerequisites
    ] + [
        {
            "source": target["id"],
            "target": node["id"],
            "type": REL_PREDICTED_PREREQ if node.get("is_predicted") else REL_PREREQUISITE_OF,
            "color": REL_COLORS[REL_PREDICTED_PREREQ] if node.get("is_predicted") else REL_COLORS[REL_PREREQUISITE_OF],
        }
        for node in dependents
    ]

    return {
        "target": target,
        "prerequisites": prerequisites,
        "dependents": dependents,
        "links": links,
    }


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


@app.post("/api/agent/chat")
def chat_with_learning_agent(payload: AgentChatRequest):
    """学习路径规划 Agent：DeepSeek + LangGraph + 知识图谱增强。"""
    try:
        from src.agent.learning_path_agent import run_learning_path_agent

        return run_learning_path_agent(
            session_id=payload.session_id,
            message=payload.message,
            known=payload.known,
            target=payload.target,
            goal=payload.goal,
            weekly_hours=payload.weekly_hours,
            level=payload.level,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent failed: {exc}") from exc


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


# ====================== P0: 本体扩展 / LLM 抽取 / 语义相似度 ======================

@app.post("/api/extract/topics")
def extract_topics(payload: ExtractTopicsRequest):
    """基于现有课程/知识点名，使用 DeepSeek 抽取 Topic/SubTopic/Domain 提案。"""
    from src.ontology.topic_extractor import extract_topics_for_major_from_neo4j

    result = extract_topics_for_major_from_neo4j(payload.major_name)
    if result is None:
        return {
            "status": "fallback",
            "message": "LLM 抽取不可用（未配置 DEEPSEEK_API_KEY 或调用失败），请检查 .env 或手动录入主题。",
            "major": payload.major_name,
            "topics": [],
        }
    return {
        "status": "success",
        "major": result.major,
        "topics": [
            {"name": t.name, "domain": t.domain, "subtopics": t.subtopics}
            for t in result.topics
        ],
        "course_to_topics": result.course_to_topics,
        "concept_to_subtopics": result.concept_to_subtopics,
    }


@app.post("/api/extract/topics/queue")
def extract_topics_and_queue(payload: ExtractTopicsRequest):
    """抽取 Topic/SubTopic 并写入待审队列，等待人工批准。"""
    from src.ontology.topic_extractor import extract_topics_for_major_from_neo4j
    from src.ontology.validator import queue_extraction_result

    result = extract_topics_for_major_from_neo4j(payload.major_name)
    if result is None:
        return {
            "status": "fallback",
            "message": "LLM 抽取不可用，无法加入队列。",
            "major": payload.major_name,
            "queued_ids": [],
        }
    queued_ids = queue_extraction_result(result)
    return {
        "status": "success",
        "major": result.major,
        "queued_count": len(queued_ids),
        "queued_ids": queued_ids,
    }


@app.get("/api/validation/pending")
def get_pending_validations(limit: int = Query(default=20, ge=1, le=100)):
    """获取待人工复核的 Topic/SubTopic 抽取提案。"""
    from src.ontology.validator import get_pending_validations

    items = get_pending_validations(limit)
    return {
        "items": [
            {
                "id": item.id,
                "major": item.major,
                "domain": item.domain,
                "topic": item.topic,
                "subtopics": item.subtopics,
                "course_mappings": item.course_mappings,
                "concept_mappings": item.concept_mappings,
                "created_at": item.created_at,
            }
            for item in items
        ],
        "count": len(items),
    }


@app.post("/api/validation/approve")
def approve_validation(payload: ValidationActionRequest):
    """批准一个待审提案并导入 Neo4j。"""
    from src.ontology.validator import approve_extraction, load_validated_items
    from src.ontology.importer import import_validated_topics

    if not approve_extraction(payload.id, payload.notes or ""):
        raise HTTPException(status_code=404, detail="待审项不存在或已处理")

    # 导入所有已批准项（幂等）
    validated = load_validated_items()
    counts = import_validated_topics(validated)
    return {"status": "approved", "id": payload.id, "import_counts": counts}


@app.post("/api/validation/reject")
def reject_validation(payload: ValidationActionRequest):
    """拒绝一个待审提案。"""
    from src.ontology.validator import reject_extraction

    if not reject_extraction(payload.id, payload.notes or ""):
        raise HTTPException(status_code=404, detail="待审项不存在或已处理")
    return {"status": "rejected", "id": payload.id}


@app.get("/api/courses/{course_name}/topics")
def get_course_topics(course_name: str):
    """获取指定课程的 Topic/SubTopic 层级。"""
    driver = get_neo4j_driver()
    with driver.session() as session:
        result = session.run(
            f"""
            MATCH (c:{LABEL_COURSE} {{name: $course_name}})-[:{REL_HAS_TOPIC}]->(t:{LABEL_TOPIC})
            OPTIONAL MATCH (t)-[:{REL_HAS_SUBTOPIC}]->(st:{LABEL_SUBTOPIC})
            OPTIONAL MATCH (t)-[:{REL_IN_DOMAIN}]->(d:{LABEL_DOMAIN})
            RETURN t.name AS topic,
                   collect(DISTINCT st.name) AS subtopics,
                   d.name AS domain
            ORDER BY topic
            """,
            course_name=course_name,
        )
        topics = []
        for r in result:
            subtopics = [s for s in r["subtopics"] if s]
            topics.append({
                "topic": r["topic"],
                "domain": r["domain"],
                "subtopics": subtopics,
            })
    return {"course": course_name, "topics": topics}


@app.post("/api/similarity/compute")
def compute_similarity(payload: ComputeSimilarityRequest):
    """计算并存储 Course/Course 或 Concept/Concept 的语义相似度边。"""
    from src.ontology.similarity import compute_and_store_similarity

    if payload.entity_label not in (LABEL_KNOWLEDGE_POINT, LABEL_COURSE):
        raise HTTPException(
            status_code=400,
            detail=f"entity_label 必须是 {LABEL_KNOWLEDGE_POINT} 或 {LABEL_COURSE}",
        )
    return compute_and_store_similarity(
        entity_label=payload.entity_label,
        threshold=payload.threshold,
        max_edges=payload.max_edges,
    )


@app.get("/api/similar/concepts/{concept_name}")
def get_similar_concepts(concept_name: str, limit: int = Query(default=10, ge=1, le=50)):
    """查询与指定知识点最相似的其它知识点（基于 SEMANTIC_SIMILARITY 关系）。"""
    driver = get_neo4j_driver()
    with driver.session() as session:
        result = session.run(
            f"""
            MATCH (a:{LABEL_KNOWLEDGE_POINT} {{name: $name}})
                  -[r:{REL_SEMANTIC_SIMILARITY}]-(b:{LABEL_KNOWLEDGE_POINT})
            RETURN b.name AS name, r.weight AS similarity
            ORDER BY r.weight DESC
            LIMIT $limit
            """,
            name=concept_name,
            limit=limit,
        )
        similar = [
            {"name": r["name"], "similarity": round(float(r["similarity"]), 3)}
            for r in result
        ]
    return {"concept": concept_name, "similar": similar}


# ====================== P1: 前置关系预测 ======================

@app.post("/api/prereq/predict")
def predict_prerequisites(payload: PredictPrereqRequest):
    """预测缺失的前置关系（PREDICTED_PREREQUISITE）。

    支持两种方法：
      - method="llm": 使用 DeepSeek 直接判断前置关系（推荐，质量更高）。
      - method="mlp": 使用 embedding + MLP 算法预测（API key 不可用时 fallback）。
    """
    if payload.entity_label not in (LABEL_KNOWLEDGE_POINT,):
        raise HTTPException(status_code=400, detail="当前仅支持 KnowledgeConcept 的前置预测")

    if payload.method == "llm":
        from src.prereq_prediction.llm_predictor import complete_prerequisites_with_llm
        return complete_prerequisites_with_llm(
            targets=None,
            top_k=payload.top_k,
            min_score=payload.threshold,
            max_targets=payload.max_targets,
            dry_run=payload.dry_run,
        )
    elif payload.method == "mlp":
        from src.prereq_prediction.predictor import predict_and_store
        return predict_and_store(
            threshold=payload.threshold,
            dry_run=payload.dry_run,
            max_predictions=payload.max_predictions,
        )
    else:
        raise HTTPException(status_code=400, detail="method 必须是 'llm' 或 'mlp'")


@app.get("/api/prereq/predicted")
def get_predicted_prerequisites(limit: int = Query(default=100, ge=1, le=500)):
    """获取已存储的预测前置边。"""
    driver = get_neo4j_driver()
    with driver.session() as session:
        result = session.run(
            f"""
            MATCH (a:{LABEL_KNOWLEDGE_POINT})-[r:{REL_PREDICTED_PREREQ}]->(b:{LABEL_KNOWLEDGE_POINT})
            RETURN a.name AS src, b.name AS dst,
                   r.confidence AS confidence, r.method AS method,
                   r.created_at AS created_at
            ORDER BY r.confidence DESC
            LIMIT $limit
            """,
            limit=limit,
        )
        edges = [dict(r) for r in result]
    return {"predicted_edges": edges, "count": len(edges)}


# ====================== P2: 高级个性化推荐 ======================

@app.post("/api/recommend")
def recommend(payload: RecommendRequest):
    """基于双图（前置图 + 相似图）的个性化学习路径推荐。"""
    from src.rl.recommend import recommend_path

    try:
        result = recommend_path(
            known_concepts=payload.known,
            target_concept=payload.target,
            major_name=payload.major,
            weekly_hours=payload.weekly_hours,
            level=payload.level or "intermediate",
            goal=payload.goal,
            include_predicted=payload.use_predicted,
        )
        return {
            "target": result.target,
            "path": result.path,
            "blocked_recovery": result.blocked_recovery,
            "difficulty_analysis": result.difficulty_analysis,
            "explanation": result.explanation,
            "method": "dual_graph_rl",
        }
    except Exception as e:
        # Fallback to existing path finder
        from src.recommendation.path_finder import find_path_to_target
        driver = get_neo4j_driver()
        with driver.session() as session:
            fallback = find_path_to_target(session, payload.known, payload.target)
        return {
            "target": payload.target,
            "path": [],
            "blocked_recovery": [],
            "difficulty_analysis": {},
            "explanation": f"高级推荐引擎暂时不可用，使用基础路径查找。错误: {str(e)}",
            "fallback_result": fallback,
            "method": "fallback_kahn",
        }


# 静态文件挂载（放在所有 API 路由之后）
static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
if os.path.exists(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
