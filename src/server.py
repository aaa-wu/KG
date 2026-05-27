"""FastAPI 后端：从 Neo4j 读取图数据并返回 3D 可视化所需的 JSON"""
import os
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from src.config import get_neo4j_driver
from src.models.schema import (
    LABEL_MAJOR, LABEL_COURSE, LABEL_KNOWLEDGE_POINT,
)
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
    "BELONGS_TO": "#00e5ff",
    "COVERS": "#26c6da",
    "PREREQUISITE_OF": "#00e5ff",
    "RELATED_TO": "#dfe6e9",
}


def _make_node_id(label: str, name: str, extra: str = "") -> str:
    """生成唯一节点 ID"""
    parts = [label, name]
    if extra:
        parts.append(extra)
    return "::".join(parts)


@app.get("/api/graph")
def get_full_graph(limit: int = Query(default=200, description="最大节点数")):
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
        links_result = session.run(
            """
            MATCH (a)-[r]->(b)
            WHERE (a:Major OR a:Course OR a:KnowledgePoint)
              AND (b:Major OR b:Course OR b:KnowledgePoint)
            RETURN labels(a)[0] AS label_a, properties(a) AS props_a,
                   labels(b)[0] AS label_b, properties(b) AS props_b,
                   type(r) AS rel_type
            LIMIT 500
            """
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
            WHERE neighbor:Major OR neighbor:Course OR neighbor:KnowledgePoint
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
            """
            MATCH (a)-[r]->(b)
            WHERE (a:Major OR a:Course OR a:KnowledgePoint)
              AND (b:Major OR b:Course OR b:KnowledgePoint)
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
                MATCH (a:{LABEL_KNOWLEDGE_POINT})-[r:PREREQUISITE_OF]->(b:{LABEL_KNOWLEDGE_POINT})
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
                    "type": "PREREQUISITE_OF",
                    "color": REL_COLORS["PREREQUISITE_OF"],
                })

    return {"nodes": nodes, "links": links, "stages": path_data.get("stages", [])}


@app.get("/api/search")
def search(q: str = Query(..., description="搜索关键词")):
    """模糊搜索节点"""
    driver = get_neo4j_driver()
    with driver.session() as session:
        result = session.run(
            """
            MATCH (n)
            WHERE (n:Major OR n:Course OR n:KnowledgePoint)
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


@app.get("/api/stats")
def get_stats():
    """获取图谱统计信息"""
    driver = get_neo4j_driver()
    with driver.session() as session:
        r = session.run(
            """
            MATCH (n) WHERE n:Major OR n:Course OR n:KnowledgePoint
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
