"""学习路径推荐：前驱链查询 + 拓扑排序 + 下一批推荐"""
from collections import deque
from src.models.schema import LABEL_KNOWLEDGE_POINT, LABEL_COURSE, LABEL_MAJOR


def _topological_sort(nodes: set[str], edges: list[tuple[str, str]]) -> list[list[str]]:
    """Kahn 算法拓扑排序，返回分层的学习阶段列表"""
    in_degree = {n: 0 for n in nodes}
    adj = {n: [] for n in nodes}

    for src, dst in edges:
        if src in nodes and dst in nodes:
            adj[src].append(dst)
            in_degree[dst] += 1

    # 第 0 层：入度为 0 的节点（无前驱依赖）
    queue = deque([n for n in nodes if in_degree[n] == 0])
    sorted_nodes = []
    visited = set()

    while queue:
        batch = []
        for _ in range(len(queue)):
            n = queue.popleft()
            if n in visited:
                continue
            visited.add(n)
            batch.append(n)
            for neighbor in adj[n]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        if batch:
            sorted_nodes.append(batch)

    # 剩余有环的节点各自成为一层
    remaining = [n for n in nodes if n not in visited]
    for n in remaining:
        sorted_nodes.append([n])

    return sorted_nodes


def find_prerequisites(session, target_name: str) -> dict:
    """给定目标知识点名，反向 BFS 查找所有前驱知识点并拓扑排序"""
    # 获取所有前驱节点
    result = session.run(
        f"""
        MATCH path = (prereq:{LABEL_KNOWLEDGE_POINT})-[:PREREQUISITE_OF*1..10]->
                     (target:{LABEL_KNOWLEDGE_POINT} {{name: $name}})
        RETURN DISTINCT prereq.name AS name, prereq.difficulty AS difficulty,
               prereq.category AS category, length(path) AS distance
        ORDER BY distance DESC
        """,
        name=target_name,
    )
    prereqs = [r for r in result]

    # 检查目标节点是否存在
    target_check = session.run(
        f"MATCH (k:{LABEL_KNOWLEDGE_POINT} {{name: $name}}) RETURN k",
        name=target_name,
    )
    if not target_check.single():
        return {"target": target_name, "error": "Knowledge point not found"}

    # 收集所有节点和边
    all_names = set()
    all_names.add(target_name)
    for r in prereqs:
        all_names.add(r["name"])

    # 获取这些节点之间的 PREREQUISITE_OF 边
    edge_result = session.run(
        f"""
        MATCH (a:{LABEL_KNOWLEDGE_POINT})-[r:PREREQUISITE_OF]->(b:{LABEL_KNOWLEDGE_POINT})
        WHERE a.name IN $names AND b.name IN $names
        RETURN a.name AS src, b.name AS dst
        """,
        names=list(all_names),
    )
    edges = [(r["src"], r["dst"]) for r in edge_result]

    # 拓扑排序
    layers = _topological_sort(all_names, edges)

    # 查找每个知识点关联的课程
    kp_courses = {}
    course_result = session.run(
        f"""
        MATCH (c:{LABEL_COURSE})-[r:COVERS]->(k:{LABEL_KNOWLEDGE_POINT})
        WHERE k.name IN $names
        RETURN k.name AS kp, c.name AS course
        """,
        names=list(all_names),
    )
    for r in course_result:
        kp_courses.setdefault(r["kp"], []).append(r["course"])

    return {
        "target": target_name,
        "total_knowledge_points": len(all_names),
        "depth": len(layers),
        "stages": [
            {
                "stage": i + 1,
                "knowledge_points": layer,
                "courses": list(set(
                    c for kp in layer for c in kp_courses.get(kp, [])
                )),
            }
            for i, layer in enumerate(layers)
        ],
    }


def recommend_next(session, known_kp_names: list[str]) -> dict:
    """给定已掌握的知识点，查找所有前驱都已满足的下一批可学知识点"""
    if not known_kp_names:
        # 如果用户没指定已知知识，返回无前驱依赖的基础知识点
        result = session.run(
            f"""
            MATCH (k:{LABEL_KNOWLEDGE_POINT})
            WHERE NOT EXISTS {{
                MATCH (other:{LABEL_KNOWLEDGE_POINT})-[:PREREQUISITE_OF]->(k)
            }}
            RETURN k.name AS name, k.category AS category, k.difficulty AS difficulty
            ORDER BY k.difficulty, k.name
            """
        )
        entry_points = [dict(r) for r in result]
        return {
            "known": known_kp_names,
            "ready_to_learn": entry_points,
            "suggestion": "你还没有指定已有基础，以下是不需要前置知识的基础知识点",
        }

    # 查找：已知集合的所有后继（即 known → next）
    result = session.run(
        f"""
        MATCH (known:{LABEL_KNOWLEDGE_POINT})-[:PREREQUISITE_OF]->(next:{LABEL_KNOWLEDGE_POINT})
        WHERE known.name IN $known_names AND NOT next.name IN $known_names
        WITH next, collect(known.name) AS prereqs_satisfied
        // 检查 next 的所有前驱是否都在已知集合中
        OPTIONAL MATCH (prereq:{LABEL_KNOWLEDGE_POINT})-[:PREREQUISITE_OF]->(next)
        WHERE NOT prereq.name IN $known_names
        WITH next, prereqs_satisfied, count(prereq) AS missing_count
        WHERE missing_count = 0
        RETURN next.name AS name, next.category AS category, next.difficulty AS difficulty,
               prereqs_satisfied
        ORDER BY next.difficulty, next.name
        """,
        known_names=known_kp_names,
    )
    ready = [dict(r) for r in result]

    return {
        "known": known_kp_names,
        "ready_to_learn": ready,
    }


def find_path_to_target(session, known_kp_names: list[str], target_name: str) -> dict:
    """从已知知识点到目标知识点，返回完整待学习前驱集合。"""
    prereq_plan = find_prerequisites(session, target_name)
    if "error" in prereq_plan:
        return prereq_plan

    known_set = set(known_kp_names)
    remaining_stages = []
    need_to_learn = []

    for stage in prereq_plan.get("stages", []):
        remaining_kps = [
            kp for kp in stage.get("knowledge_points", [])
            if kp not in known_set
        ]
        if not remaining_kps:
            continue

        need_to_learn.extend(remaining_kps)
        remaining_stages.append({
            "stage": len(remaining_stages) + 1,
            "knowledge_points": remaining_kps,
            "courses": stage.get("courses", []),
        })

    return {
        "known": known_kp_names,
        "target": target_name,
        "path": need_to_learn,
        "need_to_learn": need_to_learn,
        "steps_from_current": len(need_to_learn),
        "total_knowledge_points": len(need_to_learn),
        "depth": len(remaining_stages),
        "stages": remaining_stages,
    }


def get_major_structure(session, major_name: str, university: str = None) -> dict:
    """获取专业完整结构：按学期排列的课程+知识点"""
    if university:
        result = session.run(
            f"""
            MATCH (m:{LABEL_MAJOR} {{name: $major, university: $uni}})
            -[:BELONGS_TO]->(c:{LABEL_COURSE})
            OPTIONAL MATCH (c)-[:COVERS]->(k:{LABEL_KNOWLEDGE_POINT})
            RETURN m.name AS major, m.university AS university, m.description AS desc,
                   c.name AS course, c.type AS type, c.semester AS semester,
                   c.credits AS credits,
                   collect(DISTINCT k.name) AS knowledge_points
            ORDER BY c.semester, c.name
            """,
            major=major_name, uni=university,
        )
    else:
        result = session.run(
            f"""
            MATCH (m:{LABEL_MAJOR} {{name: $major}})
            -[:BELONGS_TO]->(c:{LABEL_COURSE})
            OPTIONAL MATCH (c)-[:COVERS]->(k:{LABEL_KNOWLEDGE_POINT})
            RETURN m.name AS major, m.university AS university, m.description AS desc,
                   c.name AS course, c.type AS type, c.semester AS semester,
                   c.credits AS credits,
                   collect(DISTINCT k.name) AS knowledge_points
            ORDER BY c.semester, c.name
            """,
            major=major_name,
        )

    records = [dict(r) for r in result]
    if not records:
        return {"error": f"Major '{major_name}' not found"}

    return {
        "major": records[0]["major"],
        "university": records[0]["university"],
        "description": records[0]["desc"],
        "courses": [
            {
                "name": r["course"],
                "semester": r["semester"],
                "type": r["type"],
                "credits": r["credits"],
                "knowledge_points": r["knowledge_points"],
            }
            for r in records
        ],
    }
