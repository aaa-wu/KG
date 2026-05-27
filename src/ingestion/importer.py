"""Neo4j 批量导入器：使用 UNWIND 高效写入节点和关系"""
from src.models.schema import (
    LABEL_MAJOR, LABEL_COURSE, LABEL_KNOWLEDGE_POINT,
    REL_BELONGS_TO, REL_COVERS, REL_PREREQUISITE_OF,
)


def _import_majors(session, majors: list[dict]) -> int:
    if not majors:
        return 0
    result = session.run(
        f"""
        UNWIND $batch AS row
        MERGE (n:{LABEL_MAJOR} {{name: row.name, university: row.university}})
        SET n.degree = row.degree,
            n.department = row.department,
            n.duration = row.duration,
            n.description = row.description
        RETURN count(n) AS c
        """,
        batch=majors,
    )
    return result.single()["c"]


def _import_courses(session, courses: list[dict]) -> int:
    if not courses:
        return 0
    result = session.run(
        f"""
        UNWIND $batch AS row
        MERGE (c:{LABEL_COURSE} {{name: row.name}})
        SET c.code = row.code,
            c.credits = row.credits,
            c.hours = row.hours,
            c.type = row.type,
            c.semester = row.semester,
            c.description = row.description
        RETURN count(c) AS c
        """,
        batch=courses,
    )
    return result.single()["c"]


def _import_knowledge_points(session, kps: list[dict]) -> int:
    if not kps:
        return 0
    result = session.run(
        f"""
        UNWIND $batch AS row
        MERGE (k:{LABEL_KNOWLEDGE_POINT} {{name: row.name}})
        SET k.category = row.category,
            k.difficulty = row.difficulty,
            k.description = coalesce(row.description, '')
        RETURN count(k) AS c
        """,
        batch=kps,
    )
    return result.single()["c"]


def _import_major_course_rels(session, rels: dict) -> int:
    """rels: {"清华-计算机": ["高等数学A", ...], ...}"""
    batch = []
    for key, course_names in rels.items():
        parts = key.rsplit("-", 1)
        if len(parts) == 2:
            uni, major = parts
        else:
            uni, major = key, ""
        for cname in course_names:
            batch.append({"university": uni, "major": major, "course": cname})

    if not batch:
        return 0

    result = session.run(
        f"""
        UNWIND $batch AS row
        MATCH (m:{LABEL_MAJOR} {{name: row.major, university: row.university}})
        MATCH (c:{LABEL_COURSE} {{name: row.course}})
        MERGE (m)-[:{REL_BELONGS_TO}]->(c)
        RETURN count(*) AS c
        """,
        batch=batch,
    )
    return result.single()["c"]


def _import_course_knowledge_rels(session, rels: dict) -> int:
    """rels: {"高等数学A": [{"kp": "...", "depth": "...", "weight": 0.2}, ...], ...}"""
    batch = []
    for course_name, kp_list in rels.items():
        for item in kp_list:
            batch.append({
                "course": course_name,
                "kp": item["kp"],
                "depth": item.get("depth", "intro"),
                "weight": item.get("weight", 1.0),
            })

    if not batch:
        return 0

    result = session.run(
        f"""
        UNWIND $batch AS row
        MATCH (c:{LABEL_COURSE} {{name: row.course}})
        MATCH (k:{LABEL_KNOWLEDGE_POINT} {{name: row.kp}})
        MERGE (c)-[r:{REL_COVERS}]->(k)
        SET r.depth = row.depth,
            r.weight = row.weight
        RETURN count(*) AS c
        """,
        batch=batch,
    )
    return result.single()["c"]


def _import_prereq_rels(session, rels: list[dict]) -> int:
    """rels: [{"from": "A", "to": "B", "strength": "strong"}, ...]"""
    if not rels:
        return 0

    result = session.run(
        f"""
        UNWIND $batch AS row
        MATCH (a:{LABEL_KNOWLEDGE_POINT} {{name: row.from}})
        MATCH (b:{LABEL_KNOWLEDGE_POINT} {{name: row.to}})
        MERGE (a)-[r:{REL_PREREQUISITE_OF}]->(b)
        SET r.strength = row.strength
        RETURN count(*) AS c
        """,
        batch=rels,
    )
    return result.single()["c"]


def import_all(session, data: dict) -> dict:
    """导入全部数据，返回各类节点/关系的导入计数"""
    counts = {}
    counts["majors"] = _import_majors(session, data.get("majors", []))
    counts["courses"] = _import_courses(session, data.get("courses", []))
    counts["knowledge_points"] = _import_knowledge_points(session, data.get("knowledge", []))
    counts["major_course_rels"] = _import_major_course_rels(
        session, data.get("major_course_rels", {})
    )
    counts["course_knowledge_rels"] = _import_course_knowledge_rels(
        session, data.get("course_knowledge_rels", {})
    )
    counts["knowledge_prereq_rels"] = _import_prereq_rels(
        session, data.get("knowledge_prereq_rels", [])
    )
    return counts
