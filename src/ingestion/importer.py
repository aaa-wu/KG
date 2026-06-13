"""Neo4j 批量导入器：使用 UNWIND 高效写入节点和关系"""
from src.models.schema import (
    LABEL_MAJOR, LABEL_COURSE, LABEL_KNOWLEDGE_POINT,
    REL_BELONGS_TO, REL_COVERS, REL_PREREQUISITE_OF,
    REL_PREREQUISITE_FOR,
)

DEFAULT_UNIVERSITY = "真实数据"


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
            n.description = row.description,
            n.id = coalesce(row.id, n.id)
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
            c.description = row.description,
            c.id = coalesce(row.id, c.id)
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
            k.description = coalesce(row.description, ''),
            k.id = coalesce(row.id, k.id)
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


def _import_course_prereq_rels(session, rels: list[dict]) -> int:
    """rels: [{"from": "Course A", "to": "Course B", "strength": "strong"}, ...]"""
    if not rels:
        return 0

    result = session.run(
        f"""
        UNWIND $batch AS row
        MATCH (a:{LABEL_COURSE} {{name: row.from}})
        MATCH (b:{LABEL_COURSE} {{name: row.to}})
        MERGE (a)-[r:{REL_PREREQUISITE_FOR}]->(b)
        SET r.strength = row.strength
        RETURN count(*) AS c
        """,
        batch=rels,
    )
    return result.single()["c"]


def _audit_lookup(audit_rows: list[dict]) -> dict:
    lookup = {}
    for row in audit_rows:
        rel_type = row.get("RelationType", "")
        if rel_type == REL_COVERS:
            key = (row.get("Course", ""), rel_type, row.get("KnowledgeConcept", ""))
        elif rel_type == REL_PREREQUISITE_OF and " -> " in row.get("KnowledgeConcept", ""):
            start, end = row["KnowledgeConcept"].split(" -> ", 1)
            key = (start, rel_type, end)
        else:
            continue
        lookup[key] = {
            "source_type": row.get("SourceType", ""),
            "source_url": row.get("SourceURL", ""),
            "source_title": row.get("SourceTitle", ""),
            "source_location": row.get("SourceLocation", ""),
            "basis": row.get("Basis", ""),
            "confidence": row.get("Confidence", ""),
            "need_manual_review": row.get("NeedManualReview", ""),
            "added_by": row.get("AddedBy", ""),
            "notes": row.get("Notes", ""),
        }
    return lookup


def _entity_rows_by_label(entities: list[dict], label: str) -> list[dict]:
    return [row for row in entities if row.get("label") == label]


def _normalize_entity_graph(data: dict) -> dict:
    entities = data.get("entities", [])
    relations = data.get("relations", [])
    by_id = {row["id"]: row for row in entities}
    audit = _audit_lookup(data.get("knowledge_concept_audit", []))

    majors = [
        {
            "id": row["id"],
            "name": row["name"],
            "university": DEFAULT_UNIVERSITY,
            "degree": "",
            "department": "",
            "duration": 0,
            "description": "",
        }
        for row in _entity_rows_by_label(entities, LABEL_MAJOR)
    ]
    courses = [
        {
            "id": row["id"],
            "name": row["name"],
            "code": row["id"],
            "credits": 0,
            "hours": 0,
            "type": "",
            "semester": 0,
            "description": "",
        }
        for row in _entity_rows_by_label(entities, LABEL_COURSE)
    ]
    knowledge = [
        {
            "id": row["id"],
            "name": row["name"],
            "category": "知识概念",
            "difficulty": 3,
            "description": "",
        }
        for row in _entity_rows_by_label(entities, LABEL_KNOWLEDGE_POINT)
    ]
    graph_rels = []
    for row in relations:
        start = by_id[row["start_id"]]
        end = by_id[row["end_id"]]
        rel_type = row["type"]
        meta = audit.get((start["name"], rel_type, end["name"]), {})
        graph_rels.append({
            "start_id": row["start_id"],
            "end_id": row["end_id"],
            "type": rel_type,
            **meta,
        })

    return {
        "majors": majors,
        "courses": courses,
        "knowledge": knowledge,
        "graph_rels": graph_rels,
    }


def _import_graph_rels_by_id(session, rels: list[dict], rel_type: str) -> int:
    batch = [row for row in rels if row["type"] == rel_type]
    if not batch:
        return 0

    result = session.run(
        f"""
        UNWIND $batch AS row
        MATCH (a {{id: row.start_id}})
        MATCH (b {{id: row.end_id}})
        MERGE (a)-[r:{rel_type}]->(b)
        SET r.source_type = coalesce(row.source_type, ''),
            r.source_url = coalesce(row.source_url, ''),
            r.source_title = coalesce(row.source_title, ''),
            r.source_location = coalesce(row.source_location, ''),
            r.basis = coalesce(row.basis, ''),
            r.confidence = coalesce(row.confidence, ''),
            r.need_manual_review = coalesce(row.need_manual_review, ''),
            r.added_by = coalesce(row.added_by, ''),
            r.notes = coalesce(row.notes, '')
        RETURN count(*) AS c
        """,
        batch=batch,
    )
    return result.single()["c"]


def _import_entity_relation_graph(session, data: dict) -> dict:
    normalized = _normalize_entity_graph(data)
    counts = {}
    counts["majors"] = _import_majors(session, normalized["majors"])
    counts["courses"] = _import_courses(session, normalized["courses"])
    counts["knowledge_points"] = _import_knowledge_points(session, normalized["knowledge"])
    counts["major_course_rels"] = _import_graph_rels_by_id(
        session, normalized["graph_rels"], REL_BELONGS_TO
    )
    counts["course_knowledge_rels"] = _import_graph_rels_by_id(
        session, normalized["graph_rels"], REL_COVERS
    )
    counts["knowledge_prereq_rels"] = _import_graph_rels_by_id(
        session, normalized["graph_rels"], REL_PREREQUISITE_OF
    )
    counts["course_prereq_rels"] = _import_graph_rels_by_id(
        session, normalized["graph_rels"], REL_PREREQUISITE_FOR
    )
    return counts


def import_all(session, data: dict) -> dict:
    """导入全部数据，返回各类节点/关系的导入计数"""
    if data.get("format") == "entity_relation_csv":
        return _import_entity_relation_graph(session, data)

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
    counts["course_prereq_rels"] = _import_course_prereq_rels(
        session, data.get("course_prereq_rels", [])
    )
    return counts
