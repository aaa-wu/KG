"""图谱数据审计：输出当前数据质量报告，为 LLM 增强提供输入。"""
import json
import os
from dataclasses import dataclass, asdict
from typing import Optional

from src.config import get_neo4j_driver
from src.models.schema import (
    LABEL_MAJOR,
    LABEL_COURSE,
    LABEL_KNOWLEDGE_POINT,
    REL_BELONGS_TO,
    REL_COVERS,
    REL_PREREQUISITE_OF,
)

DEFAULT_AUDIT_PATH = "data/audit_report.json"


@dataclass
class AuditReport:
    node_counts: dict
    relation_counts: dict
    isolated_knowledge_points: list[str]
    knowledge_without_prereq_or_covered: list[str]
    courses_without_knowledge: list[str]
    majors_without_courses: list[str]
    missing_descriptions: dict

    def to_dict(self):
        return asdict(self)


def _count_nodes(session, label: str) -> int:
    result = session.run(f"MATCH (n:{label}) RETURN count(n) AS cnt")
    return result.single()["cnt"]


def _count_rels(session, rel_type: str) -> int:
    result = session.run(f"MATCH ()-[r:{rel_type}]->() RETURN count(r) AS cnt")
    return result.single()["cnt"]


def _find_isolated_knowledge(session) -> list[str]:
    """没有任何 COVERS 或 PREREQ 关系进出的知识点"""
    result = session.run(f"""
        MATCH (k:{LABEL_KNOWLEDGE_POINT})
        WHERE NOT (k)<-[:{REL_COVERS}]-(:{LABEL_COURSE})
          AND NOT (k)-[:{REL_PREREQUISITE_OF}]->(:{LABEL_KNOWLEDGE_POINT})
          AND NOT (k)<-[:{REL_PREREQUISITE_OF}]-(:{LABEL_KNOWLEDGE_POINT})
        RETURN k.name AS name
        LIMIT 200
    """)
    return [r["name"] for r in result]


def _find_knowledge_without_coverage(session) -> list[str]:
    """没有被任何课程覆盖，也没有前置关系知识点"""
    result = session.run(f"""
        MATCH (k:{LABEL_KNOWLEDGE_POINT})
        WHERE NOT (k)<-[:{REL_COVERS}]-(:{LABEL_COURSE})
          AND NOT (k)-[:{REL_PREREQUISITE_OF}]->(:{LABEL_KNOWLEDGE_POINT})
          AND NOT (k)<-[:{REL_PREREQUISITE_OF}]-(:{LABEL_KNOWLEDGE_POINT})
        RETURN k.name AS name
        LIMIT 200
    """)
    return [r["name"] for r in result]


def _find_courses_without_knowledge(session) -> list[str]:
    result = session.run(f"""
        MATCH (c:{LABEL_COURSE})
        WHERE NOT (c)-[:{REL_COVERS}]->(:{LABEL_KNOWLEDGE_POINT})
        RETURN c.name AS name
        LIMIT 200
    """)
    return [r["name"] for r in result]


def _find_majors_without_courses(session) -> list[str]:
    result = session.run(f"""
        MATCH (m:{LABEL_MAJOR})
        WHERE NOT (m)-[:{REL_BELONGS_TO}]->(:{LABEL_COURSE})
        RETURN m.name AS name
        LIMIT 50
    """)
    return [r["name"] for r in result]


def _find_missing_descriptions(session) -> dict:
    result = session.run(f"""
        MATCH (n)
        WHERE n:{LABEL_MAJOR} OR n:{LABEL_COURSE} OR n:{LABEL_KNOWLEDGE_POINT}
        RETURN labels(n)[0] AS label,
               count(n) AS total,
               count(CASE WHEN n.description IS NULL OR n.description = '' THEN 1 END) AS missing
    """)
    rows = {r["label"]: {"total": r["total"], "missing": r["missing"]} for r in result}
    return rows


def audit_graph(output_path: str = DEFAULT_AUDIT_PATH) -> AuditReport:
    """执行全图审计并保存报告。"""
    driver = get_neo4j_driver()
    with driver.session() as session:
        node_counts = {
            LABEL_MAJOR: _count_nodes(session, LABEL_MAJOR),
            LABEL_COURSE: _count_nodes(session, LABEL_COURSE),
            LABEL_KNOWLEDGE_POINT: _count_nodes(session, LABEL_KNOWLEDGE_POINT),
        }
        relation_counts = {
            REL_BELONGS_TO: _count_rels(session, REL_BELONGS_TO),
            REL_COVERS: _count_rels(session, REL_COVERS),
            REL_PREREQUISITE_OF: _count_rels(session, REL_PREREQUISITE_OF),
        }
        report = AuditReport(
            node_counts=node_counts,
            relation_counts=relation_counts,
            isolated_knowledge_points=_find_isolated_knowledge(session),
            knowledge_without_prereq_or_covered=_find_knowledge_without_coverage(session),
            courses_without_knowledge=_find_courses_without_knowledge(session),
            majors_without_courses=_find_majors_without_courses(session),
            missing_descriptions=_find_missing_descriptions(session),
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)

    return report


def load_audit_report(path: str = DEFAULT_AUDIT_PATH) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    report = audit_graph()
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
