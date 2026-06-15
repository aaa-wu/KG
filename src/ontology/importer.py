"""将已审批的 ValidationItem 导入 Neo4j 图数据库。"""
import warnings
from datetime import datetime
from typing import TYPE_CHECKING

from src.config import get_neo4j_driver
from src.models.schema import (
    LABEL_COURSE,
    LABEL_TOPIC,
    LABEL_SUBTOPIC,
    LABEL_DOMAIN,
    REL_HAS_TOPIC,
    REL_HAS_SUBTOPIC,
    REL_COVERS_SUBTOPIC,
    REL_IN_DOMAIN,
)

if TYPE_CHECKING:
    from .validator import ValidationItem


def import_validated_topics(validated_items: list["ValidationItem"]) -> dict:
    """
    把已审批的 ValidationItem 写入 Neo4j。

    创建的节点/关系：
      - Topic（name 唯一）
      - Domain（name 唯一）
      - SubTopic（name + parent_topic 复合唯一）
      - Course ->[:HAS_TOPIC]-> Topic
      - Topic ->[:HAS_SUBTOPIC]-> SubTopic
      - Course ->[:COVERS_SUBTOPIC]-> SubTopic
      - Topic ->[:IN_DOMAIN]-> Domain
      - SubTopic ->[:IN_DOMAIN]-> Domain

    如果 course_mappings 中的课程在 Neo4j 中不存在，则跳过并打印警告。

    Returns:
        dict: 各计数器，包含 topics_created, domains_created, subtopics_created,
              has_topic_rel_created, has_subtopic_rel_created,
              covers_subtopic_rel_created, in_domain_rel_created, skipped_courses。
    """
    driver = get_neo4j_driver()
    counts = {
        "topics_created": 0,
        "domains_created": 0,
        "subtopics_created": 0,
        "has_topic_rel_created": 0,
        "has_subtopic_rel_created": 0,
        "covers_subtopic_rel_created": 0,
        "in_domain_rel_created": 0,
        "skipped_courses": 0,
    }

    with driver.session() as session:
        for item in validated_items:
            # 1. 创建/合并 Topic 节点
            result_topic = session.run(
                f"""
                MERGE (t:{LABEL_TOPIC} {{name: $topic}})
                ON CREATE SET t.created_at = $now
                RETURN t.created_at AS created
                """,
                topic=item.topic,
                now=datetime.utcnow().isoformat(),
            )
            record = result_topic.single()
            if record and record["created"]:
                counts["topics_created"] += 1

            # 2. 创建/合并 Domain 节点
            result_domain = session.run(
                f"""
                MERGE (d:{LABEL_DOMAIN} {{name: $domain}})
                ON CREATE SET d.created_at = $now
                RETURN d.created_at AS created
                """,
                domain=item.domain,
                now=datetime.utcnow().isoformat(),
            )
            record = result_domain.single()
            if record and record["created"]:
                counts["domains_created"] += 1

            # 3. Topic ->[:IN_DOMAIN]-> Domain
            result_in_domain = session.run(
                f"""
                MATCH (t:{LABEL_TOPIC} {{name: $topic}})
                MATCH (d:{LABEL_DOMAIN} {{name: $domain}})
                MERGE (t)-[r:{REL_IN_DOMAIN}]->(d)
                ON CREATE SET r.created_at = $now
                RETURN r.created_at AS created
                """,
                topic=item.topic,
                domain=item.domain,
                now=datetime.utcnow().isoformat(),
            )
            record = result_in_domain.single()
            if record and record["created"]:
                counts["in_domain_rel_created"] += 1

            # 4. 创建/合并 SubTopic 节点
            for st in item.subtopics:
                result_st = session.run(
                    f"""
                    MERGE (st:{LABEL_SUBTOPIC} {{name: $st_name, parent_topic: $topic}})
                    ON CREATE SET st.created_at = $now
                    RETURN st.created_at AS created
                    """,
                    st_name=st,
                    topic=item.topic,
                    now=datetime.utcnow().isoformat(),
                )
                record = result_st.single()
                if record and record["created"]:
                    counts["subtopics_created"] += 1

                # Topic ->[:HAS_SUBTOPIC]-> SubTopic
                result_has_sub = session.run(
                    f"""
                    MATCH (t:{LABEL_TOPIC} {{name: $topic}})
                    MATCH (st:{LABEL_SUBTOPIC} {{name: $st_name, parent_topic: $topic}})
                    MERGE (t)-[r:{REL_HAS_SUBTOPIC}]->(st)
                    ON CREATE SET r.created_at = $now
                    RETURN r.created_at AS created
                    """,
                    topic=item.topic,
                    st_name=st,
                    now=datetime.utcnow().isoformat(),
                )
                record = result_has_sub.single()
                if record and record["created"]:
                    counts["has_subtopic_rel_created"] += 1

                # SubTopic ->[:IN_DOMAIN]-> Domain
                result_st_domain = session.run(
                    f"""
                    MATCH (st:{LABEL_SUBTOPIC} {{name: $st_name, parent_topic: $topic}})
                    MATCH (d:{LABEL_DOMAIN} {{name: $domain}})
                    MERGE (st)-[r:{REL_IN_DOMAIN}]->(d)
                    ON CREATE SET r.created_at = $now
                    RETURN r.created_at AS created
                    """,
                    st_name=st,
                    topic=item.topic,
                    domain=item.domain,
                    now=datetime.utcnow().isoformat(),
                )
                record = result_st_domain.single()
                if record and record["created"]:
                    counts["in_domain_rel_created"] += 1

            # 5. 处理 course_mappings
            for course_name in item.course_mappings:
                # 检查课程是否存在
                check = session.run(
                    f"""
                    MATCH (c:{LABEL_COURSE} {{name: $course_name}})
                    RETURN c.name AS name
                    """,
                    course_name=course_name,
                )
                if check.single() is None:
                    warnings.warn(
                        f"Course '{course_name}' not found in Neo4j; skipping "
                        f"HAS_TOPIC/COVERS_SUBTOPIC for topic '{item.topic}'.",
                        stacklevel=2,
                    )
                    counts["skipped_courses"] += 1
                    continue

                # Course ->[:HAS_TOPIC]-> Topic
                result_has_topic = session.run(
                    f"""
                    MATCH (c:{LABEL_COURSE} {{name: $course_name}})
                    MATCH (t:{LABEL_TOPIC} {{name: $topic}})
                    MERGE (c)-[r:{REL_HAS_TOPIC}]->(t)
                    ON CREATE SET r.created_at = $now
                    RETURN r.created_at AS created
                    """,
                    course_name=course_name,
                    topic=item.topic,
                    now=datetime.utcnow().isoformat(),
                )
                record = result_has_topic.single()
                if record and record["created"]:
                    counts["has_topic_rel_created"] += 1

                # Course ->[:COVERS_SUBTOPIC]-> SubTopic（对每个 subtopic）
                for st in item.subtopics:
                    result_covers = session.run(
                        f"""
                        MATCH (c:{LABEL_COURSE} {{name: $course_name}})
                        MATCH (st:{LABEL_SUBTOPIC} {{name: $st_name, parent_topic: $topic}})
                        MERGE (c)-[r:{REL_COVERS_SUBTOPIC}]->(st)
                        ON CREATE SET r.created_at = $now
                        RETURN r.created_at AS created
                        """,
                        course_name=course_name,
                        st_name=st,
                        topic=item.topic,
                        now=datetime.utcnow().isoformat(),
                    )
                    record = result_covers.single()
                    if record and record["created"]:
                        counts["covers_subtopic_rel_created"] += 1

    return counts
