"""图数据模型定义：节点标签、关系类型常量 以及 Schema 初始化 Cypher"""

# 节点标签
LABEL_MAJOR = "Major"
LABEL_COURSE = "Course"
LABEL_KNOWLEDGE_POINT = "KnowledgeConcept"
LABEL_TOPIC = "Topic"
LABEL_SUBTOPIC = "SubTopic"
LABEL_DOMAIN = "Domain"
LABEL_USER = "User"

# 关系类型
REL_BELONGS_TO = "HAS_COURSE"          # Major → Course
REL_COVERS = "COVERS_KNOWLEDGE"        # Course → KnowledgeConcept
REL_PREREQUISITE_OF = "CONCEPT_PREREQUISITE_FOR" # KnowledgeConcept → KnowledgeConcept
REL_PREREQUISITE_FOR = "PREREQUISITE_FOR" # Course → Course
REL_RELATED_TO = "RELATED_TO"           # KnowledgeConcept ↔ KnowledgeConcept

# 新增：P0 本体扩展
REL_HAS_TOPIC = "HAS_TOPIC"             # Course → Topic
REL_HAS_SUBTOPIC = "HAS_SUBTOPIC"       # Topic → SubTopic
REL_COVERS_SUBTOPIC = "COVERS_SUBTOPIC" # Course → SubTopic
REL_IN_DOMAIN = "IN_DOMAIN"             # Topic/SubTopic/Course → Domain
REL_SEMANTIC_SIMILARITY = "SEMANTIC_SIMILARITY"  # Concept↔Concept / Course↔Course

# 新增：P1 预测前置关系（与原始人工边分离，避免覆盖）
REL_PREDICTED_PREREQ = "PREDICTED_PREREQUISITE"  # KnowledgeConcept → KnowledgeConcept

# 创建约束和索引的 Cypher 语句
CREATE_CONSTRAINTS = [
    # Major 的 name+university 唯一复合约束
    f"""
    CREATE CONSTRAINT major_unique IF NOT EXISTS
    FOR (m:{LABEL_MAJOR})
    REQUIRE (m.name, m.university) IS UNIQUE
    """,
    # KnowledgeConcept 的 name 唯一约束
    f"""
    CREATE CONSTRAINT knowledge_unique IF NOT EXISTS
    FOR (k:{LABEL_KNOWLEDGE_POINT})
    REQUIRE k.name IS UNIQUE
    """,
    # Topic 的 name 唯一约束
    f"""
    CREATE CONSTRAINT topic_unique IF NOT EXISTS
    FOR (t:{LABEL_TOPIC})
    REQUIRE t.name IS UNIQUE
    """,
    # SubTopic 的 (name, parent_topic) 复合唯一约束
    f"""
    CREATE CONSTRAINT subtopic_unique IF NOT EXISTS
    FOR (st:{LABEL_SUBTOPIC})
    REQUIRE (st.name, st.parent_topic) IS UNIQUE
    """,
    # Domain 的 name 唯一约束
    f"""
    CREATE CONSTRAINT domain_unique IF NOT EXISTS
    FOR (d:{LABEL_DOMAIN})
    REQUIRE d.name IS UNIQUE
    """,
    # User 的 user_id 唯一约束
    f"""
    CREATE CONSTRAINT user_unique IF NOT EXISTS
    FOR (u:{LABEL_USER})
    REQUIRE u.user_id IS UNIQUE
    """,
]

CREATE_INDEXES = [
    f"""
    CREATE INDEX course_name IF NOT EXISTS
    FOR (c:{LABEL_COURSE})
    ON (c.name)
    """,
    f"""
    CREATE INDEX topic_name IF NOT EXISTS
    FOR (t:{LABEL_TOPIC})
    ON (t.name)
    """,
    f"""
    CREATE INDEX subtopic_name IF NOT EXISTS
    FOR (st:{LABEL_SUBTOPIC})
    ON (st.name)
    """,
    f"""
    CREATE INDEX domain_name IF NOT EXISTS
    FOR (d:{LABEL_DOMAIN})
    ON (d.name)
    """,
]

# 完整 Schema 初始化：先约束再索引
SCHEMA_CYPHER = CREATE_CONSTRAINTS + CREATE_INDEXES


def init_schema(driver):
    """在 Neo4j 中执行所有 Schema 初始化语句"""
    with driver.session() as session:
        for stmt in SCHEMA_CYPHER:
            session.run(stmt)
    return len(SCHEMA_CYPHER)
