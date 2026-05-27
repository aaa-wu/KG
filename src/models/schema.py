"""图数据模型定义：节点标签、关系类型常量 以及 Schema 初始化 Cypher"""

# 节点标签
LABEL_MAJOR = "Major"
LABEL_COURSE = "Course"
LABEL_KNOWLEDGE_POINT = "KnowledgePoint"

# 关系类型
REL_BELONGS_TO = "BELONGS_TO"          # Major → Course
REL_COVERS = "COVERS"                   # Course → KnowledgePoint
REL_PREREQUISITE_OF = "PREREQUISITE_OF" # KnowledgePoint → KnowledgePoint
REL_RELATED_TO = "RELATED_TO"           # KnowledgePoint ↔ KnowledgePoint

# 创建约束和索引的 Cypher 语句
CREATE_CONSTRAINTS = [
    # Major 的 name+university 唯一复合约束
    f"""
    CREATE CONSTRAINT major_unique IF NOT EXISTS
    FOR (m:{LABEL_MAJOR})
    REQUIRE (m.name, m.university) IS UNIQUE
    """,
    # KnowledgePoint 的 name 唯一约束
    f"""
    CREATE CONSTRAINT knowledge_unique IF NOT EXISTS
    FOR (k:{LABEL_KNOWLEDGE_POINT})
    REQUIRE k.name IS UNIQUE
    """,
]

CREATE_INDEXES = [
    f"""
    CREATE INDEX course_name IF NOT EXISTS
    FOR (c:{LABEL_COURSE})
    ON (c.name)
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
