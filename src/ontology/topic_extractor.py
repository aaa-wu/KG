"""基于 DeepSeek 的 Topic/SubTopic/Domain 自动抽取。

约束：项目没有原始讲义/PPT，因此输入只能是现有的课程名、知识点名和已有描述。
DeepSeek 被提示基于这些名称推断知识模块层次结构。
"""
import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Optional

from src.config import get_deepseek_client, DEEPSEEK_MODEL
from src.models.schema import LABEL_MAJOR, LABEL_COURSE, LABEL_KNOWLEDGE_POINT


@dataclass
class ExtractedTopic:
    """单个 Topic 提案"""
    name: str
    domain: str
    subtopics: list[str] = field(default_factory=list)
    source_major: str = ""
    source_course: str = ""
    concept_hints: list[str] = field(default_factory=list)
    raw_extraction: str = ""


@dataclass
class LLMExtractionResult:
    """一次 Major 级别的完整抽取结果"""
    major: str
    topics: list[ExtractedTopic]
    course_to_topics: dict[str, list[str]]
    concept_to_subtopics: dict[str, list[str]]
    raw_response: str


EXTRACTION_SYSTEM_PROMPT = """你是一位高等教育课程知识图谱专家。请根据下面提供的专业、课程列表和知识点列表，推断出合理的知识主题（Topic）、子主题（SubTopic）和学科领域（Domain）层次结构。

要求：
1. Topic 是比课程更细、比知识点更粗的知识模块，例如"监督学习"、"计算机系统结构"。
2. SubTopic 是具体可学习的单元，例如"随机梯度下降"、"CPU 流水线"。
3. Domain 是更高层的学科领域，例如"人工智能"、"计算机科学"、"数学基础"。
4. 每个 Topic 必须属于一个 Domain；每个 Topic 下包含 2-8 个 SubTopic。
5. 课程到 Topic 的映射：一门课程可以覆盖多个 Topic。
6. 知识点到 SubTopic 的映射：每个知识点尽量归属到最相关的 SubTopic；允许一个知识点对应多个 SubTopic。
7. 不要编造输入中没有出现过的具体课程或知识点名称。
8. 输出严格为 JSON，不要 Markdown 代码块。

输出格式：
{
  "domain": "顶层学科领域",
  "topics": [
    {
      "name": "Topic名称",
      "domain": "所属Domain",
      "subtopics": ["SubTopic1", "SubTopic2", ...]
    }
  ],
  "course_to_topics": {
    "课程名1": ["TopicA", "TopicB"],
    "课程名2": ["TopicB"]
  },
  "concept_to_subtopics": {
    "知识点名1": ["SubTopicX"],
    "知识点名2": ["SubTopicX", "SubTopicY"]
  }
}
"""


def _build_prompt_for_major(
    major_name: str,
    courses: list[dict],
    concepts: list[dict],
) -> str:
    """把 Major 下的课程和知识点组织成 Prompt。"""
    course_lines = []
    for c in courses[:60]:  # 限制 token
        desc = (c.get("description") or "").strip()
        line = f"- {c['name']}" + (f"（{desc[:60]}）" if desc else "")
        course_lines.append(line)

    concept_lines = []
    for k in concepts[:120]:
        desc = (k.get("description") or "").strip()
        line = f"- {k['name']}" + (f"（{desc[:40]}）" if desc else "")
        concept_lines.append(line)

    return f"""专业：{major_name}

课程列表（{len(course_lines)} 门）：
{chr(10).join(course_lines)}

知识点列表（{len(concept_lines)} 个）：
{chr(10).join(concept_lines)}

请基于以上信息生成 Topic/SubTopic/Domain 结构。
"""


def _parse_extraction_json(raw: str) -> Optional[dict]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _deduplicate_topics(topics: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for t in topics:
        key = t.get("name", "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


def extract_topics_for_major(
    major_name: str,
    courses: list[dict],
    concepts: list[dict],
) -> Optional[LLMExtractionResult]:
    """使用 DeepSeek 从现有课程/知识点名中抽取 Topic/SubTopic/Domain 结构。

    如果未配置 DEEPSEEK_API_KEY 或调用失败，返回 None，由调用方 fallback。
    """
    if not os.getenv("DEEPSEEK_API_KEY"):
        return None

    prompt = _build_prompt_for_major(major_name, courses, concepts)

    try:
        client = get_deepseek_client()
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            timeout=60,
        )
        raw = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"DeepSeek extraction failed for major {major_name}: {e}")
        return None

    parsed = _parse_extraction_json(raw)
    if not parsed:
        return None

    topic_dicts = _deduplicate_topics(parsed.get("topics", []))
    topics = []
    for t in topic_dicts:
        subtopics = [s.strip() for s in t.get("subtopics", []) if s and s.strip()]
        topics.append(ExtractedTopic(
            name=t.get("name", "").strip(),
            domain=t.get("domain", parsed.get("domain", "")).strip(),
            subtopics=subtopics,
            source_major=major_name,
            raw_extraction=raw,
        ))

    course_to_topics = {
        k.strip(): [v.strip() for v in vals if v and v.strip()]
        for k, vals in parsed.get("course_to_topics", {}).items()
        if k and k.strip()
    }
    concept_to_subtopics = {
        k.strip(): [v.strip() for v in vals if v and v.strip()]
        for k, vals in parsed.get("concept_to_subtopics", {}).items()
        if k and k.strip()
    }

    return LLMExtractionResult(
        major=major_name,
        topics=topics,
        course_to_topics=course_to_topics,
        concept_to_subtopics=concept_to_subtopics,
        raw_response=raw,
    )


def extract_topics_for_major_from_neo4j(major_name: str) -> Optional[LLMExtractionResult]:
    """直接从 Neo4j 读取指定 Major 的课程和知识点，然后调用 LLM 抽取。"""
    from src.config import get_neo4j_driver
    from src.models.schema import REL_BELONGS_TO, REL_COVERS

    driver = get_neo4j_driver()
    with driver.session() as session:
        courses_result = session.run(f"""
            MATCH (m:{LABEL_MAJOR} {{name: $major}})-[:{REL_BELONGS_TO}]->(c:{LABEL_COURSE})
            RETURN c.name AS name, c.description AS description
            ORDER BY c.name
        """, major=major_name)
        courses = [dict(r) for r in courses_result]

        concepts_result = session.run(f"""
            MATCH (m:{LABEL_MAJOR} {{name: $major}})-[:{REL_BELONGS_TO}]->(c:{LABEL_COURSE})
                  -[:{REL_COVERS}]->(k:{LABEL_KNOWLEDGE_POINT})
            RETURN DISTINCT k.name AS name, k.description AS description
            ORDER BY k.name
        """, major=major_name)
        concepts = [dict(r) for r in concepts_result]

    if not courses:
        return None
    return extract_topics_for_major(major_name, courses, concepts)


def compute_major_fingerprint(major_name: str, courses: list[dict], concepts: list[dict]) -> str:
    """用于校验队列去重的指纹。"""
    text = major_name + "".join(sorted(c["name"] for c in courses))
    text += "".join(sorted(k["name"] for k in concepts))
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]
