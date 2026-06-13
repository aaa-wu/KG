"""DeepSeek 大模型接口：意图解析 + 实体匹配 + 结果格式化"""
import json
from src.config import get_neo4j_driver, get_deepseek_client, DEEPSEEK_MODEL
from src.models.schema import LABEL_COURSE, LABEL_KNOWLEDGE_POINT, LABEL_MAJOR


INTENT_SYSTEM_PROMPT = """你是一个学习路径推荐系统的意图解析器。用户会用自然语言描述他们想学什么、已掌握什么。

请分析用户输入，输出一个 JSON，格式如下：
{
  "action": "recommend_path" | "compare_majors" | "explore_major" | "unknown",
  "target": "用户想学的目标专业或知识点名称",
  "known": ["已掌握的知识点1", "已掌握的知识点2", ...],
  "university": "如果用户提到具体学校则提取，否则null"
}

规则：
- action: recommend_path=推荐学习路径, compare_majors=对比两个学校的专业, explore_major=了解某专业学什么, unknown=无法理解
- target: 提取用户想学的目标（专业名或知识点名），尽量简洁
- known: 提取用户说已掌握/学过/会的知识点列表
- 只输出 JSON，不要有其他内容"""


MATCH_SYSTEM_PROMPT = """你是一个知识图谱实体匹配器。用户用口语化的方式描述一个概念，你需要从候选列表中找到最匹配的一项或几项。

请输出一个 JSON 数组，包含匹配到的候选名称，按匹配度排序：
["候选名称1", "候选名称2"]

只输出 JSON 数组，不要有其他内容。如果没有匹配的，输出空数组 []。"""


FORMAT_SYSTEM_PROMPT = """你是一个学习路径规划助手。请根据给出的学习路径数据，用自然、友好的语言为用户规划学习计划。

要求：
- 分阶段呈现学习路径，说明每个阶段的重点
- 解释为什么某些知识需要先学（前驱关系）
- 如果用户已有基础，明确指出可以跳过哪些内容
- 给出建议的学习顺序和时间分配建议
- 用中文回复，语气亲切但不啰嗦"""


def _call_deepseek(system_prompt: str, user_message: str) -> str:
    client = get_deepseek_client()
    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,
    )
    return response.choices[0].message.content.strip()


def parse_intent(user_input: str) -> dict:
    """将用户口语输入解析为结构化意图"""
    raw = _call_deepseek(INTENT_SYSTEM_PROMPT, user_input)
    # 清理可能的 markdown 代码块标记
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {"action": "unknown", "target": None, "known": [], "university": None}
    return result


def match_entities(user_terms: list[str], candidates: list[str]) -> list[str]:
    """将用户的口语化术语匹配到图中的精确节点名称"""
    if not user_terms or not candidates:
        return []

    user_msg = f"用户描述的概念：{json.dumps(user_terms, ensure_ascii=False)}\n候选列表：{json.dumps(candidates, ensure_ascii=False)}"
    raw = _call_deepseek(MATCH_SYSTEM_PROMPT, user_msg)
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return []


def format_plan(path_result: dict) -> str:
    """将图查询结果格式化为自然语言学习计划"""
    user_msg = json.dumps(path_result, ensure_ascii=False)
    return _call_deepseek(FORMAT_SYSTEM_PROMPT, user_msg)


def get_all_knowledge_point_names() -> list[str]:
    """从 Neo4j 获取所有知识点名称列表，用于实体匹配"""
    driver = get_neo4j_driver()
    with driver.session() as session:
        result = session.run(
            f"MATCH (k:{LABEL_KNOWLEDGE_POINT}) RETURN k.name AS name ORDER BY name"
        )
        return [r["name"] for r in result]


def get_all_major_names() -> list[str]:
    """从 Neo4j 获取所有专业名称列表"""
    driver = get_neo4j_driver()
    with driver.session() as session:
        result = session.run(
            f"MATCH (m:{LABEL_MAJOR}) RETURN DISTINCT m.name AS name ORDER BY name"
        )
        return [r["name"] for r in result]


def get_all_course_names() -> list[str]:
    """从 Neo4j 获取所有课程名称列表"""
    driver = get_neo4j_driver()
    with driver.session() as session:
        result = session.run(
            f"MATCH (c:{LABEL_COURSE}) RETURN DISTINCT c.name AS name ORDER BY name"
        )
        return [r["name"] for r in result]


def resolve_target_and_known(user_input: str) -> dict:
    """完整流程：意图解析 + 实体匹配，返回可直接用于图查询的结构化参数"""
    intent = parse_intent(user_input)

    # 匹配目标实体
    all_kps = get_all_knowledge_point_names()
    all_majors = get_all_major_names()
    all_courses = get_all_course_names()

    target = intent.get("target", "")
    known_terms = intent.get("known", [])

    # 尝试匹配 target：优先级 专业 > 课程 > 知识点
    matched_target = None
    target_type = None
    if target:
        all_names = all_majors + all_courses + all_kps
        matches = match_entities([target], all_names)
        if matches:
            matched_name = matches[0]
            if matched_name in all_majors:
                target_type = "Major"
                matched_target = matched_name
            elif matched_name in all_courses:
                target_type = "Course"
                matched_target = matched_name
            elif matched_name in all_kps:
                target_type = LABEL_KNOWLEDGE_POINT
                matched_target = matched_name

    # 匹配已知知识点
    matched_known = match_entities(known_terms, all_kps) if known_terms else []

    return {
        "action": intent.get("action", "unknown"),
        "target": matched_target,
        "target_type": target_type,
        "known": matched_known,
        "university": intent.get("university"),
        "raw_target": target,
        "raw_known": known_terms,
    }
