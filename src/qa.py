"""Learning-domain KBQA over the Neo4j knowledge graph."""
import json
import os
import re
from difflib import SequenceMatcher, get_close_matches
from typing import Optional

from src.config import DEEPSEEK_MODEL, get_deepseek_client
from src.models.schema import (
    LABEL_COURSE,
    LABEL_KNOWLEDGE_POINT,
    LABEL_MAJOR,
    REL_BELONGS_TO,
    REL_COVERS,
    REL_PREREQUISITE_OF,
    REL_PREREQUISITE_FOR,
)
from src.recommendation.path_finder import find_path_to_target

REL_SHARES_KNOWLEDGE = "SHARES_KNOWLEDGE"


SUGGESTED_QUESTIONS = [
    "人工智能专业有哪些课程？",
    "机器学习覆盖哪些知识点？",
    "线性代数属于哪些课程？",
    "深度学习需要先学什么？",
    "给我推荐学习深度学习的路径",
]

LLM_PARSE_SYSTEM_PROMPT = """你是学习图谱问答的解析器。只抽取用户问题里的结构化信息，不回答问题。

输出严格 JSON，不要 Markdown，不要解释。字段：
{
  "intent": "major_courses" | "major_overview" | "course_knowledge" | "knowledge_courses" | "prerequisites" | "learning_path" | "unknown",
  "target": "用户主要想了解或学习的专业/课程/知识点原词；没有则为空字符串",
  "target_type": "Major" | "Course" | "KnowledgeConcept" | null,
  "known": ["用户明确说已经会/学过/掌握的知识点或课程原词"],
  "goal": "就业/考研/科研/考试/入门/补基础/转专业等；没有则null",
  "weekly_hours": 每周学习小时数；如果用户说每天N小时则换算成N*7；没有则null,
  "level": "beginner" | "intermediate" | "advanced" | null
}

规则：
- target 只填用户想学、想查、想规划的主要对象，不要填已掌握内容。
- known 只填用户已有基础，不要包含 target。
- 如果用户问“X需要先学什么/怎么学/路径”，intent 用 learning_path 或 prerequisites。
- 如果用户问“X有哪些课程”，专业用 major_courses，知识点用 knowledge_courses。
"""

QUESTION_FILLER_TOKENS = [
    "应该",
    "怎么",
    "如何",
    "哪些",
    "什么",
    "学会",
    "想学",
    "想要",
    "达成",
    "目标",
    "路径",
    "路线",
    "推荐",
    "需要",
    "先学",
    "前置",
    "基础",
    "课程",
    "专业",
]

PROFILE_TERM_ALIASES = {
    "概率统计": "概率论与数理统计",
    "概率与统计": "概率论与数理统计",
}


def _make_node_id(label: str, name: str, extra: str = "") -> str:
    parts = [label, name]
    if extra:
        parts.append(extra)
    return "::".join(parts)


def _node(label: str, props: dict) -> dict:
    return {
        "id": _make_node_id(label, props.get("name", ""), props.get("university", "")),
        "name": props.get("name", ""),
        "label": label,
        "properties": props,
    }


def _link(source: dict, target: dict, rel_type: str) -> dict:
    return {
        "source": source["id"],
        "target": target["id"],
        "type": rel_type,
    }


def _fetch_entities(session) -> dict[str, list[dict]]:
    result = session.run(
        f"""
        MATCH (n)
        WHERE n:{LABEL_MAJOR} OR n:{LABEL_COURSE} OR n:{LABEL_KNOWLEDGE_POINT}
        RETURN labels(n)[0] AS label, properties(n) AS props
        """
    )
    entities = {LABEL_MAJOR: [], LABEL_COURSE: [], LABEL_KNOWLEDGE_POINT: []}
    for row in result:
        entities.setdefault(row["label"], []).append(dict(row["props"]))
    return entities


def _question_terms(question: str) -> list[str]:
    text = question
    for token in QUESTION_FILLER_TOKENS:
        text = text.replace(token, " ")
    terms = re.split(r"[\s，。！？、,.!?;；:：()（）\"'“”]+", text)
    terms = [term.strip() for term in terms if len(term.strip()) >= 2]
    return sorted(set(terms), key=len, reverse=True)


def _best_entity(question: str, entities: dict[str, list[dict]], label: str) -> Optional[dict]:
    candidates = entities.get(label, [])
    if not candidates:
        return None

    text = question.strip()
    exact = [item for item in candidates if item.get("name") == text]
    if exact:
        return exact[0]

    contained = [item for item in candidates if item.get("name") and item["name"] in text]
    if contained:
        return sorted(contained, key=lambda item: len(item.get("name", "")), reverse=True)[0]

    terms = _question_terms(question)
    for term in terms:
        exact = [item for item in candidates if item.get("name") == term]
        if exact:
            return exact[0]
        matches = [item for item in candidates if item.get("name") and term in item["name"]]
        if matches:
            return sorted(matches, key=lambda item: len(item.get("name", "")))[0]

    names = [item.get("name", "") for item in candidates if item.get("name")]
    for term in terms:
        matches = get_close_matches(term, names, n=1, cutoff=0.68)
        if matches:
            return next(item for item in candidates if item.get("name") == matches[0])

    scored = []
    for term in terms:
        scored.extend(
            (SequenceMatcher(None, term, item.get("name", "")).ratio(), item)
            for item in candidates
        )
    scored.sort(reverse=True, key=lambda x: x[0])
    return scored[0][1] if scored and scored[0][0] >= 0.62 else None


def _matched(label: str, props: Optional[dict]) -> list[dict]:
    if not props:
        return []
    return [{"label": label, "name": props.get("name", ""), "properties": props}]


def _unknown_answer(question: str, entities: dict[str, list[dict]]) -> dict:
    names = []
    for label in (LABEL_MAJOR, LABEL_COURSE, LABEL_KNOWLEDGE_POINT):
        names.extend(item.get("name", "") for item in entities.get(label, [])[:4])
    sample = "、".join([name for name in names if name][:8])
    answer = "我还没有匹配到明确的专业、课程或知识概念。"
    if sample:
        answer += f" 可以试试这样问：{sample} 相关的问题。"
    return {
        "answer": answer,
        "intent": "unknown",
        "matched_entities": [],
        "evidence_nodes": [],
        "evidence_links": [],
        "suggested_questions": SUGGESTED_QUESTIONS,
        "raw_question": question,
    }


def _clean_llm_json(raw: str) -> str:
    raw = raw.strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return raw


def _parse_question_with_llm(question: str) -> dict:
    if not os.getenv("DEEPSEEK_API_KEY"):
        return {}
    try:
        client = get_deepseek_client()
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": LLM_PARSE_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=0,
            timeout=5,
        )
        parsed = json.loads(_clean_llm_json(response.choices[0].message.content or ""))
    except Exception:
        return {}
    if not isinstance(parsed, dict):
        return {}
    known = parsed.get("known") or []
    if not isinstance(known, list):
        known = []
    return {
        "intent": parsed.get("intent") or "",
        "target": parsed.get("target") or "",
        "target_type": parsed.get("target_type"),
        "known": [str(item).strip() for item in known if str(item).strip()],
        "goal": parsed.get("goal"),
        "weekly_hours": parsed.get("weekly_hours"),
        "level": parsed.get("level"),
    }


def _number_or_none(value) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _split_profile_terms(text: str) -> list[str]:
    terms = re.split(r"[\s，。！？、,.!?;；:：/和与及]+", text)
    noise = {"我", "已经", "已", "会", "学过", "掌握", "有", "基础", "一点", "一些"}
    result = []
    for term in terms:
        clean = term.strip()
        for token in noise:
            clean = clean.replace(token, "")
        if len(clean) >= 2:
            result.append(clean)
    return result


def _extract_known_terms(question: str) -> list[str]:
    patterns = [
        r"(?:我)?(?:已经|已)?(?:(?<!不)会|学过|掌握)(.+?)(?:，|。|；|;|,|\.|想|希望|目标|接下来|然后|请|帮|怎么|如何|推荐|应该|需要|学习路径|路线|$)",
        r"(?:我)?有(.+?)基础",
    ]
    terms = []
    for pattern in patterns:
        for match in re.finditer(pattern, question):
            terms.extend(_split_profile_terms(match.group(1)))
    return terms


def _extract_goal(question: str) -> Optional[str]:
    for token in ["就业", "找工作", "考研", "科研", "考试", "转专业", "入门", "补基础"]:
        if token in question:
            return token
    return None


def _extract_weekly_hours(question: str) -> Optional[float]:
    match = re.search(r"每周\s*(\d+(?:\.\d+)?)\s*(?:个)?小时", question)
    if match:
        return float(match.group(1))
    match = re.search(r"每天\s*(\d+(?:\.\d+)?)\s*(?:个)?小时", question)
    if match:
        return float(match.group(1)) * 7
    return None


def _extract_level(question: str) -> Optional[str]:
    if any(token in question for token in ["零基础", "小白", "初学"]):
        return "beginner"
    if any(token in question for token in ["有基础", "学过一些", "入门过"]):
        return "intermediate"
    if any(token in question for token in ["进阶", "深入", "高级"]):
        return "advanced"
    return None


def _extract_target_text(question: str) -> str:
    patterns = [
        r"(?:想学|想要学|学习|学会|掌握|目标是|目标为)(.+?)(?:，|。|；|;|,|\.|每周|每天|已经|已|我会|学过|有.+?基础|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, question)
        if match:
            text = match.group(1).strip()
            if len(text) >= 2:
                return text
    return question


def _match_known_terms(terms: list[str], entities: dict[str, list[dict]]) -> list[dict]:
    matched = []
    seen = set()
    for term in terms:
        item = _best_entity(PROFILE_TERM_ALIASES.get(term, term), entities, LABEL_KNOWLEDGE_POINT)
        if item and item.get("name") not in seen:
            seen.add(item.get("name"))
            matched.append(item)
    return matched


def _build_profile(
    question: str,
    entities: dict[str, list[dict]],
    known: Optional[list[str]] = None,
    goal: Optional[str] = None,
    weekly_hours: Optional[float] = None,
    level: Optional[str] = None,
    parsed: Optional[dict] = None,
) -> dict:
    known_terms = []
    if known:
        known_terms.extend(known)
    if parsed and parsed.get("known"):
        known_terms.extend(parsed["known"])
    known_terms.extend(_extract_known_terms(question))
    known_nodes = _match_known_terms(known_terms, entities)
    parsed_hours = _number_or_none(parsed.get("weekly_hours")) if parsed else None
    return {
        "known": [node.get("name", "") for node in known_nodes if node.get("name")],
        "goal": goal or (parsed.get("goal") if parsed else None) or _extract_goal(question),
        "weekly_hours": (
            weekly_hours
            if weekly_hours is not None
            else parsed_hours if parsed_hours is not None else _extract_weekly_hours(question)
        ),
        "level": level or (parsed.get("level") if parsed else None) or _extract_level(question),
    }


def _personalized_path_answer(session, question: str, knowledge: dict, profile: dict) -> Optional[dict]:
    known_names = profile.get("known", [])
    if not _has_profile(profile):
        return None

    path = find_path_to_target(session, known_names, knowledge["name"])
    if "error" in path:
        return None

    # 尝试使用双图 RL 推荐增强路径（P2）
    rl_enhanced = None
    try:
        from src.rl.recommend import recommend_path
        rl_enhanced = recommend_path(
            known_concepts=known_names,
            target_concept=knowledge["name"],
            level=profile.get("level", "intermediate"),
            weekly_hours=profile.get("weekly_hours"),
            goal=profile.get("goal"),
        )
    except Exception:
        rl_enhanced = None

    all_names = list(dict.fromkeys([
        *known_names,
        *path.get("need_to_learn", []),
        knowledge["name"],
        *[item["concept"] for item in (rl_enhanced.path if rl_enhanced else [])],
    ]))
    node_result = session.run(
        f"""
        MATCH (k:{LABEL_KNOWLEDGE_POINT})
        WHERE k.name IN $names
        RETURN properties(k) AS props
        """,
        names=all_names,
    )
    nodes = [_node(LABEL_KNOWLEDGE_POINT, dict(row["props"])) for row in node_result]
    node_ids = {node["id"] for node in nodes}
    edge_result = session.run(
        f"""
        MATCH (a:{LABEL_KNOWLEDGE_POINT})-[:{REL_PREREQUISITE_OF}]->(b:{LABEL_KNOWLEDGE_POINT})
        WHERE a.name IN $names AND b.name IN $names
        RETURN properties(a) AS source, properties(b) AS target
        LIMIT 160
        """,
        names=all_names,
    )
    links = []
    for row in edge_result:
        source = _node(LABEL_KNOWLEDGE_POINT, dict(row["source"]))
        target = _node(LABEL_KNOWLEDGE_POINT, dict(row["target"]))
        if source["id"] in node_ids and target["id"] in node_ids:
            links.append(_link(source, target, REL_PREREQUISITE_OF))

    remaining = path.get("need_to_learn", [])
    parts = [f"针对你想学 {knowledge['name']} 的目标"]
    if known_names:
        parts.append(f"已识别你掌握：{'、'.join(known_names)}")
    if remaining:
        parts.append(f"还需要学习 {len(remaining)} 个知识点：{'、'.join(remaining[:18])}{' 等' if len(remaining) > 18 else ''}")
    else:
        parts.append("当前图谱中的前置知识已经被你的已掌握知识覆盖，可以直接进入目标内容")
    if rl_enhanced and rl_enhanced.path:
        rl_steps = [item["concept"] for item in rl_enhanced.path if item["type"] != "target"][:8]
        if rl_steps:
            parts.append(f"智能推荐引擎给出的关键学习步骤：{' → '.join(rl_steps)}")
        if rl_enhanced.blocked_recovery:
            bridges = [f"{b['concept']}（辅助理解 {b['reason'].split('缺失前置')[1].split('（')[0]}）" for b in rl_enhanced.blocked_recovery[:2] if '缺失前置' in b['reason']]
            if bridges:
                parts.append(f"若遇到前置阻塞，可借助相似知识突破：{'；'.join(bridges)}")
    if profile.get("level") == "beginner":
        parts.append("因为你是零基础/初学者，建议按阶段顺序推进，不要跳过第一阶段")
    elif profile.get("level") == "advanced":
        parts.append("如果你已有进阶基础，可以优先核对未掌握清单，再进入目标知识")
    if profile.get("weekly_hours"):
        hours = profile["weekly_hours"]
        stage_count = max(path.get("depth", 1), 1)
        parts.append(f"按每周约 {hours:g} 小时估算，可把 {stage_count} 个阶段拆成每阶段 1-2 周推进")
    if profile.get("goal"):
        parts.append(f"你的目标偏向“{profile['goal']}”，当前会先保证依赖顺序正确；图谱暂未包含岗位/考试权重")

    return {
        "answer": "；".join(parts) + "。",
        "intent": "personalized_path",
        "matched_entities": _matched(LABEL_KNOWLEDGE_POINT, knowledge),
        "evidence_nodes": nodes,
        "evidence_links": links,
        "suggested_questions": [
            f"我已经会{knowledge['name']}的前置基础，下一步学什么？",
            f"零基础每周 6 小时怎么学{knowledge['name']}？",
            f"{knowledge['name']}属于哪些课程？",
        ],
        "personalization": profile,
        "path": path,
        "raw_question": question,
    }


def _has_profile(profile: dict) -> bool:
    return any([
        profile.get("known"),
        profile.get("goal"),
        profile.get("weekly_hours"),
        profile.get("level"),
    ])


def _personalized_course_answer(session, question: str, course: dict, profile: dict) -> Optional[dict]:
    if not _has_profile(profile):
        return None

    result = session.run(
        f"""
        MATCH (c:{LABEL_COURSE} {{name: $name}})
        OPTIONAL MATCH (pc:{LABEL_COURSE})-[:{REL_PREREQUISITE_FOR}]->(c)
        OPTIONAL MATCH (c)-[:{REL_COVERS}]->(k:{LABEL_KNOWLEDGE_POINT})
        OPTIONAL MATCH (p:{LABEL_KNOWLEDGE_POINT})-[:{REL_PREREQUISITE_OF}]->(k)
        RETURN properties(c) AS course,
               collect(DISTINCT properties(pc)) AS prereq_courses,
               collect(DISTINCT properties(k)) AS covered_knowledge,
               collect(DISTINCT properties(p)) AS prereq_knowledge
        """,
        name=course["name"],
    )
    row = result.single()
    if not row:
        return None

    course_node = _node(LABEL_COURSE, dict(row["course"]) if row["course"] else course)
    prereq_courses = [p for p in row["prereq_courses"] if p]
    covered_knowledge = [p for p in row["covered_knowledge"] if p]
    prereq_knowledge = [p for p in row["prereq_knowledge"] if p]
    known_set = set(profile.get("known", []))
    remaining_prereqs = [p for p in prereq_knowledge if p.get("name") not in known_set]
    covered_nodes = [_node(LABEL_KNOWLEDGE_POINT, p) for p in covered_knowledge[:24]]
    prereq_nodes = [_node(LABEL_KNOWLEDGE_POINT, p) for p in remaining_prereqs[:24]]
    prereq_course_nodes = [_node(LABEL_COURSE, p) for p in prereq_courses[:16]]

    parts = [f"针对你想学 {course['name']}"]
    if profile.get("known"):
        parts.append(f"已识别你掌握：{'、'.join(profile['known'])}")
    if prereq_courses:
        parts.append(f"前置课程包括：{'、'.join(p.get('name', '') for p in prereq_courses[:8])}")
    if remaining_prereqs:
        parts.append(f"仍建议补齐：{'、'.join(p.get('name', '') for p in remaining_prereqs[:12])}")
    elif prereq_knowledge:
        parts.append("当前记录的前置知识已被你的已掌握清单覆盖")
    elif covered_knowledge:
        parts.append(f"图谱没有明确前置知识，可先了解课程覆盖的核心概念：{'、'.join(p.get('name', '') for p in covered_knowledge[:10])}")
    else:
        parts.append("当前图谱没有记录明确前置课程或知识概念")
    if profile.get("weekly_hours"):
        parts.append(f"按每周约 {profile['weekly_hours']:g} 小时，建议先用 1-2 周补前置，再进入课程主体")
    if profile.get("goal"):
        parts.append(f"你的目标偏向“{profile['goal']}”，当前会先保证依赖顺序正确；图谱暂未包含岗位/考试权重")
    if profile.get("level") == "beginner":
        parts.append("零基础建议先从前置课程和基础概念开始")

    links = [_link(node, course_node, REL_PREREQUISITE_FOR) for node in prereq_course_nodes]
    links.extend(_link(course_node, node, REL_COVERS) for node in covered_nodes)
    return {
        "answer": "；".join(parts) + "。",
        "intent": "personalized_course_path",
        "matched_entities": _matched(LABEL_COURSE, course),
        "evidence_nodes": [course_node, *prereq_course_nodes, *covered_nodes, *prereq_nodes],
        "evidence_links": links,
        "suggested_questions": [
            f"{course['name']}覆盖哪些知识点？",
            f"学习{course['name']}前需要哪些基础？",
            "人工智能专业有哪些课程？",
        ],
        "personalization": profile,
        "raw_question": question,
    }


def _major_courses(session, major: dict) -> dict:
    result = session.run(
        f"""
        MATCH (m:{LABEL_MAJOR} {{name: $name}})-[:{REL_BELONGS_TO}]->(c:{LABEL_COURSE})
        RETURN properties(m) AS major, properties(c) AS course
        ORDER BY c.name
        LIMIT 80
        """,
        name=major["name"],
    )
    rows = [dict(r) for r in result]
    major_node = _node(LABEL_MAJOR, rows[0]["major"] if rows else major)
    course_nodes = [_node(LABEL_COURSE, row["course"]) for row in rows]
    course_ids = [node["properties"].get("id") for node in course_nodes if node["properties"].get("id")]
    prereq_result = session.run(
        f"""
        MATCH (a:{LABEL_COURSE})-[:{REL_PREREQUISITE_FOR}]->(b:{LABEL_COURSE})
        WHERE a.id IN $course_ids AND b.id IN $course_ids
        RETURN properties(a) AS source, properties(b) AS target
        ORDER BY source.name, target.name
        LIMIT 120
        """,
        course_ids=course_ids,
    )
    prereq_links = [
        _link(_node(LABEL_COURSE, row["source"]), _node(LABEL_COURSE, row["target"]), REL_PREREQUISITE_FOR)
        for row in prereq_result
    ]
    related_result = session.run(
        f"""
        MATCH (a:{LABEL_COURSE})-[:{REL_COVERS}]->(k:{LABEL_KNOWLEDGE_POINT})<-[:{REL_COVERS}]-(b:{LABEL_COURSE})
        WHERE a.id IN $course_ids AND b.id IN $course_ids AND a.id < b.id
        WITH a, b, count(DISTINCT k) AS shared_count
        ORDER BY shared_count DESC, a.name, b.name
        LIMIT 80
        RETURN properties(a) AS source, properties(b) AS target, shared_count
        """,
        course_ids=course_ids,
    )
    prereq_keys = {
        (link["source"], link["target"])
        for link in prereq_links
    }
    related_links = []
    for row in related_result:
        link = _link(
            _node(LABEL_COURSE, row["source"]),
            _node(LABEL_COURSE, row["target"]),
            REL_SHARES_KNOWLEDGE,
        )
        if (link["source"], link["target"]) not in prereq_keys:
            link["shared_count"] = row["shared_count"]
            related_links.append(link)
    names = [n["name"] for n in course_nodes]
    answer = f"{major['name']}专业关联 {len(names)} 门课程"
    if names:
        answer += f"，包括：{'、'.join(names[:18])}"
        if len(names) > 18:
            answer += f" 等。"
        else:
            answer += "。"
        if prereq_links:
            answer += f" 这些课程之间还有 {len(prereq_links)} 条先后/前置关系，已在图中用学习路径线标出。"
        if related_links:
            answer += f" 另外补充显示了 {len(related_links)} 条共享知识点的课程关联线，用来观察课程之间的内容联系。"
    else:
        answer += "，当前图谱里还没有课程关系。"
    return {
        "answer": answer,
        "intent": "major_courses",
        "matched_entities": _matched(LABEL_MAJOR, major),
        "evidence_nodes": [major_node, *course_nodes],
        "evidence_links": [
            *[_link(major_node, node, REL_BELONGS_TO) for node in course_nodes],
            *prereq_links,
            *related_links,
        ],
        "suggested_questions": [
            f"{major['name']}专业学什么？",
            "机器学习覆盖哪些知识点？",
            "给我推荐学习深度学习的路径",
        ],
    }


def _major_overview(session, major: dict) -> dict:
    result = session.run(
        f"""
        MATCH (m:{LABEL_MAJOR} {{name: $name}})-[:{REL_BELONGS_TO}]->(c:{LABEL_COURSE})
        OPTIONAL MATCH (c)-[:{REL_COVERS}]->(k:{LABEL_KNOWLEDGE_POINT})
        RETURN properties(m) AS major,
               collect(DISTINCT properties(c)) AS courses,
               collect(DISTINCT properties(k)) AS knowledge
        """,
        name=major["name"],
    )
    row = result.single()
    if not row:
        return _major_courses(session, major)
    major_node = _node(LABEL_MAJOR, dict(row["major"]))
    courses = [p for p in row["courses"] if p]
    knowledge = [p for p in row["knowledge"] if p]
    course_nodes = [_node(LABEL_COURSE, p) for p in courses[:30]]
    knowledge_nodes = [_node(LABEL_KNOWLEDGE_POINT, p) for p in knowledge[:30]]
    answer = (
        f"{major['name']}专业在当前图谱中关联 {len(courses)} 门课程、"
        f"{len(knowledge)} 个知识概念。核心课程示例："
        f"{'、'.join([c.get('name', '') for c in courses[:12]]) or '暂无'}。"
    )
    return {
        "answer": answer,
        "intent": "major_overview",
        "matched_entities": _matched(LABEL_MAJOR, major),
        "evidence_nodes": [major_node, *course_nodes, *knowledge_nodes],
        "evidence_links": [_link(major_node, node, REL_BELONGS_TO) for node in course_nodes],
        "suggested_questions": [
            f"{major['name']}专业有哪些课程？",
            "这些课程覆盖哪些知识点？",
            "哪些知识点需要先学？",
        ],
    }


def _course_knowledge(session, course: dict) -> dict:
    result = session.run(
        f"""
        MATCH (c:{LABEL_COURSE} {{name: $name}})-[:{REL_COVERS}]->(k:{LABEL_KNOWLEDGE_POINT})
        RETURN properties(c) AS course, properties(k) AS knowledge
        ORDER BY k.name
        LIMIT 80
        """,
        name=course["name"],
    )
    rows = [dict(r) for r in result]
    course_node = _node(LABEL_COURSE, rows[0]["course"] if rows else course)
    knowledge_nodes = [_node(LABEL_KNOWLEDGE_POINT, row["knowledge"]) for row in rows]
    names = [n["name"] for n in knowledge_nodes]
    answer = f"{course['name']}覆盖 {len(names)} 个知识概念"
    if names:
        answer += f"，包括：{'、'.join(names[:20])}"
        answer += " 等。" if len(names) > 20 else "。"
    else:
        answer += "，当前图谱里还没有覆盖关系。"
    return {
        "answer": answer,
        "intent": "course_knowledge",
        "matched_entities": _matched(LABEL_COURSE, course),
        "evidence_nodes": [course_node, *knowledge_nodes],
        "evidence_links": [_link(course_node, node, REL_COVERS) for node in knowledge_nodes],
        "suggested_questions": [
            f"{course['name']}属于哪些专业？",
            "这些知识点需要先学什么？",
            "线性代数属于哪些课程？",
        ],
    }


def _knowledge_courses(session, knowledge: dict) -> dict:
    result = session.run(
        f"""
        MATCH (c:{LABEL_COURSE})-[:{REL_COVERS}]->(k:{LABEL_KNOWLEDGE_POINT} {{name: $name}})
        RETURN properties(c) AS course, properties(k) AS knowledge
        ORDER BY c.name
        LIMIT 80
        """,
        name=knowledge["name"],
    )
    rows = [dict(r) for r in result]
    knowledge_node = _node(LABEL_KNOWLEDGE_POINT, rows[0]["knowledge"] if rows else knowledge)
    course_nodes = [_node(LABEL_COURSE, row["course"]) for row in rows]
    names = [n["name"] for n in course_nodes]
    answer = f"{knowledge['name']}出现在 {len(names)} 门课程中"
    if names:
        answer += f"，包括：{'、'.join(names[:20])}"
        answer += " 等。" if len(names) > 20 else "。"
    else:
        answer += "，当前图谱里还没有课程覆盖它。"
    return {
        "answer": answer,
        "intent": "knowledge_courses",
        "matched_entities": _matched(LABEL_KNOWLEDGE_POINT, knowledge),
        "evidence_nodes": [knowledge_node, *course_nodes],
        "evidence_links": [_link(node, knowledge_node, REL_COVERS) for node in course_nodes],
        "suggested_questions": [
            f"{knowledge['name']}需要先学什么？",
            "机器学习覆盖哪些知识点？",
            "人工智能专业有哪些课程？",
        ],
    }


def _knowledge_prerequisites(session, knowledge: dict) -> dict:
    result = session.run(
        f"""
        MATCH path = (p:{LABEL_KNOWLEDGE_POINT})-[:{REL_PREREQUISITE_OF}*1..8]->
                     (k:{LABEL_KNOWLEDGE_POINT} {{name: $name}})
        RETURN DISTINCT properties(p) AS prereq, length(path) AS distance
        ORDER BY distance DESC, prereq.name
        LIMIT 80
        """,
        name=knowledge["name"],
    )
    prereqs = [dict(r["prereq"]) for r in result if r["prereq"]]
    target_node = _node(LABEL_KNOWLEDGE_POINT, knowledge)
    prereq_nodes = [_node(LABEL_KNOWLEDGE_POINT, p) for p in prereqs]
    names = [n["name"] for n in prereq_nodes]
    answer = f"学习 {knowledge['name']} 前"
    if names:
        answer += f"，建议先掌握：{'、'.join(names[:20])}"
        answer += " 等。" if len(names) > 20 else "。"
    else:
        answer += "，当前图谱没有记录明确的前置知识。"
    return {
        "answer": answer,
        "intent": "knowledge_prerequisites",
        "matched_entities": _matched(LABEL_KNOWLEDGE_POINT, knowledge),
        "evidence_nodes": [target_node, *prereq_nodes],
        "evidence_links": [_link(node, target_node, REL_PREREQUISITE_OF) for node in prereq_nodes],
        "suggested_questions": [
            f"{knowledge['name']}属于哪些课程？",
            "给我推荐学习深度学习的路径",
            "人工智能专业有哪些课程？",
        ],
    }


def _course_prerequisites(session, course: dict) -> dict:
    result = session.run(
        f"""
        MATCH (c:{LABEL_COURSE} {{name: $name}})
        OPTIONAL MATCH (pc:{LABEL_COURSE})-[:{REL_PREREQUISITE_FOR}]->(c)
        OPTIONAL MATCH (c)-[:{REL_COVERS}]->(k:{LABEL_KNOWLEDGE_POINT})
        OPTIONAL MATCH (p:{LABEL_KNOWLEDGE_POINT})-[:{REL_PREREQUISITE_OF}]->(k)
        RETURN properties(c) AS course,
               collect(DISTINCT properties(pc)) AS prereq_courses,
               collect(DISTINCT properties(k)) AS covered_knowledge,
               collect(DISTINCT properties(p)) AS prereq_knowledge
        """,
        name=course["name"],
    )
    row = result.single()
    course_node = _node(LABEL_COURSE, dict(row["course"]) if row and row["course"] else course)
    prereq_courses = [p for p in (row["prereq_courses"] if row else []) if p]
    covered_knowledge = [p for p in (row["covered_knowledge"] if row else []) if p]
    prereq_knowledge = [p for p in (row["prereq_knowledge"] if row else []) if p]
    prereq_course_nodes = [_node(LABEL_COURSE, p) for p in prereq_courses[:20]]
    covered_nodes = [_node(LABEL_KNOWLEDGE_POINT, p) for p in covered_knowledge[:20]]
    prereq_nodes = [_node(LABEL_KNOWLEDGE_POINT, p) for p in prereq_knowledge[:20]]
    knowledge_edge_rows = []
    if covered_knowledge and prereq_knowledge:
        edge_result = session.run(
            f"""
            MATCH (p:{LABEL_KNOWLEDGE_POINT})-[:{REL_PREREQUISITE_OF}]->(k:{LABEL_KNOWLEDGE_POINT})
            WHERE p.name IN $prereq_names AND k.name IN $covered_names
            RETURN properties(p) AS source, properties(k) AS target
            ORDER BY source.name, target.name
            LIMIT 80
            """,
            prereq_names=[p.get("name", "") for p in prereq_knowledge],
            covered_names=[p.get("name", "") for p in covered_knowledge],
        )
        knowledge_edge_rows = [dict(row) for row in edge_result]

    parts = []
    if prereq_courses:
        parts.append(f"前置课程：{'、'.join(p.get('name', '') for p in prereq_courses[:12])}")
    if prereq_knowledge:
        parts.append(f"建议先掌握：{'、'.join(p.get('name', '') for p in prereq_knowledge[:12])}")
    if not parts and covered_knowledge:
        parts.append(f"当前没有明确前置记录，可先了解它覆盖的知识概念：{'、'.join(p.get('name', '') for p in covered_knowledge[:12])}")
    answer = f"学习 {course['name']} 前，" + ("；".join(parts) + "。" if parts else "当前图谱没有记录明确的前置课程或前置知识。")

    links = [_link(node, course_node, REL_PREREQUISITE_FOR) for node in prereq_course_nodes]
    links.extend(_link(course_node, node, REL_COVERS) for node in covered_nodes)
    links.extend(
        _link(
            _node(LABEL_KNOWLEDGE_POINT, row["source"]),
            _node(LABEL_KNOWLEDGE_POINT, row["target"]),
            REL_PREREQUISITE_OF,
        )
        for row in knowledge_edge_rows
    )
    return {
        "answer": answer,
        "intent": "course_prerequisites",
        "matched_entities": _matched(LABEL_COURSE, course),
        "evidence_nodes": [course_node, *prereq_course_nodes, *covered_nodes, *prereq_nodes],
        "evidence_links": links,
        "suggested_questions": [
            f"{course['name']}覆盖哪些知识点？",
            "线性代数属于哪些课程？",
            "人工智能专业有哪些课程？",
        ],
    }


def _course_lookup(session, course: dict) -> dict:
    term = course.get("name", "").split("(")[0].strip() or course.get("name", "")
    result = session.run(
        f"""
        MATCH (c:{LABEL_COURSE})
        WHERE c.name CONTAINS $term OR c.name = $name
        RETURN properties(c) AS course
        ORDER BY c.name
        LIMIT 30
        """,
        term=term,
        name=course["name"],
    )
    courses = [dict(r["course"]) for r in result if r["course"]]
    course_nodes = [_node(LABEL_COURSE, p) for p in courses]
    names = [node["name"] for node in course_nodes]
    answer = f"{term}更像课程名称。当前图谱中匹配到 {len(names)} 门相关课程"
    if names:
        answer += f"：{'、'.join(names[:20])}"
        answer += " 等。" if len(names) > 20 else "。"
    else:
        answer += "，但没有找到更具体的课程条目。"
    return {
        "answer": answer,
        "intent": "course_lookup",
        "matched_entities": _matched(LABEL_COURSE, course),
        "evidence_nodes": course_nodes,
        "evidence_links": [],
        "suggested_questions": [
            f"{course['name']}覆盖哪些知识点？",
            "机器学习覆盖哪些知识点？",
            "人工智能专业有哪些课程？",
        ],
    }


def _infer_intent(
    question: str,
    entities: dict[str, list[dict]],
    parsed: Optional[dict] = None,
) -> tuple[str, str, Optional[dict]]:
    target_text = (parsed or {}).get("target") or _extract_target_text(question)
    parsed_type = (parsed or {}).get("target_type")
    parsed_intent = (parsed or {}).get("intent") or ""

    major = _best_entity(target_text, entities, LABEL_MAJOR)
    course = _best_entity(target_text, entities, LABEL_COURSE)
    knowledge = _best_entity(target_text, entities, LABEL_KNOWLEDGE_POINT)
    path_tokens = [
        "先学",
        "前置",
        "前驱",
        "基础",
        "路径",
        "推荐",
        "怎么学",
        "学习路线",
        "学习路径",
        "达成",
        "目标",
        "掌握",
        "想学",
        "学会",
    ]

    if parsed_intent in ("learning_path", "prerequisites"):
        if parsed_type == LABEL_KNOWLEDGE_POINT and knowledge:
            return "knowledge_prerequisites", LABEL_KNOWLEDGE_POINT, knowledge
        if parsed_type == LABEL_COURSE and course:
            return "course_prerequisites", LABEL_COURSE, course
        if knowledge:
            return "knowledge_prerequisites", LABEL_KNOWLEDGE_POINT, knowledge
        if course:
            return "course_prerequisites", LABEL_COURSE, course
    if parsed_intent == "major_courses" and major:
        return "major_courses", LABEL_MAJOR, major
    if parsed_intent == "major_overview" and major:
        return "major_overview", LABEL_MAJOR, major
    if parsed_intent == "course_knowledge" and course:
        return "course_knowledge", LABEL_COURSE, course
    if parsed_intent == "knowledge_courses" and knowledge:
        return "knowledge_courses", LABEL_KNOWLEDGE_POINT, knowledge

    if knowledge and any(token in question for token in ["属于哪些课程", "哪些课程", "课程", "哪里学"]):
        return "knowledge_courses", LABEL_KNOWLEDGE_POINT, knowledge
    if course and any(token in question for token in ["属于哪些课程", "哪里学"]):
        return "course_lookup", LABEL_COURSE, course
    if knowledge and any(token in question for token in path_tokens):
        return "knowledge_prerequisites", LABEL_KNOWLEDGE_POINT, knowledge
    if course and any(token in question for token in path_tokens):
        return "course_prerequisites", LABEL_COURSE, course
    if major and any(token in question for token in ["有哪些课程", "课程", "课"]):
        return "major_courses", LABEL_MAJOR, major
    if major and any(token in question for token in ["学什么", "概览", "介绍", "是什么"]):
        return "major_overview", LABEL_MAJOR, major
    if course and any(token in question for token in ["覆盖", "知识点", "知识概念", "包含"]):
        return "course_knowledge", LABEL_COURSE, course

    if major:
        return "major_overview", LABEL_MAJOR, major
    if course:
        return "course_knowledge", LABEL_COURSE, course
    if knowledge:
        return "knowledge_courses", LABEL_KNOWLEDGE_POINT, knowledge
    return "unknown", "", None


def _polish_answer(question: str, answer: str) -> str:
    if not os.getenv("DEEPSEEK_API_KEY"):
        return answer
    try:
        client = get_deepseek_client()
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "你只负责润色学习问答答案，不得新增未给出的事实。保持简洁中文。",
                },
                {
                    "role": "user",
                    "content": f"问题：{question}\n图谱答案：{answer}",
                },
            ],
            temperature=0.1,
            timeout=4,
        )
        text = response.choices[0].message.content.strip()
        return text or answer
    except Exception:
        return answer


def answer_question(
    session,
    question: str,
    known: Optional[list[str]] = None,
    goal: Optional[str] = None,
    weekly_hours: Optional[float] = None,
    level: Optional[str] = None,
) -> dict:
    question = question.strip()
    if not question:
        return {
            "answer": "请输入一个关于专业、课程或知识点的问题。",
            "intent": "unknown",
            "matched_entities": [],
            "evidence_nodes": [],
            "evidence_links": [],
            "suggested_questions": SUGGESTED_QUESTIONS,
            "raw_question": question,
        }

    entities = _fetch_entities(session)
    parsed = _parse_question_with_llm(question)
    profile = _build_profile(question, entities, known, goal, weekly_hours, level, parsed)
    intent, label, entity = _infer_intent(question, entities, parsed)
    if intent == "unknown" or not entity:
        result = _unknown_answer(question, entities)
        result["llm_parse"] = parsed
        return result

    if label == LABEL_KNOWLEDGE_POINT and intent in ("knowledge_prerequisites", "knowledge_courses"):
        personalized = _personalized_path_answer(session, question, entity, profile)
        if personalized:
            personalized["llm_parse"] = parsed
            return personalized
    if label == LABEL_COURSE and intent in ("course_prerequisites", "course_knowledge", "course_lookup"):
        personalized = _personalized_course_answer(session, question, entity, profile)
        if personalized:
            personalized["llm_parse"] = parsed
            return personalized

    handlers = {
        "major_courses": _major_courses,
        "major_overview": _major_overview,
        "course_knowledge": _course_knowledge,
        "knowledge_courses": _knowledge_courses,
        "knowledge_prerequisites": _knowledge_prerequisites,
        "course_prerequisites": _course_prerequisites,
        "course_lookup": _course_lookup,
    }
    result = handlers[intent](session, entity)
    result["answer"] = _polish_answer(question, result["answer"])
    result["raw_question"] = question
    result["llm_parse"] = parsed
    return result
