"""LangGraph learning-path planning agent backed by the Neo4j graph."""
from __future__ import annotations

import json
import os
import re
import uuid
from difflib import SequenceMatcher, get_close_matches
from typing import Any, Optional, TypedDict

try:
    from langchain_openai import ChatOpenAI
    from langgraph.graph import END, StateGraph
except Exception:  # pragma: no cover - exercised when optional deps are absent.
    ChatOpenAI = None
    StateGraph = None
    END = "__end__"

from src.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, get_neo4j_driver
from src.models.schema import (
    LABEL_COURSE,
    LABEL_KNOWLEDGE_POINT,
    LABEL_MAJOR,
    REL_BELONGS_TO,
    REL_COVERS,
    REL_PREDICTED_PREREQ,
    REL_PREREQUISITE_FOR,
    REL_PREREQUISITE_OF,
)
from src.recommendation.path_finder import find_path_to_target
from src.recommendation.roadmap_classifier import build_module_roadmap


MAX_SESSION_MESSAGES = 20
AGENT_SESSIONS: dict[str, dict[str, Any]] = {}
_COMPILED_GRAPH = None


class AgentState(TypedDict, total=False):
    session_id: str
    message: str
    messages: list[dict[str, str]]
    known: list[str]
    target: Optional[str]
    goal: Optional[str]
    weekly_hours: Optional[float]
    level: Optional[str]
    matched_entities: list[dict[str, Any]]
    target_entity: Optional[dict[str, Any]]
    evidence_nodes: list[dict[str, Any]]
    evidence_links: list[dict[str, Any]]
    plan_steps: list[dict[str, Any]]
    suggested_questions: list[str]
    tool_trace: list[dict[str, Any]]
    answer: str
    error: Optional[str]


def _make_node_id(label: str, name: str, extra: str = "") -> str:
    parts = [label, name]
    if extra:
        parts.append(extra)
    return "::".join(parts)


def _course_key(course: dict) -> str:
    return course.get("id") or _make_node_id(LABEL_COURSE, course.get("name", ""))


def _node(label: str, props: dict) -> dict:
    return {
        "id": _make_node_id(label, props.get("name", ""), props.get("university", "")),
        "name": props.get("name", ""),
        "label": label,
        "properties": props,
    }


def _llm() -> Any:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY is not configured")
    if ChatOpenAI is None:
        raise RuntimeError("langchain-openai is not installed")
    return ChatOpenAI(
        model=DEEPSEEK_MODEL,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=0.1,
    )


def _clean_json(raw: str) -> str:
    return raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()


def _split_terms(text: str) -> list[str]:
    terms = re.split(r"[\s，。！？、,.!?;；:：/和与及]+", text or "")
    return [term.strip() for term in terms if len(term.strip()) >= 2]


def _number_or_none(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fallback_parse(message: str) -> dict[str, Any]:
    known: list[str] = []
    for pattern in (
        r"(?:我)?(?:已经|已)?(?:(?<!不)会|学过|掌握)(.+?)(?:，|。|；|;|,|\.|想|希望|目标|接下来|然后|请|帮|怎么|如何|推荐|应该|需要|学习路径|路线|$)",
        r"(?:我)?有(.+?)基础",
    ):
        for match in re.finditer(pattern, message):
            known.extend(_split_terms(match.group(1)))

    target = ""
    match = re.search(r"(?:想学|学习|目标是|目标为|学会|掌握)(.+?)(?:，|。|；|;|,|\.|每周|$)", message)
    if match:
        target = _split_terms(match.group(1))[0] if _split_terms(match.group(1)) else ""

    hours = None
    weekly = re.search(r"每周\s*(\d+(?:\.\d+)?)\s*(?:小时|h)", message, re.I)
    daily = re.search(r"每天\s*(\d+(?:\.\d+)?)\s*(?:小时|h)", message, re.I)
    if weekly:
        hours = float(weekly.group(1))
    elif daily:
        hours = float(daily.group(1)) * 7

    level = None
    if any(token in message for token in ("零基础", "初学", "入门")):
        level = "beginner"
    elif any(token in message for token in ("进阶", "高级", "研究")):
        level = "advanced"
    elif any(token in message for token in ("有基础", "中级")):
        level = "intermediate"

    return {
        "known": known,
        "target": target,
        "goal": None,
        "weekly_hours": hours,
        "level": level,
    }


def _parse_with_llm(message: str) -> dict[str, Any]:
    prompt = """你是学习路径规划 Agent 的画像解析器。只输出 JSON，不要 Markdown。
字段：
{
  "known": ["用户已经会/学过/掌握的知识点或课程"],
  "target": "用户想学习或达成的主要目标",
  "goal": "入门/补基础/考试/科研/就业/项目等，没有则 null",
  "weekly_hours": 每周学习小时数，没有则 null,
  "level": "beginner" | "intermediate" | "advanced" | null
}
不要把目标放进 known。"""
    try:
        result = _llm().invoke([("system", prompt), ("human", message)])
        parsed = json.loads(_clean_json(result.content or ""))
    except Exception:
        return {}
    if not isinstance(parsed, dict):
        return {}
    known = parsed.get("known") or []
    return {
        "known": [str(item).strip() for item in known if str(item).strip()],
        "target": str(parsed.get("target") or "").strip(),
        "goal": parsed.get("goal"),
        "weekly_hours": _number_or_none(parsed.get("weekly_hours")),
        "level": parsed.get("level"),
    }


def _merge_unique(*groups: list[str]) -> list[str]:
    seen = set()
    merged = []
    for group in groups:
        for item in group or []:
            clean = str(item).strip()
            if clean and clean not in seen:
                seen.add(clean)
                merged.append(clean)
    return merged


def _is_vague_target(value: Any) -> bool:
    text = re.sub(r"[\s？?。！，!,；;呢吧啊呀]+", "", str(value or ""))
    if not text:
        return True
    if text in {"下一步", "接下来", "然后", "路径", "路线", "学习路径", "推荐", "规划"}:
        return True
    return len(text) <= 8 and any(token in text for token in ("下一步", "接下来", "然后"))


def _fetch_entities(session) -> list[dict[str, Any]]:
    result = session.run(
        f"""
        MATCH (n)
        WHERE n:{LABEL_MAJOR} OR n:{LABEL_COURSE} OR n:{LABEL_KNOWLEDGE_POINT}
        RETURN labels(n)[0] AS label, properties(n) AS props
        """
    )
    return [
        {"label": row["label"], "name": dict(row["props"]).get("name", ""), "properties": dict(row["props"])}
        for row in result
        if dict(row["props"]).get("name")
    ]


def _score_match(query: str, name: str) -> float:
    if not query or not name:
        return 0.0
    if query == name:
        return 1.0
    if query in name or name in query:
        return min(0.94, 0.72 + min(len(query), len(name)) / max(len(query), len(name)) * 0.2)
    return SequenceMatcher(None, query.lower(), name.lower()).ratio()


def search_entities(query: str, limit: int = 8) -> list[dict[str, Any]]:
    """Find matching majors, courses, and knowledge concepts."""
    if not query:
        return []
    with get_neo4j_driver().session() as session:
        entities = _fetch_entities(session)
    scored = [
        (_score_match(query, item["name"]), item)
        for item in entities
        if _score_match(query, item["name"]) >= 0.54
    ]
    if not scored:
        names = [item["name"] for item in entities]
        close = get_close_matches(query, names, n=limit, cutoff=0.5)
        scored = [(0.55, item) for item in entities if item["name"] in close]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [{**item, "score": round(score, 3)} for score, item in scored[:limit]]


def get_knowledge_relations(kp_name: str) -> dict[str, Any]:
    with get_neo4j_driver().session() as session:
        target = session.run(
            f"MATCH (k:{LABEL_KNOWLEDGE_POINT} {{name: $name}}) RETURN properties(k) AS props",
            name=kp_name,
        ).single()
        if not target:
            return {"target": None, "prerequisites": [], "dependents": [], "links": []}
        prereq_rows = session.run(
            f"""
            MATCH (pre:{LABEL_KNOWLEDGE_POINT})-[:{REL_PREREQUISITE_OF}]->(target:{LABEL_KNOWLEDGE_POINT} {{name: $name}})
            RETURN DISTINCT properties(pre) AS props
            ORDER BY props.name
            """,
            name=kp_name,
        )
        dep_rows = session.run(
            f"""
            MATCH (target:{LABEL_KNOWLEDGE_POINT} {{name: $name}})-[:{REL_PREREQUISITE_OF}]->(dep:{LABEL_KNOWLEDGE_POINT})
            RETURN DISTINCT properties(dep) AS props
            ORDER BY props.name
            """,
            name=kp_name,
        )
        target_node = _node(LABEL_KNOWLEDGE_POINT, dict(target["props"]))
        prerequisites = [_node(LABEL_KNOWLEDGE_POINT, dict(row["props"])) for row in prereq_rows]
        dependents = [_node(LABEL_KNOWLEDGE_POINT, dict(row["props"])) for row in dep_rows]
    return {
        "target": target_node,
        "prerequisites": prerequisites,
        "dependents": dependents,
        "links": [
            {"source": item["id"], "target": target_node["id"], "type": REL_PREREQUISITE_OF}
            for item in prerequisites
        ] + [
            {"source": target_node["id"], "target": item["id"], "type": REL_PREREQUISITE_OF}
            for item in dependents
        ],
    }


def get_course_knowledge(course_name: str) -> dict[str, Any]:
    with get_neo4j_driver().session() as session:
        row = session.run(
            f"""
            MATCH (c:{LABEL_COURSE} {{name: $name}})
            OPTIONAL MATCH (c)-[:{REL_COVERS}]->(k:{LABEL_KNOWLEDGE_POINT})
            RETURN properties(c) AS course, collect(DISTINCT properties(k)) AS knowledge
            """,
            name=course_name,
        ).single()
        if not row or not row["course"]:
            return {"course": None, "knowledge_points": [], "prerequisites": []}
        knowledge = [dict(item) for item in row["knowledge"] if item]
        names = [item.get("name", "") for item in knowledge if item.get("name")]
        prereq_rows = session.run(
            f"""
            MATCH (pre:{LABEL_KNOWLEDGE_POINT})-[:{REL_PREREQUISITE_OF}]->(k:{LABEL_KNOWLEDGE_POINT})
            WHERE k.name IN $names AND NOT pre.name IN $names
            RETURN DISTINCT properties(pre) AS props
            ORDER BY props.name
            LIMIT 24
            """,
            names=names,
        )
        prereqs = [dict(row["props"]) for row in prereq_rows]
    return {
        "course": _node(LABEL_COURSE, dict(row["course"])),
        "knowledge_points": [_node(LABEL_KNOWLEDGE_POINT, item) for item in knowledge],
        "prerequisites": [_node(LABEL_KNOWLEDGE_POINT, item) for item in prereqs],
    }


def get_major_roadmap(major_name: str) -> dict[str, Any]:
    with get_neo4j_driver().session() as session:
        rows = list(session.run(
            f"""
            MATCH (m:{LABEL_MAJOR} {{name: $major_name}})-[:{REL_BELONGS_TO}]->(c:{LABEL_COURSE})
            OPTIONAL MATCH (c)-[:{REL_COVERS}]->(k:{LABEL_KNOWLEDGE_POINT})
            RETURN properties(m) AS major, properties(c) AS course, collect(DISTINCT properties(k)) AS knowledge
            ORDER BY course.name
            """,
            major_name=major_name,
        ))
        if not rows:
            return {"major": None, "modules": [], "courses": [], "links": []}
        raw_courses = []
        for row in rows:
            course = dict(row["course"])
            knowledge = [dict(item) for item in row["knowledge"] if item]
            raw_courses.append({
                "id": _course_key(course),
                "name": course.get("name", ""),
                "properties": course,
                "knowledge_points": [{"name": item.get("name", "")} for item in knowledge if item.get("name")],
                "knowledge_count": len(knowledge),
            })
        prereq_rows = list(session.run(
            f"""
            MATCH (a:{LABEL_COURSE})-[r:{REL_PREREQUISITE_FOR}]->(b:{LABEL_COURSE})
            WHERE (a.id IN $course_ids OR a.name IN $course_names)
              AND (b.id IN $course_ids OR b.name IN $course_names)
            RETURN properties(a) AS source, properties(b) AS target, type(r) AS rel_type
            """,
            course_ids=[item["properties"].get("id") for item in raw_courses],
            course_names=[item["name"] for item in raw_courses],
        ))
    course_id_set = {course["id"] for course in raw_courses}
    links = []
    for row in prereq_rows:
        source = _course_key(dict(row["source"]))
        target = _course_key(dict(row["target"]))
        if source in course_id_set and target in course_id_set:
            links.append({"source": source, "target": target, "type": row["rel_type"]})
    roadmap = build_module_roadmap(major_name, raw_courses, links)
    return {
        "major": _node(LABEL_MAJOR, dict(rows[0]["major"])),
        "modules": roadmap.get("modules", []),
        "module_links": roadmap.get("module_links", []),
        "courses": roadmap.get("courses", []),
        "links": roadmap.get("links", []),
    }


def recommend_learning_path(profile: dict[str, Any]) -> dict[str, Any]:
    try:
        from src.rl.recommend import recommend_path

        result = recommend_path(
            known_concepts=profile.get("known", []),
            target_concept=profile.get("target"),
            major_name=profile.get("major"),
            weekly_hours=profile.get("weekly_hours"),
            level=profile.get("level") or "intermediate",
            goal=profile.get("goal"),
            include_predicted=False,
        )
        path = [item.get("concept", "") for item in result.path if item.get("concept")]
        return {
            "method": "recommend_path",
            "path": path,
            "blocked_recovery": result.blocked_recovery,
            "explanation": result.explanation,
        }
    except Exception as exc:
        return {"method": "recommend_path", "path": [], "error": str(exc)}


def _parse_profile_node(state: AgentState) -> AgentState:
    session = AGENT_SESSIONS.get(state["session_id"], {})
    previous = session.get("profile", {})
    parsed = _parse_with_llm(state["message"]) or _fallback_parse(state["message"])
    explicit_target = state.get("target")
    parsed_target = parsed.get("target")
    if _is_vague_target(explicit_target):
        explicit_target = None
    if _is_vague_target(parsed_target):
        parsed_target = None
    state["target"] = explicit_target or parsed_target or previous.get("target")
    state["known"] = _merge_unique(previous.get("known", []), parsed.get("known", []), state.get("known", []))
    # 防止解析器把目标也当成已掌握
    if state["target"]:
        state["known"] = [k for k in state["known"] if k != state["target"]]
    state["goal"] = state.get("goal") or parsed.get("goal") or previous.get("goal")
    state["weekly_hours"] = (
        state.get("weekly_hours")
        if state.get("weekly_hours") is not None
        else parsed.get("weekly_hours") if parsed.get("weekly_hours") is not None else previous.get("weekly_hours")
    )
    state["level"] = state.get("level") or parsed.get("level") or previous.get("level") or "intermediate"
    state.setdefault("tool_trace", []).append({"tool": "parse_profile", "ok": True})
    return state


def _match_entities_node(state: AgentState) -> AgentState:
    query_parts = [state.get("target") or "", state["message"]]
    matches: list[dict[str, Any]] = []
    for query in query_parts:
        for item in search_entities(query, limit=10):
            if not any(existing["label"] == item["label"] and existing["name"] == item["name"] for existing in matches):
                matches.append(item)

    known_matches = []
    for term in state.get("known", []):
        known_matches.extend(search_entities(term, limit=3))

    target_type_order = [LABEL_KNOWLEDGE_POINT, LABEL_COURSE, LABEL_MAJOR]
    if "专业" in state["message"]:
        target_type_order = [LABEL_MAJOR, LABEL_COURSE, LABEL_KNOWLEDGE_POINT]
    target_entity = None
    target_text = state.get("target") or ""
    for label in target_type_order:
        candidates = [item for item in matches if item["label"] == label]
        exact = [item for item in candidates if target_text and item["name"] == target_text]
        target_entity = (exact or candidates or [None])[0]
        if target_entity:
            break

    state["matched_entities"] = matches[:12]
    state["target_entity"] = target_entity
    state["known"] = _merge_unique(
        state.get("known", []),
        [item["name"] for item in known_matches if item.get("label") == LABEL_KNOWLEDGE_POINT],
    )
    state.setdefault("tool_trace", []).append({
        "tool": "search_entities",
        "ok": True,
        "matches": len(matches),
        "target": target_entity["name"] if target_entity else None,
    })
    return state


def _rows_to_nodes_and_links(names: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not names:
        return [], []
    with get_neo4j_driver().session() as session:
        node_rows = session.run(
            f"""
            MATCH (k:{LABEL_KNOWLEDGE_POINT})
            WHERE k.name IN $names
            RETURN DISTINCT properties(k) AS props
            """,
            names=names,
        )
        nodes = [_node(LABEL_KNOWLEDGE_POINT, dict(row["props"])) for row in node_rows]
        edge_rows = session.run(
            f"""
            MATCH (a:{LABEL_KNOWLEDGE_POINT})-[:{REL_PREREQUISITE_OF}]->(b:{LABEL_KNOWLEDGE_POINT})
            WHERE a.name IN $names AND b.name IN $names
            RETURN DISTINCT a.name AS source, b.name AS target
            LIMIT 180
            """,
            names=names,
        )
        links = [
            {
                "source": _make_node_id(LABEL_KNOWLEDGE_POINT, row["source"]),
                "target": _make_node_id(LABEL_KNOWLEDGE_POINT, row["target"]),
                "type": REL_PREREQUISITE_OF,
            }
            for row in edge_rows
        ]
    return nodes, links


def _retrieve_graph_node(state: AgentState) -> AgentState:
    target = state.get("target_entity")
    plan_steps: list[dict[str, Any]] = []
    evidence_nodes: list[dict[str, Any]] = []
    evidence_links: list[dict[str, Any]] = []
    trace = state.setdefault("tool_trace", [])

    profile = {
        "known": state.get("known", []),
        "target": target["name"] if target else state.get("target"),
        "goal": state.get("goal"),
        "weekly_hours": state.get("weekly_hours"),
        "level": state.get("level"),
    }
    rl_result = recommend_learning_path(profile)
    trace.append({"tool": "recommend_learning_path", "ok": not rl_result.get("error"), "path_count": len(rl_result.get("path", []))})

    if not target:
        state["plan_steps"] = []
        state["evidence_nodes"] = []
        state["evidence_links"] = []
        return state

    if target["label"] == LABEL_KNOWLEDGE_POINT:
        with get_neo4j_driver().session() as session:
            path = find_path_to_target(session, state.get("known", []), target["name"])
            # Also query direct predicted prerequisites as supplementary references.
            predicted_rows = session.run(
                f"""
                MATCH (pre:{LABEL_KNOWLEDGE_POINT})-[r:{REL_PREDICTED_PREREQ}]->(target:{LABEL_KNOWLEDGE_POINT} {{name: $name}})
                RETURN pre.name AS name, r.confidence AS confidence
                ORDER BY r.confidence DESC
                LIMIT 12
                """,
                name=target["name"],
            )
            predicted_prereqs = [(r["name"], r["confidence"]) for r in predicted_rows if r["name"]]

        manual_names = path.get("need_to_learn", []) or path.get("path", [])
        rl_names = rl_result.get("path", [])

        # Prefer the real prerequisite path; only use predicted/RL results as supplementary.
        # A longer predicted path is not necessarily a better path.
        plan_steps: list[dict[str, Any]] = []
        used_names = set()
        if manual_names:
            for index, name in enumerate(manual_names[:24], 1):
                plan_steps.append({"order": index, "name": name, "type": "knowledge", "reason": "图谱前置路径"})
                used_names.add(name)

        # Add direct predicted prerequisites as "reference" steps (controlled, not BFS-wandering).
        for name, conf in predicted_prereqs:
            if name in used_names or name in state.get("known", []):
                continue
            plan_steps.append({
                "order": len(plan_steps) + 1,
                "name": name,
                "type": "knowledge",
                "reason": "学习过程中可能遇到的相关概念（预测参考）",
            })
            used_names.add(name)

        # Append any remaining RL suggestions that are not already covered.
        rl_extra = [n for n in rl_names if n not in used_names and n != target["name"]][:12]
        if rl_extra:
            for name in rl_extra:
                plan_steps.append({
                    "order": len(plan_steps) + 1,
                    "name": name,
                    "type": "knowledge",
                    "reason": "学习过程中可能遇到的相关概念（预测参考）",
                })
                used_names.add(name)

        if not plan_steps:
            plan_steps.append({"order": 1, "name": target["name"], "type": "knowledge", "reason": "目标知识点"})

        evidence_names = _merge_unique(
            state.get("known", []),
            [n["name"] for n in plan_steps],
            [target["name"]],
        )
        evidence_nodes, evidence_links = _rows_to_nodes_and_links(evidence_names)
        trace.append({"tool": "find_path_to_target", "ok": "error" not in path, "path_count": len(plan_steps)})
    elif target["label"] == LABEL_COURSE:
        data = get_course_knowledge(target["name"])
        nodes = [node for node in [data.get("course")] if node] + data.get("prerequisites", []) + data.get("knowledge_points", [])
        evidence_nodes = nodes[:80]
        for pre in data.get("prerequisites", [])[:10]:
            plan_steps.append({"order": len(plan_steps) + 1, "name": pre["name"], "type": "knowledge", "reason": "课程外部前置知识"})
        for kp in data.get("knowledge_points", [])[:18]:
            plan_steps.append({"order": len(plan_steps) + 1, "name": kp["name"], "type": "knowledge", "reason": f"{target['name']} 覆盖知识点"})
        trace.append({"tool": "get_course_knowledge", "ok": True, "knowledge_count": len(data.get("knowledge_points", []))})
    else:
        data = get_major_roadmap(target["name"])
        evidence_nodes = [data["major"]] if data.get("major") else []
        for index, module in enumerate(data.get("modules", [])[:12], 1):
            plan_steps.append({
                "order": index,
                "name": module.get("name", ""),
                "type": "module",
                "reason": f"{len(module.get('courses', []))} 门课程",
            })
        trace.append({"tool": "get_major_roadmap", "ok": bool(data.get("major")), "module_count": len(data.get("modules", []))})

    state["plan_steps"] = plan_steps
    state["evidence_nodes"] = evidence_nodes
    state["evidence_links"] = evidence_links
    return state


def _fallback_answer(state: AgentState) -> str:
    target = state.get("target_entity") or {}
    target_name = target.get("name") or state.get("target") or "你的目标"
    known = state.get("known", [])
    steps = state.get("plan_steps", [])
    lines = [f"我会把目标定为「{target_name}」。"]
    if known:
        lines.append(f"已掌握部分按「{'、'.join(known[:8])}」处理。")
    if state.get("weekly_hours"):
        lines.append(f"按每周约 {state['weekly_hours']:g} 小时，建议先做小阶段推进。")
    if steps:
        path = " → ".join(step["name"] for step in steps[:10])
        lines.append(f"推荐路径：{path}。")
        lines.append("下一步先完成第 1-2 个未掌握节点，再回到 Agent 更新已掌握内容。")
    else:
        lines.append("我还没有在图谱里匹配到足够明确的目标，请补充想学的知识点、课程或专业。")
    return "\n".join(lines)


def _compose_answer_node(state: AgentState) -> AgentState:
    if not DEEPSEEK_API_KEY:
        state["error"] = "DEEPSEEK_API_KEY is not configured"
        state["answer"] = "DeepSeek API Key 未配置，Agent 无法生成个性化回复；请检查后端 .env。"
        return state
    prompt = """你是一位有亲和力的学习路径规划老师，正在给学生做一对一学习建议。请用自然、温暖、鼓励性的中文回复，不要像机器人或系统提示那样生硬。

要求：
1. 依据给定的 profile、plan_steps 和 evidence 给出建议，不要编造图谱没有支持的前置关系。
2. 推荐路径只能使用 plan_steps 里的 name；不要自行补充新步骤。
3. 不要假设用户已经掌握了 profile.known 里没列出的知识。如果用户没有声明已知基础，就把他当成初学者，不要写"以你已有的 XX 基础"这类话。
4. 把 plan_steps 中的节点分成两类处理：
   - reason 为 "图谱前置路径" 的节点：这是**核心学习路径**，按 order 顺序写成 "A → B → C"，并解释每一步为什么要学。
   - reason 包含 "预测参考" 或 "参考前置" 的节点：这是**学习过程中可能会遇到的相关概念**，单独放在一段里介绍，不要编进主路径的序号，不要让学生觉得必须按顺序提前学完。
5. 如果真实前置路径很短或只有目标本身，不要写"真实前置路径为空""数据不足""图谱未找到"之类的话。老师的说法应该是：
   - "以你目前的 XX 基础，可以直接进入 YY 的学习。"
   - "YY 本身已经是一个比较独立的目标，你可以先从它开始。"
   - "在学 YY 的过程中，你可能会用到 ZZ，遇到的时候再补也来得及。"
6. 结合 weekly_hours 给出大致的阶段划分（例如"按每周 X 小时，大约 Y 周可以过完第一步"），让老师的感觉更自然。
7. 不要把参考概念排成 "A → B → C" 的顺序，也不要给它们分配 stage/周数。参考概念只在遇到时按需补充即可。
8. 结尾用一句简短的鼓励或下一步行动建议。

回复结构参考（不必严格照搬，保持自然）：
- 先肯定学生已有的基础（仅限 profile.known 里明确列出的）；
- 给出核心推荐学习顺序（用 → 连接，只包含 "图谱前置路径" 的节点）；
- 说明每一步学什么、为什么；
- 用一小段列出可能遇到的参考概念；
- 给出时间拆分和下一步行动；
- 一句鼓励。"""
    context = {
        "user_message": state["message"],
        "profile": {
            "known": state.get("known", []),
            "target": state.get("target_entity") or state.get("target"),
            "goal": state.get("goal"),
            "weekly_hours": state.get("weekly_hours"),
            "level": state.get("level"),
        },
        "plan_steps": state.get("plan_steps", [])[:18],
        "evidence_nodes": [
            {"label": item.get("label"), "name": item.get("name")}
            for item in state.get("evidence_nodes", [])[:30]
        ],
        "tool_trace": state.get("tool_trace", []),
    }
    try:
        result = _llm().invoke([("system", prompt), ("human", json.dumps(context, ensure_ascii=False))])
        state["answer"] = result.content or _fallback_answer(state)
        state.setdefault("tool_trace", []).append({"tool": "deepseek_compose", "ok": True})
    except Exception as exc:
        state["answer"] = _fallback_answer(state)
        state.setdefault("tool_trace", []).append({"tool": "deepseek_compose", "ok": False, "error": str(exc)})
    target_name = (state.get("target_entity") or {}).get("name") or state.get("target") or "目标"
    hours = state.get("weekly_hours") or 6
    state["suggested_questions"] = [
        f"按每周 {hours:g} 小时拆成 4 周计划",
        f"我学到「{target_name}」之后下一步是什么？",
        "哪些已掌握内容可以跳过？",
    ]
    return state


def _get_graph():
    global _COMPILED_GRAPH
    if StateGraph is None:
        raise RuntimeError("langgraph is not installed")
    if _COMPILED_GRAPH is None:
        graph = StateGraph(AgentState)
        graph.add_node("parse_profile", _parse_profile_node)
        graph.add_node("match_entities", _match_entities_node)
        graph.add_node("retrieve_graph", _retrieve_graph_node)
        graph.add_node("compose_answer", _compose_answer_node)
        graph.set_entry_point("parse_profile")
        graph.add_edge("parse_profile", "match_entities")
        graph.add_edge("match_entities", "retrieve_graph")
        graph.add_edge("retrieve_graph", "compose_answer")
        graph.add_edge("compose_answer", END)
        _COMPILED_GRAPH = graph.compile()
    return _COMPILED_GRAPH


def _save_session(state: AgentState) -> None:
    session_id = state["session_id"]
    session = AGENT_SESSIONS.setdefault(session_id, {"messages": [], "profile": {}})
    session["messages"].append({"role": "user", "content": state["message"]})
    session["messages"].append({"role": "assistant", "content": state.get("answer", "")})
    session["messages"] = session["messages"][-MAX_SESSION_MESSAGES:]
    session["profile"] = {
        "known": state.get("known", []),
        "target": (state.get("target_entity") or {}).get("name") or state.get("target"),
        "goal": state.get("goal"),
        "weekly_hours": state.get("weekly_hours"),
        "level": state.get("level"),
    }


def run_learning_path_agent(
    *,
    message: str,
    session_id: Optional[str] = None,
    known: Optional[list[str]] = None,
    target: Optional[str] = None,
    goal: Optional[str] = None,
    weekly_hours: Optional[float] = None,
    level: Optional[str] = None,
) -> dict[str, Any]:
    """Run one Agent turn and return a JSON-serializable response."""
    if not message.strip():
        raise ValueError("message is required")
    session_id = session_id or uuid.uuid4().hex
    session = AGENT_SESSIONS.get(session_id, {})
    initial: AgentState = {
        "session_id": session_id,
        "message": message.strip(),
        "messages": session.get("messages", []),
        "known": known or [],
        "target": target,
        "goal": goal,
        "weekly_hours": weekly_hours,
        "level": level,
        "tool_trace": [],
    }
    state = _get_graph().invoke(initial)
    _save_session(state)
    return {
        "session_id": session_id,
        "answer": state.get("answer", ""),
        "profile": AGENT_SESSIONS[session_id]["profile"],
        "plan_steps": state.get("plan_steps", []),
        "evidence_nodes": state.get("evidence_nodes", []),
        "evidence_links": state.get("evidence_links", []),
        "suggested_questions": state.get("suggested_questions", []),
        "tool_trace": state.get("tool_trace", []),
        "error": state.get("error"),
    }
