"""Recommendation entry point: build dual graph, run RL agent, format result."""
from dataclasses import dataclass
from typing import Optional

from src.rl.dual_graph import build_dual_graph
from src.rl.agent import SimpleRLAgent, UserState


@dataclass
class RecommendationResult:
    target: Optional[str]
    path: list[dict]
    blocked_recovery: list[dict]
    difficulty_analysis: dict
    explanation: str


def recommend_path(
    known_concepts: list[str],
    target_concept: Optional[str] = None,
    major_name: Optional[str] = None,
    weekly_hours: Optional[float] = None,
    level: str = "intermediate",
    goal: Optional[str] = None,
    include_predicted: bool = False,
    predicted_min_confidence: float = 0.9,
) -> RecommendationResult:
    """Build dual graph, run agent, and format recommendation result.

    Returns a RecommendationResult even if Neo4j is empty or embeddings
    are unavailable, so callers never crash.
    """
    try:
        dual_graph = build_dual_graph(
            major_name=major_name,
            include_predicted=include_predicted,
            predicted_min_confidence=predicted_min_confidence,
            sim_threshold=0.5,
            max_sim_edges=2000,
        )
    except Exception:
        dual_graph = build_dual_graph()

    agent = SimpleRLAgent(dual_graph)
    state = UserState(
        known_concepts=set(known_concepts or []),
        target_concept=target_concept,
        weekly_hours=weekly_hours or 0.0,
        level=level,
        goal=goal,
    )

    try:
        actions = agent.act(state)
    except Exception:
        actions = []

    path = []
    blocked_recovery = []
    difficulty_analysis = {"overall_difficulty": 0.0, "prereq_count": 0, "bridge_count": 0}

    prereq_actions = [a for a in actions if a.action_type == "prereq"]
    target_actions = [a for a in actions if a.action_type == "target"]
    bridge_actions = [a for a in actions if a.action_type == "bridge"]

    for a in prereq_actions:
        path.append({
            "concept": a.concept,
            "type": a.action_type,
            "reason": a.reason,
            "expected_difficulty": a.expected_difficulty,
            "confidence": a.confidence,
        })

    for a in target_actions:
        path.append({
            "concept": a.concept,
            "type": a.action_type,
            "reason": a.reason,
            "expected_difficulty": a.expected_difficulty,
            "confidence": a.confidence,
        })

    for a in bridge_actions:
        blocked_recovery.append({
            "concept": a.concept,
            "type": a.action_type,
            "reason": a.reason,
            "expected_difficulty": a.expected_difficulty,
            "confidence": a.confidence,
        })

    if actions:
        avg_diff = sum(a.expected_difficulty for a in actions) / len(actions)
        difficulty_analysis = {
            "overall_difficulty": round(avg_diff, 2),
            "prereq_count": len(prereq_actions),
            "bridge_count": len(bridge_actions),
            "target_count": len(target_actions),
        }

    # Build Chinese explanation
    parts = []
    if target_concept:
        parts.append(f"目标知识：「{target_concept}」")
    if known_concepts:
        parts.append(f"已掌握知识：{'、'.join(known_concepts[:10])}")
    if path:
        parts.append(f"推荐学习路径包含 {len(path)} 个知识点")
    if blocked_recovery:
        parts.append(f"发现 {len(blocked_recovery)} 个可通过相似知识辅助理解的桥接概念")
    if not path and not blocked_recovery:
        if target_concept and target_concept in state.known_concepts:
            parts.append("目标知识已在已掌握列表中，无需额外学习。")
        else:
            parts.append("当前图谱数据不足，无法生成推荐路径。")

    if weekly_hours:
        parts.append(f"按每周约 {weekly_hours:g} 小时估算，建议分阶段推进")
    if goal:
        parts.append(f"学习目标：{goal}")

    if level == "beginner":
        parts.append("建议零基础学习者按顺序逐步完成前置知识，不要跳过基础阶段")
    elif level == "advanced":
        parts.append("已有进阶基础，可优先核对未掌握清单，再进入目标知识")

    explanation = "；".join(parts) + "。" if parts else "暂无推荐内容。"

    return RecommendationResult(
        target=target_concept,
        path=path,
        blocked_recovery=blocked_recovery,
        difficulty_analysis=difficulty_analysis,
        explanation=explanation,
    )
