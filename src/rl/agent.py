"""Simple RL agent for recommending learning paths over a dual graph."""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.rl.dual_graph import DualGraph


@dataclass
class UserState:
    known_concepts: set[str]
    target_concept: Optional[str]
    weekly_hours: float
    level: str
    goal: Optional[str]


@dataclass
class Action:
    concept: str
    action_type: str  # "prereq" | "target" | "bridge"
    reason: str
    expected_difficulty: float
    confidence: float


class SimpleRLAgent:
    def __init__(self, dual_graph: DualGraph):
        self.graph = dual_graph

    def _difficulty_score(self, concept: str, known: set[str]) -> float:
        """Combine missing-prereq ratio (0.6) and normalized embedding distance
        to nearest known concept (0.4).
        """
        prereqs = self.graph.get_prereqs(concept)
        if not prereqs:
            missing_ratio = 0.0
        else:
            missing = sum(1 for pr, _ in prereqs if pr not in known)
            missing_ratio = missing / len(prereqs)

        emb_dist = 1.0
        if concept in self.graph.embeddings and known:
            emb_c = np.array(self.graph.embeddings[concept])
            min_dist = float("inf")
            for k in known:
                if k in self.graph.embeddings:
                    emb_k = np.array(self.graph.embeddings[k])
                    d = np.linalg.norm(emb_c - emb_k)
                    if d < min_dist:
                        min_dist = d
            if min_dist != float("inf"):
                # Normalize by average embedding norm for stability
                avg_norm = np.mean([
                    np.linalg.norm(np.array(self.graph.embeddings[n]))
                    for n in known if n in self.graph.embeddings
                ]) or 1.0
                emb_dist = min(min_dist / (avg_norm + 1e-9), 1.0)

        return 0.6 * missing_ratio + 0.4 * emb_dist

    def _select_by_difficulty(self, candidates: list[str], state: UserState) -> list[tuple[str, float]]:
        """Rank candidates by difficulty score (ascending)."""
        scored = [(c, self._difficulty_score(c, state.known_concepts)) for c in candidates]
        scored.sort(key=lambda x: x[1])
        return scored

    def act(self, state: UserState) -> list[Action]:
        """Generate actions for the user state.

        If target is set, find prerequisite path via BFS backwards.
        Otherwise find learnable concepts.
        Then find blocked recovery bridges.
        Sort by confidence desc.
        """
        actions: list[Action] = []

        if state.target_concept and state.target_concept in self.graph.nodes:
            path = self._find_prereq_path(state.target_concept, state.known_concepts)
            for i, concept in enumerate(path):
                diff = round(self._difficulty_score(concept, state.known_concepts), 2)
                conf = max(0.5, 1.0 - i * 0.05)
                actions.append(Action(
                    concept=concept,
                    action_type="prereq" if concept != state.target_concept else "target",
                    reason=f"学习{state.target_concept}的前置知识" if concept != state.target_concept else f"目标知识：{state.target_concept}",
                    expected_difficulty=diff,
                    confidence=round(conf, 2),
                ))

            # Find bridge concepts for missing prerequisites
            blocked = self._find_blocked(state)
            for node, missing, bridge, sim in blocked:
                actions.append(Action(
                    concept=bridge,
                    action_type="bridge",
                    reason=f"通过已掌握的「{bridge}」辅助理解缺失前置「{missing}」（相似度 {sim:.2f}），以解锁「{node}」",
                    expected_difficulty=round(self._difficulty_score(bridge, state.known_concepts), 2),
                    confidence=round(sim, 2),
                ))
        else:
            learnable = self._find_learnable(state.known_concepts)
            ranked = self._select_by_difficulty(learnable, state)
            for i, (concept, diff) in enumerate(ranked):
                conf = max(0.5, 1.0 - i * 0.05)
                actions.append(Action(
                    concept=concept,
                    action_type="prereq",
                    reason=f"基于当前已掌握知识，推荐学习「{concept}」",
                    expected_difficulty=round(diff, 2),
                    confidence=round(conf, 2),
                ))

        actions.sort(key=lambda a: a.confidence, reverse=True)
        return actions

    def _find_prereq_path(self, target: str, known: set[str]) -> list[str]:
        """BFS backwards from target to find shortest path of missing prerequisites."""
        if target in known:
            return []

        from collections import deque

        queue = deque([(target, [target])])
        visited = {target}

        while queue:
            current, path = queue.popleft()
            prereqs = self.graph.get_prereqs(current)
            missing = [(pr, w) for pr, w in prereqs if pr not in known]

            if not missing:
                # All prerequisites met; return path in learning order (prereqs first)
                return list(reversed(path))

            # Sort by weight desc and pick the highest-weight missing prereq to explore first
            missing.sort(key=lambda x: x[1], reverse=True)
            for pr, _ in missing:
                if pr not in visited:
                    visited.add(pr)
                    queue.append((pr, path + [pr]))

        # No complete path found; return reversed path of best effort
        return list(reversed(path))

    def _find_learnable(self, known: set[str]) -> list[str]:
        """Find concepts whose prerequisites are all satisfied."""
        learnable = []
        for node in self.graph.nodes:
            if node in known:
                continue
            prereqs = self.graph.get_prereqs(node)
            if all(pr in known for pr, _ in prereqs):
                learnable.append(node)
        return learnable

    def _find_blocked(self, state: UserState) -> list[tuple[str, str, str, float]]:
        """Find blocked concepts and their recovery bridges.

        Returns list of (node, missing_prereq, bridge_concept, sim).
        """
        if not state.target_concept:
            return []

        # Find concepts on the path to target that are blocked
        path = self._find_prereq_path(state.target_concept, state.known_concepts)
        blocked_nodes = [n for n in path if self.graph.is_blocked(n, state.known_concepts)]

        results = []
        for node in blocked_nodes:
            bridges = self.graph.find_bridge_concepts(
                node, state.known_concepts, threshold=0.5
            )
            for bridge, missing, sim in bridges:
                results.append((node, missing, bridge, sim))

        # Sort by similarity desc
        results.sort(key=lambda x: x[3], reverse=True)
        return results
