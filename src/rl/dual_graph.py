"""Dual graph builder: prerequisite graph + semantic similarity graph."""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.config import get_neo4j_driver
from src.models.schema import (
    LABEL_KNOWLEDGE_POINT,
    LABEL_MAJOR,
    REL_BELONGS_TO,
    REL_COVERS,
    REL_PREREQUISITE_OF,
    REL_PREDICTED_PREREQ,
    REL_SEMANTIC_SIMILARITY,
)
from src.prereq_prediction.embedder import KnowledgeEmbedder


@dataclass
class DualGraph:
    nodes: list[str]
    prereq_edges: list[tuple[str, str, float]]  # (src, dst, weight)
    sim_edges: list[tuple[str, str, float]]      # (a, b, sim)
    embeddings: dict[str, list[float]]

    def get_prereqs(self, node: str) -> list[tuple[str, float]]:
        """Return list of (prerequisite_node, weight) for given node."""
        return [(src, w) for src, dst, w in self.prereq_edges if dst == node]

    def get_similar(self, node: str, threshold: float = 0.5) -> list[tuple[str, float]]:
        """Return list of (similar_node, similarity) for given node above threshold."""
        result = []
        for a, b, sim in self.sim_edges:
            if a == node and sim >= threshold:
                result.append((b, sim))
            elif b == node and sim >= threshold:
                result.append((a, sim))
        return result

    def is_blocked(self, node: str, known: set[str]) -> bool:
        """Check if node has unmet prerequisites."""
        prereqs = self.get_prereqs(node)
        if not prereqs:
            return False
        return any(pr not in known for pr, _ in prereqs)

    def find_bridge_concepts(
        self,
        target: str,
        known: set[str],
        threshold: float = 0.5,
    ) -> list[tuple[str, str, float]]:
        """Find bridge concepts: known concepts similar to missing prerequisites of target.

        Returns list of (known_bridge, missing_prereq, sim_score) sorted by sim desc.
        """
        if target not in self.nodes:
            return []

        missing_prereqs = []
        for src, dst, w in self.prereq_edges:
            if dst == target and src not in known:
                missing_prereqs.append(src)

        if not missing_prereqs:
            return []

        bridges = []
        for missing in missing_prereqs:
            for a, b, sim in self.sim_edges:
                if sim < threshold:
                    continue
                if a == missing and b in known:
                    bridges.append((b, missing, sim))
                elif b == missing and a in known:
                    bridges.append((a, missing, sim))

        bridges.sort(key=lambda x: x[2], reverse=True)
        return bridges


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def build_dual_graph(
    major_name: Optional[str] = None,
    include_predicted: bool = True,
    predicted_min_confidence: float = 0.85,
    sim_threshold: float = 0.5,
    max_sim_edges: int = 2000,
) -> DualGraph:
    """Build a dual graph from Neo4j.

    Loads KnowledgeConcept nodes (filtered by major if given).
    Loads prereq edges from CONCEPT_PREREQUISITE_FOR and optionally
    PREDICTED_PREREQUISITE (weight = confidence or 1.0).
    Predicted edges below ``predicted_min_confidence`` are dropped so that
    low-quality LLM/embedding predictions do not pollute the prerequisite path.
    Builds similarity edges from SEMANTIC_SIMILARITY or from embeddings
    cached by prereq_prediction.embedder if no such edges exist.
    """
    driver = get_neo4j_driver()

    nodes = []
    embeddings: dict[str, list[float]] = {}
    prereq_edges: list[tuple[str, str, float]] = []
    sim_edges: list[tuple[str, str, float]] = []

    try:
        with driver.session() as session:
            # Load nodes
            if major_name:
                result = session.run(
                    f"""
                    MATCH (m:{LABEL_MAJOR} {{name: $major_name}})
                          -[:{REL_BELONGS_TO}]->(c:{LABEL_COURSE})
                          -[:{REL_COVERS}]->(k:{LABEL_KNOWLEDGE_POINT})
                    RETURN k.name AS name
                    """,
                    major_name=major_name,
                )
            else:
                result = session.run(
                    f"""
                    MATCH (k:{LABEL_KNOWLEDGE_POINT})
                    RETURN k.name AS name
                    """
                )

            for row in result:
                name = row.get("name")
                if name:
                    nodes.append(name)

            nodes_set = set(nodes)

            # Load embeddings from cache (do not rely on Neo4j property)
            try:
                embedder = KnowledgeEmbedder()
                embeddings = embedder.compute_embeddings()
            except Exception:
                embeddings = {}

            # Load prerequisite edges
            result = session.run(
                f"""
                MATCH (a:{LABEL_KNOWLEDGE_POINT})-[r:{REL_PREREQUISITE_OF}]->(b:{LABEL_KNOWLEDGE_POINT})
                RETURN a.name AS src, b.name AS dst, r.weight AS weight
                """
            )
            for row in result:
                src, dst = row.get("src"), row.get("dst")
                if src in nodes_set and dst in nodes_set:
                    w = row.get("weight")
                    weight = float(w) if w is not None else 1.0
                    prereq_edges.append((src, dst, weight))

            if include_predicted:
                result = session.run(
                    f"""
                    MATCH (a:{LABEL_KNOWLEDGE_POINT})-[r:{REL_PREDICTED_PREREQ}]->(b:{LABEL_KNOWLEDGE_POINT})
                    RETURN a.name AS src, b.name AS dst, r.confidence AS confidence
                    """
                )
                for row in result:
                    src, dst = row.get("src"), row.get("dst")
                    if src in nodes_set and dst in nodes_set:
                        c = row.get("confidence")
                        conf = float(c) if c is not None else 1.0
                        if conf < predicted_min_confidence:
                            continue
                        prereq_edges.append((src, dst, conf))

            # Load similarity edges from DB
            result = session.run(
                f"""
                MATCH (a:{LABEL_KNOWLEDGE_POINT})-[r:{REL_SEMANTIC_SIMILARITY}]-(b:{LABEL_KNOWLEDGE_POINT})
                RETURN a.name AS a, b.name AS b, r.weight AS weight
                """
            )
            db_sim_edges = []
            for row in result:
                a, b = row.get("a"), row.get("b")
                if a in nodes_set and b in nodes_set and a != b:
                    w = row.get("weight")
                    weight = float(w) if w is not None else 0.0
                    db_sim_edges.append((a, b, weight))

            if db_sim_edges:
                db_sim_edges.sort(key=lambda x: x[2], reverse=True)
                sim_edges = db_sim_edges[:max_sim_edges]
            else:
                # Build similarity from embeddings
                nodes_with_emb = [n for n in nodes if n in embeddings]
                if len(nodes_with_emb) >= 2:
                    emb_matrix = np.array([embeddings[n] for n in nodes_with_emb])
                    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
                    norms[norms == 0] = 1e-9
                    normalized = emb_matrix / norms
                    sim_matrix = np.dot(normalized, normalized.T)

                    pairs = []
                    for i in range(len(nodes_with_emb)):
                        for j in range(i + 1, len(nodes_with_emb)):
                            sim = float(sim_matrix[i, j])
                            if sim >= sim_threshold:
                                pairs.append((nodes_with_emb[i], nodes_with_emb[j], sim))
                    pairs.sort(key=lambda x: x[2], reverse=True)
                    sim_edges = pairs[:max_sim_edges]

    except Exception:
        # Return empty graph on any error so callers don't crash
        pass

    return DualGraph(
        nodes=nodes,
        prereq_edges=prereq_edges,
        sim_edges=sim_edges,
        embeddings=embeddings,
    )
