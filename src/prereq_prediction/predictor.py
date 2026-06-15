"""Main predictor: compute scores and store predicted prerequisites in Neo4j."""
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from src.config import get_neo4j_driver
from src.models.schema import (
    LABEL_KNOWLEDGE_POINT,
    REL_PREREQUISITE_OF,
    REL_PREDICTED_PREREQ,
)
from .embedder import KnowledgeEmbedder
from .graph_mlp import train_predictor, predict_prerequisite_score


def predict_and_store(
    threshold: float = 0.7,
    max_predictions: int = 500,
    dry_run: bool = False,
) -> dict:
    """训练预测器，基于 embedding 邻域生成候选对，评分后存储 Top 预测边。

    为了避免 O(N^2) 全对评分，先用 embedding cosine 筛选候选对
    （阈值 threshold * 0.5，至少 0.25），再对候选对调用 MLP/规则评分。
    """
    # 1. 计算 embeddings
    embedder = KnowledgeEmbedder()
    embeddings = embedder.compute_embeddings()
    if not embeddings:
        return {
            "status": "error",
            "message": "Embeddings unavailable (sentence-transformers may not be installed).",
            "stored_count": 0,
            "threshold": threshold,
            "method": "none",
        }

    # 2. 训练预测器
    clf = train_predictor()
    method = "mlp" if clf is not None else "cosine"

    driver = get_neo4j_driver()

    # 3. 读取所有概念名和已有边
    with driver.session() as session:
        names_result = session.run(f"""
            MATCH (k:{LABEL_KNOWLEDGE_POINT})
            RETURN k.name AS name
        """)
        names = [r["name"] for r in names_result]

        existing_result = session.run(f"""
            MATCH (a:{LABEL_KNOWLEDGE_POINT})-[r:{REL_PREREQUISITE_OF}|{REL_PREDICTED_PREREQ}]->(b:{LABEL_KNOWLEDGE_POINT})
            RETURN a.name AS src, b.name AS dst, type(r) AS rel_type
        """)
        existing_edges = set((r["src"], r["dst"]) for r in existing_result)

    if len(names) < 2:
        return {
            "status": "error",
            "message": "Too few knowledge concepts to predict prerequisites.",
            "stored_count": 0,
            "threshold": threshold,
            "method": method,
        }

    # 4. 用 embedding 快速筛选候选对（避免全对评分）
    vectors = np.array([embeddings.get(name, np.zeros(384)) for name in names], dtype=np.float32)
    # L2 归一化，并处理 NaN/Inf
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    vectors = vectors / norms
    vectors = np.nan_to_num(vectors, nan=0.0, posinf=0.0, neginf=0.0)

    candidate_threshold = max(threshold * 0.5, 0.25)
    max_candidates_for_mlp = 1000  # 限制进入 MLP 评分的候选对数量

    # 先收集所有满足 embedding 阈值的候选对
    embedding_candidates = []
    for i, src in enumerate(names):
        sims = vectors @ vectors[i]
        sims = np.nan_to_num(sims, nan=0.0, posinf=0.0, neginf=0.0)
        for j in range(len(names)):
            if i == j:
                continue
            sim = float(sims[j])
            if sim < candidate_threshold:
                continue
            dst = names[j]
            if (src, dst) in existing_edges:
                continue
            embedding_candidates.append((sim, src, dst))

    # 按 embedding 相似度排序，取 Top-K 进入 MLP/规则评分
    embedding_candidates.sort(reverse=True, key=lambda x: x[0])
    if len(embedding_candidates) > max_candidates_for_mlp:
        embedding_candidates = embedding_candidates[:max_candidates_for_mlp]

    candidates = []
    for sim, src, dst in embedding_candidates:
        score = predict_prerequisite_score(src, dst, embedder, clf=clf)
        if score >= threshold:
            candidates.append((score, src, dst))

    # 5. 排序并取 Top
    candidates.sort(reverse=True, key=lambda x: x[0])
    top_candidates = candidates[:max_predictions]

    stored_count = 0
    if not dry_run and top_candidates:
        with driver.session() as session:
            for score, src, dst in top_candidates:
                session.run(f"""
                    MATCH (a:{LABEL_KNOWLEDGE_POINT} {{name: $src}})
                    MATCH (b:{LABEL_KNOWLEDGE_POINT} {{name: $dst}})
                    MERGE (a)-[r:{REL_PREDICTED_PREREQ}]->(b)
                    ON CREATE SET r.confidence = $confidence,
                                  r.method = $method,
                                  r.created_at = $created_at
                    ON MATCH SET  r.confidence = $confidence,
                                  r.method = $method,
                                  r.updated_at = $created_at
                """, src=src, dst=dst, confidence=round(score, 6),
                    method=method,
                    created_at=datetime.now(timezone.utc).isoformat())
                stored_count += 1

    return {
        "status": "success" if not dry_run else "dry_run",
        "stored_count": stored_count,
        "threshold": threshold,
        "method": method,
        "scored_pairs": len(candidates),
        "top_predictions": [
            {"src": src, "dst": dst, "score": round(score, 4)}
            for score, src, dst in top_candidates
        ],
    }
