"""语义相似度计算：为节点计算 embedding 并写入 SEMANTIC_SIMILARITY 关系。"""
import math
import os
import pickle
import warnings
from datetime import datetime
from typing import Optional

import numpy as np

from src.config import get_neo4j_driver
from src.models.schema import (
    LABEL_KNOWLEDGE_POINT,
    LABEL_COURSE,
    REL_SEMANTIC_SIMILARITY,
)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个向量之间的余弦相似度。"""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _load_cached_embeddings(entity_label: str) -> Optional[dict]:
    """尝试从本地缓存加载 embeddings。"""
    cache_file = (
        "data/knowledge_embeddings.pkl"
        if entity_label == LABEL_KNOWLEDGE_POINT
        else "data/course_embeddings.pkl"
    )
    if not os.path.exists(cache_file):
        return None
    with open(cache_file, "rb") as f:
        return pickle.load(f)


def _save_cached_embeddings(entity_label: str, embeddings: dict):
    """将 embeddings 缓存到本地文件。"""
    os.makedirs("data", exist_ok=True)
    cache_file = (
        "data/knowledge_embeddings.pkl"
        if entity_label == LABEL_KNOWLEDGE_POINT
        else "data/course_embeddings.pkl"
    )
    with open(cache_file, "wb") as f:
        pickle.dump(embeddings, f)


def compute_and_store_similarity(
    entity_label: str = LABEL_KNOWLEDGE_POINT,
    threshold: float = 0.75,
    max_edges: int = 5000,
    batch_size: int = 1000,
) -> dict:
    """
    为指定标签的节点计算语义相似度，并将结果写入 Neo4j。

    参数：
        entity_label: 节点标签，默认 LABEL_KNOWLEDGE_POINT（"KnowledgeConcept"），
                      也可以是 LABEL_COURSE（"Course"）。
        threshold:    余弦相似度阈值，只有 >= threshold 的边才会被写入。
        max_edges:    最多写入的边数（按相似度从高到低取 top N）。
        batch_size:   从 Neo4j 读取节点时的批次大小。

    返回：
        dict，包含 status、stored_count、threshold、entity_label。
    """
    # 尝试加载 sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return {
            "status": "fallback",
            "message": (
                "sentence-transformers is not installed. "
                "Please install it to enable semantic similarity computation."
            ),
            "stored_count": 0,
            "threshold": threshold,
            "entity_label": entity_label,
        }

    model_name = "paraphrase-multilingual-MiniLM-L12-v2"
    model = SentenceTransformer(model_name)

    driver = get_neo4j_driver()

    # 1. 从 Neo4j 读取所有节点（name + description）
    nodes = []
    with driver.session() as session:
        result = session.run(
            f"""
            MATCH (n:{entity_label})
            RETURN n.name AS name, n.description AS description
            """
        )
        for record in result:
            name = record["name"]
            desc = record["description"] or ""
            text = f"{name} {desc}".strip()
            nodes.append({"name": name, "text": text})

    if not nodes:
        return {
            "status": "no_nodes",
            "message": f"No nodes found with label '{entity_label}'.",
            "stored_count": 0,
            "threshold": threshold,
            "entity_label": entity_label,
        }

    # 2. 尝试加载缓存的 embeddings
    cached = _load_cached_embeddings(entity_label)
    embeddings: dict[str, np.ndarray] = {}

    if cached is not None:
        embeddings = cached
        # 检查是否有新节点需要重新计算
        cached_names = set(embeddings.keys())
        current_names = {n["name"] for n in nodes}
        if cached_names != current_names:
            warnings.warn(
                "Cached embeddings mismatch with current nodes; recomputing all.",
                stacklevel=2,
            )
            embeddings = {}

    if not embeddings:
        texts = [n["text"] for n in nodes]
        vectors = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        vectors = np.nan_to_num(vectors, nan=0.0, posinf=0.0, neginf=0.0)
        embeddings = {n["name"]: vectors[i] for i, n in enumerate(nodes)}
        _save_cached_embeddings(entity_label, embeddings)

    # 3. 计算相似度并收集候选边
    names = [n["name"] for n in nodes]
    candidate_edges = []
    total = len(names)
    for i in range(total):
        for j in range(i + 1, total):
            sim = _cosine_similarity(embeddings[names[i]], embeddings[names[j]])
            if sim >= threshold:
                candidate_edges.append((names[i], names[j], sim))

    # 按相似度降序排序，取 top max_edges
    candidate_edges.sort(key=lambda x: x[2], reverse=True)
    edges_to_store = candidate_edges[:max_edges]

    # 4. 写入 Neo4j（使用 MERGE 避免重复）
    stored_count = 0
    with driver.session() as session:
        for a_name, b_name, sim in edges_to_store:
            result = session.run(
                f"""
                MATCH (a:{entity_label} {{name: $a_name}})
                MATCH (b:{entity_label} {{name: $b_name}})
                MERGE (a)-[r:{REL_SEMANTIC_SIMILARITY}]->(b)
                ON CREATE SET r.weight = $sim, r.created_at = $now
                ON MATCH SET r.weight = $sim
                RETURN r.created_at AS created
                """,
                a_name=a_name,
                b_name=b_name,
                sim=sim,
                now=datetime.utcnow().isoformat(),
            )
            record = result.single()
            if record and record["created"]:
                stored_count += 1

    return {
        "status": "success",
        "stored_count": stored_count,
        "threshold": threshold,
        "entity_label": entity_label,
        "total_candidates": len(candidate_edges),
        "max_edges": max_edges,
    }
