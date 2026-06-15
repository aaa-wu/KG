"""Graph-aware MLP classifier for prerequisite prediction."""

import os
import pickle
from typing import Optional

import numpy as np

from src.config import get_neo4j_driver
from src.models.schema import (
    LABEL_KNOWLEDGE_POINT,
    REL_PREREQUISITE_OF,
    REL_PREDICTED_PREREQ,
)
from .embedder import KnowledgeEmbedder


DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data",
    "prereq_predictor.pkl",
)


def _get_neighbors(session, name: str, direction: str = "out") -> set[str]:
    """Return neighbor names for a given node via REL_PREREQUISITE_OF."""
    if direction == "out":
        query = f"""
            MATCH (a:{LABEL_KNOWLEDGE_POINT} {{name: $name}})
                  -[:{REL_PREREQUISITE_OF}]->(b:{LABEL_KNOWLEDGE_POINT})
            RETURN b.name AS neighbor
        """
    else:
        query = f"""
            MATCH (a:{LABEL_KNOWLEDGE_POINT} {{name: $name}})
                  <-[:{REL_PREREQUISITE_OF}]-(b:{LABEL_KNOWLEDGE_POINT})
            RETURN b.name AS neighbor
        """
    result = session.run(query, name=name)
    return {r["neighbor"] for r in result if r["neighbor"] is not None}


def _degree(session, name: str, direction: str = "out") -> int:
    """Return degree count for a node via REL_PREREQUISITE_OF."""
    if direction == "out":
        query = f"""
            MATCH (a:{LABEL_KNOWLEDGE_POINT} {{name: $name}})
                  -[:{REL_PREREQUISITE_OF}]->(b:{LABEL_KNOWLEDGE_POINT})
            RETURN count(b) AS deg
        """
    else:
        query = f"""
            MATCH (a:{LABEL_KNOWLEDGE_POINT} {{name: $name}})
                  <-[:{REL_PREREQUISITE_OF}]-(b:{LABEL_KNOWLEDGE_POINT})
            RETURN count(b) AS deg
        """
    result = session.run(query, name=name)
    record = result.single()
    return record["deg"] if record else 0


def _shortest_path_length(session, name_a: str, name_b: str) -> int:
    """Return shortest path length between a and b, or 999 if none."""
    query = f"""
        MATCH (a:{LABEL_KNOWLEDGE_POINT} {{name: $name_a}}),
              (b:{LABEL_KNOWLEDGE_POINT} {{name: $name_b}})
        MATCH p = shortestPath((a)-[:{REL_PREREQUISITE_OF}|{REL_PREDICTED_PREREQ}*]-(b))
        RETURN length(p) AS plen
    """
    result = session.run(query, name_a=name_a, name_b=name_b)
    record = result.single()
    return record["plen"] if record else 999


def _compute_graph_features(
    session,
    name_a: str,
    name_b: str,
    embedder: KnowledgeEmbedder,
) -> Optional[np.ndarray]:
    """Compute feature vector for a candidate (a -> b) prerequisite pair.

    Features (6-dim):
      0. embedding cosine similarity
      1. common neighbors count
      2. Jaccard similarity of neighbor sets
      3. out-degree of a
      4. in-degree of b
      5. L2 embedding difference

    Note: shortestPath is excluded from inference to keep latency low;
    structural information is captured by local neighborhood features.
    """
    emb_a = embedder.get_embedding(name_a)
    emb_b = embedder.get_embedding(name_b)
    if emb_a is None or emb_b is None:
        return None

    cos_sim = KnowledgeEmbedder.cosine_similarity(emb_a, emb_b)

    a_arr = np.array(emb_a, dtype=np.float32)
    b_arr = np.array(emb_b, dtype=np.float32)
    l2_diff = float(np.linalg.norm(a_arr - b_arr))

    neigh_a = _get_neighbors(session, name_a, direction="out")
    neigh_b_in = _get_neighbors(session, name_b, direction="in")

    common = len(neigh_a & neigh_b_in)

    union = neigh_a | neigh_b_in
    jaccard = len(neigh_a & neigh_b_in) / len(union) if union else 0.0

    out_deg_a = _degree(session, name_a, direction="out")
    in_deg_b = _degree(session, name_b, direction="in")

    features = np.array([
        cos_sim,
        common,
        jaccard,
        out_deg_a,
        in_deg_b,
        l2_diff,
    ], dtype=np.float32)
    return features


def build_training_data(
    session,
    embedder: KnowledgeEmbedder,
) -> tuple[np.ndarray, np.ndarray]:
    """Build training matrix X and label vector y.

    Positive samples = existing CONCEPT_PREREQUISITE_FOR edges.
    Negative samples = random non-edges (same count as positives).
    """
    # Positive edges
    pos_result = session.run(f"""
        MATCH (a:{LABEL_KNOWLEDGE_POINT})-[r:{REL_PREREQUISITE_OF}]->(b:{LABEL_KNOWLEDGE_POINT})
        RETURN a.name AS src, b.name AS dst
    """)
    pos_pairs = [(r["src"], r["dst"]) for r in pos_result]

    if not pos_pairs:
        return np.array([]), np.array([])

    # All possible pairs (src, dst) among nodes that have embeddings
    names = list(embedder._embeddings.keys())
    name_set = set(names)

    existing_edges = set(pos_pairs)

    # Also exclude predicted edges from negative sampling to keep training clean
    pred_result = session.run(f"""
        MATCH (a:{LABEL_KNOWLEDGE_POINT})-[r:{REL_PREDICTED_PREREQ}]->(b:{LABEL_KNOWLEDGE_POINT})
        RETURN a.name AS src, b.name AS dst
    """)
    for r in pred_result:
        existing_edges.add((r["src"], r["dst"]))

    # Generate negative samples: random non-edges
    np.random.seed(42)
    neg_pairs = []
    max_attempts = len(pos_pairs) * 20
    attempts = 0
    while len(neg_pairs) < len(pos_pairs) and attempts < max_attempts:
        a = np.random.choice(names)
        b = np.random.choice(names)
        attempts += 1
        if a == b:
            continue
        if (a, b) in existing_edges:
            continue
        neg_pairs.append((a, b))

    X_list = []
    y_list = []

    for src, dst in pos_pairs:
        feat = _compute_graph_features(session, src, dst, embedder)
        if feat is not None:
            X_list.append(feat)
            y_list.append(1)

    for src, dst in neg_pairs:
        feat = _compute_graph_features(session, src, dst, embedder)
        if feat is not None:
            X_list.append(feat)
            y_list.append(0)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    return X, y


def train_predictor(force_retrain: bool = False) -> Optional[object]:
    """Train and cache an MLPClassifier for prerequisite prediction.

    Returns the trained classifier or None if sklearn is unavailable
    or there are fewer than 10 training samples.
    """
    if not force_retrain and os.path.exists(DEFAULT_MODEL_PATH):
        with open(DEFAULT_MODEL_PATH, "rb") as f:
            clf = pickle.load(f)
        print(f"[graph_mlp] Loaded cached predictor from {DEFAULT_MODEL_PATH}")
        return clf

    try:
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:
        print(f"[graph_mlp] Warning: scikit-learn not available ({exc})")
        return None

    embedder = KnowledgeEmbedder()
    embeddings = embedder.compute_embeddings()
    if not embeddings:
        print("[graph_mlp] No embeddings available; cannot train.")
        return None

    driver = get_neo4j_driver()
    with driver.session() as session:
        X, y = build_training_data(session, embedder)

    if len(y) < 10:
        print(f"[graph_mlp] Too few samples ({len(y)}); skipping training.")
        return None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=500,
        early_stopping=True,
        random_state=42,
    )
    clf.fit(X_scaled, y)
    # Attach scaler so we can reuse it at inference time
    clf._scaler = scaler

    os.makedirs(os.path.dirname(DEFAULT_MODEL_PATH), exist_ok=True)
    with open(DEFAULT_MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)
    print(f"[graph_mlp] Trained and cached predictor ({len(y)} samples).")
    return clf


def predict_prerequisite_score(
    src: str,
    dst: str,
    embedder: KnowledgeEmbedder,
    clf: Optional[object] = None,
) -> float:
    """Predict prerequisite score for src -> dst.

    If clf is provided, use the MLP probability.
    Otherwise fall back to embedding cosine similarity.
    """
    driver = get_neo4j_driver()
    with driver.session() as session:
        feat = _compute_graph_features(session, src, dst, embedder)

    if feat is None:
        return 0.0

    if clf is not None:
        scaler = getattr(clf, "_scaler", None)
        if scaler is not None:
            feat_scaled = scaler.transform(feat.reshape(1, -1))
        else:
            feat_scaled = feat.reshape(1, -1)
        prob = clf.predict_proba(feat_scaled)[0][1]
        return float(prob)

    # Fallback: embedding cosine similarity
    emb_a = embedder.get_embedding(src)
    emb_b = embedder.get_embedding(dst)
    if emb_a is None or emb_b is None:
        return 0.0
    return KnowledgeEmbedder.cosine_similarity(emb_a, emb_b)
