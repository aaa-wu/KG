"""Knowledge embedder using sentence-transformers."""

import os
import pickle
from typing import Optional

import numpy as np

from src.config import get_neo4j_driver
from src.models.schema import LABEL_KNOWLEDGE_POINT


# Default cache path relative to project root
DEFAULT_EMBEDDING_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data",
    "knowledge_embeddings.pkl",
)


class KnowledgeEmbedder:
    """Compute and cache sentence embeddings for KnowledgeConcept nodes."""

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
                 cache_path: str = DEFAULT_EMBEDDING_PATH):
        self.model_name = model_name
        self.cache_path = cache_path
        self._model = None
        self._embeddings: dict[str, list[float]] = {}

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        except Exception as exc:
            print(f"[KnowledgeEmbedder] Warning: sentence-transformers not available ({exc})")
            self._model = None
        return self._model

    def compute_embeddings(self, force_refresh: bool = False) -> dict[str, list[float]]:
        """Load from cache or compute embeddings for all KnowledgeConcept nodes."""
        if not force_refresh and os.path.exists(self.cache_path):
            with open(self.cache_path, "rb") as f:
                self._embeddings = pickle.load(f)
            print(f"[KnowledgeEmbedder] Loaded {len(self._embeddings)} embeddings from cache.")
            return self._embeddings

        model = self._load_model()
        if model is None:
            print("[KnowledgeEmbedder] Warning: sentence-transformers not installed, returning empty embeddings.")
            return {}

        driver = get_neo4j_driver()
        texts = []
        names = []
        with driver.session() as session:
            result = session.run(f"""
                MATCH (k:{LABEL_KNOWLEDGE_POINT})
                RETURN k.name AS name, k.description AS description
            """)
            for record in result:
                name = record["name"]
                description = record["description"] or ""
                text = f"{name} {description}".strip()
                texts.append(text)
                names.append(name)

        if not texts:
            print("[KnowledgeEmbedder] No knowledge concepts found in graph.")
            self._embeddings = {}
            return self._embeddings

        vectors = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        # Sanitize: replace NaN/Inf with 0 to avoid downstream numerical issues
        vectors = np.nan_to_num(vectors, nan=0.0, posinf=0.0, neginf=0.0)
        self._embeddings = {name: vec.tolist() for name, vec in zip(names, vectors)}

        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, "wb") as f:
            pickle.dump(self._embeddings, f)
        print(f"[KnowledgeEmbedder] Computed and cached {len(self._embeddings)} embeddings.")
        return self._embeddings

    def get_embedding(self, name: str) -> Optional[list[float]]:
        """Return the embedding vector for a given concept name."""
        if not self._embeddings:
            self.compute_embeddings()
        return self._embeddings.get(name)

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        a_arr = np.array(a, dtype=np.float32)
        b_arr = np.array(b, dtype=np.float32)
        norm_a = np.linalg.norm(a_arr)
        norm_b = np.linalg.norm(b_arr)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))
