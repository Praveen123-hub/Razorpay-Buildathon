"""
Local Semantic Search Module
Powered by sentence-transformers/all-MiniLM-L6-v2 running locally in Python.
Embeddings are computed, normalized, and cached per merchant, referencing products
strictly via the Merchant Data Access Layer.
"""

import threading
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

from backend.data_access import dal, MerchantNotFoundError

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def format_product_text(product: Dict[str, Any]) -> str:
    """
    Constructs a rich text representation of a product for semantic embedding,
    incorporating name, category, detailed description, and key specifications.
    """
    name = product.get("p_name", "").strip()
    category = product.get("category", "").strip()
    description = product.get("description", "").strip()

    parts = [
        f"Product Name: {name}",
        f"Category: {category}",
        f"Description: {description}"
    ]

    attributes = product.get("attributes", {})
    if isinstance(attributes, dict) and attributes:
        attr_parts = []
        for key, value in attributes.items():
            if value is None:
                continue
            if isinstance(value, list):
                attr_parts.append(f"{key}: {', '.join(str(v) for v in value)}")
            else:
                attr_parts.append(f"{key}: {value}")
        if attr_parts:
            parts.append("Specifications: " + "; ".join(attr_parts))

    return ". ".join(parts)


class SemanticSearchEngine:
    """
    Local Semantic Search Engine managing embeddings and cosine similarity searches
    for merchant product catalogs.
    """

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self._model: Optional[SentenceTransformer] = None
        self._model_lock = threading.Lock()

        # In-memory caches per merchant:
        # _embeddings_cache: merchant -> np.ndarray of shape (N, 384) (normalized)
        # _products_cache: merchant -> List[Dict[str, Any]] (matching rows in embedding matrix)
        self._embeddings_cache: Dict[str, np.ndarray] = {}
        self._products_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._cache_lock = threading.Lock()

    @property
    def model(self) -> SentenceTransformer:
        """Lazy loader for SentenceTransformer to optimize startup time."""
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    self._model = SentenceTransformer(self.model_name)
        return self._model

    def index_merchant(self, merchant: str, force_reload: bool = False) -> None:
        """
        Fetches products for a merchant via the Merchant Data Access Layer,
        computes normalized embeddings using all-MiniLM-L6-v2, and caches them in memory.
        """
        merchant_clean = merchant.strip().lower()
        if merchant_clean not in ("shopnest", "cartwave"):
            raise MerchantNotFoundError(f"Unknown merchant '{merchant}'. Must be 'shopnest' or 'cartwave'.")

        with self._cache_lock:
            if not force_reload and merchant_clean in self._embeddings_cache:
                return

            products = dal.get_products(merchant_clean)
            if not products:
                self._embeddings_cache[merchant_clean] = np.empty((0, 384), dtype=np.float32)
                self._products_cache[merchant_clean] = []
                return

            texts = [format_product_text(p) for p in products]
            embeddings = self.model.encode(
                texts,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )

            self._embeddings_cache[merchant_clean] = embeddings
            self._products_cache[merchant_clean] = products

    def search(
        self,
        merchant: str,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Encodes the natural language query, performs cosine similarity calculation
        against pre-computed product embeddings for the merchant, and returns the
        top-k ranked products with similarity scores.
        """
        merchant_clean = merchant.strip().lower()
        query_clean = query.strip()

        if not query_clean:
            return []

        # Ensure embeddings are pre-computed and cached
        if merchant_clean not in self._embeddings_cache:
            self.index_merchant(merchant_clean)

        with self._cache_lock:
            embeddings = self._embeddings_cache[merchant_clean]
            products = self._products_cache[merchant_clean]

        if len(products) == 0 or embeddings.shape[0] == 0:
            return []

        # Encode query to normalized 384-dimensional vector
        query_embedding = self.model.encode(
            [query_clean],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0]

        # Cosine similarity is exact dot product of normalized vectors
        scores = np.dot(embeddings, query_embedding)

        # Rank all products by similarity descending
        ranked_indices = np.argsort(scores)[::-1]

        results = []
        for idx in ranked_indices:
            score = float(scores[idx])
            if score < min_score:
                continue

            product = products[idx]
            results.append({
                "p_id": product.get("p_id"),
                "p_name": product.get("p_name"),
                "category": product.get("category"),
                "description": product.get("description"),
                "price": product.get("price"),
                "rating": product.get("rating"),
                "merchant": merchant_clean,
                "similarity_score": round(score, 4)
            })

            if len(results) >= top_k:
                break

        return results

    def warm_up(self) -> None:
        """Pre-indexes and caches embeddings for both supported merchants."""
        for m in ("shopnest", "cartwave"):
            self.index_merchant(m)


# Global singleton instance for use across the application
semantic_search_engine = SemanticSearchEngine()
