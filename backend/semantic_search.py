import os
import json
import threading
from typing import List, Dict, Any, Optional
import numpy as np

from backend.data_access import dal, MerchantNotFoundError

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "catalog_embeddings.json")


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
    Local Semantic Search Engine managing precomputed embeddings and cosine similarity searches
    for merchant product catalogs with lazy loading and low-memory fallbacks.
    """

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self._model = None
        self._model_lock = threading.Lock()

        # In-memory caches per merchant:
        self._embeddings_cache: Dict[str, np.ndarray] = {}
        self._products_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._cache_lock = threading.Lock()

        # Load precomputed catalog embeddings immediately on init (sub-millisecond, zero PyTorch RAM)
        self._load_precomputed_embeddings()

    def _load_precomputed_embeddings(self) -> None:
        """Loads precomputed catalog embeddings from JSON if available."""
        if not os.path.exists(EMBEDDINGS_FILE):
            return
        try:
            with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            with self._cache_lock:
                for merchant in ("shopnest", "cartwave"):
                    if merchant in data:
                        products = dal.get_products(merchant)
                        p_map = {p["p_id"]: p for p in products}
                        p_ids = data[merchant].get("p_ids", [])
                        embeddings_list = data[merchant].get("embeddings", [])
                        
                        ordered_products = [p_map[pid] for pid in p_ids if pid in p_map]
                        if ordered_products and len(ordered_products) == len(embeddings_list):
                            self._products_cache[merchant] = ordered_products
                            self._embeddings_cache[merchant] = np.array(embeddings_list, dtype=np.float32)
        except Exception as e:
            print(f"[SemanticSearch] Warning loading precomputed embeddings: {e}")

    @property
    def model(self):
        """Lazy loader for SentenceTransformer to optimize memory and startup time."""
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    try:
                        import torch
                        torch.set_num_threads(1)
                    except Exception:
                        pass
                    from sentence_transformers import SentenceTransformer
                    self._model = SentenceTransformer(self.model_name)
        return self._model

    def index_merchant(self, merchant: str, force_reload: bool = False) -> None:
        """
        Computes normalized embeddings and caches them in memory.
        """
        merchant_clean = merchant.strip().lower()
        if merchant_clean not in ("shopnest", "cartwave"):
            raise MerchantNotFoundError(f"Unknown merchant '{merchant}'. Must be 'shopnest' or 'cartwave'.")

        with self._cache_lock:
            if not force_reload and merchant_clean in self._embeddings_cache and len(self._products_cache.get(merchant_clean, [])) > 0:
                return

            products = dal.get_products(merchant_clean)
            if not products:
                self._embeddings_cache[merchant_clean] = np.empty((0, 384), dtype=np.float32)
                self._products_cache[merchant_clean] = []
                return

            try:
                texts = [format_product_text(p) for p in products]
                embeddings = self.model.encode(
                    texts,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
                self._embeddings_cache[merchant_clean] = embeddings
                self._products_cache[merchant_clean] = products
            except Exception as e:
                print(f"[SemanticSearch] Model encode fallback: {e}")
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

        # Ensure embeddings are loaded
        if merchant_clean not in self._embeddings_cache:
            self.index_merchant(merchant_clean)

        with self._cache_lock:
            embeddings = self._embeddings_cache.get(merchant_clean)
            products = self._products_cache.get(merchant_clean, [])

        if not products:
            return []

        # Try dense neural vector search
        scores = None
        if embeddings is not None and len(embeddings) == len(products) and embeddings.shape[0] > 0:
            try:
                query_embedding = self.model.encode(
                    [query_clean],
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )[0]
                scores = np.dot(embeddings, query_embedding)
            except Exception as e:
                print(f"[SemanticSearch] Query embedding error: {e}")
                scores = None

        # Fallback to lexical-semantic overlap if neural encoder fails or OOM
        if scores is None:
            scores = []
            query_tokens = set(query_clean.lower().split())
            for p in products:
                text = f"{p.get('p_name', '')} {p.get('category', '')} {p.get('description', '')}".lower()
                matches = sum(1 for t in query_tokens if t in text)
                score = matches / max(1, len(query_tokens))
                scores.append(score)
            scores = np.array(scores, dtype=np.float32)

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
            if m not in self._embeddings_cache:
                self.index_merchant(m)


# Global singleton instance for use across the application
semantic_search_engine = SemanticSearchEngine()

