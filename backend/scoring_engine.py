"""
Deterministic Filtering and Scoring Engine
Implements the exact mathematical models, hard constraint filters,
priority weights, and deterministic tie-breaking rules specified in ai_growth.pdf.
"""

from typing import List, Dict, Any, Optional, Union, Tuple
from backend.data_access import dal, MerchantNotFoundError

# -----------------------------------------------------------------------------
# Exact Priority Weights Specification
# -----------------------------------------------------------------------------
PRIORITY_WEIGHTS: Dict[str, Dict[str, float]] = {
    "cheapest": {"price_weight": 0.90, "rating_weight": 0.10},
    "highest_rated": {"price_weight": 0.10, "rating_weight": 0.90},
    "best_balance": {"price_weight": 0.50, "rating_weight": 0.50}
}


def normalize_priority(priority: str) -> str:
    """Normalizes priority string into canonical key ('cheapest', 'highest_rated', 'best_balance')."""
    p = priority.strip().lower().replace(" ", "_").replace("-", "_")
    if p in ("cheapest", "price", "low_price"):
        return "cheapest"
    if p in ("highest_rated", "rating", "top_rated", "highest_rating"):
        return "highest_rated"
    if p in ("best_balance", "balanced", "value", "balance"):
        return "best_balance"
    raise ValueError(f"Invalid priority '{priority}'. Must be one of: 'cheapest', 'highest_rated', 'best_balance'.")


def check_size_match(requested_size: Union[str, int], available_sizes: List[Any]) -> bool:
    """Case-insensitive and type-flexible check for size matching."""
    req_str = str(requested_size).strip().lower()
    for s in available_sizes:
        if str(s).strip().lower() == req_str:
            return True
    return False


def check_color_match(requested_color: str, available_colors: List[Any]) -> bool:
    """Case-insensitive check for color matching."""
    req_col = requested_color.strip().lower()
    for c in available_colors:
        if str(c).strip().lower() == req_col:
            return True
    return False


def passes_hard_constraints(
    product: Dict[str, Any],
    stock: int,
    budget: Optional[int] = None,
    category: Optional[str] = None,
    size: Optional[Union[str, int]] = None,
    color: Optional[str] = None,
    required_quantity: int = 1,
    attributes: Optional[Dict[str, Any]] = None
) -> Tuple[bool, Optional[str]]:
    """
    Evaluates all deterministic hard constraints before scoring.
    Returns (True, None) if product passes, or (False, reason) if rejected.
    """
    # 1. Stock availability constraint
    if stock < required_quantity:
        return False, f"Insufficient stock: required {required_quantity}, available {stock}"

    # 2. Budget constraint (price ceiling)
    price = product.get("price", 0)
    if budget is not None and price > budget:
        return False, f"Price ₹{price} exceeds budget ₹{budget}"

    # 3. Category constraint
    if category is not None:
        prod_cat = product.get("category", "").strip().lower()
        if prod_cat != category.strip().lower():
            return False, f"Category '{prod_cat}' does not match required '{category}'"

    # 4. Product attributes constraints
    prod_attrs = product.get("attributes", {})
    if not isinstance(prod_attrs, dict):
        prod_attrs = {}

    # Size constraint
    if size is not None:
        avail_sizes = prod_attrs.get("sizes", [])
        if not isinstance(avail_sizes, list) or not check_size_match(size, avail_sizes):
            return False, f"Size '{size}' not available in {avail_sizes}"

    # Color constraint
    if color is not None:
        avail_colors = prod_attrs.get("colors", [])
        if not isinstance(avail_colors, list) or not check_color_match(color, avail_colors):
            return False, f"Color '{color}' not available in {avail_colors}"

    # Custom attributes matching
    if attributes:
        for k, v in attributes.items():
            if k in ("sizes", "colors", "size", "color"):
                continue
            prod_val = prod_attrs.get(k)
            if prod_val is None:
                return False, f"Missing required attribute '{k}'"
            if isinstance(prod_val, list):
                v_str = str(v).strip().lower()
                if not any(str(item).strip().lower() == v_str for item in prod_val):
                    return False, f"Attribute '{k}' value '{v}' not found in {prod_val}"
            else:
                if str(prod_val).strip().lower() != str(v).strip().lower():
                    return False, f"Attribute '{k}' value '{prod_val}' does not match required '{v}'"

    return True, None


def calculate_price_score(price: float, min_price: float, max_price: float) -> float:
    """
    Price Score = (Max Price - Product Price) / (Max Price - Min Price)
    If Max Price == Min Price, returns 1.0 (no price penalty / single price point).
    """
    if max_price <= min_price:
        return 1.0
    score = (max_price - price) / (max_price - min_price)
    return round(float(score), 4)


def calculate_rating_score(rating: float) -> float:
    """
    Rating Score = Rating / 5.0
    """
    return round(float(rating) / 5.0, 4)


def calculate_value_score(price_score: float, rating_score: float, priority: str) -> float:
    """
    Value Score = (Price Weight * Price Score) + (Rating Weight * Rating Score)
    """
    canonical_priority = normalize_priority(priority)
    weights = PRIORITY_WEIGHTS[canonical_priority]
    pw = weights["price_weight"]
    rw = weights["rating_weight"]
    value_score = (pw * price_score) + (rw * rating_score)
    return round(float(value_score), 4)


class ScoringEngine:
    """
    Deterministic Filtering & Scoring Engine for multi-merchant product offers.
    """

    def score_offers(
        self,
        candidate_offers: List[Dict[str, Any]],
        priority: str,
        budget: Optional[int] = None,
        category: Optional[str] = None,
        size: Optional[Union[str, int]] = None,
        color: Optional[str] = None,
        required_quantity: int = 1,
        attributes: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Filters candidate offers by hard constraints, then computes exact
        Price Score, Rating Score, and Value Score, and deterministically ranks them.
        """
        canonical_priority = normalize_priority(priority)

        # 1. Apply hard constraint filtering
        valid_offers = []
        for offer in candidate_offers:
            stock = offer.get("available_stock", offer.get("stock", 0))
            is_valid, reason = passes_hard_constraints(
                product=offer,
                stock=stock,
                budget=budget,
                category=category,
                size=size,
                color=color,
                required_quantity=required_quantity,
                attributes=attributes
            )
            if is_valid:
                valid_offers.append(offer)

        if not valid_offers:
            return []

        # 2. Determine Min Price and Max Price across valid candidate offers
        prices = [float(o["price"]) for o in valid_offers]
        min_price = min(prices)
        max_price = max(prices)

        # 3. Calculate scores for each valid offer
        scored_offers = []
        for offer in valid_offers:
            price = float(offer["price"])
            rating = float(offer["rating"])
            stock = int(offer.get("available_stock", offer.get("stock", 0)))

            price_score = calculate_price_score(price, min_price, max_price)
            rating_score = calculate_rating_score(rating)
            value_score = calculate_value_score(price_score, rating_score, canonical_priority)

            scored_offers.append({
                "p_id": offer["p_id"],
                "p_name": offer["p_name"],
                "category": offer["category"],
                "description": offer.get("description", ""),
                "merchant": offer["merchant"],
                "price": int(offer["price"]),
                "rating": float(offer["rating"]),
                "available_stock": stock,
                "price_score": price_score,
                "rating_score": rating_score,
                "value_score": value_score,
                "selected_priority": canonical_priority,
                "similarity_score": offer.get("similarity_score")
            })

        # 4. Deterministic Tie-Breaking Sort
        # Rule:
        # 1. Higher Value Score wins (-value_score)
        # 2. If tied, Higher Rating Score wins (-rating_score)
        # 3. If tied, Lower Price wins (price)
        # 4. If tied, Higher Stock wins (-available_stock)
        scored_offers.sort(
            key=lambda x: (
                -x["value_score"],
                -x["rating_score"],
                x["price"],
                -x["available_stock"]
            )
        )

        return scored_offers

    def compare_merchants_for_product(
        self,
        p_id: str,
        priority: str,
        budget: Optional[int] = None,
        size: Optional[Union[str, int]] = None,
        color: Optional[str] = None,
        required_quantity: int = 1,
        attributes: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetches live offers for a single p_id from both ShopNest and CartWave via the DAL,
        applies hard constraints, and ranks the offers using the exact scoring engine.
        """
        candidate_offers = []
        for merchant in ("shopnest", "cartwave"):
            product = dal.get_product(merchant, p_id)
            if not product:
                continue
            stock = dal.get_stock(merchant, p_id)
            offer = dict(product)
            offer["merchant"] = merchant
            offer["available_stock"] = stock
            candidate_offers.append(offer)

        if not candidate_offers:
            return []

        return self.score_offers(
            candidate_offers=candidate_offers,
            priority=priority,
            budget=budget,
            category=candidate_offers[0].get("category") if candidate_offers else None,
            size=size,
            color=color,
            required_quantity=required_quantity,
            attributes=attributes
        )

    def evaluate_candidates(
        self,
        candidate_items: List[Dict[str, Any]],
        priority: str,
        budget: Optional[int] = None,
        category: Optional[str] = None,
        size: Optional[Union[str, int]] = None,
        color: Optional[str] = None,
        required_quantity: int = 1,
        attributes: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Expands candidate items (e.g. from semantic search or product IDs) across both
        ShopNest and CartWave via DAL, filters by hard constraints, and deterministically
        scores and ranks all valid options.
        """
        seen_pairs = set()
        expanded_offers = []

        for item in candidate_items:
            p_id = item["p_id"]
            target_merchants = item.get("target_merchants", ["shopnest", "cartwave"])

            for merchant in target_merchants:
                pair_key = (merchant, p_id)
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                product = dal.get_product(merchant, p_id)
                if not product:
                    continue

                stock = dal.get_stock(merchant, p_id)
                offer = dict(product)
                offer["merchant"] = merchant
                offer["available_stock"] = stock
                if "similarity_score" in item:
                    offer["similarity_score"] = item["similarity_score"]
                expanded_offers.append(offer)

        return self.score_offers(
            candidate_offers=expanded_offers,
            priority=priority,
            budget=budget,
            category=category,
            size=size,
            color=color,
            required_quantity=required_quantity,
            attributes=attributes
        )


# Global singleton instance
scoring_engine = ScoringEngine()
