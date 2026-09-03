"""
Agent 1 — AI Buyer Agent Backend Service
Orchestrates:
1. Understanding user shopping requests and extracting requirements.
2. Managing multi-turn slot-filling dialogues (priority, size, budget, quantity).
3. Invoking Step 4A Local Semantic Search tool.
4. Invoking Step 4B Deterministic Filtering and Scoring engine tool.
5. Presenting deterministic winning offer and managing cart additions.
6. Invoking Step 5 Agent 2 Sales Improvement Agent after main product confirmation.
7. Presenting complementary recommendation and calculating final cart summary.
8. Returning READY_FOR_PURCHASE upon final order confirmation.
"""

import re
import json
import uuid
import urllib.request
import urllib.error
from typing import Dict, Any, List, Optional, Tuple

from backend.data_access import dal
from backend.semantic_search import semantic_search_engine
from backend.scoring_engine import scoring_engine
from backend.agent2_service import agent2_service
from backend.config import get_openrouter_api_key, get_agent1_model, OPENROUTER_BASE_URL
from backend.trust_layer import TrustLayer
from backend.schemas import Agent1ChatRequest, Agent1ChatResponse


PRIORITY_KEYWORDS = {
    "cheapest": ["cheapest", "lowest price", "most affordable", "lowest", "cheap", "least expensive", "budget friendly"],
    "highest_rated": [
        "highest rated", "best reviewed", "top rated", "best rating", "highest rating", "highest", "top review", "best reviews",
        "highly rated", "highly-rated", "best rated", "best-rated", "top-rated", "top reviewed", "highest_rated"
    ],
    "best_balance": ["best balance", "balance", "balanced", "value for money", "best value", "good price and good rating", "sweet spot", "best_balance"]
}

AFFIRMATIVE_KEYWORDS = ["yes", "yeah", "yup", "sure", "add", "add it", "please", "proceed", "ok", "okay", "confirm", "definitely", "yep", "do it", "place order"]
NEGATIVE_KEYWORDS = ["no", "nope", "nah", "skip", "don't", "dont", "cancel", "pass", "not now", "no thanks"]


def singularize_word(word: str) -> str:
    """Generic singularization of a single word."""
    lower = word.lower()
    if lower in ('jeans', 'pants', 'shorts', 'trousers', 'glasses', 'sunglasses'):
        return word
    if lower.endswith('ies') and not lower.endswith('hoodies'):
        return word[:-3] + 'y'
    if lower.endswith('s'):
        if lower.endswith('es'):
            for suffix in ('sh', 'ch', 's', 'x', 'z'):
                if lower[:-2].endswith(suffix):
                    return word[:-2]
            if lower.endswith('oes'):
                if lower.endswith('shoes'):
                    return word[:-1]
                return word[:-2]
        if not lower.endswith('ss') and not lower.endswith('us') and not lower.endswith('is') and not lower.endswith('as'):
            return word[:-1]
    return word


def singularize_query(query: str) -> str:
    """Normalizes plural product nouns in search query to singular form."""
    words = query.split()
    return ' '.join(singularize_word(w) for w in words)


def clean_and_get_last_word(text: str) -> str:
    """Extracts the singularized head noun of a query or product name."""
    text_clean = text.lower()
    # Split by common prepositions to isolate the primary item
    if ' for ' in text_clean:
        text_clean = text_clean.split(' for ')[0]
    if ' pack of ' in text_clean:
        text_clean = text_clean.split(' pack of ')[0]
    if ' set of ' in text_clean:
        text_clean = text_clean.split(' set of ')[0]
    # Remove dimensions, capacities, and unit specifications
    text_clean = re.sub(r'\b\d+mm\b', '', text_clean)
    text_clean = re.sub(r'\b\d+%\b', '', text_clean)
    text_clean = re.sub(r'\b\d+w\b', '', text_clean)
    text_clean = re.sub(r'\b\d+ml\b', '', text_clean)
    text_clean = re.sub(r'\b\d+-meter\b', '', text_clean)
    text_clean = re.sub(r'\b\d+\s*(?:units?|pcs?|pieces?)\b', '', text_clean)
    # Remove non-alphabetic characters except spaces and hyphens
    text_clean = re.sub(r'[^a-z\s-]', '', text_clean)
    words = text_clean.split()
    # Strip common product adjectives/specifiers at the end
    while words and words[-1] in ('pro', 'amoled', 'lite', 'max', 'generation', 'gen', 'plus', 'active', 'smart', 'new', 'classic', 'luxury'):
        words.pop()
    if not words:
        return ""
    return singularize_word(words[-1])


def detect_priority_from_text(text: str) -> Optional[str]:
    """Detects priority mode from natural language text."""
    lower = text.lower()
    for priority_key, phrases in PRIORITY_KEYWORDS.items():
        for phrase in phrases:
            if re.search(r'\b' + re.escape(phrase) + r'\b', lower):
                return priority_key
    return None


def detect_affirmation(text: str) -> Optional[bool]:
    """Detects user confirmation (True for Yes/Add, False for No/Skip, None if ambiguous)."""
    lower = text.lower().strip()
    if lower in ("y", "yes", "add", "ok", "okay", "sure", "confirm", "proceed", "yes please"):
        return True
    if lower in ("n", "no", "skip", "cancel", "pass", "no thanks", "dont add"):
        return True is False
    for aff in AFFIRMATIVE_KEYWORDS:
        if re.search(r'\b' + re.escape(aff) + r'\b', lower):
            return True
    for neg in NEGATIVE_KEYWORDS:
        if re.search(r'\b' + re.escape(neg) + r'\b', lower):
            return False
    return None


def extract_budget_from_text(text: str) -> Optional[int]:
    """Extracts budget limit in INR from text."""
    lower = text.lower()
    # Matches: under 5000, below 5000, under rs 5000, under ₹5000, 5000 budget, max 5000
    patterns = [
        r'(?:under|below|max|budget|within|upto|up to|less than)\s*(?:rs\.?|inr|₹)?\s*([0-9]+(?:,[0-9]+)*)',
        r'(?:rs\.?|inr|₹)\s*([0-9]+(?:,[0-9]+)*)\s*(?:budget|max|limit)?',
        r'([0-9]+(?:,[0-9]+)*)\s*(?:rs|rupees|inr|budget)'
    ]
    for pattern in patterns:
        m = re.search(pattern, lower)
        if m:
            num_str = m.group(1).replace(",", "")
            try:
                val = int(num_str)
                if val > 50:  # Reasonable budget threshold
                    return val
            except ValueError:
                pass
    return None


def extract_size_from_text(text: str) -> Optional[Any]:
    """Extracts size specification from text (numeric shoe size or S/M/L/XL/waist)."""
    lower = text.lower()
    m_size = re.search(r'\bsize\s*(?:is|=|:)?\s*([0-9]{1,2}|s|m|l|xl|xxl)\b', lower)
    if m_size:
        val = m_size.group(1).upper()
        return int(val) if val.isdigit() else val

    m_waist = re.search(r'\bwaist\s*(?:is|=|:)?\s*([0-9]{2})\b', lower)
    if m_waist:
        return int(m_waist.group(1))

    # Single standalone numbers if context implies size
    if re.search(r'\b(size|number|no\.?)\s*([0-9]+)\b', lower):
        m = re.search(r'\b(size|number|no\.?)\s*([0-9]+)\b', lower)
        return int(m.group(2))
    return None


CATALOG_COLORS = [
    # Multi-word colors first (matched in priority order)
    "Space Grey", "Rose Gold", "Carbon Fiber", "Ocean Blue", "Navy Blue", 
    "Olive Green", "Dark Blue", "Light Blue", "Light Wash Blue", "Dark Indigo",
    # Single-word colors
    "Black", "White", "Blue", "Red", "Green", "Grey", "Gray", "Silver",
    "Gold", "Navy", "Pink", "Purple", "Brown", "Olive", "Charcoal",
    "Orange", "Tan", "Yellow", "Tortoise", "Clear", "RGB"
]


def extract_color_from_text(text: str) -> Optional[str]:
    """Extracts catalog color names from text."""
    lower = text.lower()
    for col in CATALOG_COLORS:
        pattern = r'\b' + re.escape(col.lower()) + r'\b'
        if re.search(pattern, lower):
            if col == "Gray":
                return "Grey"
            return col
    return None


def extract_quantity_from_text(text: str) -> Optional[int]:
    """Detects and extracts numeric quantity from natural language text."""
    lower = text.lower()
    # 5. Standalone quantity matching (e.g. "3", "3 each", "3 please")
    m_standalone = re.search(r'^\s*([0-9]+)\s*(?:each|please|thanks|thank\s+you)?\s*$', lower)
    if m_standalone:
        val = int(m_standalone.group(1))
        if val <= 100:
            return val

    # 1. Matches "quantity 2", "qty 2", "quantity: 2", "qty: 2"
    m = re.search(r'\b(?:quantity|qty)\s*(?:is|=|:)?\s*([0-9]+)\b', lower)
    if m:
        return int(m.group(1))

    # 2. Matches "2 pairs", "2 pairs of", "2 units", "2 pieces", "2 pcs", "2 packs"
    m_unit = re.search(r'\b([0-9]+)\s*(?:pairs?(?:\s+of)?|pcs?|pieces?|items?|units?|packs?)\b', lower)
    if m_unit:
        return int(m_unit.group(1))

    # 3. Matches "I need 3", "I want 3", "get me 3", "buy 3", "give me 3"
    # where the number is at the end or followed only by optional thanks/please/each
    m_verb_num = re.search(
        r'\b(need|want|get|buy|give|qty|quantity|have)\s+(?:me\s+|the\s+|a\s+|an\s+)?([0-9]+)\s*(?:each|please|thanks|thank\s+you)?\s*$',
        lower
    )
    if m_verb_num:
        return int(m_verb_num.group(2))

    # 4. Generic "2 <product_noun>" like "2 smartwatches", "2 laptops", "3 running shoes", "4 shirts"
    matches = re.finditer(r'\b([0-9]+)\s+([a-zA-Z-]+)\b', lower)
    for match in matches:
        num = int(match.group(1))
        word = match.group(2)
        # Exclude common non-product descriptors
        if word in ("rs", "inr", "rupees", "bucks", "size", "waist", "budget", "priority", "under", "below", "above", "each", "pack", "pairs", "pcs", "pieces", "items", "units", "inch", "inches", "gb", "tb", "cm", "mm", "hz", "v", "w", "ah", "mah", "ml", "g", "kg", "in", "oz", "liter", "liters", "l", "m", "meter", "meters", "yard", "yards", "ft", "feet", "pair", "packs"):
            continue
        # Exclude color names
        if word in [
            "black", "white", "blue", "red", "green", "grey", "gray", "silver", "gold", "maroon",
            "navy", "pink", "purple", "brown", "olive", "beige", "charcoal", "dark indigo", "light wash blue",
            "washed black", "neon lime", "neon pink"
        ]:
            continue
        # Exclude priority words
        is_prio = False
        for prio_list in PRIORITY_KEYWORDS.values():
            if any(p in word for p in prio_list):
                is_prio = True
                break
        if is_prio:
            continue
        return num
    return None


def extract_product_query_from_text(text: str) -> Optional[str]:
    """Extracts the primary product intent query string, stripped of conversational and constraint phrases."""
    cleaned = text

    # 1. Strip budget expressions
    cleaned = re.sub(r'(?:under|below|max|budget|within|upto|up to|less than|around|approx|about)\s*(?:rs\.?|inr|₹)?\s*[0-9]+(?:,[0-9]+)*', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'(?:rs\.?|inr|₹)\s*[0-9]+(?:,[0-9]+)*', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'[0-9]+(?:,[0-9]+)*\s*(?:rs|rupees|inr|bucks)', '', cleaned, flags=re.IGNORECASE)

    # 2. Strip size / waist expressions
    cleaned = re.sub(r'\bsize\s*(?:is|=|:)?\s*[0-9a-zA-Z]+\b', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\bwaist\s*(?:is|=|:)?\s*[0-9]+\b', '', cleaned, flags=re.IGNORECASE)

    # 3. Strip quantity phrases: "quantity 2", "qty 2", "2 pairs of", "2 pairs", "2 units", "2 pcs", "2 pieces", etc.
    cleaned = re.sub(r'\b(?:quantity|qty)\s*(?:is|=|:)?\s*[0-9]+\b', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\b[0-9]+\s*(?:pairs?(?:\s+of)?|pcs?|pieces?|items?|units?|packs?)\b', '', cleaned, flags=re.IGNORECASE)
    # Strip leading counts before nouns (e.g. "2 smartwatches" -> "smartwatches")
    cleaned = re.sub(r'\b[0-9]+\s+(?=[a-zA-Z-])', '', cleaned, flags=re.IGNORECASE)

    # 4. Strip color names
    for col in [
        "black", "white", "blue", "red", "green", "grey", "gray", "silver",
        "gold", "maroon", "navy", "pink", "purple", "brown", "olive", "beige",
        "charcoal", "dark indigo", "light wash blue", "washed black", "neon lime", "neon pink",
        "lime", "indigo", "orange", "tan"
    ]:
        cleaned = re.sub(r'\b' + re.escape(col) + r'\b', '', cleaned, flags=re.IGNORECASE)

    # 5. Strip priority phrases
    for phrases in PRIORITY_KEYWORDS.values():
        for phrase in phrases:
            cleaned = re.sub(r'\b' + re.escape(phrase) + r'\b', '', cleaned, flags=re.IGNORECASE)

    # 6. Combined Single-Pass Prefix Pattern matching conversational search templates
    prefix_pattern = (
        r'\b(?:'
        r'(?:can\s+you\s+)?(?:find|show|get|search(?:\s+for)?)\s*(?:me\s+)?'
        r'|'
        r'(?:i[\'’]m|i\s+am|i\s+)?\s*(?:want|need|would\s+like|looking\s+for|searching\s+for)\s*(?:to\s+)?(?:buy|purchase|find|get|show|search(?:\s+for)?|place\s+order\s+for)?'
        r')\s*'
        r'(?:a\s+pair\s+of|the|a|an|some)?\b'
    )
    cleaned = re.sub(prefix_pattern, ' ', cleaned, flags=re.IGNORECASE)

    # 7. Strip conversational clauses, instructions, and other prefixes
    conversational_phrases = [
        r'\b(?:my\s+)?priority\s*(?:is|=|:)?\b',
        r'\b(?:i\s+)?(?:prefer|would\s+prefer|preference\s+is)\b',
        r'\b(?:with\s+a\s+budget\s+of|budget\s+of|budget\s+is)\b',
        r'\b(?:with|in)\b',
        r'\b(?:please|for\s+me|for\s+myself|thanks|thank\s+you|each|check)\b'
    ]
    for cp in conversational_phrases:
        cleaned = re.sub(cp, '', cleaned, flags=re.IGNORECASE)

    # 8. Strip punctuation except hyphen, plus, and percent signs
    cleaned = re.sub(r'[^\w\s\-\+%]', ' ', cleaned).strip()

    # 9. Clean up leading/trailing residue particles and articles
    cleaned = re.sub(r'^(?:to|for|me|a|an|the|some|please|buy|purchase|find|show|get|search)\s+', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s+(?:to|for|me|a|an|the|some|please)$', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    if cleaned.isdigit():
        return None
    return cleaned or None


def detect_search_intent(text: str) -> bool:
    """Detects explicit user search intent keywords in natural language text."""
    lower = text.lower()
    intent_keywords = [
        "find", "search", "looking for", "want", "need", 
        "show me", "can you find", "buy", "purchase", "show"
    ]
    for kw in intent_keywords:
        if kw in lower:
            return True
    return False


def queries_overlap(q1: str, q2: str) -> bool:
    """Checks if two queries share meaningful non-noise words."""
    if not q1 or not q2:
        return False
    w1 = set(singularize_word(w) for w in q1.lower().split())
    w2 = set(singularize_word(w) for w in q2.lower().split())
    # Noise/filler words to ignore in query overlap checks
    noise = {"a", "an", "the", "with", "for", "of", "and", "or", "in", "to", "at", "by", "from", "me", "i", "need", "want", "find"}
    w1 = w1 - noise
    w2 = w2 - noise
    return not w1.isdisjoint(w2)


def is_footwear_or_apparel(category: str) -> bool:
    if not category:
        return False
    cat_lower = category.lower()
    return "footwear" in cat_lower or "apparel" in cat_lower


def is_meaningful_sizes(sizes) -> bool:
    if not sizes or not isinstance(sizes, list):
        return False
    if len(sizes) <= 1:
        return False
    # If the options are "Free Size", "Adjustable", etc.
    noise = {"free size", "adjustable", "one-size", "trim", "trim-to-fit"}
    for s in sizes:
        if str(s).lower() in noise or any(n in str(s).lower() for n in noise):
            return False
    return True


def is_meaningful_colors(colors) -> bool:
    if not colors or not isinstance(colors, list):
        return False
    if len(colors) <= 1:
        return False
    return True


def extract_standalone_size(text: str) -> Optional[Any]:
    clean = text.strip()
    if clean.isdigit():
        return int(clean)
    if clean.upper() in {"XS", "S", "M", "L", "XL", "XXL", "XXXL"}:
        return clean.upper()
    return None


class BuyerAgent:
    """
    Agent 1 Service responsible for multi-turn shopping orchestration:
    - Requirements extraction & missing slot identification
    - Tool invocation of Step 4A Semantic Search & Step 4B Scoring Engine
    - Cart addition & Agent 2 recommendation handoff
    - Final cart summary & purchase readiness confirmation
    """

    def __init__(self):
        # In-memory session store: session_id -> state dict
        self._sessions: Dict[str, Dict[str, Any]] = {}

    def get_session(self, session_id: Optional[str]) -> Tuple[str, Dict[str, Any]]:
        """Retrieves or creates a session state."""
        sid = session_id.strip() if session_id and session_id.strip() else str(uuid.uuid4())
        if sid not in self._sessions:
            self._sessions[sid] = {
                "session_id": sid,
                "current_state": "COLLECTING_REQUIREMENTS",
                "requirements": {
                    "product_query": None,
                    "budget": None,
                    "quantity": 1,
                    "size": None,
                    "color": None,
                    "category": None,
                    "attributes": {},
                    "priority": None
                },
                "missing_requirements": [],
                "winning_offer": None,
                "agent2_recommendation": None,
                "agent2_recommendation_history": None,
                "cart_contents": [],
                "cart_total": 0,
                "last_bot_question": None,
                "original_user_request": None,
                "conversation_history": [],
                "user_consent_complementary": None,
                "scored_offers": [],
                "trust_layer_results": {
                    "primary": None,
                    "complementary": None,
                    "cart_consistency": None
                }
            }
        return sid, self._sessions[sid]

    def _enrich_cart_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        merchant = item["merchant"].strip().lower()
        p_id = item["p_id"]
        prod = dal.get_product(merchant, p_id)
        if prod:
            stock = dal.get_stock(merchant, p_id) or 0
            enriched = {
                "p_id": prod["p_id"],
                "p_name": prod["p_name"],
                "merchant": merchant,
                "price": prod["price"],
                "rating": prod.get("rating", 0.0),
                "quantity": item.get("quantity", 1),
                "available_stock": stock,
                "category": prod.get("category"),
                "attributes": prod.get("attributes", {})
            }
            if item.get("color"):
                enriched["color"] = item["color"]
            if item.get("size"):
                enriched["size"] = item["size"]
            return enriched
        return item

    def _get_candidates(self, q: str) -> List[Dict[str, Any]]:
        normalized_q = singularize_query(q) if q else ""
        if not normalized_q:
            return []
        
        # 1. Semantic Search across merchants using normalized query
        hits = []
        for merchant in ("shopnest", "cartwave"):
            hits.extend(semantic_search_engine.search(merchant, normalized_q, top_k=8))
            
        if not hits:
            return []

        valid_hits = [h for h in hits if h.get("similarity_score", 0.0) >= 0.40]
        if not valid_hits:
            return []

        q_clean = re.sub(r'[^a-zA-Z0-9\s]', ' ', normalized_q).lower()
        q_words = set(singularize_word(w) for w in q_clean.split() if len(w) > 1)
        noise = {
            "a", "an", "the", "under", "with", "from", "each", "need", "want", 
            "find", "get", "buy", "looking", "for", "please", "priority", 
            "highest", "rated", "cheapest", "best", "balance", "over", 
            "above", "below", "price", "budget", "size", "color", "i", "would", "like"
        }
        meaningful_q_words = q_words - noise
        if not meaningful_q_words:
            meaningful_q_words = q_words

        accessory_intent_words = {"insole", "sock", "strap", "protector", "coaster", "case", "cable", "stand", "accessory"}
        user_wants_accessory = bool(meaningful_q_words & accessory_intent_words)
        
        bag_synonyms = {"bag", "backpack", "duffel"}
        has_generic_bag_intent = ("bag" in meaningful_q_words) and not any(k in meaningful_q_words for k in ("duffel", "sports", "backpack", "laptop"))

        scored = []
        seen = set()
        for h in valid_hits:
            pid = h["p_id"]
            merchant = h["merchant"]
            key = (merchant, pid)
            if key in seen:
                continue
            seen.add(key)
            
            prod = dal.get_product(merchant, pid)
            if not prod:
                continue
                
            p_name = prod.get("p_name", "").lower()
            p_name_words = set(singularize_word(w) for w in re.sub(r'[^a-zA-Z0-9]', ' ', p_name).split())
            category = prod.get("category", "")
            is_accessory_cat = "accessories" in category.lower()
            
            # Check keyword overlap
            overlap = bool(meaningful_q_words & p_name_words)
            if has_generic_bag_intent and (p_name_words & bag_synonyms):
                overlap = True
                
            acc_specific_match = user_wants_accessory and bool(meaningful_q_words & accessory_intent_words & p_name_words)
            
            # If user did NOT ask for accessory, but product is accessory (e.g. insole when asking for shoe), reject accessory
            if not user_wants_accessory and is_accessory_cat and not (meaningful_q_words & (p_name_words - {"shoe", "watch", "earbud", "laptop", "coffee"})):
                if not (has_generic_bag_intent and pid == "BACK_001"):
                    overlap = False
                
            if user_wants_accessory and not is_accessory_cat and not acc_specific_match:
                overlap = False
                
            is_primary = not is_accessory_cat
            scored.append((acc_specific_match, overlap, is_primary, h["similarity_score"], prod))
            
        scored.sort(key=lambda x: (x[0], x[1], x[2], x[3]), reverse=True)
        
        if any(s[1] for s in scored):
            candidates = [s[4] for s in scored if s[1]]
        else:
            candidates = [s[4] for s in scored[:4]]
            
        return candidates

    def extract_slots(self, text: str, current_reqs: Dict[str, Any], last_question: Optional[str] = None) -> Dict[str, Any]:
        """Extracts shopping requirement slots from user text, strictly preserving existing filled slots."""
        reqs = dict(current_reqs)
        if "attributes" not in reqs:
            reqs["attributes"] = {}

        # 1. Parse natural query from text
        detected_query = extract_product_query_from_text(text)

        # Detect whether the message is a new product search
        is_new_search = False
        old_query = reqs.get("product_query")
        
        if old_query and detected_query:
            has_intent = detect_search_intent(text)
            overlap = queries_overlap(old_query, detected_query)
            
            # Avoid treating conversational confirmations or common responses as new search queries
            is_noise_query = detected_query.lower() in {
                "yes", "no", "maybe", "sure", "ok", "okay", "yup", "yep", "nope", "nah",
                "hello", "hi", "hey", "please", "thanks", "thank you",
                "xs", "s", "m", "l", "xl", "xxl", "free size", "one size"
            }

            if last_question in ("size", "color", "priority") and not has_intent:
                is_noise_query = True
            
            # If the last question was dynamic attributes (e.g. levels), keep it as noise too if not explicit search intent
            if last_question and last_question not in ("product_query", "priority", "budget", "quantity", "size", "color") and not has_intent:
                is_noise_query = True

            if not is_noise_query:
                if has_intent or not overlap:
                    is_new_search = True

        # 2. If it is a new search, replace the old product query and clear stale product-specific requirements.
        if is_new_search:
            reqs = {
                "product_query": detected_query,
                "budget": None,
                "quantity": 1,
                "size": None,
                "color": None,
                "category": None,
                "attributes": {},
                "priority": None
            }
        elif not reqs.get("product_query") and detected_query:
            reqs["product_query"] = detected_query

        # 3. Detect explicit values for known generic slots: budget, quantity, size, color, priority.
        detected_prio = detect_priority_from_text(text)
        detected_budget = extract_budget_from_text(text)
        detected_size = extract_size_from_text(text)
        detected_color = extract_color_from_text(text)
        detected_qty = extract_quantity_from_text(text)

        # Retrieve candidates for the current active product query to support dynamic attributes and colors matching
        q = reqs.get("product_query")
        candidates = self._get_candidates(q) if q else []

        # 4. Map the response to the currently requested slot whenever the conversation context is unambiguous
        if last_question == "size":
            standalone_size = extract_standalone_size(text)
            if standalone_size is not None:
                detected_size = standalone_size
                # Protect budget and quantity from matching the standalone size
                detected_qty = None
                detected_budget = None

        elif last_question == "color" and detected_color is None:
            # Check if user mentioned any candidate colors
            for p in candidates:
                attrs = p.get("attributes", {})
                colors = attrs.get("colors") or attrs.get("color")
                if colors and isinstance(colors, list):
                    for col in colors:
                        if re.search(r'\b' + re.escape(col.lower()) + r'\b', text.lower()):
                            detected_color = col
                            break
                    if detected_color is not None:
                        break

        elif last_question == "priority" and detected_prio is None:
            clean = text.lower().strip()
            # Map standalone priority indicators
            if clean in ("1", "first", "cheapest", "cheap", "price", "cheapest option"):
                detected_prio = "cheapest"
            elif clean in ("2", "second", "highest rated", "top rated", "rating", "highest-rated option"):
                detected_prio = "highest_rated"
            elif clean in ("3", "third", "best balance", "balance", "value", "best balance option"):
                detected_prio = "best_balance"

        elif last_question == "budget" and detected_budget is None:
            clean = text.strip()
            if clean.isdigit():
                detected_budget = int(clean)

        elif last_question == "quantity" and detected_qty is None:
            clean = text.strip()
            if clean.isdigit():
                detected_qty = int(clean)

        elif last_question and last_question not in ("product_query", "priority", "budget", "quantity", "size", "color"):
            # Dynamic attribute question (e.g. levels)
            # Check if any candidate's dynamic attribute value matches the user text
            dynamic_val = None
            for p in candidates:
                attrs = p.get("attributes", {})
                vals = attrs.get(last_question)
                if vals and isinstance(vals, list):
                    for val in vals:
                        if re.search(r'\b' + re.escape(str(val).lower()) + r'\b', text.lower()):
                            dynamic_val = val
                            break
                    if dynamic_val is not None:
                        break
            if dynamic_val is not None:
                reqs["attributes"][last_question] = dynamic_val

        # 5. Detect and save dynamic catalog attributes from candidates (like levels)
        for p in candidates:
            attrs = p.get("attributes", {})
            for k, v in attrs.items():
                if k not in ("sizes", "size", "waist", "colors", "color") and isinstance(v, list):
                    if k in ("description", "category", "p_id", "merchant", "p_name", "price", "rating"):
                        continue
                    for option in v:
                        if re.search(r'\b' + re.escape(str(option).lower()) + r'\b', text.lower()):
                            reqs["attributes"][k] = option
                            break

        # Merge extracted values into requirements
        if detected_prio:
            reqs["priority"] = detected_prio
        if detected_budget is not None:
            reqs["budget"] = detected_budget
        if detected_size is not None:
            reqs["size"] = detected_size
        if detected_color is not None:
            reqs["color"] = detected_color
        if detected_qty is not None:
            reqs["quantity"] = detected_qty

        # Default quantity to 1 if not set
        if not reqs.get("quantity"):
            reqs["quantity"] = 1

        return reqs

    def compute_missing_slots(self, reqs: Dict[str, Any]) -> List[str]:
        """Determines which required slots are still missing before semantic search & scoring."""
        missing = []
        q = (reqs.get("product_query") or "").lower()

        # Product query is fundamental
        if not q:
            missing.append("product_query")
            return missing

        # Determine size and color and dynamic sensitivity dynamically from candidates
        candidates = self._get_candidates(q)
        if not candidates:
            return []

        # Find all list-valued selectable attributes dynamically across candidates
        has_size_option = False
        has_color_option = False
        custom_list_attrs = {}

        is_loop_band_query = "band" in q or "loop" in q or "resistance" in q
        
        all_sizes = set()
        all_colors = []

        for p in candidates:
            attrs = p.get("attributes", {})
            
            # 1. Size mapping (sizes, size, waist)
            sizes = attrs.get("sizes") or attrs.get("size") or attrs.get("waist")
            if sizes and isinstance(sizes, list):
                if len(sizes) > 1 and is_meaningful_sizes(sizes):
                    all_sizes.update(str(s) for s in sizes)
            
            # 2. Color mapping (colors, color)
            colors = attrs.get("colors") or attrs.get("color")
            if colors and isinstance(colors, list):
                if len(colors) > 1 and is_meaningful_colors(colors):
                    for c in colors:
                        if c not in all_colors:
                            all_colors.append(c)
            
            # 3. Custom dynamic list attributes
            for k, v in attrs.items():
                if k not in ("sizes", "size", "waist", "colors", "color") and isinstance(v, list) and len(v) > 1:
                    if k in ("description", "category", "p_id", "merchant", "p_name", "price", "rating"):
                        continue
                    if k not in custom_list_attrs:
                        custom_list_attrs[k] = set()
                    custom_list_attrs[k].update(str(x) for x in v)

        if len(all_sizes) > 1:
            has_size_option = True
            
        if len(all_colors) > 1:
            has_color_option = True

        if has_size_option and reqs.get("size") is None:
            missing.append("size")
            
        if has_color_option and reqs.get("color") is None:
            missing.append("color")

        for attr_name, options in custom_list_attrs.items():
            is_selectable = False
            if is_loop_band_query and attr_name == "levels":
                is_selectable = True
            elif attr_name in ("levels", "resistance_levels"):
                is_selectable = True

            if is_selectable and len(options) > 1:
                supplied_attrs = reqs.get("attributes", {})
                if supplied_attrs.get(attr_name) is None:
                    missing.append(attr_name)

        if not reqs.get("priority"):
            missing.append("priority")

        return missing

    def process_message(self, request: Agent1ChatRequest) -> Agent1ChatResponse:
        """
        Main orchestration entry point: executes deterministic state transitions,
        invokes Step 4A, Step 4B, and Agent 2 tools, and generates user-facing responses.
        """
        sid, session = self.get_session(request.session_id)

        # Allow caller to override state if passed in request
        if request.state and isinstance(request.state, dict):
            session.update(request.state)

        user_text = request.message.strip()
        
        # Store user message in conversation history
        session.setdefault("conversation_history", []).append({"role": "user", "message": user_text})

        # Store original user request on first meaningful input
        if session.get("original_user_request") is None:
            detected_q = extract_product_query_from_text(user_text)
            if detected_q and detected_q.lower().strip() not in {"yes", "no", "maybe", "sure", "ok", "okay", "yup", "yep", "nope", "nah", "hello", "hi", "hey"}:
                session["original_user_request"] = user_text

        state = session.get("current_state", "COLLECTING_REQUIREMENTS")
        reqs = session.get("requirements", {})

        # Fallback to COLLECTING_REQUIREMENTS if the user specifies a constraint in confirmation/cart states
        if state in ("AWAITING_MAIN_CART_CONFIRMATION", "AWAITING_COMPLEMENTARY_CART_CONFIRMATION", "AWAITING_ORDER_CONFIRMATION") and detect_affirmation(user_text) is None:
            has_constraint = (
                extract_color_from_text(user_text) is not None or
                extract_size_from_text(user_text) is not None or
                extract_budget_from_text(user_text) is not None or
                detect_priority_from_text(user_text) is not None or
                extract_quantity_from_text(user_text) is not None or
                extract_product_query_from_text(user_text) is not None
            )
            if has_constraint:
                state = "COLLECTING_REQUIREMENTS"
                session["current_state"] = "COLLECTING_REQUIREMENTS"

        # =====================================================================
        # STATE 1: COLLECTING_REQUIREMENTS
        # =====================================================================
        if state == "COLLECTING_REQUIREMENTS":
            # Extract slots from user text
            old_q = reqs.get("product_query")
            reqs = self.extract_slots(user_text, reqs, session.get("last_bot_question"))
            new_q = reqs.get("product_query")
            
            # Reset winner if query changed to avoid carrying over previous winning product/recommendation
            if old_q and new_q and old_q != new_q:
                session["winning_offer"] = None
                session["agent2_recommendation"] = None

            session["requirements"] = reqs

            missing = self.compute_missing_slots(reqs)
            session["missing_requirements"] = missing

            if "product_query" in missing:
                reply = "Hello! What product are you looking to buy today?"
                return self._build_response(session, reply, "COLLECTING_REQUIREMENTS", "ASK_CLARIFICATION")

            if "size" in missing:
                session["last_bot_question"] = "size"
                prod_name = reqs.get("product_query", "item").lower()
                if "jeans" in prod_name or "pant" in prod_name or "belt" in prod_name:
                    reply = "What waist size do you need?"
                else:
                    reply = "What size do you need?"
                return self._build_response(session, reply, "COLLECTING_REQUIREMENTS", "ASK_CLARIFICATION")

            if "color" in missing:
                session["last_bot_question"] = "color"
                prod_name = reqs.get("product_query", "item")
                candidates = self._get_candidates(reqs.get("product_query"))
                available_colors = []
                for p in candidates:
                    attrs = p.get("attributes", {})
                    colors = attrs.get("colors") or attrs.get("color")
                    if colors and isinstance(colors, list):
                        for c in colors:
                            if c not in available_colors:
                                available_colors.append(c)
                if available_colors:
                    if len(available_colors) > 1:
                        color_str = ", ".join(available_colors[:-1]) + f", or {available_colors[-1]}"
                    else:
                        color_str = available_colors[0]
                    reply = f"Which color would you prefer — {color_str}?"
                else:
                    reply = f"Which color would you prefer for your {prod_name}?"
                return self._build_response(session, reply, "COLLECTING_REQUIREMENTS", "ASK_CLARIFICATION")

            # Dynamic custom list-valued attributes prompting (e.g. levels)
            dynamic_attr = None
            for m_item in missing:
                if m_item not in ("product_query", "size", "color", "priority"):
                    dynamic_attr = m_item
                    break

            if dynamic_attr:
                session["last_bot_question"] = dynamic_attr
                prod_name = reqs.get("product_query", "item")
                candidates = self._get_candidates(prod_name)
                available_opts = []
                for p in candidates:
                    attrs = p.get("attributes", {})
                    vals = attrs.get(dynamic_attr)
                    if vals and isinstance(vals, list):
                        for v in vals:
                            if v not in available_opts:
                                available_opts.append(v)
                if available_opts:
                    if len(available_opts) > 1:
                        opts_str = ", ".join(str(o) for o in available_opts[:-1]) + f", or {available_opts[-1]}"
                    else:
                        opts_str = str(available_opts[0])
                    attr_label = dynamic_attr.replace("_", " ")
                    if attr_label.endswith("s") and len(attr_label) > 3:
                        attr_label = attr_label[:-1]
                    reply = f"Which {attr_label} would you prefer — {opts_str}?"
                else:
                    reply = f"Which {dynamic_attr} would you prefer for your {prod_name}?"
                return self._build_response(session, reply, "COLLECTING_REQUIREMENTS", "ASK_CLARIFICATION")

            if "priority" in missing:
                session["last_bot_question"] = "priority"
                reply = "What matters most to you for this purchase: the **cheapest** option, the **highest-rated** option, or the **best balance** between price and rating?"
                return self._build_response(session, reply, "COLLECTING_REQUIREMENTS", "ASK_CLARIFICATION")

            # All slots ready -> Trigger Step 4A & Step 4B
            return self._execute_search_and_scoring(session)

        # =====================================================================
        # STATE 2: AWAITING_MAIN_CART_CONFIRMATION
        # =====================================================================
        elif state == "AWAITING_MAIN_CART_CONFIRMATION":
            aff = detect_affirmation(user_text)
            winner = session.get("winning_offer")

            if aff is True:
                # User confirmed adding main product to cart
                if not winner:
                    return self._execute_search_and_scoring(session)

                quantity = reqs.get("quantity", 1)
                trust_res = TrustLayer.validate_primary_product(winner, quantity)
                session.setdefault("trust_layer_results", {})["primary"] = trust_res
                if not trust_res["approved"]:
                    reason = trust_res.get("reason", "VALIDATION_FAILED")
                    reply = f"❌ **Validation Rejected by Trust Layer:** {reason}. We cannot add this product to your cart."
                    return self._build_response(session, reply, "COLLECTING_REQUIREMENTS", "ASK_CLARIFICATION")

                cart_item = {
                    "p_id": winner["p_id"],
                    "p_name": winner["p_name"],
                    "merchant": winner["merchant"],
                    "price": winner["price"],
                    "rating": winner.get("rating", 0.0),
                    "quantity": quantity,
                    "color": reqs.get("color") or winner.get("color"),
                    "size": reqs.get("size") or winner.get("size")
                }
                session["cart_contents"].append(self._enrich_cart_item(cart_item))
                session["cart_total"] = sum(i["price"] * i["quantity"] for i in session["cart_contents"])

                # Trigger Step 5: Agent 2 Sales Improvement Agent
                try:
                    agent2_resp = agent2_service.get_recommendation(
                        merchant=winner["merchant"],
                        selected_product_id=winner["p_id"],
                        current_cart_items=[winner["p_id"]]
                    )
                    session["agent2_recommendation"] = agent2_resp.dict()
                    session["agent2_recommendation_history"] = agent2_resp.dict()
                except Exception as e:
                    agent2_resp = None
                    session["agent2_recommendation"] = None

                if agent2_resp and agent2_resp.recommendation_available:
                    session["current_state"] = "AWAITING_COMPLEMENTARY_CART_CONFIRMATION"
                    # Use Agent 2 recommendation message or structured fallback
                    rec_pname = agent2_resp.recommended_product_name
                    rec_price = agent2_resp.price
                    msg = agent2_resp.recommendation_message
                    if not msg:
                        msg = f"Customers who bought {winner['p_name']} also frequently bought {rec_pname} (₹{rec_price:,}). Would you like to add it to your order?"

                    reply = f"Added **{winner['p_name']}** to your cart!\n\n{msg}"
                    return self._build_response(session, reply, "AWAITING_COMPLEMENTARY_CART_CONFIRMATION", "ASK_COMPLEMENTARY_CONFIRMATION")
                else:
                    # No complementary recommendation available -> proceed directly to final cart
                    return self._present_final_cart(session)

            elif aff is False:
                # User rejected main product
                session["current_state"] = "COLLECTING_REQUIREMENTS"
                pname = winner["p_name"] if winner else "this product"
                reply = f"Understood, I won't add {pname} to your cart. Is there anything else you're looking for?"
                return self._build_response(session, reply, "COLLECTING_REQUIREMENTS", "ASK_CLARIFICATION")

            else:
                # Ambiguous answer -> re-prompt
                pname = winner["p_name"] if winner else "the selected product"
                reply = f"Would you like to add **{pname}** to your cart? Please reply with **Yes** or **No**."
                return self._build_response(session, reply, "AWAITING_MAIN_CART_CONFIRMATION", "ASK_MAIN_CART_CONFIRMATION")

        # =====================================================================
        # STATE 3: AWAITING_COMPLEMENTARY_CART_CONFIRMATION
        # =====================================================================
        elif state == "AWAITING_COMPLEMENTARY_CART_CONFIRMATION":
            aff = detect_affirmation(user_text)
            rec = session.get("agent2_recommendation") or session.get("agent2_recommendation_history")

            if aff is True:
                session["user_consent_complementary"] = "YES"
                if rec and rec.get("recommendation_available"):
                    trust_res = TrustLayer.validate_complementary_product(rec, user_consent=True)
                    session.setdefault("trust_layer_results", {})["complementary"] = trust_res
                    if trust_res["approved"]:
                        comp_item = {
                            "p_id": rec["recommended_product_id"],
                            "p_name": rec["recommended_product_name"],
                            "merchant": rec["merchant"],
                            "price": rec["price"],
                            "rating": rec.get("rating", 0.0),
                            "quantity": 1
                        }
                        session["cart_contents"].append(self._enrich_cart_item(comp_item))
                        session["cart_total"] = sum(i["price"] * i["quantity"] for i in session["cart_contents"])
                # Clear the recommendation from state
                session["agent2_recommendation"] = None
                return self._present_final_cart(session)
            elif aff is False:
                session["user_consent_complementary"] = "NO"
                if rec and rec.get("recommendation_available"):
                    trust_res = TrustLayer.validate_complementary_product(rec, user_consent=False)
                    session.setdefault("trust_layer_results", {})["complementary"] = trust_res
                # Clear the recommendation from state
                session["agent2_recommendation"] = None
                return self._present_final_cart(session)
            else:
                # Ambiguous response -> re-prompt for complementary decision
                rec_pname = rec.get("recommended_product_name") if rec else "the recommended product"
                rec_price = rec.get("price") if rec else 0
                reply = f"Would you like to add **{rec_pname}** (₹{rec_price:,}) to your cart? Please reply with **Yes** or **No**."
                return self._build_response(session, reply, "AWAITING_COMPLEMENTARY_CART_CONFIRMATION", "ASK_COMPLEMENTARY_CONFIRMATION")

        # =====================================================================
        # STATE 4: AWAITING_ORDER_CONFIRMATION
        # =====================================================================
        elif state == "AWAITING_ORDER_CONFIRMATION":
            aff = detect_affirmation(user_text)

            if aff is True:
                cart = session.get("cart_contents", [])
                if not cart:
                    reply = "Your cart is empty. Please select a product first."
                    return self._build_response(session, reply, "COLLECTING_REQUIREMENTS", "ASK_CLARIFICATION")

                trust_res = TrustLayer.validate_cart_consistency(cart)
                session.setdefault("trust_layer_results", {})["cart_consistency"] = trust_res
                if not trust_res["approved"]:
                    reason = trust_res.get("reason", "CART_VALIDATION_FAILED")
                    reply = f"❌ **Checkout failed: Stopped by Trust Layer:** {reason}. Please review your cart contents."
                    return self._build_response(session, reply, "AWAITING_ORDER_CONFIRMATION", "CHECKOUT_FAILED")
                
                from backend.payment_service import create_razorpay_order_for_session
                try:
                    res = create_razorpay_order_for_session(session["session_id"])
                    razorpay_order_id = res["razorpay_order_id"]
                    total_amount = res["cart_total"]
                    
                    session["current_state"] = "AWAITING_PAYMENT"
                    
                    reply = (
                        f"💳 **Order validated by Trust Layer!**\n\n"
                        f"To complete your purchase of **₹{total_amount:,}**, please make a test payment.\n"
                        f"• Razorpay Order ID: `{razorpay_order_id}`\n\n"
                        f"Please complete checkout to place your order."
                    )
                    return self._build_response(session, reply, "AWAITING_PAYMENT", "AWAITING_PAYMENT")
                except Exception as e:
                    reply = f"❌ Payment initialization failed: {str(e)}"
                    return self._build_response(session, reply, "AWAITING_ORDER_CONFIRMATION", "CHECKOUT_FAILED")
            elif aff is False:
                reply = "Understood. Your cart has been saved and order placement was cancelled."
                return self._build_response(session, reply, "COLLECTING_REQUIREMENTS", "ORDER_CANCELLED")
            else:
                reply = "Would you like to place this order now? Please reply with **Yes** or **No**."
                return self._build_response(session, reply, "AWAITING_ORDER_CONFIRMATION", "ASK_ORDER_CONFIRMATION")

        # =====================================================================
        # STATE 4.5: AWAITING_PAYMENT
        # =====================================================================
        elif state == "AWAITING_PAYMENT":
            razorpay_order_id = session.get("razorpay_order_id", "Unknown")
            cart_total = session.get("cart_total", 0)
            reply = (
                f"We are still awaiting payment of **₹{cart_total:,}** for your order.\n"
                f"• Razorpay Order ID: `{razorpay_order_id}`\n\n"
                f"Please verify the payment using Swagger endpoint `/api/payment/verify`."
            )
            return self._build_response(session, reply, "AWAITING_PAYMENT", "AWAITING_PAYMENT")

        # =====================================================================
        # STATE 5: READY_FOR_PURCHASE (Terminal)
        # =====================================================================
        elif state == "READY_FOR_PURCHASE":
            reply = "Your order is already confirmed and ready for purchase."
            return self._build_response(session, reply, "READY_FOR_PURCHASE", "READY_FOR_PURCHASE")

        # Fallback
        session["current_state"] = "COLLECTING_REQUIREMENTS"
        return self._build_response(session, "How can I assist you with your shopping?", "COLLECTING_REQUIREMENTS", "ASK_CLARIFICATION")

    def _execute_search_and_scoring(self, session: Dict[str, Any]) -> Agent1ChatResponse:
        """
        Executes Step 4A (Semantic Search) and Step 4B (Deterministic Scoring)
        and formats the winning offer proposal.
        """
        reqs = session["requirements"]
        q = reqs.get("product_query", "")

        # Normalize plural product nouns to singular for search compatibility
        normalized_q = singularize_query(q) if q else ""

        # Retrieve candidates with accessory intent protection
        candidates = self._get_candidates(q)

        # Retrieve raw semantic search hits for fallback detection
        search_hits = semantic_search_engine.search("shopnest", normalized_q, top_k=5)
        if not search_hits:
            search_hits = semantic_search_engine.search("cartwave", normalized_q, top_k=5)

        if not candidates:
            if search_hits:
                # Accessories were found but filtered out since query represents primary product
                reply = f"I could not find an actual '{q}' in our catalog. It is currently unavailable."
            else:
                reply = f"I could not find any products matching '{q}'. Please try another search term."
            return self._build_response(session, reply, "COLLECTING_REQUIREMENTS", "ASK_CLARIFICATION")

        # 2. Step 4B: Deterministic Scoring & Merchant Comparison for top semantic matches
        winner = None
        for p in candidates:
            pid = p["p_id"]
            offers = scoring_engine.compare_merchants_for_product(
                p_id=pid,
                priority=reqs.get("priority", "best_balance"),
                budget=reqs.get("budget"),
                size=reqs.get("size"),
                color=reqs.get("color"),
                required_quantity=reqs.get("quantity", 1)
            )
            if offers:
                session["scored_offers"] = offers
                winner = offers[0]
                break

        if not winner:
            session["winning_offer"] = None
            reply = f"I found items matching '{q}', but none met all your constraints (budget: ₹{reqs.get('budget')}, size: {reqs.get('size')}, in-stock). Would you like to adjust your requirements?"
            return self._build_response(session, reply, "COLLECTING_REQUIREMENTS", "ASK_CLARIFICATION")

        if reqs.get("color"):
            winner["color"] = reqs.get("color")
        if reqs.get("size"):
            winner["size"] = reqs.get("size")
        session["winning_offer"] = winner
        session["current_state"] = "AWAITING_MAIN_CART_CONFIRMATION"

        p_name = winner["p_name"]
        merchant = winner["merchant"].title()
        price = winner["price"]
        rating = winner["rating"]
        prio_label = reqs.get("priority", "best balance").replace("_", " ").title()

        reply = (
            f"Based on your requirements, I found **{p_name}** at **{merchant}** for **₹{price:,}** "
            f"({rating}★ rating, matching your **{prio_label}** preference).\n\n"
            f"Would you like to add this to your cart?"
        )

        return self._build_response(session, reply, "AWAITING_MAIN_CART_CONFIRMATION", "ASK_MAIN_CART_CONFIRMATION")

    def _present_final_cart(self, session: Dict[str, Any]) -> Agent1ChatResponse:
        """Presents the final cart summary and prompts for order confirmation."""
        cart = session.get("cart_contents", [])
        total = sum(item["price"] * item["quantity"] for item in cart)
        session["cart_total"] = total
        session["current_state"] = "AWAITING_ORDER_CONFIRMATION"

        cart_lines = "\n".join(
            f"• {i['quantity']} × **{i['p_name']}** ({i['merchant'].title()}) — ₹{i['price'] * i['quantity']:,}"
            for i in cart
        )

        reply = (
            f"**Your Cart Summary:**\n"
            f"{cart_lines}\n\n"
            f"**Total Amount: ₹{total:,}**\n\n"
            f"Would you like to place the order?"
        )

        return self._build_response(session, reply, "AWAITING_ORDER_CONFIRMATION", "ASK_ORDER_CONFIRMATION")

    def _build_response(
        self,
        session: Dict[str, Any],
        reply: str,
        current_state: str,
        next_action: str,
        cart_override: Optional[List[Dict[str, Any]]] = None
    ) -> Agent1ChatResponse:
        """Builds the standardized Agent1ChatResponse object."""
        session["current_state"] = current_state
        session.setdefault("conversation_history", []).append({"role": "agent", "message": reply})
        
        return Agent1ChatResponse(
            reply=reply,
            current_state=current_state,
            next_action=next_action,
            requirements=session.get("requirements", {}),
            missing_requirements=session.get("missing_requirements", []),
            winning_offer=session.get("winning_offer"),
            agent2_recommendation=session.get("agent2_recommendation"),
            cart_contents=session.get("cart_contents", []),
            cart_total=session.get("cart_total", 0),
            purchased_items=cart_override,
            session_id=session["session_id"],
            razorpay_order_id=session.get("razorpay_order_id"),
            razorpay_key_id=session.get("razorpay_key_id")
        )


# Global singleton instance for Agent 1
agent1_service = BuyerAgent()

