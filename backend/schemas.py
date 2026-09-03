from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field


class ProductAttributes(BaseModel):
    sizes: Optional[List[Any]] = None
    colors: Optional[List[str]] = None
    gender: Optional[str] = None
    material: Optional[str] = None
    fit: Optional[str] = None
    sleeve: Optional[str] = None
    waist: Optional[List[int]] = None
    gsm: Optional[int] = None
    features: Optional[str] = None
    capacity: Optional[str] = None
    battery_hours: Optional[int] = None
    anc_db: Optional[int] = None
    screen_size: Optional[str] = None
    battery_days: Optional[int] = None
    max_output_w: Optional[int] = None
    wattage: Optional[int] = None
    ports: Optional[List[str]] = None
    length_m: Optional[int] = None
    power_w: Optional[int] = None
    switch_types: Optional[List[str]] = None
    dpi: Optional[int] = None
    connectivity: Optional[List[str]] = None
    thickness: Optional[str] = None
    levels: Optional[List[str]] = None
    shoe_compartment: Optional[bool] = None
    modes: Optional[int] = None
    spf: Optional[str] = None
    fragrance_type: Optional[str] = None
    active_ingredient: Optional[str] = None
    hold: Optional[str] = None
    finish: Optional[str] = None
    pack_of: Optional[int] = None
    runtime: Optional[str] = None
    volume: Optional[str] = None
    weight: Optional[str] = None
    hardness: Optional[str] = None
    clarity: Optional[str] = None
    uv_protection: Optional[str] = None
    laptop_fit: Optional[str] = None
    thermal_hours: Optional[int] = None
    power: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None

    class Config:
        extra = "allow"


class Product(BaseModel):
    p_id: str = Field(..., description="Unique product ID, e.g., SHOE_001")
    p_name: str = Field(..., description="Full display name of the product")
    category: str = Field(..., description="Product category")
    description: str = Field(..., description="Detailed description")
    price: int = Field(..., ge=0, description="Price in INR")
    rating: float = Field(..., ge=0.0, le=5.0, description="Rating from 0 to 5")
    attributes: Dict[str, Any] = Field(default_factory=dict, description="Product attributes map")


class InventoryItem(BaseModel):
    p_id: str = Field(..., description="Unique product ID")
    stock: int = Field(..., ge=0, description="Current stock quantity")


class OrderItem(BaseModel):
    p_id: str = Field(..., description="Unique product ID")
    quantity: int = Field(..., ge=1, description="Quantity ordered")
    unit_price: int = Field(..., ge=0, description="Unit price in INR at time of purchase")
    color: Optional[str] = Field(None, description="Selected color variant")
    size: Optional[Union[str, int]] = Field(None, description="Selected size variant")


class OrderCreate(BaseModel):
    order_id: Optional[str] = None
    products: List[OrderItem] = Field(..., min_items=1, description="List of items in the order")
    total_amount: int = Field(..., ge=0, description="Total amount in INR")
    payment_status: str = Field(default="paid", description="Payment status")
    order_status: str = Field(default="confirmed", description="Order fulfillment status")
    created_at: Optional[str] = None
    is_user_order: Optional[bool] = Field(default=True, description="Whether this is a live user/agent order")
    order_source: Optional[str] = Field(default="agent_purchase", description="Source of the order")


class Order(BaseModel):
    order_id: str
    products: List[OrderItem]
    total_amount: int
    payment_status: str
    order_status: str
    created_at: str
    is_user_order: Optional[bool] = None
    order_source: Optional[str] = None


class InventoryUpdateRequest(BaseModel):
    stock_delta: Optional[int] = Field(None, description="Delta to adjust stock by (negative to decrease)")
    new_stock: Optional[int] = Field(None, ge=0, description="Exact new stock quantity")


class SemanticSearchResult(BaseModel):
    p_id: str = Field(..., description="Unique product ID")
    p_name: str = Field(..., description="Full display name of the product")
    category: str = Field(..., description="Product category")
    description: str = Field(..., description="Detailed description")
    price: int = Field(..., ge=0, description="Price in INR")
    rating: float = Field(..., ge=0.0, le=5.0, description="Rating from 0 to 5")
    merchant: str = Field(..., description="Merchant identifier (shopnest or cartwave)")
    similarity_score: float = Field(..., description="Cosine similarity score between query and product")


class HardConstraints(BaseModel):
    budget: Optional[int] = Field(None, ge=0, description="Maximum budget / price ceiling")
    category: Optional[str] = Field(None, description="Exact product category filter")
    size: Optional[Union[str, int]] = Field(None, description="Required size (e.g. 9, 'L', 'M')")
    color: Optional[str] = Field(None, description="Required color (e.g. 'Black', 'Blue')")
    required_quantity: int = Field(1, ge=1, description="Required purchase quantity (must be <= stock)")
    attributes: Optional[Dict[str, Any]] = Field(None, description="Additional attribute filters")


class ScoredOffer(BaseModel):
    p_id: str = Field(..., description="Unique product ID")
    p_name: str = Field(..., description="Full display name of the product")
    category: str = Field(..., description="Product category")
    description: str = Field(..., description="Detailed description")
    merchant: str = Field(..., description="Merchant offering the product ('shopnest' or 'cartwave')")
    price: int = Field(..., ge=0, description="Unit price in INR at this merchant")
    rating: float = Field(..., ge=0.0, le=5.0, description="Merchant rating")
    available_stock: int = Field(..., ge=0, description="Current stock available at this merchant")
    price_score: float = Field(..., ge=0.0, le=1.0, description="Calculated Price Score [0..1]")
    rating_score: float = Field(..., ge=0.0, le=1.0, description="Calculated Rating Score [0..1]")
    value_score: float = Field(..., ge=0.0, le=1.0, description="Calculated Value Score [0..1]")
    selected_priority: str = Field(..., description="Selected priority mode (cheapest, highest_rated, best_balance)")
    similarity_score: Optional[float] = Field(None, description="Semantic similarity score if derived from search")


class Agent2RecommendRequest(BaseModel):
    merchant: str = Field(..., description="Selected merchant ('shopnest' or 'cartwave')")
    selected_product_id: str = Field(..., description="Product ID selected by Agent 1 (e.g. 'SHOE_001')")
    current_cart_items: Optional[List[str]] = Field(default_factory=list, description="List of product IDs currently in user's cart to exclude")
    required_quantity: int = Field(1, ge=1, description="Quantity context for inventory validation")


class Agent2RecommendResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    recommendation_available: bool = Field(..., description="Whether a valid complementary recommendation was found")
    merchant: str = Field(..., description="Merchant context")
    selected_product_id: str = Field(..., description="Selected product ID (Product A)")
    selected_product_name: Optional[str] = Field(None, description="Display name of Product A")
    recommended_product_id: Optional[str] = Field(None, description="Recommended complementary product ID (Product B)")
    recommended_product_name: Optional[str] = Field(None, description="Display name of Product B")
    category: Optional[str] = Field(None, description="Category of Product B")
    description: Optional[str] = Field(None, description="Description of Product B")
    price: Optional[int] = Field(None, description="Unit price of Product B in INR")
    rating: Optional[float] = Field(None, description="Rating of Product B")
    available_stock: Optional[int] = Field(None, description="Current available stock of Product B")
    co_purchase_probability: Optional[float] = Field(None, description="Calculated P(B|A) conditional probability")
    orders_with_selected: int = Field(0, description="Denominator: total orders containing Product A")
    orders_with_both: int = Field(0, description="Numerator: orders containing both Product A and B")
    recommendation_message: Optional[str] = Field(None, description="Natural language recommendation generated for Agent 1")
    model_used: Optional[str] = Field(None, description="LLM model identifier used (e.g. 'qwen/qwen3-4b:free')")
    llm_status: str = Field(..., description="Status of LLM generation: 'success', 'pending_api_key', 'llm_error', etc.")
    audit_evidence: Optional[Dict[str, Any]] = Field(None, description="Structured audit log evidence for auditing")


class Agent1ChatRequest(BaseModel):
    message: str = Field(..., description="User's natural-language message or response")
    session_id: Optional[str] = Field(None, description="Unique conversation session ID")
    state: Optional[Dict[str, Any]] = Field(None, description="Optional explicit conversation state dictionary")


class Agent1ChatResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    reply: str = Field(..., description="Agent 1's conversational response text to the user")
    current_state: str = Field(..., description="Current state machine step (e.g. COLLECTING_REQUIREMENTS, AWAITING_MAIN_CART_CONFIRMATION, READY_FOR_PURCHASE)")
    next_action: str = Field(..., description="Action recommendation (e.g. ASK_CLARIFICATION, SHOW_WINNER, READY_FOR_PURCHASE)")
    requirements: Dict[str, Any] = Field(default_factory=dict, description="Extracted shopping requirements slots")
    missing_requirements: List[str] = Field(default_factory=list, description="List of required slots still missing")
    winning_offer: Optional[Dict[str, Any]] = Field(None, description="Deterministic winning offer from Step 4B")
    agent2_recommendation: Optional[Dict[str, Any]] = Field(None, description="Agent 2 complementary product recommendation")
    cart_contents: List[Dict[str, Any]] = Field(default_factory=list, description="Current confirmed items in user's cart")
    cart_total: int = Field(0, ge=0, description="Calculated total price in INR for confirmed cart items")
    purchased_items: Optional[List[Dict[str, Any]]] = Field(None, description="Purchased items for final rendering")
    session_id: str = Field(..., description="Conversation session ID")
    razorpay_order_id: Optional[str] = Field(None, description="Razorpay order ID if payment is initiated")
    razorpay_key_id: Optional[str] = Field(None, description="Razorpay key ID if payment is initiated")


class RazorpayOrderCreateRequest(BaseModel):
    session_id: str = Field(..., description="Active session ID for the cart")


class RazorpayOrderCreateResponse(BaseModel):
    success: bool
    razorpay_order_id: str
    razorpay_key_id: str
    amount: int
    currency: str = "INR"
    cart_total: int


class RazorpayPaymentVerifyRequest(BaseModel):
    session_id: str = Field(..., description="Active session ID for the cart")
    razorpay_order_id: str = Field(..., description="Razorpay order ID")
    razorpay_payment_id: str = Field(..., description="Razorpay payment ID")
    razorpay_signature: str = Field(..., description="Razorpay payment signature")


class RazorpayPaymentVerifyResponse(BaseModel):
    success: bool
    payment_status: str
    order_status: Optional[str] = None
    razorpay_order_id: Optional[str] = None
    razorpay_payment_id: Optional[str] = None
    order_id: Optional[str] = None
    total: Optional[int] = None
    items: Optional[List[Dict[str, Any]]] = None
    shipping_address: Optional[Dict[str, Any]] = None
    message: Optional[str] = None



# =============================================================================
# User Authentication, Address & Per-User Order Schemas
# =============================================================================
class UserRegisterRequest(BaseModel):
    name: str = Field(..., min_length=1, description="Full Name of the user")
    email: str = Field(..., min_length=3, description="Email address")
    password: str = Field(..., min_length=1, description="Password")


class UserLoginRequest(BaseModel):
    email: str = Field(..., min_length=3, description="Email address")
    password: str = Field(..., min_length=1, description="Password")


class UserResponse(BaseModel):
    id: int
    name: str
    email: str
    created_at: Optional[str] = None


class AuthResponse(BaseModel):
    token: str
    user: UserResponse


class AddressRequest(BaseModel):
    recipient_name: str = Field(..., min_length=1, description="Recipient full name")
    phone: str = Field(..., min_length=5, description="Contact phone number")
    address_line1: str = Field(..., min_length=1, description="House/Street address")
    address_line2: Optional[str] = Field("", description="Apartment, Landmark (Optional)")
    city: str = Field(..., min_length=1, description="City")
    state: str = Field(..., min_length=1, description="State")
    pincode: str = Field(..., min_length=3, description="Postal / Pincode")
    country: str = Field("India", description="Country")


class AddressResponse(BaseModel):
    id: Optional[int] = None
    user_id: Optional[int] = None
    recipient_name: str
    phone: str
    address_line1: str
    address_line2: Optional[str] = ""
    city: str
    state: str
    pincode: str
    country: str = "India"
class UserOrderResponse(BaseModel):
    id: int
    user_id: int
    order_id: str
    merchant: str
    total_amount: float
    payment_status: str
    order_status: str
    items: List[Dict[str, Any]]
    shipping_address: Dict[str, Any]
    created_at: str


# =============================================================================
# UAP (Universal Agent Protocol) & AP2 (Agent Payment Protocol) Schemas
# =============================================================================
class UAPMessageEnvelope(BaseModel):
    protocol_version: str = Field(default="UAP/1.0", description="Protocol version identifier")
    message_id: str = Field(..., description="Unique message UUID")
    sender_id: str = Field(..., description="Verifiable ID of sender agent (e.g., agent1_buyer)")
    recipient_id: str = Field(..., description="Verifiable ID of recipient agent (e.g., agent2_merchant_shopnest)")
    intent: str = Field(..., description="Standardized intent (e.g., QUERY_OFFER, RECOMMEND_CROSS_SELL, MANDATE_SYNC)")
    timestamp: str = Field(..., description="ISO 8601 timestamp of message dispatch")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Standardized intent payload data")
    signature: Optional[str] = Field(None, description="Cryptographic signature of the envelope")


class UAPAgentManifest(BaseModel):
    agent_id: str
    name: str
    role: str
    protocol_version: str = "UAP/1.0"
    endpoints: Dict[str, str]
    capabilities: List[str]
    supported_intents: List[str]
    public_key: Optional[str] = None


class AP2MandateRequest(BaseModel):
    session_id: str = Field(..., description="Active shopping session ID")
    max_amount: int = Field(..., ge=1, description="Upper bound spending limit in INR")
    authorized_merchants: List[str] = Field(default=["shopnest", "cartwave"], description="Merchants permitted to charge against this mandate")
    validity_minutes: int = Field(default=60, ge=1, le=1440, description="Mandate validity window in minutes")
    purpose: Optional[str] = Field(default="Agentic Shopping Checkout", description="Human-readable purpose of delegation")
    cart_items: Optional[List[Dict[str, Any]]] = Field(default=None, description="Optional explicit cart contents for delegation mandate generation")


class AP2Mandate(BaseModel):
    mandate_id: str = Field(..., description="Unique mandate identifier, e.g. ap2_man_...")
    session_id: str
    user_id: Optional[int] = None
    agent_id: str = Field(default="agent1_buyer")
    authorized_merchants: List[str]
    max_amount: int = Field(..., description="Hard upper spending bound in INR")
    currency: str = "INR"
    created_at: str
    expires_at: str
    cart_hash: str = Field(..., description="SHA-256 fingerprint of authorized cart items")
    status: str = Field(default="AUTHORIZED", description="Mandate state: AUTHORIZED, CLAIMED, EXPIRED, REVOKED")
    signature: str = Field(..., description="HMAC-SHA256 signature guaranteeing tamper-proof mandate")


class AP2PaymentClaimRequest(BaseModel):
    mandate_id: str = Field(..., description="Authorized AP2 mandate ID")
    session_id: str
    merchant: str
    claim_amount: int = Field(..., ge=1, description="Actual settlement amount in INR")
    cart_items: List[Dict[str, Any]]
    razorpay_payment_id: Optional[str] = None


class AP2SettlementReceipt(BaseModel):
    receipt_id: str
    mandate_id: str
    session_id: str
    merchant: str
    amount_paid: int
    currency: str = "INR"
    settled_at: str
    razorpay_payment_id: str
    status: str = "SETTLED"
    receipt_hash: str


