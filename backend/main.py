import os
from fastapi import FastAPI, HTTPException, Query, status, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from typing import List, Optional, Dict, Any

from backend.database import init_db
from backend.auth_service import (
    auth_service,
    UserAlreadyExistsError,
    InvalidCredentialsError,
    AuthenticationError
)
from backend.data_access import (
    dal,
    MerchantNotFoundError,
    ProductNotFoundError,
    InsufficientStockError,
    PriceMismatchError,
    MerchantDataError
)
from backend.schemas import (
    Product,
    InventoryItem,
    Order,
    OrderCreate,
    InventoryUpdateRequest,
    SemanticSearchResult,
    HardConstraints,
    ScoredOffer,
    Agent2RecommendRequest,
    Agent2RecommendResponse,
    Agent1ChatRequest,
    Agent1ChatResponse,
    RazorpayOrderCreateRequest,
    RazorpayOrderCreateResponse,
    RazorpayPaymentVerifyRequest,
    RazorpayPaymentVerifyResponse,
    UserRegisterRequest,
    UserLoginRequest,
    AuthResponse,
    UserResponse,
    AddressRequest,
    AddressResponse,
    UserOrderResponse
)
from backend.payment_service import create_razorpay_order_for_session, verify_payment_for_session
from backend.semantic_search import semantic_search_engine
from backend.scoring_engine import scoring_engine
from backend.agent2_service import agent2_service
from backend.agent1_service import agent1_service

app = FastAPI(
    title="Agentic Commerce Platform API",
    description="Merchant Data Access Layer and Endpoints for ShopNest, CartWave and AI Buyer Agent",
    version="1.0.0"
)

# Initialize SQLite database schema on startup without wiping existing tables
@app.on_event("startup")
def startup_event():
    init_db()

# Enable CORS for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
IMAGES_DIR = os.path.join(FRONTEND_DIR, "Images") if os.path.exists(os.path.join(FRONTEND_DIR, "Images")) else os.path.join(FRONTEND_DIR, "images")

# Direct route for product images with smart fallback resolution
@app.get("/images/{image_name}", tags=["Images"])
@app.get("/Images/{image_name}", tags=["Images"])
def get_product_image(image_name: str):
    """Serve product image directly by filename with intelligent fallback support."""
    clean_name = os.path.basename(image_name).replace(" ", "_")
    img_path = os.path.join(IMAGES_DIR, clean_name)
    if os.path.exists(img_path) and os.path.isfile(img_path):
        return FileResponse(img_path, media_type="image/jpeg")
    
    # Try case-insensitive matching
    if os.path.exists(IMAGES_DIR):
        all_files = os.listdir(IMAGES_DIR)
        for f in all_files:
            if f.lower() == clean_name.lower():
                return FileResponse(os.path.join(IMAGES_DIR, f), media_type="image/jpeg")
                
        # Fallback: if p_id.jpg is requested without color variant, serve the first matching variant
        base_name = os.path.splitext(clean_name)[0]
        for f in all_files:
            if f.lower().startswith(f"{base_name.lower()}_") and f.lower().endswith((".jpg", ".jpeg", ".png")):
                return FileResponse(os.path.join(IMAGES_DIR, f), media_type="image/jpeg")

    raise HTTPException(status_code=404, detail=f"Image '{image_name}' not found")

# Mount static files directory as fallback/static route
if os.path.exists(IMAGES_DIR):
    app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")
    app.mount("/Images", StaticFiles(directory=IMAGES_DIR), name="Images")


# Serve application logo directly if requested
@app.get("/Logo.jpeg", response_class=FileResponse, tags=["Images"])
def get_app_logo():
    logo_path = os.path.join(FRONTEND_DIR, "Logo.jpeg")
    if os.path.exists(logo_path):
        return FileResponse(logo_path, media_type="image/jpeg")
    raise HTTPException(status_code=404, detail="Logo.jpeg not found")


@app.get("/api/health", tags=["System"])
def health_check():
    """Health check endpoint to verify backend operational readiness."""
    return {"status": "ok", "service": "agentic-commerce-backend"}


# -----------------------------------------------------------------------------
# Authentication Dependency
# -----------------------------------------------------------------------------
def get_current_user(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    """
    Extracts Bearer token from Authorization header, validates session,
    and returns authenticated user dictionary. Never trusts client user_id.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please log in."
        )
    token = authorization.split(" ", 1)[1].strip()
    user = auth_service.get_user_by_token(token)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session. Please log in."
        )
    user["token"] = token
    return user


def get_optional_current_user(authorization: Optional[str] = Header(None)) -> Optional[Dict[str, Any]]:
    """
    Optional authentication dependency for non-strictly-protected endpoints.
    """
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization.split(" ", 1)[1].strip()
    user = auth_service.get_user_by_token(token)
    if user:
        user["token"] = token
    return user


# -----------------------------------------------------------------------------
# Frontend Storefront & App Routes
# -----------------------------------------------------------------------------
@app.get("/login", response_class=FileResponse, tags=["Storefronts"])
@app.get("/login.html", response_class=FileResponse, tags=["Storefronts"])
def get_login_page():
    """Serves the standalone Login & Registration page."""
    path = os.path.join(FRONTEND_DIR, "login.html")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="login.html not found")
    return FileResponse(path)


@app.get("/address", response_class=FileResponse, tags=["Storefronts"])
@app.get("/address.html", response_class=FileResponse, tags=["Storefronts"])
def get_address_page():
    """Serves the delivery address entry page."""
    path = os.path.join(FRONTEND_DIR, "address.html")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="address.html not found")
    return FileResponse(path)


@app.get("/agent1", response_class=FileResponse, tags=["Storefronts"])
@app.get("/agent1.html", response_class=FileResponse, tags=["Storefronts"])
def get_agent1_page():
    """Serves the AI Buyer Agent conversational shopping interface."""
    path = os.path.join(FRONTEND_DIR, "agent1.html")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="agent1.html not found")
    return FileResponse(path)


@app.get("/shopnest", response_class=FileResponse, tags=["Storefronts"])
@app.get("/shopnest.html", response_class=FileResponse, tags=["Storefronts"])
def get_shopnest_storefront():
    """Serves the ShopNest Amazon-style merchant storefront."""
    path = os.path.join(FRONTEND_DIR, "shopnest.html")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="shopnest.html not found")
    return FileResponse(path)


@app.get("/cartwave", response_class=FileResponse, tags=["Storefronts"])
@app.get("/cartwave.html", response_class=FileResponse, tags=["Storefronts"])
def get_cartwave_storefront():
    """Serves the CartWave Flipkart-style merchant storefront."""
    path = os.path.join(FRONTEND_DIR, "cartwave.html")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="cartwave.html not found")
    return FileResponse(path)


@app.get("/guide", response_class=FileResponse, tags=["Storefronts"])
@app.get("/guide.html", response_class=FileResponse, tags=["Storefronts"])
def get_guide_page():
    """Serves the Complete System Architecture & Logic Guide page."""
    path = os.path.join(FRONTEND_DIR, "guide.html")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="guide.html not found")
    return FileResponse(path)


@app.get("/", response_class=FileResponse, tags=["Storefronts"])
def get_root_page():
    """Serves the Login & Registration page by default when opening the root application URL."""
    path = os.path.join(FRONTEND_DIR, "login.html")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="login.html not found")
    return FileResponse(path)


@app.get("/hub", response_class=HTMLResponse, tags=["Storefronts"])
def get_hub():
    """Navigation hub to access the Login, Buyer Agent, Guide and merchant storefronts."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
      <title>Agentic Commerce Platform</title>
      <style>
        body { font-family: sans-serif; display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 80vh; background: #f0f2f5; margin: 0; padding: 20px; }
        .card-container { display: flex; flex-wrap: wrap; justify-content: center; gap: 20px; margin-top: 24px; }
        .store-card { background: white; padding: 24px 28px; border-radius: 12px; text-decoration: none; color: #111; box-shadow: 0 4px 14px rgba(0,0,0,0.08); text-align: center; width: 200px; font-weight: bold; font-size: 17px; transition: transform 0.2s, box-shadow 0.2s; }
        .store-card:hover { transform: translateY(-4px); box-shadow: 0 8px 20px rgba(0,0,0,0.12); }
        .ag { border-top: 5px solid #2563eb; }
        .gd { border-top: 5px solid #7c3aed; }
        .lg { border-top: 5px solid #10b981; }
        .sn { border-top: 5px solid #febd69; }
        .cw { border-top: 5px solid #2874f0; }
      </style>
    </head>
    <body>
      <h2>Agentic Commerce Platform</h2>
      <p style="color: #666; font-size: 14px;">Select an application module to open:</p>
      <div class="card-container">
        <a href="/login" class="store-card lg">Sign In / Register<br><small style="font-size: 13px; color: #666;">Authentication Portal</small></a>
        <a href="/agent1" class="store-card ag">Buyer Agent<br><small style="font-size: 13px; color: #666;">AI Shopping Assistant</small></a>
        <a href="/guide" class="store-card gd">System Guide<br><small style="font-size: 13px; color: #666;">Architecture & Logic</small></a>
        <a href="/shopnest" class="store-card sn">ShopNest<br><small style="font-size: 13px; color: #666;">Amazon-Style Store</small></a>
        <a href="/cartwave" class="store-card cw">CartWave<br><small style="font-size: 13px; color: #666;">Flipkart-Style Store</small></a>
      </div>
    </body>
    </html>
    """


# =============================================================================
# User Authentication API Endpoints
# =============================================================================
@app.post("/api/auth/register", response_model=AuthResponse, tags=["Authentication"])
def register(req: UserRegisterRequest):
    """
    Registers a new user account.
    Rejects duplicate email addresses with HTTP 409 Conflict.
    """
    try:
        res = auth_service.register_user(name=req.name, email=req.email, password=req.password)
        return res
    except UserAlreadyExistsError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/api/auth/login", response_model=AuthResponse, tags=["Authentication"])
def login(req: UserLoginRequest):
    """
    Authenticates an existing user and creates a secure session token.
    """
    try:
        res = auth_service.login_user(email=req.email, password=req.password)
        return res
    except InvalidCredentialsError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/api/auth/logout", tags=["Authentication"])
def logout(current_user: Dict[str, Any] = Depends(get_current_user)):
    """
    Invalidates the current session token in the database.
    """
    token = current_user.get("token")
    if token:
        auth_service.logout_user(token)
    return {"success": True, "message": "Logged out successfully"}


@app.get("/api/auth/me", response_model=UserResponse, tags=["Authentication"])
def get_current_user_profile(current_user: Dict[str, Any] = Depends(get_current_user)):
    """
    Returns the currently authenticated user's profile info.
    """
    return {
        "id": current_user["id"],
        "name": current_user["name"],
        "email": current_user["email"],
        "created_at": current_user.get("created_at")
    }


# =============================================================================
# User Address Management API Endpoints
# =============================================================================
@app.get("/api/user/address", tags=["User Address"])
def get_user_delivery_address(current_user: Dict[str, Any] = Depends(get_current_user)):
    """
    Retrieves the authenticated user's current default delivery address.
    """
    addr = auth_service.get_user_address(current_user["id"])
    return addr


@app.post("/api/user/address", response_model=AddressResponse, tags=["User Address"])
def save_user_delivery_address(
    req: AddressRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Saves or updates the authenticated user's current delivery address.
    """
    try:
        saved_addr = auth_service.save_user_address(current_user["id"], req.dict())
        return saved_addr
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# =============================================================================
# Per-User Order History API Endpoints
# =============================================================================
@app.get("/api/user/orders", response_model=List[UserOrderResponse], tags=["User Orders"])
def get_user_order_history(current_user: Dict[str, Any] = Depends(get_current_user)):
    """
    Retrieves the isolated order history belonging strictly to the authenticated user.
    """
    orders = auth_service.get_user_orders(current_user["id"])
    return orders


@app.get("/api/user/orders/{order_id}", response_model=UserOrderResponse, tags=["User Orders"])
def get_user_order_details(
    order_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Retrieves a single order only if it belongs to the authenticated user.
    """
    order = auth_service.get_user_order_by_id(current_user["id"], order_id)
    if not order:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Order not found or does not belong to the authenticated user."
        )
    return order




# -----------------------------------------------------------------------------
# Products Endpoints
# -----------------------------------------------------------------------------
@app.get("/api/{merchant}/products", response_model=List[Product], tags=["Products"])
def get_merchant_products(merchant: str, category: Optional[str] = Query(None, description="Filter by category")):
    """
    Retrieve all products in a merchant's catalog, optionally filtered by category.
    """
    try:
        return dal.get_products(merchant, category=category)
    except MerchantNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.get("/api/{merchant}/products/{p_id}", response_model=Product, tags=["Products"])
def get_merchant_product(merchant: str, p_id: str):
    """
    Retrieve a single product by its unique product ID (p_id) from the specified merchant.
    """
    try:
        product = dal.get_product(merchant, p_id)
        if not product:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Product '{p_id}' not found in merchant '{merchant}'"
            )
        return product
    except MerchantNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# -----------------------------------------------------------------------------
# Inventory Endpoints
# -----------------------------------------------------------------------------
@app.get("/api/{merchant}/inventory", response_model=List[InventoryItem], tags=["Inventory"])
def get_merchant_inventory(merchant: str):
    """
    Retrieve current stock levels for all products in the merchant inventory.
    """
    try:
        return dal.get_inventory(merchant)
    except MerchantNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.get("/api/{merchant}/inventory/{p_id}", response_model=InventoryItem, tags=["Inventory"])
def get_merchant_stock(merchant: str, p_id: str):
    """
    Retrieve current stock level for a specific product ID.
    """
    try:
        stock = dal.get_stock(merchant, p_id)
        if stock is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Product '{p_id}' not found in '{merchant}' inventory"
            )
        return {"p_id": p_id, "stock": stock}
    except MerchantNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.patch("/api/{merchant}/inventory/{p_id}", response_model=InventoryItem, tags=["Inventory"])
def update_merchant_stock(merchant: str, p_id: str, req: InventoryUpdateRequest):
    """
    Update stock for a specific product by applying a delta or setting exact stock.
    """
    try:
        updated_stock = dal.update_stock(
            merchant=merchant,
            p_id=p_id,
            stock_delta=req.stock_delta,
            new_stock=req.new_stock
        )
        return {"p_id": p_id, "stock": updated_stock}
    except MerchantNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except (ProductNotFoundError, InsufficientStockError, ValueError) as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# -----------------------------------------------------------------------------
# Orders Endpoints
# -----------------------------------------------------------------------------
@app.get("/api/{merchant}/orders", response_model=List[Order], tags=["Orders"])
def get_merchant_orders(merchant: str, user_only: bool = Query(False, description="Filter for user/agent purchases only")):
    """
    Retrieve order history for a merchant, optionally filtered to live user/agent purchases.
    """
    try:
        return dal.get_orders(merchant, user_only=user_only)
    except MerchantNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.get("/api/{merchant}/orders/{order_id}", response_model=Order, tags=["Orders"])
def get_merchant_order(merchant: str, order_id: str):
    """
    Retrieve a specific order by its order ID.
    """
    try:
        order = dal.get_order(merchant, order_id)
        if not order:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Order '{order_id}' not found in merchant '{merchant}'"
            )
        return order
    except MerchantNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/api/{merchant}/orders", response_model=Order, status_code=status.HTTP_201_CREATED, tags=["Orders"])
def create_merchant_order(merchant: str, order_in: OrderCreate):
    """
    Create and append a new confirmed order to the merchant's orders.json.
    Guarantees inventory validation & deduction, authoritative price check, and rollback safety.
    """
    try:
        order_dict = order_in.dict(exclude_unset=True)
        created_order = dal.append_order(merchant, order_dict)
        return created_order
    except MerchantNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except (ProductNotFoundError, InsufficientStockError, PriceMismatchError, ValueError) as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# -----------------------------------------------------------------------------
# Semantic Search Endpoint
# -----------------------------------------------------------------------------
@app.get("/api/{merchant}/semantic-search", response_model=List[SemanticSearchResult], tags=["Search"])
def semantic_search_endpoint(
    merchant: str,
    q: str = Query(..., description="Natural language search query"),
    top_k: int = Query(5, ge=1, le=50, description="Maximum number of results to return")
):
    """
    Local Semantic Search endpoint powered by sentence-transformers/all-MiniLM-L6-v2.
    Ranks merchant products by cosine similarity against the natural language query.
    """
    try:
        return semantic_search_engine.search(merchant=merchant, query=q, top_k=top_k)
    except MerchantNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# -----------------------------------------------------------------------------
# Deterministic Scoring & Merchant Comparison Endpoints
# -----------------------------------------------------------------------------
@app.get("/api/scoring/compare/{p_id}", response_model=List[ScoredOffer], tags=["Scoring"])
def compare_product_offers(
    p_id: str,
    priority: str = Query("best_balance", description="Priority mode: 'cheapest', 'highest_rated', 'best_balance'"),
    budget: Optional[int] = Query(None, description="Max budget constraint"),
    size: Optional[str] = Query(None, description="Required size"),
    color: Optional[str] = Query(None, description="Required color"),
    required_quantity: int = Query(1, ge=1, description="Required purchase quantity")
):
    """
    Compares live offers for a specific product ID across ShopNest and CartWave,
    applies deterministic hard constraint filtering, and scores/ranks the offers.
    """
    try:
        return scoring_engine.compare_merchants_for_product(
            p_id=p_id,
            priority=priority,
            budget=budget,
            size=size,
            color=color,
            required_quantity=required_quantity
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# -----------------------------------------------------------------------------
# Agent 2 — Sales Improvement Agent Recommendation Endpoints
# -----------------------------------------------------------------------------
@app.post("/api/agent2/recommend", response_model=Agent2RecommendResponse, tags=["Agent 2"])
def agent2_recommend_post(request_in: Agent2RecommendRequest):
    """
    Agent 2 — Sales Improvement Agent endpoint.
    Dynamically computes co-purchase probabilities from historical orders, validates inventory,
    and returns a single complementary product recommendation with natural-language message.
    """
    try:
        return agent2_service.get_recommendation(
            merchant=request_in.merchant,
            selected_product_id=request_in.selected_product_id,
            current_cart_items=request_in.current_cart_items,
            required_quantity=request_in.required_quantity
        )
    except (MerchantNotFoundError, ProductNotFoundError) as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.get("/api/agent2/recommend", response_model=Agent2RecommendResponse, tags=["Agent 2"])
def agent2_recommend_get(
    merchant: str = Query(..., description="Merchant name ('shopnest' or 'cartwave')"),
    p_id: str = Query(..., description="Selected product ID (Product A)"),
    cart_items: Optional[str] = Query(None, description="Comma-separated product IDs currently in cart"),
    quantity: int = Query(1, ge=1, description="Required quantity context")
):
    """
    Convenience GET endpoint for Agent 2 recommendation testing.
    """
    try:
        cart_list = [item.strip() for item in cart_items.split(",")] if cart_items else []
        return agent2_service.get_recommendation(
            merchant=merchant,
            selected_product_id=p_id,
            current_cart_items=cart_list,
            required_quantity=quantity
        )
    except (MerchantNotFoundError, ProductNotFoundError) as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# -----------------------------------------------------------------------------
# Agent 1 — AI Buyer Agent Orchestration Endpoints
# -----------------------------------------------------------------------------
@app.post("/api/agent1/chat", response_model=Agent1ChatResponse, tags=["Agent 1"])
def agent1_chat_post(
    request_in: Agent1ChatRequest,
    authorization: Optional[str] = Header(None)
):
    """
    Agent 1 — AI Buyer Agent dialogue orchestration endpoint.
    Extracts shopping slots, triggers Step 4A Semantic Search & Step 4B Scoring,
    manages cart additions, coordinates Step 5 Agent 2 recommendations,
    and returns READY_FOR_PURCHASE upon user confirmation.
    """
    try:
        if authorization and authorization.startswith("Bearer "):
            token = authorization.split(" ", 1)[1].strip()
            user = auth_service.get_user_by_token(token)
            if user:
                sid, session = agent1_service.get_session(request_in.session_id)
                session["user_id"] = user["id"]
        return agent1_service.process_message(request_in)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))



@app.get("/api/agent1/chat", response_model=Agent1ChatResponse, tags=["Agent 1"])
def agent1_chat_get(
    message: str = Query(..., description="User message"),
    session_id: Optional[str] = Query(None, description="Optional conversation session ID")
):
    """
    Convenience GET endpoint for Agent 1 chat testing.
    """
    try:
        req = Agent1ChatRequest(message=message, session_id=session_id)
        return agent1_service.process_message(req)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# -----------------------------------------------------------------------------
# Agent 1 Session Cart Synchronization Endpoints
# -----------------------------------------------------------------------------
@app.get("/api/session/{session_id}/cart", tags=["Agent 1 Session"])
def get_session_cart(session_id: str):
    """
    Retrieve the authoritative Master Cart state for the given session ID.
    """
    try:
        _, session = agent1_service.get_session(session_id)
        return {
            "cart_contents": session.get("cart_contents", []),
            "cart_total": session.get("cart_total", 0),
            "current_state": session.get("current_state", "COLLECTING_REQUIREMENTS")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/session/{session_id}/cart", tags=["Agent 1 Session"])
def update_session_cart(session_id: str, cart_data: dict):
    """
    Overwrites the authoritative Master Cart state for the given session ID.
    """
    try:
        _, session = agent1_service.get_session(session_id)
        new_contents = cart_data.get("cart_contents", [])
        
        validated = []
        for item in new_contents:
            if not all(k in item for k in ("p_id", "p_name", "merchant", "price", "quantity")):
                raise HTTPException(status_code=400, detail="Missing required keys in cart item specification")
            enriched = agent1_service._enrich_cart_item({
                "p_id": item["p_id"],
                "p_name": item["p_name"],
                "merchant": item["merchant"].strip().lower(),
                "price": int(item["price"]),
                "rating": float(item.get("rating", 0.0)),
                "quantity": int(item["quantity"])
            })
            validated.append(enriched)
            
        session["cart_contents"] = validated
        session["cart_total"] = sum(i["price"] * i["quantity"] for i in validated)
        return {
            "cart_contents": session["cart_contents"],
            "cart_total": session["cart_total"],
            "current_state": session.get("current_state")
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------------------------------------------
# Razorpay Test Mode Payment Endpoints
# -----------------------------------------------------------------------------
@app.post("/api/payment/create-order", response_model=RazorpayOrderCreateResponse, tags=["Payment"])
def create_payment_order(
    req: RazorpayOrderCreateRequest,
    authorization: Optional[str] = Header(None)
):
    """
    Validates the session cart via Trust Layer, verifies user address requirement,
    and creates a Razorpay Order.
    """
    try:
        auth_token = None
        if authorization and authorization.startswith("Bearer "):
            auth_token = authorization.split(" ", 1)[1].strip()
        res = create_razorpay_order_for_session(req.session_id, auth_token=auth_token)
        return res
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/api/payment/verify", response_model=RazorpayPaymentVerifyResponse, tags=["Payment"])
def verify_payment(
    req: RazorpayPaymentVerifyRequest,
    authorization: Optional[str] = Header(None)
):
    """
    Verifies payment signature, runs idempotency checks, records the order in DAL,
    stores user order history in SQLite, and clears the cart on success.
    """
    try:
        auth_token = None
        if authorization and authorization.startswith("Bearer "):
            auth_token = authorization.split(" ", 1)[1].strip()
        res = verify_payment_for_session(
            session_id=req.session_id,
            razorpay_order_id=req.razorpay_order_id,
            razorpay_payment_id=req.razorpay_payment_id,
            razorpay_signature=req.razorpay_signature,
            auth_token=auth_token
        )
        return res
    except ValueError as e:
        return {
            "success": False,
            "payment_status": "FAILED",
            "message": str(e)
        }
    except Exception as e:
        return {
            "success": False,
            "payment_status": "FAILED",
            "message": f"An internal server error occurred: {str(e)}"
        }


@app.get("/api/session/{session_id}/audit", tags=["Audit"])
def get_session_audit(session_id: str):
    """
    Retrieve the canonical audit log details for a conversation session.
    """
    try:
        from backend.audit_builder import build_audit
        return build_audit(session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate session audit: {str(e)}")







