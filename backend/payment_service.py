import os
import copy
from typing import Dict, Any, List, Optional
import razorpay

# Import config first — this triggers load_dotenv() so that RAZORPAY_KEY_ID
# and RAZORPAY_KEY_SECRET are available from .env before we read them.
import backend.config  # noqa: F401

from backend.data_access import dal
from backend.trust_layer import TrustLayer


def get_razorpay_client() -> razorpay.Client:
    """
    Initializes and returns the Razorpay client.
    Credentials are read fresh from the environment on every call so that
    .env loading (via config.py) is guaranteed to have already happened.
    Raises ValueError if credentials are missing.
    """
    key_id = os.getenv("RAZORPAY_KEY_ID", "").strip()
    key_secret = os.getenv("RAZORPAY_KEY_SECRET", "").strip()
    if not key_id or not key_secret:
        raise ValueError(
            "Razorpay credentials (RAZORPAY_KEY_ID / RAZORPAY_KEY_SECRET) are missing in environment. "
            "Please check your .env file."
        )
    return razorpay.Client(auth=(key_id, key_secret))

def check_duplicate_payment(razorpay_payment_id: str) -> Optional[dict]:
    """
    Checks if a verified razorpay_payment_id has already been processed.
    Enforces idempotency and returns the existing order result.
    """
    matching_orders = []
    for merchant in ["shopnest", "cartwave"]:
        orders = dal.get_orders(merchant)
        for order in orders:
            if order.get("razorpay_payment_id") == razorpay_payment_id:
                matching_orders.append((merchant, order))
                
    if not matching_orders:
        return None
        
    total_amount = sum(item[1]["total_amount"] for item in matching_orders)
    razorpay_order_id = matching_orders[0][1].get("razorpay_order_id")
    order_ids = ", ".join(item[1]["order_id"] for item in matching_orders)
    
    items = []
    for merchant, order in matching_orders:
        for p in order.get("products", []):
            prod = dal.get_product(merchant, p["p_id"])
            p_name = prod.get("p_name") if prod else p["p_id"]
            items.append({
                "p_id": p["p_id"],
                "p_name": p_name,
                "merchant": merchant,
                "price": p["unit_price"],
                "quantity": p["quantity"],
                "color": p.get("color"),
                "size": p.get("size")
            })
            
    return {
        "success": True,
        "payment_status": "PAID",
        "order_status": "CONFIRMED",
        "razorpay_order_id": razorpay_order_id,
        "razorpay_payment_id": razorpay_payment_id,
        "order_id": order_ids,
        "total": total_amount,
        "items": items
    }

def create_razorpay_order_for_session(session_id: str, auth_token: Optional[str] = None) -> dict:
    """
    Identifies the session, retrieves the server-side cart, validates it
    via the Trust Layer, validates user delivery address, converts total to paise,
    and creates a Razorpay order.
    """
    from backend.agent1_service import agent1_service
    from backend.auth_service import auth_service
    
    # 1. Retrieve session
    sid, session = agent1_service.get_session(session_id)
    cart = session.get("cart_contents", [])
    if not cart:
        raise ValueError("Cannot create payment order: Cart is empty.")
        
    # 2. Validate cart consistency via Trust Layer
    trust_res = TrustLayer.validate_cart_consistency(cart)
    if not trust_res["approved"]:
        reason = trust_res.get("reason", "Cart validation failed")
        raise ValueError(f"Cart validation failed by Trust Layer: {reason}")

    # 3. Validate user delivery address
    user_id = None
    if auth_token:
        current_user = auth_service.get_user_by_token(auth_token)
        if current_user:
            user_id = current_user["id"]
            session["user_id"] = user_id
            addr = auth_service.get_user_address(user_id)
            if not addr:
                raise ValueError("Delivery address required. Please add your delivery address before checkout.")
            session["shipping_address"] = addr
        
    # 4. Calculate final amount and convert to paise
    cart_total = sum(i["price"] * i["quantity"] for i in cart)
    amount_paise = cart_total * 100
    
    # 5. Create Razorpay Test Mode order
    client = get_razorpay_client()
    order_data = {
        "amount": amount_paise,
        "currency": "INR",
        "receipt": f"receipt_{sid}",
        "notes": {
            "session_id": sid
        }
    }
    
    try:
        razorpay_order = client.order.create(data=order_data)
    except Exception as e:
        raise ValueError(f"Razorpay API failure: {str(e)}")
        
    # 6. Store order ID in session mapping
    key_id = os.getenv("RAZORPAY_KEY_ID", "").strip()
    session["razorpay_order_id"] = razorpay_order["id"]
    session["razorpay_key_id"] = key_id
    session["razorpay_order_amount"] = amount_paise
    session["current_state"] = "AWAITING_PAYMENT"
    
    return {
        "success": True,
        "razorpay_order_id": razorpay_order["id"],
        "razorpay_key_id": key_id,
        "amount": amount_paise,
        "currency": "INR",
        "cart_total": cart_total
    }

def verify_payment_for_session(
    session_id: str,
    razorpay_order_id: str,
    razorpay_payment_id: str,
    razorpay_signature: str,
    auth_token: Optional[str] = None
) -> dict:
    """
    Verifies payment signature, runs idempotency checks, records the order
    and stock updates in DAL, saves immutable per-user order records in user_orders,
    and clears the cart on success.
    """
    from backend.agent1_service import agent1_service
    from backend.auth_service import auth_service
    
    # 1. Check idempotency
    dup_res = check_duplicate_payment(razorpay_payment_id)
    if dup_res:
        return dup_res
        
    # 2. Check credentials readiness
    client = get_razorpay_client()
    
    # 3. Retrieve session and match Razorpay order ID
    sid, session = agent1_service.get_session(session_id)
    saved_order_id = session.get("razorpay_order_id")
    if not saved_order_id or saved_order_id != razorpay_order_id:
        raise ValueError("Razorpay order ID mismatch or no active payment request found for this session.")
        
    # 4. Verify signature via SDK
    try:
        client.utility.verify_payment_signature({
            "razorpay_order_id": razorpay_order_id,
            "razorpay_payment_id": razorpay_payment_id,
            "razorpay_signature": razorpay_signature
        })
    except Exception as e:
        raise ValueError(f"Razorpay payment signature verification failed: {str(e)}")
        
    # 5. Validate cart consistency again before checkout
    cart = session.get("cart_contents", [])
    if not cart:
        raise ValueError("Cannot complete payment verification: Cart is empty.")
        
    trust_res = TrustLayer.validate_cart_consistency(cart)
    session.setdefault("trust_layer_results", {})["cart_consistency"] = trust_res
    if not trust_res["approved"]:
        reason = trust_res.get("reason", "Cart validation failed")
        raise ValueError(f"Cart validation failed by Trust Layer: {reason}")
        
    # 6. Group items by merchant
    by_merchant = {}
    for item in cart:
        m = item["merchant"].strip().lower()
        if m not in by_merchant:
            by_merchant[m] = []
        by_merchant[m].append(item)
        
    # 7. Record final order(s) inside database/DAL
    placed_orders = []
    try:
        for m, items in by_merchant.items():
            order_items = [{
                "p_id": i["p_id"],
                "quantity": i["quantity"],
                "unit_price": i["price"],
                "color": i.get("color") or i.get("selected_color"),
                "size": i.get("size")
            } for i in items]
            expected_total = sum(i["price"] * i["quantity"] for i in items)
            
            # Merchant DAL order execution
            new_order = dal.execute_purchase_transaction(
                merchant=m,
                order_items=order_items,
                expected_total=expected_total,
                razorpay_order_id=razorpay_order_id,
                razorpay_payment_id=razorpay_payment_id
            )
            placed_orders.append((m, new_order, items))
    except Exception as e:
        raise ValueError(f"Failed to record final order transaction: {str(e)}")
        
    # 8. Identify authenticated user and record immutable per-user order history in SQLite
    user_id = session.get("user_id")
    if not user_id and auth_token:
        user_info = auth_service.get_user_by_token(auth_token)
        if user_info:
            user_id = user_info["id"]

    shipping_addr_snapshot = session.get("shipping_address") or {}
    if user_id and not shipping_addr_snapshot:
        shipping_addr_snapshot = auth_service.get_user_address(user_id) or {}

    if user_id:
        for m, o, items in placed_orders:
            items_payload = []
            for i in items:
                col = i.get("color") or i.get("selected_color")
                clean_col = str(col).replace(" ", "_") if col else None
                img_url = f"/images/{i['p_id']}_{clean_col}.jpg" if clean_col else f"/images/{i['p_id']}.jpg"
                items_payload.append({
                    "p_id": i["p_id"],
                    "p_name": i.get("p_name", i["p_id"]),
                    "merchant": m,
                    "color": col,
                    "size": i.get("size"),
                    "quantity": i["quantity"],
                    "unit_price": i["price"],
                    "image": img_url
                })
            try:
                auth_service.record_user_order(
                    user_id=user_id,
                    order_id=o["order_id"],
                    merchant=m,
                    total_amount=o["total_amount"],
                    payment_status="PAID",
                    order_status="CONFIRMED",
                    items=items_payload,
                    shipping_address=shipping_addr_snapshot
                )
            except Exception as e:
                # Log without breaking payment response
                print(f"[Warning] Failed to record user order history: {e}")


    # Record metadata for audit trail before clearing the cart
    session["pre_checkout_cart"] = copy.deepcopy(cart)
    session["razorpay_payment_id"] = razorpay_payment_id
    session["payment_status"] = "PAID"
    session["payment_signature_verified"] = True
    session["placed_orders"] = [
        {
            "merchant": m,
            "order_id": o["order_id"],
            "total_amount": o["total_amount"],
            "items": [{
                "p_id": i["p_id"],
                "p_name": i["p_name"],
                "price": i["price"],
                "quantity": i["quantity"],
                "color": i.get("color") or i.get("selected_color"),
                "size": i.get("size")
            } for i in items]
        }
        for m, o, items in placed_orders
    ]
        
    # 9. Clear cart only after successful order confirmation
    session["cart_contents"] = []
    session["cart_total"] = 0
    session["current_state"] = "READY_FOR_PURCHASE"
    
    # 10. Format response payload
    total_amount = sum(o["total_amount"] for _, o, _ in placed_orders)
    order_ids = ", ".join(o["order_id"] for _, o, _ in placed_orders)
    
    items_out = []
    for m, o, items in placed_orders:
        for i in items:
            item_data = {
                "p_id": i["p_id"],
                "p_name": i["p_name"],
                "merchant": m,
                "price": i["price"],
                "quantity": i["quantity"]
            }
            if i.get("color") or i.get("selected_color"):
                item_data["color"] = i.get("color") or i.get("selected_color")
            if i.get("size"):
                item_data["size"] = i.get("size")
            items_out.append(item_data)
            
    return {
        "success": True,
        "payment_status": "PAID",
        "order_status": "CONFIRMED",
        "razorpay_order_id": razorpay_order_id,
        "razorpay_payment_id": razorpay_payment_id,
        "order_id": order_ids,
        "total": total_amount,
        "items": items_out,
        "shipping_address": shipping_addr_snapshot if user_id else {}
    }



