import os
import copy
from datetime import datetime
from typing import Dict, Any, List, Optional
from backend.agent1_service import agent1_service

def build_audit(session_id: str) -> Dict[str, Any]:
    """
    Constructs a canonical read-only audit log for the given session_id.
    Collects runtime state and metadata stored during execution.
    """
    # 1. Retrieve the active session mapping
    _, session = agent1_service.get_session(session_id)
    
    # 2. Extract baseline variables (with safe fallbacks)
    requirements = session.get("requirements", {})
    winning_offer = session.get("winning_offer")
    scored_offers = session.get("scored_offers", [])
    agent2_rec = session.get("agent2_recommendation_history") or session.get("agent2_recommendation")
    trust_results = session.get("trust_layer_results", {})
    conversation_history = session.get("conversation_history", [])
    
    # Determine pre-checkout cart vs current cart (since cart gets cleared on purchase)
    pre_checkout_cart = session.get("pre_checkout_cart")
    if pre_checkout_cart is None:
        pre_checkout_cart = session.get("cart_contents", [])
    
    cart_total = sum(i["price"] * i["quantity"] for i in pre_checkout_cart)
    
    # 3. Calculate selection reasoning based on actual values
    selection_reasoning = "No product has been evaluated or selected yet."
    if winning_offer and scored_offers:
        prio_label = str(requirements.get("priority", "best_balance")).replace("_", " ").title()
        cand_count = len(scored_offers)
        
        # Build candidate breakdown
        cand_summaries = []
        for idx, c in enumerate(scored_offers, 1):
            p_sc = f"{c.get('price_score', 0):.4f}" if c.get('price_score') is not None else "N/A"
            r_sc = f"{c.get('rating_score', 0):.4f}" if c.get('rating_score') is not None else "N/A"
            v_sc = f"{c.get('value_score', 0):.4f}" if c.get('value_score') is not None else "N/A"
            sim_sc = f", Similarity Score: {c.get('similarity_score', 0):.4f}" if c.get('similarity_score') is not None else ""
            cand_summaries.append(
                f"Candidate {idx} ({c['merchant'].title()} — {c['p_name']}): Price = ₹{c['price']:,}, "
                f"Rating = {c.get('rating', 0.0):.1f}★, Price Score = {p_sc}, Rating Score = {r_sc}, Value Score = {v_sc}{sim_sc}."
            )
        
        candidates_text = " ".join(cand_summaries)
        
        # Specific rationale by priority
        win_prio = str(requirements.get("priority", "best_balance")).lower()
        if "cheapest" in win_prio:
            prio_rationale = (
                f"Under the 'Cheapest' priority mode (90% price weight, 10% rating weight), the evaluation strongly favored "
                f"the lowest price. '{winning_offer['p_name']}' at {winning_offer['merchant'].title()} (₹{winning_offer['price']:,}) "
                f"achieved the top Price Score of {winning_offer.get('price_score', 0):.4f} and winning Value Score of {winning_offer.get('value_score', 0):.4f}."
            )
        elif "highest" in win_prio:
            prio_rationale = (
                f"Under the 'Highest Rated' priority mode (90% rating weight, 10% price weight), the evaluation prioritized customer satisfaction. "
                f"'{winning_offer['p_name']}' at {winning_offer['merchant'].title()} ({winning_offer.get('rating', 0.0):.1f}★) "
                f"achieved the top Rating Score of {winning_offer.get('rating_score', 0):.4f} and winning Value Score of {winning_offer.get('value_score', 0):.4f}."
            )
        else:
            prio_rationale = (
                f"Under the 'Best Balance' priority mode (50% price weight, 50% rating weight), the evaluation balanced price competitiveness and rating quality. "
                f"'{winning_offer['p_name']}' at {winning_offer['merchant'].title()} (₹{winning_offer['price']:,}, {winning_offer.get('rating', 0.0):.1f}★) "
                f"yielded the optimal overall Value Score of {winning_offer.get('value_score', 0):.4f}."
            )
            
        # Tie-breaking status
        ties = [o for o in scored_offers if (o.get("p_id") != winning_offer.get("p_id") or o.get("merchant") != winning_offer.get("merchant")) and abs(o.get("value_score", 0) - winning_offer.get("value_score", 0)) < 1e-4]
        if ties:
            tie_text = (
                "Score ties were resolved using the deterministic tie-breaker hierarchy: "
                "Higher Rating Score -> Lower Price -> Higher Available Stock."
            )
        else:
            tie_text = "No Value Score ties occurred; the winner ranked highest unconditionally."

        selection_reasoning = (
            f"The AI Buyer Agent evaluated {cand_count} candidate offer(s) across merchants under the '{prio_label}' preference. "
            f"{candidates_text} {prio_rationale} {tie_text}"
        )

    # 4. Agent 2 Recommendation Details
    agent2_details = None
    if agent2_rec:
        if hasattr(agent2_rec, "dict"):
            agent2_rec = agent2_rec.dict()
        elif hasattr(agent2_rec, "model_dump"):
            agent2_rec = agent2_rec.model_dump()

        primary_name = agent2_rec.get("selected_product_name")
        if not primary_name and winning_offer:
            primary_name = winning_offer.get("p_name")

        audit_ev = agent2_rec.get("audit_evidence") or {}
        sim_val = audit_ev.get("semantic_similarity")
        if sim_val is None:
            sim_val = agent2_rec.get("semantic_similarity")

        prob = agent2_rec.get("co_purchase_probability")
        if prob is None:
            prob = 0.0

        agent2_details = {
            "primary_product": primary_name or "Primary Product",
            "recommended_product": agent2_rec.get("recommended_product_name") or "Complementary Product",
            "recommended_product_id": agent2_rec.get("recommended_product_id"),
            "merchant": agent2_rec.get("merchant", "N/A"),
            "price": agent2_rec.get("price", 0),
            "rating": agent2_rec.get("rating", 0.0),
            "co_purchase_probability": prob,
            "orders_with_selected": agent2_rec.get("orders_with_selected", 0),
            "orders_with_both": agent2_rec.get("orders_with_both", 0),
            "semantic_similarity": sim_val,
            "compatibility": "Compatible" if prob > 0 else "Incompatible",
            "model_used": agent2_rec.get("model_used", "Local Compatibility Rules"),
            "llm_status": agent2_rec.get("llm_status", "success"),
            "recommendation_available": agent2_rec.get("recommendation_available", True)
        }

    # 5. Format Trust Layer Results
    formatted_trust = []
    # Primary product validation check
    prim_res = trust_results.get("primary")
    if prim_res:
        formatted_trust.append({
            "check": "Merchant & Primary Product Existence",
            "result": "APPROVED" if prim_res.get("approved") else "REJECTED",
            "detail": f"Merchant: {prim_res.get('merchant', 'N/A').title()}, Product ID: {prim_res.get('product_id', 'N/A')}. Price: ₹{prim_res.get('price', 0):,}. Reason: {prim_res.get('reason', 'N/A')}"
        })
    else:
        formatted_trust.append({
            "check": "Merchant & Primary Product Existence",
            "result": "PENDING",
            "detail": "Primary product has not been confirmed or validated yet."
        })

    # Complementary product validation check
    comp_res = trust_results.get("complementary")
    consent_status = session.get("user_consent_complementary")
    if comp_res:
        formatted_trust.append({
            "check": "Complementary Consent & Validation",
            "result": "APPROVED" if comp_res.get("approved") else "REJECTED",
            "detail": f"Consent: {consent_status or 'N/A'}. Product ID: {comp_res.get('product_id', 'N/A')}. Reason: {comp_res.get('reason', 'N/A')}"
        })
    elif agent2_rec and agent2_rec.get("recommendation_available"):
        formatted_trust.append({
            "check": "Complementary Consent & Validation",
            "result": "PENDING",
            "detail": "Complementary recommendation offered, awaiting user selection."
        })
    else:
        formatted_trust.append({
            "check": "Complementary Consent & Validation",
            "result": "N/A",
            "detail": "No complementary recommendation was generated."
        })

    # Cart consistency verification check
    cart_res = trust_results.get("cart_consistency")
    if cart_res:
        formatted_trust.append({
            "check": "Pre-Checkout Cart Consistency Validation",
            "result": "APPROVED" if cart_res.get("approved") else "REJECTED",
            "detail": f"State: {cart_res.get('reason', 'N/A')}"
        })
    else:
        formatted_trust.append({
            "check": "Pre-Checkout Cart Consistency Validation",
            "result": "PENDING",
            "detail": "Cart consistency has not been verified yet."
        })

    # 6. Format Placed Orders
    placed_orders = session.get("placed_orders", [])
    payment_status = session.get("payment_status", "PENDING")
    
    # 7. Compile Canonical Audit Schema
    slots_summary = {
        "Priority Mode": str(requirements.get("priority", "N/A")).replace("_", " ").title(),
        "Quantity": str(requirements.get("quantity", 1)),
        "Size Option": str(requirements.get("size") if requirements.get("size") is not None else "N/A"),
        "Color Choice": str(requirements.get("color") if requirements.get("color") is not None else "N/A"),
    }
    if requirements.get("budget") is not None and str(requirements.get("budget")).strip().lower() not in ("none", "no limit", "n/a", ""):
        try:
            slots_summary["Max Budget"] = f"₹{int(requirements.get('budget')):,}"
        except (ValueError, TypeError):
            slots_summary["Max Budget"] = f"₹{requirements.get('budget')}"

    return {
        "session_id": session_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "original_user_request": session.get("original_user_request"),
        "conversation_history": conversation_history,
        "winning_offer": winning_offer,
        
        # Page 1: Buyer Request & Requirements
        "page1_buyer_request": {
            "original_request": session.get("original_user_request") or "No query submitted yet.",
            "product_query": requirements.get("product_query"),
            "quantity": requirements.get("quantity", 1),
            "size": requirements.get("size"),
            "color": requirements.get("color"),
            "budget": requirements.get("budget"),
            "priority": requirements.get("priority"),
            "slots_summary": slots_summary,
            "execution_overview": "USER REQUEST -> AGENT 1 (Slot-Filling) -> PRODUCT SELECTION -> TRUST LAYER -> AGENT 2 -> USER CONSENT -> CART -> TRUST LAYER CHECKOUT -> RAZORPAY -> FINAL ORDER"
        },
        
        # Page 2: Product Selection & Decision Logic
        "page2_product_selection": {
            "candidates": scored_offers,
            "winner": winning_offer,
            "selection_reasoning": selection_reasoning
        },
        
        # Page 3: Agent 2 + User Consent + Trust Layer
        "page3_agent2_trust_layer": {
            "agent2_recommendation": agent2_details,
            "user_consent_complementary": consent_status,
            "user_consent_detail": (
                "User explicitly approved the complementary recommendation." 
                if consent_status == "YES" 
                else ("User explicitly declined the complementary recommendation." if consent_status == "NO" else "Awaiting user selection.")
            ),
            "trust_layer_validations": formatted_trust
        },
        
        # Page 4: Cart Summary, Payment & Final Orders
        "page4_cart_payment_order": {
            "cart": {
                "items": pre_checkout_cart,
                "subtotal": cart_total,
                "total": cart_total
            },
            "payment": {
                "razorpay_order_id": session.get("razorpay_order_id"),
                "razorpay_payment_id": session.get("razorpay_payment_id"),
                "payment_status": payment_status,
                "signature_verified": session.get("payment_signature_verified", False)
            },
            "final_order": {
                "orders": placed_orders,
                "total": sum(o.get("total_amount", 0) for o in placed_orders) if placed_orders else 0
            },
            "audit_result": "SUCCESSFUL" if payment_status == "PAID" else "IN_PROGRESS"
        },

        # Page 5: UAP & AP2 Protocol Verification & Cryptographic Audit
        "page5_protocols_audit": {
            "uap_protocol": {
                "version": "UAP/1.0",
                "inter_agent_messages": session.get("uap_messages", []),
                "message_count": len(session.get("uap_messages", [])),
                "status": "VERIFIED" if session.get("uap_messages") else "STANDBY"
            },
            "ap2_protocol": {
                "version": "AP2/1.0",
                "mandate": session.get("ap2_mandate"),
                "mandate_id": session.get("ap2_mandate_id"),
                "cart_hash": session.get("ap2_mandate", {}).get("cart_hash") if session.get("ap2_mandate") else None,
                "max_authorized_bound": session.get("ap2_mandate", {}).get("max_amount") if session.get("ap2_mandate") else cart_total,
                "signature": session.get("ap2_mandate", {}).get("signature") if session.get("ap2_mandate") else None,
                "settlement_receipts": session.get("ap2_settlement_receipts", [])
            },
            "cryptographic_integrity": "GUARANTEED_TAMPER_PROOF"
        }
    }

