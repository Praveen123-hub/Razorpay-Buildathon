import json
import urllib.request
import urllib.error
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

from backend.data_access import dal, MerchantNotFoundError, ProductNotFoundError
from backend.config import get_openrouter_api_key, get_agent2_model, OPENROUTER_BASE_URL
from backend.schemas import Agent2RecommendResponse


class SalesImprovementAgent:
    """
    Agent 2 Service responsible for dynamic co-purchase analysis and
    persuasive, context-aware complementary product recommendations.
    """

    def __init__(self):
        pass

    def check_compatibility(self, selected_product: Dict[str, Any], candidate_product: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Generic, data-driven compatibility validator checks product categories, descriptions, and semantic similarity.
        """
        sel_name = selected_product.get("p_name", "").lower()
        sel_cat = selected_product.get("category", "").lower()
        
        cand_name = candidate_product.get("p_name", "").lower()
        cand_cat = candidate_product.get("category", "").lower()
        
        # 1. Category Cluster check:
        cat_clusters = [
            {"footwear", "footwear accessories", "fitness gear", "fitness accessories"},
            {"apparel", "apparel accessories"},
            {"electronics", "electronics accessories", "lifestyle accessories"},
            {"personal care", "personal-care"},
            {"home & kitchen", "home and kitchen"}
        ]
        
        sel_cluster = None
        cand_cluster = None
        for cluster in cat_clusters:
            if sel_cat in cluster:
                sel_cluster = cluster
            if cand_cat in cluster:
                cand_cluster = cluster
                
        if not (sel_cluster and cand_cluster and sel_cluster == cand_cluster):
            return False, f"Incompatible: Category '{sel_cat}' and '{cand_cat}' belong to different clusters."
            
        # 2. Semantic Similarity Check using precomputed catalog vector embeddings
        from backend.semantic_search import semantic_search_engine
        
        try:
            sel_pid = selected_product.get("p_id")
            cand_pid = candidate_product.get("p_id")
            merchant = candidate_product.get("merchant", selected_product.get("merchant", "shopnest"))
            similarity = semantic_search_engine.compute_product_similarity(merchant, sel_pid, cand_pid)
        except Exception:
            similarity = 0.5  # Fallback in case of exceptions
            
        # Determine dynamic similarity threshold based on category types
        # Footwear/fitness and home & kitchen items typically have lower direct description
        # overlaps but are highly compatible (different physical materials, same use context)
        if ("footwear" in sel_cat or "footwear" in cand_cat
                or ("home" in sel_cat and "home" in cand_cat)):
            threshold = 0.30
        else:
            threshold = 0.40
            
        if similarity < threshold:
            return False, f"Incompatible: Semantic similarity {similarity:.4f} is below the threshold of {threshold}."
            
        return True, f"Compatible: Semantic similarity {similarity:.4f} satisfies threshold {threshold}."

    def calculate_co_purchase_statistics(
        self,
        merchant: str,
        selected_product_id: str,
        current_cart_items: Optional[List[str]] = None,
        required_quantity: int = 1
    ) -> Dict[str, Any]:
        """
        Dynamically analyzes historical orders for the selected merchant and calculates:
        P(B | A) = (Orders containing BOTH A and B) / (Orders containing A)

        Deterministic Python logic excludes:
        - Product A itself
        - Products in current cart
        - Out of stock / insufficient stock items
        - Uncatalogued product IDs
        """
        norm_merchant = merchant.strip().lower()
        selected_pid = selected_product_id.strip()
        cart_set = set(item.strip() for item in (current_cart_items or []))

        # Validate product exists in catalog
        selected_product = dal.get_product(norm_merchant, selected_pid)
        if not selected_product:
            raise ProductNotFoundError(f"Product '{selected_pid}' not found in {norm_merchant} catalog.")

        # Read orders exclusively for the selected merchant
        orders = dal.get_orders(norm_merchant)

        # Identify all historical orders containing Product A
        # (Unit of calculation is unique order, regardless of quantity within order)
        matching_orders = []
        for order in orders:
            order_pids = {item.get("p_id") for item in order.get("products", []) if item.get("p_id")}
            if selected_pid in order_pids:
                matching_orders.append(order_pids)

        denominator = len(matching_orders)
        if denominator == 0:
            return {
                "recommendation_available": False,
                "reason": f"No historical orders contain product '{selected_pid}' in {norm_merchant}.",
                "selected_product": selected_product,
                "denominator": 0,
                "candidates": []
            }

        # Count co-occurrences of every other product B in these matching orders
        co_counts: Dict[str, int] = {}
        for order_pids in matching_orders:
            for pid in order_pids:
                if pid == selected_pid:
                    continue  # Exclude Product A itself
                if pid in cart_set:
                    continue  # Exclude products already in user's cart
                co_counts[pid] = co_counts.get(pid, 0) + 1

        # Build and validate candidate list
        valid_candidates = []
        for candidate_pid, numerator in co_counts.items():
            if numerator <= 0:
                continue

            # Fetch live product details and stock via DAL
            prod_info = dal.get_product(norm_merchant, candidate_pid)
            if not prod_info:
                continue

            available_stock = dal.get_stock(norm_merchant, candidate_pid)
            # Exclude if stock is insufficient
            if available_stock < required_quantity:
                continue

            probability = numerator / denominator
            is_compat, reason = self.check_compatibility(selected_product, prod_info)

            valid_candidates.append({
                "p_id": candidate_pid,
                "product": prod_info,
                "available_stock": available_stock,
                "both_count": numerator,
                "denominator": denominator,
                "probability": round(probability, 6),
                "compatibility_result": is_compat,
                "compatibility_reason": reason
            })

        if not valid_candidates:
            return {
                "recommendation_available": False,
                "reason": "No valid in-stock complementary products found.",
                "selected_product": selected_product,
                "denominator": denominator,
                "candidates": []
            }

        # Dynamically apply Price Reasonableness filtering among candidates:
        # A candidate is filtered out for price when it is clearly disproportionate (ratio > 2.0)
        # AND a substantially more reasonable alternative exists (ratio <= 1.5).
        sel_price = selected_product.get("price", 0)
        has_reasonable_alt = any(
            c["compatibility_result"] and (c["product"].get("price", 0) / sel_price <= 1.5 if sel_price > 0 else True)
            for c in valid_candidates
        )

        if has_reasonable_alt and sel_price > 0:
            for c in valid_candidates:
                if c["compatibility_result"]:
                    ratio = c["product"].get("price", 0) / sel_price
                    if ratio > 2.0:
                        c["compatibility_result"] = False
                        c["compatibility_reason"] = f"Incompatible: Price ratio is disproportionately high ({ratio:.2f} > 2.0) and a more reasonable alternative exists."

        # Filter out clearly incompatible candidates for ranking
        compat_candidates = [c for c in valid_candidates if c["compatibility_result"]]

        if not compat_candidates:
            return {
                "recommendation_available": False,
                "reason": "No compatible complementary products found.",
                "selected_product": selected_product,
                "denominator": denominator,
                "candidates": valid_candidates
            }

        # Deterministic Ranking Rules for compatible items:
        # 1. Highest P(B|A)
        # 2. Higher co-occurrence count (numerator)
        # 3. Higher product rating
        # 4. Lower unit price
        # 5. Higher available stock
        compat_candidates.sort(
            key=lambda c: (
                -c["probability"],
                -c["both_count"],
                -c["product"].get("rating", 0.0),
                c["product"].get("price", 0),
                -c["available_stock"]
            )
        )

        best_candidate = compat_candidates[0]
        return {
            "recommendation_available": True,
            "selected_product": selected_product,
            "denominator": denominator,
            "top_candidate": best_candidate,
            "all_candidates": valid_candidates
        }

    def generate_llm_message(
        self,
        merchant_name: str,
        selected_product: Dict[str, Any],
        recommended_candidate: Dict[str, Any]
    ) -> Tuple[str, str, str]:
        """
        Calls OpenRouter (qwen/qwen3-4b:free) to craft a persuasive, concise
        recommendation message based on deterministic context.

        Returns: (recommendation_message, model_used, llm_status)
        """
        api_key = get_openrouter_api_key()
        model_name = get_agent2_model()

        rec_prod = recommended_candidate["product"]
        sel_name = selected_product.get("p_name", "your selected item")
        rec_name = rec_prod.get("p_name", "this complementary item")
        rec_price = rec_prod.get("price", 0)
        percentage = round(recommended_candidate["probability"] * 100, 1)

        if not api_key:
            return None, model_name, "pending_api_key"

        # Structured prompt for Agent 2
        system_prompt = (
            "You are Agent 2 (Sales Improvement Agent) in an e-commerce platform. "
            "Your role is to write a single, friendly, natural recommendation sentence "
            "suggesting a verified complementary product to the buyer.\n\n"
            "CRITICAL RULES:\n"
            "1. Recommend ONLY the exact complementary product provided in the context.\n"
            "2. Do NOT mention other products or suggest alternatives.\n"
            "3. Naturally mention that customers frequently purchase these items together.\n"
            "4. Keep the message concise (1-2 sentences max).\n"
            "5. Conclude by asking if the customer would like to add it to their cart."
        )

        user_content = json.dumps({
            "selected_product": sel_name,
            "merchant": merchant_name.title(),
            "recommended_product": rec_name,
            "price_inr": rec_price,
            "category": rec_prod.get("category"),
            "co_purchase_percentage": f"{percentage}% of buyers purchased both"
        })

        request_body = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "temperature": 0.3,
            "max_tokens": 400
        }

        url = f"{OPENROUTER_BASE_URL}/chat/completions"
        req = urllib.request.Request(
            url,
            data=json.dumps(request_body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:8000",
                "X-Title": "Agentic Commerce Platform - Agent 2"
            },
            method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=20) as response:
                if response.status == 200:
                    resp_data = json.loads(response.read().decode("utf-8"))
                    choices = resp_data.get("choices", [])
                    if choices:
                        c0 = choices[0]
                        msg = c0.get("message", {})
                        raw_content = msg.get("content")
                        if isinstance(raw_content, str) and raw_content.strip():
                            return raw_content.strip(), model_name, "success"

                        # Handle edge case where completion text is returned directly
                        raw_text = c0.get("text")
                        if isinstance(raw_text, str) and raw_text.strip():
                            return raw_text.strip(), model_name, "success"

                        finish_reason = c0.get("finish_reason")
                        if finish_reason == "length":
                            return None, model_name, "llm_token_limit_reached"

                    return None, model_name, "llm_empty_response"
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8") if hasattr(e, "read") else str(e)
            return None, model_name, f"llm_http_error: {e.code} - {err_body[:120]}"
        except Exception as e:
            return None, model_name, f"llm_error: {str(e)}"

    def get_recommendation(
        self,
        merchant: str,
        selected_product_id: str,
        current_cart_items: Optional[List[str]] = None,
        required_quantity: int = 1
    ) -> Agent2RecommendResponse:
        """
        Main entry point: performs deterministic co-purchase calculation,
        validates inventory, generates LLM copy, and returns structured response.
        """
        stats = self.calculate_co_purchase_statistics(
            merchant=merchant,
            selected_product_id=selected_product_id,
            current_cart_items=current_cart_items,
            required_quantity=required_quantity
        )

        norm_merchant = merchant.strip().lower()
        selected_pid = selected_product_id.strip()
        selected_prod = stats.get("selected_product") or {}
        selected_pname = selected_prod.get("p_name", selected_pid)

        if not stats.get("recommendation_available"):
            return Agent2RecommendResponse(
                recommendation_available=False,
                merchant=norm_merchant,
                selected_product_id=selected_pid,
                selected_product_name=selected_pname,
                orders_with_selected=stats.get("denominator", 0),
                orders_with_both=0,
                recommendation_message="No complementary product recommendation available for this selection.",
                model_used=get_agent2_model(),
                llm_status="not_applicable",
                audit_evidence={
                    "timestamp": datetime.utcnow().isoformat(),
                    "merchant": norm_merchant,
                    "selected_product_id": selected_pid,
                    "reason": stats.get("reason", "No valid co-purchase candidate"),
                    "denominator": stats.get("denominator", 0)
                }
            )

        top_candidate = stats["top_candidate"]
        rec_prod = top_candidate["product"]

        # Generate recommendation message (via OpenRouter or structured template)
        msg_text, model_used, llm_status = self.generate_llm_message(
            merchant_name=norm_merchant,
            selected_product=selected_prod,
            recommended_candidate=top_candidate
        )

        audit_evidence = {
            "timestamp": datetime.utcnow().isoformat(),
            "merchant": norm_merchant,
            "selected_product_id": selected_pid,
            "selected_product_name": selected_pname,
            "recommended_product_id": rec_prod.get("p_id"),
            "recommended_product_name": rec_prod.get("p_name"),
            "denominator_orders_with_selected": top_candidate["denominator"],
            "numerator_orders_with_both": top_candidate["both_count"],
            "co_purchase_probability": top_candidate["probability"],
            "available_stock": top_candidate["available_stock"],
            "unit_price": rec_prod.get("price"),
            "model_used": model_used,
            "llm_status": llm_status,
            "candidate_rankings": [
                {
                    "product_id": c["p_id"],
                    "product_name": c["product"].get("p_name"),
                    "category": c["product"].get("category"),
                    "price": c["product"].get("price"),
                    "co_purchase_probability": c["probability"],
                    "compatibility_result": c.get("compatibility_result"),
                    "compatibility_reason": c.get("compatibility_reason")
                }
                for c in stats.get("all_candidates", [])
            ]
        }

        return Agent2RecommendResponse(
            recommendation_available=True,
            merchant=norm_merchant,
            selected_product_id=selected_pid,
            selected_product_name=selected_pname,
            recommended_product_id=rec_prod.get("p_id"),
            recommended_product_name=rec_prod.get("p_name"),
            category=rec_prod.get("category"),
            description=rec_prod.get("description"),
            price=rec_prod.get("price"),
            rating=rec_prod.get("rating"),
            available_stock=top_candidate["available_stock"],
            co_purchase_probability=top_candidate["probability"],
            orders_with_selected=top_candidate["denominator"],
            orders_with_both=top_candidate["both_count"],
            recommendation_message=msg_text,
            model_used=model_used,
            llm_status=llm_status,
            audit_evidence=audit_evidence
        )


# Global singleton instance for Agent 2
agent2_service = SalesImprovementAgent()
