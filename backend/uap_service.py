"""
UAP (Universal Agent Protocol) Service — Version 1.0
Implements standardized agent-to-agent communication, discovery, and intent exchange
aligned with NPCI / open agentic commerce interoperability standards.
"""

import hmac
import hashlib
import json
import time
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from backend.schemas import UAPMessageEnvelope, UAPAgentManifest

UAP_PROTOCOL_VERSION = "UAP/1.0"
UAP_SIGNING_SECRET = "uap_universal_agent_protocol_secret_2026"

# Registered Agent Directory
AGENT_REGISTRY: Dict[str, Dict[str, Any]] = {
    "agent1_buyer": {
        "agent_id": "agent1_buyer",
        "name": "AI Buyer Orchestrator Agent",
        "role": "BUYER_ORCHESTRATOR",
        "protocol_version": UAP_PROTOCOL_VERSION,
        "endpoints": {
            "chat": "/api/agent1/chat",
            "uap_inbox": "/api/uap/message"
        },
        "capabilities": [
            "natural_language_understanding",
            "cross_merchant_search",
            "deterministic_scoring",
            "cart_orchestration",
            "mandate_delegation"
        ],
        "supported_intents": [
            "INTENT_DISCOVERY",
            "QUERY_OFFER",
            "REQUEST_CROSS_SELL",
            "MANDATE_SYNC"
        ],
        "public_key": "uap_pub_agent1_buyer_key_2026"
    },
    "agent2_shopnest": {
        "agent_id": "agent2_shopnest",
        "name": "ShopNest Merchant Sales Optimizer Agent",
        "role": "MERCHANT_SALES_OPTIMIZER",
        "protocol_version": UAP_PROTOCOL_VERSION,
        "endpoints": {
            "recommend": "/api/agent2/recommend",
            "uap_inbox": "/api/uap/message"
        },
        "capabilities": [
            "co_purchase_probability_mining",
            "realtime_inventory_gating",
            "cross_sell_generation",
            "merchant_catalog_inspection"
        ],
        "supported_intents": [
            "QUERY_OFFER",
            "RECOMMEND_CROSS_SELL",
            "INVENTORY_CHECK"
        ],
        "public_key": "uap_pub_agent2_shopnest_key_2026"
    },
    "agent2_cartwave": {
        "agent_id": "agent2_cartwave",
        "name": "CartWave Merchant Sales Optimizer Agent",
        "role": "MERCHANT_SALES_OPTIMIZER",
        "protocol_version": UAP_PROTOCOL_VERSION,
        "endpoints": {
            "recommend": "/api/agent2/recommend",
            "uap_inbox": "/api/uap/message"
        },
        "capabilities": [
            "co_purchase_probability_mining",
            "realtime_inventory_gating",
            "cross_sell_generation",
            "merchant_catalog_inspection"
        ],
        "supported_intents": [
            "QUERY_OFFER",
            "RECOMMEND_CROSS_SELL",
            "INVENTORY_CHECK"
        ],
        "public_key": "uap_pub_agent2_cartwave_key_2026"
    }
}


class UAPService:
    def __init__(self, secret: str = UAP_SIGNING_SECRET):
        self.secret = secret

    def sign_payload(self, sender_id: str, recipient_id: str, intent: str, timestamp: str, payload: Dict[str, Any]) -> str:
        """Generates deterministic HMAC-SHA256 signature for a UAP envelope."""
        canonical_str = f"{UAP_PROTOCOL_VERSION}|{sender_id}|{recipient_id}|{intent}|{timestamp}|{json.dumps(payload, sort_keys=True)}"
        return hmac.new(self.secret.encode("utf-8"), canonical_str.encode("utf-8"), hashlib.sha256).hexdigest()

    def create_envelope(
        self,
        sender_id: str,
        recipient_id: str,
        intent: str,
        payload: Dict[str, Any]
    ) -> UAPMessageEnvelope:
        """Constructs and signs a standardized UAP message envelope."""
        now_iso = datetime.now(timezone.utc).isoformat()
        msg_id = f"uap_msg_{uuid.uuid4().hex[:16]}"
        sig = self.sign_payload(sender_id, recipient_id, intent, now_iso, payload)

        return UAPMessageEnvelope(
            protocol_version=UAP_PROTOCOL_VERSION,
            message_id=msg_id,
            sender_id=sender_id,
            recipient_id=recipient_id,
            intent=intent,
            timestamp=now_iso,
            payload=payload,
            signature=sig
        )

    def verify_envelope(self, envelope: UAPMessageEnvelope) -> bool:
        """Verifies integrity and cryptographic signature of a UAP message envelope."""
        if envelope.protocol_version != UAP_PROTOCOL_VERSION:
            return False
        if not envelope.signature:
            return False
        expected_sig = self.sign_payload(
            envelope.sender_id,
            envelope.recipient_id,
            envelope.intent,
            envelope.timestamp,
            envelope.payload
        )
        return hmac.compare_digest(expected_sig, envelope.signature)

    def get_discovery_manifest(self) -> Dict[str, Any]:
        """Returns the complete Universal Agent Protocol manifest for agent discovery."""
        return {
            "protocol": "Universal Agent Protocol (UAP)",
            "version": UAP_PROTOCOL_VERSION,
            "specification": "https://standards.npci.org.in/uap/v1",
            "host": "Agentic Commerce Platform",
            "active_agents": list(AGENT_REGISTRY.values())
        }

    def get_agent_manifest(self, agent_id: str) -> Optional[UAPAgentManifest]:
        """Returns individual agent discovery manifest."""
        data = AGENT_REGISTRY.get(agent_id)
        if not data:
            return None
        return UAPAgentManifest(**data)

    def dispatch_message(self, envelope: UAPMessageEnvelope) -> UAPMessageEnvelope:
        """
        Processes an incoming UAP message envelope and dispatches to appropriate internal service.
        Returns a signed UAP response envelope.
        """
        if not self.verify_envelope(envelope):
            return self.create_envelope(
                sender_id="uap_gateway",
                recipient_id=envelope.sender_id,
                intent="ERROR_RESPONSE",
                payload={"error": "INVALID_UAP_SIGNATURE", "message": "Cryptographic signature mismatch in UAP envelope"}
            )

        intent = envelope.intent
        payload = envelope.payload

        if intent == "QUERY_OFFER":
            from backend.scoring_engine import scoring_engine
            p_id = payload.get("product_id")
            priority = payload.get("priority", "best_balance")
            budget = payload.get("budget")
            size = payload.get("size")
            color = payload.get("color")
            qty = payload.get("quantity", 1)

            try:
                offers = scoring_engine.compare_merchants_for_product(
                    p_id=p_id,
                    priority=priority,
                    budget=budget,
                    size=size,
                    color=color,
                    required_quantity=qty
                )
                response_payload = {
                    "status": "SUCCESS",
                    "product_id": p_id,
                    "offers": [o.dict() for o in offers]
                }
            except Exception as e:
                response_payload = {"status": "ERROR", "error": str(e)}

            return self.create_envelope(
                sender_id=envelope.recipient_id,
                recipient_id=envelope.sender_id,
                intent="OFFER_RESPONSE",
                payload=response_payload
            )

        elif intent == "RECOMMEND_CROSS_SELL":
            from backend.agent2_service import agent2_service
            merchant = payload.get("merchant")
            p_id = payload.get("product_id")
            cart_items = payload.get("current_cart_items", [])
            qty = payload.get("quantity", 1)

            try:
                rec_res = agent2_service.get_recommendation(
                    merchant=merchant,
                    selected_product_id=p_id,
                    current_cart_items=cart_items,
                    required_quantity=qty
                )
                response_payload = {
                    "status": "SUCCESS",
                    "recommendation": rec_res.dict()
                }
            except Exception as e:
                response_payload = {"status": "ERROR", "error": str(e)}

            return self.create_envelope(
                sender_id=envelope.recipient_id,
                recipient_id=envelope.sender_id,
                intent="CROSS_SELL_RESPONSE",
                payload=response_payload
            )

        elif intent == "INVENTORY_CHECK":
            from backend.data_access import dal
            merchant = payload.get("merchant")
            p_id = payload.get("product_id")
            stock = dal.get_stock(merchant, p_id) if (merchant and p_id) else None
            return self.create_envelope(
                sender_id=envelope.recipient_id,
                recipient_id=envelope.sender_id,
                intent="INVENTORY_RESPONSE",
                payload={"merchant": merchant, "product_id": p_id, "stock": stock, "available": (stock is not None and stock > 0)}
            )

        else:
            return self.create_envelope(
                sender_id="uap_gateway",
                recipient_id=envelope.sender_id,
                intent="UNKNOWN_INTENT_RESPONSE",
                payload={"error": "UNSUPPORTED_INTENT", "intent_received": intent}
            )


# Global UAP service singleton
uap_service = UAPService()
