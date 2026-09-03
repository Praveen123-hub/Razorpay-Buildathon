"""
Comprehensive Test Suite for AP2 and UAP Protocol Integration
Tests:
1. UAP discovery manifest and agent lookup
2. UAP message envelope cryptographic signing and intent dispatch
3. AP2 discovery manifest
4. AP2 delegation mandate generation and SHA-256 cart hashing
5. AP2 spending bound and expiration enforcement via Trust Layer Guardrail #8
6. AP2 settlement receipt issuance
7. End-to-end Agent 1 + Agent 2 conversation with UAP trace & AP2 Mandate in Audit
"""

import sys
import unittest
from datetime import datetime, timezone, timedelta
from fastapi.testclient import TestClient

from backend.main import app
from backend.uap_service import uap_service
from backend.ap2_service import ap2_service
from backend.trust_layer import TrustLayer
from backend.agent1_service import agent1_service
from backend.schemas import UAPMessageEnvelope, AP2MandateRequest

client = TestClient(app)


class TestProtocols(unittest.TestCase):

    def setUp(self):
        # Reset any test state
        pass

    def test_01_uap_discovery_manifest(self):
        """Test /.well-known/agent.json and /api/uap/discovery"""
        resp = client.get("/.well-known/agent.json")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["version"], "UAP/1.0")
        self.assertIn("active_agents", data)
        agent_ids = [a["agent_id"] for a in data["active_agents"]]
        self.assertIn("agent1_buyer", agent_ids)
        self.assertIn("agent2_shopnest", agent_ids)

        resp2 = client.get("/api/uap/agents/agent1_buyer")
        self.assertEqual(resp2.status_code, 200)
        self.assertEqual(resp2.json()["role"], "BUYER_ORCHESTRATOR")

    def test_02_uap_message_exchange(self):
        """Test inter-agent UAP message signing and dispatch"""
        envelope = uap_service.create_envelope(
            sender_id="agent1_buyer",
            recipient_id="agent2_shopnest",
            intent="RECOMMEND_CROSS_SELL",
            payload={
                "merchant": "shopnest",
                "product_id": "SHOE_001",
                "current_cart_items": ["SHOE_001"],
                "quantity": 1
            }
        )
        self.assertTrue(uap_service.verify_envelope(envelope))

        # Send envelope through FastAPI endpoint
        resp = client.post("/api/uap/message", json=envelope.dict())
        self.assertEqual(resp.status_code, 200)
        resp_envelope = resp.json()
        self.assertEqual(resp_envelope["intent"], "CROSS_SELL_RESPONSE")
        self.assertIn("recommendation", resp_envelope["payload"])
        self.assertTrue(resp_envelope["payload"]["recommendation"]["recommendation_available"])

    def test_03_ap2_discovery_manifest(self):
        """Test /.well-known/ap2.json"""
        resp = client.get("/.well-known/ap2.json")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["protocol"], "Agent Payment Protocol (AP2)")
        self.assertEqual(data["version"], "AP2/1.0")
        self.assertIn("INR", data["supported_currencies"])
        self.assertTrue(data["delegation_features"]["bounded_budget_mandates"])

    def test_04_ap2_mandate_creation_and_cart_hash(self):
        """Test AP2 delegation mandate creation, SHA-256 fingerprinting, and signature"""
        cart = [
            {"p_id": "SHOE_001", "p_name": "Nike Air Zoom", "merchant": "shopnest", "price": 2400, "quantity": 1}
        ]
        mandate = ap2_service.create_delegation_mandate(
            session_id="test_sess_01",
            cart_items=cart,
            max_amount=3000,
            authorized_merchants=["shopnest"]
        )
        self.assertTrue(mandate.mandate_id.startswith("ap2_man_"))
        self.assertEqual(mandate.status, "AUTHORIZED")
        self.assertEqual(mandate.max_amount, 3000)
        self.assertTrue(len(mandate.cart_hash) == 64)  # SHA-256 is 64 hex chars
        self.assertTrue(len(mandate.signature) == 64)

        # Test verification through Trust Layer Guardrail #8
        val_res = TrustLayer.validate_ap2_mandate(
            mandate_id=mandate.mandate_id,
            cart_items=cart,
            expected_total=2400,
            merchant="shopnest"
        )
        self.assertTrue(val_res["approved"])
        self.assertEqual(val_res["action"], "ALLOW_SETTLEMENT")

    def test_05_ap2_guardrail_bounds_enforcement(self):
        """Test Trust Layer rejects claims exceeding max authorized amount or tampered cart"""
        cart = [
            {"p_id": "SHOE_001", "p_name": "Nike Air Zoom", "merchant": "shopnest", "price": 2400, "quantity": 1}
        ]
        mandate = ap2_service.create_delegation_mandate(
            session_id="test_sess_bounds",
            cart_items=cart,
            max_amount=2500,
            authorized_merchants=["shopnest"]
        )

        # Attempt to claim ₹3000 (exceeding ₹2500 max ceiling)
        val_res = TrustLayer.validate_ap2_mandate(
            mandate_id=mandate.mandate_id,
            cart_items=cart,
            expected_total=3000,
            merchant="shopnest"
        )
        self.assertFalse(val_res["approved"])
        self.assertEqual(val_res["action"], "REJECT")
        self.assertIn("CLAIM_EXCEEDS_MANDATE_LIMIT", val_res["reason"])

        # Attempt to claim with tampered cart items (hash mismatch)
        tampered_cart = [
            {"p_id": "SHOE_001", "p_name": "Nike Air Zoom", "merchant": "shopnest", "price": 9999, "quantity": 1}
        ]
        val_res_tampered = TrustLayer.validate_ap2_mandate(
            mandate_id=mandate.mandate_id,
            cart_items=tampered_cart,
            expected_total=2400,
            merchant="shopnest"
        )
        self.assertFalse(val_res_tampered["approved"])
        self.assertEqual(val_res_tampered["action"], "REJECT")

    def test_06_ap2_settlement_receipt(self):
        """Test AP2 settlement receipt creation and receipt hash integrity"""
        cart = [{"p_id": "SHOE_001", "merchant": "shopnest", "price": 2400, "quantity": 1}]
        mandate = ap2_service.create_delegation_mandate(
            session_id="test_sess_settle",
            cart_items=cart,
            max_amount=2500
        )
        receipt = ap2_service.issue_settlement_receipt(
            mandate_id=mandate.mandate_id,
            session_id="test_sess_settle",
            merchant="shopnest",
            amount_paid=2400,
            razorpay_payment_id="pay_test_12345"
        )
        self.assertTrue(receipt.receipt_id.startswith("ap2_rec_"))
        self.assertEqual(receipt.status, "SETTLED")
        self.assertEqual(receipt.amount_paid, 2400)
        self.assertTrue(len(receipt.receipt_hash) == 64)

        # Mandate status should now be CLAIMED
        updated_mandate = ap2_service.get_mandate(mandate.mandate_id)
        self.assertEqual(updated_mandate.status, "CLAIMED")

    def test_07_end_to_end_uap_ap2_flow(self):
        """Test full conversation flow generating UAP inter-agent trace and AP2 Mandate in session audit"""
        sess_id = "test_e2e_proto_session"

        # 1. Ask for running shoes with all slots filled (product, size, color, budget, priority)
        resp1 = client.post("/api/agent1/chat", json={
            "message": "I want black running shoes size 9 under 3000 cheapest",
            "session_id": sess_id
        })
        self.assertEqual(resp1.status_code, 200)
        self.assertEqual(resp1.json()["current_state"], "AWAITING_MAIN_CART_CONFIRMATION")

        # 2. Confirm adding main product (triggers Step 5 via UAP to Agent 2)
        resp2 = client.post("/api/agent1/chat", json={
            "message": "yes",
            "session_id": sess_id
        })
        self.assertEqual(resp2.status_code, 200)

        # 3. Reject complementary recommendation to transition to final cart & AP2 mandate binding
        resp3 = client.post("/api/agent1/chat", json={
            "message": "no",
            "session_id": sess_id
        })
        self.assertEqual(resp3.status_code, 200)
        self.assertEqual(resp3.json()["current_state"], "AWAITING_ORDER_CONFIRMATION")

        # 4. Fetch Audit log and verify UAP & AP2 details in page 5
        resp_audit = client.get(f"/api/session/{sess_id}/audit")
        self.assertEqual(resp_audit.status_code, 200)
        audit = resp_audit.json()
        self.assertIn("page5_protocols_audit", audit)
        p5 = audit["page5_protocols_audit"]
        self.assertEqual(p5["uap_protocol"]["version"], "UAP/1.0")
        self.assertGreater(p5["uap_protocol"]["message_count"], 0)
        self.assertEqual(p5["ap2_protocol"]["version"], "AP2/1.0")
        self.assertIsNotNone(p5["ap2_protocol"]["mandate_id"])
        self.assertIsNotNone(p5["ap2_protocol"]["signature"])



if __name__ == "__main__":
    unittest.main()
