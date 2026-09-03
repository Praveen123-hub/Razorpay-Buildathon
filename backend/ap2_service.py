"""
AP2 (Agent Payment Protocol) Service — Version 1.0
Implements standardized agent payment discovery, bounded cryptographic mandates,
integrity cart hashing, payment claiming, and verifiable settlement receipts.
"""

import hmac
import hashlib
import json
import time
import uuid
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone, timedelta

from backend.schemas import AP2Mandate, AP2SettlementReceipt

AP2_PROTOCOL_VERSION = "AP2/1.0"
AP2_SIGNING_SECRET = "ap2_agent_payment_protocol_secret_key_2026"

# In-memory registry of active mandates
_MANDATES_REGISTRY: Dict[str, AP2Mandate] = {}
_SETTLEMENT_RECEIPTS: Dict[str, AP2SettlementReceipt] = {}


class AP2Service:
    def __init__(self, secret: str = AP2_SIGNING_SECRET):
        self.secret = secret

    @staticmethod
    def compute_cart_hash(cart_items: List[Dict[str, Any]]) -> str:
        """
        Computes deterministic SHA-256 fingerprint for cart contents.
        Normalizes p_id, quantity, merchant, price to prevent tampering.
        """
        normalized_items = []
        for item in cart_items:
            normalized_items.append({
                "p_id": str(item.get("p_id", "")).strip(),
                "merchant": str(item.get("merchant", "")).strip().lower(),
                "price": int(item.get("price", 0)),
                "quantity": int(item.get("quantity", 1))
            })
        # Sort items deterministically by p_id and merchant
        normalized_items.sort(key=lambda x: (x["merchant"], x["p_id"]))
        canonical_bytes = json.dumps(normalized_items, sort_keys=True).encode("utf-8")
        return hashlib.sha256(canonical_bytes).hexdigest()

    def sign_mandate(
        self,
        mandate_id: str,
        session_id: str,
        max_amount: int,
        authorized_merchants: List[str],
        created_at: str,
        expires_at: str,
        cart_hash: str
    ) -> str:
        """Generates HMAC-SHA256 signature binding mandate constraints."""
        canonical_str = (
            f"{AP2_PROTOCOL_VERSION}|{mandate_id}|{session_id}|{max_amount}|"
            f"{','.join(sorted(authorized_merchants))}|{created_at}|{expires_at}|{cart_hash}"
        )
        return hmac.new(self.secret.encode("utf-8"), canonical_str.encode("utf-8"), hashlib.sha256).hexdigest()

    def create_delegation_mandate(
        self,
        session_id: str,
        cart_items: List[Dict[str, Any]],
        max_amount: Optional[int] = None,
        authorized_merchants: Optional[List[str]] = None,
        validity_minutes: int = 60,
        user_id: Optional[int] = None
    ) -> AP2Mandate:
        """
        Creates a new cryptographically bound AP2 delegation mandate for the session.
        If max_amount is not explicitly passed, defaults to the cart total with safety buffer.
        """
        if not cart_items:
            raise ValueError("Cannot create AP2 mandate for empty cart.")

        cart_total = sum(int(i.get("price", 0)) * int(i.get("quantity", 1)) for i in cart_items)
        if max_amount is None or max_amount < cart_total:
            max_amount = cart_total

        if authorized_merchants is None:
            # Extract unique merchants from current cart
            authorized_merchants = list(set(str(i.get("merchant", "")).strip().lower() for i in cart_items if i.get("merchant")))
            if not authorized_merchants:
                authorized_merchants = ["shopnest", "cartwave"]

        now = datetime.now(timezone.utc)
        expires = now + timedelta(minutes=validity_minutes)
        created_at_str = now.isoformat()
        expires_at_str = expires.isoformat()

        mandate_id = f"ap2_man_{uuid.uuid4().hex[:16]}"
        cart_hash = self.compute_cart_hash(cart_items)

        signature = self.sign_mandate(
            mandate_id=mandate_id,
            session_id=session_id,
            max_amount=max_amount,
            authorized_merchants=authorized_merchants,
            created_at=created_at_str,
            expires_at=expires_at_str,
            cart_hash=cart_hash
        )

        mandate = AP2Mandate(
            mandate_id=mandate_id,
            session_id=session_id,
            user_id=user_id,
            agent_id="agent1_buyer",
            authorized_merchants=authorized_merchants,
            max_amount=max_amount,
            currency="INR",
            created_at=created_at_str,
            expires_at=expires_at_str,
            cart_hash=cart_hash,
            status="AUTHORIZED",
            signature=signature
        )

        _MANDATES_REGISTRY[mandate_id] = mandate
        return mandate

    def get_mandate(self, mandate_id: str) -> Optional[AP2Mandate]:
        """Retrieves mandate by ID."""
        return _MANDATES_REGISTRY.get(mandate_id)

    def verify_mandate_for_claim(
        self,
        mandate_id: str,
        cart_items: List[Dict[str, Any]],
        claim_amount: int,
        merchant: str
    ) -> Tuple[bool, str]:
        """
        Rigorous cryptographic and constraint verification of an AP2 mandate:
        1. Mandate existence & status
        2. Cryptographic signature check
        3. Merchant authorization check
        4. Expiration check
        5. Hard budget bound check (claim_amount <= max_amount)
        6. SHA-256 Cart Fingerprint matching
        """
        mandate = self.get_mandate(mandate_id)
        if not mandate:
            return False, "MANDATE_NOT_FOUND"

        if mandate.status != "AUTHORIZED":
            return False, f"MANDATE_INVALID_STATUS_{mandate.status}"

        # 1. Cryptographic signature verification
        expected_sig = self.sign_mandate(
            mandate_id=mandate.mandate_id,
            session_id=mandate.session_id,
            max_amount=mandate.max_amount,
            authorized_merchants=mandate.authorized_merchants,
            created_at=mandate.created_at,
            expires_at=mandate.expires_at,
            cart_hash=mandate.cart_hash
        )
        if not hmac.compare_digest(expected_sig, mandate.signature):
            return False, "MANDATE_SIGNATURE_TAMPERED"

        # 2. Merchant validation
        m_clean = merchant.strip().lower()
        if m_clean not in [m.lower() for m in mandate.authorized_merchants]:
            return False, f"MERCHANT_NOT_AUTHORIZED_FOR_MANDATE: {merchant}"

        # 3. Expiration validation
        try:
            expires_dt = datetime.fromisoformat(mandate.expires_at)
            if datetime.now(timezone.utc) > expires_dt:
                mandate.status = "EXPIRED"
                return False, "MANDATE_EXPIRED"
        except Exception:
            return False, "MANDATE_INVALID_EXPIRY_FORMAT"

        # 4. Spending ceiling bound validation
        if claim_amount > mandate.max_amount:
            return False, f"CLAIM_EXCEEDS_MANDATE_LIMIT: {claim_amount} > {mandate.max_amount}"

        # 5. Cart hash integrity validation
        current_cart_hash = self.compute_cart_hash(cart_items)
        if current_cart_hash != mandate.cart_hash:
            return False, "CART_INTEGRITY_TAMPERED_HASH_MISMATCH"

        return True, "MANDATE_VERIFIED"

    def issue_settlement_receipt(
        self,
        mandate_id: str,
        session_id: str,
        merchant: str,
        amount_paid: int,
        razorpay_payment_id: str
    ) -> AP2SettlementReceipt:
        """Issues an immutable, cryptographically verifiable AP2 settlement receipt."""
        mandate = self.get_mandate(mandate_id)
        if mandate:
            mandate.status = "CLAIMED"

        now_str = datetime.now(timezone.utc).isoformat()
        receipt_id = f"ap2_rec_{uuid.uuid4().hex[:16]}"
        
        # Compute receipt verification hash
        receipt_data = f"{AP2_PROTOCOL_VERSION}|{receipt_id}|{mandate_id}|{session_id}|{merchant}|{amount_paid}|{razorpay_payment_id}|{now_str}"
        receipt_hash = hashlib.sha256(receipt_data.encode("utf-8")).hexdigest()

        receipt = AP2SettlementReceipt(
            receipt_id=receipt_id,
            mandate_id=mandate_id,
            session_id=session_id,
            merchant=merchant,
            amount_paid=amount_paid,
            currency="INR",
            settled_at=now_str,
            razorpay_payment_id=razorpay_payment_id,
            status="SETTLED",
            receipt_hash=receipt_hash
        )

        _SETTLEMENT_RECEIPTS[receipt_id] = receipt
        return receipt

    def get_discovery_manifest(self) -> Dict[str, Any]:
        """Returns the complete AP2 protocol capability discovery document."""
        return {
            "protocol": "Agent Payment Protocol (AP2)",
            "version": AP2_PROTOCOL_VERSION,
            "specification": "https://standards.ap2.org/v1",
            "supported_currencies": ["INR"],
            "supported_gateways": ["Razorpay_Test_Mode"],
            "signature_algorithm": "HMAC-SHA256",
            "delegation_features": {
                "bounded_budget_mandates": True,
                "sha256_cart_integrity_verification": True,
                "single_use_settlement": True,
                "cryptographic_receipts": True
            },
            "endpoints": {
                "mandate_create": "/api/ap2/mandates",
                "payment_claim": "/api/ap2/payments/claim",
                "discovery": "/.well-known/ap2.json"
            }
        }


# Global AP2 service singleton
ap2_service = AP2Service()
