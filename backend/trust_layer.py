from typing import Dict, Any, List, Optional
from backend.data_access import dal

VALID_MERCHANTS = {"shopnest", "cartwave"}

class TrustLayer:
    @staticmethod
    def validate_primary_product(winner: Optional[Dict[str, Any]], quantity: int) -> Dict[str, Any]:
        """
        Validates:
        1. Merchant validity.
        2. Product ID validity.
        3. Product existence under merchant catalog.
        4. Current catalog price consistency.
        5. Stock availability for the requested quantity.
        """
        if not winner:
            return {
                "approved": False,
                "action": "REJECT",
                "reason": "PRODUCT_NOT_FOUND"
            }

        # 1. Validate quantity
        if quantity <= 0:
            return {
                "approved": False,
                "action": "REJECT",
                "reason": "INVALID_QUANTITY"
            }

        # 2. Validate merchant
        m = winner.get("merchant", "").strip().lower()
        if m not in VALID_MERCHANTS:
            return {
                "approved": False,
                "action": "REJECT",
                "merchant": m,
                "reason": "MERCHANT_NOT_FOUND"
            }

        # 3. Validate product ID
        p_id = winner.get("p_id")
        if not p_id:
            return {
                "approved": False,
                "action": "REJECT",
                "reason": "PRODUCT_NOT_FOUND"
            }

        # 4. Validate product existence under merchant
        prod = dal.get_product(m, p_id)
        if not prod:
            return {
                "approved": False,
                "action": "REJECT",
                "product_id": p_id,
                "merchant": m,
                "reason": "PRODUCT_NOT_FOUND"
            }

        # 5. Validate product ID identity consistency
        if prod.get("p_id") != p_id:
            return {
                "approved": False,
                "action": "REJECT",
                "product_id": p_id,
                "merchant": m,
                "reason": "PRODUCT_ID_MISMATCH"
            }

        # 6. Validate price consistency against current catalog
        catalog_price = prod.get("price")
        if winner.get("price") != catalog_price:
            return {
                "approved": False,
                "action": "REJECT",
                "product_id": p_id,
                "merchant": m,
                "price": winner.get("price"),
                "reason": "PRICE_MISMATCH"
            }

        # 7. Validate stock availability
        stock = dal.get_stock(m, p_id)
        if stock is None:
            return {
                "approved": False,
                "action": "REJECT",
                "product_id": p_id,
                "merchant": m,
                "reason": "PRODUCT_NOT_FOUND"
            }
        if stock < quantity:
            return {
                "approved": False,
                "action": "REJECT",
                "product_id": p_id,
                "merchant": m,
                "price": catalog_price,
                "reason": "PRODUCT_OUT_OF_STOCK"
            }

        return {
            "approved": True,
            "action": "ADD_TO_CART",
            "product_id": p_id,
            "merchant": m,
            "price": catalog_price,
            "reason": "VALID_PRODUCT_STOCK_AND_PRICE"
        }

    @staticmethod
    def validate_complementary_product(rec: Optional[Dict[str, Any]], user_consent: bool) -> Dict[str, Any]:
        """
        Validates optional complementary products using explicit user consent,
        merchant existence, catalog presence, price consistency, and stock.
        """
        if not rec:
            return {
                "approved": False,
                "action": "REJECT",
                "reason": "PRODUCT_NOT_FOUND"
            }

        p_id = rec.get("recommended_product_id")
        m = rec.get("merchant", "").strip().lower()

        # 1. Validate explicit user consent
        if not user_consent:
            return {
                "approved": False,
                "action": "REJECT",
                "product_id": p_id,
                "merchant": m,
                "price": rec.get("price"),
                "user_consent": False,
                "reason": "USER_DECLINED"
            }

        # 2. Validate merchant
        if m not in VALID_MERCHANTS:
            return {
                "approved": False,
                "action": "REJECT",
                "merchant": m,
                "user_consent": True,
                "reason": "MERCHANT_NOT_FOUND"
            }

        # 3. Validate product ID
        if not p_id:
            return {
                "approved": False,
                "action": "REJECT",
                "user_consent": True,
                "reason": "PRODUCT_NOT_FOUND"
            }

        # 4. Validate product existence
        prod = dal.get_product(m, p_id)
        if not prod:
            return {
                "approved": False,
                "action": "REJECT",
                "product_id": p_id,
                "merchant": m,
                "user_consent": True,
                "reason": "PRODUCT_NOT_FOUND"
            }

        # 5. Validate product ID identity
        if prod.get("p_id") != p_id:
            return {
                "approved": False,
                "action": "REJECT",
                "product_id": p_id,
                "merchant": m,
                "user_consent": True,
                "reason": "PRODUCT_ID_MISMATCH"
            }

        # 6. Validate price consistency against catalog
        catalog_price = prod.get("price")
        if rec.get("price") != catalog_price:
            return {
                "approved": False,
                "action": "REJECT",
                "product_id": p_id,
                "merchant": m,
                "price": rec.get("price"),
                "user_consent": True,
                "reason": "PRICE_MISMATCH"
            }

        # 7. Validate current stock
        stock = dal.get_stock(m, p_id)
        if stock is None:
            return {
                "approved": False,
                "action": "REJECT",
                "product_id": p_id,
                "merchant": m,
                "user_consent": True,
                "reason": "PRODUCT_NOT_FOUND"
            }
        if stock < 1:
            return {
                "approved": False,
                "action": "REJECT",
                "product_id": p_id,
                "merchant": m,
                "price": catalog_price,
                "user_consent": True,
                "reason": "PRODUCT_OUT_OF_STOCK"
            }

        return {
            "approved": True,
            "action": "ADD_COMPLEMENTARY_TO_CART",
            "product_id": p_id,
            "merchant": m,
            "price": catalog_price,
            "user_consent": True,
            "reason": "USER_EXPLICITLY_APPROVED"
        }

    @staticmethod
    def validate_cart_consistency(cart: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Audits every item currently present in the cart.
        """
        if not cart:
            return {
                "approved": True,
                "reason": "CART_VALIDATION_SUCCESS"
            }


        for item in cart:
            p_id = item.get("p_id")
            m = item.get("merchant", "").strip().lower()
            quantity = item.get("quantity", 1)

            # 1. Validate merchant
            if m not in VALID_MERCHANTS:
                return {
                    "approved": False,
                    "reason": "MERCHANT_NOT_FOUND"
                }



            # 2. Validate product ID
            if not p_id:
                return {
                    "approved": False,
                    "reason": "PRODUCT_NOT_FOUND"
                }

            # 3. Validate product exists under merchant catalog
            prod = dal.get_product(m, p_id)
            if not prod:
                return {
                    "approved": False,
                    "reason": "PRODUCT_NOT_FOUND"
                }

            # 4. Validate product ID identity
            if prod.get("p_id") != p_id:
                return {
                    "approved": False,
                    "reason": "PRODUCT_ID_MISMATCH"
                }

            # 5. Validate merchant consistency
            if item.get("merchant") != prod.get("merchant", m):
                return {
                    "approved": False,
                    "reason": "MERCHANT_MISMATCH"
                }

            # 6. Validate price consistency
            catalog_price = prod.get("price")
            if item.get("price") != catalog_price:
                return {
                    "approved": False,
                    "reason": "PRICE_MISMATCH"
                }

            # 7. Validate quantity
            if quantity <= 0:
                return {
                    "approved": False,
                    "reason": "INVALID_QUANTITY"
                }

            # 8. Validate stock availability
            stock = dal.get_stock(m, p_id)
            if stock is None:
                return {
                    "approved": False,
                    "reason": "PRODUCT_NOT_FOUND"
                }
            if stock < quantity:
                return {
                    "approved": False,
                    "reason": "PRODUCT_OUT_OF_STOCK"
                }

        return {
            "approved": True,
            "reason": "CART_VALIDATION_SUCCESS"
        }
