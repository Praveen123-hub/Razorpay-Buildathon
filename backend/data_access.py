import os
import json
import threading
import tempfile
import copy
from typing import List, Dict, Any, Optional
from datetime import datetime


VALID_MERCHANTS = {"shopnest", "cartwave"}


class MerchantDataError(Exception):
    """Base exception for data access layer operations."""
    pass


class MerchantNotFoundError(MerchantDataError):
    """Raised when an invalid merchant identifier is requested."""
    pass


class ProductNotFoundError(MerchantDataError):
    """Raised when a product ID does not exist in the merchant's catalog."""
    pass


class InsufficientStockError(MerchantDataError):
    """Raised when requested quantity exceeds available stock."""
    pass


class PriceMismatchError(MerchantDataError):
    """Raised when supplied price or total does not match authoritative catalog calculations."""
    pass


class MerchantDataAccess:
    """
    Merchant Data Access Layer (DAL).
    Provides unified, concurrency-safe, atomic operations over mock merchant JSON files.
    The six JSON files are the SINGLE SOURCE OF TRUTH.
    """

    def __init__(self, base_data_dir: Optional[str] = None):
        if base_data_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.base_data_dir = os.path.join(current_dir, "data")
        else:
            self.base_data_dir = os.path.abspath(base_data_dir)

        # Re-entrant locks per merchant to ensure thread-safe read/write operations
        self._locks = {
            "shopnest": threading.RLock(),
            "cartwave": threading.RLock()
        }

    def _normalize_merchant(self, merchant: str) -> str:
        norm = merchant.strip().lower()
        if norm not in VALID_MERCHANTS:
            raise MerchantNotFoundError(
                f"Invalid merchant '{merchant}'. Supported merchants are: {', '.join(sorted(VALID_MERCHANTS))}"
            )
        return norm

    def _get_file_path(self, merchant: str, file_type: str) -> str:
        merchant_name = self._normalize_merchant(merchant)
        filename = f"{file_type}.json"
        path = os.path.join(self.base_data_dir, merchant_name, filename)
        if not os.path.exists(path):
            raise MerchantDataError(f"Required data file does not exist: {path}")
        return path

    def _read_json(self, merchant: str, file_type: str) -> Any:
        m = self._normalize_merchant(merchant)
        file_path = self._get_file_path(m, file_type)
        with self._locks[m]:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)

    def _write_json_atomic(self, merchant: str, file_type: str, data: Any) -> None:
        """
        Atomically writes data to a target JSON file by writing to a temporary file
        in the same directory and replacing the destination file.
        """
        m = self._normalize_merchant(merchant)
        file_path = self._get_file_path(m, file_type)
        dir_name = os.path.dirname(file_path)

        with self._locks[m]:
            temp_name = None
            try:
                with tempfile.NamedTemporaryFile("w", dir=dir_name, delete=False, encoding="utf-8") as tf:
                    temp_name = tf.name
                    json.dump(data, tf, indent=2, ensure_ascii=False)
                    tf.flush()
                    os.fsync(tf.fileno())

                os.replace(temp_name, file_path)
            except Exception:
                if temp_name and os.path.exists(temp_name):
                    try:
                        os.remove(temp_name)
                    except OSError:
                        pass
                raise

    # -------------------------------------------------------------------------
    # Product Operations
    # -------------------------------------------------------------------------
    def get_products(self, merchant: str, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieves all products for a given merchant, optionally filtered by category."""
        products = self._read_json(merchant, "products")
        if category:
            cat_clean = category.strip().lower()
            return [p for p in products if p.get("category", "").strip().lower() == cat_clean]
        return products

    def get_product(self, merchant: str, p_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a single product by ID."""
        products = self._read_json(merchant, "products")
        p_id_clean = p_id.strip()
        for p in products:
            if p.get("p_id") == p_id_clean:
                return p
        return None

    def product_exists(self, merchant: str, p_id: str) -> bool:
        """Checks if a product exists in the merchant's catalog."""
        return self.get_product(merchant, p_id) is not None

    # -------------------------------------------------------------------------
    # Inventory Operations
    # -------------------------------------------------------------------------
    def get_inventory(self, merchant: str) -> List[Dict[str, Any]]:
        """Retrieves complete inventory list for a given merchant."""
        return self._read_json(merchant, "inventory")

    def get_stock(self, merchant: str, p_id: str) -> Optional[int]:
        """Retrieves current stock for a specific product ID."""
        inventory = self._read_json(merchant, "inventory")
        p_id_clean = p_id.strip()
        for item in inventory:
            if item.get("p_id") == p_id_clean:
                return int(item.get("stock", 0))
        return None

    def update_stock(
        self,
        merchant: str,
        p_id: str,
        stock_delta: Optional[int] = None,
        new_stock: Optional[int] = None
    ) -> int:
        """
        Updates stock for a product either by delta or by setting an exact value.
        Guarantees atomic file update and stock >= 0.
        """
        m = self._normalize_merchant(merchant)
        p_id_clean = p_id.strip()

        with self._locks[m]:
            inventory = self._read_json(m, "inventory")
            found = False
            updated_value = 0

            for item in inventory:
                if item.get("p_id") == p_id_clean:
                    found = True
                    current_stock = int(item.get("stock", 0))
                    if new_stock is not None:
                        if new_stock < 0:
                            raise ValueError(f"Stock cannot be negative: {new_stock}")
                        updated_value = new_stock
                    elif stock_delta is not None:
                        updated_value = current_stock + stock_delta
                        if updated_value < 0:
                            raise InsufficientStockError(
                                f"Cannot decrease stock below 0. Current: {current_stock}, delta: {stock_delta}"
                            )
                    else:
                        raise ValueError("Either stock_delta or new_stock must be provided")

                    item["stock"] = updated_value
                    break

            if not found:
                raise ProductNotFoundError(f"Product '{p_id}' not found in {m} inventory")

            self._write_json_atomic(m, "inventory", inventory)
            return updated_value

    # -------------------------------------------------------------------------
    # Order Operations
    # -------------------------------------------------------------------------
    def get_orders(self, merchant: str, user_only: bool = False) -> List[Dict[str, Any]]:
        """Retrieves completed orders for a merchant, optionally filtered to live user purchases."""
        orders = self._read_json(merchant, "orders")
        if user_only:
            return [o for o in orders if o.get("is_user_order") is True or o.get("order_source") == "agent_purchase"]
        return orders

    def get_order(self, merchant: str, order_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a specific order by order_id."""
        orders = self._read_json(merchant, "orders")
        order_id_clean = order_id.strip()
        for o in orders:
            if o.get("order_id") == order_id_clean:
                return o
        return None

    def append_order(self, merchant: str, order_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Appends an order through the unified transaction-safe pathway, ensuring
        inventory availability is strictly checked and deducted alongside order creation.
        """
        products = order_dict.get("products", [])
        expected_total = order_dict.get("total_amount")
        order_id = order_dict.get("order_id")
        created_at = order_dict.get("created_at")

        return self.execute_purchase_transaction(
            merchant=merchant,
            order_items=products,
            expected_total=expected_total,
            order_id=order_id,
            created_at=created_at
        )

    # -------------------------------------------------------------------------
    # Hardened Cross-File Transaction-Safe Purchase State Update
    # -------------------------------------------------------------------------
    def execute_purchase_transaction(
        self,
        merchant: str,
        order_items: List[Dict[str, Any]],
        expected_total: Optional[int] = None,
        order_id: Optional[str] = None,
        created_at: Optional[str] = None,
        razorpay_order_id: Optional[str] = None,
        razorpay_payment_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes a complete, hardened transaction-safe state update for confirmed purchases:
        1. Acquires merchant lock.
        2. Captures backup snapshots of existing inventory and orders state.
        3. Validates all products against current catalog and verifies prices.
        4. Validates that current stock >= requested quantity for every item.
        5. Verifies calculated total matches expected total if provided.
        6. Atomically writes updated inventory.json.
        7. Atomically writes updated orders.json.
        8. If writing orders.json fails, automatically ROLLS BACK inventory.json to its prior state.
        """
        m = self._normalize_merchant(merchant)
        if not order_items or not isinstance(order_items, list):
            raise ValueError("Order must contain a non-empty list of products")

        with self._locks[m]:
            # Step 1: Capture previous clean state snapshots for rollback safety
            previous_inventory = self._read_json(m, "inventory")
            previous_orders = self._read_json(m, "orders")

            # Work on deep copies during preparation
            inventory = copy.deepcopy(previous_inventory)
            orders = copy.deepcopy(previous_orders)

            catalog = {p["p_id"]: p for p in self.get_products(m)}
            inv_map = {item["p_id"]: item["stock"] for item in inventory}

            validated_products = []
            computed_total = 0

            # Step 2: Validate catalog, stock availability, and prices
            for item in order_items:
                p_id = item.get("p_id")
                qty = int(item.get("quantity", 0))
                if qty <= 0:
                    raise ValueError(f"Quantity must be >= 1 for item {p_id}")
                if p_id not in catalog:
                    raise ProductNotFoundError(f"Product {p_id} does not exist in {m} catalog")
                if p_id not in inv_map or inv_map[p_id] < qty:
                    avail = inv_map.get(p_id, 0)
                    raise InsufficientStockError(
                        f"Insufficient stock for product {p_id} in {m}. Available: {avail}, requested: {qty}"
                    )

                catalog_price = catalog[p_id]["price"]
                supplied_unit_price = item.get("unit_price")
                if supplied_unit_price is not None and int(supplied_unit_price) != catalog_price:
                    raise PriceMismatchError(
                        f"Price changed for product {p_id}. Current catalog: {catalog_price}, supplied: {supplied_unit_price}"
                    )

                val_item = {
                    "p_id": p_id,
                    "quantity": qty,
                    "unit_price": catalog_price
                }
                if item.get("color"):
                    val_item["color"] = item["color"]
                if item.get("size"):
                    val_item["size"] = item["size"]
                validated_products.append(val_item)
                computed_total += catalog_price * qty

            # Step 3: Validate total amount
            if expected_total is not None and int(expected_total) != computed_total:
                raise PriceMismatchError(
                    f"Total amount mismatch. Current calculated: {computed_total}, expected: {expected_total}"
                )

            # Step 4: Apply inventory deductions
            for item in validated_products:
                inv_map[item["p_id"]] -= item["quantity"]

            for item in inventory:
                item["stock"] = inv_map[item["p_id"]]

            # Step 5: Prepare order record
            prefix = "SN_ORD" if m == "shopnest" else "CW_ORD"
            if not order_id:
                existing_ids = [int(o["order_id"].split("_")[-1]) for o in orders if "_" in o.get("order_id", "")]
                next_num = max(existing_ids) + 1 if existing_ids else 1001
                order_id = f"{prefix}_{next_num}"

            timestamp = created_at or datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

            new_order = {
                "order_id": order_id,
                "products": validated_products,
                "total_amount": computed_total,
                "payment_status": "paid",
                "order_status": "confirmed",
                "created_at": timestamp,
                "is_user_order": True,
                "order_source": "agent_purchase"
            }
            if razorpay_order_id:
                new_order["razorpay_order_id"] = razorpay_order_id
            if razorpay_payment_id:
                new_order["razorpay_payment_id"] = razorpay_payment_id
            orders.append(new_order)

            # Step 6: Atomic writes with cross-file rollback protection
            inventory_written = False
            try:
                self._write_json_atomic(m, "inventory", inventory)
                inventory_written = True
                self._write_json_atomic(m, "orders", orders)
            except Exception as e:
                # If inventory was written but orders failed, rollback inventory immediately
                if inventory_written:
                    try:
                        self._write_json_atomic(m, "inventory", previous_inventory)
                    except Exception as rollback_err:
                        raise MerchantDataError(
                            f"Critical write failure on orders.json and inventory rollback failed: {rollback_err}"
                        ) from e
                raise

            return new_order


# Singleton instance for direct imports across the backend
dal = MerchantDataAccess()
