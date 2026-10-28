
import pandas as pd
from pathlib import Path
from datetime import datetime
import uuid

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# ==========================================
# PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "chroma_db"


# ==========================================
# TOOL 1: BUSINESS ORDER INVESTIGATION
# ==========================================

def investigate_order(order_id: str) -> dict:
    """Investigate an order and calculate available stock and shortage."""

    orders = pd.read_csv(DATA_DIR / "orders.csv")
    inventory = pd.read_csv(DATA_DIR / "inventory.csv")
    products = pd.read_csv(DATA_DIR / "products.csv")

    order = orders[orders["order_id"] == order_id]

    if order.empty:
        return {"error": f"Order {order_id} was not found."}

    order = order.iloc[0]

    stock = inventory[
        (inventory["product_id"] == order["product_id"])
    ]

    product = products[
        products["product_id"] == order["product_id"]
    ]

    if stock.empty:
        return {"error": "No inventory record found for this product."}

    stock = stock.iloc[0]

    product_name = (
        product.iloc[0]["product_name"]
        if not product.empty
        else order["product_id"]
    )

    physical = int(stock["physical_stock_cartons"])
    reserved = int(stock["reserved_cartons"])
    damaged = int(stock["damaged_cartons"])
    required = int(order["quantity_cartons"])

    available = max(0, physical - reserved - damaged)
    shortage = max(0, required - available)

    return {
        "order_id": order_id,
        "customer": order["customer_name"],
        "product": product_name,
        "required_cartons": required,
        "available_cartons": available,
        "shortage_cartons": shortage,
        "stock_sufficient": shortage == 0,
        "order_status": order["order_status"],
        "priority": order["priority"],
    }


# ==========================================
# TOOL 2: RAG DOCUMENT RETRIEVAL
# ==========================================

def search_business_documents(question: str) -> list:
    """Retrieve relevant passages from indexed business documents."""

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_db = Chroma(
        persist_directory=str(DB_DIR),
        embedding_function=embeddings,
        collection_name="wholesale_documents",
    )

    results = vector_db.similarity_search(question, k=3)

    return [
        {
            "content": document.page_content,
            "source": document.metadata.get("source", "Unknown"),
        }
        for document in results
    ]


# ==========================================
# TOOL 3: RECORD A CUSTOMER DELIVERY
# ==========================================

def record_delivery(
    customer_id: str,
    customer_name: str,
    product_id: str,
    quantity_cartons: int,
    warehouse_id: str = "WH-HYD-01",
    notes: str = "",
) -> dict:
    """
    Record a customer delivery and reduce physical inventory.

    The delivery is rejected if the requested quantity exceeds
    available stock after reservations and damaged cartons.
    """

    # Validate quantity.
    if (
        not isinstance(quantity_cartons, int)
        or isinstance(quantity_cartons, bool)
        or quantity_cartons <= 0
    ):
        return {
            "success": False,
            "message": "Quantity must be a positive whole number of cartons.",
        }

    inventory_path = DATA_DIR / "inventory.csv"
    transactions_path = DATA_DIR / "stock_transactions.csv"

    # Check required files.
    if not inventory_path.exists():
        return {
            "success": False,
            "message": "Inventory file was not found.",
        }

    if not transactions_path.exists():
        return {
            "success": False,
            "message": (
                "Stock transaction file was not found. "
                "Create data/stock_transactions.csv first."
            ),
        }

    # Read inventory.
    try:
        inventory = pd.read_csv(inventory_path)
    except Exception as error:
        return {
            "success": False,
            "message": f"Could not read inventory: {error}",
        }

    # Match both warehouse and product.
    matching_rows = inventory[
        (inventory["warehouse_id"] == warehouse_id)
        & (inventory["product_id"] == product_id)
    ]

    if len(matching_rows) == 0:
        return {
            "success": False,
            "message": (
                f"No inventory record found for product {product_id} "
                f"in warehouse {warehouse_id}."
            ),
        }

    if len(matching_rows) > 1:
        return {
            "success": False,
            "message": (
                "Multiple inventory rows match this product and warehouse. "
                "Resolve the duplicate rows before recording a delivery."
            ),
        }

    row_index = matching_rows.index[0]
    row = inventory.loc[row_index]

    try:
        physical = int(row["physical_stock_cartons"])
        reserved = int(row["reserved_cartons"])
        damaged = int(row["damaged_cartons"])
    except (ValueError, TypeError):
        return {
            "success": False,
            "message": "Inventory contains an invalid stock quantity.",
        }

    if physical < 0 or reserved < 0 or damaged < 0:
        return {
            "success": False,
            "message": "Inventory quantities cannot be negative.",
        }

    available = physical - reserved - damaged

    if available < 0:
        return {
            "success": False,
            "message": (
                "Inventory is inconsistent: reserved and damaged cartons "
                "exceed physical stock. Reconcile inventory first."
            ),
        }

    # Reject deliveries that would use stock reserved for other customers.
    if quantity_cartons > available:
        return {
            "success": False,
            "message": (
                f"Delivery rejected. Available stock: {available} cartons; "
                f"requested: {quantity_cartons} cartons. "
                "Inventory has not been changed."
            ),
        }

    # Prepare a unique transaction record.
    transaction = {
        "transaction_id": f"TXN-{uuid.uuid4().hex[:10].upper()}",
        "date": datetime.now().astimezone().isoformat(timespec="seconds"),
        "transaction_type": "CUSTOMER_DELIVERY",
        "customer_id": customer_id,
        "customer_name": customer_name,
        "warehouse_id": warehouse_id,
        "product_id": product_id,
        "quantity_cartons": quantity_cartons,
        "notes": notes,
    }

    # Load existing transaction records before appending.
    try:
        columns = [
            "transaction_id",
            "date",
            "transaction_type",
            "customer_id",
            "customer_name",
            "warehouse_id",
            "product_id",
            "quantity_cartons",
            "notes",
        ]

        if transactions_path.stat().st_size > 0:
            transactions = pd.read_csv(transactions_path)

            missing_columns = [
                column for column in columns
                if column not in transactions.columns
            ]

            if missing_columns:
                return {
                    "success": False,
                    "message": (
                        "Transaction file is missing columns: "
                        + ", ".join(missing_columns)
                    ),
                }
        else:
            transactions = pd.DataFrame(columns=columns)

        updated_transactions = pd.concat(
            [transactions, pd.DataFrame([transaction])],
            ignore_index=True,
        )

        # Save the transaction first.
        updated_transactions.to_csv(transactions_path, index=False)

    except Exception as error:
        return {
            "success": False,
            "message": f"Could not save delivery transaction: {error}",
        }

    # Reduce physical stock. Reservations and damages remain unchanged.
    new_physical = physical - quantity_cartons

    inventory.loc[row_index, "physical_stock_cartons"] = new_physical
    inventory.loc[row_index, "last_updated"] = (
        datetime.now().astimezone().isoformat(timespec="seconds")
    )

    try:
        inventory.to_csv(inventory_path, index=False)

    except Exception as error:
        return {
            "success": False,
            "transaction_id": transaction["transaction_id"],
            "message": (
                "The transaction was saved, but inventory could not be "
                "updated. Reconcile the transaction and inventory before "
                f"retrying. Details: {error}"
            ),
        }

    new_available = new_physical - reserved - damaged

    return {
        "success": True,
        "transaction_id": transaction["transaction_id"],
        "customer": customer_name,
        "product_id": product_id,
        "delivered_cartons": quantity_cartons,
        "previous_physical_stock": physical,
        "remaining_physical_stock": new_physical,
        "remaining_available_stock": new_available,
        "message": "Delivery recorded and inventory updated successfully.",
    }


# ==========================================
# TEST BOTH EXISTING TOOLS
# ==========================================

if __name__ == "__main__":

    print("\n========== ORDER INVESTIGATION ==========")

    order_result = investigate_order("ORD-1001")

    for key, value in order_result.items():
        print(f"{key}: {value}")

    print("\n========== RAG DOCUMENT SEARCH ==========")

    question = (
        "Can we substitute another biscuit product "
        "if stock is insufficient?"
    )

    document_results = search_business_documents(question)

    for index, result in enumerate(document_results, start=1):
        print(f"\n--- Document Result {index} ---")
        print(f"Source: {result['source']}")
        print(result["content"])

    print("\n========== TEST COMPLETE ==========")
