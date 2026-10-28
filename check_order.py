
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")

# Load business data
inventory = pd.read_csv(DATA_DIR / "inventory.csv")
orders = pd.read_csv(DATA_DIR / "orders.csv")
products = pd.read_csv(DATA_DIR / "products.csv")

# Select the order to investigate
order_id = "ORD-1001"

order_rows = orders[orders["order_id"] == order_id]

if order_rows.empty:
    print(f"Order {order_id} was not found.")
    raise SystemExit

order = order_rows.iloc[0]

product_id = order["product_id"]
required_quantity = int(order["quantity_cartons"])

# Find product details
product_rows = products[products["product_id"] == product_id]

if product_rows.empty:
    print(f"Product {product_id} was not found.")
    raise SystemExit

product_name = product_rows.iloc[0]["product_name"]

# Find inventory for this product
stock_rows = inventory[inventory["product_id"] == product_id]

if stock_rows.empty:
    print(f"No inventory record found for {product_id}.")
    raise SystemExit

stock = stock_rows.iloc[0]

physical_stock = int(stock["physical_stock_cartons"])
reserved_stock = int(stock["reserved_cartons"])
damaged_stock = int(stock["damaged_cartons"])

available_stock = physical_stock - reserved_stock - damaged_stock
shortage = max(0, required_quantity - available_stock)

# Display the report
print("\n========== ORDER INVESTIGATION ==========")
print(f"Order ID: {order_id}")
print(f"Customer: {order['customer_name']}")
print(f"Product: {product_name}")
print(f"Required quantity: {required_quantity} cartons")
print(f"Physical stock: {physical_stock} cartons")
print(f"Reserved stock: {reserved_stock} cartons")
print(f"Damaged stock: {damaged_stock} cartons")
print(f"Available stock: {available_stock} cartons")
print(f"Shortage: {shortage} cartons")

if shortage == 0:
    print("Result: Stock quantity is sufficient.")
else:
    print("Result: Stock quantity is insufficient.")

print("==========================================")
