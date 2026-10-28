
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data")

csv_files = [
    "products.csv",
    "inventory.csv",
    "orders.csv",
    "suppliers.csv",
    "purchase_orders.csv",
    "sales_history.csv",
    "price_rules.csv",
    "complaints.csv",
]

print("=" * 60)
print("WHOLESALE DISTRIBUTION DATA INSPECTION")
print("=" * 60)

for filename in csv_files:
    file_path = DATA_DIR / filename
    print(f"\n--- {filename} ---")

    if not file_path.exists():
        print("File not found!")
        continue

    try:
        df = pd.read_csv(file_path)
        print(f"Rows: {len(df)}")
        print(f"Columns: {list(df.columns)}")
        print("\nFirst 3 records:")
        print(df.head(3).to_string(index=False))
    except Exception as error:
        print(f"Error reading file: {error}")

print("\nInspection completed.")
