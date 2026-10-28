
import streamlit as st
import pandas as pd

from decision_agent import create_decision_report
from business_tools import record_delivery


# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Wholesale Intelligence Agent",
    page_icon="📦",
    layout="wide",
)


# --------------------------------------------------
# 2. Application header
# --------------------------------------------------

st.title("📦 Wholesale Distribution Intelligence Agent")

st.write(
    "Investigate order fulfillment problems, manage stock, "
    "and record customer deliveries."
)

st.divider()


# --------------------------------------------------
# 3. Load project data
# --------------------------------------------------

from business_tools import DATA_DIR

inventory_path = DATA_DIR / "inventory.csv"
products_path = DATA_DIR / "products.csv"


# --------------------------------------------------
# 4. Sidebar
# --------------------------------------------------

with st.sidebar:
    st.header("About")

    st.write("**Agent 1:** Business Investigation")
    st.write("**Agent 2:** Decision Support")
    st.write("**Inventory:** Stock and delivery tracking")

    st.caption(
        "Data sources: CSV files and indexed business PDFs."
    )


# --------------------------------------------------
# 5. Inventory overview
# --------------------------------------------------

st.header("Current Inventory")

try:
    inventory = pd.read_csv(inventory_path)
    products = pd.read_csv(products_path)

    inventory_view = inventory.merge(
        products[["product_id", "product_name"]],
        on="product_id",
        how="left",
    )

    inventory_view["available_cartons"] = (
        inventory_view["physical_stock_cartons"]
        - inventory_view["reserved_cartons"]
        - inventory_view["damaged_cartons"]
    )

    st.dataframe(
        inventory_view[
            [
                "warehouse_id",
                "product_id",
                "product_name",
                "physical_stock_cartons",
                "reserved_cartons",
                "damaged_cartons",
                "available_cartons",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

except Exception as error:
    st.error(f"Could not load inventory: {error}")


st.divider()


# --------------------------------------------------
# 6. Record a customer delivery
# --------------------------------------------------

st.header("Record Customer Delivery")

st.info(
    "Record a delivery after the cartons have actually been supplied. "
    "A successful submission reduces physical stock automatically."
)

try:
    inventory = pd.read_csv(inventory_path)
    products = pd.read_csv(products_path)

    warehouse_options = sorted(
        inventory["warehouse_id"].dropna().unique().tolist()
    )

    with st.form("delivery_form"):
        if warehouse_options:
            warehouse_id = st.selectbox(
                "Warehouse",
                warehouse_options,
            )
        else:
            warehouse_id = None
            st.warning("No warehouses are available in inventory.")

        if warehouse_id:
            warehouse_inventory = inventory[
                inventory["warehouse_id"] == warehouse_id
            ]

            product_options = sorted(
                warehouse_inventory["product_id"]
                .dropna()
                .unique()
                .tolist()
            )
        else:
            product_options = []

        if product_options:
            product_labels = {}

            for product_id in product_options:
                matching_product = products[
                    products["product_id"] == product_id
                ]

                if not matching_product.empty:
                    name = matching_product.iloc[0]["product_name"]
                else:
                    name = product_id

                product_labels[product_id] = f"{product_id} - {name}"

            product_id = st.selectbox(
                "Product",
                product_options,
                format_func=lambda value: product_labels[value],
            )
        else:
            product_id = None
            st.warning("No products found for this warehouse.")

        customer_id = st.text_input(
            "Customer ID",
            placeholder="For example: CUST-1001",
        )

        customer_name = st.text_input(
            "Wholesale shop name",
            placeholder="For example: Kiran Wholesale",
        )

        quantity_cartons = st.number_input(
            "Cartons delivered",
            min_value=1,
            step=1,
            value=1,
        )

        notes = st.text_input(
            "Notes (optional)",
            placeholder="Delivery reference or additional details",
        )

        submitted = st.form_submit_button(
            "Record Delivery",
            type="primary",
        )

    if submitted:
        if not warehouse_id or not product_id:
            st.error("Please select a valid warehouse and product.")

        elif not customer_id.strip() or not customer_name.strip():
            st.error("Please enter both the customer ID and shop name.")

        else:
            result = record_delivery(
                customer_id=customer_id.strip(),
                customer_name=customer_name.strip(),
                product_id=product_id,
                quantity_cartons=int(quantity_cartons),
                warehouse_id=warehouse_id,
                notes=notes.strip(),
            )

            if result.get("success"):
                st.success(result["message"])

                st.write(
                    f"**Transaction ID:** "
                    f"{result['transaction_id']}"
                )
                st.write(
                    f"**Cartons delivered:** "
                    f"{result['delivered_cartons']}"
                )
                st.write(
                    f"**Physical stock remaining:** "
                    f"{result['remaining_physical_stock']}"
                )
                st.write(
                    f"**Available stock remaining:** "
                    f"{result['remaining_available_stock']}"
                )

                st.rerun()

            else:
                st.error(result.get("message", "Delivery could not be recorded."))

except Exception as error:
    st.error(f"Could not prepare the delivery form: {error}")


st.divider()


# --------------------------------------------------
# 7. Investigate a business problem
# --------------------------------------------------

st.header("Investigate a Business Problem")


question = st.text_area(
    "Enter your question",
    value="",
    height=120,
    placeholder="For example: Investigate order ORD-1001",
    key="business_question",
)



# --------------------------------------------------
# 8. Run the agents
# --------------------------------------------------

if st.button("Investigate", type="primary"):

    if not question.strip():
        st.warning("Please enter a business question.")

    else:
        with st.spinner(
            "Agents are investigating the problem "
            "and preparing recommendations..."
        ):
            try:
                report = create_decision_report(question)

                clean_report = (
                    report
                    .replace("\\<br>", " ")
                    .replace("<br />", " ")
                    .replace("<br/>", " ")
                    .replace("<br>", " ")
                    .replace("\\\n", "\n")
                )

                st.success("Investigation completed!")
                st.markdown(clean_report)

                st.download_button(
                    label="Download Decision Report",
                    data=clean_report,
                    file_name="wholesale_decision_report.md",
                    mime="text/markdown",
                )

            except Exception as error:
                st.error(
                    "The investigation failed. Check the terminal "
                    "and verify your API configuration."
                )
                st.exception(error)


# --------------------------------------------------
# 9. Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Decision support only. Verify inventory, supplier commitments, "
    "and customer approvals before taking business action."
)
