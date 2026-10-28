
import os
import re
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from business_agent import agent as business_agent


# --------------------------------------------------
# 1. Load environment variables and project paths
# --------------------------------------------------

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
ORDERS_FILE = BASE_DIR / "data" / "orders.csv"

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise RuntimeError(
        "GROQ_API_KEY was not found. Check that your .env file "
        "exists in the project root and contains your API key."
    )


# --------------------------------------------------
# 2. Initialize the decision-support LLM
# --------------------------------------------------

decision_llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=api_key,
)


# --------------------------------------------------
# 3. Validate order IDs against the orders dataset
# --------------------------------------------------

def check_order_exists(order_id: str) -> bool:
    """
    Check whether an order ID exists in data/orders.csv.

    This verifies only the dataset used by this application.
    It does not check other company systems.
    """

    if not ORDERS_FILE.exists():
        raise FileNotFoundError(
            f"Orders file not found: {ORDERS_FILE}"
        )

    orders = pd.read_csv(
        ORDERS_FILE,
        dtype={"order_id": str},
    )

    order_ids = (
        orders["order_id"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    return order_id.strip().upper() in order_ids.values


def find_order_ids(question: str) -> list:
    """
    Extract order IDs such as ORD-1001 from a question.
    """

    return re.findall(
        r"\bORD-[A-Za-z0-9-]+\b",
        question,
        flags=re.IGNORECASE,
    )


# --------------------------------------------------
# 4. Create the decision-support report
# --------------------------------------------------

def create_decision_report(business_question: str) -> str:
    """
    Run Agent 1 to investigate the business problem,
    then run Agent 2 to produce a decision-support report.

    Unknown order IDs are handled before calling the agents.
    """

    # Step A: Validate any explicitly mentioned order IDs.
    order_ids = find_order_ids(business_question)

    for order_id in order_ids:
        if not check_order_exists(order_id):
            return (
                "========== ORDER VALIDATION ==========\n\n"
                f"Order ID: {order_id}\n\n"
                "Result: No matching record was found in the "
                "configured data/orders.csv file.\n\n"
                "Verified finding:\n"
                "The requested order ID was not found in the "
                "orders dataset searched by this application.\n\n"
                "Unknowns:\n"
                "- Whether the order exists in another system.\n"
                "- Whether the order ID was entered incorrectly.\n"
                "- Whether historical records exist elsewhere.\n\n"
                "Recommended next steps:\n"
                "1. Verify the order ID against the customer's "
                "confirmation or invoice.\n"
                "2. If the ID is correct, ask the responsible "
                "team to check other relevant systems, if available.\n\n"
                "No customer statements, deadlines, order details, "
                "or system-wide conclusions have been assumed."
            )

    # Step B: Agent 1 investigates the business problem.
    investigation = business_agent.invoke(
        {
            "messages": [
                HumanMessage(content=business_question)
            ]
        }
    )

    investigation_report = investigation["messages"][-1].content

    # Step C: Agent 2 evaluates the findings and recommends actions.
    system_prompt = """
You are a Decision Support Agent for a wholesale distribution business.

Your job is to analyze the investigation report and produce a
concise, evidence-based decision-support report.

RELIABILITY RULES

1. Use only facts supported by the investigation report or the user's
   question.

2. Separate verified facts, unknown information, recommendations,
   and assumptions.

3. Never invent customer statements, order details, stock levels,
   supplier commitments, prices, approvals, deadlines, or owners.

4. Do not assume the searched CSV files contain every record in every
   company system.

5. If an order cannot be found, state only that it was not found in
   the dataset searched. Do not generate a normal fulfillment report
   for a missing order.

6. If essential information is missing, explicitly say it is unknown.

7. Make recommendations conditional whenever they depend on
   unverified facts, customer preferences, or approvals.

8. Customer approval is required before delivering a substitute product.

9. Do not claim an email was sent, an order was updated, or a shipment
   was arranged unless a tool actually performed that action.

10. Do not create arbitrary deadlines or invent urgency.

11. Do not convert a supplier's standard lead time into a confirmed
    delivery date or shipment promise. State the event that starts
    the lead-time clock and identify any missing confirmation.

12. Do not claim a substitute is in stock, can fulfill the full
    quantity, or can arrive sooner unless the investigation verifies it.

13. Distinguish inventory currently available from inventory that
    can actually be dispatched. Do not promise immediate shipment
    unless dispatch readiness is confirmed.

14. Do not make customer approval a mandatory condition for partial
    shipment unless the retrieved policy explicitly requires it.

15. Do not present a possible option as a confirmed capability.
    For example, alternate-supplier sourcing is only an option until
    supplier availability, timing, and any required approval are verified.

16. Do not invent discounts, savings, product equivalence, or pricing.

17. Preserve the meaning of policy documents. Do not turn a general
    policy into a guarantee for this particular order.

18. Preserve policy wording exactly when describing lead times.
    If the source says "calendar days", never rewrite it as
    "business days".

19. Do not state that customer approval is or is not required
    for partial fulfilment unless the retrieved policy explicitly
    establishes that requirement.

20. Recommendations must be framed as proposed actions, not completed
    actions or confirmed commitments. Use wording such as
    "verify dispatch readiness, then arrange shipment if appropriate"
    when dispatch readiness has not been checked.


REPORT FORMAT

Produce a concise report with these sections:

1. Problem Summary
2. Verified Facts and Sources
3. Unknown Information
4. Options, Benefits, and Risks
5. Conditional Recommendation
6. Practical Next Steps

For each option:
- Explain the benefit only when it is reasonably supported.
- Identify important risks and missing information.
- State any required approval or verification.

For recommendations:
- Distinguish stock availability from dispatch readiness.
- Distinguish standard lead time from confirmed delivery timing.
- Identify which details must be checked before making a commitment.
- Never imply that a proposed action has already occurred.

If evidence is insufficient for a section, say so rather than speculating.

Use readable Markdown headings and bullet lists.
Avoid HTML tags such as <br> in Markdown tables.
Prefer bullet lists over wide tables when listing multiple facts.
"""

    decision_input = f"""
BUSINESS QUESTION:
{business_question}

AGENT 1 INVESTIGATION REPORT:
{investigation_report}

Prepare the decision-support report using only the evidence above
and following all reliability rules.
"""

    decision = decision_llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=decision_input),
        ]
    )

    decision_report = decision.content

    # Step D: Return both agents' reports.
    return (
        "========== AGENT 1: INVESTIGATION ==========\n\n"
        f"{investigation_report}\n\n"
        "========== AGENT 2: DECISION REPORT ==========\n\n"
        f"{decision_report}"
    )


# --------------------------------------------------
# 5. Allow testing directly from the terminal
# --------------------------------------------------

if __name__ == "__main__":
    print("\nWholesale Distribution Decision Support Agent")
    print("-" * 48)

    question = input(
        "\nEnter your business question: "
    ).strip()

    if not question:
        print("Please enter a business question.")
    else:
        try:
            report = create_decision_report(question)
            print("\n" + report)

        except Exception as error:
            print(
                "\nAn error occurred while generating the report:"
            )
            print(error)
