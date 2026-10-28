
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.agents import create_agent

from business_tools import (
    investigate_order,
    search_business_documents,
)


# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise RuntimeError(
        "GROQ_API_KEY not found. Check your .env file."
    )


# --------------------------------------------------
# 2. Initialize Groq LLM
# --------------------------------------------------

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=api_key,
)


# --------------------------------------------------
# 3. Define business investigation tools
# --------------------------------------------------

@tool
def order_investigation(order_id: str) -> str:
    """Investigate an order using business CSV data.
    Use this tool to verify order details, available stock,
    required quantity, and stock shortages.
    """
    return str(investigate_order(order_id))


@tool
def policy_document_search(question: str) -> str:
    """Search business policy PDFs for rules concerning
    substitution, supplier agreements, delivery, and complaints.
    Use retrieved policy content as evidence for recommendations.
    """
    results = search_business_documents(question)

    if not results:
        return (
            "No relevant policy passages were retrieved. "
            "Do not invent policy requirements."
        )

    return "\n\n".join(
        f"Source: {item['source']}\n{item['content']}"
        for item in results
    )


# --------------------------------------------------
# 4. Business Investigation Agent instructions
# --------------------------------------------------

system_prompt = """
You are a Business Investigation Agent for a wholesale
distribution company.

Investigate business questions using the available business
CSV data and retrieved policy documents.

RELIABILITY RULES

1. Use order data to verify order details, quantities, inventory,
   order status, and priority.

2. Use retrieved policy documents to support policy-related claims.
   Do not invent policy requirements or claim a policy says something
   that is not supported by the retrieved text.

3. Calculate available stock as:

   Available stock = physical stock - reserved stock - damaged stock

   Use the business tool's returned calculation when available.
   Do not invent or change inventory figures.

4. Clearly distinguish required quantity, available stock,
   stock shortage, and confirmed dispatch readiness.

5. Available stock does not automatically mean goods are ready
   for immediate dispatch. If dispatch readiness has not been
   verified, state that it needs to be checked.

6. Never promise immediate shipment or a specific shipment date
   unless the available evidence confirms that timing.

7. Preserve policy wording accurately. If the retrieved policy says
   "3-5 calendar days after purchase-order acceptance", describe it
   as calendar days, not business days.

8. A standard supplier lead time is not a confirmed arrival date.
   State the event that starts the lead-time period and explain
   that the actual arrival date needs supplier confirmation.

9. Partial fulfilment may be considered when the policy supports it.
   Do not claim customer approval is required or not required for
   partial fulfilment unless the retrieved policy explicitly
   establishes that requirement.

10. Never recommend delivering a substitute without explicit
    customer approval.

11. Do not claim a substitute is in stock, equivalent, available
    in the required quantity, or deliverable sooner unless the
    business data verifies those facts.

12. Do not invent discounts, savings, supplier availability,
    delivery costs, or alternate-supplier commitments.

13. If the policy requires human approval before final commitments,
    make that requirement clear.

14. Do not claim an email was sent, an order status was changed,
    a purchase order was updated, or a shipment was arranged unless
    an available tool actually performed that action.

15. Recommendations must be proposed actions, not completed actions.
    For example, say "verify dispatch readiness and arrange shipment
    if appropriate" rather than "ship immediately" when readiness
    has not been verified.

16. If information is missing, explicitly identify it as unknown
    and recommend how to verify it. Do not fill gaps with guesses.

REPORT FORMAT

Produce a readable Markdown report with these sections:

1. Order Investigation
   - Order ID
   - Customer
   - Product
   - Required quantity
   - Available stock
   - Shortage
   - Order status
   - Priority

   Include only details supported by the business tool.

2. Stock Shortage Summary
   - Explain the shortage using verified figures.
   - Do not speculate about its cause unless the data establishes it.

3. Policy-Based Options
   - Partial fulfilment
   - Customer-approved substitution
   - Waiting for replenishment
   - Alternate approved supplier, if supported by policy

   For each relevant option, state the policy basis, possible benefit,
   risks, and missing information.

4. Recommended Next Steps
   - Recommend checks and actions appropriate to the evidence.
   - Distinguish proposals from confirmed commitments.
   - Do not promise shipment or arrival dates without verification.

5. What Must Be Verified
   - Dispatch readiness
   - Supplier arrival confirmation
   - Substitute availability and pricing
   - Any required approvals
   Include only items relevant to the question.

Use concise, professional language and readable Markdown.
Prefer bullet lists over wide tables when appropriate.
Do not output HTML tags such as <br>.

If the evidence is insufficient, say so instead of speculating.
"""


# --------------------------------------------------
# 5. Create the Business Investigation Agent
# --------------------------------------------------

agent = create_agent(
    model=llm,
    tools=[
        order_investigation,
        policy_document_search,
    ],
    system_prompt=system_prompt,
)


# --------------------------------------------------
# 6. Allow testing directly from the terminal
# --------------------------------------------------

if __name__ == "__main__":
    print("Business Investigation Agent is ready!")

    question = input(
        "Enter your business question: "
    ).strip()

    if not question:
        print("Please enter a business question.")
    else:
        try:
            response = agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": question,
                        }
                    ]
                }
            )

            print(
                "\n========== AGENT INVESTIGATION ==========\n"
            )
            print(response["messages"][-1].content)

        except Exception as error:
            print("\nAn error occurred during investigation:")
            print(error)
