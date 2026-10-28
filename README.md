
# Wholesale Distribution Intelligence Agent using Agentic RAG

An AI-powered wholesale distribution decision-support system that investigates order shortages, retrieves business policies, recommends possible solutions, and records inventory deliveries using **Agentic RAG, LangChain, Groq, ChromaDB, and Streamlit**.

---

## 1. Project Overview

Wholesale distribution businesses handle customer orders, warehouse inventory, supplier deliveries, product substitutions, and stock shortages.

When a customer places an order larger than the available inventory, employees must investigate the shortage, review relevant business policies, and decide how to fulfill the order.

This project provides an **AI-powered Wholesale Distribution Intelligence Agent** to support these decisions.

The application uses two AI agents:

- **Agent 1 — Business Investigation Agent:** Investigates orders, checks inventory, calculates shortages, and retrieves relevant business policies.
- **Agent 2 — Decision-Support Agent:** Analyzes Agent 1's findings and generates recommendations, risks, and next steps.

The application also provides an inventory delivery-recording feature.

---

## 2. Problem Statement

Wholesale distributors frequently face challenges such as:

- Insufficient stock to fulfill customer orders.
- Delayed supplier deliveries.
- Difficulty finding relevant company policies.
- Product substitution decisions.
- Manual inventory calculations.
- Time-consuming order investigations.
- Lack of structured decision-support reports.

### Proposed Solution

Build an AI-powered system that combines:

1. Structured business data from CSV files.
2. Unstructured business policies from PDF documents.
3. Retrieval-Augmented Generation (RAG).
4. LLM-powered agents.
5. Business tools for order investigation and inventory updates.
6. An interactive Streamlit application.

The system helps users investigate business problems and make informed decisions based on available data and relevant policies.

---

## 3. Project Objectives

The main objectives of this project are:

- Investigate wholesale customer orders.
- Calculate available inventory and shortages.
- Retrieve relevant business policies using semantic search.
- Generate policy-grounded recommendations.
- Identify missing information and business risks.
- Provide structured decision-support reports.
- Record deliveries and update inventory.
- Reduce unsupported conclusions by validating business data.
- Build a practical multi-agent AI application.

---

## 4. Key Features

### 4.1 Order Investigation

The system retrieves:

- Order ID
- Customer details
- Product information
- Required quantity
- Available inventory
- Stock shortage
- Order status
- Order priority

### 4.2 Inventory Shortage Calculation

The system calculates available stock using:

**Available Stock = Physical Stock - Reserved Stock - Damaged Stock**

The shortage is calculated using:

**Shortage = max(0, Required Quantity - Available Stock)**

### 4.3 Retrieval-Augmented Generation (RAG)

The system searches business documents to retrieve relevant information about:

- Supplier agreements
- Product substitution policies
- Bulk-order policies
- Delivery procedures
- Complaint-handling procedures
- Product catalogues

### 4.4 Multi-Agent Decision Support

Two agents work sequentially:

**Agent 1: Investigation Agent**

- Uses business tools.
- Retrieves order and inventory data.
- Searches relevant documents.
- Generates an investigation report.

**Agent 2: Decision-Support Agent**

- Receives the investigation findings.
- Identifies verified facts and unknown information.
- Compares possible solutions.
- Explains benefits and risks.
- Generates conditional recommendations.

### 4.5 Inventory Delivery Recording

Users can record a delivery through the Streamlit interface.

The system:

- Validates the delivery quantity.
- Checks available inventory.
- Rejects deliveries that exceed available stock.
- Records the delivery transaction.
- Updates the inventory CSV file.

### 4.6 Invalid Order Validation

The application checks whether the requested order ID exists in the configured orders dataset.

If the order is not found, the system reports that result instead of inventing customer details or stock quantities.

---

## 5. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core application development |
| Streamlit | Interactive web application |
| LangChain | Agent creation and tool integration |
| LangGraph / LangChain Agent Framework | Agent workflow orchestration |
| Groq API | Language-model inference |
| GPT-OSS 20B | LLM used for investigation and decision support |
| ChromaDB | Vector database |
| Sentence Transformers | Text embeddings |
| all-MiniLM-L6-v2 | Embedding model |
| Pandas | CSV data processing |
| PyMuPDF | PDF text extraction |
| python-dotenv | Environment variable management |

### LLM Configuration

The project uses the following model through the Groq API:

`openai/gpt-oss-20b`

### Embedding Model

The project uses:

`sentence-transformers/all-MiniLM-L6-v2`

---

## 6. System Architecture

```text
                    USER
                     |
                     v
              STREAMLIT APP
                     |
          +----------+----------+
          |                     |
          v                     v
   BUSINESS INVESTIGATION   RECORD DELIVERY
          |                     |
          v                     v
   AGENT 1: INVESTIGATION   INVENTORY TOOL
          |
          +--------------------------+
          |                          |
          v                          v
    CSV BUSINESS DATA          RAG RETRIEVAL
          |                          |
          |                          v
          |                     CHROMADB
          |                          |
          |                          v
          |                   BUSINESS POLICIES
          |                          |
          +------------+-------------+
                       |
                       v
              INVESTIGATION REPORT
                       |
                       v
             AGENT 2: DECISION SUPPORT
                       |
                       v
                FINAL REPORT
                       |
                       v
                     USER
```

---

## 7. Project Folder Structure

```text
wholesale_intelligence_agent/
|
|-- app.py
|-- business_agent.py
|-- decision_agent.py
|-- business_tools.py
|-- rag_pipeline.py
|-- test_rag.py
|-- test_groq.py
|-- inspect_data.py
|-- check_order.py
|-- requirements.txt
|-- .gitignore
|
|-- data/
|   |-- products.csv
|   |-- inventory.csv
|   |-- orders.csv
|   |-- suppliers.csv
|   |-- purchase_orders.csv
|   |-- sales_history.csv
|   |-- price_rules.csv
|   |-- complaints.csv
|   |-- README.txt
|   |
|   |-- documents/
|       |-- agent_workflow_and_tool_rules.pdf
|       |-- bulk_order_and_substitution_policy.pdf
|       |-- delivery_and_complaint_procedure.pdf
|       |-- product_catalogue.pdf
|       |-- supplier_agreement_sunrise_foods.pdf
|
|-- chroma_db/          # Generated vector database
|-- .env                # Local API key configuration
```

**Note:** The `.env`, virtual environment, backup folders, and generated vector database should not be uploaded to a public repository.

---

## 8. Dataset Description

The project uses sample wholesale distribution datasets.

### 8.1 products.csv

Contains product details such as:

- Product ID
- Product name
- Supplier name
- Product category
- Units per carton
- Price per carton
- Suggested alternative product

### 8.2 inventory.csv

Contains warehouse inventory information:

- Warehouse ID
- Product ID
- Physical stock
- Reserved stock
- Damaged stock
- Last updated timestamp

### 8.3 orders.csv

Contains customer order information:

- Order ID
- Customer ID
- Customer name
- Product ID
- Quantity
- Required delivery date
- Order status
- Priority
- Credit status

### 8.4 suppliers.csv

Contains supplier information and lead-time details.

### 8.5 purchase_orders.csv

Contains purchase orders, expected arrival dates, and shipment status.

### 8.6 sales_history.csv

Contains historical sales information.

### 8.7 price_rules.csv

Contains product pricing rules.

### 8.8 complaints.csv

Contains customer complaints and complaint status.

---

## 9. How Agentic RAG Works

### Step 1: Load Business Documents

The system loads business policy documents stored in:

`data/documents/`

### Step 2: Extract PDF Text

PyMuPDF extracts text from the PDF documents.

### Step 3: Split Documents into Chunks

The extracted text is divided into smaller chunks for retrieval.

### Step 4: Generate Embeddings

The embedding model converts document chunks into numerical vector representations.

Embedding model:

`sentence-transformers/all-MiniLM-L6-v2`

### Step 5: Store Embeddings

The embeddings and associated document content are stored in ChromaDB.

### Step 6: Retrieve Relevant Documents

When a user asks a question, the system performs semantic similarity search to retrieve relevant policy information.

### Step 7: Generate Investigation Report

Agent 1 combines business tool results with retrieved policy information.

### Step 8: Generate Decision Report

Agent 2 analyzes the investigation findings and generates a structured decision-support report.

---

## 10. Business Tools

### 10.1 investigate_order()

This tool:

- Reads customer orders.
- Retrieves product details.
- Checks inventory.
- Calculates available stock.
- Calculates stock shortages.
- Returns structured order information.

### 10.2 search_business_documents()

This tool:

- Searches the ChromaDB vector database.
- Retrieves relevant policy passages.
- Provides supporting context for business investigations.

### 10.3 record_delivery()

This tool:

- Validates the delivery quantity.
- Checks product and warehouse inventory.
- Verifies stock availability.
- Records the transaction.
- Updates physical inventory.

**Important:** Delivery recording changes inventory data. The same delivery should not be submitted multiple times.

---

## 11. Example: Order Shortage Investigation

### User Question

> Investigate order ORD-1001. How much stock is available, what is the shortage, and what do our policies recommend?

### Sample Order Details

| Field | Value |
|---|---|
| Order ID | ORD-1001 |
| Customer | Kiran Wholesale |
| Product | Sunrise Glucose Biscuits |
| Required quantity | 500 cartons |
| Physical stock | 350 cartons |
| Reserved stock | 100 cartons |
| Damaged stock | 0 cartons |
| Available stock | 250 cartons |
| Shortage | 250 cartons |
| Order status | Pending |
| Priority | High |

### Stock Calculation

```text
Available Stock = 350 - 100 - 0
Available Stock = 250 cartons

Shortage = 500 - 250
Shortage = 250 cartons
```

### Agent 1 Output

The investigation agent identifies:

- The customer and product.
- The required order quantity.
- The available stock.
- The shortage.
- Relevant business policies.
- Possible fulfillment options.

### Agent 2 Output

The decision-support agent evaluates:

1. Partial fulfillment.
2. Customer-approved substitution.
3. Supplier replenishment.
4. Alternate approved suppliers.

It also identifies risks, unknown information, and recommended next steps.

The system does not automatically execute these recommendations.

---

## 12. Business Policy Rules

The system uses business policy documents to support its recommendations.

### 12.1 Product Substitution

A substitute product must not be delivered without the required wholesale customer approval.

### 12.2 Supplier Lead Time

The sample supplier agreement describes a standard lead time of:

**3-5 calendar days after purchase-order acceptance.**

This is not a guaranteed delivery date.

### 12.3 Partial Fulfillment

Partial fulfillment may be considered when available inventory is insufficient for the complete order.

The responsible teams must verify dispatch readiness and applicable business requirements.

### 12.4 Inventory Accuracy

Available inventory must account for reserved and damaged stock.

### 12.5 Human Decision-Making

The system provides recommendations. Responsible personnel must verify the information and approve actions where required.

---

## 13. Handling Invalid Orders

### Example Question

> Investigate order ORD-9999. Tell me its customer, stock availability, shortage, and recommended action.

### Expected Behavior

If the order does not exist in the configured dataset, the system reports:

```text
ORDER VALIDATION

Order ID: ORD-9999

Result:
No matching record was found in data/orders.csv.

Recommended Next Steps:
1. Verify the order ID.
2. Check the customer's confirmation or invoice.
3. Consult other relevant systems if available.
```

The system avoids fabricating customer information or order details.

---

## 14. Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/(https://github.com/sadanandeshruthi554)/wholesale-distribution-intelligence-agent.git

cd wholesale-distribution-intelligence-agent
```

### Step 2: Create a Virtual Environment

For Windows:

```cmd
python -m venv env
```

Activate the environment:

```cmd
env\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure the Groq API Key

Create a `.env` file in the project root.

Add:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Replace the placeholder with your own Groq API key.

### Step 5: Prepare the Vector Database

If the ChromaDB database has not already been generated, run the RAG indexing process defined in `rag_pipeline.py`.

If the script has an executable indexing entry point, use:

```bash
python rag_pipeline.py
```

The indexing process must read the PDF documents and create the persistent vector database expected by the application.

### Step 6: Run the Streamlit Application

```bash
streamlit run app.py
```

Open the application in your browser:

```text
http://localhost:8501
```

---
## 16. Testing and Validation

The following scenarios were tested during development:

| Test Case | Result |
|---|---|
| Groq API connectivity | Passed |
| Business document retrieval | Passed |
| Order investigation for ORD-1001 | Passed |
| Available stock calculation | Passed |
| Stock shortage calculation | Passed |
| Two-agent report generation | Passed |
| Unknown order validation | Passed |
| Product substitution policy reasoning | Passed |
| Streamlit application execution | Passed |
| Delivery transaction recording | Tested |
| Inventory update after delivery | Tested |

These results describe development testing, not comprehensive production-level validation.

---
## 18. Future Enhancements

Possible future improvements include:

1. PostgreSQL or MySQL database integration.
2. Real-time supplier API integration.
3. Live inventory synchronization.
4. Automated supplier notifications.
5. Customer communication workflows.
6. Advanced demand forecasting.
7. Inventory optimization.
8. Role-based authentication.
9. Audit logs and transaction history.
10. Duplicate delivery prevention.
11. Automated testing and monitoring.
12. Deployment to a cloud platform.
13. Improved report formatting and citations.
14. Additional agent evaluation and reliability checks.

---

## 19. Learning Outcomes

Through this project, I gained practical experience in:

- Python application development.
- Retrieval-Augmented Generation (RAG).
- Agentic AI concepts.
- LLM tool calling.
- Multi-agent workflow design.
- LangChain integration.
- Groq API integration.
- Vector databases.
- Text embeddings.
- Semantic search.
- PDF document processing.
- CSV-based business data analysis.
- Streamlit application development.
- Business decision-support systems.
- Validation and responsible AI behavior.

---

## 20. Conclusion

The **Wholesale Distribution Intelligence Agent using Agentic RAG** demonstrates how AI agents, structured business data, and document retrieval can be combined to support wholesale distribution operations.

The system investigates customer orders, calculates inventory shortages, retrieves relevant policies, and produces structured decision-support reports.

It also supports inventory delivery recording through an interactive Streamlit interface.

The project demonstrates a practical application of **Agentic RAG for business investigation and decision support**, while keeping final operational decisions under human control.


The recommendations generated by the AI agents are intended to support human decision-making and should be verified before operational use.
