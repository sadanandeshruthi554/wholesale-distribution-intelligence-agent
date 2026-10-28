
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DB_DIR = Path("chroma_db")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = Chroma(
    persist_directory=str(DB_DIR),
    embedding_function=embeddings,
    collection_name="wholesale_documents"
)

question = input("Ask a question about your wholesale business: ")

results = vector_db.similarity_search(question, k=3)

print("\n========== RAG SEARCH RESULTS ==========")

if not results:
    print("No relevant document passages found.")

for i, doc in enumerate(results, start=1):
    print(f"\nResult {i}")
    print(f"Source: {doc.metadata.get('source', 'Unknown')}")
    print(f"Page: {doc.metadata.get('page', 'Unknown')}")
    print("Content:")
    print(doc.page_content)
    print("-" * 50)

print("\nRetrieval test completed.")
