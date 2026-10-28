
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DATA_DIR = Path("data/documents")
DB_DIR = Path("chroma_db")

def build_vector_database():
    pdf_files = list(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF documents found in {DATA_DIR.resolve()}"
        )

    documents = []

    for pdf_file in pdf_files:
        print(f"Loading: {pdf_file.name}")
        loader = PyMuPDFLoader(str(pdf_file))
        documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120
    )

    chunks = splitter.split_documents(documents)

    print(f"\nLoaded {len(documents)} PDF pages.")
    print(f"Created {len(chunks)} text chunks.")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(DB_DIR),
        collection_name="wholesale_documents"
    )

    print(f"Vector database saved in: {DB_DIR.resolve()}")
    return vector_db


if __name__ == "__main__":
    build_vector_database()
