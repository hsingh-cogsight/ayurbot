import os
from langchain_community.document_loaders import PyPDFLoader  # Corrected import
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS  # Corrected import
from langchain.embeddings.base import Embeddings

# Define a custom embedding wrapper for LangChain
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=True)
    
    def embed_query(self, text):
        return self.model.encode([text], show_progress_bar=False)[0]

def prepare_backend(pdf_path, faiss_index_path="faiss_index"):
    # Step 1: Load PDF and Extract Text
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Step 2: Split Text into Chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    # Step 3: Use LangChain-compatible Embeddings
    embedding_model = SentenceTransformerEmbeddings()
    texts = [chunk.page_content for chunk in chunks]
    db = FAISS.from_texts(texts, embedding_model)

    # Save the FAISS index
    db.save_local(faiss_index_path)
    print(f"FAISS index saved to: {faiss_index_path}")

# Path to your PDF
pdf_path = "AyurvedaBook.pdf"  # Replace with your PDF file path
prepare_backend(pdf_path)
