import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Define a custom embedding wrapper for LangChain
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=True)
    
    def embed_query(self, text):
        return self.model.encode([text], show_progress_bar=False)[0]

# Path to FAISS index
faiss_index_path = "faiss_index"  # Path where the FAISS index was saved in backend

# Load FAISS Index with dangerous deserialization enabled
embedding_model = SentenceTransformerEmbeddings()  # Use the same embedding model as in backend
db = FAISS.load_local(faiss_index_path, embedding_model, allow_dangerous_deserialization=True)

# Configure LLaMA via Ollama
llm = Ollama(
    model="llama3.2:3b",  # Updated to the desired model
    base_url="http://localhost:11434",  # Ensure Ollama server is running locally
)


# Define Prompt Template
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are an Ayurvedic expert with deep knowledge of Ayurvedic practices, remedies, and diagnostics. "
        "Use the provided Ayurvedic context to answer the question thoughtfully and accurately.\n\n"
        "Context:\n{context}\n\n"
        "Question:\n{question}\n\n"
        "Answer as an Ayurvedic expert:"
    )
)

# Define the RetrievalQA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(),
    return_source_documents=False,  # Do not return source documents
    chain_type_kwargs={"prompt": prompt_template},
)

# Streamlit UI
st.set_page_config(page_title="Ayurveda Chatbot", layout="wide")
st.title("Ayurveda Chatbot")

st.subheader("Ask your Ayurvedic Question")
query = st.text_input("Enter your query:")
if query:
    with st.spinner("Retrieving answer..."):
        response = qa_chain.run(query)
        st.markdown(f"### Answer:\n{response}")
