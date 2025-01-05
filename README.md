# Ayurveda Chatbot using LLaMA and RAG

This project is an interactive Ayurveda chatbot that uses a **Retrieval-Augmented Generation (RAG)** pipeline powered by the **LLaMA language model via Ollama**. The chatbot provides Ayurvedic knowledge and answers user queries based on pre-trained PDF content.

---

## Features

- **PDF Knowledge Base**: Pretrained on Ayurvedic texts for domain-specific answers.
- **RAG Pipeline**: Combines FAISS vector retrieval and LLaMA for context-aware responses.
- **Streamlit Interface**: Easy-to-use frontend for interacting with the chatbot.

---

## Requirements

- Python 3.8+
- GPU support (optional but recommended for faster LLM inference)
- LLaMA model via [Ollama](https://ollama.ai)

---

## Installation

### 1. Clone the Repository
```bash
git clone git@git.digimantra.com:DigiMantra/AyurvedaChatbot.git
```


### 2. Create and Activate a Virtual Environment
On Linux/macOS:
```bash
python3 -m venv env
source env/bin/activate
```
On Windows:
```
python -m venv env
env\Scripts\activate
```
3. Install Dependencies
```bash
pip install -r requirements.txt
```

4. Install Ollama and Pull LLaMA Model
Download and install Ollama from here.

Pull the desired LLaMA model:

```bash
ollama pull llama3.2:3b
```

5. Start the Ollama Server
Ensure the Ollama server is running:

```bash
ollama serve
```

Usage
1. Preprocess PDF and Create FAISS Index
Ensure the PDF file (e.g., ayurveda_text.pdf) is placed in the project directory.

Run the backend script to preprocess the data and create a FAISS index:

```bash
python3 backend.py
```

2. Start the Chatbot
Launch the Streamlit interface:

```bash
streamlit run frontend.py
```
Access the chatbot in your browser at http://localhost:8501.
