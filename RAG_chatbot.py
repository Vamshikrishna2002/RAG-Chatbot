import os
import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
import faiss
import numpy as np

# 🔑 Load Gemini API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# --------------------------
# Helper functions
# --------------------------
def load_pdf(file):
    """Extract text from PDF file"""
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() or ""
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks"""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def embed_texts(texts):
    """Get embeddings from Gemini API"""
    embeddings = []
    for txt in texts:
        res = genai.embed_content(
            model="gemini-embedding-001",
            content=txt
        )
        embeddings.append(res["embedding"])
    return np.array(embeddings).astype("float32")

def build_faiss_index(chunks):
    """Build FAISS index for fast similarity search"""
    embeddings = embed_texts(chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings

def search_index(query, index, chunks, k=3):
    """Search FAISS index for relevant chunks"""
    q_emb = embed_texts([query])
    D, I = index.search(q_emb, k)
    return [chunks[i] for i in I[0]]

def ask_gemini(question, context, history):
    """Ask Gemini with context and chat history"""
    prompt = f"""You are a helpful assistant. 
Use the following context from the PDF to answer the user’s question.

Context:
{context}

Chat history:
{history}

Question: {question}
Answer:"""

    response = genai.GenerativeModel("gemini-2.0-flash").generate_content(prompt)
    return response.text

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="📄 PDF RAG Chatbot", layout="wide")
st.title("📄 PDF RAG Chatbot with Gemini")

# Sidebar for PDF upload
uploaded_file = st.sidebar.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    text = load_pdf(uploaded_file)
    chunks = chunk_text(text)
    index, embeddings = build_faiss_index(chunks)

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = ""

    # User input
    question = st.chat_input("Ask a question about the PDF...")
    if question:
        results = search_index(question, index, chunks)
        context = "\n".join(results)
        answer = ask_gemini(question, context, st.session_state.chat_history)

        # Update history
        st.session_state.chat_history += f"\nUser: {question}\nBot: {answer}"

        # Show conversation
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            st.write(answer)
else:
    st.info("👆 Upload a PDF to get started.")





