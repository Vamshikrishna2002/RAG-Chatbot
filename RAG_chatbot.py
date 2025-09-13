import os
import streamlit as st
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# --------------------------
# Config
# --------------------------
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
EMBED_DIM = EMBED_MODEL.get_sentence_embedding_dimension()
api_key = os.getenv("GEMINI_API_KEY")  # load from env
if not api_key:
    st.error("❌ Please set your GEMINI_API_KEY environment variable.")
else:
    genai.configure(api_key=api_key)


# --------------------------
# Functions
# --------------------------
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def make_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

def embed_texts(texts):
    return EMBED_MODEL.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype("float32")

def build_index(vectors):
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(vectors)
    return index

def retrieve(query, index, chunks, k=3):
    qvec = embed_texts([query])
    D, I = index.search(qvec, k)
    return [chunks[i] for i in I[0]]

def ask_gemini(question, chunks, chat_history):
    context = "\n\n".join(chunks)
    history_text = "\n".join([f"Q: {q}\nA: {a}" for q, a in chat_history])

    prompt = f"""You are a helpful PDF assistant.
Here is the conversation so far:
{history_text}

Relevant PDF context:
{context}

Now answer the new question: {question}"""

    model = genai.GenerativeModel("gemini-2.0-flash")
    return model.generate_content(prompt).text
# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="📄 PDF RAG Chatbot", layout="wide")
st.title("📄 PDF RAG Chatbot (Gemini)")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "index" not in st.session_state:
    st.session_state.index = None

# File uploader
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file and st.session_state.index is None:
    st.success("✅ PDF uploaded and processed!")
    text = extract_text_from_pdf(uploaded_file)
    chunks = make_chunks(text)
    vectors = embed_texts(chunks)
    index = build_index(vectors)
    st.session_state.chunks = chunks
    st.session_state.index = index

# Display chat history in ChatGPT style
for q, a in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        st.markdown(a)

# Chat input at the bottom
if st.session_state.index is not None:
    if question := st.chat_input("Ask something about your PDF..."):
        # Show user message
        with st.chat_message("user"):
            st.markdown(question)

        # Retrieve relevant chunks
        results = retrieve(question, st.session_state.index, st.session_state.chunks)

        # Get answer from Gemini
        answer = ask_gemini(question, results, st.session_state.chat_history)

        # Save to history
        st.session_state.chat_history.append((question, answer))

        # Show bot response
        with st.chat_message("assistant"):
            st.markdown(answer)