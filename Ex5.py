"""
PDF Chatbot – Streamlit app
- Uses FREE Llama-2-7B (GGML) via CTransformers  – model loaded only once
- Embeds PDF with HuggingFace ‘all-MiniLM-L6-v2’   – embeddings built only once per file
- Keeps chain in session_state so **every next question answers instantly**
"""

import os, hashlib, logging, streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# ---------- logging ----------
logging.basicConfig(filename="chatbot.log",
                    level=logging.INFO,
                    format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)

# ---------- page ----------
st.set_page_config(page_title="PDF Chatbot", layout="wide")
st.header("📄 My First Chatbot")

# ---------- helpers ----------
def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

def pdf_hash(data) -> str:
    return hashlib.sha256(data.getbuffer()).hexdigest()[:16]

# ---------- sidebar ----------
with st.sidebar:
    st.title("Your Documents")
    uploaded = st.file_uploader("Upload a PDF file", type="pdf")

# ---------- session keys ----------
if "pdf_id" not in st.session_state:
    st.session_state.pdf_id = None
# NEW: cache the complete chain so we do NOT rebuild it every run
if "chain" not in st.session_state:
    st.session_state.chain = None

# ---------- FREE LLM (cached globally) ----------
@st.cache_resource(show_spinner=False)   # ← one load per session
def load_llama():
    from langchain_community.llms import CTransformers
    return CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGML",
        model_file="llama-2-7b-chat.ggmlv3.q4_0.bin",   # local path ok too
        config={'max_new_tokens': 256, 'temperature': 0.0}
    )

# ---------- main ----------
if uploaded is None:
    st.info("👆 Upload a PDF to start")
    st.stop()

uid = pdf_hash(uploaded)

# NEW: only re-process when user uploads a **different** PDF
if uid != st.session_state.pdf_id:
    log.info("New PDF uploaded %s", uid)
    st.session_state.pdf_id = uid
    st.session_state.chain = None          # wipe old chain

    # 1. extract text
    reader = PdfReader(uploaded)
    text = "".join(page.extract_text() or "" for page in reader.pages)
    if not text.strip():
        st.error("❌ Could not extract text from this PDF.")
        st.stop()

    # 2. chunk
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n"], chunk_size=1000, chunk_overlap=150
    )
    chunks = splitter.split_text(text)

    # 3. embed & index
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vs = FAISS.from_texts(chunks, embeddings)
    retriever = vs.as_retriever(search_kwargs={"k": 1})

    # 4. build chain **once** and store in session_state
    llm = load_llama()
    tmpl = ChatPromptTemplate.from_template(
        "Context:\n{context}\n\n"
        "Use only the context above. If the answer is not in the context, say 'I don't know.'\n\n"
        "Question: {question}\nAnswer:"
    )
    st.session_state.chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | tmpl
        | llm
        | StrOutputParser()
    )
    log.info("Vector store & chain ready for %s", uid)

# ---------- Q&A ----------
chain = st.session_state.chain      # pull cached chain
question = st.text_input("Ask a question about your PDF")
if question:
    with st.spinner("🤖 Thinking…"):
        answer = chain.invoke(question)   # fast: no re-loading, no re-indexing
    log.info("Q: %s  | A: %s", question, answer)
    st.success("Answer")
    st.write(answer)