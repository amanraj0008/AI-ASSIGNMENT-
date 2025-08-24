# app.py
import os
import io
import time
import shutil
from pathlib import Path
from typing import List, Dict, Any

import streamlit as st

# ---- Core NLP / RAG deps (local-friendly) ----
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

# ---- PDF parsing (fast & reliable) ----
import fitz  # PyMuPDF

# ---- OpenAI (optional) ----
OPENAI_AVAILABLE = False
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

# ---- HuggingFace Transformers fallback (local) ----
HF_AVAILABLE = False
try:
    from transformers import pipeline
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False


# =========================
# Configuration
# =========================
st.set_page_config(
    page_title="PDF RAG (PDF-only)",
    page_icon="📄",
    layout="wide"
)

# Persistent directories
PERSIST_DIR = Path("./chroma_db")
UPLOADS_DIR = Path("./uploads")
PERSIST_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# Embeddings model (small, fast, widely cached)
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking defaults
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Retrieval defaults
TOP_K = 4

# =========================
# Helpers
# =========================
@st.cache_resource(show_spinner=False)
def get_embeddings():
    # CPU-friendly local embeddings
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

def _read_pdf_bytes(pdf_bytes: bytes, file_name: str) -> List[Document]:
    """
    Parse a single PDF (bytes) into per-page Documents with metadata: source, page, total_pages.
    """
    docs: List[Document] = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        total_pages = doc.page_count
        for i in range(total_pages):
            page = doc.load_page(i)
            text = page.get_text("text")
            if not text or not text.strip():
                # Fallback: extract blocks
                text = page.get_text("blocks")
                text = "\n".join([b[4] for b in text]) if text else ""
            metadata = {
                "source": file_name,
                "page": i + 1,
                "total_pages": total_pages,
            }
            docs.append(Document(page_content=text or "", metadata=metadata))
    return docs

def _chunk_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)

@st.cache_resource(show_spinner=False)
def get_vectorstore():
    # Chroma persistent vector DB
    embeddings = get_embeddings()
    vs = Chroma(
        persist_directory=str(PERSIST_DIR),
        embedding_function=embeddings
    )
    return vs

def clear_vectorstore():
    if PERSIST_DIR.exists():
        shutil.rmtree(PERSIST_DIR)
    PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    # Also clear Streamlit cache for vectorstore & embeddings
    st.cache_resource.clear()

def save_uploaded_file(uploaded_file) -> Path:
    # Saves uploaded file to persistent disk
    dest = UPLOADS_DIR / uploaded_file.name
    with open(dest, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return dest

def format_sources(docs: List[Document]) -> str:
    lines = []
    for idx, d in enumerate(docs, 1):
        src = d.metadata.get("source", "unknown.pdf")
        page = d.metadata.get("page", "?")
        snippet = (d.page_content or "").strip().replace("\n", " ")
        if len(snippet) > 320:
            snippet = snippet[:320] + "..."
        lines.append(f"**[{idx}] {src} — p.{page}**\n> {snippet}")
    return "\n\n".join(lines)

def openai_chain_answer(user_q: str, contexts: List[Document]) -> str:
    """
    Use OpenAI Chat Completions with strict, grounded prompt.
    Requires OPENAI_API_KEY in env.
    """
    client = OpenAI()
    context_text = "\n\n---\n\n".join([c.page_content for c in contexts])
    sys_prompt = (
        "You are a helpful assistant that answers strictly using the provided context from PDFs. "
        "If the answer is not found in the context, respond with: "
        "\"I couldn't find that in the uploaded PDFs.\" "
        "Cite sources as [n] referencing the order of the provided contexts."
    )
    # Compose citations inline like [1], [2] based on order
    user_prompt = (
        f"Question: {user_q}\n\n"
        f"Context (ordered):\n{context_text}\n\n"
        f"Instructions:\n"
        f"- Only use the context to answer.\n"
        f"- When you use a piece of context, cite it as [n] where n is the index above.\n"
        f"- If unknown from context, say you couldn't find it."
    )
    resp = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.0,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content.strip()

@st.cache_resource(show_spinner=False)
def get_hf_qa_pipeline():
    # Light-weight extractive QA pipeline; downloads once
    # You can change to 'deepset/roberta-base-squad2' if preferred
    return pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

def hf_chain_answer(user_q: str, contexts: List[Document]) -> str:
    """
    Extractive QA on concatenated top chunks using HF pipeline.
    """
    qa = get_hf_qa_pipeline()
    # Concatenate top chunks into a single context (trim if too long)
    ctx = "\n\n".join([c.page_content for c in contexts])[:4000]
    if not ctx.strip():
        return "I couldn't find that in the uploaded PDFs."
    try:
        ans = qa(question=user_q, context=ctx)
        text = ans.get("answer", "").strip()
        if not text:
            return "I couldn't find that in the uploaded PDFs."
        return text
    except Exception:
        return "I couldn't find that in the uploaded PDFs."

def answer_with_rag(user_q: str, k: int = TOP_K) -> Dict[str, Any]:
    """
    1) Embed question
    2) Retrieve top-k chunks
    3) Generate answer via OpenAI (if key present) else HF extractive QA
    """
    vs = get_vectorstore()
    retriever = vs.as_retriever(search_kwargs={"k": k})
    retrieved = retriever.get_relevant_documents(user_q)

    if os.getenv("OPENAI_API_KEY") and OPENAI_AVAILABLE:
        answer = openai_chain_answer(user_q, retrieved)
    elif HF_AVAILABLE:
        answer = hf_chain_answer(user_q, retrieved)
    else:
        # Worst-case fallback: return stitched extractive context
        stitched = "\n\n".join([d.page_content for d in retrieved])[:1500]
        answer = stitched if stitched else "I couldn't find that in the uploaded PDFs."
    return {"answer": answer, "contexts": retrieved}

def validate_pdf_file(uploaded_file) -> bool:
    # Enforce PDF only by MIME + extension
    if uploaded_file.type != "application/pdf":
        return False
    if not uploaded_file.name.lower().endswith(".pdf"):
        return False
    return True

def index_pdfs(files: List[Any]) -> Dict[str, Any]:
    """
    Ingest & index uploaded PDFs into Chroma (persistent).
    """
    vs = get_vectorstore()
    embeddings = get_embeddings()
    total_pages = 0
    total_chunks = 0
    added: List[str] = []

    for f in files:
        if not validate_pdf_file(f):
            st.warning(f"Skipped non-PDF: {f.name}")
            continue

        # Save to disk
        saved_path = save_uploaded_file(f)
        added.append(saved_path.name)

        # Read and chunk
        docs_per_page = _read_pdf_bytes(f.getvalue(), f.name)
        total_pages += len(docs_per_page)
        chunks = _chunk_documents(docs_per_page)
        total_chunks += len(chunks)

        # Add to vector store with metadata
        # Note: Chroma handles batching internally
        vs.add_documents(chunks, embedding=embeddings)

    # Persist Chroma on disk
    vs.persist()
    return {
        "files": added,
        "pages": total_pages,
        "chunks": total_chunks
    }


# =========================
# UI
# =========================
st.title("📄 PDF RAG — PDF-Only Doc QA")
st.caption("Upload PDFs → Index → Ask questions. Uses local embeddings (MiniLM) + Chroma persistence. "
           "Generation: OpenAI (if API key set) or local extractive QA fallback.")

with st.sidebar:
    st.header("⚙️ Settings")
    st.write("**Embeddings:** sentence-transformers/all-MiniLM-L6-v2 (local)")
    st.write("**Vector DB:** Chroma (persistent)")
    st.write("**LLM:** OpenAI Chat Completions (if `OPENAI_API_KEY` is set) → else HF extractive QA fallback")

    st.divider()
    k = st.slider("Top-k chunks", 2, 10, TOP_K, 1)
    chunk_size = st.number_input("Chunk size", 300, 3000, CHUNK_SIZE, 50)
    chunk_overlap = st.number_input("Chunk overlap", 0, 1000, CHUNK_OVERLAP, 25)
    apply_chunking = st.button("Apply Chunking Settings")

    if apply_chunking:
        global CHUNK_SIZE, CHUNK_OVERLAP
        CHUNK_SIZE = int(chunk_size)
        CHUNK_OVERLAP = int(chunk_overlap)
        st.success("Chunking settings updated. Re-ingest PDFs to apply.")

    st.divider()
    if st.button("🧹 Clear Vector DB"):
        clear_vectorstore()
        st.success("Cleared Chroma DB. You can re-ingest PDFs now.")

    st.divider()
    openai_key_present = bool(os.getenv("OPENAI_API_KEY"))
    st.write(f"OpenAI key detected: {'✅' if openai_key_present else '❌'}")
    if not openai_key_present:
        st.info("Set environment variable `OPENAI_API_KEY` to enable OpenAI RAG generation.", icon="🔑")

    st.caption("Note: This app **accepts PDF format only**. Other formats are rejected.")

tab_ingest, tab_chat = st.tabs(["📥 Upload & Ingest PDFs", "💬 Ask Questions"])

with tab_ingest:
    st.subheader("Upload your PDFs")
    uploaded_files = st.file_uploader(
        "Drag-and-drop or browse — PDF files only",
        type=["pdf"],  # Enforces PDF-only in UI
        accept_multiple_files=True
    )

    col_a, col_b = st.columns([1, 1])
    with col_a:
        do_ingest = st.button("📚 Ingest / Update Index", type="primary")
    with col_b:
        do_list = st.button("📁 List Uploaded Files")

    if do_ingest:
        if not uploaded_files:
            st.warning("Please select at least one PDF.")
        else:
            with st.spinner("Indexing PDFs..."):
                stats = index_pdfs(uploaded_files)
                st.success(
                    f"Ingested: {len(stats['files'])} file(s) | Pages: {stats['pages']} | Chunks: {stats['chunks']}"
                )
                st.write("Files:", ", ".join(stats["files"]))

    if do_list:
        files = sorted([p.name for p in UPLOADS_DIR.glob("*.pdf")])
        if files:
            st.write("**Uploaded PDFs (persistent):**")
            for f in files:
                st.write(f"- {f}")
        else:
            st.info("No PDFs uploaded yet.")

with tab_chat:
    st.subheader("Ask a question about your PDFs")
    query = st.text_input("Your question", placeholder="e.g., What is the main conclusion in the report?")
    col1, col2 = st.columns([1, 1])
    with col1:
        ask_btn = st.button("🔎 Retrieve & Answer", type="primary")
    with col2:
        show_ctx = st.checkbox("Show retrieved sources", value=True)

    if ask_btn:
        if not query.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Thinking..."):
                result = answer_with_rag(query, k=k)
            st.markdown("### ✅ Answer")
            st.write(result["answer"])

            if show_ctx:
                st.markdown("### 📚 Sources")
                st.markdown(format_sources(result["contexts"]))

    st.divider()
    st.caption("Tip: Answers are grounded in your uploaded PDFs only. If not found in context, the app will tell you.")


# =========================
# Footer / Notes
# =========================
with st.expander("ℹ️ Notes & How to Run"):
    st.markdown(
        """
**Run locally**
```bash
# 1) Create venv (recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate

# 2) Install requirements
pip install -r requirements.txt

# 3) (Optional) Enable OpenAI responses
export OPENAI_API_KEY=sk-...   # Windows PowerShell: $env:OPENAI_API_KEY="sk-..."

# 4) Start app
streamlit run app.py
