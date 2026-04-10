"""
rag_engine.py
-------------
Handles document ingestion (PDF, DOCX, PPTX) and RAG retrieval.
Uses HuggingFace embeddings locally — no OpenAI key required.

Public API
----------
ingest_files(uploaded_files) -> dict   # {"docs": N, "chunks": N, "names": [...]}
query_rag(question, k=4)    -> dict   # {"context": str, "docs_used": N, "chunks_used": N, "sources": [...]}
get_rag_stats()             -> dict   # {"docs": N, "chunks": N, "names": [...]}
clear_rag()                 -> None
"""

import os
import tempfile
from typing import List

from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# ── Embeddings — runs fully locally, no API key needed ──────────────────────
# all-MiniLM-L6-v2 is small (80 MB), fast, and great for semantic search.
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# ── Module-level state ───────────────────────────────────────────────────────
_vector_store: FAISS | None = None
_ingested_files: list[str] = []
_total_chunks: int = 0

# ── Text splitter ─────────────────────────────────────────────────────────────
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=120,
    separators=["\n\n", "\n", ". ", " ", ""],
)


# ── File loader ───────────────────────────────────────────────────────────────

def _load_file(uploaded_file) -> List[Document]:
    """Write UploadedFile to a temp path, load with the right LangChain loader."""
    name: str = uploaded_file.name
    ext = name.rsplit(".", 1)[-1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        if ext == "pdf":
            loader = PyPDFLoader(tmp_path)
        elif ext in ("doc", "docx"):
            loader = UnstructuredWordDocumentLoader(tmp_path)
        elif ext in ("ppt", "pptx"):
            loader = UnstructuredPowerPointLoader(tmp_path)
        else:
            raise ValueError(f"Unsupported file type: .{ext}")

        docs = loader.load()
        for d in docs:
            d.metadata["source_file"] = name
        return docs
    finally:
        os.unlink(tmp_path)


# ── Public API ────────────────────────────────────────────────────────────────

def ingest_files(uploaded_files) -> dict:
    """
    Load, chunk, and index uploaded files into FAISS.
    Already-indexed files are skipped (incremental).
    """
    global _vector_store, _ingested_files, _total_chunks

    all_chunks: List[Document] = []

    for uf in uploaded_files:
        if uf.name in _ingested_files:
            continue
        raw_docs = _load_file(uf)
        chunks = _splitter.split_documents(raw_docs)
        all_chunks.extend(chunks)
        _ingested_files.append(uf.name)

    if not all_chunks:
        return {"docs": len(_ingested_files), "chunks": _total_chunks, "names": _ingested_files}

    if _vector_store is None:
        _vector_store = FAISS.from_documents(all_chunks, embeddings)
    else:
        _vector_store.add_documents(all_chunks)

    _total_chunks += len(all_chunks)

    return {
        "docs": len(_ingested_files),
        "chunks": _total_chunks,
        "names": _ingested_files,
    }


def query_rag(question: str, k: int = 4) -> dict:
    """Retrieve top-k chunks most relevant to question."""
    if _vector_store is None:
        return {"context": "", "chunks_used": 0, "docs_used": 0, "sources": []}

    results = _vector_store.similarity_search(question, k=k)

    context_parts = []
    sources = []
    seen_files = set()

    for i, doc in enumerate(results, 1):
        file_name = doc.metadata.get("source_file", "unknown")
        page = doc.metadata.get("page", None)
        context_parts.append(f"[Chunk {i} | {file_name}]:\n{doc.page_content}")
        sources.append({"file": file_name, "page": page})
        seen_files.add(file_name)

    return {
        "context": "\n\n".join(context_parts),
        "chunks_used": len(results),
        "docs_used": len(seen_files),
        "sources": sources,
    }


def get_rag_stats() -> dict:
    return {
        "docs": len(_ingested_files),
        "chunks": _total_chunks,
        "names": list(_ingested_files),
    }


def clear_rag() -> None:
    global _vector_store, _ingested_files, _total_chunks
    _vector_store = None
    _ingested_files = []
    _total_chunks = 0
