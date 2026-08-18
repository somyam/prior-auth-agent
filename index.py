"""Build (or load a cached) FAISS index over the local CMS policy PDFs."""
import glob
import json
import os

import faiss
import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(__file__)
POLICY_DIR = os.path.join(BASE_DIR, "docs", "cms_policies")
CACHE_DIR = os.path.join(BASE_DIR, ".cache")
INDEX_PATH = os.path.join(CACHE_DIR, "policy_index.faiss")
CHUNKS_PATH = os.path.join(CACHE_DIR, "policy_chunks.json")


def _chunk_pdf(path: str, words_per_chunk: int = 500, stride: int = 450) -> list[str]:
    doc = fitz.open(path)
    text = "".join(page.get_text() for page in doc)
    doc.close()
    words = text.split()
    return [" ".join(words[i:i + words_per_chunk]) for i in range(0, len(words), stride)]


def build_index(force_rebuild: bool = False):
    """Returns (embedding_model, faiss_index, chunks, sources). Cached to disk after the first build."""
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    if not force_rebuild and os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
        faiss_index = faiss.read_index(INDEX_PATH)
        meta = json.load(open(CHUNKS_PATH))
        return embedding_model, faiss_index, meta["chunks"], meta["sources"]

    chunks, sources = [], []
    for pdf_path in sorted(glob.glob(os.path.join(POLICY_DIR, "*.pdf"))):
        pdf_chunks = _chunk_pdf(pdf_path)
        chunks.extend(pdf_chunks)
        sources.extend([os.path.basename(pdf_path)] * len(pdf_chunks))

    embeddings = embedding_model.encode(chunks)
    faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
    faiss_index.add(np.array(embeddings, dtype=np.float32))

    os.makedirs(CACHE_DIR, exist_ok=True)
    faiss.write_index(faiss_index, INDEX_PATH)
    with open(CHUNKS_PATH, "w") as f:
        json.dump({"chunks": chunks, "sources": sources}, f)

    return embedding_model, faiss_index, chunks, sources
