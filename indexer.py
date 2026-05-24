"""
Vector Indexing Module.
Creates FAISS vector index from text chunks using sentence embeddings.
"""

import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import (
    EMBEDDING_MODEL, INDEX_DIR, CHUNKS_PATH,
    FAISS_INDEX_PATH, METADATA_PATH, CHUNK_SIZE, CHUNK_OVERLAP
)
from pdf_extractor import extract_text_from_pdf, chunk_text


def build_index():
    """
    Full pipeline: extract PDF -> chunk -> embed -> build FAISS index.
    Saves index and metadata to disk.
    """
    # Step 1: Extract text
    print("=" * 60)
    print("STEP 1: Extracting text from PDF")
    print("=" * 60)
    pages = extract_text_from_pdf()

    # Step 2: Chunk text
    print("\n" + "=" * 60)
    print("STEP 2: Chunking text")
    print("=" * 60)
    chunks = chunk_text(pages, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

    # Step 3: Generate embeddings
    print("\n" + "=" * 60)
    print("STEP 3: Generating embeddings")
    print("=" * 60)
    print(f"[INFO] Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [c["text"] for c in chunks]
    print(f"[INFO] Encoding {len(texts)} chunks...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=64,
        normalize_embeddings=True
    )
    embeddings = np.array(embeddings, dtype="float32")
    print(f"[INFO] Embedding shape: {embeddings.shape}")

    # Step 4: Build FAISS index
    print("\n" + "=" * 60)
    print("STEP 4: Building FAISS index")
    print("=" * 60)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity with normalized vectors)
    index.add(embeddings)
    print(f"[INFO] FAISS index built with {index.ntotal} vectors.")

    # Step 5: Save to disk
    print("\n" + "=" * 60)
    print("STEP 5: Saving index to disk")
    print("=" * 60)
    os.makedirs(INDEX_DIR, exist_ok=True)

    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"[INFO] FAISS index saved: {FAISS_INDEX_PATH}")

    # Save chunk texts and metadata
    chunk_data = np.array([c["text"] for c in chunks], dtype=object)
    np.save(CHUNKS_PATH, chunk_data)
    print(f"[INFO] Chunks saved: {CHUNKS_PATH}")

    metadata = np.array([(c["page"], c["chunk_id"]) for c in chunks])
    np.save(METADATA_PATH, metadata)
    print(f"[INFO] Metadata saved: {METADATA_PATH}")

    print("\n" + "=" * 60)
    print("INDEX BUILD COMPLETE")
    print(f"Total chunks indexed: {len(chunks)}")
    print(f"Embedding dimension: {dimension}")
    print(f"Index directory: {INDEX_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    build_index()
