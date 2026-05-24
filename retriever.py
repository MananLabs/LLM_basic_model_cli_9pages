"""
Retrieval Module.
Handles semantic search over the FAISS index.
"""

import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDING_MODEL, FAISS_INDEX_PATH,
    CHUNKS_PATH, METADATA_PATH, TOP_K
)


class Retriever:
    """Semantic retrieval engine over the indexed engineering book."""

    def __init__(self):
        self.model = None
        self.index = None
        self.chunks = None
        self.metadata = None
        self._loaded = False

    def load(self):
        """Load the FAISS index, chunks, and embedding model."""
        if self._loaded:
            return

        if not os.path.exists(FAISS_INDEX_PATH):
            raise FileNotFoundError(
                "Vector index not found. Run 'python indexer.py' first to build the index."
            )

        print("[INFO] Loading embedding model...")
        self.model = SentenceTransformer(EMBEDDING_MODEL)

        print("[INFO] Loading FAISS index...")
        self.index = faiss.read_index(FAISS_INDEX_PATH)

        print("[INFO] Loading chunks...")
        self.chunks = np.load(CHUNKS_PATH, allow_pickle=True)

        print("[INFO] Loading metadata...")
        self.metadata = np.load(METADATA_PATH, allow_pickle=True)

        self._loaded = True
        print(f"[INFO] Retriever ready. Index contains {self.index.ntotal} vectors.")

    def retrieve(self, query, top_k=None):
        """
        Retrieve top-k most relevant chunks for a query.
        Returns list of dicts: {text, page, score, chunk_id}
        """
        if not self._loaded:
            self.load()

        if top_k is None:
            top_k = TOP_K

        # Encode query
        query_embedding = self.model.encode(
            [query], normalize_embeddings=True
        ).astype("float32")

        # Search
        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0:
                continue
            results.append({
                "text": str(self.chunks[idx]),
                "page": int(self.metadata[idx][0]),
                "chunk_id": int(self.metadata[idx][1]),
                "score": float(scores[0][i]),
            })

        return results

    def get_context(self, query, top_k=None):
        """
        Get concatenated context string from top-k results.
        Used as input context for the response generator.
        """
        results = self.retrieve(query, top_k)

        if not results:
            return None, []

        context_parts = []
        for r in results:
            context_parts.append(
                f"[Page {r['page']}] {r['text']}"
            )

        context = "\n\n---\n\n".join(context_parts)
        return context, results


if __name__ == "__main__":
    retriever = Retriever()
    retriever.load()

    test_query = "buck converter steady state analysis"
    print(f"\n[TEST] Query: {test_query}")
    print("-" * 60)

    results = retriever.retrieve(test_query)
    for i, r in enumerate(results):
        print(f"\n[Result {i+1}] Score: {r['score']:.4f} | Page: {r['page']}")
        print(f"  {r['text'][:150]}...")
