"""
Web UI for Engineering Knowledge Intelligence System.
Flask backend serving a ChatGPT-style interface.
"""

import time
import sys
from flask import Flask, render_template, request, jsonify

from config import CONFIDENCE_THRESHOLD, OLLAMA_MODEL
from domain_guard import validate_query, validate_retrieval_confidence
from retriever import Retriever
from response_generator import generate_response

app = Flask(__name__)

# Initialize retriever once at startup
retriever = Retriever()
retriever.load()


def log(msg):
    """Print timestamped log to terminal."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"  [{timestamp}] {msg}")
    sys.stdout.flush()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat_endpoint():
    """Handle a user question and return the answer."""
    data = request.get_json()
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"error": "Empty query"}), 400

    log(f"QUERY: \"{query}\"")

    # Step 1: Domain guard
    is_valid, refusal_msg = validate_query(query)
    if not is_valid:
        log(f"REJECTED (off-topic)")
        return jsonify({
            "answer": refusal_msg,
            "sources": [],
            "relevance": 0,
            "retrieval_time": 0,
            "generation_time": 0,
            "status": "rejected"
        })

    # Step 2: Retrieve
    log("Retrieving context...")
    t0 = time.time()
    context, results = retriever.get_context(query)
    retrieval_time = time.time() - t0
    log(f"Retrieved {len(results)} chunks in {retrieval_time:.2f}s")

    # Step 3: Confidence check
    if not results or not validate_retrieval_confidence(results, CONFIDENCE_THRESHOLD):
        top_score = max(r["score"] for r in results) if results else 0.0
        log(f"LOW CONFIDENCE (best score: {top_score:.4f})")
        return jsonify({
            "answer": "The provided textbook data does not contain enough information to answer this question.",
            "sources": [],
            "relevance": round(top_score, 4),
            "retrieval_time": round(retrieval_time, 2),
            "generation_time": 0,
            "status": "low_confidence"
        })

    # Step 4: Generate LLM response
    log(f"Generating answer with {OLLAMA_MODEL} (CPU — please wait)...")
    answer, generation_time = generate_response(query, context, results)

    # Build sources
    pages = sorted(set(r["page"] for r in results))
    top_score = results[0]["score"] if results else 0.0

    log(f"DONE — {generation_time:.1f}s generation | relevance: {top_score:.4f}")
    log(f"ANSWER: {answer[:120]}...")
    log(f"SOURCES: {', '.join(f'Page {p}' for p in pages)}")
    print()  # blank line between queries

    return jsonify({
        "answer": answer,
        "sources": [f"Page {p}" for p in pages],
        "relevance": round(top_score, 4),
        "retrieval_time": round(retrieval_time, 2),
        "generation_time": round(generation_time, 2),
        "status": "success"
    })


@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint."""
    try:
        from ollama import list as ollama_list
        models = ollama_list()
        names = [m.model for m in models.models] if models.models else []
        ollama_ok = any(OLLAMA_MODEL in n for n in names)
    except Exception:
        ollama_ok = False

    return jsonify({
        "status": "ok",
        "model": OLLAMA_MODEL,
        "ollama_connected": ollama_ok,
        "index_loaded": retriever._loaded,
        "total_vectors": retriever.index.ntotal if retriever.index else 0
    })


if __name__ == "__main__":
    print(f"\n{'═' * 50}")
    print(f"  Power Electronics AI — Web UI")
    print(f"  Model: {OLLAMA_MODEL}")
    print(f"  Vectors: {retriever.index.ntotal}")
    print(f"{'═' * 50}")
    print(f"  Open: http://localhost:5000")
    print(f"{'═' * 50}\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
