"""
Engineering Knowledge Intelligence System - Main CLI Interface.
Closed-domain RAG assistant using Ollama + Qwen3 1.7B.

Usage:
    python main.py          - Start interactive CLI
    python main.py --build  - Build/rebuild the vector index
"""

import sys
import time

from config import CONFIDENCE_THRESHOLD, OLLAMA_MODEL
from domain_guard import validate_query, validate_retrieval_confidence, REFUSAL_MESSAGE
from retriever import Retriever
from response_generator import (
    generate_response, format_response, format_insufficient_context,
)

BANNER = f"""
════════════════════════════════════════════════════════════════
  ENGINEERING KNOWLEDGE INTELLIGENCE SYSTEM
  Closed-Domain RAG | Ollama + {OLLAMA_MODEL}
════════════════════════════════════════════════════════════════
  Dataset: Fundamentals of Power Electronics
  Authors: Robert W. Erickson, Dragan Maksimovic (2e, 2001)
────────────────────────────────────────────────────────────────
  Commands: 'help' | 'quit'
════════════════════════════════════════════════════════════════
"""

HELP_TEXT = """
  ALLOWED TOPICS: DC-DC converters, buck/boost/flyback/Cuk/SEPIC,
  steady-state analysis, volt-second balance, small-signal modeling,
  transfer functions, Bode plots, feedback design, magnetics,
  switching losses, PWM, CCM/DCM, resonant converters, PFC.

  EXAMPLE QUERIES:
  > What is discontinuous conduction mode?
  > Explain the buck converter steady-state operation
  > Derive the conversion ratio of a boost converter
"""


def validate_startup():
    """Check that Ollama is reachable and model is available."""
    try:
        from ollama import list as ollama_list
        models = ollama_list()
        names = [m.model for m in models.models] if models.models else []
        if not any(OLLAMA_MODEL in n for n in names):
            print(f"  [WARN] Model '{OLLAMA_MODEL}' not found in Ollama.")
            print(f"  [WARN] Available: {names or 'none'}")
            print(f"  [WARN] Run: ollama pull {OLLAMA_MODEL}")
            return False
        return True
    except Exception as e:
        print(f"  [ERROR] Cannot connect to Ollama: {e}")
        print(f"  [ERROR] Ensure Ollama is running: ollama serve")
        return False


def run_cli():
    """Main interactive CLI loop."""
    print(BANNER)

    # Validate Ollama
    print("  Checking Ollama...", end=" ")
    if validate_startup():
        print("OK")
    else:
        print("\n  Continuing anyway (LLM calls will fail).\n")

    # Load retriever
    retriever = Retriever()
    try:
        retriever.load()
    except FileNotFoundError as e:
        print(f"\n  [ERROR] {e}")
        print("  [ERROR] Run: python indexer.py")
        sys.exit(1)

    print("\n  Ready. Ask your engineering question.\n")

    while True:
        try:
            query = input(">> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n  [SESSION ENDED]")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("\n  [SESSION ENDED]")
            break
        if query.lower() == "help":
            print(HELP_TEXT)
            continue

        # Step 1: Domain guard
        is_valid, refusal_msg = validate_query(query)
        if not is_valid:
            print(f"\n  {refusal_msg}\n")
            continue

        # Step 2: Retrieve
        t0 = time.time()
        context, results = retriever.get_context(query)
        retrieval_time = time.time() - t0

        # Step 3: Confidence check
        if not results or not validate_retrieval_confidence(results, CONFIDENCE_THRESHOLD):
            print(f"\n{format_insufficient_context(query, results, retrieval_time)}\n")
            continue

        # Step 4: Generate LLM response
        print("\n  Generating answer (this may take a minute on CPU)...")
        try:
            answer, generation_time = generate_response(query, context, results)
        except KeyboardInterrupt:
            print("\n  [Generation cancelled]\n")
            continue

        # Step 5: Format and display
        output = format_response(query, answer, results, retrieval_time, generation_time)
        print(f"\n{output}\n")


def build_index():
    """Build the vector index."""
    from indexer import build_index as _build
    _build()


if __name__ == "__main__":
    if "--build" in sys.argv:
        build_index()
    else:
        run_cli()
