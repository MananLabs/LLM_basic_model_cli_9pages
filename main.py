"""
Engineering Knowledge Intelligence System - Main CLI Interface.

A closed-domain conversational interface for:
"Fundamentals of Power Electronics" by Robert W. Erickson & Dragan Maksimovic (2e, 2001)

Usage:
    python main.py          - Start interactive CLI
    python main.py --build  - Build/rebuild the vector index
"""

import sys
import os

from domain_guard import validate_query, validate_retrieval_confidence, REFUSAL_MESSAGE
from retriever import Retriever
from response_generator import format_engineering_response, format_low_confidence_response


BANNER = """
================================================================
  ENGINEERING KNOWLEDGE INTELLIGENCE SYSTEM
================================================================
  Dataset: Fundamentals of Power Electronics
  Authors: Robert W. Erickson, Dragan Maksimovic
  Edition: 2nd Edition (2001)
================================================================
  Type your engineering question below.
  Type 'quit' or 'exit' to end the session.
  Type 'help' for usage information.
================================================================
"""

HELP_TEXT = """
----------------------------------------------------------------
  USAGE GUIDE
----------------------------------------------------------------
  This system answers questions ONLY from the engineering book:
  "Fundamentals of Power Electronics" (Erickson & Maksimovic, 2e)

  ALLOWED TOPICS:
  - DC-DC converters (buck, boost, buck-boost, Cuk, SEPIC, etc.)
  - Steady-state converter analysis
  - Inductor volt-second balance
  - Capacitor charge balance
  - Small-signal modeling
  - Transfer functions and Bode plots
  - Feedback loop design and compensation
  - Magnetics design (inductors, transformers)
  - Switching losses and efficiency
  - Rectifiers and power factor correction
  - Resonant converters
  - Pulse-width modulation (PWM)
  - CCM and DCM operation

  EXAMPLE QUERIES:
  > Explain the buck converter steady-state operation
  > What is inductor volt-second balance?
  > Derive the conversion ratio of a boost converter
  > How does DCM affect the buck converter?
  > Explain small-signal AC modeling
  > What is the canonical model?

  OFF-TOPIC QUERIES WILL BE REJECTED.
----------------------------------------------------------------
"""


def run_cli():
    """Main interactive CLI loop."""
    print(BANNER)

    # Initialize retriever
    retriever = Retriever()
    try:
        retriever.load()
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("[ERROR] Please run: python indexer.py")
        print("[ERROR] This builds the vector index from the PDF.")
        sys.exit(1)

    print("\nSystem ready. Ask your engineering question.\n")

    while True:
        try:
            query = input(">> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n[SESSION ENDED]")
            break

        if not query:
            continue

        if query.lower() in ("quit", "exit", "q"):
            print("\n[SESSION ENDED]")
            break

        if query.lower() == "help":
            print(HELP_TEXT)
            continue

        # Step 1: Domain validation
        is_valid, refusal_msg = validate_query(query)
        if not is_valid:
            print(f"\n{refusal_msg}\n")
            continue

        # Step 2: Retrieve relevant content
        context, results = retriever.get_context(query)

        # Step 3: Validate retrieval confidence
        if not results or not validate_retrieval_confidence(results):
            response = format_low_confidence_response(query, results)
            print(f"\n{response}\n")
            continue

        # Step 4: Generate structured response
        response = format_engineering_response(query, context, results)
        print(f"\n{response}\n")


def build_index():
    """Build the vector index."""
    from indexer import build_index as _build
    _build()


if __name__ == "__main__":
    if "--build" in sys.argv:
        build_index()
    else:
        run_cli()
