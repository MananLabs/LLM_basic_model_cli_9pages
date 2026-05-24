"""
Domain Guard Module.
Enforces strict domain boundaries - rejects off-topic queries.
"""

import re
from config import DOMAIN_KEYWORDS, PROHIBITED_PATTERNS


# The standard refusal message
REFUSAL_MESSAGE = (
    "ERROR: This query is outside the scope of the engineering dataset. "
    "Please ask questions strictly related to the provided engineering book."
)


def is_prohibited(query):
    """
    Check if query matches any prohibited pattern.
    Returns True if the query should be REFUSED.
    """
    query_lower = query.lower().strip()

    # Check against prohibited patterns
    for pattern in PROHIBITED_PATTERNS:
        if pattern in query_lower:
            return True

    return False


def is_domain_relevant(query):
    """
    Check if query contains domain-relevant keywords.
    Returns True if the query appears to be about power electronics.
    """
    query_lower = query.lower().strip()

    # Check for domain keywords
    for keyword in DOMAIN_KEYWORDS:
        if keyword in query_lower:
            return True

    return False


def has_engineering_context(query):
    """
    Additional heuristic: check if query has engineering-like structure.
    Looks for patterns like equations, units, technical terms.
    """
    query_lower = query.lower()

    # Engineering patterns
    engineering_patterns = [
        r'\d+\s*(v|volt|amp|watt|ohm|hz|khz|mhz)',  # values with units
        r'(derive|calculate|explain|analyze|find|determine|design)',  # action verbs
        r'(equation|formula|expression|relation)',  # math terms
        r'(circuit|converter|topology|waveform)',  # circuit terms
        r'(chapter|section|figure|table|example)',  # book references
        r'(steady.?state|transient|dynamic|small.?signal)',  # analysis types
        r'(input|output|transfer|gain|phase|margin)',  # system terms
    ]

    for pattern in engineering_patterns:
        if re.search(pattern, query_lower):
            return True

    return False


def validate_query(query):
    """
    Main validation function.
    Returns (is_valid, message).
    - is_valid=True: query is allowed, proceed with retrieval
    - is_valid=False: query is rejected, return the refusal message
    """
    if not query or not query.strip():
        return False, REFUSAL_MESSAGE

    # Rule 1: Check prohibited patterns first
    if is_prohibited(query):
        return False, REFUSAL_MESSAGE

    # Rule 2: Check domain relevance
    if is_domain_relevant(query):
        return True, None

    # Rule 3: Check engineering context heuristics
    if has_engineering_context(query):
        return True, None

    # Rule 4: If query is very short and has no domain signals, reject
    if len(query.split()) < 3:
        return False, REFUSAL_MESSAGE

    # Rule 5: For longer queries without clear domain signals,
    # allow them through to retrieval (retriever will handle low-confidence)
    # But apply a stricter check - at least some technical content
    technical_words = [
        "how", "what", "why", "when", "where", "which",
        "explain", "describe", "derive", "calculate", "compare",
        "difference", "between", "operation", "principle", "method",
        "analysis", "design", "model", "system", "control",
    ]

    query_words = query.lower().split()
    has_technical = any(w in query_words for w in technical_words)

    if has_technical and len(query_words) >= 4:
        # Allow through - retriever confidence will filter
        return True, None

    return False, REFUSAL_MESSAGE


def validate_retrieval_confidence(results, threshold=0.25):
    """
    Post-retrieval validation.
    If the best retrieval score is below threshold,
    the dataset likely doesn't contain relevant information.
    """
    if not results:
        return False

    best_score = max(r["score"] for r in results)
    return best_score >= threshold


if __name__ == "__main__":
    # Test cases
    test_queries = [
        "Tell me a joke",
        "Who is the Prime Minister of India?",
        "Write me a poem",
        "Explain the buck converter steady-state operation",
        "What is the duty cycle of a boost converter?",
        "How does a flyback converter work?",
        "What is your favorite color?",
        "Calculate the output voltage ripple",
        "Hello how are you",
    ]

    print("DOMAIN GUARD TEST")
    print("=" * 60)
    for q in test_queries:
        is_valid, msg = validate_query(q)
        status = "ALLOWED" if is_valid else "REJECTED"
        print(f"[{status}] {q}")
