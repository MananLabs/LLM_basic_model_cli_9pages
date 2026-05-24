"""
Response Generator Module.
Generates structured engineering responses from retrieved context.
Uses template-based generation grounded strictly in retrieved content.
"""

import re


def format_engineering_response(query, context, results):
    """
    Generate a structured engineering response based on retrieved context.
    This is a retrieval-augmented response - content comes ONLY from the dataset.
    """
    if not context or not results:
        return (
            "INSUFFICIENT DATA: The dataset does not contain sufficient "
            "information to answer this query with engineering certainty."
        )

    # Extract page references
    pages = sorted(set(r["page"] for r in results))
    page_ref = ", ".join(str(p) for p in pages)

    # Build structured response
    response_parts = []

    response_parts.append("-" * 50)
    response_parts.append("QUERY UNDERSTANDING")
    response_parts.append("-" * 50)
    response_parts.append(f"Query: {query}")
    response_parts.append(f"Relevance Score: {results[0]['score']:.4f}")
    response_parts.append("")

    response_parts.append("-" * 50)
    response_parts.append("ENGINEERING CONTENT (FROM DATASET)")
    response_parts.append("-" * 50)

    # Present retrieved content in a clean format
    for i, r in enumerate(results):
        response_parts.append(f"\n[Source: Page {r['page']}]")
        # Clean up the text
        text = r["text"].strip()
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        response_parts.append(text)

    response_parts.append("")
    response_parts.append("-" * 50)
    response_parts.append("DATASET REFERENCE")
    response_parts.append("-" * 50)
    response_parts.append(f"Source: Fundamentals of Power Electronics")
    response_parts.append(f"Authors: Robert W. Erickson, Dragan Maksimovic")
    response_parts.append(f"Edition: 2nd Edition (2001)")
    response_parts.append(f"Referenced Pages: {page_ref}")

    return "\n".join(response_parts)


def format_low_confidence_response(query, results):
    """
    Response when retrieval confidence is low.
    """
    best_score = max(r["score"] for r in results) if results else 0.0

    return (
        f"LOW CONFIDENCE RESPONSE\n"
        f"{'-' * 50}\n"
        f"Query: {query}\n"
        f"Best match score: {best_score:.4f}\n\n"
        f"The dataset does not contain sufficiently relevant information\n"
        f"to provide a reliable engineering answer to this query.\n\n"
        f"Possible reasons:\n"
        f"- The topic may not be covered in this edition of the book\n"
        f"- The query may need to be rephrased using terminology from the book\n"
        f"- The specific detail requested may not be present in the dataset\n\n"
        f"Please rephrase your question or ask about a topic covered in:\n"
        f"'Fundamentals of Power Electronics' by Erickson & Maksimovic (2e, 2001)"
    )
