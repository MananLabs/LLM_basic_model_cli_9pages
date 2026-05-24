"""
Response Generator Module.
Generates grounded engineering responses using Ollama + Qwen3 1.7B.
Strictly closed-domain: answers ONLY from retrieved context.
"""

import time
from ollama import chat

from config import (
    OLLAMA_MODEL, OLLAMA_TIMEOUT,
    LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_TOP_P,
)

SYSTEM_PROMPT = """You are a closed-domain engineering assistant. You answer questions ONLY using the provided textbook context below.

STRICT RULES:
1. Use ONLY the information in the CONTEXT section to answer.
2. NEVER use outside knowledge, prior training data, or general knowledge.
3. NEVER invent formulas, equations, values, or citations not present in the context.
4. If the context does not contain enough information to answer, respond EXACTLY with: "The provided textbook data does not contain enough information to answer this question."
5. Keep answers concise, technical, and grounded.
6. Do NOT speculate or extrapolate beyond what the context states.
7. Reference page numbers from the context when relevant.

You are answering from: "Fundamentals of Power Electronics" by Erickson & Maksimovic (2nd Edition, 2001)."""

NO_CONTEXT_RESPONSE = "The provided textbook data does not contain enough information to answer this question."


def build_prompt(query, context):
    """Build the user message with retrieved context injected."""
    return (
        f"CONTEXT (from textbook):\n"
        f"---\n{context}\n---\n\n"
        f"QUESTION: {query}\n\n"
        f"Answer using ONLY the context above. Cite page numbers where applicable."
    )


def generate_response(query, context, results):
    """
    Call Ollama with grounded prompt and return the LLM response.
    Returns (answer_text, generation_time_seconds).
    """
    user_message = build_prompt(query, context)

    start = time.time()
    try:
        response = chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            options={
                "temperature": LLM_TEMPERATURE,
                "num_predict": LLM_MAX_TOKENS,
                "top_p": LLM_TOP_P,
            },
        )
        elapsed = time.time() - start
        answer = response.message.content.strip()
        return answer, elapsed
    except Exception as e:
        elapsed = time.time() - start
        return f"[LLM ERROR] {e}", elapsed


def format_response(query, answer, results, retrieval_time, generation_time):
    """Format the final structured output."""
    pages = sorted(set(r["page"] for r in results))
    top_score = results[0]["score"] if results else 0.0

    lines = []
    lines.append("─" * 60)
    lines.append(f"  Question: {query}")
    lines.append("─" * 60)
    lines.append("")
    lines.append(f"  Answer:\n")
    lines.append(f"  {answer}")
    lines.append("")
    lines.append("─" * 60)
    lines.append(f"  Sources: {', '.join(f'Page {p}' for p in pages)}")
    lines.append(f"  Relevance: {top_score:.4f} | Retrieval: {retrieval_time:.2f}s | Generation: {generation_time:.2f}s")
    lines.append("─" * 60)

    return "\n".join(lines)


def format_insufficient_context(query, results, retrieval_time):
    """Format response when confidence is too low."""
    top_score = max(r["score"] for r in results) if results else 0.0

    lines = []
    lines.append("─" * 60)
    lines.append(f"  Question: {query}")
    lines.append("─" * 60)
    lines.append("")
    lines.append(f"  Answer:\n")
    lines.append(f"  {NO_CONTEXT_RESPONSE}")
    lines.append("")
    lines.append("─" * 60)
    lines.append(f"  Best score: {top_score:.4f} (below threshold) | Retrieval: {retrieval_time:.2f}s")
    lines.append("─" * 60)

    return "\n".join(lines)
