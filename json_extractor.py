"""
JSON-based Text Extraction Module.
Extracts text from pre-converted PDF JSON (structured layout data).
Much faster and more accurate than OCR.
"""

import json
import os
from config import INDEX_DIR


# Path to the converted JSON
JSON_PATH = "/home/vispl/Downloads/326880829_Fundamentals_of_Power_Electronics_Robert_W_Erickson_Dragan_removed (1).json"


def extract_text_from_json(json_path=None):
    """
    Extract text from the structured JSON file.
    Returns list of (page_number, text) tuples.
    """
    if json_path is None:
        json_path = JSON_PATH

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found at: {json_path}")

    print(f"[INFO] Loading JSON: {os.path.basename(json_path)}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pages = []
    total_pages = len(data.get("pages", []))
    print(f"[INFO] Total pages in JSON: {total_pages}")

    for page_data in data["pages"]:
        page_num = page_data["page_id"]
        content_blocks = page_data.get("content", [])

        # Sort blocks by vertical position (top to bottom)
        content_blocks.sort(key=lambda b: b["position"][1] if b.get("position") else 0)

        # Extract text from each block, filtering out low-quality/empty blocks
        page_texts = []
        for block in content_blocks:
            text = block.get("text", "").strip()
            block_type = block.get("type", "")
            score = block.get("score", 0)

            # Skip empty or whitespace-only blocks
            if not text or len(text) < 3:
                continue

            # Skip very low confidence blocks
            if score < 0.5 and block_type not in ("title", "paragraph"):
                continue

            # Add section markers for titles to help retrieval
            if block_type == "title":
                page_texts.append(f"\n{text}\n")
            elif block_type == "figure_caption":
                page_texts.append(f"[Figure: {text}]")
            else:
                page_texts.append(text)

        full_text = "\n".join(page_texts).strip()

        if full_text and len(full_text) > 30:
            pages.append((page_num, full_text))

    print(f"[INFO] Extracted text from {len(pages)} pages.")
    return pages


def chunk_text(pages, chunk_size=1024, overlap=128):
    """
    Split extracted pages into overlapping chunks.
    Each chunk includes metadata about source page.
    Returns list of dicts: {text, page, chunk_id}
    """
    chunks = []
    chunk_id = 0

    for page_num, text in pages:
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text_content = text[start:end]

            # Try to break at sentence/paragraph boundary
            if end < len(text) and len(chunk_text_content) > 100:
                # Look for last period, newline, or sentence end
                last_break = max(
                    chunk_text_content.rfind(". "),
                    chunk_text_content.rfind(".\n"),
                    chunk_text_content.rfind("\n\n"),
                )
                if last_break > chunk_size * 0.5:  # Only break if past halfway
                    chunk_text_content = chunk_text_content[:last_break + 1]
                    end = start + last_break + 1

            if len(chunk_text_content.strip()) > 30:
                chunks.append({
                    "text": chunk_text_content.strip(),
                    "page": page_num,
                    "chunk_id": chunk_id,
                })
                chunk_id += 1

            start = end - overlap

    print(f"[INFO] Created {len(chunks)} text chunks.")
    return chunks


if __name__ == "__main__":
    pages = extract_text_from_json()
    for page_num, text in pages[:2]:
        print(f"\n--- Page {page_num} ---")
        print(text[:500])
    
    chunks = chunk_text(pages)
    if chunks:
        print(f"\n\n--- Sample Chunk (page {chunks[0]['page']}) ---")
        print(chunks[0]['text'][:400])
