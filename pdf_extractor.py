"""
PDF Text Extraction Module (OCR-based).
Extracts text from image-based PDF using EasyOCR.
Supports incremental extraction with progress saving.
"""

import os
import json
import io
import numpy as np
import fitz
import easyocr
from PIL import Image
from tqdm import tqdm
from config import PDF_PATH, INDEX_DIR


# Cache file for extracted text
TEXT_CACHE_PATH = os.path.join(INDEX_DIR, "extracted_text.json")
PROGRESS_PATH = os.path.join(INDEX_DIR, "ocr_progress.json")


def get_ocr_reader():
    """Initialize EasyOCR reader."""
    print("[INFO] Initializing OCR engine...")
    reader = easyocr.Reader(['en'], gpu=True, verbose=False)
    return reader


def extract_page_image(doc, page_idx, dpi=150):
    """Render a PDF page to a numpy image array."""
    page = doc[page_idx]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    img_bytes = pix.tobytes("png")
    img = Image.open(io.BytesIO(img_bytes))
    return np.array(img)


def ocr_page(reader, img_np):
    """Run OCR on a page image, return extracted text."""
    results = reader.readtext(img_np, detail=0, paragraph=True)
    return "\n".join(results)


def extract_text_from_pdf(pdf_path=None, use_cache=True, dpi=150, save_every=25):
    """
    Extract text from all pages using OCR.
    Saves progress incrementally every `save_every` pages.
    Returns list of (page_number, text) tuples.
    """
    if pdf_path is None:
        pdf_path = PDF_PATH

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at: {pdf_path}")

    os.makedirs(INDEX_DIR, exist_ok=True)

    # Check if full extraction is already cached
    if use_cache and os.path.exists(TEXT_CACHE_PATH):
        print(f"[INFO] Loading cached text from: {TEXT_CACHE_PATH}")
        with open(TEXT_CACHE_PATH, "r", encoding="utf-8") as f:
            cached = json.load(f)
        pages = [(item["page"], item["text"]) for item in cached]
        print(f"[INFO] Loaded {len(pages)} pages from cache.")
        return pages

    # Check for partial progress
    completed_pages = {}
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH, "r", encoding="utf-8") as f:
            progress_data = json.load(f)
        completed_pages = {item["page"]: item["text"] for item in progress_data}
        print(f"[INFO] Resuming from progress: {len(completed_pages)} pages already done.")

    # Open PDF
    print(f"[INFO] Loading PDF: {os.path.basename(pdf_path)}")
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    print(f"[INFO] Total pages: {total_pages}")
    print(f"[INFO] DPI: {dpi} | Save interval: every {save_every} pages")
    print(f"[INFO] Starting OCR extraction...")

    reader = get_ocr_reader()
    failed = 0
    new_count = 0

    for i in tqdm(range(total_pages), desc="OCR Processing"):
        page_num = i + 1

        # Skip already completed pages
        if page_num in completed_pages:
            continue

        try:
            img_np = extract_page_image(doc, i, dpi=dpi)
            text = ocr_page(reader, img_np)

            if text and len(text.strip()) > 30:
                completed_pages[page_num] = text.strip()
            else:
                completed_pages[page_num] = ""

            new_count += 1

        except Exception:
            completed_pages[page_num] = ""
            failed += 1

        # Save progress periodically
        if new_count > 0 and new_count % save_every == 0:
            _save_progress(completed_pages)

    doc.close()

    # Final save
    _save_progress(completed_pages)

    # Build final output (only non-empty pages)
    pages = [
        (p, t) for p, t in sorted(completed_pages.items())
        if t and len(t) > 30
    ]

    print(f"[INFO] OCR complete. Extracted text from {len(pages)} pages.")
    if failed > 0:
        print(f"[WARN] Failed on {failed} pages.")

    # Save final cache
    cache_data = [{"page": p, "text": t} for p, t in pages]
    with open(TEXT_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, ensure_ascii=False)
    print(f"[INFO] Final text cached to: {TEXT_CACHE_PATH}")

    # Clean up progress file
    if os.path.exists(PROGRESS_PATH):
        os.remove(PROGRESS_PATH)

    return pages


def _save_progress(completed_pages):
    """Save current progress to disk."""
    progress_data = [{"page": p, "text": t} for p, t in sorted(completed_pages.items())]
    with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
        json.dump(progress_data, f, ensure_ascii=False)


def chunk_text(pages, chunk_size=512, overlap=64):
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

            if len(chunk_text_content.strip()) > 30:
                chunks.append({
                    "text": chunk_text_content.strip(),
                    "page": page_num,
                    "chunk_id": chunk_id,
                })
                chunk_id += 1

            start += chunk_size - overlap

    print(f"[INFO] Created {len(chunks)} text chunks.")
    return chunks


if __name__ == "__main__":
    pages = extract_text_from_pdf()
    chunks = chunk_text(pages)
    if chunks:
        print(f"\n[SAMPLE] First chunk (page {chunks[0]['page']}):")
        print(chunks[0]['text'][:300])
