"""
Module 1 - Step 2: Document Chunker (chunker.py)
-----------------------------------------------
Purpose:
    Split a continuous string into smaller, overlapping chunks of text.

Why this is needed:
    1. Embedding models have context length limits (e.g. 256/512 tokens).
    2. Small, focused chunks preserve specific facts and improve retrieval precision.
    3. Overlap prevents critical context from being cut in half across boundaries.
"""

from typing import List, Dict, Any


def split_text_into_chunks(
    text: str,
    chunk_size: int = 350,
    chunk_overlap: int = 50
) -> List[Dict[str, Any]]:
    """
    Splits a raw text string into overlapping chunks using a sliding window.

    Args:
        text (str): The raw text to split.
        chunk_size (int): Target character count for each chunk. Default is 350.
        chunk_overlap (int): Number of overlapping characters between consecutive chunks. Default is 50.

    Returns:
        List[Dict[str, Any]]: A list of chunk objects, each containing:
            - 'chunk_id': Integer index (0, 1, 2, ...)
            - 'text': The chunk string
            - 'start_idx': Character start position in original text
            - 'end_idx': Character end position in original text
    """
    if chunk_size <= chunk_overlap:
        raise ValueError("chunk_size must be strictly greater than chunk_overlap.")

    if not text.strip():
        return []

    chunks = []
    step = chunk_size - chunk_overlap
    start = 0
    text_length = len(text)
    chunk_id = 0

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk_text = text[start:end].strip()

        if chunk_text:  # Avoid empty chunks
            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "start_idx": start,
                "end_idx": end
            })
            chunk_id += 1

        if end >= text_length:
            break

        start += step

    return chunks


if __name__ == "__main__":
    # Quick standalone test for Step 2
    from loader import load_document
    import os

    sample_file = os.path.join("data", "college_policy.txt")
    raw_text = load_document(sample_file)
    chunks = split_text_into_chunks(raw_text, chunk_size=300, chunk_overlap=50)

    print("--- [Step 2: Chunker Test] ---")
    print(f"Total Chunks Created: {len(chunks)}")
    for i, c in enumerate(chunks[:3]):
        print(f"\n[Chunk {c['chunk_id']}] (Chars {c['start_idx']} -> {c['end_idx']}):")
        print(f"\"{c['text']}\"")

