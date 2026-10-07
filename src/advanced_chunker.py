"""
Module 2: Advanced Boundary-Aware Chunker (advanced_chunker.py)
--------------------------------------------------------------
Purpose:
    Split text into chunks that respect semantic boundaries (paragraphs, sentences, words)
    rather than blindly slicing at arbitrary character counts.

Why this is needed:
    Naive character slicing cuts words in half (e.g. "e Curfew:" instead of "4.1 In-Time Curfew:").
    Boundary-aware chunking keeps sentences and paragraphs intact, preserving clean semantics.
"""

import re
from typing import List, Dict, Any


def split_into_sentences(text: str) -> List[str]:
    """
    Splits text into sentences using punctuation markers (.!? and newlines).
    """
    # Split by double newline (paragraphs) or sentence terminators (.!?) followed by space/newline
    raw_sentences = re.split(r'(?<=[.!?])\s+|\n\n+', text)
    return [s.strip() for s in raw_sentences if s.strip()]


def recursive_boundary_chunker(
    text: str,
    target_chunk_size: int = 350,
    chunk_overlap: int = 50
) -> List[Dict[str, Any]]:
    """
    Creates chunks by packing complete sentences up to `target_chunk_size`.
    Maintains a sliding sentence overlap between consecutive chunks.

    Args:
        text (str): Input raw text.
        target_chunk_size (int): Target maximum characters per chunk.
        chunk_overlap (int): Desired character overlap across chunks.

    Returns:
        List[Dict[str, Any]]: List of clean, boundary-respected chunks.
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return []

    chunks = []
    current_chunk_sentences = []
    current_length = 0
    chunk_id = 0

    idx = 0
    while idx < len(sentences):
        sentence = sentences[idx]
        sentence_len = len(sentence)

        # If single sentence exceeds target size, include it as its own chunk
        if sentence_len >= target_chunk_size:
            if current_chunk_sentences:
                chunk_str = " ".join(current_chunk_sentences)
                chunks.append({"chunk_id": chunk_id, "text": chunk_str})
                chunk_id += 1
                current_chunk_sentences = []
                current_length = 0

            chunks.append({"chunk_id": chunk_id, "text": sentence})
            chunk_id += 1
            idx += 1
            continue

        # If adding this sentence exceeds target chunk size, commit current chunk
        if current_length + sentence_len + 1 > target_chunk_size and current_chunk_sentences:
            chunk_str = " ".join(current_chunk_sentences)
            chunks.append({"chunk_id": chunk_id, "text": chunk_str})
            chunk_id += 1

            # Build overlap from the tail of current sentences
            overlap_sentences = []
            overlap_len = 0
            for s in reversed(current_chunk_sentences):
                if overlap_len + len(s) + 1 <= chunk_overlap:
                    overlap_sentences.insert(0, s)
                    overlap_len += len(s) + 1
                else:
                    break

            current_chunk_sentences = overlap_sentences
            current_length = overlap_len

        current_chunk_sentences.append(sentence)
        current_length += sentence_len + 1
        idx += 1

    # Add any remaining sentences as the final chunk
    if current_chunk_sentences:
        chunk_str = " ".join(current_chunk_sentences)
        chunks.append({"chunk_id": chunk_id, "text": chunk_str})

    return chunks


if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from loader import load_document

    sample_file = os.path.join("data", "college_policy.txt")
    raw_text = load_document(sample_file)

    print("--- [Module 2: Boundary-Aware Chunker Test] ---")
    clean_chunks = recursive_boundary_chunker(raw_text, target_chunk_size=350, chunk_overlap=50)
    print(f"Total Clean Chunks: {len(clean_chunks)}\n")

    for c in clean_chunks[:3]:
        print(f"[Chunk {c['chunk_id']}] (Length: {len(c['text'])} chars):")
        print(f"\"{c['text']}\"\n")

