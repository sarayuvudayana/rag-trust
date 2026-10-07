"""
Module 1 - Step 1: Document Loader (loader.py)
---------------------------------------------
Purpose:
    Extract raw text from a document file (.txt or .pdf) and return it as a single string.

Why this is needed:
    Raw files cannot be directly searched mathematically. We need clean,
    unformatted text that can be chunked and embedded.
"""

import os
from pypdf import PdfReader


def load_document(file_path: str) -> str:
    """
    Reads a text or PDF file and returns the full text as a string.

    Args:
        file_path (str): The path to the file to be loaded.

    Returns:
        str: Cleaned text extracted from the document.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Document not found at: {file_path}")

    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return text.strip()

    elif ext == ".pdf":
        reader = PdfReader(file_path)
        extracted_pages = []
        for page_idx, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                extracted_pages.append(page_text.strip())
        return "\n\n".join(extracted_pages)

    else:
        raise ValueError(f"Unsupported file format: '{ext}'. Supported formats: .txt, .pdf")


if __name__ == "__main__":
    # Quick standalone test for Step 1
    sample_file = os.path.join("data", "college_policy.txt")
    content = load_document(sample_file)
    print("--- [Step 1: Document Loader Test] ---")
    print(f"Loaded {len(content)} characters from {sample_file}")
    print("\nFirst 200 characters preview:")
    print(content[:200] + "...")

