"""
Comparison Tool: Version 1 (Naive Chunker) vs Version 2 (Boundary-Aware Chunker)
--------------------------------------------------------------------------------
Run this script to see the side-by-side difference between:
- Version 1: Blind character slicing (cuts words/sentences)
- Version 2: Smart boundary-aware slicing (preserves complete sentences)
"""

import os
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from loader import load_document
from chunker import split_text_into_chunks
from advanced_chunker import recursive_boundary_chunker


def compare_v1_vs_v2(doc_path: str):
    print("=" * 80)
    print("⚔️  SIDE-BY-SIDE COMPARISON: VERSION 1 vs VERSION 2 CHUNKING")
    print("=" * 80)

    raw_text = load_document(doc_path)

    # V1: Naive Chunker
    v1_chunks = split_text_into_chunks(raw_text, chunk_size=300, chunk_overlap=50)

    # V2: Boundary-Aware Chunker
    v2_chunks = recursive_boundary_chunker(raw_text, target_chunk_size=350, chunk_overlap=50)

    print(f"\n[Stats]")
    print(f"• Document Length: {len(raw_text)} characters")
    print(f"• V1 Total Chunks: {len(v1_chunks)} (Naive character window)")
    print(f"• V2 Total Chunks: {len(v2_chunks)} (Sentence & boundary-aware)")

    print("\n" + "=" * 80)
    print("🔍 INSPECTING SPECIFIC CHUNKS (LOOK AT THE START AND END OF TEXT)")
    print("=" * 80)

    # Compare Chunk with Hostel Rules
    print("\n[V1 - Naive Chunk 9] (Notice the broken word at the start!):")
    print("-" * 50)
    print(f"\"{v1_chunks[9]['text']}\"")
    print("-" * 50)
    print("❌ Flaw: Notice it starts with 'e Curfew:' because '4.1 In-Time Curfew:' was chopped in half!")

    print("\n[V2 - Boundary-Aware Chunk 8] (Clean sentence boundaries):")
    print("-" * 50)
    print(f"\"{v2_chunks[8]['text']}\"")
    print("-" * 50)
    print("✅ Improvement: Starts cleanly with 'SECTION 4: HOSTEL AND CURFEW RULES' with zero broken words.")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    policy_doc = os.path.join("data", "college_policy.txt")
    compare_v1_vs_v2(policy_doc)

