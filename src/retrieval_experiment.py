"""
Module 2: Retrieval Experiment & Tuning Benchmarks (retrieval_experiment.py)
---------------------------------------------------------------------------
Purpose:
    Experimentally evaluate different chunk sizes (Small: 150, Medium: 350, Large: 700)
    and chunking strategies to measure their direct effect on similarity scores and retrieval precision.
"""

import os
import sys

# Ensure UTF-8 output encoding on Windows consoles
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
from embedder import TextEmbedder
from vector_store import VectorStoreManager


def run_experiment(doc_path: str, test_queries: list):
    print("=" * 75)
    print("🔬 MODULE 2: RETRIEVAL ENGINE EXPERIMENT & TUNING BENCHMARK")
    print("=" * 75)

    raw_text = load_document(doc_path)
    embedder = TextEmbedder()

    configurations = [
        {"name": "Small (150 chars)", "size": 150, "overlap": 30, "method": "boundary"},
        {"name": "Medium (350 chars - Recommended)", "size": 350, "overlap": 50, "method": "boundary"},
        {"name": "Large (700 chars)", "size": 700, "overlap": 100, "method": "boundary"},
    ]

    for config in configurations:
        print("\n" + "-" * 75)
        print(f"📦 TESTING CONFIGURATION: {config['name']}")
        print(f"   Target Size: {config['size']} chars | Overlap: {config['overlap']} chars")
        print("-" * 75)

        # 1. Chunk
        chunks = recursive_boundary_chunker(
            raw_text,
            target_chunk_size=config["size"],
            chunk_overlap=config["overlap"]
        )
        print(f"   [Info] Total chunks produced: {len(chunks)}")

        # 2. Embed
        chunk_texts = [c["text"] for c in chunks]
        embeddings = embedder.embed_texts(chunk_texts)

        # 3. Store in dedicated test collection
        safe_name = config["name"].split()[0].lower()
        col_name = f"exp_col_{safe_name}"
        store = VectorStoreManager(collection_name=col_name, persist_dir="chroma_experiments_db")
        store.reset_collection()
        store.add_chunks(chunks, embeddings, source_name="policy")

        # 4. Run test queries
        for q in test_queries:
            q_vec = embedder.embed_query(q)
            results = store.query(q_vec, top_k=1)
            top = results[0] if results else None
            if top:
                print(f"\n   ❓ Query: \"{q}\"")
                print(f"   ⭐ Top-1 Similarity: {top['similarity_score']:.4f} (Distance: {top['cosine_distance']:.4f})")
                print(f"   📄 Excerpt: \"{top['text'][:120]}...\"")

    print("\n" + "=" * 75)
    print("✅ Experiment completed.")
    print("=" * 75)


if __name__ == "__main__":
    policy_doc = os.path.join("data", "college_policy.txt")
    queries = [
        "What is the minimum attendance requirement to write exams?",
        "What is the overdue fine for returning books late?",
        "What is the curfew time for hostel students on weekends?"
    ]
    run_experiment(policy_doc, queries)

