"""
Module 1: End-to-End Semantic Retrieval Pipeline (search_demo.py)
----------------------------------------------------------------
This script ties together all 5 steps of Module 1:
1. Document Ingestion (loader.py)
2. Text Chunking (chunker.py)
3. Embedding Generation (embedder.py)
4. Vector Indexing (vector_store.py)
5. Query Vector Search (Top-k Retrieval with Similarity Scores)
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

# Ensure src modules are resolvable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from loader import load_document
from chunker import split_text_into_chunks
from advanced_chunker import recursive_boundary_chunker
from embedder import TextEmbedder
from vector_store import VectorStoreManager


def run_pipeline(doc_path: str, query: str, top_k: int = 2, use_v2_chunker: bool = True):
    print("=" * 60)
    version_label = "V0.2 (Boundary-Aware Chunker)" if use_v2_chunker else "V0.1 (Naive Chunker)"
    print(f"RUNNING MODULE 2 RETRIEVAL PIPELINE: {version_label}")
    print("=" * 60)

    # 1. Load Document
    print(f"\n[Step 1] Loading document from: {doc_path}")
    raw_text = load_document(doc_path)
    print(f"[OK] Loaded {len(raw_text)} characters.")

    # 2. Chunk Text
    if use_v2_chunker:
        print(f"\n[Step 2] Chunking with V2 Boundary-Aware Chunker (target_size=350, overlap=50)...")
        chunks = recursive_boundary_chunker(raw_text, target_chunk_size=350, chunk_overlap=50)
    else:
        print(f"\n[Step 2] Chunking with V1 Naive Chunker (chunk_size=300, chunk_overlap=50)...")
        chunks = split_text_into_chunks(raw_text, chunk_size=300, chunk_overlap=50)

    print(f"[OK] Generated {len(chunks)} text chunks.")


    # 3. Embed Chunks
    print(f"\n[Step 3] Initializing embedder and generating chunk embeddings...")
    embedder = TextEmbedder(model_name="all-MiniLM-L6-v2")
    chunk_texts = [c["text"] for c in chunks]
    chunk_vectors = embedder.embed_texts(chunk_texts)
    print(f"[OK] Generated {len(chunk_vectors)} vectors (dimension={len(chunk_vectors[0])}).")

    # 4. Store in Vector Database
    print(f"\n[Step 4] Storing chunks and vectors in ChromaDB...")
    doc_name = os.path.basename(doc_path)
    vector_store = VectorStoreManager(collection_name="module1_policy_store", persist_dir="chroma_db")
    vector_store.reset_collection()  # Fresh index for demo
    vector_store.add_chunks(chunks, chunk_vectors, source_name=doc_name)
    print("[OK] Vector DB indexed and ready.")

    # 5. Query & Retrieve
    print(f"\n[Step 5] Executing Query:")
    print(f"Question: \"{query}\"")
    
    query_vector = embedder.embed_query(query)
    results = vector_store.query(query_vector, top_k=top_k)

    print("\n" + "=" * 60)
    print(f"RETRIEVED TOP-{top_k} RESULTS FOR QUESTION")
    print("=" * 60)

    for rank, res in enumerate(results, start=1):
        print(f"\n--- Result #{rank} [Similarity: {res['similarity_score']:.4f} | Cosine Dist: {res['cosine_distance']:.4f}] ---")
        print(f"Source: {res['source']} (Chunk ID: {res['chunk_id']})")
        print(f"Content:")
        print(res["text"])

    print("\n" + "=" * 60)



if __name__ == "__main__":
    doc_file = os.path.join("data", "college_policy.txt")
    test_query = "What is hostel rules?"
    
    # Allow passing custom question from CLI if desired
    if len(sys.argv) > 1:
        test_query = " ".join(sys.argv[1:])

    run_pipeline(doc_file, test_query, top_k=2)

