"""
Module 3: Complete End-to-End RAG Generation Pipeline (rag_pipeline.py)
----------------------------------------------------------------------
Purpose:
    Execute the full RAG workflow:
    User Question -> Retrieval (ChromaDB) -> Prompt Composition -> LLM Generation -> Final Grounded Answer
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
from advanced_chunker import recursive_boundary_chunker
from embedder import TextEmbedder
from vector_store import VectorStoreManager
from prompt import build_rag_prompt
from llm import LLMClient


class RAGSystem:
    def __init__(
        self,
        collection_name: str = "rag_knowledge_base",
        persist_dir: str = "chroma_db",
        llm_provider: str = "auto",
        llm_model: str = "llama3.2"
    ):
        """
        Initializes all RAG sub-components: Embedder, Vector Store, and LLM Client.
        """
        print("[RAGSystem] Initializing Pipeline...")
        self.embedder = TextEmbedder(model_name="all-MiniLM-L6-v2")
        self.vector_store = VectorStoreManager(collection_name=collection_name, persist_dir=persist_dir)
        self.llm = LLMClient(provider=llm_provider, model_name=llm_model)

    def ingest_document(self, doc_path: str, chunk_size: int = 350, chunk_overlap: int = 50):
        """
        Loads, chunks, embeds, and indexes a document in ChromaDB.
        """
        print(f"\n[Ingestion] Loading: {doc_path}")
        raw_text = load_document(doc_path)
        chunks = recursive_boundary_chunker(raw_text, target_chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        chunk_texts = [c["text"] for c in chunks]
        embeddings = self.embedder.embed_texts(chunk_texts)
        
        doc_name = os.path.basename(doc_path)
        self.vector_store.reset_collection()
        self.vector_store.add_chunks(chunks, embeddings, source_name=doc_name)
        print(f"[Ingestion] Successfully indexed {len(chunks)} chunks from '{doc_name}'.")

    def answer_query(self, query: str, top_k: int = 2) -> dict:
        """
        Performs vector retrieval, constructs grounded prompt, and calls LLM to generate answer.
        """
        print("\n" + "=" * 65)
        print(f"❓ USER QUESTION: \"{query}\"")
        print("=" * 65)

        # 1. Retrieve relevant chunks
        query_vector = self.embedder.embed_query(query)
        retrieved_chunks = self.vector_store.query(query_vector, top_k=top_k)

        print(f"\n[Retriever] Found Top-{len(retrieved_chunks)} relevant chunks:")
        for idx, chk in enumerate(retrieved_chunks, start=1):
            print(f"  • Match #{idx} | Similarity: {chk['similarity_score']:.4f} | Chunk ID: {chk['chunk_id']}")

        # 2. Build Prompt
        prompt_bundle = build_rag_prompt(query, retrieved_chunks)

        # 3. Generate Answer
        print(f"\n[LLM] Generating grounded response...")
        answer = self.llm.generate(
            system_prompt=prompt_bundle["system_prompt"],
            user_prompt=prompt_bundle["user_prompt"],
            temperature=0.0
        )

        print("\n" + "=" * 65)
        print("🤖 RAG GENERATED ANSWER:")
        print("=" * 65)
        print(answer)
        print("=" * 65)

        return {
            "query": query,
            "retrieved_chunks": retrieved_chunks,
            "prompt": prompt_bundle["full_prompt"],
            "answer": answer
        }


if __name__ == "__main__":
    doc_file = os.path.join("data", "college_policy.txt")
    test_question = "What is the hostel fees?"

    if len(sys.argv) > 1:
        test_question = " ".join(sys.argv[1:])

    rag = RAGSystem()
    rag.ingest_document(doc_file)
    rag.answer_query(test_question, top_k=2)

