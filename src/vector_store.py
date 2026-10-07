"""
Module 1 - Step 4 & 5: Vector Store & Similarity Retriever (vector_store.py)
---------------------------------------------------------------------------
Purpose:
    Store text chunks and their embeddings in ChromaDB, and perform
    cosine similarity search to retrieve the most relevant chunks for a query.

Why this is needed:
    A vector database indexes vectors geometrically so we can retrieve
    the top-k nearest chunks in milliseconds, even across thousands of documents.
"""

import os
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings


class VectorStoreManager:
    def __init__(self, collection_name: str = "rag_knowledge_base", persist_dir: str = "chroma_db"):
        """
        Initializes the ChromaDB persistent client and collection.

        Args:
            collection_name (str): Name of the Chroma collection.
            persist_dir (str): Directory where Chroma stores vector indexes on disk.
        """
        self.persist_dir = persist_dir
        self.collection_name = collection_name

        # Create persistent Chroma client on disk
        self.client = chromadb.PersistentClient(path=self.persist_dir)

        # Use cosine similarity space for vector distance calculations
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def reset_collection(self):
        """Deletes and recreates the collection to start fresh if needed."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
        source_name: str = "doc"
    ):
        """
        Inserts chunks and their vector embeddings into the vector store.

        Args:
            chunks (List[Dict[str, Any]]): List of chunk objects from chunker.py
            embeddings (List[List[float]]): List of vector embeddings from embedder.py
            source_name (str): The filename/source identifier.
        """
        if not chunks or not embeddings:
            return

        ids = [f"{source_name}_chunk_{c['chunk_id']}" for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [
            {
                "source": source_name,
                "chunk_id": c.get("chunk_id", idx),
                "start_idx": c.get("start_idx", 0),
                "end_idx": c.get("end_idx", len(c.get("text", "")))
            }
            for idx, c in enumerate(chunks)
        ]


        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        print(f"[VectorStore] Added {len(chunks)} chunks from '{source_name}' to ChromaDB.")

    def query(self, query_embedding: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Searches ChromaDB for the closest vectors to the given query embedding.

        Args:
            query_embedding (List[float]): The 384-dimensional query vector.
            top_k (int): Number of most similar chunks to return.

        Returns:
            List[Dict[str, Any]]: Ranked list of matching chunks with similarity scores.
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        ranked_results = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]

            for doc, meta, dist in zip(docs, metadatas, distances):
                # In Chroma with cosine space: similarity = 1 - distance
                similarity_score = 1.0 - dist
                ranked_results.append({
                    "chunk_id": meta.get("chunk_id"),
                    "source": meta.get("source"),
                    "text": doc,
                    "cosine_distance": round(dist, 4),
                    "similarity_score": round(similarity_score, 4)
                })

        return ranked_results


if __name__ == "__main__":
    # Quick standalone test for Vector Store
    v_store = VectorStoreManager(persist_dir="chroma_db_test")
    print(f"--- [Step 4 & 5: Vector Store Initialized] ---")
    print(f"Collection Name: {v_store.collection_name}")
    print(f"Current Count: {v_store.collection.count()} chunks in DB")

