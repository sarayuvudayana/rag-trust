"""
Module 1 - Step 3: Text Embedder (embedder.py)
---------------------------------------------
Purpose:
    Convert raw text strings into dense mathematical vector embeddings (list of floats).

Why this is needed:
    Vector search relies on calculating semantic distance (cosine similarity)
    between query vectors and document vectors. We use 'all-MiniLM-L6-v2',
    a fast and reliable sentence transformer that outputs 384-dimensional vectors.
"""

from typing import List
from sentence_transformers import SentenceTransformer


class TextEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initializes the SentenceTransformer embedding model.

        Args:
            model_name (str): HuggingFace model identifier.
                              Defaults to 'all-MiniLM-L6-v2' (384 dimensions).
        """
        print(f"[Embedder] Loading embedding model: '{model_name}'...")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Encodes a list of text strings into a list of dense vector embeddings.

        Args:
            texts (List[str]): List of sentences or chunks.

        Returns:
            List[List[float]]: List of 384-dimensional floating point vectors.
        """
        if not texts:
            return []
        
        # normalize_embeddings=True ensures unit length vectors (magnitude = 1),
        # making dot product identical to cosine similarity.
        embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """
        Encodes a single query string into a vector embedding.

        Args:
            query (str): The search question.

        Returns:
            List[float]: A single 384-dimensional vector.
        """
        embeddings = self.embed_texts([query])
        return embeddings[0]


if __name__ == "__main__":
    # Quick standalone test for Step 3
    embedder = TextEmbedder()
    test_sentences = [
        "Students must maintain 75% attendance.",
        "What is the required class attendance percentage?",
        "The library is open until 8 PM."
    ]
    vectors = embedder.embed_texts(test_sentences)

    print("--- [Step 3: Embedder Test] ---")
    print(f"Number of sentences encoded: {len(vectors)}")
    print(f"Embedding dimension size: {len(vectors[0])} floats per sentence")
    print("\nSample vector slice (first 5 numbers of Chunk 1):")
    print(vectors[0][:5])

