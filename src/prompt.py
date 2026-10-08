"""
Module 3 - Step 1: Grounded Prompt Composer (prompt.py)
-------------------------------------------------------
Purpose:
    Combine retrieved document chunks and the user's question into a strict,
    grounded prompt that prevents the LLM from making up outside facts.

Why this is needed:
    LLMs hallucinate when asked open questions. By giving the LLM clear boundaries
    and strict negative constraints ("answer only using context"), we minimize false claims.
"""

from typing import List, Dict, Any


def format_context(chunks: List[Dict[str, Any]]) -> str:
    """
    Formats a list of retrieved chunk dictionaries into a clean text block.

    Args:
        chunks (List[Dict[str, Any]]): Retrieved chunks with metadata and text.

    Returns:
        str: Formatted context string with chunk boundaries.
    """
    if not chunks:
        return "No relevant context found."

    context_blocks = []
    for idx, chunk in enumerate(chunks, start=1):
        source = chunk.get("source", "document")
        chunk_id = chunk.get("chunk_id", idx - 1)
        sim = chunk.get("similarity_score", 0.0)
        text = chunk.get("text", "").strip()

        block = f"[Context #{idx} | Source: {source} (Chunk {chunk_id}) | Similarity: {sim:.2f}]\n{text}"
        context_blocks.append(block)

    return "\n\n".join(context_blocks)


def build_rag_prompt(question: str, chunks: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Builds the system and user prompt for RAG generation.

    Args:
        question (str): The user's query.
        chunks (List[Dict[str, Any]]): Retrieved context chunks.

    Returns:
        Dict[str, str]: Dictionary containing 'system_prompt' and 'user_prompt'.
    """
    system_prompt = (
        "You are a precise, factual AI assistant.\n"
        "Your task is to answer the user's question STRICTLY and ONLY using the provided CONTEXT.\n"
        "Rules:\n"
        "1. Do not use outside knowledge or make assumptions.\n"
        "2. If the context does not contain enough information to answer, state: "
        "\"The provided document does not contain this information.\"\n"
        "3. Keep your response concise, factual, and directly tied to the facts in the context."
    )

    context_str = format_context(chunks)

    user_prompt = (
        f"CONTEXT:\n"
        f"============================================================\n"
        f"{context_str}\n"
        f"============================================================\n\n"
        f"QUESTION: {question}\n\n"
        f"ANSWER:"
    )

    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "full_prompt": f"{system_prompt}\n\n{user_prompt}"
    }


if __name__ == "__main__":
    # Test prompt formatting
    sample_chunks = [
        {
            "chunk_id": 0,
            "source": "college_policy.txt",
            "similarity_score": 0.8287,
            "text": "All resident students must return to their respective hostel premises before 9:30 PM on weekdays and 10:30 PM on weekends."
        }
    ]
    prompt_bundle = build_rag_prompt("When is curfew on Friday?", sample_chunks)
    print("--- [Module 3: Prompt Composer Test] ---")
    print(prompt_bundle["full_prompt"])

