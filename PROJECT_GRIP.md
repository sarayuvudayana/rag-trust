# PROJECT GRIP DOCUMENT: RAG Reliability Monitor

This document tracks **every component, decision, input, output, and technical reasoning** across our 7 modules.

---

## 🗺️ System Architecture Overview

```
                      [ Raw Document (PDF/Text) ]
                                  │
                                  ▼ (1. Ingestion)
                        [ Extracted Raw Text ]
                                  │
                                  ▼ (2. Chunking)
                        [ List of Text Chunks ]
                                  │
                                  ▼ (3. Embedding Model)
                   [ Dense Numerical Vectors (Float Arrays) ]
                                  │
                                  ▼ (4. Vector Database)
                      [ Indexed Vector Store (Chroma) ]
                                  │
    [ User Query ] ──► [ Query Embedding ] ──► [ Similarity Search (Cosine) ]
                                                       │
                                                       ▼
                                            [ Top-k Relevant Chunks ]
```

---

## MODULE 1 — Core Document Ingestion & Retrieval Pipeline (V0.1)

### 1. High-Level Objective
Given any document (e.g., college policy, technical manual) and a user's question, retrieve the exact text sections that contain the answer with similarity scores, **without relying on an LLM yet**.

### 2. File & Component Breakdown

| Step | File | Purpose | Key Functions / Classes | What happens inside |
| :--- | :--- | :--- | :--- | :--- |
| **1.1** | [`src/loader.py`](file:///g:/Rag@project/src/loader.py) | Document Loader | `load_document(file_path)` | Checks file type (`.txt` / `.pdf`), extracts raw string text using UTF-8 or `pypdf.PdfReader`. |
| **1.2** | [`src/chunker.py`](file:///g:/Rag@project/src/chunker.py) | Text Chunker | `split_text_into_chunks(text, chunk_size, chunk_overlap)` | Uses a sliding window algorithm to split long text into overlapping chunks, tracking start and end indices. |
| **1.3** | [`src/embedder.py`](file:///g:/Rag@project/src/embedder.py) | Semantic Embedder | `TextEmbedder` (`embed_texts`, `embed_query`) | Uses `sentence-transformers` (`all-MiniLM-L6-v2`) to convert text into normalized 384-dimensional dense vectors. |
| **1.4** | [`src/vector_store.py`](file:///g:/Rag@project/src/vector_store.py) | Vector Database Manager | `VectorStoreManager` (`add_chunks`, `query`) | Stores vectors + metadata in ChromaDB with cosine distance indexing (`hnsw:space: cosine`). |
| **1.5** | [`src/search_demo.py`](file:///g:/Rag@project/src/search_demo.py) | End-to-End Pipeline | `run_pipeline(doc_path, query, top_k)` | Glues Steps 1–4 together: Loads file $\rightarrow$ Chunks $\rightarrow$ Embeds $\rightarrow$ Indexes in Chroma $\rightarrow$ Searches by query $\rightarrow$ Prints Top-$k$ results. |

---

## 🔬 Mathematical & Conceptual Deep Dive

### 1. What is an Embedding Vector?
An embedding converts words into coordinates in a 384-dimensional geometric space:
$$\vec{v} = [x_1, x_2, x_3, \dots, x_{384}]$$
- Semantically identical sentences have vectors that point in almost the exact same direction.
- Unrelated sentences point in orthogonal (perpendicular) or different directions.

### 2. Cosine Similarity Formula
When vectors are normalized ($\|\vec{A}\| = 1, \|\vec{B}\| = 1$), Cosine Similarity is simply the dot product:
$$\text{Cosine Similarity}(\vec{A}, \vec{B}) = \cos(\theta) = \frac{\vec{A} \cdot \vec{B}}{\|\vec{A}\| \|\vec{B}\|} = \sum_{i=1}^{n} A_i B_i$$
- **Score = 1.0**: Identical semantic meaning.
- **Score = 0.0**: Unrelated / orthogonal.
- **Score < 0.0**: Opposite meaning.

### 3. Cosine Distance vs Cosine Similarity in ChromaDB:
$$\text{Cosine Distance} = 1 - \text{Cosine Similarity}$$
$$\text{Cosine Similarity} = 1 - \text{Cosine Distance}$$
- When distance is **0.2586**, similarity is **0.7414** (74.14% match).

---

## 🧪 Module 1 Verification Results (V0.1 Milestone Achieved)

### Test Query 1: *"What is the minimum attendance requirement to write exams?"*
- **Top Match (Chunk 0)**:
  > *"All undergraduate and postgraduate students must maintain a minimum attendance of 75% in each registered course to be eligible to appear for the End-Semester Final Examinations."*
- **Similarity Score**: `0.6443` (Cosine Distance: `0.3557`)
- **Status**: ✅ Correct factual chunk retrieved.

### Test Query 2: *"What is the curfew time for hostel students on weekends?"*
- **Top Match (Chunk 9)**:
  > *"All resident students must return to their respective hostel premises before 9:30 PM on weekdays and 10:30 PM on weekends."*
- **Similarity Score**: `0.7414` (Cosine Distance: `0.2586`)
- **Status**: ✅ Correct factual chunk retrieved without matching unrelated sections (attendance, grading, library).

---

## ⚠️ Key Insights & Observations from Module 1
1. **Word-Splitting Artifacts**: Naive character slicing cut words in half.
2. **Need for Boundary Awareness**: Chunks need to respect sentence and paragraph boundaries.

---

## MODULE 2 — Retrieval Engine Optimization & Tuning (V0.2 Milestone)

### 1. High-Level Objective
Analyze and scientifically optimize the retrieval mechanism by solving:
1. **Sentence Boundary Fragmentation** (preserving complete sentences/paragraphs).
2. **Chunk Dilution vs Context Loss** (determining optimal chunk sizes).
3. **Top-$k$ Retrieval Behavior** (evaluating how many chunks to pass downstream).

### 2. File & Component Breakdown

| Step | File | Purpose | Key Concept |
| :--- | :--- | :--- | :--- |
| **2.1** | [`src/advanced_chunker.py`](file:///g:/Rag@project/src/advanced_chunker.py) | Sentence & Paragraph Boundary Chunker | Uses regex hierarchy (`\n\n` $\rightarrow$ `[.!?]` $\rightarrow$ words) to pack complete sentences up to target size without clipping words. |
| **2.2** | [`src/retrieval_experiment.py`](file:///g:/Rag@project/src/retrieval_experiment.py) | Retrieval Tuning Benchmark | Tests Small ($150$), Medium ($350$), and Large ($700$) configurations across standard queries to measure Cosine Similarity impact. |

---

### 3. Quantitative Experimental Results (Benchmark Log)

| Query | Small Chunks (150 chars) | Medium Chunks (350 chars) | Large Chunks (700 chars) | Key Phenomenon Observed |
| :--- | :---: | :---: | :---: | :--- |
| **1. Minimum Attendance** | **0.7175** | 0.6565 | 0.6314 | Focused chunk gives highest semantic density. |
| **2. Overdue Book Fines** | **0.8633** | 0.6670 | 0.6809 | **Vector Dilution**: Mixing book lending + fines diluted similarity by ~20%. |
| **3. Hostel Curfew** | **0.8287** | **0.8287** | 0.7174 | Large chunk blended unrelated *Grading Rules* into the vector. |

---

### 4. Critical Discoveries for RAG Engineering

#### 💡 The Vector Dilution Effect
When a chunk contains **two different topics** (e.g., *Letter Grades* + *Hostel Rules*), the embedding vector is pulled toward the geometric center of both topics. This reduces the cosine similarity score for queries targeting only one of those topics.

---

## MODULE 3 — Grounded LLM Generation Pipeline (V0.3 Milestone)

### 1. High-Level Objective
Connect the retrieval engine from Modules 1 & 2 to an LLM (e.g. Llama 3 / Ollama / Cloud API) to synthesize fluent, strictly grounded answers based **only** on retrieved context chunks.

### 2. File & Component Breakdown

| Step | File | Purpose | Key Concept |
| :--- | :--- | :--- | :--- |
| **3.1** | [`src/prompt.py`](file:///g:/Rag@project/src/prompt.py) | Grounded Prompt Composer | Assembles retrieved chunks with system instructions and user queries, applying negative constraints ("answer only using context"). |
| **3.2** | [`src/llm.py`](file:///g:/Rag@project/src/llm.py) | Universal LLM Client | Decoupled LLM interface supporting **Ollama** (`llama3.2`), **Groq / OpenAI API**, and **Deterministic Extractive Fallback**. |
| **3.3** | [`src/rag_pipeline.py`](file:///g:/Rag@project/src/rag_pipeline.py) | End-to-End RAG Engine | Orchestrates full pipeline: Question $\rightarrow$ ChromaDB Search $\rightarrow$ Prompt Assembly $\rightarrow$ LLM Generation $\rightarrow$ Final Grounded Answer. |

---

## 🚨 **IMPORTANT DETAILS OF LLM MODEL & EXECUTION MECHANICS**

### 1. What is "Offline Extractive" Mode in `llm.py`?
- **The Problem**: Full Large Language Models (like Llama 3 8B or Llama 3.2) require 2GB to 8GB of memory and need either a local server (Ollama) or a paid/free Cloud API key (Groq/OpenAI). Without these, a standard script would crash with `ConnectionRefusedError: No LLM found`.
- **The Solution**: We built a built-in safety net in [`src/llm.py`](file:///g:/Rag@project/src/llm.py). If neither Ollama nor an API key is detected, it automatically activates the **Offline Extractor**, which parses the question keywords and synthesizes the exact factual sentences directly from the retrieved context chunks.
- **Why this is critical**: The entire RAG pipeline can be run, tested, and verified on any machine with zero external dependencies, and automatically upgrades to real Llama 3 the instant Ollama or an API key is supplied.

### 2. How Did the Embedding Model Download Without Manual Installation?
- When the code runs `SentenceTransformer("all-MiniLM-L6-v2")` in [`src/embedder.py`](file:///g:/Rag@project/src/embedder.py), the `sentence-transformers` library checks your local disk cache at `C:\Users\HP\.cache\huggingface\hub`.
- If the model is not found, it automatically connects to the **Hugging Face Model Hub** via HTTPS, downloads the compact 80 MB model weights, and saves them permanently in your local cache.
- On all subsequent executions, the model loads from your local disk with zero network latency.

### 3. What Does the Warning *"You are sending unauthenticated requests to the HF Hub"* Mean?
- **Is it an error?** No. It is purely an informational message.
- **Why does it appear?** `all-MiniLM-L6-v2` is a 100% public, open-source model that anyone can download freely without logging in. Hugging Face simply displays this message to let developers know that setting an optional `HF_TOKEN` environment variable gives higher rate limits for high-volume enterprise downloads. For standard development, anonymous downloads work without restrictions.

---


### 1. What Exactly is ChromaDB Storing?

ChromaDB is a specialized database that stores **4 distinct components** for every chunk:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 A SINGLE CHROMADB RECORD                               │
├────────────┬───────────────────────────────────────────────────────────────────────────┤
│ 1. ID      │ "college_policy.txt_chunk_0"                                              │
│            │ Unique key used for lookups, updates, and deduplication.                  │
├────────────┼───────────────────────────────────────────────────────────────────────────┤
│ 2. VECTOR  │ [0.0241, -0.1932, 0.8120, ..., 0.0514] (384 float numbers)                 │
│            │ Dense coordinates in semantic vector space for geometric nearest-neighbor  │
│            │ search (indexed via HNSW graph).                                          │
├────────────┼───────────────────────────────────────────────────────────────────────────┤
│ 3. DOCUMENT│ "SECTION 1: ATTENDANCE POLICY\nAll students must maintain 75%..."         │
│            │ Raw human-readable text string. The LLM cannot read 384 numbers; once the  │
│            │ vector match is found, this text is fetched and injected into the prompt. │
├────────────┼───────────────────────────────────────────────────────────────────────────┤
│ 4. METADATA│ {"source": "college_policy.txt", "chunk_id": 0, "start_idx": 0}           │
│            │ Key-value dictionary for metadata filtering (e.g., filter by department).  │
└────────────┴───────────────────────────────────────────────────────────────────────────┘
```

#### ❓ Is ChromaDB storing our questions?
- **No.** The knowledge base chunks are stored **permanently** in `chroma_db/` on disk.
- The **User Question** is ephemeral (temporary). At query time, the question is converted into a vector in RAM, passed to ChromaDB to calculate distances against all stored chunk vectors, and discarded after returning results.

#### ⚡ How ChromaDB Searches in Milliseconds (HNSW Index):
Instead of scanning every vector linearly ($O(N)$ brute force), ChromaDB organizes vectors into a **Hierarchical Navigable Small World (HNSW)** multi-layered graph. It navigates across layers like an express highway system, finding the nearest neighbors in logarithmic time ($O(\log N)$).

---

### 2. Deep Dive: What is `all-MiniLM-L6-v2`?

`all-MiniLM-L6-v2` is a distilled Transformer encoder fine-tuned by Microsoft / Hugging Face for sentence embeddings.

#### Decoding the Model Name:
- **`all-`**: Trained on an all-purpose dataset of over **1 billion sentence pairs** (Wikipedia, Reddit, StackExchange, news, scientific papers).
- **`MiniLM`**: **Miniature Language Model**. Distilled and compressed from BERT/RoBERTa to be lightweight ($\sim 80\text{ MB}$) and run in milliseconds on standard CPUs.
- **`L6`**: Contains **6 Transformer Encoder Layers** (BERT-base has 12, BERT-large has 24). This provides $5\times$ faster inference speed while maintaining $99\%$ semantic accuracy.
- **`v2`**: Second generation version tuned specifically for **Cosine Similarity** mapping.

#### How Text Becomes a 384-Dimensional Vector:
```
 "Students must maintain 75% attendance."
                    │
                    ▼  (1. Tokenizer)
 [Token IDs: 3124, 2442, 5809, 1021, 2389, ...]
                    │
                    ▼  (2. 6 Transformer Attention Layers)
 Tokens interact through self-attention to build context-aware vectors for every word.
                    │
                    ▼  (3. Mean Pooling)
 Averages all token output vectors into a single fixed vector of 384 dimensions.
                    │
                    ▼  (4. L2 Normalization: ||v|| = 1.0)
 [ 0.0241, -0.1932, 0.8120, ..., 0.0514 ]  <-- 384 floats (Normalized Unit Vector)
```

#### Why 384 Dimensions?
Every dimension represents a latent semantic feature in vector space (e.g., dimension 12 might track *administrative policies*, dimension 88 might track *numbers and percentages*). Semantically similar sentences point in nearly identical directions in this 384-dimensional hypersphere.

---

## Technical Log & Learnings

### Why do we need chunking instead of embedding the whole document?
1. **Model Context Windows**: Embedding models have maximum token limits (e.g., 256 or 512 tokens).
2. **Semantic Specificity**: A single long document covers 50 different topics. Embedding the entire document averages out the meaning. Small chunks preserve specific facts (e.g., "75% attendance").
3. **Retrieval Precision**: We only want to feed relevant excerpts to the LLM, not 50 pages of irrelevant text.

### Why do we need overlap between chunks?
If a sentence or idea is cut right in the middle across a chunk boundary, critical context is lost. Overlap (e.g., 50 characters or words) ensures sentences spanning a boundary remain understandable in at least one chunk.


