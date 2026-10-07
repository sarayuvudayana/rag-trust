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

#### 💡 The Small Chunk vs Context Trade-off
- **Smaller chunks ($150$–$300$ chars)** $\rightarrow$ Higher similarity scores for narrow factual questions.
- **Danger with too small**: A complex query requiring multi-sentence reasoning may find that premise $A$ is in Chunk $1$ and conclusion $B$ is in Chunk $2$.
- **Best Practice for Module 3**: Use **boundary-aware chunks of $250$–$350$ chars with $50$ char overlap**, and retrieve **Top-$k = 2$ or $3$** so the LLM gets full context without vector dilution.



## Technical Log & Learnings

### Why do we need chunking instead of embedding the whole document?
1. **Model Context Windows**: Embedding models have maximum token limits (e.g., 256 or 512 tokens).
2. **Semantic Specificity**: A single long document covers 50 different topics. Embedding the entire document averages out the meaning. Small chunks preserve specific facts (e.g., "75% attendance").
3. **Retrieval Precision**: We only want to feed relevant excerpts to the LLM, not 50 pages of irrelevant text.

### Why do we need overlap between chunks?
If a sentence or idea is cut right in the middle across a chunk boundary, critical context is lost. Overlap (e.g., 50 characters or words) ensures sentences spanning a boundary remain understandable in at least one chunk.

