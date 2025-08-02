## Summary of Phase 1: Document Processing & Indexing

In this initial phase, our primary goal was to prepare your raw PDF policy documents for effective **semantic search**. We focused on transforming them into a collection of clean, semantically meaningful, and well-indexed text chunks.

---

### ✅ Key Steps Implemented

#### 1. **PDF Text Extraction**
We started by using the **PyMuPDF** library to reliably extract all text content from your PDF files, **page by page**.  
This is the foundational step, converting the visual layout of PDFs into raw text that our system can process.

---

#### 2. **Dynamic Header and Footer Management**

Repetitive headers and footers can introduce noise into search results, so we implemented a **two-step intelligent approach**:

- **Identification:**  
  Dynamically detect common header and footer lines by analyzing text patterns that repeat across a significant number of pages. The **first page is excluded**, as it often contains unique and vital metadata (e.g., policy title, dates).

- **Removal:**  
  Once identified, these repetitive elements are removed from relevant pages, ensuring our embeddings focus only on **unique, meaningful content**.

---

#### 3. **Structure-Aware Document Chunking**

Instead of using fixed-size splitting, we used the **RecursiveCharacterTextSplitter** from `langchain-text-splitters`.  
This tool splits text intelligently using the following hierarchy:

- Paragraphs (`\n\n`)  
- Lines (`\n`)  
- Words (` `)  
- Characters (fallback only)

This preserves logical units (like **complete clauses or sentences**), especially across page breaks—crucial for policy documents.

---

#### 4. **Accurate Tokenization**

We integrated **tiktoken**, OpenAI's tokenizer, to precisely count tokens in each chunk.  
This ensures each chunk is **optimized for the LLM's context window**, preventing truncation and maximizing usable information during inference.

---

#### 5. **Rich Metadata Tagging**

Each chunk is tagged with valuable metadata, including:

- `doc_id` – Original document identifier  
- `page_number` – Source page  
- `clause_id` – Unique identifier for traceability

This metadata enables **precise mapping** from answers back to source clauses and allows **filtered semantic search** in later phases.

---

### ✅ Embedding & Vector Indexing

#### 6. **Embedding Generation**

We used a powerful embedding model: **`BAAI/bge-large-en-v1.5`**, to convert each of the cleaned, token-optimized chunks into high-dimensional **semantic vectors (embeddings)**.  
These embeddings capture the **meaning and intent** of each clause or segment.

---

#### 7. **Vector Indexing**

All generated embeddings—along with their **original text and metadata**—were stored in a **Qdrant** collection.  
This vector database allows us to **efficiently search** semantically similar content based on natural language queries.

---

#### 8. **Verification**

To validate the setup, we ran test queries against the Qdrant vector store.  
The system successfully retrieved the **most relevant text chunks**, confirming the **end-to-end pipeline** from PDF to retrievable vector index is working as intended.

---

### ✅ Outcome

By completing this phase, we now have:

- **Clean**, de-noised text  
- **Semantically coherent** chunks  
- **Token-count aware** segments  
- **Traceable metadata** for each clause  
- **Vector embeddings** stored in a performant vector database  
- **Verified query functionality**

This dataset is now **fully ready** for Phase 2: Semantic Search & LLM-based Reasoning.

---

## Summary of Phase 2: Retrieval Techniques

In the context of a **Retrieval-Augmented Generation (RAG)** pipeline, Phase 2 focuses on the **Retrieval** stage.  
This is where the system processes a user's query and searches through the indexed policy documents (from Phase 1) to find the **most relevant and legally precise information**.

We implemented **multiple advanced retrieval strategies** designed to go far beyond simple keyword or semantic matching.

---

### ✅ Retrieval Methods Implemented

#### 1. **Standard RAG**
A baseline method that retrieves document chunks based solely on **semantic similarity** between the user's query and text embeddings.  
Simple and fast, but may struggle with domain-specific legal terms or multi-step queries.

---

#### 2. **Hybrid RAG**
Combines both **semantic vector search** and **keyword-based filtering**.  
This dual-layer approach is especially effective for policy documents that contain precise language and industry jargon, ensuring no critical clause is overlooked.

---

#### 3. **Re-ranking RAG**
Retrieves a **larger set of candidate chunks**, then uses a **fine-tuned cross-encoder or reranker model** to re-score and sort them.  
This significantly improves **precision**, especially when clauses are semantically similar but legally distinct.

---

#### 4. **Query Transformation RAG**
Uses an LLM to **break down a complex query** into simpler, atomic sub-queries.  
Each sub-query is then used independently to retrieve focused and complementary pieces of evidence.

---

#### 5. **Step-Back RAG**
A powerful method for complex reasoning tasks.  
The LLM first "steps back" to a **foundational or broader question**, retrieves documents for both the original and foundational query, and combines them for a deeper, **more complete answer**.

---

#### 6. **HyDE (Hypothetical Document Embeddings)**
Generates a **hypothetical answer** to the user's query, embeds it, and then uses this vector to search for **real documents** that are most semantically similar.  
Great for queries where exact wording doesn't exist in the document but the concept is present.

---

#### 7. **FLARE (Forward-Looking Active REtrieval)**
Dynamically anticipates which information might be needed next.  
The LLM **generates new sub-queries on-the-fly** and retrieves relevant documents as the reasoning unfolds.  
Useful in exploratory or multi-part queries.

---

#### 8. **Self-RAG**
A reflective approach where the LLM **evaluates its own retrieval and generation process**.  
If the answer appears weak or incomplete, it **requests additional context** and repeats the retrieval step—improving both coverage and confidence.

---

### 🎯 Why Re-ranking and Step-Back Are Especially Important

For complex and legally sensitive documents like **insurance policies**, the following retrieval methods are **particularly beneficial**:

- **Re-ranking RAG**  
  Ensures high-precision clause selection by surfacing **only the most legally relevant and accurate content**, avoiding confusion caused by near-duplicate phrasing.

- **Step-Back RAG**  
  Supports depth and clarity in multi-step reasoning by anchoring answers in **core definitions**, then layering on specific conditions—critical for compliance-heavy domains.

---

### ✅ Outcome

By completing Phase 2, the system now has:

- **Multiple retrieval strategies** tailored for both precision and recall  
- A flexible pipeline that adapts to query complexity  
- Tools to handle **both exact and abstract queries**  
- Strong foundation for Phase 3: LLM reasoning, logic validation, and JSON output generation
