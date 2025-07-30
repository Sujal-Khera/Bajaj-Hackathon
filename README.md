# Bajaj-Hackathon
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

### ✅ Outcome

By completing this phase, we now have:

- **Clean**, de-noised text  
- **Semantically coherent** chunks  
- **Token-count aware** segments  
- **Traceable metadata** for each clause

This dataset is now **fully ready** for transformation into numerical embeddings, powering **efficient and accurate semantic retrieval** in Phase 2.
