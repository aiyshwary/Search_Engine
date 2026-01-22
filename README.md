## 🔍 Multilingual Hybrid Search Engine (BM25 + FAISS)
A production-style multilingual hybrid search engine designed for Indian-language documents.
The system combines lexical search (BM25 via OpenSearch) with semantic search (FAISS IVF-PQ) and supports OCR-based PDFs, multiple Indic languages, and document-level ranking.

## ✨ Key Features
### 📄 PDF & DOCX ingestion
Digital text extraction
OCR fallback using Tesseract for scanned documents

### 🌐 Multilingual support
Indian languages: Tamil, Hindi, Telugu, Kannada, Bengali, Marathi, Gujarati, etc.
Language detection using FastText (LID176) with Lingua fallback
ISO-639-1 → ISO-639-3 normalization across the pipeline

### ✂️ Token-aware chunking
Sentence-based splitting
Fixed token windows with overlap
Page-level metadata preserved

### 🧹 Document deduplication
Near-duplicate detection using SimHash

### 🧠 Semantic embeddings
Multilingual Sentence Transformers
Normalized embeddings for cosine similarity

### ⚡ Vector search with FAISS
IVF-PQ index for scalable ANN search
Configurable nprobe for recall/latency tradeoff

### 🔎 Hybrid retrieval
BM25 (OpenSearch) + ANN (FAISS)
Union and document-level score aggregation

### 📊 Document-level ranking
Chunk-level evidence
Best-snippet selection per document

🏗️ Architecture Overview
Documents (PDF/DOCX)
        ↓
Optimized Ingestion
(text + OCR + language detection)
        ↓
SimHash Deduplication
        ↓
Token-aware Chunking
        ↓
Embeddings (Sentence Transformers)
        ↓
FAISS Index (IVF-PQ)
        ↓
OpenSearch BM25 Index
        ↓
Hybrid Search (BM25 + ANN)
        ↓
Document-level Ranking

📁 Repository Structure
 ```
 ├── ingest.py                # Basic ingestion pipeline
 ├── optimized_ingest.py      # Parallel, OCR-aware ingestion (recommended)
 ├── dedupe_simhash.py        # Near-duplicate document detection
 ├── chunker.py               # Token-aware chunking with overlap
 ├── embedder.py              # Multilingual embeddings generation
 ├── build_faiss.py           # FAISS index builders (IVF-PQ, HNSW, Flat)
 ├── search_service.py        # Hybrid search API (BM25 + FAISS)
 ├── dataset/                 # Input documents (PDF/DOCX)
 └── output/
    ├── documents.jsonl
    ├── chunks.jsonl
    ├── chunks_embeddings.npy
    ├── embeddings_meta.jsonl
    └── faiss_ivfpq.index ```

## 🚀 End-to-End Pipeline
1️⃣ Ingest documents
``` python optimized_ingest.py ```
Outputs:
output/documents.jsonl

2️⃣ Deduplicate documents
python dedupe_simhash.py
Outputs:
output/deduped_docs.jsonl

3️⃣ Chunk documents
python chunker.py
Outputs:
output/chunks.jsonl

4️⃣ Generate embeddings
python embedder.py
Outputs:
chunks_embeddings.npy
embeddings_meta.jsonl

5️⃣ Build FAISS index
python build_faiss.py
Outputs:
faiss_ivfpq.index

6️⃣ Search (Hybrid BM25 + ANN)
python search_service.py
Example query:
q = "தமிழ்நாடு அரசு புதிய திட்டம்"
lang = "ta"

