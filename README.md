# RAG Portfolio Intelligence Engine

RAG Portfolio Intelligence Engine is an end-to-end Retrieval-Augmented Generation (RAG) system that transforms professional portfolio data into an intelligent, low-latency AI assistant. The system autonomously crawls portfolio sources, synthesizes structured documentation, generates embeddings, and serves context-aware responses through a persistent in-memory retrieval engine.

---

![Asad's AI Assistant](Asads%20AI%20Assistant.gif)

---

## 1. Project Directory Structure

```text
RAG/
├── runner.py                  # Manual Pipeline Trigger
├── app.py                     # FastAPI Persistent Inference Server
├── rag_engine.py              # Core RAG Architecture / Engines
├── .env                       # Environment Variables / API Keys
├── requirements.txt           # Dependencies
└── Asad_Ahmed_Master_RAG.pdf  # Generated Structured Portfolio Knowledge Base
```

---

## 2. Engineering Architecture (Class-Based Design)

The platform follows a modular class-based architecture for maintainability and scalability:

| Class | Responsibility |
|--------|----------------|
| ContentCrawler | Fetches and cleans portfolio website data via DOM parsing |
| AIEngine | Converts raw crawled corpus into structured markdown documentation using Gemini |
| PDFGenerator | Generates portable structured knowledge-base PDF from synthesized content |
| PortfolioRetriever | Orchestrates crawling → synthesis → PDF generation pipeline |
| DocumentChunker | Parses PDF and creates semantic chunks using markdown-aware splitting |
| EmbedEngine | Generates transformer embeddings using `all-MiniLM-L6-v2` |
| SearchEngine | Maintains normalized in-memory vector matrix for ultra-fast retrieval |
| ChatEngine | Injects retrieved context into LLM prompt for grounded response generation |

---

## 3. Autonomous Data Synchronization Pipeline

The system supports a fully automated refresh pipeline:

1. **Crawl Latest Portfolio Data**  
2. **Generate Structured Master PDF**  
3. **Chunk Generated PDF into Semantic Sections**  
4. **Embed Chunks into Dense Vectors**  
5. **Refresh Search Index in RAM Without Restart**  

This enables hot data synchronization without redeploying or restarting the inference server.

---

## 4. FastAPI Persistent Serving Architecture

The FastAPI server keeps embeddings and retrieval matrices resident in RAM for sub-second inference.

### Startup Behavior
On application boot:

- Existing PDF is automatically loaded  
- Chunking and embedding are performed  
- Search matrix is precomputed in memory  

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ask` | GET | Query the RAG assistant |
| `/pipeline/update` | POST | Trigger full background sync pipeline |

---

## 5. Technical Optimizations

### Persistent Hot-Loaded Retrieval Index
Embedding matrix remains in RAM after startup for zero cold-start search latency.

### Vectorized Similarity Search
Uses normalized NumPy matrix operations instead of iterative loops:

```python
matrix = np.array([item['embedding'] for item in self.data]).astype('float32')
norms = np.linalg.norm(matrix, axis=1, keepdims=True)
self.embeddings_matrix = matrix / (norms + 1e-10)

similarities = np.dot(self.embeddings_matrix, query_vector)
```

### Background Pipeline Execution
Portfolio refresh runs asynchronously in background tasks without blocking API availability.

### Markdown-Aware Chunking
Preserves semantic hierarchy using header-aware recursive chunk splitting.

---

## 6. End-to-End Pipeline Flow

```text
Portfolio URLs
   ↓
ContentCrawler
   ↓
AIEngine (Gemini Synthesis)
   ↓
PDFGenerator
   ↓
DocumentChunker
   ↓
EmbedEngine
   ↓
SearchEngine (RAM Matrix)
   ↓
ChatEngine
   ↓
User Query Response
```

---

## 7. Execution Guide

### Run Manual Pipeline

```bash
python runner.py full
```

### Start Production Server

```bash
uvicorn app:app --reload
```

---

## 8. Example Query Endpoint

```http
GET /ask?query=Tell me about Asad's IoT experience
```

---

## Engineer Profile

- **Engineer:** Asad Ahmed  
- **Designation:** Senior Full Stack AI & IoT Engineer  
- **Core Expertise:** Distributed Systems, AI Inference Pipelines, Scalable Backend Architectures  
