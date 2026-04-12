# Pro-RAG: High-Performance Autonomous Portfolio Intelligence

Pro-RAG is an end-to-end Retrieval-Augmented Generation (RAG) system that transforms professional portfolio data into an intelligent, low-latency AI assistant. The project was engineered from scratch with a strong focus on modular software architecture, maintainability, and performance optimization.

---

## 1. Project Directory Structure

The project follows clean Separation of Concerns (SoC) principles:

```text
RAG/
├── runner.py              # Pipeline Controller (Entry Point)
├── app.py                 # FastAPI Server (Persistent Model Serving)
├── rag_engine.py          # Core Logic (Classes & Engineering Engines)
├── .env                   # Secret Management (API Keys)
├── requirements.txt       # Dependency Management
├── chunks.json            # Semantic Document Chunks
├── embedded_chunks.json   # Vector Embeddings (JSON Vector Store)
└── Master_RAG.pdf         # AI Structured Master Document
```

---

## 2. Engineering Architecture (Class-Based Design)

The system is divided into modular classes for maintainability and scalability:

| Class | Engineering Responsibility |
|--------|-----------------------------|
| ContentCrawler | Targeted URL scraping with DOM-based noise removal |
| AIEngine | LLM orchestration and raw-text-to-structured-Markdown synthesis via Gemini 3 Flash |
| DocumentChunker | Header-aware recursive splitting for high-fidelity context preservation |
| EmbedEngine | Local transformer-based embedding generation using `all-MiniLM-L6-v2` |
| SearchEngine | High-speed retrieval engine using NumPy-based matrix normalization |
| ChatEngine | Context-injected prompt engineering and deterministic response generation |

---

## 3. Conversational Memory & Context Preservation

The system is designed to support persistent conversational continuity through optimized context management:

1. **Incremental Context Updates:** New conversation history can be appended continuously over time  
2. **Previous Chat Referencing:** Historical chat references can be injected into the retrieval pipeline  
3. **Context Compression:** Prior interactions can be summarized/optimized before storage to reduce token overhead  
4. **Semantic Memory Retrieval:** Relevant prior conversations can be retrieved contextually during inference  
5. **Long-Term Context Maintenance:** Enables sustained multi-turn intelligence across sessions  

---

## 4. The 5-Step Data Pipeline

The system follows a linear professional data-processing pipeline:

1. **Extraction:** Asynchronously fetch clean data from portfolio sources  
2. **Synthesis:** Convert raw text into structured technical documentation using generative AI  
3. **Vectorization:** Transform document chunks into high-dimensional embedding vectors  
4. **Optimization:** Use NumPy matrix mathematics to achieve microsecond-level retrieval latency  
5. **Serving:** Keep model persistently loaded in RAM via FastAPI ASGI server for hot-start inference  

---

## 5. Technical Optimizations & Clean Code

Advanced engineering decisions were made to maximize scalability and performance:

### Persistent Model Loading
The model is loaded during server startup to eliminate repeated initialization overhead (~3.5s).

### Vectorized Search Operations
Optimized NumPy matrix operations replace traditional Python loops:

```python
# Vectorized Normalization & Similarity Scoring Logic
self.embeddings_matrix = np.array(
    [item['embedding'] for item in self.data]
).astype('float32')

self.embeddings_matrix /= (
    np.linalg.norm(self.embeddings_matrix, axis=1, keepdims=True) + 1e-10
)

# Instant dot product for similarity search
similarities = np.dot(self.embeddings_matrix, query_vector)
```

### Decoupled Architecture
Ingestion, processing, and inference layers are fully independent and modular.

---

## 6. Execution Guide

### Pipeline Execution (Full Flow)

To run the complete pipeline from scraping to vector indexing:

```bash
python runner.py full
```

### Production Serving (Sub-Second Response)

To start the high-speed inference server:

```bash
uvicorn app:app --reload
```

Once started, the model remains persistent in RAM and serves instant responses via:

```http
GET /ask?query=...
```

---

## Engineer Profile

- **Engineer:** Asad Ahmed  
- **Designation:** Senior Full Stack AI & IoT Engineer  
- **Core Expertise:** Distributed Systems, AI Inference Pipelines, Scalable Backend Architectures  
