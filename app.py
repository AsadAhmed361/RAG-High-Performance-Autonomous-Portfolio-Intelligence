#uvicorn app:app --reload
import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks, HTTPException
from rag_engine import PortfolioRetriever, DocumentChunker, EmbedEngine, SearchEngine, ChatEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- GLOBAL STORAGE ---
temp_storage = {
    "raw_chunks": [],
    "embedded_data": []
}

search_engine = SearchEngine()
chat_engine = ChatEngine()

# --- THE MASTER UPDATE FUNCTION ---
async def full_sync_pipeline():
    """
    Sequence: Crawl -> PDF -> Chunk -> Embed -> RAM Update
    """
    try:
        logger.info("Starting Full Sync: Crawling latest data...")
        # 1. Crawling & PDF Generation
        retriever = PortfolioRetriever()
        await retriever.run_update() 
        
        # 2. Chunking (Reading from the newly generated PDF)
        logger.info("Syncing: Chunking updated PDF...")
        chunker = DocumentChunker()
        temp_storage["raw_chunks"] = chunker.process_pdf("Asad_Ahmed_Master_RAG.pdf")
        
        # 3. Embedding
        logger.info("Syncing: Generating new vectors...")
        embedder = EmbedEngine()
        temp_storage["embedded_data"] = embedder.generate_vectors(temp_storage["raw_chunks"])
        
        # 4. Update Search Engine Matrix in RAM
        search_engine.update_index(temp_storage["embedded_data"])
        
        logger.info("Full Sync Complete: RAM is now updated with latest data.")
    except Exception as e:
        logger.error(f"Sync Pipeline Error: {e}")

# --- LIFESPAN EVENT ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup pe existing PDF ko load kar lete hain taake app khali na ho
    logger.info("Initial startup load from existing PDF...")
    try:
        chunker = DocumentChunker()
        temp_storage["raw_chunks"] = chunker.process_pdf("Asad_Ahmed_Master_RAG.pdf")
        embedder = EmbedEngine()
        temp_storage["embedded_data"] = embedder.generate_vectors(temp_storage["raw_chunks"])
        search_engine.update_index(temp_storage["embedded_data"])
    except Exception as e:
        logger.warning(f"Initial load failed (Normal if PDF doesn't exist yet): {e}")
    
    yield

app = FastAPI(title="Pro-RAG Master Sync API", lifespan=lifespan)

# --- THE TRIGGER ENDPOINT ---
@app.post("/pipeline/update")
async def trigger_update(background_tasks: BackgroundTasks):
    """
    Ye endpoint ab teeno steps (Crawl, Chunk, Embed) background mein chalayega.
    """
    background_tasks.add_task(full_sync_pipeline)
    return {"message": "Master update triggered. Crawling, chunking, and embedding are running in background."}

@app.get("/ask")
async def ask_asad(query: str):
    try:
        results = search_engine.get_top_matches(query)
        if not results:
            return {"answer": "Engine is empty. Please run /pipeline/update first."}
        
        answer = chat_engine.generate_response(query, results)
        return {"query": query, "answer": answer}
    except Exception as e:
        if "429" in str(e):
            raise HTTPException(status_code=429, detail="API Quota Exceeded.")
        raise HTTPException(status_code=500, detail=str(e))
