#uvicorn app:app --reload
import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks, HTTPException
from rag_engine import PortfolioRetriever, DocumentChunker, EmbedEngine, SearchEngine, ChatEngine
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str


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

# Initialize empty list for memory
chat_history_buffer = [] 

@app.get("/ask")
async def ask_asad(query: str):
    try:
        # 1. Pehle context nikalein
        results = search_engine.get_top_matches(query)
        
        # 2. AI se jawab lein (History bhej kar)
        answer = chat_engine.generate_response(
            query=query, 
            search_results=results, 
            history=chat_history_buffer
        )
        
        # 3. History Update Logic (Maintain exactly 5)
        chat_history_buffer.append({"user": query, "assistant": answer})
        
        if len(chat_history_buffer) > 5:
            chat_history_buffer.pop(0) # Sabse purana message (index 0) remove kar do
            
        return {"query": query, "answer": answer}
    except Exception as e:
        logger.error(f"Error: {e}")
        return {"error": str(e)}

@app.get("/", response_class=HTMLResponse)
async def get_ui():
    return FileResponse("index.html")

# 2. UI ki POST request handle karne ke liye naya endpoint
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        query = request.message
        
        # 1. Search (rag_engine logic)
        results = search_engine.get_top_matches(query)
        
        # 2. Generate Response with History
        # Note: 'chat_history_buffer' aapne pehle hi banaya hua hai
        answer = chat_engine.generate_response(query, results, chat_history_buffer)
        
        # 3. History update (sliding window of 5)
        chat_history_buffer.append({"user": query, "assistant": answer})
        if len(chat_history_buffer) > 5:
            chat_history_buffer.pop(0)
            
        return {"reply": answer} # 'reply' key matches your JS 'data.reply'
        
    except Exception as e:
        logger.error(f"Chat Error: {e}")
        return {"reply": "Sorry, I am having trouble connecting to my brain right now."}