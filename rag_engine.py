import os
import requests
import asyncio
import logging
from bs4 import BeautifulSoup
from fpdf import FPDF
from google import genai
from google.genai import types
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np
import time



# Load Environment Variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContentCrawler:
    def __init__(self, urls):
        self.urls = urls

    def fetch_all_clean_data(self):
        master_data = ""
        for url in self.urls:
            logger.info(f"🌐 Crawling: {url}")
            try:
                r = requests.get(url, timeout=15)
                soup = BeautifulSoup(r.text, 'html.parser')
                for noise in soup(["script", "style", "nav", "footer", "header", "button"]):
                    noise.decompose()
                clean_text = soup.get_text(separator=' ', strip=True)
                master_data += f"\n\n--- SOURCE: {url} ---\n{clean_text}"
            except Exception as e:
                logger.error(f"Error fetching {url}: {e}")
        return master_data

class AIEngine:
    def __init__(self, model_id="gemini-3-flash-preview"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("❌ GEMINI_API_KEY not found in .env file!")
        self.client = genai.Client(api_key=api_key)
        self.model_id = model_id

    async def synthesize(self, raw_corpus):
        prompt = f"""
        Act as a Professional Document Architect. Convert the following raw web-scraped data into a structured, HIGH-FIDELITY document for a RAG system.
        1. NO DATA LOSS: Preserve ALL technical details, project descriptions, and core information.
        2. STRUCTURE: Organise using Markdown headers (## for sections).
        3. FORMAT: Use bullet points and bold tech terms.
        DATA: {raw_corpus}
        """
        for attempt in range(3):
            try:
                response = self.client.models.generate_content(
                    model=self.model_id, contents=prompt,
                    config=types.GenerateContentConfig(temperature=0.1, max_output_tokens=8192)
                )
                # Logging token usage
                usage = response.usage_metadata
                logger.info(f"📊 Tokens used: {usage.total_token_count}")
                return response.text
            except Exception as e:
                logger.warning(f"AI Retry {attempt+1}: {e}")
                await asyncio.sleep(20)
        return "AI Error"

class PDFGenerator:
    def __init__(self, output_name):
        self.output_name = output_name

    def generate(self, content):
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("helvetica", size=10)
        safe_text = content.encode('latin-1', 'ignore').decode('latin-1')
        pdf.multi_cell(0, 6, safe_text)
        pdf.output(self.output_name)
        logger.info(f"✅ PDF Saved: {self.output_name}")

# --- INTEGRATED RETRIEVER CLASS WITH URLS ---
class PortfolioRetriever:
    """
    Ab saara control is class ke paas hai. 
    URLs bhi yahan hard-coded hain.
    """
    DEFAULT_URLS = [
        "https://asadahmed361.vercel.app",
        "https://asadahmed361.vercel.app/case_study_01",
        "https://asadahmed361.vercel.app/case_study_02",
        "https://asadahmed361.vercel.app/case_study_03",
        "https://asadahmed361.vercel.app/case_study_04",
        "https://asadahmed361.vercel.app/case_study_05"
    ]

    def __init__(self, output_pdf="Asad_Ahmed_Master_RAG.pdf"):
        self.output_pdf = output_pdf
        self.crawler = ContentCrawler(self.DEFAULT_URLS)
        self.ai = AIEngine()
        self.pdf_gen = PDFGenerator(output_pdf)

    async def run_update(self):
        print("🚀 Portfolio Update Service Triggered...")
        
        # Step 1: Fetch
        data = self.crawler.fetch_all_clean_data()
        
        # Step 2: Process
        structured_text = await self.ai.synthesize(data)
        
        # Step 3: Generate
        if structured_text != "AI Error":
            self.pdf_gen.generate(structured_text)
            print(f"✅ Success: {self.output_pdf} created with latest data.")
            return True
        else:
            print("❌ Failure: Could not process data.")
            return False
            
import fitz
import json
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter
)


class DocumentChunker:
    def __init__(self, chunk_size=600, chunk_overlap=100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.headers_to_split_on = [("#", "h1"), ("##", "h2"), ("###", "h3")]
        self.markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=self.headers_to_split_on, strip_headers=False)
        self.recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)

    def process_pdf(self, pdf_path):
        doc = fitz.open(pdf_path)
        text = "".join([page.get_text("text") for page in doc])
        header_chunks = self.markdown_splitter.split_text(text)
        
        final_chunks = []
        for chunk in header_chunks:
            sub_chunks = self.recursive_splitter.split_text(chunk.page_content)
            for sub_chunk in sub_chunks:
                final_chunks.append({"content": sub_chunk.strip(), "metadata": chunk.metadata})
        return final_chunks # Returning List directly    


class EmbedEngine:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """
        Constructor: Model aur initial settings load karta hai.
        """
        print(f"Initializing EmbedEngine with model: {model_name}")
        self.model = HuggingFaceEmbeddings(model_name=model_name)

class EmbedEngine:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = HuggingFaceEmbeddings(model_name=model_name)

    def generate_vectors(self, chunks):
        """Variable (list) input, returns List of dicts with embeddings"""
        texts = [c["content"] for c in chunks]
        vectors = self.model.embed_documents(texts)
        
        for i, chunk in enumerate(chunks):
            chunk["embedding"] = vectors[i]
        return chunks # Return embedded data in RAM
        


# class SearchEngine:
    # def __init__(self, data_path="embedded_chunks.json"):
        # """
        # Constructor: Model aur data load karta hai.
        # """
        # print("Initializing SearchEngine...")
        # self.model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # with open(data_path, "r", encoding="utf-8") as f:
            # self.data = json.load(f)
        # print(f"SearchEngine ready with {len(self.data)} chunks.")

    # def _get_similarity(self, v1, v2):
        # return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    # def get_top_matches(self, query, top_k=3, output_file="search_results.json"):
        # """
        # Search karta hai aur results ko JSON file mein save karta hai.
        # """
        # start_time = time.time()
        # query_vector = self.model.embed_query(query)
        
        # all_results = []
        # for item in self.data:
            # score = self._get_similarity(query_vector, item['embedding'])
            # all_results.append({
                # "score": float(score),
                # "content": item['content'],
                # "metadata": item['metadata']
            # })
        
        # # Sorting
        # top_results = sorted(all_results, key=lambda x: x['score'], reverse=True)[:top_k]
        
        # # Response object for JSON
        # final_output = {
            # "query": query,
            # "search_metadata": {
                # "top_k": top_k,
                # "execution_time_sec": round(time.time() - start_time, 4),
                # "total_pool_size": len(self.data)
            # },
            # "results": top_results
        # }

        # # Saving to JSON
        # with open(output_file, "w", encoding="utf-8") as f:
            # json.dump(final_output, f, indent=2, ensure_ascii=False)
        
        # print(f"Search results saved to {output_file}")
        # return top_results
        
class SearchEngine:
    def __init__(self, initial_data=None):
        self.model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.data = []
        self.embeddings_matrix = None
        if initial_data:
            self.update_index(initial_data)

    def update_index(self, embedded_data):
        """Update matrix in RAM without reading files"""
        self.data = embedded_data
        matrix = np.array([item['embedding'] for item in self.data]).astype('float32')
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        self.embeddings_matrix = matrix / (norms + 1e-10)
        logger.info(f"Search index refreshed with {len(self.data)} chunks.")

    def get_top_matches(self, query, top_k=3):
        if self.embeddings_matrix is None: return []
        
        query_vector = np.array(self.model.embed_query(query)).astype('float32')
        query_vector /= (np.linalg.norm(query_vector) + 1e-10)
        
        similarities = np.dot(self.embeddings_matrix, query_vector)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [{
            "score": float(similarities[i]),
            "content": self.data[i]['content'],
            "metadata": self.data[i]['metadata']
        } for i in top_indices]

class ChatEngine:
    def __init__(self, model_id="gemini-3-flash-preview"):
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.model_id = model_id

    def generate_response(self, query, search_results):
        context = "\n".join([f"SOURCE {i+1}:\n{m['content']}" for i, m in enumerate(search_results)])
        prompt = f"Act as Asad's AI. Answer based ONLY on context:\n\nCONTEXT:\n{context}\n\nUSER: {query}"
        
        response = self.client.models.generate_content(
            model=self.model_id, contents=prompt,
            config=types.GenerateContentConfig(temperature=0.1, max_output_tokens=800)
        )
        return response.text


