from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging
import uvicorn
import os                                                           ###

from environment import ENABLE_WEB_SEARCH, MAX_SEARCH_RESULTS
from environment import GROQ_MODEL, GROQ_API_KEY

from server_tools.vector_store import VectorStore                # Your VectorStore class
from server_tools.llm_service_OLD import LLMService
from server_tools.agent_OLD import NetworkIntegrationAgent
from server_tools.update_checker import UpdateChecker            # Your UpdateChecker class
from server_tools.web_search import WebSearcher
# From main.py
from server_tools.ingestion import DataIngestionPipeline         # Your DataIngestionPipeline class
from server_tools.scraper import NetworkDocScraper               # Your NetworkDocScraper class

from server_tools.scraper_utils_NEW import warm_scraper_cache_from_link_files
from server_routes.mcp_api_OLD import router as mcp_router  # <- Import the MCP router
from server_routes.topology_api import router as topology_router

# Topology related.
from fastapi import UploadFile, File, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from topology_analyser import TopologyAnalyzer
# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Initialize Core Components ---
vector_store_instance = VectorStore()
llm_service_instance = LLMService(model_name=GROQ_MODEL)
web_searcher_instance = WebSearcher(max_results=MAX_SEARCH_RESULTS) if ENABLE_WEB_SEARCH else None   
agent_instance = NetworkIntegrationAgent(vector_store_instance, llm_service_instance, web_searcher_instance)

ingestion_pipeline = DataIngestionPipeline()
update_checker_instance = UpdateChecker()

# Topology related.
topology_analyser_instance = TopologyAnalyzer()

# --- Initialize FastAPI App ---
app = FastAPI(
    title="Network Integration Assistant API",
    description="API for assisting with network equipment integration across vendors",
    version="1.0.0"
)

app.include_router(mcp_router)  # <- Attach the MCP API routes here
app.include_router(topology_router, prefix="/topology")

# --- Schema Definitions ---
class QueryRequest(BaseModel):
    query: str
    metadata: Optional[Dict[str, Any]] = None

class QueryResponse(BaseModel):
    response: str
    analysis: Optional[Dict[str, Any]] = None

class IngestRequest(BaseModel):
    path: str
    vendor: str

# BigBro..
# RAG Query Request/Response
class RAGQueryRequest(BaseModel):
    user_query: str

class RAGQueryResponse(BaseModel):
    answer: str
    source_docs: Optional[list[Dict[str, Any]]] = None

# --- Core API Routes ---


@app.post("/query_rag", response_model=RAGQueryResponse)
async def process_rag_query(request: RAGQueryRequest):
    """Process a user query and return a response"""
    try:
        analysis = agent_instance.analyze_query(request.query)       
        
        answer = agent_instance.generate_response(request.query)
        
        return RAGQueryResponse(
            answer = answer,
            source_documents = [analysis]
        )
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
# end process_query()

# --- Admin API Routes ---
@app.post("/admin/reset")       # Resets the vector DB, ingestion tracking, and update checker states
def reset_all():
    try:
        # ingestion_pipeline = DataIngestionPipeline()
        vector_store_instance.reset_database()
        ingestion_pipeline.reset_ingestion_tracking()
        if os.path.exists(update_checker_instance.state_file_path):
            # os.remove(update_checker_instance.state_file_path)
            try: os.remove(update_checker_instance.state_file_path); logger.info("Removed UpdateChecker state file.")
            except OSError as e_remove_state: logger.error(f"Error removing UpdateChecker state file: {e_remove_state}")
        update_checker_instance.mother_pages_seen_links = {}
        logger.info("Vector DB, ingestion tracking, and update checker states has been reset.")
        return {"message": "Reset completed."}
    except Exception as e:
        logger.error(f"Error in /admin/reset: {str(e)}")
        raise HTTPException(status_code=500, detail="Reset failed")
# end reset_all().

@app.post("/admin/clear-cache") # Clears scraper diskcache.
def clear_scraper_cache():
    try:
        scraper = NetworkDocScraper()
        if scraper.cache:
            scraper.cache.clear()
            scraper.cache.close()
        scraper._quit_selenium_driver()
        logger.info("Scraper diskcache cleared.")
        return {"message": "Scraper diskcache cleared."}
    except Exception as e:
        logger.error(f"Error clearing scraper diskcache: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear cache.")
# end clear_scraper_cache().

# Needs checking of links.
@app.post("/admin/warm-cache") # Warms cache.
def warm_cache(background_tasks: BackgroundTasks):
    """
    def warm():
        logger.info("Cache warming started...")
        scraper = NetworkDocScraper()
        link_sources = [
            {"path": "./aristalinks.txt", "vendor": "Arista"},
            {"path": "./arubalinks.txt", "vendor": "Aruba"},
            {"path": "./sitemap_urls.txt", "vendor": "Juniper"},
        ]
        for src in link_sources:
            if not os.path.exists(src["path"]):
                logger.warning(f"File missing: {src['path']}")
                continue
            with open(src["path"], 'r', encoding='utf-8') as f:
                urls = [line.strip() for line in f if line.strip()]
            for url in urls:
                try:
                    doc_type = "pdf" if url.lower().endswith(".pdf") else "html"
                    doc_meta = {
                        "url": url,
                        "title": f"API CacheWarming: {src['vendor']}",
                        "doc_type": doc_type,
                        "vendor": src["vendor"]
                    }
                    scraper.extract_document_content(doc_meta, force_live=False)
                except Exception as e:
                    logger.error(f"Failed warming cache for {url}: {str(e)}")
        scraper._quit_selenium_driver()
        logger.info("Cache warming complete.")
    """
    def warm():
        link_sources = [
            {"path": "./aristalinks.txt", "vendor": "Arista"},
            {"path": "./arubalinks.txt", "vendor": "Aruba"},
            {"path": "./sitemap_urls.txt", "vendor": "Juniper"},
        ]
        warm_scraper_cache_from_link_files(link_sources)

    background_tasks.add_task(warm)
    return {"message": "Cache warming started in background."}
# end warm_cache().

# This allows frontend/admin interface to upload or refer to a file path and start ingestion.
@app.post("/admin/ingest-json") # Ingests a JSON file.
def ingest_json(req: IngestRequest):
    try:
        if not os.path.exists(req.path):
            raise FileNotFoundError(f"File not found: {req.path}")
        if req.vendor.lower() == "error_codes":
            ingestion_pipeline.ingest_scraped_data(req.path)
        else:
            ingestion_pipeline.ingest_json_file(req.path, req.vendor)
        return {"message": f"Ingested {req.path} for vendor {req.vendor}"}
    except Exception as e:
        logger.error(f"Error in /admin/ingest-json: {str(e)}")
        raise HTTPException(status_code=500, detail="Ingestion failed")
# end ingest_json().

@app.post("/admin/start-update-checker") # Starts the update checker.
def start_update_checker(background_tasks: BackgroundTasks):
    def run_checker():
        logger.info("Update checker started...")
        update_checker_instance.start(force_reingest_all_updates_this_run=False)
        logger.info("Update checker completed.")

    background_tasks.add_task(run_checker)
    return {"message": "Update checker started in background."}
# end start_update_checker().


@app.get("/status")     # Returns status info
def get_status():
    return {
        "vector_store_ready": True,
        "llm_model": GROQ_MODEL,
        "web_search_enabled": ENABLE_WEB_SEARCH,
        "update_checker_running": update_checker_instance.running
    }
# end get_status().


# BigBro ...
# RAG Endpoint:
# @app.post("/query_rag", response_model=RAGQueryResponse)
# async def rag_query_handler(request: RAGQueryRequest):
#     """
#     Accepts a natural language query from the user and processes it via the NetworkIntegrationAgent,
#     returning the answer along with supporting source documents.
#     """
#     try:
#         logger.info(f"[RAG] Received query: {request.query}")

#         # Using the agent_instance to process the query.
        


# @app.post("/query", response_model=QueryResponse)
# async def process_query(request: QueryRequest):
#     """Process a user query and return a response"""
#     try:
#         analysis = agent_instance.analyze_query(request.query)       
        
#         response = agent_instance.generate_response(request.query)
        
#         return QueryResponse(
#             response=response,
#             analysis=analysis
#         )
#     except Exception as e:
#         logger.error(f"Error processing query: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

# if get_user_confirmation("Do you want to reset the database AND ALL TRACKING files?"):
#             logger.info("Proceeding with full reset...")
#             vector_store_instance.reset_database()
#             ingestion_pipeline.reset_ingestion_tracking()
#             if os.path.exists(update_checker_instance.state_file_path):
#                 try: os.remove(update_checker_instance.state_file_path); logger.info("Removed UpdateChecker state file.")
#                 except OSError as e_remove_state: logger.error(f"Error removing UpdateChecker state file: {e_remove_state}")
#             update_checker_instance.mother_pages_seen_links = {}
#             logger.info("Database, ingestion tracking, and update checker state have been reset.")



"""
BackgroundTasks: a FastAPI utility that lets you run functions asynchronously in the background, after the response has been sent to the client.
"""