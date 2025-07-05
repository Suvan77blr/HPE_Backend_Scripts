from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging
import uvicorn
from vector_store import VectorStore
from llm_service import LLMService
from OLD_agent import NetworkIntegrationAgent
from update_checker import UpdateChecker
from web_search import WebSearcher
from environment import ENABLE_WEB_SEARCH, MAX_SEARCH_RESULTS
from environment import GROQ_MODEL, GROQ_API_KEY

from mcp_api import router as mcp_router
from topology_api import router as topology_router
from admin_api import router as admin_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

vector_store = VectorStore()
llm_service = LLMService(model_name=GROQ_MODEL)

web_searcher = None
if ENABLE_WEB_SEARCH:
    web_searcher = WebSearcher(max_results=MAX_SEARCH_RESULTS)

agent = NetworkIntegrationAgent(vector_store, llm_service, web_searcher)

app = FastAPI(
    title="Network Integration Assistant API",
    description="API for assisting with network equipment integration across vendors",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# app.include_router(mcp_router)  # <- Attach the MCP API routes here
app.include_router(topology_router, prefix="/topology")
app.include_router(admin_router, prefix="/admin")

class QueryRequest(BaseModel):
    query: str
    metadata: Optional[Dict[str, Any]] = None

class QueryResponse(BaseModel):
    response: str
    analysis: Optional[Dict[str, Any]] = None

@app.post("/query_rag", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a user query and return a response"""
    try:
        analysis = agent.analyze_query(request.query)
        
        response = agent.generate_response(request.query)
        
        return QueryResponse(
            response=response,
            analysis=analysis
        )
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    

# # --- Admin API Routes ---
# @app.post("/admin/reset")       # Resets the vector DB, ingestion tracking, and update checker states
# def reset_all():
#     try:
#         # ingestion_pipeline = DataIngestionPipeline()
#         vector_store_instance.reset_database()
#         ingestion_pipeline.reset_ingestion_tracking()
#         if os.path.exists(update_checker_instance.state_file_path):
#             # os.remove(update_checker_instance.state_file_path)
#             try: os.remove(update_checker_instance.state_file_path); logger.info("Removed UpdateChecker state file.")
#             except OSError as e_remove_state: logger.error(f"Error removing UpdateChecker state file: {e_remove_state}")
#         update_checker_instance.mother_pages_seen_links = {}
#         logger.info("Vector DB, ingestion tracking, and update checker states has been reset.")
#         return {"message": "Reset completed."}
#     except Exception as e:
#         logger.error(f"Error in /admin/reset: {str(e)}")
#         raise HTTPException(status_code=500, detail="Reset failed")
# # end reset_all().

# @app.post("/admin/clear-cache") # Clears scraper diskcache.
# def clear_scraper_cache():
#     try:
#         scraper = NetworkDocScraper()
#         if scraper.cache:
#             scraper.cache.clear()
#             scraper.cache.close()
#         scraper._quit_selenium_driver()
#         logger.info("Scraper diskcache cleared.")
#         return {"message": "Scraper diskcache cleared."}
#     except Exception as e:
#         logger.error(f"Error clearing scraper diskcache: {e}")
#         raise HTTPException(status_code=500, detail="Failed to clear cache.")
# # end clear_scraper_cache().

# @app.post("/admin/warm-cache") # Warms cache.
# def warm_cache(background_tasks: BackgroundTasks):
#     def warm():
#         link_sources = [
#             {"path": "./aristalinks.txt", "vendor": "Arista"},
#             {"path": "./arubalinks.txt", "vendor": "Aruba"},
#             {"path": "./sitemap_urls.txt", "vendor": "Juniper"},
#         ]
#         warm_scraper_cache_from_link_files(link_sources)

#     background_tasks.add_task(warm)
#     return {"message": "Cache warming started in background."}
# # end warm_cache().

# # This allows frontend/admin interface to upload or refer to a file path and start ingestion.
# @app.post("/admin/ingest-json") # Ingests a JSON file.
# def ingest_json(req: IngestRequest):
#     try:
#         if not os.path.exists(req.path):
#             raise FileNotFoundError(f"File not found: {req.path}")
#         if req.vendor.lower() == "error_codes":
#             ingestion_pipeline.ingest_scraped_data(req.path)
#         else:
#             ingestion_pipeline.ingest_json_file(req.path, req.vendor)
#         return {"message": f"Ingested {req.path} for vendor {req.vendor}"}
#     except Exception as e:
#         logger.error(f"Error in /admin/ingest-json: {str(e)}")
#         raise HTTPException(status_code=500, detail="Ingestion failed")
# # end ingest_json().

# @app.post("/admin/start-update-checker") # Starts the update checker.
# def start_update_checker(background_tasks: BackgroundTasks):
#     def run_checker():
#         logger.info("Update checker started...")
#         update_checker_instance.start(force_reingest_all_updates_this_run=False)
#         logger.info("Update checker completed.")

#     background_tasks.add_task(run_checker)
#     return {"message": "Update checker started in background."}
# # end start_update_checker().


# @app.get("/status")     # Returns status info
# def get_status():
#     return {
#         "vector_store_ready": True,
#         "llm_model": GROQ_MODEL,
#         "web_search_enabled": ENABLE_WEB_SEARCH,
#         "update_checker_running": update_checker_instance.running
#     }
# # end get_status().

# --- End Admin API Routes ---


# --- MCP API Routes ---
