# routes/admin_api.py

from fastapi import APIRouter, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import logging
import uvicorn
import os

from environment import ENABLE_WEB_SEARCH, MAX_SEARCH_RESULTS
from environment import GROQ_MODEL, GROQ_API_KEY

from server_tools.vector_store import VectorStore
from server_tools.llm_service_OLD import LLMService
from server_tools.agent_OLD import NetworkIntegrationAgent
from server_tools.update_checker import UpdateChecker
from server_tools.web_search import WebSearcher
from server_tools.ingestion import DataIngestionPipeline
from server_tools.scraper import NetworkDocScraper

from server_tools.web_search import WebSearcher
from server_tools.scraper_utils_NEW import warm_scraper_cache_from_link_files

router = APIRouter(prefix="/admin", tags=["Admin Panel"])

logger = logging.getLogger(__name__)

vector_store_instance = VectorStore()
llm_service_instance = LLMService(model_name=GROQ_MODEL)
ingestion_pipeline = DataIngestionPipeline()
update_checker_instance = UpdateChecker()
web_searcher_instance = WebSearcher(max_results=MAX_SEARCH_RESULTS) if ENABLE_WEB_SEARCH else None   

# ---Schema Definitions---
class IngestRequest(BaseModel):
    path: str
    vendor: str

# --- Admin API Routes ---
@router.post("/reset")       # Resets the vector DB, ingestion tracking, and update checker states
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

@router.post("/clear-cache") # Clears scraper diskcache.
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
@router.post("/warm-cache") # Warms cache.
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
@router.post("/ingest-json") # Ingests a JSON file.
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

@router.post("/start-update-checker") # Starts the update checker.
def start_update_checker(background_tasks: BackgroundTasks):
    def run_checker():
        logger.info("Update checker started...")
        update_checker_instance.start(force_reingest_all_updates_this_run=False)
        logger.info("Update checker completed.")

    background_tasks.add_task(run_checker)
    return {"message": "Update checker started in background."}
# end start_update_checker().


@router.get("/status")     # Returns status info
def get_status():
    return {
        "vector_store_ready": True,
        "llm_model": GROQ_MODEL,
        "web_search_enabled": ENABLE_WEB_SEARCH,
        "update_checker_running": update_checker_instance.running
    }
# end get_status().
