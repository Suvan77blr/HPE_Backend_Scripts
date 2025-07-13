import uvicorn
from WRKING_api import app # Your FastAPI app instance
# from api_NEW import app # Your FastAPI app instance
from server_tools.update_checker import UpdateChecker # Your UpdateChecker class
import logging
from server_tools.ingestion import DataIngestionPipeline # Your DataIngestionPipeline class
from server_tools.vector_store import VectorStore # Your VectorStore class
from server_tools.scraper import NetworkDocScraper # Your NetworkDocScraper class
import os
from environment import GROQ_API_KEY, ENABLE_WEB_SEARCH # Assuming these are in your environment.py
# --- Configure logging (do this once at the application start) ---
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler() 
    ]
)
logger = logging.getLogger(__name__)

# --- Check for essential configurations ---
if not GROQ_API_KEY:
    logger.warning("GROQ_API_KEY is not set in environment. LLM functionality will be limited.")
if not ENABLE_WEB_SEARCH: 
    logger.info("Web search functionality is disabled via environment settings.")


def get_user_confirmation(message: str) -> bool:
    while True:
        try:
            response = input(f"{message} (y/n): ").lower().strip()
            if response in ['y', 'yes']:
                return True
            elif response in ['n', 'no']:
                return False
            else:
                print("Invalid input. Please enter 'y' or 'n'.")
        except EOFError: 
            logger.warning("EOFError during user confirmation, defaulting to 'no'.")
            return False


# --- MODIFIED: Expects list of dicts: [{"path": "file.txt", "vendor": "VendorName"}, ...] ---
def warm_scraper_cache_from_link_files(link_sources_with_vendor: list[dict]):
    logger.info("Starting scraper cache warming process...")
    scraper_for_warming = NetworkDocScraper() 
    
    total_urls_processed = 0
    total_contents_cached_or_verified = 0

    for source_info in link_sources_with_vendor:
        file_path = source_info["path"]
        vendor_for_warming = source_info["vendor"]

        if not os.path.exists(file_path):
            logger.warning(f"Link file for cache warming not found: {file_path}. Skipping.")
            continue
        
        logger.info(f"Processing links for {vendor_for_warming} from cache warming file: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                urls_from_file = [line.strip() for line in f if line.strip()]
        except Exception as e_read:
            logger.error(f"Error reading link file {file_path}: {e_read}")
            continue
        
        if not urls_from_file:
            logger.info(f"No URLs found in {file_path}.")
            continue

        for i, url in enumerate(urls_from_file):
            logger.info(f"Warming cache for {vendor_for_warming} URL ({i+1}/{len(urls_from_file)} from {os.path.basename(file_path)}): {url}")
            total_urls_processed += 1
            try:
                doc_type_for_warming = "pdf" if url.lower().endswith(".pdf") else "html"
                
                doc_meta = {
                    'url': url, 
                    'title': f"CacheWarming: {vendor_for_warming} - {os.path.basename(url.split('?')[0])}",
                    'doc_type': doc_type_for_warming,
                    'vendor': vendor_for_warming 
                }
                
                # Use extract_document_content which handles both PDF and HTML and their caching
                content = scraper_for_warming.extract_document_content(doc_meta, force_live=False) 
                
                if content is not None: 
                    logger.info(f"Successfully ensured content is cached for: {url} (Content length: {len(content)})")
                    total_contents_cached_or_verified += 1
                else:
                    logger.warning(f"Failed to obtain/cache content for: {url} (see scraper logs).")
            except Exception as e_warm:
                logger.error(f"Unexpected error during cache warming for URL {url}: {e_warm}", exc_info=True) 
    
    if scraper_for_warming.cache:
        try: scraper_for_warming.cache.close()
        except Exception as e_cache_close: logger.warning(f"Error closing cache for warming scraper: {e_cache_close}")
    scraper_for_warming._quit_selenium_driver()

    logger.info(f"Scraper cache warming completed. Processed {total_urls_processed} URLs, successfully obtained/verified content for {total_contents_cached_or_verified} items in cache.")


if __name__ == "__main__":
    logger.info("Application starting...")
    update_checker_instance = None 
    
    try:
        ingestion_pipeline = DataIngestionPipeline()
        vector_store_instance = VectorStore() # Used for reset
        update_checker_instance = UpdateChecker()

        if get_user_confirmation("Do you want to reset the database AND ALL TRACKING files?"):
            logger.info("Proceeding with full reset...")
            vector_store_instance.reset_database()
            ingestion_pipeline.reset_ingestion_tracking()
            if os.path.exists(update_checker_instance.state_file_path):
                try: os.remove(update_checker_instance.state_file_path); logger.info("Removed UpdateChecker state file.")
                except OSError as e_remove_state: logger.error(f"Error removing UpdateChecker state file: {e_remove_state}")
            update_checker_instance.mother_pages_seen_links = {}
            logger.info("Database, ingestion tracking, and update checker state have been reset.")

            if get_user_confirmation("Also clear the scraper's diskcache? (Recommended after full reset)"):
                temp_scraper_for_clear = NetworkDocScraper()
                if temp_scraper_for_clear.cache:
                    try: temp_scraper_for_clear.cache.clear(); logger.info("Scraper diskcache has been cleared.")
                    except Exception as e_clear_cache: logger.error(f"Error clearing scraper diskcache: {e_clear_cache}")
                    finally:
                        try: temp_scraper_for_clear.cache.close()
                        except: pass
                temp_scraper_for_clear._quit_selenium_driver()
        
        # --- MODIFIED: Define paths and structure for link files for warming ---
        arista_links_file = "./aristalinks.txt" 
        aruba_links_file = "./arubalinks.txt"
        juniper_links_file = "./sitemap_urls.txt" # Your Juniper file path
        
        link_files_sources_for_warming = [] 
        if os.path.exists(arista_links_file): 
            link_files_sources_for_warming.append({"path": arista_links_file, "vendor": "Arista"})
        else: logger.warning(f"Cache warming: aristalinks.txt not found at {arista_links_file}")
        
        if os.path.exists(aruba_links_file): 
            link_files_sources_for_warming.append({"path": aruba_links_file, "vendor": "Aruba"})
        else: logger.warning(f"Cache warming: arubalinks.txt not found at {aruba_links_file}")

        if os.path.exists(juniper_links_file): 
            link_files_sources_for_warming.append({"path": juniper_links_file, "vendor": "Juniper"})
        else: logger.warning(f"Cache warming: sitemap_urls.txt (Juniper) not found at {juniper_links_file}")

        if link_files_sources_for_warming:
            if get_user_confirmation(f"Do you want to pre-populate/warm the scraper's diskcache using configured link files?"):
                warm_scraper_cache_from_link_files(link_files_sources_for_warming) # Pass list of dicts
        else:
            logger.info("No link files found for cache warming, or paths are incorrect.")

        json_ingestion_sources = [
            {"path": "./scraped_data_0-1000.json", "vendor": "Juniper"},
            # {"path": "./JUNIPER_merged_scraped_data.json", "vendor": "Juniper"},
            # {"path": "./ARUBA_merged_scraped_data.json", "vendor": "Aruba"},
            # {"path": "./CISCO_merged_scraped_data.json", "vendor": "Cisco"},
            # {"path": "./CISCO_8000_merged_scraped_data.json", "vendor": "Cisco"},
            # {"path": "./arista_data.json", "vendor": "Arista"},
            # {"path": "./scraped_data.json", "vendor": "error_codes"},
        ]

        for source in json_ingestion_sources:
            if os.path.exists(source["path"]):
                if get_user_confirmation(f"Do you want to ingest data from {source['vendor']} JSON file '{os.path.basename(source['path'])}'?"):
                    if source["vendor"] == "error_codes":
                        ingestion_pipeline.ingest_scraped_data(source["path"]) 
                    else:
                        ingestion_pipeline.ingest_json_file(source["path"], source["vendor"])
            else:
                logger.warning(f"{source['vendor']} JSON file not found at {source['path']}. Skipping.")
        
        if get_user_confirmation("Do you want to run specific web scraping tasks (e.g., initial docs scrape)?"):
            # Add any specific, one-off scraping tasks here if needed
            logger.info("No specific one-off web scraping tasks defined in main.py for this run.")

        if get_user_confirmation("Enable and start automatic background checks for document updates?"):
            update_checker_instance.start(force_reingest_all_updates_this_run=False) 
        else:
            logger.info("Automatic background update checker is disabled by user for this session.")

        logger.info("Starting FastAPI server on http://0.0.0.0:8000")
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

    except KeyboardInterrupt:
        logger.info("Application shutting down due to KeyboardInterrupt...")
    except Exception as e_main:
        logger.critical(f"CRITICAL UNHANDLED ERROR IN MAIN APPLICATION FLOW: {str(e_main)}", exc_info=True)
    finally:
        logger.info("Initiating application cleanup...")
        if update_checker_instance and update_checker_instance.running:
            logger.info("Stopping background update checker...")
            update_checker_instance.stop()
        logger.info("Application shutdown sequence complete.")

