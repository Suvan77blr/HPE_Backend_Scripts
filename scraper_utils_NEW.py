import os, logging
from scraper import NetworkDocScraper

logger = logging.getLogger(__name__)
LOG_TAG = "[CACHE-WARMING]"

def warm_scraper_cache_from_link_files(link_sources_with_vendor: list[dict]):
    logger.info(f"{LOG_TAG} Starting scraper cache warming process...")
    scraper_for_warming = NetworkDocScraper() 
    
    total_urls_processed = 0
    total_contents_cached_or_verified = 0

    for source_info in link_sources_with_vendor:
        file_path = source_info["path"]
        vendor_for_warming = source_info["vendor"]

        if not os.path.exists(file_path):
            logger.warning(f"{LOG_TAG} Link file for cache warming not found: {file_path}. Skipping.")
            continue
        
        logger.info(f"{LOG_TAG} Processing links for {vendor_for_warming} from cache warming file: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                urls_from_file = [line.strip() for line in f if line.strip()]
        except Exception as e_read:
            logger.error(f"{LOG_TAG} Error reading link file {file_path}: {e_read}")
            continue
        
        if not urls_from_file:
            logger.info(f"{LOG_TAG} No URLs found in {file_path}.")
            continue

        for i, url in enumerate(urls_from_file):
            logger.info(f"{LOG_TAG} Warming cache for {vendor_for_warming} URL ({i+1}/{len(urls_from_file)} from {os.path.basename(file_path)}): {url}")
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
                    logger.info(f"{LOG_TAG} Successfully ensured content is cached for: {url} (Content length: {len(content)})")
                    total_contents_cached_or_verified += 1
                else:
                    logger.warning(f"{LOG_TAG} Failed to obtain/cache content for: {url} (see scraper logs).")
            except Exception as e_warm:
                logger.error(f"{LOG_TAG} Unexpected error during cache warming for URL {url}: {e_warm}", exc_info=True) 
    
    if scraper_for_warming.cache:
        try: scraper_for_warming.cache.close()
        except Exception as e_cache_close: logger.warning(f"{LOG_TAG} Error closing cache for warming scraper: {e_cache_close}")
    scraper_for_warming._quit_selenium_driver()

    logger.info(f"{LOG_TAG} Scraper cache warming completed. Processed {total_urls_processed} URLs, successfully obtained/verified content for {total_contents_cached_or_verified} items in cache.")
