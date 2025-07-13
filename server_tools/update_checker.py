import schedule
import time
import logging
from typing import Dict, List, Any, Optional # Added Optional
from datetime import datetime as dt 
import threading
from server_tools.scraper import NetworkDocScraper 
from server_tools.ingestion import DataIngestionPipeline 
from environment import UPDATE_CHECK_INTERVAL 
import json
import os

logger = logging.getLogger(__name__)

def get_user_confirmation(message: str) -> bool:
    while True:
        try:
            response = input(f"{message} (y/n): ").lower().strip()
            if response in ['y', 'yes']: return True
            elif response in ['n', 'no']: return False
            else: print("Invalid input. Please enter 'y' or 'n'.")
        except EOFError: logger.warning("EOFError during user confirmation, defaulting to 'no'."); return False

class UpdateChecker:
    def __init__(self):
        self.scraper = NetworkDocScraper() 
        self.ingestion_pipeline = DataIngestionPipeline() 
        self.state_file_path = os.path.join(self.ingestion_pipeline.tracking_dir, "update_checker_state.json")
        self.mother_pages_seen_links = self._load_checker_state() 
        self.running = False
        self.thread = None
        logger.info("UpdateChecker initialized.")

    def _load_checker_state(self) -> dict:
        if os.path.exists(self.state_file_path):
            try:
                with open(self.state_file_path, 'r', encoding='utf-8') as f: state = json.load(f)
                logger.info(f"Loaded UpdateChecker state from {self.state_file_path} ({len(state)} mother pages).")
                return state
            except Exception as e: logger.error(f"Error loading UpdateChecker state: {e}. Starting fresh.", exc_info=True)
        return {}

    def _save_checker_state(self):
        os.makedirs(os.path.dirname(self.state_file_path), exist_ok=True)
        try:
            with open(self.state_file_path, 'w', encoding='utf-8') as f: json.dump(self.mother_pages_seen_links, f, indent=2)
            logger.debug(f"Saved UpdateChecker state to {self.state_file_path}.")
        except Exception as e: logger.error(f"Failed to save UpdateChecker state: {e}", exc_info=True)

    # --- ADDITION: arista_page_type parameter ---
    def get_current_documents_from_mother_page(self, mother_url: str, vendor: str, arista_page_type: Optional[str] = None) -> list[dict]:
        current_docs_metadata = []
        # --- MODIFICATION: Log arista_page_type ---
        logger.info(f"Scraping page for {vendor} (type: {arista_page_type or 'default'}) at {mother_url}")

        try:
            if vendor.lower() == "arista":
                # --- MODIFICATION: Call specific Arista parsers based on arista_page_type ---
                if arista_page_type == "software_general":
                    current_docs_metadata = self.scraper.parse_arista_software_documentation_page(mother_url)
                elif arista_page_type == "hardware":
                    current_docs_metadata = self.scraper.parse_arista_hardware_documentation_page(mother_url)
                else:
                    logger.warning(f"Unknown or missing arista_page_type '{arista_page_type}' for Arista URL: {mother_url}. No Arista-specific parsing will occur.")
                    # If you had a generic `parse_arista_documentation_mother_page` for other Arista URLs, you could call it here.
                    # For now, this will result in no docs for un-typed Arista URLs.
            elif vendor.lower() == "aruba":
                current_docs_metadata = self.scraper.parse_aruba_documentation_series_page(mother_url)
            elif vendor.lower() == "juniper":
                current_docs_metadata = self.scraper.parse_juniper_sitemap_xml(mother_url)
            elif vendor.lower() == "cisco":
                if "release-notes-list.html" in mother_url: 
                     current_docs_metadata = self.scraper.parse_cisco_release_notes(mother_url)
                elif "configuration-guides-list.html" in mother_url:
                     current_docs_metadata = self.scraper.parse_cisco_config_guides(mother_url)
                else: logger.warning(f"No specific Cisco parsing logic matched for mother URL: {mother_url}")
            else:
                logger.warning(f"No specific mother page parsing logic for vendor '{vendor}' in UpdateChecker.")
        except Exception as e_scrape:
            logger.error(f"Error during scraping mother page {mother_url} for {vendor} (type: {arista_page_type or 'default'}): {e_scrape}", exc_info=True)
            return []

        valid_docs = []
        if current_docs_metadata:
            for doc_meta in current_docs_metadata:
                if isinstance(doc_meta, dict) and 'url' in doc_meta and 'title' in doc_meta:
                    doc_meta['vendor'] = vendor 
                    valid_docs.append(doc_meta)
                else: logger.warning(f"Skipping invalid doc_meta from {vendor} {mother_url}: {str(doc_meta)[:200]}")
        logger.info(f"Found {len(valid_docs)} valid document links on {vendor} page: {mother_url} (type: {arista_page_type or 'default'})")
        return valid_docs

    # --- ADDITION: arista_page_type parameter ---
    def check_and_process_vendor_updates(self, mother_url: str, vendor: str, arista_page_type: Optional[str] = None, force_reingest_discovered_docs: bool = False):
        # --- MODIFICATION: Log arista_page_type ---
        logger.info(f"Update check for {vendor} (type: {arista_page_type or 'default'}) at {mother_url}. Force re-ingest: {force_reingest_discovered_docs}")
        
        # --- MODIFICATION: Pass arista_page_type ---
        current_document_metadatas_on_page = self.get_current_documents_from_mother_page(mother_url, vendor, arista_page_type)
        
        if not current_document_metadatas_on_page:
            logger.info(f"No documents currently found on {vendor} page: {mother_url} (type: {arista_page_type or 'default'}).")
            self.mother_pages_seen_links[mother_url] = {} 
            self._save_checker_state()
            return

        previously_seen_doc_urls_map = self.mother_pages_seen_links.get(mother_url, {})
        docs_to_process_this_cycle = []
        current_urls_on_mother_page_map = {}
        for doc_meta in current_document_metadatas_on_page:
            doc_url = doc_meta['url']
            doc_title = doc_meta.get('title', 'Unknown Title')
            current_urls_on_mother_page_map[doc_url] = True
            if doc_url not in previously_seen_doc_urls_map:
                logger.info(f"NEW document link on {vendor} page: '{doc_title}' ({doc_url})")
                docs_to_process_this_cycle.append(doc_meta)
            elif force_reingest_discovered_docs:
                logger.info(f"FORCING RE-PROCESS of previously seen link: '{doc_title}' ({doc_url})")
                docs_to_process_this_cycle.append(doc_meta)
        
        if docs_to_process_this_cycle:
            logger.info(f"Found {len(docs_to_process_this_cycle)} new/force-reprocess document(s) for {vendor} from {mother_url} (type: {arista_page_type or 'default'}).")
            # --- MODIFICATION: Include arista_page_type in doc_type_prefix for better tracking if needed ---
            page_type_suffix = f"_{arista_page_type}" if arista_page_type else ""
            doc_type_prefix = f"{vendor.lower()}{page_type_suffix}_update"
            self.ingestion_pipeline._process_document_list(
                document_links=docs_to_process_this_cycle, vendor=vendor, 
                doc_type_prefix=doc_type_prefix, force_reingest=force_reingest_discovered_docs 
            )
        else:
            logger.info(f"No new or force-reprocess documents for {vendor} from {mother_url} (type: {arista_page_type or 'default'}) this cycle.")
        
        self.mother_pages_seen_links[mother_url] = current_urls_on_mother_page_map
        self._save_checker_state()
        logger.info(f"Finished update check for {vendor} at {mother_url} (type: {arista_page_type or 'default'}). State saved.")


    


    def start(self, force_reingest_all_updates_this_run: bool = False):
        if self.running: logger.info("Update checker already running."); return
        self.running = True
        logger.info(f"Update checker starting. Interval: {UPDATE_CHECK_INTERVAL} hours. Immediate run force re-ingest: {force_reingest_all_updates_this_run}")

        aruba_series_index_pages_to_monitor = [
            "https://arubanetworking.hpe.com/techdocs/AOS-CX/Consolidated_RNs/HTML-10000/Content/PDFs.htm",
            "https://arubanetworking.hpe.com/techdocs/AOS-CX/Consolidated_RNs/HTML-9300/Content/PDFs.htm",
            "https://arubanetworking.hpe.com/techdocs/AOS-CX/Consolidated_RNs/HTML-8400/Content/PDFs.htm",
            "https://arubanetworking.hpe.com/techdocs/AOS-CX/Consolidated_RNs/HTML-8360/Content/PDFs.htm",
            "https://arubanetworking.hpe.com/techdocs/AOS-CX/Consolidated_RNs/HTML-8325/Content/PDFs.htm",
            "https://arubanetworking.hpe.com/techdocs/AOS-CX/Consolidated_RNs/HTML-8320/Content/PDFs.htm",
            "https://arubanetworking.hpe.com/techdocs/AOS-CX/Consolidated_RNs/HTML-8100/Content/PDFs.htm",
            "https://arubanetworking.hpe.com/techdocs/AOS-CX/Consolidated_RNs/HTML-6300-6400/Content/PDFs.htm",
            "https://arubanetworking.hpe.com/techdocs/AOS-CX/Consolidated_RNs/HTML-6200/Content/PDFs.htm",
            "https://arubanetworking.hpe.com/techdocs/AOS-CX/Consolidated_RNs/HTML-6100/Content/PDFs.htm",
            "https://arubanetworking.hpe.com/techdocs/AOS-CX/Consolidated_RNs/HTML-6000/Content/PDFs.htm",
            "https://arubanetworking.hpe.com/techdocs/AOS-CX/Consolidated_RNs/HTML-4100i/Content/PDFs.htm",
        ]
        
        vendors_and_pages_to_monitor = [
            # --- ADDITION: Specific entries for the two Arista pages with their types ---
            {"vendor": "Arista", "url": "https://www.arista.com/en/support/product-documentation", "arista_page_type": "software_general"},
            {"vendor": "Arista", "url": "https://www.arista.com/en/support/product-documentation/hardware", "arista_page_type": "hardware"},
            {"vendor": "Juniper", "url": "https://www.juniper.net/documentation/sitemap/us/en/sitemap3.xml", "arista_page_type": None},
            # (Your Cisco URLs would go here if you add them, with arista_page_type: None)
        ]
        # --- Aruba URLs added to the list, keeping their arista_page_type as None ---
        for aruba_series_url in aruba_series_index_pages_to_monitor:
            if aruba_series_url not in [job['url'] for job in vendors_and_pages_to_monitor]:
                vendors_and_pages_to_monitor.append({"vendor": "Aruba", "url": aruba_series_url, "arista_page_type": None}) 
        
        for job_details in vendors_and_pages_to_monitor:
            schedule.every(UPDATE_CHECK_INTERVAL).hours.do(
                self.check_and_process_vendor_updates, 
                mother_url=job_details["url"],
                vendor=job_details["vendor"],
                # --- ADDITION: Pass arista_page_type from job_details ---
                arista_page_type=job_details.get("arista_page_type"), 
                force_reingest_discovered_docs=False 
            )
            # --- MODIFICATION: Log arista_page_type ---
            logger.info(f"Scheduled update check for {job_details['vendor']} (type: {job_details.get('arista_page_type','default')}) from {job_details['url']}")

        def run_scheduler_thread():
            logger.info("Update checker scheduler thread started.")
            if get_user_confirmation("Run all scheduled update checks once immediately now? (y/n)"):
                 logger.info("Running all scheduled update checks immediately...")
                 for job_details_startup in vendors_and_pages_to_monitor:
                     try:
                         # --- MODIFICATION: Log and pass arista_page_type ---
                         logger.info(f"Immediate run for: {job_details_startup['vendor']} (type: {job_details_startup.get('arista_page_type','default')}) - {job_details_startup['url']}")
                         self.check_and_process_vendor_updates(
                             mother_url=job_details_startup["url"], vendor=job_details_startup["vendor"],
                             # --- ADDITION: Pass arista_page_type from job_details_startup ---
                             arista_page_type=job_details_startup.get("arista_page_type"),
                             force_reingest_discovered_docs=force_reingest_all_updates_this_run
                         )
                     except Exception as e_immediate:
                         logger.error(f"Error during immediate check for {job_details_startup['vendor']} ({job_details_startup['url']}): {e_immediate}", exc_info=True)
                 logger.info("Finished immediate run of all scheduled update checks.")
            while self.running:
                schedule.run_pending()
                time.sleep(60) 
            logger.info("Update checker scheduler thread loop exited.")

        self.thread = threading.Thread(target=run_scheduler_thread, daemon=True)
        self.thread.start()
        logger.info("Update checker background thread launched.")

    def stop(self):
        if not self.running: logger.info("Update checker not running."); return
        logger.info("Attempting to stop update checker...")
        self.running = False
        if self.thread and self.thread.is_alive():
            logger.info("Waiting for update checker thread to join...")
            self.thread.join(timeout=10) 
            if self.thread.is_alive(): logger.warning("Update checker thread did not stop gracefully.")
            else: logger.info("Update checker thread stopped successfully.")
        else: logger.info("Update checker thread not active or already joined.")
        schedule.clear() 
        logger.info("Scheduled jobs cleared. Update checker fully stopped.")
