import logging
import json
import os
import datetime
from scraper import NetworkDocScraper
from document_processor import DocumentProcessor
from vector_store import VectorStore # Make sure this import is correct

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataIngestionPipeline:
    def __init__(self):
        self.scraper = NetworkDocScraper()
        self.processor = DocumentProcessor()
        self.vector_store = VectorStore()
        
        self.tracking_dir = os.path.join(os.getcwd(), "tracking")
        os.makedirs(self.tracking_dir, exist_ok=True)
        self.tracking_file = os.path.join(self.tracking_dir, "ingested_files.json")
        
        self._load_tracking_data() # Load existing tracking data

    def _load_tracking_data(self):
        """Load tracking data from file."""
        if os.path.exists(self.tracking_file):
            try:
                with open(self.tracking_file, 'r', encoding='utf-8') as f:
                    self.ingested_files = json.load(f)
                    logger.info(f"Loaded {len(self.ingested_files)} ingested file records from {self.tracking_file}")
            except json.JSONDecodeError:
                logger.error(f"Error decoding JSON from tracking file: {self.tracking_file}. Starting with empty tracking.")
                self.ingested_files = {}
            except Exception as e:
                logger.error(f"Unexpected error loading tracking file {self.tracking_file}: {e}. Starting with empty tracking.")
                self.ingested_files = {}
        else:
            self.ingested_files = {}
            logger.info(f"Tracking file {self.tracking_file} not found. Starting with empty tracking.")

    def _save_tracking_data(self): # Renamed to be public if UpdateChecker needs it, or keep protected if only used internally
        """Save tracking data to file"""
        os.makedirs(self.tracking_dir, exist_ok=True)
        try:
            with open(self.tracking_file, 'w', encoding='utf-8') as f:
                json.dump(self.ingested_files, f, indent=2)
            logger.debug(f"Saved ingestion tracking data to {self.tracking_file}")
        except Exception as e:
            logger.error(f"Failed to save tracking data to {self.tracking_file}: {e}")

    def save_tracking_data(self): # Public alias if needed by UpdateChecker
        self._save_tracking_data()

    def ingest_json_file(self, file_path: str, vendor_name: str, force_reingest=False):
        """
        Ingests data from a specified JSON file for a given vendor.
        The JSON file can be a single large object or a list of objects.
        Each object (or the single object) will be treated as a document to be chunked.
        """
        logger.info(f"Starting ingestion of JSON file: {file_path} for vendor: {vendor_name}")
        
        if not os.path.exists(file_path):
            logger.error(f"JSON file not found: {file_path}. Aborting ingestion for this file.")
            return

        file_stat = os.stat(file_path)
        # Key includes vendor, basename, and mod time to uniquely identify this version of the file
        file_key = f"json_ingest_{vendor_name.lower()}_{os.path.basename(file_path)}_{file_stat.st_mtime}"

        if not force_reingest and file_key in self.ingested_files:
            logger.info(f"File {file_path} for vendor {vendor_name} (key: {file_key}) was already ingested and force_reingest is False. Skipping.")
            return
        elif force_reingest and file_key in self.ingested_files:
            logger.info(f"Force re-ingesting file {file_path} for vendor {vendor_name} (key: {file_key}).")


        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from file {file_path}: {e}")
            return
        except Exception as e:
            logger.error(f"An unexpected error occurred while reading {file_path}: {e}")
            return

        documents_to_process = []
        if isinstance(data, list):
            documents_to_process = data
        elif isinstance(data, dict):
            documents_to_process = [data] # Treat a single dict as a list with one item
        else:
            logger.error(f"Unsupported JSON structure in {file_path}. Expected a list of objects or a single object.")
            return

        total_chunks_added_for_file = 0
        processed_item_count = 0
        batch_size = 50 # Process items in batches to manage memory for very large JSON arrays
        
        for batch_start_idx in range(0, len(documents_to_process), batch_size):
            batch_items = documents_to_process[batch_start_idx : batch_start_idx + batch_size]
            all_chunks_for_batch = []

            for item_index_in_batch, item in enumerate(batch_items):
                actual_item_index = batch_start_idx + item_index_in_batch
                logger.debug(f"Processing item {actual_item_index + 1}/{len(documents_to_process)} from {file_path}")
                try:
                    content_str = json.dumps(item, indent=2, ensure_ascii=False)
                    
                    item_metadata = {
                        "vendor": vendor_name.lower(),
                        "source_file": os.path.basename(file_path),
                        "original_item_index": actual_item_index, # For traceability
                        "title": item.get("title", f"Item {actual_item_index + 1} from {os.path.basename(file_path)}"),
                        "url": item.get("url", item.get("link", f"json://{os.path.basename(file_path)}/item_{actual_item_index}")), # Use URL if present
                        "doc_type": "json_import"
                    }
                    
                    # Attempt to extract more specific metadata if fields exist
                    if isinstance(item.get("metadata"), dict): # If item has its own metadata field
                        item_metadata.update(item["metadata"])
                        item_metadata["vendor"] = vendor_name.lower() # Ensure file-level vendor overrides

                    product_info = self.processor.extract_product_info(content_str)
                    item_metadata.update(product_info)
                    item_metadata["vendor"] = vendor_name.lower() # Re-affirm vendor

                    item_chunks = self.processor.chunk_document(content_str, item_metadata)
                    
                    if item_chunks:
                        all_chunks_for_batch.extend(item_chunks)
                    else:
                        logger.warning(f"No chunks generated for item {actual_item_index + 1} from {file_path}.")
                    processed_item_count += 1
                except Exception as e_item:
                    logger.error(f"Error processing item {actual_item_index + 1} from JSON file {file_path}: {e_item}", exc_info=True)
            
            if all_chunks_for_batch:
                self.vector_store.add_documents(vendor_name.lower(), all_chunks_for_batch)
                total_chunks_added_for_file += len(all_chunks_for_batch)
                logger.info(f"Added batch of {len(all_chunks_for_batch)} chunks for {vendor_name} from {file_path}. Total for file so far: {total_chunks_added_for_file}")

        if processed_item_count > 0:
            self.ingested_files[file_key] = {
                "timestamp": datetime.datetime.now().isoformat(),
                "items_processed_in_file": processed_item_count,
                "total_chunks_added_from_file": total_chunks_added_for_file,
                "source_file_path": file_path # Store original path for clarity
            }
            self._save_tracking_data()
            logger.info(f"Completed ingestion of JSON file: {file_path} for vendor: {vendor_name}. Processed {processed_item_count} items, added {total_chunks_added_for_file} chunks.")
        elif len(documents_to_process) > 0: # If there were items but none processed successfully
             logger.warning(f"No items were successfully processed from JSON file: {file_path} for vendor: {vendor_name}.")
        else: # If the JSON file was empty or had invalid top-level structure
            logger.info(f"JSON file {file_path} for vendor {vendor_name} was empty or had no processable items.")



    
    def _process_document_list(self, document_links: list[dict], vendor: str, doc_type_prefix: str, force_reingest=False):
        """Helper function to process a list of scraped document metadata."""
        processed_count = 0
        total_links = len(document_links)
        for i, doc_meta in enumerate(document_links):
            logger.debug(f"Processing link {i+1}/{total_links} for {vendor} {doc_type_prefix}: {doc_meta.get('title', doc_meta.get('url'))}")
            doc_url = doc_meta.get("url")
            if not doc_url:
                logger.warning(f"Skipping document with no URL: {doc_meta}")
                continue

            doc_key = f"{doc_type_prefix}_{vendor.lower()}_{doc_url}"
            
            if not force_reingest and doc_key in self.ingested_files:
                logger.info(f"Document already ingested (key: {doc_key}): '{doc_meta.get('title', doc_url)}'. Skipping.")
                continue
            elif force_reingest and doc_key in self.ingested_files:
                 logger.info(f"Force re-ingesting document (key: {doc_key}): '{doc_meta.get('title', doc_url)}'.")


            try:
                content = self.scraper.extract_document_content(doc_meta) # Pass full doc_meta
                if not content:
                    logger.warning(f"No content extracted from {doc_url}")
                    continue
                
                # Prepare metadata
                metadata_for_processor = doc_meta.copy() # Start with metadata from scraper
                metadata_for_processor['vendor'] = vendor.lower() # Ensure vendor is set

                product_info = self.processor.extract_product_info(content)
                final_metadata = {**metadata_for_processor, **product_info}
                final_metadata['vendor'] = vendor.lower() # Re-affirm vendor

                chunks = self.processor.chunk_document(content, final_metadata)
                
                if chunks:
                    self.vector_store.add_documents(vendor.lower(), chunks)
                    self.ingested_files[doc_key] = {
                        "timestamp": datetime.datetime.now().isoformat(),
                        "title": doc_meta.get("title", "N/A"),
                        "url": doc_url,
                        "chunks": len(chunks)
                    }
                    self._save_tracking_data()
                    processed_count += 1
                    logger.info(f"Processed {processed_count}/{total_links} for {vendor} {doc_type_prefix}: '{doc_meta.get('title', doc_url)}'")
                else:
                    logger.warning(f"No chunks generated for document: '{doc_meta.get('title', doc_url)}'")
            except Exception as e:
                logger.error(f"Error processing document '{doc_meta.get('title', doc_url)}' ({doc_url}): {e}", exc_info=True)
        
        logger.info(f"Completed ingestion of {processed_count} documents for {vendor} {doc_type_prefix}.")
        return processed_count



    

    def ingest_cisco_release_notes(self, url: str, links: list[dict] = None, force_reingest=False):
        logger.info(f"Starting ingestion of Cisco release notes from {url}")
        if links is None:
            # Assuming scraper has a method like parse_cisco_main_page_for_release_notes
            document_links = self.scraper.parse_cisco_release_notes(url) # Or specific parser
        else:
            document_links = links
        
        if not document_links:
            logger.info(f"No Cisco release note links found or provided for {url}.")
            return
        
        self._process_document_list(document_links, "Cisco", "cisco_release_notes", force_reingest)


    def ingest_cisco_config_guides(self, url: str, links: list[dict] = None, force_reingest=False):
        logger.info(f"Starting ingestion of Cisco configuration guides from {url}")
        if links is None:
            document_links = self.scraper.parse_cisco_config_guides(url) # Or specific parser
        else:
            document_links = links

        if not document_links:
            logger.info(f"No Cisco config guide links found or provided for {url}.")
            return

        self._process_document_list(document_links, "Cisco", "cisco_config_guides", force_reingest)

    def ingest_scraped_data(self, json_file_path: str, force_reingest=False):
        """Ingest data from a generic JSON file, typically for error codes."""
        if not os.path.exists(json_file_path):
            logger.error(f"Error code JSON file not found: {json_file_path}. Aborting.")
            return

        file_stat = os.stat(json_file_path)
        file_key = f"error_codes_json_{os.path.basename(json_file_path)}_{file_stat.st_mtime}"

        if not force_reingest and file_key in self.ingested_files:
            logger.info(f"Error codes file {json_file_path} (key: {file_key}) was already ingested. Skipping.")
            return
        elif force_reingest and file_key in self.ingested_files:
            logger.info(f"Force re-ingesting error codes file {json_file_path} (key: {file_key}).")

        logger.info(f"Ingesting error codes from {json_file_path}")
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            content_to_chunk = ""
            if isinstance(data, list):
                content_to_chunk = "\n\n".join([json.dumps(item, ensure_ascii=False) for item in data])
            elif isinstance(data, dict):
                content_to_chunk = json.dumps(data, ensure_ascii=False)
            else:
                logger.warning(f"Unsupported data type in {json_file_path} for error codes. Expected list or dict.")
                return

            metadata = {
                "doc_type": "Error Documentation",
                "source_file": os.path.basename(json_file_path),
                "url": f"file://{os.path.abspath(json_file_path)}" # Represent as a file URL
            }
            
            error_chunks = self.processor.chunk_document(content_to_chunk, metadata)
            if error_chunks:
                # Add to the specific error_codes collection in VectorStore
                self.vector_store.add_documents(VectorStore.ERROR_CODES_COLLECTION_NAME, error_chunks)
                self.ingested_files[file_key] = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "chunks": len(error_chunks),
                    "source_file_path": json_file_path
                }
                self._save_tracking_data()
                logger.info(f"Ingested {len(error_chunks)} chunks from {json_file_path} into '{VectorStore.ERROR_CODES_COLLECTION_NAME}'.")
            else:
                logger.info(f"No chunks generated from error codes file {json_file_path}.")
        except Exception as e:
            logger.error(f"Error ingesting error codes from {json_file_path}: {e}", exc_info=True)

    def ingest_aruba_documentation(self, mother_url: str, links: list[dict] = None, force_reingest=False, force_scrape_mother_page=False):
        """
        Ingest Aruba AOS-CX documentation.
        Can take a mother_url to scrape or a pre-fetched list of links.
        """
        logger.info(f"Starting ingestion of Aruba documentation. Mother URL: {mother_url}, Pre-fetched links provided: {links is not None}")
        
        # Key for tracking if the mother_url itself has been processed for link discovery
        # This is different from individual document tracking keys.
        # We add force_scrape_mother_page to allow re-discovery of links even if this key exists.
        mother_url_discovery_key = f"aruba_mother_page_discovery_{mother_url}"
        
        document_links = []
        if links is not None:
            document_links = links
            logger.info(f"Using {len(document_links)} pre-fetched links for Aruba.")
        elif force_scrape_mother_page or mother_url_discovery_key not in self.ingested_files:
            logger.info(f"Scraping Aruba mother page for document links: {mother_url}")
            # Assuming scraper.parse_aruba_documentation_mother_page is defined and works
            main_page_links = self.scraper.parse_aruba_documentation_mother_page(mother_url)
            # Assuming scraper.scrape_dropdown_options for Aruba specific dropdowns if applicable
            # dropdown_links = self.scraper.scrape_dropdown_options(mother_url) # Example
            document_links.extend(main_page_links)
            # document_links.extend(dropdown_links) # Example
            
            if document_links:
                self.ingested_files[mother_url_discovery_key] = { # Track that we've discovered links from this mother page
                    "timestamp": datetime.datetime.now().isoformat(),
                    "links_found_count": len(document_links)
                }
                self._save_tracking_data()
            else:
                logger.warning(f"No document links found on Aruba mother page: {mother_url}")
                return # Nothing to process further if no links found
        else:
            logger.info(f"Aruba mother page {mother_url} link discovery already processed (key: {mother_url_discovery_key}). To re-scrape, use force_scrape_mother_page=True. No new links will be discovered now.")
            # If we are not re-scraping the mother page, and no links were passed, there's nothing to do.
            # If you wanted to process previously discovered links, you'd need to load them from somewhere.
            # For now, this means if mother page isn't re-scraped and no links passed, we exit.
            return


        if not document_links:
            logger.info(f"No Aruba document links to process for {mother_url}.")
            return
        
        processed_doc_count = self._process_document_list(document_links, "Aruba", "aruba_doc", force_reingest)
        
        # Optionally save the raw scraped content if needed (as in your original stub)
        # This part should be conditional and perhaps save to a structured location
        # For example, if you are collecting raw data for other purposes.
        # documents_data = [] # Logic to collect raw content if needed
        # json_raw_content_path = 'aruba_documents_raw_content.json'
        # try:
        #     with open(json_raw_content_path, 'w', encoding='utf-8') as json_file:
        #         json.dump(documents_data, json_file, ensure_ascii=False, indent=4)
        #     logger.info(f"Saved raw Aruba content to {json_raw_content_path}")
        # except Exception as e:
        #     logger.error(f"Error saving raw Aruba content: {e}")

        logger.info(f"Completed Aruba documentation ingestion from {mother_url}. Processed {processed_doc_count} individual documents.")


    def run_full_web_ingestion(self, force_reingest_all_docs=False, force_scrape_all_mother_pages=False):
        """Run the complete web scraping and ingestion pipeline for all configured vendors."""
        logger.info(f"Starting full WEB ingestion pipeline. Force re-ingest all docs: {force_reingest_all_docs}, Force re-scrape all mother pages: {force_scrape_all_mother_pages}")
        
        # Cisco
        self.ingest_cisco_release_notes(
            "https://www.cisco.com/c/en/us/support/switches/nexus-9000-series-switches/products-release-notes-list.html",
            force_reingest=force_reingest_all_docs # Links will be re-scraped by default inside the method
        )
        self.ingest_cisco_config_guides(
            "https://www.cisco.com/c/en/us/support/switches/nexus-9000-series-switches/products-installation-and-configuration-guides-list.html",
            force_reingest=force_reingest_all_docs
        )
        
        # Aruba
        self.ingest_aruba_documentation(
            "https://arubanetworking.hpe.com/techdocs/AOS-CX/help_portal/Content/ArubaTopics/Switches/pdfs.htm",
            force_reingest=force_reingest_all_docs,
            force_scrape_mother_page=force_scrape_all_mother_pages
        )
        
        # Note: `ingest_scraped_data` (for local error_codes.json) is separate from web ingestion.
        # Call it explicitly in main.py if needed.
        
        logger.info("Full WEB ingestion pipeline completed.")

    def reset_ingestion_tracking(self):
        self.ingested_files = {}
        self._save_tracking_data()
        logger.info("Ingestion tracking has been reset.")
















