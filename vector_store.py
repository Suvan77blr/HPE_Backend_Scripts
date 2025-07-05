import chromadb
import logging
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from environment import CHROMA_PERSIST_DIRECTORY, EMBEDDING_MODEL
import os
import json
import datetime

logger = logging.getLogger(__name__)

class VectorStore:
    ALL_VENDOR_DOCS_COLLECTION_NAME = "all_vendor_docs"
    ERROR_CODES_COLLECTION_NAME = "error_codes"
    HACKER_NEWS_COLLECTION_NAME = "hacker_news_posts"

    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        # Ensure embedding_dimension matches your model's output
        # For 'sentence-transformers/all-mpnet-base-v2', it's 768.
        # If you change the model, update this or get it dynamically.
        try:
            self.embedding_dimension = self.embedding_model.get_sentence_embedding_dimension()
            logger.info(f"Dynamically determined embedding dimension: {self.embedding_dimension}")
        except Exception as e:
            logger.warning(f"Could not dynamically determine embedding dimension, defaulting to 768. Error: {e}")
            self.embedding_dimension = 768

        logger.info(f"Using embedding model: {EMBEDDING_MODEL} with dimension: {self.embedding_dimension}")
        self._store_embedding_info()
        self.collections = self._initialize_collections()
        self.ingested_files = self._load_ingestion_tracking() # Should be _load_tracking_data based on ingestion.py usage
        
        if not self.check_embedding_consistency():
            logger.warning("Embedding dimension mismatch detected. Consider resetting the database or using the original embedding model.")

    def _store_embedding_info(self):
        info_dir = os.path.join(CHROMA_PERSIST_DIRECTORY, "metadata")
        os.makedirs(info_dir, exist_ok=True)
        info_file = os.path.join(info_dir, "embedding_info.json")
        embedding_info = {
            "model_name": EMBEDDING_MODEL,
            "dimension": self.embedding_dimension,
            "last_updated": str(datetime.datetime.now())
        }
        with open(info_file, 'w') as f:
            json.dump(embedding_info, f, indent=2)

    def _initialize_collections(self):
        collections = {}
        
        # Single collection for all vendor documents
        collections[self.ALL_VENDOR_DOCS_COLLECTION_NAME] = self.client.get_or_create_collection(
            name=self.ALL_VENDOR_DOCS_COLLECTION_NAME,
            metadata={"type": "vendor_documentation", "embedding_dimension": self.embedding_dimension, "model": EMBEDDING_MODEL}
        )
        
        # Special collections
        collections[self.ERROR_CODES_COLLECTION_NAME] = self.client.get_or_create_collection(
            name=self.ERROR_CODES_COLLECTION_NAME,
            metadata={"type": "error_codes", "embedding_dimension": self.embedding_dimension, "model": EMBEDDING_MODEL}
        )
        
        collections[self.HACKER_NEWS_COLLECTION_NAME] = self.client.get_or_create_collection(
            name=self.HACKER_NEWS_COLLECTION_NAME,
            metadata={"type": "hacker_news", "embedding_dimension": self.embedding_dimension, "model": EMBEDDING_MODEL}
        )
        logger.info(f"Initialized collections: {list(collections.keys())}")
        return collections

    def _load_ingestion_tracking(self): # Renamed from _load_tracking_data to match original __init__
        tracking_file = os.path.join(CHROMA_PERSIST_DIRECTORY, "ingestion_tracking.json")
        if os.path.exists(tracking_file):
            try:
                with open(tracking_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.error(f"Error decoding JSON from tracking file: {tracking_file}. Starting with empty tracking.")
                return {}
        return {}

    def _save_tracking_data(self):
        tracking_file = os.path.join(CHROMA_PERSIST_DIRECTORY, "ingestion_tracking.json")
        os.makedirs(CHROMA_PERSIST_DIRECTORY, exist_ok=True)
        try:
            with open(tracking_file, 'w') as f:
                json.dump(self.ingested_files, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save tracking data to {tracking_file}: {e}")


    def check_embedding_consistency(self):
        for name, collection in self.collections.items():
            try:
                collection_metadata = collection.metadata # Corrected variable name
                if collection_metadata and 'embedding_dimension' in collection_metadata:
                    stored_dimension = collection_metadata['embedding_dimension']
                    if stored_dimension != self.embedding_dimension:
                        logger.warning(f"Collection {name} was created with embedding dimension {stored_dimension}, but current model uses {self.embedding_dimension}")
                        return False
            except Exception as e:
                logger.warning(f"Error checking collection {name} metadata: {str(e)}")
        return True

    def add_documents(self, collection_key: str, documents: List[Dict[str, Any]]):
        """
        Adds documents to the specified collection or to the main vendor collection.
        If adding to ALL_VENDOR_DOCS_COLLECTION_NAME, collection_key is treated as the vendor tag.
        """
        if not documents:
            logger.warning("No documents to add")
            return

        actual_collection_name = ""
        vendor_tag_for_metadata = collection_key.lower()

        if collection_key.lower() == self.ERROR_CODES_COLLECTION_NAME:
            actual_collection_name = self.ERROR_CODES_COLLECTION_NAME
            vendor_tag_for_metadata = "error_codes" # Or derive from doc metadata if appropriate
        elif collection_key.lower() == self.HACKER_NEWS_COLLECTION_NAME:
            actual_collection_name = self.HACKER_NEWS_COLLECTION_NAME
            vendor_tag_for_metadata = "hackernews" # Or derive
        else:
            # Assumes collection_key is a vendor tag (e.g., "cisco", "aruba")
            # and documents go into the single vendor collection.
            actual_collection_name = self.ALL_VENDOR_DOCS_COLLECTION_NAME
            # vendor_tag_for_metadata is already set to collection_key.lower()

        if actual_collection_name not in self.collections:
            logger.error(f"Target collection '{actual_collection_name}' (derived from key '{collection_key}') not found. Aborting add.")
            return
            
        collection = self.collections[actual_collection_name]

        ids, contents, metadatas_list = [], [], []
        for i, doc in enumerate(documents):
            # Ensure doc["content"] exists and is a string
            doc_content = doc.get("content")
            if not isinstance(doc_content, str):
                logger.warning(f"Document content is not a string or is missing, skipping: {doc.get('metadata', {}).get('title', 'Unknown title')}")
                continue

            # Generate a unique ID for the document chunk
            # Consider a more robust hashing or UUID if needed
            doc_id_base = doc.get("metadata", {}).get("url", "") + doc_content
            doc_id = str(hash(doc_id_base))[:16] + f"_{i}"
            
            ids.append(doc_id)
            contents.append(doc_content)
            
            # Prepare metadata
            doc_metadata = doc.get("metadata", {}).copy() # Start with existing metadata from the document
            doc_metadata["embedding_model"] = EMBEDDING_MODEL
            doc_metadata["embedding_dimension"] = self.embedding_dimension
            
            # Ensure the 'vendor' field in metadata is correctly set
            if actual_collection_name == self.ALL_VENDOR_DOCS_COLLECTION_NAME:
                # If 'vendor' is already in doc_metadata from DocumentProcessor, prioritize it.
                # Otherwise, use the vendor_tag_for_metadata derived from collection_key.
                doc_metadata["vendor"] = doc_metadata.get("vendor", vendor_tag_for_metadata).lower()
            else: # For special collections like error_codes, hacker_news
                doc_metadata["vendor"] = vendor_tag_for_metadata # e.g., "error_codes", "hackernews"

            # Standardize other metadata fields
            doc_metadata["product_line"] = doc_metadata.get("product_line", "Unknown")
            doc_metadata["release"] = doc_metadata.get("release", "Unknown")
            # Ensure features, categories, deployment are strings if they exist, or empty strings
            doc_metadata["features"] = ",".join(doc_metadata.get("features", [])) if isinstance(doc_metadata.get("features"), list) else doc_metadata.get("features", "")
            doc_metadata["categories"] = ",".join(doc_metadata.get("categories", [])) if isinstance(doc_metadata.get("categories"), list) else doc_metadata.get("categories", "")
            doc_metadata["deployment"] = ",".join(doc_metadata.get("deployment", [])) if isinstance(doc_metadata.get("deployment"), list) else doc_metadata.get("deployment", "")
            
            metadatas_list.append(doc_metadata)

        if not contents:
            logger.warning("No valid documents with content to process after filtering.")
            return

        embeddings = self.embedding_model.encode(contents).tolist()
        
        try:
            collection.add(
                ids=ids,
                documents=contents,
                metadatas=metadatas_list,
                embeddings=embeddings
            )
            logger.info(f"Added {len(documents)} documents to collection '{collection.name}' (derived from key '{collection_key}').")
        except Exception as e:
            logger.error(f"Error adding documents to collection '{collection.name}': {str(e)}")


    

    def query(self, collection_name: str, query_text: str, n_results: int = 3, where_filter: Optional[Dict[str, Any]] = None):
        """
        Queries a specific collection with an optional metadata filter.
        """
        if collection_name not in self.collections:
            logger.error(f"Collection '{collection_name}' not found for querying.")
            return {"documents": [[]], "metadatas": [[]], "distances": [[]], "ids": [[]]}

        collection = self.collections[collection_name]
        embedding = self.embedding_model.encode(query_text).tolist()
        
        try:
            query_params = {
                "query_embeddings": [embedding],
                "n_results": n_results
            }
            if where_filter:
                query_params["where"] = where_filter
            
            return collection.query(**query_params)
        except Exception as e:
            logger.error(f"Error querying collection '{collection_name}': {str(e)}")
            return {"documents": [[]], "metadatas": [[]], "distances": [[]], "ids": [[]]}

    def query_all_collections(self, query_text: str, n_results: int = 3):
        """
        Queries all initialized collections.
        """
        embedding = self.embedding_model.encode(query_text).tolist()
        all_results = {}
        
        for name, collection in self.collections.items():
            try:
                results = collection.query(
                    query_embeddings=[embedding],
                    n_results=n_results
                )
                all_results[name] = results
            except Exception as e:
                logger.error(f"Error querying collection {name}: {str(e)}")
                all_results[name] = {"documents": [[]], "metadatas": [[]], "distances": [[]], "ids": [[]]}
        return all_results
        
    def clear_collection(self, collection_name: str):
        if collection_name in self.collections:
            try:
                # Fetch all IDs in the collection to delete them
                # ChromaDB's delete method requires IDs. If no IDs, it means empty or error.
                # A more robust way if collection.get() is too slow for large collections
                # would be to delete the collection and recreate it, but that loses metadata.
                # For now, this is standard for clearing.
                existing_items = self.collections[collection_name].get(limit=10000) # Adjust limit as needed, or paginate
                if existing_items and existing_items['ids']:
                    self.collections[collection_name].delete(ids=existing_items['ids'])
                    logger.info(f"Cleared collection: {collection_name}")
                else:
                    logger.info(f"Collection {collection_name} is already empty or IDs could not be fetched.")
            except Exception as e:
                logger.error(f"Error clearing collection {collection_name}: {str(e)}")
        else:
            logger.warning(f"Collection {collection_name} not found for clearing.")

    def reset_database(self):
        logger.info("Resetting database...")
        collection_names_to_delete = list(self.client.list_collections()) # Get all actual collections
        for collection_obj in collection_names_to_delete:
            try:
                self.client.delete_collection(name=collection_obj.name)
                logger.info(f"Deleted collection: {collection_obj.name}")
            except Exception as e:
                logger.error(f"Error deleting collection {collection_obj.name} during reset: {e}")
        
        self.collections = self._initialize_collections() # Re-initialize
        self.ingested_files = {}
        self._save_tracking_data()
        logger.info("All collections have been cleared and re-initialized, and ingestion tracking reset.")





















