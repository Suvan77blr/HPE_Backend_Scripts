#!/usr/bin/env python3
"""
R1-RAG MCP Server

Exposes the Network Documentation RAG system capabilities through the Model Context Protocol.
This allows AI assistants to access and query your network documentation database.
"""

import asyncio
import json
import logging
import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Sequence
from pathlib import Path

# Add stderr debugging as suggested by the error message
print("Starting R1-RAG MCP Server...", file=sys.stderr)

try:
    import mcp.server.stdio
    import mcp.types as types
    from mcp.server import NotificationOptions, Server
    from mcp.server.models import InitializationOptions
    print("MCP imports successful", file=sys.stderr)
except ImportError as e:
    print(f"MCP import error: {e}", file=sys.stderr)
    sys.exit(1)

# Import environment only - defer all other heavy imports until needed
try:
    from environment import *
    print("Environment variables imported successfully", file=sys.stderr)
except ImportError as e:
    print(f"Environment import error: {e}", file=sys.stderr)
    sys.exit(1)

# Lazy import helpers - imports happen only when components are first used
def lazy_import_vector_store():
def lazy_import_agent():
def lazy_import_web_searcher():
def lazy_import_scraper():
def lazy_import_ingestion():
def lazy_import_document_processor():
def lazy_import_llm_service():

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("r1-rag-mcp")

# Define collection names locally to avoid importing VectorStore during startup
ALL_VENDOR_DOCS_COLLECTION_NAME = "all_vendor_docs"
ERROR_CODES_COLLECTION_NAME = "error_codes"
HACKER_NEWS_COLLECTION_NAME = "hacker_news_posts"

class R1RAGMCPServer:
    """MCP Server wrapper for R1-RAG functionality"""
    
    def __init__(self):
        self.vector_store = None
        self.agent = None
        self.web_searcher = None
        self.scraper = None
        self.ingestion_pipeline = None
        self.doc_processor = None
        self.llm_service = None
        self.initialization_errors = []
        self._components_initialized = False
        self._initialization_in_progress = False
        self._initialization_start_time = None
        self._initialization_task = None
        self._extracted_docs = None  # Cache for extracted documents
        self._vector_store_loading = False
        self._vector_store_ready = False
        
    def get_initialization_status(self):
        """Get current initialization status"""
        if self._components_initialized:
            return "completed"
        elif self._initialization_in_progress:
            elapsed = time.time() - self._initialization_start_time if self._initialization_start_time else 0
            return f"in_progress ({elapsed:.1f}s elapsed)"
        else:
            return "not_started"
            
    async def start_background_initialization(self):
        """Start initialization in background without blocking"""
        if self._components_initialized or self._initialization_in_progress:
            return
            
        self._initialization_task = asyncio.create_task(self._background_init())
        
    async def _background_init(self):
        """Background initialization task"""
        try:
            await self.initialize_components()
        except Exception as e:
            logger.error(f"Background initialization failed: {e}")
            
    async def ensure_components_initialized(self, timeout_seconds=30):
        """Ensure components are initialized with timeout protection"""
        if self._components_initialized:
            return True
            
        if not self._initialization_in_progress:
            await self.start_background_initialization()
            
        # Wait for initialization with timeout
        start_time = time.time()
        while self._initialization_in_progress and (time.time() - start_time) < timeout_seconds:
            await asyncio.sleep(0.1)
            
        return self._components_initialized
        
    async def initialize_components(self):
        """Initialize R1-RAG components with better error handling"""
        if self._components_initialized or self._initialization_in_progress:
            return
            
        self._initialization_in_progress = True
        self._initialization_start_time = time.time()
        print("Starting component initialization...", file=sys.stderr)
        
        # Initialize vector store first - it's the most critical component
        try:
            print("Initializing VectorStore...", file=sys.stderr)
            VectorStore = lazy_import_vector_store()
            self.vector_store = VectorStore()
            print("VectorStore initialized successfully", file=sys.stderr)
        except Exception as e:
            error_msg = f"Failed to initialize VectorStore: {e}"
            print(error_msg, file=sys.stderr)
            self.initialization_errors.append(error_msg)
            
        # Initialize lightweight components only
        try:
            print("Initializing LLMService...", file=sys.stderr)
            LLMService = lazy_import_llm_service()
            self.llm_service = LLMService()
            print("LLMService initialized successfully", file=sys.stderr)
        except Exception as e:
            error_msg = f"Failed to initialize LLMService: {e}"
            print(error_msg, file=sys.stderr)
            self.initialization_errors.append(error_msg)
            
        try:
            print("Initializing WebSearcher...", file=sys.stderr)
            WebSearcher = lazy_import_web_searcher()
            self.web_searcher = WebSearcher()
            print("WebSearcher initialized successfully", file=sys.stderr)
        except Exception as e:
            error_msg = f"Failed to initialize WebSearcher: {e}"
            print(error_msg, file=sys.stderr)
            self.initialization_errors.append(error_msg)
            
        # Initialize agent (requires vector_store, llm_service, web_searcher)
        try:
            if self.vector_store and self.llm_service and self.web_searcher:
                print("Initializing NetworkIntegrationAgent...", file=sys.stderr)
                NetworkIntegrationAgent = lazy_import_agent()
                self.agent = NetworkIntegrationAgent(
                    self.vector_store, 
                    self.llm_service,
                    self.web_searcher
                )
                print("NetworkIntegrationAgent initialized successfully", file=sys.stderr)
            else:
                error_msg = "Cannot initialize NetworkIntegrationAgent: missing dependencies"
                print(error_msg, file=sys.stderr)
                self.initialization_errors.append(error_msg)
        except Exception as e:
            error_msg = f"Failed to initialize NetworkIntegrationAgent: {e}"
            print(error_msg, file=sys.stderr)
            self.initialization_errors.append(error_msg)
            
        # Skip heavy components (scraper, ingestion) to speed up initialization
        # They'll be initialized on-demand when needed
        
        self._components_initialized = True
        self._initialization_in_progress = False
        elapsed = time.time() - self._initialization_start_time
            
        if self.initialization_errors:
            print(f"Core initialization completed with {len(self.initialization_errors)} errors in {elapsed:.1f}s", file=sys.stderr)
            for error in self.initialization_errors:
                print(f"  - {error}", file=sys.stderr)
        else:
            print(f"Core R1-RAG components initialized successfully in {elapsed:.1f}s", file=sys.stderr)
            
    async def ensure_heavy_components(self):
        """Initialize heavy components only when needed"""
        if not self.scraper:
            try:
                print("Initializing NetworkDocScraper on-demand...", file=sys.stderr)
                NetworkDocScraper = lazy_import_scraper()
                self.scraper = NetworkDocScraper()
                print("NetworkDocScraper initialized successfully", file=sys.stderr)
            except Exception as e:
                error_msg = f"Failed to initialize NetworkDocScraper: {e}"
                print(error_msg, file=sys.stderr)
                self.initialization_errors.append(error_msg)
                
        if not self.ingestion_pipeline:
            try:
                print("Initializing DataIngestionPipeline on-demand...", file=sys.stderr)
                DataIngestionPipeline = lazy_import_ingestion()
                self.ingestion_pipeline = DataIngestionPipeline()
                print("DataIngestionPipeline initialized successfully", file=sys.stderr)
            except Exception as e:
                error_msg = f"Failed to initialize DataIngestionPipeline: {e}"
                print(error_msg, file=sys.stderr)
                self.initialization_errors.append(error_msg)
                
        if not self.doc_processor:
            try:
                print("Initializing DocumentProcessor on-demand...", file=sys.stderr)
                DocumentProcessor = lazy_import_document_processor()
                self.doc_processor = DocumentProcessor()
                print("DocumentProcessor initialized successfully", file=sys.stderr)
            except Exception as e:
                error_msg = f"Failed to initialize DocumentProcessor: {e}"
                print(error_msg, file=sys.stderr)
                self.initialization_errors.append(error_msg)

    def load_extracted_documents(self):
        """Load documents from extracted JSON file as fallback"""
        if self._extracted_docs is not None:
            return self._extracted_docs
            
        try:
            if os.path.exists("extracted_documents.json"):
                print("Loading documents from extracted JSON file...", file=sys.stderr)
                with open("extracted_documents.json", 'r', encoding='utf-8') as f:
                    self._extracted_docs = json.load(f)
                doc_count = len(self._extracted_docs.get('all_vendor_docs', []))
                print(f"Loaded {doc_count} documents from backup", file=sys.stderr)
                return self._extracted_docs
            else:
                print("No extracted_documents.json file found", file=sys.stderr)
        except Exception as e:
            print(f"Failed to load extracted documents: {e}", file=sys.stderr)
        
        return None
    
    def search_extracted_documents(self, query_text, collection="all_vendor_docs", n_results=5):
        """Search extracted documents using simple text matching"""
        extracted_docs = self.load_extracted_documents()
        if not extracted_docs or collection not in extracted_docs:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]], "ids": [[]]}
        
        documents = extracted_docs[collection]
        query_lower = query_text.lower()
        
        # Simple keyword matching
        matches = []
        for doc in documents:
            doc_text = doc.get('document', '').lower()
            metadata = doc.get('metadata', {})
            
            # Score based on keyword matches
            score = 0
            query_words = query_lower.split()
            for word in query_words:
                if word in doc_text:
                    score += doc_text.count(word)
            
            if score > 0:
                matches.append({
                    'document': doc.get('document', ''),
                    'metadata': metadata,
                    'id': doc.get('id', ''),
                    'score': score
                })
        
        # Sort by score and limit results
        matches.sort(key=lambda x: x['score'], reverse=True)
        matches = matches[:n_results]
        
        # Format like ChromaDB results
        result_docs = [match['document'] for match in matches]
        result_metadata = [match['metadata'] for match in matches]
        result_ids = [match['id'] for match in matches]
        result_distances = [1.0 - (match['score'] / 100) for match in matches]  # Fake distances
        
        return {
            "documents": [result_docs],
            "metadatas": [result_metadata], 
            "distances": [result_distances],
            "ids": [result_ids],
            "fallback_used": True,
            "total_documents_searched": len(documents)
        }

    async def ultra_fast_init(self):
        """Ultra-fast initialization - only core services, no heavy ML models"""
        if self._components_initialized or self._initialization_in_progress:
            return
            
        self._initialization_in_progress = True
        self._initialization_start_time = time.time()
        print("Starting ultra-fast initialization (core services only)...", file=sys.stderr)
        
        # Only initialize absolutely essential services
        try:
            print("Initializing WebSearcher...", file=sys.stderr)
            WebSearcher = lazy_import_web_searcher()
            self.web_searcher = WebSearcher()
            print("WebSearcher initialized successfully", file=sys.stderr)
        except Exception as e:
            error_msg = f"Failed to initialize WebSearcher: {e}"
            print(error_msg, file=sys.stderr)
            self.initialization_errors.append(error_msg)
            
        try:
            print("Initializing LLMService...", file=sys.stderr)
            LLMService = lazy_import_llm_service()
            self.llm_service = LLMService()
            print("LLMService initialized successfully", file=sys.stderr)
        except Exception as e:
            error_msg = f"Failed to initialize LLMService: {e}"
            print(error_msg, file=sys.stderr)
            self.initialization_errors.append(error_msg)
        
        # Skip vector store and agent initialization to avoid ML model loading
        
        self._components_initialized = True
        self._initialization_in_progress = False
        elapsed = time.time() - self._initialization_start_time
            
        print(f"Ultra-fast initialization completed in {elapsed:.1f}s", file=sys.stderr)
        
        # Only start background vector store loading if explicitly requested
        # This avoids timeouts during MCP initialization
        
    async def delayed_initialization(self):
        """Initialize components AFTER MCP handshake to avoid timeout"""
        try:
            # Wait a moment to ensure MCP handshake completes
            await asyncio.sleep(1)
            print("Starting delayed ultra-fast initialization...", file=sys.stderr)
            await self.ultra_fast_init()
            
            # Now start vector store loading in background
            print("Starting background vector store loading...", file=sys.stderr)
            await self.start_background_vector_loading()
            
            print("Delayed initialization completed", file=sys.stderr)
        except Exception as e:
            print(f"Delayed initialization failed: {e}", file=sys.stderr)
            self.initialization_errors.append(f"Delayed initialization failed: {e}")
        
    async def ensure_vector_store(self):
        """Initialize vector store only when needed"""
        if self.vector_store:
            return True
            
        try:
            print("Initializing VectorStore on-demand...", file=sys.stderr)
            VectorStore = lazy_import_vector_store()
            self.vector_store = VectorStore()
            print("VectorStore initialized successfully", file=sys.stderr)
            
            # Also initialize agent if we have all dependencies
            if self.llm_service and self.web_searcher:
                try:
                    print("Initializing NetworkIntegrationAgent...", file=sys.stderr)
                    NetworkIntegrationAgent = lazy_import_agent()
                    self.agent = NetworkIntegrationAgent(
                        self.vector_store, 
                        self.llm_service,
                        self.web_searcher
                    )
                    print("NetworkIntegrationAgent initialized successfully", file=sys.stderr)
                except Exception as e:
                    print(f"Failed to initialize NetworkIntegrationAgent: {e}", file=sys.stderr)
            
            return True
        except Exception as e:
            print(f"Failed to initialize VectorStore: {e}", file=sys.stderr)
            return False

    async def start_background_vector_loading(self):
        """Start vector store loading in background immediately"""
        if self._vector_store_loading or self._vector_store_ready or self.vector_store:
            return
            
        self._vector_store_loading = True
        print("Starting background vector store initialization...", file=sys.stderr)
        
        # Create a background task for vector store loading
        asyncio.create_task(self._load_vector_store_background())
        
    async def _load_vector_store_background(self):
        """Background task to load vector store"""
        try:
            print("Loading VectorStore in background...", file=sys.stderr)
            VectorStore = lazy_import_vector_store()
            self.vector_store = VectorStore()
            print("VectorStore loaded successfully in background", file=sys.stderr)
            
            # Initialize agent if we have all dependencies
            if self.llm_service and self.web_searcher:
                try:
                    print("Initializing NetworkIntegrationAgent in background...", file=sys.stderr)
                    NetworkIntegrationAgent = lazy_import_agent()
                    self.agent = NetworkIntegrationAgent(
                        self.vector_store, 
                        self.llm_service,
                        self.web_searcher
                    )
                    print("NetworkIntegrationAgent loaded successfully in background", file=sys.stderr)
                except Exception as e:
                    print(f"Failed to initialize NetworkIntegrationAgent in background: {e}", file=sys.stderr)
                    
            self._vector_store_ready = True
            self._vector_store_loading = False
            
        except Exception as e:
            print(f"Background vector store loading failed: {e}", file=sys.stderr)
            self._vector_store_loading = False
            # Don't set ready to True, so fallback will be used

# Create the MCP server
server = Server("r1-rag")
r1_rag = R1RAGMCPServer()

@server.list_resources()
async def handle_list_resources() -> list[types.Resource]:
    """List available resources (document collections)"""
    resources = []
    
    try:
        # Return basic resources without initializing components for faster startup
        # Components will be initialized on-demand when tools are called
        
        # Add basic collection resources
        collections = [
            "all_vendor_docs",
            "error_codes", 
            "hacker_news_posts"
        ]
        
        for collection in collections:
            resources.append(types.Resource(
                uri=f"r1-rag://collections/{collection}",
                name=f"Collection: {collection}",
                description=f"Network documentation collection (initialize on first use)",
                mimeType="application/json"
            ))
        
        # Add vendor URL collections as resources
        url_files = ["arubalinks.txt", "aristalinks.txt", "juniperlinks.txt"]
        for url_file in url_files:
            if Path(url_file).exists():
                vendor = url_file.replace("links.txt", "").capitalize()
                resources.append(types.Resource(
                    uri=f"r1-rag://urls/{vendor.lower()}",
                    name=f"{vendor} URL Collection", 
                    description=f"List of {vendor} documentation URLs for scraping",
                    mimeType="text/plain"
                ))
        
        # Get collection info from vector store if already initialized
        if r1_rag.vector_store:
            collections = [
                ALL_VENDOR_DOCS_COLLECTION_NAME,
                ERROR_CODES_COLLECTION_NAME,
                HACKER_NEWS_COLLECTION_NAME
            ]
            
            for collection in collections:
                try:
                    # Get collection stats
                    collection_obj = r1_rag.vector_store.client.get_collection(collection)
                    count = collection_obj.count()
                    
                    resources.append(types.Resource(
                        uri=f"r1-rag://collections/{collection}",
                        name=f"Collection: {collection}",
                        description=f"Network documentation collection with {count} documents",
                        mimeType="application/json"
                    ))
                except Exception as e:
                    logger.warning(f"Could not get info for collection {collection}: {e}")
                    
            # Add vendor URL collections as resources
            url_files = ["arubalinks.txt", "aristalinks.txt", "juniperlinks.txt"]
            for url_file in url_files:
                if Path(url_file).exists():
                    vendor = url_file.replace("links.txt", "").capitalize()
                    resources.append(types.Resource(
                        uri=f"r1-rag://urls/{vendor.lower()}",
                        name=f"{vendor} URL Collection",
                        description=f"List of {vendor} documentation URLs for scraping",
                        mimeType="text/plain"
                    ))
                    
    except Exception as e:
        logger.error(f"Error listing resources: {e}")
        
    return resources

@server.read_resource()
async def handle_read_resource(uri: str) -> str:
    """Read resource content"""
    try:
        if uri.startswith("r1-rag://collections/"):
            # Only initialize components when actually needed
            initialized = await r1_rag.ensure_components_initialized(timeout_seconds=10)
            
            if not initialized:
                return json.dumps({
                    "error": "Components still initializing",
                    "status": r1_rag.get_initialization_status(),
                    "message": "Try again in a few moments"
                })
            
            collection_name = uri.split("/")[-1]
            if r1_rag.vector_store:
                # Get sample documents from collection
                results = r1_rag.vector_store.query(
                    collection_name=collection_name,
                    query_text="documentation overview",
                    n_results=5
                )
                
                if results and results.get('documents'):
                    return json.dumps({
                        "collection": collection_name,
                        "document_count": len(results['documents'][0]),
                        "sample_documents": results['documents'][0][:3],
                        "sample_metadata": results.get('metadatas', [{}])[0][:3] if results.get('metadatas') else []
                    }, indent=2)
                    
        elif uri.startswith("r1-rag://urls/"):
            vendor = uri.split("/")[-1]
            url_file = f"{vendor}links.txt"
            if Path(url_file).exists():
                with open(url_file, 'r') as f:
                    urls = f.read().strip().split('\n')
                return json.dumps({
                    "vendor": vendor,
                    "url_count": len(urls),
                    "urls": urls[:10]  # First 10 URLs as sample
                }, indent=2)
                
    except Exception as e:
        logger.error(f"Error reading resource {uri}: {e}")
        
    return "Resource not found or error occurred"

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available tools"""
    return [
        types.Tool(
            name="query_documentation",
            description="Search and query network documentation using RAG",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The question or search query about network documentation"
                    },
                    "vendor": {
                        "type": "string",
                        "description": "Specific vendor to search (aruba, cisco, juniper) or 'all' for all vendors",
                        "default": "all"
                    }
                },
                "required": ["query"]
            }
        ),
        types.Tool(
            name="search_vector_database",
            description="Direct vector similarity search in the documentation database",
            inputSchema={
                "type": "object",
                "properties": {
                    "query_text": {
                        "type": "string",
                        "description": "Text to search for using vector similarity"
                    },
                    "collection": {
                        "type": "string",
                        "description": "Collection to search in: 'all_vendor_docs' (all vendors), 'aruba_docs', 'cisco_docs', 'juniper_docs', 'arista_docs' (vendor-specific), 'error_codes', 'hacker_news_posts'",
                        "default": "all_vendor_docs"
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 20
                    }
                },
                "required": ["query_text"]
            }
        ),
        types.Tool(
            name="scrape_url",
            description="Scrape content from a network documentation URL",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL to scrape documentation from"
                    },
                    "vendor": {
                        "type": "string",
                        "description": "Vendor name (aruba, cisco, juniper, arista)"
                    }
                },
                "required": ["url", "vendor"]
            }
        ),
        types.Tool(
            name="web_search",
            description="Search the web for network documentation and information",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for web search"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 10
                    }
                },
                "required": ["query"]
            }
        ),
        types.Tool(
            name="get_collection_stats",
            description="Get statistics about document collections in the vector database",
            inputSchema={
                "type": "object",
                "properties": {
                    "collection": {
                        "type": "string",
                        "description": "Collection name to get stats for, or 'all' for all collections",
                        "default": "all"
                    }
                }
            }
        ),
        types.Tool(
            name="ingest_url_list",
            description="Ingest and process a list of URLs into the vector database",
            inputSchema={
                "type": "object",
                "properties": {
                    "vendor": {
                        "type": "string",
                        "description": "Vendor name (aruba, cisco, juniper, arista)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of URLs to process",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 100
                    }
                },
                "required": ["vendor"]
            }
        ),
        types.Tool(
            name="clear_web_search_cache",
            description="Clear the web search cache to resolve rate limiting issues",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        types.Tool(
            name="get_web_search_stats",
            description="Get web search cache statistics and rate limiting status",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Handle tool calls with improved error handling and timeout protection"""
    try:
        # Check initialization status first
        init_status = r1_rag.get_initialization_status()
        
        # For some tools, we can provide immediate responses without full initialization
        if name == "get_collection_stats":
            if not r1_rag._components_initialized:
                return [types.TextContent(
                    type="text",
                    text=json.dumps({
                        "initialization_status": init_status,
                        "message": "Components not yet initialized. Collections will be empty until data is ingested.",
                        "all_vendor_docs": {"document_count": 0, "status": "not_initialized"},
                        "error_codes": {"document_count": 0, "status": "not_initialized"},
                        "hacker_news_posts": {"document_count": 0, "status": "not_initialized"}
                    }, indent=2)
                )]
        
        # For other tools, attempt ultra-fast initialization 
        if not r1_rag._components_initialized:
            # Use ultra-fast initialization for immediate response
            await r1_rag.ultra_fast_init()
        
        # Proceed with tool execution
        if name == "query_documentation":
            query = arguments.get("query", "")
            vendor = arguments.get("vendor", "all")
            
            if not query:
                return [types.TextContent(type="text", text="Error: query parameter is required")]
            
            # Check if vector store is ready or loading
            if r1_rag._vector_store_ready and r1_rag.agent:
                # Vector store is ready, use full RAG
                try:
                    print("Using full RAG with ready vector store...", file=sys.stderr)
                    response = r1_rag.agent.generate_response(query)
                    return [types.TextContent(type="text", text=response)]
                except Exception as e:
                    print(f"RAG failed, falling back: {e}", file=sys.stderr)
                    # Fall through to fallback
            
            elif r1_rag._vector_store_loading:
                # Vector store is loading, provide immediate response with extracted docs + web search
                print("Vector store loading, using hybrid fallback...", file=sys.stderr)
                
                # Quick response while loading
                response = f"**Network Documentation Query Response** (Vector database loading...)\n\n"
                response += f"**Query**: {query}\n"
                if vendor != "all":
                    response += f"**Vendor**: {vendor}\n"
                response += "\n"
                
                # Try extracted documents search first
                try:
                    extracted_results = r1_rag.search_extracted_documents(query, "all_vendor_docs", 3)
                    if extracted_results.get('documents') and extracted_results['documents'][0]:
                        response += "**Documentation Extracts**:\n\n"
                        for i, doc in enumerate(extracted_results['documents'][0][:2], 1):
                            meta = extracted_results.get('metadatas', [[]])[0][i-1] if i-1 < len(extracted_results.get('metadatas', [[]])[0]) else {}
                            response += f"**{i}. {meta.get('title', 'Document')}**\n"
                            response += f"{doc[:300]}...\n"
                            if meta.get('url'):
                                response += f"Source: {meta['url']}\n"
                            response += "\n"
                    else:
                        response += "No relevant documentation extracts found.\n\n"
                except Exception as e:
                    print(f"Extracted docs search failed: {e}", file=sys.stderr)
                    response += "Documentation search temporarily unavailable.\n\n"
                
                # Add web search results
                if r1_rag.web_searcher:
                    try:
                        search_query = f"{vendor} {query}" if vendor != "all" else query
                        search_query += " network configuration documentation"
                        
                        web_results = r1_rag.web_searcher.search(search_query, max_results=3)
                        if web_results:
                            response += "**Web Search Results**:\n\n"
                            for i, result in enumerate(web_results, 1):
                                response += f"**{i}. {result.get('title', 'No Title')}**\n"
                                response += f"{result.get('snippet', 'No snippet available')}\n"
                                response += f"Source: {result.get('link', 'No link')}\n\n"
                    except Exception as e:
                        print(f"Web search failed: {e}", file=sys.stderr)
                        response += "Web search temporarily unavailable.\n"
                
                response += "\n*Note: Vector database is initializing in the background. Full RAG capabilities will be available shortly.*"
                return [types.TextContent(type="text", text=response)]
                
            else:
                # Vector store not ready and not loading, start initialization and provide immediate fallback
                print("Vector store not ready, providing immediate fallback response...", file=sys.stderr)
                
                # Start vector store loading if not already started
                if not r1_rag._vector_store_loading and not r1_rag.vector_store:
                    await r1_rag.start_background_vector_loading()
                
                # Provide immediate web search response
                if r1_rag.web_searcher:
                    try:
                        search_query = f"{vendor} {query}" if vendor != "all" else query
                        search_query += " network configuration enterprise documentation"
                        
                        web_results = r1_rag.web_searcher.search(search_query, max_results=5)
                        
                        if web_results:
                            response = f"**Network Documentation Search Results** (Initializing vector database...)\n\n"
                            response += f"**Query**: {query}\n"
                            if vendor != "all":
                                response += f"**Vendor**: {vendor}\n"
                            response += "\n**Web Search Results**:\n\n"
                            
                            for i, result in enumerate(web_results, 1):
                                response += f"**{i}. {result.get('title', 'No Title')}**\n"
                                response += f"{result.get('snippet', 'No snippet available')}\n"
                                response += f"Source: {result.get('link', 'No link')}\n\n"
                            
                            response += "\n*Note: Full RAG with 15,235 documents will be available once vector database initialization completes.*"
                            return [types.TextContent(type="text", text=response)]
                        else:
                            return [types.TextContent(type="text", text=f"Unable to find web search results for '{query}'. Vector database is initializing - please try again in a moment.")]
                    except Exception as web_e:
                        return [types.TextContent(type="text", text=f"Query processing failed: {str(web_e)}. Vector database is initializing.")]
                else:
                    return [types.TextContent(type="text", text="Both vector database and web search are initializing. Please try again in a moment.")]
            
        elif name == "search_vector_database":
            query_text = arguments.get("query_text", "")
            collection = arguments.get("collection", ALL_VENDOR_DOCS_COLLECTION_NAME)
            n_results = arguments.get("n_results", 5)
            
            if not query_text:
                return [types.TextContent(type="text", text="Error: query_text parameter is required")]
            
            # Try to initialize vector store, but use fallback if it fails
            if not r1_rag.vector_store:
                vector_store_ready = await r1_rag.ensure_vector_store()
                if not vector_store_ready:
                    # Use fallback immediately
                    print("Vector store initialization failed, using fallback...", file=sys.stderr)
                    fallback_results = r1_rag.search_extracted_documents(query_text, collection, n_results)
                    
                    if fallback_results.get('documents') and fallback_results['documents'][0]:
                        fallback_results["fallback_message"] = "Using extracted documents (vector store unavailable)"
                        fallback_results["suggestion"] = "Vector store initialization failed - using backup search"
                        return [types.TextContent(
                            type="text", 
                            text=json.dumps(fallback_results, indent=2)
                        )]
                    else:
                        return [types.TextContent(
                            type="text", 
                            text=json.dumps({
                                "error": "Vector store unavailable and no fallback data found",
                                "collection": collection,
                                "query": query_text,
                                "suggestion": "Check if extracted_documents.json exists or run migration",
                                "documents": [[]],
                                "metadatas": [[]],
                                "distances": [[]],
                                "ids": [[]]
                            }, indent=2)
                        )]
            
            # Handle vendor-specific collection requests by mapping to all_vendor_docs with vendor filter
            vendor_filter = None
            actual_collection = collection
            vendor_collections = ["aruba_docs", "cisco_docs", "juniper_docs", "arista_docs"]
            
            if collection in vendor_collections:
                # Map vendor collection to all_vendor_docs with vendor filter
                vendor_name = collection.replace("_docs", "")
                actual_collection = ALL_VENDOR_DOCS_COLLECTION_NAME
                vendor_filter = {"vendor": vendor_name}
                print(f"Mapping {collection} to {actual_collection} with vendor filter: {vendor_filter}", file=sys.stderr)
            
            # Check if collection exists and try to query
            try:
                coll = r1_rag.vector_store.client.get_collection(actual_collection)
                doc_count = coll.count()
                logger.info(f"Searching collection '{actual_collection}' with {doc_count} documents for query: '{query_text}'")
                
                if doc_count == 0:
                    return [types.TextContent(
                        type="text", 
                        text=json.dumps({
                            "message": f"Collection '{actual_collection}' is empty (0 documents)",
                            "collection": collection,
                            "query": query_text,
                            "suggestion": "Use the 'ingest_url_list' tool to populate the database with documentation.",
                            "documents": [[]],
                            "metadatas": [[]],
                            "distances": [[]],
                            "ids": [[]]
                        }, indent=2)
                    )]
                
                # Try to query the collection with vendor filter if specified
                if vendor_filter:
                    results = r1_rag.vector_store.query(
                        collection_name=actual_collection,
                        query_text=query_text,
                        n_results=n_results,
                        where_filter=vendor_filter
                    )
                else:
                    results = r1_rag.vector_store.query(
                        collection_name=actual_collection,
                        query_text=query_text,
                        n_results=n_results
                    )
                    
            except Exception as e:
                error_msg = str(e)
                if "'dict' object has no attribute 'dimensionality'" in error_msg:
                    # Try fallback to extracted documents
                    print("ChromaDB compatibility issue, trying fallback...", file=sys.stderr)
                    fallback_results = r1_rag.search_extracted_documents(query_text, collection, n_results)
                    
                    if fallback_results.get('documents') and fallback_results['documents'][0]:
                        fallback_results["fallback_message"] = "Using extracted documents due to ChromaDB compatibility issue"
                        fallback_results["suggestion"] = "Run 'python migrate_db.py' to restore full vector search functionality"
                        return [types.TextContent(
                            type="text", 
                            text=json.dumps(fallback_results, indent=2)
                        )]
                    else:
                        return [types.TextContent(
                            type="text", 
                            text=json.dumps({
                                "error": "ChromaDB compatibility issue detected",
                                "message": "The database was created with an older version of ChromaDB and needs migration",
                                "collection": collection,
                                "query": query_text,
                                "documents_available": doc_count if 'doc_count' in locals() else "unknown",
                                "solution": "Run 'python migrate_db.py' to fix database compatibility",
                                "technical_error": error_msg,
                                "documents": [[]],
                                "metadatas": [[]],
                                "distances": [[]],
                                "ids": [[]]
                            }, indent=2)
                        )]
                else:
                    return [types.TextContent(
                        type="text", 
                        text=json.dumps({
                            "error": f"Database error: {error_msg}",
                            "collection": collection,
                            "query": query_text,
                            "available_collections": list(r1_rag.vector_store.collections.keys()) if r1_rag.vector_store else [],
                            "documents": [[]],
                            "metadatas": [[]],
                            "distances": [[]],
                            "ids": [[]]
                        }, indent=2)
                    )]
            
            # Add query metadata to results for debugging
            results["query_info"] = {
                "query_text": query_text,
                "requested_collection": collection,
                "actual_collection": actual_collection,
                "vendor_filter": vendor_filter,
                "n_results": n_results,
                "total_docs_in_collection": doc_count
            }
            
            return [types.TextContent(
                type="text", 
                text=json.dumps(results, indent=2)
            )]
            
        elif name == "scrape_url":
            url = arguments.get("url", "")
            vendor = arguments.get("vendor", "")
            
            # Initialize scraper on-demand
            await r1_rag.ensure_heavy_components()
            
            if not r1_rag.scraper:
                return [types.TextContent(type="text", text="Scraper not initialized")]
                
            if not url:
                return [types.TextContent(type="text", text="Error: URL parameter is required")]
                
            # Create document metadata for the scraper
            doc_meta = {
                'url': url,
                'vendor': vendor.lower() if vendor else 'unknown',
                'doc_type': 'pdf' if url.lower().endswith('.pdf') else 'html',
                'title': f"{vendor} documentation from {url}" if vendor else f"Documentation from {url}"
            }
            
            # Extract content using the correct method
            try:
                content = r1_rag.scraper.extract_document_content(doc_meta)
                
                if content:
                    return [types.TextContent(
                        type="text",
                        text=f"Successfully scraped {url}\nVendor: {vendor}\nDocument type: {doc_meta['doc_type']}\nContent length: {len(content)} characters\n\nContent preview (first 500 chars):\n{content[:500]}..."
                    )]
                else:
                    return [types.TextContent(type="text", text=f"Failed to extract content from {url}")]
                    
            except Exception as e:
                return [types.TextContent(type="text", text=f"Error scraping {url}: {str(e)}")]
                
        elif name == "web_search":
            query = arguments.get("query", "")
            max_results = arguments.get("max_results", 5)
            
            if not r1_rag.web_searcher:
                return [types.TextContent(type="text", text="Web searcher not initialized")]
            
            # Update the web searcher's max_results if needed
            original_max_results = r1_rag.web_searcher.max_results
            r1_rag.web_searcher.max_results = max_results
            
            try:
                results = r1_rag.web_searcher.search(query)
                return [types.TextContent(
                    type="text",
                    text=json.dumps(results, indent=2)
                )]
            finally:
                # Restore original max_results
                r1_rag.web_searcher.max_results = original_max_results
            
        elif name == "get_collection_stats":
            collection = arguments.get("collection", "all")
            
            if not r1_rag.vector_store:
                return [types.TextContent(type="text", text="Vector store not initialized")]
                
            stats = {}
            collections_to_check = [
                ALL_VENDOR_DOCS_COLLECTION_NAME,
                ERROR_CODES_COLLECTION_NAME,
                HACKER_NEWS_COLLECTION_NAME
            ]
            
            if collection != "all":
                collections_to_check = [collection]
                
            for coll_name in collections_to_check:
                try:
                    coll = r1_rag.vector_store.client.get_collection(coll_name)
                    stats[coll_name] = {
                        "document_count": coll.count(),
                        "name": coll_name
                    }
                except Exception as e:
                    stats[coll_name] = {"error": str(e)}
            
            # Add initialization status
            stats["initialization_status"] = r1_rag.get_initialization_status()
            if r1_rag.initialization_errors:
                stats["initialization_errors"] = r1_rag.initialization_errors
                    
            return [types.TextContent(
                type="text",
                text=json.dumps(stats, indent=2)
            )]
            
        elif name == "ingest_url_list":
            vendor = arguments.get("vendor", "")
            limit = arguments.get("limit", 10)
            
            # Initialize heavy components on-demand  
            await r1_rag.ensure_heavy_components()
            
            if not r1_rag.ingestion_pipeline:
                return [types.TextContent(type="text", text="Ingestion pipeline not initialized")]
                
            if not vendor:
                return [types.TextContent(type="text", text="Error: vendor parameter is required")]
                
            # Read URLs from file
            url_file = f"{vendor.lower()}links.txt"
            if not os.path.exists(url_file):
                return [types.TextContent(type="text", text=f"URL file {url_file} not found. Available files: {[f for f in os.listdir('.') if f.endswith('links.txt')]}")]
                
            try:
                with open(url_file, 'r', encoding='utf-8') as f:
                    urls = [line.strip() for line in f if line.strip()][:limit]
                    
                if not urls:
                    return [types.TextContent(type="text", text=f"No URLs found in {url_file}")]
                    
                # Process URLs using the scraper and add to vector store
                results = []
                processed_count = 0
                
                for i, url in enumerate(urls):
                    try:
                        logger.info(f"Processing URL {i+1}/{len(urls)}: {url}")
                        
                        # Create document metadata
                        doc_meta = {
                            'url': url,
                            'title': f"{vendor} documentation from {url}",
                            'doc_type': 'pdf' if url.lower().endswith('.pdf') else 'html',
                            'vendor': vendor.lower()
                        }
                        
                        # Extract content using scraper
                        content = r1_rag.scraper.extract_document_content(doc_meta)
                        
                        if content:
                            # Process and add to vector store
                            processed_doc = {
                                'content': content,
                                'metadata': doc_meta
                            }
                            
                            chunks = r1_rag.ingestion_pipeline.processor.chunk_document(content, doc_meta)
                            if chunks:
                                r1_rag.vector_store.add_documents(vendor.lower(), chunks)
                                processed_count += 1
                                results.append(f"✓ {url}: Success ({len(chunks)} chunks)")
                            else:
                                results.append(f"✗ {url}: No chunks generated")
                        else:
                            results.append(f"✗ {url}: No content extracted")
                            
                    except Exception as e:
                        results.append(f"✗ {url}: Error - {str(e)}")
                        logger.error(f"Error processing URL {url}: {e}")
                        
                return [types.TextContent(
                    type="text",
                    text=f"Ingestion completed for {vendor}:\nProcessed {processed_count}/{len(urls)} URLs successfully\n\nResults:\n" + "\n".join(results)
                )]
                
            except Exception as e:
                return [types.TextContent(type="text", text=f"Error reading {url_file}: {str(e)}")]
                
        elif name == "clear_web_search_cache":
            if r1_rag.web_searcher:
                try:
                    r1_rag.web_searcher.clear_cache()
                    return [types.TextContent(type="text", text="Web search cache cleared successfully. This should help resolve rate limiting issues.")]
                except Exception as e:
                    return [types.TextContent(type="text", text=f"Error clearing web search cache: {str(e)}")]
            else:
                return [types.TextContent(type="text", text="Web searcher not initialized")]
                
        elif name == "get_web_search_stats":
            if r1_rag.web_searcher:
                try:
                    stats = r1_rag.web_searcher.get_cache_stats()
                    if hasattr(r1_rag.web_searcher, 'consecutive_failures'):
                        stats["consecutive_failures"] = r1_rag.web_searcher.consecutive_failures
                        stats["last_request_time"] = r1_rag.web_searcher.last_request_time
                    return [types.TextContent(type="text", text=json.dumps(stats, indent=2))]
                except Exception as e:
                    return [types.TextContent(type="text", text=f"Error getting web search stats: {str(e)}")]
            else:
                return [types.TextContent(type="text", text="Web searcher not initialized")]
            
        else:
            return [types.TextContent(type="text", text=f"Unknown tool: {name}")]
            
    except Exception as e:
        logger.error(f"Error in tool {name}: {e}")
        return [types.TextContent(type="text", text=f"Error: {str(e)}")]

async def main():
    """Main entry point for the MCP server"""
    print("Entering main function", file=sys.stderr)
    
    try:
        # NO initialization during startup - start completely clean for instant MCP handshake
        print("Skipping initialization during startup for instant MCP connection...", file=sys.stderr)
        
        # Run the server
        print("Starting MCP server...", file=sys.stderr)
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            print("MCP server streams established", file=sys.stderr)
            
            # Start background initialization AFTER MCP handshake completes
            print("Starting background initialization after MCP handshake...", file=sys.stderr)
            asyncio.create_task(r1_rag.delayed_initialization())
            
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="r1-rag",
                    server_version="1.0.0",
                    capabilities=server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )
            print("MCP server run completed", file=sys.stderr)
    except Exception as e:
        print(f"Fatal error in main: {e}", file=sys.stderr)
        import traceback
        print(f"Traceback: {traceback.format_exc()}", file=sys.stderr)
        raise

if __name__ == "__main__":
    print("Starting asyncio.run(main())", file=sys.stderr)
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Exception in asyncio.run: {e}", file=sys.stderr)
        import traceback
        print(f"Traceback: {traceback.format_exc()}", file=sys.stderr)
        sys.exit(1) 