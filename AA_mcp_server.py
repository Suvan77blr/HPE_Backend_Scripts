
from fastapi import FastAPI, HTTPException
from groq_orchestrator import GroqOrchestrator
from vector_store import VectorStore
from web_search import WebSearcher
from utils.AA_mcp_handler import MCPHandler
import os

app = FastAPI()
vector_store = VectorStore()
web_searcher = WebSearcher()
orchestrator = GroqOrchestrator(vector_store, web_searcher)
mcp_handler = MCPHandler()

@app.post("/mcp/tools")
async def handle_mcp_tool_request(request: dict):
    try:
        # Route query through Groq orchestrator
        plan = orchestrator.route_query(
            query=request["query"],
            session_context=request.get("session", {})
        )
        
        # Execute workflow
        primary_result = orchestrator.execute_tool(
            plan['primary_tool'], 
            plan['parameters']
        )
        
        # Execute secondary tools
        secondary_results = []
        for tool in plan.get('secondary_tools', []):
            result = orchestrator.execute_tool(tool['name'], tool.get('parameters', {}))
            secondary_results.append(result)
        
        # Format MCP response
        return mcp_handler.format_response(
            primary_result, 
            secondary_results,
            request
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Existing topology routes
from routes.AA_topology import router as topology_router
app.include_router(topology_router, prefix="/topology")






# # Copying contents from earlier version
# """
# R1-RAG MCP Server

# Exposes the Network Documentation RAG system capabilities through the Model Context Protocol.
# This allows AI assistants to access and query your network documentation database.
# """

# from fastapi import FastAPI, HTTPException
# from AA_groq_orchestrator import GroqOrchestrator
# from vector_store import VectorStore

# ### Addition ###
# # Add to imports
# from AA_groq_orchestrator import GroqOrchestrator
# from typing import Dict


# # Lazy import helpers - imports happen only when components are first used
# def lazy_import_vector_store():
#     try:
#         from vector_store import VectorStore
#         print("VectorStore import successful", file=sys.stderr)
#         return VectorStore
#     except ImportError as e:
#         print(f"VectorStore import error: {e}", file=sys.stderr)
#         raise

# def lazy_import_agent():
#     try:
#         from agent import NetworkIntegrationAgent
#         print("NetworkIntegrationAgent import successful", file=sys.stderr)
#         return NetworkIntegrationAgent
#     except ImportError as e:
#         print(f"NetworkIntegrationAgent import error: {e}", file=sys.stderr)
#         raise

# def lazy_import_web_searcher():
#     try:
#         from web_search import WebSearcher
#         print("WebSearcher import successful", file=sys.stderr)
#         return WebSearcher
#     except ImportError as e:
#         print(f"WebSearcher import error: {e}", file=sys.stderr)
#         raise

# def lazy_import_scraper():
#     try:
#         from scraper import NetworkDocScraper
#         print("NetworkDocScraper import successful", file=sys.stderr)
#         return NetworkDocScraper
#     except ImportError as e:
#         print(f"NetworkDocScraper import error: {e}", file=sys.stderr)
#         raise

# def lazy_import_ingestion():
#     try:
#         from ingestion import DataIngestionPipeline
#         print("DataIngestionPipeline import successful", file=sys.stderr)
#         return DataIngestionPipeline
#     except ImportError as e:
#         print(f"DataIngestionPipeline import error: {e}", file=sys.stderr)
#         raise

# def lazy_import_document_processor():
#     try:
#         from document_processor import DocumentProcessor
#         print("DocumentProcessor import successful", file=sys.stderr)
#         return DocumentProcessor
#     except ImportError as e:
#         print(f"DocumentProcessor import error: {e}", file=sys.stderr)
#         raise

# def lazy_import_llm_service():
#     try:
#         from llm_service import LLMService
#         print("LLMService import successful", file=sys.stderr)
#         return LLMService
#     except ImportError as e:
#         print(f"LLMService import error: {e}", file=sys.stderr)
#         raise

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("r1-rag-mcp")

# # Define collection names locally to avoid importing VectorStore during startup
# ALL_VENDOR_DOCS_COLLECTION_NAME = "all_vendor_docs"
# ERROR_CODES_COLLECTION_NAME = "error_codes"
# HACKER_NEWS_COLLECTION_NAME = "hacker_news_posts"


# class R1RAGMCPServer:
#     def __init__(self):
#         self.vector_store = None
#         self.agent = None
#         self.web_searcher = None
#         self.scraper = None
#         self.ingestion_pipeline = None
#         self.doc_processor = None
#         self.llm_service = None
#         self.initialization_errors = []
#         self._components_initialized = False
#         self._initialization_in_progress = False
#         self._initialization_start_time = None
#         self._initialization_task = None
#         self._extracted_docs = None  # Cache for extracted documents
#         self._vector_store_loading = False
#         self._vector_store_ready = False
#         # ... existing code ...
#         ''' Addition '''
#         self.orchestrator = None        
    
#     async def handle_call_tool(self, name: str, arguments: dict) -> list:
#         # Replace Claude-based tool calls
#         await self.initialize_components()
        
#         # Route and execute using Groq orchestrator
#         plan = self.orchestrator.route_query(arguments.get("query", ""))
#         response = self.orchestrator.execute_workflow(plan)
        
#         return [types.TextContent(type="text", text=response)]

#     # async def initialize_components(self):
#     #     # ... existing code ...
#     #     if not self.orchestrator:
#     #         self.orchestrator = GroqOrchestrator(
#     #             self.vector_store,
#     #             self.llm_service,
#     #             self.web_searcher
#     #         )
    
#     async def handle_call_tool(self, name: str, arguments: dict) -> list:
#         # ... existing code ...
        
#         if name == "query_documentation":
#             # Use orchestrator for this tool
#             await self.initialize_components()
            
#             plan = self.orchestrator.route_query(
#                 arguments.get("query", ""),
#                 session_context
#             )
            
#             response = self.orchestrator.execute_workflow(
#                 plan,
#                 session_context
#             )
            
#             return [types.TextContent(type="text", text=response)]
        
#         # ... handle other tools normally ...

#     # Execution workflow. .. not sure where this fits.
#     def execute_workflow(plan):
#         vector_results = call_tool('search_vector_database', plan['parameters'])
#         web_results = call_tool('web_search', {'query': 'Aruba CX examples'})
#         return synthesize_results([vector_results, web_results])

#     # fallback_mechanism.. 
#     def route_query(self, query: str) -> Dict:
#         try:
#             # Groq-based routing
#             pass
#         except Exception as e:
#             # Fallback to predefined workflows
#             return {
#                 "primary_tool": "query_documentation",
#                 "parameters": {"query": query},
#                 "fallback_used": True
#             }
