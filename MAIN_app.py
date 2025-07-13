from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging
import uvicorn
from server_tools.vector_store import VectorStore
from server_tools.llm_service_OLD import LLMService
from server_tools.AA_agent import NetworkIntegrationAgent
from server_tools.web_search import WebSearcher
from server_tools.topology_analyzer import TopologyAnalyzer

from environment import ENABLE_WEB_SEARCH, MAX_SEARCH_RESULTS, GROQ_MODEL, GROQ_API_KEY

# Import Groq orchestrator
from groq_orchestrator import GroqOrchestrator

# Import routers
from server_routes.AA_mcp_api import router as mcp_router
from server_routes.topology_api import router as topology_router
from server_routes.admin_api import router as admin_router
from server_routes.simple_rag_api import build_rag_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize core components
vector_store = VectorStore()
llm_service = LLMService(model_name=GROQ_MODEL)
web_searcher = WebSearcher(max_results=MAX_SEARCH_RESULTS) if ENABLE_WEB_SEARCH else None
agent = NetworkIntegrationAgent(vector_store, llm_service, web_searcher)
topology_analyzer = TopologyAnalyzer(vector_store, web_searcher)

# Initialize Groq orchestrator
groq_orchestrator = GroqOrchestrator(
    vector_store=vector_store,
    llm_service=llm_service,
    web_searcher=web_searcher,
    topology_analyzer=topology_analyzer,
    agent=agent
)

app = FastAPI(
    title="Network Integration Assistant API",
    description="API for assisting with network equipment integration across vendors",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers with dependency injection
app.include_router( mcp_router(groq_orchestrator), prefix="/mcp" )
app.include_router(topology_router, prefix="/topology")
app.include_router(admin_router, prefix="/admin")
app.include_router(
    build_rag_router(
        llm_service=llm_service,
        vector_store=vector_store,
        web_searcher=web_searcher,
        groq_orchestrator=groq_orchestrator
    ), 
    prefix="/rag"
)

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
        # Use Groq orchestrator for tool-based queries
        if groq_orchestrator.is_tool_query(request.query):
            plan = groq_orchestrator.route_query(request.query)
            response = groq_orchestrator.execute_workflow(plan)
            return QueryResponse(response=response)
        else:
            # Fallback to original RAG for non-tool queries
            analysis = agent.analyze_query(request.query)
            response = agent.generate_response(request.query)
            return QueryResponse(response=response, analysis=analysis)
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
