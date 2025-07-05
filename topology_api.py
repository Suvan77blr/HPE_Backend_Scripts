# routes/topology_router.py

# from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
# from fastapi.staticfiles import StaticFiles
# from fastapi.templating import Jinja2Templates
# from fastapi.responses import HTMLResponse
# from pydantic import BaseModel
# from typing import Optional, Dict, Any
# import logging
# import uvicorn
# from vector_store import VectorStore
# from llm_service import LLMService
# from agent import NetworkIntegrationAgent
# from update_checker import UpdateChecker
# from web_search import WebSearcher
# from topology_analyzer import TopologyAnalyzer
# from environment import ENABLE_WEB_SEARCH, MAX_SEARCH_RESULTS
# from environment import GROQ_MODEL, GROQ_API_KEY

from fastapi import APIRouter, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging

from environment import ENABLE_WEB_SEARCH, MAX_SEARCH_RESULTS, GROQ_MODEL

from topology_analyzer_Latest import TopologyAnalyzer
# from topology_analyzer import TopologyAnalyzer
from OLD_agent import NetworkIntegrationAgent
from vector_store import VectorStore
from llm_service import LLMService
from web_search import WebSearcher

# Setup
router = APIRouter()
templates = Jinja2Templates(directory="templates")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Initialize services
vector_store = VectorStore()
llm_service = LLMService(model_name=GROQ_MODEL)
web_searcher = WebSearcher(max_results=MAX_SEARCH_RESULTS) if ENABLE_WEB_SEARCH else None
# web_searcher = None
# if ENABLE_WEB_SEARCH:
#     web_searcher = WebSearcher(max_results=MAX_SEARCH_RESULTS)

agent = NetworkIntegrationAgent(vector_store, llm_service, web_searcher)
topology_analyzer = TopologyAnalyzer(vector_store, web_searcher)


# # Initialize FastAPI app
# app = FastAPI(
#     title="Network Topology Analyzer API",
#     description="AI-powered network topology analysis and device replacement recommendations",
#     version="2.0.0"
# )

# Mount static files and templates
# app.mount("/static", StaticFiles(directory="static"), name="static")
# templates = Jinja2Templates(directory="templates")

# Existing models
class QueryRequest(BaseModel):
    query: str
    metadata: Optional[Dict[str, Any]] = None

class QueryResponse(BaseModel):
    response: str
    analysis: Optional[Dict[str, Any]] = None

# Enhanced models for topology analysis
class TopologyAnalysisResponse(BaseModel):
    success: bool
    original_topology: Optional[Dict[str, Any]] = None
    modified_topology: Optional[Dict[str, Any]] = None
    recommendations: Optional[Dict[str, Any]] = None
    diagrams: Optional[Dict[str, str]] = None
    analysis_summary: Optional[str] = None
    topology_explanation: Optional[str] = None
    context_sources: Optional[str] = None
    modification_details: Optional[Dict[str, Any]] = None
    implementation_guidance: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# Routes
@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main topology analyzer interface"""
    return templates.TemplateResponse("index.html", {"request": request})

@router.post("/analyze-topology", response_model=TopologyAnalysisResponse)
async def analyze_topology(
    image: UploadFile = File(...),
    replacement_query: str = Form(...),
):
    """Analyze network topology image and generate comprehensive replacement recommendations"""
    try:
        # Enhanced validation
        if not image.content_type or not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image (PNG, JPEG, GIF, etc.)")
        
        # Read image data
        image_data = await image.read()
        
        if len(image_data) == 0:
            raise HTTPException(status_code=400, detail="Empty image file received")
        
        # Enhanced query validation
        if not replacement_query.strip():
            raise HTTPException(status_code=400, detail="Replacement query cannot be empty")
        
        if len(replacement_query.strip()) < 20:
            raise HTTPException(status_code=400, detail="Please provide more detailed replacement requirements (minimum 20 characters)")
        
        logger.info(f"Processing comprehensive topology analysis for image: {image.filename}")
        logger.info(f"Image size: {len(image_data)} bytes")
        logger.info(f"Replacement query length: {len(replacement_query)} characters")
        logger.info(f"Query preview: {replacement_query[:100]}...")
        
        # Perform comprehensive analysis
        result = await topology_analyzer.analyze_and_replace_topology(
            image_data, replacement_query, agent
        )
        
        # # Augment result with additional RAG-based context
        # rag_context = agent.generate_response(replacement_query)
        # result["context_sources"] = rag_context

        logger.info(f"Analysis completed successfully: {result.get('success', False)}")
        
        return TopologyAnalysisResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in comprehensive topology analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a user query and return a response (existing functionality)"""
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

@router.get("/health")
async def health_check():
    """Enhanced application health check"""
    return {
        "status": "healthy", 
        "version": "2.0.0",
        "features": [
            "topology_analysis", 
            "device_replacement", 
            "mermaid_diagrams",
            "comprehensive_explanations",
            "implementation_guidance",
            "risk_assessment",
            "cost_analysis"
        ],
        "ai_models": {
            "image_analysis": "Gemini 1.5 Pro",
            "text_generation": GROQ_MODEL,
            "vector_search": "ChromaDB"
        }
    }

# if __name__ == "__main__":
    # uvicorn.run(app, host="0.0.0.0", port=8000)
