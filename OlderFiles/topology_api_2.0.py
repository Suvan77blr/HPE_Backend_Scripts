# routes/topology_router.py

from fastapi import APIRouter, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging

from environment import ENABLE_WEB_SEARCH, MAX_SEARCH_RESULTS, GROQ_MODEL

from topology_analyser import TopologyAnalyzer
from OLD_agent import NetworkIntegrationAgent
from vector_store import VectorStore
from llm_service import LLMService
from web_search import WebSearcher

# Setup
router = APIRouter()
templates = Jinja2Templates(directory="templates")
logger = logging.getLogger(__name__)

# Services
vector_store = VectorStore()
llm_service = LLMService(model_name=GROQ_MODEL)
web_searcher = WebSearcher(max_results=MAX_SEARCH_RESULTS) if ENABLE_WEB_SEARCH else None
agent = NetworkIntegrationAgent(vector_store, llm_service, web_searcher)
topology_analyzer = TopologyAnalyzer(vector_store, web_searcher)

# Models
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
    return templates.TemplateResponse("index.html", {"request": request})

@router.post("/analyze-topology", response_model=TopologyAnalysisResponse)
async def analyze_topology(image: UploadFile = File(...), replacement_query: str = Form(...)):
    try:
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        image_data = await image.read()

        if len(image_data) == 0:
            raise HTTPException(status_code=400, detail="Empty image file received")

        if len(replacement_query.strip()) < 20:
            raise HTTPException(status_code=400, detail="Please provide more detailed replacement requirements")

        result = await topology_analyzer.analyze_and_replace_topology(image_data, replacement_query, agent)
        return TopologyAnalysisResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in topology analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")