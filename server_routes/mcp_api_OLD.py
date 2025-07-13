from fastapi import FastAPI, HTTPException, Request, APIRouter
from pydantic import BaseModel
from typing import Optional, Dict, Any
import asyncio
import logging

from mcp_server import R1RAGMCPServer as r1_rag  # Assuming MCP server class is imported

# app = FastAPI(title="R1-RAG MCP Backend API", version="1.0")
router = APIRouter(prefix="/mcp", tags=["MCP Server"])

logger = logging.getLogger(__name__)

# ---------- Models ----------
class ToolCallRequest(BaseModel):
    arguments: Dict[str, Any]

class IngestRequest(BaseModel):
    vendor: str
    limit: Optional[int] = 10

class WarmCacheRequest(BaseModel):
    link_sources_with_vendor: list[dict]

# ---------- Endpoints ----------

# @app.get("/mcp/status")
@router.get("/status")
async def get_status():
    return {"status": r1_rag.get_initialization_status()}

# @app.post("/mcp/init")
@router.post("/init")
async def trigger_init():
    await r1_rag.start_background_initialization()
    return {"message": "Initialization started in background"}

# @app.post("/mcp/vector_store/init")
@router.post("/vector_store/init")
async def ensure_vector_store():
    ready = await r1_rag.ensure_vector_store()
    return {"vector_store_ready": ready}

# @app.get("/mcp/tools")
@router.get("/tools")
async def list_tools():
    from mcp_tools import handle_list_tools  # from your MCP handler
    return await handle_list_tools()

# @app.get("/mcp/resources")
@router.get("/resources")
async def list_resources():
    from mcp_resources import handle_list_resources
    return await handle_list_resources()

# @app.get("/mcp/resource/{uri:path}")
@router.get("/resource/{uri:path}")
async def read_resource(uri: str):
    from mcp_resources import handle_read_resource
    return await handle_read_resource(uri)

# @app.post("/mcp/tool/{tool_name}")
@router.post("/tool/{tool_name}")
async def call_tool(tool_name: str, request: ToolCallRequest):
    from mcp_tools import handle_call_tool
    try:
        results = await handle_call_tool(tool_name, request.arguments)
        return {"results": [r.dict() for r in results]}
    except Exception as e:
        logger.error(f"Error calling tool {tool_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# @app.post("/mcp/warm_cache")
@router.post("/warm_cache")
async def warm_cache(req: WarmCacheRequest):
    from warm_utils import warm_scraper_cache_from_link_files
    try:
        warm_scraper_cache_from_link_files(req.link_sources_with_vendor)
        return {"status": "Cache warming complete"}
    except Exception as e:
        logger.error(f"Cache warming failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# @app.post("/mcp/ingest")
@router.post("/ingest")
async def ingest_urls(req: IngestRequest):
    await r1_rag.ensure_heavy_components()
    try:
        success = await r1_rag.ingestion_pipeline.ingest_urls(req.vendor, req.limit)
        return {"status": "Ingestion complete", "success": success}
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
