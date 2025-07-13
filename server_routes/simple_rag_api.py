from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import logging
import json

logger = logging.getLogger(__name__)

# Request/Response Models
class QueryRequest(BaseModel):
    query: str
    metadata: Optional[Dict[str, Any]] = None
    use_orchestrator: Optional[bool] = True
    force_tool_routing: Optional[bool] = False

class QueryResponse(BaseModel):
    response: str
    analysis: Optional[Dict[str, Any]] = None
    orchestrator_used: bool = False
    tools_executed: Optional[List[str]] = None
    workflow_plan: Optional[Dict[str, Any]] = None
    execution_details: Optional[Dict[str, Any]] = None
    rag_context_used: bool = False
    documents_retrieved: Optional[int] = None

class ToolRequest(BaseModel):
    tool_name: str
    parameters: Dict[str, Any]

class ToolResponse(BaseModel):
    result: Any
    tool_name: str
    success: bool
    error: Optional[str] = None

class VectorQueryRequest(BaseModel):
    query: str
    collection: Optional[str] = "all_vendor_docs"
    n_results: Optional[int] = 3

class StatusResponse(BaseModel):
    status: str
    llm_model: str
    web_search_enabled: bool
    groq_api_key_set: bool
    orchestrator_ready: bool
    vector_store_ready: bool
    available_tools: List[str]
    collections_available: List[str]

def build_rag_router(llm_service, vector_store, web_searcher, groq_orchestrator) -> APIRouter:
    router = APIRouter()

    @router.get("/status", response_model=StatusResponse)
    async def get_status():
        return StatusResponse(
            status="running",
            llm_model=llm_service.model_name,
            web_search_enabled=web_searcher is not None,
            groq_api_key_set=True,  # Assume valid key already checked
            orchestrator_ready=True,
            vector_store_ready=True,
            available_tools=[tool["name"] for tool in groq_orchestrator.tool_definitions],
            collections_available=list(vector_store.collections.values())
        )

    @router.post("/agentic_query", response_model=QueryResponse)
    async def agentic_query(request: QueryRequest):
        try:
            logger.info(f"Agentic query: {request.query}")

            analysis = {
                "original_query": request.query,
                "intent": "general",
                "vendors_detected": []
            }

            vendors = ["Cisco", "Juniper", "Arista", "Aruba", "HPE", "Huawei"]
            for vendor in vendors:
                if vendor.lower() in request.query.lower():
                    analysis["vendors_detected"].append(vendor)

            if "compare" in request.query.lower():
                analysis["intent"] = "comparison"
            elif "configure" in request.query.lower():
                analysis["intent"] = "configuration"
            elif any(w in request.query.lower() for w in ["troubleshoot", "error", "problem"]):
                analysis["intent"] = "troubleshooting"

            should_use_orch = request.use_orchestrator and (
                request.force_tool_routing or
                groq_orchestrator.is_tool_query(request.query) or
                analysis["intent"] in ["comparison", "configuration", "troubleshooting"]
            )

            if should_use_orch:
                plan = groq_orchestrator.route_query(request.query)
                response = groq_orchestrator.execute_workflow(plan)
                # tools_executed = [plan['primary_tool']] + [t['name'] for t in plan.get('secondary_tools', [])]
                tools_executed = [plan['primary_tool']]
                for tool in plan.get('secondary_tools', []):
                    tool_name = tool.get('name') or tool.get('tool')  # fallback key support
                    if not tool_name:
                        logger.warning(f"Skipping secondary tool due to missing name: {tool}")
                        continue
                tools_executed.append(tool_name)
                rag_used = 'query_documentation' in tools_executed
                try:
                    docs = vector_store.query("all_vendor_docs", request.query, 5)
                    docs_count = len(docs)
                except:
                    docs_count = 0
                return QueryResponse(
                    response=response,
                    analysis=analysis,
                    orchestrator_used=True,
                    tools_executed=tools_executed,
                    workflow_plan=plan,
                    execution_details={
                        "primary_tool": plan.get('primary_tool'),
                        "secondary_tools": plan.get('secondary_tools', [])
                    },
                    rag_context_used=rag_used,
                    documents_retrieved=docs_count
                )
            else:
                response = llm_service.generate_response(request.query)
                return QueryResponse(
                    response=response,
                    analysis=analysis,
                    orchestrator_used=False,
                    tools_executed=["direct_llm"],
                    rag_context_used=False
                )
        except Exception as e:
            logger.error(f"Agentic query failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Agentic query error: {str(e)}")

    @router.post("/orchestrator_query", response_model=QueryResponse)
    async def orchestrator_query(request: QueryRequest):
        request.use_orchestrator = True
        request.force_tool_routing = True
        return await agentic_query(request)

    @router.post("/direct_llm_query", response_model=QueryResponse)
    async def direct_llm_query(request: QueryRequest):
        try:
            response = llm_service.generate_response(request.query)
            return QueryResponse(
                response=response,
                orchestrator_used=False,
                tools_executed=["direct_llm"],
                rag_context_used=False
            )
        except Exception as e:
            logger.error(f"Direct LLM query failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Direct query failed: {str(e)}")

    @router.post("/tool_execute", response_model=ToolResponse)
    async def tool_execute(request: ToolRequest):
        try:
            result = groq_orchestrator.execute_tool(request.tool_name, request.parameters)
            return ToolResponse(result=result, tool_name=request.tool_name, success=True)
        except Exception as e:
            return ToolResponse(result=None, tool_name=request.tool_name, success=False, error=str(e))

    @router.post("/vector_query", response_model=Dict[str, Any])
    async def vector_query(request: VectorQueryRequest):
        try:
            docs = vector_store.query(request.collection, request.query, request.n_results)
            if isinstance(docs, dict):
                docs = docs.get("documents", [])
            return {
                "query": request.query,
                "collection": request.collection,
                "documents": docs,
                "total_documents": len(docs)
            }
        except Exception as e:
            logger.error(f"Vector query failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Vector query failed: {str(e)}")

    @router.get("/tools", response_model=List[Dict[str, Any]])
    async def list_tools():
        return groq_orchestrator.tool_definitions

    @router.get("/test")
    async def test():
        return {"message": "Simple RAG API merged and running!"}

    return router
