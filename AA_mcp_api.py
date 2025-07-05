from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import json
from AA_groq_orchestrator import GroqOrchestrator
import logging

logger = logging.getLogger(__name__)

log_label = "[MCP API]"
def router(groq_orchestrator: GroqOrchestrator) -> APIRouter:
    router = APIRouter()
    
    class MCPToolRequest(BaseModel):
        name: str
        arguments: Dict[str, Any]
        session: Optional[Dict[str, Any]] = None
        
    class MCPToolResponse(BaseModel):
        result: str
        session: Dict[str, Any]

    @router.post("/call_tool", response_model=MCPToolResponse)
    async def call_tool(request: MCPToolRequest):
        try:
            # # Ensure arguments is a dict, not a string
            # if not isinstance(request.arguments, dict):
            #     raise ValueError("arguments must be a dictionary")

            # Composing the query safely.
            query_str = f"{request.name} with arguments: {json.dumps(request.arguments)}"
            
            logger.info(f"{log_label} Query: {query_str}")
            # Getting plan from Groq orchestrator.
            plan = groq_orchestrator.route_query(query_str)

            logger.info(f"{log_label} Plan type: {type(plan)}\n")            
            logger.info(f"{log_label} Plan: {plan}")
            result = groq_orchestrator.execute_workflow(plan)
            
            return MCPToolResponse(
                result=result,
                session=request.session or {}
            )
            
        except Exception as e:
            logger.exception(f"{log_label} Full error traceback:")
            raise HTTPException(status_code=500, detail=str(e))
            
    return router
