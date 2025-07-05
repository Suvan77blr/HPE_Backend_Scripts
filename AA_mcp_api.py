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
            # Dynamic tool selection
            arguments = request.arguments
            tool_name = "analyze_topology" if "image_data" in arguments else request.name

            # Extract the original query for normalization
            original_query = arguments.get("query") or arguments.get("replacement_query") or ""
            logger.info(f"{log_label} Tool: {tool_name}, Arguments: {arguments}")
            
            # Get the plan from the orchestrator
            plan = groq_orchestrator.route_query(original_query if tool_name != "analyze_topology" else tool_name)

            logger.info(f"{log_label} Plan type: {type(plan)}")
            logger.info(f"{log_label} Plan: {plan}")

            result = groq_orchestrator.execute_workflow(plan)
            return MCPToolResponse(result=result, session=request.session or {})
        except Exception as e:
            logger.exception(f"{log_label} Full error traceback:")
            raise HTTPException(status_code=500, detail=str(e))
    # end call_tool().
            
    return router



# Earlier version ..
# @router.post("/call_tool", response_model=MCPToolResponse)
# async def call_tool(request: MCPToolRequest):
#     try:
#         # Dynamic tool selection for topology analysis
#         arguments = request.arguments
#         tool_name = request.name

#         # If image_data is present, force analyze_topology as the tool
#         if "image_data" in arguments:
#             tool_name = "analyze_topology"

#         # Extract the original query for normalization (if present)
#         original_query = arguments.get("query") or arguments.get("replacement_query") or ""

#         # Log the incoming request
#         logger.info(f"{log_label} Tool: {tool_name}, Arguments: {arguments}")

#         # Get the plan from the orchestrator
#         plan = groq_orchestrator.route_query(original_query if tool_name != "analyze_topology" else tool_name)
#         # Optionally, you can pass the full plan here if your orchestrator supports it:
#         # plan = {
#         #     "primary_tool": tool_name,
#         #     "parameters": arguments,
#         #     "secondary_tools": []
#         # }
#         # plan = groq_orchestrator.normalize_plan(plan, original_query)

#         logger.info(f"{log_label} Plan type: {type(plan)}")
#         logger.info(f"{log_label} Plan: {plan}")

#         result = groq_orchestrator.execute_workflow(plan)

#         return MCPToolResponse(
#             result=result,
#             session=request.session or {}
#         )

#     except Exception as e:
#         logger.exception(f"{log_label} Full error traceback:")
#         raise HTTPException(status_code=500, detail=str(e))