import os
import json
from groq import Groq
from typing import Dict, Any, List, Union
import logging

logger = logging.getLogger(__name__)
ALL_VENDOR_DOCS_COLLECTION_NAME = "all_vendor_docs"
PLACEHOLDER = "<analysis from query_documentation>"
class GroqOrchestrator:
    QUERY_TOOLS = {'query_documentation', 'search_vector_database', 'web_search', 'analyze_topology'}
    def __init__(self, vector_store, llm_service, web_searcher=None, topology_analyzer=None, agent=None):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.vector_store = vector_store
        self.llm_service = llm_service
        self.web_searcher = web_searcher
        self.topology_analyzer = topology_analyzer
        self.agent = agent
        self.tool_definitions = self._load_tool_definitions()
    
    def _load_tool_definitions(self) -> List[Dict]:
        return [
            {
                "name": "query_documentation",
                "description": "Search network docs using RAG",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "vendor": {"type": "string", "default": "aruba"}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "search_vector_database",
                "description": "Search ChromaDB vector store",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "collection": {"type": "string", "default": ALL_VENDOR_DOCS_COLLECTION_NAME},
                        # "n_results": {"type": "integer", "default": 5, "minimum": 1, "maximum": 20}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "web_search",
                "description": "Search the web for information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "analyze_topology",
                "description": "Analyze a network topology image and generate device replacement recommendations.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_data": {"type": "string", "description": "Base64-encoded image data"},
                        "replacement_query": {"type": "string", "description": "Replacement requirements"}
                    },
                    "required": ["image_data", "replacement_query"]
                }
            }
        ]
    
    def is_tool_query(self, query: str) -> bool:
        """Determine if query should use tool routing"""
        return any(keyword in query.lower() for keyword in 
                  ["search", "find", "query", "documentation", "database"])
    
    def route_query(self, query: str) -> Dict:
        prompt = f"""
        Given user query: "{query}"
        Available tools: {json.dumps(self.tool_definitions)}
        
        Output JSON with:
        - primary_tool: tool name
        - parameters: dict of tool parameters
        - secondary_tools: list of subsequent tools (optional)
        """
        
        response = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192",
            response_format={"type": "json_object"},
            temperature=0.3
        )

        # Extract and parse the JSON plan
        plan_json = response.choices[0].message.content
        
        # Parse JSON to dict
        plan = json.loads(plan_json)
        plan = self.normalize_plan(plan, query)
    
        # Transform secondary_tools to list of dicts if needed
        if isinstance(plan.get('secondary_tools'), list) and all(isinstance(tool, str) for tool in plan['secondary_tools']):
            plan['secondary_tools'] = [{'name': tool, 'parameters': {}} for tool in plan['secondary_tools']]

        # Adding validationto ensure response is dict
        result = plan
        try:
            if isinstance(result, str):
                result = json.loads(result) # parse if string.
            return result   # Already a dict
        except:
            # Fallback if parsing fails.
            return {
                "primary_tool": "query_documentation",
                "parameters": {"query": query},
                "error": "Failed to parse Groq response"
            }
    # end route_query().
    
    def execute_tool(self, tool_name: str, params: Dict) -> Any:
        """Execute tool using existing components"""

        # Handle the missing 'query' for relevant tools.
        if tool_name in self.QUERY_TOOLS and 'query' not in params:
            # For now having a fallback query.
            params['query'] = 'Network configuration best practises'

        # Tool execution.
        if tool_name == "query_documentation":
            # Use existing RAG pipeline
            return self.llm_service.generate_response(params["query"])
            
        elif tool_name == "search_vector_database":
            return self.vector_store.query(
                # collection_name=params.get("collection", "aruba_docs"),
                collection_name=params.get("collection", ALL_VENDOR_DOCS_COLLECTION_NAME),
                query_text=params["query"],
                n_results=5
            )
            
        elif tool_name == "web_search" and self.web_searcher:
            return self.web_searcher.search(params["query"])

        elif tool_name == "analyze_topology":
            # image_data should be base64 encoded; decode before passing it.
            import base64
            image_bytes = base64.b64decode(params["image_data"])
            replacement_query = params["replacement_query"]

            # Using the existing TopologyAnalyzer instance.
            result = self.topology_analyzer.analyze_topology(image_bytes, replacement_query, self.agent)

        raise ValueError(f"Unsupported tool: {tool_name}")

    def execute_workflow(self, plan: Dict) -> str:
        # Type validation
        if not isinstance(plan, dict):
            logger.error(f"Invalid plan type: {type(plan)}. Expected dict.")
            return "Error: Invalid workflow plan"
        
        # Extract primary query
        primary_query = plan['parameters'].get('query', '')
        
        results = []
        
        # 1. Run query_documentation first.
        try:
            doc_analysis = self.execute_tool(plan['primary_tool'], plan['parameters'])
            results.append(doc_analysis)
            # primary_result = self.execute_tool(plan['primary_tool'], plan['parameters'])
            # results.append(primary_result)
        except Exception as e:
            logger.error(f"Primary tool execution failed: {str(e)}")
            results.append(f"Error in primary tool: {str(e)}")
            doc_analysis = ""
        
        # 2. For all secondary tools, use doc_analysis as the query
        # Execute secondary tools
        for tool in plan.get('secondary_tools', []):
            tool_name = tool.get('name') or tool.get('tool')  # Handle both 'name' and 'tool' keys
            if not tool_name:
                logger.error(f"Missing tool name in: {tool}")
                results.append(f"Error: Missing tool name in {tool}")
                continue
            try:
                params = tool.get('parameters', {})
                # Inherit query from primary if missing
                # Only replace if placeholder is present
                if params.get('query', '') == PLACEHOLDER:
                    params['query'] = doc_analysis
                elif 'query' not in params and doc_analysis:
                    params['query'] = doc_analysis
                # if 'query' not in params and primary_query:
                #     params['query'] = primary_query
                # params['query'] = doc_analysis if doc_analysis else params.get('query', '')
                result = self.execute_tool(tool_name, params)
                results.append(result)
            except Exception as e:
                logger.error(f"Secondary tool {tool['name']} failed: {str(e)}")
                results.append(f"Error in {tool['name']}: {str(e)}")
        
        # Build prompt with safe truncation
        MAX_PROMPT_LENGTH = 3000  # Characters (adjust based on token limits)
        prompt = f"""
        Original query: {primary_query}
        Tool outputs: {json.dumps(results)}
        Synthesize a comprehensive response.
        """
        
        # Truncate if needed
        if len(prompt) > MAX_PROMPT_LENGTH:
            truncated_msg = f"... [truncated {len(prompt)-MAX_PROMPT_LENGTH} chars]"
            prompt = prompt[:MAX_PROMPT_LENGTH] + truncated_msg
        
        # Generate response with error handling
        try:
            return self.llm_service.generate_response(prompt)
        except Exception as e:
            logger.error(f"LLM synthesis failed: {str(e)}")
            # Fallback: Return tool results directly
            return json.dumps({
                "status": "partial_response",
                "tools_executed": [plan['primary_tool']] + [t['name'] for t in plan.get('secondary_tools', [])],
                "results": results,
                "error": f"LLM synthesis failed: {str(e)}"
            })

    def normalize_plan(self, plan: Dict, original_query: str) -> Dict:
        # Do nothing, if query_documentation is already the primary_tool.
        if plan.get('primary_tool') == 'query_documentation':
            return plan

        # Otherwise, insert query_documentation as the primary tool
        normalized_plan = {
            "primary_tool": "query_documentation",
            "parameters": {
                "query": original_query,
                "vendor": plan['parameters'].get('vendor', 'aruba')
            },
            "secondary_tools": []
        }

        # Move the original primary tool to the front of secondary_tools
        orig_primary = {
            "name": plan['primary_tool'],
            "parameters": plan.get('parameters', {})
        }
        # Ensure the query for the original primary tool is set to the output of query_documentation
        orig_primary['parameters']['query'] = "<analysis from query_documentation>"

        # Update all secondary tools to use the output of query_documentation as their query if not already set
        secondary_tools = plan.get('secondary_tools', [])
        for tool in secondary_tools:
            if 'query' not in tool.get('parameters', {}):
                tool['parameters']['query'] = "<analysis from query_documentation>"

        normalized_plan['secondary_tools'] = [orig_primary] + secondary_tools
        return normalized_plan

# end GroqOrchestrator.

#     '''
#         Tool Mapping:
#             query_documentation → Existing RAG pipeline
#             search_vector_database → Direct vector store access
#             scrape_url → WebSearcher component
#             ingest_url_list → Existing ingestion pipeline
#     '''