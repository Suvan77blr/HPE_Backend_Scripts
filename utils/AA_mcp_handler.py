class MCPHandler:
    def __init__(self):
        self.tool_registry = {}
    
    def register_tool(self, tool_name: str, tool_definition: dict):
        self.tool_registry[tool_name] = tool_definition
    
    def format_response(self, primary_result, secondary_results, request: dict) -> dict:
        return {
            "mcp_version": "1.0",
            "server": "groq-mcp",
            "response": {
                "primary": self._format_content(primary_result),
                "secondary": [self._format_content(r) for r in secondary_results]
            },
            "session": request.get("session", {}),
            "metadata": {
                "model": "llama3-70b-8192",
                "tool_sequence": [request['tool']] + [t['name'] for t in request.get('secondary_tools', [])]
            }
        }
    
    def _format_content(self, content):
        if isinstance(content, str):
            return {"type": "text", "content": content}
        elif isinstance(content, dict):
            return {"type": "structured", "content": content}
        # Add other content types
        return {"type": "raw", "content": str(content)}
