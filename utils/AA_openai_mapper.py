from fastapi.openapi.utils import get_openapi

def generate_mcp_tools(app):
    """Convert FastAPI routes to MCP tools"""
    openapi_spec = get_openapi(
        title=app.title,
        version=app.version,
        routes=app.routes,
    )
    
    mcp_tools = []
    for path, methods in openapi_spec["paths"].items():
        for method, spec in methods.items():
            mcp_tools.append({
                "name": spec["operationId"],
                "description": spec.get("description", ""),
                "parameters": {
                    "type": "object",
                    "properties": self._map_parameters(spec.get("parameters", [])),
                    "required": [p["name"] for p in spec.get("parameters", []) if p.get("required")]
                }
            })
    return mcp_tools

def _map_parameters(self, parameters: list) -> dict:
    # Convert OpenAPI parameters to JSON Schema
    properties = {}
    for param in parameters:
        properties[param["name"]] = {
            "type": param["schema"]["type"] if "schema" in param else "string",
            "description": param.get("description", "")
        }
    return properties
