from autogen import AssistantAgent, UserProxyAgent
from AA_orchestrator import MCPOrchestrator 

class MCPAgentSystem:
    def __init__(self):
        self.orchestrator = MCPOrchestrator()
        self.agents = {
            "query_documentation": AssistantAgent(
                name="doc_agent",
                system_message="Expert in technical documentation",
                llm_config={"config_list": [self.orchestrator.llm_config]}
            ),
            "search_vector_database": AssistantAgent(
                name="vector_agent",
                system_message="ChromaDB search specialist",
                llm_config={"config_list": [self.orchestrator.llm_config]}
            ),
            # ... other agents
        }
    
    def execute_workflow(self, query: str, session_id: str):
        # Get tool routing decision
        tool_plan = self.orchestrator.route_query(query, self._get_context(session_id))
        
        # Execute primary tool
        primary_agent = self.agents[tool_plan["primary_tool"]]
        result = primary_agent.initiate_chat(
            message=query,
            summary_method="reflection_with_llm"
        )
        
        # Handle secondary tools
        for tool in tool_plan.get("secondary_tools", []):
            secondary_agent = self.agents[tool]
            secondary_agent.initiate_chat(
                message=f"Process based on: {result.summary}",
                summary_method="reflection_with_llm"
            )
        
        return self._format_output(result)
