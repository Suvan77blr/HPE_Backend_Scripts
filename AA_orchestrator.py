from groq import Groq
from crewai import Agent, Task, Crew
import json

class MCPOrchestrator:
    def __init__(self):
        self.client = Groq()
        self.llm_config = {
            "model": "llama3-70b-8192",
            "temperature": 0.3
        }
    
    def route_query(self, query: str, context: dict) -> dict:
        prompt = f"""
        Given MCP tools: 
        1. query_documentation (technical docs)
        2. search_vector_database (ChromaDB search)
        3. scrape_url (web scraping)
        4. web_search (external search)
        5. get_collection_stats (DB metrics)
        6. ingest_url_list (batch ingestion)
        
        User query: "{query}"
        Context: {context}
        
        Select primary tool and secondary tools (if needed). Respond in JSON:
        {{"primary_tool": "tool_name", "secondary_tools": ["tool1", "tool2"]}}
        """
        response = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            **self.llm_config
        )
        return json.loads(response.choices[0].message.content)
