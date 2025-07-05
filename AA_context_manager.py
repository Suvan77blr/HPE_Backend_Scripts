from chromadb import PersistentClient
import json
from typing import Dict

class ContextManager:
    def __init__(self):
        self.client = PersistentClient()
        self.collection = self.client.get_or_create_collection("mcp_sessions")
    
    def update_context(self, session_id: str, context: dict):
        self.collection.upsert(
            ids=[session_id],
            documents=[json.dumps(context)]
        )
    
    def get_context(self, session_id: str):
        return self.collection.get(ids=[session_id])

    def update_session_plan(self, session_id: str, plan: Dict):
        self.collection.upsert(
            ids=[f"session_{session_id}"],
            documents=[json.dumps({
                "current_step": 0,
                "workflow": plan
            })]
        )

    def get_next_tool(self, session_id: str) -> Dict:
        session_data = self.get_context(session_id)
        if not session_data: return None
        
        workflow = session_data.get("workflow", {})
        current_step = session_data.get("current_step", 0)
        
        if current_step == 0:
            return workflow['primary_tool']
        elif current_step - 1 < len(workflow.get('secondary_tools', [])):
            return workflow['secondary_tools'][current_step - 1]
        
        return None
