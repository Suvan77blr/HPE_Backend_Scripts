import groq
from typing import Dict, Any, List
import logging
import os
from environment import GROQ_API_KEY, GROQ_MODEL


logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self, model_name=GROQ_MODEL):
        logger.info(f"Initializing Groq LLM with model: {model_name}")
        self.model_name = model_name
        
        self.client = groq.Client(api_key=GROQ_API_KEY)
        logger.info("Groq client initialized successfully")
        
        self.is_deepseek = "deepseek" in model_name.lower()

    def generate_response(self, prompt, max_tokens=2048):
        """Generate a response from the Groq LLM based on the prompt"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                stream=False
            )
            
            response_text = response.choices[0].message.content
            
            if self.is_deepseek:
                response_text = self._remove_thinking(response_text)
                
            return response_text
        except Exception as e:
            logger.error(f"Error generating response with Groq: {str(e)}")
            return "I encountered an error while processing your request."
    
    def _remove_thinking(self, response):
        """Remove the thinking process from DeepSeek model responses"""
        import re
        
        # Pattern 1: Remove content between <think> and </think> tags
        cleaned_response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        
        # Pattern 2: Remove content up to </think> tag (if opening tag is missing)
        cleaned_response = re.sub(r'^.*?</think>', '', cleaned_response, flags=re.DOTALL)
        
        # Remove any remaining tags that might be part of the thinking process
        cleaned_response = re.sub(r'<think>|</think>', '', cleaned_response)
        
        # Clean up any extra whitespace
        cleaned_response = cleaned_response.strip()
        
        # If the response is empty after cleaning, return the original
        if not cleaned_response and response:
            return response
            
        return cleaned_response

    def format_context_for_prompt(self, context_results: Dict[str, Any]) -> str:
        """Format retrieved context into a prompt-friendly format"""
        # Existing implementation remains the same
        formatted_context = "CONTEXT INFORMATION:\n\n"
        for collection_name, results in context_results.items():
            if not results.get('documents'):
                continue
            formatted_context += f"--- {collection_name.replace('_', ' ').upper()} ---\n"
            for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                formatted_context += f"Document {i+1}:\n"
                if metadata.get('title'):
                    formatted_context += f"Title: {metadata['title']}\n"
                if metadata.get('vendor'):
                    formatted_context += f"Vendor: {metadata['vendor']}\n"
                if metadata.get('product_line'):
                    formatted_context += f"Product: {metadata['product_line']}\n"
                formatted_context += f"Content: {doc[:500]}...\n\n"
        return formatted_context

    def create_integration_prompt(self, user_query: str, context: str) -> str:
        """Create a prompt for network integration questions"""

        formatting_instructions = """
# Response Formatting Instructions
When you are providing a replacement for a device from the CURRENT TOPOLOGY CONTEXT, you MUST format the recommendation on its own line using the following exact syntax:
`Replacement for <Original Device ID>: <Full Recommended Device Model>`

For example:
`Replacement for cisco_catalyst_9300_dist1: Aruba CX 6300M Switch`

Do not add any other text to this line. Provide your reasoning and justification in the surrounding text.
"""

        # For DeepSeek models via Groq
        if self.is_deepseek:
            return f"""You are assisting a project/product line manager or an account manager at an OEM firm (e.g., Cisco, Juniper, Aruba) in designing network solutions. Your task is to analyze customer queries and generate a suitable response in the form of an RFP and/or a network topology. Consider the available product features and SKUs to ensure the proposed solution aligns with the customer's requirements.

# Context Information

{context}

# User Query

{user_query}

{formatting_instructions}

Based on the context provided and your knowledge of networking, provide a detailed, step-by-step response that helps the user integrate their networking equipment. Include specific commands, configuration examples, and links to documentation when available. If the information in the context is not sufficient, clearly state what additional information would be needed.
"""
        else:
            # For other Groq models
            return f"""
You are a network integration specialist assistant. Use the following context information to answer the user's question about integrating networking equipment from different vendors.

# project/product line manager and an account manager of an OEM firm like Cisco Juniper Aruba, etc & the objective is to provide a network solution for a customer query in the form of an RFP and/or a network topology by gauging the features, SKUs provided

{context}

USER QUERY: {user_query}

{formatting_instructions}

Based on the context provided and your knowledge of networking, provide a detailed, step-by-step response that helps the user integrate their networking equipment. Include specific commands, configuration examples, and links to documentation when available. If the information in the context is not sufficient, clearly state what additional information would be needed.
"""
