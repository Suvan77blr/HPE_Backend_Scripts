import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from server_tools.web_search import WebSearcher  # <- You must provide this implementation
import openai
import time
import os

from chromadb.config import Settings
from groq import Groq
# --- Configuration ---
# GROQ_API_KEY = "your-groq-api-key"  # Replace with your Groq API key

GROQ_MODEL = "llama3-8b-8192"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
ALL_VENDOR_DOCS_COLLECTION_NAME = "all_vendor_docs"

openai.api_key = GROQ_API_KEY
openai.api_base = "https://api.groq.com/openai/v1"
groq_client = Groq(api_key=GROQ_API_KEY)

# --------------------- QUERY ANALYZER ---------------------
class QueryAnalyzerAgent:
    def __init__(self):
        print("[DEBUG] QueryAnalyzerAgent initialized!")  # <-- Add this line
        self.vendors = ["Cisco", "Juniper", "Arista", "Aruba", "HPE", "Huawei", "Fortinet", "Palo Alto", "F5", "Checkpoint"]
        self.intent_keywords = {
            "comparison": ["compare", "vs", "versus", "difference between", "differentiate"],
            "integration": ["integrate", "integration", "connect", "interoperate"],
            "configuration": ["config", "configure", "setup", "set up", "installation"],
            "troubleshooting": ["troubleshoot", "problem", "issue", "error", "not working", "fix"],
            "migration": ["migrate", "migration", "move from", "upgrade"],
            "product_info": ["features", "capabilities", "specifications", "specs"],
            "security": ["security", "firewall", "authentication", "encryption"],
            "performance": ["performance", "speed", "throughput", "latency"]
        }

    def analyze_query(self, query: str) -> dict:
        analysis = {
            "original_query": query,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "analysis_version": "1.0"
        }

        analysis.update(self._extract_vendors(query))
        analysis["intent"] = self._detect_intent(query)
        analysis.update(self._extract_products(query))
        analysis["confidence"] = self._calculate_confidence(analysis)

        return analysis

    def _extract_vendors(self, query: str) -> dict:
        found_vendors = {}
        q_lower = query.lower()
        import re
        from_to = re.search(r"(?:from|integrate)\s+(\w+)\s+(?:to|with)\s+(\w+)", q_lower)
        if from_to:
            source, target = from_to.groups()
            for v in self.vendors:
                if v.lower() == source:
                    found_vendors["source_vendor"] = v
                if v.lower() == target:
                    found_vendors["target_vendor"] = v
        if not found_vendors:
            for v in self.vendors:
                if v.lower() in q_lower:
                    if "source_vendor" not in found_vendors:
                        found_vendors["source_vendor"] = v
                    elif "target_vendor" not in found_vendors:
                        found_vendors["target_vendor"] = v
        return found_vendors

    def _detect_intent(self, query: str) -> str:
        q_lower = query.lower()
        for intent, keywords in self.intent_keywords.items():
            if any(k in q_lower for k in keywords):
                return intent
        return "general"

    def _extract_products(self, query: str) -> dict:
        q_lower = query.lower()
        products = {
            "switch": ["switch", "switches"],
            "router": ["router", "routers"],
            "firewall": ["firewall", "firewalls"],
            "access_point": ["access point", "ap", "aps"],
            "controller": ["controller", "controllers"],
            "server": ["server", "servers"]
        }
        for product, terms in products.items():
            if any(term in q_lower for term in terms):
                return {"product_type": product}
        return {}

    def _calculate_confidence(self, analysis: dict) -> float:
        confidence = 0.3
        if "source_vendor" in analysis or "target_vendor" in analysis:
            confidence += 0.3
        if analysis.get("intent") != "general":
            confidence += 0.2
        if "product_type" in analysis:
            confidence += 0.2
        return min(confidence, 1.0)

from sentence_transformers import SentenceTransformer

class MyEmbeddingFunction:
    def __init__(self, model):
        self.model = model
    def __call__(self, texts):
        return self.model.encode(texts).tolist()

# --------------------- RAG AGENT ---------------------
class RAGAgent:
    def __init__(self, embedding_fn=None):
        # self.embedding_model = SentenceTransformer("all-mpnet-base-v2")
        
        self.client = chromadb.Client(Settings(persist_directory="./chroma_db"))

        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-mpnet-base-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name = ALL_VENDOR_DOCS_COLLECTION_NAME,
            embedding_function = self.embedding_fn
        )
        print("[DEBUG] Listing all collections in ChromaDB...")
        print([c.name for c in self.client.list_collections()])
        self.collection = self.client.get_collection(
            name = ALL_VENDOR_DOCS_COLLECTION_NAME
        )

        try:
            peeked = self.collection.peek(5)
            print("[DEBUG] Peek sample from collection:")
            for i, doc in enumerate(peeked["documents"]):
                print(f"  Doc {i+1}: {doc}")
        except Exception as e:
            print("[ERROR] Could not peek:", e)
        # print("[DEBUG] Available metadata keys in ChromaDB collection:")
        # print(self.collection.peek(1))  # Will show one document and its metadata

    def retrieve(self, query: str, filter_metadata=None, n_results=3):
        try:
            print(f"[DEBUG] Collection count: {self.collection.count()} documents")

            print(f"[DEBUG][ChromaDB] Querying: '{query}'")
            print(f"[DEBUG][ChromaDB] Metadata Filter: {filter_metadata}")
            result = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=filter_metadata or None
            )
            print(f"[DEBUG][ChromaDB] Raw Result: {result}")
            # Return all documents (as a flat list)
            return sum(result["documents"], [])  # Flattens list of lists
        except Exception as e:
            print(f"[ERROR][ChromaDB] Retrieval failed: {e}")
            return []


# --------------------- WEB SEARCH AGENT ---------------------
class WebSearchAgent:
    def __init__(self, max_results=5):
        self.web_searcher = WebSearcher(max_results=max_results)

    def search(self, query, analysis=None):
        enhanced = query
        if analysis:
            if "target_vendor" in analysis:
                enhanced += f" {analysis['target_vendor']}"
            if "product_type" in analysis:
                enhanced += f" {analysis['product_type']}"
            if analysis.get("intent") == "troubleshooting":
                enhanced += " error solution troubleshooting"
        return self.web_searcher.search(enhanced)

# --------------------- LLM COMBINER ---------------------
# import openai

def ask_groq_llm(query, rag_texts, web_texts):
    combined_context = "\n\n".join(["[RAG] " + r for r in rag_texts] + ["[Web] " + w for w in web_texts])
    # system_prompt = """You are a network assistant. Based on the following combined context from internal documents and web search, generate a concise, helpful, and technically accurate answer to the user query."""
    system_prompt = """You are a highly knowledgeable network assistant. 
Use the [RAG] internal knowledge first whenever it contains relevant information. 
Only use [Web] context if [RAG] does not provide a sufficient answer. 
Ensure all answers are technically accurate, actionable, and concise."""


    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Query: {query}\n\nContext:\n{combined_context}"}
    ]

    # response = openai.chat.completions.create(
    #     model=GROQ_MODEL,
    #     messages=messages,
    #     temperature=0.3,
    #     max_tokens=500
    # )
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=500
    )

    return response.choices[0].message.content

# --------------------- STREAMLIT APP ---------------------
def main():
    st.title("🤖 A2A RAG + WebSearch with Groq LLM")

    query = st.text_input("🔍 Enter your network query:", value="How to configure VLAN on Aruba switch?")
    run = st.button("🚀 Run A2A Workflow")

    if run and query.strip():
        with st.spinner("Analyzing..."):
            analyzer = QueryAnalyzerAgent()
            analysis = analyzer.analyze_query(query)

        with st.spinner("Retrieving from ChromaDB..."):
            # rag_agent = RAGAgent(
            #     embedding_functions.SentenceTransformerEmbeddingFunction("all-MiniLM-L6-v2")
            # )
            rag_agent = RAGAgent(None)
            filter_metadata = {}
            if "target_vendor" in analysis:
                filter_metadata["vendor"] = analysis["target_vendor"].lower()
            elif "source_vendor" in analysis:
                filter_metadata["vendor"] = analysis["source_vendor"].lower()
            # rag_docs = rag_agent.retrieve(query, filter_metadata)
            rag_docs = rag_agent.retrieve(query)

            print(f"[DEBUG][MAIN] Retrieved {len(rag_docs)} documents from ChromaDB")

        with st.spinner("Performing Web Search..."):
            web_agent = WebSearchAgent(max_results=3)
            web_results = web_agent.search(query, analysis)
            web_snippets = [r.get("snippet", "") for r in web_results]

        with st.expander("📚 Retrieved from ChromaDB"):
            if not rag_docs:
                st.warning("No results retrieved from ChromaDB.")
            else:
                for i, doc in enumerate(rag_docs):
                    st.markdown(f"**Doc {i+1}:** {doc}")

        with st.expander("🌐 Snippets from Web Search"):
            if not web_snippets:
                st.warning("Web Search did not return any relevant results.")
            else:
                for i, snippet in enumerate(web_snippets):
                    st.markdown(f"**Snippet {i+1}:** {snippet}")

        with st.spinner("Generating final answer with Groq..."):
            final_answer = ask_groq_llm(query, rag_docs, web_snippets)

        st.markdown("### 🧠 Final Answer")
        # st.success(final_answer)
        st.markdown(final_answer)  # instead of st.success() to allow rich Markdown

if __name__ == "__main__":
    main()