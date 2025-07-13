import logging
import re
import json
from typing import Dict, List, Any, Optional

from server_tools.vector_store import VectorStore # For collection name constants
from server_tools.llm_service_OLD import LLMService
from server_tools.web_search import WebSearcher # Assuming this is your web search class

logger = logging.getLogger(__name__)

class NetworkIntegrationAgent:
    def __init__(self, vector_store: VectorStore, llm_service: LLMService, web_searcher: Optional[WebSearcher] = None):
        self.vector_store = vector_store
        self.llm_service = llm_service
        self.web_searcher = web_searcher

    def extract_vendors_from_query(self, query: str) -> Dict[str, Any]:
        """Extract vendor information from the user query with default fallback to Aruba"""
        vendors = ["Cisco", "Juniper", "Arista", "Aruba", "HPE", "Huawei", "Fortinet", "Palo Alto", "F5", "Checkpoint"]
        found = {}
        lower_query = query.lower()

        # Check if it's a comparison to all others
        if re.search(r"(all (other|vendors)|every (other|vendor)|across vendors)", lower_query):
            found["source"] = "Aruba"
            found["target"] = "ALL"
            found["compare_with_all"] = True
            found["other_vendors"] = [v for v in vendors if v != "Aruba"]
            return found

        # Try explicit "from X to Y" or "integrate X with Y"
        from_to_match = re.search(r"(?:from|integrate)\s+([A-Za-z0-9-]+)\s+(?:to|with)\s+([A-Za-z0-9-]+)", query, re.IGNORECASE)
        if from_to_match:
            source_candidate = from_to_match.group(1)
            target_candidate = from_to_match.group(2)
            for v in vendors:
                if v.lower() == source_candidate.lower():
                    found["source"] = v
                if v.lower() == target_candidate.lower():
                    found["target"] = v
            return found

        # Otherwise just collect mentioned vendors
        mentioned = [v for v in vendors if v.lower() in lower_query]
        if mentioned:
            if "Aruba" in mentioned:
                found["source"] = "Aruba"
                others = [v for v in mentioned if v != "Aruba"]
                if others:
                    found["target"] = others[0]
            else:
                found["source"] = mentioned[0]
                found["target"] = "Aruba"  # ✅ Default fallback
        else:
            # No vendor mentioned, assume it's Aruba vs all
            found["source"] = "Aruba"
            found["target"] = "ALL"
            found["compare_with_all"] = True
            found["other_vendors"] = [v for v in vendors if v != "Aruba"]
        return found
    # end extract_vendors_from_query()


    def extract_product_from_query(self, query: str) -> Dict[str, str]:
        """Extract product information from the user query"""
        product_patterns = [
            r"Nexus\s+\d+[A-Za-z0-9-]*", r"Catalyst\s+\d+[A-Za-z0-9-]*",
            r"MX\s*\d+[A-Za-z0-9-]*", r"EX\s*\d+[A-Za-z0-9-]*", r"SRX\s*\d+[A-Za-z0-9-]*", r"QFX\s*\d+[A-Za-z0-9-]*",
            r"DCS-\d+[A-Za-z0-9-]*", r"7\d{3}X[A-Za-z0-9-]*",
            r"FortiGate\s+\d+[A-Za-z0-9-]*", r"PA-\d+[A-Za-z0-9-]*",
            r"CX\s+\d+[A-Za-z0-9-]*",  # Aruba CX
            r"AOS-CX\s+\d+\.\d+"     # AOS-CX versions
        ]
        found_products = {}
        # This logic is also simplistic for determining source/target product
        for pattern in product_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                # Basic assignment, could be improved with proximity to "from", "to", "new", "old"
                if "source_product" not in found_products:
                    found_products["source_product"] = match
                elif "target_product" not in found_products:
                    found_products["target_product"] = match
                break # Take first match for pattern
            if "source_product" in found_products and "target_product" in found_products:
                break
        return found_products

    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze the user query to extract key information"""
        analysis = {"original_query": query}
        
        vendors = self.extract_vendors_from_query(query)
        analysis.update(vendors)
        
        # If it's a broad comparison, convert it into specific Aruba-vs-others
        if vendors.get("compare_with_all"):
            analysis["intent"] = "comparison"
            analysis["compare_with_all"] = True
            analysis["source"] = "Aruba"
            analysis["compare_item_1_vendor"] = "Aruba"
            analysis["other_vendors"] = vendors.get("other_vendors", [])

        products = self.extract_product_from_query(query)
        analysis.update(products)
        
        # Intent detection refinement
        is_comparison_query = False
        comparison_keywords = [r"compare", r"vs", r"versus", r"difference between", r"differentiate"]
        for keyword_pattern in comparison_keywords:
            if re.search(keyword_pattern, query, re.IGNORECASE):
                is_comparison_query = True
                break
        
        # if is_comparison_query and analysis.get("source") and analysis.get("target"):
        if is_comparison_query and not analysis.get("intent"):
            analysis["intent"] = "comparison"
            analysis["compare_item_1_desc"] = analysis.get("source_product") or analysis.get("source")
            analysis["compare_item_2_desc"] = analysis.get("target_product") or analysis.get("target")
            analysis["compare_item_1_vendor"] = analysis.get("source")
            analysis["compare_item_2_vendor"] = analysis.get("target")
        elif re.search(r"how\\s+to\\s+integrate|integrat(e|ion)", query, re.IGNORECASE):
            analysis["intent"] = "integration"
        elif re.search(r"configur(e|ation)|setup|set\s+up", query, re.IGNORECASE):
            analysis["intent"] = "configuration"
        elif re.search(r"troubleshoot|problem|issue|error|not\s+working", query, re.IGNORECASE):
            analysis["intent"] = "troubleshooting"
        elif re.search(r"migrat(e|ion)|move\s+from", query, re.IGNORECASE):
            analysis["intent"] = "migration"
        elif re.search(r"features|capabilities|specifications", query, re.IGNORECASE) and not is_comparison_query: # ensure compare takes precedence
            analysis["intent"] = "product_info"
        else:
            analysis["intent"] = "general"
            
        # If a vendor is mentioned, assume it's the target if not specified otherwise
        if "source" not in analysis and "target" not in analysis:
            all_vendors = ["Cisco", "Juniper", "Arista", "Aruba"] # Extend as needed
            for v_name in all_vendors:
                if v_name.lower() in query.lower():
                    analysis["target"] = v_name # Default to target if only one vendor mentioned
                    break
        return analysis

    def retrieve_relevant_context(self, query: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve relevant context based on the query analysis using single vendor collection"""
        context_results = {}
        
        # Determine the vendor to filter by, prioritizing target vendor
        filter_vendor = analysis.get("target", analysis.get("source")) 
        
        search_text = query # Default search text
        
        if analysis.get("intent") == "comparison":
            item1_vendor = analysis.get("compare_item_1_vendor")
            item1_desc = analysis.get("compare_item_1_desc")
            item2_vendor = analysis.get("compare_item_2_vendor")
            item2_desc = analysis.get("compare_item_2_desc")

            if item1_vendor and item1_desc:
                search_text_item1 = f"{item1_vendor} {item1_desc} specifications features"
                item1_results = self.vector_store.query(
                    collection_name=VectorStore.ALL_VENDOR_DOCS_COLLECTION_NAME,
                    query_text=search_text_item1,
                    n_results=2, # Get a couple of docs for each item
                    where_filter={"vendor": item1_vendor.lower()}
                )
                if item1_results and item1_results.get('documents') and item1_results['documents'][0]:
                    context_results[f"{item1_vendor.lower()}_{item1_desc.replace(' ', '_').lower()}_docs"] = item1_results
            
            if item2_vendor and item2_desc:
                search_text_item2 = f"{item2_vendor} {item2_desc} specifications features"
                item2_results = self.vector_store.query(
                    collection_name=VectorStore.ALL_VENDOR_DOCS_COLLECTION_NAME,
                    query_text=search_text_item2,
                    n_results=2, # Get a couple of docs for each item
                    where_filter={"vendor": item2_vendor.lower()}
                )
                if item2_results and item2_results.get('documents') and item2_results['documents'][0]:
                     context_results[f"{item2_vendor.lower()}_{item2_desc.replace(' ', '_').lower()}_docs"] = item2_results
            logger.debug(f"Comparison context: {context_results}")
            return context_results # Return after attempting to fetch for both items
        
        # Refine search text or target collection based on intent
        if analysis.get("intent") == "troubleshooting":
            # Query error codes collection
            error_results = self.vector_store.query(
                collection_name=VectorStore.ERROR_CODES_COLLECTION_NAME,
                query_text=query,
                n_results=3
            )
            if error_results and error_results.get('documents') and error_results['documents'][0]:
                 context_results[VectorStore.ERROR_CODES_COLLECTION_NAME] = error_results

            # Also search vendor docs if a vendor is identified for troubleshooting context
            if filter_vendor:
                 search_text_vendor_troubleshoot = f"{filter_vendor} {analysis.get('target_product','')} troubleshooting {query}"
                 vendor_troubleshoot_results = self.vector_store.query(
                    collection_name=VectorStore.ALL_VENDOR_DOCS_COLLECTION_NAME,
                    query_text=search_text_vendor_troubleshoot,
                    n_results=2,
                    where_filter={"vendor": filter_vendor.lower()}
                )
                 if vendor_troubleshoot_results and vendor_troubleshoot_results.get('documents') and vendor_troubleshoot_results['documents'][0]:
                    # Merge or add separately
                    if VectorStore.ALL_VENDOR_DOCS_COLLECTION_NAME in context_results: # Should not happen if error_codes is first
                        # This merging logic can be complex, for now, let's keep them separate if error_codes was already populated
                        context_results[VectorStore.ALL_VENDOR_DOCS_COLLECTION_NAME + "_troubleshooting"] = vendor_troubleshoot_results
                    else:
                        context_results[VectorStore.ALL_VENDOR_DOCS_COLLECTION_NAME] = vendor_troubleshoot_results
            logger.debug(f"Early returning context results: {context_results}")
            return context_results # Early return for troubleshooting focus

        # For other intents, primarily query the all_vendor_docs collection
        if filter_vendor:
            # Construct a more specific query if product info is available
            if analysis.get("target_product"):
                search_text = f"{filter_vendor} {analysis.get('target_product')} {analysis.get('intent')} {query}"
            elif analysis.get("source_product") and analysis.get("intent") == "migration":
                 search_text = f"migrate from {analysis.get('source')} {analysis.get('source_product')} to {filter_vendor} {query}"
            else:
                search_text = f"{filter_vendor} {analysis.get('intent')} {query}"

            vendor_docs_results = self.vector_store.query(
                collection_name=VectorStore.ALL_VENDOR_DOCS_COLLECTION_NAME,
                query_text=search_text,
                n_results=3,
                where_filter={"vendor": filter_vendor.lower()}
            )
            if vendor_docs_results and vendor_docs_results.get('documents') and vendor_docs_results['documents'][0]:
                context_results[VectorStore.ALL_VENDOR_DOCS_COLLECTION_NAME] = vendor_docs_results
        
        # If no specific vendor context or if the query is general, search all vendor docs without filter or HackerNews
        if not context_results or analysis.get("intent") == "general":
            # If intent is general and no specific vendor, could search hacker news or all vendor docs broadly
            # For now, let's query all_vendor_docs broadly if no specific vendor results.
            if not context_results.get(VectorStore.ALL_VENDOR_DOCS_COLLECTION_NAME):
                general_vendor_results = self.vector_store.query(
                    collection_name=VectorStore.ALL_VENDOR_DOCS_COLLECTION_NAME,
                    query_text=query, # Original query
                    n_results=2
                )
                if general_vendor_results and general_vendor_results.get('documents') and general_vendor_results['documents'][0]:
                    context_results[VectorStore.ALL_VENDOR_DOCS_COLLECTION_NAME] = general_vendor_results
            
            # Optionally, add Hacker News results for general queries if relevant
            # hn_results = self.vector_store.query(VectorStore.HACKER_NEWS_COLLECTION_NAME, query, n_results=1)
            # if hn_results and hn_results.get('documents') and hn_results['documents'][0]:
            #    context_results[VectorStore.HACKER_NEWS_COLLECTION_NAME] = hn_results

        # Log the final context results
        logger.debug(f"Final context results: {context_results}")
        return context_results

    def _is_sufficient(self, context_results: Dict[str, Any]) -> bool:
        """Determine if the retrieved context is sufficient to answer the query"""
        total_documents = 0
        if not context_results: return False

        for collection_name, results_data in context_results.items():
            # ChromaDB query results are nested, documents are in results_data['documents'][0]
            if isinstance(results_data, dict) and \
               'documents' in results_data and \
               isinstance(results_data['documents'], list) and \
               len(results_data['documents']) > 0 and \
               isinstance(results_data['documents'][0], list):
                total_documents += len(results_data['documents'][0])
        
        # If we have at least 1-2 relevant documents, consider it sufficient for now.
        # This threshold might need adjustment.
        is_sufficient = total_documents >= 1 
        logger.debug(f"Context sufficiency check: {total_documents} documents found. Sufficient: {is_sufficient}")
        return is_sufficient


    def _combine_results(self, context_results: Dict[str, Any], web_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine vector store results with web search results into context_results structure."""
        if not web_results:
            return context_results

        # Prepare web results in a format similar to ChromaDB's output for one "collection"
        web_search_collection_key = "web_search_results" # Or VectorStore.WEB_SEARCH_COLLECTION_NAME if defined
        
        web_docs = []
        web_metadatas = []
        web_ids = []
        web_distances = [] # Dummy distances

        for i, result in enumerate(web_results):
            web_docs.append(f"Title: {result.get('title', '')}\nSnippet: {result.get('snippet', '')}\nSource: {result.get('link', '')}")
            web_metadatas.append({
                'title': result.get('title', ''),
                'source': result.get('link', ''),
                'doc_type': 'Web Search Result',
                'vendor': 'web_search' # Generic vendor tag for web results
            })
            web_ids.append(f"web_{i}")
            web_distances.append(1.0) # Assign a dummy distance

        if web_docs:
            context_results[web_search_collection_key] = {
                'documents': [web_docs],
                'metadatas': [web_metadatas],
                'ids': [web_ids],
                'distances': [web_distances]
            }
        return context_results

    def generate_response(self, query: str, topology_context: Optional[Dict[str, Any]] = None) -> str:
        """Generate a response to the user's query with optional topology context and multi-vendor support"""
        try:
            analysis = self.analyze_query(query)
            logger.info(f"Query analysis: {json.dumps(analysis, indent=2)}")

            # ✅ If multi-vendor comparison: loop and collate
            if analysis.get("compare_with_all") and analysis.get("intent") == "comparison":
                full_response = "## Multi-Vendor Comparison Report (Aruba vs Others)\n\n"
                for other_vendor in analysis["other_vendors"]:
                    logger.info(f"Handling comparison between Aruba and {other_vendor}")
                    mini_analysis = {
                        **analysis,
                        "source": "Aruba",
                        "target": other_vendor,
                        "compare_item_1_vendor": "Aruba",
                        "compare_item_2_vendor": other_vendor,
                        "compare_item_1_desc": analysis.get("source_product") or "Aruba",
                        "compare_item_2_desc": other_vendor,
                        "intent": "comparison"
                    }

                    context_results = self.retrieve_relevant_context(query, mini_analysis)

                    # Use web search fallback if needed
                    if not self._is_sufficient(context_results) and self.web_searcher:
                        logger.info("Context insufficient. Performing web search.")
                        search_query_for_web = f"{other_vendor} vs Aruba {query}"
                        web_results = self.web_searcher.search(search_query_for_web)
                        if web_results:
                            logger.info(f"Found {len(web_results)} web results.")
                            context_results = self._combine_results(context_results, web_results)

                    formatted_context = self.llm_service.format_context_for_prompt(context_results)

                    # Inject topology if available
                    if topology_context:
                        topology_str = json.dumps(topology_context, indent=2)
                        formatted_context = (
                            f"CURRENT TOPOLOGY CONTEXT:\n{topology_str}\n\n"
                            f"ADDITIONAL CONTEXT FROM DOCUMENTS:\n{formatted_context}"
                        )

                    prompt = self.llm_service.create_integration_prompt(query, formatted_context, mini_analysis)
                    sub_response = self.llm_service.generate_response(prompt)
                    full_response += f"### Aruba vs {other_vendor}\n{sub_response.strip()}\n\n---\n\n"

                return full_response.strip()

            # ✅ Else: handle single vendor analysis as before
            context_results = self.retrieve_relevant_context(query, analysis)

            if not self._is_sufficient(context_results) and self.web_searcher:
                logger.info("Vector store results insufficient. Performing web search.")
                search_query_for_web = query
                if analysis.get("target"):
                    search_query_for_web = f"{analysis.get('target')} {query}"
                web_results = self.web_searcher.search(search_query_for_web)
                if web_results:
                    logger.info(f"Found {len(web_results)} web results.")
                    context_results = self._combine_results(context_results, web_results)

            formatted_context = self.llm_service.format_context_for_prompt(context_results)

            if topology_context:
                topology_str = json.dumps(topology_context, indent=2)
                formatted_context = (
                    f"CURRENT TOPOLOGY CONTEXT:\n{topology_str}\n\n"
                    f"ADDITIONAL CONTEXT FROM DOCUMENTS:\n{formatted_context}"
                )

            prompt = self.llm_service.create_integration_prompt(query, formatted_context, analysis)
            return self.llm_service.generate_response(prompt)

        except Exception as e:
            logger.error(f"Error in NetworkIntegrationAgent.generate_response: {str(e)}", exc_info=True)
            return "I encountered an error while processing your request. Please try again or rephrase your question."


# Earlier version. Keeping for reference
    # def generate_response(self, query: str) -> str:
    #     """Generate a response to the user's query with web search fallback"""
    #     try:
    #         analysis = self.analyze_query(query)
    #         logger.info(f"Query analysis: {json.dumps(analysis, indent=2)}")

    #         context_results = self.retrieve_relevant_context(query, analysis)
            
    #         if not self._is_sufficient(context_results) and self.web_searcher:
    #             logger.info("Vector store results insufficient or empty. Performing web search.")
    #             # Construct a more specific search query
    #             search_query_for_web = query
    #             if analysis.get("target"):
    #                 search_query_for_web = f"{analysis.get('target')} "
    #                 if analysis.get("target_product"):
    #                     search_query_for_web += f"{analysis.get('target_product')} "
    #                 search_query_for_web += query # Append original query for context
                
    #             web_search_items = self.web_searcher.search(search_query_for_web)
    #             if web_search_items:
    #                 logger.info(f"Found {len(web_search_items)} results from web search.")
    #                 context_results = self._combine_results(context_results, web_search_items)
    #             else:
    #                 logger.info("No results found from web search.")
            
    #         formatted_context = self.llm_service.format_context_for_prompt(context_results)
            
    #         prompt_to_llm = self.llm_service.create_integration_prompt(query, formatted_context, analysis)
            
    #         response = self.llm_service.generate_response(prompt_to_llm)
    #         return response

    #     except Exception as e:
    #         logger.error(f"Error in NetworkIntegrationAgent.generate_response: {str(e)}", exc_info=True)
    #         return "I encountered an error while processing your request. Please try again or rephrase your question."

