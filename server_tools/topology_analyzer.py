import json
import logging
import re
import zlib
import base64
import markdown2
from typing import Dict, Any, Optional, List
from server_tools.gemini_service import GeminiService
from server_tools.mermaid_generator import MermaidGenerator
from server_tools.vector_store import VectorStore
from server_tools.web_search import WebSearcher

logger = logging.getLogger(__name__)

class TopologyAnalyzer:
    def __init__(self, vector_store: VectorStore, web_searcher: Optional[WebSearcher] = None):
        """Initialize topology analyzer with enhanced error handling"""
        self.gemini_service = GeminiService()
        self.mermaid_generator = MermaidGenerator()
        self.vector_store = vector_store
        self.web_searcher = web_searcher

    def _generate_kroki_url(self, mermaid_code: str, output_format: str = "png") -> str:
        """Generates a Kroki.io URL for a given Mermaid diagram."""
        compressed = zlib.compress(mermaid_code.encode('utf-8'))
        encoded = base64.urlsafe_b64encode(compressed).decode('utf-8')
        return f"https://kroki.io/mermaid/{output_format}/{encoded}"

    def parse_gemini_response(self, response_text: str) -> Dict[str, Any]:
        """Robust JSON parsing with multiple fallback strategies"""
        logger.info(f"Parsing Gemini response of {len(response_text)} characters")

        # Strategy 1: Extract JSON from markdown code blocks ```json ... ```
        match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
            try:
                parsed = json.loads(json_str)
                logger.info("Successfully extracted JSON from ```json block")
                return parsed
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON from ```json block: {e}")

        # Strategy 2: Find the first '{' and last '}' and try to parse the content
        try:
            start_index = response_text.find('{')
            end_index = response_text.rfind('}')
            if start_index != -1 and end_index != -1 and end_index > start_index:
                json_str = response_text[start_index : end_index + 1]
                parsed = json.loads(json_str)
                logger.info("Successfully parsed JSON between first '{' and last '}'")
                return parsed
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse content between first '{{' and last '}}': {e}")

        # Strategy 3: Fallback for generic markdown code blocks
        match = re.search(r'```(.*?)```', response_text, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
            if json_str.startswith('json'):
                json_str = json_str[4:].lstrip()
            try:
                parsed = json.loads(json_str)
                logger.info("Successfully extracted JSON from generic ``` block")
                return parsed
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON from generic ``` block: {e}")

        # Strategy 4: Direct parsing of the whole text (as a last resort for clean JSON)
        try:
            parsed = json.loads(response_text.strip())
            logger.info("Successfully parsed JSON directly")
            return parsed
        except json.JSONDecodeError:
            pass

        # Strategy 5: Parse 'Replacement for...' text format
        logger.info("JSON parsing failed, trying to parse 'Replacement for...' format.")
        replacements = []
        # This regex is designed to be flexible with device names
        pattern = re.compile(r'replacement for\s+([\w\d\s.-]+?):\s*(.*)', re.IGNORECASE)
        for line in response_text.splitlines():
            match = pattern.search(line.strip())
            if match:
                original_device_name = match.group(1).strip()
                recommended_model_name = match.group(2).strip()
                
                # The comprehensive apply function will handle the name variations
                replacements.append({
                    "original_device": {"id": original_device_name},
                    "recommended_device": {"model": recommended_model_name}
                })
        
        if replacements:
            logger.info(f"Successfully parsed {len(replacements)} replacements from text format.")
            # Return a structure that the rest of the system expects
            return {
                "replacements": replacements,
                "justification": "Parsed from text-based recommendations.",
                "cost_analysis": "Cost analysis needs to be regenerated based on parsed text.",
                "feature_comparison": "Feature comparison needs to be regenerated based on parsed text."
            }

        # Strategy 6: Create a fallback structure if all else fails
        logger.warning("All parsing strategies failed. Creating fallback structure.")
        return self.create_fallback_structure(response_text)

    def create_fallback_structure(self, text: str) -> Dict[str, Any]:
        """Create structured data from unstructured text"""
        devices = []

        device_patterns = [
            (r'catalyst\s*(\d+)', 'cisco', 'switch'),
            (r'nexus\s*(\d+)', 'cisco', 'switch'),
            (r'c(\d{4})', 'cisco', 'switch'),
            (r'ex(\d+)', 'juniper', 'switch'),
            (r'mx(\d+)', 'juniper', 'router'),
            (r'srx(\d+)', 'juniper', 'firewall'),
            (r'(cx\s*\d+)', 'aruba', 'switch'),  # Captures full model like 'CX 6400'
            (r'aruba\s+(central)', 'aruba', 'management'), # Detects Aruba Central
            (r'aruba\s+(\d{4})\s*(?:series)?\s*router', 'aruba', 'router'), # Flexible router detection
            (r'procurve\s*(\d+)', 'hpe', 'switch'),
            (r'dcs[-\s]*(\d+)', 'arista', 'switch'),
            (r'(\d+[A-Z]+\d*)', 'generic', 'switch')
        ]

        device_id_counter = 1
        found_devices = set()

        for pattern, vendor, device_type in device_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                device_name = f"{vendor}_{device_type}_{match}"
                if device_name not in found_devices:
                    devices.append({
                        "id": f"device_{device_id_counter}",
                        "type": device_type,
                        "vendor": vendor,
                        "model": f"{vendor.upper()} {match}",
                        "specifications": {"ports": "unknown", "speed": "unknown"},
                        "connections": [],
                        "location": "detected_in_text",
                        "role": "unknown"
                    })
                    found_devices.add(device_name)
                    device_id_counter += 1

        if not devices:
            logger.warning("No devices detected in text. Creating generic structure.")
            devices = [
                {
                    "id": "generic_device_1",
                    "type": "switch",
                    "vendor": "generic",
                    "model": "Unknown Switch",
                    "specifications": {"ports": "unknown"},
                    "connections": [],
                    "location": "topology_center",
                    "role": "core"
                }
            ]

        return {
            "devices": devices,
            "topology_structure": "Hierarchical network topology",
            "network_segments": ["management", "production"],
            "connection_types": ["ethernet", "trunk"],
            "deployment_context": "enterprise"
        }

    async def analyze_and_replace_topology(self, image_data: bytes, replacement_query: str, agent) -> Dict[str, Any]:
        """Main method with enhanced error handling and logging"""
        try:
            logger.info(f"Starting topology analysis - Image size: {len(image_data)} bytes")

            topology_analysis = await self.gemini_service.analyze_topology_image(
                image_data,
                "Focus on identifying all network devices, their types, models, and interconnections"
            )

            current_topology = self.parse_gemini_response(topology_analysis)
            devices_count = len(current_topology.get("devices", []))
            logger.info(f"Successfully parsed topology with {devices_count} devices")

            # Generate the RAG-powered justification which includes recommendations
            rag_justification_and_recs = agent.generate_response(replacement_query, current_topology)

            # --- CRITICAL DEBUG LOG ---
            logger.info(f"--- FULL RAW RAG RESPONSE ---\n{rag_justification_and_recs}\n--- END RAW RAG RESPONSE ---")
            # --- END CRITICAL DEBUG LOG ---

            initial_topology = current_topology
            modified_topology = initial_topology # Default to original

            # Determine which recommendations to use
            recommendations = self.parse_gemini_response(rag_justification_and_recs)
            final_recommendations = recommendations
            if not final_recommendations or not final_recommendations.get("replacements"):
                logger.warning("No structured replacements found. Attempting dynamic fallback.")
                final_recommendations = self._create_dynamic_fallback_recommendations(initial_topology, rag_justification_and_recs)

            # Apply the chosen recommendations if they exist
            if final_recommendations and final_recommendations.get("replacements"):
                logger.info(f"Applying {len(final_recommendations.get('replacements', []))} replacements using comprehensive logic.")
                modified_topology = self._apply_comprehensive_recommendations(initial_topology, final_recommendations)
            else:
                logger.warning("No replacements to apply, either from structured response or fallback.")

            original_diagram = self.mermaid_generator.generate_network_diagram(
                current_topology, "Original Network Topology"
            )

            modified_mermaid = self.mermaid_generator.generate_network_diagram(modified_topology, "Proposed Topology")
            comparison_mermaid = self.mermaid_generator.generate_comparison_diagram(current_topology, modified_topology)

            # Generate Kroki URLs for download
            proposed_png_url = self._generate_kroki_url(modified_mermaid, "png")
            proposed_svg_url = self._generate_kroki_url(modified_mermaid, "svg")

            # Convert RAG context to HTML for proper frontend rendering
            context_html = markdown2.markdown(
                rag_justification_and_recs,
                extras=["tables", "fenced-code-blocks", "strike"]
            )

            return {
                "success": True,
                "original_topology": current_topology,
                "modified_topology": modified_topology,
                "recommendations": recommendations,
                "diagrams": {
                    "original": original_diagram,
                    "modified": modified_mermaid,
                    "comparison": comparison_mermaid,
                    "proposed_png_url": proposed_png_url,
                    "proposed_svg_url": proposed_svg_url
                },
                "analysis_summary": "Analysis completed successfully",
                "topology_explanation": "Network topology analyzed and recommendations generated",
                "context_sources": context_html,
                "modification_details": {"total_replacements": len(recommendations.get("replacements", []))},
                "implementation_guidance": {"phases": []},
            }

        except Exception as e:
            logger.error(f"Error in topology analysis: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "diagrams": {"original": "", "modified": "", "comparison": ""},
                "analysis_summary": f"Analysis failed: {str(e)}"
            }

    def _generate_identifier_variants(self, device: Dict[str, Any]) -> List[str]:
        """
        Generate all reasonable identifier variants for flexible matching:
        - Original ID/model/label
        - Lowercase
        - Remove underscores
        - Remove spaces
        - Concatenated forms
        """
        id_ = device.get("id", "") or ""
        model = device.get("model", "") or ""
        label = device.get("label", id_) or ""
        variants = set()

        for val in [id_, model, label]:
            if not val:
                continue
            v = str(val)
            variants.add(v)
            variants.add(v.lower())
            variants.add(v.replace("_", ""))
            variants.add(v.replace("_", " ").lower())
            variants.add(v.replace("_", ""))
            variants.add(v.replace(" ", ""))
            variants.add(v.replace(" ", "").lower())
            variants.add(v.replace("_", "").lower())
        # Remove empty strings
        return [v for v in variants if v]

    def _parse_replacement_model_from_text(self, device: Dict[str, Any], text: str) -> Optional[str]:
        """
        Parses the recommended device model from a structured RAG response.
        Tries multiple device identifier variants to find the
        structured replacement line: `Replacement for <identifier>: <model_name>`.
        """
        identifier_variants = self._generate_identifier_variants(device)
        logger.info(f"[PARSER-FLEX] Trying to find replacement for device '{device.get('id')}' using variants: {identifier_variants}")

        clean_text = re.sub(r'[*_`]', '', text)

        # Create a normalized set of variants for robust matching
        # Normalization: lowercase, remove spaces and underscores
        normalized_variants = {re.sub(r'[\s_]', '', v.lower()) for v in identifier_variants if v}

        # General pattern to find any 'Replacement for' line
        general_pattern = re.compile(r"Replacement for\s+([\w\d\s._-]+?)\s*:\s*(.+)", re.IGNORECASE)

        try:
            for line in clean_text.splitlines():
                match = general_pattern.search(line)
                if match:
                    found_identifier = match.group(1).strip()
                    model_name = match.group(2).strip()

                    # Normalize the identifier found in the text for comparison
                    normalized_found = re.sub(r'[\s_]', '', found_identifier.lower())

                    if normalized_found in normalized_variants:
                        logger.info(f"[PARSER-FLEX] SUCCESS: Matched '{found_identifier}' to device '{device.get('id')}'. Found model: '{model_name}'")
                        return model_name
            
            logger.warning(f"[PARSER-FLEX] FAILED: Could not find a structured replacement for device '{device.get('id')}'.")
            return None
        except Exception as e:
            logger.error(f"[PARSER-FLEX] An unexpected error occurred during parsing for '{device.get('id')}': {e}", exc_info=True)
            return None

    def _create_dynamic_fallback_recommendations(self, topology: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Create completely dynamic fallback recommendations based on query analysis."""
        devices = topology.get("devices", [])
        replacements = []
        query_lower = (query or "").lower()

        detected_vendors = {d.get("vendor", "").lower() for d in devices if d.get("vendor")}
        detected_vendors.discard("generic")
        logger.info(f"Detected vendors in topology: {detected_vendors}")

        source_vendors, target_vendors = self._determine_vendor_replacement_scope(query_lower, detected_vendors)
        logger.info(f"Dynamic vendor detection - Source: {source_vendors}, Target: {target_vendors}")

        for device in devices:
            device_vendor = (device.get("vendor") or "").lower()
            device_id = device.get("id", "")
            device_model = (device.get("model") or "").lower()

            should_replace, target_vendor = self._check_device_replacement_necessity(
                device, source_vendors, target_vendors, query_lower
            )

            if should_replace and target_vendor:
                replacement_model = self._parse_replacement_model_from_text(device, query)
                if not replacement_model:
                    logger.warning(f"Could not parse specific model for '{device_id}'. Creating descriptive fallback.")
                    replacement_model = f"ARUBA Replacement for {device.get('model', 'device')}"

                replacements.append(self._build_replacement_object(device, target_vendor, replacement_model, query_lower))

        return {
            "replacements": replacements,
            "topology_modifications": {
                "structural_changes": "Dynamic device replacements based on query analysis.",
                "performance_impact": "Expected improvements based on vendor capabilities.",
                "security_enhancements": "Enhanced security features with new vendor ecosystem."
            },
            "implementation_plan": {
                "phases": [{
                    "phase_number": 1,
                    "description": "Dynamic device replacement implementation.",
                    "duration": "2-4 weeks",
                    "risk_level": "medium"
                }]
            }
        }

    def _determine_vendor_replacement_scope(self, query_lower: str, detected_vendors: set) -> (List[str], List[str]):
        """Determine the source and target vendors for replacement from the query."""
        source_vendors, target_vendors = [], []
        KNOWN_VENDORS = {"cisco", "juniper", "aruba", "hpe", "dell", "fortinet", "palo alto", "vmware", "meraki", "arista"}

        # Handle common typos
        query_lower = query_lower.replace("ciso", "cisco")

        replace_patterns = [
            r'replace\s+([,\w\s-]+?)\s+with\s+([,\w\s-]+)',
            r'from\s+([,\w\s-]+?)\s+to\s+([,\w\s-]+)',
            r'migrate\s+([,\w\s-]+?)\s+to\s+([,\w\s-]+)',
        ]

        explicit_replace_found = False
        for pattern in replace_patterns:
            matches = re.findall(pattern, query_lower)
            if matches:
                explicit_replace_found = True
                for match in matches:
                    source, target = match[0].strip(), match[1].strip()
                    source_vendors.append(source)
                    target_vendors.append(target)

        if not explicit_replace_found:
            # Fallback for implicit queries. Only act if source and target are clear.
            mentioned_vendors = {v for v in KNOWN_VENDORS if re.search(r'\b' + v + r'\b', query_lower, re.IGNORECASE)}
            potential_target = None
            query_words = query_lower.split()
            for i, word in enumerate(query_words):
                if word in ["with", "to", "into"] and i + 1 < len(query_words):
                    next_word = query_words[i + 1].rstrip('.,')
                    if next_word in mentioned_vendors:
                        potential_target = next_word
                        break

            if potential_target:
                # We have a target. The sources are the *other* mentioned vendors.
                potential_sources = mentioned_vendors - {potential_target}
                if potential_sources:
                    target_vendors = [potential_target]
                    source_vendors = list(potential_sources)
            else:
                # No explicit target. Infer one if sources are mentioned.
                if mentioned_vendors:
                    source_vendors = list(mentioned_vendors)
                    target_vendors = self._infer_target_vendors(source_vendors, query_lower)

        def clean_vendor_list(phrases, known_set):
            """Cleans a list of phrases to extract known vendor names."""
            cleaned = set()
            for phrase in phrases:
                # Also split by common conjunctions
                sub_phrases = re.split(r'\s+and\s+|,', phrase)
                for sub_phrase in sub_phrases:
                    sub_phrase = sub_phrase.strip()
                    if not sub_phrase:
                        continue
                    for known in known_set:
                        if re.search(r'\b' + re.escape(known) + r'\b', sub_phrase, re.IGNORECASE):
                            cleaned.add(known)
            return list(cleaned)

        return clean_vendor_list(source_vendors, KNOWN_VENDORS), clean_vendor_list(target_vendors, KNOWN_VENDORS)

    def _check_device_replacement_necessity(self, device: Dict[str, Any], source_vendors: List[str], target_vendors: List[str], query_lower: str) -> (bool, Optional[str]):
        """Check if a device should be replaced based on vendor matching."""
        device_vendor = (device.get("vendor") or "").lower()
        device_id = (device.get("id") or "").lower()
        device_model = (device.get("model") or "").lower()

        for source_vendor in source_vendors:
            if source_vendor is None:
                continue
            if (source_vendor in device_vendor) or \
               (source_vendor in device_model) or \
               (source_vendor in device_id):
                target_vendor = target_vendors[0] if target_vendors else self._infer_single_target_vendor(device_vendor, query_lower)
                return True, target_vendor
        return False, None

    def _build_replacement_object(self, device: Dict[str, Any], target_vendor: str, replacement_model: str, query_lower: str) -> Dict[str, Any]:
        """Construct the dictionary for a single device replacement."""
        return {
            "original_device": {
                "id": device.get("id"),
                "vendor": device.get("vendor"),
                "model": device.get("model")
            },
            "recommended_device": {
                "vendor": target_vendor.capitalize(),
                "model": replacement_model,
                "features": self._generate_dynamic_features(device, target_vendor, device.get("type", "")),
                "specifications": device.get("specifications", {}),
                "justification": self._generate_dynamic_justification(device, target_vendor, query_lower),
                "cost_benefit": self._generate_dynamic_cost_benefit((device.get("vendor") or "").lower(), target_vendor),
                "migration_complexity": self._assess_migration_complexity((device.get("vendor") or "").lower(), target_vendor)
            }
        }

    def _infer_target_vendors(self, source_vendors: List[str], query: str) -> List[str]:
        """Dynamically infer target vendors from query context"""
        safe_query = (query or "").lower()  # Ensure query is a safe, lowercase string

        # Common vendor alternatives (dynamically built)
        vendor_alternatives = {
            'cisco': ['aruba', 'juniper', 'arista', 'fortinet'],
            'aruba': ['cisco', 'juniper', 'extreme'],
            'juniper': ['cisco', 'aruba', 'arista'],
            'arista': ['cisco', 'juniper'],
            'fortinet': ['cisco', 'palo alto'],
            'extreme': ['aruba', 'cisco']
        }
        
        # Look for vendor mentions in query
        all_vendors = set()
        for alternatives in vendor_alternatives.values():
            all_vendors.update(alternatives)
        
        mentioned_vendors = []
        for vendor in all_vendors:
            if vendor in safe_query and vendor not in source_vendors:
                mentioned_vendors.append(vendor)
        
        if mentioned_vendors:
            return mentioned_vendors
        
        # Default to most common alternatives
        targets = []
        for source in source_vendors:
            alternatives = vendor_alternatives.get(source, ['cisco'])
            targets.extend(alternatives[:1])  # Take first alternative
        
        return list(set(targets))

    def _infer_single_target_vendor(self, source_vendor: str, query: str) -> str:
        """Infer single target vendor"""
        targets = self._infer_target_vendors([source_vendor], query)
        return targets[0] if targets else 'cisco'

    def _generate_dynamic_replacement_model(self, device: Dict[str, Any], target_vendor: str, device_type: str) -> str:
        """Generate appropriate replacement model based on device characteristics - FIXED None handling"""
        device_model = (device.get("model") or "").lower()
        device_id = (device.get("id") or "").lower()
        
        # Ensure target_vendor is not None
        target_vendor = target_vendor or "generic"
        device_type = device_type or "device"
        
        # Dynamic model generation based on device characteristics
        if device_type == "switch":
            if any(keyword in device_id or keyword in device_model for keyword in ['core', 'catalyst', '9500', '8400']):
                return f"{target_vendor.title()} Core Switch Series"
            elif any(keyword in device_id or keyword in device_model for keyword in ['access', '2960', 'edge']):
                return f"{target_vendor.title()} Access Switch Series"
            else:
                return f"{target_vendor.title()} Switch Series"
        
        elif device_type == "router":
            return f"{target_vendor.title()} Router Series"
        
        elif device_type == "firewall":
            return f"{target_vendor.title()} Security Appliance"
        
        # Default
        return f"{target_vendor.title()} {device_type.title()}"

    def _generate_dynamic_features(self, device: Dict[str, Any], target_vendor: str, device_type: str) -> List[str]:
        """Generate appropriate features based on vendor and device type"""
        features = []
        
        # Base features for device types
        if device_type == "switch":
            features.extend(["Layer 2/3 switching", "VLAN support", "Link aggregation"])
        elif device_type == "router":
            features.extend(["Advanced routing", "VPN support", "QoS"])
        elif device_type == "firewall":
            features.extend(["Stateful inspection", "VPN", "Intrusion prevention"])
        
        # Add vendor-agnostic enhancements
        features.extend([
            "Centralized management",
            "Enhanced security",
            "Performance optimization",
            "Simplified operations"
        ])
        
        return features

    def _generate_dynamic_justification(self, device: Dict[str, Any], target_vendor: str, query: str) -> str:
        """Generate contextual justification based on query and device - FIXED None handling"""
        device_vendor = device.get("vendor") or "Unknown"
        device_model = device.get("model") or "Unknown"
        target_vendor = target_vendor or "Generic"
        query = query or ""
        
        # Extract motivations from query
        motivations = []
        if any(word in query for word in ['cost', 'budget', 'save', 'cheaper']):
            motivations.append("cost optimization")
        if any(word in query for word in ['performance', 'speed', 'faster', 'better']):
            motivations.append("performance improvement")
        if any(word in query for word in ['management', 'unified', 'centralized', 'simple']):
            motivations.append("unified management")
        if any(word in query for word in ['security', 'secure', 'protection']):
            motivations.append("enhanced security")
        if any(word in query for word in ['modern', 'upgrade', 'latest', 'new']):
            motivations.append("technology modernization")
        
        # Build justification
        justification = f"Replace {device_vendor} {device_model} with {target_vendor.title()} equivalent"
        if motivations:
            justification += f" for {', '.join(motivations)}."
        else:
            justification += "."
        
        return justification

    def _apply_recommendations(self, topology: Dict[str, Any], recommendations: Dict[str, Any]) -> Dict[str, Any]:
        """Apply replacement recommendations to the topology"""
        modified_topology = json.loads(json.dumps(topology))  # Deep copy
        
        replacements = recommendations.get("replacements", [])
        if not replacements:
            logger.info("No replacements to apply.")
            return modified_topology
        
        # Create a map of device IDs to their index in the devices list
        device_id_to_index = {device['id']: i for i, device in enumerate(modified_topology.get("devices", []))}

        for replacement in replacements:
            original_device_info = replacement.get("original_device", {})
            original_id = original_device_info.get("id")
            
            if original_id in device_id_to_index:
                recommended_device_info = replacement.get("recommended_device", {})
                
                # Get the index of the device to be replaced
                device_index = device_id_to_index[original_id]
                device_to_update = modified_topology["devices"][device_index]

                # Store original details before updating
                original_vendor = device_to_update.get('vendor')
                original_model = device_to_update.get('model')

                # Update the device with new information
                device_to_update['vendor'] = recommended_device_info.get('vendor', original_vendor)
                device_to_update['model'] = recommended_device_info.get('model', original_model)
                device_to_update['specifications'] = recommended_device_info.get('specifications', device_to_update.get('specifications', {}))
                device_to_update['role'] = recommended_device_info.get('role', device_to_update.get('role', 'upgraded'))
                
                # Add metadata about the replacement for clarity
                device_to_update['is_replaced'] = True
                device_to_update['original_vendor'] = original_vendor
                device_to_update['original_model'] = original_model
                device_to_update['replacement_reason'] = recommended_device_info.get('justification', 'General recommendation.')

                logger.info(f"Applied replacement for device {original_id}")
            else:
                logger.warning(f"Device with ID '{original_id}' not found in topology for replacement.")

        return modified_topology

    def _generate_dynamic_cost_benefit(self, source_vendor: str, target_vendor: str) -> str:
        """Generate cost-benefit analysis based on vendor transition - FIXED None handling"""
        source_vendor = (source_vendor or "Unknown").title()
        target_vendor = (target_vendor or "Unknown").title()
        return f"Migration from {source_vendor} to {target_vendor} ecosystem provides operational efficiency gains and potential cost optimization"

    def _assess_migration_complexity(self, source_vendor: str, target_vendor: str) -> str:
        """Assess migration complexity based on vendor compatibility - FIXED None handling"""
        source_vendor = source_vendor or ""
        target_vendor = target_vendor or ""
        
        # Simple complexity assessment
        if source_vendor.lower() == target_vendor.lower():
            return 'low'
        else:
            return 'medium'

    def _apply_comprehensive_recommendations(self, original_topology: Dict[str, Any],
                                       recommendations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply recommendations with COMPLETELY DYNAMIC matching.
        This version ensures all original devices are carried over to the modified topology.
        """
        # Start with a deep copy to preserve all other topology data
        modified_topology = json.loads(json.dumps(original_topology))

        replacements = recommendations.get("replacements", [])
        if not replacements:
            logger.warning("No replacements found in recommendations")
            return modified_topology

        # Create a list of devices that have already been matched to a replacement
        matched_device_ids = set()

        # Process each device from the original topology
        new_devices = []
        for device in original_topology.get("devices", []):
            device_copy = device.copy()  # Work with a copy

            # Find the best replacement recommendation for the current device
            best_match_score = 0.3  # Minimum threshold for a match
            best_replacement = None

            for replacement in replacements:
                original_device_info = replacement.get("original_device", {})

                target_criteria = {
                    'vendor': (original_device_info.get("vendor") or "").lower().strip(),
                    'model': (original_device_info.get("model") or "").lower().strip(),
                    'type': (original_device_info.get("type") or "").lower().strip(),
                    'id': (original_device_info.get("id") or "").lower().strip()
                }

                match_score = self._calculate_dynamic_match_score(device_copy, target_criteria)

                if match_score > best_match_score:
                    best_match_score = match_score
                    best_replacement = replacement

            # If a suitable replacement is found, apply it
            if best_replacement:
                recommended_device = best_replacement.get("recommended_device", {})
                new_vendor = (recommended_device.get("vendor") or "").lower().strip()
                new_model = (recommended_device.get("model") or "").strip()

                # Store original details
                device_copy['original_vendor'] = device_copy.get("vendor", "Unknown")
                device_copy['original_model'] = device_copy.get("model", "Unknown")
                device_copy['is_replaced'] = True
                device_copy['replacement_reason'] = recommended_device.get("justification", "Recommended upgrade")

                # Apply new details
                device_copy['vendor'] = new_vendor
                device_copy['model'] = new_model
                device_copy['features'] = recommended_device.get("features", [])

                logger.info(f"Device {device_copy.get('id')} marked for replacement with {new_vendor} {new_model}")

            new_devices.append(device_copy)

        modified_topology['devices'] = new_devices
        modified_topology["devices_replaced_count"] = sum(1 for d in new_devices if d.get("is_replaced"))

        logger.info(f"Comprehensive recommendation application complete. Total devices in new topology: {len(new_devices)}")
        return modified_topology

    def _calculate_dynamic_match_score(self, device: Dict[str, Any], target_criteria: Dict[str, str]) -> float:
        """Calculate match score using COMPLETELY DYNAMIC criteria - FIXED None handling"""
        score = 0.0
        max_score = 0.0
        
        device_data = {
            'vendor': (device.get("vendor") or "").lower().strip(),
            'model': (device.get("model") or "").lower().strip(),
            'type': (device.get("type") or "").lower().strip(),
            'id': (device.get("id") or "").lower().strip()
        }
        
        # Dynamic scoring for each criterion
        for criterion, target_value in target_criteria.items():
            if not target_value:  # Skip empty criteria
                continue
                
            max_score += 1.0  # Each criterion can contribute max 1.0 to score
            device_value = device_data.get(criterion, "")
            
            if not device_value:
                continue
            
            # Exact match gets full score
            if device_value == target_value:
                score += 1.0
                logger.debug(f"Exact match on {criterion}: {device_value}")
            
            # Partial match gets proportional score
            elif target_value in device_value or device_value in target_value:
                partial_score = 0.7
                score += partial_score
                logger.debug(f"Partial match on {criterion}: {device_value} ~ {target_value}")
            
            # Fuzzy match for similar strings
            elif self._dynamic_fuzzy_match(device_value, target_value):
                fuzzy_score = 0.5
                score += fuzzy_score
                logger.debug(f"Fuzzy match on {criterion}: {device_value} ~ {target_value}")
        
        # Normalize score
        normalized_score = score / max_score if max_score > 0 else 0
        logger.debug(f"Device {device.get('id')} match score: {normalized_score:.2f} ({score}/{max_score})")
        
        return normalized_score

    def _dynamic_fuzzy_match(self, str1: str, str2: str) -> bool:
        """Completely dynamic fuzzy string matching"""
        if not str1 or not str2:
            return False
        
        # Tokenize and compare
        tokens1 = set(str1.lower().split())
        tokens2 = set(str2.lower().split())
        
        if not tokens1 or not tokens2:
            return False
        
        # Calculate Jaccard similarity
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        
        similarity = intersection / union if union > 0 else 0
        return similarity > 0.3  # Dynamic threshold

    def _generate_completely_dynamic_id(self, vendor: str, model: str, device_type: str) -> str:
        """Generate device ID completely dynamically - FIXED None handling"""
        import re
        
        # Clean inputs - handle None values
        vendor = vendor or "device"
        model = model or ""
        device_type = device_type or "device"
        
        vendor_clean = re.sub(r'[^a-zA-Z0-9]', '', vendor.lower())
        model_clean = re.sub(r'[^a-zA-Z0-9]', '', model.lower()) if model else device_type
        
        # Generate ID from available information
        if vendor_clean and model_clean:
            return f"{vendor_clean}_{model_clean}"
        elif vendor_clean:
            return f"{vendor_clean}_{device_type}"
        else:
            return f"new_{device_type}"

    async def _get_comprehensive_device_context(self, topology: Dict[str, Any], query: str) -> str:
        """Get comprehensive context about devices from multiple sources"""
        context_parts = []
        
        devices = topology.get("devices", [])
        for device in devices:
            vendor = device.get("vendor", "")
            model = device.get("model", "")
            device_type = device.get("type", "")
            
            if vendor and (model or device_type):
                search_queries = [
                    f"{vendor} {model} specifications features",
                    f"{vendor} {device_type} replacement alternatives"
                ]
                
                for search_query in search_queries:
                    # Search vector database
                    try:
                        collection_names = [f"{vendor.lower()}_docs", "all_vendor_docs"]
                        for collection_name in collection_names:
                            try:
                                results = self.vector_store.query(
                                    collection_name=collection_name,
                                    query_text=search_query,
                                    n_results=2
                                )
                                
                                if results and results.get('documents') and results['documents']:
                                    for doc in results['documents']:
                                        if isinstance(doc, list) and doc:
                                            context_parts.append(f"Vector DB ({vendor}): {doc[0][:400]}...")
                                        elif isinstance(doc, str):
                                            context_parts.append(f"Vector DB ({vendor}): {doc[:400]}...")
                                    break
                            except Exception as e:
                                logger.warning(f"Vector search failed for {collection_name}: {e}")
                                continue
                                
                    except Exception as e:
                        logger.warning(f"Vector database search failed for {vendor}: {str(e)}")
                    
                    # Web search fallback
                    if self.web_searcher and len(context_parts) < 5:
                        try:
                            web_results = self.web_searcher.search(search_query)
                            for result in web_results[:1]:
                                context_parts.append(f"Web: {result['title']} - {result['snippet']}")
                        except Exception as e:
                            logger.warning(f"Web search failed: {str(e)}")
        
        return "\n\n".join(context_parts) if context_parts else "No additional context found"

    def _generate_comprehensive_summary(self, original_topology: Dict[str, Any], 
                                      recommendations: Dict[str, Any],
                                      modified_topology: Dict[str, Any]) -> str:
        """Generate comprehensive human-readable summary"""
        summary_parts = []
        
        # Executive Summary
        original_devices = original_topology.get("devices", [])
        replacements = recommendations.get("replacements", [])
        devices_replaced = modified_topology.get("devices_replaced_count", 0)
        
        summary_parts.append("🎯 EXECUTIVE SUMMARY")
        summary_parts.append("=" * 50)
        summary_parts.append(f"Network Analysis: {len(original_devices)} devices analyzed")
        summary_parts.append(f"Recommended Changes: {len(replacements)} device replacements")
        summary_parts.append(f"Devices Actually Replaced: {devices_replaced}")
        summary_parts.append(f"Deployment Context: {original_topology.get('deployment_context', 'enterprise')}")
        
        summary_parts.append("")
        
        # Current Network Analysis
        summary_parts.append("📊 CURRENT NETWORK ANALYSIS")
        summary_parts.append("=" * 50)
        summary_parts.append(f"Total Infrastructure: {len(original_devices)} network devices")
        
        # Device breakdown
        device_types = {}
        for device in original_devices:
            device_type = (device.get('type') or 'Unknown').lower()
            device_types[device_type] = device_types.get(device_type, 0) + 1
        
        summary_parts.append("Device Breakdown:")
        for device_type, count in device_types.items():
            summary_parts.append(f"  -  {device_type.title()}: {count} devices")
        
        summary_parts.append("")
        
        # Replacement Summary
        if devices_replaced > 0:
            summary_parts.append("🔄 DEVICE REPLACEMENTS APPLIED")
            summary_parts.append("=" * 50)
            
            modified_devices = modified_topology.get("devices", [])
            for device in modified_devices:
                if device.get("is_replaced"):
                    original_vendor = device.get("original_vendor", "Unknown")
                    original_model = device.get("original_model", "Unknown")
                    new_vendor = device.get("vendor", "Unknown")
                    new_model = device.get("model", "Unknown")
                    
                    summary_parts.append(f"✅ {device.get('id', 'Unknown')}")
                    summary_parts.append(f"   Original: {original_vendor} {original_model}")
                    summary_parts.append(f"   Replaced with: {new_vendor} {new_model}")
                    summary_parts.append(f"   Reason: {device.get('replacement_reason', 'Not specified')}")
                    summary_parts.append("")
        else:
            summary_parts.append("⚠️ NO DEVICE REPLACEMENTS APPLIED")
            summary_parts.append("=" * 50)
            summary_parts.append("No devices were replaced in the modified topology.")
            summary_parts.append("This may indicate issues with the replacement algorithm or recommendations.")
            summary_parts.append("")
        
        return "\n".join(summary_parts)

    def _generate_topology_explanation(self, original_topology: Dict[str, Any], 
                                     modified_topology: Dict[str, Any]) -> str:
        """Generate detailed topology explanation"""
        explanation_parts = []
        
        explanation_parts.append("NETWORK TOPOLOGY ANALYSIS AND MODIFICATIONS")
        explanation_parts.append("=" * 60)
        explanation_parts.append("")
        
        # Count replaced devices
        modified_devices = modified_topology.get("devices", [])
        replaced_count = sum(1 for d in modified_devices if d.get('is_replaced'))
        
        if replaced_count > 0:
            explanation_parts.append("TOPOLOGY MODIFICATIONS APPLIED:")
            explanation_parts.append(f"Successfully replaced {replaced_count} devices in the network topology.")
            explanation_parts.append("")
            
            for device in modified_devices:
                if device.get('is_replaced'):
                    explanation_parts.append(f"-  {device.get('id')}: {device.get('replacement_reason')}")
        else:
            explanation_parts.append("NO TOPOLOGY MODIFICATIONS:")
            explanation_parts.append("The modified topology is identical to the original topology.")
            explanation_parts.append("No device replacements were applied.")
        
        return "\n".join(explanation_parts)

    def _extract_modification_details(self, recommendations: Dict[str, Any]) -> Dict[str, Any]:
        """Extract modification details for frontend display"""
        return {
            "total_replacements": len(recommendations.get("replacements", [])),
            "topology_changes": recommendations.get("topology_modifications", {}),
            "implementation_phases": len(recommendations.get("implementation_plan", {}).get("phases", [])),
            "risk_level": "medium",
            "estimated_timeline": "2-4 weeks",
            "cost_category": "medium"
        }

    def _extract_implementation_guidance(self, recommendations: Dict[str, Any]) -> Dict[str, Any]:
        """Extract implementation guidance for frontend display"""
        impl_plan = recommendations.get("implementation_plan", {})
        return {
            "phases": impl_plan.get("phases", []),
            "prerequisites": impl_plan.get("prerequisites", []),
            "testing_strategy": impl_plan.get("testing_strategy", ""),
            "rollback_procedures": [],
            "success_criteria": []
        }
