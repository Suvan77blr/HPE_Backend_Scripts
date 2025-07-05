import json
import logging
import re
import zlib
import base64
import markdown2
from typing import Dict, Any, Optional, List
from gemini_service import GeminiService
from mermaid_generator import MermaidGenerator
from vector_store import VectorStore
from web_search import WebSearcher

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

        # Strategy 5: Create a fallback structure if all else fails
        logger.warning("All JSON parsing strategies failed. Creating fallback structure.")
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
            (r'cx\s*(\d+)', 'aruba', 'switch'),
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
            logger.info(f"Raw agent response for recommendations: {rag_justification_and_recs}")

            # Parse the response to get recommendations
            recommendations = self.parse_gemini_response(rag_justification_and_recs)
            logger.info(f"Parsed recommendations: {recommendations}")

            # If parsing fails or no recommendations are found, generate dynamic fallback recommendations
            if not recommendations.get("replacements"):
                logger.warning("No structured recommendations found in agent response. Generating dynamic fallback recommendations.")
                recommendations = self._create_dynamic_fallback_recommendations(current_topology, replacement_query)

            # Apply the recommendations
            modified_topology = self._apply_recommendations(current_topology, recommendations)

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

    def _create_dynamic_fallback_recommendations(self, topology: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Create completely dynamic fallback recommendations - FIXED None handling"""
        devices = topology.get("devices", [])
        replacements = []
        
        query_lower = (query or "").lower()

        # Get all vendors present in the current topology
        detected_vendors = {(d.get("vendor") or "").lower() for d in devices if d.get("vendor")}
        detected_vendors.discard("generic")
        logger.info(f"Detected vendors in topology: {detected_vendors}")

        source_vendors = []
        target_vendors = []

        # --- New, more robust vendor extraction logic ---
        KNOWN_VENDORS = {"cisco", "juniper", "aruba", "hpe", "dell", "fortinet", "palo alto", "vmware", "meraki"}

        # Strategy 1: Look for explicit 'replace X with Y' patterns
        replace_patterns = [
            r'replace\s+([\w\s-]+?)\s+with\s+([\w\s-]+)',
            r'from\s+([\w\s-]+?)\s+to\s+([\w\s-]+)',
            r'migrate\s+([\w\s-]+?)\s+to\s+([\w\s-]+)',
        ]

        for pattern in replace_patterns:
            matches = re.findall(pattern, query_lower)
            for match in matches:
                potential_source = match[0].strip()
                potential_target = match[1].strip()
                logger.info(f"Fallback regex matched pattern '{pattern}': source='{potential_source}', target='{potential_target}'")

                # Check if the matched source is in the topology
                if any(v in potential_source for v in detected_vendors):
                    source_vendors.append(potential_source)
                    target_vendors.append(potential_target)

        # Strategy 2: If no explicit pattern, look for target vendor keywords
        if not target_vendors:
            logger.info("No explicit 'replace' pattern found. Searching for target vendor keywords.")
            query_words = query_lower.split()
            for i, word in enumerate(query_words):
                if word in ["with", "to", "into"] and i + 1 < len(query_words):
                    potential_target = query_words[i + 1].rstrip('.,')
                    if potential_target in KNOWN_VENDORS:
                        logger.info(f"Found explicit target vendor '{potential_target}' after keyword '{word}'.")
                        target_vendors = [potential_target]
                        source_vendors = [v for v in detected_vendors if v != potential_target]
                        logger.info(f"Inferred source vendors: {source_vendors}")
                        break

        # Strategy 3: Fallback to old logic if still nothing found
        if not source_vendors or not target_vendors:
            logger.warning("Could not determine vendors from explicit patterns. Falling back to inference.")
            vendor_mentions = [v for v in detected_vendors if v in query_lower]
            if vendor_mentions:
                source_vendors = vendor_mentions
                target_vendors = self._infer_target_vendors(source_vendors, query_lower)

        def clean_vendor_list(vendors_phrases, known_set):
            """Extracts known vendor names from phrases."""
            cleaned = set()
            for phrase in vendors_phrases:
                for known in known_set:
                    if re.search(r'\b' + re.escape(known) + r'\b', phrase):
                        cleaned.add(known)
            return list(cleaned)

        # Clean up vendor names to find the actual vendor in the matched phrases
        source_vendors = clean_vendor_list(source_vendors, detected_vendors)
        target_vendors = clean_vendor_list(target_vendors, KNOWN_VENDORS)
        
        logger.info(f"Dynamic vendor detection - Source: {source_vendors}, Target: {target_vendors}")
        
        # Generate replacements dynamically
        for device in devices:
            device_vendor = (device.get("vendor") or "").lower()
            device_type = device.get("type") or ""
            device_id = device.get("id") or ""
            device_model = (device.get("model") or "").lower()
            
            # Check if this device should be replaced
            should_replace = False
            target_vendor = None
            
            # Dynamic matching against source vendors
            for source_vendor in source_vendors:
                if (device_vendor == source_vendor or 
                    source_vendor in device_vendor or 
                    source_vendor in device_model or
                    source_vendor in device_id.lower()):
                    should_replace = True
                    target_vendor = target_vendors[0] if target_vendors else self._infer_single_target_vendor(device_vendor, query_lower)
                    break
            
            if should_replace and target_vendor:
                # Generate dynamic replacement
                replacement_model = self._generate_dynamic_replacement_model(device, target_vendor, device_type)
                features = self._generate_dynamic_features(device, target_vendor, device_type)
                justification = self._generate_dynamic_justification(device, target_vendor, query_lower)
                
                replacements.append({
                    "original_device": {
                        "id": device_id,
                        "vendor": device.get("vendor") or "",
                        "model": device.get("model") or "",
                        "type": device_type
                    },
                    "recommended_device": {
                        "vendor": target_vendor,
                        "model": replacement_model,
                        "features": features,
                        "specifications": device.get("specifications", {}),
                        "justification": justification,
                        "cost_benefit": self._generate_dynamic_cost_benefit(device_vendor, target_vendor),
                        "migration_complexity": self._assess_migration_complexity(device_vendor, target_vendor)
                    }
                })
        
        return {
            "replacements": replacements,
            "topology_modifications": {
                "structural_changes": f"Dynamic device replacements based on query analysis",
                "performance_impact": "Expected improvements based on vendor capabilities",
                "security_enhancements": "Enhanced security features with new vendor ecosystem"
            },
            "implementation_plan": {
                "phases": [{
                    "phase_number": 1,
                    "description": "Dynamic device replacement implementation",
                    "duration": "2-4 weeks",
                    "risk_level": "medium"
                }]
            }
        }

    def _infer_target_vendors(self, source_vendors: List[str], query: str) -> List[str]:
        """Dynamically infer target vendors from query context"""
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
            if vendor in query and vendor not in source_vendors:
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

        device_map = {device['id']: device for device in modified_topology.get("devices", [])}

        for replacement in replacements:
            original_device_info = replacement.get("original_device", {})
            original_id = original_device_info.get("id")
            
            if original_id in device_map:
                recommended_device = replacement.get("recommended_device", {})
                
                # Update device attributes
                device_to_update = device_map[original_id]
                device_to_update['vendor'] = recommended_device.get('vendor', device_to_update.get('vendor'))
                device_to_update['model'] = recommended_device.get('model', device_to_update.get('model'))
                device_to_update['specifications'] = recommended_device.get('specifications', device_to_update.get('specifications'))
                device_to_update['role'] = recommended_device.get('role', 'upgraded') # Default role
                
                # Add a note about the replacement
                device_to_update['notes'] = f"Replaced based on recommendation. Original: {original_device_info.get('vendor')} {original_device_info.get('model')}"
                
                logger.info(f"Applied replacement for device {original_id}")

        return modified_topology
        
        if motivations:
            justification += f" to achieve {', '.join(motivations)}"
        else:
            justification += f" for improved capabilities and vendor standardization"
        
        return justification

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
        """Apply recommendations with COMPLETELY DYNAMIC matching - FIXED None handling"""
        modified_topology = json.loads(json.dumps(original_topology))  # Deep copy
        
        replacements = recommendations.get("replacements", [])
        logger.info(f"Applying {len(replacements)} device replacements with DYNAMIC matching")
        
        if not replacements:
            logger.warning("No replacements found in recommendations")
            return modified_topology
        
        devices_replaced = 0
        
        for replacement in replacements:
            original_device_info = replacement.get("original_device", {})
            recommended_device = replacement.get("recommended_device", {})
            
            if not original_device_info or not recommended_device:
                logger.warning("Skipping replacement due to missing device information")
                continue
            
            # Extract ALL possible matching criteria dynamically - FIXED None handling
            target_criteria = {
                'vendor': (original_device_info.get("vendor") or "").lower().strip(),
                'model': (original_device_info.get("model") or "").lower().strip(),
                'type': (original_device_info.get("type") or "").lower().strip(),
                'id': (original_device_info.get("id") or "").lower().strip()
            }
            
            # Extract new device information - FIXED None handling
            new_vendor = (recommended_device.get("vendor") or "").lower().strip()
            new_model = (recommended_device.get("model") or "").strip()
            new_features = recommended_device.get("features", [])
            justification = recommended_device.get("justification") or ""
            
            logger.info(f"Dynamic search for device matching criteria: {target_criteria}")
            logger.info(f"Replacement target: {new_vendor} {new_model}")
            
            # COMPLETELY DYNAMIC MATCHING: Score-based approach
            best_match = None
            best_score = 0
            
            for device in modified_topology.get("devices", []):
                if device.get("is_replaced"):  # Skip already replaced devices
                    continue
                
                # Calculate dynamic match score
                match_score = self._calculate_dynamic_match_score(device, target_criteria)
                
                if match_score > best_score:
                    best_score = match_score
                    best_match = device
                    logger.info(f"New best match: {device.get('id')} with score {match_score:.2f}")
            
            # Apply replacement if we found a good match (dynamic threshold)
            min_threshold = 0.3  # Configurable threshold
            if best_match and best_score >= min_threshold:
                # Store original information - FIXED None handling
                old_vendor = best_match.get("vendor") or "Unknown"
                old_model = best_match.get("model") or "Unknown"
                old_id = best_match.get("id") or "Unknown"
                
                # APPLY THE REPLACEMENT DYNAMICALLY
                best_match["vendor"] = new_vendor
                best_match["model"] = new_model
                best_match["features"] = new_features
                best_match["replacement_reason"] = justification
                best_match["original_vendor"] = old_vendor
                best_match["original_model"] = old_model
                best_match["original_id"] = old_id
                best_match["is_replaced"] = True
                
                # Generate completely dynamic new ID
                best_match["id"] = self._generate_completely_dynamic_id(new_vendor, new_model, best_match.get("type") or "device")
                
                devices_replaced += 1
                logger.info(f"🎯 DYNAMIC REPLACEMENT SUCCESS: {old_vendor} {old_model} → {new_vendor} {new_model} (score: {best_score:.2f})")
            else:
                logger.warning(f"❌ NO SUITABLE MATCH FOUND: best score {best_score:.2f} below threshold {min_threshold}")
        
        logger.info(f"🔄 TOTAL DEVICES REPLACED: {devices_replaced}")
        
        # Update topology metadata
        modified_topology["devices_replaced_count"] = devices_replaced
        modified_topology["topology_modifications"] = f"Dynamically replaced {devices_replaced} devices based on AI recommendations"
        
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
            device_type = device.get('type', 'Unknown')
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
