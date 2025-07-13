import google.generativeai as genai
import logging
from PIL import Image, ImageEnhance, ImageFilter
import io
import json
import re
import asyncio
import random
from typing import Dict, Any, Optional
from environment import GEMINI_API_KEY

logger = logging.getLogger(__name__)

class GeminiService:
    def __init__(self):
        """Initialize Gemini service with enhanced error handling"""
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        genai.configure(api_key=GEMINI_API_KEY)
        # Use gemini-1.5-flash for higher rate limits
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.max_retries = 3
        logger.info("Gemini service initialized with Flash model")

    def preprocess_image(self, image_data: bytes) -> bytes:
        """Enhance image quality for better AI analysis"""
        try:
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Enhance contrast and sharpness for better text recognition
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.5)
            
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.2)
            
            # Resize if too large (max 2048x2048 for optimal processing)
            max_size = 2048
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Save enhanced image
            output = io.BytesIO()
            image.save(output, format='PNG', optimize=True, quality=95)
            return output.getvalue()
            
        except Exception as e:
            logger.warning(f"Image preprocessing failed: {e}. Using original image.")
            return image_data

    async def analyze_topology_image(self, image_data: bytes, analysis_prompt: str = "") -> str:
        """Analyze network topology image with enhanced prompting and retry logic"""
        try:
            # Preprocess image for better recognition
            enhanced_image_data = self.preprocess_image(image_data)
            image = Image.open(io.BytesIO(enhanced_image_data))
            
            # Create comprehensive analysis prompt
            prompt = self.create_enhanced_analysis_prompt(analysis_prompt)
            
            # Analyze with retry logic for IMAGE analysis
            response_text = await self.analyze_image_with_retry(image, prompt)
            
            return response_text
            
        except Exception as e:
            logger.error(f"Error analyzing topology image: {str(e)}")
            raise Exception(f"Failed to analyze topology: {str(e)}")

    def create_enhanced_analysis_prompt(self, user_context: str = "") -> str:
        """Create detailed prompt for better device detection"""
        return f"""
NETWORK TOPOLOGY ANALYSIS TASK:

Analyze this network diagram image with extreme attention to detail. Extract ALL visible network devices and their specifications.

CRITICAL REQUIREMENTS:
1. Identify EVERY device visible in the image (switches, routers, firewalls, access points, servers, workstations)
2. Extract device labels, model numbers, and vendor information from ANY text visible in the image
3. Map all visible connections between devices (lines, cables, links)
4. Identify network segments, VLANs, or subnets if labeled
5. Note any performance specifications, port counts, or technical details visible

DEVICE DETECTION FOCUS:
- Look for Cisco devices (Catalyst, Nexus, ASR, ISR, C9300, C9500 series)
- Identify Juniper equipment (EX, MX, SRX, QFX, EX4300, EX4600 series)
- Find Aruba/HPE devices (CX, ProCurve, CX6300, CX8360 series)
- Detect Arista switches (DCS, 7050, 7280, 7320 series)
- Note any generic devices, servers, workstations, or unlabeled equipment

USER CONTEXT: {user_context}

OUTPUT FORMAT: Provide response as valid JSON only, no markdown formatting or code blocks:
{{
    "devices": [
        {{
            "id": "unique_device_name_from_image_or_generated",
            "type": "switch|router|firewall|access_point|server|workstation|gateway",
            "vendor": "cisco|juniper|aruba|arista|hpe|generic",
            "model": "exact_model_from_image_text_or_estimated",
            "specifications": {{
                "ports": "port_count_if_visible",
                "speed": "bandwidth_if_indicated",
                "interfaces": "interface_types_if_visible"
            }},
            "connections": ["list_of_connected_device_ids"],
            "location": "position_description_in_topology",
            "role": "core|distribution|access|edge|management"
        }}
    ],
    "topology_structure": "detailed_description_of_network_architecture_and_layout",
    "network_segments": ["identified_vlans_subnets_or_zones"],
    "connection_types": ["ethernet", "fiber", "wireless", "trunk", "access"],
    "deployment_context": "datacenter|campus|branch|wan|hybrid"
}}

CRITICAL: Return ONLY the JSON object, no additional text, explanations, or markdown formatting.
"""

    async def analyze_image_with_retry(self, image: Image.Image, prompt: str) -> str:
        """Analyze image with exponential backoff retry logic - FOR IMAGE ANALYSIS ONLY"""
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Gemini image analysis attempt {attempt + 1}/{self.max_retries}")
                response = self.model.generate_content([prompt, image])
                
                if response and response.text:
                    logger.info(f"Received image analysis response of {len(response.text)} characters")
                    return response.text
                else:
                    raise Exception("Empty response from Gemini")
                    
            except Exception as e:
                error_str = str(e)
                logger.warning(f"Image analysis attempt {attempt + 1} failed: {error_str}")
                
                if "429" in error_str and attempt < self.max_retries - 1:
                    # Rate limit hit - exponential backoff
                    wait_time = (2 ** attempt) + random.uniform(1, 3)
                    logger.info(f"Rate limited. Waiting {wait_time:.1f} seconds...")
                    await asyncio.sleep(wait_time)
                    continue
                elif attempt < self.max_retries - 1:
                    # Other error - shorter wait
                    wait_time = random.uniform(1, 2)
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    # Final attempt failed
                    raise e

    async def generate_replacement_recommendations(self, 
                                                 current_topology: Dict[str, Any], 
                                                 replacement_query: str,
                                                 context_info: str = "") -> str:
        """Generate comprehensive device replacement recommendations - TEXT ONLY"""
        try:
            prompt = f"""
Based on the current network topology and replacement requirements, provide intelligent device replacement recommendations.

Current Topology Analysis:
{json.dumps(current_topology, indent=2)}

User Replacement Requirements:
{replacement_query}

Additional Context from Vector Database and Web Search:
{context_info}

Provide detailed recommendations in valid JSON format (no markdown):
{{
    "replacements": [
        {{
            "original_device": {{
                "id": "device_id",
                "vendor": "current_vendor",
                "model": "current_model",
                "limitations": ["current_limitations"]
            }},
            "recommended_device": {{
                "vendor": "recommended_vendor",
                "model": "specific_model",
                "features": ["enhanced_features"],
                "specifications": {{"ports": "count", "throughput": "capacity"}},
                "justification": "detailed_technical_and_business_reasons",
                "cost_benefit": "cost_analysis_and_benefits",
                "migration_complexity": "low|medium|high"
            }},
            "compatibility_analysis": {{
                "existing_connections": "connection_compatibility",
                "protocol_support": "protocol_compatibility",
                "potential_issues": ["possible_challenges"]
            }}
        }}
    ],
    "topology_modifications": {{
        "structural_changes": "how_topology_will_change",
        "performance_impact": "expected_performance_improvements",
        "security_enhancements": "security_improvements"
    }},
    "implementation_plan": {{
        "phases": [
            {{
                "phase_number": 1,
                "description": "phase_description",
                "duration": "estimated_time",
                "devices_affected": ["device_list"],
                "risk_level": "low|medium|high"
            }}
        ],
        "prerequisites": ["preparation_requirements"],
        "testing_strategy": "validation_approach"
    }},
    "cost_analysis": {{
        "hardware_costs": "equipment_investment",
        "implementation_costs": "professional_services",
        "operational_savings": "ongoing_cost_reductions",
        "total_project_cost": "comprehensive_cost_estimate"
    }}
}}

Return ONLY the JSON object, no additional text or formatting.
"""
            
            # Use TEXT-ONLY generation for recommendations (NO IMAGE)
            response = await self.generate_text_with_retry(prompt)
            return response
            
        except Exception as e:
            logger.error(f"Error generating replacement recommendations: {str(e)}")
            raise Exception(f"Failed to generate recommendations: {str(e)}")

    async def generate_text_with_retry(self, prompt: str) -> str:
        """Generate text-only content with retry logic - NO IMAGE PARAMETER"""
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Gemini text generation attempt {attempt + 1}/{self.max_retries}")
                # TEXT-ONLY generation - no image parameter
                response = self.model.generate_content(prompt)
                
                if response and response.text:
                    logger.info(f"Received text response of {len(response.text)} characters")
                    return response.text
                else:
                    raise Exception("Empty response from Gemini")
                    
            except Exception as e:
                error_str = str(e)
                logger.warning(f"Text generation attempt {attempt + 1} failed: {error_str}")
                
                if "429" in error_str and attempt < self.max_retries - 1:
                    # Rate limit hit - exponential backoff
                    wait_time = (2 ** attempt) + random.uniform(1, 3)
                    logger.info(f"Rate limited. Waiting {wait_time:.1f} seconds...")
                    await asyncio.sleep(wait_time)
                    continue
                elif attempt < self.max_retries - 1:
                    # Other error - shorter wait
                    wait_time = random.uniform(1, 2)
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    # Final attempt failed
                    raise e
