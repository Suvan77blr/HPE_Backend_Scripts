import re
import logging
from typing import Dict, List, Any, Optional
import json

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        self.chunk_size = 1000
        self.chunk_overlap = 200
    
    def extract_product_info(self, content: str) -> Dict[str, str]:
        """Extract product information from document content"""
        metadata = {}
        
        vendor_patterns = {
            "Cisco": r"Cisco|Nexus|Catalyst",
            "Juniper": r"Juniper|MX|EX|SRX|QFX",
            "Aruba": r"Aruba|AOS-CX|CX",
            "Arista": r"Arista|DCS|EOS",
            "HPE": r"HPE|ProCurve|Comware"
        }
        
        for vendor, pattern in vendor_patterns.items():
            if re.search(pattern, content, re.IGNORECASE):
                metadata["vendor"] = vendor
                break
        
        product_patterns = [
            (r"Nexus\s+(\d+)", "Cisco", "Nexus"),
            (r"Catalyst\s+(\d+)", "Cisco", "Catalyst"),
            (r"MX\s+(\d+)", "Juniper", "MX Series"),
            (r"EX\s+(\d+)", "Juniper", "EX Series"),
            (r"SRX\s+(\d+)", "Juniper", "SRX Series"),
            (r"CX\s+(\d+)", "Aruba", "CX Series"),
            (r"(\d+)\s*[A-Z]\s*Switch", "Aruba", "Switch")
        ]
        
        for pattern, vendor, product_line in product_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                metadata["vendor"] = vendor
                metadata["product_line"] = product_line
                metadata["sku"] = match.group(0)
                break
        
        release_match = re.search(r"(Release|Version)\s+([\d\.]+)", content, re.IGNORECASE)
        if release_match:
            metadata["release"] = release_match.group(2)
        
        feature_categories = {
            "switching": r"switch(ing)?|vlan|spanning tree|stp|lacp|trunk",
            "routing": r"rout(ing|er)|ospf|bgp|eigrp|rip|static route",
            "vpn": r"vpn|ipsec|ssl|tunnel",
            "security": r"firewall|acl|access list|authentication|encryption"
        }
        
        detected_categories = []
        for category, pattern in feature_categories.items():
            if re.search(pattern, content, re.IGNORECASE):
                detected_categories.append(category)
        
        if detected_categories:
            metadata["feature_categories"] = ",".join(detected_categories)
        
        deployment_contexts = {
            "datacenter": r"data\s*center|rack|server|virtualization",
            "campus": r"campus|office|building|enterprise",
            "wan": r"wan|wide area network|branch|remote"
        }
        
        detected_contexts = []
        for context, pattern in deployment_contexts.items():
            if re.search(pattern, content, re.IGNORECASE):
                detected_contexts.append(context)
        
        if detected_contexts:
            metadata["deployment"] = ",".join(detected_contexts)
        
        if re.search(r"hardware|physical|port|interface|chassis", content, re.IGNORECASE):
            metadata["feature_type"] = "hardware" if "feature_type" not in metadata else metadata["feature_type"] + ",hardware"
        
        if re.search(r"software|firmware|os|operating system|configuration", content, re.IGNORECASE):
            metadata["feature_type"] = "software" if "feature_type" not in metadata else metadata["feature_type"] + ",software"
        
        return metadata
    
    def chunk_document(self, content: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split document into chunks with metadata"""
        chunks = []
        
        paragraphs = re.split(r'\n\s*\n', content)
        
        current_chunk = ""
        current_size = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            paragraph_size = len(paragraph)
            
            if current_size + paragraph_size > self.chunk_size and current_size > 0:
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_size"] = current_size
                chunks.append({
                    "content": current_chunk,
                    "metadata": chunk_metadata
                })
                
                words = current_chunk.split()
                overlap_words = words[-min(len(words), self.chunk_overlap // 5):]
                current_chunk = " ".join(overlap_words) + "\n\n" + paragraph
                current_size = len(current_chunk)
            else:
                if current_size > 0:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
                current_size += paragraph_size
        
        if current_size > 0:
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_size"] = current_size
            chunks.append({
                "content": current_chunk,
                "metadata": chunk_metadata
            })
        
        return chunks