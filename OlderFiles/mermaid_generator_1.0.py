# from BigBro.
import json
import logging
import re
from typing import Dict, Any, List, Optional, Callable

logger = logging.getLogger(__name__)

class MermaidGenerator:
    def __init__(self):
        self.device_icons = {
            "router": "🔀",
            "switch": "🔄",
            "firewall": "🛡️",
            "server": "💽",
            "workstation": "💻",
            "access_point": "📡",
            "load_balancer": "⚖️",
            "storage": "💿",
            "gateway": "🚪"
        }

        self.vendor_colors = {
            "cisco": "#1BA0D7",
            "juniper": "#84BD00",
            "aruba": "#FF6900",
            "arista": "#F80000",
            "hpe": "#01A982",
            "fortinet": "#E31837",
            "extreme": "#8B0000",
            "dell": "#007DB8",
            "generic": "#6C757D"
        }

    def generate_network_diagram(self, topology_data: Dict[str, Any], title: str = "Network Topology") -> str:
        try:
            devices = topology_data.get("devices", [])
            if not devices:
                return self._generate_empty_diagram(title)

            mermaid_code = f"""---
title: {title}
---
graph TD
    %% Vendor-specific styling
"""

            detected_vendors = {d.get("vendor", "generic").lower() for d in devices}
            for vendor in detected_vendors:
                color = self.vendor_colors.get(vendor, self.vendor_colors["generic"])
                mermaid_code += f"    classDef {vendor} fill:{color},stroke:#333,stroke-width:2px,color:#fff\n"
            mermaid_code += "    classDef replaced fill:#FFE4B5,stroke:#FF8C00,stroke-width:3px,color:#333\n\n"

            mermaid_code += "    %% Device nodes\n"
            used_ids = set()

            def get_unique_device_id(base_id):
                uid = base_id
                counter = 1
                while uid in used_ids:
                    uid = f"{base_id}_{counter}"
                    counter += 1
                used_ids.add(uid)
                return uid

            id_map = {}

            for device in devices:
                raw_id = self._sanitize_id(device.get("id", "unknown"))
                device_id = get_unique_device_id(raw_id)
                id_map[device["id"]] = device_id

                device_type = device.get("type", "generic")
                vendor = device.get("vendor", "generic").lower()
                model = device.get("model", "")
                specs = device.get("specifications", {})
                is_replaced = device.get("is_replaced", False)
                original_vendor = device.get("original_vendor", "")
                original_model = device.get("original_model", "")
                features = device.get("features", [])

                icon = self.device_icons.get(device_type, "📦")
                label_parts = [f"{icon} {device_id}"]

                if is_replaced:
                    label_parts.append(f"NEW: {vendor.upper()} {model}")
                    if original_vendor and original_model:
                        label_parts.append(f"WAS: {original_vendor} {original_model}")
                    for f in features[:2]:
                        if len(f) < 30:
                            label_parts.append(f"✓ {f}")
                    label_parts.append("✅ REPLACED")
                else:
                    if model:
                        label_parts.append(f"{vendor.upper()} {model}")

                if specs.get("ports") and specs["ports"] != "unknown":
                    label_parts.append(f"{specs['ports']} ports")
                if specs.get("speed") and specs["speed"] != "unknown":
                    label_parts.append(f"{specs['speed']}")

                label = "<br/>".join(label_parts)
                mermaid_code += f'    {device_id}["{label}"]\n'
                mermaid_code += f"    class {device_id} {'replaced' if is_replaced else vendor}\n"

            mermaid_code += "\n    %% Network connections\n"
            added_connections = set()
            for device in devices:
                src_id = id_map.get(device.get("id"))
                for conn in device.get("connections", []):
                    tgt_id = id_map.get(conn)
                    if src_id and tgt_id:
                        key = tuple(sorted([src_id, tgt_id]))
                        if key not in added_connections:
                            mermaid_code += f"    {src_id} ---|Link| {tgt_id}\n"
                            added_connections.add(key)

            return mermaid_code

        except Exception as e:
            logger.error(f"Error generating Mermaid diagram: {str(e)}")
            return self._generate_error_diagram(str(e))

    def generate_comparison_diagram(self, original: Dict[str, Any], modified: Dict[str, Any],
                                    id_generator: Optional[Callable[[Dict[str, Any], int], str]] = None) -> str:
        try:
            mermaid_code = """---
title: Network Topology Comparison - Before and After
---
graph LR
    subgraph "🔄 ORIGINAL"
        direction TD
"""
            for i, device in enumerate(original.get("devices", [])):
                dev_id = id_generator(device, i) if id_generator else f"orig_{self._sanitize_id(device.get('id'))}"
                icon = self.device_icons.get(device.get("type", "generic"), "📦")
                label = f"{icon} {device.get('vendor', '').title()}<br/>{device.get('model', '')}"
                mermaid_code += f'        {dev_id}["{label}"]\n'

            mermaid_code += """    end

    subgraph "✨ MODIFIED"
        direction TD
"""
            for i, device in enumerate(modified.get("devices", [])):
                dev_id = id_generator(device, i) if id_generator else f"mod_{self._sanitize_id(device.get('id'))}"
                icon = self.device_icons.get(device.get("type", "generic"), "📦")
                label = f"{icon} {device.get('vendor', '').title()}<br/>{device.get('model', '')}"
                if device.get("is_replaced"):
                    label += "<br/>⬆️ UPGRADED"
                mermaid_code += f'        {dev_id}["{label}"]\n'

            mermaid_code += "    end\n"
            return mermaid_code
        except Exception as e:
            logger.error(f"Error generating comparison diagram: {str(e)}")
            return self._generate_error_diagram(str(e))

    def _sanitize_id(self, device_id: str) -> str:
        return re.sub(r'[^a-zA-Z0-9_]', '_', str(device_id or "unknown"))

    def _generate_empty_diagram(self, title: str) -> str:
        return f"""---
title: {title}
---
graph TD
    NoDevices["⚠️ No devices detected<br/>Please check image quality<br/>and device visibility"]
    classDef warning fill:#fff3cd,stroke:#856404,stroke-width:2px,color:#856404
    class NoDevices warning
"""

    def _generate_error_diagram(self, error_message: str) -> str:
        return f"""graph TD
    Error["❌ Error generating diagram<br/>{error_message}"]
    classDef errorClass fill:#ffcccc,stroke:#ff0000,stroke-width:2px
    class Error errorClass
"""


# import json
# import logging
# from typing import Dict, Any, List, Optional, Callable

# logger = logging.getLogger(__name__)

# class MermaidGenerator:
#     def __init__(self):
#         """Initialize Mermaid diagram generator with enhanced styling"""
#         self.device_icons = {
#             "router": "🔀",
#             "switch": "🔄", 
#             "firewall": "🛡️",
#             "server": "🖥️",
#             "workstation": "💻",
#             "access_point": "📡",
#             "load_balancer": "⚖️",
#             "storage": "💾",
#             "gateway": "🚪"
#         }
        
#         self.vendor_colors = {
#             "cisco": "#1BA0D7",
#             "juniper": "#84BD00",
#             "aruba": "#FF6900", 
#             "arista": "#F80000",
#             "hpe": "#01A982",
#             "generic": "#6C757D"
#         }
        
#     def generate_network_diagram(self, topology_data: Dict[str, Any], title: str = "Network Topology") -> str:
#         """Generate comprehensive Mermaid network diagram"""
#         try:
#             devices = topology_data.get("devices", [])
            
#             if not devices:
#                 return self._generate_empty_diagram(title)
            
#             mermaid_code = f"""---
# title: {title}
# ---
# graph TD
#     %% Vendor-specific styling
#     classDef cisco fill:{self.vendor_colors['cisco']},stroke:#333,stroke-width:2px,color:#fff
#     classDef juniper fill:{self.vendor_colors['juniper']},stroke:#333,stroke-width:2px,color:#fff  
#     classDef aruba fill:{self.vendor_colors['aruba']},stroke:#333,stroke-width:2px,color:#fff
#     classDef arista fill:{self.vendor_colors['arista']},stroke:#333,stroke-width:2px,color:#fff
#     classDef hpe fill:{self.vendor_colors['hpe']},stroke:#333,stroke-width:2px,color:#fff
#     classDef generic fill:{self.vendor_colors['generic']},stroke:#333,stroke-width:2px,color:#fff
#     classDef replaced fill:#FFE4B5,stroke:#FF8C00,stroke-width:3px,color:#333
    
#     %% Device nodes
# """
            
#             # Add device nodes with enhanced information
#             for device in devices:
#                 device_id = self._sanitize_id(device.get("id", "unknown"))
#                 device_type = device.get("type", "generic")
#                 vendor = device.get("vendor", "generic").lower()
#                 model = device.get("model", "")
#                 specs = device.get("specifications", {})
#                 is_replaced = device.get("replacement_reason", "")
                
#                 icon = self.device_icons.get(device_type, "📦")
                
#                 # Build device label
#                 label_parts = [f"{icon} {device.get('id', 'Unknown')}"]
                
#                 if model:
#                     label_parts.append(f"Model: {model}")
                
#                 # Add specifications
#                 if specs:
#                     if specs.get("ports"):
#                         label_parts.append(f"Ports: {specs['ports']}")
#                     if specs.get("speed"):
#                         label_parts.append(f"Speed: {specs['speed']}")
                
#                 # Add replacement indicator
#                 if is_replaced:
#                     label_parts.append("🔄 REPLACED")
                
#                 label = "<br/>".join(label_parts)
                
#                 # Add device node
#                 mermaid_code += f'    {device_id}["{label}"]\n'
                
#                 # Apply styling
#                 if is_replaced:
#                     mermaid_code += f"    class {device_id} replaced\n"
#                 else:
#                     vendor_class = vendor if vendor in self.vendor_colors else "generic"
#                     mermaid_code += f"    class {device_id} {vendor_class}\n"
            
#             # Add connections
#             mermaid_code += "\n    %% Network connections\n"
#             processed_connections = set()
            
#             for device in devices:
#                 device_id = self._sanitize_id(device.get("id"))
#                 connections = device.get("connections", [])
                
#                 for connected_device in connections:
#                     connected_id = self._sanitize_id(connected_device)
                    
#                     # Avoid duplicate connections
#                     connection_key = tuple(sorted([device_id, connected_id]))
#                     if connection_key not in processed_connections:
#                         mermaid_code += f"    {device_id} ---|Link| {connected_id}\n"
#                         processed_connections.add(connection_key)
            
#             # If no connections defined, create a simple layout
#             if not any(device.get("connections") for device in devices):
#                 for i in range(len(devices) - 1):
#                     device1_id = self._sanitize_id(devices[i].get("id"))
#                     device2_id = self._sanitize_id(devices[i + 1].get("id"))
#                     mermaid_code += f"    {device1_id} ---|Network| {device2_id}\n"
            
#             return mermaid_code
            
#         except Exception as e:
#             logger.error(f"Error generating Mermaid diagram: {str(e)}")
#             return self._generate_error_diagram(str(e))
    
# #     def generate_comparison_diagram(self, 
# #                                     original_topology: Dict[str, Any], 
# #                                     modified_topology: Dict[str, Any],
# #                                     id_generator: Optional[Callable[[Dict[str, Any], int], str]] = None
# #                                     ) -> str:
# #         """Generate side-by-side comparison diagram"""
# #         try:
# #             mermaid_code = """---
# # title: Network Topology Comparison - Before and After
# # ---
# # graph LR
# #     subgraph "🔄 ORIGINAL"
# #         direction TD
# # """
            
# #             # Add original devices
# #             original_devices = original_topology.get("devices", [])
# #             for device in original_devices:
# #                 device_id = f"orig_{self._sanitize_id(device.get('id'))}"
# #                 device_type = device.get("type", "generic")
# #                 vendor = device.get("vendor", "generic")
# #                 model = device.get("model", "")
                
# #                 icon = self.device_icons.get(device_type, "📦")
# #                 label = f"{icon} {device.get('id')}"
# #                 if vendor and model:
# #                     label += f"<br/>{vendor} {model}"
                
# #                 mermaid_code += f'        {device_id}["{label}"]\n'
            
# #             mermaid_code += """    end
    
# #     subgraph "✨ MODIFIED"
# #         direction TD
# # """
            
# #             # Add modified devices
# #             modified_devices = modified_topology.get("devices", [])
# #             for device in modified_devices:
# #                 device_id = f"mod_{self._sanitize_id(device.get('id'))}"
# #                 device_type = device.get("type", "generic")
# #                 vendor = device.get("vendor", "generic")
# #                 model = device.get("model", "")
# #                 is_replaced = device.get("replacement_reason", "")
                
# #                 icon = self.device_icons.get(device_type, "📦")
# #                 label = f"{icon} {device.get('id')}"
# #                 if vendor and model:
# #                     label += f"<br/>{vendor} {model}"
# #                 if is_replaced:
# #                     label += f"<br/>🔄 UPGRADED"
                
# #                 mermaid_code += f'        {device_id}["{label}"]\n'
            
# #             mermaid_code += "    end\n"
            
# #             return mermaid_code
            
# #         except Exception as e:
# #             logger.error(f"Error generating comparison diagram: {str(e)}")
# #             return self._generate_error_diagram(str(e))
    
#     def generate_comparison_diagram(self, 
#                                     original_topology: Dict[str, Any], 
#                                     modified_topology: Dict[str, Any], 
#                                     id_generator: Optional[Callable[[Dict[str, Any], int], str]] = None
#                                 ) -> str:
#         """Generate side-by-side comparison diagram with safe unique IDs"""
#         try:
#             mermaid_code = """---
# title: Network Topology Comparison - Before and After
# ---
# graph LR
#     subgraph "🔄 ORIGINAL"
#         direction TD
# """

#             # Add original devices
#             original_devices = original_topology.get("devices", [])
#             for index, device in enumerate(original_devices):
#                 device_id = (
#                     id_generator(device, index)
#                     if id_generator else f"orig_{self._sanitize_id(device.get('id'))}"
#                 )
#                 device_type = device.get("type", "generic")
#                 vendor = device.get("vendor", "generic")
#                 model = device.get("model", "")
#                 icon = self.device_icons.get(device_type, "📦")

#                 label = f"{icon} {device.get('id')}"
#                 if vendor and model:
#                     label += f"<br/>{vendor} {model}"

#                 mermaid_code += f'        {device_id}["{label}"]\n'

#             mermaid_code += """    end

#         subgraph "✨ MODIFIED"
#             direction TD
#     """

#             # Add modified devices
#             modified_devices = modified_topology.get("devices", [])
#             for index, device in enumerate(modified_devices):
#                 device_id = (
#                     id_generator(device, index)
#                     if id_generator else f"mod_{self._sanitize_id(device.get('id'))}"
#                 )
#                 device_type = device.get("type", "generic")
#                 vendor = device.get("vendor", "generic")
#                 model = device.get("model", "")
#                 icon = self.device_icons.get(device_type, "📦")

#                 label = f"{icon} {device.get('id')}"
#                 if vendor and model:
#                     label += f"<br/>{vendor} {model}"
#                 if device.get("notes"):
#                     label += "<br/>🔄 UPGRADED"

#                 mermaid_code += f'        {device_id}["{label}"]\n'

#             mermaid_code += "    end\n"

#             return mermaid_code

#         except Exception as e:
#             logger.error(f"Error generating comparison diagram: {str(e)}")
#             return self._generate_error_diagram(str(e))

#     def _sanitize_id(self, device_id: str) -> str:
#         """Sanitize device ID for Mermaid compatibility"""
#         if not device_id:
#             return "unknown"
#         import re
#         return re.sub(r'[^a-zA-Z0-9_]', '_', str(device_id))
    
#     def _generate_empty_diagram(self, title: str) -> str:
#         """Generate diagram when no devices are found"""
#         return f"""---
# title: {title}
# ---
# graph TD
#     NoDevices["⚠️ No devices detected<br/>Please check image quality<br/>and device visibility"]
#     classDef warning fill:#fff3cd,stroke:#856404,stroke-width:2px,color:#856404
#     class NoDevices warning
# """
    
#     def _generate_error_diagram(self, error_message: str) -> str:
#         """Generate error diagram when something goes wrong"""
#         return f"""graph TD
#     Error["❌ Error generating diagram<br/>{error_message}"]
#     classDef errorClass fill:#ffcccc,stroke:#ff0000,stroke-width:2px
#     class Error errorClass
# """

# From HPE-Update-Checker-Network-Topology-DONE. .. latest Sampada's version
# import json
# import logging
# from typing import Dict, Any, List

# logger = logging.getLogger(__name__)

# class MermaidGenerator:
#     def __init__(self):
#         """Initialize Mermaid diagram generator with enhanced styling"""
#         self.device_icons = {
#             "router": "🔀",
#             "switch": "🔄", 
#             "firewall": "🛡️",
#             "server": "🖥️",
#             "workstation": "💻",
#             "access_point": "📡",
#             "load_balancer": "⚖️",
#             "storage": "💾",
#             "gateway": "🚪"
#         }
        
#         self.vendor_colors = {
#             "cisco": "#1BA0D7",
#             "juniper": "#84BD00",
#             "aruba": "#FF6900", 
#             "arista": "#F80000",
#             "hpe": "#01A982",
#             "fortinet": "#E31837",
#             "extreme": "#8B0000",
#             "dell": "#007DB8",
#             "generic": "#6C757D"
#         }
        
#     def generate_network_diagram(self, topology_data: Dict[str, Any], title: str = "Network Topology") -> str:
#         """Generate COMPLETELY DYNAMIC network diagram"""
#         try:
#             devices = topology_data.get("devices", [])
            
#             if not devices:
#                 return self._generate_empty_diagram(title)
            
#             # Dynamic vendor detection and styling
#             detected_vendors = set()
#             for device in devices:
#                 vendor = device.get("vendor", "generic").lower()
#                 detected_vendors.add(vendor)
            
#             mermaid_code = f"""---
# title: {title}
# ---
# graph TD
#     %% DYNAMIC styling based on detected vendors
# """
            
#             # Generate dynamic CSS classes for detected vendors
#             for vendor in detected_vendors:
#                 color = self.vendor_colors.get(vendor, "#6C757D")  # Default gray
#                 mermaid_code += f"    classDef {vendor} fill:{color},stroke:#333,stroke-width:2px,color:#fff\n"
            
#             mermaid_code += "    classDef replaced fill:#28A745,stroke:#155724,stroke-width:4px,color:#fff\n\n"
#             mermaid_code += "    %% DYNAMIC device nodes\n"
            
#             # Generate device nodes completely dynamically
#             for device in devices:
#                 device_id = self._sanitize_id(device.get("id", "unknown"))
#                 device_type = device.get("type", "generic")
#                 vendor = device.get("vendor", "generic").lower()
#                 model = device.get("model", "")
#                 is_replaced = device.get("is_replaced", False)
#                 original_vendor = device.get("original_vendor", "")
#                 original_model = device.get("original_model", "")
#                 features = device.get("features", [])
                
#                 # Dynamic icon selection
#                 icon = self.device_icons.get(device_type, "📦")
                
#                 # Build COMPLETELY DYNAMIC device label
#                 if is_replaced:
#                     label_parts = [f"🔄 {icon} {device_id}"]
#                     if model:
#                         label_parts.append(f"NEW: {vendor.upper()} {model}")
#                     if original_vendor and original_model:
#                         label_parts.append(f"WAS: {original_vendor} {original_model}")
                    
#                     # Add key features dynamically
#                     if features:
#                         key_features = features[:2]  # Show top 2 features
#                         for feature in key_features:
#                             if len(feature) < 30:  # Only show concise features
#                                 label_parts.append(f"✓ {feature}")
                    
#                     label_parts.append("✅ REPLACED")
#                 else:
#                     label_parts = [f"{icon} {device_id}"]
#                     if model:
#                         label_parts.append(f"{vendor.upper()} {model}")
                
#                 # Add specifications dynamically
#                 specs = device.get("specifications", {})
#                 if specs.get("ports") and specs["ports"] != "unknown":
#                     label_parts.append(f"{specs['ports']} ports")
                
#                 label = "<br/>".join(label_parts)
#                 mermaid_code += f'    {device_id}["{label}"]\n'
                
#                 # Apply DYNAMIC styling
#                 if is_replaced:
#                     mermaid_code += f"    class {device_id} replaced\n"
#                 else:
#                     vendor_class = vendor if vendor in detected_vendors else "generic"
#                     mermaid_code += f"    class {device_id} {vendor_class}\n"
            
#             # Generate DYNAMIC connections based on topology data
#             mermaid_code += "\n    %% DYNAMIC network connections\n"
#             connections_added = set()
            
#             for device in devices:
#                 device_id = self._sanitize_id(device.get("id"))
#                 connections = device.get("connections", [])
                
#                 for connected_device in connections:
#                     connected_id = self._sanitize_id(connected_device)
                    
#                     # Check if target device exists or is special (like internet)
#                     target_exists = (any(self._sanitize_id(d.get("id")) == connected_id for d in devices) or 
#                                    connected_device.lower() in ['internet', 'wan', 'cloud'])
                    
#                     if target_exists:
#                         connection_key = tuple(sorted([device_id, connected_id]))
#                         if connection_key not in connections_added:
#                             mermaid_code += f"    {device_id} ---|Link| {connected_id}\n"
#                             connections_added.add(connection_key)
            
#             # Dynamic fallback connections if none exist
#             if not connections_added:
#                 mermaid_code += self._generate_dynamic_fallback_connections(devices)
            
#             return mermaid_code
            
#         except Exception as e:
#             logger.error(f"Error generating dynamic diagram: {str(e)}")
#             return self._generate_error_diagram(str(e))

#     def generate_comparison_diagram(self, original_topology: Dict[str, Any], 
#                                   modified_topology: Dict[str, Any]) -> str:
#         """Generate DYNAMIC side-by-side comparison showing actual device changes"""
#         try:
#             mermaid_code = """---
# title: Network Transformation - Device Replacements
# ---
# graph LR
#     subgraph "🔄 BEFORE (Original)"
#         direction TD
# """
            
#             # Add original devices with clean labels
#             original_devices = original_topology.get("devices", [])
#             for device in original_devices:
#                 device_id = f"orig_{self._sanitize_id(device.get('id'))}"
#                 device_type = device.get("type", "generic")
#                 vendor = device.get("vendor", "generic")
#                 model = device.get("model", "")
                
#                 icon = self.device_icons.get(device_type, "📦")
                
#                 # Clean original device label
#                 if model:
#                     label = f"{icon} {vendor.upper()}<br/>{model}"
#                 else:
#                     label = f"{icon} {vendor.upper()}<br/>{device_type.title()}"
                
#                 mermaid_code += f'        {device_id}["{label}"]\n'
            
#             mermaid_code += """    end
    
#     subgraph "✨ AFTER (Modified)"
#         direction TD
# """
            
#             # Add modified devices showing NEW vendor/model clearly
#             modified_devices = modified_topology.get("devices", [])
#             for device in modified_devices:
#                 device_id = f"mod_{self._sanitize_id(device.get('id'))}"
#                 device_type = device.get("type", "generic")
#                 vendor = device.get("vendor", "generic")
#                 model = device.get("model", "")
#                 is_replaced = device.get("is_replaced", False)
                
#                 icon = self.device_icons.get(device_type, "📦")
                
#                 if is_replaced:
#                     # Show NEW device clearly with upgrade indicator
#                     if model:
#                         label = f"⬆️ {icon} {vendor.upper()}<br/>{model}<br/>UPGRADED"
#                     else:
#                         label = f"⬆️ {icon} {vendor.upper()}<br/>{device_type.title()}<br/>UPGRADED"
#                 else:
#                     # Show unchanged device
#                     if model:
#                         label = f"{icon} {vendor.upper()}<br/>{model}"
#                     else:
#                         label = f"{icon} {vendor.upper()}<br/>{device_type.title()}"
                
#                 mermaid_code += f'        {device_id}["{label}"]\n'
            
#             mermaid_code += """    end
    
#     %% Clean styling for comparison
#     classDef originalStyle fill:#E8F4FD,stroke:#1BA0D7,stroke-width:2px,color:#333
#     classDef upgradedStyle fill:#D4EDDA,stroke:#28A745,stroke-width:3px,color:#333
#     classDef unchangedStyle fill:#F8F9FA,stroke:#6C757D,stroke-width:2px,color:#333
# """
            
#             # Apply styles to original devices
#             for device in original_devices:
#                 device_id = f"orig_{self._sanitize_id(device.get('id'))}"
#                 mermaid_code += f"    class {device_id} originalStyle\n"
            
#             # Apply styles to modified devices
#             for device in modified_devices:
#                 device_id = f"mod_{self._sanitize_id(device.get('id'))}"
#                 if device.get("is_replaced"):
#                     mermaid_code += f"    class {device_id} upgradedStyle\n"
#                 else:
#                     mermaid_code += f"    class {device_id} unchangedStyle\n"
            
#             return mermaid_code
            
#         except Exception as e:
#             logger.error(f"Error generating comparison diagram: {str(e)}")
#             return self._generate_error_diagram(str(e))

#     def _generate_dynamic_fallback_connections(self, devices: List[Dict[str, Any]]) -> str:
#         """Generate COMPLETELY DYNAMIC fallback connections"""
#         connections = ""
        
#         if len(devices) <= 1:
#             return connections
        
#         # Dynamic device categorization based on actual device properties
#         device_categories = {
#             'core': [],
#             'distribution': [],
#             'access': [],
#             'edge': [],
#             'endpoint': []
#         }
        
#         for device in devices:
#             device_id = device.get("id", "").lower()
#             device_type = device.get("type", "").lower()
#             device_model = device.get("model", "").lower()
#             vendor = device.get("vendor", "").lower()
            
#             # Dynamic categorization logic - NO HARDCODED VALUES
#             core_indicators = ['core', 'catalyst', '9500', '8400', '8360', 'nexus']
#             access_indicators = ['access', 'edge', '2960', '2930', '6300']
#             distribution_indicators = ['distribution', 'dist', 'aggregation']
#             edge_indicators = ['router', 'firewall', 'gateway', 'border']
#             endpoint_indicators = ['workstation', 'server', 'access_point', 'wireless', 'ap']
            
#             # Check for core devices
#             if any(indicator in device_id or indicator in device_model for indicator in core_indicators):
#                 device_categories['core'].append(device)
#             # Check for access devices
#             elif any(indicator in device_id or indicator in device_model for indicator in access_indicators):
#                 device_categories['access'].append(device)
#             # Check for distribution devices
#             elif any(indicator in device_id or indicator in device_model for indicator in distribution_indicators):
#                 device_categories['distribution'].append(device)
#             # Check for edge devices
#             elif device_type in ['router', 'firewall'] or any(indicator in device_id for indicator in edge_indicators):
#                 device_categories['edge'].append(device)
#             # Check for endpoint devices
#             elif device_type in ['workstation', 'server', 'access_point'] or any(indicator in device_id for indicator in endpoint_indicators):
#                 device_categories['endpoint'].append(device)
#             else:
#                 # Default categorization based on device type
#                 if device_type == 'switch':
#                     device_categories['distribution'].append(device)
#                 elif device_type in ['router', 'firewall']:
#                     device_categories['edge'].append(device)
#                 else:
#                     device_categories['access'].append(device)
        
#         # Generate connections dynamically based on hierarchy
#         for core in device_categories['core']:
#             for dist in device_categories['distribution']:
#                 core_id = self._sanitize_id(core.get("id"))
#                 dist_id = self._sanitize_id(dist.get("id"))
#                 connections += f"    {core_id} ---|Link| {dist_id}\n"
        
#         for dist in device_categories['distribution']:
#             for access in device_categories['access']:
#                 dist_id = self._sanitize_id(dist.get("id"))
#                 access_id = self._sanitize_id(access.get("id"))
#                 connections += f"    {dist_id} ---|Link| {access_id}\n"
        
#         for access in device_categories['access']:
#             for endpoint in device_categories['endpoint']:
#                 access_id = self._sanitize_id(access.get("id"))
#                 endpoint_id = self._sanitize_id(endpoint.get("id"))
#                 connections += f"    {access_id} ---|Link| {endpoint_id}\n"
        
#         # Connect edge devices
#         for edge in device_categories['edge']:
#             edge_id = self._sanitize_id(edge.get("id"))
#             if device_categories['core']:
#                 core_id = self._sanitize_id(device_categories['core'].get("id"))
#                 connections += f"    {edge_id} ---|Link| {core_id}\n"
#             elif device_categories['distribution']:
#                 dist_id = self._sanitize_id(device_categories['distribution'].get("id"))
#                 connections += f"    {edge_id} ---|Link| {dist_id}\n"
#             connections += f"    {edge_id} ---|Link| internet\n"
        
#         # If no hierarchical connections, create simple chain
#         if not connections:
#             for i in range(len(devices) - 1):
#                 device1_id = self._sanitize_id(devices[i].get("id"))
#                 device2_id = self._sanitize_id(devices[i + 1].get("id"))
#                 connections += f"    {device1_id} ---|Network| {device2_id}\n"
        
#         return connections
    
#     def _sanitize_id(self, device_id: str) -> str:
#         """Sanitize device ID for Mermaid compatibility"""
#         if not device_id:
#             return "unknown"
#         import re
#         return re.sub(r'[^a-zA-Z0-9_]', '_', str(device_id))
    
#     def _generate_empty_diagram(self, title: str) -> str:
#         """Generate diagram when no devices are found"""
#         return f"""---
# title: {title}
# ---
# graph TD
#     NoDevices["⚠️ No devices detected<br/>Please check image quality<br/>and device visibility"]
#     classDef warning fill:#FFF3CD,stroke:#856404,stroke-width:2px,color:#856404
#     class NoDevices warning
# """
    
#     def _generate_error_diagram(self, error_message: str) -> str:
#         """Generate error diagram when something goes wrong"""
#         return f"""graph TD
#     Error["❌ Error generating diagram<br/>{error_message[:50]}..."]
#     classDef errorClass fill:#F8D7DA,stroke:#721C24,stroke-width:2px,color:#721C24
#     class Error errorClass
# """
