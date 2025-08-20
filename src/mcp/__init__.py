"""
Model Context Protocol (MCP) Integration for WordOfPrompt.

This module provides a complete MCP-based architecture that replaces traditional
middleware with standardized protocol communication between AI models and external tools.

The MCP integration includes:
- MCP Servers for exposing services (Amazon search, intent analysis, etc.)
- MCP Clients for consuming services
- MCP Tools for standardized functionality
- Protocol handlers for message routing and processing
"""

from .servers import *
from .clients import *
from .tools import *
from .protocols import *

__all__ = [
    # Core MCP components
    "MCPServer",
    "MCPClient", 
    "MCPTool",
    "MCPProtocolHandler",
    
    # Specific servers
    "AmazonSearchServer",
    "IntentAnalysisServer",
    "KeywordExtractionServer",
    "ProductRecommendationServer",
    
    # Clients
    "WordOfPromptClient",
    "UnifiedMCPClient",
    
    # Tools
    "AmazonSearchTool",
    "IntentAnalysisTool",
    "KeywordExtractionTool",
    "ProductAnalysisTool",
    
    # Protocol utilities
    "MCPMessage",
    "MCPRequest",
    "MCPResponse",
    "MCPError",
]
