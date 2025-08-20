"""
Amazon Search MCP Server.

This server exposes Amazon product search functionality through the Model Context Protocol,
replacing traditional REST API middleware with standardized MCP communication.
"""

import logging
from typing import Any, Dict, List, Optional

import sys
from pathlib import Path

# Add the src directory to Python path
src_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_dir))

from mcp.servers.base import WordOfPromptMCPServer
from tools.amazon_search import AmazonSearchTool
from core.exceptions import ProductSearchError


logger = logging.getLogger(__name__)


class AmazonSearchServer(WordOfPromptMCPServer):
    """
    MCP Server for Amazon product search functionality.
    
    This server exposes Amazon search capabilities through MCP tools,
    replacing the traditional middleware approach with standardized protocol communication.
    """
    
    def __init__(self):
        super().__init__(
            name="amazon-search-server",
            version="1.0.0",
            description="MCP server providing Amazon product search capabilities via Rainforest API"
        )
        self.search_tool: Optional[AmazonSearchTool] = None
    
    async def setup_tools(self):
        """Setup Amazon search tools."""
        
        # Initialize the search tool
        try:
            self.search_tool = AmazonSearchTool(
                api_key=self.config.get("rainforest_api_key"),
                affiliate_id=self.config.get("amazon_affiliate_id"),
                default_domain=self.config.get("default_amazon_domain", "amazon.com")
            )
        except Exception as e:
            logger.error(f"Failed to initialize Amazon search tool: {e}")
            raise
        
        # Register search products tool
        self.register_tool(
            name="search_products",
            description="Search for products on Amazon using keywords and filters",
            input_schema={
                "type": "object",
                "properties": {
                    "keywords": {
                        "type": "string",
                        "description": "Search keywords for products"
                    },
                    "domain": {
                        "type": "string",
                        "description": "Amazon domain to search (e.g., amazon.com, amazon.co.uk)",
                        "default": "amazon.com"
                    },
                    "sort_by": {
                        "type": "string",
                        "description": "Sort order for results",
                        "enum": ["relevance", "price_low_to_high", "price_high_to_low", "newest_arrivals"],
                        "default": "relevance"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "minimum": 1,
                        "maximum": 20,
                        "default": 5
                    },
                    "generate_affiliate_links": {
                        "type": "boolean",
                        "description": "Whether to generate affiliate links",
                        "default": True
                    }
                },
                "required": ["keywords"]
            },
            handler=self._search_products
        )
        
        # Register get product details tool
        self.register_tool(
            name="get_product_details",
            description="Get detailed information about a specific product by ASIN",
            input_schema={
                "type": "object",
                "properties": {
                    "asin": {
                        "type": "string",
                        "description": "Amazon Standard Identification Number (ASIN) of the product"
                    },
                    "domain": {
                        "type": "string",
                        "description": "Amazon domain",
                        "default": "amazon.com"
                    }
                },
                "required": ["asin"]
            },
            handler=self._get_product_details
        )
        
        # Register search suggestions tool
        self.register_tool(
            name="get_search_suggestions",
            description="Get search suggestions based on partial keywords",
            input_schema={
                "type": "object",
                "properties": {
                    "partial_query": {
                        "type": "string",
                        "description": "Partial search query to get suggestions for"
                    },
                    "domain": {
                        "type": "string",
                        "description": "Amazon domain",
                        "default": "amazon.com"
                    },
                    "max_suggestions": {
                        "type": "integer",
                        "description": "Maximum number of suggestions",
                        "minimum": 1,
                        "maximum": 10,
                        "default": 5
                    }
                },
                "required": ["partial_query"]
            },
            handler=self._get_search_suggestions
        )
    
    async def setup_resources(self):
        """Setup Amazon search resources."""
        
        # Register domains resource
        self.register_resource(
            uri="amazon://domains",
            name="Amazon Domains",
            description="List of supported Amazon domains",
            mime_type="application/json",
            handler=self._get_domains_resource
        )
        
        # Register categories resource
        self.register_resource(
            uri="amazon://categories",
            name="Amazon Categories",
            description="List of Amazon product categories",
            mime_type="application/json",
            handler=self._get_categories_resource
        )
        
        # Register search statistics resource
        self.register_resource(
            uri="amazon://stats",
            name="Search Statistics",
            description="Amazon search usage statistics",
            mime_type="application/json",
            handler=self._get_stats_resource
        )
    
    async def setup_prompts(self):
        """Setup Amazon search prompts."""
        
        # Register product search prompt
        self.register_prompt(
            name="product_search",
            description="Generate optimized product search queries",
            arguments=[
                {
                    "name": "user_intent",
                    "description": "User's search intent or requirement",
                    "required": True
                },
                {
                    "name": "budget_range",
                    "description": "Budget range for the product",
                    "required": False
                },
                {
                    "name": "specific_features",
                    "description": "Specific features the user is looking for",
                    "required": False
                }
            ],
            handler=self._generate_search_prompt
        )
        
        # Register product comparison prompt
        self.register_prompt(
            name="product_comparison",
            description="Generate product comparison analysis",
            arguments=[
                {
                    "name": "products",
                    "description": "List of products to compare",
                    "required": True
                },
                {
                    "name": "comparison_criteria",
                    "description": "Criteria for comparison (price, features, reviews, etc.)",
                    "required": False
                }
            ],
            handler=self._generate_comparison_prompt
        )
    
    # Tool handlers
    async def _search_products(self, keywords: str, domain: str = "amazon.com", 
                              sort_by: str = "relevance", max_results: int = 5,
                              generate_affiliate_links: bool = True) -> Dict[str, Any]:
        """Search for products on Amazon."""
        try:
            result = self.search_tool.search(
                keywords=keywords,
                domain=domain,
                sort_by=sort_by,
                max_results=max_results,
                generate_affiliate_links=generate_affiliate_links
            )
            
            return {
                "success": True,
                "data": result.to_dict(),
                "message": f"Found {len(result.products)} products for '{keywords}'"
            }
            
        except ProductSearchError as e:
            logger.error(f"Product search failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Product search failed"
            }
    
    async def _get_product_details(self, asin: str, domain: str = "amazon.com") -> Dict[str, Any]:
        """Get detailed product information by ASIN."""
        try:
            # This would use a specific product details API call
            # For now, we'll simulate it with a search
            result = self.search_tool.search(
                keywords=asin,
                domain=domain,
                max_results=1
            )
            
            if result.products:
                product = result.products[0]
                return {
                    "success": True,
                    "data": product.to_dict(),
                    "message": f"Retrieved details for product {asin}"
                }
            else:
                return {
                    "success": False,
                    "error": f"Product with ASIN {asin} not found",
                    "message": "Product not found"
                }
                
        except Exception as e:
            logger.error(f"Failed to get product details: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to retrieve product details"
            }
    
    async def _get_search_suggestions(self, partial_query: str, domain: str = "amazon.com",
                                    max_suggestions: int = 5) -> Dict[str, Any]:
        """Get search suggestions for partial query."""
        try:
            # This is a simplified implementation
            # In a real scenario, you'd use Amazon's autocomplete API
            suggestions = [
                f"{partial_query} best",
                f"{partial_query} cheap",
                f"{partial_query} reviews",
                f"{partial_query} 2024",
                f"{partial_query} sale"
            ][:max_suggestions]
            
            return {
                "success": True,
                "data": {
                    "query": partial_query,
                    "suggestions": suggestions
                },
                "message": f"Generated {len(suggestions)} suggestions"
            }
            
        except Exception as e:
            logger.error(f"Failed to get suggestions: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to get search suggestions"
            }
    
    # Resource handlers
    async def _get_domains_resource(self, uri: str) -> str:
        """Get supported Amazon domains."""
        domains = [
            {"domain": "amazon.com", "country": "United States", "currency": "USD"},
            {"domain": "amazon.co.uk", "country": "United Kingdom", "currency": "GBP"},
            {"domain": "amazon.de", "country": "Germany", "currency": "EUR"},
            {"domain": "amazon.fr", "country": "France", "currency": "EUR"},
            {"domain": "amazon.it", "country": "Italy", "currency": "EUR"},
            {"domain": "amazon.es", "country": "Spain", "currency": "EUR"},
            {"domain": "amazon.ca", "country": "Canada", "currency": "CAD"},
            {"domain": "amazon.in", "country": "India", "currency": "INR"},
        ]
        
        import json
        return json.dumps({"domains": domains}, indent=2)
    
    async def _get_categories_resource(self, uri: str) -> str:
        """Get Amazon product categories."""
        categories = [
            "Electronics", "Computers", "Home & Kitchen", "Sports & Outdoors",
            "Books", "Movies & TV", "Music", "Clothing", "Health & Beauty",
            "Toys & Games", "Automotive", "Industrial & Scientific"
        ]
        
        import json
        return json.dumps({"categories": categories}, indent=2)
    
    async def _get_stats_resource(self, uri: str) -> str:
        """Get search statistics."""
        stats = self.get_stats()
        
        import json
        return json.dumps(stats, indent=2)
    
    # Prompt handlers
    async def _generate_search_prompt(self, user_intent: str, budget_range: str = None,
                                    specific_features: str = None) -> List[Dict[str, Any]]:
        """Generate optimized search prompt."""
        messages = [
            {
                "role": "system",
                "content": "You are an expert at creating optimized Amazon search queries. Generate the best keywords to find products that match the user's intent."
            },
            {
                "role": "user",
                "content": f"User intent: {user_intent}"
            }
        ]
        
        if budget_range:
            messages[-1]["content"] += f"\nBudget range: {budget_range}"
        
        if specific_features:
            messages[-1]["content"] += f"\nSpecific features: {specific_features}"
        
        messages[-1]["content"] += "\n\nGenerate 3-5 optimized search keywords that will help find the best matching products on Amazon."
        
        return messages
    
    async def _generate_comparison_prompt(self, products: str, comparison_criteria: str = None) -> List[Dict[str, Any]]:
        """Generate product comparison prompt."""
        messages = [
            {
                "role": "system",
                "content": "You are an expert product analyst. Create a comprehensive comparison of the given products, highlighting their strengths, weaknesses, and best use cases."
            },
            {
                "role": "user",
                "content": f"Compare these products:\n{products}"
            }
        ]
        
        if comparison_criteria:
            messages[-1]["content"] += f"\n\nFocus on these criteria: {comparison_criteria}"
        
        messages[-1]["content"] += "\n\nProvide a detailed comparison with pros, cons, and recommendations for different user needs."
        
        return messages
