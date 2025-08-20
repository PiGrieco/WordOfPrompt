"""
Unified WordOfPrompt MCP Server.

This server combines all WordOfPrompt functionality into a single MCP server,
providing a complete replacement for the traditional middleware architecture.
"""

import logging
from typing import Any, Dict, List, Optional

import sys
from pathlib import Path

# Add the src directory to Python path
src_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_dir))

from mcp.servers.base import WordOfPromptMCPServer
from core.prompt_analyzer import PromptAnalyzer
from core.keyword_extractor import KeywordExtractor
from tools.amazon_search import AmazonSearchTool
from core.exceptions import WordOfPromptError


logger = logging.getLogger(__name__)


class UnifiedWordOfPromptServer(WordOfPromptMCPServer):
    """
    Unified MCP Server providing complete WordOfPrompt functionality.
    
    This server replaces the entire middleware stack with a single MCP server
    that provides all WordOfPrompt capabilities through standardized tools.
    """
    
    def __init__(self):
        super().__init__(
            name="wordofprompt-unified-server",
            version="1.0.0",
            description="Unified MCP server providing complete AI-powered advertising recommendation system"
        )
        
        # Core components
        self.prompt_analyzer: Optional[PromptAnalyzer] = None
        self.keyword_extractor: Optional[KeywordExtractor] = None
        self.amazon_tool: Optional[AmazonSearchTool] = None
        
        # Sub-servers (for modular functionality)
        self.amazon_server: Optional[AmazonSearchServer] = None
        self.intent_server: Optional[IntentAnalysisServer] = None
        self.keyword_server: Optional[KeywordExtractionServer] = None
        self.recommendation_server: Optional[ProductRecommendationServer] = None
    
    async def setup_tools(self):
        """Setup all WordOfPrompt tools in the unified server."""
        
        # Initialize core components
        await self._initialize_components()
        
        # Setup main workflow tool
        self.register_tool(
            name="analyze_and_recommend",
            description="Complete WordOfPrompt workflow: analyze intent, extract keywords, search products, and generate recommendations",
            input_schema={
                "type": "object",
                "properties": {
                    "user_prompt": {
                        "type": "string",
                        "description": "User's input prompt to analyze and process"
                    },
                    "intent_threshold": {
                        "type": "number",
                        "description": "Threshold for intent detection (0.0-1.0)",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "default": 0.95
                    },
                    "max_keywords": {
                        "type": "integer",
                        "description": "Maximum number of keywords to extract",
                        "minimum": 1,
                        "maximum": 20,
                        "default": 5
                    },
                    "max_products": {
                        "type": "integer",
                        "description": "Maximum number of products to recommend",
                        "minimum": 1,
                        "maximum": 20,
                        "default": 5
                    },
                    "amazon_domain": {
                        "type": "string",
                        "description": "Amazon domain to search",
                        "default": "amazon.com"
                    },
                    "sort_by": {
                        "type": "string",
                        "description": "Sort order for product results",
                        "enum": ["relevance", "price_low_to_high", "price_high_to_low", "newest_arrivals"],
                        "default": "relevance"
                    }
                },
                "required": ["user_prompt"]
            },
            handler=self._analyze_and_recommend
        )
        
        # Setup individual component tools
        await self._setup_intent_tools()
        await self._setup_keyword_tools()
        await self._setup_search_tools()
        await self._setup_recommendation_tools()
        
        # Setup utility tools
        await self._setup_utility_tools()
    
    async def setup_resources(self):
        """Setup unified resources."""
        
        # System configuration resource
        self.register_resource(
            uri="wordofprompt://config",
            name="System Configuration",
            description="Current WordOfPrompt system configuration",
            mime_type="application/json",
            handler=self._get_config_resource
        )
        
        # System health resource
        self.register_resource(
            uri="wordofprompt://health",
            name="System Health",
            description="WordOfPrompt system health status",
            mime_type="application/json",
            handler=self._get_health_resource
        )
        
        # Analytics resource
        self.register_resource(
            uri="wordofprompt://analytics",
            name="System Analytics",
            description="Usage analytics and statistics",
            mime_type="application/json",
            handler=self._get_analytics_resource
        )
        
        # Model information resource
        self.register_resource(
            uri="wordofprompt://models",
            name="Model Information",
            description="Information about loaded AI models",
            mime_type="application/json",
            handler=self._get_models_resource
        )
    
    async def setup_prompts(self):
        """Setup unified prompts."""
        
        # Complete workflow prompt
        self.register_prompt(
            name="complete_workflow",
            description="Execute complete WordOfPrompt workflow with customizable parameters",
            arguments=[
                {
                    "name": "user_query",
                    "description": "User's search query or intent",
                    "required": True
                },
                {
                    "name": "workflow_type",
                    "description": "Type of workflow to execute (product_search, intent_analysis, keyword_extraction)",
                    "required": False
                }
            ],
            handler=self._generate_workflow_prompt
        )
        
        # Product recommendation prompt
        self.register_prompt(
            name="product_recommendation",
            description="Generate personalized product recommendations",
            arguments=[
                {
                    "name": "user_profile",
                    "description": "User preferences and profile information",
                    "required": True
                },
                {
                    "name": "context",
                    "description": "Additional context for recommendations",
                    "required": False
                }
            ],
            handler=self._generate_recommendation_prompt
        )
    
    # Core workflow implementation
    async def _analyze_and_recommend(self, user_prompt: str, intent_threshold: float = 0.95,
                                   max_keywords: int = 5, max_products: int = 5,
                                   amazon_domain: str = "amazon.com", sort_by: str = "relevance") -> Dict[str, Any]:
        """
        Complete WordOfPrompt workflow implementation.
        
        This method replaces the entire middleware stack with a single MCP tool call.
        """
        try:
            workflow_result = {
                "user_prompt": user_prompt,
                "workflow_steps": [],
                "final_result": None,
                "metadata": {
                    "intent_threshold": intent_threshold,
                    "max_keywords": max_keywords,
                    "max_products": max_products,
                    "amazon_domain": amazon_domain,
                    "sort_by": sort_by
                }
            }
            
            # Step 1: Analyze intent
            logger.info("Step 1: Analyzing user intent")
            intent_result = self.prompt_analyzer.analyze(user_prompt)
            
            workflow_result["workflow_steps"].append({
                "step": "intent_analysis",
                "result": intent_result.to_dict(),
                "status": "completed"
            })
            
            # Check if intent meets threshold
            if not intent_result.has_intent:
                # No purchase intent detected - return general response
                workflow_result["final_result"] = {
                    "type": "general_response",
                    "message": "No purchase intent detected. This appears to be a general conversation.",
                    "intent_score": intent_result.intent_score,
                    "suggestion": "Try asking about specific products you'd like to buy."
                }
                return workflow_result
            
            # Step 2: Extract keywords
            logger.info("Step 2: Extracting keywords")
            keywords = intent_result.keywords if intent_result.keywords else self.keyword_extractor.extract(user_prompt, max_keywords)
            
            workflow_result["workflow_steps"].append({
                "step": "keyword_extraction",
                "result": {
                    "keywords": keywords,
                    "count": len(keywords)
                },
                "status": "completed"
            })
            
            # Step 3: Search products
            logger.info("Step 3: Searching products on Amazon")
            search_keywords = " ".join(keywords[:3])  # Use top 3 keywords
            
            search_result = self.amazon_tool.search(
                keywords=search_keywords,
                domain=amazon_domain,
                sort_by=sort_by,
                max_results=max_products
            )
            
            workflow_result["workflow_steps"].append({
                "step": "product_search",
                "result": {
                    "query": search_keywords,
                    "products_found": len(search_result.products),
                    "search_time": search_result.search_time
                },
                "status": "completed"
            })
            
            # Step 4: Generate recommendations
            logger.info("Step 4: Generating product recommendations")
            recommendations = []
            
            for i, product in enumerate(search_result.products, 1):
                recommendation = {
                    "rank": i,
                    "product": product.to_dict(),
                    "recommendation_score": self._calculate_recommendation_score(product, intent_result),
                    "pros": self._generate_pros(product),
                    "cons": self._generate_cons(product),
                    "best_for": self._generate_best_for(product, intent_result)
                }
                recommendations.append(recommendation)
            
            workflow_result["workflow_steps"].append({
                "step": "recommendation_generation",
                "result": {
                    "recommendations_count": len(recommendations)
                },
                "status": "completed"
            })
            
            # Final result
            workflow_result["final_result"] = {
                "type": "product_recommendations",
                "intent_analysis": intent_result.to_dict(),
                "keywords": keywords,
                "search_query": search_keywords,
                "recommendations": recommendations,
                "summary": f"Found {len(recommendations)} product recommendations based on your intent to {intent_result.intent_label}"
            }
            
            return workflow_result
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Workflow execution failed",
                "user_prompt": user_prompt
            }
    
    # Component initialization
    async def _initialize_components(self):
        """Initialize all core components."""
        try:
            # Initialize prompt analyzer
            self.prompt_analyzer = PromptAnalyzer(
                config=self.config,
                intent_threshold=self.config.get("intent_threshold", 0.95),
                extract_keywords=True
            )
            
            # Initialize keyword extractor
            self.keyword_extractor = KeywordExtractor(
                method=self.config.get("keyword_method", "rake"),
                max_keywords=self.config.get("max_keywords", 5)
            )
            
            # Initialize Amazon search tool
            self.amazon_tool = AmazonSearchTool(
                api_key=self.config.get("rainforest_api_key"),
                affiliate_id=self.config.get("amazon_affiliate_id"),
                default_domain=self.config.get("default_amazon_domain", "amazon.com")
            )
            
            logger.info("All core components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    # Individual tool setups
    async def _setup_intent_tools(self):
        """Setup intent analysis tools."""
        self.register_tool(
            name="analyze_intent",
            description="Analyze user intent for purchase behavior",
            input_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to analyze"},
                    "threshold": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.95}
                },
                "required": ["text"]
            },
            handler=self._analyze_intent
        )
    
    async def _setup_keyword_tools(self):
        """Setup keyword extraction tools."""
        self.register_tool(
            name="extract_keywords",
            description="Extract keywords from text using various algorithms",
            input_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to extract keywords from"},
                    "method": {"type": "string", "enum": ["rake", "yake", "hybrid"], "default": "rake"},
                    "max_keywords": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5}
                },
                "required": ["text"]
            },
            handler=self._extract_keywords
        )
    
    async def _setup_search_tools(self):
        """Setup Amazon search tools."""
        self.register_tool(
            name="search_amazon",
            description="Search for products on Amazon",
            input_schema={
                "type": "object",
                "properties": {
                    "keywords": {"type": "string", "description": "Search keywords"},
                    "domain": {"type": "string", "default": "amazon.com"},
                    "sort_by": {"type": "string", "default": "relevance"},
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5}
                },
                "required": ["keywords"]
            },
            handler=self._search_amazon
        )
    
    async def _setup_recommendation_tools(self):
        """Setup product recommendation tools."""
        self.register_tool(
            name="generate_recommendations",
            description="Generate product recommendations with pros/cons analysis",
            input_schema={
                "type": "object",
                "properties": {
                    "products": {"type": "array", "description": "List of products to analyze"},
                    "user_intent": {"type": "string", "description": "User's intent or requirements"}
                },
                "required": ["products"]
            },
            handler=self._generate_recommendations
        )
    
    async def _setup_utility_tools(self):
        """Setup utility tools."""
        self.register_tool(
            name="get_system_status",
            description="Get current system status and health information",
            input_schema={"type": "object", "properties": {}},
            handler=self._get_system_status
        )
    
    # Tool handlers
    async def _analyze_intent(self, text: str, threshold: float = 0.95) -> Dict[str, Any]:
        """Analyze intent tool handler."""
        result = self.prompt_analyzer.analyze(text)
        return result.to_dict()
    
    async def _extract_keywords(self, text: str, method: str = "rake", max_keywords: int = 5) -> Dict[str, Any]:
        """Extract keywords tool handler."""
        extractor = KeywordExtractor(method=method, max_keywords=max_keywords)
        keywords = extractor.extract(text)
        return {"keywords": keywords, "method": method, "count": len(keywords)}
    
    async def _search_amazon(self, keywords: str, domain: str = "amazon.com", 
                           sort_by: str = "relevance", max_results: int = 5) -> Dict[str, Any]:
        """Amazon search tool handler."""
        result = self.amazon_tool.search(keywords, domain, sort_by, max_results)
        return result.to_dict()
    
    async def _generate_recommendations(self, products: List[Dict], user_intent: str = "") -> Dict[str, Any]:
        """Generate recommendations tool handler."""
        recommendations = []
        for i, product in enumerate(products, 1):
            rec = {
                "rank": i,
                "product": product,
                "score": self._calculate_recommendation_score(product, None),
                "analysis": f"Analysis for {product.get('title', 'Unknown Product')}"
            }
            recommendations.append(rec)
        
        return {"recommendations": recommendations, "count": len(recommendations)}
    
    async def _get_system_status(self) -> Dict[str, Any]:
        """Get system status tool handler."""
        return await self.health_check()
    
    # Resource handlers
    async def _get_config_resource(self, uri: str) -> str:
        """Get system configuration."""
        import json
        config_info = {
            "server_name": self.name,
            "version": self.version,
            "configuration": {
                "intent_threshold": self.config.get("intent_threshold", 0.95),
                "keyword_method": self.config.get("keyword_method", "rake"),
                "default_amazon_domain": self.config.get("default_amazon_domain", "amazon.com"),
                "max_keywords": self.config.get("max_keywords", 5)
            },
            "features_enabled": {
                "intent_analysis": bool(self.prompt_analyzer),
                "keyword_extraction": bool(self.keyword_extractor),
                "amazon_search": bool(self.amazon_tool),
                "affiliate_links": bool(self.config.get("amazon_affiliate_id"))
            }
        }
        return json.dumps(config_info, indent=2)
    
    async def _get_health_resource(self, uri: str) -> str:
        """Get system health."""
        health = await self.health_check()
        import json
        return json.dumps(health, indent=2)
    
    async def _get_analytics_resource(self, uri: str) -> str:
        """Get system analytics."""
        analytics = self.get_stats()
        import json
        return json.dumps(analytics, indent=2)
    
    async def _get_models_resource(self, uri: str) -> str:
        """Get model information."""
        models_info = {
            "intent_model": {
                "loaded": bool(self.prompt_analyzer),
                "threshold": self.config.get("intent_threshold", 0.95),
                "model_id": self.config.get("huggingface_model_id", "PiGrieco/OpenSesame")
            },
            "keyword_extractor": {
                "loaded": bool(self.keyword_extractor),
                "method": self.config.get("keyword_method", "rake"),
                "max_keywords": self.config.get("max_keywords", 5)
            }
        }
        import json
        return json.dumps(models_info, indent=2)
    
    # Prompt handlers
    async def _generate_workflow_prompt(self, user_query: str, workflow_type: str = "product_search") -> List[Dict[str, Any]]:
        """Generate workflow execution prompt."""
        messages = [
            {
                "role": "system",
                "content": f"You are WordOfPrompt AI assistant. Execute a {workflow_type} workflow for the user's query."
            },
            {
                "role": "user", 
                "content": f"Query: {user_query}\nWorkflow: {workflow_type}"
            }
        ]
        return messages
    
    async def _generate_recommendation_prompt(self, user_profile: str, context: str = "") -> List[Dict[str, Any]]:
        """Generate personalized recommendation prompt."""
        messages = [
            {
                "role": "system",
                "content": "You are a product recommendation expert. Generate personalized recommendations based on user profile and context."
            },
            {
                "role": "user",
                "content": f"User Profile: {user_profile}"
            }
        ]
        
        if context:
            messages[-1]["content"] += f"\nContext: {context}"
        
        return messages
    
    # Helper methods
    def _calculate_recommendation_score(self, product, intent_result) -> float:
        """Calculate recommendation score for a product."""
        # Simplified scoring algorithm
        base_score = 0.5
        
        # Boost score based on rating
        if hasattr(product, 'rating') and product.rating:
            base_score += (product.rating / 5.0) * 0.3
        
        # Boost score based on review count
        if hasattr(product, 'reviews_count') and product.reviews_count:
            review_boost = min(product.reviews_count / 1000.0, 0.2)
            base_score += review_boost
        
        return min(base_score, 1.0)
    
    def _generate_pros(self, product) -> List[str]:
        """Generate pros for a product."""
        pros = []
        if hasattr(product, 'rating') and product.rating and product.rating >= 4.0:
            pros.append(f"Highly rated ({product.rating}/5 stars)")
        if hasattr(product, 'price') and product.price:
            pros.append("Competitive pricing")
        return pros
    
    def _generate_cons(self, product) -> List[str]:
        """Generate cons for a product."""
        cons = []
        if hasattr(product, 'rating') and product.rating and product.rating < 3.5:
            cons.append(f"Lower rating ({product.rating}/5 stars)")
        return cons
    
    def _generate_best_for(self, product, intent_result) -> str:
        """Generate best use case for a product."""
        return "General use based on search criteria"
