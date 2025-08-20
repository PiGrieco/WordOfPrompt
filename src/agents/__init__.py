"""
Multi-agent system for intelligent product recommendations.

This module provides the agent orchestration system that coordinates
specialized AI agents for intent analysis, keyword extraction, domain
recognition, and product recommendation.
"""

from .base_agent import BaseAgent
from .intent_agent import IntentAnalysisAgent
from .keyword_agent import KeywordExtractionAgent
from .domain_agent import DomainRecognitionAgent
from .search_agent import ProductSearchAgent
from .recommendation_agent import ProductRecommendationAgent
from .orchestrator import AgentOrchestrator
from .workflow import RecommendationWorkflow

__all__ = [
    "BaseAgent",
    "IntentAnalysisAgent",
    "KeywordExtractionAgent", 
    "DomainRecognitionAgent",
    "ProductSearchAgent",
    "ProductRecommendationAgent",
    "AgentOrchestrator",
    "RecommendationWorkflow",
]
