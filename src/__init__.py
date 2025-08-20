"""
WordOfPrompt - AI-Powered Advertising Recommendation System

A sophisticated platform combining multi-agent workflows, real-time intent analysis,
and Amazon product recommendations for intelligent advertising ecosystems.
"""

__version__ = "1.0.0"
__author__ = "Piermatteo Grieco, Youness ElBrag"
__email__ = "piermatteo.grieco@aigot.com"
__license__ = "MIT"

from .core import *
from .models import *

__all__ = [
    # Core components
    "PromptAnalyzer",
    "IntentClassifier", 
    "KeywordExtractor",
    "LLMClient",
    
    # Models
    "UserModel",
    "ProductModel",
    "RecommendationModel",
    
    # Version info
    "__version__",
]
