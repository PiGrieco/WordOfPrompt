"""
External API integrations and tools for product search and recommendations.

This module provides integrations with external services including
Amazon product search, affiliate link generation, and other e-commerce APIs.
"""

from .amazon_search import AmazonSearchTool, RainforestAPI
from .affiliate_links import AffiliateLinkGenerator
from .product_analyzer import ProductAnalyzer
from .validation_tools import ValidationTools

__all__ = [
    "AmazonSearchTool",
    "RainforestAPI", 
    "AffiliateLinkGenerator",
    "ProductAnalyzer",
    "ValidationTools",
]
