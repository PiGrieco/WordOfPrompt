"""
Data models for WordOfPrompt.

This module contains all data models, schemas, and database models
used throughout the WordOfPrompt system.
"""

from .user import UserModel
from .product import ProductModel
from .recommendation import RecommendationModel

__all__ = [
    "UserModel",
    "ProductModel", 
    "RecommendationModel",
]
