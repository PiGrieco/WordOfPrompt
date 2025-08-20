"""
Recommendation data models for WordOfPrompt.
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime

from .product import ProductModel


@dataclass
class RecommendationModel:
    """Recommendation model for WordOfPrompt system."""
    
    id: str
    user_id: str
    query: str
    products: List[ProductModel]
    intent_score: float
    keywords: List[str]
    created_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "query": self.query,
            "products": [p.to_dict() for p in self.products],
            "intent_score": self.intent_score,
            "keywords": self.keywords,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "metadata": self.metadata or {}
        }
