"""
Product data models for WordOfPrompt.
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass


@dataclass
class ProductModel:
    """Product model for WordOfPrompt system."""
    
    asin: str
    title: str
    price: Optional[Dict[str, Any]] = None
    rating: Optional[float] = None
    reviews_count: Optional[int] = None
    image_url: Optional[str] = None
    product_url: Optional[str] = None
    affiliate_url: Optional[str] = None
    features: Optional[List[str]] = None
    availability: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "asin": self.asin,
            "title": self.title,
            "price": self.price,
            "rating": self.rating,
            "reviews_count": self.reviews_count,
            "image_url": self.image_url,
            "product_url": self.product_url,
            "affiliate_url": self.affiliate_url,
            "features": self.features or [],
            "availability": self.availability
        }
