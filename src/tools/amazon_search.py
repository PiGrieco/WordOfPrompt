"""
Amazon product search integration using Rainforest API.

This module provides comprehensive Amazon product search capabilities
with support for multiple domains, sorting options, and affiliate links.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from ..core.exceptions import APIError, ProductSearchError


logger = logging.getLogger(__name__)


class AmazonDomain(Enum):
    """Supported Amazon domains."""
    US = "amazon.com"
    UK = "amazon.co.uk"
    DE = "amazon.de"
    FR = "amazon.fr"
    IT = "amazon.it"
    ES = "amazon.es"
    CA = "amazon.ca"
    IN = "amazon.in"
    AU = "amazon.com.au"
    JP = "amazon.co.jp"


class SortBy(Enum):
    """Supported sorting options."""
    RELEVANCE = "relevance"
    PRICE_LOW_TO_HIGH = "price_low_to_high"
    PRICE_HIGH_TO_LOW = "price_high_to_low"
    NEWEST_ARRIVALS = "newest_arrivals"
    CUSTOMER_REVIEWS = "customer_reviews"
    FEATURED = "featured"


@dataclass
class ProductResult:
    """Individual product search result."""
    
    asin: str
    title: str
    price: Optional[Dict[str, Any]] = None
    rating: Optional[float] = None
    reviews_count: Optional[int] = None
    image_url: Optional[str] = None
    product_url: Optional[str] = None
    affiliate_url: Optional[str] = None
    position: Optional[int] = None
    availability: Optional[str] = None
    features: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    
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
            "position": self.position,
            "availability": self.availability,
            "features": self.features or [],
            "metadata": self.metadata or {}
        }


@dataclass
class SearchResult:
    """Complete search result with products and metadata."""
    
    products: List[ProductResult]
    query: str
    domain: str
    sort_by: str
    total_results: int
    search_time: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "products": [product.to_dict() for product in self.products],
            "query": self.query,
            "domain": self.domain,
            "sort_by": self.sort_by,
            "total_results": self.total_results,
            "search_time": self.search_time,
            "metadata": self.metadata
        }


class RainforestAPI:
    """
    Rainforest API client for Amazon product search.
    
    This class provides a robust interface to the Rainforest API
    with error handling, retries, and response parsing.
    """
    
    BASE_URL = "https://api.rainforestapi.com/request"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3
    ):
        """
        Initialize the Rainforest API client.

        Args:
            api_key: Rainforest API key (defaults to RAINFOREST_API_KEY env var)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.api_key = api_key or os.getenv("RAINFOREST_API_KEY")
        if not self.api_key:
            raise ValueError("Rainforest API key is required")
        
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        
        # Statistics
        self.request_count = 0
        self.error_count = 0
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def search_products(
        self,
        query: str,
        domain: str = "amazon.com",
        sort_by: str = "relevance",
        max_results: int = 10,
        **kwargs
    ) -> SearchResult:
        """
        Search for products on Amazon.

        Args:
            query: Search query/keywords
            domain: Amazon domain to search
            sort_by: Sort order for results
            max_results: Maximum number of results to return
            **kwargs: Additional search parameters

        Returns:
            SearchResult: Parsed search results

        Raises:
            ProductSearchError: If search fails
        """
        import time
        start_time = time.time()
        
        try:
            # Prepare parameters
            params = {
                "api_key": self.api_key,
                "type": "search",
                "amazon_domain": domain,
                "search_term": query,
                "sort_by": sort_by.replace("-", "_"),
                "max_page": min(max_results // 16 + 1, 10),  # Rainforest returns ~16 per page
                **kwargs
            }
            
            logger.info(f"Searching Amazon: query='{query}', domain='{domain}', sort='{sort_by}'")
            
            # Make request
            response = self.session.get(
                self.BASE_URL,
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            # Update statistics
            self.request_count += 1
            
            # Parse response
            data = response.json()
            search_time = time.time() - start_time
            
            return self._parse_search_response(
                data, query, domain, sort_by, search_time, max_results
            )
            
        except requests.RequestException as e:
            self.error_count += 1
            logger.error(f"Rainforest API request failed: {e}")
            raise ProductSearchError(f"Search request failed: {e}")
        except Exception as e:
            self.error_count += 1
            logger.error(f"Product search failed: {e}")
            raise ProductSearchError(f"Search failed: {e}")
    
    def _parse_search_response(
        self,
        data: Dict[str, Any],
        query: str,
        domain: str,
        sort_by: str,
        search_time: float,
        max_results: int
    ) -> SearchResult:
        """Parse Rainforest API response into SearchResult."""
        try:
            search_results = data.get("search_results", [])
            products = []
            
            for i, item in enumerate(search_results[:max_results]):
                try:
                    product = self._parse_product_item(item, i + 1)
                    products.append(product)
                except Exception as e:
                    logger.warning(f"Failed to parse product item: {e}")
                    continue
            
            return SearchResult(
                products=products,
                query=query,
                domain=domain,
                sort_by=sort_by,
                total_results=len(search_results),
                search_time=search_time,
                metadata={
                    "api_response_size": len(str(data)),
                    "products_parsed": len(products),
                    "products_skipped": len(search_results) - len(products)
                }
            )
            
        except Exception as e:
            raise ProductSearchError(f"Failed to parse search response: {e}")
    
    def _parse_product_item(self, item: Dict[str, Any], position: int) -> ProductResult:
        """Parse individual product item from API response."""
        try:
            # Extract basic information
            asin = item.get("asin", "")
            title = item.get("title", "")
            
            # Extract price information
            price_info = None
            if "price" in item and item["price"]:
                price_info = {
                    "value": item["price"].get("value"),
                    "currency": item["price"].get("currency"),
                    "symbol": item["price"].get("symbol"),
                    "raw": item["price"].get("raw")
                }
            
            # Extract rating and reviews
            rating = None
            reviews_count = None
            if "rating" in item:
                rating = float(item["rating"]) if item["rating"] else None
            if "ratings_total" in item:
                try:
                    reviews_count = int(item["ratings_total"])
                except (ValueError, TypeError):
                    reviews_count = None
            
            # Extract URLs
            image_url = item.get("image")
            product_url = item.get("link")
            
            # Extract features
            features = []
            if "features" in item and isinstance(item["features"], list):
                features = [str(feature) for feature in item["features"]]
            
            return ProductResult(
                asin=asin,
                title=title,
                price=price_info,
                rating=rating,
                reviews_count=reviews_count,
                image_url=image_url,
                product_url=product_url,
                position=position,
                availability=item.get("availability"),
                features=features,
                metadata={
                    "raw_item": item,
                    "extracted_at": time.time()
                }
            )
            
        except Exception as e:
            raise ProductSearchError(f"Failed to parse product item: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get API usage statistics."""
        return {
            "request_count": self.request_count,
            "error_count": self.error_count,
            "success_rate": (
                (self.request_count - self.error_count) / self.request_count
                if self.request_count > 0 else 0
            ),
            "api_key_configured": bool(self.api_key)
        }


class AmazonSearchTool:
    """
    High-level Amazon search tool with affiliate link generation.
    
    This class provides a user-friendly interface for Amazon product search
    with automatic affiliate link generation and domain optimization.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        affiliate_id: Optional[str] = None,
        default_domain: str = "amazon.com"
    ):
        """
        Initialize the Amazon search tool.

        Args:
            api_key: Rainforest API key
            affiliate_id: Amazon affiliate ID for link generation
            default_domain: Default Amazon domain to search
        """
        self.rainforest_api = RainforestAPI(api_key)
        self.affiliate_id = affiliate_id or os.getenv("AMAZON_AFFILIATE_ID")
        self.default_domain = default_domain
        
        # Statistics
        self.search_count = 0
        self.products_found = 0
    
    def search(
        self,
        keywords: str,
        domain: Optional[str] = None,
        sort_by: str = "relevance",
        max_results: int = 5,
        generate_affiliate_links: bool = True
    ) -> SearchResult:
        """
        Search for products with optional affiliate link generation.

        Args:
            keywords: Search keywords
            domain: Amazon domain (defaults to default_domain)
            sort_by: Sort order
            max_results: Maximum results to return
            generate_affiliate_links: Whether to generate affiliate links

        Returns:
            SearchResult: Search results with affiliate links
        """
        domain = domain or self.default_domain
        
        try:
            # Perform search
            result = self.rainforest_api.search_products(
                query=keywords,
                domain=domain,
                sort_by=sort_by,
                max_results=max_results
            )
            
            # Generate affiliate links if requested
            if generate_affiliate_links and self.affiliate_id:
                for product in result.products:
                    product.affiliate_url = self._generate_affiliate_link(
                        product.asin, domain
                    )
            
            # Update statistics
            self.search_count += 1
            self.products_found += len(result.products)
            
            logger.info(f"Found {len(result.products)} products for query: {keywords}")
            return result
            
        except Exception as e:
            logger.error(f"Amazon search failed: {e}")
            raise ProductSearchError(f"Search failed: {e}")
    
    def _generate_affiliate_link(self, asin: str, domain: str) -> str:
        """Generate affiliate link for a product."""
        if not self.affiliate_id or not asin:
            return f"https://{domain}/dp/{asin}"
        
        return f"https://{domain}/dp/{asin}/ref=nosim?tag={self.affiliate_id}"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get search tool statistics."""
        rainforest_stats = self.rainforest_api.get_stats()
        
        return {
            "search_count": self.search_count,
            "products_found": self.products_found,
            "avg_products_per_search": (
                self.products_found / self.search_count
                if self.search_count > 0 else 0
            ),
            "affiliate_id_configured": bool(self.affiliate_id),
            "rainforest_api_stats": rainforest_stats
        }


# Legacy compatibility
class SearchTools:
    """Legacy compatibility wrapper for old SearchTools interface."""
    
    def __init__(self):
        self.amazon_tool = AmazonSearchTool()
    
    @staticmethod
    def search_rainforest(keywords: str, domain: str, sort_by: str, n: int = 5) -> Dict[str, Any]:
        """Legacy method for Rainforest API search."""
        tool = AmazonSearchTool()
        result = tool.search(
            keywords=keywords,
            domain=domain,
            sort_by=sort_by,
            max_results=n
        )
        
        # Convert to legacy format
        return {
            "search_results": [product.to_dict() for product in result.products]
        }
