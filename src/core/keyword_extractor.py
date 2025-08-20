"""
Advanced keyword extraction with multiple algorithms and optimization.

This module provides sophisticated keyword extraction capabilities using
various algorithms including RAKE, YAKE, TextRank, and custom methods.
"""

import logging
from typing import List, Dict, Optional, Union, Tuple, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

try:
    import nltk
    from rake_nltk import Rake
    RAKE_AVAILABLE = True
except ImportError:
    RAKE_AVAILABLE = False
    logging.warning("RAKE dependencies not available")

try:
    from yake import KeywordExtractor as YakeExtractor
    YAKE_AVAILABLE = True
except ImportError:
    YAKE_AVAILABLE = False
    logging.warning("YAKE dependencies not available")

try:
    import textstat
    TEXTSTAT_AVAILABLE = True
except ImportError:
    TEXTSTAT_AVAILABLE = False
    logging.warning("TextStat dependencies not available")

from core.exceptions import KeywordExtractionError


logger = logging.getLogger(__name__)


@dataclass
class KeywordResult:
    """Result of keyword extraction with metadata."""
    
    keywords: List[str]
    scores: List[float]
    method: str
    confidence: float
    metadata: Dict = None
    
    def to_dict(self) -> Dict:
        """Convert result to dictionary format."""
        return {
            "keywords": self.keywords,
            "scores": self.scores,
            "method": self.method,
            "confidence": self.confidence,
            "metadata": self.metadata or {}
        }


class BaseKeywordExtractor(ABC):
    """Abstract base class for keyword extractors."""
    
    @abstractmethod
    def extract(self, text: str, max_keywords: int = 5) -> KeywordResult:
        """Extract keywords from text."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get extractor name."""
        pass


class RakeKeywordExtractor(BaseKeywordExtractor):
    """RAKE (Rapid Automatic Keyword Extraction) algorithm implementation."""
    
    def __init__(self, language: str = "en"):
        self.language = language
        if RAKE_AVAILABLE:
            self.extractor = Rake()
        else:
            raise KeywordExtractionError("RAKE dependencies not available")
        
    def extract(self, text: str, max_keywords: int = 5) -> KeywordResult:
        """Extract keywords using RAKE algorithm."""
        try:
            self.extractor.extract_keywords_from_text(text)
            ranked_phrases = self.extractor.get_ranked_phrases()
            scores = self.extractor.get_ranked_phrases_with_scores()
            
            keywords = ranked_phrases[:max_keywords]
            keyword_scores = [score for score, _ in scores[:max_keywords]]
            
            # Normalize scores to 0-1 range
            if keyword_scores:
                max_score = max(keyword_scores)
                normalized_scores = [score / max_score for score in keyword_scores]
            else:
                normalized_scores = []
            
            confidence = self._calculate_confidence(keywords, normalized_scores)
            
            return KeywordResult(
                keywords=keywords,
                scores=normalized_scores,
                method="rake",
                confidence=confidence,
                metadata={"total_phrases": len(ranked_phrases)}
            )
            
        except Exception as e:
            raise KeywordExtractionError(f"RAKE extraction failed: {e}")
    
    def get_name(self) -> str:
        return "rake"
    
    def _calculate_confidence(self, keywords: List[str], scores: List[float]) -> float:
        """Calculate confidence based on keyword quality."""
        if not keywords:
            return 0.0
        
        # Base confidence from average score
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        # Boost for longer keywords (more specific)
        length_bonus = min(sum(len(kw.split()) for kw in keywords) / len(keywords) * 0.1, 0.2)
        
        return min(avg_score + length_bonus, 1.0)


class YakeKeywordExtractor(BaseKeywordExtractor):
    """YAKE (Yet Another Keyword Extractor) algorithm implementation."""
    
    def __init__(self, language: str = "en", n: int = 3, deduplim: float = 0.9):
        self.language = language
        self.n = n
        self.deduplim = deduplim
        
    def extract(self, text: str, max_keywords: int = 5) -> KeywordResult:
        """Extract keywords using YAKE algorithm."""
        try:
            extractor = YakeExtractor(
                lan=self.language,
                n=self.n,
                dedupLim=self.deduplim,
                top=max_keywords,
                features=None
            )
            
            keyword_tuples = extractor.extract_keywords(text)
            keywords = [kw for kw, _ in keyword_tuples]
            scores = [score for _, score in keyword_tuples]
            
            # YAKE scores are lower = better, so invert them
            if scores:
                max_score = max(scores)
                normalized_scores = [(max_score - score) / max_score for score in scores]
            else:
                normalized_scores = []
            
            confidence = self._calculate_confidence(keywords, normalized_scores)
            
            return KeywordResult(
                keywords=keywords,
                scores=normalized_scores,
                method="yake",
                confidence=confidence,
                metadata={
                    "n_gram": self.n,
                    "dedup_limit": self.deduplim
                }
            )
            
        except Exception as e:
            raise KeywordExtractionError(f"YAKE extraction failed: {e}")
    
    def get_name(self) -> str:
        return "yake"
    
    def _calculate_confidence(self, keywords: List[str], scores: List[float]) -> float:
        """Calculate confidence based on keyword quality."""
        if not keywords:
            return 0.0
        
        return sum(scores) / len(scores) if scores else 0.0


class SimpleKeywordExtractor(BaseKeywordExtractor):
    """Simple keyword extractor using basic text analysis (fallback)."""
    
    def __init__(self, language: str = "en"):
        self.language = language
        
    def extract(self, text: str, max_keywords: int = 5) -> KeywordResult:
        """Extract keywords using simple text analysis."""
        try:
            import re
            
            # Simple keyword extraction using regex and word frequency
            # Remove punctuation and convert to lowercase
            clean_text = re.sub(r'[^\w\s]', ' ', text.lower())
            words = clean_text.split()
            
            # Filter common stop words
            stop_words = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
                'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
                'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us',
                'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'this',
                'that', 'these', 'those'
            }
            
            # Filter words
            filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
            
            # Count word frequency
            word_freq = {}
            for word in filtered_words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            # Sort by frequency and take top keywords
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            keywords = [word for word, freq in sorted_words[:max_keywords]]
            scores = [freq / max(word_freq.values()) for word, freq in sorted_words[:max_keywords]]
            
            confidence = sum(scores) / len(scores) if scores else 0.0
            
            return KeywordResult(
                keywords=keywords,
                scores=scores,
                method="simple",
                confidence=confidence,
                metadata={"total_words": len(words), "unique_words": len(word_freq)}
            )
            
        except Exception as e:
            raise KeywordExtractionError(f"Simple extraction failed: {e}")
    
    def get_name(self) -> str:
        return "simple"


class HybridKeywordExtractor(BaseKeywordExtractor):
    """Hybrid extractor combining multiple algorithms."""
    
    def __init__(self, methods: List[str] = None, weights: Dict[str, float] = None):
        self.methods = methods or ["rake", "yake"]
        self.weights = weights or {"rake": 0.6, "yake": 0.4}
        
        self.extractors = {}
        for method in self.methods:
            if method == "rake":
                self.extractors[method] = RakeKeywordExtractor()
            elif method == "yake":
                self.extractors[method] = YakeKeywordExtractor()
    
    def extract(self, text: str, max_keywords: int = 5) -> KeywordResult:
        """Extract keywords using multiple algorithms and combine results."""
        try:
            all_results = {}
            keyword_scores = {}
            
            # Run all extractors
            for method, extractor in self.extractors.items():
                result = extractor.extract(text, max_keywords * 2)  # Get more for merging
                all_results[method] = result
                
                # Accumulate weighted scores
                for kw, score in zip(result.keywords, result.scores):
                    weight = self.weights.get(method, 1.0)
                    if kw in keyword_scores:
                        keyword_scores[kw] += score * weight
                    else:
                        keyword_scores[kw] = score * weight
            
            # Sort by combined scores and take top keywords
            sorted_keywords = sorted(
                keyword_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:max_keywords]
            
            keywords = [kw for kw, _ in sorted_keywords]
            scores = [score for _, score in sorted_keywords]
            
            # Normalize scores
            if scores:
                max_score = max(scores)
                normalized_scores = [score / max_score for score in scores]
            else:
                normalized_scores = []
            
            confidence = sum(normalized_scores) / len(normalized_scores) if normalized_scores else 0.0
            
            return KeywordResult(
                keywords=keywords,
                scores=normalized_scores,
                method="hybrid",
                confidence=confidence,
                metadata={
                    "methods_used": self.methods,
                    "weights": self.weights,
                    "individual_results": {k: v.to_dict() for k, v in all_results.items()}
                }
            )
            
        except Exception as e:
            raise KeywordExtractionError(f"Hybrid extraction failed: {e}")
    
    def get_name(self) -> str:
        return "hybrid"


class KeywordExtractor:
    """
    Main keyword extractor with support for multiple algorithms.
    
    This class provides a unified interface for keyword extraction using
    various algorithms and optimization techniques.
    """
    
    SUPPORTED_METHODS = ["rake", "yake", "hybrid", "simple"]
    
    def __init__(
        self,
        method: str = "rake",
        language: str = "en",
        max_keywords: int = 5,
        **kwargs
    ):
        """
        Initialize the KeywordExtractor.

        Args:
            method: Extraction method ('rake', 'yake', 'hybrid')
            language: Language code for text processing
            max_keywords: Maximum number of keywords to extract
            **kwargs: Additional method-specific parameters
        """
        self.method = method
        self.language = language
        self.max_keywords = max_keywords
        self.kwargs = kwargs
        
        # Statistics
        self.extraction_count = 0
        self.total_keywords_extracted = 0
        
        # Initialize extractor
        self.extractor = self._create_extractor()
    
    def _create_extractor(self) -> BaseKeywordExtractor:
        """Create the appropriate extractor based on method."""
        if self.method == "rake":
            if RAKE_AVAILABLE:
                return RakeKeywordExtractor(language=self.language)
            else:
                # Fallback to simple extractor
                return SimpleKeywordExtractor(language=self.language)
        elif self.method == "yake":
            if YAKE_AVAILABLE:
                return YakeKeywordExtractor(
                    language=self.language,
                    **{k: v for k, v in self.kwargs.items() if k in ["n", "deduplim"]}
                )
            else:
                return SimpleKeywordExtractor(language=self.language)
        elif self.method == "hybrid":
            return HybridKeywordExtractor(
                methods=self.kwargs.get("methods", ["simple"]),
                weights=self.kwargs.get("weights", {"simple": 1.0})
            )
        elif self.method == "simple":
            return SimpleKeywordExtractor(language=self.language)
        else:
            raise ValueError(f"Unsupported method: {self.method}")
    
    def extract(self, text: str, max_keywords: Optional[int] = None) -> List[str]:
        """
        Extract keywords from text.

        Args:
            text: Input text to extract keywords from
            max_keywords: Maximum number of keywords (overrides default)

        Returns:
            List[str]: Extracted keywords

        Raises:
            KeywordExtractionError: If extraction fails
        """
        if not text or not text.strip():
            return []
        
        max_kw = max_keywords or self.max_keywords
        
        try:
            result = self.extractor.extract(text, max_kw)
            
            # Update statistics
            self.extraction_count += 1
            self.total_keywords_extracted += len(result.keywords)
            
            logger.debug(f"Extracted {len(result.keywords)} keywords using {self.method}")
            return result.keywords
            
        except Exception as e:
            logger.error(f"Keyword extraction failed: {e}")
            raise KeywordExtractionError(f"Extraction failed: {e}")
    
    def extract_with_scores(self, text: str, max_keywords: Optional[int] = None) -> KeywordResult:
        """
        Extract keywords with confidence scores and metadata.

        Args:
            text: Input text to extract keywords from
            max_keywords: Maximum number of keywords (overrides default)

        Returns:
            KeywordResult: Detailed extraction results
        """
        if not text or not text.strip():
            return KeywordResult([], [], self.method, 0.0)
        
        max_kw = max_keywords or self.max_keywords
        result = self.extractor.extract(text, max_kw)
        
        # Update statistics
        self.extraction_count += 1
        self.total_keywords_extracted += len(result.keywords)
        
        return result
    
    def get_stats(self) -> Dict:
        """Get extractor statistics."""
        return {
            "method": self.method,
            "language": self.language,
            "max_keywords": self.max_keywords,
            "extraction_count": self.extraction_count,
            "total_keywords_extracted": self.total_keywords_extracted,
            "avg_keywords_per_extraction": (
                self.total_keywords_extracted / self.extraction_count
                if self.extraction_count > 0 else 0
            )
        }
    
    @classmethod
    def get_supported_methods(cls) -> List[str]:
        """Get list of supported extraction methods."""
        return cls.SUPPORTED_METHODS.copy()


# Legacy compatibility
class KeywordExtractors:
    """Legacy compatibility wrapper for old KeywordExtractors interface."""
    
    def __init__(self, prompt: str, language: str = "en", max_keywords: int = 5, method: str = "rake"):
        self.extractor = KeywordExtractor(
            method=method,
            language=language,
            max_keywords=max_keywords
        )
        self.keywords = self.extractor.extract(prompt)
    
    def extract_keywords(self, prompt: str) -> List[str]:
        """Legacy method for keyword extraction."""
        return self.extractor.extract(prompt)
