"""
Custom exceptions for the WordOfPrompt platform (Fixed version).
"""

from typing import Optional, Dict, Any


class WordOfPromptError(Exception):
    """Base exception class for all WordOfPrompt errors."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary format."""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details
        }


class IntentAnalysisError(WordOfPromptError):
    """Exception raised when intent analysis fails."""
    pass


class KeywordExtractionError(WordOfPromptError):
    """Exception raised when keyword extraction fails."""
    pass


class LLMError(WordOfPromptError):
    """Exception raised when LLM operations fail."""
    pass


class ConfigurationError(WordOfPromptError):
    """Exception raised when configuration is invalid."""
    pass


class APIError(WordOfPromptError):
    """Exception raised when external API calls fail."""
    pass


class ProductSearchError(WordOfPromptError):
    """Exception raised when product search operations fail."""
    pass


class RecommendationError(WordOfPromptError):
    """Exception raised when recommendation generation fails."""
    pass
