"""
Advanced prompt analysis with intent detection and keyword extraction.

This module provides sophisticated prompt analysis capabilities including
purchase intent detection, keyword extraction, and multi-modal analysis.
"""

import logging
from typing import Dict, List, Tuple, Union, Optional, Any
from dataclasses import dataclass
import sys
from pathlib import Path

# Add the src directory to Python path
src_dir = Path(__file__).parent.parent
sys.path.insert(0, str(src_dir))

from core.intent_classifier import IntentClassifier
from core.keyword_extractor import KeywordExtractor
from core.exceptions import IntentAnalysisError


logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Result of prompt analysis containing intent and keywords."""
    
    has_intent: bool
    intent_score: float
    intent_label: str
    keywords: List[str]
    original_prompt: str
    confidence: float
    metadata: Dict = None
    
    def to_dict(self) -> Dict:
        """Convert result to dictionary format."""
        return {
            "has_intent": self.has_intent,
            "intent_score": self.intent_score,
            "intent_label": self.intent_label,
            "keywords": self.keywords,
            "original_prompt": self.original_prompt,
            "confidence": self.confidence,
            "metadata": self.metadata or {}
        }


class PromptAnalyzer:
    """
    Advanced prompt analyzer with intent detection and keyword extraction.
    
    This class provides comprehensive prompt analysis capabilities including:
    - Purchase intent detection using fine-tuned models
    - Keyword extraction with multiple algorithms
    - Confidence scoring and threshold management
    - Multi-modal analysis support
    """

    def __init__(
        self,
        config: Dict,
        intent_threshold: float = 0.95,
        extract_keywords: bool = True,
        keyword_method: str = "rake"
    ):
        """
        Initialize the PromptAnalyzer.

        Args:
            config: Configuration dictionary with API keys and model settings
            intent_threshold: Threshold for intent detection (0.0-1.0)
            extract_keywords: Whether to extract keywords from prompts
            keyword_method: Method for keyword extraction ('rake', 'yake', 'textrank')
        """
        if not config:
            raise ValueError("Configuration is required for PromptAnalyzer")
            
        self.config = config
        self.intent_threshold = intent_threshold
        self.extract_keywords = extract_keywords
        self.keyword_method = keyword_method
        
        # Initialize components
        try:
            self.intent_classifier = IntentClassifier(config)
            self.keyword_extractor = KeywordExtractor(method=keyword_method)
        except Exception as e:
            logger.error(f"Failed to initialize PromptAnalyzer components: {e}")
            raise IntentAnalysisError(f"Initialization failed: {e}")
    
    def analyze(self, prompt: str, **kwargs) -> AnalysisResult:
        """
        Analyze a prompt for intent and extract keywords.

        Args:
            prompt: The text prompt to analyze
            **kwargs: Additional analysis parameters

        Returns:
            AnalysisResult: Comprehensive analysis results

        Raises:
            IntentAnalysisError: If analysis fails
        """
        if not prompt or not prompt.strip():
            raise IntentAnalysisError("Prompt cannot be empty")
            
        try:
            # Perform intent classification
            intent_result = self.intent_classifier.classify(prompt)
            
            # Extract keywords if enabled
            keywords = []
            if self.extract_keywords:
                keywords = self.keyword_extractor.extract(prompt)
            
            # Determine if intent threshold is met
            has_intent = intent_result.score >= self.intent_threshold
            
            # Calculate overall confidence
            confidence = self._calculate_confidence(intent_result, keywords)
            
            # Create result object
            result = AnalysisResult(
                has_intent=has_intent,
                intent_score=intent_result.score,
                intent_label=intent_result.label,
                keywords=keywords,
                original_prompt=prompt,
                confidence=confidence,
                metadata={
                    "threshold": self.intent_threshold,
                    "keyword_method": self.keyword_method,
                    "model_version": intent_result.model_version,
                    **kwargs
                }
            )
            
            logger.info(f"Analyzed prompt: intent={has_intent}, score={intent_result.score:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Prompt analysis failed: {e}")
            raise IntentAnalysisError(f"Analysis failed: {e}")
    
    def analyze_batch(self, prompts: List[str], **kwargs) -> List[AnalysisResult]:
        """
        Analyze multiple prompts in batch for efficiency.

        Args:
            prompts: List of text prompts to analyze
            **kwargs: Additional analysis parameters

        Returns:
            List[AnalysisResult]: Results for each prompt
        """
        results = []
        for prompt in prompts:
            try:
                result = self.analyze(prompt, **kwargs)
                results.append(result)
            except IntentAnalysisError as e:
                logger.warning(f"Failed to analyze prompt '{prompt[:50]}...': {e}")
                # Create error result
                error_result = AnalysisResult(
                    has_intent=False,
                    intent_score=0.0,
                    intent_label="error",
                    keywords=[],
                    original_prompt=prompt,
                    confidence=0.0,
                    metadata={"error": str(e)}
                )
                results.append(error_result)
        
        return results
    
    def update_threshold(self, new_threshold: float) -> None:
        """
        Update the intent detection threshold.

        Args:
            new_threshold: New threshold value (0.0-1.0)
        """
        if not 0.0 <= new_threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
        
        old_threshold = self.intent_threshold
        self.intent_threshold = new_threshold
        logger.info(f"Updated intent threshold: {old_threshold} -> {new_threshold}")
    
    def get_stats(self) -> Dict:
        """
        Get analyzer statistics and configuration.

        Returns:
            Dict: Current analyzer statistics
        """
        return {
            "intent_threshold": self.intent_threshold,
            "extract_keywords": self.extract_keywords,
            "keyword_method": self.keyword_method,
            "intent_classifier_stats": self.intent_classifier.get_stats(),
            "keyword_extractor_stats": self.keyword_extractor.get_stats()
        }
    
    def _calculate_confidence(self, intent_result, keywords: List[str]) -> float:
        """
        Calculate overall confidence score for the analysis.

        Args:
            intent_result: Intent classification result
            keywords: Extracted keywords

        Returns:
            float: Confidence score (0.0-1.0)
        """
        # Base confidence from intent score
        confidence = intent_result.score
        
        # Boost confidence if keywords are found
        if keywords:
            keyword_boost = min(len(keywords) * 0.05, 0.2)  # Max 20% boost
            confidence = min(confidence + keyword_boost, 1.0)
        
        # Penalize if score is close to threshold (uncertainty region)
        threshold_distance = abs(intent_result.score - self.intent_threshold)
        if threshold_distance < 0.1:
            uncertainty_penalty = (0.1 - threshold_distance) * 0.1
            confidence = max(confidence - uncertainty_penalty, 0.0)
        
        return confidence


# Legacy compatibility
class LegacyPromptAnalyzer:
    """Legacy compatibility wrapper for old PromptAnalyzer interface."""
    
    def __init__(self, config, keywords_extracted: bool = False):
        self.analyzer = PromptAnalyzer(
            config=config,
            extract_keywords=keywords_extracted
        )
    
    def analyze_prompt(self, prompt: str) -> Union[Tuple[bool, List[str], float, str], Tuple[bool, float, str]]:
        """Legacy method signature for backward compatibility."""
        result = self.analyzer.analyze(prompt)
        
        if self.analyzer.extract_keywords:
            return (
                result.has_intent,
                result.keywords,
                result.intent_score,
                result.original_prompt
            )
        else:
            return (
                result.has_intent,
                result.intent_score,
                result.original_prompt
            )
    
    @staticmethod
    def extract_keywords(prompt: str) -> List[str]:
        """Legacy static method for keyword extraction."""
        extractor = KeywordExtractor(method="rake")
        return extractor.extract(prompt)
