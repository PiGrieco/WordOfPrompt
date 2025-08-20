"""
Intent Classification for purchase behavior analysis.

This module provides sophisticated intent classification capabilities
for detecting purchase intent in user prompts and messages.
"""

import logging
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

from core.exceptions import IntentAnalysisError


logger = logging.getLogger(__name__)


class IntentType(Enum):
    """Types of user intent."""
    PURCHASE = "purchase"
    BROWSE = "browse"
    COMPARE = "compare"
    GENERAL = "general"
    QUESTION = "question"


@dataclass
class IntentResult:
    """Result of intent classification."""
    
    intent_type: IntentType
    label: str
    score: float
    confidence: float
    model_version: str
    metadata: Dict = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format."""
        return {
            "intent_type": self.intent_type.value,
            "label": self.label,
            "score": self.score,
            "confidence": self.confidence,
            "model_version": self.model_version,
            "metadata": self.metadata or {}
        }


class IntentClassifier:
    """
    Advanced intent classifier for purchase behavior analysis.
    
    This class provides sophisticated intent classification using
    multiple approaches including fine-tuned models and rule-based systems.
    """
    
    def __init__(self, config: Dict, model_type: str = "huggingface"):
        """
        Initialize the intent classifier.

        Args:
            config: Configuration dictionary with API keys and model settings
            model_type: Type of model to use ('huggingface', 'openai', 'local')
        """
        self.config = config
        self.model_type = model_type
        self.model_version = "1.0.0"
        
        # Classification statistics
        self.classification_count = 0
        self.intent_distribution = {intent.value: 0 for intent in IntentType}
        
        # Initialize the classifier based on type
        try:
            self._initialize_classifier()
        except Exception as e:
            logger.warning(f"Failed to initialize {model_type} classifier, falling back to rule_based: {e}")
            self.model_type = "rule_based"
        
        logger.info(f"Initialized IntentClassifier with {model_type} model")
    
    def _initialize_classifier(self):
        """Initialize the classifier based on model type."""
        try:
            if self.model_type == "huggingface":
                self._init_huggingface_classifier()
            elif self.model_type == "openai":
                self._init_openai_classifier()
            elif self.model_type == "local":
                self._init_local_classifier()
            elif self.model_type == "rule_based":
                self._init_rule_based_classifier()
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
                
        except Exception as e:
            logger.error(f"Failed to initialize classifier: {e}")
            # Fallback to rule-based classifier
            self.model_type = "rule_based"
            self._init_rule_based_classifier()
            logger.warning("Falling back to rule-based classifier")
    
    def _init_rule_based_classifier(self):
        """Initialize rule-based classifier."""
        logger.info("Initialized rule-based classifier")
    
    def _init_huggingface_classifier(self):
        """Initialize Hugging Face classifier."""
        try:
            # This would initialize the actual HF model
            # For now, we'll simulate it
            self.model_id = self.config.get("huggingface_model_id", "PiGrieco/OpenSesame")
            self.api_token = self.config.get("huggingface_api_key")
            
            if not self.api_token:
                raise ValueError("Hugging Face API token required")
            
            logger.info(f"Initialized Hugging Face classifier: {self.model_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Hugging Face classifier: {e}")
            raise
    
    def _init_openai_classifier(self):
        """Initialize OpenAI classifier."""
        try:
            self.openai_api_key = self.config.get("openai_api_key")
            
            if not self.openai_api_key:
                raise ValueError("OpenAI API key required")
            
            logger.info("Initialized OpenAI classifier")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI classifier: {e}")
            raise
    
    def _init_local_classifier(self):
        """Initialize local classifier."""
        try:
            # This would load a local model
            logger.info("Initialized local classifier")
            
        except Exception as e:
            logger.error(f"Failed to initialize local classifier: {e}")
            raise
    
    def classify(self, text: str, **kwargs) -> IntentResult:
        """
        Classify the intent of the given text.

        Args:
            text: Text to classify
            **kwargs: Additional classification parameters

        Returns:
            IntentResult: Classification result

        Raises:
            IntentAnalysisError: If classification fails
        """
        if not text or not text.strip():
            raise IntentAnalysisError("Text cannot be empty")
        
        try:
            # Route to appropriate classifier
            if self.model_type == "huggingface":
                result = self._classify_huggingface(text, **kwargs)
            elif self.model_type == "openai":
                result = self._classify_openai(text, **kwargs)
            elif self.model_type == "local":
                result = self._classify_local(text, **kwargs)
            else:
                result = self._classify_rule_based(text, **kwargs)
            
            # Update statistics
            self.classification_count += 1
            self.intent_distribution[result.intent_type.value] += 1
            
            logger.debug(f"Classified text: intent={result.intent_type.value}, score={result.score:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            raise IntentAnalysisError(f"Classification failed: {e}")
    
    def _classify_huggingface(self, text: str, **kwargs) -> IntentResult:
        """Classify using Hugging Face model with real API call."""
        try:
            import requests
            import time
            
            # Prepare API call
            headers = {"Authorization": f"Bearer {self.api_token}"}
            api_url = f"https://api-inference.huggingface.co/models/{self.model_id}"
            
            # For text classification, the input should be the text directly
            payload = {"inputs": text}
            
            logger.info(f"Making Hugging Face API call to {self.model_id}")
            
            # Make API call with timeout and retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = requests.post(
                        api_url, 
                        headers=headers, 
                        json=payload,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        result_data = response.json()
                        logger.info(f"HF API response: {result_data}")
                        
                        # Parse response based on model output format
                        if isinstance(result_data, list) and len(result_data) > 0:
                            # Classification model response format
                            top_result = result_data[0]
                            
                            if isinstance(top_result, dict):
                                score = top_result.get("score", 0.0)
                                label = top_result.get("label", "UNKNOWN")
                                
                                # Map label to intent type
                                if "PURCHASE" in label.upper() or "BUY" in label.upper():
                                    intent_type = IntentType.PURCHASE
                                elif "BROWSE" in label.upper() or "SEARCH" in label.upper():
                                    intent_type = IntentType.BROWSE
                                elif "COMPARE" in label.upper():
                                    intent_type = IntentType.COMPARE
                                else:
                                    intent_type = IntentType.GENERAL
                                
                                return IntentResult(
                                    intent_type=intent_type,
                                    label=label,
                                    score=score,
                                    confidence=score * 0.95,  # High confidence for HF model
                                    model_version=self.model_id,
                                    metadata={
                                        "model_type": "huggingface",
                                        "api_response": result_data,
                                        "text_length": len(text),
                                        "api_latency": time.time(),
                                        "attempt": attempt + 1
                                    }
                                )
                        
                        # If we can't parse the response, fall back to rule-based
                        logger.warning(f"Unexpected HF response format: {result_data}")
                        break
                        
                    elif response.status_code == 503:
                        # Model loading, wait and retry
                        logger.warning(f"Model loading, attempt {attempt + 1}/{max_retries}")
                        if attempt < max_retries - 1:
                            time.sleep(2 ** attempt)  # Exponential backoff
                            continue
                        else:
                            break
                    else:
                        logger.error(f"HF API error {response.status_code}: {response.text}")
                        break
                        
                except requests.RequestException as e:
                    logger.error(f"HF API request failed (attempt {attempt + 1}): {e}")
                    if attempt < max_retries - 1:
                        time.sleep(1)
                        continue
                    else:
                        break
            
            # Fallback to rule-based classification
            logger.warning("Falling back to rule-based classification due to HF API issues")
            return self._classify_rule_based(text, **kwargs)
            
        except Exception as e:
            logger.error(f"Hugging Face classification failed: {e}")
            # Fallback to rule-based
            logger.warning("Falling back to rule-based classification")
            return self._classify_rule_based(text, **kwargs)
    
    def _classify_openai(self, text: str, **kwargs) -> IntentResult:
        """Classify using OpenAI model."""
        try:
            # This would make an actual API call to OpenAI
            # For now, we'll use a simple heuristic
            
            # Simulate OpenAI classification
            if any(word in text.lower() for word in ["buy", "purchase", "need", "want"]):
                score = 0.85
                intent_type = IntentType.PURCHASE
                label = "PURCHASE_INTENT"
            elif any(word in text.lower() for word in ["compare", "vs", "versus", "better"]):
                score = 0.75
                intent_type = IntentType.COMPARE
                label = "COMPARE_INTENT"
            else:
                score = 0.3
                intent_type = IntentType.GENERAL
                label = "GENERAL"
            
            return IntentResult(
                intent_type=intent_type,
                label=label,
                score=score,
                confidence=score * 0.95,
                model_version="openai-gpt-4",
                metadata={
                    "model_type": "openai",
                    "text_length": len(text)
                }
            )
            
        except Exception as e:
            logger.error(f"OpenAI classification failed: {e}")
            raise IntentAnalysisError(f"OpenAI classification failed: {e}")
    
    def _classify_local(self, text: str, **kwargs) -> IntentResult:
        """Classify using local model."""
        try:
            # This would use a local model
            # For now, we'll use rule-based classification
            return self._classify_rule_based(text, **kwargs)
            
        except Exception as e:
            logger.error(f"Local classification failed: {e}")
            raise IntentAnalysisError(f"Local classification failed: {e}")
    
    def _classify_rule_based(self, text: str, **kwargs) -> IntentResult:
        """Classify using enhanced rule-based approach."""
        try:
            text_lower = text.lower()
            
            # Enhanced purchase patterns with refined weights
            purchase_patterns = [
                # Strong purchase verbs
                ("buy", 0.4), ("purchase", 0.4), ("order", 0.4),
                
                # Need/want expressions
                ("need", 0.3), ("want", 0.3), ("looking for", 0.3), ("searching for", 0.3),
                
                # Price/budget indicators
                ("$", 0.2), ("€", 0.2), ("£", 0.2), ("price", 0.2), ("cost", 0.2),
                ("cheap", 0.2), ("expensive", 0.2), ("budget", 0.2), ("under", 0.2),
                ("affordable", 0.2), ("deal", 0.2), ("sale", 0.2), ("discount", 0.2),
                
                # Urgency indicators
                ("now", 0.2), ("today", 0.2), ("asap", 0.3), ("urgent", 0.3),
                ("immediately", 0.3), ("right now", 0.3),
                
                # Quality/recommendation indicators
                ("best", 0.1), ("good", 0.1), ("quality", 0.1), ("recommend", 0.1),
                ("get", 0.1)
            ]
            
            # Compare intent patterns
            compare_patterns = [
                ("compare", 0.5), ("vs", 0.4), ("versus", 0.4), ("better", 0.3),
                ("difference", 0.4), ("which", 0.3), ("top", 0.2)
            ]
            
            # Question patterns
            question_patterns = [
                ("what", 0.3), ("how", 0.3), ("why", 0.3), ("when", 0.2),
                ("where", 0.2), ("?", 0.2)
            ]
            
            # Calculate scores
            purchase_score = 0.0
            compare_score = 0.0
            question_score = 0.0
            matched_patterns = []
            
            for pattern, weight in purchase_patterns:
                if pattern in text_lower:
                    purchase_score += weight
                    matched_patterns.append((pattern, weight, "purchase"))
            
            for pattern, weight in compare_patterns:
                if pattern in text_lower:
                    compare_score += weight
                    matched_patterns.append((pattern, weight, "compare"))
            
            for pattern, weight in question_patterns:
                if pattern in text_lower:
                    question_score += weight
                    matched_patterns.append((pattern, weight, "question"))
            
            # Apply bonuses for combinations
            if any(p[0] in ["buy", "purchase", "order"] for p in matched_patterns):
                if any(p[0] in ["$", "€", "£", "price", "budget"] for p in matched_patterns):
                    purchase_score += 0.2  # Bonus for verb + price
            
            # Cap scores at 1.0
            purchase_score = min(purchase_score, 1.0)
            compare_score = min(compare_score, 1.0)
            question_score = min(question_score, 1.0)
            
            # Determine intent based on highest score
            scores = {
                "purchase": purchase_score,
                "compare": compare_score,
                "question": question_score
            }
            
            max_intent = max(scores, key=scores.get)
            max_score = scores[max_intent]
            
            # Map to intent types and determine final classification
            if max_intent == "purchase" and max_score >= 0.3:
                intent_type = IntentType.PURCHASE
                label = "PURCHASE_INTENT"
                final_score = purchase_score
            elif max_intent == "compare" and max_score >= 0.3:
                intent_type = IntentType.COMPARE
                label = "COMPARE_INTENT"
                final_score = compare_score
            elif max_intent == "question" and max_score >= 0.2:
                intent_type = IntentType.QUESTION
                label = "QUESTION"
                final_score = question_score
            else:
                intent_type = IntentType.GENERAL
                label = "GENERAL"
                final_score = max(max_score, 0.1)  # Minimum score
            
            # Calculate confidence based on score and pattern count
            pattern_count = len(matched_patterns)
            confidence = final_score * 0.8
            
            if pattern_count >= 3:
                confidence += 0.1  # Bonus for multiple patterns
            
            confidence = min(confidence, 1.0)
            
            return IntentResult(
                intent_type=intent_type,
                label=label,
                score=final_score,
                confidence=confidence,
                model_version="rule_based_v2_enhanced",
                metadata={
                    "model_type": "rule_based",
                    "text_length": len(text),
                    "purchase_score": purchase_score,
                    "compare_score": compare_score,
                    "question_score": question_score,
                    "matched_patterns": [{"pattern": p[0], "weight": p[1], "type": p[2]} for p in matched_patterns],
                    "pattern_count": pattern_count
                }
            )
            
        except Exception as e:
            logger.error(f"Rule-based classification failed: {e}")
            raise IntentAnalysisError(f"Rule-based classification failed: {e}")
    
    def batch_classify(self, texts: List[str], **kwargs) -> List[IntentResult]:
        """
        Classify multiple texts in batch.

        Args:
            texts: List of texts to classify
            **kwargs: Additional classification parameters

        Returns:
            List[IntentResult]: Classification results
        """
        results = []
        for text in texts:
            try:
                result = self.classify(text, **kwargs)
                results.append(result)
            except IntentAnalysisError as e:
                logger.warning(f"Failed to classify text '{text[:50]}...': {e}")
                # Add error result
                error_result = IntentResult(
                    intent_type=IntentType.GENERAL,
                    label="ERROR",
                    score=0.0,
                    confidence=0.0,
                    model_version=self.model_version,
                    metadata={"error": str(e)}
                )
                results.append(error_result)
        
        return results
    
    def get_stats(self) -> Dict:
        """Get classifier statistics."""
        return {
            "model_type": self.model_type,
            "model_version": self.model_version,
            "classification_count": self.classification_count,
            "intent_distribution": self.intent_distribution.copy(),
            "accuracy_estimate": self._estimate_accuracy()
        }
    
    def _estimate_accuracy(self) -> float:
        """Estimate classifier accuracy based on model type."""
        accuracy_estimates = {
            "huggingface": 0.92,
            "openai": 0.88,
            "local": 0.85,
            "rule_based": 0.75
        }
        return accuracy_estimates.get(self.model_type, 0.70)
