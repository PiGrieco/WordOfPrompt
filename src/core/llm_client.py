"""
LLM Client for unified language model interactions.

This module provides a unified interface for interacting with various
language models including OpenAI, Hugging Face, and self-hosted models.
"""

import logging
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

from core.exceptions import LLMError, ConfigurationError


logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    SELF_HOSTED = "self_hosted"
    ANTHROPIC = "anthropic"


@dataclass
class LLMResponse:
    """Response from LLM interaction."""
    
    content: str
    model: str
    provider: str
    tokens_used: Optional[int] = None
    cost: Optional[float] = None
    latency: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "content": self.content,
            "model": self.model,
            "provider": self.provider,
            "tokens_used": self.tokens_used,
            "cost": self.cost,
            "latency": self.latency,
            "metadata": self.metadata or {}
        }


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate text using the LLM."""
        pass
    
    @abstractmethod
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Chat with the LLM using message history."""
        pass
    
    @abstractmethod
    def get_provider(self) -> LLMProvider:
        """Get the LLM provider."""
        pass


class OpenAIClient(BaseLLMClient):
    """OpenAI LLM client implementation."""
    
    def __init__(self, api_key: str, model: str = "gpt-4-turbo", organization: str = None):
        self.api_key = api_key
        self.model = model
        self.organization = organization
        
        # Statistics
        self.request_count = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        
        logger.info(f"Initialized OpenAI client with model: {model}")
    
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate text using OpenAI model."""
        try:
            import time
            start_time = time.time()
            
            # This would make an actual OpenAI API call
            # For now, we'll simulate the response
            response_text = f"OpenAI response to: {prompt[:100]}..."
            
            # Simulate token counting
            estimated_tokens = len(prompt.split()) + len(response_text.split())
            estimated_cost = estimated_tokens * 0.00003  # Rough estimate
            
            latency = time.time() - start_time
            
            # Update statistics
            self.request_count += 1
            self.total_tokens += estimated_tokens
            self.total_cost += estimated_cost
            
            return LLMResponse(
                content=response_text,
                model=self.model,
                provider=LLMProvider.OPENAI.value,
                tokens_used=estimated_tokens,
                cost=estimated_cost,
                latency=latency,
                metadata={
                    "prompt_length": len(prompt),
                    "response_length": len(response_text)
                }
            )
            
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise LLMError(f"OpenAI generation failed: {e}")
    
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Chat with OpenAI model."""
        try:
            # Convert messages to prompt
            prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
            return await self.generate(prompt, **kwargs)
            
        except Exception as e:
            logger.error(f"OpenAI chat failed: {e}")
            raise LLMError(f"OpenAI chat failed: {e}")
    
    def get_provider(self) -> LLMProvider:
        """Get the provider."""
        return LLMProvider.OPENAI


class HuggingFaceClient(BaseLLMClient):
    """Hugging Face LLM client implementation."""
    
    def __init__(self, api_key: str, model_id: str = "PiGrieco/OpenSesame"):
        self.api_key = api_key
        self.model_id = model_id
        self.api_url = f"https://api-inference.huggingface.co/models/{model_id}"
        
        # Statistics
        self.request_count = 0
        
        logger.info(f"Initialized Hugging Face client with model: {model_id}")
    
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate text using Hugging Face model."""
        try:
            import time
            start_time = time.time()
            
            # This would make an actual HF API call
            # For now, we'll simulate the response
            response_text = f"HuggingFace response to: {prompt[:100]}..."
            
            latency = time.time() - start_time
            
            # Update statistics
            self.request_count += 1
            
            return LLMResponse(
                content=response_text,
                model=self.model_id,
                provider=LLMProvider.HUGGINGFACE.value,
                latency=latency,
                metadata={
                    "prompt_length": len(prompt),
                    "response_length": len(response_text)
                }
            )
            
        except Exception as e:
            logger.error(f"Hugging Face generation failed: {e}")
            raise LLMError(f"Hugging Face generation failed: {e}")
    
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Chat with Hugging Face model."""
        # Convert messages to prompt
        prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        return await self.generate(prompt, **kwargs)
    
    def get_provider(self) -> LLMProvider:
        """Get the provider."""
        return LLMProvider.HUGGINGFACE


class SelfHostedClient(BaseLLMClient):
    """Self-hosted LLM client implementation."""
    
    def __init__(self, base_url: str, model: str = "llama3", api_key: str = None):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        
        # Statistics
        self.request_count = 0
        
        logger.info(f"Initialized self-hosted client: {base_url}/{model}")
    
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate text using self-hosted model."""
        try:
            import time
            start_time = time.time()
            
            # This would make an actual API call to self-hosted model
            # For now, we'll simulate the response
            response_text = f"Self-hosted response to: {prompt[:100]}..."
            
            latency = time.time() - start_time
            
            # Update statistics
            self.request_count += 1
            
            return LLMResponse(
                content=response_text,
                model=self.model,
                provider=LLMProvider.SELF_HOSTED.value,
                latency=latency,
                metadata={
                    "base_url": self.base_url,
                    "prompt_length": len(prompt),
                    "response_length": len(response_text)
                }
            )
            
        except Exception as e:
            logger.error(f"Self-hosted generation failed: {e}")
            raise LLMError(f"Self-hosted generation failed: {e}")
    
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Chat with self-hosted model."""
        # Convert messages to prompt
        prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        return await self.generate(prompt, **kwargs)
    
    def get_provider(self) -> LLMProvider:
        """Get the provider."""
        return LLMProvider.SELF_HOSTED


class LLMClient:
    """
    Unified LLM client that can work with multiple providers.
    
    This class provides a unified interface for interacting with various
    language models through a single API.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the unified LLM client.

        Args:
            config: Configuration dictionary with provider settings
        """
        self.config = config
        self.clients: Dict[LLMProvider, BaseLLMClient] = {}
        self.default_provider = LLMProvider.OPENAI
        
        # Statistics
        self.total_requests = 0
        self.provider_usage = {provider.value: 0 for provider in LLMProvider}
        
        # Initialize clients based on configuration
        self._initialize_clients()
        
        logger.info("Initialized unified LLM client")
    
    def _initialize_clients(self):
        """Initialize LLM clients based on configuration."""
        try:
            # Initialize OpenAI client
            if self.config.get("openai_api_key"):
                self.clients[LLMProvider.OPENAI] = OpenAIClient(
                    api_key=self.config["openai_api_key"],
                    model=self.config.get("openai_model", "gpt-4-turbo"),
                    organization=self.config.get("openai_org_id")
                )
                logger.info("Initialized OpenAI client")
            
            # Initialize Hugging Face client
            if self.config.get("huggingface_api_key"):
                self.clients[LLMProvider.HUGGINGFACE] = HuggingFaceClient(
                    api_key=self.config["huggingface_api_key"],
                    model_id=self.config.get("huggingface_model_id", "PiGrieco/OpenSesame")
                )
                logger.info("Initialized Hugging Face client")
            
            # Initialize self-hosted client
            if self.config.get("self_hosted_url"):
                self.clients[LLMProvider.SELF_HOSTED] = SelfHostedClient(
                    base_url=self.config["self_hosted_url"],
                    model=self.config.get("self_hosted_model", "llama3"),
                    api_key=self.config.get("self_hosted_api_key")
                )
                logger.info("Initialized self-hosted client")
            
            if not self.clients:
                raise ConfigurationError("No LLM clients could be initialized")
            
            # Set default provider to the first available
            self.default_provider = list(self.clients.keys())[0]
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM clients: {e}")
            raise ConfigurationError(f"LLM client initialization failed: {e}")
    
    async def generate(self, prompt: str, provider: Optional[LLMProvider] = None, **kwargs) -> LLMResponse:
        """
        Generate text using the specified or default provider.

        Args:
            prompt: Text prompt to generate from
            provider: LLM provider to use (defaults to default_provider)
            **kwargs: Additional generation parameters

        Returns:
            LLMResponse: Generated response

        Raises:
            LLMError: If generation fails
        """
        provider = provider or self.default_provider
        
        if provider not in self.clients:
            raise LLMError(f"Provider {provider.value} not available")
        
        try:
            client = self.clients[provider]
            response = await client.generate(prompt, **kwargs)
            
            # Update statistics
            self.total_requests += 1
            self.provider_usage[provider.value] += 1
            
            logger.debug(f"Generated text using {provider.value}: {len(response.content)} chars")
            return response
            
        except Exception as e:
            logger.error(f"Text generation failed with {provider.value}: {e}")
            raise LLMError(f"Generation failed: {e}")
    
    async def chat(self, messages: List[Dict[str, str]], provider: Optional[LLMProvider] = None, **kwargs) -> LLMResponse:
        """
        Chat with the specified or default provider.

        Args:
            messages: List of chat messages
            provider: LLM provider to use (defaults to default_provider)
            **kwargs: Additional chat parameters

        Returns:
            LLMResponse: Chat response

        Raises:
            LLMError: If chat fails
        """
        provider = provider or self.default_provider
        
        if provider not in self.clients:
            raise LLMError(f"Provider {provider.value} not available")
        
        try:
            client = self.clients[provider]
            response = await client.chat(messages, **kwargs)
            
            # Update statistics
            self.total_requests += 1
            self.provider_usage[provider.value] += 1
            
            logger.debug(f"Chat completed using {provider.value}: {len(response.content)} chars")
            return response
            
        except Exception as e:
            logger.error(f"Chat failed with {provider.value}: {e}")
            raise LLMError(f"Chat failed: {e}")
    
    def get_available_providers(self) -> List[LLMProvider]:
        """Get list of available providers."""
        return list(self.clients.keys())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {
            "total_requests": self.total_requests,
            "provider_usage": self.provider_usage.copy(),
            "available_providers": [p.value for p in self.get_available_providers()],
            "default_provider": self.default_provider.value
        }
