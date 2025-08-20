"""
Core utilities for WordOfPrompt (Fixed version).
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Setup logging configuration."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler()]
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=handlers
    )


def get_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from environment variables."""
    config = {
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "rainforest_api_key": os.getenv("RAINFOREST_API_KEY"),
        "amazon_affiliate_id": os.getenv("AMAZON_AFFILIATE_ID"),
        "huggingface_api_key": os.getenv("HUGGINGFACE_API_KEY"),
        "intent_threshold": float(os.getenv("INTENT_THRESHOLD", "0.95")),
        "keyword_method": os.getenv("KEYWORD_METHOD", "simple"),
        "max_keywords": int(os.getenv("MAX_KEYWORDS", "5")),
        "default_amazon_domain": os.getenv("DEFAULT_AMAZON_DOMAIN", "amazon.com"),
        "host": os.getenv("HOST", "0.0.0.0"),
        "port": int(os.getenv("PORT", "5000")),
        "debug": os.getenv("DEBUG", "false").lower() == "true",
        "log_level": os.getenv("LOG_LEVEL", "INFO"),
    }
    
    return config


def sanitize_input(text: str, max_length: int = 10000) -> str:
    """Sanitize user input text."""
    if not text:
        return ""
    
    sanitized = text.replace('\x00', '').strip()
    
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length] + "..."
    
    return sanitized


def create_response(success: bool, data: Any = None, message: str = "", 
                   error: str = "", metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create a standardized response dictionary."""
    response = {
        "success": success,
        "timestamp": __import__("time").time()
    }
    
    if success:
        response["data"] = data
        if message:
            response["message"] = message
    else:
        response["error"] = error
    
    if metadata:
        response["metadata"] = metadata
    
    return response
