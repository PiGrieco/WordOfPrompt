#!/usr/bin/env python3
"""
WordOfPrompt MCP Server Runner

This script provides a convenient way to run WordOfPrompt MCP servers
with proper configuration and error handling.
"""

import sys
import os
import argparse
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp.servers.unified import UnifiedWordOfPromptServer
from mcp.servers.amazon_search import AmazonSearchServer
from core.exceptions import WordOfPromptError


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config_from_env() -> Dict[str, Any]:
    """Load configuration from environment variables."""
    config = {
        # API Keys
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "rainforest_api_key": os.getenv("RAINFOREST_API_KEY"),
        "amazon_affiliate_id": os.getenv("AMAZON_AFFILIATE_ID"),
        "huggingface_api_key": os.getenv("HUGGINGFACE_API_KEY"),
        
        # Model Configuration
        "huggingface_model_id": os.getenv("HUGGINGFACE_MODEL_ID", "PiGrieco/OpenSesame"),
        "intent_threshold": float(os.getenv("INTENT_THRESHOLD", "0.95")),
        "keyword_method": os.getenv("KEYWORD_METHOD", "rake"),
        "max_keywords": int(os.getenv("MAX_KEYWORDS", "5")),
        
        # Amazon Configuration
        "default_amazon_domain": os.getenv("DEFAULT_AMAZON_DOMAIN", "amazon.com"),
        
        # Server Configuration
        "log_level": os.getenv("LOG_LEVEL", "INFO"),
        "debug": os.getenv("DEBUG", "false").lower() == "true"
    }
    
    return config


def validate_config(config: Dict[str, Any], server_type: str) -> bool:
    """Validate configuration for the specified server type."""
    required_keys = {
        "unified": ["openai_api_key", "rainforest_api_key"],
        "amazon-search": ["rainforest_api_key"],
        "intent-analysis": ["openai_api_key"],
        "keyword-extraction": [],
        "product-recommendation": ["openai_api_key"]
    }
    
    required = required_keys.get(server_type, [])
    
    missing = []
    for key in required:
        if not config.get(key):
            missing.append(key)
    
    if missing:
        logger.error(f"Missing required configuration for {server_type}: {', '.join(missing)}")
        return False
    
    return True


async def run_unified_server(config: Dict[str, Any]):
    """Run the unified WordOfPrompt MCP server."""
    logger.info("Starting Unified WordOfPrompt MCP Server")
    
    server = UnifiedWordOfPromptServer()
    server.load_config(config)
    
    await server.initialize_server()
    await server.start_stdio()


async def run_amazon_search_server(config: Dict[str, Any]):
    """Run the Amazon search MCP server."""
    logger.info("Starting Amazon Search MCP Server")
    
    server = AmazonSearchServer()
    server.load_config(config)
    
    await server.initialize_server()
    await server.start_stdio()


async def run_server(server_type: str, config: Dict[str, Any]):
    """Run the specified MCP server."""
    try:
        if server_type == "unified":
            await run_unified_server(config)
        elif server_type == "amazon-search":
            await run_amazon_search_server(config)
        else:
            logger.error(f"Unknown server type: {server_type}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    except Exception as e:
        logger.error(f"Server failed: {e}")
        sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run WordOfPrompt MCP Servers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_mcp_server.py unified
  python run_mcp_server.py amazon-search
  python run_mcp_server.py --config-file config.json unified
  python run_mcp_server.py --debug unified
        """
    )
    
    parser.add_argument(
        "server_type",
        choices=["unified", "amazon-search", "intent-analysis", "keyword-extraction", "product-recommendation"],
        help="Type of MCP server to run"
    )
    
    parser.add_argument(
        "--config-file",
        type=str,
        help="Path to configuration file (JSON format)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate configuration, don't start server"
    )
    
    args = parser.parse_args()
    
    # Set log level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Load configuration
    config = load_config_from_env()
    
    if args.config_file:
        try:
            import json
            with open(args.config_file, 'r') as f:
                file_config = json.load(f)
            config.update(file_config)
            logger.info(f"Loaded configuration from {args.config_file}")
        except Exception as e:
            logger.error(f"Failed to load configuration file: {e}")
            sys.exit(1)
    
    # Validate configuration
    if not validate_config(config, args.server_type):
        logger.error("Configuration validation failed")
        sys.exit(1)
    
    if args.validate_only:
        logger.info("Configuration validation passed")
        return
    
    # Run the server
    logger.info(f"Starting {args.server_type} MCP server")
    asyncio.run(run_server(args.server_type, config))


if __name__ == "__main__":
    main()
