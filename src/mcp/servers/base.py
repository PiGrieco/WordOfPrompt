"""
Base MCP Server implementation for WordOfPrompt.

This module provides the base server class that all WordOfPrompt MCP servers
inherit from, providing common functionality and standardized interfaces.
"""

import asyncio
import logging
import sys
from typing import Any, Dict, List, Optional, TextIO
from abc import ABC, abstractmethod

import sys
import os
from pathlib import Path

# Add the src directory to Python path
src_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_dir))

from mcp.protocols.base import MCPProtocolHandler, MCPMessage


logger = logging.getLogger(__name__)


class MCPServer(MCPProtocolHandler):
    """
    Base MCP Server for WordOfPrompt services.
    
    This class provides the foundation for all WordOfPrompt MCP servers,
    handling protocol communication and providing common functionality.
    """
    
    def __init__(self, name: str, version: str = "1.0.0", description: str = ""):
        super().__init__(name, version)
        self.description = description
        self.running = False
        
        # I/O streams for stdio transport
        self.stdin: Optional[TextIO] = None
        self.stdout: Optional[TextIO] = None
        
        logger.info(f"Initialized MCP server: {name} v{version}")
    
    async def start_stdio(self, stdin: TextIO = None, stdout: TextIO = None):
        """
        Start the server with stdio transport.
        
        Args:
            stdin: Input stream (defaults to sys.stdin)
            stdout: Output stream (defaults to sys.stdout)
        """
        self.stdin = stdin or sys.stdin
        self.stdout = stdout or sys.stdout
        self.running = True
        
        logger.info(f"Starting MCP server {self.name} with stdio transport")
        
        try:
            while self.running:
                # Read message from stdin
                line = await self._read_line()
                if not line:
                    break
                
                try:
                    # Parse and handle message
                    message = MCPMessage.from_json(line)
                    response = await self.handle_message(message)
                    
                    # Send response if any
                    if response:
                        await self._write_message(response)
                        
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    # Send error response
                    error_response = self._create_error_response(
                        None,
                        -32603,  # Internal error
                        str(e)
                    )
                    await self._write_message(error_response)
        
        except KeyboardInterrupt:
            logger.info("Server interrupted by user")
        except Exception as e:
            logger.error(f"Server error: {e}")
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the server gracefully."""
        self.running = False
        logger.info(f"Stopped MCP server {self.name}")
    
    async def _read_line(self) -> Optional[str]:
        """Read a line from stdin asynchronously."""
        loop = asyncio.get_event_loop()
        
        def read_stdin():
            try:
                return self.stdin.readline().strip()
            except Exception:
                return None
        
        return await loop.run_in_executor(None, read_stdin)
    
    async def _write_message(self, message: MCPMessage):
        """Write a message to stdout."""
        try:
            json_str = message.to_json()
            self.stdout.write(json_str + "\n")
            self.stdout.flush()
        except Exception as e:
            logger.error(f"Error writing message: {e}")
    
    @abstractmethod
    async def setup_tools(self):
        """Setup MCP tools for this server. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    async def setup_resources(self):
        """Setup MCP resources for this server. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    async def setup_prompts(self):
        """Setup MCP prompts for this server. Must be implemented by subclasses."""
        pass
    
    async def initialize_server(self):
        """Initialize the server by setting up tools, resources, and prompts."""
        logger.info(f"Initializing server {self.name}")
        
        await self.setup_tools()
        await self.setup_resources()
        await self.setup_prompts()
        
        logger.info(f"Server {self.name} initialized with {len(self.tools)} tools, "
                   f"{len(self.resources)} resources, {len(self.prompts)} prompts")
    
    def get_server_info(self) -> Dict[str, Any]:
        """Get server information."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "tools_count": len(self.tools),
            "resources_count": len(self.resources),
            "prompts_count": len(self.prompts),
            "initialized": self.initialized,
            "running": self.running
        }


class WordOfPromptMCPServer(MCPServer):
    """
    Base class for WordOfPrompt-specific MCP servers.
    
    This class provides WordOfPrompt-specific functionality and
    common patterns used across all WordOfPrompt MCP servers.
    """
    
    def __init__(self, name: str, version: str = "1.0.0", description: str = ""):
        super().__init__(name, version, description)
        self.config: Dict[str, Any] = {}
        self.stats: Dict[str, Any] = {
            "requests_handled": 0,
            "errors_count": 0,
            "tools_called": {},
            "resources_accessed": {},
            "prompts_used": {}
        }
    
    def load_config(self, config: Dict[str, Any]):
        """Load configuration for the server."""
        self.config = config
        logger.info(f"Loaded configuration for {self.name}")
    
    async def handle_message(self, message: MCPMessage) -> Optional[MCPMessage]:
        """Enhanced message handling with statistics."""
        self.stats["requests_handled"] += 1
        
        try:
            response = await super().handle_message(message)
            
            # Track tool usage
            if message.method == "tools/call" and message.params:
                tool_name = message.params.get("name")
                if tool_name:
                    self.stats["tools_called"][tool_name] = self.stats["tools_called"].get(tool_name, 0) + 1
            
            # Track resource usage
            elif message.method == "resources/read" and message.params:
                uri = message.params.get("uri")
                if uri:
                    self.stats["resources_accessed"][uri] = self.stats["resources_accessed"].get(uri, 0) + 1
            
            # Track prompt usage
            elif message.method == "prompts/get" and message.params:
                prompt_name = message.params.get("name")
                if prompt_name:
                    self.stats["prompts_used"][prompt_name] = self.stats["prompts_used"].get(prompt_name, 0) + 1
            
            return response
            
        except Exception as e:
            self.stats["errors_count"] += 1
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        return {
            **self.stats,
            "server_info": self.get_server_info()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the server."""
        return {
            "status": "healthy" if self.running else "stopped",
            "initialized": self.initialized,
            "uptime": self.stats["requests_handled"],
            "error_rate": self.stats["errors_count"] / max(self.stats["requests_handled"], 1)
        }
