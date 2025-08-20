"""
Base MCP protocol implementation following the Model Context Protocol specification.

This module provides the core protocol classes and message handling for
standardized communication between AI models and external tools.
"""

import json
import uuid
import asyncio
import logging
from typing import Any, Dict, List, Optional, Union, Callable, Awaitable
from dataclasses import dataclass, asdict
from enum import Enum
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class MCPMessageType(Enum):
    """MCP message types according to the specification."""
    
    # Core protocol messages
    INITIALIZE = "initialize"
    INITIALIZED = "initialized"
    PING = "ping"
    PONG = "pong"
    
    # Tool-related messages
    LIST_TOOLS = "tools/list"
    CALL_TOOL = "tools/call"
    
    # Resource-related messages
    LIST_RESOURCES = "resources/list"
    READ_RESOURCE = "resources/read"
    
    # Prompt-related messages
    LIST_PROMPTS = "prompts/list"
    GET_PROMPT = "prompts/get"
    
    # Notification messages
    NOTIFICATION = "notification"
    
    # Error handling
    ERROR = "error"


@dataclass
class MCPMessage:
    """Base MCP message structure."""
    
    jsonrpc: str = "2.0"
    id: Optional[str] = None
    method: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.id is None and self.method is not None:
            self.id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary format."""
        data = asdict(self)
        return {k: v for k, v in data.items() if v is not None}
    
    def to_json(self) -> str:
        """Convert message to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCPMessage":
        """Create message from dictionary."""
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> "MCPMessage":
        """Create message from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)


@dataclass
class MCPRequest(MCPMessage):
    """MCP request message."""
    
    def __post_init__(self):
        super().__post_init__()
        if self.method is None:
            raise ValueError("Request must have a method")


@dataclass
class MCPResponse(MCPMessage):
    """MCP response message."""
    
    def __post_init__(self):
        super().__post_init__()
        if self.id is None:
            raise ValueError("Response must have an id")


@dataclass
class MCPError:
    """MCP error structure."""
    
    code: int
    message: str
    data: Optional[Dict[str, Any]] = None
    
    # Standard error codes
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary format."""
        result = {"code": self.code, "message": self.message}
        if self.data:
            result["data"] = self.data
        return result


@dataclass
class MCPTool:
    """MCP tool definition."""
    
    name: str
    description: str
    input_schema: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tool to dictionary format."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema
        }


@dataclass
class MCPResource:
    """MCP resource definition."""
    
    uri: str
    name: str
    description: Optional[str] = None
    mime_type: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert resource to dictionary format."""
        result = {
            "uri": self.uri,
            "name": self.name
        }
        if self.description:
            result["description"] = self.description
        if self.mime_type:
            result["mimeType"] = self.mime_type
        return result


@dataclass
class MCPPrompt:
    """MCP prompt template definition."""
    
    name: str
    description: str
    arguments: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert prompt to dictionary format."""
        return {
            "name": self.name,
            "description": self.description,
            "arguments": self.arguments
        }


class MCPProtocolHandler(ABC):
    """
    Abstract base class for MCP protocol handlers.
    
    This class defines the interface for handling MCP messages and
    implementing the protocol specification.
    """
    
    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.tools: Dict[str, Callable] = {}
        self.resources: Dict[str, Callable] = {}
        self.prompts: Dict[str, Callable] = {}
        self.initialized = False
        
        # Message handlers
        self.handlers: Dict[str, Callable] = {
            MCPMessageType.INITIALIZE.value: self._handle_initialize,
            MCPMessageType.PING.value: self._handle_ping,
            MCPMessageType.LIST_TOOLS.value: self._handle_list_tools,
            MCPMessageType.CALL_TOOL.value: self._handle_call_tool,
            MCPMessageType.LIST_RESOURCES.value: self._handle_list_resources,
            MCPMessageType.READ_RESOURCE.value: self._handle_read_resource,
            MCPMessageType.LIST_PROMPTS.value: self._handle_list_prompts,
            MCPMessageType.GET_PROMPT.value: self._handle_get_prompt,
        }
    
    async def handle_message(self, message: MCPMessage) -> Optional[MCPMessage]:
        """
        Handle incoming MCP message.
        
        Args:
            message: The incoming MCP message
            
        Returns:
            Optional response message
        """
        try:
            if message.method in self.handlers:
                handler = self.handlers[message.method]
                return await handler(message)
            else:
                return self._create_error_response(
                    message.id,
                    MCPError.METHOD_NOT_FOUND,
                    f"Method not found: {message.method}"
                )
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            return self._create_error_response(
                message.id,
                MCPError.INTERNAL_ERROR,
                str(e)
            )
    
    def register_tool(self, name: str, description: str, input_schema: Dict[str, Any], handler: Callable):
        """Register a tool with the MCP handler."""
        self.tools[name] = {
            "description": description,
            "input_schema": input_schema,
            "handler": handler
        }
        logger.info(f"Registered MCP tool: {name}")
    
    def register_resource(self, uri: str, name: str, handler: Callable, description: str = None, mime_type: str = None):
        """Register a resource with the MCP handler."""
        self.resources[uri] = {
            "name": name,
            "description": description,
            "mime_type": mime_type,
            "handler": handler
        }
        logger.info(f"Registered MCP resource: {uri}")
    
    def register_prompt(self, name: str, description: str, arguments: List[Dict[str, Any]], handler: Callable):
        """Register a prompt template with the MCP handler."""
        self.prompts[name] = {
            "description": description,
            "arguments": arguments,
            "handler": handler
        }
        logger.info(f"Registered MCP prompt: {name}")
    
    async def _handle_initialize(self, message: MCPMessage) -> MCPMessage:
        """Handle initialization request."""
        self.initialized = True
        return MCPResponse(
            id=message.id,
            result={
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {"listChanged": True},
                    "resources": {"subscribe": True, "listChanged": True},
                    "prompts": {"listChanged": True}
                },
                "serverInfo": {
                    "name": self.name,
                    "version": self.version
                }
            }
        )
    
    async def _handle_ping(self, message: MCPMessage) -> MCPMessage:
        """Handle ping request."""
        return MCPResponse(
            id=message.id,
            result={}
        )
    
    async def _handle_list_tools(self, message: MCPMessage) -> MCPMessage:
        """Handle list tools request."""
        tools = []
        for name, tool_info in self.tools.items():
            tools.append({
                "name": name,
                "description": tool_info["description"],
                "inputSchema": tool_info["input_schema"]
            })
        
        return MCPResponse(
            id=message.id,
            result={"tools": tools}
        )
    
    async def _handle_call_tool(self, message: MCPMessage) -> MCPMessage:
        """Handle tool call request."""
        if not message.params:
            return self._create_error_response(
                message.id,
                MCPError.INVALID_PARAMS,
                "Missing parameters"
            )
        
        tool_name = message.params.get("name")
        arguments = message.params.get("arguments", {})
        
        if tool_name not in self.tools:
            return self._create_error_response(
                message.id,
                MCPError.METHOD_NOT_FOUND,
                f"Tool not found: {tool_name}"
            )
        
        try:
            tool_handler = self.tools[tool_name]["handler"]
            
            # Call the tool handler
            if asyncio.iscoroutinefunction(tool_handler):
                result = await tool_handler(**arguments)
            else:
                result = tool_handler(**arguments)
            
            return MCPResponse(
                id=message.id,
                result={
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(result, indent=2)
                        }
                    ]
                }
            )
        except Exception as e:
            logger.error(f"Error calling tool {tool_name}: {e}")
            return self._create_error_response(
                message.id,
                MCPError.INTERNAL_ERROR,
                str(e)
            )
    
    async def _handle_list_resources(self, message: MCPMessage) -> MCPMessage:
        """Handle list resources request."""
        resources = []
        for uri, resource_info in self.resources.items():
            resource_data = {
                "uri": uri,
                "name": resource_info["name"]
            }
            if resource_info.get("description"):
                resource_data["description"] = resource_info["description"]
            if resource_info.get("mime_type"):
                resource_data["mimeType"] = resource_info["mime_type"]
            
            resources.append(resource_data)
        
        return MCPResponse(
            id=message.id,
            result={"resources": resources}
        )
    
    async def _handle_read_resource(self, message: MCPMessage) -> MCPMessage:
        """Handle read resource request."""
        if not message.params:
            return self._create_error_response(
                message.id,
                MCPError.INVALID_PARAMS,
                "Missing parameters"
            )
        
        uri = message.params.get("uri")
        
        if uri not in self.resources:
            return self._create_error_response(
                message.id,
                MCPError.METHOD_NOT_FOUND,
                f"Resource not found: {uri}"
            )
        
        try:
            resource_handler = self.resources[uri]["handler"]
            
            # Call the resource handler
            if asyncio.iscoroutinefunction(resource_handler):
                content = await resource_handler(uri)
            else:
                content = resource_handler(uri)
            
            return MCPResponse(
                id=message.id,
                result={
                    "contents": [
                        {
                            "uri": uri,
                            "mimeType": self.resources[uri].get("mime_type", "text/plain"),
                            "text": content
                        }
                    ]
                }
            )
        except Exception as e:
            logger.error(f"Error reading resource {uri}: {e}")
            return self._create_error_response(
                message.id,
                MCPError.INTERNAL_ERROR,
                str(e)
            )
    
    async def _handle_list_prompts(self, message: MCPMessage) -> MCPMessage:
        """Handle list prompts request."""
        prompts = []
        for name, prompt_info in self.prompts.items():
            prompts.append({
                "name": name,
                "description": prompt_info["description"],
                "arguments": prompt_info["arguments"]
            })
        
        return MCPResponse(
            id=message.id,
            result={"prompts": prompts}
        )
    
    async def _handle_get_prompt(self, message: MCPMessage) -> MCPMessage:
        """Handle get prompt request."""
        if not message.params:
            return self._create_error_response(
                message.id,
                MCPError.INVALID_PARAMS,
                "Missing parameters"
            )
        
        prompt_name = message.params.get("name")
        arguments = message.params.get("arguments", {})
        
        if prompt_name not in self.prompts:
            return self._create_error_response(
                message.id,
                MCPError.METHOD_NOT_FOUND,
                f"Prompt not found: {prompt_name}"
            )
        
        try:
            prompt_handler = self.prompts[prompt_name]["handler"]
            
            # Call the prompt handler
            if asyncio.iscoroutinefunction(prompt_handler):
                messages = await prompt_handler(**arguments)
            else:
                messages = prompt_handler(**arguments)
            
            return MCPResponse(
                id=message.id,
                result={
                    "description": self.prompts[prompt_name]["description"],
                    "messages": messages
                }
            )
        except Exception as e:
            logger.error(f"Error getting prompt {prompt_name}: {e}")
            return self._create_error_response(
                message.id,
                MCPError.INTERNAL_ERROR,
                str(e)
            )
    
    def _create_error_response(self, request_id: Optional[str], code: int, message: str, data: Dict[str, Any] = None) -> MCPResponse:
        """Create an error response message."""
        error = MCPError(code=code, message=message, data=data)
        return MCPResponse(
            id=request_id,
            error=error.to_dict()
        )
