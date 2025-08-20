"""
MCP Tool and Server Registry.

This module provides registry functionality for managing MCP tools,
servers, and their capabilities in a centralized way.
"""

import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class RegistryType(Enum):
    """Types of registry entries."""
    TOOL = "tool"
    RESOURCE = "resource"
    PROMPT = "prompt"
    SERVER = "server"


@dataclass
class RegistryEntry:
    """Base registry entry."""
    name: str
    type: RegistryType
    description: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ToolEntry(RegistryEntry):
    """Tool registry entry."""
    input_schema: Dict[str, Any]
    handler: Callable
    
    def __post_init__(self):
        self.type = RegistryType.TOOL


@dataclass
class ResourceEntry(RegistryEntry):
    """Resource registry entry."""
    uri: str
    mime_type: str
    handler: Callable
    
    def __post_init__(self):
        self.type = RegistryType.RESOURCE


@dataclass
class PromptEntry(RegistryEntry):
    """Prompt registry entry."""
    arguments: List[Dict[str, Any]]
    handler: Callable
    
    def __post_init__(self):
        self.type = RegistryType.PROMPT


@dataclass
class ServerEntry(RegistryEntry):
    """Server registry entry."""
    version: str
    capabilities: Dict[str, Any]
    endpoint: str
    
    def __post_init__(self):
        self.type = RegistryType.SERVER


class MCPRegistry:
    """
    Base registry for MCP components.
    
    This class provides a centralized way to register and discover
    MCP tools, resources, prompts, and servers.
    """
    
    def __init__(self, name: str = "default"):
        self.name = name
        self.entries: Dict[str, RegistryEntry] = {}
        self.stats = {
            "tools": 0,
            "resources": 0,
            "prompts": 0,
            "servers": 0,
            "total_registrations": 0
        }
        
        logger.info(f"Initialized MCP registry: {name}")
    
    def register(self, entry: RegistryEntry) -> bool:
        """Register an entry in the registry."""
        try:
            if entry.name in self.entries:
                logger.warning(f"Overwriting existing entry: {entry.name}")
            
            self.entries[entry.name] = entry
            self.stats[entry.type.value + "s"] += 1
            self.stats["total_registrations"] += 1
            
            logger.info(f"Registered {entry.type.value}: {entry.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register {entry.name}: {e}")
            return False
    
    def unregister(self, name: str) -> bool:
        """Unregister an entry from the registry."""
        if name not in self.entries:
            logger.warning(f"Entry not found for unregistration: {name}")
            return False
        
        try:
            entry = self.entries.pop(name)
            self.stats[entry.type.value + "s"] -= 1
            
            logger.info(f"Unregistered {entry.type.value}: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unregister {name}: {e}")
            return False
    
    def get(self, name: str) -> Optional[RegistryEntry]:
        """Get an entry by name."""
        return self.entries.get(name)
    
    def list_by_type(self, entry_type: RegistryType) -> List[RegistryEntry]:
        """List all entries of a specific type."""
        return [entry for entry in self.entries.values() if entry.type == entry_type]
    
    def list_all(self) -> List[RegistryEntry]:
        """List all entries."""
        return list(self.entries.values())
    
    def search(self, query: str) -> List[RegistryEntry]:
        """Search entries by name or description."""
        results = []
        query_lower = query.lower()
        
        for entry in self.entries.values():
            if (query_lower in entry.name.lower() or 
                query_lower in entry.description.lower()):
                results.append(entry)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        return {
            "name": self.name,
            "stats": self.stats.copy(),
            "entries_count": len(self.entries)
        }
    
    def clear(self):
        """Clear all entries."""
        self.entries.clear()
        self.stats = {
            "tools": 0,
            "resources": 0,
            "prompts": 0,
            "servers": 0,
            "total_registrations": 0
        }
        logger.info(f"Cleared registry: {self.name}")


class MCPToolRegistry(MCPRegistry):
    """Specialized registry for MCP tools."""
    
    def __init__(self):
        super().__init__("tools")
    
    def register_tool(self, name: str, description: str, input_schema: Dict[str, Any], 
                     handler: Callable, metadata: Dict[str, Any] = None) -> bool:
        """Register a tool."""
        entry = ToolEntry(
            name=name,
            description=description,
            input_schema=input_schema,
            handler=handler,
            metadata=metadata or {}
        )
        return self.register(entry)
    
    def get_tool(self, name: str) -> Optional[ToolEntry]:
        """Get a tool by name."""
        entry = self.get(name)
        return entry if isinstance(entry, ToolEntry) else None
    
    def list_tools(self) -> List[ToolEntry]:
        """List all tools."""
        return [entry for entry in self.entries.values() if isinstance(entry, ToolEntry)]
    
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get schemas for all tools."""
        schemas = []
        for tool in self.list_tools():
            schemas.append({
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.input_schema
            })
        return schemas


class MCPResourceRegistry(MCPRegistry):
    """Specialized registry for MCP resources."""
    
    def __init__(self):
        super().__init__("resources")
    
    def register_resource(self, name: str, uri: str, description: str, 
                         mime_type: str, handler: Callable, 
                         metadata: Dict[str, Any] = None) -> bool:
        """Register a resource."""
        entry = ResourceEntry(
            name=name,
            uri=uri,
            description=description,
            mime_type=mime_type,
            handler=handler,
            metadata=metadata or {}
        )
        return self.register(entry)
    
    def get_resource(self, name: str) -> Optional[ResourceEntry]:
        """Get a resource by name."""
        entry = self.get(name)
        return entry if isinstance(entry, ResourceEntry) else None
    
    def get_resource_by_uri(self, uri: str) -> Optional[ResourceEntry]:
        """Get a resource by URI."""
        for entry in self.entries.values():
            if isinstance(entry, ResourceEntry) and entry.uri == uri:
                return entry
        return None
    
    def list_resources(self) -> List[ResourceEntry]:
        """List all resources."""
        return [entry for entry in self.entries.values() if isinstance(entry, ResourceEntry)]
    
    def get_resource_list(self) -> List[Dict[str, Any]]:
        """Get list of all resources."""
        resources = []
        for resource in self.list_resources():
            resources.append({
                "uri": resource.uri,
                "name": resource.name,
                "description": resource.description,
                "mimeType": resource.mime_type
            })
        return resources


class MCPPromptRegistry(MCPRegistry):
    """Specialized registry for MCP prompts."""
    
    def __init__(self):
        super().__init__("prompts")
    
    def register_prompt(self, name: str, description: str, arguments: List[Dict[str, Any]],
                       handler: Callable, metadata: Dict[str, Any] = None) -> bool:
        """Register a prompt."""
        entry = PromptEntry(
            name=name,
            description=description,
            arguments=arguments,
            handler=handler,
            metadata=metadata or {}
        )
        return self.register(entry)
    
    def get_prompt(self, name: str) -> Optional[PromptEntry]:
        """Get a prompt by name."""
        entry = self.get(name)
        return entry if isinstance(entry, PromptEntry) else None
    
    def list_prompts(self) -> List[PromptEntry]:
        """List all prompts."""
        return [entry for entry in self.entries.values() if isinstance(entry, PromptEntry)]
    
    def get_prompt_list(self) -> List[Dict[str, Any]]:
        """Get list of all prompts."""
        prompts = []
        for prompt in self.list_prompts():
            prompts.append({
                "name": prompt.name,
                "description": prompt.description,
                "arguments": prompt.arguments
            })
        return prompts


class MCPServerRegistry(MCPRegistry):
    """Specialized registry for MCP servers."""
    
    def __init__(self):
        super().__init__("servers")
    
    def register_server(self, name: str, description: str, version: str,
                       capabilities: Dict[str, Any], endpoint: str,
                       metadata: Dict[str, Any] = None) -> bool:
        """Register a server."""
        entry = ServerEntry(
            name=name,
            description=description,
            version=version,
            capabilities=capabilities,
            endpoint=endpoint,
            metadata=metadata or {}
        )
        return self.register(entry)
    
    def get_server(self, name: str) -> Optional[ServerEntry]:
        """Get a server by name."""
        entry = self.get(name)
        return entry if isinstance(entry, ServerEntry) else None
    
    def list_servers(self) -> List[ServerEntry]:
        """List all servers."""
        return [entry for entry in self.entries.values() if isinstance(entry, ServerEntry)]
    
    def get_server_list(self) -> List[Dict[str, Any]]:
        """Get list of all servers."""
        servers = []
        for server in self.list_servers():
            servers.append({
                "name": server.name,
                "description": server.description,
                "version": server.version,
                "capabilities": server.capabilities,
                "endpoint": server.endpoint
            })
        return servers


# Global registries
_global_tool_registry = MCPToolRegistry()
_global_resource_registry = MCPResourceRegistry()
_global_prompt_registry = MCPPromptRegistry()
_global_server_registry = MCPServerRegistry()


def get_tool_registry() -> MCPToolRegistry:
    """Get the global tool registry."""
    return _global_tool_registry


def get_resource_registry() -> MCPResourceRegistry:
    """Get the global resource registry."""
    return _global_resource_registry


def get_prompt_registry() -> MCPPromptRegistry:
    """Get the global prompt registry."""
    return _global_prompt_registry


def get_server_registry() -> MCPServerRegistry:
    """Get the global server registry."""
    return _global_server_registry
