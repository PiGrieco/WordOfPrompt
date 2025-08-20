"""
MCP Transport implementations for different communication methods.

This module provides transport layer implementations for the Model Context Protocol,
supporting stdio, websocket, and other communication methods.
"""

import asyncio
import json
import logging
import sys
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, TextIO, Callable, Awaitable

from .base import MCPMessage

logger = logging.getLogger(__name__)


class MCPTransport(ABC):
    """Abstract base class for MCP transport implementations."""
    
    def __init__(self):
        self.connected = False
        self.message_handler: Optional[Callable[[MCPMessage], Awaitable[Optional[MCPMessage]]]] = None
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the transport. Returns True if successful."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the transport."""
        pass
    
    @abstractmethod
    async def send_message(self, message: MCPMessage) -> None:
        """Send a message through the transport."""
        pass
    
    @abstractmethod
    async def receive_message(self) -> Optional[MCPMessage]:
        """Receive a message from the transport."""
        pass
    
    def set_message_handler(self, handler: Callable[[MCPMessage], Awaitable[Optional[MCPMessage]]]):
        """Set the message handler for incoming messages."""
        self.message_handler = handler


class StdioTransport(MCPTransport):
    """
    Standard input/output transport for MCP.
    
    This transport uses stdin/stdout for communication, which is the
    standard method for MCP server implementations.
    """
    
    def __init__(self, stdin: Optional[TextIO] = None, stdout: Optional[TextIO] = None):
        super().__init__()
        self.stdin = stdin or sys.stdin
        self.stdout = stdout or sys.stdout
        self.running = False
    
    async def connect(self) -> bool:
        """Connect to stdio transport."""
        try:
            self.connected = True
            self.running = True
            logger.info("Connected to stdio transport")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to stdio: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from stdio transport."""
        self.connected = False
        self.running = False
        logger.info("Disconnected from stdio transport")
    
    async def send_message(self, message: MCPMessage) -> None:
        """Send a message to stdout."""
        if not self.connected:
            raise RuntimeError("Transport not connected")
        
        try:
            json_str = message.to_json()
            self.stdout.write(json_str + "\n")
            self.stdout.flush()
            logger.debug(f"Sent message: {message.method or 'response'}")
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            raise
    
    async def receive_message(self) -> Optional[MCPMessage]:
        """Receive a message from stdin."""
        if not self.connected:
            raise RuntimeError("Transport not connected")
        
        try:
            # Read line asynchronously
            loop = asyncio.get_event_loop()
            line = await loop.run_in_executor(None, self._read_line)
            
            if not line:
                return None
            
            # Parse message
            message = MCPMessage.from_json(line)
            logger.debug(f"Received message: {message.method or 'response'}")
            return message
            
        except Exception as e:
            logger.error(f"Failed to receive message: {e}")
            return None
    
    def _read_line(self) -> Optional[str]:
        """Read a line from stdin synchronously."""
        try:
            line = self.stdin.readline()
            return line.strip() if line else None
        except Exception:
            return None
    
    async def start_message_loop(self):
        """Start the message processing loop."""
        if not self.message_handler:
            raise ValueError("Message handler not set")
        
        logger.info("Starting stdio message loop")
        
        try:
            while self.running:
                # Receive message
                message = await self.receive_message()
                if not message:
                    break
                
                # Handle message
                try:
                    response = await self.message_handler(message)
                    if response:
                        await self.send_message(response)
                except Exception as e:
                    logger.error(f"Error handling message: {e}")
                    # Send error response
                    error_response = MCPMessage(
                        id=message.id,
                        error={
                            "code": -32603,
                            "message": str(e)
                        }
                    )
                    await self.send_message(error_response)
        
        except KeyboardInterrupt:
            logger.info("Message loop interrupted")
        except Exception as e:
            logger.error(f"Message loop error: {e}")
        finally:
            await self.disconnect()


class WebSocketTransport(MCPTransport):
    """
    WebSocket transport for MCP.
    
    This transport uses WebSocket connections for communication,
    useful for web-based integrations.
    """
    
    def __init__(self, url: str, headers: Optional[Dict[str, str]] = None):
        super().__init__()
        self.url = url
        self.headers = headers or {}
        self.websocket = None
    
    async def connect(self) -> bool:
        """Connect to WebSocket."""
        try:
            import websockets
            
            self.websocket = await websockets.connect(
                self.url,
                extra_headers=self.headers
            )
            self.connected = True
            logger.info(f"Connected to WebSocket: {self.url}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to WebSocket: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from WebSocket."""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
        
        self.connected = False
        logger.info("Disconnected from WebSocket")
    
    async def send_message(self, message: MCPMessage) -> None:
        """Send a message through WebSocket."""
        if not self.connected or not self.websocket:
            raise RuntimeError("WebSocket not connected")
        
        try:
            json_str = message.to_json()
            await self.websocket.send(json_str)
            logger.debug(f"Sent WebSocket message: {message.method or 'response'}")
        except Exception as e:
            logger.error(f"Failed to send WebSocket message: {e}")
            raise
    
    async def receive_message(self) -> Optional[MCPMessage]:
        """Receive a message from WebSocket."""
        if not self.connected or not self.websocket:
            raise RuntimeError("WebSocket not connected")
        
        try:
            message_str = await self.websocket.recv()
            message = MCPMessage.from_json(message_str)
            logger.debug(f"Received WebSocket message: {message.method or 'response'}")
            return message
            
        except Exception as e:
            logger.error(f"Failed to receive WebSocket message: {e}")
            return None
    
    async def start_message_loop(self):
        """Start the WebSocket message processing loop."""
        if not self.message_handler:
            raise ValueError("Message handler not set")
        
        logger.info("Starting WebSocket message loop")
        
        try:
            while self.connected and self.websocket:
                # Receive message
                message = await self.receive_message()
                if not message:
                    break
                
                # Handle message
                try:
                    response = await self.message_handler(message)
                    if response:
                        await self.send_message(response)
                except Exception as e:
                    logger.error(f"Error handling WebSocket message: {e}")
                    # Send error response
                    error_response = MCPMessage(
                        id=message.id,
                        error={
                            "code": -32603,
                            "message": str(e)
                        }
                    )
                    await self.send_message(error_response)
        
        except Exception as e:
            logger.error(f"WebSocket message loop error: {e}")
        finally:
            await self.disconnect()


class MockTransport(MCPTransport):
    """
    Mock transport for testing purposes.
    
    This transport simulates message passing without actual I/O,
    useful for unit testing MCP components.
    """
    
    def __init__(self):
        super().__init__()
        self.sent_messages = []
        self.received_messages = []
        self.message_queue = []
    
    async def connect(self) -> bool:
        """Connect to mock transport."""
        self.connected = True
        logger.debug("Connected to mock transport")
        return True
    
    async def disconnect(self) -> None:
        """Disconnect from mock transport."""
        self.connected = False
        logger.debug("Disconnected from mock transport")
    
    async def send_message(self, message: MCPMessage) -> None:
        """Send a message (store in sent_messages)."""
        if not self.connected:
            raise RuntimeError("Mock transport not connected")
        
        self.sent_messages.append(message)
        logger.debug(f"Mock sent message: {message.method or 'response'}")
    
    async def receive_message(self) -> Optional[MCPMessage]:
        """Receive a message (from message_queue)."""
        if not self.connected:
            raise RuntimeError("Mock transport not connected")
        
        if self.message_queue:
            message = self.message_queue.pop(0)
            self.received_messages.append(message)
            logger.debug(f"Mock received message: {message.method or 'response'}")
            return message
        
        return None
    
    def add_message_to_queue(self, message: MCPMessage):
        """Add a message to the receive queue."""
        self.message_queue.append(message)
    
    def clear_messages(self):
        """Clear all message logs."""
        self.sent_messages.clear()
        self.received_messages.clear()
        self.message_queue.clear()
