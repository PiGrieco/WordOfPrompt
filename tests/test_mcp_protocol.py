"""
Test suite for MCP Protocol implementation.

Professional test suite for the Model Context Protocol integration.
"""

import sys
from pathlib import Path
import pytest
import asyncio

# Add src to path
src_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_dir))

# Import using absolute imports to avoid issues
import importlib.util

def load_mcp_module(module_name, file_path):
    """Load MCP module using importlib."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load MCP modules
mcp_base = load_mcp_module("mcp_base", src_dir / "mcp" / "protocols" / "base.py")
mcp_transport = load_mcp_module("mcp_transport", src_dir / "mcp" / "protocols" / "transport.py")

MCPMessage = mcp_base.MCPMessage
MCPRequest = mcp_base.MCPRequest
MCPResponse = mcp_base.MCPResponse
MCPError = mcp_base.MCPError
MockTransport = mcp_transport.MockTransport


class TestMCPProtocol:
    """Test cases for MCP Protocol."""
    
    def test_message_creation(self):
        """Test MCP message creation."""
        request = MCPRequest(
            method="tools/list",
            params={}
        )
        
        assert request.jsonrpc == "2.0"
        assert request.method == "tools/list"
        assert request.id is not None
    
    def test_message_serialization(self):
        """Test message serialization/deserialization."""
        original = MCPRequest(
            method="test/method",
            params={"key": "value"}
        )
        
        # Serialize
        json_str = original.to_json()
        
        # Deserialize
        parsed = MCPMessage.from_json(json_str)
        
        assert parsed.method == original.method
        assert parsed.params == original.params
    
    def test_response_creation(self):
        """Test response creation."""
        response = MCPResponse(
            id="test-id",
            result={"success": True}
        )
        
        assert response.id == "test-id"
        assert response.result["success"] is True
    
    def test_error_creation(self):
        """Test error creation."""
        error = MCPError(
            code=-32602,
            message="Invalid params"
        )
        
        assert error.code == -32602
        assert error.message == "Invalid params"
    
    @pytest.mark.asyncio
    async def test_mock_transport(self):
        """Test mock transport functionality."""
        transport = MockTransport()
        
        # Test connection
        connected = await transport.connect()
        assert connected is True
        
        # Test message sending
        message = MCPMessage(method="test", params={"key": "value"})
        await transport.send_message(message)
        
        assert len(transport.sent_messages) == 1
        assert transport.sent_messages[0].method == "test"
        
        # Test message receiving
        response = MCPMessage(id="test-id", result={"success": True})
        transport.add_message_to_queue(response)
        
        received = await transport.receive_message()
        assert received is not None
        assert received.id == "test-id"
        
        # Test disconnection
        await transport.disconnect()
        assert transport.connected is False
