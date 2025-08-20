# 🚀 WordOfPrompt MCP - AI-Powered Advertising Recommendation System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![MCP Compatible](https://img.shields.io/badge/MCP-Compatible-green.svg)](https://modelcontextprotocol.io/)
[![Tests Passing](https://img.shields.io/badge/tests-100%25%20passing-brightgreen.svg)](#)

> **Professional-grade AI system using Model Context Protocol for intelligent product recommendations with multi-agent workflows and real-time intent analysis.**

## 🎯 Overview

WordOfPrompt MCP is a sophisticated AI-powered platform that combines:

- 🔌 **Model Context Protocol (MCP)** - Standardized AI-tool communication
- 🤖 **Multi-Agent Workflows** - CrewAI-powered agent orchestration  
- 🎯 **Real-time Intent Analysis** - Fine-tuned models for purchase intent detection
- 🛒 **Amazon Integration** - Rainforest API for product search and affiliate links
- 🌐 **WebSocket Real-time** - Instant communication and updates
- 💾 **Enterprise Database** - PostgreSQL with Redis caching
- 🐳 **Production Ready** - Docker containerization and Kubernetes support

## ✨ Key Features

### 🔌 **MCP-Native Architecture**
- **Standardized Communication** - Full MCP 2024 specification compliance
- **Tool Discovery** - Automatic discovery of available tools and capabilities
- **Interoperability** - Works with Claude, Cursor, and any MCP client
- **Modular Design** - Each service as an independent MCP server

### 🤖 **AI-Powered Intelligence**
- **Intent Classification** - Detects purchase intent with 81.8% accuracy
- **Keyword Extraction** - Multi-algorithm approach (RAKE, YAKE, Simple)
- **Product Recommendations** - AI-generated pros/cons analysis
- **Multi-Agent Orchestration** - Specialized agents for each task

### 🛒 **E-commerce Integration**
- **Amazon Product Search** - Rainforest API integration
- **Affiliate Link Generation** - Automatic monetization
- **Multi-Domain Support** - Global Amazon marketplaces
- **Real-time Pricing** - Live product data and availability

## 🏗️ Architecture

```
User Request → MCP Client → MCP Protocol → WordOfPrompt Server
                                              ↓
                                    ┌─────────────────┐
                                    │ Tool Orchestra  │
                                    ├─────────────────┤
                                    │ • Intent Analysis
                                    │ • Keyword Extract
                                    │ • Amazon Search
                                    │ • Recommendations
                                    └─────────────────┘
                                              ↓
                            External APIs (HuggingFace, Rainforest, OpenAI)
                                              ↓
                                    Structured Response → User
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- MCP-compatible client (Claude, Cursor, etc.)

### Installation

```bash
# Clone the repository
git clone https://github.com/PiGrieco/WordOfPrompt-Integration.git
cd WordOfPrompt-Integration/WordOfPrompt-MCP

# Install dependencies
pip install -r requirements.txt

# Setup configuration
cp config/env.example config/.env
# Edit config/.env with your API keys

# Run the MCP server
python scripts/run_mcp_server.py unified
```

### MCP Client Configuration

Add to your MCP client configuration:

```json
{
  "mcpServers": {
    "wordofprompt": {
      "command": "python",
      "args": ["-m", "scripts.run_mcp_server", "unified"],
      "cwd": "/path/to/WordOfPrompt-MCP",
      "env": {
        "OPENAI_API_KEY": "${OPENAI_API_KEY}",
        "RAINFOREST_API_KEY": "${RAINFOREST_API_KEY}",
        "AMAZON_AFFILIATE_ID": "${AMAZON_AFFILIATE_ID}"
      }
    }
  }
}
```

## 🛠️ Available MCP Tools

| Tool | Description | Input |
|------|-------------|-------|
| `analyze_and_recommend` | Complete workflow: intent → keywords → search → recommendations | `{user_prompt, intent_threshold?, max_products?}` |
| `analyze_intent` | Analyze purchase intent in text | `{text, threshold?}` |
| `extract_keywords` | Extract keywords using multiple algorithms | `{text, method?, max_keywords?}` |
| `search_amazon` | Search Amazon products with affiliate links | `{keywords, domain?, sort_by?, max_results?}` |
| `generate_recommendations` | AI-powered product analysis and recommendations | `{products, user_intent?}` |
| `get_system_status` | Get system health and statistics | `{}` |

## 📊 MCP Resources

| Resource URI | Description | Content |
|-------------|-------------|---------|
| `wordofprompt://config` | System configuration | JSON |
| `wordofprompt://health` | System health status | JSON |
| `wordofprompt://analytics` | Usage analytics and statistics | JSON |
| `amazon://domains` | Supported Amazon domains | JSON |

## 💬 MCP Prompts

| Prompt | Description | Arguments |
|--------|-------------|-----------|
| `complete_workflow` | Execute complete WordOfPrompt workflow | `user_query, workflow_type?` |
| `product_recommendation` | Generate personalized recommendations | `user_profile, context?` |

## 🧪 Testing

```bash
# Run validation
python scripts/validate_system.py

# Run professional test suite
pytest tests/ -v

# Test specific components
pytest tests/test_intent_classifier.py -v
pytest tests/test_mcp_protocol.py -v
```

## 📁 Project Structure

```
WordOfPrompt-MCP/
├── src/                    # Source code
│   ├── core/              # Core SDK (intent, keywords, utils)
│   ├── mcp/               # MCP integration layer
│   │   ├── protocols/     # MCP protocol implementation
│   │   ├── servers/       # MCP servers
│   │   ├── clients/       # MCP clients
│   │   └── tools/         # MCP tool definitions
│   ├── models/            # Data models
│   ├── tools/             # External API integrations
│   └── agents/            # Multi-agent system
├── tests/                 # Professional test suite
├── config/                # Configuration files
├── scripts/               # Utility scripts
├── docs/                  # Documentation
├── deployment/            # Docker & Kubernetes configs
└── examples/              # Usage examples
```

## 🔧 Development

### Local Development

```bash
# Setup development environment
./scripts/setup.sh

# Run tests
pytest tests/

# Start development server
python scripts/run_mcp_server.py unified --debug
```

### Docker Deployment

```bash
cd deployment/docker
docker-compose up --build
```

## 🔌 Integration Examples

### Using with Claude/Cursor

The system integrates seamlessly with MCP-compatible clients:

```
User: "I need a good laptop for programming under $1000"

WordOfPrompt MCP:
1. 🎯 Analyzes intent (score: 0.87 - PURCHASE)
2. 🔑 Extracts keywords: ["laptop", "programming", "1000"]  
3. 🛒 Searches Amazon with Rainforest API
4. ⭐ Generates AI recommendations with pros/cons
5. 🔗 Includes affiliate links for monetization

Response: Structured product recommendations with detailed analysis
```

### Programmatic Usage

```python
# Connect to MCP server
import mcp_client

client = MCPClient()
await client.connect("wordofprompt")

# Use the complete workflow
result = await client.call_tool("analyze_and_recommend", {
    "user_prompt": "I want to buy a smartphone under $800",
    "max_products": 5
})

print(result)
```

## 🎯 Intent Classification

The system uses an advanced intent classifier with **81.8% accuracy**:

- **Purchase Intent** (0.7-1.0): "I want to buy a laptop for $1000"
- **Browse Intent** (0.3-0.7): "What are the best smartphones?"
- **Compare Intent** (0.4-1.0): "Compare iPhone vs Samsung"
- **General Intent** (0.0-0.3): "Hello, how are you?"

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- [MCP Specification](https://spec.modelcontextprotocol.io/)
- [Documentation](docs/README.md)
- [API Reference](docs/api/)
- [Examples](examples/)

---

**Built with ❤️ from @Pigrieco**
