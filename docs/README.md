# WordOfPrompt Documentation

Welcome to the comprehensive documentation for WordOfPrompt, the AI-powered advertising recommendation system.

## 📚 Documentation Structure

### For Users
- [**User Guide**](user/README.md) - Complete guide for end users
- [**Quick Start**](user/quickstart.md) - Get started in 5 minutes
- [**API Reference**](api/README.md) - Complete API documentation
- [**Examples**](../examples/) - Code examples and use cases

### For Developers
- [**Developer Guide**](dev/README.md) - Complete development documentation
- [**Architecture Overview**](dev/architecture.md) - System architecture and design
- [**Contributing**](dev/contributing.md) - How to contribute to the project
- [**Testing Guide**](dev/testing.md) - Testing strategies and guidelines

### For DevOps
- [**Deployment Guide**](../deployment/README.md) - Production deployment
- [**Configuration**](dev/configuration.md) - Environment configuration
- [**Monitoring**](dev/monitoring.md) - System monitoring and observability

## 🎯 What is WordOfPrompt?

WordOfPrompt is a sophisticated AI-powered platform that combines:

- **Multi-Agent Workflows** using CrewAI for intelligent task orchestration
- **Real-time Intent Analysis** with fine-tuned language models
- **Amazon Product Integration** via Rainforest API
- **Affiliate Link Generation** for monetization
- **WebSocket Communication** for real-time interactions
- **Enterprise-grade Infrastructure** with PostgreSQL and Redis

## 🚀 Key Features

### For Business Users
- **Intelligent Product Recommendations** - AI-powered product suggestions
- **Real-time Chat Interface** - Instant communication with the system
- **Multi-domain Support** - Works across different Amazon marketplaces
- **Analytics Dashboard** - Comprehensive usage and performance metrics

### For Developers
- **Modular Architecture** - Easy to extend and customize
- **Comprehensive APIs** - RESTful and WebSocket APIs
- **Docker Support** - Containerized deployment
- **Extensive Testing** - Unit, integration, and load tests

### For Enterprise
- **Scalable Infrastructure** - Handles high-volume requests
- **Security First** - Rate limiting, authentication, and data protection
- **Monitoring & Observability** - Prometheus metrics and logging
- **High Availability** - Production-ready deployment options

## 🛠️ Technology Stack

### Core Technologies
- **Python 3.9+** - Primary programming language
- **FastAPI** - High-performance web framework
- **CrewAI** - Multi-agent orchestration
- **LangChain** - Language model integration

### AI & ML
- **OpenAI GPT-4** - Language model capabilities
- **Transformers** - Hugging Face model integration
- **NLTK/spaCy** - Natural language processing
- **Custom Models** - Fine-tuned intent classification

### Database & Caching
- **PostgreSQL** - Primary database
- **Redis** - Caching and session management
- **SQLAlchemy** - ORM and database abstraction
- **Alembic** - Database migrations

### Infrastructure
- **Docker** - Containerization
- **Kubernetes** - Container orchestration
- **Nginx** - Reverse proxy and load balancing
- **Prometheus** - Metrics and monitoring

## 📖 Getting Started

### Quick Installation

```bash
# Clone the repository
git clone https://github.com/PiGrieco/WordOfPrompt-Integration.git
cd WordOfPrompt-Integration/wordofprompt

# Install dependencies
pip install -r requirements.txt

# Set up configuration
cp config/env.example config/.env
# Edit config/.env with your API keys

# Run the application
python -m src.api.main
```

### Docker Installation

```bash
# Using Docker Compose
cd deployment/docker
docker-compose up --build
```

### Next Steps

1. **Configure your environment** - See [Configuration Guide](dev/configuration.md)
2. **Set up your API keys** - Get keys for OpenAI, Rainforest, and other services
3. **Run your first search** - Try the example in [Quick Start](user/quickstart.md)
4. **Explore the API** - Check out the [API Documentation](api/README.md)

## 🤝 Community & Support

- **GitHub Issues** - Bug reports and feature requests
- **Documentation** - Comprehensive guides and references
- **Examples** - Real-world usage examples
- **Contributing** - How to contribute to the project

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

---

**Ready to build intelligent product recommendations? Let's get started! 🚀**
