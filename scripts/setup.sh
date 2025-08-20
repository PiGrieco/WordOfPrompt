#!/bin/bash
# WordOfPrompt Setup Script
# This script sets up the development environment for WordOfPrompt

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Main setup function
main() {
    print_status "🚀 Setting up WordOfPrompt development environment..."
    echo
    
    # Check Python version
    print_status "Checking Python version..."
    if command_exists python3; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        if [[ $(echo "$PYTHON_VERSION >= 3.9" | bc -l) -eq 1 ]]; then
            print_success "Python $PYTHON_VERSION found ✓"
        else
            print_error "Python 3.9+ required, found $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "Python 3 not found. Please install Python 3.9+"
        exit 1
    fi
    
    # Check if we're in the right directory
    if [[ ! -f "pyproject.toml" ]]; then
        print_error "Please run this script from the wordofprompt directory"
        exit 1
    fi
    
    # Create virtual environment
    print_status "Creating virtual environment..."
    if [[ ! -d "venv" ]]; then
        python3 -m venv venv
        print_success "Virtual environment created ✓"
    else
        print_warning "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    print_status "Activating virtual environment..."
    source venv/bin/activate
    
    # Upgrade pip
    print_status "Upgrading pip..."
    pip install --upgrade pip
    
    # Install development dependencies
    print_status "Installing development dependencies..."
    pip install -r requirements-dev.txt
    print_success "Dependencies installed ✓"
    
    # Set up pre-commit hooks
    print_status "Setting up pre-commit hooks..."
    pre-commit install
    print_success "Pre-commit hooks installed ✓"
    
    # Create configuration file
    print_status "Setting up configuration..."
    if [[ ! -f "config/.env" ]]; then
        cp config/env.example config/.env
        print_success "Configuration file created from template ✓"
        print_warning "Please edit config/.env with your API keys and settings"
    else
        print_warning "Configuration file already exists"
    fi
    
    # Create necessary directories
    print_status "Creating directories..."
    mkdir -p logs data tmp
    print_success "Directories created ✓"
    
    # Download NLTK data
    print_status "Downloading NLTK data..."
    python -c "
import nltk
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    print('NLTK data downloaded successfully')
except Exception as e:
    print(f'Warning: Could not download NLTK data: {e}')
"
    
    # Check Docker installation
    print_status "Checking Docker installation..."
    if command_exists docker; then
        print_success "Docker found ✓"
        if command_exists docker-compose; then
            print_success "Docker Compose found ✓"
        else
            print_warning "Docker Compose not found. Install it for full development experience."
        fi
    else
        print_warning "Docker not found. Install it for containerized development."
    fi
    
    # Run tests to verify setup
    print_status "Running tests to verify setup..."
    if python -m pytest tests/ -v --tb=short; then
        print_success "All tests passed ✓"
    else
        print_warning "Some tests failed. This might be due to missing API keys."
    fi
    
    echo
    print_success "🎉 WordOfPrompt development environment setup complete!"
    echo
    echo "Next steps:"
    echo "1. Edit config/.env with your API keys"
    echo "2. Run 'source venv/bin/activate' to activate the virtual environment"
    echo "3. Run 'python -m src.api.main' to start the development server"
    echo "4. Visit http://localhost:5000 to access the application"
    echo
    echo "For more information, see docs/dev/README.md"
}

# Run main function
main "$@"
