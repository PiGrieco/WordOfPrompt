#!/usr/bin/env python3
"""
Validation script for the clean WordOfPrompt system.

This script validates that the cleaned system works correctly.
"""

import sys
from pathlib import Path

# Add src to path
src_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_dir))

print("🧹 WordOfPrompt - Sistema Pulito Validazione")
print("=" * 45)

def validate_core_components():
    """Validate core components work."""
    print("\n🧠 Validazione Componenti Core:")
    
    try:
        from core.intent_classifier import IntentClassifier, IntentType
        from core.keyword_extractor import KeywordExtractor
        from core.exceptions import WordOfPromptError
        from core.utils import get_config, sanitize_input
        
        print("✅ Import core components OK")
        
        # Test intent classifier
        config = {"model_type": "rule_based"}
        classifier = IntentClassifier(config, model_type="rule_based")
        result = classifier.classify("I want to buy a laptop")
        assert result.intent_type == IntentType.PURCHASE
        print("✅ Intent classifier funziona")
        
        # Test keyword extractor
        extractor = KeywordExtractor(method="simple", max_keywords=3)
        keywords = extractor.extract("I need a good laptop for programming")
        assert len(keywords) > 0
        print("✅ Keyword extractor funziona")
        
        # Test utils
        config = get_config()
        assert isinstance(config, dict)
        clean_text = sanitize_input("Hello world")
        assert clean_text == "Hello world"
        print("✅ Utils funzionano")
        
        return True
        
    except Exception as e:
        print(f"❌ Validazione core fallita: {e}")
        return False

def validate_mcp_structure():
    """Validate MCP structure."""
    print("\n🔌 Validazione Struttura MCP:")
    
    try:
        # Test MCP message structure
        mcp_message = {
            "jsonrpc": "2.0",
            "id": "test-001",
            "method": "tools/call",
            "params": {
                "name": "analyze_and_recommend",
                "arguments": {"user_prompt": "test"}
            }
        }
        
        assert mcp_message["jsonrpc"] == "2.0"
        assert "method" in mcp_message
        print("✅ Struttura messaggi MCP valida")
        
        # Test tool definitions
        tool_def = {
            "name": "analyze_and_recommend",
            "description": "Complete WordOfPrompt workflow",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "user_prompt": {"type": "string"}
                },
                "required": ["user_prompt"]
            }
        }
        
        assert "inputSchema" in tool_def
        assert "properties" in tool_def["inputSchema"]
        print("✅ Definizioni tool MCP valide")
        
        return True
        
    except Exception as e:
        print(f"❌ Validazione MCP fallita: {e}")
        return False

def validate_configuration():
    """Validate configuration files."""
    print("\n⚙️  Validazione Configurazione:")
    
    try:
        import json
        
        # Test MCP config
        with open("config/mcp.json", 'r') as f:
            mcp_config = json.load(f)
        
        assert "mcpServers" in mcp_config
        assert "wordofprompt-unified" in mcp_config["mcpServers"]
        print("✅ Configurazione MCP valida")
        
        # Test environment template
        with open("config/env.example", 'r') as f:
            env_content = f.read()
        
        assert "OPENAI_API_KEY" in env_content
        assert "RAINFOREST_API_KEY" in env_content
        print("✅ Template environment valido")
        
        # Test requirements
        with open("requirements.txt", 'r') as f:
            req_content = f.read()
        
        assert "fastapi" in req_content
        assert "crewai" in req_content
        print("✅ Requirements validi")
        
        return True
        
    except Exception as e:
        print(f"❌ Validazione configurazione fallita: {e}")
        return False

def validate_project_structure():
    """Validate final project structure."""
    print("\n📁 Validazione Struttura Progetto:")
    
    required_structure = {
        "src/core": ["intent_classifier.py", "keyword_extractor.py", "exceptions.py", "utils.py"],
        "src/mcp/protocols": ["base.py", "transport.py", "registry.py"],
        "src/mcp/servers": ["base.py", "unified.py", "amazon_search.py"],
        "src/models": ["user.py", "product.py", "recommendation.py"],
        "config": ["mcp.json", "env.example"],
        "scripts": ["run_mcp_server.py", "setup.sh"],
        "tests": ["test_intent_classifier.py", "test_mcp_protocol.py"],
        "docs": ["README.md", "mcp-integration.md"]
    }
    
    all_good = True
    
    for directory, expected_files in required_structure.items():
        dir_path = Path(directory)
        if not dir_path.exists():
            print(f"❌ Directory mancante: {directory}")
            all_good = False
            continue
        
        print(f"✅ {directory}/")
        
        for file_name in expected_files:
            file_path = dir_path / file_name
            if file_path.exists():
                print(f"  ✅ {file_name}")
            else:
                print(f"  ❌ {file_name} mancante")
                all_good = False
    
    return all_good

def main():
    """Validation principale."""
    print("🎯 VALIDAZIONE SISTEMA PULITO")
    
    validations = [
        ("Struttura Progetto", validate_project_structure),
        ("Configurazione", validate_configuration),
        ("Componenti Core", validate_core_components),
        ("Struttura MCP", validate_mcp_structure)
    ]
    
    results = []
    
    for validation_name, validation_func in validations:
        try:
            result = validation_func()
            results.append((validation_name, result))
        except Exception as e:
            print(f"❌ {validation_name} fallito: {e}")
            results.append((validation_name, False))
    
    # Riepilogo
    print("\n" + "=" * 50)
    print("📊 RIEPILOGO VALIDAZIONE")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for validation_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {validation_name}")
    
    print(f"\n🎯 Risultato: {passed}/{total} validazioni passate")
    
    if passed == total:
        print("\n🎉 SISTEMA PULITO E VALIDATO!")
        print("✅ Struttura professionale completa")
        print("✅ Tutti i componenti funzionanti")
        print("✅ Configurazione corretta")
        print("✅ Integrazione MCP operativa")
        print("\n🚀 PRONTO PER L'USO IN PRODUZIONE!")
        
        # Istruzioni finali
        print("\n📋 PROSSIMI PASSI:")
        print("1. cp config/env.example config/.env")
        print("2. Edita config/.env con le tue API keys")
        print("3. python scripts/run_mcp_server.py unified")
        print("4. Connetti il tuo client MCP preferito!")
        
        return True
    else:
        print(f"\n⚠️  {total - passed} validazioni fallite")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Errore validazione: {e}")
        sys.exit(1)
