
import os
import sys
from unittest.mock import MagicMock

# Add current directory to path
# Add src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

try:
    from advanced_multi_agent_system import AdvancedCarSalesSystem, AgentRole
    print("✅ Imports successful")
    
    # Mock LLMs to avoid API calls
    mock_llm = MagicMock()
    
    # Create system with mocks
    # Note: We need to mock the LLM initialization in __init__ or just mock the whole thing
    # For a quick check, let's just see if the class can be instantiated if we mock the LLMs
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

print("✅ LangGraph migration seems syntactically correct.")
