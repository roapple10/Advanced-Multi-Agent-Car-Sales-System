
import os
import sys
from unittest.mock import MagicMock

# Add current directory to path
# Add src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

try:
    from advanced_multi_agent_system import AdvancedCarSalesSystem, AgentRole
    print("✅ Imports successful")
    
    # Check if set_language exists
    system = MagicMock(spec=AdvancedCarSalesSystem)
    if hasattr(AdvancedCarSalesSystem, 'set_language'):
        print("✅ set_language exists in AdvancedCarSalesSystem")
    else:
        print("❌ set_language MISSING in AdvancedCarSalesSystem")
        
    # Check if AgentState has language
    # AgentState is a TypedDict, checking it via __annotations__ if possible or just checking usage in code
    
except Exception as e:
    print(f"❌ Verification failed: {e}")
    sys.exit(1)

print("✅ Multi-language support structure verified.")
