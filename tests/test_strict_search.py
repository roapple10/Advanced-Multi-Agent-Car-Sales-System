
import sys
import os

# Add current directory to path
# Add src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from enhanced_inventory_manager import get_inventory_manager

manager = get_inventory_manager()

print("🔍 Testing search for '2025 Audi A4'...")
results = manager.intelligent_search("Show me all 2025 Audi A4")
print(f"Results found: {len(results)}")

if len(results) == 0:
    print("✅ SUCCESS: Strict year filtering works.")
else:
    print("❌ FAILURE: Still finding vehicles for non-existent year.")
    for r in results:
        print(f" - Found: {r.year} {r.make} {r.model}")

print("\n🔍 Testing search for 'Audi A4' (no year)...")
results_any = manager.intelligent_search("Audi A4")
print(f"Results found: {len(results_any)}")
for r in results_any:
    print(f" - Found: {r.year} {r.make} {r.model}")
