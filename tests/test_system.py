#!/usr/bin/env python3
"""
Test script para verificar que el sistema multiagente funcione correctamente
"""

import os
import sys

# Add src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import os

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_system():
    """Test the multi-agent system"""
    print("🧪 Iniciando pruebas del sistema...")
    
    # Test 1: Import modules
    try:
        from enhanced_inventory_manager import get_inventory_manager
        from advanced_multi_agent_system import get_advanced_multi_agent_system
        print("✅ Test 1: Importaciones exitosas")
    except Exception as e:
        print(f"❌ Test 1: Error en importaciones: {e}")
        return False
    
    # Test 2: Initialize inventory manager
    try:
        inventory_manager = get_inventory_manager()
        stats = inventory_manager.get_inventory_stats()
        print(f"✅ Test 2: Inventario cargado - {stats.get('total_vehicles', 0)} vehículos")
    except Exception as e:
        print(f"❌ Test 2: Error en inventario: {e}")
        return False
    
    # Test 3: Test inventory search
    try:
        results = inventory_manager.intelligent_search("BMW sedan rojo", max_results=3)
        print(f"✅ Test 3: Búsqueda inteligente - {len(results)} resultados")
    except Exception as e:
        print(f"❌ Test 3: Error en búsqueda: {e}")
        return False
    
    # Test 4: Initialize multi-agent system (only if API key is available)
    databricks_token = os.getenv('DATABRICKS_TOKEN')
    if databricks_token and databricks_token.startswith('dapi'):
        try:
            agent_system = get_advanced_multi_agent_system(databricks_token)
            print("✅ Test 4: Sistema multiagente inicializado")
            
            # Test 5: Simple interaction
            try:
                response = agent_system.process_customer_input("Hola")
                print(f"✅ Test 5: Interacción básica exitosa")
                print(f"   Respuesta: {response[:100]}...")
            except Exception as e:
                print(f"❌ Test 5: Error en interacción: {e}")
                return False
                
        except Exception as e:
            print(f"❌ Test 4: Error inicializando sistema: {e}")
            return False
    else:
        print("⚠️  Test 4: Saltado - No hay Databricks Token válido")
        print("⚠️  Test 5: Saltado - Requiere Databricks Token")
    
    print("\n🎉 ¡Todos los tests completados exitosamente!")
    return True

if __name__ == "__main__":
    success = test_system()
    exit(0 if success else 1) 