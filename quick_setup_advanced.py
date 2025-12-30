#!/usr/bin/env python3
"""
CarBot Pro - Advanced Multi-Agent System Quick Setup
====================================================

Quick setup script for the advanced multi-agent car sales system.
Configures the environment, installs dependencies and prepares the system for the demo.

Author: Eduardo Hilario, CTO IA For Transport
For: AI Agents Day Demo
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import json

def print_header():
    """Print setup header"""
    print("=" * 70)
    print("🚗 CarBot Pro - Advanced Multi-Agent System Setup")
    print("=" * 70)
    print("Demo for AI Agents Day")
    print("Author: Eduardo Hilario, CTO IA For Transport")
    print("=" * 70)
    print()

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    
    print(f"✅ Python {sys.version.split()[0]} - Compatible")
    return True

def create_virtual_environment():
    """Create and activate virtual environment"""
    print("\n🔧 Setting up virtual environment...")
    
    venv_path = Path(".venv")
    
    if venv_path.exists():
        print("⚠️  Existing virtual environment found")
        response = input("Do you want to recreate it? (y/N): ").lower().strip()
        if response == 'y':
            print("🗑️  Removing existing virtual environment...")
            shutil.rmtree(venv_path)
        else:
            print("✅ Using existing virtual environment")
            return True
    
    try:
        print("📦 Creating new virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", ".venv"], check=True)
        print("✅ Virtual environment created successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error creating virtual environment: {e}")
        return False

def get_pip_command():
    """Get the correct pip command for the platform"""
    if os.name == 'nt':  # Windows
        return [".venv/Scripts/pip"]
    else:  # Unix/Linux/macOS
        return [".venv/bin/pip"]

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    
    pip_cmd = get_pip_command()
    
    try:
        # Upgrade pip first
        print("🔄 Updating pip...")
        subprocess.run(pip_cmd + ["install", "--upgrade", "pip"], check=True)
        
        # Install requirements
        print("📥 Installing project dependencies...")
        subprocess.run(pip_cmd + ["install", "-r", "requirements.txt"], check=True)
        
        print("✅ Dependencies installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def setup_environment_file():
    """Setup environment configuration file"""
    print("\n🔑 Setting up environment variables...")
    
    env_file = Path(".env")
    config_file = Path("config.env")
    
    if env_file.exists():
        print("⚠️  Existing .env file found")
        response = input("Do you want to overwrite it? (y/N): ").lower().strip()
        if response != 'y':
            print("✅ Keeping existing configuration")
            return True
    
    if config_file.exists():
        print("📋 Copying configuration from config.env...")
        shutil.copy(config_file, env_file)
    else:
        print("📝 Creating .env file...")
        env_content = """# CarBot Pro - API Keys Configuration
# Add your real keys here

# REQUIRED: Databricks Token for language models
# How to get your Databricks token: https://docs.databricks.com/en/dev-tools/auth/pat.html
DATABRICKS_TOKEN=your_databricks_token_here

# Databricks serving endpoint base URL
DATABRICKS_BASE_URL=your_databricks_base_url_here

# OPTIONAL: SerpAPI Key for real-time web search
SERPAPI_API_KEY=your_serpapi_key_here

# Database configuration
INVENTORY_PATH=data/enhanced_inventory.csv

# System configuration
DEBUG_MODE=true
LOG_LEVEL=INFO
"""
        with open(env_file, 'w', encoding='utf-8') as f:
            f.write(env_content)
    
    print("✅ .env file configured")
    print("⚠️  IMPORTANT: Edit the .env file with your real API keys")
    return True

def verify_data_files():
    """Verify that data files exist"""
    print("\n📊 Checking data files...")
    
    data_dir = Path("data")
    enhanced_inventory = data_dir / "enhanced_inventory.csv"
    
    if not data_dir.exists():
        print("📁 Creating data directory...")
        data_dir.mkdir()
    
    if enhanced_inventory.exists():
        print("✅ Enhanced inventory found")
        # Check file size
        file_size = enhanced_inventory.stat().st_size
        if file_size > 1000:  # At least 1KB
            print(f"✅ Valid inventory file ({file_size} bytes)")
        else:
            print("⚠️  Inventory file appears to be empty")
    else:
        print("❌ Enhanced inventory file not found")
        print("   Required: data/enhanced_inventory.csv")
        return False
    
    return True

def verify_system_files():
    """Verify that all system files exist"""
    print("\n🔍 Checking system files...")
    
    required_files = [
        "src/enhanced_app.py",
        "src/advanced_multi_agent_system.py",
        "src/enhanced_inventory_manager.py",
        "requirements.txt"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
        else:
            print(f"✅ {file_path}")
    
    if missing_files:
        print(f"❌ Missing files: {', '.join(missing_files)}")
        return False
    
    print("✅ All system files are present")
    return True

def create_demo_script():
    """Create demo script file"""
    print("\n🎬 Creating demo script...")
    
    demo_script = {
        "demo_title": "CarBot Pro - Advanced Multi-Agent System",
        "presenter": "Eduardo Hilario, CTO IA For Transport",
        "duration": "30 minutes",
        "sections": {
            "1_demo": {
                "title": "Live Demonstration (8-10 min)",
                "prompts": [
                    {
                        "step": 1,
                        "role": "Customer",
                        "prompt": "Hello, I'm looking for a car",
                        "expected": "Carlos greets and builds rapport"
                    },
                    {
                        "step": 2,
                        "role": "Customer", 
                        "prompt": "I need a bigger and safer car because we had a baby",
                        "expected": "Carlos updates profile and shows understanding"
                    },
                    {
                        "step": 3,
                        "role": "Customer",
                        "prompt": "I want a red sedan that's no more than 2 years old",
                        "expected": "Carlos consults manager and searches inventory"
                    },
                    {
                        "step": 4,
                        "role": "Customer",
                        "prompt": "I'm interested in BMWs",
                        "expected": "Carlos refines search and presents options"
                    },
                    {
                        "step": 5,
                        "role": "Customer",
                        "prompt": "What safety features does it have for babies?",
                        "expected": "Carlos consults Maria for research"
                    },
                    {
                        "step": 6,
                        "role": "Customer",
                        "prompt": "What trunk space does the BMW X3 have?",
                        "expected": "Maria provides specific data"
                    },
                    {
                        "step": 7,
                        "role": "Customer",
                        "prompt": "What's the price of the black BMW X3?",
                        "expected": "Carlos consults manager for price"
                    },
                    {
                        "step": 8,
                        "role": "Customer",
                        "prompt": "Can you offer any discount?",
                        "expected": "Negotiation between Carlos and manager"
                    },
                    {
                        "step": 9,
                        "role": "Customer",
                        "prompt": "I'll take it",
                        "expected": "Carlos finalizes sale and updates inventory"
                    }
                ]
            },
            "2_code_review": {
                "title": "Code Review (20-22 min)",
                "topics": [
                    "Multi-agent architecture",
                    "Intelligent inventory management",
                    "Inter-agent communication system",
                    "Real-time logs and analytics",
                    "External API integration",
                    "State and memory management"
                ]
            }
        },
        "key_features": [
            "Multi-agent system with specialized roles",
            "Intelligent search in enhanced inventory",
            "Real-time web research",
            "Automatic negotiation between agents",
            "Dynamic customer profiling",
            "Detailed logs and analytics",
            "Modern interface with Streamlit"
        ]
    }
    
    with open("demo_script_advanced.json", 'w', encoding='utf-8') as f:
        json.dump(demo_script, f, indent=2, ensure_ascii=False)
    
    print("✅ Demo script created: demo_script_advanced.json")

def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "=" * 70)
    print("🎉 SETUP COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print()
    print("📋 NEXT STEPS:")
    print()
    print("1. 🔑 CONFIGURE API KEYS:")
    print("   - Edit the .env file")
    print("   - Add your Databricks Token (REQUIRED)")
    print("   - Add your Databricks Base URL (if different from default)")
    print("   - Add your SerpAPI Key (OPTIONAL)")
    print()
    print("2. 🚀 RUN THE APPLICATION:")
    
    if os.name == 'nt':  # Windows
        print("   .venv\\Scripts\\activate")
        print("   streamlit run src/enhanced_app.py")
    else:  # Unix/Linux/macOS
        print("   source .venv/bin/activate")
        print("   streamlit run src/enhanced_app.py")
    
    print()
    print("3. 🎬 PREPARE DEMO:")
    print("   - Review demo_script_advanced.json")
    print("   - Practice the suggested prompts")
    print("   - Verify that all agents respond")
    print()
    print("4. 🔧 DEBUG MODE:")
    print("   - Enable detailed logs in the interface")
    print("   - Monitor inter-agent communications")
    print("   - Verify real-time analytics")
    print()
    print("=" * 70)
    print("🎯 READY FOR AI AGENTS DAY DEMO!")
    print("=" * 70)

def main():
    """Main setup function"""
    print_header()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create virtual environment
    if not create_virtual_environment():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Setup environment file
    if not setup_environment_file():
        sys.exit(1)
    
    # Verify data files
    if not verify_data_files():
        sys.exit(1)
    
    # Verify system files
    if not verify_system_files():
        sys.exit(1)
    
    # Create demo script
    create_demo_script()
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    main() 