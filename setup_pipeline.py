#!/usr/bin/env python3
"""
Setup script for AI Stock GPT with Parallel AI Data Pipeline
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def print_banner():
    """Print setup banner"""
    print("=" * 60)
    print("🚀 AI Stock GPT - Data Pipeline Setup")
    print("=" * 60)
    print()

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    
    requirements_file = "requirements_pipeline.txt"
    if not os.path.exists(requirements_file):
        print(f"❌ Requirements file {requirements_file} not found")
        return False
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
        print("✅ Packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install packages: {e}")
        return False

def setup_environment():
    """Setup environment variables"""
    print("\n🔧 Environment Setup")
    print("-" * 30)
    
    # Check if API key is already set
    api_key = os.getenv("PARALLEL_API_KEY")
    if api_key:
        print(f"✅ PARALLEL_API_KEY already set: {api_key[:8]}...")
        return True
    
    print("To use the data pipeline, you need a Parallel AI API key.")
    print("Get your API key from: https://parallel.ai/")
    print()
    
    # Create .env file
    env_file = ".env"
    api_key = input("Enter your Parallel AI API key (or press Enter to skip): ").strip()
    
    if api_key:
        try:
            with open(env_file, "w") as f:
                f.write(f"PARALLEL_API_KEY={api_key}\n")
            print(f"✅ API key saved to {env_file}")
            
            # Set environment variable for current session
            os.environ["PARALLEL_API_KEY"] = api_key
            return True
        except Exception as e:
            print(f"❌ Failed to save API key: {e}")
            return False
    else:
        print("⚠️  No API key provided. Data pipeline will not be available.")
        return False

def create_data_directory():
    """Create data directory for pipeline results"""
    data_dir = Path("data")
    try:
        data_dir.mkdir(exist_ok=True)
        print("✅ Data directory created")
        return True
    except Exception as e:
        print(f"❌ Failed to create data directory: {e}")
        return False

def test_pipeline():
    """Test the data pipeline"""
    print("\n🧪 Testing Data Pipeline")
    print("-" * 30)
    
    try:
        from data_pipeline import StockMarketDataPipeline
        print("✅ Data pipeline module imported successfully")
        
        api_key = os.getenv("PARALLEL_API_KEY")
        if not api_key:
            print("⚠️  No API key found. Skipping pipeline test.")
            return True
        
        print("🔄 Testing pipeline initialization...")
        pipeline = StockMarketDataPipeline(api_key)
        print("✅ Pipeline initialized successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import data pipeline: {e}")
        print("Make sure you've installed the requirements: pip install -r requirements_pipeline.txt")
        return False
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        return False

def create_startup_script():
    """Create startup script"""
    script_content = """#!/bin/bash
# AI Stock GPT Enhanced Backend Startup Script

echo "🚀 Starting AI Stock GPT Enhanced Backend..."

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | xargs)
    echo "✅ Environment variables loaded"
fi

# Check if API key is set
if [ -z "$PARALLEL_API_KEY" ]; then
    echo "⚠️  PARALLEL_API_KEY not set. Data pipeline will not be available."
fi

# Start the enhanced backend
cd backend
python enhanced_backend_with_pipeline.py
"""
    
    try:
        with open("start_enhanced_backend.sh", "w") as f:
            f.write(script_content)
        os.chmod("start_enhanced_backend.sh", 0o755)
        print("✅ Startup script created: start_enhanced_backend.sh")
        return True
    except Exception as e:
        print(f"❌ Failed to create startup script: {e}")
        return False

def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "=" * 60)
    print("🎉 Setup Complete!")
    print("=" * 60)
    print()
    print("📋 Next Steps:")
    print("1. Start the enhanced backend:")
    print("   ./start_enhanced_backend.sh")
    print("   or")
    print("   cd backend && python enhanced_backend_with_pipeline.py")
    print()
    print("2. Start the frontend (in a new terminal):")
    print("   npm start")
    print()
    print("3. Access the application:")
    print("   Frontend: http://localhost:3000")
    print("   Backend: http://localhost:8000")
    print()
    print("🔧 Available Commands:")
    print("- 'run pipeline' - Start market analysis")
    print("- 'market analysis' - View latest analysis")
    print("- 'predict AAPL' - Stock predictions")
    print("- 'technical analysis TSLA' - Technical analysis")
    print()
    print("📚 Documentation:")
    print("- Data Pipeline: data_pipeline.py")
    print("- Enhanced Backend: enhanced_backend_with_pipeline.py")
    print("- API Documentation: http://localhost:8000/docs")
    print()

def main():
    """Main setup function"""
    print_banner()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        print("❌ Setup failed during package installation")
        sys.exit(1)
    
    # Setup environment
    setup_environment()
    
    # Create data directory
    if not create_data_directory():
        print("❌ Setup failed during directory creation")
        sys.exit(1)
    
    # Test pipeline
    test_pipeline()
    
    # Create startup script
    create_startup_script()
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    main()
