#!/usr/bin/env python3
"""
Start Backend Script for AI Stock GPT
=====================================

This script starts the FastAPI backend server for the AI Stock GPT application.
"""

import os
import sys
import subprocess
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import fastapi
        import uvicorn
        import numpy
        import pandas
        import tensorflow
        import sklearn
        import yfinance
        logger.info("✅ All required dependencies are installed")
        return True
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        logger.info("Please install dependencies with: pip install -r api_requirements.txt")
        return False

def install_dependencies():
    """Install required dependencies."""
    logger.info("Installing dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "api_requirements.txt"], 
                      check=True, capture_output=True, text=True)
        logger.info("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install dependencies: {e}")
        return False

def start_backend():
    """Start the FastAPI backend server."""
    logger.info("🚀 Starting AI Stock GPT Backend...")
    
    # Check if we're in the right directory
    if not os.path.exists("backend/main.py"):
        logger.error("❌ backend/main.py not found. Please run this script from the project root.")
        return False
    
    # Change to backend directory
    os.chdir("backend")
    
    try:
        # Start the server
        logger.info("Starting FastAPI server on http://localhost:8000")
        logger.info("Press Ctrl+C to stop the server")
        
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "main:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ])
        
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Error starting server: {e}")
        return False
    
    return True

def main():
    """Main function."""
    print("=" * 60)
    print("🤖 AI Stock GPT Backend Startup")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        logger.info("Attempting to install dependencies...")
        if not install_dependencies():
            logger.error("Failed to install dependencies. Please install manually.")
            return False
    
    # Start backend
    return start_backend()

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
