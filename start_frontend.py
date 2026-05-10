#!/usr/bin/env python3
"""
Start Frontend Script for AI Stock GPT
======================================

This script starts the React frontend for the AI Stock GPT application.
"""

import os
import sys
import subprocess
import time
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_node_installed():
    """Check if Node.js is installed."""
    try:
        result = subprocess.run(["node", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"✅ Node.js installed: {result.stdout.strip()}")
            return True
        else:
            logger.error("❌ Node.js not found")
            return False
    except FileNotFoundError:
        logger.error("❌ Node.js not installed. Please install Node.js from https://nodejs.org/")
        return False

def check_npm_installed():
    """Check if npm is installed."""
    try:
        result = subprocess.run(["npm", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"✅ npm installed: {result.stdout.strip()}")
            return True
        else:
            logger.error("❌ npm not found")
            return False
    except FileNotFoundError:
        logger.error("❌ npm not installed. Please install npm with Node.js")
        return False

def install_dependencies():
    """Install React dependencies."""
    logger.info("Installing React dependencies...")
    try:
        subprocess.run(["npm", "install"], check=True, capture_output=True, text=True)
        logger.info("✅ React dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install React dependencies: {e}")
        return False

def check_dependencies_installed():
    """Check if React dependencies are installed."""
    return os.path.exists("node_modules") and os.path.exists("package-lock.json")

def start_frontend():
    """Start the React frontend development server."""
    logger.info("🚀 Starting AI Stock GPT Frontend...")
    
    # Check if we're in the right directory
    if not os.path.exists("package.json"):
        logger.error("❌ package.json not found. Please run this script from the project root.")
        return False
    
    try:
        # Start the development server
        logger.info("Starting React development server on http://localhost:3000")
        logger.info("Press Ctrl+C to stop the server")
        
        subprocess.run(["npm", "start"])
        
    except KeyboardInterrupt:
        logger.info("🛑 Frontend server stopped by user")
    except Exception as e:
        logger.error(f"❌ Error starting frontend server: {e}")
        return False
    
    return True

def main():
    """Main function."""
    print("=" * 60)
    print("🎨 AI Stock GPT Frontend Startup")
    print("=" * 60)
    
    # Check Node.js and npm
    if not check_node_installed():
        return False
    
    if not check_npm_installed():
        return False
    
    # Check if dependencies are installed
    if not check_dependencies_installed():
        logger.info("React dependencies not found. Installing...")
        if not install_dependencies():
            logger.error("Failed to install React dependencies.")
            return False
    
    # Start frontend
    return start_frontend()

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
