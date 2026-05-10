#!/usr/bin/env python3
"""
Master Startup Script for AI Stock GPT
======================================

This script can start both the frontend and backend servers for the AI Stock GPT application.
"""

import os
import sys
import subprocess
import time
import threading
import signal
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AppManager:
    """Manages the startup and shutdown of the AI Stock GPT application."""
    
    def __init__(self):
        self.frontend_process = None
        self.backend_process = None
        self.running = False
        
    def check_prerequisites(self):
        """Check if all prerequisites are met."""
        logger.info("🔍 Checking prerequisites...")
        
        # Check Python version
        if sys.version_info < (3, 8):
            logger.error("❌ Python 3.8+ is required")
            return False
        
        # Check Node.js
        try:
            result = subprocess.run(["node", "--version"], capture_output=True, text=True)
            if result.returncode != 0:
                logger.error("❌ Node.js is not installed")
                return False
            logger.info(f"✅ Node.js: {result.stdout.strip()}")
        except FileNotFoundError:
            logger.error("❌ Node.js is not installed")
            return False
        
        # Check npm
        try:
            result = subprocess.run(["npm", "--version"], capture_output=True, text=True)
            if result.returncode != 0:
                logger.error("❌ npm is not installed")
                return False
            logger.info(f"✅ npm: {result.stdout.strip()}")
        except FileNotFoundError:
            logger.error("❌ npm is not installed")
            return False
        
        return True
    
    def install_dependencies(self):
        """Install required dependencies."""
        logger.info("📦 Installing dependencies...")
        
        # Install Python dependencies
        try:
            logger.info("Installing Python dependencies...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "api_requirements.txt"], 
                          check=True, capture_output=True, text=True)
            logger.info("✅ Python dependencies installed")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install Python dependencies: {e}")
            return False
        
        # Install React dependencies
        try:
            logger.info("Installing React dependencies...")
            subprocess.run(["npm", "install"], check=True, capture_output=True, text=True)
            logger.info("✅ React dependencies installed")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install React dependencies: {e}")
            return False
        
        return True
    
    def start_backend(self):
        """Start the FastAPI backend server."""
        logger.info("🚀 Starting FastAPI backend...")
        
        try:
            # Change to backend directory
            backend_dir = Path("backend")
            if not backend_dir.exists():
                logger.error("❌ Backend directory not found")
                return False
            
            # Start the server
            self.backend_process = subprocess.Popen([
                sys.executable, "-m", "uvicorn", 
                "main:app", 
                "--host", "0.0.0.0", 
                "--port", "8000", 
                "--reload"
            ], cwd=backend_dir)
            
            logger.info("✅ Backend server started on http://localhost:8000")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start backend: {e}")
            return False
    
    def start_frontend(self):
        """Start the React frontend server."""
        logger.info("🎨 Starting React frontend...")
        
        try:
            # Start the development server
            self.frontend_process = subprocess.Popen([
                "npm", "start"
            ])
            
            logger.info("✅ Frontend server started on http://localhost:3000")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start frontend: {e}")
            return False
    
    def wait_for_backend(self, timeout=30):
        """Wait for backend to be ready."""
        logger.info("⏳ Waiting for backend to be ready...")
        
        import requests
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get("http://localhost:8000/status", timeout=5)
                if response.status_code == 200:
                    logger.info("✅ Backend is ready!")
                    return True
            except:
                pass
            
            time.sleep(1)
        
        logger.warning("⚠️ Backend may not be fully ready")
        return False
    
    def start(self, install_deps=False):
        """Start both frontend and backend."""
        logger.info("=" * 60)
        logger.info("🤖 AI Stock GPT - Starting Application")
        logger.info("=" * 60)
        
        # Check prerequisites
        if not self.check_prerequisites():
            return False
        
        # Install dependencies if requested
        if install_deps:
            if not self.install_dependencies():
                return False
        
        # Start backend
        if not self.start_backend():
            return False
        
        # Wait for backend to be ready
        self.wait_for_backend()
        
        # Start frontend
        if not self.start_frontend():
            self.stop()
            return False
        
        self.running = True
        logger.info("=" * 60)
        logger.info("🎉 AI Stock GPT is now running!")
        logger.info("📱 Frontend: http://localhost:3000")
        logger.info("🔧 Backend:  http://localhost:8000")
        logger.info("📚 API Docs: http://localhost:8000/docs")
        logger.info("=" * 60)
        logger.info("Press Ctrl+C to stop the application")
        
        return True
    
    def stop(self):
        """Stop both frontend and backend."""
        logger.info("🛑 Stopping AI Stock GPT...")
        
        if self.frontend_process:
            self.frontend_process.terminate()
            self.frontend_process.wait()
            logger.info("✅ Frontend stopped")
        
        if self.backend_process:
            self.backend_process.terminate()
            self.backend_process.wait()
            logger.info("✅ Backend stopped")
        
        self.running = False
        logger.info("👋 AI Stock GPT stopped")

def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info(f"\n🛑 Received signal {signum}, shutting down...")
    if app_manager.running:
        app_manager.stop()
    sys.exit(0)

def main():
    """Main function."""
    global app_manager
    
    # Parse command line arguments
    install_deps = "--install" in sys.argv or "-i" in sys.argv
    
    # Create app manager
    app_manager = AppManager()
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start the application
        if app_manager.start(install_deps=install_deps):
            # Keep the main thread alive
            while app_manager.running:
                time.sleep(1)
        else:
            logger.error("❌ Failed to start AI Stock GPT")
            return 1
            
    except KeyboardInterrupt:
        logger.info("\n🛑 Stopped by user")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return 1
    finally:
        if app_manager.running:
            app_manager.stop()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
