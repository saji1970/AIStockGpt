#!/usr/bin/env python3
"""
Simple script to set up Parallel AI API key
"""

import os
import sys

def setup_api_key():
    """Set up the Parallel AI API key"""
    
    print("🔑 Parallel AI API Key Setup")
    print("=" * 40)
    print()
    
    # Check if API key is already set
    existing_key = os.getenv("PARALLEL_API_KEY")
    if existing_key:
        print(f"✅ API key already set: {existing_key[:8]}...")
        response = input("Do you want to update it? (y/n): ").lower()
        if response != 'y':
            print("Keeping existing API key.")
            return True
    
    print("📋 Instructions:")
    print("1. Go to https://parallel.ai/")
    print("2. Sign up for an account")
    print("3. Go to your dashboard")
    print("4. Find 'API Keys' or 'Developer' section")
    print("5. Generate a new API key")
    print("6. Copy the API key (starts with 'pk_')")
    print()
    
    # Get API key from user
    api_key = input("Enter your Parallel AI API key: ").strip()
    
    if not api_key:
        print("❌ No API key provided. Setup cancelled.")
        return False
    
    # Validate API key format (basic check)
    if not api_key.startswith('pk_'):
        print("⚠️  Warning: API key should start with 'pk_'. Continue anyway? (y/n): ")
        response = input().lower()
        if response != 'y':
            return False
    
    # Create .env file
    try:
        with open(".env", "w") as f:
            f.write(f"PARALLEL_API_KEY={api_key}\n")
        print("✅ API key saved to .env file")
        
        # Set environment variable for current session
        os.environ["PARALLEL_API_KEY"] = api_key
        print("✅ API key set for current session")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving API key: {e}")
        return False

def test_api_key():
    """Test if the API key works"""
    print("\n🧪 Testing API Key...")
    
    try:
        from data_pipeline import StockMarketDataPipeline
        
        api_key = os.getenv("PARALLEL_API_KEY")
        if not api_key:
            print("❌ No API key found")
            return False
        
        print("🔄 Initializing pipeline...")
        pipeline = StockMarketDataPipeline(api_key)
        print("✅ Pipeline initialized successfully!")
        
        return True
        
    except ImportError:
        print("❌ Data pipeline not available. Install: pip install parallel-web")
        return False
    except Exception as e:
        print(f"❌ API key test failed: {e}")
        print("This might be due to:")
        print("- Invalid API key")
        print("- Network connectivity issues")
        print("- Insufficient API credits")
        return False

def print_next_steps():
    """Print next steps"""
    print("\n🎉 API Key Setup Complete!")
    print("=" * 30)
    print()
    print("📋 Next Steps:")
    print("1. Start the enhanced backend:")
    print("   cd backend")
    print("   python enhanced_backend_with_pipeline.py")
    print()
    print("2. Start the frontend (in new terminal):")
    print("   npm start")
    print()
    print("3. Test the pipeline:")
    print("   - Ask 'run pipeline' in the chat")
    print("   - Ask 'market analysis' to view results")
    print()
    print("🔧 Available Commands:")
    print("- 'run pipeline' - Start market analysis")
    print("- 'market analysis' - View latest analysis")
    print("- 'predict AAPL' - Stock predictions")
    print("- 'technical analysis TSLA' - Technical analysis")

def main():
    """Main function"""
    print("🚀 AI Stock GPT - API Key Setup")
    print("=" * 40)
    print()
    
    # Setup API key
    if not setup_api_key():
        print("❌ Setup failed")
        sys.exit(1)
    
    # Test API key
    if test_api_key():
        print("✅ API key is working!")
    else:
        print("⚠️  API key test failed, but you can still try using the system")
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    main()
