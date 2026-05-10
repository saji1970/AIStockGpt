#!/usr/bin/env python3
"""
Test script for the data pipeline using the actual API key
"""

from parallel import Parallel
import os
import json
from datetime import datetime

def test_basic_pipeline():
    """Test the basic pipeline functionality"""
    
    # Set the API key
    api_key = "VkvkaHQWl8twbuEhWsgagDp2sNYNdIOuIQMC1NqI"
    os.environ["PARALLEL_API_KEY"] = api_key
    
    print("🚀 Testing Data Pipeline with Your API Key")
    print("=" * 50)
    print()
    
    try:
        # Initialize the client
        client = Parallel(api_key=api_key)
        print("✅ Parallel AI client initialized successfully")
        
        # Test the exact prompt from your code
        input_prompt = """
        find all the news that can impact US stock market that can be used in sensitivity analysis
        """
        
        print("🔄 Creating task...")
        response = client.task_run.create(
            input=input_prompt,
            processor="ultra"
        )
        
        print(f"✅ Task created with ID: {response.run_id}")
        print("⏳ Waiting for results...")
        
        # Wait for the run to complete
        run_result = client.task_run.result(response.run_id, api_timeout=3600)
        
        print("✅ Task completed successfully!")
        print()
        print("📰 Market News Analysis Results:")
        print("-" * 40)
        print(run_result.output)
        
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"market_news_test_{timestamp}.json"
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "task_id": response.run_id,
            "api_key": api_key[:8] + "...",  # Only show first 8 chars for security
            "output": run_result.output
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_enhanced_pipeline():
    """Test the enhanced pipeline with more detailed analysis"""
    
    api_key = "VkvkaHQWl8twbuEhWsgagDp2sNYNdIOuIQMC1NqI"
    
    print("\n🔄 Testing Enhanced Pipeline...")
    print("-" * 30)
    
    try:
        client = Parallel(api_key=api_key)
        
        # Enhanced prompt for sector analysis
        enhanced_prompt = """
        Analyze the latest news that can impact the US stock market and provide:
        
        1. **Economic Indicators**: GDP, inflation, employment, interest rates
        2. **Sector-Specific News**: Technology, healthcare, finance, energy
        3. **Geopolitical Events**: Trade wars, political decisions
        4. **Company-Specific News**: Major earnings, mergers, scandals
        5. **Market Sentiment**: Analyst ratings, institutional moves
        
        For each news item, provide:
        - Headline and summary
        - Potential market impact (positive/negative/neutral)
        - Affected sectors or companies
        - Confidence level in the impact assessment
        
        Focus on high-impact news that could significantly move markets.
        """
        
        print("🔄 Creating enhanced analysis task...")
        response = client.task_run.create(
            input=enhanced_prompt,
            processor="ultra"
        )
        
        print(f"✅ Enhanced task created: {response.run_id}")
        print("⏳ Waiting for enhanced analysis...")
        
        run_result = client.task_run.result(response.run_id, api_timeout=3600)
        
        print("✅ Enhanced analysis completed!")
        print()
        print("📊 Enhanced Market Analysis:")
        print("-" * 40)
        print(run_result.output)
        
        # Save enhanced results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"enhanced_analysis_{timestamp}.json"
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "task_id": response.run_id,
            "analysis_type": "enhanced_market_analysis",
            "output": run_result.output
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Enhanced results saved to: {filename}")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced pipeline error: {e}")
        return False

def setup_environment():
    """Set up the environment with the API key"""
    
    print("🔧 Setting up environment...")
    
    # Create .env file
    try:
        with open(".env", "w") as f:
            f.write("PARALLEL_API_KEY=VkvkaHQWl8twbuEhWsgagDp2sNYNdIOuIQMC1NqI\n")
        print("✅ .env file created")
        
        # Set environment variable
        os.environ["PARALLEL_API_KEY"] = "VkvkaHQWl8twbuEhWsgagDp2sNYNdIOuIQMC1NqI"
        print("✅ Environment variable set")
        
        return True
        
    except Exception as e:
        print(f"❌ Error setting up environment: {e}")
        return False

def main():
    """Main test function"""
    
    print("🧪 Data Pipeline Test with Your API Key")
    print("=" * 50)
    print()
    
    # Setup environment
    if not setup_environment():
        print("❌ Environment setup failed")
        return
    
    # Test basic pipeline
    if test_basic_pipeline():
        print("\n✅ Basic pipeline test successful!")
    else:
        print("\n❌ Basic pipeline test failed")
        return
    
    # Test enhanced pipeline
    if test_enhanced_pipeline():
        print("\n✅ Enhanced pipeline test successful!")
    else:
        print("\n❌ Enhanced pipeline test failed")
    
    print("\n" + "=" * 50)
    print("🎉 Pipeline Testing Complete!")
    print("=" * 50)
    print()
    print("📋 Next Steps:")
    print("1. Start the enhanced backend:")
    print("   cd backend")
    print("   python enhanced_backend_with_pipeline.py")
    print()
    print("2. Start the frontend:")
    print("   npm start")
    print()
    print("3. Test in the chat interface:")
    print("   - Type 'run pipeline' to start analysis")
    print("   - Type 'market analysis' to view results")
    print("   - Type 'predict AAPL' for stock predictions")
    print()
    print("🔑 Your API key is now configured and ready to use!")

if __name__ == "__main__":
    main()
