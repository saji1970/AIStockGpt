#!/usr/bin/env python3
"""
AI Stock Prediction System - Demo
=================================

A simple demonstration of the professional AI stock prediction system.
This script shows the basic functionality without running the full analysis.

Author: AI Assistant
Date: 2024
"""

import sys
import warnings
warnings.filterwarnings('ignore')

def main():
    """Run a simple demonstration of the AI Stock Prediction System."""
    
    print("🚀 AI STOCK PREDICTION SYSTEM - DEMO")
    print("=" * 50)
    print("Professional LSTM-based stock prediction with sensitivity analysis")
    print("=" * 50)
    
    try:
        # Import our modules
        from data_collector import StockDataCollector
        from lstm_model import LSTMModel
        from sensitivity_analysis import SensitivityAnalysis
        from main import AIStockPredictor
        
        print("\n✅ All modules imported successfully!")
        
        # Test data collection
        print("\n📊 Testing Data Collection...")
        collector = StockDataCollector(symbol='AAPL', start_date='2023-01-01')
        data = collector.fetch_data()
        print(f"   ✅ Collected {len(data)} days of data for AAPL")
        print(f"   📅 Date range: {data.index[0].date()} to {data.index[-1].date()}")
        
        # Test data preparation
        print("\n🔧 Testing Data Preparation...")
        data_tuple = collector.prepare_data(sequence_length=30, test_size=0.2)
        X_train, X_test, y_train, y_test, scaler_X, scaler_y, features = data_tuple
        print(f"   ✅ Prepared {X_train.shape[0]} training samples")
        print(f"   ✅ Prepared {X_test.shape[0]} test samples")
        print(f"   🔢 Using {len(features)} features")
        
        # Test model creation
        print("\n🧠 Testing Model Creation...")
        model = LSTMModel(sequence_length=30, n_features=len(features))
        model.build_model(lstm_units=[64, 32], dropout_rate=0.2)
        
        # Build the model with dummy data
        import numpy as np
        dummy_data = np.random.random((1, 30, len(features)))
        model.model(dummy_data)
        
        print(f"   ✅ LSTM model created with {model.model.count_params()} parameters")
        
        # Test sensitivity analysis
        print("\n📈 Testing Sensitivity Analysis...")
        analyzer = SensitivityAnalysis(model=model, feature_names=features)
        print("   ✅ Sensitivity analyzer created")
        
        # Test AI Stock Predictor
        print("\n🤖 Testing AI Stock Predictor...")
        predictor = AIStockPredictor(symbol='AAPL', start_date='2023-01-01', sequence_length=30)
        print("   ✅ AI Stock Predictor initialized")
        
        print("\n" + "=" * 50)
        print("🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print("\nThe AI Stock Prediction System is ready for professional use!")
        print("\nNext steps:")
        print("1. Run 'python main.py' for full analysis")
        print("2. Run 'python example_usage.py' for various examples")
        print("3. Check 'README.md' for detailed documentation")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demo failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
