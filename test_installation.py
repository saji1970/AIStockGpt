#!/usr/bin/env python3
"""
Test Installation Script for AI Stock Prediction System
======================================================

This script tests the installation and basic functionality of the AI Stock Prediction System.
Run this script to verify that all dependencies are installed correctly.

Author: AI Assistant
Date: 2024
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

def test_imports():
    """Test if all required packages can be imported."""
    print("🔍 Testing package imports...")
    
    required_packages = [
        'numpy',
        'pandas', 
        'matplotlib',
        'seaborn',
        'sklearn',
        'tensorflow',
        'keras',
        'yfinance',
        'plotly',
        'dash',
        'ta',
        'scipy',
        'joblib'
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package}: {str(e)}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        print("Please install missing packages using: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All packages imported successfully!")
        return True

def test_custom_modules():
    """Test if our custom modules can be imported."""
    print("\n🔍 Testing custom module imports...")
    
    custom_modules = [
        'data_collector',
        'lstm_model', 
        'sensitivity_analysis',
        'main'
    ]
    
    failed_imports = []
    
    for module in custom_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {str(e)}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n❌ Failed to import custom modules: {', '.join(failed_imports)}")
        return False
    else:
        print("\n✅ All custom modules imported successfully!")
        return True

def test_data_collection():
    """Test basic data collection functionality."""
    print("\n🔍 Testing data collection...")
    
    try:
        from data_collector import StockDataCollector
        
        # Test data collection
        collector = StockDataCollector(symbol='AAPL', start_date='2023-01-01')
        data = collector.fetch_data()
        
        if data is not None and len(data) > 0:
            print(f"✅ Data collection successful: {len(data)} records")
            print(f"   Date range: {data.index[0].date()} to {data.index[-1].date()}")
            print(f"   Columns: {list(data.columns)}")
            return True
        else:
            print("❌ Data collection failed: No data retrieved")
            return False
            
    except Exception as e:
        print(f"❌ Data collection error: {str(e)}")
        return False

def test_lstm_model():
    """Test LSTM model creation."""
    print("\n🔍 Testing LSTM model creation...")
    
    try:
        from lstm_model import LSTMModel
        
        # Create a simple model
        model = LSTMModel(sequence_length=10, n_features=5)
        model.build_model(lstm_units=[32, 16], dropout_rate=0.2)
        
        # Create dummy data to build the model
        import numpy as np
        dummy_data = np.random.random((1, 10, 5))
        model.model(dummy_data)  # This builds the model
        
        print("✅ LSTM model created successfully")
        print(f"   Model summary: {model.model.count_params()} parameters")
        return True
        
    except Exception as e:
        print(f"❌ LSTM model error: {str(e)}")
        return False

def test_sensitivity_analysis():
    """Test sensitivity analysis functionality."""
    print("\n🔍 Testing sensitivity analysis...")
    
    try:
        from sensitivity_analysis import SensitivityAnalysis
        from lstm_model import LSTMModel
        
        # Create a simple model for testing
        model = LSTMModel(sequence_length=10, n_features=5)
        model.build_model(lstm_units=[32], dropout_rate=0.2)
        
        # Create sensitivity analyzer
        analyzer = SensitivityAnalysis(model=model, feature_names=['f1', 'f2', 'f3', 'f4', 'f5'])
        
        print("✅ Sensitivity analysis module created successfully")
        return True
        
    except Exception as e:
        print(f"❌ Sensitivity analysis error: {str(e)}")
        return False

def test_basic_prediction():
    """Test basic prediction functionality."""
    print("\n🔍 Testing basic prediction system...")
    
    try:
        from main import AIStockPredictor
        
        # Create predictor (this will test the integration)
        predictor = AIStockPredictor(
            symbol='AAPL',
            start_date='2023-01-01',
            sequence_length=30  # Shorter for faster testing
        )
        
        print("✅ AI Stock Predictor created successfully")
        return True
        
    except Exception as e:
        print(f"❌ Basic prediction error: {str(e)}")
        return False

def test_tensorflow_gpu():
    """Test TensorFlow GPU availability."""
    print("\n🔍 Testing TensorFlow GPU...")
    
    try:
        import tensorflow as tf
        
        # Check TensorFlow version
        print(f"   TensorFlow version: {tf.__version__}")
        
        # Check for GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"   GPU devices found: {len(gpus)}")
            for gpu in gpus:
                print(f"     - {gpu.name}")
        else:
            print("   No GPU devices found (will use CPU)")
        
        # Test basic TensorFlow operation
        a = tf.constant([[1, 2], [3, 4]])
        b = tf.constant([[5, 6], [7, 8]])
        c = tf.matmul(a, b)
        print(f"   Basic TensorFlow operation successful: {c.numpy()}")
        
        return True
        
    except Exception as e:
        print(f"❌ TensorFlow error: {str(e)}")
        return False

def run_quick_test():
    """Run a quick end-to-end test."""
    print("\n🔍 Running quick end-to-end test...")
    
    try:
        from main import AIStockPredictor
        
        # Create predictor with minimal settings
        predictor = AIStockPredictor(
            symbol='AAPL',
            start_date='2023-01-01',
            sequence_length=30
        )
        
        # Test data collection
        print("   Testing data collection...")
        data_tuple = predictor.collect_and_prepare_data(test_size=0.3)
        X_train, X_test, y_train, y_test, scaler_X, scaler_y, feature_names = data_tuple
        
        print(f"   Data prepared: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
        print(f"   Features: {len(feature_names)}")
        
        # Test model creation and training (minimal)
        print("   Testing model creation and training...")
        predictor.build_and_train_model(lstm_units=[32, 16], dropout_rate=0.2, epochs=5, batch_size=16)
        
        print("   Testing model evaluation...")
        metrics, predictions = predictor.evaluate_model()
        
        print(f"   Model performance: R² = {metrics['R2']:.4f}")
        
        print("✅ Quick end-to-end test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Quick test error: {str(e)}")
        return False

def main():
    """Main test function."""
    print("🚀 AI STOCK PREDICTION SYSTEM - INSTALLATION TEST")
    print("=" * 60)
    
    # Run all tests
    tests = [
        ("Package Imports", test_imports),
        ("Custom Modules", test_custom_modules),
        ("Data Collection", test_data_collection),
        ("LSTM Model", test_lstm_model),
        ("Sensitivity Analysis", test_sensitivity_analysis),
        ("Basic Prediction", test_basic_prediction),
        ("TensorFlow GPU", test_tensorflow_gpu),
        ("Quick End-to-End Test", run_quick_test)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {str(e)}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Your installation is ready to use.")
        print("\nNext steps:")
        print("1. Run 'python main.py' to start the full analysis")
        print("2. Run 'python example_usage.py' to see various examples")
        print("3. Check the README.md for detailed usage instructions")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
        print("\nTroubleshooting:")
        print("1. Make sure all dependencies are installed: pip install -r requirements.txt")
        print("2. Check your Python version (3.8+ recommended)")
        print("3. For GPU issues, check TensorFlow installation")
        print("4. For data collection issues, check internet connection")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
