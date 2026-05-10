#!/usr/bin/env python3
"""
Example Usage Script for AI Stock Prediction System
==================================================

This script demonstrates various ways to use the AI Stock Prediction System
with different configurations and analysis methods.

Author: AI Assistant
Date: 2024
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from main import AIStockPredictor
from data_collector import StockDataCollector
from lstm_model import LSTMModel
from sensitivity_analysis import SensitivityAnalysis

def example_1_basic_usage():
    """
    Example 1: Basic usage with default parameters
    """
    print("=" * 60)
    print("EXAMPLE 1: BASIC USAGE")
    print("=" * 60)
    
    # Initialize predictor with default settings
    predictor = AIStockPredictor(
        symbol='AAPL',
        start_date='2020-01-01',
        sequence_length=60
    )
    
    # Run complete analysis
    results = predictor.run_complete_analysis(save_model=True)
    
    # Display key results
    print(f"\n📊 RESULTS SUMMARY:")
    print(f"Stock: {predictor.symbol}")
    print(f"R² Score: {results['evaluation_results']['metrics']['R2']:.4f}")
    print(f"Directional Accuracy: {results['evaluation_results']['metrics']['Directional_Accuracy']:.2f}%")
    print(f"Predicted Price Change: {results['prediction_results']['price_change_pct']:+.2f}%")
    
    return results

def example_2_multiple_stocks():
    """
    Example 2: Analyze multiple stocks
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 2: MULTIPLE STOCKS ANALYSIS")
    print("=" * 60)
    
    stocks = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
    results_summary = {}
    
    for symbol in stocks:
        print(f"\n🔍 Analyzing {symbol}...")
        
        try:
            predictor = AIStockPredictor(
                symbol=symbol,
                start_date='2020-01-01',
                sequence_length=60
            )
            
            # Run analysis
            results = predictor.run_complete_analysis(save_model=False)
            
            # Store summary
            results_summary[symbol] = {
                'R2': results['evaluation_results']['metrics']['R2'],
                'Directional_Accuracy': results['evaluation_results']['metrics']['Directional_Accuracy'],
                'Predicted_Change': results['prediction_results']['price_change_pct'],
                'Current_Price': results['prediction_results']['current_price'],
                'Predicted_Price': results['prediction_results']['predicted_price']
            }
            
            print(f"✅ {symbol} analysis completed")
            
        except Exception as e:
            print(f"❌ Error analyzing {symbol}: {str(e)}")
            results_summary[symbol] = None
    
    # Display comparison
    print(f"\n📈 STOCKS COMPARISON:")
    print("-" * 80)
    print(f"{'Stock':<8} {'R²':<8} {'Dir Acc%':<10} {'Pred Chg%':<12} {'Current':<10} {'Predicted':<10}")
    print("-" * 80)
    
    for symbol, data in results_summary.items():
        if data:
            print(f"{symbol:<8} {data['R2']:<8.4f} {data['Directional_Accuracy']:<10.2f} "
                  f"{data['Predicted_Change']:<12.2f} ${data['Current_Price']:<9.2f} ${data['Predicted_Price']:<9.2f}")
    
    return results_summary

def example_3_parameter_optimization():
    """
    Example 3: Parameter optimization for better performance
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 3: PARAMETER OPTIMIZATION")
    print("=" * 60)
    
    # Test different parameter combinations
    param_combinations = [
        {'lstm_units': [64, 32], 'dropout_rate': 0.1, 'learning_rate': 0.001},
        {'lstm_units': [128, 64], 'dropout_rate': 0.2, 'learning_rate': 0.001},
        {'lstm_units': [256, 128, 64], 'dropout_rate': 0.3, 'learning_rate': 0.0005},
        {'lstm_units': [512, 256, 128], 'dropout_rate': 0.4, 'learning_rate': 0.0001}
    ]
    
    best_score = 0
    best_params = None
    best_results = None
    
    for i, params in enumerate(param_combinations):
        print(f"\n🧪 Testing combination {i+1}/{len(param_combinations)}:")
        print(f"LSTM Units: {params['lstm_units']}")
        print(f"Dropout Rate: {params['dropout_rate']}")
        print(f"Learning Rate: {params['learning_rate']}")
        
        try:
            predictor = AIStockPredictor(
                symbol='AAPL',
                start_date='2020-01-01',
                sequence_length=60
            )
            
            # Prepare data
            predictor.collect_and_prepare_data()
            
            # Train with specific parameters
            predictor.build_and_train_model(
                lstm_units=params['lstm_units'],
                dropout_rate=params['dropout_rate'],
                learning_rate=params['learning_rate'],
                epochs=50  # Reduced for faster testing
            )
            
            # Evaluate
            results = predictor.evaluate_model()
            score = results['metrics']['R2']
            
            print(f"R² Score: {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_params = params
                best_results = results
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    print(f"\n🏆 BEST PARAMETERS:")
    print(f"LSTM Units: {best_params['lstm_units']}")
    print(f"Dropout Rate: {best_params['dropout_rate']}")
    print(f"Learning Rate: {best_params['learning_rate']}")
    print(f"Best R² Score: {best_score:.4f}")
    
    return best_params, best_results

def example_4_sensitivity_analysis_demo():
    """
    Example 4: Detailed sensitivity analysis demonstration
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 4: DETAILED SENSITIVITY ANALYSIS")
    print("=" * 60)
    
    # Initialize predictor
    predictor = AIStockPredictor(
        symbol='AAPL',
        start_date='2020-01-01',
        sequence_length=60
    )
    
    # Prepare data and train model
    predictor.collect_and_prepare_data()
    predictor.build_and_train_model(epochs=50)
    predictor.evaluate_model()
    
    # Perform comprehensive sensitivity analysis
    sensitivity_results = predictor.perform_sensitivity_analysis([
        'perturbation', 'correlation', 'gradient'
    ])
    
    # Analyze feature importance
    if 'perturbation' in sensitivity_results:
        results = sensitivity_results['perturbation']
        feature_ranking = results['feature_ranking'][:10]
        overall_sensitivity = results['overall_sensitivity']
        
        print(f"\n🔍 TOP 10 MOST IMPORTANT FEATURES:")
        print("-" * 50)
        for i, feature_idx in enumerate(feature_ranking):
            feature_name = predictor.training_data['feature_names'][feature_idx]
            score = overall_sensitivity[feature_idx]
            print(f"{i+1:2d}. {feature_name}: {score:.4f}")
    
    # Market condition analysis
    if 'market_conditions' in sensitivity_results:
        market_results = sensitivity_results['market_conditions']
        print(f"\n📊 MARKET CONDITION SENSITIVITY:")
        print("-" * 40)
        for scenario, data in market_results.items():
            print(f"{scenario}: {data['percentage_change']:+.2f}% change")
    
    return sensitivity_results

def example_5_real_time_predictions():
    """
    Example 5: Real-time predictions and monitoring
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 5: REAL-TIME PREDICTIONS")
    print("=" * 60)
    
    # Initialize predictor
    predictor = AIStockPredictor(
        symbol='AAPL',
        start_date='2020-01-01',
        sequence_length=60
    )
    
    # Train model
    predictor.collect_and_prepare_data()
    predictor.build_and_train_model(epochs=50)
    predictor.evaluate_model()
    
    # Make predictions for different time horizons
    time_horizons = [1, 3, 5, 10, 15]
    predictions = {}
    
    print(f"\n🔮 PREDICTIONS FOR DIFFERENT TIME HORIZONS:")
    print("-" * 60)
    print(f"{'Days Ahead':<12} {'Predicted Price':<15} {'Change':<12} {'Change %':<10}")
    print("-" * 60)
    
    current_price = predictor.data_collector.raw_data['Close'].iloc[-1]
    
    for days in time_horizons:
        try:
            # Get latest data
            latest_data = predictor.data_collector.get_latest_data(days=predictor.sequence_length)
            
            # Make prediction
            prediction_scaled = predictor.model.predict(latest_data)
            prediction = predictor.training_data['scaler_y'].inverse_transform(prediction_scaled)[0][0]
            
            # Calculate changes
            price_change = prediction - current_price
            price_change_pct = (price_change / current_price) * 100
            
            predictions[days] = {
                'predicted_price': prediction,
                'price_change': price_change,
                'price_change_pct': price_change_pct
            }
            
            print(f"{days:<12} ${prediction:<14.2f} ${price_change:<11.2f} {price_change_pct:<10.2f}%")
            
        except Exception as e:
            print(f"❌ Error predicting {days} days ahead: {str(e)}")
    
    return predictions

def example_6_ensemble_model():
    """
    Example 6: Ensemble model approach
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 6: ENSEMBLE MODEL APPROACH")
    print("=" * 60)
    
    # Initialize predictor
    predictor = AIStockPredictor(
        symbol='AAPL',
        start_date='2020-01-01',
        sequence_length=60
    )
    
    # Prepare data
    predictor.collect_and_prepare_data()
    
    # Train multiple models with different architectures
    models = []
    model_configs = [
        {'lstm_units': [64, 32], 'dropout_rate': 0.1},
        {'lstm_units': [128, 64], 'dropout_rate': 0.2},
        {'lstm_units': [256, 128, 64], 'dropout_rate': 0.3},
        {'lstm_units': [512, 256], 'dropout_rate': 0.4},
        {'lstm_units': [128, 64, 32], 'dropout_rate': 0.25}
    ]
    
    print("🏗️ Training ensemble models...")
    
    for i, config in enumerate(model_configs):
        print(f"Training model {i+1}/{len(model_configs)}...")
        
        try:
            # Create new model instance
            model = LSTMModel(
                sequence_length=predictor.sequence_length,
                n_features=predictor.training_data['X_train'].shape[2],
                model_name=f'ensemble_model_{i+1}'
            )
            
            # Build and train
            model.build_model(
                lstm_units=config['lstm_units'],
                dropout_rate=config['dropout_rate']
            )
            
            model.train(
                predictor.training_data['X_train'],
                predictor.training_data['y_train'],
                epochs=30,  # Reduced for faster training
                validation_split=0.2
            )
            
            models.append(model)
            print(f"✅ Model {i+1} trained successfully")
            
        except Exception as e:
            print(f"❌ Error training model {i+1}: {str(e)}")
    
    # Make ensemble predictions
    if models:
        print(f"\n🔮 Making ensemble predictions...")
        
        # Individual predictions
        individual_predictions = []
        for i, model in enumerate(models):
            pred = model.predict(predictor.training_data['X_test'])
            individual_predictions.append(pred.flatten())
        
        # Ensemble prediction (average)
        ensemble_pred = np.mean(individual_predictions, axis=0)
        
        # Evaluate ensemble
        from sklearn.metrics import mean_squared_error, r2_score
        
        y_test_original = predictor.training_data['scaler_y'].inverse_transform(
            predictor.training_data['y_test'].reshape(-1, 1)
        ).flatten()
        
        ensemble_pred_original = predictor.training_data['scaler_y'].inverse_transform(
            ensemble_pred.reshape(-1, 1)
        ).flatten()
        
        ensemble_mse = mean_squared_error(y_test_original, ensemble_pred_original)
        ensemble_r2 = r2_score(y_test_original, ensemble_pred_original)
        
        print(f"Ensemble Performance:")
        print(f"MSE: {ensemble_mse:.4f}")
        print(f"R² Score: {ensemble_r2:.4f}")
        
        # Compare with individual models
        print(f"\n📊 MODEL COMPARISON:")
        print("-" * 40)
        for i, pred in enumerate(individual_predictions):
            pred_original = predictor.training_data['scaler_y'].inverse_transform(
                pred.reshape(-1, 1)
            ).flatten()
            mse = mean_squared_error(y_test_original, pred_original)
            r2 = r2_score(y_test_original, pred_original)
            print(f"Model {i+1}: MSE={mse:.4f}, R²={r2:.4f}")
        
        return {
            'ensemble_prediction': ensemble_pred_original,
            'individual_predictions': individual_predictions,
            'ensemble_metrics': {'MSE': ensemble_mse, 'R2': ensemble_r2}
        }
    
    return None

def example_7_custom_analysis():
    """
    Example 7: Custom analysis with specific requirements
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 7: CUSTOM ANALYSIS")
    print("=" * 60)
    
    # Custom configuration
    symbol = 'TSLA'
    start_date = '2019-01-01'  # Longer history for Tesla
    sequence_length = 90  # Longer sequence for more complex patterns
    
    print(f"Analyzing {symbol} with custom parameters...")
    print(f"Data period: {start_date} to present")
    print(f"Sequence length: {sequence_length} days")
    
    # Initialize predictor with custom settings
    predictor = AIStockPredictor(
        symbol=symbol,
        start_date=start_date,
        sequence_length=sequence_length
    )
    
    # Custom data preparation
    predictor.collect_and_prepare_data(test_size=0.15)  # More training data
    
    # Custom model with specific architecture
    predictor.build_and_train_model(
        lstm_units=[512, 256, 128, 64],  # Deeper network
        dropout_rate=0.3,
        learning_rate=0.0005,
        epochs=150,
        batch_size=16
    )
    
    # Evaluate
    results = predictor.evaluate_model()
    
    # Custom sensitivity analysis
    print(f"\n🔍 Performing custom sensitivity analysis...")
    sensitivity_results = predictor.perform_sensitivity_analysis([
        'perturbation', 'correlation'
    ])
    
    # Custom predictions
    predictions = predictor.make_predictions(days_ahead=7)
    
    # Generate custom report
    predictor.generate_comprehensive_report(output_dir=f'reports/{symbol}_custom')
    
    print(f"\n✅ Custom analysis completed for {symbol}")
    print(f"Results saved to reports/{symbol}_custom/")
    
    return {
        'evaluation': results,
        'sensitivity': sensitivity_results,
        'predictions': predictions
    }

def main():
    """
    Main function to run all examples
    """
    print("🚀 AI STOCK PREDICTION SYSTEM - EXAMPLE USAGE")
    print("=" * 60)
    
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    # Run examples
    examples = [
        ("Basic Usage", example_1_basic_usage),
        ("Multiple Stocks", example_2_multiple_stocks),
        ("Parameter Optimization", example_3_parameter_optimization),
        ("Sensitivity Analysis", example_4_sensitivity_analysis_demo),
        ("Real-time Predictions", example_5_real_time_predictions),
        ("Ensemble Model", example_6_ensemble_model),
        ("Custom Analysis", example_7_custom_analysis)
    ]
    
    results = {}
    
    for name, func in examples:
        try:
            print(f"\n{'='*20} RUNNING {name.upper()} {'='*20}")
            results[name] = func()
            print(f"✅ {name} completed successfully")
        except Exception as e:
            print(f"❌ Error in {name}: {str(e)}")
            results[name] = None
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 EXAMPLE EXECUTION SUMMARY")
    print("=" * 60)
    
    for name, result in results.items():
        status = "✅ Completed" if result is not None else "❌ Failed"
        print(f"{name:<25} {status}")
    
    print(f"\n🎯 NEXT STEPS:")
    print("1. Check the 'models' directory for saved models")
    print("2. Review the 'reports' directory for detailed analysis")
    print("3. Modify parameters in main.py for your specific needs")
    print("4. Experiment with different stocks and time periods")

if __name__ == "__main__":
    main()
