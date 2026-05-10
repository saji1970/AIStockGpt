#!/usr/bin/env python3
"""
AI Stock Prediction System using LSTM and Sensitivity Analysis
=============================================================

This script demonstrates a complete AI system for stock price prediction using:
1. LSTM (Long Short-Term Memory) neural networks
2. Comprehensive sensitivity analysis
3. Technical indicators and feature engineering
4. Model evaluation and visualization

Author: AI Assistant
Date: 2024
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from data_collector import StockDataCollector
from lstm_model import LSTMModel
from sensitivity_analysis import SensitivityAnalysis

# Cloud Functions entry point for AI Stock GPT Backend
import functions_framework
from enhanced_backend_with_pipeline import app

# Set environment variables
os.environ['PARALLEL_API_KEY'] = "VkvkaHQWl8twbuEhWsgagDp2sNYNdIOuIQMC1NqI"

@functions_framework.http
def ai_stock_gpt(request):
    """HTTP Cloud Function for AI Stock GPT Backend"""
    return app(request.environ, lambda x, y: y)

class AIStockPredictor:
    """
    Main class for the AI Stock Prediction System.
    """
    
    def __init__(self, symbol='AAPL', start_date='2020-01-01', sequence_length=60):
        """
        Initialize the AI Stock Predictor.
        
        Args:
            symbol (str): Stock symbol to predict
            start_date (str): Start date for data collection
            sequence_length (int): Number of time steps for LSTM
        """
        self.symbol = symbol
        self.start_date = start_date
        self.sequence_length = sequence_length
        
        # Initialize components
        self.data_collector = None
        self.model = None
        self.sensitivity_analyzer = None
        
        # Results storage
        self.training_results = {}
        self.prediction_results = {}
        self.sensitivity_results = {}
        
        print(f"AI Stock Predictor initialized for {symbol}")
        print(f"Data period: {start_date} to present")
        print(f"Sequence length: {sequence_length} days")
    
    def collect_and_prepare_data(self, test_size=0.2):
        """
        Collect and prepare data for training.
        
        Args:
            test_size (float): Proportion of data for testing
            
        Returns:
            tuple: Prepared data (X_train, X_test, y_train, y_test, scalers, feature_names)
        """
        print("\n" + "="*50)
        print("STEP 1: DATA COLLECTION AND PREPARATION")
        print("="*50)
        
        # Initialize data collector
        self.data_collector = StockDataCollector(
            symbol=self.symbol,
            start_date=self.start_date
        )
        
        # Fetch data
        print(f"Fetching data for {self.symbol}...")
        raw_data = self.data_collector.fetch_data()
        
        if raw_data is None:
            raise ValueError(f"Failed to fetch data for {self.symbol}")
        
        print(f"Raw data shape: {raw_data.shape}")
        print(f"Date range: {raw_data.index[0].date()} to {raw_data.index[-1].date()}")
        
        # Prepare data for LSTM
        print("Preparing data for LSTM model...")
        data_tuple = self.data_collector.prepare_data(
            target_column='Close',
            sequence_length=self.sequence_length,
            test_size=test_size
        )
        
        X_train, X_test, y_train, y_test, scaler_X, scaler_y, feature_names = data_tuple
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Testing data shape: {X_test.shape}")
        print(f"Number of features: {len(feature_names)}")
        
        # Store data
        self.training_data = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'scaler_X': scaler_X,
            'scaler_y': scaler_y,
            'feature_names': feature_names
        }
        
        return data_tuple
    
    def build_and_train_model(self, lstm_units=[128, 64], dropout_rate=0.2, 
                            learning_rate=0.001, epochs=100, batch_size=32):
        """
        Build and train the LSTM model.
        
        Args:
            lstm_units (list): LSTM layer units
            dropout_rate (float): Dropout rate
            learning_rate (float): Learning rate
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            
        Returns:
            dict: Training results
        """
        print("\n" + "="*50)
        print("STEP 2: MODEL BUILDING AND TRAINING")
        print("="*50)
        
        if not hasattr(self, 'training_data'):
            raise ValueError("Data must be prepared first. Call collect_and_prepare_data()")
        
        # Initialize model
        n_features = self.training_data['X_train'].shape[2]
        self.model = LSTMModel(
            sequence_length=self.sequence_length,
            n_features=n_features,
            model_name=f'{self.symbol}_lstm'
        )
        
        print(f"Building LSTM model with {n_features} features...")
        print(f"LSTM units: {lstm_units}")
        print(f"Dropout rate: {dropout_rate}")
        print(f"Learning rate: {learning_rate}")
        
        # Build model
        self.model.build_model(
            lstm_units=lstm_units,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate
        )
        
        # Display model summary
        print("\nModel Architecture:")
        self.model.model.summary()
        
        # Train model
        print(f"\nTraining model for {epochs} epochs...")
        history = self.model.train(
            X_train=self.training_data['X_train'],
            y_train=self.training_data['y_train'],
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2
        )
        
        # Store training results
        self.training_results = {
            'history': history,
            'model_params': {
                'lstm_units': lstm_units,
                'dropout_rate': dropout_rate,
                'learning_rate': learning_rate,
                'epochs': epochs,
                'batch_size': batch_size
            }
        }
        
        print("Model training completed!")
        return self.training_results
    
    def evaluate_model(self):
        """
        Evaluate the trained model.
        
        Returns:
            dict: Evaluation metrics
        """
        print("\n" + "="*50)
        print("STEP 3: MODEL EVALUATION")
        print("="*50)
        
        if self.model is None:
            raise ValueError("Model must be trained first. Call build_and_train_model()")
        
        # Evaluate model
        metrics, y_pred = self.model.evaluate(
            X_test=self.training_data['X_test'],
            y_test=self.training_data['y_test'],
            scaler_y=self.training_data['scaler_y']
        )
        
        # Store evaluation results
        self.evaluation_results = {
            'metrics': metrics,
            'predictions': y_pred,
            'actual': self.training_data['scaler_y'].inverse_transform(
                self.training_data['y_test'].reshape(-1, 1)
            ).flatten()
        }
        
        # Print results
        print("Model Performance Metrics:")
        print("-" * 30)
        for metric, value in metrics.items():
            if metric in ['MSE', 'RMSE', 'MAE']:
                print(f"{metric}: {value:.4f}")
            elif metric == 'R2':
                print(f"{metric}: {value:.4f}")
            elif metric == 'MAPE':
                print(f"{metric}: {value:.2f}%")
            elif metric == 'Directional_Accuracy':
                print(f"{metric}: {value:.2f}%")
        
        return self.evaluation_results
    
    def perform_sensitivity_analysis(self, analysis_methods=['perturbation', 'correlation']):
        """
        Perform comprehensive sensitivity analysis.
        
        Args:
            analysis_methods (list): Methods to use for sensitivity analysis
            
        Returns:
            dict: Sensitivity analysis results
        """
        print("\n" + "="*50)
        print("STEP 4: SENSITIVITY ANALYSIS")
        print("="*50)
        
        if self.model is None:
            raise ValueError("Model must be trained first. Call build_and_train_model()")
        
        # Initialize sensitivity analyzer
        self.sensitivity_analyzer = SensitivityAnalysis(
            model=self.model,
            scaler_X=self.training_data['scaler_X'],
            scaler_y=self.training_data['scaler_y'],
            feature_names=self.training_data['feature_names']
        )
        
        # Use a sample of test data for sensitivity analysis
        X_sample = self.training_data['X_test'][:100]  # Use first 100 samples
        y_sample = self.training_data['y_test'][:100]
        
        sensitivity_results = {}
        
        # Perform different types of sensitivity analysis
        for method in analysis_methods:
            print(f"Performing {method} sensitivity analysis...")
            
            if method == 'perturbation':
                results = self.sensitivity_analyzer.feature_sensitivity_analysis(
                    X_sample, method='perturbation'
                )
            elif method == 'correlation':
                results = self.sensitivity_analyzer.feature_sensitivity_analysis(
                    X_sample, y_sample, method='correlation'
                )
            elif method == 'gradient':
                results = self.sensitivity_analyzer.feature_sensitivity_analysis(
                    X_sample, method='gradient'
                )
            
            sensitivity_results[method] = results
            
            # Plot results
            self.sensitivity_analyzer.plot_sensitivity_results(method, top_n=10)
        
        # Market condition sensitivity analysis
        print("Performing market condition sensitivity analysis...")
        market_scenarios = {
            'High_Volatility': {
                0: {'type': 'noise', 'value': 0.5},  # Price volatility
                4: {'type': 'multiply', 'value': 1.5}  # Volume increase
            },
            'Low_Volatility': {
                0: {'type': 'noise', 'value': 0.1},  # Price stability
                4: {'type': 'multiply', 'value': 0.5}  # Volume decrease
            },
            'Trend_Up': {
                0: {'type': 'add', 'value': 0.1},  # Price increase
                1: {'type': 'add', 'value': 0.1},  # High increase
                2: {'type': 'add', 'value': 0.1}   # Low increase
            },
            'Trend_Down': {
                0: {'type': 'add', 'value': -0.1},  # Price decrease
                1: {'type': 'add', 'value': -0.1},  # High decrease
                2: {'type': 'add', 'value': -0.1}   # Low decrease
            }
        }
        
        market_results = self.sensitivity_analyzer.market_condition_sensitivity(
            X_sample, market_scenarios
        )
        sensitivity_results['market_conditions'] = market_results
        
        # Plot market condition results
        self.sensitivity_analyzer.plot_sensitivity_results('market_conditions')
        
        # Store results
        self.sensitivity_results = sensitivity_results
        
        return sensitivity_results
    
    def make_predictions(self, days_ahead=5):
        """
        Make future predictions.
        
        Args:
            days_ahead (int): Number of days to predict ahead
            
        Returns:
            dict: Prediction results
        """
        print("\n" + "="*50)
        print("STEP 5: FUTURE PREDICTIONS")
        print("="*50)
        
        if self.model is None:
            raise ValueError("Model must be trained first. Call build_and_train_model()")
        
        # Get latest data for prediction
        latest_data = self.data_collector.get_latest_data(days=self.sequence_length)
        
        # Make prediction
        prediction_scaled = self.model.predict(latest_data)
        prediction = self.training_data['scaler_y'].inverse_transform(prediction_scaled)[0][0]
        
        # Get current price
        current_price = self.data_collector.raw_data['Close'].iloc[-1]
        
        # Calculate change
        price_change = prediction - current_price
        price_change_pct = (price_change / current_price) * 100
        
        # Store prediction results
        self.prediction_results = {
            'current_price': current_price,
            'predicted_price': prediction,
            'price_change': price_change,
            'price_change_pct': price_change_pct,
            'prediction_date': datetime.now().strftime('%Y-%m-%d'),
            'days_ahead': days_ahead
        }
        
        # Print results
        print(f"Current Price ({self.symbol}): ${current_price:.2f}")
        print(f"Predicted Price ({days_ahead} days ahead): ${prediction:.2f}")
        print(f"Price Change: ${price_change:.2f} ({price_change_pct:+.2f}%)")
        
        if price_change > 0:
            print("Prediction: BULLISH 📈")
        else:
            print("Prediction: BEARISH 📉")
        
        return self.prediction_results
    
    def generate_comprehensive_report(self, output_dir='reports'):
        """
        Generate a comprehensive report of all analyses.
        
        Args:
            output_dir (str): Directory to save reports
        """
        print("\n" + "="*50)
        print("STEP 6: GENERATING COMPREHENSIVE REPORT")
        print("="*50)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate plots
        self._generate_plots(output_dir)
        
        # Generate sensitivity report
        if hasattr(self, 'sensitivity_analyzer'):
            sensitivity_report_path = os.path.join(output_dir, 'sensitivity_report.html')
            self.sensitivity_analyzer.generate_sensitivity_report(sensitivity_report_path)
        
        # Generate summary report
        self._generate_summary_report(output_dir)
        
        print(f"All reports saved to '{output_dir}' directory")
    
    def _generate_plots(self, output_dir):
        """Generate and save all plots."""
        # Training history
        if hasattr(self, 'training_results') and self.training_results.get('history'):
            plt.figure(figsize=(15, 5))
            self.model.plot_training_history()
            plt.savefig(os.path.join(output_dir, 'training_history.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        # Predictions vs actual
        if hasattr(self, 'evaluation_results'):
            plt.figure(figsize=(12, 6))
            self.model.plot_predictions(
                self.evaluation_results['actual'],
                self.evaluation_results['predictions'],
                title=f'{self.symbol} Stock Price Predictions'
            )
            plt.savefig(os.path.join(output_dir, 'predictions_vs_actual.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        # Feature importance
        if hasattr(self, 'sensitivity_results') and 'perturbation' in self.sensitivity_results:
            self.sensitivity_analyzer.plot_sensitivity_results('perturbation', top_n=15)
            plt.savefig(os.path.join(output_dir, 'feature_importance.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def _generate_summary_report(self, output_dir):
        """Generate a summary report."""
        report_path = os.path.join(output_dir, 'summary_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("AI STOCK PREDICTION SYSTEM - SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Stock Symbol: {self.symbol}\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data Period: {self.start_date} to present\n")
            f.write(f"Sequence Length: {self.sequence_length} days\n\n")
            
            # Model performance
            if hasattr(self, 'evaluation_results'):
                f.write("MODEL PERFORMANCE:\n")
                f.write("-" * 20 + "\n")
                for metric, value in self.evaluation_results['metrics'].items():
                    if metric in ['MSE', 'RMSE', 'MAE']:
                        f.write(f"{metric}: {value:.4f}\n")
                    elif metric == 'R2':
                        f.write(f"{metric}: {value:.4f}\n")
                    elif metric in ['MAPE', 'Directional_Accuracy']:
                        f.write(f"{metric}: {value:.2f}%\n")
                f.write("\n")
            
            # Predictions
            if hasattr(self, 'prediction_results'):
                f.write("PREDICTIONS:\n")
                f.write("-" * 12 + "\n")
                f.write(f"Current Price: ${self.prediction_results['current_price']:.2f}\n")
                f.write(f"Predicted Price: ${self.prediction_results['predicted_price']:.2f}\n")
                f.write(f"Price Change: ${self.prediction_results['price_change']:.2f} "
                       f"({self.prediction_results['price_change_pct']:+.2f}%)\n")
                f.write(f"Prediction: {'BULLISH' if self.prediction_results['price_change'] > 0 else 'BEARISH'}\n\n")
            
            # Top features
            if hasattr(self, 'sensitivity_results') and 'perturbation' in self.sensitivity_results:
                f.write("TOP 10 MOST IMPORTANT FEATURES:\n")
                f.write("-" * 35 + "\n")
                results = self.sensitivity_results['perturbation']
                feature_ranking = results['feature_ranking'][:10]
                overall_sensitivity = results['overall_sensitivity']
                
                for i, feature_idx in enumerate(feature_ranking):
                    feature_name = self.training_data['feature_names'][feature_idx]
                    score = overall_sensitivity[feature_idx]
                    f.write(f"{i+1:2d}. {feature_name}: {score:.4f}\n")
                f.write("\n")
        
        print(f"Summary report saved to {report_path}")
    
    def run_complete_analysis(self, save_model=True):
        """
        Run the complete analysis pipeline.
        
        Args:
            save_model (bool): Whether to save the trained model
            
        Returns:
            dict: Complete analysis results
        """
        print("🚀 STARTING COMPLETE AI STOCK PREDICTION ANALYSIS")
        print("=" * 60)
        
        try:
            # Step 1: Data collection and preparation
            self.collect_and_prepare_data()
            
            # Step 2: Model building and training
            self.build_and_train_model()
            
            # Step 3: Model evaluation
            self.evaluate_model()
            
            # Step 4: Sensitivity analysis
            self.perform_sensitivity_analysis()
            
            # Step 5: Future predictions
            self.make_predictions()
            
            # Step 6: Generate reports
            self.generate_comprehensive_report()
            
            # Save model if requested
            if save_model:
                model_path = f'models/{self.symbol}_lstm_complete'
                self.model.save_model(
                    model_path=model_path,
                    scaler_X=self.training_data['scaler_X'],
                    scaler_y=self.training_data['scaler_y'],
                    feature_names=self.training_data['feature_names']
                )
            
            print("\n" + "=" * 60)
            print("✅ ANALYSIS COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            
            return {
                'training_results': self.training_results,
                'evaluation_results': self.evaluation_results,
                'sensitivity_results': self.sensitivity_results,
                'prediction_results': self.prediction_results
            }
            
        except Exception as e:
            print(f"\n❌ ERROR DURING ANALYSIS: {str(e)}")
            raise

def main():
    """
    Main function to demonstrate the AI Stock Prediction System.
    """
    print("AI Stock Prediction System using LSTM and Sensitivity Analysis")
    print("=" * 60)
    
    # Configuration
    symbol = 'AAPL'  # Change this to any stock symbol
    start_date = '2020-01-01'
    sequence_length = 60
    
    # Create predictor
    predictor = AIStockPredictor(
        symbol=symbol,
        start_date=start_date,
        sequence_length=sequence_length
    )
    
    # Run complete analysis
    results = predictor.run_complete_analysis(save_model=True)
    
    print("\n📊 ANALYSIS SUMMARY:")
    print(f"Stock: {symbol}")
    print(f"R² Score: {results['evaluation_results']['metrics']['R2']:.4f}")
    print(f"Directional Accuracy: {results['evaluation_results']['metrics']['Directional_Accuracy']:.2f}%")
    print(f"Predicted Price Change: {results['prediction_results']['price_change_pct']:+.2f}%")
    
    print("\n🎯 NEXT STEPS:")
    print("1. Check the 'reports' directory for detailed analysis")
    print("2. Review the sensitivity analysis to understand feature importance")
    print("3. Use the saved model for future predictions")
    print("4. Consider retraining with different parameters or more data")

if __name__ == "__main__":
    main()
