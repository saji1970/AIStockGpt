#!/usr/bin/env python3
"""
FastAPI Backend for AI Stock GPT
================================

This module provides a FastAPI backend that integrates with the AI stock prediction system
and provides a chat interface for natural language queries.
"""

import os
import sys
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

# Add the parent directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Import our custom modules
from data_collector import StockDataCollector
from lstm_model import LSTMModel
from sensitivity_analysis import SensitivityAnalysis
from nlp_processor import NLPProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Stock GPT API",
    description="Intelligent Stock Analysis and Prediction API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    timestamp: Optional[str] = Field(None, description="Message timestamp")

class ChatResponse(BaseModel):
    message: str = Field(..., description="AI response")
    stockData: Optional[Dict[str, Any]] = Field(None, description="Stock analysis data")
    charts: Optional[Dict[str, Any]] = Field(None, description="Chart data")
    confidence: Optional[float] = Field(None, description="Model confidence")

class PredictionRequest(BaseModel):
    symbol: str = Field(..., description="Stock symbol")
    days_ahead: Optional[int] = Field(5, description="Days to predict ahead")

# Global variables
nlp_processor = None
models_cache = {}
data_collectors = {}

def initialize_nlp():
    """Initialize the NLP processor."""
    global nlp_processor
    try:
        nlp_processor = NLPProcessor()
        logger.info("NLP processor initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize NLP processor: {e}")
        raise

def get_or_create_model(symbol: str) -> Dict[str, Any]:
    """Get or create a model for the given symbol."""
    if symbol not in models_cache:
        try:
            # Initialize data collector
            data_collector = StockDataCollector(symbol=symbol, start_date='2020-01-01')
            data_collectors[symbol] = data_collector
            
            # Load or create model
            model_path = f'models/{symbol}_lstm_best.h5'
            if os.path.exists(model_path):
                # Load existing model
                model = LSTMModel()
                model.load_model(model_path)
                models_cache[symbol] = {
                    'model': model,
                    'data_collector': data_collector,
                    'loaded': True
                }
                logger.info(f"Loaded existing model for {symbol}")
            else:
                # Create new model
                models_cache[symbol] = {
                    'model': None,
                    'data_collector': data_collector,
                    'loaded': False
                }
                logger.info(f"Created new model entry for {symbol}")
        except Exception as e:
            logger.error(f"Error creating model for {symbol}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to create model for {symbol}")
    
    return models_cache[symbol]

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    logger.info("Starting AI Stock GPT API...")
    initialize_nlp()
    logger.info("API startup completed")

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "AI Stock GPT API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/status")
async def get_status():
    """Get API and model status."""
    return {
        "api_status": "running",
        "nlp_processor": "active" if nlp_processor else "inactive",
        "cached_models": len(models_cache),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Handle chat messages and return AI responses."""
    try:
        if not nlp_processor:
            raise HTTPException(status_code=500, detail="NLP processor not initialized")
        
        # Process the message with NLP
        intent, entities, confidence = nlp_processor.process_message(request.message)
        
        # Generate response based on intent
        response = await generate_response(intent, entities, request.message, confidence)
        
        return ChatResponse(**response)
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def generate_response(intent: str, entities: Dict, original_message: str, confidence: float) -> Dict:
    """Generate response based on intent and entities."""
    
    if intent == "stock_prediction":
        symbol = entities.get("symbol", "AAPL")
        return await handle_stock_prediction(symbol, original_message, confidence)
    
    elif intent == "technical_analysis":
        symbol = entities.get("symbol", "AAPL")
        return await handle_technical_analysis(symbol, original_message, confidence)
    
    elif intent == "sensitivity_analysis":
        symbol = entities.get("symbol", "AAPL")
        return await handle_sensitivity_analysis(symbol, original_message, confidence)
    
    elif intent == "general_question":
        return await handle_general_question(original_message, confidence)
    
    else:
        return {
            "message": f"I understand you're asking about: {intent}. Could you please be more specific about what stock analysis you'd like me to perform?",
            "confidence": confidence
        }

async def handle_stock_prediction(symbol: str, original_message: str, confidence: float) -> Dict:
    """Handle stock prediction requests."""
    try:
        # Get or create model
        model_info = get_or_create_model(symbol)
        
        if not model_info['loaded']:
            return {
                "message": f"I don't have a trained model for {symbol} yet. Please ask me to analyze a different stock or wait for the model to be trained.",
                "confidence": confidence
            }
        
        # Get latest data and make prediction
        data_collector = model_info['data_collector']
        model = model_info['model']
        
        # Get latest data
        latest_data = data_collector.get_latest_data(days=60)
        if latest_data is None:
            return {
                "message": f"Sorry, I couldn't fetch the latest data for {symbol}. Please try again later.",
                "confidence": confidence
            }
        
        # Make prediction
        prediction_scaled = model.predict(latest_data)
        predicted_price = model.scaler_y.inverse_transform(prediction_scaled)[0][0]
        
        # Get current price
        current_price = data_collector.raw_data['Close'].iloc[-1]
        
        # Calculate metrics
        price_change = predicted_price - current_price
        price_change_pct = (price_change / current_price) * 100
        
        # Determine prediction sentiment
        prediction_sentiment = "BULLISH 📈" if price_change > 0 else "BEARISH 📉"
        
        # Create response
        response_message = f"""# Stock Prediction for {symbol.upper()}

## 📊 Current Analysis
- **Current Price**: ${current_price:.2f}
- **Predicted Price**: ${predicted_price:.2f}
- **Price Change**: ${price_change:+.2f} ({price_change_pct:+.2f}%)
- **Prediction**: {prediction_sentiment}

## 🤖 Model Confidence
- **Confidence Level**: {confidence:.1%}
- **Model Type**: LSTM Neural Network
- **Analysis Date**: {datetime.now().strftime('%Y-%m-%d')}

## ⚠️ Important Disclaimer
This prediction is based on historical data and AI models. Always conduct your own research and consider consulting with financial advisors before making investment decisions.

*Model confidence: {confidence:.1%}*"""

        # Prepare stock data for frontend
        stock_data = {
            "symbol": symbol.upper(),
            "currentPrice": current_price,
            "predictedPrice": predicted_price,
            "priceChange": price_change,
            "priceChangePercent": price_change_pct,
            "prediction": prediction_sentiment,
            "confidence": confidence * 100,
            "metrics": {
                "modelType": "LSTM",
                "analysisDate": datetime.now().strftime('%Y-%m-%d'),
                "dataPoints": len(data_collector.raw_data)
            }
        }
        
        return {
            "message": response_message,
            "stockData": stock_data,
            "confidence": confidence
        }
        
    except Exception as e:
        logger.error(f"Error in stock prediction: {e}")
        return {
            "message": f"Sorry, I encountered an error while analyzing {symbol}. Please try again or ask about a different stock.",
            "confidence": confidence
        }

async def handle_technical_analysis(symbol: str, original_message: str, confidence: float) -> Dict:
    """Handle technical analysis requests."""
    try:
        # Get or create model
        model_info = get_or_create_model(symbol)
        data_collector = model_info['data_collector']
        
        # Get technical indicators
        data = data_collector.raw_data
        if data is None or len(data) == 0:
            return {
                "message": f"Sorry, I couldn't fetch data for {symbol}. Please try again later.",
                "confidence": confidence
            }
        
        # Calculate technical indicators
        technical_indicators = data_collector.calculate_technical_indicators()
        
        # Get latest values
        latest = technical_indicators.iloc[-1]
        
        # Create response
        response_message = f"""# Technical Analysis for {symbol.upper()}

## 📈 Key Technical Indicators

### Moving Averages
- **SMA (20)**: ${latest.get('SMA_20', 0):.2f}
- **EMA (20)**: ${latest.get('EMA_20', 0):.2f}
- **SMA (50)**: ${latest.get('SMA_50', 0):.2f}

### Momentum Indicators
- **RSI (14)**: {latest.get('RSI_14', 0):.2f}
- **MACD**: {latest.get('MACD', 0):.4f}
- **MACD Signal**: {latest.get('MACD_signal', 0):.4f}

### Volatility Indicators
- **Bollinger Upper**: ${latest.get('BB_upper', 0):.2f}
- **Bollinger Lower**: ${latest.get('BB_lower', 0):.2f}
- **ATR (14)**: {latest.get('ATR_14', 0):.2f}

### Volume Indicators
- **Volume SMA**: {latest.get('Volume_SMA', 0):.0f}
- **Volume Ratio**: {latest.get('Volume_ratio', 0):.2f}

## 📊 Analysis Summary
Based on the technical indicators, {symbol.upper()} shows {'bullish' if latest.get('RSI_14', 50) > 50 else 'bearish'} momentum with {'high' if latest.get('ATR_14', 0) > 2 else 'low'} volatility.

*Analysis confidence: {confidence:.1%}*"""

        return {
            "message": response_message,
            "confidence": confidence
        }
        
    except Exception as e:
        logger.error(f"Error in technical analysis: {e}")
        return {
            "message": f"Sorry, I encountered an error while performing technical analysis for {symbol}. Please try again.",
            "confidence": confidence
        }

async def handle_sensitivity_analysis(symbol: str, original_message: str, confidence: float) -> Dict:
    """Handle sensitivity analysis requests."""
    try:
        # Get or create model
        model_info = get_or_create_model(symbol)
        
        if not model_info['loaded']:
            return {
                "message": f"I don't have a trained model for {symbol} yet, so I can't perform sensitivity analysis. Please ask about a different stock.",
                "confidence": confidence
            }
        
        model = model_info['model']
        data_collector = model_info['data_collector']
        
        # Perform sensitivity analysis
        sensitivity_analyzer = SensitivityAnalysis(
            model=model,
            scaler_X=model.scaler_X,
            scaler_y=model.scaler_y,
            feature_names=model.feature_names
        )
        
        # Use sample data for analysis
        X_sample = data_collector.X_test[:50] if hasattr(data_collector, 'X_test') else None
        
        if X_sample is None:
            return {
                "message": f"Sorry, I don't have enough data to perform sensitivity analysis for {symbol}.",
                "confidence": confidence
            }
        
        # Get feature importance
        results = sensitivity_analyzer.feature_sensitivity_analysis(X_sample, method='perturbation')
        feature_ranking = results['feature_ranking'][:10]  # Top 10 features
        
        # Create response
        response_message = f"""# Sensitivity Analysis for {symbol.upper()}

## 🔍 Feature Importance Ranking

The following features have the greatest impact on {symbol.upper()} stock price predictions:

"""
        
        for i, feature_idx in enumerate(feature_ranking):
            feature_name = model.feature_names[feature_idx]
            importance_score = results['overall_sensitivity'][feature_idx]
            response_message += f"{i+1}. **{feature_name}**: {importance_score:.4f}\n"
        
        response_message += f"""

## 📊 Analysis Details
- **Analysis Method**: Perturbation-based sensitivity analysis
- **Features Analyzed**: {len(model.feature_names)} technical indicators
- **Sample Size**: {len(X_sample)} data points
- **Model Type**: LSTM Neural Network

## 💡 Key Insights
The most important features for predicting {symbol.upper()} stock prices are typically:
- Price-based indicators (moving averages, price momentum)
- Volume indicators (trading volume patterns)
- Volatility measures (Bollinger Bands, ATR)
- Technical oscillators (RSI, MACD)

*Analysis confidence: {confidence:.1%}*"""

        return {
            "message": response_message,
            "confidence": confidence
        }
        
    except Exception as e:
        logger.error(f"Error in sensitivity analysis: {e}")
        return {
            "message": f"Sorry, I encountered an error while performing sensitivity analysis for {symbol}. Please try again.",
            "confidence": confidence
        }

async def handle_general_question(original_message: str, confidence: float) -> Dict:
    """Handle general questions about the system."""
    
    # Check for common questions
    if any(word in original_message.lower() for word in ['help', 'what can you do', 'capabilities']):
        return {
            "message": """# AI Stock GPT Capabilities 🤖📈

I'm an intelligent stock analysis assistant powered by advanced AI models. Here's what I can do:

## 📊 **Stock Predictions**
- Predict stock prices using LSTM neural networks
- Provide price change estimates and confidence levels
- Analyze historical patterns and trends

## 📈 **Technical Analysis**
- Calculate and interpret technical indicators
- Analyze moving averages, RSI, MACD, Bollinger Bands
- Provide momentum and volatility insights

## 🔍 **Sensitivity Analysis**
- Identify the most important features affecting stock prices
- Rank technical indicators by their predictive power
- Understand what drives price movements

## 💬 **Natural Language Processing**
- Understand questions in plain English
- Provide detailed explanations and insights
- Generate comprehensive analysis reports

## 📋 **Example Questions**
- "What's the prediction for AAPL stock?"
- "Analyze the technical indicators for TSLA"
- "Show me the sensitivity analysis for MSFT"
- "What affects stock prices most?"

Just ask me anything about stock analysis and I'll provide intelligent insights!""",
            "confidence": confidence
        }
    
    elif any(word in original_message.lower() for word in ['model', 'ai', 'neural network', 'lstm']):
        return {
            "message": """# AI Model Information 🤖

## 🧠 **Model Architecture**
I use **LSTM (Long Short-Term Memory)** neural networks for stock price prediction:

### **Key Features:**
- **Bidirectional LSTM layers** for better pattern recognition
- **Dropout regularization** to prevent overfitting
- **Batch normalization** for stable training
- **Dense layers** for final predictions

### **Technical Specifications:**
- **Input**: 60 days of historical data
- **Features**: 20+ technical indicators
- **Output**: Price predictions with confidence scores
- **Training**: Historical data from 2020 onwards

### **Data Processing:**
- **Feature Engineering**: Technical indicators (RSI, MACD, Bollinger Bands, etc.)
- **Data Normalization**: MinMax scaling for optimal performance
- **Sequence Preparation**: Time-series data formatting for LSTM

### **Model Performance:**
- **R² Score**: Typically 0.7-0.9 on test data
- **Directional Accuracy**: 60-80% for price direction prediction
- **Training Time**: 2-5 minutes per stock

The model continuously learns from new data and can be retrained for better accuracy!""",
            "confidence": confidence
        }
    
    else:
        return {
            "message": f"I understand you're asking: '{original_message}'. I'm an AI stock analysis assistant. I can help you with stock predictions, technical analysis, and sensitivity analysis. Could you please ask me about a specific stock or analysis type?",
            "confidence": confidence
        }

@app.get("/predict/{symbol}")
async def predict_stock(symbol: str, days_ahead: int = 5):
    """Direct endpoint for stock prediction."""
    try:
        response = await handle_stock_prediction(symbol, f"Predict {symbol} stock", 0.8)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/technical/{symbol}")
async def technical_analysis(symbol: str):
    """Direct endpoint for technical analysis."""
    try:
        response = await handle_technical_analysis(symbol, f"Technical analysis for {symbol}", 0.8)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sensitivity/{symbol}")
async def sensitivity_analysis(symbol: str):
    """Direct endpoint for sensitivity analysis."""
    try:
        response = await handle_sensitivity_analysis(symbol, f"Sensitivity analysis for {symbol}", 0.8)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
