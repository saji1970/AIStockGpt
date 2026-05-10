#!/usr/bin/env python3
"""
Simplified FastAPI Backend for AI Stock GPT
===========================================

This is a simplified version that works without TensorFlow for testing purposes.
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

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import yfinance as yf
import pandas as pd
import numpy as np

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
    confidence: Optional[float] = Field(None, description="Model confidence")

class SimpleNLPProcessor:
    """Simplified NLP processor for testing."""
    
    def __init__(self):
        self.stock_symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
            'ADBE', 'CRM', 'ORCL', 'INTC', 'AMD', 'IBM', 'CSCO', 'QCOM'
        ]
    
    def process_message(self, message: str):
        """Process user message and extract intent and entities."""
        message_lower = message.lower()
        
        # Extract stock symbol
        symbol = None
        for stock in self.stock_symbols:
            if stock.lower() in message_lower:
                symbol = stock
                break
        
        # Determine intent
        if any(word in message_lower for word in ['predict', 'forecast', 'price', 'prediction']):
            intent = "stock_prediction"
        elif any(word in message_lower for word in ['technical', 'rsi', 'macd', 'indicator']):
            intent = "technical_analysis"
        elif any(word in message_lower for word in ['help', 'what can you do', 'capabilities']):
            intent = "general_question"
        else:
            intent = "general_question"
        
        entities = {"symbol": symbol} if symbol else {}
        confidence = 0.8 if symbol else 0.5
        
        return intent, entities, confidence

# Global variables
nlp_processor = SimpleNLPProcessor()

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    logger.info("Starting AI Stock GPT API...")
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
        "nlp_processor": "active",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Handle chat messages and return AI responses."""
    try:
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
        # Get stock data
        stock = yf.Ticker(symbol)
        hist = stock.history(period="1mo")
        
        if hist.empty:
            return {
                "message": f"Sorry, I couldn't fetch data for {symbol}. Please try again later.",
                "confidence": confidence
            }
        
        # Calculate simple prediction (for demo purposes)
        current_price = hist['Close'].iloc[-1]
        avg_price = hist['Close'].mean()
        price_change = current_price - avg_price
        price_change_pct = (price_change / avg_price) * 100
        
        # Simple prediction logic
        if price_change > 0:
            prediction_sentiment = "BULLISH 📈"
            predicted_price = current_price * 1.02  # 2% increase
        else:
            prediction_sentiment = "BEARISH 📉"
            predicted_price = current_price * 0.98  # 2% decrease
        
        # Create response
        response_message = f"""# Stock Prediction for {symbol.upper()}

## 📊 Current Analysis
- **Current Price**: ${current_price:.2f}
- **Predicted Price**: ${predicted_price:.2f}
- **Price Change**: ${price_change:+.2f} ({price_change_pct:+.2f}%)
- **Prediction**: {prediction_sentiment}

## 🤖 Analysis Details
- **Analysis Method**: Simple Statistical Analysis
- **Data Period**: 1 month
- **Analysis Date**: {datetime.now().strftime('%Y-%m-%d')}

## ⚠️ Important Disclaimer
This is a simplified demo prediction. Always conduct your own research and consider consulting with financial advisors before making investment decisions.

*Analysis confidence: {confidence:.1%}*"""

        # Prepare stock data for frontend
        stock_data = {
            "symbol": symbol.upper(),
            "currentPrice": current_price,
            "predictedPrice": predicted_price,
            "priceChange": predicted_price - current_price,
            "priceChangePercent": ((predicted_price - current_price) / current_price) * 100,
            "prediction": prediction_sentiment,
            "confidence": confidence * 100,
            "metrics": {
                "modelType": "Statistical",
                "analysisDate": datetime.now().strftime('%Y-%m-%d'),
                "dataPoints": len(hist)
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
        # Get stock data
        stock = yf.Ticker(symbol)
        hist = stock.history(period="1mo")
        
        if hist.empty:
            return {
                "message": f"Sorry, I couldn't fetch data for {symbol}. Please try again later.",
                "confidence": confidence
            }
        
        # Calculate simple technical indicators
        close_prices = hist['Close']
        sma_20 = close_prices.rolling(window=20).mean().iloc[-1]
        current_price = close_prices.iloc[-1]
        
        # Simple RSI calculation
        delta = close_prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]
        
        # Create response
        response_message = f"""# Technical Analysis for {symbol.upper()}

## 📈 Key Technical Indicators

### Moving Averages
- **SMA (20)**: ${sma_20:.2f}
- **Current Price**: ${current_price:.2f}

### Momentum Indicators
- **RSI (14)**: {rsi:.2f}

### Price Analysis
- **Price vs SMA**: {'Above' if current_price > sma_20 else 'Below'} 20-day average
- **Trend**: {'Bullish' if current_price > sma_20 else 'Bearish'}

## 📊 Analysis Summary
Based on the technical indicators, {symbol.upper()} shows {'bullish' if current_price > sma_20 else 'bearish'} momentum.

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

async def handle_general_question(original_message: str, confidence: float) -> Dict:
    """Handle general questions about the system."""
    
    # Check for common questions
    if any(word in original_message.lower() for word in ['help', 'what can you do', 'capabilities']):
        return {
            "message": """# AI Stock GPT Capabilities 🤖📈

I'm an intelligent stock analysis assistant. Here's what I can do:

## 📊 **Stock Predictions**
- Predict stock prices using statistical analysis
- Provide price change estimates and confidence levels
- Analyze historical patterns and trends

## 📈 **Technical Analysis**
- Calculate and interpret technical indicators
- Analyze moving averages, RSI, and other indicators
- Provide momentum and trend insights

## 💬 **Natural Language Processing**
- Understand questions in plain English
- Provide detailed explanations and insights
- Generate comprehensive analysis reports

## 📋 **Example Questions**
- "What's the prediction for AAPL stock?"
- "Analyze the technical indicators for TSLA"
- "Show me the analysis for MSFT"

Just ask me anything about stock analysis and I'll provide intelligent insights!""",
            "confidence": confidence
        }
    
    else:
        return {
            "message": f"I understand you're asking: '{original_message}'. I'm an AI stock analysis assistant. I can help you with stock predictions and technical analysis. Could you please ask me about a specific stock or analysis type?",
            "confidence": confidence
        }

@app.get("/predict/{symbol}")
async def predict_stock(symbol: str):
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

if __name__ == "__main__":
    uvicorn.run(
        "simple_main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
