#!/usr/bin/env python3
"""
Enhanced FastAPI Backend for AI Stock GPT v2.0
==============================================

This module provides an enhanced FastAPI backend that integrates with the AI stock prediction system
and provides advanced features including user authentication, portfolio management, and enhanced analytics.
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

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import uvicorn

# Import our custom modules (from original main.py)
try:
    from data_collector import StockDataCollector
    from lstm_model import LSTMModel
    from sensitivity_analysis import SensitivityAnalysis
    from nlp_processor import NLPProcessor
    ORIGINAL_MODULES_AVAILABLE = True
except ImportError:
    ORIGINAL_MODULES_AVAILABLE = False
    print("Warning: Original modules not available, running in enhanced-only mode")

# Import enhanced NLP and LLM modules
try:
    from backend.nlp_enhanced import EnhancedNLPProcessor
    from backend.llm_provider import LLMProvider, llm_provider
    ENHANCED_NLP_AVAILABLE = True
except ImportError:
    try:
        from nlp_enhanced import EnhancedNLPProcessor
        from llm_provider import LLMProvider, llm_provider
        ENHANCED_NLP_AVAILABLE = True
    except ImportError:
        ENHANCED_NLP_AVAILABLE = False
        llm_provider = None
        print("Warning: Enhanced NLP/LLM modules not available")

# Import new enhanced modules
try:
    from backend.database import db_manager
    from backend.auth import auth_manager, get_current_active_user, UserCreate, UserLogin, Token
    from backend.security import (
        SecurityMiddleware,
        rate_limit_public,
        rate_limit_authenticated,
        rate_limit_sensitive,
        validate_api_request,
        check_suspicious_activity,
        StockSymbolRequest,
        ChatMessageRequest,
        UserRegistrationRequest
    )
    ENHANCED_MODULES_AVAILABLE = True
except ImportError:
    try:
        from database import db_manager
        from auth import auth_manager, get_current_active_user, UserCreate, UserLogin, Token
        from security import (
            SecurityMiddleware,
            rate_limit_public,
            rate_limit_authenticated,
            rate_limit_sensitive,
            validate_api_request,
            check_suspicious_activity,
            StockSymbolRequest,
            ChatMessageRequest,
            UserRegistrationRequest
        )
        ENHANCED_MODULES_AVAILABLE = True
    except ImportError:
        ENHANCED_MODULES_AVAILABLE = False
        print("Warning: Enhanced modules not available, running in basic mode")

# Create fallback decorators when enhanced modules are not available
if not ENHANCED_MODULES_AVAILABLE:
    def rate_limit_public(func):
        return func
    
    def rate_limit_authenticated(func):
        return func
    
    def rate_limit_sensitive(func):
        return func
    
    def validate_api_request(request):
        pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Stock GPT Enhanced API",
    description="Intelligent Stock Analysis and Prediction API with User Management",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize security middleware if available
if ENHANCED_MODULES_AVAILABLE:
    try:
        security_middleware = SecurityMiddleware(app)
    except Exception as e:
        logger.warning(f"Security middleware initialization failed: {e}")

# Global variables
nlp_processor = None
models_cache = {}
data_collectors = {}

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

class PortfolioCreate(BaseModel):
    name: str = Field(..., description="Portfolio name")
    description: Optional[str] = Field(None, description="Portfolio description")

class StockAdd(BaseModel):
    symbol: str = Field(..., description="Stock symbol")
    shares: float = Field(..., description="Number of shares")
    purchase_price: float = Field(..., description="Purchase price per share")
    purchase_date: str = Field(..., description="Purchase date (YYYY-MM-DD)")

class EmailAlert(BaseModel):
    symbol: str = Field(..., description="Stock symbol")
    alert_type: str = Field(..., description="Alert type (price, technical, prediction)")
    threshold: float = Field(..., description="Alert threshold")
    email: str = Field(..., description="Email address")

def initialize_nlp():
    """Initialize the NLP processor (enhanced or basic)."""
    global nlp_processor

    # Try enhanced NLP first (sentence-transformers)
    if ENHANCED_NLP_AVAILABLE:
        try:
            nlp_processor = EnhancedNLPProcessor()
            logger.info("Enhanced NLP processor initialized (sentence-transformers)")
            return
        except Exception as e:
            logger.warning(f"Enhanced NLP init failed, falling back to basic: {e}")

    # Fall back to basic regex NLP
    if ORIGINAL_MODULES_AVAILABLE:
        try:
            nlp_processor = NLPProcessor()
            logger.info("Basic NLP processor initialized (regex)")
        except Exception as e:
            logger.error(f"Failed to initialize NLP processor: {e}")
            raise
    else:
        logger.warning("No NLP processor available")

def get_or_create_model(symbol: str) -> Dict[str, Any]:
    """Get or create a model for the given symbol."""
    if not ORIGINAL_MODULES_AVAILABLE:
        return {"error": "Original modules not available"}
    
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
            else:
                # Create new model
                model = LSTMModel()
                models_cache[symbol] = {
                    'model': model,
                    'data_collector': data_collector,
                    'loaded': False
                }
        except Exception as e:
            logger.error(f"Error creating model for {symbol}: {e}")
            return {"error": str(e)}
    
    return models_cache[symbol]

def generate_response(message: str, user_id: Optional[str] = None) -> Dict[str, Any]:
    """Generate AI response to user message using NLP + LLM."""
    try:
        if nlp_processor is None:
            initialize_nlp()

        if nlp_processor is None:
            # No NLP available at all - use LLM directly or template
            if llm_provider:
                response_text = llm_provider.generate_response(
                    "general_question", {}, message
                )
            else:
                response_text = (
                    "I'm AI Stock GPT. I can help with stock predictions, "
                    "technical analysis, and portfolio management. "
                    "Try asking about a specific stock like AAPL or TSLA."
                )
            return {
                "message": response_text,
                "stockData": None,
                "charts": None,
                "confidence": 0.0,
            }

        # Process message with NLP (returns tuple: intent, entities, confidence)
        intent, entities, confidence = nlp_processor.process_message(message)

        # Generate response using LLM provider (with fallback chain)
        if llm_provider:
            response_text = llm_provider.generate_response(intent, entities, message)
        else:
            response_text = handle_general_question(message)

        # Save chat history if user is authenticated
        if user_id and ENHANCED_MODULES_AVAILABLE:
            try:
                db_manager.save_chat_message(user_id, message, response_text)
            except Exception as e:
                logger.warning(f"Failed to save chat history: {e}")

        return {
            "message": response_text,
            "stockData": None,
            "charts": None,
            "confidence": confidence,
        }
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return {
            "message": f"Sorry, I encountered an error: {str(e)}",
            "stockData": None,
            "charts": None,
            "confidence": 0.0,
        }

def handle_general_question(message: str) -> str:
    """Handle general questions about the system."""
    general_responses = {
        "hello": "Hello! I'm AI Stock GPT, your intelligent stock analysis assistant. How can I help you today?",
        "help": "I can help you with stock analysis, predictions, technical indicators, and portfolio management. Try asking about a specific stock or analysis type.",
        "capabilities": "I can analyze stocks, make predictions, provide technical analysis, manage portfolios, and answer questions about market trends.",
        "features": "My features include: Stock price analysis, LSTM predictions, technical indicators, portfolio tracking, user authentication, and email alerts."
    }
    
    message_lower = message.lower()
    for key, response in general_responses.items():
        if key in message_lower:
            return response
    
    return "I'm here to help with stock analysis and predictions. What would you like to know?"

# API Endpoints

@app.get("/")
@rate_limit_public
async def root():
    """Root endpoint with API information."""
    return {
        "message": "AI Stock GPT Enhanced API v2.0",
        "version": "2.0.0",
        "status": "operational",
        "features": [
            "Stock Analysis & Predictions",
            "User Authentication",
            "Portfolio Management", 
            "Email Alerts",
            "Enhanced Security",
            "Rate Limiting"
        ],
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/status")
@rate_limit_public
async def get_status():
    """Get API status."""
    return {
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "modules": {
            "original": ORIGINAL_MODULES_AVAILABLE,
            "enhanced": ENHANCED_MODULES_AVAILABLE
        }
    }

@app.get("/health")
@rate_limit_public
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0"
    }

@app.post("/chat")
@rate_limit_authenticated
async def chat(
    request: ChatRequest,
    current_user: Optional[Dict] = Depends(get_current_active_user) if ENHANCED_MODULES_AVAILABLE else None
):
    """Enhanced chat endpoint with user authentication."""
    try:
        user_id = current_user["id"] if current_user else None
        
        # Validate input
        if ENHANCED_MODULES_AVAILABLE:
            validate_api_request(request.message)
        
        # Generate response
        response = generate_response(request.message, user_id)
        
        return ChatResponse(**response)
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict/{symbol}")
@rate_limit_authenticated
async def predict_stock(
    symbol: str,
    days_ahead: int = 5,
    current_user: Optional[Dict] = Depends(get_current_active_user) if ENHANCED_MODULES_AVAILABLE else None
):
    """Enhanced stock prediction endpoint."""
    try:
        if not ORIGINAL_MODULES_AVAILABLE:
            raise HTTPException(status_code=503, detail="Prediction service not available")
        
        # Validate symbol
        if ENHANCED_MODULES_AVAILABLE:
            validate_api_request(symbol)
        
        # Get or create model
        model_data = get_or_create_model(symbol)
        if "error" in model_data:
            raise HTTPException(status_code=500, detail=model_data["error"])
        
        # Make prediction
        model = model_data['model']
        data_collector = model_data['data_collector']
        
        # Get latest data
        latest_data = data_collector.get_latest_data()
        if latest_data is None or latest_data.empty:
            raise HTTPException(status_code=404, detail=f"No data available for {symbol}")
        
        # Make prediction
        prediction = model.predict(latest_data, days_ahead)
        
        # Save prediction to database if user is authenticated
        if current_user and ENHANCED_MODULES_AVAILABLE:
            try:
                db_manager.save_prediction(
                    current_user["id"], 
                    symbol, 
                    prediction, 
                    days_ahead
                )
            except Exception as e:
                logger.warning(f"Failed to save prediction: {e}")
        
        return {
            "symbol": symbol,
            "prediction": prediction.tolist(),
            "days_ahead": days_ahead,
            "timestamp": datetime.now().isoformat(),
            "confidence": model.get_confidence() if hasattr(model, 'get_confidence') else 0.8
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/technical/{symbol}")
@rate_limit_authenticated
async def technical_analysis(
    symbol: str,
    current_user: Optional[Dict] = Depends(get_current_active_user) if ENHANCED_MODULES_AVAILABLE else None
):
    """Enhanced technical analysis endpoint."""
    try:
        if not ORIGINAL_MODULES_AVAILABLE:
            raise HTTPException(status_code=503, detail="Technical analysis service not available")
        
        # Validate symbol
        if ENHANCED_MODULES_AVAILABLE:
            validate_api_request(symbol)
        
        # Get data
        data_collector = StockDataCollector(symbol=symbol, start_date='2020-01-01')
        data = data_collector.get_data()
        
        if data is None or data.empty:
            raise HTTPException(status_code=404, detail=f"No data available for {symbol}")
        
        # Calculate technical indicators
        analysis = data_collector.calculate_technical_indicators(data)
        
        return {
            "symbol": symbol,
            "analysis": analysis,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Technical analysis error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Enhanced Authentication Endpoints
if ENHANCED_MODULES_AVAILABLE:
    @app.post("/auth/register")
    @rate_limit_public
    async def register_user(user_data: UserCreate):
        """Register a new user."""
        try:
            user = auth_manager.register_user(user_data)
            return {"message": "User registered successfully", "user_id": user.user_id}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/auth/login")
    @rate_limit_public
    async def login_user(user_data: UserLogin):
        """Login user and return tokens."""
        try:
            tokens = auth_manager.login_user(user_data)
            return tokens
        except Exception as e:
            raise HTTPException(status_code=401, detail=str(e))

    @app.post("/auth/refresh")
    @rate_limit_public
    async def refresh_token(token: Token):
        """Refresh access token."""
        try:
            new_token = auth_manager.refresh_access_token(token.refresh_token)
            return {"access_token": new_token}
        except Exception as e:
            raise HTTPException(status_code=401, detail=str(e))

    @app.get("/auth/me")
    @rate_limit_authenticated
    async def get_user_profile(current_user: Dict = Depends(get_current_active_user)):
        """Get current user profile."""
        return current_user

    # Portfolio Management Endpoints
    @app.post("/portfolio/create")
    @rate_limit_authenticated
    async def create_portfolio(
        portfolio_data: PortfolioCreate,
        current_user: Dict = Depends(get_current_active_user)
    ):
        """Create a new portfolio."""
        try:
            portfolio = db_manager.create_portfolio(
                current_user["id"],
                portfolio_data.name,
                portfolio_data.description
            )
            return portfolio
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/portfolio/{portfolio_id}/add-stock")
    @rate_limit_authenticated
    async def add_stock_to_portfolio(
        portfolio_id: str,
        stock_data: StockAdd,
        current_user: Dict = Depends(get_current_active_user)
    ):
        """Add stock to portfolio."""
        try:
            result = db_manager.add_stock_to_portfolio(
                portfolio_id,
                stock_data.symbol,
                stock_data.shares,
                stock_data.purchase_price,
                stock_data.purchase_date
            )
            return result
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/portfolio/{portfolio_id}")
    @rate_limit_authenticated
    async def get_portfolio(
        portfolio_id: str,
        current_user: Dict = Depends(get_current_active_user)
    ):
        """Get portfolio details."""
        try:
            portfolio = db_manager.get_portfolio(portfolio_id)
            return portfolio
        except Exception as e:
            raise HTTPException(status_code=404, detail=str(e))

    @app.get("/portfolio/list")
    @rate_limit_authenticated
    async def list_portfolios(current_user: Dict = Depends(get_current_active_user)):
        """List user portfolios."""
        try:
            portfolios = db_manager.get_user_portfolios(current_user["id"])
            return {"portfolios": portfolios}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # Chat History Endpoint
    @app.get("/chat/history")
    @rate_limit_authenticated
    async def get_chat_history(current_user: Dict = Depends(get_current_active_user)):
        """Get user chat history."""
        try:
            history = db_manager.get_chat_history(current_user["id"])
            return {"history": history}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # Analytics Endpoints
    @app.get("/analytics/user")
    @rate_limit_authenticated
    async def get_user_analytics(current_user: Dict = Depends(get_current_active_user)):
        """Get user analytics."""
        try:
            analytics = db_manager.get_user_analytics(current_user["id"])
            return analytics
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # Email Alerts Endpoints
    @app.post("/alerts/create")
    @rate_limit_authenticated
    async def create_email_alert(
        alert_data: EmailAlert,
        current_user: Dict = Depends(get_current_active_user)
    ):
        """Create email alert."""
        try:
            alert = db_manager.create_email_alert(
                current_user["id"],
                alert_data.symbol,
                alert_data.alert_type,
                alert_data.threshold,
                alert_data.email
            )
            return alert
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/alerts/list")
    @rate_limit_authenticated
    async def list_email_alerts(current_user: Dict = Depends(get_current_active_user)):
        """List user email alerts."""
        try:
            alerts = db_manager.get_user_alerts(current_user["id"])
            return {"alerts": alerts}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

# Fallback endpoints for when enhanced modules are not available
if not ENHANCED_MODULES_AVAILABLE:
    @app.post("/chat")
    @rate_limit_public
    async def chat_fallback(request: ChatRequest):
        """Fallback chat endpoint without authentication."""
        try:
            response = generate_response(request.message)
            return ChatResponse(**response)
        except Exception as e:
            logger.error(f"Chat error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/predict/{symbol}")
    @rate_limit_public
    async def predict_stock_fallback(symbol: str, days_ahead: int = 5):
        """Fallback prediction endpoint without authentication."""
        try:
            if not ORIGINAL_MODULES_AVAILABLE:
                raise HTTPException(status_code=503, detail="Prediction service not available")
            
            model_data = get_or_create_model(symbol)
            if "error" in model_data:
                raise HTTPException(status_code=500, detail=model_data["error"])
            
            model = model_data['model']
            data_collector = model_data['data_collector']
            
            latest_data = data_collector.get_latest_data()
            if latest_data is None or latest_data.empty:
                raise HTTPException(status_code=404, detail=f"No data available for {symbol}")
            
            prediction = model.predict(latest_data, days_ahead)
            
            return {
                "symbol": symbol,
                "prediction": prediction.tolist(),
                "days_ahead": days_ahead,
                "timestamp": datetime.now().isoformat(),
                "confidence": model.get_confidence() if hasattr(model, 'get_confidence') else 0.8
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Prediction error for {symbol}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

# Initialize the application
@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    logger.info("Starting AI Stock GPT Enhanced API v2.0")

    # Create database tables if they don't exist
    try:
        try:
            from backend.db_session import engine
            from backend.models import Base
        except ImportError:
            from db_session import engine
            from models import Base
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database tables: {e}")

    try:
        initialize_nlp()
        logger.info("Application initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
