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

def fetch_stock_data(symbol: str) -> Optional[Dict[str, Any]]:
    """Fetch real-time stock data via RapidAPI, with Yahoo Finance and yfinance fallbacks."""
    import requests as req

    # Try Real-Time Finance Data API (RapidAPI) first - works in Docker/Railway
    rapidapi_key = os.getenv("RAPIDAPI_KEY", "")
    if rapidapi_key:
        for attempt in range(2):
            try:
                url = "https://real-time-finance-data.p.rapidapi.com/stock-quote"
                headers = {
                    "x-rapidapi-key": rapidapi_key,
                    "x-rapidapi-host": "real-time-finance-data.p.rapidapi.com",
                }
                resp = req.get(url, headers=headers, params={"symbol": symbol, "language": "en"}, timeout=20)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("status") == "OK" and data.get("data"):
                        q = data["data"]
                        return {
                            "symbol": symbol,
                            "price": round(float(q.get("price", 0)), 2),
                            "previousClose": round(float(q.get("previous_close", 0)), 2),
                            "change": round(float(q.get("change", 0)), 2),
                            "changePercent": round(float(q.get("change_percent", 0)), 2),
                            "high": round(float(q.get("high", 0)), 2),
                            "low": round(float(q.get("low", 0)), 2),
                            "volume": int(q.get("volume", 0)),
                            "name": q.get("name", symbol),
                            "open": round(float(q.get("open", 0)), 2),
                        }
                logger.warning(f"RapidAPI returned {resp.status_code} for {symbol} (attempt {attempt+1})")
            except Exception as e:
                logger.warning(f"RapidAPI stock fetch failed for {symbol} (attempt {attempt+1}): {e}")

    # Fallback: Direct Yahoo Finance v8 API (no library needed)
    try:
        yahoo_url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        yahoo_headers = {"User-Agent": "Mozilla/5.0"}
        resp = req.get(yahoo_url, headers=yahoo_headers, params={"range": "5d", "interval": "1d"}, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            chart = data.get("chart", {}).get("result", [{}])[0]
            meta = chart.get("meta", {})
            indicators = chart.get("indicators", {}).get("quote", [{}])[0]
            closes = indicators.get("close", [])
            highs = indicators.get("high", [])
            lows = indicators.get("low", [])
            volumes = indicators.get("volume", [])

            # Filter out None values
            valid_closes = [c for c in closes if c is not None]
            if valid_closes:
                current_price = valid_closes[-1]
                prev_close = meta.get("chartPreviousClose") or meta.get("previousClose") or (valid_closes[-2] if len(valid_closes) >= 2 else current_price)
                change = current_price - prev_close
                change_pct = (change / prev_close) * 100 if prev_close else 0

                valid_highs = [h for h in highs if h is not None]
                valid_lows = [l for l in lows if l is not None]
                valid_volumes = [v for v in volumes if v is not None]

                result = {
                    "symbol": symbol,
                    "price": round(float(current_price), 2),
                    "previousClose": round(float(prev_close), 2),
                    "change": round(float(change), 2),
                    "changePercent": round(float(change_pct), 2),
                    "high": round(float(valid_highs[-1]), 2) if valid_highs else 0,
                    "low": round(float(valid_lows[-1]), 2) if valid_lows else 0,
                    "volume": int(valid_volumes[-1]) if valid_volumes else 0,
                    "name": meta.get("shortName") or meta.get("symbol", symbol),
                    "open": round(float(meta.get("regularMarketPrice", current_price)), 2),
                }
                logger.info(f"Stock data fetched via Yahoo Finance v8 API for {symbol}")
                return result
        logger.warning(f"Yahoo Finance v8 API returned {resp.status_code} for {symbol}")
    except Exception as e:
        logger.warning(f"Yahoo Finance v8 API failed for {symbol}: {e}")

    # Fallback: yfinance library
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="5d")

        if hist.empty:
            # Try 1mo if 5d returns nothing (e.g., new listings)
            hist = ticker.history(period="1mo")

        if hist.empty:
            return None

        current_price = hist['Close'].iloc[-1]
        prev_close = hist['Close'].iloc[-2] if len(hist) >= 2 else current_price
        change = current_price - prev_close
        change_pct = (change / prev_close) * 100 if prev_close else 0

        result = {
            "symbol": symbol,
            "price": round(float(current_price), 2),
            "previousClose": round(float(prev_close), 2),
            "change": round(float(change), 2),
            "changePercent": round(float(change_pct), 2),
            "high": round(float(hist['High'].iloc[-1]), 2),
            "low": round(float(hist['Low'].iloc[-1]), 2),
            "volume": int(hist['Volume'].iloc[-1]),
            "name": symbol,
        }

        try:
            info = ticker.info
            result["name"] = info.get("shortName", symbol)
            result["marketCap"] = info.get("marketCap")
            result["peRatio"] = info.get("trailingPE")
            result["fiftyTwoWeekHigh"] = info.get("fiftyTwoWeekHigh")
            result["fiftyTwoWeekLow"] = info.get("fiftyTwoWeekLow")
        except Exception:
            pass

        return result
    except Exception as e:
        logger.error(f"Error fetching stock data for {symbol}: {e}")
        return None


def generate_response(message: str, user_id: Optional[str] = None) -> Dict[str, Any]:
    """Generate AI response to user message using NLP + LLM + live stock data."""
    try:
        if nlp_processor is None:
            initialize_nlp()

        if nlp_processor is None:
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

        # Fetch real stock data if a symbol was detected
        stock_data = None
        symbol = entities.get("symbol")
        if symbol and intent in ("stock_prediction", "technical_analysis", "sensitivity_analysis"):
            stock_data = fetch_stock_data(symbol)

        # Handle portfolio_management intent
        portfolio_response = None
        if intent == "portfolio_management" and user_id and ENHANCED_MODULES_AVAILABLE:
            try:
                portfolios = db_manager.get_user_portfolios(user_id)
                if portfolios:
                    portfolio_lines = []
                    for portfolio in portfolios:
                        portfolio_lines.append(f"\n**{portfolio['name']}**")
                        p_total_invested = 0
                        p_total_current = 0
                        for s in portfolio.get('stocks', []):
                            s_symbol = s['symbol']
                            s_shares = s.get('shares', 0)
                            s_purchase = s.get('purchase_price', 0)
                            s_cost = s_shares * s_purchase
                            p_total_invested += s_cost

                            live = fetch_stock_data(s_symbol)
                            if live:
                                s_current_price = live['price']
                                s_current_val = s_shares * s_current_price
                                p_total_current += s_current_val
                                s_gain = s_current_val - s_cost
                                s_gain_pct = (s_gain / s_cost * 100) if s_cost > 0 else 0
                                sign = "+" if s_gain >= 0 else ""
                                portfolio_lines.append(
                                    f"- **{s_symbol}**: {s_shares} shares | "
                                    f"Bought ${s_purchase:.2f} -> Now ${s_current_price:.2f} | "
                                    f"{sign}${s_gain:.2f} ({sign}{s_gain_pct:.1f}%)"
                                )
                            else:
                                portfolio_lines.append(
                                    f"- **{s_symbol}**: {s_shares} shares @ ${s_purchase:.2f} "
                                    f"(live price unavailable)"
                                )

                        if p_total_invested > 0:
                            p_gain = p_total_current - p_total_invested
                            p_gain_pct = (p_gain / p_total_invested * 100)
                            sign = "+" if p_gain >= 0 else ""
                            portfolio_lines.append(
                                f"\n**Portfolio Total**: ${p_total_current:,.2f} | "
                                f"Invested: ${p_total_invested:,.2f} | "
                                f"P/L: {sign}${p_gain:,.2f} ({sign}{p_gain_pct:.1f}%)"
                            )

                    portfolio_response = (
                        "Here's your portfolio overview:\n"
                        + "\n".join(portfolio_lines)
                        + "\n\nNote: This is not financial advice. Always do your own research."
                    )
                else:
                    portfolio_response = (
                        "You don't have any portfolios yet. "
                        "Create one from the Portfolio page and add your stocks, ETFs, "
                        "or crypto holdings to track their performance!"
                    )
            except Exception as e:
                logger.warning(f"Failed to fetch portfolio for chat: {e}")

        # Build response text with real data
        if stock_data:
            price = stock_data["price"]
            change = stock_data["change"]
            change_pct = stock_data["changePercent"]
            direction = "up" if change >= 0 else "down"
            sign = "+" if change >= 0 else ""
            name = stock_data.get("name", symbol)

            response_text = (
                f"{name} ({symbol}) is currently trading at ${price:.2f}, "
                f"{direction} {sign}{change:.2f} ({sign}{change_pct:.2f}%) from the previous close.\n\n"
            )

            if stock_data.get("open"):
                response_text += f"Open: ${stock_data['open']:.2f}\n"
            if stock_data.get("high") and stock_data.get("low"):
                response_text += f"Day range: ${stock_data['low']:.2f} - ${stock_data['high']:.2f}\n"
            if stock_data.get("volume"):
                response_text += f"Volume: {stock_data['volume']:,}\n"
            if stock_data.get("previousClose"):
                response_text += f"Previous close: ${stock_data['previousClose']:.2f}\n"
            if stock_data.get("fiftyTwoWeekHigh") and stock_data.get("fiftyTwoWeekLow"):
                response_text += f"52-week range: ${stock_data['fiftyTwoWeekLow']:.2f} - ${stock_data['fiftyTwoWeekHigh']:.2f}\n"
            if stock_data.get("peRatio"):
                response_text += f"P/E Ratio: {stock_data['peRatio']:.2f}\n"
            if stock_data.get("marketCap"):
                cap = stock_data["marketCap"]
                if cap >= 1e12:
                    response_text += f"Market Cap: ${cap/1e12:.2f}T\n"
                elif cap >= 1e9:
                    response_text += f"Market Cap: ${cap/1e9:.2f}B\n"
                else:
                    response_text += f"Market Cap: ${cap/1e6:.2f}M\n"

            response_text += "\nNote: This is not financial advice. Always do your own research before making investment decisions."
        elif portfolio_response:
            response_text = portfolio_response
        else:
            # No stock data - use LLM/template response
            if intent == "market_advice":
                if llm_provider:
                    response_text = llm_provider.generate_response(intent, entities, message)
                else:
                    response_text = handle_market_advice(message)
            elif llm_provider:
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
            "stockData": stock_data,
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


def handle_market_advice(message: str) -> str:
    """Handle broad market and investment advice queries."""
    message_lower = message.lower()

    # Top stocks / best stocks queries
    if re.search(r"top\s*\d+\s*stocks?|best\s*stocks?", message_lower):
        return (
            "## Top Stocks to Watch\n\n"
            "Here are some widely-followed stocks across key sectors:\n\n"
            "**Technology:** AAPL (Apple), MSFT (Microsoft), NVDA (NVIDIA), GOOGL (Alphabet), META (Meta)\n\n"
            "**Consumer:** AMZN (Amazon), TSLA (Tesla), WMT (Walmart), COST (Costco)\n\n"
            "**Healthcare:** UNH (UnitedHealth), JNJ (Johnson & Johnson), ABBV (AbbVie)\n\n"
            "**Finance:** JPM (JPMorgan), V (Visa), MA (Mastercard)\n\n"
            "**ETFs for Diversification:** SPY (S&P 500), QQQ (Nasdaq-100), VTI (Total Market), VGT (Tech Sector)\n\n"
            "To get live prices and analysis for any of these, ask me: *\"Predict AAPL stock\"* or *\"Technical analysis of NVDA\"*\n\n"
            "**Disclaimer:** This is not financial advice. Always do your own research and consider consulting a financial advisor before making investment decisions."
        )

    # Investment amount queries
    if re.search(r"invest\s*\$?\d+|where.*(?:invest|put.*money)|what.*(?:should|would).*(?:invest|buy)", message_lower):
        return (
            "## Investment Strategy Guide\n\n"
            "Here are common approaches based on different goals:\n\n"
            "### For Beginners / Lower Risk\n"
            "- **Index Funds / ETFs:** SPY (S&P 500), VTI (Total Market) - broad diversification with low fees\n"
            "- **Bond ETFs:** BND (Total Bond) - lower volatility\n\n"
            "### For Growth\n"
            "- **Tech Leaders:** AAPL, MSFT, NVDA, GOOGL - established companies with growth potential\n"
            "- **Growth ETFs:** QQQ (Nasdaq-100), VGT (Tech Sector)\n\n"
            "### For Income / Dividends\n"
            "- **Dividend Stocks:** JNJ, KO, PEP, PG - consistent dividend payers\n"
            "- **Dividend ETFs:** VYM, SCHD - diversified dividend income\n\n"
            "### Hedge Funds & Alternatives\n"
            "- Most hedge funds require accredited investor status ($200K+ income or $1M+ net worth)\n"
            "- **Accessible alternatives:** BTAL (anti-beta), DBMF (managed futures), QMOM (momentum)\n"
            "- **REITs:** VNQ (real estate) - real estate exposure without direct ownership\n\n"
            "### General Tips\n"
            "- Diversify across sectors and asset classes\n"
            "- Consider your risk tolerance and time horizon\n"
            "- Dollar-cost averaging reduces timing risk\n\n"
            "Ask me about any specific stock for live prices and AI predictions!\n\n"
            "**Disclaimer:** This is not financial advice. Always do your own research and consider consulting a financial advisor."
        )

    # Hedge fund queries
    if re.search(r"hedge\s*fund|mutual\s*fund", message_lower):
        return (
            "## Hedge Funds & Mutual Funds\n\n"
            "### Hedge Funds\n"
            "- Typically require accredited investor status and high minimums ($100K-$1M+)\n"
            "- Use strategies like long/short equity, global macro, and event-driven\n"
            "- **Accessible alternatives via ETFs:**\n"
            "  - DBMF - managed futures strategy\n"
            "  - BTAL - anti-beta / market neutral\n"
            "  - QMOM - quantitative momentum\n"
            "  - MNA - merger arbitrage\n\n"
            "### Mutual Funds (More Accessible)\n"
            "- **Vanguard 500 (VFIAX):** Tracks S&P 500, 0.04% expense ratio\n"
            "- **Fidelity Total Market (FSKAX):** Broad US market exposure\n"
            "- **Schwab International (SWISX):** International diversification\n\n"
            "### ETF Alternatives (No Minimums)\n"
            "- **SPY / VOO:** S&P 500 ETFs\n"
            "- **QQQ:** Nasdaq-100\n"
            "- **VTI:** Total US stock market\n"
            "- **VXUS:** International stocks\n\n"
            "Ask me about any specific stock or ETF for live prices and analysis!\n\n"
            "**Disclaimer:** This is not financial advice. Always do your own research and consider consulting a financial advisor."
        )

    # Sector / market trend queries
    if re.search(r"sector|market.*(?:trend|outlook)|(?:current|today).*market|s.?p\s*500|nasdaq|dow", message_lower):
        return (
            "## Market Overview & Sector Insights\n\n"
            "### Major Indices to Track\n"
            "- **S&P 500 (SPY):** Broad market benchmark - 500 large-cap US companies\n"
            "- **Nasdaq-100 (QQQ):** Tech-heavy index\n"
            "- **Dow Jones (DIA):** 30 blue-chip industrial companies\n"
            "- **Russell 2000 (IWM):** Small-cap stocks\n\n"
            "### Key Sectors & Representative ETFs\n"
            "- **Technology (XLK):** AAPL, MSFT, NVDA\n"
            "- **Healthcare (XLV):** UNH, JNJ, PFE\n"
            "- **Financials (XLF):** JPM, BAC, GS\n"
            "- **Energy (XLE):** XOM, CVX\n"
            "- **Consumer Discretionary (XLY):** AMZN, TSLA\n\n"
            "For live prices on any of these, ask me: *\"Predict SPY\"* or *\"Technical analysis of QQQ\"*\n\n"
            "**Disclaimer:** This is not financial advice. Always do your own research and consider consulting a financial advisor."
        )

    # Default market advice response
    return (
        "## Market & Investment Insights\n\n"
        "I can help you with:\n\n"
        "- **Top stocks by sector** - Ask: *\"What are the top 10 stocks?\"*\n"
        "- **Investment strategies** - Ask: *\"If I have $500 to invest, what should I buy?\"*\n"
        "- **Hedge funds & ETFs** - Ask: *\"Tell me about hedge fund alternatives\"*\n"
        "- **Sector analysis** - Ask: *\"Which sectors are performing best?\"*\n"
        "- **Specific stock analysis** - Ask: *\"Predict AAPL stock\"* or *\"Technical analysis of TSLA\"*\n\n"
        "For the most detailed analysis, ask about a specific stock symbol and I'll fetch live data with AI predictions.\n\n"
        "**Disclaimer:** This is not financial advice. Always do your own research and consider consulting a financial advisor."
    )


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
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/auth/login")
    @rate_limit_public
    async def login_user(user_data: UserLogin):
        """Login user and return tokens."""
        try:
            tokens = auth_manager.login_user(user_data)
            return tokens
        except HTTPException:
            raise
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

    # NOTE: /portfolio/list must be registered BEFORE /portfolio/{portfolio_id}
    # so that "list" is not captured as a portfolio_id path parameter.
    @app.get("/portfolio/list")
    @rate_limit_authenticated
    async def list_portfolios(current_user: Dict = Depends(get_current_active_user)):
        """List user portfolios."""
        try:
            portfolios = db_manager.get_user_portfolios(current_user["id"])
            return {"portfolios": portfolios}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/portfolio/{portfolio_id}/summary")
    @rate_limit_authenticated
    async def get_portfolio_summary(
        portfolio_id: str,
        current_user: Dict = Depends(get_current_active_user)
    ):
        """Get portfolio with live prices, gain/loss per stock, and totals."""
        try:
            portfolio = db_manager.get_portfolio(portfolio_id)
            if not portfolio or portfolio.get('user_id') != current_user['id']:
                raise HTTPException(status_code=404, detail="Portfolio not found")

            stocks = portfolio.get('stocks', [])
            enriched_stocks = []
            total_invested = 0.0
            total_current_value = 0.0

            for stock in stocks:
                symbol = stock['symbol']
                shares = stock.get('shares', 0)
                purchase_price = stock.get('purchase_price', 0)
                cost_basis = shares * purchase_price
                total_invested += cost_basis

                live_data = fetch_stock_data(symbol)
                current_price = live_data['price'] if live_data else None
                current_value = (shares * current_price) if current_price else None
                gain_loss = (current_value - cost_basis) if current_value is not None else None
                gain_loss_pct = ((gain_loss / cost_basis) * 100) if (gain_loss is not None and cost_basis > 0) else None

                if current_value:
                    total_current_value += current_value

                enriched_stocks.append({
                    'symbol': symbol,
                    'shares': shares,
                    'purchase_price': purchase_price,
                    'purchase_date': stock.get('purchase_date'),
                    'current_price': current_price,
                    'cost_basis': round(cost_basis, 2),
                    'current_value': round(current_value, 2) if current_value else None,
                    'gain_loss': round(gain_loss, 2) if gain_loss is not None else None,
                    'gain_loss_percent': round(gain_loss_pct, 2) if gain_loss_pct is not None else None,
                    'name': live_data.get('name', symbol) if live_data else symbol,
                })

            total_gain_loss = total_current_value - total_invested
            total_gain_loss_pct = ((total_gain_loss / total_invested) * 100) if total_invested > 0 else 0

            return {
                'id': portfolio.get('id'),
                'name': portfolio.get('name'),
                'description': portfolio.get('description'),
                'stocks': enriched_stocks,
                'summary': {
                    'total_invested': round(total_invested, 2),
                    'total_current_value': round(total_current_value, 2),
                    'total_gain_loss': round(total_gain_loss, 2),
                    'total_gain_loss_percent': round(total_gain_loss_pct, 2),
                    'stock_count': len(enriched_stocks),
                }
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Portfolio summary error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

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

    @app.delete("/portfolio/{portfolio_id}/stock/{symbol}")
    @rate_limit_authenticated
    async def delete_stock_from_portfolio_endpoint(
        portfolio_id: str,
        symbol: str,
        current_user: Dict = Depends(get_current_active_user)
    ):
        """Remove a stock from a portfolio."""
        try:
            portfolio = db_manager.get_portfolio(portfolio_id)
            if not portfolio or portfolio.get('user_id') != current_user['id']:
                raise HTTPException(status_code=404, detail="Portfolio not found")
            result = db_manager.delete_stock_from_portfolio(portfolio_id, symbol)
            if not result:
                raise HTTPException(status_code=404, detail=f"Stock {symbol} not found in portfolio")
            return {"message": f"Stock {symbol} removed from portfolio"}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

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
