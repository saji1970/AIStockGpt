#!/usr/bin/env python3
"""
Enhanced FastAPI Backend for AI Stock GPT v2.0
==============================================

This module provides an enhanced FastAPI backend that integrates with the AI stock prediction system
and provides advanced features including user authentication, portfolio management, and enhanced analytics.
"""

import os
import sys
from pathlib import Path

# Load .env before any backend module reads DATABASE_URL (critical for local login)
try:
    from dotenv import load_dotenv

    _repo = Path(__file__).resolve().parent.parent
    load_dotenv(_repo / "backend" / ".env")
    load_dotenv(_repo / ".env")
except ImportError:
    pass

# Before any TensorFlow import (e.g. via lstm_model): drop TF INFO/WARNING noise on CPU hosts.
# Some XLA/CUDA "Unable to register ... factory" ERROR lines can still appear with GPU-enabled TF wheels on CPU-only machines; they are usually harmless.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
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
    from backend.training_retriever import TrainingRetriever
    ENHANCED_NLP_AVAILABLE = True
except ImportError:
    try:
        from nlp_enhanced import EnhancedNLPProcessor
        from llm_provider import LLMProvider, llm_provider
        from training_retriever import TrainingRetriever
        ENHANCED_NLP_AVAILABLE = True
    except ImportError:
        ENHANCED_NLP_AVAILABLE = False
        llm_provider = None
        TrainingRetriever = None
        print("Warning: Enhanced NLP/LLM modules not available")

# Import new enhanced modules
try:
    from backend.database import db_manager
    from backend.email_service import email_service
    from backend.auth import (
        auth_manager, get_current_active_user, UserCreate, UserLogin, Token,
        ChangePasswordRequest, ForgotPasswordRequest, ResetPasswordRequest
    )
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
        from email_service import email_service
        from auth import (
            auth_manager, get_current_active_user, UserCreate, UserLogin, Token,
            ChangePasswordRequest, ForgotPasswordRequest, ResetPasswordRequest
        )
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

    try:
        from backend.admin_routes import router as admin_router
        app.include_router(admin_router)
        logger.info("Admin API routes mounted at /api/admin")
    except ImportError:
        try:
            from admin_routes import router as admin_router
            app.include_router(admin_router)
            logger.info("Admin API routes mounted at /api/admin")
        except ImportError as e:
            logger.warning(f"Admin routes not available: {e}")

# Global variables
nlp_processor = None
training_retriever = None
models_cache = {}
data_collectors = {}

# ML Engine instances
feature_pipeline = None
xgboost_predictor = None
monte_carlo_sim = None
portfolio_optimizer = None
sentiment_analyzer = None
regime_detector = None
av_collector = None
fundamental_collector = None
buffett_scorer = None
jhunjhunwala_scorer = None

# Pydantic models
class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    timestamp: Optional[str] = Field(None, description="Message timestamp")
    session_id: Optional[str] = Field(None, description="Learn session ID for multi-turn @learn")

class ChatResponse(BaseModel):
    message: str = Field(..., description="AI response")
    stockData: Optional[Dict[str, Any]] = Field(None, description="Stock analysis data")
    charts: Optional[Dict[str, Any]] = Field(None, description="Chart data")
    confidence: Optional[float] = Field(None, description="Model confidence")
    session_id: Optional[str] = Field(None, description="Learn session ID (non-null during @learn)")

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

class PortfolioRecommendRequest(BaseModel):
    amount: float = Field(..., description="Investment amount in USD")
    risk_level: str = Field('moderate', description="conservative, moderate, or aggressive")
    horizon_months: int = Field(12, description="Investment horizon in months")
    goals: Optional[str] = Field(None, description="Investment goals")

class RiskAnalysisRequest(BaseModel):
    portfolio_id: Optional[str] = Field(None, description="Existing portfolio ID")
    symbols: Optional[List[str]] = Field(None, description="Stock symbols")
    weights: Optional[List[float]] = Field(None, description="Portfolio weights")

class ForecastRequest(BaseModel):
    symbols: List[str] = Field(..., description="Stock symbols")
    weights: List[float] = Field(..., description="Portfolio weights (must sum to 1)")
    amount: float = Field(..., description="Initial investment amount")
    months: int = Field(12, description="Forecast horizon in months")

def initialize_nlp():
    """Initialize the NLP processor, training retriever, and wire RAG into LLM."""
    global nlp_processor, training_retriever

    # Try enhanced NLP first (sentence-transformers)
    if ENHANCED_NLP_AVAILABLE:
        try:
            nlp_processor = EnhancedNLPProcessor()
            logger.info("Enhanced NLP processor initialized (sentence-transformers)")

            # Initialize TrainingRetriever sharing the NLP model for efficiency
            if TrainingRetriever is not None:
                try:
                    shared_model = nlp_processor.model if nlp_processor.use_ml else None
                    training_retriever = TrainingRetriever(model=shared_model)
                    if training_retriever.ready and llm_provider:
                        llm_provider.set_training_retriever(training_retriever)
                        logger.info("Training retriever wired into LLM provider (RAG enabled)")
                    elif training_retriever.ready:
                        logger.info("Training retriever ready but no LLM provider to attach to")
                    else:
                        logger.warning("Training retriever initialized but not ready")
                except Exception as e:
                    logger.warning(f"Training retriever init failed: {e}")

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

def _to_yf_symbol(symbol: str) -> str:
    """Convert Alpha Vantage / BSE symbol to yfinance equivalent."""
    _map = {'.BSE': '.BO', '.NSE': '.NS'}
    for av_suffix, yf_suffix in _map.items():
        if symbol.upper().endswith(av_suffix):
            return symbol[: -len(av_suffix)] + yf_suffix
    return symbol


def _is_indian_symbol(symbol: str) -> bool:
    """Check if a symbol belongs to an Indian exchange."""
    upper = symbol.upper()
    return upper.endswith('.BSE') or upper.endswith('.NSE')


def _currency_symbol(symbol: str) -> str:
    """Return the currency symbol (₹ or $) for a stock."""
    return '₹' if _is_indian_symbol(symbol) else '$'


def _currency_code(symbol: str) -> str:
    """Return the currency code (INR or USD) for a stock."""
    return 'INR' if _is_indian_symbol(symbol) else 'USD'


def fetch_stock_data(symbol: str) -> Optional[Dict[str, Any]]:
    """Fetch real-time stock data. Alpha Vantage primary, RapidAPI and Yahoo Finance fallbacks."""
    import requests as req

    # Tier 1: Alpha Vantage (premium, most reliable on Railway)
    global av_collector
    if av_collector:
        av_quote = av_collector.get_quote(symbol)
        if av_quote:
            logger.info(f"Stock data fetched via Alpha Vantage for {symbol}")
            return av_quote

    # Tier 2: RapidAPI Real-Time Finance Data
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
                            "currency": _currency_code(symbol),
                            "currencySymbol": _currency_symbol(symbol),
                        }
                logger.warning(f"RapidAPI returned {resp.status_code} for {symbol} (attempt {attempt+1})")
            except Exception as e:
                logger.warning(f"RapidAPI stock fetch failed for {symbol} (attempt {attempt+1}): {e}")

    # Tier 3: Direct Yahoo Finance v8 API (no library needed)
    yf_sym = _to_yf_symbol(symbol)
    try:
        yahoo_url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yf_sym}"
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
                    "currency": _currency_code(symbol),
                    "currencySymbol": _currency_symbol(symbol),
                }
                logger.info(f"Stock data fetched via Yahoo Finance v8 API for {symbol}")
                return result
        logger.warning(f"Yahoo Finance v8 API returned {resp.status_code} for {symbol}")
    except Exception as e:
        logger.warning(f"Yahoo Finance v8 API failed for {symbol}: {e}")

    # Fallback: yfinance library
    try:
        import yfinance as yf
        ticker = yf.Ticker(yf_sym)
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
            "currency": _currency_code(symbol),
            "currencySymbol": _currency_symbol(symbol),
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


_GROWTH_VALUE_SCREENING_PATTERNS = [
    r"(?:which|what)\s+stock(?:s)?.*(?:growth|cheap|undervalued|promising|low[-\s]?cost|affordable|inexpensive)",
    r"(?:promising|strong)\s+growth.*(?:low|cheap|cost|affordable|undervalued|price)",
    r"(?:cheap|low[-\s]?cost|affordable|undervalued|inexpensive).*(?:growth|growing|potential|upside)",
    r"(?:growth|growing).*(?:cheap|low[-\s]?cost|affordable|undervalued|low\s+price)",
    r"(?:best|good|interesting)\s+(?:cheap|low[-\s]?cost).*(?:growth|grow)",
    r"(?:under|below)\s*\$?\s*\d+.*(?:growth|grow)",
]


def _is_growth_value_screening_message(text: str) -> bool:
    """Broad stock-picking questions (growth + low cost / cheap) — not a single-ticker TA request."""
    t = (text or "").lower()
    if not t:
        return False
    return any(re.search(p, t) for p in _GROWTH_VALUE_SCREENING_PATTERNS)


# ---------------------------------------------------------------------------
# Startup / high-growth stock screening
# ---------------------------------------------------------------------------
_STARTUP_SCREENING_PATTERNS = [
    r"\bstartup\b.*\b(?:stock|invest|compan|equit)",
    r"\b(?:stock|invest|compan|equit).*\bstartup",
    r"\b(?:high[-\s]?growth|hyper[-\s]?growth|emerging)\s+(?:stock|compan|pick|ticker|name)",
    r"\b(?:stock|compan|pick|ticker|name).*\b(?:high[-\s]?growth|hyper[-\s]?growth|emerging)",
    r"\b(?:early[-\s]?stage|pre[-\s]?profit|disruptive|disruptor).*\b(?:stock|invest|compan)",
    r"\b(?:next\s+big|moonshot|10x|tenbagger|multi[-\s]?bagger).*\b(?:stock|invest|compan|pick)",
    r"\b(?:stock|invest|compan|pick).*\b(?:moonshot|10x|tenbagger|multi[-\s]?bagger)",
    r"\b(?:small[-\s]?cap|micro[-\s]?cap|mid[-\s]?cap)\s+(?:growth|stock|pick|idea)",
]

# Broader "stock ideas / what to invest" questions (no amount, no specific ticker)
_STOCK_IDEAS_PATTERNS = [
    r"(?:which|what)\s+(?:stock|stocks|companies|tickers)\s+(?:to|should\s+i|can\s+i|do\s+you\s+recommend)\s+(?:invest|buy|pick|hold)",
    r"(?:which|what)\s+(?:are|is)\s+(?:the\s+)?(?:best|good|top|hot|trending)\s+(?:stock|stocks|companies)\s+(?:to\s+)?(?:invest|buy|pick|hold)?",
    r"(?:recommend|suggest)\s+(?:me\s+)?(?:some\s+)?(?:stock|stocks|companies|tickers)\s+(?:to\s+)?(?:invest|buy)",
    r"(?:best|good|top|hot|trending)\s+(?:stock|stocks|companies)\s+(?:to\s+)?(?:invest|buy|pick)",
    r"(?:where|what)\s+(?:should\s+i|to)\s+invest\s+(?:in\s+)?(?:right\s+now|today|now|this\s+year)",
]


def _is_startup_screening_message(text: str) -> bool:
    """Startup / high-growth stock questions."""
    t = (text or "").lower()
    if not t:
        return False
    return any(re.search(p, t) for p in _STARTUP_SCREENING_PATTERNS)


def _is_stock_ideas_message(text: str) -> bool:
    """Broad 'what stocks to invest in' without a dollar amount or specific ticker."""
    t = (text or "").lower()
    if not t:
        return False
    # Only trigger if there is NO dollar/rupee amount already in the message
    if re.search(r"(?:\$|₹|usd|inr)\s*\d", t) or re.search(r"\d+\s*(?:dollars?|rupees?|usd|inr)", t):
        return False
    return any(re.search(p, t) for p in _STOCK_IDEAS_PATTERNS)


# ---------------------------------------------------------------------------
# @learn command detection & inline parsing
# ---------------------------------------------------------------------------
import re as _re_learn

def _is_learn_command(text: str) -> bool:
    """Detect @learn at the start of a message or after whitespace."""
    return bool(re.search(r'(?:^|\s)@learn\b', (text or ''), re.IGNORECASE))

def _is_cancel_command(text: str) -> bool:
    """Detect cancel/quit/exit during a @learn session."""
    t = (text or '').strip().lower()
    return t in ('cancel', 'quit', 'exit', '@cancel', 'stop', 'nevermind')

def _extract_inline_learn_info(text: str) -> Dict[str, Any]:
    """Parse as much as possible from a single @learn message.
    e.g. '@learn AAPL predicted bullish last week but dropped, earnings miss'
    Returns dict with keys: symbols, direction_error, date_hint, context_hint
    """
    t = (text or '')
    # Remove the @learn prefix
    t_clean = re.sub(r'(?:^|\s)@learn\s*', ' ', t, flags=re.IGNORECASE).strip()

    info: Dict[str, Any] = {
        'symbols': [],
        'direction_error': None,
        'date_hint': None,
        'context_hint': None,
    }

    # Extract symbols (1-5 uppercase letters, optionally with .BSE, =X, =F, -USD suffixes)
    sym_pattern = r'\b([A-Z]{1,5}(?:\.[A-Z]{2,4})?(?:[-=][A-Z]+)?)\b'
    # Exclude common English words that look like tickers
    _STOP_WORDS = {'I', 'A', 'IT', 'IS', 'AT', 'IN', 'ON', 'TO', 'DO', 'GO',
                    'UP', 'SO', 'IF', 'OR', 'AN', 'AS', 'BY', 'WE', 'MY', 'NO',
                    'OF', 'BE', 'HE', 'ME', 'OK', 'THE', 'BUT', 'AND', 'FOR',
                    'NOT', 'YOU', 'ALL', 'CAN', 'HAS', 'HER', 'WAS', 'ONE',
                    'OUR', 'OUT', 'ARE', 'HIS', 'HAD', 'HOW', 'ITS', 'MAY',
                    'NEW', 'NOW', 'OLD', 'SEE', 'WAY', 'WHO', 'DID', 'GET',
                    'HIM', 'LET', 'SAY', 'SHE', 'TOO', 'USE', 'WENT', 'SAID',
                    'WEEK', 'LAST', 'WHEN', 'WHAT', 'WRONG', 'DROP', 'FELL',
                    'MISS', 'DOWN', 'ROSE', 'SKIP', 'NONE', 'YES'}
    for m in re.finditer(sym_pattern, t):
        sym = m.group(1)
        if sym not in _STOP_WORDS and len(sym) >= 2:
            info['symbols'].append(sym)

    # Direction error
    t_lower = t_clean.lower()
    if re.search(r'bullish.*(?:drop|fell|went\s+(?:down|bearish)|bear|crash|lost|decline)', t_lower):
        info['direction_error'] = 'bullish_went_bearish'
    elif re.search(r'bearish.*(?:rose|went\s+(?:up|bullish)|bull|gain|rally|surge)', t_lower):
        info['direction_error'] = 'bearish_went_bullish'

    # Date hints
    if re.search(r'last\s+week', t_lower):
        info['date_hint'] = 'last_week'
    elif re.search(r'last\s+month', t_lower):
        info['date_hint'] = 'last_month'
    elif re.search(r'yesterday', t_lower):
        info['date_hint'] = 'yesterday'
    else:
        date_match = re.search(r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})', t_clean)
        if date_match:
            info['date_hint'] = date_match.group(1)

    # Context hint: anything after a comma or "because" / "due to"
    ctx_match = re.search(r'(?:because|due\s+to|,)\s*(.+?)$', t_clean, re.IGNORECASE)
    if ctx_match:
        ctx = ctx_match.group(1).strip().rstrip('.')
        if len(ctx) > 2:
            info['context_hint'] = ctx

    return info


def _format_growth_value_screening_markdown(india: bool) -> str:
    """ChatGPT-style structured answer: criteria + table + categories + follow-ups."""
    if india:
        return (
            "## Growth-oriented ideas (India)\n\n"
            "If you want **growth with a lower absolute share price**, screen for "
            "**revenue growth**, **margin improvement**, and **valuation vs growth**—not share price alone.\n\n"
            "| Stock | Why it can look interesting | Risk |\n"
            "|------|-----------------------------|------|\n"
            "| **INFY.BSE** (Infosys) | IT scale; AI/digital deal narrative | Medium |\n"
            "| **HDFCBANK.BSE** (HDFC Bank) | Retail banking growth; strong franchise | Medium |\n"
            "| **RELIANCE.BSE** (Reliance) | Conglomerate optionality (energy, retail, telecom) | Medium–High |\n"
            "| **TCS.BSE** (TCS) | Large IT; cash generation | Medium |\n"
            "| **BAJFINANCE.BSE** (Bajaj Finance) | Consumer/lending growth; often premium valuation | Medium–High |\n"
            "| **LT.BSE** (L&T) | Infra / engineering cycle exposure | Medium |\n\n"
            "### Ask next in AI Stock GPT\n"
            "For **live price + ML context**, name a ticker: *\"Predict INFY.BSE\"* or *\"Technical analysis of HDFCBANK.BSE\"*.\n\n"
            "---\n\n"
            "*Not financial advice. Do your own research.*"
        )
    return (
        "## Growth vs \"low cost\" (US)\n\n"
        "If you want **growth at a relatively lower share price** (or **better value vs growth**), "
        "investors often look for **revenue growth**, **profitability inflection**, **theme tailwinds** "
        "(AI, fintech, semis), and **valuation vs peers**.\n\n"
        "Here is an **illustrative** shortlist (not a buy ranking):\n\n"
        "| Stock | Why it looks interesting | Risk level |\n"
        "|------|---------------------------|------------|\n"
        "| **SOFI** (SoFi Technologies) | Fintech + banking ecosystem; often cited as a lower-priced growth name | Medium–High |\n"
        "| **MU** (Micron) | AI/memory demand; discussed as relatively cheaper vs some AI leaders | Medium |\n"
        "| **MRVL** (Marvell) | AI networking / custom silicon | Medium–High |\n"
        "| **QCOM** (Qualcomm) | Edge AI + handsets + automotive; often less extreme mega-cap multiples | Medium |\n"
        "| **PYPL** (PayPal) | Turnaround + commerce / payments cyclicality | Medium |\n"
        "| **AMD** (AMD) | AI accelerator + CPU narrative vs larger peers | Medium–High |\n\n"
        "### By angle (how people *talk* about buckets — not advice)\n"
        "- **Lower-priced fintech / digital growth**: **SOFI**\n"
        "- **AI memory at a relatively cheaper bar vs some leaders**: **MU**\n"
        "- **AI connectivity / silicon**: **MRVL**\n"
        "- **Cash flow + devices + automotive optionality**: **QCOM**\n\n"
        "### Follow-ups you can ask\n"
        "- \"Best stocks under $20\"\n"
        "- \"AI stocks with upside but cheaper than NVDA\"\n"
        "- \"Conservative vs aggressive $5,000 growth portfolio\"\n\n"
        "---\n\n"
        "*Not financial advice. Prices and fundamentals change—verify before acting.*"
    )


def _format_startup_screening_markdown(india: bool) -> str:
    """Startup / high-growth stock ideas with specific analysis, risk levels, and strategy buckets."""
    if india:
        return (
            "## Startup & High-Growth Stock Ideas (India)\n\n"
            "If you want **startup-style growth** with higher upside potential, look for companies with "
            "**rapid revenue growth**, **market disruption**, **expanding TAM**, and **strong management**.\n\n"
            "| Stock | Why it's interesting | Risk Level |\n"
            "|------|----------------------|------------|\n"
            "| **ZOMATO.BSE** (Zomato) | Food-delivery + quick-commerce (Blinkit); turning profitable with massive scale | Medium–High |\n"
            "| **PAYTM.BSE** (One97 / Paytm) | Fintech payments + lending; restructured, improving path to profitability | High |\n"
            "| **NYKAA.BSE** (FSN E-Commerce) | Beauty & fashion e-commerce; strong brand + D2C growth | Medium–High |\n"
            "| **POLICYBZR.BSE** (PB Fintech) | Insurtech leader; digital insurance distribution at scale | High |\n"
            "| **DELHIVERY.BSE** (Delhivery) | Logistics-tech; benefiting from e-commerce boom | High |\n"
            "| **MAPMYINDIA.BSE** (C.E. Info Systems) | India's mapping & location-tech; AI + autonomous driving play | Medium–High |\n\n"
            "### Strategy buckets\n"
            "- **Safer growth**: ZOMATO, NYKAA\n"
            "- **Aggressive high-upside**: PAYTM, POLICYBZR\n"
            "- **Speculative moonshot**: DELHIVERY, MAPMYINDIA\n\n"
            "### Practical allocation idea (example only)\n"
            "- 40% in a stronger core growth stock (ZOMATO)\n"
            "- 40% split across 2–3 emerging growth names\n"
            "- 20% in speculative bets\n\n"
            "### Ask next\n"
            "- *\"Predict ZOMATO.BSE\"* for ML price forecast\n"
            "- *\"Technical analysis of PAYTM.BSE\"* for indicators\n\n"
            "---\n\n"
            "*Not financial advice. Startup stocks carry higher volatility—do your own research.*"
        )
    return (
        "## Startup & High-Growth Stock Ideas (US)\n\n"
        "If you want **startup-style growth** with higher upside potential, focus on "
        "**AI infrastructure**, **fintech**, **defense tech**, and **emerging software** "
        "rather than pure speculative penny stocks.\n\n"
        "| Stock | Why it's interesting | Risk Level |\n"
        "|------|----------------------|------------|\n"
        "| **PLTR** (Palantir Technologies) | Strong AI + government + enterprise growth; becoming core AI infrastructure for defense and enterprise analytics. Expensive valuation, but momentum is strong | Medium–High |\n"
        "| **PGNY** (Pagaya Technologies) | AI-based lending/fintech platform; fast revenue growth and still relatively small compared to larger fintechs | High |\n"
        "| **NVTS** (Navitas Semiconductor) | Power semiconductors tied to AI data centers and next-gen energy systems; big long-term upside if AI infra demand continues | High |\n"
        "| **FRSH** (Freshworks) | Profitable SaaS company with AI customer-support tools; more stable than many startups | Medium |\n"
        "| **OPRA** (Opera Limited) | Browser + AI assistant + gaming ecosystem; still underfollowed and profitable | Medium |\n"
        "| **BBAI** (BigBear.ai Holdings) | AI + defense analytics; highly speculative but can move aggressively during AI rallies | Very High |\n"
        "| **INOD** (Innodata) | Provides AI training and data engineering for LLMs; smaller company benefiting from the AI boom | High |\n"
        "| **RKLB** (Rocket Lab) | Space infrastructure + satellite launch; high growth, high volatility | High |\n\n"
        "### Strategy buckets\n"
        "| Category | Stocks |\n"
        "|----------|--------|\n"
        "| **Safer growth** | PLTR, FRSH |\n"
        "| **Aggressive high-upside** | PGNY, NVTS, INOD |\n"
        "| **Speculative moonshot** | BBAI, RKLB |\n\n"
        "### Practical allocation strategy (example only)\n"
        "- **40%** in a stronger core growth stock (PLTR / FRSH)\n"
        "- **40%** split across 2–3 emerging growth names\n"
        "- **20%** in speculative AI / small-cap bets\n\n"
        "Avoid putting everything into one stock—most fail even if the sector trend is right.\n\n"
        "### Ask next\n"
        "- *\"Predict PLTR\"* for ML-based price forecast\n"
        "- *\"Technical analysis of NVTS\"* for RSI, MACD, and more\n"
        "- *\"Best stocks under $20\"* for price-filtered ideas\n\n"
        "---\n\n"
        "*Not financial advice. Startup stocks carry higher volatility—always verify fundamentals before acting.*"
    )


def _format_stock_ideas_screening_markdown(india: bool) -> str:
    """General 'what stocks to invest in' response with categorized ideas."""
    if india:
        return (
            "## Top Stock Ideas to Consider (India)\n\n"
            "Here are **specific stocks** across different categories based on current market themes:\n\n"
            "### Blue-chip growth\n"
            "| Stock | Why it's interesting | Risk |\n"
            "|------|----------------------|------|\n"
            "| **RELIANCE.BSE** (Reliance) | Conglomerate optionality: energy, retail, telecom (Jio) | Medium |\n"
            "| **HDFCBANK.BSE** (HDFC Bank) | Largest private bank; strong retail franchise | Medium |\n"
            "| **INFY.BSE** (Infosys) | IT services; AI & digital transformation play | Medium |\n\n"
            "### High-growth / emerging\n"
            "| Stock | Why it's interesting | Risk |\n"
            "|------|----------------------|------|\n"
            "| **ZOMATO.BSE** (Zomato) | Food delivery + quick commerce; path to profitability | Medium–High |\n"
            "| **BAJFINANCE.BSE** (Bajaj Finance) | Consumer lending growth; premium valuation | Medium–High |\n"
            "| **LT.BSE** (L&T) | Infrastructure + engineering cycle exposure | Medium |\n\n"
            "### Ask next\n"
            "- *\"Predict RELIANCE.BSE\"* for ML forecast\n"
            "- *\"What startup stocks to invest\"* for higher-risk growth ideas\n\n"
            "---\n\n"
            "*Not financial advice. Do your own research.*"
        )
    return (
        "## Top Stock Ideas to Consider (US)\n\n"
        "Here are **specific stocks** across different categories based on current market themes:\n\n"
        "### Large-cap growth leaders\n"
        "| Stock | Why it's interesting | Risk |\n"
        "|------|----------------------|------|\n"
        "| **NVDA** (NVIDIA) | AI chip leader; data center + inference demand | Medium |\n"
        "| **MSFT** (Microsoft) | Cloud (Azure) + AI integration (Copilot) | Medium |\n"
        "| **AAPL** (Apple) | Services growth + massive cash generation | Low–Medium |\n\n"
        "### Mid-cap growth with upside\n"
        "| Stock | Why it's interesting | Risk |\n"
        "|------|----------------------|------|\n"
        "| **PLTR** (Palantir) | AI + government + enterprise analytics momentum | Medium–High |\n"
        "| **SOFI** (SoFi Technologies) | Fintech + banking ecosystem; member growth | Medium–High |\n"
        "| **CRWD** (CrowdStrike) | Cybersecurity leader; AI-driven threat detection | Medium |\n\n"
        "### Emerging / higher-risk\n"
        "| Stock | Why it's interesting | Risk |\n"
        "|------|----------------------|------|\n"
        "| **RKLB** (Rocket Lab) | Space infrastructure; government + commercial launch | High |\n"
        "| **NVTS** (Navitas) | Power semiconductors for AI data centers | High |\n\n"
        "### By investment style\n"
        "| Style | Stocks |\n"
        "|-------|--------|\n"
        "| **Conservative** | AAPL, MSFT |\n"
        "| **Growth** | NVDA, PLTR, CRWD |\n"
        "| **Aggressive** | SOFI, RKLB, NVTS |\n\n"
        "### Ask next\n"
        "- *\"What startup stocks to invest\"* for high-growth / startup ideas\n"
        "- *\"Predict PLTR\"* for ML-based price forecast\n"
        "- *\"Invest $5000 in aggressive stocks\"* for a specific allocation plan\n\n"
        "---\n\n"
        "*Not financial advice. Always verify fundamentals before acting.*"
    )


def _parse_under_price_screening_cap_usd(text: str) -> Optional[float]:
    """Detect 'best stock under $X / below X USD' style questions; return price cap or None."""
    tl = (text or "").lower()
    if not tl:
        return None
    if re.search(r"\b(india|inr|rupee|₹|nifty|sensex|bse|nse)\b", tl) and not re.search(
        r"\b(usd|dollar|bucks|u\.s\.|nyse|nasdaq)\b", tl
    ):
        return None
    if not re.search(r"\b(?:stock|stocks|ticker|pick|invest|investing|equit|compan(?:y|ies)|names?)\b", tl):
        return None
    caps: List[float] = []
    for rx in (
        r"\b(?:under|below|less\s+than|cheaper\s+than)\s*\$\s*(\d+(?:\.\d+)?)\b",
        r"\b(?:under|below|less\s+than|cheaper\s+than)\s*(\d+(?:\.\d+)?)\s*(?:usd|dollars?|bucks)\b",
    ):
        m = re.search(rx, tl)
        if m:
            caps.append(float(m.group(1)))
    if not caps:
        return None
    cap = min(caps)
    if cap < 1 or cap > 2500:
        return None
    return cap


def _parse_under_price_screening_cap_inr(text: str) -> Optional[float]:
    """India: share price under ₹X / X rupees with stock-picking context."""
    tl = (text or "").lower()
    if not tl or not re.search(r"\b(india|inr|rupee|₹|nifty|sensex|bse|nse)\b", tl):
        return None
    if not re.search(r"\b(?:stock|stocks|ticker|pick|invest|investing)\b", tl):
        return None
    caps: List[float] = []
    for rx in (
        r"\b(?:under|below|less\s+than|cheaper\s+than)\s*₹\s*(\d+(?:\.\d+)?)\b",
        r"\b(?:under|below|less\s+than|cheaper\s+than)\s*(\d+(?:\.\d+)?)\s*(?:rupees?|inr|rs\.?)\b",
    ):
        m = re.search(rx, tl)
        if m:
            caps.append(float(m.group(1)))
    if not caps:
        return None
    cap = min(caps)
    if cap < 10 or cap > 500000:
        return None
    return cap


def _format_under_price_screening_usd(cap: float) -> str:
    cap_disp = f"{cap:.0f}" if float(cap).is_integer() else f"{cap:g}"
    return (
        f"## Stocks under **${cap_disp}** (illustrative ideas)\n\n"
        "People often screen for **growth**, **improving fundamentals**, **theme tailwinds** "
        "(fintech, AI, SaaS), and **liquidity**—then check that the **live quote** is still under your cap.\n\n"
        "### Top name frequently discussed in this bucket\n"
        "**SoFi (SOFI)** — digital banking + investing ecosystem; often cited in “under $20 growth” threads "
        "because of member growth and expanding product surface (not a recommendation).\n\n"
        "### Other names people group in “under $20” conversations\n"
        "| Stock | Theme / why it comes up | Risk (typical) |\n"
        "|------|-------------------------|----------------|\n"
        "| **NVTS** (Navitas) | Power / GaN tied to data-center & AI power chains | Medium–High |\n"
        "| **FRSH** (Freshworks) | Profitable SaaS; growth vs larger software multiples | Medium |\n"
        "| **BBAI** (BigBear.ai) | Defense + AI; very speculative | High |\n"
        "| **RKLB** (Rocket Lab) | Space infrastructure; high volatility | High |\n"
        "| **F** (Ford) | More mature; often “lower drama” than pure growth | Medium |\n\n"
        "### By style (conversation-style buckets — not advice)\n"
        "| Style | Name often mentioned |\n"
        "|-------|----------------------|\n"
        "| Balanced growth / fintech | SOFI |\n"
        "| AI / power semis angle | NVTS |\n"
        "| SaaS growth | FRSH |\n"
        "| Aggressive / speculative | BBAI |\n"
        "| Thematic “moonshot” | RKLB |\n\n"
        "### If you want **lower volatility** than small growth\n"
        "Larger caps like **QCOM** sometimes trade closer to this range during drawdowns—still verify the current price vs your **$"
        f"{cap_disp}** cap.\n\n"
        "### Optional “basket” framing (example only)\n"
        "Some discussions split exposure across themes—for example **40% SOFI / 25% FRSH / 20% NVTS / 15% RKLB** "
        "as a *conceptual* mix (not a managed portfolio).\n\n"
        "Ask next: *“Quote SOFI”* or *“Predict NVTS”* for live price + ML context in this app.\n\n"
        "---\n\n"
        "*Not financial advice. Tickers can move above your cap quickly—always verify price and fundamentals.*"
    )


def _format_under_price_screening_inr(cap: float) -> str:
    cap_disp = f"{cap:,.0f}"
    return (
        f"## Stocks under **₹{cap_disp}** per share (India — illustrative)\n\n"
        "Low **share price** ≠ “cheap” valuation—check **P/E, earnings quality, and debt** before acting.\n\n"
        "| Stock | Why it sometimes appears in “low price” screens | Risk |\n"
        "|------|--------------------------------------------------|------|\n"
        "| **SUZLON.BSE** | Renewable / infra theme; volatile | High |\n"
        "| **NHPC.BSE** | Utilities / yield angle; often lower nominal price | Medium |\n"
        "| **PNB.BSE** | Large PSU bank; deep value narratives; asset quality cycles | Medium–High |\n"
        "| **NMDC.BSE** | Commodities / mining cycle | Medium–High |\n"
        "| **ITC.BSE** | Consumer + cash generation; often used as a steadier large-cap example | Medium |\n\n"
        "For **live quotes + ML**, ask: *“Quote ITC.BSE”* or *“Predict NHPC.BSE”*.\n\n"
        "---\n\n"
        "*Not financial advice.*"
    )


# ---------------------------------------------------------------------------
# @learn multi-turn handler
# ---------------------------------------------------------------------------
from backend.learn_session import (
    create_session, get_session, update_session, delete_session, session_exists,
)

def _is_trained_symbol(symbol: str) -> bool:
    """Check if a trained model exists for the symbol."""
    safe = symbol.replace('=', '_').replace('^', '_')
    return os.path.exists(os.path.join('models', f'{safe}_xgboost.joblib'))


def _resolve_date_range(hint: Optional[str]):
    """Convert a date hint string into (start_date, end_date) strings."""
    from datetime import date, timedelta as _td
    today = date.today()
    if not hint or hint == 'last_month':
        start = today - _td(days=30)
        return str(start), str(today)
    if hint == 'last_week':
        start = today - _td(days=7)
        return str(start), str(today)
    if hint == 'yesterday':
        yest = today - _td(days=1)
        return str(yest), str(today)
    # Try parsing YYYY-MM-DD
    try:
        from datetime import datetime as _dt
        d = _dt.strptime(hint, '%Y-%m-%d').date()
        return str(d), str(today)
    except Exception:
        # Default to last 30 days
        start = today - _td(days=30)
        return str(start), str(today)


def _learn_do_retrospect(sess: Dict) -> str:
    """Build a retrospective analysis for the symbol(s) in the session."""
    lines = []
    for symbol in sess.get('symbols', []):
        try:
            # Build features to get actual price data and current prediction
            features = feature_pipeline.build_features(symbol, include_live_bar=True)

            date_range = sess.get('date_range')
            if date_range:
                start_str, end_str = date_range
            else:
                start_str, end_str = _resolve_date_range(None)

            import pandas as _pd
            start_dt = _pd.Timestamp(start_str)
            end_dt = _pd.Timestamp(end_str)

            # Get price data in the date range
            mask = (features.index >= start_dt) & (features.index <= end_dt)
            period_data = features.loc[mask]

            if period_data.empty:
                lines.append(f"### {symbol}\nNo data available for {start_str} to {end_str}.\n")
                continue

            # Actual move
            start_close = float(period_data['close'].iloc[0])
            end_close = float(period_data['close'].iloc[-1])
            actual_return = (end_close - start_close) / start_close
            actual_dir = 'bullish (up)' if actual_return >= 0 else 'bearish (down)'

            # Current model prediction
            prediction = xgboost_predictor.predict(symbol, features)
            model_dir = prediction['direction']
            model_prob = prediction['probability']
            top_features = prediction.get('feature_importance', {})

            # Direction error context
            dir_error = sess.get('direction_error', '')
            if dir_error == 'bullish_went_bearish':
                error_desc = 'Model predicted **bullish** but price went **bearish**'
            elif dir_error == 'bearish_went_bullish':
                error_desc = 'Model predicted **bearish** but price went **bullish**'
            else:
                error_desc = f'Model predicted **{model_dir}** — actual was **{actual_dir}**'

            # Identify potentially misleading features
            misleading = []
            feat_names = list(top_features.keys())[:5]
            for fn in feat_names:
                if fn in period_data.columns:
                    val = float(period_data[fn].iloc[-1]) if not period_data[fn].isna().all() else None
                    if val is not None:
                        misleading.append(f"  - **{fn}** = {val:.4f} (importance: {top_features[fn]:.4f})")

            user_ctx = sess.get('user_context')
            ctx_note = ''
            if user_ctx and user_ctx.lower() not in ('skip', 'none', 'n/a'):
                ctx_note = (
                    f"\n**Your context:** {user_ctx} — the model currently has no explicit "
                    f"event/catalyst feature, so it couldn't anticipate this.\n"
                )

            lines.append(
                f"### Retrospective for {symbol} ({start_str} to {end_str})\n\n"
                f"- {error_desc}\n"
                f"- Current model signal: **{model_dir}** ({model_prob:.0%} probability)\n"
                f"- Actual price move over period: **{actual_return:+.2%}** "
                f"(${start_close:,.2f} → ${end_close:,.2f})\n\n"
                f"**Top features that may have misled the model:**\n"
                + ('\n'.join(misleading) if misleading else '  - (feature data not available)')
                + '\n'
                + ctx_note
            )

        except Exception as e:
            lines.append(f"### {symbol}\nCould not run retrospective: {e}\n")

    return '\n'.join(lines)


def _trigger_retrain(symbols: list) -> str:
    """POST to the training pipeline server to retrain specific symbols."""
    import urllib.request
    import urllib.error
    syms_csv = ','.join(symbols)
    url = f'http://127.0.0.1:8090/train?symbols={syms_csv}'
    try:
        req = urllib.request.Request(url, method='POST')
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        return (
            f"Retraining **{syms_csv}** triggered successfully.\n"
            f"Check progress: `curl http://localhost:8090/status`"
        )
    except urllib.error.URLError:
        return (
            f"Could not reach the training server at `localhost:8090`.\n"
            f"Start it with `python train_pipe.py`, then retry."
        )
    except Exception as e:
        return f"Retrain request failed: {e}"


def _save_learn_feedback(sess: Dict, retrained: bool) -> None:
    """Append feedback record to models/learn_feedback.json."""
    record = {
        'timestamp': datetime.now().isoformat(),
        'symbols': sess.get('symbols', []),
        'direction_error': sess.get('direction_error'),
        'date_range': sess.get('date_range'),
        'user_context': sess.get('user_context'),
        'retrained': retrained,
    }
    fb_path = os.path.join('models', 'learn_feedback.json')
    try:
        existing = []
        if os.path.exists(fb_path):
            with open(fb_path, 'r') as f:
                existing = json.load(f)
        existing.append(record)
        with open(fb_path, 'w') as f:
            json.dump(existing, f, indent=2, default=str)
    except Exception as e:
        logger.warning(f"Failed to save learn feedback: {e}")


def _handle_learn_message(message: str, user_id: Optional[str] = None,
                          session_id: Optional[str] = None) -> Dict[str, Any]:
    """Multi-turn @learn conversation handler. Returns a ChatResponse-shaped dict."""

    key = session_id or user_id or None
    sess = get_session(key) if key else None

    # Cancel at any step
    if sess and _is_cancel_command(message):
        delete_session(key)
        return {
            'message': 'Learning session cancelled.',
            'stockData': None, 'charts': None, 'confidence': None, 'session_id': None,
        }

    # New @learn command — create session and try to parse inline info
    if sess is None:
        inline = _extract_inline_learn_info(message)
        key, sess = create_session(key)

        # Pre-fill whatever was parsed inline
        if inline['symbols']:
            valid = [s for s in inline['symbols'] if _is_trained_symbol(s)]
            if valid:
                sess['symbols'] = valid
                sess['step'] = 'ask_direction'
        if inline['direction_error'] and sess['step'] == 'ask_direction':
            sess['direction_error'] = inline['direction_error']
            sess['step'] = 'ask_timeframe'
        if inline['date_hint'] and sess['step'] == 'ask_timeframe':
            sess['date_range'] = _resolve_date_range(inline['date_hint'])
            sess['step'] = 'ask_context'
        if inline['context_hint'] and sess['step'] == 'ask_context':
            sess['user_context'] = inline['context_hint']
            sess['step'] = 'confirm_retrain'

    step = sess['step']

    # ── ask_symbol ────────────────────────────────────────────
    if step == 'ask_symbol':
        # Try extracting symbols from the current message
        sym_pattern = r'\b([A-Z]{1,5}(?:\.[A-Z]{2,4})?(?:[-=][A-Z]+)?)\b'
        _STOP = {'I', 'A', 'IT', 'IS', 'AT', 'IN', 'ON', 'TO', 'DO', 'UP',
                 'SO', 'IF', 'OR', 'AN', 'AS', 'BY', 'WE', 'MY', 'NO', 'OF',
                 'BE', 'OK', 'THE', 'BUT', 'AND', 'FOR', 'NOT', 'YOU', 'ALL',
                 'CAN', 'HAS', 'WAS', 'ONE', 'ARE', 'HOW', 'ITS', 'MAY',
                 'NOW', 'WHO', 'DID', 'GET', 'LET', 'SAY', 'TOO', 'USE',
                 'YES', 'WHAT', 'WHEN', 'LAST', 'WEEK', 'NONE', 'SKIP'}
        found = [m.group(1) for m in re.finditer(sym_pattern, message)
                 if m.group(1) not in _STOP and len(m.group(1)) >= 2]
        valid = [s for s in found if _is_trained_symbol(s)]

        if valid:
            update_session(key, {'symbols': valid, 'step': 'ask_direction'})
            syms_str = ', '.join(valid)
            return {
                'message': (
                    f"Got it — looking at **{syms_str}**.\n\n"
                    f"What happened? Did the model predict bullish when it went bearish, "
                    f"or bearish when it went bullish?\n\n"
                    f"*(bullish went bearish / bearish went bullish)*"
                ),
                'stockData': None, 'charts': None, 'confidence': None,
                'session_id': key,
            }
        else:
            return {
                'message': (
                    "Which symbol(s) did the model get wrong?\n\n"
                    "Type the ticker symbol(s) — e.g. **AAPL**, **TSLA**, **BTC-USD**.\n\n"
                    "*(Type 'cancel' to exit)*"
                ),
                'stockData': None, 'charts': None, 'confidence': None,
                'session_id': key,
            }

    # ── ask_direction ─────────────────────────────────────────
    if step == 'ask_direction':
        t = message.lower()
        direction = None
        if re.search(r'bullish.*bear|up.*down|bull.*drop|bull.*fell|bull.*crash', t):
            direction = 'bullish_went_bearish'
        elif re.search(r'bearish.*bull|down.*up|bear.*rose|bear.*rally|bear.*gain', t):
            direction = 'bearish_went_bullish'
        elif 'bullish' in t or 'up' in t or 'rose' in t:
            # If they just say the actual outcome
            direction = 'bearish_went_bullish'
        elif 'bearish' in t or 'down' in t or 'drop' in t or 'fell' in t:
            direction = 'bullish_went_bearish'

        if direction:
            update_session(key, {'direction_error': direction, 'step': 'ask_timeframe'})
            return {
                'message': (
                    "When did this happen?\n\n"
                    "Examples: **last week**, **last month**, **yesterday**, "
                    "**2026-05-10**, or a range like **May 5-12**.\n\n"
                    "*(Type 'cancel' to exit)*"
                ),
                'stockData': None, 'charts': None, 'confidence': None,
                'session_id': key,
            }
        else:
            return {
                'message': (
                    "I didn't catch the direction. What happened?\n\n"
                    "- The model said **bullish** but the price **dropped** (bearish)\n"
                    "- The model said **bearish** but the price **rose** (bullish)\n\n"
                    "*(Type one of the options above, or 'cancel' to exit)*"
                ),
                'stockData': None, 'charts': None, 'confidence': None,
                'session_id': key,
            }

    # ── ask_timeframe ─────────────────────────────────────────
    if step == 'ask_timeframe':
        t = message.lower().strip()
        hint = None
        if 'last week' in t:
            hint = 'last_week'
        elif 'last month' in t:
            hint = 'last_month'
        elif 'yesterday' in t:
            hint = 'yesterday'
        else:
            date_match = re.search(r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})', message)
            if date_match:
                hint = date_match.group(1)
            elif t in ('recent', 'recently', 'this week', 'past week'):
                hint = 'last_week'
            else:
                hint = 'last_month'  # Default

        date_range = _resolve_date_range(hint)
        update_session(key, {'date_range': date_range, 'step': 'ask_context'})
        return {
            'message': (
                f"Looking at **{date_range[0]}** to **{date_range[1]}**.\n\n"
                f"Any context you noticed? (earnings miss, sector sell-off, "
                f"news event, Fed announcement, etc.)\n\n"
                f"*(Type your observation, or 'skip' if none)*"
            ),
            'stockData': None, 'charts': None, 'confidence': None,
            'session_id': key,
        }

    # ── ask_context ───────────────────────────────────────────
    if step == 'ask_context':
        ctx = message.strip()
        if ctx.lower() in ('skip', 'none', 'n/a', 'no', 'nothing'):
            ctx = None
        update_session(key, {'user_context': ctx, 'step': 'confirm_retrain'})

        # Run retrospective analysis
        sess = get_session(key)
        retro = _learn_do_retrospect(sess)
        update_session(key, {'retrospect_text': retro})

        syms_str = ', '.join(sess['symbols'])
        return {
            'message': (
                f"{retro}\n\n---\n\n"
                f"Shall I retrain the model for **{syms_str}** with the latest data?\n\n"
                f"*(yes / no)*"
            ),
            'stockData': None, 'charts': None, 'confidence': None,
            'session_id': key,
        }

    # ── confirm_retrain ───────────────────────────────────────
    if step == 'confirm_retrain':
        t = message.strip().lower()
        sess = get_session(key)
        if t in ('yes', 'y', 'sure', 'ok', 'do it', 'retrain', 'train'):
            retrain_msg = _trigger_retrain(sess['symbols'])
            _save_learn_feedback(sess, retrained=True)
            delete_session(key)
            return {
                'message': (
                    f"{retrain_msg}\n\n"
                    f"Your feedback has been saved for future reference. Thank you!"
                ),
                'stockData': None, 'charts': None, 'confidence': None,
                'session_id': None,
            }
        else:
            _save_learn_feedback(sess, retrained=False)
            delete_session(key)
            return {
                'message': (
                    "No problem — feedback saved without retraining.\n\n"
                    "You can retrain anytime with "
                    "`curl -X POST \"http://localhost:8090/train?symbols="
                    + ','.join(sess['symbols']) + "\"`"
                ),
                'stockData': None, 'charts': None, 'confidence': None,
                'session_id': None,
            }

    # Fallback
    delete_session(key)
    return {
        'message': 'Learning session reset. Type **@learn** to start again.',
        'stockData': None, 'charts': None, 'confidence': None, 'session_id': None,
    }


def generate_response(message: str, user_id: Optional[str] = None,
                      session_id: Optional[str] = None) -> Dict[str, Any]:
    """Generate AI response to user message using NLP + ML Engine + LLM + live stock data."""
    try:
        # ---- @learn intercept (before NLP) ---- #
        learn_key = session_id or user_id
        active_learn = get_session(learn_key) if learn_key else None
        if _is_learn_command(message) or active_learn is not None:
            return _handle_learn_message(message, user_id=user_id, session_id=session_id)

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

        # Broad "growth + cheap / low cost" questions are not single-ticker TA (also avoids false symbols like LOW from "low cost")
        if _is_growth_value_screening_message(message):
            intent = "market_advice"
            entities.pop("symbol", None)
            entities["screening_growth_value"] = True
            confidence = max(confidence, 0.85)
        # Startup / high-growth stock questions
        elif _is_startup_screening_message(message):
            intent = "market_advice"
            entities.pop("symbol", None)
            entities["screening_startup"] = True
            confidence = max(confidence, 0.85)
        # Broad "what stocks to invest" without dollar amount
        elif _is_stock_ideas_message(message):
            intent = "market_advice"
            entities.pop("symbol", None)
            entities["screening_stock_ideas"] = True
            confidence = max(confidence, 0.85)
        # Buffett investment analysis
        elif re.search(
            r"buffett|warren\s*buff|economic\s*moat|intrinsic\s*value|margin\s*of\s*safety"
            r"|buffett\s*score|would\s*buffett\s*buy|buffett\s*style|durable\s*competitive\s*advantage",
            message.lower(),
        ):
            intent = "buffett_analysis"
            confidence = max(confidence, 0.90)
        # Jhunjhunwala / GARP analysis
        elif re.search(
            r"jhunjhunwala|rakesh|multibagger|multi\s*bagger"
            r"|garp\s*analysis|growth.*reasonable\s*price|jhunjhunwala\s*score"
            r"|jhunjhunwala\s*pick|jhunjhunwala\s*style|10x\s*potential|hundred\s*bagger",
            message.lower(),
        ):
            intent = "jhunjhunwala_analysis"
            confidence = max(confidence, 0.90)
        # Fundamental analysis (generic)
        elif re.search(
            r"fundamental\s*analysis|fundamentals?\s*(?:of|for)"
            r"|balance\s*sheet\s*analysis|income\s*statement\s*review"
            r"|earnings\s*quality|financial\s*health\s*(?:of|for)",
            message.lower(),
        ):
            intent = "fundamental_analysis"
            confidence = max(confidence, 0.90)
        # Currency investment questions (must check before price cap to avoid false positives)
        elif re.search(
            r"(?:best|top|safe|strong).*(?:currency|currencies).*(?:invest|buy|hold)"
            r"|(?:invest|trading|trade).*(?:currency|currencies|forex)"
            r"|(?:currency|currencies|forex).*(?:invest|profitable|profit|good|worth|strategy)"
            r"|(?:should|can|is).*(?:invest|trade).*(?:currency|currencies|forex)"
            r"|(?:forex|currency)\s*(?:trading|investment)\s*(?:guide|beginner|strategy|profitable)?",
            message.lower(),
        ):
            intent = "currency_investment"
            entities.pop("symbol", None)
            confidence = max(confidence, 0.85)
        # Currency conversion questions ("usd to inr", "convert dollars to rupees", "exchange rate")
        elif re.search(
            r"(?:convert|change).*(?:usd|dollar|eur|euro|gbp|pound|jpy|yen|inr|rupee|aud|cad|chf)"
            r"|(?:usd|dollar|eur|euro|gbp|pound|jpy|yen|aud|cad|chf)\s*(?:to|into|in)\s*(?:usd|dollar|eur|euro|gbp|pound|jpy|yen|inr|rupee|aud|cad|chf)"
            r"|exchange\s*rate"
            r"|(?:how\s*(?:much|many)).*(?:dollar|rupee|euro|pound|yen)"
            r"|(?:currency|forex)\s*(?:rate|conversion)",
            message.lower(),
        ):
            intent = "currency_conversion"
            confidence = max(confidence, 0.85)
        elif not entities.get("screening_growth_value"):
            cap_usd = _parse_under_price_screening_cap_usd(message)
            cap_inr = None if cap_usd is not None else _parse_under_price_screening_cap_inr(message)
            if cap_usd is not None:
                intent = "market_advice"
                entities.pop("symbol", None)
                entities.pop("amount", None)
                entities.pop("currency", None)
                entities["under_price_cap_usd"] = cap_usd
                if not entities.get("market"):
                    entities["market"] = "us"
                confidence = max(confidence, 0.85)
            elif cap_inr is not None:
                intent = "market_advice"
                entities.pop("symbol", None)
                entities.pop("amount", None)
                entities.pop("currency", None)
                entities["under_price_cap_inr"] = cap_inr
                entities["market"] = "india"
                confidence = max(confidence, 0.85)

        # Fetch real stock data if a symbol was detected (for ALL intents)
        stock_data = None
        symbol = entities.get("symbol")
        if symbol:
            stock_data = fetch_stock_data(symbol)

        # ---- ML Engine Integration (runs for ALL queries) ---- #
        ml_results = {}

        # XGBoost prediction for ANY query with a symbol
        if symbol and feature_pipeline and xgboost_predictor:
            try:
                features = feature_pipeline.build_features(symbol, include_live_bar=True)
                ml_results['prediction'] = xgboost_predictor.predict(symbol, features)
            except Exception as e:
                logger.warning(f"XGBoost prediction failed for {symbol}: {e}")

        # Technical indicators for ANY query with a symbol
        if symbol and feature_pipeline:
            try:
                ml_results['indicators'] = feature_pipeline.compute_indicators(symbol)
            except Exception as e:
                logger.warning(f"Feature pipeline indicators failed for {symbol}: {e}")

        # Sentiment analysis for ANY query with a symbol
        if symbol and sentiment_analyzer:
            try:
                ml_results['sentiment'] = sentiment_analyzer.analyze_symbol(symbol)
            except Exception as e:
                logger.warning(f"Sentiment analysis failed for {symbol}: {e}")

        # Buffett / Jhunjhunwala scores for fundamental-related intents
        if symbol and intent in ("buffett_analysis", "jhunjhunwala_analysis", "fundamental_analysis"):
            if buffett_scorer:
                try:
                    ml_results['buffett_score'] = buffett_scorer.score(symbol)
                except Exception as e:
                    logger.warning(f"Buffett scoring failed for {symbol}: {e}")
            if jhunjhunwala_scorer:
                try:
                    ml_results['jhunjhunwala_score'] = jhunjhunwala_scorer.score(symbol)
                except Exception as e:
                    logger.warning(f"Jhunjhunwala scoring failed for {symbol}: {e}")

        # Portfolio optimizer + Monte Carlo for intents that benefit from allocation advice
        _allocation_intents = {
            "market_advice":       {"default_risk": "moderate", "default_horizon": 12},
            "retirement_planning": {"default_risk": "conservative", "default_horizon": 120},
            "income_strategy":     {"default_risk": "conservative", "default_horizon": 60},
            "risk_assessment":     {"default_risk": "conservative", "default_horizon": 12},
            "financial_planning":  {"default_risk": "moderate", "default_horizon": 60},
            "beginner_guidance":   {"default_risk": "moderate", "default_horizon": 36},
        }
        _skip_alloc = (
            entities.get("screening_growth_value")
            or entities.get("screening_startup")
            or entities.get("screening_stock_ideas")
            or entities.get("under_price_cap_usd") is not None
            or entities.get("under_price_cap_inr") is not None
        )
        if intent in _allocation_intents and portfolio_optimizer and not _skip_alloc:
            defaults = _allocation_intents[intent]
            amount = entities.get('amount', 10000)
            risk = entities.get('risk_level', defaults["default_risk"])
            horizon = entities.get('horizon_months', defaults["default_horizon"])
            market = entities.get('market')  # 'india', 'us', or None
            # Infer market from currency when not explicitly specified
            if not market:
                currency = entities.get('currency', '').upper()
                if currency == 'USD':
                    market = 'us'
                elif currency == 'INR':
                    market = 'india'
            # Infer market from detected symbol asset class
            if not market and symbol:
                try:
                    from backend.ml.feature_pipeline import AssetClass, FeaturePipeline
                    _ac = FeaturePipeline.classify_asset(symbol)
                    _ac_market = {
                        AssetClass.FOREX: 'forex', AssetClass.COMMODITY: 'commodity',
                        AssetClass.CRYPTO: 'crypto', AssetClass.INDIA_STOCK: 'india',
                        AssetClass.US_STOCK: 'us',
                    }
                    market = _ac_market.get(_ac)
                except Exception:
                    pass
            try:
                ml_results['allocation'] = portfolio_optimizer.recommend_allocation(amount, risk, horizon, market=market)
                # Run Monte Carlo on the recommended allocation
                alloc = ml_results['allocation']
                symbols_alloc = list(alloc['allocations'].keys())
                weights_alloc = [alloc['allocations'][s]['weight'] for s in symbols_alloc]
                if monte_carlo_sim and symbols_alloc and weights_alloc:
                    ml_results['forecast'] = monte_carlo_sim.simulate(symbols_alloc, weights_alloc, amount, horizon)
            except Exception as e:
                logger.warning(f"Portfolio optimization failed: {e}")

        # Risk analysis for portfolio_management with user holdings
        if intent == "portfolio_management" and user_id and portfolio_optimizer and ENHANCED_MODULES_AVAILABLE:
            try:
                portfolios = db_manager.get_user_portfolios(user_id)
                if portfolios and portfolios[0].get('stocks'):
                    stocks = portfolios[0]['stocks']
                    syms = [s['symbol'] for s in stocks]
                    vals = [s['shares'] * s['purchase_price'] for s in stocks]
                    total = sum(vals)
                    wts = [v / total for v in vals] if total > 0 else []
                    if syms and wts:
                        ml_results['risk'] = portfolio_optimizer.risk_analysis(syms, wts)
            except Exception as e:
                logger.warning(f"Portfolio risk analysis failed: {e}")

        # ---- Handle portfolio_management display ---- #
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

        # ---- Currency intent short-circuits ---- #
        if intent == "currency_conversion":
            # Extract forex pair from user message if symbol wasn't detected by NLP
            if not symbol:
                symbol = _extract_forex_symbol(message)
            if not stock_data and symbol:
                try:
                    stock_data = fetch_stock_data(symbol)
                except Exception:
                    pass
            response_text = _handle_currency_conversion(symbol or 'USDINR=X', stock_data)
            if user_id and ENHANCED_MODULES_AVAILABLE:
                try:
                    db_manager.save_chat_message(user_id, message, response_text)
                except Exception:
                    pass
            return {
                "message": response_text,
                "stockData": stock_data,
                "charts": None,
                "confidence": confidence,
            }

        if intent == "currency_investment":
            response_text = _handle_currency_investment()
            # Attach a representative forex card
            if stock_data is None:
                try:
                    stock_data = fetch_stock_data('USDINR=X')
                except Exception:
                    pass
            if user_id and ENHANCED_MODULES_AVAILABLE:
                try:
                    db_manager.save_chat_message(user_id, message, response_text)
                except Exception:
                    pass
            return {
                "message": response_text,
                "stockData": stock_data,
                "charts": None,
                "confidence": confidence,
            }

        # ---- Buffett / Jhunjhunwala / Fundamental short-circuits ---- #
        if intent in ("buffett_analysis", "jhunjhunwala_analysis", "fundamental_analysis"):
            if not symbol:
                response_text = (
                    "Please specify a stock symbol so I can run the analysis. "
                    "For example: *\"Buffett analysis of AAPL\"* or *\"Jhunjhunwala score for RELIANCE.BSE\"*"
                )
            else:
                response_text = _handle_investment_philosophy(intent, symbol, ml_results, stock_data)
            if user_id and ENHANCED_MODULES_AVAILABLE:
                try:
                    db_manager.save_chat_message(user_id, message, response_text)
                except Exception:
                    pass
            return {
                "message": response_text,
                "stockData": stock_data,
                "charts": None,
                "confidence": confidence,
            }

        # ---- Build response text ---- #
        if ml_results:
            # ML results available - use LLM or formatted template
            if llm_provider:
                response_text = llm_provider.generate_response(intent, entities, message, ml_results=ml_results)
            else:
                response_text = llm_provider._format_ml_results(ml_results) if llm_provider else ""
                if not response_text:
                    response_text = json.dumps(ml_results, indent=2, default=str)

            # Prepend stock data summary if available
            if stock_data:
                price = stock_data["price"]
                change = stock_data["change"]
                change_pct = stock_data["changePercent"]
                direction = "up" if change >= 0 else "down"
                sign = "+" if change >= 0 else ""
                name = stock_data.get("name", symbol)
                cs = stock_data.get("currencySymbol", _currency_symbol(symbol))
                stock_summary = (
                    f"{name} ({symbol}) is currently trading at {cs}{price:,.2f}, "
                    f"{direction} {sign}{change:.2f} ({sign}{change_pct:.2f}%).\n\n"
                )
                response_text = stock_summary + response_text
        elif stock_data:
            price = stock_data["price"]
            change = stock_data["change"]
            change_pct = stock_data["changePercent"]
            direction = "up" if change >= 0 else "down"
            sign = "+" if change >= 0 else ""
            name = stock_data.get("name", symbol)
            cs = stock_data.get("currencySymbol", _currency_symbol(symbol))

            response_text = (
                f"{name} ({symbol}) is currently trading at {cs}{price:,.2f}, "
                f"{direction} {sign}{change:.2f} ({sign}{change_pct:.2f}%) from the previous close.\n\n"
            )

            if stock_data.get("open"):
                response_text += f"Open: {cs}{stock_data['open']:,.2f}\n"
            if stock_data.get("high") and stock_data.get("low"):
                response_text += f"Day range: {cs}{stock_data['low']:,.2f} - {cs}{stock_data['high']:,.2f}\n"
            if stock_data.get("volume"):
                response_text += f"Volume: {stock_data['volume']:,}\n"
            if stock_data.get("previousClose"):
                response_text += f"Previous close: {cs}{stock_data['previousClose']:,.2f}\n"
            if stock_data.get("fiftyTwoWeekHigh") and stock_data.get("fiftyTwoWeekLow"):
                response_text += f"52-week range: {cs}{stock_data['fiftyTwoWeekLow']:,.2f} - {cs}{stock_data['fiftyTwoWeekHigh']:,.2f}\n"
            if stock_data.get("peRatio"):
                response_text += f"P/E Ratio: {stock_data['peRatio']:.2f}\n"
            if stock_data.get("marketCap"):
                cap = stock_data["marketCap"]
                if cap >= 1e12:
                    response_text += f"Market Cap: {cs}{cap/1e12:.2f}T\n"
                elif cap >= 1e9:
                    response_text += f"Market Cap: {cs}{cap/1e9:.2f}B\n"
                else:
                    response_text += f"Market Cap: {cs}{cap/1e6:.2f}M\n"

            response_text += "\nNote: This is not financial advice. Always do your own research before making investment decisions."
        elif portfolio_response:
            response_text = portfolio_response
            # Append risk analysis if available
            if ml_results.get('risk'):
                risk = ml_results['risk']
                response_text += (
                    f"\n\n**Portfolio Risk Metrics**\n"
                    f"- Sharpe Ratio: {risk.get('sharpe_ratio', 0):.2f}\n"
                    f"- Annual Volatility: {risk.get('annual_volatility', 0):.1%}\n"
                    f"- Max Drawdown: {risk.get('max_drawdown', 0):.1%}\n"
                    f"- Beta: {risk.get('beta', 0):.2f}"
                )
        else:
            # No stock data and no ML results - use LLM/template response
            if intent == "market_advice":
                if entities.get("screening_growth_value"):
                    ml = (message or "").lower()
                    india = (
                        entities.get("market") == "india"
                        or any(k in ml for k in ("india", "indian", "nifty", "sensex", "bse", "nse", "rupee", "inr"))
                    )
                    response_text = _format_growth_value_screening_markdown(india)
                elif entities.get("screening_startup"):
                    ml = (message or "").lower()
                    india = (
                        entities.get("market") == "india"
                        or any(k in ml for k in ("india", "indian", "nifty", "sensex", "bse", "nse", "rupee", "inr"))
                    )
                    response_text = _format_startup_screening_markdown(india)
                elif entities.get("screening_stock_ideas"):
                    ml = (message or "").lower()
                    india = (
                        entities.get("market") == "india"
                        or any(k in ml for k in ("india", "indian", "nifty", "sensex", "bse", "nse", "rupee", "inr"))
                    )
                    response_text = _format_stock_ideas_screening_markdown(india)
                elif entities.get("under_price_cap_usd") is not None:
                    response_text = _format_under_price_screening_usd(float(entities["under_price_cap_usd"]))
                elif entities.get("under_price_cap_inr") is not None:
                    response_text = _format_under_price_screening_inr(float(entities["under_price_cap_inr"]))
                elif llm_provider:
                    response_text = llm_provider.generate_response(intent, entities, message)
                else:
                    response_text = handle_market_advice(message, entities)
            elif llm_provider:
                response_text = llm_provider.generate_response(intent, entities, message)
            else:
                response_text = handle_general_question(message)

        # Attach a representative stock card for screening responses
        _screening_spot_map = {
            "screening_growth_value": {"us": "SOFI", "india": "HDFCBANK.BSE"},
            "screening_startup":     {"us": "PLTR", "india": "ZOMATO.BSE"},
            "screening_stock_ideas": {"us": "NVDA", "india": "RELIANCE.BSE"},
        }
        if stock_data is None:
            for _skey, _sym_map in _screening_spot_map.items():
                if entities.get(_skey):
                    try:
                        _ml_spot = (message or "").lower()
                        _india_spot = (
                            entities.get("market") == "india"
                            or any(
                                k in _ml_spot
                                for k in ("india", "indian", "nifty", "sensex", "bse", "nse", "rupee", "inr")
                            )
                        )
                        _sp = _sym_map["india"] if _india_spot else _sym_map["us"]
                        spot = fetch_stock_data(_sp)
                        if spot:
                            stock_data = spot
                    except Exception:
                        pass
                    break
        if entities.get("under_price_cap_usd") is not None and stock_data is None:
            try:
                spot = fetch_stock_data("SOFI")
                if spot:
                    stock_data = spot
            except Exception:
                pass
        elif entities.get("under_price_cap_inr") is not None and stock_data is None:
            try:
                spot = fetch_stock_data("ITC.BSE")
                if spot:
                    stock_data = spot
            except Exception:
                pass

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


def _extract_forex_symbol(message: str) -> Optional[str]:
    """Extract a Yahoo Finance forex symbol from a user message like 'usd to inr'."""
    msg = message.lower()
    # Map common currency names/codes to standard 3-letter codes
    _currency_aliases = {
        'usd': 'USD', 'dollar': 'USD', 'dollars': 'USD', 'us dollar': 'USD',
        'inr': 'INR', 'rupee': 'INR', 'rupees': 'INR', 'indian rupee': 'INR',
        'eur': 'EUR', 'euro': 'EUR', 'euros': 'EUR',
        'gbp': 'GBP', 'pound': 'GBP', 'pounds': 'GBP', 'british pound': 'GBP', 'sterling': 'GBP',
        'jpy': 'JPY', 'yen': 'JPY', 'japanese yen': 'JPY',
        'aud': 'AUD', 'australian dollar': 'AUD',
        'cad': 'CAD', 'canadian dollar': 'CAD',
        'chf': 'CHF', 'swiss franc': 'CHF', 'franc': 'CHF',
        'nzd': 'NZD', 'new zealand dollar': 'NZD',
    }
    # Known Yahoo Finance forex pairs (base/quote)
    _valid_pairs = {
        ('USD', 'INR'): 'USDINR=X', ('INR', 'USD'): 'USDINR=X',
        ('EUR', 'USD'): 'EURUSD=X', ('USD', 'EUR'): 'EURUSD=X',
        ('GBP', 'USD'): 'GBPUSD=X', ('USD', 'GBP'): 'GBPUSD=X',
        ('USD', 'JPY'): 'USDJPY=X', ('JPY', 'USD'): 'USDJPY=X',
        ('AUD', 'USD'): 'AUDUSD=X', ('USD', 'AUD'): 'AUDUSD=X',
        ('USD', 'CAD'): 'USDCAD=X', ('CAD', 'USD'): 'USDCAD=X',
        ('USD', 'CHF'): 'USDCHF=X', ('CHF', 'USD'): 'USDCHF=X',
        ('NZD', 'USD'): 'NZDUSD=X', ('USD', 'NZD'): 'NZDUSD=X',
        ('EUR', 'GBP'): 'EURGBP=X', ('GBP', 'EUR'): 'EURGBP=X',
        ('EUR', 'JPY'): 'EURJPY=X', ('JPY', 'EUR'): 'EURJPY=X',
        ('EUR', 'INR'): 'EURINR=X', ('INR', 'EUR'): 'EURINR=X',
        ('GBP', 'INR'): 'GBPINR=X', ('INR', 'GBP'): 'GBPINR=X',
    }
    # Try to find "X to Y" or "X into Y" or "X in Y" pattern
    m = re.search(
        r'(\b(?:us dollar|australian dollar|canadian dollar|new zealand dollar|british pound'
        r'|swiss franc|japanese yen|indian rupee'
        r'|usd|dollar|dollars|eur|euro|euros|gbp|pound|pounds|sterling'
        r'|jpy|yen|aud|cad|chf|nzd|inr|rupee|rupees|franc)\b)'
        r'\s+(?:to|into|in|vs|versus)\s+'
        r'(\b(?:us dollar|australian dollar|canadian dollar|new zealand dollar|british pound'
        r'|swiss franc|japanese yen|indian rupee'
        r'|usd|dollar|dollars|eur|euro|euros|gbp|pound|pounds|sterling'
        r'|jpy|yen|aud|cad|chf|nzd|inr|rupee|rupees|franc)\b)',
        msg,
    )
    if m:
        from_code = _currency_aliases.get(m.group(1))
        to_code = _currency_aliases.get(m.group(2))
        if from_code and to_code and from_code != to_code:
            pair = _valid_pairs.get((from_code, to_code))
            if pair:
                return pair
            # Fallback: construct symbol (base + quote + =X)
            return f"{from_code}{to_code}=X"
    # Fallback: look for any two distinct currency codes in the message
    found = []
    # Check longer aliases first to avoid partial matches
    for alias in sorted(_currency_aliases.keys(), key=len, reverse=True):
        if re.search(r'\b' + re.escape(alias) + r'\b', msg):
            code = _currency_aliases[alias]
            if code not in found:
                found.append(code)
            if len(found) == 2:
                break
    if len(found) == 2:
        pair = _valid_pairs.get((found[0], found[1]))
        if pair:
            return pair
        return f"{found[0]}{found[1]}=X"
    # Single currency mentioned: assume USD as the other side
    if len(found) == 1 and found[0] != 'USD':
        pair = _valid_pairs.get(('USD', found[0]))
        if pair:
            return pair
        return f"USD{found[0]}=X"
    return None


def _handle_currency_conversion(symbol: str, stock_data: Optional[Dict] = None) -> str:
    """Handle currency conversion queries with rich, detailed output."""
    _pair_info = {
        'USDINR=X': {'from_flag': '\U0001f1fa\U0001f1f8', 'from_name': 'US Dollar', 'from_sym': '$', 'from_code': 'USD',
                      'to_flag': '\U0001f1ee\U0001f1f3', 'to_name': 'Indian Rupee', 'to_sym': '\u20b9', 'to_code': 'INR'},
        'EURUSD=X': {'from_flag': '\U0001f1ea\U0001f1fa', 'from_name': 'Euro', 'from_sym': '\u20ac', 'from_code': 'EUR',
                      'to_flag': '\U0001f1fa\U0001f1f8', 'to_name': 'US Dollar', 'to_sym': '$', 'to_code': 'USD'},
        'GBPUSD=X': {'from_flag': '\U0001f1ec\U0001f1e7', 'from_name': 'British Pound', 'from_sym': '\u00a3', 'from_code': 'GBP',
                      'to_flag': '\U0001f1fa\U0001f1f8', 'to_name': 'US Dollar', 'to_sym': '$', 'to_code': 'USD'},
        'USDJPY=X': {'from_flag': '\U0001f1fa\U0001f1f8', 'from_name': 'US Dollar', 'from_sym': '$', 'from_code': 'USD',
                      'to_flag': '\U0001f1ef\U0001f1f5', 'to_name': 'Japanese Yen', 'to_sym': '\u00a5', 'to_code': 'JPY'},
        'AUDUSD=X': {'from_flag': '\U0001f1e6\U0001f1fa', 'from_name': 'Australian Dollar', 'from_sym': 'A$', 'from_code': 'AUD',
                      'to_flag': '\U0001f1fa\U0001f1f8', 'to_name': 'US Dollar', 'to_sym': '$', 'to_code': 'USD'},
        'USDCAD=X': {'from_flag': '\U0001f1fa\U0001f1f8', 'from_name': 'US Dollar', 'from_sym': '$', 'from_code': 'USD',
                      'to_flag': '\U0001f1e8\U0001f1e6', 'to_name': 'Canadian Dollar', 'to_sym': 'C$', 'to_code': 'CAD'},
        'USDCHF=X': {'from_flag': '\U0001f1fa\U0001f1f8', 'from_name': 'US Dollar', 'from_sym': '$', 'from_code': 'USD',
                      'to_flag': '\U0001f1e8\U0001f1ed', 'to_name': 'Swiss Franc', 'to_sym': 'CHF', 'to_code': 'CHF'},
        'NZDUSD=X': {'from_flag': '\U0001f1f3\U0001f1ff', 'from_name': 'New Zealand Dollar', 'from_sym': 'NZ$', 'from_code': 'NZD',
                      'to_flag': '\U0001f1fa\U0001f1f8', 'to_name': 'US Dollar', 'to_sym': '$', 'to_code': 'USD'},
        'EURGBP=X': {'from_flag': '\U0001f1ea\U0001f1fa', 'from_name': 'Euro', 'from_sym': '\u20ac', 'from_code': 'EUR',
                      'to_flag': '\U0001f1ec\U0001f1e7', 'to_name': 'British Pound', 'to_sym': '\u00a3', 'to_code': 'GBP'},
        'EURJPY=X': {'from_flag': '\U0001f1ea\U0001f1fa', 'from_name': 'Euro', 'from_sym': '\u20ac', 'from_code': 'EUR',
                      'to_flag': '\U0001f1ef\U0001f1f5', 'to_name': 'Japanese Yen', 'to_sym': '\u00a5', 'to_code': 'JPY'},
    }

    info = _pair_info.get(symbol)
    rate = stock_data.get('price', 0) if stock_data else 0
    change = stock_data.get('change', 0) if stock_data else 0
    change_pct = stock_data.get('changePercent', 0) if stock_data else 0

    if not info:
        # Generic fallback for unknown pairs
        return (
            f"## {symbol} Exchange Rate\n\n"
            f"**Current Rate:** {rate:,.4f}\n"
            f"**Change:** {'+' if change >= 0 else ''}{change:.4f} ({'+' if change_pct >= 0 else ''}{change_pct:.2f}%)\n\n"
            f"Exchange rates fluctuate throughout the day based on the forex market.\n\n"
            f"*Ask: \"Predict {symbol}\" for AI-based forecast.*\n\n"
            f"*Not financial advice. Rates are indicative.*"
        )

    sign = '+' if change >= 0 else ''

    response = (
        f"## {info['from_code']}/{info['to_code']} Exchange Rate\n\n"
        f"{info['from_flag']} **{info['from_name']}** ({info['from_sym']})"
        f"  \u2192  "
        f"{info['to_flag']} **{info['to_name']}** ({info['to_sym']})\n\n"
        f"**Current Rate:** 1 {info['from_code']} = {info['to_sym']}{rate:,.4f}\n"
        f"**Change:** {sign}{change:.4f} ({sign}{change_pct:.2f}%)\n\n"
    )

    # Conversion examples
    if rate > 0:
        multipliers = [1, 10, 100, 1000, 10000]
        response += "### Quick Conversion\n\n"
        response += f"| {info['from_code']} | {info['to_code']} |\n"
        response += "|------|------|\n"
        for m in multipliers:
            response += f"| {info['from_sym']}{m:,} | {info['to_sym']}{m * rate:,.2f} |\n"
        response += "\n"

    response += (
        "### Key Factors Affecting This Rate\n\n"
        "- **Interest rate differentials** between central banks\n"
        "- **Inflation rates** in both economies\n"
        "- **Trade balance** and capital flows\n"
        "- **Economic growth** (GDP) differentials\n"
        "- **Geopolitical events** and risk sentiment\n\n"
        f"### Next Steps\n\n"
        f"- *\"Predict {symbol}\"* for ML-based exchange rate forecast\n"
        f"- *\"Technical analysis {symbol}\"* for trend indicators\n\n"
        f"---\n\n"
        f"*Exchange rates fluctuate throughout the day. Not financial advice.*"
    )

    return response


def _handle_currency_investment() -> str:
    """Handle queries about investing in currencies with educational, detailed guidance."""
    return (
        "## Currency Investment Guide\n\n"
        "Currency swaps and forex trading can be profitable, but they are "
        "usually **much riskier** than long-term stock or ETF investing.\n\n"
        "### Two Common Meanings\n\n"
        "**1. Forex Trading / Currency Speculation**\n"
        "- Buying one currency and selling another to profit from exchange-rate changes\n"
        "- Example: betting that INR will strengthen against USD\n"
        "- High risk because currencies move based on interest rates, geopolitics, "
        "inflation, and central bank actions\n\n"
        "**2. Currency Swap Instruments**\n"
        "- Mostly used by banks, corporations, and large investors to hedge currency exposure\n"
        "- Not typically suitable for beginners\n\n"
        "### Strongest Major Currencies\n\n"
        "| Currency | Country | Risk Level | Use Case |\n"
        "|----------|---------|------------|----------|\n"
        "| **USD** \U0001f1fa\U0001f1f8 | United States | Low-Medium | Global reserve currency, strongest liquidity |\n"
        "| **CHF** \U0001f1e8\U0001f1ed | Switzerland | Low | Safe-haven during crises |\n"
        "| **SGD** \U0001f1f8\U0001f1ec | Singapore | Low | Stable economy, strong monetary policy |\n"
        "| **EUR** \U0001f1ea\U0001f1fa | Eurozone | Medium | Diversification from USD |\n"
        "| **JPY** \U0001f1ef\U0001f1f5 | Japan | Medium | Often rises during market fear |\n"
        "| **GBP** \U0001f1ec\U0001f1e7 | United Kingdom | Medium-High | Strong financial market presence |\n\n"
        "### Challenges of Forex Trading\n\n"
        "- High leverage can **magnify losses**\n"
        "- Markets move 24/5 and are volatile\n"
        "- Requires macroeconomic knowledge and strong risk management\n"
        "- Many beginners lose money due to overtrading\n\n"
        "### A More Stable Approach\n\n"
        "For most retail investors, a balanced strategy works better:\n\n"
        "| Allocation | Strategy |\n"
        "|------------|----------|\n"
        "| 70-80% | Long-term diversified investments (ETFs, index funds, quality stocks) |\n"
        "| 10-20% | High-growth opportunities |\n"
        "| 5% or less | Forex / crypto / speculative trades |\n\n"
        "### Ways to Get Currency Exposure\n\n"
        "- **Investing in U.S. stocks** while earning/spending in INR\n"
        "- **Holding USD-denominated assets** for diversification\n"
        "- **International ETFs** (VXUS, VEA, EFA)\n"
        "- **Currency ETFs**: FXE (Euro), FXY (Yen), FXB (Pound)\n"
        "- **Government bonds** in different currencies\n\n"
        "### Protecting Against INR Depreciation\n\n"
        "Historically, these currencies have strengthened against INR:\n"
        "- USD, CHF, GBP, EUR\n\n"
        "A practical structure:\n"
        "- 50% USD assets\n"
        "- 20% CHF/SGD\n"
        "- 20% Global equity ETFs\n"
        "- 10% Speculative/high-growth bets\n\n"
        "### Ask Me More\n\n"
        "- *\"USD to INR\"* - live exchange rate\n"
        "- *\"Predict EURUSD=X\"* - ML-based forex forecast\n"
        "- *\"Technical analysis dollar rupee\"* - trend indicators\n"
        "- *\"Best stocks to invest in\"* - equity alternatives\n\n"
        "---\n\n"
        "*Currency markets are volatile. This is educational content, not financial advice.*"
    )


def _handle_investment_philosophy(intent: str, symbol: str, ml_results: Dict, stock_data: Optional[Dict] = None) -> str:
    """Handle Buffett, Jhunjhunwala, and fundamental analysis queries."""
    lines = []

    # Header with stock price if available
    if stock_data:
        price = stock_data.get("price", 0)
        change = stock_data.get("change", 0)
        change_pct = stock_data.get("changePercent", 0)
        name = stock_data.get("name", symbol)
        sign = "+" if change >= 0 else ""
        lines.append(f"## {name} ({symbol}) - ${price:,.2f} ({sign}{change_pct:.2f}%)\n")
    else:
        lines.append(f"## {symbol} Investment Philosophy Analysis\n")

    buffett = ml_results.get('buffett_score')
    jhunjhunwala = ml_results.get('jhunjhunwala_score')

    no_data = not buffett and not jhunjhunwala

    if no_data:
        lines.append(
            "Fundamental data is not available for this asset. "
            "Investment philosophy scoring requires equity fundamentals "
            "(income statement, balance sheet, cash flow) which are available "
            "for US and Indian equities.\n\n"
            "Try a stock like **AAPL**, **MSFT**, **RELIANCE.BSE**, or **INFY.BSE**."
        )
        return "\n".join(lines)

    # ---- Buffett Section ---- #
    if buffett and intent in ("buffett_analysis", "fundamental_analysis"):
        score = buffett.get('overall_score', 0)
        grade = buffett.get('grade', 'N/A')
        verdict = buffett.get('verdict', '')
        lines.append(f"### Warren Buffett Analysis\n")
        lines.append(f"**Overall Score: {score:.0f}/100 ({grade})** - {verdict}\n")
        lines.append("| Criterion | Weight | Score | Details |")
        lines.append("|-----------|--------|-------|---------|")
        for criterion in buffett.get('criteria', []):
            name = criterion.get('name', '')
            weight = criterion.get('weight', 0)
            cscore = criterion.get('score', 0)
            detail = criterion.get('detail', '')
            lines.append(f"| {name} | {weight:.0%} | {cscore:.0f}/100 | {detail} |")
        narrative = buffett.get('narrative', '')
        if narrative:
            lines.append(f"\n{narrative}\n")

    # ---- Jhunjhunwala Section ---- #
    if jhunjhunwala and intent in ("jhunjhunwala_analysis", "fundamental_analysis"):
        score = jhunjhunwala.get('overall_score', 0)
        grade = jhunjhunwala.get('grade', 'N/A')
        verdict = jhunjhunwala.get('verdict', '')
        lines.append(f"### Rakesh Jhunjhunwala (GARP) Analysis\n")
        lines.append(f"**Overall Score: {score:.0f}/100 ({grade})** - {verdict}\n")
        lines.append("| Criterion | Weight | Score | Details |")
        lines.append("|-----------|--------|-------|---------|")
        for criterion in jhunjhunwala.get('criteria', []):
            name = criterion.get('name', '')
            weight = criterion.get('weight', 0)
            cscore = criterion.get('score', 0)
            detail = criterion.get('detail', '')
            lines.append(f"| {name} | {weight:.0%} | {cscore:.0f}/100 | {detail} |")
        narrative = jhunjhunwala.get('narrative', '')
        if narrative:
            lines.append(f"\n{narrative}\n")

    # ---- ML prediction if available ---- #
    pred = ml_results.get('prediction')
    if pred:
        direction = pred.get('direction', 'N/A')
        prob = pred.get('probability', 0)
        exp_ret = pred.get('expected_return', 0)
        horizon = pred.get('horizon_days', 21)
        lines.append(f"### ML Prediction ({horizon}-day horizon)\n")
        lines.append(f"- Direction: **{direction}** (probability: {prob:.0%})")
        lines.append(f"- Expected return: **{exp_ret:.1%}**\n")

    # Next steps
    lines.append("### Explore Further\n")
    if intent != "buffett_analysis" and buffett:
        lines.append(f"- *\"Buffett analysis of {symbol}\"* - Detailed value investing view")
    if intent != "jhunjhunwala_analysis" and jhunjhunwala:
        lines.append(f"- *\"Jhunjhunwala analysis of {symbol}\"* - GARP / multibagger view")
    lines.append(f"- *\"Predict {symbol}\"* - ML-based price prediction")
    lines.append(f"- *\"Technical analysis {symbol}\"* - Chart indicators\n")

    lines.append("---\n\n*Investment philosophy scores are educational tools based on publicly available "
                 "financial data. Not financial advice. Always do your own research.*")

    return "\n".join(lines)


def handle_market_advice(message: str, entities: Optional[Dict] = None) -> str:
    """Handle broad market and investment advice queries with market-specific recommendations."""
    message_lower = message.lower()
    entities = entities or {}
    market = entities.get('market')
    amount = entities.get('amount')
    cs = '₹' if market == 'india' else '$'

    # Detect India context from message if not already detected
    if not market:
        india_keywords = ['india', 'indian', 'nifty', 'sensex', 'bse', 'nse', 'rupee', 'inr']
        if any(kw in message_lower for kw in india_keywords):
            market = 'india'
            cs = '₹'

    # ---- INDIA-SPECIFIC RESPONSES ---- #
    if market == 'india':
        if re.search(r"top\s*\d+\s*stocks?|best\s*stocks?", message_lower):
            return (
                "## Top Indian Stocks to Watch\n\n"
                "Here are widely-followed Indian stocks across key sectors:\n\n"
                "**IT/Technology:** INFY.BSE (Infosys), TCS.BSE (TCS), WIPRO.BSE (Wipro), HCLTECH.BSE (HCL Tech)\n\n"
                "**Banking & Finance:** HDFCBANK.BSE (HDFC Bank), ICICIBANK.BSE (ICICI Bank), SBIN.BSE (SBI), KOTAKBANK.BSE (Kotak), BAJFINANCE.BSE (Bajaj Finance)\n\n"
                "**Consumer:** HINDUNILVR.BSE (HUL), ITC.BSE (ITC), TITAN.BSE (Titan), NESTLEIND.BSE (Nestle India)\n\n"
                "**Energy & Infra:** RELIANCE.BSE (Reliance), LT.BSE (L&T), NTPC.BSE (NTPC), POWERGRID.BSE (Power Grid)\n\n"
                "**Auto:** MARUTI.BSE (Maruti Suzuki), TATAMOTORS.BSE (Tata Motors)\n\n"
                "**Pharma:** SUNPHARMA.BSE (Sun Pharma), DRREDDY.BSE (Dr Reddy's), CIPLA.BSE (Cipla)\n\n"
                "To get live prices and analysis, ask me: *\"Predict INFY.BSE\"* or *\"Infosys stock price\"*\n\n"
                "**Disclaimer:** This is not financial advice. Always do your own research."
            )

        if amount or re.search(r"invest|where.*(?:invest|put)|what.*(?:should|would).*(?:invest|buy)", message_lower):
            amt = amount or 10000
            return (
                f"## Indian Market Investment Guide ({cs}{amt:,.0f})\n\n"
                f"Here's a suggested allocation of {cs}{amt:,.0f} across Indian stocks:\n\n"
                f"### Conservative (Lower Risk)\n"
                f"- **HDFCBANK.BSE** (HDFC Bank): {cs}{amt*0.20:,.0f} - India's largest private bank\n"
                f"- **SBIN.BSE** (SBI): {cs}{amt*0.15:,.0f} - India's largest public bank\n"
                f"- **ITC.BSE** (ITC): {cs}{amt*0.15:,.0f} - Diversified conglomerate, strong dividends\n"
                f"- **HINDUNILVR.BSE** (HUL): {cs}{amt*0.15:,.0f} - FMCG leader\n"
                f"- **POWERGRID.BSE** (Power Grid): {cs}{amt*0.10:,.0f} - Stable utility stock\n\n"
                f"### Growth (Higher Risk, Higher Potential)\n"
                f"- **INFY.BSE** (Infosys): {cs}{amt*0.20:,.0f} - IT sector leader\n"
                f"- **RELIANCE.BSE** (Reliance): {cs}{amt*0.20:,.0f} - India's largest company\n"
                f"- **BAJFINANCE.BSE** (Bajaj Finance): {cs}{amt*0.15:,.0f} - Leading NBFC\n"
                f"- **TITAN.BSE** (Titan): {cs}{amt*0.15:,.0f} - Premium consumer brand\n"
                f"- **ICICIBANK.BSE** (ICICI Bank): {cs}{amt*0.15:,.0f} - Strong private bank\n\n"
                f"### General Tips for Indian Market\n"
                f"- Consider Nifty 50 index funds for broad diversification\n"
                f"- SIP (Systematic Investment Plan) reduces timing risk\n"
                f"- Banking and IT sectors are the largest in Indian markets\n\n"
                f"Ask me about any specific Indian stock for live prices and AI predictions!\n\n"
                f"**Disclaimer:** This is not financial advice. Always do your own research."
            )

        return (
            "## Indian Market Investment Insights\n\n"
            "I can help you with Indian stock analysis:\n\n"
            "- **Top Indian stocks** - Ask: *\"Best Indian stocks to buy\"*\n"
            "- **Investment allocation** - Ask: *\"Invest ₹50,000 in India\"*\n"
            "- **Specific stock analysis** - Ask: *\"Predict Infosys stock\"* or *\"Reliance stock price\"*\n\n"
            "**Key Indian Indices:** Nifty 50, Sensex\n"
            "**Major Sectors:** Banking, IT, FMCG, Pharma, Energy\n\n"
            "**Disclaimer:** This is not financial advice. Always do your own research."
        )

    # ---- FOREX RESPONSES ---- #
    if market == 'forex':
        return (
            "## Forex Market Guide\n\n"
            "### Major Pairs\n"
            "| Pair | Description | Typical Volatility |\n"
            "|------|-------------|-------------------|\n"
            "| **EURUSD=X** | Euro / US Dollar | Medium |\n"
            "| **GBPUSD=X** | British Pound / US Dollar | Medium-High |\n"
            "| **USDJPY=X** | US Dollar / Japanese Yen | Medium |\n"
            "| **USDINR=X** | US Dollar / Indian Rupee | Low-Medium |\n"
            "| **AUDUSD=X** | Australian Dollar / US Dollar | Medium-High |\n\n"
            "### Cross Pairs\n"
            "| Pair | Description |\n"
            "|------|-------------|\n"
            "| **EURGBP=X** | Euro / British Pound |\n"
            "| **EURJPY=X** | Euro / Japanese Yen |\n\n"
            "### Ask next\n"
            "- *\"Predict EURUSD=X\"* for ML-based forecast\n"
            "- *\"Technical analysis of dollar rupee\"* for indicators\n\n"
            "---\n\n"
            "*Forex trading involves significant risk. Not financial advice.*"
        )

    # ---- CRYPTO RESPONSES ---- #
    if market == 'crypto':
        return (
            "## Cryptocurrency Market Guide\n\n"
            "### Top Cryptocurrencies\n"
            "| Asset | Why it's notable | Risk |\n"
            "|-------|-----------------|------|\n"
            "| **BTC-USD** (Bitcoin) | Digital gold, largest market cap, institutional adoption | Medium-High |\n"
            "| **ETH-USD** (Ethereum) | Smart contracts, DeFi ecosystem, staking | High |\n"
            "| **SOL-USD** (Solana) | High-speed blockchain, growing DeFi/NFT ecosystem | High |\n"
            "| **BNB-USD** (Binance Coin) | Exchange token, utility in Binance ecosystem | High |\n"
            "| **XRP-USD** (Ripple) | Cross-border payments, institutional use case | High |\n"
            "| **ADA-USD** (Cardano) | Research-driven blockchain, proof of stake | High |\n"
            "| **DOGE-USD** (Dogecoin) | Meme coin with large community | Very High |\n\n"
            "### Strategy buckets\n"
            "- **Core holdings**: BTC-USD, ETH-USD (60-70%)\n"
            "- **Growth alts**: SOL-USD, BNB-USD, XRP-USD (20-30%)\n"
            "- **Speculative**: DOGE-USD, AVAX-USD (5-10%)\n\n"
            "### Ask next\n"
            "- *\"Predict bitcoin\"* or *\"Predict BTC-USD\"* for ML forecast\n"
            "- *\"Technical analysis of ethereum\"* for indicators\n\n"
            "---\n\n"
            "*Crypto markets are extremely volatile. Not financial advice.*"
        )

    # ---- COMMODITY RESPONSES ---- #
    if market == 'commodity':
        return (
            "## Commodity Futures Guide\n\n"
            "### Precious Metals\n"
            "| Asset | Symbol | Why it's notable |\n"
            "|-------|--------|------------------|\n"
            "| **Gold** | GC=F | Safe haven, inflation hedge, central bank demand |\n"
            "| **Silver** | SI=F | Industrial + precious metal, higher volatility than gold |\n"
            "| **Platinum** | PL=F | Industrial demand, automotive catalysts |\n\n"
            "### Energy\n"
            "| Asset | Symbol | Why it's notable |\n"
            "|-------|--------|------------------|\n"
            "| **Crude Oil (WTI)** | CL=F | Global energy benchmark, geopolitical sensitivity |\n"
            "| **Natural Gas** | NG=F | Seasonal demand, weather-driven volatility |\n\n"
            "### Agriculture\n"
            "| Asset | Symbol | Why it's notable |\n"
            "|-------|--------|------------------|\n"
            "| **Corn** | ZC=F | Staple crop, ethanol demand |\n"
            "| **Wheat** | ZW=F | Global food supply, geopolitical risks |\n"
            "| **Soybeans** | ZS=F | Animal feed, biodiesel |\n\n"
            "### Ask next\n"
            "- *\"Predict gold\"* or *\"Predict GC=F\"* for ML forecast\n"
            "- *\"Technical analysis of crude oil\"* for indicators\n\n"
            "---\n\n"
            "*Commodity futures carry substantial risk. Not financial advice.*"
        )

    # ---- US / GLOBAL RESPONSES ---- #
    if re.search(r"top\s*\d+\s*stocks?|best\s*stocks?", message_lower):
        return (
            "## Top US Stocks to Watch\n\n"
            "Here are widely-followed stocks across key sectors:\n\n"
            "**Technology:** AAPL (Apple), MSFT (Microsoft), NVDA (NVIDIA), GOOGL (Alphabet), META (Meta)\n\n"
            "**Consumer:** AMZN (Amazon), TSLA (Tesla), WMT (Walmart), COST (Costco)\n\n"
            "**Healthcare:** UNH (UnitedHealth), JNJ (Johnson & Johnson), ABBV (AbbVie)\n\n"
            "**Finance:** JPM (JPMorgan), V (Visa), MA (Mastercard)\n\n"
            "**ETFs for Diversification:** SPY (S&P 500), QQQ (Nasdaq-100), VTI (Total Market), VGT (Tech Sector)\n\n"
            "To get live prices and analysis for any of these, ask me: *\"Predict AAPL stock\"* or *\"Technical analysis of NVDA\"*\n\n"
            "**Disclaimer:** This is not financial advice. Always do your own research."
        )

    if amount or re.search(r"invest|where.*(?:invest|put.*money)|what.*(?:should|would).*(?:invest|buy)", message_lower):
        amt = amount or 10000
        return (
            f"## US Market Investment Guide (${amt:,.0f})\n\n"
            f"Here's a suggested allocation of ${amt:,.0f}:\n\n"
            f"### For Beginners / Lower Risk\n"
            f"- **SPY** (S&P 500 ETF): ${amt*0.30:,.0f} - Broad market diversification\n"
            f"- **BND** (Total Bond ETF): ${amt*0.20:,.0f} - Lower volatility, income\n"
            f"- **VTI** (Total Market ETF): ${amt*0.20:,.0f} - Full US market exposure\n"
            f"- **GLD** (Gold ETF): ${amt*0.10:,.0f} - Inflation hedge\n\n"
            f"### For Growth\n"
            f"- **AAPL** (Apple): ${amt*0.20:,.0f} - Largest tech company\n"
            f"- **MSFT** (Microsoft): ${amt*0.20:,.0f} - Cloud & AI leader\n"
            f"- **NVDA** (NVIDIA): ${amt*0.15:,.0f} - AI/GPU leader\n"
            f"- **GOOGL** (Alphabet): ${amt*0.15:,.0f} - Search & cloud\n"
            f"- **QQQ** (Nasdaq-100 ETF): ${amt*0.15:,.0f} - Tech-heavy index\n\n"
            f"### For Income / Dividends\n"
            f"- **SCHD** (Dividend ETF): ${amt*0.25:,.0f} - High-quality dividends\n"
            f"- **JNJ** (Johnson & Johnson): ${amt*0.20:,.0f} - Healthcare dividend king\n"
            f"- **KO** (Coca-Cola): ${amt*0.15:,.0f} - Consistent dividend payer\n\n"
            f"### Hedge Fund Alternatives\n"
            f"- **DBMF** (Managed Futures): ${amt*0.15:,.0f} - Hedge fund strategy ETF\n"
            f"- **BTAL** (Anti-Beta): ${amt*0.10:,.0f} - Market-neutral strategy\n\n"
            f"### General Tips\n"
            f"- Diversify across sectors and asset classes\n"
            f"- Dollar-cost averaging reduces timing risk\n"
            f"- Consider your risk tolerance and time horizon\n\n"
            f"Ask me about any specific stock for live prices and AI predictions!\n\n"
            f"**Disclaimer:** This is not financial advice. Always do your own research."
        )

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
            "**Disclaimer:** This is not financial advice. Always do your own research."
        )

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
            "**Disclaimer:** This is not financial advice. Always do your own research."
        )

    # Default market advice response
    return (
        "## Market & Investment Insights\n\n"
        "I can help you with:\n\n"
        "- **Top US stocks** - Ask: *\"What are the top 10 stocks?\"*\n"
        "- **Indian market** - Ask: *\"Best Indian stocks to invest in\"*\n"
        "- **Investment allocation** - Ask: *\"If I have $500 to invest, what should I buy?\"*\n"
        "- **Indian allocation** - Ask: *\"Invest ₹50,000 in Indian stocks\"*\n"
        "- **Hedge funds & ETFs** - Ask: *\"Tell me about hedge fund alternatives\"*\n"
        "- **Sector analysis** - Ask: *\"Which sectors are performing best?\"*\n"
        "- **Specific stock analysis** - Ask: *\"Predict AAPL stock\"* or *\"Infosys stock price\"*\n\n"
        "For the most detailed analysis, ask about a specific stock symbol and I'll fetch live data with AI predictions.\n\n"
        "**Disclaimer:** This is not financial advice. Always do your own research."
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

@app.get("/stock/quote/{symbol}")
@rate_limit_public
async def get_stock_quote(symbol: str):
    """Public endpoint to fetch live stock quote."""
    try:
        data = fetch_stock_data(symbol.upper())
        if data:
            return {"status": "ok", "data": data}
        return {"status": "error", "message": f"Could not fetch data for {symbol}", "data": None}
    except Exception as e:
        logger.error(f"Stock quote error for {symbol}: {e}")
        return {"status": "error", "message": str(e), "data": None}

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
        response = generate_response(request.message, user_id,
                                     session_id=request.session_id)

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
    """Enhanced stock prediction endpoint using XGBoost+LightGBM ensemble."""
    try:
        if ENHANCED_MODULES_AVAILABLE:
            validate_api_request(symbol)

        # Try enhanced ensemble model first
        if feature_pipeline and xgboost_predictor:
            try:
                features = feature_pipeline.build_features(symbol, include_live_bar=True)
                prediction = xgboost_predictor.predict(symbol, features)
                # Save to database
                if current_user and ENHANCED_MODULES_AVAILABLE:
                    try:
                        db_manager.save_prediction(
                            current_user["id"], symbol,
                            prediction, prediction.get('horizon_days', 21)
                        )
                    except Exception as e:
                        logger.warning(f"Failed to save prediction: {e}")

                return {
                    "symbol": symbol,
                    "direction": prediction['direction'],
                    "probability": prediction['probability'],
                    "probability_raw": prediction.get('probability_raw', prediction['probability']),
                    "confidence": prediction['confidence'],
                    "expected_return": prediction['expected_return'],
                    "horizon_days": prediction.get('horizon_days', 21),
                    "ensemble": prediction.get('ensemble', False),
                    "feature_importance": prediction.get('feature_importance', {}),
                    "timestamp": datetime.now().isoformat(),
                }
            except Exception as e:
                logger.warning(f"Enhanced prediction failed for {symbol}, trying legacy: {e}")

        # Fallback to legacy LSTM model
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

        if current_user and ENHANCED_MODULES_AVAILABLE:
            try:
                db_manager.save_prediction(current_user["id"], symbol, prediction, days_ahead)
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

    @app.post("/auth/change-password")
    @rate_limit_sensitive
    async def change_password(
        request: ChangePasswordRequest,
        current_user: Dict = Depends(get_current_active_user)
    ):
        """Change password for authenticated user."""
        try:
            auth_manager.change_password(
                current_user["id"],
                request.current_password,
                request.new_password
            )
            return {"message": "Password changed successfully"}
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Change password error: {e}")
            raise HTTPException(status_code=500, detail="Failed to change password")

    @app.post("/auth/forgot-password")
    @rate_limit_sensitive
    async def forgot_password(request: ForgotPasswordRequest):
        """Request password reset email. Always returns success to prevent email enumeration."""
        try:
            token = auth_manager.create_password_reset(request.email)
            if token:
                user = db_manager.get_user_by_email(request.email)
                first_name = user.get("first_name", "User") if user else "User"
                email_service.send_password_reset_email(request.email, token, first_name)

            return {
                "message": "If an account with that email exists, a password reset link has been sent."
            }
        except Exception as e:
            logger.error(f"Forgot password error: {e}")
            return {
                "message": "If an account with that email exists, a password reset link has been sent."
            }

    @app.post("/auth/reset-password")
    @rate_limit_sensitive
    async def reset_password(request: ResetPasswordRequest):
        """Reset password using a valid token."""
        try:
            auth_manager.reset_password(request.token, request.new_password)
            return {"message": "Password has been reset successfully. You can now log in with your new password."}
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Reset password error: {e}")
            raise HTTPException(status_code=500, detail="Failed to reset password")

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
        """List user portfolios with live-computed totals."""
        try:
            portfolios = db_manager.get_user_portfolios(current_user["id"])
            for portfolio in portfolios:
                total_invested = 0.0
                total_current = 0.0
                has_live_data = False
                for stock in portfolio.get('stocks', []):
                    shares = stock.get('shares', 0)
                    purchase_price = stock.get('purchase_price', 0)
                    cost = shares * purchase_price
                    total_invested += cost
                    try:
                        live = fetch_stock_data(stock['symbol'])
                        if live and live.get('price'):
                            total_current += shares * live['price']
                            has_live_data = True
                            logger.info(f"Portfolio list: {stock['symbol']} live price=${live['price']}")
                        else:
                            total_current += cost
                            logger.warning(f"Portfolio list: No live data for {stock['symbol']}, using cost ${cost}")
                    except Exception as e:
                        total_current += cost
                        logger.error(f"Portfolio list: fetch_stock_data failed for {stock['symbol']}: {e}")
                portfolio['total_value'] = round(total_current, 2)
                portfolio['total_gain_loss'] = round(total_current - total_invested, 2)
                portfolio['has_live_data'] = has_live_data
            return {"portfolios": portfolios}
        except Exception as e:
            logger.error(f"Portfolio list error: {e}")
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

    @app.delete("/portfolio/{portfolio_id}")
    @rate_limit_authenticated
    async def delete_portfolio_endpoint(
        portfolio_id: str,
        current_user: Dict = Depends(get_current_active_user)
    ):
        """Delete an entire portfolio and all its stocks."""
        try:
            result = db_manager.delete_portfolio(portfolio_id, current_user['id'])
            if not result:
                raise HTTPException(status_code=404, detail="Portfolio not found")
            return {"message": "Portfolio deleted successfully"}
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

    # ML-powered endpoints
    @app.post("/portfolio/recommend")
    @rate_limit_authenticated
    async def portfolio_recommend(
        request: PortfolioRecommendRequest,
        current_user: Dict = Depends(get_current_active_user)
    ):
        """AI-optimized portfolio allocation recommendation."""
        try:
            if not portfolio_optimizer:
                raise HTTPException(status_code=503, detail="Portfolio optimizer not available")

            result = portfolio_optimizer.recommend_allocation(
                request.amount, request.risk_level, request.horizon_months
            )

            # Run Monte Carlo forecast
            forecast = None
            if monte_carlo_sim and result.get('allocations'):
                symbols = list(result['allocations'].keys())
                weights = [result['allocations'][s]['weight'] for s in symbols]
                try:
                    forecast = monte_carlo_sim.simulate(
                        symbols, weights, request.amount, request.horizon_months
                    )
                except Exception as e:
                    logger.warning(f"Monte Carlo forecast failed: {e}")

            return {
                "allocations": result.get('allocations', {}),
                "expected_return_range": result.get('expected_return_range', {}),
                "risk_metrics": result.get('risk_metrics', {}),
                "forecast": forecast,
                "method": result.get('method', ''),
                "risk_level": result.get('risk_level', ''),
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Portfolio recommend error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/market-summary")
    @rate_limit_authenticated
    async def market_summary(current_user: Dict = Depends(get_current_active_user)):
        """Market regime, sector analysis, macro indicators."""
        try:
            # Fetch major indices
            indices = {}
            for sym in ['SPY', 'QQQ', 'DIA']:
                data = fetch_stock_data(sym)
                if data:
                    indices[sym] = data

            # Fetch macro indicators from DB
            macro_data = {}
            try:
                from datetime import date
                end = date.today()
                start = date(end.year - 1, end.month, end.day)
                indicators = db_manager.get_macro_indicators(
                    ['DFF', 'CPIAUCSL', 'UNRATE', 'T10Y2Y'], start, end
                )
                # Get latest value for each indicator
                for ind in indicators:
                    name = ind['indicator_name']
                    if name not in macro_data or ind['date'] > macro_data[name]['date']:
                        macro_data[name] = {'value': ind['value'], 'date': str(ind['date'])}
            except Exception as e:
                logger.warning(f"Failed to fetch macro data: {e}")

            # Sector sentiment (if sentiment analyzer available)
            sector_sentiment = {}
            if sentiment_analyzer:
                for sector_sym in ['XLK', 'XLV', 'XLF', 'XLE', 'XLY']:
                    try:
                        sent = sentiment_analyzer.analyze_symbol(sector_sym, days=3)
                        sector_sentiment[sector_sym] = {
                            'sentiment': sent.get('overall_sentiment', 0),
                            'label': sent.get('overall_label', 'neutral'),
                        }
                    except Exception:
                        pass

            # Determine regime using ML-based regime detector
            regime_info = {'regime': 'neutral', 'confidence': 0.0}
            if regime_detector and feature_pipeline:
                try:
                    spy_features = feature_pipeline.build_features('SPY')
                    regime_info = regime_detector.detect_regime(spy_features)
                except Exception as e:
                    logger.warning(f"Regime detection failed: {e}")
                    # Fallback to simple price-based
                    spy_data = indices.get('SPY', {})
                    if spy_data:
                        change_pct = spy_data.get('changePercent', 0)
                        if change_pct > 1:
                            regime_info = {'regime': 'bull_normal_vol', 'confidence': 0.6}
                        elif change_pct < -1:
                            regime_info = {'regime': 'bear_normal_vol', 'confidence': 0.6}

            return {
                "indices": indices,
                "macro_indicators": macro_data,
                "sector_sentiment": sector_sentiment,
                "regime": regime_info.get('regime', 'neutral'),
                "regime_confidence": regime_info.get('confidence', 0),
                "regime_details": regime_info.get('details', {}),
            }
        except Exception as e:
            logger.error(f"Market summary error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/risk-analysis")
    @rate_limit_authenticated
    async def risk_analysis(
        request: RiskAnalysisRequest,
        current_user: Dict = Depends(get_current_active_user)
    ):
        """Portfolio risk analysis."""
        try:
            if not portfolio_optimizer:
                raise HTTPException(status_code=503, detail="Portfolio optimizer not available")

            symbols = request.symbols
            weights = request.weights

            # If portfolio_id provided, load from DB
            if request.portfolio_id:
                portfolio = db_manager.get_portfolio(request.portfolio_id)
                if not portfolio or portfolio.get('user_id') != current_user['id']:
                    raise HTTPException(status_code=404, detail="Portfolio not found")

                stocks = portfolio.get('stocks', [])
                if not stocks:
                    raise HTTPException(status_code=400, detail="Portfolio has no stocks")

                symbols = [s['symbol'] for s in stocks]
                vals = [s['shares'] * s['purchase_price'] for s in stocks]
                total = sum(vals)
                weights = [v / total for v in vals] if total > 0 else []

            if not symbols or not weights:
                raise HTTPException(status_code=400, detail="Symbols and weights are required")

            result = portfolio_optimizer.risk_analysis(symbols, weights)
            return result
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Risk analysis error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/sentiment/{symbol}")
    @rate_limit_authenticated
    async def get_sentiment(
        symbol: str,
        current_user: Dict = Depends(get_current_active_user)
    ):
        """NLP sentiment analysis for a stock."""
        try:
            if not sentiment_analyzer:
                raise HTTPException(status_code=503, detail="Sentiment analyzer not available")

            result = sentiment_analyzer.analyze_symbol(symbol.upper())
            return result
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Sentiment error for {symbol}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/forecast")
    @rate_limit_authenticated
    async def forecast(
        request: ForecastRequest,
        current_user: Dict = Depends(get_current_active_user)
    ):
        """Monte Carlo portfolio forecast."""
        try:
            if not monte_carlo_sim:
                raise HTTPException(status_code=503, detail="Monte Carlo simulator not available")

            result = monte_carlo_sim.simulate(
                request.symbols, request.weights, request.amount, request.months
            )
            return result
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Forecast error: {e}")
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
            response = generate_response(request.message,
                                         session_id=request.session_id)
            return ChatResponse(**response)
        except Exception as e:
            logger.error(f"Chat error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/predict/{symbol}")
    @rate_limit_public
    async def predict_stock_fallback(symbol: str, days_ahead: int = 5):
        """Fallback prediction endpoint without authentication."""
        try:
            # Try enhanced ensemble model first
            if feature_pipeline and xgboost_predictor:
                try:
                    features = feature_pipeline.build_features(symbol, include_live_bar=True)
                    prediction = xgboost_predictor.predict(symbol, features)
                    return {
                        "symbol": symbol,
                        "direction": prediction['direction'],
                        "probability": prediction['probability'],
                        "confidence": prediction['confidence'],
                        "expected_return": prediction['expected_return'],
                        "horizon_days": prediction.get('horizon_days', 21),
                        "ensemble": prediction.get('ensemble', False),
                        "timestamp": datetime.now().isoformat(),
                    }
                except Exception as e:
                    logger.warning(f"Enhanced prediction failed for {symbol}: {e}")

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
    global feature_pipeline, xgboost_predictor, monte_carlo_sim
    global portfolio_optimizer, sentiment_analyzer, regime_detector
    global av_collector

    logger.info("Starting AI Stock GPT Enhanced API v2.0")

    # Create database tables if they don't exist
    try:
        try:
            from backend.db_session import engine, DATABASE_URL
            from backend.models import Base
        except ImportError:
            from db_session import engine, DATABASE_URL
            from models import Base
        db_host = DATABASE_URL.split("@")[-1].split("/")[0] if "@" in DATABASE_URL else "unknown"
        logger.info("Database target: %s", db_host)
        Base.metadata.create_all(bind=engine)
        if ENHANCED_MODULES_AVAILABLE:
            db_manager.ensure_admin_schema()
            admin_emails = os.getenv("ADMIN_EMAILS", "")
            if admin_emails.strip():
                promoted = db_manager.promote_admin_by_emails(
                    [e.strip() for e in admin_emails.split(",") if e.strip()]
                )
                if promoted:
                    logger.info(f"Promoted {promoted} user(s) to admin via ADMIN_EMAILS")
        logger.info("Database tables initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database tables: {e}")

    try:
        initialize_nlp()
        logger.info("Application initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")

    # Initialize ML Engine
    try:
        from backend.data.alphavantage_collector import AlphaVantageCollector
        from backend.ml.feature_pipeline import FeaturePipeline
        from backend.ml.xgboost_model import XGBoostPredictor
        from backend.ml.monte_carlo import MonteCarloSimulator
        from backend.ml.portfolio_optimizer import PortfolioOptimizer
        from backend.ml.sentiment import SentimentAnalyzer
        from backend.ml.regime_detector import RegimeDetector

        _db = db_manager if ENHANCED_MODULES_AVAILABLE else None
        av_collector = AlphaVantageCollector()

        # Initialize fundamental data collector and investment advisors
        try:
            from backend.data.fundamental_collector import FundamentalCollector
            from backend.advisor.investment_advisor import BuffettScore, JhunjhunwalaScore
            fundamental_collector = FundamentalCollector()
            buffett_scorer = BuffettScore(fundamental_collector)
            jhunjhunwala_scorer = JhunjhunwalaScore(fundamental_collector)
            logger.info("Fundamental data collector and investment advisors initialized (Buffett + Jhunjhunwala)")
        except Exception as e:
            logger.warning(f"Fundamental/advisor init failed (will use fallbacks): {e}")

        feature_pipeline = FeaturePipeline(db_manager=_db, av_collector=av_collector, fundamental_collector=fundamental_collector)
        xgboost_predictor = XGBoostPredictor()
        monte_carlo_sim = MonteCarloSimulator(av_collector=av_collector)
        portfolio_optimizer = PortfolioOptimizer(av_collector=av_collector)
        sentiment_analyzer = SentimentAnalyzer()
        regime_detector = RegimeDetector()
        if _db:
            sentiment_analyzer.set_db_manager(_db)
        logger.info("ML Engine initialized: XGBoost+LightGBM Ensemble, Monte Carlo, PyPortfolioOpt, FinBERT, RegimeDetector")
    except Exception as e:
        logger.warning(f"ML Engine init failed (will use fallbacks): {e}")

    # Initialize Data Scheduler
    try:
        from backend.data.scheduler import DataScheduler
        from backend.data.fred_collector import FREDCollector

        _db = db_manager if ENHANCED_MODULES_AVAILABLE else None
        fred_collector = FREDCollector()
        data_scheduler = DataScheduler(_db, fred_collector, sentiment_analyzer)
        data_scheduler.start()
        logger.info("Data scheduler started")
    except Exception as e:
        logger.warning(f"Data scheduler failed to start: {e}")


def _mount_web_ui_if_present() -> None:
    """Serve React build from static/ui (Railway combined deploy).

    StaticFiles with html=True only maps ``/`` -> ``index.html``.  For SPA
    client-side routes like ``/admin``, ``/login``, ``/portfolio`` we need an
    explicit catch-all that returns ``index.html`` so React Router can handle
    the path.
    """
    ui_dir = os.getenv(
        "STATIC_UI_DIR",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "static", "ui"),
    )
    if not os.path.isdir(ui_dir):
        return
    try:
        from fastapi.staticfiles import StaticFiles
        from starlette.responses import FileResponse

        index_html = os.path.join(ui_dir, "index.html")

        # Serve JS/CSS/media assets from the build directory
        app.mount("/static", StaticFiles(directory=os.path.join(ui_dir, "static")), name="web-static")

        # SPA catch-all: any GET that didn't match an API route returns index.html
        @app.get("/{full_path:path}")
        async def _spa_fallback(full_path: str):
            # If the request maps to a real file in ui_dir, serve it (manifest.json, etc.)
            file_path = os.path.join(ui_dir, full_path)
            if full_path and os.path.isfile(file_path):
                return FileResponse(file_path)
            return FileResponse(index_html)

        logger.info("Web UI mounted from %s with SPA fallback (admin: /admin, login: /login)", ui_dir)
    except Exception as e:
        logger.warning(f"Could not mount web UI: {e}")


_mount_web_ui_if_present()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
