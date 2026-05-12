# AI Portfolio Intelligence Platform - Build Specification

> **Instructions for Claude:** This is a self-contained build specification. Read the entire document before starting. Section 1 describes everything that already exists -- DO NOT rebuild it. Section 2 defines the architecture. Sections 3-6 contain ordered build tasks. Execute them in order. Each task lists exact files, function signatures, and integration points.

## Project Info

- **Repository:** github.com/saji1970/AIStockGpt (branch: `main`)
- **Runtime:** Python 3.11, Node 18, React Native 0.85
- **Backend entry:** `uvicorn backend.main_enhanced:app --host 0.0.0.0 --port ${PORT:-8080}`
- **Database:** PostgreSQL (Railway-hosted, auto-created tables via `Base.metadata.create_all`)
- **Deployment:** Railway auto-deploy from GitHub push (uses `Dockerfile.backend`)
- **Production URL:** `https://aistockgpt-production.up.railway.app`

---

## 1. EXISTING CODEBASE (DO NOT REBUILD)

Everything below already exists and works. Reference these files, classes, and functions -- do not recreate them.

### 1.1 Backend Files

#### `backend/main_enhanced.py` (1160 lines)

FastAPI app with all current endpoints.

**App instance:** `app = FastAPI(title="AI Stock GPT Enhanced API", version="2.0.0")`

**Globals:**
```python
nlp_processor = None          # EnhancedNLPProcessor instance
models_cache = {}             # Dict[symbol, {model, data_collector, loaded}]
data_collectors = {}          # Dict[symbol, StockDataCollector]
```

**Pydantic models (already defined):**
- `ChatRequest(message: str, timestamp: Optional[str])`
- `ChatResponse(message: str, stockData: Optional[Dict], charts: Optional[Dict], confidence: Optional[float])`
- `PredictionRequest(symbol: str, days_ahead: Optional[int] = 5)`
- `PortfolioCreate(name: str, description: Optional[str])`
- `StockAdd(symbol: str, shares: float, purchase_price: float, purchase_date: str)`
- `EmailAlert(symbol: str, alert_type: str, threshold: float, email: str)`

**Key functions:**
- `initialize_nlp()` -- loads EnhancedNLPProcessor or falls back to regex NLP
- `get_or_create_model(symbol: str) -> Dict` -- loads/creates LSTM model for a symbol
- `fetch_stock_data(symbol: str) -> Optional[Dict]` -- 3-tier fallback: RapidAPI -> Yahoo Finance v8 -> yfinance. Returns `{symbol, price, previousClose, change, changePercent, high, low, volume, name, open}`
- `generate_response(message: str, user_id: Optional[str] = None) -> Dict` -- **THIS IS THE MAIN INTEGRATION POINT.** Routes NLP intent to handlers, fetches stock data, calls LLM, saves chat history. Returns `{message, stockData, charts, confidence}`
- `handle_general_question(message: str) -> str` -- template responses
- `handle_market_advice(message: str) -> str` -- market/investment template responses

**Existing endpoints (21 total):**

| Method | Route | Auth | Handler |
|--------|-------|------|---------|
| GET | `/` | No | `root()` |
| GET | `/status` | No | `get_status()` |
| GET | `/health` | No | `health_check()` |
| GET | `/stock/quote/{symbol}` | No | `get_stock_quote()` |
| POST | `/chat` | Yes* | `chat()` -- calls `generate_response()` |
| GET | `/predict/{symbol}` | Yes* | `predict_stock()` |
| GET | `/technical/{symbol}` | Yes* | `technical_analysis()` |
| POST | `/auth/register` | No | `register_user()` |
| POST | `/auth/login` | No | `login_user()` |
| POST | `/auth/refresh` | No | `refresh_token()` |
| GET | `/auth/me` | Yes | `get_user_profile()` |
| POST | `/portfolio/create` | Yes | `create_portfolio()` |
| GET | `/portfolio/list` | Yes | `list_portfolios()` -- computes live totals |
| GET | `/portfolio/{id}/summary` | Yes | `get_portfolio_summary()` -- per-stock gain/loss |
| GET | `/portfolio/{id}` | Yes | `get_portfolio()` |
| POST | `/portfolio/{id}/add-stock` | Yes | `add_stock_to_portfolio()` |
| DELETE | `/portfolio/{id}/stock/{symbol}` | Yes | `delete_stock_from_portfolio_endpoint()` |
| GET | `/chat/history` | Yes | `get_chat_history()` |
| GET | `/analytics/user` | Yes | `get_user_analytics()` |
| POST | `/alerts/create` | Yes | `create_email_alert()` |
| GET | `/alerts/list` | Yes | `list_email_alerts()` |

**Startup event:** `Base.metadata.create_all(bind=engine)` then `initialize_nlp()`

**Feature flags (import guards):**
- `ORIGINAL_MODULES_AVAILABLE` -- data_collector, lstm_model, sensitivity_analysis, nlp_processor
- `ENHANCED_NLP_AVAILABLE` -- nlp_enhanced, llm_provider
- `ENHANCED_MODULES_AVAILABLE` -- database, auth, security

---

#### `backend/models.py` (118 lines)

SQLAlchemy ORM models. `Base = declarative_base()`

**Table: users**
```
id: String(64) PK
email: String(255) UNIQUE INDEX
hashed_password: Text
first_name: String(100)
last_name: String(100)
username: String(50) UNIQUE INDEX nullable
is_active: Boolean default=True
created_at: DateTime(tz)
updated_at: DateTime(tz)
last_login: DateTime(tz) nullable
Relationships: portfolios, chat_messages, predictions, email_alerts (cascade delete)
```

**Table: portfolios**
```
id: UUID PK default=uuid4
user_id: String(64) FK->users.id CASCADE INDEX
name: String(255)
description: Text nullable
total_value: Float default=0.0
total_gain_loss: Float default=0.0
created_at, updated_at: DateTime(tz)
Relationships: user (back), stocks (cascade delete)
```

**Table: stocks**
```
id: UUID PK default=uuid4
portfolio_id: UUID FK->portfolios.id CASCADE INDEX
symbol: String(10) INDEX
shares: Float default=0.0
purchase_price: Float default=0.0
purchase_date: Date nullable
added_at, updated_at: DateTime(tz)
UniqueConstraint('portfolio_id', 'symbol')
Relationship: portfolio (back)
```

**Table: chat_messages**
```
id: UUID PK, user_id: String(64) FK INDEX, message: Text, response: Text nullable, timestamp: DateTime(tz) INDEX
```

**Table: predictions**
```
id: UUID PK, user_id: String(64) FK INDEX, symbol: String(10) INDEX, prediction: JSONB, days_ahead: Integer default=5, status: String(20) INDEX default='pending', actual_price: Float nullable, accuracy: Float nullable, created_at, completed_at
```

**Table: email_alerts**
```
id: UUID PK, user_id: String(64) FK INDEX, symbol: String(10) INDEX, alert_type: String(50), threshold: Float, email: String(255), is_active: Boolean default=True, created_at, triggered_at nullable
```

---

#### `backend/database.py` (625 lines)

`class DatabaseManager` with global instance `db_manager = DatabaseManager()`

**Methods:**
```python
# Users
create_user(user_id: str, user_data: Dict) -> bool
get_user(user_id: str) -> Optional[Dict]
get_user_by_email(email: str) -> Optional[Dict]
get_user_by_username(username: str) -> Optional[Dict]
update_user(user_id: str, updates: Dict) -> bool

# Portfolios
create_portfolio(user_id, name, description) -> str  # returns portfolio_id
get_user_portfolios(user_id: str) -> List[Dict]       # includes stocks
get_portfolio(portfolio_id: str) -> Optional[Dict]     # includes stocks
add_stock_to_portfolio(portfolio_id, symbol, shares, purchase_price, purchase_date) -> bool
delete_stock_from_portfolio(portfolio_id: str, symbol: str) -> bool

# Chat
save_chat_message(user_id, message, response) -> str
get_chat_history(user_id: str, limit: int = 50) -> List[Dict]

# Predictions
save_prediction(user_id, symbol, prediction, days_ahead) -> str
update_prediction_accuracy(prediction_id, actual_price, accuracy) -> bool

# Analytics
get_user_analytics(user_id: str) -> Dict
get_system_analytics() -> Dict

# Alerts
create_email_alert(user_id, symbol, alert_type, threshold, email) -> Dict
get_user_alerts(user_id: str) -> List[Dict]
```

---

#### `backend/db_session.py` (34 lines)

```python
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/aistockgpt")
# Auto-fixes postgres:// -> postgresql:// for Railway
engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=10, max_overflow=20)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
def get_db(): ...  # FastAPI dependency
```

---

#### `backend/auth.py` (295 lines)

```python
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Pydantic: UserCreate, UserLogin, UserResponse, Token
# class AuthManager: hash_password, verify_password, create_access_token, create_refresh_token, verify_token, register_user, authenticate_user, login_user, refresh_access_token
# Global: auth_manager = AuthManager()
# DI: get_current_user(credentials) -> Dict, get_current_active_user(current_user) -> Dict
```

---

#### `backend/nlp_enhanced.py` (419 lines)

```python
# 7 intents: stock_prediction, technical_analysis, sensitivity_analysis, portfolio_management, market_news, market_advice, general_question
# INTENT_REFERENCES dict -- ML reference sentences per intent
# REGEX_INTENT_PATTERNS dict -- regex fallback patterns per intent
# STOCK_SYMBOLS list -- 68 known symbols

class EnhancedNLPProcessor:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2")
    def process_message(message: str) -> Tuple[str, Dict[str, Any], float]  # (intent, entities, confidence)
    def _classify_intent_ml(message: str) -> Tuple[str, float]              # cosine similarity
    def _classify_intent_regex(message: str) -> Tuple[str, float]           # regex fallback
    def _extract_entities(normalized, original) -> Dict                     # symbol, time_period, analysis_type
    def _extract_stock_symbols(normalized, original) -> List[str]
    def _extract_time_periods(message) -> List[str]
    def _extract_analysis_types(message) -> List[str]
    def validate_stock_symbol(symbol: str) -> bool
    def get_supported_symbols() -> List[str]
```

---

#### `backend/llm_provider.py` (309 lines)

```python
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

class LLMProvider:
    def generate_response(intent: str, entities: Dict, user_message: str) -> str
    def _build_prompt(intent, entities, user_message) -> str
    def _call_ollama(prompt: str) -> Optional[str]
    def _call_huggingface(prompt: str) -> Optional[str]
    def _template_response(intent, entities, user_message) -> str

# Global: llm_provider = LLMProvider()
```

---

#### `backend/security.py` (328 lines)

SecurityMiddleware, rate limit decorators (`rate_limit_public`, `rate_limit_authenticated`, `rate_limit_sensitive`), input validation (`SecurityConfig`, `StockSymbolRequest`, `ChatMessageRequest`, `UserRegistrationRequest`), sanitization utilities.

---

#### `data_collector.py` (261 lines, root level)

```python
class StockDataCollector:
    def __init__(self, symbol='AAPL', start_date='2020-01-01', end_date=None)
    def fetch_data() -> pd.DataFrame                    # yfinance OHLCV
    def add_technical_indicators(data) -> pd.DataFrame   # SMA, EMA, MACD, RSI, BB, Stoch, Volume_SMA, Price_Change, Volatility
    def create_features(data) -> pd.DataFrame            # time-based, lag, rolling stats
    def prepare_data(...) -> Tuple                       # train/test split, scaling, sequences
    def get_latest_data(days=60) -> np.array
```

---

#### `lstm_model.py` (root level)

```python
class LSTMModel:
    def build_model(lstm_units, dropout_rate, learning_rate) -> tf.keras.Model
    def train(X_train, y_train, ...) -> dict
    def predict(X) -> np.array
    def save_model(path, scaler_X, scaler_y, feature_names)
    def load_model(path)
    def get_feature_importance(X_sample, feature_names) -> dict
```

---

### 1.2 Frontend (React 18 + Tailwind)

```
src/App.js              -- Router: /login, / (Chat), /portfolio. ProtectedRoute wrapper
src/pages/LoginPage.js   -- Login/register form, useAuthStore
src/pages/ChatPage.js    -- Chat messages, quick prompts, StockCard sidebar, markdown rendering
src/pages/PortfolioPage.js -- Portfolio CRUD, add/remove stocks, gain/loss display
src/components/ChatMessage.js, StockCard.js, PortfolioCard.js, TypingIndicator.js
src/services/api.js      -- Axios client with JWT interceptor. Exports: sendMessage, getStockPrediction, getTechnicalAnalysis, getSensitivityAnalysis, loginUser, registerUser, getProfile, createPortfolio, listPortfolios, getPortfolio, getPortfolioSummary, addStockToPortfolio, deleteStockFromPortfolio
src/store/authStore.js   -- Zustand store: user, token, isAuthenticated, login(), register(), logout()
```

**Key deps:** react 18.2, react-router-dom 6, axios, zustand, tailwindcss 3.3, recharts 2.7, framer-motion, react-markdown

---

### 1.3 Mobile (React Native 0.85 + TypeScript)

```
mobile/src/screens/       -- LoginScreen, ChatScreen, PortfolioScreen, AlertsScreen, ProfileScreen
mobile/src/components/    -- ChatMessage, StockCard, PortfolioCard, QuickPrompts, TypingIndicator
mobile/src/api/client.ts  -- Axios with JWT. Same endpoint functions as web
mobile/src/auth/AuthContext.tsx -- React Context: login, register, logout, user
mobile/src/navigation/AppNavigator.tsx -- Bottom tabs: Chat, Portfolio, Alerts, Profile
mobile/src/config.ts      -- API_BASE_URL = 'https://aistockgpt-production.up.railway.app'
```

---

### 1.4 Infrastructure

```
Dockerfile.backend       -- Python 3.11-slim. COPY backend/ + root .py files. CMD uvicorn
docker-compose.yml       -- db (postgres:16-alpine), backend (:8080), frontend (:80)
railway.toml             -- DOCKERFILE builder, /health healthcheck
requirements.txt         -- 41 deps (see file for exact versions)
.gitignore               -- .env, models/*.h5, models/*.pkl, data/*.json, data/*.csv
```

---

### 1.5 Environment Variables (currently set)

```
DATABASE_URL              # PostgreSQL connection (Railway auto-sets)
JWT_SECRET_KEY            # JWT signing key
RAPIDAPI_KEY              # Real-Time Finance Data API
OLLAMA_BASE_URL           # Default: http://localhost:11434
OLLAMA_MODEL              # Default: llama3.2
HF_API_TOKEN              # HuggingFace (optional)
```

---

## 2. ARCHITECTURE

### Two-Layer AI System

```
Layer 1: LLM (Explanation Layer)
  - Receives structured data from ML models
  - Generates natural language explanations
  - Ollama -> HuggingFace -> Template fallback

Layer 2: ML Engine (Intelligence Layer)
  - XGBoost: Stock return probability predictions
  - Monte Carlo: Probabilistic outcome simulation (GBM, 10,000 paths)
  - PyPortfolioOpt: Portfolio allocation optimization (Sharpe, min-vol, risk parity)
  - FinBERT: Financial sentiment scoring from news headlines
  - Feature Pipeline: Technical/volatility/momentum indicators
```

**Design rule:** The LLM does NOT make predictions. It explains predictions made by ML models. Every ML output is consumed through the `/chat` endpoint via `generate_response()`.

### Query Routing (through `generate_response()` in `main_enhanced.py`)

```
User Query
  -> nlp_processor.process_message(message) -> (intent, entities, confidence)
  -> Route by intent:
       stock_prediction     -> XGBoostPredictor.predict() + fetch_stock_data()
       market_advice        -> PortfolioOptimizer + MonteCarloSimulator (if amount/risk/horizon in entities)
       market_news          -> SentimentAnalyzer.analyze_symbol()
       portfolio_management -> PortfolioOptimizer.risk_analysis() (if user has holdings)
       technical_analysis   -> FeaturePipeline.compute_indicators() + fetch_stock_data()
       sensitivity_analysis -> XGBoostPredictor feature importance
       general_question     -> LLM template
  -> llm_provider.generate_response(intent, entities, message, ml_results=...)
  -> Return ChatResponse {message, stockData, charts, confidence}
```

---

## 3. DEPENDENCIES TO ADD

Add these to `requirements.txt`:

```
# ML Engine
xgboost>=2.0.0
pyportfolioopt>=1.5.6

# FinBERT Sentiment
transformers>=4.35.0
torch>=2.0.0                 # CPU-only, needed by transformers for FinBERT

# Data Sources
finnhub-python>=2.4.19
fredapi>=0.5.2
alpha-vantage>=2.3.1

# Scheduler
apscheduler>=3.10.4
```

**New environment variables** (add to Railway dashboard):
```
FINNHUB_API_KEY=             # Free: finnhub.io (60 req/min)
ALPHA_VANTAGE_KEY=           # Free: alphavantage.co (25 req/day)
FRED_API_KEY=                # Free: fred.stlouisfed.org (unlimited)
```

---

## 4. DATABASE SCHEMA ADDITIONS

Add these 3 new models to `backend/models.py` (after the existing EmailAlert class):

### Table: market_data
```python
class MarketData(Base):
    __tablename__ = 'market_data'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    symbol = Column(String(10), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
    adjusted_close = Column(Float)
    source = Column(String(20), nullable=False)  # 'yfinance', 'alphavantage', 'finnhub'
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint('symbol', 'date', 'source', name='uq_market_data_symbol_date_source'),
    )
```

### Table: sentiment_scores
```python
class SentimentScore(Base):
    __tablename__ = 'sentiment_scores'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    symbol = Column(String(10), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    sentiment_score = Column(Float, nullable=False)       # -1.0 to 1.0
    sentiment_label = Column(String(10), nullable=False)  # 'positive', 'negative', 'neutral'
    confidence = Column(Float, nullable=False)
    source = Column(String(50))                            # e.g. 'finnhub', 'yahoo_rss'
    headline = Column(Text)
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
```

### Table: macro_indicators
```python
class MacroIndicator(Base):
    __tablename__ = 'macro_indicators'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    indicator_name = Column(String(50), nullable=False, index=True)  # 'DFF', 'CPIAUCSL', 'UNRATE', etc.
    date = Column(Date, nullable=False, index=True)
    value = Column(Float, nullable=False)
    source = Column(String(20), nullable=False, default='fred')
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint('indicator_name', 'date', name='uq_macro_indicator_date'),
    )
```

### New DatabaseManager methods

Add to `backend/database.py` (class `DatabaseManager`):

```python
# Market Data
def save_market_data(self, symbol: str, date, ohlcv: Dict, source: str) -> str
def get_market_data(self, symbol: str, start_date, end_date) -> List[Dict]
def get_tracked_symbols(self) -> List[str]  # unique symbols from stocks table

# Sentiment
def save_sentiment_score(self, symbol: str, date, score: float, label: str, confidence: float, source: str, headline: str) -> str
def get_sentiment_scores(self, symbol: str, days: int = 30) -> List[Dict]

# Macro
def save_macro_indicator(self, name: str, date, value: float, source: str = 'fred') -> str
def get_macro_indicators(self, names: List[str], start_date, end_date) -> List[Dict]
```

---

## 5. BUILD TASKS (execute in order)

### Task 1: Add DB models and methods

**Modify:** `backend/models.py`
- Add `MarketData`, `SentimentScore`, `MacroIndicator` classes (schemas above)
- Add imports if needed: existing imports cover everything

**Modify:** `backend/database.py`
- Add imports: `from .models import MarketData, SentimentScore, MacroIndicator`
- Add 7 new methods to `DatabaseManager` (signatures above)
- Use upsert pattern (query first, update or insert) for market_data and macro_indicators to handle UniqueConstraint

**Verify:** Start app. `GET /health` returns 200. Check PostgreSQL for new tables.

---

### Task 2: Feature engineering pipeline

**Create:** `backend/ml/__init__.py` (empty)

**Create:** `backend/ml/feature_pipeline.py`

```python
class FeaturePipeline:
    """Unified feature engineering: technical, volatility, momentum, macro."""

    def __init__(self, db_manager=None):
        """Uses db_manager for macro data. Falls back to data_collector for price data."""

    def build_features(self, symbol: str, lookback_days: int = 504) -> pd.DataFrame:
        """
        Fetch price data via yfinance, compute all features, return clean DataFrame.
        Columns (~50 total):
          - OHLCV (5)
          - Technical: SMA_20, SMA_50, EMA_12, EMA_26, MACD, MACD_signal, RSI_14, BB_upper, BB_lower, BB_middle, ATR_14, OBV, Stoch_K, Stoch_D, ADX_14 (15)
          - Volatility: hist_vol_20, hist_vol_60, garman_klass, atr_ratio, vol_of_vol (5)
          - Momentum: return_5d, return_21d, return_63d, return_252d, beta_spy, rolling_sharpe_21 (6)
          - Lag: close_lag_1, close_lag_5, volume_lag_1, volume_lag_5 (4)
          - Rolling: close_rolling_mean_20, close_rolling_std_20, volume_rolling_mean_20 (3)
          - Time: day_of_week, month, quarter (3)
          - Macro: fed_rate, cpi_yoy, unemployment, t10y2y (4) -- from macro_indicators table
        Drop NaN rows. Return DataFrame indexed by date.
        """

    def compute_indicators(self, symbol: str) -> Dict[str, Any]:
        """
        Compute latest indicator values for a symbol.
        Returns dict with current RSI, MACD, BB position, ATR, etc.
        Used by technical_analysis intent in chat.
        """
```

**Integration:** Import in `main_enhanced.py`:
```python
try:
    from backend.ml.feature_pipeline import FeaturePipeline
    feature_pipeline = FeaturePipeline(db_manager=db_manager)
except ImportError:
    feature_pipeline = None
```

**Verify:** `python -c "from backend.ml.feature_pipeline import FeaturePipeline; fp = FeaturePipeline(); df = fp.build_features('AAPL'); print(df.shape)"`

---

### Task 3: XGBoost model

**Create:** `backend/ml/xgboost_model.py`

```python
class XGBoostPredictor:
    """XGBoost model with walk-forward validation for stock prediction."""

    def __init__(self, models_dir: str = 'models'):
        """models_dir: where .joblib files are saved/loaded."""

    def train(self, symbol: str, features_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Walk-forward validation: 252-day train window, 21-day test window, rolling.
        Target: binary -- 1 if 21-day forward return > 0, else 0.
        Returns: {accuracy, precision, recall, f1, feature_importance: Dict[str, float]}
        Saves model to models/{symbol}_xgboost.joblib
        """

    def predict(self, symbol: str, features_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Load model, predict on latest features.
        Returns: {
            symbol: str,
            direction: 'bullish' | 'bearish',
            probability: float,          # probability of positive 21-day return
            expected_return: float,       # calibrated expected return
            confidence: float,            # model confidence (0-1)
            feature_importance: Dict[str, float],  # top 10 features
            horizon_days: 21
        }
        """

    def load_model(self, symbol: str) -> bool
    def save_model(self, symbol: str) -> bool
    def _create_target(self, df: pd.DataFrame, horizon: int = 21) -> pd.Series
```

**Dependencies:** `xgboost`, `joblib` (already in requirements.txt), `scikit-learn` (already)

**Integration:** Called from `generate_response()` when intent is `stock_prediction`:
```python
if feature_pipeline and xgboost_predictor:
    features = feature_pipeline.build_features(symbol)
    ml_prediction = xgboost_predictor.predict(symbol, features)
    # Pass ml_prediction to LLM for natural language explanation
```

**Verify:** Train on AAPL, check .joblib file created, predict returns valid dict.

---

### Task 4: Monte Carlo simulation

**Create:** `backend/ml/monte_carlo.py`

```python
class MonteCarloSimulator:
    """Monte Carlo simulation using Geometric Brownian Motion with correlated returns."""

    def simulate(
        self,
        symbols: List[str],
        weights: List[float],
        initial_amount: float,
        months: int,
        n_simulations: int = 10000
    ) -> Dict[str, Any]:
        """
        Simulate portfolio value over time.
        Uses historical returns to estimate mu (drift) and covariance matrix.
        Cholesky decomposition for correlated multi-asset simulation.

        Returns: {
            initial_amount: float,
            months: int,
            median_value: float,
            mean_value: float,
            percentiles: {5: float, 25: float, 50: float, 75: float, 95: float},
            probability_positive: float,     # % of paths with positive return
            probability_double: float,       # % of paths doubling initial amount
            expected_annual_return: float,
            var_95: float,                   # 95% Value at Risk (dollar loss)
            cvar_95: float,                  # Conditional VaR
            best_case: float,                # 95th percentile final value
            worst_case: float,               # 5th percentile final value
        }
        """

    def _estimate_parameters(self, symbols: List[str], lookback_days: int = 252) -> Tuple[np.array, np.array]:
        """Estimate annualized mu and covariance from historical yfinance data."""

    def _gbm_paths(self, mu, cov, weights, initial, days, n_sims) -> np.array:
        """Generate correlated GBM paths. Returns array of shape (n_sims, days)."""
```

**Dependencies:** `numpy`, `scipy` (already in requirements.txt), `yfinance` (already)

**Integration:** Called from `generate_response()` when intent is `market_advice` and entities contain investment amount/horizon.

**Verify:** `mc.simulate(['AAPL','MSFT'], [0.6,0.4], 1000, 12)` returns valid dict.

---

### Task 5: Portfolio optimizer

**Create:** `backend/ml/portfolio_optimizer.py`

```python
class PortfolioOptimizer:
    """Portfolio optimization using PyPortfolioOpt."""

    def optimize(
        self,
        symbols: List[str],
        method: str = 'max_sharpe',
        constraints: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Optimize portfolio allocation.
        Methods: 'max_sharpe', 'min_volatility', 'risk_parity'
        Fetches 2 years of price data via yfinance.

        Returns: {
            weights: Dict[str, float],       # {symbol: weight} -- weights sum to 1.0
            expected_annual_return: float,
            annual_volatility: float,
            sharpe_ratio: float,
            method: str,
            symbols: List[str]
        }
        """

    def risk_analysis(
        self,
        symbols: List[str],
        weights: List[float]
    ) -> Dict[str, Any]:
        """
        Compute risk metrics for a given allocation.

        Returns: {
            var_95: float,                  # 95% Value at Risk (% daily loss)
            cvar_95: float,                 # Conditional VaR
            max_drawdown: float,            # historical max drawdown
            sharpe_ratio: float,
            annual_volatility: float,
            beta: float,                    # vs SPY
            correlation_matrix: Dict        # pairwise correlations
        }
        """

    def recommend_allocation(
        self,
        amount: float,
        risk_level: str,                # 'conservative', 'moderate', 'aggressive'
        horizon_months: int
    ) -> Dict[str, Any]:
        """
        Generate a recommended allocation based on risk profile.
        Selects appropriate symbols and optimization method.

        Returns: {
            allocations: Dict[str, Dict],  # {symbol: {weight, amount, shares_approx}}
            risk_level: str,
            expected_return_range: Dict,    # {low, mid, high}
            risk_metrics: Dict,
            method: str
        }
        """

    def _get_price_data(self, symbols: List[str], period_days: int = 504) -> pd.DataFrame:
        """Fetch adjusted close prices via yfinance. Returns DataFrame with symbols as columns."""
```

**Dependencies:** `pyportfolioopt`, `yfinance` (already)

**Integration:** Called from `generate_response()` when intent is `market_advice` (with amount/risk/horizon) or `portfolio_management` (for risk analysis of user holdings).

**Verify:** `opt.optimize(['AAPL','MSFT','GOOGL','BND','GLD'], 'max_sharpe')` returns weights summing to ~1.0.

---

### Task 6: FinBERT sentiment analysis

**Create:** `backend/ml/sentiment.py`

```python
class SentimentAnalyzer:
    """FinBERT-based financial sentiment analysis."""

    def __init__(self):
        """
        Load ProsusAI/finbert from HuggingFace.
        Model is ~400MB, runs on CPU. Auto-downloads on first use.
        Uses transformers pipeline('sentiment-analysis', model='ProsusAI/finbert')
        """

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze a single text.
        Returns: {label: 'positive'|'negative'|'neutral', score: float, confidence: float}
        """

    def analyze_symbol(self, symbol: str, days: int = 7) -> Dict[str, Any]:
        """
        Fetch recent news headlines for symbol, analyze each, return aggregate.
        Returns: {
            symbol: str,
            overall_sentiment: float,        # -1 to 1 weighted average
            overall_label: str,              # 'positive', 'negative', 'neutral'
            confidence: float,
            trend: str,                      # 'improving', 'declining', 'stable'
            headline_count: int,
            recent_headlines: List[Dict],    # [{headline, label, score, date}]
        }
        Stores individual scores in sentiment_scores table via db_manager.
        """

    def _fetch_news(self, symbol: str, days: int = 7) -> List[Dict]:
        """
        Fetch news headlines.
        1. Try Finnhub API (env: FINNHUB_API_KEY) -- company_news endpoint
        2. Fallback: Yahoo Finance RSS feed
        Returns: [{headline: str, date: str, source: str}]
        """
```

**Dependencies:** `transformers`, `torch` (CPU), `finnhub-python`

**Note:** On Railway, FinBERT downloads ~400MB on first load. Subsequent starts use cached model. If Railway memory is tight, add lazy loading (load model only when first sentiment request comes in).

**Integration:** Called from `generate_response()` when intent is `market_news`.

**Verify:** `analyzer.analyze_text("Apple reports record quarterly revenue")` returns `{label: 'positive', ...}`

---

### Task 7: Data ingestion and scheduler

**Create:** `backend/data/__init__.py` (empty)

**Create:** `backend/data/fred_collector.py`

```python
class FREDCollector:
    """Fetch macroeconomic indicators from FRED API."""

    INDICATORS = {
        'DFF': 'Federal Funds Rate',
        'CPIAUCSL': 'CPI (inflation)',
        'UNRATE': 'Unemployment Rate',
        'GDP': 'Gross Domestic Product',
        'T10Y2Y': '10Y-2Y Treasury Spread (yield curve)',
    }

    def __init__(self, api_key: str = None):
        """api_key from env FRED_API_KEY. Uses fredapi library."""

    def fetch_all(self, start_date: str = None) -> Dict[str, pd.Series]:
        """Fetch all indicators. Returns {indicator_name: pd.Series}."""

    def save_to_db(self, db_manager) -> int:
        """Fetch and store in macro_indicators table. Returns count of new rows."""
```

**Create:** `backend/data/finnhub_collector.py`

```python
class FinnhubCollector:
    """Fetch news and basic data from Finnhub API."""

    def __init__(self, api_key: str = None):
        """api_key from env FINNHUB_API_KEY. Uses finnhub-python."""

    def get_company_news(self, symbol: str, days: int = 7) -> List[Dict]:
        """Returns [{headline, summary, source, datetime, url}]"""

    def get_quote(self, symbol: str) -> Dict:
        """Returns {current_price, change, percent_change, high, low, open, previous_close}"""
```

**Create:** `backend/data/scheduler.py`

```python
class DataScheduler:
    """APScheduler-based data refresh."""

    def __init__(self, db_manager, fred_collector=None, sentiment_analyzer=None):

    def start(self):
        """
        Start scheduler with two jobs:
        1. Daily at 18:00 ET: refresh FRED macro data, refresh market_data for tracked symbols
        2. Every 4 hours: refresh sentiment for top-held symbols (from stocks table)
        """

    def stop(self):
        """Shutdown scheduler gracefully."""

    def _daily_refresh(self):
        """Job: fetch FRED data, fetch market_data for tracked symbols via yfinance."""

    def _sentiment_refresh(self):
        """Job: run SentimentAnalyzer on top-held symbols."""
```

**Integration in `main_enhanced.py` startup event:**
```python
# After initialize_nlp()
try:
    from backend.data.scheduler import DataScheduler
    data_scheduler = DataScheduler(db_manager, fred_collector, sentiment_analyzer)
    data_scheduler.start()
    logger.info("Data scheduler started")
except Exception as e:
    logger.warning(f"Data scheduler failed to start: {e}")
```

**Verify:** Trigger `_daily_refresh()` manually, check `macro_indicators` table for FRED data.

---

### Task 8: Enhance NLP entity extraction

**Modify:** `backend/nlp_enhanced.py`

Add 3 new entity extractors to `_extract_entities()`:

```python
def _extract_dollar_amount(self, message: str) -> Optional[float]:
    """Extract dollar amounts: '$500', '500 dollars', '500 usd', 'invest 500'"""
    # Patterns: \$\d+, \d+\s*(?:dollars?|usd|bucks), invest\s*\$?\d+

def _extract_risk_level(self, message: str) -> Optional[str]:
    """Extract risk level: 'conservative', 'moderate', 'aggressive', 'low risk', 'high risk'"""
    # Map synonyms: safe/low/conservative -> 'conservative', balanced/moderate/medium -> 'moderate', aggressive/high/growth -> 'aggressive'

def _extract_horizon(self, message: str) -> Optional[int]:
    """Extract investment horizon in months: '1 year' -> 12, '6 months' -> 6, '5 years' -> 60"""
```

Update `_extract_entities()` to call these and add to entities dict:
```python
entities['amount'] = self._extract_dollar_amount(normalized)
entities['risk_level'] = self._extract_risk_level(normalized)
entities['horizon_months'] = self._extract_horizon(normalized)
```

**Verify:** `nlp.process_message("I have $500, moderate risk, 1 year")` returns entities `{amount: 500, risk_level: 'moderate', horizon_months: 12}`.

---

### Task 9: Wire ML models into generate_response()

**Modify:** `backend/main_enhanced.py`

Add global ML instances after `nlp_processor`:
```python
# ML Engine instances
feature_pipeline = None
xgboost_predictor = None
monte_carlo_sim = None
portfolio_optimizer = None
sentiment_analyzer = None
```

Add initialization in `startup_event()` (after `initialize_nlp()`):
```python
try:
    from backend.ml.feature_pipeline import FeaturePipeline
    from backend.ml.xgboost_model import XGBoostPredictor
    from backend.ml.monte_carlo import MonteCarloSimulator
    from backend.ml.portfolio_optimizer import PortfolioOptimizer
    from backend.ml.sentiment import SentimentAnalyzer
    feature_pipeline = FeaturePipeline(db_manager=db_manager)
    xgboost_predictor = XGBoostPredictor()
    monte_carlo_sim = MonteCarloSimulator()
    portfolio_optimizer = PortfolioOptimizer()
    sentiment_analyzer = SentimentAnalyzer()
    logger.info("ML Engine initialized: XGBoost, Monte Carlo, PyPortfolioOpt, FinBERT")
except Exception as e:
    logger.warning(f"ML Engine init failed (will use fallbacks): {e}")
```

**Modify `generate_response()` to call ML models based on intent:**

```python
# After NLP processing and before LLM response:
ml_results = {}

if intent == "stock_prediction" and symbol and xgboost_predictor:
    try:
        features = feature_pipeline.build_features(symbol)
        ml_results['prediction'] = xgboost_predictor.predict(symbol, features)
    except Exception as e:
        logger.warning(f"XGBoost prediction failed: {e}")

if intent == "market_advice" and portfolio_optimizer:
    amount = entities.get('amount')
    risk = entities.get('risk_level', 'moderate')
    horizon = entities.get('horizon_months', 12)
    if amount:
        try:
            ml_results['allocation'] = portfolio_optimizer.recommend_allocation(amount, risk, horizon)
            # Run Monte Carlo on the recommended allocation
            alloc = ml_results['allocation']
            symbols = list(alloc['allocations'].keys())
            weights = [alloc['allocations'][s]['weight'] for s in symbols]
            ml_results['forecast'] = monte_carlo_sim.simulate(symbols, weights, amount, horizon)
        except Exception as e:
            logger.warning(f"Portfolio optimization failed: {e}")

if intent == "market_news" and symbol and sentiment_analyzer:
    try:
        ml_results['sentiment'] = sentiment_analyzer.analyze_symbol(symbol)
    except Exception as e:
        logger.warning(f"Sentiment analysis failed: {e}")

if intent == "portfolio_management" and user_id and portfolio_optimizer:
    # Run risk analysis on user's actual holdings
    try:
        portfolios = db_manager.get_user_portfolios(user_id)
        if portfolios and portfolios[0].get('stocks'):
            stocks = portfolios[0]['stocks']
            syms = [s['symbol'] for s in stocks]
            vals = [s['shares'] * s['purchase_price'] for s in stocks]
            total = sum(vals)
            wts = [v/total for v in vals] if total > 0 else []
            if syms and wts:
                ml_results['risk'] = portfolio_optimizer.risk_analysis(syms, wts)
    except Exception as e:
        logger.warning(f"Portfolio risk analysis failed: {e}")

# Pass ml_results to LLM for formatting
if ml_results:
    # Include ML results in the response
    if llm_provider:
        response_text = llm_provider.generate_response(intent, entities, message, ml_results=ml_results)
    else:
        response_text = _format_ml_results(ml_results, intent)
```

**Modify `llm_provider.py`:**

Update `LLMProvider.generate_response()` signature:
```python
def generate_response(self, intent, entities, user_message, ml_results=None) -> str:
```

Update `_build_prompt()` to include ML results in the prompt context:
```python
if ml_results:
    prompt += "\n\n## ML Model Results (use these to inform your response):\n"
    prompt += json.dumps(ml_results, indent=2, default=str)
    prompt += "\n\nExplain these results in clear, natural language. Include the key numbers."
```

Update `_template_response()` to format ML results when LLM is unavailable:
```python
if ml_results and ml_results.get('prediction'):
    pred = ml_results['prediction']
    return f"**{pred['symbol']} Analysis**\n\nDirection: {pred['direction']}\nProbability of positive return: {pred['probability']:.0%}\nConfidence: {pred['confidence']:.0%}\n\nTop factors: {', '.join(list(pred.get('feature_importance',{}).keys())[:5])}\n\n*21-day horizon. Not financial advice.*"

if ml_results and ml_results.get('allocation'):
    alloc = ml_results['allocation']
    lines = [f"**Recommended Allocation ({alloc['risk_level']})**\n"]
    for sym, info in alloc['allocations'].items():
        lines.append(f"- {sym}: {info['weight']:.0%} (${info['amount']:.0f})")
    if ml_results.get('forecast'):
        fc = ml_results['forecast']
        lines.append(f"\n**Monte Carlo ({fc['months']}mo, 10K simulations)**")
        lines.append(f"- Median outcome: ${fc['median_value']:,.0f}")
        lines.append(f"- {fc['probability_positive']:.0%} chance of positive return")
        lines.append(f"- Best case (95th): ${fc['best_case']:,.0f}")
        lines.append(f"- Worst case (5th): ${fc['worst_case']:,.0f}")
    return '\n'.join(lines) + "\n\n*Not financial advice.*"
```

**Verify:** Send chat "Predict AAPL stock" -- response should include XGBoost probability and direction. Send "I have $500 moderate risk 1 year" -- response should include allocation and Monte Carlo.

---

### Task 10: Add 5 new API endpoints

**Modify:** `backend/main_enhanced.py`

Add new Pydantic models:
```python
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
```

Add endpoints (inside the `if ENHANCED_MODULES_AVAILABLE:` block):

```python
@app.post("/portfolio/recommend")
@rate_limit_authenticated
async def portfolio_recommend(request: PortfolioRecommendRequest, current_user: Dict = Depends(get_current_active_user)):
    """AI-optimized portfolio allocation recommendation."""
    # Calls portfolio_optimizer.recommend_allocation() + monte_carlo_sim.simulate()
    # Returns: {allocations, expected_return_range, risk_metrics, forecast, sentiment_context}

@app.get("/market-summary")
@rate_limit_authenticated
async def market_summary(current_user: Dict = Depends(get_current_active_user)):
    """Market regime, sector analysis, macro indicators."""
    # Fetches SPY/QQQ/DIA via fetch_stock_data(), FRED macro data, sector sentiment
    # Returns: {indices, sectors, macro_indicators, overall_sentiment, regime}

@app.post("/risk-analysis")
@rate_limit_authenticated
async def risk_analysis(request: RiskAnalysisRequest, current_user: Dict = Depends(get_current_active_user)):
    """Portfolio risk analysis."""
    # If portfolio_id: load from DB. Else use symbols/weights from request.
    # Calls portfolio_optimizer.risk_analysis()
    # Returns: {var_95, cvar_95, max_drawdown, sharpe, beta, correlation_matrix}

@app.get("/sentiment/{symbol}")
@rate_limit_authenticated
async def get_sentiment(symbol: str, current_user: Dict = Depends(get_current_active_user)):
    """NLP sentiment analysis for a stock."""
    # Calls sentiment_analyzer.analyze_symbol()
    # Returns: {sentiment_score, confidence, trend, recent_headlines}

@app.post("/forecast")
@rate_limit_authenticated
async def forecast(request: ForecastRequest, current_user: Dict = Depends(get_current_active_user)):
    """Monte Carlo portfolio forecast."""
    # Calls monte_carlo_sim.simulate()
    # Returns: {percentiles, probability_positive, var_95, scenarios}
```

**Verify:** `curl -X POST /portfolio/recommend -d '{"amount":500,"risk_level":"moderate","horizon_months":12}'` with auth header returns allocations.

---

### Task 11: Frontend additions

**Modify:** `src/services/api.js` -- add:
```javascript
export const getMarketSummary = async () => { const r = await api.get('/market-summary'); return r.data; };
export const getPortfolioRecommendation = async (data) => { const r = await api.post('/portfolio/recommend', data); return r.data; };
export const getRiskAnalysis = async (data) => { const r = await api.post('/risk-analysis', data); return r.data; };
export const getSentiment = async (symbol) => { const r = await api.get(`/sentiment/${symbol}`); return r.data; };
export const getForecast = async (data) => { const r = await api.post('/forecast', data); return r.data; };
```

**Modify:** `src/App.js`
- Add dark mode toggle (Tailwind `dark:` classes, store preference in localStorage)
- Add route: `/dashboard` -> DashboardPage

**Create:** `src/pages/DashboardPage.js`
- Calls `getMarketSummary()` on mount
- Displays: index cards (SPY, QQQ, DIA with price/change), macro indicator cards, sector sentiment bars
- Uses Recharts BarChart for sector performance

**Create:** `src/components/AllocationPie.js`
- Recharts PieChart for portfolio allocation weights
- Props: `{ allocations: {symbol: {weight, amount}} }`

**Create:** `src/components/MonteCarloChart.js`
- Recharts AreaChart showing percentile bands (5th, 25th, median, 75th, 95th)
- Fan chart visualization
- Props: `{ forecast: {percentiles, months, initial_amount} }`

**Create:** `src/components/ProbabilityCard.js`
- Displays Monte Carlo probability metrics
- Props: `{ forecast: {probability_positive, median_value, var_95, best_case, worst_case} }`

**Modify:** `src/pages/ChatPage.js`
- When `response.stockData` contains `ml_results.allocation`: render AllocationPie
- When `response.stockData` contains `ml_results.forecast`: render MonteCarloChart + ProbabilityCard

**Modify:** `src/pages/PortfolioPage.js`
- Add "Analyze Risk" button per portfolio -> calls `getRiskAnalysis({portfolio_id})`
- Display risk metrics card (VaR, Sharpe, max drawdown)

**Modify:** `mobile/src/api/client.ts` -- add same 5 new API functions

**Verify:** Visual check -- dark mode toggle works, dashboard loads, recommendation with pie chart renders in chat.

---

### Task 12: Dockerfile and CI/CD

**Modify:** `Dockerfile.backend` -- ensure new directories are copied:
```dockerfile
COPY backend/ ./backend/
# backend/ml/ and backend/data/ are inside backend/, so already covered
```
(The existing `COPY backend/ ./backend/` already copies subdirectories. No change needed if backend/ml/ and backend/data/ are inside backend/.)

**Modify:** `requirements.txt` -- add deps from Section 3

**Modify:** `docker-compose.yml` -- add new env vars:
```yaml
- FINNHUB_API_KEY=${FINNHUB_API_KEY}
- ALPHA_VANTAGE_KEY=${ALPHA_VANTAGE_KEY}
- FRED_API_KEY=${FRED_API_KEY}
```

**Create:** `.github/workflows/ci.yml`
```yaml
name: CI
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: python -m pytest tests/ -v --tb=short
```

**Create:** `tests/__init__.py` (empty)

**Create:** `tests/test_ml.py` -- smoke tests:
```python
def test_feature_pipeline_import():
    from backend.ml.feature_pipeline import FeaturePipeline

def test_xgboost_import():
    from backend.ml.xgboost_model import XGBoostPredictor

def test_monte_carlo_basic():
    from backend.ml.monte_carlo import MonteCarloSimulator
    mc = MonteCarloSimulator()
    result = mc.simulate(['AAPL'], [1.0], 1000, 6, n_simulations=100)
    assert 'median_value' in result
    assert result['probability_positive'] >= 0

def test_portfolio_optimizer():
    from backend.ml.portfolio_optimizer import PortfolioOptimizer
    opt = PortfolioOptimizer()
    result = opt.optimize(['AAPL', 'MSFT', 'BND'], 'max_sharpe')
    assert abs(sum(result['weights'].values()) - 1.0) < 0.01
```

**Create:** `tests/test_api.py` -- endpoint tests:
```python
from fastapi.testclient import TestClient
from backend.main_enhanced import app

client = TestClient(app)

def test_health():
    r = client.get('/health')
    assert r.status_code == 200

def test_status():
    r = client.get('/status')
    assert r.status_code == 200
    assert r.json()['status'] == 'operational'
```

**Verify:** `python -m pytest tests/ -v` passes. `docker-compose up --build` succeeds.

---

## 6. VERIFICATION CHECKLIST

After all tasks are complete, verify:

1. `GET /health` -> 200 `{"status": "healthy"}`
2. `GET /market-summary` -> returns indices, macro, sentiment
3. `POST /portfolio/recommend {"amount":500, "risk_level":"moderate", "horizon_months":12}` -> returns allocations + forecast
4. `GET /sentiment/AAPL` -> returns sentiment score and headlines
5. `POST /forecast {"symbols":["AAPL","MSFT"], "weights":[0.6,0.4], "amount":1000, "months":12}` -> returns percentiles
6. `POST /risk-analysis {"symbols":["AAPL","MSFT","GOOGL"], "weights":[0.4,0.3,0.3]}` -> returns VaR, Sharpe
7. Chat: "Predict AAPL stock" -> response includes XGBoost direction/probability (not just live price)
8. Chat: "I have $500, moderate risk, 1 year" -> response includes allocation percentages + Monte Carlo probabilities
9. Chat: "What's the sentiment for TSLA?" -> response includes FinBERT sentiment score
10. Chat: "Show my portfolio" (logged-in user with holdings) -> response includes risk metrics
11. Web: dark mode toggle works
12. Web: `/dashboard` page loads with market data
13. `docker-compose up --build` succeeds
14. `python -m pytest tests/ -v` passes
15. PostgreSQL has 9 tables (6 existing + 3 new: market_data, sentiment_scores, macro_indicators)

---

## 7. COST ANALYSIS

| Service | Current | After Build | Notes |
|---------|---------|-------------|-------|
| Railway backend | ~$5/mo | ~$12/mo | FinBERT model uses ~400MB RAM |
| Railway PostgreSQL | ~$5/mo | ~$7/mo | 3 new tables |
| RapidAPI | Free tier | Free tier | 500 req/mo |
| Finnhub | - | Free | 60 req/min |
| Alpha Vantage | - | Free | 25 req/day |
| FRED | - | Free | Unlimited |
| HuggingFace | Free | Free | FinBERT runs locally |
| GitHub Actions | - | Free | 2000 min/mo |
| **Total** | **~$10/mo** | **~$19/mo** | Under $50/mo budget |
