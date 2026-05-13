"""
Enhanced NLP Processor for AI Stock GPT
=========================================

Uses sentence-transformers (Hugging Face) for ML-based intent classification
with cosine similarity, while keeping regex-based entity extraction.
Falls back to regex intent classification if sentence-transformers is unavailable.
"""

import re
import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

logger = logging.getLogger(__name__)

# Try to import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import cos_sim
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not installed, using regex fallback")


# Intent reference sentences for embedding-based classification
INTENT_REFERENCES = {
    "stock_prediction": [
        "predict the stock price",
        "what will the stock price be",
        "forecast stock movement",
        "stock price prediction",
        "where is the stock going",
        "is the stock going up or down",
        "will the stock go up",
        "buy or sell recommendation",
        "bullish or bearish outlook",
        "investment advice for this stock",
        "stock trend prediction",
        "future price estimate",
    ],
    "technical_analysis": [
        "technical analysis of the stock",
        "show me technical indicators",
        "RSI MACD bollinger bands analysis",
        "moving average crossover",
        "momentum indicators for the stock",
        "chart analysis support resistance",
        "oscillator values for the stock",
        "volume analysis trend",
        "technical signals buy sell",
        "indicator values for trading",
    ],
    "sensitivity_analysis": [
        "sensitivity analysis of stock price",
        "feature importance for prediction",
        "what affects the stock price most",
        "key factors influencing price",
        "important features driving stock",
        "model sensitivity to inputs",
        "variable importance ranking",
        "what drives the stock price",
    ],
    "portfolio_management": [
        "manage my portfolio",
        "add stock to my portfolio",
        "portfolio performance",
        "diversification strategy",
        "asset allocation",
        "portfolio tracking",
        "my investment portfolio",
        "rebalance portfolio",
    ],
    "market_news": [
        "market news today",
        "what is happening in the market",
        "market trends and updates",
        "sector performance",
        "market conditions",
        "latest financial news",
        "stock market overview",
    ],
    "market_advice": [
        "top 10 stocks to invest in",
        "best stocks to buy right now",
        "what are the top stocks",
        "if I have 500 dollars to invest what should I buy",
        "where should I invest my money",
        "best investment options available",
        "recommend stocks for beginners",
        "hedge fund alternatives and ETFs",
        "mutual fund recommendations",
        "best ETFs to buy",
        "which sectors are performing best",
        "market overview and sector trends",
        "index fund recommendations",
        "S&P 500 versus Nasdaq",
        "safe stocks for long term investment",
        "dividend stocks to buy",
        "growth stocks recommendations",
        "which stock shows promising growth and is low cost",
        "cheap stocks with strong growth potential",
        "undervalued growth stocks to buy now",
        "low price stocks with high upside",
        "best growth stocks under 20 dollars",
        "crypto investment advice",
        "how to diversify my investments",
        "best options in current market",
        "best Indian stocks to invest in",
        "invest 50000 rupees in India",
        "top Nifty 50 stocks to buy",
        "best stocks in Indian market",
        "where to invest in Indian market",
        "invest in India",
    ],
    "retirement_planning": [
        "I am retiring in 10 years how should I adjust my portfolio",
        "retirement planning for a 55 year old",
        "how to build a retirement income portfolio",
        "age based portfolio allocation",
        "glide path for retirement investing",
        "I am 60 and need income from my investments",
        "how should older investors reduce risk",
        "retirement countdown investment strategy",
        "sequence of returns risk in retirement",
        "transitioning portfolio from growth to income for retirement",
    ],
    "behavioral_coaching": [
        "I panic during market crashes what should I do",
        "I keep chasing hype stocks and losing money",
        "how do I stop emotional investing",
        "I sold everything during the crash and regret it",
        "I have FOMO about stocks going up without me",
        "I fear market crashes how do I manage anxiety",
        "should I sell everything the market is crashing",
        "I missed out on a stock that went up 500 percent",
        "how to stay disciplined during market volatility",
        "I cannot stop checking my portfolio every hour",
    ],
    "income_strategy": [
        "I want passive income from my investments",
        "best dividend stocks and ETFs for income",
        "build me a dividend portfolio",
        "how to generate monthly income from investments",
        "compare dividend vs growth investing",
        "which ETFs pay the highest dividends",
        "REIT investing for rental income",
        "income producing investments for retirees",
    ],
    "macro_analysis": [
        "how do interest rates affect the stock market",
        "what happens to stocks during inflation",
        "how should I position for a recession",
        "what is stagflation and how to invest during it",
        "impact of Fed rate cuts on my portfolio",
        "which sectors perform best during inflation",
        "how does monetary policy affect investments",
        "what happens to bonds when rates rise",
        "sector rotation based on business cycle",
        "macroeconomic outlook and portfolio positioning",
    ],
    "risk_assessment": [
        "I cannot afford to lose money",
        "build me a capital preservation portfolio",
        "what are safe haven assets",
        "how to protect my portfolio from a crash",
        "recession resistant portfolio construction",
        "what is the safest way to invest",
        "downside protection strategies",
        "my portfolio is too risky how do I reduce risk",
    ],
    "comparative_analysis": [
        "compare Apple vs Microsoft stock",
        "VOO vs QQQ which is better",
        "should I buy growth or value stocks",
        "compare index funds vs individual stocks",
        "ETF comparison for long term investing",
        "which is better dividend stocks or growth stocks",
        "compare active vs passive investing",
        "Nifty 50 vs S&P 500 comparison",
    ],
    "beginner_guidance": [
        "I am a complete beginner how do I start investing",
        "I have 1000 dollars what should I invest in",
        "explain investing in simple terms",
        "what is the simplest way to invest",
        "first time investor what should I know",
        "investing basics for beginners",
        "how to start investing with little money",
        "simple portfolio for someone who knows nothing",
    ],
    "financial_planning": [
        "how to save for my child education",
        "I want financial independence how much do I need",
        "plan my investments for a house down payment",
        "529 plan for college savings",
        "how much do I need to retire at 50",
        "create a goal based investment plan",
        "dollar cost averaging vs lump sum investing",
        "what is the 4 percent rule for retirement withdrawal",
        "how to plan investments for multiple financial goals",
        "fire movement financial independence retire early",
    ],
    "general_question": [
        "hello how are you",
        "what can you do",
        "help me understand",
        "how does this work",
        "capabilities of the system",
        "tell me about the AI model",
        "how accurate are predictions",
        "what features do you have",
    ],
}

# Regex intent patterns (fallback)
REGEX_INTENT_PATTERNS = {
    "stock_prediction": [
        r"predict.*stock", r"forecast.*stock", r"what.*price.*(?:will|going to)",
        r"stock.*prediction", r"price.*prediction", r"future.*price",
        r"where.*stock.*going", r"stock.*trend", r"price.*trend",
        r"bullish|bearish", r"buy|sell.*recommendation", r"investment.*advice",
        r"stock.*outlook", r"price.*outlook",
    ],
    "technical_analysis": [
        r"technical.*analysis", r"technical.*indicators",
        r"rsi|macd|bollinger|moving.*average", r"momentum.*indicators",
        r"volatility.*indicators", r"volume.*analysis", r"chart.*analysis",
        r"support.*resistance", r"trend.*analysis", r"oscillator",
        r"technical.*signals", r"indicator.*values",
    ],
    "sensitivity_analysis": [
        r"sensitivity.*analysis", r"feature.*importance", r"what.*affects.*price",
        r"key.*factors", r"important.*features", r"drivers.*price",
        r"factors.*influence", r"model.*sensitivity", r"feature.*ranking",
        r"variable.*importance",
    ],
    "portfolio_management": [
        r"portfolio", r"my.*stocks", r"holdings", r"diversif",
        r"asset.*allocation", r"rebalance",
    ],
    "market_news": [
        r"market.*news", r"market.*trend", r"sector.*performance",
        r"market.*condition", r"financial.*news", r"market.*overview",
    ],
    "market_advice": [
        r"top\s*\d+\s*stocks?", r"best\s*stocks?", r"best.*(?:invest|option|pick|buy)",
        r"invest\s*(?:\$|₹|rs\.?)?\s*\d+", r"where.*(?:invest|put.*money)",
        r"hedge\s*fund", r"mutual\s*fund", r"etf.*(?:recommend|best|top|buy)",
        r"(?:recommend|suggest).*(?:stock|invest|fund|etf|portfolio)",
        r"good.*(?:stock|invest|fund|etf).*(?:buy|now|today|current)",
        r"market.*(?:recommend|advice|tip|outlook|trend)",
        r"what.*(?:should|would).*(?:invest|buy)",
        r"(?:safe|risky|growth|value|dividend).*(?:stock|invest|fund)",
        r"(?:which|what)\s+stock.*(?:growth|cheap|undervalued|promising|low\s*cost|affordable)",
        r"(?:promising|strong)\s+growth.*(?:low|cheap|cost|affordable|undervalued)",
        r"(?:cheap|low[-\s]?cost|affordable|undervalued).*(?:growth|growing|potential|upside)",
        r"(?:growth|growing).*(?:cheap|low\s*price|affordable|undervalued)",
        r"(?:stock|invest|fund).*(?:beginner|starter|start)",
        r"diversif", r"sector.*(?:perform|best|top|hot)", r"asset.*allocation",
        r"(?:current|today).*market", r"index.*fund",
        r"s.?p\s*500", r"nasdaq|dow\s*jones", r"crypto.*(?:invest|buy|best)",
        r"(?:india|indian|nifty|sensex|bse|nse).*(?:invest|stock|buy|best)",
        r"(?:invest|stock|buy|best).*(?:india|indian|nifty|sensex|bse|nse)",
    ],
    "retirement_planning": [
        r"retir", r"retirement", r"retire.*in.*\d+.*years?",
        r"age.*based.*(?:portfolio|invest|allocation)",
        r"(?:i.?m|i am)\s*\d{2,}.*(?:year|old).*(?:retir|invest|portfolio)",
        r"glide.*path", r"pension", r"(?:retirement|retire).*income",
        r"(?:older|senior|elderly).*invest", r"life.*stage",
    ],
    "behavioral_coaching": [
        r"panic.*(?:sell|crash|market)", r"(?:fear|scared|afraid|anxious|anxiety).*(?:market|invest|crash|losing)",
        r"emotional.*invest", r"(?:fomo|fear.*missing.*out)",
        r"(?:chas|chase).*(?:hype|hot|meme).*stock", r"regret.*(?:sell|buy|miss)",
        r"(?:stop|quit).*(?:worry|panic|fear).*(?:market|invest|portfolio)",
        r"(?:sold|sell).*everything.*(?:crash|fear|panic)",
        r"(?:can.?t|cannot).*stop.*check.*portfolio",
    ],
    "income_strategy": [
        r"(?:passive|monthly|regular).*income.*(?:invest|portfolio|stock|dividend)",
        r"dividend.*(?:stock|etf|invest|portfolio|income|strategy)",
        r"(?:income|yield).*(?:invest|portfolio|generat)",
        r"reit.*(?:invest|income)", r"(?:high|best).*(?:yield|dividend)",
        r"(?:invest|portfolio).*(?:income|dividend)",
    ],
    "macro_analysis": [
        r"interest.*rate.*(?:affect|impact|stock|bond|market)",
        r"(?:inflation|deflation).*(?:invest|stock|portfolio|sector|affect|impact)",
        r"recession.*(?:invest|portfolio|proof|resist|protect)",
        r"stagflation", r"fed(?:eral)?.*(?:rate|policy|cut|hike)",
        r"(?:monetary|fiscal).*policy", r"macro.*(?:econom|analysis|outlook)",
        r"(?:sector|business).*(?:cycle|rotation)",
        r"(?:bond|stock|market).*(?:when|during|if).*rate",
    ],
    "risk_assessment": [
        r"(?:can.?t|cannot).*(?:afford|risk).*(?:lose|loss)",
        r"capital.*preserv", r"safe.*haven",
        r"(?:protect|shield|hedge).*(?:portfolio|investment).*(?:crash|loss|downturn)",
        r"(?:recession|crash|downturn).*(?:resist|proof|protect)",
        r"(?:safest|safe|low.*risk|minimal.*risk).*(?:invest|way|portfolio)",
        r"(?:reduce|lower|decrease).*(?:risk|volatility|exposure)",
        r"(?:downside|loss).*protect",
    ],
    "comparative_analysis": [
        r"(?:compare|vs|versus|or).*(?:stock|etf|fund|invest)",
        r"(?:which|what).*(?:better|prefer|choose).*(?:stock|etf|fund)",
        r"(?:growth|value|dividend|index|active|passive).*(?:vs|versus|or|compared)",
        r"(?:apple|aapl).*(?:vs|versus|or).*(?:microsoft|msft)",
        r"(?:voo|spy|qqq|vti).*(?:vs|versus|or).*(?:voo|spy|qqq|vti)",
        r"(?:nifty|sensex).*(?:vs|versus|or).*(?:s.?p|nasdaq)",
    ],
    "beginner_guidance": [
        r"(?:beginner|newbie|new.*to|first.*time|starting|start).*(?:invest|stock|market|portfolio)",
        r"(?:how|where).*(?:start|begin).*invest",
        r"(?:invest|stock|market).*(?:basics?|101|simple|easy|beginner)",
        r"(?:know|learn).*(?:nothing|little).*(?:invest|stock|market)",
        r"(?:explain|teach).*(?:invest|stock|market).*(?:simple|basic|easy)",
        r"(?:i have|invest).*(?:\$|₹|rs)?\s*\d{3,4}\b.*(?:what|how|where)",
    ],
    "financial_planning": [
        r"(?:financial|money).*(?:independence|freedom|fire)",
        r"(?:save|saving).*(?:child|kid|college|education|school|university|house|home|wedding)",
        r"(?:529|education).*(?:plan|save|fund)",
        r"(?:how.*much|when).*(?:need|enough).*(?:retire|fi\b|financial.*independ)",
        r"(?:goal|target).*(?:based|driven).*(?:invest|plan|portfolio)",
        r"(?:dollar.*cost|dca|lump.*sum).*(?:invest|average|vs)",
        r"(?:4|four).*percent.*rule", r"(?:withdrawal|drawdown).*(?:rate|strategy)",
        r"(?:plan|planning).*(?:invest|financial|money).*(?:goal|target|future)",
    ],
    "general_question": [
        r"help", r"what.*can.*you.*do", r"capabilities",
        r"how.*does.*it.*work", r"model.*information", r"ai.*model",
        r"neural.*network", r"lstm", r"how.*accurate", r"model.*performance",
    ],
}

# Common stock symbols
STOCK_SYMBOLS = [
    # US Stocks
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
    'ADBE', 'CRM', 'ORCL', 'INTC', 'AMD', 'IBM', 'CSCO', 'QCOM',
    'AVGO', 'TXN', 'MU', 'MRVL', 'AMAT', 'KLAC', 'LRCX', 'ADI', 'MCHP',
    'PYPL', 'SQ', 'SOFI', 'V', 'MA', 'JPM', 'BAC', 'WFC', 'GS', 'MS',
    'DIS', 'NKE', 'SBUX', 'MCD', 'KO', 'PEP', 'WMT', 'TGT', 'HD',
    'LOW', 'COST', 'TMO', 'ABBV', 'JNJ', 'PFE', 'MRK', 'UNH',
    'CVS', 'ANTM', 'CI', 'HUM', 'ELV', 'DHR', 'GE', 'BA', 'CAT',
    'DE', 'MMM', 'HON', 'RTX', 'LMT', 'NOC', 'GD', 'LHX',
    'SPY', 'QQQ', 'DIA', 'IWM', 'VOO', 'VTI', 'XOM', 'CVX', 'PG',
    'XLK', 'XLV', 'XLF', 'XLE', 'XLY', 'VGT', 'BND', 'AGG', 'TLT',
    'VIG', 'SCHD', 'VYM', 'VNQ', 'GLD', 'VXUS',
    # India - BSE (Nifty 50 / Sensex constituents)
    'INFY.BSE', 'TCS.BSE', 'WIPRO.BSE', 'HCLTECH.BSE', 'TECHM.BSE',
    'HDFCBANK.BSE', 'ICICIBANK.BSE', 'SBIN.BSE', 'KOTAKBANK.BSE',
    'AXISBANK.BSE', 'BAJFINANCE.BSE', 'BAJAJFINSV.BSE', 'INDUSINDBK.BSE',
    'HINDUNILVR.BSE', 'ITC.BSE', 'NESTLEIND.BSE', 'BRITANNIA.BSE',
    'TATACONSUM.BSE', 'ASIANPAINT.BSE', 'TITAN.BSE',
    'RELIANCE.BSE', 'LT.BSE', 'MARUTI.BSE', 'TATAMOTORS.BSE',
    'EICHERMOT.BSE', 'ULTRACEMCO.BSE', 'GRASIM.BSE',
    'SUNPHARMA.BSE', 'DRREDDY.BSE', 'CIPLA.BSE', 'DIVISLAB.BSE',
    'APOLLOHOSP.BSE',
    'ONGC.BSE', 'NTPC.BSE', 'POWERGRID.BSE', 'BPCL.BSE', 'COALINDIA.BSE',
    'TATASTEEL.BSE', 'HINDALCO.BSE', 'JSWSTEEL.BSE',
    'BHARTIARTL.BSE', 'ADANIENT.BSE', 'ADANIPORTS.BSE',
]

# Tickers that match common English words when the user types lowercase ("low cost" -> LOW)
AMBIGUOUS_TICKERS = frozenset(
    s for s in {
        "LOW", "COST", "CAT", "DE", "DIS", "GS", "MS", "KO", "MA", "BA", "V", "CRM",
        "WMT", "TGT", "HD", "PG", "META", "ALL", "KEY", "NET", "NOW", "FAST", "LITE",
    }
    if s in STOCK_SYMBOLS
)

# Lowercase ticker spellings users often type intentionally
ALLOW_LOWER_TICKERS = frozenset({
    "aapl", "msft", "googl", "goog", "amzn", "tsla", "meta", "fb", "nvda", "nflx",
    "spy", "qqq", "iwm", "dia", "voo", "vti", "amd", "intc", "mu", "avgo",
    "pypl", "coin", "hood", "sofi", "pltr", "rivn", "lcid", "f", "gm", "mrvl",
    "qcom", "shop", "sq", "uber", "abnb", "nke", "jpm", "bac", "xom", "pfe",
})

# Lowercase ticker spellings users often type intentionally
ALLOW_LOWER_TICKERS = frozenset({
    'aapl', 'msft', 'googl', 'goog', 'amzn', 'tsla', 'meta', 'fb', 'nvda', 'nflx',
    'spy', 'qqq', 'iwm', 'dia', 'voo', 'vti', 'amd', 'intc', 'mu', 'avgo',
    'pypl', 'coin', 'hood', 'sofi', 'pltr', 'rivn', 'lcid', 'f', 'gm',
})

# Company name to symbol mapping (case-insensitive lookup)
COMPANY_NAME_MAP = {
    # US Companies
    'apple': 'AAPL', 'microsoft': 'MSFT', 'google': 'GOOGL', 'alphabet': 'GOOGL',
    'amazon': 'AMZN', 'tesla': 'TSLA', 'meta': 'META', 'facebook': 'META',
    'nvidia': 'NVDA', 'netflix': 'NFLX', 'adobe': 'ADBE', 'salesforce': 'CRM',
    'oracle': 'ORCL', 'intel': 'INTC', 'amd': 'AMD', 'ibm': 'IBM',
    'cisco': 'CSCO', 'qualcomm': 'QCOM', 'broadcom': 'AVGO',
    'paypal': 'PYPL', 'square': 'SQ', 'block': 'SQ',
    'sofi': 'SOFI', 'sofi technologies': 'SOFI',
    'micron': 'MU', 'micron technology': 'MU',
    'marvell': 'MRVL', 'marvell technology': 'MRVL',
    'visa': 'V', 'mastercard': 'MA', 'jpmorgan': 'JPM', 'jp morgan': 'JPM',
    'bank of america': 'BAC', 'wells fargo': 'WFC', 'goldman sachs': 'GS',
    'morgan stanley': 'MS', 'disney': 'DIS', 'nike': 'NKE',
    'starbucks': 'SBUX', 'mcdonalds': 'MCD', "mcdonald's": 'MCD',
    'coca cola': 'KO', 'coca-cola': 'KO', 'pepsi': 'PEP', 'pepsico': 'PEP',
    'walmart': 'WMT', 'target': 'TGT', 'home depot': 'HD', 'costco': 'COST',
    'johnson & johnson': 'JNJ', 'johnson and johnson': 'JNJ',
    'pfizer': 'PFE', 'merck': 'MRK', 'unitedhealth': 'UNH',
    'boeing': 'BA', 'caterpillar': 'CAT', 'general electric': 'GE',
    'exxon': 'XOM', 'exxon mobil': 'XOM', 'exxonmobil': 'XOM', 'chevron': 'CVX',
    'procter & gamble': 'PG', 'procter and gamble': 'PG', 'p&g': 'PG',
    # India - BSE Companies
    'infosys': 'INFY.BSE', 'infy': 'INFY.BSE',
    'tcs': 'TCS.BSE', 'tata consultancy': 'TCS.BSE', 'tata consultancy services': 'TCS.BSE',
    'wipro': 'WIPRO.BSE',
    'hcl tech': 'HCLTECH.BSE', 'hcl technologies': 'HCLTECH.BSE', 'hcltech': 'HCLTECH.BSE',
    'tech mahindra': 'TECHM.BSE',
    'hdfc bank': 'HDFCBANK.BSE', 'hdfc': 'HDFCBANK.BSE', 'hdfcbank': 'HDFCBANK.BSE',
    'icici bank': 'ICICIBANK.BSE', 'icici': 'ICICIBANK.BSE',
    'sbi': 'SBIN.BSE', 'state bank': 'SBIN.BSE', 'state bank of india': 'SBIN.BSE',
    'kotak bank': 'KOTAKBANK.BSE', 'kotak mahindra': 'KOTAKBANK.BSE', 'kotak': 'KOTAKBANK.BSE',
    'axis bank': 'AXISBANK.BSE', 'axis': 'AXISBANK.BSE',
    'bajaj finance': 'BAJFINANCE.BSE', 'bajfinance': 'BAJFINANCE.BSE',
    'bajaj finserv': 'BAJAJFINSV.BSE',
    'indusind bank': 'INDUSINDBK.BSE', 'indusind': 'INDUSINDBK.BSE',
    'hindustan unilever': 'HINDUNILVR.BSE', 'hul': 'HINDUNILVR.BSE',
    'itc': 'ITC.BSE',
    'nestle india': 'NESTLEIND.BSE', 'nestle': 'NESTLEIND.BSE',
    'britannia': 'BRITANNIA.BSE',
    'tata consumer': 'TATACONSUM.BSE',
    'asian paints': 'ASIANPAINT.BSE', 'asian paint': 'ASIANPAINT.BSE',
    'titan': 'TITAN.BSE',
    'reliance': 'RELIANCE.BSE', 'reliance industries': 'RELIANCE.BSE', 'ril': 'RELIANCE.BSE',
    'larsen & toubro': 'LT.BSE', 'l&t': 'LT.BSE', 'larsen and toubro': 'LT.BSE',
    'maruti': 'MARUTI.BSE', 'maruti suzuki': 'MARUTI.BSE',
    'tata motors': 'TATAMOTORS.BSE',
    'eicher motors': 'EICHERMOT.BSE', 'royal enfield': 'EICHERMOT.BSE',
    'ultratech cement': 'ULTRACEMCO.BSE', 'ultratech': 'ULTRACEMCO.BSE',
    'grasim': 'GRASIM.BSE',
    'sun pharma': 'SUNPHARMA.BSE', 'sun pharmaceutical': 'SUNPHARMA.BSE',
    'dr reddy': 'DRREDDY.BSE', "dr reddy's": 'DRREDDY.BSE', 'dr reddys': 'DRREDDY.BSE',
    'cipla': 'CIPLA.BSE',
    "divi's lab": 'DIVISLAB.BSE', 'divis lab': 'DIVISLAB.BSE',
    'apollo hospitals': 'APOLLOHOSP.BSE', 'apollo': 'APOLLOHOSP.BSE',
    'ongc': 'ONGC.BSE', 'ntpc': 'NTPC.BSE',
    'power grid': 'POWERGRID.BSE', 'powergrid': 'POWERGRID.BSE',
    'bpcl': 'BPCL.BSE', 'coal india': 'COALINDIA.BSE',
    'tata steel': 'TATASTEEL.BSE',
    'hindalco': 'HINDALCO.BSE',
    'jsw steel': 'JSWSTEEL.BSE',
    'bharti airtel': 'BHARTIARTL.BSE', 'airtel': 'BHARTIARTL.BSE',
    'adani enterprises': 'ADANIENT.BSE', 'adani': 'ADANIENT.BSE',
    'adani ports': 'ADANIPORTS.BSE',
}


def _is_indian_symbol(symbol: str) -> bool:
    """Check if a symbol belongs to an Indian exchange."""
    return symbol.upper().endswith('.BSE') or symbol.upper().endswith('.NSE')


class EnhancedNLPProcessor:
    """
    Enhanced NLP Processor using sentence-transformers for intent classification
    and regex for entity extraction.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = None
        self.intent_embeddings = {}
        self.use_ml = False

        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                logger.info(f"Loading sentence-transformer model: {model_name}")
                self.model = SentenceTransformer(model_name)
                self._precompute_intent_embeddings()
                self.use_ml = True
                logger.info("Enhanced NLP Processor initialized with ML-based classification")
            except Exception as e:
                logger.error(f"Failed to load sentence-transformer model: {e}")
                logger.info("Falling back to regex-based classification")
        else:
            logger.info("Enhanced NLP Processor initialized with regex-based classification")

    def _precompute_intent_embeddings(self):
        """Precompute embeddings for all intent reference sentences."""
        for intent, sentences in INTENT_REFERENCES.items():
            embeddings = self.model.encode(sentences, convert_to_tensor=False)
            # Store as numpy array, average the embeddings for this intent
            self.intent_embeddings[intent] = np.mean(embeddings, axis=0)
        logger.info(f"Precomputed embeddings for {len(self.intent_embeddings)} intents")

    def process_message(self, message: str) -> Tuple[str, Dict[str, Any], float]:
        """
        Process a user message and extract intent, entities, and confidence.

        Args:
            message: User message

        Returns:
            Tuple of (intent, entities, confidence)
        """
        try:
            normalized = self._normalize_message(message)

            if self.use_ml:
                intent, confidence = self._classify_intent_ml(message)
            else:
                intent, confidence = self._classify_intent_regex(normalized)

            entities = self._extract_entities(normalized, message)
            confidence = self._adjust_confidence(confidence, entities)

            logger.info(
                f"Processed message: intent={intent}, entities={entities}, "
                f"confidence={confidence:.3f}, method={'ml' if self.use_ml else 'regex'}"
            )

            return intent, entities, confidence

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return "general_question", {}, 0.5

    def _normalize_message(self, message: str) -> str:
        """Normalize message for processing."""
        normalized = message.lower()
        normalized = re.sub(r'\s+', ' ', normalized)
        normalized = re.sub(r'[^\w\s]', ' ', normalized)
        return normalized.strip()

    def _classify_intent_ml(self, message: str) -> Tuple[str, float]:
        """Classify intent using sentence-transformer embeddings + cosine similarity."""
        try:
            message_embedding = self.model.encode([message], convert_to_tensor=False)[0]

            best_intent = "general_question"
            best_score = -1.0

            for intent, intent_embedding in self.intent_embeddings.items():
                # Cosine similarity
                similarity = float(np.dot(message_embedding, intent_embedding) / (
                    np.linalg.norm(message_embedding) * np.linalg.norm(intent_embedding)
                ))

                if similarity > best_score:
                    best_score = similarity
                    best_intent = intent

            # Convert similarity to confidence (similarity ranges ~0.0-1.0)
            confidence = max(0.3, min(0.95, best_score))

            return best_intent, confidence

        except Exception as e:
            logger.error(f"ML classification failed: {e}")
            return self._classify_intent_regex(self._normalize_message(message))

    def _classify_intent_regex(self, message: str) -> Tuple[str, float]:
        """Classify intent using regex patterns (fallback)."""
        intent_scores = {}

        for intent, patterns in REGEX_INTENT_PATTERNS.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, message, re.IGNORECASE):
                    score += 1
            if score > 0:
                intent_scores[intent] = score

        if not intent_scores:
            return "general_question", 0.3

        best_intent = max(intent_scores, key=intent_scores.get)
        max_score = intent_scores[best_intent]
        confidence = min(0.9, 0.3 + (max_score * 0.2))

        return best_intent, confidence

    def _extract_entities(self, normalized: str, original: str) -> Dict[str, Any]:
        """Extract entities from the message using regex."""
        entities = {}

        # Extract stock symbols (including company name resolution)
        symbols = self._extract_stock_symbols(normalized, original)
        if symbols:
            entities["symbol"] = symbols[0]

        # Extract time periods
        time_periods = self._extract_time_periods(normalized)
        if time_periods:
            entities["time_period"] = time_periods[0]

        # Extract analysis types
        analysis_types = self._extract_analysis_types(normalized)
        if analysis_types:
            entities["analysis_type"] = analysis_types[0]

        # Extract market context (india vs us)
        market = self._extract_market(normalized)
        if market:
            entities["market"] = market
        elif entities.get("symbol") and _is_indian_symbol(entities["symbol"]):
            entities["market"] = "india"

        # Extract amount (supports both USD and INR)
        amount = self._extract_amount(normalized)
        if amount is not None:
            entities["amount"] = amount["value"]
            if amount.get("currency"):
                entities["currency"] = amount["currency"]
                if amount["currency"] == "INR" and not entities.get("market"):
                    entities["market"] = "india"

        # Extract risk level
        risk_level = self._extract_risk_level(normalized)
        if risk_level is not None:
            entities["risk_level"] = risk_level

        # Extract investment horizon
        horizon = self._extract_horizon(normalized)
        if horizon is not None:
            entities["horizon_months"] = horizon

        return entities

    def _extract_stock_symbols(self, normalized: str, original: str) -> List[str]:
        """Extract stock symbols from the message, including company name resolution."""
        found = []

        # 1. Check company name mapping first (e.g., "Infosys" -> "INFY.BSE")
        msg_lower = original.lower()
        # Sort by length descending so multi-word names match first
        for name in sorted(COMPANY_NAME_MAP.keys(), key=len, reverse=True):
            if name in msg_lower:
                sym = COMPANY_NAME_MAP[name]
                if sym not in found:
                    found.append(sym)
                break  # Use first (longest) match

        # 2. Known symbols as words — require $, ALL CAPS, allowlist, or long non-ambiguous tickers
        for m in re.finditer(r"\$?\b([A-Za-z][A-Za-z0-9.]{0,14})\b", original):
            raw = m.group(1)
            if raw.startswith("$"):
                raw = raw[1:]
            clean = re.sub(r"[^\w.]", "", raw).strip(".")
            if not clean:
                continue

            upper_clean = clean.upper()
            sym: Optional[str] = None
            if upper_clean in STOCK_SYMBOLS:
                sym = upper_clean
            else:
                base = upper_clean.split(".", 1)[0]
                for s in STOCK_SYMBOLS:
                    if s.split(".")[0] == base:
                        sym = s
                        break
            if not sym or sym in found:
                continue

            rlow = raw.lower()
            all_caps = raw == raw.upper()
            base_len = len(sym.split(".")[0])

            if rlow in ALLOW_LOWER_TICKERS or all_caps:
                found.append(sym)
                continue
            if sym in AMBIGUOUS_TICKERS:
                continue
            if base_len <= 4:
                continue
            found.append(sym)

        # 3. Pattern-based extraction from original message (US symbols)
        symbol_patterns = [
            r"([A-Z]{1,10}(?:\.[A-Z]{1,4})?)\s+stock",
            r"stock\s+([A-Z]{1,10}(?:\.[A-Z]{1,4})?)",
            r"analyze\s+([A-Z]{1,10}(?:\.[A-Z]{1,4})?)",
            r"predict\s+([A-Z]{1,10}(?:\.[A-Z]{1,4})?)",
            r"([A-Z]{1,10}(?:\.[A-Z]{1,4})?)\s+analysis",
        ]

        for pattern in symbol_patterns:
            for match in re.findall(pattern, original.upper()):
                if match not in found:
                    found.append(match)

        return found

    def _extract_time_periods(self, message: str) -> List[str]:
        """Extract time periods from the message."""
        patterns = [
            r"(\d+)\s*days?",
            r"(\d+)\s*weeks?",
            r"(\d+)\s*months?",
            r"(\d+)\s*years?",
            r"next\s*(\d+)\s*days?",
            r"in\s*(\d+)\s*days?",
        ]
        found = []
        for pattern in patterns:
            found.extend(re.findall(pattern, message))
        return found

    def _extract_analysis_types(self, message: str) -> List[str]:
        """Extract analysis types from the message."""
        keywords = ["technical", "fundamental", "sensitivity", "prediction", "forecast", "trend"]
        return [k for k in keywords if k in message]

    def _extract_amount(self, message: str) -> Optional[Dict[str, Any]]:
        """Extract monetary amounts with currency detection.

        Supports: '$500', '500 dollars', '500 usd', '₹500', '500 rupees',
                  '500 inr', 'invest 500', 'invest Rs 500', 'invest Rs.500'
        Returns: {'value': float, 'currency': 'USD'|'INR'|None}
        """
        # INR-specific patterns (check first to avoid fallback to generic)
        inr_patterns = [
            (r'(?:₹|rs\.?|inr)\s*([\d,]+(?:\.\d{1,2})?)', 'INR'),
            (r'([\d,]+(?:\.\d{1,2})?)\s*(?:rupees?|inr|rs)', 'INR'),
        ]
        for pattern, currency in inr_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                try:
                    return {'value': float(match.group(1).replace(',', '')), 'currency': currency}
                except ValueError:
                    continue

        # USD-specific patterns
        usd_patterns = [
            (r'\$\s*([\d,]+(?:\.\d{1,2})?)', 'USD'),
            (r'([\d,]+(?:\.\d{1,2})?)\s*(?:dollars?|usd|bucks)', 'USD'),
        ]
        for pattern, currency in usd_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                try:
                    return {'value': float(match.group(1).replace(',', '')), 'currency': currency}
                except ValueError:
                    continue

        # Generic amount patterns (no currency detected)
        generic_patterns = [
            r'invest\s*(?:\$|₹|rs\.?)?\s*([\d,]+(?:\.\d{1,2})?)',
            r'have\s*(?:\$|₹|rs\.?)?\s*([\d,]+(?:\.\d{1,2})?)',
            r'put\s*(?:\$|₹|rs\.?)?\s*([\d,]+(?:\.\d{1,2})?)',
            r'([\d,]+(?:\.\d{1,2})?)\s*(?:to invest|to put)',
        ]
        for pattern in generic_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                try:
                    return {'value': float(match.group(1).replace(',', '')), 'currency': None}
                except ValueError:
                    continue
        return None

    def _extract_market(self, message: str) -> Optional[str]:
        """Detect target market (india or us) from the message."""
        india_keywords = [
            'india', 'indian', 'nifty', 'sensex', 'bse', 'nse',
            'rupee', 'rupees', 'inr', '₹', ' rs ', ' rs.',
            'mumbai', 'dalal street',
        ]
        us_keywords = [
            ' us ', ' usa ', 'united states', 'american', 'wall street',
            'nasdaq', 'nyse', 's&p', 's p 500', 'dow jones',
        ]
        for kw in india_keywords:
            if kw in message:
                return 'india'
        for kw in us_keywords:
            if kw in message:
                return 'us'
        return None

    def _extract_risk_level(self, message: str) -> Optional[str]:
        """Extract risk level from message."""
        risk_map = {
            'conservative': ['conservative', 'safe', 'low risk', 'low-risk', 'minimal risk', 'cautious'],
            'moderate': ['moderate', 'balanced', 'medium risk', 'medium-risk', 'average risk'],
            'aggressive': ['aggressive', 'high risk', 'high-risk', 'growth', 'risky', 'maximum growth'],
        }
        for level, keywords in risk_map.items():
            for kw in keywords:
                if kw in message:
                    return level
        return None

    def _extract_horizon(self, message: str) -> Optional[int]:
        """Extract investment horizon in months."""
        patterns = [
            (r'(\d+)\s*years?', lambda m: int(m) * 12),
            (r'(\d+)\s*months?', lambda m: int(m)),
            (r'(\d+)\s*weeks?', lambda m: max(1, int(m) // 4)),
        ]
        for pattern, converter in patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                return converter(match.group(1))

        # Named horizons
        horizon_map = {
            'short term': 3, 'short-term': 3,
            'medium term': 12, 'medium-term': 12, 'mid term': 12, 'mid-term': 12,
            'long term': 36, 'long-term': 36,
        }
        for phrase, months in horizon_map.items():
            if phrase in message:
                return months
        return None

    def _adjust_confidence(self, confidence: float, entities: Dict[str, Any]) -> float:
        """Adjust confidence based on entity extraction results."""
        if entities.get("symbol"):
            confidence += 0.1
        if entities.get("analysis_type"):
            confidence += 0.05
        if entities.get("time_period"):
            confidence += 0.05
        return min(0.95, confidence)

    def validate_stock_symbol(self, symbol: str) -> bool:
        """Validate if a stock symbol is supported."""
        return symbol.upper() in STOCK_SYMBOLS

    def get_supported_symbols(self) -> List[str]:
        """Get list of supported stock symbols."""
        return STOCK_SYMBOLS.copy()

    def get_suggested_queries(self, intent: Optional[str] = None) -> List[str]:
        """Get suggested queries based on intent."""
        suggestions = {
            "stock_prediction": [
                "What's the prediction for AAPL stock?",
                "Forecast the price of TSLA",
                "Will MSFT stock go up?",
            ],
            "technical_analysis": [
                "Analyze technical indicators for AAPL",
                "Show me RSI and MACD for TSLA",
                "Technical analysis of MSFT",
            ],
            "sensitivity_analysis": [
                "What affects AAPL stock price most?",
                "Show sensitivity analysis for TSLA",
                "Key factors influencing MSFT",
            ],
            "general_question": [
                "What can you do?",
                "How does the AI model work?",
                "What's your accuracy?",
            ],
        }

        if intent and intent in suggestions:
            return suggestions[intent]

        all_suggestions = []
        for s in suggestions.values():
            all_suggestions.extend(s[:2])
        return all_suggestions
