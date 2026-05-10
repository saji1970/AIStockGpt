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
    "general_question": [
        r"help", r"what.*can.*you.*do", r"capabilities",
        r"how.*does.*it.*work", r"model.*information", r"ai.*model",
        r"neural.*network", r"lstm", r"how.*accurate", r"model.*performance",
    ],
}

# Common stock symbols
STOCK_SYMBOLS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
    'ADBE', 'CRM', 'ORCL', 'INTC', 'AMD', 'IBM', 'CSCO', 'QCOM',
    'AVGO', 'TXN', 'MU', 'AMAT', 'KLAC', 'LRCX', 'ADI', 'MCHP',
    'PYPL', 'SQ', 'V', 'MA', 'JPM', 'BAC', 'WFC', 'GS', 'MS',
    'DIS', 'NKE', 'SBUX', 'MCD', 'KO', 'PEP', 'WMT', 'TGT', 'HD',
    'LOW', 'COST', 'TMO', 'ABBV', 'JNJ', 'PFE', 'MRK', 'UNH',
    'CVS', 'ANTM', 'CI', 'HUM', 'ELV', 'DHR', 'GE', 'BA', 'CAT',
    'DE', 'MMM', 'HON', 'RTX', 'LMT', 'NOC', 'GD', 'LHX',
]


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

        # Extract stock symbols
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

        return entities

    def _extract_stock_symbols(self, normalized: str, original: str) -> List[str]:
        """Extract stock symbols from the message."""
        found = []

        # Check known symbols in original (case-sensitive)
        words = original.upper().split()
        for word in words:
            clean = re.sub(r'[^\w]', '', word)
            if clean in STOCK_SYMBOLS and clean not in found:
                found.append(clean)

        # Pattern-based extraction from original message
        symbol_patterns = [
            r"([A-Z]{1,5})\s+stock",
            r"stock\s+([A-Z]{1,5})",
            r"analyze\s+([A-Z]{1,5})",
            r"predict\s+([A-Z]{1,5})",
            r"([A-Z]{1,5})\s+analysis",
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
