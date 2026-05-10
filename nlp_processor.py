#!/usr/bin/env python3
"""
NLP Processor for AI Stock GPT
==============================

This module provides natural language processing capabilities for understanding
user queries and extracting intents and entities for stock analysis.
"""

import re
import json
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)

class NLPProcessor:
    """
    Natural Language Processor for stock analysis queries.
    """
    
    def __init__(self):
        """Initialize the NLP processor."""
        self.stock_symbols = self._load_stock_symbols()
        self.intent_patterns = self._load_intent_patterns()
        self.entity_patterns = self._load_entity_patterns()
        
        logger.info("NLP Processor initialized successfully")
    
    def _load_stock_symbols(self) -> List[str]:
        """Load common stock symbols."""
        return [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
            'ADBE', 'CRM', 'ORCL', 'INTC', 'AMD', 'IBM', 'CSCO', 'QCOM',
            'AVGO', 'TXN', 'MU', 'AMAT', 'KLAC', 'LRCX', 'ADI', 'MCHP',
            'PYPL', 'SQ', 'V', 'MA', 'JPM', 'BAC', 'WFC', 'GS', 'MS',
            'DIS', 'NKE', 'SBUX', 'MCD', 'KO', 'PEP', 'WMT', 'TGT', 'HD',
            'LOW', 'COST', 'TMO', 'ABBV', 'JNJ', 'PFE', 'MRK', 'UNH',
            'CVS', 'ANTM', 'CI', 'HUM', 'ELV', 'DHR', 'GE', 'BA', 'CAT',
            'DE', 'MMM', 'HON', 'RTX', 'LMT', 'NOC', 'GD', 'LHX'
        ]
    
    def _load_intent_patterns(self) -> Dict[str, List[str]]:
        """Load intent recognition patterns."""
        return {
            "stock_prediction": [
                r"predict.*stock",
                r"forecast.*stock",
                r"what.*price.*(?:will|going to)",
                r"stock.*prediction",
                r"price.*prediction",
                r"future.*price",
                r"where.*stock.*going",
                r"stock.*trend",
                r"price.*trend",
                r"bullish|bearish",
                r"buy|sell.*recommendation",
                r"investment.*advice",
                r"stock.*outlook",
                r"price.*outlook"
            ],
            "technical_analysis": [
                r"technical.*analysis",
                r"technical.*indicators",
                r"rsi|macd|bollinger|moving.*average",
                r"momentum.*indicators",
                r"volatility.*indicators",
                r"volume.*analysis",
                r"chart.*analysis",
                r"support.*resistance",
                r"trend.*analysis",
                r"oscillator",
                r"technical.*signals",
                r"indicator.*values"
            ],
            "sensitivity_analysis": [
                r"sensitivity.*analysis",
                r"feature.*importance",
                r"what.*affects.*price",
                r"key.*factors",
                r"important.*features",
                r"drivers.*price",
                r"factors.*influence",
                r"model.*sensitivity",
                r"feature.*ranking",
                r"variable.*importance"
            ],
            "general_question": [
                r"help",
                r"what.*can.*you.*do",
                r"capabilities",
                r"how.*does.*it.*work",
                r"model.*information",
                r"ai.*model",
                r"neural.*network",
                r"lstm",
                r"how.*accurate",
                r"model.*performance"
            ]
        }
    
    def _load_entity_patterns(self) -> Dict[str, List[str]]:
        """Load entity extraction patterns."""
        return {
            "symbol": [
                r"\b([A-Z]{1,5})\b",  # Stock symbols like AAPL, TSLA
                r"stock.*([A-Z]{1,5})",
                r"([A-Z]{1,5}).*stock",
                r"analyze.*([A-Z]{1,5})",
                r"predict.*([A-Z]{1,5})"
            ],
            "time_period": [
                r"(\d+)\s*days?",
                r"(\d+)\s*weeks?",
                r"(\d+)\s*months?",
                r"(\d+)\s*years?",
                r"next\s*(\d+)\s*days?",
                r"in\s*(\d+)\s*days?"
            ],
            "analysis_type": [
                r"technical",
                r"fundamental",
                r"sensitivity",
                r"prediction",
                r"forecast",
                r"trend"
            ]
        }
    
    def process_message(self, message: str) -> Tuple[str, Dict[str, Any], float]:
        """
        Process a user message and extract intent, entities, and confidence.
        
        Args:
            message (str): User message
            
        Returns:
            Tuple[str, Dict[str, Any], float]: (intent, entities, confidence)
        """
        try:
            # Normalize message
            normalized_message = self._normalize_message(message)
            
            # Extract intent
            intent, intent_confidence = self._extract_intent(normalized_message)
            
            # Extract entities
            entities = self._extract_entities(normalized_message)
            
            # Calculate overall confidence
            confidence = self._calculate_confidence(intent_confidence, entities)
            
            logger.info(f"Processed message: intent={intent}, entities={entities}, confidence={confidence:.3f}")
            
            return intent, entities, confidence
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return "general_question", {}, 0.5
    
    def _normalize_message(self, message: str) -> str:
        """Normalize the message for processing."""
        # Convert to lowercase
        normalized = message.lower()
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Remove punctuation except for important patterns
        normalized = re.sub(r'[^\w\s]', ' ', normalized)
        
        return normalized.strip()
    
    def _extract_intent(self, message: str) -> Tuple[str, float]:
        """Extract the primary intent from the message."""
        intent_scores = {}
        
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, message, re.IGNORECASE):
                    score += 1
            
            if score > 0:
                intent_scores[intent] = score
        
        if not intent_scores:
            return "general_question", 0.3
        
        # Get the intent with highest score
        best_intent = max(intent_scores, key=intent_scores.get)
        max_score = intent_scores[best_intent]
        
        # Calculate confidence based on score and pattern matches
        confidence = min(0.9, 0.3 + (max_score * 0.2))
        
        return best_intent, confidence
    
    def _extract_entities(self, message: str) -> Dict[str, Any]:
        """Extract entities from the message."""
        entities = {}
        
        # Extract stock symbols
        symbols = self._extract_stock_symbols(message)
        if symbols:
            entities["symbol"] = symbols[0]  # Take the first symbol found
        
        # Extract time periods
        time_periods = self._extract_time_periods(message)
        if time_periods:
            entities["time_period"] = time_periods[0]
        
        # Extract analysis types
        analysis_types = self._extract_analysis_types(message)
        if analysis_types:
            entities["analysis_type"] = analysis_types[0]
        
        return entities
    
    def _extract_stock_symbols(self, message: str) -> List[str]:
        """Extract stock symbols from the message."""
        found_symbols = []
        
        # Look for exact matches in our stock symbols list
        words = message.upper().split()
        for word in words:
            if word in self.stock_symbols:
                found_symbols.append(word)
        
        # Look for patterns like "AAPL stock" or "stock AAPL"
        symbol_patterns = [
            r"([A-Z]{1,5})\s+stock",
            r"stock\s+([A-Z]{1,5})",
            r"analyze\s+([A-Z]{1,5})",
            r"predict\s+([A-Z]{1,5})",
            r"([A-Z]{1,5})\s+analysis"
        ]
        
        for pattern in symbol_patterns:
            matches = re.findall(pattern, message.upper())
            for match in matches:
                if match not in found_symbols:
                    found_symbols.append(match)
        
        return found_symbols
    
    def _extract_time_periods(self, message: str) -> List[str]:
        """Extract time periods from the message."""
        time_patterns = [
            r"(\d+)\s*days?",
            r"(\d+)\s*weeks?",
            r"(\d+)\s*months?",
            r"(\d+)\s*years?",
            r"next\s*(\d+)\s*days?",
            r"in\s*(\d+)\s*days?"
        ]
        
        found_periods = []
        for pattern in time_patterns:
            matches = re.findall(pattern, message)
            found_periods.extend(matches)
        
        return found_periods
    
    def _extract_analysis_types(self, message: str) -> List[str]:
        """Extract analysis types from the message."""
        analysis_types = []
        
        analysis_keywords = [
            "technical", "fundamental", "sensitivity", 
            "prediction", "forecast", "trend"
        ]
        
        for keyword in analysis_keywords:
            if keyword in message:
                analysis_types.append(keyword)
        
        return analysis_types
    
    def _calculate_confidence(self, intent_confidence: float, entities: Dict[str, Any]) -> float:
        """Calculate overall confidence score."""
        base_confidence = intent_confidence
        
        # Boost confidence if we found relevant entities
        if entities.get("symbol"):
            base_confidence += 0.1
        
        if entities.get("analysis_type"):
            base_confidence += 0.05
        
        if entities.get("time_period"):
            base_confidence += 0.05
        
        # Cap confidence at 0.95
        return min(0.95, base_confidence)
    
    def get_suggested_queries(self, intent: str = None) -> List[str]:
        """Get suggested queries based on intent."""
        suggestions = {
            "stock_prediction": [
                "What's the prediction for AAPL stock?",
                "Forecast the price of TSLA",
                "Will MSFT stock go up?",
                "Predict GOOGL stock price",
                "What's the outlook for AMZN?"
            ],
            "technical_analysis": [
                "Analyze technical indicators for AAPL",
                "Show me RSI and MACD for TSLA",
                "Technical analysis of MSFT",
                "What are the moving averages for GOOGL?",
                "Bollinger Bands analysis for AMZN"
            ],
            "sensitivity_analysis": [
                "What affects AAPL stock price most?",
                "Show sensitivity analysis for TSLA",
                "Key factors influencing MSFT",
                "Feature importance for GOOGL",
                "What drives AMZN stock price?"
            ],
            "general_question": [
                "What can you do?",
                "How does the AI model work?",
                "What's your accuracy?",
                "Tell me about the LSTM model",
                "Show me your capabilities"
            ]
        }
        
        if intent and intent in suggestions:
            return suggestions[intent]
        else:
            # Return a mix of suggestions
            all_suggestions = []
            for intent_suggestions in suggestions.values():
                all_suggestions.extend(intent_suggestions[:2])  # Take first 2 from each
            return all_suggestions
    
    def validate_stock_symbol(self, symbol: str) -> bool:
        """Validate if a stock symbol is supported."""
        return symbol.upper() in self.stock_symbols
    
    def get_supported_symbols(self) -> List[str]:
        """Get list of supported stock symbols."""
        return self.stock_symbols.copy()
