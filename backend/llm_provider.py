"""
LLM Provider for AI Stock GPT
==============================

Provides LLM-powered text generation with a 3-tier fallback:
1. Ollama (local, free)
2. HuggingFace Inference API (cloud, free tier)
3. Template responses (offline fallback)
"""

import os
import json
import logging
from typing import Dict, Any, Optional

import requests

logger = logging.getLogger(__name__)

# Ollama configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "30"))

# HuggingFace configuration
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
HF_MODEL = os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
HF_API_URL = "https://api-inference.huggingface.co/models"
HF_TIMEOUT = int(os.getenv("HF_TIMEOUT", "30"))


class LLMProvider:
    """LLM provider with Ollama -> HuggingFace -> template fallback chain."""

    def __init__(self):
        self.ollama_available = False
        self.hf_available = False
        self._check_providers()

    def _check_providers(self):
        """Check which LLM providers are available."""
        # Check Ollama
        try:
            resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
            if resp.status_code == 200:
                self.ollama_available = True
                logger.info(f"Ollama available at {OLLAMA_BASE_URL}")
        except Exception:
            logger.info("Ollama not available, will use fallback providers")

        # Check HuggingFace
        if HF_API_TOKEN:
            self.hf_available = True
            logger.info("HuggingFace Inference API configured")
        else:
            # HF free tier can work without token for some models
            self.hf_available = True
            logger.info("HuggingFace Inference API available (no token, free tier)")

    def generate_response(
        self, intent: str, entities: Dict[str, Any], user_message: str
    ) -> str:
        """
        Generate a response using the best available LLM provider.

        Args:
            intent: Detected intent (e.g., 'stock_prediction', 'technical_analysis')
            entities: Extracted entities (e.g., {'symbol': 'AAPL'})
            user_message: Original user message

        Returns:
            Generated response string
        """
        prompt = self._build_prompt(intent, entities, user_message)

        # Try Ollama first
        if self.ollama_available:
            response = self._call_ollama(prompt)
            if response:
                return response

        # Try HuggingFace Inference API
        if self.hf_available:
            response = self._call_huggingface(prompt)
            if response:
                return response

        # Fall back to templates
        return self._template_response(intent, entities, user_message)

    def _build_prompt(
        self, intent: str, entities: Dict[str, Any], user_message: str
    ) -> str:
        """Build a context-aware prompt for the LLM."""
        symbol = entities.get("symbol", "")
        time_period = entities.get("time_period", "")
        analysis_type = entities.get("analysis_type", "")

        system_context = (
            "You are AI Stock GPT, an intelligent stock market analysis assistant. "
            "You provide helpful, accurate, and concise financial analysis. "
            "Always include a disclaimer that this is not financial advice. "
            "Keep responses under 200 words."
        )

        intent_context = {
            "stock_prediction": (
                f"The user is asking about stock price predictions"
                f"{' for ' + symbol if symbol else ''}. "
                "Discuss potential price movements, trends, and factors that could "
                "influence the stock. Mention that predictions are based on historical "
                "data and ML models."
            ),
            "technical_analysis": (
                f"The user wants technical analysis"
                f"{' for ' + symbol if symbol else ''}. "
                "Discuss relevant technical indicators like RSI, MACD, Bollinger Bands, "
                "moving averages, and support/resistance levels."
            ),
            "sensitivity_analysis": (
                f"The user is asking about sensitivity analysis"
                f"{' for ' + symbol if symbol else ''}. "
                "Discuss which factors most influence the stock price, feature importance, "
                "and key drivers of price movement."
            ),
            "portfolio_management": (
                "The user is asking about portfolio management. "
                "Discuss diversification, risk management, asset allocation, "
                "and portfolio optimization strategies."
            ),
            "market_news": (
                "The user is asking about market news and trends. "
                "Discuss current market conditions, sector performance, "
                "and notable market events."
            ),
            "general_question": (
                "The user has a general question about the AI Stock GPT system. "
                "Explain your capabilities: stock predictions using LSTM neural networks, "
                "technical analysis, sensitivity analysis, portfolio management, "
                "and market insights."
            ),
        }

        context = intent_context.get(intent, intent_context["general_question"])

        return (
            f"System: {system_context}\n\n"
            f"Context: {context}\n\n"
            f"User: {user_message}\n\n"
            f"Assistant:"
        )

    def _call_ollama(self, prompt: str) -> Optional[str]:
        """Call Ollama REST API for text generation."""
        try:
            resp = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "num_predict": 300,
                    },
                },
                timeout=OLLAMA_TIMEOUT,
            )

            if resp.status_code == 200:
                data = resp.json()
                response_text = data.get("response", "").strip()
                if response_text:
                    logger.info("Response generated via Ollama")
                    return response_text

        except requests.exceptions.Timeout:
            logger.warning("Ollama request timed out")
        except requests.exceptions.ConnectionError:
            logger.warning("Ollama connection failed")
            self.ollama_available = False
        except Exception as e:
            logger.error(f"Ollama error: {e}")

        return None

    def _call_huggingface(self, prompt: str) -> Optional[str]:
        """Call HuggingFace Inference API for text generation."""
        try:
            headers = {"Content-Type": "application/json"}
            if HF_API_TOKEN:
                headers["Authorization"] = f"Bearer {HF_API_TOKEN}"

            resp = requests.post(
                f"{HF_API_URL}/{HF_MODEL}",
                headers=headers,
                json={
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": 300,
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "return_full_text": False,
                    },
                },
                timeout=HF_TIMEOUT,
            )

            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list) and len(data) > 0:
                    response_text = data[0].get("generated_text", "").strip()
                    if response_text:
                        logger.info("Response generated via HuggingFace API")
                        return response_text
            elif resp.status_code == 503:
                logger.warning("HuggingFace model is loading, using fallback")
            else:
                logger.warning(f"HuggingFace API returned {resp.status_code}")

        except requests.exceptions.Timeout:
            logger.warning("HuggingFace request timed out")
        except Exception as e:
            logger.error(f"HuggingFace error: {e}")

        return None

    def _template_response(
        self, intent: str, entities: Dict[str, Any], user_message: str
    ) -> str:
        """Generate a template-based response as final fallback."""
        symbol = entities.get("symbol", "the stock")

        templates = {
            "stock_prediction": (
                f"Based on our LSTM neural network analysis of {symbol}, "
                f"I can provide insights into potential price movements. "
                f"Our model analyzes historical price data, volume patterns, "
                f"and technical indicators to generate predictions. "
                f"For detailed predictions with confidence scores, please use "
                f"the /predict/{symbol} endpoint. "
                f"Note: This is not financial advice. Always do your own research "
                f"before making investment decisions."
            ),
            "technical_analysis": (
                f"For technical analysis of {symbol}, our system evaluates key indicators "
                f"including RSI (Relative Strength Index), MACD (Moving Average Convergence "
                f"Divergence), Bollinger Bands, and various moving averages. "
                f"Use the /technical/{symbol} endpoint for detailed indicator values. "
                f"Note: Technical analysis is one tool among many for investment decisions."
            ),
            "sensitivity_analysis": (
                f"Sensitivity analysis for {symbol} examines which factors have the greatest "
                f"impact on price predictions. Key factors typically include trading volume, "
                f"moving average crossovers, RSI values, and market volatility. "
                f"Use the /sensitivity/{symbol} endpoint for detailed feature importance rankings."
            ),
            "portfolio_management": (
                "I can help you manage your investment portfolio. Our system supports "
                "tracking multiple stocks, monitoring performance, and setting up alerts. "
                "Use the portfolio endpoints to add stocks, view performance, and get "
                "diversification recommendations."
            ),
            "market_news": (
                "I can provide insights on current market trends and conditions. "
                "Our AI analyzes multiple data points to identify market patterns. "
                "For specific stock analysis, try asking about a particular symbol."
            ),
            "general_question": (
                "I'm AI Stock GPT, your intelligent stock market analysis assistant. "
                "I can help you with:\n"
                "- Stock price predictions using LSTM neural networks\n"
                "- Technical indicator analysis (RSI, MACD, Bollinger Bands)\n"
                "- Sensitivity analysis to understand price drivers\n"
                "- Portfolio management and tracking\n"
                "Try asking something like 'Predict AAPL stock' or "
                "'Technical analysis of TSLA'."
            ),
        }

        response = templates.get(intent, templates["general_question"])
        logger.info("Response generated via template fallback")
        return response


# Global instance
llm_provider = LLMProvider()
