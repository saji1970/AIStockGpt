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
from typing import Dict, Any, Optional, List

import requests

logger = logging.getLogger(__name__)

# Lazy import to avoid circular dependencies
_training_retriever = None

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
        self.training_retriever = None
        self._check_providers()

    def set_training_retriever(self, retriever):
        """Set the training retriever for RAG-based few-shot prompting."""
        self.training_retriever = retriever
        if retriever and retriever.ready:
            logger.info(f"LLM provider: training retriever attached ({len(retriever.examples)} examples)")
        else:
            logger.info("LLM provider: training retriever attached (not ready or empty)")

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
        self, intent: str, entities: Dict[str, Any], user_message: str,
        ml_results: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate a response using the best available LLM provider.

        Args:
            intent: Detected intent (e.g., 'stock_prediction', 'technical_analysis')
            entities: Extracted entities (e.g., {'symbol': 'AAPL'})
            user_message: Original user message
            ml_results: Optional ML model results to include in prompt context

        Returns:
            Generated response string
        """
        prompt = self._build_prompt(intent, entities, user_message, ml_results)

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
        return self._template_response(intent, entities, user_message, ml_results)

    def _build_prompt(
        self, intent: str, entities: Dict[str, Any], user_message: str,
        ml_results: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build a context-aware prompt for the LLM."""
        symbol = entities.get("symbol", "")
        time_period = entities.get("time_period", "")
        analysis_type = entities.get("analysis_type", "")

        system_context = (
            "You are AI Stock GPT, an advanced AI investment advisor and portfolio strategist "
            "powered by XGBoost ML models, Monte Carlo simulations, and FinBERT sentiment analysis. "
            "You behave like an experienced wealth management advisor: you explain reasoning, "
            "evaluate risk, discuss diversification, and think long-term. "
            "Always include specific ticker symbols with company names (e.g., AAPL (Apple)). "
            "For Indian stocks, use ₹ for prices. For US stocks, use $. "
            "When recommending allocations, always include specific amounts per stock/ETF. "
            "When ML model results are provided, reference the key numbers "
            "(direction, probability, confidence, indicators) in your response. "
            "Discuss risks and tradeoffs. Never guarantee returns or encourage speculation. "
            "Always include a disclaimer that this is not financial advice. "
            "Keep responses under 400 words."
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
            "market_advice": (
                "The user is asking for broad market or investment advice. "
                "Provide helpful guidance on top stocks, ETFs, sectors, hedge funds, "
                "mutual funds, index funds, diversification strategies, and asset allocation. "
                "Include specific ticker symbols (with company names) and ETFs when relevant. "
                "Cover different risk levels and investment goals. "
                "When allocation data is provided, always include the SPECIFIC amount per stock "
                "(e.g., 'Invest $200 in AAPL (Apple)' or 'Invest ₹10,000 in INFY.BSE (Infosys)'). "
                "Use ₹ for Indian stocks (.BSE/.NSE) and $ for US stocks. "
                "If the user asked about India, only recommend Indian stocks."
            ),
            "market_news": (
                "The user is asking about market news and trends. "
                "Discuss current market conditions, sector performance, "
                "and notable market events."
            ),
            "retirement_planning": (
                "The user is asking about retirement or age-based investing. "
                "Discuss age-appropriate asset allocation, glide paths (shifting from stocks to bonds over time), "
                "sequence-of-returns risk, retirement income strategies, and the importance of de-risking "
                "as retirement approaches. Include specific allocation percentages by age bracket. "
                "Mention dividend income, bond laddering, and cash buffers for retirees."
            ),
            "behavioral_coaching": (
                "The user is expressing emotional distress or behavioral challenges with investing. "
                "Provide empathetic, evidence-based behavioral guidance. Reference historical data showing "
                "that every crash has recovered. Discuss the psychology of panic selling, FOMO, hype chasing, "
                "and regret. Suggest practical strategies: written investment plans, automation, reduced "
                "portfolio checking frequency, right-sizing risk tolerance. Be supportive but honest."
            ),
            "income_strategy": (
                "The user wants income from their investments. Discuss dividend ETFs (SCHD, VYM, VIG, DGRO), "
                "REITs (VNQ), bond funds (BND, AGG), and high-yield options. Compare dividend yield vs total return. "
                "Explain dividend reinvestment, tax implications of dividends, and how to build a portfolio "
                "that generates regular income. Include specific yields and ticker symbols."
            ),
            "macro_analysis": (
                "The user is asking about macroeconomic factors and their impact on investments. "
                "Discuss interest rates, inflation, recession indicators, Fed monetary policy, "
                "stagflation, and business cycle sectors. Reference historical examples (1970s stagflation, "
                "2008 crisis, 2022 rate hikes). Explain which asset classes and sectors perform best "
                "in each economic environment. Be educational and data-driven."
            ),
            "risk_assessment": (
                "The user wants capital preservation or defensive investment strategy. "
                "Discuss safe-haven assets (Treasury bonds TLT/SHY, gold GLD, cash), defensive sectors "
                "(Consumer Staples XLP, Healthcare XLV, Utilities XLU), and portfolio construction "
                "for crash resistance. Include expected max drawdowns for different allocations. "
                "Emphasize that risk reduction means accepting lower returns in exchange for stability."
            ),
            "comparative_analysis": (
                "The user is comparing two or more investments. Provide a balanced, data-driven comparison "
                "covering: historical returns, volatility, expense ratios, dividend yield, sector exposure, "
                "and risk characteristics. Use tables where helpful. Present pros and cons of each option "
                "without bias. Conclude with guidance on when each option is more appropriate."
            ),
            "beginner_guidance": (
                "The user is a beginner investor. Keep the language simple and jargon-free. "
                "Recommend starting with broad index funds (VTI, VOO) or target-date funds. "
                "Explain core concepts: diversification, compound growth, long-term perspective. "
                "Suggest a simple 2-3 fund portfolio. Emphasize starting early and investing consistently. "
                "Warn against common beginner mistakes: stock picking, timing the market, "
                "taking advice from social media."
            ),
            "financial_planning": (
                "The user has a specific financial goal (education savings, financial independence, "
                "home purchase, etc.). Discuss goal-based investing: calculate required savings, "
                "recommend appropriate time-horizon allocations, explain 529 plans for education, "
                "the 4% rule for retirement withdrawal, and dollar-cost averaging. "
                "Provide specific numbers and timelines based on the user's stated goals."
            ),
            "general_question": (
                "The user has a general question about the AI Stock GPT system. "
                "Explain your capabilities: stock predictions using LSTM neural networks, "
                "technical analysis, sensitivity analysis, portfolio management, "
                "and market insights."
            ),
        }

        context = intent_context.get(intent, intent_context["general_question"])

        prompt = (
            f"System: {system_context}\n\n"
            f"Context: {context}\n\n"
        )

        # Inject few-shot example from training data via RAG
        if self.training_retriever and self.training_retriever.ready:
            try:
                examples = self.training_retriever.find_best_example(user_message, top_k=1, min_similarity=0.3)
                if examples:
                    ex = examples[0]
                    prompt += (
                        f"## Reference Example (similar query, similarity={ex['similarity']:.2f}):\n"
                        f"User: {ex['user']}\n"
                        f"Assistant: {ex['assistant'][:600]}\n\n"
                        f"Use the style, depth, and structure of the above example to guide your response. "
                        f"Adapt the content to the user's specific question below.\n\n"
                    )
            except Exception as e:
                logger.warning(f"Training retriever lookup failed: {e}")

        if ml_results:
            prompt += "\n\n## ML Model Results (use these to inform your response):\n"
            prompt += json.dumps(ml_results, indent=2, default=str)
            prompt += "\n\nExplain these results in clear, natural language. Include the key numbers."

        prompt += f"\n\nUser: {user_message}\n\nAssistant:"
        return prompt

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
        self, intent: str, entities: Dict[str, Any], user_message: str,
        ml_results: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate a template-based response as final fallback."""
        # Format ML results if available
        if ml_results:
            formatted = self._format_ml_results(ml_results)
            if formatted:
                return formatted

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
            "market_advice": (
                "Here are popular investment options with suggested allocations:\n\n"
                "**Top Large-Cap Stocks:** AAPL (Apple), MSFT (Microsoft), NVDA (NVIDIA), GOOGL (Alphabet), AMZN (Amazon)\n"
                "**Dividend Stocks:** JNJ (Johnson & Johnson), KO (Coca-Cola), PEP (PepsiCo), PG (Procter & Gamble)\n"
                "**Index ETFs:** SPY (S&P 500), QQQ (Nasdaq-100), VTI (Total Market)\n"
                "**Sector ETFs:** XLK (Tech), XLV (Healthcare), XLF (Financials)\n"
                "**Hedge Fund Alternatives:** DBMF (Managed Futures), BTAL (Anti-Beta), MNA (Merger Arbitrage)\n"
                "**Bond ETFs:** BND (Total Bond), AGG (Aggregate Bond) for lower risk\n\n"
                "**Indian Market:** INFY.BSE (Infosys), TCS.BSE (TCS), HDFCBANK.BSE (HDFC Bank), RELIANCE.BSE (Reliance)\n\n"
                "For specific allocation advice, try: *\"Invest $500 in US stocks\"* or *\"Invest ₹50,000 in Indian stocks\"*\n\n"
                "Note: This is not financial advice. Always do your own research."
            ),
            "market_news": (
                "I can provide insights on current market trends and conditions. "
                "Our AI analyzes multiple data points to identify market patterns. "
                "For specific stock analysis, try asking about a particular symbol."
            ),
            "retirement_planning": (
                "**Retirement Portfolio Planning**\n\n"
                "As you approach retirement, the key is gradually shifting from growth to preservation and income:\n\n"
                "**Age-Based Guidelines:**\n"
                "- Age 50-55: 60-70% stocks / 25-30% bonds / 5-10% cash\n"
                "- Age 55-60: 50-60% stocks / 30-35% bonds / 10-15% cash\n"
                "- Age 60-65: 40-50% stocks / 35-40% bonds / 10-20% cash\n"
                "- Age 65+: 30-40% stocks / 40-50% bonds / 15-25% cash\n\n"
                "**Key Strategies:**\n"
                "- Shift from growth (QQQ) to dividend/value (SCHD, VYM)\n"
                "- Build a 2-3 year cash buffer for living expenses\n"
                "- Diversify bonds: BND (broad), TIP (inflation-protected), SHY (short-term)\n"
                "- Consider REITs (VNQ) for income\n\n"
                "Try asking with your specific age and amount for a personalized allocation.\n\n"
                "Note: This is not financial advice."
            ),
            "behavioral_coaching": (
                "**Investment Behavioral Guidance**\n\n"
                "Emotional reactions to markets are completely normal. Here's what the data shows:\n\n"
                "- Every market crash in history has been followed by recovery\n"
                "- Missing the 10 best trading days over 20 years roughly halves your returns\n"
                "- Most of the best days occur during or immediately after crashes\n\n"
                "**Practical Strategies:**\n"
                "1. Write an investment plan now, before the next crash\n"
                "2. Automate your contributions so emotions can't intervene\n"
                "3. Check your portfolio quarterly, not daily\n"
                "4. If crashes cause panic, your allocation is too aggressive - reduce risk\n"
                "5. Keep 6-12 months expenses in cash as a psychological safety net\n\n"
                "The investors who build the most wealth invest THROUGH crashes, not around them.\n\n"
                "Note: If investment anxiety significantly impacts your life, consider a fee-only financial advisor."
            ),
            "income_strategy": (
                "**Income Investing Strategies**\n\n"
                "Here are the main approaches to generating investment income:\n\n"
                "**Dividend ETFs:**\n"
                "- SCHD (Schwab Dividend Equity): ~3.5% yield, quality companies\n"
                "- VYM (Vanguard High Dividend): ~3% yield, broad diversification\n"
                "- VIG (Vanguard Dividend Appreciation): ~2% yield, growing dividends\n\n"
                "**Bond ETFs:**\n"
                "- BND (Total Bond Market): ~4-5% yield, broad bond exposure\n"
                "- AGG (Aggregate Bond): ~4-5% yield, investment grade\n"
                "- TLT (Long Treasury): ~4% yield, government safety\n\n"
                "**Real Estate:**\n"
                "- VNQ (Vanguard REIT): ~4% yield, real estate income\n\n"
                "**A balanced income portfolio** might combine: SCHD (40%) + BND (30%) + VNQ (15%) + VYM (15%)\n\n"
                "Try asking with a specific amount for detailed allocation.\n\n"
                "Note: This is not financial advice."
            ),
            "macro_analysis": (
                "**Macroeconomic Analysis**\n\n"
                "Different economic environments favor different asset classes:\n\n"
                "**Rising Interest Rates:**\n"
                "- Bonds fall (especially long-term TLT)\n"
                "- Banks benefit (XLF)\n"
                "- Growth stocks hurt (high valuations compressed)\n\n"
                "**High Inflation:**\n"
                "- TIPS outperform (inflation-protected)\n"
                "- Commodities and energy rise (XLE)\n"
                "- Gold tends to benefit (GLD)\n"
                "- Consumer staples maintain pricing power (XLP)\n\n"
                "**Recession:**\n"
                "- Defensive sectors outperform: Healthcare (XLV), Staples (XLP), Utilities (XLU)\n"
                "- Treasury bonds rally (TLT)\n"
                "- Cyclical sectors underperform\n\n"
                "**Business Cycle Sectors:**\n"
                "- Early Recovery: Financials, Consumer Discretionary\n"
                "- Mid-Expansion: Technology, Industrials\n"
                "- Late Expansion: Energy, Materials\n"
                "- Recession: Healthcare, Staples, Utilities\n\n"
                "Note: This is educational analysis, not a prediction."
            ),
            "risk_assessment": (
                "**Capital Preservation & Risk Management**\n\n"
                "**Safe-Haven Assets:**\n"
                "- US Treasury Bonds (TLT/SHY): Government-backed, rise during stock crashes\n"
                "- Gold (GLD): 5,000-year store of value, low correlation with stocks\n"
                "- Cash/Money Market: Zero price risk, immediate liquidity\n\n"
                "**Defensive Portfolio Example:**\n"
                "- Consumer Staples (XLP): 15% — essential goods companies\n"
                "- Healthcare (XLV): 15% — non-discretionary spending\n"
                "- Quality Dividend (SCHD): 12% — strong balance sheet companies\n"
                "- Treasury Bonds (TLT): 10% — crash cushion\n"
                "- Short-Term Bonds (SHY): 8% — stability\n"
                "- Gold (GLD): 7% — crisis hedge\n"
                "- VOO (S&P 500): 15% — core equity (reduced)\n"
                "- Cash: 5% — dry powder\n\n"
                "**Expected drawdowns:** -10% to -18% in severe recessions (vs -35% to -50% for aggressive portfolios)\n\n"
                "Note: This is not financial advice."
            ),
            "comparative_analysis": (
                "**Investment Comparison**\n\n"
                "I can compare stocks, ETFs, and investment strategies. Here's an example framework:\n\n"
                "**VOO (S&P 500) vs QQQ (Nasdaq-100):**\n"
                "- VOO: Broader diversification (500 stocks), lower volatility, ~0.03% expense ratio\n"
                "- QQQ: Tech-heavy (100 stocks), higher growth potential, higher volatility, ~0.20% expense ratio\n"
                "- VOO max drawdown: ~-34% | QQQ max drawdown: ~-50%\n\n"
                "**Growth vs Dividend Investing:**\n"
                "- Growth (QQQ): Higher returns pre-retirement, more volatile, tax-efficient\n"
                "- Dividend (SCHD): Lower volatility, regular income, easier to hold psychologically\n"
                "- Best approach: Blend both based on your age and risk tolerance\n\n"
                "Try asking about specific stocks or ETFs to compare.\n\n"
                "Note: This is not financial advice."
            ),
            "beginner_guidance": (
                "**Getting Started with Investing**\n\n"
                "Welcome! Here's a simple framework:\n\n"
                "**Step 1: Build an emergency fund** (6 months expenses in savings account)\n\n"
                "**Step 2: Start with a simple portfolio:**\n"
                "- VTI (Total US Stock Market): 70% — owns 4,000+ US stocks\n"
                "- VXUS (Total International): 20% — owns 8,000+ global stocks\n"
                "- BND (Total Bond Market): 10% — stability anchor\n\n"
                "**Step 3: Automate** — Set up automatic monthly investments\n\n"
                "**Key principles:**\n"
                "- Start early, invest consistently\n"
                "- Don't try to time the market\n"
                "- Keep costs low (index funds charge ~0.03%)\n"
                "- Don't check your portfolio daily\n"
                "- Ignore stock tips from social media\n\n"
                "This simple 3-fund portfolio has outperformed 80-90% of professional fund managers over 20 years.\n\n"
                "Note: This is not financial advice."
            ),
            "financial_planning": (
                "**Goal-Based Financial Planning**\n\n"
                "**Financial Independence (4% Rule):**\n"
                "- Annual expenses × 25 = your FI number\n"
                "- $50K/year expenses → need $1.25M invested\n"
                "- $75K/year expenses → need $1.875M invested\n\n"
                "**Education Savings (529 Plan):**\n"
                "- Tax-advantaged growth for college expenses\n"
                "- Start early: $250/month for 18 years at 7% return ≈ $100K\n\n"
                "**Dollar-Cost Averaging vs Lump Sum:**\n"
                "- Lump sum wins ~68% of the time (markets go up more than down)\n"
                "- DCA is better psychologically for nervous investors\n"
                "- Most important: invest consistently, don't wait for the 'perfect' time\n\n"
                "**Savings Rate Impact (starting from $0, 5% real return):**\n"
                "- 20% savings rate → ~37 years to FI\n"
                "- 40% savings rate → ~22 years to FI\n"
                "- 60% savings rate → ~12.5 years to FI\n\n"
                "Try asking with your specific goal and amount for a detailed plan.\n\n"
                "Note: This is not financial advice."
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


    def _format_ml_results(self, ml_results: Dict[str, Any]) -> Optional[str]:
        """Format ML results into readable text when LLM is unavailable."""
        lines = []

        if ml_results.get('prediction'):
            pred = ml_results['prediction']
            model_label = "XGBoost+LightGBM Ensemble" if pred.get('ensemble') else "XGBoost ML Model"
            lines.append(f"**{pred.get('symbol', '')} Analysis ({model_label})**\n")
            lines.append(f"Direction: **{pred.get('direction', 'N/A')}**")
            prob = pred.get('probability', 0)
            lines.append(f"Calibrated probability: **{prob:.0%}**")
            prob_raw = pred.get('probability_raw')
            if prob_raw and abs(prob_raw - prob) > 0.01:
                lines.append(f"Raw probability: {prob_raw:.0%}")
            conf = pred.get('confidence', 0)
            lines.append(f"Confidence: **{conf:.0%}**")
            exp_ret = pred.get('expected_return', 0)
            horizon = pred.get('horizon_days', 21)
            lines.append(f"Expected {horizon}-day return: **{exp_ret:.1%}**")
            importance = pred.get('feature_importance', {})
            if importance:
                top = list(importance.keys())[:5]
                lines.append(f"\nTop factors: {', '.join(top)}")
            lines.append(f"\n*{horizon}-day horizon. Not financial advice.*")

        if ml_results.get('allocation'):
            alloc = ml_results['allocation']
            risk = alloc.get('risk_level', 'moderate')
            alloc_market = alloc.get('market', 'global')
            cs = alloc.get('currency_symbol', '$')
            market_label = {'india': 'Indian Market', 'us': 'US Market', 'global': 'Global'}.get(alloc_market, 'Global')
            lines.append(f"\n**Recommended Allocation - {market_label} ({risk})**\n")
            for sym, info in alloc.get('allocations', {}).items():
                w = info.get('weight', 0)
                amt = info.get('amount', 0)
                # Use ₹ for Indian symbols, $ for US
                sym_cs = '₹' if (sym.endswith('.BSE') or sym.endswith('.NSE')) else '$'
                lines.append(f"- **{sym}**: {w:.0%} — Invest **{sym_cs}{amt:,.0f}**")

            ret_range = alloc.get('expected_return_range', {})
            if ret_range:
                lines.append(f"\nExpected annual return: {ret_range.get('low', 0):.1%} to {ret_range.get('high', 0):.1%}")

        if ml_results.get('forecast'):
            fc = ml_results['forecast']
            # Use allocation currency if available, else default to $
            fc_cs = '$'
            if ml_results.get('allocation'):
                fc_cs = ml_results['allocation'].get('currency_symbol', '$')
            lines.append(f"\n**Monte Carlo Forecast ({fc.get('months', 12)}mo, 10K simulations)**\n")
            lines.append(f"- Median outcome: **{fc_cs}{fc.get('median_value', 0):,.0f}**")
            lines.append(f"- {fc.get('probability_positive', 0):.0%} chance of positive return")
            lines.append(f"- Best case (95th): {fc_cs}{fc.get('best_case', 0):,.0f}")
            lines.append(f"- Worst case (5th): {fc_cs}{fc.get('worst_case', 0):,.0f}")
            var95 = fc.get('var_95', 0)
            if var95:
                lines.append(f"- Value at Risk (95%): {fc_cs}{var95:,.0f}")

        if ml_results.get('sentiment'):
            sent = ml_results['sentiment']
            lines.append(f"\n**{sent.get('symbol', '')} Sentiment Analysis (FinBERT)**\n")
            lines.append(f"Overall sentiment: **{sent.get('overall_label', 'neutral')}** ({sent.get('overall_sentiment', 0):.2f})")
            lines.append(f"Confidence: {sent.get('confidence', 0):.0%}")
            lines.append(f"Trend: {sent.get('trend', 'stable')}")
            lines.append(f"Headlines analyzed: {sent.get('headline_count', 0)}")

        if ml_results.get('indicators'):
            ind = ml_results['indicators']
            sym = ind.get('symbol', '')
            lines.append(f"\n**{sym} Technical Indicators**\n")
            rsi = ind.get('rsi_14', 0)
            rsi_sig = ind.get('rsi_signal', 'neutral')
            lines.append(f"- RSI (14): **{rsi:.1f}** ({rsi_sig})")
            lines.append(f"- MACD: {ind.get('macd', 0):.2f} | Signal: {ind.get('macd_signal', 0):.2f} ({ind.get('macd_signal_direction', 'neutral')})")
            lines.append(f"- Bollinger Bands: {ind.get('bb_signal', 'within bands')}")
            lines.append(f"- SMA 20: ${ind.get('sma_20', 0):.2f} | SMA 50: ${ind.get('sma_50', 0):.2f}")
            lines.append(f"- ATR (14): {ind.get('atr_14', 0):.2f}")
            lines.append(f"- Stochastic: K={ind.get('stoch_k', 0):.1f} D={ind.get('stoch_d', 0):.1f}")
            lines.append(f"- ADX: {ind.get('adx_14', 0):.1f}")
            hist_vol = ind.get('hist_vol_20', 0)
            lines.append(f"- Volatility (20d): {hist_vol:.1%}")
            sharpe = ind.get('rolling_sharpe_21', 0)
            lines.append(f"- Rolling Sharpe (21d): {sharpe:.2f}")

        if ml_results.get('risk'):
            risk = ml_results['risk']
            lines.append("\n**Portfolio Risk Analysis**\n")
            lines.append(f"- Sharpe Ratio: {risk.get('sharpe_ratio', 0):.2f}")
            lines.append(f"- Annual Volatility: {risk.get('annual_volatility', 0):.1%}")
            lines.append(f"- Max Drawdown: {risk.get('max_drawdown', 0):.1%}")
            lines.append(f"- Value at Risk (95%): {risk.get('var_95', 0):.2%} daily")
            lines.append(f"- Beta: {risk.get('beta', 0):.2f}")

        if not lines:
            return None

        lines.append("\n\n*Not financial advice.*")
        return '\n'.join(lines)


# Global instance
llm_provider = LLMProvider()
