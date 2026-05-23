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
        chat_history: Optional[List[Dict[str, Any]]] = None,
        investor_profile: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate a response using the best available LLM provider.

        Args:
            intent: Detected intent (e.g., 'stock_prediction', 'technical_analysis')
            entities: Extracted entities (e.g., {'symbol': 'AAPL'})
            user_message: Original user message
            ml_results: Optional ML model results to include in prompt context
            chat_history: Optional recent chat messages for conversational context
            investor_profile: Optional persistent investor profile from database

        Returns:
            Generated response string
        """
        prompt = self._build_prompt(intent, entities, user_message, ml_results, chat_history, investor_profile)

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
        return self._template_response(intent, entities, user_message, ml_results, investor_profile)

    def _build_prompt(
        self, intent: str, entities: Dict[str, Any], user_message: str,
        ml_results: Optional[Dict[str, Any]] = None,
        chat_history: Optional[List[Dict[str, Any]]] = None,
        investor_profile: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build a context-aware prompt for the LLM."""
        symbol = entities.get("symbol", "")
        time_period = entities.get("time_period", "")
        analysis_type = entities.get("analysis_type", "")

        system_context = (
            "You are AI Stock GPT, an advanced AI investment advisor and portfolio strategist "
            "powered by XGBoost ML models, Monte Carlo simulations, and FinBERT sentiment analysis. "
            "You behave like an experienced wealth management advisor at a top firm: you give "
            "personalized, actionable advice tailored to each client's specific situation. "
            "When the user provides personal details (age, risk tolerance, timeline, amount), "
            "use them to give SPECIFIC recommendations — not generic guidelines. "
            "When key details are MISSING, ask 2-3 brief clarifying questions before giving advice. "
            "Structure allocation recommendations as markdown tables when possible. "
            "Always include specific ticker symbols with company names (e.g., AAPL (Apple)). "
            "For Indian stocks, use ₹ for prices. For US stocks, use $. "
            "When recommending allocations, always include specific amounts per stock/ETF. "
            "When ML model results are provided, reference the key numbers "
            "(direction, probability, confidence, indicators) in your response. "
            "Discuss risks and tradeoffs. Never guarantee returns or encourage speculation. "
            "Always include a disclaimer that this is not financial advice. "
            "Keep responses under 500 words."
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
                "Act like an experienced portfolio strategist. "
                "IMPORTANT: If the user's question is vague (e.g., 'how should I invest', "
                "'suggest investments', 'best way to invest'), ask 2-3 clarifying questions first: "
                "(1) investment amount, (2) time horizon, (3) risk tolerance (conservative/moderate/aggressive), "
                "and (4) any goals (retirement, income, growth, wealth building). "
                "If details ARE provided, give a SPECIFIC allocation plan as a markdown table "
                "with columns: Asset Type | Allocation % | Amount | Example ETFs/Tickers. "
                "Tailor recommendations by risk profile:\n"
                "- Conservative: heavy bonds/dividend ETFs (BND, SCHD, VYM), minimal growth\n"
                "- Moderate: balanced mix (VOO 40%, SCHD 20%, BND 20%, VXUS 15%, cash 5%)\n"
                "- Aggressive: growth-heavy (QQQ, VGT, individual growth stocks), minimal bonds\n"
                "Include specific ticker symbols with company names. "
                "When allocation data is provided, include the SPECIFIC amount per stock "
                "(e.g., 'Invest $200 in AAPL (Apple)' or 'Invest ₹10,000 in INFY.BSE (Infosys)'). "
                "Use ₹ for Indian stocks (.BSE/.NSE) and $ for US stocks. "
                "If the user asked about India, only recommend Indian stocks. "
                "When the question is about finding growth stocks, cheap stocks, or stock ideas, "
                "structure the answer like a research note with a markdown table. "
                "Discuss rebalancing frequency, tax implications, and diversification."
            ),
            "market_news": (
                "The user is asking about market news and trends. "
                "Discuss current market conditions, sector performance, "
                "and notable market events."
            ),
            "retirement_planning": (
                "The user is asking about retirement or age-based investing. "
                "Act like an experienced wealth management advisor giving a personal consultation. "
                "IMPORTANT: If the user has NOT provided their age, approximate retirement timeline, "
                "risk comfort level, or investment amount, START by asking 2-3 brief clarifying questions "
                "(e.g., 'To give you a tailored plan, could you share: (1) your approximate age or birth year, "
                "(2) when you'd like to retire, and (3) how much you're planning to invest?'). "
                "If the user HAS provided age or timeline details, give SPECIFIC personalized advice: "
                "provide a markdown table with columns: Asset Type | Suggested Allocation | Example ETFs/Funds. "
                "Tailor allocations to their age bracket:\n"
                "- Under 40: 80-90% equities (VTI, QQQ), 10-20% bonds (BND)\n"
                "- 40-50: 70-80% equities, 15-25% bonds, 5% cash\n"
                "- 50-55: 60-70% equities (shift toward dividend: SCHD, VYM), 25-30% bonds, 5-10% cash\n"
                "- 55-60: 50-60% equities, 30-35% bonds (BND, TIP), 10-15% cash buffer\n"
                "- 60-65: 40-50% equities, 35-40% bonds, 10-20% cash\n"
                "- 65+: 30-40% equities, 40-50% bonds/income, 15-25% cash\n"
                "Discuss: glide path strategy, sequence-of-returns risk near retirement, "
                "tax-efficient placement (bonds in tax-deferred, equities in taxable), "
                "building a 2-3 year cash buffer, dividend income vs total return, "
                "and bond laddering for retirees. Keep it actionable and specific."
            ),
            "behavioral_coaching": (
                "The user is expressing emotional distress or behavioral challenges with investing. "
                "Provide empathetic, evidence-based behavioral guidance. Reference historical data showing "
                "that every crash has recovered. Discuss the psychology of panic selling, FOMO, hype chasing, "
                "and regret. Suggest practical strategies: written investment plans, automation, reduced "
                "portfolio checking frequency, right-sizing risk tolerance. Be supportive but honest."
            ),
            "income_strategy": (
                "The user wants income from their investments. Act like a wealth advisor specializing in income portfolios. "
                "If the user hasn't specified their investment amount, age, or monthly income target, "
                "ask briefly: 'To build your income plan, I'd like to know: (1) how much you're investing, "
                "(2) your target monthly/annual income, and (3) your age or retirement timeline.' "
                "When details are available, provide a specific income portfolio as a markdown table "
                "with columns: Asset Type | Allocation % | Ticker | Current Yield | Annual Income (on their amount). "
                "Cover: dividend ETFs (SCHD ~3.5%, VYM ~3%, VIG ~2%, DGRO ~2.3%), "
                "REITs (VNQ ~4%, SCHH ~3.5%), bond funds (BND ~4.5%, AGG ~4.5%, TLT ~4%), "
                "and high-yield options (JEPI ~7%, JEPQ ~9%) with risk tradeoffs. "
                "Discuss: dividend growth vs high yield, tax implications (qualified vs ordinary dividends), "
                "DRIP (dividend reinvestment), and sustainable withdrawal rates. "
                "Include total projected annual income from the portfolio."
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
                "Act like a risk management advisor. "
                "If the user hasn't specified their investment amount or what they're protecting against "
                "(market crash, inflation, recession, near-term spending need), ask briefly. "
                "When details are available, provide a specific defensive portfolio as a markdown table "
                "with columns: Asset Type | Allocation % | Ticker | Max Drawdown | Role. "
                "Cover safe-haven assets (Treasury bonds TLT/SHY, gold GLD, cash), "
                "defensive sectors (Consumer Staples XLP, Healthcare XLV, Utilities XLU), "
                "and low-volatility options (USMV, SPLV). "
                "Include expected portfolio max drawdown vs S&P 500 drawdown in past crises "
                "(2008: S&P -50%, defensive portfolio -15 to -20%). "
                "Discuss: the tradeoff of lower returns for stability, inflation risk of being too conservative, "
                "and the concept of 'risk capacity' vs 'risk tolerance'."
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
                "The user has a specific financial goal. Act like a certified financial planner. "
                "If the user hasn't specified their goal amount, timeline, current savings, or monthly contribution capacity, "
                "ask briefly: 'To build your plan, I'd like to know: (1) your financial goal and target amount, "
                "(2) your timeline, (3) current savings, and (4) how much you can invest monthly.' "
                "When details are available, provide a specific plan with: "
                "a savings trajectory table (Year | Contribution | Growth | Balance), "
                "recommended asset allocation by time horizon, and specific fund recommendations. "
                "Cover: 529 plans for education, 4% rule for retirement withdrawal, "
                "dollar-cost averaging vs lump sum, and tax-advantaged accounts (401k, IRA, Roth). "
                "Calculate specific numbers: required monthly savings to reach the goal, "
                "expected portfolio value at different return assumptions (conservative 5%, moderate 7%, optimistic 9%)."
            ),
            "currency_conversion": (
                f"The user is asking about currency exchange rates"
                f"{' for ' + symbol if symbol else ''}. "
                "Provide the current rate if available, show example conversions at different "
                "amounts, and explain what factors affect exchange rates (interest rate differentials, "
                "inflation, trade balances, economic growth, geopolitics). "
                "Include country flags and currency symbols for visual richness. "
                "Mention that exchange rates fluctuate throughout the day based on the forex market."
            ),
            "currency_investment": (
                "The user is asking about investing in currencies or forex trading. "
                "Explain that currency trading is usually much riskier than stock/ETF investing. "
                "Cover: forex trading vs currency swaps, strongest major currencies (USD, CHF, SGD, "
                "EUR, JPY, GBP) with risk levels, challenges of forex trading (leverage, volatility, "
                "24/5 markets), and safer alternatives (international ETFs, USD-denominated assets). "
                "Suggest a balanced allocation: 70-80% diversified long-term investments, 10-20% "
                "high-growth, 5% or less for speculative forex/crypto trades. "
                "Include ways to get currency exposure indirectly: US stocks from India, "
                "international ETFs, currency ETFs (FXE, FXY, FXB). "
                "Be educational and balanced - acknowledge that forex can be profitable but "
                "emphasize the risks for beginners."
            ),
            "general_question": (
                "The user has a general question about the AI Stock GPT system. "
                "Explain your capabilities: stock predictions using LSTM neural networks, "
                "technical analysis, sensitivity analysis, portfolio management, "
                "and market insights."
            ),
            "buffett_analysis": (
                f"The user is asking for a Warren Buffett-style value investing analysis"
                f"{' of ' + symbol if symbol else ''}. "
                "Discuss economic moat (durable competitive advantage), intrinsic value, "
                "margin of safety, financial health (low debt), earnings consistency, "
                "and management quality (capital allocation, ROIC, buybacks). "
                "Reference Buffett's principles: buy wonderful companies at fair prices, "
                "invest within your circle of competence, and hold for the long term. "
                "If Buffett score data is provided, incorporate the scores and sub-criteria "
                "into your analysis with specific numbers."
            ),
            "jhunjhunwala_analysis": (
                f"The user is asking for a Rakesh Jhunjhunwala-style GARP (Growth at "
                f"Reasonable Price) analysis{' of ' + symbol if symbol else ''}. "
                "Discuss growth acceleration (revenue and earnings CAGR), turnaround "
                "potential (improving margins, earnings surprises), multibagger characteristics "
                "(operating leverage, high growth rate), and value-for-growth (PEG ratio, "
                "forward PE relative to growth). Reference Jhunjhunwala's philosophy: "
                "buy growth stocks before the market recognizes them, hold conviction "
                "picks for the long term, and identify turnaround stories. "
                "If Jhunjhunwala score data is provided, incorporate the scores into analysis."
            ),
            "fundamental_analysis": (
                f"The user is asking for a comprehensive fundamental analysis"
                f"{' of ' + symbol if symbol else ''}. "
                "Cover key areas: valuation (PE, PB, PEG, EV/EBITDA), profitability "
                "(ROE, ROA, margins), financial health (debt/equity, current ratio, "
                "interest coverage), growth (revenue and EPS CAGR), and cash flow quality "
                "(FCF yield, FCF-to-net-income). If both Buffett and Jhunjhunwala scores "
                "are provided, compare the two perspectives and explain what each philosophy "
                "would say about the stock."
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

        # Inject client profile from persistent data + extracted entities
        _profile_intents = {
            "retirement_planning", "market_advice", "income_strategy",
            "financial_planning", "risk_assessment", "beginner_guidance",
            "portfolio_management",
        }
        if intent in _profile_intents:
            profile_lines = []

            # Persistent profile data (from database)
            if investor_profile:
                name = investor_profile.get('first_name', '')
                if name:
                    profile_lines.append(f"- Client name: {name}")
                if investor_profile.get('occupation'):
                    profile_lines.append(f"- Occupation: {investor_profile['occupation']}")
                if investor_profile.get('investment_experience'):
                    profile_lines.append(f"- Investment experience: {investor_profile['investment_experience']}")
                if investor_profile.get('investment_goal'):
                    profile_lines.append(f"- Primary investment goal: {investor_profile['investment_goal']}")

            # Age (calculated from DOB, injected into entities by main_enhanced.py)
            if entities.get("user_age"):
                age = entities["user_age"]
                profile_lines.append(f"- Age: {age} years old")
                if age < 30:
                    profile_lines.append("  (Young investor — long time horizon, can take more risk)")
                elif age < 40:
                    profile_lines.append("  (Early-career — growth-focused with some diversification)")
                elif age < 50:
                    profile_lines.append("  (Mid-career — balanced approach, start thinking about retirement)")
                elif age < 60:
                    profile_lines.append("  (Pre-retirement — shift toward income and capital preservation)")
                else:
                    profile_lines.append("  (Near/in retirement — focus on income, preservation, and cash buffer)")

            # NLP-extracted or profile-merged entities
            if entities.get("amount"):
                profile_lines.append(f"- Investment amount: {entities['amount']}")
            if entities.get("risk_level"):
                profile_lines.append(f"- Risk tolerance: {entities['risk_level']}")
            if entities.get("horizon_months"):
                months = entities["horizon_months"]
                years = months / 12
                profile_lines.append(
                    f"- Investment horizon: {months} months ({years:.1f} years)"
                )
            if entities.get("market"):
                profile_lines.append(f"- Market preference: {entities['market']}")
            if entities.get("currency"):
                profile_lines.append(f"- Currency: {entities['currency']}")

            if profile_lines:
                prompt += "\n## Client Profile (known about this investor):\n"
                prompt += "\n".join(profile_lines) + "\n"
                prompt += (
                    "Use ALL of these details to personalize your response. "
                    "Address the client by name if known. "
                    "Tailor allocations to their age, risk tolerance, and goals.\n\n"
                )
            else:
                prompt += (
                    "\n## Client Profile: No profile data available. Ask the user to set up "
                    "their investor profile for personalized advice, or ask 2-3 clarifying "
                    "questions (age, amount, risk tolerance, timeline).\n\n"
                )

        # Inject recent chat history for conversational context
        if chat_history:
            prompt += "## Recent Conversation (for context — the user may have shared personal details earlier):\n"
            for msg in chat_history:
                user_msg = msg.get("message", "")
                assistant_msg = msg.get("response", "")
                if user_msg:
                    prompt += f"User: {user_msg[:200]}\n"
                if assistant_msg:
                    prompt += f"Assistant: {assistant_msg[:200]}\n"
            prompt += "\nUse any relevant context from the conversation above (age, goals, preferences) to personalize your response.\n\n"

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

    def _retirement_template(self, entities: Dict[str, Any], investor_profile: Optional[Dict[str, Any]] = None) -> str:
        """Generate personalized retirement planning template."""
        risk = entities.get("risk_level")
        horizon = entities.get("horizon_months")
        amount = entities.get("amount")
        age = entities.get("user_age")
        name = investor_profile.get("first_name", "") if investor_profile else ""
        goal = investor_profile.get("investment_goal", "") if investor_profile else ""
        experience = investor_profile.get("investment_experience", "") if investor_profile else ""
        occupation = investor_profile.get("occupation", "") if investor_profile else ""

        # Derive sensible defaults from age when explicit values are missing
        if age and not risk:
            if age >= 55:
                risk = "conservative"
            elif age >= 40:
                risk = "moderate"
            else:
                risk = "aggressive" if experience == "advanced" else "moderate"

        if age and not horizon:
            years_to_retire = max(1, 65 - age)
            horizon = years_to_retire * 12

        # If no personal details at all, prompt for them
        if not risk and not horizon and not amount and not age:
            return (
                "**Retirement Investment Planning**\n\n"
                "I'd like to give you a personalized retirement plan. "
                "To tailor my recommendations, you can either:\n\n"
                "**Option 1:** Set up your investor profile (Settings > Investor Profile) "
                "with your date of birth, risk tolerance, and investment goal for "
                "automatic personalization on every query.\n\n"
                "**Option 2:** Tell me in this chat:\n"
                "1. **Your age** or approximate birth year\n"
                "2. **When you'd like to retire** (e.g., in 10 years, at age 65)\n"
                "3. **How much** you're planning to invest\n"
                "4. **Risk comfort**: Conservative (protect capital), Moderate (balanced growth), "
                "or Aggressive (maximize growth)\n\n"
                "In the meantime, here's a general framework:\n\n"
                "| Age Bracket | Stocks | Bonds | Cash | Key Shift |\n"
                "| --- | --- | --- | --- | --- |\n"
                "| Under 40 | 80-90% | 10-15% | 0-5% | Maximize growth |\n"
                "| 40-50 | 70-80% | 15-25% | 5% | Start diversifying |\n"
                "| 50-55 | 60-70% | 25-30% | 5-10% | Shift to dividend/value |\n"
                "| 55-60 | 50-60% | 30-35% | 10-15% | Build cash buffer |\n"
                "| 60-65 | 40-50% | 35-40% | 10-20% | De-risk, income focus |\n"
                "| 65+ | 30-40% | 40-50% | 15-25% | Preservation + income |\n\n"
                "**Example:** *'I'm 55, want to retire in 10 years, have $100,000 to invest, moderate risk'*\n\n"
                "Note: This is not financial advice."
            )

        # Personalized response based on available details
        risk_label = (risk or "moderate").capitalize()
        horizon_years = (horizon or 120) / 12

        # Select allocation based on risk and horizon
        if risk == "conservative" or horizon_years <= 5:
            alloc_table = (
                "| Asset Type | Allocation | Example ETFs | Role |\n"
                "| --- | --- | --- | --- |\n"
                "| U.S. Stock Index | 35% | VOO (S&P 500), VTI | Core equity |\n"
                "| Dividend/Value | 15% | SCHD, VYM | Income + stability |\n"
                "| International | 10% | VXUS | Diversification |\n"
                "| Bonds (Broad) | 25% | BND, AGG | Stability |\n"
                "| Inflation-Protected | 5% | TIP | Inflation hedge |\n"
                "| Cash/Short-Term | 10% | SHY, Money Market | Liquidity buffer |\n"
            )
        elif risk == "aggressive" and horizon_years >= 15:
            alloc_table = (
                "| Asset Type | Allocation | Example ETFs | Role |\n"
                "| --- | --- | --- | --- |\n"
                "| U.S. Stock Index | 50% | VOO, VTI | Core growth |\n"
                "| Growth/Tech | 20% | QQQ, VGT | High growth |\n"
                "| International | 15% | VXUS, VWO | Global diversification |\n"
                "| Dividend | 10% | SCHD | Income + stability |\n"
                "| Bonds | 5% | BND | Minimal stability anchor |\n"
            )
        else:  # moderate
            alloc_table = (
                "| Asset Type | Allocation | Example ETFs | Role |\n"
                "| --- | --- | --- | --- |\n"
                "| U.S. Stock Index | 40% | VOO (S&P 500), VTI | Core equity |\n"
                "| Dividend/Value | 15% | SCHD, VYM | Income + lower volatility |\n"
                "| International | 15% | VXUS | Global diversification |\n"
                "| Bonds (Broad) | 20% | BND, AGG | Stability |\n"
                "| REITs | 5% | VNQ | Real estate income |\n"
                "| Cash/Short-Term | 5% | SHY, Money Market | Liquidity |\n"
            )

        amount_section = ""
        if amount:
            amount_val = float(amount)
            rows = []
            if risk == "conservative" or horizon_years <= 5:
                splits = [("VOO", 0.35), ("SCHD", 0.15), ("VXUS", 0.10), ("BND", 0.25), ("TIP", 0.05), ("SHY", 0.10)]
            elif risk == "aggressive" and horizon_years >= 15:
                splits = [("VOO", 0.50), ("QQQ", 0.20), ("VXUS", 0.15), ("SCHD", 0.10), ("BND", 0.05)]
            else:
                splits = [("VOO", 0.40), ("SCHD", 0.15), ("VXUS", 0.15), ("BND", 0.20), ("VNQ", 0.05), ("SHY", 0.05)]
            for ticker, pct in splits:
                rows.append(f"| {ticker} | ${amount_val * pct:,.0f} |")
            amount_section = (
                f"\n**Suggested Dollar Allocation (${amount_val:,.0f}):**\n\n"
                "| ETF | Amount |\n| --- | --- |\n"
                + "\n".join(rows) + "\n"
            )

        # Build personalized header
        greeting = f"**{name}'s " if name else "**Your "
        profile_parts = [f"{risk_label} risk"]
        if age:
            profile_parts.append(f"Age {age}")
        profile_parts.append(f"{horizon_years:.0f}-year horizon")
        if amount:
            profile_parts.append(f"${float(amount):,.0f} to invest")
        if goal:
            profile_parts.append(f"Goal: {goal}")
        profile_str = " | ".join(profile_parts)

        # Age-specific strategy notes
        age_strategies = ""
        if age:
            if age >= 50 and age < 60:
                age_strategies = (
                    f"**Age-Specific Guidance (Age {age}, pre-retirement):**\n"
                    "- Focus on **steady growth** while avoiding large losses close to retirement\n"
                    "- Shift toward **dividend-paying equities** (SCHD, VYM) for income + lower volatility\n"
                    "- Keep retirement money **separate** from experimental or speculative trading\n"
                    "- Begin building a **2-3 year cash buffer** for sequence-of-returns risk protection\n\n"
                )
            elif age >= 60:
                age_strategies = (
                    f"**Age-Specific Guidance (Age {age}, near/in retirement):**\n"
                    "- Prioritize **capital preservation** and **income generation**\n"
                    "- Maintain a **3+ year cash buffer** to avoid selling during downturns\n"
                    "- Consider **bond laddering** for predictable income\n"
                    "- Keep some equity exposure (30-40%) to outpace inflation over a 20-30 year retirement\n\n"
                )
            elif age >= 40:
                age_strategies = (
                    f"**Age-Specific Guidance (Age {age}, mid-career):**\n"
                    "- **Balanced approach** — still enough time for growth, but start diversifying\n"
                    "- Maximize **tax-advantaged accounts** (401k, IRA, Roth) before taxable investing\n"
                    "- Start adding **dividend/value** positions alongside growth for stability\n\n"
                )

        occupation_note = ""
        if occupation:
            occupation_lower = occupation.lower()
            if any(t in occupation_lower for t in ["tech", "software", "engineer", "developer", "it "]):
                occupation_note = (
                    "**Note on sector exposure:** As a technology professional, you may have significant "
                    "tech exposure through employer equity (RSUs, stock options). Consider **underweighting** "
                    "tech ETFs (QQQ, VGT) in your retirement portfolio to avoid concentration risk.\n\n"
                )

        return (
            f"{greeting}Retirement Investment Plan**\n\n"
            f"**Profile:** {profile_str}\n\n"
            f"{age_strategies}"
            f"{occupation_note}"
            f"**Recommended Allocation:**\n\n{alloc_table}"
            f"{amount_section}\n"
            "**Key Strategies:**\n"
            "- **Glide path:** Gradually shift 2-3% per year from stocks to bonds as retirement nears\n"
            "- **Cash buffer:** Build 2-3 years of living expenses in cash/short-term bonds before retiring\n"
            "- **Tax placement:** Hold bonds in tax-deferred accounts (401k/IRA), equities in taxable accounts\n"
            "- **Rebalance** annually to maintain target allocation\n\n"
            "Note: This is not financial advice. Consider consulting a fee-only financial advisor."
        )

    def _market_advice_template(self, entities: Dict[str, Any], investor_profile: Optional[Dict[str, Any]] = None) -> str:
        """Generate personalized market advice template."""
        risk = entities.get("risk_level")
        amount = entities.get("amount")
        market = entities.get("market")
        age = entities.get("user_age")
        name = investor_profile.get("first_name", "") if investor_profile else ""

        # Derive risk from age if available but risk not specified
        if not risk and age:
            if age >= 55:
                risk = "conservative"
            elif age >= 40:
                risk = "moderate"
            else:
                risk = "moderate"

        if not risk and not amount and not age:
            return (
                "**Investment Advisory**\n\n"
                "To give you the best recommendations, I'd like to understand your situation:\n\n"
                "1. **How much** are you looking to invest?\n"
                "2. **Risk tolerance**: Conservative, Moderate, or Aggressive?\n"
                "3. **Time horizon**: Short-term (<1 year), Medium (1-5 years), or Long-term (5+ years)?\n"
                "4. **Goal**: Growth, income, retirement, or wealth preservation?\n\n"
                "**Quick-Start Options by Risk Level:**\n\n"
                "| Risk Level | Core Holdings | Expected Return | Max Drawdown |\n"
                "| --- | --- | --- | --- |\n"
                "| Conservative | BND 40%, VOO 30%, SCHD 20%, Cash 10% | 5-7% | -10 to -15% |\n"
                "| Moderate | VOO 40%, SCHD 20%, VXUS 15%, BND 20%, Cash 5% | 7-9% | -20 to -30% |\n"
                "| Aggressive | QQQ 35%, VOO 30%, VXUS 15%, VGT 15%, BND 5% | 9-12% | -30 to -45% |\n\n"
                "**Example:** *'Invest $10,000, moderate risk, for 5 years'*\n\n"
                "Note: This is not financial advice. Always do your own research."
            )

        # Personalized
        risk_label = (risk or "moderate").capitalize()
        if market == "india":
            if risk == "conservative":
                table = (
                    "| Asset Type | Allocation | Ticker | Role |\n| --- | --- | --- | --- |\n"
                    "| Large-Cap Index | 40% | NIFTYBEES.NSE | Core stability |\n"
                    "| Banking | 15% | HDFCBANK.BSE | Sector strength |\n"
                    "| IT Services | 15% | INFY.BSE, TCS.BSE | Export earnings |\n"
                    "| Debt/Bonds | 25% | Gilt Funds | Stability |\n"
                    "| Gold | 5% | GOLDBEES.NSE | Hedge |\n"
                )
            else:
                table = (
                    "| Asset Type | Allocation | Ticker | Role |\n| --- | --- | --- | --- |\n"
                    "| Large-Cap | 35% | RELIANCE.BSE, HDFCBANK.BSE | Core |\n"
                    "| Mid-Cap Growth | 20% | ZOMATO.BSE, PAYTM.BSE | Growth |\n"
                    "| IT Services | 20% | INFY.BSE, TCS.BSE | Quality |\n"
                    "| Nifty Index | 15% | NIFTYBEES.NSE | Diversification |\n"
                    "| Debt | 10% | Liquid Funds | Stability |\n"
                )
        else:
            if risk == "conservative":
                table = (
                    "| Asset Type | Allocation | Ticker | Role |\n| --- | --- | --- | --- |\n"
                    "| S&P 500 Index | 30% | VOO | Core equity |\n"
                    "| Dividend | 25% | SCHD, VYM | Income + stability |\n"
                    "| Bonds | 30% | BND, AGG | Stability |\n"
                    "| International | 10% | VXUS | Diversification |\n"
                    "| Cash | 5% | Money Market | Liquidity |\n"
                )
            elif risk == "aggressive":
                table = (
                    "| Asset Type | Allocation | Ticker | Role |\n| --- | --- | --- | --- |\n"
                    "| Growth/Tech | 35% | QQQ, VGT | High growth |\n"
                    "| S&P 500 | 30% | VOO | Core equity |\n"
                    "| International | 15% | VXUS, VWO | Global growth |\n"
                    "| Individual Stocks | 15% | NVDA, MSFT, AMZN | Alpha |\n"
                    "| Bonds | 5% | BND | Minimal anchor |\n"
                )
            else:
                table = (
                    "| Asset Type | Allocation | Ticker | Role |\n| --- | --- | --- | --- |\n"
                    "| S&P 500 Index | 40% | VOO, VTI | Core equity |\n"
                    "| Dividend/Value | 15% | SCHD | Income + stability |\n"
                    "| International | 15% | VXUS | Global diversification |\n"
                    "| Bonds | 20% | BND | Stability |\n"
                    "| REITs | 5% | VNQ | Real estate income |\n"
                    "| Cash | 5% | Money Market | Liquidity |\n"
                )

        amount_note = ""
        if amount:
            amount_val = float(amount)
            amount_note = f"\n**Investment Amount:** ${amount_val:,.0f}\n"

        greeting = f"**{name}'s " if name else "**Your "
        return (
            f"{greeting}Investment Plan**\n\n"
            f"**Profile:** {risk_label} risk"
            f"{f'  |  Age {age}' if age else ''}"
            f"{'  |  ' + market.upper() + ' market' if market else ''}"
            f"{amount_note}\n"
            f"**Recommended Allocation:**\n\n{table}\n"
            "**Key Principles:**\n"
            "- Diversify across asset classes and geographies\n"
            "- Rebalance quarterly or when allocations drift >5% from target\n"
            "- Keep total expense ratios under 0.20%\n"
            "- Stay invested through volatility — time in market beats timing the market\n\n"
            "Note: This is not financial advice. Always do your own research."
        )

    def _income_strategy_template(self, entities: Dict[str, Any], investor_profile: Optional[Dict[str, Any]] = None) -> str:
        """Generate personalized income strategy template."""
        amount = entities.get("amount")

        if amount:
            amt = float(amount)
            annual_income = amt * 0.035  # ~3.5% blended yield
            monthly_income = annual_income / 12
            amount_section = (
                f"\n**Projected Income on ${amt:,.0f}:**\n\n"
                "| ETF | Allocation | Amount | Yield | Annual Income |\n"
                "| --- | --- | --- | --- | --- |\n"
                f"| SCHD | 35% | ${amt*0.35:,.0f} | ~3.5% | ${amt*0.35*0.035:,.0f} |\n"
                f"| VYM | 15% | ${amt*0.15:,.0f} | ~3.0% | ${amt*0.15*0.030:,.0f} |\n"
                f"| BND | 25% | ${amt*0.25:,.0f} | ~4.5% | ${amt*0.25*0.045:,.0f} |\n"
                f"| VNQ | 15% | ${amt*0.15:,.0f} | ~4.0% | ${amt*0.15*0.040:,.0f} |\n"
                f"| JEPI | 10% | ${amt*0.10:,.0f} | ~7.0% | ${amt*0.10*0.070:,.0f} |\n\n"
                f"**Estimated Total:** ~${annual_income:,.0f}/year (~${monthly_income:,.0f}/month)\n"
            )
        else:
            amount_section = (
                "\nTo see specific dollar amounts, try: *'I want to invest $50,000 for income'*\n"
            )

        return (
            "**Income Portfolio Strategy**\n\n"
            "| Category | Allocation | Top Picks | Yield Range |\n"
            "| --- | --- | --- | --- |\n"
            "| Dividend Equity | 35% | SCHD (Schwab Dividend Equity) | 3.0-3.5% |\n"
            "| High Dividend | 15% | VYM (Vanguard High Dividend) | 2.8-3.2% |\n"
            "| Bonds | 25% | BND (Total Bond Market) | 4.0-5.0% |\n"
            "| REITs | 15% | VNQ (Vanguard Real Estate) | 3.5-4.5% |\n"
            "| Covered Call | 10% | JEPI (JPMorgan Equity Premium) | 6.5-8.0% |\n\n"
            f"{amount_section}\n"
            "**Key Considerations:**\n"
            "- **Qualified dividends** (SCHD, VYM) are taxed at lower capital gains rates\n"
            "- **JEPI** provides high yield via covered calls but caps upside growth\n"
            "- **Dividend growth** (VIG, DGRO) sacrifices current yield for rising income over time\n"
            "- **DRIP** (dividend reinvestment) compounds returns if you don't need the income yet\n\n"
            "Note: This is not financial advice."
        )

    def _financial_planning_template(self, entities: Dict[str, Any], investor_profile: Optional[Dict[str, Any]] = None) -> str:
        """Generate personalized financial planning template."""
        amount = entities.get("amount")
        horizon = entities.get("horizon_months")

        if not amount and not horizon:
            return (
                "**Financial Planning Guide**\n\n"
                "To build a personalized plan, I'd like to know:\n\n"
                "1. **Your financial goal** (retirement, home, education, financial independence)\n"
                "2. **Target amount** you need\n"
                "3. **Timeline** (when do you need the money?)\n"
                "4. **Current savings** and monthly contribution capacity\n\n"
                "**Quick Reference:**\n\n"
                "| Goal | Rule of Thumb | Key Vehicle |\n"
                "| --- | --- | --- |\n"
                "| Financial Independence | Annual expenses x 25 | 401k, IRA, Taxable |\n"
                "| Education (18 yrs) | $250/mo at 7% = ~$100K | 529 Plan |\n"
                "| Home Down Payment | 20% of target price | High-yield savings |\n"
                "| Emergency Fund | 6 months expenses | Savings account |\n\n"
                "**Example:** *'I want to save $500,000 for retirement in 15 years, can invest $2,000/month'*\n\n"
                "Note: This is not financial advice."
            )

        # Personalized
        horizon_years = (horizon or 60) / 12
        if amount:
            amt = float(amount)
            # Show growth projections
            conservative = amt * (1.05 ** horizon_years)
            moderate = amt * (1.07 ** horizon_years)
            optimistic = amt * (1.09 ** horizon_years)
            projection = (
                f"\n**Growth Projections for ${amt:,.0f} over {horizon_years:.0f} years:**\n\n"
                "| Scenario | Annual Return | Projected Value |\n"
                "| --- | --- | --- |\n"
                f"| Conservative | 5% | ${conservative:,.0f} |\n"
                f"| Moderate | 7% | ${moderate:,.0f} |\n"
                f"| Optimistic | 9% | ${optimistic:,.0f} |\n"
            )
        else:
            projection = ""

        alloc_label = "Growth-oriented" if horizon_years >= 10 else "Balanced" if horizon_years >= 5 else "Conservative"
        return (
            f"**Your Financial Plan**\n\n"
            f"**Timeline:** {horizon_years:.0f} years | **Approach:** {alloc_label}\n"
            f"{projection}\n"
            "**Recommended Allocation:**\n\n"
            + (
                "| Asset Type | Allocation | Example |\n| --- | --- | --- |\n"
                "| US Equity | 60% | VOO, VTI |\n"
                "| International | 20% | VXUS |\n"
                "| Bonds | 15% | BND |\n"
                "| Cash | 5% | Money Market |\n"
                if horizon_years >= 10 else
                "| Asset Type | Allocation | Example |\n| --- | --- | --- |\n"
                "| US Equity | 40% | VOO |\n"
                "| Dividend | 15% | SCHD |\n"
                "| International | 10% | VXUS |\n"
                "| Bonds | 25% | BND, TIP |\n"
                "| Cash | 10% | SHY, Money Market |\n"
            ) +
            "\n**Key Actions:**\n"
            "- Automate monthly contributions (dollar-cost averaging)\n"
            "- Maximize tax-advantaged accounts first (401k match, then Roth IRA, then taxable)\n"
            "- Rebalance annually\n"
            "- Review and adjust plan yearly as your situation changes\n\n"
            "Note: This is not financial advice."
        )

    def _template_response(
        self, intent: str, entities: Dict[str, Any], user_message: str,
        ml_results: Optional[Dict[str, Any]] = None,
        investor_profile: Optional[Dict[str, Any]] = None,
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
            "market_advice": self._market_advice_template(entities, investor_profile),
            "market_news": (
                "I can provide insights on current market trends and conditions. "
                "Our AI analyzes multiple data points to identify market patterns. "
                "For specific stock analysis, try asking about a particular symbol."
            ),
            "retirement_planning": self._retirement_template(entities, investor_profile),
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
            "income_strategy": self._income_strategy_template(entities, investor_profile),
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
            "financial_planning": self._financial_planning_template(entities, investor_profile),
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

        if ml_results.get('buffett_score'):
            bs = ml_results['buffett_score']
            lines.append(f"\n**Warren Buffett Score: {bs.get('overall_score', 0):.0f}/100 ({bs.get('grade', 'N/A')})**\n")
            lines.append(f"Verdict: {bs.get('verdict', 'N/A')}")
            for c in bs.get('criteria', []):
                lines.append(f"- {c.get('name', '')}: {c.get('score', 0):.0f}/100 - {c.get('detail', '')}")
            narrative = bs.get('narrative', '')
            if narrative:
                lines.append(f"\n{narrative}")

        if ml_results.get('jhunjhunwala_score'):
            js = ml_results['jhunjhunwala_score']
            lines.append(f"\n**Rakesh Jhunjhunwala (GARP) Score: {js.get('overall_score', 0):.0f}/100 ({js.get('grade', 'N/A')})**\n")
            lines.append(f"Verdict: {js.get('verdict', 'N/A')}")
            for c in js.get('criteria', []):
                lines.append(f"- {c.get('name', '')}: {c.get('score', 0):.0f}/100 - {c.get('detail', '')}")
            narrative = js.get('narrative', '')
            if narrative:
                lines.append(f"\n{narrative}")

        if not lines:
            return None

        lines.append("\n\n*Not financial advice.*")
        return '\n'.join(lines)


# Global instance
llm_provider = LLMProvider()
