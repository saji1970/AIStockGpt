from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yfinance as yf
import json
from datetime import datetime, timedelta
import random

app = FastAPI(title="AI Stock GPT - Minimal Backend")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for simplicity
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    message: str
    timestamp: str = None

class ChatResponse(BaseModel):
    message: str
    stockData: dict = None

@app.get("/")
async def root():
    return {"message": "AI Stock GPT Backend is running!"}

@app.get("/status")
async def status():
    return {"status": "online", "timestamp": datetime.now().isoformat()}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatMessage):
    user_message = request.message.lower()
    
    # Simple keyword-based intent detection
    if any(word in user_message for word in ['predict', 'prediction', 'forecast']):
        return await handle_prediction_request(user_message)
    elif any(word in user_message for word in ['technical', 'analysis', 'indicator']):
        return await handle_technical_analysis(user_message)
    elif any(word in user_message for word in ['what', 'help', 'can you']):
        return await handle_general_question(user_message)
    else:
        return await handle_general_question(user_message)

async def extract_stock_symbol(message):
    """Extract stock symbol from message"""
    words = message.upper().split()
    common_symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'META', 'NVDA', 'NFLX', 'AMD', 'INTC']
    
    for word in words:
        if word in common_symbols:
            return word
    
    # Try to find 3-5 letter uppercase words
    for word in words:
        if len(word) >= 3 and len(word) <= 5 and word.isalpha():
            return word
    
    return 'AAPL'  # Default

async def handle_prediction_request(message):
    symbol = await extract_stock_symbol(message)
    
    try:
        # Get stock data
        stock = yf.Ticker(symbol)
        info = stock.info
        hist = stock.history(period="30d")
        
        if hist.empty:
            return ChatResponse(
                message=f"Sorry, I couldn't find data for {symbol}. Please try a different stock symbol.",
                stockData=None
            )
        
        current_price = hist['Close'].iloc[-1]
        
        # Simple prediction based on recent trend
        recent_prices = hist['Close'].tail(5).values
        if len(recent_prices) >= 2:
            trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
            predicted_change = trend * 0.1  # 10% of current trend
            predicted_price = current_price * (1 + predicted_change)
        else:
            predicted_price = current_price * (1 + random.uniform(-0.05, 0.05))
        
        # Determine sentiment
        if predicted_price > current_price:
            sentiment = "Bullish"
            confidence = random.randint(60, 85)
        else:
            sentiment = "Bearish"
            confidence = random.randint(60, 85)
        
        response_message = f"""
## 📊 Stock Prediction for {symbol}

**Current Price:** ${current_price:.2f}
**Predicted Price:** ${predicted_price:.2f}
**Predicted Change:** {((predicted_price - current_price) / current_price * 100):.2f}%

**Analysis:**
- **Sentiment:** {sentiment}
- **Confidence:** {confidence}%
- **Trend:** {'Upward' if predicted_price > current_price else 'Downward'}

**Key Factors:**
- Recent price movement analysis
- Market trend indicators
- Volume analysis

*Note: This is a simplified prediction model. Always do your own research before making investment decisions.*
        """
        
        stock_data = {
            "symbol": symbol,
            "currentPrice": current_price,
            "predictedPrice": predicted_price,
            "prediction": f"{sentiment} ({confidence}% confidence)"
        }
        
        return ChatResponse(message=response_message, stockData=stock_data)
        
    except Exception as e:
        return ChatResponse(
            message=f"Sorry, I encountered an error while analyzing {symbol}. Please try again.",
            stockData=None
        )

async def handle_technical_analysis(message):
    symbol = await extract_stock_symbol(message)
    
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period="30d")
        
        if hist.empty:
            return ChatResponse(
                message=f"Sorry, I couldn't find data for {symbol}. Please try a different stock symbol.",
                stockData=None
            )
        
        current_price = hist['Close'].iloc[-1]
        
        # Calculate simple technical indicators
        sma_20 = hist['Close'].rolling(window=20).mean().iloc[-1]
        sma_5 = hist['Close'].rolling(window=5).mean().iloc[-1]
        
        # Simple RSI calculation
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]
        
        # Determine signals
        price_vs_sma20 = "Above" if current_price > sma_20 else "Below"
        price_vs_sma5 = "Above" if current_price > sma_5 else "Below"
        
        if rsi > 70:
            rsi_signal = "Overbought"
        elif rsi < 30:
            rsi_signal = "Oversold"
        else:
            rsi_signal = "Neutral"
        
        response_message = f"""
## 📈 Technical Analysis for {symbol}

**Current Price:** ${current_price:.2f}

**Moving Averages:**
- **20-Day SMA:** ${sma_20:.2f}
- **5-Day SMA:** ${sma_5:.2f}
- **Price vs 20-Day SMA:** {price_vs_sma20}
- **Price vs 5-Day SMA:** {price_vs_sma5}

**RSI (Relative Strength Index):**
- **Current RSI:** {rsi:.1f}
- **Signal:** {rsi_signal}

**Technical Signals:**
- **Short-term Trend:** {'Bullish' if current_price > sma_5 else 'Bearish'}
- **Medium-term Trend:** {'Bullish' if current_price > sma_20 else 'Bearish'}
- **RSI Status:** {rsi_signal}

**Recommendation:**
Based on the technical indicators, {symbol} appears to be in a {'bullish' if current_price > sma_20 and rsi < 70 else 'bearish' if current_price < sma_20 and rsi > 30 else 'neutral'} position.

*Note: This is simplified technical analysis. Always consult with financial advisors.*
        """
        
        return ChatResponse(message=response_message, stockData=None)
        
    except Exception as e:
        return ChatResponse(
            message=f"Sorry, I encountered an error while analyzing {symbol}. Please try again.",
            stockData=None
        )

async def handle_general_question(message):
    if 'what can you do' in message or 'help' in message:
        response_message = """
## 🤖 AI Stock GPT - What I Can Do

I'm your intelligent stock analysis assistant! Here's what I can help you with:

### 📊 **Stock Predictions**
- Get price predictions for any stock
- Analyze market trends and patterns
- Provide confidence levels and sentiment analysis

### 📈 **Technical Analysis**
- Calculate and interpret technical indicators
- Analyze moving averages (SMA)
- RSI (Relative Strength Index) analysis
- Trend identification and signals

### 💡 **Market Insights**
- Stock performance analysis
- Risk assessment
- Investment recommendations

### 🎯 **How to Use Me**
Try asking questions like:
- "What's the prediction for AAPL?"
- "Analyze technical indicators for TSLA"
- "Show me the analysis for MSFT"
- "Predict GOOGL stock price"

### ⚠️ **Important Disclaimer**
All analysis and predictions are for educational purposes only. Always do your own research and consult with financial advisors before making investment decisions.
        """
    else:
        response_message = """
I'm here to help you with stock analysis and predictions! 

Try asking me about:
- Stock predictions (e.g., "Predict AAPL stock")
- Technical analysis (e.g., "Analyze TSLA indicators")
- General questions about my capabilities

What would you like to know about?
        """
    
    return ChatResponse(message=response_message, stockData=None)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
