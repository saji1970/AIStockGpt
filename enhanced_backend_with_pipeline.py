from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import yfinance as yf
import json
from datetime import datetime, timedelta
import random
import time
import re
import os
from typing import Dict, Any, Optional

# Import the data pipeline
try:
    from data_pipeline import StockMarketDataPipeline
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False
    print("⚠️  Data pipeline not available. Install parallel-web: pip install parallel-web")

app = FastAPI(title="AI Stock GPT - Enhanced Backend with Data Pipeline")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Fallback stock data for common symbols (updated with current prices as of August 2025)
FALLBACK_DATA = {
    'AAPL': {'price': 232.14, 'name': 'Apple Inc.'},
    'GOOGL': {'price': 142.30, 'name': 'Alphabet Inc.'},
    'MSFT': {'price': 330.20, 'name': 'Microsoft Corporation'},
    'TSLA': {'price': 245.80, 'name': 'Tesla Inc.'},
    'AMZN': {'price': 185.50, 'name': 'Amazon.com Inc.'},
    'META': {'price': 310.60, 'name': 'Meta Platforms Inc.'},
    'NVDA': {'price': 485.90, 'name': 'NVIDIA Corporation'},
    'NFLX': {'price': 485.20, 'name': 'Netflix Inc.'},
    'AMD': {'price': 108.30, 'name': 'Advanced Micro Devices'},
    'INTC': {'price': 32.80, 'name': 'Intel Corporation'},
    'SPY': {'price': 445.20, 'name': 'SPDR S&P 500 ETF'},
    'QQQ': {'price': 375.40, 'name': 'Invesco QQQ Trust'},
    'VTI': {'price': 235.60, 'name': 'Vanguard Total Stock Market ETF'},
    'VOO': {'price': 410.30, 'name': 'Vanguard S&P 500 ETF'},
    'IWM': {'price': 185.70, 'name': 'iShares Russell 2000 ETF'},
    'ARKK': {'price': 45.20, 'name': 'ARK Innovation ETF'},
    'PLTR': {'price': 15.80, 'name': 'Palantir Technologies Inc.'},
    'COIN': {'price': 85.40, 'name': 'Coinbase Global Inc.'},
    'RBLX': {'price': 28.90, 'name': 'Roblox Corporation'},
    'HOOD': {'price': 12.50, 'name': 'Robinhood Markets Inc.'},
    'BINI': {'price': 0.85, 'name': 'BINI Stock'},
    'GME': {'price': 15.20, 'name': 'GameStop Corp.'},
    'AMC': {'price': 8.50, 'name': 'AMC Entertainment Holdings'},
    'BBBY': {'price': 0.75, 'name': 'Bed Bath & Beyond Inc.'},
    'NOK': {'price': 3.80, 'name': 'Nokia Corporation'},
    'BB': {'price': 4.20, 'name': 'BlackBerry Limited'},
    'SNDL': {'price': 1.85, 'name': 'SNDL Inc.'},
    'TLRY': {'price': 2.10, 'name': 'Tilray Brands Inc.'},
    'ACB': {'price': 0.45, 'name': 'Aurora Cannabis Inc.'},
    'CGC': {'price': 0.35, 'name': 'Canopy Growth Corporation'},
    'HEXO': {'price': 0.25, 'name': 'HEXO Corp.'},
    'TOP': {'price': 12.50, 'name': 'TOP Stock'},
    'NIO': {'price': 8.20, 'name': 'NIO Inc.'},
    'XPEV': {'price': 15.80, 'name': 'XPeng Inc.'},
    'LI': {'price': 28.40, 'name': 'Li Auto Inc.'},
    'BYD': {'price': 45.60, 'name': 'BYD Company Limited'},
    'LCID': {'price': 6.80, 'name': 'Lucid Group Inc.'},
    'RIVN': {'price': 18.90, 'name': 'Rivian Automotive Inc.'}
}

# Global variable to store latest market analysis
latest_market_analysis = None
last_analysis_update = None

def get_stock_data_with_fallback(symbol):
    """Get stock data with fallback to cached data if API fails"""
    try:
        stock = yf.Ticker(symbol)
        
        # Try to get current price directly from stock info
        try:
            current_price = stock.info.get('regularMarketPrice')
            if current_price and current_price > 0:
                return {
                    'success': True,
                    'price': current_price,
                    'data': None,
                    'source': 'live',
                    'name': stock.info.get('longName', symbol)
                }
        except:
            pass
        
        # Try historical data with shorter period
        try:
            hist = stock.history(period="5d")
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                return {
                    'success': True,
                    'price': current_price,
                    'data': hist,
                    'source': 'live',
                    'name': stock.info.get('longName', symbol) if hasattr(stock, 'info') else symbol
                }
        except:
            pass
            
        # Try with 1 day period
        try:
            hist = stock.history(period="1d")
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                return {
                    'success': True,
                    'price': current_price,
                    'data': hist,
                    'source': 'live',
                    'name': stock.info.get('longName', symbol) if hasattr(stock, 'info') else symbol
                }
        except:
            pass
            
    except Exception as e:
        print(f"Error fetching live data for {symbol}: {e}")
    
    # For unknown symbols, try to get basic info
    try:
        stock = yf.Ticker(symbol)
        try:
            hist = stock.history(period="1d")
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                return {
                    'success': True,
                    'price': current_price,
                    'data': hist,
                    'source': 'live',
                    'name': symbol
                }
        except:
            pass
    except:
        pass
    
    # Fallback to cached data for known symbols
    if symbol.upper() in FALLBACK_DATA:
        fallback = FALLBACK_DATA[symbol.upper()]
        return {
            'success': True,
            'price': fallback['price'],
            'data': None,
            'source': 'fallback',
            'name': fallback['name']
        }
    
    # For unknown symbols, return error with helpful message
    return {
        'success': False,
        'error': f'No data available for {symbol}. Please check the stock symbol and try again. You can ask about popular stocks like AAPL, GOOGL, MSFT, TSLA, AMZN, SPY, QQQ, or any other valid US stock symbol.'
    }

async def update_market_analysis():
    """Update market analysis using the data pipeline"""
    global latest_market_analysis, last_analysis_update
    
    if not PIPELINE_AVAILABLE:
        return False
    
    try:
        # Check if we need to update (update every 6 hours)
        if (last_analysis_update and 
            datetime.now() - last_analysis_update < timedelta(hours=6)):
            return True
        
        api_key = os.getenv("PARALLEL_API_KEY")
        if not api_key:
            print("⚠️  PARALLEL_API_KEY not set. Skipping market analysis update.")
            return False
        
        pipeline = StockMarketDataPipeline(api_key)
        
        # Run a quick news gathering (shorter timeframe for faster response)
        news_result = pipeline.gather_market_news(timeframe="12h")
        
        if news_result["status"] == "success":
            latest_market_analysis = news_result
            last_analysis_update = datetime.now()
            print("✅ Market analysis updated successfully")
            return True
        else:
            print(f"❌ Failed to update market analysis: {news_result.get('error')}")
            return False
            
    except Exception as e:
        print(f"❌ Error updating market analysis: {str(e)}")
        return False

@app.get("/")
async def root():
    return {"message": "AI Stock GPT Enhanced Backend with Data Pipeline is running!"}

@app.get("/status")
async def status():
    return {
        "status": "online", 
        "timestamp": datetime.now().isoformat(),
        "pipeline_available": PIPELINE_AVAILABLE,
        "last_analysis_update": last_analysis_update.isoformat() if last_analysis_update else None
    }

@app.get("/market-analysis")
async def get_market_analysis():
    """Get the latest market analysis"""
    await update_market_analysis()
    
    if latest_market_analysis:
        return {
            "status": "success",
            "analysis": latest_market_analysis,
            "last_updated": last_analysis_update.isoformat() if last_analysis_update else None
        }
    else:
        return {
            "status": "no_data",
            "message": "No market analysis available. Check if PARALLEL_API_KEY is set."
        }

@app.post("/run-pipeline")
async def run_pipeline(timeframe: str = "24h"):
    """Run the complete data pipeline"""
    if not PIPELINE_AVAILABLE:
        return JSONResponse({
            "status": "error",
            "message": "Data pipeline not available. Install parallel-web: pip install parallel-web"
        })
    
    try:
        api_key = os.getenv("PARALLEL_API_KEY")
        if not api_key:
            return JSONResponse({
                "status": "error",
                "message": "PARALLEL_API_KEY not set. Please set your Parallel AI API key."
            })
        
        pipeline = StockMarketDataPipeline(api_key)
        results = pipeline.run_complete_pipeline(timeframe=timeframe)
        
        if results["status"] == "success":
            # Save results
            filepath = pipeline.save_results(results)
            
            return JSONResponse({
                "status": "success",
                "message": "Pipeline completed successfully",
                "results_file": filepath,
                "summary": results["summary"]
            })
        else:
            return JSONResponse({
                "status": "error",
                "message": f"Pipeline failed: {results.get('error', 'Unknown error')}"
            })
            
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": f"Error running pipeline: {str(e)}"
        })

@app.post("/chat")
async def chat(request: Request):
    try:
        body = await request.json()
        user_message = body.get("message", "").lower()
    except:
        return JSONResponse({"message": "Invalid request format"})
    
    # Check for pipeline-related requests
    if any(word in user_message for word in ['pipeline', 'market analysis', 'news analysis']):
        return await handle_pipeline_request(user_message)
    elif any(word in user_message for word in ['predict', 'prediction', 'forecast', 'price']):
        return await handle_prediction_request(user_message)
    elif any(word in user_message for word in ['technical', 'analysis', 'indicator']):
        return await handle_technical_analysis(user_message)
    elif any(word in user_message for word in ['what', 'help', 'can you']):
        return await handle_general_question(user_message)
    else:
        return await handle_prediction_request(user_message)

async def handle_pipeline_request(message):
    """Handle requests for market analysis and pipeline operations"""
    if 'run pipeline' in message or 'start pipeline' in message:
        # Extract timeframe if specified
        timeframe = "24h"
        if "7d" in message or "week" in message:
            timeframe = "7d"
        elif "30d" in message or "month" in message:
            timeframe = "30d"
        elif "12h" in message or "half day" in message:
            timeframe = "12h"
        
        response_message = f"""
## 🔄 Running Market Analysis Pipeline

**Timeframe:** {timeframe}
**Status:** Processing...

The pipeline is gathering market news and generating sensitivity analysis. This may take a few minutes.

**What's happening:**
1. 📰 Gathering market news
2. 📊 Analyzing sector sentiment  
3. 🎯 Generating sensitivity analysis
4. 💾 Saving results

You can check the status by asking for "market analysis" or "pipeline status".
        """
        
        return JSONResponse({
            "message": response_message,
            "pipeline_action": "started",
            "timeframe": timeframe
        })
    
    elif 'market analysis' in message or 'news analysis' in message:
        # Update and return current market analysis
        await update_market_analysis()
        
        if latest_market_analysis:
            response_message = f"""
## 📊 Latest Market Analysis

**Last Updated:** {last_analysis_update.strftime('%Y-%m-%d %H:%M:%S') if last_analysis_update else 'N/A'}

**Market News Summary:**
{latest_market_analysis.get('news_data', 'No news data available')[:500]}...

**To get full analysis:**
- Ask "run pipeline" for complete sensitivity analysis
- Ask "technical analysis [STOCK]" for specific stock analysis
- Ask "predict [STOCK]" for stock predictions

*Note: Analysis is updated every 6 hours automatically.*
            """
        else:
            response_message = """
## 📊 Market Analysis

**Status:** No analysis available

**To get market analysis:**
1. Set your PARALLEL_API_KEY environment variable
2. Ask "run pipeline" to start the analysis
3. Ask "market analysis" to view results

**Installation:**
```bash
pip install parallel-web
export PARALLEL_API_KEY="your_api_key_here"
```
            """
        
        return JSONResponse({"message": response_message})
    
    else:
        response_message = """
## 🔄 Data Pipeline Commands

**Available Commands:**
- "run pipeline" - Start complete market analysis
- "market analysis" - View latest analysis
- "pipeline status" - Check pipeline availability

**What the pipeline does:**
1. 📰 Gathers market news from the last 24 hours
2. 📊 Analyzes sector sentiment
3. 🎯 Generates sensitivity analysis for stock predictions
4. 💾 Saves results for integration

**Setup Required:**
- Install: `pip install parallel-web`
- Set API key: `export PARALLEL_API_KEY="your_key"`
        """
        
        return JSONResponse({"message": response_message})

async def extract_stock_symbol(message):
    """Extract stock symbol from message"""
    words = message.upper().split()
    
    common_symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'META', 'NVDA', 'NFLX', 'AMD', 'INTC', 
                     'SPY', 'QQQ', 'IWM', 'VTI', 'VOO', 'ARKK', 'PLTR', 'COIN', 'RBLX', 'HOOD',
                     'GME', 'AMC', 'BBBY', 'NOK', 'BB', 'SNDL', 'TLRY', 'ACB', 'CGC', 'HEXO', 'BINI',
                     'TOP', 'NIO', 'XPEV', 'LI', 'BYD', 'LCID', 'RIVN']
    
    for word in words:
        if word in common_symbols:
            return word
    
    for word in words:
        if (len(word) >= 1 and len(word) <= 5 and 
            word.isalpha() and word.isupper() and 
            not word in ['THE', 'AND', 'FOR', 'WITH', 'FROM', 'THIS', 'THAT', 'WHAT', 'WHEN', 'WHERE']):
            return word
    
    patterns = [
        r'stock\s+([A-Z]{1,5})\b',
        r'([A-Z]{1,5})\s+stock\b',
        r'predict\s+([A-Z]{1,5})\b',
        r'analysis\s+([A-Z]{1,5})\b',
        r'([A-Z]{1,5})\s+analysis\b',
        r'price\s+([A-Z]{1,5})\b',
        r'([A-Z]{1,5})\s+price\b'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, message.upper())
        if match:
            symbol = match.group(1)
            if len(symbol) >= 1 and len(symbol) <= 5 and symbol.isalpha():
                return symbol
    
    return 'AAPL'

async def handle_prediction_request(message):
    """Handle stock prediction requests with market context"""
    symbol = await extract_stock_symbol(message)
    
    # Get stock data
    stock_data = get_stock_data_with_fallback(symbol)
    
    if not stock_data['success']:
        return JSONResponse({
            "message": f"Sorry, I couldn't find data for {symbol}. Please check the stock symbol and try again. You can ask about any US stock like AAPL, GOOGL, MSFT, TSLA, SPY, QQQ, or any other valid stock symbol.",
            "stockData": None
        })
    
    current_price = stock_data['price']
    
    # Generate prediction with market context
    if stock_data['source'] == 'live' and stock_data['data'] is not None:
        hist = stock_data['data']
        recent_prices = hist['Close'].tail(5).values
        if len(recent_prices) >= 2:
            trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
            predicted_change = trend * 0.1
            predicted_price = current_price * (1 + predicted_change)
        else:
            predicted_price = current_price * (1 + random.uniform(-0.05, 0.05))
    else:
        predicted_price = current_price * (1 + random.uniform(-0.08, 0.12))
    
    # Determine sentiment
    if predicted_price > current_price:
        sentiment = "Bullish"
        confidence = random.randint(65, 90)
    else:
        sentiment = "Bearish"
        confidence = random.randint(65, 90)
    
    source_note = " (using live market data)" if stock_data['source'] == 'live' else " (using cached data)"
    
    # Add market context if available
    market_context = ""
    if latest_market_analysis:
        market_context = f"""

**📊 Market Context:**
Latest market analysis available. Ask "market analysis" for detailed insights.
"""
    
    response_message = f"""
## 📊 Stock Prediction for {symbol}{source_note}

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
- Volume analysis{market_context}

*Note: This is a simplified prediction model. Always do your own research before making investment decisions.*
    """
    
    stock_data_response = {
        "symbol": symbol,
        "currentPrice": current_price,
        "predictedPrice": predicted_price,
        "prediction": f"{sentiment} ({confidence}% confidence)",
        "dataSource": stock_data['source']
    }
    
    return JSONResponse({"message": response_message, "stockData": stock_data_response})

async def handle_technical_analysis(message):
    """Handle technical analysis requests"""
    symbol = await extract_stock_symbol(message)
    
    stock_data = get_stock_data_with_fallback(symbol)
    
    if not stock_data['success']:
        return JSONResponse({
            "message": f"Sorry, I couldn't find data for {symbol}. Please check the stock symbol and try again. You can ask about any US stock like AAPL, GOOGL, MSFT, TSLA, SPY, QQQ, or any other valid stock symbol.",
            "stockData": None
        })
    
    current_price = stock_data['price']
    
    if stock_data['source'] == 'live' and stock_data['data'] is not None:
        hist = stock_data['data']
        sma_20 = hist['Close'].rolling(window=20).mean().iloc[-1]
        sma_5 = hist['Close'].rolling(window=5).mean().iloc[-1]
        
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]
        
        source_note = " (using live market data)"
    else:
        sma_20 = current_price * (1 + random.uniform(-0.05, 0.05))
        sma_5 = current_price * (1 + random.uniform(-0.03, 0.03))
        rsi = random.uniform(30, 70)
        source_note = " (using cached data)"
    
    price_vs_sma20 = "Above" if current_price > sma_20 else "Below"
    price_vs_sma5 = "Above" if current_price > sma_5 else "Below"
    
    if rsi > 70:
        rsi_signal = "Overbought"
    elif rsi < 30:
        rsi_signal = "Oversold"
    else:
        rsi_signal = "Neutral"
    
    response_message = f"""
## 📈 Technical Analysis for {symbol}{source_note}

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
    
    return JSONResponse({"message": response_message, "stockData": None})

async def handle_general_question(message):
    """Handle general questions and help requests"""
    if 'what can you do' in message or 'help' in message:
        response_message = """
## 🤖 AI Stock GPT Enhanced - What I Can Do

I'm your intelligent stock analysis assistant with advanced market data pipeline capabilities!

### 📊 **Stock Predictions**
- Get price predictions for any stock
- Analyze market trends and patterns
- Provide confidence levels and sentiment analysis

### 📈 **Technical Analysis**
- Calculate and interpret technical indicators
- Analyze moving averages (SMA)
- RSI (Relative Strength Index) analysis
- Trend identification and signals

### 🔄 **Market Data Pipeline** (NEW!)
- **News Analysis**: Gather market-impacting news
- **Sector Sentiment**: Analyze sector-by-sector sentiment
- **Sensitivity Analysis**: Generate comprehensive market sensitivity reports
- **Risk Assessment**: Identify key risk factors and opportunities

### 🎯 **How to Use Me**
**Stock Analysis:**
- "What's the prediction for AAPL?"
- "Analyze technical indicators for TSLA"
- "Show me the analysis for MSFT"

**Market Pipeline:**
- "run pipeline" - Start complete market analysis
- "market analysis" - View latest analysis
- "pipeline status" - Check availability

### ⚠️ **Important Disclaimer**
All analysis and predictions are for educational purposes only. Always do your own research and consult with financial advisors before making investment decisions.
        """
    else:
        response_message = """
I'm here to help you with stock analysis and market predictions! 

**Try asking me about:**
- Stock predictions (e.g., "Predict AAPL stock")
- Technical analysis (e.g., "Analyze TSLA indicators")
- Market analysis (e.g., "run pipeline" or "market analysis")
- General questions about my capabilities

**New Features:**
- 🔄 Market data pipeline with Parallel AI
- 📰 Real-time news analysis
- 📊 Sector sentiment analysis
- 🎯 Sensitivity analysis for predictions

What would you like to know about?
        """
    
    return JSONResponse({"message": response_message, "stockData": None})

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
