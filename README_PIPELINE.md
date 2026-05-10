# AI Stock GPT - Data Pipeline with Parallel AI

This enhanced version of AI Stock GPT includes a powerful data pipeline that uses Parallel AI to gather market news, analyze sentiment, and generate sensitivity analysis for stock predictions.

## 🚀 Features

### Core Stock Analysis
- **Stock Predictions**: Get price predictions for any US stock
- **Technical Analysis**: RSI, Moving Averages, Trend Analysis
- **Universal Stock Support**: Works with any valid US stock symbol

### 🔄 Data Pipeline (NEW!)
- **Market News Gathering**: Real-time news that impacts US stock market
- **Sector Sentiment Analysis**: Bullish/Bearish analysis for all market sectors
- **Sensitivity Analysis**: Comprehensive risk and opportunity assessment
- **Automated Updates**: Pipeline runs every 6 hours automatically

## 📦 Installation

### Quick Setup
```bash
# Run the automated setup script
python setup_pipeline.py
```

### Manual Setup
```bash
# 1. Install dependencies
pip install -r requirements_pipeline.txt

# 2. Set your Parallel AI API key
export PARALLEL_API_KEY="your_api_key_here"

# 3. Create data directory
mkdir data
```

## 🔧 Configuration

### Parallel AI API Key
1. Get your API key from [Parallel AI](https://parallel.ai/)
2. Set it as an environment variable:
   ```bash
   export PARALLEL_API_KEY="your_api_key_here"
   ```
3. Or create a `.env` file:
   ```
   PARALLEL_API_KEY=your_api_key_here
   ```

## 🚀 Usage

### Starting the Application

#### Option 1: Using the startup script
```bash
./start_enhanced_backend.sh
```

#### Option 2: Manual startup
```bash
# Terminal 1: Start enhanced backend
cd backend
python enhanced_backend_with_pipeline.py

# Terminal 2: Start frontend
npm start
```

### Accessing the Application
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## 💬 Available Commands

### Stock Analysis
- `"predict AAPL"` - Get stock prediction
- `"technical analysis TSLA"` - Technical analysis
- `"price GOOGL"` - Current price and analysis

### Data Pipeline Commands
- `"run pipeline"` - Start complete market analysis
- `"market analysis"` - View latest analysis
- `"pipeline status"` - Check pipeline availability

### Help and Information
- `"what can you do"` - Show all capabilities
- `"help"` - Get help information

## 📊 Data Pipeline Details

### Pipeline Components

#### 1. Market News Gathering
- **Timeframe**: Configurable (12h, 24h, 7d, 30d)
- **Categories**:
  - Economic Indicators (GDP, inflation, employment)
  - Sector-Specific News (Tech, Healthcare, Finance)
  - Geopolitical Events
  - Company-Specific News
  - Market Sentiment

#### 2. Sector Sentiment Analysis
- **Sectors Covered**:
  - Technology
  - Healthcare
  - Financial
  - Energy
  - Consumer Discretionary
  - Consumer Staples
  - Industrials
  - Materials
  - Real Estate
  - Utilities

#### 3. Sensitivity Analysis
- **Market Sensitivity Factors**:
  - Interest rate sensitivity
  - Inflation sensitivity
  - Economic growth sensitivity
  - Geopolitical risk sensitivity
  - Sector rotation sensitivity

- **Stock-Specific Sensitivity**:
  - High-beta stocks identification
  - Defensive stocks recommendations
  - Growth vs Value preferences
  - Small-cap vs Large-cap analysis

### Pipeline Output
The pipeline generates comprehensive reports saved as JSON files in the `data/` directory:
- Market news summary
- Sector sentiment scores
- Risk factors and opportunities
- Portfolio recommendations
- Monitoring checklist

## 🔌 API Endpoints

### Core Endpoints
- `GET /` - Health check
- `GET /status` - System status with pipeline info
- `POST /chat` - Main chat interface

### Pipeline Endpoints
- `GET /market-analysis` - Get latest market analysis
- `POST /run-pipeline` - Run complete data pipeline

## 📁 File Structure

```
AIStockGpt/
├── data_pipeline.py              # Main pipeline implementation
├── enhanced_backend_with_pipeline.py  # Enhanced backend
├── setup_pipeline.py             # Setup script
├── requirements_pipeline.txt     # Dependencies
├── start_enhanced_backend.sh     # Startup script
├── data/                         # Pipeline results
├── backend/                      # Backend files
└── src/                          # Frontend files
```

## 🛠️ Development

### Adding New Features
1. **Extend the Pipeline**: Modify `data_pipeline.py`
2. **Add API Endpoints**: Update `enhanced_backend_with_pipeline.py`
3. **Update Frontend**: Modify React components in `src/`

### Testing the Pipeline
```python
from data_pipeline import StockMarketDataPipeline

# Initialize pipeline
pipeline = StockMarketDataPipeline("your_api_key")

# Run complete pipeline
results = pipeline.run_complete_pipeline(timeframe="24h")

# Run individual components
news = pipeline.gather_market_news("12h")
sentiment = pipeline.analyze_sector_sentiment(news["news_data"])
sensitivity = pipeline.generate_sensitivity_analysis(news["news_data"], sentiment["sentiment_analysis"])
```

## 🔍 Troubleshooting

### Common Issues

#### 1. Pipeline Not Available
```
⚠️  Data pipeline not available. Install parallel-web: pip install parallel-web
```
**Solution**: Install the required package
```bash
pip install parallel-web
```

#### 2. API Key Not Set
```
⚠️  PARALLEL_API_KEY not set. Data pipeline will not be available.
```
**Solution**: Set your API key
```bash
export PARALLEL_API_KEY="your_api_key_here"
```

#### 3. Pipeline Timeout
```
❌ Pipeline failed: Task timeout
```
**Solution**: 
- Check your internet connection
- Verify API key has sufficient credits
- Try shorter timeframe (12h instead of 24h)

#### 4. Import Errors
```
❌ Failed to import data pipeline
```
**Solution**: Install all requirements
```bash
pip install -r requirements_pipeline.txt
```

### Performance Optimization
- **Shorter Timeframes**: Use 12h instead of 24h for faster results
- **Cached Results**: Pipeline results are cached for 6 hours
- **Background Processing**: Pipeline runs in background to avoid blocking

## 📈 Use Cases

### 1. Stock Prediction Enhancement
Use pipeline insights to improve stock predictions:
```python
# Get market context for predictions
market_analysis = await update_market_analysis()
if market_analysis:
    # Adjust prediction based on market sentiment
    prediction = adjust_prediction_with_market_context(stock_data, market_analysis)
```

### 2. Portfolio Management
Generate portfolio recommendations based on market analysis:
```python
# Run sensitivity analysis
sensitivity = pipeline.generate_sensitivity_analysis(news_data, sentiment_data)
# Extract portfolio recommendations
recommendations = extract_portfolio_recommendations(sensitivity)
```

### 3. Risk Assessment
Identify and monitor market risks:
```python
# Get risk factors from pipeline
risk_factors = extract_risk_factors(sensitivity_analysis)
# Set up monitoring alerts
setup_risk_monitoring(risk_factors)
```

## 🔒 Security & Privacy

- **API Keys**: Stored securely in environment variables
- **Data Storage**: All pipeline results stored locally
- **No External Sharing**: Pipeline data stays on your system
- **Rate Limiting**: Built-in protection against API abuse

## 📞 Support

### Getting Help
1. Check the troubleshooting section above
2. Review API documentation at http://localhost:8000/docs
3. Check the logs for detailed error messages

### Contributing
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

This project is for educational purposes. Always do your own research before making investment decisions.

## ⚠️ Disclaimer

**Important**: All analysis and predictions are for educational purposes only. The data pipeline provides insights but should not be used as the sole basis for investment decisions. Always:

- Do your own research
- Consult with financial advisors
- Consider your risk tolerance
- Diversify your investments
- Never invest more than you can afford to lose

---

**Happy Trading! 📈**
