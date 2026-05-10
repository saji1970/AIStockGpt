# AI Stock Prediction System using LSTM and Sensitivity Analysis

A comprehensive AI system for stock price prediction using Long Short-Term Memory (LSTM) neural networks and advanced sensitivity analysis techniques.

## 🚀 Features

- **Advanced LSTM Architecture**: Bidirectional LSTM with multiple layers, batch normalization, and dropout for robust predictions
- **Comprehensive Data Processing**: Automatic data collection, technical indicators, and feature engineering
- **Sensitivity Analysis**: Multiple analysis methods including perturbation, correlation, and market condition sensitivity
- **Technical Indicators**: 20+ technical indicators including MACD, RSI, Bollinger Bands, and more
- **Model Evaluation**: Comprehensive metrics including directional accuracy, MAPE, and R² score
- **Visualization**: Interactive plots and comprehensive reporting
- **Future Predictions**: Real-time stock price predictions with confidence metrics

## 📊 System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Collector│    │   LSTM Model    │    │Sensitivity      │
│                 │    │                 │    │Analysis         │
│ • Yahoo Finance │───▶│ • Bidirectional │───▶│ • Perturbation  │
│ • Technical     │    │ • Multi-layer   │    │ • Correlation   │
│   Indicators    │    │ • Dropout       │    │ • Market        │
│ • Feature       │    │ • Batch Norm    │    │   Conditions    │
│   Engineering   │    │ • Early Stop    │    │ • Parameter     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd AIStockGpt
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Verify installation**:
```bash
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
```

## 📈 Quick Start

### Basic Usage

```python
from main import AIStockPredictor

# Initialize the predictor
predictor = AIStockPredictor(
    symbol='AAPL',           # Stock symbol
    start_date='2020-01-01', # Data start date
    sequence_length=60       # LSTM sequence length
)

# Run complete analysis
results = predictor.run_complete_analysis()

# View results
print(f"R² Score: {results['evaluation_results']['metrics']['R2']:.4f}")
print(f"Directional Accuracy: {results['evaluation_results']['metrics']['Directional_Accuracy']:.2f}%")
```

### Advanced Usage

```python
# Custom model parameters
predictor = AIStockPredictor(symbol='TSLA', start_date='2019-01-01')

# Step-by-step analysis
predictor.collect_and_prepare_data(test_size=0.2)
predictor.build_and_train_model(
    lstm_units=[256, 128, 64],
    dropout_rate=0.3,
    learning_rate=0.0005,
    epochs=150
)
predictor.evaluate_model()
predictor.perform_sensitivity_analysis(['perturbation', 'correlation', 'gradient'])
predictor.make_predictions(days_ahead=10)
predictor.generate_comprehensive_report()
```

## 📋 Technical Indicators Included

- **Moving Averages**: SMA (20, 50), EMA (12, 26)
- **Momentum**: RSI, Stochastic Oscillator, MACD
- **Volatility**: Bollinger Bands, Price Volatility
- **Volume**: Volume SMA, Volume-based indicators
- **Price Action**: Price changes, Rolling statistics
- **Time Features**: Day of week, month, quarter, year

## 🔍 Sensitivity Analysis Methods

### 1. Perturbation Analysis
- Tests model sensitivity to feature perturbations
- Multiple perturbation levels (0.1, 0.2, 0.5, 1.0)
- Ranks features by importance

### 2. Correlation Analysis
- Pearson and Spearman correlations
- Feature-prediction relationships
- Absolute correlation strength ranking

### 3. Market Condition Sensitivity
- High/Low volatility scenarios
- Trend up/down scenarios
- Impact assessment on predictions

### 4. Parameter Sensitivity
- LSTM units optimization
- Dropout rate analysis
- Learning rate tuning

## 📊 Model Performance Metrics

- **MSE (Mean Squared Error)**: Overall prediction accuracy
- **RMSE (Root Mean Squared Error)**: Error in original price units
- **MAE (Mean Absolute Error)**: Average absolute prediction error
- **R² Score**: Model fit quality (0-1)
- **MAPE (Mean Absolute Percentage Error)**: Percentage error
- **Directional Accuracy**: Correct price direction predictions

## 📁 Project Structure

```
AIStockGpt/
├── main.py                 # Main execution script
├── data_collector.py       # Data collection and preprocessing
├── lstm_model.py          # LSTM model architecture and training
├── sensitivity_analysis.py # Sensitivity analysis methods
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── models/               # Saved model files
├── reports/              # Generated reports and plots
└── examples/             # Example scripts and notebooks
```

## 🎯 Example Results

### Model Performance
```
Model Performance Metrics:
MSE: 0.0023
RMSE: 0.0479
MAE: 0.0381
R2: 0.8947
MAPE: 2.34%
Directional_Accuracy: 67.89%
```

### Feature Importance (Top 5)
1. Close Price (0.2341)
2. RSI (0.1892)
3. MACD (0.1567)
4. Volume (0.1345)
5. Bollinger Band Position (0.1123)

### Prediction Example
```
Current Price (AAPL): $150.25
Predicted Price (5 days ahead): $152.80
Price Change: $2.55 (+1.70%)
Prediction: BULLISH 📈
```

## 🔧 Configuration Options

### Model Parameters
```python
model_params = {
    'lstm_units': [128, 64],      # LSTM layer sizes
    'dropout_rate': 0.2,          # Dropout for regularization
    'learning_rate': 0.001,       # Adam optimizer learning rate
    'epochs': 100,                # Training epochs
    'batch_size': 32,             # Batch size for training
    'sequence_length': 60         # Time steps for LSTM
}
```

### Data Parameters
```python
data_params = {
    'symbol': 'AAPL',             # Stock symbol
    'start_date': '2020-01-01',   # Data start date
    'test_size': 0.2,             # Test set proportion
    'target_column': 'Close'      # Target variable
}
```

## 📈 Advanced Features

### Custom Technical Indicators
```python
# Add custom indicators to data_collector.py
def add_custom_indicator(self, data):
    # Your custom indicator logic
    data['Custom_Indicator'] = your_calculation(data)
    return data
```

### Ensemble Models
```python
# Train multiple models and ensemble predictions
models = []
for i in range(5):
    model = LSTMModel()
    model.train(X_train, y_train)
    models.append(model)

# Ensemble prediction
predictions = [model.predict(X_test) for model in models]
ensemble_pred = np.mean(predictions, axis=0)
```

### Real-time Predictions
```python
# Load saved model and make real-time predictions
predictor = AIStockPredictor(symbol='AAPL')
predictor.model.load_model('models/AAPL_lstm_complete')

# Get latest data and predict
latest_data = predictor.data_collector.get_latest_data()
prediction = predictor.model.predict(latest_data)
```

## 🚨 Important Notes

### Risk Disclaimer
- This system is for educational and research purposes only
- Stock predictions are inherently uncertain
- Never invest based solely on AI predictions
- Always conduct thorough research and consult financial advisors

### Model Limitations
- Historical performance doesn't guarantee future results
- Market conditions change rapidly
- Models may not capture all market factors
- Regular retraining is recommended

### Best Practices
- Use multiple timeframes for analysis
- Combine with fundamental analysis
- Monitor model performance regularly
- Keep models updated with recent data

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Yahoo Finance for providing stock data
- TensorFlow/Keras for the deep learning framework
- TA-Lib for technical indicators
- The open-source community for various libraries and tools

## 📞 Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Check the documentation in the code comments
- Review the example scripts

---

**Happy Trading! 📈📉**

*Remember: Past performance does not guarantee future results. Always do your own research and consider consulting with financial professionals before making investment decisions.*
