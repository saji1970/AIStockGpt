import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ta
import warnings
warnings.filterwarnings('ignore')

class StockDataCollector:
    """
    A class to collect and preprocess stock data for LSTM prediction models.
    """
    
    def __init__(self, symbol='AAPL', start_date='2020-01-01', end_date=None):
        """
        Initialize the data collector.
        
        Args:
            symbol (str): Stock symbol (default: 'AAPL')
            start_date (str): Start date for data collection
            end_date (str): End date for data collection (default: today)
        """
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date if end_date else datetime.now().strftime('%Y-%m-%d')
        self.raw_data = None
        self.processed_data = None
        
    def fetch_data(self):
        """
        Fetch stock data from Yahoo Finance.
        
        Returns:
            pd.DataFrame: Raw stock data
        """
        try:
            ticker = yf.Ticker(self.symbol)
            self.raw_data = ticker.history(start=self.start_date, end=self.end_date)
            print(f"Successfully fetched {len(self.raw_data)} days of data for {self.symbol}")
            return self.raw_data
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
    
    def add_technical_indicators(self, data):
        """
        Add technical indicators to the dataset.
        
        Args:
            data (pd.DataFrame): Raw stock data
            
        Returns:
            pd.DataFrame: Data with technical indicators
        """
        df = data.copy()
        
        # Moving averages
        df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df['EMA_12'] = ta.trend.ema_indicator(df['Close'], window=12)
        df['EMA_26'] = ta.trend.ema_indicator(df['Close'], window=26)
        
        # MACD
        df['MACD'] = ta.trend.macd_diff(df['Close'])
        df['MACD_signal'] = ta.trend.macd_signal(df['Close'])
        
        # RSI
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['Close'])
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_lower'] = bb.bollinger_lband()
        df['BB_middle'] = bb.bollinger_mavg()
        
        # Stochastic Oscillator
        df['Stoch_K'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'])
        df['Stoch_D'] = ta.momentum.stoch_signal(df['High'], df['Low'], df['Close'])
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        
        # Price changes
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Change_5'] = df['Close'].pct_change(periods=5)
        
        # Volatility
        df['Volatility'] = df['Price_Change'].rolling(window=20).std()
        
        return df
    
    def create_features(self, data):
        """
        Create additional features for the model.
        
        Args:
            data (pd.DataFrame): Data with technical indicators
            
        Returns:
            pd.DataFrame: Data with additional features
        """
        df = data.copy()
        
        # Time-based features
        df['Day_of_Week'] = df.index.dayofweek
        df['Month'] = df.index.month
        df['Quarter'] = df.index.quarter
        df['Year'] = df.index.year
        
        # Lag features
        for lag in [1, 2, 3, 5, 10]:
            df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
            df[f'Volume_lag_{lag}'] = df['Volume'].shift(lag)
        
        # Rolling statistics
        for window in [5, 10, 20]:
            df[f'Close_rolling_mean_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'Close_rolling_std_{window}'] = df['Close'].rolling(window=window).std()
            df[f'Volume_rolling_mean_{window}'] = df['Volume'].rolling(window=window).mean()
        
        return df
    
    def prepare_data(self, target_column='Close', sequence_length=60, test_size=0.2):
        """
        Prepare data for LSTM model training.
        
        Args:
            target_column (str): Target variable column
            sequence_length (int): Number of time steps for LSTM
            test_size (float): Proportion of data for testing
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test, scaler, feature_names)
        """
        if self.raw_data is None:
            self.fetch_data()
        
        # Add technical indicators
        df_with_indicators = self.add_technical_indicators(self.raw_data)
        
        # Add features
        df_features = self.create_features(df_with_indicators)
        
        # Remove rows with NaN values
        df_features = df_features.dropna()
        
        # Select features for the model
        feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
            'MACD', 'MACD_signal', 'RSI',
            'BB_upper', 'BB_lower', 'BB_middle',
            'Stoch_K', 'Stoch_D', 'Volume_SMA',
            'Price_Change', 'Price_Change_5', 'Volatility',
            'Day_of_Week', 'Month', 'Quarter', 'Year'
        ]
        
        # Add lag features
        for lag in [1, 2, 3, 5, 10]:
            feature_columns.extend([f'Close_lag_{lag}', f'Volume_lag_{lag}'])
        
        # Add rolling features
        for window in [5, 10, 20]:
            feature_columns.extend([
                f'Close_rolling_mean_{window}',
                f'Close_rolling_std_{window}',
                f'Volume_rolling_mean_{window}'
            ])
        
        # Filter available columns
        available_features = [col for col in feature_columns if col in df_features.columns]
        
        # Prepare features and target
        X = df_features[available_features].values
        y = df_features[target_column].values
        
        # Split data
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale the data
        from sklearn.preprocessing import MinMaxScaler
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
        
        # Create sequences for LSTM
        X_train_seq, y_train_seq = self.create_sequences(X_train_scaled, y_train_scaled, sequence_length)
        X_test_seq, y_test_seq = self.create_sequences(X_test_scaled, y_test_scaled, sequence_length)
        
        self.processed_data = {
            'X_train': X_train_seq,
            'X_test': X_test_seq,
            'y_train': y_train_seq,
            'y_test': y_test_seq,
            'scaler_X': scaler_X,
            'scaler_y': scaler_y,
            'feature_names': available_features,
            'original_data': df_features
        }
        
        return (X_train_seq, X_test_seq, y_train_seq, y_test_seq, 
                scaler_X, scaler_y, available_features)
    
    def create_sequences(self, X, y, sequence_length):
        """
        Create sequences for LSTM model.
        
        Args:
            X (np.array): Feature array
            y (np.array): Target array
            sequence_length (int): Number of time steps
            
        Returns:
            tuple: (X_sequences, y_sequences)
        """
        X_sequences, y_sequences = [], []
        
        for i in range(sequence_length, len(X)):
            X_sequences.append(X[i-sequence_length:i])
            y_sequences.append(y[i])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def get_latest_data(self, days=60):
        """
        Get the latest data for making predictions.
        
        Args:
            days (int): Number of days to fetch
            
        Returns:
            np.array: Latest data sequence
        """
        if self.processed_data is None:
            raise ValueError("Data must be prepared first. Call prepare_data() method.")
        
        latest_data = self.processed_data['original_data'].tail(days)
        feature_names = self.processed_data['feature_names']
        
        # Select features
        X_latest = latest_data[feature_names].values
        
        # Scale the data
        X_latest_scaled = self.processed_data['scaler_X'].transform(X_latest)
        
        return X_latest_scaled.reshape(1, X_latest_scaled.shape[0], X_latest_scaled.shape[1])

if __name__ == "__main__":
    # Example usage
    collector = StockDataCollector(symbol='AAPL', start_date='2020-01-01')
    data = collector.fetch_data()
    print(f"Data shape: {data.shape}")
    print(f"Columns: {data.columns.tolist()}")
