import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from datetime import datetime

class LSTMModel:
    """
    LSTM model for stock price prediction with comprehensive evaluation and sensitivity analysis.
    """
    
    def __init__(self, sequence_length=60, n_features=None, model_name='stock_lstm'):
        """
        Initialize the LSTM model.
        
        Args:
            sequence_length (int): Number of time steps for LSTM
            n_features (int): Number of features
            model_name (str): Name for saving the model
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model_name = model_name
        self.model = None
        self.history = None
        self.scaler_X = None
        self.scaler_y = None
        self.feature_names = None
        
    def build_model(self, lstm_units=[128, 64], dropout_rate=0.2, learning_rate=0.001):
        """
        Build the LSTM model architecture.
        
        Args:
            lstm_units (list): List of LSTM units for each layer
            dropout_rate (float): Dropout rate for regularization
            learning_rate (float): Learning rate for optimizer
            
        Returns:
            tf.keras.Model: Compiled LSTM model
        """
        if self.n_features is None:
            raise ValueError("n_features must be set before building the model")
        
        model = Sequential()
        
        # First LSTM layer
        model.add(Bidirectional(
            LSTM(lstm_units[0], return_sequences=True, 
                 input_shape=(self.sequence_length, self.n_features))
        ))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        
        # Additional LSTM layers
        for units in lstm_units[1:]:
            model.add(Bidirectional(LSTM(units, return_sequences=True)))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))
        
        # Final LSTM layer
        model.add(Bidirectional(LSTM(lstm_units[-1])))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        
        # Dense layers
        model.add(Dense(64, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        
        model.add(Dense(32, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        
        # Output layer
        model.add(Dense(1, activation='linear'))
        
        # Compile the model
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs=100, batch_size=32, validation_split=0.2):
        """
        Train the LSTM model.
        
        Args:
            X_train (np.array): Training features
            y_train (np.array): Training targets
            X_val (np.array): Validation features
            y_val (np.array): Validation targets
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            validation_split (float): Validation split if no validation data provided
            
        Returns:
            dict: Training history
        """
        if self.model is None:
            self.build_model()
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                f'models/{self.model_name}_best.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Prepare validation data
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
            validation_split = None
        else:
            validation_data = None
        
        # Train the model
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Args:
            X (np.array): Input features
            
        Returns:
            np.array: Predictions
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test, scaler_y=None):
        """
        Evaluate the model performance.
        
        Args:
            X_test (np.array): Test features
            y_test (np.array): Test targets
            scaler_y: Scaler for inverse transforming predictions
            
        Returns:
            dict: Evaluation metrics
        """
        # Make predictions
        y_pred_scaled = self.predict(X_test)
        
        # Inverse transform if scaler is provided
        if scaler_y is not None:
            y_pred = scaler_y.inverse_transform(y_pred_scaled)
            y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
        else:
            y_pred = y_pred_scaled.flatten()
            y_test_original = y_test
        
        # Calculate metrics
        mse = mean_squared_error(y_test_original, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_original, y_pred)
        r2 = r2_score(y_test_original, y_pred)
        
        # Calculate MAPE
        mape = np.mean(np.abs((y_test_original - y_pred) / y_test_original)) * 100
        
        # Calculate directional accuracy
        direction_accuracy = self.calculate_directional_accuracy(y_test_original, y_pred)
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape,
            'Directional_Accuracy': direction_accuracy
        }
        
        return metrics, y_pred
    
    def calculate_directional_accuracy(self, y_true, y_pred):
        """
        Calculate directional accuracy (percentage of correct price direction predictions).
        
        Args:
            y_true (np.array): True values
            y_pred (np.array): Predicted values
            
        Returns:
            float: Directional accuracy percentage
        """
        # Calculate price changes
        true_changes = np.diff(y_true)
        pred_changes = np.diff(y_pred)
        
        # Determine directions
        true_direction = np.sign(true_changes)
        pred_direction = np.sign(pred_changes)
        
        # Calculate accuracy
        correct_directions = np.sum(true_direction == pred_direction)
        total_predictions = len(true_direction)
        
        return (correct_directions / total_predictions) * 100 if total_predictions > 0 else 0
    
    def plot_training_history(self, save_path=None):
        """
        Plot training history.
        
        Args:
            save_path (str): Path to save the plot
        """
        if self.history is None:
            print("No training history available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(self.history.history['loss'], label='Training Loss')
        ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot MAE
        ax2.plot(self.history.history['mae'], label='Training MAE')
        ax2.plot(self.history.history['val_mae'], label='Validation MAE')
        ax2.set_title('Model MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_predictions(self, y_true, y_pred, title='Stock Price Predictions', save_path=None):
        """
        Plot actual vs predicted values.
        
        Args:
            y_true (np.array): True values
            y_pred (np.array): Predicted values
            title (str): Plot title
            save_path (str): Path to save the plot
        """
        plt.figure(figsize=(12, 6))
        
        # Plot actual vs predicted
        plt.plot(y_true, label='Actual', color='blue', alpha=0.7)
        plt.plot(y_pred, label='Predicted', color='red', alpha=0.7)
        
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def save_model(self, model_path=None, scaler_X=None, scaler_y=None, feature_names=None):
        """
        Save the trained model and related components.
        
        Args:
            model_path (str): Path to save the model
            scaler_X: Feature scaler
            scaler_y: Target scaler
            feature_names (list): List of feature names
        """
        if model_path is None:
            model_path = f'models/{self.model_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Save the model
        self.model.save(f'{model_path}.h5')
        
        # Save scalers and metadata
        metadata = {
            'sequence_length': self.sequence_length,
            'n_features': self.n_features,
            'feature_names': feature_names,
            'model_name': self.model_name
        }
        
        if scaler_X is not None:
            joblib.dump(scaler_X, f'{model_path}_scaler_X.pkl')
        if scaler_y is not None:
            joblib.dump(scaler_y, f'{model_path}_scaler_y.pkl')
        
        joblib.dump(metadata, f'{model_path}_metadata.pkl')
        
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path):
        """
        Load a trained model and related components.
        
        Args:
            model_path (str): Path to the saved model
        """
        # Load the model
        self.model = load_model(f'{model_path}.h5')
        
        # Load metadata
        metadata = joblib.load(f'{model_path}_metadata.pkl')
        self.sequence_length = metadata['sequence_length']
        self.n_features = metadata['n_features']
        self.feature_names = metadata['feature_names']
        self.model_name = metadata['model_name']
        
        # Load scalers if they exist
        try:
            self.scaler_X = joblib.load(f'{model_path}_scaler_X.pkl')
            self.scaler_y = joblib.load(f'{model_path}_scaler_y.pkl')
        except FileNotFoundError:
            print("Scalers not found, loading model only")
        
        print(f"Model loaded from {model_path}")
    
    def get_feature_importance(self, X_sample, feature_names=None):
        """
        Calculate feature importance using permutation importance.
        
        Args:
            X_sample (np.array): Sample data for importance calculation
            feature_names (list): List of feature names
            
        Returns:
            dict: Feature importance scores
        """
        if self.model is None:
            raise ValueError("Model must be trained before calculating feature importance")
        
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(X_sample.shape[2])]
        
        # Get baseline prediction
        baseline_pred = self.predict(X_sample)
        
        # Calculate importance for each feature
        importance_scores = {}
        
        for i, feature_name in enumerate(feature_names):
            # Create perturbed data
            X_perturbed = X_sample.copy()
            np.random.shuffle(X_perturbed[:, :, i])
            
            # Get prediction with perturbed feature
            perturbed_pred = self.predict(X_perturbed)
            
            # Calculate importance as increase in MSE
            importance = np.mean((perturbed_pred - baseline_pred) ** 2)
            importance_scores[feature_name] = importance
        
        # Sort by importance
        importance_scores = dict(sorted(importance_scores.items(), 
                                      key=lambda x: x[1], reverse=True))
        
        return importance_scores

if __name__ == "__main__":
    # Example usage
    print("LSTM Model class created successfully!")
    print("Use this class with the StockDataCollector to train and evaluate stock prediction models.")
