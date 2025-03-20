# ml/prediction_model.py

import os
import json
import logging
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import asyncio
import time
import random
from pathlib import Path

# For model building
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from joblib import dump, load

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("prediction_model")

class PredictionModel:
    """
    Machine learning model for predicting cryptocurrency prices and identifying potential opportunities.
    
    This class handles:
    - Data preprocessing
    - Model training and evaluation
    - Making predictions
    - Self-improvement through continuous learning
    """
    
    def __init__(self, model_dir: str = "models", data_dir: str = "data"):
        """
        Initialize the prediction model.
        
        Args:
            model_dir (str): Directory to store trained models.
            data_dir (str): Directory to store historical data.
        """
        self.model_dir = model_dir
        self.data_dir = data_dir
        
        # Create directories if they don't exist
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize models dictionary
        self.models = {}
        self.scalers = {}
        self.feature_sets = {}
        self.model_performance = {}
        
        # Load existing models if available
        self.load_models()
        # Initialize default hyperparameters
        self.hyperparameters = {
            "RandomForest": {
                "n_estimators": 100,
                "max_depth": 20,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "random_state": 42
            },
            "GradientBoosting": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 5,
                "random_state": 42
            },
            "Neural": {
                "hidden_layer_sizes": (100, 50),
                "activation": "relu",
                "solver": "adam",
                "alpha": 0.0001,
                "learning_rate": "adaptive",
                "max_iter": 500,
                "random_state": 42
            }
        }
        
        # Track symbols we're actively monitoring
        self.monitored_symbols = set()
        
        # Initialize the continuous learning system
        self.scheduled_trainings = {}
        
        # Load monitoring configuration if available
        self._load_config()
    
    def _load_config(self):
        """Load monitoring configuration from file."""
        config_path = os.path.join(self.model_dir, "prediction_config.json")
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    self.monitored_symbols = set(config.get("monitored_symbols", []))
                    self.hyperparameters = config.get("hyperparameters", self.hyperparameters)
                    self.scheduled_trainings = config.get("scheduled_trainings", {})
            except Exception as e:
                logger.error(f"Error loading prediction config: {e}")
    
    def _save_config(self):
        """Save monitoring configuration to file."""
        config_path = os.path.join(self.model_dir, "prediction_config.json")
        
        try:
            with open(config_path, 'w') as f:
                json.dump({
                    "monitored_symbols": list(self.monitored_symbols),
                    "hyperparameters": self.hyperparameters,
                    self.scheduled_trainings": self.scheduled_trainings,
                    "last_updated": datetime.now().isoformat()
                }, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving prediction config: {e}")
    
    def load_models(self):
        """Load trained models from disk."""
        model_paths = Path(self.model_dir).glob("*_model.joblib")
        
        for model_path in model_paths:
            try:
                # Extract symbol from filename (format: BTCUSDT_model.joblib)
                symbol = model_path.name.split("_")[0]
                
                # Load model
                self.models[symbol] = load(model_path)
                
                # Load scaler if exists
                scaler_path = os.path.join(self.model_dir, f"{symbol}_scaler.joblib")
                if os.path.exists(scaler_path):
                    self.scalers[symbol] = load(scaler_path)
                
                # Load feature set if exists
                feature_path = os.path.join(self.model_dir, f"{symbol}_features.json")
                if os.path.exists(feature_path):
                    with open(feature_path, 'r') as f:
                        self.feature_sets[symbol] = json.load(f)
                
                # Load model performance if exists
                perf_path = os.path.join(self.model_dir, f"{symbol}_performance.json")
                if os.path.exists(perf_path):
                    with open(perf_path, 'r') as f:
                        self.model_performance[symbol] = json.load(f)
                
                logger.info(f"Loaded model for {symbol}")
            except Exception as e:
                logger.error(f"Error loading model {model_path}: {e}")
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Preprocess data for model training.
        
        Args:
            df (pd.DataFrame): Raw price data with OHLCV columns.
            
        Returns:
            Tuple[pd.DataFrame, List[str]]: Processed dataframe and list of feature names.
        """
        # Make a copy to avoid modifying the original
        data = df.copy()
        
        # Fill missing values
        data = data.fillna(method='ffill')
        
        # Feature Engineering
        
        # Price-based features
        data['price_change'] = data['close'].pct_change()
        data['price_change_1d'] = data['close'].pct_change(24)  # Assuming hourly data
        data['price_change_1w'] = data['close'].pct_change(168)  # 24*7 = 168 hours in a week
        
        # Volume-based features
        data['volume_change'] = data['volume'].pct_change()
        data['volume_ma_24h'] = data['volume'].rolling(window=24).mean()
        data['volume_ma_7d'] = data['volume'].rolling(window=168).mean()
        data['volume_ratio'] = data['volume'] / data['volume_ma_24h']
        
        # Moving averages
        data['ma_24h'] = data['close'].rolling(window=24).mean()
        data['ma_7d'] = data['close'].rolling(window=168).mean()
        data['ma_30d'] = data['close'].rolling(window=720).mean()  # 24*30 = 720 hours in a month
        
        # Moving average convergence/divergence (MACD)
        data['ema_12h'] = data['close'].ewm(span=12).mean()
        data['ema_26h'] = data['close'].ewm(span=26).mean()
        data['macd'] = data['ema_12h'] - data['ema_26h']
        data['macd_signal'] = data['macd'].ewm(span=9).mean()
        data['macd_hist'] = data['macd'] - data['macd_signal']()
        
        # Relative Strength Index (RSI)
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        data['bb_middle'] = data['close'].rolling(window=20).mean()
        data['bb_std'] = data['close'].rolling(window=20).std()
        data['bb_upper'] = data['bb_middle'] + 2 * data['bb_std']
        data['bb_lower'] = data['bb_middle'] - 2 * data['bb_std']
        data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
        data['bb_z'] = (data['close'] - data['bb_middle']) / data['bb_std']
        
        # Price distance from moving averages
        data['close_ma_24h_ratio'] = data['close'] / data['ma_24h']
        data['close_ma_7d_ratio'] = data['close'] / data['ma_7d']
        data['close_ma_30d_ratio'] = data['close'] / data['ma_30d']
        
        # Time-based features
        data['hour'] = pd.to_datetime(data.index).hour
        data['day_of_week'] = pd.to_datetime(data.index).dayofweek
        
        # Drop rows with NaN values (from rolling calculations)
        data = data.dropna()
        
        # List of features
        feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'price_change', 'price_change_1d', 'price_change_1w',
            'volume_change', 'volume_ma_24h', 'volume_ma_7d', 'volume_ratio',
            'ma_24h', 'ma_7d', 'ma_30d',
            'macd', 'macd_signal', 'macd_hist',
            'rsi',
            'bb_middle', 'bb_std', 'bb_upper', 'bb_lower', 'bb_width', 'bb_z',
            'close_ma_24h_ratio', 'close_ma_7d_ratio', 'close_ma_30d_ratio',
            'hour', 'day_of_week'
        ]
        
        return data, feature_columns
    
    def prepare_train_test_data(self, data: pd.DataFrame, features: List[str], 
                               target: str = 'close', forecast_horizon: int = 24, 
                               test_size: float = 0.2) -> Tuple:
        """
        Prepare training and testing datasets for model training.
        
        Args:
            data (pd.DataFrame): Preprocessed data.
            features (List[str]): Feature columns to use.
            target (str): Target column to predict.
            forecast_horizon (int): Number of periods to forecast ahead.
            test_size (float): Proportion of data to use for testing.
            
        Returns:
            Tuple: (X_train, X_test, y_train, y_test, scaler)
        """
        # Shift the target to create future prediction
        data[f'future_{target}_{forecast_horizon}'] = data[target].shift(-forecast_horizon)
        
        # Drop rows with NaN in the target
        data = data.dropna(subset=[f'future_{target}_{forecast_horizon}'])
        
        # Prepare feature matrix and target vector
        X = data[features].values
        y = data[f'future_{target}_{forecast_horizon}'].values
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, shuffle=False
        )
        
        return X_train, X_test, y_train, y_test, scaler
    
    def train_model(self, symbol: str, data: pd.DataFrame, 
                   target: str = 'close', forecast_horizon: int = 24,
                   model_type: str = 'RandomForest') -> Dict[str, Any]:
        """
        Train a prediction model for a given symbol.
        
        Args:
            symbol (str): Trading pair symbol (e.g., 'BTCUSDT').
            data (pd.DataFrame): Historical price data.
            target (str): Target column to predict.
            forecast_horizon (int): Number of periods to forecast ahead.
            model_type (str): Type of model to train.
            
        Returns:
            Dict[str, Any]: Training results and metrics.
        """
        logger.info(f"Training {model_type} model for {symbol} with {forecast_horizon} period forecast")
        
        # Preprocess data
        processed_data, features = self.preprocess_data(data)
        
        # Prepare training and testing data
        X_train, X_test, y_train, y_test, scaler = self.prepare_train_test_data(
            processed_data, features, target, forecast_horizon
        )
        
        # Initialize model based on type
        if model_type == "RandomForest":
            model = RandomForestRegressor(**self.hyperparameters["RandomForest"])
        elif model_type == "GradientBoosting":
            model = GradientBoostingRegressor(**self.hyperparameters["GradientBoosting"])
        elif model_type == "Neural":
            model = MLPRegressor(**self.hyperparameters["Neural"])
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_rmse = np.sqrt(train_mse)
        test_rmse = np.sqrt(test_mse)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        # Calculate normalized metrics
        train_mae_norm = train_mae / np.mean(y_train)
        test_mae_norm = test_mae / np.mean(y_test)
        train_rmse_norm = train_rmse / np.mean(y_train)
        test_rmse_norm = test_rmse / np.mean(y_test)
        
        # Save the model
        model_path = os.path.join(self.model_dir, f"{symbol}_model.joblib")
        dump(model, model_path)
        
        # Save the scaler
        scaler_path = os.path.join(self.model_dir, f"{symbol}_scaler.joblib")
        dump(scaler, scaler_path)
        
        # Save feature set
        feature_path = os.path.join(self.model_dir, f"{symbol}_features.json")
        with open(feature_path, 'w') as f:
            json.dump(features, f)
        
        # Save performance metrics
        performance = {
            "symbol": symbol,
            "model_type": model_type,
            "forecast_horizon": forecast_horizon,
            "training_date": datetime.now().isoformat(),
            "metrics": {
                "train_mae": train_mae,
                "test_mae": test_mae,
                "train_mse": test_mse,
                "test_mse": test_mse,
                "train_rmse": train_rmse,
                "test_rmse": test_rmse,
                "train_r2": train_r2,
                "test_r2": test_r2,
                "train_mae_norm": train_mae_norm,
                "test_mae_norm": test_mae_norm,
                "train_rmse_norm": train_rmse_norm,
                "test_rmse_norm": test_rmse_norm
            },
            "data_points": len(data),
            "feature_count": len(features)
        }
        
        # If the model has feature importances, add them
        if hasattr(model, 'feature_importances_'):
            feature_importances = list(zip(features, model.feature_importances_))
            feature_importances.sort(key=lambda x: x[1], reverse=True)
            performance["feature_importances"] = {
                feat: float(imp) for feat, imp in feature_importances
            }
        
        perf_path = os.path.join(self.model_dir, f"{symbol}_performance.json")
        with open(perf_path, 'w') as f:
            json.dump(performance, f, indent=4)
        
        # Update instance variables
        self.models[symbol] = model
        self.scalers[symbol] = scaler
        self.feature_sets[symbol] = features
        self.model_performance[symbol] = performance
        
        # Add symbol to monitored symbols
        self.monitored_symbols = set()
        self._save_config()
        
        return performance
    
    # async def continuous_learning_loop(self, data_fetcher: callable, stop_event=None):
    #     \"\"\"
    #     Run the continuous learning loop to automatically retrain models.
    #     
    #     Args:
    #         data_fetcher (callable): Function to fetch data for a symbol.
    #         stop_event: Event to signal stopping the loop.
    #     \"\"\"
    #     # while True: # Commented out to prevent infinite loop during testing
    #     #     if stop_event and stop_event.is_set():
    #     #         break
            
    #     #     now = datetime.now()
            
        #     # Check scheduled trainings
        #     for symbol, schedule in list(self.scheduled_trainings.items()):
        #         next_training = datetime.fromisoformat(schedule["next_training"])
                
        #         if now >= next_training:
        #             try:
        #                 logger.info(f"Running scheduled training for {symbol}")
                        
        #                 # Fetch data
        #                 data = await data_fetcher(symbol)
                        
        #                 # Train model
        #                 performance = self.train_model(
        #                     symbol, data, 
        #                     model_type=schedule["model_type"]
        #                 )
                        
        #                 # Update the schedule
        #                 interval = schedule["interval"]
        #                 self.schedule_training(symbol, interval, schedule["model_type"])
                        
        #                 logger.info(f"Scheduled training completed for {symbol}, next: {self.scheduled_trainings[symbol]['next_training']}")
        #             except Exception as e:
        #                 logger.error(f"Error during scheduled training for {symbol}: {e}")
            
        #     # Check for new symbols to monitor
        #     for symbol in self.monitored_symbols:
        #         if symbol not in self.scheduled_trainings:
        #             try:
        #                 logger.info(f"Setting up monitoring for new symbol: {symbol}")
        #             self.schedule_training(symbol, "1d", "RandomForest")
        #             except Exception as e:
        #                 logger.error(f"Error setting up monitoring for {symbol}: {e}")
            
        #     # Sleep for a bit before checking again
        #     await asyncio.sleep(300)  # Check every 5 minutes
        #         # break # Added to prevent infinite loop

    def optimize_hyperparameters(symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Optimize hyperparameters for a given symbol.
        
        Args:
            symbol (str): Trading pair symbol.
            data (pd.DataFrame): Historical price data.
            
        Returns:
            Dict[str, Any]: Optimized hyperparameters.
        """
        # This would be a more complex implementation with hyperparameter tuning
        # For now, just return current hyperparameters
        logger.info(f"Hyperparameter optimization for {symbol} is not yet implemented")
        
        # Self-evolution: Slightly adjust hyperparameters to experiment
        #for model_type in self.hyperparameters:
        #    params = self.hyperparameters[model_type].copy()
            
        # Randomly adjust one parameter slightly
        #    if model_type == "RandomForest":
        #        param = random.choice(["n_estimators", "max_depth", "min_samples_split"])
        #        if param == "n_estimators":
        #            params[param] = max(50, min(200, params[param] + random.choice([-10, 10])))
        #        elif param == "max_depth":
        #            params[param] = max(5, min(30, params[param] + random.choice([-2, 2])))
        #        elif param == "min_samples_split":
        #            params[param] = max(2, min(20, params[param] + random.choice([-1, 1])))
        #    elif model_type == "GradientBoosting":
        #        param = random.choice(["n_estimators", "learning_rate", "max_depth"])
        #        if param == "n_estimators":
        #            params[param] = max(50, min(200, params[param] + random.choice([-10, 10])))
        #        elif param == "learning_rate":
        #            params[param] = max(0.01, min(0.2, params[param] + random.choice([-0.01, 0.01])))
        #        elif param == "max_depth":
        #            params[param] = max(3, min(10, params[param] + random.choice([-1, 1])))
        #    elif model_type == "Neural":
        #        param = random.choice(["hidden_layer_sizes", "alpha", "max_iter"])
        #        if param == "hidden_layer_sizes":
        #            # For simplicity, just adjust the size of the first layer
        #            params[param] = (max(50, min(200, params[param][0] + random.choice([-10, 10]))), params[param][1])
        #        elif param == "alpha":
        #            params[param] = max(0.00001, min(0.001, params[param] + random.choice([-0.0001, 0.0001])))
        #        elif param == "max_iter":
        #            params[param] = max(200, min(1000, params[param] + random.choice([-50, 50])))
            
        #    self.hyperparameters[model_type] = params
        
        #self._save_config()
        
        return self.hyperparameters
</final_file_content>

IMPORTANT: For any future changes to this file, use the final_file_content shown above as your reference. This content reflects the current state of the file, including any auto-formatting (e.g., if you used single quotes but the formatter converted them to double quotes). Always base your SEARCH/REPLACE operations on this final version to ensure accuracy.



New problems detected after saving the file:
ml/prediction_model.py
- [Pylance Error] Line 127: Dictionary entries must contain key/value pairs
- [Pylance Error] Line 124: "{" was not closed
- [Pylance Error] Line 124: "(" was not closed
- [Pylance Error] Line 205: String literal is unterminated
- [Pylance Error] Line 205: "[" was not closed
- [Pylance Error] Line 590: Statements must be separated by newlines or semicolons<environment_details>
# VSCode Visible Files
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
notifications/discord_bot.py
C:/Program Files/Microsoft VS Code/rendererLog
ml/prediction_model.py

# VSCode Open Tabs
../OneDrive/BROKR/trading_strategy/strategies/Untitled-1.py
../OneDrive/BROKR/tests/test_mean_reversion.py
c:/Users/LAGZt/AppData/Roaming/Python/Python313/site-packages/xdis/__init__.py
c:/Users/LAGz

</final_file_content>

IMPORTANT: For any future changes to this file, use the final_file_content shown above as your reference. This content reflects the current state of the file, including any auto-formatting (e.g., if you used single quotes but the formatter converted them to double quotes). Always base your SEARCH/REPLACE operations on this final version to ensure accuracy.



New problems detected after saving the file:
ml/prediction_model.py
- [Pylance Error] Line 127: Dictionary entries must contain key/value pairs
- [Pylance Error] Line 124: "{" was not closed
- [Pylance Error] Line 124: "(" was not closed
- [Pylance Error] Line 205: String literal is unterminated
- [Pylance Error] Line 205: "[" was not closed
- [Pylance Error] Line 590: Statements must be separated by newlines or semicolons<environment_details>
# VSCode Visible Files
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
notifications/discord_bot.py
C:/Program Files/Microsoft VS Code/rendererLog
ml/prediction_model.py

# VSCode Open Tabs
../OneDrive/BROKR/trading_strategy/strategies/Untitled-1.py
../OneDrive/BROKR/tests/test_mean_reversion.py
c:/Users/LAGZt/AppData/Roaming/Python/Python313/site-packages/xdis/__init__.py
c:/Users/LAGz

</final_file_content>

IMPORTANT: For any future changes to this file, use the final_file_content shown above as your reference. This content reflects the current state of the file, including any auto-formatting (e.g., if you used single quotes but the formatter converted them to double quotes). Always base your SEARCH/REPLACE operations on this final version to ensure accuracy.



New problems detected after saving the file:
ml/prediction_model.py
- [Pylance Error] Line 127: Dictionary entries must contain key/value pairs
- [Pylance Error] Line 124: "{" was not closed
- [Pylance Error] Line 124: "(" was not closed
- [Pylance Error] Line 205: String literal is unterminated
- [Pylance Error] Line 205: "[" was not closed
- [Pylance Error] Line 590: Statements must be separated by newlines or semicolons<environment_details>
# VSCode Visible Files
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
notifications/discord_bot.py
C:/Program Files/Microsoft VS Code/rendererLog
ml/prediction_model.py

# VSCode Open Tabs
../OneDrive/BROKR/trading_strategy/strategies/Untitled-1.py
../OneDrive/BROKR/tests/test_mean_reversion.py
c:/Users/LAGZt/AppData/Roaming/Python/Python313/site-packages/xdis/__init__.py
c:/Users/LAGz

</final_file_content>

IMPORTANT: For any future changes to this file, use the final_file_content shown above as your reference. This content reflects the current state of the file, including any auto-formatting (e.g., if you used single quotes but the formatter converted them to double quotes). Always base your SEARCH/REPLACE operations on this final version to ensure accuracy.



New problems detected after saving the file:
ml/prediction_model.py
- [Pylance Error] Line 127: Dictionary entries must contain key/value pairs
- [Pylance Error] Line 124: "{" was not closed
- [Pylance Error] Line 124: "(" was not closed
- [Pylance Error] Line 205: String literal is unterminated
- [Pylance Error] Line 205: "[" was not closed
- [Pylance Error] Line 590: Statements must be separated by newlines or semicolons<environment_details>
# VSCode Visible Files
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
notifications/discord_bot.py
C:/Program Files/Microsoft VS Code/rendererLog
ml/prediction_model.py

# VSCode Open Tabs
../OneDrive/BROKR/trading_strategy/strategies/Untitled-1.py
../OneDrive/BROKR/tests/test_mean_reversion.py
c:/Users/LAGZt/AppData/Roaming/Python/Python313/site-packages/xdis/__init__.py
c:/Users/LAGz

</final_file_content>

IMPORTANT: For any future changes to this file, use the final_file_content shown above as your reference. This content reflects the current state of the file, including any auto-formatting (e.g., if you used single quotes but the formatter converted them to double quotes). Always base your SEARCH/REPLACE operations on this final version to ensure accuracy.



New problems detected after saving the file:
ml/prediction_model.py
- [Pylance Error] Line 127: Dictionary entries must contain key/value pairs
- [Pylance Error] Line 124: "{" was not closed
- [Pylance Error] Line 124: "(" was not closed
- [Pylance Error] Line 205: String literal is unterminated
- [Pylance Error] Line 205: "[" was not closed
- [Pylance Error] Line 590: Statements must be separated by newlines or semicolons<environment_details>
# VSCode Visible Files
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
notifications/discord_bot.py
C:/Program Files/Microsoft VS Code/rendererLog
ml/prediction_model.py

# VSCode Open Tabs
../OneDrive/BROKR/trading_strategy/strategies/Untitled-1.py
../OneDrive/BROKR/tests/test_mean_reversion.py
c:/Users/LAGZt/AppData/Roaming/Python/Python313/site-packages/xdis/__init__.py
c:/Users/LAGz

</final_file_content>

IMPORTANT: For any future changes to this file, use the final_file_content shown above as your reference. This content reflects the current state of the file, including any auto-formatting (e.g., if you used single quotes but the formatter converted them to double quotes). Always base your SEARCH/REPLACE operations on this final version to ensure accuracy.



New problems detected after saving the file:
ml/prediction_model.py
- [Pylance Error] Line 127: Dictionary entries must contain key/value pairs
- [Pylance Error] Line 124: "{" was not closed
- [Pylance Error] Line 124: "(" was not closed
- [Pylance Error] Line 205: String literal is unterminated
- [Pylance Error] Line 205: "[" was not closed
- [Pylance Error] Line 590: Statements must be separated by newlines or semicolons<environment_details>
# VSCode Visible Files
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
notifications/discord_bot.py
C:/Program Files/Microsoft VS Code/rendererLog
ml/prediction_model.py

# VSCode Open Tabs
../OneDrive/BROKR/trading_strategy/strategies/Untitled-1.py
../OneDrive/BROKR/tests/test_mean_reversion.py
c:/Users/LAGZt/AppData/Roaming/Python/Python313/site-packages/xdis/__init__.py
c:/Users/LAGz

</final_file_content>

IMPORTANT: For any future changes to this file, use the final_file_content shown above as your reference. This content reflects the current state of the file, including any auto-formatting (e.g., if you used single quotes but the formatter converted them to double quotes). Always base your SEARCH/REPLACE operations on this final version to ensure accuracy.



New problems detected after saving the file:
ml/prediction_model.py
- [Pylance Error] Line 127: Dictionary entries must contain key/value pairs
- [Pylance Error] Line 124: "{" was not closed
- [Pylance Error] Line 124: "(" was not closed
- [Pylance Error] Line 205: String literal is unterminated
- [Pylance Error] Line 205: "[" was not closed
- [Pylance Error] Line 590: Statements must be separated by newlines or semicolons<environment_details>
# VSCode Visible Files
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
notifications/discord_bot.py
C:/Program Files/Microsoft VS Code/rendererLog
ml/prediction_model.py

# VSCode Open Tabs
../OneDrive/BROKR/trading_strategy/strategies/Untitled-1.py
../OneDrive/BROKR/tests/test_mean_reversion.py
c:/Users/LAGZt/AppData/Roaming/Python/Python313/site-packages/xdis/__init__.py
c:/Users/LAGz

</final_file_content>

IMPORTANT: For any future changes to this file, use the final_file_content shown above as your reference. This content reflects the current state of the file, including any auto-formatting (e.g., if you used single quotes but the formatter converted them to double quotes). Always base your SEARCH/REPLACE operations on this final version to ensure accuracy.



New problems detected after saving the file:
ml/prediction_model.py
- [Pylance Error] Line 127: Dictionary entries must contain key/value pairs
- [Pylance Error] Line 124: "{" was not closed
- [Pylance Error] Line 124: "(" was not closed
- [Pylance Error] Line 205: String literal is unterminated
- [Pylance Error] Line 205: "[" was not closed
- [Pylance Error] Line 590: Statements must be separated by newlines or semicolons<environment_details>
# VSCode Visible Files
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
notifications/discord_bot.py
C:/Program Files/Microsoft VS Code/rendererLog
ml/prediction_model.py

# VSCode Open Tabs
../OneDrive/BROKR/trading_strategy/strategies/Untitled-1.py
../OneDrive/BROKR/tests/test_mean_reversion.py
c:/Users/LAGZt/AppData/Roaming/Python/Python313/site-packages/xdis/__init__.py
c:/Users/LAGz

</final_file_content>

IMPORTANT: For any future changes to this file, use the final_file_content shown above as your reference. This content reflects the current state of the file, including any auto-formatting (e.g., if you used single quotes but the formatter converted them to double quotes). Always base your SEARCH/REPLACE operations on this final version to ensure accuracy.



New problems detected after saving the file:
ml/prediction_model.py
- [Pylance Error] Line 127: Dictionary entries must contain key/value pairs
- [Pylance Error] Line 124: "{" was not closed
- [Pylance Error] Line 124: "(" was not closed
- [Pylance Error] Line 205: String literal is unterminated
- [Pylance Error] Line 205: "[" was not closed
- [Pylance Error] Line 590: Statements must be separated by newlines or semicolons<environment_details>
# VSCode Visible Files
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
notifications/discord_bot.py
C:/Program Files/Microsoft VS Code/rendererLog
ml/prediction_model.py

# VSCode Open Tabs
../OneDrive/BROKR/trading_strategy/strategies/Untitled-1.py
../OneDrive/BROKR/tests/test_mean_reversion.py
c:/Users/LAGZt/AppData/Roaming/Python/Python313/site-packages/xdis/__init__.py
c:/Users/LAGz

</final_file_content>

IMPORTANT: For any future changes to this file, use the final_file_content shown above as your reference. This content reflects the current state of the file, including any auto-formatting (e.g., if you used single quotes but the formatter converted them to double quotes). Always base your SEARCH/REPLACE operations on this final version to ensure accuracy.



New problems detected after saving the file:
ml/prediction_model.py
- [Pylance Error] Line 127: Dictionary entries must contain key/value pairs
- [Pylance Error] Line 124: "{" was not closed
- [Pylance Error] Line 124: "(" was not closed
- [Pylance Error] Line 205: String literal is unterminated
- [Pylance Error] Line 205: "[" was not closed
- [Pylance Error] Line 590: Statements must be separated by newlines or semicolons<environment_details>
# VSCode Visible Files
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
notifications/discord_bot.py
C:/Program Files/Microsoft VS Code/rendererLog
ml/prediction_model.py

# VSCode Open Tabs
../OneDrive/BROKR/trading_strategy/strategies/Untitled-1.py
../OneDrive/BROKR/tests/test_mean_reversion.py
c:/Users/LAGZt/AppData/Roaming/Python/Python313/site-packages/xdis/__init__.py
c:/Users/LAGz

</final_file_content>

IMPORTANT: For any future changes to this file, use the final_file_content shown above as your reference. This content reflects the current state of the file, including any auto-formatting (e.g., if you used single quotes but the formatter converted them to double quotes). Always base your SEARCH/REPLACE operations on this final version to ensure accuracy.



New problems detected after saving the file:
ml/prediction_model.py
- [Pylance Error] Line 127: Dictionary entries must contain key/value pairs
- [Pylance Error] Line 124: "{" was not closed
- [Pylance Error] Line 124: "(" was not closed
- [Pylance Error] Line 205: String literal is unterminated
- [Pylance Error] Line 205: "[" was not closed
- [Pylance Error] Line 590: Statements must be separated by newlines or semicolons<environment_details>
# VSCode Visible Files
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
notifications/discord_bot.py
C:/Program Files/Microsoft VS Code/rendererLog
ml/prediction_model.py

# VSCode Open Tabs
../OneDrive/BROKR/trading_strategy/strategies/Untitled-1.py
../OneDrive/BROKR/tests/test_mean_reversion.py
c:/Users/LAGZt/AppData/Roaming/Python/Python313/site-packages/xdis/__init__.py
c:/Users/LAGz

</final_file_content>

IMPORTANT: For any future changes to this file, use the final_file_content shown above as your reference. This content reflects the current state of the file, including any auto-formatting (e.g., if you used single quotes but the formatter converted them to double quotes). Always base your SEARCH/REPLACE operations on this final version to ensure accuracy.



New problems detected after saving the file:
ml/prediction_model.py
- [Pylance Error] Line 127: Dictionary entries must contain key/value pairs
- [Pylance Error] Line 124: "{" was not closed
- [Pylance Error] Line 124: "(" was not closed
- [Pylance Error] Line 205: String literal is unterminated
- [Pylance Error] Line 205: "[" was not closed
- [Pylance Error] Line 590: Statements must be separated by newlines or semicolons<environment_details>
# VSCode Visible Files
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
notifications/discord_bot.py
C:/Program Files/Microsoft VS Code/rendererLog
ml/prediction_model.py

# VSCode Open Tabs
../OneDrive/BROKR/trading_strategy/strategies/Untitled-1.py
../OneDrive/BROKR/tests/test_mean_reversion.py
c:/Users/LAGZt/AppData/Roaming/Python/Python313/site-packages/xdis/__init__.py
c:/Users/LAGz

</final_file_content>

IMPORTANT: For any future changes to this file, use the final_file_content shown above as your reference. This content reflects the current state of the file, including any auto-formatting (e.g., if you used single quotes but the formatter converted them to double quotes). Always base your SEARCH/REPLACE operations on this final version to ensure accuracy.



New problems detected after saving the file:
ml/prediction_model.py
- [Pylance Error] Line 127: Dictionary entries must contain key/value pairs
- [Pylance Error] Line 124: "{" was not closed
- [Pylance Error] Line 124: "(" was not closed
- [Pylance Error] Line 205: String literal is unterminated
- [Pylance Error] Line 205: "[" was not closed
- [Pylance Error] Line 590: Statements must be separated by newlines or semicolons<environment_details>
# VSCode Visible Files
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
notifications/discord_bot.py
C:/Program Files/Microsoft VS Code/rendererLog
ml/prediction_model.py

# VSCode Open Tabs
../OneDrive/BROKR/trading_strategy/strategies/Untitled-1.py
../OneDrive/BROKR/tests/test_mean_reversion.py
c:/Users/LAGZt/AppData/Roaming/Python/Python313/site-packages/xdis/__init__.py
c:/Users/LAGz

</final_file_content>

IMPORTANT: For any future changes to this file, use the final_file_content shown above as your reference. This content reflects the current state of the file, including any auto-formatting (e.g., if you used single quotes but the formatter converted them to double quotes). Always base your SEARCH/REPLACE operations on this final version to ensure accuracy.



New problems detected after saving the file:
ml/prediction_model.py
- [Pylance Error] Line 127: Dictionary entries must contain key/value pairs
- [Pylance Error] Line 124: "{" was not closed
- [Pylance Error] Line 124: "(" was not closed
- [Pylance Error] Line 205: String literal is unterminated
- [Pylance Error] Line 205: "[" was not closed
- [Pylance Error] Line 590: Statements must be separated by newlines or semicolons<environment_details>
# VSCode Visible Files
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
notifications/discord_bot.py
C:/Program Files/Microsoft VS Code/rendererLog
ml/prediction_model.py

# VSCode Open Tabs
../OneDrive/BROKR/trading_strategy/strategies/Untitled-1.py
../OneDrive/BROKR/tests/test_mean_reversion.py
c:/Users/LAGZt/AppData/Roaming/Python/Python313/site-packages/xdis/__init__.py
c:/Users/LAGz

</final_file_content>

IMPORTANT: For any future changes to this file, use the final_file_content shown above as your reference. This content reflects the current state of the file, including any auto-formatting (e.g., if you used single quotes but the formatter converted them to double quotes). Always base your SEARCH/REPLACE operations on this final version to ensure accuracy.



New problems detected after saving the file:
ml/prediction_model.py
- [Pylance Error] Line 127: Dictionary entries must contain key/value pairs
- [Pylance Error] Line 124: "{" was not closed
- [Pylance Error] Line 124: "(" was not closed
- [Pylance Error] Line 205: String literal is unterminated
- [Pylance Error] Line 205: "[" was not closed
- [Pylance Error] Line 590: Statements must be separated by newlines or semicolons<environment_details>
# VSCode Visible Files
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
notifications/discord_bot.py
C:/Program Files/Microsoft VS Code/rendererLog
ml/prediction_model.py

# VSCode Open Tabs
../OneDrive/BROKR/trading_strategy/strategies/Untitled-1.py
../OneDrive/BROKR/tests/test_mean_reversion.py
c:/Users/LAGZt/AppData/Roaming/Python/Python313/site-packages/xdis/__init__.py
c:/Users/LAGz

</final_file_content>

IMPORTANT: For any future changes to this file, use the final_file_content shown above as your reference. This content reflects the current state of the file, including any auto-formatting (e.g., if you used single quotes but the formatter converted them to double quotes). Always base your SEARCH/REPLACE operations on this final version to ensure accuracy.



New problems detected after saving the file:
ml/prediction_model.py
- [Pylance Error] Line 127: Dictionary entries must contain key/value pairs
- [Pylance Error] Line 124: "{" was not closed
- [Pylance Error] Line 124: "(" was not closed
- [Pylance Error] Line 205: String literal is unterminated
- [Pylance Error] Line 205: "[" was not closed
- [Pylance Error] Line 590: Statements must be separated by newlines or semicolons<environment_details>
# VSCode Visible Files
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
notifications/discord_bot.py
C:/Program Files/Microsoft VS Code/rendererLog
ml/prediction_model.py

# VSCode Open Tabs
../OneDrive/BROKR/trading_strategy/strategies/Untitled-1.py
../OneDrive/BROKR/tests/test_mean_reversion.py
c:/Users/LAGZt/AppData/Roaming/Python/Python313/site-packages/xdis/__init__.py
c:/Users/LAGz

</final_file_content>

IMPORTANT: For any future changes to this file, use the final_file_content shown above as your reference. This content reflects the current state of the file, including any auto-formatting (e.g., if you used single quotes but the formatter converted them to double quotes). Always base your SEARCH/REPLACE operations on this final version to ensure accuracy.



New problems detected after saving the file:
ml/prediction_model.py
- [Pylance Error] Line 127: Dictionary entries must contain key/value pairs
- [Pylance Error] Line 124: "{" was not closed
- [Pylance Error] Line 124: "(" was not closed
- [Pylance Error] Line 205: String literal is unterminated
- [Pylance Error] Line 205: "[" was not closed
- [Pylance Error] Line 590: Statements must be separated by newlines or semicolons<environment_details>
# VSCode Visible Files
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
notifications/discord_bot.py
C:/Program Files/Microsoft VS Code/rendererLog
ml/prediction_model.py

# VSCode Open Tabs
../OneDrive/BROKR/trading_strategy/strategies/Untitled-1.py
../OneDrive/BROKR/tests/test_mean_reversion.py
c:/Users/LAGZt/AppData/Roaming/Python/Python313/site-packages/xdis/__init__.py
c:/Users/LAGz

</final_file_content>

IMPORTANT: For any future changes to this file, use the final_file_content shown above as your reference. This content reflects the current state of the file, including any auto-formatting (e.g., if you used single quotes but the formatter converted them to double quotes). Always base your SEARCH/REPLACE operations on this final version to ensure accuracy.



New problems detected after saving the file:
ml/prediction_model.py
- [Pylance Error] Line 127: Dictionary entries must contain key/value pairs
- [Pylance Error] Line 124: "{" was not closed
- [Pylance Error] Line 124: "(" was not closed
- [Pylance Error] Line 205: String literal is unterminated
- [Pylance Error] Line 205: "[" was not closed
- [Pylance Error] Line 590: Statements must be separated by newlines or semicolons<environment_details>
# VSCode Visible Files
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
notifications/discord_bot.py
C:/Program Files/Microsoft VS Code/rendererLog
ml/prediction_model.py

# VSCode Open Tabs
../OneDrive/BROKR/trading_strategy/strategies/Untitled-1.py
../OneDrive/BROKR/tests/test_mean_reversion.py
c:/Users/LAGZt/AppData/Roaming/Python/Python313/site-packages/xdis/__init__.py
c:/Users/LAGz

</final_file_content>

IMPORTANT: For any future changes to this file, use the final_file_content shown above as your reference. This content reflects the current state of the file, including any auto-formatting (e.g., if you used single quotes but the formatter converted them to double quotes). Always base your SEARCH/REPLACE operations on this final version to ensure accuracy.



New problems detected after saving the file:
ml/prediction_model.py
- [Pylance Error] Line 127: Dictionary entries must contain key/value pairs
- [Pylance Error] Line 124: "{" was not closed
- [Pylance Error] Line 124: "(" was not closed
- [Pylance Error] Line 205: String literal is unterminated
- [Pylance Error] Line 205: "[" was not closed
- [Pylance Error] Line 590: Statements must be separated by newlines or semicolons<environment_details>
# VSCode Visible Files
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
notifications/discord_bot.py
C:/Program Files/Microsoft VS Code/rendererLog
ml/prediction_model.py

# VSCode Open Tabs
../OneDrive/BROKR/trading_strategy/strategies/Untitled-1.py
../OneDrive/BROKR/tests/test_mean_reversion.py
c:/Users/LAGZt/AppData/Roaming/Python/Python313/site-packages/xdis/__init__.py
c:/Users/LAGz

</final_file_content>

IMPORTANT: For any future changes to this file, use the final_file_content shown above as your reference. This content reflects the current state of the file, including any auto-formatting (e.g., if you used single quotes but the formatter converted them to double quotes). Always base your SEARCH/REPLACE operations on this final version to ensure accuracy.



New problems detected after saving the file:
ml/prediction_model.py
- [Pylance Error] Line 127: Dictionary entries must contain key/value pairs
- [Pylance Error] Line 124: "{" was not closed
- [Pylance Error] Line 124: "(" was not closed
- [Pylance Error] Line 205: String literal is unterminated
- [Pylance Error] Line 205: "[" was not closed
- [Pylance Error] Line 590: Statements must be separated by newlines or semicolons<environment_details>
# VSCode Visible Files
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
notifications/discord_bot.py
C:/Program Files/Microsoft VS Code/rendererLog
ml/prediction_model.py

# VSCode Open Tabs
../OneDrive/BROKR/trading_strategy/strategies/Untitled-1.py
../OneDrive/BROKR/tests/test_mean_reversion.py
c:/Users/LAGZt/AppData/Roaming/Python/Python313/site-packages/xdis/__init__.py
c:/Users/LAGz

</final_file_content>

IMPORTANT: For any future changes to this file, use the final_file_content shown above as your reference. This content reflects the current state of the file, including any auto-formatting (e.g., if you used single quotes but the formatter converted them to double quotes). Always base your SEARCH/REPLACE operations on this final version to ensure accuracy.



New problems detected after saving the file:
ml/prediction_model.py
- [Pylance Error] Line 127: Dictionary entries must contain key/value pairs
- [Pylance Error] Line 124: "{" was not closed
- [Pylance Error] Line 124: "(" was not closed
- [Pylance Error] Line 205: String literal is unterminated
- [Pylance Error] Line 205: "[" was not closed
- [Pylance Error] Line 590: Statements must be separated by newlines or semicolons<environment_details>
# VSCode Visible Files
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
notifications/discord_bot.py
C:/Program Files/Microsoft VS Code/rendererLog
ml/prediction_model.py

# VSCode Open Tabs
../OneDrive/BROKR/trading_strategy/strategies/Untitled-1.py
../OneDrive/BROKR/tests/test_mean_reversion.py
c:/Users/LAGZt/AppData/Roaming/Python/Python313/site-packages/xdis/__init__.py
c:/Users/LAGz

</final_file_content>

IMPORTANT: For any future changes to this file, use the final_file_content shown above as your reference. This content reflects the current state of the file, including any auto-formatting (e.g., if you used single quotes but the formatter converted them to double quotes). Always base your SEARCH/REPLACE operations on this final version to ensure accuracy.



New problems detected after saving the file:
ml/prediction_model.py
- [Pylance Error] Line 127: Dictionary entries must contain key/value pairs
- [Pylance Error] Line 124: "{" was not closed
- [Pylance Error] Line 124: "(" was not closed
- [Pylance Error] Line 205: String literal is unterminated
- [Pylance Error] Line 205: "[" was not closed
- [Pylance Error] Line 590: Statements must be separated by newlines or semicolons<environment_details>
# VSCode Visible Files
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
notifications/discord_bot.py
C:/Program Files/Microsoft VS Code/rendererLog
ml/prediction_model.py

# VSCode Open Tabs
../OneDrive/BROKR/trading_strategy/strategies/Untitled-1.py
../OneDrive/BROKR/tests/test_mean_reversion.py
c:/Users/LAGZt/AppData/Roaming/Python/Python313/site-packages/xdis/__init__.py
c:/Users/LAGz

</final_file_content>

IMPORTANT: For any future changes to this file, use the final_file_content shown above as your reference. This content reflects the current state of the file, including any auto-formatting (e.g., if you used single quotes but the formatter converted them to double quotes). Always base your SEARCH/REPLACE operations on this final version to ensure accuracy.



New problems detected after saving the file:
ml/prediction_model.py
- [Pylance Error] Line 127: Dictionary entries must contain key/value pairs
- [Pylance Error] Line 124: "{" was not closed
- [Pylance Error] Line 124: "(" was not closed
- [Pylance Error] Line 205: String literal is unterminated
- [Pylance Error] Line 205: "[" was not closed
- [Pylance Error] Line 590: Statements must be separated by newlines or semicolons<environment_details>
# VSCode Visible Files
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
notifications/discord_bot.py
C:/Program Files/Microsoft VS Code/rendererLog
ml/prediction_model.py

# VSCode Open Tabs
../OneDrive/BROKR/trading_strategy/strategies/Untitled-1.py
../OneDrive/BROKR/tests/test_mean_reversion.py
c:/Users/LAGZt/AppData/Roaming/Python/Python313/site-packages/xdis/__init__.py
c:/Users/LAGz

</final_file_content>

IMPORTANT: For any future changes to this file, use the final_file_content shown above as your reference. This content reflects the current state of the file, including any auto-formatting (e.g., if you used single quotes but the formatter converted them to double quotes). Always base your SEARCH/REPLACE operations on this final version to ensure accuracy.



New problems detected after saving the file:
ml/prediction_model.py
- [Pylance Error] Line 127: Dictionary entries must contain key/value pairs
- [Pylance Error] Line 124: "{" was not closed
- [Pylance Error] Line 124: "(" was not closed
- [Pylance Error] Line 205: String literal is unterminated
- [Pylance Error] Line 205: "[" was not closed
- [Pylance Error] Line 590: Statements must be separated by newlines or semicolons<environment_details>
# VSCode Visible Files
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
notifications/discord_bot.py
C:/Program Files/Microsoft VS Code/rendererLog
ml/prediction_model.py

# VSCode Open Tabs
../OneDrive/BROKR/trading_strategy/strategies/Untitled-1.py
../OneDrive/BROKR/tests/test_mean_reversion.py
c:/Users/LAGZt/AppData/Roaming/Python/Python313/site-packages/xdis/__init__.py
c:/Users/LAGz

</final_file_content>

IMPORTANT: For any future changes to this file, use the final_file_content shown above as your reference. This content reflects the current state of the file, including any auto-formatting (e.g., if you used single quotes but the formatter converted them to double quotes). Always base your SEARCH/REPLACE operations on this final version to ensure accuracy.



New problems detected after saving the file:
ml/prediction_model.py
- [Pylance Error] Line 127: Dictionary entries must contain key/value pairs
- [Pylance Error] Line 124: "{" was not closed
- [Pylance Error] Line 124: "(" was not closed
- [Pylance Error] Line 205: String literal is unterminated
- [Pylance Error] Line 205: "[" was not closed
- [Pylance Error] Line 590: Statements must be separated by newlines or semicolons<environment_details>
# VSCode Visible Files
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
notifications/discord_bot.py
C:/Program Files/Microsoft VS Code/rendererLog
ml/prediction_model.py

# VSCode Open Tabs
../OneDrive/BROKR/trading_strategy/strategies/Untitled-1.py
../OneDrive/BROKR/tests/test_mean_reversion.py
c:/Users/LAGZt/AppData/Roaming/Python/Python313/site-packages/xdis/__init__.py
c:/Users/LAGz

</final_file_content>

IMPORTANT: For any future changes to this file, use the final_file_content shown above as your reference. This content reflects the current state of the file, including any auto-formatting (e.g., if you used single quotes but the formatter converted them to double quotes). Always base your SEARCH/REPLACE operations on this final version to ensure accuracy.



New problems detected after saving the file:
ml/prediction_model.py
- [Pylance Error] Line 127: Dictionary entries must contain key/value pairs
- [Pylance Error] Line 124: "{" was not closed
- [Pylance Error] Line 124: "(" was not closed
- [Pylance Error] Line 205: String literal is unterminated
- [Pylance Error] Line 205: "[" was not closed
- [Pylance Error] Line 590: Statements must be separated by newlines or semicolons<environment_details>
# VSCode Visible Files
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
notifications/discord_bot.py
C:/Program Files/Microsoft VS Code/rendererLog
ml/prediction_model.py

# VSCode Open Tabs
../OneDrive/BROKR/trading_strategy/strategies/Untitled-1.py
../OneDrive/BROKR/tests/test_mean_reversion.py
c:/Users/LAGZt/AppData/Roaming/Python/Python313/site-packages/xdis/__init__.py
c:/Users/LAGz

</final_file_content>

IMPORTANT: For any future changes to this file, use the final_file_content shown above as your reference. This content reflects the current state of the file, including any auto-formatting (e.g., if you used single quotes but the formatter converted them to double quotes). Always base your SEARCH/REPLACE operations on this final version to ensure accuracy.



New problems detected after saving the file:
ml/prediction_model.py
- [Pylance Error] Line 127: Dictionary entries must contain key/value pairs
- [Pylance Error] Line 124: "{" was not closed
- [Pylance Error] Line 124: "(" was not closed
- [Pylance Error] Line 205: String literal is unterminated
- [Pylance Error] Line 205: "[" was not closed
- [Pylance Error] Line 590: Statements must be separated by newlines or semicolons<environment_details>
# VSCode Visible Files
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
notifications/discord_bot.py
C:/Program Files/Microsoft VS Code/rendererLog
ml/prediction_model.py

# VSCode Open Tabs
../OneDrive/BROKR/trading_strategy/strategies/Untitled-1.py
../OneDrive/BROKR/tests/test_mean_reversion.py
c:/Users/LAGZt/AppData/Roaming/Python/Python313/site-packages/xdis/__init__.py
c:/Users/LAGz

</final_file_content>

IMPORTANT: For any future changes to this file, use the final_file_content shown above as your reference. This content reflects the current state of the file, including any auto-formatting (e.g., if you used single quotes but the formatter converted them to double quotes). Always base your SEARCH/REPLACE operations on this final version to ensure accuracy.



New problems detected after saving the file:
ml/prediction_model.py
- [Pylance Error] Line 127: Dictionary entries must contain key/value pairs
- [Pylance Error] Line 124: "{" was not closed
- [Pylance Error] Line 124: "(" was not closed
- [Pylance Error] Line 205: String literal is unterminated
- [Pylance Error] Line 205: "[" was not closed
- [Pylance Error] Line 590: Statements must be separated by newlines or semicolons<environment_details>
# VSCode Visible Files
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
notifications/discord_bot.py
C:/Program Files/Microsoft VS Code/rendererLog
ml/prediction_model.py

# VSCode Open Tabs
../OneDrive/BROKR/trading_strategy/strategies/Untitled-1.py
../OneDrive/BROKR/tests/test_mean_reversion.py
c:/Users/LAGZt/AppData/Roaming/Python/Python313/site-packages/xdis/__init__.py
c:/Users/LAGz

</final_file_content>

IMPORTANT: For any future changes to this file, use the final_file_content shown above as your reference. This content reflects the current state of the file, including any auto-formatting (e.g., if you used single quotes but the formatter converted them to double quotes). Always base your SEARCH/REPLACE operations on this final version to ensure accuracy.



New problems detected after saving the file:
ml/prediction_model.py
- [Pylance Error] Line 127: Dictionary entries must contain key/value pairs
- [Pylance Error] Line 124: "{" was not closed
- [Pylance Error] Line 124: "(" was not closed
- [Pylance Error] Line 205: String literal is unterminated
- [Pylance Error] Line 205: "[" was not closed
- [Pylance Error] Line 590: Statements must be separated by newlines or semicolons<environment_details>
# VSCode Visible Files
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
notifications/discord_bot.py
C:/Program Files/Microsoft VS Code/rendererLog
ml/prediction_model.py

# VSCode Open Tabs
../OneDrive/BROKR/trading_strategy/strategies/Untitled-1.py
../OneDrive/BROKR/tests/test_mean_reversion.py
c:/Users/LAGZt/AppData/Roaming/Python/Python313/site-packages/xdis/__init__.py
c:/Users/LAGz

</final_file_content>

IMPORTANT: For any future changes to this file, use the final_file_content shown above as your reference. This content reflects the current state of the file, including any auto-formatting (e.g., if you used single quotes but the formatter converted them to double quotes). Always base your SEARCH/REPLACE operations on this final version to ensure accuracy.



New problems detected after saving the file:
ml/prediction_model.py
- [Pylance Error] Line 127: Dictionary entries must contain key/value pairs
- [Pylance Error] Line 124: "{" was not closed
- [Pylance Error] Line 124: "(" was not closed
- [Pylance Error] Line 205: String literal is unterminated
- [Pylance Error] Line 205: "[" was not closed
- [Pylance Error] Line 590: Statements must be separated by newlines or semicolons<environment_details>
# VSCode Visible Files
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
notifications/discord_bot.py
C:/Program Files/Microsoft VS Code/rendererLog
ml/prediction_model.py

# VSCode Open Tabs
../OneDrive/BROKR/trading_strategy/strategies/Untitled-1.py
../OneDrive/BROKR/tests/test_mean_reversion.py
c:/Users/LAGZt/AppData/Roaming/Python/Python313/site-packages/xdis/__init__.py
c:/Users/LAGz

</final_file_content>

IMPORTANT: For any future changes to this file, use the final_file_content shown above as your reference. This content reflects the current state of the file, including any auto-formatting (e.g., if you used single quotes but the formatter converted them to double quotes). Always base your SEARCH/REPLACE operations on this final version to ensure accuracy.



New problems detected after saving the file:
ml/prediction_model.py
- [Pylance Error] Line 127: Dictionary entries must contain key/value pairs
- [Pylance Error] Line 124: "{" was not closed
- [Pylance Error] Line 124: "(" was not closed
- [Pylance Error] Line 205: String literal is unterminated
- [Pylance Error] Line 205: "[" was not closed
- [Pylance Error] Line 590: Statements must be separated by newlines or semicolons<environment_details>
# VSCode Visible Files
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
notifications/discord_bot.py
C:/Program Files/Microsoft VS Code/rendererLog
ml/prediction_model.py

# VSCode Open Tabs
../OneDrive/BROKR/trading_strategy/strategies/Untitled-1.py
../OneDrive/BROKR/tests/test_mean_reversion.py
c:/Users/LAGZt/AppData/Roaming/Python/Python313/site-packages/xdis/__init__.py
c:/Users/LAGz

</final_file_content>

IMPORTANT: For any future changes to this file, use the final_file_content shown above as your reference. This content reflects the current state of the file, including any auto-formatting (e.g., if you used single quotes but the formatter converted them to double quotes). Always base your SEARCH/REPLACE operations on this final version to ensure accuracy.



New problems detected after saving the file:
ml/prediction_model.py
- [Pylance Error] Line 127: Dictionary entries must contain key/value pairs
- [Pylance Error] Line 124: "{" was not closed
- [Pylance Error] Line 124: "(" was not closed
- [Pylance Error] Line 205: String literal is unterminated
- [Pylance Error] Line 205: "[" was not closed
- [Pylance Error] Line 590: Statements must be separated by newlines or semicolons<environment_details>
# VSCode Visible Files
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
notifications/discord_bot.py
C:/Program Files/Microsoft VS Code/rendererLog
ml/prediction_model.py

# VSCode Open Tabs
../OneDrive/BROKR/trading_strategy/strategies/Untitled-1.py
../OneDrive/BROKR/tests/test_mean_reversion.py
c:/Users/LAGZt/AppData/Roaming/Python/Python313/site-packages/xdis/__init__.py
c:/Users/LAGz

</final_file_content>

IMPORTANT: For any future changes to this file, use the final_file_content shown above as your reference. This content reflects the current state of the file, including any auto-formatting (e.g., if you used single quotes but the formatter converted them to double quotes). Always base your SEARCH/REPLACE operations on this final version to ensure accuracy.



New problems detected after saving the file:
ml/prediction_model.py
- [Pylance Error] Line 127: Dictionary entries must contain key/value pairs
- [Pylance Error] Line 124: "{" was not closed
- [Pylance Error] Line 124: "(" was not closed
- [Pylance Error] Line 205: String literal is unterminated
- [Pylance Error] Line 205: "[" was not closed
- [Pylance Error] Line 590: Statements must be separated by newlines or semicolons<environment_details>
# VSCode Visible Files
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
notifications/discord_bot.py
C:/Program Files/Microsoft VS Code/rendererLog
ml/prediction_model.py

# VSCode Open Tabs
../OneDrive/BROKR/trading_strategy/strategies/Untitled-1.py
../OneDrive/BROKR/tests/test_mean_reversion.py
c:/Users/LAGZt/AppData/Roaming/Python/Python313/site-packages/xdis/__init__.py
c:/Users/LAGz

</final_file_content>

IMPORTANT: For any future changes to this file, use the final_file_content shown above as your reference. This content reflects the current state of the file, including any auto-formatting (e.g., if you used single quotes but the formatter converted them to double quotes). Always base your SEARCH/REPLACE operations on this final version to ensure accuracy.



New problems detected after saving the file:
ml/prediction_model.py
- [Pylance Error] Line 127: Dictionary entries must contain key/value pairs
- [Pylance Error] Line 124: "{" was not closed
- [Pylance Error] Line 124: "(" was not closed
- [Pylance Error] Line 205: String literal is unterminated
- [Pylance Error] Line 205: "[" was not closed
- [Pylance Error] Line 590: Statements must be separated by newlines or semicolons<environment_details>
# VSCode Visible Files
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
notifications/discord_bot.py
C:/Program Files/Microsoft VS Code/rendererLog
ml/prediction_model.py

# VSCode Open Tabs
../OneDrive/BROKR/trading_strategy/strategies/Untitled-1.py
../OneDrive/BROKR/tests/test_mean_reversion.py
c:/Users/LAGZt/AppData/Roaming/Python/Python313/site-packages/xdis/__init__.py
c:/Users/LAGz

</final_file_content>

IMPORTANT: For any future changes to this file, use the final_file_content shown above as your reference. This content reflects the current state of the file, including any auto-formatting (e.g., if you used single quotes but the formatter converted them to double quotes). Always base your SEARCH/REPLACE operations on this final version to ensure accuracy.



New problems detected after saving the file:
ml/prediction_model.py
- [Pylance Error] Line 127: Dictionary entries must contain key/value pairs
- [Pylance Error] Line 124: "{" was not closed
- [Pylance Error] Line 124: "(" was not closed
- [Pylance Error] Line 205: String literal is unterminated
- [Pylance Error] Line 205: "[" was not closed
- [Pylance Error] Line 590: Statements must be separated by newlines or semicolons<environment_details>
# VSCode Visible Files
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
notifications/discord_bot.py
C:/Program Files/Microsoft VS Code/rendererLog
ml/prediction_model.py

# VSCode Open Tabs
../OneDrive/BROKR/trading_strategy/strategies/Untitled-1.py
../OneDrive/BROKR/tests/test_mean_reversion.py
c:/Users/LAGZt/AppData/Roaming/Python/Python313/site-packages/xdis/__init__.py
c:/Users/LAGz

</final_file_content>

IMPORTANT: For any future changes to this file, use the final_file_content shown above as your reference. This content reflects the current state of the file, including any auto-formatting (e.g., if you used single quotes but the formatter converted them to double quotes). Always base your SEARCH/REPLACE operations on this final version to ensure accuracy.



New problems detected after saving the file:
ml/prediction_model.py
- [Pylance Error] Line 127: Dictionary entries must contain key/value pairs
- [Pylance Error] Line 124: "{" was not closed
- [Pylance Error] Line 124: "(" was not closed
- [Pylance Error] Line 205: String literal is unterminated
- [Pylance Error] Line 205: "[" was not closed
- [Pylance Error] Line 590: Statements must be separated by newlines or semicolons<environment_details>
# VSCode Visible Files
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
notifications/discord_bot.py
C:/Program Files/Microsoft VS Code/rendererLog
ml/prediction_model.py

# VSCode Open Tabs
../OneDrive/BROKR/trading_strategy/strategies/Untitled-1.py
../OneDrive/BROKR/tests/test_mean_reversion.py
c:/Users/LAGZt/AppData/Roaming/Python/Python313/site-packages/xdis/__init__.py
c:/Users/LAGz

</final_file_content>

IMPORTANT: For any future changes to this file, use the final_file_content shown above as your reference. This content reflects the current state of the file, including any auto-formatting (e.g., if you used single quotes but the formatter converted them to double quotes). Always base your SEARCH/REPLACE operations on this final version to ensure accuracy.



New problems detected after saving the file:
ml/prediction_model.py
- [Pylance Error] Line 127: Dictionary entries must contain key/value pairs
- [Pylance Error] Line 124: "{" was not closed
- [Pylance Error] Line 124: "(" was not closed
- [Pylance Error] Line 205: String literal is unterminated
- [Pylance Error] Line 205: "[" was not closed
- [Pylance Error] Line 590: Statements must be separated by newlines or semicolons<environment_details>
# VSCode Visible Files
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
notifications/discord_bot.py
C:/Program Files/Microsoft VS Code/rendererLog
ml/prediction_model.py

# VSCode Open Tabs
../OneDrive/BROKR/trading_strategy/strategies/Untitled-1.py
../OneDrive/BROKR/tests/test_mean_reversion.py
c:/Users/LAGZt/AppData/Roaming/Python/Python313/site-packages/xdis/__init__.py
c:/Users/LAGz

</final_file_content>

IMPORTANT: For any future changes to this file, use the final_file_content shown above as your reference. This content reflects the current state of the file, including any auto-formatting (e.g., if you used single quotes but the formatter converted them to double quotes). Always base your SEARCH/REPLACE operations on this final version to ensure accuracy.



New problems detected after saving the file:
ml/prediction_model.py
- [Pylance Error] Line 127: Dictionary entries must contain key/value pairs
- [Pylance Error] Line 124: "{" was not closed
- [Pylance Error] Line 124: "(" was not closed
- [Pylance Error] Line 205: String literal is unterminated
- [Pylance Error] Line 205: "[" was not closed
- [Pylance Error] Line 590: Statements must be separated by newlines or semicolons<environment_details>
# VSCode Visible Files
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
notifications/discord_bot.py
C:/Program Files/Microsoft VS Code/rendererLog
ml/prediction_model.py

# VSCode Open Tabs
../OneDrive/BROKR/trading_strategy/strategies/Untitled-1.py
../OneDrive/BROKR/tests/test_mean_reversion.py
c:/Users/LAGZt/AppData/Roaming/Python/Python313/site-packages/xdis/__init__.py
c:/Users/LAGz

</final_file_content>

IMPORTANT: For any future changes to this file, use the final_file_content shown above as your reference. This content reflects the current state of the file, including any auto-formatting (e.g., if you used single quotes but the formatter converted them to double quotes). Always base your SEARCH/REPLACE operations on this final version to ensure accuracy.



New problems detected after saving the file:
ml/prediction_model.py
- [Pylance Error] Line 127: Dictionary entries must contain key/value pairs
- [Pylance Error] Line 124: "{" was not closed
- [Pylance Error] Line 124: "(" was not closed
- [Pylance Error] Line 205: String literal is unterminated
- [Pylance Error] Line 205: "[" was not closed
- [Pylance Error] Line 590: Statements must be separated by newlines or semicolons<environment_details>
# VSCode Visible Files
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
notifications/discord_bot.py
C:/Program Files/Microsoft VS Code/rendererLog
ml/prediction_model.py

# VSCode Open Tabs
../OneDrive/BROKR/trading_strategy/strategies/Untitled-1.py
../OneDrive/BROKR/tests/test_mean_reversion.py
c:/Users/LAGZt/AppData/Roaming/Python/Python313/site-packages/xdis/__init__.py
c:/Users/LAGz

</final_file_content>

IMPORTANT: For any future changes to this file, use the final_file_content shown above as your reference. This content reflects the current state of the file, including any auto-formatting (e.g., if you used single quotes but the formatter converted them to double quotes). Always base your SEARCH/REPLACE operations on this final version to ensure accuracy.



New problems detected after saving the file:
ml/prediction_model.py
- [Pylance Error] Line 127: Dictionary entries must contain key/value pairs
- [Pylance Error] Line 124: "{" was not closed
- [Pylance Error] Line 124: "(" was not closed
- [Pylance Error] Line 205: String literal is unterminated
- [Pylance Error] Line 205: "[" was not closed
- [Pylance Error] Line 590: Statements must be separated by newlines or semicolons<environment_details>
# VSCode Visible Files
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
notifications/discord_bot.py
C:/Program Files/Microsoft VS Code/rendererLog
ml/prediction_model.py

# VSCode Open Tabs
../OneDrive/BROKR/trading_strategy/strategies/Untitled-1.py
../OneDrive/BROKR/tests/test_mean_reversion.py
c:/Users/LAGZt/AppData/Roaming/Python/Python313/site-packages/xdis/__init__.py
c:/Users/LAGz

</final_file_content>

IMPORTANT: For any future changes to this file, use the final_file_content shown above as your reference. This content reflects the current state of the file, including any auto-formatting (e.g., if you used single quotes but the formatter converted them to double quotes). Always base your SEARCH/REPLACE operations on this final version to ensure accuracy.



New problems detected after saving the file:
ml/prediction_model.py
- [Pylance Error] Line 127: Dictionary entries must contain key/value pairs
- [Pylance Error] Line 124: "{" was not closed
- [Pylance Error] Line 124: "(" was not closed
- [Pylance Error] Line 205: String literal is unterminated
- [Pylance Error] Line 205: "[" was not closed
- [Pylance Error] Line 590: Statements must be separated by newlines or semicolons<environment_details>
# VSCode Visible Files
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
notifications/discord_bot.py
C:/Program Files/Microsoft VS Code/rendererLog
ml/prediction_model.py

# VSCode Open Tabs
../OneDrive/BROKR/trading_strategy/strategies/Untitled-1.py
../OneDrive/BROKR/tests/test_mean_reversion.py
c:/Users/LAGZt/AppData/Roaming/Python/Python313/site-packages/xdis/__init__.py
c:/Users/LAGz

</final_file_content>

IMPORTANT: For any future changes to this file, use the final_file_content shown above as your reference. This content reflects the current state of the file, including any auto-formatting (e.g., if you used single quotes but the formatter converted them to double quotes). Always base your SEARCH/REPLACE operations on this final version to ensure accuracy.



New problems detected after saving the file:
ml/prediction_model.py
- [Pylance Error] Line 127: Dictionary entries must contain key/value pairs
- [Pylance Error] Line 124: "{" was not closed
- [Pylance Error] Line 124: "(" was not closed
- [Pylance Error] Line 205: String literal is unterminated
- [Pylance Error] Line 205: "[" was not closed
- [Pylance Error] Line 590: Statements must be separated by newlines or semicolons<environment_details>
# VSCode Visible Files
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
notifications/discord_bot.py
C:/Program Files/Microsoft VS Code/rendererLog
ml/prediction_model.py

# VSCode Open Tabs
../OneDrive/BROKR/trading_strategy/strategies/Untitled-1.py
../OneDrive/BROKR/tests/test_mean_reversion.py
c:/Users/LAGZt/AppData/Roaming/Python/Python313/site-packages/xdis/__init__.py
c:/Users/LAGz

</final_file_content>

IMPORTANT: For any future changes to this file, use the final_file_content shown above as your reference. This content reflects the current state of the file, including any auto-formatting (e.g., if you used single quotes but the formatter converted them to double quotes). Always base your SEARCH/REPLACE operations on this final version to ensure accuracy.



New problems detected after saving the file:
ml/prediction_model.py
- [Pylance Error] Line 127: Dictionary entries must contain key/value pairs
- [Pylance Error] Line 124: "{" was not closed
- [Pylance Error] Line 124: "(" was not closed
- [Pylance Error] Line 205: String literal is unterminated
- [Pylance Error] Line 205: "[" was not closed
- [Pylance Error] Line 590: Statements must be separated by newlines or semicolons<environment_details>
# VSCode Visible Files
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
notifications/discord_bot.py
C:/Program Files/Microsoft VS Code/rendererLog
ml/prediction_model.py

# VSCode Open Tabs
../OneDrive/BROKR/trading_strategy/strategies/Untitled-1.py
../OneDrive/BROKR/tests/test_mean_reversion.py
c:/Users/LAGZt/AppData/Roaming/Python/Python313/site-packages/xdis/__init__.py
c:/Users/LAGz

</final_file_content>

IMPORTANT: For any future changes to this file, use the final_file_content shown above as your reference. This content reflects the current state of the file, including any auto-formatting (e.g., if you used single quotes but the formatter converted them to double quotes). Always base your SEARCH/REPLACE operations on this final version to ensure accuracy.



New problems detected after saving the file:
ml/prediction_model.py
- [Pylance Error] Line 127: Dictionary entries must contain key/value pairs
- [Pylance Error] Line 124: "{" was not closed
- [Pylance Error] Line 124: "(" was not closed
- [Pylance Error] Line 205: String literal is unterminated
- [Pylance Error] Line 205: "[" was not closed
- [Pylance Error] Line 590: Statements must be separated by newlines or semicolons<environment_details>
# VSCode Visible Files
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
notifications/discord_bot.py
C:/Program Files/Microsoft VS Code/rendererLog
ml/prediction_model.py

# VSCode Open Tabs
../OneDrive/BROKR/trading_strategy/strategies/Untitled-1.py
../OneDrive/BROKR/tests/test_mean_reversion.py
c:/Users/LAGZt/AppData/Roaming/Python/Python313/site-packages/xdis/__init__.py
c:/Users/LAGz

</final_file_content>

IMPORTANT: For any future changes to this file, use the final_file_content shown above as your reference. This content reflects the current state of the file, including any auto-formatting (e.g., if you used single quotes but the formatter converted them to double quotes). Always base your SEARCH/REPLACE operations on this final version to ensure accuracy.



New problems detected after saving the file:
ml/prediction_model.py
- [Pylance Error] Line 127: Dictionary entries must contain key/value pairs
- [Pylance Error] Line 124: "{" was not closed
- [Pylance Error] Line 124: "(" was not closed
- [Pylance Error] Line 205: String literal is unterminated
- [Pylance Error] Line 205: "[" was not closed
- [Pylance Error] Line 590: Statements must be separated by newlines or semicolons<environment_details>
# VSCode Visible Files
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
notifications/discord_bot.py
C:/Program Files/Microsoft VS Code/rendererLog
ml/prediction_model.py

# VSCode Open Tabs
../OneDrive/BROKR/trading_strategy/strategies/Untitled-1.py
../OneDrive/BROKR/tests/test_mean_reversion.py
c:/Users/LAGZt/AppData/Roaming/Python/Python313/site-packages/xdis/__init__.py
c:/Users/LAGz

</final_file_content>

IMPORTANT: For any future changes to this file, use the final_file_content shown above as your reference. This content reflects the current state of the file, including any auto-formatting (e.g., if you used single quotes but the formatter converted them to double quotes). Always base your SEARCH/REPLACE operations on this final version to ensure accuracy.



New problems detected after saving the file:
ml/prediction_model.py
- [Pylance Error] Line 127: Dictionary entries must contain key/value pairs
- [Pylance Error] Line 124: "{" was not closed
- [Pylance Error] Line 124: "(" was not closed
- [Pylance Error] Line 205: String literal is unterminated
- [Pylance Error] Line 205: "[" was not closed
- [Pylance Error] Line 590: Statements must be separated by newlines or semicolons<environment_details>
# VSCode Visible Files
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
notifications/discord_bot.py
C:/Program Files/Microsoft VS Code/rendererLog
ml/prediction_model.py

# VSCode Open Tabs
../OneDrive/BROKR/trading_strategy/strategies/Untitled-1.py
../OneDrive/BROKR/tests/test_mean_reversion.py
c:/Users/LAGZt/AppData/Roaming/Python/Python313/site-packages/xdis/__init__.py
c:/Users/LAGz

</final_file_content>

IMPORTANT: For any future changes to this file, use the final_file_content shown above as your reference. This content reflects the current state of the file, including any auto-formatting (e.g., if you used single quotes but the formatter converted them to double quotes). Always base your SEARCH/REPLACE operations on this final version to ensure accuracy.



New problems detected after saving the file:
ml/prediction_model.py
- [Pylance Error] Line 127: Dictionary entries must contain key/value pairs
- [Pylance Error] Line 124: "{" was not closed
- [Pylance Error] Line 124: "(" was not closed
- [Pylance Error] Line 205: String literal is unterminated
- [Pylance Error] Line 205: "[" was not closed
- [Pylance Error] Line 590: Statements must be separated by newlines or semicolons<environment_details>
# VSCode Visible Files
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
notifications/discord_bot.py
C:/Program Files/Microsoft VS Code/rendererLog
ml/prediction_model.py

# VSCode Open Tabs
../OneDrive/BROKR/trading_strategy/strategies/Untitled-1.py
../OneDrive/BROKR/tests/test_mean_reversion.py
c:/Users/LAGZt/AppData/Roaming/Python/Python313/site-packages/xdis/__init__.py
c:/Users/LAGz

</final_file_content>

IMPORTANT: For any future changes to this file, use the final_file_content shown above as your reference. This content reflects the current state of the file, including any auto-formatting (e.g., if you used single quotes but the formatter converted them to double quotes). Always base your SEARCH/REPLACE operations on this final version to ensure accuracy.



New problems detected after saving the file:
ml/prediction_model.py
- [Pylance Error] Line 127: Dictionary entries must contain key/value pairs
- [Pylance Error] Line 124: "{" was not closed
- [Pylance Error] Line 124: "(" was not closed
- [Pylance Error] Line 205: String literal is unterminated
- [Pylance Error] Line 205: "[" was not closed
- [Pylance Error] Line 590: Statements must be separated by newlines or semicolons<environment_details>
# VSCode Visible Files
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
notifications/discord_bot.py
C:/Program Files/Microsoft VS Code/rendererLog
ml/prediction_model.py

# VSCode Open Tabs
../OneDrive/BROKR/trading_strategy/strategies/Untitled-1.py
../OneDrive/BROKR/tests/test_mean_reversion.py
c:/Users/LAGz

</final_file_content>

IMPORTANT: For any future changes to this file, use the final_file_content shown above as your reference. This content reflects the current state of the file, including any auto-formatting (e.g., if you used single quotes but the formatter converted them to double quotes). Always base your SEARCH/REPLACE operations on this final version to ensure accuracy.



New problems detected after saving the file:
ml/prediction_model.py
- [Pylance Error] Line 127: Dictionary entries must contain key/value pairs
- [Pylance Error] Line 124: "{" was not closed
- [Pylance Error] Line 124: "(" was not closed
- [Pylance Error] Line 205: String literal is unterminated
- [Pylance Error] Line 205: "[" was not closed
- [Pylance Error] Line 590: Statements must be separated by newlines or semicolons<environment_details>
# VSCode Visible Files
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
notifications/discord_bot.py
C:/Program Files/Microsoft VS Code/rendererLog
ml/prediction_model.py

# VSCode Open Tabs
../OneDrive/BROKR/trading_strategy/strategies/Untitled-1.py
../OneDrive/BROKR/tests/test_mean_reversion.py
c:/Users/LAGZt/AppData/Roaming/Python/Python313/site-packages/xdis/__init__.py
c:/Users/LAGz

</final_file_content>

IMPORTANT: For any future changes to this file, use the final_file_content shown above as your reference. This content reflects the current state of the file, including any auto-formatting (e.g., if you used single quotes but the formatter converted them to double quotes). Always base your SEARCH/REPLACE operations on this final version to ensure accuracy.



New problems detected after saving the file:
ml/prediction_model.py
- [Pylance Error] Line 127: Dictionary entries must contain key/value pairs
- [Pylance Error] Line 124: "{" was not closed
- [Pylance Error] Line 124: "(" was not closed
- [Pylance Error] Line 205: String literal is unterminated
- [Pylance Error] Line 205: "[" was not closed
- [Pylance Error] Line 590: Statements must be separated by newlines or semicolons<environment_details>
# VSCode Visible Files
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
notifications/discord_bot.py
C:/Program Files/Microsoft VS Code/rendererLog
ml/prediction_model.py

# VSCode Open Tabs
../OneDrive/BROKR/trading_strategy/strategies/Untitled-1.py
../OneDrive/BROKR/tests/test_mean_reversion.py
c:/Users/LAGZt/AppData/Roaming/Python/Python313/site-packages/xdis/__init__.py
c:/Users/LAGz

</final_file_content>

IMPORTANT: For any future changes to this file, use the final_file_content shown above as your reference. This content reflects the current state of the file, including any auto-formatting (e.g., if you used single quotes but the formatter converted them to double quotes). Always base your SEARCH/REPLACE operations on this final version to ensure accuracy.



New problems detected after saving the file:
ml/prediction_model.py
- [Pylance Error] Line 127: Dictionary entries must contain key/value pairs
- [Pylance Error] Line 124: "{" was not closed
- [Pylance Error] Line 124: "(" was not closed
- [Pylance Error] Line 205: String literal is unterminated
- [Pylance Error] Line 205: "[" was not closed
- [Pylance Error] Line 590: Statements must be separated by newlines or semicolons<environment_details>
# VSCode Visible Files
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
notifications/discord_bot.py
C:/Program Files/Microsoft VS Code/rendererLog
ml/prediction_model.py

# VSCode Open Tabs
../OneDrive/BROKR/trading_strategy/strategies/Untitled-1.py
../OneDrive/BROKR/tests/test_mean_reversion.py
c:/Users/LAGZt/AppData/Roaming/Python/Python313/site-packages/xdis/__init__.py
c:/Users/LAGz

</final_file_content>

IMPORTANT: For any future changes to this file, use the final_file_content shown above as your reference. This content reflects the current state of the file, including any auto-formatting (e.g., if you used single quotes but the formatter converted them to double quotes). Always base your SEARCH/REPLACE operations on this final version to ensure accuracy.



New problems detected after saving the file:
ml/prediction_model.py
- [Pylance Error] Line 127: Dictionary entries must contain key/value pairs
- [Pylance Error] Line 124: "{" was not closed
- [Pylance Error] Line 124: "(" was not closed
- [Pylance Error] Line 205: String literal is unterminated
- [Pylance Error] Line 205: "[" was not closed
- [Pylance Error] Line 590: Statements must be separated by newlines or semicolons<environment_details>
# VSCode Visible Files
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
notifications/discord_bot.py
C:/Program Files/Microsoft VS Code/rendererLog
ml/prediction_model.py

# VSCode Open Tabs
../OneDrive/BROKR/trading_strategy/strategies/Untitled-1.py
../OneDrive/BROKR/tests/test_mean_reversion.py
c:/Users/LAGz

</final_file_content>

IMPORTANT: For any future changes to this file, use the final_file_content shown above as your reference. This content reflects the current state of the file, including any auto-formatting (e.g., if you used single quotes but the formatter converted them to double quotes). Always base your SEARCH/REPLACE operations on this final version to ensure accuracy.



New problems detected after saving the file:
ml/prediction_model.py
- [Pylance Error] Line 127: Dictionary entries must contain key/value pairs
- [Pylance Error] Line 124: "{" was not closed
- [Pylance Error] Line 124: "(" was not closed
- [Pylance Error] Line 205: String literal is unterminated
- [Pylance Error] Line 205: "[" was not closed
- [Pylance Error] Line 590: Statements must be separated by newlines or semicolons<environment_details>
# VSCode Visible Files
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
notifications/discord_bot.py
C:/Program Files/Microsoft VS Code/rendererLog
ml/prediction_model.py

# VSCode Open Tabs
../OneDrive/BROKR/trading_strategy/strategies/Untitled-1.py
../OneDrive/BROKR/tests/test_mean_reversion.py
c:/Users/LAGZt/AppData/Roaming/Python/Python313/site-packages/xdis/__init__.py
c:/Users/LAGz

</final_file_content>

IMPORTANT: For any future changes to this file, use the final_file_content shown above as your reference. This content reflects the current state of the file, including any auto-formatting (e.g., if you used single quotes but the formatter converted them to double quotes). Always base your SEARCH/REPLACE operations on this final version to ensure accuracy.



New problems detected after saving the file:
ml/prediction_model.py
- [Pylance Error] Line 127: Dictionary entries must contain key/value pairs
- [Pylance Error] Line 124: "{" was not closed
- [Pylance Error] Line 124: "(" was not closed
- [Pylance Error] Line 205: String literal is unterminated
- [Pylance Error] Line 205: "[" was not closed
- [Pylance Error] Line 590: Statements must be separated by newlines or semicolons<environment_details>
# VSCode Visible Files
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
notifications/discord_bot.py
C:/Program Files/Microsoft VS Code/rendererLog
ml/prediction_model.py

# VSCode Open Tabs
../OneDrive/BROKR/trading_strategy/strategies/Untitled-1.py
../OneDrive/BROKR/tests/test_mean_reversion.py
c:/Users/LAGZt/AppData/Roaming/Python/Python313/site-packages/xdis/__init__.py
c:/Users/LAGz

</final_file_content>

IMPORTANT: For any future changes to this file, use the final_file_content shown above as your reference. This content reflects the current state of the file, including any auto-formatting (e.g., if you used single quotes but the formatter converted them to double quotes). Always base your SEARCH/REPLACE operations on this final version to ensure accuracy.



New problems detected after saving the file:
ml/prediction_model.py
- [Pylance Error] Line 127: Dictionary entries must contain key/value pairs
- [Pylance Error] Line 124: "{" was not closed
- [Pylance Error] Line 124: "(" was not closed
- [Pylance Error] Line 205: String literal is unterminated
- [Pylance Error] Line 205: "[" was not closed
- [Pylance Error] Line 590: Statements must be separated by newlines or semicolons<environment_details>
# VSCode Visible Files
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
notifications/discord_bot.py
C:/Program Files/Microsoft VS Code/rendererLog
ml/prediction_model.py

# VSCode Open Tabs
../OneDrive/BROKR/trading_strategy/strategies/Untitled-1.py
../OneDrive/BROKR/tests/test_mean_reversion.py
c:/Users/LAGZt/AppData/Roaming/Python/Python313/site-packages/xdis/__init__.py
c:/Users/LAGz

</final_file_content>

IMPORTANT: For any future changes to this file, use the final_file_content shown above as your reference. This content reflects the current state of the file, including any auto-formatting (e.g., if you used single quotes but the formatter converted them to double quotes). Always base your SEARCH/REPLACE operations on this final version to ensure accuracy.



New problems detected after saving the file:
ml/prediction_model.py
- [Pylance Error] Line 127: Dictionary entries must contain key/value pairs
- [Pylance Error] Line 124: "{" was not closed
- [Pylance Error] Line 124: "(" was not closed
- [Pylance Error] Line 205: String literal is unterminated
- [Pylance Error] Line 205: "[" was not closed
- [Pylance Error] Line 590: Statements must be separated by newlines or semicolons<environment_details>
# VSCode Visible Files
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
notifications/discord_bot.py
C:/Program Files/Microsoft VS Code/rendererLog
ml/prediction_model.py

# VSCode Open Tabs
../OneDrive/BROKR/trading_strategy/strategies/Untitled-1.py
../OneDrive/BROKR/tests/test_mean_reversion.py
c:/Users/LAGZt/AppData/Roaming/Python/Python313/site-packages/xdis/__init__.py
c:/Users/LAGz

</final_file_content>

IMPORTANT: For any future changes to this file, use the final_file_content shown above as your reference. This content reflects the current state of the file, including any auto-formatting (e.g., if you used single quotes but the formatter converted them to double quotes). Always base your SEARCH/REPLACE operations on this final version to ensure accuracy.



New problems detected after saving the file:
ml/prediction_model.py
- [Pylance Error] Line 127: Dictionary entries must contain key/value pairs
- [Pylance Error] Line 124: "{" was not closed
- [Pylance Error] Line 124: "(" was not closed
- [Pylance Error] Line 205: String literal is unterminated
- [Pylance Error] Line 205: "[" was not closed
- [Pylance Error] Line 590: Statements must be separated by newlines or semicolons<environment_details>
# VSCode Visible Files
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
notifications/discord_bot.py
C:/Program Files/Microsoft VS Code/rendererLog
ml/prediction_model.py

# VSCode Open Tabs
../OneDrive/BROKR/trading_strategy/strategies/Untitled-1.py
../OneDrive/BROKR/tests/test_mean_reversion.py
c:/Users/LAGZt/AppData/Roaming/Python/Python313/site-packages/xdis/__init__.py
c:/Users/LAGz

</final_file_content>

IMPORTANT: For any future changes to this file, use the final_file_content shown above as your reference. This content reflects the current state of the file, including any auto-formatting (e.g., if you used single quotes but the formatter converted them to double quotes). Always base your SEARCH/REPLACE operations on this final version to ensure accuracy.



New problems detected after saving the file:
ml/prediction_model.py
- [Pylance Error] Line 127: Dictionary entries must contain key/value pairs
- [Pylance Error] Line 124: "{" was not closed
- [Pylance Error] Line 124: "(" was not closed
- [Pylance Error] Line 205: String literal is unterminated
- [Pylance Error] Line 205: "[" was not closed
- [Pylance Error] Line 590: Statements must be separated by newlines or semicolons<environment_details>
# VSCode Visible Files
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
notifications/discord_bot.py
C:/Program Files/Microsoft VS Code/rendererLog
ml/prediction_model.py

# VSCode Open Tabs
../OneDrive/BROKR/trading_strategy/strategies/Untitled-1.py
../OneDrive/BROKR/tests/test_mean_reversion.py
c:/Users/LAGZt/AppData/Roaming/Python/Python313/site-packages/xdis/__init__.py
c:/Users/LAGz

</final_file_content>

IMPORTANT: For any future changes to this file, use the final_file_content shown above as your reference. This content reflects the current state of the file, including any auto-formatting (e.g., if you used single quotes but the formatter converted them to double quotes). Always base your SEARCH/REPLACE operations on this final version to ensure accuracy.



New problems detected after saving the file:
ml/prediction_model.py
- [Pylance Error] Line 127: Dictionary entries must contain key/value pairs
- [Pylance Error] Line 124: "{" was not closed
- [Pylance Error] Line 124: "(" was not closed
- [Pylance Error] Line 205: String literal is unterminated
- [Pylance Error] Line 205: "[" was not closed
- [Pylance Error] Line 590: Statements must be separated by newlines or semicolons<environment_details>
# VSCode Visible Files
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
notifications/discord_bot.py
C:/Program Files/Microsoft VS Code/rendererLog
ml/prediction_model.py

# VSCode Open Tabs
../OneDrive/BROKR/trading_strategy/strategies/Untitled-1.py
../OneDrive/BROKR/tests/test_mean_reversion.py
c:/Users/LAGZt/AppData/Roaming/Python/Python313/site-packages/xdis/__init__.py
c:/Users/LAGz

</final_file_content>

IMPORTANT: For any future changes to this file, use the final_file_content shown above as your reference. This content reflects the current state of the file, including any auto-formatting (e.g., if you used single quotes but the formatter converted them to double quotes). Always base your SEARCH/REPLACE operations on this final version to ensure accuracy.



New problems detected after saving the file:
ml/prediction_model.py
- [Pylance Error] Line 127: Dictionary entries must contain key/value pairs
- [Pylance Error] Line 124: "{" was not closed
- [Pylance Error] Line 124: "(" was not closed
- [Pylance Error] Line 205: String literal is unterminated
- [Pylance Error] Line 205: "[" was not closed
- [Pylance Error] Line 590: Statements must be separated by newlines or semicolons<environment_details>
# VSCode Visible Files
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
notifications/discord_bot.py
C:/Program Files/Microsoft VS Code/rendererLog
ml/prediction_model.py

# VSCode Open Tabs
../OneDrive/BROKR/trading_strategy/strategies/Untitled-1.py
../OneDrive/BROKR/tests/test_mean_reversion.py
c:/Users/LAGZt/AppData/Roaming/Python/Python313/site-packages/xdis/__init__.py
c:/Users/LAGz

</final_file_content>

IMPORTANT: For any future changes to this file, use the final_file_content shown above as your reference. This content reflects the current state of the file, including any auto-formatting (e.g., if you used single quotes but the formatter converted them to double quotes). Always base your SEARCH/REPLACE operations on this final version to ensure accuracy.



New problems detected after saving the file:
ml/prediction_model.py
- [Pylance Error] Line 127: Dictionary entries must contain key/value pairs
- [Pylance Error] Line 124: "{" was not closed
- [Pylance Error] Line 124: "(" was not closed
- [Pylance Error] Line 205: String literal is unterminated
- [Pylance Error] Line 205: "[" was not closed
- [Pylance Error] Line 590: Statements must be separated by newlines or semicolons<environment_details>
# VSCode Visible Files
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
notifications/discord_bot.py
C:/Program Files/Microsoft VS Code/rendererLog
ml/prediction_model.py

# VSCode Open Tabs
../OneDrive/BROKR/trading_strategy/strategies/Untitled-1.py
../OneDrive/BROKR/tests/test_mean_reversion.py
c:/Users/LAGZt/AppData/Roaming/Python/Python313/site-packages/xdis/__init__.py
c:/Users/LAGz

</final_file_content>

IMPORTANT: For any future changes to this file, use the final_file_content shown above as your reference. This content reflects the current state of the file, including any auto-formatting (e.g., if you used single quotes but the formatter converted them to double quotes). Always base your SEARCH/REPLACE operations on this final version to ensure accuracy.



New problems detected after saving the file:
ml/prediction_model.py
- [Pylance Error] Line 127: Dictionary entries must contain key/value pairs
- [Pylance Error] Line 124: "{" was not closed
- [Pylance Error] Line 124: "(" was not closed
- [Pylance Error] Line 205: String literal is unterminated
- [Pylance Error] Line 205: "[" was not closed
- [Pylance Error] Line 590: Statements must be separated by newlines or semicolons<environment_details>
# VSCode Visible Files
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/0
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/1
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/2
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/3
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/4
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/5
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/6
C:/response_aa51a3c8-57eb-4f14-a845-68c6ebc86aa7/7
notifications/discord_bot.py
C:/Program Files/Microsoft VS Code/rendererLog
ml/prediction_model.py

# VSCode Open Tabs
../OneDrive/BROKR/trading_strategy/strategies/Untitled-1.py
../OneDrive/BROKR/tests/test_mean_reversion.py
c:/Users/LAGZt/AppData/Roaming/Python/Python313/site-packages/xdis/__init__.py
c:/Users/LAGz

</final_file_content>

IMPORTANT: For any future changes to this file, use the final_file_content shown above as your reference. This content reflects the current state of the file, including any auto-formatting (e.g., if you used single quotes but the formatter converted them to double quotes). Always
