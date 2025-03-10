# sentiment_ml.py
import os
import json
import time
import schedule
import argparse
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import joblib
from prometheus_client import Counter, Gauge, Summary, Histogram, start_http_server
from database_manager import DatabaseManager
from utils.logging_config import get_module_logger

# Prometheus metrics
ML_MODEL_TRAINS = Counter('sentiment_ml_model_trains_total', 'Number of model training events', ['symbol', 'horizon', 'model_type'])
ML_PREDICTIONS = Counter('sentiment_ml_predictions_total', 'Number of prediction events', ['symbol', 'horizon'])
ML_ERRORS = Counter('sentiment_ml_errors_total', 'Number of errors', ['operation', 'symbol', 'reason'])
DB_OPERATIONS = Counter('sentiment_ml_db_operations_total', 'Number of database operations', ['operation'])
FEATURE_ENGINEERING_RUNS = Counter('sentiment_ml_feature_engineering_total', 'Number of feature engineering operations')
PREDICTION_VALUES = Gauge('sentiment_ml_prediction_value', 'Prediction values', ['symbol', 'horizon', 'direction'])
PREDICTION_CONFIDENCE = Gauge('sentiment_ml_prediction_confidence', 'Prediction confidence', ['symbol', 'horizon'])
MODEL_R2_SCORE = Gauge('sentiment_ml_model_r2', 'R-squared value of models', ['symbol', 'horizon'])
MODEL_MAE = Gauge('sentiment_ml_model_mae', 'Mean absolute error of models', ['symbol', 'horizon'])
MODEL_AGE_DAYS = Gauge('sentiment_ml_model_age_days', 'Age of model in days', ['symbol', 'horizon'])
ACTIVE_MODELS = Gauge('sentiment_ml_active_models', 'Number of active models')
FEATURE_COUNT = Gauge('sentiment_ml_feature_count', 'Number of features used in model', ['symbol', 'horizon'])
SAMPLE_COUNT = Gauge('sentiment_ml_sample_count', 'Number of samples used in training', ['symbol', 'horizon'])
CACHE_SIZE = Gauge('sentiment_ml_cache_size', 'Size of feature cache')
TRAIN_DURATION = Summary('sentiment_ml_train_duration_seconds', 'Time spent training models', ['symbol', 'horizon'])
PREDICTION_DURATION = Summary('sentiment_ml_prediction_duration_seconds', 'Time spent making predictions', ['symbol', 'horizon'])
FEATURE_ENGINEERING_DURATION = Summary('sentiment_ml_feature_engineering_duration_seconds', 'Time spent on feature engineering')
DATA_FETCH_DURATION = Summary('sentiment_ml_data_fetch_duration_seconds', 'Time spent fetching data')
PREDICTION_DISTRIBUTION = Histogram('sentiment_ml_prediction_distribution', 'Distribution of prediction values', 
                                   ['symbol', 'horizon'], buckets=[-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5])

logger = get_module_logger("SentimentML")

class SentimentML:
    """Machine learning model for sentiment-based price prediction"""
    
    def __init__(self, config_file="sentiment_ml_config.json"):
        """Initialize the ML system"""
        self.feature_cache = {}
        self.data_cache_ttl = 3600  # Cache data for 1 hour
        self.active_model_count = 0
        ACTIVE_MODELS.set(self.active_model_count)
        
        self.load_config(config_file)
        self.setup_database()
        self._start_metrics_server()
        
        CACHE_SIZE.set(len(self.feature_cache))
        logger.info("SentimentML initialized")

    def _start_metrics_server(self):
        """Start the Prometheus metrics server"""
        try:
            metrics_port = int(os.environ.get("METRICS_PORT", 8002))
            start_http_server(metrics_port)
            logger.info(f"Prometheus metrics server started on port {metrics_port}")
        except Exception as e:
            logger.error(f"Error starting metrics server: {e}")

    def load_config(self, config_file):
        """Load configuration from JSON file"""
        with FEATURE_ENGINEERING_DURATION.time():
            try:
                # Check for containerized config
                container_config = os.path.join("/app/config", os.path.basename(config_file))
                if os.path.exists(container_config):
                    config_path = container_config
                else:
                    config_path = config_file
                
                with open(config_path, "r") as f:
                    self.config = json.load(f)
                
                self.model_type = self.config.get("model_type", "random_forest")
                self.prediction_horizons = self.config.get("prediction_horizons", [1, 3, 7])
                self.feature_windows = self.config.get("feature_windows", [1, 3, 7, 14])
                self.min_training_samples = self.config.get("min_training_samples", 30)
                self.retrain_interval_days = self.config.get("retrain_interval_days", 7)
                self.model_save_path = self.config.get("model_save_path", "ml_models")
                self.symbols = self.config.get("symbols", ["BTC", "ETH"])
                
                # Override with environment variables if available
                db_uri = os.environ.get("DB_URI")
                if db_uri:
                    logger.info(f"Using database URI from environment: {db_uri}")
                
                # Create model directory if it doesn't exist
                os.makedirs(self.model_save_path, exist_ok=True)
                
                logger.info("Configuration loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load configuration: {str(e)}")
                ML_ERRORS.labels(operation="config_load", symbol="all", reason="file_error").inc()
                raise

    def setup_database(self):
        """Set up database connection"""
        with DATA_FETCH_DURATION.time():
            try:
                self.db_manager = DatabaseManager(
                    host=os.environ.get("POSTGRES_HOST", "localhost"),
                    port=os.environ.get("POSTGRES_PORT", "5432"),
                    database=os.environ.get("POSTGRES_DB", "trading_db"),
                    user=os.environ.get("POSTGRES_USER", "bot_user"),
                    password=os.environ.get("POSTGRES_PASSWORD", "secure_password")
                )
                self.engine = self.db_manager.engine
                
                # Test connection
                self.db_connection_test = self.engine.connect()
                self.db_connection_test.close()
                
                DB_OPERATIONS.labels("connect").inc()
                logger.info("Database connection established")
            except Exception as e:
                logger.error(f"Failed to set up database: {str(e)}")
                ML_ERRORS.labels(operation="db_setup", symbol="all", reason="connection_error").inc()
                raise

    def get_training_data(self, symbol, force_refresh=False):
        """
        Fetch training data from the database
        
        Parameters:
        symbol (str): Cryptocurrency symbol
        force_refresh (bool): Force refresh the data even if cached
        
        Returns:
        DataFrame: Combined price and sentiment data
        """
        with DATA_FETCH_DURATION.time():
            cache_key = f"training_data_{symbol}"
            current_time = datetime.now()
            
            # Check cache if not forced to refresh
            if not force_refresh and cache_key in self.feature_cache and \
               f"{cache_key}_timestamp" in self.feature_cache and \
               (current_time - self.feature_cache[f"{cache_key}_timestamp"]).total_seconds() < self.data_cache_ttl:
                logger.info(f"Using cached training data for {symbol}")
                return self.feature_cache[cache_key]
            
            try:
                # Query for sentiment data aggregated by day
                sentiment_query = f"""
                    SELECT 
                        DATE_TRUNC('day', processed_date) as date,
                        COUNT(*) as record_count,
                        AVG(combined_score) as avg_sentiment,
                        SUM(bullish_keywords) as total_bullish,
                        SUM(bearish_keywords) as total_bearish,
                        EXTRACT(DOW FROM DATE_TRUNC('day', processed_date)) as day_of_week
                    FROM 
                        sentiment_youtube
                    WHERE 
                        processed_date >= NOW() - INTERVAL '60 days'
                    GROUP BY 
                        DATE_TRUNC('day', processed_date),
                        EXTRACT(DOW FROM DATE_TRUNC('day', processed_date))
                    ORDER BY 
                        date
                """
                
                with self.engine.connect() as conn:
                    DB_OPERATIONS.labels("query").inc()
                    sentiment_df = pd.read_sql(sentiment_query, conn)
                
                if len(sentiment_df) == 0:
                    logger.warning(f"No sentiment data available for {symbol}")
                    ML_ERRORS.labels(operation="data_fetch", symbol=symbol, reason="no_sentiment_data").inc()
                    return None
                
                # Convert date to datetime
                sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
                
                # Get price data
                price_df = pd.DataFrame()
                try:
                    # First try to get from database
                    price_query = f"""
                        SELECT 
                            DATE_TRUNC('day', timestamp) as date,
                            AVG(price) as price_mean,
                            MIN(price) as price_low,
                            MAX(price) as price_high,
                            FIRST_VALUE(price) OVER (PARTITION BY DATE_TRUNC('day', timestamp) ORDER BY timestamp) as price_open,
                            LAST_VALUE(price) OVER (PARTITION BY DATE_TRUNC('day', timestamp) ORDER BY timestamp) as price_close
                        FROM 
                            price_history
                        WHERE 
                            symbol = '{symbol}' AND
                            timestamp >= NOW() - INTERVAL '60 days'
                        GROUP BY 
                            DATE_TRUNC('day', timestamp)
                        ORDER BY 
                            date
                    """
                    
                    with self.engine.connect() as conn:
                        price_df = pd.read_sql(price_query, conn)
                    
                    # If database query returned no results, fall back to JSON file
                    if len(price_df) == 0:
                        raise ValueError("No price data in database")
                        
                except Exception as e:
                    # Fall back to reading from price_history.json if database fetch fails
                    logger.warning(f"Failed to get price data from database: {e}. Falling back to file.")
                    try:
                        with open("price_history.json", "r") as f:
                            price_history = json.load(f)
                            price_data = price_history.get(symbol, [])
                            
                            if not price_data:
                                logger.warning(f"No price data found for {symbol}")
                                ML_ERRORS.labels(operation="data_fetch", symbol=symbol, reason="no_price_data").inc()
                                return None
                                
                            price_df = pd.DataFrame(price_data)
                            price_df['timestamp'] = pd.to_datetime(price_df['timestamp'])
                            
                            # Group by day
                            price_df['date'] = pd.to_datetime(price_df['timestamp']).dt.date
                            price_df['date'] = pd.to_datetime(price_df['date'])
                            
                            price_df = price_df.groupby('date').agg({
                                'price': ['mean', 'min', 'max', 'first', 'last']
                            }).reset_index()
                            
                            # Flatten the MultiIndex columns
                            price_df.columns = ['date', 'price_mean', 'price_low', 'price_high', 'price_open', 'price_close']
                    except Exception as e2:
                        logger.error(f"Error loading price data: {str(e2)}")
                        ML_ERRORS.labels(operation="data_fetch", symbol=symbol, reason="price_file_error").inc()
                        return None
                
                # Merge sentiment and price data
                df = pd.merge(sentiment_df, price_df, on='date', how='inner')
                df = df.sort_values('date')
                
                # Calculate price change for each horizon
                for days in self.prediction_horizons:
                    df[f'price_change_{days}d'] = df['price_close'].pct_change(periods=days).shift(-days)
                    df[f'price_direction_{days}d'] = np.sign(df[f'price_change_{days}d'])
                
                # Remove rows with missing target values
                df = df.dropna(subset=[f'price_change_{days}d' for days in self.prediction_horizons])
                
                # Cache the result
                self.feature_cache[cache_key] = df
                self.feature_cache[f"{cache_key}_timestamp"] = current_time
                CACHE_SIZE.set(len(self.feature_cache))
                
                logger.info(f"Prepared training data for {symbol}: {len(df)} samples")
                SAMPLE_COUNT.labels(symbol=symbol, horizon="all").set(len(df))
                
                return df
                
            except Exception as e:
                logger.error(f"Error preparing training data: {str(e)}")
                ML_ERRORS.labels(operation="data_fetch", symbol=symbol, reason="query_error").inc()
                return None

    def engineer_features(self, df, horizon=None, cache_key=None):
        """
        Engineer features for ML model
        
        Parameters:
        df (DataFrame): Input dataframe with sentiment and price data
        horizon (int): Prediction horizon in days
        cache_key (str): Cache key for storing results
        
        Returns:
        DataFrame: DataFrame with engineered features
        """
        FEATURE_ENGINEERING_RUNS.inc()
        
        try:
            # Check cache first
            if cache_key and f"{cache_key}_features" in self.feature_cache:
                cached_features = self.feature_cache[f"{cache_key}_features"]
                if len(cached_features) == len(df):
                    logger.info(f"Using cached features for {cache_key}")
                    return cached_features
            
            # Engineer features
            features_df = df.copy()
            
            # Create rolling window features
            for window in self.feature_windows:
                if len(df) > window:
                    # Sentiment features
                    features_df[f'sentiment_ma_{window}d'] = df['avg_sentiment'].rolling(window=window).mean()
                    features_df[f'sentiment_std_{window}d'] = df['avg_sentiment'].rolling(window=window).std()
                    features_df[f'sentiment_slope_{window}d'] = df['avg_sentiment'].diff(window) / window
                    
                    # Bullish/bearish ratio
                    features_df[f'bull_bear_ratio_{window}d'] = (
                        df['total_bullish'].rolling(window=window).sum() / 
                        (df['total_bearish'].rolling(window=window).sum() + 1)  # Add 1 to avoid division by zero
                    )
                    
                    # Price features
                    features_df[f'price_ma_{window}d'] = df['price_mean'].rolling(window=window).mean()
                    features_df[f'price_std_{window}d'] = df['price_mean'].rolling(window=window).std()
                    features_df[f'price_slope_{window}d'] = df['price_mean'].diff(window) / window
                    
                    # Try to calculate RSI if enough data
                    try:
                        features_df[f'rsi_{window}d'] = self._calculate_rsi(df['price_close'], window)
                    except:
                        pass
            
            # Add volatility
            if len(df) > 1:
                features_df['price_volatility_1d'] = abs(df['price_mean'].pct_change())
                
            # Add MACD (Moving Average Convergence Divergence)
            if len(df) > 26:
                ema12 = df['price_close'].ewm(span=12, adjust=False).mean()
                ema26 = df['price_close'].ewm(span=26, adjust=False).mean()
                features_df['macd'] = ema12 - ema26
                features_df['macd_signal'] = features_df['macd'].ewm(span=9, adjust=False).mean()
            
            # Momentum indicators
            features_df['sentiment_momentum'] = features_df['avg_sentiment'].diff(5)
            
            # Day of week cyclical encoding
            features_df['day_sin'] = np.sin(2 * np.pi * features_df['day_of_week'] / 7)
            features_df['day_cos'] = np.cos(2 * np.pi * features_df['day_of_week'] / 7)
            
            # Correlation features
            if len(df) > 7:
                features_df['sentiment_price_correlation'] = features_df['avg_sentiment'].rolling(window=7).corr(features_df['price_close'])
            
            # Drop rows with missing values
            features_df = features_df.dropna()
            
            # Cache engineered features
            if cache_key:
                self.feature_cache[f"{cache_key}_features"] = features_df
                CACHE_SIZE.set(len(self.feature_cache))
            
            # Get list of feature columns (excluding target variables)
            target_cols = [col for col in features_df.columns if col.startswith('price_change_') or col.startswith('price_direction_')]
            feature_cols = [col for col in features_df.columns if col not in target_cols and col not in ['date', 'avg_sentiment', 'total_bullish', 'total_bearish', 'record_count']]
            
            if horizon:
                FEATURE_COUNT.labels(symbol="all", horizon=str(horizon)).set(len(feature_cols))
            
            logger.info(f"Engineered features: {len(features_df)} samples with {len(features_df.columns)} features")
            return features_df
            
        except Exception as e:
            logger.error(f"Error engineering features: {str(e)}")
            ML_ERRORS.labels(operation="feature_engineering", symbol="all", reason="processing_error").inc()
            return None

    def _calculate_rsi(self, prices, window):
        """Calculate Relative Strength Index"""
        deltas = prices.diff()
        
        # Get rid of the first row, which is NaN since diff loses one row
        deltas = deltas[1:]
        
        # Make the positive gains (up) and negative gains (down) series
        up = deltas.copy()
        up[up < 0] = 0
        down = -1 * deltas.copy()
        down[down < 0] = 0
        
        # Calculate the EWMA
        roll_up = up.ewm(com=window-1, min_periods=window).mean()
        roll_down = down.ewm(com=window-1, min_periods=window).mean()
        
        # Calculate the RSI based on EWMA
        rs = roll_up / roll_down
        rsi = 100.0 - (100.0 / (1.0 + rs))
        
        return rsi

    def train_model(self, symbol, horizon):
        """
        Train a model for a specific symbol and prediction horizon
        
        Parameters:
        symbol (str): Cryptocurrency symbol
        horizon (int): Prediction horizon in days
        
        Returns:
        dict: Model data including the trained model, scaler, and performance metrics
        """
        with TRAIN_DURATION.labels(symbol=symbol, horizon=str(horizon)).time():
            try:
                logger.info(f"Training model for {symbol} with {horizon}-day horizon")
                
                # Get training data
                df = self.get_training_data(symbol)
                if df is None or len(df) < self.min_training_samples:
                    logger.warning(f"Not enough data to train model for {symbol}")
                    ML_ERRORS.labels(operation="training", symbol=symbol, reason="insufficient_data").inc()
                    return None
                
                # Engineer features
                cache_key = f"{symbol}_{horizon}d"
                features_df = self.engineer_features(
                    df, 
                    horizon=horizon,
                    cache_key=cache_key
                )
                
                if features_df is None or len(features_df) < self.min_training_samples:
                    logger.warning(f"Not enough featured data to train model for {symbol}")
                    ML_ERRORS.labels(operation="training", symbol=symbol, reason="insufficient_features").inc()
                    return None
                
                # Prepare features and target
                target_col = f"price_change_{horizon}d"
                feature_cols = [col for col in features_df.columns if col != target_col and not col.startswith('price_change_') and not col.startswith('price_direction_') and col not in ['date', 'avg_sentiment', 'total_bullish', 'total_bearish', 'record_count']]
                
                X = features_df[feature_cols].copy()
                y = features_df[target_col].values
                
                FEATURE_COUNT.labels(symbol=symbol, horizon=str(horizon)).set(len(feature_cols))
                SAMPLE_COUNT.labels(symbol=symbol, horizon=str(horizon)).set(len(X))
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, shuffle=False
                )
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Select and train model
                if self.model_type == 'random_forest':
                    model = RandomForestRegressor(
                        n_estimators=100,
                        max_depth=10,
                        random_state=42,
                        n_jobs=-1
                    )
                elif self.model_type == 'gradient_boosting':
                    model = GradientBoostingRegressor(
                        n_estimators=100,
                        max_depth=5,
                        random_state=42
                    )
                else:
                    model = LinearRegression()
                
                model.fit(X_train_scaled, y_train)
                ML_MODEL_TRAINS.labels(symbol=symbol, horizon=str(horizon), model_type=self.model_type).inc()
                
                # Evaluate model
                y_pred = model.predict(X_test_scaled)
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                MODEL_R2_SCORE.labels(symbol=symbol, horizon=str(horizon)).set(r2)
                MODEL_MAE.labels(symbol=symbol, horizon=str(horizon)).set(mae)
                
                logger.info(f"Model evaluation for {symbol} {horizon}-day horizon:")
                logger.info(f"  Mean Squared Error: {mse:.6f}")
                logger.info(f"  Mean Absolute Error: {mae:.6f}")
                logger.info(f"  R² Score: {r2:.6f}")
                
                # Save model, scaler and feature list
                model_file = os.path.join(self.model_save_path, f"{symbol}_{horizon}d_model.joblib")
                scaler_file = os.path.join(self.model_save_path, f"{symbol}_{horizon}d_scaler.joblib")
                feature_file = os.path.join(self.model_save_path, f"{symbol}_{horizon}d_features.json")
                
                joblib.dump(model, model_file)
                joblib.dump(scaler, scaler_file)
                
                with open(feature_file, 'w') as f:
                    json.dump(feature_cols, f)
                
                # Save feature importances if available
                if hasattr(model, 'feature_importances_'):
                    importance_df = pd.DataFrame({
                        'feature': feature_cols,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    importance_file = os.path.join(self.model_save_path, f"{symbol}_{horizon}d_importance.csv")
                    importance_df.to_csv(importance_file, index=False)
                    
                    logger.info(f"Top 5 important features:")
                    for i, row in importance_df.head(5).iterrows():
                        logger.info(f"  {row['feature']}: {row['importance']:.4f}")
                
                # Update active model count
                self.active_model_count += 1
                ACTIVE_MODELS.set(self.active_model_count)
                
                # Return model data
                return {
                    'model': model,
                    'scaler': scaler,
                    'feature_columns': feature_cols,
                    'metrics': {
                        'mse': mse,
                        'mae': mae,
                        'r2': r2
                    },
                    'trained_at': datetime.now(),
                    'samples': len(features_df)
                }
                
            except Exception as e:
                logger.error(f"Error training model: {str(e)}")
                ML_ERRORS.labels(operation="training", symbol=symbol, reason="model_error").inc()
                return None

    def load_model(self, symbol, horizon):
        """
        Load a trained model from disk
        
        Parameters:
        symbol (str): Cryptocurrency symbol
        horizon (int): Prediction horizon in days
        
        Returns:
        dict: Model data including the model, scaler, feature columns and metrics
        """
        try:
            model_file = os.path.join(self.model_save_path, f"{symbol}_{horizon}d_model.joblib")
            scaler_file = os.path.join(self.model_save_path, f"{symbol}_{horizon}d_scaler.joblib")
            feature_file = os.path.join(self.model_save_path, f"{symbol}_{horizon}d_features.json")
            importance_file = os.path.join(self.model_save_path, f"{symbol}_{horizon}d_importance.csv")
            
            if not os.path.exists(model_file) or not os.path.exists(scaler_file):
                logger.warning(f"Model files for {symbol} {horizon}d not found")
                return None
                
            model = joblib.load(model_file)
            scaler = joblib.load(scaler_file)
            
            try:
                # Load feature columns
                if os.path.exists(feature_file):
                    with open(feature_file, 'r') as f:
                        feature_cols = json.load(f)
                else:
                    # Try to get feature columns from importance file
                    if os.path.exists(importance_file):
                        importance_df = pd.read_csv(importance_file)
                        feature_cols = importance_df['feature'].tolist()
                    else:
                        logger.warning(f"Feature files for {symbol} {horizon}d not found")
                        feature_cols = []
            except Exception as e:
                logger.warning(f"Could not load feature columns: {str(e)}")
                feature_cols = []
                
            # Get model age
            model_timestamp = datetime.fromtimestamp(os.path.getmtime(model_file))
            model_age_days = (datetime.now() - model_timestamp).days
            MODEL_AGE_DAYS.labels(symbol=symbol, horizon=str(horizon)).set(model_age_days)
            
            logger.info(f"Loaded model for {symbol} {horizon}d (age: {model_age_days} days)")
            
            # Return model data
            return {
                'model': model,
                'scaler': scaler,
                'feature_columns': feature_cols,
                'metrics': {},  # Could add metrics from the model if available
                'trained_at': model_timestamp
            }
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            ML_ERRORS.labels(operation="model_load", symbol=symbol, reason="file_error").inc()
            return None

    def predict(self, symbol, horizon, retrain=False):
        """
        Generate a price prediction for a specific symbol and horizon
        
        Parameters:
        symbol (str): Cryptocurrency symbol
        horizon (int): Prediction horizon in days
        retrain (bool): Force model retraining even if a model exists
        
        Returns:
        dict: Prediction results with predicted change, direction, confidence, etc.
        """
        with PREDICTION_DURATION.labels(symbol=symbol, horizon=str(horizon)).time():
            try:
                # Load or train model
                model_data = self.load_model(symbol, horizon)
                
                # Retrain if requested or if model doesn't exist
                if retrain or model_data is None:
                    logger.info(f"Training new model for {symbol} {horizon}d")
                    model_data = self.train_model(symbol, horizon)
                    
                    if model_data is None:
                        logger.warning(f"Could not load or train model for {symbol} {horizon}d")
                        ML_ERRORS.labels(operation="prediction", symbol=symbol, reason="no_model").inc()
                        return None
                
                # Get latest data for prediction
                df = self.get_training_data(symbol, force_refresh=True)
                if df is None or len(df) < max(self.feature_windows):
                    logger.warning(f"Not enough recent data for prediction")
                    ML_ERRORS.labels(operation="prediction", symbol=symbol, reason="insufficient_data").inc()
                    return None
                
                # Engineer features
                features_df = self.engineer_features(
                    df, 
                    horizon=horizon,
                    cache_key=f"{symbol}_{horizon}d_pred"
                )
                
                if features_df is None or len(features_df) == 0:
                    logger.warning(f"Could not engineer features for prediction")
                    ML_ERRORS.labels(operation="prediction", symbol=symbol, reason="feature_error").inc()
                    return None
                
                # Get latest data point for prediction
                latest_data = features_df.iloc[-1:].copy()
                
                # Get feature columns used by the model
                feature_cols = model_data.get('feature_columns', [])
                if not feature_cols:
                    logger.warning(f"No feature columns available for {symbol} {horizon}d model")
                    feature_cols = [col for col in latest_data.columns if col not in ['date', 'avg_sentiment', 'total_bullish', 'total_bearish', 'record_count'] and not col.startswith('price_change_') and not col.startswith('price_direction_')]
                
                # Select features for prediction
                X = latest_data[feature_cols].copy()
                
                # Check for missing features
                for feature in model_data.get('feature_columns', []):
                    if feature not in X.columns:
                        logger.warning(f"Missing feature: {feature}")
                        X[feature] = 0  # Add with default value
                
                # Scale features
                X_scaled = model_data['scaler'].transform(X)
                
                # Make prediction
                prediction = model_data['model'].predict(X_scaled)[0]
                
                # Convert to percentage
                prediction_pct = prediction * 100
                
                # Determine direction
                direction = "UP" if prediction > 0 else "DOWN" if prediction < 0 else "NEUTRAL"
                
                # Calculate prediction confidence based on model R2
                confidence = min(1.0, max(0.1, model_data.get('metrics', {}).get('r2', 0.5)))
                
                # Calculate target price
                current_price = latest_data['price_close'].iloc[0]
                target_price = current_price * (1 + prediction)
                
                # Update metrics
                ML_PREDICTIONS.labels(symbol=symbol, horizon=str(horizon)).inc()
                PREDICTION_VALUES.labels(symbol=symbol, horizon=str(horizon), direction=direction).set(prediction)
                PREDICTION_CONFIDENCE.labels(symbol=symbol, horizon=str(horizon)).set(confidence)
                PREDICTION_DISTRIBUTION.labels(symbol=symbol, horizon=str(horizon)).observe(prediction)
                
                # Create result dict
                result = {
                    'symbol': symbol,
                    'horizon': horizon,
                    'timestamp': datetime.now(),
                    'predicted_change': prediction,
                    'predicted_change_pct': prediction_pct,
                    'direction': direction,
                    'confidence': confidence,
                    'current_price': current_price,
                    'target_price': target_price,
                    'target_date': latest_data['date'].iloc[0] + timedelta(days=horizon),
                    'model_trained_at': model_data.get('trained_at', None)
                }
                
                logger.info(f"Prediction for {symbol} {horizon}d: {prediction:.2%} change (confidence: {confidence:.2f})")
                return result
                
            except Exception as e:
                logger.error(f"Error making prediction: {str(e)}")
                ML_ERRORS.labels(operation="prediction", symbol=symbol, reason="prediction_error").inc()
                return None

    def run_predictions(self, retrain=False):
        """
        Run predictions for all configured symbols and horizons
        
        Parameters:
        retrain (bool): Force retraining of all models
        
        Returns:
        list: List of prediction results
        """
        try:
            start_time = time.time()
            results = []
            
            for symbol in self.symbols:
                for horizon in self.prediction_horizons:
                    try:
                        prediction = self.predict(symbol, horizon, retrain)
                        if prediction:
                            results.append(prediction)
                    except Exception as e:
                        logger.error(f"Error predicting {symbol} {horizon}d: {str(e)}")
            
            # Save predictions to file
            os.makedirs("predictions", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            prediction_file = f"predictions/prediction_{timestamp}.json"
            
            with open(prediction_file, "w") as f:
                json.dump(results, f, indent=2, default=str)
            
            total_duration = time.time() - start_time
            logger.info(f"Completed all predictions in {total_duration:.2f} seconds")
            
            return results
            
        except Exception as e:
            logger.error(f"Error running predictions: {str(e)}")
            ML_ERRORS.labels(operation="batch_prediction", symbol="all", reason="batch_error").inc()
            return []

    def start_scheduled_training(self):
        """Start scheduled training and prediction jobs"""
        logger.info(f"Starting scheduled training with {self.retrain_interval_days} day interval")
        
        # Initial run with training
        self.run_predictions(retrain=True)
        
        # Schedule periodic training
        schedule.every(self.retrain_interval_days).days.do(
            lambda: self.run_predictions(retrain=True)
        )
        
        # Schedule daily predictions without retraining
        schedule.every().day.at("00:01").do(
            lambda: self.run_predictions(retrain=False)
        )
        
        # Run the scheduler
        try:
            while True:
                schedule.run_pending()
                time.sleep(3600)  # Check every hour
        except KeyboardInterrupt:
            logger.info("Scheduled training stopped by user")

    def clear_cache(self):
        """Clear the feature cache"""
        self.feature_cache = {}
        CACHE_SIZE.set(0)
        logger.info("Feature cache cleared")

    def vacuum_models(self, keep_days=90):
        """Remove old model files to save disk space"""
        try:
            logger.info(f"Running model vacuum - keeping models from last {keep_days} days")
            
            cutoff_time = datetime.now() - timedelta(days=keep_days)
            removed_count = 0
            kept_count = 0
            
            if not os.path.exists(self.model_save_path):
                logger.warning(f"Model directory {self.model_save_path} does not exist")
                return removed_count, kept_count
                
            for filename in os.listdir(self.model_save_path):
                file_path = os.path.join(self.model_save_path, filename)
                
                # Skip directories
                if not os.path.isfile(file_path):
                    continue
                    
                # Check file age
                file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                if file_time < cutoff_time:
                    os.remove(file_path)
                    removed_count += 1
                else:
                    kept_count += 1
                    
            logger.info(f"Model vacuum complete: removed {removed_count} files, kept {kept_count} files")
            return removed_count, kept_count
            
        except Exception as e:
            logger.error(f"Error during model vacuum: {e}")
            ML_ERRORS.labels(operation="vacuum", symbol="all", reason="file_error").inc()
            return 0, 0

# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentiment-based ML price predictor")
    parser.add_argument("--train", action="store_true", help="Train models")
    parser.add_argument("--predict", action="store_true", help="Run predictions")
    parser.add_argument("--schedule", action="store_true", help="Start scheduled training")
    parser.add_argument("--symbol", type=str, help="Symbol to use (default: use all configured symbols)")
    parser.add_argument("--horizon", type=int, help="Prediction horizon in days (default: use all configured horizons)")
    parser.add_argument("--clear-cache", action="store_true", help="Clear the feature cache")
    parser.add_argument("--vacuum", action="store_true", help="Remove old model files")
    parser.add_argument("--vacuum-days", type=int, default=90, help="Keep models from last N days")
    
    args = parser.parse_args()
    
    ml = SentimentML()
    
    if args.clear_cache:
        ml.clear_cache()
        
    if args.vacuum:
        ml.vacuum_models(keep_days=args.vacuum_days)
        
    if args.train:
        if args.symbol and args.horizon:
            ml.train_model(args.symbol, args.horizon)
        elif args.symbol:
            for horizon in ml.prediction_horizons:
                ml.train_model(args.symbol, horizon)
        elif args.horizon:
            for symbol in ml.symbols:
                ml.train_model(symbol, args.horizon)
        else:
            for symbol in ml.symbols:
                for horizon in ml.prediction_horizons:
                    ml.train_model(symbol, horizon)
                    
    if args.predict:
        if args.symbol and args.horizon:
            prediction = ml.predict(args.symbol, args.horizon)
            print(f"\nPrediction for {args.symbol} {args.horizon}d: {prediction}")
        elif args.symbol:
            for horizon in ml.prediction_horizons:
                prediction = ml.predict(args.symbol, horizon)
                print(f"\nPrediction for {args.symbol} {horizon}d: {prediction}")
        elif args.horizon:
            for symbol in ml.symbols:
                prediction = ml.predict(symbol, args.horizon)
                print(f"\nPrediction for {symbol} {args.horizon}d: {prediction}")
        else:
            results = ml.run_predictions()
            print("\nPredictions:")
            for result in results:
                print(f"{result['symbol']} {result['horizon']}d: {result['predicted_change']:.2%} change ({result['direction']}) with {result['confidence']:.2f} confidence")
                print(f"  Current price: ${result['current_price']:.2f}, Target price: ${result['target_price']:.2f} by {result['target_date'].strftime('%Y-%m-%d')}")
                
    if args.schedule:
        ml.start_scheduled_training()
        
    if not any([args.train, args.predict, args.schedule, args.clear_cache, args.vacuum]):
        parser.print_help()