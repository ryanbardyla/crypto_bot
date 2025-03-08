import os
import json
import logging
import numpy as np
import pandas as pd
import schedule
import time
import argparse
import joblib
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logging.basicConfig(
        logging.FileHandler("sentiment_ml.log"),
        logging.StreamHandler()
)
logger = logging.getLogger("SentimentML")

class SentimentML:
    def __init__(self, config_file="sentiment_ml_config.json"):
        self.load_config(config_file)
        self.setup_database()
        
    def load_config(self, config_file):
        try:
            with open(config_file, "r") as f:
                self.config = json.load(f)
                
            self.model_type = self.config.get("model_type", "random_forest")
            self.prediction_horizons = self.config.get("prediction_horizons", [1, 3, 7])
            self.feature_windows = self.config.get("feature_windows", [1, 3, 7, 14])
            self.min_training_samples = self.config.get("min_training_samples", 30)
            self.retrain_interval_days = self.config.get("retrain_interval_days", 7)
            self.model_save_path = self.config.get("model_save_path", "ml_models")
            self.symbols = self.config.get("symbols", ["BTC", "ETH"])
            self.db_path = self.config.get("db_path", "sqlite:///sentiment_database.db")
            
            logger.info("Configuration loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            
    def setup_database(self):
        try:
            self.engine = create_engine(self.db_path)
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Failed to set up database: {str(e)}")
            
    def get_training_data(self, symbol):
        try:
            # Base sentiment query - modified to work with either Twitter or YouTube or both
            sentiment_query = """
            SELECT 
                date(processed_date) as date, 
                avg(combined_score) as avg_sentiment,
                sum(bullish_keywords) as total_bullish,
                sum(bearish_keywords) as total_bearish,
                count(*) as record_count
            FROM sentiment_records
            GROUP BY date(processed_date)
            ORDER BY date(processed_date)
            """
            
            with self.engine.connect() as conn:
                sentiment_df = pd.read_sql(sentiment_query, conn)
            
            if sentiment_df.empty:
                logger.warning(f"No sentiment data available for {symbol}")
                return None
                
            sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
            
            # Load price data
            price_df = pd.DataFrame()
            try:
                with open("price_history.json", "r") as f:
                    price_history = json.load(f)
                    price_data = price_history.get(symbol, [])
                    
                    if not price_data:
                        logger.warning(f"No price data found for {symbol}")
                        return None
                        
                    price_df = pd.DataFrame(price_data)
                    price_df['timestamp'] = pd.to_datetime(price_df['timestamp'])
                    price_df['date'] = pd.to_datetime(price_df['timestamp']).dt.date
                    price_df['date'] = pd.to_datetime(price_df['date'])
                    
                    # Aggregate price data by date
                    price_df = price_df.groupby('date').agg({
                        'price': ['open', 'close', 'high', 'low', 'mean']
                    }).reset_index()
                    
                    # Flatten multi-level columns
                    price_df.columns = ['date' if col[0] == 'date' else f"price_{col[1]}" for col in price_df.columns]
            except Exception as e:
                logger.error(f"Error loading price data: {str(e)}")
                return None
            
            # Merge sentiment and price data
            df = pd.merge(sentiment_df, price_df, on='date', how='inner')
            df = df.sort_values('date')
            
            # Add target variables for each prediction horizon
            for days in self.prediction_horizons:
                df[f'price_change_{days}d'] = df['price_close'].pct_change(periods=days).shift(-days)
                df[f'price_direction_{days}d'] = np.sign(df[f'price_change_{days}d'])
            
            # Drop rows with missing target values
            df = df.dropna(subset=[f'price_change_{days}d' for days in self.prediction_horizons])
            
            logger.info(f"Prepared training data for {symbol}: {len(df)} samples")
            return df
            
        except Exception as e:
            logger.error(f"Error preparing training data: {str(e)}")
            return None
            
    def engineer_features(self, df):
        try:
            features_df = df.copy()
            
            # Create various features for each time window
            for window in self.feature_windows:
                if len(df) > window:
                    # Sentiment features
                    features_df[f'sentiment_ma_{window}d'] = df['avg_sentiment'].rolling(window=window).mean()
                    features_df[f'sentiment_std_{window}d'] = df['avg_sentiment'].rolling(window=window).std()
                    features_df[f'sentiment_slope_{window}d'] = df['avg_sentiment'].diff(window) / window
                    
                    # Keyword ratio (bullish/bearish)
                    features_df[f'keyword_ratio_{window}d'] = (
                        df['total_bullish'].rolling(window=window).sum() / 
                        (df['total_bearish'].rolling(window=window).sum() + 1)  # Add 1 to avoid division by zero
                    )
                    
                    # Price features
                    features_df[f'price_ma_{window}d'] = df['price_mean'].rolling(window=window).mean()
                    features_df[f'price_std_{window}d'] = df['price_mean'].rolling(window=window).std()
                    features_df[f'price_slope_{window}d'] = df['price_mean'].diff(window) / window
            
            # Add recent volatility
            if len(df) > 1:
                features_df['price_volatility_1d'] = abs(df['price_mean'].pct_change())
            
            # Drop rows with missing values
            features_df = features_df.dropna()
            
            logger.info(f"Engineered features: {len(features_df)} samples with {len(features_df.columns)} features")
            return features_df
            
        except Exception as e:
            logger.error(f"Error engineering features: {str(e)}")
            return None
            
    def train_model(self, symbol, horizon):
        try:
            logger.info(f"Training model for {symbol} with {horizon}-day horizon")
            
            # Get and prepare data
            df = self.get_training_data(symbol)
            if df is None or len(df) < self.min_training_samples:
                logger.warning(f"Not enough data to train model for {symbol}")
                return None
            
            # Engineer features
            features_df = self.engineer_features(df)
            if features_df is None or len(features_df) < self.min_training_samples:
                logger.warning(f"Not enough featured data to train model for {symbol}")
                return None
            
            # Prepare training data
            X = features_df.drop(columns=['date', 'avg_sentiment', 'total_bullish', 'total_bearish', 'record_count'] + 
                               [f'price_change_{d}d' for d in self.prediction_horizons] + 
                               [f'price_direction_{d}d' for d in self.prediction_horizons])
            
            y = features_df[f'price_change_{horizon}d']
            
            # Split into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Standardize features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model based on selected type
            if self.model_type == "random_forest":
                model = RandomForestRegressor(
                    n_estimators=100, 
                    max_depth=None, 
                    min_samples_split=2, 
                    random_state=42
                )
            elif self.model_type == "gradient_boosting":
                model = GradientBoostingRegressor(
                    n_estimators=100, 
                    learning_rate=0.1, 
                    max_depth=3, 
                    random_state=42
                )
            else:
                model = LinearRegression()
            
            model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            logger.info(f"Model evaluation for {symbol} {horizon}-day horizon:")
            logger.info(f"  Mean Squared Error: {mse:.6f}")
            logger.info(f"  Mean Absolute Error: {mae:.6f}")
            logger.info(f"  R² Score: {r2:.6f}")
            
            # Save model and related data
            os.makedirs(self.model_save_path, exist_ok=True)
            model_file = os.path.join(self.model_save_path, f"{symbol}_{horizon}d_model.joblib")
            scaler_file = os.path.join(self.model_save_path, f"{symbol}_{horizon}d_scaler.joblib")
            
            joblib.dump(model, model_file)
            joblib.dump(scaler, scaler_file)
            
            # Save feature importance if available
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                importance_file = os.path.join(self.model_save_path, f"{symbol}_{horizon}d_importance.csv")
                importance_df.to_csv(importance_file, index=False)
                
                logger.info(f"Top 5 important features:")
                for i, row in importance_df.head(5).iterrows():
                    logger.info(f"  {row['feature']}: {row['importance']:.4f}")
            
            return {
                'model': model,
                'scaler': scaler,
                'feature_columns': X.columns.tolist(),
                'metrics': {'mse': mse, 'mae': mae, 'r2': r2},
                'trained_at': datetime.now(),
                'samples': len(features_df)
            }
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return None
            
    def load_model(self, symbol, horizon):
        try:
            model_file = os.path.join(self.model_save_path, f"{symbol}_{horizon}d_model.joblib")
            scaler_file = os.path.join(self.model_save_path, f"{symbol}_{horizon}d_scaler.joblib")
            importance_file = os.path.join(self.model_save_path, f"{symbol}_{horizon}d_importance.csv")
            
            if not os.path.exists(model_file) or not os.path.exists(scaler_file):
                logger.warning(f"Model files for {symbol} {horizon}d not found")
                return None
            
            model = joblib.load(model_file)
            scaler = joblib.load(scaler_file)
            
            feature_cols = []
            try:
                if os.path.exists(importance_file):
                    importance_df = pd.read_csv(importance_file)
                    feature_cols = importance_df['feature'].tolist()
                else:
                    logger.warning(f"Feature importance file for {symbol} {horizon}d not found")
            except Exception as e:
                logger.warning(f"Could not load feature columns: {str(e)}")
            
            logger.info(f"Loaded model for {symbol} {horizon}d")
            
            return {
                'model': model,
                'scaler': scaler,
                'feature_columns': feature_cols,
                'trained_at': datetime.fromtimestamp(os.path.getmtime(model_file)),
                'metrics': {}  # Metrics not saved when loading
            }
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return None
            
    def predict(self, symbol, horizon, retrain=False):
        try:
            # Load or train model
            model_data = self.load_model(symbol, horizon)
            if model_data is None or retrain:
                logger.info(f"Training new model for {symbol} {horizon}d")
                model_data = self.train_model(symbol, horizon)
                
            if model_data is None:
                logger.warning(f"Could not load or train model for {symbol} {horizon}d")
                return None
            
            # Get recent data for prediction
            df = self.get_training_data(symbol)
            if df is None or len(df) < max(self.feature_windows):
                logger.warning(f"Not enough recent data for prediction")
                return None
            
            # Engineer features
            features_df = self.engineer_features(df)
            if features_df is None:
                logger.warning(f"Could not engineer features for prediction")
                return None
            
            # Prepare prediction data (most recent data point)
            latest_data = features_df.iloc[-1:].copy()
            
            # Ensure all required features are present
            X = latest_data.drop(columns=['date', 'avg_sentiment', 'total_bullish', 'total_bearish', 'record_count'] + 
                               [f'price_change_{d}d' for d in self.prediction_horizons] + 
                               [f'price_direction_{d}d' for d in self.prediction_horizons])
            
            for feature in model_data['feature_columns']:
                if feature not in X.columns:
                    logger.warning(f"Missing feature: {feature}")
            
            # Make prediction
            X_scaled = model_data['scaler'].transform(X)
            prediction = model_data['model'].predict(X_scaled)[0]
            
            # Calculate confidence based on model R² score
            confidence = min(1.0, max(0.1, model_data.get('metrics', {}).get('r2', 0.5)))
            
            # Current price from most recent data
            current_price = df.iloc[-1]['price_close']
            
            # Predicted price
            target_price = current_price * (1 + prediction)
            
            # Determine direction
            direction = "up" if prediction > 0 else "down"
            
            result = {
                'symbol': symbol,
                'horizon': horizon,
                'predicted_change': prediction,
                'predicted_change_pct': prediction * 100,
                'current_price': current_price,
                'target_price': target_price,
                'direction': direction,
                'confidence': confidence,
                'timestamp': datetime.now(),
                'target_date': latest_data['date'].iloc[0] + timedelta(days=horizon),
                'model_trained_at': model_data.get('trained_at', None)
            }
            
            logger.info(f"Prediction for {symbol} {horizon}d: {prediction:.2%} change (confidence: {confidence:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return None
            
    def run_predictions(self, retrain=False):
        """Run predictions for all symbols and horizons"""
        results = []
        
        for symbol in self.symbols:
            for horizon in self.prediction_horizons:
                prediction = self.predict(symbol, horizon, retrain)
                if prediction:
                    results.append(prediction)
                    
        # Save results
        if results:
            os.makedirs("predictions", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(f"predictions/prediction_{timestamp}.json", "w") as f:
                json.dump(results, f, indent=2, default=str)
                
        return results
    
    def start_scheduled_training(self):
        """Start scheduled model training and prediction"""
        logger.info(f"Starting scheduled training with {self.retrain_interval_days} day interval")
        
        # Run once immediately
        self.run_predictions(retrain=True)
        
        # Schedule regular retraining
        schedule.every(self.retrain_interval_days).days.do(
            lambda: self.run_predictions(retrain=True)
        )
        
        # Schedule daily predictions
        schedule.every().day.at("00:01").do(
            lambda: self.run_predictions(retrain=False)
        )
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(3600)  # Check every hour
        except KeyboardInterrupt:
            logger.info("Scheduled training stopped by user")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentiment-based ML price predictor")
    parser.add_argument("--train", action="store_true", help="Train models")
    parser.add_argument("--predict", action="store_true", help="Run predictions")
    parser.add_argument("--schedule", action="store_true", help="Start scheduled training")
    parser.add_argument("--symbol", type=str, help="Symbol to use (default: use all configured symbols)")
    parser.add_argument("--horizon", type=int, help="Prediction horizon in days (default: use all configured horizons)")
    
    args = parser.parse_args()
    ml = SentimentML()
    
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
    
    elif args.predict:
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
    
    elif args.schedule:
        ml.start_scheduled_training()
    
    else:
        parser.print_help()