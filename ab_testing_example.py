# ab_testing_example.py

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Any

# Import our A/B testing frameworks
from ab_testing_framework import ABTestingFramework
from trading_strategy_ab_testing import TradingStrategyABTesting
from sentiment_metrics_ab_testing import SentimentAnalysisABTesting
from enhanced_performance_metrics import EnhancedPerformanceMetrics
from metrics_collector import MetricsCollector

# Import bot components (these would be your actual components)
from multi_api_price_fetcher import CryptoPriceFetcher
from paper_trader import PaperTrader
from simple_backtester import SimpleBacktester
from sentiment_analyzer import SentimentAnalyzer
from crypto_analyzer import CryptoAnalyzer
from trading_strategy import TradingStrategy
from advanced_strategy import AdvancedStrategy

def setup_ab_testing():
    """Set up A/B testing for all components"""
    
    # Initialize components
    price_fetcher = CryptoPriceFetcher()
    paper_trader = PaperTrader(initial_capital=10000)
    backtest_engine = SimpleBacktester()
    sentiment_analyzer = SentimentAnalyzer()
    crypto_analyzer = CryptoAnalyzer(price_fetcher)
    
    # Set up metrics collector
    metrics_collector = MetricsCollector()
    metrics_collector.register_component('price_fetcher', price_fetcher)
    metrics_collector.register_component('paper_trader', paper_trader)
    metrics_collector.register_component('sentiment_analyzer', sentiment_analyzer)
    metrics_collector.register_component('crypto_analyzer', crypto_analyzer)
    
    # Start metrics collection
    metrics_collector.start_collection(interval=300)  # Collect every 5 minutes
    
    # Set up Trading Strategy A/B Testing
    strategy_testing = TradingStrategyABTesting(
        price_fetcher=price_fetcher,
        paper_trader=paper_trader,
        backtest_engine=backtest_engine
    )
    
    # Register different trading strategies for testing
    def simple_ma_strategy(df, fast_period=10, slow_period=30):
        """Simple moving average crossover strategy"""
        df['fast_ma'] = df['price'].rolling(window=fast_period).mean()
        df['slow_ma'] = df['price'].rolling(window=slow_period).mean()
        
        # Initialize signal columns
        df['signal'] = 0
        df['signal_strength'] = 0
        
        # Generate signals based on MA crossover
        for i in range(1, len(df)):
            if df['fast_ma'].iloc[i] > df['slow_ma'].iloc[i] and df['fast_ma'].iloc[i-1] <= df['slow_ma'].iloc[i-1]:
                df.loc[df.index[i], 'signal'] = 1
                df.loc[df.index[i], 'signal_strength'] = 0.5
            elif df['fast_ma'].iloc[i] < df['slow_ma'].iloc[i] and df['fast_ma'].iloc[i-1] >= df['slow_ma'].iloc[i-1]:
                df.loc[df.index[i], 'signal'] = -1
                df.loc[df.index[i], 'signal_strength'] = 0.5
                
        return df
    
    def rsi_strategy(df, rsi_period=14, overbought=70, oversold=30):
        """RSI-based strategy"""
        # Calculate RSI
        delta = df['price'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=rsi_period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=rsi_period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Initialize signal columns
        df['signal'] = 0
        df['signal_strength'] = 0
        
        # Generate signals based on RSI
        for i in range(1, len(df)):
            if df['rsi'].iloc[i] < oversold and df['rsi'].iloc[i-1] >= oversold:
                df.loc[df.index[i], 'signal'] = 1
                df.loc[df.index[i], 'signal_strength'] = 0.7
            elif df['rsi'].iloc[i] > overbought and df['rsi'].iloc[i-1] <= overbought:
                df.loc[df.index[i], 'signal'] = -1
                df.loc[df.index[i], 'signal_strength'] = 0.7
                
        return df
    
    def combined_strategy(df, fast_period=10, slow_period=30, rsi_period=14, overbought=70, oversold=30):
        """Combined MA and RSI strategy"""
        # Calculate MAs
        df['fast_ma'] = df['price'].rolling(window=fast_period).mean()
        df['slow_ma'] = df['price'].rolling(window=slow_period).mean()
        
        # Calculate RSI
        delta = df['price'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=rsi_period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=rsi_period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Initialize signal columns
        df['signal'] = 0
        df['signal_strength'] = 0
        
        # Generate signals based on both indicators
        for i in range(1, len(df)):
            # MA signals
            ma_signal = 0
            if df['fast_ma'].iloc[i] > df['slow_ma'].iloc[i] and df['fast_ma'].iloc[i-1] <= df['slow_ma'].iloc[i-1]:
                ma_signal = 1
            elif df['fast_ma'].iloc[i] < df['slow_ma'].iloc[i] and df['fast_ma'].iloc[i-1] >= df['slow_ma'].iloc[i-1]:
                ma_signal = -1
                
            # RSI signals
            rsi_signal = 0
            if df['rsi'].iloc[i] < oversold:
                rsi_signal = 1
            elif df['rsi'].iloc[i] > overbought:
                rsi_signal = -1
                
            # Combined signal
            # Only generate a signal if both indicators agree
            if ma_signal == rsi_signal and ma_signal != 0:
                df.loc[df.index[i], 'signal'] = ma_signal
                df.loc[df.index[i], 'signal_strength'] = 0.8  # Stronger signal when both agree
                
        return df
    
    # Register strategies
    strategy_testing.register_strategy("simple_ma", simple_ma_strategy)
    strategy_testing.register_strategy("rsi", rsi_strategy)
    strategy_testing.register_strategy("combined", combined_strategy)
    
    # Create a test for trading strategies
    strategy_test = strategy_testing.create_strategy_test(
        test_id="strategy_test_2025_mar",
        description="Testing different trading strategies for March 2025",
        strategy_variants=[
            {
                "id": "ma_default",
                "strategy_id": "simple_ma",
                "description": "Default MA Crossover (10/30)",
                "fast_period": 10,
                "slow_period": 30
            },
            {
                "id": "ma_short",
                "strategy_id": "simple_ma",
                "description": "Short-term MA Crossover (5/15)",
                "fast_period": 5,
                "slow_period": 15
            },
            {
                "id": "rsi_default",
                "strategy_id": "rsi",
                "description": "Default RSI (14/70/30)",
                "rsi_period": 14,
                "overbought": 70,
                "oversold": 30
            },
            {
                "id": "rsi_sensitive",
                "strategy_id": "rsi",
                "description": "More Sensitive RSI (10/65/35)",
                "rsi_period": 10,
                "overbought": 65,
                "oversold": 35
            },
            {
                "id": "combined_default",
                "strategy_id": "combined",
                "description": "Combined MA/RSI with default settings",
                "fast_period": 10,
                "slow_period": 30,
                "rsi_period": 14,
                "overbought": 70,
                "oversold": 30
            }
        ],
        symbols=["BTC", "ETH", "SOL"],
        test_duration_days=14,
        metrics=["return_pct", "win_rate", "profit_factor", "max_drawdown", "expectancy"]
    )
    
    print(f"Created strategy A/B test: {strategy_test['test_id']}")
    
    # Set up Sentiment Analysis A/B Testing
    sentiment_testing = SentimentAnalysisABTesting()
    
    # Register different sentiment analyzers for testing
    def standard_sentiment_analyzer(text, **kwargs):
        """Standard sentiment analyzer using default settings"""
        return sentiment_analyzer.analyze_text(text)
    
    def enhanced_sentiment_analyzer(text, weight_factor=0.7, threshold=1.5, **kwargs):
        """Enhanced sentiment analyzer with custom weights and thresholds"""
        # Get base sentiment
        base_sentiment = sentiment_analyzer.analyze_text(text)
        
        # Apply custom weighting
        vader_sentiment = base_sentiment.get("vader_sentiment", {})
        keyword_sentiment = base_sentiment.get("keyword_sentiment", 0)
        
        # Adjust the weighting between VADER and keyword sentiment
        combined_score = (
            weight_factor * vader_sentiment.get("compound", 0) * 10 +
            (1 - weight_factor) * keyword_sentiment
        )
        
        # Apply threshold
        if abs(combined_score) < threshold:
            combined_score = 0  # Neutral if below threshold
        
        # Update the result
        result = dict(base_sentiment)
        result["combined_score"] = combined_score
        
        return result
    
    def crypto_focused_analyzer(text, crypto_weight=2.0, market_terms=None, **kwargs):
        """Crypto-focused analyzer with emphasis on market-specific terms"""
        if market_terms is None:
            market_terms = {
                "bullish": 3.0,
                "bearish": -3.0,
                "moon": 2.0,
                "dump": -2.0,
                "hodl": 1.5,
                "fud": -2.0,
                "buy": 1.0,
                "sell": -1.0,
                "long": 1.0,
                "short": -1.0,
                "pump": 1.5,
                "correction": -1.0
            }
            
        # Get base sentiment
        base_sentiment = sentiment_analyzer.analyze_text(text)
        
        # Add extra weight for crypto-specific terms
        extra_score = 0
        text_lower = text.lower()
        
        for term, weight in market_terms.items():
            if term in text_lower:
                count = text_lower.count(term)
                extra_score += count * weight
        
        # Scale the extra score and combine with base sentiment
        scaled_extra = extra_score / 10  # Scale to roughly -10 to 10 range
        
        # Combine original score with extra score
        combined_score = base_sentiment.get("combined_score", 0) + (scaled_extra * crypto_weight)
        
        # Cap at -10 to 10 range
        combined_score = max(-10, min(10, combined_score))
        
        # Update the result
        result = dict(base_sentiment)
        result["combined_score"] = combined_score
        result["market_term_score"] = scaled_extra
        
        return result
    
    # Register analyzers
    sentiment_testing.register_analyzer("standard", standard_sentiment_analyzer)
    sentiment_testing.register_analyzer("enhanced", enhanced_sentiment_analyzer)
    sentiment_testing.register_analyzer("crypto_focused", crypto_focused_analyzer)
    
    # Create a test for sentiment analyzers
    sentiment_test = sentiment_testing.create_sentiment_test(
        test_id="sentiment_test_2025_mar",
        description="Comparing sentiment analyzer approaches for March 2025",
        analyzer_variants=[
            {
                "id": "standard",
                "analyzer_id": "standard",
                "description": "Standard sentiment analyzer with default settings"
            },
            {
                "id": "enhanced_balanced",
                "analyzer_id": "enhanced",
                "description": "Enhanced analyzer with balanced weights",
                "weight_factor": 0.6,
                "threshold": 1.5
            },
            {
                "id": "enhanced_keyword",
                "analyzer_id": "enhanced",
                "description": "Enhanced analyzer with emphasis on keywords",
                "weight_factor": 0.3,
                "threshold": 1.0
            },
            {
                "id": "crypto_focused_standard",
                "analyzer_id": "crypto_focused",
                "description": "Crypto-focused analyzer with standard weights",
                "crypto_weight": 1.5
            },
            {
                "id": "crypto_focused_heavy",
                "analyzer_id": "crypto_focused",
                "description": "Crypto-focused analyzer with heavy market term weights",
                "crypto_weight": 3.0
            }
        ],
        data_sources=["youtube", "twitter"],
        test_duration_days=7,
        metrics=["accuracy", "correlation", "mae", "precision_positive", "recall_positive", "f1_positive"]
    )
    
    print(f"Created sentiment A/B test: {sentiment_test['test_id']}")
    
    # Return all testing objects for further use
    return {
        "metrics_collector": metrics_collector,
        "strategy_testing": strategy_testing,
        "sentiment_testing": sentiment_testing
    }

def run_backtest_evaluation(strategy_testing):
    """Run backtest evaluation of all strategy variants"""
    
    # Run backtest comparison
    results = strategy_testing.run_backtest_comparison(
        test_id="strategy_test_2025_mar",
        historical_days=60,
        initial_capital=10000
    )
    
    # Print summary of results
    print("\nBacktest Results Summary:")
    print("=" * 50)
    
    for symbol, symbol_results in results["results"].items():
        print(f"\nSymbol: {symbol}")
        print("-" * 30)
        
        for variant_id, metrics in symbol_results.items():
            print(f"  Variant: {variant_id}")
            print(f"    Return: {metrics['return_pct']:.2f}%")
            print(f"    Win Rate: {metrics['win_rate']:.2f}%")
            print(f"    Profit Factor: {metrics['profit_factor']:.2f}")
            print(f"    Max Drawdown: {metrics['max_drawdown']:.2f}%")
            print(f"    Expectancy: {metrics['expectancy']:.2f}")
    
    # Generate test report
    strategy_testing.ab_framework.generate_test_report(
        test_id="strategy_test_2025_mar",
        path="reports/strategy_backtest_results.html"
    )
    
    # Get best strategy
    best_strategy = strategy_testing.get_best_strategy("strategy_test_2025_mar")
    print("\nBest Strategy:")
    print(f"  Variant: {best_strategy['variant_id']}")
    print(f"  Strategy: {best_strategy['strategy_id']}")
    print(f"  Return: {best_strategy['performance']['return_pct']:.2f}%")
    
    return results

def evaluate_sentiment_analyzers(sentiment_testing):
    """Evaluate sentiment analyzer variants on labeled data"""
    
    # Load some labeled test data
    # In a real implementation, you would have a set of labeled data
    labeled_data = [
        {"text": "Bitcoin is breaking out! I think we're going to see new all time highs very soon.", "true_label": 1},
        {"text": "The current price action is concerning. BTC could crash further.", "true_label": -1},
        {"text": "ETH remains in a consolidation pattern, waiting for the next move.", "true_label": 0},
        {"text": "I'm super bullish on cryptocurrencies for the long term. HODL!", "true_label": 1},
        {"text": "This market is about to crash. Sell everything and wait for better prices.", "true_label": -1}
        # Add more labeled examples...
    ]
    
    # Evaluate on YouTube data
    youtube_results = sentiment_testing.evaluate_on_labeled_data(
        test_id="sentiment_test_2025_mar",
        labeled_data=labeled_data,
        data_source="youtube"
    )
    
    # Print summary of results
    print("\nSentiment Analyzer Evaluation Results (YouTube):")
    print("=" * 50)
    
    for variant_id, metrics in youtube_results["results"].items():
        print(f"  Variant: {variant_id}")
        print(f"    Accuracy: {metrics.get('accuracy', 0):.2f}")
        print(f"    Correlation: {metrics.get('correlation', 0):.2f}")
        print(f"    MAE: {metrics.get('mae', 0):.2f}")
        print(f"    F1 (Positive): {metrics.get('f1_positive', 0):.2f}")
    
    # Generate test report
    sentiment_testing.ab_framework.generate_test_report(
        test_id="sentiment_test_2025_mar",
        path="reports/sentiment_evaluation_results.html"
    )
    
    # Get best analyzer
    best_analyzer = sentiment_testing.get_best_sentiment_analyzer("sentiment_test_2025_mar")
    print("\nBest Sentiment Analyzer:")
    print(f"  Variant: {best_analyzer['variant_id']}")
    print(f"  Analyzer: {best_analyzer['analyzer_id']}")
    print(f"  Accuracy: {best_analyzer['performance']['accuracy']:.2f}")
    
    return youtube_results

if __name__ == "__main__":
    # Create reports directory
    os.makedirs("reports", exist_ok=True)
    
    # Setup A/B testing
    testing = setup_ab_testing()
    
    # Ask user what to test
    print("\nWhat would you like to evaluate?")
    print("1. Run trading strategy backtest comparison")
    print("2. Evaluate sentiment analyzers")
    print("3. Both")
    choice = input("Enter your choice (1-3): ")
    
    if choice == "1" or choice == "3":
        run_backtest_evaluation(testing["strategy_testing"])
        
    if choice == "2" or choice == "3":
        evaluate_sentiment_analyzers(testing["sentiment_testing"])
    
    # Generate comprehensive performance report
    testing["metrics_collector"].generate_report("reports/performance_report.html")
    
    print("\nEvaluation complete! Reports have been saved to the 'reports' directory.")