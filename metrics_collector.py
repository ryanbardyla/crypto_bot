# metrics_collector.py

import os
import json
import time
import psutil
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any
from enhanced_performance_metrics import EnhancedPerformanceMetrics

class MetricsCollector:
    """
    Collects performance metrics from various components of the trading bot
    """
    def __init__(self, metrics_dir: str = "metrics_data"):
        """
        Initialize the metrics collector
        
        Args:
            metrics_dir: Directory to store metrics data
        """
        self.metrics_dir = metrics_dir
        os.makedirs(metrics_dir, exist_ok=True)
        
        self.performance_metrics = EnhancedPerformanceMetrics(data_dir=metrics_dir)
        self.collection_thread = None
        self.collecting = False
        self.collection_interval = 60  # seconds
        
        # Store component references
        self.components = {}
        
    def register_component(self, component_name: str, component: Any) -> None:
        """
        Register a component for metrics collection
        
        Args:
            component_name: Name of the component
            component: Component instance
        """
        self.components[component_name] = component
        print(f"Registered component for metrics collection: {component_name}")
    
    def collect_system_metrics(self, component_name: str) -> Dict:
        """
        Collect system-level metrics for a component
        
        Args:
            component_name: Name of the component
            
        Returns:
            Dict containing system metrics
        """
        # Get CPU and memory usage for the current process
        process = psutil.Process(os.getpid())
        cpu_usage = process.cpu_percent(interval=1.0)
        memory_usage = process.memory_info().rss / 1024 / 1024  # Convert to MB
        
        # Get component-specific metrics if available
        component = self.components.get(component_name)
        api_latency = 0
        execution_time = 0
        
        if component:
            if hasattr(component, 'get_api_latency'):
                api_latency = component.get_api_latency()
                
            if hasattr(component, 'get_execution_time'):
                execution_time = component.get_execution_time()
        
        # Record in performance metrics
        metrics = self.performance_metrics.update_system_metrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            api_latency=api_latency,
            execution_time=execution_time,
            component=component_name
        )
        
        return metrics
    
    def collect_trading_metrics(self, symbol: str, period: str, strategy_name: str = "default") -> Dict:
        """
        Collect trading metrics from the paper trader or live trader
        
        Args:
            symbol: Trading symbol
            period: Time period (e.g., '1d', '7d', '30d')
            strategy_name: Name of the trading strategy
            
        Returns:
            Dict containing trading metrics
        """
        # Get paper trader or live trader component
        trader = self.components.get('paper_trader') or self.components.get('live_trader')
        
        if not trader:
            print("No trader component registered for metrics collection")
            return {}
            
        # Get trading metrics
        trades = []
        initial_capital = 10000
        final_capital = initial_capital
        
        if hasattr(trader, 'get_trades'):
            trades = trader.get_trades(symbol)
            
        if hasattr(trader, 'get_initial_capital'):
            initial_capital = trader.get_initial_capital()
            
        if hasattr(trader, 'get_current_value'):
            final_capital = trader.get_current_value()
        
        # Record in performance metrics
        metrics = self.performance_metrics.update_trading_metrics(
            symbol=symbol,
            period=period,
            trades=trades,
            initial_capital=initial_capital,
            final_capital=final_capital,
            strategy_name=strategy_name
        )
        
        return metrics
    
    def collect_sentiment_metrics(self, source: str, period: str = "7d") -> Dict:
        """
        Collect sentiment analysis metrics
        
        Args:
            source: Source of sentiment data (e.g., 'youtube', 'twitter')
            period: Time period
            
        Returns:
            Dict containing sentiment metrics
        """
        # Get sentiment analyzer component
        analyzer = self.components.get('sentiment_analyzer')
        
        if not analyzer:
            print("No sentiment analyzer component registered for metrics collection")
            return {}
            
        # Get true labels and predicted scores
        true_labels = []
        predicted_scores = []
        
        if hasattr(analyzer, 'get_labeled_sentiment_data'):
            labeled_data = analyzer.get_labeled_sentiment_data(source, period)
            true_labels = [item.get('true_label', 0) for item in labeled_data]
            predicted_scores = [item.get('predicted_score', 0) for item in labeled_data]
        
        # Record in performance metrics
        metrics = self.performance_metrics.update_sentiment_metrics(
            source=source,
            true_labels=true_labels,
            predicted_scores=predicted_scores,
            threshold=0.0,
            period=period
        )
        
        return metrics
    
    def collect_signal_metrics(self, symbol: str, strategy_name: str = "default") -> Dict:
        """
        Collect signal generation metrics
        
        Args:
            symbol: Trading symbol
            strategy_name: Name of the signal strategy
            
        Returns:
            Dict containing signal metrics
        """
        # Get signal generator component
        signal_generator = self.components.get('signal_generator')
        
        if not signal_generator:
            print("No signal generator component registered for metrics collection")
            return {}
            
        # Get signals and actual prices
        signals = []
        actual_prices = []
        time_periods = [24, 48, 72]  # hours
        
        if hasattr(signal_generator, 'get_signal_performance_data'):
            signal_data = signal_generator.get_signal_performance_data(symbol)
            signals = signal_data.get('signals', [])
            actual_prices = signal_data.get('actual_prices', [])
        
        # Record in performance metrics
        metrics = self.performance_metrics.update_signal_metrics(
            symbol=symbol,
            signals=signals,
            actual_prices=actual_prices,
            time_periods=time_periods,
            strategy_name=strategy_name
        )
        
        return metrics
    
    def _collect_metrics_task(self) -> None:
        """Background task to periodically collect metrics"""
        while self.collecting:
            try:
                # Collect system metrics for all registered components
                for component_name in self.components:
                    self.collect_system_metrics(component_name)
                
                # Collect trading metrics for configured symbols
                # This would normally come from a configuration
                symbols = ["BTC", "ETH", "SOL"]
                for symbol in symbols:
                    self.collect_trading_metrics(symbol, period="1d")
                
                # Collect sentiment metrics
                sources = ["youtube", "twitter"]
                for source in sources:
                    self.collect_sentiment_metrics(source)
                
                # Collect signal metrics
                for symbol in symbols:
                    self.collect_signal_metrics(symbol)
                
                # Generate performance report periodically
                current_hour = datetime.now().hour
                if current_hour == 0:  # Midnight
                    self.performance_metrics.generate_performance_report()
                
            except Exception as e:
                print(f"Error collecting metrics: {str(e)}")
            
            # Sleep until next collection interval
            time.sleep(self.collection_interval)
    
    def start_collection(self, interval: int = 60) -> None:
        """
        Start collecting metrics in the background
        
        Args:
            interval: Collection interval in seconds
        """
        if self.collecting:
            print("Metrics collection already running")
            return
            
        self.collection_interval = interval
        self.collecting = True
        
        # Start collection in a background thread
        self.collection_thread = threading.Thread(target=self._collect_metrics_task)
        self.collection_thread.daemon = True
        self.collection_thread.start()
        
        print(f"Started metrics collection with {interval} second interval")
    
    def stop_collection(self) -> None:
        """Stop collecting metrics"""
        if not self.collecting:
            print("Metrics collection not running")
            return
            
        self.collecting = False
        
        # Wait for collection thread to terminate
        if self.collection_thread and self.collection_thread.is_alive():
            self.collection_thread.join(timeout=5.0)
            
        print("Stopped metrics collection")
    
    def generate_report(self, path: Optional[str] = None) -> None:
        """
        Generate a comprehensive metrics report
        
        Args:
            path: Optional path to save the report
        """
        if not path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(self.metrics_dir, f"performance_report_{timestamp}.html")
            
        self.performance_metrics.generate_performance_report(path)
        
        print(f"Generated performance report: {path}")
    
    def plot_metrics(self, metric_type: str, **kwargs) -> None:
        """
        Plot specific metrics
        
        Args:
            metric_type: Type of metrics to plot ('trading', 'sentiment', 'signal', 'system')
            **kwargs: Additional parameters for the specific plot function
        """
        if metric_type == 'trading':
            self.performance_metrics.plot_trading_performance(**kwargs)
        elif metric_type == 'sentiment':
            self.performance_metrics.plot_sentiment_accuracy(**kwargs)
        elif metric_type == 'signal':
            self.performance_metrics.plot_signal_accuracy(**kwargs)
        elif metric_type == 'system':
            self.performance_metrics.plot_system_performance(**kwargs)
        else:
            print(f"Unknown metric type: {metric_type}")