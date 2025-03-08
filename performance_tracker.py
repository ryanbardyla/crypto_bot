# performance_tracker.py
import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("performance_tracker.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("PerformanceTracker")

class PerformanceTracker:
    def __init__(self, config_file="performance_tracker_config.json"):
        self.load_config(config_file)
        self.load_performance_data()
        
    def load_config(self, config_file):
        try:
            with open(config_file, "r") as f:
                self.config = json.load(f)
                
            # Performance settings
            self.performance_log_file = self.config.get("performance_log_file", "performance_history.json")
            self.trade_log_file = self.config.get("trade_log_file", "paper_trading/trade_history.json")
            self.report_interval_days = self.config.get("report_interval_days", 7)
            self.benchmark_symbol = self.config.get("benchmark_symbol", "BTC")
            
            # Metrics to calculate
            self.calculate_sharpe = self.config.get("calculate_sharpe", True)
            self.calculate_sortino = self.config.get("calculate_sortino", True)
            self.calculate_drawdown = self.config.get("calculate_drawdown", True)
            self.calculate_win_rate = self.config.get("calculate_win_rate", True)
            
            # Report output settings
            self.output_dir = self.config.get("output_dir", "performance_reports")
            self.generate_charts = self.config.get("generate_charts", True)
            
            logger.info("Configuration loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            raise
            
    def load_performance_data(self):
        """Load historical performance data"""
        try:
            # Load performance log
            if os.path.exists(self.performance_log_file):
                with open(self.performance_log_file, "r") as f:
                    self.performance_history = json.load(f)
                logger.info(f"Loaded {len(self.performance_history)} performance records")
            else:
                self.performance_history = []
                logger.warning(f"Performance log file {self.performance_log_file} not found")
                
            # Load trade history
            if os.path.exists(self.trade_log_file):
                with open(self.trade_log_file, "r") as f:
                    self.trade_history = json.load(f)
                logger.info(f"Loaded {len(self.trade_history)} trade records")
            else:
                self.trade_history = []
                logger.warning(f"Trade log file {self.trade_log_file} not found")
                
            # Convert to pandas DataFrame for easier analysis
            if self.performance_history:
                self.performance_df = pd.DataFrame(self.performance_history)
                self.performance_df['timestamp'] = pd.to_datetime(self.performance_df['timestamp'])
                self.performance_df = self.performance_df.sort_values('timestamp')
            else:
                self.performance_df = pd.DataFrame()
                
            if self.trade_history:
                self.trades_df = pd.DataFrame(self.trade_history)
                self.trades_df['timestamp'] = pd.to_datetime(self.trades_df['timestamp'])
                self.trades_df = self.trades_df.sort_values('timestamp')
            else:
                self.trades_df = pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error loading performance data: {str(e)}")
            self.performance_history = []
            self.trade_history = []
            self.performance_df = pd.DataFrame()
            self.trades_df = pd.DataFrame()
            
    def load_benchmark_data(self):
        """Load benchmark data (e.g., BTC price) for comparison"""
        try:
            # Load price history from file
            with open("price_history.json", "r") as f:
                price_history = json.load(f)
                
            if self.benchmark_symbol in price_history:
                benchmark_data = price_history[self.benchmark_symbol]
                benchmark_df = pd.DataFrame(benchmark_data)
                benchmark_df['timestamp'] = pd.to_datetime(benchmark_df['timestamp'])
                benchmark_df = benchmark_df.sort_values('timestamp')
                
                # Resample to daily frequency
                benchmark_df['date'] = benchmark_df['timestamp'].dt.date
                benchmark_df['date'] = pd.to_datetime(benchmark_df['date'])
                daily_df = benchmark_df.groupby('date').agg({
                    'price': 'last'
                }).reset_index()
                
                return daily_df
            else:
                logger.warning(f"Benchmark symbol {self.benchmark_symbol} not found in price history")
                return None
        except Exception as e:
            logger.error(f"Error loading benchmark data: {str(e)}")
            return None
            
    def calculate_metrics(self, days=30):
        """Calculate performance metrics"""
        try:
            if self.performance_df.empty:
                logger.warning("No performance data available")
                return None
                
            # Filter for the specified time period
            cutoff_date = datetime.now() - timedelta(days=days)
            period_df = self.performance_df[self.performance_df['timestamp'] >= cutoff_date].copy()
            
            if period_df.empty:
                logger.warning(f"No performance data available for the last {days} days")
                return None
                
            # Basic metrics
            initial_value = period_df['total_value'].iloc[0]
            final_value = period_df['total_value'].iloc[-1]
            return_pct = (final_value / initial_value - 1) * 100
            
            # Calculate daily returns
            period_df['date'] = period_df['timestamp'].dt.date
            period_df['date'] = pd.to_datetime(period_df['date'])
            daily_df = period_df.groupby('date').agg({
                'total_value': 'last'
            }).reset_index()
            
            daily_df['daily_return'] = daily_df['total_value'].pct_change()
            
            # Sharpe Ratio (annualized)
            risk_free_rate = 0.0  # For simplicity, assuming 0% risk-free rate
            if self.calculate_sharpe and len(daily_df) > 1:
                mean_daily_return = daily_df['daily_return'].mean()
                std_daily_return = daily_df['daily_return'].std()
                sharpe_ratio = (mean_daily_return - risk_free_rate) / std_daily_return if std_daily_return > 0 else 0
                # Annualize
                sharpe_ratio = sharpe_ratio * np.sqrt(365)
            else:
                sharpe_ratio = None
                
            # Sortino Ratio (downside risk only, annualized)
            if self.calculate_sortino and len(daily_df) > 1:
                mean_daily_return = daily_df['daily_return'].mean()
                # Calculate downside deviation (standard deviation of negative returns only)
                negative_returns = daily_df[daily_df['daily_return'] < 0]['daily_return']
                downside_deviation = negative_returns.std() if len(negative_returns) > 0 else 0
                sortino_ratio = (mean_daily_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
                # Annualize
                sortino_ratio = sortino_ratio * np.sqrt(365)
            else:
                sortino_ratio = None
                
            # Maximum Drawdown
            if self.calculate_drawdown:
                daily_df['cumulative_return'] = (1 + daily_df['daily_return']).cumprod()
                daily_df['cumulative_max'] = daily_df['cumulative_return'].cummax()
                daily_df['drawdown'] = (daily_df['cumulative_return'] / daily_df['cumulative_max'] - 1) * 100
                max_drawdown = abs(daily_df['drawdown'].min())
                max_drawdown_date = daily_df.loc[daily_df['drawdown'].idxmin(), 'date']
            else:
                max_drawdown = None
                max_drawdown_date = None
                
            # Win Rate
            if self.calculate_win_rate and not self.trades_df.empty:
                # Filter trades for the specified period
                period_trades = self.trades_df[
                    (self.trades_df['timestamp'] >= cutoff_date) & 
                    (self.trades_df['type'] == 'SELL')
                ]
                
                if not period_trades.empty:
                    # Count winning and losing trades
                    winning_trades = period_trades[period_trades['profit_loss'] > 0]
                    win_rate = len(winning_trades) / len(period_trades) * 100
                    
                    # Calculate average win and loss
                    avg_win = winning_trades['profit_loss'].mean() if not winning_trades.empty else 0
                    losing_trades = period_trades[period_trades['profit_loss'] <= 0]
                    avg_loss = losing_trades['profit_loss'].mean() if not losing_trades.empty else 0
                    
                    # Calculate profit factor
                    total_profit = winning_trades['profit_loss'].sum() if not winning_trades.empty else 0
                    total_loss = abs(losing_trades['profit_loss'].sum()) if not losing_trades.empty else 0
                    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
                else:
                    win_rate = None
                    avg_win = None
                    avg_loss = None
                    profit_factor = None
            else:
                win_rate = None
                avg_win = None
                avg_loss = None
                profit_factor = None
                
            # Compare to benchmark
            benchmark_df = self.load_benchmark_data()
            if benchmark_df is not None:
                # Filter benchmark data for the specified period
                benchmark_period = benchmark_df[benchmark_df['date'] >= cutoff_date].copy()
                
                if not benchmark_period.empty:
                    benchmark_initial = benchmark_period['price'].iloc[0]
                    benchmark_final = benchmark_period['price'].iloc[-1]
                    benchmark_return = (benchmark_final / benchmark_initial - 1) * 100
                    
                    # Calculate alpha (excess return over benchmark)
                    alpha = return_pct - benchmark_return
                else:
                    benchmark_return = None
                    alpha = None
            else:
                benchmark_return = None
                alpha = None
                
            return {
                "period_days": days,
                "start_date": period_df['timestamp'].iloc[0],
                "end_date": period_df['timestamp'].iloc[-1],
                "initial_value": initial_value,
                "final_value": final_value,
                "return_pct": return_pct,
                "sharpe_ratio": sharpe_ratio,
                "sortino_ratio": sortino_ratio,
                "max_drawdown": max_drawdown,
                "max_drawdown_date": max_drawdown_date,
                "win_rate": win_rate,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "profit_factor": profit_factor,
                "benchmark_return": benchmark_return,
                "alpha": alpha,
                "daily_returns": daily_df.to_dict(orient="records") if not daily_df.empty else []
            }
                
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return None
            
    def generate_performance_charts(self, metrics, output_file=None):
        """Generate performance charts"""
        if metrics is None or not self.generate_charts:
            return False
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Convert daily returns to DataFrame
            daily_df = pd.DataFrame(metrics["daily_returns"])
            
            if daily_df.empty:
                logger.warning("No daily return data available for charts")
                return False
                
            # Create figure with multiple subplots
            fig, axs = plt.subplots(3, 1, figsize=(12, 18), gridspec_kw={'height_ratios': [3, 1, 1]})
            
            # Plot equity curve
            axs[0].plot(daily_df['date'], daily_df['total_value'], 'b-', linewidth=2)
            axs[0].set_title('Trading Equity Curve')
            axs[0].set_ylabel('Account Value ($)')
            axs[0].grid(True, alpha=0.3)
            
            # Plot drawdown
            if 'drawdown' in daily_df.columns:
                axs[1].fill_between(daily_df['date'], daily_df['drawdown'], 0, color='red', alpha=0.3)
                axs[1].plot(daily_df['date'], daily_df['drawdown'], 'r-', linewidth=1)
                axs[1].set_title('Drawdown (%)')
                axs[1].set_ylabel('Drawdown %')
                axs[1].grid(True, alpha=0.3)
                
                # Set y-limits
                min_drawdown = min(daily_df['drawdown'].min() * 1.1, -1)  # At least -1%
                axs[1].set_ylim([min_drawdown, 1])
            
            # Plot daily returns
            daily_df['positive_returns'] = daily_df['daily_return'].apply(lambda x: max(0, x))
            daily_df['negative_returns'] = daily_df['daily_return'].apply(lambda x: min(0, x))
            
            axs[2].bar(daily_df['date'], daily_df['positive_returns'] * 100, color='green', alpha=0.7)
            axs[2].bar(daily_df['date'], daily_df['negative_returns'] * 100, color='red', alpha=0.7)
            axs[2].set_title('Daily Returns (%)')
            axs[2].set_ylabel('Return %')
            axs[2].grid(True, alpha=0.3)
            
            # Add performance metrics as text
            metrics_text = [
                f"Period: {metrics['start_date'].strftime('%Y-%m-%d')} to {metrics['end_date'].strftime('%Y-%m-%d')}",
                f"Total Return: {metrics['return_pct']:.2f}%",
                f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}" if metrics['sharpe_ratio'] is not None else "",
                f"Sortino Ratio: {metrics['sortino_ratio']:.2f}" if metrics['sortino_ratio'] is not None else "",
                f"Max Drawdown: {metrics['max_drawdown']:.2f}%" if metrics['max_drawdown'] is not None else "",
                f"Win Rate: {metrics['win_rate']:.2f}%" if metrics['win_rate'] is not None else "",
                f"Profit Factor: {metrics['profit_factor']:.2f}" if metrics['profit_factor'] is not None else "",
                f"Benchmark Return: {metrics['benchmark_return']:.2f}%" if metrics['benchmark_return'] is not None else "",
                f"Alpha: {metrics['alpha']:.2f}%" if metrics['alpha'] is not None else ""
            ]
            
            # Remove empty strings
            metrics_text = [text for text in metrics_text if text]
            
            plt.figtext(0.02, 0.01, '\n'.join(metrics_text), fontsize=12, 
                       bbox=dict(facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.15)  # Make room for metrics text
            
            # Save the figure
            if output_file is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = os.path.join(self.output_dir, f"performance_report_{timestamp}.png")
                
            plt.savefig(output_file)
            plt.close()
            
            logger.info(f"Performance chart saved to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating performance charts: {str(e)}")
            return False
            
    def generate_performance_report(self, days=30):
        """Generate a complete performance report"""
        try:
            # Calculate metrics
            metrics = self.calculate_metrics(days=days)
            
            if metrics is None:
                logger.warning("Could not calculate performance metrics")
                return False
                
            # Create output directory
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Generate report timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save metrics to JSON
            metrics_file = os.path.join(self.output_dir, f"performance_metrics_{timestamp}.json")
            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=2, default=str)
                
            # Generate charts
            chart_file = os.path.join(self.output_dir, f"performance_chart_{timestamp}.png")
            self.generate_performance_charts(metrics, output_file=chart_file)
            
            logger.info(f"Performance report generated: {metrics_file}")
            
            # Return the metrics for display
            return metrics
            
        except Exception as e:
            logger.error(f"Error generating performance report: {str(e)}")
            return None
            
    def run_scheduled_reporting(self):
        """Start scheduled performance reporting"""
        import schedule
        import time
        
        logger.info(f"Starting scheduled reporting with {self.report_interval_days} day interval")
        
        # Run once immediately
        self.generate_performance_report()
        
        # Schedule regular reporting
        schedule.every(self.report_interval_days).days.do(
            lambda: self.generate_performance_report()
        )
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(3600)  # Check every hour
        except KeyboardInterrupt:
            logger.info("Scheduled reporting stopped by user")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Trading Performance Tracker")
    parser.add_argument("--days", type=int, default=30, help="Number of days to analyze")
    parser.add_argument("--schedule", action="store_true", help="Start scheduled reporting")
    
    args = parser.parse_args()
    
    tracker = PerformanceTracker()
    
    if args.schedule:
        tracker.run_scheduled_reporting()
    else:
        metrics = tracker.generate_performance_report(days=args.days)
        
        if metrics:
            print("\nPerformance Report:")
            print(f"Period: {metrics['start_date'].strftime('%Y-%m-%d')} to {metrics['end_date'].strftime('%Y-%m-%d')}")
            print(f"Total Return: {metrics['return_pct']:.2f}%")
            
            if metrics['sharpe_ratio'] is not None:
                print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
                
            if metrics['sortino_ratio'] is not None:
                print(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")
                
            if metrics['max_drawdown'] is not None:
                print(f"Maximum Drawdown: {metrics['max_drawdown']:.2f}%")
                
            if metrics['win_rate'] is not None:
                print(f"Win Rate: {metrics['win_rate']:.2f}%")
                print(f"Profit Factor: {metrics['profit_factor']:.2f}")
                
            if metrics['benchmark_return'] is not None:
                print(f"\nBenchmark ({tracker.benchmark_symbol}) Return: {metrics['benchmark_return']:.2f}%")
                print(f"Alpha: {metrics['alpha']:.2f}%")
                
            print(f"\nDetailed report saved to {tracker.output_dir}")