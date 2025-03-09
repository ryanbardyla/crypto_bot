# enhanced_performance_metrics.py

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.metrics import confusion_matrix, classification_report
from typing import Dict, List, Tuple, Optional, Union

class EnhancedPerformanceMetrics:
    """
    Comprehensive performance metrics tracking system for crypto trading bot components
    """
    def __init__(self, data_dir: str = "performance_data"):
        """
        Initialize the performance metrics system
        
        Args:
            data_dir: Directory to store performance data
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize metrics storage
        self.metrics = {
            "trading": {},
            "sentiment": {},
            "signals": {},
            "system": {}
        }
        
        # Load existing metrics if available
        self._load_metrics()
        
    def _load_metrics(self) -> None:
        """Load metrics from saved files if they exist"""
        for metric_type in self.metrics.keys():
            filename = os.path.join(self.data_dir, f"{metric_type}_metrics.json")
            if os.path.exists(filename):
                try:
                    with open(filename, 'r') as f:
                        self.metrics[metric_type] = json.load(f)
                    print(f"Loaded {metric_type} metrics from {filename}")
                except Exception as e:
                    print(f"Error loading metrics from {filename}: {str(e)}")
    
    def _save_metrics(self, metric_type: str) -> None:
        """Save metrics to file"""
        filename = os.path.join(self.data_dir, f"{metric_type}_metrics.json")
        try:
            with open(filename, 'w') as f:
                json.dump(self.metrics[metric_type], f, indent=2, default=str)
            print(f"Saved {metric_type} metrics to {filename}")
        except Exception as e:
            print(f"Error saving metrics to {filename}: {str(e)}")
    
    def update_trading_metrics(self, 
                              symbol: str,
                              period: str,
                              trades: List[Dict],
                              initial_capital: float,
                              final_capital: float,
                              strategy_name: str = "default") -> Dict:
        """
        Update trading performance metrics
        
        Args:
            symbol: Trading symbol (e.g., 'BTC')
            period: Time period (e.g., '1d', '7d', '30d')
            trades: List of trade dictionaries
            initial_capital: Starting capital
            final_capital: Ending capital
            strategy_name: Name of the trading strategy
            
        Returns:
            Dict of calculated metrics
        """
        # Calculate core metrics
        total_return = final_capital - initial_capital
        return_pct = (final_capital / initial_capital - 1) * 100
        
        # Trading metrics
        winning_trades = [t for t in trades if t.get("profit_loss", 0) > 0]
        losing_trades = [t for t in trades if t.get("profit_loss", 0) <= 0]
        
        win_count = len(winning_trades)
        lose_count = len(losing_trades)
        total_trades = win_count + lose_count
        
        # Avoid division by zero
        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
        
        # Calculate profit factor (sum of profits / sum of losses)
        total_profits = sum(t.get("profit_loss", 0) for t in winning_trades)
        total_losses = abs(sum(t.get("profit_loss", 0) for t in losing_trades))
        profit_factor = total_profits / total_losses if total_losses > 0 else float('inf')
        
        # Calculate average winning and losing trade
        avg_win = total_profits / win_count if win_count > 0 else 0
        avg_loss = total_losses / lose_count if lose_count > 0 else 0
        
        # Calculate expectancy (average amount you can expect to win/lose per trade)
        expectancy = (win_rate/100 * avg_win) - ((100-win_rate)/100 * avg_loss) if total_trades > 0 else 0
        
        # Calculate drawdown
        max_drawdown = 0
        peak_capital = initial_capital
        for trade in trades:
            capital_after = trade.get("balance_after", peak_capital)
            if capital_after > peak_capital:
                peak_capital = capital_after
            else:
                drawdown = (peak_capital - capital_after) / peak_capital * 100
                max_drawdown = max(max_drawdown, drawdown)
        
        # Calculate Sharpe Ratio (if we have daily returns)
        sharpe_ratio = None
        if period == '30d' and len(trades) >= 30:
            # Create a dictionary of daily returns
            daily_returns = {}
            for trade in trades:
                if 'timestamp' in trade and 'profit_loss' in trade:
                    date_str = trade['timestamp'].split(' ')[0] if isinstance(trade['timestamp'], str) else trade['timestamp'].strftime('%Y-%m-%d')
                    daily_returns[date_str] = daily_returns.get(date_str, 0) + trade['profit_loss'] / initial_capital
            
            # Convert to numpy array for calculation
            returns_array = np.array(list(daily_returns.values()))
            if len(returns_array) > 0:
                mean_return = np.mean(returns_array)
                std_return = np.std(returns_array, ddof=1)
                if std_return > 0:
                    sharpe_ratio = (mean_return / std_return) * np.sqrt(252)  # Annualized Sharpe
        
        # Store the metrics
        timestamp = datetime.now().isoformat()
        
        metrics = {
            "timestamp": timestamp,
            "symbol": symbol,
            "period": period,
            "strategy": strategy_name,
            "initial_capital": initial_capital,
            "final_capital": final_capital,
            "total_return": total_return,
            "return_pct": return_pct,
            "total_trades": total_trades,
            "winning_trades": win_count,
            "losing_trades": lose_count,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "expectancy": expectancy,
            "max_drawdown": max_drawdown,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "sharpe_ratio": sharpe_ratio
        }
        
        # Update metrics storage
        key = f"{symbol}_{period}_{strategy_name}"
        self.metrics["trading"][key] = metrics
        
        # Save to file
        self._save_metrics("trading")
        
        return metrics
    
    def update_sentiment_metrics(self,
                                source: str,
                                true_labels: List[int],
                                predicted_scores: List[float],
                                threshold: float = 0.0,
                                period: str = "7d") -> Dict:
        """
        Update sentiment analysis performance metrics
        
        Args:
            source: Source of sentiment data (e.g., 'youtube', 'twitter')
            true_labels: List of true sentiment labels (1 for positive, -1 for negative, 0 for neutral)
            predicted_scores: List of predicted sentiment scores (continuous values)
            threshold: Threshold to convert scores to labels
            period: Time period covered
            
        Returns:
            Dict of calculated metrics
        """
        if len(true_labels) != len(predicted_scores):
            raise ValueError("true_labels and predicted_scores must have the same length")
        
        if len(true_labels) == 0:
            return {}
        
        # Convert continuous scores to labels
        predicted_labels = []
        for score in predicted_scores:
            if score > threshold:
                predicted_labels.append(1)  # Positive
            elif score < -threshold:
                predicted_labels.append(-1)  # Negative
            else:
                predicted_labels.append(0)  # Neutral
        
        # Calculate confusion matrix
        # First convert to 0, 1, 2 for confusion matrix calculation
        true_labels_cm = [label + 1 for label in true_labels]  # Convert -1,0,1 to 0,1,2
        predicted_labels_cm = [label + 1 for label in predicted_labels]
        
        # Compute confusion matrix
        cm = confusion_matrix(true_labels_cm, predicted_labels_cm, labels=[0, 1, 2])
        
        # Extract TP, FP, TN, FN for each class
        metrics_by_class = {}
        classes = ['negative', 'neutral', 'positive']
        
        for i, class_name in enumerate(classes):
            # For the current class, every other class is considered negative
            tp = cm[i, i]
            fp = sum(cm[:, i]) - tp
            fn = sum(cm[i, :]) - tp
            tn = np.sum(cm) - tp - fp - fn
            
            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics_by_class[class_name] = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            }
        
        # Calculate overall metrics
        correct_predictions = sum(t == p for t, p in zip(true_labels, predicted_labels))
        accuracy = correct_predictions / len(true_labels)
        
        # Calculate correlation between predicted and true values
        correlation = np.corrcoef(true_labels, predicted_scores)[0, 1]
        
        # Calculate mean absolute error
        mae = np.mean(np.abs(np.array(predicted_scores) - np.array(true_labels)))
        
        # Store the metrics
        timestamp = datetime.now().isoformat()
        
        metrics = {
            "timestamp": timestamp,
            "source": source,
            "period": period,
            "sample_size": len(true_labels),
            "accuracy": accuracy,
            "correlation": correlation,
            "mae": mae,
            "class_metrics": metrics_by_class,
            "confusion_matrix": cm.tolist()
        }
        
        # Update metrics storage
        key = f"{source}_{period}"
        self.metrics["sentiment"][key] = metrics
        
        # Save to file
        self._save_metrics("sentiment")
        
        return metrics
    
    def update_signal_metrics(self,
                             symbol: str,
                             signals: List[Dict],
                             actual_prices: List[float],
                             time_periods: List[int],
                             strategy_name: str = "default") -> Dict:
        """
        Update signal generation performance metrics
        
        Args:
            symbol: Trading symbol (e.g., 'BTC')
            signals: List of signal dictionaries with 'timestamp', 'signal', 'price' fields
            actual_prices: List of actual prices that followed each signal
            time_periods: List of time periods (in hours) to evaluate signals
            strategy_name: Name of the signal strategy
            
        Returns:
            Dict of calculated metrics
        """
        if len(signals) != len(actual_prices):
            raise ValueError("signals and actual_prices must have the same length")
        
        if len(signals) == 0:
            return {}
        
        # Calculate signal accuracy metrics
        metrics_by_period = {}
        
        for period in time_periods:
            correct_signals = 0
            incorrect_signals = 0
            neutral_signals = 0
            
            for signal, future_price in zip(signals, actual_prices):
                signal_type = signal.get('signal', 0)  # 1 for buy, -1 for sell, 0 for neutral
                signal_price = signal.get('price', 0)
                
                if signal_type == 0:
                    neutral_signals += 1
                    continue
                
                price_change = (future_price - signal_price) / signal_price
                
                # Buy signal is correct if price went up
                if signal_type == 1 and price_change > 0:
                    correct_signals += 1
                # Sell signal is correct if price went down
                elif signal_type == -1 and price_change < 0:
                    correct_signals += 1
                else:
                    incorrect_signals += 1
            
            total_directional_signals = correct_signals + incorrect_signals
            accuracy = correct_signals / total_directional_signals if total_directional_signals > 0 else 0
            
            metrics_by_period[str(period)] = {
                "correct_signals": correct_signals,
                "incorrect_signals": incorrect_signals,
                "neutral_signals": neutral_signals,
                "accuracy": accuracy
            }
        
        # Calculate average signal strength vs. actual price movement correlation
        signal_strengths = [signal.get('signal_strength', 0) * (1 if signal.get('signal', 0) > 0 else -1 if signal.get('signal', 0) < 0 else 0) for signal in signals]
        price_movements = [(actual - signal.get('price', 0)) / signal.get('price', 0) for actual, signal in zip(actual_prices, signals)]
        
        correlation = np.corrcoef(signal_strengths, price_movements)[0, 1] if len(signal_strengths) > 1 else 0
        
        # Calculate profit potential if traded on signals
        profit_potential = 0
        for signal, future_price in zip(signals, actual_prices):
            signal_type = signal.get('signal', 0)
            signal_price = signal.get('price', 0)
            
            if signal_type != 0 and signal_price > 0:
                price_change_pct = (future_price - signal_price) / signal_price
                # Buy signal: profit if price went up
                if signal_type == 1:
                    profit_potential += price_change_pct
                # Sell signal: profit if price went down
                elif signal_type == -1:
                    profit_potential -= price_change_pct
        
        # Store the metrics
        timestamp = datetime.now().isoformat()
        
        metrics = {
            "timestamp": timestamp,
            "symbol": symbol,
            "strategy": strategy_name,
            "total_signals": len(signals),
            "buy_signals": sum(1 for s in signals if s.get('signal', 0) == 1),
            "sell_signals": sum(1 for s in signals if s.get('signal', 0) == -1),
            "neutral_signals": sum(1 for s in signals if s.get('signal', 0) == 0),
            "correlation": correlation,
            "profit_potential": profit_potential,
            "by_period": metrics_by_period
        }
        
        # Update metrics storage
        key = f"{symbol}_{strategy_name}"
        self.metrics["signals"][key] = metrics
        
        # Save to file
        self._save_metrics("signals")
        
        return metrics
    
    def update_system_metrics(self,
                             cpu_usage: float,
                             memory_usage: float,
                             api_latency: float,
                             execution_time: float,
                             component: str) -> Dict:
        """
        Update system performance metrics
        
        Args:
            cpu_usage: CPU usage percentage (0-100)
            memory_usage: Memory usage in MB
            api_latency: API latency in milliseconds
            execution_time: Time to execute component in seconds
            component: Name of the component
            
        Returns:
            Dict of updated metrics
        """
        timestamp = datetime.now().isoformat()
        
        metrics = {
            "timestamp": timestamp,
            "component": component,
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage, 
            "api_latency": api_latency,
            "execution_time": execution_time
        }
        
        # Update metrics storage
        if component not in self.metrics["system"]:
            self.metrics["system"][component] = []
            
        self.metrics["system"][component].append(metrics)
        
        # Keep only the last 1000 system metrics
        if len(self.metrics["system"][component]) > 1000:
            self.metrics["system"][component] = self.metrics["system"][component][-1000:]
        
        # Save to file
        self._save_metrics("system")
        
        return metrics
    
    def plot_trading_performance(self, symbol: str, period: str, strategy_name: str = "default", save_path: Optional[str] = None) -> None:
        """
        Generate and display trading performance plots
        
        Args:
            symbol: Trading symbol to plot
            period: Time period to plot
            strategy_name: Strategy name to plot
            save_path: Path to save the plot
        """
        key = f"{symbol}_{period}_{strategy_name}"
        if key not in self.metrics["trading"]:
            print(f"No trading metrics found for {key}")
            return
        
        metrics = self.metrics["trading"][key]
        
        # Create figure with several subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Trading Performance: {symbol} ({period}) - {strategy_name}", fontsize=16)
        
        # Plot returns
        axs[0, 0].bar(['Total Return'], [metrics['return_pct']], color='green' if metrics['return_pct'] > 0 else 'red')
        axs[0, 0].set_ylabel('Return %')
        axs[0, 0].set_title('Total Return')
        axs[0, 0].axhline(y=0, color='k', linestyle='-', alpha=0.2)
        
        # Plot win rate and max drawdown
        bars = axs[0, 1].bar(['Win Rate', 'Max Drawdown'], [metrics['win_rate'], metrics['max_drawdown']], 
                         color=['blue', 'red'])
        axs[0, 1].set_ylabel('Percentage')
        axs[0, 1].set_title('Win Rate & Max Drawdown')
        # Add data labels
        for bar in bars:
            height = bar.get_height()
            axs[0, 1].text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.2f}%', ha='center', va='bottom')
        
        # Plot trade distribution
        axs[1, 0].pie([metrics['winning_trades'], metrics['losing_trades']], 
                  labels=['Winning', 'Losing'], autopct='%1.1f%%',
                  colors=['green', 'red'])
        axs[1, 0].set_title('Trade Distribution')
        
        # Plot key metrics
        metrics_to_show = ['profit_factor', 'expectancy', 'sharpe_ratio']
        values = [metrics.get(m, 0) for m in metrics_to_show]
        bars = axs[1, 1].bar(metrics_to_show, values)
        axs[1, 1].set_title('Key Performance Metrics')
        # Add data labels
        for bar in bars:
            height = bar.get_height()
            axs[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}', ha='center', va='bottom')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        if save_path:
            plt.savefig(save_path)
            print(f"Saved trading performance plot to {save_path}")
        else:
            plt.show()
    
    def plot_sentiment_accuracy(self, source: str, period: str, save_path: Optional[str] = None) -> None:
        """
        Generate and display sentiment analysis accuracy plots
        
        Args:
            source: Source of sentiment data
            period: Time period to plot
            save_path: Path to save the plot
        """
        key = f"{source}_{period}"
        if key not in self.metrics["sentiment"]:
            print(f"No sentiment metrics found for {key}")
            return
        
        metrics = self.metrics["sentiment"][key]
        
        # Create figure with several subplots
        fig, axs = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f"Sentiment Analysis Performance: {source} ({period})", fontsize=16)
        
        # Plot accuracy
        axs[0].bar(['Accuracy'], [metrics['accuracy'] * 100], color='blue')
        axs[0].set_ylabel('Percentage')
        axs[0].set_title('Overall Accuracy')
        axs[0].set_ylim([0, 100])
        
        # Plot class metrics
        class_metrics = metrics['class_metrics']
        classes = list(class_metrics.keys())
        precision = [class_metrics[c]['precision'] * 100 for c in classes]
        recall = [class_metrics[c]['recall'] * 100 for c in classes]
        f1 = [class_metrics[c]['f1_score'] * 100 for c in classes]
        
        x = np.arange(len(classes))
        width = 0.25
        
        axs[1].bar(x - width, precision, width, label='Precision')
        axs[1].bar(x, recall, width, label='Recall')
        axs[1].bar(x + width, f1, width, label='F1')
        
        axs[1].set_ylabel('Percentage')
        axs[1].set_title('Class Metrics')
        axs[1].set_xticks(x)
        axs[1].set_xticklabels(classes)
        axs[1].legend()
        axs[1].set_ylim([0, 100])
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        if save_path:
            plt.savefig(save_path)
            print(f"Saved sentiment accuracy plot to {save_path}")
        else:
            plt.show()
    
    def plot_signal_accuracy(self, symbol: str, strategy_name: str = "default", save_path: Optional[str] = None) -> None:
        """
        Generate and display signal accuracy plots
        
        Args:
            symbol: Trading symbol
            strategy_name: Strategy name
            save_path: Path to save the plot
        """
        key = f"{symbol}_{strategy_name}"
        if key not in self.metrics["signals"]:
            print(f"No signal metrics found for {key}")
            return
        
        metrics = self.metrics["signals"][key]
        by_period = metrics['by_period']
        
        # Create figure with several subplots
        fig, axs = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f"Signal Generation Performance: {symbol} - {strategy_name}", fontsize=16)
        
        # Plot signal distribution
        signal_counts = [metrics['buy_signals'], metrics['sell_signals'], metrics['neutral_signals']]
        axs[0].pie(signal_counts, labels=['Buy', 'Sell', 'Neutral'], autopct='%1.1f%%',
                 colors=['green', 'red', 'gray'])
        axs[0].set_title('Signal Distribution')
        
        # Plot accuracy by period
        periods = list(by_period.keys())
        accuracy = [by_period[p]['accuracy'] * 100 for p in periods]
        
        axs[1].bar(periods, accuracy, color='blue')
        axs[1].set_xlabel('Time Period')
        axs[1].set_ylabel('Accuracy (%)')
        axs[1].set_title('Signal Accuracy by Time Period')
        axs[1].set_ylim([0, 100])
        
        # Add correlation and profit potential as text
        text = f"Correlation with Price Movement: {metrics['correlation']:.2f}\n"
        text += f"Profit Potential: {metrics['profit_potential']:.2f}%"
        fig.text(0.5, 0.01, text, ha='center', fontsize=12, 
                 bbox=dict(facecolor='white', alpha=0.5))
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        if save_path:
            plt.savefig(save_path)
            print(f"Saved signal accuracy plot to {save_path}")
        else:
            plt.show()
    
    def plot_system_performance(self, component: str, metric: str, window: int = 100, save_path: Optional[str] = None) -> None:
        """
        Plot system performance metrics over time
        
        Args:
            component: Component name to plot
            metric: Metric to plot ('cpu_usage', 'memory_usage', 'api_latency', 'execution_time')
            window: Number of data points to include
            save_path: Path to save the plot
        """
        if component not in self.metrics["system"]:
            print(f"No system metrics found for {component}")
            return
        
        metrics = self.metrics["system"][component]
        
        # Extract timestamps and values
        timestamps = [datetime.fromisoformat(m["timestamp"]) for m in metrics[-window:]]
        values = [m.get(metric, 0) for m in metrics[-window:]]
        
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, values, marker='o', linestyle='-', markersize=3)
        
        plt.title(f"{component} - {metric} over time")
        plt.xlabel('Time')
        
        # Set appropriate y-axis label based on metric
        if metric == 'cpu_usage':
            plt.ylabel('CPU Usage (%)')
        elif metric == 'memory_usage':
            plt.ylabel('Memory Usage (MB)')
        elif metric == 'api_latency':
            plt.ylabel('API Latency (ms)')
        elif metric == 'execution_time':
            plt.ylabel('Execution Time (s)')
        else:
            plt.ylabel(metric)
        
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Saved system performance plot to {save_path}")
        else:
            plt.show()
    
    def generate_performance_report(self, path: str = "performance_report.html") -> None:
        """
        Generate a comprehensive HTML performance report
        
        Args:
            path: Path to save the HTML report
        """
        # We'll create a simple HTML report with all metrics
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Crypto Trading Bot Performance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
                .section {{ margin-bottom: 30px; }}
            </style>
        </head>
        <body>
            <h1>Crypto Trading Bot Performance Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="section">
                <h2>Trading Performance</h2>
        """
        
        # Add trading metrics
        if self.metrics["trading"]:
            html += """
                <table>
                    <tr>
                        <th>Symbol</th>
                        <th>Period</th>
                        <th>Strategy</th>
                        <th>Return %</th>
                        <th>Win Rate</th>
                        <th>Profit Factor</th>
                        <th>Max Drawdown</th>
                        <th>Sharpe Ratio</th>
                    </tr>
            """
            
            for key, metrics in self.metrics["trading"].items():
                html += f"""
                    <tr>
                        <td>{metrics['symbol']}</td>
                        <td>{metrics['period']}</td>
                        <td>{metrics['strategy']}</td>
                        <td class="{'positive' if metrics['return_pct'] > 0 else 'negative'}">{metrics['return_pct']:.2f}%</td>
                        <td>{metrics['win_rate']:.2f}%</td>
                        <td>{metrics['profit_factor']:.2f}</td>
                        <td class="negative">{metrics['max_drawdown']:.2f}%</td>
                        <td>{metrics.get('sharpe_ratio', 'N/A')}</td>
                    </tr>
                """
            
            html += "</table>"
        else:
            html += "<p>No trading metrics available</p>"
        
        # Add sentiment metrics
        html += """
            </div>
            
            <div class="section">
                <h2>Sentiment Analysis Performance</h2>
        """
        
        if self.metrics["sentiment"]:
            html += """
                <table>
                    <tr>
                        <th>Source</th>
                        <th>Period</th>
                        <th>Sample Size</th>
                        <th>Accuracy</th>
                        <th>Correlation</th>
                        <th>MAE</th>
                    </tr>
            """
            
            for key, metrics in self.metrics["sentiment"].items():
                html += f"""
                    <tr>
                        <td>{metrics['source']}</td>
                        <td>{metrics['period']}</td>
                        <td>{metrics['sample_size']}</td>
                        <td>{metrics['accuracy']*100:.2f}%</td>
                        <td>{metrics['correlation']:.2f}</td>
                        <td>{metrics['mae']:.2f}</td>
                    </tr>
                """
            
            html += "</table>"
        else:
            html += "<p>No sentiment metrics available</p>"
        
        # Add signal metrics
        html += """
            </div>
            
            <div class="section">
                <h2>Signal Generation Performance</h2>
        """
        
        if self.metrics["signals"]:
            html += """
                <table>
                    <tr>
                        <th>Symbol</th>
                        <th>Strategy</th>
                        <th>Total Signals</th>
                        <th>Correlation</th>"
                        <th>Profit Potential</th>
                        <th>Buy Signals</th>
                        <th>Sell Signals</th>
                        <th>Neutral Signals</th>
                    </tr>
            """
            
            for key, metrics in self.metrics["signals"].items():
                html += f"""
                    <tr>
                        <td>{metrics['symbol']}</td>
                        <td>{metrics['strategy']}</td>
                        <td>{metrics['total_signals']}</td>
                        <td>{metrics['correlation']:.2f}</td>
                        <td class="{'positive' if metrics['profit_potential'] > 0 else 'negative'}">{metrics['profit_potential']:.2f}%</td>
                        <td>{metrics['buy_signals']}</td>
                        <td>{metrics['sell_signals']}</td>
                        <td>{metrics['neutral_signals']}</td>
                    </tr>
                """
            
            html += "</table>"
            
            # Add signal accuracy by period for the most recent symbol
            if self.metrics["signals"] and 'by_period' in next(iter(self.metrics["signals"].values())):
                latest_key = list(self.metrics["signals"].keys())[-1]
                latest_metrics = self.metrics["signals"][latest_key]
                
                html += f"""
                    <h3>Signal Accuracy by Period for {latest_metrics['symbol']}</h3>
                    <table>
                        <tr>
                            <th>Period</th>
                            <th>Accuracy</th>
                            <th>Correct Signals</th>
                            <th>Incorrect Signals</th>
                        </tr>
                """
                
                for period, period_metrics in latest_metrics['by_period'].items():
                    html += f"""
                        <tr>
                            <td>{period} hours</td>
                            <td>{period_metrics['accuracy']*100:.2f}%</td>
                            <td>{period_metrics['correct_signals']}</td>
                            <td>{period_metrics['incorrect_signals']}</td>
                        </tr>
                    """
                
                html += "</table>"
        else:
            html += "<p>No signal metrics available</p>"
        
        # Add system metrics summary
        html += """
            </div>
            
            <div class="section">
                <h2>System Performance</h2>
        """
        
        if self.metrics["system"]:
            html += """
                <table>
                    <tr>
                        <th>Component</th>
                        <th>Average CPU Usage</th>
                        <th>Average Memory Usage</th>
                        <th>Average API Latency</th>
                        <th>Average Execution Time</th>
                        <th>Data Points</th>
                    </tr>
            """
            
            for component, metrics_list in self.metrics["system"].items():
                if not metrics_list:
                    continue
                    
                # Calculate averages
                avg_cpu = sum(m.get('cpu_usage', 0) for m in metrics_list) / len(metrics_list)
                avg_memory = sum(m.get('memory_usage', 0) for m in metrics_list) / len(metrics_list)
                avg_latency = sum(m.get('api_latency', 0) for m in metrics_list) / len(metrics_list)
                avg_execution = sum(m.get('execution_time', 0) for m in metrics_list) / len(metrics_list)
                
                html += f"""
                    <tr>
                        <td>{component}</td>
                        <td>{avg_cpu:.2f}%</td>
                        <td>{avg_memory:.2f} MB</td>
                        <td>{avg_latency:.2f} ms</td>
                        <td>{avg_execution:.2f} s</td>
                        <td>{len(metrics_list)}</td>
                    </tr>
                """
            
            html += "</table>"
        else:
            html += "<p>No system metrics available</p>"
        
        # Close HTML
        html += """
            </div>
        </body>
        </html>
        """
        
        # Write the report to file
        with open(path, 'w') as f:
            f.write(html)
        
        print(f"Performance report generated and saved to {path}")