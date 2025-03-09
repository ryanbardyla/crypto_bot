# trading_strategy_ab_testing.py

import os
import json
import time
import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Callable, Optional, Any
from ab_testing_framework import ABTestingFramework

class TradingStrategyABTesting:
    """
    A/B testing framework specifically for evaluating different trading strategies
    """
    def __init__(self, 
                price_fetcher: Any = None, 
                paper_trader: Any = None,
                backtest_engine: Any = None):
        """
        Initialize the trading strategy A/B testing framework
        
        Args:
            price_fetcher: Instance of a price fetcher class
            paper_trader: Instance of a paper trader class
            backtest_engine: Instance of a backtesting engine
        """
        self.ab_framework = ABTestingFramework(data_dir="strategy_ab_testing_data")
        self.price_fetcher = price_fetcher
        self.paper_trader = paper_trader
        self.backtest_engine = backtest_engine
        
        # Dictionary to store strategy implementations
        self.strategies = {}
        
    def register_strategy(self, strategy_id: str, strategy_func: Callable) -> None:
        """
        Register a trading strategy for testing
        
        Args:
            strategy_id: Unique identifier for the strategy
            strategy_func: Function that implements the strategy
        """
        self.strategies[strategy_id] = strategy_func
        print(f"Registered strategy: {strategy_id}")
    
    def create_strategy_test(self, 
                            test_id: str,
                            description: str,
                            strategy_variants: List[Dict],
                            symbols: List[str],
                            test_duration_days: int = 30,
                            metrics: List[str] = None) -> Dict:
        """
        Create a new A/B test for trading strategies
        
        Args:
            test_id: Unique identifier for the test
            description: Test description
            strategy_variants: List of strategy configurations to test
            symbols: List of symbols to test with
            test_duration_days: Duration of the test in days
            metrics: List of metrics to evaluate (defaults to standard trading metrics)
            
        Returns:
            Dict containing the test configuration
        """
        # Default metrics if none provided
        if metrics is None:
            metrics = [
                "return_pct", 
                "win_rate", 
                "profit_factor", 
                "max_drawdown", 
                "expectancy"
            ]
        
        # Ensure each variant has a strategy_id that exists
        for variant in strategy_variants:
            if "strategy_id" not in variant:
                raise ValueError(f"Each variant must specify a strategy_id: {variant}")
            
            if variant["strategy_id"] not in self.strategies:
                raise ValueError(f"Strategy {variant['strategy_id']} not registered")
        
        # Calculate end date
        start_date = datetime.datetime.now().isoformat()
        end_date = (datetime.datetime.now() + datetime.timedelta(days=test_duration_days)).isoformat()
        
        # Create test in the A/B framework
        test = self.ab_framework.create_test(
            test_id=test_id,
            description=description,
            variants=strategy_variants,
            metrics=metrics,
            start_date=start_date,
            end_date=end_date,
            symbols=symbols
        )
        
        return test
    
    def run_backtest_comparison(self, 
                               test_id: str,
                               historical_days: int = 60,
                               initial_capital: float = 10000) -> Dict:
        """
        Run a backtest comparison of all variants in a test
        
        Args:
            test_id: ID of the test to run
            historical_days: Number of days of historical data to use
            initial_capital: Initial capital for backtesting
            
        Returns:
            Dict containing the backtest results
        """
        if test_id not in self.ab_framework.tests:
            raise ValueError(f"Test with ID {test_id} not found")
            
        if not self.backtest_engine:
            raise ValueError("Backtest engine not provided")
        
        test = self.ab_framework.tests[test_id]
        results = {}
        
        # Run backtest for each variant on each symbol
        for symbol in test["symbols"]:
            results[symbol] = {}
            
            for variant in test["variants"]:
                variant_id = variant["id"]
                strategy_id = variant["strategy_id"]
                
                if strategy_id not in self.strategies:
                    print(f"Warning: Strategy {strategy_id} not found, skipping")
                    continue
                
                strategy_func = self.strategies[strategy_id]
                
                # Configure the strategy with variant parameters
                strategy_params = {k: v for k, v in variant.items() if k not in ["id", "strategy_id"]}
                
                # Run backtest
                print(f"Running backtest for symbol {symbol}, variant {variant_id}")
                backtest_result = self.backtest_engine.backtest(
                    symbol=symbol,
                    strategy_func=lambda df: strategy_func(df, **strategy_params),
                    days=historical_days,
                    initial_capital=initial_capital
                )
                
                if backtest_result:
                    # Record results in A/B framework
                    metrics = {
                        metric: backtest_result.get(metric, 0)
                        for metric in test["metrics"]
                    }
                    
                    user_id = f"backtest_{symbol}_{variant_id}"
                    self.ab_framework.record_result(
                        test_id=test_id,
                        user_id=user_id,
                        metrics=metrics,
                        symbol=symbol
                    )
                    
                    # Store in results dictionary
                    results[symbol][variant_id] = backtest_result
                    
        # Analyze test
        analysis = self.ab_framework.analyze_test(test_id)
        
        return {
            "results": results,
            "analysis": analysis
        }
    
    def run_paper_trading_test(self, 
                              test_id: str, 
                              days_per_variant: int = 3,
                              simultaneous: bool = False,
                              initial_capital: float = 10000) -> None:
        """
        Run a paper trading test cycling through all variants
        
        Args:
            test_id: ID of the test to run
            days_per_variant: Number of days to test each variant
            simultaneous: If True, run all variants simultaneously (requires multiple paper trader instances)
            initial_capital: Initial capital for paper trading
        """
        if test_id not in self.ab_framework.tests:
            raise ValueError(f"Test with ID {test_id} not found")
            
        if not self.paper_trader:
            raise ValueError("Paper trader not provided")
        
        test = self.ab_framework.tests[test_id]
        
        if simultaneous:
            # Create a paper trader instance for each variant
            paper_traders = {}
            for variant in test["variants"]:
                variant_id = variant["id"]
                strategy_id = variant["strategy_id"]
                
                if strategy_id not in self.strategies:
                    print(f"Warning: Strategy {strategy_id} not found, skipping")
                    continue
                
                # Configure paper trader for this variant
                # This would require creating multiple instances of your paper trader
                # Implementation depends on your paper trader class
                print(f"Setting up paper trader for variant {variant_id}")
                
                # Record the start of this variant's test
                print(f"Started paper trading test for variant {variant_id}")
                
            # Let them run for the specified duration
            print(f"Running paper trading test for {days_per_variant} days")
            time.sleep(days_per_variant * 24 * 60 * 60)  # Sleep for days_per_variant days
            
            # Collect and record results
            for variant_id, paper_trader in paper_traders.items():
                # Get performance metrics
                performance = paper_trader.get_performance_metrics()
                
                # Record in A/B testing framework
                for symbol in test["symbols"]:
                    symbol_performance = performance.get(symbol, {})
                    metrics = {
                        metric: symbol_performance.get(metric, 0)
                        for metric in test["metrics"]
                    }
                    
                    user_id = f"paper_{symbol}_{variant_id}"
                    self.ab_framework.record_result(
                        test_id=test_id,
                        user_id=user_id,
                        metrics=metrics,
                        symbol=symbol
                    )
        else:
            # Sequential testing - run one variant at a time
            for variant in test["variants"]:
                variant_id = variant["id"]
                strategy_id = variant["strategy_id"]
                
                if strategy_id not in self.strategies:
                    print(f"Warning: Strategy {strategy_id} not found, skipping")
                    continue
                
                strategy_func = self.strategies[strategy_id]
                
                # Configure the strategy with variant parameters
                strategy_params = {k: v for k, v in variant.items() if k not in ["id", "strategy_id"]}
                
                # Apply the strategy to the paper trader
                print(f"Applying strategy {strategy_id} with variant {variant_id}")
                self.paper_trader.set_strategy(
                    lambda df: strategy_func(df, **strategy_params)
                )
                
                # Reset paper trader to initial state
                self.paper_trader.reset(initial_capital=initial_capital)
                
                # Run for specified duration
                print(f"Running paper trading for variant {variant_id} for {days_per_variant} days")
                start_time = time.time()
                end_time = start_time + (days_per_variant * 24 * 60 * 60)
                
                # This is a placeholder - in a real implementation, you would
                # likely use a different mechanism to let the paper trader run
                # Maybe you'd use a scheduler or just let it run in a separate process
                while time.time() < end_time:
                    # Let the paper trader run its schedule
                    time.sleep(3600)  # Check every hour
                    
                    # You might want to add a way to check if the test should be terminated early
                    if test_id not in self.ab_framework.tests or self.ab_framework.tests[test_id]["status"] != "active":
                        print(f"Test {test_id} is no longer active, stopping paper trading")
                        break
                
                # Get performance metrics
                performance = self.paper_trader.get_performance_metrics()
                
                # Record in A/B testing framework
                for symbol in test["symbols"]:
                    symbol_performance = performance.get(symbol, {})
                    metrics = {
                        metric: symbol_performance.get(metric, 0)
                        for metric in test["metrics"]
                    }
                    
                    user_id = f"paper_{symbol}_{variant_id}"
                    self.ab_framework.record_result(
                        test_id=test_id,
                        user_id=user_id,
                        metrics=metrics,
                        symbol=symbol
                    )
    
    def generate_strategy_comparison_report(self, test_id: str, path: Optional[str] = None) -> str:
        """
        Generate a comprehensive report comparing all strategy variants
        
        Args:
            test_id: ID of the test to analyze
            path: Optional path to save the HTML report
            
        Returns:
            HTML report content
        """
        # Leverage the A/B testing framework's report generator
        html = self.ab_framework.generate_test_report(test_id, path)
        
        # Add additional trading-specific visualizations
        # This could include performance charts, drawdown comparisons, etc.
        # Depends on the specific metrics you want to highlight
        
        return html
    
    def get_best_strategy(self, test_id: str, primary_metric: str = "return_pct") -> Dict:
        """
        Identify the best performing strategy variant based on a primary metric
        
        Args:
            test_id: ID of the test to analyze
            primary_metric: Primary metric to use for determining the best strategy
            
        Returns:
            Dict containing the best strategy configuration
        """
        analysis = self.ab_framework.analyze_test(test_id)
        winner = analysis["winner"].get(primary_metric, {})
        
        if not winner:
            return {
                "status": "no_winner",
                "message": f"No winner found for metric {primary_metric}"
            }
            
        winning_variant_id = winner["variant_id"]
        
        # Find the winning variant configuration
        test = self.ab_framework.tests[test_id]
        winning_variant = None
        for variant in test["variants"]:
            if variant["id"] == winning_variant_id:
                winning_variant = variant
                break
                
        if not winning_variant:
            return {
                "status": "error",
                "message": f"Winning variant {winning_variant_id} not found in test configuration"
            }
                
        return {
            "status": "success",
            "test_id": test_id,
            "primary_metric": primary_metric,
            "variant_id": winning_variant_id,
            "strategy_id": winning_variant["strategy_id"],
            "configuration": {k: v for k, v in winning_variant.items() if k not in ["id", "strategy_id"]},
            "performance": {
                metric: analysis["winner"].get(metric, {}).get("value", 0)
                for metric in analysis["metrics_analyzed"]
            }
        }
    
    def implement_winning_strategy(self, test_id: str, primary_metric: str = "return_pct") -> bool:
        """
        Implement the winning strategy in the production system
        
        Args:
            test_id: ID of the test to analyze
            primary_metric: Primary metric to use for determining the best strategy
            
        Returns:
            Boolean indicating success
        """
        # Get the best strategy
        best_strategy = self.get_best_strategy(test_id, primary_metric)
        
        if best_strategy["status"] != "success":
            print(f"Cannot implement winning strategy: {best_strategy['message']}")
            return False
            
        # Update the production configuration
        strategy_id = best_strategy["strategy_id"]
        config = best_strategy["configuration"]
        
        try:
            # This is where you would update your production configuration
            # For example, saving to a configuration file
            production_config = {
                "strategy_id": strategy_id,
                "parameters": config,
                "implemented_from_test": test_id,
                "implemented_at": datetime.datetime.now().isoformat(),
                "primary_metric": primary_metric,
                "performance": best_strategy["performance"]
            }
            
            # Save to a file
            os.makedirs("production_config", exist_ok=True)
            with open("production_config/strategy_config.json", "w") as f:
                json.dump(production_config, f, indent=2)
                
            print(f"Successfully implemented winning strategy {strategy_id} from test {test_id}")
            
            # End the test
            self.ab_framework.end_test(test_id)
            
            return True
            
        except Exception as e:
            print(f"Error implementing winning strategy: {str(e)}")
            return False