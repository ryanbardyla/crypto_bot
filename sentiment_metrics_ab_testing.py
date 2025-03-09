# sentiment_metrics_ab_testing.py

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from collections import defaultdict
from ab_testing_framework import ABTestingFramework

class SentimentAnalysisABTesting:
    """
    A/B testing framework specifically for evaluating sentiment analysis approaches
    """
    def __init__(self, data_dir: str = "sentiment_ab_testing_data"):
        """
        Initialize the sentiment analysis A/B testing framework
        
        Args:
            data_dir: Directory to store test data
        """
        self.ab_framework = ABTestingFramework(data_dir=data_dir)
        
        # Dictionary to store sentiment analyzer implementations
        self.analyzers = {}
        
    def register_analyzer(self, analyzer_id: str, analyzer_func: Callable) -> None:
        """
        Register a sentiment analyzer for testing
        
        Args:
            analyzer_id: Unique identifier for the analyzer
            analyzer_func: Function that implements the sentiment analysis
        """
        self.analyzers[analyzer_id] = analyzer_func
        print(f"Registered sentiment analyzer: {analyzer_id}")
    
    def create_sentiment_test(self, 
                            test_id: str,
                            description: str,
                            analyzer_variants: List[Dict],
                            data_sources: List[str],
                            test_duration_days: int = 7,
                            metrics: List[str] = None) -> Dict:
        """
        Create a new A/B test for sentiment analyzers
        
        Args:
            test_id: Unique identifier for the test
            description: Test description
            analyzer_variants: List of analyzer configurations to test
            data_sources: List of data sources to test with (e.g., ['youtube', 'twitter'])
            test_duration_days: Duration of the test in days
            metrics: List of metrics to evaluate (defaults to standard sentiment metrics)
            
        Returns:
            Dict containing the test configuration
        """
        # Default metrics if none provided
        if metrics is None:
            metrics = [
                "accuracy", 
                "correlation", 
                "mae",
                "precision_positive",
                "recall_positive",
                "f1_positive"
            ]
        
        # Ensure each variant has an analyzer_id that exists
        for variant in analyzer_variants:
            if "analyzer_id" not in variant:
                raise ValueError(f"Each variant must specify an analyzer_id: {variant}")
            
            if variant["analyzer_id"] not in self.analyzers:
                raise ValueError(f"Analyzer {variant['analyzer_id']} not registered")
        
        # Calculate end date
        start_date = datetime.now().isoformat()
        end_date = (datetime.now() + timedelta(days=test_duration_days)).isoformat()
        
        # Create test in the A/B framework
        test = self.ab_framework.create_test(
            test_id=test_id,
            description=description,
            variants=analyzer_variants,
            metrics=metrics,
            start_date=start_date,
            end_date=end_date,
            symbols=data_sources  # Reusing the symbols field for data sources
        )
        
        return test
    
    def evaluate_on_labeled_data(self, 
                                test_id: str,
                                labeled_data: List[Dict],
                                data_source: str) -> Dict:
        """
        Evaluate all variants on a set of labeled data
        
        Args:
            test_id: ID of the test to run
            labeled_data: List of dictionaries containing text and true labels
            data_source: Source of the labeled data (e.g., 'youtube', 'twitter')
            
        Returns:
            Dict containing the evaluation results
        """
        if test_id not in self.ab_framework.tests:
            raise ValueError(f"Test with ID {test_id} not found")
        
        test = self.ab_framework.tests[test_id]
        results = {}
        
        # Check if this data source is included in the test
        if data_source not in test["symbols"]:
            raise ValueError(f"Data source {data_source} not included in test {test_id}")
        
        # Evaluate each variant
        for variant in test["variants"]:
            variant_id = variant["id"]
            analyzer_id = variant["analyzer_id"]
            
            if analyzer_id not in self.analyzers:
                print(f"Warning: Analyzer {analyzer_id} not found, skipping")
                continue
            
            analyzer_func = self.analyzers[analyzer_id]
            
            # Configure the analyzer with variant parameters
            analyzer_params = {k: v for k, v in variant.items() if k not in ["id", "analyzer_id"]}
            
            # Evaluate on labeled data
            print(f"Evaluating variant {variant_id} on {data_source} data")
            
            true_labels = []
            predicted_scores = []
            
            for item in labeled_data:
                text = item.get("text", "")
                true_label = item.get("true_label", 0)
                
                # Skip if text is empty
                if not text:
                    continue
                
                # Apply the analyzer
                analysis_result = analyzer_func(text, **analyzer_params)
                predicted_score = analysis_result.get("combined_score", 0)
                
                true_labels.append(true_label)
                predicted_scores.append(predicted_score)
            
            # Calculate metrics
            # We're reusing the sentiment metrics functionality from EnhancedPerformanceMetrics
            from enhanced_performance_metrics import EnhancedPerformanceMetrics
            perf_metrics = EnhancedPerformanceMetrics()
            
            metrics = perf_metrics.update_sentiment_metrics(
                source=data_source,
                true_labels=true_labels,
                predicted_scores=predicted_scores,
                threshold=variant.get("threshold", 0.0),
                period="test"
            )
            
            # Extract the metrics we want to record
            result_metrics = {}
            for metric in test["metrics"]:
                if metric == "accuracy":
                    result_metrics[metric] = metrics.get("accuracy", 0)
                elif metric == "correlation":
                    result_metrics[metric] = metrics.get("correlation", 0)
                elif metric == "mae":
                    result_metrics[metric] = metrics.get("mae", 0)
                elif metric.startswith("precision_"):
                    class_name = metric.split("_")[1]
                    result_metrics[metric] = metrics.get("class_metrics", {}).get(class_name, {}).get("precision", 0)
                elif metric.startswith("recall_"):
                    class_name = metric.split("_")[1]
                    result_metrics[metric] = metrics.get("class_metrics", {}).get(class_name, {}).get("recall", 0)
                elif metric.startswith("f1_"):
                    class_name = metric.split("_")[1]
                    result_metrics[metric] = metrics.get("class_metrics", {}).get(class_name, {}).get("f1_score", 0)
            
            # Record results in A/B framework
            user_id = f"evaluation_{data_source}_{variant_id}"
            self.ab_framework.record_result(
                test_id=test_id,
                user_id=user_id,
                metrics=result_metrics,
                symbol=data_source
            )
            
            # Store in results dictionary
            results[variant_id] = result_metrics
        
        # Analyze test
        analysis = self.ab_framework.analyze_test(test_id)
        
        return {
            "results": results,
            "analysis": analysis
        }
    
    def run_real_time_sentiment_test(self, 
                                    test_id: str,
                                    content_provider: Any,
                                    manual_rating_callback: Callable = None,
                                    days_per_variant: int = 2) -> None:
        """
        Run a real-time test of sentiment analyzers on new content
        
        Args:
            test_id: ID of the test to run
            content_provider: Object that provides new content for analysis
            manual_rating_callback: Optional function to get manual ratings for content
            days_per_variant: Number of days to test each variant
        """
        if test_id not in self.ab_framework.tests:
            raise ValueError(f"Test with ID {test_id} not found")
        
        test = self.ab_framework.tests[test_id]
        
        # Cycle through variants sequentially
        for variant in test["variants"]:
            variant_id = variant["id"]
            analyzer_id = variant["analyzer_id"]
            
            if analyzer_id not in self.analyzers:
                print(f"Warning: Analyzer {analyzer_id} not found, skipping")
                continue
            
            analyzer_func = self.analyzers[analyzer_id]
            
            # Configure the analyzer with variant parameters
            analyzer_params = {k: v for k, v in variant.items() if k not in ["id", "analyzer_id"]}
            
            # Set this analyzer as the active one in the content provider
            print(f"Setting analyzer {analyzer_id} as active for variant {variant_id}")
            content_provider.set_analyzer(
                lambda text: analyzer_func(text, **analyzer_params)
            )
            
            # Run for specified duration
            print(f"Running real-time test for variant {variant_id} for {days_per_variant} days")
            start_time = datetime.now()
            end_time = start_time + timedelta(days=days_per_variant)
            
            while datetime.now() < end_time:
                # Wait for the content provider to collect data
                time.sleep(3600)  # Check every hour
                
                # Check if test is still active
                if test_id not in self.ab_framework.tests or self.ab_framework.tests[test_id]["status"] != "active":
                    print(f"Test {test_id} is no longer active, stopping real-time test")
                    break
                
                # Collect results if available
                if hasattr(content_provider, 'get_sentiment_results'):
                    results = content_provider.get_sentiment_results()
                    
                    for data_source, source_results in results.items():
                        if data_source not in test["symbols"]:
                            continue
                        
                        # Calculate metrics for this source
                        true_labels = source_results.get("true_labels", [])
                        predicted_scores = source_results.get("predicted_scores", [])
                        
                        if not true_labels or not predicted_scores:
                            continue
                        
                        # Calculate metrics
                        from enhanced_performance_metrics import EnhancedPerformanceMetrics
                        perf_metrics = EnhancedPerformanceMetrics()
                        
                        metrics = perf_metrics.update_sentiment_metrics(
                            source=data_source,
                            true_labels=true_labels,
                            predicted_scores=predicted_scores,
                            threshold=variant.get("threshold", 0.0),
                            period="realtime"
                        )
                        
                        # Extract the metrics we want to record
                        result_metrics = {}
                        for metric in test["metrics"]:
                            if metric == "accuracy":
                                result_metrics[metric] = metrics.get("accuracy", 0)
                            elif metric == "correlation":
                                result_metrics[metric] = metrics.get("correlation", 0)
                            elif metric == "mae":
                                result_metrics[metric] = metrics.get("mae", 0)
                            elif metric.startswith("precision_"):
                                class_name = metric.split("_")[1]
                                result_metrics[metric] = metrics.get("class_metrics", {}).get(class_name, {}).get("precision", 0)
                            elif metric.startswith("recall_"):
                                class_name = metric.split("_")[1]
                                result_metrics[metric] = metrics.get("class_metrics", {}).get(class_name, {}).get("recall", 0)
                            elif metric.startswith("f1_"):
                                class_name = metric.split("_")[1]
                                result_metrics[metric] = metrics.get("class_metrics", {}).get(class_name, {}).get("f1_score", 0)
                        
                        # Record results in A/B framework
                        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                        user_id = f"realtime_{data_source}_{variant_id}_{timestamp}"
                        self.ab_framework.record_result(
                            test_id=test_id,
                            user_id=user_id,
                            metrics=result_metrics,
                            symbol=data_source
                        )
    
    def get_best_sentiment_analyzer(self, test_id: str, primary_metric: str = "accuracy") -> Dict:
        """
        Identify the best performing sentiment analyzer based on a primary metric
        
        Args:
            test_id: ID of the test to analyze
            primary_metric: Primary metric to use for determining the best analyzer
            
        Returns:
            Dict containing the best analyzer configuration
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
            "analyzer_id": winning_variant["analyzer_id"],
            "configuration": {k: v for k, v in winning_variant.items() if k not in ["id", "analyzer_id"]},
            "performance": {
                metric: analysis["winner"].get(metric, {}).get("value", 0)
                for metric in analysis["metrics_analyzed"]
            }
        }
    
    def implement_winning_analyzer(self, test_id: str, primary_metric: str = "accuracy") -> bool:
        """
        Implement the winning sentiment analyzer in the production system
        
        Args:
            test_id: ID of the test to analyze
            primary_metric: Primary metric to use for determining the best analyzer
            
        Returns:
            Boolean indicating success
        """
        # Get the best analyzer
        best_analyzer = self.get_best_sentiment_analyzer(test_id, primary_metric)
        
        if best_analyzer["status"] != "success":
            print(f"Cannot implement winning analyzer: {best_analyzer['message']}")
            return False
            
        # Update the production configuration
        analyzer_id = best_analyzer["analyzer_id"]
        config = best_analyzer["configuration"]
        
        try:
            # This is where you would update your production configuration
            # For example, saving to a configuration file
            production_config = {
                "analyzer_id": analyzer_id,
                "parameters": config,
                "implemented_from_test": test_id,
                "implemented_at": datetime.now().isoformat(),
                "primary_metric": primary_metric,
                "performance": best_analyzer["performance"]
            }
            
            # Save to a file
            os.makedirs("production_config", exist_ok=True)
            with open("production_config/sentiment_analyzer_config.json", "w") as f:
                json.dump(production_config, f, indent=2)
                
            print(f"Successfully implemented winning analyzer {analyzer_id} from test {test_id}")
            
            # End the test
            self.ab_framework.end_test(test_id)
            
            return True
            
        except Exception as e:
            print(f"Error implementing winning analyzer: {str(e)}")
            return False