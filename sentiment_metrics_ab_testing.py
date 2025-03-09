import os
import json
from typing import Dict, List, Callable, Optional
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from ab_testing_framework import ABTestingFramework
from enhanced_performance_metrics import EnhancedPerformanceMetrics

class SentimentAnalysisABTesting:
    """
    A class for conducting A/B testing on sentiment analysis techniques.
    
    This class provides methods to register, evaluate, and compare different 
    sentiment analysis approaches using various performance metrics.
    """

    def __init__(self, data_dir: str = "sentiment_ab_testing_data"):
        """
        Initialize the A/B testing framework for sentiment analysis.
        
        Args:
            data_dir (str, optional): Directory to store A/B testing data. 
                                      Defaults to "sentiment_ab_testing_data".
        """
        self.ab_framework = ABTestingFramework(data_dir=data_dir)
        self.registered_analyzers = {}

    def register_analyzer(self, analyzer_id: str, analyzer_func: Callable) -> None:
        """
        Register a sentiment analysis function for A/B testing.
        
        Args:
            analyzer_id (str): Unique identifier for the analyzer.
            analyzer_func (Callable): Function to analyze sentiment.
        """
        if not isinstance(analyzer_id, str):
            raise ValueError("Analyzer ID must be a string")
        
        if not callable(analyzer_func):
            raise ValueError("Analyzer must be a callable function")
        
        self.registered_analyzers[analyzer_id] = analyzer_func
        print(f"Registered sentiment analyzer: {analyzer_id}")

    def create_sentiment_test(self, 
                               test_description: str, 
                               variants: List[Dict], 
                               test_duration_days: int = 7) -> str:
        """
        Create a new sentiment analysis A/B test.
        
        Args:
            test_description (str): Description of the test.
            variants (List[Dict]): List of analyzer variants to test.
            test_duration_days (int, optional): Duration of the test. Defaults to 7.
        
        Returns:
            str: Unique test ID
        """
        # Validate variants
        for variant in variants:
            if 'analyzer_id' not in variant:
                raise ValueError(f"Each variant must specify an analyzer_id: {variant}")
            
            if variant['analyzer_id'] not in self.registered_analyzers:
                raise ValueError(f"Analyzer {variant['analyzer_id']} not registered")

        # Create test with start and end dates
        start_date = datetime.now().isoformat()
        end_date = (datetime.now() + timedelta(days=test_duration_days)).isoformat()
        
        test_id = self.ab_framework.create_test(
            test_id=f"sentiment_test_{datetime.now().strftime('%Y_%b')}",
            description=test_description,
            variants=variants,
            start_date=start_date,
            end_date=end_date
        )
        
        return test_id

    def evaluate_on_labeled_data(self, 
                                  test_id: str, 
                                  data_source: str, 
                                  labeled_data: List[Dict]) -> Dict:
        """
        Evaluate sentiment analyzers on labeled data.
        
        Args:
            test_id (str): Unique identifier for the test.
            data_source (str): Source of the labeled data.
            labeled_data (List[Dict]): List of labeled sentiment data.
        
        Returns:
            Dict: Performance metrics for each analyzer variant
        """
        # Validate test
        test = self.ab_framework.get_test(test_id)
        if not test:
            raise ValueError(f"Test with ID {test_id} not found")

        # Aggregate results per variant
        result_metrics = {}
        for variant in test['variants']:
            analyzer_id = variant['analyzer_id']
            analyzer_func = self.registered_analyzers.get(analyzer_id)
            
            if not analyzer_func:
                print(f"Warning: Analyzer {analyzer_id} not found, skipping")
                continue

            # Collect performance metrics
            true_labels = []
            predicted_scores = []

            for item in labeled_data:
                text = item.get("text", "")
                true_label = item.get("true_label", 0)

                # Apply analyzer with variant-specific parameters
                analyzer_params = {k: v for k, v in variant.items() if k not in ["id", "analyzer_id"]}
                analysis_result = analyzer_func(text, **analyzer_params)
                
                predicted_score = analysis_result.get("combined_score", 0)
                
                true_labels.append(true_label)
                predicted_scores.append(predicted_score)

            # Calculate performance metrics
            perf_metrics = EnhancedPerformanceMetrics()
            metrics = perf_metrics.update_sentiment_metrics(
                true_labels=true_labels, 
                predicted_scores=predicted_scores
            )

            # Store results
            result_metrics[analyzer_id] = {
                "accuracy": metrics.get("accuracy", 0),
                "correlation": metrics.get("correlation", 0),
                "mae": metrics.get("mae", 0)
            }

        # Record results in A/B testing framework
        self.ab_framework.record_result(
            test_id=test_id,
            variant_id=variant.get('id'),
            metrics=result_metrics
        )

        # Analyze test results
        analysis = self.ab_framework.analyze_test(test_id)

        return {
            "results": result_metrics,
            "analysis": analysis
        }

    def run_real_time_sentiment_test(self, 
                                     test_id: str, 
                                     content_provider, 
                                     days_per_variant: int = 7):
        """
        Run a real-time sentiment analysis test across different variants.
        
        Args:
            test_id (str): Unique test identifier.
            content_provider: Object providing streaming content and sentiment analysis.
            days_per_variant (int, optional): Duration for each variant. Defaults to 7.
        """
        import time

        # Validate test and variants
        test = self.ab_framework.get_test(test_id)
        if not test:
            raise ValueError(f"Test {test_id} is not active")

        # Process each variant
        for variant in test['variants']:
            analyzer_id = variant.get('analyzer_id')
            analyzer_func = self.registered_analyzers.get(analyzer_id)
            
            if not analyzer_func:
                print(f"Warning: Analyzer {analyzer_id} not found, skipping")
                continue

            # Variant-specific parameters
            analyzer_params = {k: v for k, v in variant.items() if k not in ["id", "analyzer_id"]}
            
            # Set analyzer for content provider
            content_provider.set_analyzer(
                lambda text: analyzer_func(text, **analyzer_params)
            )

            # Run test for specified duration
            start_time = datetime.now()
            end_time = start_time + timedelta(days=days_per_variant)
            
            while datetime.now() < end_time:
                # Check if test is still active
                if not self.ab_framework.is_test_active(test_id):
                    print(f"Test {test_id} is no longer active, stopping real-time test")
                    break

                # Collect sentiment results periodically
                if hasattr(content_provider, 'get_sentiment_results'):
                    results = content_provider.get_sentiment_results()
                    
                    for data_source, source_results in results.items():
                        true_labels = source_results.get("true_labels", [])
                        predicted_scores = source_results.get("predicted_scores", [])
                        
                        # Record results
                        self.ab_framework.record_result(
                            test_id=test_id,
                            variant_id=variant.get('id'),
                            metrics={
                                "data_source": data_source,
                                "true_labels": true_labels,
                                "predicted_scores": predicted_scores
                            }
                        )

                # Wait before next iteration
                time.sleep(3600)  # Check every hour

    def get_best_sentiment_analyzer(self, test_id: str, primary_metric: str = "accuracy") -> Dict:
        """
        Determine the best sentiment analyzer based on test results.
        
        Args:
            test_id (str): Unique test identifier.
            primary_metric (str, optional): Metric to use for determining the best analyzer. 
                                            Defaults to "accuracy".
        
        Returns:
            Dict: Details of the best performing analyzer
        """
        # Analyze test results
        analysis = self.ab_framework.analyze_test(test_id)
        
        # Find winning variant
        winner = analysis["winner"].get(primary_metric, {})
        winning_variant = winner.get("variant", {})
        
        if not winning_variant:
            return {
                "success": False,
                "message": "No clear winner found in the test"
            }

        return {
            "success": True,
            "analyzer_id": winning_variant.get("analyzer_id"),
            "configuration": {k: v for k, v in winning_variant.items() if k not in ["id", "analyzer_id"]},
            "performance": {
                metric: analysis["winner"].get(metric, {}).get("value", 0)
                for metric in analysis["winner"]
            }
        }

    def implement_winning_analyzer(self, test_id: str, primary_metric: str = "accuracy") -> bool:
        """
        Implement the winning sentiment analyzer in production.
        
        Args:
            test_id (str): Unique test identifier.
            primary_metric (str, optional): Metric used to determine the best analyzer. 
                                            Defaults to "accuracy".
        
        Returns:
            bool: Whether implementation was successful
        """
        best_analyzer = self.get_best_sentiment_analyzer(test_id, primary_metric)
        
        if not best_analyzer["success"]:
            print(f"Cannot implement winning analyzer: {best_analyzer['message']}")
            return False

        try:
            # Prepare production configuration
            production_config = {
                "analyzer_id": best_analyzer["analyzer_id"],
                "implemented_at": datetime.now().isoformat(),
                "primary_metric": primary_metric,
                "configuration": best_analyzer["configuration"]
            }

            # Ensure production config directory exists
            os.makedirs("production_config", exist_ok=True)

            # Save configuration
            with open("production_config/sentiment_analyzer_config.json", "w") as f:
                json.dump(production_config, f, indent=2)

            print(f"Successfully implemented winning analyzer {best_analyzer['analyzer_id']} from test {test_id}")
            
            # End the test
            self.ab_framework.end_test(test_id)
            
            return True

        except Exception as e:
            print(f"Error implementing winning analyzer: {str(e)}")
            return False