# ab_testing_framework.py

import os
import json
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Callable
from collections import defaultdict

class ABTestingFramework:
    """
    A/B testing framework for crypto trading strategies
    """
    def __init__(self, data_dir: str = "ab_testing_data"):
        """
        Initialize the A/B testing framework
        
        Args:
            data_dir: Directory to store test data
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize test storage
        self.tests = {}
        self.test_results = {}
        
        # Load existing tests if available
        self._load_tests()
    
    def _load_tests(self) -> None:
        """Load tests from saved files if they exist"""
        filename = os.path.join(self.data_dir, "ab_tests.json")
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    data = json.load(f)
                    self.tests = data.get("tests", {})
                    self.test_results = data.get("test_results", {})
                print(f"Loaded A/B tests from {filename}")
            except Exception as e:
                print(f"Error loading A/B tests from {filename}: {str(e)}")
    
    def _save_tests(self) -> None:
        """Save tests to file"""
        filename = os.path.join(self.data_dir, "ab_tests.json")
        try:
            data = {
                "tests": self.tests,
                "test_results": self.test_results
            }
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            print(f"Saved A/B tests to {filename}")
        except Exception as e:
            print(f"Error saving A/B tests to {filename}: {str(e)}")
    
    def create_test(self, 
                   test_id: str, 
                   description: str,
                   variants: List[Dict],
                   metrics: List[str],
                   start_date: Optional[str] = None,
                   end_date: Optional[str] = None,
                   symbols: Optional[List[str]] = None) -> Dict:
        """
        Create a new A/B test
        
        Args:
            test_id: Unique identifier for the test
            description: Description of the test
            variants: List of variant configurations
            metrics: List of metrics to compare (e.g., 'return_pct', 'win_rate')
            start_date: Optional start date for the test (defaults to now)
            end_date: Optional end date for the test
            symbols: List of symbols to include in the test
            
        Returns:
            Dict containing the test configuration
        """
        if test_id in self.tests:
            raise ValueError(f"Test with ID {test_id} already exists")
        
        # Set default start date to now
        if start_date is None:
            start_date = datetime.now().isoformat()
            
        # Assign variant IDs if not provided
        for i, variant in enumerate(variants):
            if 'id' not in variant:
                variant['id'] = f"variant_{i+1}"
        
        # Create test configuration
        test = {
            "test_id": test_id,
            "description": description,
            "variants": variants,
            "metrics": metrics,
            "start_date": start_date,
            "end_date": end_date,
            "status": "active",
            "symbols": symbols or ["BTC", "ETH"],
            "created_at": datetime.now().isoformat()
        }
        
        self.tests[test_id] = test
        self._save_tests()
        
        return test
    
    def end_test(self, test_id: str) -> Dict:
        """
        End an active A/B test
        
        Args:
            test_id: ID of the test to end
            
        Returns:
            Updated test configuration
        """
        if test_id not in self.tests:
            raise ValueError(f"Test with ID {test_id} not found")
        
        test = self.tests[test_id]
        if test["status"] != "active":
            raise ValueError(f"Test {test_id} is not active (current status: {test['status']})")
        
        test["status"] = "completed"
        test["end_date"] = datetime.now().isoformat()
        
        self.tests[test_id] = test
        self._save_tests()
        
        return test
    
    def assign_variant(self, test_id: str, user_id: str = None) -> Dict:
        """
        Assign a variant to a user/session for a specific test
        
        Args:
            test_id: ID of the test
            user_id: Optional user/session ID (if not provided, a random one will be generated)
            
        Returns:
            Dict containing the assigned variant
        """
        if test_id not in self.tests:
            raise ValueError(f"Test with ID {test_id} not found")
        
        test = self.tests[test_id]
        if test["status"] != "active":
            raise ValueError(f"Test {test_id} is not active (current status: {test['status']})")
        
        # Generate a random user ID if none provided
        if user_id is None:
            user_id = f"user_{random.randint(1, 1000000)}"
            
        # Deterministically assign variant based on user ID
        # This ensures the same user gets the same variant consistently
        variants = test["variants"]
        variant_index = hash(user_id + test_id) % len(variants)
        variant = variants[variant_index]
        
        # Record the assignment
        test_results = self.test_results.get(test_id, {"assignments": {}, "results": []})
        test_results["assignments"][user_id] = variant["id"]
        self.test_results[test_id] = test_results
        self._save_tests()
        
        return variant
    
    def record_result(self, 
                     test_id: str, 
                     user_id: str, 
                     metrics: Dict[str, float],
                     symbol: str = None) -> None:
        """
        Record a result for a specific test and user
        
        Args:
            test_id: ID of the test
            user_id: User ID
            metrics: Dictionary of metric values
            symbol: Optional symbol associated with this result
        """
        if test_id not in self.tests:
            raise ValueError(f"Test with ID {test_id} not found")
            
        if test_id not in self.test_results:
            self.test_results[test_id] = {"assignments": {}, "results": []}
            
        # Get variant assignment for this user
        variant_id = self.test_results[test_id]["assignments"].get(user_id)
        if variant_id is None:
            # If user wasn't assigned, assign them now
            variant = self.assign_variant(test_id, user_id)
            variant_id = variant["id"]
        
        # Record the result
        result = {
            "user_id": user_id,
            "variant_id": variant_id,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "symbol": symbol
        }
        
        self.test_results[test_id]["results"].append(result)
        self._save_tests()
    
    def analyze_test(self, test_id: str) -> Dict:
        """
        Analyze the results of an A/B test
        
        Args:
            test_id: ID of the test to analyze
            
        Returns:
            Dict containing the analysis results
        """
        if test_id not in self.tests:
            raise ValueError(f"Test with ID {test_id} not found")
            
        if test_id not in self.test_results:
            raise ValueError(f"No results found for test {test_id}")
            
        test = self.tests[test_id]
        results = self.test_results[test_id]["results"]
        
        if not results:
            return {"status": "no_data", "message": "No results recorded for this test"}
            
        # Group results by variant
        variant_results = defaultdict(list)
        for result in results:
            variant_id = result["variant_id"]
            variant_results[variant_id].append(result)
        
        # Calculate statistics for each metric in each variant
        analysis = {
            "test_id": test_id,
            "description": test["description"],
            "total_results": len(results),
            "metrics_analyzed": test["metrics"],
            "variants": {},
            "winner": {},
            "analyzed_at": datetime.now().isoformat()
        }
        
        # For each metric, determine the winner
        for metric in test["metrics"]:
            metric_data = {}
            best_value = None
            best_variant = None
            
            for variant_id, variant_data in variant_results.items():
                # Extract all values for this metric
                values = [result["metrics"].get(metric, 0) for result in variant_data]
                
                if not values:
                    continue
                    
                # Calculate statistics
                mean = np.mean(values)
                median = np.median(values)
                std_dev = np.std(values)
                count = len(values)
                
                # Store in analysis
                if variant_id not in analysis["variants"]:
                    analysis["variants"][variant_id] = {}
                    
                analysis["variants"][variant_id][metric] = {
                    "mean": mean,
                    "median": median,
                    "std_dev": std_dev,
                    "count": count,
                    "min": min(values),
                    "max": max(values)
                }
                
                # Determine if this is the best variant for this metric
                # For simplicity, we'll assume higher is better for all metrics
                # You might want to customize this based on the metric
                if best_value is None or mean > best_value:
                    best_value = mean
                    best_variant = variant_id
            
            # Record the winner for this metric
            if best_variant:
                analysis["winner"][metric] = {
                    "variant_id": best_variant,
                    "value": best_value
                }
        
        return analysis
    
    def plot_test_results(self, test_id: str, metric: str, save_path: Optional[str] = None) -> None:
        """
        Plot the results of an A/B test for a specific metric
        
        Args:
            test_id: ID of the test to plot
            metric: Metric to plot
            save_path: Optional path to save the plot
        """
        if test_id not in self.tests:
            raise ValueError(f"Test with ID {test_id} not found")
            
        if test_id not in self.test_results:
            raise ValueError(f"No results found for test {test_id}")
            
        test = self.tests[test_id]
        results = self.test_results[test_id]["results"]
        
        if not results:
            print("No results recorded for this test")
            return
            
        # Group results by variant
        variant_data = defaultdict(list)
        for result in results:
            if metric in result["metrics"]:
                variant_id = result["variant_id"]
                variant_data[variant_id].append(result["metrics"][metric])
        
        if not variant_data:
            print(f"No data found for metric {metric}")
            return
            
        # Create box plot
        plt.figure(figsize=(12, 6))
        
        variant_ids = list(variant_data.keys())
        data = [variant_data[v_id] for v_id in variant_ids]
        
        plt.boxplot(data, labels=variant_ids)
        plt.title(f"A/B Test Results: {test['description']} - {metric}")
        plt.ylabel(metric)
        plt.grid(True, alpha=0.3)
        
        # Add mean values as text
        means = [np.mean(values) for values in data]
        for i, mean in enumerate(means):
            plt.text(i+1, plt.ylim()[0], f"Mean: {mean:.2f}", 
                    ha='center', va='bottom', rotation=45)
        
        if save_path:
            plt.savefig(save_path)
            print(f"Saved test results plot to {save_path}")
        else:
            plt.show()
    
    def generate_test_report(self, test_id: str, path: Optional[str] = None) -> str:
        """
        Generate an HTML report for a specific A/B test
        
        Args:
            test_id: ID of the test
            path: Optional path to save the HTML report
            
        Returns:
            HTML report content
        """
        if test_id not in self.tests:
            raise ValueError(f"Test with ID {test_id} not found")
        
        # Analyze the test
        analysis = self.analyze_test(test_id)
        test = self.tests[test_id]
        
        # Create HTML report
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>A/B Test Report: {test_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .highlight {{ background-color: #e6f7e6; }}
                .section {{ margin-bottom: 30px; }}
            </style>
        </head>
        <body>
            <h1>A/B Test Report: {test_id}</h1>
            
            <div class="section">
                <h2>Test Information</h2>
                <table>
                    <tr>
                        <th>ID</th>
                        <td>{test_id}</td>
                    </tr>
                    <tr>
                        <th>Description</th>
                        <td>{test["description"]}</td>
                    </tr>
                    <tr>
                        <th>Status</th>
                        <td>{test["status"]}</td>
                    </tr>
                    <tr>
                        <th>Start Date</th>
                        <td>{test["start_date"]}</td>
                    </tr>
                    <tr>
                        <th>End Date</th>
                        <td>{test.get("end_date", "Active")}</td>
                    </tr>
                    <tr>
                        <th>Symbols</th>
                        <td>{", ".join(test["symbols"])}</td>
                    </tr>
                    <tr>
                        <th>Total Results</th>
                        <td>{analysis["total_results"]}</td>
                    </tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Variants</h2>
                <table>
                    <tr>
                        <th>Variant ID</th>
                        <th>Description</th>
                        <th>Configuration</th>
                    </tr>
        """
        
        # Add variant information
        for variant in test["variants"]:
            variant_id = variant["id"]
            config = {k: v for k, v in variant.items() if k != "id"}
            html += f"""
                    <tr>
                        <td>{variant_id}</td>
                        <td>{variant.get("description", "")}</td>
                        <td><pre>{json.dumps(config, indent=2)}</pre></td>
                    </tr>
            """
        
        html += """
                </table>
            </div>
            
            <div class="section">
                <h2>Results by Metric</h2>
        """
        
        # Add results for each metric
        for metric in test["metrics"]:
            html += f"""
                <h3>Metric: {metric}</h3>
                <table>
                    <tr>
                        <th>Variant</th>
                        <th>Mean</th>
                        <th>Median</th>
                        <th>Std Dev</th>
                        <th>Min</th>
                        <th>Max</th>
                        <th>Count</th>
                    </tr>
            """
            
            # Determine winner for this metric
            winner = analysis["winner"].get(metric, {}).get("variant_id", "")
            
            # Add results for each variant
            for variant_id, metrics in analysis["variants"].items():
                if metric not in metrics:
                    continue
                    
                metric_data = metrics[metric]
                is_winner = variant_id == winner
                
                html += f"""
                    <tr class="{'highlight' if is_winner else ''}">
                        <td>{variant_id}{' (Winner)' if is_winner else ''}</td>
                        <td>{metric_data['mean']:.4f}</td>
                        <td>{metric_data['median']:.4f}</td>
                        <td>{metric_data['std_dev']:.4f}</td>
                        <td>{metric_data['min']:.4f}</td>
                        <td>{metric_data['max']:.4f}</td>
                        <td>{metric_data['count']}</td>
                    </tr>
                """
            
            html += "</table>"
        
        # Add overall summary
        html += """
            </div>
            
            <div class="section">
                <h2>Summary</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Winning Variant</th>
                        <th>Value</th>
                    </tr>
        """
        
        for metric, winner in analysis["winner"].items():
            html += f"""
                    <tr>
                        <td>{metric}</td>
                        <td>{winner['variant_id']}</td>
                        <td>{winner['value']:.4f}</td>
                    </tr>
            """
        
        html += """
                </table>
            </div>
            
            <p>Report generated at: {}</p>
        </body>
        </html>
        """.format(datetime.now().isoformat())
        
        # Save the report if path provided
        if path:
            with open(path, 'w') as f:
                f.write(html)
            print(f"Test report saved to {path}")
        
        return html