#!/usr/bin/env python3
"""
Statistical analysis and reporting nodes.
"""

import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd

from dag_engine import BaseNode, NodeResult, NodeStatus, ExecutionContext, register_node


@register_node("StatisticalSignificanceTest")
class StatisticalSignificanceTestNode(BaseNode):
    """Compare two models with statistical significance testing."""
    
    def run(self, context: ExecutionContext) -> NodeResult:
        try:
            model1_results = self.config.get("model1_results")
            model2_results = self.config.get("model2_results")
            
            if not model1_results or not model2_results:
                return NodeResult(
                    status=NodeStatus.FAILED,
                    error="Both model1_results and model2_results must be specified in config"
                )
            
            if not context.has_artifact(model1_results):
                return NodeResult(
                    status=NodeStatus.FAILED,
                    error=f"Model 1 results '{model1_results}' not found in context"
                )
            
            if not context.has_artifact(model2_results):
                return NodeResult(
                    status=NodeStatus.FAILED,
                    error=f"Model 2 results '{model2_results}' not found in context"
                )
            
            results1 = context.get_artifact(model1_results)
            results2 = context.get_artifact(model2_results)
            
            # Extract accuracies/scores for comparison
            if isinstance(results1, dict) and "accuracy" in results1:
                score1 = results1["accuracy"]
                n1 = results1.get("total_predictions", results1.get("total_examples", 0))
            else:
                return NodeResult(
                    status=NodeStatus.FAILED,
                    error="Model 1 results must contain 'accuracy' field"
                )
            
            if isinstance(results2, dict) and "accuracy" in results2:
                score2 = results2["accuracy"]
                n2 = results2.get("total_predictions", results2.get("total_examples", 0))
            else:
                return NodeResult(
                    status=NodeStatus.FAILED,
                    error="Model 2 results must contain 'accuracy' field"
                )
            
            # Perform statistical significance test (McNemar's test for paired samples)
            try:
                from scipy.stats import chi2_contingency
                import numpy as np
                
                # For simplicity, use a z-test for proportions
                # In practice, you'd want to use the actual predictions for McNemar's test
                p1, p2 = score1, score2
                n1, n2 = max(n1, 1), max(n2, 1)  # Avoid division by zero
                
                # Pooled proportion and standard error
                p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
                se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
                
                # Z-statistic
                z_stat = (p1 - p2) / se if se > 0 else 0
                
                # P-value (two-tailed)
                from scipy.stats import norm
                p_value = 2 * (1 - norm.cdf(abs(z_stat)))
                
                # Effect size (Cohen's h)
                h = 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))
                
                significance_test = {
                    "model1_accuracy": p1,
                    "model2_accuracy": p2,
                    "difference": p1 - p2,
                    "z_statistic": z_stat,
                    "p_value": p_value,
                    "is_significant": p_value < 0.05,
                    "effect_size_h": h,
                    "sample_size_1": n1,
                    "sample_size_2": n2,
                    "test_type": "two_proportion_z_test"
                }
                
            except ImportError:
                # Fallback if scipy is not available
                difference = score1 - score2
                significance_test = {
                    "model1_accuracy": score1,
                    "model2_accuracy": score2,
                    "difference": difference,
                    "note": "Statistical significance test requires scipy",
                    "substantial_difference": abs(difference) > 0.05  # Simple threshold
                }
            
            outputs = {"significance_test": significance_test}
            
            return NodeResult(
                status=NodeStatus.SUCCESS,
                outputs=outputs
            )
            
        except Exception as e:
            return NodeResult(
                status=NodeStatus.FAILED,
                error=f"Failed to perform statistical significance test: {str(e)}"
            )


@register_node("ReportGenerator")
class ReportGeneratorNode(BaseNode):
    """Generate evaluation reports in various formats."""
    
    def run(self, context: ExecutionContext) -> NodeResult:
        try:
            if not context.has_artifact("evaluation_results"):
                return NodeResult(
                    status=NodeStatus.FAILED,
                    error="evaluation_results not found in context"
                )
            
            results = context.get_artifact("evaluation_results")
            format_type = self.config.get("format", "text")
            output_file = self.config.get("output_file")
            
            # Generate report based on format
            if format_type == "text":
                report = self._generate_text_report(results, context)
            elif format_type == "json":
                report = self._generate_json_report(results, context)
            elif format_type == "html":
                report = self._generate_html_report(results, context)
            else:
                return NodeResult(
                    status=NodeStatus.FAILED,
                    error=f"Unsupported report format: {format_type}"
                )
            
            # Save to file if specified
            if output_file:
                try:
                    with open(output_file, 'w') as f:
                        f.write(report)
                except Exception as e:
                    return NodeResult(
                        status=NodeStatus.FAILED,
                        error=f"Failed to write report to {output_file}: {str(e)}"
                    )
            
            outputs = {
                "report": report,
                "format": format_type,
                "output_file": output_file
            }
            
            return NodeResult(
                status=NodeStatus.SUCCESS,
                outputs=outputs
            )
            
        except Exception as e:
            return NodeResult(
                status=NodeStatus.FAILED,
                error=f"Failed to generate report: {str(e)}"
            )
    
    def _generate_text_report(self, results: Dict[str, Any], context: ExecutionContext) -> str:
        """Generate a text-based evaluation report."""
        lines = []
        lines.append("=" * 60)
        lines.append("EVALUATION REPORT")
        lines.append("=" * 60)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Model information
        model_info = {}
        try:
            if context.has_artifact("model_info"):
                mi = context.get_artifact("model_info")
                if isinstance(mi, dict):
                    model_info = mi
        except Exception:
            model_info = {}
        if model_info:
            lines.append("MODEL INFORMATION:")
            lines.append(f"  Provider: {model_info.get('provider', 'unknown')}")
            lines.append(f"  Model: {model_info.get('model', 'unknown')}")
            if model_info.get('temperature') is not None:
                lines.append(f"  Temperature: {model_info.get('temperature')}")
            if model_info.get('max_tokens'):
                lines.append(f"  Max Tokens: {model_info.get('max_tokens')}")
            lines.append("")
        else:
            # Fallback to model name from results
            if 'model_name' in results:
                lines.append("MODEL INFORMATION:")
                lines.append(f"  Model: {results.get('model_name')}")
                lines.append("")
        
        # Dataset information
        dataset_info = {}
        try:
            if context.has_artifact("dataset_info"):
                di = context.get_artifact("dataset_info")
                if isinstance(di, dict):
                    dataset_info = di
        except Exception:
            dataset_info = {}
        if dataset_info:
            lines.append("DATASET INFORMATION:")
            lines.append(f"  Total Examples: {dataset_info.get('rows', 'unknown')}")
            if 'positive' in dataset_info:
                lines.append(f"  Positive: {dataset_info['positive']}")
                lines.append(f"  Negative: {dataset_info['negative']}")
                lines.append(f"  Neutral: {dataset_info['neutral']}")
            lines.append("")
        
        # Evaluation results
        lines.append("EVALUATION RESULTS:")
        lines.append(f"  Overall Accuracy: {results.get('accuracy', 0):.1%}")
        lines.append(f"  Correct Predictions: {results.get('correct_predictions', 0)}")
        lines.append(f"  Total Predictions: {results.get('total_predictions', 0)}")
        
        if results.get('prediction_errors', 0) > 0:
            lines.append(f"  Prediction Errors: {results['prediction_errors']}")
        
        lines.append("")
        
        # Per-class results
        if "class_stats" in results:
            lines.append("PER-CLASS RESULTS:")
            for class_name, stats in results["class_stats"].items():
                lines.append(f"  {class_name}:")
                lines.append(f"    Accuracy: {stats.get('accuracy', 0):.1%}")
                lines.append(f"    Correct: {stats.get('correct', 0)}/{stats.get('total', 0)}")
        
        # Multi-metric results
        if "per_class_metrics" in results:
            lines.append("")
            lines.append("DETAILED METRICS:")
            for class_name, metrics in results["per_class_metrics"].items():
                lines.append(f"  {class_name}:")
                lines.append(f"    Precision: {metrics.get('precision', 0):.3f}")
                lines.append(f"    Recall: {metrics.get('recall', 0):.3f}")
                lines.append(f"    F1-Score: {metrics.get('f1', 0):.3f}")
                lines.append(f"    Support: {metrics.get('support', 0)}")
            
            if "macro_avg" in results:
                lines.append("")
                lines.append("MACRO AVERAGES:")
                lines.append(f"  Precision: {results['macro_avg'].get('precision', 0):.3f}")
                lines.append(f"  Recall: {results['macro_avg'].get('recall', 0):.3f}")
                lines.append(f"  F1-Score: {results['macro_avg'].get('f1', 0):.3f}")
        
        # Performance information
        prediction_summary = {}
        try:
            if context.has_artifact("prediction_summary"):
                ps = context.get_artifact("prediction_summary")
                if isinstance(ps, dict):
                    prediction_summary = ps
        except Exception:
            prediction_summary = {}
        if prediction_summary:
            lines.append("")
            lines.append("PERFORMANCE:")
            lines.append(f"  Total Time: {prediction_summary.get('prediction_time', 0):.2f}s")
            lines.append(f"  Avg Time/Prediction: {prediction_summary.get('avg_time_per_prediction', 0):.3f}s")
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def _generate_json_report(self, results: Dict[str, Any], context: ExecutionContext) -> str:
        """Generate a JSON evaluation report."""
        # Safely fetch optional artifacts
        model_info = {}
        dataset_info = {}
        prediction_summary = {}
        try:
            if context.has_artifact("model_info"):
                mi = context.get_artifact("model_info")
                if isinstance(mi, dict):
                    model_info = mi
            if context.has_artifact("dataset_info"):
                di = context.get_artifact("dataset_info")
                if isinstance(di, dict):
                    dataset_info = di
            if context.has_artifact("prediction_summary"):
                ps = context.get_artifact("prediction_summary")
                if isinstance(ps, dict):
                    prediction_summary = ps
        except Exception:
            pass

        report_data = {
            "timestamp": datetime.now().isoformat(),
            "evaluation_results": results,
            "model_info": model_info,
            "dataset_info": dataset_info,
            "prediction_summary": prediction_summary
        }
        
        return json.dumps(report_data, indent=2)
    
    def _generate_html_report(self, results: Dict[str, Any], context: ExecutionContext) -> str:
        """Generate an HTML evaluation report."""
        html = []
        html.append("<!DOCTYPE html>")
        html.append("<html><head><title>Evaluation Report</title>")
        html.append("<style>")
        html.append("body { font-family: Arial, sans-serif; margin: 40px; }")
        html.append("table { border-collapse: collapse; width: 100%; }")
        html.append("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
        html.append("th { background-color: #f2f2f2; }")
        html.append(".metric { font-weight: bold; }")
        html.append("</style></head><body>")
        
        html.append("<h1>Evaluation Report</h1>")
        html.append(f"<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
        
        # Summary table
        html.append("<h2>Summary</h2>")
        html.append("<table>")
        html.append("<tr><th>Metric</th><th>Value</th></tr>")
        html.append(f"<tr><td class='metric'>Accuracy</td><td>{results.get('accuracy', 0):.1%}</td></tr>")
        html.append(f"<tr><td>Correct Predictions</td><td>{results.get('correct_predictions', 0)}</td></tr>")
        html.append(f"<tr><td>Total Predictions</td><td>{results.get('total_predictions', 0)}</td></tr>")
        html.append("</table>")
        
        # Per-class results
        if "class_stats" in results:
            html.append("<h2>Per-Class Results</h2>")
            html.append("<table>")
            html.append("<tr><th>Class</th><th>Accuracy</th><th>Correct</th><th>Total</th></tr>")
            for class_name, stats in results["class_stats"].items():
                html.append(f"<tr>")
                html.append(f"<td>{class_name}</td>")
                html.append(f"<td>{stats.get('accuracy', 0):.1%}</td>")
                html.append(f"<td>{stats.get('correct', 0)}</td>")
                html.append(f"<td>{stats.get('total', 0)}</td>")
                html.append(f"</tr>")
            html.append("</table>")
        
        html.append("</body></html>")
        
        return "\n".join(html)


@register_node("CostAnalysis")
class CostAnalysisNode(BaseNode):
    """Analyze API costs for model evaluations."""
    
    def run(self, context: ExecutionContext) -> NodeResult:
        try:
            # Get model info and prediction summary
            model_info = {}
            prediction_summary = {}
            try:
                if context.has_artifact("model_info"):
                    mi = context.get_artifact("model_info")
                    if isinstance(mi, dict):
                        model_info = mi
                if context.has_artifact("prediction_summary"):
                    ps = context.get_artifact("prediction_summary")
                    if isinstance(ps, dict):
                        prediction_summary = ps
            except Exception:
                pass
            
            provider = model_info.get("provider", "unknown")
            model = model_info.get("model", "unknown")
            total_predictions = prediction_summary.get("total_predictions", 0)
            
            # Estimate costs based on provider and model
            cost_per_1k_tokens = self._get_cost_per_1k_tokens(provider, model)
            estimated_tokens_per_prediction = self.config.get("tokens_per_prediction", 100)
            
            total_tokens = total_predictions * estimated_tokens_per_prediction
            estimated_cost = (total_tokens / 1000) * cost_per_1k_tokens
            
            cost_analysis = {
                "provider": provider,
                "model": model,
                "total_predictions": total_predictions,
                "estimated_tokens_per_prediction": estimated_tokens_per_prediction,
                "total_estimated_tokens": total_tokens,
                "cost_per_1k_tokens": cost_per_1k_tokens,
                "estimated_total_cost": estimated_cost,
                "currency": "USD"
            }
            
            outputs = {"cost_analysis": cost_analysis}
            
            return NodeResult(
                status=NodeStatus.SUCCESS,
                outputs=outputs
            )
            
        except Exception as e:
            return NodeResult(
                status=NodeStatus.FAILED,
                error=f"Failed to analyze costs: {str(e)}"
            )
    
    def _get_cost_per_1k_tokens(self, provider: str, model: str) -> float:
        """Get estimated cost per 1K tokens for different models."""
        # Simplified cost estimates (would need regular updates in practice)
        costs = {
            "openai": {
                "gpt-4o": 0.015,
                "gpt-4o-mini": 0.00015,
                "gpt-4": 0.03,
                "gpt-3.5-turbo": 0.001,
            },
            "anthropic": {
                "claude-3-5-sonnet": 0.003,
                "claude-3-haiku": 0.00025,
                "claude-3-opus": 0.015,
            },
            "ollama": {
                # Local models - no API cost
                "default": 0.0
            }
        }
        
        provider_costs = costs.get(provider, {})
        return provider_costs.get(model, provider_costs.get("default", 0.001))
