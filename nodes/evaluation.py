#!/usr/bin/env python3
"""
Model evaluation and comparison nodes.
"""

import dspy
from typing import Dict, List, Any, Optional
import time

from dag_engine import BaseNode, NodeResult, NodeStatus, ExecutionContext, register_node
from .core import SentimentSignature

# Import synthetic nodes
from .synthetic import SyntheticGeneratorNode, ManualTestCaseLoaderNode, TestCaseMergerNode, BatchEvaluatorNode


@register_node("SentimentAnalysis")
class SentimentAnalysisNode(BaseNode):
    """Run sentiment analysis using DSPy."""
    
    def run(self, context: ExecutionContext) -> NodeResult:
        try:
            # Get the DSPy language model from context
            if not context.has_artifact("dspy_lm"):
                return NodeResult(
                    status=NodeStatus.FAILED,
                    error="dspy_lm not found in context. Run DSPySetup node first."
                )
            
            if not context.has_artifact("dataset"):
                return NodeResult(
                    status=NodeStatus.FAILED,
                    error="dataset not found in context. Create or load dataset first."
                )
            
            # Get dataset and language model
            dataset = context.get_artifact("dataset")
            dspy_lm = context.get_artifact("dspy_lm")
            
            # Configure DSPy with the language model
            dspy.configure(lm=dspy_lm)
            
            # Create sentiment predictor
            sentiment_predictor = dspy.Predict(SentimentSignature)
            
            # Run predictions
            predictions = []
            start_time = time.time()
            
            for i, example in enumerate(dataset):
                try:
                    prediction = sentiment_predictor(sentence=example.sentence)
                    
                    pred_result = {
                        "example_id": i,
                        "sentence": example.sentence,
                        "prediction": prediction.sentiment,
                        "actual": getattr(example, "sentiment", None)
                    }
                    predictions.append(pred_result)
                    
                except Exception as e:
                    pred_result = {
                        "example_id": i,
                        "sentence": example.sentence,
                        "prediction": None,
                        "actual": getattr(example, "sentiment", None),
                        "error": str(e)
                    }
                    predictions.append(pred_result)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            outputs = {
                "predictions": predictions,
                "prediction_time": total_time,
                "examples_processed": len(predictions)
            }
            
            # Add prediction summary to context
            context.set_artifact("prediction_summary", {
                "total_predictions": len(predictions),
                "successful_predictions": sum(1 for p in predictions if p.get("prediction") is not None),
                "failed_predictions": sum(1 for p in predictions if p.get("error") is not None),
                "prediction_time": total_time,
                "avg_time_per_prediction": total_time / len(predictions) if predictions else 0
            })
            
            # Also set predictions for downstream nodes
            context.set_artifact("predictions", predictions)
            
            return NodeResult(
                status=NodeStatus.SUCCESS,
                outputs=outputs
            )
            
        except Exception as e:
            return NodeResult(
                status=NodeStatus.FAILED,
                error=f"Failed to run sentiment analysis: {str(e)}"
            )


@register_node("EvaluateModel")
class EvaluateModelNode(BaseNode):
    """Evaluate model predictions against ground truth."""
    
    def run(self, context: ExecutionContext) -> NodeResult:
        try:
            if not context.has_artifact("dataset"):
                return NodeResult(
                    status=NodeStatus.FAILED,
                    error="Dataset not found in context"
                )
            
            if not context.has_artifact("predictions"):
                return NodeResult(
                    status=NodeStatus.FAILED,
                    error="Predictions not found in context"
                )
            
            dataset = context.get_artifact("dataset")
            predictions = context.get_artifact("predictions")
            
            # Calculate accuracy
            correct_predictions = 0
            total_predictions = 0
            prediction_errors = 0
            
            for pred in predictions:
                if pred.get("error"):
                    prediction_errors += 1
                    continue
                    
                if pred.get("actual") and pred.get("prediction"):
                    total_predictions += 1
                    if pred["actual"] == pred["prediction"]:
                        correct_predictions += 1
            
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
            
            # Calculate per-class metrics
            class_stats = {}
            for pred in predictions:
                if pred.get("actual") and pred.get("prediction") and not pred.get("error"):
                    actual = pred["actual"]
                    predicted = pred["prediction"]
                    
                    if actual not in class_stats:
                        class_stats[actual] = {"total": 0, "correct": 0}
                    
                    class_stats[actual]["total"] += 1
                    if actual == predicted:
                        class_stats[actual]["correct"] += 1
            
            # Calculate per-class accuracy
            for class_name, stats in class_stats.items():
                stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
            
            # Determine model name (optional)
            model_info = {}
            try:
                if context.has_artifact("model_info"):
                    mi = context.get_artifact("model_info")
                    if isinstance(mi, dict):
                        model_info = mi
            except Exception:
                model_info = {}

            evaluation_results = {
                "accuracy": accuracy,
                "total_examples": len(dataset),
                "total_predictions": total_predictions,
                "correct_predictions": correct_predictions,
                "prediction_errors": prediction_errors,
                "class_stats": class_stats,
                "model_name": model_info.get("model", "unknown")
            }
            
            outputs = {"evaluation_results": evaluation_results}
            
            # Also set evaluation_results in context for downstream nodes
            context.set_artifact("evaluation_results", evaluation_results)
            
            return NodeResult(
                status=NodeStatus.SUCCESS,
                outputs=outputs
            )
            
        except Exception as e:
            return NodeResult(
                status=NodeStatus.FAILED,
                error=f"Failed to evaluate model: {str(e)}"
            )


@register_node("MultiMetricEvaluateModel")
class MultiMetricEvaluateModelNode(BaseNode):
    """Evaluate model with multiple metrics including precision, recall, F1."""
    
    def run(self, context: ExecutionContext) -> NodeResult:
        try:
            if not context.has_artifact("predictions"):
                return NodeResult(
                    status=NodeStatus.FAILED,
                    error="Predictions not found in context"
                )
            
            predictions = context.get_artifact("predictions")
            
            # Collect all classes
            all_classes = set()
            valid_predictions = []
            
            for pred in predictions:
                if pred.get("actual") and pred.get("prediction") and not pred.get("error"):
                    all_classes.add(pred["actual"])
                    all_classes.add(pred["prediction"])
                    valid_predictions.append(pred)
            
            if not valid_predictions:
                return NodeResult(
                    status=NodeStatus.FAILED,
                    error="No valid predictions found for evaluation"
                )
            
            # Calculate metrics for each class
            metrics = {}
            for class_name in all_classes:
                tp = sum(1 for p in valid_predictions if p["actual"] == class_name and p["prediction"] == class_name)
                fp = sum(1 for p in valid_predictions if p["actual"] != class_name and p["prediction"] == class_name)
                fn = sum(1 for p in valid_predictions if p["actual"] == class_name and p["prediction"] != class_name)
                tn = sum(1 for p in valid_predictions if p["actual"] != class_name and p["prediction"] != class_name)
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                
                metrics[class_name] = {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "support": tp + fn,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn
                }
            
            # Calculate overall accuracy
            correct = sum(1 for p in valid_predictions if p["actual"] == p["prediction"])
            accuracy = correct / len(valid_predictions)
            
            # Calculate macro averages
            macro_precision = sum(m["precision"] for m in metrics.values()) / len(metrics)
            macro_recall = sum(m["recall"] for m in metrics.values()) / len(metrics)
            macro_f1 = sum(m["f1"] for m in metrics.values()) / len(metrics)
            
            # Calculate weighted averages
            total_support = sum(m["support"] for m in metrics.values())
            weighted_precision = sum(m["precision"] * m["support"] for m in metrics.values()) / total_support if total_support > 0 else 0
            weighted_recall = sum(m["recall"] * m["support"] for m in metrics.values()) / total_support if total_support > 0 else 0
            weighted_f1 = sum(m["f1"] * m["support"] for m in metrics.values()) / total_support if total_support > 0 else 0
            
            evaluation_results = {
                "accuracy": accuracy,
                "per_class_metrics": metrics,
                "macro_avg": {
                    "precision": macro_precision,
                    "recall": macro_recall,
                    "f1": macro_f1
                },
                "weighted_avg": {
                    "precision": weighted_precision,
                    "recall": weighted_recall,
                    "f1": weighted_f1
                },
                "total_predictions": len(valid_predictions),
                "total_examples": len(predictions),
                "prediction_errors": len(predictions) - len(valid_predictions)
            }
            
            outputs = {"evaluation_results": evaluation_results}
            
            return NodeResult(
                status=NodeStatus.SUCCESS,
                outputs=outputs
            )
            
        except Exception as e:
            return NodeResult(
                status=NodeStatus.FAILED,
                error=f"Failed to evaluate model with multiple metrics: {str(e)}"
            )


@register_node("PromptVariationGenerator")
class PromptVariationGeneratorNode(BaseNode):
    """Generate variations of prompts for robustness testing."""
    
    def run(self, context: ExecutionContext) -> NodeResult:
        try:
            if not context.has_artifact("base_prompt"):
                return NodeResult(
                    status=NodeStatus.FAILED,
                    error="Base prompt not found in context"
                )
            
            base_prompt = context.get_artifact("base_prompt")
            num_variations = self.config.get("num_variations", 5)
            variation_strategies = self.config.get("strategies", [
                "rephrase", "simplify", "elaborate", "formal", "casual"
            ])
            
            variations = []
            
            # Add base prompt as first variation
            variations.append({
                "id": "base",
                "prompt": base_prompt,
                "strategy": "original",
                "metadata": {"is_base": True}
            })
            
            # Generate variations using different strategies
            for i, strategy in enumerate(variation_strategies[:num_variations-1]):
                variation = {
                    "id": f"variation_{i+1}",
                    "prompt": self._generate_variation(base_prompt, strategy),
                    "strategy": strategy,
                    "metadata": {"variation_index": i+1}
                }
                variations.append(variation)
            
            outputs = {
                "prompt_variations": variations,
                "num_variations": len(variations),
                "base_prompt": base_prompt
            }
            
            # Store in context for downstream nodes
            context.set_artifact("prompt_variations", variations)
            
            return NodeResult(
                status=NodeStatus.SUCCESS,
                outputs=outputs
            )
            
        except Exception as e:
            return NodeResult(
                status=NodeStatus.FAILED,
                error=f"Failed to generate prompt variations: {str(e)}"
            )
    
    def _generate_variation(self, base_prompt: str, strategy: str) -> str:
        """Generate a prompt variation using the specified strategy."""
        
        if strategy == "rephrase":
            return f"Please {base_prompt.lower().lstrip('please ')}"
        elif strategy == "simplify":
            # Remove complex words and shorten
            words = base_prompt.split()
            simplified = " ".join(words[:min(len(words), 15)])
            return simplified
        elif strategy == "elaborate":
            return f"{base_prompt} Please provide detailed explanations and examples."
        elif strategy == "formal":
            return f"I would appreciate it if you could {base_prompt.lower().lstrip('please ')}"
        elif strategy == "casual":
            return f"Hey, can you {base_prompt.lower().lstrip('please ')}"
        else:
            # Default: minor rephrasing
            return base_prompt.replace("Please", "Could you please")


@register_node("DatasetAblationGenerator")
class DatasetAblationGeneratorNode(BaseNode):
    """Generate ablated versions of datasets for robustness testing."""
    
    def run(self, context: ExecutionContext) -> NodeResult:
        try:
            if not context.has_artifact("dataset"):
                return NodeResult(
                    status=NodeStatus.FAILED,
                    error="Dataset not found in context"
                )
            
            dataset = context.get_artifact("dataset")
            ablation_strategies = self.config.get("strategies", [
                "remove_random", "remove_class", "subsample", "add_noise"
            ])
            ablation_ratios = self.config.get("ratios", [0.1, 0.25, 0.5])
            
            ablated_datasets = {}
            
            for strategy in ablation_strategies:
                for ratio in ablation_ratios:
                    ablated_key = f"{strategy}_{ratio}"
                    ablated_data = self._apply_ablation(dataset, strategy, ratio)
                    
                    ablated_datasets[ablated_key] = {
                        "data": ablated_data,
                        "strategy": strategy,
                        "ratio": ratio,
                        "original_size": len(dataset),
                        "ablated_size": len(ablated_data),
                        "metadata": {
                            "ablation_type": strategy,
                            "ablation_ratio": ratio,
                            "reduction_factor": len(ablated_data) / len(dataset) if dataset else 0
                        }
                    }
            
            outputs = {
                "ablated_datasets": ablated_datasets,
                "num_ablations": len(ablated_datasets),
                "original_dataset_size": len(dataset)
            }
            
            # Store in context for downstream nodes
            context.set_artifact("ablated_datasets", ablated_datasets)
            
            return NodeResult(
                status=NodeStatus.SUCCESS,
                outputs=outputs
            )
            
        except Exception as e:
            return NodeResult(
                status=NodeStatus.FAILED,
                error=f"Failed to generate dataset ablations: {str(e)}"
            )
    
    def _apply_ablation(self, dataset: list, strategy: str, ratio: float) -> list:
        """Apply ablation strategy to dataset."""
        import random
        
        if strategy == "remove_random":
            # Randomly remove ratio of samples
            keep_count = int(len(dataset) * (1 - ratio))
            return random.sample(dataset, keep_count)
            
        elif strategy == "subsample":
            # Keep only ratio of samples
            keep_count = int(len(dataset) * ratio)
            return random.sample(dataset, min(keep_count, len(dataset)))
            
        elif strategy == "remove_class":
            # Remove samples from specific classes
            if not dataset:
                return dataset
                
            # Try to find class information
            classes = set()
            for item in dataset:
                if hasattr(item, 'sentiment'):
                    classes.add(item.sentiment)
                elif isinstance(item, dict) and 'label' in item:
                    classes.add(item['label'])
                elif isinstance(item, dict) and 'category' in item:
                    classes.add(item['category'])
            
            if classes:
                # Remove ratio of classes
                classes_to_remove = random.sample(list(classes), int(len(classes) * ratio))
                filtered_data = []
                for item in dataset:
                    item_class = None
                    if hasattr(item, 'sentiment'):
                        item_class = item.sentiment
                    elif isinstance(item, dict) and 'label' in item:
                        item_class = item['label']
                    elif isinstance(item, dict) and 'category' in item:
                        item_class = item['category']
                    
                    if item_class not in classes_to_remove:
                        filtered_data.append(item)
                
                return filtered_data
            else:
                # Fallback to random removal
                return self._apply_ablation(dataset, "remove_random", ratio)
                
        elif strategy == "add_noise":
            # Add noise to ratio of samples (for text, this means minor modifications)
            noisy_data = []
            for item in dataset:
                if random.random() < ratio:
                    # Add noise to this item
                    noisy_item = self._add_noise_to_item(item)
                    noisy_data.append(noisy_item)
                else:
                    noisy_data.append(item)
            return noisy_data
            
        else:
            # Default: return original dataset
            return dataset
    
    def _add_noise_to_item(self, item):
        """Add noise to a single data item."""
        import random
        
        if hasattr(item, 'sentence'):
            # Add typos or minor changes to sentence
            sentence = item.sentence
            words = sentence.split()
            if words:
                # Randomly duplicate a word
                random_word = random.choice(words)
                words.append(random_word)
                
                # Create new item with noisy sentence
                noisy_item = type(item)(
                    sentence=" ".join(words),
                    sentiment=getattr(item, 'sentiment', None)
                )
                return noisy_item
        
        elif isinstance(item, dict) and 'text' in item:
            # Duplicate the item and modify text
            noisy_item = item.copy()
            words = item['text'].split()
            if words:
                random_word = random.choice(words)
                words.append(random_word)
                noisy_item['text'] = " ".join(words)
            return noisy_item
        
        # Return original if we can't add noise
        return item
