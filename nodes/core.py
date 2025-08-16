#!/usr/bin/env python3
"""
Core node utilities and base classes.
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import dspy

from dag_engine import BaseNode, NodeResult, NodeStatus, ExecutionContext, register_node


class SentimentSignature(dspy.Signature):
    """Classify the sentiment of a sentence."""
    sentence = dspy.InputField(desc="A sentence to classify.")
    sentiment = dspy.OutputField(desc="The sentiment of the sentence, which can be Positive, Negative, or Neutral.")


@register_node("LoadCSV")
class LoadCSVNode(BaseNode):
    """Load data from a CSV file."""
    
    def run(self, context: ExecutionContext) -> NodeResult:
        filepath = self.config.get("filepath")
        if not filepath:
            return NodeResult(status=NodeStatus.FAILED, error="No filepath specified")
        
        try:
            df = pd.read_csv(filepath)
            outputs = {"dataset": df}
            
            context.set_artifact("dataset_info", {
                "rows": len(df),
                "columns": list(df.columns),
                "filepath": filepath
            })
            
            return NodeResult(
                status=NodeStatus.SUCCESS,
                outputs=outputs
            )
        except Exception as e:
            return NodeResult(
                status=NodeStatus.FAILED,
                error=f"Failed to load CSV: {str(e)}"
            )


@register_node("CreateTestData")
class CreateTestDataNode(BaseNode):
    """Create synthetic test data for evaluation."""
    
    def run(self, context: ExecutionContext) -> NodeResult:
        try:
            # Create test examples similar to sentiment_eval.py
            examples = [
                {"sentence": "It's a beautiful day!", "sentiment": "Positive"},
                {"sentence": "I'm not sure how I feel about this.", "sentiment": "Neutral"},
                {"sentence": "The service was terrible.", "sentiment": "Negative"},
                {"sentence": "This movie was absolutely fantastic.", "sentiment": "Positive"},
                {"sentence": "I hate waiting in long lines.", "sentiment": "Negative"},
                {"sentence": "The weather is okay today.", "sentiment": "Neutral"},
                {"sentence": "I love spending time with my family.", "sentiment": "Positive"},
                {"sentence": "This restaurant has average food.", "sentiment": "Neutral"},
                {"sentence": "The customer support was unhelpful.", "sentiment": "Negative"},
                {"sentence": "This book changed my life!", "sentiment": "Positive"}
            ]
            
            # Convert to DSPy examples
            dspy_examples = []
            for ex in examples:
                dspy_example = dspy.Example(
                    sentence=ex["sentence"], 
                    sentiment=ex["sentiment"]
                ).with_inputs("sentence")
                dspy_examples.append(dspy_example)
            
            # Map outputs to requested names if provided; default to dataset/raw_data
            outputs = {}
            if self.outputs:
                # First output: dataset examples
                outputs[self.outputs[0]] = dspy_examples
                # Optional second output: raw_data
                if len(self.outputs) > 1:
                    outputs[self.outputs[1]] = examples
            else:
                outputs = {"dataset": dspy_examples, "raw_data": examples}
            
            context.set_artifact("dataset_info", {
                "rows": len(examples),
                "positive": sum(1 for ex in examples if ex["sentiment"] == "Positive"),
                "negative": sum(1 for ex in examples if ex["sentiment"] == "Negative"), 
                "neutral": sum(1 for ex in examples if ex["sentiment"] == "Neutral")
            })
            
            return NodeResult(
                status=NodeStatus.SUCCESS,
                outputs=outputs
            )
            
        except Exception as e:
            return NodeResult(
                status=NodeStatus.FAILED,
                error=f"Failed to create test data: {str(e)}"
            )
