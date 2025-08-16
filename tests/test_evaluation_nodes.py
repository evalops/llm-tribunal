#!/usr/bin/env python3
"""
Tests for evaluation node implementations.
"""

import pytest
import pandas as pd
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from evaluation_nodes import (
    CreateTestDataNode, SetupOllamaNode, SentimentAnalysisNode,
    EvaluateModelNode, ReportGeneratorNode
)
from dag_engine import ExecutionContext, NodeStatus


class TestCreateTestDataNode:
    """Test CreateTestDataNode functionality."""
    
    def test_create_test_data_success(self):
        """Test successful test data creation."""
        node = CreateTestDataNode("test_data")
        context = ExecutionContext()
        
        result = node.run(context)
        
        assert result.status == NodeStatus.SUCCESS
        assert "dataset" in result.outputs
        assert "raw_data" in result.outputs
        
        # Check dataset structure
        dataset = result.outputs["dataset"]
        assert len(dataset) == 10  # Default test examples
        
        # Check that context has dataset info
        assert context.has_artifact("dataset_info")
        info = context.get_artifact("dataset_info")
        assert info["rows"] == 10
        assert "positive" in info
        assert "negative" in info
        assert "neutral" in info
    
    def test_create_test_data_dspy_format(self):
        """Test that created data is in correct DSPy format."""
        node = CreateTestDataNode("test_data")
        context = ExecutionContext()
        
        result = node.run(context)
        dataset = result.outputs["dataset"]
        
        # Check first example
        example = dataset[0]
        assert hasattr(example, "sentence")
        assert hasattr(example, "sentiment")
        assert example.sentence == "It's a beautiful day!"
        assert example.sentiment == "Positive"


class TestSetupOllamaNode:
    """Test SetupOllamaNode functionality using gpt-oss:20b."""
    
    @patch('requests.get')
    @patch('dspy.LM')
    def test_setup_ollama(self, mock_lm, mock_get):
        mock_get.return_value = Mock(status_code=200)
        node = SetupOllamaNode("setup", model="gpt-oss:20b")
        context = ExecutionContext()
        result = node.run(context)
        assert result.status == NodeStatus.SUCCESS
        assert result.outputs.get("provider") == "ollama"
        mock_lm.assert_called_once()


class TestSentimentAnalysisNode:
    """Test SentimentAnalysisNode functionality."""
    
    @patch('dspy.configure')
    @patch('dspy.Predict')
    def test_sentiment_analysis_success(self, mock_predict_class, mock_configure):
        """Test successful sentiment analysis."""
        # Mock DSPy predictor
        mock_predictor = Mock()
        mock_predict_class.return_value = mock_predictor
        
        # Mock prediction results
        mock_predictor.side_effect = [
            Mock(sentiment="Positive"),
            Mock(sentiment="Negative"),
            Mock(sentiment="Neutral")
        ]
        
        node = SentimentAnalysisNode("sentiment")
        context = ExecutionContext()
        
        # Mock DSPy LM and dataset
        context.set_artifact("dspy_lm", Mock())
        context.set_artifact("dataset", [
            Mock(sentence="Good"),
            Mock(sentence="Bad"),
            Mock(sentence="Okay")
        ])
        
        result = node.run(context)
        
        assert result.status == NodeStatus.SUCCESS
        assert "predictions" in result.outputs
        assert len(result.outputs["predictions"]) == 3
        
        # Check predictions
        predictions = result.outputs["predictions"]
        assert predictions[0]["prediction"] == "Positive"
        assert predictions[1]["prediction"] == "Negative"
        assert predictions[2]["prediction"] == "Neutral"
    
    def test_sentiment_analysis_missing_dataset(self):
        """Test sentiment analysis with missing dataset."""
        node = SentimentAnalysisNode("sentiment")
        context = ExecutionContext()
        context.set_artifact("dspy_lm", Mock())
        
        result = node.run(context)
        assert result.status == NodeStatus.FAILED
        assert "dataset" in result.error
    
    def test_sentiment_analysis_missing_dspy_lm(self):
        """Test sentiment analysis with missing DSPy LM."""
        node = SentimentAnalysisNode("sentiment")
        context = ExecutionContext()
        context.set_artifact("dataset", [])
        
        result = node.run(context)
        assert result.status == NodeStatus.FAILED
        assert "dspy_lm" in result.error


class TestEvaluateModelNode:
    """Test EvaluateModelNode functionality."""
    
    def test_evaluate_model_success(self):
        """Test successful model evaluation."""
        node = EvaluateModelNode("evaluate")
        context = ExecutionContext()
        
        # Mock dataset with ground truth
        dataset = [
            Mock(sentence="Good", sentiment="Positive"),
            Mock(sentence="Bad", sentiment="Negative"),
            Mock(sentence="Okay", sentiment="Neutral")
        ]
        
        # Mock predictions
        predictions = [
            {"sentence": "Good", "prediction": "Positive", "actual": "Positive"},
            {"sentence": "Bad", "prediction": "Negative", "actual": "Negative"},
            {"sentence": "Okay", "prediction": "Positive", "actual": "Neutral"}  # Wrong prediction
        ]
        
        context.set_artifact("dataset", dataset)
        context.set_artifact("predictions", predictions)
        
        result = node.run(context)
        
        assert result.status == NodeStatus.SUCCESS
        assert "evaluation_results" in result.outputs
        
        eval_results = result.outputs["evaluation_results"]
        assert eval_results["accuracy"] == 2/3  # 2 correct out of 3
        assert eval_results["total_examples"] == 3
        assert eval_results["correct_predictions"] == 2
    
    def test_evaluate_model_perfect_score(self):
        """Test model evaluation with perfect predictions."""
        node = EvaluateModelNode("evaluate")
        context = ExecutionContext()
        
        dataset = [Mock(sentiment="Positive")]
        predictions = [{"prediction": "Positive", "actual": "Positive"}]
        
        context.set_artifact("dataset", dataset)
        context.set_artifact("predictions", predictions)
        
        result = node.run(context)
        
        assert result.status == NodeStatus.SUCCESS
        eval_results = result.outputs["evaluation_results"]
        assert eval_results["accuracy"] == 1.0
    
    def test_evaluate_model_missing_inputs(self):
        """Test evaluation with missing required inputs."""
        node = EvaluateModelNode("evaluate")
        context = ExecutionContext()
        
        result = node.run(context)
        assert result.status == NodeStatus.FAILED


class TestReportGeneratorNode:
    """Test ReportGeneratorNode functionality."""
    
    def test_generate_report_success(self):
        """Test successful report generation."""
        node = ReportGeneratorNode("report", format="text")
        context = ExecutionContext()
        
        # Mock evaluation results
        eval_results = {
            "accuracy": 0.85,
            "total_examples": 100,
            "correct_predictions": 85,
            "model_name": "gpt-4o-mini"
        }
        
        context.set_artifact("evaluation_results", eval_results)
        context.set_artifact("dataset_info", {"rows": 100})
        
        result = node.run(context)
        
        assert result.status == NodeStatus.SUCCESS
        assert "report" in result.outputs
        
        report = result.outputs["report"]
        assert "85.0%" in report  # Accuracy percentage
        assert "gpt-4o-mini" in report  # Model name
        assert "100" in report  # Total examples
    
    def test_generate_json_report(self):
        """Test JSON report generation."""
        node = ReportGeneratorNode("report", format="json")
        context = ExecutionContext()
        
        eval_results = {"accuracy": 0.9, "total_examples": 50}
        context.set_artifact("evaluation_results", eval_results)
        
        result = node.run(context)
        
        assert result.status == NodeStatus.SUCCESS
        report = result.outputs["report"]
        
        # Should be valid JSON string
        import json
        parsed = json.loads(report)
        assert parsed["evaluation_results"]["accuracy"] == 0.9
    
    def test_report_with_file_output(self):
        """Test report generation with file output."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = os.path.join(temp_dir, "test_report.txt")
            node = ReportGeneratorNode("report", format="text", output_file=output_file)
            context = ExecutionContext()
            
            eval_results = {"accuracy": 0.8}
            context.set_artifact("evaluation_results", eval_results)
            
            result = node.run(context)
            
            assert result.status == NodeStatus.SUCCESS
            assert os.path.exists(output_file)
            
            # Check file contents
            with open(output_file, 'r') as f:
                content = f.read()
                assert "80.0%" in content
    
    def test_generate_report_missing_results(self):
        """Test report generation with missing evaluation results."""
        node = ReportGeneratorNode("report")
        context = ExecutionContext()
        
        result = node.run(context)
        assert result.status == NodeStatus.FAILED
        assert "evaluation_results" in result.error


class TestNodeInputValidation:
    """Test input validation across different nodes."""
    
    def test_nodes_validate_required_inputs(self):
        """Test that nodes properly validate required inputs."""
        # Nodes that require specific inputs
        test_cases = [
            (SentimentAnalysisNode("test"), ["dspy_lm", "dataset"]),
            (EvaluateModelNode("test"), ["dataset", "predictions"]),
            (ReportGeneratorNode("test"), ["evaluation_results"])
        ]
        
        for node, required_inputs in test_cases:
            context = ExecutionContext()
            
            # Should fail with missing inputs
            result = node.run(context)
            assert result.status == NodeStatus.FAILED
            
            # Add inputs one by one (except last)
            for input_name in required_inputs[:-1]:
                context.set_artifact(input_name, Mock())
                result = node.run(context)
                assert result.status == NodeStatus.FAILED
            
            # Add final input - now should work (or at least not fail on missing inputs)
            context.set_artifact(required_inputs[-1], Mock())
            # Note: May still fail due to mock data, but not due to missing inputs


if __name__ == "__main__":
    pytest.main([__file__])
