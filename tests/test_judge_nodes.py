#!/usr/bin/env python3
"""
Tests for LLM-as-Evaluator judge nodes.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from judge_nodes import (
    RatingScale, BinaryScale, CategoryScale,
    EvaluatorNode, MultiCriticNode, SynthesisNode, ValidationNode
)
from dag_engine import ExecutionContext, NodeStatus


class TestEvaluationScales:
    """Test evaluation scale implementations."""
    
    def test_rating_scale(self):
        """Test RatingScale functionality."""
        scale = RatingScale((1, 5))
        
        # Test validation
        assert scale.validate(3)
        assert scale.validate(1)
        assert scale.validate(5)
        assert not scale.validate(0)
        assert not scale.validate(6)
        assert not scale.validate("3")
        
        # Test parsing
        assert scale.parse("I rate this 4 out of 5") == 4
        assert scale.parse("Score: 2") == 2
        
        with pytest.raises(ValueError):
            scale.parse("No numbers here")
        
        with pytest.raises(ValueError):
            scale.parse("I rate this 0 out of 5")  # Out of range
        
        # Test prompt formatting
        prompt = scale.format_prompt()
        assert "1" in prompt and "5" in prompt
    
    def test_rating_scale_with_labels(self):
        """Test RatingScale with labels."""
        labels = {1: "Poor", 3: "Average", 5: "Excellent"}
        scale = RatingScale((1, 5), labels)
        
        prompt = scale.format_prompt()
        assert "Poor" in prompt
        assert "Average" in prompt
        assert "Excellent" in prompt
    
    def test_binary_scale(self):
        """Test BinaryScale functionality."""
        scale = BinaryScale()
        
        # Test validation
        assert scale.validate(True)
        assert scale.validate(False)
        assert not scale.validate("yes")
        assert not scale.validate(1)
        
        # Test parsing
        assert scale.parse("Yes, this is correct") == True
        assert scale.parse("No way") == False
        assert scale.parse("The answer is true") == True
        assert scale.parse("This is false") == False
        
        with pytest.raises(ValueError):
            scale.parse("Maybe")
        
        # Test prompt formatting
        prompt = scale.format_prompt()
        assert "yes" in prompt.lower() or "no" in prompt.lower()
    
    def test_binary_scale_custom_values(self):
        """Test BinaryScale with custom values."""
        scale = BinaryScale(["pass", "good"], ["fail", "bad"])
        
        assert scale.parse("This will pass") == True
        assert scale.parse("This is bad") == False
        
        prompt = scale.format_prompt()
        assert "pass" in prompt.lower()
    
    def test_category_scale(self):
        """Test CategoryScale functionality."""
        categories = ["SAFE", "RISKY", "DANGEROUS"]
        scale = CategoryScale(categories)
        
        # Test validation
        assert scale.validate("SAFE")
        assert scale.validate("RISKY")
        assert not scale.validate("UNKNOWN")
        assert not scale.validate("safe")  # Case sensitive
        
        # Test parsing
        assert scale.parse("This content is SAFE") == "SAFE"
        assert scale.parse("I think this is RISKY behavior") == "RISKY"
        
        with pytest.raises(ValueError):
            scale.parse("This is UNKNOWN")
        
        # Test prompt formatting
        prompt = scale.format_prompt()
        for category in categories:
            assert category in prompt


class TestEvaluatorNode:
    """Test EvaluatorNode functionality with Ollama (gpt-oss:20b)."""
    
    @patch('requests.post')
    def test_evaluator_node_ollama(self, mock_post):
        mock_post.return_value = Mock(status_code=200, json=lambda: {"response": "I rate this content 4 out of 5"})
        scale = RatingScale((1, 5))
        evaluator = EvaluatorNode("test_eval", scale, "Rate this content: {content}", model="gpt-oss:20b")
        context = ExecutionContext()
        context.set_artifact("content", "This is test content")
        result = evaluator.run(context)
        assert result.status == NodeStatus.SUCCESS
        assert result.outputs["score"] == 4
        assert "metadata" in result.outputs
    
    def test_evaluator_node_missing_input(self):
        """Test EvaluatorNode with missing required input."""
        scale = RatingScale((1, 5))
        evaluator = EvaluatorNode("test", scale, "Rate: {content}")
        
        # Context missing required artifact
        context = ExecutionContext()
        
        result = evaluator.run(context)
        assert result.status == NodeStatus.FAILED
        assert "not found" in result.error
    
    @patch('openai.OpenAI')
    def test_evaluator_node_api_error(self, mock_openai_class):
        """Test EvaluatorNode handling API errors."""
        # Mock API error
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        
        scale = RatingScale((1, 5))
        evaluator = EvaluatorNode("test", scale, "Rate: {content}")
        
        context = ExecutionContext()
        context.set_artifact("content", "test")
        
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            result = evaluator.run(context)
        
        assert result.status == NodeStatus.FAILED
        assert "API Error" in result.error


class TestMultiCriticNode:
    """Test MultiCriticNode functionality."""
    
    @patch('judge_nodes.EvaluatorNode')
    def test_multi_critic_consensus(self, mock_evaluator_class):
        """Test MultiCriticNode reaching consensus."""
        # Mock evaluator responses
        mock_evaluators = []
        for i in range(3):
            mock_eval = Mock()
            mock_eval.run.return_value = Mock(
                status=NodeStatus.SUCCESS,
                outputs={"score": "SAFE", "explanation": "This content is safe"}
            )
            mock_evaluators.append(mock_eval)
        
        mock_evaluator_class.side_effect = mock_evaluators
        
        # Create multi-critic node
        multi_critic = MultiCriticNode(
            "safety_critics",
            ["gpt-4o-mini", "claude-3-haiku", "gpt-4o-mini"],
            "Evaluate safety: {content}",
            rounds=1
        )
        
        context = ExecutionContext()
        context.set_artifact("content", "test content")
        
        result = multi_critic.run(context)
        
        assert result.status == NodeStatus.SUCCESS
        assert "debate_history" in result.outputs
        # All evaluators should have been called
        for mock_eval in mock_evaluators:
            mock_eval.run.assert_called()
    
    @patch('judge_nodes.EvaluatorNode')
    def test_multi_critic_disagreement(self, mock_evaluator_class):
        """Test MultiCriticNode with disagreeing critics."""
        # Mock evaluator responses with disagreement
        responses = ["SAFE", "RISKY", "SAFE"]
        mock_evaluators = []
        for response in responses:
            mock_eval = Mock()
            mock_eval.run.return_value = Mock(
                status=NodeStatus.SUCCESS,
                outputs={"score": response, "explanation": f"I think this is {response}"}
            )
            mock_evaluators.append(mock_eval)
        
        mock_evaluator_class.side_effect = mock_evaluators
        
        multi_critic = MultiCriticNode(
            "critics",
            ["model1", "model2", "model3"],
            "Evaluate: {content}",
            rounds=1
        )
        
        context = ExecutionContext()
        context.set_artifact("content", "test")
        
        result = multi_critic.run(context)
        
        assert result.status == NodeStatus.SUCCESS
        deliberation = result.outputs["debate_history"]
        
        # Should capture disagreement
        deliberation_str = str(deliberation)
        assert "SAFE" in deliberation_str
        assert "RISKY" in deliberation_str


class TestSynthesisNode:
    """Test SynthesisNode functionality."""
    
    def test_majority_vote_synthesis(self):
        """Test majority vote synthesis."""
        synthesis = SynthesisNode("synthesis", method="majority_vote")
        
        # Mock deliberation data
        deliberation = {
            "round_0": {
                "critic_0": "SAFE",
                "critic_1": "SAFE", 
                "critic_2": "RISKY"
            }
        }
        
        context = ExecutionContext()
        # Prepare context as if a debate node ran
        from dag_engine import NodeResult
        context.node_results["debate"] = NodeResult(status=NodeStatus.SUCCESS, outputs={
            "final_positions": {"c0": {"score": "SAFE", "reasoning": ""}, "c1": {"score": "SAFE", "reasoning": ""}, "c2": {"score": "RISKY", "reasoning": ""}}
        })
        result = synthesis.run(context)
        assert result.status == NodeStatus.SUCCESS
        assert result.outputs["final_score"] == "SAFE"
        assert result.outputs["confidence"] > 0.5
    
    def test_consensus_synthesis(self):
        """Test consensus synthesis."""
        synthesis = SynthesisNode("synthesis", method="consensus")
        
        # All critics agree
        deliberation = {
            "round_0": {
                "critic_0": "SAFE",
                "critic_1": "SAFE",
                "critic_2": "SAFE"
            }
        }
        
        context = ExecutionContext()
        from dag_engine import NodeResult
        context.node_results["debate"] = NodeResult(status=NodeStatus.SUCCESS, outputs={
            "final_positions": {"c0": {"score": "SAFE", "reasoning": ""}, "c1": {"score": "SAFE", "reasoning": ""}, "c2": {"score": "SAFE", "reasoning": ""}}
        })
        result = synthesis.run(context)
        assert result.status == NodeStatus.SUCCESS
        assert result.outputs["final_score"] == "SAFE"
        assert result.outputs["confidence"] == 1.0
    
    def test_synthesis_missing_input(self):
        """Test synthesis with missing deliberation data."""
        synthesis = SynthesisNode("synthesis")
        context = ExecutionContext()
        
        result = synthesis.run(context)
        assert result.status == NodeStatus.FAILED


class TestValidationNode:
    """Test ValidationNode functionality."""
    
    @patch('requests.post')
    def test_validation_node(self, mock_post):
        """Test ValidationNode validation."""
        # Mock validation response
        mock_post.return_value = Mock(status_code=200, json=lambda: {"response": "Yes, the reasoning is sound"})
        
        validator = ValidationNode(
            "validator",
            "Validate this reasoning: {reasoning}",
            verifier_model="gpt-oss:20b"
        )
        
        context = ExecutionContext()
        context.set_artifact("reasoning", "Test reasoning")
        
        result = validator.run(context)
        
        assert result.status == NodeStatus.SUCCESS
        assert result.outputs["validation"]
        assert "explanation" in result.outputs


if __name__ == "__main__":
    pytest.main([__file__])
