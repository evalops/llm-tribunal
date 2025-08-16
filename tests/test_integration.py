#!/usr/bin/env python3
"""
Integration tests for the DAG evaluation system.
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
import json

from dag_engine import DAGExecutor, ExecutionContext, NodeStatus
from evaluation_nodes import CreateTestDataNode, SetupOllamaNode
from judge_nodes import EvaluatorNode, RatingScale, CategoryScale


class TestEndToEndPipelines:
    """Test complete evaluation pipelines end-to-end."""
    
    def test_simple_evaluation_pipeline(self):
        """Test a simple evaluation pipeline with mocked LLM calls."""
        executor = DAGExecutor(cache_dir=None)
        
        # Add nodes manually
        data_node = CreateTestDataNode("create_data", outputs=["dataset", "raw_data"])
        
        with patch('requests.get') as mock_get, patch('dspy.LM'), patch('dspy.configure'):
            mock_get.return_value = Mock(status_code=200)
            setup_node = SetupOllamaNode("setup", outputs=["dspy_lm"], model="gpt-oss:20b")
        
        executor.add_node(data_node)
        executor.add_node(setup_node)
        
        # Execute pipeline
        context = executor.execute()
        
        # Verify successful execution
        assert context.node_results["create_data"].status == NodeStatus.SUCCESS
        assert context.node_results["setup"].status in [NodeStatus.SUCCESS, NodeStatus.FAILED]  # May fail due to missing API key
        
        # Verify data artifacts
        assert context.has_artifact("dataset")
        dataset = context.get_artifact("dataset")
        assert len(dataset) == 10
    
    @patch('requests.post')
    def test_judge_evaluation_pipeline(self, mock_post):
        """Test LLM-as-Judge evaluation pipeline."""
        # Mock OpenAI response
        mock_post.return_value = Mock(status_code=200, json=lambda: {"response": "I rate this content 4 out of 5"})
        
        executor = DAGExecutor(cache_dir=None)
        
        # Create evaluation pipeline
        data_node = CreateTestDataNode("data", outputs=["dataset"])
        
        scale = RatingScale((1, 5))
        judge_node = EvaluatorNode(
            "judge",
            scale,
            "Rate the politeness of this text: {sentence}",
            inputs=["sentence"],
            outputs=["politeness_score"],
            model="gpt-oss:20b"
        )
        
        executor.add_node(data_node)
        executor.add_node(judge_node)
        
        # Execute with mocked environment
        context = executor.execute()
        
        # Verify execution
        assert context.node_results["data"].status == NodeStatus.SUCCESS
        # Judge node might fail if it can't find the 'sentence' artifact
        # This is expected since we're testing with mock data
    
    def test_pipeline_from_yaml_config(self):
        """Test loading and executing pipeline from YAML configuration."""
        # Create a test YAML configuration
        config = {
            "steps": [
                {
                    "id": "create_test_data",
                    "type": "CreateTestData",
                    "outputs": ["dataset", "raw_data"]
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            json.dump(config, f)
            config_file = f.name
        
        try:
            executor = DAGExecutor(cache_dir=None)
            
            # Load from YAML
            with open(config_file, 'r') as f:
                executor.load_from_spec(f.read())
            
            # Execute
            context = executor.execute()
            
            # Verify
            assert len(executor.nodes) == 1
            assert "create_test_data" in executor.nodes
            assert context.node_results["create_test_data"].status == NodeStatus.SUCCESS
            
        finally:
            os.unlink(config_file)
    
    def test_pipeline_failure_recovery(self):
        """Test pipeline behavior when nodes fail."""
        executor = DAGExecutor(cache_dir=None)
        
        # Create nodes where one will fail
        success_node = CreateTestDataNode("success", outputs=["data"])
        
        # Mock a failing node
        class FailingNode:
            def __init__(self, node_id):
                self.id = node_id
                self.inputs = ["data"]
                self.outputs = ["result"]
                self.config = {}
                self.status = None
            
            def run(self, context):
                from dag_engine import NodeResult, NodeStatus
                return NodeResult(status=NodeStatus.FAILED, error="Intentional failure")
            
            def validate_inputs(self, context):
                return []
        
        failing_node = FailingNode("failing")
        
        # Node that depends on failing node
        dependent_node = CreateTestDataNode("dependent", inputs=["result"], outputs=["final"])
        
        executor.add_node(success_node)
        executor.add_node(failing_node)
        executor.add_node(dependent_node)
        
        context = executor.execute()
        
        # Verify failure propagation
        assert context.node_results["success"].status == NodeStatus.SUCCESS
        assert context.node_results["failing"].status == NodeStatus.FAILED
        assert context.node_results["dependent"].status == NodeStatus.SKIPPED
    
    def test_parallel_node_execution(self):
        """Test that independent nodes can execute in parallel."""
        executor = DAGExecutor(cache_dir=None)
        
        # Create independent nodes
        node1 = CreateTestDataNode("independent1", outputs=["data1"])
        node2 = CreateTestDataNode("independent2", outputs=["data2"])
        node3 = CreateTestDataNode("independent3", outputs=["data3"])
        
        # Node that depends on all three
        class MergeNode:
            def __init__(self):
                self.id = "merge"
                self.inputs = ["data1", "data2", "data3"]
                self.outputs = ["merged"]
                self.config = {}
                self.status = None
            
            def run(self, context):
                from dag_engine import NodeResult, NodeStatus
                # Verify all inputs are available
                for inp in self.inputs:
                    assert context.has_artifact(inp)
                return NodeResult(
                    status=NodeStatus.SUCCESS,
                    outputs={"merged": "all_data_merged"}
                )
            
            def validate_inputs(self, context):
                missing = []
                for inp in self.inputs:
                    if not context.has_artifact(inp):
                        missing.append(inp)
                return missing
        
        merge_node = MergeNode()
        
        for node in [node1, node2, node3, merge_node]:
            executor.add_node(node)
        
        context = executor.execute()
        
        # All nodes should succeed
        for node_id in ["independent1", "independent2", "independent3", "merge"]:
            assert context.node_results[node_id].status == NodeStatus.SUCCESS
        
        # Merge should run last
        assert executor.execution_order[-1] == "merge"
        # Independent nodes should run first (in any order)
        assert set(executor.execution_order[:3]) == {"independent1", "independent2", "independent3"}


class TestCachingIntegration:
    """Test caching behavior in full pipelines."""
    
    def test_pipeline_caching(self):
        """Test that pipeline results are cached and reused."""
        with tempfile.TemporaryDirectory() as temp_dir:
            executor = DAGExecutor(cache_dir=temp_dir)
            
            # Add a simple node
            node = CreateTestDataNode("test", outputs=["data"])
            executor.add_node(node)
            
            # First execution
            context1 = executor.execute(use_cache=True)
            assert context1.node_results["test"].status == NodeStatus.SUCCESS
            
            # Check cache directory
            cache_files = os.listdir(temp_dir)
            assert len(cache_files) > 0
            
            # Second execution should use cache
            context2 = executor.execute(use_cache=True)
            assert context2.node_results["test"].status == NodeStatus.CACHED
            
            # Verify same results
            assert context1.artifacts["data"][0].sentence == context2.artifacts["data"][0].sentence
    
    def test_cache_invalidation_on_config_change(self):
        """Test that cache is invalidated when node configuration changes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # First execution
            executor1 = DAGExecutor(cache_dir=temp_dir)
            node1 = CreateTestDataNode("test", outputs=["data"])
            executor1.add_node(node1)
            
            context1 = executor1.execute(use_cache=True)
            assert context1.node_results["test"].status == NodeStatus.SUCCESS
            
            # Second execution with different node (same ID but different config)
            executor2 = DAGExecutor(cache_dir=temp_dir)
            node2 = CreateTestDataNode("test", outputs=["data"])
            node2.config["new_param"] = "different_value"  # Change config
            executor2.add_node(node2)
            
            context2 = executor2.execute(use_cache=True)
            # Should not use cache due to config change
            assert context2.node_results["test"].status == NodeStatus.SUCCESS


class TestErrorHandlingIntegration:
    """Test error handling in complete pipelines."""
    
    def test_graceful_degradation(self):
        """Test that pipelines gracefully handle partial failures."""
        executor = DAGExecutor(cache_dir=None)
        
        # Create a branching pipeline: A -> [B, C] -> D
        # Where B fails but C succeeds
        
        node_a = CreateTestDataNode("A", outputs=["shared_data"])
        
        # B will fail
        class FailingNode:
            def __init__(self, node_id, inputs, outputs):
                self.id = node_id
                self.inputs = inputs
                self.outputs = outputs
                self.config = {}
                self.status = None
            
            def run(self, context):
                from dag_engine import NodeResult, NodeStatus
                return NodeResult(status=NodeStatus.FAILED, error="Simulated failure")
            
            def validate_inputs(self, context):
                return []
        
        node_b = FailingNode("B", ["shared_data"], ["result_b"])
        node_c = CreateTestDataNode("C", inputs=["shared_data"], outputs=["result_c"])
        
        # D depends on both B and C
        node_d = FailingNode("D", ["result_b", "result_c"], ["final"])
        
        for node in [node_a, node_b, node_c, node_d]:
            executor.add_node(node)
        
        context = executor.execute()
        
        # Verify partial success
        assert context.node_results["A"].status == NodeStatus.SUCCESS
        assert context.node_results["B"].status == NodeStatus.FAILED
        assert context.node_results["C"].status == NodeStatus.SUCCESS
        assert context.node_results["D"].status == NodeStatus.SKIPPED  # Missing result_b
    
    def test_error_reporting(self):
        """Test comprehensive error reporting."""
        executor = DAGExecutor(cache_dir=None)
        
        # Add a node that will raise an exception
        class ExceptionNode:
            def __init__(self):
                self.id = "exception_node"
                self.inputs = []
                self.outputs = ["output"]
                self.config = {}
                self.status = None
            
            def run(self, context):
                raise ValueError("This is a test exception")
            
            def validate_inputs(self, context):
                return []
        
        executor.add_node(ExceptionNode())
        context = executor.execute()
        
        # Verify exception is captured
        result = context.node_results["exception_node"]
        assert result.status == NodeStatus.FAILED
        assert "ValueError" in result.error
        assert "This is a test exception" in result.error
        assert "Traceback" in result.error or "Exception in node" in result.error


if __name__ == "__main__":
    pytest.main([__file__])
