#!/usr/bin/env python3
"""
Tests for the DAG execution engine.
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch
from typing import Dict, Any

from dag_engine import (
    DAGExecutor, BaseNode, NodeResult, NodeStatus, ExecutionContext,
    DAGValidationError, NodeRegistry
)


class MockNode(BaseNode):
    """Mock node for testing."""
    
    def __init__(self, node_id: str, inputs=None, outputs=None, should_fail=False, **kwargs):
        super().__init__(node_id, inputs, outputs, **kwargs)
        self.should_fail = should_fail
        self.run_called = False
    
    def run(self, context: ExecutionContext) -> NodeResult:
        self.run_called = True
        
        if self.should_fail:
            return NodeResult(status=NodeStatus.FAILED, error="Mock failure")
        
        # Set outputs in context
        for output in self.outputs:
            context.set_artifact(output, f"output_from_{self.id}")
        
        return NodeResult(
            status=NodeStatus.SUCCESS,
            outputs={output: f"output_from_{self.id}" for output in self.outputs}
        )


class TestDAGExecutor:
    """Test cases for DAGExecutor."""
    
    def test_add_node(self):
        """Test adding nodes to DAG."""
        executor = DAGExecutor()
        node = MockNode("test_node", outputs=["output1"])
        
        executor.add_node(node)
        assert "test_node" in executor.nodes
        assert executor.nodes["test_node"] == node
    
    def test_add_duplicate_node_fails(self):
        """Test that adding duplicate node IDs fails."""
        executor = DAGExecutor()
        node1 = MockNode("test_node", outputs=["output1"])
        node2 = MockNode("test_node", outputs=["output2"])
        
        executor.add_node(node1)
        with pytest.raises(ValueError, match="already exists"):
            executor.add_node(node2)
    
    def test_simple_dependency_chain(self):
        """Test execution of a simple dependency chain."""
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            executor = DAGExecutor(cache_dir=temp_dir)  # Use temp dir for isolated caching
            
            # Create nodes: A -> B -> C
            node_a = MockNode("A", outputs=["data"])
            node_b = MockNode("B", inputs=["data"], outputs=["processed"])
            node_c = MockNode("C", inputs=["processed"], outputs=["final"])
            
            executor.add_node(node_a)
            executor.add_node(node_b)
            executor.add_node(node_c)
            
            context = executor.execute()
            
            # Verify all nodes ran successfully
            assert all(node.run_called for node in [node_a, node_b, node_c])
            assert context.node_results["A"].status == NodeStatus.SUCCESS
            assert context.node_results["B"].status == NodeStatus.SUCCESS
            assert context.node_results["C"].status == NodeStatus.SUCCESS
            
            # Verify execution order
            assert executor.execution_order == ["A", "B", "C"]
    
    def test_parallel_execution(self):
        """Test execution of parallel nodes."""
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            executor = DAGExecutor(cache_dir=temp_dir)
            
            # Create nodes: A -> [B, C] -> D
            node_a = MockNode("A", outputs=["data"])
            node_b = MockNode("B", inputs=["data"], outputs=["result1"])
            node_c = MockNode("C", inputs=["data"], outputs=["result2"])
            node_d = MockNode("D", inputs=["result1", "result2"], outputs=["final"])
            
            for node in [node_a, node_b, node_c, node_d]:
                executor.add_node(node)
            
            context = executor.execute()
            
            # All nodes should succeed
            assert all(result.status in [NodeStatus.SUCCESS, NodeStatus.CACHED] for result in context.node_results.values())
            
            # A should run first, D should run last, B and C can run in parallel
            assert executor.execution_order[0] == "A"
            assert executor.execution_order[-1] == "D"
            assert set(executor.execution_order[1:3]) == {"B", "C"}
    
    def test_failure_propagation(self):
        """Test that node failures cause dependent nodes to be skipped."""
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            executor = DAGExecutor(cache_dir=temp_dir)
            
            # Create chain where B fails
            node_a = MockNode("A", outputs=["data"])
            node_b = MockNode("B", inputs=["data"], outputs=["processed"], should_fail=True)
            node_c = MockNode("C", inputs=["processed"], outputs=["final"])
            
            for node in [node_a, node_b, node_c]:
                executor.add_node(node)
            
            context = executor.execute()
            
            # A should succeed, B should fail, C should be skipped
            assert context.node_results["A"].status in [NodeStatus.SUCCESS, NodeStatus.CACHED]
            assert context.node_results["B"].status == NodeStatus.FAILED  # Fresh node should fail
            assert context.node_results["C"].status == NodeStatus.SKIPPED
            
            # C should not have run
            assert not node_c.run_called
    
    def test_cycle_detection(self):
        """Test that cycles are detected during validation."""
        executor = DAGExecutor()
        
        # Create cycle: A -> B -> C -> A
        node_a = MockNode("A", inputs=["data_c"], outputs=["data_a"])
        node_b = MockNode("B", inputs=["data_a"], outputs=["data_b"])
        node_c = MockNode("C", inputs=["data_b"], outputs=["data_c"])
        
        for node in [node_a, node_b, node_c]:
            executor.add_node(node)
        
        with pytest.raises(DAGValidationError, match="cycles"):
            executor.validate()
    
    def test_missing_input_validation(self):
        """Test validation catches missing input dependencies."""
        executor = DAGExecutor()
        
        # Node requires input that no other node produces
        node = MockNode("test", inputs=["missing_input"], outputs=["output"])
        executor.add_node(node)
        
        with pytest.raises(DAGValidationError, match="no node produces"):
            executor.validate()
    
    def test_multiple_producers_validation(self):
        """Test validation catches multiple producers for same artifact."""
        executor = DAGExecutor()
        
        # Two nodes produce the same output
        node_a = MockNode("A", outputs=["data"])
        node_b = MockNode("B", outputs=["data"])
        node_c = MockNode("C", inputs=["data"])
        
        for node in [node_a, node_b, node_c]:
            executor.add_node(node)
        
        with pytest.raises(DAGValidationError, match="Multiple nodes produce"):
            executor.validate()


class TestExecutionContext:
    """Test cases for ExecutionContext."""
    
    def test_artifact_management(self):
        """Test artifact get/set operations."""
        context = ExecutionContext()
        
        # Test setting and getting artifacts
        context.set_artifact("test_data", {"key": "value"})
        assert context.has_artifact("test_data")
        assert context.get_artifact("test_data") == {"key": "value"}
        
        # Test missing artifact
        assert not context.has_artifact("missing")
        with pytest.raises(KeyError):
            context.get_artifact("missing")


class TestNodeRegistry:
    """Test cases for NodeRegistry."""
    
    def test_register_and_create_node(self):
        """Test node registration and creation."""
        # Register a test node type
        NodeRegistry.register("TestNode", MockNode)
        
        # Create instance
        node = NodeRegistry.create_node("TestNode", node_id="test", outputs=["output"])
        assert isinstance(node, MockNode)
        assert node.id == "test"
        assert node.outputs == ["output"]
    
    def test_unknown_node_type_fails(self):
        """Test that unknown node types raise error."""
        with pytest.raises(ValueError, match="Unknown node type"):
            NodeRegistry.create_node("UnknownType", node_id="test")
    
    def test_invalid_node_class_fails(self):
        """Test that non-BaseNode classes can't be registered."""
        class NotANode:
            pass
        
        with pytest.raises(ValueError, match="must inherit from BaseNode"):
            NodeRegistry.register("InvalidNode", NotANode)


class TestPipelineSpecLoading:
    """Test loading DAG from YAML/JSON specifications."""
    
    def test_load_yaml_spec(self):
        """Test loading DAG from YAML specification."""
        # Register our mock node
        NodeRegistry.register("MockNode", MockNode)
        
        yaml_spec = """
        steps:
          - id: node_a
            type: MockNode
            outputs: [data]
          - id: node_b
            type: MockNode
            inputs: [data]
            outputs: [result]
        """
        
        executor = DAGExecutor()
        executor.load_from_spec(yaml_spec)
        
        assert len(executor.nodes) == 2
        assert "node_a" in executor.nodes
        assert "node_b" in executor.nodes
        assert executor.nodes["node_a"].outputs == ["data"]
        assert executor.nodes["node_b"].inputs == ["data"]
    
    def test_load_dict_spec(self):
        """Test loading DAG from dictionary specification."""
        NodeRegistry.register("MockNode", MockNode)
        
        dict_spec = {
            "steps": [
                {"id": "node_a", "type": "MockNode", "outputs": ["data"]},
                {"id": "node_b", "type": "MockNode", "inputs": ["data"]}
            ]
        }
        
        executor = DAGExecutor()
        executor.load_from_spec(dict_spec)
        
        assert len(executor.nodes) == 2
    
    def test_invalid_spec_fails(self):
        """Test that invalid specifications raise errors."""
        executor = DAGExecutor()
        
        # Missing 'steps' key
        with pytest.raises(ValueError, match="must contain 'steps'"):
            executor.load_from_spec({"invalid": "spec"})
        
        # Missing required fields
        with pytest.raises(ValueError, match="must have 'id' and 'type'"):
            executor.load_from_spec({"steps": [{"id": "test"}]})


class TestCaching:
    """Test caching functionality."""
    
    def test_caching_disabled(self):
        """Test execution with caching disabled."""
        executor = DAGExecutor(cache_dir=None)
        node = MockNode("test", outputs=["output"])
        executor.add_node(node)
        
        # First execution
        context1 = executor.execute(use_cache=False)
        assert context1.node_results["test"].status == NodeStatus.SUCCESS
        
        # Second execution should run again (no caching)
        node.run_called = False  # Reset flag
        context2 = executor.execute(use_cache=False)
        assert node.run_called  # Should have run again
    
    def test_caching_enabled(self):
        """Test execution with caching enabled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            executor = DAGExecutor(cache_dir=temp_dir)
            node = MockNode("test", outputs=["output"])
            executor.add_node(node)
            
            # First execution
            context1 = executor.execute(use_cache=True)
            assert context1.node_results["test"].status == NodeStatus.SUCCESS
            
            # Second execution should use cache
            node.run_called = False  # Reset flag
            context2 = executor.execute(use_cache=True)
            assert not node.run_called  # Should not have run again
            assert context2.node_results["test"].status == NodeStatus.CACHED


if __name__ == "__main__":
    pytest.main([__file__])