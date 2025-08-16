#!/usr/bin/env python3
"""
DAG-based Execution Engine for Evaluation Systems

This module provides a flexible, extensible DAG execution engine that can
run evaluation pipelines with proper dependency management, cycle detection,
and failure handling.
"""

import yaml
import json
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Set, Optional, Tuple, Union
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from dataclasses import dataclass, field
from enum import Enum
import traceback
import os
import hashlib
import pickle
import logging
import inspect
import time
import uuid

from config import get_config, SystemConfig, setup_logging
from utils.logging_setup import get_logger, create_execution_tracker
from utils.metrics import dag_metrics, timer


class NodeStatus(Enum):
    """Execution status of a node."""
    PENDING = "pending"
    RUNNING = "running" 
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    CACHED = "cached"


@dataclass
class NodeResult:
    """Result of a node execution."""
    status: NodeStatus
    outputs: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    execution_time: Optional[float] = None


class ExecutionContext:
    """Context passed between nodes during execution."""
    
    def __init__(self):
        self.artifacts: Dict[str, Any] = {}
        self.node_results: Dict[str, NodeResult] = {}
        self.metadata: Dict[str, Any] = {}
    
    def get_artifact(self, name: str) -> Any:
        """Get an artifact by name."""
        if name not in self.artifacts:
            raise KeyError(f"Artifact '{name}' not found in context")
        return self.artifacts[name]
    
    def set_artifact(self, name: str, value: Any) -> None:
        """Set an artifact in the context."""
        self.artifacts[name] = value
    
    def has_artifact(self, name: str) -> bool:
        """Check if an artifact exists."""
        return name in self.artifacts


class BaseNode(ABC):
    """Base class for all DAG nodes."""
    
    def __init__(self, node_id: str, inputs: List[str] = None, outputs: List[str] = None, **kwargs):
        self.id = node_id
        self.inputs = inputs or []
        self.outputs = outputs or []
        self.config = kwargs
        self.status = NodeStatus.PENDING
    
    @abstractmethod
    def run(self, context: ExecutionContext) -> NodeResult:
        """Execute the node and return results."""
        pass
    
    def validate_inputs(self, context: ExecutionContext) -> List[str]:
        """Validate that all required inputs are available."""
        missing = []
        for input_name in self.inputs:
            if not context.has_artifact(input_name):
                missing.append(input_name)
        return missing
    
    def __repr__(self):
        return f"{self.__class__.__name__}(id='{self.id}', inputs={self.inputs}, outputs={self.outputs})"


class NodeRegistry:
    """Registry for node types to enable extensibility."""
    
    _registry: Dict[str, type] = {}
    
    @classmethod
    def register(cls, node_type: str, node_class: type):
        """Register a node class with a type name."""
        if not issubclass(node_class, BaseNode):
            raise ValueError(f"Node class must inherit from BaseNode")
        cls._registry[node_type] = node_class
    
    @classmethod
    def create_node(cls, node_type: str, **kwargs) -> BaseNode:
        """Create a node instance from its type."""
        if node_type not in cls._registry:
            raise ValueError(f"Unknown node type: {node_type}")
        return cls._registry[node_type](**kwargs)
    
    @classmethod
    def get_registered_types(cls) -> List[str]:
        """Get all registered node types."""
        return list(cls._registry.keys())


class DAGValidationError(Exception):
    """Raised when DAG validation fails."""
    pass


class DAGExecutor:
    """DAG-based execution engine."""
    
    def __init__(self, config: Optional[SystemConfig] = None, cache_dir: Optional[str] = None):
        self.config = config or get_config()
        # Ensure logging is configured once per process based on config
        try:
            setup_logging(self.config.logging)
        except Exception:
            pass
        self.nodes: Dict[str, BaseNode] = {}
        self.dependencies: Dict[str, Set[str]] = defaultdict(set)
        self.dependents: Dict[str, Set[str]] = defaultdict(set)
        self.execution_order: List[str] = []
        
        # Cache directory behavior:
        # - If cache_dir is provided (even None), honor it. Passing None disables caching.
        # - If omitted (not detectable), defaults to disabled when None.
        self.cache_dir = cache_dir
            
        self.artifact_hashes: Dict[str, str] = {}
        self.logger = get_logger(__name__)
        self.execution_id = str(uuid.uuid4())
        self.execution_tracker = None

        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            # Prune cache according to policy
            try:
                self._prune_cache()
            except Exception:
                pass

    def add_node(self, node: BaseNode) -> None:
        """Add a node to the DAG."""
        if node.id in self.nodes:
            raise ValueError(f"Node with id '{node.id}' already exists")
        
        self.nodes[node.id] = node
    
    def _build_dependency_graph(self) -> None:
        """Build the complete dependency graph after all nodes are added."""
        self.dependencies.clear()
        self.dependents.clear()
        
        for node_id, node in self.nodes.items():
            self.dependencies[node_id] = set()  # Initialize empty set
            
        for node_id, node in self.nodes.items():
            # Build dependency graph
            for input_artifact in node.inputs:
                # Find which nodes produce this artifact
                producers = self._find_artifact_producers(input_artifact)
                for producer_id in producers:
                    self.dependencies[node_id].add(producer_id)
                    self.dependents[producer_id].add(node_id)
    
    def _find_artifact_producers(self, artifact: str) -> List[str]:
        """Find which nodes produce a given artifact."""
        producers = []
        for node_id, node in self.nodes.items():
            if artifact in node.outputs:
                producers.append(node_id)
        return producers
    
    def _interpolate_config(self, obj: Any, global_config: dict) -> Any:
        """Recursively interpolate config values in an object."""
        import re
        
        if isinstance(obj, str):
            # Check if the entire string is just a config reference
            full_match = re.match(r'^\{(config\.[^}]+)\}$', obj)
            if full_match:
                path = full_match.group(1)
                if path.startswith("config."):
                    path = path[7:]  # Remove "config." prefix
                
                # Navigate the nested config structure
                value = global_config
                for key in path.split("."):
                    if isinstance(value, dict) and key in value:
                        value = value[key]
                    else:
                        # If path doesn't exist, return original string
                        return obj
                
                # Return the actual value (preserving type)
                return value
            
            # For partial interpolations, use string replacement
            def replace_config(match):
                path = match.group(1)  # Extract the path after {config.
                if path.startswith("config."):
                    path = path[7:]  # Remove "config." prefix
                
                # Navigate the nested config structure
                value = global_config
                for key in path.split("."):
                    if isinstance(value, dict) and key in value:
                        value = value[key]
                    else:
                        # If path doesn't exist, return original string
                        return match.group(0)
                
                return str(value)
            
            # Replace all {config.path.to.value} patterns
            result = re.sub(r'\{(config\.[^}]+)\}', replace_config, obj)
            
            # Try to convert to appropriate type if it's a pure interpolated value
            if result != obj and result.replace('.', '').replace('-', '').isdigit():
                try:
                    if '.' in result:
                        return float(result)
                    else:
                        return int(result)
                except ValueError:
                    pass
            
            return result
            
        elif isinstance(obj, dict):
            return {k: self._interpolate_config(v, global_config) for k, v in obj.items()}
            
        elif isinstance(obj, list):
            return [self._interpolate_config(item, global_config) for item in obj]
            
        else:
            return obj

    def load_from_spec(self, spec: Union[str, dict]) -> None:
        """Load DAG from a pipeline specification."""
        if isinstance(spec, str):
            # Try to parse as YAML first, then JSON
            try:
                config = yaml.safe_load(spec)
            except yaml.YAMLError:
                try:
                    config = json.loads(spec)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid YAML/JSON spec: {e}")
        else:
            config = spec
        
        if "steps" not in config:
            raise ValueError("Pipeline spec must contain 'steps'")
        
        # Clear existing nodes
        self.nodes.clear()
        self.dependencies.clear()
        self.dependents.clear()
        
        # Extract global config for interpolation
        global_config = config.get("config", {})
        
        # Create nodes from spec
        for step in config["steps"]:
            if "id" not in step or "type" not in step:
                raise ValueError("Each step must have 'id' and 'type'")
            
            node_id = step["id"]
            node_type = step["type"]
            inputs = step.get("inputs", [])
            outputs = step.get("outputs", [])
            
            # Extract other config parameters
            node_config = {k: v for k, v in step.items() 
                          if k not in ["id", "type", "inputs", "outputs"]}
            
            # Interpolate config values
            node_config = self._interpolate_config(node_config, global_config)
            
            try:
                node = NodeRegistry.create_node(
                    node_type,
                    node_id=node_id,
                    inputs=inputs,
                    outputs=outputs,
                    **node_config
                )
                self.add_node(node)
            except Exception as e:
                raise ValueError(f"Failed to create node '{node_id}': {e}")
        
        # Build dependency graph after all nodes are added
        self._build_dependency_graph()
    
    def validate(self, strict: bool = True) -> None:
        """Validate the DAG structure.
        
        If strict is False, missing input producers are treated as warnings
        and do not raise, allowing execution where some nodes may fail at
        runtime due to missing artifacts.
        """
        # Build dependency graph first
        self._build_dependency_graph()
        
        errors = []
        
        # Check that all input dependencies can be satisfied
        for node_id, node in self.nodes.items():
            for input_artifact in node.inputs:
                producers = self._find_artifact_producers(input_artifact)
                if not producers:
                    if strict:
                        errors.append(f"Node '{node_id}' requires input '{input_artifact}' but no node produces it")
                elif len(producers) > 1:
                    errors.append(f"Multiple nodes produce artifact '{input_artifact}': {producers}")
        
        # Check for cycles
        if self._has_cycles():
            errors.append("DAG contains cycles")
        
        if errors:
            raise DAGValidationError("DAG validation failed:\n" + "\n".join(f"- {error}" for error in errors))
        
        # Build execution order
        self.execution_order = self._topological_sort()
    
    def _has_cycles(self) -> bool:
        """Check if the DAG has cycles using DFS."""
        WHITE, GRAY, BLACK = 0, 1, 2
        colors = {node_id: WHITE for node_id in self.nodes}
        
        def dfs(node_id: str) -> bool:
            if colors[node_id] == GRAY:
                return True  # Back edge found - cycle detected
            if colors[node_id] == BLACK:
                return False  # Already processed
            
            colors[node_id] = GRAY
            for dependent_id in self.dependents[node_id]:
                if dfs(dependent_id):
                    return True
            colors[node_id] = BLACK
            return False
        
        for node_id in self.nodes:
            if colors[node_id] == WHITE:
                if dfs(node_id):
                    return True
        return False
    
    def _topological_sort(self) -> List[str]:
        """Return nodes in topological order using Kahn's algorithm."""
        in_degree = {node_id: len(self.dependencies[node_id]) for node_id in self.nodes}
        queue = deque([node_id for node_id, degree in in_degree.items() if degree == 0])
        result = []
        
        while queue:
            current = queue.popleft()
            result.append(current)
            
            for dependent in self.dependents[current]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        if len(result) != len(self.nodes):
            raise DAGValidationError("Cannot compute topological sort - DAG has cycles")
        
        return result

    def _get_node_hash(self, node_id: str) -> str:
        """Generate a unique hash for a node's execution state."""
        node = self.nodes[node_id]
        hasher = hashlib.sha256()

        # Hash node configuration
        hasher.update(node.id.encode())
        hasher.update(json.dumps(node.config, sort_keys=True).encode())

        # Include node class identity and source signature to reduce stale cache risk
        try:
            hasher.update(node.__class__.__name__.encode())
            src = inspect.getsource(node.__class__.run)
            hasher.update(hashlib.sha256(src.encode()).hexdigest().encode())
        except Exception:
            pass

        # Hash input artifact hashes
        for input_name in sorted(node.inputs):
            if input_name in self.artifact_hashes:
                hasher.update(self.artifact_hashes[input_name].encode())

        return hasher.hexdigest()

    def _dependency_closure(self, targets: List[str]) -> Set[str]:
        """Return all nodes required to run the targets (including themselves)."""
        need: Set[str] = set(targets)
        changed = True
        while changed:
            changed = False
            for nid in list(need):
                for dep in self.dependencies.get(nid, set()):
                    if dep not in need:
                        need.add(dep)
                        changed = True
        return need

    def _downstream_closure(self, sources: List[str]) -> Set[str]:
        """Return all nodes reachable from sources (including themselves)."""
        seen: Set[str] = set(sources)
        queue = list(sources)
        while queue:
            cur = queue.pop(0)
            for dep in self.dependents.get(cur, set()):
                if dep not in seen:
                    seen.add(dep)
                    queue.append(dep)
        return seen

    def explain_plan(self) -> str:
        """Return a human-readable execution plan summary."""
        if not self.execution_order:
            self.validate()
        lines = []
        lines.append(f"Nodes: {len(self.nodes)}")
        lines.append(f"Order: {' -> '.join(self.execution_order)}")
        # Longest path (by edges) via DP
        dist = {nid: 0 for nid in self.nodes}
        for nid in self.execution_order:
            for dep in self.dependents.get(nid, set()):
                dist[dep] = max(dist.get(dep, 0), dist.get(nid, 0) + 1)
        if dist:
            lines.append(f"Critical path length (edges): {max(dist.values())}")
        lines.append("Node degrees:")
        for nid in self.execution_order:
            indeg = len(self.dependencies.get(nid, []))
            outdeg = len(self.dependents.get(nid, []))
            lines.append(f"  - {nid}: in={indeg}, out={outdeg}")
        return "\n".join(lines)

    def execute(self, context: ExecutionContext = None, use_cache: Optional[bool] = None, selected_nodes: Optional[Set[str]] = None) -> ExecutionContext:
        """Execute the DAG in dependency order."""
        if not self.execution_order:
            self.validate(strict=False)  # This will set execution_order
        
        if context is None:
            context = ExecutionContext()
        
        # Use config default if not specified
        if use_cache is None:
            use_cache = self.config.cache.enabled
        
        print(f"🚀 Executing DAG with {len(self.nodes)} nodes (Cache: {'Enabled' if use_cache else 'Disabled'})")
        active_order = [nid for nid in self.execution_order if (selected_nodes is None or nid in selected_nodes)]
        print(f"📋 Execution order: {' -> '.join(active_order)}")
        
        # Choose serial or parallel execution based on config
        if not self.config.evaluation.parallel_execution or self.config.evaluation.max_concurrent_nodes <= 1:
            for node_id in active_order:
                self._execute_node_internal(node_id, context, use_cache)
        else:
            lock = Lock()
            in_degree = {nid: len(self.dependencies[nid]) for nid in self.nodes}
            ready = {nid for nid, deg in in_degree.items() if deg == 0 and (selected_nodes is None or nid in selected_nodes)}
            submitted = set()
            futures = {}
            with ThreadPoolExecutor(max_workers=self.config.evaluation.max_concurrent_nodes) as pool:
                # Submit initial ready set
                for nid in list(ready):
                    futures[pool.submit(self._execute_node_internal, nid, context, use_cache)] = nid
                    submitted.add(nid)
                    ready.remove(nid)
                # Process as tasks complete
                while futures:
                    for future in as_completed(list(futures.keys()), timeout=None):
                        completed_nid = futures.pop(future)
                        # Update dependents
                        for dep in self.dependents[completed_nid]:
                            if selected_nodes is not None and dep not in selected_nodes:
                                continue
                            in_degree[dep] -= 1
                            # If any dependency failed, mark skip immediately
                            deps_failed = any(
                                context.node_results.get(d, NodeResult(NodeStatus.PENDING)).status == NodeStatus.FAILED
                                for d in self.dependencies[dep]
                            )
                            if deps_failed and dep not in context.node_results:
                                res = NodeResult(status=NodeStatus.SKIPPED, error="Skipped due to failed dependencies")
                                context.node_results[dep] = res
                                self.nodes[dep].status = NodeStatus.SKIPPED
                                print(f"⏭️  Skipped {dep} (failed dependencies)")
                                # Reduce in-degree of its dependents to allow propagation
                                continue
                            if in_degree[dep] == 0 and dep not in submitted and dep not in context.node_results:
                                futures[pool.submit(self._execute_node_internal, dep, context, use_cache)] = dep
                                submitted.add(dep)
                    # loop repeats until all futures consumed
        
        # Print summary
        successful = sum(1 for r in context.node_results.values() if r.status == NodeStatus.SUCCESS)
        failed = sum(1 for r in context.node_results.values() if r.status == NodeStatus.FAILED)
        skipped = sum(1 for r in context.node_results.values() if r.status == NodeStatus.SKIPPED)
        cached = sum(1 for r in context.node_results.values() if r.status == NodeStatus.CACHED)

        print(f"\n🎯 Execution Summary:")
        print(f"   ✅ Success: {successful}")
        print(f"   💾 Cached: {cached}")
        print(f"   ❌ Failed: {failed}")
        print(f"   ⏭️  Skipped: {skipped}")
        print(f"   📊 Total: {len(self.nodes)}")
        
        return context

    def _execute_node_internal(self, node_id: str, context: ExecutionContext, use_cache: bool) -> None:
        node = self.nodes[node_id]
        # Check if any dependencies failed
        failed_dependencies = []
        for dep_id in self.dependencies[node_id]:
            if context.node_results.get(dep_id, NodeResult(NodeStatus.PENDING)).status == NodeStatus.FAILED:
                failed_dependencies.append(dep_id)
        
        if failed_dependencies:
            # Skip this node due to failed dependencies
            result = NodeResult(
                status=NodeStatus.SKIPPED,
                error=f"Skipped due to failed dependencies: {failed_dependencies}"
            )
            context.node_results[node_id] = result
            node.status = NodeStatus.SKIPPED
            print(f"⏭️  Skipped {node_id} (failed dependencies: {failed_dependencies})")
            return

        # Caching logic
        node_hash = self._get_node_hash(node_id)
        cache_path = os.path.join(self.cache_dir, f"{node_hash}.pkl") if self.cache_dir else None

        if use_cache and cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    result = pickle.load(f)
                
                result.status = NodeStatus.CACHED
                print(f"CACHE: Loading result for {node_id} from cache.")

                # Store outputs in context and update artifact hashes
                for output_name, output_value in result.outputs.items():
                    context.set_artifact(output_name, output_value)
                    output_hash = hashlib.sha256(pickle.dumps(output_value)).hexdigest()
                    self.artifact_hashes[output_name] = output_hash

                context.node_results[node_id] = result
                node.status = NodeStatus.CACHED
                print(f"✅ Completed {node_id} (from cache)")
                return

            except Exception as e:
                print(f"CACHE: Failed to load from cache for {node_id}: {e}. Re-running.")
        
        # Execute the node
        print(f"⚡ Running {node_id}...")
        node.status = NodeStatus.RUNNING
        
        try:
            start_time = time.time()
            result = node.run(context)
            
            end_time = time.time()
            result.execution_time = end_time - start_time
            
            # Store outputs in context and update artifact hashes
            for output_name, output_value in result.outputs.items():
                context.set_artifact(output_name, output_value)
                output_hash = hashlib.sha256(pickle.dumps(output_value)).hexdigest()
                self.artifact_hashes[output_name] = output_hash
            
            node.status = result.status
            context.node_results[node_id] = result
            
            
            if result.status == NodeStatus.SUCCESS:
                print(f"✅ Completed {node_id} ({result.execution_time:.2f}s)")
                # Save to cache if successful
                if use_cache and cache_path:
                    try:
                        with open(cache_path, "wb") as f:
                            pickle.dump(result, f)
                        print(f"CACHE: Saved result for {node_id} to cache.")
                    except Exception:
                        pass
            else:
                print(f"❌ Failed {node_id}: {result.error}")
                
        except Exception as e:
            error_msg = f"Exception in node {node_id}: {str(e)}\n{traceback.format_exc()}"
            result = NodeResult(
                status=NodeStatus.FAILED,
                error=error_msg,
                execution_time=time.time() - start_time if 'start_time' in locals() else None
            )
            node.status = NodeStatus.FAILED
            context.node_results[node_id] = result
            print(f"❌ Failed {node_id}: {str(e)}")

    def _prune_cache(self) -> None:
        """Prune cache entries based on TTL and max size from config."""
        if not self.cache_dir:
            return
        ttl_seconds = int(self.config.cache.cache_ttl_hours * 3600)
        max_bytes = int(self.config.cache.max_cache_size_mb * 1024 * 1024)

        try:
            entries = []
            total_size = 0
            now = time.time()
            for fname in os.listdir(self.cache_dir):
                path = os.path.join(self.cache_dir, fname)
                if not os.path.isfile(path):
                    continue
                try:
                    stat = os.stat(path)
                except FileNotFoundError:
                    continue
                age = now - stat.st_mtime
                size = stat.st_size
                # Remove expired
                if ttl_seconds > 0 and age > ttl_seconds:
                    try:
                        os.remove(path)
                    except Exception:
                        pass
                    continue
                entries.append((path, stat.st_mtime, size))
                total_size += size

            # Enforce max size by deleting oldest
            if total_size > max_bytes:
                entries.sort(key=lambda x: x[1])  # oldest first
                bytes_to_free = total_size - max_bytes
                freed = 0
                for path, _, size in entries:
                    try:
                        os.remove(path)
                        freed += size
                    except Exception:
                        pass
                    if freed >= bytes_to_free:
                        break
        except Exception:
            # Never crash pruning
            pass
    
    def visualize(self) -> str:
        """Generate Graphviz DOT representation of the DAG."""
        lines = ["digraph DAG {"]
        lines.append("    rankdir=TB;")
        lines.append("    node [shape=box, style=rounded];")
        
        # Add nodes
        for node_id, node in self.nodes.items():
            color = {
                NodeStatus.PENDING: "lightgray",
                NodeStatus.RUNNING: "yellow", 
                NodeStatus.SUCCESS: "lightgreen",
                NodeStatus.FAILED: "lightcoral",
                NodeStatus.SKIPPED: "lightblue",
                NodeStatus.CACHED: "mediumpurple1"
            }.get(node.status, "white")
            
            lines.append(f'    "{node_id}" [label="{node_id}\n({node.__class__.__name__})" fillcolor="{color}" style="filled,rounded"];')
        
        # Add edges
        for node_id in self.nodes:
            for dep_id in self.dependencies[node_id]:
                lines.append(f'    "{dep_id}" -> "{node_id}";')
        
        lines.append("}")
        return "\n".join(lines)
    
    def get_execution_stats(self, context: ExecutionContext = None) -> Dict[str, Any]:
        """Get execution statistics."""
        if context is None:
            return {"total_nodes": len(self.nodes), "message": "No execution context provided"}
        
        stats = {
            "total_nodes": len(self.nodes),
            "successful": 0,
            "failed": 0, 
            "skipped": 0,
            "cached": 0,
            "total_time": 0.0,
            "node_times": {}
        }
        
        for node_id, result in context.node_results.items():
            if result.status == NodeStatus.SUCCESS:
                stats["successful"] += 1
            elif result.status == NodeStatus.FAILED:
                stats["failed"] += 1
            elif result.status == NodeStatus.SKIPPED:
                stats["skipped"] += 1
            elif result.status == NodeStatus.CACHED:
                stats["cached"] += 1
            
            if result.execution_time:
                stats["total_time"] += result.execution_time
                stats["node_times"][node_id] = result.execution_time
        
        return stats



# Register decorator for easy node registration
def register_node(node_type: str):
    """Decorator to register a node class."""
    def decorator(cls):
        NodeRegistry.register(node_type, cls)
        return cls
    return decorator


# Import and register evaluator nodes
try:
    from judge_nodes import (
        EvaluatorNode, MultiCriticNode, SynthesisNode, ValidationNode,
        RatingScale, BinaryScale, CategoryScale
    )
    
    # Register evaluator node types
    NodeRegistry.register("EvaluatorNode", EvaluatorNode)
    NodeRegistry.register("MultiCriticNode", MultiCriticNode)
    NodeRegistry.register("SynthesisNode", SynthesisNode)
    NodeRegistry.register("ValidationNode", ValidationNode)
    
except ImportError as e:
    print(f"Warning: Could not import judge nodes: {e}")

# Import and register evaluation nodes  
try:
    from evaluation_nodes import (
        DSPySetupNode, SentimentAnalysisNode, EvaluateModelNode, 
        ReportGeneratorNode, MultiMetricEvaluateModelNode,
        StatisticalSignificanceTestNode,
        DatasetAblationGeneratorNode, CostAnalysisNode, SetupOllamaNode,
        PromptVariationGeneratorNode
    )
    
    # Register evaluation node types
    NodeRegistry.register("DSPySetupNode", DSPySetupNode)
    NodeRegistry.register("SentimentAnalysisNode", SentimentAnalysisNode)
    NodeRegistry.register("EvaluateModelNode", EvaluateModelNode)
    NodeRegistry.register("ReportGeneratorNode", ReportGeneratorNode)
    NodeRegistry.register("MultiMetricEvaluateModelNode", MultiMetricEvaluateModelNode)
    NodeRegistry.register("StatisticalSignificanceTestNode", StatisticalSignificanceTestNode)
    NodeRegistry.register("DatasetAblationGeneratorNode", DatasetAblationGeneratorNode)
    NodeRegistry.register("CostAnalysisNode", CostAnalysisNode)
    NodeRegistry.register("SetupOllamaNode", SetupOllamaNode)
    NodeRegistry.register("PromptVariationGeneratorNode", PromptVariationGeneratorNode)
    
except ImportError as e:
    print(f"Warning: Could not import evaluation nodes: {e}")

# Import and register adversarial nodes
try:
    from nodes.adversarial import RedTeamGeneratorNode, AttackEvaluatorNode

    NodeRegistry.register("RedTeamGenerator", RedTeamGeneratorNode)
    NodeRegistry.register("AttackEvaluator", AttackEvaluatorNode)

except ImportError as e:
    print(f"Warning: Could not import adversarial nodes: {e}")
