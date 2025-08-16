#!/usr/bin/env python3
"""
Metrics collection and monitoring utilities.
"""

import time
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
import json

from utils.logging_setup import get_logger


@dataclass
class MetricPoint:
    """A single metric measurement."""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags
        }


class MetricsCollector:
    """Collects and manages metrics."""
    
    def __init__(self, max_points: int = 10000):
        self.max_points = max_points
        self.metrics: deque = deque(maxlen=max_points)
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = {}
        self.timers: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()
        self.logger = get_logger(__name__)
    
    def counter(self, name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a counter metric."""
        with self._lock:
            self.counters[name] += value
            self._record_metric(name, value, tags or {})
    
    def gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a gauge metric."""
        with self._lock:
            self.gauges[name] = value
            self._record_metric(name, value, tags or {})
    
    def timer(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a timer metric."""
        with self._lock:
            self.timers[name].append(value)
            self._record_metric(name, value, tags or {})
    
    def histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a histogram metric (alias for timer)."""
        self.timer(name, value, tags)
    
    def _record_metric(self, name: str, value: float, tags: Dict[str, str]) -> None:
        """Internal method to record a metric point."""
        point = MetricPoint(name, value, datetime.now(), tags)
        self.metrics.append(point)
    
    def get_counter(self, name: str) -> float:
        """Get current counter value."""
        with self._lock:
            return self.counters.get(name, 0.0)
    
    def get_gauge(self, name: str) -> Optional[float]:
        """Get current gauge value."""
        with self._lock:
            return self.gauges.get(name)
    
    def get_timer_stats(self, name: str) -> Dict[str, float]:
        """Get timer statistics."""
        with self._lock:
            values = self.timers.get(name, [])
            if not values:
                return {}
            
            sorted_values = sorted(values)
            count = len(sorted_values)
            
            return {
                "count": count,
                "min": min(sorted_values),
                "max": max(sorted_values),
                "mean": sum(sorted_values) / count,
                "p50": sorted_values[int(count * 0.5)],
                "p95": sorted_values[int(count * 0.95)],
                "p99": sorted_values[int(count * 0.99)]
            }
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all current metrics."""
        with self._lock:
            return {
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "timers": {name: self.get_timer_stats(name) for name in self.timers}
            }
    
    def get_recent_points(self, since: Optional[datetime] = None, limit: Optional[int] = None) -> List[MetricPoint]:
        """Get recent metric points."""
        with self._lock:
            points = list(self.metrics)
            
            if since:
                points = [p for p in points if p.timestamp >= since]
            
            if limit:
                points = points[-limit:]
            
            return points
    
    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        with self._lock:
            # Export counters
            for name, value in self.counters.items():
                lines.append(f"# TYPE {name} counter")
                lines.append(f"{name} {value}")
            
            # Export gauges
            for name, value in self.gauges.items():
                lines.append(f"# TYPE {name} gauge")
                lines.append(f"{name} {value}")
            
            # Export timer summaries
            for name, values in self.timers.items():
                if values:
                    stats = self.get_timer_stats(name)
                    lines.append(f"# TYPE {name} summary")
                    lines.append(f"{name}_count {stats['count']}")
                    lines.append(f"{name}_sum {sum(values)}")
                    for quantile in [0.5, 0.95, 0.99]:
                        key = f"p{int(quantile * 100)}"
                        if key in stats:
                            lines.append(f'{name}{{quantile="{quantile}"}} {stats[key]}')
        
        return "\n".join(lines)
    
    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self.metrics.clear()
            self.counters.clear()
            self.gauges.clear()
            self.timers.clear()


class TimerContext:
    """Context manager for timing operations."""
    
    def __init__(self, metrics: MetricsCollector, name: str, tags: Optional[Dict[str, str]] = None):
        self.metrics = metrics
        self.name = name
        self.tags = tags or {}
        self.start_time: Optional[float] = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.time() - self.start_time
            self.metrics.timer(self.name, duration, self.tags)


class DAGMetrics:
    """Specialized metrics for DAG execution."""
    
    def __init__(self, metrics: MetricsCollector):
        self.metrics = metrics
        self.logger = get_logger(__name__)
    
    def record_dag_start(self, dag_name: str, node_count: int) -> None:
        """Record DAG execution start."""
        self.metrics.counter("dag_executions_started", tags={"dag_name": dag_name})
        self.metrics.gauge("dag_node_count", node_count, tags={"dag_name": dag_name})
        self.logger.info("DAG execution started", extra={
            "dag_name": dag_name,
            "node_count": node_count,
            "event_type": "dag_start"
        })
    
    def record_dag_completion(self, dag_name: str, success: bool, duration: float, stats: Dict[str, int]) -> None:
        """Record DAG execution completion."""
        status = "success" if success else "failure"
        self.metrics.counter(f"dag_executions_{status}", tags={"dag_name": dag_name})
        self.metrics.timer("dag_execution_duration", duration, tags={"dag_name": dag_name, "status": status})
        
        for stat_name, count in stats.items():
            self.metrics.gauge(f"dag_nodes_{stat_name}", count, tags={"dag_name": dag_name})
        
        self.logger.info("DAG execution completed", extra={
            "dag_name": dag_name,
            "success": success,
            "duration": duration,
            "stats": stats,
            "event_type": "dag_completion"
        })
    
    def record_node_execution(self, node_id: str, node_type: str, status: str, duration: float, error: Optional[str] = None) -> None:
        """Record node execution metrics."""
        self.metrics.counter("node_executions_total", tags={"node_type": node_type, "status": status})
        self.metrics.timer("node_execution_duration", duration, tags={"node_type": node_type, "status": status})
        
        if error:
            self.metrics.counter("node_errors_total", tags={"node_type": node_type, "error_type": type(error).__name__})
        
        self.logger.info("Node execution recorded", extra={
            "node_id": node_id,
            "node_type": node_type,
            "status": status,
            "duration": duration,
            "error": error,
            "event_type": "node_execution"
        })
    
    def record_cache_operation(self, operation: str, hit: bool) -> None:
        """Record cache operation metrics."""
        status = "hit" if hit else "miss"
        self.metrics.counter(f"cache_{operation}_{status}")
        self.metrics.counter("cache_operations_total", tags={"operation": operation, "status": status})
    
    def record_api_call(self, provider: str, model: str, duration: float, success: bool, tokens: Optional[int] = None) -> None:
        """Record API call metrics."""
        status = "success" if success else "failure"
        self.metrics.counter("api_calls_total", tags={"provider": provider, "model": model, "status": status})
        self.metrics.timer("api_call_duration", duration, tags={"provider": provider, "model": model})
        
        if tokens:
            self.metrics.counter("api_tokens_total", tokens, tags={"provider": provider, "model": model})
            self.metrics.timer("api_tokens_per_call", tokens, tags={"provider": provider, "model": model})


# Global metrics collector
_metrics_collector = MetricsCollector()
dag_metrics = DAGMetrics(_metrics_collector)


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector."""
    return _metrics_collector


def timer(name: str, tags: Optional[Dict[str, str]] = None) -> TimerContext:
    """Create a timer context manager."""
    return TimerContext(_metrics_collector, name, tags)


def counter(name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None) -> None:
    """Record a counter metric."""
    _metrics_collector.counter(name, value, tags)


def gauge(name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
    """Record a gauge metric."""
    _metrics_collector.gauge(name, value, tags)


def histogram(name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
    """Record a histogram metric."""
    _metrics_collector.histogram(name, value, tags)


class PerformanceMonitor:
    """Monitor performance of DAG executions."""
    
    def __init__(self):
        self.metrics = get_metrics_collector()
        self.logger = get_logger(__name__)
        self.thresholds = {
            "node_execution_time": 60.0,  # seconds
            "dag_execution_time": 300.0,  # seconds
            "memory_usage_mb": 1000.0,    # MB
            "error_rate": 0.1             # 10%
        }
    
    def check_performance_thresholds(self) -> List[Dict[str, Any]]:
        """Check if any performance thresholds are exceeded."""
        alerts = []
        
        # Check node execution times
        for name in self.metrics.timers:
            if "node_execution_duration" in name:
                stats = self.metrics.get_timer_stats(name)
                if stats and stats.get("p95", 0) > self.thresholds["node_execution_time"]:
                    alerts.append({
                        "type": "performance",
                        "metric": "node_execution_time",
                        "value": stats["p95"],
                        "threshold": self.thresholds["node_execution_time"],
                        "message": f"Node execution time P95 ({stats['p95']:.2f}s) exceeds threshold"
                    })
        
        # Check error rates
        total_executions = self.metrics.get_counter("node_executions_total")
        failed_executions = self.metrics.get_counter("node_executions_failed")
        
        if total_executions > 0:
            error_rate = failed_executions / total_executions
            if error_rate > self.thresholds["error_rate"]:
                alerts.append({
                    "type": "error_rate",
                    "metric": "error_rate",
                    "value": error_rate,
                    "threshold": self.thresholds["error_rate"],
                    "message": f"Error rate ({error_rate:.1%}) exceeds threshold"
                })
        
        return alerts
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of performance metrics."""
        return {
            "timestamp": datetime.now().isoformat(),
            "metrics": self.metrics.get_all_metrics(),
            "alerts": self.check_performance_thresholds()
        }