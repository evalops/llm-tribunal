#!/usr/bin/env python3
"""
Centralized logging configuration and setup.
"""

import logging
import logging.handlers
import json
import sys
from typing import Dict, Any, Optional
from datetime import datetime
import os
from pathlib import Path

from config import get_config, LoggingConfig


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 
                          'msecs', 'relativeCreated', 'thread', 'threadName', 
                          'processName', 'process', 'getMessage', 'exc_info', 
                          'exc_text', 'stack_info']:
                log_entry[key] = value
        
        return json.dumps(log_entry)


class ContextFilter(logging.Filter):
    """Filter to add contextual information to log records."""
    
    def __init__(self, context: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.context = context or {}
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add context information to the record."""
        for key, value in self.context.items():
            setattr(record, key, value)
        return True
    
    def update_context(self, context: Dict[str, Any]) -> None:
        """Update the context information."""
        self.context.update(context)


class LoggingManager:
    """Manages logging configuration and setup."""
    
    def __init__(self, config: Optional[LoggingConfig] = None):
        self.config = config or get_config().logging
        self.context_filter = ContextFilter()
        self._configured = False
    
    def setup_logging(self) -> None:
        """Setup logging configuration based on config."""
        if self._configured:
            return
        
        # Get root logger
        root_logger = logging.getLogger()
        
        # Clear existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Set log level
        level = getattr(logging, self.config.level.upper(), logging.INFO)
        root_logger.setLevel(level)
        
        # Setup console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        if self.config.structured_logging:
            console_formatter = StructuredFormatter()
        else:
            console_formatter = logging.Formatter(self.config.format)
        
        console_handler.setFormatter(console_formatter)
        console_handler.addFilter(self.context_filter)
        root_logger.addHandler(console_handler)
        
        # Setup file handler if configured
        if self.config.file_path:
            file_path = Path(self.config.file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Use rotating file handler
            file_handler = logging.handlers.RotatingFileHandler(
                filename=file_path,
                maxBytes=self.config.max_file_size_mb * 1024 * 1024,
                backupCount=self.config.backup_count
            )
            file_handler.setLevel(level)
            
            if self.config.structured_logging:
                file_formatter = StructuredFormatter()
            else:
                file_formatter = logging.Formatter(self.config.format)
            
            file_handler.setFormatter(file_formatter)
            file_handler.addFilter(self.context_filter)
            root_logger.addHandler(file_handler)
        
        self._configured = True
        
        # Log configuration
        logger = logging.getLogger(__name__)
        logger.info("Logging configured", extra={
            "log_level": self.config.level,
            "structured_logging": self.config.structured_logging,
            "file_logging": bool(self.config.file_path)
        })
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger with the specified name."""
        if not self._configured:
            self.setup_logging()
        return logging.getLogger(name)
    
    def add_context(self, context: Dict[str, Any]) -> None:
        """Add context information to all log records."""
        self.context_filter.update_context(context)
    
    def create_execution_logger(self, execution_id: str, dag_name: Optional[str] = None) -> logging.Logger:
        """Create a logger for a specific DAG execution."""
        logger = self.get_logger(f"dag_execution.{execution_id}")
        
        # Add execution context
        context = {
            "execution_id": execution_id,
            "component": "dag_executor"
        }
        
        if dag_name:
            context["dag_name"] = dag_name
        
        # Create a context filter specific to this execution
        execution_filter = ContextFilter(context)
        
        # Add the filter to all handlers
        for handler in logging.getLogger().handlers:
            handler.addFilter(execution_filter)
        
        return logger


# Global logging manager instance
logging_manager = LoggingManager()


def setup_logging(config: Optional[LoggingConfig] = None) -> None:
    """Setup logging with the specified configuration."""
    global logging_manager
    if config:
        logging_manager = LoggingManager(config)
    logging_manager.setup_logging()


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name."""
    return logging_manager.get_logger(name)


def add_logging_context(context: Dict[str, Any]) -> None:
    """Add context information to all log records."""
    logging_manager.add_context(context)


class ExecutionTracker:
    """Tracks execution metrics and performance."""
    
    def __init__(self, execution_id: str):
        self.execution_id = execution_id
        self.logger = get_logger(f"execution_tracker.{execution_id}")
        self.start_time = datetime.now()
        self.metrics: Dict[str, Any] = {
            "execution_id": execution_id,
            "start_time": self.start_time.isoformat(),
            "nodes_executed": 0,
            "nodes_successful": 0,
            "nodes_failed": 0,
            "nodes_skipped": 0,
            "nodes_cached": 0,
            "total_execution_time": 0.0,
            "node_timings": {},
            "errors": []
        }
    
    def track_node_start(self, node_id: str) -> None:
        """Track the start of node execution."""
        self.logger.info(f"Starting node execution", extra={
            "node_id": node_id,
            "event_type": "node_start"
        })
    
    def track_node_completion(self, node_id: str, status: str, execution_time: float, error: Optional[str] = None) -> None:
        """Track the completion of node execution."""
        self.metrics["nodes_executed"] += 1
        
        if status == "success":
            self.metrics["nodes_successful"] += 1
        elif status == "failed":
            self.metrics["nodes_failed"] += 1
            if error:
                self.metrics["errors"].append({
                    "node_id": node_id,
                    "error": error,
                    "timestamp": datetime.now().isoformat()
                })
        elif status == "skipped":
            self.metrics["nodes_skipped"] += 1
        elif status == "cached":
            self.metrics["nodes_cached"] += 1
        
        self.metrics["node_timings"][node_id] = execution_time
        
        self.logger.info(f"Node execution completed", extra={
            "node_id": node_id,
            "status": status,
            "execution_time": execution_time,
            "error": error,
            "event_type": "node_completion"
        })
    
    def finalize(self) -> Dict[str, Any]:
        """Finalize tracking and return metrics."""
        end_time = datetime.now()
        self.metrics["end_time"] = end_time.isoformat()
        self.metrics["total_execution_time"] = (end_time - self.start_time).total_seconds()
        
        self.logger.info("DAG execution completed", extra={
            "metrics": self.metrics,
            "event_type": "execution_completion"
        })
        
        return self.metrics


def create_execution_tracker(execution_id: str) -> ExecutionTracker:
    """Create an execution tracker for a DAG execution."""
    return ExecutionTracker(execution_id)