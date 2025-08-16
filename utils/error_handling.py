#!/usr/bin/env python3
"""
Enhanced error handling with retry policies and circuit breakers.
"""

import time
import random
from typing import Callable, Any, Optional, Dict, List, Type, Union
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta
import functools

from utils.logging_setup import get_logger


class RetryStrategy(Enum):
    """Different retry strategies."""
    FIXED = "fixed"
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    JITTER = "jitter"


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    backoff_multiplier: float = 2.0
    jitter_range: float = 0.1
    
    # Which exceptions to retry on
    retryable_exceptions: tuple = (Exception,)
    # Which exceptions to never retry
    non_retryable_exceptions: tuple = (KeyboardInterrupt, SystemExit)


class RetryableError(Exception):
    """Base class for retryable errors."""
    pass


class NonRetryableError(Exception):
    """Base class for non-retryable errors."""
    pass


class APIError(RetryableError):
    """API-related error that can be retried."""
    pass


class ValidationError(NonRetryableError):
    """Validation error that should not be retried."""
    pass


class ConfigurationError(NonRetryableError):
    """Configuration error that should not be retried."""
    pass


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""
    pass


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Circuit breaker implementation."""
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 recovery_timeout: float = 60.0,
                 expected_exception: Type[Exception] = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = CircuitBreakerState.CLOSED
        self.logger = get_logger(__name__)
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to apply circuit breaker to a function."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Call function with circuit breaker protection."""
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
                self.logger.info("Circuit breaker transitioning to HALF_OPEN")
            else:
                raise CircuitBreakerError("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        return datetime.now() - self.last_failure_time > timedelta(seconds=self.recovery_timeout)
    
    def _on_success(self) -> None:
        """Handle successful call."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.CLOSED
            self.logger.info("Circuit breaker transitioning to CLOSED")
        self.failure_count = 0
    
    def _on_failure(self) -> None:
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            self.logger.warning(f"Circuit breaker transitioning to OPEN after {self.failure_count} failures")


class RetryHandler:
    """Handles retry logic with different strategies."""
    
    def __init__(self, config: RetryConfig):
        self.config = config
        self.logger = get_logger(__name__)
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to apply retry logic to a function."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.execute_with_retry(func, *args, **kwargs)
        return wrapper
    
    def execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic."""
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                result = func(*args, **kwargs)
                if attempt > 0:
                    self.logger.info(f"Function succeeded on attempt {attempt + 1}")
                return result
                
            except Exception as e:
                last_exception = e
                
                # Check if exception is retryable
                if isinstance(e, self.config.non_retryable_exceptions):
                    self.logger.error(f"Non-retryable exception occurred: {e}")
                    raise e
                
                if not isinstance(e, self.config.retryable_exceptions):
                    self.logger.error(f"Exception not in retryable list: {e}")
                    raise e
                
                # Don't delay after the last attempt
                if attempt < self.config.max_attempts - 1:
                    delay = self._calculate_delay(attempt)
                    self.logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s")
                    time.sleep(delay)
                else:
                    self.logger.error(f"All {self.config.max_attempts} attempts failed. Last error: {e}")
        
        # If we get here, all retries failed
        raise last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay based on retry strategy."""
        if self.config.strategy == RetryStrategy.FIXED:
            delay = self.config.base_delay
        
        elif self.config.strategy == RetryStrategy.EXPONENTIAL:
            delay = self.config.base_delay * (self.config.backoff_multiplier ** attempt)
        
        elif self.config.strategy == RetryStrategy.LINEAR:
            delay = self.config.base_delay * (attempt + 1)
        
        elif self.config.strategy == RetryStrategy.JITTER:
            base_delay = self.config.base_delay * (self.config.backoff_multiplier ** attempt)
            jitter = random.uniform(-self.config.jitter_range, self.config.jitter_range) * base_delay
            delay = base_delay + jitter
        
        else:
            delay = self.config.base_delay
        
        # Ensure delay doesn't exceed max_delay
        return min(delay, self.config.max_delay)


class ErrorContext:
    """Context manager for error handling and logging."""
    
    def __init__(self, operation: str, logger: Optional[logging.Logger] = None, 
                 context: Optional[Dict[str, Any]] = None):
        self.operation = operation
        self.logger = logger or get_logger(__name__)
        self.context = context or {}
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.logger.info(f"Starting {self.operation}", extra=self.context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time if self.start_time else 0
        
        if exc_type is None:
            self.logger.info(f"Completed {self.operation} successfully", 
                           extra={**self.context, "duration": duration})
        else:
            self.logger.error(f"Failed {self.operation}: {exc_val}", 
                            extra={**self.context, "duration": duration, "error_type": exc_type.__name__})
        
        return False  # Don't suppress exceptions


def create_retry_decorator(max_attempts: int = 3, 
                          base_delay: float = 1.0,
                          strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
                          retryable_exceptions: tuple = (Exception,),
                          non_retryable_exceptions: tuple = (KeyboardInterrupt, SystemExit)) -> Callable:
    """Create a retry decorator with the specified configuration."""
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        strategy=strategy,
        retryable_exceptions=retryable_exceptions,
        non_retryable_exceptions=non_retryable_exceptions
    )
    return RetryHandler(config)


def create_circuit_breaker(failure_threshold: int = 5,
                          recovery_timeout: float = 60.0,
                          expected_exception: Type[Exception] = Exception) -> CircuitBreaker:
    """Create a circuit breaker with the specified configuration."""
    return CircuitBreaker(failure_threshold, recovery_timeout, expected_exception)


# Pre-configured retry decorators
api_retry = create_retry_decorator(
    max_attempts=3,
    base_delay=1.0,
    strategy=RetryStrategy.EXPONENTIAL,
    retryable_exceptions=(APIError, ConnectionError, TimeoutError),
    non_retryable_exceptions=(ValidationError, ConfigurationError, KeyboardInterrupt, SystemExit)
)

io_retry = create_retry_decorator(
    max_attempts=2,
    base_delay=0.5,
    strategy=RetryStrategy.FIXED,
    retryable_exceptions=(IOError, OSError),
    non_retryable_exceptions=(KeyboardInterrupt, SystemExit)
)

# Pre-configured circuit breakers
api_circuit_breaker = create_circuit_breaker(
    failure_threshold=5,
    recovery_timeout=60.0,
    expected_exception=APIError
)


class ErrorAggregator:
    """Aggregates and analyzes errors across the system."""
    
    def __init__(self, max_errors: int = 1000):
        self.max_errors = max_errors
        self.errors: List[Dict[str, Any]] = []
        self.error_counts: Dict[str, int] = {}
        self.logger = get_logger(__name__)
    
    def record_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
        """Record an error with context."""
        error_info = {
            "timestamp": datetime.now().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context or {}
        }
        
        self.errors.append(error_info)
        self.error_counts[error_info["error_type"]] = self.error_counts.get(error_info["error_type"], 0) + 1
        
        # Keep only the most recent errors
        if len(self.errors) > self.max_errors:
            self.errors = self.errors[-self.max_errors:]
        
        self.logger.error(f"Error recorded: {error}", extra=error_info)
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get a summary of recorded errors."""
        return {
            "total_errors": len(self.errors),
            "error_counts": dict(self.error_counts),
            "recent_errors": self.errors[-10:] if self.errors else []
        }
    
    def get_error_rate(self, time_window_minutes: int = 60) -> float:
        """Get error rate within a time window."""
        cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
        
        recent_errors = [
            error for error in self.errors
            if datetime.fromisoformat(error["timestamp"]) > cutoff_time
        ]
        
        return len(recent_errors) / time_window_minutes  # Errors per minute
    
    def clear_errors(self) -> None:
        """Clear all recorded errors."""
        self.errors.clear()
        self.error_counts.clear()


# Global error aggregator
error_aggregator = ErrorAggregator()


def record_error(error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
    """Record an error globally."""
    error_aggregator.record_error(error, context)


def get_error_summary() -> Dict[str, Any]:
    """Get global error summary."""
    return error_aggregator.get_error_summary()