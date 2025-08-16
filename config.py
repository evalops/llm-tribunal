#!/usr/bin/env python3
"""
Centralized configuration management for the DAG evaluation system.
"""

import os
import json
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from pathlib import Path
import yaml
from dotenv import load_dotenv
import logging
from logging.handlers import RotatingFileHandler

# Load environment variables
load_dotenv()


@dataclass
class APIConfig:
    """Configuration for API providers."""
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None
    anthropic_base_url: Optional[str] = None
    default_timeout: int = 30
    max_retries: int = 3
    
    def __post_init__(self):
        """Load API keys from environment if not provided."""
        if self.openai_api_key is None:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if self.anthropic_api_key is None:
            self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if self.openai_base_url is None:
            self.openai_base_url = os.getenv("OPENAI_BASE_URL")
        if self.anthropic_base_url is None:
            self.anthropic_base_url = os.getenv("ANTHROPIC_BASE_URL")
    
    def validate(self) -> List[str]:
        """Validate API configuration and return list of errors."""
        errors = []
        # Do not require API keys at validation time.
        # Many workflows (e.g., local Ollama) do not need them. Individual
        # provider clients should validate presence when actually used.
        
        if self.default_timeout <= 0:
            errors.append("Default timeout must be positive")
        
        if self.max_retries < 0:
            errors.append("Max retries must be non-negative")
        
        return errors


@dataclass 
class CacheConfig:
    """Configuration for caching."""
    enabled: bool = True
    cache_dir: str = ".dag_cache"
    max_cache_size_mb: int = 1000
    cache_ttl_hours: int = 24
    
    def validate(self) -> List[str]:
        """Validate cache configuration."""
        errors = []
        
        if self.max_cache_size_mb <= 0:
            errors.append("Cache size must be positive")
        
        if self.cache_ttl_hours <= 0:
            errors.append("Cache TTL must be positive")
        
        return errors


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_file_size_mb: int = 10
    backup_count: int = 5
    structured_logging: bool = False
    
    def validate(self) -> List[str]:
        """Validate logging configuration."""
        errors = []
        
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.level.upper() not in valid_levels:
            errors.append(f"Log level must be one of: {valid_levels}")
        
        if self.max_file_size_mb <= 0:
            errors.append("Max file size must be positive")
        
        if self.backup_count < 0:
            errors.append("Backup count must be non-negative")
        
        return errors


def setup_logging(logging_config: "LoggingConfig") -> None:
    """Configure Python logging according to LoggingConfig.

    Applies root logger level/format and optional rotating file handler.
    Safe to call multiple times; subsequent calls update handlers/levels.
    """
    try:
        level = getattr(logging, logging_config.level.upper(), logging.INFO)
    except Exception:
        level = logging.INFO

    # Basic console handler
    logging.basicConfig(level=level, format=logging_config.format)

    # Optional file handler
    root_logger = logging.getLogger()
    # Remove existing RotatingFileHandler if present to avoid duplicates
    for h in list(root_logger.handlers):
        if isinstance(h, RotatingFileHandler):
            root_logger.removeHandler(h)

    if logging_config.file_path:
        try:
            file_handler = RotatingFileHandler(
                logging_config.file_path,
                maxBytes=int(logging_config.max_file_size_mb * 1024 * 1024),
                backupCount=logging_config.backup_count,
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(logging.Formatter(logging_config.format))
            root_logger.addHandler(file_handler)
        except Exception:
            # Fall back silently if file handler can't be created
            pass


@dataclass
class EvaluationConfig:
    """Configuration for evaluation parameters."""
    default_model: str = "gpt-4o-mini"
    default_temperature: float = 0.0
    default_max_tokens: int = 1000
    parallel_execution: bool = True
    max_concurrent_nodes: int = 4
    default_retry_attempts: int = 3
    retry_delay_seconds: float = 1.0
    
    def validate(self) -> List[str]:
        """Validate evaluation configuration."""
        errors = []
        
        if self.default_temperature < 0 or self.default_temperature > 2:
            errors.append("Temperature must be between 0 and 2")
        
        if self.default_max_tokens <= 0:
            errors.append("Max tokens must be positive")
        
        if self.max_concurrent_nodes <= 0:
            errors.append("Max concurrent nodes must be positive")
        
        if self.default_retry_attempts < 0:
            errors.append("Retry attempts must be non-negative")
        
        if self.retry_delay_seconds < 0:
            errors.append("Retry delay must be non-negative")
        
        return errors


@dataclass
class SystemConfig:
    """Main system configuration."""
    api: APIConfig = field(default_factory=APIConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    # Environment settings
    environment: str = "development"
    debug: bool = False
    
    def validate(self) -> List[str]:
        """Validate entire configuration."""
        errors = []
        
        # Validate each section
        errors.extend([f"API: {err}" for err in self.api.validate()])
        errors.extend([f"Cache: {err}" for err in self.cache.validate()])
        errors.extend([f"Logging: {err}" for err in self.logging.validate()])
        errors.extend([f"Evaluation: {err}" for err in self.evaluation.validate()])
        
        # Validate environment
        valid_environments = ["development", "staging", "production"]
        if self.environment not in valid_environments:
            errors.append(f"Environment must be one of: {valid_environments}")
        
        return errors
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"


class ConfigurationError(Exception):
    """Raised when configuration validation fails."""
    pass


class ConfigManager:
    """Manages system configuration loading and validation."""
    
    def __init__(self):
        self._config: Optional[SystemConfig] = None
        self._config_file_path: Optional[str] = None
    
    def load_from_file(self, config_path: str) -> SystemConfig:
        """Load configuration from a file (JSON or YAML)."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() in ['.yml', '.yaml']:
                    data = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    data = json.load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {config_path.suffix}")
            
            return self._parse_config_data(data)
            
        except Exception as e:
            raise ConfigurationError(f"Failed to load config from {config_path}: {e}")
    
    def load_from_env(self) -> SystemConfig:
        """Load configuration from environment variables."""
        # Create config with environment-based overrides
        config = SystemConfig()
        
        # Override with environment variables
        if os.getenv("ENVIRONMENT"):
            config.environment = os.getenv("ENVIRONMENT")
        
        if os.getenv("DEBUG"):
            config.debug = os.getenv("DEBUG").lower() in ("true", "1", "yes")
        
        # API configuration
        if os.getenv("DEFAULT_TIMEOUT"):
            config.api.default_timeout = int(os.getenv("DEFAULT_TIMEOUT"))
        
        if os.getenv("MAX_RETRIES"):
            config.api.max_retries = int(os.getenv("MAX_RETRIES"))
        
        # Cache configuration
        if os.getenv("CACHE_ENABLED"):
            config.cache.enabled = os.getenv("CACHE_ENABLED").lower() in ("true", "1", "yes")
        
        if os.getenv("CACHE_DIR"):
            config.cache.cache_dir = os.getenv("CACHE_DIR")
        
        # Logging configuration
        if os.getenv("LOG_LEVEL"):
            config.logging.level = os.getenv("LOG_LEVEL")
        
        if os.getenv("LOG_FILE"):
            config.logging.file_path = os.getenv("LOG_FILE")
        
        # Evaluation configuration
        if os.getenv("DEFAULT_MODEL"):
            config.evaluation.default_model = os.getenv("DEFAULT_MODEL")
        
        if os.getenv("DEFAULT_TEMPERATURE"):
            config.evaluation.default_temperature = float(os.getenv("DEFAULT_TEMPERATURE"))
        
        if os.getenv("MAX_CONCURRENT_NODES"):
            config.evaluation.max_concurrent_nodes = int(os.getenv("MAX_CONCURRENT_NODES"))
        
        return config
    
    def load_default(self) -> SystemConfig:
        """Load default configuration."""
        return SystemConfig()
    
    def load(self, config_path: Optional[str] = None) -> SystemConfig:
        """Load configuration with automatic fallback."""
        config = None
        
        # Try to load from specified file
        if config_path:
            try:
                config = self.load_from_file(config_path)
                self._config_file_path = config_path
            except Exception as e:
                print(f"Warning: Failed to load config from {config_path}: {e}")
        
        # Try to load from default locations
        if config is None:
            default_paths = [
                "config.yaml",
                "config.yml", 
                "config.json",
                ".config/eval_config.yaml",
                os.path.expanduser("~/.eval_config.yaml")
            ]
            
            for path in default_paths:
                try:
                    config = self.load_from_file(path)
                    self._config_file_path = path
                    break
                except (FileNotFoundError, ConfigurationError):
                    continue
        
        # Fallback to environment and defaults
        if config is None:
            config = self.load_from_env()
        
        # Validate configuration
        errors = config.validate()
        if errors:
            raise ConfigurationError(f"Configuration validation failed:\n" + 
                                   "\n".join(f"  - {error}" for error in errors))
        
        self._config = config
        return config
    
    def get(self) -> SystemConfig:
        """Get current configuration, loading if necessary."""
        if self._config is None:
            self._config = self.load()
        return self._config
    
    def _parse_config_data(self, data: Dict[str, Any]) -> SystemConfig:
        """Parse configuration data into SystemConfig object."""
        config = SystemConfig()
        
        # Parse API config
        if "api" in data:
            api_data = data["api"]
            config.api = APIConfig(
                openai_api_key=api_data.get("openai_api_key"),
                anthropic_api_key=api_data.get("anthropic_api_key"),
                openai_base_url=api_data.get("openai_base_url"),
                anthropic_base_url=api_data.get("anthropic_base_url"),
                default_timeout=api_data.get("default_timeout", config.api.default_timeout),
                max_retries=api_data.get("max_retries", config.api.max_retries)
            )
        
        # Parse cache config
        if "cache" in data:
            cache_data = data["cache"]
            config.cache = CacheConfig(
                enabled=cache_data.get("enabled", config.cache.enabled),
                cache_dir=cache_data.get("cache_dir", config.cache.cache_dir),
                max_cache_size_mb=cache_data.get("max_cache_size_mb", config.cache.max_cache_size_mb),
                cache_ttl_hours=cache_data.get("cache_ttl_hours", config.cache.cache_ttl_hours)
            )
        
        # Parse logging config
        if "logging" in data:
            log_data = data["logging"]
            config.logging = LoggingConfig(
                level=log_data.get("level", config.logging.level),
                format=log_data.get("format", config.logging.format),
                file_path=log_data.get("file_path"),
                max_file_size_mb=log_data.get("max_file_size_mb", config.logging.max_file_size_mb),
                backup_count=log_data.get("backup_count", config.logging.backup_count),
                structured_logging=log_data.get("structured_logging", config.logging.structured_logging)
            )
        
        # Parse evaluation config
        if "evaluation" in data:
            eval_data = data["evaluation"]
            config.evaluation = EvaluationConfig(
                default_model=eval_data.get("default_model", config.evaluation.default_model),
                default_temperature=eval_data.get("default_temperature", config.evaluation.default_temperature),
                default_max_tokens=eval_data.get("default_max_tokens", config.evaluation.default_max_tokens),
                parallel_execution=eval_data.get("parallel_execution", config.evaluation.parallel_execution),
                max_concurrent_nodes=eval_data.get("max_concurrent_nodes", config.evaluation.max_concurrent_nodes),
                default_retry_attempts=eval_data.get("default_retry_attempts", config.evaluation.default_retry_attempts),
                retry_delay_seconds=eval_data.get("retry_delay_seconds", config.evaluation.retry_delay_seconds)
            )
        
        # Parse system-level config
        if "environment" in data:
            config.environment = data["environment"]
        
        if "debug" in data:
            config.debug = data["debug"]
        
        return config
    
    def save_to_file(self, config_path: str, config: Optional[SystemConfig] = None) -> None:
        """Save configuration to a file."""
        if config is None:
            config = self.get()
        
        config_path = Path(config_path)
        
        # Convert config to dictionary
        config_dict = {
            "environment": config.environment,
            "debug": config.debug,
            "api": {
                "openai_api_key": config.api.openai_api_key,
                "anthropic_api_key": config.api.anthropic_api_key,
                "openai_base_url": config.api.openai_base_url,
                "anthropic_base_url": config.api.anthropic_base_url,
                "default_timeout": config.api.default_timeout,
                "max_retries": config.api.max_retries
            },
            "cache": {
                "enabled": config.cache.enabled,
                "cache_dir": config.cache.cache_dir,
                "max_cache_size_mb": config.cache.max_cache_size_mb,
                "cache_ttl_hours": config.cache.cache_ttl_hours
            },
            "logging": {
                "level": config.logging.level,
                "format": config.logging.format,
                "file_path": config.logging.file_path,
                "max_file_size_mb": config.logging.max_file_size_mb,
                "backup_count": config.logging.backup_count,
                "structured_logging": config.logging.structured_logging
            },
            "evaluation": {
                "default_model": config.evaluation.default_model,
                "default_temperature": config.evaluation.default_temperature,
                "default_max_tokens": config.evaluation.default_max_tokens,
                "parallel_execution": config.evaluation.parallel_execution,
                "max_concurrent_nodes": config.evaluation.max_concurrent_nodes,
                "default_retry_attempts": config.evaluation.default_retry_attempts,
                "retry_delay_seconds": config.evaluation.retry_delay_seconds
            }
        }
        
        # Save based on file extension
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            if config_path.suffix.lower() in ['.yml', '.yaml']:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            elif config_path.suffix.lower() == '.json':
                json.dump(config_dict, f, indent=2)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")


# Global configuration manager instance
config_manager = ConfigManager()


def get_config() -> SystemConfig:
    """Get the current system configuration."""
    return config_manager.get()


def load_config(config_path: Optional[str] = None) -> SystemConfig:
    """Load configuration from file or environment."""
    return config_manager.load(config_path)
