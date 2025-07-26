"""
Configuration Management for miStudioExplain Service

Centralized configuration with environment variable support and validation.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing."""
    pass


@dataclass
class OllamaConfig:
    """Ollama service configuration."""
    endpoint: str = "http://localhost:11434"
    namespace: str = "mistudio"
    models: List[str] = field(default_factory=lambda: ["llama3.1:8b", "llama3.1:70b"])
    timeout: int = 300
    max_retries: int = 3
    
    @classmethod
    def from_env(cls) -> 'OllamaConfig':
        """Create configuration from environment variables."""
        return cls(
            endpoint=os.getenv("OLLAMA_ENDPOINT", cls.endpoint),
            namespace=os.getenv("OLLAMA_NAMESPACE", cls.namespace),
            models=os.getenv("OLLAMA_MODELS", ",".join(cls.models)).split(","),
            timeout=int(os.getenv("OLLAMA_TIMEOUT", str(cls.timeout))),
            max_retries=int(os.getenv("OLLAMA_MAX_RETRIES", str(cls.max_retries)))
        )


@dataclass
class ProcessingConfig:
    """Processing and quality configuration."""
    default_quality_threshold: float = 0.4
    high_quality_threshold: float = 0.6
    excellent_threshold: float = 0.8
    max_batch_size: int = 10
    max_concurrent_explanations: int = 4
    explanation_timeout: int = 120
    
    @classmethod
    def from_env(cls) -> 'ProcessingConfig':
        """Create configuration from environment variables."""
        return cls(
            default_quality_threshold=float(os.getenv("QUALITY_THRESHOLD", str(cls.default_quality_threshold))),
            high_quality_threshold=float(os.getenv("HIGH_QUALITY_THRESHOLD", str(cls.high_quality_threshold))),
            excellent_threshold=float(os.getenv("EXCELLENT_THRESHOLD", str(cls.excellent_threshold))),
            max_batch_size=int(os.getenv("MAX_BATCH_SIZE", str(cls.max_batch_size))),
            max_concurrent_explanations=int(os.getenv("MAX_CONCURRENT_EXPLANATIONS", str(cls.max_concurrent_explanations))),
            explanation_timeout=int(os.getenv("EXPLANATION_TIMEOUT", str(cls.explanation_timeout)))
        )


@dataclass
class StorageConfig:
    """Storage and file path configuration."""
    data_path: str = "./data"
    cache_path: str = "./data/cache"
    input_path: str = "./data/input"
    output_path: str = "./data/output"
    logs_path: str = "./logs"
    max_cache_size_gb: int = 10
    
    @classmethod
    def from_env(cls) -> 'StorageConfig':
        """Create configuration from environment variables."""
        return cls(
            data_path=os.getenv("MISTUDIO_DATA_PATH", cls.data_path),
            cache_path=os.getenv("MISTUDIO_CACHE_PATH", cls.cache_path),
            input_path=os.getenv("MISTUDIO_INPUT_PATH", cls.input_path),
            output_path=os.getenv("MISTUDIO_OUTPUT_PATH", cls.output_path),
            logs_path=os.getenv("MISTUDIO_LOGS_PATH", cls.logs_path),
            max_cache_size_gb=int(os.getenv("MAX_CACHE_SIZE_GB", str(cls.max_cache_size_gb)))
        )


@dataclass
class APIConfig:
    """API service configuration."""
    host: str = "0.0.0.0"
    port: int = 8003
    workers: int = 1
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    api_key: Optional[str] = None
    rate_limit_per_minute: int = 100
    
    @classmethod
    def from_env(cls) -> 'APIConfig':
        """Create configuration from environment variables."""
        return cls(
            host=os.getenv("API_HOST", cls.host),
            port=int(os.getenv("API_PORT", str(cls.port))),
            workers=int(os.getenv("API_WORKERS", str(cls.workers))),
            cors_origins=os.getenv("CORS_ORIGINS", ",".join(cls.cors_origins)).split(","),
            api_key=os.getenv("API_KEY"),
            rate_limit_per_minute=int(os.getenv("RATE_LIMIT_PER_MINUTE", str(cls.rate_limit_per_minute)))
        )


@dataclass
class Config:
    """Main configuration container."""
    service_name: str = "miStudioExplain"
    service_version: str = "1.0.0"
    log_level: str = "INFO"
    debug: bool = False
    environment: str = "development"
    
    # Sub-configurations
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    api: APIConfig = field(default_factory=APIConfig)
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Create complete configuration from environment variables."""
        return cls(
            service_name=os.getenv("SERVICE_NAME", cls.service_name),
            service_version=os.getenv("SERVICE_VERSION", cls.service_version),
            log_level=os.getenv("LOG_LEVEL", cls.log_level).upper(),
            debug=os.getenv("DEBUG", "false").lower() == "true",
            environment=os.getenv("ENVIRONMENT", cls.environment),
            ollama=OllamaConfig.from_env(),
            processing=ProcessingConfig.from_env(),
            storage=StorageConfig.from_env(),
            api=APIConfig.from_env()
        )
    
    def validate(self) -> None:
        """Validate configuration values."""
        # Validate thresholds
        if not (0.0 <= self.processing.default_quality_threshold <= 1.0):
            raise ConfigurationError("Quality threshold must be between 0.0 and 1.0")
        
        if self.processing.high_quality_threshold <= self.processing.default_quality_threshold:
            raise ConfigurationError("High quality threshold must be greater than default threshold")
        
        if self.processing.excellent_threshold <= self.processing.high_quality_threshold:
            raise ConfigurationError("Excellent threshold must be greater than high quality threshold")
        
        # Validate paths exist or can be created
        paths_to_check = [
            self.storage.data_path,
            self.storage.cache_path,
            self.storage.input_path,
            self.storage.output_path,
            self.storage.logs_path
        ]
        
        for path_str in paths_to_check:
            path = Path(path_str)
            try:
                path.mkdir(parents=True, exist_ok=True)
            except PermissionError:
                raise ConfigurationError(f"Cannot create directory: {path}")
        
        # Validate numeric ranges
        if self.processing.max_batch_size < 1:
            raise ConfigurationError("Max batch size must be at least 1")
        
        if self.processing.max_concurrent_explanations < 1:
            raise ConfigurationError("Max concurrent explanations must be at least 1")
        
        if self.api.port < 1 or self.api.port > 65535:
            raise ConfigurationError("API port must be between 1 and 65535")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "service_name": self.service_name,
            "service_version": self.service_version,
            "log_level": self.log_level,
            "debug": self.debug,
            "environment": self.environment,
            "ollama": {
                "endpoint": self.ollama.endpoint,
                "namespace": self.ollama.namespace,
                "models": self.ollama.models,
                "timeout": self.ollama.timeout,
                "max_retries": self.ollama.max_retries
            },
            "processing": {
                "default_quality_threshold": self.processing.default_quality_threshold,
                "high_quality_threshold": self.processing.high_quality_threshold,
                "excellent_threshold": self.processing.excellent_threshold,
                "max_batch_size": self.processing.max_batch_size,
                "max_concurrent_explanations": self.processing.max_concurrent_explanations,
                "explanation_timeout": self.processing.explanation_timeout
            },
            "storage": {
                "data_path": self.storage.data_path,
                "cache_path": self.storage.cache_path,
                "input_path": self.storage.input_path,
                "output_path": self.storage.output_path,
                "logs_path": self.storage.logs_path,
                "max_cache_size_gb": self.storage.max_cache_size_gb
            },
            "api": {
                "host": self.api.host,
                "port": self.api.port,
                "workers": self.api.workers,
                "cors_origins": self.api.cors_origins,
                "api_key_set": bool(self.api.api_key),
                "rate_limit_per_minute": self.api.rate_limit_per_minute
            }
        }


class ConfigManager:
    """Configuration manager with caching and validation."""
    
    def __init__(self):
        self._config: Optional[Config] = None
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def get_config(self, force_reload: bool = False) -> Config:
        """
        Get configuration with caching.
        
        Args:
            force_reload: Force reload from environment
            
        Returns:
            Validated configuration object
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        if self._config is None or force_reload:
            try:
                self._config = Config.from_env()
                self._config.validate()
                self.logger.info("Configuration loaded and validated successfully")
                
                if self._config.debug:
                    self.logger.debug(f"Configuration: {self._config.to_dict()}")
                    
            except Exception as e:
                self.logger.error(f"Failed to load configuration: {e}")
                raise ConfigurationError(f"Configuration error: {e}")
        
        return self._config
    
    def reload_config(self) -> Config:
        """Reload configuration from environment."""
        return self.get_config(force_reload=True)
    
    def validate_config(self) -> bool:
        """
        Validate current configuration.
        
        Returns:
            True if configuration is valid
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        config = self.get_config()
        config.validate()
        return True


# Global configuration manager instance
config_manager = ConfigManager()


def get_config() -> Config:
    """Get the global configuration instance."""
    return config_manager.get_config()


def reload_config() -> Config:
    """Reload the global configuration from environment."""
    return config_manager.reload_config()


if __name__ == "__main__":
    # Test configuration loading
    config = get_config()
    print(f"Loaded configuration for {config.service_name} v{config.service_version}")
    print(f"Environment: {config.environment}")
    print(f"Debug mode: {config.debug}")