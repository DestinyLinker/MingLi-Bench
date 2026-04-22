"""
Base class for model clients.
"""

import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List

from ..utils.logger import get_logger

logger = get_logger(__name__)


class ModelClient(ABC):
    """Abstract base class for LLM model clients."""
    
    # Shared system prompt for all models
    SYSTEM_PROMPT = "你是一位精通中国传统命理学的专家，包括八字命理、紫微斗数等。请根据给定的信息进行分析和回答。"
    
    # Default generation parameters
    DEFAULT_TEMPERATURE = 0.0
    DEFAULT_MAX_TOKENS = 8192
    
    def __init__(self, 
                 model_name: str,
                 api_key: Optional[str] = None,
                 api_key_env_vars: Optional[List[str]] = None,
                 temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None,
                 **kwargs):
        """
        Initialize the model client.
        
        Args:
            model_name: Name of the model to use
            api_key: API key (optional, will check env vars if not provided)
            api_key_env_vars: List of environment variable names to check
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional configuration options
        """
        self.model_name = model_name
        
        # Get API key if env vars provided
        if api_key_env_vars:
            self.api_key = self._get_api_key(api_key, api_key_env_vars)
        else:
            self.api_key = api_key
            
        # Set generation parameters with defaults
        self.temperature = temperature if temperature is not None else self.DEFAULT_TEMPERATURE
        env_max_tokens = os.getenv("MAX_TOKENS")
        self.max_tokens = (
            max_tokens
            if max_tokens is not None
            else int(env_max_tokens) if env_max_tokens is not None
            else self.DEFAULT_MAX_TOKENS
        )
        
        # Store additional config
        self.config = kwargs
        
        logger.info(f"Initialized {self.__class__.__name__} with model: {model_name}")
    
    def _get_api_key(self, api_key: Optional[str], env_vars: List[str]) -> str:
        """
        Get API key from provided value or environment variables.
        
        Args:
            api_key: Directly provided API key
            env_vars: List of environment variable names to check
            
        Returns:
            API key string
            
        Raises:
            ValueError: If no API key found
        """
        if api_key:
            return api_key
            
        for var in env_vars:
            if key := os.getenv(var):
                return key
                
        provider = self.__class__.__name__.replace('Client', '')
        raise ValueError(
            f"{provider} API key not found. "
            f"Set one of: {', '.join(env_vars)} environment variable."
        )
    
    def handle_api_error(self, operation: str, error: Exception):
        """
        Handle API errors consistently.
        
        Args:
            operation: Description of the operation
            error: The exception that occurred
        """
        logger.error(f"Error in {operation} for {self.model_name}: {error}")
        
    def get_generation_params(self, **kwargs) -> Dict[str, Any]:
        """
        Get generation parameters with defaults and overrides.
        
        Args:
            **kwargs: Override parameters
            
        Returns:
            Merged parameters dictionary
        """
        params = {
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
        }
        params.update(kwargs)
        return params
        
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a response from the model.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
        """
        pass
    
    def validate_api_key(self) -> bool:
        """
        Base implementation for API key validation.
        Subclasses should override with specific validation logic.
        
        Returns:
            True if API key exists, False otherwise
        """
        return bool(self.api_key)
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get current configuration.
        
        Returns:
            Configuration dictionary
        """
        return {
            'model_name': self.model_name,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            **self.config
        }
