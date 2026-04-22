"""
DeepSeek model client implementation.
"""

import os
from typing import Optional
from openai import OpenAI

from .base import ModelClient
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DeepSeekClient(ModelClient):
    """Client for DeepSeek models (DeepSeek uses OpenAI-compatible API)"""
    
    # API key environment variables
    API_KEY_ENV_VARS = ["DEEPSEEK_API_KEY"]
    
    # Default base URL for DeepSeek
    DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
    
    def __init__(self, 
                 model_name: str = "deepseek-chat",
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 **kwargs):
        """
        Initialize DeepSeek client.
        
        Args:
            model_name: Model to use (e.g., "deepseek-chat", "deepseek-coder")
            api_key: DeepSeek API key
            base_url: API base URL (default: DeepSeek's API)
            **kwargs: Additional configuration
        """
        # Initialize parent with common functionality
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            api_key_env_vars=self.API_KEY_ENV_VARS,
            **kwargs
        )
        
        # Initialize OpenAI-compatible client
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=base_url or os.getenv("DEEPSEEK_BASE_URL", self.DEFAULT_BASE_URL)
        )
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate response using DeepSeek API.
        
        Args:
            prompt: Input prompt
            **kwargs: Override generation parameters
            
        Returns:
            Generated text
        """
        try:
            # Get parameters with overrides
            params = self.get_generation_params(**kwargs)
            
            # Make API call (OpenAI-compatible)
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                **params
            )
            
            # Extract text
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            self.handle_api_error("DeepSeek generation", e)
            raise
    
    def validate_api_key(self) -> bool:
        """
        Validate DeepSeek API key.
        
        Returns:
            True if valid, False otherwise
        """
        try:
            # Try a simple API call
            self.client.models.list()
            return True
        except Exception as e:
            logger.error(f"Invalid DeepSeek API key: {e}")
            return False