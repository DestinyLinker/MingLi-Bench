"""
OpenAI model client implementation.
"""

import os
from typing import Optional
from openai import OpenAI

from .base import ModelClient
from ..utils.logger import get_logger

logger = get_logger(__name__)


class OpenAIClient(ModelClient):
    """Client for OpenAI models (GPT-3.5, GPT-4, etc.)"""
    
    def __init__(self, 
                 model_name: str = "gpt-4",
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 temperature: float = 0.0,
                 max_tokens: Optional[int] = None,
                 **kwargs):
        """
        Initialize OpenAI client.
        
        Args:
            model_name: Model to use (e.g., "gpt-4", "gpt-3.5-turbo")
            api_key: OpenAI API key (if not set, uses environment variable)
            base_url: Custom API base URL
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional configuration
        """
        # Initialize parent with API key handling
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            api_key_env_vars=["OPENAI_API_KEY"],
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        # Initialize client with timeout
        timeout_seconds = int(os.getenv("TIMEOUT", "30"))
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
            timeout=timeout_seconds
        )
        
        logger.info(f"Initialized OpenAI client with model: {model_name}, timeout: {timeout_seconds}s")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate response using OpenAI API.
        
        Args:
            prompt: Input prompt
            **kwargs: Override generation parameters
            
        Returns:
            Generated text
        """
        try:
            # Get parameters with overrides using base class method
            gen_params = self.get_generation_params(**kwargs)
            
            # Build API call parameters
            params = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                **gen_params
            }
            
            logger.debug(f"Making API call with model: {params['model']}")
            
            # Make API call
            response = self.client.chat.completions.create(**params)
            
            # Extract text
            text = response.choices[0].message.content.strip()
            
            return text
            
        except Exception as e:
            self.handle_api_error("OpenAI generation", e)
            raise
    
    def validate_api_key(self) -> bool:
        """
        Validate OpenAI API key.
        
        Returns:
            True if valid, False otherwise
        """
        try:
            # Try a simple API call
            self.client.models.list()
            return True
        except Exception as e:
            logger.error(f"Invalid OpenAI API key: {e}")
            return False
