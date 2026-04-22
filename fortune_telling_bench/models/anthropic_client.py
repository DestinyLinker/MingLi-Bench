"""
Anthropic Claude model client implementation.
"""

import os
from typing import Optional
import anthropic

from .base import ModelClient
from ..utils.logger import get_logger

logger = get_logger(__name__)


class AnthropicClient(ModelClient):
    """Client for Anthropic Claude models"""
    
    def __init__(self, 
                 model_name: str = "claude-3-sonnet-20240229",
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 temperature: float = 0.0,
                 max_tokens: Optional[int] = None,
                 **kwargs):
        """
        Initialize Anthropic client.
        
        Args:
            model_name: Model to use (e.g., "claude-3-sonnet-20240229")
            api_key: Anthropic API key (if not set, uses environment variable)
            base_url: Custom API base URL (if not set, uses environment variable)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional configuration
        """
        # Initialize parent with API key handling
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            api_key_env_vars=["CLAUDE_API_KEY", "ANTHROPIC_API_KEY"],
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        # Set base URL if provided
        self.base_url = base_url or os.getenv("CLAUDE_BASE_URL") or os.getenv("ANTHROPIC_BASE_URL")
        
        # Initialize client
        if self.base_url:
            self.client = anthropic.Anthropic(api_key=self.api_key, base_url=self.base_url)
            logger.info(f"Using custom base URL: {self.base_url}")
        else:
            self.client = anthropic.Anthropic(api_key=self.api_key)
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate response using Anthropic API.
        
        Args:
            prompt: Input prompt
            **kwargs: Override generation parameters
            
        Returns:
            Generated text
        """
        try:
            # Get parameters with overrides using base class method
            gen_params = self.get_generation_params(**kwargs)
            
            # Create message
            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=gen_params.get("max_tokens"),
                temperature=gen_params.get("temperature"),
                system=self.SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract text
            text = message.content[0].text.strip()
            
            return text
            
        except Exception as e:
            self.handle_api_error("Anthropic generation", e)
            raise
    
    def validate_api_key(self) -> bool:
        """
        Validate Anthropic API key.
        
        Returns:
            True if valid, False otherwise
        """
        try:
            # Try a simple API call
            self.client.messages.create(
                model=self.model_name,
                max_tokens=10,
                messages=[{"role": "user", "content": "Hello"}]
            )
            return True
        except Exception as e:
            logger.error(f"Invalid Anthropic API key: {e}")
            return False
