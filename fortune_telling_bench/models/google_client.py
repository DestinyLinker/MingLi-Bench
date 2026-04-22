"""
Google Gemini model client implementation.
"""

import os
from typing import Optional
import google.generativeai as genai
from google.api_core import client_options as client_options_lib

from .base import ModelClient
from ..utils.logger import get_logger

logger = get_logger(__name__)


class GoogleClient(ModelClient):
    """Client for Google Gemini models"""
    
    def __init__(self, 
                 model_name: str = "gemini-pro",
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 temperature: float = 0.0,
                 max_tokens: Optional[int] = None,
                 **kwargs):
        """
        Initialize Google client.
        
        Args:
            model_name: Model to use (e.g., "gemini-pro")
            api_key: Google API key (if not set, uses environment variable)
            base_url: Custom API base URL (if not set, uses environment variable)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional configuration
        """
        # Initialize parent with API key handling
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            api_key_env_vars=["GEMINI_API_KEY", "GOOGLE_API_KEY"],
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        # Set base URL if provided
        self.base_url = base_url or os.getenv("GEMINI_BASE_URL") or os.getenv("GOOGLE_BASE_URL")
        
        # Configure API
        if self.base_url:
            client_options = client_options_lib.ClientOptions(api_endpoint=self.base_url)
            genai.configure(api_key=self.api_key, client_options=client_options)
            logger.info(f"Using custom base URL: {self.base_url}")
        else:
            genai.configure(api_key=self.api_key)
        
        # Initialize model
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens,
            },
            system_instruction=self.SYSTEM_PROMPT
        )
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate response using Google API.
        
        Args:
            prompt: Input prompt
            **kwargs: Override generation parameters
            
        Returns:
            Generated text
        """
        try:
            # Get parameters with overrides using base class method
            gen_params = self.get_generation_params(**kwargs)
            
            # Update generation config if needed
            if kwargs:
                generation_config = {
                    "temperature": gen_params.get("temperature"),
                    "max_output_tokens": gen_params.get("max_tokens"),
                }
                response = self.model.generate_content(prompt, generation_config=generation_config)
            else:
                response = self.model.generate_content(prompt)
            
            # Extract text
            text = response.text.strip()
            
            return text
            
        except Exception as e:
            self.handle_api_error("Google generation", e)
            raise
    
    def validate_api_key(self) -> bool:
        """
        Validate Google API key.
        
        Returns:
            True if valid, False otherwise
        """
        try:
            # Try a simple API call
            self.model.generate_content("Hello")
            return True
        except Exception as e:
            logger.error(f"Invalid Google API key: {e}")
            return False
