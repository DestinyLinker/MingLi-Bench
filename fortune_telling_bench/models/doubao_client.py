"""
ByteDance Doubao (豆包) model client implementation.
"""

import os
from typing import Optional, Dict, Any
import requests

from .base import ModelClient
from ..utils.logger import get_logger
from ..utils.decorators import retry_on_error

logger = get_logger(__name__)


class DoubaoClient(ModelClient):
    """Client for ByteDance Doubao models"""
    
    # API key environment variables
    API_KEY_ENV_VARS = ["DOUBAO_API_KEY", "BYTEDANCE_API_KEY", "ARK_API_KEY"]
    
    # Default base URL for Doubao
    DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
    
    def __init__(self, 
                 model_name: str = "doubao-pro",
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 endpoint_id: Optional[str] = None,
                 **kwargs):
        """
        Initialize Doubao client.
        
        Args:
            model_name: Model to use (e.g., "doubao-pro", "doubao-lite")
            api_key: Doubao API key
            base_url: API base URL
            endpoint_id: Endpoint ID for the model
            **kwargs: Additional configuration
        """
        # Initialize parent with common functionality
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            api_key_env_vars=self.API_KEY_ENV_VARS,
            **kwargs
        )
        
        # Set base URL and endpoint
        self.base_url = base_url or os.getenv("DOUBAO_BASE_URL", self.DEFAULT_BASE_URL)
        self.endpoint_id = endpoint_id or os.getenv("DOUBAO_ENDPOINT_ID")
        
        if not self.endpoint_id:
            raise ValueError(
                "Doubao endpoint ID not found. "
                "Set DOUBAO_ENDPOINT_ID environment variable or pass endpoint_id parameter."
            )
        
        # Set headers
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    @retry_on_error(max_retries=3, delay=1.0)
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate response using Doubao API.
        
        Args:
            prompt: Input prompt
            **kwargs: Override generation parameters
            
        Returns:
            Generated text
        """
        try:
            # Get parameters
            params = self.get_generation_params(**kwargs)
            
            # Build request
            url = f"{self.base_url}/chat/completions"
            data = {
                "model": self.endpoint_id,  # Doubao uses endpoint_id instead of model name
                "messages": [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                "temperature": params.get("temperature", self.temperature),
                "max_tokens": params.get("max_tokens", self.max_tokens),
            }
            
            # Make API call
            response = requests.post(url, headers=self.headers, json=data)
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            
            # Extract text
            if "choices" in result and result["choices"]:
                return result["choices"][0]["message"]["content"].strip()
            else:
                raise ValueError(f"Unexpected response format: {result}")
                
        except requests.RequestException as e:
            self.handle_api_error("Doubao API request", e)
            raise
        except Exception as e:
            self.handle_api_error("Doubao generation", e)
            raise
    
    def validate_api_key(self) -> bool:
        """
        Validate Doubao API key.
        
        Returns:
            True if valid, False otherwise
        """
        try:
            # Try a simple API call
            url = f"{self.base_url}/models"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Invalid Doubao API key: {e}")
            return False
