"""
Astraflow model client implementation.
Astraflow (by UCloud) is an OpenAI-compatible platform supporting 200+ models.
Global endpoint : https://api-us-ca.umodelverse.ai/v1  (ASTRAFLOW_API_KEY)
China  endpoint : https://api.modelverse.cn/v1          (ASTRAFLOW_CN_API_KEY)
"""

import os
from typing import Optional
from openai import OpenAI

from .base import ModelClient
from ..utils.logger import get_logger

logger = get_logger(__name__)


class AstraflowClient(ModelClient):
    """Client for Astraflow models (OpenAI-compatible API — global endpoint)."""

    # API key environment variables
    API_KEY_ENV_VARS = ["ASTRAFLOW_API_KEY"]

    # Default base URL for Astraflow global endpoint
    DEFAULT_BASE_URL = "https://api-us-ca.umodelverse.ai/v1"

    def __init__(
        self,
        model_name: str = "deepseek-ai/DeepSeek-R1",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize Astraflow client (global endpoint).

        Args:
            model_name: Model to use (any model supported by Astraflow).
            api_key: Astraflow API key (falls back to ASTRAFLOW_API_KEY env var).
            base_url: API base URL (default: global endpoint).
            **kwargs: Additional configuration.
        """
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            api_key_env_vars=self.API_KEY_ENV_VARS,
            **kwargs,
        )

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=base_url or os.getenv("ASTRAFLOW_BASE_URL", self.DEFAULT_BASE_URL),
        )

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate response using the Astraflow API.

        Args:
            prompt: Input prompt.
            **kwargs: Override generation parameters.

        Returns:
            Generated text.
        """
        try:
            params = self.get_generation_params(**kwargs)

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                **params,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            self.handle_api_error("Astraflow generation", e)
            raise

    def validate_api_key(self) -> bool:
        """Validate Astraflow API key."""
        try:
            self.client.models.list()
            return True
        except Exception as e:
            logger.error(f"Invalid Astraflow API key: {e}")
            return False


class AstraflowCNClient(AstraflowClient):
    """Client for Astraflow models (OpenAI-compatible API — China endpoint)."""

    API_KEY_ENV_VARS = ["ASTRAFLOW_CN_API_KEY"]
    DEFAULT_BASE_URL = "https://api.modelverse.cn/v1"

    def __init__(
        self,
        model_name: str = "deepseek-ai/DeepSeek-R1",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize Astraflow client (China endpoint).

        Args:
            model_name: Model to use.
            api_key: Astraflow CN API key (falls back to ASTRAFLOW_CN_API_KEY env var).
            base_url: API base URL (default: China endpoint).
            **kwargs: Additional configuration.
        """
        # Call grandparent (ModelClient) directly so we can set our own env vars
        # before the OpenAI client is constructed.
        ModelClient.__init__(
            self,
            model_name=model_name,
            api_key=api_key,
            api_key_env_vars=self.API_KEY_ENV_VARS,
            **kwargs,
        )

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=base_url or os.getenv("ASTRAFLOW_CN_BASE_URL", self.DEFAULT_BASE_URL),
        )
