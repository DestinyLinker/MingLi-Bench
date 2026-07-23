"""
Atlas Cloud model client implementation.
"""

import os
from typing import Optional

from openai import OpenAI

from .base import ModelClient
from ..utils.logger import get_logger

logger = get_logger(__name__)


class AtlasCloudClient(ModelClient):
    """Client for Atlas Cloud models through the OpenAI-compatible API."""

    API_KEY_ENV_VARS = ["ATLASCLOUD_API_KEY", "ATLAS_CLOUD_API_KEY"]
    BASE_URL_ENV_VARS = [
        "ATLASCLOUD_API_BASE",
        "ATLASCLOUD_BASE_URL",
        "ATLAS_CLOUD_API_BASE",
        "ATLAS_CLOUD_BASE_URL",
    ]
    DEFAULT_BASE_URL = "https://api.atlascloud.ai/v1"

    def __init__(
        self,
        model_name: str = "qwen/qwen3.5-flash",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize Atlas Cloud client.

        Args:
            model_name: Atlas Cloud model id, e.g. "qwen/qwen3.5-flash"
            api_key: Atlas Cloud API key
            base_url: OpenAI-compatible base URL
            **kwargs: Additional configuration
        """
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            api_key_env_vars=self.API_KEY_ENV_VARS,
            **kwargs,
        )

        timeout_seconds = int(os.getenv("TIMEOUT", "30"))
        self.base_url = base_url or self._get_base_url()
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=timeout_seconds,
        )

        logger.info(
            "Initialized Atlas Cloud client with model: %s, base_url: %s, timeout: %ss",
            model_name,
            self.base_url,
            timeout_seconds,
        )

    def _get_base_url(self) -> str:
        """Resolve Atlas Cloud base URL from aliases or default."""
        for env_var in self.BASE_URL_ENV_VARS:
            if value := os.getenv(env_var):
                return value
        return self.DEFAULT_BASE_URL

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate response using Atlas Cloud OpenAI-compatible API.

        Args:
            prompt: Input prompt
            **kwargs: Override generation parameters

        Returns:
            Generated text
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
            self.handle_api_error("Atlas Cloud generation", e)
            raise

    def validate_api_key(self) -> bool:
        """
        Validate Atlas Cloud API key.

        Returns:
            True if valid, False otherwise
        """
        try:
            self.client.models.list()
            return True
        except Exception as e:
            logger.error(f"Invalid Atlas Cloud API key: {e}")
            return False
