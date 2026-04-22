"""
Model factory for creating LLM clients.
"""

from typing import Dict, Any, Type, Optional
import re

from .base import ModelClient
from .openai_client import OpenAIClient
from .anthropic_client import AnthropicClient
from .google_client import GoogleClient
from .deepseek_client import DeepSeekClient
from .doubao_client import DoubaoClient
from ..utils.logger import get_logger
from ..utils.config import load_config

logger = get_logger(__name__)


class ModelFactory:
    """Factory for creating model clients."""

    # Provider to client class mapping
    _registry = {
        'openai': OpenAIClient,
        'anthropic': AnthropicClient,
        'google': GoogleClient,
        'deepseek': DeepSeekClient,
        'doubao': DoubaoClient,
        'openrouter': OpenAIClient,  # OpenRouter uses OpenAI-compatible API
    }
    
    @classmethod
    def register_provider(cls, provider: str, client_class: Type[ModelClient]):
        """
        Register a new provider.

        Args:
            provider: Provider name
            client_class: Client class
        """
        cls._registry[provider] = client_class
    
    @classmethod
    def get_provider(cls, model_name: str) -> Optional[str]:
        """
        Determine provider from model name.

        Logic:
        - Models with '/' prefix (e.g., openai/gpt-4, anthropic/claude-3) → openrouter
        - Models without prefix → native API (gpt-4 → openai, claude-3 → anthropic)

        Args:
            model_name: Name of the model

        Returns:
            Provider name or None
        """
        # OpenRouter models (with prefix)
        if '/' in model_name:
            prefix = model_name.split('/')[0].lower()
            # Special case: bytedance prefix for Doubao native API
            if prefix == 'bytedance':
                return 'doubao'
            # All other prefixed models go through OpenRouter
            return 'openrouter'

        # Native API models (without prefix)
        if model_name.startswith('gpt-') or model_name.startswith('o1-') or model_name.startswith('o3-') or model_name.startswith('o4-'):
            return 'openai'
        if model_name.startswith('claude-'):
            return 'anthropic'
        if model_name.startswith('gemini-'):
            return 'google'
        if model_name.startswith('deepseek-'):
            return 'deepseek'
        if model_name.startswith('doubao-'):
            return 'doubao'

        return None
    
    @classmethod
    def create(cls,
               model_name: str,
               provider: Optional[str] = None,
               config: Optional[Dict[str, Any]] = None,
               **kwargs) -> ModelClient:
        """
        Create a model client.

        Args:
            model_name: Name of the model (e.g., 'gpt-4', 'openai/gpt-4', 'claude-3-sonnet')
            provider: Optional provider override
            config: Optional configuration dictionary
            **kwargs: Additional parameters

        Returns:
            ModelClient instance

        Raises:
            ValueError: If provider cannot be determined or is not supported
        """
        # Load configuration
        if config is None:
            config = load_config()

        logger.debug(f"ModelFactory.create called with model_name: '{model_name}'")

        # Determine provider from model name
        if not provider:
            provider = cls.get_provider(model_name)

        if not provider:
            # Raise error for unknown models instead of defaulting to OpenAI
            raise ValueError(
                f"Cannot determine provider for model '{model_name}'. "
                f"Supported patterns: gpt-*, o1-*, o3-*, o4-*, claude-*, gemini-*, deepseek-*, doubao-*, "
                f"or use OpenRouter format: provider/model-name (e.g., openai/gpt-4, nvidia/llama-3)"
            )

        logger.info(f"Determined provider: {provider} for model: {model_name}")

        # Get provider configuration
        provider_config = config.get(provider, {})

        # Get client class
        if provider not in cls._registry:
            logger.warning(f"Unknown provider: {provider}, defaulting to OpenAI client")
            client_class = cls._registry['openai']
        else:
            client_class = cls._registry[provider]

        # Build merged configuration
        merged_config = {
            'model_name': model_name,
            'api_key': provider_config.get('api_key'),
            'base_url': provider_config.get('base_url'),
            'temperature': provider_config.get('temperature'),
            'max_tokens': provider_config.get('max_tokens'),
        }

        # Update with kwargs (highest priority)
        merged_config.update(kwargs)

        # Remove None values
        merged_config = {k: v for k, v in merged_config.items() if v is not None}

        logger.info(f"Creating {client_class.__name__} for model: {model_name}")

        # Create and return client
        return client_class(**merged_config)
    
    @classmethod
    def list_providers(cls) -> list:
        """Get list of available providers."""
        return list(cls._registry.keys())
    
    @classmethod
    def list_supported_models(cls) -> Dict[str, list]:
        """Get supported models by provider."""
        return {
            'openai': ['gpt-4', 'gpt-4-turbo', 'gpt-3.5-turbo', 'o1-preview', 'o1-mini'],
            'anthropic': ['claude-3-opus', 'claude-3-sonnet', 'claude-3-haiku', 'claude-3-5-sonnet'],
            'google': ['gemini-pro', 'gemini-1.5-pro', 'gemini-1.5-flash'],
            'deepseek': ['deepseek-chat', 'deepseek-coder'],
            'doubao': ['doubao-pro', 'doubao-lite'],
            'openrouter': [
                'openai/gpt-4', 'anthropic/claude-3-sonnet', 'google/gemini-2.0-flash',
                'x-ai/grok-4', 'moonshotai/kimi-k2', 'deepseek/deepseek-r1'
            ],
        }