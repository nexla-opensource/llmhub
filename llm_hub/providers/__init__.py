"""
Provider registry for LLM Hub
"""

from typing import Dict, Type

from .base import BaseProvider
from .openai import OpenAIProvider
from .anthropic import AnthropicProvider
from .gemini import GeminiProvider

# Registry of available providers
PROVIDERS: Dict[str, Type[BaseProvider]] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
}


def register_provider(name: str, provider_class: Type[BaseProvider]) -> None:
    """
    Register a new provider with LLM Hub
    
    Args:
        name: Name of the provider
        provider_class: Provider implementation class
    """
    PROVIDERS[name] = provider_class


def get_provider_class(name: str) -> Type[BaseProvider]:
    """
    Get a provider implementation class by name
    
    Args:
        name: Name of the provider
        
    Returns:
        The provider implementation class
        
    Raises:
        KeyError: If the provider is not found
    """
    if name not in PROVIDERS:
        raise KeyError(f"Provider '{name}' not found. Available providers: {', '.join(PROVIDERS.keys())}")
    
    return PROVIDERS[name]


def list_available_providers() -> Dict[str, Type[BaseProvider]]:
    """
    Get a dictionary of all available providers
    
    Returns:
        Dictionary of provider names to provider classes
    """
    return PROVIDERS.copy()