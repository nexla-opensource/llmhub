"""
Standardized exceptions for LLM Hub
"""

from typing import Any, Dict, Optional


class LLMHubError(Exception):
    """Base exception for all LLM Hub errors"""
    
    def __init__(self, message: str, provider: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.provider = provider
        self.details = details or {}
        super().__init__(self.message)
    
    def __str__(self) -> str:
        error_str = self.message
        if self.provider:
            error_str = f"[{self.provider}] {error_str}"
        if self.details:
            error_str = f"{error_str} - Details: {self.details}"
        return error_str


class ProviderError(LLMHubError):
    """Exception raised for errors returned by the LLM provider"""
    pass


class ConfigurationError(LLMHubError):
    """Exception raised for errors in the configuration"""
    pass


class AuthenticationError(ProviderError):
    """Exception raised for authentication failures"""
    pass


class RateLimitError(ProviderError):
    """Exception raised when rate limits are exceeded"""
    
    def __init__(
        self, 
        message: str, 
        provider: Optional[str] = None, 
        details: Optional[Dict[str, Any]] = None,
        retry_after: Optional[int] = None
    ):
        super().__init__(message, provider, details)
        self.retry_after = retry_after


class TimeoutError(ProviderError):
    """Exception raised when a request times out"""
    pass


class TokenLimitError(ProviderError):
    """Exception raised when token context limits are exceeded"""
    
    def __init__(
        self, 
        message: str, 
        provider: Optional[str] = None, 
        details: Optional[Dict[str, Any]] = None,
        token_count: Optional[int] = None,
        token_limit: Optional[int] = None
    ):
        super().__init__(message, provider, details)
        self.token_count = token_count
        self.token_limit = token_limit


class ContentError(LLMHubError):
    """Exception raised for content-related errors"""
    pass


class ToolError(LLMHubError):
    """Exception raised for tool-related errors"""
    pass


class MiddlewareError(LLMHubError):
    """Exception raised for middleware-related errors"""
    pass


class StreamingError(LLMHubError):
    """Exception raised for streaming-related errors"""
    pass


class FileHandlingError(LLMHubError):
    """Exception raised for file handling errors"""
    pass


class StructuredOutputError(LLMHubError):
    """Exception raised for structured output errors"""
    pass