"""
Retry middleware for LLM Hub
"""

import asyncio
import functools
import logging
import random
import time
from typing import Any, Callable, List, Optional, Set, Type, Union

from ..core.exceptions import (
    AuthenticationError,
    LLMHubError,
    RateLimitError,
    TimeoutError,
)

# Set up logger
logger = logging.getLogger(__name__)


class RetryMiddleware:
    """
    Middleware for automatic retries of failed LLM requests
    
    This middleware will automatically retry requests that fail due to
    rate limits, timeouts, or other transient errors.
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: Optional[Set[Type[Exception]]] = None,
        non_retryable_exceptions: Optional[Set[Type[Exception]]] = None,
    ):
        """
        Initialize the retry middleware
        
        Args:
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay between retries in seconds
            max_delay: Maximum delay between retries in seconds
            backoff_factor: Exponential backoff multiplier
            jitter: Whether to add random jitter to delay times
            retryable_exceptions: Set of exceptions that should trigger a retry
            non_retryable_exceptions: Set of exceptions that should never trigger a retry
        """
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter
        
        # Default retryable exceptions if not provided
        self.retryable_exceptions = retryable_exceptions or {
            TimeoutError,
            RateLimitError,
            ConnectionError,
            ConnectionRefusedError,
            ConnectionResetError,
        }
        
        # Default non-retryable exceptions if not provided
        self.non_retryable_exceptions = non_retryable_exceptions or {
            AuthenticationError,
        }
    
    def wrap(self, func: Callable) -> Callable:
        """
        Wrap a function with retry logic
        
        Args:
            func: The function to wrap
            
        Returns:
            Wrapped function
        """
        @functools.wraps(func)
        def wrapped_sync(*args, **kwargs):
            """
            Synchronous wrapper with retry logic
            """
            # Get max_retries from kwargs if provided, otherwise use the instance value
            max_retries = kwargs.pop("max_retries", None)
            if max_retries is None:
                max_retries = self.max_retries
            
            # Retry loop
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # Check if this is the last attempt
                    if attempt >= max_retries:
                        logger.warning(f"Max retries ({max_retries}) reached, giving up")
                        raise
                    
                    # Check if the exception is retryable
                    if not self._is_retryable(e):
                        logger.info(f"Non-retryable exception: {type(e).__name__}, giving up")
                        raise
                    
                    # Calculate retry delay
                    delay = self._calculate_delay(attempt)
                    
                    # Log the retry
                    logger.info(
                        f"Retry {attempt + 1}/{max_retries} after exception: "
                        f"{type(e).__name__}: {str(e)}. Waiting {delay:.2f}s"
                    )
                    
                    # Wait before retrying
                    time.sleep(delay)
        
        @functools.wraps(func)
        async def wrapped_async(*args, **kwargs):
            """
            Asynchronous wrapper with retry logic
            """
            # Get max_retries from kwargs if provided, otherwise use the instance value
            max_retries = kwargs.pop("max_retries", None)
            if max_retries is None:
                max_retries = self.max_retries
            
            # Retry loop
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    # Check if this is the last attempt
                    if attempt >= max_retries:
                        logger.warning(f"Max retries ({max_retries}) reached, giving up")
                        raise
                    
                    # Check if the exception is retryable
                    if not self._is_retryable(e):
                        logger.info(f"Non-retryable exception: {type(e).__name__}, giving up")
                        raise
                    
                    # Calculate retry delay
                    delay = self._calculate_delay(attempt)
                    
                    # Use retry-after header value if available and it's a RateLimitError
                    if isinstance(e, RateLimitError) and e.retry_after is not None:
                        delay = max(delay, e.retry_after)
                    
                    # Log the retry
                    logger.info(
                        f"Retry {attempt + 1}/{max_retries} after exception: "
                        f"{type(e).__name__}: {str(e)}. Waiting {delay:.2f}s"
                    )
                    
                    # Wait before retrying
                    await asyncio.sleep(delay)
        
        # Check if the wrapped function is async
        if asyncio.iscoroutinefunction(func):
            return wrapped_async
        else:
            return wrapped_sync
    
    def _is_retryable(self, exception: Exception) -> bool:
        """
        Determine if an exception should trigger a retry
        
        Args:
            exception: The exception to check
            
        Returns:
            Whether the exception is retryable
        """
        # Check if it's in the non-retryable exceptions
        for exc_type in self.non_retryable_exceptions:
            if isinstance(exception, exc_type):
                return False
        
        # Check if it's in the retryable exceptions
        for exc_type in self.retryable_exceptions:
            if isinstance(exception, exc_type):
                return True
        
        # Default to not retryable
        return False
    
    def _calculate_delay(self, attempt: int) -> float:
        """
        Calculate the delay for a retry attempt
        
        Args:
            attempt: The current attempt number (0-indexed)
            
        Returns:
            Delay time in seconds
        """
        # Calculate exponential backoff
        delay = min(
            self.max_delay,
            self.initial_delay * (self.backoff_factor ** attempt)
        )
        
        # Add jitter if enabled
        if self.jitter:
            jitter_multiplier = random.uniform(0.5, 1.5)
            delay *= jitter_multiplier
        
        return delay