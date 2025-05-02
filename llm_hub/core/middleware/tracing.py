"""
Tracing middleware for LLM Hub
"""

import functools
import json
import logging
import time
import uuid
from typing import Any, Callable, Dict, Optional

from ..core.exceptions import MiddlewareError

# Set up logger
logger = logging.getLogger(__name__)


class TracingMiddleware:
    """
    Middleware for tracing LLM requests and responses
    
    This middleware logs detailed information about each request and response,
    including timing, tokens used, and other metadata.
    """
    
    def __init__(
        self, 
        log_level: int = logging.INFO,
        include_content: bool = True,
        trace_id_generator: Optional[Callable[[], str]] = None
    ):
        """
        Initialize the tracing middleware
        
        Args:
            log_level: Logging level for trace logs
            include_content: Whether to include request/response content in logs
            trace_id_generator: Function to generate trace IDs (defaults to UUID)
        """
        self.log_level = log_level
        self.include_content = include_content
        self.trace_id_generator = trace_id_generator or (lambda: str(uuid.uuid4()))
    
    def wrap(self, func: Callable) -> Callable:
        """
        Wrap a function with tracing
        
        Args:
            func: The function to wrap
            
        Returns:
            Wrapped function
        """
        @functools.wraps(func)
        def wrapped_sync(*args, **kwargs):
            """
            Synchronous wrapper with tracing
            """
            trace_id = self.trace_id_generator()
            start_time = time.time()
            
            # Extract metadata from kwargs if present
            metadata = kwargs.get("metadata", {})
            
            # Add trace_id to metadata
            if metadata is None:
                metadata = {}
            metadata["trace_id"] = trace_id
            kwargs["metadata"] = metadata
            
            # Log request
            self._log_request(trace_id, args, kwargs)
            
            try:
                # Call the wrapped function
                response = func(*args, **kwargs)
                
                # Calculate duration
                duration = time.time() - start_time
                
                # Log response
                self._log_response(trace_id, response, duration)
                
                return response
            
            except Exception as e:
                # Log error
                self._log_error(trace_id, e, time.time() - start_time)
                raise
        
        @functools.wraps(func)
        async def wrapped_async(*args, **kwargs):
            """
            Asynchronous wrapper with tracing
            """
            trace_id = self.trace_id_generator()
            start_time = time.time()
            
            # Extract metadata from kwargs if present
            metadata = kwargs.get("metadata", {})
            
            # Add trace_id to metadata
            if metadata is None:
                metadata = {}
            metadata["trace_id"] = trace_id
            kwargs["metadata"] = metadata
            
            # Log request
            self._log_request(trace_id, args, kwargs)
            
            try:
                # Call the wrapped function
                response = await func(*args, **kwargs)
                
                # Calculate duration
                duration = time.time() - start_time
                
                # Log response
                self._log_response(trace_id, response, duration)
                
                return response
            
            except Exception as e:
                # Log error
                self._log_error(trace_id, e, time.time() - start_time)
                raise
        
        # Check if the wrapped function is async
        if asyncio.iscoroutinefunction(func):
            return wrapped_async
        else:
            return wrapped_sync
    
    def _log_request(self, trace_id: str, args: tuple, kwargs: dict) -> None:
        """
        Log a request
        
        Args:
            trace_id: The trace ID
            args: Positional arguments
            kwargs: Keyword arguments
        """
        try:
            # Extract relevant request data
            log_data = {
                "trace_id": trace_id,
                "timestamp": time.time(),
                "event_type": "llm_request",
            }
            
            # Add metadata if present
            if "metadata" in kwargs and kwargs["metadata"]:
                log_data["metadata"] = kwargs["metadata"]
            
            # Add content if enabled
            if self.include_content:
                # Add filtered request data
                filtered_kwargs = kwargs.copy()
                
                # Remove API keys
                if "api_key" in filtered_kwargs:
                    filtered_kwargs["api_key"] = "***"
                
                log_data["request"] = {
                    "args": [str(arg) for arg in args],
                    "kwargs": self._sanitize_dict(filtered_kwargs)
                }
            
            # Log the data
            logger.log(self.log_level, json.dumps(log_data))
            
        except Exception as e:
            logger.warning(f"Error logging request: {str(e)}")
    
    def _log_response(self, trace_id: str, response: Any, duration: float) -> None:
        """
        Log a response
        
        Args:
            trace_id: The trace ID
            response: The response object
            duration: Request duration in seconds
        """
        try:
            # Extract relevant response data
            log_data = {
                "trace_id": trace_id,
                "timestamp": time.time(),
                "event_type": "llm_response",
                "duration": duration,
            }
            
            # Extract metadata from response if present
            if hasattr(response, "metadata"):
                log_data["model"] = getattr(response.metadata, "model", "unknown")
                log_data["provider"] = getattr(response.metadata, "provider", "unknown")
                
                # Add usage information if present
                if hasattr(response.metadata, "usage"):
                    log_data["usage"] = {
                        "prompt_tokens": getattr(response.metadata.usage, "prompt_tokens", 0),
                        "completion_tokens": getattr(response.metadata.usage, "completion_tokens", 0),
                        "total_tokens": getattr(response.metadata.usage, "total_tokens", 0),
                    }
                
                # Add cost information if present
                if hasattr(response.metadata, "cost"):
                    log_data["cost"] = {
                        "prompt_cost": getattr(response.metadata.cost, "prompt_cost", 0),
                        "completion_cost": getattr(response.metadata.cost, "completion_cost", 0),
                        "total_cost": getattr(response.metadata.cost, "total_cost", 0),
                    }
            
            # Add content if enabled
            if self.include_content:
                # Check if it's a streaming response
                if hasattr(response, "__iter__") or hasattr(response, "__aiter__"):
                    log_data["response_type"] = "stream"
                else:
                    log_data["response_type"] = "complete"
                    
                    # Add response content for non-streaming responses
                    if hasattr(response, "message"):
                        log_data["content"] = getattr(response.message, "content", "")
                        
                        # Add tool calls if present
                        if hasattr(response.message, "tool_calls") and response.message.tool_calls:
                            log_data["tool_calls"] = [
                                {
                                    "type": tool_call.type,
                                    "function": {
                                        "name": tool_call.function.name,
                                        "arguments": tool_call.function.arguments
                                    }
                                }
                                for tool_call in response.message.tool_calls
                            ]
            
            # Log the data
            logger.log(self.log_level, json.dumps(log_data))
            
        except Exception as e:
            logger.warning(f"Error logging response: {str(e)}")
    
    def _log_error(self, trace_id: str, error: Exception, duration: float) -> None:
        """
        Log an error
        
        Args:
            trace_id: The trace ID
            error: The error that occurred
            duration: Request duration in seconds
        """
        try:
            # Extract relevant error data
            log_data = {
                "trace_id": trace_id,
                "timestamp": time.time(),
                "event_type": "llm_error",
                "duration": duration,
                "error_type": type(error).__name__,
                "error_message": str(error)
            }
            
            # Add more details for LLMHub errors
            if hasattr(error, "provider"):
                log_data["provider"] = getattr(error, "provider", "unknown")
            
            if hasattr(error, "details"):
                log_data["error_details"] = getattr(error, "details", {})
            
            # Log the data
            logger.log(self.log_level, json.dumps(log_data))
            
        except Exception as e:
            logger.warning(f"Error logging error: {str(e)}")
    
    def _sanitize_dict(self, d: Dict) -> Dict:
        """
        Sanitize a dictionary for logging
        
        Args:
            d: Dictionary to sanitize
            
        Returns:
            Sanitized dictionary
        """
        result = {}
        
        for k, v in d.items():
            if k in ["api_key", "password", "secret", "token"]:
                result[k] = "***"
            elif isinstance(v, dict):
                result[k] = self._sanitize_dict(v)
            elif isinstance(v, list):
                result[k] = [
                    self._sanitize_dict(item) if isinstance(item, dict) else item
                    for item in v
                ]
            else:
                result[k] = v
        
        return result