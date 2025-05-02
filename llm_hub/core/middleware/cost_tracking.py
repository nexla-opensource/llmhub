"""
Cost tracking middleware for LLM Hub
"""

import asyncio
import functools
import logging
from typing import Any, Callable, Dict, Optional

# Set up logger
logger = logging.getLogger(__name__)


class CostTrackingMiddleware:
    """
    Middleware for tracking costs of LLM requests
    
    This middleware aggregates cost information from responses and
    provides methods to get total costs and usage statistics.
    """
    
    def __init__(
        self,
        log_level: int = logging.INFO,
        cost_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        """
        Initialize the cost tracking middleware
        
        Args:
            log_level: Logging level for cost logs
            cost_callback: Optional callback function for cost tracking
        """
        self.log_level = log_level
        self.cost_callback = cost_callback
        
        # Initialize tracking counters
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_cost = 0.0
        self.prompt_cost = 0.0
        self.completion_cost = 0.0
        
        # Track costs by model
        self.costs_by_model: Dict[str, Dict[str, float]] = {}
        self.tokens_by_model: Dict[str, Dict[str, int]] = {}
    
    def wrap(self, func: Callable) -> Callable:
        """
        Wrap a function with cost tracking
        
        Args:
            func: The function to wrap
            
        Returns:
            Wrapped function
        """
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            """
            Wrapper with cost tracking
            """
            # Call the wrapped function
            response = func(*args, **kwargs)
            
            # Track costs for non-streaming responses
            if not kwargs.get("stream", False):
                self._track_costs(response)
            
            return response
        
        @functools.wraps(func)
        async def wrapped_async(*args, **kwargs):
            """
            Async wrapper with cost tracking
            """
            # Call the wrapped function
            response = await func(*args, **kwargs)
            
            # Track costs for non-streaming responses
            if not kwargs.get("stream", False):
                self._track_costs(response)
            
            return response
        
        # Check if the wrapped function is async
        if asyncio.iscoroutinefunction(func):
            return wrapped_async
        else:
            return wrapped
    
    def _track_costs(self, response: Any) -> None:
        """
        Track costs from a response
        
        Args:
            response: The response to extract cost information from
        """
        # Check if response has metadata
        if not hasattr(response, "metadata"):
            return
        
        # Check if metadata has usage and cost information
        if not (hasattr(response.metadata, "usage") and hasattr(response.metadata, "cost")):
            return
        
        # Extract model name
        model = getattr(response.metadata, "model", "unknown")
        provider = getattr(response.metadata, "provider", "unknown")
        model_key = f"{provider}/{model}"
        
        # Extract usage information
        usage = response.metadata.usage
        prompt_tokens = getattr(usage, "prompt_tokens", 0)
        completion_tokens = getattr(usage, "completion_tokens", 0)
        total_tokens = getattr(usage, "total_tokens", 0)
        
        # Extract cost information
        cost = response.metadata.cost
        prompt_cost = getattr(cost, "prompt_cost", 0.0)
        completion_cost = getattr(cost, "completion_cost", 0.0)
        total_cost = getattr(cost, "total_cost", 0.0)
        
        # Update global counters
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += total_tokens
        self.prompt_cost += prompt_cost
        self.completion_cost += completion_cost
        self.total_cost += total_cost
        
        # Update model-specific counters
        if model_key not in self.costs_by_model:
            self.costs_by_model[model_key] = {
                "prompt_cost": 0.0,
                "completion_cost": 0.0,
                "total_cost": 0.0
            }
            self.tokens_by_model[model_key] = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        
        self.costs_by_model[model_key]["prompt_cost"] += prompt_cost
        self.costs_by_model[model_key]["completion_cost"] += completion_cost
        self.costs_by_model[model_key]["total_cost"] += total_cost
        
        self.tokens_by_model[model_key]["prompt_tokens"] += prompt_tokens
        self.tokens_by_model[model_key]["completion_tokens"] += completion_tokens
        self.tokens_by_model[model_key]["total_tokens"] += total_tokens
        
        # Log cost information
        logger.log(
            self.log_level,
            f"Cost: ${total_cost:.6f} ({model_key}) - "
            f"Tokens: {total_tokens} ({prompt_tokens} prompt, {completion_tokens} completion)"
        )
        
        # Call the callback if provided
        if self.cost_callback:
            cost_data = {
                "model": model,
                "provider": provider,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens
                },
                "cost": {
                    "prompt_cost": prompt_cost,
                    "completion_cost": completion_cost,
                    "total_cost": total_cost
                }
            }
            self.cost_callback(cost_data)
    
    def get_total_cost(self) -> float:
        """
        Get the total cost tracked so far
        
        Returns:
            Total cost in USD
        """
        return self.total_cost
    
    def get_total_tokens(self) -> int:
        """
        Get the total number of tokens used
        
        Returns:
            Total token count
        """
        return self.total_tokens
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all tracked costs
        
        Returns:
            Dictionary with cost and usage statistics
        """
        return {
            "overall": {
                "usage": {
                    "prompt_tokens": self.prompt_tokens,
                    "completion_tokens": self.completion_tokens,
                    "total_tokens": self.total_tokens
                },
                "cost": {
                    "prompt_cost": self.prompt_cost,
                    "completion_cost": self.completion_cost,
                    "total_cost": self.total_cost
                }
            },
            "by_model": {
                model: {
                    "usage": self.tokens_by_model[model],
                    "cost": self.costs_by_model[model]
                }
                for model in self.costs_by_model
            }
        }
    
    def reset_tracking(self) -> None:
        """
        Reset all cost tracking counters
        """
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_cost = 0.0
        self.prompt_cost = 0.0
        self.completion_cost = 0.0
        self.costs_by_model = {}
        self.tokens_by_model = {}
