"""
Main LLMHub class that provides the unified interface for all LLM providers
"""

import importlib
import time
from typing import Any, Dict, List, Optional, Union, Iterator, AsyncGenerator

from ..core.types import (
    GenerateRequest, 
    GenerateResponse, 
    LLMHubConfig,
    Provider,
    Message,
    ResponseFormat,
    ReasoningConfig,
    StreamingResponse,
    Tool
)
from ..core.exceptions import ConfigurationError, LLMHubError
from ..providers.base import BaseProvider
from ..middleware.tracing import TracingMiddleware
from ..middleware.cost_tracking import CostTrackingMiddleware
from ..middleware.retry import RetryMiddleware


class LLMHub:
    """
    Unified interface for interacting with various LLM providers
    """
    
    def __init__(
        self,
        provider: Union[str, Provider],
        api_key: str,
        model: Optional[str] = None,
        tracing: bool = False,
        cost_tracking: bool = False,
        retries: int = 0,
        timeout: int = 60,
        metadata: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        """
        Initialize the LLM Hub
        
        Args:
            provider: The LLM provider to use
            api_key: API key for the provider
            model: Specific model to use (defaults to provider's default)
            tracing: Whether to enable request tracing
            cost_tracking: Whether to track usage costs
            retries: Number of retries for failed requests
            timeout: Default timeout in seconds
            metadata: Additional metadata for logging and tracing
            **kwargs: Additional provider-specific options
        """
        config = LLMHubConfig(
            provider=Provider(provider) if isinstance(provider, str) else provider,
            api_key=api_key,
            model=model,
            tracing=tracing,
            cost_tracking=cost_tracking,
            retries=retries,
            timeout=timeout,
            metadata=metadata or {},
        )
        
        # Initialize the provider
        self.provider = self._initialize_provider(config, **kwargs)
        
        # Initialize middleware stack
        self.middleware = []
        
        # Add middleware based on configuration
        if config.tracing:
            self.middleware.append(TracingMiddleware())
        
        if config.cost_tracking:
            self.middleware.append(CostTrackingMiddleware())
        
        if config.retries > 0:
            self.middleware.append(RetryMiddleware(max_retries=config.retries))
        
        self.config = config
    
    def _initialize_provider(self, config: LLMHubConfig, **kwargs) -> BaseProvider:
        """
        Initialize the appropriate provider based on configuration
        
        Args:
            config: LLMHub configuration
            **kwargs: Additional provider-specific options
            
        Returns:
            Initialized provider instance
            
        Raises:
            ConfigurationError: If the provider could not be initialized
        """
        try:
            # Dynamically import the provider module
            provider_module = importlib.import_module(f"..providers.{config.provider.value.lower()}", package=__name__)
            
            # Get the provider class (convention: provider name + "Provider")
            provider_class_name = f"{config.provider.value.capitalize()}Provider"
            provider_class = getattr(provider_module, provider_class_name)
            
            # Initialize the provider with the API key and options
            provider_options = {**kwargs}
            if config.model:
                provider_options["model"] = config.model
                
            return provider_class(api_key=config.api_key, **provider_options)
        
        except (ImportError, AttributeError) as e:
            raise ConfigurationError(
                f"Failed to initialize provider '{config.provider}': {str(e)}"
            )
    
    def _apply_middleware(self, func, *args, **kwargs):
        """
        Apply middleware to a provider function call
        
        Args:
            func: The provider function to call
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Function result after passing through middleware
        """
        # Start with the original function
        result_func = func
        
        # Apply middleware in reverse order (so the first middleware is the outermost)
        for middleware in reversed(self.middleware):
            result_func = middleware.wrap(result_func)
        
        # Call the wrapped function
        return result_func(*args, **kwargs)
    
    def generate(
        self,
        instructions: Optional[str] = None,
        messages: List[Union[Dict[str, Any], Message]] = None,
        tools: Optional[List[Union[Dict[str, Any], Tool]]] = None,
        tool_choice: Optional[str] = "auto",
        parallel_tool_calls: bool = False,
        output_format: Optional[Union[Dict[str, Any], ResponseFormat]] = None,
        max_retries: Optional[int] = None,
        timeout: Optional[int] = None,
        stream: bool = False,
        reasoning: Optional[Union[Dict[str, Any], ReasoningConfig]] = None,
        metadata: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Union[GenerateResponse, Iterator[StreamingResponse]]:
        """
        Generate a response from the LLM
        
        Args:
            instructions: System instructions for the model
            messages: List of messages in the conversation
            tools: List of tools available to the model
            tool_choice: How the model should choose tools
            parallel_tool_calls: Whether to allow parallel tool calls
            output_format: Format specification for the output
            max_retries: Maximum number of retries (overrides instance setting)
            timeout: Request timeout in seconds (overrides instance setting)
            stream: Whether to stream the response
            reasoning: Configuration for model reasoning
            metadata: Additional metadata for tracing and logging
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Either a complete response or a streaming response iterator
        """
        # Convert dictionary inputs to their proper types if needed
        if messages and isinstance(messages[0], dict):
            messages = [Message(**msg) if isinstance(msg, dict) else msg for msg in messages]
        
        if tools and isinstance(tools[0], dict):
            tools = [Tool(**tool) if isinstance(tool, dict) else tool for tool in tools]
        
        if output_format and isinstance(output_format, dict):
            output_format = ResponseFormat(**output_format)
        
        if reasoning and isinstance(reasoning, dict):
            reasoning = ReasoningConfig(**reasoning)
        
        # Merge request metadata with instance metadata
        request_metadata = {**self.config.metadata}
        if metadata:
            request_metadata.update(metadata)
        
        # Create request for validation
        request = GenerateRequest(
            instructions=instructions,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            output_format=output_format,
            max_retries=max_retries if max_retries is not None else self.config.retries,
            timeout=timeout if timeout is not None else self.config.timeout,
            stream=stream,
            reasoning=reasoning,
            metadata=request_metadata,
        )
        
        # Validate the request against the provider's capabilities
        self.provider.validate_request(request)
        
        # Record start time for latency tracking
        start_time = time.time()
        
        # Call the provider's generate method through middleware
        try:
            result = self._apply_middleware(
                self.provider.generate,
                instructions=instructions,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                parallel_tool_calls=parallel_tool_calls,
                output_format=output_format,
                timeout=timeout if timeout is not None else self.config.timeout,
                stream=stream,
                reasoning=reasoning, 
                metadata=request_metadata,
                **kwargs
            )
            
            # For non-streaming responses, add latency information
            if not stream and hasattr(result, "metadata"):
                result.metadata.latency = time.time() - start_time
                
            return result
            
        except Exception as e:
            # Re-raise any exceptions
            if isinstance(e, LLMHubError):
                raise
            else:
                raise LLMHubError(f"Unexpected error: {str(e)}", provider=self.provider.provider_name)
    
    async def agenerate(
        self,
        instructions: Optional[str] = None,
        messages: List[Union[Dict[str, Any], Message]] = None,
        tools: Optional[List[Union[Dict[str, Any], Tool]]] = None,
        tool_choice: Optional[str] = "auto",
        parallel_tool_calls: bool = False,
        output_format: Optional[Union[Dict[str, Any], ResponseFormat]] = None,
        max_retries: Optional[int] = None,
        timeout: Optional[int] = None,
        stream: bool = False,
        reasoning: Optional[Union[Dict[str, Any], ReasoningConfig]] = None,
        metadata: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Union[GenerateResponse, AsyncGenerator[StreamingResponse, None]]:
        """
        Asynchronously generate a response from the LLM
        
        Args:
            instructions: System instructions for the model
            messages: List of messages in the conversation
            tools: List of tools available to the model
            tool_choice: How the model should choose tools
            parallel_tool_calls: Whether to allow parallel tool calls
            output_format: Format specification for the output
            max_retries: Maximum number of retries (overrides instance setting)
            timeout: Request timeout in seconds (overrides instance setting)
            stream: Whether to stream the response
            reasoning: Configuration for model reasoning
            metadata: Additional metadata for tracing and logging
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Either a complete response or an async streaming response generator
        """
        # Convert dictionary inputs to their proper types if needed
        if messages and isinstance(messages[0], dict):
            messages = [Message(**msg) if isinstance(msg, dict) else msg for msg in messages]
        
        if tools and isinstance(tools[0], dict):
            tools = [Tool(**tool) if isinstance(tool, dict) else tool for tool in tools]
        
        if output_format and isinstance(output_format, dict):
            output_format = ResponseFormat(**output_format)
        
        if reasoning and isinstance(reasoning, dict):
            reasoning = ReasoningConfig(**reasoning)
        
        # Merge request metadata with instance metadata
        request_metadata = {**self.config.metadata}
        if metadata:
            request_metadata.update(metadata)
        
        # Create request for validation
        request = GenerateRequest(
            instructions=instructions,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            output_format=output_format,
            max_retries=max_retries if max_retries is not None else self.config.retries,
            timeout=timeout if timeout is not None else self.config.timeout,
            stream=stream,
            reasoning=reasoning,
            metadata=request_metadata,
        )
        
        # Validate the request against the provider's capabilities
        self.provider.validate_request(request)
        
        # Record start time for latency tracking
        start_time = time.time()
        
        # Call the provider's agenerate method through middleware
        try:
            result = await self._apply_middleware(
                self.provider.agenerate,
                instructions=instructions,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                parallel_tool_calls=parallel_tool_calls,
                output_format=output_format,
                timeout=timeout if timeout is not None else self.config.timeout,
                stream=stream,
                reasoning=reasoning,
                metadata=request_metadata,
                **kwargs
            )
            
            # For non-streaming responses, add latency information
            if not stream and hasattr(result, "metadata"):
                result.metadata.latency = time.time() - start_time
                
            return result
            
        except Exception as e:
            # Re-raise any exceptions
            if isinstance(e, LLMHubError):
                raise
            else:
                raise LLMHubError(f"Unexpected error: {str(e)}", provider=self.provider.provider_name)
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get a list of available models from the current provider
        
        Returns:
            List of model information dictionaries
        """
        return self.provider.get_available_models()
    
    def supports_feature(self, feature: str) -> bool:
        """
        Check if the current provider supports a specific feature
        
        Args:
            feature: Feature name
            
        Returns:
            Boolean indicating feature support
        """
        return self.provider.supports_feature(feature)
