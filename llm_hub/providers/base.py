"""
Base provider interface for LLM Hub
"""

import abc
from typing import Any, Dict, List, Optional, Union, AsyncGenerator, Iterator

from ..core.types import (
    GenerateRequest,
    GenerateResponse,
    StreamingResponse,
    ToolCall,
    ResponseFormat,
    Tool,
    ReasoningConfig,
    Message
)


class BaseProvider(abc.ABC):
    """
    Abstract base class for all LLM providers
    
    All provider implementations must inherit from this class and implement
    the required methods.
    """
    
    def __init__(self, api_key: str, **kwargs):
        """
        Initialize the provider with the necessary credentials and options
        
        Args:
            api_key: API key for the provider
            **kwargs: Additional provider-specific options
        """
        self.api_key = api_key
        self.options = kwargs
    
    @abc.abstractmethod
    def generate(
        self, 
        instructions: Optional[str],
        messages: List[Message],
        tools: Optional[List[Tool]] = None,
        tool_choice: Optional[str] = "auto",
        parallel_tool_calls: bool = False,
        output_format: Optional[ResponseFormat] = None,
        timeout: Optional[int] = None,
        stream: bool = False,
        reasoning: Optional[ReasoningConfig] = None,
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
            timeout: Request timeout in seconds
            stream: Whether to stream the response
            reasoning: Configuration for model reasoning
            metadata: Additional metadata for tracing and logging
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Either a complete response or a streaming response iterator
        """
        pass
    
    @abc.abstractmethod
    async def agenerate(
        self, 
        instructions: Optional[str],
        messages: List[Message],
        tools: Optional[List[Tool]] = None,
        tool_choice: Optional[str] = "auto",
        parallel_tool_calls: bool = False,
        output_format: Optional[ResponseFormat] = None,
        timeout: Optional[int] = None,
        stream: bool = False,
        reasoning: Optional[ReasoningConfig] = None,
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
            timeout: Request timeout in seconds
            stream: Whether to stream the response
            reasoning: Configuration for model reasoning
            metadata: Additional metadata for tracing and logging
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Either a complete response or an async streaming response generator
        """
        pass
    
    @abc.abstractmethod
    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get a list of available models from this provider
        
        Returns:
            List of model information dictionaries
        """
        pass
    
    @abc.abstractmethod
    def calculate_cost(
        self, 
        prompt_tokens: int, 
        completion_tokens: int, 
        model: str
    ) -> Dict[str, float]:
        """
        Calculate the cost of a request/response
        
        Args:
            prompt_tokens: Number of tokens in the prompt
            completion_tokens: Number of tokens in the completion
            model: The model used
            
        Returns:
            Dictionary with prompt_cost, completion_cost, and total_cost
        """
        pass
    
    @property
    @abc.abstractmethod
    def provider_name(self) -> str:
        """
        Get the name of the provider
        
        Returns:
            Provider name string
        """
        pass
    
    @abc.abstractmethod
    def supports_feature(self, feature: str) -> bool:
        """
        Check if the provider supports a specific feature
        
        Args:
            feature: Feature name
            
        Returns:
            Boolean indicating feature support
        """
        pass
    
    @abc.abstractmethod
    def validate_request(self, request: GenerateRequest) -> None:
        """
        Validate a request against this provider's capabilities
        
        Args:
            request: The request to validate
            
        Raises:
            LLMHubError: If the request contains unsupported features
        """
        pass