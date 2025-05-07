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
    ResponseFormatType,
    Tool,
    ReasoningConfig,
    Message
)
from ..core.exceptions import LLMHubError, ConfigurationError


class BaseProvider(abc.ABC):
    """
    Abstract base class for all LLM providers
    
    All provider implementations must inherit from this class and implement
    the required methods.
    """
    
    # Every provider should define this with supported model names
    STRUCTURED_OUTPUT_MODELS = []
    
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
    
    def _validate_structured_output_request(self, model: str, output_format: Optional[ResponseFormat]) -> None:
        """
        Validate if a structured output request is valid for the given model
        
        Args:
            model: The model name
            output_format: The output format specification
            
        Raises:
            ConfigurationError: If the model does not support structured outputs
        """
        if not output_format:
            return
            
        # Check for JSON schema support
        if output_format.type == ResponseFormatType.JSON_SCHEMA:
            if not self.STRUCTURED_OUTPUT_MODELS or model not in self.STRUCTURED_OUTPUT_MODELS:
                supported_models = ', '.join(self.STRUCTURED_OUTPUT_MODELS) if self.STRUCTURED_OUTPUT_MODELS else 'None'
                raise ConfigurationError(
                    f"Model '{model}' does not support structured outputs with JSON schema. "
                    f"Supported models for {self.provider_name}: {supported_models}"
                )
            
            # Validate schema requirements
            if not output_format.schema:
                raise ConfigurationError("JSON schema is required for structured outputs")
                
            schema = output_format.schema
            
            # Schema must be an object
            if schema.get("type") != "object":
                raise ConfigurationError("JSON schema root must be an object type")
                
            # Check for required fields
            if "required" not in schema:
                raise ConfigurationError("JSON schema must specify required fields")
                
            # Check for additionalProperties: false
            if schema.get("additionalProperties") is not False:
                raise ConfigurationError("JSON schema must set additionalProperties to false")
    
    def _validate_output_format(self, output_format: Optional[ResponseFormat]) -> None:
        """
        Validate the output format specification
        
        Args:
            output_format: The output format specification
            
        Raises:
            ConfigurationError: If the output format is invalid
        """
        if not output_format:
            return
            
        # Validate output format type
        if not hasattr(output_format, "type") or not output_format.type:
            raise ConfigurationError("Output format must specify a type")
            
        # If it's a JSON schema, validate the schema
        if output_format.type == ResponseFormatType.JSON_SCHEMA:
            self._validate_structured_output_request(getattr(self, "model", ""), output_format)
    
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