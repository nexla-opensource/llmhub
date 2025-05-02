"""
OpenAI provider implementation for LLM Hub
"""

import asyncio
import base64
import json
import os
from typing import Any, Dict, List, Optional, Union, AsyncGenerator, Iterator

from openai import OpenAI, AsyncOpenAI, OpenAIError
from pydantic import ValidationError

from ..core.exceptions import (
    AuthenticationError,
    ConfigurationError,
    LLMHubError,
    ProviderError,
    RateLimitError,
    TimeoutError,
    TokenLimitError,
)
from ..core.types import (
    ContentType,
    GenerateRequest,
    GenerateResponse,
    StreamingResponse,
    Message,
    MessageContent,
    ResponseMessage,
    ResponseMetadata,
    Tool,
    ToolCall,
    UsageInfo,
    CostInfo,
    ResponseFormat,
    ReasoningConfig,
    Role,
)
from .base import BaseProvider


class OpenAIProvider(BaseProvider):
    """
    Provider implementation for OpenAI
    """
    
    # Feature support flags
    SUPPORTED_FEATURES = {
        "function_calling": True,
        "vision": True,
        "streaming": True,
        "structured_output": True,
        "file_upload": True,
        "reasoning": False,  # OpenAI doesn't have an explicit reasoning parameter
    }
    
    # Default models
    DEFAULT_MODEL = "gpt-4o"
    VISION_MODELS = ["gpt-4-vision-preview", "gpt-4o", "gpt-4o-2024-05-13"]
    
    # Models that support function calling
    FUNCTION_CALLING_MODELS = [
        "gpt-3.5-turbo-0125",
        "gpt-3.5-turbo-1106",
        "gpt-4-0613",
        "gpt-4-turbo-preview",
        "gpt-4o",
        "gpt-4o-2024-05-13",
    ]
    
    # Token pricing (per 1K tokens as of May 2023)
    # Format: {model_name: (prompt_price, completion_price)}
    TOKEN_PRICING = {
        "gpt-3.5-turbo": (0.0015, 0.002),
        "gpt-3.5-turbo-0125": (0.0015, 0.002),
        "gpt-3.5-turbo-1106": (0.0015, 0.002),
        "gpt-3.5-turbo-instruct": (0.0015, 0.002),
        "gpt-4": (0.03, 0.06),
        "gpt-4-32k": (0.06, 0.12),
        "gpt-4-0613": (0.03, 0.06),
        "gpt-4-32k-0613": (0.06, 0.12),
        "gpt-4-turbo-preview": (0.01, 0.03),
        "gpt-4o": (0.01, 0.03),
        "gpt-4o-2024-05-13": (0.01, 0.03),
        "gpt-4-vision-preview": (0.01, 0.03),
    }
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the OpenAI provider
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Model to use (defaults to DEFAULT_MODEL)
            base_url: Base URL for API requests (for Azure OpenAI)
            organization: OpenAI organization ID
            **kwargs: Additional options
        """
        # Use provided API key or get from environment
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ConfigurationError("OpenAI API key not provided and OPENAI_API_KEY environment variable not set")
        
        super().__init__(api_key, **kwargs)
        
        # Set default model if not provided
        self.model = model or self.DEFAULT_MODEL
        
        # Initialize clients
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            **kwargs
        )
        
        self.async_client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            **kwargs
        )
    
    @property
    def provider_name(self) -> str:
        return "openai"
    
    def supports_feature(self, feature: str) -> bool:
        return self.SUPPORTED_FEATURES.get(feature, False)
    
    def _convert_messages(self, instructions: Optional[str], messages: List[Message]) -> List[Dict[str, Any]]:
        """
        Convert LLMHub messages to OpenAI format
        
        Args:
            instructions: System instructions
            messages: LLMHub messages
            
        Returns:
            List of OpenAI-formatted messages
        """
        openai_messages = []
        
        # Add system message if instructions provided
        if instructions:
            openai_messages.append({
                "role": "system",
                "content": instructions
            })
        
        # Convert LLMHub messages to OpenAI format
        for message in messages:
            # Start with basic message structure
            openai_message = {
                "role": message.role.value,
            }
            
            # Handle message content
            content_items = []
            
            for content_item in message.content:
                if content_item.type == ContentType.TEXT:
                    # Text content
                    content_items.append({
                        "type": "text",
                        "text": content_item.text
                    })
                
                elif content_item.type == ContentType.IMAGE:
                    # Image content
                    image_data = None
                    
                    # Use URL if provided
                    if hasattr(content_item, "image_url") and content_item.image_url:
                        image_data = {
                            "url": str(content_item.image_url),
                        }
                    
                    # Use file path if provided
                    elif hasattr(content_item, "image_path") and content_item.image_path:
                        # Read image file and convert to base64
                        try:
                            with open(content_item.image_path, "rb") as f:
                                image_bytes = f.read()
                                base64_image = base64.b64encode(image_bytes).decode("utf-8")
                                image_data = {
                                    "url": f"data:image/png;base64,{base64_image}",
                                }
                        except Exception as e:
                            raise LLMHubError(f"Failed to read image file: {str(e)}")
                    
                    # Set image detail if specified
                    if hasattr(content_item, "detail") and content_item.detail:
                        image_data["detail"] = content_item.detail
                    
                    content_items.append({
                        "type": "image_url",
                        "image_url": image_data
                    })
                
                # Document content not supported natively by OpenAI API
                # It would be handled via file upload APIs separately
            
            # Set appropriate message content format
            if len(content_items) == 1 and content_items[0]["type"] == "text":
                # Use string content for text-only messages
                openai_message["content"] = content_items[0]["text"]
            else:
                # Use array content for multi-modal messages
                openai_message["content"] = content_items
            
            openai_messages.append(openai_message)
        
        return openai_messages
    
    def _convert_tools(self, tools: Optional[List[Tool]]) -> List[Dict[str, Any]]:
        """
        Convert LLMHub tools to OpenAI format
        
        Args:
            tools: LLMHub tools
            
        Returns:
            List of OpenAI-formatted tools
        """
        if not tools:
            return []
        
        openai_tools = []
        
        for tool in tools:
            if isinstance(tool, dict) and "__root__" in tool:
                tool = tool["__root__"]
            elif hasattr(tool, "root"):
                tool = tool.root
            
            if tool.type == "function":
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.function.name,
                        "description": tool.function.description,
                        "parameters": {
                            "type": tool.function.parameters.type,
                            "properties": tool.function.parameters.properties,
                            "required": tool.function.parameters.required or [],
                        }
                    }
                })
        
        return openai_tools
    
    def _convert_tool_calls(self, tool_calls: List[Any]) -> List[ToolCall]:
        """
        Convert OpenAI tool calls to LLMHub format
        
        Args:
            tool_calls: OpenAI tool calls
            
        Returns:
            List of LLMHub ToolCall objects
        """
        llm_hub_tool_calls = []
        
        for call in tool_calls:
            try:
                arguments = json.loads(call.function.arguments)
            except json.JSONDecodeError:
                arguments = {"raw_arguments": call.function.arguments}
            
            llm_hub_tool_calls.append(
                ToolCall(
                    id=call.id,
                    type="function",
                    function={
                        "name": call.function.name,
                        "arguments": arguments
                    }
                )
            )
        
        return llm_hub_tool_calls
    
    def _create_response(self, openai_response, stream=False) -> Union[GenerateResponse, Iterator[StreamingResponse]]:
        """
        Convert OpenAI response to LLMHub format
        
        Args:
            openai_response: Response from OpenAI API
            stream: Whether this is a streaming response
            
        Returns:
            LLMHub response object
        """
        if stream:
            return self._create_streaming_response(openai_response)
        
        # Extract useful response data
        message = openai_response.choices[0].message
        usage = openai_response.usage
        
        # Create response message
        response_message = ResponseMessage(
            role=Role.ASSISTANT,
            content=message.content or ""
        )
        
        # Add tool calls if present
        if hasattr(message, "tool_calls") and message.tool_calls:
            response_message.tool_calls = self._convert_tool_calls(message.tool_calls)
        
        # Create usage info
        usage_info = UsageInfo(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens
        )
        
        # Calculate cost
        cost_info = CostInfo(**self.calculate_cost(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            model=openai_response.model
        ))
        
        # Create metadata
        metadata = ResponseMetadata(
            model=openai_response.model,
            provider="openai",
            usage=usage_info,
            cost=cost_info,
            request_id=openai_response.id
        )
        
        # Create and return response
        return GenerateResponse(
            message=response_message,
            metadata=metadata
        )
    
    def _create_streaming_response(self, response_stream) -> Iterator[StreamingResponse]:
        """
        Convert OpenAI streaming response to LLMHub format
        
        Args:
            response_stream: Streaming response from OpenAI API
            
        Returns:
            Iterator of LLMHub StreamingResponse objects
        """
        accumulated_text = ""
        metadata = None
        
        for chunk in response_stream:
            delta = chunk.choices[0].delta
            
            # Extract content from the delta
            content = ""
            if hasattr(delta, "content") and delta.content:
                content = delta.content
                accumulated_text += content
            
            # Create metadata on the first chunk
            if metadata is None:
                metadata = ResponseMetadata(
                    model=chunk.model,
                    provider="openai",
                    request_id=chunk.id
                )
            
            # Check if this is the last chunk
            finished = chunk.choices[0].finish_reason is not None
            
            yield StreamingResponse(
                chunk=content,
                finished=finished,
                metadata=metadata if finished else None
            )
    
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
        Generate a response from OpenAI
        
        Args:
            instructions: System instructions for the model
            messages: List of messages in the conversation
            tools: List of tools available to the model
            tool_choice: How the model should choose tools
            parallel_tool_calls: Whether to allow parallel tool calls
            output_format: Format specification for the output
            timeout: Request timeout in seconds
            stream: Whether to stream the response
            reasoning: Configuration for model reasoning (not used for OpenAI)
            metadata: Additional metadata for tracing and logging
            **kwargs: Additional OpenAI-specific parameters
            
        Returns:
            Either a complete response or a streaming response iterator
        """
        try:
            # Convert messages to OpenAI format
            openai_messages = self._convert_messages(instructions, messages)
            
            # Convert tools to OpenAI format
            openai_tools = None
            if tools:
                openai_tools = self._convert_tools(tools)
            
            # Prepare the request parameters
            request_params = {
                "model": self.model,
                "messages": openai_messages,
                "timeout": timeout,
                "stream": stream,
            }
            
            # Add tools if provided
            if openai_tools:
                request_params["tools"] = openai_tools
                request_params["tool_choice"] = tool_choice
                
                # Add parallel tool calls parameter
                if parallel_tool_calls:
                    request_params["parallel_tool_calls"] = True
            
            # Add structured output format if specified
            if output_format and output_format.type == "json_object":
                request_params["response_format"] = {"type": "json_object"}
            
            # Add additional parameters
            request_params.update(kwargs)
            
            # Make the API call
            response = self.client.chat.completions.create(**request_params)
            
            # Convert and return the response
            return self._create_response(response, stream)
            
        except OpenAIError as e:
            # Map OpenAI errors to LLMHub exceptions
            error_msg = str(e)
            
            if "authentication" in error_msg.lower() or "api key" in error_msg.lower():
                raise AuthenticationError(error_msg, provider=self.provider_name)
            
            elif "rate limit" in error_msg.lower():
                raise RateLimitError(error_msg, provider=self.provider_name)
            
            elif "timeout" in error_msg.lower():
                raise TimeoutError(error_msg, provider=self.provider_name)
            
            elif "context length" in error_msg.lower() or "token" in error_msg.lower():
                raise TokenLimitError(error_msg, provider=self.provider_name)
            
            else:
                raise ProviderError(error_msg, provider=self.provider_name)
        
        except Exception as e:
            # Handle other errors
            raise LLMHubError(f"Unexpected error: {str(e)}", provider=self.provider_name)
    
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
        Asynchronously generate a response from OpenAI
        
        Args:
            instructions: System instructions for the model
            messages: List of messages in the conversation
            tools: List of tools available to the model
            tool_choice: How the model should choose tools
            parallel_tool_calls: Whether to allow parallel tool calls
            output_format: Format specification for the output
            timeout: Request timeout in seconds
            stream: Whether to stream the response
            reasoning: Configuration for model reasoning (not used for OpenAI)
            metadata: Additional metadata for tracing and logging
            **kwargs: Additional OpenAI-specific parameters
            
        Returns:
            Either a complete response or an async streaming response generator
        """
        try:
            # Convert messages to OpenAI format
            openai_messages = self._convert_messages(instructions, messages)
            
            # Convert tools to OpenAI format
            openai_tools = None
            if tools:
                openai_tools = self._convert_tools(tools)
            
            # Prepare the request parameters
            request_params = {
                "model": self.model,
                "messages": openai_messages,
                "timeout": timeout,
                "stream": stream,
            }
            
            # Add tools if provided
            if openai_tools:
                request_params["tools"] = openai_tools
                request_params["tool_choice"] = tool_choice
                
                # Add parallel tool calls parameter
                if parallel_tool_calls:
                    request_params["parallel_tool_calls"] = True
            
            # Add structured output format if specified
            if output_format and output_format.type == "json_object":
                request_params["response_format"] = {"type": "json_object"}
            
            # Add additional parameters
            request_params.update(kwargs)
            
            # Make the API call
            response = await self.async_client.chat.completions.create(**request_params)
            
            if not stream:
                # Convert and return the response
                return self._create_response(response)
            else:
                # Return the streaming response
                return self._acreate_streaming_response(response)
            
        except OpenAIError as e:
            # Map OpenAI errors to LLMHub exceptions
            error_msg = str(e)
            
            if "authentication" in error_msg.lower() or "api key" in error_msg.lower():
                raise AuthenticationError(error_msg, provider=self.provider_name)
            
            elif "rate limit" in error_msg.lower():
                raise RateLimitError(error_msg, provider=self.provider_name)
            
            elif "timeout" in error_msg.lower():
                raise TimeoutError(error_msg, provider=self.provider_name)
            
            elif "context length" in error_msg.lower() or "token" in error_msg.lower():
                raise TokenLimitError(error_msg, provider=self.provider_name)
            
            else:
                raise ProviderError(error_msg, provider=self.provider_name)
        
        except Exception as e:
            # Handle other errors
            raise LLMHubError(f"Unexpected error: {str(e)}", provider=self.provider_name)
    
    async def _acreate_streaming_response(self, response_stream) -> AsyncGenerator[StreamingResponse, None]:
        """
        Convert OpenAI async streaming response to LLMHub format
        
        Args:
            response_stream: Async streaming response from OpenAI API
            
        Returns:
            Async generator of LLMHub StreamingResponse objects
        """
        accumulated_text = ""
        metadata = None
        
        async for chunk in response_stream:
            delta = chunk.choices[0].delta
            
            # Extract content from the delta
            content = ""
            if hasattr(delta, "content") and delta.content:
                content = delta.content
                accumulated_text += content
            
            # Create metadata on the first chunk
            if metadata is None:
                metadata = ResponseMetadata(
                    model=chunk.model,
                    provider="openai",
                    request_id=chunk.id
                )
            
            # Check if this is the last chunk
            finished = chunk.choices[0].finish_reason is not None
            
            yield StreamingResponse(
                chunk=content,
                finished=finished,
                metadata=metadata if finished else None
            )
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get a list of available models from OpenAI
        
        Returns:
            List of model information dictionaries
        """
        try:
            models = self.client.models.list()
            
            # Filter for chat models only and format the response
            chat_models = []
            for model in models.data:
                if "gpt" in model.id.lower():
                    model_info = {
                        "id": model.id,
                        "provider": self.provider_name,
                        "capabilities": {
                            "vision": model.id in self.VISION_MODELS,
                            "function_calling": model.id in self.FUNCTION_CALLING_MODELS,
                        }
                    }
                    chat_models.append(model_info)
            
            return chat_models
            
        except Exception as e:
            raise LLMHubError(f"Failed to get available models: {str(e)}", provider=self.provider_name)
    
    def calculate_cost(self, prompt_tokens: int, completion_tokens: int, model: str) -> Dict[str, float]:
        """
        Calculate the cost of a request/response
        
        Args:
            prompt_tokens: Number of tokens in the prompt
            completion_tokens: Number of tokens in the completion
            model: The model used
            
        Returns:
            Dictionary with prompt_cost, completion_cost, and total_cost
        """
        # Default to gpt-4 pricing if model not found
        prompt_price, completion_price = self.TOKEN_PRICING.get(
            model, self.TOKEN_PRICING.get("gpt-4", (0.03, 0.06))
        )
        
        prompt_cost = (prompt_tokens / 1000) * prompt_price
        completion_cost = (completion_tokens / 1000) * completion_price
        total_cost = prompt_cost + completion_cost
        
        return {
            "prompt_cost": prompt_cost,
            "completion_cost": completion_cost,
            "total_cost": total_cost
        }
    
    def validate_request(self, request: GenerateRequest) -> None:
        """
        Validate a request against this provider's capabilities
        
        Args:
            request: The request to validate
            
        Raises:
            LLMHubError: If the request contains unsupported features
        """
        model = getattr(self, "model", self.DEFAULT_MODEL)
        
        # Check for vision capabilities if images are present
        has_images = any(
            any(content.type == ContentType.IMAGE for content in message.content)
            for message in request.messages
        )
        
        if has_images and model not in self.VISION_MODELS:
            raise LLMHubError(
                f"Model '{model}' does not support vision features",
                provider=self.provider_name
            )
        
        # Check for function calling support
        if request.tools and model not in self.FUNCTION_CALLING_MODELS:
            raise LLMHubError(
                f"Model '{model}' does not support function calling",
                provider=self.provider_name
            )
        
        # Validate reasoning config (OpenAI doesn't support explicit reasoning)
        if request.reasoning:
            raise LLMHubError(
                f"Model '{model}' does not support explicit reasoning configuration",
                provider=self.provider_name
            )