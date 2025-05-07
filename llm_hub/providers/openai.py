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
    ResponseFormatType,
    ReasoningConfig,
    Role,
)
from .base import BaseProvider


class OpenAIProvider(BaseProvider):
    """
    Provider implementation for OpenAI
    """
    DEFAULT_MODEL = "gpt-4o"
    
    # Feature support flags
    SUPPORTED_FEATURES = {
        "function_calling": True,
        "vision": True,
        "streaming": True,
        "structured_output": True,
        "json_schema": True,
        "file_upload": True,
        "reasoning": True,
        "prompt_caching": True,
    }
    
    # Default models
    VISION_MODELS = ["gpt-4-vision-preview", "gpt-4o", "gpt-4o-2024-05-13", "gpt-4.1", "gpt-4.1-mini"]
    
    # Models that support function calling
    FUNCTION_CALLING_MODELS = [
        "gpt-3.5-turbo-0125",
        "gpt-3.5-turbo-1106",
        "gpt-4-0613",
        "gpt-4-turbo-preview",
        "gpt-4o",
        "gpt-4o-2024-05-13",
        "gpt-4.1",
        "gpt-4.1-mini",
    ]
    
    # Models that support structured output with JSON schema
    STRUCTURED_OUTPUT_MODELS = [
        "gpt-4o",
        "gpt-4o-2024-05-13",
        "gpt-4o-mini",
        "gpt-4o-mini-2024-07-18",
        "gpt-4.1",
        "gpt-4.1-mini"
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
        "gpt-4.1": (0.01, 0.03),
        "gpt-4.1-mini": (0.005, 0.015),
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
    
    def _convert_messages_to_api_format(self, instructions: Optional[str], messages: List[Message]) -> List[Dict[str, Any]]:
        """
        Convert LLMHub messages to OpenAI API format
        
        Args:
            instructions: System instructions
            messages: LLMHub messages
            
        Returns:
            List of OpenAI API-formatted messages
        """
        api_messages = []
        
        # Add system message if instructions provided
        if instructions:
            api_messages.append({
                "role": "system",
                "content": [{"type": "input_text", "text": instructions}]
            })
        
        # Convert LLMHub messages to API format
        for message in messages:
            # Start with basic message structure
            api_message = {
                "role": message.role.value,
                "content": []
            }
            
            # Handle tool messages separately
            if message.role == Role.TOOL:
                if hasattr(message, "tool_call_id") and message.tool_call_id:
                    # Handle tool message with result format
                    api_message["type"] = "function_call_output"
                    api_message["call_id"] = message.tool_call_id
                    api_message["output"] = message.content
                    api_messages.append(api_message)
                continue
            
            # Handle assistant messages with tool calls
            if message.role == Role.ASSISTANT and hasattr(message, "tool_calls") and message.tool_calls:
                # Add each tool call to API message list
                for tool_call in message.tool_calls:
                    api_messages.append({
                        "type": "function_call",
                        "call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "arguments": json.dumps(tool_call.function.arguments)
                    })
                
                # If there's no text content, continue to next message
                if message.content is None or message.content == "":
                    continue
            
            # Prepare content items
            if isinstance(message.content, str):
                # String content
                api_message["content"].append({
                    "type": "input_text", 
                    "text": message.content
                })
            elif message.content is None:
                # Skip messages with no content
                continue
            else:
                # Process content items list
                for content_item in message.content:
                    if content_item.type == ContentType.TEXT:
                        # Text content
                        api_message["content"].append({
                            "type": "input_text",
                            "text": content_item.text
                        })
                    
                    elif content_item.type == ContentType.IMAGE:
                        # Image content
                        image_data = {}
                        
                        # Use URL if provided
                        if hasattr(content_item, "image_url") and content_item.image_url:
                            image_data["image_url"] = str(content_item.image_url)
                        
                        # Use file path if provided
                        elif hasattr(content_item, "image_path") and content_item.image_path:
                            # Read image file and convert to base64
                            try:
                                with open(content_item.image_path, "rb") as f:
                                    image_bytes = f.read()
                                    base64_image = base64.b64encode(image_bytes).decode("utf-8")
                                    image_data["image_url"] = f"data:image/jpeg;base64,{base64_image}"
                            except Exception as e:
                                raise LLMHubError(f"Failed to read image file: {str(e)}")
                        
                        # Set image detail if specified
                        if hasattr(content_item, "detail") and content_item.detail:
                            image_data["detail"] = content_item.detail
                        
                        api_message["content"].append({
                            "type": "input_image",
                            **image_data
                        })
                    
                    elif content_item.type == ContentType.FILE:
                        # File content (e.g., PDF)
                        if hasattr(content_item, "file_path") and content_item.file_path:
                            api_message["content"].append({
                                "type": "input_file",
                                "file_path": content_item.file_path
                            })
            
            # Add message to input list if it has content
            if api_message["content"]:
                api_messages.append(api_message)
        
        return api_messages
    
    def _convert_tools_to_api_format(self, tools: Optional[List[Tool]]) -> Optional[List[Dict[str, Any]]]:
        """
        Convert LLMHub tools to OpenAI API format
        
        Args:
            tools: LLMHub tools
            
        Returns:
            List of OpenAI API-formatted tools
        """
        if not tools:
            return None
        
        openai_tools = []
        
        for tool in tools:
            if isinstance(tool, dict) and "__root__" in tool:
                tool = tool["__root__"]
            elif hasattr(tool, "root"):
                tool = tool.root
            
            if tool.type == "function":
                # Create standard OpenAI tool format
                tool_def = {
                    "type": "function",
                    "function": {
                        "name": tool.function.name,
                        "description": tool.function.description,
                        "parameters": {
                            "type": tool.function.parameters.type,
                            "properties": tool.function.parameters.properties,
                        }
                    }
                }
                
                # Add required fields if present
                if tool.function.parameters.required:
                    tool_def["function"]["parameters"]["required"] = tool.function.parameters.required
                
                # Add strict mode and additionalProperties: false for better schema enforcement
                if "additionalProperties" not in tool_def["function"]["parameters"]:
                    tool_def["function"]["parameters"]["additionalProperties"] = False
                
                # Add strict mode unless explicitly set to false in the parameters
                tool_def["strict"] = True
                
                openai_tools.append(tool_def)
        
        return openai_tools
    
    def _extract_tool_calls_from_response(self, response_items):
        """
        Extract tool calls from OpenAI API response
        
        Args:
            response_items: Response items from OpenAI API
            
        Returns:
            List of LLMHub ToolCall objects
        """
        tool_calls = []
        
        for item in response_items:
            if item.get("type") == "function_call":
                # Parse arguments
                try:
                    args = json.loads(item["arguments"])
                except (json.JSONDecodeError, KeyError):
                    args = {"raw_arguments": item.get("arguments", "")}
                
                # Create tool call
                tool_calls.append(
                    ToolCall(
                        id=item.get("id", "") or item.get("call_id", ""),
                        type="function",
                        function={
                            "name": item.get("name", ""),
                            "arguments": args
                        }
                    )
                )
        
        return tool_calls
    
    def _create_response_from_api(self, openai_response, stream=False) -> GenerateResponse:
        """
        Convert OpenAI API response to LLMHub format
        
        Args:
            openai_response: Response from OpenAI API
            stream: Whether this is a streaming response
            
        Returns:
            LLMHub response object
        """
        # Extract text content if present
        content = ""
        if hasattr(openai_response, "output_text") and openai_response.output_text:
            content = openai_response.output_text
        
        # Create response message
        response_message = ResponseMessage(
            role=Role.ASSISTANT,
            content=content
        )
        
        # Add tool calls if present
        if hasattr(openai_response, "output") and openai_response.output:
            tool_calls = self._extract_tool_calls_from_response(openai_response.output)
            if tool_calls:
                response_message.tool_calls = tool_calls
        
        # Create metadata - usage info if available, otherwise use placeholders
        usage_info = UsageInfo(
            prompt_tokens=getattr(openai_response, "usage_prompt_tokens", 0),
            completion_tokens=getattr(openai_response, "usage_completion_tokens", 0),
            total_tokens=getattr(openai_response, "usage_total_tokens", 0)
        )
        
        # Calculate cost if usage information is available
        if usage_info.total_tokens > 0:
            cost_info = CostInfo(**self.calculate_cost(
                prompt_tokens=usage_info.prompt_tokens,
                completion_tokens=usage_info.completion_tokens,
                model=openai_response.model
            ))
        else:
            cost_info = CostInfo(prompt_cost=0, completion_cost=0, total_cost=0)
        
        # Create metadata
        metadata = ResponseMetadata(
            model=openai_response.model,
            provider="openai",
            usage=usage_info,
            cost=cost_info,
            request_id=getattr(openai_response, "id", None)
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
        accumulated_tool_calls = {}
        
        for chunk in response_stream:
            # Extract content from the chunk
            content = ""
            
            # Handle different event types
            if hasattr(chunk, "type"):
                event_type = chunk.type
                
                # Handle function call events
                if event_type == "response.output_item.added" and chunk.item.get("type") == "function_call":
                    # New function call initialization
                    output_index = chunk.output_index
                    accumulated_tool_calls[output_index] = {
                        "id": chunk.item.get("id", ""),
                        "call_id": chunk.item.get("call_id", ""),
                        "name": chunk.item.get("name", ""),
                        "arguments": ""
                    }
                    continue
                
                # Handle argument deltas
                elif event_type == "response.function_call_arguments.delta":
                    output_index = chunk.output_index
                    if output_index in accumulated_tool_calls:
                        accumulated_tool_calls[output_index]["arguments"] += chunk.delta
                    continue
            
            # Handle regular text content
            if hasattr(chunk, "delta") and chunk.delta and hasattr(chunk.delta, "output_text"):
                content = chunk.delta.output_text
                accumulated_text += content
            
            # Create metadata on the first chunk
            if metadata is None:
                metadata = ResponseMetadata(
                    model=getattr(chunk, "model", "unknown"),
                    provider="openai"
                )
            
            # Check if this is the last chunk
            finished = hasattr(chunk, "done") and chunk.done
            
            yield StreamingResponse(
                chunk=content,
                finished=finished,
                metadata=metadata if finished else None
            )
    
    def _convert_output_format_to_api_format(self, output_format: Optional[ResponseFormat]) -> Optional[Dict[str, Any]]:
        """
        Convert LLMHub output format to OpenAI API format
        
        Args:
            output_format: LLMHub output format specification
            
        Returns:
            OpenAI API-formatted response_format
        """
        if output_format is None:
            return None
            
        # Handle simple JSON mode (legacy support)
        if output_format.type == ResponseFormatType.JSON_OBJECT:
            return {"type": "json_object"}
            
        # Handle structured output with schema
        elif output_format.type == ResponseFormatType.JSON_SCHEMA:
            if not self.model in self.STRUCTURED_OUTPUT_MODELS:
                raise ConfigurationError(
                    f"Model {self.model} does not support structured outputs with JSON schema. "
                    f"Use one of: {', '.join(self.STRUCTURED_OUTPUT_MODELS)}"
                )
                
            # Construct the format specification for structured outputs
            format_spec = {
                "type": "json_schema",
                "schema": output_format.schema,
                "strict": output_format.strict
            }
            
            # Add optional fields if provided
            if output_format.name:
                format_spec["name"] = output_format.name
                
            if output_format.description:
                format_spec["description"] = output_format.description
                
            return format_spec
            
        # Handle other format types
        elif output_format.type == ResponseFormatType.TEXT:
            return {"type": "text"}
            
        return None
    
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
            reasoning: Configuration for model reasoning
            metadata: Additional metadata for tracing and logging
            **kwargs: Additional OpenAI-specific parameters
            
        Returns:
            Either a complete response or a streaming response iterator
        """
        try:
            # Validate output format
            if output_format:
                self._validate_output_format(output_format)
            
            # Convert messages to API format
            api_messages = self._convert_messages_to_api_format(instructions, messages)
            
            # Prepare the request parameters
            request_params = {
                "model": self.model,
                "input": api_messages,
                "timeout": timeout,
                "stream": stream,
            }
            
            # Add tools if provided
            if tools:
                request_params["tools"] = self._convert_tools_to_api_format(tools)
                if tool_choice != "auto":
                    request_params["tool_choice"] = tool_choice
                
                # Add parallel tool calls parameter if needed
                if parallel_tool_calls:
                    request_params["parallel_tool_calls"] = True
            
            # Add structured output format if specified
            if output_format:
                response_format = self._convert_output_format_to_api_format(output_format)
                if response_format:
                    if "schema" in response_format and response_format["type"] == "json_schema":
                        # Use the new format structure for text param in newer models
                        request_params["text"] = {"format": response_format}
                    else:
                        # Use the older response_format param for backward compatibility
                        request_params["response_format"] = response_format
            
            # Add reasoning configuration if specified
            if reasoning:
                request_params["reasoning"] = {}
                
                if hasattr(reasoning, "effort") and reasoning.effort:
                    request_params["reasoning"]["effort"] = reasoning.effort.value
                
                if hasattr(reasoning, "summary") and reasoning.summary:
                    request_params["reasoning"]["summary"] = reasoning.summary.value
                
                if hasattr(reasoning, "max_tokens") and reasoning.max_tokens:
                    request_params["reasoning"]["max_tokens"] = reasoning.max_tokens
            
            # Add additional parameters
            request_params.update(kwargs)
            
            # Make the API call
            if stream:
                response_stream = self.client.responses.streaming.create(**request_params)
                return self._create_streaming_response(response_stream)
            else:
                response = self.client.responses.create(**request_params)
                return self._create_response_from_api(response)
            
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
        accumulated_tool_calls = {}
        
        async for chunk in response_stream:
            # Extract content from the chunk
            content = ""
            
            # Handle different event types
            if hasattr(chunk, "type"):
                event_type = chunk.type
                
                # Handle function call events
                if event_type == "response.output_item.added" and chunk.item.get("type") == "function_call":
                    # New function call initialization
                    output_index = chunk.output_index
                    accumulated_tool_calls[output_index] = {
                        "id": chunk.item.get("id", ""),
                        "call_id": chunk.item.get("call_id", ""),
                        "name": chunk.item.get("name", ""),
                        "arguments": ""
                    }
                    continue
                
                # Handle argument deltas
                elif event_type == "response.function_call_arguments.delta":
                    output_index = chunk.output_index
                    if output_index in accumulated_tool_calls:
                        accumulated_tool_calls[output_index]["arguments"] += chunk.delta
                    continue
            
            # Handle regular text content
            if hasattr(chunk, "delta") and chunk.delta and hasattr(chunk.delta, "output_text"):
                content = chunk.delta.output_text
                accumulated_text += content
            
            # Create metadata on the first chunk
            if metadata is None:
                metadata = ResponseMetadata(
                    model=getattr(chunk, "model", "unknown"),
                    provider="openai"
                )
            
            # Check if this is the last chunk
            finished = hasattr(chunk, "done") and chunk.done
            
            yield StreamingResponse(
                chunk=content,
                finished=finished,
                metadata=metadata if finished else None
            )
    
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
            reasoning: Configuration for model reasoning
            metadata: Additional metadata for tracing and logging
            **kwargs: Additional OpenAI-specific parameters
            
        Returns:
            Either a complete response or an async streaming response generator
        """
        try:
            # Validate output format
            if output_format:
                self._validate_output_format(output_format)
            
            # Convert messages to API format
            api_messages = self._convert_messages_to_api_format(instructions, messages)
            
            # Prepare the request parameters
            request_params = {
                "model": self.model,
                "input": api_messages,
                "timeout": timeout,
                "stream": stream,
            }
            
            # Add tools if provided
            if tools:
                request_params["tools"] = self._convert_tools_to_api_format(tools)
                if tool_choice != "auto":
                    request_params["tool_choice"] = tool_choice
                
                # Add parallel tool calls parameter if needed
                if parallel_tool_calls:
                    request_params["parallel_tool_calls"] = True
            
            # Add structured output format if specified
            if output_format:
                response_format = self._convert_output_format_to_api_format(output_format)
                if response_format:
                    if "schema" in response_format and response_format["type"] == "json_schema":
                        # Use the new format structure for text param in newer models
                        request_params["text"] = {"format": response_format}
                    else:
                        # Use the older response_format param for backward compatibility
                        request_params["response_format"] = response_format
            
            # Add reasoning configuration if specified
            if reasoning:
                request_params["reasoning"] = {}
                
                if hasattr(reasoning, "effort") and reasoning.effort:
                    request_params["reasoning"]["effort"] = reasoning.effort.value
                
                if hasattr(reasoning, "summary") and reasoning.summary:
                    request_params["reasoning"]["summary"] = reasoning.summary.value
                
                if hasattr(reasoning, "max_tokens") and reasoning.max_tokens:
                    request_params["reasoning"]["max_tokens"] = reasoning.max_tokens
            
            # Add additional parameters
            request_params.update(kwargs)
            
            # Make the API call
            if stream:
                response_stream = await self.async_client.responses.streaming.create(**request_params)
                return self._acreate_streaming_response(response_stream)
            else:
                response = await self.async_client.responses.create(**request_params)
                return self._create_response_from_api(response)
            
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
            any(isinstance(content, dict) and content.get("type") == ContentType.IMAGE
                for content in (message.content if isinstance(message.content, list) else []))
            for message in request.messages
            if hasattr(message, "content") and message.content is not None
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
            
        # Validate structured output format if present
        if request.output_format:
            self._validate_output_format(request.output_format)