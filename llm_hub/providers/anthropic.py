"""
Complete Anthropic (Claude) provider implementation for LLM Hub
"""

import asyncio
import base64
import json
import os
from typing import Any, Dict, List, Optional, Union, AsyncGenerator, Iterator

import anthropic
from anthropic import Anthropic, AsyncAnthropic

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


class AnthropicProvider(BaseProvider):
    """
    Provider implementation for Anthropic (Claude)
    """
    
    # Feature support flags
    SUPPORTED_FEATURES = {
        "function_calling": True,
        "vision": True,
        "streaming": True,
        "structured_output": True,
        "file_upload": True,
        "reasoning": True,
    }
    
    # Default models
    DEFAULT_MODEL = "claude-3-sonnet-20240229"
    VISION_MODELS = ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"]
    
    # Models that support function calling
    FUNCTION_CALLING_MODELS = ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"]
    
    # Token pricing (per 1K tokens as of May 2024)
    # Format: {model_name: (prompt_price, completion_price)}
    TOKEN_PRICING = {
        "claude-3-opus-20240229": (0.015, 0.075),
        "claude-3-sonnet-20240229": (0.003, 0.015),
        "claude-3-haiku-20240307": (0.00025, 0.00125),
        "claude-2.1": (0.008, 0.024),
        "claude-2.0": (0.008, 0.024),
        "claude-instant-1.2": (0.0008, 0.0024),
    }
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the Anthropic provider
        
        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            model: Model to use (defaults to DEFAULT_MODEL)
            base_url: Custom API base URL (optional)
            **kwargs: Additional options
        """
        # Use provided API key or get from environment
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ConfigurationError("Anthropic API key not provided and ANTHROPIC_API_KEY environment variable not set")
        
        super().__init__(api_key, **kwargs)
        
        # Set default model if not provided
        self.model = model or self.DEFAULT_MODEL
        
        # Initialize clients
        client_params = {"api_key": api_key}
        if base_url:
            client_params["base_url"] = base_url
            
        # Add additional parameters
        for key, value in kwargs.items():
            if key in ["timeout", "max_retries"]:
                client_params[key] = value
        
        self.client = Anthropic(**client_params)
        self.async_client = AsyncAnthropic(**client_params)
    
    @property
    def provider_name(self) -> str:
        return "anthropic"
    
    def supports_feature(self, feature: str) -> bool:
        return self.SUPPORTED_FEATURES.get(feature, False)
    
    def _convert_messages(self, instructions: Optional[str], messages: List[Message]) -> List[Dict[str, Any]]:
        """
        Convert LLMHub messages to Anthropic format
        
        Args:
            instructions: System instructions
            messages: LLMHub messages
            
        Returns:
            List of Anthropic-formatted messages
        """
        anthropic_messages = []
        
        # Add system message if instructions provided
        if instructions:
            anthropic_messages.append({
                "role": "system",
                "content": instructions
            })
        
        # Convert LLMHub messages to Anthropic format
        for message in messages:
            # Start with basic message structure
            anthropic_message = {
                "role": message.role.value,
                "content": []
            }
            
            # Handle message content
            for content_item in message.content:
                if content_item.type == ContentType.TEXT:
                    # Text content
                    anthropic_message["content"].append({
                        "type": "text",
                        "text": content_item.text
                    })
                
                elif content_item.type == ContentType.IMAGE:
                    # Image content
                    if hasattr(content_item, "image_url") and content_item.image_url:
                        anthropic_message["content"].append({
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": str(content_item.image_url)
                            }
                        })
                    
                    elif hasattr(content_item, "image_path") and content_item.image_path:
                        # Read image file and convert to base64
                        try:
                            with open(content_item.image_path, "rb") as f:
                                image_bytes = f.read()
                                base64_image = base64.b64encode(image_bytes).decode("utf-8")
                                
                                # Try to determine media type from file extension
                                file_ext = content_item.image_path.split('.')[-1].lower()
                                media_type = {
                                    'jpg': 'image/jpeg',
                                    'jpeg': 'image/jpeg',
                                    'png': 'image/png',
                                    'gif': 'image/gif',
                                    'webp': 'image/webp',
                                }.get(file_ext, 'image/jpeg')
                                
                                anthropic_message["content"].append({
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": base64_image
                                    }
                                })
                        except Exception as e:
                            raise LLMHubError(f"Failed to read image file: {str(e)}")
                
                elif content_item.type == ContentType.DOCUMENT:
                    # Document content
                    if hasattr(content_item, "file") and content_item.file:
                        # Handle file data
                        file_data = content_item.file
                        
                        # Convert to format expected by Anthropic
                        try:
                            file_bytes = None
                            
                            # If file_data is a path, read the file
                            if isinstance(file_data.file_data, str) and os.path.isfile(file_data.file_data):
                                with open(file_data.file_data, "rb") as f:
                                    file_bytes = f.read()
                            elif isinstance(file_data.file_data, str):
                                # Try to decode base64
                                try:
                                    file_bytes = base64.b64decode(file_data.file_data)
                                except Exception:
                                    # If not base64, treat as raw string data
                                    file_bytes = file_data.file_data.encode("utf-8")
                            elif isinstance(file_data.file_data, bytes):
                                file_bytes = file_data.file_data
                            
                            if file_bytes is None:
                                raise LLMHubError(f"Could not process file data: {file_data.file_data}")
                            
                            # Encode as base64
                            base64_file = base64.b64encode(file_bytes).decode("utf-8")
                            
                            # Determine media type from file extension
                            file_ext = file_data.file_name.split('.')[-1].lower()
                            media_type = {
                                'pdf': 'application/pdf',
                                'txt': 'text/plain',
                                'csv': 'text/csv',
                                'json': 'application/json',
                                'html': 'text/html',
                                'md': 'text/markdown',
                                'doc': 'application/msword',
                                'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                                'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                                'pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
                            }.get(file_ext, 'application/octet-stream')
                            
                            anthropic_message["content"].append({
                                "type": "file",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": base64_file
                                }
                            })
                        except Exception as e:
                            raise LLMHubError(f"Failed to process document: {str(e)}")
                
                elif content_item.type == ContentType.AUDIO:
                    # Audio content
                    if hasattr(content_item, "audio_url") and content_item.audio_url:
                        # Not directly supported by Claude API yet
                        # Future implementation may involve audio transcription first
                        raise LLMHubError("Audio URL content not yet supported by Claude API")
                    
                    elif hasattr(content_item, "audio_path") and content_item.audio_path:
                        # Not directly supported by Claude API yet
                        # Future implementation may involve audio transcription first
                        raise LLMHubError("Audio file content not yet supported by Claude API")
            
            anthropic_messages.append(anthropic_message)
        
        return anthropic_messages
    
    def _convert_tools(self, tools: Optional[List[Tool]]) -> List[Dict[str, Any]]:
        """
        Convert LLMHub tools to Anthropic format
        
        Args:
            tools: LLMHub tools
            
        Returns:
            List of Anthropic-formatted tools
        """
        if not tools:
            return []
        
        anthropic_tools = []
        
        for tool in tools:
            if isinstance(tool, dict) and "__root__" in tool:
                tool = tool["__root__"]
            elif hasattr(tool, "root"):
                tool = tool.root
            
            if tool.type == "function":
                anthropic_tools.append({
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
        
        return anthropic_tools
    
    def _convert_tool_calls(self, tool_calls: List[Any]) -> List[ToolCall]:
        """
        Convert Anthropic tool calls to LLMHub format
        
        Args:
            tool_calls: Anthropic tool calls
            
        Returns:
            List of LLMHub ToolCall objects
        """
        llm_hub_tool_calls = []
        
        for call in tool_calls:
            # Parse arguments if they are a string
            args = call.input
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    # Keep as string if not valid JSON
                    args = {"raw_input": args}
            
            llm_hub_tool_calls.append(
                ToolCall(
                    id=call.id,
                    type="function",
                    function={
                        "name": call.name,
                        "arguments": args
                    }
                )
            )
        
        return llm_hub_tool_calls
    
    def _convert_content_to_message_content(self, 
                                    content_items: List[Dict[str, Any]]) -> Union[str, List[MessageContent]]:
        """
        Convert Anthropic content items to LLMHub format
        
        Args:
            content_items: Anthropic content items
            
        Returns:
            Either a string (for text-only content) or a list of MessageContent objects
        """
        # Check if it's text-only content
        if len(content_items) == 1 and content_items[0].get("type") == "text":
            return content_items[0].get("text", "")
        
        # Handle multi-modal content
        message_content = []
        
        for item in content_items:
            item_type = item.get("type")
            
            if item_type == "text":
                message_content.append(
                    MessageContent(
                        type=ContentType.TEXT,
                        text=item.get("text", "")
                    )
                )
            # Other content types would be handled here in a future implementation
        
        return message_content
    
    def _create_response(self, anthropic_response, stream=False) -> Union[GenerateResponse, Iterator[StreamingResponse]]:
        """
        Convert Anthropic response to LLMHub format
        
        Args:
            anthropic_response: Response from Anthropic API
            stream: Whether this is a streaming response
            
        Returns:
            LLMHub response object
        """
        if stream:
            return self._create_streaming_response(anthropic_response)
        
        # Extract response data
        role = Role.ASSISTANT
        
        # Handle content - convert to LLMHub format
        message_content = self._convert_content_to_message_content(anthropic_response.content)
        
        # Create response message
        response_message = ResponseMessage(
            role=role,
            content=message_content
        )
        
        # Add tool calls if present
        if hasattr(anthropic_response, "tool_use") and anthropic_response.tool_use:
            response_message.tool_calls = self._convert_tool_calls(anthropic_response.tool_use)
        
        # Create usage info
        usage_info = UsageInfo(
            prompt_tokens=anthropic_response.usage.input_tokens,
            completion_tokens=anthropic_response.usage.output_tokens,
            total_tokens=anthropic_response.usage.input_tokens + anthropic_response.usage.output_tokens
        )
        
        # Calculate cost
        cost_info = CostInfo(**self.calculate_cost(
            prompt_tokens=anthropic_response.usage.input_tokens,
            completion_tokens=anthropic_response.usage.output_tokens,
            model=anthropic_response.model
        ))
        
        # Create metadata
        metadata = ResponseMetadata(
            model=anthropic_response.model,
            provider="anthropic",
            usage=usage_info,
            cost=cost_info,
            request_id=anthropic_response.id
        )
        
        # Create and return response
        return GenerateResponse(
            message=response_message,
            metadata=metadata
        )
    
    def _create_streaming_response(self, response_stream) -> Iterator[StreamingResponse]:
        """
        Convert Anthropic streaming response to LLMHub format
        
        Args:
            response_stream: Streaming response from Anthropic API
            
        Returns:
            Iterator of LLMHub StreamingResponse objects
        """
        request_id = None
        model = None
        metadata = None
        accumulated_text = ""  # To accumulate text for tool extraction at the end
        
        for event in response_stream:
            # Store model and request ID from the first event
            if request_id is None and hasattr(event, "id"):
                request_id = event.id
            
            if model is None and hasattr(event, "model"):
                model = event.model
            
            # Process different event types
            if hasattr(event, "type"):
                if event.type == "content_block_delta":
                    # Text content delta
                    if event.delta.type == "text":
                        delta_text = event.delta.text
                        accumulated_text += delta_text
                        
                        yield StreamingResponse(
                            chunk=delta_text,
                            finished=False,
                            metadata=None
                        )
                        
                elif event.type == "content_block_start":
                    # Beginning of a content block
                    pass
                
                elif event.type == "content_block_stop":
                    # End of a content block
                    pass
                
                elif event.type == "message_delta":
                    # Message-level update
                    pass
                
                elif event.type == "message_start":
                    # Beginning of a message
                    pass
                
                elif event.type == "message_stop":
                    # Final event with usage info
                    if hasattr(event, "usage"):
                        usage = event.usage
                        
                        # Create usage info
                        usage_info = UsageInfo(
                            prompt_tokens=usage.input_tokens,
                            completion_tokens=usage.output_tokens,
                            total_tokens=usage.input_tokens + usage.output_tokens
                        )
                        
                        # Calculate cost
                        cost_info = CostInfo(**self.calculate_cost(
                            prompt_tokens=usage.input_tokens,
                            completion_tokens=usage.output_tokens,
                            model=model
                        ))
                        
                        # Create metadata
                        metadata = ResponseMetadata(
                            model=model,
                            provider="anthropic",
                            usage=usage_info,
                            cost=cost_info,
                            request_id=request_id
                        )
                    
                    # Yield final empty chunk with metadata
                    yield StreamingResponse(
                        chunk="",
                        finished=True,
                        metadata=metadata
                    )
                
                elif event.type == "tool_use":
                    # Tool use event (not expected in streaming, but handled just in case)
                    pass
    
    async def _acreate_streaming_response(self, response_stream) -> AsyncGenerator[StreamingResponse, None]:
        """
        Convert Anthropic async streaming response to LLMHub format
        
        Args:
            response_stream: Async streaming response from Anthropic API
            
        Returns:
            Async generator of LLMHub StreamingResponse objects
        """
        request_id = None
        model = None
        metadata = None
        accumulated_text = ""  # To accumulate text for tool extraction at the end
        
        async for event in response_stream:
            # Store model and request ID from the first event
            if request_id is None and hasattr(event, "id"):
                request_id = event.id
            
            if model is None and hasattr(event, "model"):
                model = event.model
            
            # Process different event types
            if hasattr(event, "type"):
                if event.type == "content_block_delta":
                    # Text content delta
                    if event.delta.type == "text":
                        delta_text = event.delta.text
                        accumulated_text += delta_text
                        
                        yield StreamingResponse(
                            chunk=delta_text,
                            finished=False,
                            metadata=None
                        )
                        
                elif event.type == "content_block_start":
                    # Beginning of a content block
                    pass
                
                elif event.type == "content_block_stop":
                    # End of a content block
                    pass
                
                elif event.type == "message_delta":
                    # Message-level update
                    pass
                
                elif event.type == "message_start":
                    # Beginning of a message
                    pass
                
                elif event.type == "message_stop":
                    # Final event with usage info
                    if hasattr(event, "usage"):
                        usage = event.usage
                        
                        # Create usage info
                        usage_info = UsageInfo(
                            prompt_tokens=usage.input_tokens,
                            completion_tokens=usage.output_tokens,
                            total_tokens=usage.input_tokens + usage.output_tokens
                        )
                        
                        # Calculate cost
                        cost_info = CostInfo(**self.calculate_cost(
                            prompt_tokens=usage.input_tokens,
                            completion_tokens=usage.output_tokens,
                            model=model
                        ))
                        
                        # Create metadata
                        metadata = ResponseMetadata(
                            model=model,
                            provider="anthropic",
                            usage=usage_info,
                            cost=cost_info,
                            request_id=request_id
                        )
                    
                    # Yield final empty chunk with metadata
                    yield StreamingResponse(
                        chunk="",
                        finished=True,
                        metadata=metadata
                    )
                
                elif event.type == "tool_use":
                    # Tool use event (not expected in streaming, but handled just in case)
                    pass
    
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
        Generate a response from Anthropic
        
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
            **kwargs: Additional Anthropic-specific parameters
            
        Returns:
            Either a complete response or a streaming response iterator
        """
        try:
            # Convert messages to Anthropic format
            anthropic_messages = self._convert_messages(instructions, messages)
            
            # Convert tools to Anthropic format
            anthropic_tools = None
            if tools:
                anthropic_tools = self._convert_tools(tools)
            
            # Prepare the request parameters
            request_params = {
                "model": self.model,
                "messages": anthropic_messages,
                "max_tokens": kwargs.pop("max_tokens", 4096),
                "stream": stream,
            }
            
            # Add tools if provided
            if anthropic_tools:
                request_params["tools"] = anthropic_tools
                
                # Configure tool choice behavior based on tool_choice parameter
                if tool_choice == "required":
                    request_params["tool_choice"] = "required"
                elif tool_choice == "none":
                    request_params["tool_choice"] = "none"
                # Default is auto
            
            # Add reasoning configuration if provided
            if reasoning:
                # Anthropic doesn't have direct reasoning parameters
                # We use system instructions to guide the model
                system_text = request_params.get("system", instructions or "")
                
                if reasoning.effort == "high":
                    system_text += "\nPlease think step-by-step and thoroughly about this query."
                elif reasoning.effort == "medium":
                    system_text += "\nPlease think through this query carefully."
                
                # If there's a system message, update it
                if anthropic_messages and anthropic_messages[0]["role"] == "system":
                    anthropic_messages[0]["content"] = system_text
                else:
                    # Add a system message if none exists
                    anthropic_messages.insert(0, {
                        "role": "system",
                        "content": system_text
                    })
                
                # Add max tokens for reasoning if specified
                if reasoning.max_tokens:
                    request_params["max_tokens"] = reasoning.max_tokens
            
            # Add structured output format if specified
            if output_format and output_format.schema:
                if isinstance(output_format.schema, dict):
                    # Convert to Anthropic's format
                    request_params["response_format"] = {"type": "json_object", "schema": output_format.schema}
                elif hasattr(output_format.schema, "schema") and callable(output_format.schema.schema):
                    # If it's a Pydantic model with schema method
                    request_params["response_format"] = {"type": "json_object", "schema": output_format.schema.schema()}
            
            # Add temperature if specified
            if "temperature" in kwargs:
                request_params["temperature"] = kwargs.pop("temperature")
            
            # Add top_p if specified
            if "top_p" in kwargs:
                request_params["top_p"] = kwargs.pop("top_p")
            
            # Add metadata if provided
            if metadata:
                request_params["metadata"] = metadata
            
            # Add timeout if specified
            if timeout:
                request_params["timeout"] = timeout
            
            # Add additional parameters
            for key, value in kwargs.items():
                request_params[key] = value
            
            # Make the API call
            if stream:
                response = self.client.messages.stream(**request_params)
                return self._create_streaming_response(response)
            else:
                response = self.client.messages.create(**request_params)
                return self._create_response(response)
            
        except anthropic.APIError as e:
            # Map Anthropic errors to LLMHub exceptions
            error_msg = str(e)
            
            if "unauthorized" in error_msg.lower() or "authentication" in error_msg.lower():
                raise AuthenticationError(error_msg, provider=self.provider_name)
            
            elif "rate limit" in error_msg.lower() or "rate_limit" in error_msg.lower():
                retry_after = getattr(e, "retry_after", None)
                raise RateLimitError(error_msg, provider=self.provider_name, retry_after=retry_after)
            
            elif "timeout" in error_msg.lower():
                raise TimeoutError(error_msg, provider=self.provider_name)
            
            elif "token limit" in error_msg.lower() or "context length" in error_msg.lower() or "context_length" in error_msg.lower():
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
        Asynchronously generate a response from Anthropic
        
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
            **kwargs: Additional Anthropic-specific parameters
            
        Returns:
            Either a complete response or an async streaming response generator
        """
        try:
            # Convert messages to Anthropic format
            anthropic_messages = self._convert_messages(instructions, messages)
            
            # Convert tools to Anthropic format
            anthropic_tools = None
            if tools:
                anthropic_tools = self._convert_tools(tools)
            
            # Prepare the request parameters
            request_params = {
                "model": self.model,
                "messages": anthropic_messages,
                "max_tokens": kwargs.pop("max_tokens", 4096),
                "stream": stream,
            }
            
            # Add tools if provided
            if anthropic_tools:
                request_params["tools"] = anthropic_tools
                
                # Configure tool choice behavior based on tool_choice parameter
                if tool_choice == "required":
                    request_params["tool_choice"] = "required"
                elif tool_choice == "none":
                    request_params["tool_choice"] = "none"
                # Default is auto
            
            # Add reasoning configuration if provided
            if reasoning:
                # Anthropic doesn't have direct reasoning parameters
                # We use system instructions to guide the model
                system_text = request_params.get("system", instructions or "")
                
                if reasoning.effort == "high":
                    system_text += "\nPlease think step-by-step and thoroughly about this query."
                elif reasoning.effort == "medium":
                    system_text += "\nPlease think through this query carefully."
                
                # If there's a system message, update it
                if anthropic_messages and anthropic_messages[0]["role"] == "system":
                    anthropic_messages[0]["content"] = system_text
                else:
                    # Add a system message if none exists
                    anthropic_messages.insert(0, {
                        "role": "system",
                        "content": system_text
                    })
                
                # Add max tokens for reasoning if specified
                if reasoning.max_tokens:
                    request_params["max_tokens"] = reasoning.max_tokens
            
            # Add structured output format if specified
            if output_format and output_format.schema:
                if isinstance(output_format.schema, dict):
                    # Convert to Anthropic's format
                    request_params["response_format"] = {"type": "json_object", "schema": output_format.schema}
                elif hasattr(output_format.schema, "schema") and callable(output_format.schema.schema):
                    # If it's a Pydantic model with schema method
                    request_params["response_format"] = {"type": "json_object", "schema": output_format.schema.schema()}
            
            # Add temperature if specified
            if "temperature" in kwargs:
                request_params["temperature"] = kwargs.pop("temperature")
            
            # Add top_p if specified
            if "top_p" in kwargs:
                request_params["top_p"] = kwargs.pop("top_p")
            
            # Add stop_sequences if specified
            if "stop_sequences" in kwargs:
                request_params["stop_sequences"] = kwargs.pop("stop_sequences")
            
            # Add metadata if provided
            if metadata:
                request_params["metadata"] = metadata
            
            # Add timeout if specified
            if timeout:
                request_params["timeout"] = timeout
            
            # Add additional parameters
            for key, value in kwargs.items():
                request_params[key] = value
            
            # Make the API call
            if stream:
                response = await self.async_client.messages.stream(**request_params)
                return self._acreate_streaming_response(response)
            else:
                response = await self.async_client.messages.create(**request_params)
                return self._create_response(response)
            
        except anthropic.APIError as e:
            # Map Anthropic errors to LLMHub exceptions
            error_msg = str(e)
            
            if "unauthorized" in error_msg.lower() or "authentication" in error_msg.lower():
                raise AuthenticationError(error_msg, provider=self.provider_name)
            
            elif "rate limit" in error_msg.lower() or "rate_limit" in error_msg.lower():
                retry_after = getattr(e, "retry_after", None)
                raise RateLimitError(error_msg, provider=self.provider_name, retry_after=retry_after)
            
            elif "timeout" in error_msg.lower():
                raise TimeoutError(error_msg, provider=self.provider_name)
            
            elif "token limit" in error_msg.lower() or "context length" in error_msg.lower() or "context_length" in error_msg.lower():
                raise TokenLimitError(error_msg, provider=self.provider_name)
            
            else:
                raise ProviderError(error_msg, provider=self.provider_name)
        
        except Exception as e:
            # Handle other errors
            raise LLMHubError(f"Unexpected error: {str(e)}", provider=self.provider_name)
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get a list of available models from Anthropic
        
        Returns:
            List of model information dictionaries
        """
        # Anthropic doesn't have a specific API endpoint for listing models
        # Return a predefined list of available models
        models = [
            {
                "id": "claude-3-opus-20240229",
                "provider": "anthropic",
                "capabilities": {
                    "vision": True,
                    "function_calling": True,
                    "reasoning": True,
                }
            },
            {
                "id": "claude-3-sonnet-20240229",
                "provider": "anthropic",
                "capabilities": {
                    "vision": True,
                    "function_calling": True,
                    "reasoning": True,
                }
            },
            {
                "id": "claude-3-haiku-20240307",
                "provider": "anthropic",
                "capabilities": {
                    "vision": True,
                    "function_calling": True,
                    "reasoning": True,
                }
            },
            {
                "id": "claude-2.1",
                "provider": "anthropic",
                "capabilities": {
                    "vision": False,
                    "function_calling": False,
                    "reasoning": False,
                }
            },
            {
                "id": "claude-2.0",
                "provider": "anthropic",
                "capabilities": {
                    "vision": False,
                    "function_calling": False,
                    "reasoning": False,
                }
            },
            {
                "id": "claude-instant-1.2",
                "provider": "anthropic",
                "capabilities": {
                    "vision": False,
                    "function_calling": False,
                    "reasoning": False,
                }
            }
        ]
        
        return models
    
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
        # Default to claude-3-sonnet pricing if model not found
        prompt_price, completion_price = self.TOKEN_PRICING.get(
            model, self.TOKEN_PRICING.get("claude-3-sonnet-20240229", (0.003, 0.015))
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
            