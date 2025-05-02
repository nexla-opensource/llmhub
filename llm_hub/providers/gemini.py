"""
Google Gemini provider implementation for LLM Hub
"""

import asyncio
import base64
import json
import mimetypes
import os
from typing import Any, Dict, List, Optional, Union, AsyncGenerator, Iterator

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from google.generativeai.types.generation_types import GenerationConfig
from google.ai.generativelanguage_v1 import FunctionDeclaration, Tool, Part

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
    Tool as LLMHubTool,
    ToolCall,
    UsageInfo,
    CostInfo,
    ResponseFormat,
    ReasoningConfig,
    Role,
)
from .base import BaseProvider


class GeminiProvider(BaseProvider):
    """
    Provider implementation for Google Gemini
    """
    
    # Feature support flags
    SUPPORTED_FEATURES = {
        "function_calling": True,
        "vision": True,
        "streaming": True,
        "structured_output": True,
        "file_upload": True,
        "reasoning": False,  # Gemini doesn't have explicit reasoning parameters
    }
    
    # Default models
    DEFAULT_MODEL = "gemini-1.5-pro"
    VISION_MODELS = ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro-vision"]
    
    # Models that support function calling
    FUNCTION_CALLING_MODELS = ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"]
    
    # Token pricing (per 1K tokens as of May 2024)
    # Format: {model_name: (prompt_price, completion_price)}
    TOKEN_PRICING = {
        "gemini-1.5-pro": (0.00175, 0.00525),
        "gemini-1.5-flash": (0.000350, 0.00105),
        "gemini-pro": (0.000125, 0.000375),
        "gemini-pro-vision": (0.000125, 0.000375),
    }
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        model: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the Gemini provider
        
        Args:
            api_key: Google AI API key (defaults to GOOGLE_API_KEY env var)
            model: Model to use (defaults to DEFAULT_MODEL)
            **kwargs: Additional options
        """
        # Use provided API key or get from environment
        api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ConfigurationError("Google API key not provided and GOOGLE_API_KEY environment variable not set")
        
        super().__init__(api_key, **kwargs)
        
        # Set default model if not provided
        self.model = model or self.DEFAULT_MODEL
        
        # Initialize the Gemini client
        genai.configure(api_key=api_key)
        
        # Store additional parameters
        self.generation_config = kwargs.get("generation_config", {})
        self.safety_settings = kwargs.get("safety_settings", None)
    
    @property
    def provider_name(self) -> str:
        return "gemini"
    
    def supports_feature(self, feature: str) -> bool:
        return self.SUPPORTED_FEATURES.get(feature, False)
    
    def _create_gemini_model(self):
        """
        Create a Gemini model instance
        
        Returns:
            Gemini model object
        """
        return genai.GenerativeModel(
            model_name=self.model,
            generation_config=self._get_generation_config(),
            safety_settings=self._get_safety_settings()
        )
    
    def _get_generation_config(self) -> GenerationConfig:
        """
        Get the generation configuration for Gemini
        
        Returns:
            Gemini GenerationConfig object
        """
        # Start with default config
        config = {}
        
        # Add parameters from initialization
        config.update(self.generation_config)
        
        return GenerationConfig(**config)
    
    def _get_safety_settings(self) -> Optional[List[Dict[str, Any]]]:
        """
        Get safety settings for Gemini
        
        Returns:
            List of safety settings or None for defaults
        """
        if self.safety_settings is not None:
            return self.safety_settings
        
        # Use default safety settings
        return None
    
    def _convert_messages(self, instructions: Optional[str], messages: List[Message]) -> List[Dict[str, Any]]:
        """
        Convert LLMHub messages to Gemini format
        
        Args:
            instructions: System instructions
            messages: LLMHub messages
            
        Returns:
            List of Gemini-formatted messages
        """
        gemini_messages = []
        
        # Add system message if instructions provided
        if instructions:
            gemini_messages.append({
                "role": "system",
                "parts": [{"text": instructions}]
            })
        
        # Convert LLMHub messages to Gemini format
        for message in messages:
            # Start with basic message structure
            gemini_message = {
                "role": self._convert_role(message.role),
                "parts": []
            }
            
            # Handle message content
            for content_item in message.content:
                if content_item.type == ContentType.TEXT:
                    # Text content
                    gemini_message["parts"].append({
                        "text": content_item.text
                    })
                
                elif content_item.type == ContentType.IMAGE:
                    # Image content
                    if hasattr(content_item, "image_url") and content_item.image_url:
                        gemini_message["parts"].append({
                            "inline_data": {
                                "mime_type": "image/jpeg",  # Default to JPEG
                                "data": self._get_base64_from_url(str(content_item.image_url))
                            }
                        })
                    
                    elif hasattr(content_item, "image_path") and content_item.image_path:
                        # Read image file and convert to base64
                        try:
                            mime_type = mimetypes.guess_type(content_item.image_path)[0] or "image/jpeg"
                            with open(content_item.image_path, "rb") as f:
                                image_bytes = f.read()
                                base64_image = base64.b64encode(image_bytes).decode("utf-8")
                                
                                gemini_message["parts"].append({
                                    "inline_data": {
                                        "mime_type": mime_type,
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
                        
                        # Convert to format expected by Gemini
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
                            mime_type = mimetypes.guess_type(file_data.file_name)[0] or "application/octet-stream"
                            
                            gemini_message["parts"].append({
                                "inline_data": {
                                    "mime_type": mime_type,
                                    "data": base64_file
                                }
                            })
                        except Exception as e:
                            raise LLMHubError(f"Failed to process document: {str(e)}")
                
                elif content_item.type == ContentType.AUDIO:
                    # Audio content
                    if hasattr(content_item, "audio_url") and content_item.audio_url:
                        # Not directly supported by Gemini API yet
                        raise LLMHubError("Audio URL content not yet supported by Gemini API")
                    
                    elif hasattr(content_item, "audio_path") and content_item.audio_path:
                        # Not directly supported by Gemini API yet
                        raise LLMHubError("Audio file content not yet supported by Gemini API")
            
            gemini_messages.append(gemini_message)
        
        return gemini_messages
    
    def _convert_role(self, role: Role) -> str:
        """
        Convert LLMHub role to Gemini role
        
        Args:
            role: LLMHub role
            
        Returns:
            Gemini role string
        """
        # Map roles
        role_map = {
            Role.USER: "user",
            Role.ASSISTANT: "model",
            Role.SYSTEM: "system",
        }
        
        return role_map.get(role, "user")
    
    def _get_base64_from_url(self, url: str) -> str:
        """
        Get base64-encoded data from a URL
        
        Args:
            url: URL to fetch
            
        Returns:
            Base64-encoded string
        """
        try:
            import requests
            
            # Check if it's already a data URL
            if url.startswith("data:"):
                # Extract the base64 part
                base64_data = url.split(",", 1)[1]
                return base64_data
            
            # Download the image
            response = requests.get(url)
            response.raise_for_status()
            
            # Convert to base64
            base64_data = base64.b64encode(response.content).decode("utf-8")
            return base64_data
            
        except Exception as e:
            raise LLMHubError(f"Failed to fetch data from URL: {str(e)}")
    
    def _convert_tools(self, tools: Optional[List[LLMHubTool]]) -> List[Dict[str, Any]]:
        """
        Convert LLMHub tools to Gemini format
        
        Args:
            tools: LLMHub tools
            
        Returns:
            List of Gemini-formatted tools
        """
        if not tools:
            return []
        
        gemini_tools = []
        
        for tool in tools:
            if isinstance(tool, dict) and "__root__" in tool:
                tool = tool["__root__"]
            elif hasattr(tool, "root"):
                tool = tool.root
            
            if tool.type == "function":
                function_declaration = {
                    "name": tool.function.name,
                    "description": tool.function.description,
                    "parameters": {
                        "type": tool.function.parameters.type,
                        "properties": tool.function.parameters.properties,
                    }
                }
                
                # Add required fields if present
                if tool.function.parameters.required:
                    function_declaration["parameters"]["required"] = tool.function.parameters.required
                
                gemini_tools.append({
                    "function_declarations": [function_declaration]
                })
        
        return gemini_tools
    
    def _convert_tool_calls(self, tool_calls: List[Any]) -> List[ToolCall]:
        """
        Convert Gemini function calls to LLMHub format
        
        Args:
            tool_calls: Gemini function calls
            
        Returns:
            List of LLMHub ToolCall objects
        """
        llm_hub_tool_calls = []
        
        for call in tool_calls:
            try:
                # Parse arguments
                if isinstance(call.args, str):
                    args = json.loads(call.args)
                else:
                    args = call.args
                
                llm_hub_tool_calls.append(
                    ToolCall(
                        id=str(hash(f"{call.name}_{json.dumps(args)}")),  # Generate a deterministic ID
                        type="function",
                        function={
                            "name": call.name,
                            "arguments": args
                        }
                    )
                )
            except Exception as e:
                # If we can't parse the arguments, include them as raw string
                llm_hub_tool_calls.append(
                    ToolCall(
                        id=str(hash(f"{call.name}_{call.args}")),
                        type="function",
                        function={
                            "name": call.name,
                            "arguments": {"raw_args": call.args}
                        }
                    )
                )
        
        return llm_hub_tool_calls
    
    def _create_response(self, gemini_response, model_name: str, stream=False) -> Union[GenerateResponse, Iterator[StreamingResponse]]:
        """
        Convert Gemini response to LLMHub format
        
        Args:
            gemini_response: Response from Gemini API
            model_name: Name of the model used
            stream: Whether this is a streaming response
            
        Returns:
            LLMHub response object
        """
        if stream:
            return self._create_streaming_response(gemini_response, model_name)
        
        # Extract response data
        candidates = gemini_response.candidates
        if not candidates:
            raise LLMHubError("Empty response from Gemini API")
        
        candidate = candidates[0]
        content = candidate.content
        
        # Get the text from parts
        text_content = ""
        for part in content.parts:
            if hasattr(part, "text") and part.text:
                text_content += part.text
        
        # Create response message
        response_message = ResponseMessage(
            role=Role.ASSISTANT,
            content=text_content
        )
        
        # Add tool calls if present
        if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
            function_calls = []
            for part in candidate.content.parts:
                if hasattr(part, "function_call"):
                    function_calls.append(part.function_call)
            
            if function_calls:
                response_message.tool_calls = self._convert_tool_calls(function_calls)
        
        # Create usage info - Gemini doesn't provide detailed token counts
        # Estimate based on characters (rough approximation)
        char_count = len(str(gemini_response))
        token_estimate = char_count // 4  # very rough estimate
        
        usage_info = UsageInfo(
            prompt_tokens=token_estimate // 2,  # Rough estimate
            completion_tokens=token_estimate // 2,  # Rough estimate
            total_tokens=token_estimate
        )
        
        # Calculate cost (rough estimate)
        cost_info = CostInfo(**self.calculate_cost(
            prompt_tokens=usage_info.prompt_tokens,
            completion_tokens=usage_info.completion_tokens,
            model=model_name
        ))
        
        # Create metadata
        metadata = ResponseMetadata(
            model=model_name,
            provider="gemini",
            usage=usage_info,
            cost=cost_info,
            request_id=getattr(gemini_response, "response_id", None)
        )
        
        # Create and return response
        return GenerateResponse(
            message=response_message,
            metadata=metadata
        )
    
    def _create_streaming_response(self, response_stream, model_name: str) -> Iterator[StreamingResponse]:
        """
        Convert Gemini streaming response to LLMHub format
        
        Args:
            response_stream: Streaming response from Gemini API
            model_name: Name of the model used
            
        Returns:
            Iterator of LLMHub StreamingResponse objects
        """
        # For cost estimation
        total_chars = 0
        metadata = None
        
        for chunk in response_stream:
            # Extract text from chunk
            text = ""
            if hasattr(chunk, "text"):
                text = chunk.text
            elif hasattr(chunk, "parts") and chunk.parts:
                for part in chunk.parts:
                    if hasattr(part, "text"):
                        text += part.text
            
            # Update total character count for cost estimation
            total_chars += len(text)
            
            # Check if this is the last chunk
            finished = not hasattr(chunk, "parts") or not chunk.parts
            
            # Create metadata on the final chunk
            if finished:
                # Rough token estimation
                token_estimate = total_chars // 4
                
                # Create usage info
                usage_info = UsageInfo(
                    prompt_tokens=token_estimate // 2,  # Rough estimate
                    completion_tokens=token_estimate // 2,  # Rough estimate
                    total_tokens=token_estimate
                )
                
                # Calculate cost
                cost_info = CostInfo(**self.calculate_cost(
                    prompt_tokens=usage_info.prompt_tokens,
                    completion_tokens=usage_info.completion_tokens,
                    model=model_name
                ))
                
                # Create metadata
                metadata = ResponseMetadata(
                    model=model_name,
                    provider="gemini",
                    usage=usage_info,
                    cost=cost_info,
                    request_id=None  # Gemini streaming doesn't provide request ID
                )
            
            yield StreamingResponse(
                chunk=text,
                finished=finished,
                metadata=metadata if finished else None
            )
    
    async def _acreate_streaming_response(self, response_stream, model_name: str) -> AsyncGenerator[StreamingResponse, None]:
        """
        Convert Gemini async streaming response to LLMHub format
        
        Args:
            response_stream: Async streaming response from Gemini API
            model_name: Name of the model used
            
        Returns:
            Async generator of LLMHub StreamingResponse objects
        """
        # For cost estimation
        total_chars = 0
        metadata = None
        
        async for chunk in response_stream:
            # Extract text from chunk
            text = ""
            if hasattr(chunk, "text"):
                text = chunk.text
            elif hasattr(chunk, "parts") and chunk.parts:
                for part in chunk.parts:
                    if hasattr(part, "text"):
                        text += part.text
            
            # Update total character count for cost estimation
            total_chars += len(text)
            
            # Check if this is the last chunk
            finished = not hasattr(chunk, "parts") or not chunk.parts
            
            # Create metadata on the final chunk
            if finished:
                # Rough token estimation
                token_estimate = total_chars // 4
                
                # Create usage info
                usage_info = UsageInfo(
                    prompt_tokens=token_estimate // 2,  # Rough estimate
                    completion_tokens=token_estimate // 2,  # Rough estimate
                    total_tokens=token_estimate
                )
                
                # Calculate cost
                cost_info = CostInfo(**self.calculate_cost(
                    prompt_tokens=usage_info.prompt_tokens,
                    completion_tokens=usage_info.completion_tokens,
                    model=model_name
                ))
                
                # Create metadata
                metadata = ResponseMetadata(
                    model=model_name,
                    provider="gemini",
                    usage=usage_info,
                    cost=cost_info,
                    request_id=None  # Gemini streaming doesn't provide request ID
                )
            
            yield StreamingResponse(
                chunk=text,
                finished=finished,
                metadata=metadata if finished else None
            )
    
    def generate(
        self,
        instructions: Optional[str],
        messages: List[Message],
        tools: Optional[List[LLMHubTool]] = None,
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
        Generate a response from Gemini
        
        Args:
            instructions: System instructions for the model
            messages: List of messages in the conversation
            tools: List of tools available to the model
            tool_choice: How the model should choose tools
            parallel_tool_calls: Whether to allow parallel tool calls
            output_format: Format specification for the output
            timeout: Request timeout in seconds
            stream: Whether to stream the response
            reasoning: Configuration for model reasoning (not used for Gemini)
            metadata: Additional metadata for tracing and logging
            **kwargs: Additional Gemini-specific parameters
            
        Returns:
            Either a complete response or a streaming response iterator
        """
        try:
            # Create the model
            model = self._create_gemini_model()
            
            # Convert messages to Gemini format
            gemini_messages = self._convert_messages(instructions, messages)
            
            # Prepare generation config
            generation_config = self._get_generation_config()
            
            # Add temperature if specified
            if "temperature" in kwargs:
                generation_config.temperature = kwargs.pop("temperature")
            
            # Add top_p if specified
            if "top_p" in kwargs:
                generation_config.top_p = kwargs.pop("top_p")
            
            # Add top_k if specified
            if "top_k" in kwargs:
                generation_config.top_k = kwargs.pop("top_k")
            
            # Add max_output_tokens if specified
            if "max_tokens" in kwargs:
                generation_config.max_output_tokens = kwargs.pop("max_tokens")
            
            # Add tools if provided
            gemini_tools = None
            if tools:
                gemini_tools = self._convert_tools(tools)
            
            # Begin chat session
            chat = model.start_chat(history=[])
            
            # Prepare content for the request
            content = []
            
            # Get the last user message
            last_user_message = None
            for message in reversed(gemini_messages):
                if message["role"] == "user":
                    last_user_message = message
                    break
            
            if last_user_message is None:
                raise LLMHubError("No user message found in the conversation")
            
            # Add parts from the last user message
            for part in last_user_message["parts"]:
                content.append(part)
            
            # Make the API call
            if stream:
                response = chat.send_message(
                    content=content,
                    generation_config=generation_config,
                    tools=gemini_tools,
                    stream=True
                )
                return self._create_streaming_response(response, self.model)
            else:
                response = chat.send_message(
                    content=content,
                    generation_config=generation_config,
                    tools=gemini_tools,
                    stream=False
                )
                return self._create_response(response, self.model)
            
        except Exception as e:
            # Map Gemini errors to LLMHub exceptions
            error_msg = str(e)
            
            if "authentication" in error_msg.lower() or "api key" in error_msg.lower():
                raise AuthenticationError(error_msg, provider=self.provider_name)
            
            elif "rate limit" in error_msg.lower() or "quota" in error_msg.lower():
                raise RateLimitError(error_msg, provider=self.provider_name)
            
            elif "timeout" in error_msg.lower():
                raise TimeoutError(error_msg, provider=self.provider_name)
            
            elif "token limit" in error_msg.lower() or "context length" in error_msg.lower():
                raise TokenLimitError(error_msg, provider=self.provider_name)
            
            elif "safety" in error_msg.lower() or "blocked" in error_msg.lower():
                raise ProviderError(f"Content filtered due to safety concerns: {error_msg}", provider=self.provider_name)
            
            else:
                raise ProviderError(error_msg, provider=self.provider_name)
    
    async def agenerate(
        self,
        instructions: Optional[str],
        messages: List[Message],
        tools: Optional[List[LLMHubTool]] = None,
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
        Asynchronously generate a response from Gemini
        
        Args:
            instructions: System instructions for the model
            messages: List of messages in the conversation
            tools: List of tools available to the model
            tool_choice: How the model should choose tools
            parallel_tool_calls: Whether to allow parallel tool calls
            output_format: Format specification for the output
            timeout: Request timeout in seconds
            stream: Whether to stream the response
            reasoning: Configuration for model reasoning (not used for Gemini)
            metadata: Additional metadata for tracing and logging
            **kwargs: Additional Gemini-specific parameters
            
        Returns:
            Either a complete response or an async streaming response generator
        """
        # Convert the sync implementation to async
        async def run_in_executor():
            return self.generate(
                instructions=instructions,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                parallel_tool_calls=parallel_tool_calls,
                output_format=output_format,
                timeout=timeout,
                stream=stream,
                reasoning=reasoning,
                metadata=metadata,
                **kwargs
            )
        
        # Run the synchronous implementation in a thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, run_in_executor)
        
        # If it's a streaming response, we need to convert it to an async generator
        if stream:
            async def async_generator():
                for chunk in result:
                    yield chunk
            
            return async_generator()
        
        # Otherwise, just return the response
        return result
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get a list of available models from Gemini
        
        Returns:
            List of model information dictionaries
        """
        try:
            # Fetch available models from the API
            models = genai.list_models()
            
            # Filter for Gemini models only
            gemini_models = []
            for model in models:
                if "gemini" in model.name.lower():
                    model_info = {
                        "id": model.name,
                        "provider": "gemini",
                        "capabilities": {
                            "vision": model.name in self.VISION_MODELS,
                            "function_calling": model.name in self.FUNCTION_CALLING_MODELS,
                        }
                    }
                    gemini_models.append(model_info)
            
            return gemini_models
            
        except Exception as e:
            # If we can't fetch models, return a predefined list
            return [
                {
                    "id": "gemini-1.5-pro",
                    "provider": "gemini",
                    "capabilities": {
                        "vision": True,
                        "function_calling": True,
                    }
                },
                {
                    "id": "gemini-1.5-flash",
                    "provider": "gemini",
                    "capabilities": {
                        "vision": True,
                        "function_calling": True,
                    }
                },
                {
                    "id": "gemini-pro",
                    "provider": "gemini",
                    "capabilities": {
                        "vision": False,
                        "function_calling": True,
                    }
                },
                {
                    "id": "gemini-pro-vision",
                    "provider": "gemini",
                    "capabilities": {
                        "vision": True,
                        "function_calling": False,
                    }
                },
            ]
    
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
        # Default to gemini-pro pricing if model not found
        prompt_price, completion_price = self.TOKEN_PRICING.get(
            model, self.TOKEN_PRICING.get("gemini-pro", (0.000125, 0.000375))
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
        
        # Validate reasoning config (Gemini doesn't support explicit reasoning)
        if request.reasoning:
            # Don't fail, but log a warning that this will be ignored
            import logging
            logging.warning(
                f"Model '{model}' does not support explicit reasoning configuration, this will be ignored"
            )