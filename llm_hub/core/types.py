"""
Core type definitions for LLM Hub using Pydantic
"""

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, HttpUrl, RootModel


class ContentType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    DOCUMENT = "file"
    AUDIO = "audio"


class CacheControlType(str, Enum):
    EPHEMERAL = "ephemeral"
    PERSISTENT = "persistent"


class CacheControl(BaseModel):
    type: CacheControlType = Field(default=CacheControlType.PERSISTENT)


class BaseContent(BaseModel):
    type: ContentType
    cache_control: Optional[CacheControl] = None


class TextContent(BaseContent):
    type: Literal[ContentType.TEXT] = ContentType.TEXT
    text: str


class ImageDetail(str, Enum):
    LOW = "low"
    HIGH = "high"
    AUTO = "auto"


class ImageContent(BaseContent):
    type: Literal[ContentType.IMAGE] = ContentType.IMAGE
    image_url: Optional[HttpUrl] = None
    image_path: Optional[str] = None
    detail: Optional[ImageDetail] = ImageDetail.AUTO

    class Config:
        extra = "forbid"

    def model_post_init(self, *args, **kwargs):
        if not self.image_url and not self.image_path:
            raise ValueError("Either image_url or image_path must be provided")
        if self.image_url and self.image_path:
            raise ValueError("Only one of image_url or image_path can be provided")


class FileData(BaseModel):
    file_name: str
    file_data: Union[str, bytes]  # URL or path


class DocumentContent(BaseContent):
    type: Literal[ContentType.DOCUMENT] = ContentType.DOCUMENT
    file: FileData


class AudioContent(BaseContent):
    type: Literal[ContentType.AUDIO] = ContentType.AUDIO
    audio_url: Optional[HttpUrl] = None
    audio_path: Optional[str] = None

    class Config:
        extra = "forbid"
    
    def model_post_init(self, *args, **kwargs):
        if not self.audio_url and not self.audio_path:
            raise ValueError("Either audio_url or audio_path must be provided")
        if self.audio_url and self.audio_path:
            raise ValueError("Only one of audio_url or audio_path can be provided")


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class Message(BaseModel):
    role: Role
    content: List[Union[TextContent, ImageContent, DocumentContent, AudioContent]] | str
    tool_call_id: Optional[str] = None
    tool_name: Optional[str] = None

class ToolType(str, Enum):
    FUNCTION = "function"


class FunctionParameter(BaseModel):
    type: str
    properties: Dict[str, Any]
    required: Optional[List[str]] = None


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: FunctionParameter


class FunctionTool(BaseModel):
    type: Literal[ToolType.FUNCTION] = ToolType.FUNCTION
    function: FunctionDefinition


class Tool(RootModel):
    root: Union[FunctionTool]


class ToolChoice(str, Enum):
    AUTO = "auto"
    REQUIRED = "required"
    NONE = "none"


class ReasoningEffort(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SummaryType(str, Enum):
    AUTO = "auto"
    DETAILED = "detailed"
    NONE = "none"


class ReasoningConfig(BaseModel):
    effort: Optional[ReasoningEffort] = ReasoningEffort.MEDIUM
    summary: Optional[SummaryType] = SummaryType.AUTO
    max_tokens: Optional[int] = None


class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


class ResponseFormatType(str, Enum):
    JSON_OBJECT = "json_object"
    JSON_SCHEMA = "json_schema"
    TEXT = "text"


class ResponseFormat(BaseModel):
    type: ResponseFormatType
    schema: Optional[Dict[str, Any]] = None
    name: Optional[str] = None  # Name for the schema, useful for documentation
    strict: bool = True  # Whether to strictly enforce schema validation
    description: Optional[str] = None  # Optional description for the schema
    model: Optional[Any] = None  # Pydantic model


class LLMHubConfig(BaseModel):
    provider: Provider
    api_key: str
    model: Optional[str] = None
    tracing: bool = False
    cost_tracking: bool = False
    retries: int = 0
    timeout: int = 60
    metadata: Optional[Dict[str, str]] = None
    

class GenerateRequest(BaseModel):
    instructions: Optional[str] = None  # System prompt
    messages: List[Message]
    tools: Optional[List[Union[FunctionTool]]] = None
    tool_choice: Optional[ToolChoice] = ToolChoice.AUTO
    parallel_tool_calls: bool = False
    output_format: Optional[ResponseFormat] = None
    max_retries: Optional[int] = None
    timeout: Optional[int] = None
    stream: bool = False
    reasoning: Optional[ReasoningConfig] = None
    metadata: Optional[Dict[str, str]] = None


class FunctionCall(BaseModel):
    name: str
    arguments: Dict[str, Any]


class ToolCall(BaseModel):
    id: str
    type: ToolType
    function: FunctionCall


class MessageContent(BaseModel):
    type: ContentType
    text: Optional[str] = None


class ResponseMessage(BaseModel):
    role: Role
    content: Union[str, List[MessageContent]]
    tool_calls: Optional[List[ToolCall]] = None


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CostInfo(BaseModel):
    prompt_cost: float
    completion_cost: float
    total_cost: float


class ResponseMetadata(BaseModel):
    model: str
    provider: Provider
    usage: Optional[UsageInfo] = None
    cost: Optional[CostInfo] = None
    request_id: Optional[str] = None
    latency: Optional[float] = None  # in seconds


class GenerateResponse(BaseModel):
    message: ResponseMessage
    metadata: ResponseMetadata


class StreamingResponse(BaseModel):
    chunk: str
    finished: bool
    metadata: Optional[ResponseMetadata] = None
