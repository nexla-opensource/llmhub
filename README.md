# LLM Hub

A unified Python interface for major Large Language Model providers (OpenAI, Claude, Gemini, and LiteLLM) with advanced features like tracing, cost tracking, retries, and structured output.

## Features

- **Unified API** across all supported LLMs
- **Extensible Architecture** for adding new LLMs
- **Synchronous and Asynchronous Interfaces** for flexibility
- **Tracing and Logging** of all requests and responses
- **Cost Tracking** and usage monitoring
- **Automatic Retries** and robust exception handling
- **Clear Error Handling** like Rate Limit, Timedout, Token Context limit exceeded.

- **Tool Use/Function Calling** for agentic capabilities
- **Vision Model Support** for processing images and documents
- **Document Uploading** (where supported by model/provider)
- **Thinking Steps** (intermediate reasoning, if supported)
- **Streaming Responses** (where supported)
- **Structured Output** (JSON, pydantic models)
- **Prompt Caching** 

## Installation

```bash
pip install llm_hub
```

## Quick Start

### Synchronous API

```python
from llm_hub import LLMHub

# Initialize with your preferred provider
hub = LLMHub(
    provider="openai",
    api_key="sk-...",
    tracing=True,
    cost_tracking=True,
    retries=3,
)

# Simple text generation
response = hub.generate(
    prompt="Explain quantum computing in simple terms.",
    max_tokens=500
)
print(response.text)

# Stream response
for chunk in hub.generate_stream(
    prompt="Write a short story about a robot learning to paint.",
    max_tokens=1000
):
    print(chunk, end="", flush=True)

# Working with documents
from llm_hub.utils import Document

doc = Document.from_file("research-paper.pdf")
summary = hub.generate(
    prompt="Summarize this research paper",
    document=doc,
    max_tokens=300
)

# Vision capabilities with images
image_doc = Document.from_file("image.jpg")
description = hub.generate(
    prompt="What's in this image?",
    document=image_doc,
    model="gpt-4o"  # or any vision-capable model
)

# Tool Use/Function Calling
from pydantic import BaseModel, Field
from typing import List

class WeatherInput(BaseModel):
    location: str = Field(..., description="City and state/country")
    unit: str = Field("celsius", description="Temperature unit")

# Define tools for the model to use
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather in a location",
            "parameters": WeatherInput.model_json_schema()
        }
    }
]

tool_response = hub.generate(
    prompt="What's the weather like in San Francisco?",
    tools=tools,
    model="gpt-4o"  # or any tool-capable model
)

# Check if tool calls were made
if "tool_calls" in tool_response.metadata:
    for tool_call in tool_response.metadata["tool_calls"]:
        print(f"Tool: {tool_call['function']['name']}")
        print(f"Arguments: {tool_call['function']['arguments']}")

# Structured output
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int
    occupation: str

result = hub.generate(
    prompt="Extract person info: John Doe is a 35-year-old software engineer",
    output_schema=Person
)
print(f"Name: {result.name}, Age: {result.age}, Job: {result.occupation}")
```

### Asynchronous API

LLM Hub also provides an asynchronous API for use with `async`/`await`:

```python
import asyncio
from llm_hub import LLMHub

async def main():
    hub = LLMHub(
        provider="openai",
        api_key="sk-...",
        tracing=True,
        cost_tracking=True
    )
    
    # Simple async text generation
    response = await hub.agenerate(
        prompt="Explain quantum computing in simple terms.",
        max_tokens=500
    )
    print(response.text)
    
    # Async streaming
    async for chunk in hub.agenerate_stream(
        prompt="Write a short story about a robot learning to paint.",
        max_tokens=1000
    ):
        print(chunk, end="", flush=True)
    
    # Async tool calling
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_database",
                "description": "Search a database for information",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}
            }
        }
    ]
    
    tool_response = await hub.agenerate(
        prompt="Find information about renewable energy technologies",
        tools=tools,
        model="claude-3-5-sonnet-20241022"  # or any tool-capable model
    )
    
    # Run multiple requests in parallel
    results = await asyncio.gather(
        hub.agenerate(prompt="Write a haiku about the ocean."),
        hub.agenerate(prompt="Give me 3 quick tips for productive programming."),
        hub.agenerate(prompt="Explain async/await in Python briefly.")
    )
    
    for i, response in enumerate(results, 1):
        print(f"\nResult {i}:\n{response.text}")

# Run the async example
asyncio.run(main())
```

## Supported Providers

- **OpenAI** (GPT models) - Supports tool calling, vision, and structured output
- **Claude** (via Anthropic) - Supports tool use with Claude 3.5, vision capabilities, and structured output
- **Gemini** (Google) - Supports multimodal capabilities
- **LiteLLM** (for access to 100+ additional LLMs)

## License

MIT 