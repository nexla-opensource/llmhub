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

### Basic Example

```python
from llm_hub import LLMHub

# Initialize with your preferred provider
llm = LLMHub(
    provider="openai",
    api_key="sk-...",
    model="gpt-4o",
    tracing=True,
    cost_tracking=True,
)

# Generate a response
response = llm.generate(
    instructions="You are a helpful assistant.",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Tell me about the history of artificial intelligence."
                }
            ]
        }
    ]
)

# Print the response
print(response.message.content)

# Print usage and cost information
usage = response.metadata.usage
cost = response.metadata.cost
print(f"Usage: {usage.total_tokens} tokens ({usage.prompt_tokens} prompt, {usage.completion_tokens} completion)")
print(f"Cost: ${cost.total_cost:.6f} (${cost.prompt_cost:.6f} prompt, ${cost.completion_cost:.6f} completion)")
```

### Multimodal Example

```python
from llm_hub import LLMHub
from llm_hub.core.types import TextContent, ImageContent, Message, Role

# Initialize LLM Hub
llm = LLMHub(provider="openai", api_key="sk-...", model="gpt-4o")

# Generate a response with image input
response = llm.generate(
    messages=[
        Message(
            role=Role.USER,
            content=[
                TextContent(
                    type="text",
                    text="What can you tell me about this image?"
                ),
                ImageContent(
                    type="image",
                    image_url="https://example.com/image.jpg",
                    detail="high"
                )
            ]
        )
    ]
)

print(response.message.content)
```

### Function Calling Example

```python
from llm_hub import LLMHub
from llm_hub.core.types import FunctionTool
import json

# Initialize LLM Hub
llm = LLMHub(provider="openai", api_key="sk-...", model="gpt-4o")

# Define a tool (function)
weather_tool = FunctionTool(
    type="function",
    function={
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The unit of temperature"
                }
            },
            "required": ["location"]
        }
    }
)

# Generate a response with function calling
response = llm.generate(
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What's the weather like in San Francisco?"
                }
            ]
        }
    ],
    tools=[weather_tool],
    tool_choice="auto"
)

# Check if the model called a function
if response.message.tool_calls:
    for tool_call in response.message.tool_calls:
        print(f"Function called: {tool_call.function.name}")
        print(f"Arguments: {tool_call.function.arguments}")
        
        # In a real application, you would call the actual function here
        # and then provide the result back to the model
        weather_data = {
            "temperature": 72,
            "unit": tool_call.function.arguments.get("unit", "fahrenheit"),
            "condition": "sunny",
            "location": tool_call.function.arguments.get("location", "San Francisco")
        }
        
        # Send the function result back to the model
        follow_up_response = llm.generate(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What's the weather like in San Francisco?"
                        }
                    ]
                },
                {
                    "role": "assistant", 
                    "content": "",
                    "tool_calls": [
                        {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": json.dumps(tool_call.function.arguments)
                            }
                        }
                    ]
                },
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(weather_data)
                }
            ]
        )
        
        print("Model response with function results:")
        print(follow_up_response.message.content)
```

### Streaming Example

```python
from llm_hub import LLMHub

# Initialize LLM Hub
llm = LLMHub(provider="openai", api_key="sk-...", model="gpt-4o")

# Generate a streaming response
stream = llm.generate(
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Write a short poem about the moon."
                }
            ]
        }
    ],
    stream=True
)

# Process the stream
for chunk in stream:
    print(chunk.chunk, end="", flush=True)
```

### Structured Output Example

```python
from llm_hub import LLMHub
from llm_hub.core.types import ResponseFormat
from llm_hub.tools.structured_output import parse_response_as_model
from pydantic import BaseModel
from typing import List, Optional

# Initialize LLM Hub
llm = LLMHub(provider="openai", api_key="sk-...", model="gpt-4o")

# Define a Pydantic model for the output
class Ingredient(BaseModel):
    item: str
    quantity: str
    state: Optional[str] = None
    
class Recipe(BaseModel):
    name: str
    ingredients: List[Ingredient]
    instructions: List[str]
    servings: Optional[str] = None
    prep_time_minutes: Optional[int] = None

class RecipeResponse(BaseModel):
    recipe: Recipe

# Create a JSON schema for the model
output_format = ResponseFormat(
    type="json_object",
    schema=RecipeResponse.model_json_schema()
)

# Generate a structured response
response = llm.generate(
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Give me a recipe for chocolate chip cookies. Format your response as JSON."
                }
            ]
        }
    ],
    output_format=output_format
)

# Parse the response into the Pydantic model
recipe_response = parse_response_as_model(response.message.content, RecipeResponse)
recipe = recipe_response.recipe

# Print the structured data
print(f"Recipe: {recipe.name}")
print(f"Prep time: {recipe.prep_time_minutes} minutes")
print(f"Servings: {recipe.servings}")
print("Ingredients:")
for ingredient in recipe.ingredients:
    ingredient_str = f"- {ingredient.quantity} {ingredient.item}"
    if ingredient.state:
        ingredient_str += f" ({ingredient.state})"
    print(ingredient_str)
print("Instructions:")
for i, step in enumerate(recipe.instructions, 1):
    print(f"{i}. {step}")
```

### Error Handling Example

```python
from llm_hub import LLMHub
from llm_hub.core.exceptions import (
    RateLimitError,
    TimeoutError,
    TokenLimitError,
    AuthenticationError,
    ProviderError,
    LLMHubError,
)

# Initialize LLM Hub
llm = LLMHub(
    provider="openai",
    api_key="sk-...",
    model="gpt-4o",
)

try:
    response = llm.generate(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Hello, world!"
                    }
                ]
            }
        ]
    )
    print(response.message.content)
    
except AuthenticationError as e:
    print(f"Authentication error: {e}")
    # In a real application, you would handle this by checking API keys
    
except RateLimitError as e:
    print(f"Rate limit exceeded: {e}")
    # In a real application, you would implement a backoff strategy
    
except TimeoutError as e:
    print(f"Request timed out: {e}")
    # In a real application, you would retry with a longer timeout
    
except TokenLimitError as e:
    print(f"Token limit exceeded: {e}")
    # In a real application, you would implement a chunking strategy
    
except ProviderError as e:
    print(f"Provider error: {e}")
    # In a real application, you would handle provider-specific issues
    
except LLMHubError as e:
    print(f"General error: {e}")
    # In a real application, you would fall back to default behavior
```

### Asynchronous API

LLM Hub also provides an asynchronous API for use with `async`/`await`:

```python
import asyncio
from llm_hub import LLMHub

async def main():
    llm = LLMHub(
        provider="openai",
        api_key="sk-...",
        model="gpt-4o",
        tracing=True,
        cost_tracking=True
    )
    
    # Simple async text generation
    response = await llm.agenerate(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Explain quantum computing in simple terms."
                    }
                ]
            }
        ]
    )
    print(response.message.content)
    
    # Run multiple requests in parallel
    results = await asyncio.gather(
        llm.agenerate(
            messages=[{"role": "user", "content": [{"type": "text", "text": "Write a haiku about the ocean."}]}]
        ),
        llm.agenerate(
            messages=[{"role": "user", "content": [{"type": "text", "text": "Give me 3 quick tips for productive programming."}]}]
        ),
        llm.agenerate(
            messages=[{"role": "user", "content": [{"type": "text", "text": "Explain async/await in Python briefly."}]}]
        )
    )
    
    for i, response in enumerate(results, 1):
        print(f"\nResult {i}:\n{response.message.content}")

# Run the async example
asyncio.run(main())
```

### Thinking Steps Example

```python
from llm_hub import LLMHub
from llm_hub.core.types import ReasoningConfig

# Initialize LLM Hub
llm = LLMHub(provider="openai", api_key="sk-...", model="gpt-4o")

# Generate a response with reasoning
stream = llm.generate(
    instructions="You are a helpful assistant that thinks step-by-step.",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Solve this math problem: If a company's revenue grew by 15% to $230,000, what was the original revenue?"
                }
            ]
        }
    ],
    # Configure reasoning to get detailed thinking steps
    reasoning=ReasoningConfig(
        effort="high",  # Request high-effort reasoning
        max_tokens=2000  # Allow sufficient tokens for detailed steps
    ),
    stream=True  # Enable streaming to see thinking in real-time
)

# Process the stream and display thinking in real-time
for chunk in stream:
    print(chunk.chunk, end="", flush=True)
```

## Supported Providers

- **OpenAI** (GPT models) - Supports tool calling, vision, and structured output
- **Claude** (via Anthropic) - Supports tool use with Claude 3.5, vision capabilities, and structured output
- **Gemini** (Google) - Supports multimodal capabilities
- **LiteLLM** (for access to 100+ additional LLMs)
