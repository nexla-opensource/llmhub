"""
LLM Hub - Usage Examples

This script demonstrates various ways to use the LLM Hub package.
"""

import os
import json
import asyncio
from typing import Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

from llm_hub import LLMHub
from llm_hub.core.types import (
    TextContent,
    ImageContent,
    DocumentContent,
    FileData,
    FunctionTool,
    ResponseFormat,
    Message,
    Role,
    ReasoningConfig,
)
from llm_hub.core.exceptions import (
    RateLimitError,
    TimeoutError,
    TokenLimitError,
    AuthenticationError,
    ProviderError,
    LLMHubError,
)
from llm_hub.tools.structured_output import parse_response_as_model
from pydantic import BaseModel


def basic_example(llm):
    """
    Basic example of using LLM Hub with OpenAI
    """
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
    print("\n=== Basic Example Response ===")
    print(response.message.content)
    
    # Print usage and cost information
    usage = response.metadata.usage
    cost = response.metadata.cost
    print(f"\nUsage: {usage.total_tokens} tokens ({usage.prompt_tokens} prompt, {usage.completion_tokens} completion)")
    print(f"Cost: ${cost.total_cost:.6f} (${cost.prompt_cost:.6f} prompt, ${cost.completion_cost:.6f} completion)")


def multimodal_example(llm):
    """
    Example of using LLM Hub with multimodal inputs (text + image)
    """
    # Create image content
    image_url = "https://nexla.com/n3x_ctx/uploads/2024/10/main-og.png"
    
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
                        image_url=image_url,
                        detail="high"
                    )
                ]
            )
        ]
    )
    
    # Option 2: Using a local image file
    # response = llm.generate(
    #     messages=[
    #         Message(
    #             role=Role.USER,
    #             content=[
    #                 TextContent(
    #                     type="text",
    #                     text="What can you tell me about this painting?"
    #                 ),
    #                 ImageContent(
    #                     type="image",
    #                     image_path="/path/to/local/image.jpg",
    #                     detail="high"
    #                 )
    #             ]
    #         )
    #     ]
    # )
    
    # Print the response
    print("\n=== Multimodal Example Response ===")
    print(response.message.content)


def function_calling_example(llm):
    """
    Example of using LLM Hub with function calling
    """
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
    
    # Print the response
    print("\n=== Function Calling Example Response ===")
    print(response.message.content)
    
    # Check if the model called a function
    if response.message.tool_calls:
        for tool_call in response.message.tool_calls:
            print(f"\nFunction called: {tool_call.function.name}")
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
            
            print("\nModel response with function results:")
            print(follow_up_response.message.content)


def streaming_example(llm):
    """
    Example of using LLM Hub with streaming responses
    """
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
    print("\n=== Streaming Example Response ===")
    for chunk in stream:
        print(chunk.chunk, end="", flush=True)
    print("\n")


async def async_example(llm):
    """
    Example of using LLM Hub with async API
    """
    # Generate a response asynchronously
    response = await llm.agenerate(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Explain quantum computing briefly."
                    }
                ]
            }
        ]
    )
    
    # Print the response
    print("\n=== Async Example Response ===")
    print(response.message.content)


def structured_output_example(llm):
    """
    Example of using LLM Hub with structured output
    """
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
    
    # Print the raw response
    print("\n=== Structured Output Example Response ===")
    print(f"Raw response: {response.message.content}")
    
    # Parse the response into the Pydantic model
    recipe_response = parse_response_as_model(response.message.content, RecipeResponse)
    recipe = recipe_response.recipe
    
    # Print the structured data
    print(f"\nRecipe: {recipe.name}")
    if recipe.prep_time_minutes:
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


def error_handling_example(provider, model):
    """
    Example of error handling with LLM Hub
    """
    # Initialize LLM Hub with an invalid API key to trigger an error
    llm = LLMHub(
        provider=provider,
        api_key="invalid-api-key",  # Invalid API key to trigger an error
        model=model,
    )
    
    print("\n=== Error Handling Example ===")
    
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


def thinking_streaming_example(llm):
    """
    Example of using LLM Hub with reasoning configuration and streaming the thinking process
    """
    # Generate a response with reasoning and stream the output
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
    print("\n=== Thinking Streaming Example Response ===")
    for chunk in stream:
        print(chunk.chunk, end="", flush=True)
    print("\n")


def advanced_thinking_example(llm):
    """
    Advanced example of extracting and processing intermediate thinking steps
    from a streaming response
    """
    # Set up custom system prompt to explicitly request thinking steps
    system_prompt = """
    You are a helpful assistant that always shows your thinking process.
    When solving problems:
    1. First, outline your thinking process clearly marked between <thinking> and </thinking> tags
    2. Then provide your final answer
    
    Always show your step-by-step reasoning before giving the final answer.
    """
    
    # Generate a streaming response with a complex problem
    stream = llm.generate(
        instructions=system_prompt,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "A store sells widgets for $12 each and gadgets for $8 each. If I bought 10 items total for $104, how many of each item did I buy?"
                    }
                ]
            }
        ],
        reasoning=ReasoningConfig(
            effort="high",
            max_tokens=3000
        ),
        stream=True
    )
    
    # Process the stream with separate handling for thinking steps and final answer
    print("\n=== Advanced Thinking Example Response ===")
    
    thinking_mode = False
    thinking_content = ""
    final_answer = ""
    
    # Process stream chunks and separate thinking from final answer
    for chunk in stream:
        content = chunk.chunk
        
        # Check for thinking tags in the content
        if "<thinking>" in content:
            thinking_mode = True
            content = content.replace("<thinking>", "")
            print("\n--- Thinking Steps ---\n", end="")
        
        if "</thinking>" in content:
            thinking_mode = False
            content = content.replace("</thinking>", "")
            print(content, end="")
            print("\n\n--- Final Answer ---\n", end="")
            continue
        
        # Store content in appropriate buffer
        if thinking_mode:
            thinking_content += content
            # Print thinking in a different color or format
            print(content, end="", flush=True)
        else:
            final_answer += content
            print(content, end="", flush=True)
    
    print("\n")
    
    # Example of post-processing the thinking steps
    if thinking_content:
        # Count the number of calculation steps
        steps = thinking_content.split("\n")
        calculation_steps = [s for s in steps if any(op in s for op in ["+", "-", "*", "/", "="])]
        
        print(f"\nAnalysis: The solution involved {len(calculation_steps)} calculation steps.")
        print(f"Total thinking: {len(thinking_content)} characters")
        print(f"Final answer: {len(final_answer)} characters")


def multiple_providers_example():
    """
    Example of using LLM Hub with multiple providers
    """
    print("\n=== Multiple Providers Example ===")
    
    # Define a simple question
    question = "What is the capital of France?"
    
    # List of providers to try
    providers = [
        {
            "name": "OpenAI",
            "provider": "openai",
            "api_key": os.environ.get("OPENAI_API_KEY"),
            "model": "gpt-4o",
        },
        {
            "name": "Anthropic",
            "provider": "anthropic",
            "api_key": os.environ.get("ANTHROPIC_API_KEY"),
            "model": "claude-3.7-sonnet",
        },
        {
            "name": "Gemini",
            "provider": "gemini",
            "api_key": os.environ.get("GOOGLE_API_KEY"),
            "model": "gemini-2.5-pro-preview-03-25",
        }
    ]
    
    # Check if any API keys are available
    available_providers = [p for p in providers if p["api_key"]]
    if not available_providers:
        print("No API keys available for any providers. Skipping example.")
        return
    
    # Try each provider
    for provider_info in providers:
        try:
            # Skip if API key is not available
            if not provider_info["api_key"]:
                print(f"{provider_info['name']}: API key not available, skipping")
                continue
            
            # Initialize LLM Hub with this provider
            llm = LLMHub(
                provider=provider_info["provider"],
                api_key=provider_info["api_key"],
                model=provider_info["model"],
            )
            
            # Generate a response
            response = llm.generate(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": question
                            }
                        ]
                    }
                ]
            )
            
            # Print the response
            print(f"\n{provider_info['name']} ({provider_info['model']}) response:")
            print(response.message.content)
            
            # Print usage and cost
            usage = response.metadata.usage
            cost = response.metadata.cost
            print(f"Usage: {usage.total_tokens} tokens")
            print(f"Cost: ${cost.total_cost:.6f}")
            
        except Exception as e:
            print(f"{provider_info['name']} error: {e}")


def main():
    """
    Run all examples
    """
    print("\n=== LLM Hub Examples ===")
    
    # Set the provider and model once here
    provider = "openai"  # Change to your preferred provider: "openai", "anthropic", etc.
    
    # Select appropriate model based on provider
    models = {
        "openai": "gpt-4o",
        "anthropic": "claude-3.7-sonnet",
        "gemini": "gemini-2.5-pro",
        # Add other providers and their default models here
    }
    model = models.get(provider)
    
    # Get the appropriate API key based on provider
    api_keys = {
        "openai": os.environ.get("OPENAI_API_KEY"),
        "anthropic": os.environ.get("ANTHROPIC_API_KEY"),
        "gemini": os.environ.get("GOOGLE_API_KEY"),
    }
    api_key = api_keys.get(provider)
    
    if not api_key:
        print(f"No API key found for provider '{provider}'. Please check your .env file.")
        return
    
    print(f"Running examples using provider: {provider}, model: {model}")
    
    # Initialize LLM Hub once
    llm = LLMHub(
        provider=provider,
        api_key=api_key,
        model=model,
        tracing=True,
        cost_tracking=True,
    )
    
    # Run examples with the initialized LLM instance
    # Uncomment examples you want to run
    
    # Basic example
    basic_example(llm)
    
    # Multimodal example - requires image-capable model
    multimodal_example(llm)
    
    # Function calling example - requires function-calling capable model
    function_calling_example(llm)
    
    # Streaming example
    streaming_example(llm)
    
    # Structured output example
    structured_output_example(llm)
    
    # Error handling example - can run without valid API key
    error_handling_example(provider, model)
    
    # Thinking streaming example - requires reasoning-capable model
    # thinking_streaming_example(llm)
    
    # Advanced thinking example - requires reasoning-capable model
    # advanced_thinking_example(llm)
    
    # Multiple providers example - can still be run separately
    # multiple_providers_example()
    
    # Async example
    # asyncio.run(async_example(llm))
    
    print("\n=== Examples Completed ===")


if __name__ == "__main__":
    main()