"""
Streaming response utilities for LLM Hub
"""

import asyncio
import json
from typing import Any, AsyncGenerator, Callable, Dict, Generator, Iterator, Optional, Union

from ..core.exceptions import StreamingError
from ..core.types import StreamingResponse


def stream_to_string(stream: Iterator[StreamingResponse]) -> str:
    """
    Convert a streaming response to a complete string
    
    Args:
        stream: Iterator of streaming responses
        
    Returns:
        Complete combined string from all chunks
        
    Raises:
        StreamingError: If there's an error processing the stream
    """
    try:
        # Combine all chunks into a string
        result = ""
        for chunk in stream:
            result += chunk.chunk
        return result
    
    except Exception as e:
        raise StreamingError(f"Error processing stream: {str(e)}")


async def astream_to_string(stream: AsyncGenerator[StreamingResponse, None]) -> str:
    """
    Convert an async streaming response to a complete string
    
    Args:
        stream: Async generator of streaming responses
        
    Returns:
        Complete combined string from all chunks
        
    Raises:
        StreamingError: If there's an error processing the stream
    """
    try:
        # Combine all chunks into a string
        result = ""
        async for chunk in stream:
            result += chunk.chunk
        return result
    
    except Exception as e:
        raise StreamingError(f"Error processing async stream: {str(e)}")


def process_stream_with_callback(
    stream: Iterator[StreamingResponse],
    callback: Callable[[str, bool], None]
) -> str:
    """
    Process a streaming response with a callback
    
    Args:
        stream: Iterator of streaming responses
        callback: Function that takes a chunk string and a boolean indicating if it's the final chunk
        
    Returns:
        Complete combined string from all chunks
        
    Raises:
        StreamingError: If there's an error processing the stream
    """
    try:
        # Combine all chunks into a string
        result = ""
        for chunk in stream:
            callback(chunk.chunk, chunk.finished)
            result += chunk.chunk
        return result
    
    except Exception as e:
        raise StreamingError(f"Error processing stream with callback: {str(e)}")


async def aprocess_stream_with_callback(
    stream: AsyncGenerator[StreamingResponse, None],
    callback: Callable[[str, bool], Any]
) -> str:
    """
    Process an async streaming response with a callback
    
    Args:
        stream: Async generator of streaming responses
        callback: Function that takes a chunk string and a boolean indicating if it's the final chunk
        
    Returns:
        Complete combined string from all chunks
        
    Raises:
        StreamingError: If there's an error processing the stream
    """
    try:
        # Combine all chunks into a string
        result = ""
        async for chunk in stream:
            # If the callback is a coroutine, await it
            if asyncio.iscoroutinefunction(callback):
                await callback(chunk.chunk, chunk.finished)
            else:
                callback(chunk.chunk, chunk.finished)
            
            result += chunk.chunk
        return result
    
    except Exception as e:
        raise StreamingError(f"Error processing async stream with callback: {str(e)}")


def json_stream_to_object(stream: Iterator[StreamingResponse]) -> Dict[str, Any]:
    """
    Convert a JSON streaming response to a complete object
    
    Args:
        stream: Iterator of streaming responses containing JSON
        
    Returns:
        Parsed JSON object
        
    Raises:
        StreamingError: If the stream doesn't contain valid JSON
    """
    try:
        # Get the complete string
        json_str = stream_to_string(stream)
        
        # Parse as JSON
        return json.loads(json_str)
    
    except json.JSONDecodeError as e:
        raise StreamingError(f"Error parsing JSON from stream: {str(e)}")
    
    except Exception as e:
        raise StreamingError(f"Error processing JSON stream: {str(e)}")


async def ajson_stream_to_object(stream: AsyncGenerator[StreamingResponse, None]) -> Dict[str, Any]:
    """
    Convert an async JSON streaming response to a complete object
    
    Args:
        stream: Async generator of streaming responses containing JSON
        
    Returns:
        Parsed JSON object
        
    Raises:
        StreamingError: If the stream doesn't contain valid JSON
    """
    try:
        # Get the complete string
        json_str = await astream_to_string(stream)
        
        # Parse as JSON
        return json.loads(json_str)
    
    except json.JSONDecodeError as e:
        raise StreamingError(f"Error parsing JSON from async stream: {str(e)}")
    
    except Exception as e:
        raise StreamingError(f"Error processing async JSON stream: {str(e)}")


def streaming_generator_to_async(
    stream: Iterator[StreamingResponse]
) -> AsyncGenerator[StreamingResponse, None]:
    """
    Convert a synchronous streaming generator to an async generator
    
    Args:
        stream: Synchronous iterator of streaming responses
        
    Returns:
        Async generator of the same streaming responses
    """
    
    async def wrapper():
        for chunk in stream:
            yield chunk
    
    return wrapper()


def create_streaming_response_generator(
    chunks: Iterator[str],
    metadata: Optional[Dict[str, Any]] = None
) -> Iterator[StreamingResponse]:
    """
    Create a streaming response generator from string chunks
    
    Args:
        chunks: Iterator of string chunks
        metadata: Optional metadata to include in the final chunk
        
    Returns:
        Generator of StreamingResponse objects
    """
    last_chunk = None
    
    for chunk in chunks:
        # If we had a previous chunk, yield it (not the final one)
        if last_chunk is not None:
            yield StreamingResponse(chunk=last_chunk, finished=False, metadata=None)
        
        # Store this chunk
        last_chunk = chunk
    
    # Yield the final chunk with metadata and finished=True
    if last_chunk is not None:
        yield StreamingResponse(chunk=last_chunk, finished=True, metadata=metadata)
    else:
        # Handle empty streams
        yield StreamingResponse(chunk="", finished=True, metadata=metadata)


async def acreate_streaming_response_generator(
    chunks: AsyncGenerator[str, None],
    metadata: Optional[Dict[str, Any]] = None
) -> AsyncGenerator[StreamingResponse, None]:
    """
    Create an async streaming response generator from string chunks
    
    Args:
        chunks: Async generator of string chunks
        metadata: Optional metadata to include in the final chunk
        
    Returns:
        Async generator of StreamingResponse objects
    """
    last_chunk = None
    
    async for chunk in chunks:
        # If we had a previous chunk, yield it (not the final one)
        if last_chunk is not None:
            yield StreamingResponse(chunk=last_chunk, finished=False, metadata=None)
        
        # Store this chunk
        last_chunk = chunk
    
    # Yield the final chunk with metadata and finished=True
    if last_chunk is not None:
        yield StreamingResponse(chunk=last_chunk, finished=True, metadata=metadata)
    else:
        # Handle empty streams
        yield StreamingResponse(chunk="", finished=True, metadata=metadata)
