"""
Structured output handling utilities for LLM Hub
"""

import json
import re
from typing import Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel, ValidationError

from ..core.exceptions import StructuredOutputError
from ..core.types import ResponseFormat, ResponseFormatType
from ..core.exceptions import LLMHubError


def extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    Extract JSON from a text response
    
    Args:
        text: Text containing JSON (possibly with non-JSON content)
        
    Returns:
        Parsed JSON as a dictionary
        
    Raises:
        StructuredOutputError: If no valid JSON can be extracted
    """
    # Try to parse the entire text as JSON first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Look for JSON inside code blocks (Markdown style)
    code_block_pattern = r"```(?:json)?\s*(.+?)```"
    matches = re.findall(code_block_pattern, text, re.DOTALL)
    
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    
    # Look for JSON enclosed in curly braces
    brace_pattern = r"\{.*\}"
    matches = re.findall(brace_pattern, text, re.DOTALL)
    
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    
    # If we couldn't extract JSON, raise an error
    raise StructuredOutputError("Could not extract valid JSON from the response")


def parse_response_as_model(response_text: str, model_class: Type[BaseModel]) -> BaseModel:
    """
    Parse a model response into a Pydantic model
    
    Args:
        response_text: The text response from the model
        model_class: The Pydantic model class to parse into
        
    Returns:
        An instance of the provided model class
        
    Raises:
        LLMHubError: If the response cannot be parsed into the model
    """
    try:
        # Try to parse the response as JSON
        json_data = json.loads(response_text)
        
        # Try to validate using the model
        return model_class.model_validate(json_data)
    
    except json.JSONDecodeError as e:
        raise LLMHubError(f"Failed to parse response as JSON: {str(e)}")
    
    except ValidationError as e:
        raise LLMHubError(f"Response does not match expected schema: {str(e)}")
    
    except Exception as e:
        raise LLMHubError(f"Unexpected error parsing response: {str(e)}")


def parse_response_as_list(response_text: str) -> List[Any]:
    """
    Parse a text response as a JSON list
    
    Args:
        response_text: Text response from the LLM
        
    Returns:
        Parsed list
        
    Raises:
        StructuredOutputError: If the response can't be parsed as a list
    """
    # Try to extract JSON from the response
    data = extract_json_from_text(response_text)
    
    # Check if it's a list
    if not isinstance(data, list):
        raise StructuredOutputError(
            f"Expected a list but got {type(data).__name__}"
        )
    
    return data


def create_json_schema(model_class: Type[BaseModel]) -> Dict[str, Any]:
    """
    Create a JSON schema from a Pydantic model for use with LLMs
    
    Args:
        model_class: Pydantic model class
        
    Returns:
        JSON schema dictionary
    """
    # Get the JSON schema using Pydantic v2 or v1 method
    try:
        # Pydantic v2 method
        schema = model_class.model_json_schema()
    except AttributeError:
        # Pydantic v1 method (fallback)
        schema = model_class.schema()
    
    # Remove Pydantic-specific fields
    schema.pop("title", None)
    
    return schema


def create_schema_prompt(
    model_class: Type[BaseModel], 
    description: Optional[str] = None
) -> str:
    """
    Create a prompt describing the expected output schema
    
    Args:
        model_class: Pydantic model class
        description: Optional description of the expected output
        
    Returns:
        Formatted prompt string
    """
    schema = create_json_schema(model_class)
    schema_str = json.dumps(schema, indent=2)
    
    prompt = "Please provide your response in the following JSON format:\n\n"
    prompt += schema_str + "\n\n"
    
    if description:
        prompt += description + "\n\n"
    
    prompt += "Ensure your response is valid JSON that exactly matches this schema."
    
    return prompt


def validate_json_against_schema(
    data: Dict[str, Any],
    schema: Dict[str, Any]
) -> None:
    """
    Validate JSON data against a JSON schema
    
    Args:
        data: JSON data to validate
        schema: JSON schema to validate against
        
    Raises:
        StructuredOutputError: If validation fails
    """
    try:
        from jsonschema import validate, ValidationError as JsonSchemaValidationError
        
        # Validate the data against the schema
        validate(instance=data, schema=schema)
    
    except ImportError:
        # jsonschema is optional, fall back to basic type checking
        if "type" in schema and schema["type"] == "object":
            if not isinstance(data, dict):
                raise StructuredOutputError(
                    f"Expected an object but got {type(data).__name__}"
                )
            
            # Check required properties
            if "required" in schema:
                for prop in schema["required"]:
                    if prop not in data:
                        raise StructuredOutputError(
                            f"Missing required property: {prop}"
                        )
    
    except JsonSchemaValidationError as e:
        raise StructuredOutputError(f"JSON validation failed: {str(e)}")


def convert_pydantic_to_json_schema(model_class: Type[BaseModel]) -> Dict[str, Any]:
    """
    Convert a Pydantic model to a JSON schema suitable for LLM function calling
    
    Args:
        model_class: Pydantic model class
        
    Returns:
        JSON schema dictionary formatted for function calling
    """
    # Get schema using either v1 or v2 API
    try:
        # Pydantic v2 method
        schema = model_class.model_json_schema()
    except AttributeError:
        # Pydantic v1 method (fallback)
        schema = model_class.schema()
    
    # Format for function calling
    result = {
        "type": "object",
        "properties": schema.get("properties", {}),
    }
    
    # Add required fields if present
    if "required" in schema:
        result["required"] = schema["required"]
    
    return result


def model_to_response_format(
    model_class: Type[BaseModel], 
    name: Optional[str] = None,
    strict: bool = True
) -> ResponseFormat:
    """
    Convert a Pydantic model to a ResponseFormat for use with LLM Hub
    
    Args:
        model_class: The Pydantic model class to convert
        name: Optional name for the schema (for documentation)
        strict: Whether to enforce strict schema validation
        
    Returns:
        A ResponseFormat object ready to use with generate methods
    """
    # Get the JSON schema for the model
    schema = model_class.model_json_schema()
    
    # Create a ResponseFormat object
    return ResponseFormat(
        type=ResponseFormatType.JSON_SCHEMA,
        schema=schema,
        name=name or model_class.__name__,
        strict=strict
    )


def create_schema_from_dict(schema_dict: Dict[str, Any], name: Optional[str] = None) -> ResponseFormat:
    """
    Create a ResponseFormat from a dictionary representing a JSON schema
    
    Args:
        schema_dict: Dictionary containing the JSON schema
        name: Optional name for the schema
        
    Returns:
        A ResponseFormat object ready to use with generate methods
        
    Raises:
        LLMHubError: If the schema is invalid
    """
    # Validate basic schema requirements
    if not isinstance(schema_dict, dict):
        raise LLMHubError("Schema must be a dictionary")
    
    if schema_dict.get("type") != "object":
        raise LLMHubError("Schema root must be an object type")
    
    if "required" not in schema_dict:
        raise LLMHubError("Schema must specify required fields")
    
    if "properties" not in schema_dict:
        raise LLMHubError("Schema must define properties")
    
    # Set additionalProperties to false if not specified
    if "additionalProperties" not in schema_dict:
        schema_dict["additionalProperties"] = False
    
    # Create and return the ResponseFormat
    return ResponseFormat(
        type=ResponseFormatType.JSON_SCHEMA,
        schema=schema_dict,
        name=name,
        strict=True
    )


def is_refusal_response(response_text: str) -> bool:
    """
    Check if a response appears to be a refusal from the model
    
    Args:
        response_text: The text response from the model
        
    Returns:
        Boolean indicating if this is likely a refusal
    """
    # Check if response is valid JSON
    try:
        json_data = json.loads(response_text)
        # If it parses as JSON, it's not a refusal
        return False
    except json.JSONDecodeError:
        # Not valid JSON, might be a refusal
        pass
    
    # Common refusal patterns
    refusal_patterns = [
        "I'm sorry",
        "I cannot",
        "I am not able",
        "I apologize",
        "I'm unable",
        "I don't have",
        "I'm not able",
    ]
    
    text_lower = response_text.lower()
    return any(pattern.lower() in text_lower for pattern in refusal_patterns)
