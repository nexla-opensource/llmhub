"""
Structured output handling utilities for LLM Hub
"""

import json
import re
from typing import Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel, ValidationError

from ..core.exceptions import StructuredOutputError


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


def parse_response_as_model(
    response_text: str, 
    model_class: Type[BaseModel]
) -> BaseModel:
    """
    Parse a text response as a Pydantic model
    
    Args:
        response_text: Text response from the LLM
        model_class: Pydantic model class to parse into
        
    Returns:
        Instantiated Pydantic model
        
    Raises:
        StructuredOutputError: If the response can't be parsed into the model
    """
    try:
        # First try to extract JSON from the response
        data = extract_json_from_text(response_text)
        
        # Then validate it against the model
        return model_class.parse_obj(data)
    
    except (StructuredOutputError, ValidationError) as e:
        raise StructuredOutputError(
            f"Failed to parse response as {model_class.__name__}: {str(e)}"
        )


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
    # Get the JSON schema
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
