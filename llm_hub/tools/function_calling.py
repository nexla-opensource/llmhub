"""
Function calling utilities for LLM Hub
"""

import inspect
import json
from typing import Any, Callable, Dict, List, Optional, Type, Union, get_type_hints

from pydantic import BaseModel, ValidationError, create_model

from ..core.exceptions import ToolError
from ..core.types import FunctionDefinition, FunctionParameter, FunctionTool


def function_to_tool(
    func: Callable,
    description: Optional[str] = None,
    parameter_descriptions: Optional[Dict[str, str]] = None
) -> FunctionTool:
    """
    Convert a Python function to a function tool definition
    
    Args:
        func: The function to convert
        description: Optional function description (falls back to docstring)
        parameter_descriptions: Optional descriptions for parameters
        
    Returns:
        FunctionTool object for the function
        
    Raises:
        ToolError: If the function cannot be converted
    """
    try:
        # Get function name
        func_name = func.__name__
        
        # Get function description from docstring if not provided
        if description is None:
            description = inspect.getdoc(func) or f"Function {func_name}"
        
        # Get parameter info
        signature = inspect.signature(func)
        type_hints = get_type_hints(func)
        
        # Create properties dictionary for parameters
        properties = {}
        required = []
        
        for param_name, param in signature.parameters.items():
            # Skip self for methods
            if param_name == "self":
                continue
            
            param_type = type_hints.get(param_name, Any)
            param_description = parameter_descriptions.get(param_name, "") if parameter_descriptions else ""
            
            # Convert type to JSON schema type
            param_schema = _type_to_json_schema(param_type)
            
            # Add description if provided
            if param_description:
                param_schema["description"] = param_description
            
            # Add to properties dictionary
            properties[param_name] = param_schema
            
            # Add to required list if no default value
            if param.default == inspect.Parameter.empty:
                required.append(param_name)
        
        # Create function parameter object
        parameters = FunctionParameter(
            type="object",
            properties=properties,
            required=required
        )
        
        # Create function definition
        function_def = FunctionDefinition(
            name=func_name,
            description=description,
            parameters=parameters
        )
        
        # Create and return function tool
        return FunctionTool(
            type="function",
            function=function_def
        )
    
    except Exception as e:
        raise ToolError(f"Failed to convert function to tool: {str(e)}")


def _type_to_json_schema(type_hint: Any) -> Dict[str, Any]:
    """
    Convert a Python type hint to a JSON schema
    
    Args:
        type_hint: Type hint to convert
        
    Returns:
        JSON schema for the type
    """
    # Handle basic types
    if type_hint == str:
        return {"type": "string"}
    elif type_hint == int:
        return {"type": "integer"}
    elif type_hint == float:
        return {"type": "number"}
    elif type_hint == bool:
        return {"type": "boolean"}
    elif type_hint == list or type_hint == List:
        return {"type": "array", "items": {}}
    elif type_hint == dict or type_hint == Dict:
        return {"type": "object"}
    elif type_hint == Any:
        return {}
    
    # Handle Optional types
    if hasattr(type_hint, "__origin__") and type_hint.__origin__ is Union:
        args = type_hint.__args__
        if len(args) == 2 and args[1] is type(None):
            # It's an Optional[...]
            return _type_to_json_schema(args[0])
    
    # Handle List[...] and Dict[...]
    if hasattr(type_hint, "__origin__"):
        if type_hint.__origin__ is list or type_hint.__origin__ is List:
            item_type = type_hint.__args__[0]
            return {
                "type": "array",
                "items": _type_to_json_schema(item_type)
            }
        elif type_hint.__origin__ is dict or type_hint.__origin__ is Dict:
            if type_hint.__args__[0] is str:
                value_type = type_hint.__args__[1]
                return {
                    "type": "object",
                    "additionalProperties": _type_to_json_schema(value_type)
                }
    
    # Handle Pydantic models
    if isinstance(type_hint, type) and issubclass(type_hint, BaseModel):
        return type_hint.schema()
    
    # Fallback for anything else
    return {"type": "string"}


def pydantic_model_to_function_tool(
    model_class: Type[BaseModel],
    name: str,
    description: Optional[str] = None
) -> FunctionTool:
    """
    Convert a Pydantic model to a function tool definition
    
    Args:
        model_class: Pydantic model class
        name: Function name
        description: Function description
        
    Returns:
        FunctionTool object for the model
        
    Raises:
        ToolError: If the model cannot be converted
    """
    try:
        # Get schema from the model
        schema = model_class.schema()
        
        # Extract properties and required fields
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        
        # Create function parameter object
        parameters = FunctionParameter(
            type="object",
            properties=properties,
            required=required
        )
        
        # Use model description if not provided
        if description is None:
            description = schema.get("description", f"Function {name}")
        
        # Create function definition
        function_def = FunctionDefinition(
            name=name,
            description=description,
            parameters=parameters
        )
        
        # Create and return function tool
        return FunctionTool(
            type="function",
            function=function_def
        )
    
    except Exception as e:
        raise ToolError(f"Failed to convert Pydantic model to tool: {str(e)}")


def execute_function_call(
    function_call: Dict[str, Any],
    available_functions: Dict[str, Callable]
) -> Any:
    """
    Execute a function call from an LLM
    
    Args:
        function_call: Function call data from the LLM
        available_functions: Dictionary of available functions (name -> function)
        
    Returns:
        Result of the function call
        
    Raises:
        ToolError: If the function call cannot be executed
    """
    try:
        # Get function name
        function_name = function_call.get("name")
        if not function_name:
            raise ToolError("Function call missing name")
        
        # Get function arguments
        function_args = function_call.get("arguments", {})
        
        # Convert arguments to Python types if they're a string
        if isinstance(function_args, str):
            try:
                function_args = json.loads(function_args)
            except json.JSONDecodeError:
                raise ToolError(f"Invalid function arguments: {function_args}")
        
        # Get the function
        if function_name not in available_functions:
            raise ToolError(f"Function '{function_name}' not found")
        
        function = available_functions[function_name]
        
        # Call the function with the arguments
        return function(**function_args)
    
    except Exception as e:
        # Re-raise ToolError
        if isinstance(e, ToolError):
            raise
        
        # Wrap other exceptions
        raise ToolError(f"Error executing function call: {str(e)}")


def create_tool_registry(
    functions: Dict[str, Callable],
    descriptions: Optional[Dict[str, str]] = None,
    parameter_descriptions: Optional[Dict[str, Dict[str, str]]] = None
) -> Dict[str, Any]:
    """
    Create a registry of tools from functions
    
    Args:
        functions: Dictionary of functions (name -> function)
        descriptions: Optional descriptions for functions
        parameter_descriptions: Optional descriptions for function parameters
        
    Returns:
        A dictionary with tools and available_functions
    """
    tools = []
    
    for name, func in functions.items():
        description = descriptions.get(name) if descriptions else None
        params_desc = parameter_descriptions.get(name) if parameter_descriptions else None
        
        tool = function_to_tool(func, description, params_desc)
        tools.append(tool)
    
    return {
        "tools": tools,
        "available_functions": functions
    }


def validate_and_execute_tool_calls(
    tool_calls: List[Dict[str, Any]],
    available_functions: Dict[str, Callable]
) -> List[Dict[str, Any]]:
    """
    Validate and execute multiple tool calls
    
    Args:
        tool_calls: List of tool calls from the LLM
        available_functions: Dictionary of available functions
        
    Returns:
        List of tool call results
    """
    results = []
    
    for call in tool_calls:
        # Validate tool call format
        if not isinstance(call, dict):
            raise ToolError(f"Invalid tool call format: {call}")
        
        # Extract function call
        if "function" not in call:
            raise ToolError(f"Tool call missing function: {call}")
        
        function_call = call["function"]
        
        # Execute the function
        try:
            result = execute_function_call(function_call, available_functions)
            
            # Add result to list
            results.append({
                "tool_call_id": call.get("id", "unknown"),
                "function_name": function_call.get("name", "unknown"),
                "result": result
            })
        
        except Exception as e:
            results.append({
                "tool_call_id": call.get("id", "unknown"),
                "function_name": function_call.get("name", "unknown"),
                "error": str(e)
            })
    
    return results


def format_tool_results_for_llm(
    tool_results: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Format tool call results for sending back to an LLM
    
    Args:
        tool_results: Results from validate_and_execute_tool_calls
        
    Returns:
        List of formatted messages
    """
    messages = []
    
    for result in tool_results:
        # Create a message for each tool call
        content = None
        
        if "error" in result:
            # Format error response
            content = f"Error: {result['error']}"
        else:
            # Format successful response
            # Convert complex objects to JSON
            if isinstance(result["result"], (dict, list)):
                content = json.dumps(result["result"])
            else:
                content = str(result["result"])
        
        messages.append({
            "role": "tool",
            "tool_call_id": result["tool_call_id"],
            "content": content
        })
    
    return messages
