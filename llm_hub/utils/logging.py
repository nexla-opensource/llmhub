"""
Logging utilities for LLM Hub
"""

import datetime
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Union

# Set up the default logger
logger = logging.getLogger("llm_hub")


class LLMHubLogFormatter(logging.Formatter):
    """
    Custom formatter for LLM Hub logs
    """
    
    def __init__(self, include_timestamp: bool = True, include_level: bool = True):
        """
        Initialize the formatter
        
        Args:
            include_timestamp: Whether to include timestamps in log messages
            include_level: Whether to include log levels in log messages
        """
        self.include_timestamp = include_timestamp
        self.include_level = include_level
        super().__init__()
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format a log record
        
        Args:
            record: Log record to format
            
        Returns:
            Formatted log string
        """
        # Check if the message is already JSON
        try:
            # Parse the message as JSON
            message_dict = json.loads(record.msg)
            is_json = True
        except (json.JSONDecodeError, TypeError):
            # Not JSON, treat as a regular string
            message_dict = {"message": record.msg}
            is_json = False
        
        # Add timestamp if requested
        if self.include_timestamp:
            message_dict["timestamp"] = datetime.datetime.fromtimestamp(
                record.created
            ).isoformat()
        
        # Add log level if requested
        if self.include_level:
            message_dict["level"] = record.levelname
        
        # Add logger name
        message_dict["logger"] = record.name
        
        # Format as JSON if the original message was JSON
        if is_json:
            return json.dumps(message_dict)
        else:
            # Format as a string if the original message was a string
            parts = []
            
            if self.include_timestamp:
                parts.append(message_dict.pop("timestamp"))
            
            if self.include_level:
                parts.append(f"[{message_dict.pop('level')}]")
            
            parts.append(f"({message_dict.pop('logger')})")
            parts.append(message_dict.pop("message"))
            
            # Add any remaining fields
            for key, value in message_dict.items():
                parts.append(f"{key}={value}")
            
            return " ".join(parts)


def configure_logging(
    level: int = logging.INFO,
    console: bool = True,
    log_file: Optional[str] = None,
    json_format: bool = False,
) -> None:
    """
    Configure logging for LLM Hub
    
    Args:
        level: Logging level
        console: Whether to log to console
        log_file: Path to log file (if None, no file logging)
        json_format: Whether to use JSON format for logs
    """
    # Create logger
    root_logger = logging.getLogger("llm_hub")
    root_logger.setLevel(level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter
    if json_format:
        formatter = logging.Formatter("%(message)s")
    else:
        formatter = LLMHubLogFormatter()
    
    # Add console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # Add file handler
    if log_file:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def log_json(
    logger_instance: logging.Logger,
    level: int,
    data: Dict[str, Any],
) -> None:
    """
    Log a dictionary as JSON
    
    Args:
        logger_instance: Logger instance to use
        level: Logging level
        data: Dictionary to log
    """
    # Convert data to JSON and log it
    json_str = json.dumps(data)
    logger_instance.log(level, json_str)


def log_request(
    request_id: str,
    provider: str,
    model: str,
    prompt: Union[str, List[Dict[str, Any]]],
    metadata: Optional[Dict[str, Any]] = None,
    level: int = logging.INFO,
) -> None:
    """
    Log an LLM request
    
    Args:
        request_id: Unique request ID
        provider: Provider name
        model: Model name
        prompt: Text prompt or message list
        metadata: Additional metadata
        level: Logging level
    """
    log_data = {
        "event": "llm_request",
        "request_id": request_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "provider": provider,
        "model": model,
        "prompt": prompt,
    }
    
    if metadata:
        log_data["metadata"] = metadata
    
    log_json(logger, level, log_data)


def log_response(
    request_id: str,
    provider: str,
    model: str,
    response: Any,
    latency: float,
    usage: Optional[Dict[str, int]] = None,
    cost: Optional[Dict[str, float]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    level: int = logging.INFO,
) -> None:
    """
    Log an LLM response
    
    Args:
        request_id: Unique request ID
        provider: Provider name
        model: Model name
        response: Response content
        latency: Response time in seconds
        usage: Token usage information
        cost: Cost information
        metadata: Additional metadata
        level: Logging level
    """
    log_data = {
        "event": "llm_response",
        "request_id": request_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "provider": provider,
        "model": model,
        "latency": latency,
    }
    
    # Add content if it's a simple value
    if isinstance(response, (str, int, float, bool)) or response is None:
        log_data["content"] = response
    else:
        # For complex objects, try to get a string representation
        try:
            log_data["content"] = str(response)
        except Exception:
            log_data["content"] = "Complex object (not serializable)"
    
    if usage:
        log_data["usage"] = usage
    
    if cost:
        log_data["cost"] = cost
    
    if metadata:
        log_data["metadata"] = metadata
    
    log_json(logger, level, log_data)


def log_error(
    request_id: str,
    provider: str,
    error_type: str,
    error_message: str,
    metadata: Optional[Dict[str, Any]] = None,
    level: int = logging.ERROR,
) -> None:
    """
    Log an LLM error
    
    Args:
        request_id: Unique request ID
        provider: Provider name
        error_type: Type of error
        error_message: Error message
        metadata: Additional metadata
        level: Logging level
    """
    log_data = {
        "event": "llm_error",
        "request_id": request_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "provider": provider,
        "error_type": error_type,
        "error_message": error_message,
    }
    
    if metadata:
        log_data["metadata"] = metadata
    
    log_json(logger, level, log_data)


class LoggingContext:
    """
    Context manager for request-scoped logging
    """
    
    def __init__(
        self,
        request_id: str,
        provider: str,
        model: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the logging context
        
        Args:
            request_id: Unique request ID
            provider: Provider name
            model: Model name
            metadata: Additional metadata
        """
        self.request_id = request_id
        self.provider = provider
        self.model = model
        self.metadata = metadata or {}
        self.start_time = None
    
    def __enter__(self):
        """
        Enter the context
        """
        self.start_time = datetime.datetime.now()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the context
        """
        # If an exception occurred, log it
        if exc_type is not None:
            log_error(
                request_id=self.request_id,
                provider=self.provider,
                error_type=exc_type.__name__,
                error_message=str(exc_val),
                metadata=self.metadata,
            )
