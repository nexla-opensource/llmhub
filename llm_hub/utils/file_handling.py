"""
File handling utilities for LLM Hub
"""

import base64
import io
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import requests

from ..core.exceptions import FileHandlingError


def get_file_mime_type(file_path: str) -> str:
    """
    Get the MIME type of a file
    
    Args:
        file_path: Path to the file
        
    Returns:
        MIME type string
    """
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        # Default to binary if we can't determine the type
        mime_type = "application/octet-stream"
    return mime_type


def read_file_as_base64(file_path: str) -> str:
    """
    Read a file and encode it as a base64 string
    
    Args:
        file_path: Path to the file
        
    Returns:
        Base64-encoded string of the file contents
        
    Raises:
        FileHandlingError: If the file cannot be read
    """
    try:
        with open(file_path, "rb") as f:
            file_bytes = f.read()
        return base64.b64encode(file_bytes).decode("utf-8")
    except Exception as e:
        raise FileHandlingError(f"Failed to read file '{file_path}': {str(e)}")


def download_file(url: str, output_path: Optional[str] = None) -> str:
    """
    Download a file from a URL
    
    Args:
        url: URL to download from
        output_path: Path to save the downloaded file (optional)
        
    Returns:
        Path to the downloaded file
        
    Raises:
        FileHandlingError: If the download fails
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # If no output path is specified, create a temporary file
        if output_path is None:
            # Try to get the filename from the URL
            filename = os.path.basename(url.split("?")[0])
            if not filename:
                # Use the content disposition header if available
                content_disposition = response.headers.get("Content-Disposition")
                if content_disposition and "filename=" in content_disposition:
                    filename = content_disposition.split("filename=")[1].strip('"\'')
                else:
                    # Fallback to a generic name
                    filename = "downloaded_file"
            
            # Create a temporary directory if it doesn't exist
            temp_dir = os.path.join(os.getcwd(), "temp")
            os.makedirs(temp_dir, exist_ok=True)
            
            output_path = os.path.join(temp_dir, filename)
        
        # Save the file
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return output_path
    
    except Exception as e:
        raise FileHandlingError(f"Failed to download file from '{url}': {str(e)}")


def file_to_base64_data_uri(file_path: str) -> str:
    """
    Convert a file to a base64 data URI
    
    Args:
        file_path: Path to the file
        
    Returns:
        Base64 data URI string
        
    Raises:
        FileHandlingError: If the file cannot be read
    """
    try:
        mime_type = get_file_mime_type(file_path)
        base64_data = read_file_as_base64(file_path)
        return f"data:{mime_type};base64,{base64_data}"
    except Exception as e:
        raise FileHandlingError(f"Failed to convert file to data URI: {str(e)}")


def extract_file_info(file_path: str) -> Dict[str, Any]:
    """
    Extract information about a file
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dictionary with file information
        
    Raises:
        FileHandlingError: If the file cannot be read
    """
    try:
        path = Path(file_path)
        
        if not path.exists():
            raise FileHandlingError(f"File does not exist: {file_path}")
        
        stats = path.stat()
        
        return {
            "name": path.name,
            "path": str(path.absolute()),
            "size": stats.st_size,
            "mime_type": get_file_mime_type(file_path),
            "extension": path.suffix,
            "last_modified": stats.st_mtime,
        }
    except Exception as e:
        if isinstance(e, FileHandlingError):
            raise
        raise FileHandlingError(f"Failed to extract file information: {str(e)}")


def split_data_uri(data_uri: str) -> Tuple[str, bytes]:
    """
    Split a data URI into MIME type and binary data
    
    Args:
        data_uri: Data URI string
        
    Returns:
        Tuple of (mime_type, data_bytes)
        
    Raises:
        FileHandlingError: If the data URI is invalid
    """
    try:
        # Check if it's a data URI
        if not data_uri.startswith("data:"):
            raise ValueError("Not a data URI")
        
        # Split at the first comma
        header, data = data_uri.split(",", 1)
        
        # Parse the header
        header = header[5:]  # remove 'data:'
        mime_type = header.split(";")[0] if ";" in header else ""
        
        # Check if data is base64 encoded
        is_base64 = ";base64" in header
        
        # Decode the data
        if is_base64:
            data_bytes = base64.b64decode(data)
        else:
            # URL-decode the data
            from urllib.parse import unquote
            data_bytes = unquote(data).encode("utf-8")
        
        return mime_type, data_bytes
    
    except Exception as e:
        raise FileHandlingError(f"Failed to split data URI: {str(e)}")


def save_data_uri_to_file(data_uri: str, output_path: str) -> str:
    """
    Save a data URI to a file
    
    Args:
        data_uri: Data URI string
        output_path: Path to save the file
        
    Returns:
        Path to the saved file
        
    Raises:
        FileHandlingError: If the file cannot be saved
    """
    try:
        _, data_bytes = split_data_uri(data_uri)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Write the data to the file
        with open(output_path, "wb") as f:
            f.write(data_bytes)
        
        return output_path
    
    except Exception as e:
        raise FileHandlingError(f"Failed to save data URI to file: {str(e)}")


def prepare_document_for_upload(
    file_path: str, 
    provider: str
) -> Dict[str, Any]:
    """
    Prepare a document for upload to a specific provider
    
    Args:
        file_path: Path to the file
        provider: Provider name ('openai', 'anthropic', etc.)
        
    Returns:
        Dictionary with provider-specific document data
        
    Raises:
        FileHandlingError: If the file cannot be prepared
    """
    try:
        file_info = extract_file_info(file_path)
        
        # Base document data
        document_data = {
            "file_name": file_info["name"],
            "mime_type": file_info["mime_type"],
            "size": file_info["size"],
        }
        
        if provider.lower() == "openai":
            # OpenAI expects a file object
            document_data["file"] = open(file_path, "rb")
            document_data["purpose"] = "assistants"  # Default purpose
        
        elif provider.lower() == "anthropic":
            # Anthropic expects a base64-encoded string
            document_data["data"] = read_file_as_base64(file_path)
        
        elif provider.lower() == "gemini":
            # Google's Gemini expects a base64-encoded string
            document_data["data"] = read_file_as_base64(file_path)
        
        else:
            # Default to base64 encoding
            document_data["data"] = read_file_as_base64(file_path)
        
        return document_data
    
    except Exception as e:
        if isinstance(e, FileHandlingError):
            raise
        raise FileHandlingError(f"Failed to prepare document for upload: {str(e)}")


def cleanup_uploaded_document(document_data: Dict[str, Any]) -> None:
    """
    Clean up resources used by an uploaded document
    
    Args:
        document_data: Document data from prepare_document_for_upload
        
    Raises:
        FileHandlingError: If cleanup fails
    """
    try:
        # Close file objects if present
        if "file" in document_data and hasattr(document_data["file"], "close"):
            document_data["file"].close()
    
    except Exception as e:
        raise FileHandlingError(f"Failed to clean up document resources: {str(e)}")


def is_supported_image_format(file_path: str) -> bool:
    """
    Check if a file is in a supported image format
    
    Args:
        file_path: Path to the file
        
    Returns:
        Boolean indicating if the file is a supported image
    """
    mime_type = get_file_mime_type(file_path)
    return mime_type.startswith("image/") and mime_type not in [
        "image/tiff",  # Some providers don't support TIFF
        "image/webp",  # Some providers don't support WebP
    ]


def is_supported_document_format(file_path: str, provider: str) -> bool:
    """
    Check if a file is in a supported document format for a provider
    
    Args:
        file_path: Path to the file
        provider: Provider name
        
    Returns:
        Boolean indicating if the file is a supported document
    """
    mime_type = get_file_mime_type(file_path)
    extension = os.path.splitext(file_path)[1].lower()
    
    # Common document formats
    if mime_type in [
        "application/pdf",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/plain",
        "text/markdown",
        "text/csv",
    ]:
        return True
    
    # Provider-specific formats
    if provider.lower() == "openai":
        # OpenAI supports additional formats
        return extension in [
            ".pdf", ".docx", ".doc", ".txt", ".md", ".csv", ".json"
        ]
    
    elif provider.lower() == "anthropic":
        # Anthropic Claude supports PDFs and text files
        return extension in [".pdf", ".txt", ".md"]
    
    elif provider.lower() == "gemini":
        # Google's Gemini supports PDFs
        return extension in [".pdf"]
    
    # Default to common formats only
    return False