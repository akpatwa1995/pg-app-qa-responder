"""Configuration settings for the application."""

import os
from typing import Optional

# Ollama server configuration
OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_API_KEY: Optional[str] = os.getenv("OLLAMA_API_KEY","qqqqa")

# Vector store configuration
PERSIST_DIRECTORY: str = os.path.join("data", "vectors")
DOCS_METADATA_DIR: str = os.path.join("data", "docs_metadata")
DATA_CHANGES_DIR: str = os.path.join("data", "changes")

# Create necessary directories
for directory in [PERSIST_DIRECTORY, DOCS_METADATA_DIR, DATA_CHANGES_DIR]:
    os.makedirs(directory, exist_ok=True)

# Log configuration
logging_config = {
    "level": os.getenv("LOG_LEVEL", "INFO"),
    "format": "%(asctime)s - %(levelname)s - %(message)s",
    "datefmt": "%Y-%m-%d %H:%M:%S"
} 