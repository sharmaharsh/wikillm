"""Configuration settings for Local DeepWiki."""

import os
from pathlib import Path
from typing import List, Dict, Any
from pydantic import BaseSettings

class Settings(BaseSettings):
    """Application settings."""
    
    # Ollama settings
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama2"  # Default model, can be changed to codellama, mistral, etc.
    OLLAMA_EMBEDDING_MODEL: str = "nomic-embed-text"
    
    # Vector database settings
    CHROMA_PERSIST_DIRECTORY: str = "./data/chroma_db"
    COLLECTION_NAME: str = "codebase_docs"
    
    # Repository settings
    REPOS_DIRECTORY: str = "./data/repos"
    DOCS_DIRECTORY: str = "./data/generated_docs"
    
    # Generation settings
    MAX_CHUNK_SIZE: int = 2000
    CHUNK_OVERLAP: int = 200
    MAX_CONTEXT_LENGTH: int = 4000
    
    # API settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    # Supported file extensions
    CODE_EXTENSIONS: List[str] = [
        ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".cpp", ".c", ".h",
        ".cs", ".php", ".rb", ".go", ".rs", ".swift", ".kt", ".scala",
        ".r", ".sql", ".sh", ".yaml", ".yml", ".json", ".xml", ".html",
        ".css", ".scss", ".less", ".vue", ".svelte"
    ]
    
    DOC_EXTENSIONS: List[str] = [".md", ".rst", ".txt", ".adoc"]
    
    # Ignored directories
    IGNORED_DIRS: List[str] = [
        ".git", ".svn", ".hg", "__pycache__", ".pytest_cache",
        "node_modules", ".npm", ".yarn", "bower_components",
        ".venv", "venv", "env", ".env", "virtualenv",
        "dist", "build", "target", "bin", "obj",
        ".idea", ".vscode", ".vs", "*.egg-info"
    ]
    
    class Config:
        env_file = ".env"

# Global settings instance
settings = Settings()

# Ensure data directories exist
Path(settings.CHROMA_PERSIST_DIRECTORY).mkdir(parents=True, exist_ok=True)
Path(settings.REPOS_DIRECTORY).mkdir(parents=True, exist_ok=True)
Path(settings.DOCS_DIRECTORY).mkdir(parents=True, exist_ok=True)
