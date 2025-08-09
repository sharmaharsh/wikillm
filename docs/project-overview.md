# Project Overview

This is Local DeepWiki - a system for analyzing code repositories using local LLMs.

## Key Features
- Repository analysis and indexing
- RAG-based question answering
- Local LLM integration via Ollama
- REST API for programmatic access

## Architecture
- **Repository Analyzer**: Parses code structure and extracts metadata
- **Vector Store**: Uses ChromaDB for semantic search
- **RAG System**: Retrieves context and generates responses
- **API Server**: FastAPI-based REST endpoints