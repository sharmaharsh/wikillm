# API Reference

## Base URL
```
http://localhost:8000
```

## Health & System Endpoints

### GET /health
Check system health and configuration.

**Response:**
```json
{
  "status": "healthy",
  "ollama_available": true,
  "ollama_url": "http://localhost:11434",
  "ollama_model": "deepseek-coder:6.7b",
  "vector_store": {
    "total_documents": 169,
    "collection_name": "codebase_docs"
  }
}
```

### GET /stats
Get system statistics.

**Response:**
```json
{
  "repositories": 1,
  "documents_in_vector_store": 169,
  "generated_docs": 0,
  "ollama_available": true,
  "ollama_model": "deepseek-coder:6.7b"
}
```

## Repository Management

### POST /repositories/analyze
Analyze a repository and add it to the vector store.

**Request:**
```json
{
  "repo_path": "/path/to/repository",
  "repo_name": "optional-name"
}
```

### GET /repositories
List all analyzed repositories.

**Response:**
```json
{
  "repositories": [
    {
      "name": "wikillm",
      "path": "C:\\Users\\chuba\\wikillm",
      "files": 14,
      "languages": ["python"]
    }
  ]
}
```

## Query & Chat Endpoints

### POST /query
Query repositories using the RAG system.

**Request:**
```json
{
  "query": "How does authentication work?",
  "repo_name": "optional-filter",
  "stream": false
}
```

**Response:**
```json
{
  "response": "Generated response text...",
  "context": {...},
  "sources": [...]
}
```

### POST /chat
Interactive chat with repository context.

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "What is this project about?"}
  ],
  "repo_name": "wikillm"
}
```

## Code Analysis Endpoints

### POST /explain
Explain code snippets.

**Request:**
```json
{
  "code": "def hello_world():\n    print('Hello, World!')",
  "language": "python",
  "context": "Optional context"
}
```

### POST /improve
Get code improvement suggestions.

**Request:**
```json
{
  "code": "def hello_world():\n    print('Hello, World!')",
  "language": "python"
}
```