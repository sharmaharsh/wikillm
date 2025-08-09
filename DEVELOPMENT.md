# Development Guide

## Setting up the development environment

1. Install Python 3.8+
2. Install dependencies: `pip install -r requirements.txt`
3. Install and start Ollama
4. Pull required models: `ollama pull deepseek-coder:6.7b`

## Running the system

### Analyze a repository
```bash
python main.py analyze . --name myproject
```

### Query the repository
```bash
python main.py query "How does authentication work?" --repo myproject
```

### Start the API server
```bash
python main.py server
```

## Key Components

- **config.py**: Configuration settings
- **repository_analyzer.py**: Code parsing and analysis
- **vector_store.py**: ChromaDB integration
- **rag_system.py**: RAG implementation
- **api.py**: FastAPI server