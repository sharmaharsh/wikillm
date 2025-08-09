# Local DeepWiki

A local implementation of DeepWiki that allows you to chat with your code repositories using Ollama and local LLMs. This system provides repository analysis, documentation generation, and conversational AI capabilities - all running locally on your machine.

## Features

- 🔍 **Repository Analysis**: Automatically parse and understand code structure
- 🤖 **Local LLM Integration**: Uses Ollama for privacy-focused AI inference
- 📚 **Documentation Generation**: Create comprehensive docs from your codebase
- 💬 **Interactive Chat**: Ask questions about your code in natural language
- 🔎 **Semantic Search**: Find relevant code and documentation quickly
- 🌐 **Web Interface**: Clean, modern UI for easy interaction
- 📖 **RAG System**: Context-aware responses grounded in your actual code

## Prerequisites

1. **Python 3.8+**
2. **Ollama** - Download from [ollama.ai](https://ollama.ai)
3. **Git** (optional, for repository information)

## Installation

1. **Clone this repository:**
```bash
git clone <repository-url>
cd local_deepwiki
```

2. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

3. **Install and start Ollama:**
```bash
# Install Ollama (see https://ollama.ai for platform-specific instructions)

# Start Ollama server
ollama serve

# Pull required models (in another terminal)
ollama pull llama2          # Main language model
ollama pull nomic-embed-text  # Embedding model (optional)
```

4. **Check prerequisites:**
```bash
python main.py check
```

## Quick Start

### 1. Analyze a Repository

```bash
# Analyze your current directory
python main.py analyze .

# Analyze a specific repository
python main.py analyze /path/to/your/repo --name my-project

# Analyze with custom name
python main.py analyze ~/projects/my-app --name my-awesome-app
```

### 2. Chat with Your Code

```bash
# Start interactive chat
python main.py chat

# Chat with specific repository
python main.py chat --repo my-project
```

### 3. Query Your Repository

```bash
# Ask a specific question
python main.py query "How does user authentication work?" --repo my-project

# Query without specifying repository (searches all)
python main.py query "Show me the main entry point"
```

### 4. Generate Documentation

```bash
# Generate comprehensive documentation
python main.py docs my-project
```

### 5. Web Interface

```bash
# Start the web server
python main.py server

# Or with custom host/port
python main.py server --host 127.0.0.1 --port 8080
```

Then open http://localhost:8000 in your browser.

## Configuration

Create a `.env` file to customize settings:

```env
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2
OLLAMA_EMBEDDING_MODEL=nomic-embed-text

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Storage Configuration
CHROMA_PERSIST_DIRECTORY=./data/chroma_db
REPOS_DIRECTORY=./data/repos
DOCS_DIRECTORY=./data/generated_docs

# Processing Configuration
MAX_CHUNK_SIZE=2000
CHUNK_OVERLAP=200
MAX_CONTEXT_LENGTH=4000
```

## Supported Languages

The system can analyze and understand code in:

- Python (.py)
- JavaScript/TypeScript (.js, .ts, .jsx, .tsx)
- Java (.java)
- C/C++ (.c, .cpp, .h)
- C# (.cs)
- PHP (.php)
- Ruby (.rb)
- Go (.go)
- Rust (.rs)
- Swift (.swift)
- Kotlin (.kt)
- Scala (.scala)
- And more...

Plus documentation files:
- Markdown (.md)
- reStructuredText (.rst)
- Plain text (.txt)

## CLI Commands

### Repository Management
```bash
# List analyzed repositories
python main.py list

# Show system statistics
python main.py stats
```

### Analysis and Documentation
```bash
# Analyze repository
python main.py analyze <path> [--name <name>]

# Generate documentation
python main.py docs <repo-name>
```

### Querying
```bash
# One-time query
python main.py query "<question>" [--repo <name>]

# Interactive chat
python main.py chat [--repo <name>]
```

### Web Server
```bash
# Start web interface
python main.py server [--host <host>] [--port <port>]
```

## API Endpoints

When running the web server, these endpoints are available:

### Health & System
- `GET /health` - System health check
- `GET /stats` - System statistics
- `GET /models` - List available Ollama models

### Repository Management
- `POST /repositories/analyze` - Analyze repository
- `GET /repositories` - List repositories
- `DELETE /repositories/{name}` - Delete repository

### Querying
- `POST /query` - Query with RAG system
- `POST /chat` - Chat interface
- `POST /explain` - Explain code snippet
- `POST /improve` - Suggest code improvements

### Documentation
- `POST /repositories/{name}/generate-docs` - Generate docs
- `GET /repositories/{name}/docs` - List generated docs
- `GET /repositories/{name}/docs/{path}` - Get specific doc

### Search
- `GET /search` - Search all repositories
- `GET /repositories/{name}/search` - Search specific repository

## How It Works

1. **Repository Analysis**: The system parses your codebase, extracting:
   - File structure and organization
   - Functions, classes, and their relationships
   - Import dependencies
   - Documentation and comments
   - Git information (if available)

2. **Vector Embeddings**: Code and documentation are chunked and converted to embeddings using:
   - Ollama embedding models (preferred)
   - Sentence Transformers (fallback)
   - ChromaDB for efficient storage and retrieval

3. **RAG System**: When you ask questions:
   - Your query is embedded and matched against the codebase
   - Relevant code snippets and documentation are retrieved
   - Context is provided to the LLM for accurate, grounded responses

4. **Documentation Generation**: Using the analyzed structure:
   - README files with project overviews
   - API documentation for each file
   - Architecture documentation
   - Usage examples and tutorials

## Example Queries

Here are some example questions you can ask:

### Code Understanding
- "How does user authentication work in this project?"
- "What is the main entry point of the application?"
- "Show me all the API endpoints"
- "How is data validation handled?"

### Architecture Questions
- "What's the overall architecture of this system?"
- "How do the different modules interact?"
- "What design patterns are used?"
- "What are the main dependencies?"

### Implementation Details
- "How is error handling implemented?"
- "Where is the database connection configured?"
- "How are user permissions checked?"
- "What testing framework is used?"

### Documentation
- "Generate API documentation for the user service"
- "Create a getting started guide"
- "Explain the deployment process"

## Troubleshooting

### Ollama Issues
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Pull required models
ollama pull llama2
ollama pull nomic-embed-text

# Check model availability
ollama list
```

### Common Problems

1. **"Ollama server not available"**
   - Make sure Ollama is installed and running (`ollama serve`)
   - Check the OLLAMA_BASE_URL in your configuration

2. **"Model not found"**
   - Pull the required model: `ollama pull llama2`
   - Update OLLAMA_MODEL in configuration if using a different model

3. **"No relevant context found"**
   - Make sure the repository has been analyzed
   - Try rephrasing your question
   - Check that the repository contains relevant code

4. **Slow responses**
   - Consider using a smaller, faster model (e.g., `llama2:7b`)
   - Reduce MAX_CONTEXT_LENGTH in configuration
   - Use more specific queries

### Performance Tips

1. **Model Selection**: 
   - Use `codellama` for better code understanding
   - Use `llama2:7b` for faster responses
   - Use `mistral` for a good balance

2. **Configuration Tuning**:
   - Adjust MAX_CHUNK_SIZE for your hardware
   - Reduce MAX_CONTEXT_LENGTH for faster responses
   - Increase CHUNK_OVERLAP for better context

3. **Repository Size**:
   - Large repositories may take time to analyze
   - Consider analyzing specific directories for faster results
   - Use .gitignore patterns to exclude unnecessary files

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Ollama](https://ollama.ai) for local LLM inference
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [LangChain](https://langchain.com/) for RAG implementation
- [FastAPI](https://fastapi.tiangolo.com/) for the web API
- [Vue.js](https://vuejs.org/) for the web interface

## Roadmap

- [ ] Support for more programming languages
- [ ] Advanced code analysis (call graphs, dependency analysis)
- [ ] Integration with popular IDEs
- [ ] Multi-repository projects support
- [ ] Custom model fine-tuning
- [ ] Collaborative features
- [ ] Plugin system for extensibility
