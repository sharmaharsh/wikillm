# System Architecture

## Overview
Local DeepWiki follows a modular architecture with clear separation of concerns:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CLI/API       │    │   RAG System    │    │  Vector Store   │
│   Interface     │◄──►│   (Core Logic)  │◄──►│   (ChromaDB)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         ▲                       ▲                       ▲
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Repository      │    │ Ollama Client   │    │ Documentation   │
│ Analyzer        │    │ (LLM Interface) │    │ Generator       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Core Components

### Repository Analyzer
- **Purpose**: Parse and understand code structure
- **Input**: File paths and directory structures  
- **Output**: Structured analysis with metadata
- **Key Features**:
  - Multi-language support
  - Intelligent file filtering
  - Metadata extraction (functions, classes, imports)
  - Tree structure generation

### Vector Store (ChromaDB)
- **Purpose**: Semantic search and retrieval
- **Storage**: Document embeddings and metadata
- **Features**:
  - Persistent storage
  - Similarity search
  - Metadata filtering
  - Batch operations

### RAG System
- **Purpose**: Context retrieval and response generation
- **Components**:
  - Query understanding
  - Context retrieval
  - Response synthesis
  - Conversation management

### Ollama Integration
- **Models Supported**:
  - Code generation: `deepseek-coder`, `codellama`
  - General purpose: `llama2`, `mistral`
  - Embeddings: `nomic-embed-text`
- **Features**:
  - Local inference
  - Streaming responses
  - Model management

## Data Flow

1. **Analysis Phase**:
   ```
   Source Code → Repository Analyzer → Structured Data → Vector Store
   ```

2. **Query Phase**:
   ```
   User Query → Vector Search → Context Retrieval → LLM → Response
   ```

3. **Training Data Flow**:
   ```
   Markdown Files → Text Chunks → Embeddings → Vector Store
   ```