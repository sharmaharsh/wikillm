"""FastAPI server for Local DeepWiki."""

import asyncio
import json
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from config import settings
from repository_analyzer import RepositoryAnalyzer
from ollama_client import OllamaClient
from vector_store import VectorStore
from rag_system import RAGSystem
from documentation_generator import DocumentationBuilder, DocumentationConfig

# Pydantic models for API
class QueryRequest(BaseModel):
    query: str
    repo_name: Optional[str] = None
    file_path: Optional[str] = None
    stream: Optional[bool] = False

class ChatMessage(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    repo_name: Optional[str] = None

class RepositoryRequest(BaseModel):
    repo_path: str
    repo_name: Optional[str] = None

class ExplainCodeRequest(BaseModel):
    code: str
    language: Optional[str] = None
    context: Optional[str] = None

class DocumentationRequest(BaseModel):
    repo_name: str
    include_overview: bool = True
    include_api_docs: bool = True
    include_examples: bool = True
    include_architecture: bool = True

# Global instances
app = FastAPI(title="Local DeepWiki", version="1.0.0")
ollama_client = OllamaClient()
vector_store = VectorStore()
rag_system = RAGSystem(vector_store, ollama_client)
repository_analyzer = RepositoryAnalyzer()
documentation_builder = DocumentationBuilder(ollama_client)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    ollama_available = ollama_client.is_available()
    vector_stats = vector_store.get_collection_stats()
    
    return {
        "status": "healthy",
        "ollama_available": ollama_available,
        "ollama_url": settings.OLLAMA_BASE_URL,
        "ollama_model": settings.OLLAMA_MODEL,
        "vector_store": vector_stats
    }

@app.get("/models")
async def list_models():
    """List available Ollama models."""
    try:
        models = ollama_client.list_models()
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")

# Repository management endpoints
@app.post("/repositories/analyze")
async def analyze_repository(request: RepositoryRequest, background_tasks: BackgroundTasks):
    """Analyze a repository and add it to the vector store."""
    repo_path = Path(request.repo_path)
    
    if not repo_path.exists():
        raise HTTPException(status_code=404, detail=f"Repository path not found: {repo_path}")
    
    repo_name = request.repo_name or repo_path.name
    
    try:
        # Start analysis in background
        background_tasks.add_task(analyze_repository_task, str(repo_path), repo_name)
        
        return {
            "message": f"Repository analysis started for {repo_name}",
            "repo_name": repo_name,
            "repo_path": str(repo_path)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting analysis: {str(e)}")

async def analyze_repository_task(repo_path: str, repo_name: str):
    """Background task to analyze repository."""
    try:
        print(f"Starting analysis of {repo_name} at {repo_path}")
        
        # Analyze repository
        analysis = repository_analyzer.analyze_repository(repo_path)
        analysis['repo_name'] = repo_name
        
        # Save analysis
        analysis_path = Path(settings.REPOS_DIRECTORY) / f"{repo_name}_analysis.json"
        repository_analyzer.save_analysis(analysis, str(analysis_path))
        
        # Add to vector store
        chunks_added = vector_store.add_repository(analysis)
        
        print(f"Analysis complete for {repo_name}: {chunks_added} chunks added to vector store")
        
    except Exception as e:
        print(f"Error in repository analysis task: {str(e)}")
        traceback.print_exc()

@app.get("/repositories")
async def list_repositories():
    """List analyzed repositories."""
    repos = []
    repos_dir = Path(settings.REPOS_DIRECTORY)
    
    if repos_dir.exists():
        for analysis_file in repos_dir.glob("*_analysis.json"):
            try:
                analysis = repository_analyzer.load_analysis(str(analysis_file))
                repos.append({
                    "name": analysis.get('repo_name', analysis_file.stem.replace('_analysis', '')),
                    "path": analysis.get('repo_path', ''),
                    "files": analysis.get('statistics', {}).get('total_files', 0),
                    "languages": list(analysis.get('statistics', {}).get('languages', {}).keys())
                })
            except Exception as e:
                print(f"Error loading analysis from {analysis_file}: {e}")
    
    return {"repositories": repos}

@app.delete("/repositories/{repo_name}")
async def delete_repository(repo_name: str):
    """Delete repository from vector store."""
    try:
        vector_store.delete_repository(repo_name)
        
        # Also delete analysis file
        analysis_path = Path(settings.REPOS_DIRECTORY) / f"{repo_name}_analysis.json"
        if analysis_path.exists():
            analysis_path.unlink()
        
        return {"message": f"Repository {repo_name} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting repository: {str(e)}")

# Query endpoints
@app.post("/query")
async def query_repository(request: QueryRequest):
    """Query repository using RAG system."""
    try:
        if request.stream:
            # Return streaming response
            return StreamingResponse(
                stream_query_response(request),
                media_type="text/plain"
            )
        else:
            # Return complete response
            result = rag_system.query(
                request.query,
                repo_name=request.repo_name,
                file_path=request.file_path,
                stream=False
            )
            return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

async def stream_query_response(request: QueryRequest):
    """Stream query response for real-time updates."""
    try:
        # Get context first
        filter_metadata = {}
        if request.repo_name:
            filter_metadata['repo_name'] = request.repo_name
        if request.file_path:
            filter_metadata['file_path'] = request.file_path
        
        retrieved_chunks = rag_system.retrieve_context(
            request.query, 
            n_results=5, 
            filter_metadata=filter_metadata or None
        )
        
        if not retrieved_chunks:
            yield "No relevant context found for query.\n"
            return
        
        context_text = rag_system.build_context_text(retrieved_chunks)
        
        # Build prompt
        system_prompt = rag_system.system_prompts['general_question']
        full_prompt = f"""{system_prompt}

CONTEXT:
{context_text}

USER QUERY: {request.query}

Please provide a helpful response based on the context above."""

        # Stream response
        async for chunk in ollama_client.generate_stream(full_prompt, temperature=0.3):
            yield chunk
            
    except Exception as e:
        yield f"Error: {str(e)}\n"

@app.post("/chat")
async def chat_with_repository(request: ChatRequest):
    """Chat interface for conversational interaction."""
    try:
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        response = rag_system.chat_with_repo(messages, request.repo_name)
        
        return {
            "response": response,
            "repo_name": request.repo_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in chat: {str(e)}")

@app.post("/explain")
async def explain_code(request: ExplainCodeRequest):
    """Explain code snippet."""
    try:
        explanation = rag_system.explain_code(
            request.code,
            request.language,
            request.context
        )
        return {"explanation": explanation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error explaining code: {str(e)}")

@app.post("/improve")
async def suggest_improvements(request: ExplainCodeRequest):
    """Suggest code improvements."""
    try:
        suggestions = rag_system.suggest_improvements(
            request.code,
            request.language
        )
        return {"suggestions": suggestions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error suggesting improvements: {str(e)}")

@app.get("/repositories/{repo_name}/overview")
async def get_repository_overview(repo_name: str):
    """Get repository overview."""
    try:
        overview = rag_system.get_repository_overview(repo_name)
        return {"overview": overview, "repo_name": repo_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting overview: {str(e)}")

@app.get("/repositories/{repo_name}/files/{file_path:path}/summary")
async def get_file_summary(repo_name: str, file_path: str):
    """Get file summary."""
    try:
        summary = rag_system.get_file_summary(file_path, repo_name)
        return {"summary": summary, "file_path": file_path, "repo_name": repo_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting file summary: {str(e)}")

# Documentation endpoints
@app.post("/repositories/{repo_name}/generate-docs")
async def generate_documentation(repo_name: str, request: DocumentationRequest, background_tasks: BackgroundTasks):
    """Generate comprehensive documentation for repository."""
    try:
        # Load repository analysis
        analysis_path = Path(settings.REPOS_DIRECTORY) / f"{repo_name}_analysis.json"
        if not analysis_path.exists():
            raise HTTPException(status_code=404, detail=f"Repository {repo_name} not found. Please analyze it first.")
        
        analysis = repository_analyzer.load_analysis(str(analysis_path))
        
        config = DocumentationConfig(
            include_overview=request.include_overview,
            include_api_docs=request.include_api_docs,
            include_examples=request.include_examples,
            include_architecture=request.include_architecture
        )
        
        # Start documentation generation in background
        background_tasks.add_task(generate_docs_task, analysis, config, repo_name)
        
        return {
            "message": f"Documentation generation started for {repo_name}",
            "repo_name": repo_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting documentation generation: {str(e)}")

async def generate_docs_task(analysis: Dict[str, Any], config: DocumentationConfig, repo_name: str):
    """Background task to generate documentation."""
    try:
        print(f"Starting documentation generation for {repo_name}")
        docs = documentation_builder.generate_full_documentation(analysis, config)
        print(f"Documentation generation complete for {repo_name}: {len(docs)} files generated")
    except Exception as e:
        print(f"Error in documentation generation task: {str(e)}")
        traceback.print_exc()

@app.get("/repositories/{repo_name}/docs")
async def list_generated_docs(repo_name: str):
    """List generated documentation files."""
    docs_dir = Path(settings.DOCS_DIRECTORY) / repo_name
    
    if not docs_dir.exists():
        return {"documents": []}
    
    docs = []
    for doc_file in docs_dir.rglob("*.md"):
        relative_path = doc_file.relative_to(docs_dir)
        docs.append({
            "name": doc_file.name,
            "path": str(relative_path),
            "size": doc_file.stat().st_size
        })
    
    return {"documents": docs}

@app.get("/repositories/{repo_name}/docs/{doc_path:path}")
async def get_documentation(repo_name: str, doc_path: str):
    """Get specific documentation file."""
    doc_file = Path(settings.DOCS_DIRECTORY) / repo_name / doc_path
    
    if not doc_file.exists():
        raise HTTPException(status_code=404, detail="Documentation file not found")
    
    try:
        with open(doc_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return {
            "content": content,
            "path": doc_path,
            "repo_name": repo_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading documentation: {str(e)}")

# Search endpoints
@app.get("/search")
async def search_all_repositories(q: str, limit: int = 10):
    """Search across all repositories."""
    try:
        results = vector_store.search(q, n_results=limit)
        return {"results": results, "query": q}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching: {str(e)}")

@app.get("/repositories/{repo_name}/search")
async def search_repository(repo_name: str, q: str, limit: int = 10):
    """Search within specific repository."""
    try:
        results = vector_store.search(
            q, 
            n_results=limit, 
            filter_metadata={"repo_name": repo_name}
        )
        return {"results": results, "query": q, "repo_name": repo_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching repository: {str(e)}")

# Statistics endpoint
@app.get("/stats")
async def get_system_stats():
    """Get system statistics."""
    try:
        vector_stats = vector_store.get_collection_stats()
        
        # Count repositories
        repos_dir = Path(settings.REPOS_DIRECTORY)
        repo_count = len(list(repos_dir.glob("*_analysis.json"))) if repos_dir.exists() else 0
        
        # Count generated docs
        docs_dir = Path(settings.DOCS_DIRECTORY)
        doc_count = len(list(docs_dir.rglob("*.md"))) if docs_dir.exists() else 0
        
        return {
            "repositories": repo_count,
            "documents_in_vector_store": vector_stats.get('total_documents', 0),
            "generated_docs": doc_count,
            "ollama_available": ollama_client.is_available(),
            "ollama_model": settings.OLLAMA_MODEL
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

# Main function to run the server
def main():
    """Main function to run the API server."""
    print(f"Starting Local DeepWiki server on {settings.API_HOST}:{settings.API_PORT}")
    print(f"Ollama URL: {settings.OLLAMA_BASE_URL}")
    print(f"Model: {settings.OLLAMA_MODEL}")
    
    # Check Ollama availability
    if not ollama_client.is_available():
        print("WARNING: Ollama server not available. Please start Ollama first.")
        print("Run: ollama serve")
    
    uvicorn.run(
        "api:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True
    )

if __name__ == "__main__":
    main()
