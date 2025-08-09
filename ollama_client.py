"""Ollama client for local LLM inference."""

import json
import requests
import asyncio
import aiohttp
from typing import Dict, List, Any, Optional, AsyncGenerator
from config import settings

class OllamaClient:
    """Client for interacting with Ollama local LLM server."""
    
    def __init__(self, base_url: str = None, model: str = None):
        self.base_url = base_url or settings.OLLAMA_BASE_URL
        self.model = model or settings.OLLAMA_MODEL
        self.embedding_model = settings.OLLAMA_EMBEDDING_MODEL
        
    def is_available(self) -> bool:
        """Check if Ollama server is available."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List available models."""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            return response.json().get('models', [])
        except Exception as e:
            print(f"Error listing models: {e}")
            return []
    
    def pull_model(self, model_name: str) -> bool:
        """Pull a model if not already available."""
        try:
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name},
                stream=True
            )
            
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if data.get('status'):
                        print(f"Pulling {model_name}: {data['status']}")
                    if data.get('error'):
                        print(f"Error: {data['error']}")
                        return False
            
            return True
        except Exception as e:
            print(f"Error pulling model {model_name}: {e}")
            return False
    
    def generate(self, prompt: str, model: str = None, **kwargs) -> str:
        """Generate text using Ollama."""
        model = model or self.model
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            **kwargs
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            return response.json()['response']
        except Exception as e:
            print(f"Error generating text: {e}")
            raise
    
    async def generate_async(self, prompt: str, model: str = None, **kwargs) -> str:
        """Generate text asynchronously."""
        model = model or self.model
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            **kwargs
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
                    return result['response']
            except Exception as e:
                print(f"Error generating text: {e}")
                raise
    
    async def generate_stream(self, prompt: str, model: str = None, **kwargs) -> AsyncGenerator[str, None]:
        """Generate text with streaming response."""
        model = model or self.model
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            **kwargs
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=None)
                ) as response:
                    response.raise_for_status()
                    
                    async for line in response.content:
                        if line:
                            try:
                                data = json.loads(line)
                                if 'response' in data:
                                    yield data['response']
                                if data.get('done', False):
                                    break
                            except json.JSONDecodeError:
                                continue
            except Exception as e:
                print(f"Error in streaming generation: {e}")
                raise
    
    def get_embeddings(self, text: str, model: str = None) -> List[float]:
        """Get embeddings for text."""
        model = model or self.embedding_model
        
        payload = {
            "model": model,
            "prompt": text
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            return response.json()['embedding']
        except Exception as e:
            print(f"Error getting embeddings: {e}")
            raise
    
    async def get_embeddings_async(self, text: str, model: str = None) -> List[float]:
        """Get embeddings asynchronously."""
        model = model or self.embedding_model
        
        payload = {
            "model": model,
            "prompt": text
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.base_url}/api/embeddings",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
                    return result['embedding']
            except Exception as e:
                print(f"Error getting embeddings: {e}")
                raise
    
    def chat(self, messages: List[Dict[str, str]], model: str = None, **kwargs) -> str:
        """Chat completion using Ollama."""
        model = model or self.model
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            **kwargs
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            return response.json()['message']['content']
        except Exception as e:
            print(f"Error in chat completion: {e}")
            raise
    
    async def chat_async(self, messages: List[Dict[str, str]], model: str = None, **kwargs) -> str:
        """Chat completion asynchronously."""
        model = model or self.model
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            **kwargs
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
                    return result['message']['content']
            except Exception as e:
                print(f"Error in chat completion: {e}")
                raise

class DocumentationGenerator:
    """Generate documentation using Ollama."""
    
    def __init__(self, ollama_client: OllamaClient):
        self.client = ollama_client
    
    def generate_file_documentation(self, file_analysis: Dict[str, Any]) -> str:
        """Generate documentation for a single file."""
        
        # Build context from file analysis
        context = []
        context.append(f"File: {file_analysis['file_path']}")
        context.append(f"Language: {file_analysis.get('language', 'Unknown')}")
        
        if file_analysis.get('imports'):
            context.append(f"Imports: {', '.join(file_analysis['imports'][:10])}")
        
        # Add code elements
        elements_summary = []
        for element in file_analysis.get('elements', []):
            elem_type = element['type']
            name = element['name']
            signature = element.get('signature', '')
            docstring = element.get('docstring', '')
            
            elem_desc = f"{elem_type}: {name}"
            if signature:
                elem_desc += f" - {signature}"
            if docstring:
                elem_desc += f" - {docstring[:100]}..."
            
            elements_summary.append(elem_desc)
        
        if elements_summary:
            context.append("Code Elements:")
            context.extend(elements_summary[:15])  # Limit to prevent prompt overflow
        
        # Add file content preview
        content = file_analysis.get('content', '')
        if content:
            content_preview = content[:2000] + "..." if len(content) > 2000 else content
            context.append(f"Content Preview:\n```\n{content_preview}\n```")
        
        context_str = "\n".join(context)
        
        prompt = f"""Analyze the following code file and generate comprehensive documentation in Markdown format.

{context_str}

Please provide:
1. A brief overview of what this file does
2. Main components and their purposes
3. Key functions/classes with descriptions
4. Usage examples if applicable
5. Dependencies and relationships

Format the output as clean Markdown with appropriate headers and code blocks."""

        try:
            return self.client.generate(prompt, temperature=0.3, max_tokens=2000)
        except Exception as e:
            print(f"Error generating documentation for {file_analysis['file_path']}: {e}")
            return f"# {file_analysis['file_path']}\n\nError generating documentation: {str(e)}"
    
    def generate_repository_overview(self, repo_analysis: Dict[str, Any]) -> str:
        """Generate high-level repository documentation."""
        
        stats = repo_analysis.get('statistics', {})
        structure = repo_analysis.get('structure', {})
        
        context = []
        context.append(f"Repository: {repo_analysis.get('repo_name', 'Unknown')}")
        context.append(f"Total Files: {stats.get('total_files', 0)}")
        context.append(f"Code Files: {stats.get('code_files', 0)}")
        context.append(f"Documentation Files: {stats.get('doc_files', 0)}")
        context.append(f"Total Lines: {stats.get('total_lines', 0)}")
        
        if stats.get('languages'):
            langs = list(stats['languages'].keys())
            context.append(f"Languages: {', '.join(langs)}")
        
        # Add git info if available
        git_info = repo_analysis.get('git_info')
        if git_info:
            context.append(f"Current Branch: {git_info.get('current_branch', 'Unknown')}")
            if git_info.get('last_commit'):
                commit = git_info['last_commit']
                context.append(f"Last Commit: {commit.get('message', '')[:100]}")
        
        # Add main files/directories
        main_files = []
        for file_data in repo_analysis.get('files', [])[:10]:
            if file_data.get('file_type') == 'documentation':
                main_files.append(file_data['file_path'])
        
        if main_files:
            context.append(f"Key Documentation: {', '.join(main_files)}")
        
        context_str = "\n".join(context)
        
        prompt = f"""Analyze this code repository and generate a comprehensive overview documentation in Markdown format.

Repository Information:
{context_str}

Please provide:
1. Project overview and purpose
2. Architecture and structure
3. Key components and modules
4. Getting started guide
5. Development setup instructions
6. Main features and capabilities

Format as professional README-style documentation with proper Markdown structure."""

        try:
            return self.client.generate(prompt, temperature=0.3, max_tokens=3000)
        except Exception as e:
            print(f"Error generating repository overview: {e}")
            return f"# Repository Overview\n\nError generating overview: {str(e)}"
