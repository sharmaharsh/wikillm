"""RAG (Retrieval-Augmented Generation) system for context-aware responses."""

import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from ollama_client import OllamaClient
from vector_store import VectorStore

@dataclass
class RAGContext:
    """Context information for RAG system."""
    query: str
    retrieved_chunks: List[Dict[str, Any]]
    context_text: str
    metadata: Dict[str, Any]

class RAGSystem:
    """Retrieval-Augmented Generation system."""
    
    def __init__(self, vector_store: VectorStore, ollama_client: OllamaClient):
        self.vector_store = vector_store
        self.ollama_client = ollama_client
        
        # System prompts for different types of queries
        self.system_prompts = {
            'code_explanation': """You are an expert code analyst. Your task is to explain code clearly and concisely based on the provided context. 

Focus on:
- What the code does
- How it works
- Key components and their relationships
- Usage examples when relevant
- Best practices and potential improvements

Be accurate and reference the specific code provided in the context.""",

            'general_question': """You are a helpful assistant that answers questions about codebases and documentation. 

Use the provided context to give accurate, helpful answers. If the context doesn't contain enough information to answer fully, say so and provide what information you can from the context.

Be concise but thorough, and always ground your answers in the provided context.""",

            'documentation': """You are a technical documentation expert. Generate clear, well-structured documentation based on the provided code and context.

Focus on:
- Clear explanations of functionality
- Proper formatting with headers and code blocks
- Usage examples and best practices
- Integration with other parts of the system

Use proper Markdown formatting.""",

            'troubleshooting': """You are a debugging expert. Help identify issues, suggest solutions, and provide troubleshooting guidance based on the code context.

Focus on:
- Identifying potential problems
- Suggesting specific solutions
- Explaining the reasoning behind recommendations
- Providing preventive measures

Be practical and actionable in your suggestions."""
        }
    
    def determine_query_type(self, query: str) -> str:
        """Determine the type of query to select appropriate prompt."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['how does', 'what does', 'explain', 'how to']):
            return 'code_explanation'
        elif any(word in query_lower for word in ['document', 'generate docs', 'create documentation']):
            return 'documentation'
        elif any(word in query_lower for word in ['error', 'bug', 'fix', 'problem', 'issue', 'debug']):
            return 'troubleshooting'
        else:
            return 'general_question'
    
    def retrieve_context(self, query: str, n_results: int = 5, filter_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Retrieve relevant context for the query."""
        return self.vector_store.search(query, n_results, filter_metadata)
    
    def build_context_text(self, retrieved_chunks: List[Dict[str, Any]], max_context_length: int = 4000) -> str:
        """Build context text from retrieved chunks."""
        context_parts = []
        current_length = 0
        
        for chunk in retrieved_chunks:
            text = chunk['text']
            metadata = chunk.get('metadata', {})
            
            # Add metadata header for context
            header = ""
            if metadata.get('file_path'):
                header += f"File: {metadata['file_path']}\n"
            if metadata.get('element_name'):
                header += f"Element: {metadata['element_name']} ({metadata.get('element_type', 'unknown')})\n"
            if metadata.get('type'):
                header += f"Type: {metadata['type']}\n"
            
            chunk_text = header + "\n" + text + "\n" + "="*50 + "\n"
            
            # Check if adding this chunk would exceed max length
            if current_length + len(chunk_text) > max_context_length:
                if not context_parts:  # If first chunk is too long, truncate it
                    context_parts.append(chunk_text[:max_context_length])
                break
            
            context_parts.append(chunk_text)
            current_length += len(chunk_text)
        
        return "\n".join(context_parts)
    
    def generate_response(self, query: str, context: RAGContext, stream: bool = False) -> str:
        """Generate response using RAG context."""
        query_type = self.determine_query_type(query)
        system_prompt = self.system_prompts.get(query_type, self.system_prompts['general_question'])
        
        # Build the full prompt
        full_prompt = f"""{system_prompt}

CONTEXT:
{context.context_text}

USER QUERY: {query}

Please provide a helpful response based on the context above. If the context doesn't contain sufficient information, clearly state what information is missing."""

        try:
            if stream:
                return self.ollama_client.generate_stream(full_prompt, temperature=0.3)
            else:
                return self.ollama_client.generate(full_prompt, temperature=0.3)
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def query(self, 
              query: str, 
              repo_name: str = None, 
              file_path: str = None,
              n_results: int = 5,
              stream: bool = False) -> Dict[str, Any]:
        """Main query interface for RAG system."""
        
        # Build filter for specific repository or file
        filter_metadata = {}
        if repo_name:
            filter_metadata['repo_name'] = repo_name
        if file_path:
            filter_metadata['file_path'] = file_path
        
        # Retrieve relevant context
        retrieved_chunks = self.retrieve_context(query, n_results, filter_metadata or None)
        
        if not retrieved_chunks:
            return {
                'response': f"No relevant context found for query: {query}",
                'context': None,
                'sources': []
            }
        
        # Build context
        context_text = self.build_context_text(retrieved_chunks)
        context = RAGContext(
            query=query,
            retrieved_chunks=retrieved_chunks,
            context_text=context_text,
            metadata={'repo_name': repo_name, 'file_path': file_path}
        )
        
        # Generate response
        response = self.generate_response(query, context, stream)
        
        # Extract sources
        sources = []
        for chunk in retrieved_chunks:
            metadata = chunk.get('metadata', {})
            source = {
                'file_path': metadata.get('file_path'),
                'element_name': metadata.get('element_name'),
                'type': metadata.get('type'),
                'distance': chunk.get('distance', 0)
            }
            # Remove None values
            source = {k: v for k, v in source.items() if v is not None}
            if source and source not in sources:
                sources.append(source)
        
        return {
            'response': response,
            'context': context,
            'sources': sources[:5]  # Limit to top 5 sources
        }
    
    def explain_code(self, code: str, language: str = None, context: str = None) -> str:
        """Explain a specific code snippet."""
        prompt = f"""Explain the following code clearly and concisely:

Language: {language or 'Unknown'}
{f'Context: {context}' if context else ''}

Code:
```{language or ''}
{code}
```

Please explain:
1. What this code does
2. How it works
3. Key components and their purpose
4. Any notable patterns or techniques used"""

        try:
            return self.ollama_client.generate(prompt, temperature=0.3)
        except Exception as e:
            return f"Error explaining code: {str(e)}"
    
    def suggest_improvements(self, code: str, language: str = None) -> str:
        """Suggest improvements for code."""
        prompt = f"""Analyze the following code and suggest improvements:

Language: {language or 'Unknown'}

Code:
```{language or ''}
{code}
```

Please provide:
1. Code quality assessment
2. Potential improvements
3. Best practices recommendations
4. Performance optimizations if applicable
5. Security considerations if relevant"""

        try:
            return self.ollama_client.generate(prompt, temperature=0.3)
        except Exception as e:
            return f"Error suggesting improvements: {str(e)}"
    
    def search_similar_code(self, code_snippet: str, language: str = None, n_results: int = 3) -> List[Dict[str, Any]]:
        """Find similar code in the repository."""
        # Create a search query from the code
        query = f"code similar to: {code_snippet[:200]}..."
        
        filter_metadata = {}
        if language:
            filter_metadata['language'] = language
        
        return self.retrieve_context(query, n_results, filter_metadata or None)
    
    def get_file_summary(self, file_path: str, repo_name: str = None) -> str:
        """Get a summary of a specific file."""
        filter_metadata = {'file_path': file_path}
        if repo_name:
            filter_metadata['repo_name'] = repo_name
        
        # Get file overview chunk
        chunks = self.vector_store.search(
            f"file overview {file_path}", 
            n_results=1, 
            filter_metadata=filter_metadata
        )
        
        if not chunks:
            return f"No information found for file: {file_path}"
        
        chunk = chunks[0]
        context_text = chunk['text']
        
        prompt = f"""Provide a concise summary of this file:

{context_text}

Summary should include:
- Purpose of the file
- Main components
- Key functionality
- Dependencies"""

        try:
            return self.ollama_client.generate(prompt, temperature=0.3)
        except Exception as e:
            return f"Error generating file summary: {str(e)}"
    
    def get_repository_overview(self, repo_name: str) -> str:
        """Get an overview of the repository."""
        chunks = self.vector_store.search(
            f"repository overview {repo_name}",
            n_results=1,
            filter_metadata={'repo_name': repo_name, 'type': 'repository_overview'}
        )
        
        if not chunks:
            return f"No overview found for repository: {repo_name}"
        
        return chunks[0]['text']
    
    def chat_with_repo(self, messages: List[Dict[str, str]], repo_name: str = None) -> str:
        """Chat interface for conversational interaction with repository."""
        if not messages:
            return "No messages provided"
        
        # Get the latest user message
        latest_message = messages[-1]['content']
        
        # Get context for the latest query
        result = self.query(latest_message, repo_name=repo_name, stream=False)
        
        # Build conversation context
        conversation_context = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in messages[:-1]  # Exclude the latest message
        ])
        
        # Enhanced prompt with conversation history
        prompt = f"""You are having a conversation about a codebase. Here's the conversation history:

{conversation_context}

Current context from the codebase:
{result['context'].context_text if result['context'] else 'No relevant context found'}

Current question: {latest_message}

Please provide a helpful response that considers both the conversation history and the current context."""

        try:
            return self.ollama_client.generate(prompt, temperature=0.4)
        except Exception as e:
            return f"Error in chat response: {str(e)}"
