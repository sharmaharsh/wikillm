"""Vector embedding and storage system using ChromaDB."""

import os
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import tiktoken
from config import settings
from ollama_client import OllamaClient

class TextChunker:
    """Utility class for chunking text into manageable pieces."""
    
    def __init__(self, max_chunk_size: int = None, overlap: int = None):
        self.max_chunk_size = max_chunk_size or settings.MAX_CHUNK_SIZE
        self.overlap = overlap or settings.CHUNK_OVERLAP
        
        # Initialize tokenizer for accurate token counting
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except:
            self.tokenizer = None
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Rough approximation: 1 token ≈ 4 characters
            return len(text) // 4
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Split text into chunks with metadata."""
        if metadata is None:
            metadata = {}
        
        chunks = []
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        current_chunk = ""
        current_tokens = 0
        
        for para in paragraphs:
            para_tokens = self.count_tokens(para)
            
            # If single paragraph exceeds max size, split it further
            if para_tokens > self.max_chunk_size:
                # Save current chunk if not empty
                if current_chunk.strip():
                    chunks.append({
                        'text': current_chunk.strip(),
                        'tokens': current_tokens,
                        **metadata
                    })
                    current_chunk = ""
                    current_tokens = 0
                
                # Split large paragraph by sentences
                sentences = para.split('. ')
                for sentence in sentences:
                    sentence_tokens = self.count_tokens(sentence)
                    
                    if current_tokens + sentence_tokens > self.max_chunk_size:
                        if current_chunk.strip():
                            chunks.append({
                                'text': current_chunk.strip(),
                                'tokens': current_tokens,
                                **metadata
                            })
                        current_chunk = sentence
                        current_tokens = sentence_tokens
                    else:
                        current_chunk += (". " if current_chunk else "") + sentence
                        current_tokens += sentence_tokens
            
            # Normal paragraph processing
            elif current_tokens + para_tokens > self.max_chunk_size:
                # Save current chunk
                if current_chunk.strip():
                    chunks.append({
                        'text': current_chunk.strip(),
                        'tokens': current_tokens,
                        **metadata
                    })
                
                # Start new chunk with overlap
                if self.overlap > 0 and chunks:
                    # Get last few words for overlap
                    last_chunk_words = current_chunk.split()
                    overlap_words = last_chunk_words[-self.overlap:] if len(last_chunk_words) > self.overlap else last_chunk_words
                    current_chunk = " ".join(overlap_words) + "\n\n" + para
                else:
                    current_chunk = para
                
                current_tokens = self.count_tokens(current_chunk)
            else:
                current_chunk += ("\n\n" if current_chunk else "") + para
                current_tokens += para_tokens
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip(),
                'tokens': current_tokens,
                **metadata
            })
        
        return chunks
    
    def chunk_code_file(self, content: str, file_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk code file based on structure."""
        chunks = []
        
        # Create chunk for file overview
        overview_text = f"File: {file_analysis['file_path']}\n"
        overview_text += f"Language: {file_analysis.get('language', 'Unknown')}\n"
        
        if file_analysis.get('imports'):
            overview_text += f"Imports: {', '.join(file_analysis['imports'])}\n"
        
        # Add summary of elements
        elements = file_analysis.get('elements', [])
        if elements:
            overview_text += "\nCode Elements:\n"
            for element in elements:
                overview_text += f"- {element['type']}: {element['name']}\n"
                if element.get('docstring'):
                    overview_text += f"  {element['docstring'][:100]}...\n"
        
        chunks.append({
            'text': overview_text,
            'type': 'file_overview',
            'file_path': file_analysis['file_path'],
            'language': file_analysis.get('language'),
            'source': 'code_analysis'
        })
        
        # Create chunks for individual code elements
        lines = content.split('\n')
        for element in elements:
            start_line = element.get('line_start', 1) - 1  # Convert to 0-based
            end_line = element.get('line_end', len(lines))
            
            if start_line < len(lines):
                element_content = '\n'.join(lines[start_line:min(end_line, len(lines))])
                
                element_text = f"Element: {element['name']} ({element['type']})\n"
                element_text += f"File: {file_analysis['file_path']}\n"
                
                if element.get('signature'):
                    element_text += f"Signature: {element['signature']}\n"
                
                if element.get('docstring'):
                    element_text += f"Documentation: {element['docstring']}\n"
                
                element_text += f"\nCode:\n```{file_analysis.get('language', '')}\n{element_content}\n```"
                
                chunks.append({
                    'text': element_text,
                    'type': 'code_element',
                    'element_type': element['type'],
                    'element_name': element['name'],
                    'file_path': file_analysis['file_path'],
                    'language': file_analysis.get('language'),
                    'line_start': element.get('line_start'),
                    'line_end': element.get('line_end'),
                    'source': 'code_analysis'
                })
        
        # If no elements found, chunk the entire file content
        if not elements and content.strip():
            file_chunks = self.chunk_text(
                content, 
                {
                    'type': 'file_content',
                    'file_path': file_analysis['file_path'],
                    'language': file_analysis.get('language'),
                    'source': 'file_content'
                }
            )
            chunks.extend(file_chunks)
        
        return chunks

class VectorStore:
    """Vector store for embeddings using ChromaDB and local embeddings."""
    
    def __init__(self, persist_directory: str = None, collection_name: str = None):
        self.persist_directory = persist_directory or settings.CHROMA_PERSIST_DIRECTORY
        self.collection_name = collection_name or settings.COLLECTION_NAME
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
        except:
            self.collection = self.client.create_collection(name=self.collection_name)
        
        # Initialize embedding models
        self.ollama_client = OllamaClient()
        
        # Fallback to sentence transformers if Ollama embeddings fail
        try:
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        except:
            print("Warning: Could not load sentence transformer model")
            self.sentence_transformer = None
        
        self.chunker = TextChunker()
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using available models."""
        # Try Ollama first
        if self.ollama_client.is_available():
            try:
                return self.ollama_client.get_embeddings(text)
            except Exception as e:
                print(f"Ollama embedding failed: {e}")
        
        # Fallback to sentence transformer
        if self.sentence_transformer:
            try:
                embedding = self.sentence_transformer.encode(text)
                return embedding.tolist()
            except Exception as e:
                print(f"Sentence transformer embedding failed: {e}")
        
        raise Exception("No embedding model available")
    
    def add_repository(self, repo_analysis: Dict[str, Any]) -> int:
        """Add entire repository to vector store."""
        repo_name = repo_analysis.get('repo_name', 'unknown')
        total_chunks = 0
        
        # Add repository overview
        if repo_analysis.get('files'):
            overview_text = f"Repository: {repo_name}\n"
            overview_text += f"Total files: {repo_analysis['statistics']['total_files']}\n"
            overview_text += f"Languages: {', '.join(repo_analysis['statistics']['languages'].keys())}\n"
            
            # Add file list
            overview_text += "\nFiles:\n"
            for file_data in repo_analysis['files'][:20]:  # Limit to first 20 files
                overview_text += f"- {file_data['file_path']}\n"
            
            total_chunks += self.add_document(
                overview_text,
                {
                    'type': 'repository_overview',
                    'repo_name': repo_name,
                    'source': 'repository_analysis'
                }
            )
        
        # Add individual files
        for file_data in repo_analysis.get('files', []):
            file_chunks = self.add_file_analysis(file_data, repo_name)
            total_chunks += file_chunks
        
        return total_chunks
    
    def add_file_analysis(self, file_analysis: Dict[str, Any], repo_name: str = None) -> int:
        """Add file analysis to vector store."""
        if file_analysis.get('file_type') == 'code':
            chunks = self.chunker.chunk_code_file(
                file_analysis.get('content', ''), 
                file_analysis
            )
        else:
            # Regular text chunking for documentation files
            chunks = self.chunker.chunk_text(
                file_analysis.get('content', ''),
                {
                    'type': 'documentation',
                    'file_path': file_analysis['file_path'],
                    'source': 'file_content'
                }
            )
        
        # Add repository name to all chunks
        for chunk in chunks:
            if repo_name:
                chunk['repo_name'] = repo_name
        
        return self.add_chunks(chunks)
    
    def add_document(self, text: str, metadata: Dict[str, Any]) -> int:
        """Add a single document to vector store."""
        chunks = self.chunker.chunk_text(text, metadata)
        return self.add_chunks(chunks)
    
    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """Add multiple chunks to vector store."""
        if not chunks:
            return 0
        
        documents = []
        metadatas = []
        ids = []
        embeddings = []
        
        for i, chunk in enumerate(chunks):
            text = chunk['text']
            
            # Create unique ID
            chunk_id = hashlib.md5(text.encode()).hexdigest()
            
            # Prepare metadata (remove 'text' key)
            metadata = {k: v for k, v in chunk.items() if k != 'text'}
            
            try:
                # Get embedding
                embedding = self.get_embedding(text)
                
                documents.append(text)
                metadatas.append(metadata)
                ids.append(chunk_id)
                embeddings.append(embedding)
                
            except Exception as e:
                print(f"Error processing chunk {i}: {e}")
                continue
        
        if documents:
            try:
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids,
                    embeddings=embeddings
                )
                print(f"Added {len(documents)} chunks to vector store")
                return len(documents)
            except Exception as e:
                print(f"Error adding to vector store: {e}")
                return 0
        
        return 0
    
    def search(self, query: str, n_results: int = 5, filter_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search vector store for relevant documents."""
        try:
            query_embedding = self.get_embedding(query)
            
            search_kwargs = {
                'query_embeddings': [query_embedding],
                'n_results': n_results
            }
            
            if filter_metadata:
                search_kwargs['where'] = filter_metadata
            
            results = self.collection.query(**search_kwargs)
            
            # Format results
            formatted_results = []
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i],
                    'id': results['ids'][0][i]
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"Error searching vector store: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        try:
            count = self.collection.count()
            return {
                'total_documents': count,
                'collection_name': self.collection_name
            }
        except Exception as e:
            print(f"Error getting collection stats: {e}")
            return {'total_documents': 0, 'collection_name': self.collection_name}
    
    def clear_collection(self):
        """Clear all documents from collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(name=self.collection_name)
            print("Collection cleared successfully")
        except Exception as e:
            print(f"Error clearing collection: {e}")
    
    def delete_repository(self, repo_name: str):
        """Delete all documents from a specific repository."""
        try:
            # Get all documents with repo_name
            results = self.collection.get(where={"repo_name": repo_name})
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                print(f"Deleted {len(results['ids'])} documents for repository {repo_name}")
        except Exception as e:
            print(f"Error deleting repository {repo_name}: {e}")
