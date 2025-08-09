#!/usr/bin/env python3
"""
Local DeepWiki - Main entry point and CLI interface.

A local implementation of DeepWiki using Ollama for LLM inference.
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional

from config import settings
from repository_analyzer import RepositoryAnalyzer
from ollama_client import OllamaClient
from vector_store import VectorStore
from rag_system import RAGSystem
from documentation_generator import DocumentationBuilder, DocumentationConfig

class LocalDeepWiki:
    """Main class for Local DeepWiki functionality."""
    
    def __init__(self):
        """Initialize LocalDeepWiki with all components."""
        self.ollama_client = OllamaClient()
        self.vector_store = VectorStore()
        self.rag_system = RAGSystem(self.vector_store, self.ollama_client)
        self.repository_analyzer = RepositoryAnalyzer()
        self.documentation_builder = DocumentationBuilder(self.ollama_client)
    
    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are met."""
        print("Checking prerequisites...")
        
        # Check Ollama
        if not self.ollama_client.is_available():
            print("❌ Ollama server is not available at", settings.OLLAMA_BASE_URL)
            print("Please start Ollama first:")
            print("  ollama serve")
            return False
        
        print("✅ Ollama server is available")
        
        # Check model
        models = self.ollama_client.list_models()
        model_names = [model['name'] for model in models]
        
        if settings.OLLAMA_MODEL not in model_names:
            print(f"❌ Model '{settings.OLLAMA_MODEL}' is not available")
            print("Available models:", model_names)
            print(f"To download the model, run:")
            print(f"  ollama pull {settings.OLLAMA_MODEL}")
            return False
        
        print(f"✅ Model '{settings.OLLAMA_MODEL}' is available")
        
        # Check embedding model
        if settings.OLLAMA_EMBEDDING_MODEL not in model_names:
            print(f"⚠️  Embedding model '{settings.OLLAMA_EMBEDDING_MODEL}' not found")
            print("Will use fallback embedding model")
        else:
            print(f"✅ Embedding model '{settings.OLLAMA_EMBEDDING_MODEL}' is available")
        
        return True
    
    def analyze_repository(self, repo_path: str, repo_name: Optional[str] = None) -> bool:
        """Analyze a repository and add it to the vector store."""
        repo_path = Path(repo_path).resolve()
        
        if not repo_path.exists():
            print(f"❌ Repository path does not exist: {repo_path}")
            return False
        
        repo_name = repo_name or repo_path.name
        print(f"🔍 Analyzing repository: {repo_name}")
        print(f"   Path: {repo_path}")
        
        try:
            # Analyze repository structure
            print("   Parsing files and extracting structure...")
            analysis = self.repository_analyzer.analyze_repository(str(repo_path))
            analysis['repo_name'] = repo_name
            
            # Save analysis
            analysis_path = Path(settings.REPOS_DIRECTORY) / f"{repo_name}_analysis.json"
            self.repository_analyzer.save_analysis(analysis, str(analysis_path))
            print(f"   Analysis saved to: {analysis_path}")
            
            # Add to vector store
            print("   Adding to vector store...")
            chunks_added = self.vector_store.add_repository(analysis)
            
            print(f"✅ Repository analysis complete!")
            print(f"   Files analyzed: {analysis['statistics']['total_files']}")
            print(f"   Code files: {analysis['statistics']['code_files']}")
            print(f"   Languages: {', '.join(analysis['statistics']['languages'].keys())}")
            print(f"   Chunks added to vector store: {chunks_added}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error analyzing repository: {e}")
            return False
    
    def query_repository(self, query: str, repo_name: Optional[str] = None, stream: bool = True):
        """Query a repository using the RAG system."""
        print(f"🤔 Query: {query}")
        if repo_name:
            print(f"   Repository: {repo_name}")
        
        try:
            result = self.rag_system.query(query, repo_name=repo_name, stream=False)
            
            if result['context']:
                print("\n📖 Response:")
                print(result['response'])
                
                if result['sources']:
                    print("\n📚 Sources:")
                    for i, source in enumerate(result['sources'], 1):
                        print(f"   {i}. {source.get('file_path', 'Unknown')}")
                        if source.get('element_name'):
                            print(f"      Element: {source['element_name']}")
            else:
                print("❌ No relevant context found for your query.")
                
        except Exception as e:
            print(f"❌ Error processing query: {e}")
    
    def interactive_chat(self, repo_name: Optional[str] = None):
        """Start an interactive chat session."""
        print("💬 Interactive Chat Mode")
        print("Type 'quit' or 'exit' to end the session")
        if repo_name:
            print(f"Querying repository: {repo_name}")
        print("-" * 50)
        
        messages = []
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                messages.append({"role": "user", "content": user_input})
                
                print("\nAssistant: ", end="", flush=True)
                response = self.rag_system.chat_with_repo(messages, repo_name)
                print(response)
                
                messages.append({"role": "assistant", "content": response})
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
    def generate_documentation(self, repo_name: str):
        """Generate documentation for a repository."""
        print(f"📝 Generating documentation for: {repo_name}")
        
        # Load analysis
        analysis_path = Path(settings.REPOS_DIRECTORY) / f"{repo_name}_analysis.json"
        if not analysis_path.exists():
            print(f"❌ Repository {repo_name} not found. Please analyze it first.")
            return False
        
        try:
            analysis = self.repository_analyzer.load_analysis(str(analysis_path))
            
            config = DocumentationConfig(
                include_overview=True,
                include_api_docs=True,
                include_examples=True,
                include_architecture=True
            )
            
            docs = self.documentation_builder.generate_full_documentation(analysis, config)
            
            print(f"✅ Documentation generated!")
            print(f"   Files created: {len(docs)}")
            print(f"   Output directory: {Path(settings.DOCS_DIRECTORY) / repo_name}")
            
            for doc_name in docs.keys():
                print(f"   - {doc_name}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error generating documentation: {e}")
            return False
    
    def list_repositories(self):
        """List all analyzed repositories."""
        repos_dir = Path(settings.REPOS_DIRECTORY)
        
        if not repos_dir.exists():
            print("No repositories found.")
            return
        
        analysis_files = list(repos_dir.glob("*_analysis.json"))
        
        if not analysis_files:
            print("No repositories found.")
            return
        
        print("📚 Analyzed Repositories:")
        print("-" * 50)
        
        for analysis_file in analysis_files:
            try:
                analysis = self.repository_analyzer.load_analysis(str(analysis_file))
                repo_name = analysis.get('repo_name', analysis_file.stem.replace('_analysis', ''))
                stats = analysis.get('statistics', {})
                
                print(f"📁 {repo_name}")
                print(f"   Path: {analysis.get('repo_path', 'Unknown')}")
                print(f"   Files: {stats.get('total_files', 0)} total, {stats.get('code_files', 0)} code")
                print(f"   Languages: {', '.join(stats.get('languages', {}).keys())}")
                print()
                
            except Exception as e:
                print(f"❌ Error loading {analysis_file}: {e}")
    
    def show_stats(self):
        """Show system statistics."""
        print("📊 System Statistics")
        print("-" * 30)
        
        # Vector store stats
        vector_stats = self.vector_store.get_collection_stats()
        print(f"Documents in vector store: {vector_stats.get('total_documents', 0)}")
        
        # Repository count
        repos_dir = Path(settings.REPOS_DIRECTORY)
        repo_count = len(list(repos_dir.glob("*_analysis.json"))) if repos_dir.exists() else 0
        print(f"Analyzed repositories: {repo_count}")
        
        # Generated docs count
        docs_dir = Path(settings.DOCS_DIRECTORY)
        doc_count = len(list(docs_dir.rglob("*.md"))) if docs_dir.exists() else 0
        print(f"Generated documentation files: {doc_count}")
        
        # Ollama status
        print(f"Ollama available: {'✅' if self.ollama_client.is_available() else '❌'}")
        print(f"Current model: {settings.OLLAMA_MODEL}")

def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Local DeepWiki - Chat with your code repositories")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Check command
    subparsers.add_parser('check', help='Check system prerequisites')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a repository')
    analyze_parser.add_argument('repo_path', help='Path to the repository')
    analyze_parser.add_argument('--name', help='Custom name for the repository')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query a repository')
    query_parser.add_argument('query', help='Your question or query')
    query_parser.add_argument('--repo', help='Repository name to query')
    
    # Chat command
    chat_parser = subparsers.add_parser('chat', help='Start interactive chat')
    chat_parser.add_argument('--repo', help='Repository name to chat with')
    
    # Docs command
    docs_parser = subparsers.add_parser('docs', help='Generate documentation')
    docs_parser.add_argument('repo_name', help='Repository name')
    
    # List command
    subparsers.add_parser('list', help='List analyzed repositories')
    
    # Stats command
    subparsers.add_parser('stats', help='Show system statistics')
    
    # Server command
    server_parser = subparsers.add_parser('server', help='Start web server')
    server_parser.add_argument('--host', default=settings.API_HOST, help='Host address')
    server_parser.add_argument('--port', type=int, default=settings.API_PORT, help='Port number')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize LocalDeepWiki
    ldw = LocalDeepWiki()
    
    if args.command == 'check':
        if ldw.check_prerequisites():
            print("\n✅ All prerequisites are met! You're ready to use Local DeepWiki.")
        else:
            print("\n❌ Please fix the issues above before using Local DeepWiki.")
            sys.exit(1)
    
    elif args.command == 'analyze':
        if not ldw.check_prerequisites():
            sys.exit(1)
        ldw.analyze_repository(args.repo_path, args.name)
    
    elif args.command == 'query':
        if not ldw.check_prerequisites():
            sys.exit(1)
        ldw.query_repository(args.query, args.repo)
    
    elif args.command == 'chat':
        if not ldw.check_prerequisites():
            sys.exit(1)
        ldw.interactive_chat(args.repo)
    
    elif args.command == 'docs':
        if not ldw.check_prerequisites():
            sys.exit(1)
        ldw.generate_documentation(args.repo_name)
    
    elif args.command == 'list':
        ldw.list_repositories()
    
    elif args.command == 'stats':
        ldw.show_stats()
    
    elif args.command == 'server':
        if not ldw.check_prerequisites():
            sys.exit(1)
        
        # Import and run the API server
        from api import main as run_server
        run_server()

if __name__ == "__main__":
    main()
