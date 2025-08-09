"""Documentation generation system using local LLMs."""

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import markdown
from jinja2 import Template
from ollama_client import OllamaClient, DocumentationGenerator
from repository_analyzer import RepositoryAnalyzer
from config import settings

@dataclass
class DocumentationConfig:
    """Configuration for documentation generation."""
    include_overview: bool = True
    include_api_docs: bool = True
    include_examples: bool = True
    include_architecture: bool = True
    output_format: str = 'markdown'  # 'markdown', 'html'
    template_style: str = 'default'  # 'default', 'minimal', 'detailed'

class DocumentationBuilder:
    """Build comprehensive documentation from repository analysis."""
    
    def __init__(self, ollama_client: OllamaClient):
        self.ollama_client = ollama_client
        self.doc_generator = DocumentationGenerator(ollama_client)
        
        # Documentation templates
        self.templates = {
            'repository_readme': """# {{ repo_name }}

{{ overview }}

## Architecture

{{ architecture }}

## Getting Started

{{ getting_started }}

## API Documentation

{{ api_docs }}

## Examples

{{ examples }}

## Contributing

{{ contributing }}

""",
            
            'api_reference': """# API Reference - {{ file_name }}

{{ file_overview }}

## Classes

{{ classes }}

## Functions

{{ functions }}

## Constants

{{ constants }}

""",
            
            'module_docs': """# {{ module_name }}

{{ description }}

## Usage

{{ usage }}

## Implementation Details

{{ implementation }}

## Dependencies

{{ dependencies }}

"""
        }
    
    def generate_full_documentation(self, 
                                  repo_analysis: Dict[str, Any], 
                                  config: DocumentationConfig = None,
                                  output_dir: str = None) -> Dict[str, str]:
        """Generate complete documentation for a repository."""
        
        if config is None:
            config = DocumentationConfig()
        
        if output_dir is None:
            output_dir = settings.DOCS_DIRECTORY
        
        output_dir = Path(output_dir)
        repo_name = repo_analysis.get('repo_name', 'unknown')
        repo_output_dir = output_dir / repo_name
        repo_output_dir.mkdir(parents=True, exist_ok=True)
        
        documentation = {}
        
        # Generate main README
        if config.include_overview:
            readme_content = self.generate_repository_readme(repo_analysis)
            readme_path = repo_output_dir / 'README.md'
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(readme_content)
            documentation['README.md'] = readme_content
        
        # Generate API documentation for each file
        if config.include_api_docs:
            api_dir = repo_output_dir / 'api'
            api_dir.mkdir(exist_ok=True)
            
            for file_data in repo_analysis.get('files', []):
                if file_data.get('file_type') == 'code':
                    api_doc = self.generate_file_api_docs(file_data)
                    file_name = Path(file_data['file_path']).stem
                    api_file_path = api_dir / f"{file_name}.md"
                    
                    with open(api_file_path, 'w', encoding='utf-8') as f:
                        f.write(api_doc)
                    documentation[f'api/{file_name}.md'] = api_doc
        
        # Generate architecture documentation
        if config.include_architecture:
            arch_doc = self.generate_architecture_docs(repo_analysis)
            arch_path = repo_output_dir / 'ARCHITECTURE.md'
            with open(arch_path, 'w', encoding='utf-8') as f:
                f.write(arch_doc)
            documentation['ARCHITECTURE.md'] = arch_doc
        
        # Generate examples
        if config.include_examples:
            examples_doc = self.generate_examples_docs(repo_analysis)
            examples_path = repo_output_dir / 'EXAMPLES.md'
            with open(examples_path, 'w', encoding='utf-8') as f:
                f.write(examples_doc)
            documentation['EXAMPLES.md'] = examples_doc
        
        # Generate table of contents
        toc = self.generate_table_of_contents(documentation)
        toc_path = repo_output_dir / 'TABLE_OF_CONTENTS.md'
        with open(toc_path, 'w', encoding='utf-8') as f:
            f.write(toc)
        documentation['TABLE_OF_CONTENTS.md'] = toc
        
        return documentation
    
    def generate_repository_readme(self, repo_analysis: Dict[str, Any]) -> str:
        """Generate main repository README."""
        repo_name = repo_analysis.get('repo_name', 'Repository')
        stats = repo_analysis.get('statistics', {})
        
        # Build context for LLM
        context = []
        context.append(f"Repository: {repo_name}")
        context.append(f"Total files: {stats.get('total_files', 0)}")
        context.append(f"Code files: {stats.get('code_files', 0)}")
        context.append(f"Languages: {', '.join(stats.get('languages', {}).keys())}")
        
        # Add git information
        git_info = repo_analysis.get('git_info')
        if git_info:
            context.append(f"Current branch: {git_info.get('current_branch', 'main')}")
            if git_info.get('last_commit'):
                commit = git_info['last_commit']
                context.append(f"Last commit: {commit.get('message', '')[:100]}")
        
        # Add file structure overview
        main_files = []
        config_files = []
        doc_files = []
        
        for file_data in repo_analysis.get('files', []):
            file_path = file_data['file_path']
            if any(name in file_path.lower() for name in ['readme', 'license', 'changelog']):
                doc_files.append(file_path)
            elif any(ext in file_path.lower() for ext in ['.json', '.yaml', '.yml', '.toml', '.ini']):
                config_files.append(file_path)
            elif file_data.get('file_type') == 'code':
                main_files.append(file_path)
        
        context.append(f"Main code files: {', '.join(main_files[:10])}")
        if config_files:
            context.append(f"Configuration files: {', '.join(config_files[:5])}")
        if doc_files:
            context.append(f"Documentation files: {', '.join(doc_files)}")
        
        context_str = '\n'.join(context)
        
        prompt = f"""Generate a comprehensive README.md for this repository based on the following information:

{context_str}

Please include:
1. Project title and brief description
2. Features and capabilities
3. Installation instructions
4. Quick start guide
5. Usage examples
6. Project structure overview
7. Contributing guidelines
8. License information

Make it professional and well-formatted with proper Markdown syntax."""

        try:
            return self.ollama_client.generate(prompt, temperature=0.3)
        except Exception as e:
            return f"# {repo_name}\n\nError generating README: {str(e)}"
    
    def generate_file_api_docs(self, file_analysis: Dict[str, Any]) -> str:
        """Generate API documentation for a single file."""
        file_path = file_analysis['file_path']
        language = file_analysis.get('language', 'Unknown')
        elements = file_analysis.get('elements', [])
        
        # Group elements by type
        classes = [e for e in elements if e['type'] == 'class']
        functions = [e for e in elements if e['type'] in ['function', 'method']]
        
        doc_content = f"# API Reference - {Path(file_path).name}\n\n"
        doc_content += f"**File**: `{file_path}`  \n"
        doc_content += f"**Language**: {language}\n\n"
        
        # File overview
        overview_prompt = f"""Provide a brief overview of this {language} file:

File: {file_path}
Elements: {len(elements)} total ({len(classes)} classes, {len(functions)} functions)

Content preview:
{file_analysis.get('content', '')[:1000]}...

Write a 2-3 sentence overview of what this file does."""

        try:
            overview = self.ollama_client.generate(overview_prompt, temperature=0.3)
            doc_content += f"## Overview\n\n{overview}\n\n"
        except:
            doc_content += f"## Overview\n\nThis file contains {len(elements)} code elements.\n\n"
        
        # Document classes
        if classes:
            doc_content += "## Classes\n\n"
            for cls in classes:
                doc_content += f"### {cls['name']}\n\n"
                if cls.get('docstring'):
                    doc_content += f"{cls['docstring']}\n\n"
                
                # Get methods for this class
                class_methods = [e for e in elements if e['type'] == 'method' and e['name'].startswith(f"{cls['name']}.")]
                if class_methods:
                    doc_content += "#### Methods\n\n"
                    for method in class_methods:
                        method_name = method['name'].split('.')[-1]
                        doc_content += f"- **{method_name}**"
                        if method.get('signature'):
                            doc_content += f": `{method['signature']}`"
                        if method.get('docstring'):
                            doc_content += f" - {method['docstring'][:100]}..."
                        doc_content += "\n"
                    doc_content += "\n"
        
        # Document functions
        if functions:
            doc_content += "## Functions\n\n"
            for func in functions:
                if func['type'] == 'method':
                    continue  # Skip methods (already documented with classes)
                
                doc_content += f"### {func['name']}\n\n"
                if func.get('signature'):
                    doc_content += f"```{language}\n{func['signature']}\n```\n\n"
                if func.get('docstring'):
                    doc_content += f"{func['docstring']}\n\n"
        
        # Add imports if available
        imports = file_analysis.get('imports', [])
        if imports:
            doc_content += "## Dependencies\n\n"
            doc_content += "This file imports:\n\n"
            for imp in imports[:10]:  # Limit to first 10 imports
                doc_content += f"- `{imp}`\n"
            if len(imports) > 10:
                doc_content += f"- ... and {len(imports) - 10} more\n"
            doc_content += "\n"
        
        return doc_content
    
    def generate_architecture_docs(self, repo_analysis: Dict[str, Any]) -> str:
        """Generate architecture documentation."""
        repo_name = repo_analysis.get('repo_name', 'Repository')
        stats = repo_analysis.get('statistics', {})
        
        # Analyze project structure
        languages = stats.get('languages', {})
        files = repo_analysis.get('files', [])
        
        # Group files by directory
        directories = {}
        for file_data in files:
            dir_path = str(Path(file_data['file_path']).parent)
            if dir_path not in directories:
                directories[dir_path] = []
            directories[dir_path].append(file_data)
        
        # Build context for architecture analysis
        context = []
        context.append(f"Repository: {repo_name}")
        context.append(f"Languages: {', '.join(languages.keys())}")
        context.append(f"Total files: {stats.get('total_files', 0)}")
        
        context.append("\nDirectory structure:")
        for dir_path, dir_files in sorted(directories.items())[:15]:  # Limit directories
            file_count = len(dir_files)
            main_types = set()
            for f in dir_files:
                if f.get('language'):
                    main_types.add(f['language'])
            context.append(f"- {dir_path}: {file_count} files ({', '.join(main_types)})")
        
        # Add key files analysis
        key_files = []
        for file_data in files:
            if any(keyword in file_data['file_path'].lower() for keyword in ['main', 'app', 'index', '__init__', 'server', 'client']):
                key_files.append(file_data['file_path'])
        
        if key_files:
            context.append(f"\nKey files: {', '.join(key_files[:10])}")
        
        context_str = '\n'.join(context)
        
        prompt = f"""Analyze the architecture of this project and generate comprehensive architecture documentation:

{context_str}

Please provide:
1. High-level architecture overview
2. Component breakdown and relationships
3. Data flow and system interactions
4. Design patterns used
5. Technology stack analysis
6. Scalability considerations
7. Deployment architecture

Format as professional technical documentation with proper Markdown structure."""

        try:
            return self.ollama_client.generate(prompt, temperature=0.3)
        except Exception as e:
            return f"# Architecture Documentation\n\nError generating architecture docs: {str(e)}"
    
    def generate_examples_docs(self, repo_analysis: Dict[str, Any]) -> str:
        """Generate usage examples documentation."""
        repo_name = repo_analysis.get('repo_name', 'Repository')
        languages = repo_analysis.get('statistics', {}).get('languages', {})
        
        # Find main entry points
        entry_points = []
        main_classes = []
        main_functions = []
        
        for file_data in repo_analysis.get('files', []):
            if file_data.get('file_type') == 'code':
                elements = file_data.get('elements', [])
                
                # Look for main functions or entry points
                for element in elements:
                    if element['name'].lower() in ['main', 'run', 'start', 'init']:
                        entry_points.append({
                            'name': element['name'],
                            'file': file_data['file_path'],
                            'signature': element.get('signature', ''),
                            'type': element['type']
                        })
                    
                    if element['type'] == 'class' and len(element.get('name', '')) > 3:
                        main_classes.append({
                            'name': element['name'],
                            'file': file_data['file_path'],
                            'signature': element.get('signature', ''),
                            'docstring': element.get('docstring', '')
                        })
                    
                    if element['type'] == 'function' and element.get('docstring'):
                        main_functions.append({
                            'name': element['name'],
                            'file': file_data['file_path'],
                            'signature': element.get('signature', ''),
                            'docstring': element.get('docstring', '')
                        })
        
        # Build context
        context = []
        context.append(f"Repository: {repo_name}")
        context.append(f"Languages: {', '.join(languages.keys())}")
        
        if entry_points:
            context.append("\nEntry points:")
            for ep in entry_points[:5]:
                context.append(f"- {ep['name']} in {ep['file']}")
        
        if main_classes:
            context.append("\nMain classes:")
            for cls in main_classes[:5]:
                context.append(f"- {cls['name']} in {cls['file']}")
        
        if main_functions:
            context.append("\nKey functions:")
            for func in main_functions[:5]:
                context.append(f"- {func['name']} in {func['file']}: {func['docstring'][:50]}...")
        
        context_str = '\n'.join(context)
        
        prompt = f"""Generate practical usage examples and tutorials for this project:

{context_str}

Please provide:
1. Quick start example
2. Basic usage patterns
3. Common use cases with code examples
4. Integration examples
5. Configuration examples
6. Troubleshooting common issues

Include actual code snippets with explanations. Format with proper Markdown and code blocks."""

        try:
            return self.ollama_client.generate(prompt, temperature=0.3)
        except Exception as e:
            return f"# Usage Examples\n\nError generating examples: {str(e)}"
    
    def generate_table_of_contents(self, documentation: Dict[str, str]) -> str:
        """Generate table of contents for all documentation."""
        toc_content = "# Table of Contents\n\n"
        toc_content += "This repository contains the following documentation:\n\n"
        
        # Main documentation
        main_docs = ['README.md', 'ARCHITECTURE.md', 'EXAMPLES.md']
        for doc in main_docs:
            if doc in documentation:
                toc_content += f"- [{doc}](./{doc})\n"
        
        # API documentation
        api_docs = [k for k in documentation.keys() if k.startswith('api/')]
        if api_docs:
            toc_content += "\n## API Documentation\n\n"
            for doc in sorted(api_docs):
                file_name = Path(doc).stem
                toc_content += f"- [{file_name}](./{doc})\n"
        
        toc_content += f"\n---\n\n*Documentation generated automatically from codebase analysis.*"
        
        return toc_content
    
    def update_existing_docs(self, repo_path: str, analysis: Dict[str, Any]) -> Dict[str, str]:
        """Update existing documentation files with new analysis."""
        repo_path = Path(repo_path)
        updates = {}
        
        # Check for existing README
        readme_path = repo_path / 'README.md'
        if readme_path.exists():
            with open(readme_path, 'r', encoding='utf-8') as f:
                existing_readme = f.read()
            
            # Generate enhancement suggestions
            prompt = f"""Analyze this existing README and suggest improvements based on the codebase analysis:

Current README:
{existing_readme[:2000]}...

Codebase info:
- Files: {analysis.get('statistics', {}).get('total_files', 0)}
- Languages: {', '.join(analysis.get('statistics', {}).get('languages', {}).keys())}
- Key components: {len(analysis.get('files', []))} files analyzed

Suggest specific improvements while preserving the existing structure and content."""

            try:
                suggestions = self.ollama_client.generate(prompt, temperature=0.3)
                updates['README_suggestions.md'] = suggestions
            except Exception as e:
                updates['README_suggestions.md'] = f"Error generating suggestions: {str(e)}"
        
        return updates
