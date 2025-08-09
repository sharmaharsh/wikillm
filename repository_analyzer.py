"""Repository analysis module for extracting code structure and content."""

import os
import ast
import re
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import git
from config import settings

@dataclass
class CodeElement:
    """Represents a code element (function, class, etc.)."""
    name: str
    type: str  # 'function', 'class', 'method', 'variable'
    file_path: str
    line_start: int
    line_end: int
    docstring: Optional[str] = None
    signature: Optional[str] = None
    content: Optional[str] = None
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []

@dataclass
class FileAnalysis:
    """Analysis results for a single file."""
    file_path: str
    file_type: str  # 'code', 'documentation', 'config'
    language: Optional[str] = None
    imports: List[str] = None
    elements: List[CodeElement] = None
    summary: Optional[str] = None
    content: Optional[str] = None
    
    def __post_init__(self):
        if self.imports is None:
            self.imports = []
        if self.elements is None:
            self.elements = []

class RepositoryAnalyzer:
    """Analyzes code repositories to extract structure and content."""
    
    def __init__(self):
        self.supported_languages = {
            '.py': 'python',
            '.js': 'javascript', 
            '.ts': 'typescript',
            '.jsx': 'javascript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.h': 'c',
            '.cs': 'csharp',
            '.php': 'php',
            '.rb': 'ruby',
            '.go': 'go',
            '.rs': 'rust',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.scala': 'scala'
        }
    
    def analyze_repository(self, repo_path: str) -> Dict[str, Any]:
        """Analyze an entire repository."""
        repo_path = Path(repo_path)
        
        if not repo_path.exists():
            raise ValueError(f"Repository path does not exist: {repo_path}")
        
        analysis = {
            'repo_path': str(repo_path),
            'repo_name': repo_path.name,
            'files': [],
            'structure': {},
            'statistics': {
                'total_files': 0,
                'code_files': 0,
                'doc_files': 0,
                'languages': {},
                'total_lines': 0
            }
        }
        
        # Get git info if available
        try:
            repo = git.Repo(repo_path)
            analysis['git_info'] = {
                'remote_url': repo.remotes.origin.url if repo.remotes else None,
                'current_branch': repo.active_branch.name,
                'last_commit': {
                    'hash': repo.head.commit.hexsha[:8],
                    'message': repo.head.commit.message.strip(),
                    'author': str(repo.head.commit.author),
                    'date': repo.head.commit.committed_datetime.isoformat()
                }
            }
        except:
            analysis['git_info'] = None
        
        # Analyze files
        for file_path in self._get_files_to_analyze(repo_path):
            file_analysis = self.analyze_file(file_path)
            if file_analysis:
                analysis['files'].append(asdict(file_analysis))
                
                # Update statistics
                analysis['statistics']['total_files'] += 1
                if file_analysis.file_type == 'code':
                    analysis['statistics']['code_files'] += 1
                    if file_analysis.language:
                        lang = file_analysis.language
                        analysis['statistics']['languages'][lang] = \
                            analysis['statistics']['languages'].get(lang, 0) + 1
                elif file_analysis.file_type == 'documentation':
                    analysis['statistics']['doc_files'] += 1
                
                # Count lines
                if file_analysis.content:
                    analysis['statistics']['total_lines'] += len(file_analysis.content.split('\n'))
        
        # Build directory structure
        analysis['structure'] = self._build_directory_structure(repo_path)
        
        return analysis
    
    def analyze_file(self, file_path: Path) -> Optional[FileAnalysis]:
        """Analyze a single file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return None
        
        file_ext = file_path.suffix.lower()
        relative_path = str(file_path)
        
        # Determine file type and language
        if file_ext in settings.CODE_EXTENSIONS:
            file_type = 'code'
            language = self.supported_languages.get(file_ext)
        elif file_ext in settings.DOC_EXTENSIONS:
            file_type = 'documentation'
            language = None
        else:
            file_type = 'config'
            language = None
        
        analysis = FileAnalysis(
            file_path=relative_path,
            file_type=file_type,
            language=language,
            content=content
        )
        
        # Language-specific analysis
        if language == 'python':
            self._analyze_python_file(analysis, content)
        elif language in ['javascript', 'typescript']:
            self._analyze_js_ts_file(analysis, content)
        
        return analysis
    
    def _analyze_python_file(self, analysis: FileAnalysis, content: str):
        """Analyze Python file for structure."""
        try:
            tree = ast.parse(content)
            
            # Extract imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        analysis.imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    for alias in node.names:
                        analysis.imports.append(f"{module}.{alias.name}")
            
            # Extract classes and functions
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    element = CodeElement(
                        name=node.name,
                        type='class',
                        file_path=analysis.file_path,
                        line_start=node.lineno,
                        line_end=getattr(node, 'end_lineno', node.lineno),
                        docstring=ast.get_docstring(node),
                        signature=f"class {node.name}"
                    )
                    analysis.elements.append(element)
                    
                    # Extract methods
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            method = CodeElement(
                                name=f"{node.name}.{item.name}",
                                type='method',
                                file_path=analysis.file_path,
                                line_start=item.lineno,
                                line_end=getattr(item, 'end_lineno', item.lineno),
                                docstring=ast.get_docstring(item),
                                signature=f"def {item.name}({self._get_function_args(item)})"
                            )
                            analysis.elements.append(method)
                
                elif isinstance(node, ast.FunctionDef):
                    # Only top-level functions (not methods)
                    if isinstance(getattr(node, 'parent', None), ast.Module) or not hasattr(node, 'parent'):
                        element = CodeElement(
                            name=node.name,
                            type='function',
                            file_path=analysis.file_path,
                            line_start=node.lineno,
                            line_end=getattr(node, 'end_lineno', node.lineno),
                            docstring=ast.get_docstring(node),
                            signature=f"def {node.name}({self._get_function_args(node)})"
                        )
                        analysis.elements.append(element)
                        
        except SyntaxError as e:
            print(f"Syntax error in {analysis.file_path}: {e}")
    
    def _analyze_js_ts_file(self, analysis: FileAnalysis, content: str):
        """Analyze JavaScript/TypeScript file for structure."""
        lines = content.split('\n')
        
        # Extract imports (simple regex-based approach)
        import_patterns = [
            r'import\s+.*?\s+from\s+[\'"]([^\'"]+)[\'"]',
            r'const\s+.*?\s*=\s*require\([\'"]([^\'"]+)[\'"]\)',
            r'import\s*\(\s*[\'"]([^\'"]+)[\'"]\s*\)'
        ]
        
        for line in lines:
            for pattern in import_patterns:
                matches = re.findall(pattern, line)
                analysis.imports.extend(matches)
        
        # Extract functions and classes (basic regex patterns)
        function_patterns = [
            r'function\s+(\w+)\s*\(',
            r'const\s+(\w+)\s*=\s*\(',
            r'(\w+)\s*:\s*function\s*\(',
            r'(\w+)\s*\([^)]*\)\s*=>',
            r'async\s+function\s+(\w+)\s*\('
        ]
        
        class_pattern = r'class\s+(\w+)'
        
        for i, line in enumerate(lines, 1):
            # Find classes
            class_match = re.search(class_pattern, line)
            if class_match:
                element = CodeElement(
                    name=class_match.group(1),
                    type='class',
                    file_path=analysis.file_path,
                    line_start=i,
                    line_end=i,  # We'd need more sophisticated parsing to find end
                    signature=line.strip()
                )
                analysis.elements.append(element)
            
            # Find functions
            for pattern in function_patterns:
                func_match = re.search(pattern, line)
                if func_match:
                    element = CodeElement(
                        name=func_match.group(1),
                        type='function',
                        file_path=analysis.file_path,
                        line_start=i,
                        line_end=i,
                        signature=line.strip()
                    )
                    analysis.elements.append(element)
    
    def _get_function_args(self, node: ast.FunctionDef) -> str:
        """Extract function arguments as string."""
        args = []
        for arg in node.args.args:
            args.append(arg.arg)
        return ', '.join(args)
    
    def _get_files_to_analyze(self, repo_path: Path) -> List[Path]:
        """Get list of files to analyze, respecting ignore patterns."""
        files = []
        
        for root, dirs, filenames in os.walk(repo_path):
            # Remove ignored directories
            dirs[:] = [d for d in dirs if not self._should_ignore_dir(d)]
            
            for filename in filenames:
                file_path = Path(root) / filename
                if self._should_analyze_file(file_path):
                    files.append(file_path)
        
        return files
    
    def _should_ignore_dir(self, dirname: str) -> bool:
        """Check if directory should be ignored."""
        return dirname in settings.IGNORED_DIRS or dirname.startswith('.')
    
    def _should_analyze_file(self, file_path: Path) -> bool:
        """Check if file should be analyzed."""
        ext = file_path.suffix.lower()
        return ext in settings.CODE_EXTENSIONS or ext in settings.DOC_EXTENSIONS
    
    def _build_directory_structure(self, repo_path: Path) -> Dict[str, Any]:
        """Build a tree representation of the directory structure."""
        def build_tree(path: Path) -> Dict[str, Any]:
            tree = {'name': path.name, 'type': 'directory', 'children': []}
            
            try:
                for item in sorted(path.iterdir()):
                    if item.is_dir() and not self._should_ignore_dir(item.name):
                        tree['children'].append(build_tree(item))
                    elif item.is_file() and self._should_analyze_file(item):
                        tree['children'].append({
                            'name': item.name,
                            'type': 'file',
                            'path': str(item.relative_to(repo_path))
                        })
            except PermissionError:
                pass
            
            return tree
        
        return build_tree(repo_path)

    def save_analysis(self, analysis: Dict[str, Any], output_path: str):
        """Save analysis results to JSON file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
    
    def load_analysis(self, input_path: str) -> Dict[str, Any]:
        """Load analysis results from JSON file."""
        with open(input_path, 'r', encoding='utf-8') as f:
            return json.load(f)
