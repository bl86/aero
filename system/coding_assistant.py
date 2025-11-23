import os
import ast
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import tree_sitter_languages
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import TerminalFormatter

class CodingAssistant:
    def __init__(self, config, llm_engine):
        self.config = config
        self.llm_engine = llm_engine

        self.base_dir = Path(config['system']['install_dir']).expanduser()
        self.workspace_dir = self.base_dir / "workspace"
        self.workspace_dir.mkdir(exist_ok=True)

        self.context_window = config['coding'].get('context_window', 16000)

        self.language_extensions = {
            'python': ['.py'],
            'javascript': ['.js', '.jsx', '.ts', '.tsx'],
            'rust': ['.rs'],
            'go': ['.go'],
            'c': ['.c', '.h'],
            'cpp': ['.cpp', '.hpp', '.cc'],
            'java': ['.java'],
            'ruby': ['.rb'],
            'php': ['.php'],
        }

    def assist(self, task: str, language: str = 'python',
              context_files: List[str] = None) -> str:

        context = self._build_context(context_files or [])

        prompt = f"""You are a senior software developer assistant.

Task: {task}
Language: {language}

Context:
{context}

Provide a complete, production-ready solution with:
1. Clean, well-documented code
2. Error handling
3. Type hints (if applicable)
4. Comments explaining complex logic
5. Best practices

Solution:
"""

        response = self.llm_engine.generate(
            prompt,
            max_tokens=4096,
            temperature=0.2
        )

        return response

    def _build_context(self, files: List[str]) -> str:
        context_parts = []
        total_length = 0

        for file_path in files:
            if not os.path.exists(file_path):
                continue

            with open(file_path, 'r') as f:
                content = f.read()

            if total_length + len(content) > self.context_window:
                break

            context_parts.append(f"File: {file_path}\n```\n{content}\n```\n")
            total_length += len(content)

        return "\n".join(context_parts)

    def generate_code(self, description: str, language: str = 'python') -> str:
        prompt = f"""Generate {language} code for:

{description}

Requirements:
- Clean, readable code
- Proper error handling
- Documentation
- Type annotations
- Following best practices

Code:
"""

        return self.llm_engine.generate(prompt, max_tokens=4096, temperature=0.2)

    def explain_code(self, code: str, language: str = 'python') -> str:
        prompt = f"""Explain the following {language} code:

```{language}
{code}
```

Provide:
1. High-level overview
2. Step-by-step explanation
3. Key concepts used
4. Potential improvements

Explanation:
"""

        return self.llm_engine.generate(prompt, max_tokens=2048, temperature=0.3)

    def debug_code(self, code: str, error: str, language: str = 'python') -> str:
        prompt = f"""Debug this {language} code:

Code:
```{language}
{code}
```

Error:
```
{error}
```

Provide:
1. Explanation of the error
2. Root cause
3. Fixed code
4. Prevention tips

Solution:
"""

        return self.llm_engine.generate(prompt, max_tokens=3072, temperature=0.2)

    def refactor_code(self, code: str, language: str = 'python') -> str:
        prompt = f"""Refactor this {language} code for better quality:

```{language}
{code}
```

Improve:
- Code structure
- Performance
- Readability
- Best practices
- Design patterns

Refactored code:
"""

        return self.llm_engine.generate(prompt, max_tokens=4096, temperature=0.2)

    def add_tests(self, code: str, language: str = 'python') -> str:
        test_frameworks = {
            'python': 'pytest',
            'javascript': 'jest',
            'rust': 'built-in test framework',
            'go': 'testing package',
            'java': 'JUnit'
        }

        framework = test_frameworks.get(language, 'appropriate test framework')

        prompt = f"""Generate comprehensive tests for this {language} code using {framework}:

```{language}
{code}
```

Include:
- Unit tests for all functions
- Edge cases
- Error cases
- Integration tests if applicable

Test code:
"""

        return self.llm_engine.generate(prompt, max_tokens=4096, temperature=0.2)

    def optimize_code(self, code: str, language: str = 'python') -> str:
        prompt = f"""Optimize this {language} code for performance:

```{language}
{code}
```

Focus on:
- Time complexity
- Space complexity
- Algorithm efficiency
- Resource usage

Provide optimized code and explain improvements.

Optimized code:
"""

        return self.llm_engine.generate(prompt, max_tokens=4096, temperature=0.2)

    def code_review(self, code: str, language: str = 'python') -> str:
        prompt = f"""Perform a code review on this {language} code:

```{language}
{code}
```

Review for:
- Code quality
- Best practices
- Security issues
- Performance concerns
- Maintainability
- Documentation

Review:
"""

        return self.llm_engine.generate(prompt, max_tokens=3072, temperature=0.3)

    def convert_code(self, code: str, from_lang: str, to_lang: str) -> str:
        prompt = f"""Convert this {from_lang} code to {to_lang}:

{from_lang} code:
```{from_lang}
{code}
```

Provide equivalent {to_lang} code following {to_lang} best practices.

{to_lang} code:
"""

        return self.llm_engine.generate(prompt, max_tokens=4096, temperature=0.2)

    def generate_documentation(self, code: str, language: str = 'python',
                             doc_type: str = 'inline') -> str:

        if doc_type == 'inline':
            prompt = f"""Add inline documentation to this {language} code:

```{language}
{code}
```

Include:
- Docstrings/comments for functions and classes
- Complex logic explanations
- Parameter descriptions
- Return value descriptions

Documented code:
"""
        else:
            prompt = f"""Generate external documentation for this {language} code:

```{language}
{code}
```

Create markdown documentation with:
- Overview
- API reference
- Usage examples
- Requirements

Documentation:
"""

        return self.llm_engine.generate(prompt, max_tokens=4096, temperature=0.2)

    def suggest_improvements(self, code: str, language: str = 'python') -> List[str]:
        prompt = f"""Analyze this {language} code and suggest improvements:

```{language}
{code}
```

List specific, actionable improvements for:
- Performance
- Security
- Maintainability
- Error handling
- Code style

Suggestions:
"""

        response = self.llm_engine.generate(prompt, max_tokens=2048, temperature=0.3)

        suggestions = [s.strip() for s in response.split('\n') if s.strip()]

        return suggestions

    def analyze_complexity(self, code: str, language: str = 'python') -> Dict:
        if language != 'python':
            return {'error': 'Complexity analysis currently only supports Python'}

        try:
            tree = ast.parse(code)

            functions = []
            classes = []

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    complexity = self._calculate_complexity(node)
                    functions.append({
                        'name': node.name,
                        'complexity': complexity,
                        'lines': node.end_lineno - node.lineno
                    })
                elif isinstance(node, ast.ClassDef):
                    classes.append({'name': node.name})

            return {
                'functions': functions,
                'classes': classes,
                'total_functions': len(functions),
                'total_classes': len(classes)
            }

        except Exception as e:
            return {'error': str(e)}

    def _calculate_complexity(self, node) -> int:
        complexity = 1

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1

        return complexity

    def create_project(self, name: str, language: str, project_type: str) -> str:
        project_dir = self.workspace_dir / name

        if project_dir.exists():
            return f"Project '{name}' already exists"

        project_dir.mkdir()

        templates = {
            'python': self._create_python_project,
            'javascript': self._create_js_project,
            'rust': self._create_rust_project,
        }

        creator = templates.get(language)
        if creator:
            creator(project_dir, project_type)
            return f"Project '{name}' created at {project_dir}"
        else:
            return f"Unsupported language: {language}"

    def _create_python_project(self, project_dir: Path, project_type: str):
        (project_dir / "src").mkdir()
        (project_dir / "tests").mkdir()

        with open(project_dir / "requirements.txt", 'w') as f:
            f.write("pytest\nblack\npylint\n")

        with open(project_dir / "src" / "__init__.py", 'w') as f:
            f.write("")

        with open(project_dir / "src" / "main.py", 'w') as f:
            f.write('def main():\n    print("Hello, World!")\n\nif __name__ == "__main__":\n    main()\n')

    def _create_js_project(self, project_dir: Path, project_type: str):
        (project_dir / "src").mkdir()
        (project_dir / "tests").mkdir()

        package_json = {
            "name": project_dir.name,
            "version": "1.0.0",
            "scripts": {
                "test": "jest"
            }
        }

        import json
        with open(project_dir / "package.json", 'w') as f:
            json.dump(package_json, f, indent=2)

    def _create_rust_project(self, project_dir: Path, project_type: str):
        subprocess.run(['cargo', 'init', str(project_dir)], check=True)

    def execute_code(self, code: str, language: str = 'python') -> Tuple[str, str]:
        if language == 'python':
            return self._execute_python(code)
        else:
            return "", "Execution only supported for Python currently"

    def _execute_python(self, code: str) -> Tuple[str, str]:
        temp_file = self.workspace_dir / "temp_exec.py"

        with open(temp_file, 'w') as f:
            f.write(code)

        try:
            result = subprocess.run(
                ['python3', str(temp_file)],
                capture_output=True,
                text=True,
                timeout=10
            )

            return result.stdout, result.stderr

        except subprocess.TimeoutExpired:
            return "", "Execution timed out"
        except Exception as e:
            return "", str(e)
        finally:
            if temp_file.exists():
                temp_file.unlink()
