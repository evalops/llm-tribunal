# Contributing to LLM Tribunal

Thank you for your interest in contributing to LLM Tribunal! This document provides guidelines for contributing to the project.

## Code of Conduct

This project is committed to providing a welcoming and inclusive environment for all contributors. Please be respectful and professional in all interactions.

## How to Contribute

### Reporting Issues

1. **Search existing issues** first to avoid duplicates
2. **Use the issue template** when creating new issues
3. **Provide clear steps to reproduce** any bugs
4. **Include system information** (Python version, OS, etc.)

### Submitting Changes

1. **Fork the repository** and create a new branch
2. **Follow the coding standards** outlined below
3. **Add tests** for new functionality
4. **Update documentation** as needed
5. **Submit a pull request** with a clear description

## Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/llm-tribunal.git
cd llm-tribunal

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-cov black flake8

# Run tests to verify setup
pytest tests/
```

## Coding Standards

### Python Style

- Follow **PEP 8** style guidelines
- Use **black** for code formatting: `black .`
- Use **flake8** for linting: `flake8 .`
- Maximum line length: 88 characters (black default)

### Node Development

When creating new evaluation nodes:

1. **Inherit from BaseNode** and use the `@register_node` decorator
2. **Implement the `run` method** with proper error handling
3. **Return NodeResult** with appropriate status and outputs
4. **Add comprehensive docstrings** and type hints
5. **Create tests** in the `tests/` directory

Example node structure:

```python
from dag_engine import BaseNode, NodeResult, NodeStatus, register_node
from typing import Dict, Any

@register_node("MyNode")
class MyNode(BaseNode):
    """Brief description of what this node does.
    
    Detailed description including:
    - Input requirements
    - Output format
    - Configuration options
    - Example usage
    """
    
    def run(self, context: ExecutionContext) -> NodeResult:
        try:
            # Validate inputs
            if not context.has_artifact("required_input"):
                return NodeResult(
                    status=NodeStatus.FAILED,
                    error="Missing required input"
                )
            
            # Process data
            result = self._process_data(context.get_artifact("required_input"))
            
            # Return success
            return NodeResult(
                status=NodeStatus.SUCCESS,
                outputs={"output_name": result}
            )
            
        except Exception as e:
            return NodeResult(
                status=NodeStatus.FAILED,
                error=f"Node failed: {str(e)}"
            )
    
    def _process_data(self, data: Any) -> Any:
        """Private helper method for data processing."""
        # Implementation here
        pass
```

### Testing

- **Write unit tests** for all new functionality
- **Aim for >80% code coverage**
- **Use descriptive test names** that explain what is being tested
- **Test both success and failure cases**
- **Mock external API calls** in tests

Example test structure:

```python
import pytest
from unittest.mock import Mock, patch
from dag_engine import ExecutionContext, NodeStatus
from nodes.mymodule import MyNode

class TestMyNode:
    def test_successful_execution(self):
        """Test that MyNode processes data correctly."""
        node = MyNode("test_node")
        context = ExecutionContext()
        context.set_artifact("required_input", "test_data")
        
        result = node.run(context)
        
        assert result.status == NodeStatus.SUCCESS
        assert "output_name" in result.outputs
    
    def test_missing_input_fails(self):
        """Test that MyNode fails gracefully with missing input."""
        node = MyNode("test_node")
        context = ExecutionContext()
        
        result = node.run(context)
        
        assert result.status == NodeStatus.FAILED
        assert "Missing required input" in result.error
```

### Documentation

- **Update README.md** for user-facing changes
- **Add docstrings** to all classes and methods
- **Include examples** in docstrings where helpful
- **Update pipeline examples** when adding new node types

## Project Structure

```
llm-tribunal/
├── cli.py                  # Command-line interface
├── dag_engine.py           # Core DAG execution engine
├── judge_nodes.py          # LLM-as-Judge evaluation nodes
├── evaluation_nodes.py     # Compatibility layer
├── config.py               # Configuration management
├── nodes/                  # Modular node implementations
│   ├── core.py            # Basic data operations
│   ├── llm_providers.py   # LLM setup nodes
│   ├── evaluation.py      # Evaluation nodes
│   ├── synthetic.py       # Synthetic test generation
│   └── analysis.py        # Analysis and reporting
├── utils/                  # Utility modules
│   ├── logging_setup.py   # Structured logging
│   ├── metrics.py         # Performance metrics
│   └── error_handling.py  # Error management
├── tests/                  # Test suite
│   ├── test_dag_engine.py
│   ├── test_judge_nodes.py
│   └── ...
├── examples/               # Example configurations
│   ├── pipelines/         # Pipeline YAML files
│   └── configs/           # Configuration examples
└── README.md              # Main documentation
```

## Adding New Features

### New Node Types

1. **Create the node class** in the appropriate `nodes/` module
2. **Add imports** in `evaluation_nodes.py`
3. **Register with DAG engine** in `dag_engine.py`
4. **Create example pipeline** in `examples/pipelines/`
5. **Add comprehensive tests**
6. **Update documentation**

### New LLM Providers

1. **Add provider setup** in `nodes/llm_providers.py`
2. **Follow existing patterns** for API integration
3. **Add error handling** and retry logic
4. **Create configuration examples**
5. **Test with multiple models**

### New Evaluation Scales

1. **Add scale class** in `judge_nodes.py`
2. **Inherit from base scale** class
3. **Implement validation logic**
4. **Add usage examples**
5. **Test edge cases**

## Pull Request Process

1. **Create descriptive branch name**: `feature/add-new-node` or `fix/bug-description`
2. **Write clear commit messages** following conventional commits format
3. **Ensure all tests pass**: `pytest tests/`
4. **Run code formatting**: `black .`
5. **Run linting**: `flake8 .`
6. **Update CHANGELOG.md** if applicable
7. **Request review** from maintainers

### Commit Message Format

```
type(scope): description

- feat: new feature
- fix: bug fix
- docs: documentation changes
- style: formatting changes
- refactor: code refactoring
- test: adding tests
- chore: maintenance tasks

Examples:
feat(nodes): add PromptVariationGenerator node
fix(dag): resolve cycle detection bug
docs(readme): update installation instructions
```

## Security Guidelines

This project is designed for **defensive security research** and **AI safety evaluation**:

### Acceptable Use
- ✅ LLM safety assessment and vulnerability detection
- ✅ Academic research on LLM threats and mitigations  
- ✅ Authorized red team evaluation and security testing
- ✅ Development of defensive AI safety tools

### Prohibited Use
- ❌ Actual attacks or malicious use against systems
- ❌ Generating harmful content for distribution
- ❌ Testing systems without proper authorization
- ❌ Violating API terms of service

### Contribution Guidelines
- **Review all synthetic content generation** for potential misuse
- **Ensure test cases are educational** and clearly marked as research
- **Follow responsible disclosure** for any vulnerabilities discovered
- **Respect rate limits** and API usage guidelines

## Release Process

1. **Version bumping** follows semantic versioning (MAJOR.MINOR.PATCH)
2. **Update CHANGELOG.md** with new features and fixes
3. **Tag releases** in GitHub with release notes
4. **Coordinate with maintainers** for release timing

## Getting Help

- **Join GitHub Discussions** for general questions
- **Create GitHub Issues** for bugs and feature requests
- **Review existing documentation** in `examples/` directory
- **Check test files** for usage examples

## Recognition

Contributors will be recognized in:
- **CONTRIBUTORS.md** file
- **Release notes** for significant contributions
- **GitHub contributors** section

Thank you for contributing to LLM Tribunal! 🚀