# Contributing to Agent Mem

Thank you for your interest in contributing to Agent Mem! This document provides guidelines and instructions for contributing.

## Code of Conduct

Be respectful, inclusive, and professional. We aim to foster a welcoming community.

## How to Contribute

### Reporting Bugs

Before creating a bug report:
1. Check the [existing issues](https://github.com/yourusername/agent-reminiscence/issues)
2. Verify you're using the latest version
3. Test with a minimal reproducible example

**Good bug reports include:**
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details (Python version, OS, dependencies)
- Error messages and stack traces

### Suggesting Features

Feature requests are welcome! Please:
1. Check if it's already been suggested
2. Explain the use case and benefits
3. Consider implementation complexity
4. Be open to discussion

### Pull Requests

1. **Fork and clone** the repository
   ```bash
   git clone https://github.com/yourusername/agent-reminiscence.git
   cd agent-reminiscence
   ```

2. **Create a branch** for your changes
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Set up development environment**
   ```bash
   pip install -e ".[dev]"
   docker-compose up -d
   ```

4. **Make your changes**
   - Write clear, documented code
   - Follow existing code style
   - Add tests for new features
   - Update documentation

5. **Run tests and linters**
   ```bash
   # Run tests
   pytest tests/ -v
   
   # Check coverage
   pytest --cov=agent_reminiscence tests/
   
   # Format code
   black agent_reminiscence/ tests/ examples/
   
   # Lint code
   ruff check agent_reminiscence/ tests/ examples/
   
   # Type checking
   mypy agent_reminiscence/
   ```

6. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add awesome feature"
   ```
   
   Use conventional commit messages:
   - `feat:` - New feature
   - `fix:` - Bug fix
   - `docs:` - Documentation
   - `test:` - Tests
   - `refactor:` - Code refactoring
   - `chore:` - Maintenance

7. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then open a Pull Request on GitHub.

## Development Guidelines

### Code Style

- Follow PEP 8
- Use type hints
- Write docstrings (Google style)
- Keep functions focused and small
- Maximum line length: 100 characters

Example:

```python
async def create_memory(
    external_id: str,
    title: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None
) -> Memory:
    """
    Create a new memory entry.
    
    Args:
        external_id: Unique identifier for the agent
        title: Memory title
        content: Memory content
        metadata: Optional metadata dictionary
        
    Returns:
        Created Memory object
        
    Raises:
        ValueError: If external_id is empty
        DatabaseError: If creation fails
    """
    if not external_id:
        raise ValueError("external_id cannot be empty")
    # Implementation...
```

### Testing

- Write tests for new features
- Maintain or improve coverage
- Use pytest fixtures for setup
- Mock external dependencies
- Test error cases

```python
import pytest
from agent_reminiscence import AgentMem

@pytest.mark.asyncio
async def test_create_memory(agent_reminiscence):
    """Test memory creation."""
    memory = await agent_reminiscence.create_active_memory(
        external_id="test-agent",
        title="Test Memory",
        template_content=TEMPLATE
    )
    assert memory.title == "Test Memory"
```

### Documentation

- Update README.md for user-facing changes
- Add docstrings to all public APIs
- Update relevant markdown docs in `docs/`
- Add examples for new features

### Database Changes

If you modify the database schema:

1. Update `agent_reminiscence/sql/schema.sql`
2. Create migration script in `agent_reminiscence/sql/migrations/`
3. Update models in `agent_reminiscence/database/models.py`
4. Document changes in PR description

## Project Structure

```
agent_reminiscence/
├── agent_reminiscence/          # Main package
│   ├── core.py        # Main AgentMem class
│   ├── config/        # Configuration
│   ├── database/      # Database managers and repositories
│   ├── services/      # Business logic
│   ├── agents/        # AI agents
│   └── utils/         # Utilities
├── tests/             # Test suite
├── examples/          # Example scripts
├── docs/              # Documentation
└── sql/               # Database schemas
```

## Getting Help

- **Questions**: Open a [GitHub Discussion](https://github.com/yourusername/agent-reminiscence/discussions)
- **Bugs**: Open a [GitHub Issue](https://github.com/yourusername/agent-reminiscence/issues)
- **Chat**: Join our community (link TBD)

## Development Setup Checklist

- [ ] Python 3.10+ installed
- [ ] Docker and docker-compose installed
- [ ] Repository cloned and dependencies installed
- [ ] Services running (`docker-compose up -d`)
- [ ] Tests passing (`pytest tests/`)
- [ ] Pre-commit hooks configured (optional)

## Review Process

1. Automated tests must pass
2. Code review by maintainer(s)
3. Documentation updated
4. No merge conflicts
5. Approved by maintainer

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Recognition

Contributors will be recognized in:
- CHANGELOG.md
- GitHub contributors page
- Release notes

Thank you for contributing to Agent Mem! 🎉

