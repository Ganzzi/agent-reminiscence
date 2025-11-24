# Agent Mem Examples

This directory contains example scripts demonstrating how to use Agent Mem.

## Prerequisites

Before running examples, ensure you have:

1. Installed Agent Mem:
   ```bash
   pip install agent-mem
   # Or for development
   pip install -e ".[dev]"
   ```

2. Started services:
   ```bash
   docker-compose up -d
   docker exec -it agent_reminiscence_ollama ollama pull nomic-embed-text
   ```

3. Configured `.env` file (copy from `.env.example`)

## Examples

### `basic_usage.py`

Comprehensive example showing all core features:

- Creating stateless AgentMem instance
- Managing multiple agents simultaneously
- Creating active memories with templates
- Updating sections with automatic tracking
- Deep search with AI synthesis

**Run it:**
```bash
python examples/basic_usage.py
```

**What it demonstrates:**
- Stateless multi-agent design
- Template-driven memory structure
- Section-level update tracking
- Automatic consolidation triggers
- AI-powered deep search with synthesis

### `token_usage_tracking.py`

Learn how to monitor LLM token usage across operations:

- Register a usage processor callback
- Track tokens for deep search operations
- Calculate estimated costs
- Implement custom usage tracking logic

**Run it:**
```bash
python examples/token_usage_tracking.py
```

**What it demonstrates:**
- Setting up usage tracking with `set_usage_processor()`
- Monitoring token consumption
- Cost estimation
- Building custom usage processors

### `search_vs_deep_search.py`

Compare the two retrieval methods:

- `search_memories()` - Fast programmatic search
- `deep_search_memories()` - AI-powered search with synthesis

**Run it:**
```bash
python examples/search_vs_deep_search.py
```

**What it demonstrates:**
- Performance differences between methods
- When to use each method
- Token cost implications
- Search result quality comparison

### `database_test.py`

Low-level database connectivity test (no AI agents):

- PostgreSQL connection and queries
- Neo4j connection and queries
- Ollama embedding generation

**Run it:**
```bash
python examples/database_test.py
```

**What it demonstrates:**
- Direct database access
- Embedding service functionality
- Connection health checks

## Creating Your Own Examples

Basic template for a new example:

```python
import asyncio
import logging
from agent_reminiscence import AgentMem
from pydantic_ai import RunUsage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def usage_processor(external_id: str, usage: RunUsage):
    """Optional: Track token usage."""
    logger.info(f"{external_id}: {usage.total_tokens} tokens used")

async def main():
    agent_mem = AgentMem()
    await agent_mem.initialize()
    
    # Optional: Enable usage tracking
    agent_mem.set_usage_processor(usage_processor)
    
    try:
        # Your code here
        pass
        
    finally:
        await agent_mem.close()

if __name__ == "__main__":
    asyncio.run(main())
```

## Troubleshooting

### Database Connection Errors

```bash
# Check services are running
docker-compose ps

# View logs
docker-compose logs postgres
docker-compose logs neo4j
docker-compose logs ollama
```

### Ollama Not Responding

```bash
# Check Ollama is accessible
curl http://localhost:11434

# Verify model is pulled
docker exec -it agent_reminiscence_ollama ollama list

# Pull model if needed
docker exec -it agent_reminiscence_ollama ollama pull nomic-embed-text
```

### Import Errors

```bash
# Ensure package is installed
pip install -e .

# Or reinstall
pip uninstall agent-mem
pip install -e ".[dev]"
```

## Next Steps

- Read the [Documentation](../docs/index.md)
- Check the [API Reference](../docs/api/agent-mem.md)
- Review [Best Practices](../docs/guide/best-practices.md)

