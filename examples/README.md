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
- Retrieving memories across all tiers

**Run it:**
```bash
python examples/basic_usage.py
```

**What it demonstrates:**
- Stateless multi-agent design
- Template-driven memory structure
- Section-level update tracking
- Automatic consolidation triggers
- Memory retrieval

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    agent_reminiscence = AgentMem()
    await agent_reminiscence.initialize()
    
    try:
        # Your code here
        pass
        
    finally:
        await agent_reminiscence.close()

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

