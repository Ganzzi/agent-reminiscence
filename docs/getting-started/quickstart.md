# Quick Start

Get started with Agent Mem in 5 minutes!

## Installation

```bash
pip install agent-mem
```

## Start Services

Agent Mem requires PostgreSQL, Neo4j, and Ollama. The easiest way is using Docker:

```bash
# Clone the repository (if not already done)
git clone https://github.com/yourusername/agent-mem.git
cd agent-mem

# Copy and configure environment
cp .env.example .env
# Edit .env with your settings

# Start services
docker-compose up -d

# Pull the embedding model
docker exec -it agent_mem_ollama ollama pull nomic-embed-text
```

## Basic Usage

Create a file `example.py`:

```python
from agent_mem import AgentMem
import asyncio

# Define a memory template
TASK_TEMPLATE = """
template:
  id: "task_memory_v1"
  name: "Task Memory"
sections:
  - id: "current_task"
    title: "Current Task"
  - id: "progress"
    title: "Progress"
"""

async def main():
    # Initialize memory manager
    agent_mem = AgentMem()
    await agent_mem.initialize()
    
    try:
        # Create active memory
        memory = await agent_mem.create_active_memory(
            external_id="my-agent",
            title="Build Dashboard",
            template_content=TASK_TEMPLATE,
            initial_sections={
                "current_task": {
                    "content": "Implementing analytics",
                    "update_count": 0
                },
                "progress": {
                    "content": "- Set up project\n- Created UI mockups",
                    "update_count": 0
                }
            }
        )
        print(f"✓ Created memory: {memory.title}")
        
        # Get all memories
        memories = await agent_mem.get_active_memories("my-agent")
        print(f"✓ Total memories: {len(memories)}")
        
        # Update a section
        await agent_mem.update_active_memory_section(
            external_id="my-agent",
            memory_id=memory.id,
            section_id="progress",
            new_content="- Set up project\n- Created UI mockups\n- Implemented charts"
        )
        print("✓ Updated progress section")
        
        # Retrieve memories
        results = await agent_mem.retrieve_memories(
            external_id="my-agent",
            query="What is the current progress?"
        )
        print(f"✓ Search results: {len(results.shortterm_results)} shortterm memories")
        
    finally:
        await agent_mem.close()

if __name__ == "__main__":
    asyncio.run(main())
```

Run it:

```bash
python example.py
```

## What's Next?

- **[Configuration Guide](configuration.md)**: Learn about all configuration options
- **[Memory Tiers](../guide/memory-tiers.md)**: Understand the three-tier memory system
- **[API Reference](../api/agent-mem.md)**: Explore the complete API
- **[Best Practices](../guide/best-practices.md)**: Tips for production use

## Troubleshooting

### Connection Errors

If you get database connection errors:

1. Verify services are running: `docker-compose ps`
2. Check logs: `docker-compose logs postgres` or `docker-compose logs neo4j`
3. Verify `.env` configuration matches service credentials

### Ollama Errors

If embedding generation fails:

1. Check Ollama is running: `curl http://localhost:11434`
2. Verify model is pulled: `docker exec -it agent_mem_ollama ollama list`
3. Pull model if missing: `docker exec -it agent_mem_ollama ollama pull nomic-embed-text`

For more help, see our [GitHub Issues](https://github.com/yourusername/agent-mem/issues).
