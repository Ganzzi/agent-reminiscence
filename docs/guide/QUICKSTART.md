# Quick Start Guide

Get started with agent-reminiscence in 5 minutes.

## Installation

### Prerequisites
- Python 3.10+
- PostgreSQL 14+ with pgvector extension
- Neo4j 5+
- Ollama (for embeddings)

### Setup with Docker (Recommended)

```bash
# Start services
docker-compose up -d

# Verify services are running
docker ps
# Should show: postgres, neo4j, ollama
```

### Install Package

```bash
pip install agent-reminiscence
```

Or install from source:

```bash
git clone https://github.com/yourusername/agent-reminiscence.git
cd agent-reminiscence
pip install -e .
```

### Configure Environment

Create `.env` file:

```bash
# PostgreSQL
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=agent_reminiscence
POSTGRES_USER=postgres
POSTGRES_PASSWORD=password

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_DIMENSION=768

# LLM (for synthesis)
LLM_MODEL=gpt-4  # or your preferred model
```

---

## Your First Memory

### 1. Basic Memory Creation

```python
import asyncio
from agent_reminiscence import AgentMem

async def main():
    # Initialize with context manager
    async with AgentMem() as memory:
        # Create a memory
        my_memory = await memory.create_active_memory(
            external_id="my-agent",
            title="Project Status",
            template_content={
                "template": {
                    "id": "project_v1",
                    "name": "Project Memory"
                },
                "sections": [
                    {
                        "id": "goals",
                        "description": "Project goals"
                    },
                    {
                        "id": "progress",
                        "description": "Current progress"
                    }
                ]
            },
            initial_sections={
                "goals": {
                    "content": "Launch v1.0 by Q1 2026"
                },
                "progress": {
                    "content": "Just started - 0% complete"
                }
            }
        )
        
        print(f"✅ Created memory: {my_memory.title}")
        print(f"   ID: {my_memory.id}")

asyncio.run(main())
```

### 2. Update Your Memory

```python
async def update_example():
    async with AgentMem() as memory:
        # Update progress
        result = await memory.update_active_memory(
            external_id="my-agent",
            memory_id=1,  # From previous example
            sections=[
                {
                    "section_id": "progress",
                    "action": "replace",
                    "new_content": "Phase 1 complete - 25% done"
                }
            ]
        )
        
        print(f"✅ Updated memory")

asyncio.run(update_example())
```

### 3. Retrieve Your Memories

```python
async def retrieve_example():
    async with AgentMem() as memory:
        # Get all memories
        memories = await memory.get_active_memories(
            external_id="my-agent"
        )
        
        print(f"Found {len(memories)} memories:")
        for mem in memories:
            print(f"  - {mem.title}")

asyncio.run(retrieve_example())
```

---

## Search Operations

### Fast Search (< 200ms)

Use for quick lookups:

```python
async def fast_search():
    async with AgentMem() as memory:
        results = await memory.search_memories(
            external_id="my-agent",
            query="project goals and timeline",
            limit=5
        )
        
        print(f"Found {len(results.chunks)} chunks")
        for chunk in results.chunks:
            print(f"  Score: {chunk.score:.2%}")
            print(f"  Content: {chunk.content[:100]}...")

asyncio.run(fast_search())
```

**Use When**:
- You need quick results
- Looking for specific information
- Response time is critical
- Building UI/interfaces

### Deep Search (500ms-2s) ⭐ NEW

Use for comprehensive analysis:

```python
async def deep_search():
    async with AgentMem() as memory:
        results = await memory.deep_search_memories(
            external_id="my-agent",
            query="How is project progress related to our goals?",
            limit=10
        )
        
        print(f"Synthesis: {results.synthesis}")
        print(f"\nKey Entities:")
        for entity in results.entities:
            print(f"  - {entity.name}: {entity.description}")

asyncio.run(deep_search())
```

**Use When**:
- You need comprehensive analysis
- Asking complex questions
- Want entity relationships
- Need AI synthesis

---

## Claude Desktop Integration ⭐

Use agent-reminiscence tools directly in Claude Desktop.

### Step 1: Start MCP Server

```bash
python -m agent_reminiscence_mcp.run
```

Output should show:
```
🚀 Starting Agent Reminiscence MCP Server...
✅ Serving on stdio
```

### Step 2: Configure Claude Desktop

**macOS**: `~/.config/claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "agent-mem": {
      "command": "python",
      "args": ["-m", "agent_reminiscence_mcp.run"],
      "env": {
        "POSTGRES_HOST": "localhost",
        "NEO4J_URI": "bolt://localhost:7687",
        "OLLAMA_BASE_URL": "http://localhost:11434"
      }
    }
  }
}
```

**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

(Same JSON format as macOS)

### Step 3: Restart Claude Desktop

Claude will automatically load all 6 memory tools.

### Step 4: Use Tools in Claude

```
You: "Create a memory to track my learning progress"

Claude: I'll create a memory for you to track your learning.
[Uses: create_active_memory(...)]
✅ Created memory with sections for goals, resources, and progress

You: "Search for information about Python concepts I've learned"

Claude: I'll search your memories for Python concepts.
[Uses: search_memories(...)]
Found 5 relevant chunks about decorators, async/await, and type hints

You: "Deep search - analyze how all my learning connects together"

Claude: I'll perform a comprehensive analysis.
[Uses: deep_search_memories(...)]
Synthesis: Your Python learning has progressed from basic concepts 
through async programming. Key relationships show connections between 
decorators and type hints through practical examples...
```

---

## Common Patterns

### Pattern 1: Track Project Progress

```python
async def project_tracking():
    async with AgentMem() as memory:
        # Create project memory
        project = await memory.create_active_memory(
            external_id="project-123",
            title="Website Redesign",
            template_content={
                "template": {"id": "project_v1", "name": "Project"},
                "sections": [
                    {"id": "milestones", "description": "Key milestones"},
                    {"id": "progress", "description": "Progress updates"},
                    {"id": "blockers", "description": "Known issues"}
                ]
            }
        )
        
        # Update progress regularly
        for i in range(1, 4):
            await memory.update_active_memory(
                external_id="project-123",
                memory_id=project.id,
                sections=[
                    {
                        "section_id": "progress",
                        "action": "replace",
                        "new_content": f"Sprint {i} complete - {i*25}%"
                    }
                ]
            )
        
        # Search for specific phase
        phase_search = await memory.search_memories(
            external_id="project-123",
            query="sprint completed",
            limit=3
        )

asyncio.run(project_tracking())
```

### Pattern 2: Knowledge Base

```python
async def knowledge_base():
    async with AgentMem() as memory:
        # Create specialized memories
        topics = ["Python", "JavaScript", "SQL"]
        
        for topic in topics:
            await memory.create_active_memory(
                external_id="learning-hub",
                title=f"{topic} Guide",
                template_content={
                    "template": {"id": f"{topic.lower()}_v1", "name": topic},
                    "sections": [
                        {"id": "basics", "description": "Basic concepts"},
                        {"id": "advanced", "description": "Advanced topics"}
                    ]
                }
            )
        
        # Search across all
        results = await memory.deep_search_memories(
            external_id="learning-hub",
            query="What are the key concepts across all languages?",
            limit=20
        )

asyncio.run(knowledge_base())
```

### Pattern 3: Meeting Notes

```python
async def meeting_notes():
    async with AgentMem() as memory:
        # Create meeting memory
        meeting = await memory.create_active_memory(
            external_id="team-meetings",
            title="Q1 Planning Meeting",
            template_content={
                "template": {"id": "meeting_v1", "name": "Meeting"},
                "sections": [
                    {"id": "attendees", "description": "Who attended"},
                    {"id": "agenda", "description": "Topics covered"},
                    {"id": "decisions", "description": "Key decisions"},
                    {"id": "action_items", "description": "Action items"}
                ]
            }
        )
        
        # Search for action items
        actions = await memory.search_memories(
            external_id="team-meetings",
            query="action items assigned to me",
            limit=5
        )

asyncio.run(meeting_notes())
```

---

## Troubleshooting

### Connection Issues

**Error**: `Failed to connect to database`

**Solution**:
1. Check Docker containers: `docker ps`
2. Verify `.env` file settings
3. Check service ports: `5432` (Postgres), `7687` (Neo4j), `11434` (Ollama)
4. Restart services: `docker-compose restart`

### Memory Not Found

**Error**: `Memory with id 999 not found`

**Solution**:
1. Verify `external_id` is correct
2. Check `memory_id` exists: `get_active_memories(external_id)`
3. Ensure memory wasn't deleted

### Search Returns No Results

**Solution**:
1. Try broader query
2. Create test memory first
3. Check embedding service is running
4. Verify query text is relevant

### MCP Tools Not Available in Claude

**Solution**:
1. Ensure MCP server is running
2. Check Claude configuration file syntax
3. Restart Claude Desktop
4. Check server logs for errors

---

## Next Steps

1. **Read Examples**: See `docs/guide/EXAMPLES.md` for 7 complete examples
2. **API Reference**: Check `docs/ref/API.md` for all methods
3. **MCP Tools**: Review `docs/ref/MCP_TOOLS.md` for tool documentation
4. **Architecture**: Understand the system in `docs/ARCHITECTURE.md`

---

## Key Concepts

### Memory Tiers
- **Active**: Working memory (not searchable)
- **Shortterm**: Recent memories (searchable, vector indexed)
- **Longterm**: Important memories (permanent, temporal tracking)

### Automatic Consolidation
- Memories consolidate automatically as you update them
- Based on `update_count` threshold
- Creates searchable shortterm memory

### Search Strategies
- **Fast**: Pointer-based (< 200ms) for quick lookups
- **Deep**: Synthesis-based (500ms-2s) for analysis

### External ID
- Universal identifier for your agent
- Can be UUID, string, or integer
- Used to organize all memories

---

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/yourusername/agent-reminiscence/issues)
- **Documentation**: [Full Docs](https://agent-reminiscence.readthedocs.io)
- **Examples**: See `docs/guide/EXAMPLES.md`

---

## Version

- Quick Start: v0.2.0
- Last Updated: November 16, 2025
