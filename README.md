# Agent Mem

A Python package for hierarchical memory management in AI agents. Provides a **stateless** interface to manage active, short-term, and long-term memories with vector search, graph relationships, and intelligent consolidation.

Whether you're building a multi-agent system, an AI assistant with persistent memory, or a complex knowledge management system, Agent Mem provides a robust, scalable foundation.

## ✨ Key Features

- **🔄 Stateless Design** - One instance serves multiple agents/workers with any ID type (UUID, string, int)
- **📚 Three-Tier Memory** - Active (template-driven) → Shortterm (searchable) → Longterm (consolidated)
- **🔍 Dual Search Modes** - Fast search (< 200ms) or deep search with AI synthesis (500ms-2s)
- **📊 Vector Search** - Semantic search powered by Ollama embeddings
- **🌐 Graph Relationships** - Entity and relationship tracking via Neo4j
- **🤖 AI Integration** - MCP server for Claude Desktop and other LLM clients
- **🎯 Template-Driven** - Structured active memory with YAML templates and sections
- **⚡ Smart Consolidation** - Automatic promotion between tiers with intelligent merging
- **🎨 Web UI** - Streamlit interface for memory management without code
- **📝 Comprehensive** - 350+ tests, full documentation, production-ready

## Installation

Install from PyPI:

```bash
pip install agent-reminiscence
```

Or with optional dependencies:

```bash
# With MCP server support
pip install "agent-reminiscence[mcp]"

# With development tools
pip install "agent-reminiscence[dev]"

# With documentation tools
pip install "agent-reminiscence[docs]"
```

## System Requirements

- **Python**: 3.10+
- **PostgreSQL**: 14+ with extensions (pgvector, pg_tokenizer, vchord_bm25)
- **Neo4j**: 5+
- **Ollama**: Latest with nomic-embed-text model

### Quick Setup with Docker

The easiest way to get started:

```bash
# Start all services (PostgreSQL, Neo4j, Ollama)
docker compose up -d

# Verify services are running
docker compose ps
```

This starts:
- **PostgreSQL** on localhost:5432 (vector storage + search)
- **Neo4j** on localhost:7687 (graph relationships)
- **Ollama** on localhost:11434 (embeddings)

## Configuration

Agent Mem supports three configuration patterns:

### Pattern 1: Environment Variables (Recommended)

```bash
# Create .env file
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password
POSTGRES_DB=agent_reminiscence

NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

OLLAMA_BASE_URL=http://localhost:11434
EMBEDDING_MODEL=nomic-embed-text
```

Then in Python:

```python
from agent_reminiscence import AgentMem

agent_mem = AgentMem()  # Uses .env file automatically
```

### Pattern 2: Direct Configuration

```python
from agent_reminiscence import AgentMem, Config

config = Config(
    postgres_host="localhost",
    postgres_port=5432,
    postgres_user="postgres",
    postgres_password="password",
    postgres_db="agent_reminiscence",
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="password",
    ollama_base_url="http://localhost:11434",
)

agent_mem = AgentMem(config=config)
```

### Pattern 3: Mixed (Config + Environment)

```python
from agent_reminiscence import AgentMem, Config

config = Config(postgres_host="custom-host")
agent_mem = AgentMem(config=config)  # Remaining values from env
```

## Quick Start

### 1. Basic Usage

```python
import asyncio
from agent_reminiscence import AgentMem

async def main():
    agent_mem = AgentMem()
    await agent_mem.initialize()
    
    try:
        # Create a memory
        memory = await agent_mem.create_active_memory(
            external_id="agent-123",
            title="Project Context",
            template_content={
                "template": {
                    "id": "project_v1",
                    "name": "Project Memory"
                },
                "sections": [
                    {"id": "goals", "description": "Project goals"},
                    {"id": "progress", "description": "Current progress"},
                    {"id": "blockers", "description": "Current blockers"}
                ]
            },
            initial_sections={
                "goals": {"content": "Launch MVP by Q1 2026"},
                "progress": {"content": "Architecture designed (40% complete)"},
            }
        )
        
        print(f"Created memory: {memory.id}")
        
        # Update a section
        await agent_mem.update_active_memory_section(
            external_id="agent-123",
            memory_id=memory.id,
            section_id="progress",
            new_content="Architecture designed, backend in progress (60% complete)"
        )
        
        # Search memories
        results = await agent_mem.search_memories(
            external_id="agent-123",
            query="What is the project progress?",
            limit=5
        )
        
        print(f"Found {len(results.shortterm_chunks)} results")
        
    finally:
        await agent_mem.close()

if __name__ == "__main__":
    asyncio.run(main())
```

### 2. Using Templates

Create structured memories with YAML templates:

```python
# Define a template (or use one of 60+ pre-built templates)
TASK_TEMPLATE = """
template:
  id: "task_v1"
  name: "Task Memory"
sections:
  - id: "title"
    description: "Task title"
  - id: "description"
    description: "Detailed description"
  - id: "status"
    description: "Current status"
  - id: "notes"
    description: "Additional notes"
"""

memory = await agent_mem.create_active_memory(
    external_id="agent-123",
    title="Implement Auth System",
    template_content=TASK_TEMPLATE,
    initial_sections={
        "title": {"content": "JWT Authentication"},
        "status": {"content": "In Progress"},
    }
)
```

### 3. Searching Memories

Agent Mem provides two search modes:

**Fast Search** (< 200ms) - Quick fact lookups:

```python
results = await agent_mem.search_memories(
    external_id="agent-123",
    query="authentication implementation",
    limit=10
)

# Access results
for chunk in results.shortterm_chunks:
    print(f"Chunk: {chunk.content}")
    print(f"Relevance: {chunk.relevance_score:.2%}")
```

**Deep Search** (500ms-2s) - Comprehensive analysis with AI synthesis:

```python
results = await agent_mem.deep_search_memories(
    external_id="agent-123",
    query="How does authentication relate to the API design?",
    limit=10
)

# Results include AI-generated synthesis
if results.synthesis:
    print(f"Analysis: {results.synthesis}")

# Plus entities and relationships
for entity in results.shortterm_triplets:
    print(f"Entity: {entity.subject} - {entity.predicate} - {entity.object}")
```

## Architecture

Agent Mem is organized into clean layers with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 1: Public API (core.py)                               │
│ - AgentMem class: 6 main methods                            │
│ - Entry point for all operations                            │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 2: Service Layer (memory_manager.py)                  │
│ - Memory operations (stateless, multi-agent)                │
│ - Search & retrieval logic                                  │
│ - Consolidation & promotion                                 │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 3: Agent Layer (agents/)                              │
│ - Memory Update Agent (LLM-powered updates)                 │
│ - Memorizer Agent (consolidation & promotion)               │
│ - Memory Retriever Agent (search & synthesis)               │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 4: Repository Layer (repositories/)                   │
│ - Active Memory Repository                                  │
│ - Shortterm Memory Repository                               │
│ - Longterm Memory Repository                                │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────┬──────────────────────────────────┐
│ PostgreSQL               │ Neo4j                            │
│ - Embeddings/vectors     │ - Entity graph                   │
│ - Text search (BM25)     │ - Relationships                  │
│ - Active memory sections │ - Entity types & confidence      │
└──────────────────────────┴──────────────────────────────────┘
```

## Memory Tiers

### Active Memory
**Working memory with template-driven structure**
- Storage: PostgreSQL (JSONB sections)
- Lifetime: Short (hours to days)
- Updates: Frequent (section-level tracking)
- Use Case: Current tasks, ongoing work, structured context
- Structure: `{template_id, title, metadata, sections[]}`

Example:
```python
{
    "id": 1,
    "title": "Project Context",
    "sections": {
        "goals": {"content": "...", "update_count": 2},
        "progress": {"content": "...", "update_count": 5}
    }
}
```

### Shortterm Memory
**Searchable recent knowledge**
- Storage: PostgreSQL (vectors, BM25) + Neo4j (entities)
- Lifetime: Medium (days to weeks)
- Updates: Occasional (from active consolidation)
- Use Case: Recent findings, conversations, implementations
- Structure: `{chunks[], entities[], relationships[]}`

### Longterm Memory
**Consolidated knowledge base**
- Storage: PostgreSQL (vectors, BM25) + Neo4j (entities)
- Lifetime: Long (persistent)
- Updates: Rare (promoted from shortterm)
- Use Case: Core knowledge, patterns, decisions
- Promotion: Intelligent merging with type union + confidence weighting

## Core API Methods

### Memory Management

**`async create_active_memory(external_id, title, template_content, initial_sections=None, metadata=None) -> ActiveMemory`**

Create a new active memory with template-driven structure.

```python
memory = await agent_mem.create_active_memory(
    external_id="agent-123",
    title="My Task",
    template_content={"template": {...}, "sections": [...]},
    initial_sections={"section_id": {"content": "..."}},
    metadata={"priority": "high"}
)
```

**`async get_active_memories(external_id) -> List[ActiveMemory]`**

Retrieve all active memories for an agent.

```python
memories = await agent_mem.get_active_memories(external_id="agent-123")
```

**`async update_active_memory_section(external_id, memory_id, section_id, new_content) -> ActiveMemory`**

Update a single section (triggers consolidation if needed).

```python
updated = await agent_mem.update_active_memory_section(
    external_id="agent-123",
    memory_id=1,
    section_id="progress",
    new_content="Updated content"
)
```

**`async update_active_memory_sections(external_id, memory_id, sections) -> ActiveMemory`**

Batch upsert multiple sections with replace/insert actions.

```python
updated = await agent_mem.update_active_memory_sections(
    external_id="agent-123",
    memory_id=1,
    sections=[
        {"section_id": "progress", "new_content": "...", "action": "replace"},
        {"section_id": "notes", "new_content": "...", "action": "insert"}
    ]
)
```

### Search & Retrieval

**`async search_memories(external_id, query, limit=10) -> RetrievalResultV2`**

Fast search across all memory tiers (< 200ms, no synthesis).

```python
results = await agent_mem.search_memories(
    external_id="agent-123",
    query="authentication implementation",
    limit=10
)

# Access results
print(f"Shortterm chunks: {len(results.shortterm_chunks)}")
print(f"Longterm chunks: {len(results.longterm_chunks)}")
for entity in results.shortterm_triplets:
    print(f"Entity: {entity.subject} -> {entity.predicate} -> {entity.object}")
```

**`async deep_search_memories(external_id, query, limit=10) -> RetrievalResultV2`**

Comprehensive search with AI-powered synthesis (500ms-2s).

```python
results = await agent_mem.deep_search_memories(
    external_id="agent-123",
    query="How does authentication relate to API security?",
    limit=10
)

# Results include AI synthesis
if results.synthesis:
    print(f"Analysis: {results.synthesis}")
```

### Management

**`async initialize() -> None`**

Initialize database connections and ensure schema exists.

```python
agent_mem = AgentMem()
await agent_mem.initialize()
```

**`async close() -> None`**

Close all database connections.

```python
await agent_mem.close()
```

## Advanced Usage

### Custom Templates

Define structured memory templates in YAML or JSON:

```yaml
template:
  id: "research_v1"
  name: "Research Notes"
sections:
  - id: "topic"
    description: "Research topic"
  - id: "findings"
    description: "Key findings"
  - id: "references"
    description: "Source references"
  - id: "next_steps"
    description: "Next research steps"
```

### Batch Updates

Update multiple sections efficiently:

```python
await agent_mem.update_active_memory_sections(
    external_id="agent-123",
    memory_id=1,
    sections=[
        {
            "section_id": "progress",
            "old_content": "60%",  # Find this pattern
            "new_content": "70%",  # Replace with this
            "action": "replace"
        },
        {
            "section_id": "notes",
            "old_content": "## Updates",  # Find pattern
            "new_content": "\n- New item",  # Insert after pattern
            "action": "insert"
        }
    ]
)
```

### Entity Relationships

Entities and relationships are automatically extracted. Access them from search results:

```python
results = await agent_mem.search_memories(
    external_id="agent-123",
    query="architecture decisions",
    limit=10
)

# Access extracted entities
for entity in results.shortterm_triplets:
    print(f"{entity.subject} --{entity.predicate}--> {entity.object}")

# Relationships show connections between concepts
```

### Stateless Multi-Agent

One AgentMem instance serves multiple agents:

```python
agent_mem = AgentMem()
await agent_mem.initialize()

# Agent 1's memory
await agent_mem.create_active_memory(external_id="agent-1", ...)

# Agent 2's memory  
await agent_mem.create_active_memory(external_id="agent-2", ...)

# Agent 3's memory
await agent_mem.create_active_memory(external_id="agent-3", ...)

# No need for separate instances - one serves all
```

## Web UI

Agent Mem includes a **Streamlit web interface** for managing memories without code:

### Features
- Browse 60+ pre-built templates
- Create/view/update/delete memories
- Live Markdown editor
- Type-to-confirm deletion

### Running

```bash
cd streamlit_app
streamlit run app.py
```

Access at `http://localhost:8501`

## Claude Desktop Integration (MCP)

Integrate Agent Mem with Claude Desktop using the Model Context Protocol:

### Setup

Add to Claude Desktop config (`%APPDATA%\Claude\claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "agent-mem": {
      "command": "python",
      "args": ["-m", "agent_reminiscence_mcp.run"],
      "env": {
        "POSTGRES_HOST": "localhost",
        "POSTGRES_PORT": "5432",
        "POSTGRES_USER": "postgres",
        "POSTGRES_PASSWORD": "your_password",
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USER": "neo4j",
        "NEO4J_PASSWORD": "your_password",
        "OLLAMA_BASE_URL": "http://localhost:11434"
      }
    }
  }
}
```

### Claude Functions

Six tools available in Claude:

1. **get_active_memories** - List all memories for an agent
2. **create_active_memory** - Create new memory with template
3. **update_memory_sections** - Batch update sections
4. **delete_active_memory** - Delete a memory
5. **search_memories** - Fast search (< 200ms)
6. **deep_search_memories** - Comprehensive search with synthesis

### Example in Claude

```
You: "Help me organize my project memory"

Claude: [Uses create_active_memory to create new memory]

You: "What's my current progress?"

Claude: [Uses deep_search_memories to analyze and synthesize your memories]
→ "Based on your memories, you've completed the architecture 
and are now working on the backend implementation. You have 
two blockers related to authentication that need attention."
```

## Documentation

Full documentation available at [docs/](docs/):

- **[Quick Start](docs/QUICKSTART.md)** - Get running in 5 minutes
- **[API Reference](docs/API.md)** - Complete method documentation
- **[Architecture](docs/ARCHITECTURE.md)** - System design deep dive
- **[Examples](docs/guide/EXAMPLES.md)** - 7 complete working examples
- **[MCP Integration](docs/guide/MCP_INTEGRATION.md)** - Claude Desktop setup
- **[Changelog](CHANGELOG.md)** - Version history

## Development

### Running Tests

```bash
# Install dev dependencies
pip install "agent-reminiscence[dev]"

# Run all tests
pytest tests/

# Run with coverage
pytest --cov=agent_reminiscence tests/

# Run specific test file
pytest tests/test_core.py
```

### Project Structure

```
agent_reminiscence/
├── core.py                    # Main AgentMem class
├── config/settings.py         # Configuration
├── database/
│   ├── repositories/          # Data access layer
│   └── models.py              # Pydantic models
├── services/
│   ├── memory_manager.py      # Memory operations
│   └── embedding.py           # Embeddings
├── agents/                    # LLM agents
└── utils/                     # Helpers
```

### Building Documentation

```bash
# Install docs dependencies
pip install "agent-reminiscence[docs]"

# Serve locally
mkdocs serve

# Build static site
mkdocs build
```

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Code style guidelines
- How to submit issues
- Pull request process
- Development setup

## Support

- **📖 Documentation**: [docs/](docs/)
- **🐛 Issues**: [GitHub Issues](https://github.com/Ganzzi/agent-reminiscence/issues)
- **💬 Discussions**: [GitHub Discussions](https://github.com/Ganzzi/agent-reminiscence/discussions)

## License

MIT License - see [LICENSE](LICENSE) for details.

---

**Made with ❤️ for AI agents that remember.**

