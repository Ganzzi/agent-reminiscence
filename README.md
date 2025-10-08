# Agent Mem

A standalone Python package for hierarchical memory management in AI agents. Provides a **stateless** interface to manage active, short-term, and long-term memories with vector search, graph relationships, and intelligent consolidation.

## 🚀 Quick Links

- **📚 Documentation Index**: See [docs/INDEX.md](docs/INDEX.md) for all documentation
- **🎯 Getting Started**: See [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) for setup and testing
- **⚡ Quick Start**: See [docs/QUICKSTART.md](docs/QUICKSTART.md) for 5-minute setup
- **🐛 Bug Fixes**: See [docs/BUG_FIXES.md](docs/BUG_FIXES.md) for resolved issues
- **🏗️ Architecture**: See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for system design
- **👨‍💻 Development**: See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for implementation guide
- **📊 Progress**: See [docs/IMPLEMENTATION_STATUS.md](docs/IMPLEMENTATION_STATUS.md) for current status
- **🤖 Phase 4**: See [docs/PHASE4_COMPLETE.md](docs/PHASE4_COMPLETE.md) for AI agent integration

## ✅ Current Status

**Overall Progress**: 89% complete (98/110 major tasks completed)

**Completed Phases**:
- ✅ **Phase 1**: Core infrastructure (PostgreSQL + Neo4j, embedding service)
- ✅ **Phase 2**: Memory tiers (Active, Shortterm, Longterm repositories)
- ✅ **Phase 3**: Memory Manager (consolidation, promotion, retrieval)
- ✅ **Phase 4**: AI Agents (ER Extractor, Memory Retrieve, Memory Update)
- ✅ **Phase 5**: Testing (27 test suites - needs rewrite to match implementation)
- ✅ **Phase 9**: Streamlit UI (100% - Web UI fully functional)
- ✅ **Phase 10**: MCP Server (100% - Claude Desktop integration ready)

**In Progress**:
- 📖 **Phase 6**: Examples and demonstrations (20% complete)
- 📚 **Phase 7**: Complete API documentation (50% complete)
- 🚀 **Phase 8**: Production deployment (not started)

**See [docs/IMPLEMENTATION_STATUS.md](docs/IMPLEMENTATION_STATUS.md) for detailed progress**

## 🎨 Streamlit Web UI

**NEW**: AgentMem now includes a web-based UI for managing memories without writing code!

### Features

- 📚 **Browse Templates** - Explore 60+ pre-built BMAD templates
- ➕ **Create Memories** - Create memories using templates or custom YAML
- 📋 **View Memories** - Browse all memories for an agent
- ✏️ **Update Sections** - Edit memory sections with live Markdown preview
- 🗑️ **Delete Memories** - Safely delete with type-to-confirm protection

### Starting the UI

```bash
cd streamlit_app
streamlit run app.py
```

The UI will open at `http://localhost:8501`

### User Guide

See [docs/STREAMLIT_UI_USER_GUIDE.md](docs/STREAMLIT_UI_USER_GUIDE.md) for complete usage instructions.

---

## 🔌 MCP Server for Claude Desktop

**NEW**: AgentMem now includes a Model Context Protocol (MCP) server for integration with Claude Desktop and other MCP clients!

### Features

- 🔍 **get_active_memories** - Retrieve all active memories for an agent
- 📝 **update_memory_sections** - Batch update multiple sections at once
- 🔎 **search_memories** - Search across memory tiers with AI-synthesized responses

### Quick Start

```powershell
# Run the MCP server (recommended: using uv)
uv run agent_mem_mcp\run.py

# Alternative: using Python directly
py agent_mem_mcp\run.py

# Add sample data for testing
uv run agent_mem_mcp\tests\add_sample_data.py

# Test with Python client
uv run agent_mem_mcp\tests\test_mcp_client.py
```

### Claude Desktop Integration

Add to your Claude Desktop config (`%APPDATA%\Claude\claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "agent-mem": {
      "command": "uv",
      "args": [
        "run",
        "path_to_agent_mem_mcp\\run.py"
      ],
      "env": {
        "POSTGRES_HOST": "localhost",
        "POSTGRES_PORT": "5432",
        "POSTGRES_USER": "postgres",
        "POSTGRES_PASSWORD": "postgres",
        "POSTGRES_DB": "agent_mem",
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USER": "neo4j",
        "NEO4J_PASSWORD": "neo4jpassword",
        "OLLAMA_BASE_URL": "http://localhost:11434"
      }
    }
  }
}
```

**Important**: 
- Use `uv` command for better dependency management
- Use absolute path (adjust to your installation directory)
- Ensure environment variables match your `.env` file
- Double backslashes required in JSON on Windows

### Documentation

- **[MCP Server README](agent_mem_mcp/README.md)** - Complete MCP server documentation
- **[Getting Started with MCP](docs/GETTING_STARTED_MCP.md)** - Quick start guide
- **[MCP Server Status](docs/MCP_SERVER_STATUS.md)** - Implementation status
- **[Implementation Details](docs/MCP_IMPLEMENTATION_COMPLETE.md)** - Technical details

---

## Key Features

- **Stateless Design**: One AgentMem instance can serve multiple agents/workers
- **Template-Driven Memory**: Active memories use YAML templates with structured sections
- **Section-Level Tracking**: Each section tracks its own update_count for consolidation
- **Three-Tier Memory**: Active (template+sections) → Shortterm (chunks+entities) → Longterm (temporal)
- **Simple API**: Just 4 methods to manage all your agent's memories
- **Vector Search**: Semantic search using embeddings via Ollama
- **Graph Relationships**: Entity and relationship tracking with Neo4j
- **Smart Consolidation**: Automatic section-level consolidation with conflict resolution
- **Generic ID Support**: Use any external ID (UUID, string, int) for your agents

## Quick Start

### Installation

**Linux/Mac:**
```bash
cd libs/agent_mem
pip install -e .
```

**Windows:**
```powershell
cd libs\agent_mem
py -m pip install -e .
```

Or add to your project's requirements:
```bash
echo "agent-mem @ file:///${PWD}/libs/agent_mem" >> requirements.txt
```

### Prerequisites

**Using Docker Compose (Recommended):**

The easiest way is to use the provided Docker Compose configuration:

```bash
# Start all services (PostgreSQL, Neo4j, Ollama)
docker compose up -d

# Check services are running
docker compose ps
```

**Manual Setup:**

1. **PostgreSQL** with extensions:
   - `pgvector` for vector storage
   - `pg_tokenizer` for text tokenization
   - `vchord_bm25` for BM25 search

2. **Neo4j** for graph storage

3. **Ollama** for embeddings:
   ```bash
   ollama pull nomic-embed-text
   ```

### Environment Setup

Create a `.env` file:

```env
# PostgreSQL
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password
POSTGRES_DB=agent_mem

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j

# Ollama
OLLAMA_BASE_URL=http://localhost:11434

# Model Configuration
EMBEDDING_MODEL=nomic-embed-text
VECTOR_DIMENSION=768

# Agent Models (Optional - defaults to Gemini)
MEMORY_UPDATE_AGENT_MODEL=google:gemini-2.0-flash
MEMORIZER_AGENT_MODEL=google:gemini-2.0-flash
MEMORY_RETRIEVE_AGENT_MODEL=google:gemini-2.0-flash
```

### Basic Usage

```python
from agent_mem import AgentMem
import asyncio

# Template defining memory structure
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
    # Initialize STATELESS memory manager (serves multiple agents)
    agent_mem = AgentMem()
    
    # Initialize database connections
    await agent_mem.initialize()
    
    try:
        # 1. Create an active memory with template
        memory = await agent_mem.create_active_memory(
            external_id="agent-123",  # Pass agent ID to method
            title="Build Dashboard",
            template_content=TASK_TEMPLATE,
            initial_sections={
                "current_task": {
                    "content": "Implement real-time analytics",
                    "update_count": 0
                },
                "progress": {
                    "content": "- Designed UI\n- Set up project",
                    "update_count": 0
                }
            },
            metadata={"priority": "high"}
        )
        print(f"Created memory: {memory.id}")
        
        # 2. Get all active memories for agent
        all_memories = await agent_mem.get_active_memories(
            external_id="agent-123"
        )
        print(f"Total memories: {len(all_memories)}")
        
        # 3. Update a specific section (auto-increments update_count)
        await agent_mem.update_active_memory_section(
            external_id="agent-123",
            memory_id=memory.id,
            section_id="progress",
            new_content="- Designed UI\n- Set up project\n- Implemented charts"
        )
        
        # 4. Retrieve memories (searches across all tiers)
        results = await agent_mem.retrieve_memories(
            external_id="agent-123",
            query="What is the current progress on the dashboard?",
            limit=10,
            synthesis=True  # Request AI summary of findings
        )
        
        # Access results
        print(f"Mode: {results.mode}")  # "pointer" or "synthesis"
        print(f"Found {len(results.chunks)} chunks, {len(results.entities)} entities")
        if results.synthesis:
            print(f"AI Summary: {results.synthesis}")
        
    finally:
        # Clean up connections
        await agent_mem.close()

if __name__ == "__main__":
    asyncio.run(main())
```

## Architecture

```
agent_mem/
├── __init__.py              # Public API exports
├── core.py                  # AgentMem main class (STATELESS)
├── config/
│   ├── __init__.py
│   ├── settings.py          # Configuration management
│   └── constants.py         # Constants and defaults
├── database/
│   ├── __init__.py
│   ├── postgres_manager.py  # PostgreSQL connection pool
│   ├── neo4j_manager.py     # Neo4j connection manager
│   ├── repositories/
│   │   ├── __init__.py
│   │   ├── active_memory.py      # Template + section CRUD
│   │   ├── shortterm_memory.py   # Shortterm memory CRUD
│   │   └── longterm_memory.py    # Longterm memory CRUD
│   └── models.py            # Pydantic models for data
├── services/
│   ├── __init__.py
│   ├── embedding.py         # Ollama embedding service
│   └── memory_manager.py    # Core memory operations (stateless)
├── agents/
│   ├── __init__.py
│   ├── memory_updater.py    # Memory Update Agent
│   ├── memorizer.py         # Memory Consolidation Agent
│   └── memory_retriever.py  # Memory Retrieve Agent
├── utils/
│   ├── __init__.py
│   └── helpers.py           # Utility functions
└── sql/
    ├── schema.sql           # PostgreSQL schema
    └── migrations/          # Future migrations
```

## API Reference

### AgentMem

The main class for memory management (STATELESS - serves multiple agents).

#### `__init__(config: Optional[Config] = None)`

Initialize stateless memory manager.

**Parameters:**
- `config`: Optional configuration object (uses environment variables by default)

#### `async initialize() -> None`

Initialize database connections and ensure schema exists.

#### `async create_active_memory(external_id: str | UUID | int, title: str, template_content: str, initial_sections: Optional[dict] = None, metadata: Optional[dict] = None) -> ActiveMemory`

Create a new active memory with template-driven structure.

**Parameters:**
- `external_id`: Unique identifier for the agent (UUID, string, or int)
- `title`: Memory title
- `template_content`: YAML template defining section structure
- `initial_sections`: Optional initial sections {section_id: {content: str, update_count: int}}
- `metadata`: Optional metadata dictionary

**Returns:** Created `ActiveMemory` object with sections

#### `async get_active_memories(external_id: str | UUID | int) -> List[ActiveMemory]`

Get all active memories for a specific agent.

**Parameters:**
- `external_id`: Unique identifier for the agent

**Returns:** List of `ActiveMemory` objects

#### `async update_active_memory_section(external_id: str | UUID | int, memory_id: int, section_id: str, new_content: str) -> ActiveMemory`

Update a specific section in an active memory.

Automatically increments the section's update_count and triggers consolidation when threshold is reached.

**Parameters:**
- `external_id`: Unique identifier for the agent
- `memory_id`: ID of memory to update
- `section_id`: Section ID to update (defined in template)
- `new_content`: New content for the section

**Returns:** Updated `ActiveMemory` object

#### `async retrieve_memories(external_id: str | UUID | int, query: str, limit: int = 10, synthesis: bool = False) -> RetrievalResult`

Search and retrieve relevant memories for a specific agent across all tiers.

**Parameters:**
- `external_id`: Unique identifier for the agent
- `query`: Natural language search query describing the context and what information is needed
- `limit`: Maximum results per tier (default: 10)
- `synthesis`: Force AI to generate a synthesized summary of results (default: False, AI decides)

**Returns:** `RetrievalResult` with matched chunks, entities, relationships, and optional AI synthesis

#### `async close() -> None`

Close all database connections.

## Memory Tiers

### Active Memory
- **Purpose**: Template-driven working memory with sections
- **Storage**: PostgreSQL with JSONB sections
- **Structure**: YAML template + sections {section_id: {content, update_count}}
- **Lifetime**: Short (hours to days)
- **Updates**: Frequent (per-section tracking)
- **Use Case**: Current task context, ongoing work, structured progress tracking
- **Consolidation**: Section-level triggers based on update_count threshold

### Shortterm Memory
- **Purpose**: Searchable recent knowledge
- **Storage**: PostgreSQL (vectors + BM25) + Neo4j (entities/relationships)
- **Lifetime**: Medium (days to weeks)
- **Updates**: Occasional (from active memory consolidation)
- **Use Case**: Recent implementations, conversations, research findings

### Longterm Memory
- **Purpose**: Consolidated knowledge base
- **Storage**: PostgreSQL (vectors + BM25) + Neo4j (entities/relationships)
- **Lifetime**: Long (persistent)
- **Updates**: Rare (promoted from shortterm with intelligent merging)
- **Use Case**: Core knowledge, patterns, historical decisions
- **Promotion**: Automatic with type merging, confidence recalculation, and state history tracking

## Advanced Usage

### Custom Configuration

```python
from agent_mem import AgentMem, Config

config = Config(
    postgres_host="custom-host",
    postgres_port=5433,
    embedding_model="custom-model",
    vector_dimension=1024,
)

agent_mem = AgentMem(external_id="agent-456", config=config)
```

### Automatic Consolidation

Memory consolidation happens automatically based on update thresholds. You can also trigger it manually:

```python
# This is done internally, but exposed for advanced use cases
from agent_mem.services import MemoryManager

manager = MemoryManager(external_id="agent-123")
await manager.consolidate_to_shortterm(active_memory_id=1)
await manager.promote_to_longterm(shortterm_memory_id=5)
```

### Intelligent Promotion to Longterm

When shortterm memories are promoted to longterm, the system performs intelligent merging:

**Entity Merging:**
- Matches entities by name (case-insensitive)
- Merges types from both memories (union operation)
- Recalculates confidence using weighted average (favors higher confidence)
- Tracks complete state history in metadata

**Relationship Merging:**
- Matches relationships by source and target entity names
- Merges types from both memories (union operation)
- Recalculates confidence and strength using weighted averages
- Tracks complete state history in metadata

**State History Example:**
```python
# Each entity/relationship tracks its evolution in metadata.state_history:
{
    "state_history": [
        {
            "timestamp": "2025-10-05T10:30:00Z",
            "source": "shortterm_promotion",
            "old_types": ["Person", "Developer"],
            "new_types": ["Person", "Developer", "Team Lead"],
            "old_confidence": 0.75,
            "new_confidence": 0.82
        }
    ]
}
```

This ensures no information is lost during promotion and provides a complete audit trail.

### Working with Entities and Relationships

```python
# Entities and relationships are automatically extracted during consolidation
# You can query them through the retrieval results

result = await agent_mem.retrieve_memories("authentication system")

# Access entities
for entity in result.entities:
    print(f"Entity: {entity.name} (type: {entity.type})")

# Access relationships
for rel in result.relationships:
    print(f"Relationship: {rel.from_entity} -> {rel.type} -> {rel.to_entity}")
```

## Development

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run with coverage
pytest --cov=agent_mem tests/
```

### Building Documentation

```bash
cd docs/
mkdocs serve  # View locally
mkdocs build  # Build static site
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Support

- **Documentation**: [Full Documentation](docs/)
- **Issues**: [GitHub Issues](https://github.com/Ganzzi/agent-mem/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Ganzzi/agent-mem/discussions)
