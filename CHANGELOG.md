# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2025-10-23

### Fixed
- **Package Naming Consistency**: Corrected package folder structure from `agent_mem` to `agent_reminiscence`
  - Renamed main package directory to match PyPI package name
  - Updated all imports throughout codebase (core, tests, examples, docs, scripts)
  - Updated module references in unit test patches and mocks
  - Updated configuration files (pyproject.toml, docker-compose.yml)
  - Updated all documentation and code examples
- **Package Configuration**: 
  - Updated `pyproject.toml` to reference correct package name in setuptools discovery
  - Updated coverage reports to use correct module name
  - Removed old build artifacts (agent_mem.egg-info, agentmem.egg-info, agent_memory.egg-info)

### Note
Users upgrading from 0.1.0 must update their imports:
- Old: `from agent_mem import AgentMem`
- New: `from agent_reminiscence import AgentMem`

## [0.1.0] - 2025-10-23

### Added
- **Streamlit Web UI**: Complete web interface for memory management (Phase 9 - Oct 3, 2025)
  - 5 fully functional pages (Browse, Create, View, Update, Delete)
  - Template browser with 60+ pre-built BMAD templates
  - Dual-mode memory creation (template or custom YAML)
  - Live Markdown editor with preview
  - Type-to-confirm deletion with safety checks
  - Responsive design with custom theme
  - Comprehensive user guide and documentation
- **MCP Server Integration**: Model Context Protocol server for Claude Desktop and MCP clients
  - Full support for active memory creation and management
  - Batch section updates with upsert capability (replace/insert actions)
  - Cross-tier memory search with AI synthesis
  - Entity and relationship extraction and tracking
  - Comprehensive API documentation
- Initial release of Agent Mem
- Stateless AgentMem interface for multi-agent memory management
- Three-tier memory system (Active, Shortterm, Longterm)
- Template-driven active memory with YAML section definitions
- Section-level update tracking with automatic consolidation
- PostgreSQL integration with pgvector for vector storage
- Neo4j integration for entity and relationship graphs
- Ollama integration for embeddings
- Pydantic AI agents for intelligent operations:
  - Memory Update Agent
  - Memory Consolidation Agent (Memorizer)
  - Memory Retrieval Agent
- Hybrid search combining vector similarity and BM25
- Docker Compose setup for easy deployment
- Comprehensive test suite with 229+ tests
- Documentation with MkDocs
- Examples demonstrating core functionality

### Features
- **Stateless Design**: Single instance serves multiple agents
- **Generic ID Support**: Use UUID, string, or int for agent identifiers
- **Simple API**: 4 core methods for all memory operations
- **Batch Updates**: Upsert multiple sections with replace/insert actions
- **Automatic Consolidation**: Section-level triggers based on update_count
- **Smart Retrieval**: AI-powered memory search with optional synthesis
- **Entity Extraction**: Automatic entity and relationship extraction
- **Production Ready**: Docker setup, comprehensive tests, full documentation

### Release Notes
Initial alpha release of Agent Mem. Suitable for testing and evaluation.

**Requirements:**
- Python 3.10+
- PostgreSQL 14+ with pgvector, pg_tokenizer, vchord_bm25
- Neo4j 5+
- Ollama with nomic-embed-text model

**Known Limitations:**
- Alpha software, APIs may change
- Performance optimization ongoing
- Limited production deployment testing

**Installation:**
```bash
pip install agent-reminiscence
```

---

## Version Guidelines

### Major Version (x.0.0)
- Breaking API changes
- Major architectural changes
- Incompatible database schema changes

### Minor Version (0.x.0)
- New features (backward compatible)
- New API methods
- Performance improvements
- Database migrations (backward compatible)

### Patch Version (0.0.x)
- Bug fixes
- Documentation updates
- Minor improvements
- Dependency updates

---

## Contribution

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to contribute to this project.

## Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/agent-reminiscence/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/agent-reminiscence/discussions)

