# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Streamlit Web UI**: Complete web interface for memory management (Phase 9 - Oct 3, 2025)
  - 5 fully functional pages (Browse, Create, View, Update, Delete)
  - Template browser with 60+ pre-built BMAD templates
  - Dual-mode memory creation (template or custom YAML)
  - Live Markdown editor with preview
  - Type-to-confirm deletion with safety checks
  - Responsive design with custom theme
  - Comprehensive user guide and documentation
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
- Comprehensive test suite with pytest
- Documentation with MkDocs
- Examples demonstrating core functionality

### Features
- **Stateless Design**: Single instance serves multiple agents
- **Generic ID Support**: Use UUID, string, or int for agent identifiers
- **Simple API**: 4 core methods for all memory operations
- **Automatic Consolidation**: Section-level triggers based on update_count
- **Smart Retrieval**: AI-powered memory search and synthesis
- **Entity Extraction**: Automatic entity and relationship extraction
- **Production Ready**: Docker setup, comprehensive tests, full documentation

## [0.1.0] - 2025-01-XX (Planned)

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
pip install agent-mem
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

- **Issues**: [GitHub Issues](https://github.com/yourusername/agent-mem/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/agent-mem/discussions)
