# Agent Mem Documentation Index

**Last Updated**: October 2, 2025  
**Package Version**: 0.1.0

This directory contains all documentation for the `agent_mem` package - a sophisticated three-tier memory system for AI agents.

---

## 📚 Quick Start

**New to agent_mem?** Start here:

1. **[README.md](../README.md)** - Package overview and quick start
2. **[QUICKSTART.md](QUICKSTART.md)** - 5-minute setup guide
3. **[GETTING_STARTED.md](GETTING_STARTED.md)** - Installation and first steps
4. **[ARCHITECTURE.md](ARCHITECTURE.md)** - System architecture and design

---

## 📖 Core Documentation

### User Guides

- **[GETTING_STARTED.md](GETTING_STARTED.md)**  
  Installation, setup, and basic usage examples
  - Prerequisites and dependencies
  - Database configuration (PostgreSQL + Neo4j)
  - Docker Compose setup (recommended)
  - Windows and Linux/Mac instructions
  - First memory operations
  - Common patterns

- **[QUICKSTART.md](QUICKSTART.md)**  
  5-minute quick start guide
  - Automated setup scripts
  - Service verification
  - Basic examples
  - Troubleshooting

- **[BUG_FIXES.md](BUG_FIXES.md)**  
  Fixed issues in v0.1.0
  - JSON serialization with psqlpy
  - Database row access patterns
  - Configuration attribute naming
  - NumPy dependency
  - Docker Compose updates

### Architecture & Design

- **[ARCHITECTURE.md](ARCHITECTURE.md)**  
  System architecture and design decisions
  - Three-tier memory system (Active → Shortterm → Longterm)
  - Database schemas (PostgreSQL + Neo4j)
  - Memory lifecycle and workflows
  - Entity/relationship graph structure

### Development

- **[DEVELOPMENT.md](DEVELOPMENT.md)**  
  Developer guide for contributing
  - Project structure
  - Coding standards
  - Testing guidelines
  - Adding new features

### Meta Documentation

- **[STRUCTURE.md](STRUCTURE.md)**  
  Documentation structure and organization
  - Complete directory tree
  - Navigation guide
  - File categories and purposes
  - Maintenance guidelines

- **[CONSOLIDATION_SUMMARY.md](CONSOLIDATION_SUMMARY.md)**  
  Documentation consolidation process
  - Before/after comparison
  - Changes made
  - Migration guide
  - Benefits achieved

---

## 🚀 Implementation Documentation

### Current Status

- **[IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)**  
  Complete implementation checklist and progress tracker
  - ✅ Phase 1: Core Infrastructure (100%)
  - ✅ Phase 2: Memory Tiers (100%)
  - ✅ Phase 3: Memory Manager (100%)
  - ✅ Phase 4: AI Agents (100%)
  - ⏸️ Phase 5: Testing (Not started)
  - Overall: **58% complete** (48/82 tasks)

### Phase 4: AI-Enhanced Memory

- **[PHASE4_COMPLETE.md](PHASE4_COMPLETE.md)**  
  Comprehensive Phase 4 documentation
  - ER Extractor Agent integration
  - Auto-resolution algorithm (similarity >= 0.85, overlap >= 0.7)
  - Entity/relationship workflows
  - Consolidation and promotion details
  - Helper functions (similarity, overlap, importance)
  - Configuration and thresholds

- **[PHASE4_INTEGRATION.md](PHASE4_INTEGRATION.md)**  
  Technical integration guide
  - ER Extractor Agent usage examples
  - Code snippets and workflows
  - Auto-resolution logic implementation
  - Memory Retrieve Agent integration
  - Error handling strategies
  - Testing recommendations

---

## 🏗️ Architecture Overview

### Three-Tier Memory System

```
┌─────────────────┐
│  Active Memory  │  ← Immediate context (conversation, task)
│  (PostgreSQL)   │     - Sections, metadata
└────────┬────────┘     - Auto-consolidates after N updates
         │
         ▼
┌─────────────────┐
│ Shortterm Memory│  ← Recent consolidated knowledge
│ (PG + Neo4j)    │     - Chunks with embeddings
└────────┬────────┘     - Entities and relationships
         │              - Hybrid search (vector + BM25)
         ▼
┌─────────────────┐
│ Longterm Memory │  ← Validated, important knowledge
│ (PG + Neo4j)    │     - Temporal tracking (start_date, end_date)
└─────────────────┘     - Confidence and importance scores
```

### AI Agents

1. **ER Extractor Agent** (`agents/predefined_agents/er_extractor_agent.py`)
   - Extracts entities and relationships from text
   - Model: google-gla:gemini-2.5-flash-lite
   - Output: ExtractionResult with entities/relationships
   - Used in: Consolidation workflow

2. **Memory Retrieve Agent** (`agent_mem/agents/memory_retriever.py`)
   - Intelligent search strategy determination
   - Result synthesis into natural language
   - Used in: Memory retrieval workflow

3. **Memory Update Agent** (`agent_mem/agents/memory_updater.py`)
   - Context-aware active memory updates
   - Not yet integrated (pending message workflows)

---

## 📦 Package Structure

```
agent_mem/
├── agent_mem/
│   ├── __init__.py           # Main AgentMem class
│   ├── config.py             # Configuration management
│   ├── agents/               # Pydantic AI agents
│   │   ├── memory_retriever.py
│   │   └── memory_updater.py
│   ├── database/
│   │   ├── neo4j_manager.py
│   │   ├── postgres_manager.py
│   │   └── repositories/
│   │       ├── active_memory.py
│   │       ├── shortterm_memory.py
│   │       └── longterm_memory.py
│   ├── models/               # Pydantic models
│   ├── services/
│   │   ├── memory_manager.py # Core orchestration
│   │   └── embedding.py      # Ollama embeddings
│   └── utils/                # Chunking, helpers
├── docs/                     # Documentation (you are here)
├── examples/                 # Usage examples
└── README.md                 # Package overview
```

---

## 🔧 Key Features

### Entity & Relationship Extraction

- **Auto-Resolution**: Automatically merges similar entities
  - Semantic similarity >= 0.85
  - Entity overlap >= 0.7
- **Conflict Detection**: Flags entities that don't meet thresholds
- **Confidence Tracking**: Updates confidence scores over time

### Memory Consolidation

- **Active → Shortterm**: Triggered after N updates
  - Chunks content with overlap
  - Generates embeddings (Ollama)
  - Extracts entities/relationships (ER Extractor Agent)
  - Stores in Neo4j graph

### Memory Promotion

- **Shortterm → Longterm**: Importance-based promotion
  - Entity confidence update: `0.7 * existing + 0.3 * new`
  - Importance scoring with type multipliers
  - Temporal tracking (start_date, end_date)
  - Relationship promotion with entity ID mapping

### Intelligent Retrieval

- **Dual-Agent System**: Strategy + Synthesis
  - Query intent analysis
  - Cross-tier search optimization
  - Vector/BM25 weight tuning
  - Natural language synthesis

---

## 🧪 Testing Status

**Current**: Phase 5 not started

**Planned Tests**:
- Unit tests: Config, models, managers, repositories (15 test files)
- Integration tests: End-to-end workflows (5 test files)
- Agent tests: ER Extractor, Memory Retrieve, Memory Update
- Total: **27 test suites** planned

See [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) for details.

---

## 📚 Historical Documentation

Archived documents from development phases:

### Archive Directory

Located in `docs/archive/`, these documents provide historical context:

- **[PHASE2_COMPLETE.md](archive/PHASE2_COMPLETE.md)** - Phase 2: Memory Tiers completion
- **[PHASE3_COMPLETE.md](archive/PHASE3_COMPLETE.md)** - Phase 3: Memory Manager completion
- **[ADJUSTMENT_PLAN_PHASE4.md](archive/ADJUSTMENT_PLAN_PHASE4.md)** - Phase 4 adjustment planning
- **[NEO4J_OPERATIONS_COMPLETE.md](archive/NEO4J_OPERATIONS_COMPLETE.md)** - Neo4j integration notes
- **[MEMORY_MANAGER_SUMMARY.md](archive/MEMORY_MANAGER_SUMMARY.md)** - Memory Manager summary
- **[REFACTORING_COMPLETE.md](archive/REFACTORING_COMPLETE.md)** - Refactoring session notes
- **[SESSION_SUMMARY.md](archive/SESSION_SUMMARY.md)** - Development session summary
- **[UPDATE_PLAN.md](archive/UPDATE_PLAN.md)** - Original update plan

**Note**: These are kept for reference but may contain outdated information. Refer to current documentation for accurate details.

---

## 🔗 Related Resources

### External Documentation

- **Main AI-Army Docs**: `../../docs/` (root project documentation)
  - Memory architecture: `../../docs/memory-architecture.md`
  - ER Extractor Agent: `../../docs/er-extractor-agent.md`
  - Neo4j entities: `../../docs/neo4j-entities-relationships.md`

### Code Examples

- **Examples Directory**: `../examples/`
  - Basic usage: `basic_usage.py`
  - Entity extraction: `entity_extraction_example.py`
  - Search examples: `search_examples.py`

---

## 🤝 Contributing

See [DEVELOPMENT.md](DEVELOPMENT.md) for:
- Setting up development environment
- Coding standards and best practices
- Testing guidelines
- Pull request process

---

## 📝 Documentation Maintenance

### Updating Documentation

When making changes to the package:

1. **Update relevant docs** in `docs/` directory
2. **Update [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)** if completing tasks
3. **Update this INDEX.md** if adding/removing documents
4. **Archive old versions** to `docs/archive/` if creating revisions

### Document Naming Conventions

- **User guides**: Descriptive names (e.g., `GETTING_STARTED.md`)
- **Technical docs**: Component names (e.g., `ARCHITECTURE.md`)
- **Phase docs**: Phase number prefix (e.g., `PHASE4_COMPLETE.md`)
- **Archived docs**: Keep original names, add README.md to archive/ if needed

---

## 📞 Support

For questions or issues:

1. Check this documentation index
2. Review [GETTING_STARTED.md](GETTING_STARTED.md) for common issues
3. Check [ARCHITECTURE.md](ARCHITECTURE.md) for design questions
4. Review [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) for feature status
5. See main project docs at `../../docs/`

---

**Package**: agent_mem v0.1.0  
**License**: MIT  
**Documentation**: https://github.com/Ganzzi/ai-army/tree/main/libs/agent_mem
