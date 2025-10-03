# Documentation Archive

This directory contains historical documentation from the development of `agent_mem`. These documents provide context about implementation decisions, refactoring sessions, and phase completions.

**⚠️ Note**: Information in these documents may be outdated. Please refer to the current documentation in the parent `docs/` directory for accurate, up-to-date information.

---

## 📁 Archived Documents

### Phase Completion Notes

- **[PHASE2_COMPLETE.md](PHASE2_COMPLETE.md)**  
  Completion notes for Phase 2: Memory Tiers implementation
  - Shortterm memory repository
  - Longterm memory repository
  - Vector and BM25 search implementations
  - Entity/relationship management in Neo4j

- **[PHASE3_COMPLETE.md](PHASE3_COMPLETE.md)**  
  Completion notes for Phase 3: Memory Manager implementation
  - Core consolidation workflow
  - Promotion workflow
  - Memory lifecycle management
  - Initial agent integration plans

### Planning Documents

- **[ADJUSTMENT_PLAN_PHASE4.md](ADJUSTMENT_PLAN_PHASE4.md)**  
  Comprehensive adjustment plan for Phase 4
  - Decision to use existing ER Extractor Agent
  - Auto-resolution algorithm specification
  - Entity/relationship handling in consolidation
  - Promotion enhancements with confidence updates
  - **Superseded by**: [../PHASE4_COMPLETE.md](../PHASE4_COMPLETE.md)

- **[UPDATE_PLAN.md](UPDATE_PLAN.md)**  
  Original update plan for stateless refactoring
  - Template-driven active memory design
  - Section-level tracking
  - Stateless API design decisions

### Implementation Notes

- **[NEO4J_OPERATIONS_COMPLETE.md](NEO4J_OPERATIONS_COMPLETE.md)**  
  Neo4j integration completion notes
  - Entity and relationship operations
  - Cypher query patterns
  - Graph traversal examples

- **[MEMORY_MANAGER_SUMMARY.md](MEMORY_MANAGER_SUMMARY.md)**  
  Summary of Memory Manager implementation
  - Component overview
  - Workflow descriptions
  - Integration points

### Session Notes

- **[REFACTORING_COMPLETE.md](REFACTORING_COMPLETE.md)**  
  Notes from major refactoring session
  - Stateless design implementation
  - Template system integration
  - Breaking changes and migration guide

- **[SESSION_SUMMARY.md](SESSION_SUMMARY.md)**  
  Development session summary
  - Key decisions made
  - Problems solved
  - Next steps planned

---

## 🔍 Why Archive?

These documents are archived rather than deleted because they:

1. **Provide Historical Context**: Show how design decisions evolved
2. **Document Refactoring**: Explain why certain changes were made
3. **Reference Implementation**: Contain detailed technical notes
4. **Audit Trail**: Track progress and decision-making process

However, they may contain:
- ❌ Outdated implementation details
- ❌ Deprecated API examples
- ❌ Superseded architectural decisions
- ❌ Old file/function names

---

## 📖 Current Documentation

For up-to-date information, see:

- **[../INDEX.md](../INDEX.md)** - Documentation index
- **[../GETTING_STARTED.md](../GETTING_STARTED.md)** - Setup guide
- **[../ARCHITECTURE.md](../ARCHITECTURE.md)** - Current architecture
- **[../PHASE4_COMPLETE.md](../PHASE4_COMPLETE.md)** - Latest phase documentation
- **[../IMPLEMENTATION_STATUS.md](../IMPLEMENTATION_STATUS.md)** - Current progress

---

## 📝 Archive Maintenance

### When to Archive

Archive documents when:
- Creating revised versions of existing docs
- Completing major refactoring that invalidates old docs
- Phase completion notes become historical

### What to Keep Current

Keep in parent `docs/` directory:
- User guides (GETTING_STARTED, ARCHITECTURE, DEVELOPMENT)
- Current implementation status
- Latest phase documentation
- API references

### Naming Convention

Archived documents keep their original names to maintain historical context and make it easy to track versions.

---

**Last Updated**: October 2, 2025  
**Archive Created**: During Phase 4 documentation consolidation
