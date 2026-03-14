---
applyTo: '**'
---

# Agent Mem Memory Management Instructions

## External ID
**ALWAYS use:** `agent_mem_dev` for all memory operations in this project.

## Memory Structure Overview

Agent Mem project maintains 8 active memories to support development:

1. **Project Overview** - Purpose, core features, tech stack, version, release timeline
2. **Architecture Design** - Layered architecture, component responsibilities, design patterns, data flow
3. **API Design** - Public API methods, MCP tools, Streamlit UI structure
4. **Configuration** - Environment variables, dependencies (uv), setup instructions, Windows-specific notes
5. **Testing Strategy** - Unit/integration/e2e tests, test organization, pytest configuration, status
6. **Development Status** - Current phase, completed tasks, next steps, blockers
7. **Library References** - Usage patterns for psqlpy, pydantic, pydantic-ai, neo4j, pytest-asyncio, numpy
8. **Issues and Bugs** - Detailed bug reports with reproduction steps, root causes, and workarounds

## When to Access Memories

### 1. **At Session Start**
- Get all active memories to understand current project state
- Check "Development Status" memory for current phase and next steps
- Review "Issues and Bugs" memory for known problems

```
Tool: mcp_agent-mem_get_active_memories
Parameters: external_id="agent_mem_dev"
```

### 2. **Before Implementing Features**
- Search memories for relevant context
- Check "Architecture Design" for patterns and component responsibilities
- Review "API Design" if working with public API or MCP tools
- Check "Configuration" for environment setup and dependencies

```
Tool: mcp_agent-mem_search_memories
Parameters:
  external_id="agent_mem_dev"
  query="Implementing deep search feature with AI synthesis, need API method signatures and pydantic-ai patterns"
  limit=10
```

### 3. **When Needing Library Documentation**
- Check "Library References" memory first for known patterns
- Search for specific library usage (psqlpy, pydantic, pydantic-ai, neo4j, pytest-asyncio)
- Only check local files in `docs/` as fallback
- Update "Library References" with new patterns discovered

**Key libraries to reference:**
- psqlpy - Async PostgreSQL driver with connection pooling
- pydantic - Data validation and settings management
- pydantic-ai - LLM agent framework with tool support
- neo4j - Graph database driver for entity relationships
- pytest-asyncio - Async test support

### 4. **When Encountering Bugs**
- First check "Issues and Bugs" memory to see if it's known
- If new bug, add detailed section to memory with reproduction steps
- Include error messages, affected components, and workarounds

## How to Update Memories

### Update Development Status

**When starting a new phase:**
```
Tool: mcp_agent-mem_update_memory_sections
Parameters:
  external_id="agent_mem_dev"
  memory_id=6
  sections=[
    {
      section_id="current_phase",
      action="replace",
      old_content="**Phase: Pre-Implementation**...",
      new_content="**Phase: Feature Development**\n\nImplementing deep search functionality with AI synthesis.\n\n**Started:** November 24, 2025"
    }
  ]
```

**When completing tasks:**
```
sections=[
  {
    section_id="completed_tasks",
    action="insert",
    old_content="- ✓ Initialized project structure\n\n**Next milestone:**",
    new_content="- ✓ Implemented AgentMem.deep_search_memories()\n- ✓ Added AI synthesis with pydantic-ai\n- ✓ Wrote comprehensive tests for search\n\n**Next milestone:**"
  }
]
```

**When encountering blockers:**
```
sections=[
  {
    section_id="blockers",
    action="replace",
    old_content="No current blockers.",
    new_content="**Blocker #1:** Neo4j entity extraction performance with large graphs\n- Impact: Memory retrieval slows down with 100k+ entities\n- Investigating: Index optimization and query patterns\n- Workaround: Implement pagination for entity results"
  }
]
```

### Add Bug/Issue

**Create a new section for each bug in Issues and Bugs memory:**
```
Tool: mcp_agent-mem_update_memory_sections
Parameters:
  external_id="agent_mem_dev"
  memory_id=8
  sections=[
    {
      section_id="issue_001_postgres_connection_timeout",
      action="replace",
      new_content="**Issue #001: PostgreSQL Connection Timeout on Large Batch Updates**\n**Status:** Open\n**Severity:** High\n**Date Found:** 2025-11-24\n**Component:** agent_reminiscence/database/repositories/shortterm_memory.py\n\n**Description:**\nPostgreSQL connection pools exhaust during batch update operations, causing timeouts.\n\n**Steps to Reproduce:**\n1. Create 10k+ memory chunks in shortterm tier\n2. Run batch consolidation with default connection pool size (10)\n3. Observe connection timeout after ~5 minutes\n\n**Expected Behavior:**\nBatch operations should complete without timeout errors.\n\n**Actual Behavior:**\npsqlpy raises connection pool exhaustion error.\n\n**Environment:**\n- Python 3.11+\n- psqlpy 0.11.0\n- PostgreSQL 14+\n- Platform: Windows/Linux\n\n**Error Message:**\n```\npsqlpy.exceptions.PoolExhausted: Connection pool exhausted, no available connections\n```\n\n**Root Cause Analysis:**\nDefault connection pool size too small for concurrent batch operations. Each batch update holds connection longer than expected due to vector embedding generation.\n\n**Suggested Solution:**\nIncrease pool size or implement connection pooling strategy with queue-based task distribution.\n\n**Related Files:**\n- agent_reminiscence/config/settings.py:POSTGRES_POOL_SIZE\n- agent_reminiscence/services/memory_manager.py:_consolidate_batch()\n- agent_reminiscence/database/postgres_manager.py"
    }
  ]
```

### Update Library References

**When discovering new patterns or gotchas:**
```
sections=[
  {
    section_id="pydantic_ai_usage",
    action="insert",
    old_content="**Base Agent Pattern:**",
    new_content="\n**Tool Registration Pattern:**\n- Use @agent.tool decorator for simple tool functions\n- Tools automatically handle type conversion from LLM output\n- Return RunUsage from tools to track token consumption\n- Use structured Pydantic models for complex tool inputs/outputs\n\n**Base Agent Pattern:**"
  }
]
```

### Update Architecture

**When design decisions change:**
```
sections=[
  {
    section_id="design_patterns",
    action="insert",
    old_content="**Service Layer Responsibilities:**",
    new_content="\n**Deep Search Design:**\n- MemoryManager orchestrates search operations\n- RetrievalAgent performs vector + BM25 hybrid search\n- SynthesisAgent generates AI summaries from results\n- EntityExtractorAgent extracts and links entities from findings\n- Results consolidated into RetrievalResult with confidence scores\n\n**Service Layer Responsibilities:**"
  }
]
```

## Search Best Practices

### Effective Search Queries

**Good queries are specific and contextual:**

✅ **Good:**
```
"Working on MCP tool integration, need to know tool schema structure and how to register custom tools with agent"
"Implementing consolidation, need to understand promotion workflow and entity merging strategy"
"Writing repository for shortterm memory, need psqlpy async patterns and vector search implementation"
```

❌ **Bad:**
```
"MCP tools"  # Too vague
"how to search"  # Not enough context
"memory"  # Too broad
```

### Multi-Memory Search Strategy

1. **Use search for cross-cutting concerns:**
   ```
   query="Implementing memory consolidation feature from start to finish"
   # Will return relevant info from Architecture, API, Config, and Library References
   ```

2. **Get specific memory when you know what you need:**
   ```
   # If you just need to check current phase:
   mcp_agent-mem_get_active_memories → check memory ID 6
   ```

## Memory Update Frequency

### Update Frequently:
- **Development Status** - Every major task or phase transition
- **Issues and Bugs** - Immediately when bug found or resolved
- **Library References** - When discovering new patterns or gotchas

### Update Occasionally:
- **Architecture Design** - When design decisions change
- **Configuration** - When adding new dependencies or env vars

### Rarely Update:
- **Project Overview** - Stable information
- **Database Design** - Only if schema changes
- **API Design** - Only if API contracts change

## Integration with Development Workflow

### Starting New Phase
1. Search memories for phase requirements
2. Update "Development Status" → current_phase
3. Check "Library References" for relevant tools
4. Begin implementation

### During Development
1. Search when stuck or need context
2. Add bugs to "Issues and Bugs" as discovered
3. Update "Development Status" → completed_tasks regularly
4. Document learnings in "Library References"

### Completing Phase
1. Update "Development Status" → mark phase complete
2. Resolve any issues in "Issues and Bugs"
3. Update "Development Status" → next_steps for next phase
4. Commit any architecture or API changes to memories

### End of Session
1. Update "Development Status" with current state
2. Document any blockers
3. List next steps clearly
4. Ensure all new bugs are recorded

## Quick Reference Commands

```python
# Get all memories
mcp_agent-mem_get_active_memories(external_id="agent_mem_dev")

# Search across memories
mcp_agent-mem_search_memories(
    external_id="agent_mem_dev",
    query="your contextual search query",
    limit=10
)

# Update single section
mcp_agent-mem_update_memory_sections(
    external_id="agent_mem_dev",
    memory_id=<memory_id>,
    sections=[{
        "section_id": "<section_name>",
        "action": "replace" | "insert",
        "old_content": "...",  # For replace: exact match, for insert: insert after
        "new_content": "..."
    }]
)

# Update multiple sections at once
sections=[
    {"section_id": "current_phase", "action": "replace", ...},
    {"section_id": "next_steps", "action": "replace", ...}
]
```

## Memory IDs Reference

| Memory ID | Title | Key Sections |
|-----------|-------|--------------|
| 1 | Project Overview | purpose, core_features, tech_stack, timeline, release_info |
| 2 | Architecture Design | layers, component_responsibilities, design_patterns, data_flow |
| 3 | API Design | core_api_methods, mcp_tools, streamlit_interface |
| 4 | Configuration | environment_variables, dependencies_uv, setup_instructions, os_specifics |
| 5 | Testing Strategy | unit_tests, integration_tests, e2e_tests, status, tools |
| 6 | Development Status | current_phase, completed_tasks, next_steps, blockers |
| 7 | Library References | psqlpy_patterns, pydantic_usage, pydantic_ai_patterns, neo4j_usage, pytest_asyncio |
| 8 | Issues and Bugs | template_for_issues, issue_XXX_name (dynamic) |

## Important Rules

1. **Always use external_id="agent_mem_dev"** - Never use different ID
2. **Search before updating** - Understand current state first
3. **Be specific in updates** - Include enough context for old_content matching
4. **Document bugs thoroughly** - Use the template in Issues memory with reproduction steps
5. **Update Development Status frequently** - Keep progress transparent
6. **Use search for context** - Don't guess, search memories
7. **Keep sections focused** - Each section has one clear purpose
8. **Update blockers immediately** - Don't let blockers go undocumented

## Memory Update Frequency

### Update Frequently:
- **Development Status** - Every major task or phase transition
- **Issues and Bugs** - Immediately when bug found or resolved
- **Library References** - When discovering new patterns or gotchas

### Update Occasionally:
- **Architecture Design** - When design decisions change
- **Configuration** - When adding new dependencies or env vars

### Rarely Update:
- **Project Overview** - Stable information
- **API Design** - Only if API contracts change
- **Testing Strategy** - Only if approach changes

## Integration with Development Workflow

### Starting New Phase
1. Search memories for phase requirements
2. Update "Development Status" → current_phase
3. Check "Library References" for relevant tools
4. Begin implementation

### During Development
1. Search when stuck or need context
2. Add bugs to "Issues and Bugs" as discovered
3. Update "Development Status" → completed_tasks regularly
4. Document learnings in "Library References"

### Completing Phase
1. Update "Development Status" → mark phase complete
2. Resolve any issues in "Issues and Bugs"
3. Update "Development Status" → next_steps for next phase
4. Commit any architecture or API changes to memories

### End of Session
1. Update "Development Status" with current state
2. Document any blockers
3. List next steps clearly
4. Ensure all new bugs are recorded