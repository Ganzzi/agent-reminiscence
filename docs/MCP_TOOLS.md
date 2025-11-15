# MCP Tools Reference

Complete documentation for all 6 MCP tools available in agent-reminiscence v0.2.0

**Note**: MCP (Model Context Protocol) tools integrate with Claude Desktop and other MCP clients.

## Table of Contents

1. [Overview](#overview)
2. [Memory Management Tools](#memory-management-tools)
3. [Search Tools](#search-tools)
4. [Tool Schemas](#tool-schemas)
5. [Claude Desktop Integration](#claude-desktop-integration)

---

## Overview

The agent-reminiscence MCP server exposes 6 tools for managing agent memories:

| # | Tool | Purpose | Mode |
|---|------|---------|------|
| 1 | `get_active_memories` | Retrieve all agent memories | Read |
| 2 | `create_active_memory` | Create new memory | Write |
| 3 | `update_memory_sections` | Update memory content | Write |
| 4 | `delete_active_memory` | Delete memory | Write |
| 5 | `search_memories` | Fast pointer-based search | Search |
| 6 | `deep_search_memories` | Full synthesis search ⭐ NEW | Search |

---

## Memory Management Tools

### 1. get_active_memories

Retrieve all active memories for an agent.

**Purpose**: List all active memories for an agent, useful for discovering existing memories before updating or searching.

**Parameters**:
- `external_id` (string, required): Agent identifier (UUID, string, or int)

**Returns**:
```json
{
  "memories": [
    {
      "id": 1,
      "external_id": "agent-123",
      "title": "Current Task Progress",
      "sections": {
        "progress": {
          "content": "Working on authentication",
          "update_count": 2
        }
      },
      "created_at": "2025-11-15T10:00:00Z",
      "updated_at": "2025-11-15T10:30:00Z"
    }
  ],
  "count": 1
}
```

**Example Usage**:
```json
{
  "external_id": "agent-123"
}
```

**Error Cases**:
- Empty or invalid `external_id`
- Database connection errors

---

### 2. create_active_memory

Create a new active memory with template-driven structure.

**Purpose**: Set up a new memory structure for an agent with predefined sections.

**Parameters**:
- `external_id` (string, required): Agent identifier
- `title` (string, required): Human-readable memory title
- `template_content` (object, required): Template structure
  - `template`: { `id`, `name` }
  - `sections`: Array of { `id`, `description` }
- `initial_sections` (object, optional): Initial content for sections
- `metadata` (object, optional): Additional metadata

**Returns**:
```json
{
  "id": 2,
  "external_id": "agent-123",
  "title": "New Memory",
  "sections": {...},
  "created_at": "2025-11-15T10:00:00Z"
}
```

**Example Usage**:
```json
{
  "external_id": "agent-123",
  "title": "Project Planning",
  "template_content": {
    "template": {
      "id": "project_memory_v1",
      "name": "Project Memory"
    },
    "sections": [
      {
        "id": "goals",
        "description": "Project goals and objectives"
      },
      {
        "id": "progress",
        "description": "Completion status"
      }
    ]
  },
  "initial_sections": {
    "goals": {
      "content": "Launch MVP by Q1 2026"
    }
  }
}
```

---

### 3. update_memory_sections

Update multiple sections in an active memory.

**Purpose**: Modify existing memory content with support for replace or insert operations.

**Parameters**:
- `external_id` (string, required): Agent identifier
- `memory_id` (integer, required): ID of memory to update
- `sections` (array, required): Section updates
  - `section_id` (string): Section to update
  - `new_content` (string): New content
  - `old_content` (string, optional): Pattern to match for replacement
  - `action` (string): "replace" or "insert"

**Returns**:
```json
{
  "id": 1,
  "sections_updated": 2,
  "total_update_count": 5
}
```

**Example Usage**:
```json
{
  "external_id": "agent-123",
  "memory_id": 1,
  "sections": [
    {
      "section_id": "progress",
      "action": "replace",
      "new_content": "Authentication module 50% complete"
    },
    {
      "section_id": "notes",
      "action": "insert",
      "new_content": "\n- Implemented JWT tokens\n- Added refresh token logic"
    }
  ]
}
```

---

### 4. delete_active_memory

Delete an active memory.

**Purpose**: Remove a memory when no longer needed.

**Parameters**:
- `external_id` (string, required): Agent identifier
- `memory_id` (integer, required): ID of memory to delete

**Returns**:
```json
{
  "success": true,
  "deleted_id": 1,
  "message": "Memory deleted successfully"
}
```

**Example Usage**:
```json
{
  "external_id": "agent-123",
  "memory_id": 1
}
```

---

## Search Tools

### 5. search_memories

Fast pointer-based memory retrieval without AI synthesis.

**Purpose**: Quickly search memories when you need direct results without AI processing.

**Mode**: Pointer-based (< 200ms)

**Parameters**:
- `external_id` (string, required): Agent identifier
- `query` (string, required): Search query
- `limit` (integer, optional): Max results (1-100, default: 10)
- `synthesis` (boolean, optional): Force synthesis (default: false)

**Returns**:
```json
{
  "mode": "pointer",
  "chunks": [
    {
      "id": "chunk-123",
      "content": "Authentication implementation details...",
      "tier": "shortterm",
      "score": 0.95,
      "importance": 0.85
    }
  ],
  "entities": [...],
  "relationships": [...],
  "synthesis": null,
  "result_counts": {
    "chunks": 5,
    "entities": 3,
    "relationships": 2
  }
}
```

**Example Usage**:
```json
{
  "external_id": "agent-123",
  "query": "JWT authentication implementation",
  "limit": 5
}
```

**Use Cases**:
- Quick lookups
- Recent information retrieval
- Direct fact finding
- Performance-critical searches

---

### 6. deep_search_memories ⭐ NEW

Comprehensive memory retrieval with full synthesis, entity extraction, and relationship analysis.

**Purpose**: Perform deep analysis of memories with AI-generated summaries and entity insights.

**Mode**: Synthesis-based (500ms - 2s)

**Parameters**:
- `external_id` (string, required): Agent identifier
- `query` (string, required): Search query (can be complex)
- `limit` (integer, optional): Results to retrieve (1-100, default: 10)
- `synthesis` (boolean, optional): Enable synthesis (default: true)

**Returns**:
```json
{
  "mode": "synthesis",
  "chunks": [...],
  "entities": [
    {
      "id": "entity-123",
      "name": "JWT Authentication",
      "types": ["Technical_Concept"],
      "description": "JSON Web Token based auth",
      "importance": 0.92
    }
  ],
  "relationships": [
    {
      "from_entity_name": "JWT Authentication",
      "to_entity_name": "API Security",
      "types": ["Implements"],
      "importance": 0.88
    }
  ],
  "synthesis": "JWT authentication is the core security mechanism for our API. It implements token-based auth...",
  "result_counts": {
    "chunks": 8,
    "entities": 5,
    "relationships": 4
  }
}
```

**Example Usage**:
```json
{
  "external_id": "agent-123",
  "query": "How does JWT authentication relate to our API security design decisions?",
  "limit": 10
}
```

**Features**:
- AI-synthesized summary
- Entity extraction and relationships
- Confidence scoring
- Temporal context
- Multi-part query support

**Use Cases**:
- Comprehensive analysis
- Understanding relationships
- Complex question answering
- Decision support
- Entity relationship analysis

**Token Cost**: Higher than `search_memories` due to AI synthesis

---

## Tool Schemas

### Common Parameter Schema

All tools use JSON objects for parameters:

```json
{
  "type": "object",
  "properties": {
    "external_id": {
      "type": "string",
      "description": "Agent identifier"
    }
  },
  "required": ["external_id"]
}
```

### Search Parameters Schema

Both search tools use:

```json
{
  "type": "object",
  "properties": {
    "external_id": {
      "type": "string",
      "description": "Agent identifier"
    },
    "query": {
      "type": "string",
      "description": "Search query"
    },
    "limit": {
      "type": "integer",
      "minimum": 1,
      "maximum": 100,
      "default": 10
    },
    "synthesis": {
      "type": "boolean",
      "default": false  // For search_memories
      // default: true   // For deep_search_memories
    }
  },
  "required": ["external_id", "query"]
}
```

---

## Claude Desktop Integration

### Setup

1. Ensure MCP server is running:
```bash
python -m agent_reminiscence_mcp.run
```

2. Configure Claude Desktop (macOS):
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

3. Restart Claude Desktop to load tools

### Available in Claude

Once configured, all 6 tools are available in Claude:

- `get_active_memories(external_id)` - List agent memories
- `create_active_memory(...)` - Create new memory
- `update_memory_sections(...)` - Update memory content
- `delete_active_memory(...)` - Delete memory
- `search_memories(...)` - Fast search
- `deep_search_memories(...)` - Deep search with synthesis

### Example Claude Conversation

```
You: "List all memories for agent-123"
Claude: I'll retrieve all memories for that agent.
[Uses: get_active_memories(external_id="agent-123")]
Result: Found 3 active memories...

You: "Search for information about authentication"
Claude: I'll search for authentication-related information.
[Uses: search_memories(external_id="agent-123", query="authentication")]
Result: Found 5 relevant chunks...

You: "Analyze how JWT connects to our API design"
Claude: I'll perform a deep analysis with synthesis.
[Uses: deep_search_memories(external_id="agent-123", 
       query="How does JWT relate to API design?")]
Result: JWT is the core security mechanism... [synthesis provided]
```

---

## Best Practices

### Choosing Between search_memories and deep_search_memories

| Use search_memories when: | Use deep_search_memories when: |
|---|---|
| You need quick results | You need comprehensive analysis |
| Looking for specific facts | Analyzing relationships |
| Response time critical | Understanding connections |
| Simple queries | Complex multi-part questions |
| < 200ms acceptable | 500ms-2s acceptable |

### Query Guidelines

**Effective queries**:
- "How does X relate to Y?"
- "List all decisions about authentication"
- "Summarize our approach to caching"

**Ineffective queries**:
- "stuff" (too vague)
- Single words (not enough context)
- "all" (too broad)

### External ID Best Practices

- Use UUID for production systems
- Use string for development/testing
- Keep ID format consistent
- Never use empty string

---

## Error Handling

### Common Errors

**Invalid external_id**:
```json
{
  "error": "ValueError: external_id cannot be empty"
}
```

**Memory not found**:
```json
{
  "error": "Memory with id 999 not found"
}
```

**Database connection error**:
```json
{
  "error": "Failed to connect to database"
}
```

### Error Recovery

1. Check external_id format
2. Verify memory exists before updating
3. Ensure MCP server is running
4. Check database connections
5. Review query syntax

---

## Version

- MCP Server Version: 0.2.0
- Tools: 6
- Last Updated: November 16, 2025
- Protocol: Model Context Protocol 1.0
