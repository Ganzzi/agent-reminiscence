# MCP Integration Guide

Complete guide to using agent-reminiscence with Model Context Protocol and Claude Desktop.

## Table of Contents

1. [What is MCP?](#what-is-mcp)
2. [MCP Architecture](#mcp-architecture)
3. [Available Tools](#available-tools)
4. [Claude Desktop Setup](#claude-desktop-setup)
5. [Usage Examples](#usage-examples)
6. [Tool Schemas](#tool-schemas)
7. [Error Handling](#error-handling)
8. [Best Practices](#best-practices)

---

## What is MCP?

**Model Context Protocol (MCP)** is an open protocol that allows AI assistants like Claude to access external tools and data sources.

### Why MCP?

- **Standardized**: Works with any MCP client (Claude, others)
- **Secure**: Client controls what tools are available
- **Flexible**: Tools can be local or remote
- **Extensible**: Easy to add new tools

### MCP in Agent Reminiscence

Agent-reminiscence v0.2.0+ provides a complete MCP server that exposes all memory management capabilities as standardized tools.

**Available Tools: 6 total**
- 4 memory management tools (CRUD operations)
- 2 search tools (fast pointer-based + deep synthesis)

---

## MCP Architecture

### MCP Communication Flow

```
┌─────────────────────────┐
│   Claude Desktop        │
│   (MCP Client)          │
└────────────┬────────────┘
             │
             │ MCP Protocol
             │ (JSON-RPC over stdio)
             ↓
┌─────────────────────────────────────────┐
│    Agent Reminiscence MCP Server        │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │  Tool Dispatcher                │   │
│  ├─────────────────────────────────┤   │
│  │ Tool 1: get_active_memories     │   │
│  │ Tool 2: create_active_memory    │   │
│  │ Tool 3: update_memory_sections  │   │
│  │ Tool 4: delete_active_memory    │   │
│  │ Tool 5: search_memories         │   │
│  │ Tool 6: deep_search_memories ⭐ │   │
│  └────────────┬────────────────────┘   │
│               │                         │
│               ↓                         │
│  ┌─────────────────────────────────┐   │
│  │  AgentMem Core API              │   │
│  │  (Python async methods)         │   │
│  └─────────────────────────────────┘   │
└────────────────────────────────────────┘
             │
             ↓
┌──────────────────────┐
│  PostgreSQL + Neo4j  │
│  (Data Storage)      │
└──────────────────────┘
```

### Request-Response Cycle

```
1. Claude: "Create a memory for project tracking"
   ↓
2. MCP Client: Calls tool with parameters
   {
     "tool": "create_active_memory",
     "params": {...}
   }
   ↓
3. MCP Server: Validates input, calls AgentMem
   ↓
4. AgentMem: Processes request (PostgreSQL + Neo4j)
   ↓
5. MCP Server: Returns structured result
   {
     "success": true,
     "data": {...}
   }
   ↓
6. Claude: Displays result in conversation
```

---

## Available Tools

### Overview

| # | Tool Name | Purpose | Type | Input | Output |
|---|-----------|---------|------|-------|--------|
| 1 | `get_active_memories` | List agent memories | Read | `external_id` | Array of memories |
| 2 | `create_active_memory` | Create new memory | Write | `external_id`, config | Memory object |
| 3 | `update_memory_sections` | Modify memory sections | Write | `external_id`, content | Update result |
| 4 | `delete_active_memory` | Remove memory | Write | `external_id`, `memory_id` | Confirmation |
| 5 | `search_memories` | Fast search (< 200ms) | Read | `external_id`, query | Search results |
| 6 | `deep_search_memories` | Full synthesis search ⭐ | Read | `external_id`, query | Results + synthesis |

### Tool Descriptions

#### 1. get_active_memories

Lists all active memories for an agent.

```
Input:
  - external_id (string): Agent identifier

Output:
  - Array of memory objects with sections
  - Timestamps and metadata
```

**Use When**: You need to discover what memories exist before querying

#### 2. create_active_memory

Creates a new templated memory.

```
Input:
  - external_id (string): Agent identifier
  - title (string): Memory name
  - template_content (object): Template structure
  - initial_sections (object, optional): Initial content

Output:
  - Created memory object with ID and timestamp
```

**Use When**: Setting up a new memory structure

#### 3. update_memory_sections

Updates existing memory sections.

```
Input:
  - external_id (string): Agent identifier
  - memory_id (number): Which memory to update
  - sections (array): What to update
    - section_id: Name of section
    - new_content: Content to write
    - action: "replace" or "insert"

Output:
  - Update confirmation with change count
```

**Use When**: Modifying existing memories with new information

#### 4. delete_active_memory

Removes a memory permanently.

```
Input:
  - external_id (string): Agent identifier
  - memory_id (number): Which memory to delete

Output:
  - Deletion confirmation
```

**Use When**: No longer need a memory

#### 5. search_memories

Fast pointer-based search (< 200ms).

```
Input:
  - external_id (string): Agent identifier
  - query (string): Search terms
  - limit (number, 1-100): Max results (default 10)

Output:
  - Chunks (matched content)
  - Entities (extracted topics)
  - Relationships (connections)
  - No synthesis
```

**Use When**: You need quick results without AI synthesis

#### 6. deep_search_memories ⭐

Comprehensive search with AI synthesis (500ms-2s).

```
Input:
  - external_id (string): Agent identifier
  - query (string): Question or search query
  - limit (number, 1-100): Results to retrieve (default 10)

Output:
  - Chunks (matched content)
  - Entities (topics with descriptions)
  - Relationships (connections with types)
  - Synthesis (AI-generated summary)
```

**Use When**: You need analysis and synthesis, not just matching

---

## Claude Desktop Setup

### Prerequisites

1. Claude Desktop installed (latest version)
2. Python 3.10+ installed
3. agent-reminiscence v0.2.0+ installed
4. PostgreSQL, Neo4j, Ollama running (via Docker)

### Step 1: Start Services

```bash
docker-compose up -d
```

Verify:
```bash
docker ps
# Should show: postgres, neo4j, ollama
```

### Step 2: Start MCP Server

```bash
python -m agent_reminiscence_mcp.run
```

Expected output:
```
🚀 Starting Agent Reminiscence MCP Server...
✅ Initializing AgentMem...
✅ Registering tools (6 total)...
✅ Ready: Serving on stdio
```

**Important**: Keep this terminal open while using Claude

### Step 3: Configure Claude Desktop

**On macOS**:

Edit: `~/.config/claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "agent-mem": {
      "command": "python",
      "args": ["-m", "agent_reminiscence_mcp.run"],
      "env": {
        "POSTGRES_HOST": "localhost",
        "POSTGRES_PORT": "5432",
        "POSTGRES_DB": "agent_reminiscence",
        "POSTGRES_USER": "postgres",
        "POSTGRES_PASSWORD": "password",
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USERNAME": "neo4j",
        "NEO4J_PASSWORD": "password",
        "OLLAMA_BASE_URL": "http://localhost:11434",
        "EMBEDDING_MODEL": "nomic-embed-text"
      }
    }
  }
}
```

**On Windows**:

Edit: `%APPDATA%\Claude\claude_desktop_config.json`

(Same JSON format as macOS)

### Step 4: Restart Claude Desktop

1. Completely close Claude Desktop
2. Wait 3 seconds
3. Reopen Claude Desktop
4. New conversation will have access to 6 tools

### Step 5: Verify Tools Are Available

In Claude, you should see a "Tools" section with 6 agent-reminiscence tools available.

---

## Usage Examples

### Example 1: Simple Query

```
You: "Create a memory to track my fitness goals"

Claude: I'll create a fitness tracking memory for you.
[Uses: create_active_memory(
  external_id="fitness-tracker",
  title="Fitness Goals",
  template_content={
    "template": {"id": "fitness_v1", "name": "Fitness"},
    "sections": [
      {"id": "goals", "description": "Fitness goals"},
      {"id": "progress", "description": "Weekly progress"}
    ]
  }
)]

✅ Created memory "Fitness Goals" (ID: 42)

You: "Update my progress to 3 workouts this week"

Claude: I'll update your progress.
[Uses: update_memory_sections(
  external_id="fitness-tracker",
  memory_id=42,
  sections=[{
    "section_id": "progress",
    "action": "replace",
    "new_content": "Week 1: 3 workouts completed"
  }]
)]

✅ Updated successfully

You: "What fitness goals did I set?"

Claude: I'll search for your fitness goals.
[Uses: search_memories(
  external_id="fitness-tracker",
  query="fitness goals",
  limit=5
)]

Found: Your goals include...
```

### Example 2: Complex Analysis

```
You: "I've been learning about web development for 3 months. 
Create memories for each major topic I should cover."

Claude: I'll create a learning structure.
[Creates 5 memories: HTML/CSS, JavaScript, Databases, 
Backend Development, Deployment]

You: "Now do a deep analysis of how these topics connect together"

Claude: I'll analyze the relationships between topics.
[Uses: deep_search_memories(
  external_id="web-learning",
  query="How do HTML, CSS, JavaScript, Databases, and Deployment 
  relate to each other in web development?",
  limit=20
)]

Synthesis: HTML and CSS form the frontend layer, which JavaScript 
enhances with interactivity. JavaScript also connects to the backend 
through APIs that communicate with Databases. Deployment ties all 
components together by managing the full stack...

Key Entities:
- Frontend: HTML/CSS (visual structure)
- Interactivity: JavaScript (client-side logic)
- Backend: Databases (data storage)
- Integration: APIs (communication layer)
- Deployment: Full stack management

Relationships:
- JavaScript uses APIs to Databases
- Frontend displays data from APIs
- Deployment manages entire stack
```

### Example 3: Project Management

```
You: "Set up a memory system for project XYZ with phases"

Claude: I'll create a comprehensive project tracking system.
[Creates: Project Overview, Phase 1, Phase 2, Phase 3, Risks]

You: "Update Phase 1 to 60% complete"

[Uses: update_memory_sections(...)]

You: "List all project memories"

[Uses: get_active_memories(external_id="project-xyz")]

Memories:
1. Project Overview - Q1 2026 launch target
2. Phase 1 - 60% complete (Design & Planning)
3. Phase 2 - Not started (Development)
4. Phase 3 - Not started (Testing & Launch)
5. Risks - 3 identified risks

You: "Search for blockers we've identified"

[Uses: search_memories(external_id="project-xyz", query="blockers risks")]

Found 3 items mentioning blockers or risks...
```

---

## Tool Schemas

### Input Schema - Search Tools

Both `search_memories` and `deep_search_memories` use the same input schema:

```json
{
  "type": "object",
  "properties": {
    "external_id": {
      "type": "string",
      "description": "Agent identifier (UUID, string, or integer)"
    },
    "query": {
      "type": "string",
      "description": "Search query or question"
    },
    "limit": {
      "type": "integer",
      "description": "Maximum results to return",
      "minimum": 1,
      "maximum": 100,
      "default": 10
    }
  },
  "required": ["external_id", "query"]
}
```

### Output Schema - Search Results

```json
{
  "mode": "pointer|synthesis",
  "chunks": [
    {
      "id": "string",
      "content": "string",
      "tier": "active|shortterm|longterm",
      "score": 0.0-1.0,
      "importance": 0.0-1.0
    }
  ],
  "entities": [
    {
      "id": "integer",
      "name": "string",
      "types": ["string"],
      "description": "string",
      "importance": 0.0-1.0
    }
  ],
  "relationships": [
    {
      "from_entity_name": "string",
      "to_entity_name": "string",
      "types": ["string"],
      "description": "string",
      "importance": 0.0-1.0
    }
  ],
  "synthesis": "string|null"
}
```

---

## Error Handling

### Common Errors

#### 1. Connection Error

```
Error: "Failed to connect to PostgreSQL"

Solution:
1. Check Docker is running: docker ps
2. Verify .env file configuration
3. Check port 5432 is available
4. Restart services: docker-compose restart
```

#### 2. Tool Not Available

```
Error: "Tool 'deep_search_memories' not found"

Solution:
1. Ensure MCP server is running
2. Check Claude Desktop config file
3. Verify MCP server started without errors
4. Restart Claude Desktop
5. Check terminal running MCP server
```

#### 3. Invalid Parameters

```
Error: "external_id is required"

Solution:
1. Check all required parameters are provided
2. Verify parameter types match schema
3. Check for typos in parameter names
4. Review error message for specific issue
```

#### 4. Memory Not Found

```
Error: "Memory with id 999 not found"

Solution:
1. Use get_active_memories to find valid IDs
2. Verify external_id is correct
3. Check memory wasn't already deleted
4. Double-check memory_id value
```

### Error Recovery Strategies

**For Transient Errors** (timeouts, connection issues):
1. Wait a few seconds
2. Retry the operation
3. If persistent, restart services

**For Configuration Errors**:
1. Verify .env file syntax
2. Check port availability
3. Verify credentials are correct
4. Restart MCP server

**For Data Errors**:
1. Check memory ID exists
2. Verify external_id format
3. Review query syntax
4. Check parameter values

---

## Best Practices

### 1. External ID Strategy

**Good**:
```python
# Use consistent format
external_id = "user-123"          # Human readable
external_id = str(uuid4())        # Unique
external_id = "project-alpha-v2"  # Descriptive
```

**Avoid**:
```python
external_id = ""                # Empty string
external_id = "x"              # Too short
external_id = "123 456"        # Contains spaces
```

### 2. Query Design

**Good**:
```
"What are the main project goals?"
"How do APIs relate to databases?"
"List all completed milestones"
```

**Avoid**:
```
"stuff"                    # Too vague
"tell me everything"       # Too broad
"api"                      # Single word
```

### 3. Search Strategy

**Fast Search** (< 200ms) for:
- Recent information lookups
- Specific fact finding
- UI/interface responsiveness
- Real-time queries

**Deep Search** (500ms-2s) for:
- Comprehensive analysis
- Relationship discovery
- Decision support
- Complex questions

### 4. Memory Organization

**Structure**:
```
Create specialized memories for different domains
- One memory per major topic
- Clear, descriptive section names
- Initial sections with seed content
```

**Avoid**:
```
Storing everything in one giant memory
- Hard to search effectively
- Updates become unwieldy
- Synthesis less focused
```

### 5. Error Handling in Claude

Claude automatically handles many errors, but:
- Request clarification for ambiguous queries
- Verify memory exists before updating
- Check external_id is consistent
- Use descriptive error messages

### 6. Performance Optimization

**For Better Results**:
1. Use specific queries
2. Keep memories focused
3. Use appropriate search type
4. Limit results to needed amount
5. Update memories regularly

**For Faster Response**:
1. Use fast search when possible
2. Reduce result limit
3. Use more specific queries
4. Batch operations efficiently

---

## Troubleshooting Checklist

### Setup Issues

- [ ] Docker services running (docker ps)
- [ ] MCP server running (python -m agent_reminiscence_mcp.run)
- [ ] Claude Desktop restarted after config change
- [ ] Config file syntax valid (JSON)
- [ ] Environment variables set correctly
- [ ] Correct paths in config

### Runtime Issues

- [ ] MCP server not showing errors in terminal
- [ ] Tools appearing in Claude (bottom toolbar)
- [ ] Parameters match required schema
- [ ] external_id not empty or invalid
- [ ] Memory ID exists before updating/deleting
- [ ] Query is specific enough

### Search Issues

- [ ] Database has content before searching
- [ ] Query is relevant to stored content
- [ ] Limit parameter is within 1-100
- [ ] No typos in external_id
- [ ] Using appropriate search type
- [ ] Ollama embeddings running

---

## Advanced Usage

### Chaining Operations

```
You: "Create a learning path for Python, update progress, 
then search for key concepts"

Claude: I'll create the structure, track progress, and analyze it.
[Creates memory] → [Updates section] → [Deep searches]
```

### Batch Operations

```
You: "Create 5 memories for different project phases"

Claude: I'll create all phases for you.
[Calls create_active_memory 5 times]
```

### Conditional Logic

```
You: "If I have a 'Fitness' memory, update it. 
Otherwise create it first."

Claude: Let me check if it exists.
[Calls get_active_memories] → [Conditionally creates or updates]
```

---

## Version

- MCP Integration Guide: v0.2.0
- MCP Server: v0.2.0
- Last Updated: November 16, 2025
- MCP Protocol: 1.0
