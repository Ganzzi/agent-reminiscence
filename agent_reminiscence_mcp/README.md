# AgentMem MCP Server

**Model Context Protocol (MCP) Server** for AgentMem - exposes memory management functionality to MCP clients like Claude Desktop.

---

## 📁 Directory Structure

```
agent_reminiscence_mcp/
├── __init__.py          # Module exports
├── server.py            # Main MCP server with 3 tools (320 lines)
├── schemas.py           # JSON Schema definitions for inputs
├── __main__.py          # CLI entry point
├── run.py               # Simple runner script
├── test_server.py       # Test server structure
└── README.md            # This file
```

---

## 🎯 Three Tools

1. **`get_active_memories`** - Get all active memories for an agent
2. **`update_memory_sections`** - Batch update multiple sections at once
3. **`search_memories`** - Search across memory tiers (shortterm/longterm)

---

## 🚀 Quick Start

### Prerequisites

Ensure these services are running:
- **PostgreSQL** (with agent_reminiscence database)
- **Neo4j**
- **Ollama** (with nomic-embed-text model)

### Environment Variables

Create a `.env` file in the project root:

```env
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password
POSTGRES_DB=agent_reminiscence

NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

OLLAMA_BASE_URL=http://localhost:11434
```

### Test Server Structure

```powershell
# Test without database connections
py agent_reminiscence_mcp/test_server.py
```

### Run the Server

```powershell
# Method 1: Using uv (recommended)
uv run agent_reminiscence_mcp/run.py

# Method 2: Direct run with Python
py agent_reminiscence_mcp/run.py

# Method 3: As module
py -m agent_reminiscence_mcp

# Method 4: With MCP Inspector
mcp dev mcp_dev.py
```

**Logs**: Server logs are written to `agent_reminiscence_mcp/logs/mcp_server_TIMESTAMP.log`

---

## 🔌 Claude Desktop Integration

Add to your Claude Desktop config file:

**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "agent-mem": {
      "command": "uv",
      "args": [
        "run",
        "path_to_agent_reminiscence_mcp\\run.py"
      ],
      "env": {
        "POSTGRES_HOST": "localhost",
        "POSTGRES_PORT": "5432",
        "POSTGRES_USER": "postgres",
        "POSTGRES_PASSWORD": "postgres",
        "POSTGRES_DB": "agent_reminiscence",
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
- Use absolute path for reliability
- Adjust the path to match your installation directory
- Double backslashes required in JSON on Windows

---

## 📋 Tool Specifications

### 1. get_active_memories

**Input**:
```json
{
  "external_id": "agent-123"
}
```

**Output**:
```json
{
  "memories": [
    {
      "id": 1,
      "external_id": "agent-123",
      "title": "Task Memory",
      "sections": {
        "current_task": {
          "content": "...",
          "update_count": 3
        }
      },
      "created_at": "2025-10-04T10:00:00",
      "updated_at": "2025-10-04T12:30:00"
    }
  ],
  "count": 1
}
```

---

### 2. update_memory_sections

**Input**:
```json
{
  "external_id": "agent-123",
  "memory_id": 1,
  "sections": [
    {
      "section_id": "current_task",
      "new_content": "Updated current task content"
    },
    {
      "section_id": "progress",
      "new_content": "Updated progress information"
    }
  ]
}
```

**Output**:
```json
{
  "memory": { ... },
  "updates": [
    {
      "section_id": "current_task",
      "previous_count": 3,
      "new_count": 4
    },
    {
      "section_id": "progress",
      "previous_count": 1,
      "new_count": 2
    }
  ],
  "total_sections_updated": 2,
  "message": "Successfully updated 2 sections"
}
```

**Features**:
- ✅ Validates memory exists
- ✅ Validates all sections exist
- ✅ Batch updates multiple sections atomically
- ✅ Tracks update count changes for each section
- ✅ Auto-triggers consolidation at threshold

---

### 3. search_memories

**Input**:
```json
{
  "external_id": "agent-123",
  "query": "How did I implement authentication?",
  "search_shortterm": true,
  "search_longterm": true,
  "limit": 10
}
```

**Output**:
```json
{
  "query": "How did I implement authentication?",
  "synthesized_response": "Based on your memories, you implemented...",
  "active_memories": [...],
  "shortterm_chunks": [
    {
      "id": 5,
      "content": "Implemented JWT authentication...",
      "similarity_score": 0.89,
      "bm25_score": 3.2
    }
  ],
  "longterm_chunks": [...],
  "entities": [
    {
      "id": 1,
      "name": "AuthService",
      "type": "class",
      "confidence": 0.95,
      "memory_tier": "shortterm"
    }
  ],
  "relationships": [...],
  "result_counts": {
    "active": 1,
    "shortterm": 3,
    "longterm": 2,
    "entities": 2,
    "relationships": 1
  }
}
```

---

## 🏗️ Architecture

```
Low-Level Server (mcp.server.lowlevel.Server)
    │
    ├── Lifespan Management
    │   ├── Initialize AgentMem on startup
    │   └── Close AgentMem on shutdown
    │
    ├── Tool Registration (@server.list_tools)
    │   ├── get_active_memories
    │   ├── update_memory_section
    │   └── search_memories
    │
    └── Tool Execution (@server.call_tool)
        ├── Route to handler
        ├── Access AgentMem from lifespan context
        └── Return JSON-formatted TextContent
```

### Key Design Decisions

1. **Low-Level Server API**
   - Uses `mcp.server.lowlevel.Server` (not FastMCP)
   - Explicit JSON Schema for inputs
   - Manual response formatting

2. **Stateless Design**
   - Single AgentMem instance in lifespan
   - external_id passed with each request
   - No per-client state

3. **Error Handling**
   - Validation in tool handlers
   - Clear error messages
   - Try-except with fallback

---

## 🧪 Testing

### Test Server Structure

```powershell
py agent_reminiscence_mcp/test_server.py
```

This verifies:
- ✅ Server can be imported
- ✅ All 3 tools are registered
- ✅ Input schemas are correct
- ✅ No database connection required

### Test with Real Database

1. Start services (PostgreSQL, Neo4j, Ollama)
2. Set environment variables
3. Add sample data:
   ```powershell
   uv run agent_reminiscence_mcp/tests/add_sample_data.py
   ```
4. Run server:
   ```powershell
   uv run agent_reminiscence_mcp/run.py
   ```
5. Test with client:
   ```powershell
   uv run agent_reminiscence_mcp/tests/test_mcp_client.py
   ```

**Server Logs**: Check `agent_reminiscence_mcp/logs/` for detailed logs of server activity.

### Test with MCP Inspector

If you have the MCP Inspector installed:

```powershell
mcp dev agent_reminiscence_mcp/tests/mcp_dev.py
```

This opens a web UI where you can:
- View all available tools
- Test tool inputs
- See formatted outputs
- Debug issues

---

## 🐛 Troubleshooting

### "Module 'mcp' has no attribute 'server'"

**Problem**: Naming conflict between our directory and mcp package.

**Solution**: We renamed directory from `mcp` to `agent_reminiscence_mcp`.

---

### "Failed to connect to Neo4j"

**Problem**: Neo4j service not running.

**Solutions**:
1. Start Neo4j: `docker start neo4j` or `neo4j start`
2. Check connection: `bolt://localhost:7687`
3. Verify credentials in `.env`

---

### "Connection refused" for PostgreSQL

**Problem**: PostgreSQL not running or wrong port.

**Solutions**:
1. Start PostgreSQL: `docker start postgres` or `pg_ctl start`
2. Check port (default: 5432)
3. Verify credentials

---

### "Model not found" for Ollama

**Problem**: Embedding model not pulled.

**Solution**:
```powershell
ollama pull nomic-embed-text
```

---

### Server starts but no output

**Expected**: The server waits for stdio input from MCP client (like Claude Desktop).

**To test manually**:
- Use MCP Inspector: `mcp dev mcp_dev.py`
- Or integrate with Claude Desktop

---

## 📝 Development Notes

### File Organization

- **server.py**: Main logic, lifespan, tool handlers (~320 lines)
- **schemas.py**: JSON Schema definitions (~75 lines)
- **run.py**: Simple runner for testing
- **test_server.py**: Structural validation without DB

### Adding New Tools

1. Add input schema to `schemas.py`
2. Add tool to `handle_list_tools()` in `server.py`
3. Add handler in `handle_call_tool()` routing
4. Create handler function `_handle_<tool_name>()`
5. Test with `test_server.py`

---

## 🔗 Related Documentation

- **Main README**: `../README.md`
- **Implementation Plan**: `../docs/MCP_SERVER_IMPLEMENTATION_PLAN.md`
- **Implementation Complete**: `../docs/MCP_IMPLEMENTATION_COMPLETE.md`
- **AgentMem Architecture**: `../docs/ARCHITECTURE.md`

---

## ✅ Status

- [x] Core implementation complete
- [x] All 3 tools working
- [x] Error handling implemented
- [x] Moved to root directory
- [ ] Tested with MCP Inspector
- [ ] Tested with Claude Desktop
- [ ] Production deployment guide

---

**Version**: 1.0.0  
**Date**: October 4, 2025  
**Status**: ✅ Ready for Testing

