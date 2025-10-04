# AgentMem MCP Server - Getting Started Guide

## ✅ What's Been Completed

### 1. Services Started
- ✅ PostgreSQL (port 5432) - **HEALTHY**
- ✅ Neo4j (ports 7474, 7687) - **RUNNING**
- ✅ Ollama (port 11434) - **RUNNING**

### 2. Models Ready
- ✅ `nomic-embed-text:latest` - Embedding model (274 MB)
- ✅ Gemini 2.0 Flash - Via API key (configured in .env)

### 3. MCP Server Implementation
- ✅ `agent_mem_mcp/` directory created at root
- ✅ Low-level Server API implementation
- ✅ 3 tools with JSON schemas
- ✅ Server structure validated
- ✅ Server currently **RUNNING** (waiting for client)

---

## 🚀 Next Steps: Testing the Server

### Option 1: Test with MCP Inspector (Recommended)

The MCP Inspector provides a web UI to test your server.

1. **Stop the current server** (Ctrl+C in terminal)

2. **Install MCP CLI** (if not already installed):
   ```powershell
   pip install mcp[cli]
   ```

3. **Start MCP Inspector**:
   ```powershell
   mcp dev mcp_dev.py
   ```

4. **Open browser** at the URL shown (usually http://localhost:5173)

5. **Test the tools**:
   - Click on each tool to see its schema
   - Enter test inputs
   - See formatted outputs

### Option 2: Integrate with Claude Desktop

1. **Locate Claude config file**:
   ```
   %APPDATA%\Claude\claude_desktop_config.json
   ```

2. **Add this configuration**:
   ```json
   {
     "mcpServers": {
       "agent-mem": {
         "command": "py",
         "args": [
           "-m",
           "agent_mem_mcp"
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
           "OLLAMA_BASE_URL": "http://localhost:11434",
           "GEMINI_API_KEY": "AIzaSyCD7ZhcN-srbeCxG8d9r0NGKeuufpOguHo"
         }
       }
     }
   }
   ```

3. **Restart Claude Desktop**

4. **Use the tools**:
   - Type: "Use get_active_memories for agent-123"
   - Type: "Search my memories for authentication"
   - etc.

---

## 🧪 Manual Testing

### Test 1: Get Active Memories

Create a test Python script:

```python
# test_mcp_client.py
import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_get_memories():
    server_params = StdioServerParameters(
        command="py",
        args=["agent_mem_mcp/run.py"],
        env=None
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # List available tools
            tools = await session.list_tools()
            print("Available tools:")
            for tool in tools.tools:
                print(f"  - {tool.name}")
            
            # Call get_active_memories
            result = await session.call_tool(
                "get_active_memories",
                arguments={"external_id": "test-agent-123"}
            )
            print("\nResult:", json.dumps(result.content, indent=2))

if __name__ == "__main__":
    asyncio.run(test_get_memories())
```

Run it:
```powershell
py test_mcp_client.py
```

---

## 📊 Current Status

### Services
```
✅ PostgreSQL:  localhost:5432  (HEALTHY)
✅ Neo4j:       bolt://localhost:7687
✅ Ollama:      http://localhost:11434
```

### MCP Server
```
✅ Location:    agent_mem_mcp/
✅ Status:      RUNNING (waiting for client)
✅ PID:         Check with: Get-Process python
```

### Database Status
- PostgreSQL: Tables initialized via SQL scripts
- Neo4j: Graph database ready
- Embeddings: nomic-embed-text model loaded

---

## 🐛 Troubleshooting

### Server won't start

**Check logs**:
```powershell
docker logs agent_mem_postgres
docker logs agent_mem_neo4j
docker logs agent_mem_ollama
```

### Database connection error

**Test connections**:
```powershell
# PostgreSQL
docker exec -it agent_mem_postgres psql -U postgres -d agent_mem -c "\dt"

# Neo4j
docker exec -it agent_mem_neo4j cypher-shell -u neo4j -p neo4jpassword "RETURN 1"

# Ollama
curl http://localhost:11434/api/tags
```

### Port conflicts

**Check if ports are in use**:
```powershell
netstat -ano | findstr "5432"  # PostgreSQL
netstat -ano | findstr "7687"  # Neo4j
netstat -ano | findstr "11434" # Ollama
```

---

## 📝 Environment Variables

Your `.env` file is already configured with:

```env
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=agent_mem

NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=neo4jpassword

OLLAMA_BASE_URL=http://localhost:11434

GEMINI_API_KEY=AIzaSyCD7ZhcN-srbeCxG8d9r0NGKeuufpOguHo
```

---

## 🎯 Tool Usage Examples

### 1. Get Active Memories

**Input**:
```json
{
  "external_id": "agent-123"
}
```

**Use case**: Retrieve all active working memories for an agent

---

### 2. Update Memory Section

**Input**:
```json
{
  "external_id": "agent-123",
  "memory_id": 1,
  "section_id": "current_task",
  "new_content": "Updated task: Implement user authentication"
}
```

**Use case**: Update a specific section when the agent learns something new

---

### 3. Search Memories

**Input**:
```json
{
  "external_id": "agent-123",
  "query": "How did I implement the login flow?",
  "search_shortterm": true,
  "search_longterm": true,
  "limit": 10
}
```

**Use case**: Search across memory tiers for relevant information

---

## 📚 Documentation

- **MCP Server README**: `agent_mem_mcp/README.md`
- **Implementation Plan**: `docs/MCP_SERVER_IMPLEMENTATION_PLAN.md`
- **Architecture**: `docs/ARCHITECTURE.md`
- **Main README**: `README.md`

---

## ✨ What You Can Do Now

1. **Test with MCP Inspector** - Interactive web UI testing
2. **Integrate with Claude Desktop** - Use tools in conversations
3. **Create test scripts** - Automated testing with Python
4. **Add sample data** - Populate with test memories
5. **Monitor logs** - Watch server behavior

---

**Status**: 🟢 **ALL SYSTEMS OPERATIONAL**

The MCP server is running and ready to accept connections from MCP clients!
