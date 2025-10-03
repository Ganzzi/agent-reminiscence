# 🚀 Quick Start Guide - Agent Mem

Get up and running with Agent Mem in 5 minutes!

## Prerequisites

- **Docker Desktop** (Windows/Mac) or **Docker Engine** (Linux)
- **Python 3.10+**
- **Git**

## Step 1: Run Setup Script

### Windows (PowerShell)

```powershell
# Run from the agent_mem directory
.\setup.ps1
```

### Linux/Mac (Bash)

```bash
# Make script executable
chmod +x setup.sh

# Run setup
./setup.sh
```

The setup script will:
1. ✅ Create `.env` file from template
2. ✅ Start Docker containers (PostgreSQL, Neo4j, Ollama)
3. ✅ Wait for services to be healthy
4. ✅ Initialize database schemas
5. ✅ Pull Ollama embedding model
6. ✅ Install Python dependencies

## Step 2: Verify Installation

### Check Docker Containers

```bash
docker compose ps
```

You should see 3 running containers:
- `agent_mem_postgres` (PostgreSQL with pgvector)
- `agent_mem_neo4j` (Neo4j graph database)
- `agent_mem_ollama` (Ollama for embeddings)

### Check Services

```bash
# PostgreSQL
docker compose exec postgres pg_isready -U agent_mem_user -d agent_mem

# Neo4j
docker compose exec neo4j cypher-shell -u neo4j -p agent_mem_neo4j_dev "RETURN 1"

# Ollama
curl http://localhost:11434/
```

## Step 3: Run Tests

### Run All Tests

```bash
pytest
```

### Run with Coverage

```bash
pytest --cov=agent_mem --cov-report=html
```

### Run Specific Test Categories

```bash
# Unit tests only
pytest -m unit

# Integration tests only
pytest -m integration

# Verbose output
pytest -v
```

## Step 4: Try Basic Example

```python
import asyncio
from agent_mem import AgentMem

async def main():
    # Initialize Agent Mem
    async with AgentMem() as agent_mem:
        # Create a memory
        memory = await agent_mem.create_active_memory(
            external_id="user-123",
            memory_type="conversation",
            sections={
                "summary": "Discussion about Python programming",
                "details": "User asked about async/await patterns",
            }
        )
        
        print(f"Created memory: {memory.id}")
        
        # Update memory
        updated = await agent_mem.update_active_memory(
            memory_id=str(memory.id),
            sections={
                "summary": "Extended discussion about Python async programming",
                "details": "Covered asyncio, async/await, and event loops",
            }
        )
        
        print(f"Updated memory (count: {updated.update_count})")
        
        # Retrieve memories
        results = await agent_mem.retrieve_memories(
            query="Tell me about Python programming",
            external_id="user-123"
        )
        
        print(f"Retrieved: {results}")

# Run
asyncio.run(main())
```

## Service URLs

### PostgreSQL
- **Host**: localhost:5432
- **Database**: agent_mem
- **User**: agent_mem_user
- **Password**: agent_mem_dev_password

**Connection String**:
```
postgresql://agent_mem_user:agent_mem_dev_password@localhost:5432/agent_mem
```

### Neo4j
- **Browser**: http://localhost:7474
- **Bolt**: bolt://localhost:7687
- **User**: neo4j
- **Password**: agent_mem_neo4j_dev

### Ollama
- **API**: http://localhost:11434
- **Model**: nomic-embed-text (768 dimensions)

## Common Commands

### Docker Management

```bash
# View logs
docker compose logs -f

# Stop containers
docker compose down

# Restart containers
docker compose restart

# Remove everything (including data)
docker compose down -v

# Rebuild containers
docker compose up -d --build
```

### Database Management

```bash
# PostgreSQL shell
docker compose exec postgres psql -U agent_mem_user -d agent_mem

# Neo4j shell
docker compose exec neo4j cypher-shell -u neo4j -p agent_mem_neo4j_dev

# View PostgreSQL logs
docker compose logs postgres

# View Neo4j logs
docker compose logs neo4j
```

### Ollama Management

```bash
# List models
docker compose exec ollama ollama list

# Pull a model
docker compose exec ollama ollama pull nomic-embed-text

# Remove a model
docker compose exec ollama ollama rm nomic-embed-text

# View Ollama logs
docker compose logs ollama
```

## Troubleshooting

### Port Already in Use

If you get "port already in use" errors:

```bash
# Check what's using the port (Windows)
netstat -ano | findstr :5432

# Stop the process or change the port in docker-compose.yml
```

### Docker Not Starting

1. Ensure Docker Desktop is running
2. Check Docker resources (CPU/Memory)
3. Try restarting Docker

### Database Connection Error

```bash
# Check container status
docker compose ps

# View logs
docker compose logs postgres
docker compose logs neo4j

# Restart containers
docker compose restart
```

### Ollama Model Not Found

```bash
# Pull the model manually
docker compose exec ollama ollama pull nomic-embed-text

# Verify model is available
docker compose exec ollama ollama list
```

### Tests Failing

```bash
# Ensure all services are running
docker compose ps

# Check service health
docker compose exec postgres pg_isready
docker compose exec neo4j cypher-shell -u neo4j -p agent_mem_neo4j_dev "RETURN 1"

# Restart services
docker compose restart

# Re-run setup
.\setup.ps1  # Windows
./setup.sh   # Linux/Mac
```

## Environment Variables

All configuration is in `.env`:

```bash
# PostgreSQL
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=agent_mem_user
POSTGRES_PASSWORD=agent_mem_dev_password
POSTGRES_DB=agent_mem

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=agent_mem_neo4j_dev

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
EMBEDDING_MODEL=nomic-embed-text

# Memory Settings
ACTIVE_MEMORY_UPDATE_THRESHOLD=5
SHORTTERM_PROMOTION_THRESHOLD=0.7
```

## Next Steps

1. **Run the tests**: `pytest -v`
2. **Check coverage**: `pytest --cov=agent_mem --cov-report=html`
3. **Try examples**: See `examples/` directory
4. **Read docs**: See `docs/` directory
5. **Build something**: Start using Agent Mem in your project!

## Clean Up

To stop and remove everything:

```bash
# Stop containers
docker compose down

# Remove volumes (deletes data)
docker compose down -v

# Remove all images
docker compose down --rmi all -v
```

## Need Help?

- **Documentation**: See `docs/` directory
- **Examples**: See `examples/` directory
- **Tests**: See `tests/` directory
- **Issues**: Check `docs/TROUBLESHOOTING.md`

---

**Ready to go!** 🎉

Start with:
```bash
pytest -v
```

Then explore the examples in `examples/basic_usage.py`.
