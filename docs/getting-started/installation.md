# Installation Guide

## Prerequisites

Agent Mem requires the following services:

1. **PostgreSQL** (v14+) with extensions
2. **Neo4j** (v5+) for graph storage
3. **Ollama** for embeddings and AI models

## Installation Options

### Option 1: Using Docker (Recommended)

This is the easiest way to get started. All services run in Docker containers.

#### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/agent-mem.git
cd agent-mem
```

#### Step 2: Configure Environment

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your preferred settings
# At minimum, change the passwords!
```

Important variables to configure:

```env
# PostgreSQL
POSTGRES_PASSWORD=your_secure_password_here

# Neo4j
NEO4J_PASSWORD=your_neo4j_password_here

# Google Gemini (for AI agents)
GEMINI_API_KEY=your_api_key_here
```

#### Step 3: Start Services

```bash
# Start all services (CPU-only)
docker-compose up -d

# Or with GPU support for Ollama
docker-compose --profile gpu up -d
```

#### Step 4: Initialize Database

The PostgreSQL schema is automatically initialized on first startup. Verify:

```bash
docker-compose logs postgres | grep "database system is ready"
```

#### Step 5: Pull Embedding Model

```bash
# Pull the embedding model
docker exec -it agent_mem_ollama ollama pull nomic-embed-text
```

#### Step 6: Install Python Package

```bash
# Install in your Python environment
pip install agent-mem

# Or for development
pip install -e ".[dev]"
```

### Option 2: Manual Installation

If you prefer to run services manually:

#### PostgreSQL Setup

1. Install PostgreSQL 14+
2. Install required extensions:

```sql
-- Connect to your database
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_tokenizer;
CREATE EXTENSION IF NOT EXISTS vchord_bm25;
```

3. Run the schema script:

```bash
psql -U postgres -d agent_mem -f agent_mem/sql/schema.sql
```

#### Neo4j Setup

1. Install Neo4j 5+ Community or Enterprise
2. Enable APOC plugin
3. Start Neo4j service

```bash
# Configure in neo4j.conf
dbms.security.procedures.unrestricted=apoc.*
```

#### Ollama Setup

1. Install Ollama from [ollama.ai](https://ollama.ai)
2. Start Ollama service
3. Pull the embedding model:

```bash
ollama pull nomic-embed-text
```

#### Python Package

```bash
pip install agent-mem
```

#### Configure Environment

Create `.env` file:

```env
# PostgreSQL
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password
POSTGRES_DB=agent_mem

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
EMBEDDING_MODEL=nomic-embed-text
VECTOR_DIMENSION=768

# Google Gemini
GEMINI_API_KEY=your_api_key_here
```

## Verify Installation

Create `test_install.py`:

```python
import asyncio
from agent_mem import AgentMem

async def test():
    agent_mem = AgentMem()
    await agent_mem.initialize()
    print("✓ Agent Mem initialized successfully!")
    await agent_mem.close()

asyncio.run(test())
```

Run it:

```bash
python test_install.py
```

If successful, you should see: `✓ Agent Mem initialized successfully!`

## Troubleshooting

### PostgreSQL Extensions

If you get errors about missing extensions:

```bash
# Install pgvector
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install

# Install other extensions as needed
```

### Neo4j APOC Plugin

If APOC procedures are not available:

1. Download APOC jar from [GitHub releases](https://github.com/neo4j-contrib/neo4j-apoc-procedures/releases)
2. Place in Neo4j plugins directory
3. Update neo4j.conf
4. Restart Neo4j

### Connection Issues

If you can't connect to services:

```bash
# Check service status
docker-compose ps

# View logs
docker-compose logs -f postgres
docker-compose logs -f neo4j
docker-compose logs -f ollama

# Restart services
docker-compose restart
```

## Next Steps

- **[Quick Start](quickstart.md)**: Run your first example
- **[Configuration](configuration.md)**: Learn about configuration options
- **[User Guide](../guide/overview.md)**: Deep dive into features
