# Configuration Guide

AgentMem supports three configuration methods: Direct Python Configuration, Environment Variables, and .env File Configuration.

## Configuration Patterns

### Pattern 1: Direct Python Configuration (Recommended for PyPI users)

```python
from agent_mem import AgentMem, Config

config = Config(
    postgres_host="localhost",
    postgres_password="secure_password",
    neo4j_uri="bolt://localhost:7687",
    neo4j_password="neo4j_password",
    ollama_base_url="http://localhost:11434",
    embedding_model="nomic-embed-text",
    vector_dimension=768
)

agent_mem = AgentMem(config=config)
```

### Pattern 2: Environment Variables (Recommended for Docker/K8s)

```bash
export POSTGRES_HOST=postgres
export POSTGRES_PASSWORD=secure_pass
export NEO4J_URI=bolt://neo4j:7687
export NEO4J_PASSWORD=neo4j_pass
export OLLAMA_BASE_URL=http://ollama:11434
python your_app.py
```

```python
from agent_mem import AgentMem

agent_mem = AgentMem()  # Uses environment variables
```

### Pattern 3: .env File Configuration (Convenient for local development)

Create a `.env` file in your project root:

```env
# PostgreSQL Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password
POSTGRES_DB=agent_mem

# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
EMBEDDING_MODEL=nomic-embed-text
VECTOR_DIMENSION=768

# Google Gemini API (for AI agents)
GOOGLE_API_KEY=your_api_key_here

# Agent Models (Optional)
MEMORY_UPDATE_AGENT_MODEL=google:gemini-2.0-flash
MEMORIZER_AGENT_MODEL=google:gemini-2.0-flash
MEMORY_RETRIEVE_AGENT_MODEL=google:gemini-2.0-flash
```

```python
# Automatically loaded if python-dotenv is available
from agent_mem import AgentMem

agent_mem = AgentMem()  # Uses .env file
```

**Note**: `python-dotenv` is optional. Install with `pip install agent-mem[dev]` for .env file support.

## Configuration Options

### Database Configuration

#### PostgreSQL

| Variable | Default | Description |
|----------|---------|-------------|
| `POSTGRES_HOST` | `localhost` | PostgreSQL server hostname |
| `POSTGRES_PORT` | `5432` | PostgreSQL server port |
| `POSTGRES_USER` | `postgres` | Database username |
| `POSTGRES_PASSWORD` | - | Database password (required) |
| `POSTGRES_DB` | `agent_mem` | Database name |

#### Neo4j

| Variable | Default | Description |
|----------|---------|-------------|
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection URI |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | - | Neo4j password (required) |
| `NEO4J_DATABASE` | `neo4j` | Neo4j database name |

### Embedding Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API base URL |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model name |
| `VECTOR_DIMENSION` | `768` | Embedding vector dimension |

### AI Agent Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | - | Google Gemini API key (required for agents) |
| `MEMORY_UPDATE_AGENT_MODEL` | `google-gla:gemini-2.0-flash` | Model for memory updates |
| `MEMORIZER_AGENT_MODEL` | `google-gla:gemini-2.0-flash` | Model for consolidation |
| `MEMORY_RETRIEVE_AGENT_MODEL` | `google-gla:gemini-2.0-flash` | Model for retrieval |

### Memory Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ACTIVE_MEMORY_UPDATE_THRESHOLD` | `5` | Updates before consolidation |
| `SHORTTERM_PROMOTION_THRESHOLD` | `0.7` | Importance score for promotion to longterm |

## Docker Configuration

Additional variables for `docker-compose.yml`:

```env
# Ollama data volume path on host
OLLAMA_HOST_VOLUME_PATH=./ollama_data

# GPU support (set to empty to disable)
ENABLE_GPU=true
```

## Production Considerations

### Security

1. **Never commit `.env` files** to version control
2. **Use strong passwords** for all services
3. **Restrict network access** to databases
4. **Use SSL/TLS connections** in production:

```env
# PostgreSQL with SSL
POSTGRES_HOST=production-db.example.com
POSTGRES_SSLMODE=require

# Neo4j with SSL
NEO4J_URI=neo4j+s://production-neo4j.example.com:7687
```

### Performance

```env
# Connection pool settings
POSTGRES_MIN_POOL_SIZE=5
POSTGRES_MAX_POOL_SIZE=20

# Neo4j memory settings (in docker-compose.yml)
NEO4J_dbms_memory_pagecache_size=2G
NEO4J_dbms_memory_heap_max__size=4G
```

### Monitoring

```env
# Enable detailed logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# Metrics endpoint (if implemented)
METRICS_ENABLED=true
METRICS_PORT=9090
```

## Testing Configuration

For running tests, create a separate `.env.test`:

```env
POSTGRES_DB=agent_mem_test
NEO4J_DATABASE=neo4j_test
```

Load it in tests:

```python
from dotenv import load_dotenv
load_dotenv('.env.test')
```

## Next Steps

- **[Quick Start](quickstart.md)**: Run your first example
- **[User Guide](../guide/overview.md)**: Learn about features
- **[Deployment](../deployment/production.md)**: Production setup guide
