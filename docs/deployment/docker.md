# Docker Deployment

This guide covers deploying Agent Mem using Docker and Docker Compose.

## Quick Start

### 1. Clone and Configure

```bash
git clone https://github.com/yourusername/agent-mem.git
cd agent-mem

# Copy environment template
cp .env.example .env

# Edit .env with your settings
nano .env
```

### 2. Start Services

```bash
# CPU-only deployment
docker-compose up -d

# Or with GPU support for Ollama
docker-compose --profile gpu up -d
```

### 3. Initialize

```bash
# Pull embedding model
docker exec -it agent_mem_ollama ollama pull nomic-embed-text

# Verify services
docker-compose ps
```

### 4. Install Python Package

```bash
pip install agent-mem
```

## Architecture

The Docker setup includes three services:

```
┌─────────────────┐
│   PostgreSQL    │  Port 5432
│   + pgvector    │  Data: postgres_data volume
│   + extensions  │
└─────────────────┘

┌─────────────────┐
│     Neo4j       │  Ports: 7687 (Bolt), 7474 (HTTP)
│   + APOC        │  Data: neo4j_data volume
└─────────────────┘

┌─────────────────┐
│     Ollama      │  Port 11434
│   + Models      │  Data: OLLAMA_HOST_VOLUME_PATH
└─────────────────┘
```

## Configuration

### Environment Variables

Key variables in `.env`:

```env
# PostgreSQL
POSTGRES_USER=agent_mem_user
POSTGRES_PASSWORD=change_this_password
POSTGRES_DB=agent_mem

# Neo4j
NEO4J_USER=neo4j
NEO4J_PASSWORD=change_this_password

# Ollama
OLLAMA_HOST_VOLUME_PATH=./ollama_data

# Models
EMBEDDING_MODEL=nomic-embed-text
GEMINI_API_KEY=your_api_key_here
```

### Volume Configuration

By default, data is stored in Docker volumes:

- `postgres_data` - PostgreSQL database
- `neo4j_data` - Neo4j graph data
- `neo4j_logs` - Neo4j logs
- `OLLAMA_HOST_VOLUME_PATH` - Ollama models (host path)

To use custom paths:

```yaml
# docker-compose.override.yml
services:
  postgres:
    volumes:
      - /your/postgres/path:/var/lib/postgresql/data
  
  neo4j:
    volumes:
      - /your/neo4j/data:/data
      - /your/neo4j/logs:/logs
```

### GPU Support

For GPU-accelerated Ollama:

1. Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
2. Start with GPU profile:
   ```bash
   docker-compose --profile gpu up -d
   ```

Without GPU:
```bash
docker-compose up -d
```

## Management

### Service Status

```bash
# Check all services
docker-compose ps

# View logs
docker-compose logs -f

# Specific service logs
docker-compose logs -f postgres
docker-compose logs -f neo4j
docker-compose logs -f ollama
```

### Restart Services

```bash
# Restart all
docker-compose restart

# Restart specific service
docker-compose restart postgres
```

### Stop Services

```bash
# Stop all services
docker-compose stop

# Stop and remove containers
docker-compose down

# Stop and remove volumes (CAUTION: data loss!)
docker-compose down -v
```

### Backup Data

```bash
# PostgreSQL backup
docker exec agent_mem_postgres pg_dump -U agent_mem_user agent_mem > backup.sql

# Neo4j backup
docker exec agent_mem_neo4j neo4j-admin database dump neo4j --to-path=/data/backups
```

### Restore Data

```bash
# PostgreSQL restore
docker exec -i agent_mem_postgres psql -U agent_mem_user agent_mem < backup.sql

# Neo4j restore
docker exec agent_mem_neo4j neo4j-admin database load neo4j --from-path=/data/backups
```

## Scaling

### PostgreSQL Connection Pool

Adjust in code:

```python
from agent_mem import AgentMem, Config

config = Config(
    postgres_pool_min_size=5,
    postgres_pool_max_size=20
)
```

### Neo4j Memory

In `docker-compose.yml` or `.env`:

```yaml
environment:
  NEO4J_dbms_memory_pagecache_size: 2G
  NEO4J_dbms_memory_heap_max__size: 4G
```

### Multiple Instances

For horizontal scaling, use a shared database:

```yaml
services:
  postgres:
    # Use external PostgreSQL cluster
    
  neo4j:
    # Use Neo4j cluster
```

## Security

### Production Checklist

- [ ] Change all default passwords
- [ ] Use strong, unique passwords
- [ ] Enable SSL/TLS for databases
- [ ] Restrict network access (firewall/security groups)
- [ ] Use secrets management (Vault, AWS Secrets Manager)
- [ ] Enable authentication for all services
- [ ] Regular security updates
- [ ] Monitor access logs

### SSL/TLS

PostgreSQL with SSL:

```yaml
postgres:
  environment:
    POSTGRES_SSL_MODE: require
  volumes:
    - ./certs/server.crt:/var/lib/postgresql/server.crt
    - ./certs/server.key:/var/lib/postgresql/server.key
```

Neo4j with SSL:

```yaml
neo4j:
  environment:
    NEO4J_dbms_ssl_policy_bolt_enabled: "true"
  volumes:
    - ./certs:/ssl
```

### Network Isolation

Create isolated network:

```yaml
networks:
  agent_mem_network:
    driver: bridge
    internal: true  # No external access
```

## Monitoring

### Health Checks

Services include health checks:

```bash
# View health status
docker-compose ps

# Check specific service
docker inspect agent_mem_postgres | grep Health
```

### Logs

Centralized logging:

```yaml
services:
  postgres:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

### Metrics

Connect Prometheus/Grafana:

```yaml
# Add to docker-compose.yml
services:
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
```

## Troubleshooting

### PostgreSQL Won't Start

```bash
# Check logs
docker-compose logs postgres

# Common issues:
# - Port 5432 in use
# - Permission issues with volume
# - Invalid configuration
```

### Neo4j Connection Failed

```bash
# Check logs
docker-compose logs neo4j

# Verify authentication
docker exec -it agent_mem_neo4j cypher-shell -u neo4j -p your_password

# Common issues:
# - Wrong password
# - APOC plugin not loaded
# - Memory settings too high
```

### Ollama Not Responding

```bash
# Check if running
curl http://localhost:11434

# Check logs
docker-compose logs ollama

# Pull model manually
docker exec -it agent_mem_ollama ollama pull nomic-embed-text

# Common issues:
# - Model not pulled
# - Insufficient disk space
# - GPU driver issues (if using GPU)
```

## Next Steps

- **[Production Guide](production.md)**: Deploy to production
- **[Configuration](../getting-started/configuration.md)**: Advanced configuration
- **[Monitoring](monitoring.md)**: Set up monitoring and alerts
