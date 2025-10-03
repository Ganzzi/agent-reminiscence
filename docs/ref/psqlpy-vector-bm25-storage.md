# PSQLPy Vector and BM25 Storage Guide

## Overview

This guide explains how to use PSQLPy for storing and searching vector embeddings and BM25 indexes in the AI Army application. The system uses PostgreSQL with the `pgvector` extension for vector similarity search and `vchord_bm25` extension for BM25 text search.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Database Extensions](#database-extensions)
3. [Vector Storage](#vector-storage)
4. [BM25 Storage](#bm25-storage)
5. [Hybrid Search](#hybrid-search)
6. [Best Practices](#best-practices)

## Prerequisites

### Required Extensions

The following PostgreSQL extensions must be installed:

```sql
-- Enable pgvector for vector similarity search
CREATE EXTENSION IF NOT EXISTS vector;

-- Enable pg_tokenizer for text tokenization
CREATE EXTENSION IF NOT EXISTS pg_tokenizer CASCADE;

-- Enable vchord_bm25 for BM25 ranking
CREATE EXTENSION IF NOT EXISTS vchord_bm25 CASCADE;

-- Additional extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS btree_gin;
CREATE EXTENSION IF NOT EXISTS btree_gist;
```

### Connection Setup

```python
from psqlpy import ConnectionPool, Connection
from psqlpy.extra_types import PgVector
import os
from dotenv import load_dotenv

load_dotenv()

class PostgreSQLManager:
    def __init__(self, host: str, port: int, username: str, password: str, database: str):
        self._pool = ConnectionPool(
            host=host,
            port=port,
            username=username,
            password=password,
            db_name=database,
            max_db_pool_size=10
        )
    
    @classmethod
    def from_env(cls):
        return cls(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", 5432)),
            username=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", ""),
            database=os.getenv("POSTGRES_DB", "ai_army")
        )
    
    async def get_connection(self) -> Connection:
        return await self._pool.connection()
```

## Vector Storage

### Table Schema

Create tables with vector columns using the `vector(dimensions)` type:

```sql
-- Short-term Memory Chunks with Vector Embeddings
CREATE TABLE IF NOT EXISTS shortterm_memory_chunk (
    id SERIAL PRIMARY KEY,
    shortterm_memory_id INTEGER NOT NULL REFERENCES shortterm_memory(id) ON DELETE CASCADE,
    chunk_order INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding vector(1024),  -- Vector dimension from config
    metadata JSONB DEFAULT '{}'
);

-- Long-term Memory Chunks with Vector Embeddings
CREATE TABLE IF NOT EXISTS longterm_memory_chunk (
    id SERIAL PRIMARY KEY,
    longterm_memory_id INTEGER NOT NULL REFERENCES longterm_memory(id) ON DELETE CASCADE,
    chunk_order INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding vector(1024),
    importance_score FLOAT DEFAULT 0.5,
    confidence_score FLOAT DEFAULT 0.5,
    valid_from TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    valid_until TIMESTAMP,
    metadata JSONB DEFAULT '{}'
);
```

### Creating Vector Indexes

For efficient similarity search, create indexes on vector columns:

```sql
-- IVFFlat index for approximate nearest neighbor search
CREATE INDEX IF NOT EXISTS shortterm_chunk_embedding_idx 
ON shortterm_memory_chunk 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

CREATE INDEX IF NOT EXISTS longterm_chunk_embedding_idx 
ON longterm_memory_chunk 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

### Storing Vectors with PSQLPy

```python
from psqlpy.extra_types import PgVector
from typing import List

async def create_chunk_with_embedding(
    memory_id: int,
    content: str,
    embedding: List[float],
    chunk_order: int = 0,
    metadata: dict = None
) -> int:
    """
    Create a memory chunk with vector embedding.
    
    Args:
        memory_id: Reference to parent memory
        content: Text content of the chunk
        embedding: Vector embedding as list of floats
        chunk_order: Order of chunk in sequence
        metadata: Additional metadata as JSON
    
    Returns:
        ID of created chunk
    """
    connection = await get_db_connection()
    
    # Convert Python list to PgVector
    pg_vector = PgVector(embedding)
    
    query = """
        INSERT INTO shortterm_memory_chunk 
        (shortterm_memory_id, chunk_order, content, embedding, metadata)
        VALUES ($1, $2, $3, $4, $5)
        RETURNING id
    """
    
    result = await connection.execute(
        query,
        [memory_id, chunk_order, content, pg_vector, metadata or {}]
    )
    
    chunk_id = result.result()[0][0]
    return chunk_id
```

### Vector Similarity Search

```python
async def search_by_vector_similarity(
    query_embedding: List[float],
    limit: int = 10,
    similarity_threshold: float = 0.7
) -> List[dict]:
    """
    Search for similar chunks using cosine similarity.
    
    Args:
        query_embedding: Query vector
        limit: Maximum number of results
        similarity_threshold: Minimum similarity score
    
    Returns:
        List of matching chunks with similarity scores
    """
    connection = await get_db_connection()
    pg_vector = PgVector(query_embedding)
    
    query = """
        SELECT 
            id,
            content,
            metadata,
            1 - (embedding <=> $1) AS similarity
        FROM shortterm_memory_chunk
        WHERE 1 - (embedding <=> $1) >= $2
        ORDER BY embedding <=> $1
        LIMIT $3
    """
    
    result = await connection.execute(
        query,
        [pg_vector, similarity_threshold, limit]
    )
    
    chunks = []
    for row in result.result():
        chunks.append({
            'id': row[0],
            'content': row[1],
            'metadata': row[2],
            'similarity': float(row[3])
        })
    
    return chunks
```

## BM25 Storage

### Table Schema with BM25

BM25 uses a special `bm25vector` type that is automatically generated via triggers:

```sql
-- Short-term Memory Chunks Table with bm25vector
CREATE TABLE IF NOT EXISTS shortterm_memory_chunk (
    id SERIAL PRIMARY KEY,
    shortterm_id INTEGER NOT NULL REFERENCES shortterm_memory (id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    content_vector vector(768),
    content_bm25 bm25vector,  -- Auto-populated by trigger
    metadata JSONB DEFAULT '{}'
);

-- Create tokenizer for BM25
SELECT create_tokenizer('bert', $$ model = "bert_base_uncased" $$);

-- Trigger to automatically populate bm25vector
CREATE OR REPLACE FUNCTION update_bm25_vector_memory_chunk()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' OR (TG_OP = 'UPDATE' AND OLD.content IS DISTINCT FROM NEW.content) THEN
        -- Automatically tokenize content into bm25vector
        NEW.content_bm25 = tokenize(NEW.content, 'bert');
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_bm25_shortterm_memory_chunk
    BEFORE INSERT OR UPDATE ON shortterm_memory_chunk
    FOR EACH ROW EXECUTE FUNCTION update_bm25_vector_memory_chunk();

-- Create BM25 index
CREATE INDEX IF NOT EXISTS idx_shortterm_memory_chunk_bm25 
ON shortterm_memory_chunk 
USING bm25 (content_bm25 bm25_ops);
```

**Important Notes:**
- The `content_bm25` column is of type `bm25vector`, not a regular column
- Content is automatically tokenized when inserted/updated via the trigger
- You don't manually populate `content_bm25` - the trigger handles it
- The tokenizer (e.g., 'bert') must be created before the trigger

### BM25 Search

BM25 search uses `to_bm25query` and `tokenize` functions to query the bm25vector column:

```python
async def search_by_bm25(
    query_text: str,
    shortterm_id: int,
    limit: int = 10,
    similarity_threshold: float = 0.0
) -> List[dict]:
    """
    Search for chunks using BM25 text ranking.
    
    Args:
        query_text: Search query text
        shortterm_id: Shortterm memory ID to search within
        limit: Maximum number of results
        similarity_threshold: Minimum BM25 score threshold
    
    Returns:
        List of matching chunks with BM25 scores
    """
    connection = await get_db_connection()
    
    # BM25 search using to_bm25query and tokenize
    query = """
        SELECT 
            id,
            shortterm_id,
            content,
            metadata,
            content_bm25 <&> to_bm25query(
                'idx_shortterm_memory_chunk_bm25',
                tokenize($2, 'bert')
            ) as score
        FROM shortterm_memory_chunk
        WHERE shortterm_id = $1
          AND content_bm25 IS NOT NULL
          AND (content_bm25 <&> to_bm25query(
              'idx_shortterm_memory_chunk_bm25',
              tokenize($2, 'bert')
          )) >= $3
        ORDER BY score DESC
        LIMIT $4
    """
    
    result = await connection.execute(
        query,
        [shortterm_id, query_text, similarity_threshold, limit]
    )
    
    chunks = []
    for row in result.result():
        chunks.append({
            'id': row[0],
            'shortterm_id': row[1],
            'content': row[2],
            'metadata': row[3],
            'bm25_score': float(row[4])
        })
    
    return chunks
```

**Key Points:**
- Use `to_bm25query(index_name, tokenized_query)` to create a BM25 query
- Use `tokenize(text, tokenizer_name)` to tokenize the search text
- The `<&>` operator performs BM25 similarity search
- Index name must match the BM25 index created earlier
- Tokenizer name must match the tokenizer used in the trigger ('bert' in this case)

## Hybrid Search

Combine vector similarity and BM25 for better search results:

```python
async def hybrid_search(
    query_text: str,
    query_embedding: List[float],
    limit: int = 10,
    vector_weight: float = 0.7,
    bm25_weight: float = 0.3
) -> List[dict]:
    """
    Perform hybrid search combining vector similarity and BM25.
    
    Args:
        query_text: Search query text
        query_embedding: Query vector embedding
        limit: Maximum number of results
        vector_weight: Weight for vector similarity (0-1)
        bm25_weight: Weight for BM25 score (0-1)
    
    Returns:
        List of chunks ranked by combined score
    """
    connection = await get_db_connection()
    pg_vector = PgVector(query_embedding)
    
    query = """
        WITH vector_search AS (
            SELECT 
                id,
                content,
                metadata,
                1 - (embedding <=> $1) AS vector_similarity
            FROM shortterm_memory_chunk
            WHERE embedding IS NOT NULL
        ),
        bm25_search AS (
            SELECT 
                id,
                content,
                metadata,
                bm25_score
            FROM shortterm_memory_chunk
            WHERE content @@@ $2
        ),
        max_scores AS (
            SELECT 
                MAX(vector_similarity) as max_vector,
                MAX(bm25_score) as max_bm25
            FROM vector_search, bm25_search
        )
        SELECT 
            COALESCE(v.id, b.id) as id,
            COALESCE(v.content, b.content) as content,
            COALESCE(v.metadata, b.metadata) as metadata,
            (
                ($3 * COALESCE(v.vector_similarity, 0) / NULLIF(m.max_vector, 0)) +
                ($4 * COALESCE(b.bm25_score, 0) / NULLIF(m.max_bm25, 0))
            ) AS combined_score
        FROM vector_search v
        FULL OUTER JOIN bm25_search b ON v.id = b.id
        CROSS JOIN max_scores m
        ORDER BY combined_score DESC
        LIMIT $5
    """
    
    result = await connection.execute(
        query,
        [pg_vector, query_text, vector_weight, bm25_weight, limit]
    )
    
    chunks = []
    for row in result.result():
        chunks.append({
            'id': row[0],
            'content': row[1],
            'metadata': row[2],
            'combined_score': float(row[3])
        })
    
    return chunks
```

### Real-World Implementation Example

From the codebase (`longterm_memory_repository.py`):

```python
async def search_chunks(
    self,
    query_vector: Optional[List[float]] = None,
    query_text: Optional[str] = None,
    limit: int = 10,
    similarity_threshold: float = 0.0,
) -> List[Tuple[LongtermMemoryChunk, float, float]]:
    """Search longterm memory chunks using hybrid search."""
    
    connection = await get_db_connection()
    
    if query_vector and query_text:
        # Hybrid search with both vector and BM25
        pg_vector = PgVector(query_vector)
        
        query = """
            WITH vector_results AS (
                SELECT 
                    lmc.*,
                    1 - (lmc.embedding <=> $1) AS vector_similarity,
                    0.0 AS bm25_score
                FROM longterm_memory_chunk lmc
                WHERE lmc.embedding IS NOT NULL
                  AND (lmc.valid_until IS NULL OR lmc.valid_until > CURRENT_TIMESTAMP)
            ),
            bm25_results AS (
                SELECT 
                    lmc.*,
                    0.0 AS vector_similarity,
                    lmc.bm25_score AS bm25_score
                FROM longterm_memory_chunk lmc
                WHERE lmc.content @@@ $2
                  AND (lmc.valid_until IS NULL OR lmc.valid_until > CURRENT_TIMESTAMP)
            ),
            combined AS (
                SELECT * FROM vector_results
                UNION
                SELECT * FROM bm25_results
            )
            SELECT 
                id, longterm_memory_id, chunk_order, content,
                importance_score, confidence_score, valid_from, valid_until,
                metadata, vector_similarity, bm25_score
            FROM combined
            WHERE vector_similarity >= $3 OR bm25_score > 0
            ORDER BY (vector_similarity + bm25_score) DESC
            LIMIT $4
        """
        
        result = await connection.execute(
            query,
            [pg_vector, query_text, similarity_threshold, limit]
        )
    
    elif query_vector:
        # Vector-only search
        pg_vector = PgVector(query_vector)
        
        query = """
            SELECT 
                id, longterm_memory_id, chunk_order, content,
                importance_score, confidence_score, valid_from, valid_until,
                metadata,
                1 - (embedding <=> $1) AS vector_similarity,
                0.0 AS bm25_score
            FROM longterm_memory_chunk
            WHERE embedding IS NOT NULL
              AND 1 - (embedding <=> $1) >= $2
              AND (valid_until IS NULL OR valid_until > CURRENT_TIMESTAMP)
            ORDER BY embedding <=> $1
            LIMIT $3
        """
        
        result = await connection.execute(
            query,
            [pg_vector, similarity_threshold, limit]
        )
    
    # Parse results into chunk objects
    chunks = []
    for row in result.result():
        chunk = LongtermMemoryChunk(
            id=row[0],
            longterm_memory_id=row[1],
            chunk_order=row[2],
            content=row[3],
            importance_score=row[4],
            confidence_score=row[5],
            valid_from=row[6],
            valid_until=row[7],
            metadata=row[8]
        )
        vector_sim = float(row[9])
        bm25_score = float(row[10])
        chunks.append((chunk, vector_sim, bm25_score))
    
    return chunks
```

## Best Practices

### 1. Vector Dimension Consistency

Always use consistent vector dimensions across your application:

```python
from config.constants import VECTOR_DIMENSION

# Ensure embeddings match configured dimension
assert len(embedding) == VECTOR_DIMENSION
```

### 2. Batch Operations

For inserting multiple chunks, use batch operations:

```python
async def batch_create_chunks(chunks_data: List[dict]) -> List[int]:
    """Create multiple chunks in a single transaction."""
    connection = await get_db_connection()
    
    query = """
        INSERT INTO shortterm_memory_chunk 
        (shortterm_memory_id, chunk_order, content, embedding, metadata)
        VALUES ($1, $2, $3, $4, $5)
        RETURNING id
    """
    
    chunk_ids = []
    async with connection.transaction():
        for chunk in chunks_data:
            result = await connection.execute(
                query,
                [
                    chunk['memory_id'],
                    chunk['chunk_order'],
                    chunk['content'],
                    PgVector(chunk['embedding']),
                    chunk.get('metadata', {})
                ]
            )
            chunk_ids.append(result.result()[0][0])
    
    return chunk_ids
```

### 3. Index Maintenance

Rebuild indexes periodically for optimal performance:

```sql
-- Reindex vector indexes
REINDEX INDEX shortterm_chunk_embedding_idx;
REINDEX INDEX longterm_chunk_embedding_idx;

-- Reindex BM25 indexes
REINDEX INDEX shortterm_chunk_bm25_idx;
REINDEX INDEX longterm_chunk_bm25_idx;
```

### 4. Connection Pooling

Always use connection pooling for better performance:

```python
# Initialize pool once at application startup
db_manager = PostgreSQLManager.from_env()

# Reuse connections from pool
async def perform_search():
    connection = await db_manager.get_connection()
    # ... perform operations
```

### 5. Error Handling

Implement proper error handling for database operations:

```python
from psqlpy.exceptions import PSQLPyException

async def safe_vector_search(query_embedding: List[float]):
    try:
        connection = await get_db_connection()
        # ... perform search
        return results
    except PSQLPyException as e:
        logger.error(f"Database error during vector search: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return []
```

### 6. Query Optimization

- Use appropriate similarity thresholds to limit results
- Create indexes on frequently queried columns
- Use EXPLAIN ANALYZE to optimize slow queries
- Consider partitioning large tables by date or memory type

### 7. Vector Normalization

Normalize vectors before storage for consistent similarity calculations:

```python
import numpy as np

def normalize_vector(vector: List[float]) -> List[float]:
    """Normalize vector to unit length."""
    arr = np.array(vector)
    norm = np.linalg.norm(arr)
    if norm == 0:
        return vector
    return (arr / norm).tolist()

# Use normalized vectors
embedding = normalize_vector(raw_embedding)
```

## Summary

- Use `pgvector` extension for vector similarity search
- Use `vchord_bm25` extension for BM25 text search
- Combine both for hybrid search with weighted scoring
- Always use `PgVector` type when passing vectors to queries
- Leverage connection pooling and batch operations for performance
- Create appropriate indexes for your search patterns
- Handle errors gracefully and log issues for debugging

For more examples, refer to:
- `database/memories/shortterm_memory_repository.py`
- `database/memories/longterm_memory_repository.py`
- `database/sql_schema/memories.sql`
