# PSQLPy Vector and BM25 Storage Guide

> **Note:** For a complete guide to PSQLPy including connection management, configuration, and best practices, see [PSQLPy Complete Guide](./psqlpy-complete-guide.md).

## Overview

This guide focuses specifically on using PSQLPy for vector embeddings (pgvector) and BM25 full-text search in the agent_reminiscence package. It covers practical patterns directly from the codebase.

## Table of Contents

1. [Vector Storage with pgvector](#1-vector-storage-with-pgvector)
2. [BM25 Full-Text Search](#2-bm25-full-text-search)
3. [Hybrid Search Implementation](#3-hybrid-search-implementation)
4. [Real-World Examples from Codebase](#4-real-world-examples-from-codebase)
5. [Best Practices and Common Errors](#5-best-practices-and-common-errors)

---

## 1. Vector Storage with pgvector

### Table Schema

From `agent_reminiscence/sql/schema.sql`:

```sql
-- Shortterm memory chunks with vector embeddings
CREATE TABLE IF NOT EXISTS shortterm_memory_chunk (
    id SERIAL PRIMARY KEY,
    shortterm_memory_id INTEGER NOT NULL REFERENCES shortterm_memory(id) ON DELETE CASCADE,
    external_id TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding vector(768),  -- Dimension from config.vector_dimension
    content_bm25 bm25vector,  -- Auto-populated by trigger
    section_id TEXT,
    metadata JSONB DEFAULT '{}',
    access_count INTEGER DEFAULT 0,
    last_access TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Vector similarity index (IVFFlat for approximate nearest neighbor)
CREATE INDEX IF NOT EXISTS idx_shortterm_chunk_embedding
ON shortterm_memory_chunk
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

### Storing Vectors - Code from `shortterm_memory.py`

The key is using `PgVector` from `psqlpy.extra_types`:

```python
from psqlpy.extra_types import PgVector
from typing import List, Optional, Dict, Any

async def create_chunk(
    self,
    shortterm_memory_id: int,
    external_id: str,
    content: str,
    embedding: Optional[List[float]] = None,
    section_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> ShorttermMemoryChunk:
    """
    Create a new chunk for a shortterm memory.
    
    The BM25 vector is auto-populated by database trigger.
    """
    query = """
        INSERT INTO shortterm_memory_chunk 
        (shortterm_memory_id, external_id, content, embedding, section_id, metadata, access_count, last_access)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        RETURNING id, shortterm_memory_id, content, section_id, metadata, access_count, last_access
    """

    # CRITICAL: Convert Python list to PgVector
    pg_vector = None
    if embedding:
        pg_vector = PgVector(embedding)

    async with self.postgres.connection() as conn:
        result = await conn.execute(
            query,
            [
                shortterm_memory_id,
                external_id,
                content,
                pg_vector,  # Use PgVector, NOT raw list
                section_id,
                metadata or {},
                0,  # access_count
                None,  # last_access
            ],
        )

        row = result.result()[0]
        chunk = self._chunk_row_to_model(row)
        
        logger.info(f"Created shortterm chunk {chunk.id} for memory {shortterm_memory_id}")
        return chunk
```

**Key Points:**
- ✅ Always convert `List[float]` to `PgVector(embedding)` before passing to query
- ✅ BM25 vector is auto-populated by trigger - don't include it in INSERT
- ✅ Use parameterized queries with `$1, $2, $3...` placeholders (PostgreSQL style)
- ❌ Don't use `%s` (psycopg2 style) or `?` (SQLite style)

### Vector Similarity Search

From `shortterm_memory.py` - using cosine distance:

```python
async def vector_search(
    self,
    external_id: str,
    query_embedding: List[float],
    limit: int = 10,
    min_similarity: float = 0.7,
) -> List[ShorttermMemoryChunk]:
    """
    Search using vector similarity only.
    """
    pg_vector = PgVector(query_embedding)

    query = """
        SELECT 
            id, shortterm_memory_id, external_id, content, 
            section_id, metadata, access_count, last_access, created_at,
            1 - (embedding <=> $1) AS similarity_score
        FROM shortterm_memory_chunk
        WHERE external_id = $2
          AND embedding IS NOT NULL
          AND 1 - (embedding <=> $1) >= $3
        ORDER BY embedding <=> $1
        LIMIT $4
    """

    async with self.postgres.connection() as conn:
        result = await conn.execute(
            query,
            [pg_vector, external_id, min_similarity, limit]
        )
        rows = result.result()
        
        chunks = []
        for row in rows:
            chunk = self._chunk_row_to_model(row[:9])  # First 9 columns
            chunk.similarity_score = float(row[9])      # 10th column
            chunks.append(chunk)
        
        return chunks
```

**Understanding PostgreSQL Vector Operators:**

- **`<=>` (Cosine Distance):** Most common for semantic search
  - Range: 0 (identical) to 2 (opposite)
  - Similarity score: `1 - (embedding <=> query)`
  - Use for normalized or unnormalized vectors

- **`<->` (L2/Euclidean Distance):** Physical distance in vector space
  - Smaller = more similar
  - Sensitive to vector magnitude

- **`<#>` (Inner Product):** Dot product
  - Larger = more similar (for normalized vectors)
  - Fastest operator but requires normalization

---

## 2. BM25 Full-Text Search

### Setting Up BM25 - From `schema.sql`

```sql
-- 1. Create tokenizer (must be done BEFORE trigger)
SELECT create_tokenizer('bert', $$ model = "bert_base_uncased" $$);

-- 2. Create trigger to auto-populate bm25vector column
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

-- 3. Create BM25 index (index name is important!)
CREATE INDEX IF NOT EXISTS idx_shortterm_chunk_bm25 
ON shortterm_memory_chunk 
USING bm25 (content_bm25 bm25_ops);
```

**Critical Setup Points:**
- 🔴 Tokenizer name ('bert') must match everywhere
- 🔴 Index name ('idx_shortterm_chunk_bm25') is used in queries
- 🔴 `content_bm25` is NEVER manually set - trigger handles it automatically
- 🔴 Only insert `content`, trigger populates `content_bm25`

### BM25 Search Query - From `shortterm_memory.py`

```python
async def bm25_search(
    self,
    external_id: str,
    query_text: str,
    limit: int = 10,
    min_score: float = 0.0,
) -> List[ShorttermMemoryChunk]:
    """
    Search using BM25 text ranking only.
    """
    query = """
        SELECT 
            id, shortterm_memory_id, external_id, content, 
            section_id, metadata, access_count, last_access, created_at,
            content_bm25 <&> to_bm25query('idx_shortterm_chunk_bm25', tokenize($1, 'bert')) AS bm25_score
        FROM shortterm_memory_chunk
        WHERE external_id = $2
          AND content_bm25 IS NOT NULL
          AND content_bm25 <&> to_bm25query('idx_shortterm_chunk_bm25', tokenize($1, 'bert')) >= $3
        ORDER BY bm25_score DESC
        LIMIT $4
    """

    async with self.postgres.connection() as conn:
        result = await conn.execute(
            query,
            [query_text, external_id, min_score, limit]
        )
        rows = result.result()
        
        chunks = []
        for row in rows:
            chunk = self._chunk_row_to_model(row[:9])
            chunk.bm25_score = float(row[9])
            chunks.append(chunk)
        
        return chunks
```

**BM25 Query Components Explained:**

1. **`tokenize($1, 'bert')`**
   - Tokenizes the query text using 'bert' tokenizer
   - Must match tokenizer created in setup

2. **`to_bm25query('idx_shortterm_chunk_bm25', tokenized_query)`**
   - Creates BM25 query object from tokens
   - Index name MUST match the BM25 index exactly

3. **`content_bm25 <&> bm25_query`**
   - BM25 similarity operator
   - Returns relevance score (higher = more relevant)

4. **Order by score DESC**
   - Most relevant results first

---

## 3. Hybrid Search Implementation

### Complete Hybrid Search - From `shortterm_memory.py`

This combines semantic understanding (vectors) with keyword matching (BM25):

```python
async def hybrid_search(
    self,
    external_id: str,
    query_text: str,
    query_embedding: List[float],
    limit: int = 10,
    vector_weight: float = 0.5,
    bm25_weight: float = 0.5,
    shortterm_memory_id: Optional[int] = None,
    min_similarity_score: Optional[float] = None,
    min_bm25_score: Optional[float] = None,
) -> List[ShorttermMemoryChunk]:
    """
    Hybrid search combining vector similarity and BM25.
    
    RECOMMENDED: This provides the best search results by combining:
    - Semantic understanding (vector similarity)
    - Keyword matching (BM25 text search)
    """
    pg_vector = PgVector(query_embedding)

    query = """
        WITH vector_results AS (
            SELECT 
                id,
                1 - (embedding <=> $1) AS vector_score
            FROM shortterm_memory_chunk
            WHERE external_id = $2 AND embedding IS NOT NULL
            AND ($7::int IS NULL OR shortterm_memory_id = $7)
        ),
        bm25_results AS (
            SELECT 
                id,
                content_bm25 <&> to_bm25query('idx_shortterm_chunk_bm25', tokenize($3, 'bert')) AS bm25_score
            FROM shortterm_memory_chunk
            WHERE external_id = $2 AND content_bm25 IS NOT NULL
            AND ($7::int IS NULL OR shortterm_memory_id = $7)
        )
        SELECT 
            c.id, c.shortterm_memory_id, c.external_id, c.content, 
            c.section_id, c.metadata, c.access_count, c.last_access, c.created_at,
            COALESCE(v.vector_score, 0) * $4 + COALESCE(b.bm25_score, 0) * $5 AS combined_score,
            COALESCE(v.vector_score, 0) AS vector_score,
            COALESCE(b.bm25_score, 0) AS bm25_score
        FROM shortterm_memory_chunk c
        LEFT JOIN vector_results v ON c.id = v.id
        LEFT JOIN bm25_results b ON c.id = b.id
        WHERE c.external_id = $2
          AND ($7::int IS NULL OR c.shortterm_memory_id = $7)
          AND (v.vector_score IS NOT NULL OR b.bm25_score IS NOT NULL)
          AND ($8::float IS NULL OR COALESCE(v.vector_score, 0) >= $8)
          AND ($9::float IS NULL OR COALESCE(b.bm25_score, 0) >= $9)
        ORDER BY combined_score DESC
        LIMIT $6
    """

    async with self.postgres.connection() as conn:
        result = await conn.execute(
            query,
            [
                pg_vector,              # $1
                external_id,            # $2
                query_text,             # $3
                vector_weight,          # $4
                bm25_weight,            # $5
                limit,                  # $6
                shortterm_memory_id,    # $7
                min_similarity_score,   # $8
                min_bm25_score,         # $9
            ],
        )
        rows = result.result()

        chunks = []
        for row in rows:
            chunk = self._chunk_row_to_model(row[:9])
            chunk.similarity_score = float(row[9])  # combined_score
            chunk.bm25_score = float(row[11]) if row[11] is not None else None
            chunks.append(chunk)

        logger.debug(f"Hybrid search found {len(chunks)} chunks for {external_id}")
        return chunks
```

### Query Structure Breakdown

1. **vector_results CTE:** Calculate cosine similarity for all matching chunks
2. **bm25_results CTE:** Calculate BM25 scores for all matching chunks
3. **Main Query:** JOIN results and compute weighted combined score
4. **Scoring Formula:** 
   ```
   combined_score = (vector_score × vector_weight) + (bm25_score × bm25_weight)
   ```

### Weight Tuning for Different Use Cases

```python
# Semantic-focused: Emphasize meaning over keywords
results = await repo.hybrid_search(
    external_id, query_text, query_embedding,
    vector_weight=0.8,  # 80% semantic similarity
    bm25_weight=0.2     # 20% keyword matching
)

# Keyword-focused: Emphasize exact matches
results = await repo.hybrid_search(
    external_id, query_text, query_embedding,
    vector_weight=0.3,  # 30% semantic similarity
    bm25_weight=0.7     # 70% keyword matching
)

# Balanced: Equal importance
results = await repo.hybrid_search(
    external_id, query_text, query_embedding,
    vector_weight=0.5,  # 50% semantic similarity
    bm25_weight=0.5     # 50% keyword matching
)
```

---

## 4. Real-World Examples from Codebase

### Example 1: Complete Workflow - Creating Memory with Search

```python
from agent_reminiscence.database.postgres_manager import PostgreSQLManager
from agent_reminiscence.database.neo4j_manager import Neo4jManager
from agent_reminiscence.database.repositories.shortterm_memory import ShorttermMemoryRepository
from agent_reminiscence.config import get_config

# Initialize (once at startup)
config = get_config()
postgres_manager = PostgreSQLManager(config)
neo4j_manager = Neo4jManager(config)

await postgres_manager.initialize()
await neo4j_manager.initialize()

# Create repository
repo = ShorttermMemoryRepository(postgres_manager, neo4j_manager)

# 1. Create a memory
memory = await repo.create_memory(
    external_id="agent-123",
    title="Python Best Practices",
    summary="Collection of Python coding standards and patterns",
    metadata={"category": "programming", "language": "python"}
)

# 2. Add chunks with embeddings
chunk1 = await repo.create_chunk(
    shortterm_memory_id=memory.id,
    external_id="agent-123",
    content="Always use type hints in function signatures for better code clarity",
    embedding=get_embedding("type hints python"),  # Your embedding function
    section_id="section-1",
    metadata={"importance": 0.9}
)

chunk2 = await repo.create_chunk(
    shortterm_memory_id=memory.id,
    external_id="agent-123",
    content="Use list comprehensions for simple transformations instead of loops",
    embedding=get_embedding("list comprehensions"),
    section_id="section-2",
    metadata={"importance": 0.8}
)

# 3. Search for relevant memories
query = "How to write clean Python code?"
query_embedding = get_embedding(query)

results = await repo.hybrid_search(
    external_id="agent-123",
    query_text=query,
    query_embedding=query_embedding,
    limit=5,
    vector_weight=0.7,  # Emphasize semantic meaning
    bm25_weight=0.3,
    min_similarity_score=0.5,  # Filter low-relevance
    min_bm25_score=0.1
)

# 4. Process results
for chunk in results:
    print(f"Combined Score: {chunk.similarity_score:.2f}")
    print(f"Vector Score: {chunk.similarity_score:.2f}")
    print(f"BM25 Score: {chunk.bm25_score:.2f}")
    print(f"Content: {chunk.content}")
    print("---")
```

### Example 2: Using PostgreSQLManager Utilities

```python
# Verify database connection
is_connected = await postgres_manager.verify_connection()
if not is_connected:
    logger.error("Database connection failed!")
    raise ConnectionError("Cannot connect to PostgreSQL")

# Ensure required extensions are installed
await postgres_manager.ensure_extensions()

# Execute custom query using manager
result = await postgres_manager.execute(
    "SELECT COUNT(*) FROM shortterm_memory_chunk WHERE external_id = $1",
    ["agent-123"]
)
count = result.result()[0][0]
print(f"Total chunks for agent-123: {count}")

# Get all memories for an agent
result = await postgres_manager.execute(
    """
    SELECT id, title, summary, created_at 
    FROM shortterm_memory 
    WHERE external_id = $1 
    ORDER BY created_at DESC 
    LIMIT $2
    """,
    ["agent-123", 10]
)

for row in result.result():
    print(f"Memory {row[0]}: {row[1]} - {row[2]}")
```

### Example 3: Batch Insert with Transaction

```python
async def batch_create_chunks(
    repo: ShorttermMemoryRepository,
    memory_id: int,
    external_id: str,
    chunks_data: List[Dict[str, Any]]
) -> List[ShorttermMemoryChunk]:
    """
    Create multiple chunks in a single transaction.
    
    All-or-nothing: If any insert fails, all are rolled back.
    """
    created_chunks = []
    
    async with repo.postgres.connection() as conn:
        # Start transaction
        await conn.execute("BEGIN")
        
        try:
            for chunk_data in chunks_data:
                chunk = await repo.create_chunk(
                    shortterm_memory_id=memory_id,
                    external_id=external_id,
                    content=chunk_data["content"],
                    embedding=chunk_data["embedding"],
                    section_id=chunk_data.get("section_id"),
                    metadata=chunk_data.get("metadata", {})
                )
                created_chunks.append(chunk)
            
            # Commit if all successful
            await conn.execute("COMMIT")
            logger.info(f"✅ Created {len(created_chunks)} chunks in transaction")
            return created_chunks
            
        except Exception as e:
            # Rollback on any error
            await conn.execute("ROLLBACK")
            logger.error(f"❌ Batch insert failed, rolled back: {e}")
            raise


# Usage
chunks_to_create = [
    {
        "content": "Python supports duck typing",
        "embedding": get_embedding("duck typing"),
        "section_id": "types-1"
    },
    {
        "content": "Use context managers for resource management",
        "embedding": get_embedding("context managers"),
        "section_id": "resources-1"
    }
]

created = await batch_create_chunks(repo, memory.id, "agent-123", chunks_to_create)
```

### Example 4: Result Processing Helper - From `shortterm_memory.py`

```python
def _chunk_row_to_model(self, row) -> ShorttermMemoryChunk:
    """
    Convert database row tuple to ShorttermMemoryChunk model.
    
    This pattern is used throughout the repository to convert
    PSQLPy results into Pydantic models.
    """
    return ShorttermMemoryChunk(
        id=row[0],
        shortterm_memory_id=row[1],
        external_id=row[2],
        content=row[3],
        section_id=row[4] if len(row) > 4 else None,
        metadata=row[5] if len(row) > 5 else {},
        access_count=row[6] if len(row) > 6 else 0,
        last_access=row[7] if len(row) > 7 else None,
        created_at=row[8] if len(row) > 8 else None,
    )

# Usage in query methods
async with self.postgres.connection() as conn:
    result = await conn.execute(query, params)
    rows = result.result()
    
    chunks = [self._chunk_row_to_model(row) for row in rows]
    return chunks
```

---

## 5. Best Practices and Common Errors

### ✅ Best Practices

#### Vector Operations
1. **Always use PgVector**
   ```python
   # ✅ CORRECT
   pg_vector = PgVector(embedding)
   await conn.execute(query, [pg_vector])
   
   # ❌ WRONG
   await conn.execute(query, [embedding])  # Will fail!
   ```

2. **Normalize vectors for consistent similarity**
   ```python
   import numpy as np
   
   def normalize_vector(vector: List[float]) -> List[float]:
       arr = np.array(vector)
       norm = np.linalg.norm(arr)
       if norm == 0:
           return vector
       return (arr / norm).tolist()
   
   # Use normalized
   embedding = normalize_vector(raw_embedding)
   pg_vector = PgVector(embedding)
   ```

3. **Create appropriate indexes**
   ```sql
   -- For datasets < 100K rows
   CREATE INDEX ON table USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
   
   -- For datasets > 100K rows
   CREATE INDEX ON table USING ivfflat (embedding vector_cosine_ops) WITH (lists = 1000);
   ```

4. **Filter NULL embeddings**
   ```sql
   WHERE embedding IS NOT NULL
   ```

#### BM25 Operations
1. **Never manually set bm25vector columns**
   ```python
   # ✅ CORRECT - trigger handles it
   await conn.execute(
       "INSERT INTO table (content) VALUES ($1)",
       [content]
   )
   
   # ❌ WRONG - don't insert content_bm25
   await conn.execute(
       "INSERT INTO table (content, content_bm25) VALUES ($1, $2)",
       [content, bm25_value]
   )
   ```

2. **Use exact index name in queries**
   ```python
   # Must match CREATE INDEX statement exactly
   to_bm25query('idx_shortterm_chunk_bm25', tokenize($1, 'bert'))
   ```

3. **Match tokenizer name everywhere**
   ```python
   # Trigger uses 'bert'
   NEW.content_bm25 = tokenize(NEW.content, 'bert');
   
   # Query must also use 'bert'
   tokenize($1, 'bert')
   ```

#### Connection Management
1. **Initialize once, reuse everywhere**
   ```python
   # ✅ At application startup
   manager = PostgreSQLManager(config)
   await manager.initialize()
   
   # ❌ Don't create new manager per request
   ```

2. **Always use context managers**
   ```python
   # ✅ CORRECT - connection returned automatically
   async with manager.connection() as conn:
       result = await conn.execute(query, params)
   
   # ❌ WRONG - manual management risks leaks
   conn = await manager.get_connection()
   result = await conn.execute(query, params)
   ```

3. **Use transactions for related operations**
   ```python
   async with manager.connection() as conn:
       await conn.execute("BEGIN")
       try:
           await conn.execute(query1, params1)
           await conn.execute(query2, params2)
           await conn.execute("COMMIT")
       except Exception:
           await conn.execute("ROLLBACK")
           raise
   ```

### ❌ Common Errors and Solutions

#### Error 1: "dimension of vector doesn't match"
```python
# Problem: Embedding dimension doesn't match table schema
embedding = [0.1, 0.2, 0.3]  # Only 3 dimensions
# Table expects vector(768)

# Solution: Verify dimension matches config
config = get_config()
assert len(embedding) == config.vector_dimension
pg_vector = PgVector(embedding)
```

#### Error 2: "column content_bm25 does not exist"
```python
# Problem: Trying to manually insert bm25vector

# Wrong:
await conn.execute(
    "INSERT INTO table (content, content_bm25) VALUES ($1, $2)",
    [content, bm25_value]
)

# Correct: Only insert content, trigger handles bm25
await conn.execute(
    "INSERT INTO table (content) VALUES ($1)",
    [content]
)
```

#### Error 3: "index 'idx_name' does not exist"
```python
# Problem: Wrong index name in to_bm25query()

# Check CREATE INDEX statement:
# CREATE INDEX idx_shortterm_chunk_bm25 ON ...

# Use exact name:
to_bm25query('idx_shortterm_chunk_bm25', tokenize($1, 'bert'))
```

#### Error 4: "tokenizer 'bert' does not exist"
```python
# Problem: Tokenizer not created before trigger

# Solution: Run this SQL first (in init_databases.py):
await conn.execute(
    "SELECT create_tokenizer('bert', $$ model = \"bert_base_uncased\" $$)"
)
```

#### Error 5: "list index out of range"
```python
# Problem: Assuming results exist

# Wrong:
result = await conn.execute(query, params)
user = result.result()[0]  # Crashes if no rows!

# Correct: Check for empty results
result = await conn.execute(query, params)
rows = result.result()
if rows:
    user = rows[0]
else:
    user = None
```

#### Error 6: "PSQLPy manager not initialized"
```python
# Problem: Using manager before calling initialize()

# Wrong:
manager = PostgreSQLManager(config)
async with manager.connection() as conn:  # RuntimeError!
    ...

# Correct:
manager = PostgreSQLManager(config)
await manager.initialize()  # Must call first!
async with manager.connection() as conn:
    ...
```

#### Error 7: Wrong placeholder style
```python
# ❌ WRONG: psycopg2 style
await conn.execute("SELECT * FROM users WHERE id = %s", [user_id])

# ❌ WRONG: SQLite style
await conn.execute("SELECT * FROM users WHERE id = ?", [user_id])

# ✅ CORRECT: PostgreSQL style
await conn.execute("SELECT * FROM users WHERE id = $1", [user_id])
```

---

## Summary

### Key Takeaways

1. **PSQLPy Basics**
   - Use `$1, $2, $3...` for parameter placeholders
   - Always use connection context managers
   - Initialize manager once at startup

2. **Vector Operations**
   - Convert to `PgVector` before passing to queries
   - Use cosine distance (`<=>`) for semantic search
   - Calculate similarity as `1 - (embedding <=> query)`

3. **BM25 Operations**
   - Never manually set `bm25vector` columns
   - Use exact index name in `to_bm25query()`
   - Match tokenizer name in trigger and queries

4. **Hybrid Search**
   - Combines semantic (vector) + keyword (BM25) search
   - Tune weights based on use case
   - Set minimum thresholds to filter noise

5. **Connection Management**
   - One manager instance per application
   - Use context managers for automatic cleanup
   - Use transactions for atomic operations

---

## References

- **Complete PSQLPy Guide:** [psqlpy-complete-guide.md](./psqlpy-complete-guide.md) - Full guide with setup, configuration, and advanced topics
- **Source Code:**
  - `agent_reminiscence/database/postgres_manager.py` - Manager implementation
  - `agent_reminiscence/database/repositories/shortterm_memory.py` - Repository with all search methods
  - `agent_reminiscence/database/init_databases.py` - Database initialization script
  - `agent_reminiscence/sql/schema.sql` - Complete database schema with triggers
  - `agent_reminiscence/config/settings.py` - Configuration management

- **PostgreSQL Extensions:**
  - [pgvector GitHub](https://github.com/pgvector/pgvector) - Vector similarity search
  - [vchord GitHub](https://github.com/tensorchord/vchord) - BM25 and vector search# PSQLPy Complete Usage Guide



> **Note:** For a complete guide to PSQLPy including connection management, configuration, and best practices, see [PSQLPy Complete Guide](./psqlpy-complete-guide.md).## Overview



## OverviewThis is a comprehensive guide for using **PSQLPy** - a high-performance PostgreSQL driver for Python with async support. PSQLPy is NOT the same as psycopg2 or asyncpg. This guide covers everything from basic setup to advanced usage patterns including vector embeddings and BM25 full-text search.



This guide focuses specifically on using PSQLPy for vector embeddings (pgvector) and BM25 full-text search in the agent_reminiscence package. It covers practical patterns used in the codebase.**Key Features of PSQLPy:**

- Fully async/await support (asyncio)

## Table of Contents- High-performance connection pooling

- Native support for PostgreSQL extensions (pgvector, vchord_bm25)

1. [Vector Storage with pgvector](#vector-storage-with-pgvector)- Type-safe parameter binding with `$1, $2` placeholders

2. [BM25 Full-Text Search](#bm25-full-text-search)- Special types like `PgVector` for vector operations

3. [Hybrid Search Implementation](#hybrid-search-implementation)- No ORM overhead - direct SQL execution

4. [Real-World Examples from Codebase](#real-world-examples-from-codebase)

## Table of Contents

---

1. [Installation and Setup](#installation-and-setup)

## Vector Storage with pgvector2. [Configuration Management](#configuration-management)

3. [Connection Pool Architecture](#connection-pool-architecture)

### Table Schema4. [Database Initialization](#database-initialization)

5. [Connection Management](#connection-management)

```sql6. [Query Execution Patterns](#query-execution-patterns)

-- Shortterm memory chunks with vector embeddings7. [Result Handling](#result-handling)

CREATE TABLE IF NOT EXISTS shortterm_memory_chunk (8. [Exception Handling](#exception-handling)

    id SERIAL PRIMARY KEY,9. [Vector Operations (pgvector)](#vector-operations-pgvector)

    shortterm_memory_id INTEGER NOT NULL REFERENCES shortterm_memory(id) ON DELETE CASCADE,10. [BM25 Full-Text Search](#bm25-full-text-search)

    external_id TEXT NOT NULL,11. [Hybrid Search](#hybrid-search)

    content TEXT NOT NULL,12. [Best Practices](#best-practices)

    embedding vector(768),  -- Dimension from config13. [Common Pitfalls](#common-pitfalls)

    content_bm25 bm25vector,  -- Auto-populated by trigger

    section_id TEXT,---

    metadata JSONB DEFAULT '{}',

    access_count INTEGER DEFAULT 0,## 1. Installation and Setup

    last_access TIMESTAMP,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP### Installing PSQLPy

);

```bash

-- Vector similarity index (IVFFlat)pip install psqlpy

CREATE INDEX IF NOT EXISTS idx_shortterm_chunk_embedding```

ON shortterm_memory_chunk

USING ivfflat (embedding vector_cosine_ops)Or with Poetry:

WITH (lists = 100);

```bash

-- BM25 index for text searchpoetry add psqlpy

CREATE INDEX IF NOT EXISTS idx_shortterm_chunk_bm25```

ON shortterm_memory_chunk

USING bm25 (content_bm25 bm25_ops);### Required PostgreSQL Extensions

```

PSQLPy works with standard PostgreSQL, but for advanced features, you'll need these extensions:

### Storing Vectors (from shortterm_memory.py)

```sql

```python-- Vector similarity search

from psqlpy.extra_types import PgVectorCREATE EXTENSION IF NOT EXISTS vector CASCADE;

from typing import List, Optional

-- Hierarchical vector search (optional)

async def create_chunk(CREATE EXTENSION IF NOT EXISTS vchord CASCADE;

    self,

    shortterm_memory_id: int,-- Text tokenization for BM25

    external_id: str,CREATE EXTENSION IF NOT EXISTS pg_tokenizer CASCADE;

    content: str,

    embedding: Optional[List[float]] = None,-- BM25 full-text search

    section_id: Optional[str] = None,CREATE EXTENSION IF NOT EXISTS vchord_bm25 CASCADE;

    metadata: Optional[Dict[str, Any]] = None,```

) -> ShorttermMemoryChunk:

    """**Important:** Extensions must be created BEFORE you run your application code.

    Create a new chunk for a shortterm memory.```

    

    The BM25 vector is auto-populated by database trigger.### Connection Setup

    """

    query = """```python

        INSERT INTO shortterm_memory_chunk from psqlpy import ConnectionPool, Connection

        (shortterm_memory_id, external_id, content, embedding, section_id, metadata, access_count, last_access)from psqlpy.extra_types import PgVector

        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)import os

        RETURNING id, shortterm_memory_id, content, section_id, metadata, access_count, last_accessfrom dotenv import load_dotenv

    """

load_dotenv()

    # Convert embedding to PostgreSQL vector format if provided

    pg_vector = Noneclass PostgreSQLManager:

    if embedding:    def __init__(self, host: str, port: int, username: str, password: str, database: str):

        pg_vector = PgVector(embedding)        self._pool = ConnectionPool(

            host=host,

    async with self.postgres.connection() as conn:            port=port,

        result = await conn.execute(            username=username,

            query,            password=password,

            [            db_name=database,

                shortterm_memory_id,            max_db_pool_size=10

                external_id,        )

                content,    

                pg_vector,  # Use PgVector, not raw list    @classmethod

                section_id,    def from_env(cls):

                metadata or {},        return cls(

                0,  # access_count            host=os.getenv("POSTGRES_HOST", "localhost"),

                None,  # last_access            port=int(os.getenv("POSTGRES_PORT", 5432)),

            ],            username=os.getenv("POSTGRES_USER", "postgres"),

        )            password=os.getenv("POSTGRES_PASSWORD", ""),

            database=os.getenv("POSTGRES_DB", "ai_army")

        row = result.result()[0]        )

        chunk = self._chunk_row_to_model(row)    

            async def get_connection(self) -> Connection:

        logger.info(f"Created shortterm chunk {chunk.id} for memory {shortterm_memory_id}")        return await self._pool.connection()

        return chunk```

```

## Vector Storage

**Key Points:**

- Always convert Python list to `PgVector` before passing to query### Table Schema

- BM25 vector is auto-populated by trigger, don't include it in INSERT

- Use parameterized queries with `$1, $2, $3...` placeholdersCreate tables with vector columns using the `vector(dimensions)` type:



### Vector Similarity Search```sql

-- Short-term Memory Chunks with Vector Embeddings

```pythonCREATE TABLE IF NOT EXISTS shortterm_memory_chunk (

async def vector_search(    id SERIAL PRIMARY KEY,

    self,    shortterm_memory_id INTEGER NOT NULL REFERENCES shortterm_memory(id) ON DELETE CASCADE,

    external_id: str,    chunk_order INTEGER NOT NULL,

    query_embedding: List[float],    content TEXT NOT NULL,

    limit: int = 10,    embedding vector(1024),  -- Vector dimension from config

    min_similarity: float = 0.7,    metadata JSONB DEFAULT '{}'

) -> List[ShorttermMemoryChunk]:);

    """

    Search using vector similarity only.-- Long-term Memory Chunks with Vector Embeddings

    """CREATE TABLE IF NOT EXISTS longterm_memory_chunk (

    pg_vector = PgVector(query_embedding)    id SERIAL PRIMARY KEY,

    longterm_memory_id INTEGER NOT NULL REFERENCES longterm_memory(id) ON DELETE CASCADE,

    query = """    chunk_order INTEGER NOT NULL,

        SELECT     content TEXT NOT NULL,

            id, shortterm_memory_id, external_id, content,     embedding vector(1024),

            section_id, metadata, access_count, last_access, created_at,    importance_score FLOAT DEFAULT 0.5,

            1 - (embedding <=> $1) AS similarity_score    confidence_score FLOAT DEFAULT 0.5,

        FROM shortterm_memory_chunk    valid_from TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

        WHERE external_id = $2    valid_until TIMESTAMP,

          AND embedding IS NOT NULL    metadata JSONB DEFAULT '{}'

          AND 1 - (embedding <=> $1) >= $3);

        ORDER BY embedding <=> $1```

        LIMIT $4

    """### Creating Vector Indexes



    async with self.postgres.connection() as conn:For efficient similarity search, create indexes on vector columns:

        result = await conn.execute(

            query,```sql

            [pg_vector, external_id, min_similarity, limit]-- IVFFlat index for approximate nearest neighbor search

        )CREATE INDEX IF NOT EXISTS shortterm_chunk_embedding_idx 

        rows = result.result()ON shortterm_memory_chunk 

        USING ivfflat (embedding vector_cosine_ops)

        chunks = []WITH (lists = 100);

        for row in rows:

            chunk = self._chunk_row_to_model(row[:9])CREATE INDEX IF NOT EXISTS longterm_chunk_embedding_idx 

            chunk.similarity_score = float(row[9])ON longterm_memory_chunk 

            chunks.append(chunk)USING ivfflat (embedding vector_cosine_ops)

        WITH (lists = 100);

        return chunks```

```

### Storing Vectors with PSQLPy

**Understanding Vector Operators:**

- `<=>` : Cosine distance (use `1 - (embedding <=> query)` for similarity)```python

- `<->` : L2/Euclidean distancefrom psqlpy.extra_types import PgVector

- `<#>` : Inner product (for normalized vectors)from typing import List



---async def create_chunk_with_embedding(

    memory_id: int,

## BM25 Full-Text Search    content: str,

    embedding: List[float],

### Setting Up BM25 (from schema)    chunk_order: int = 0,

    metadata: dict = None

```sql) -> int:

-- 1. Create tokenizer    """

SELECT create_tokenizer('bert', $$ model = "bert_base_uncased" $$);    Create a memory chunk with vector embedding.

    

-- 2. Create trigger to auto-populate bm25vector    Args:

CREATE OR REPLACE FUNCTION update_bm25_vector_memory_chunk()        memory_id: Reference to parent memory

RETURNS TRIGGER AS $$        content: Text content of the chunk

BEGIN        embedding: Vector embedding as list of floats

    IF TG_OP = 'INSERT' OR (TG_OP = 'UPDATE' AND OLD.content IS DISTINCT FROM NEW.content) THEN        chunk_order: Order of chunk in sequence

        -- Automatically tokenize content into bm25vector        metadata: Additional metadata as JSON

        NEW.content_bm25 = tokenize(NEW.content, 'bert');    

    END IF;    Returns:

    RETURN NEW;        ID of created chunk

END;    """

$$ LANGUAGE plpgsql;    connection = await get_db_connection()

    

CREATE TRIGGER trigger_update_bm25_shortterm_memory_chunk    # Convert Python list to PgVector

    BEFORE INSERT OR UPDATE ON shortterm_memory_chunk    pg_vector = PgVector(embedding)

    FOR EACH ROW EXECUTE FUNCTION update_bm25_vector_memory_chunk();    

    query = """

-- 3. Create BM25 index        INSERT INTO shortterm_memory_chunk 

CREATE INDEX IF NOT EXISTS idx_shortterm_chunk_bm25         (shortterm_memory_id, chunk_order, content, embedding, metadata)

ON shortterm_memory_chunk         VALUES ($1, $2, $3, $4, $5)

USING bm25 (content_bm25 bm25_ops);        RETURNING id

```    """

    

**Critical Points:**    result = await connection.execute(

- Tokenizer name ('bert') must match in trigger and queries        query,

- Index name is used in `to_bm25query()` function        [memory_id, chunk_order, content, pg_vector, metadata or {}]

- `content_bm25` is NEVER manually set - trigger handles it    )

- Only insert `content`, the trigger populates `content_bm25`    

    chunk_id = result.result()[0][0]

### BM25 Search Query    return chunk_id

```

```python

async def bm25_search(### Vector Similarity Search

    self,

    external_id: str,```python

    query_text: str,async def search_by_vector_similarity(

    limit: int = 10,    query_embedding: List[float],

    min_score: float = 0.0,    limit: int = 10,

) -> List[ShorttermMemoryChunk]:    similarity_threshold: float = 0.7

    """) -> List[dict]:

    Search using BM25 text ranking only.    """

    """    Search for similar chunks using cosine similarity.

    query = """    

        SELECT     Args:

            id, shortterm_memory_id, external_id, content,         query_embedding: Query vector

            section_id, metadata, access_count, last_access, created_at,        limit: Maximum number of results

            content_bm25 <&> to_bm25query('idx_shortterm_chunk_bm25', tokenize($1, 'bert')) AS bm25_score        similarity_threshold: Minimum similarity score

        FROM shortterm_memory_chunk    

        WHERE external_id = $2    Returns:

          AND content_bm25 IS NOT NULL        List of matching chunks with similarity scores

          AND content_bm25 <&> to_bm25query('idx_shortterm_chunk_bm25', tokenize($1, 'bert')) >= $3    """

        ORDER BY bm25_score DESC    connection = await get_db_connection()

        LIMIT $4    pg_vector = PgVector(query_embedding)

    """    

    query = """

    async with self.postgres.connection() as conn:        SELECT 

        result = await conn.execute(            id,

            query,            content,

            [query_text, external_id, min_score, limit]            metadata,

        )            1 - (embedding <=> $1) AS similarity

        rows = result.result()        FROM shortterm_memory_chunk

                WHERE 1 - (embedding <=> $1) >= $2

        chunks = []        ORDER BY embedding <=> $1

        for row in rows:        LIMIT $3

            chunk = self._chunk_row_to_model(row[:9])    """

            chunk.bm25_score = float(row[9])    

            chunks.append(chunk)    result = await connection.execute(

                query,

        return chunks        [pg_vector, similarity_threshold, limit]

```    )

    

**BM25 Query Components:**    chunks = []

1. `tokenize($1, 'bert')` - Tokenize the query text using 'bert' tokenizer    for row in result.result():

2. `to_bm25query('idx_shortterm_chunk_bm25', tokenized)` - Create BM25 query from tokens using index name        chunks.append({

3. `content_bm25 <&> bm25_query` - BM25 similarity operator, returns relevance score            'id': row[0],

4. Order by score DESC - Most relevant first            'content': row[1],

            'metadata': row[2],

---            'similarity': float(row[3])

        })

## Hybrid Search Implementation    

    return chunks

### Complete Hybrid Search (from shortterm_memory.py)```



```python## BM25 Storage

async def hybrid_search(

    self,### Table Schema with BM25

    external_id: str,

    query_text: str,BM25 uses a special `bm25vector` type that is automatically generated via triggers:

    query_embedding: List[float],

    limit: int = 10,```sql

    vector_weight: float = 0.5,-- Short-term Memory Chunks Table with bm25vector

    bm25_weight: float = 0.5,CREATE TABLE IF NOT EXISTS shortterm_memory_chunk (

    shortterm_memory_id: Optional[int] = None,    id SERIAL PRIMARY KEY,

    min_similarity_score: Optional[float] = None,    shortterm_id INTEGER NOT NULL REFERENCES shortterm_memory (id) ON DELETE CASCADE,

    min_bm25_score: Optional[float] = None,    content TEXT NOT NULL,

) -> List[ShorttermMemoryChunk]:    content_vector vector(768),

    """    content_bm25 bm25vector,  -- Auto-populated by trigger

    Hybrid search combining vector similarity and BM25.    metadata JSONB DEFAULT '{}'

    );

    This is the recommended search method as it combines:

    - Semantic understanding (vector similarity)-- Create tokenizer for BM25

    - Keyword matching (BM25)SELECT create_tokenizer('bert', $$ model = "bert_base_uncased" $$);

    """

    pg_vector = PgVector(query_embedding)-- Trigger to automatically populate bm25vector

CREATE OR REPLACE FUNCTION update_bm25_vector_memory_chunk()

    query = """RETURNS TRIGGER AS $$

        WITH vector_results AS (BEGIN

            SELECT     IF TG_OP = 'INSERT' OR (TG_OP = 'UPDATE' AND OLD.content IS DISTINCT FROM NEW.content) THEN

                id,        -- Automatically tokenize content into bm25vector

                1 - (embedding <=> $1) AS vector_score        NEW.content_bm25 = tokenize(NEW.content, 'bert');

            FROM shortterm_memory_chunk    END IF;

            WHERE external_id = $2 AND embedding IS NOT NULL    RETURN NEW;

            AND ($7::int IS NULL OR shortterm_memory_id = $7)END;

        ),$$ LANGUAGE plpgsql;

        bm25_results AS (

            SELECT CREATE TRIGGER trigger_update_bm25_shortterm_memory_chunk

                id,    BEFORE INSERT OR UPDATE ON shortterm_memory_chunk

                content_bm25 <&> to_bm25query('idx_shortterm_chunk_bm25', tokenize($3, 'bert')) AS bm25_score    FOR EACH ROW EXECUTE FUNCTION update_bm25_vector_memory_chunk();

            FROM shortterm_memory_chunk

            WHERE external_id = $2 AND content_bm25 IS NOT NULL-- Create BM25 index

            AND ($7::int IS NULL OR shortterm_memory_id = $7)CREATE INDEX IF NOT EXISTS idx_shortterm_memory_chunk_bm25 

        )ON shortterm_memory_chunk 

        SELECT USING bm25 (content_bm25 bm25_ops);

            c.id, c.shortterm_memory_id, c.external_id, c.content, ```

            c.section_id, c.metadata, c.access_count, c.last_access, c.created_at,

            COALESCE(v.vector_score, 0) * $4 + COALESCE(b.bm25_score, 0) * $5 AS combined_score,**Important Notes:**

            COALESCE(v.vector_score, 0) AS vector_score,- The `content_bm25` column is of type `bm25vector`, not a regular column

            COALESCE(b.bm25_score, 0) AS bm25_score- Content is automatically tokenized when inserted/updated via the trigger

        FROM shortterm_memory_chunk c- You don't manually populate `content_bm25` - the trigger handles it

        LEFT JOIN vector_results v ON c.id = v.id- The tokenizer (e.g., 'bert') must be created before the trigger

        LEFT JOIN bm25_results b ON c.id = b.id

        WHERE c.external_id = $2### BM25 Search

          AND ($7::int IS NULL OR c.shortterm_memory_id = $7)

          AND (v.vector_score IS NOT NULL OR b.bm25_score IS NOT NULL)BM25 search uses `to_bm25query` and `tokenize` functions to query the bm25vector column:

          AND ($8::float IS NULL OR COALESCE(v.vector_score, 0) >= $8)

          AND ($9::float IS NULL OR COALESCE(b.bm25_score, 0) >= $9)```python

        ORDER BY combined_score DESCasync def search_by_bm25(

        LIMIT $6    query_text: str,

    """    shortterm_id: int,

    limit: int = 10,

    async with self.postgres.connection() as conn:    similarity_threshold: float = 0.0

        result = await conn.execute() -> List[dict]:

            query,    """

            [    Search for chunks using BM25 text ranking.

                pg_vector,    

                external_id,    Args:

                query_text,        query_text: Search query text

                vector_weight,        shortterm_id: Shortterm memory ID to search within

                bm25_weight,        limit: Maximum number of results

                limit,        similarity_threshold: Minimum BM25 score threshold

                shortterm_memory_id,    

                min_similarity_score,    Returns:

                min_bm25_score,        List of matching chunks with BM25 scores

            ],    """

        )    connection = await get_db_connection()

        rows = result.result()    

    # BM25 search using to_bm25query and tokenize

        chunks = []    query = """

        for row in rows:        SELECT 

            chunk = self._chunk_row_to_model(row[:9])            id,

            chunk.similarity_score = float(row[9])  # combined_score            shortterm_id,

            chunk.bm25_score = float(row[11]) if row[11] is not None else None            content,

            chunks.append(chunk)            metadata,

            content_bm25 <&> to_bm25query(

        logger.debug(f"Hybrid search found {len(chunks)} chunks for {external_id}")                'idx_shortterm_memory_chunk_bm25',

        return chunks                tokenize($2, 'bert')

```            ) as score

        FROM shortterm_memory_chunk

### Query Structure Breakdown        WHERE shortterm_id = $1

          AND content_bm25 IS NOT NULL

1. **Vector CTE:** Calculate vector similarity for all chunks          AND (content_bm25 <&> to_bm25query(

2. **BM25 CTE:** Calculate BM25 scores for all chunks              'idx_shortterm_memory_chunk_bm25',

3. **Main Query:** Join results and compute weighted combined score              tokenize($2, 'bert')

4. **Scoring Formula:** `combined_score = (vector_score * vector_weight) + (bm25_score * bm25_weight)`          )) >= $3

        ORDER BY score DESC

### Weight Tuning Examples        LIMIT $4

    """

```python    

# Semantic-focused search (emphasize meaning)    result = await connection.execute(

results = await hybrid_search(        query,

    external_id, query_text, query_embedding,        [shortterm_id, query_text, similarity_threshold, limit]

    vector_weight=0.8,  # 80% vector    )

    bm25_weight=0.2     # 20% BM25    

)    chunks = []

    for row in result.result():

# Keyword-focused search (emphasize exact matches)        chunks.append({

results = await hybrid_search(            'id': row[0],

    external_id, query_text, query_embedding,            'shortterm_id': row[1],

    vector_weight=0.3,  # 30% vector            'content': row[2],

    bm25_weight=0.7     # 70% BM25            'metadata': row[3],

)            'bm25_score': float(row[4])

        })

# Balanced search    

results = await hybrid_search(    return chunks

    external_id, query_text, query_embedding,```

    vector_weight=0.5,  # 50% vector

    bm25_weight=0.5     # 50% BM25**Key Points:**

)- Use `to_bm25query(index_name, tokenized_query)` to create a BM25 query

```- Use `tokenize(text, tokenizer_name)` to tokenize the search text

- The `<&>` operator performs BM25 similarity search

---- Index name must match the BM25 index created earlier

- Tokenizer name must match the tokenizer used in the trigger ('bert' in this case)

## Real-World Examples from Codebase

## Hybrid Search

### Example 1: Creating Memory with Chunks (from repositories)

Combine vector similarity and BM25 for better search results:

```python

from agent_reminiscence.database.postgres_manager import PostgreSQLManager```python

from agent_reminiscence.database.repositories.shortterm_memory import ShorttermMemoryRepositoryasync def hybrid_search(

from agent_reminiscence.config import get_config    query_text: str,

    query_embedding: List[float],

# Initialize manager    limit: int = 10,

config = get_config()    vector_weight: float = 0.7,

postgres_manager = PostgreSQLManager(config)    bm25_weight: float = 0.3

await postgres_manager.initialize()) -> List[dict]:

    """

# Create repository    Perform hybrid search combining vector similarity and BM25.

repo = ShorttermMemoryRepository(postgres_manager, neo4j_manager)    

    Args:

# Create memory        query_text: Search query text

memory = await repo.create_memory(        query_embedding: Query vector embedding

    external_id="agent-123",        limit: Maximum number of results

    title="Python Best Practices",        vector_weight: Weight for vector similarity (0-1)

    summary="Collection of Python coding standards",        bm25_weight: Weight for BM25 score (0-1)

    metadata={"category": "programming"}    

)    Returns:

        List of chunks ranked by combined score

# Add chunks with embeddings    """

chunk1 = await repo.create_chunk(    connection = await get_db_connection()

    shortterm_memory_id=memory.id,    pg_vector = PgVector(query_embedding)

    external_id="agent-123",    

    content="Always use type hints in function signatures",    query = """

    embedding=get_embedding("type hints python"),  # Your embedding function        WITH vector_search AS (

    section_id="section-1",            SELECT 

    metadata={"importance": 0.9}                id,

)                content,

                metadata,

chunk2 = await repo.create_chunk(                1 - (embedding <=> $1) AS vector_similarity

    shortterm_memory_id=memory.id,            FROM shortterm_memory_chunk

    external_id="agent-123",            WHERE embedding IS NOT NULL

    content="Use list comprehensions for simple transformations",        ),

    embedding=get_embedding("list comprehensions"),        bm25_search AS (

    section_id="section-2",            SELECT 

    metadata={"importance": 0.8}                id,

)                content,

```                metadata,

                bm25_score

### Example 2: Hybrid Search for Memory Retrieval            FROM shortterm_memory_chunk

            WHERE content @@@ $2

```python        ),

# Search for relevant memories        max_scores AS (

query = "How to write clean Python code?"            SELECT 

query_embedding = get_embedding(query)                MAX(vector_similarity) as max_vector,

                MAX(bm25_score) as max_bm25

results = await repo.hybrid_search(            FROM vector_search, bm25_search

    external_id="agent-123",        )

    query_text=query,        SELECT 

    query_embedding=query_embedding,            COALESCE(v.id, b.id) as id,

    limit=5,            COALESCE(v.content, b.content) as content,

    vector_weight=0.7,  # Emphasize semantic similarity            COALESCE(v.metadata, b.metadata) as metadata,

    bm25_weight=0.3,            (

    min_similarity_score=0.5,  # Filter low-relevance results                ($3 * COALESCE(v.vector_similarity, 0) / NULLIF(m.max_vector, 0)) +

    min_bm25_score=0.1                ($4 * COALESCE(b.bm25_score, 0) / NULLIF(m.max_bm25, 0))

)            ) AS combined_score

        FROM vector_search v

for chunk in results:        FULL OUTER JOIN bm25_search b ON v.id = b.id

    print(f"Score: {chunk.similarity_score:.2f}")        CROSS JOIN max_scores m

    print(f"Content: {chunk.content}")        ORDER BY combined_score DESC

    print(f"BM25: {chunk.bm25_score:.2f}")        LIMIT $5

    print("---")    """

```    

    result = await connection.execute(

### Example 3: Using PostgreSQLManager Utilities        query,

        [pg_vector, query_text, vector_weight, bm25_weight, limit]

```python    )

# Verify connection    

is_connected = await postgres_manager.verify_connection()    chunks = []

if not is_connected:    for row in result.result():

    logger.error("Database connection failed!")        chunks.append({

            'id': row[0],

# Ensure extensions are installed            'content': row[1],

await postgres_manager.ensure_extensions()            'metadata': row[2],

            'combined_score': float(row[3])

# Execute custom query        })

result = await postgres_manager.execute(    

    "SELECT COUNT(*) FROM shortterm_memory_chunk WHERE external_id = $1",    return chunks

    ["agent-123"]```

)

count = result.result()[0][0]### Real-World Implementation Example

print(f"Total chunks: {count}")

```From the codebase (`longterm_memory_repository.py`):



### Example 4: Batch Insert with Transaction```python

async def search_chunks(

```python    self,

async def batch_create_chunks(    query_vector: Optional[List[float]] = None,

    repo: ShorttermMemoryRepository,    query_text: Optional[str] = None,

    memory_id: int,    limit: int = 10,

    external_id: str,    similarity_threshold: float = 0.0,

    chunks_data: List[dict]) -> List[Tuple[LongtermMemoryChunk, float, float]]:

) -> List[ShorttermMemoryChunk]:    """Search longterm memory chunks using hybrid search."""

    """Create multiple chunks in a transaction."""    

    created_chunks = []    connection = await get_db_connection()

        

    async with repo.postgres.connection() as conn:    if query_vector and query_text:

        await conn.execute("BEGIN")        # Hybrid search with both vector and BM25

                pg_vector = PgVector(query_vector)

        try:        

            for chunk_data in chunks_data:        query = """

                chunk = await repo.create_chunk(            WITH vector_results AS (

                    shortterm_memory_id=memory_id,                SELECT 

                    external_id=external_id,                    lmc.*,

                    content=chunk_data["content"],                    1 - (lmc.embedding <=> $1) AS vector_similarity,

                    embedding=chunk_data["embedding"],                    0.0 AS bm25_score

                    section_id=chunk_data.get("section_id"),                FROM longterm_memory_chunk lmc

                    metadata=chunk_data.get("metadata", {})                WHERE lmc.embedding IS NOT NULL

                )                  AND (lmc.valid_until IS NULL OR lmc.valid_until > CURRENT_TIMESTAMP)

                created_chunks.append(chunk)            ),

                        bm25_results AS (

            await conn.execute("COMMIT")                SELECT 

            logger.info(f"Created {len(created_chunks)} chunks")                    lmc.*,

            return created_chunks                    0.0 AS vector_similarity,

                                lmc.bm25_score AS bm25_score

        except Exception as e:                FROM longterm_memory_chunk lmc

            await conn.execute("ROLLBACK")                WHERE lmc.content @@@ $2

            logger.error(f"Batch insert failed: {e}")                  AND (lmc.valid_until IS NULL OR lmc.valid_until > CURRENT_TIMESTAMP)

            raise            ),

```            combined AS (

                SELECT * FROM vector_results

### Example 5: Result Processing                UNION

                SELECT * FROM bm25_results

```python            )

def _chunk_row_to_model(self, row) -> ShorttermMemoryChunk:            SELECT 

    """                id, longterm_memory_id, chunk_order, content,

    Convert database row to ShorttermMemoryChunk model.                importance_score, confidence_score, valid_from, valid_until,

                    metadata, vector_similarity, bm25_score

    This is the pattern used throughout the codebase.            FROM combined

    """            WHERE vector_similarity >= $3 OR bm25_score > 0

    return ShorttermMemoryChunk(            ORDER BY (vector_similarity + bm25_score) DESC

        id=row[0],            LIMIT $4

        shortterm_memory_id=row[1],        """

        external_id=row[2],        

        content=row[3],        result = await connection.execute(

        section_id=row[4] if len(row) > 4 else None,            query,

        metadata=row[5] if len(row) > 5 else {},            [pg_vector, query_text, similarity_threshold, limit]

        access_count=row[6] if len(row) > 6 else 0,        )

        last_access=row[7] if len(row) > 7 else None,    

        created_at=row[8] if len(row) > 8 else None,    elif query_vector:

    )        # Vector-only search

```        pg_vector = PgVector(query_vector)

        

---        query = """

            SELECT 

## Best Practices Summary                id, longterm_memory_id, chunk_order, content,

                importance_score, confidence_score, valid_from, valid_until,

### Vector Operations                metadata,

1. ✅ Always use `PgVector(embedding)` when passing to queries                1 - (embedding <=> $1) AS vector_similarity,

2. ✅ Normalize vectors before storage for consistent similarity                0.0 AS bm25_score

3. ✅ Create vector indexes with appropriate list size (100-1000)            FROM longterm_memory_chunk

4. ✅ Use cosine distance (`<=>`) for semantic similarity            WHERE embedding IS NOT NULL

5. ✅ Filter with `WHERE embedding IS NOT NULL`              AND 1 - (embedding <=> $1) >= $2

              AND (valid_until IS NULL OR valid_until > CURRENT_TIMESTAMP)

### BM25 Operations            ORDER BY embedding <=> $1

1. ✅ Never manually insert into `bm25vector` columns            LIMIT $3

2. ✅ Use exact index name in `to_bm25query()`        """

3. ✅ Match tokenizer name across trigger and queries        

4. ✅ Filter with `WHERE content_bm25 IS NOT NULL`        result = await connection.execute(

5. ✅ Create BM25 index on bm25vector column            query,

            [pg_vector, similarity_threshold, limit]

### Hybrid Search        )

1. ✅ Use hybrid search for best results    

2. ✅ Tune weights based on use case (semantic vs keyword)    # Parse results into chunk objects

3. ✅ Set minimum thresholds to filter noise    chunks = []

4. ✅ Use CTEs for clean separation of concerns    for row in result.result():

5. ✅ Return both individual and combined scores        chunk = LongtermMemoryChunk(

            id=row[0],

### Connection Management            longterm_memory_id=row[1],

1. ✅ Initialize manager once at startup            chunk_order=row[2],

2. ✅ Use connection context managers            content=row[3],

3. ✅ Close pool on shutdown            importance_score=row[4],

4. ✅ Use transactions for related operations            confidence_score=row[5],

5. ✅ Handle exceptions appropriately            valid_from=row[6],

            valid_until=row[7],

---            metadata=row[8]

        )

## Common Errors and Solutions        vector_sim = float(row[9])

        bm25_score = float(row[10])

### Error: "dimension mismatch"        chunks.append((chunk, vector_sim, bm25_score))

```python    

# Problem: Embedding dimension doesn't match table schema    return chunks

embedding = [0.1, 0.2]  # Only 2 dimensions```

# Table expects vector(768)

## Best Practices

# Solution: Ensure dimension matches

assert len(embedding) == 768### 1. Vector Dimension Consistency

pg_vector = PgVector(embedding)

```Always use consistent vector dimensions across your application:



### Error: "column content_bm25 does not exist"```python

```python
from agent_reminiscence.config import get_config

config = get_config()

# Ensure embeddings match configured dimension
assert len(embedding) == config.vector_dimension
```

await conn.execute(```

    "INSERT INTO table (content, content_bm25) VALUES ($1, $2)",

    [content, bm25_value]### 2. Batch Operations

)

For inserting multiple chunks, use batch operations:

# Correct:

await conn.execute(```python

    "INSERT INTO table (content) VALUES ($1)",async def batch_create_chunks(chunks_data: List[dict]) -> List[int]:

    [content]  # Trigger populates content_bm25    """Create multiple chunks in a single transaction."""

)    connection = await get_db_connection()

```    

    query = """

### Error: "index 'idx_name' does not exist"        INSERT INTO shortterm_memory_chunk 

```python        (shortterm_memory_id, chunk_order, content, embedding, metadata)

# Problem: Wrong index name in to_bm25query()        VALUES ($1, $2, $3, $4, $5)

        RETURNING id

# Check your CREATE INDEX statement and use exact name:    """

to_bm25query('idx_shortterm_chunk_bm25', tokenize($1, 'bert'))    

```    chunk_ids = []

    async with connection.transaction():

### Error: "tokenizer 'bert' does not exist"        for chunk in chunks_data:

```python            result = await connection.execute(

# Problem: Tokenizer not created                query,

                [

# Solution: Run this SQL first:                    chunk['memory_id'],

SELECT create_tokenizer('bert', $$ model = "bert_base_uncased" $$);                    chunk['chunk_order'],

```                    chunk['content'],

                    PgVector(chunk['embedding']),

---                    chunk.get('metadata', {})

                ]

## References            )

            chunk_ids.append(result.result()[0][0])

- **Complete PSQLPy Guide:** [psqlpy-complete-guide.md](./psqlpy-complete-guide.md)    

- **Source Code:**    return chunk_ids

  - `agent_reminiscence/database/postgres_manager.py` - Manager implementation```

  - `agent_reminiscence/database/repositories/shortterm_memory.py` - Repository with search

  - `agent_reminiscence/database/init_databases.py` - Schema initialization### 3. Index Maintenance

  - `agent_reminiscence/sql/schema.sql` - Complete database schema

Rebuild indexes periodically for optimal performance:

- **PostgreSQL Extensions:**

  - [pgvector documentation](https://github.com/pgvector/pgvector)```sql

  - [vchord_bm25 documentation](https://github.com/tensorchord/vchord)-- Reindex vector indexes

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

