# Neo4j Entities and Relationships Storage Guide

## Overview

This guide explains how to use Neo4j to store and query entities and relationships in the AI Army application. The system uses a dual-storage approach with PostgreSQL for chunks and Neo4j for graph data, supporting both shortterm and longterm memory.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Connection Setup](#connection-setup)
3. [Schema Design](#schema-design)
4. [Shortterm Entities](#shortterm-entities)
5. [Longterm Entities](#longterm-entities)
6. [Relationships](#relationships)
7. [Querying Patterns](#querying-patterns)
8. [Best Practices](#best-practices)

## Prerequisites

### Neo4j Setup

Ensure Neo4j is installed and running. Set environment variables:

```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j
```

### Required Constraints

Run these Cypher commands to create constraints:

```cypher
// Unique constraints for entity IDs
CREATE CONSTRAINT shortterm_entity_unique IF NOT EXISTS 
FOR (e:ShorttermEntity) REQUIRE e.id IS UNIQUE;

CREATE CONSTRAINT longterm_entity_unique IF NOT EXISTS 
FOR (e:LongtermEntity) REQUIRE e.id IS UNIQUE;

// Unique constraint for relationship IDs
CREATE CONSTRAINT relationship_id_unique IF NOT EXISTS 
FOR ()-[r:RELATES_TO]-() REQUIRE r.id IS UNIQUE;
```

## Connection Setup

### Connection Manager

```python
from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv

load_dotenv()

class Neo4jManager:
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        self._uri = uri
        self._user = user
        self._password = password
        self._database = database
        self._driver: Optional[AsyncDriver] = None
    
    @classmethod
    def from_env(cls):
        """Create manager from environment variables."""
        return cls(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            user=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "password"),
            database=os.getenv("NEO4J_DATABASE", "neo4j")
        )
    
    async def initialize(self):
        """Initialize the Neo4j driver."""
        if not self._driver:
            self._driver = AsyncGraphDatabase.driver(
                self._uri,
                auth=(self._user, self._password)
            )
    
    async def close(self):
        """Close the driver connection."""
        if self._driver:
            await self._driver.close()
            self._driver = None
    
    @asynccontextmanager
    async def get_write_session(self) -> AsyncSession:
        """Get a write session for write operations."""
        if not self._driver:
            await self.initialize()
        
        async with self._driver.session(database=self._database) as session:
            yield session
    
    @asynccontextmanager
    async def get_read_session(self) -> AsyncSession:
        """Get a read session for read-only operations."""
        if not self._driver:
            await self.initialize()
        
        async with self._driver.session(
            database=self._database,
            default_access_mode="READ"
        ) as session:
            yield session

# Global instance
neo4j_manager = Neo4jManager.from_env()
```

### Session Context Managers

```python
from database.neo4j_conn import get_neo4j_write_session, get_neo4j_read_session

# For write operations
async def create_entity():
    async with get_neo4j_write_session() as session:
        result = await session.run(query, parameters)
        return result

# For read operations
async def query_entities():
    async with get_neo4j_read_session() as session:
        result = await session.run(query, parameters)
        return result
```

## Schema Design

### Entity Node Structure

Entities are stored as nodes with labels indicating their memory type:

```cypher
// Shortterm Entity Node
(:ShorttermEntity {
    id: INTEGER,              // Unique identifier (from PostgreSQL)
    shortterm_memory_id: INTEGER,  // Reference to PostgreSQL shortterm_memory.id
    worker_id: INTEGER,       // Worker ID for filtering
    name: STRING,             // Entity name
    type: STRING,             // Entity type (Person, Organization, Concept, etc.)
    description: STRING,      // Description of entity
    first_seen: DATETIME,     // When entity was first observed
    last_seen: DATETIME,      // When entity was last observed
    confidence: FLOAT,        // Confidence score (0-1)
    metadata: MAP             // Additional properties
})

// Longterm Entity Node
(:LongtermEntity {
    id: INTEGER,
    worker_id: INTEGER,
    name: STRING,
    type: STRING,
    description: STRING,
    first_seen: DATETIME,
    last_seen: DATETIME,
    confidence: FLOAT,
    importance: FLOAT,        // Importance score (0-1)
    metadata: MAP
})
```

### Relationship Structure

Relationships connect entities with temporal and confidence properties. Longterm relationships use `start_date` and `end_date` to distinguish between different relationships across time:

```cypher
// Shortterm Relationship structure
(source:ShorttermEntity)-[r:RELATES_TO {
    id: INTEGER,              // Unique identifier
    shortterm_memory_id: INTEGER,  // Reference to PostgreSQL shortterm_memory.id
    worker_id: INTEGER,       // Worker ID for filtering
    from_entity_id: INTEGER,  // Source entity ID
    to_entity_id: INTEGER,    // Target entity ID
    type: STRING,             // Relationship type
    description: STRING,      // Description of relationship
    confidence: FLOAT,        // Confidence score (0-1)
    strength: FLOAT,          // Relationship strength (0-1)
    first_observed: DATETIME, // When first observed
    last_observed: DATETIME,  // When last observed
    metadata: MAP             // Additional properties
}]->(target:ShorttermEntity)

// Longterm Relationship structure with temporal constraints
(source:LongtermEntity)-[r:RELATES_TO {
    id: INTEGER,
    worker_id: INTEGER,
    from_entity_id: INTEGER,
    to_entity_id: INTEGER,
    type: STRING,
    description: STRING,
    confidence: FLOAT,
    strength: FLOAT,
    start_date: DATETIME,     // Relationship start date (for temporal tracking)
    last_updated: DATETIME,   // When relationship was last updated
    metadata: MAP
}]->(target:LongtermEntity)
```

**Longterm Relationship Constraints:**

To prevent duplicate relationships:

```cypher
// Constraint: Unique relationship per (from_entity, to_entity, type, start_date)
CREATE CONSTRAINT relationship_temporal_unique IF NOT EXISTS 
FOR ()-[r:RELATES_TO]-()
REQUIRE (r.from_entity_id, r.to_entity_id, r.type, r.start_date) IS UNIQUE;
```

This constraint ensures that:
- The same relationship type between two entities can exist across different time periods
- Each period is uniquely identified by its `start_date`
- When a relationship changes over time, a new relationship with a new `start_date` is created
- The `last_updated` timestamp tracks when the relationship was last modified

## Shortterm Entities

### Creating Shortterm Entities

```python
from typing import Dict, Any, Optional
from datetime import datetime

async def create_shortterm_entity(
    entity_id: int,
    shortterm_memory_id: int,
    worker_id: int,
    name: str,
    entity_type: str,
    description: str = "",
    confidence: float = 0.5,
    metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Create a shortterm entity node in Neo4j.
    
    Args:
        entity_id: Unique entity ID (from PostgreSQL)
        shortterm_memory_id: Reference to shortterm_memory.id
        worker_id: Worker ID
        name: Entity name
        entity_type: Type of entity
        description: Entity description
        confidence: Confidence score
        metadata: Additional properties
    
    Returns:
        True if created successfully
    """
    async with get_neo4j_write_session() as session:
        query = """
        CREATE (e:ShorttermEntity {
            id: $id,
            shortterm_memory_id: $shortterm_memory_id,
            worker_id: $worker_id,
            name: $name,
            type: $type,
            description: $description,
            first_seen: datetime($first_seen),
            last_seen: datetime($last_seen),
            confidence: $confidence,
            metadata: $metadata
        })
        RETURN e.id as id
        """
        
        now = datetime.now().isoformat()
        result = await session.run(
            query,
            id=entity_id,
            shortterm_memory_id=shortterm_memory_id,
            worker_id=worker_id,
            name=name,
            type=entity_type,
            description=description,
            first_seen=now,
            last_seen=now,
            confidence=confidence,
            metadata=metadata or {}
        )
        
        record = await result.single()
        return record is not None
```
        metadata: Additional properties
    
    Returns:
        True if created successfully
    """
    async with get_neo4j_write_session() as session:
        query = """
        CREATE (e:ShorttermEntity {
            id: $id,
            name: $name,
            type: $type,
            description: $description,
            first_seen: datetime($first_seen),
            last_seen: datetime($last_seen),
            confidence: $confidence,
            metadata: $metadata
        })
        RETURN e.id as id
        """
        
        now = datetime.now().isoformat()
        result = await session.run(
            query,
            id=entity_id,
            name=name,
            type=entity_type,
            description=description,
            first_seen=now,
            last_seen=now,
            confidence=confidence,
            metadata=metadata or {}
        )
        
        record = await result.single()
        return record is not None
```

### Updating Shortterm Entities

```python
async def update_shortterm_entity(
    entity_id: int,
    name: Optional[str] = None,
    description: Optional[str] = None,
    confidence: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Update an existing shortterm entity.
    
    Args:
        entity_id: Entity ID to update
        name: New name (if provided)
        description: New description (if provided)
        confidence: New confidence score (if provided)
        metadata: Metadata to merge (if provided)
    
    Returns:
        True if updated successfully
    """
    async with get_neo4j_write_session() as session:
        # Build dynamic SET clause
        set_clauses = ["e.last_seen = datetime($last_seen)"]
        params = {"id": entity_id, "last_seen": datetime.now().isoformat()}
        
        if name is not None:
            set_clauses.append("e.name = $name")
            params["name"] = name
        
        if description is not None:
            set_clauses.append("e.description = $description")
            params["description"] = description
        
        if confidence is not None:
            set_clauses.append("e.confidence = $confidence")
            params["confidence"] = confidence
        
        if metadata is not None:
            set_clauses.append("e.metadata = e.metadata + $metadata")
            params["metadata"] = metadata
        
        query = f"""
        MATCH (e:ShorttermEntity {{id: $id}})
        SET {', '.join(set_clauses)}
        RETURN e.id as id
        """
        
        result = await session.run(query, **params)
        record = await result.single()
        return record is not None
```

### Querying Shortterm Entities

```python
async def get_shortterm_entity_by_id(entity_id: int) -> Optional[Dict[str, Any]]:
    """Get a shortterm entity by ID."""
    async with get_neo4j_read_session() as session:
        query = """
        MATCH (e:ShorttermEntity {id: $id})
        RETURN e
        """
        
        result = await session.run(query, id=entity_id)
        record = await result.single()
        
        if record:
            entity = record["e"]
            return {
                "id": entity["id"],
                "name": entity["name"],
                "type": entity["type"],
                "description": entity["description"],
                "first_seen": entity["first_seen"],
                "last_seen": entity["last_seen"],
                "confidence": entity["confidence"],
                "metadata": entity["metadata"]
            }
        return None

async def search_shortterm_entities_by_name(
    name_pattern: str,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """Search shortterm entities by name pattern."""
    async with get_neo4j_read_session() as session:
        query = """
        MATCH (e:ShorttermEntity)
        WHERE e.name CONTAINS $pattern
        RETURN e
        ORDER BY e.confidence DESC, e.last_seen DESC
        LIMIT $limit
        """
        
        result = await session.run(query, pattern=name_pattern, limit=limit)
        
        entities = []
        async for record in result:
            entity = record["e"]
            entities.append({
                "id": entity["id"],
                "name": entity["name"],
                "type": entity["type"],
                "description": entity["description"],
                "confidence": entity["confidence"]
            })
        
        return entities
```

## Longterm Entities

### Creating Longterm Entities

```python
async def create_longterm_entity(
    entity_id: int,
    name: str,
    entity_type: str,
    description: str = "",
    confidence: float = 0.8,
    importance: float = 0.5,
    valid_from: Optional[datetime] = None,
    valid_until: Optional[datetime] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Create a longterm entity node in Neo4j.
    
    Longterm entities include importance scores and temporal validity.
    """
    async with get_neo4j_write_session() as session:
        query = """
        CREATE (e:LongtermEntity {
            id: $id,
            name: $name,
            type: $type,
            description: $description,
            first_seen: datetime($first_seen),
            last_seen: datetime($last_seen),
            confidence: $confidence,
            importance: $importance,
            valid_from: datetime($valid_from),
            valid_until: CASE WHEN $valid_until IS NULL 
                THEN NULL 
                ELSE datetime($valid_until) 
            END,
            metadata: $metadata
        })
        RETURN e.id as id
        """
        
        now = datetime.now()
        result = await session.run(
            query,
            id=entity_id,
            name=name,
            type=entity_type,
            description=description,
            first_seen=now.isoformat(),
            last_seen=now.isoformat(),
            confidence=confidence,
            importance=importance,
            valid_from=(valid_from or now).isoformat(),
            valid_until=valid_until.isoformat() if valid_until else None,
            metadata=metadata or {}
        )
        
        record = await result.single()
        return record is not None
```

### Querying Valid Longterm Entities

```python
async def get_valid_longterm_entities(
    entity_type: Optional[str] = None,
    min_importance: float = 0.0,
    as_of: Optional[datetime] = None
) -> List[Dict[str, Any]]:
    """
    Get valid longterm entities as of a specific time.
    
    Args:
        entity_type: Filter by entity type
        min_importance: Minimum importance score
        as_of: Check validity as of this datetime (default: now)
    
    Returns:
        List of valid entities
    """
    async with get_neo4j_read_session() as session:
        as_of_time = (as_of or datetime.now()).isoformat()
        
        type_filter = ""
        if entity_type:
            type_filter = "AND e.type = $entity_type"
        
        query = f"""
        MATCH (e:LongtermEntity)
        WHERE e.importance >= $min_importance
          AND e.valid_from <= datetime($as_of)
          AND (e.valid_until IS NULL OR e.valid_until > datetime($as_of))
          {type_filter}
        RETURN e
        ORDER BY e.importance DESC, e.confidence DESC
        """
        
        params = {
            "min_importance": min_importance,
            "as_of": as_of_time
        }
        if entity_type:
            params["entity_type"] = entity_type
        
        result = await session.run(query, **params)
        
        entities = []
        async for record in result:
            entity = record["e"]
            entities.append({
                "id": entity["id"],
                "name": entity["name"],
                "type": entity["type"],
                "description": entity["description"],
                "confidence": entity["confidence"],
                "importance": entity["importance"],
                "valid_from": entity["valid_from"],
                "valid_until": entity.get("valid_until")
            })
        
        return entities
```

## Relationships

### Creating Relationships

```python
async def create_shortterm_relationship(
    relationship_id: int,
    shortterm_memory_id: int,
    worker_id: int,
    source_entity_id: int,
    target_entity_id: int,
    relationship_type: str,
    description: str = "",
    confidence: float = 0.5,
    strength: float = 0.5,
    metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Create a relationship between two shortterm entities.
    
    Args:
        relationship_id: Unique relationship ID
        shortterm_memory_id: Reference to shortterm_memory.id
        worker_id: Worker ID
        source_entity_id: Source entity ID
        target_entity_id: Target entity ID
        relationship_type: Type of relationship
        description: Relationship description
        confidence: Confidence score
        strength: Relationship strength
        metadata: Additional properties
    
    Returns:
        True if created successfully
    """
    async with get_neo4j_write_session() as session:
        query = """
        MATCH (source:ShorttermEntity {id: $source_id})
        MATCH (target:ShorttermEntity {id: $target_id})
        CREATE (source)-[r:RELATES_TO {
            id: $rel_id,
            shortterm_memory_id: $shortterm_memory_id,
            worker_id: $worker_id,
            from_entity_id: $source_id,
            to_entity_id: $target_id,
            type: $rel_type,
            description: $description,
            confidence: $confidence,
            strength: $strength,
            first_observed: datetime($first_observed),
            last_observed: datetime($last_observed),
            metadata: $metadata
        }]->(target)
        RETURN r.id as id
        """
        
        now = datetime.now().isoformat()
        result = await session.run(
            query,
            source_id=source_entity_id,
            target_id=target_entity_id,
            rel_id=relationship_id,
            shortterm_memory_id=shortterm_memory_id,
            worker_id=worker_id,
            rel_type=relationship_type,
            description=description,
            confidence=confidence,
            strength=strength,
            first_observed=now,
            last_observed=now,
            metadata=metadata or {}
        )
        
        record = await result.single()
        return record is not None
```

### Querying Relationships

```python
async def get_entity_relationships(
    entity_id: int,
    direction: str = "both",  # "outgoing", "incoming", "both"
    min_confidence: float = 0.0
) -> List[Dict[str, Any]]:
    """
    Get all relationships for an entity.
    
    Args:
        entity_id: Entity ID
        direction: Relationship direction
        min_confidence: Minimum confidence threshold
    
    Returns:
        List of relationships with connected entities
    """
    async with get_neo4j_read_session() as session:
        if direction == "outgoing":
            pattern = "(source)-[r:RELATES_TO]->(target)"
            entity_match = "source.id = $entity_id"
            other_entity = "target"
        elif direction == "incoming":
            pattern = "(source)-[r:RELATES_TO]->(target)"
            entity_match = "target.id = $entity_id"
            other_entity = "source"
        else:  # both
            pattern = "(e1)-[r:RELATES_TO]-(e2)"
            entity_match = "e1.id = $entity_id"
            other_entity = "e2"
            
        query = f"""
        MATCH {pattern}
        WHERE {entity_match}
          AND r.confidence >= $min_confidence
        RETURN r, {other_entity} as other
        ORDER BY r.strength DESC, r.confidence DESC
        """
        
        result = await session.run(
            query,
            entity_id=entity_id,
            min_confidence=min_confidence
        )
        
        relationships = []
        async for record in result:
            rel = record["r"]
            other = record["other"]
            
            relationships.append({
                "id": rel["id"],
                "type": rel["type"],
                "description": rel["description"],
                "confidence": rel["confidence"],
                "strength": rel["strength"],
                "first_observed": rel["first_observed"],
                "last_observed": rel["last_observed"],
                "connected_entity": {
                    "id": other["id"],
                    "name": other["name"],
                    "type": other["type"]
                }
            })
        
        return relationships
```

### Advanced Relationship Queries

```python
async def find_path_between_entities(
    source_id: int,
    target_id: int,
    max_depth: int = 5
) -> List[List[Dict[str, Any]]]:
    """
    Find paths between two entities.
    
    Args:
        source_id: Source entity ID
        target_id: Target entity ID
        max_depth: Maximum path length
    
    Returns:
        List of paths (each path is a list of nodes and relationships)
    """
    async with get_neo4j_read_session() as session:
        query = """
        MATCH path = (source {id: $source_id})-[*1..$max_depth]-(target {id: $target_id})
        WHERE ALL(r IN relationships(path) WHERE r.confidence > 0.5)
        RETURN [node IN nodes(path) | node] as nodes,
               [rel IN relationships(path) | rel] as rels
        ORDER BY length(path) ASC
        LIMIT 10
        """
        
        result = await session.run(
            query,
            source_id=source_id,
            target_id=target_id,
            max_depth=max_depth
        )
        
        paths = []
        async for record in result:
            nodes = record["nodes"]
            rels = record["rels"]
            
            path = []
            for i, node in enumerate(nodes):
                path.append({
                    "type": "entity",
                    "id": node["id"],
                    "name": node["name"]
                })
                
                if i < len(rels):
                    path.append({
                        "type": "relationship",
                        "id": rels[i]["id"],
                        "rel_type": rels[i]["type"],
                        "confidence": rels[i]["confidence"]
                    })
            
            paths.append(path)
        
        return paths
```

## Querying Patterns

### Entity with Relationships

```python
async def get_entity_with_relationships(
    entity_id: int
) -> Dict[str, Any]:
    """
    Get entity with all its relationships in a single query.
    """
    async with get_neo4j_read_session() as session:
        query = """
        MATCH (e {id: $entity_id})
        OPTIONAL MATCH (e)-[r:RELATES_TO]-(other)
        RETURN e,
               collect({
                   relationship: r,
                   entity: other
               }) as relationships
        """
        
        result = await session.run(query, entity_id=entity_id)
        record = await result.single()
        
        if not record:
            return None
        
        entity = record["e"]
        relationships = record["relationships"]
        
        return {
            "entity": {
                "id": entity["id"],
                "name": entity["name"],
                "type": entity["type"],
                "description": entity["description"],
                "confidence": entity.get("confidence", 0.0)
            },
            "relationships": [
                {
                    "id": rel["relationship"]["id"],
                    "type": rel["relationship"]["type"],
                    "confidence": rel["relationship"]["confidence"],
                    "connected_to": {
                        "id": rel["entity"]["id"],
                        "name": rel["entity"]["name"],
                        "type": rel["entity"]["type"]
                    }
                }
                for rel in relationships if rel["relationship"] is not None
            ]
        }
```

### Batch Operations

```python
async def batch_create_entities_and_relationships(
    entities: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]]
) -> bool:
    """
    Create multiple entities and relationships in a single transaction.
    """
    async with get_neo4j_write_session() as session:
        async with session.begin_transaction() as tx:
            # Create entities
            entity_query = """
            UNWIND $entities as entity
            CREATE (e:ShorttermEntity {
                id: entity.id,
                name: entity.name,
                type: entity.type,
                description: entity.description,
                first_seen: datetime(entity.first_seen),
                last_seen: datetime(entity.last_seen),
                confidence: entity.confidence,
                metadata: entity.metadata
            })
            """
            
            await tx.run(entity_query, entities=entities)
            
            # Create relationships
            rel_query = """
            UNWIND $relationships as rel
            MATCH (source:ShorttermEntity {id: rel.source_id})
            MATCH (target:ShorttermEntity {id: rel.target_id})
            CREATE (source)-[r:RELATES_TO {
                id: rel.id,
                type: rel.type,
                description: rel.description,
                confidence: rel.confidence,
                strength: rel.strength,
                first_observed: datetime(rel.first_observed),
                last_observed: datetime(rel.last_observed),
                metadata: rel.metadata
            }]->(target)
            """
            
            await tx.run(rel_query, relationships=relationships)
            await tx.commit()
        
        return True
```

## Best Practices

### 1. Use Parameterized Queries

Always use parameters to prevent Cypher injection:

```python
# Good
query = "MATCH (e:Entity {id: $id}) RETURN e"
result = await session.run(query, id=entity_id)

# Bad - vulnerable to injection
query = f"MATCH (e:Entity {{id: {entity_id}}}) RETURN e"
```

### 2. Leverage Indexes

Create indexes on frequently queried properties:

```cypher
CREATE INDEX entity_name_idx IF NOT EXISTS 
FOR (e:ShorttermEntity) ON (e.name);

CREATE INDEX entity_type_idx IF NOT EXISTS 
FOR (e:ShorttermEntity) ON (e.type);

CREATE INDEX entity_confidence_idx IF NOT EXISTS 
FOR (e:ShorttermEntity) ON (e.confidence);
```

### 3. Use Transactions for Consistency

Wrap related operations in transactions:

```python
async with session.begin_transaction() as tx:
    # Multiple related operations
    await tx.run(query1, params1)
    await tx.run(query2, params2)
    await tx.commit()
```

### 4. Handle Temporal Validity

Always check temporal validity for longterm entities:

```cypher
WHERE e.valid_from <= datetime($now)
  AND (e.valid_until IS NULL OR e.valid_until > datetime($now))
```

### 5. Limit Result Sets

Always use LIMIT in queries to prevent memory issues:

```cypher
MATCH (e:Entity)
WHERE e.confidence > 0.7
RETURN e
ORDER BY e.confidence DESC
LIMIT 100
```

### 6. Connection Management

Properly manage driver lifecycle:

```python
# Initialize at startup
await neo4j_manager.initialize()

# Use context managers for sessions
async with get_neo4j_write_session() as session:
    # Operations
    pass

# Close at shutdown
await neo4j_manager.close()
```

### 7. Error Handling

Implement robust error handling:

```python
from neo4j.exceptions import Neo4jError

try:
    async with get_neo4j_write_session() as session:
        result = await session.run(query, **params)
except Neo4jError as e:
    logger.error(f"Neo4j error: {e.code} - {e.message}")
    raise
```

## Summary

- Use separate node labels for shortterm and longterm entities
- Store temporal information (first_seen, last_seen, valid_from, valid_until)
- Use confidence and importance scores for entity ranking
- Leverage relationships with detailed properties
- Always use parameterized queries
- Create appropriate indexes for performance
- Use transactions for data consistency
- Handle temporal validity for longterm entities

For more examples, refer to:
- `database/memories/neo4j_memory_repository.py`
- `database/sql_schema/memories.cypher`
- `database/neo4j_conn.py`

