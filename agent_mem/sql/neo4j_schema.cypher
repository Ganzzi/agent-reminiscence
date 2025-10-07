// ============================================================================
// Neo4j Schema for AgentMem
// ============================================================================
// This file defines the graph schema for entities and relationships
// in both shortterm and longterm memory tiers.
//
// Usage:
//   cat agent_mem/sql/neo4j_schema.cypher | cypher-shell -u neo4j -p password
//
// Or in Neo4j Browser:
//   Copy and paste sections into the query editor
// ============================================================================

// ============================================================================
// INDEXES
// ============================================================================

// Shortterm Entity Indexes
CREATE INDEX shortterm_entity_external_id IF NOT EXISTS
FOR (e:ShorttermEntity)
ON (e.external_id);

CREATE INDEX shortterm_entity_name IF NOT EXISTS
FOR (e:ShorttermEntity)
ON (e.name);

CREATE INDEX shortterm_entity_composite IF NOT EXISTS
FOR (e:ShorttermEntity)
ON (e.external_id, e.name);

CREATE INDEX shortterm_entity_memory_id IF NOT EXISTS
FOR (e:ShorttermEntity)
ON (e.shortterm_memory_id);

// Longterm Entity Indexes
CREATE INDEX longterm_entity_external_id IF NOT EXISTS
FOR (e:LongtermEntity)
ON (e.external_id);

CREATE INDEX longterm_entity_name IF NOT EXISTS
FOR (e:LongtermEntity)
ON (e.name);

CREATE INDEX longterm_entity_composite IF NOT EXISTS
FOR (e:LongtermEntity)
ON (e.external_id, e.name);

// Relationship Indexes
CREATE INDEX shortterm_relationship_external_id IF NOT EXISTS
FOR ()-[r:SHORTTERM_RELATES]-()
ON (r.external_id);

CREATE INDEX shortterm_relationship_memory_id IF NOT EXISTS
FOR ()-[r:SHORTTERM_RELATES]-()
ON (r.shortterm_memory_id);

CREATE INDEX longterm_relationship_external_id IF NOT EXISTS
FOR ()-[r:LONGTERM_RELATES]-()
ON (r.external_id);

// ============================================================================
// CONSTRAINTS
// ============================================================================

// Ensure unique entity names per agent in shortterm
CREATE CONSTRAINT shortterm_entity_unique IF NOT EXISTS
FOR (e:ShorttermEntity)
REQUIRE (e.external_id, e.shortterm_memory_id, e.name) IS UNIQUE;

// Ensure unique entity names per agent in longterm
CREATE CONSTRAINT longterm_entity_unique IF NOT EXISTS
FOR (e:LongtermEntity)
REQUIRE (e.external_id, e.name) IS UNIQUE;

// ============================================================================
// SCHEMA DOCUMENTATION
// ============================================================================

// NODE LABELS
// -----------

// :ShorttermEntity
// Properties:
//   - external_id: String (agent identifier)
//   - shortterm_memory_id: Integer (reference to shortterm memory)
//   - name: String (entity name, e.g., "John Doe")
//   - types: List<String> (entity types, e.g., ["PERSON", "DEVELOPER"])
//   - description: String (optional, entity description)
//   - confidence: Float (0.0-1.0, confidence score)
//   - first_seen: DateTime (ISO 8601 timestamp)
//   - last_seen: DateTime (ISO 8601 timestamp)
//   - metadata: String (JSON string of additional properties)
// Note: Use elementId(e) to get the node's ID, not the deprecated id(e) function

// :LongtermEntity
// Properties:
//   - external_id: String (agent identifier)
//   - name: String (entity name)
//   - types: List<String> (entity types)
//   - description: String (optional, entity description)
//   - confidence: Float (0.0-1.0, confidence score)
//   - importance: Float (0.0-1.0, importance score)
//   - first_seen: DateTime (ISO 8601 timestamp)
//   - last_seen: DateTime (ISO 8601 timestamp)
//   - metadata: String (JSON string of additional properties)
// Note: Use elementId(e) to get the node's ID, not the deprecated id(e) function

// RELATIONSHIP TYPES
// ------------------

// :SHORTTERM_RELATES
// Connects (:ShorttermEntity)-[:SHORTTERM_RELATES]->(:ShorttermEntity)
// Properties:
//   - external_id: String (agent identifier)
//   - shortterm_memory_id: Integer (reference to shortterm memory)
//   - types: List<String> (relationship types, e.g., ["WORKS_WITH", "COLLABORATES"])
//   - description: String (optional, relationship description)
//   - confidence: Float (0.0-1.0, confidence score)
//   - strength: Float (0.0-1.0, relationship strength)
//   - first_observed: DateTime (ISO 8601 timestamp)
//   - last_observed: DateTime (ISO 8601 timestamp)
//   - metadata: String (JSON string of additional properties)
// Note: Use elementId(r) to get the relationship's ID, not the deprecated id(r) function

// :LONGTERM_RELATES
// Connects (:LongtermEntity)-[:LONGTERM_RELATES]->(:LongtermEntity)
// Properties:
//   - external_id: String (agent identifier)
//   - types: List<String> (relationship types)
//   - description: String (optional, relationship description)
//   - confidence: Float (0.0-1.0, confidence score)
//   - strength: Float (0.0-1.0, relationship strength)
//   - importance: Float (0.0-1.0, importance score for prioritization)
//   - start_date: DateTime (ISO 8601 timestamp, when relationship became valid)
//   - last_updated: DateTime (ISO 8601 timestamp, last time relationship was updated)
//   - metadata: String (JSON string of additional properties)
// Note: Use elementId(r) to get the relationship's ID, not the deprecated id(r) function

// ============================================================================
// EXAMPLE QUERIES
// ============================================================================

// Find all entities for an agent
// MATCH (e:ShorttermEntity {external_id: 'agent-123'})
// RETURN e;

// Find entities by type
// MATCH (e:ShorttermEntity {external_id: 'agent-123'})
// WHERE 'PERSON' IN e.types
// RETURN e;

// Find all entities with multiple types
// MATCH (e:ShorttermEntity {external_id: 'agent-123'})
// WHERE size(e.types) > 1
// RETURN e.name, e.types;

// Find relationships between entities
// MATCH (e1:ShorttermEntity {external_id: 'agent-123'})-[r:SHORTTERM_RELATES]-(e2:ShorttermEntity)
// RETURN e1, r, e2;

// Find relationships by type
// MATCH (e1:ShorttermEntity)-[r:SHORTTERM_RELATES]-(e2:ShorttermEntity)
// WHERE r.external_id = 'agent-123' AND 'WORKS_WITH' IN r.types
// RETURN e1, r, e2;

// Get entity with all relationships
// MATCH (e:ShorttermEntity {external_id: 'agent-123', name: 'John Doe'})
// OPTIONAL MATCH (e)-[r:SHORTTERM_RELATES]-(other)
// RETURN e, collect(r) AS relationships, collect(other) AS connected_entities;

// Find entities with high confidence
// MATCH (e:ShorttermEntity {external_id: 'agent-123'})
// WHERE e.confidence > 0.8
// RETURN e.name, e.types, e.confidence
// ORDER BY e.confidence DESC;

// Find most important longterm entities
// MATCH (e:LongtermEntity {external_id: 'agent-123'})
// WHERE e.importance > 0.7
// RETURN e.name, e.types, e.importance, e.confidence
// ORDER BY e.importance DESC
// LIMIT 10;

// Find entities connected to a specific entity
// MATCH (e:ShorttermEntity {external_id: 'agent-123', name: 'Python'})
// MATCH (e)-[r:SHORTTERM_RELATES]-(connected)
// RETURN connected.name, r.types, r.strength
// ORDER BY r.strength DESC;

// Count entities by type
// MATCH (e:ShorttermEntity {external_id: 'agent-123'})
// UNWIND e.types AS type
// RETURN type, count(*) AS count
// ORDER BY count DESC;

// Find entities with specific type combinations
// MATCH (e:ShorttermEntity {external_id: 'agent-123'})
// WHERE 'PERSON' IN e.types AND 'DEVELOPER' IN e.types
// RETURN e.name, e.types, e.confidence;

// ============================================================================
// MIGRATION FROM SINGLE TYPE TO ARRAY
// ============================================================================
// If migrating from old schema where 'type' was a single string:

// Migrate shortterm entities
// MATCH (e:ShorttermEntity)
// WHERE e.type IS NOT NULL AND e.types IS NULL
// SET e.types = [e.type]
// REMOVE e.type
// RETURN count(e) AS migrated_entities;

// Migrate shortterm relationships
// MATCH ()-[r:SHORTTERM_RELATES]-()
// WHERE r.type IS NOT NULL AND r.types IS NULL
// SET r.types = [r.type]
// REMOVE r.type
// RETURN count(r) AS migrated_relationships;

// Migrate longterm entities
// MATCH (e:LongtermEntity)
// WHERE e.type IS NOT NULL AND e.types IS NULL
// SET e.types = [e.type]
// REMOVE e.type
// RETURN count(e) AS migrated_entities;

// Migrate longterm relationships
// MATCH ()-[r:LONGTERM_RELATES]-()
// WHERE r.type IS NOT NULL AND r.types IS NULL
// SET r.types = [r.type]
// REMOVE r.type
// RETURN count(r) AS migrated_relationships;

// ============================================================================
// CLEANUP (Development/Testing Only)
// ============================================================================
// WARNING: These queries delete all data! Use with caution in development only.

// Delete all shortterm entities and relationships
// MATCH (e:ShorttermEntity)
// DETACH DELETE e;

// Delete all longterm entities and relationships
// MATCH (e:LongtermEntity)
// DETACH DELETE e;

// Delete entities for specific agent
// MATCH (e:ShorttermEntity {external_id: 'agent-123'})
// DETACH DELETE e;

// Delete entities from specific shortterm memory
// MATCH (e:ShorttermEntity {shortterm_memory_id: 1})
// DETACH DELETE e;

// ============================================================================
// PERFORMANCE TIPS
// ============================================================================

// 1. Always filter by external_id first (indexed)
//    ✅ MATCH (e:ShorttermEntity {external_id: 'agent-123'})
//    ❌ MATCH (e:ShorttermEntity) WHERE e.name = 'John'

// 2. Use composite indexes for better performance
//    ✅ MATCH (e:ShorttermEntity {external_id: 'agent-123', name: 'John'})

// 3. Use PROFILE or EXPLAIN to analyze query performance
//    PROFILE MATCH (e:ShorttermEntity {external_id: 'agent-123'}) RETURN e;

// 4. For type filtering, use IN operator
//    ✅ WHERE 'PERSON' IN e.types
//    ❌ WHERE e.types CONTAINS 'PERSON'

// 5. Limit result sets in production
//    MATCH (e:ShorttermEntity {external_id: 'agent-123'})
//    RETURN e
//    LIMIT 100;

// ============================================================================
// VERIFICATION
// ============================================================================

// Show all indexes
// SHOW INDEXES;

// Show all constraints
// SHOW CONSTRAINTS;

// Count all entities
// MATCH (e:ShorttermEntity)
// RETURN count(e) AS shortterm_entities
// UNION ALL
// MATCH (e:LongtermEntity)
// RETURN count(e) AS longterm_entities;

// Count all relationships
// MATCH ()-[r:SHORTTERM_RELATES]-()
// RETURN count(r) AS shortterm_relationships
// UNION ALL
// MATCH ()-[r:LONGTERM_RELATES]-()
// RETURN count(r) AS longterm_relationships;

// Show schema
// CALL db.schema.visualization();

// ============================================================================
// END OF SCHEMA
// ============================================================================
