"""
Integration tests for shortterm memory entity search functionality.

Tests the search_entities_with_relationships method with real database connections.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timezone
from typing import List

import pytest

# Add parent directory to path to import modules directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_reminiscence.config import Config
from agent_reminiscence.database.postgres_manager import PostgreSQLManager
from agent_reminiscence.database.neo4j_manager import Neo4jManager
from agent_reminiscence.database.repositories.shortterm_memory import ShorttermMemoryRepository

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@pytest.fixture
async def db_managers():
    """Initialize database managers."""
    config = Config()

    postgres = PostgreSQLManager(config)
    neo4j = Neo4jManager(config)

    await postgres.initialize()
    await neo4j.initialize()

    yield postgres, neo4j

    await postgres.close()
    await neo4j.close()


@pytest.fixture
async def repo(db_managers):
    """Create repository instance."""
    postgres, neo4j = db_managers
    return ShorttermMemoryRepository(postgres, neo4j)


@pytest.fixture
async def test_data(repo):
    """Create test data for entity search tests."""
    external_id = "test-entity-search"

    # Clean up any existing test data
    logger.info("Cleaning up existing test data...")
    async with repo.neo4j.session() as session:
        await session.run(
            "MATCH (e:ShorttermEntity {external_id: $external_id}) DETACH DELETE e",
            external_id=external_id,
        )

    # Create a shortterm memory
    logger.info("Creating test shortterm memory...")
    memory = await repo.create_memory(
        external_id=external_id,
        title="Entity Search Test Memory",
        summary="Test memory for entity search",
        metadata={"test": "entity_search"},
    )

    # Create entities
    logger.info("Creating test entities...")
    entity1 = await repo.create_entity(
        external_id=external_id,
        shortterm_memory_id=memory.id,
        name="CentralStorage",
        types=["Component", "Storage"],
        description="Central storage component",
        importance=0.9,
        metadata={"source": "test"},
    )

    entity2 = await repo.create_entity(
        external_id=external_id,
        shortterm_memory_id=memory.id,
        name="PostgreSQL",
        types=["Database", "Storage"],
        description="PostgreSQL database",
        importance=0.8,
        metadata={"source": "test"},
    )

    entity3 = await repo.create_entity(
        external_id=external_id,
        shortterm_memory_id=memory.id,
        name="Neo4j",
        types=["Database", "Graph"],
        description="Neo4j graph database",
        importance=0.8,
        metadata={"source": "test"},
    )

    entity4 = await repo.create_entity(
        external_id=external_id,
        shortterm_memory_id=memory.id,
        name="MemoryRetriever",
        types=["Agent", "Service"],
        description="Memory retriever agent",
        importance=0.7,
        metadata={"source": "test"},
    )

    # Create relationships
    logger.info("Creating test relationships...")
    rel1 = await repo.create_relationship(
        external_id=external_id,
        shortterm_memory_id=memory.id,
        from_entity_id=entity1.id,
        to_entity_id=entity2.id,
        types=["USES", "STORES_IN"],
        description="CentralStorage uses PostgreSQL",
        importance=0.8,
        metadata={"source": "test"},
    )

    rel2 = await repo.create_relationship(
        external_id=external_id,
        shortterm_memory_id=memory.id,
        from_entity_id=entity1.id,
        to_entity_id=entity3.id,
        types=["USES", "STORES_IN"],
        description="CentralStorage uses Neo4j",
        importance=0.8,
        metadata={"source": "test"},
    )

    rel3 = await repo.create_relationship(
        external_id=external_id,
        shortterm_memory_id=memory.id,
        from_entity_id=entity4.id,
        to_entity_id=entity1.id,
        types=["QUERIES", "ACCESSES"],
        description="MemoryRetriever queries CentralStorage",
        importance=0.7,
        metadata={"source": "test"},
    )

    logger.info(
        f"Created test data: memory={memory.id}, entities={[entity1.id, entity2.id, entity3.id, entity4.id]}"
    )

    return {
        "external_id": external_id,
        "memory": memory,
        "entities": {
            "storage": entity1,
            "postgres": entity2,
            "neo4j": entity3,
            "retriever": entity4,
        },
        "relationships": {
            "storage_postgres": rel1,
            "storage_neo4j": rel2,
            "retriever_storage": rel3,
        },
    }


@pytest.mark.asyncio
async def test_search_single_entity_exact_match(repo, test_data):
    """Test searching for a single entity with exact name match."""
    logger.info("\n=== Test: Single Entity Exact Match ===")

    result = await repo.search_entities_with_relationships(
        entity_names=["CentralStorage"], external_id=test_data["external_id"], limit=10
    )

    logger.info(f"Matched entities: {len(result.matched_entities)}")
    logger.info(f"Related entities: {len(result.related_entities)}")
    logger.info(f"Relationships: {len(result.relationships)}")

    # Assertions
    assert len(result.matched_entities) == 1, "Should find exactly 1 matched entity"
    assert result.matched_entities[0].name == "CentralStorage"

    # Should have 2 related entities (PostgreSQL and Neo4j as outgoing, MemoryRetriever as incoming)
    assert (
        len(result.related_entities) >= 2
    ), f"Should find at least 2 related entities, got {len(result.related_entities)}"

    # Should have 3 relationships total (2 outgoing, 1 incoming)
    assert (
        len(result.relationships) == 3
    ), f"Should find 3 relationships, got {len(result.relationships)}"

    # Log details
    for entity in result.matched_entities:
        logger.info(f"  Matched: {entity.name} (importance={entity.importance})")
    for entity in result.related_entities:
        logger.info(f"  Related: {entity.name} (importance={entity.importance})")
    for rel in result.relationships:
        logger.info(f"  Relationship: {rel.from_entity_name} -> {rel.to_entity_name} ({rel.types})")


@pytest.mark.asyncio
async def test_search_partial_name_match(repo, test_data):
    """Test searching with partial name matching (case-insensitive)."""
    logger.info("\n=== Test: Partial Name Match ===")

    result = await repo.search_entities_with_relationships(
        entity_names=["storage"],  # lowercase partial match
        external_id=test_data["external_id"],
        limit=10,
    )

    logger.info(f"Matched entities: {len(result.matched_entities)}")
    logger.info(f"Related entities: {len(result.related_entities)}")
    logger.info(f"Relationships: {len(result.relationships)}")

    # Should find CentralStorage
    assert len(result.matched_entities) >= 1, "Should find at least 1 entity with 'storage' in name"

    storage_found = any(e.name == "CentralStorage" for e in result.matched_entities)
    assert storage_found, "Should find CentralStorage entity"

    for entity in result.matched_entities:
        logger.info(f"  Matched: {entity.name}")


@pytest.mark.asyncio
async def test_search_multiple_entities(repo, test_data):
    """Test searching for multiple entities at once."""
    logger.info("\n=== Test: Multiple Entity Search ===")

    result = await repo.search_entities_with_relationships(
        entity_names=["PostgreSQL", "Neo4j"], external_id=test_data["external_id"], limit=10
    )

    logger.info(f"Matched entities: {len(result.matched_entities)}")
    logger.info(f"Related entities: {len(result.related_entities)}")
    logger.info(f"Relationships: {len(result.relationships)}")

    # Should find both databases
    assert (
        len(result.matched_entities) == 2
    ), f"Should find 2 matched entities, got {len(result.matched_entities)}"

    matched_names = {e.name for e in result.matched_entities}
    assert "PostgreSQL" in matched_names, "Should find PostgreSQL"
    assert "Neo4j" in matched_names, "Should find Neo4j"

    # Should have CentralStorage as related entity
    related_names = {e.name for e in result.related_entities}
    assert "CentralStorage" in related_names, "Should find CentralStorage as related entity"

    # Should have 2 relationships (incoming from CentralStorage)
    assert (
        len(result.relationships) == 2
    ), f"Should find 2 relationships, got {len(result.relationships)}"

    for entity in result.matched_entities:
        logger.info(f"  Matched: {entity.name}")
    for entity in result.related_entities:
        logger.info(f"  Related: {entity.name}")


@pytest.mark.asyncio
async def test_search_with_importance_filter(repo, test_data):
    """Test searching with minimum importance threshold."""
    logger.info("\n=== Test: Importance Filter ===")

    # Search with high importance threshold (should only get CentralStorage)
    result = await repo.search_entities_with_relationships(
        entity_names=["storage", "retriever"],
        external_id=test_data["external_id"],
        min_importance=0.85,
        limit=10,
    )

    logger.info(f"Matched entities (min_importance=0.85): {len(result.matched_entities)}")

    # Only CentralStorage has importance >= 0.85
    assert len(result.matched_entities) == 1, "Should find only 1 entity with importance >= 0.85"
    assert result.matched_entities[0].name == "CentralStorage"

    # Search with lower threshold
    result_low = await repo.search_entities_with_relationships(
        entity_names=["storage", "retriever"],
        external_id=test_data["external_id"],
        min_importance=0.6,
        limit=10,
    )

    logger.info(f"Matched entities (min_importance=0.6): {len(result_low.matched_entities)}")

    # Should get both entities
    assert len(result_low.matched_entities) == 2, "Should find 2 entities with importance >= 0.6"


@pytest.mark.asyncio
async def test_search_with_memory_filter(repo, test_data):
    """Test searching with specific memory ID filter."""
    logger.info("\n=== Test: Memory ID Filter ===")

    # Search with correct memory ID
    result = await repo.search_entities_with_relationships(
        entity_names=["CentralStorage"],
        external_id=test_data["external_id"],
        shortterm_memory_id=test_data["memory"].id,
        limit=10,
    )

    logger.info(f"Matched entities (correct memory): {len(result.matched_entities)}")
    assert len(result.matched_entities) == 1, "Should find entity in correct memory"

    # Search with wrong memory ID
    result_wrong = await repo.search_entities_with_relationships(
        entity_names=["CentralStorage"],
        external_id=test_data["external_id"],
        shortterm_memory_id=99999,
        limit=10,
    )

    logger.info(f"Matched entities (wrong memory): {len(result_wrong.matched_entities)}")
    assert len(result_wrong.matched_entities) == 0, "Should find no entities with wrong memory ID"


@pytest.mark.asyncio
async def test_search_no_results(repo, test_data):
    """Test searching for non-existent entity."""
    logger.info("\n=== Test: No Results ===")

    result = await repo.search_entities_with_relationships(
        entity_names=["NonExistentEntity"], external_id=test_data["external_id"], limit=10
    )

    logger.info(f"Matched entities: {len(result.matched_entities)}")
    logger.info(f"Related entities: {len(result.related_entities)}")
    logger.info(f"Relationships: {len(result.relationships)}")

    assert len(result.matched_entities) == 0, "Should find no matched entities"
    assert len(result.related_entities) == 0, "Should find no related entities"
    assert len(result.relationships) == 0, "Should find no relationships"


@pytest.mark.asyncio
async def test_search_relationship_directions(repo, test_data):
    """Test that both incoming and outgoing relationships are captured."""
    logger.info("\n=== Test: Relationship Directions ===")

    result = await repo.search_entities_with_relationships(
        entity_names=["CentralStorage"], external_id=test_data["external_id"], limit=10
    )

    # Check relationship directions
    outgoing = [r for r in result.relationships if r.from_entity_name == "CentralStorage"]
    incoming = [r for r in result.relationships if r.to_entity_name == "CentralStorage"]

    logger.info(f"Outgoing relationships: {len(outgoing)}")
    logger.info(f"Incoming relationships: {len(incoming)}")

    assert len(outgoing) == 2, "Should have 2 outgoing relationships from CentralStorage"
    assert len(incoming) == 1, "Should have 1 incoming relationship to CentralStorage"

    # Verify outgoing targets
    outgoing_targets = {r.to_entity_name for r in outgoing}
    assert "PostgreSQL" in outgoing_targets, "Should have relationship to PostgreSQL"
    assert "Neo4j" in outgoing_targets, "Should have relationship to Neo4j"

    # Verify incoming source
    incoming_sources = {r.from_entity_name for r in incoming}
    assert "MemoryRetriever" in incoming_sources, "Should have relationship from MemoryRetriever"


@pytest.mark.asyncio
async def test_search_metadata_parsing(repo, test_data):
    """Test that metadata is correctly parsed from JSON strings."""
    logger.info("\n=== Test: Metadata Parsing ===")

    result = await repo.search_entities_with_relationships(
        entity_names=["CentralStorage"], external_id=test_data["external_id"], limit=10
    )

    # Check matched entity metadata
    assert len(result.matched_entities) == 1
    entity = result.matched_entities[0]

    logger.info(f"Entity metadata type: {type(entity.metadata)}")
    logger.info(f"Entity metadata: {entity.metadata}")

    assert isinstance(entity.metadata, dict), "Metadata should be a dict, not a string"
    assert entity.metadata.get("source") == "test", "Metadata should contain source=test"

    # Check relationship metadata
    if result.relationships:
        rel = result.relationships[0]
        logger.info(f"Relationship metadata type: {type(rel.metadata)}")
        logger.info(f"Relationship metadata: {rel.metadata}")

        assert isinstance(rel.metadata, dict), "Relationship metadata should be a dict"
        assert (
            rel.metadata.get("source") == "test"
        ), "Relationship metadata should contain source=test"


@pytest.mark.asyncio
async def test_search_limit(repo, test_data):
    """Test that limit parameter works correctly."""
    logger.info("\n=== Test: Search Limit ===")

    # Search with limit=1
    result = await repo.search_entities_with_relationships(
        entity_names=["storage", "postgres", "neo4j", "retriever"],
        external_id=test_data["external_id"],
        limit=1,
    )

    logger.info(f"Matched entities with limit=1: {len(result.matched_entities)}")
    assert len(result.matched_entities) <= 1, "Should respect limit of 1"

    # Search with limit=2
    result = await repo.search_entities_with_relationships(
        entity_names=["storage", "postgres", "neo4j", "retriever"],
        external_id=test_data["external_id"],
        limit=2,
    )

    logger.info(f"Matched entities with limit=2: {len(result.matched_entities)}")
    assert len(result.matched_entities) <= 2, "Should respect limit of 2"


if __name__ == "__main__":
    """Run tests directly without pytest."""

    async def run_tests():
        config = Config()

        postgres = PostgreSQLManager(config)
        neo4j = Neo4jManager(config)

        await postgres.initialize()
        await neo4j.initialize()

        repo = ShorttermMemoryRepository(postgres, neo4j)

        try:
            # Create test data
            logger.info("Setting up test data...")
            external_id = "test-entity-search"

            # Clean up
            async with neo4j.session() as session:
                await session.run(
                    "MATCH (e:ShorttermEntity {external_id: $external_id}) DETACH DELETE e",
                    external_id=external_id,
                )

            # Create memory
            memory = await repo.create_memory(
                external_id=external_id,
                title="Entity Search Test Memory",
                summary="Test memory for entity search",
                metadata={"test": "entity_search"},
            )

            # Create entities
            entity1 = await repo.create_entity(
                external_id=external_id,
                shortterm_memory_id=memory.id,
                name="CentralStorage",
                types=["Component", "Storage"],
                description="Central storage component",
                importance=0.9,
                metadata={"source": "test"},
            )

            entity2 = await repo.create_entity(
                external_id=external_id,
                shortterm_memory_id=memory.id,
                name="PostgreSQL",
                types=["Database", "Storage"],
                description="PostgreSQL database",
                importance=0.8,
                metadata={"source": "test"},
            )

            entity3 = await repo.create_entity(
                external_id=external_id,
                shortterm_memory_id=memory.id,
                name="Neo4j",
                types=["Database", "Graph"],
                description="Neo4j graph database",
                importance=0.8,
                metadata={"source": "test"},
            )

            entity4 = await repo.create_entity(
                external_id=external_id,
                shortterm_memory_id=memory.id,
                name="MemoryRetriever",
                types=["Agent", "Service"],
                description="Memory retriever agent",
                importance=0.7,
                metadata={"source": "test"},
            )

            # Create relationships
            await repo.create_relationship(
                external_id=external_id,
                shortterm_memory_id=memory.id,
                from_entity_id=entity1.id,
                to_entity_id=entity2.id,
                types=["USES", "STORES_IN"],
                description="CentralStorage uses PostgreSQL",
                importance=0.8,
                metadata={"source": "test"},
            )

            await repo.create_relationship(
                external_id=external_id,
                shortterm_memory_id=memory.id,
                from_entity_id=entity1.id,
                to_entity_id=entity3.id,
                types=["USES", "STORES_IN"],
                description="CentralStorage uses Neo4j",
                importance=0.8,
                metadata={"source": "test"},
            )

            await repo.create_relationship(
                external_id=external_id,
                shortterm_memory_id=memory.id,
                from_entity_id=entity4.id,
                to_entity_id=entity1.id,
                types=["QUERIES", "ACCESSES"],
                description="MemoryRetriever queries CentralStorage",
                importance=0.7,
                metadata={"source": "test"},
            )

            logger.info("✅ Test data created successfully")

            # Run test searches
            logger.info("\n" + "=" * 60)
            logger.info("TEST 1: Search for CentralStorage")
            logger.info("=" * 60)
            result = await repo.search_entities_with_relationships(
                entity_names=["CentralStorage"], external_id=external_id, limit=10
            )
            logger.info(f"✅ Found {len(result.matched_entities)} matched entities")
            logger.info(f"✅ Found {len(result.related_entities)} related entities")
            logger.info(f"✅ Found {len(result.relationships)} relationships")

            for entity in result.matched_entities:
                logger.info(
                    f"  Matched: {entity.name} (importance={entity.importance}, metadata={entity.metadata})"
                )
            for entity in result.related_entities:
                logger.info(f"  Related: {entity.name}")
            for rel in result.relationships:
                logger.info(
                    f"  Relationship: {rel.from_entity_name} -> {rel.to_entity_name} ({rel.types})"
                )

            logger.info("\n" + "=" * 60)
            logger.info("TEST 2: Search with partial name 'storage'")
            logger.info("=" * 60)
            result = await repo.search_entities_with_relationships(
                entity_names=["storage"], external_id=external_id, limit=10
            )
            logger.info(f"✅ Found {len(result.matched_entities)} matched entities")
            for entity in result.matched_entities:
                logger.info(f"  Matched: {entity.name}")

            logger.info("\n" + "=" * 60)
            logger.info("TEST 3: Search for multiple entities")
            logger.info("=" * 60)
            result = await repo.search_entities_with_relationships(
                entity_names=["PostgreSQL", "Neo4j"], external_id=external_id, limit=10
            )
            logger.info(f"✅ Found {len(result.matched_entities)} matched entities")
            logger.info(f"✅ Found {len(result.related_entities)} related entities")
            logger.info(f"✅ Found {len(result.relationships)} relationships")

            logger.info("\n🎉 All tests completed successfully!")

        finally:
            await postgres.close()
            await neo4j.close()

    asyncio.run(run_tests())


