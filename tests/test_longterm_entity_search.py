"""
Integration tests for longterm memory entity search functionality.

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

from agent_mem.config import Config
from agent_mem.database.postgres_manager import PostgreSQLManager
from agent_mem.database.neo4j_manager import Neo4jManager
from agent_mem.database.repositories.longterm_memory import LongtermMemoryRepository

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
    return LongtermMemoryRepository(postgres, neo4j)


@pytest.fixture
async def test_data(repo):
    """Create test data for entity search tests."""
    external_id = "test-longterm-entity-search"

    # Clean up any existing test data
    logger.info("Cleaning up existing test data...")
    async with repo.neo4j.session() as session:
        await session.run(
            "MATCH (e:LongtermEntity {external_id: $external_id}) DETACH DELETE e",
            external_id=external_id,
        )

    # Create entities
    logger.info("Creating test entities...")
    entity1 = await repo.create_entity(
        external_id=external_id,
        name="Multi-tier Architecture",
        types=["Architecture", "Design Pattern"],
        description="Layered software architecture pattern",
        importance=0.9,
        metadata={"source": "test", "domain": "architecture"},
    )

    entity2 = await repo.create_entity(
        external_id=external_id,
        name="Vector Embeddings",
        types=["Technology", "AI"],
        description="Dense vector representations for semantic search",
        importance=0.85,
        metadata={"source": "test", "domain": "ml"},
    )

    entity3 = await repo.create_entity(
        external_id=external_id,
        name="Token Optimization",
        types=["Technique", "Performance"],
        description="Reducing token usage in LLM interactions",
        importance=0.8,
        metadata={"source": "test", "domain": "optimization"},
    )

    entity4 = await repo.create_entity(
        external_id=external_id,
        name="Pydantic AI",
        types=["Framework", "Library"],
        description="Type-safe AI framework for Python",
        importance=0.75,
        metadata={"source": "test", "domain": "tools"},
    )

    entity5 = await repo.create_entity(
        external_id=external_id,
        name="Google Gemini",
        types=["LLM", "Service"],
        description="Google's multimodal AI model",
        importance=0.7,
        metadata={"source": "test", "domain": "llm"},
    )

    # Create relationships
    logger.info("Creating test relationships...")
    rel1 = await repo.create_relationship(
        external_id=external_id,
        from_entity_id=entity1.id,
        to_entity_id=entity2.id,
        types=["USES", "IMPLEMENTS"],
        description="Architecture uses vector embeddings for semantic search",
        importance=0.85,
        metadata={"source": "test"},
    )

    rel2 = await repo.create_relationship(
        external_id=external_id,
        from_entity_id=entity1.id,
        to_entity_id=entity3.id,
        types=["REQUIRES", "APPLIES"],
        description="Architecture requires token optimization",
        importance=0.8,
        metadata={"source": "test"},
    )

    rel3 = await repo.create_relationship(
        external_id=external_id,
        from_entity_id=entity4.id,
        to_entity_id=entity5.id,
        types=["INTEGRATES_WITH", "SUPPORTS"],
        description="Pydantic AI integrates with Google Gemini",
        importance=0.75,
        metadata={"source": "test"},
    )

    rel4 = await repo.create_relationship(
        external_id=external_id,
        from_entity_id=entity3.id,
        to_entity_id=entity5.id,
        types=["OPTIMIZES", "REDUCES_COST"],
        description="Token optimization reduces Gemini API costs",
        importance=0.7,
        metadata={"source": "test"},
    )

    logger.info(
        f"Created test data: entities={[entity1.id, entity2.id, entity3.id, entity4.id, entity5.id]}"
    )

    return {
        "external_id": external_id,
        "entities": {
            "architecture": entity1,
            "embeddings": entity2,
            "optimization": entity3,
            "pydantic": entity4,
            "gemini": entity5,
        },
        "relationships": {
            "arch_embeddings": rel1,
            "arch_optimization": rel2,
            "pydantic_gemini": rel3,
            "optimization_gemini": rel4,
        },
    }


@pytest.mark.asyncio
async def test_search_single_entity_exact_match(repo, test_data):
    """Test searching for a single entity with exact name match."""
    logger.info("\n=== Test: Single Entity Exact Match ===")

    result = await repo.search_entities_with_relationships(
        entity_names=["Multi-tier Architecture"], external_id=test_data["external_id"], limit=10
    )

    logger.info(f"Matched entities: {len(result.matched_entities)}")
    logger.info(f"Related entities: {len(result.related_entities)}")
    logger.info(f"Relationships: {len(result.relationships)}")

    # Assertions
    assert len(result.matched_entities) == 1, "Should find exactly 1 matched entity"
    assert result.matched_entities[0].name == "Multi-tier Architecture"

    # Should have 2 related entities (Vector Embeddings and Token Optimization)
    assert (
        len(result.related_entities) >= 2
    ), f"Should find at least 2 related entities, got {len(result.related_entities)}"

    # Should have 2 relationships (2 outgoing)
    assert (
        len(result.relationships) == 2
    ), f"Should find 2 relationships, got {len(result.relationships)}"

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
        entity_names=["optimization"],  # lowercase partial match
        external_id=test_data["external_id"],
        limit=10,
    )

    logger.info(f"Matched entities: {len(result.matched_entities)}")
    logger.info(f"Related entities: {len(result.related_entities)}")
    logger.info(f"Relationships: {len(result.relationships)}")

    # Should find Token Optimization
    assert (
        len(result.matched_entities) >= 1
    ), "Should find at least 1 entity with 'optimization' in name"

    optimization_found = any(e.name == "Token Optimization" for e in result.matched_entities)
    assert optimization_found, "Should find Token Optimization entity"

    for entity in result.matched_entities:
        logger.info(f"  Matched: {entity.name}")


@pytest.mark.asyncio
async def test_search_multiple_entities(repo, test_data):
    """Test searching for multiple entities at once."""
    logger.info("\n=== Test: Multiple Entity Search ===")

    result = await repo.search_entities_with_relationships(
        entity_names=["Pydantic", "Gemini"], external_id=test_data["external_id"], limit=10
    )

    logger.info(f"Matched entities: {len(result.matched_entities)}")
    logger.info(f"Related entities: {len(result.related_entities)}")
    logger.info(f"Relationships: {len(result.relationships)}")

    # Should find both
    assert (
        len(result.matched_entities) == 2
    ), f"Should find 2 matched entities, got {len(result.matched_entities)}"

    matched_names = {e.name for e in result.matched_entities}
    assert "Pydantic AI" in matched_names, "Should find Pydantic AI"
    assert "Google Gemini" in matched_names, "Should find Google Gemini"

    # Should have 1 relationship between them
    assert (
        len(result.relationships) >= 1
    ), f"Should find at least 1 relationship, got {len(result.relationships)}"

    for entity in result.matched_entities:
        logger.info(f"  Matched: {entity.name}")
    for entity in result.related_entities:
        logger.info(f"  Related: {entity.name}")


@pytest.mark.asyncio
async def test_search_with_importance_filter(repo, test_data):
    """Test searching with minimum importance threshold."""
    logger.info("\n=== Test: Importance Filter ===")

    # Search with high importance threshold (should only get Multi-tier Architecture and Vector Embeddings)
    result = await repo.search_entities_with_relationships(
        entity_names=["architecture", "embeddings", "optimization"],
        external_id=test_data["external_id"],
        min_importance=0.83,
        limit=10,
    )

    logger.info(f"Matched entities (min_importance=0.83): {len(result.matched_entities)}")

    # Only Multi-tier Architecture (0.9) and Vector Embeddings (0.85) have importance >= 0.83
    assert (
        len(result.matched_entities) >= 1
    ), "Should find at least 1 entity with importance >= 0.83"

    # Search with lower threshold
    result_low = await repo.search_entities_with_relationships(
        entity_names=["architecture", "embeddings", "optimization"],
        external_id=test_data["external_id"],
        min_importance=0.7,
        limit=10,
    )

    logger.info(f"Matched entities (min_importance=0.7): {len(result_low.matched_entities)}")

    # Should get all three entities
    assert len(result_low.matched_entities) == 3, "Should find 3 entities with importance >= 0.7"


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
        entity_names=["Google Gemini"], external_id=test_data["external_id"], limit=10
    )

    # Check relationship directions
    outgoing = [r for r in result.relationships if r.from_entity_name == "Google Gemini"]
    incoming = [r for r in result.relationships if r.to_entity_name == "Google Gemini"]

    logger.info(f"Outgoing relationships: {len(outgoing)}")
    logger.info(f"Incoming relationships: {len(incoming)}")

    # Google Gemini should have 2 incoming relationships
    assert (
        len(incoming) == 2
    ), f"Should have 2 incoming relationships to Google Gemini, got {len(incoming)}"
    assert (
        len(outgoing) == 0
    ), f"Should have 0 outgoing relationships from Google Gemini, got {len(outgoing)}"

    # Verify incoming sources
    incoming_sources = {r.from_entity_name for r in incoming}
    assert "Pydantic AI" in incoming_sources, "Should have relationship from Pydantic AI"
    assert (
        "Token Optimization" in incoming_sources
    ), "Should have relationship from Token Optimization"


@pytest.mark.asyncio
async def test_search_metadata_parsing(repo, test_data):
    """Test that metadata is correctly parsed from JSON strings."""
    logger.info("\n=== Test: Metadata Parsing ===")

    result = await repo.search_entities_with_relationships(
        entity_names=["Multi-tier Architecture"], external_id=test_data["external_id"], limit=10
    )

    # Check matched entity metadata
    assert len(result.matched_entities) == 1
    entity = result.matched_entities[0]

    logger.info(f"Entity metadata type: {type(entity.metadata)}")
    logger.info(f"Entity metadata: {entity.metadata}")

    assert isinstance(entity.metadata, dict), "Metadata should be a dict, not a string"
    assert entity.metadata.get("source") == "test", "Metadata should contain source=test"
    assert (
        entity.metadata.get("domain") == "architecture"
    ), "Metadata should contain domain=architecture"

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
        entity_names=["architecture", "embeddings", "optimization", "pydantic", "gemini"],
        external_id=test_data["external_id"],
        limit=1,
    )

    logger.info(f"Matched entities with limit=1: {len(result.matched_entities)}")
    assert len(result.matched_entities) <= 1, "Should respect limit of 1"

    # Search with limit=3
    result = await repo.search_entities_with_relationships(
        entity_names=["architecture", "embeddings", "optimization", "pydantic", "gemini"],
        external_id=test_data["external_id"],
        limit=3,
    )

    logger.info(f"Matched entities with limit=3: {len(result.matched_entities)}")
    assert len(result.matched_entities) <= 3, "Should respect limit of 3"


@pytest.mark.asyncio
async def test_search_complex_graph(repo, test_data):
    """Test searching in a more complex graph with multiple hops."""
    logger.info("\n=== Test: Complex Graph Search ===")

    # Search for Token Optimization, which connects to both Architecture and Gemini
    result = await repo.search_entities_with_relationships(
        entity_names=["Token Optimization"], external_id=test_data["external_id"], limit=10
    )

    logger.info(f"Matched entities: {len(result.matched_entities)}")
    logger.info(f"Related entities: {len(result.related_entities)}")
    logger.info(f"Relationships: {len(result.relationships)}")

    # Should find Token Optimization
    assert len(result.matched_entities) == 1, "Should find exactly 1 matched entity"
    assert result.matched_entities[0].name == "Token Optimization"

    # Should have 2 relationships (1 incoming from Architecture, 1 outgoing to Gemini)
    assert (
        len(result.relationships) == 2
    ), f"Should find 2 relationships, got {len(result.relationships)}"

    # Check both directions exist
    has_incoming = any(r.to_entity_name == "Token Optimization" for r in result.relationships)
    has_outgoing = any(r.from_entity_name == "Token Optimization" for r in result.relationships)

    assert has_incoming, "Should have at least one incoming relationship"
    assert has_outgoing, "Should have at least one outgoing relationship"


if __name__ == "__main__":
    """Run tests directly without pytest."""

    async def run_tests():
        config = Config()

        postgres = PostgreSQLManager(config)
        neo4j = Neo4jManager(config)

        await postgres.initialize()
        await neo4j.initialize()

        repo = LongtermMemoryRepository(postgres, neo4j)

        try:
            # Create test data
            logger.info("Setting up test data...")
            external_id = "test-longterm-entity-search"

            # Clean up
            async with neo4j.session() as session:
                await session.run(
                    "MATCH (e:LongtermEntity {external_id: $external_id}) DETACH DELETE e",
                    external_id=external_id,
                )

            # Create entities
            entity1 = await repo.create_entity(
                external_id=external_id,
                name="Multi-tier Architecture",
                types=["Architecture", "Design Pattern"],
                description="Layered software architecture pattern",
                importance=0.9,
                metadata={"source": "test", "domain": "architecture"},
            )

            entity2 = await repo.create_entity(
                external_id=external_id,
                name="Vector Embeddings",
                types=["Technology", "AI"],
                description="Dense vector representations for semantic search",
                importance=0.85,
                metadata={"source": "test", "domain": "ml"},
            )

            entity3 = await repo.create_entity(
                external_id=external_id,
                name="Token Optimization",
                types=["Technique", "Performance"],
                description="Reducing token usage in LLM interactions",
                importance=0.8,
                metadata={"source": "test", "domain": "optimization"},
            )

            entity4 = await repo.create_entity(
                external_id=external_id,
                name="Pydantic AI",
                types=["Framework", "Library"],
                description="Type-safe AI framework for Python",
                importance=0.75,
                metadata={"source": "test", "domain": "tools"},
            )

            entity5 = await repo.create_entity(
                external_id=external_id,
                name="Google Gemini",
                types=["LLM", "Service"],
                description="Google's multimodal AI model",
                importance=0.7,
                metadata={"source": "test", "domain": "llm"},
            )

            # Create relationships
            await repo.create_relationship(
                external_id=external_id,
                from_entity_id=entity1.id,
                to_entity_id=entity2.id,
                types=["USES", "IMPLEMENTS"],
                description="Architecture uses vector embeddings for semantic search",
                importance=0.85,
                metadata={"source": "test"},
            )

            await repo.create_relationship(
                external_id=external_id,
                from_entity_id=entity1.id,
                to_entity_id=entity3.id,
                types=["REQUIRES", "APPLIES"],
                description="Architecture requires token optimization",
                importance=0.8,
                metadata={"source": "test"},
            )

            await repo.create_relationship(
                external_id=external_id,
                from_entity_id=entity4.id,
                to_entity_id=entity5.id,
                types=["INTEGRATES_WITH", "SUPPORTS"],
                description="Pydantic AI integrates with Google Gemini",
                importance=0.75,
                metadata={"source": "test"},
            )

            await repo.create_relationship(
                external_id=external_id,
                from_entity_id=entity3.id,
                to_entity_id=entity5.id,
                types=["OPTIMIZES", "REDUCES_COST"],
                description="Token optimization reduces Gemini API costs",
                importance=0.7,
                metadata={"source": "test"},
            )

            logger.info("✅ Test data created successfully")

            # Run test searches
            logger.info("\n" + "=" * 60)
            logger.info("TEST 1: Search for Multi-tier Architecture")
            logger.info("=" * 60)
            result = await repo.search_entities_with_relationships(
                entity_names=["Multi-tier Architecture"], external_id=external_id, limit=10
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
            logger.info("TEST 2: Search with partial name 'optimization'")
            logger.info("=" * 60)
            result = await repo.search_entities_with_relationships(
                entity_names=["optimization"], external_id=external_id, limit=10
            )
            logger.info(f"✅ Found {len(result.matched_entities)} matched entities")
            for entity in result.matched_entities:
                logger.info(f"  Matched: {entity.name}")

            logger.info("\n" + "=" * 60)
            logger.info("TEST 3: Search for multiple entities")
            logger.info("=" * 60)
            result = await repo.search_entities_with_relationships(
                entity_names=["Pydantic", "Gemini"], external_id=external_id, limit=10
            )
            logger.info(f"✅ Found {len(result.matched_entities)} matched entities")
            logger.info(f"✅ Found {len(result.related_entities)} related entities")
            logger.info(f"✅ Found {len(result.relationships)} relationships")

            logger.info("\n" + "=" * 60)
            logger.info("TEST 4: Search Google Gemini (incoming relationships)")
            logger.info("=" * 60)
            result = await repo.search_entities_with_relationships(
                entity_names=["Google Gemini"], external_id=external_id, limit=10
            )
            logger.info(f"✅ Found {len(result.matched_entities)} matched entities")
            logger.info(f"✅ Found {len(result.related_entities)} related entities")
            logger.info(f"✅ Found {len(result.relationships)} relationships")

            incoming = [r for r in result.relationships if r.to_entity_name == "Google Gemini"]
            logger.info(f"  Incoming relationships: {len(incoming)}")
            for rel in incoming:
                logger.info(f"    {rel.from_entity_name} -> {rel.to_entity_name}")

            logger.info("\n🎉 All tests completed successfully!")

        finally:
            await postgres.close()
            await neo4j.close()

    asyncio.run(run_tests())
