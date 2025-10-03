"""
Pytest configuration and fixtures for agent_mem tests.
"""

import asyncio
import os
import pytest
from datetime import datetime, timezone
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock

from agent_mem.config.settings import Config
from agent_mem.database.postgres_manager import PostgreSQLManager
from agent_mem.database.neo4j_manager import Neo4jManager
from agent_mem.database.repositories.active_memory import ActiveMemoryRepository
from agent_mem.database.repositories.shortterm_memory import ShorttermMemoryRepository
from agent_mem.database.repositories.longterm_memory import LongtermMemoryRepository
from agent_mem.services.embedding import EmbeddingService
from agent_mem.services.memory_manager import MemoryManager


# ============================================================================
# Event Loop Fixture
# ============================================================================


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# Configuration Fixtures
# ============================================================================


@pytest.fixture
def test_config() -> Config:
    """Create a test configuration."""
    return Config(
        postgres_host=os.getenv("POSTGRES_HOST", "localhost"),
        postgres_port=int(os.getenv("POSTGRES_PORT", "5432")),
        postgres_db=os.getenv("POSTGRES_DB", "agent_mem_test"),
        postgres_user=os.getenv("POSTGRES_USER", "postgres"),
        postgres_password=os.getenv("POSTGRES_PASSWORD", "postgres"),
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", "password"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
        consolidation_threshold=3,
        promotion_importance_threshold=0.7,
        entity_similarity_threshold=0.85,
        entity_overlap_threshold=0.7,
    )


@pytest.fixture
def mock_config() -> Config:
    """Create a mock configuration for isolated tests."""
    return Config(
        postgres_host="mock_host",
        postgres_port=5432,
        postgres_db="mock_db",
        postgres_user="mock_user",
        postgres_password="mock_password",
        neo4j_uri="bolt://mock:7687",
        neo4j_user="mock_user",
        neo4j_password="mock_password",
        ollama_base_url="http://mock:11434",
        embedding_model="mock-model",
        consolidation_threshold=3,
        promotion_importance_threshold=0.7,
        entity_similarity_threshold=0.85,
        entity_overlap_threshold=0.7,
    )


# ============================================================================
# Database Manager Fixtures
# ============================================================================


@pytest.fixture
async def postgres_manager(test_config: Config) -> AsyncGenerator[PostgreSQLManager, None]:
    """Create a PostgreSQLManager with test configuration."""
    manager = PostgreSQLManager(test_config)
    await manager.initialize()
    yield manager
    await manager.close()


@pytest.fixture
async def neo4j_manager(test_config: Config) -> AsyncGenerator[Neo4jManager, None]:
    """Create a Neo4jManager with test configuration."""
    manager = Neo4jManager(test_config)
    yield manager
    await manager.close()


@pytest.fixture
def mock_postgres_manager() -> MagicMock:
    """Create a mock PostgreSQLManager."""
    mock = MagicMock(spec=PostgreSQLManager)
    mock.execute_query = AsyncMock(return_value=None)
    mock.execute_query_one = AsyncMock(return_value=None)
    mock.execute_query_many = AsyncMock(return_value=[])
    return mock


@pytest.fixture
def mock_neo4j_manager() -> MagicMock:
    """Create a mock Neo4jManager."""
    mock = MagicMock(spec=Neo4jManager)
    mock.execute_read = AsyncMock(return_value=[])
    mock.execute_write = AsyncMock(return_value=None)
    return mock


# ============================================================================
# Repository Fixtures
# ============================================================================


@pytest.fixture
async def active_memory_repository(
    postgres_manager: PostgreSQLManager,
) -> ActiveMemoryRepository:
    """Create an ActiveMemoryRepository with real database."""
    return ActiveMemoryRepository(postgres_manager)


@pytest.fixture
async def shortterm_memory_repository(
    postgres_manager: PostgreSQLManager,
    neo4j_manager: Neo4jManager,
) -> ShorttermMemoryRepository:
    """Create a ShorttermMemoryRepository with real databases."""
    return ShorttermMemoryRepository(postgres_manager, neo4j_manager)


@pytest.fixture
async def longterm_memory_repository(
    postgres_manager: PostgreSQLManager,
    neo4j_manager: Neo4jManager,
) -> LongtermMemoryRepository:
    """Create a LongtermMemoryRepository with real databases."""
    return LongtermMemoryRepository(postgres_manager, neo4j_manager)


@pytest.fixture
def mock_active_memory_repository() -> MagicMock:
    """Create a mock ActiveMemoryRepository."""
    mock = MagicMock(spec=ActiveMemoryRepository)
    mock.create = AsyncMock()
    mock.get_by_id = AsyncMock()
    mock.get_all_by_external_id = AsyncMock(return_value=[])
    mock.update = AsyncMock()
    mock.delete = AsyncMock()
    return mock


@pytest.fixture
def mock_shortterm_memory_repository() -> MagicMock:
    """Create a mock ShorttermMemoryRepository."""
    mock = MagicMock(spec=ShorttermMemoryRepository)
    mock.create = AsyncMock()
    mock.get_by_id = AsyncMock()
    mock.search_hybrid = AsyncMock(return_value=[])
    mock.create_entity = AsyncMock()
    mock.create_relationship = AsyncMock()
    return mock


@pytest.fixture
def mock_longterm_memory_repository() -> MagicMock:
    """Create a mock LongtermMemoryRepository."""
    mock = MagicMock(spec=LongtermMemoryRepository)
    mock.create = AsyncMock()
    mock.get_by_id = AsyncMock()
    mock.search_hybrid = AsyncMock(return_value=[])
    mock.create_entity = AsyncMock()
    mock.create_relationship = AsyncMock()
    return mock


# ============================================================================
# Service Fixtures
# ============================================================================


@pytest.fixture
async def embedding_service(test_config: Config) -> EmbeddingService:
    """Create an EmbeddingService with test configuration."""
    return EmbeddingService(test_config)


@pytest.fixture
def mock_embedding_service() -> MagicMock:
    """Create a mock EmbeddingService."""
    mock = MagicMock(spec=EmbeddingService)
    # Return a 768-dimensional zero vector (typical embedding size)
    mock.generate_embedding = AsyncMock(return_value=[0.0] * 768)
    mock.generate_embeddings = AsyncMock(return_value=[[0.0] * 768])
    return mock


@pytest.fixture
async def memory_manager(
    postgres_manager: PostgreSQLManager,
    neo4j_manager: Neo4jManager,
    test_config: Config,
) -> MemoryManager:
    """Create a MemoryManager with real dependencies."""
    return MemoryManager(postgres_manager, neo4j_manager, test_config)


@pytest.fixture
def mock_memory_manager() -> MagicMock:
    """Create a mock MemoryManager."""
    mock = MagicMock(spec=MemoryManager)
    mock.create_active_memory = AsyncMock()
    mock.get_active_memory = AsyncMock()
    mock.update_active_memory = AsyncMock()
    mock.retrieve_memories = AsyncMock(return_value="Mock retrieved memories")
    return mock


# ============================================================================
# Test Data Fixtures
# ============================================================================


@pytest.fixture
def sample_active_memory_data() -> dict:
    """Sample data for creating active memories."""
    return {
        "external_id": "test-conversation-1",
        "memory_type": "conversation",
        "sections": {
            "summary": "Test conversation about AI",
            "key_points": ["AI is powerful", "Machine learning basics"],
            "context": "Technical discussion",
        },
        "metadata": {
            "user_id": "user-123",
            "session_id": "session-456",
        },
    }


@pytest.fixture
def sample_shortterm_chunk_data() -> dict:
    """Sample data for creating shortterm memory chunks."""
    return {
        "chunk_text": "This is a test chunk about artificial intelligence and machine learning.",
        "chunk_index": 0,
        "embedding": [0.1] * 768,
        "metadata": {"source": "test", "timestamp": datetime.now(timezone.utc).isoformat()},
    }


@pytest.fixture
def sample_longterm_chunk_data() -> dict:
    """Sample data for creating longterm memory chunks."""
    return {
        "chunk_text": "This is a validated knowledge chunk about neural networks.",
        "chunk_index": 0,
        "embedding": [0.2] * 768,
        "confidence": 0.9,
        "importance": 0.85,
        "metadata": {"source": "consolidated", "validated": True},
    }


@pytest.fixture
def sample_entity_data() -> dict:
    """Sample data for creating entities."""
    return {
        "name": "Neural Network",
        "type": "TECHNOLOGY",
        "properties": {
            "description": "A computing system inspired by biological neural networks",
            "category": "Machine Learning",
        },
        "confidence": 0.9,
    }


@pytest.fixture
def sample_relationship_data() -> dict:
    """Sample data for creating relationships."""
    return {
        "type": "USES",
        "properties": {
            "context": "Machine learning applications",
            "frequency": "often",
        },
        "confidence": 0.85,
    }


# ============================================================================
# Cleanup Fixtures
# ============================================================================


@pytest.fixture
async def cleanup_test_data(
    postgres_manager: PostgreSQLManager,
    neo4j_manager: Neo4jManager,
) -> AsyncGenerator[None, None]:
    """Clean up test data before and after tests."""
    # Cleanup before test
    await _cleanup_databases(postgres_manager, neo4j_manager)

    yield

    # Cleanup after test
    await _cleanup_databases(postgres_manager, neo4j_manager)


async def _cleanup_databases(
    postgres_manager: PostgreSQLManager,
    neo4j_manager: Neo4jManager,
) -> None:
    """Helper to clean up all test data."""
    # Clean PostgreSQL
    try:
        await postgres_manager.execute_query(
            "DELETE FROM active_memory WHERE external_id LIKE 'test-%'"
        )
        await postgres_manager.execute_query(
            "DELETE FROM shortterm_memory WHERE external_id LIKE 'test-%'"
        )
        await postgres_manager.execute_query(
            "DELETE FROM longterm_memory_chunks WHERE external_id LIKE 'test-%'"
        )
    except Exception as e:
        print(f"Error cleaning PostgreSQL: {e}")

    # Clean Neo4j
    try:
        await neo4j_manager.execute_write(
            "MATCH (n) WHERE n.memory_id STARTS WITH 'test-' DETACH DELETE n"
        )
    except Exception as e:
        print(f"Error cleaning Neo4j: {e}")


# ============================================================================
# Mock Agent Fixtures
# ============================================================================


@pytest.fixture
def mock_er_extractor_result() -> dict:
    """Mock result from ER Extractor Agent."""
    return {
        "entities": [
            {
                "name": "Python",
                "type": "TECHNOLOGY",
                "properties": {"description": "Programming language"},
                "confidence": 0.9,
            },
            {
                "name": "Machine Learning",
                "type": "TECHNOLOGY",
                "properties": {"description": "AI technique"},
                "confidence": 0.85,
            },
        ],
        "relationships": [
            {
                "source": "Python",
                "target": "Machine Learning",
                "type": "USED_FOR",
                "properties": {"context": "data science"},
                "confidence": 0.8,
            }
        ],
    }


@pytest.fixture
def mock_memory_retrieve_strategy() -> dict:
    """Mock strategy from Memory Retrieve Agent."""
    return {
        "search_tiers": ["active", "shortterm", "longterm"],
        "search_type": "hybrid",
        "vector_weight": 0.7,
        "bm25_weight": 0.3,
        "filters": {},
        "confidence": 0.9,
    }
