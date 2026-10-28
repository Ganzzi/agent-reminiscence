"""
Pytest configuration and fixtures for agent_mem tests.
"""

import asyncio
import os
from pathlib import Path
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from agent_reminiscence.config.settings import Config
from agent_reminiscence.database.postgres_manager import PostgreSQLManager
from agent_reminiscence.database.neo4j_manager import Neo4jManager
from agent_reminiscence.database.repositories.active_memory import ActiveMemoryRepository
from agent_reminiscence.database.repositories.shortterm_memory import ShorttermMemoryRepository
from agent_reminiscence.database.repositories.longterm_memory import LongtermMemoryRepository
from agent_reminiscence.services.embedding import EmbeddingService
from agent_reminiscence.services.memory_manager import MemoryManager


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Apply default test markers based on the file path."""
    for item in items:
        path = Path(str(item.fspath))

        if "e2e" in path.parts:
            item.add_marker(pytest.mark.e2e)
        elif "integration" in path.parts:
            item.add_marker(pytest.mark.integration)
        elif "unit" in path.parts:
            item.add_marker(pytest.mark.unit)


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
    """Create a test configuration.

    Reads from environment variables for local QA service addresses.
    Falls back to localhost defaults consistent with docker-compose.qa.yml port mappings.
    """
    return Config(
        postgres_host=os.getenv("POSTGRES_HOST", "localhost"),
        postgres_port=int(os.getenv("POSTGRES_PORT", "5433")),  # Local QA PostgreSQL uses port 5433
        postgres_db=os.getenv("POSTGRES_DB", "agent_mem_test"),
        postgres_user=os.getenv("POSTGRES_USER", "backend"),
        postgres_password=os.getenv("POSTGRES_PASSWORD", "agent-mem-test-password"),
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", "neo4jpassword"),
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
        embedding_provider="ollama",
        openai_embedding_model="text-embedding-3-small",
        google_embedding_model="text-embedding-004",
        vector_dimension=768,
        consolidation_threshold=3,
        promotion_importance_threshold=0.7,
        entity_similarity_threshold=0.85,
        entity_overlap_threshold=0.7,
        # Agent model configurations (required for tests using agents)
        er_extractor_agent_model="google:gemini-2.5-flash",
        memory_update_agent_model="google:gemini-2.5-flash",
        memorizer_agent_model="google:gemini-2.5-flash",
        memory_retrieve_agent_model="google:gemini-2.5-flash",
    )


# ============================================================================
# Database Manager Fixtures
# ============================================================================


@pytest.fixture
def postgres_manager() -> None:
    raise RuntimeError(
        "postgres_manager requires explicit run-owned integration targeting "
        "(NGU-100): request 'integration_config' under tests/integration/ with "
        "AGENT_MEM_INTEGRATION_TARGETING set. Ambient-default real resources "
        "are refused."
    )


@pytest.fixture
def neo4j_manager() -> None:
    raise RuntimeError(
        "neo4j_manager requires explicit run-owned integration targeting "
        "(NGU-100): request 'integration_config' under tests/integration/ with "
        "AGENT_MEM_INTEGRATION_TARGETING set. Ambient-default real resources "
        "are refused."
    )


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
def active_memory_repository() -> None:
    raise RuntimeError(
        "active_memory_repository requires explicit run-owned integration targeting "
        "(NGU-100): request 'integration_config' under tests/integration/ with "
        "AGENT_MEM_INTEGRATION_TARGETING set. Ambient-default real resources "
        "are refused."
    )


@pytest.fixture
def shortterm_memory_repository() -> None:
    raise RuntimeError(
        "shortterm_memory_repository requires explicit run-owned integration targeting "
        "(NGU-100): request 'integration_config' under tests/integration/ with "
        "AGENT_MEM_INTEGRATION_TARGETING set. Ambient-default real resources "
        "are refused."
    )


@pytest.fixture
def longterm_memory_repository() -> None:
    raise RuntimeError(
        "longterm_memory_repository requires explicit run-owned integration targeting "
        "(NGU-100): request 'integration_config' under tests/integration/ with "
        "AGENT_MEM_INTEGRATION_TARGETING set. Ambient-default real resources "
        "are refused."
    )


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
def embedding_service() -> None:
    raise RuntimeError(
        "embedding_service requires explicit run-owned integration targeting "
        "(NGU-100): request 'integration_config' under tests/integration/ with "
        "AGENT_MEM_INTEGRATION_TARGETING set. Ambient-default real resources "
        "are refused."
    )


@pytest.fixture
def mock_embedding_service() -> MagicMock:
    """Create a mock EmbeddingService."""
    mock = MagicMock(spec=EmbeddingService)
    # Return a 768-dimensional zero vector (typical embedding size)
    mock.get_embedding = AsyncMock(return_value=[0.0] * 768)
    mock.get_embeddings_batch = AsyncMock(return_value=[[0.0] * 768])
    return mock


@pytest.fixture
def memory_manager() -> None:
    raise RuntimeError(
        "memory_manager requires explicit run-owned integration targeting "
        "(NGU-100): request 'integration_config' under tests/integration/ with "
        "AGENT_MEM_INTEGRATION_TARGETING set. Ambient-default real resources "
        "are refused."
    )


@pytest.fixture
def mock_memory_manager() -> MagicMock:
    """Create a mock MemoryManager."""
    from agent_reminiscence.database.models import RetrievalResult

    mock = MagicMock(spec=MemoryManager)
    mock.create_active_memory = AsyncMock()
    mock.get_active_memory = AsyncMock()
    mock.update_active_memory = AsyncMock()

    # Return proper RetrievalResult structure
    mock_result = RetrievalResult(
        mode="pointer",
        chunks=[],
        entities=[],
        relationships=[],
        synthesis=None,
        search_strategy="Mock search",
        confidence=1.0,
        metadata={},
    )
    mock.retrieve_memories = AsyncMock(return_value=mock_result)
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
def cleanup_test_data() -> None:
    raise RuntimeError(
        "cleanup_test_data requires explicit run-owned integration targeting "
        "(NGU-100): request 'integration_config' under tests/integration/ with "
        "AGENT_MEM_INTEGRATION_TARGETING set. Ambient-default real resources "
        "and broad ambient deletes are refused."
    )


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
