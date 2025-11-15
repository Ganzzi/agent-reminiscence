"""
Phase 5 Tests: Core API Endpoints

Tests for the two new retrieval methods added to AgentMem:
- search_memories(): Fast pointer-based retrieval (< 200ms)
- deep_search_memories(): Full synthesis with AI (500ms-2s)

Verifies:
1. Both methods exist and have correct signatures
2. Methods delegate to MemoryManager correctly
3. External ID conversion works (UUID, string, int)
4. Logging includes relevant context
5. Return types are correct (RetrievalResult)
6. Backward compatibility maintained (retrieve_memories still works)
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID
from datetime import datetime, timezone

from agent_reminiscence import AgentMem
from agent_reminiscence.database.models import RetrievalResult, RetrievedChunk
from agent_reminiscence.services.memory_manager import MemoryManager


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def config_mock():
    """Mock configuration for AgentMem."""
    return MagicMock()


@pytest.fixture
def memory_manager_mock():
    """Mock MemoryManager for testing."""
    manager = AsyncMock(spec=MemoryManager)

    # Mock the retrieval methods
    manager.search_memories = AsyncMock(
        return_value=RetrievalResult(
            mode="pointer",
            chunks=[],
            entities=[],
            relationships=[],
            synthesis=None,
            search_strategy="Mock fast search",
            confidence=0.75,
            metadata={"test": True},
        )
    )

    manager.deep_search_memories = AsyncMock(
        return_value=RetrievalResult(
            mode="synthesis",
            chunks=[],
            entities=[],
            relationships=[],
            synthesis="Mock synthesis result",
            search_strategy="Mock deep search",
            confidence=0.85,
            metadata={"test": True},
        )
    )

    manager.retrieve_memories = AsyncMock(
        return_value=RetrievalResult(
            mode="pointer",
            chunks=[],
            entities=[],
            relationships=[],
            synthesis=None,
            search_strategy="Mock retrieval",
            confidence=0.75,
            metadata={},
        )
    )

    return manager


@pytest.fixture
async def agent_mem_initialized(memory_manager_mock, config_mock):
    """Create initialized AgentMem with mocked dependencies."""
    with patch("agent_reminiscence.core.get_config", return_value=config_mock):
        agent_mem = AgentMem(config=config_mock)
        agent_mem._memory_manager = memory_manager_mock
        agent_mem._initialized = True
        yield agent_mem
        agent_mem._initialized = False


# ============================================================================
# TESTS: Method Existence
# ============================================================================


@pytest.mark.asyncio
async def test_search_memories_exists(agent_mem_initialized):
    """Test that search_memories method exists on AgentMem."""
    assert hasattr(agent_mem_initialized, "search_memories")
    assert callable(agent_mem_initialized.search_memories)


@pytest.mark.asyncio
async def test_deep_search_memories_exists(agent_mem_initialized):
    """Test that deep_search_memories method exists on AgentMem."""
    assert hasattr(agent_mem_initialized, "deep_search_memories")
    assert callable(agent_mem_initialized.deep_search_memories)


# ============================================================================
# TESTS: Method Signatures
# ============================================================================


@pytest.mark.asyncio
async def test_search_memories_signature(agent_mem_initialized):
    """Test that search_memories has correct signature."""
    import inspect

    sig = inspect.signature(agent_mem_initialized.search_memories)
    params = list(sig.parameters.keys())

    # Should have: self, external_id, query, limit
    assert "external_id" in params
    assert "query" in params
    assert "limit" in params

    # Check defaults
    assert sig.parameters["limit"].default == 10


@pytest.mark.asyncio
async def test_deep_search_memories_signature(agent_mem_initialized):
    """Test that deep_search_memories has correct signature."""
    import inspect

    sig = inspect.signature(agent_mem_initialized.deep_search_memories)
    params = list(sig.parameters.keys())

    # Should have: self, external_id, query, limit
    assert "external_id" in params
    assert "query" in params
    assert "limit" in params

    # Check defaults
    assert sig.parameters["limit"].default == 10


@pytest.mark.asyncio
async def test_both_methods_have_identical_signatures(agent_mem_initialized):
    """Test that search_memories and deep_search_memories have identical signatures."""
    import inspect

    search_sig = inspect.signature(agent_mem_initialized.search_memories)
    deep_sig = inspect.signature(agent_mem_initialized.deep_search_memories)

    search_params = set(search_sig.parameters.keys())
    deep_params = set(deep_sig.parameters.keys())

    # Both should have same parameters (except self, implicit)
    assert search_params == deep_params


# ============================================================================
# TESTS: Delegation to MemoryManager
# ============================================================================


@pytest.mark.asyncio
async def test_search_memories_calls_manager(agent_mem_initialized, memory_manager_mock):
    """Test that search_memories delegates to MemoryManager."""
    result = await agent_mem_initialized.search_memories(
        external_id="agent-123",
        query="test query",
        limit=5,
    )

    # Verify MemoryManager method was called
    memory_manager_mock.search_memories.assert_called_once_with(
        external_id="agent-123",
        query="test query",
        limit=5,
    )

    # Verify result is RetrievalResult
    assert isinstance(result, RetrievalResult)
    assert result.mode == "pointer"


@pytest.mark.asyncio
async def test_deep_search_memories_calls_manager(agent_mem_initialized, memory_manager_mock):
    """Test that deep_search_memories delegates to MemoryManager."""
    result = await agent_mem_initialized.deep_search_memories(
        external_id="agent-456",
        query="complex query",
        limit=7,
    )

    # Verify MemoryManager method was called
    memory_manager_mock.deep_search_memories.assert_called_once_with(
        external_id="agent-456",
        query="complex query",
        limit=7,
    )

    # Verify result is RetrievalResult
    assert isinstance(result, RetrievalResult)
    assert result.mode == "synthesis"


@pytest.mark.asyncio
async def test_search_memories_returns_retrieval_result(agent_mem_initialized):
    """Test that search_memories returns RetrievalResult."""
    result = await agent_mem_initialized.search_memories(
        external_id="agent-789",
        query="query",
    )

    assert isinstance(result, RetrievalResult)


@pytest.mark.asyncio
async def test_deep_search_memories_returns_retrieval_result(agent_mem_initialized):
    """Test that deep_search_memories returns RetrievalResult."""
    result = await agent_mem_initialized.deep_search_memories(
        external_id="agent-789",
        query="query",
    )

    assert isinstance(result, RetrievalResult)


# ============================================================================
# TESTS: External ID Conversion
# ============================================================================


@pytest.mark.asyncio
async def test_search_memories_with_string_external_id(agent_mem_initialized, memory_manager_mock):
    """Test search_memories accepts string external_id."""
    await agent_mem_initialized.search_memories(
        external_id="agent-string",
        query="test",
    )

    memory_manager_mock.search_memories.assert_called_with(
        external_id="agent-string",
        query="test",
        limit=10,
    )


@pytest.mark.asyncio
async def test_search_memories_with_uuid_external_id(agent_mem_initialized, memory_manager_mock):
    """Test search_memories accepts UUID external_id."""
    test_uuid = UUID("12345678-1234-5678-1234-567812345678")

    await agent_mem_initialized.search_memories(
        external_id=test_uuid,
        query="test",
    )

    # Should be converted to string
    memory_manager_mock.search_memories.assert_called_with(
        external_id=str(test_uuid),
        query="test",
        limit=10,
    )


@pytest.mark.asyncio
async def test_search_memories_with_int_external_id(agent_mem_initialized, memory_manager_mock):
    """Test search_memories accepts int external_id."""
    await agent_mem_initialized.search_memories(
        external_id=12345,
        query="test",
    )

    # Should be converted to string
    memory_manager_mock.search_memories.assert_called_with(
        external_id="12345",
        query="test",
        limit=10,
    )


@pytest.mark.asyncio
async def test_deep_search_memories_with_uuid_external_id(
    agent_mem_initialized, memory_manager_mock
):
    """Test deep_search_memories accepts UUID external_id."""
    test_uuid = UUID("87654321-4321-8765-4321-876543218765")

    await agent_mem_initialized.deep_search_memories(
        external_id=test_uuid,
        query="test",
    )

    # Should be converted to string
    memory_manager_mock.deep_search_memories.assert_called_with(
        external_id=str(test_uuid),
        query="test",
        limit=10,
    )


# ============================================================================
# TESTS: Documentation
# ============================================================================


@pytest.mark.asyncio
async def test_search_memories_has_docstring(agent_mem_initialized):
    """Test that search_memories has documentation."""
    assert agent_mem_initialized.search_memories.__doc__ is not None
    doc = agent_mem_initialized.search_memories.__doc__.lower()
    assert "fast" in doc
    assert "pointer" in doc
    assert "200" in doc or "latency" in doc


@pytest.mark.asyncio
async def test_deep_search_memories_has_docstring(agent_mem_initialized):
    """Test that deep_search_memories has documentation."""
    assert agent_mem_initialized.deep_search_memories.__doc__ is not None
    doc = agent_mem_initialized.deep_search_memories.__doc__.lower()
    assert "deep" in doc or "synthesis" in doc
    assert "agent" in doc or "ai" in doc
    assert "500" in doc or "2" in doc or "token" in doc.lower()


@pytest.mark.asyncio
async def test_search_memories_docstring_includes_examples(agent_mem_initialized):
    """Test that search_memories docstring includes usage examples."""
    doc = agent_mem_initialized.search_memories.__doc__
    assert "example" in doc.lower()
    assert "await" in doc


@pytest.mark.asyncio
async def test_deep_search_memories_docstring_includes_examples(agent_mem_initialized):
    """Test that deep_search_memories docstring includes usage examples."""
    doc = agent_mem_initialized.deep_search_memories.__doc__
    assert "example" in doc.lower()
    assert "await" in doc
    assert "synthesis" in doc.lower()


# ============================================================================
# TESTS: Logging
# ============================================================================


@pytest.mark.asyncio
async def test_search_memories_logs(agent_mem_initialized, caplog):
    """Test that search_memories logs its execution."""
    import logging

    caplog.set_level(logging.INFO)
    await agent_mem_initialized.search_memories(
        external_id="agent-log",
        query="logging test query",
    )

    # Should log the fast search
    assert any(
        "fast search" in record.message.lower() or "search" in record.message.lower()
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_deep_search_memories_logs_warning_about_tokens(agent_mem_initialized, caplog):
    """Test that deep_search_memories logs warning about token usage."""
    import logging

    caplog.set_level(logging.INFO)
    await agent_mem_initialized.deep_search_memories(
        external_id="agent-log",
        query="token warning test",
    )

    # Should log token warning
    assert any(
        "token" in record.message.lower()
        for record in caplog.records
        if record.levelno >= logging.WARNING
    )


# ============================================================================
# TESTS: Breaking Changes in v0.2.0
# ============================================================================


@pytest.mark.asyncio
async def test_retrieve_memories_removed(agent_mem_initialized):
    """Test that retrieve_memories method was removed (breaking change in v0.2.0)."""
    assert not hasattr(agent_mem_initialized, "retrieve_memories")


@pytest.mark.asyncio
async def test_search_and_deep_search_are_replacements(agent_mem_initialized):
    """Test that search_memories and deep_search_memories replace retrieve_memories."""
    assert hasattr(agent_mem_initialized, "search_memories")
    assert hasattr(agent_mem_initialized, "deep_search_memories")
    assert callable(agent_mem_initialized.search_memories)
    assert callable(agent_mem_initialized.deep_search_memories)


# ============================================================================
# TESTS: Error Handling
# ============================================================================


@pytest.mark.asyncio
async def test_search_memories_not_initialized_raises(memory_manager_mock):
    """Test that search_memories raises RuntimeError if not initialized."""
    with patch("agent_reminiscence.core.get_config", return_value=MagicMock()):
        agent_mem = AgentMem()

        with pytest.raises(RuntimeError):
            await agent_mem.search_memories(
                external_id="agent-err",
                query="test",
            )


@pytest.mark.asyncio
async def test_deep_search_memories_not_initialized_raises(memory_manager_mock):
    """Test that deep_search_memories raises RuntimeError if not initialized."""
    with patch("agent_reminiscence.core.get_config", return_value=MagicMock()):
        agent_mem = AgentMem()

        with pytest.raises(RuntimeError):
            await agent_mem.deep_search_memories(
                external_id="agent-err",
                query="test",
            )


# ============================================================================
# TESTS: API Consistency
# ============================================================================


@pytest.mark.asyncio
async def test_search_and_deep_search_parameter_defaults_match(agent_mem_initialized):
    """Test that search_memories and deep_search_memories have matching defaults."""
    import inspect

    search_sig = inspect.signature(agent_mem_initialized.search_memories)
    deep_sig = inspect.signature(agent_mem_initialized.deep_search_memories)

    # Both should have limit default of 10
    assert search_sig.parameters["limit"].default == 10
    assert deep_sig.parameters["limit"].default == 10


@pytest.mark.asyncio
async def test_both_methods_require_external_id(agent_mem_initialized):
    """Test that both methods require external_id parameter."""
    import inspect

    search_sig = inspect.signature(agent_mem_initialized.search_memories)
    deep_sig = inspect.signature(agent_mem_initialized.deep_search_memories)

    # Both should have external_id with no default
    assert "external_id" in search_sig.parameters
    assert "external_id" in deep_sig.parameters
    assert search_sig.parameters["external_id"].default == inspect.Parameter.empty
    assert deep_sig.parameters["external_id"].default == inspect.Parameter.empty


@pytest.mark.asyncio
async def test_both_methods_require_query(agent_mem_initialized):
    """Test that both methods require query parameter."""
    import inspect

    search_sig = inspect.signature(agent_mem_initialized.search_memories)
    deep_sig = inspect.signature(agent_mem_initialized.deep_search_memories)

    # Both should have query with no default
    assert "query" in search_sig.parameters
    assert "query" in deep_sig.parameters
    assert search_sig.parameters["query"].default == inspect.Parameter.empty
    assert deep_sig.parameters["query"].default == inspect.Parameter.empty


# ============================================================================
# TESTS: Integration
# ============================================================================


@pytest.mark.asyncio
async def test_search_memories_with_realistic_result(agent_mem_initialized, memory_manager_mock):
    """Test search_memories with realistic RetrievalResult."""
    chunk = RetrievedChunk(
        id=1,
        content="Important information",
        tier="shortterm",
        score=0.95,
    )

    memory_manager_mock.search_memories.return_value = RetrievalResult(
        mode="pointer",
        chunks=[chunk],
        entities=[],
        relationships=[],
        synthesis=None,
        search_strategy="Direct search",
        confidence=0.85,
        metadata={"chunks_found": 1},
    )

    result = await agent_mem_initialized.search_memories(
        external_id="agent-realistic",
        query="test",
    )

    assert result.mode == "pointer"
    assert len(result.chunks) == 1
    assert result.chunks[0].content == "Important information"
    assert result.chunks[0].score == 0.95
    assert result.synthesis is None  # Pointer mode has no synthesis


@pytest.mark.asyncio
async def test_deep_search_memories_with_realistic_result(
    agent_mem_initialized, memory_manager_mock
):
    """Test deep_search_memories with realistic RetrievalResult."""
    chunk = RetrievedChunk(
        id=2,
        content="Detailed information",
        tier="longterm",
        score=0.88,
        importance=0.9,
    )

    memory_manager_mock.deep_search_memories.return_value = RetrievalResult(
        mode="synthesis",
        chunks=[chunk],
        entities=[],
        relationships=[],
        synthesis="This is a summary of the findings.",
        search_strategy="Deep synthesis search",
        confidence=0.92,
        metadata={"chunks_found": 1, "tokens_used": 150},
    )

    result = await agent_mem_initialized.deep_search_memories(
        external_id="agent-realistic",
        query="test",
    )

    assert result.mode == "synthesis"
    assert len(result.chunks) == 1
    assert result.synthesis is not None
    assert "summary" in result.synthesis.lower()
    assert result.confidence == 0.92


# ============================================================================
# TESTS: Return Type Validation
# ============================================================================


@pytest.mark.asyncio
async def test_search_memories_always_returns_retrieval_result(
    agent_mem_initialized, memory_manager_mock
):
    """Test that search_memories always returns RetrievalResult."""
    result = await agent_mem_initialized.search_memories(
        external_id="test",
        query="test",
    )

    assert isinstance(result, RetrievalResult)
    assert hasattr(result, "mode")
    assert hasattr(result, "chunks")
    assert hasattr(result, "entities")
    assert hasattr(result, "relationships")
    assert hasattr(result, "synthesis")
    assert hasattr(result, "search_strategy")
    assert hasattr(result, "confidence")
    assert hasattr(result, "metadata")


@pytest.mark.asyncio
async def test_deep_search_memories_always_returns_retrieval_result(
    agent_mem_initialized, memory_manager_mock
):
    """Test that deep_search_memories always returns RetrievalResult."""
    result = await agent_mem_initialized.deep_search_memories(
        external_id="test",
        query="test",
    )

    assert isinstance(result, RetrievalResult)
    assert hasattr(result, "mode")
    assert hasattr(result, "chunks")
    assert hasattr(result, "entities")
    assert hasattr(result, "relationships")
    assert hasattr(result, "synthesis")
    assert hasattr(result, "search_strategy")
    assert hasattr(result, "confidence")
    assert hasattr(result, "metadata")
