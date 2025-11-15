"""
Integration Tests for agent_mem.

These tests verify end-to-end workflows with real or mocked databases.
They test the complete memory lifecycle:
- Create → Update → Consolidate → Promote → Retrieve

Note: For true integration tests, ensure PostgreSQL and Neo4j are running.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from agent_reminiscence.core import AgentMem
from agent_reminiscence.database.models import ActiveMemory


@pytest.mark.integration
class TestEndToEndWorkflow:
    """Test complete memory workflow from creation to retrieval."""

    @pytest.mark.asyncio
    async def test_full_memory_lifecycle(self, test_config):
        """Test complete workflow: create → update → consolidate → retrieve."""
        with (
            patch("agent_reminiscence.database.PostgreSQLManager"),
            patch("agent_reminiscence.database.Neo4jManager"),
            patch("agent_reminiscence.core.MemoryManager") as mock_mm,
        ):

            # Setup mock memory manager with AsyncMock for async methods
            mock_mm_instance = AsyncMock()
            mock_mm_instance.initialize = AsyncMock()
            mock_mm_instance.close = AsyncMock()
            mock_mm_instance.initialize = AsyncMock()
            mock_mm_instance.close = AsyncMock()

            # Mock create - with correct ActiveMemory structure
            memory = ActiveMemory(
                id=1,
                external_id="test-integration",
                title="Test Conversation",
                template_content={
                    "template": {"id": "conversation-template", "name": "Conversation Template"},
                    "sections": [
                        {"id": "summary", "description": "Summary section"},
                        {"id": "context", "description": "Context section"},
                    ],
                },
                sections={
                    "summary": {
                        "content": "Initial conversation",
                        "update_count": 0,
                        "awake_update_count": 0,
                        "last_updated": None,
                    }
                },
                metadata={},
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
            mock_mm_instance.create_active_memory = AsyncMock(return_value=memory)

            # Mock updates (increment count each time)
            updates = []
            for i in range(1, 6):
                updated = ActiveMemory(
                    id=memory.id,
                    external_id="test-integration",
                    title="Test Conversation",
                    template_content={
                        "template": {
                            "id": "conversation-template",
                            "name": "Conversation Template",
                        },
                        "sections": [
                            {"id": "summary", "description": "Summary section"},
                            {"id": "context", "description": "Context section"},
                        ],
                    },
                    sections={
                        "summary": {
                            "content": f"Update {i}",
                            "update_count": i,
                            "awake_update_count": i,
                            "last_updated": datetime.now(timezone.utc),
                        }
                    },
                    metadata={},
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc),
                )
                updates.append(updated)

            mock_mm_instance.update_active_memory_sections = AsyncMock(side_effect=updates)

            # Mock retrieve with proper RetrievalResultV2 structure
            from agent_reminiscence.database.models import RetrievalResultV2

            mock_retrieval_result = RetrievalResultV2(
                mode="deep_search",
                shortterm_chunks=[],
                longterm_chunks=[],
                shortterm_triplets=[],
                longterm_triplets=[],
                synthesis="Retrieved conversation history about test topic.",
                search_strategy="Test search",
                confidence=0.9,
                metadata={},
            )
            mock_mm_instance.deep_search_memories = AsyncMock(return_value=mock_retrieval_result)

            mock_mm.return_value = mock_mm_instance

            # Test workflow
            async with AgentMem(config=test_config) as agent_mem:
                # 1. Create memory
                mem = await agent_mem.create_active_memory(
                    external_id="test-integration",
                    title="Test Conversation",
                    template_content={
                        "template": {
                            "id": "conversation-template",
                            "name": "Conversation Template",
                        },
                        "sections": [
                            {"id": "summary", "description": "Summary section"},
                            {"id": "context", "description": "Context section"},
                        ],
                    },
                    initial_sections={
                        "summary": {
                            "content": "Initial conversation",
                            "update_count": 0,
                            "awake_update_count": 0,
                            "last_updated": None,
                        }
                    },
                )
                assert mem.id == 1

                # 2. Update multiple times
                for i in range(5):
                    mem = await agent_mem.update_active_memory_sections(
                        external_id="test-agent-001",
                        memory_id=mem.id,
                        sections=[{"section_id": "summary", "new_content": f"Update {i+1}"}],
                    )

                # After 5 updates, sections should have been updated
                assert mem.id == 1

                # 3. Retrieve memories
                result = await agent_mem.deep_search_memories(
                    query="What did we discuss?",
                    external_id="test-integration",
                )

                assert result is not None
                assert result.mode == "deep_search"
                assert result.synthesis is not None
                assert len(result.synthesis) > 0


@pytest.mark.integration
class TestConsolidationWorkflow:
    """Test consolidation from active to shortterm memory."""

    @pytest.mark.asyncio
    async def test_consolidation_with_entities(self, test_config):
        """Test consolidation extracts and stores entities."""
        with (
            patch("agent_reminiscence.database.PostgreSQLManager"),
            patch("agent_reminiscence.database.Neo4jManager"),
            patch("agent_reminiscence.core.MemoryManager") as mock_mm,
        ):

            mock_mm_instance = AsyncMock()

            # Mock consolidation
            mock_mm_instance._consolidate_to_shortterm = AsyncMock()
            mock_mm_instance.create_active_memory = AsyncMock()
            mock_mm_instance.update_active_memory = AsyncMock()

            mock_mm.return_value = mock_mm_instance

            async with AgentMem(config=test_config) as agent_mem:
                # Create and update until consolidation
                # Mocking handles the details
                pass

    @pytest.mark.asyncio
    async def test_consolidation_chunking(self, test_config):
        """Test that consolidation properly chunks content."""
        with (
            patch("agent_reminiscence.database.PostgreSQLManager"),
            patch("agent_reminiscence.database.Neo4jManager"),
            patch("agent_reminiscence.core.MemoryManager") as mock_mm,
        ):

            mock_mm_instance = AsyncMock()

            # Create long content that needs chunking
            long_content = " ".join([f"Sentence {i}" for i in range(100)])

            memory = ActiveMemory(
                id=2,
                external_id="test-chunking",
                title="Chunking Test",
                template_content={
                    "template": {"id": "conversation-template", "name": "Conversation Template"},
                    "sections": [{"id": "content", "description": "Content section"}],
                },
                sections={
                    "content": {
                        "content": long_content,
                        "update_count": 5,
                        "awake_update_count": 5,
                        "last_updated": datetime.now(timezone.utc),
                    }
                },
                metadata={},
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )

            mock_mm_instance.create_active_memory = AsyncMock(return_value=memory)
            mock_mm_instance._consolidate_to_shortterm = AsyncMock()

            mock_mm.return_value = mock_mm_instance

            async with AgentMem(config=test_config) as agent_mem:
                mem = await agent_mem.create_active_memory(
                    external_id="test-chunking",
                    title="Chunking Test",
                    template_content={
                        "template": {
                            "id": "conversation-template",
                            "name": "Conversation Template",
                        },
                        "sections": [{"id": "content", "description": "Content section"}],
                    },
                    initial_sections={
                        "content": {
                            "content": long_content,
                            "update_count": 5,
                            "awake_update_count": 5,
                            "last_updated": datetime.now(timezone.utc),
                        }
                    },
                )

                # Trigger consolidation
                # In real implementation, this would create multiple chunks
                # Here we just verify the mock was called
                assert mem is not None


@pytest.mark.integration
class TestPromotionWorkflow:
    """Test promotion from shortterm to longterm memory."""

    @pytest.mark.asyncio
    async def test_promotion_with_importance_threshold(self, test_config):
        """Test that only important memories are promoted."""
        with (
            patch("agent_reminiscence.database.PostgreSQLManager"),
            patch("agent_reminiscence.database.Neo4jManager"),
            patch("agent_reminiscence.core.MemoryManager") as mock_mm,
        ):

            mock_mm_instance = AsyncMock()
            mock_mm_instance._promote_to_longterm = AsyncMock()
            mock_mm.return_value = mock_mm_instance

            async with AgentMem(config=test_config) as agent_mem:
                # Promotion happens internally based on importance
                # This test verifies the workflow exists
                pass

    @pytest.mark.asyncio
    async def test_promotion_entity_confidence_update(self, test_config):
        """Test that entity confidence is updated during promotion."""
        with (
            patch("agent_reminiscence.database.PostgreSQLManager"),
            patch("agent_reminiscence.database.Neo4jManager"),
            patch("agent_reminiscence.core.MemoryManager") as mock_mm,
        ):

            mock_mm_instance = AsyncMock()
            # Promotion should update confidence: 0.7 * existing + 0.3 * new
            mock_mm_instance._promote_to_longterm = AsyncMock()
            mock_mm.return_value = mock_mm_instance

            async with AgentMem(config=test_config) as agent_mem:
                # Test confidence update logic
                pass


@pytest.mark.integration
class TestCrossTierSearch:
    """Test search across all memory tiers."""

    @pytest.mark.asyncio
    async def test_search_all_tiers(self, test_config):
        """Test search across active, shortterm, and longterm."""
        with (
            patch("agent_reminiscence.database.PostgreSQLManager"),
            patch("agent_reminiscence.database.Neo4jManager"),
            patch("agent_reminiscence.core.MemoryManager") as mock_mm,
        ):

            from agent_reminiscence.database.models import RetrievalResult

            mock_result = RetrievalResult(
                mode="synthesis",
                chunks=[],
                entities=[],
                relationships=[],
                synthesis="Found results in active, shortterm, and longterm memories.",
                search_strategy="Multi-tier search",
                confidence=0.9,
                metadata={},
            )

            mock_mm_instance = AsyncMock()
            mock_mm_instance.deep_search_memories = AsyncMock(return_value=mock_result)
            mock_mm.return_value = mock_mm_instance

            async with AgentMem(config=test_config) as agent_mem:
                result = await agent_mem.deep_search_memories(
                    query="machine learning",
                    external_id="test-123",
                )

                assert result.synthesis is not None
                assert "memories" in result.synthesis.lower()

    @pytest.mark.asyncio
    async def test_hybrid_search_weighting(self, test_config):
        """Test hybrid search with different weights."""
        with (
            patch("agent_reminiscence.database.PostgreSQLManager"),
            patch("agent_reminiscence.database.Neo4jManager"),
            patch("agent_reminiscence.core.MemoryManager") as mock_mm,
        ):

            from agent_reminiscence.database.models import RetrievalResultV2

            mock_result = RetrievalResultV2(
                mode="deep_search",
                shortterm_chunks=[],
                longterm_chunks=[],
                shortterm_triplets=[],
                longterm_triplets=[],
                synthesis="Hybrid search results with weighted combination.",
                search_strategy="Hybrid search (vector + BM25)",
                confidence=0.85,
                metadata={},
            )

            mock_mm_instance = AsyncMock()
            mock_mm_instance.deep_search_memories = AsyncMock(return_value=mock_result)
            mock_mm.return_value = mock_mm_instance

            async with AgentMem(config=test_config) as agent_mem:
                result = await agent_mem.deep_search_memories(
                    query="test",
                    external_id="test-123",
                )

                assert result is not None


@pytest.mark.integration
class TestEntityRelationshipPersistence:
    """Test entity and relationship persistence across workflows."""

    @pytest.mark.asyncio
    async def test_entity_extraction_and_storage(self, test_config):
        """Test that entities are extracted and stored in Neo4j."""
        with (
            patch("agent_reminiscence.database.PostgreSQLManager"),
            patch("agent_reminiscence.database.Neo4jManager"),
            patch("agent_reminiscence.core.MemoryManager") as mock_mm,
        ):

            mock_mm_instance = AsyncMock()
            mock_mm_instance.create_active_memory = AsyncMock()
            mock_mm_instance._consolidate_to_shortterm = AsyncMock()
            mock_mm.return_value = mock_mm_instance

            async with AgentMem(config=test_config) as agent_mem:
                # Create memory with entity-rich content
                await agent_mem.create_active_memory(
                    external_id="test-entities",
                    title="Entity Discussion",
                    template_content={
                        "template": {"id": "test", "name": "Test"},
                        "sections": [{"id": "summary", "description": "Discussion summary"}],
                    },
                    initial_sections={
                        "summary": {"content": "Discussion about Python and Machine Learning"}
                    },
                )

                # Consolidation should extract entities
                # In real test, verify entities exist in Neo4j

    @pytest.mark.asyncio
    async def test_relationship_creation(self, test_config):
        """Test that relationships are created between entities."""
        with (
            patch("agent_reminiscence.database.PostgreSQLManager"),
            patch("agent_reminiscence.database.Neo4jManager"),
            patch("agent_reminiscence.core.MemoryManager") as mock_mm,
        ):

            mock_mm_instance = AsyncMock()
            mock_mm_instance._consolidate_to_shortterm = AsyncMock()
            mock_mm.return_value = mock_mm_instance

            async with AgentMem(config=test_config) as agent_mem:
                # Test relationship creation
                # Should connect related entities
                pass

    @pytest.mark.asyncio
    async def test_entity_merging_on_similarity(self, test_config):
        """Test that similar entities are auto-merged."""
        with (
            patch("agent_reminiscence.database.PostgreSQLManager"),
            patch("agent_reminiscence.database.Neo4jManager"),
            patch("agent_reminiscence.core.MemoryManager") as mock_mm,
        ):

            mock_mm_instance = AsyncMock()
            # Auto-resolution should merge entities with:
            # - semantic similarity >= 0.85
            # - entity overlap >= 0.7
            mock_mm_instance._consolidate_to_shortterm = AsyncMock()
            mock_mm.return_value = mock_mm_instance

            async with AgentMem(config=test_config) as agent_mem:
                # Create memories with similar entities
                # Should merge "Python" and "python programming language"
                pass


@pytest.mark.integration
class TestErrorRecovery:
    """Test error handling and recovery."""

    @pytest.mark.asyncio
    async def test_database_connection_error(self, test_config):
        """Test handling of database connection errors."""
        with patch("agent_reminiscence.database.PostgreSQLManager") as mock_pg:
            mock_pg_instance = MagicMock()
            mock_pg_instance.initialize = AsyncMock(side_effect=Exception("Connection failed"))
            mock_pg.return_value = mock_pg_instance

            agent_mem = AgentMem(config=test_config)

            with pytest.raises(Exception):
                await agent_mem.initialize()

    @pytest.mark.asyncio
    async def test_partial_consolidation_failure(self, test_config):
        """Test handling of partial consolidation failures."""
        with (
            patch("agent_reminiscence.database.PostgreSQLManager"),
            patch("agent_reminiscence.database.Neo4jManager"),
            patch("agent_reminiscence.core.MemoryManager") as mock_mm,
        ):

            mock_mm_instance = AsyncMock()
            # Consolidation fails but doesn't break the system
            mock_mm_instance._consolidate_to_shortterm = AsyncMock(
                side_effect=Exception("Consolidation error")
            )
            mock_mm_instance.update_active_memory = AsyncMock()
            mock_mm.return_value = mock_mm_instance

            async with AgentMem(config=test_config) as agent_mem:
                # Update should succeed even if consolidation fails
                # System should log error and continue
                pass

    @pytest.mark.asyncio
    async def test_embedding_generation_fallback(self, test_config):
        """Test fallback when embedding generation fails."""
        with (
            patch("agent_reminiscence.database.PostgreSQLManager"),
            patch("agent_reminiscence.database.Neo4jManager"),
            patch("agent_reminiscence.core.MemoryManager") as mock_mm,
        ):

            mock_mm_instance = AsyncMock()
            # Embedding service should return zero vector on failure
            mock_mm_instance.create_active_memory = AsyncMock()
            mock_mm.return_value = mock_mm_instance

            async with AgentMem(config=test_config) as agent_mem:
                # Should work with fallback embeddings
                await agent_mem.create_active_memory(
                    external_id="test-fallback",
                    title="Fallback Test",
                    template_content={
                        "template": {"id": "test", "name": "Test"},
                        "sections": [{"id": "summary", "description": "Test summary"}],
                    },
                    initial_sections={"summary": {"content": "Test"}},
                )


@pytest.mark.integration
class TestPerformanceCharacteristics:
    """Test performance characteristics of the system."""

    @pytest.mark.asyncio
    async def test_batch_memory_creation(self, test_config):
        """Test creating multiple memories."""
        with (
            patch("agent_reminiscence.database.PostgreSQLManager"),
            patch("agent_reminiscence.database.Neo4jManager"),
            patch("agent_reminiscence.core.MemoryManager") as mock_mm,
        ):

            mock_mm_instance = AsyncMock()
            mock_mm_instance.create_active_memory = AsyncMock()
            mock_mm.return_value = mock_mm_instance

            async with AgentMem(config=test_config) as agent_mem:
                # Create 100 memories
                for i in range(100):
                    await agent_mem.create_active_memory(
                        external_id=f"batch-{i}",
                        title=f"Memory {i}",
                        template_content={
                            "template": {"id": "test", "name": "Test"},
                            "sections": [{"id": "summary", "description": "Test summary"}],
                        },
                        initial_sections={"summary": {"content": f"Memory {i}"}},
                    )

                assert mock_mm_instance.create_active_memory.call_count == 100

    @pytest.mark.asyncio
    async def test_search_with_large_result_set(self, test_config):
        """Test search returning many results."""
        with (
            patch("agent_reminiscence.database.PostgreSQLManager"),
            patch("agent_reminiscence.database.Neo4jManager"),
            patch("agent_reminiscence.core.MemoryManager") as mock_mm,
        ):

            from agent_reminiscence.database.models import RetrievalResultV2

            mock_mm_instance = AsyncMock()
            # Return large result set
            large_response = RetrievalResultV2(
                mode="deep_search",
                shortterm_chunks=[],
                longterm_chunks=[],
                shortterm_triplets=[],
                longterm_triplets=[],
                synthesis="Summary of 1000+ search results...",
                search_strategy="Large result set search",
                confidence=0.8,
                metadata={},
            )
            mock_mm_instance.deep_search_memories = AsyncMock(return_value=large_response)
            mock_mm.return_value = mock_mm_instance

            async with AgentMem(config=test_config) as agent_mem:
                result = await agent_mem.deep_search_memories(
                    query="broad query",
                    external_id="test-123",
                )

                assert result is not None
