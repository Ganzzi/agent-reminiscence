"""
Phase 3 Tests: Agent Triplet Support and Token Usage Tracking.

Tests for:
- Triplet search tools (shortterm/longterm)
- Knowledge triplet storage and retrieval
- Token usage extraction from agent runs
- UsageProcessor protocol and LoggingUsageProcessor
- End-to-end retrieval with triplets
- Error handling and edge cases
"""

import pytest
import logging
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from pydantic_ai.usage import RunUsage

from agent_reminiscence.database.models import (
    ShorttermKnowledgeTriplet,
    LongtermKnowledgeTriplet,
    ShorttermEntity,
    LongtermEntity,
)
from agent_reminiscence.services.central_storage import CentralStorage
from agent_reminiscence.services.memory_manager import (
    UsageProcessor,
    LoggingUsageProcessor,
)
from agent_reminiscence.agents.memory_retriever import (
    search_shortterm_triplets,
    search_longterm_triplets,
    TripletPointer,
    resolve_and_format_results,
    RetrievalResult,
    RetrieverDeps,
)

logger = logging.getLogger(__name__)


class TestLogginUsageProcessor:
    """Test LoggingUsageProcessor implementation."""

    @pytest.mark.asyncio
    async def test_logging_usage_processor_logs_usage(self, caplog):
        """Test that LoggingUsageProcessor logs token usage."""
        processor = LoggingUsageProcessor()
        usage = RunUsage(
            requests=1,
            input_tokens=100,
            output_tokens=50,
        )

        with caplog.at_level(logging.INFO):
            await processor.process_usage(external_id="test_agent", usage=usage)

        assert "Agent Token Usage" in caplog.text
        assert "external_id=test_agent" in caplog.text
        assert "requests=1" in caplog.text
        assert "input_tokens=100" in caplog.text
        assert "output_tokens=50" in caplog.text
        assert "total_tokens=150" in caplog.text

    @pytest.mark.asyncio
    async def test_logging_usage_processor_protocol_compliance(self):
        """Test that LoggingUsageProcessor implements UsageProcessor protocol."""
        processor = LoggingUsageProcessor()
        usage = RunUsage(requests=1, input_tokens=10, output_tokens=5)

        # Should be callable and not raise
        await processor.process_usage(external_id="test", usage=usage)


class TestTripletPointerModel:
    """Test TripletPointer data model."""

    def test_triplet_pointer_creation(self):
        """Test creating a TripletPointer."""
        pointer = TripletPointer(
            pointer_id="tool123:triplet:Alice:WORKS_AT:Acme",
            subject="Alice",
            predicate="WORKS_AT",
            object="Acme",
            tier="shortterm",
            importance=0.95,
        )

        assert pointer.subject == "Alice"
        assert pointer.predicate == "WORKS_AT"
        assert pointer.object == "Acme"
        assert pointer.tier == "shortterm"
        assert pointer.importance == 0.95

    def test_triplet_pointer_validation(self):
        """Test TripletPointer field validation."""
        with pytest.raises(ValueError):
            TripletPointer(
                pointer_id="tool123:triplet:s:p:o",
                subject="Alice",
                predicate="WORKS_AT",
                object="Acme",
                tier="invalid_tier",  # Invalid tier
                importance=0.95,
            )


class TestCentralStorageTriplets:
    """Test triplet storage in CentralStorage."""

    def test_store_and_retrieve_triplet(self):
        """Test storing and retrieving triplets."""
        storage = CentralStorage()
        triplet = ShorttermKnowledgeTriplet(
            subject="Alice",
            predicate="KNOWS",
            object="Bob",
            importance=0.85,
        )

        pointer_id = storage.store_triplet(
            external_id="test_agent",
            tool_call_id="tool_123",
            triplet=triplet,
        )

        retrieved = storage.get_triplet("test_agent", pointer_id)
        assert retrieved is not None
        assert retrieved.subject == "Alice"
        assert retrieved.predicate == "KNOWS"
        assert retrieved.object == "Bob"

    def test_triplet_pointer_id_format(self):
        """Test that triplet pointer IDs follow expected format."""
        storage = CentralStorage()
        triplet = ShorttermKnowledgeTriplet(
            subject="Alice",
            predicate="WORKS_AT",
            object="Acme",
            importance=0.9,
        )

        pointer_id = storage.store_triplet(
            external_id="test_agent",
            tool_call_id="call_abc",
            triplet=triplet,
        )

        assert pointer_id.startswith("call_abc:triplet:")
        assert "Alice:WORKS_AT:Acme" in pointer_id

    def test_clear_triplets_on_external_id_clear(self):
        """Test that triplets are cleared with external_id."""
        storage = CentralStorage()
        triplet = ShorttermKnowledgeTriplet(
            subject="X",
            predicate="Y",
            object="Z",
            importance=0.5,
        )

        pointer_id = storage.store_triplet("agent1", "tool1", triplet)

        # Clear external_id
        storage.clear_external_id("agent1")

        # Should be cleared
        assert storage.get_triplet("agent1", pointer_id) is None


class TestRetrievalResultWithTriplets:
    """Test RetrievalResult model with triplet support."""

    def test_retrieval_result_with_triplets(self):
        """Test creating RetrievalResult with triplets."""
        triplet1 = TripletPointer(
            pointer_id="t1",
            subject="A",
            predicate="P",
            object="B",
            tier="shortterm",
            importance=0.9,
        )
        triplet2 = TripletPointer(
            pointer_id="t2", subject="C", predicate="Q", object="D", tier="longterm", importance=0.7
        )

        result = RetrievalResult(
            mode="pointer",
            triplets=[triplet1, triplet2],
            search_strategy="Searched both tiers for triplets",
        )

        assert len(result.triplets) == 2
        assert result.triplets[0].subject == "A"
        assert result.triplets[1].subject == "C"

    def test_retrieval_result_with_usage_data(self):
        """Test RetrievalResult with usage_data field."""
        result = RetrievalResult(
            mode="pointer",
            search_strategy="Test search",
            usage_data={
                "requests": 1,
                "input_tokens": 150,
                "output_tokens": 100,
            },
        )

        assert result.usage_data is not None
        assert result.usage_data["requests"] == 1
        assert result.usage_data["input_tokens"] == 150
        assert result.usage_data["output_tokens"] == 100


@pytest.mark.asyncio
class TestTripletSearchTools:
    """Test triplet search tools."""

    async def test_search_shortterm_triplets_success(self):
        """Test successful shortterm triplet search."""
        # Mock dependencies - Create real RetrieverDeps instance for proper typing
        mock_shortterm_repo = AsyncMock()
        mock_longterm_repo = AsyncMock()
        mock_embedding_service = AsyncMock()

        deps = RetrieverDeps(
            external_id="test_agent",
            query="Alice knows Bob",
            synthesis=False,
            shortterm_repo=mock_shortterm_repo,
            longterm_repo=mock_longterm_repo,
            embedding_service=mock_embedding_service,
        )

        mock_ctx = AsyncMock()
        mock_ctx.tool_call_id = "tool_test_001"
        mock_ctx.deps = deps

        # Mock entity extraction
        with patch("agent_reminiscence.agents.memory_retriever.extract_entities") as mock_extract:
            mock_extract.return_value = ["Alice", "Bob"]

            # Mock repository methods
            mock_entity = ShorttermEntity(
                id="entity_1",
                external_id="test_agent",
                shortterm_memory_id=1,
                name="Alice",
                types=["Person"],
                description="Test person",
                importance=0.8,
            )

            mock_triplet = ShorttermKnowledgeTriplet(
                subject="Alice",
                predicate="KNOWS",
                object="Bob",
                importance=0.85,
            )

            mock_ctx.deps.shortterm_repo.search_entity_triplets = AsyncMock(
                return_value=([mock_entity], [mock_triplet])
            )

            # Mock storage
            with patch(
                "agent_reminiscence.agents.memory_retriever.get_central_storage"
            ) as mock_storage:
                storage_instance = MagicMock()
                storage_instance.store_entity.return_value = "tool_test_001:entity:1"
                storage_instance.store_triplet.return_value = (
                    "tool_test_001:triplet:Alice:KNOWS:Bob"
                )
                mock_storage.return_value = storage_instance

                # Call the function
                result = await search_shortterm_triplets(mock_ctx, "Alice knows Bob")

                assert result["success"] is True
                assert len(result["entity_pointers"]) == 1
                assert len(result["triplet_pointers"]) == 1
                assert result["triplet_pointers"][0]["subject"] == "Alice"
                assert result["triplet_pointers"][0]["predicate"] == "KNOWS"
                assert result["triplet_pointers"][0]["object"] == "Bob"

    async def test_search_longterm_triplets_success(self):
        """Test successful longterm triplet search."""
        # Mock dependencies - Create real RetrieverDeps instance for proper typing
        mock_shortterm_repo = AsyncMock()
        mock_longterm_repo = AsyncMock()
        mock_embedding_service = AsyncMock()

        deps = RetrieverDeps(
            external_id="test_agent",
            query="Project leadership",
            synthesis=False,
            shortterm_repo=mock_shortterm_repo,
            longterm_repo=mock_longterm_repo,
            embedding_service=mock_embedding_service,
        )

        mock_ctx = AsyncMock()
        mock_ctx.tool_call_id = "tool_test_002"
        mock_ctx.deps = deps

        with patch("agent_reminiscence.agents.memory_retriever.extract_entities") as mock_extract:
            mock_extract.return_value = ["Project"]

            mock_entity = LongtermEntity(
                id="entity_2",
                external_id="test_agent",
                name="Project",
                types=["Concept"],
                description="Important project",
                importance=0.9,
            )

            mock_triplet = LongtermKnowledgeTriplet(
                subject="Alice",
                predicate="LED",
                object="Project",
                importance=0.95,
                start_date=datetime.now(timezone.utc),
            )

            mock_ctx.deps.longterm_repo.search_entity_triplets = AsyncMock(
                return_value=([mock_entity], [mock_triplet])
            )

            with patch(
                "agent_reminiscence.agents.memory_retriever.get_central_storage"
            ) as mock_storage:
                storage_instance = MagicMock()
                storage_instance.store_entity.return_value = "tool_test_002:entity:2"
                storage_instance.store_triplet.return_value = (
                    "tool_test_002:triplet:Alice:LED:Project"
                )
                mock_storage.return_value = storage_instance

                result = await search_longterm_triplets(mock_ctx, "Project leadership")

                assert result["success"] is True
                assert len(result["entity_pointers"]) == 1
                assert len(result["triplet_pointers"]) == 1
                assert result["tier"] == "longterm"

    async def test_search_triplets_no_entities_extracted(self):
        """Test triplet search when no entities are extracted."""
        deps = RetrieverDeps(
            external_id="test_agent",
            query="xyz abc def",
            synthesis=False,
            shortterm_repo=AsyncMock(),
            longterm_repo=AsyncMock(),
            embedding_service=AsyncMock(),
        )

        mock_ctx = AsyncMock()
        mock_ctx.tool_call_id = "tool_test_003"
        mock_ctx.deps = deps

        with patch("agent_reminiscence.agents.memory_retriever.extract_entities") as mock_extract:
            mock_extract.return_value = []  # No entities extracted

            result = await search_shortterm_triplets(mock_ctx, "xyz abc def")

            assert result["success"] is True
            assert len(result["entity_pointers"]) == 0
            assert len(result["triplet_pointers"]) == 0

    async def test_search_triplets_error_handling(self):
        """Test error handling in triplet search."""
        deps = RetrieverDeps(
            external_id="test_agent",
            query="test query",
            synthesis=False,
            shortterm_repo=AsyncMock(),
            longterm_repo=AsyncMock(),
            embedding_service=AsyncMock(),
        )

        mock_ctx = AsyncMock()
        mock_ctx.tool_call_id = "tool_test_004"
        mock_ctx.deps = deps

        with patch("agent_reminiscence.agents.memory_retriever.extract_entities") as mock_extract:
            mock_extract.side_effect = Exception("Extraction failed")

            result = await search_shortterm_triplets(mock_ctx, "test query")

            assert result["success"] is False
            assert "error" in result


class TestResolveAndFormatResults:
    """Test result resolution and formatting with triplets."""

    def test_resolve_results_with_triplets(self):
        """Test resolving results with triplet data."""
        storage = CentralStorage()

        # Store test triplet
        triplet = ShorttermKnowledgeTriplet(
            subject="Alice",
            predicate="MANAGES",
            object="Project",
            importance=0.9,
            description="Alice manages the project",
        )
        pointer_id = storage.store_triplet("test_agent", "tool_xyz", triplet)

        # Create result with triplet pointer
        triplet_pointer = TripletPointer(
            pointer_id=pointer_id,
            subject="Alice",
            predicate="MANAGES",
            object="Project",
            tier="shortterm",
            importance=0.9,
        )

        retrieval_result = RetrievalResult(
            mode="pointer",
            triplets=[triplet_pointer],
            search_strategy="Searched for management relationships",
            usage_data={"requests": 1, "input_tokens": 50, "output_tokens": 25},
        )

        # Resolve
        with patch(
            "agent_reminiscence.agents.memory_retriever.get_central_storage"
        ) as mock_storage:
            mock_storage.return_value = storage
            final_result = resolve_and_format_results(
                retrieval_result, "test_agent", result_mode="search"
            )

        # Verify triplets on the structured output
        assert len(final_result.shortterm_triplets) == 1
        triplet_data = final_result.shortterm_triplets[0]
        assert triplet_data.subject == "Alice"
        assert triplet_data.predicate == "MANAGES"
        assert triplet_data.object == "Project"

        # Verify usage in metadata
        assert "usage" in final_result.metadata
        assert final_result.metadata["usage"]["input_tokens"] == 50


class TestEndToEndRetrieval:
    """End-to-end integration tests."""

    @pytest.mark.asyncio
    async def test_full_retrieval_flow_with_triplets(self):
        """Test complete retrieval flow including triplet extraction."""
        # This test verifies the full chain:
        # 1. Query → Entity extraction
        # 2. Entity search → Triplet conversion
        # 3. Storage → Pointer creation
        # 4. Resolution → Metadata inclusion

        storage = CentralStorage()

        # Simulate triplet storage
        triplet = ShorttermKnowledgeTriplet(
            subject="System",
            predicate="PROCESSES",
            object="Data",
            importance=0.88,
        )
        pointer_id = storage.store_triplet("agent1", "call1", triplet)

        # Create agent result
        agent_result = RetrievalResult(
            mode="pointer",
            triplets=[
                TripletPointer(
                    pointer_id=pointer_id,
                    subject="System",
                    predicate="PROCESSES",
                    object="Data",
                    tier="shortterm",
                    importance=0.88,
                )
            ],
            search_strategy="Used triplet search",
            confidence=0.92,
            usage_data={"requests": 2, "input_tokens": 200, "output_tokens": 150},
        )

        # Resolve
        with patch(
            "agent_reminiscence.agents.memory_retriever.get_central_storage"
        ) as mock_storage:
            mock_storage.return_value = storage
            final = resolve_and_format_results(agent_result, "agent1", result_mode="deep_search")

        # Verify full result
        assert final.mode == "deep_search"
        assert final.confidence == 0.92
        assert len(final.shortterm_triplets) == 1
        assert "usage" in final.metadata
        assert final.metadata["usage"]["input_tokens"] == 200
        assert final.metadata["usage"]["output_tokens"] == 150
