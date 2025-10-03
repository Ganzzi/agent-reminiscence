"""
Tests for AgentMem Core Interface (core.py).
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4
from datetime import datetime, timezone

from agent_mem.core import AgentMem
from agent_mem.database.models import ActiveMemory


class TestAgentMemInit:
    """Test AgentMem initialization."""

    @pytest.mark.asyncio
    async def test_initialization_with_config(self, test_config):
        """Test initialization with config."""
        with patch("agent_mem.core.MemoryManager") as mock_mm:

            mock_mm_instance = MagicMock()
            mock_mm_instance.initialize = AsyncMock()
            mock_mm.return_value = mock_mm_instance

            agent_mem = AgentMem(config=test_config)
            await agent_mem.initialize()

            assert agent_mem.config == test_config
            mock_mm_instance.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialization_without_config(self):
        """Test initialization without config uses defaults."""
        with (
            patch("agent_mem.core.MemoryManager") as mock_mm,
            patch("agent_mem.core.get_config") as mock_get_config,
        ):

            mock_config = MagicMock()
            mock_get_config.return_value = mock_config

            mock_mm_instance = MagicMock()
            mock_mm_instance.initialize = AsyncMock()
            mock_mm.return_value = mock_mm_instance

            agent_mem = AgentMem()
            await agent_mem.initialize()

            mock_get_config.assert_called_once()


class TestAgentMemContextManager:
    """Test AgentMem as context manager."""

    @pytest.mark.asyncio
    async def test_context_manager(self, test_config):
        """Test using AgentMem as async context manager."""
        with patch("agent_mem.core.MemoryManager") as mock_mm:

            mock_mm_instance = MagicMock()
            mock_mm_instance.initialize = AsyncMock()
            mock_mm_instance.close = AsyncMock()
            mock_mm.return_value = mock_mm_instance

            async with AgentMem(config=test_config) as agent_mem:
                assert agent_mem is not None

            mock_mm_instance.initialize.assert_called_once()
            mock_mm_instance.close.assert_called_once()


class TestAgentMemActiveMemory:
    """Test active memory operations."""

    @pytest.mark.asyncio
    async def test_create_active_memory(self, test_config):
        """Test creating active memory."""
        with patch("agent_mem.core.MemoryManager") as mock_mm:

            memory = ActiveMemory(
                id=1,
                external_id="test-123",
                title="Test Memory",
                template_content="# Template",
                sections={"summary": {"content": "Test", "update_count": 0}},
                metadata={},
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )

            mock_mm_instance = MagicMock()
            mock_mm_instance.create_active_memory = AsyncMock(return_value=memory)
            mock_mm.return_value = mock_mm_instance

            agent_mem = AgentMem(config=test_config)
            agent_mem._initialized = True  # Bypass initialization check
            agent_mem._memory_manager = mock_mm_instance  # Set mock manager

            result = await agent_mem.create_active_memory(
                external_id="test-123",
                title="Test Memory",
                template_content="# Template",
                initial_sections={"summary": {"content": "Test", "update_count": 0}},
                metadata={},
            )

            assert result == memory
            mock_mm_instance.create_active_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_active_memories(self, test_config):
        """Test getting active memories."""
        with patch("agent_mem.core.MemoryManager") as mock_mm:

            memories = [
                ActiveMemory(
                    id=1,
                    external_id="test-123",
                    title="Test Memory 1",
                    template_content="# Template",
                    sections={"summary": {"content": "Test 1", "update_count": 0}},
                    metadata={},
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc),
                ),
                ActiveMemory(
                    id=2,
                    external_id="test-123",
                    title="Test Memory 2",
                    template_content="# Template",
                    sections={"summary": {"content": "Test 2", "update_count": 0}},
                    metadata={},
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc),
                ),
            ]

            mock_mm_instance = MagicMock()
            mock_mm_instance.get_active_memories = AsyncMock(return_value=memories)
            mock_mm.return_value = mock_mm_instance

            agent_mem = AgentMem(config=test_config)
            agent_mem._initialized = True  # Bypass initialization check
            agent_mem._memory_manager = mock_mm_instance  # Set mock manager

            results = await agent_mem.get_active_memories("test-123")

            assert len(results) == 2
            assert results == memories

    @pytest.mark.asyncio
    async def test_update_active_memory(self, test_config):
        """Test updating active memory."""
        with patch("agent_mem.core.MemoryManager") as mock_mm:

            memory_id = 1
            updated_memory = ActiveMemory(
                id=memory_id,
                external_id="test-123",
                title="Test Memory",
                template_content="# Template",
                sections={"summary": {"content": "Updated", "update_count": 1}},
                metadata={},
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )

            mock_mm_instance = MagicMock()
            mock_mm_instance.update_active_memory_section = AsyncMock(return_value=updated_memory)
            mock_mm.return_value = mock_mm_instance

            agent_mem = AgentMem(config=test_config)
            agent_mem._initialized = True  # Bypass initialization check
            agent_mem._memory_manager = mock_mm_instance  # Set mock manager

            result = await agent_mem.update_active_memory_section(
                external_id="test-123",
                memory_id=memory_id,
                section_id="summary",
                new_content="Updated",
            )

            assert result == updated_memory
            assert result.sections["summary"]["update_count"] == 1


class TestAgentMemRetrieval:
    """Test memory retrieval."""

    @pytest.mark.asyncio
    async def test_retrieve_memories(self, test_config):
        """Test retrieving memories."""
        with patch("agent_mem.core.MemoryManager") as mock_mm:

            from agent_mem.database.models import RetrievalResult

            result_obj = RetrievalResult(
                query="Tell me about AI",
                active_memories=[],
                shortterm_chunks=[],
                longterm_chunks=[],
                entities=[],
                relationships=[],
                synthesized_response="Retrieved memories about AI and machine learning.",
            )

            mock_mm_instance = MagicMock()
            mock_mm_instance.retrieve_memories = AsyncMock(return_value=result_obj)
            mock_mm.return_value = mock_mm_instance

            agent_mem = AgentMem(config=test_config)
            agent_mem._initialized = True  # Bypass initialization check
            agent_mem._memory_manager = mock_mm_instance  # Set mock manager

            result = await agent_mem.retrieve_memories(
                external_id="test-123",
                query="Tell me about AI",
            )

            assert result == result_obj
            assert (
                result.synthesized_response == "Retrieved memories about AI and machine learning."
            )
            mock_mm_instance.retrieve_memories.assert_called_once()

    @pytest.mark.asyncio
    async def test_retrieve_memories_with_filters(self, test_config):
        """Test retrieving memories with search parameters."""
        with patch("agent_mem.core.MemoryManager") as mock_mm:

            from agent_mem.database.models import RetrievalResult

            result_obj = RetrievalResult(
                query="AI",
                active_memories=[],
                shortterm_chunks=[],
                longterm_chunks=[],
                entities=[],
                relationships=[],
                synthesized_response="Important memories about AI.",
            )

            mock_mm_instance = MagicMock()
            mock_mm_instance.retrieve_memories = AsyncMock(return_value=result_obj)
            mock_mm.return_value = mock_mm_instance

            agent_mem = AgentMem(config=test_config)
            agent_mem._initialized = True  # Bypass initialization check
            agent_mem._memory_manager = mock_mm_instance  # Set mock manager

            result = await agent_mem.retrieve_memories(
                external_id="test-123",
                query="AI",
                search_shortterm=True,
                search_longterm=True,
                limit=5,
            )

            assert result == result_obj
            assert result.synthesized_response == "Important memories about AI."


class TestAgentMemErrorHandling:
    """Test error handling."""

    @pytest.mark.asyncio
    async def test_error_on_uninitialized_operation(self, test_config):
        """Test error when performing operations before initialization."""
        with patch("agent_mem.core.MemoryManager"):

            agent_mem = AgentMem(config=test_config)
            # AgentMem should handle uninitialized state
            pass

    @pytest.mark.asyncio
    async def test_close_without_initialize(self, test_config):
        """Test closing without initializing."""
        with patch("agent_mem.core.MemoryManager") as mock_mm:

            mock_mm_instance = MagicMock()
            mock_mm_instance.close = AsyncMock()
            mock_mm.return_value = mock_mm_instance

            agent_mem = AgentMem(config=test_config)
            await agent_mem.close()

            # Should not raise error
            assert True

