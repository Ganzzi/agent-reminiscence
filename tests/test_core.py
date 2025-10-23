"""
Tests for AgentMem Core Interface (core.py).
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4
from datetime import datetime, timezone

from agent_reminiscence.core import AgentMem
from agent_reminiscence.database.models import ActiveMemory


class TestAgentMemInit:
    """Test AgentMem initialization."""

    @pytest.mark.asyncio
    async def test_initialization_with_config(self, test_config):
        """Test initialization with config."""
        with patch("agent_reminiscence.core.MemoryManager") as mock_mm:

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
            patch("agent_reminiscence.core.MemoryManager") as mock_mm,
            patch("agent_reminiscence.core.get_config") as mock_get_config,
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
        with patch("agent_reminiscence.core.MemoryManager") as mock_mm:

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
        """Test creating active memory with dict template."""
        with patch("agent_reminiscence.core.MemoryManager") as mock_mm:

            memory = ActiveMemory(
                id=1,
                external_id="test-123",
                title="Test Memory",
                template_content={
                    "template": {"id": "test-template", "name": "Test Template"},
                    "sections": [{"id": "summary", "description": "Summary section"}]
                },
                sections={
                    "summary": {
                        "content": "Test", 
                        "update_count": 0,
                        "awake_update_count": 0,
                        "last_updated": None
                    }
                },
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
                template_content={
                    "template": {"id": "test-template", "name": "Test Template"},
                    "sections": [{"id": "summary", "description": "Summary section"}]
                },
                initial_sections={
                    "summary": {
                        "content": "Test", 
                        "update_count": 0,
                        "awake_update_count": 0,
                        "last_updated": None
                    }
                },
                metadata={},
            )

            assert result == memory
            mock_mm_instance.create_active_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_active_memories(self, test_config):
        """Test getting active memories."""
        with patch("agent_reminiscence.core.MemoryManager") as mock_mm:

            memories = [
                ActiveMemory(
                    id=1,
                    external_id="test-123",
                    title="Test Memory 1",
                    template_content={
                        "template": {"id": "template-1", "name": "Template 1"},
                        "sections": [{"id": "summary", "description": "Summary"}]
                    },
                    sections={
                        "summary": {
                            "content": "Test 1", 
                            "update_count": 0,
                            "awake_update_count": 0,
                            "last_updated": None
                        }
                    },
                    metadata={},
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc),
                ),
                ActiveMemory(
                    id=2,
                    external_id="test-123",
                    title="Test Memory 2",
                    template_content={
                        "template": {"id": "template-2", "name": "Template 2"},
                        "sections": [{"id": "summary", "description": "Summary"}]
                    },
                    sections={
                        "summary": {
                            "content": "Test 2", 
                            "update_count": 0,
                            "awake_update_count": 0,
                            "last_updated": None
                        }
                    },
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
        with patch("agent_reminiscence.core.MemoryManager") as mock_mm:

            memory_id = 1
            updated_memory = ActiveMemory(
                id=memory_id,
                external_id="test-123",
                title="Test Memory",
                template_content={
                    "template": {"id": "test-template", "name": "Test Template"},
                    "sections": [{"id": "summary", "description": "Summary section"}]
                },
                sections={
                    "summary": {
                        "content": "Updated", 
                        "update_count": 1,
                        "awake_update_count": 1,
                        "last_updated": datetime.now(timezone.utc)
                    }
                },
                metadata={},
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )

            mock_mm_instance = MagicMock()
            mock_mm_instance.update_active_memory_sections = AsyncMock(return_value=updated_memory)
            mock_mm.return_value = mock_mm_instance

            agent_mem = AgentMem(config=test_config)
            agent_mem._initialized = True  # Bypass initialization check
            agent_mem._memory_manager = mock_mm_instance  # Set mock manager

            result = await agent_mem.update_active_memory_sections(
                external_id="test-123",
                memory_id=memory_id,
                sections=[{"section_id": "summary", "new_content": "Updated"}],
            )

            assert result == updated_memory
            assert result.sections["summary"]["update_count"] == 1

    @pytest.mark.asyncio
    async def test_create_active_memory_yaml_backward_compatibility(self, test_config):
        """Test creating active memory with YAML string (backward compatibility)."""
        with patch("agent_reminiscence.core.MemoryManager") as mock_mm:

            memory = ActiveMemory(
                id=1,
                external_id="test-123",
                title="Test Memory",
                template_content={
                    "template": {"id": "test-template", "name": "Test Template"},
                    "sections": [{"id": "summary", "description": "Summary section"}]
                },
                sections={
                    "summary": {
                        "content": "Test", 
                        "update_count": 0,
                        "awake_update_count": 0,
                        "last_updated": None
                    }
                },
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

            # Test with YAML string (should be converted to dict)
            yaml_template = """
template:
  id: "test-template"
  name: "Test Template"
sections:
  - id: "summary"
    description: "Summary section"
"""

            result = await agent_mem.create_active_memory(
                external_id="test-123",
                title="Test Memory",
                template_content=yaml_template,
                initial_sections={
                    "summary": {
                        "content": "Test", 
                        "update_count": 0,
                        "awake_update_count": 0,
                        "last_updated": None
                    }
                },
                metadata={},
            )

            assert result == memory
            # Verify the YAML was converted to dict before passing to manager
            call_args = mock_mm_instance.create_active_memory.call_args
            template_arg = call_args[1]['template_content']
            assert isinstance(template_arg, dict)
            assert template_arg["template"]["id"] == "test-template"

    @pytest.mark.asyncio
    async def test_update_active_memory_sections_upsert(self, test_config):
        """Test upserting active memory sections with new schema."""
        with patch("agent_reminiscence.core.MemoryManager") as mock_mm:

            memory_id = 1
            updated_memory = ActiveMemory(
                id=memory_id,
                external_id="test-123",
                title="Test Memory",
                template_content={
                    "template": {"id": "test-template", "name": "Test Template"},
                    "sections": [
                        {"id": "summary", "description": "Summary section"},
                        {"id": "new_section", "description": "Dynamically added section"}
                    ]
                },
                sections={
                    "summary": {
                        "content": "Updated content", 
                        "update_count": 1,
                        "awake_update_count": 1,
                        "last_updated": "2025-10-14T10:30:00Z"
                    },
                    "new_section": {
                        "content": "New section content",
                        "update_count": 0,
                        "awake_update_count": 0,
                        "last_updated": "2025-10-14T10:30:00Z"
                    }
                },
                metadata={},
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )

            mock_mm_instance = MagicMock()
            mock_mm_instance.update_active_memory_sections = AsyncMock(return_value=updated_memory)
            mock_mm.return_value = mock_mm_instance

            agent_mem = AgentMem(config=test_config)
            agent_mem._initialized = True  # Bypass initialization check
            agent_mem._memory_manager = mock_mm_instance  # Set mock manager

            # Test upsert with replace action
            result = await agent_mem.update_active_memory_sections(
                external_id="test-123",
                memory_id=memory_id,
                sections=[
                    {
                        "section_id": "summary", 
                        "old_content": "old text",
                        "new_content": "Updated content",
                        "action": "replace"
                    },
                    {
                        "section_id": "new_section",
                        "new_content": "New section content",
                        "action": "replace"
                    }
                ],
            )

            assert result == updated_memory
            assert result.sections["summary"]["update_count"] == 1
            assert result.sections["summary"]["awake_update_count"] == 1
            assert result.sections["new_section"]["update_count"] == 0  # New section
            assert result.sections["new_section"]["awake_update_count"] == 0

    @pytest.mark.asyncio
    async def test_update_active_memory_sections_insert_action(self, test_config):
        """Test inserting content into active memory sections."""
        with patch("agent_reminiscence.core.MemoryManager") as mock_mm:

            memory_id = 1
            updated_memory = ActiveMemory(
                id=memory_id,
                external_id="test-123",
                title="Test Memory",
                template_content={
                    "template": {"id": "test-template", "name": "Test Template"},
                    "sections": [{"id": "summary", "description": "Summary section"}]
                },
                sections={
                    "summary": {
                        "content": "Original content\nNew appended content", 
                        "update_count": 1,
                        "awake_update_count": 1,
                        "last_updated": "2025-10-14T10:30:00Z"
                    }
                },
                metadata={},
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )

            mock_mm_instance = MagicMock()
            mock_mm_instance.update_active_memory_sections = AsyncMock(return_value=updated_memory)
            mock_mm.return_value = mock_mm_instance

            agent_mem = AgentMem(config=test_config)
            agent_mem._initialized = True  # Bypass initialization check
            agent_mem._memory_manager = mock_mm_instance  # Set mock manager

            # Test insert action (append without pattern)
            result = await agent_mem.update_active_memory_sections(
                external_id="test-123",
                memory_id=memory_id,
                sections=[
                    {
                        "section_id": "summary", 
                        "new_content": "\nNew appended content",
                        "action": "insert"
                    }
                ],
            )

            assert result == updated_memory
            assert "Original content" in result.sections["summary"]["content"]
            assert "New appended content" in result.sections["summary"]["content"]


class TestAgentMemRetrieval:
    """Test memory retrieval."""

    @pytest.mark.asyncio
    async def test_retrieve_memories(self, test_config):
        """Test retrieving memories."""
        with patch("agent_reminiscence.core.MemoryManager") as mock_mm:

            from agent_reminiscence.database.models import RetrievalResult

            result_obj = RetrievalResult(
                mode="synthesis",
                chunks=[],
                entities=[],
                relationships=[],
                synthesis="Retrieved memories about AI and machine learning.",
                search_strategy="Hybrid search across shortterm and longterm tiers",
                confidence=0.95,
                metadata={"total_results": 5},
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
            assert result.mode == "synthesis"
            assert result.synthesis == "Retrieved memories about AI and machine learning."
            assert result.confidence == 0.95
            mock_mm_instance.retrieve_memories.assert_called_once()

    @pytest.mark.asyncio
    async def test_retrieve_memories_with_filters(self, test_config):
        """Test retrieving memories with search parameters."""
        with patch("agent_reminiscence.core.MemoryManager") as mock_mm:

            from agent_reminiscence.database.models import RetrievalResult

            result_obj = RetrievalResult(
                mode="synthesis",
                chunks=[],
                entities=[],
                relationships=[],
                synthesis="Important memories about AI.",
                search_strategy="Filtered search with limit=5",
                confidence=0.88,
                metadata={"limit": 5},
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
                limit=5,
            )

            assert result == result_obj
            assert result.synthesis == "Important memories about AI."


class TestUpsertFunctionality:
    """Test upsert functionality in core API."""

    @pytest.mark.asyncio
    async def test_upsert_active_memory_sections_replace(self, test_config):
        """Test upserting active memory sections with replace action."""
        with patch("agent_reminiscence.core.MemoryManager") as mock_mm:

            memory_id = 1
            updated_memory = ActiveMemory(
                id=memory_id,
                external_id="test-123",
                title="Test Memory",
                template_content={
                    "template": {"id": "test-template", "name": "Test Template"},
                    "sections": [{"id": "summary", "description": "Summary section"}]
                },
                sections={
                    "summary": {
                        "content": "Replaced content",
                        "update_count": 2,
                        "awake_update_count": 2,
                        "last_updated": datetime.now(timezone.utc)
                    }
                },
                metadata={},
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )

            mock_mm_instance = MagicMock()
            mock_mm_instance.update_active_memory_sections = AsyncMock(return_value=updated_memory)
            mock_mm.return_value = mock_mm_instance

            agent_mem = AgentMem(config=test_config)
            agent_mem._initialized = True  # Bypass initialization check
            agent_mem._memory_manager = mock_mm_instance  # Set mock manager

            result = await agent_mem.update_active_memory_sections(
                external_id="test-123",
                memory_id=memory_id,
                sections=[{
                    "section_id": "summary",
                    "new_content": "Replaced content",
                    "action": "replace",
                    "old_content": "Old content"
                }],
            )

            assert result == updated_memory
            assert result.sections["summary"]["content"] == "Replaced content"
            assert result.sections["summary"]["update_count"] == 2
            assert result.sections["summary"]["awake_update_count"] == 2

    @pytest.mark.asyncio
    async def test_upsert_active_memory_sections_insert(self, test_config):
        """Test upserting active memory sections with insert action."""
        with patch("agent_reminiscence.core.MemoryManager") as mock_mm:

            memory_id = 1
            updated_memory = ActiveMemory(
                id=memory_id,
                external_id="test-123",
                title="Test Memory",
                template_content={
                    "template": {"id": "test-template", "name": "Test Template"},
                    "sections": [
                        {"id": "summary", "description": "Summary section"},
                        {"id": "notes", "description": "Notes section"}
                    ]
                },
                sections={
                    "summary": {
                        "content": "Original content",
                        "update_count": 0,
                        "awake_update_count": 0,
                        "last_updated": None
                    },
                    "notes": {
                        "content": "New notes content",
                        "update_count": 1,
                        "awake_update_count": 1,
                        "last_updated": datetime.now(timezone.utc)
                    }
                },
                metadata={},
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )

            mock_mm_instance = MagicMock()
            mock_mm_instance.update_active_memory_sections = AsyncMock(return_value=updated_memory)
            mock_mm.return_value = mock_mm_instance

            agent_mem = AgentMem(config=test_config)
            agent_mem._initialized = True  # Bypass initialization check
            agent_mem._memory_manager = mock_mm_instance  # Set mock manager

            result = await agent_mem.update_active_memory_sections(
                external_id="test-123",
                memory_id=memory_id,
                sections=[{
                    "section_id": "notes",
                    "new_content": "New notes content",
                    "action": "insert"
                }],
            )

            assert result == updated_memory
            assert "notes" in result.sections
            assert result.sections["notes"]["content"] == "New notes content"
            assert result.sections["notes"]["update_count"] == 1
            assert result.sections["notes"]["awake_update_count"] == 1


class TestAgentMemErrorHandling:
    """Test error handling."""

    @pytest.mark.asyncio
    async def test_error_on_uninitialized_operation(self, test_config):
        """Test error when performing operations before initialization."""
        with patch("agent_reminiscence.core.MemoryManager"):

            agent_mem = AgentMem(config=test_config)
            # AgentMem should handle uninitialized state
            pass

    @pytest.mark.asyncio
    async def test_close_without_initialize(self, test_config):
        """Test closing without initializing."""
        with patch("agent_reminiscence.core.MemoryManager") as mock_mm:

            mock_mm_instance = MagicMock()
            mock_mm_instance.close = AsyncMock()
            mock_mm.return_value = mock_mm_instance

            agent_mem = AgentMem(config=test_config)
            await agent_mem.close()

            # Should not raise error
            assert True


