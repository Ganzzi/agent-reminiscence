"""
Tests for MCP Server Batch Update Functionality.

Tests the update_memory_sections tool in the MCP server.
"""

import pytest
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from agent_mem import AgentMem
from agent_mem.database.models import ActiveMemory


class TestMCPBatchUpdate:
    """Test MCP server batch update handler."""

    @pytest.mark.asyncio
    async def test_handle_update_memory_sections_success(self):
        """Test successful batch update through MCP server."""
        # Mock AgentMem instance
        mock_agent_mem = MagicMock(spec=AgentMem)

        # Mock current memory state
        current_memory = ActiveMemory(
            id=1,
            external_id="agent-123",
            title="Test Memory",
            template_content="",
            sections={
                "progress": {"content": "Old progress", "update_count": 0},
                "notes": {"content": "Old notes", "update_count": 0},
                "blockers": {"content": "No blockers", "update_count": 0},
            },
            metadata={},
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        # Mock updated memory state
        updated_memory = ActiveMemory(
            id=1,
            external_id="agent-123",
            title="Test Memory",
            template_content="",
            sections={
                "progress": {"content": "New progress", "update_count": 1},
                "notes": {"content": "New notes", "update_count": 1},
                "blockers": {"content": "No blockers", "update_count": 0},
            },
            metadata={},
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        # Mock methods
        mock_agent_mem.get_active_memories = AsyncMock(return_value=[current_memory])
        mock_agent_mem.update_active_memory_sections = AsyncMock(return_value=updated_memory)

        # Import handler after mocking
        from agent_mem_mcp.server import _handle_update_memory_sections

        # Prepare arguments
        arguments = {
            "external_id": "agent-123",
            "memory_id": 1,
            "sections": [
                {"section_id": "progress", "new_content": "New progress"},
                {"section_id": "notes", "new_content": "New notes"},
            ],
        }

        # Call handler
        result = await _handle_update_memory_sections(mock_agent_mem, arguments)

        # Verify result
        assert len(result) == 1
        assert result[0].type == "text"

        # Parse response JSON
        response = json.loads(result[0].text)

        # Verify response structure
        assert "memory" in response
        assert "updates" in response
        assert "total_sections_updated" in response
        assert "consolidation_info" in response
        assert "message" in response

        # Verify memory data
        assert response["memory"]["id"] == 1
        assert response["memory"]["external_id"] == "agent-123"

        # Verify updates
        assert response["total_sections_updated"] == 2
        assert len(response["updates"]) == 2

        # Verify consolidation info
        assert "total_update_count" in response["consolidation_info"]
        assert "threshold" in response["consolidation_info"]
        assert "will_consolidate" in response["consolidation_info"]

        # Verify the batch method was called (not loop)
        mock_agent_mem.update_active_memory_sections.assert_called_once_with(
            external_id="agent-123",
            memory_id=1,
            sections=arguments["sections"],
        )

    @pytest.mark.asyncio
    async def test_handle_update_memory_sections_validation(self):
        """Test validation in batch update handler."""
        mock_agent_mem = MagicMock(spec=AgentMem)

        from agent_mem_mcp.server import _handle_update_memory_sections

        # Test empty external_id
        with pytest.raises(ValueError, match="external_id cannot be empty"):
            await _handle_update_memory_sections(
                mock_agent_mem,
                {"external_id": "", "memory_id": 1, "sections": []},
            )

        # Test empty sections array
        with pytest.raises(ValueError, match="sections array cannot be empty"):
            await _handle_update_memory_sections(
                mock_agent_mem,
                {"external_id": "agent-123", "memory_id": 1, "sections": []},
            )

    @pytest.mark.asyncio
    async def test_handle_update_memory_sections_invalid_section(self):
        """Test error when section doesn't exist."""
        mock_agent_mem = MagicMock(spec=AgentMem)

        # Mock memory with limited sections
        current_memory = ActiveMemory(
            id=1,
            external_id="agent-123",
            title="Test Memory",
            template_content="",
            sections={
                "progress": {"content": "Content", "update_count": 0},
                "notes": {"content": "Content", "update_count": 0},
            },
            metadata={},
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        mock_agent_mem.get_active_memories = AsyncMock(return_value=[current_memory])

        from agent_mem_mcp.server import _handle_update_memory_sections

        # Try to update non-existent section
        arguments = {
            "external_id": "agent-123",
            "memory_id": 1,
            "sections": [
                {"section_id": "invalid_section", "new_content": "New content"},
            ],
        }

        with pytest.raises(ValueError, match="Section 'invalid_section' not found"):
            await _handle_update_memory_sections(mock_agent_mem, arguments)

    @pytest.mark.asyncio
    async def test_handle_update_memory_sections_empty_content(self):
        """Test error when new_content is empty."""
        mock_agent_mem = MagicMock(spec=AgentMem)

        # Mock memory
        current_memory = ActiveMemory(
            id=1,
            external_id="agent-123",
            title="Test Memory",
            template_content="",
            sections={
                "progress": {"content": "Content", "update_count": 0},
            },
            metadata={},
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        mock_agent_mem.get_active_memories = AsyncMock(return_value=[current_memory])

        from agent_mem_mcp.server import _handle_update_memory_sections

        # Try to update with empty content
        arguments = {
            "external_id": "agent-123",
            "memory_id": 1,
            "sections": [
                {"section_id": "progress", "new_content": "   "},  # Empty after strip
            ],
        }

        with pytest.raises(ValueError, match="new_content.*cannot be empty"):
            await _handle_update_memory_sections(mock_agent_mem, arguments)

    @pytest.mark.asyncio
    async def test_consolidation_info_calculation(self):
        """Test that consolidation info is calculated correctly."""
        mock_agent_mem = MagicMock(spec=AgentMem)

        # Mock memory with high update counts
        current_memory = ActiveMemory(
            id=1,
            external_id="agent-123",
            title="Test Memory",
            template_content="",
            sections={
                "section1": {"content": "Content", "update_count": 0},
                "section2": {"content": "Content", "update_count": 0},
                "section3": {"content": "Content", "update_count": 0},
            },
            metadata={},
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        # After update, counts are high
        updated_memory = ActiveMemory(
            id=1,
            external_id="agent-123",
            title="Test Memory",
            template_content="",
            sections={
                "section1": {"content": "New content", "update_count": 5},
                "section2": {"content": "New content", "update_count": 5},
                "section3": {"content": "New content", "update_count": 5},
            },
            metadata={},
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        mock_agent_mem.get_active_memories = AsyncMock(return_value=[current_memory])
        mock_agent_mem.update_active_memory_sections = AsyncMock(return_value=updated_memory)

        from agent_mem_mcp.server import _handle_update_memory_sections

        arguments = {
            "external_id": "agent-123",
            "memory_id": 1,
            "sections": [
                {"section_id": "section1", "new_content": "New content"},
                {"section_id": "section2", "new_content": "New content"},
                {"section_id": "section3", "new_content": "New content"},
            ],
        }

        result = await _handle_update_memory_sections(mock_agent_mem, arguments)
        response = json.loads(result[0].text)

        # Check consolidation info
        consolidation_info = response["consolidation_info"]
        assert consolidation_info["total_update_count"] == 15  # 5 + 5 + 5
        # With avg_section_update_count=5.0 and 3 sections, threshold = 15.0
        assert consolidation_info["will_consolidate"] is True


class TestMCPIntegration:
    """Integration tests for MCP server with full AgentMem."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_mcp_batch_update_full_flow(self):
        """
        Test full MCP batch update flow with real AgentMem instance.

        This would require a test database and full setup.
        Marked as integration test to run separately.
        """
        # This test would require:
        # 1. Test database setup
        # 2. Real AgentMem initialization
        # 3. Real MCP server handler
        # 4. Cleanup after test

        # For now, this is a placeholder showing the structure
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
