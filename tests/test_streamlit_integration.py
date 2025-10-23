"""
Integration Tests for Streamlit UI Memory Service

These tests validate the complete workflow from UI through to database.
Run with: pytest tests/test_streamlit_integration.py -v
"""

import pytest
import asyncio
import yaml
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from streamlit_app.services.memory_service import MemoryService
from streamlit_app.services.template_service import TemplateService
from agent_reminiscence import AgentMem
from agent_reminiscence.config import Config


# Test configuration - uses test database
TEST_DB_CONFIG = {
    "postgres_host": "localhost",
    "postgres_port": 5432,
    "postgres_db": "agent_mem_test",
    "postgres_user": "postgres",
    "postgres_password": "postgres",
    "neo4j_uri": "bolt://localhost:7687",
    "neo4j_user": "neo4j",
    "neo4j_password": "password",
}


@pytest.fixture
async def memory_service():
    """Create a memory service instance for testing"""
    service = MemoryService(TEST_DB_CONFIG)
    yield service
    # Cleanup after tests
    if service._agent_mem:
        await service._agent_mem.close()


@pytest.fixture
def template_service():
    """Create a template service for loading test templates"""
    templates_dir = Path(__file__).parent.parent / "prebuilt-memory-tmpl" / "bmad"
    return TemplateService(templates_dir)


@pytest.mark.asyncio
class TestMemoryServiceIntegration:
    """Integration tests for MemoryService"""

    async def test_connection(self, memory_service):
        """Test database connection"""
        is_connected, message = await memory_service.test_connection()
        assert is_connected, f"Failed to connect: {message}"

    async def test_create_memory_with_template(self, memory_service, template_service):
        """Test creating a memory using a pre-built template"""
        # Load a template
        templates = template_service.get_all_templates()
        assert len(templates) > 0, "No templates found"

        template = templates[0]
        external_id = "test-agent-create-001"

        # Prepare initial sections
        initial_sections = {}
        if template.get("sections"):
            for section in template["sections"][:2]:  # Use first 2 sections
                initial_sections[section["id"]] = {
                    "content": f"Test content for {section['id']}",
                    "update_count": 0,
                    "awake_update_count": 0,
                    "last_updated": None,
                }

        # Create memory
        memory = await memory_service.create_active_memory(
            external_id=external_id,
            title="Test Memory - Template Creation",
            template_content=template,
            initial_sections=initial_sections,
            metadata={"priority": "high", "test": True},
        )

        assert memory is not None, "Failed to create memory"
        assert memory.id > 0
        assert memory.title == "Test Memory - Template Creation"
        assert memory.external_id == external_id
        assert len(memory.sections) == len(initial_sections)

    async def test_get_active_memories(self, memory_service):
        """Test retrieving memories for an agent"""
        external_id = "test-agent-get-001"

        # First create a test memory
        template = {
            "template_id": "test.simple.v1",
            "name": "Simple Test Template",
            "sections": [
                {
                    "id": "section1",
                    "title": "Section 1",
                    "update_strategy": "replace",
                    "consolidation_trigger": {"update_threshold": 5},
                }
            ],
        }

        created = await memory_service.create_active_memory(
            external_id=external_id,
            title="Test Memory for Get",
            template_content=template,
            initial_sections={
                "section1": {
                    "content": "Test content", 
                    "update_count": 0,
                    "awake_update_count": 0,
                    "last_updated": None
                }
            },
            metadata={},
        )

        assert created is not None

        # Now retrieve memories
        memories = await memory_service.get_active_memories(external_id)
        assert len(memories) >= 1
        assert any(m.id == created.id for m in memories)

    async def test_update_memory_section(self, memory_service):
        """Test updating a section in a memory"""
        external_id = "test-agent-update-001"

        # Create a memory first
        template = {
            "template_id": "test.update.v1",
            "name": "Update Test Template",
            "sections": [
                {
                    "id": "editable_section",
                    "title": "Editable Section",
                    "update_strategy": "append",
                    "consolidation_trigger": {"update_threshold": 5},
                }
            ],
        }

        memory = await memory_service.create_active_memory(
            external_id=external_id,
            title="Test Memory for Update",
            template_content=template,
            initial_sections={
                "editable_section": {
                    "content": "Initial content", 
                    "update_count": 0,
                    "awake_update_count": 0,
                    "last_updated": None
                }
            },
            metadata={},
        )

        assert memory is not None

        # Update the section
        new_content = "Updated content - test"
        updated_memory = await memory_service.update_active_memory_section(
            external_id=external_id,
            memory_id=memory.id,
            section_id="editable_section",
            new_content=new_content,
        )

        assert updated_memory is not None
        assert updated_memory.sections["editable_section"]["content"] == new_content
        assert updated_memory.sections["editable_section"]["update_count"] == 1

    async def test_delete_memory(self, memory_service):
        """Test deleting a memory"""
        external_id = "test-agent-delete-001"

        # Create a memory to delete
        template = {
            "template_id": "test.delete.v1",
            "name": "Delete Test Template",
            "sections": [
                {
                    "id": "section1",
                    "title": "Section 1",
                    "update_strategy": "replace",
                    "consolidation_trigger": {"update_threshold": 5},
                }
            ],
        }

        memory = await memory_service.create_active_memory(
            external_id=external_id,
            title="Test Memory for Delete",
            template_content=template,
            initial_sections={
                "section1": {
                    "content": "Will be deleted", 
                    "update_count": 0,
                    "awake_update_count": 0,
                    "last_updated": None
                }
            },
            metadata={},
        )

        assert memory is not None
        memory_id = memory.id

        # Delete the memory
        success, message = await memory_service.delete_active_memory(
            external_id=external_id,
            memory_id=memory_id,
        )

        assert success, f"Failed to delete memory: {message}"

        # Verify deletion - memory should not be retrievable
        memories = await memory_service.get_active_memories(external_id)
        assert not any(m.id == memory_id for m in memories)

    async def test_get_memory_by_id(self, memory_service):
        """Test getting a specific memory by ID"""
        external_id = "test-agent-getbyid-001"

        # Create a memory
        template = {
            "template_id": "test.getbyid.v1",
            "name": "Get By ID Template",
            "sections": [
                {
                    "id": "section1",
                    "title": "Section 1",
                    "update_strategy": "replace",
                    "consolidation_trigger": {"update_threshold": 5},
                }
            ],
        }

        created = await memory_service.create_active_memory(
            external_id=external_id,
            title="Test Memory Get By ID",
            template_content=template,
            initial_sections={
                "section1": {
                    "content": "Test content", 
                    "update_count": 0,
                    "awake_update_count": 0,
                    "last_updated": None
                }
            },
            metadata={},
        )

        assert created is not None

        # Get memory by ID
        retrieved = await memory_service.get_memory_by_id(external_id, created.id)

        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.title == created.title

    async def test_consolidation_warning(self, memory_service):
        """Test consolidation warning threshold detection"""
        section_data = {
            "content": "Test content",
            "update_count": 4,
        }

        needs_consolidation, warning = memory_service.check_consolidation_needed(
            section_data, threshold=5
        )

        assert not needs_consolidation
        assert warning == ""

        # Test at threshold
        section_data["update_count"] = 5
        needs_consolidation, warning = memory_service.check_consolidation_needed(
            section_data, threshold=5
        )

        assert needs_consolidation
        assert "Warning" in warning

    async def test_format_memory_for_display(self, memory_service):
        """Test formatting memory for UI display"""
        external_id = "test-agent-format-001"

        # Create a memory
        template = {
            "template_id": "test.format.v1",
            "name": "Format Test Template",
            "sections": [
                {
                    "id": "section1",
                    "title": "Section 1",
                    "update_strategy": "replace",
                    "consolidation_trigger": {"update_threshold": 5},
                }
            ],
        }

        memory = await memory_service.create_active_memory(
            external_id=external_id,
            title="Test Memory Format",
            template_content=template,
            initial_sections={
                "section1": {
                    "content": "Test content", 
                    "update_count": 0,
                    "awake_update_count": 0,
                    "last_updated": None
                }
            },
            metadata={"priority": "high"},
        )

        assert memory is not None

        # Format for display
        formatted = memory_service.format_memory_for_display(memory)

        assert formatted["id"] == memory.id
        assert formatted["title"] == memory.title
        assert formatted["section_count"] == 1
        assert "sections" in formatted
        assert "metadata" in formatted


@pytest.mark.asyncio
class TestErrorHandling:
    """Test error handling and edge cases"""

    async def test_create_with_invalid_external_id(self, memory_service):
        """Test creating memory with invalid external ID"""
        template = {
            "template_id": "test.error.v1",
            "name": "Error Test",
            "sections": [],
        }

        # Empty external ID should still work (converted to string)
        memory = await memory_service.create_active_memory(
            external_id="",
            title="Test",
            template_content=template,
            initial_sections={},
            metadata={},
        )

        # Should handle gracefully or return None
        assert memory is None or memory.external_id == ""

    async def test_update_nonexistent_section(self, memory_service):
        """Test updating a section that doesn't exist"""
        external_id = "test-agent-error-002"

        # Try to update a memory that doesn't exist
        result = await memory_service.update_active_memory_section(
            external_id=external_id,
            memory_id=99999,  # Non-existent ID
            section_id="nonexistent",
            new_content="Should fail",
        )

        assert result is None

    async def test_delete_nonexistent_memory(self, memory_service):
        """Test deleting a memory that doesn't exist"""
        success, message = await memory_service.delete_active_memory(
            external_id="nonexistent-agent",
            memory_id=99999,
        )

        assert not success

    async def test_get_memories_for_nonexistent_agent(self, memory_service):
        """Test getting memories for an agent that doesn't exist"""
        memories = await memory_service.get_active_memories("nonexistent-agent-123456")

        assert isinstance(memories, list)
        assert len(memories) == 0


# Run tests with: pytest tests/test_streamlit_integration.py -v -s
# Or: pytest tests/test_streamlit_integration.py::TestMemoryServiceIntegration::test_create_memory_with_template -v


