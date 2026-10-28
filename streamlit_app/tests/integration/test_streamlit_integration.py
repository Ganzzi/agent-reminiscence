"""
Integration Tests for Streamlit UI Memory Service

These tests validate the complete workflow from UI through to database.
Run with: pytest tests/test_streamlit_integration.py -v
"""

import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit")

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from conftest import (
    REQUIRED_ENV,
    external_id,
    resolve_test_db_config,
    to_template_content,
)
from streamlit_app.utils.yaml_validator import YAMLValidator


@pytest.mark.asyncio
class TestMemoryServiceIntegration:
    """Integration tests for MemoryService"""

    async def test_connection(self, memory_service):
        """Test database connection"""
        is_connected, message = await memory_service.test_connection()
        assert is_connected, f"Failed to connect: {message}"

    async def test_create_memory_with_template(self, memory_service, template_service):
        """Test creating a memory using a pre-built template"""
        # Load a template through the real grouped service contract
        templates = template_service.get_all_templates(use_cache=False)
        assert len(templates) > 0, "No templates found"

        agent_group = sorted(templates)[0]  # deterministic group selection
        template = to_template_content(templates[agent_group][0])
        assert len(template["sections"]) >= 2
        # Storage materializes every template section (initial_sections only
        # overrides content), so the original first-2-sections intent must be
        # encoded in the template shape itself for the count assertion below.
        template["sections"] = template["sections"][:2]

        external_id_val = external_id("create-001")

        # Prepare initial sections
        initial_sections = {}
        for section in template["sections"][:2]:  # Use first 2 sections
            initial_sections[section["id"]] = {
                "content": f"Test content for {section['id']}",
                "update_count": 0,
                "awake_update_count": 0,
                "last_updated": None,
            }

        # Create memory
        memory = await memory_service.create_active_memory(
            external_id=external_id_val,
            title="Test Memory - Template Creation",
            template_content=template,
            initial_sections=initial_sections,
            metadata={"priority": "high", "test": True},
        )

        assert memory is not None, "Failed to create memory"
        assert memory.id > 0
        assert memory.title == "Test Memory - Template Creation"
        assert memory.external_id == external_id_val
        assert len(memory.sections) == len(initial_sections)

    async def test_get_active_memories(self, memory_service):
        """Test retrieving memories for an agent"""
        external_id_val = external_id("get-001")

        # First create a test memory
        template = to_template_content(
            {
                "template": {
                    "id": "bmad.test.simple.v1",
                    "name": "Simple Test Template",
                    "sections": [
                        {
                            "id": "section1",
                            "title": "Section 1",
                            "description": "Deterministic single section for retrieval",
                            "update_strategy": "replace",
                            "consolidation_trigger": {"update_threshold": 5},
                        }
                    ],
                }
            }
        )

        created = await memory_service.create_active_memory(
            external_id=external_id_val,
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
        memories = await memory_service.get_active_memories(external_id_val)
        assert len(memories) >= 1
        assert any(m.id == created.id for m in memories)

    async def test_update_memory_section(self, memory_service):
        """Test updating a section in a memory"""
        external_id_val = external_id("update-001")

        # Create a memory first
        template = to_template_content(
            {
                "template": {
                    "id": "bmad.test.update.v1",
                    "name": "Update Test Template",
                    "sections": [
                        {
                            "id": "editable_section",
                            "title": "Editable Section",
                            "description": "Deterministic single section for update",
                            "update_strategy": "append",
                            "consolidation_trigger": {"update_threshold": 5},
                        }
                    ],
                }
            }
        )

        memory = await memory_service.create_active_memory(
            external_id=external_id_val,
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
            external_id=external_id_val,
            memory_id=memory.id,
            section_id="editable_section",
            new_content=new_content,
        )

        assert updated_memory is not None
        assert updated_memory.sections["editable_section"]["content"] == new_content
        assert updated_memory.sections["editable_section"]["update_count"] == 1

    async def test_delete_memory(self, memory_service):
        """Test deleting a memory"""
        external_id_val = external_id("delete-001")

        # Create a memory to delete
        template = to_template_content(
            {
                "template": {
                    "id": "bmad.test.delete.v1",
                    "name": "Delete Test Template",
                    "sections": [
                        {
                            "id": "section1",
                            "title": "Section 1",
                            "description": "Deterministic single section for deletion",
                            "update_strategy": "replace",
                            "consolidation_trigger": {"update_threshold": 5},
                        }
                    ],
                }
            }
        )

        memory = await memory_service.create_active_memory(
            external_id=external_id_val,
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
            external_id=external_id_val,
            memory_id=memory_id,
        )

        assert success, f"Failed to delete memory: {message}"

        # Verify deletion - memory should not be retrievable
        memories = await memory_service.get_active_memories(external_id_val)
        assert not any(m.id == memory_id for m in memories)

    async def test_get_memory_by_id(self, memory_service):
        """Test getting a specific memory by ID"""
        external_id_val = external_id("getbyid-001")

        # Create a memory
        template = to_template_content(
            {
                "template": {
                    "id": "bmad.test.getbyid.v1",
                    "name": "Get By ID Template",
                    "sections": [
                        {
                            "id": "section1",
                            "title": "Section 1",
                            "description": "Deterministic single section for get-by-id",
                            "update_strategy": "replace",
                            "consolidation_trigger": {"update_threshold": 5},
                        }
                    ],
                }
            }
        )

        created = await memory_service.create_active_memory(
            external_id=external_id_val,
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
        retrieved = await memory_service.get_memory_by_id(external_id_val, created.id)

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
        external_id_val = external_id("format-001")

        # Create a memory
        template = to_template_content(
            {
                "template": {
                    "id": "bmad.test.format.v1",
                    "name": "Format Test Template",
                    "sections": [
                        {
                            "id": "section1",
                            "title": "Section 1",
                            "description": "Deterministic single section for formatting",
                            "update_strategy": "replace",
                            "consolidation_trigger": {"update_threshold": 5},
                        }
                    ],
                }
            }
        )

        memory = await memory_service.create_active_memory(
            external_id=external_id_val,
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
        template = to_template_content(
            {
                "template": {
                    "id": "bmad.test.error.v1",
                    "name": "Error Test",
                    "sections": [],
                }
            }
        )

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
        external_id_val = external_id("error-002")

        # Try to update a memory that doesn't exist
        result = await memory_service.update_active_memory_section(
            external_id=external_id_val,
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


class TestIsolatedConfigSafety:
    """R1/R2: the isolated-topology config contract fails before any network contact."""

    def test_missing_isolated_config_fails_before_any_contact(self, monkeypatch):
        for var in REQUIRED_ENV.values():
            monkeypatch.delenv(var, raising=False)
        with pytest.raises(RuntimeError, match="AGENT_MEM_STREAMLIT_TEST_"):
            resolve_test_db_config()  # no I/O has occurred

    def test_invalid_isolated_config_fails_before_any_contact(self, monkeypatch):
        # Non-integer port
        _set_valid_env(monkeypatch)
        monkeypatch.setenv("AGENT_MEM_STREAMLIT_TEST_PG_PORT", "not-a-port")
        with pytest.raises(RuntimeError, match="integer"):
            resolve_test_db_config()

        # Non-bolt:// URI
        _set_valid_env(monkeypatch)
        monkeypatch.setenv("AGENT_MEM_STREAMLIT_TEST_NEO4J_URI", "http://127.0.0.1:17687")
        with pytest.raises(RuntimeError, match="bolt://"):
            resolve_test_db_config()

        # Password shorter than the Config validator contract
        _set_valid_env(monkeypatch)
        monkeypatch.setenv("AGENT_MEM_STREAMLIT_TEST_PG_PASSWORD", "short")
        with pytest.raises(RuntimeError, match="at least 8 characters"):
            resolve_test_db_config()


class TestTemplateFixtureContract:
    """R3: the committed fixture tree loads through the real TemplateService."""

    def test_fixture_loads_grouped_through_real_service(self, template_service):
        templates = template_service.get_all_templates(use_cache=False)
        assert set(templates) == {"bmad-test"}
        assert len(templates["bmad-test"]) == 2
        for loaded in templates["bmad-test"]:
            assert "template" in loaded
            assert loaded["_agent_type"] == "bmad-test"
            assert loaded["_file_path"]
            assert YAMLValidator.validate_template_structure(loaded) == (True, [])
        ids = {t["template"]["id"] for t in templates["bmad-test"]}
        assert ids == {"bmad.test.alpha.v1", "bmad.test.beta.v1"}


def _set_valid_env(monkeypatch):
    """Seed the nine required isolated-topology variables with valid values (no I/O)."""
    monkeypatch.setenv(REQUIRED_ENV["run_label"], "config-safety-check")
    monkeypatch.setenv(REQUIRED_ENV["pg_host"], "127.0.0.1")
    monkeypatch.setenv(REQUIRED_ENV["pg_port"], "15432")
    monkeypatch.setenv(REQUIRED_ENV["pg_db"], "config_safety_db")
    monkeypatch.setenv(REQUIRED_ENV["pg_user"], "config_safety_user")
    monkeypatch.setenv(REQUIRED_ENV["pg_password"], "config-safety-pg-pass")
    monkeypatch.setenv(REQUIRED_ENV["neo4j_uri"], "bolt://127.0.0.1:17687")
    monkeypatch.setenv(REQUIRED_ENV["neo4j_user"], "config_safety_neo")
    monkeypatch.setenv(REQUIRED_ENV["neo4j_password"], "config-safety-neo-pass")


# Run tests with: pytest tests/test_streamlit_integration.py -v -s
# Or: pytest tests/test_streamlit_integration.py::TestMemoryServiceIntegration::test_create_memory_with_template -v
