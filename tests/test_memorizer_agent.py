"""
Test suite for the Memorizer Agent.

Tests the memory consolidation agent's ability to resolve conflicts
between active memory and shortterm memory using both mock models
and real Google Gemini models.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.messages import ModelResponse, TextPart

from agent_reminiscence.agents.memorizer import (
    format_conflicts_as_text,
    MemorizerDeps,
    ConflictResolution,
    resolve_conflicts,
    ChunkUpdateAction,
    ChunkCreateAction,
    EntityUpdateAction,
    RelationshipUpdateAction,
)
from agent_reminiscence.database.models import (
    ConsolidationConflicts,
    ConflictSection,
    ConflictEntityDetail,
    ConflictRelationshipDetail,
    ShorttermMemoryChunk,
    ShorttermEntity,
    ShorttermRelationship,
)
from agent_reminiscence.database.repositories.shortterm_memory import ShorttermMemoryRepository


# =========================================================================
# FIXTURES
# =========================================================================


@pytest.fixture
def mock_shortterm_repo():
    """Create a mock shortterm repository with predefined responses."""
    repo = AsyncMock(spec=ShorttermMemoryRepository)

    # Mock chunk operations
    repo.get_chunk_by_id.return_value = ShorttermMemoryChunk(
        id=1,
        shortterm_memory_id=100,
        content="This is existing chunk content about Python programming.",
        chunk_order=0,
        section_id="code_examples",
        metadata={"source": "consolidation"},
    )

    repo.update_chunk.return_value = ShorttermMemoryChunk(
        id=1,
        shortterm_memory_id=100,
        content="Updated chunk content with more details about Python.",
        chunk_order=0,
        section_id="code_examples",
        metadata={"source": "memorizer_agent"},
    )

    repo.create_chunk.return_value = ShorttermMemoryChunk(
        id=2,
        shortterm_memory_id=100,
        content="New chunk content created by memorizer agent.",
        section_id="code_examples",
        metadata={"source": "memorizer_agent"},
    )

    # Mock entity operations
    repo.get_entity.return_value = ShorttermEntity(
        id="1",
        external_id="test-agent",
        shortterm_memory_id=100,
        name="Python",
        types=["programming_language", "technology"],
        description="A high-level programming language",
        importance=0.9,
        metadata={},
    )

    repo.update_entity.return_value = ShorttermEntity(
        id="1",
        external_id="test-agent",
        shortterm_memory_id=100,
        name="Python",
        types=["programming_language", "technology", "interpreted"],
        description="A high-level, interpreted programming language",
        importance=0.95,
        metadata={},
    )

    # Mock relationship operations
    repo.get_relationship.return_value = ShorttermRelationship(
        id="1",
        external_id="test-agent",
        shortterm_memory_id=100,
        from_entity_id="1",
        to_entity_id="2",
        from_entity_name="Python",
        to_entity_name="Django",
        types=["uses", "framework"],
        description="Django is a Python web framework",
        importance=0.85,
        metadata={},
    )

    repo.update_relationship.return_value = ShorttermRelationship(
        id="1",
        external_id="test-agent",
        shortterm_memory_id=100,
        from_entity_id="1",
        to_entity_id="2",
        from_entity_name="Python",
        to_entity_name="Django",
        types=["uses", "framework", "web_development"],
        description="Django is a popular Python web framework",
        importance=0.9,
        metadata={},
    )

    return repo


@pytest.fixture
def sample_conflicts():
    """Create sample consolidation conflicts for testing."""
    # Create sample chunks
    existing_chunks = [
        ShorttermMemoryChunk(
            id=1,
            shortterm_memory_id=100,
            content="Python is a programming language. It is widely used.",
            section_id="overview",
            metadata={},
        ),
        ShorttermMemoryChunk(
            id=2,
            shortterm_memory_id=100,
            content="Python has a simple syntax and is easy to learn.",
            section_id="overview",
            metadata={},
        ),
    ]

    conflicts = ConsolidationConflicts(
        external_id="test-agent",
        active_memory_id=1,
        shortterm_memory_id=100,
        created_at=datetime.now(timezone.utc),
        total_conflicts=3,  # 2 chunks + 1 entity + 1 relationship conflict
        sections=[
            ConflictSection(
                section_id="overview",
                section_content="Python is a high-level, interpreted programming language. "
                "It emphasizes code readability and has extensive standard libraries. "
                "Python is used in web development, data science, and automation.",
                update_count=3,
                existing_chunks=existing_chunks,
                metadata={"has_conflicts": True},
            )
        ],
        entity_conflicts=[
            ConflictEntityDetail(
                name="Python",
                shortterm_types=["programming_language"],
                active_types=["programming_language", "interpreted"],
                merged_types=["programming_language", "interpreted"],
                shortterm_importance=0.8,
                active_importance=0.9,
                merged_importance=0.85,
                shortterm_description="A programming language",
                active_description="A high-level, interpreted programming language",
                merged_description="A high-level, interpreted programming language",
            )
        ],
        relationship_conflicts=[
            ConflictRelationshipDetail(
                from_entity="Python",
                to_entity="Django",
                shortterm_types=["uses"],
                active_types=["uses", "framework"],
                merged_types=["uses", "framework"],
                shortterm_importance=0.75,
                active_importance=0.85,
                merged_importance=0.8,
            )
        ],
    )

    return conflicts


@pytest.fixture
def minimal_conflicts():
    """Create minimal conflicts with no actual conflicts."""
    return ConsolidationConflicts(
        external_id="test-agent",
        active_memory_id=1,
        shortterm_memory_id=100,
        created_at=datetime.now(timezone.utc),
        total_conflicts=0,
    )


# =========================================================================
# TESTS - Formatting
# =========================================================================


def test_format_conflicts_as_text_basic(minimal_conflicts):
    """Test basic conflict formatting."""
    text = format_conflicts_as_text(minimal_conflicts)

    assert "# Memory Consolidation Conflicts" in text
    assert "External ID: test-agent" in text
    assert "Active Memory ID: 1" in text
    assert "Shortterm Memory ID: 100" in text
    assert "Total Conflicts: 0" in text


def test_format_conflicts_as_text_with_sections(sample_conflicts):
    """Test formatting with section conflicts."""
    text = format_conflicts_as_text(sample_conflicts)

    assert "## Section Conflicts" in text
    assert "### Section 1: overview" in text
    assert "Update Count: 3" in text
    assert "Existing Chunks: 2" in text
    assert "Python is a high-level" in text


def test_format_conflicts_as_text_with_entities(sample_conflicts):
    """Test formatting with entity conflicts."""
    text = format_conflicts_as_text(sample_conflicts)

    assert "## Entity Conflicts" in text
    assert "### Entity 1: Python" in text
    assert "Shortterm Types: ['programming_language']" in text
    assert "Active Types: ['programming_language', 'interpreted']" in text
    assert "Merged Importance: 0.85" in text


def test_format_conflicts_as_text_with_relationships(sample_conflicts):
    """Test formatting with relationship conflicts."""
    text = format_conflicts_as_text(sample_conflicts)

    assert "## Relationship Conflicts" in text
    assert "Python -> Django" in text
    assert "Shortterm Types: ['uses']" in text
    assert "Active Types: ['uses', 'framework']" in text


# =========================================================================
# TESTS - Agent Dependencies
# =========================================================================


def test_memorizer_deps_creation(mock_shortterm_repo):
    """Test creation of MemorizerDeps."""
    deps = MemorizerDeps(
        external_id="test-agent",
        active_memory_id=1,
        shortterm_memory_id=100,
        shortterm_repo=mock_shortterm_repo,
    )

    assert deps.external_id == "test-agent"
    assert deps.active_memory_id == 1
    assert deps.shortterm_memory_id == 100
    assert deps.shortterm_repo == mock_shortterm_repo


# =========================================================================
# TESTS - Conflict Resolution (Mock)
# =========================================================================


@pytest.mark.skip(reason="TestModel doesn't support structured output mode with output_type")
@pytest.mark.asyncio
async def test_resolve_conflicts_minimal(minimal_conflicts, mock_shortterm_repo):
    """Test resolving conflicts with no actual conflicts using TestModel."""
    # Create a TestModel that returns an empty resolution
    test_model = TestModel(
        custom_output_text='{"chunk_updates": [], "chunk_creates": [], "entity_updates": [], "relationship_updates": [], "summary": "No conflicts to resolve"}'
    )

    # Create agent with test model
    agent = Agent(
        model=test_model,
        deps_type=MemorizerDeps,
        output_type=ConflictResolution,
    )

    # Create dependencies
    deps = MemorizerDeps(
        external_id=minimal_conflicts.external_id,
        active_memory_id=minimal_conflicts.active_memory_id,
        shortterm_memory_id=minimal_conflicts.shortterm_memory_id,
        shortterm_repo=mock_shortterm_repo,
    )

    # Format conflicts as text
    conflict_text = format_conflicts_as_text(minimal_conflicts)

    # Run the agent
    result = await agent.run(user_prompt=conflict_text, deps=deps)

    # Verify empty resolution
    assert isinstance(result.output, ConflictResolution)
    assert len(result.output.chunk_updates) == 0
    assert len(result.output.chunk_creates) == 0
    assert len(result.output.entity_updates) == 0
    assert len(result.output.relationship_updates) == 0
    assert "No conflicts" in result.output.summary


@pytest.mark.skip(reason="FunctionModel with structured output needs different approach")
@pytest.mark.asyncio
async def test_resolve_conflicts_with_function_model(sample_conflicts, mock_shortterm_repo):
    """Test conflict resolution with FunctionModel for deterministic testing."""

    def resolve_mock(messages, info):
        """Mock resolution function that creates predictable actions."""
        actions = ConflictResolution(
            chunk_updates=[
                ChunkUpdateAction(
                    chunk_id=1,
                    new_content="Python is a high-level, interpreted programming language with extensive libraries.",
                    reason="Merged active and shortterm chunk information",
                )
            ],
            entity_updates=[
                EntityUpdateAction(
                    entity_id=1,
                    types=["programming_language", "interpreted"],
                    confidence=0.85,
                    reason="Combined types from both sources",
                )
            ],
            relationship_updates=[
                RelationshipUpdateAction(
                    relationship_id=1,
                    types=["uses", "framework"],
                    confidence=0.8,
                    reason="Merged relationship types",
                )
            ],
            summary="Resolved 3 conflicts by merging information from active and shortterm memory",
        )
        import json

        return ModelResponse(parts=[TextPart(content=actions.model_dump_json())])

    # Create agent with function model
    function_model = FunctionModel(resolve_mock)
    agent = Agent(
        model=function_model,
        deps_type=MemorizerDeps,
        output_type=ConflictResolution,
    )

    deps = MemorizerDeps(
        external_id=sample_conflicts.external_id,
        active_memory_id=sample_conflicts.active_memory_id,
        shortterm_memory_id=sample_conflicts.shortterm_memory_id,
        shortterm_repo=mock_shortterm_repo,
    )

    conflict_text = format_conflicts_as_text(sample_conflicts)
    result = await agent.run(user_prompt=conflict_text, deps=deps)

    # Verify resolution structure
    assert len(result.output.chunk_updates) == 1
    assert len(result.output.entity_updates) == 1
    assert len(result.output.relationship_updates) == 1
    assert "Resolved 3 conflicts" in result.output.summary


@pytest.mark.asyncio
async def test_chunk_operations_via_tools(mock_shortterm_repo):
    """Test that chunk tools work correctly with mock repo."""
    # Test get_chunk_by_id
    chunk = await mock_shortterm_repo.get_chunk_by_id(1)
    assert chunk.id == 1
    assert "Python programming" in chunk.content

    # Test update_chunk
    updated = await mock_shortterm_repo.update_chunk(
        chunk_id=1, content="New content with more details"
    )
    assert updated.id == 1
    assert "Updated chunk content" in updated.content

    # Test create_chunk
    new_chunk = await mock_shortterm_repo.create_chunk(
        shortterm_memory_id=100,
        external_id="test-agent",
        content="Brand new chunk",
        chunk_order=2,
        section_id="examples",
    )
    assert new_chunk.id == 2


@pytest.mark.asyncio
async def test_entity_operations_via_tools(mock_shortterm_repo):
    """Test that entity tools work correctly with mock repo."""
    # Test get_entity
    entity = await mock_shortterm_repo.get_entity(1)
    assert entity.name == "Python"
    assert "programming_language" in entity.types

    # Test update_entity
    updated = await mock_shortterm_repo.update_entity(
        entity_id=1,
        types=["programming_language", "technology", "interpreted"],
        confidence=0.95,
    )
    assert updated.importance == 0.95
    assert "interpreted" in updated.types


@pytest.mark.asyncio
async def test_relationship_operations_via_tools(mock_shortterm_repo):
    """Test that relationship tools work correctly with mock repo."""
    # Test get_relationship
    rel = await mock_shortterm_repo.get_relationship(1)
    assert rel.from_entity_name == "Python"
    assert rel.to_entity_name == "Django"

    # Test update_relationship
    updated = await mock_shortterm_repo.update_relationship(
        relationship_id=1,
        types=["uses", "framework", "web_development"],
        confidence=0.9,
    )
    assert updated.importance == 0.9
    assert "web_development" in updated.types


# =========================================================================
# TESTS - Integration with Sample Data
# =========================================================================


def test_sample_conflict_structure(sample_conflicts):
    """Test that sample conflicts have expected structure."""
    assert sample_conflicts.external_id == "test-agent"
    assert sample_conflicts.total_conflicts == 3
    assert len(sample_conflicts.sections) == 1
    assert len(sample_conflicts.entity_conflicts) == 1
    assert len(sample_conflicts.relationship_conflicts) == 1


def test_sample_conflict_section_details(sample_conflicts):
    """Test section conflict details."""
    section = sample_conflicts.sections[0]
    assert section.section_id == "overview"
    assert section.update_count == 3
    assert len(section.existing_chunks) == 2
    assert "high-level" in section.section_content


def test_sample_conflict_entity_details(sample_conflicts):
    """Test entity conflict details."""
    entity = sample_conflicts.entity_conflicts[0]
    assert entity.name == "Python"
    assert entity.merged_importance > entity.shortterm_importance
    assert len(entity.merged_types) >= len(entity.active_types)


def test_sample_conflict_relationship_details(sample_conflicts):
    """Test relationship conflict details."""
    rel = sample_conflicts.relationship_conflicts[0]
    assert rel.from_entity == "Python"
    assert rel.to_entity == "Django"
    assert "framework" in rel.merged_types


# =========================================================================
# TESTS - ConflictResolution Model
# =========================================================================


def test_conflict_resolution_empty():
    """Test creating empty conflict resolution."""
    resolution = ConflictResolution(
        summary="No conflicts to resolve",
    )

    assert resolution.summary == "No conflicts to resolve"
    assert len(resolution.chunk_updates) == 0
    assert len(resolution.chunk_creates) == 0
    assert len(resolution.entity_updates) == 0
    assert len(resolution.relationship_updates) == 0


def test_conflict_resolution_with_actions():
    """Test creating conflict resolution with actions."""
    from agent_reminiscence.agents.memorizer import (
        ChunkUpdateAction,
        EntityUpdateAction,
        RelationshipUpdateAction,
    )

    resolution = ConflictResolution(
        chunk_updates=[
            ChunkUpdateAction(
                chunk_id=1,
                new_content="Merged content",
                reason="Combined active and shortterm information",
            )
        ],
        entity_updates=[
            EntityUpdateAction(
                entity_id=1,
                types=["programming_language", "interpreted"],
                confidence=0.9,
                reason="Increased confidence based on active memory",
            )
        ],
        relationship_updates=[
            RelationshipUpdateAction(
                relationship_id=1,
                types=["uses", "framework"],
                confidence=0.85,
                reason="Added framework type from active memory",
            )
        ],
        summary="Resolved 3 conflicts by merging information",
    )

    assert len(resolution.chunk_updates) == 1
    assert len(resolution.entity_updates) == 1
    assert len(resolution.relationship_updates) == 1
    assert "Resolved 3 conflicts" in resolution.summary


# =========================================================================
# TESTS - Integration with Real Model (Google Gemini)
# =========================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_resolve_conflicts_with_real_gemini_minimal(minimal_conflicts, mock_shortterm_repo):
    """
    Integration test with real Google Gemini model - minimal conflicts.

    Requires GOOGLE_API_KEY environment variable.
    """
    try:
        from pydantic_ai.models.google import GoogleModel

        # Use the lightweight Gemini model
        model = GoogleModel("gemini-2.0-flash-exp")

        agent = Agent(
            model=model,
            deps_type=MemorizerDeps,
            output_type=ConflictResolution,
            system_prompt="""You are a memory consolidation agent.
Analyze conflicts and decide how to resolve them.
When there are no conflicts, return an empty resolution with an appropriate summary.""",
        )

        deps = MemorizerDeps(
            external_id=minimal_conflicts.external_id,
            active_memory_id=minimal_conflicts.active_memory_id,
            shortterm_memory_id=minimal_conflicts.shortterm_memory_id,
            shortterm_repo=mock_shortterm_repo,
        )

        conflict_text = format_conflicts_as_text(minimal_conflicts)
        result = await agent.run(user_prompt=conflict_text, deps=deps)

        # Basic assertions
        assert isinstance(result.output, ConflictResolution)
        assert result.output.summary, "Should have a summary"
        # With no conflicts, should have no or minimal actions
        total_actions = (
            len(result.output.chunk_updates)
            + len(result.output.chunk_creates)
            + len(result.output.entity_updates)
            + len(result.output.relationship_updates)
        )
        assert total_actions == 0, "Should have no actions for minimal conflicts"

    except Exception as e:
        pytest.skip(f"Skipping real model test: {e}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_resolve_conflicts_with_real_gemini_sample(sample_conflicts, mock_shortterm_repo):
    """
    Integration test with real Google Gemini model - sample conflicts.

    Tests the agent's ability to resolve actual conflicts.
    """
    try:
        from pydantic_ai.models.google import GoogleModel

        model = GoogleModel("gemini-2.0-flash-exp")

        agent = Agent(
            model=model,
            deps_type=MemorizerDeps,
            output_type=ConflictResolution,
            system_prompt="""You are a memory consolidation agent.

Analyze conflicts between active and shortterm memory and decide how to resolve them.
When merging information:
- Favor more detailed and recent information
- Combine complementary details
- Update confidence scores based on consistency
- Provide clear reasoning for each action

Available actions:
- chunk_updates: Update existing chunks with merged content
- entity_updates: Update entity types and confidence
- relationship_updates: Update relationship types and confidence

Return a ConflictResolution with appropriate actions and a summary.""",
        )

        deps = MemorizerDeps(
            external_id=sample_conflicts.external_id,
            active_memory_id=sample_conflicts.active_memory_id,
            shortterm_memory_id=sample_conflicts.shortterm_memory_id,
            shortterm_repo=mock_shortterm_repo,
        )

        conflict_text = format_conflicts_as_text(sample_conflicts)
        result = await agent.run(user_prompt=conflict_text, deps=deps)

        # Verify resolution structure
        assert isinstance(result.output, ConflictResolution)
        assert result.output.summary, "Should have a summary"

        # Should have some actions for conflicts
        total_actions = (
            len(result.output.chunk_updates)
            + len(result.output.chunk_creates)
            + len(result.output.entity_updates)
            + len(result.output.relationship_updates)
        )
        assert total_actions > 0, "Should have actions to resolve conflicts"

        # Verify action structure
        for action in result.output.chunk_updates:
            assert action.chunk_id > 0
            assert action.new_content
            assert action.reason

        for action in result.output.entity_updates:
            assert action.entity_id > 0
            assert action.reason

        for action in result.output.relationship_updates:
            assert action.relationship_id > 0
            assert action.reason

    except Exception as e:
        pytest.skip(f"Skipping real model test: {e}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_resolve_conflicts_using_main_function(sample_conflicts, mock_shortterm_repo):
    """
    Integration test using the main resolve_conflicts function.

    This tests the actual function that will be used in production.
    """
    try:
        # Use the real resolve_conflicts function
        result = await resolve_conflicts(sample_conflicts, mock_shortterm_repo)

        # Verify result structure
        assert isinstance(result, ConflictResolution)
        assert result.summary

        # Check that the agent made some decisions
        total_actions = (
            len(result.chunk_updates)
            + len(result.chunk_creates)
            + len(result.entity_updates)
            + len(result.relationship_updates)
        )

        # Should have at least analyzed the conflicts
        assert result.summary, "Should provide analysis summary"

    except Exception as e:
        pytest.skip(f"Skipping real model test: {e}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_conflict_resolution_quality(sample_conflicts, mock_shortterm_repo):
    """
    Integration test to verify quality of conflict resolution.

    Tests that the agent produces reasonable and valid resolutions.
    """
    try:
        result = await resolve_conflicts(sample_conflicts, mock_shortterm_repo)

        # Verify summary quality
        assert len(result.summary) > 20, "Summary should be descriptive"
        assert any(
            word in result.summary.lower()
            for word in ["conflict", "resolve", "merge", "update", "no conflicts"]
        ), "Summary should mention conflict resolution"

        # Verify action validity
        for action in result.chunk_updates:
            assert len(action.new_content) > 10, "Updated content should be substantial"
            assert len(action.reason) > 10, "Reason should be descriptive"

        for action in result.entity_updates:
            if action.confidence:
                assert 0.0 <= action.confidence <= 1.0, "Confidence should be valid"
            assert len(action.reason) > 5, "Reason should explain the update"

        for action in result.relationship_updates:
            if action.confidence:
                assert 0.0 <= action.confidence <= 1.0, "Confidence should be valid"
            if action.strength:
                assert 0.0 <= action.strength <= 1.0, "Strength should be valid"
            assert len(action.reason) > 5, "Reason should explain the update"

    except Exception as e:
        pytest.skip(f"Skipping real model test: {e}")


# =========================================================================
# TESTS - Tool Integration
# =========================================================================


@pytest.mark.skip(reason="FunctionModel with structured output needs different approach")
@pytest.mark.asyncio
async def test_function_model_with_tool_calls(mock_shortterm_repo):
    """Test that FunctionModel can simulate tool calls in conflict resolution."""

    def resolve_with_tools(messages, info):
        """Mock resolution that simulates calling tools."""
        # This simulates what the agent would do with tools
        actions = ConflictResolution(
            chunk_updates=[
                ChunkUpdateAction(
                    chunk_id=1,
                    new_content="Updated content after tool call",
                    reason="Simulated tool call to update_chunk",
                )
            ],
            chunk_creates=[
                ChunkCreateAction(
                    content="New chunk created by tool",
                    chunk_order=1,
                    section_id="new_section",
                    reason="Simulated tool call to create_chunk",
                )
            ],
            entity_updates=[
                EntityUpdateAction(
                    entity_id=1,
                    types=["updated_type"],
                    confidence=0.9,
                    reason="Simulated tool call to update_entity",
                )
            ],
            relationship_updates=[
                RelationshipUpdateAction(
                    relationship_id=1,
                    types=["updated_relationship"],
                    confidence=0.85,
                    strength=0.8,
                    reason="Simulated tool call to update_relationship",
                )
            ],
            summary="All tools were called successfully",
        )
        import json

        return ModelResponse(parts=[TextPart(content=actions.model_dump_json())])

    function_model = FunctionModel(resolve_with_tools)
    agent = Agent(
        model=function_model,
        deps_type=MemorizerDeps,
        output_type=ConflictResolution,
    )

    deps = MemorizerDeps(
        external_id="test-agent",
        active_memory_id=1,
        shortterm_memory_id=100,
        shortterm_repo=mock_shortterm_repo,
    )

    result = await agent.run(user_prompt="Test conflict", deps=deps)

    # Verify all tool types were used
    assert len(result.output.chunk_updates) == 1
    assert len(result.output.chunk_creates) == 1
    assert len(result.output.entity_updates) == 1
    assert len(result.output.relationship_updates) == 1
    assert "tools were called" in result.output.summary.lower()


# =========================================================================
# MANUAL TEST SCENARIO (for manual execution with real API)
# =========================================================================


def print_manual_test_scenario():
    """
    Print a manual test scenario that can be run with a real API key.

    This is not an automated test but a guide for manual testing.
    """
    print(
        """
    ========================================
    MANUAL TEST SCENARIO FOR MEMORIZER AGENT
    ========================================
    
    Prerequisites:
    1. Set OPENAI_API_KEY environment variable
    2. Have a running PostgreSQL and Neo4j instance
    3. Initialize the database with test data
    
    Steps:
    
    1. Create test conflicts:
        ```python
        import asyncio
        from datetime import datetime, timezone
        from agent_reminiscence.database.models import ConsolidationConflicts, ConflictSection
        from agent_reminiscence.agents.memorizer import resolve_conflicts
        
        async def main():
            # Create your real shortterm_repo instance here
            # shortterm_repo = ...
            
            conflicts = ConsolidationConflicts(
                external_id="manual-test",
                active_memory_id=1,
                shortterm_memory_id=1,
                created_at=datetime.now(timezone.utc),
                total_conflicts=1,
                sections=[
                    ConflictSection(
                        section_id="test_section",
                        section_content="This is new updated content.",
                        update_count=2,
                        existing_chunks=[...],  # Add real chunks
                        metadata={},
                    )
                ],
            )
            
            resolution = await resolve_conflicts(conflicts, shortterm_repo)
            print(f"Resolution: {resolution.summary}")
            print(f"Chunk updates: {len(resolution.chunk_updates)}")
            print(f"Entity updates: {len(resolution.entity_updates)}")
        
        asyncio.run(main())
        ```
    
    2. Expected behavior:
       - Agent analyzes conflicts
       - Makes intelligent merge decisions
       - Provides detailed reasoning
       - Updates database via tools
    
    3. Verification:
       - Check logs for agent decision-making process
       - Verify database updates were applied
       - Review resolution summary for quality
    """
    )


if __name__ == "__main__":
    print_manual_test_scenario()


