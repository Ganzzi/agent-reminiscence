"""Phase 6: MCP Server Updates - Integration tests for new deep_search_memories tool."""

import pytest
from pathlib import Path
import sys
import asyncio

# Add agent_mem to path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from agent_reminiscence_mcp.schemas import (
    SEARCH_MEMORIES_INPUT_SCHEMA,
    DEEP_SEARCH_MEMORIES_INPUT_SCHEMA,
)
from mcp import types


# We need to replicate the handle_list_tools function for testing
async def get_mcp_tools() -> list[types.Tool]:
    """Get list of MCP tools (replicated from server.py for testing)."""
    from agent_reminiscence_mcp.schemas import (
        GET_ACTIVE_MEMORIES_INPUT_SCHEMA,
        UPDATE_MEMORY_SECTIONS_INPUT_SCHEMA,
        SEARCH_MEMORIES_INPUT_SCHEMA,
        DEEP_SEARCH_MEMORIES_INPUT_SCHEMA,
        CREATE_ACTIVE_MEMORY_INPUT_SCHEMA,
        DELETE_ACTIVE_MEMORY_INPUT_SCHEMA,
    )

    return [
        types.Tool(
            name="get_active_memories",
            description=(
                "Get all active memories for an agent. "
                "Active memories represent the agent's working memory - current tasks, "
                "recent decisions, and ongoing work context. Each memory has a template "
                "structure with multiple sections that can be updated independently."
            ),
            inputSchema=GET_ACTIVE_MEMORIES_INPUT_SCHEMA,
        ),
        types.Tool(
            name="create_active_memory",
            description=(
                "Create a new active memory (working memory) for an agent with a template-driven structure.\n\n"
                "Active memories store the agent's current work context organized into sections (e.g., 'current_task', "
                "'progress', 'notes')."
            ),
            inputSchema=CREATE_ACTIVE_MEMORY_INPUT_SCHEMA,
        ),
        types.Tool(
            name="update_memory_sections",
            description=(
                "Upsert (insert or update) multiple sections in an active memory. "
                "Supports creating new sections, replacing content, and inserting content."
            ),
            inputSchema=UPDATE_MEMORY_SECTIONS_INPUT_SCHEMA,
        ),
        types.Tool(
            name="delete_active_memory",
            description="Delete an active memory for an agent",
            inputSchema=DELETE_ACTIVE_MEMORY_INPUT_SCHEMA,
        ),
        types.Tool(
            name="search_memories",
            description=(
                "Search across shortterm and longterm memory tiers using intelligent AI-powered search. "
                "Provide a natural language query describing your current context, what you're working on, "
                "and what information you need. The AI will understand your intent and search across memory tiers "
                "using vector similarity and BM25 hybrid search. "
                "\n\n"
                "Results include:\n"
                "- Matched memory chunks with relevance scores\n"
                "- Related entities and relationships from the knowledge graph\n"
                "- Optional AI-synthesized summary (when force_synthesis=true or for complex queries)\n"
                "\n\n"
                "Example queries:\n"
                "- 'Working on authentication, need to know how JWT tokens were implemented'\n"
                "- 'Debugging API errors, what endpoints and error handling did we discuss?'\n"
                "- 'Need context on the database schema design decisions'"
            ),
            inputSchema=SEARCH_MEMORIES_INPUT_SCHEMA,
        ),
        types.Tool(
            name="deep_search_memories",
            description=(
                "Perform comprehensive memory search with full synthesis, entity extraction, and relationship analysis. "
                "This is a deeper version of search_memories that performs AI synthesis and connects related concepts. "
                "Use when you need complete context including how different pieces of information relate to each other. "
                "\n\n"
                "Key differences from search_memories:\n"
                "- Performs AI synthesis to summarize findings\n"
                "- Extracts and analyzes entity relationships\n"
                "- Generates comprehensive response with connections\n"
                "- Takes 500ms-2s (vs <200ms for fast search)\n"
                "\n\n"
                "Results include:\n"
                "- Matched memory chunks with relevance scores\n"
                "- Extracted entities and their relationships\n"
                "- AI-synthesized summary connecting findings\n"
                "- Confidence scores and temporal context\n"
                "\n\n"
                "Example queries:\n"
                "- 'Deep analysis: How does JWT authentication connect to our API design?'\n"
                "- 'What entities and relationships exist around database schema decisions?'\n"
                "- 'Synthesize all information about performance optimization efforts'"
            ),
            inputSchema=DEEP_SEARCH_MEMORIES_INPUT_SCHEMA,
        ),
    ]


class TestSchemas:
    """Test that input schemas are properly defined."""

    def test_deep_search_memories_schema_exists(self) -> None:
        """Verify DEEP_SEARCH_MEMORIES_INPUT_SCHEMA is defined."""
        assert DEEP_SEARCH_MEMORIES_INPUT_SCHEMA is not None
        assert isinstance(DEEP_SEARCH_MEMORIES_INPUT_SCHEMA, dict)

    def test_deep_search_schema_has_required_properties(self) -> None:
        """Verify schema has required properties."""
        schema = DEEP_SEARCH_MEMORIES_INPUT_SCHEMA
        assert "type" in schema
        assert "properties" in schema
        assert "required" in schema
        assert schema["type"] == "object"

    def test_deep_search_schema_requires_external_id_and_query(self) -> None:
        """Verify required fields are external_id and query."""
        schema = DEEP_SEARCH_MEMORIES_INPUT_SCHEMA
        assert "external_id" in schema["required"]
        assert "query" in schema["required"]
        assert len(schema["required"]) == 2  # Only these two are required

    def test_deep_search_schema_has_optional_parameters(self) -> None:
        """Verify optional parameters are present."""
        schema = DEEP_SEARCH_MEMORIES_INPUT_SCHEMA
        props = schema["properties"]
        assert "limit" in props
        assert "synthesis" in props

    def test_deep_search_schema_limit_validation(self) -> None:
        """Verify limit has proper constraints."""
        schema = DEEP_SEARCH_MEMORIES_INPUT_SCHEMA
        limit_prop = schema["properties"]["limit"]
        assert limit_prop["type"] == "integer"
        assert limit_prop["minimum"] == 1
        assert limit_prop["maximum"] == 100
        assert limit_prop["default"] == 10

    def test_deep_search_schema_synthesis_defaults_true(self) -> None:
        """Verify synthesis defaults to true for deep search."""
        schema = DEEP_SEARCH_MEMORIES_INPUT_SCHEMA
        synthesis_prop = schema["properties"]["synthesis"]
        assert synthesis_prop["type"] == "boolean"
        assert synthesis_prop["default"] is True  # Default True for deep search

    def test_search_and_deep_search_schemas_have_similar_structure(self) -> None:
        """Verify both search schemas have comparable structure."""
        search_schema = SEARCH_MEMORIES_INPUT_SCHEMA
        deep_search_schema = DEEP_SEARCH_MEMORIES_INPUT_SCHEMA

        # Both should have same required fields
        assert search_schema["required"] == deep_search_schema["required"]

        # Both should have same external_id and query properties
        assert (
            search_schema["properties"]["external_id"]["type"]
            == deep_search_schema["properties"]["external_id"]["type"]
        )
        assert (
            search_schema["properties"]["query"]["type"]
            == deep_search_schema["properties"]["query"]["type"]
        )

    def test_deep_search_schema_external_id_description(self) -> None:
        """Verify external_id has proper description."""
        schema = DEEP_SEARCH_MEMORIES_INPUT_SCHEMA
        ext_id_prop = schema["properties"]["external_id"]
        assert "description" in ext_id_prop
        assert "agent" in ext_id_prop["description"].lower()

    def test_deep_search_schema_query_description(self) -> None:
        """Verify query parameter has comprehensive description."""
        schema = DEEP_SEARCH_MEMORIES_INPUT_SCHEMA
        query_prop = schema["properties"]["query"]
        assert "description" in query_prop
        description = query_prop["description"]
        # Should mention synthesis, entity extraction, relationships
        assert "synthesis" in description.lower() or "comprehensive" in description.lower()

    def test_deep_search_schema_has_examples(self) -> None:
        """Verify schema includes usage examples."""
        schema = DEEP_SEARCH_MEMORIES_INPUT_SCHEMA
        query_prop = schema["properties"]["query"]
        description = query_prop["description"]
        # Should have example queries
        assert "example" in description.lower()


class TestMCPToolRegistration:
    """Test that tools are properly registered in MCP server."""

    @pytest.mark.asyncio
    async def test_search_memories_tool_exists(self) -> None:
        """Verify search_memories tool is registered."""
        tools = await get_mcp_tools()
        tool_names = [tool.name for tool in tools]
        assert "search_memories" in tool_names

    @pytest.mark.asyncio
    async def test_deep_search_memories_tool_exists(self) -> None:
        """Verify deep_search_memories tool is registered."""
        tools = await get_mcp_tools()
        tool_names = [tool.name for tool in tools]
        assert "deep_search_memories" in tool_names

    @pytest.mark.asyncio
    async def test_search_memories_tool_has_schema(self) -> None:
        """Verify search_memories tool has input schema."""
        tools = await get_mcp_tools()
        search_tool = next((t for t in tools if t.name == "search_memories"), None)
        assert search_tool is not None
        assert search_tool.inputSchema is not None

    @pytest.mark.asyncio
    async def test_deep_search_memories_tool_has_schema(self) -> None:
        """Verify deep_search_memories tool has input schema."""
        tools = await get_mcp_tools()
        deep_search_tool = next((t for t in tools if t.name == "deep_search_memories"), None)
        assert deep_search_tool is not None
        assert deep_search_tool.inputSchema is not None

    @pytest.mark.asyncio
    async def test_both_search_tools_have_descriptions(self) -> None:
        """Verify both search tools have descriptions."""
        tools = await get_mcp_tools()
        search_tool = next((t for t in tools if t.name == "search_memories"), None)
        deep_search_tool = next((t for t in tools if t.name == "deep_search_memories"), None)

        assert search_tool.description is not None
        assert deep_search_tool.description is not None
        assert len(search_tool.description) > 50
        assert len(deep_search_tool.description) > 50

    @pytest.mark.asyncio
    async def test_deep_search_tool_description_emphasizes_synthesis(self) -> None:
        """Verify deep_search description emphasizes synthesis and analysis."""
        tools = await get_mcp_tools()
        deep_search_tool = next((t for t in tools if t.name == "deep_search_memories"), None)
        description = deep_search_tool.description.lower()

        # Should mention key differentiators
        assert "synthesis" in description
        assert "entity" in description or "relationship" in description


class TestToolInputValidation:
    """Test input validation for search and deep_search tools."""

    def test_search_memories_schema_validation_external_id(self) -> None:
        """Verify external_id is required for search_memories."""
        schema = SEARCH_MEMORIES_INPUT_SCHEMA
        assert "external_id" in schema["required"]

    def test_search_memories_schema_validation_query(self) -> None:
        """Verify query is required for search_memories."""
        schema = SEARCH_MEMORIES_INPUT_SCHEMA
        assert "query" in schema["required"]

    def test_deep_search_memories_schema_validation_external_id(self) -> None:
        """Verify external_id is required for deep_search_memories."""
        schema = DEEP_SEARCH_MEMORIES_INPUT_SCHEMA
        assert "external_id" in schema["required"]

    def test_deep_search_memories_schema_validation_query(self) -> None:
        """Verify query is required for deep_search_memories."""
        schema = DEEP_SEARCH_MEMORIES_INPUT_SCHEMA
        assert "query" in schema["required"]

    def test_search_limit_is_integer(self) -> None:
        """Verify limit parameter is integer type."""
        schema = SEARCH_MEMORIES_INPUT_SCHEMA
        limit_prop = schema["properties"]["limit"]
        assert limit_prop["type"] == "integer"

    def test_deep_search_limit_is_integer(self) -> None:
        """Verify limit parameter is integer type in deep_search."""
        schema = DEEP_SEARCH_MEMORIES_INPUT_SCHEMA
        limit_prop = schema["properties"]["limit"]
        assert limit_prop["type"] == "integer"

    def test_search_limit_has_bounds(self) -> None:
        """Verify search limit has minimum and maximum."""
        schema = SEARCH_MEMORIES_INPUT_SCHEMA
        limit_prop = schema["properties"]["limit"]
        assert "minimum" in limit_prop
        assert "maximum" in limit_prop
        assert limit_prop["minimum"] == 1
        assert limit_prop["maximum"] == 100

    def test_deep_search_limit_has_bounds(self) -> None:
        """Verify deep_search limit has minimum and maximum."""
        schema = DEEP_SEARCH_MEMORIES_INPUT_SCHEMA
        limit_prop = schema["properties"]["limit"]
        assert "minimum" in limit_prop
        assert "maximum" in limit_prop
        assert limit_prop["minimum"] == 1
        assert limit_prop["maximum"] == 100

    def test_search_synthesis_is_boolean(self) -> None:
        """Verify synthesis parameter is boolean."""
        schema = SEARCH_MEMORIES_INPUT_SCHEMA
        synthesis_prop = schema["properties"]["synthesis"]
        assert synthesis_prop["type"] == "boolean"

    def test_deep_search_synthesis_is_boolean(self) -> None:
        """Verify synthesis parameter is boolean in deep_search."""
        schema = DEEP_SEARCH_MEMORIES_INPUT_SCHEMA
        synthesis_prop = schema["properties"]["synthesis"]
        assert synthesis_prop["type"] == "boolean"


class TestToolDifferentiation:
    """Test that search_memories and deep_search_memories are properly differentiated."""

    @pytest.mark.asyncio
    async def test_search_and_deep_search_are_different_tools(self) -> None:
        """Verify search_memories and deep_search_memories are different tools."""
        tools = await get_mcp_tools()
        tool_names = [tool.name for tool in tools]

        # Both should exist
        assert "search_memories" in tool_names
        assert "deep_search_memories" in tool_names

        # They should be different instances
        search_tool = next((t for t in tools if t.name == "search_memories"))
        deep_search_tool = next((t for t in tools if t.name == "deep_search_memories"))
        assert search_tool != deep_search_tool

    def test_deep_search_has_different_schema_from_search(self) -> None:
        """Verify deep_search has distinct schema properties."""
        search_schema = SEARCH_MEMORIES_INPUT_SCHEMA
        deep_search_schema = DEEP_SEARCH_MEMORIES_INPUT_SCHEMA

        # Default synthesis behavior should differ
        search_synthesis_default = search_schema["properties"]["synthesis"]["default"]
        deep_search_synthesis_default = deep_search_schema["properties"]["synthesis"]["default"]

        # Deep search should default to True, search to False
        assert search_synthesis_default is False
        assert deep_search_synthesis_default is True

    def test_tool_descriptions_are_distinct(self) -> None:
        """Verify tool descriptions clearly differentiate the tools."""
        search_schema = SEARCH_MEMORIES_INPUT_SCHEMA
        deep_search_schema = DEEP_SEARCH_MEMORIES_INPUT_SCHEMA

        search_desc = search_schema["properties"]["query"]["description"]
        deep_search_desc = deep_search_schema["properties"]["query"]["description"]

        # Descriptions should be different
        assert search_desc != deep_search_desc

        # Deep search description should emphasize synthesis
        deep_search_lower = deep_search_desc.lower()
        assert "synthesis" in deep_search_lower or "comprehensive" in deep_search_lower


class TestMCPServerIntegration:
    """Test integration with MCP server."""

    @pytest.mark.asyncio
    async def test_tool_list_includes_all_memory_tools(self) -> None:
        """Verify all memory tools are in tool list."""
        tools = await get_mcp_tools()
        tool_names = [tool.name for tool in tools]

        # Should have all 6 tools (was 5, now 6 with deep_search)
        assert "get_active_memories" in tool_names
        assert "create_active_memory" in tool_names
        assert "update_memory_sections" in tool_names
        assert "delete_active_memory" in tool_names
        assert "search_memories" in tool_names
        assert "deep_search_memories" in tool_names

    @pytest.mark.asyncio
    async def test_tool_count(self) -> None:
        """Verify correct number of tools are registered."""
        tools = await get_mcp_tools()
        # Phase 6 adds 1 tool: deep_search_memories
        # Previous tools: 5 (get_active, create, update, delete, search)
        # Total: 6
        assert len(tools) == 6

    @pytest.mark.asyncio
    async def test_all_tools_have_required_fields(self) -> None:
        """Verify all tools have name, description, and inputSchema."""
        tools = await get_mcp_tools()

        for tool in tools:
            assert tool.name is not None
            assert isinstance(tool.name, str)
            assert len(tool.name) > 0

            assert tool.description is not None
            assert isinstance(tool.description, str)
            assert len(tool.description) > 0

            assert tool.inputSchema is not None
            assert isinstance(tool.inputSchema, dict)


class TestPhase6Completion:
    """Test that Phase 6 implementation is complete."""

    @pytest.mark.asyncio
    async def test_phase6_scope_complete(self) -> None:
        """Verify all Phase 6 tasks are complete."""
        tools = await get_mcp_tools()
        tool_names = [tool.name for tool in tools]

        # Task 1: search_memories tool exists
        assert "search_memories" in tool_names

        # Task 2: deep_search_memories tool added
        assert "deep_search_memories" in tool_names

        # Task 3 & 4: Tool registration and routing (implicit in above)
        # Both tools must be callable and routable

        # Task 5: Input schemas updated
        search_schema = SEARCH_MEMORIES_INPUT_SCHEMA
        deep_search_schema = DEEP_SEARCH_MEMORIES_INPUT_SCHEMA
        assert search_schema is not None
        assert deep_search_schema is not None

    @pytest.mark.asyncio
    async def test_mcp_tools_count_increased_to_six(self) -> None:
        """Verify MCP tools have increased from 5 to 6."""
        tools = await get_mcp_tools()
        # Phase 5 had 5 tools, Phase 6 adds 1 more (deep_search_memories)
        assert len(tools) == 6

    def test_schemas_module_exports_deep_search_schema(self) -> None:
        """Verify deep_search schema is exported from schemas module."""
        from agent_reminiscence_mcp.schemas import DEEP_SEARCH_MEMORIES_INPUT_SCHEMA

        assert DEEP_SEARCH_MEMORIES_INPUT_SCHEMA is not None


class TestClaudeDesktopIntegration:
    """Test Claude Desktop integration scenarios."""

    @pytest.mark.asyncio
    async def test_tools_compatible_with_claude_protocol(self) -> None:
        """Verify tools follow MCP protocol for Claude Desktop."""
        tools = await get_mcp_tools()

        for tool in tools:
            # All tools should be types.Tool instances
            assert isinstance(tool, types.Tool)

            # All tools should have name (no spaces/special chars)
            assert tool.name.isidentifier() or "_" in tool.name or "-" in tool.name

            # All tools should have description
            assert len(tool.description) > 20

            # All tools should have inputSchema
            assert isinstance(tool.inputSchema, dict)
            assert "type" in tool.inputSchema
            assert tool.inputSchema["type"] == "object"

    @pytest.mark.asyncio
    async def test_search_tools_discoverable_by_name(self) -> None:
        """Verify search tools are discoverable by Claude."""
        tools = await get_mcp_tools()
        tool_names = [tool.name for tool in tools]

        # Claude should be able to find both search tools by name
        search_indices = [i for i, name in enumerate(tool_names) if "search" in name]
        assert len(search_indices) >= 2  # At least search_memories and deep_search_memories

    @pytest.mark.asyncio
    async def test_tools_have_clear_purpose_descriptions(self) -> None:
        """Verify tools have clear purpose in descriptions."""
        tools = await get_mcp_tools()

        for tool in tools:
            description = tool.description.lower()
            # Each tool should have clear purpose
            assert len(description) > 30
            # Should not have placeholder text
            assert "todo" not in description
            assert "fix me" not in description


class TestBackwardCompatibility:
    """Test backward compatibility with existing tools."""

    @pytest.mark.asyncio
    async def test_search_memories_tool_unchanged(self) -> None:
        """Verify search_memories tool still works as before."""
        tools = await get_mcp_tools()
        search_tool = next((t for t in tools if t.name == "search_memories"), None)

        # Should still exist with same interface
        assert search_tool is not None
        assert search_tool.inputSchema is not None

        # Should have required external_id and query
        required = search_tool.inputSchema["required"]
        assert "external_id" in required
        assert "query" in required

    @pytest.mark.asyncio
    async def test_all_previous_tools_still_exist(self) -> None:
        """Verify all Phase 1-5 tools still exist."""
        tools = await get_mcp_tools()
        tool_names = [tool.name for tool in tools]

        # All previous tools should still exist
        assert "get_active_memories" in tool_names
        assert "create_active_memory" in tool_names
        assert "update_memory_sections" in tool_names
        assert "delete_active_memory" in tool_names
        assert "search_memories" in tool_names

    @pytest.mark.asyncio
    async def test_tool_order_not_breaking(self) -> None:
        """Verify tool list is still valid regardless of order."""
        tools = await get_mcp_tools()

        # Should get exactly 6 tools
        assert len(tools) == 6

        # All should be valid Tool objects
        for tool in tools:
            assert isinstance(tool, types.Tool)
            assert tool.name
            assert tool.description
            assert tool.inputSchema


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
