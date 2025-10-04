"""MCP Server for AgentMem - Exposes memory management tools using low-level Server API."""

import sys
from pathlib import Path

# Add agent_mem to Python path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import mcp.types as types
from mcp.server.lowlevel import Server

from agent_mem import AgentMem
from .schemas import (
    GET_ACTIVE_MEMORIES_INPUT_SCHEMA,
    UPDATE_MEMORY_SECTION_INPUT_SCHEMA,
    SEARCH_MEMORIES_INPUT_SCHEMA,
)


@asynccontextmanager
async def server_lifespan(_server: Server) -> AsyncIterator[dict[str, Any]]:
    """
    Manage server lifecycle - initialize AgentMem on startup.

    This creates a singleton AgentMem instance that's shared across
    all tool calls, maintaining database connections efficiently.
    """
    # Startup: Initialize AgentMem with configuration
    from agent_mem.config import get_config

    config = get_config()

    agent_mem = AgentMem(config=config)
    await agent_mem.initialize()

    try:
        yield {"agent_mem": agent_mem}
    finally:
        # Shutdown: Close AgentMem connections
        await agent_mem.close()


# Create MCP server with lifespan management
server = Server("agent-mem", lifespan=server_lifespan)


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available memory management tools."""
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
            name="update_memory_section",
            description=(
                "Update a specific section in an active memory. "
                "This updates the content of a single section within an active memory. "
                "The section's update_count is automatically incremented, and when it "
                "reaches a threshold, the memory is automatically consolidated to shortterm memory."
            ),
            inputSchema=UPDATE_MEMORY_SECTION_INPUT_SCHEMA,
        ),
        types.Tool(
            name="search_memories",
            description=(
                "Search across shortterm and longterm memory tiers. "
                "This performs an intelligent search across the agent's memory tiers, "
                "using vector similarity and BM25 search to find relevant information. "
                "Results include matched chunks, related entities, relationships, and "
                "an AI-synthesized response summarizing the findings."
            ),
            inputSchema=SEARCH_MEMORIES_INPUT_SCHEMA,
        ),
    ]


@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict[str, Any]
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """Handle tool calls by routing to appropriate handler."""

    # Access AgentMem from lifespan context
    ctx = server.request_context
    agent_mem: AgentMem = ctx.lifespan_context["agent_mem"]

    try:
        if name == "get_active_memories":
            return await _handle_get_active_memories(agent_mem, arguments)
        elif name == "update_memory_section":
            return await _handle_update_memory_section(agent_mem, arguments)
        elif name == "search_memories":
            return await _handle_search_memories(agent_mem, arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")
    except Exception as e:
        # Return error as text content
        return [
            types.TextContent(
                type="text",
                text=f"Error executing {name}: {str(e)}",
            )
        ]


async def _handle_get_active_memories(
    agent_mem: AgentMem, arguments: dict[str, Any]
) -> list[types.TextContent]:
    """Handle get_active_memories tool call."""
    external_id = arguments["external_id"]

    # Get memories
    memories = await agent_mem.get_active_memories(external_id=external_id)

    # Format response
    if not memories:
        response = {
            "memories": [],
            "count": 0,
            "message": f"No active memories found for agent {external_id}",
        }
    else:
        response = {
            "memories": [
                {
                    "id": mem.id,
                    "external_id": mem.external_id,
                    "title": mem.title,
                    "template_content": mem.template_content,
                    "sections": mem.sections,
                    "metadata": mem.metadata,
                    "created_at": mem.created_at.isoformat(),
                    "updated_at": mem.updated_at.isoformat(),
                }
                for mem in memories
            ],
            "count": len(memories),
        }

    import json

    return [types.TextContent(type="text", text=json.dumps(response, indent=2))]


async def _handle_update_memory_section(
    agent_mem: AgentMem, arguments: dict[str, Any]
) -> list[types.TextContent]:
    """Handle update_memory_section tool call."""
    external_id = arguments["external_id"]
    memory_id = arguments["memory_id"]
    section_id = arguments["section_id"]
    new_content = arguments["new_content"]

    # Validate inputs
    if not external_id or not external_id.strip():
        raise ValueError("external_id cannot be empty")
    if not section_id or not section_id.strip():
        raise ValueError("section_id cannot be empty")
    if not new_content or not new_content.strip():
        raise ValueError("new_content cannot be empty")

    # Get current memory to track update count
    current_memories = await agent_mem.get_active_memories(external_id=external_id)
    current_memory = next((m for m in current_memories if m.id == memory_id), None)

    if not current_memory:
        raise ValueError(f"Memory {memory_id} not found for agent {external_id}")

    if section_id not in current_memory.sections:
        available_sections = ", ".join(current_memory.sections.keys())
        raise ValueError(
            f"Section '{section_id}' not found in memory. "
            f"Available sections: {available_sections}"
        )

    previous_count = current_memory.sections[section_id].get("update_count", 0)

    # Update the section
    updated_memory = await agent_mem.update_active_memory_section(
        external_id=external_id,
        memory_id=memory_id,
        section_id=section_id,
        new_content=new_content,
    )

    new_count = updated_memory.sections[section_id].get("update_count", 0)

    # Format response
    response = {
        "memory": {
            "id": updated_memory.id,
            "external_id": updated_memory.external_id,
            "title": updated_memory.title,
            "sections": updated_memory.sections,
            "updated_at": updated_memory.updated_at.isoformat(),
        },
        "section_id": section_id,
        "previous_update_count": previous_count,
        "new_update_count": new_count,
        "message": f"Section '{section_id}' updated successfully ({previous_count} -> {new_count} updates)",
    }

    import json

    return [types.TextContent(type="text", text=json.dumps(response, indent=2))]


async def _handle_search_memories(
    agent_mem: AgentMem, arguments: dict[str, Any]
) -> list[types.TextContent]:
    """Handle search_memories tool call."""
    external_id = arguments["external_id"]
    query = arguments["query"]
    search_shortterm = arguments.get("search_shortterm", True)
    search_longterm = arguments.get("search_longterm", True)
    limit = arguments.get("limit", 10)

    # Validate inputs
    if not external_id or not external_id.strip():
        raise ValueError("external_id cannot be empty")
    if not query or not query.strip():
        raise ValueError("query cannot be empty")

    # Perform search
    result = await agent_mem.retrieve_memories(
        external_id=external_id,
        query=query,
        search_shortterm=search_shortterm,
        search_longterm=search_longterm,
        limit=limit,
    )

    # Format response
    response = {
        "query": result.query,
        "synthesized_response": result.synthesized_response,
        "active_memories": [
            {
                "id": mem.id,
                "title": mem.title,
                "sections": mem.sections,
            }
            for mem in result.active_memories
        ],
        "shortterm_chunks": [
            {
                "id": chunk.id,
                "content": chunk.content,
                "similarity_score": chunk.similarity_score,
                "bm25_score": chunk.bm25_score,
            }
            for chunk in result.shortterm_chunks
        ],
        "longterm_chunks": [
            {
                "id": chunk.id,
                "content": chunk.content,
                "similarity_score": chunk.similarity_score,
                "bm25_score": chunk.bm25_score,
            }
            for chunk in result.longterm_chunks
        ],
        "entities": [
            {
                "id": entity.id,
                "name": entity.name,
                "type": entity.type,
                "description": entity.description,
                "confidence": entity.confidence,
                "importance": entity.importance,
                "memory_tier": entity.memory_tier,
            }
            for entity in result.entities
        ],
        "relationships": [
            {
                "id": rel.id,
                "from_entity_name": rel.from_entity_name,
                "to_entity_name": rel.to_entity_name,
                "type": rel.type,
                "description": rel.description,
                "confidence": rel.confidence,
                "strength": rel.strength,
                "memory_tier": rel.memory_tier,
            }
            for rel in result.relationships
        ],
        "result_counts": {
            "active": len(result.active_memories),
            "shortterm": len(result.shortterm_chunks),
            "longterm": len(result.longterm_chunks),
            "entities": len(result.entities),
            "relationships": len(result.relationships),
        },
    }

    import json

    return [types.TextContent(type="text", text=json.dumps(response, indent=2))]
