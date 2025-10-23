"""MCP Server for AgentMem - Exposes memory management tools using low-level Server API."""

import json
import sys
import logging
from pathlib import Path
from datetime import datetime

# Add agent_mem to Python path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from mcp import types
from mcp.server.lowlevel import Server

from agent_reminiscence import AgentMem
from .schemas import (
    GET_ACTIVE_MEMORIES_INPUT_SCHEMA,
    UPDATE_MEMORY_SECTIONS_INPUT_SCHEMA,
    SEARCH_MEMORIES_INPUT_SCHEMA,
    CREATE_ACTIVE_MEMORY_INPUT_SCHEMA,
    DELETE_ACTIVE_MEMORY_INPUT_SCHEMA,
)


class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles datetime objects."""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger("agent_mem_mcp")


@asynccontextmanager
async def server_lifespan(_server: Server) -> AsyncIterator[dict[str, Any]]:
    """
    Manage server lifecycle - initialize AgentMem on startup.

    This creates a singleton AgentMem instance that's shared across
    all tool calls, maintaining database connections efficiently.
    """
    logger.info("Starting AgentMem MCP server...")

    # Startup: Initialize AgentMem with configuration
    from agent_reminiscence.config import get_config

    try:
        config = get_config()
        logger.info("Configuration loaded successfully")
        logger.debug(f"PostgreSQL: {config.postgres_host}:{config.postgres_port}")
        logger.debug(f"Neo4j: {config.neo4j_uri}")
        logger.debug(f"Ollama: {config.ollama_base_url}")

        agent_mem = AgentMem(config=config)
        logger.info("Initializing AgentMem instance...")
        await agent_mem.initialize()
        logger.info("AgentMem initialized successfully")
        logger.info("MCP server ready to accept requests")

        yield {"agent_mem": agent_mem}
    except Exception as e:
        logger.error(f"Failed to initialize AgentMem: {e}", exc_info=True)
        raise
    finally:
        # Cleanup: Close database connections
        logger.info("Shutting down MCP server...")
        pass


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
            name="create_active_memory",
            description=(
                "Create a new active memory (working memory) for an agent with a template-driven structure.\n\n"
                "Active memories store the agent's current work context organized into sections (e.g., 'current_task', "
                "'progress', 'notes'). The template_content defines WHAT sections exist and their PURPOSE, while "
                "initial_sections optionally sets their initial content VALUES.\n\n"
                "Required parameters:\n"
                "- external_id: Agent identifier\n"
                "- title: Memory title (e.g., 'Task Memory', 'Project Context')\n"
                "- template_content: JSON object defining the memory structure with TWO required keys:\n"
                "  * 'template': {id: 'template_id', name: 'Template Name'}\n"
                "  * 'sections': [{id: 'section_id', description: 'Purpose of this section'}, ...]\n\n"
                "Optional parameters:\n"
                "- initial_sections: Set initial content for sections: {section_id: {content: '...', update_count: 0}}\n"
                "- metadata: Additional metadata for the memory\n\n"
                "Example template_content:\n"
                "{\n"
                '  "template": {"id": "task_memory_v1", "name": "Task Memory"},\n'
                '  "sections": [\n'
                '    {"id": "current_task", "description": "What is being worked on now"},\n'
                '    {"id": "progress", "description": "Status and completion tracking"}\n'
                "  ]\n"
                "}"
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
    ]


@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict[str, Any]
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """Handle tool calls by routing to appropriate handler."""
    logger.info(f"Received tool call: {name}")
    logger.debug(f"Arguments: {arguments}")

    # Access AgentMem from lifespan context
    ctx = server.request_context
    agent_mem: AgentMem = ctx.lifespan_context["agent_mem"]

    try:
        if name == "get_active_memories":
            result = await _handle_get_active_memories(agent_mem, arguments)
            logger.info(f"Successfully executed {name}")
            return result
        elif name == "create_active_memory":
            result = await _handle_create_active_memory(agent_mem, arguments)
            logger.info(f"Successfully executed {name}")
            return result
        elif name == "update_memory_sections":
            result = await _handle_update_memory_sections(agent_mem, arguments)
            logger.info(f"Successfully executed {name}")
            return result
        elif name == "delete_active_memory":
            result = await _handle_delete_active_memory(agent_mem, arguments)
            logger.info(f"Successfully executed {name}")
            return result
        elif name == "search_memories":
            result = await _handle_search_memories(agent_mem, arguments)
            logger.info(f"Successfully executed {name}")
            return result
        else:
            logger.error(f"Unknown tool requested: {name}")
            raise ValueError(f"Unknown tool: {name}")
    except Exception as e:
        logger.error(f"Error executing {name}: {e}", exc_info=True)
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
                    # "template_content": mem.template_content,
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

    return [types.TextContent(type="text", text=json.dumps(response, indent=2, cls=DateTimeEncoder))]


async def _handle_update_memory_sections(
    agent_mem: AgentMem, arguments: dict[str, Any]
) -> list[types.TextContent]:
    """Handle update_memory_sections tool call (batch update)."""
    external_id = arguments["external_id"]
    memory_id = arguments["memory_id"]
    sections = arguments["sections"]

    # Validate inputs
    if not external_id or not external_id.strip():
        raise ValueError("external_id cannot be empty")

    if not sections or len(sections) == 0:
        raise ValueError("sections array cannot be empty")

    # Get current memory to track update counts
    current_memories = await agent_mem.get_active_memories(external_id=external_id)
    current_memory = next((m for m in current_memories if m.id == memory_id), None)

    if not current_memory:
        raise ValueError(f"Memory {memory_id} not found for agent {external_id}")

    # Validate section updates - check that sections exist in current memory
    for section_update in sections:
        section_id = section_update["section_id"]
        new_content = section_update["new_content"]

        if section_id not in current_memory.sections:
            raise ValueError(f"Section '{section_id}' not found in memory")

        if not new_content or not new_content.strip():
            raise ValueError(f"new_content for section '{section_id}' cannot be empty")

    # Track previous counts for response
    previous_counts = {}
    previous_awake_counts = {}
    for section_update in sections:
        section_id = section_update["section_id"]
        previous_counts[section_id] = current_memory.sections[section_id].get("update_count", 0)
        previous_awake_counts[section_id] = current_memory.sections[section_id].get(
            "awake_update_count", 0
        )

    # Use NEW batch update method (single call)
    updated_memory = await agent_mem.update_active_memory_sections(
        external_id=external_id,
        memory_id=memory_id,
        sections=sections,  # Pass entire sections list
    )

    # Build section updates info
    section_updates = []
    for section_update in sections:
        section_id = section_update["section_id"]
        new_count = updated_memory.sections[section_id].get("update_count", 0)
        new_awake_count = updated_memory.sections[section_id].get("awake_update_count", 0)
        section_updates.append(
            {
                "section_id": section_id,
                "previous_count": previous_counts[section_id],
                "new_count": new_count,
                "previous_awake_count": previous_awake_counts[section_id],
                "new_awake_count": new_awake_count,
            }
        )

    # Calculate total update count for consolidation info
    total_update_count = sum(
        section.get("update_count", 0) for section in updated_memory.sections.values()
    )
    num_sections = len(updated_memory.sections)

    # Get config to show threshold info (import here to avoid circular imports)
    from agent_reminiscence.config import get_config

    config = get_config()
    threshold = config.avg_section_update_count_for_consolidation * num_sections

    # Format response
    response = {
        "memory": {
            "id": updated_memory.id,
            "external_id": updated_memory.external_id,
            "title": updated_memory.title,
            "sections": updated_memory.sections,
            "updated_at": updated_memory.updated_at.isoformat(),
        },
        "updates": section_updates,
        "total_sections_updated": len(section_updates),
        "consolidation_info": {
            "total_update_count": total_update_count,
            "threshold": threshold,
            "will_consolidate": total_update_count >= threshold,
        },
        "message": f"Successfully updated {len(section_updates)} sections in single batch operation",
    }

    import json

    return [types.TextContent(type="text", text=json.dumps(response, indent=2, cls=DateTimeEncoder))]


async def _handle_search_memories(
    agent_mem: AgentMem, arguments: dict[str, Any]
) -> list[types.TextContent]:
    """Handle search_memories tool call."""
    external_id = arguments["external_id"]
    query = arguments["query"]
    limit = arguments.get("limit", 10)
    synthesis = arguments.get("synthesis", False)

    # Validate inputs
    if not external_id or not external_id.strip():
        raise ValueError("external_id cannot be empty")
    if not query or not query.strip():
        raise ValueError("query cannot be empty")

    # Perform search
    result = await agent_mem.retrieve_memories(
        external_id=external_id,
        query=query,
        limit=limit,
        synthesis=synthesis,
    )

    # Format response using the new RetrievalResult structure
    response = {
        "mode": result.mode,
        "search_strategy": result.search_strategy,
        "confidence": result.confidence,
        "chunks": [
            {
                "id": chunk.id,
                "content": chunk.content,
                "tier": chunk.tier,
                "score": chunk.score,
                **({"importance": chunk.importance} if chunk.importance is not None else {}),
                **(
                    {"start_date": chunk.start_date.isoformat()}
                    if chunk.start_date is not None
                    else {}
                ),
            }
            for chunk in result.chunks
        ],
        "entities": [
            {
                "id": entity.id,
                "name": entity.name,
                "types": entity.types,
                "description": entity.description,
                "tier": entity.tier,
                "importance": entity.importance,
            }
            for entity in result.entities
        ],
        "relationships": [
            {
                "id": rel.id,
                "from_entity_name": rel.from_entity_name,
                "to_entity_name": rel.to_entity_name,
                "types": rel.types,
                "description": rel.description,
                "tier": rel.tier,
                "importance": rel.importance,
            }
            for rel in result.relationships
        ],
        "synthesis": result.synthesis,
        "metadata": result.metadata,
        "result_counts": {
            "chunks": len(result.chunks),
            "entities": len(result.entities),
            "relationships": len(result.relationships),
        },
    }

    import json

    return [types.TextContent(type="text", text=json.dumps(response, indent=2, cls=DateTimeEncoder))]


async def _handle_create_active_memory(
    agent_mem: AgentMem, arguments: dict[str, Any]
) -> list[types.TextContent]:
    """Handle create_active_memory tool call."""
    external_id = arguments["external_id"]
    title = arguments["title"]
    template_content = arguments["template_content"]
    initial_sections = arguments.get("initial_sections", {})
    metadata = arguments.get("metadata", {})

    # Validate inputs with clear error messages
    if not external_id or not external_id.strip():
        return [
            types.TextContent(
                type="text",
                text=json.dumps(
                    {"error": "Validation Error: 'external_id' cannot be empty"}, indent=2
                ),
            )
        ]

    if not title or not title.strip():
        return [
            types.TextContent(
                type="text",
                text=json.dumps({"error": "Validation Error: 'title' cannot be empty"}, indent=2),
            )
        ]

    # Validate template_content structure with detailed error messages
    if not isinstance(template_content, dict):
        return [
            types.TextContent(
                type="text",
                text=json.dumps(
                    {
                        "error": "Validation Error: 'template_content' must be a JSON object (dictionary), not a string or other type",
                        "example": {
                            "template": {"id": "my_template", "name": "My Template"},
                            "sections": [{"id": "section1", "description": "Purpose of section 1"}],
                        },
                    },
                    indent=2,
                ),
            )
        ]

    if not template_content:
        return [
            types.TextContent(
                type="text",
                text=json.dumps(
                    {
                        "error": "Validation Error: 'template_content' cannot be empty",
                        "required_structure": {
                            "template": {"id": "...", "name": "..."},
                            "sections": [{"id": "...", "description": "..."}],
                        },
                    },
                    indent=2,
                ),
            )
        ]

    if "template" not in template_content:
        return [
            types.TextContent(
                type="text",
                text=json.dumps(
                    {
                        "error": "Validation Error: 'template_content' must have a 'template' key",
                        "received": template_content,
                        "required_structure": {
                            "template": {"id": "template_identifier", "name": "Template Name"},
                            "sections": [{"id": "section_id", "description": "Section purpose"}],
                        },
                    },
                    indent=2,
                ),
            )
        ]

    if "sections" not in template_content:
        return [
            types.TextContent(
                type="text",
                text=json.dumps(
                    {
                        "error": "Validation Error: 'template_content' must have a 'sections' key",
                        "received": template_content,
                        "required_structure": {
                            "template": {"id": "template_identifier", "name": "Template Name"},
                            "sections": [{"id": "section_id", "description": "Section purpose"}],
                        },
                    },
                    indent=2,
                ),
            )
        ]

    # Validate template object
    if not isinstance(template_content.get("template"), dict):
        return [
            types.TextContent(
                type="text",
                text=json.dumps(
                    {
                        "error": "Validation Error: 'template_content.template' must be an object",
                        "required_fields": {"id": "string", "name": "string"},
                    },
                    indent=2,
                ),
            )
        ]

    template_obj = template_content["template"]
    if "id" not in template_obj or "name" not in template_obj:
        return [
            types.TextContent(
                type="text",
                text=json.dumps(
                    {
                        "error": "Validation Error: 'template_content.template' must have 'id' and 'name' fields",
                        "received": template_obj,
                        "example": {"id": "task_memory_v1", "name": "Task Memory Template"},
                    },
                    indent=2,
                ),
            )
        ]

    # Validate sections array
    if not isinstance(template_content.get("sections"), list):
        return [
            types.TextContent(
                type="text",
                text=json.dumps(
                    {
                        "error": "Validation Error: 'template_content.sections' must be an array",
                        "example": [
                            {"id": "current_task", "description": "What is being worked on now"},
                            {"id": "progress", "description": "Status and completion tracking"},
                        ],
                    },
                    indent=2,
                ),
            )
        ]

    sections = template_content["sections"]
    if len(sections) == 0:
        return [
            types.TextContent(
                type="text",
                text=json.dumps(
                    {
                        "error": "Validation Error: 'template_content.sections' must have at least one section",
                        "example": [{"id": "main_section", "description": "Main content area"}],
                    },
                    indent=2,
                ),
            )
        ]

    # Validate each section
    for i, section in enumerate(sections):
        if not isinstance(section, dict):
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "error": f"Validation Error: Section at index {i} must be an object",
                            "received": section,
                            "required_fields": {"id": "string", "description": "string"},
                        },
                        indent=2,
                    ),
                )
            ]

        if "id" not in section or "description" not in section:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "error": f"Validation Error: Section at index {i} must have 'id' and 'description' fields",
                            "received": section,
                            "example": {
                                "id": "section_name",
                                "description": "Purpose of this section",
                            },
                        },
                        indent=2,
                    ),
                )
            ]

        if not section["id"] or not str(section["id"]).strip():
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {"error": f"Validation Error: Section at index {i} has empty 'id' field"},
                        indent=2,
                    ),
                )
            ]

    # Create memory
    try:
        memory = await agent_mem.create_active_memory(
            external_id=external_id,
            title=title,
            template_content=template_content,
            initial_sections=initial_sections,
            metadata=metadata,
        )

        # Format response
        response = {
            "success": True,
            "memory": {
                "id": memory.id,
                "external_id": memory.external_id,
                "title": memory.title,
                "template": memory.template_content.get("template", {}),
                "sections": memory.sections,
                "metadata": memory.metadata,
                "created_at": memory.created_at.isoformat(),
                "updated_at": memory.updated_at.isoformat(),
            },
            "message": f"Successfully created memory with {len(memory.sections)} sections",
        }

        logger.info(
            f"Created memory {memory.id} for {external_id} with {len(memory.sections)} sections"
        )

        return [types.TextContent(type="text", text=json.dumps(response, indent=2, cls=DateTimeEncoder))]

    except Exception as e:
        logger.error(f"Error creating memory: {e}", exc_info=True)
        return [
            types.TextContent(
                type="text",
                text=json.dumps({"error": f"Failed to create memory: {str(e)}"}, indent=2),
            )
        ]


async def _handle_delete_active_memory(
    agent_mem: AgentMem, arguments: dict[str, Any]
) -> list[types.TextContent]:
    """Handle delete_active_memory tool call."""
    external_id = arguments["external_id"]
    memory_id = arguments["memory_id"]

    # Validate inputs
    if not external_id or not external_id.strip():
        return [
            types.TextContent(
                type="text", text=json.dumps({"error": "external_id cannot be empty"}, indent=2)
            )
        ]

    # Delete memory
    try:
        success = await agent_mem.delete_active_memory(
            external_id=external_id,
            memory_id=memory_id,
        )

        if success:
            response = {
                "success": True,
                "message": f"Successfully deleted memory {memory_id}",
            }
            logger.info(f"Deleted memory {memory_id} for {external_id}")
        else:
            response = {
                "success": False,
                "error": f"Memory {memory_id} not found or does not belong to agent {external_id}",
            }
            logger.warning(f"Failed to delete memory {memory_id} for {external_id}")

        return [types.TextContent(type="text", text=json.dumps(response, indent=2, cls=DateTimeEncoder))]

    except Exception as e:
        logger.error(f"Error deleting memory: {e}", exc_info=True)
        return [
            types.TextContent(
                type="text",
                text=json.dumps({"error": f"Failed to delete memory: {str(e)}"}, indent=2),
            )
        ]


