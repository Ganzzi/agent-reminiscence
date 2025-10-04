"""JSON schemas for MCP tool inputs and outputs."""

from typing import Any

# Tool 1: get_active_memories
GET_ACTIVE_MEMORIES_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "external_id": {
            "type": "string",
            "description": "Unique identifier for the agent (UUID, string, or int)",
        }
    },
    "required": ["external_id"],
}

# Tool 2: update_memory_section
UPDATE_MEMORY_SECTION_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "external_id": {
            "type": "string",
            "description": "Unique identifier for the agent",
        },
        "memory_id": {
            "type": "integer",
            "description": "ID of the memory to update",
        },
        "section_id": {
            "type": "string",
            "description": "ID of the section to update (from template)",
        },
        "new_content": {
            "type": "string",
            "description": "New content for the section",
        },
    },
    "required": ["external_id", "memory_id", "section_id", "new_content"],
}

# Tool 3: search_memories
SEARCH_MEMORIES_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "external_id": {
            "type": "string",
            "description": "Unique identifier for the agent",
        },
        "query": {
            "type": "string",
            "description": "Search query describing what information is needed",
        },
        "search_shortterm": {
            "type": "boolean",
            "description": "Whether to search shortterm memory",
            "default": True,
        },
        "search_longterm": {
            "type": "boolean",
            "description": "Whether to search longterm memory",
            "default": True,
        },
        "limit": {
            "type": "integer",
            "description": "Maximum results per tier",
            "default": 10,
            "minimum": 1,
            "maximum": 100,
        },
    },
    "required": ["external_id", "query"],
}
