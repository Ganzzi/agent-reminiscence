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

# Tool 2: update_memory_sections (batch update)
UPDATE_MEMORY_SECTIONS_INPUT_SCHEMA: dict[str, Any] = {
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
        "sections": {
            "type": "array",
            "description": "Array of section updates to apply",
            "items": {
                "type": "object",
                "properties": {
                    "section_id": {
                        "type": "string",
                        "description": "ID/Name of the section to update",
                    },
                    "new_content": {
                        "type": "string",
                        "description": "New content for the section",
                    },
                },
                "required": ["section_id", "new_content"],
            },
            "minItems": 1,
        },
    },
    "required": ["external_id", "memory_id", "sections"],
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
            "description": (
                "Natural language query describing the context, problem, or information needed. "
                "This should be a clear description that combines:\n"
                "- Current context: What the user is working on or doing\n"
                "- Problem/question: What issue they're facing or what they need to know\n"
                "- Relevant details: Any specific aspects or constraints\n"
                "- Time period: When applicable, include time context (helpful for longterm search)\n\n"
                "Examples:\n"
                "- 'Working on authentication system, need to know how JWT tokens were implemented'\n"
                "- 'Debugging database connection issues, what configuration was used before?'\n"
                "- 'Writing documentation for API endpoints, what are the main features?'\n"
                "- 'Last week we discussed caching strategy, what were the decisions?'\n\n"
                "The AI will understand and search for relevant memories based on this description."
            ),
        },
        "limit": {
            "type": "integer",
            "description": "Maximum results per tier",
            "default": 10,
            "minimum": 1,
            "maximum": 100,
        },
        "synthesis": {
            "type": "boolean",
            "description": "Generate AI summary of search results (default: false, AI decides)",
            "default": False,
        },
    },
    "required": ["external_id", "query"],
}
