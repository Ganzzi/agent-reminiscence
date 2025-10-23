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
            "description": "Array of section updates to apply (supports upsert)",
            "items": {
                "type": "object",
                "properties": {
                    "section_id": {
                        "type": "string",
                        "description": "ID/Name of the section (creates new if doesn't exist)",
                    },
                    "old_content": {
                        "type": "string",
                        "description": (
                            "Optional: Pattern to find in section content. "
                            "If null/empty: action applies to entire content. "
                            "If provided: action applies relative to this pattern."
                        ),
                    },
                    "new_content": {
                        "type": "string",
                        "description": "New content for the section or content to insert",
                    },
                    "action": {
                        "type": "string",
                        "enum": ["replace", "insert"],
                        "default": "replace",
                        "description": (
                            "Action to perform:\n"
                            "- 'replace': Replace content (entire or pattern)\n"
                            "- 'insert': Insert/append content (at end or after pattern)"
                        ),
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

# Tool: create_active_memory
CREATE_ACTIVE_MEMORY_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "external_id": {
            "type": "string",
            "description": "Unique identifier for the agent (e.g., 'agent-123', UUID, etc.)",
        },
        "title": {
            "type": "string",
            "description": "Title for this memory (e.g., 'Task Memory', 'Project Context')",
        },
        "template_content": {
            "type": "object",
            "description": (
                "REQUIRED: Template structure defining what sections this memory should have.\n\n"
                "This is a JSON object that MUST contain two required keys:\n"
                "1. 'template' (object): Metadata about the template\n"
                "   - id: Unique template identifier (e.g., 'task_memory_v1')\n"
                "   - name: Human-readable template name (e.g., 'Task Memory')\n"
                "2. 'sections' (array): List of section definitions that this memory will have\n"
                "   - Each section must have:\n"
                "     * id: Section identifier (e.g., 'current_task', 'progress', 'notes')\n"
                "     * description: What this section is for (NOT the content, just explains the purpose)\n\n"
                "IMPORTANT: 'description' field describes the PURPOSE of the section, not its initial content.\n"
                "Use 'initial_sections' parameter if you want to set initial content values.\n\n"
                "Example:\n"
                "{\n"
                '  "template": {\n'
                '    "id": "task_memory_v1",\n'
                '    "name": "Task Memory Template"\n'
                "  },\n"
                '  "sections": [\n'
                "    {\n"
                '      "id": "current_task",\n'
                '      "description": "Describes the current task being worked on"\n'
                "    },\n"
                "    {\n"
                '      "id": "progress",\n'
                '      "description": "Tracks progress and completion status"\n'
                "    },\n"
                "    {\n"
                '      "id": "notes",\n'
                '      "description": "Additional notes and observations"\n'
                "    }\n"
                "  ]\n"
                "}"
            ),
            "required": ["template", "sections"],
            "properties": {
                "template": {
                    "type": "object",
                    "required": ["id", "name"],
                    "properties": {
                        "id": {"type": "string"},
                        "name": {"type": "string"},
                    },
                },
                "sections": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "required": ["id", "description"],
                        "properties": {
                            "id": {"type": "string"},
                            "description": {"type": "string"},
                        },
                    },
                },
            },
        },
        "initial_sections": {
            "type": "object",
            "description": (
                "Optional: Set initial content values for sections when creating the memory.\n\n"
                "Format: {section_id: {content: '...', update_count: 0, ...}}\n\n"
                "- If NOT provided: All sections will be created with empty content\n"
                "- If provided: Only specified sections will have their content set, others remain empty\n"
                "- Section IDs must match those defined in template_content.sections\n\n"
                "Example:\n"
                "{\n"
                '  "current_task": {\n'
                '    "content": "# Current Task\\nImplement authentication system",\n'
                '    "update_count": 0\n'
                "  },\n"
                '  "progress": {\n'
                '    "content": "Started implementation, 30% complete",\n'
                '    "update_count": 0\n'
                "  }\n"
                "}"
            ),
        },
        "metadata": {
            "type": "object",
            "description": (
                "Optional: Additional metadata for the memory (e.g., tags, category, priority).\n"
                'Example: {"category": "development", "priority": "high"}'
            ),
        },
    },
    "required": ["external_id", "title", "template_content"],
}

# Tool: delete_active_memory
DELETE_ACTIVE_MEMORY_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "external_id": {
            "type": "string",
            "description": "Unique identifier for the agent",
        },
        "memory_id": {
            "type": "integer",
            "description": "ID of the memory to delete",
        },
    },
    "required": ["external_id", "memory_id"],
}


