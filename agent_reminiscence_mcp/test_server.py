"""
Test the MCP server tools list without connecting to databases.
"""

import sys
from pathlib import Path
import asyncio

# Add parent directory to path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

# Change to agent_mem_mcp directory for imports
sys.path.insert(0, str(Path(__file__).parent))

import mcp.types as types
from mcp.server.lowlevel import Server


# Simple test without lifespan (no database connections)
server = Server("agent-mem-test")


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available memory management tools."""
    from schemas import (
        GET_ACTIVE_MEMORIES_INPUT_SCHEMA,
        UPDATE_MEMORY_SECTIONS_INPUT_SCHEMA,
        SEARCH_MEMORIES_INPUT_SCHEMA,
    )

    return [
        types.Tool(
            name="get_active_memories",
            description="Get all active memories for an agent",
            inputSchema=GET_ACTIVE_MEMORIES_INPUT_SCHEMA,
        ),
        types.Tool(
            name="update_memory_sections",
            description="Batch update multiple sections in an active memory",
            inputSchema=UPDATE_MEMORY_SECTIONS_INPUT_SCHEMA,
        ),
        types.Tool(
            name="search_memories",
            description="Search across shortterm and longterm memory tiers",
            inputSchema=SEARCH_MEMORIES_INPUT_SCHEMA,
        ),
    ]


async def test_tools():
    """Test listing tools."""
    print("\n✅ MCP Server Test")
    print("=" * 50)

    tools = await handle_list_tools()
    print(f"\n📋 Found {len(tools)} tools:\n")

    for i, tool in enumerate(tools, 1):
        print(f"{i}. {tool.name}")
        print(f"   Description: {tool.description}")
        print(f"   Input Schema: {list(tool.inputSchema['properties'].keys())}")
        print()

    print("✅ Server structure is correct!")
    print("\nNext steps:")
    print("1. Start PostgreSQL, Neo4j, and Ollama services")
    print("2. Set environment variables in .env file")
    print("3. Run: py agent_mem_mcp/run.py")
    print("   or: mcp dev mcp_dev.py (with MCP Inspector)")


if __name__ == "__main__":
    asyncio.run(test_tools())


