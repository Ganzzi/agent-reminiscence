"""CLI entry point for AgentMem MCP server."""

import sys
import asyncio
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mcp.server.stdio
from agent_reminiscence_mcp.server import server


def main():
    """Run the MCP server with stdio transport."""
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("AgentMem MCP Server")
        print()
        print("Usage:")
        print("  python -m mcp           # Run with stdio transport")
        print("  python mcp              # Run with stdio transport (alternative)")
        print("  python -m mcp --help    # Show this help")
        print()
        print("Tools available:")
        print("  - get_active_memories: Get all active memories for an agent")
        print("  - update_memory_section: Update a specific section in a memory")
        print("  - search_memories: Search across memory tiers")
        print()
        return

    print("Starting AgentMem MCP server with stdio transport...", file=sys.stderr)

    # Run with stdio transport (for Claude Desktop)
    async def run_server():
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())

    asyncio.run(run_server())


if __name__ == "__main__":
    main()
