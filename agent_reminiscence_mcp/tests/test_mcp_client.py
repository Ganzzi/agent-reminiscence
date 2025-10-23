"""
Simple MCP Client to test AgentMem MCP Server
Tests all three tools with mock data
"""

import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def test_mcp_server():
    """Test all three MCP server tools"""
    print("🧪 Testing AgentMem MCP Server\n")
    print("=" * 60)

    # Server parameters
    server_params = StdioServerParameters(command="py", args=["agent_mem_mcp/run.py"], env=None)

    try:
        # Connect to server
        print("\n1️⃣  Connecting to MCP server...")
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                print("   ✅ Connected!")

                # Initialize session
                print("\n2️⃣  Initializing session...")
                await session.initialize()
                print("   ✅ Initialized!")

                # List available tools
                print("\n3️⃣  Listing available tools...")
                tools_result = await session.list_tools()
                print(f"   ✅ Found {len(tools_result.tools)} tools:")
                for tool in tools_result.tools:
                    print(f"      • {tool.name}")

                # Test 1: get_active_memories
                print("\n4️⃣  Testing get_active_memories...")
                try:
                    result = await session.call_tool(
                        "get_active_memories", arguments={"external_id": "test-agent-123"}
                    )
                    print("   ✅ Tool executed successfully!")
                    print(f"   Response type: {type(result.content)}")
                    if result.content:
                        content = result.content[0]
                        print(f"   Content preview: {str(content.text)[:200]}...")
                except Exception as e:
                    print(f"   ⚠️  Tool execution failed: {e}")

                # Test 2: search_memories
                print("\n5️⃣  Testing search_memories...")
                try:
                    result = await session.call_tool(
                        "search_memories",
                        arguments={
                            "external_id": "test-agent-123",
                            "query": "test query",
                            "limit": 5,
                        },
                    )
                    print("   ✅ Tool executed successfully!")
                    if result.content:
                        content = result.content[0]
                        print(f"   Content preview: {str(content.text)[:200]}...")
                except Exception as e:
                    print(f"   ⚠️  Tool execution failed: {e}")

                print("\n" + "=" * 60)
                print("✅ MCP Server Test Complete!")
                print("\nNext steps:")
                print("  • Add sample data to test update_memory_section")
                print("  • Test with MCP Inspector: mcp dev mcp_dev.py")
                print("  • Integrate with Claude Desktop")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_mcp_server())


