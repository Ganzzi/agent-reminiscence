"""
Search Methods Comparison Example

This example demonstrates the difference between:
1. search_memories() - Fast programmatic search without AI
2. deep_search_memories() - AI-powered search with synthesis

Use this to understand when to use each method.
"""

import asyncio
import time
from agent_reminiscence import AgentMem


async def main():
    """Compare search methods."""
    
    # Initialize
    agent_mem = AgentMem()
    await agent_mem.initialize()
    
    print("Search Methods Comparison")
    print("="*60 + "\n")
    
    try:
        external_id = "demo-user"
        
        # Create test memory
        print("Setting up test memory...")
        await agent_mem.create_active_memory(
            external_id=external_id,
            title="Technical Documentation",
            sections={
                "architecture": {
                    "content": "The system uses a microservices architecture with "
                              "FastAPI for the backend, React for the frontend, "
                              "PostgreSQL for relational data, and Redis for caching. "
                              "All services communicate via REST APIs."
                },
                "authentication": {
                    "content": "We use JWT tokens for authentication. OAuth 2.0 is "
                              "implemented for third-party integrations. Tokens expire "
                              "after 1 hour and are automatically refreshed."
                }
            }
        )
        print("✓ Test memory created\n")
        
        query = "How does authentication work in our system?"
        
        # Method 1: Fast programmatic search
        print("1. FAST SEARCH (search_memories)")
        print("-" * 60)
        print("   Use when: Quick results, no synthesis needed")
        print("   Token usage: None (no LLM calls)")
        print()
        
        start = time.time()
        result1 = await agent_mem.search_memories(
            external_id=external_id,
            query=query,
            limit=10
        )
        elapsed1 = time.time() - start
        
        print(f"   ⏱️  Time: {elapsed1:.3f}s")
        print(f"   Mode: {result1.mode}")
        print(f"   Confidence: {result1.confidence:.2f}")
        print(f"   Shortterm chunks: {len(result1.shortterm_chunks)}")
        print(f"   Longterm chunks: {len(result1.longterm_chunks)}")
        print(f"   Shortterm triplets: {len(result1.shortterm_triplets)}")
        print(f"   Longterm triplets: {len(result1.longterm_triplets)}")
        print(f"   Synthesis: {result1.synthesis}")
        print()
        
        # Method 2: AI-powered deep search
        print("2. DEEP SEARCH (deep_search_memories)")
        print("-" * 60)
        print("   Use when: Need AI synthesis, complex queries")
        print("   Token usage: Yes (LLM processes results)")
        print()
        
        start = time.time()
        result2 = await agent_mem.deep_search_memories(
            external_id=external_id,
            query=query,
            synthesis=True
        )
        elapsed2 = time.time() - start
        
        print(f"   ⏱️  Time: {elapsed2:.3f}s")
        print(f"   Mode: {result2.mode}")
        print(f"   Confidence: {result2.confidence:.2f}")
        print(f"   Shortterm chunks: {len(result2.shortterm_chunks)}")
        print(f"   Longterm chunks: {len(result2.longterm_chunks)}")
        print(f"   Shortterm triplets: {len(result2.shortterm_triplets)}")
        print(f"   Longterm triplets: {len(result2.longterm_triplets)}")
        if result2.synthesis:
            print(f"   Synthesis: {result2.synthesis[:200]}...")
        print()
        
        # Comparison summary
        print("COMPARISON")
        print("="*60)
        print(f"Speed difference: {elapsed2/elapsed1:.1f}x slower for deep search")
        print(f"\nProgrammatic search: {elapsed1:.3f}s, no tokens")
        print(f"Deep search: {elapsed2:.3f}s, uses LLM tokens")
        print()
        print("When to use PROGRAMMATIC SEARCH (search_memories):")
        print("  • You know exactly what you're looking for")
        print("  • Speed is critical")
        print("  • Don't need AI to explain/synthesize results")
        print("  • Want to minimize LLM costs")
        print()
        print("When to use DEEP SEARCH (deep_search_memories):")
        print("  • Query is complex or ambiguous")
        print("  • Need AI to synthesize information")
        print("  • Want intelligent context understanding")
        print("  • Quality over speed")
        print()
        
    finally:
        await agent_mem.close()


if __name__ == "__main__":
    asyncio.run(main())
