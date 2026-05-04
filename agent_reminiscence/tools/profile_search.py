#!/usr/bin/env python3
"""
Search performance profiling tool for agent_mem.

Usage:
    uv run python agent_reminiscence/tools/profile_search.py [--memories N] [--chunks M]
"""
import asyncio
import argparse
import time
import sys
import os
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_reminiscence import AgentMem


async def create_test_data(agent_mem: AgentMem, num_memories: int, chunks_per_memory: int):
    """Create test memories and chunks for profiling."""
    external_id = "profiling-test-agent"

    print(f"Creating {num_memories} memories with {chunks_per_memory} chunks each...")

    for i in range(num_memories):
        title = f"Test Memory {i}"
        summary = f"This is test memory number {i} with various content for search testing."

        memory = await agent_mem.create_memory(
            external_id=external_id,
            title=title,
            summary=summary,
            metadata={"test_index": i}
        )

        for j in range(chunks_per_memory):
            content = f"Chunk {j} for memory {i}. " + " ".join([f"keyword{j}_{k}" for k in range(10)])
            await agent_mem.add_memory_chunk(
                external_id=external_id,
                memory_id=memory.id,
                content=content,
                metadata={"chunk_index": j}
            )

    print(f"Created {num_memories * chunks_per_memory} chunks")


async def profile_search(agent_mem: AgentMem, external_id: str, queries: List[str]):
    """Profile search_memories performance."""
    results = []

    for query in queries:
        start = time.perf_counter()
        result = await agent_mem.search_memories(
            external_id=external_id,
            query=query,
            limit=10
        )
        elapsed = (time.perf_counter() - start) * 1000

        results.append({
            "query": query[:50],
            "elapsed_ms": elapsed,
            "shortterm_chunks": len(result.shortterm_chunks),
            "longterm_chunks": len(result.longterm_chunks),
        })

        print(f"Query: {query[:30]}... | Time: {elapsed:.1f}ms | Results: {len(result.shortterm_chunks) + len(result.longterm_chunks)}")

    return results


async def profile_deep_search(agent_mem: AgentMem, external_id: str, queries: List[str]):
    """Profile deep_search_memories performance."""
    results = []

    for query in queries:
        start = time.perf_counter()
        result = await agent_mem.deep_search_memories(
            external_id=external_id,
            query=query,
            limit=10
        )
        elapsed = (time.perf_counter() - start) * 1000

        results.append({
            "query": query[:50],
            "elapsed_ms": elapsed,
            "chunks": len(result.chunks),
            "has_synthesis": result.synthesis is not None,
        })

        print(f"Deep: {query[:30]}... | Time: {elapsed:.1f}ms | Chunks: {len(result.chunks)}")

    return results


async def main():
    parser = argparse.ArgumentParser(description="Profile search performance")
    parser.add_argument("--memories", type=int, default=10, help="Number of memories to create")
    parser.add_argument("--chunks", type=int, default=100, help="Chunks per memory")
    parser.add_argument("--profile-only", action="store_true", help="Skip data creation, just profile")
    args = parser.parse_args()

    agent_mem = AgentMem(
        postgres_host=os.getenv("POSTGRES_HOST", "localhost"),
        postgres_port=int(os.getenv("POSTGRES_PORT", "5432")),
        postgres_user=os.getenv("POSTGRES_USER", "postgres"),
        postgres_password=os.getenv("POSTGRES_PASSWORD", "postgres"),
        postgres_db=os.getenv("POSTGRES_DB", "agent_reminiscence"),
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", "neo4j"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
    )

    external_id = "profiling-test-agent"

    if not args.profile_only:
        await create_test_data(agent_mem, args.memories, args.chunks)

    test_queries = [
        "test keyword0_0",
        "chunk memory",
        "profiling search performance",
        "keyword5_5",
    ]

    print("\n=== Fast Search Profiling ===")
    fast_results = await profile_search(agent_mem, external_id, test_queries)

    avg_fast = sum(r["elapsed_ms"] for r in fast_results) / len(fast_results)
    print(f"\nAverage fast search: {avg_fast:.1f}ms")

    print("\n=== Deep Search Profiling ===")
    deep_results = await profile_deep_search(agent_mem, external_id, test_queries[:2])

    avg_deep = sum(r["elapsed_ms"] for r in deep_results) / len(deep_results)
    print(f"\nAverage deep search: {avg_deep:.1f}ms")

    print("\n=== Summary ===")
    print(f"Fast search avg: {avg_fast:.1f}ms (target: <200ms)")
    print(f"Deep search avg: {avg_deep:.1f}ms (target: <2000ms)")


if __name__ == "__main__":
    asyncio.run(main())