"""
Token Usage Tracking Example

This example demonstrates how to track LLM token usage in AgentMem operations.
Token tracking is crucial for monitoring costs and optimizing usage.
"""

import asyncio
from pydantic_ai import RunUsage
from agent_reminiscence import AgentMem


class TokenUsageTracker:
    """Track and log token usage across operations."""
    
    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_requests = 0
        self.operations = []
    
    async def process_usage(self, external_id: str, usage: RunUsage) -> None:
        """
        Process token usage data.
        
        Args:
            external_id: Agent identifier
            usage: RunUsage object from pydantic-ai
        """
        # Update totals
        self.total_input_tokens += usage.input_tokens or 0
        self.total_output_tokens += usage.output_tokens or 0
        self.total_requests += usage.requests or 0
        
        # Log individual operation
        operation = {
            "external_id": external_id,
            "requests": usage.requests,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "total_tokens": (usage.input_tokens or 0) + (usage.output_tokens or 0)
        }
        self.operations.append(operation)
        
        print(f"[Usage] Agent {external_id}: {operation['total_tokens']} tokens "
              f"(in: {usage.input_tokens}, out: {usage.output_tokens})")
    
    def print_summary(self):
        """Print usage summary."""
        total_tokens = self.total_input_tokens + self.total_output_tokens
        print("\n" + "="*60)
        print("TOKEN USAGE SUMMARY")
        print("="*60)
        print(f"Total Requests:      {self.total_requests}")
        print(f"Total Input Tokens:  {self.total_input_tokens:,}")
        print(f"Total Output Tokens: {self.total_output_tokens:,}")
        print(f"Total Tokens:        {total_tokens:,}")
        print(f"\nOperations Tracked:  {len(self.operations)}")
        
        # Cost estimation (example rates)
        # Note: Update these based on your actual model pricing
        input_cost_per_1m = 3.00  # $3 per 1M input tokens
        output_cost_per_1m = 15.00  # $15 per 1M output tokens
        
        estimated_cost = (
            (self.total_input_tokens / 1_000_000) * input_cost_per_1m +
            (self.total_output_tokens / 1_000_000) * output_cost_per_1m
        )
        
        print(f"\nEstimated Cost:      ${estimated_cost:.6f}")
        print("="*60 + "\n")


async def main():
    """Demonstrate token usage tracking."""
    
    # Initialize tracker
    tracker = TokenUsageTracker()
    
    # Initialize AgentMem and register usage processor
    agent_mem = AgentMem()
    await agent_mem.initialize()
    agent_mem.set_usage_processor(tracker.process_usage)
    
    print("Token Usage Tracking Demo")
    print("="*60 + "\n")
    
    try:
        external_id = "demo-user"
        
        # 1. Create active memory (no tokens used)
        print("1. Creating active memory...")
        memory = await agent_mem.create_active_memory(
            external_id=external_id,
            title="Project Planning",
            sections={
                "overview": {
                    "content": "We are building a web application with React and FastAPI. "
                              "The frontend uses TypeScript and TailwindCSS. "
                              "The backend uses PostgreSQL and Redis for caching."
                },
                "authentication": {
                    "content": "JWT tokens are used for authentication. "
                              "Refresh tokens are stored in Redis. "
                              "Access tokens expire after 15 minutes."
                }
            }
        )
        print(f"   Created memory: {memory.title}\n")
        
        # 2. Deep search (USES TOKENS - agent powered)
        print("2. Running deep search (AI-powered, uses tokens)...")
        result = await agent_mem.deep_search_memories(
            external_id=external_id,
            query="What authentication method are we using?",
            synthesis=True
        )
        print(f"   Search confidence: {result.confidence:.2f}")
        if result.synthesis:
            print(f"   Synthesis: {result.synthesis[:100]}...\n")
        
        # 3. Another deep search
        print("3. Running another deep search...")
        result2 = await agent_mem.deep_search_memories(
            external_id=external_id,
            query="Explain the technology stack",
            synthesis=True
        )
        print(f"   Search confidence: {result2.confidence:.2f}\n")
        
        # 4. Print usage summary
        tracker.print_summary()
        
        # Example: Simple logging usage processor
        print("\nAlternative: Simple logging processor")
        print("="*60)
        print("Example code:")
        print("""
async def simple_logger(external_id: str, usage: RunUsage):
    print(f"{external_id}: {usage.total_tokens} tokens")

agent_mem.set_usage_processor(simple_logger)
        """)
        
    finally:
        await agent_mem.close()


if __name__ == "__main__":
    asyncio.run(main())
