# Overview

Agent Mem provides a **stateless**, hierarchical memory system for AI agents. It manages three tiers of memory—Active, Shortterm, and Longterm—with automatic consolidation and intelligent retrieval.

## Key Concepts

### Stateless Design

Agent Mem is **stateless**, meaning a single `AgentMem` instance can serve multiple agents simultaneously. Each method call requires an `external_id` parameter to identify which agent's memory to access.

```python
agent_mem = AgentMem()  # One instance for all agents
await agent_mem.initialize()

# Different agents using the same instance
await agent_mem.create_active_memory(external_id="agent-1", ...)
await agent_mem.create_active_memory(external_id="agent-2", ...)
await agent_mem.create_active_memory(external_id="agent-3", ...)
```

### Template-Driven Memory

Active memories use YAML templates to define their structure. Each template specifies sections that organize information:

```yaml
template:
  id: "task_memory_v1"
  name: "Task Memory"
  version: "1.0.0"
sections:
  - id: "current_task"
    title: "Current Task"
    description: "What is being worked on now"
  - id: "progress"
    title: "Progress"
    description: "Steps completed"
  - id: "blockers"
    title: "Blockers"
    description: "Issues preventing progress"
```

### Section-Level Updates

Each section in an active memory tracks its own `update_count`. When a section reaches the update threshold, it triggers automatic consolidation to shortterm memory.

```python
# Update a section (auto-increments update_count)
await agent_mem.update_active_memory_section(
    external_id="agent-123",
    memory_id=1,
    section_id="progress",
    new_content="New progress info"
)
# If update_count >= threshold, consolidates to shortterm automatically
```

## The Four Core Methods

Agent Mem provides just **4 methods** to manage all memory operations:

### 1. Create Active Memory

Create a new template-driven working memory:

```python
memory = await agent_mem.create_active_memory(
    external_id="agent-123",
    title="Build Dashboard",
    template_content=TEMPLATE_YAML,
    initial_sections={
        "current_task": {"content": "...", "update_count": 0}
    },
    metadata={"priority": "high"}
)
```

### 2. Get Active Memories

Retrieve all active memories for an agent:

```python
memories = await agent_mem.get_active_memories(
    external_id="agent-123"
)
```

### 3. Update Section

Update a specific section (triggers consolidation when threshold reached):

```python
await agent_mem.update_active_memory_section(
    external_id="agent-123",
    memory_id=1,
    section_id="progress",
    new_content="Updated content"
)
```

### 4. Retrieve Memories

Search shortterm and longterm memories:

```python
results = await agent_mem.retrieve_memories(
    external_id="agent-123",
    query="How do I implement authentication?",
    search_shortterm=True,
    search_longterm=True,
    limit=10
)
```

## Memory Flow

```
┌─────────────────┐
│ Active Memory   │  Template + Sections (working memory)
│ (Hours-Days)    │  ├─ Section: update_count tracking
└────────┬────────┘  └─ Auto-consolidate when threshold reached
         │
         ▼ Consolidation (triggered by update_count)
┌─────────────────┐
│ Shortterm       │  Chunks + Entities + Relationships
│ (Days-Weeks)    │  ├─ Vector search (semantic)
└────────┬────────┘  ├─ BM25 search (keyword)
         │            └─ Graph relationships (Neo4j)
         │
         ▼ Promotion (based on importance score)
┌─────────────────┐
│ Longterm        │  Consolidated knowledge base
│ (Persistent)    │  ├─ Vector + BM25 search
└─────────────────┘  └─ Graph relationships
```

## Use Cases

### Task Management

```python
# Agent tracks current task
memory = await agent_mem.create_active_memory(
    external_id="agent-123",
    title="User Authentication",
    template_content=TASK_TEMPLATE,
    initial_sections={
        "current_task": {"content": "Implement OAuth", "update_count": 0},
        "progress": {"content": "Set up OAuth provider", "update_count": 0}
    }
)

# Update as work progresses
await agent_mem.update_active_memory_section(
    external_id="agent-123",
    memory_id=memory.id,
    section_id="progress",
    new_content="Completed OAuth setup, implementing token refresh"
)
```

### Knowledge Retrieval

```python
# Agent searches for relevant information
results = await agent_mem.retrieve_memories(
    external_id="agent-123",
    query="How did we implement the payment flow?"
)

# Access results
print(results.synthesized_response)
for chunk in results.shortterm_results:
    print(f"- {chunk.content}")
```

### Multi-Agent Coordination

```python
# Multiple agents share the same AgentMem instance
agent_mem = AgentMem()
await agent_mem.initialize()

# Agent 1: Backend developer
await agent_mem.create_active_memory(
    external_id="backend-agent",
    title="API Development",
    ...
)

# Agent 2: Frontend developer
await agent_mem.create_active_memory(
    external_id="frontend-agent",
    title="UI Development",
    ...
)

# Each agent manages its own memories independently
```

## Next Steps

- **[Memory Tiers](memory-tiers.md)**: Deep dive into each memory tier
- **[Active Memory](active-memory.md)**: Working with templates and sections
- **[Memory Retrieval](memory-retrieval.md)**: Advanced search techniques
- **[Best Practices](best-practices.md)**: Production tips and patterns
