# Usage Examples

Complete examples demonstrating agent-reminiscence v0.2.0 capabilities.

## Table of Contents

1. [Basic Memory Operations](#basic-memory-operations)
2. [Search Operations](#search-operations)
3. [Advanced Workflows](#advanced-workflows)
4. [Claude Desktop Integration](#claude-desktop-integration)
5. [Error Handling](#error-handling)
6. [Batch Operations](#batch-operations)

---

## Basic Memory Operations

### Example 1: Creating and Updating Memory

Create an agent memory and track progress over time.

```python
import asyncio
from agent_reminiscence import AgentMem
from uuid import uuid4

async def example_basic_operations():
    """Demonstrate creating and updating memory."""
    
    # Initialize AgentMem
    agent_id = "agent-task-tracker"
    async with AgentMem() as memory_service:
        
        # Step 1: Create a new memory for project planning
        project_memory = await memory_service.create_active_memory(
            external_id=agent_id,
            title="Project Planning",
            template_content={
                "template": {
                    "id": "project_v1",
                    "name": "Project Memory Template"
                },
                "sections": [
                    {
                        "id": "goals",
                        "description": "Project goals and objectives"
                    },
                    {
                        "id": "progress",
                        "description": "Current progress status"
                    },
                    {
                        "id": "blockers",
                        "description": "Known issues and blockers"
                    }
                ]
            },
            initial_sections={
                "goals": {
                    "content": "Build authentication system\n"
                               "- User registration\n"
                               "- Login/logout\n"
                               "- Token refresh"
                },
                "progress": {
                    "content": "Starting: 0% complete"
                }
            }
        )
        
        print(f"✅ Created memory: {project_memory.title}")
        print(f"   Memory ID: {project_memory.id}")
        print(f"   Sections: {list(project_memory.sections.keys())}")
        
        # Step 2: Update progress as work happens
        memory_id = project_memory.id
        
        # First update
        await memory_service.update_active_memory(
            external_id=agent_id,
            memory_id=memory_id,
            sections=[
                {
                    "section_id": "progress",
                    "action": "replace",
                    "new_content": "In Progress: 30% complete\n"
                                   "- User registration: ✅ Complete\n"
                                   "- Login/logout: 🔄 In progress\n"
                                   "- Token refresh: ⏳ Queued"
                }
            ]
        )
        
        print("\n✅ Updated progress (30%)")
        
        # Second update
        await memory_service.update_active_memory(
            external_id=agent_id,
            memory_id=memory_id,
            sections=[
                {
                    "section_id": "progress",
                    "action": "replace",
                    "new_content": "In Progress: 60% complete\n"
                                   "- User registration: ✅ Complete\n"
                                   "- Login/logout: ✅ Complete\n"
                                   "- Token refresh: 🔄 In progress"
                },
                {
                    "section_id": "blockers",
                    "action": "replace",
                    "new_content": "None currently. On track for completion."
                }
            ]
        )
        
        print("✅ Updated progress (60%)")
        
        # Step 3: Retrieve all memories for verification
        all_memories = await memory_service.get_active_memories(
            external_id=agent_id
        )
        
        print(f"\n✅ Retrieved {len(all_memories)} total memories")
        for mem in all_memories:
            print(f"   - {mem.title} (ID: {mem.id})")


# Run example
# asyncio.run(example_basic_operations())
```

**Key Points**:
- Initialize `AgentMem` with context manager
- Create memory with template and initial sections
- Update specific sections over time
- Retrieve all memories for an agent

---

## Search Operations

### Example 2: Fast Search (Pointer-based)

Quick lookups for recent information.

```python
async def example_fast_search():
    """Demonstrate fast pointer-based search."""
    
    agent_id = "data-analyst"
    async with AgentMem() as memory_service:
        
        # Create several memories
        analysis_memory = await memory_service.create_active_memory(
            external_id=agent_id,
            title="Data Analysis Results",
            template_content={
                "template": {"id": "analysis_v1", "name": "Analysis"},
                "sections": [
                    {"id": "findings", "description": "Key findings"},
                    {"id": "metrics", "description": "Important metrics"}
                ]
            },
            initial_sections={
                "findings": {
                    "content": "Q4 2025 Analysis:\n"
                               "- Customer retention: 94.5% (up from 92%)\n"
                               "- Average transaction value: $156.32\n"
                               "- Peak usage hours: 2-4 PM EST"
                }
            }
        )
        
        # Perform fast search
        print("🔍 Performing fast search for 'customer retention'...")
        
        results = await memory_service.search_memories(
            external_id=agent_id,
            query="customer retention metrics",
            limit=5
        )
        
        print(f"✅ Found {len(results.chunks)} relevant chunks")
        print(f"   Entities: {len(results.entities)}")
        print(f"   Relationships: {len(results.relationships)}")
        
        # Display results
        if results.chunks:
            print("\n📋 First Result:")
            chunk = results.chunks[0]
            print(f"   Content: {chunk.content[:100]}...")
            print(f"   Score: {chunk.score:.2%}")
            print(f"   Importance: {chunk.importance:.2%}")


# Run example
# asyncio.run(example_fast_search())
```

**Key Points**:
- `search_memories()` returns results in < 200ms
- No AI synthesis (pointer-based retrieval)
- Returns chunks, entities, relationships
- Perfect for quick fact lookups

---

### Example 3: Deep Search (Synthesis-based) ⭐

Comprehensive analysis with AI-generated insights.

```python
async def example_deep_search():
    """Demonstrate deep search with synthesis."""
    
    agent_id = "strategic-planner"
    async with AgentMem() as memory_service:
        
        # Create memories with multiple perspectives
        await memory_service.create_active_memory(
            external_id=agent_id,
            title="Market Analysis",
            template_content={
                "template": {"id": "market_v1", "name": "Market"},
                "sections": [
                    {"id": "trends", "description": "Market trends"},
                    {"id": "competitors", "description": "Competitor analysis"}
                ]
            },
            initial_sections={
                "trends": {
                    "content": "Market Trends 2025:\n"
                               "- Shift toward AI-powered solutions\n"
                               "- Increased focus on sustainability\n"
                               "- Remote-first becoming standard"
                }
            }
        )
        
        await memory_service.create_active_memory(
            external_id=agent_id,
            title="Strategy Planning",
            template_content={
                "template": {"id": "strategy_v1", "name": "Strategy"},
                "sections": [
                    {"id": "approach", "description": "Strategic approach"},
                    {"id": "roadmap", "description": "Implementation roadmap"}
                ]
            },
            initial_sections={
                "approach": {
                    "content": "Strategic Approach:\n"
                               "- Leverage AI capabilities\n"
                               "- Focus on sustainability initiatives\n"
                               "- Build remote-first infrastructure"
                }
            }
        )
        
        # Perform deep search for insights
        print("🔍 Performing deep search for market insights...")
        
        results = await memory_service.deep_search_memories(
            external_id=agent_id,
            query="How do market trends align with our strategic approach?",
            limit=10
        )
        
        print(f"✅ Search complete")
        print(f"   Chunks: {len(results.chunks)}")
        print(f"   Entities: {len(results.entities)}")
        print(f"   Relationships: {len(results.relationships)}")
        
        # Display synthesis
        if results.synthesis:
            print("\n🤖 AI Synthesis:")
            print(results.synthesis[:500] + "...")
        
        # Display entities and relationships
        if results.entities:
            print(f"\n📊 Key Entities ({len(results.entities)}):")
            for entity in results.entities[:3]:
                print(f"   - {entity.name}: {entity.description}")
        
        if results.relationships:
            print(f"\n🔗 Key Relationships ({len(results.relationships)}):")
            for rel in results.relationships[:3]:
                print(f"   - {rel.from_entity_name} {rel.types[0]} {rel.to_entity_name}")


# Run example
# asyncio.run(example_deep_search())
```

**Key Points**:
- `deep_search_memories()` takes 500ms-2s
- Includes AI-generated synthesis summary
- Extracts and returns entities and relationships
- Better for complex, multi-part questions
- Returns full context for decision making

---

## Advanced Workflows

### Example 4: Complete Knowledge Base Workflow

Build and query a multi-faceted knowledge base.

```python
async def example_knowledge_base():
    """Demonstrate complete knowledge base workflow."""
    
    agent_id = "documentation-bot"
    async with AgentMem() as memory_service:
        
        # Step 1: Create multiple specialized memories
        print("📚 Building knowledge base...")
        
        # API documentation memory
        api_memory = await memory_service.create_active_memory(
            external_id=agent_id,
            title="API Documentation",
            template_content={
                "template": {"id": "api_docs_v1", "name": "API Docs"},
                "sections": [
                    {"id": "endpoints", "description": "API endpoints"},
                    {"id": "authentication", "description": "Auth methods"},
                    {"id": "errors", "description": "Error codes"}
                ]
            },
            initial_sections={
                "endpoints": {
                    "content": "REST Endpoints:\n"
                               "- GET /api/users - List users\n"
                               "- POST /api/users - Create user\n"
                               "- GET /api/users/{id} - Get user\n"
                               "- PUT /api/users/{id} - Update user\n"
                               "- DELETE /api/users/{id} - Delete user"
                },
                "authentication": {
                    "content": "Authentication:\n"
                               "- Bearer token in Authorization header\n"
                               "- Token expires after 24 hours\n"
                               "- Refresh endpoint: POST /api/auth/refresh"
                }
            }
        )
        
        # Best practices memory
        practices_memory = await memory_service.create_active_memory(
            external_id=agent_id,
            title="Best Practices",
            template_content={
                "template": {"id": "practices_v1", "name": "Best Practices"},
                "sections": [
                    {"id": "design", "description": "API design principles"},
                    {"id": "security", "description": "Security guidelines"}
                ]
            },
            initial_sections={
                "design": {
                    "content": "API Design Principles:\n"
                               "- RESTful URL structure\n"
                               "- Consistent response formats\n"
                               "- Versioning strategy: /api/v1/...\n"
                               "- Pagination for list endpoints"
                },
                "security": {
                    "content": "Security Guidelines:\n"
                               "- Always validate input\n"
                               "- Use HTTPS only\n"
                               "- Implement rate limiting\n"
                               "- Log security events"
                }
            }
        )
        
        print("✅ Created 2 knowledge base memories")
        
        # Step 2: Update memories with additional details
        print("\n📝 Updating with detailed information...")
        
        await memory_service.update_active_memory(
            external_id=agent_id,
            memory_id=api_memory.id,
            sections=[
                {
                    "section_id": "errors",
                    "action": "replace",
                    "new_content": "Common Error Codes:\n"
                                   "- 400 Bad Request: Invalid parameters\n"
                                   "- 401 Unauthorized: Missing or invalid token\n"
                                   "- 403 Forbidden: Insufficient permissions\n"
                                   "- 404 Not Found: Resource not found\n"
                                   "- 429 Too Many Requests: Rate limited"
                }
            ]
        )
        
        print("✅ Updated API documentation with error codes")
        
        # Step 3: Search for specific information
        print("\n🔍 Querying knowledge base...")
        
        # Query 1: API design
        auth_search = await memory_service.deep_search_memories(
            external_id=agent_id,
            query="How should we authenticate API requests and what are the security considerations?",
            limit=5
        )
        
        print(f"✅ Found {len(auth_search.chunks)} chunks on authentication")
        
        # Query 2: Best practices
        design_search = await memory_service.deep_search_memories(
            external_id=agent_id,
            query="What are the key API design principles and security guidelines?",
            limit=5
        )
        
        print(f"✅ Found {len(design_search.chunks)} chunks on design practices")
        
        # Step 4: Display comprehensive results
        print("\n📊 Knowledge Base Summary:")
        all_memories = await memory_service.get_active_memories(
            external_id=agent_id
        )
        print(f"   Total Memories: {len(all_memories)}")
        print(f"   Total Chunks Indexed: {len(auth_search.chunks) + len(design_search.chunks)}")


# Run example
# asyncio.run(example_knowledge_base())
```

**Key Points**:
- Create multiple specialized memories
- Update memories with additional details
- Perform deep searches across knowledge base
- Use synthesis to answer complex questions

---

## Claude Desktop Integration

### Example 5: Claude Desktop Tool Usage

Using agent-reminiscence tools within Claude Desktop conversations.

**Prerequisites**:
1. Start MCP server:
```bash
python -m agent_reminiscence_mcp.run
```

2. Configure Claude Desktop:
```json
{
  "mcpServers": {
    "agent-mem": {
      "command": "python",
      "args": ["-m", "agent_reminiscence_mcp.run"],
      "env": {
        "POSTGRES_HOST": "localhost",
        "NEO4J_URI": "bolt://localhost:7687",
        "OLLAMA_BASE_URL": "http://localhost:11434"
      }
    }
  }
}
```

**Claude Conversation Example**:

```
You: "I'm working on a project and need to track my progress. 
Create a memory for me with sections for goals, progress, and blockers."

Claude: I'll create a project memory for you with those sections.
[Uses: create_active_memory(
    external_id="agent-project-001",
    title="Project Progress Tracker",
    template_content={...},
    initial_sections={...}
)]
Result: Created memory successfully with ID 123

You: "Now update the progress section to 40% complete"

Claude: I'll update your progress.
[Uses: update_memory_sections(
    external_id="agent-project-001",
    memory_id=123,
    sections=[{section_id: "progress", action: "replace", 
              new_content: "Progress: 40% complete..."}]
)]
Result: Updated successfully

You: "Search for information about my project blockers"

Claude: I'll search for blocker-related information in your memories.
[Uses: search_memories(
    external_id="agent-project-001",
    query="blockers and issues",
    limit=5
)]
Result: Found 2 relevant chunks...

You: "Now give me a deep analysis of how my progress relates to my goals"

Claude: I'll perform a comprehensive analysis.
[Uses: deep_search_memories(
    external_id="agent-project-001",
    query="How does current progress compare to original goals?",
    limit=10
)]
Result: [synthesis from AI]
Your progress on authentication (60% complete) aligns well with the first goal 
of implementing login/logout. The blockers identified are mainly around token 
refresh logic, which is the remaining 40% of work...
```

**Key Points**:
- All 6 MCP tools available in Claude
- Seamless integration with conversations
- Tool results flow into chat context
- Can chain multiple tool calls

---

## Error Handling

### Example 6: Robust Error Handling

Handle common errors gracefully.

```python
async def example_error_handling():
    """Demonstrate error handling patterns."""
    
    agent_id = "error-demo"
    async with AgentMem() as memory_service:
        
        # Error 1: Invalid external_id
        print("Testing error handling...")
        
        try:
            await memory_service.get_active_memories(external_id="")
        except ValueError as e:
            print(f"✅ Caught expected error: {e}")
        
        # Error 2: Memory not found
        try:
            await memory_service.update_active_memory(
                external_id=agent_id,
                memory_id=9999,  # Non-existent
                sections=[]
            )
        except ValueError as e:
            print(f"✅ Caught expected error: {e}")
        
        # Error 3: Invalid parameters
        try:
            memory = await memory_service.create_active_memory(
                external_id=agent_id,
                title="",  # Empty title
                template_content={"template": {}, "sections": []}
            )
        except ValueError as e:
            print(f"✅ Caught expected error: {e}")
        
        # Error 4: Graceful handling in search
        print("\n🔍 Searching with potentially invalid query...")
        
        try:
            results = await memory_service.search_memories(
                external_id=agent_id,
                query="",  # Empty query
                limit=5
            )
        except ValueError as e:
            print(f"✅ Caught expected error: {e}")
        
        # Success case with proper error handling
        print("\n✅ Creating memory with error handling...")
        
        try:
            memory = await memory_service.create_active_memory(
                external_id=agent_id,
                title="Safe Memory",
                template_content={
                    "template": {"id": "safe_v1", "name": "Safe"},
                    "sections": [
                        {"id": "content", "description": "Content section"}
                    ]
                },
                initial_sections={
                    "content": {"content": "Some safe content"}
                }
            )
            print(f"✅ Successfully created memory: {memory.title}")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            # Log error and alert
            raise


# Run example
# asyncio.run(example_error_handling())
```

**Common Errors to Handle**:
- `ValueError`: Invalid parameters or missing required fields
- `RuntimeError`: AgentMem not initialized
- Database connection errors
- Timeout errors for deep search

---

## Batch Operations

### Example 7: Batch Memory Operations

Efficiently handle multiple memory operations.

```python
async def example_batch_operations():
    """Demonstrate batch memory operations."""
    
    agent_id = "batch-processor"
    async with AgentMem() as memory_service:
        
        print("📦 Starting batch operations...")
        
        # Create multiple memories efficiently
        memories = []
        topics = [
            ("Python Basics", ["variables", "functions", "classes"]),
            ("Web Development", ["HTTP", "REST", "APIs"]),
            ("Databases", ["SQL", "NoSQL", "Indexing"])
        ]
        
        for title, sections in topics:
            section_list = [
                {"id": sec, "description": f"Information about {sec}"}
                for sec in sections
            ]
            
            memory = await memory_service.create_active_memory(
                external_id=agent_id,
                title=title,
                template_content={
                    "template": {"id": f"{title.lower()}_v1", "name": title},
                    "sections": section_list
                }
            )
            memories.append(memory)
            print(f"✅ Created: {title}")
        
        # Batch update all memories
        print("\n📝 Batch updating all memories...")
        
        for memory in memories:
            await memory_service.update_active_memory(
                external_id=agent_id,
                memory_id=memory.id,
                sections=[
                    {
                        "section_id": memory.sections[list(memory.sections.keys())[0]],
                        "action": "replace",
                        "new_content": f"Updated content for {memory.title}. Last updated: 2025-11-16"
                    }
                ]
            )
        
        print(f"✅ Updated {len(memories)} memories")
        
        # Batch search across all memories
        print("\n🔍 Batch searching all memories...")
        
        search_queries = [
            "What are the key programming concepts?",
            "How do APIs work?",
            "What is database indexing?"
        ]
        
        all_results = []
        for query in search_queries:
            results = await memory_service.deep_search_memories(
                external_id=agent_id,
                query=query,
                limit=3
            )
            all_results.append(results)
            print(f"✅ Searched: '{query}' - {len(results.chunks)} chunks found")
        
        # Summary
        print(f"\n📊 Batch Summary:")
        print(f"   Memories Created: {len(memories)}")
        print(f"   Memories Updated: {len(memories)}")
        print(f"   Search Queries: {len(search_queries)}")
        print(f"   Total Chunks: {sum(len(r.chunks) for r in all_results)}")


# Run example
# asyncio.run(example_batch_operations())
```

**Key Points**:
- Create multiple memories programmatically
- Update memories in batch
- Perform multiple searches efficiently
- Track results for analysis

---

## Quick Reference

### Memory Creation Pattern

```python
memory = await service.create_active_memory(
    external_id="agent-123",
    title="Memory Title",
    template_content={
        "template": {"id": "template_id", "name": "Template Name"},
        "sections": [
            {"id": "section_1", "description": "First section"},
            {"id": "section_2", "description": "Second section"}
        ]
    },
    initial_sections={
        "section_1": {"content": "Initial content"}
    }
)
```

### Search Pattern

```python
# Fast search (< 200ms)
results = await service.search_memories(
    external_id="agent-123",
    query="search query",
    limit=5
)

# Deep search (500ms-2s, with synthesis)
results = await service.deep_search_memories(
    external_id="agent-123",
    query="complex question",
    limit=10
)
```

### Update Pattern

```python
await service.update_active_memory(
    external_id="agent-123",
    memory_id=1,
    sections=[
        {
            "section_id": "section_name",
            "action": "replace",  # or "insert"
            "new_content": "New content"
        }
    ]
)
```

---

## Version

- Examples Version: 1.0
- Agent Reminiscence: v0.2.0
- Last Updated: November 16, 2025
