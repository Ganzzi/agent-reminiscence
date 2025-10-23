# Memory Management Agents Documentation

## Overview

This document describes three specialized memory management agents in the AI Army system. These agents work together to maintain a three-tier memory hierarchy (Active → Shortterm → Longterm) and help worker agents manage their knowledge and context effectively.

## Table of Contents

1. [Memory Update Agent](#memory-update-agent)
2. [Memory Consolidate Agent (Memorizer)](#memory-consolidate-agent-memorizer)
3. [Memory Retrieve Agent](#memory-retrieve-agent)
4. [Memory Workflow](#memory-workflow)
5. [Testing Memory Agents](#testing-memory-agents)

---

## Memory Update Agent

### Role and Purpose

The **Memory Update Agent** (also called **Memory Consolidation Specialist**) is responsible for maintaining the integrity and relevance of the **active memory tier** by processing new information from worker agents. It acts as a working memory manager that helps agents create and update their short-term context.

### Key Characteristics

- **Agent Name**: `MEMORY_UPDATER`
- **Model**: Google Gemini Flash 2.5
- **Temperature**: 0.5 (balanced between precision and creativity)
- **Dependencies**: `DepsBase` (worker_id, agent_name)
- **Output**: Text confirmation of memory operations

### Tools Available

The Memory Update Agent has access to three specialized tools:

#### 1. `get_detailed_active_memories`

```python
async def get_detailed_active_memories(
    ctx: RunContext[DepsBase], 
    active_memory_ids: List[int]
) -> List[ActiveMemory]
```

**Purpose**: Retrieves full content of active memories by their IDs.

**When to Use**: Before updating a memory, fetch its current state to understand what needs to be modified.

**Example**:
```python
# Fetch memories with IDs 5 and 7 to check their content
memories = await get_detailed_active_memories(
    active_memory_ids=[5, 7]
)
```

#### 2. `create_active_memory`

```python
async def create_active_memory(
    ctx: RunContext[DepsBase],
    title: str,
    content: str,
    description: Optional[str] = None,
    metadata: Optional[MetadataModel] = None,
) -> ActiveMemory
```

**Purpose**: Creates a new active memory for the worker.

**When to Use**: When the worker has new information that doesn't relate to existing memories.

**Example**:
```python
# Create a memory for a new task
memory = await create_active_memory(
    title="Q3 Marketing Plan",
    description="Marketing strategy for Q3 2025",
    content="The Q3 marketing plan focuses on three key areas: social media expansion, email campaigns, and partnership outreach. Budget allocated: $50,000."
)
```

#### 3. `update_active_memory`

```python
async def update_active_memory(
    ctx: RunContext[DepsBase],
    memory_id: int,
    title: Optional[str] = None,
    description: Optional[str] = None,
    content: Optional[str] = None,
    metadata: Optional[MetadataModel] = None,
) -> ActiveMemory
```

**Purpose**: Updates an existing active memory.

**When to Use**: When new information extends or modifies an existing memory.

**Example**:
```python
# Update the content of memory ID 5 with new information
updated_memory = await update_active_memory(
    memory_id=5,
    content="Updated Q3 marketing plan: Added influencer partnerships. New budget: $65,000."
)
```

### Workflow

The Memory Update Agent follows a strict workflow to ensure consistent memory management:

```
┌─────────────────────────────────────┐
│  1. REVIEW INPUTS                   │
│  - Worker instruction (optional)    │
│  - Message history                  │
│  - Active memory summaries          │
└─────────────┬───────────────────────┘
              │
              ↓
┌─────────────────────────────────────┐
│  2. ASSESS MEMORY RELEVANCE         │
│  - Identify new information         │
│  - Find related memories            │
│  - Fetch detailed content           │
└─────────────┬───────────────────────┘
              │
              ↓
┌─────────────────────────────────────┐
│  3. DECIDE ACTION                   │
│  - Update existing memory?          │
│  - Create new memory?               │
└─────────────┬───────────────────────┘
              │
              ↓
┌─────────────────────────────────────┐
│  4. PERFORM MEMORY OPERATION        │
│  - MUST call tool                   │
│  - Never just describe action       │
└─────────────┬───────────────────────┘
              │
              ↓
┌─────────────────────────────────────┐
│  5. CONFIRM AND DOCUMENT            │
│  - Brief confirmation               │
│  - What was updated/created         │
└─────────────────────────────────────┘
```

### Universal Active Memory Categories

When creating new memories, the agent considers these standard categories:

#### 1. Session Context
- Current task/project name and objective
- Task priority level and deadline
- Current progress status and next steps

#### 2. Communication
- Recent messages exchanged with other workers
- Pending questions or clarifications needed
- Relevant external resources or references

#### 3. Operational State
- Recently accessed tools and their outcomes
- Current working directory/environment
- Active constraints and limitations
- Recent errors and their resolutions

#### 4. Decision History
- Last 5-10 critical decisions made
- Reasoning behind key choices
- Outcomes and lessons learned
- Patterns in decision-making

### Critical Rules

**⚠️ MANDATORY TOOL USAGE**

The Memory Update Agent MUST ALWAYS call a tool. It cannot provide conversational responses without performing actual memory operations.

❌ **WRONG**:
```
"I would update the memory about the project status with the new information."
```

✅ **CORRECT**:
```
[Calls update_active_memory with actual parameters]
"Updated active memory #5 'Project Status' with new deployment information and timeline changes."
```

### Example Scenarios

#### Example 1: Creating New Memory

**Input**:
```
User provided general requirements for the application including scalability, 
performance, security, usability, maintainability, and error handling.
Created task #1 for Coordinator to incorporate them.
```

**Agent Actions**:
1. Reviews active memory summaries
2. Determines this is new comprehensive information
3. Calls `create_active_memory`:
   ```python
   create_active_memory(
       title="General Application Requirements",
       description="Comprehensive requirements provided by user",
       content="User provided the following requirements: Scalability - handle increased load, Performance - fast response times, Security - robust protection, Usability - intuitive interface, Maintainability - easy updates, Error handling - graceful error management. Task #1 created for Coordinator."
   )
   ```

**Output**:
```
Created new active memory 'General Application Requirements' with comprehensive 
user requirements including scalability, performance, security, and error handling.
```

#### Example 2: Updating Existing Memory

**Input**:
```
Fixed the token expiration bug in the authentication module. 
Updated logic in auth.py line 87.
```

**Agent Actions**:
1. Finds existing memory "Authentication Module Issues" (ID: 42)
2. Calls `get_detailed_active_memories(active_memory_ids=[42])`
3. Calls `update_active_memory`:
   ```python
   update_active_memory(
       memory_id=42,
       content="Resolved token expiration bug. Updated logic in auth.py line 87."
   )
   ```

**Output**:
```
Updated active memory #42 'Authentication Module Issues' to reflect the token 
expiration bug fix in auth.py.
```

#### Example 3: Error Case - Missing Required Field

**Input**:
```
Update the Q3 Marketing Plan with new budget allocation of $65,000.
```

**Agent Actions** (If done incorrectly):
1. Attempts to call `update_active_memory` without fetching current content
2. Misses required `content` field
3. Error: Required field `content` is missing

**Correct Actions**:
1. Calls `get_detailed_active_memories(active_memory_ids=[21])`
2. Retrieves full content
3. Calls `update_active_memory` with complete content + new budget info

### Implementation

```python
# agents/predefined_agents/memory_updater_agent.py

from typing import List
from pydantic_ai import Agent
from agents.deps_models import DepsBase
from config.agent import AgentName
from ..model_provider import model_provider
from ..tool_provider import tool_provider

def get_memory_updater_agent():
    """Factory function to create the Memory Updater Agent."""
    return Agent(
        model=model_provider.get_model(AgentName.MEMORY_UPDATER),
        deps_type=DepsBase,
        tools=tool_provider.get_tools(AgentName.MEMORY_UPDATER),
        system_prompt=get_system_prompt(AgentName.MEMORY_UPDATER),
        model_settings={"temperature": 0.5},
        retries=3,
    )

memory_updater_agent = get_memory_updater_agent()

async def run_memory_updater_agent(
    worker_id: int,
    prompt: str,
    active_summaries: List[ActiveMemorySummary],
    monitor: bool = False,
    wait_for_completion: bool = False,
    message_history: list[ModelMessage] | None = None,
) -> OutputBase:
    """
    Run the Memory Updater Agent.
    
    Args:
        worker_id: ID of the worker running the agent
        prompt: Instruction for memory update
        active_summaries: List of existing active memory summaries
        monitor: Enable monitoring hooks
        wait_for_completion: Wait for agent completion
        message_history: Previous conversation messages
        
    Returns:
        OutputBase: Confirmation of memory operations
    """
    # Format memory context
    memory_context = f"""
Here are the summaries of existing active memories:
{
    "\\n\\n".join([
        f"ID: {am.id}\\nTitle: {am.title}\\nDescription: {am.description or 'No description'}"
        for am in active_summaries
    ]) if active_summaries else "No active memories available."
}
"""
    
    final_prompt = f"{prompt}\n\n{memory_context}"
    
    return await run_agent(
        agent=memory_updater_agent,
        user_prompt=final_prompt,
        deps=DepsBase(worker_id=worker_id, agent_name=AgentName.MEMORY_UPDATER.value),
        monitor=monitor,
        wait_for_completion=wait_for_completion,
        message_history=message_history,
    )
```

### Usage in Worker Agents

Worker agents use the Memory Update Agent through the `sync_active_memory` tool:

```python
# In a worker agent's workflow
from agents.tools.memory import sync_active_memory

@worker_agent.tool
async def complete_task(ctx: RunContext[WorkerDeps]) -> str:
    """Complete a task and update memory."""
    # ... perform task work ...
    
    # Sync memory after completing work
    result = await sync_active_memory(
        ctx,
        instruction="Update memory with task completion status and outcomes"
    )
    
    return f"Task completed. {result}"
```

---

## Memory Consolidate Agent (Memorizer)

### Role and Purpose

The **Memorizer Agent** (also called **Memory Consolidation Agent**) is responsible for consolidating **active memories** into **shortterm memories** and resolving conflicts. It bridges the gap between working memory (active) and searchable, structured memory (shortterm).

### Key Characteristics

- **Agent Name**: `MEMORIZER`
- **Model**: Google Gemini Flash 2.5
- **Temperature**: 0.6 (more creative for conflict resolution)
- **Dependencies**: `MemorizerDeps` (extends DepsBase with memory IDs)
- **Output**: Confirmation of consolidation operations

### Tools Available

The Memorizer Agent has access to six specialized tools for consolidation:

#### 1. `auto_resolve`

```python
async def auto_resolve(
    ctx: RunContext[MemorizerDeps],
    shortterm_memory_id: int,
) -> AutoResolveResult
```

**Purpose**: Automatically analyzes differences between active memory and shortterm memory, extracting entities and relationships.

**Returns**:
- `active_content`: Content from active memory (for chunk operations ONLY)
- `similar_chunks`: Existing shortterm chunks to compare
- `entity_differences`: Entity conflicts (entities already extracted and created)
- `relationship_differences`: Relationship conflicts (relationships already extracted and created)

**Critical**: This tool has ALREADY extracted and created all entities and relationships. The agent only needs to resolve conflicts.

#### 2. `create_shortterm_memory`

```python
async def create_shortterm_memory(
    ctx: RunContext[DepsBase],
    title: str,
    summary: Optional[str] = None,
    metadata: Optional[MetadataModel] = None,
) -> ShorttermMemory
```

**Purpose**: Creates a new shortterm memory entry.

#### 3. `update_shortterm_memory`

```python
async def update_shortterm_memory(
    ctx: RunContext[DepsBase],
    shortterm_memory_id: int,
    title: Optional[str] = None,
    summary: Optional[str] = None,
    metadata: Optional[MetadataModel] = None,
) -> ShorttermMemory
```

**Purpose**: Updates shortterm memory metadata (title, summary).

#### 4. `update_shortterm_memory_chunks`

```python
async def update_shortterm_memory_chunks(
    ctx: RunContext[DepsBase],
    shortterm_memory_id: int,
    chunk_updates: List[ChunkUpdateData],
) -> ChunkUpdateResult
```

**Purpose**: Updates existing chunks with enhanced content.

**Example**:
```python
chunk_updates = [
    ChunkUpdateData(
        chunk_id=15,
        new_content="Enhanced content with additional context from active memory"
    ),
    ChunkUpdateData(
        chunk_id=16,
        new_content="Updated chunk with merged information"
    )
]
```

#### 5. `add_new_shortterm_memory_chunks`

```python
async def add_new_shortterm_memory_chunks(
    ctx: RunContext[DepsBase],
    shortterm_memory_id: int,
    new_chunks: List[NewChunkData],
) -> ChunkCreateResult
```

**Purpose**: Adds new chunks for content not covered by existing chunks.

**Important**: Each new chunk must include context for standalone coherence:
```python
new_chunks = [
    NewChunkData(
        content="[Context: Implementation details section] The authentication system uses JWT tokens with 24-hour expiration. Token refresh is handled automatically by the middleware.",
        metadata={"section": "implementation", "topic": "authentication"}
    ),
    NewChunkData(
        content="[Context: Security considerations] All tokens are signed with RS256. Public keys are rotated every 30 days.",
        metadata={"section": "security", "topic": "token_management"}
    )
]
```

#### 6. `update_shortterm_memory_entities_relationships`

```python
async def update_shortterm_memory_entities_relationships(
    ctx: RunContext[DepsBase],
    shortterm_memory_id: int,
    entity_updates: List[EntityUpdateData],
    relationship_updates: List[RelationshipUpdateData],
) -> EntityRelationshipUpdateResult
```

**Purpose**: Resolves conflicts in entities and relationships.

**Example**:
```python
entity_updates = [
    EntityUpdateData(
        entity_id=42,
        name="AuthenticationService",
        description="Updated: Handles JWT authentication with refresh tokens"
    )
]

relationship_updates = [
    RelationshipUpdateData(
        relationship_id=23,
        type="USES",
        description="Updated: AuthService uses JWTManager for token operations"
    )
]
```

### Workflow

The Memorizer Agent follows a strict, optimized workflow:

```
┌─────────────────────────────────────┐
│  1. ANALYSIS                        │
│  - Choose best shortterm memory     │
│  - Based on similarity/relevance    │
└─────────────┬───────────────────────┘
              │
              ↓
┌─────────────────────────────────────┐
│  2. AUTO-RESOLVE                    │
│  - Call auto_resolve tool           │
│  - Get active_content               │
│  - Get similar_chunks               │
│  - Get entity_differences           │
│  - Get relationship_differences     │
└─────────────┬───────────────────────┘
              │
              ↓
┌─────────────────────────────────────┐
│  3. CHUNK OPERATIONS                │
│  - Update similar_chunks            │
│  - Add new chunks for remaining     │
│    content (with context)           │
└─────────────┬───────────────────────┘
              │
              ↓
┌─────────────────────────────────────┐
│  4. ENTITY/RELATIONSHIP OPERATIONS  │
│  - Resolve entity_differences       │
│  - Resolve relationship_differences │
└─────────────────────────────────────┘
```

### Critical Rules

**⚠️ DATA SOURCE SEPARATION**

This is the most important concept for the Memorizer Agent:

| Data Source | Use For | Do NOT Use For |
|-------------|---------|----------------|
| `active_content` | Chunk operations only | Entity/relationship extraction |
| `entity_differences` | Entity conflict resolution | Anything else |
| `relationship_differences` | Relationship conflict resolution | Anything else |

**Why This Matters**:
- The `auto_resolve` tool has ALREADY extracted all entities and relationships from active memory
- The `auto_resolve` tool has ALREADY created all entities and relationships in the database
- The agent ONLY needs to resolve conflicts, not create new entities/relationships

**❌ FORBIDDEN ACTIONS**:
1. Do NOT extract entities/relationships from `active_content`
2. Do NOT analyze `active_content` for entity/relationship information
3. Do NOT create new entities/relationships (they're already created)

**✅ CORRECT APPROACH**:
1. Use `active_content` ONLY for updating/adding chunks
2. Use `entity_differences` ONLY for resolving entity conflicts
3. Use `relationship_differences` ONLY for resolving relationship conflicts

### Example Scenario

**Input**:
```
Active Memory Content:
"Implemented authentication system using JWT tokens. The AuthenticationService 
class handles login and token validation. It uses the JWTManager for token operations."

Shortterm Memory Chunks:
- Chunk 1: "Authentication system planned"
- Chunk 2: "Will use token-based auth"
```

**Agent Actions**:

**Step 1: Auto-Resolve**
```python
result = await auto_resolve(shortterm_memory_id=10)

# Returns:
# active_content: "Implemented authentication system using JWT tokens..."
# similar_chunks: [Chunk 1, Chunk 2]
# entity_differences: {
#   "new": [Entity(name="AuthenticationService"), Entity(name="JWTManager")],
#   "conflicts": []
# }
# relationship_differences: {
#   "new": [Relationship(from="AuthenticationService", to="JWTManager", type="USES")],
#   "conflicts": []
# }
```

**Step 2: Update Similar Chunks**
```python
await update_shortterm_memory_chunks(
    shortterm_memory_id=10,
    chunk_updates=[
        ChunkUpdateData(
            chunk_id=1,
            new_content="Authentication system implemented using JWT tokens. The AuthenticationService class handles login and token validation."
        ),
        ChunkUpdateData(
            chunk_id=2,
            new_content="Token-based authentication now live. Uses JWTManager for token operations including generation and validation."
        )
    ]
)
```

**Step 3: Resolve Entity Differences** (if conflicts exist)
```python
# In this case, no conflicts exist - entities were already created by auto_resolve
# If conflicts existed:
await update_shortterm_memory_entities_relationships(
    shortterm_memory_id=10,
    entity_updates=[
        EntityUpdateData(
            entity_id=42,
            description="Updated description with more detail"
        )
    ],
    relationship_updates=[]
)
```

### Implementation

```python
# agents/predefined_agents/memorizer_agent.py

from pydantic_ai import Agent
from agents.deps_models import MemorizerDeps
from config.agent import AgentName
from ..model_provider import model_provider
from ..tool_provider import tool_provider

memorizer_agent = Agent(
    model=model_provider.get_model(AgentName.MEMORIZER),
    deps_type=MemorizerDeps,
    tools=tool_provider.get_tools(AgentName.MEMORIZER),
    system_prompt=get_system_prompt(AgentName.MEMORIZER),
    model_settings={"temperature": 0.6},
    retries=3,
)
```

### Chunk Context Best Practices

When adding new chunks, always include context to make them standalone:

**❌ BAD** (lacks context):
```python
NewChunkData(
    content="JWT tokens expire after 24 hours."
)
```

**✅ GOOD** (includes context):
```python
NewChunkData(
    content="[Context: Authentication System - Token Management] JWT tokens expire after 24 hours. The system automatically handles token refresh through the middleware layer."
)
```

---

## Memory Retrieve Agent

### Role and Purpose

The **Memory Retrieve Agent** (also called **Memory Retrieval Specialist**) assists worker agents by searching for and retrieving the most relevant information from all three memory tiers: Active, Shortterm, and Longterm. It acts as an intelligent search engine that understands context and query intent.

### Key Characteristics

- **Agent Name**: `MEMORY_RETRIEVOR`
- **Model**: Google Gemini Flash 2
- **Temperature**: 0.6 (balanced for search precision)
- **Dependencies**: `DepsBase` (worker_id, agent_name)
- **Output**: Synthesized search results

### Tools Available

The Memory Retrieve Agent has access to three search tools, one for each memory tier:

#### 1. `get_detailed_active_memories`

```python
async def get_detailed_active_memories(
    ctx: RunContext[DepsBase],
    active_memory_ids: List[int]
) -> List[ActiveMemory]
```

**Purpose**: Retrieves detailed content of active memories by their IDs.

**When to Use**: When active memory summaries contain relevant information and you need the full details.

**Example**:
```python
# Get active memories with IDs 3 and 5
memories = await get_detailed_active_memories(
    active_memory_ids=[3, 5]
)
```

#### 2. `search_shortterm_memory`

```python
async def search_shortterm_memory(
    ctx: RunContext[DepsBase],
    search_queries: List[ShorttermSearchCriteria],
    similarity_threshold: float = 0.7,
    limit: int = 10,
) -> ShorttermMemorySearchResult
```

**Purpose**: Searches shortterm memories using semantic similarity and BM25.

**Parameters**:
- `search_queries`: List of `{shortterm_memory_id, search_content}` pairs
- `similarity_threshold`: 0.0-1.0 (lower = broader search, higher = specific)
- `limit`: Maximum results per shortterm memory

**When to Use**: When shortterm memory summaries contain relevant information.

**Example**:
```python
# Search two shortterm memories
results = await search_shortterm_memory(
    search_queries=[
        {"shortterm_memory_id": 10, "search_content": "authentication implementation JWT tokens"},
        {"shortterm_memory_id": 12, "search_content": "error handling patterns"}
    ],
    similarity_threshold=0.75,
    limit=5
)
```

**Returns**:
- Matched chunks with similarity scores
- Related entities
- Related relationships

#### 3. `search_longterm_memory`

```python
async def search_longterm_memory(
    ctx: RunContext[DepsBase],
    search_content: str,
    time_range: Optional[Tuple[datetime, datetime]] = None,
    confidence_threshold: float = 0.7,
    limit: int = 10,
) -> LongtermMemorySearchResult
```

**Purpose**: Searches longterm consolidated knowledge base.

**Parameters**:
- `search_content`: Query string for semantic search
- `time_range`: Optional temporal filter (start_date, end_date)
- `confidence_threshold`: 0.0-1.0 minimum confidence score
- `limit`: Maximum results

**When to Use**: 
- Query asks for historical/foundational knowledge
- Active and shortterm summaries don't contain relevant information
- Query suggests consolidated knowledge base information

**Example**:
```python
# Search for historical design patterns
results = await search_longterm_memory(
    search_content="microservices architecture patterns authentication",
    confidence_threshold=0.65,
    limit=10
)
```

### Workflow

The Memory Retrieve Agent follows an intelligent search strategy:

```
┌─────────────────────────────────────┐
│  1. DECONSTRUCT REQUEST             │
│  - Understand query intent          │
│  - Identify information needed      │
│  - Detect time frame implications   │
└─────────────┬───────────────────────┘
              │
              ↓
┌─────────────────────────────────────┐
│  2. IDENTIFY RELEVANT MEMORIES      │
│  - Review active summaries          │
│  - Review shortterm summaries       │
│  - Select relevant memory IDs       │
└─────────────┬───────────────────────┘
              │
              ↓
┌─────────────────────────────────────┐
│  3. STRATEGIZE AND EXECUTE          │
│  - Search active if relevant        │
│  - Search shortterm if relevant     │
│  - Search longterm if needed        │
└─────────────┬───────────────────────┘
              │
              ↓
┌─────────────────────────────────────┐
│  4. AGGREGATE RESULTS               │
│  - Collect all search results       │
│  - Identify key information         │
└─────────────┬───────────────────────┘
              │
              ↓
┌─────────────────────────────────────┐
│  5. SYNTHESIZE AND RESPOND          │
│  - Analyze aggregated results       │
│  - Synthesize key findings          │
│  - Formulate clear response         │
└─────────────────────────────────────┘
```

### Search Strategy

The agent uses intelligent heuristics to decide which memory tier to search:

#### Search Active Memory When:
- Query is about current session/task
- Active memory summaries contain relevant keywords
- Query asks for "current", "recent", "now" information
- Examples: "what am I working on?", "current task status"

#### Search Shortterm Memory When:
- Query is about recent work (hours to days)
- Shortterm memory summaries match query context
- Query needs detailed chunks, entities, or relationships
- Examples: "how did I implement X?", "what errors did I encounter?"

#### Search Longterm Memory When:
- Query asks for foundational/historical knowledge
- No relevant active or shortterm summaries found
- Query uses terms like "general", "always", "historically"
- Query is about patterns, best practices, or principles
- Examples: "what are our authentication patterns?", "historical user requirements"

#### DO NOT Search When:
- Query is about very recent information (current message)
- No summaries are provided and query suggests session-level info
- Previous search returned no results (avoid redundant searches)

### Example Scenarios

#### Example 1: Intelligent Longterm Search

**Input**:
```
Query: "previous user requirements or project history"
Active Summaries: None provided
Shortterm Summaries: None provided
```

**Agent Analysis**:
- No active/shortterm summaries available
- Query explicitly asks for "previous" and "history"
- Indicates historical/foundational information

**Agent Actions**:
```python
results = await search_longterm_memory(
    search_content="user requirements project history application specifications",
    confidence_threshold=0.6,
    limit=10
)
```

**Agent Response**:
```
Based on long-term memory search, I found consolidated information about previous 
user requirements and project history. This includes:
- Feature specifications from Q1 2024
- User feedback from beta testing
- Project milestone documentation
- Design decisions and rationale
```

#### Example 2: No Search Needed

**Input**:
```
Query: "what did the user just say in the last message"
Active Summaries: None provided
Shortterm Summaries: None provided
```

**Agent Analysis**:
- Query about very recent/current information
- This would be in active memory if it existed
- No summaries provided
- Current session information not in longterm memory

**Agent Actions**:
- NO search performed

**Agent Response**:
```
No relevant memories found for recent user messages. This type of current session 
information would typically be in active memory, but no active memory summaries 
were provided.
```

#### Example 3: Multi-Tier Search

**Input**:
```
Query: "implementation details for the authentication system"
Active Summaries: 
- ID 3: "Authentication Implementation Progress"
Shortterm Summaries:
- ID 10: "Authentication System Design and Code"
- ID 12: "Security Patterns and Best Practices"
```

**Agent Analysis**:
- Active summary #3 matches query
- Shortterm summaries #10 and #12 are relevant
- This is recent implementation work, not historical

**Agent Actions**:
```python
# Search active memory
active_results = await get_detailed_active_memories(
    active_memory_ids=[3]
)

# Search shortterm memory
shortterm_results = await search_shortterm_memory(
    search_queries=[
        {"shortterm_memory_id": 10, "search_content": "authentication implementation JWT"},
        {"shortterm_memory_id": 12, "search_content": "authentication security patterns"}
    ],
    similarity_threshold=0.75,
    limit=5
)
```

**Agent Response**:
```
Found implementation details across active and shortterm memories:

Current Implementation (Active Memory):
- JWT-based authentication system in progress
- Token generation and validation implemented
- Refresh token mechanism completed

Previous Work (Shortterm Memory):
- Chunks: Detailed authentication flow diagrams
- Entities: AuthenticationService, JWTManager, TokenValidator
- Relationships: AuthService USES JWTManager, TokenValidator VALIDATES tokens

Security Patterns Applied:
- RS256 signature algorithm
- 24-hour token expiration
- Automatic token refresh middleware
```

#### Example 4: Vague Query with Broad Search

**Input**:
```
Query: "I remember someone mentioning a design system update but can't recall details"
Active Summaries: None matching
Shortterm Summaries:
- ID 25: "Recent Slack Discussions"
```

**Agent Analysis**:
- Vague query requires broad search
- Lower threshold for wider net
- Search both shortterm and longterm

**Agent Actions**:
```python
# Search shortterm with lower threshold
shortterm_results = await search_shortterm_memory(
    search_queries=[
        {"shortterm_memory_id": 25, "search_content": "design system update"}
    ],
    similarity_threshold=0.5,  # Lower for broader search
    limit=10
)

# Search longterm for official documentation
longterm_results = await search_longterm_memory(
    search_content="design system principles and components",
    confidence_threshold=0.6,
    limit=10
)
```

**Agent Response**:
```
Found a few potentially relevant items:

Recent Discussion (Shortterm Memory):
- Slack thread mentioning new color palette
- Discussion about component library updates

Official Documentation (Longterm Memory):
- Phoenix Design System v2.0 documentation
- Component guidelines and usage patterns
- Design principles and best practices

I suggest reviewing both sources for context on the design system update.
```

### Implementation

```python
# agents/predefined_agents/memory_retrievor_agent.py

from typing import List
from pydantic_ai import Agent
from agents.deps_models import DepsBase
from config.agent import AgentName
from ..model_provider import model_provider
from ..tool_provider import tool_provider

def get_memory_retriever_agent():
    """Factory function to create the Memory Retriever Agent."""
    return Agent(
        model=model_provider.get_model(AgentName.MEMORY_RETRIEVOR),
        deps_type=DepsBase,
        tools=tool_provider.get_tools(AgentName.MEMORY_RETRIEVOR),
        system_prompt=get_system_prompt(AgentName.MEMORY_RETRIEVOR),
        model_settings={"temperature": 0.6},
        retries=3,
    )

memory_retrievor_agent = get_memory_retriever_agent()

async def run_memory_retrievor_agent(
    worker_id: int,
    prompt: str,
    shortterm_summaries: List[ShorttermMemorySummary],
    active_summaries: List[ActiveMemorySummary],
    monitor: bool = False,
    wait_for_completion: bool = False,
) -> OutputBase:
    """
    Run the Memory Retriever Agent.
    
    Args:
        worker_id: ID of the worker running the agent
        prompt: Search query
        shortterm_summaries: List of shortterm memory summaries
        active_summaries: List of active memory summaries
        monitor: Enable monitoring hooks
        wait_for_completion: Wait for agent completion
        
    Returns:
        OutputBase: Synthesized search results
    """
    # Format summaries for context
    active_context = format_active_summaries(active_summaries)
    shortterm_context = format_shortterm_summaries(shortterm_summaries)
    
    full_prompt = f"""
Search Query: {prompt}

Active Memory Summaries:
{active_context}

Shortterm Memory Summaries:
{shortterm_context}
"""
    
    return await run_agent(
        agent=memory_retrievor_agent,
        user_prompt=full_prompt,
        deps=DepsBase(worker_id=worker_id, agent_name=AgentName.MEMORY_RETRIEVOR.value),
        monitor=monitor,
        wait_for_completion=wait_for_completion,
    )
```

### Usage in Worker Agents

Worker agents use the Memory Retrieve Agent through the `retrieve_memories` tool:

```python
from agents.tools.memory import retrieve_memories

@worker_agent.tool
async def find_implementation_details(
    ctx: RunContext[WorkerDeps],
    feature_name: str
) -> str:
    """Find implementation details for a feature."""
    
    # Search memories for implementation details
    search_results = await retrieve_memories(
        ctx,
        query=f"implementation details for {feature_name} including code and design decisions"
    )
    
    return search_results
```

---

## Memory Workflow

### Complete Memory Lifecycle

```
┌──────────────────────┐
│   Worker Activity    │
│  (Actions, Events)   │
└──────────┬───────────┘
           │
           │ sync_active_memory()
           ↓
┌──────────────────────────────┐
│  MEMORY UPDATE AGENT         │
│  - Analyze message history   │
│  - Create/update active mem  │
└──────────┬───────────────────┘
           │
           │ Active Memory Updated
           ↓
┌──────────────────────────────┐
│  Active Memory (Tier 1)      │
│  - Current task context      │
│  - Recent activities         │
│  - Working memory            │
└──────────┬───────────────────┘
           │
           │ Periodic Migration
           │ (triggered by update threshold)
           ↓
┌──────────────────────────────┐
│  MEMORIZER AGENT             │
│  - Auto-resolve differences  │
│  - Update/add chunks         │
│  - Resolve entity conflicts  │
└──────────┬───────────────────┘
           │
           │ Consolidation Complete
           ↓
┌──────────────────────────────┐
│  Shortterm Memory (Tier 2)   │
│  - Detailed chunks           │
│  - Vector embeddings         │
│  - Entities & relationships  │
└──────────┬───────────────────┘
           │
           │ Importance-Based Promotion
           │ (background process)
           ↓
┌──────────────────────────────┐
│  Longterm Memory (Tier 3)    │
│  - Consolidated knowledge    │
│  - Temporal tracking         │
│  - High-confidence entities  │
└──────────────────────────────┘
           │
           │ retrieve_memories()
           ↓
┌──────────────────────────────┐
│  MEMORY RETRIEVE AGENT       │
│  - Search all tiers          │
│  - Synthesize results        │
│  - Return relevant info      │
└──────────┬───────────────────┘
           │
           ↓
┌──────────────────────┐
│   Worker Agent       │
│  (Uses information)  │
└──────────────────────┘
```

### Interaction Patterns

#### Pattern 1: Worker Updates Memory
```python
# Worker completes a task
async def complete_coding_task(ctx: RunContext[WorkerDeps]) -> str:
    # Perform work
    code_changes = await write_code(...)
    
    # Update memory
    memory_result = await sync_active_memory(
        ctx,
        instruction="Update memory with code changes and completion status"
    )
    
    return f"Task completed. {memory_result}"
```

#### Pattern 2: Worker Retrieves Information
```python
# Worker needs to recall previous work
async def continue_feature_development(
    ctx: RunContext[WorkerDeps],
    feature_name: str
) -> str:
    # Retrieve relevant memories
    relevant_info = await retrieve_memories(
        ctx,
        query=f"previous work on {feature_name} including design and implementation"
    )
    
    # Use information to continue work
    return f"Continuing feature development based on: {relevant_info}"
```

#### Pattern 3: Periodic Consolidation
```python
# Background process triggers consolidation
async def consolidate_worker_memories(worker_id: int):
    """Periodic background task to consolidate memories."""
    
    # Get eligible active memories (high update count)
    active_memories = await memory_manager.get_eligible_active_memories(
        worker_id, min_update_count=5
    )
    
    for active_memory in active_memories:
        # Find best matching shortterm memory
        best_match = await find_best_shortterm_match(
            worker_id, active_memory
        )
        
        # Run memorizer agent to consolidate
        await run_memorizer_agent(
            worker_id=worker_id,
            active_memory_id=active_memory.id,
            shortterm_memory_id=best_match.id if best_match else None
        )
```

---

## Testing Memory Agents

### Test Structure

All memory agent tests follow a similar structure:

```python
# scripts/predefined_agents/test_[agent]_agent.py

import asyncio
import logging
from datetime import datetime

class MemoryAgentTester:
    """Test class for memory agent."""
    
    def __init__(self):
        self.memory_manager = MemoryManager()
        self.test_worker_id = None
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    async def setup_test_environment(self):
        """Create test user, project, and worker."""
        await init_both_databases()
        # Create test entities
        self.test_worker_id = await self._create_test_worker()
    
    async def cleanup_test_environment(self):
        """Clean up test data."""
        # Delete test entities
        pass
    
    async def test_basic_functionality(self):
        """Test basic agent operations."""
        pass
    
    async def test_edge_cases(self):
        """Test edge cases and error handling."""
        pass

async def main():
    tester = MemoryAgentTester()
    try:
        await tester.setup_test_environment()
        await tester.test_basic_functionality()
        await tester.test_edge_cases()
        print("✅ All tests passed!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise
    finally:
        await tester.cleanup_test_environment()

if __name__ == "__main__":
    asyncio.run(main())
```

### Testing Memory Update Agent

```python
# scripts/predefined_agents/test_memory_updater_agent.py

async def test_create_new_memory(self):
    """Test creating a new active memory."""
    
    # Simulate message history with new information
    messages = [
        ModelRequest(parts=[
            SystemPromptPart(content="You are a coding agent..."),
            UserPromptPart(content="Implement authentication system")
        ]),
        ModelResponse(parts=[
            TextPart(content="Implemented JWT-based auth with refresh tokens...")
        ])
    ]
    
    # Run memory updater
    result = await run_memory_updater_agent(
        worker_id=self.test_worker_id,
        prompt="Update memory with authentication implementation",
        active_summaries=[],  # No existing memories
        message_history=messages
    )
    
    # Verify memory was created
    memories = await self.memory_manager.get_active_memory_summaries(
        self.test_worker_id
    )
    assert len(memories) > 0
    assert "authentication" in memories[0].title.lower()

async def test_update_existing_memory(self):
    """Test updating an existing active memory."""
    
    # Create initial memory
    initial_memory = await self.memory_manager.create_active_memory(
        worker_id=self.test_worker_id,
        title="Project Status",
        content="Project initialized. Setup complete."
    )
    
    # Simulate new information
    messages = [
        ModelRequest(parts=[
            UserPromptPart(content="Completed user authentication module")
        ]),
        ModelResponse(parts=[
            TextPart(content="Auth module completed successfully")
        ])
    ]
    
    # Run memory updater
    result = await run_memory_updater_agent(
        worker_id=self.test_worker_id,
        prompt="Update project status with auth completion",
        active_summaries=[initial_memory],
        message_history=messages
    )
    
    # Verify memory was updated
    updated_memory = await self.memory_manager.get_active_memory_by_id(
        initial_memory.id
    )
    assert "auth" in updated_memory.content.lower()
```

### Testing Memorizer Agent

```python
# scripts/predefined_agents/test_memorizer_agent.py

async def test_consolidate_active_to_shortterm(self):
    """Test consolidating active memory into shortterm."""
    
    # Create active memory
    active_memory = await self.memory_manager.create_active_memory(
        worker_id=self.test_worker_id,
        title="Authentication Implementation",
        content="Implemented JWT authentication with AuthService and JWTManager classes..."
    )
    
    # Create shortterm memory
    shortterm_memory = await self.memory_manager.create_shortterm_memory(
        worker_id=self.test_worker_id,
        title="Authentication System",
        summary="Auth system planning"
    )
    
    # Run memorizer agent
    result = await run_memorizer_agent(
        worker_id=self.test_worker_id,
        active_memory_id=active_memory.id,
        shortterm_memory_id=shortterm_memory.id
    )
    
    # Verify consolidation
    updated_shortterm = await self.memory_manager.get_detailed_shortterm_memory(
        worker_id=self.test_worker_id,
        shortterm_memory_id=shortterm_memory.id
    )
    
    assert len(updated_shortterm.chunks) > 0
    assert len(updated_shortterm.entities) > 0
    assert "AuthService" in str(updated_shortterm.entities)
```

### Testing Memory Retrieve Agent

```python
# scripts/predefined_agents/test_memory_retrievor_agent.py

async def test_search_active_memory(self):
    """Test searching active memory."""
    
    # Create test memories
    memory1 = await self.memory_manager.create_active_memory(
        worker_id=self.test_worker_id,
        title="Authentication Work",
        content="Working on JWT token implementation..."
    )
    
    memory2 = await self.memory_manager.create_active_memory(
        worker_id=self.test_worker_id,
        title="Database Schema",
        content="Designing user table schema..."
    )
    
    # Get summaries
    summaries = await self.memory_manager.get_active_memory_summaries(
        self.test_worker_id
    )
    
    # Run retriever agent
    result = await run_memory_retrievor_agent(
        worker_id=self.test_worker_id,
        prompt="Find information about authentication implementation",
        active_summaries=summaries,
        shortterm_summaries=[]
    )
    
    # Verify results mention authentication
    assert "authentication" in result.content.lower() or "JWT" in result.content

async def test_intelligent_tier_selection(self):
    """Test that agent selects appropriate memory tier."""
    
    # Test longterm search for historical query
    result = await run_memory_retrievor_agent(
        worker_id=self.test_worker_id,
        prompt="What are our general design principles?",
        active_summaries=[],  # No active memories
        shortterm_summaries=[]  # No shortterm memories
    )
    
    # Agent should search longterm for general/historical info
    # Verify it attempted longterm search (check logs or response)
    assert "design principles" in result.content.lower()
```

### Integration Testing

```python
async def test_full_memory_workflow(self):
    """Test complete memory workflow across all agents."""
    
    # Step 1: Create active memory (via Memory Update Agent)
    update_result = await run_memory_updater_agent(
        worker_id=self.test_worker_id,
        prompt="Create memory for new feature development",
        active_summaries=[],
        message_history=create_test_messages()
    )
    
    # Step 2: Consolidate to shortterm (via Memorizer Agent)
    active_memories = await self.memory_manager.get_active_memory_summaries(
        self.test_worker_id
    )
    
    consolidate_result = await run_memorizer_agent(
        worker_id=self.test_worker_id,
        active_memory_id=active_memories[0].id,
        shortterm_memory_id=None  # Will create new shortterm
    )
    
    # Step 3: Retrieve information (via Memory Retrieve Agent)
    shortterm_summaries = await self.memory_manager.get_shortterm_memory_summaries(
        self.test_worker_id
    )
    
    retrieve_result = await run_memory_retrievor_agent(
        worker_id=self.test_worker_id,
        prompt="Find feature development information",
        active_summaries=active_memories,
        shortterm_summaries=shortterm_summaries
    )
    
    # Verify complete workflow
    assert "feature" in retrieve_result.content.lower()
```

## Summary

The three memory management agents work together to provide a comprehensive memory system:

1. **Memory Update Agent**: Maintains working memory (active tier)
   - Creates and updates active memories
   - Processes message history
   - Follows strict tool usage rules

2. **Memorizer Agent**: Consolidates working memory into searchable knowledge (shortterm tier)
   - Auto-resolves differences
   - Updates and adds chunks
   - Resolves entity/relationship conflicts
   - Follows strict data source separation

3. **Memory Retrieve Agent**: Searches and synthesizes information (all tiers)
   - Intelligent tier selection
   - Context-aware search
   - Synthesized responses

Together, they enable AI workers to maintain context, recall information, and build knowledge over time.

