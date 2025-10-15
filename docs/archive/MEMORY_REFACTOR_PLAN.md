# AgentMem Memory System Refactor Plan

**Created:** October 14, 2025  
**Status:** Planning Phase  
**Priority:** High

---

## Table of Contents

1. [Overview](#overview)
2. [Changes Summary](#changes-summary)
3. [Phase 1: Schema and Database Layer](#phase-1-schema-and-database-layer)
4. [Phase 2: Core Service Layer](#phase-2-core-service-layer)
5. [Phase 3: MCP Server Updates](#phase-3-mcp-server-updates)
6. [Phase 4: New MCP Tools](#phase-4-new-mcp-tools)
7. [Phase 5: Streamlit UI Updates](#phase-5-streamlit-ui-updates)
8. [Phase 6: Tests and Documentation](#phase-6-tests-and-documentation)
9. [Migration Strategy](#migration-strategy)
10. [Files Impact Matrix](#files-impact-matrix)

---

## Overview

This refactor implements four major improvements to the AgentMem memory system:

1. **Template Content JSON Refactor**: Convert `template_content` from YAML string to JSON structure with default section values
2. **Sections Upsert Logic**: Change section updates from update-only to upsert (insert or update) with content manipulation
3. **Enhanced Section Tracking**: Add `awake_update_count` (persistent counter) and `last_updated` timestamp to sections
4. **New MCP Tools**: Add `create_active_memory` and `delete_active_memory` MCP tools

---

## Changes Summary

### 1. Template Content Refactor (YAML → JSON)

**Current State:**
```yaml
# template_content field (TEXT)
template:
  id: "bmad.architect.api-design.v1"
  name: "Architect — API Design"
sections:
  - id: "api_purpose"
    title: "API Purpose"
    description: "What the API enables..."
```

**New State:**
```json
// template_content field (JSONB)
{
  "template": {
    "id": "bmad.architect.api-design.v1",
    "name": "Architect — API Design"
  },
  "sections": [
    {
      "id": "api_purpose",
      "description": "What the API enables. Example: 'Public REST API for third-party integrations to read/write user tasks and projects. Supports webhooks for real-time updates.'"
    }
  ]
}
```

**Behavior:**
- `template_content` defines default values for sections
- `initial_sections` parameter can override defaults when creating memory
- Template is stored in JSONB for easier querying and updates
- Section `description` becomes the default content when creating sections

### 2. Enhanced Section Structure

**Current Section Structure:**
```json
{
  "section_id": {
    "content": "markdown content...",
    "update_count": 5  // Reset to 0 on consolidation
  }
}
```

**New Section Structure:**
```json
{
  "section_id": {
    "content": "markdown content...",
    "update_count": 5,              // Reset to 0 on consolidation
    "awake_update_count": 23,       // NEVER reset, tracks total updates
    "last_updated": "2025-10-14T10:30:00Z"  // ISO timestamp
  }
}
```

**Tracking Logic:**
- `update_count`: Incremented on update, reset to 0 after consolidation (used for consolidation trigger)
- `awake_update_count`: Incremented on update, NEVER reset (for future sleep/wake features)
- `last_updated`: Updated with current timestamp on every update

### 3. Upsert Section Logic

**Current Behavior:**
- `update_active_memory_sections` only updates existing sections
- Throws error if section doesn't exist
- Always replaces entire content

**New Behavior:**
- Supports **upsert**: insert new sections or update existing ones
- If section doesn't exist, creates it (also updates template_content to include new section)
- Supports two actions:
  - `replace`: Replace content (with old_content pattern matching)
  - `insert`: Append new content

**New Section Update Schema:**
```python
{
    "section_id": "progress",
    "old_content": "# Old section header",  # Optional: pattern to replace
    "new_content": "# New content",
    "action": "replace"  # "replace" or "insert"
}
```

**Action Logic:**

1. **Replace (action="replace")**:
   - If `old_content` is `null` or empty: Replace entire section content
   - If `old_content` is provided: Replace that specific substring with `new_content`

2. **Insert (action="insert")**:
   - If `old_content` is `null` or empty: Append `new_content` at the end
   - If `old_content` is provided: Insert `new_content` right after `old_content`

3. **New Section (section doesn't exist)**:
   - Create new section with `new_content` as initial content
   - Add section definition to `template_content.sections` array

### 4. New MCP Tools

Add two new MCP tools for memory lifecycle management:

1. **`create_active_memory`**
   - Create new active memory
   - Parameters: external_id, title, template_content, initial_sections, metadata

2. **`delete_active_memory`**
   - Delete active memory
   - Parameters: external_id, memory_id

---

## Phase 1: Schema and Database Layer

### 1.1 Update SQL Schema

**File:** `agent_mem/sql/schema.sql`

**Changes:**

```sql
-- Update active_memory table
CREATE TABLE IF NOT EXISTS active_memory (
    id SERIAL PRIMARY KEY,
    external_id VARCHAR(255) NOT NULL,
    title VARCHAR(500) NOT NULL,
    template_content JSONB NOT NULL,  -- Changed from TEXT to JSONB
    sections JSONB DEFAULT '{}',       -- Updated structure with new fields
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Add index for template_content JSONB queries
CREATE INDEX IF NOT EXISTS idx_active_memory_template ON active_memory USING gin (template_content);

-- Comment updates
COMMENT ON COLUMN active_memory.template_content IS 
  'JSONB template with structure: {template: {id, name}, sections: [{id, description}]}';

COMMENT ON COLUMN active_memory.sections IS 
  'JSONB sections map: {section_id: {content, update_count, awake_update_count, last_updated}}';
```

**Checklist:**
- [ ] Change `template_content` from `TEXT` to `JSONB`
- [ ] Add GIN index for `template_content`
- [ ] Update comments for both `template_content` and `sections`
- [ ] Document new section structure in comments

---

### 1.2 Update Active Memory Repository

**File:** `agent_mem/database/repositories/active_memory.py`

#### 1.2.1 Update `create()` Method

**Changes:**
- Accept `template_content` as dict (JSONB), not string
- Merge `template_content.sections` with `initial_sections` (initial_sections override defaults)
- Initialize sections with all new fields: `content`, `update_count`, `awake_update_count`, `last_updated`

**Checklist:**
- [ ] Change `template_content` parameter type from `str` to `Dict[str, Any]`
- [ ] Add logic to extract default sections from template_content
- [ ] Merge defaults with initial_sections (initial_sections take precedence)
- [ ] Initialize all section fields (update_count=0, awake_update_count=0, last_updated=None)
- [ ] Store template_content as JSONB in database
- [ ] Update docstring with new structure

**Implementation:**
```python
async def create(
    self,
    external_id: str,
    title: str,
    template_content: Dict[str, Any],  # Changed from str
    initial_sections: Dict[str, Dict[str, Any]],
    metadata: Dict[str, Any],
) -> ActiveMemory:
    """
    Create a new active memory with template and sections.
    
    Args:
        external_id: Agent identifier
        title: Memory title
        template_content: JSON template with structure:
            {
                "template": {"id": "...", "name": "..."},
                "sections": [
                    {
                        "id": "section_id",
                        "description": "Default content for the section"
                    }
                ]
            }
        initial_sections: Initial sections that override template defaults
            {"section_id": {"content": "...", "update_count": 0, ...}}
        metadata: Metadata dictionary
    """
    from datetime import datetime, timezone
    
    # Extract default sections from template
    template_sections = template_content.get("sections", [])
    
    # Build sections dict with defaults from template
    sections = {}
    for tmpl_section in template_sections:
        section_id = tmpl_section["id"]
        sections[section_id] = {
            "content": tmpl_section.get("description", ""),  # description becomes default content
            "update_count": tmpl_section.get("update_count", 0),
            "awake_update_count": tmpl_section.get("awake_update_count", 0),
            "last_updated": tmpl_section.get("last_updated")
        }
    
    # Override with initial_sections if provided
    for section_id, section_data in initial_sections.items():
        if section_id in sections:
            # Override existing defaults
            sections[section_id].update(section_data)
        else:
            # Add new section (ensure all fields)
            sections[section_id] = {
                "content": section_data.get("content", ""),
                "update_count": section_data.get("update_count", 0),
                "awake_update_count": section_data.get("awake_update_count", 0),
                "last_updated": section_data.get("last_updated")
            }
    
    # Ensure last_updated is ISO string if datetime
    for section_id, section_data in sections.items():
        if isinstance(section_data.get("last_updated"), datetime):
            sections[section_id]["last_updated"] = section_data["last_updated"].isoformat()
    
    query = """
        INSERT INTO active_memory 
        (external_id, title, template_content, sections, metadata)
        VALUES ($1, $2, $3, $4, $5)
        RETURNING id, external_id, title, template_content, sections, 
                  metadata, created_at, updated_at
    """
    
    async with self.postgres.connection() as conn:
        result = await conn.execute(
            query, 
            [external_id, title, template_content, sections, metadata]
        )
        rows = result.result()
        return self._row_to_model(rows[0])
```

#### 1.2.2 Create New `upsert_sections()` Method

**Purpose:** Replace `update_sections()` with upsert logic supporting insert, replace, and pattern matching

**Checklist:**
- [ ] Create new `upsert_sections()` method
- [ ] Support section insertion (if section doesn't exist)
- [ ] Support content replacement (with optional pattern matching)
- [ ] Support content insertion/appending
- [ ] Update `template_content.sections` when adding new sections
- [ ] Increment both `update_count` and `awake_update_count`
- [ ] Update `last_updated` timestamp

**Implementation:**
```python
async def upsert_sections(
    self,
    memory_id: int,
    section_updates: List[Dict[str, Any]],
) -> Optional[ActiveMemory]:
    """
    Upsert multiple sections in an active memory (batch operation).
    
    Supports:
    - Inserting new sections (adds to template_content.sections)
    - Updating existing sections
    - Content replacement (with optional pattern matching)
    - Content insertion/appending
    
    Args:
        memory_id: Memory ID
        section_updates: List of section updates:
            [
                {
                    "section_id": "progress",
                    "old_content": "# Old",  # Optional: pattern to find
                    "new_content": "# New",
                    "action": "replace"  # "replace" or "insert"
                }
            ]
    
    Returns:
        Updated ActiveMemory or None if not found
    
    Action Behaviors:
        - replace + no old_content: Replace entire section content
        - replace + old_content: Replace substring matching old_content
        - insert + no old_content: Append new_content at end
        - insert + old_content: Insert new_content after old_content
    """
    from datetime import datetime, timezone
    
    # Get current state
    current = await self.get_by_id(memory_id)
    if not current:
        logger.warning(f"Active memory {memory_id} not found")
        return None
    
    updated_sections = current.sections.copy()
    updated_template = current.template_content.copy()
    template_sections = updated_template.get("sections", [])
    
    for update in section_updates:
        section_id = update["section_id"]
        old_content = update.get("old_content")
        new_content = update.get("new_content", "")
        action = update.get("action", "replace")
        
        # Check if section exists
        if section_id not in updated_sections:
            # INSERT NEW SECTION
            logger.info(f"Inserting new section '{section_id}' in memory {memory_id}")
            
            # Add to sections
            updated_sections[section_id] = {
                "content": new_content,
                "update_count": 1,
                "awake_update_count": 1,
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
            
            # Add to template_content.sections
            template_sections.append({
                "id": section_id,
                "description": "Dynamically added section"
            })
            updated_template["sections"] = template_sections
            
        else:
            # UPDATE EXISTING SECTION
            current_content = updated_sections[section_id]["content"]
            current_update_count = updated_sections[section_id].get("update_count", 0)
            current_awake_count = updated_sections[section_id].get("awake_update_count", 0)
            
            if action == "replace":
                if not old_content:
                    # Replace entire content
                    final_content = new_content
                else:
                    # Replace pattern
                    if old_content in current_content:
                        final_content = current_content.replace(old_content, new_content)
                    else:
                        logger.warning(
                            f"Pattern '{old_content[:50]}...' not found in section '{section_id}'. "
                            f"Replacing entire content."
                        )
                        final_content = new_content
            
            elif action == "insert":
                if not old_content:
                    # Append at end
                    final_content = current_content + "\n" + new_content
                else:
                    # Insert after pattern
                    if old_content in current_content:
                        parts = current_content.split(old_content, 1)
                        final_content = parts[0] + old_content + "\n" + new_content + parts[1]
                    else:
                        logger.warning(
                            f"Pattern '{old_content[:50]}...' not found in section '{section_id}'. "
                            f"Appending at end."
                        )
                        final_content = current_content + "\n" + new_content
            
            else:
                raise ValueError(f"Invalid action '{action}'. Must be 'replace' or 'insert'.")
            
            # Update section
            updated_sections[section_id] = {
                "content": final_content,
                "update_count": current_update_count + 1,
                "awake_update_count": current_awake_count + 1,
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
    
    # Update database
    query = """
        UPDATE active_memory
        SET sections = $1, template_content = $2, updated_at = CURRENT_TIMESTAMP
        WHERE id = $3
        RETURNING id, external_id, title, template_content, sections, 
                  metadata, created_at, updated_at
    """
    
    async with self.postgres.connection() as conn:
        result = await conn.execute(
            query, 
            [updated_sections, updated_template, memory_id]
        )
        rows = result.result()
        
        if not rows:
            logger.warning(f"Active memory {memory_id} not found for upsert")
            return None
        
        memory = self._row_to_model(rows[0])
        logger.info(f"Upserted {len(section_updates)} sections in memory {memory_id}")
        return memory
```

#### 1.2.3 Update `reset_all_update_counts()` Method

**Changes:**
- Only reset `update_count` to 0
- Do NOT reset `awake_update_count` (it's permanent)
- Keep `last_updated` unchanged

**Checklist:**
- [ ] Update method to preserve `awake_update_count`
- [ ] Update method to preserve `last_updated`
- [ ] Update docstring

**Implementation:**
```python
async def reset_all_update_counts(self, memory_id: int) -> bool:
    """
    Reset update_count to 0 for all sections in a memory.
    
    Called after successful consolidation to shortterm memory.
    Does NOT reset awake_update_count (permanent counter).
    
    Args:
        memory_id: Memory ID
    
    Returns:
        True if successful, False if memory not found
    """
    current = await self.get_by_id(memory_id)
    if not current:
        logger.warning(f"Active memory {memory_id} not found")
        return False
    
    # Reset only update_count, preserve awake_update_count and last_updated
    reset_sections = {}
    for section_id, section_data in current.sections.items():
        reset_sections[section_id] = {
            "content": section_data.get("content", ""),
            "update_count": 0,  # RESET
            "awake_update_count": section_data.get("awake_update_count", 0),  # PRESERVE
            "last_updated": section_data.get("last_updated")  # PRESERVE
        }
    
    query = """
        UPDATE active_memory
        SET sections = $1, updated_at = CURRENT_TIMESTAMP
        WHERE id = $2
    """
    
    async with self.postgres.connection() as conn:
        await conn.execute(query, [reset_sections, memory_id])
        logger.info(f"Reset update_count for all sections in memory {memory_id}")
        return True
```

#### 1.2.4 Deprecate `update_section()` and `update_sections()`

**Checklist:**
- [ ] Mark `update_section()` as deprecated in docstring
- [ ] Mark `update_sections()` as deprecated in docstring
- [ ] Add deprecation warnings when called
- [ ] Point to new `upsert_sections()` method

---

### 1.3 Update Database Models

**File:** `agent_mem/database/models.py`

**Changes:**

```python
class ActiveMemory(BaseModel):
    """
    Active memory model representing working memory.
    
    Uses template-driven structure with sections:
    - template_content: JSON template with section definitions and defaults
    - sections: JSONB with section_id -> {content, update_count, awake_update_count, last_updated}
    """
    
    id: int
    external_id: str
    title: str
    template_content: Dict[str, Any]  # Changed from str to Dict
    sections: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime
    
    model_config = ConfigDict(from_attributes=True)
```

**Checklist:**
- [ ] Change `template_content` type from `str` to `Dict[str, Any]`
- [ ] Update docstring to reflect new structure
- [ ] Update section structure comment

---

## Phase 2: Core Service Layer

### 2.1 Update Memory Manager

**File:** `agent_mem/services/memory_manager.py`

#### 2.1.1 Update `create_active_memory()`

**Changes:**
- Accept `template_content` as dict
- Pass through to repository

**Checklist:**
- [ ] Change `template_content` parameter from `str` to `Dict[str, Any]`
- [ ] Update docstring
- [ ] Validate template structure (has "template" and "sections" keys)

**Implementation:**
```python
async def create_active_memory(
    self,
    external_id: str,
    title: str,
    template_content: Dict[str, Any],  # Changed from str
    initial_sections: Dict[str, Dict[str, Any]],
    metadata: Dict[str, Any],
) -> ActiveMemory:
    """
    Create a new active memory with template and sections.
    
    Args:
        external_id: Agent identifier
        title: Memory title
        template_content: JSON template dict with structure:
            {
                "template": {"id": "...", "name": "..."},
                "sections": [{"id": "...", "description": "..."}]
            }
        initial_sections: Initial sections {section_id: {content, update_count, ...}}
        metadata: Metadata dictionary
    
    Returns:
        Created ActiveMemory object
    """
    self._ensure_initialized()
    
    # Validate template structure
    if "template" not in template_content:
        raise ValueError("template_content must have 'template' key")
    if "sections" not in template_content:
        raise ValueError("template_content must have 'sections' key")
    
    logger.info(f"Creating active memory for {external_id}: {title}")
    
    memory = await self.active_repo.create(
        external_id=external_id,
        title=title,
        template_content=template_content,
        sections=initial_sections,
        metadata=metadata,
    )
    
    logger.info(f"Created active memory {memory.id} for {external_id}")
    return memory
```

#### 2.1.2 Replace `update_active_memory_sections()` with Upsert Logic

**Changes:**
- Use new `upsert_sections()` repository method
- Update consolidation threshold logic to account for new fields

**Checklist:**
- [ ] Call `active_repo.upsert_sections()` instead of `update_sections()`
- [ ] Update section_updates schema in docstring
- [ ] Handle new fields in threshold calculation

**Implementation:**
```python
async def update_active_memory_sections(
    self,
    external_id: str,
    memory_id: int,
    sections: List[Dict[str, Any]],  # Updated schema
) -> ActiveMemory:
    """
    Upsert multiple sections in an active memory (batch operation).
    
    Supports inserting new sections and updating existing ones.
    
    Args:
        external_id: Agent identifier
        memory_id: Memory ID
        sections: List of section updates:
            [
                {
                    "section_id": "progress",
                    "old_content": "# Old",  # Optional
                    "new_content": "# New",
                    "action": "replace"  # "replace" or "insert", default "replace"
                }
            ]
    
    Returns:
        Updated ActiveMemory object
    """
    self._ensure_initialized()
    
    logger.info(f"Upserting {len(sections)} sections in memory {memory_id} for {external_id}")
    
    # Upsert all sections in repository
    memory = await self.active_repo.upsert_sections(
        memory_id=memory_id,
        section_updates=sections,
    )
    
    if not memory:
        raise ValueError(f"Active memory {memory_id} not found")
    
    logger.info(f"Upserted {len(sections)} sections in memory {memory_id}")
    
    # Calculate threshold and check consolidation (same logic as before)
    num_sections = len(memory.sections)
    threshold = self.config.avg_section_update_count_for_consolidation * num_sections
    
    total_update_count = sum(
        section.get("update_count", 0) for section in memory.sections.values()
    )
    
    logger.debug(
        f"Total update count: {total_update_count}, "
        f"Threshold: {threshold} ({num_sections} sections)"
    )
    
    # Check if consolidation threshold is met
    if total_update_count >= threshold:
        logger.info(
            f"Total update count ({total_update_count}) >= threshold ({threshold}). "
            f"Triggering consolidation in background..."
        )
        
        task = asyncio.create_task(self._consolidate_with_lock(external_id, memory.id))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
    
    return memory
```

---

### 2.2 Update AgentMem Core Interface

**File:** `agent_mem/core.py`

#### 2.2.1 Update `create_active_memory()`

**Changes:**
- Accept `template_content` as dict or str (for backward compatibility)
- If string (YAML), parse to dict
- Pass dict to memory manager

**Checklist:**
- [ ] Support both `str` (YAML) and `Dict` for `template_content`
- [ ] Add YAML parsing logic if string provided
- [ ] Update docstring with new structure
- [ ] Add examples for both formats

**Implementation:**
```python
async def create_active_memory(
    self,
    external_id: str | UUID | int,
    title: str,
    template_content: str | Dict[str, Any],  # Support both
    initial_sections: Optional[Dict[str, Dict[str, Any]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> ActiveMemory:
    """
    Create a new active memory with template-driven structure.
    
    Args:
        external_id: Unique identifier for the agent (UUID, string, or int)
        title: Memory title
        template_content: Template defining section structure. Can be:
            - Dict (JSON): {"template": {...}, "sections": [...]}
            - Str (YAML): Parsed to dict automatically
        initial_sections: Optional initial sections that override template defaults
            {"section_id": {"content": "...", "update_count": 0, ...}}
        metadata: Optional metadata dictionary
    
    Returns:
        Created ActiveMemory object
    
    Example (JSON):
        ```python
        memory = await agent_mem.create_active_memory(
            external_id="agent-123",
            title="Task Memory",
            template_content={
                "template": {
                    "id": "task_memory_v1",
                    "name": "Task Memory"
                },
                "sections": [
                    {
                        "id": "current_task",
                        "description": "What is being worked on now"
                    }
                ]
            },
            initial_sections={
                "current_task": {"content": "# Task\nImplement feature"}
            }
        )
        ```
    
    Example (YAML - backward compatible):
        ```python
        memory = await agent_mem.create_active_memory(
            external_id="agent-123",
            title="Task Memory",
            template_content='''
template:
  id: "task_memory_v1"
  name: "Task Memory"
sections:
  - id: "current_task"
    title: "Current Task"
''',
            initial_sections={"current_task": {"content": "..."}}
        )
        ```
    """
    self._ensure_initialized()
    
    # Validate inputs
    if not title or not title.strip():
        raise ValueError("title cannot be empty")
    if not template_content:
        raise ValueError("template_content cannot be empty")
    
    external_id_str = str(external_id)
    
    # Parse template if string (YAML)
    if isinstance(template_content, str):
        import yaml
        try:
            template_dict = yaml.safe_load(template_content)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML template: {e}")
    else:
        template_dict = template_content
    
    logger.info(f"Creating active memory for {external_id_str}: {title}")
    return await self._memory_manager.create_active_memory(
        external_id=external_id_str,
        title=title,
        template_content=template_dict,
        initial_sections=initial_sections or {},
        metadata=metadata or {},
    )
```

#### 2.2.2 Update `update_active_memory_sections()`

**Changes:**
- Update docstring with new section update schema
- Update examples

**Checklist:**
- [ ] Update method signature to accept new section schema
- [ ] Update docstring with action types and behaviors
- [ ] Add examples for replace, insert, and upsert actions

**Implementation:**
```python
async def update_active_memory_sections(
    self,
    external_id: str | UUID | int,
    memory_id: int,
    sections: List[Dict[str, Any]],
) -> ActiveMemory:
    """
    Upsert multiple sections in an active memory (batch operation).
    
    Supports:
    - Creating new sections (automatically added to template)
    - Updating existing sections
    - Content replacement with pattern matching
    - Content insertion/appending
    
    Args:
        external_id: Unique identifier for the agent
        memory_id: ID of the memory to update
        sections: List of section updates:
            [
                {
                    "section_id": "progress",
                    "old_content": "# Old header",  # Optional: pattern to find
                    "new_content": "# New content",
                    "action": "replace"  # "replace" or "insert", default "replace"
                }
            ]
    
    Returns:
        Updated ActiveMemory object
    
    Action Behaviors:
        **replace**:
        - If old_content is null/empty: Replaces entire section content
        - If old_content is provided: Replaces that substring with new_content
        
        **insert**:
        - If old_content is null/empty: Appends new_content at end
        - If old_content is provided: Inserts new_content right after old_content
        
        **New Section** (section doesn't exist):
        - Creates section with new_content
        - Adds section definition to template_content
    
    Examples:
        ```python
        # Replace entire section
        await agent_mem.update_active_memory_sections(
            external_id="agent-123",
            memory_id=1,
            sections=[
                {
                    "section_id": "progress",
                    "new_content": "# Progress\n- All done!",
                    "action": "replace"
                }
            ]
        )
        
        # Replace specific part
        await agent_mem.update_active_memory_sections(
            external_id="agent-123",
            memory_id=1,
            sections=[
                {
                    "section_id": "progress",
                    "old_content": "- Step 1: In progress",
                    "new_content": "- Step 1: Complete",
                    "action": "replace"
                }
            ]
        )
        
        # Append new content
        await agent_mem.update_active_memory_sections(
            external_id="agent-123",
            memory_id=1,
            sections=[
                {
                    "section_id": "progress",
                    "new_content": "\n- Step 4: Started",
                    "action": "insert"
                }
            ]
        )
        
        # Insert new section
        await agent_mem.update_active_memory_sections(
            external_id="agent-123",
            memory_id=1,
            sections=[
                {
                    "section_id": "new_section",
                    "new_content": "# New Section\nContent...",
                    "action": "replace"
                }
            ]
        )
        ```
    """
    self._ensure_initialized()
    external_id_str = str(external_id)
    
    logger.info(
        f"Upserting {len(sections)} sections in memory {memory_id} for {external_id_str}"
    )
    return await self._memory_manager.update_active_memory_sections(
        external_id=external_id_str,
        memory_id=memory_id,
        sections=sections,
    )
```

---

## Phase 3: MCP Server Updates

### 3.1 Update Existing MCP Tool Schemas

**File:** `agent_mem_mcp/schemas.py`

#### 3.1.1 Update `UPDATE_MEMORY_SECTIONS_INPUT_SCHEMA`

**Changes:**
- Add `old_content` field (optional)
- Add `action` field (optional, default "replace")
- Update descriptions

**Checklist:**
- [ ] Add `old_content` to section item schema
- [ ] Add `action` enum field to section item schema
- [ ] Update descriptions

**Implementation:**
```python
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
```

---

### 3.2 Update MCP Server Handlers

**File:** `agent_mem_mcp/server.py`

#### 3.2.1 Update `_handle_get_active_memories()`

**Changes:**
- Format new section fields in response
- Show `awake_update_count` and `last_updated`

**Checklist:**
- [ ] Include `awake_update_count` in response
- [ ] Include `last_updated` in response
- [ ] Update response example

#### 3.2.2 Update `_handle_update_memory_sections()`

**Changes:**
- Handle new section schema
- Show awake_update_count in response

**Checklist:**
- [ ] Pass through new `old_content` and `action` fields
- [ ] Show `awake_update_count` changes in response
- [ ] Update consolidation threshold info

---

## Phase 4: New MCP Tools

### 4.1 Add Create Memory Tool

**File:** `agent_mem_mcp/schemas.py`

**Checklist:**
- [ ] Create `CREATE_ACTIVE_MEMORY_INPUT_SCHEMA`

**Implementation:**
```python
# Tool: create_active_memory
CREATE_ACTIVE_MEMORY_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "external_id": {
            "type": "string",
            "description": "Unique identifier for the agent",
        },
        "title": {
            "type": "string",
            "description": "Title for the memory",
        },
        "template_content": {
            "type": "object",
            "description": (
                "JSON template defining memory structure:\n"
                "{\n"
                '  "template": {"id": "...", "name": "..."},\n'
                '  "sections": [\n'
                '    {"id": "section_id", "description": "Default content..."}\n'
                "  ]\n"
                "}"
            ),
            "required": ["template", "sections"],
        },
        "initial_sections": {
            "type": "object",
            "description": (
                "Optional: Initial section content overriding template defaults. "
                "Format: {section_id: {content: '...', update_count: 0, ...}}"
            ),
        },
        "metadata": {
            "type": "object",
            "description": "Optional: Additional metadata for the memory",
        },
    },
    "required": ["external_id", "title", "template_content"],
}
```

---

### 4.2 Add Delete Memory Tool

**File:** `agent_mem_mcp/schemas.py`

**Checklist:**
- [ ] Create `DELETE_ACTIVE_MEMORY_INPUT_SCHEMA`

**Implementation:**
```python
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
```

---

### 4.3 Update MCP Server Tool List

**File:** `agent_mem_mcp/server.py`

#### 4.3.1 Add Tools to `handle_list_tools()`

**Checklist:**
- [ ] Add `create_active_memory` tool definition
- [ ] Add `delete_active_memory` tool definition

**Implementation:**
```python
@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available MCP tools."""
    return [
        types.Tool(
            name="get_active_memories",
            description="Get all active memories for an agent",
            inputSchema=GET_ACTIVE_MEMORIES_INPUT_SCHEMA,
        ),
        types.Tool(
            name="create_active_memory",
            description=(
                "Create a new active memory for an agent with template-driven structure. "
                "Supports JSON templates with default section values."
            ),
            inputSchema=CREATE_ACTIVE_MEMORY_INPUT_SCHEMA,
        ),
        types.Tool(
            name="update_memory_sections",
            description=(
                "Upsert (insert or update) multiple sections in an active memory. "
                "Supports creating new sections, replacing content, and inserting content."
            ),
            inputSchema=UPDATE_MEMORY_SECTIONS_INPUT_SCHEMA,
        ),
        types.Tool(
            name="delete_active_memory",
            description="Delete an active memory for an agent",
            inputSchema=DELETE_ACTIVE_MEMORY_INPUT_SCHEMA,
        ),
        types.Tool(
            name="search_memories",
            description="Search across active, shortterm, and longterm memories",
            inputSchema=SEARCH_MEMORIES_INPUT_SCHEMA,
        ),
    ]
```

#### 4.3.2 Add Tool Handlers

**Checklist:**
- [ ] Implement `_handle_create_active_memory()`
- [ ] Implement `_handle_delete_active_memory()`
- [ ] Update `handle_call_tool()` routing

**Implementation:**
```python
@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict[str, Any]
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """Handle MCP tool calls."""
    # Get agent_mem from server state
    agent_mem = server._server_state.get("agent_mem")
    
    if name == "get_active_memories":
        return await _handle_get_active_memories(agent_mem, arguments)
    elif name == "create_active_memory":
        return await _handle_create_active_memory(agent_mem, arguments)
    elif name == "update_memory_sections":
        return await _handle_update_memory_sections(agent_mem, arguments)
    elif name == "delete_active_memory":
        return await _handle_delete_active_memory(agent_mem, arguments)
    elif name == "search_memories":
        return await _handle_search_memories(agent_mem, arguments)
    else:
        raise ValueError(f"Unknown tool: {name}")


async def _handle_create_active_memory(
    agent_mem: AgentMem, arguments: dict[str, Any]
) -> list[types.TextContent]:
    """Handle create_active_memory tool call."""
    external_id = arguments["external_id"]
    title = arguments["title"]
    template_content = arguments["template_content"]
    initial_sections = arguments.get("initial_sections", {})
    metadata = arguments.get("metadata", {})
    
    # Validate inputs
    if not external_id or not external_id.strip():
        return [types.TextContent(type="text", text=json.dumps({
            "error": "external_id cannot be empty"
        }, indent=2))]
    
    if not title or not title.strip():
        return [types.TextContent(type="text", text=json.dumps({
            "error": "title cannot be empty"
        }, indent=2))]
    
    # Validate template structure
    if not isinstance(template_content, dict):
        return [types.TextContent(type="text", text=json.dumps({
            "error": "template_content must be a JSON object"
        }, indent=2))]
    
    if "template" not in template_content or "sections" not in template_content:
        return [types.TextContent(type="text", text=json.dumps({
            "error": "template_content must have 'template' and 'sections' keys"
        }, indent=2))]
    
    # Create memory
    try:
        memory = await agent_mem.create_active_memory(
            external_id=external_id,
            title=title,
            template_content=template_content,
            initial_sections=initial_sections,
            metadata=metadata,
        )
        
        # Format response
        response = {
            "success": True,
            "memory": {
                "id": memory.id,
                "external_id": memory.external_id,
                "title": memory.title,
                "template": memory.template_content.get("template", {}),
                "sections": memory.sections,
                "metadata": memory.metadata,
                "created_at": memory.created_at.isoformat(),
                "updated_at": memory.updated_at.isoformat(),
            },
            "message": f"Successfully created memory with {len(memory.sections)} sections",
        }
        
        logger.info(
            f"Created memory {memory.id} for {external_id} with {len(memory.sections)} sections"
        )
        
        return [types.TextContent(type="text", text=json.dumps(response, indent=2))]
        
    except Exception as e:
        logger.error(f"Error creating memory: {e}", exc_info=True)
        return [types.TextContent(type="text", text=json.dumps({
            "error": f"Failed to create memory: {str(e)}"
        }, indent=2))]


async def _handle_delete_active_memory(
    agent_mem: AgentMem, arguments: dict[str, Any]
) -> list[types.TextContent]:
    """Handle delete_active_memory tool call."""
    external_id = arguments["external_id"]
    memory_id = arguments["memory_id"]
    
    # Validate inputs
    if not external_id or not external_id.strip():
        return [types.TextContent(type="text", text=json.dumps({
            "error": "external_id cannot be empty"
        }, indent=2))]
    
    # Delete memory
    try:
        success = await agent_mem.delete_active_memory(
            external_id=external_id,
            memory_id=memory_id,
        )
        
        if success:
            response = {
                "success": True,
                "message": f"Successfully deleted memory {memory_id}",
            }
            logger.info(f"Deleted memory {memory_id} for {external_id}")
        else:
            response = {
                "success": False,
                "error": f"Memory {memory_id} not found or does not belong to agent {external_id}",
            }
            logger.warning(f"Failed to delete memory {memory_id} for {external_id}")
        
        return [types.TextContent(type="text", text=json.dumps(response, indent=2))]
        
    except Exception as e:
        logger.error(f"Error deleting memory: {e}", exc_info=True)
        return [types.TextContent(type="text", text=json.dumps({
            "error": f"Failed to delete memory: {str(e)}"
        }, indent=2))]
```

---

## Phase 5: Streamlit UI Updates

### 5.1 Update Memory Service

**File:** `streamlit_app/services/memory_service.py`

#### 5.1.1 Update `create_active_memory()`

**Changes:**
- Handle dict template_content

**Checklist:**
- [ ] Support both YAML string and dict input
- [ ] Parse YAML if string
- [ ] Pass dict to AgentMem

#### 5.1.2 Update `format_memory_for_display()`

**Changes:**
- Show new section fields

**Checklist:**
- [ ] Display `awake_update_count`
- [ ] Display `last_updated`
- [ ] Parse template_content as dict

---

### 5.2 Update Template Service

**File:** `streamlit_app/services/template_service.py`

**Changes:**
- Load templates as JSON instead of YAML (if stored as JSON)
- Or convert YAML to JSON when loading

**Checklist:**
- [ ] Update `load_all_templates()` to return dict format
- [ ] Convert YAML files to dict on load
- [ ] Ensure structure matches new template_content format

---

### 5.3 Update Create Memory Page

**File:** `streamlit_app/pages/2_Create_Memory.py`

**Changes:**
- Display new section fields
- Pass dict template to API

**Checklist:**
- [ ] Convert template YAML to dict before creating memory
- [ ] Show section structure preview
- [ ] Handle initial_sections properly

---

### 5.4 Update Edit Memory Page

**File:** `streamlit_app/pages/3_Edit_Memory.py`

**Changes:**
- Show new section fields
- Support upsert actions

**Checklist:**
- [ ] Display `awake_update_count` (read-only)
- [ ] Display `last_updated` (read-only)
- [ ] Add action selector (replace/insert)
- [ ] Add old_content input for pattern matching

---

## Phase 6: Tests and Documentation

### 6.1 Update Unit Tests

**Files:**
- `tests/test_core.py`
- `tests/test_memory_manager.py`
- `tests/test_active_memory.py` (repository tests)

**Checklist:**
- [ ] Update `create_active_memory` tests with dict template
- [ ] Add tests for upsert logic
- [ ] Test new section fields (awake_update_count, last_updated)
- [ ] Test section insertion
- [ ] Test content replacement with patterns
- [ ] Test content insertion/appending
- [ ] Test reset_all_update_counts preserves awake_update_count

---

### 6.2 Update Integration Tests

**Files:**
- `tests/test_integration.py`
- `tests/test_streamlit_integration.py`
- `tests/test_mcp_batch_update.py`

**Checklist:**
- [ ] Test end-to-end create with JSON template
- [ ] Test upsert workflow
- [ ] Test MCP create_active_memory tool
- [ ] Test MCP delete_active_memory tool
- [ ] Test consolidation with new fields

---

### 6.3 Update MCP Tests

**Files:**
- `agent_mem_mcp/test_server.py`

**Checklist:**
- [ ] Add test for create_active_memory tool
- [ ] Add test for delete_active_memory tool
- [ ] Update update_memory_sections tests with new schema

---

### 6.4 Update Documentation

**Files:**
- `README.md`
- `docs/GETTING_STARTED.md`
- `docs/guide/overview.md`
- `docs/guide/memory-tiers.md`
- `docs/ref/memory-architecture.md`

**Checklist:**
- [ ] Update README with new template format
- [ ] Update examples with dict templates
- [ ] Document upsert logic and actions
- [ ] Document new section fields
- [ ] Update MCP tool list
- [ ] Add migration guide for YAML → JSON

---

### 6.5 Update Examples

**Files:**
- `examples/basic_usage.py`
- `examples/batch_update_example.py`

**Checklist:**
- [ ] Update examples with JSON templates
- [ ] Add upsert examples
- [ ] Show new section fields

---

## Migration Strategy

### Option 1: Automatic Migration (Recommended)

Add migration logic to `ActiveMemoryRepository` that automatically converts YAML `template_content` to JSON on first read:

```python
def _row_to_model(self, row) -> ActiveMemory:
    """Convert database row to ActiveMemory model."""
    # ... existing code ...
    
    # Migrate YAML template_content to JSON if needed
    template_content = row[3]
    if isinstance(template_content, str):
        # Old format (YAML string) - convert to JSON
        import yaml
        try:
            template_dict = yaml.safe_load(template_content)
            # Ensure proper structure
            if "sections" in template_dict:
                for section in template_dict["sections"]:
                    # Convert old format to new format
                    if "title" in section:
                        del section["title"]  # Remove title
                    if "content" in section:
                        section["description"] = section["content"]  # content becomes description
                        del section["content"]
                    section.setdefault("description", "")  # Ensure description exists
            
            # Update database with JSON format (async migration)
            asyncio.create_task(self._migrate_template(row[0], template_dict))
            template_content = template_dict
        except Exception as e:
            logger.error(f"Failed to migrate template for memory {row[0]}: {e}")
    
    # Migrate sections to add new fields if missing
    sections = row[4]
    for section_id, section_data in sections.items():
        section_data.setdefault("awake_update_count", section_data.get("update_count", 0))
        section_data.setdefault("last_updated", None)
    
    return ActiveMemory(...)
```

### Option 2: Manual Migration Script

Create a migration script that updates all existing records:

```python
# scripts/migrate_to_json_templates.py
async def migrate_templates():
    """Migrate all YAML templates to JSON format."""
    async with postgres_manager.connection() as conn:
        # Get all memories with TEXT template_content
        result = await conn.execute("""
            SELECT id, template_content, sections
            FROM active_memory
        """)
        
        for row in result.result():
            memory_id = row[0]
            template_content = row[1]
            sections = row[2]
            
            # Convert YAML to JSON
            if isinstance(template_content, str):
                template_dict = yaml.safe_load(template_content)
                # Convert old format to new format
                if "sections" in template_dict:
                    for section in template_dict["sections"]:
                        if "title" in section:
                            del section["title"]  # Remove title
                        if "content" in section:
                            section["description"] = section["content"]  # content becomes description
                            del section["content"]
                        section.setdefault("description", "")  # Ensure description exists
            
            # Migrate sections
            for section_id, section_data in sections.items():
                section_data["awake_update_count"] = section_data.get("update_count", 0)
                section_data["last_updated"] = None
            
            # Update database
            await conn.execute("""
                UPDATE active_memory
                SET template_content = $1, sections = $2
                WHERE id = $3
            """, [template_dict, sections, memory_id])
```

---

## Files Impact Matrix

| File | Changes | Priority | Complexity |
|------|---------|----------|------------|
| `agent_mem/sql/schema.sql` | Update column types, add indexes | HIGH | LOW |
| `agent_mem/database/models.py` | Change template_content type | HIGH | LOW |
| `agent_mem/database/repositories/active_memory.py` | Major refactor: create, upsert_sections, reset | HIGH | HIGH |
| `agent_mem/services/memory_manager.py` | Update create, update methods | HIGH | MEDIUM |
| `agent_mem/core.py` | Update create, update signatures | HIGH | MEDIUM |
| `agent_mem_mcp/schemas.py` | Add/update schemas | HIGH | LOW |
| `agent_mem_mcp/server.py` | Add new tools, update handlers | HIGH | MEDIUM |
| `streamlit_app/services/memory_service.py` | Update template handling | MEDIUM | LOW |
| `streamlit_app/services/template_service.py` | Convert templates to dict | MEDIUM | LOW |
| `streamlit_app/pages/2_Create_Memory.py` | Update UI | MEDIUM | MEDIUM |
| `streamlit_app/pages/3_Edit_Memory.py` | Add upsert UI | MEDIUM | MEDIUM |
| `tests/test_core.py` | Update tests | HIGH | MEDIUM |
| `tests/test_memory_manager.py` | Update tests | HIGH | MEDIUM |
| `tests/test_active_memory.py` | Add upsert tests | HIGH | MEDIUM |
| `tests/test_integration.py` | Update integration tests | HIGH | MEDIUM |
| `agent_mem_mcp/test_server.py` | Add new tool tests | HIGH | LOW |
| `README.md` | Update docs | MEDIUM | LOW |
| `docs/**/*.md` | Update all docs | MEDIUM | LOW |
| `examples/*.py` | Update examples | LOW | LOW |

---

## Implementation Phases Timeline

### Phase 1: Schema and Database Layer (2-3 days)
- Day 1: Schema updates, model changes
- Day 2-3: Repository refactor (create, upsert_sections, reset)

### Phase 2: Core Service Layer (1-2 days)
- Day 1: Memory manager updates
- Day 2: Core interface updates

### Phase 3: MCP Server Updates (1 day)
- Update existing tool schemas and handlers

### Phase 4: New MCP Tools (1 day)
- Implement create and delete tools

### Phase 5: Streamlit UI Updates (2 days)
- Day 1: Service updates
- Day 2: Page updates (create, edit)

### Phase 6: Tests and Documentation (2-3 days)
- Day 1-2: Update and write tests
- Day 3: Update documentation

**Total Estimated Time: 9-12 days**

---

## Rollback Strategy

If issues arise during implementation:

1. **Database Schema**: Keep both `template_content` TEXT and JSONB columns temporarily
2. **Backward Compatibility**: Support both old and new formats in code
3. **Feature Flags**: Add config flag to enable/disable new features
4. **Migration Toggle**: Allow reverting to old YAML format

---

## Success Criteria

- [ ] All existing tests pass
- [ ] New tests for upsert logic pass
- [ ] MCP tools work correctly
- [ ] Streamlit UI displays new fields
- [ ] Documentation updated
- [ ] Examples work with new format
- [ ] Migration script tested on sample data
- [ ] Performance benchmarks meet requirements

---

## Notes

- **Breaking Change**: Template format changes from YAML to JSON
- **Migration Required**: Existing memories need schema updates
- **Backward Compatibility**: Core API should support both formats initially
- **Testing**: Extensive testing required for upsert logic edge cases
- **Documentation**: Clear migration guide needed for users

---

## Related Documents

- [ARCHITECTURE.md](ARCHITECTURE.md)
- [DEVELOPMENT.md](DEVELOPMENT.md)
- [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)
- [MCP_SERVER_IMPLEMENTATION_PLAN.md](MCP_SERVER_IMPLEMENTATION_PLAN.md)
