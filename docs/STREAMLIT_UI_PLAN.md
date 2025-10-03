# Streamlit UI Plan

**Feature Branch**: `feature/streamlit-ui`  
**Created**: October 3, 2025  
**Status**: Planning Phase

## Overview

Add a Streamlit web UI to `agent_mem` package for managing active memories with pre-built BMAD templates. The UI will provide a user-friendly interface to view templates, create/manage active memories, and interact with the agent memory system.

---

## Goals

1. **Template Discovery**: Browse and preview 62 pre-built BMAD templates across 10 agent types
2. **Memory Creation**: Create active memories using pre-built templates or custom YAML
3. **Memory Management**: View, update, and delete active memories for agents
4. **User-Friendly**: Intuitive UI for non-technical users to manage agent memories
5. **Integration**: Seamless integration with existing `AgentMem` API

---

## Architecture

### Components

```
streamlit_app/
├── app.py                      # Main Streamlit application
├── pages/
│   ├── 1_📚_Browse_Templates.py   # Browse pre-built templates
│   ├── 2_➕_Create_Memory.py      # Create active memory
│   ├── 3_📋_View_Memories.py      # View agent memories
│   ├── 4_✏️_Update_Memory.py      # Update memory sections
│   └── 5_🗑️_Delete_Memory.py     # Delete memories
├── components/
│   ├── __init__.py
│   ├── template_viewer.py      # Template preview component
│   ├── yaml_editor.py          # YAML editor with validation
│   ├── memory_card.py          # Memory display card
│   └── section_editor.py       # Section content editor
├── services/
│   ├── __init__.py
│   ├── template_service.py     # Template loading/parsing
│   └── memory_service.py       # AgentMem wrapper
├── utils/
│   ├── __init__.py
│   ├── yaml_validator.py       # YAML validation
│   ├── template_loader.py      # Load templates from filesystem
│   └── formatters.py           # Display formatters
├── config.py                   # Streamlit app configuration
└── requirements.txt            # Streamlit dependencies
```

### Template Service

Manages pre-built template discovery and parsing:
- Scan `prebuilt-memory-tmpl/bmad/` directory
- Parse YAML templates
- Categorize by agent type
- Provide search/filter functionality

### Memory Service

Wrapper around `AgentMem` for UI operations:
- Initialize AgentMem connection
- CRUD operations with error handling
- State management for Streamlit
- Session caching

---

## Features Breakdown

### 1. Browse Templates (Page 1)

**UI Components:**
- Agent type selector (dropdown with 10 options)
- Template list with cards showing:
  - Template ID
  - Template name
  - Number of sections
  - Usage type
  - Priority
- Template preview modal:
  - Full YAML content
  - Section details with descriptions
  - Example content from descriptions
- Search/filter bar

**Functionality:**
- Load all templates from `prebuilt-memory-tmpl/bmad/`
- Display templates grouped by agent
- Preview template structure
- Copy template ID for use

**Technical Requirements:**
- Template caching (session state)
- YAML parsing with error handling
- Responsive card layout

---

### 2. Create Memory (Page 2)

**UI Components:**
- External ID input (text/UUID/int)
- Creation mode selector:
  - **Pre-built Template**: Browse and select
  - **Custom YAML**: Paste/upload YAML
- Template selector (when using pre-built)
- YAML editor (when using custom)
- Auto-populated fields based on template:
  - **Title**: Text input (from template name or custom)
  - **Initial Sections**: Expandable section editors
    - Section ID (from template)
    - Content (markdown editor)
    - Update count (default: 0)
- Metadata editor (JSON)
- Create button

**Functionality:**
- Parse selected template
- Auto-populate title from template name
- Generate initial section forms from template
- Validate YAML structure
- Call `create_active_memory()` API
- Show success/error messages
- Redirect to view page after creation

**Technical Requirements:**
- YAML validation before submission
- Template structure validation
- Error handling and user feedback
- Session state management
- Markdown preview for sections

---

### 3. View Memories (Page 3)

**UI Components:**
- External ID input
- Load button
- Memory list:
  - Memory cards with:
    - Title
    - Template name/ID
    - Created/updated timestamps
    - Number of sections
    - Metadata badges
  - Expand to view sections:
    - Section ID + title
    - Content preview
    - Update count
    - Last updated
- Empty state message
- Refresh button

**Functionality:**
- Call `get_active_memories(external_id)` API
- Display all memories for agent
- Expandable section details
- Navigate to update/delete pages

**Technical Requirements:**
- Efficient memory loading
- Pagination for many memories
- Section content truncation with "Show more"
- Link to update/delete actions

---

### 4. Update Memory (Page 4)

**UI Components:**
- External ID input
- Memory selector (dropdown or search)
- Template display (read-only)
- Section selector
- Section content editor:
  - Markdown editor
  - Preview pane
  - Character count
  - Update count display
- Metadata editor
- Update button
- Cancel button

**Functionality:**
- Load memory by ID
- Display current sections
- Edit section content
- Call `update_active_memory_section()` API
- Auto-increment update_count
- Show consolidation warnings (if threshold reached)
- Success/error feedback

**Technical Requirements:**
- Real-time markdown preview
- Dirty state tracking
- Unsaved changes warning
- Section-level updates (not full memory)
- Update count display and warnings

---

### 5. Delete Memory (Page 5)

**UI Components:**
- External ID input
- Memory selector (dropdown with preview)
- Memory details display (read-only)
- Confirmation dialog:
  - "Are you sure?" message
  - Type memory title to confirm
  - Warning about irreversible action
- Delete button
- Cancel button

**Functionality:**
- Load memory for deletion
- Display full memory details
- Require explicit confirmation
- Call delete API (to be added to AgentMem)
- Success message and redirect

**Technical Requirements:**
- Confirmation safeguards
- Soft delete option (future)
- Cascade delete handling
- Error handling

---

## API Extensions Needed

### New Methods for `AgentMem`

Add these methods to support delete operations:

```python
async def delete_active_memory(
    self,
    external_id: str | UUID | int,
    memory_id: int,
) -> bool:
    """
    Delete an active memory.
    
    Args:
        external_id: Agent identifier
        memory_id: Memory ID to delete
        
    Returns:
        True if deleted, False if not found
    """
    pass

async def get_active_memory_by_id(
    self,
    external_id: str | UUID | int,
    memory_id: int,
) -> Optional[ActiveMemory]:
    """
    Get a specific active memory by ID.
    
    Args:
        external_id: Agent identifier
        memory_id: Memory ID
        
    Returns:
        ActiveMemory object or None
    """
    pass
```

---

## Implementation Checklist

### Phase 1: Foundation (Week 1)

**Setup & Structure**
- [ ] Create `streamlit_app/` directory structure
- [ ] Add Streamlit dependencies to `requirements.txt`
- [ ] Create `streamlit_app/requirements.txt`
- [ ] Set up base `app.py` with navigation
- [ ] Create `config.py` for Streamlit settings
- [ ] Add `.streamlit/config.toml` for theme

**Template Service**
- [ ] Implement `template_loader.py`
  - [ ] Scan `prebuilt-memory-tmpl/bmad/` directory
  - [ ] Parse YAML files
  - [ ] Handle errors (invalid YAML, missing files)
- [ ] Implement `template_service.py`
  - [ ] Load all templates
  - [ ] Categorize by agent type
  - [ ] Search/filter functionality
  - [ ] Cache templates in session state
- [ ] Add unit tests for template loading

**Memory Service**
- [ ] Implement `memory_service.py`
  - [ ] Initialize `AgentMem` connection
  - [ ] Wrapper methods for CRUD operations
  - [ ] Error handling and logging
  - [ ] Session state management
- [ ] Add connection pooling/caching

### Phase 2: Browse Templates Page (Week 1-2)

**UI Components**
- [ ] Create `pages/1_📚_Browse_Templates.py`
- [ ] Implement agent type selector
- [ ] Create template card component (`components/template_viewer.py`)
  - [ ] Display template metadata
  - [ ] Section list with descriptions
  - [ ] Usage and priority badges
- [ ] Add template preview modal
  - [ ] YAML syntax highlighting
  - [ ] Section details view
  - [ ] Copy to clipboard functionality
- [ ] Implement search/filter bar
- [ ] Add responsive layout (columns/grid)

**Functionality**
- [ ] Load templates on page load
- [ ] Filter by agent type
- [ ] Search by template name/ID
- [ ] Preview full template
- [ ] Copy template ID
- [ ] Error handling for missing templates

### Phase 3: Create Memory Page (Week 2)

**UI Components**
- [ ] Create `pages/2_➕_Create_Memory.py`
- [ ] External ID input with validation
- [ ] Creation mode toggle (pre-built vs custom)
- [ ] Template selector (dropdown with preview)
- [ ] Custom YAML editor (`components/yaml_editor.py`)
  - [ ] Syntax highlighting
  - [ ] Real-time validation
  - [ ] Error messages
- [ ] Auto-populated title field
- [ ] Section editors (`components/section_editor.py`)
  - [ ] Generate from template
  - [ ] Markdown editor
  - [ ] Content preview
  - [ ] Update count input
- [ ] Metadata JSON editor
- [ ] Create button with loading state

**Functionality**
- [ ] Parse selected template
- [ ] Auto-generate section forms
- [ ] Validate YAML structure
- [ ] Validate section IDs match template
- [ ] Call `create_active_memory()` API
- [ ] Handle creation errors
- [ ] Success message and redirect
- [ ] Form reset on success

**Validation**
- [ ] Implement `yaml_validator.py`
  - [ ] YAML syntax validation
  - [ ] Template structure validation
  - [ ] Required fields check
- [ ] Client-side validation
- [ ] Server-side validation feedback

### Phase 4: View Memories Page (Week 2-3)

**UI Components**
- [ ] Create `pages/3_📋_View_Memories.py`
- [ ] External ID input with session persistence
- [ ] Load memories button
- [ ] Memory card component (`components/memory_card.py`)
  - [ ] Title and metadata
  - [ ] Template info
  - [ ] Timestamps
  - [ ] Section count
  - [ ] Expand/collapse sections
- [ ] Section detail view
  - [ ] Section ID and title
  - [ ] Content (truncated with "Show more")
  - [ ] Update count badge
  - [ ] Last updated timestamp
- [ ] Empty state message
- [ ] Refresh button
- [ ] Pagination controls

**Functionality**
- [ ] Call `get_active_memories()` API
- [ ] Display memory list
- [ ] Expand/collapse sections
- [ ] Content truncation and expansion
- [ ] Navigate to update/delete pages
- [ ] Auto-refresh option
- [ ] Error handling for API failures

### Phase 5: Update Memory Page (Week 3)

**UI Components**
- [ ] Create `pages/4_✏️_Update_Memory.py`
- [ ] External ID input
- [ ] Memory selector (dropdown)
- [ ] Template display (read-only card)
- [ ] Section selector (radio/dropdown)
- [ ] Section content editor
  - [ ] Current content display
  - [ ] Markdown editor
  - [ ] Live preview
  - [ ] Character count
  - [ ] Update count display
- [ ] Metadata editor (optional)
- [ ] Update/Cancel buttons
- [ ] Unsaved changes warning

**Functionality**
- [ ] Load memory details
- [ ] Display current section content
- [ ] Edit section content
- [ ] Call `update_active_memory_section()` API
- [ ] Show consolidation warnings
- [ ] Dirty state tracking
- [ ] Confirm before navigation if unsaved changes
- [ ] Success/error feedback
- [ ] Redirect after update

### Phase 6: Delete Memory Page (Week 3)

**UI Components**
- [ ] Create `pages/5_🗑️_Delete_Memory.py`
- [ ] External ID input
- [ ] Memory selector with preview
- [ ] Memory details card (read-only)
  - [ ] Full memory info
  - [ ] All sections preview
  - [ ] Warning badge
- [ ] Confirmation dialog
  - [ ] "Type title to confirm" input
  - [ ] Warning message
  - [ ] Checkbox: "I understand this is irreversible"
- [ ] Delete/Cancel buttons

**Functionality**
- [ ] Load memory for deletion
- [ ] Display full details
- [ ] Require explicit confirmation
- [ ] Call delete API
- [ ] Success message
- [ ] Redirect to view page
- [ ] Error handling

**API Extension**
- [ ] Add `delete_active_memory()` to `AgentMem`
- [ ] Add `delete_active_memory()` to `MemoryManager`
- [ ] Add `delete()` to `ActiveMemoryRepository`
- [ ] Add database cascade delete
- [ ] Add tests for delete functionality

### Phase 7: Polish & Testing (Week 4)

**UI Polish**
- [ ] Add consistent styling across all pages
- [ ] Implement custom theme in `.streamlit/config.toml`
- [ ] Add loading spinners for async operations
- [ ] Add toast notifications for actions
- [ ] Add keyboard shortcuts
- [ ] Responsive design testing
- [ ] Accessibility improvements (ARIA labels)

**Error Handling**
- [ ] Graceful error messages
- [ ] Connection error handling
- [ ] Database error handling
- [ ] Invalid input handling
- [ ] Session expiry handling
- [ ] Retry mechanisms

**Documentation**
- [ ] Add `STREAMLIT_UI.md` user guide
- [ ] Update main `README.md` with UI section
- [ ] Add screenshots to documentation
- [ ] Create video demo (optional)
- [ ] Add inline help tooltips in UI
- [ ] Add FAQ section

**Testing**
- [ ] Unit tests for template service
- [ ] Unit tests for memory service
- [ ] Integration tests for UI workflows
- [ ] Manual testing checklist
- [ ] Cross-browser testing
- [ ] Performance testing (many templates/memories)
- [ ] Error scenario testing

### Phase 8: Deployment & Documentation (Week 4)

**Deployment**
- [ ] Add Streamlit run script
- [ ] Docker support for UI (optional)
- [ ] Update `docker-compose.yml` to include UI
- [ ] Add environment variable configuration
- [ ] Add production deployment guide

**Documentation**
- [ ] Update `docs/INDEX.md`
- [ ] Update `docs/GETTING_STARTED.md`
- [ ] Create `docs/UI_USER_GUIDE.md`
- [ ] Add troubleshooting section
- [ ] Update `README.md` with UI screenshots

**Final Review**
- [ ] Code review
- [ ] Security review (input validation)
- [ ] Performance review
- [ ] Accessibility review
- [ ] Documentation review
- [ ] Merge to main branch

---

## Technical Requirements

### Dependencies

Add to `streamlit_app/requirements.txt`:

```txt
streamlit>=1.28.0
pyyaml>=6.0.1
markdown>=3.5.0
pygments>=2.16.0
streamlit-ace>=0.1.1
streamlit-aggrid>=0.3.4
```

### Streamlit Configuration

`.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#4A90E2"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[server]
maxUploadSize = 5
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false
```

### Session State Management

Use Streamlit session state for:
- Loaded templates (cache)
- Current agent ID
- Selected memory
- Form data
- Navigation state

### Error Handling Strategy

1. **User-Friendly Messages**: Never show raw stack traces
2. **Retry Logic**: Auto-retry on connection errors
3. **Fallbacks**: Graceful degradation if services unavailable
4. **Logging**: Log all errors for debugging
5. **Validation**: Client-side + server-side validation

---

## UI/UX Design Principles

### Layout
- **Sidebar**: Navigation + agent ID persistence
- **Main Area**: Page content
- **Top Bar**: Page title + actions
- **Footer**: Status messages + help

### Color Scheme
- **Primary**: Blue (#4A90E2) - actions, links
- **Success**: Green (#10B981) - successful operations
- **Warning**: Yellow (#F59E0B) - consolidation warnings
- **Danger**: Red (#EF4444) - delete, errors
- **Neutral**: Gray - borders, backgrounds

### Typography
- **Headings**: Bold, clear hierarchy
- **Body**: Readable font size (16px)
- **Code**: Monospace for YAML/IDs
- **Labels**: Descriptive, concise

### Interactions
- **Loading States**: Spinners for async operations
- **Feedback**: Toast notifications for actions
- **Confirmation**: Dialogs for destructive actions
- **Help**: Tooltips for complex fields

---

## Security Considerations

1. **Input Validation**: All user inputs validated
2. **SQL Injection**: Use parameterized queries (already done)
3. **XSS Prevention**: Escape user content in displays
4. **YAML Injection**: Validate YAML structure
5. **Access Control**: Optional agent ID authentication (future)
6. **Rate Limiting**: Prevent API abuse (future)

---

## Future Enhancements

### Phase 9+ (Future Releases)

- [ ] **Template Editor**: Create/edit custom templates in UI
- [ ] **Batch Operations**: Bulk create/update/delete
- [ ] **Memory Search**: Full-text search across memories
- [ ] **Analytics Dashboard**: Memory usage statistics
- [ ] **Export/Import**: Download/upload memories (JSON/YAML)
- [ ] **Collaboration**: Multi-user support with permissions
- [ ] **Notifications**: Email/webhook on consolidation
- [ ] **API Documentation**: Interactive API docs in UI
- [ ] **Template Marketplace**: Share community templates
- [ ] **Version Control**: Track memory changes over time
- [ ] **Scheduled Tasks**: Automated consolidation/cleanup
- [ ] **Mobile App**: React Native companion app

---

## Success Criteria

### Functional
- ✅ All 62 BMAD templates browsable
- ✅ Users can create memories with pre-built templates
- ✅ Users can create memories with custom YAML
- ✅ Users can view all memories for an agent
- ✅ Users can update memory sections
- ✅ Users can delete memories
- ✅ All operations validated and error-handled

### Non-Functional
- ✅ UI loads in < 2 seconds
- ✅ Template browsing handles 100+ templates smoothly
- ✅ Memory list paginated for 1000+ memories
- ✅ Works on Chrome, Firefox, Safari, Edge
- ✅ Mobile-responsive design
- ✅ WCAG 2.1 AA accessibility compliance

### User Experience
- ✅ Intuitive navigation (< 3 clicks to any action)
- ✅ Clear error messages with actionable steps
- ✅ Help tooltips for complex features
- ✅ Consistent design across all pages
- ✅ Fast feedback on all actions

---

## Timeline

**Total Estimate**: 4 weeks

- **Week 1**: Foundation + Browse Templates
- **Week 2**: Create Memory + View Memories
- **Week 3**: Update Memory + Delete Memory
- **Week 4**: Polish, Testing, Documentation

---

## Resources

### Documentation
- Streamlit Docs: https://docs.streamlit.io/
- YAML Spec: https://yaml.org/spec/
- Markdown Guide: https://www.markdownguide.org/

### Design Inspiration
- Streamlit Component Gallery
- Material Design
- GitHub UI patterns

---

## Notes

- Keep UI stateless where possible (use session state judiciously)
- Optimize template loading (cache, lazy load)
- Test with large numbers of memories/templates
- Consider internationalization (i18n) in future
- Document all UI components for reusability

---

**Next Steps**: Begin Phase 1 implementation after plan approval.
