# Streamlit UI User Guide

**AgentMem UI** - Manage agent memories with an intuitive web interface

---

## Overview

The AgentMem Streamlit UI provides a user-friendly interface for managing active memories using pre-built BMAD templates. This guide will help you get started and make the most of all features.

## Getting Started

### Starting the UI

```bash
cd streamlit_app
streamlit run app.py
```

The UI will open in your default browser at `http://localhost:8501`

### Navigation

Use the sidebar to navigate between pages:

- 🏠 **app** - Home page with overview
- 📚 **Browse Templates** - Explore 60+ pre-built templates
- ➕ **Create Memory** - Create new active memories
- 📋 **View Memories** - View and browse existing memories
- ✏️ **Update Memory** - Edit memory sections
- 🗑️ **Delete Memory** - Permanently delete memories

---

## Pages Guide

### 1. Browse Templates 📚

**Purpose**: Explore and preview 60+ pre-built BMAD templates organized by agent type.

**Features**:
- **Agent Filter**: Filter templates by agent type (10 categories)
- **Search**: Find templates by name or ID
- **Template Cards**: View template metadata
  - Template name and ID
  - Number of sections
  - Usage type and priority
  - Description
- **Preview Modal**: Click any template to view:
  - Complete YAML structure
  - Section definitions with descriptions
  - Usage and priority information

**How to Use**:
1. Select an agent type from the dropdown (or "All Agents")
2. Use the search bar to find specific templates
3. Click a template card to open the preview modal
4. Review the YAML structure and sections
5. Note the template ID for use in memory creation

**Tips**:
- Use the agent filter to narrow down relevant templates
- Read section descriptions to understand what content is expected
- Copy template IDs for quick reference

---

### 2. Create Memory ➕

**Purpose**: Create new active memories using pre-built templates or custom YAML.

**Two Creation Modes**:

#### Pre-built Template Mode
1. Enter the agent's **External ID**
2. Select **ID Type** (String, UUID, or Integer)
3. Choose a **Template** from the dropdown
   - Filter by agent type for relevant templates
4. Review auto-populated **Title** and **Metadata**
5. Fill in **Section Content** (7 expandable editors):
   - Each section has a description from the template
   - Use Markdown formatting for rich content
   - Update count starts at 0
6. Click **Create Memory**

#### Custom YAML Mode
1. Enter the agent's **External ID**
2. Toggle to **Custom YAML** mode
3. Paste or write your YAML memory definition:
   ```yaml
   title: "My Custom Memory"
   template_id: "custom.memory.v1"
   priority: "high"
   usage_type: "conversation"
   metadata:
     custom_field: "value"
   sections:
     - id: "section_1"
       title: "Section Title"
       content: "Section content here"
       update_count: 0
   ```
4. Click **Validate YAML** (optional)
5. Click **Create Memory**

**Tips**:
- Pre-built mode is faster and ensures proper structure
- Custom YAML gives full control but requires correct formatting
- Section descriptions provide guidance on what content to include
- Use Markdown for formatting (lists, bold, headers, etc.)

---

### 3. View Memories 📋

**Purpose**: Browse and view all active memories for a specific agent.

**Features**:
- **Memory Cards**: Display key information
  - Title
  - Priority badge (🔴 High, 🟡 Medium, 🟢 Low)
  - Usage type badge
  - Memory ID, Template, Created date, Section count
- **Expandable Sections**: Click to view details
  - Section ID and title
  - Full content (truncated if > 500 characters)
  - Update count badge
  - Last updated timestamp
  - Action buttons (Update Section, Copy Content)
- **Empty States**: Helpful messages when no data
- **Session Persistence**: External ID remembered across pages

**How to Use**:
1. Enter the agent's **External ID** in the sidebar
2. Select **ID Type**
3. Click **Load Memories**
4. View the list of memories (sorted by creation date)
5. Click a section to expand and view content
6. Use action buttons to update or copy sections

**Tips**:
- External ID persists while you navigate between pages
- Update count shows how close a section is to consolidation
- "Show more" button appears for long content
- Use Refresh button to reload after changes

---

### 4. Update Memory ✏️

**Purpose**: Edit and update individual sections of active memories.

**Features**:
- **Memory Selector**: Choose which memory to edit
- **Section Selector**: Pick the specific section to update
- **Dual-Pane Editor**:
  - **Left**: Markdown editor with character count
  - **Right**: Live preview of formatted content
- **Section Metrics**:
  - Update count (X/threshold)
  - Last updated timestamp
  - Status (Active vs Near Consolidation)
- **Unsaved Changes Detection**: Warning when content is modified
- **Consolidation Warnings**: Alert when approaching threshold

**How to Use**:
1. Enter **External ID** and click **Load Memories**
2. Select a **Memory** from the dropdown
3. Choose a **Section** to edit
4. View current metrics (update count, status, etc.)
5. Edit content in the **Markdown Editor**
   - Changes appear immediately in the preview
   - Character count updates in real-time
6. Review the **Live Preview** on the right
7. Click **Update Section** to save changes

**Tips**:
- Unsaved changes show a warning banner
- Update button only enables when content changes
- Pay attention to consolidation warnings (threshold alerts)
- Use Reset button to discard changes
- Section Details expander shows full metadata

---

### 5. Delete Memory 🗑️

**Purpose**: Permanently delete active memories with safety confirmations.

**Safety Features**:
- **Type-to-Confirm**: Must type exact memory title
- **Irreversibility Checkbox**: Must acknowledge action cannot be undone
- **Full Preview**: Shows all sections before deletion
- **DANGER ZONE**: Prominent warnings throughout
- **Multiple Confirmation Steps**: Prevents accidental deletion

**How to Use**:
1. Enter **External ID** and click **Load Memories**
2. Select a **Memory** to delete
3. **Review Memory Details**:
   - Priority, usage type, template
   - Created date, section count
   - All sections and content (expandable)
   - Metadata (if present)
4. **Type the Memory Title** exactly as shown
   - Green checkmark appears when correct
5. **Check the Acknowledgment Checkbox**
   - "I understand this action is irreversible..."
6. Click **Delete Memory** (now enabled)
7. Confirmation message appears with balloons

**Safety Requirements**:
- Title must match exactly (case-sensitive)
- Checkbox must be checked
- Both requirements must be met to enable Delete button

**Tips**:
- Triple-check before deleting (action cannot be undone!)
- Review all sections in the preview
- Use Cancel button to abort at any time
- DANGER ZONE warning in sidebar reminds you of risk

---

## Common Workflows

### Workflow 1: Creating a Memory from Template

1. **Browse Templates** → Find suitable template → Note template ID
2. **Create Memory** → Enter External ID → Select template
3. **Fill Sections** → Add content using template guidance
4. **Create** → Memory is created
5. **View Memories** → Verify memory was created successfully

### Workflow 2: Updating Memory Sections

1. **View Memories** → Find memory → Expand section → Click "Update Section"
2. **Update Memory** page opens → Edit content → Preview changes
3. **Update Section** → Success message
4. **View Memories** → Verify update count incremented

### Workflow 3: Monitoring Consolidation

1. **View Memories** → Check update count badges
2. If section shows "3 updates" or "4 updates":
   - Click to expand and review content
   - Consider if consolidation is needed soon
3. **Update Memory** → Warning appears if at threshold
4. Plan for consolidation before hitting limit

---

## Tips & Best Practices

### Content Formatting

**Use Markdown** for rich formatting:
- `**bold**` for emphasis
- `*italic*` for subtle emphasis
- `#` for headers
- `-` or `1.` for lists
- `` `code` `` for inline code
- ` ``` ` for code blocks

### Memory Organization

- **Clear Titles**: Use descriptive, searchable titles
- **Consistent Templates**: Stick to templates for similar tasks
- **Regular Updates**: Update sections incrementally vs. big rewrites
- **Monitor Thresholds**: Watch update counts to plan consolidation

### External IDs

- **Consistency**: Use the same ID format across all memories for an agent
- **Meaningful IDs**: Use agent-001, user-abc123, etc. (not random strings)
- **ID Type**: Match the actual type (String for text, UUID for UUIDs, etc.)

### Performance

- **Session Persistence**: External ID is remembered while navigating
- **Lazy Loading**: Templates load once and are cached
- **Efficient Updates**: Only modified sections are updated

---

## Keyboard Shortcuts

- `Enter` in text inputs → Commit value and enable buttons
- `Ctrl+Enter` in text areas → Commit multi-line content
- `Esc` → Close modals
- `Tab` → Navigate between form fields

---

## Troubleshooting

### "Load Memories" button is disabled
- **Solution**: Enter an External ID and press Enter to enable

### Template list is empty
- **Solution**: Check that `prebuilt-memory-tmpl/bmad/` directory contains templates
- Restart Streamlit if templates were recently added

### Update button stays disabled
- **Solution**: Make changes to the content and press Ctrl+Enter to commit
- Ensure content differs from original

### Consolidation warning appears
- **Meaning**: Section is approaching threshold (4/5 updates)
- **Action**: Next update may trigger consolidation
- **Plan**: Review if consolidation is acceptable

### Delete button won't enable
- **Check**: Title typed exactly matches (case-sensitive)
- **Check**: Acknowledgment checkbox is checked
- **Both** requirements must be met

---

## Demo Mode

Currently, the UI operates in **Demo Mode**:
- Uses mock data (2 sample memories)
- No actual database operations
- Changes are not persisted
- Displayed banners: "🚧 Demo Mode: This is a preview..."

**To connect to a real database**:
1. Uncomment `MemoryService` initialization in each page
2. Provide valid `db_config` with connection details
3. Remove or update mock data sections
4. Test with real agent IDs

---

## API Integration (Future)

When connecting to the AgentMem API, the following operations will be available:

- `create_active_memory(external_id, memory_data)` - Create new memory
- `get_active_memories(external_id)` - Fetch all memories for agent
- `update_active_memory_section(external_id, memory_id, section_id, content)` - Update section
- `delete_active_memory(external_id, memory_id)` - Delete memory

Each page has TODO markers showing where API calls should be made.

---

## Support & Resources

- **GitHub Issues**: Report bugs or request features
- **Documentation**: See `/docs` directory for technical details
- **BMAD Templates**: Explore `/prebuilt-memory-tmpl/bmad` for template examples

---

**Version**: 1.0  
**Last Updated**: October 3, 2025  
**Status**: Demo Mode - Ready for API Integration
