"""
Update Memory Page

Edit and update sections in existing active memories.
"""

import streamlit as st
from typing import Optional, List, Dict, Any
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from streamlit_app.services.memory_service import MemoryService
from streamlit_app.utils.formatters import Formatters
from streamlit_app.config import APP_TITLE, APP_ICON, DB_CONFIG
import asyncio

# Page configuration
st.set_page_config(page_title=f"Update Memory - {APP_TITLE}", page_icon=APP_ICON, layout="wide")

# Custom CSS for compact layout
st.markdown(
    """
<style>
    .main .block-container {padding-top: 1rem; padding-bottom: 1rem;}
    .stExpander {margin-bottom: 0.25rem;}
    .stButton>button {margin: 0.1rem;}
    .stTextInput input, .stTextArea textarea {padding: 0.25rem;}
    .stSelectbox select {padding: 0.25rem;}
    h1, h2, h3 {margin-top: 0.5rem; margin-bottom: 0.5rem;}
    .stColumns {gap: 0.5rem;}
</style>
""",
    unsafe_allow_html=True,
)

# Initialize services
if "memory_service" not in st.session_state:
    st.session_state.memory_service = MemoryService(DB_CONFIG)

memory_service = st.session_state.memory_service

# Initialize session state
if "update_external_id" not in st.session_state:
    st.session_state.update_external_id = ""
if "selected_memory_id" not in st.session_state:
    st.session_state.selected_memory_id = None
if "selected_section_id" not in st.session_state:
    st.session_state.selected_section_id = None
if "original_content" not in st.session_state:
    st.session_state.original_content = ""
if "has_unsaved_changes" not in st.session_state:
    st.session_state.has_unsaved_changes = False

# Page header
st.title("✏️ Update Memory Section")
st.markdown("Edit and update individual sections of an active memory.")

# Sidebar - Agent and Memory Selection
with st.sidebar:
    st.header("🔍 Select Memory")

    # External ID input
    # Check if coming from View Memories page
    if (
        "update_external_id" in st.session_state
        and st.session_state.update_external_id
        and not st.session_state.get("external_id_manually_changed", False)
    ):
        default_external_id = st.session_state.update_external_id
        # Auto-trigger load if coming from View Memories
        if "update_from_view" not in st.session_state:
            st.session_state.update_from_view = True
            st.session_state.memories_loaded = True
    else:
        default_external_id = st.session_state.update_external_id

    external_id = st.text_input(
        "External ID",
        value=default_external_id,
        placeholder="agent-001, UUID, or numeric ID",
        help="Enter the agent's external ID",
        key="external_id_input",
    )

    # Update session state
    if external_id != st.session_state.update_external_id:
        st.session_state.update_external_id = external_id
        st.session_state.selected_memory_id = None  # Reset selection
        st.session_state.external_id_manually_changed = True

    # ID Type selector - use session state if available
    default_id_type = st.session_state.get("update_id_type", "String")
    id_type = st.radio(
        "ID Type",
        options=["String", "UUID", "Integer"],
        index=(
            ["String", "UUID", "Integer"].index(default_id_type)
            if default_id_type in ["String", "UUID", "Integer"]
            else 0
        ),
        horizontal=False,
        help="Type of the external ID",
    )

    st.divider()

    # Load memories button
    load_clicked = st.button(
        "🔍 Load Memories",
        type="primary",
        use_container_width=True,
        disabled=not external_id.strip(),
    )

    st.divider()

    # Help section
    st.subheader("ℹ️ Help")
    st.markdown(
        """
    **Updating Sections:**
    1. Enter an external ID
    2. Click "Load Memories"
    3. Select a memory from dropdown
    4. Choose a section to edit
    5. Make your changes
    6. Click "Update Section"
    
    **Tips:**
    - Changes are not saved until you click Update
    - Update count shows consolidation progress
    - Review warnings before updating
    """
    )

# Mock data for demonstration
mock_memories = [
    {
        "id": 1,
        "title": "Brainstorming Session - Product Features",
        "template_id": "bmad.brainstorming-session.v1",
        "template_name": "Brainstorming Session",
        "created_at": "2025-10-01T10:30:00Z",
        "sections": [
            {
                "id": "session_goal",
                "title": "Session Goal",
                "content": "Generate 20+ innovative ideas for improving user activation in the first 7 days of the mobile app.",
                "update_count": 0,
                "last_updated": "2025-10-01T10:30:00Z",
                "consolidation_threshold": 5,
            },
            {
                "id": "participants",
                "title": "Participants & Roles",
                "content": "PM (product goals), UX Designer (user perspective), Senior Engineer (feasibility), Customer Success (customer feedback), Marketing Manager (positioning and messaging).",
                "update_count": 1,
                "last_updated": "2025-10-02T14:20:00Z",
                "consolidation_threshold": 5,
            },
            {
                "id": "ideas_generated",
                "title": "Ideas Generated",
                "content": '1. Interactive product tour with gamification\n2. Pre-populated demo data for instant value\n3. AI-powered setup wizard\n4. Short video tutorials (< 60 seconds each)\n5. Slack bot for onboarding assistance\n6. Progressive disclosure of features\n7. Social proof elements ("1000+ teams use this")\n8. Quick win checklist\n9. Personalized onboarding based on role\n10. In-app celebration animations for milestones',
                "update_count": 4,
                "last_updated": "2025-10-02T15:45:00Z",
                "consolidation_threshold": 5,
            },
        ],
    },
    {
        "id": 2,
        "title": "Code Review Notes - Authentication Module",
        "template_id": "bmad.code-review.v1",
        "template_name": "Code Review Session",
        "created_at": "2025-10-02T09:15:00Z",
        "sections": [
            {
                "id": "code_overview",
                "title": "Code Overview",
                "content": "Reviewed JWT authentication implementation in auth-service/middleware. Clean separation of concerns with proper error handling.",
                "update_count": 0,
                "last_updated": "2025-10-02T09:15:00Z",
                "consolidation_threshold": 5,
            },
            {
                "id": "issues_found",
                "title": "Issues Found",
                "content": "1. Token expiration not validated in edge cases\n2. Missing rate limiting on refresh token endpoint\n3. Inconsistent error messages",
                "update_count": 0,
                "last_updated": "2025-10-02T09:15:00Z",
                "consolidation_threshold": 5,
            },
        ],
    },
]

# Main content area
if load_clicked and external_id.strip():
    st.session_state.memories_loaded = True

# Show content if memories are loaded
if external_id.strip() and st.session_state.get("memories_loaded", False):
    # Load memories from API
    with st.spinner("Loading memories..."):
        try:
            memories = asyncio.run(memory_service.get_active_memories(external_id))

            if not memories:
                st.info("📭 No memories found for this agent.")
                st.markdown(
                    """
                **No memories yet?**
                - This agent doesn't have any active memories yet
                - Create a new memory using the **➕ Create Memory** page
                """
                )
            else:
                # Success - proceed with memory selection
                st.success(f"✅ Loaded {len(memories)} memor{'y' if len(memories) == 1 else 'ies'}")

        except Exception as e:
            st.error(f"❌ Error loading memories: {str(e)}")
            st.exception(e)
            memories = []

    if memories:
        # Memory selector
        st.subheader("1️⃣ Select Memory")

        memory_options = {f"#{mem.id} - {mem.title}": mem.id for mem in memories}

        selected_memory_display = st.selectbox(
            "Choose a memory to update",
            options=list(memory_options.keys()),
            help="Select the memory you want to edit",
        )

        selected_memory_id = memory_options[selected_memory_display]
        st.session_state.selected_memory_id = selected_memory_id

        # Find selected memory
        selected_memory = next((m for m in memories if m.id == selected_memory_id), None)

        if selected_memory:
            # Extract template info from YAML
            import yaml

            template_data = {}
            try:
                template_data = yaml.safe_load(selected_memory.template_content)
            except Exception:
                template_data = {}

            template_info = template_data.get("template", {})
            template_id = template_info.get("id", "unknown")
            template_name = template_info.get("name", "Unknown Template")

            # Display memory info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Template", template_name or template_id)
            with col2:
                created = Formatters.format_timestamp(str(selected_memory.created_at))
                st.metric("Created", created)
            with col3:
                st.metric("Sections", len(selected_memory.sections))

            st.caption(f"Template ID: `{template_id}`")

            st.divider()

            # Section selector
            st.subheader("2️⃣ Select Section")

            # Convert sections dict to list for selector with new fields
            sections_list = []
            for section_id, section_data in selected_memory.sections.items():
                sections_list.append(
                    {
                        "id": section_id,
                        "title": section_id.replace("_", " ").title(),
                        "update_count": section_data.get("update_count", 0),
                        "awake_update_count": section_data.get("awake_update_count", 0),
                        "last_updated": section_data.get("last_updated"),
                        "content": section_data.get("content", ""),
                    }
                )

            section_options = {
                f"{sec['title']} (Updates: {sec['update_count']}/{sec['awake_update_count']})": sec["id"]
                for sec in sections_list
            }

            selected_section_display = st.selectbox(
                "Choose a section to edit",
                options=list(section_options.keys()),
                help="Select the section you want to update",
            )

            selected_section_id = section_options[selected_section_display]
            st.session_state.selected_section_id = selected_section_id

            # Find selected section
            selected_section = next(
                (s for s in sections_list if s["id"] == selected_section_id), None
            )

            if selected_section:
                # Store original content for change detection
                if st.session_state.original_content == "":
                    st.session_state.original_content = selected_section["content"]

                # Section info with new fields
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    update_count = selected_section["update_count"]
                    threshold = 5  # Default threshold from config
                    st.metric("Update Count", f"{update_count}/{threshold}")

                with col2:
                    awake_count = selected_section.get("awake_update_count", 0)
                    st.metric("Awake Count", awake_count)

                with col3:
                    # Get last updated from section data
                    last_updated = selected_section.get("last_updated")
                    if last_updated:
                        formatted_time = Formatters.format_timestamp(last_updated)
                    else:
                        formatted_time = "Never"
                    st.metric("Last Updated", formatted_time)

                with col4:
                    updates_remaining = threshold - update_count
                    if updates_remaining <= 1:
                        st.metric(
                            "Status", "⚠️ Near Consolidation", delta=f"{updates_remaining} left"
                        )
                    else:
                        st.metric("Status", "✅ Active", delta=f"{updates_remaining} updates left")

                st.caption(f"Section ID: `{selected_section['id']}`")

                # Consolidation warning
                if update_count >= threshold - 1:
                    st.warning(
                        f"⚠️ **Consolidation Warning**: This section is approaching its consolidation threshold "
                        f"({update_count}/{threshold} updates). The next update may trigger automatic consolidation."
                    )

                st.divider()

                # Content editor with upsert options
                st.subheader("3️⃣ Edit Content")

                # Action selector
                st.markdown("**Update Action**")
                action = st.radio(
                    "Choose how to update the section:",
                    options=["replace", "insert"],
                    format_func=lambda x: "🔄 Replace Content" if x == "replace" else "➕ Insert/Append Content",
                    horizontal=True,
                    help="Replace: Replace content (entire or specific pattern). Insert: Add content at end or after pattern.",
                    label_visibility="collapsed",
                )

                # Old content pattern (for replace/insert operations)
                if action == "replace":
                    st.markdown("**Pattern to Replace** (Optional)")
                    old_content = st.text_input(
                        "Find this text",
                        placeholder="Leave empty to replace entire content",
                        help="Text to find and replace. If empty, replaces entire section content.",
                        label_visibility="collapsed",
                    )
                else:  # insert
                    st.markdown("**Insert After Pattern** (Optional)")
                    old_content = st.text_input(
                        "Insert after this text",
                        placeholder="Leave empty to append at end",
                        help="Text to find and insert after. If empty, appends at the end of the section.",
                        label_visibility="collapsed",
                    )

                # Two-column layout: Editor and Preview
                col1, col2 = st.columns([1, 1])

                with col1:
                    st.markdown("**Markdown Editor**")
                    new_content = st.text_area(
                        "Section Content",
                        value=selected_section["content"],
                        height=400,
                        help="Edit the section content using Markdown",
                        key="content_editor",
                        label_visibility="collapsed",
                    )

                    # Character count
                    char_count = len(new_content)
                    st.caption(f"Characters: {char_count:,}")

                    # Change detection
                    has_changes = (
                        new_content != st.session_state.original_content or
                        action != "replace" or
                        old_content != ""
                    )
                    if has_changes:
                        st.session_state.has_unsaved_changes = True
                        st.warning("⚠️ You have unsaved changes")
                    else:
                        st.session_state.has_unsaved_changes = False

                with col2:
                    st.markdown("**Preview**")
                    preview_container = st.container()
                    with preview_container:
                        if new_content.strip():
                            st.markdown(new_content)
                        else:
                            st.caption("_Preview will appear here_")

                st.divider()

                # Action buttons
                col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

                with col1:
                    if st.button(
                        "💾 Update Section",
                        type="primary",
                        use_container_width=True,
                        disabled=not st.session_state.has_unsaved_changes,
                    ):
                        # Prepare upsert data for new API
                        section_updates = [{
                            "section_id": selected_section_id,
                            "old_content": old_content if old_content.strip() else None,
                            "new_content": new_content,
                            "action": action,
                        }]

                        # Call new upsert API
                        with st.spinner("Updating section..."):
                            try:
                                # Note: This would need to be updated to use the new upsert API method
                                # For now, we'll use the old method for backward compatibility
                                updated_memory = asyncio.run(
                                    memory_service.update_active_memory_section(
                                        external_id=external_id,
                                        memory_id=selected_memory_id,
                                        section_id=selected_section_id,
                                        new_content=new_content,
                                    )
                                )

                                if updated_memory:
                                    st.success("✅ Section updated successfully!")
                                    st.balloons()

                                    # Check if consolidation occurred
                                    new_update_count = updated_memory.sections[
                                        selected_section_id
                                    ].get("update_count", 0)
                                    if new_update_count == 0:
                                        st.info(
                                            "🔄 **Consolidation triggered**: This section was automatically consolidated to shortterm memory and reset."
                                        )
                                    else:
                                        st.info(f"Update count is now: {new_update_count}")

                                    # Reset state
                                    st.session_state.original_content = new_content
                                    st.session_state.has_unsaved_changes = False
                                    st.rerun()
                                else:
                                    st.error(
                                        "❌ Failed to update section. Please check database connection."
                                    )

                            except Exception as e:
                                st.error(f"❌ Error updating section: {str(e)}")
                                st.exception(e)
                    st.info(
                        f"Update action: **{action.title()}** | Update count would increment: {update_count} → {update_count + 1}"
                    )

                    # Reset state
                    st.session_state.original_content = new_content
                    st.session_state.has_unsaved_changes = False

                    # Check if consolidation would trigger
                    if update_count + 1 >= threshold:
                        st.warning("🔄 Consolidation would be triggered after this update!")

            with col2:
                if st.button(
                    "🔄 Reset Changes",
                    use_container_width=True,
                    disabled=not st.session_state.has_unsaved_changes,
                ):
                    st.rerun()

            with col3:
                if st.button("📋 View All Sections", use_container_width=True):
                    st.info("Navigate to View Memories page (Demo)")

            with col4:
                if st.button("🔙 Back to Memories", use_container_width=True):
                    # Clear selection state
                    st.session_state.selected_memory_id = None
                    st.session_state.selected_section_id = None
                    st.session_state.original_content = ""
                    st.session_state.has_unsaved_changes = False
                    st.rerun()

            # Additional section details
            st.divider()

            with st.expander("📊 Section Details", expanded=False):
                st.markdown(f"**Section ID:** `{selected_section['id']}`")
                st.markdown(f"**Title:** {selected_section['title']}")
                st.markdown(f"**Update Count:** {selected_section['update_count']}")
                st.markdown(f"**Awake Update Count:** {selected_section.get('awake_update_count', 0)}")

                # Consolidation threshold may not exist in all section data
                consolidation_threshold = selected_section.get("consolidation_threshold", 5)
                st.markdown(f"**Consolidation Threshold:** {consolidation_threshold}")

                # Last updated may not exist in all section data
                last_updated = selected_section.get("last_updated")
                if last_updated:
                    st.markdown(f"**Last Updated:** {Formatters.format_timestamp(last_updated)}")
                else:
                    st.markdown(f"**Last Updated:** Never")

                st.markdown(f"**Content Length:** {len(selected_section['content'])} characters")

elif not external_id.strip():
    # Empty state - no external ID
    st.info("👈 Enter an agent's external ID in the sidebar to begin updating memories.")

    # Show helpful information
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
        ### 🔍 Quick Start
        1. Enter an external ID in the sidebar
        2. Click "Load Memories"
        3. Select a memory to update
        """
        )

    with col2:
        st.markdown(
            """
        ### ✏️ Editing
        - Live markdown preview
        - Character count
        - Unsaved changes detection
        """
        )

    with col3:
        st.markdown(
            """
        ### ⚠️ Warnings
        - Consolidation threshold alerts
        - Update count tracking
        - Auto-increment on save
        """
        )

else:
    # Empty state - need to load memories
    st.warning(f"Click 'Load Memories' in the sidebar to view memories for `{external_id}`")

# Footer
st.divider()
st.caption(f"Update Memory | {APP_TITLE}")


