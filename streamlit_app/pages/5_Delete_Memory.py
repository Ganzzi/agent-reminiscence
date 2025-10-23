"""
Delete Memory Page

Permanently delete active memories with safety confirmations.
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
st.set_page_config(page_title=f"Delete Memory - {APP_TITLE}", page_icon=APP_ICON, layout="wide")

# Custom CSS for compact layout
st.markdown(
    """
<style>
    .main .block-container {padding-top: 1rem; padding-bottom: 1rem;}
    .stExpander {margin-bottom: 0.25rem;}
    .stButton>button {margin: 0.1rem;}
    .stTextInput input {padding: 0.25rem;}
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
if "delete_external_id" not in st.session_state:
    st.session_state.delete_external_id = ""
if "delete_memories_loaded" not in st.session_state:
    st.session_state.delete_memories_loaded = False
if "selected_memory_for_deletion" not in st.session_state:
    st.session_state.selected_memory_for_deletion = None
if "deletion_confirmed" not in st.session_state:
    st.session_state.deletion_confirmed = False
if "confirmation_text" not in st.session_state:
    st.session_state.confirmation_text = ""
if "understand_irreversible" not in st.session_state:
    st.session_state.understand_irreversible = False

# Page header
st.title("🗑️ Delete Memory")
st.markdown(
    "⚠️ **Warning**: This action is **irreversible**. All memory data and sections will be permanently deleted."
)

# Sidebar - Agent and Memory Selection
with st.sidebar:
    st.header("🔍 Select Memory")

    # External ID input
    # Check if coming from View Memories page
    if (
        "delete_external_id" in st.session_state
        and st.session_state.delete_external_id
        and not st.session_state.get("delete_external_id_manually_changed", False)
    ):
        default_external_id = st.session_state.delete_external_id
        # Auto-trigger load if coming from View Memories
        if "delete_from_view" not in st.session_state:
            st.session_state.delete_from_view = True
            st.session_state.delete_memories_loaded = True
    else:
        default_external_id = st.session_state.delete_external_id

    external_id = st.text_input(
        "External ID",
        value=default_external_id,
        placeholder="agent-001, UUID, or numeric ID",
        help="Enter the agent's external ID",
        key="delete_external_id_input",
    )

    # Update session state
    if external_id != st.session_state.delete_external_id:
        st.session_state.delete_external_id = external_id
        st.session_state.selected_memory_for_deletion = None  # Reset selection
        st.session_state.deletion_confirmed = False
        st.session_state.confirmation_text = ""
        st.session_state.understand_irreversible = False
        st.session_state.delete_external_id_manually_changed = True

    # ID Type selector - use session state if available
    default_id_type = st.session_state.get("delete_id_type", "String")
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
    **Deletion Process:**
    1. Enter an external ID
    2. Click "Load Memories"
    3. Select a memory to delete
    4. Review memory details
    5. Type the memory title to confirm
    6. Check "I understand" checkbox
    7. Click "Delete Memory"
    
    **Safety Measures:**
    - Type exact title to confirm
    - Irreversibility acknowledgment
    - Final confirmation required
    - Cannot be undone
    """
    )

    st.divider()

    # Danger zone indicator
    st.error("⚠️ **DANGER ZONE**\n\nDeleted memories cannot be recovered!")

# Mock data for demonstration
mock_memories = [
    {
        "id": 1,
        "title": "Brainstorming Session - Product Features",
        "template_id": "bmad.brainstorming-session.v1",
        "template_name": "Brainstorming Session",
        "created_at": "2025-10-01T10:30:00Z",
        "updated_at": "2025-10-02T15:45:00Z",
        "priority": "high",
        "usage_type": "conversation",
        "metadata": {"team": "product", "sprint": "Q4-W1"},
        "sections": [
            {
                "id": "session_goal",
                "title": "Session Goal",
                "content": "Generate 20+ innovative ideas for improving user activation in the first 7 days of the mobile app.",
                "update_count": 0,
                "last_updated": "2025-10-01T10:30:00Z",
            },
            {
                "id": "participants",
                "title": "Participants & Roles",
                "content": "PM (product goals), UX Designer (user perspective), Senior Engineer (feasibility), Customer Success (customer feedback), Marketing Manager (positioning and messaging).",
                "update_count": 1,
                "last_updated": "2025-10-02T14:20:00Z",
            },
            {
                "id": "ideas_generated",
                "title": "Ideas Generated",
                "content": '1. Interactive product tour with gamification\n2. Pre-populated demo data for instant value\n3. AI-powered setup wizard\n4. Short video tutorials (< 60 seconds each)\n5. Slack bot for onboarding assistance\n6. Progressive disclosure of features\n7. Social proof elements ("1000+ teams use this")\n8. Quick win checklist\n9. Personalized onboarding based on role\n10. In-app celebration animations for milestones',
                "update_count": 4,
                "last_updated": "2025-10-02T15:45:00Z",
            },
        ],
    },
    {
        "id": 2,
        "title": "Code Review Notes - Authentication Module",
        "template_id": "bmad.code-review.v1",
        "template_name": "Code Review Session",
        "created_at": "2025-10-02T09:15:00Z",
        "updated_at": "2025-10-02T09:15:00Z",
        "priority": "medium",
        "usage_type": "task",
        "metadata": {"repo": "auth-service", "pr": "PR-142"},
        "sections": [
            {
                "id": "code_overview",
                "title": "Code Overview",
                "content": "Reviewed JWT authentication implementation in auth-service/middleware. Clean separation of concerns with proper error handling.",
                "update_count": 0,
                "last_updated": "2025-10-02T09:15:00Z",
            },
            {
                "id": "issues_found",
                "title": "Issues Found",
                "content": "1. Token expiration not validated in edge cases\n2. Missing rate limiting on refresh token endpoint\n3. Inconsistent error messages",
                "update_count": 0,
                "last_updated": "2025-10-02T09:15:00Z",
            },
        ],
    },
]

# Main content area
if load_clicked and external_id.strip():
    st.session_state.delete_memories_loaded = True

# Show content if memories are loaded
if external_id.strip() and st.session_state.get("delete_memories_loaded", False):
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
                st.success(f"✅ Loaded {len(memories)} memor{'y' if len(memories) == 1 else 'ies'}")

        except Exception as e:
            st.error(f"❌ Error loading memories: {str(e)}")
            st.exception(e)
            memories = []

    if memories:
        # Memory selector
        st.subheader("1️⃣ Select Memory to Delete")

        memory_options = {f"#{mem.id} - {mem.title}": mem.id for mem in memories}

        selected_memory_display = st.selectbox(
            "Choose a memory to permanently delete",
            options=list(memory_options.keys()),
            help="⚠️ Selected memory will be permanently deleted",
        )

        selected_memory_id = memory_options[selected_memory_display]
        st.session_state.selected_memory_for_deletion = selected_memory_id

        # Find selected memory
        selected_memory = next((m for m in memories if m.id == selected_memory_id), None)

        if selected_memory:
            st.divider()

            # Display full memory details
            st.subheader("2️⃣ Memory Details")
            st.warning(
                "⚠️ Review all details before proceeding. This memory will be **permanently deleted**."
            )

            # Memory info card
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                priority = selected_memory.metadata.get("priority", "medium")
                priority_emoji, priority_color = Formatters.format_priority(priority)
                st.metric("Priority", priority_emoji)

            with col2:
                usage_type = selected_memory.metadata.get("usage", "session")
                usage_emoji = Formatters.format_usage_type(usage_type)
                st.metric("Usage Type", usage_emoji)

            with col3:
                created = Formatters.format_timestamp(str(selected_memory.created_at))
                st.metric("Created", created)

            with col4:
                st.metric("Sections", len(selected_memory.sections))

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

            # Template info
            st.markdown(f"**Template**: {template_name or template_id}")
            st.caption(f"Template ID: `{template_id}`")

            # Metadata
            if selected_memory.metadata:
                with st.expander("📊 Metadata", expanded=False):
                    st.json(selected_memory.metadata)

            # Sections preview
            with st.expander("📄 All Sections (will be deleted)", expanded=True):
                sections_list = []
                for section_id, section_data in selected_memory.sections.items():
                    sections_list.append(
                        {
                            "id": section_id,
                            "title": section_id.replace("_", " ").title(),
                            "content": section_data.get("content", ""),
                            "update_count": section_data.get("update_count", 0),
                        }
                    )

                for i, section in enumerate(sections_list, 1):
                    st.markdown(f"**{i}. {section['title']}** (`{section['id']}`)")
                    content = section["content"]
                    st.markdown(f"> {content[:150]}..." if len(content) > 150 else f"> {content}")
                    st.caption(
                        f"Updates: {section['update_count']} | Last updated: {Formatters.format_timestamp(str(selected_memory.updated_at))}"
                    )
                    if i < len(sections_list):
                        st.markdown("---")

            st.divider()

            # Confirmation section
            st.subheader("3️⃣ Confirm Deletion")
            st.error(
                "⚠️ **FINAL WARNING**: This action cannot be undone. All sections and metadata will be permanently deleted."
            )

            # Confirmation requirements
            st.markdown("**Safety Requirements:**")
            st.markdown("1. Type the exact memory title below")
            st.markdown("2. Check the acknowledgment checkbox")
            st.markdown("3. Click the Delete button")

            st.markdown("")

            # Type to confirm
            confirmation_input = st.text_input(
                f"Type the memory title to confirm: **{selected_memory.title}**",
                placeholder=selected_memory.title,
                help="Type the exact title to enable deletion",
                key="confirmation_input",
            )

            # Store in session state
            st.session_state.confirmation_text = confirmation_input

            # Check if confirmation matches
            confirmation_matches = confirmation_input.strip() == selected_memory.title

            if confirmation_input and not confirmation_matches:
                st.warning("⚠️ Title does not match. Please type the exact title.")
            elif confirmation_matches:
                st.success("✅ Title confirmed")

            # Irreversibility acknowledgment
            understand_checked = st.checkbox(
                "I understand this action is **irreversible** and will permanently delete all data",
                key="understand_checkbox",
            )

            st.session_state.understand_irreversible = understand_checked

            st.divider()

            # Action buttons
            col1, col2, col3 = st.columns([1, 1, 2])

            with col1:
                delete_enabled = confirmation_matches and understand_checked

                if st.button(
                    "🗑️ Delete Memory",
                    type="primary",
                    use_container_width=True,
                    disabled=not delete_enabled,
                ):
                    # Call actual API
                    with st.spinner("Deleting memory..."):
                        try:
                            success, message = asyncio.run(
                                memory_service.delete_active_memory(
                                    external_id=external_id,
                                    memory_id=selected_memory_id,
                                )
                            )

                            if success:
                                st.success("✅ Memory deleted successfully!")
                                st.balloons()
                                st.info(
                                    f"Memory #{selected_memory_id} has been permanently deleted."
                                )

                                # Reset state
                                st.session_state.memories_loaded = False
                                st.session_state.selected_memory_for_deletion = None
                                st.session_state.confirmation_text = ""
                                st.session_state.understand_irreversible = False

                                # Wait a moment before rerun
                                import time

                                time.sleep(2)
                                st.rerun()
                            else:
                                st.error(f"❌ Failed to delete memory: {message}")

                                # Check if it's the "not implemented" message
                                if "not yet implemented" in message.lower():
                                    st.info(
                                        "ℹ️ **Note**: The delete API is not yet implemented in AgentMem core. "
                                        "Complete Phase 8.6 to add full delete functionality."
                                    )

                        except Exception as e:
                            st.error(f"❌ Error deleting memory: {str(e)}")
                            st.exception(e)

                st.success(f"✅ Memory #{selected_memory_id} deleted successfully! (Demo)")
                st.info(
                    "In production, this memory would be permanently removed from the database."
                )
                st.balloons()

                # Reset state
                st.session_state.selected_memory_for_deletion = None
                st.session_state.confirmation_text = ""
                st.session_state.understand_irreversible = False

        with col2:
            if st.button("🔙 Cancel", use_container_width=True):
                # Clear selection state
                st.session_state.selected_memory_for_deletion = None
                st.session_state.confirmation_text = ""
                st.session_state.understand_irreversible = False
                st.rerun()

        with col3:
            if not delete_enabled:
                if not confirmation_matches:
                    st.caption("⚠️ Type exact title to enable deletion")
                elif not understand_checked:
                    st.caption("⚠️ Check acknowledgment to enable deletion")

elif not external_id.strip():
    # Empty state - no external ID
    st.info("👈 Enter an agent's external ID in the sidebar to begin.")

    # Show helpful information
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
        ### 🔍 Quick Start
        1. Enter an external ID
        2. Click "Load Memories"
        3. Select memory to delete
        """
        )

    with col2:
        st.markdown(
            """
        ### 🛡️ Safety Features
        - Type-to-confirm
        - Irreversibility checkbox
        - Full memory preview
        """
        )

    with col3:
        st.markdown(
            """
        ### ⚠️ Important
        - Cannot be undone
        - All sections deleted
        - Metadata removed
        """
        )

else:
    # Empty state - need to load memories
    st.warning(f"Click 'Load Memories' in the sidebar to view memories for `{external_id}`")

# Footer
st.divider()
st.caption(f"Delete Memory | {APP_TITLE}")


