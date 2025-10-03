"""
Navigation Enhancement Module

Provides enhanced navigation and quick action features.
"""

import streamlit as st
from typing import Optional, Dict, List


def add_quick_actions_sidebar() -> None:
    """Add a quick actions menu to the sidebar."""
    with st.sidebar:
        st.markdown("### ⚡ Quick Actions")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("➕ New", use_container_width=True, help="Create new memory"):
                st.switch_page("pages/2_Create_Memory.py")

        with col2:
            if st.button("📋 View", use_container_width=True, help="View memories"):
                st.switch_page("pages/3_View_Memories.py")


def show_recent_activity(activities: List[Dict]) -> None:
    """
    Show recent activity in sidebar.

    Args:
        activities: List of activity dicts with keys: icon, text, time
    """
    with st.sidebar:
        st.markdown("### 📌 Recent Activity")

        if not activities:
            st.caption("No recent activity")
            return

        for activity in activities[:5]:  # Show max 5
            st.caption(f"{activity.get('icon', '•')} {activity.get('text', '')}")
            st.caption(f"   _{activity.get('time', '')}_")


def add_stats_sidebar(stats: Dict[str, any]) -> None:
    """
    Add statistics to sidebar.

    Args:
        stats: Dictionary of stat name to value
    """
    with st.sidebar:
        st.markdown("### 📊 Statistics")

        for label, value in stats.items():
            col1, col2 = st.columns([2, 1])
            with col1:
                st.caption(label)
            with col2:
                st.markdown(f"**{value}**")


def add_search_shortcut() -> Optional[str]:
    """
    Add a global search box to sidebar.

    Returns:
        Search query string if entered
    """
    with st.sidebar:
        st.markdown("### 🔍 Quick Search")
        query = st.text_input(
            "Search...", placeholder="Search templates, memories...", label_visibility="collapsed"
        )

        if query:
            return query.strip()
        return None


def add_favorites_sidebar(favorites: List[str]) -> None:
    """
    Show favorite templates or memories.

    Args:
        favorites: List of favorite item names
    """
    with st.sidebar:
        with st.expander("⭐ Favorites"):
            if not favorites:
                st.caption("No favorites yet")
            else:
                for fav in favorites:
                    st.markdown(f"- {fav}")


def add_help_links() -> None:
    """Add helpful links to sidebar."""
    with st.sidebar:
        st.markdown("### 📖 Resources")
        st.markdown("- [User Guide](../docs/STREAMLIT_UI_USER_GUIDE.md)")
        st.markdown("- [Documentation](../docs/INDEX.md)")
        st.markdown("- [GitHub](https://github.com/Ganzzi/agent-mem)")


def show_connection_status(connected: bool = False) -> None:
    """
    Show database connection status.

    Args:
        connected: Whether database is connected
    """
    with st.sidebar:
        st.markdown("### 🔌 Status")

        if connected:
            st.success("✅ Connected")
        else:
            st.warning("⚠️ Demo Mode")
            st.caption("Connect database for full functionality")


def add_user_profile(username: Optional[str] = None) -> None:
    """
    Show user profile in sidebar.

    Args:
        username: Optional username to display
    """
    with st.sidebar:
        st.markdown("---")

        if username:
            st.markdown(f"👤 **{username}**")
        else:
            st.caption("👤 Guest User")


def add_theme_toggle() -> str:
    """
    Add theme toggle button.

    Returns:
        Selected theme ('light' or 'dark')
    """
    with st.sidebar:
        theme = st.radio(
            "🎨 Theme", options=["Light", "Dark"], horizontal=True, label_visibility="collapsed"
        )
        return theme.lower()


def add_page_navigation(current_page: str, pages: List[Dict]) -> None:
    """
    Add enhanced page navigation with icons.

    Args:
        current_page: Current page name
        pages: List of page dicts with keys: name, icon, path
    """
    st.markdown("### 🧭 Navigation")

    for page in pages:
        is_current = page["name"] == current_page
        button_type = "primary" if is_current else "secondary"
        disabled = is_current

        if st.button(
            f"{page['icon']} {page['name']}",
            key=f"nav_{page['name']}",
            type=button_type,
            disabled=disabled,
            use_container_width=True,
        ):
            st.switch_page(page["path"])


def show_notification_badge(count: int) -> str:
    """
    Create a notification badge.

    Args:
        count: Notification count

    Returns:
        HTML badge string
    """
    if count == 0:
        return ""

    return f"""
    <span style="
        background-color: #EF4444;
        color: white;
        padding: 2px 6px;
        border-radius: 10px;
        font-size: 10px;
        font-weight: bold;
        margin-left: 5px;
    ">{count}</span>
    """


def add_beta_badge(feature_name: str) -> None:
    """
    Add a beta badge to a feature.

    Args:
        feature_name: Name of the feature
    """
    st.markdown(
        f"{feature_name} "
        '<span style="background-color: #3B82F6; color: white; padding: 2px 6px; '
        'border-radius: 4px; font-size: 10px; font-weight: bold;">BETA</span>',
        unsafe_allow_html=True,
    )


def show_progress_tracker(steps: List[str], current_step: int, completed_steps: List[int]) -> None:
    """
    Show a progress tracker for multi-step processes.

    Args:
        steps: List of step names
        current_step: Current step index (0-based)
        completed_steps: List of completed step indices
    """
    st.markdown("### 📋 Progress")

    for i, step in enumerate(steps):
        if i in completed_steps:
            icon = "✅"
            style = "color: #10B981; font-weight: bold;"
        elif i == current_step:
            icon = "▶️"
            style = "color: #3B82F6; font-weight: bold;"
        else:
            icon = "⭕"
            style = "color: #9CA3AF;"

        st.markdown(f'<div style="{style}">{icon} {i+1}. {step}</div>', unsafe_allow_html=True)


def add_keyboard_shortcuts_help() -> None:
    """Add keyboard shortcuts help to sidebar."""
    with st.sidebar:
        with st.expander("⌨️ Keyboard Shortcuts"):
            st.caption("**Navigation**")
            st.caption("• `Ctrl+K` - Quick search")
            st.caption("• `Esc` - Close dialogs")

            st.caption("**Forms**")
            st.caption("• `Enter` - Submit input")
            st.caption("• `Ctrl+Enter` - Submit text area")
            st.caption("• `Tab` - Next field")
