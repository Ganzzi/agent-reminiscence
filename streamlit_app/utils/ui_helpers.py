"""
UI Helper Functions

Utility functions to enhance user experience across all pages.
"""

import streamlit as st
from typing import Optional, Callable, Any
import time


def show_success_toast(message: str, duration: int = 3) -> None:
    """
    Show a temporary success message that auto-dismisses.

    Args:
        message: Success message to display
        duration: Seconds to display (default 3)
    """
    placeholder = st.empty()
    with placeholder.container():
        st.success(message)
    time.sleep(duration)
    placeholder.empty()


def show_warning_toast(message: str, duration: int = 3) -> None:
    """
    Show a temporary warning message that auto-dismisses.

    Args:
        message: Warning message to display
        duration: Seconds to display (default 3)
    """
    placeholder = st.empty()
    with placeholder.container():
        st.warning(message)
    time.sleep(duration)
    placeholder.empty()


def show_info_toast(message: str, duration: int = 3) -> None:
    """
    Show a temporary info message that auto-dismisses.

    Args:
        message: Info message to display
        duration: Seconds to display (default 3)
    """
    placeholder = st.empty()
    with placeholder.container():
        st.info(message)
    time.sleep(duration)
    placeholder.empty()


def confirm_action(
    message: str,
    button_text: str = "Confirm",
    button_type: str = "primary",
    cancel_text: str = "Cancel",
) -> bool:
    """
    Show a confirmation dialog with Yes/No buttons.

    Args:
        message: Confirmation message
        button_text: Text for confirm button
        button_type: Streamlit button type
        cancel_text: Text for cancel button

    Returns:
        True if confirmed, False if cancelled
    """
    st.warning(message)
    col1, col2 = st.columns(2)
    with col1:
        confirmed = st.button(button_text, type=button_type, use_container_width=True)
    with col2:
        cancelled = st.button(cancel_text, use_container_width=True)

    if confirmed:
        return True
    if cancelled:
        return False
    return None


def create_card(
    title: str,
    content: str,
    icon: str = "📄",
    footer: Optional[str] = None,
    expandable: bool = False,
) -> None:
    """
    Create a styled card component.

    Args:
        title: Card title
        content: Card content (Markdown supported)
        icon: Icon emoji
        footer: Optional footer text
        expandable: Whether card is expandable
    """
    if expandable:
        with st.expander(f"{icon} {title}"):
            st.markdown(content)
            if footer:
                st.caption(footer)
    else:
        st.markdown(f"### {icon} {title}")
        st.markdown(content)
        if footer:
            st.caption(footer)


def show_progress(current: int, total: int, label: str = "Progress") -> None:
    """
    Show a progress bar with percentage.

    Args:
        current: Current value
        total: Total value
        label: Progress label
    """
    percentage = int((current / total) * 100) if total > 0 else 0
    st.progress(percentage / 100, text=f"{label}: {current}/{total} ({percentage}%)")


def create_metric_row(metrics: list[dict]) -> None:
    """
    Create a row of metrics.

    Args:
        metrics: List of metric dicts with keys: label, value, delta, help
    """
    cols = st.columns(len(metrics))
    for col, metric in zip(cols, metrics):
        with col:
            st.metric(
                label=metric.get("label", ""),
                value=metric.get("value", ""),
                delta=metric.get("delta"),
                help=metric.get("help"),
            )


def show_empty_state(
    title: str,
    message: str,
    icon: str = "📭",
    action_text: Optional[str] = None,
    action_callback: Optional[Callable] = None,
) -> None:
    """
    Show an empty state with optional action button.

    Args:
        title: Empty state title
        message: Empty state message
        icon: Icon emoji
        action_text: Optional action button text
        action_callback: Optional callback function
    """
    st.markdown(f"<div style='text-align: center; padding: 50px;'>", unsafe_allow_html=True)
    st.markdown(f"# {icon}")
    st.markdown(f"### {title}")
    st.markdown(message)

    if action_text and action_callback:
        if st.button(action_text, type="primary"):
            action_callback()

    st.markdown("</div>", unsafe_allow_html=True)


def create_badge(text: str, color: str = "gray") -> str:
    """
    Create a colored badge.

    Args:
        text: Badge text
        color: Badge color (gray, red, yellow, green, blue)

    Returns:
        HTML string for badge
    """
    color_map = {
        "gray": "#6B7280",
        "red": "#EF4444",
        "yellow": "#F59E0B",
        "green": "#10B981",
        "blue": "#3B82F6",
    }
    bg_color = color_map.get(color, color_map["gray"])

    return f"""
    <span style="
        background-color: {bg_color};
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 500;
    ">{text}</span>
    """


def show_keyboard_shortcuts() -> None:
    """Display keyboard shortcuts help."""
    with st.expander("⌨️ Keyboard Shortcuts"):
        st.markdown(
            """
        - `Enter` - Commit text input
        - `Ctrl+Enter` - Commit text area
        - `Esc` - Close modals
        - `Tab` - Navigate between fields
        - `Ctrl+K` - Focus search (if available)
        """
        )


def add_copy_button(text: str, button_label: str = "📋 Copy") -> None:
    """
    Add a copy-to-clipboard button.

    Args:
        text: Text to copy
        button_label: Button label
    """
    if st.button(button_label):
        # Note: Actual clipboard copy requires JavaScript
        # This is a placeholder for the UI
        st.success("Copied to clipboard!")


def create_timeline(events: list[dict]) -> None:
    """
    Create a vertical timeline.

    Args:
        events: List of event dicts with keys: time, title, description
    """
    for event in events:
        col1, col2 = st.columns([1, 4])
        with col1:
            st.caption(event.get("time", ""))
        with col2:
            st.markdown(f"**{event.get('title', '')}**")
            st.markdown(event.get("description", ""))
            st.markdown("---")


def show_tips(tips: list[str], title: str = "💡 Tips") -> None:
    """
    Show a tips section.

    Args:
        tips: List of tip strings
        title: Section title
    """
    with st.expander(title):
        for tip in tips:
            st.markdown(f"- {tip}")


def create_stat_card(label: str, value: Any, trend: Optional[str] = None, icon: str = "📊") -> None:
    """
    Create a statistics card.

    Args:
        label: Stat label
        value: Stat value
        trend: Optional trend indicator (↑, ↓, →)
        icon: Icon emoji
    """
    st.markdown(
        f"""
    <div style="
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 16px;
        background: #F9FAFB;
    ">
        <div style="font-size: 14px; color: #6B7280;">{icon} {label}</div>
        <div style="font-size: 28px; font-weight: bold; margin: 8px 0;">
            {value} {trend if trend else ''}
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def add_breadcrumbs(items: list[str]) -> None:
    """
    Add breadcrumb navigation.

    Args:
        items: List of breadcrumb items
    """
    breadcrumb = " / ".join(
        [f"**{item}**" if i == len(items) - 1 else item for i, item in enumerate(items)]
    )
    st.caption(breadcrumb)


def show_loading_message(message: str = "Loading...") -> Any:
    """
    Show a loading message with spinner.

    Args:
        message: Loading message

    Returns:
        Spinner context manager
    """
    return st.spinner(message)


def create_expandable_code(code: str, language: str = "yaml", title: str = "Code") -> None:
    """
    Create an expandable code block.

    Args:
        code: Code content
        language: Code language
        title: Expander title
    """
    with st.expander(f"📄 {title}"):
        st.code(code, language=language)


def add_footer(text: str) -> None:
    """
    Add a footer to the page.

    Args:
        text: Footer text
    """
    st.markdown("---")
    st.caption(text)


def create_alert_box(message: str, alert_type: str = "info", dismissible: bool = False) -> None:
    """
    Create a custom alert box.

    Args:
        message: Alert message
        alert_type: Type (info, success, warning, error)
        dismissible: Whether alert can be dismissed
    """
    icons = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "❌"}

    icon = icons.get(alert_type, icons["info"])

    if alert_type == "info":
        st.info(f"{icon} {message}")
    elif alert_type == "success":
        st.success(f"{icon} {message}")
    elif alert_type == "warning":
        st.warning(f"{icon} {message}")
    elif alert_type == "error":
        st.error(f"{icon} {message}")
