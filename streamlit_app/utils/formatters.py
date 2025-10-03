"""
Display formatters for the UI
"""

from datetime import datetime
from typing import Any, Dict, Optional
import json


class Formatters:
    """Utility functions for formatting data for display"""

    @staticmethod
    def truncate_text(text: str, max_length: int = 200, suffix: str = "...") -> str:
        """
        Truncate text to a maximum length

        Args:
            text: Text to truncate
            max_length: Maximum length
            suffix: Suffix to add if truncated

        Returns:
            Truncated text
        """
        if not text:
            return ""

        if len(text) <= max_length:
            return text

        return text[: max_length - len(suffix)] + suffix

    @staticmethod
    def format_timestamp(timestamp: Any) -> str:
        """
        Format a timestamp for display

        Args:
            timestamp: Timestamp (datetime, string, or None)

        Returns:
            Formatted timestamp string
        """
        if timestamp is None:
            return "N/A"

        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except:
                return timestamp

        if isinstance(timestamp, datetime):
            return timestamp.strftime("%Y-%m-%d %H:%M:%S")

        return str(timestamp)

    @staticmethod
    def format_template_id(template_id: str) -> str:
        """
        Format template ID for display (remove 'bmad.' prefix)

        Args:
            template_id: Full template ID

        Returns:
            Formatted template ID
        """
        if not template_id:
            return ""

        # Remove 'bmad.' prefix for cleaner display
        if template_id.startswith("bmad."):
            parts = template_id.split(".")
            if len(parts) >= 3:
                return f"{parts[1]}.{parts[2]}"

        return template_id

    @staticmethod
    def format_agent_type(agent_type: str) -> str:
        """
        Format agent type for display (convert kebab-case to Title Case)

        Args:
            agent_type: Agent type identifier

        Returns:
            Formatted agent type
        """
        if not agent_type:
            return ""

        # Convert kebab-case to Title Case
        words = agent_type.replace("-", " ").split()
        return " ".join(word.capitalize() for word in words)

    @staticmethod
    def format_metadata(metadata: Optional[Dict]) -> str:
        """
        Format metadata dictionary for display

        Args:
            metadata: Metadata dictionary

        Returns:
            Formatted JSON string
        """
        if not metadata:
            return "{}"

        try:
            return json.dumps(metadata, indent=2)
        except:
            return str(metadata)

    @staticmethod
    def format_section_count(count: int) -> str:
        """
        Format section count with proper pluralization

        Args:
            count: Number of sections

        Returns:
            Formatted string
        """
        if count == 1:
            return "1 section"
        return f"{count} sections"

    @staticmethod
    def format_update_count(count: int, threshold: int = 8) -> tuple[str, str]:
        """
        Format update count with color indicator

        Args:
            count: Number of updates
            threshold: Warning threshold

        Returns:
            Tuple of (formatted_string, color)
        """
        if count >= threshold + 2:
            return f"⚠️ {count} updates", "error"
        elif count >= threshold:
            return f"⚡ {count} updates", "warning"
        else:
            return f"{count} updates", "normal"

    @staticmethod
    def format_priority(priority: str) -> tuple[str, str]:
        """
        Format priority with emoji and color

        Args:
            priority: Priority level (high, medium, low)

        Returns:
            Tuple of (formatted_string, color)
        """
        priority_map = {
            "high": ("🔴 High", "error"),
            "medium": ("🟡 Medium", "warning"),
            "low": ("🟢 Low", "success"),
        }

        return priority_map.get(priority.lower(), (priority, "normal"))

    @staticmethod
    def format_usage_type(usage_type: str) -> str:
        """
        Format usage type with emoji

        Args:
            usage_type: Usage type identifier

        Returns:
            Formatted string with emoji
        """
        usage_map = {
            "session": "💬 Session",
            "project": "📁 Project",
            "task": "✅ Task",
            "analysis": "🔍 Analysis",
            "planning": "📋 Planning",
        }

        return usage_map.get(usage_type.lower(), usage_type.capitalize())

    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """
        Format file size in human-readable format

        Args:
            size_bytes: Size in bytes

        Returns:
            Formatted string (e.g., "1.5 KB")
        """
        for unit in ["B", "KB", "MB", "GB"]:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"

    @staticmethod
    def format_external_id(external_id: Any) -> str:
        """
        Format external ID for display

        Args:
            external_id: External ID (can be string, UUID, or int)

        Returns:
            Formatted string
        """
        if external_id is None:
            return "N/A"

        return str(external_id)
