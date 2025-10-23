"""
Utility functions for logging AI agent runs.

Provides functions to save agent execution details including messages and usage statistics.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Optional
from pydantic_ai import RunUsage, AgentRunResult

logger = logging.getLogger(__name__)

DEFAULT_LOG_DIR = Path("agent_logs")


def ensure_log_directory(log_dir: Optional[Path] = None) -> Path:
    """
    Ensure the log directory exists.

    Args:
        log_dir: Directory to store logs. Defaults to 'agent_logs' in current directory.

    Returns:
        Path to the log directory
    """
    dir_path = log_dir or DEFAULT_LOG_DIR
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def save_agent_run(
    result: AgentRunResult,
    agent_name: str,
    run_id: str,
    log_dir: Optional[Path] = None,
    include_usage: bool = True,
    include_messages: bool = True,
) -> dict[str, Optional[Path]]:
    """
    Save agent run details to files.

    Args:
        result: The AgentRunResult from agent execution
        agent_name: Name of the agent (e.g., 'memorizer', 'er_extractor')
        run_id: Unique identifier for this run (e.g., external_id, task_id)
        log_dir: Directory to store logs. Defaults to 'agent_logs'
        include_usage: Whether to save usage statistics
        include_messages: Whether to save message history

    Returns:
        Dictionary with paths to saved files: {'usage': Path, 'messages': Path}
    """
    log_dir = ensure_log_directory(log_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    saved_files = {
        "usage": None,
        "messages": None,
    }

    # Save usage statistics
    if include_usage:
        try:
            usage = result.usage()
            usage_data = {
                "agent_name": agent_name,
                "run_id": run_id,
                "timestamp": timestamp,
                "requests": usage.requests,
                "tool_calls": usage.tool_calls,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "total_tokens": usage.total_tokens,
                "cache_read_tokens": usage.cache_read_tokens,
                "cache_write_tokens": usage.cache_write_tokens,
                "details": usage.details,
            }

            usage_filename = f"{agent_name}_{run_id}_{timestamp}_usage.json"
            usage_path = log_dir / usage_filename

            with open(usage_path, "w", encoding="utf-8") as f:
                json.dump(usage_data, f, indent=2, ensure_ascii=False)

            saved_files["usage"] = usage_path
            logger.debug(f"Saved usage statistics to {usage_path}")

        except Exception as e:
            logger.warning(f"Failed to save usage statistics: {e}")

    # Save message history
    if include_messages:
        try:
            json_msg: bytes = result.all_messages_json()
            json_str = json_msg.decode("utf-8")

            messages_filename = f"{agent_name}_{run_id}_{timestamp}_messages.json"
            messages_path = log_dir / messages_filename

            # Use utf-8 encoding to handle special characters like arrows (→)
            with open(messages_path, "w", encoding="utf-8") as f:
                f.write(json_str)

            saved_files["messages"] = messages_path
            logger.debug(f"Saved message history to {messages_path}")

        except Exception as e:
            logger.warning(f"Failed to save message history: {e}")

    return saved_files


def log_usage_summary(usage: RunUsage, agent_name: str) -> None:
    """
    Log a summary of usage statistics.

    Args:
        usage: Usage statistics from agent run
        agent_name: Name of the agent
    """
    logger.info(
        f"[{agent_name}] Usage: {usage.requests} requests, {usage.tool_calls} tool calls, "
        f"{usage.input_tokens} input tokens, {usage.output_tokens} output tokens, "
        f"{usage.total_tokens} total tokens"
    )

    if usage.cache_read_tokens > 0:
        logger.info(f"[{agent_name}] Cache: {usage.cache_read_tokens} tokens read")

    if usage.details:
        logger.debug(f"[{agent_name}] Usage details: {usage.details}")


