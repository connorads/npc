"""Logging configuration with timestamped log files."""

import logging
import json
from datetime import datetime
from pathlib import Path


def setup_logging(log_dir: str = "logs") -> Path:
    """Configure logging to write to timestamped file and console.
    
    Args:
        log_dir: Directory to store log files
        
    Returns:
        Path to the created log file
    """
    # Create logs directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Create timestamped log file name
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_path / f"{timestamp}.log"
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # File handler - captures everything including DEBUG
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    # Console handler - INFO and above
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # Suppress verbose logging from httpx and openai libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    
    logging.info(f"Logging initialized. Log file: {log_file}")
    
    return log_file


def log_openai_request(logger: logging.Logger, model: str, messages: list, response_format: type | None = None) -> None:
    """Log an OpenAI API request (concise version).
    
    Args:
        logger: Logger instance to use
        model: Model name being used
        messages: Messages being sent to the API
        response_format: The response format type if using structured outputs
    """
    # Extract just the essential info from messages
    message_summary = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        
        if isinstance(content, str):
            # System or simple user message - truncate if too long
            preview = content[:200] + "..." if len(content) > 200 else content
            message_summary.append({"role": role, "content": preview})
        elif isinstance(content, list):
            # Multimodal message (text + image)
            parts = []
            for item in content:
                if item.get("type") == "text":
                    parts.append(f"text: {item.get('text', '')}")
                elif item.get("type") == "image_url":
                    parts.append("image: [screenshot]")
            message_summary.append({"role": role, "content": " | ".join(parts)})
    
    logger.debug(f"OpenAI Request -> model={model}, response_format={response_format.__name__ if response_format else None}")
    for msg in message_summary:
        logger.debug(f"  [{msg['role']}] {msg['content']}")


def log_openai_response(logger: logging.Logger, response: object, parsed_result: object | None = None) -> None:
    """Log an OpenAI API response (concise version).
    
    Args:
        logger: Logger instance to use
        response: Raw response from OpenAI
        parsed_result: Parsed structured output if applicable
    """
    # Log usage stats
    usage = getattr(response, "usage", None)
    if usage:
        logger.debug(
            f"OpenAI Response -> tokens: {usage.prompt_tokens} prompt + "
            f"{usage.completion_tokens} completion = {usage.total_tokens} total"
        )
    
    # Log parsed result
    if parsed_result and hasattr(parsed_result, "model_dump"):
        result_dict = parsed_result.model_dump()
        logger.debug(f"Parsed result: {json.dumps(result_dict, default=str)}")


def log_game_state(logger: logging.Logger, state: object, update: object | None = None) -> None:
    """Log the current game state.
    
    Args:
        logger: Logger instance to use
        state: Current GameState object
        update: The StateUpdate that was applied (if any)
    """
    if hasattr(state, "model_dump"):
        state_dict = state.model_dump()
    else:
        state_dict = str(state)
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "game_state": state_dict
    }
    
    if update:
        if hasattr(update, "model_dump"):
            log_entry["applied_update"] = update.model_dump()
        else:
            log_entry["applied_update"] = str(update)
    
    logger.info(f"Game State Update:\n{json.dumps(log_entry, indent=2, default=str)}")
