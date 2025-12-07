"""Configuration and OpenAI client setup."""

import os
from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

# Load environment variables from root .env file
load_dotenv(find_dotenv())

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-5.1-2025-11-13"

# Screenshot interval in seconds
CAPTURE_INTERVAL = 5

# Redis configuration (loaded from .env via existing load_dotenv call)
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
GAME_STATE_KEY = "game:state:latest"


def get_openai_client() -> OpenAI:
    """Get configured OpenAI client."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    return OpenAI(api_key=OPENAI_API_KEY)
