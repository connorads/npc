"""Configuration and OpenAI client setup."""

import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-5.1-2025-11-13"

# Screenshot interval in seconds
CAPTURE_INTERVAL = 5

def get_openai_client() -> OpenAI:
    """Get configured OpenAI client."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    return OpenAI(api_key=OPENAI_API_KEY)

