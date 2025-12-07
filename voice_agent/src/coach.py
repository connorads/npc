"""Gaming Coach LLM using OpenAI with semantic caching."""

import os
import base64
import logging

from openai import OpenAI

from .semantic_cache import SemanticCache

logger = logging.getLogger(__name__)


DEFAULT_SYSTEM_PROMPT = """You are a friendly and knowledgeable gaming coach specializing in Clair Obscur: Expedition 33, the critically acclaimed turn-based RPG set in a dark fantasy Belle Époque world.

You help players master the game's unique combat system combining turn-based mechanics with real-time dodge, parry, and jump actions. You understand:
- Character abilities: Gustave's engineering attacks, Maelle's Stance switching, Lune's elemental Stains, Sciel's Foretell cards, Verso's Perfection system, and Monoco's enemy transformations
- Combat mechanics: Action Points, Gradient Attacks/Counters/Skills, Breaking enemies, parry timing, and status effects
- Progression systems: Pictos, Luminas, attribute allocation (Vitality, Might, Agility, Defense, Luck), and weapon upgrades via Chroma Catalysts
- Story and lore: The Gommage, the Paintress, Expedition 33's journey, and the mysteries of the Canvas

Keep your responses concise (2-3 sentences) since they will be spoken aloud. Be encouraging but direct. Help players with combat strategies, boss fights, character builds, and exploration tips."""


class Coach:
    """Gaming coach powered by OpenAI."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4.1",
        system_prompt: str | None = None,
        max_history: int = 20,
        enable_cache: bool = True,
    ):
        """
        Initialize the coach.

        Args:
            api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
            model: Model to use (gpt-4.1, gpt-4o, etc.)
            system_prompt: Custom system prompt for the coach.
            max_history: Maximum number of messages to keep in history.
            enable_cache: Whether to enable semantic caching.
        """
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key."
            )
        self._client = OpenAI(api_key=self._api_key)
        self._model = model
        self._system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self._max_history = max_history
        self._history: list[dict] = []
        self._cache = SemanticCache() if enable_cache else None

    def get_response(
        self,
        user_message: str,
        screenshot: bytes | None = None,
    ) -> str:
        """
        Get coaching response.

        Args:
            user_message: The user's transcribed question.
            screenshot: Optional screenshot bytes (JPEG) for vision models.

        Returns:
            Coach's response text.
        """
        if not user_message.strip():
            return "I didn't catch that. Could you repeat your question?"

        # Try semantic cache first (only for text-only queries without screenshots)
        if not screenshot and self._cache and self._cache.enabled:
            cached_response = self._cache.search(user_message)
            if cached_response:
                logger.info(f"Cache hit for: {user_message[:50]}...")
                # Add to history so conversation context stays consistent
                self._history.append({"role": "user", "content": user_message})
                self._history.append({"role": "assistant", "content": cached_response})
                if len(self._history) > self._max_history:
                    self._history = self._history[-self._max_history :]
                return cached_response

        # Build the user message content
        if screenshot:
            # Vision-enabled request with image
            b64_image = base64.b64encode(screenshot).decode("utf-8")
            content = [
                {"type": "text", "text": user_message},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
                },
            ]
        else:
            content = user_message

        # Add user message to history
        self._history.append({"role": "user", "content": content})

        # Trim history if needed
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history :]

        # Build messages with system prompt
        messages = [
            {"role": "system", "content": self._system_prompt},
            *self._history,
        ]

        # Get response from OpenAI
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=300,  # Keep responses concise for speech
        )

        assistant_message = response.choices[0].message.content or ""

        # Add assistant response to history
        self._history.append({"role": "assistant", "content": assistant_message})

        # Store in cache for future queries (only text-only queries)
        if not screenshot and self._cache and self._cache.enabled:
            self._cache.store(user_message, assistant_message)

        return assistant_message

    def clear_history(self) -> None:
        """Clear conversation history."""
        self._history = []
