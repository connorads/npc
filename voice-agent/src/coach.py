"""Gaming Coach LLM using OpenAI."""

import os
import base64

from openai import OpenAI


DEFAULT_SYSTEM_PROMPT = """You are a friendly and knowledgeable gaming coach. You help players improve their skills, understand game mechanics, and have more fun.

Keep your responses concise (2-3 sentences) since they will be spoken aloud. Be encouraging but direct. If you don't know something specific about a game, say so and offer general advice instead."""


class Coach:
    """Gaming coach powered by OpenAI."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4.1",
        system_prompt: str | None = None,
        max_history: int = 20,
    ):
        """
        Initialize the coach.

        Args:
            api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
            model: Model to use (gpt-4.1, gpt-4o, etc.)
            system_prompt: Custom system prompt for the coach.
            max_history: Maximum number of messages to keep in history.
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

        return assistant_message

    def clear_history(self) -> None:
        """Clear conversation history."""
        self._history = []
