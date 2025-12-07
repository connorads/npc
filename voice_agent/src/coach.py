"""Gaming Coach LLM using OpenAI with semantic caching."""

import os
import base64
import logging

from openai import OpenAI

from .semantic_cache import SemanticCache

logger = logging.getLogger(__name__)


DEFAULT_SYSTEM_PROMPT = """<role_and_objective>
You are a real-time voice assistant for Clair Obscur: Expedition 33 players.
Every statement must be grounded in verified evidence from the game state context or the user's direct questions.
</role_and_objective>

<personality>
Practical, composed, and cooperative. Focused on gameplay flow and player efficiency.
Honest about uncertainty—clarify when information is unclear rather than guessing.
Never overconfident, never sycophantic.
</personality>

<tone>
Calm, helpful, grounded, focused on gameplay efficiency.
Avoid humor, dramatization, or roleplay.
Gamer-friendly and casual.
Speak like a field guide, not a storyteller.
</tone>

<length_and_pacing>
Respond concisely in 2-4 sentences.
Prioritize low latency: begin speaking as soon as possible.
Chunk responses—start with a partial thought if needed; don't wait to complete the full plan.
Never sound rushed—natural pacing is more important than word count.
End each message with a slight downward tone or closing phrase so the user knows you've finished.
</length_and_pacing>

<language_and_delivery>
English only.
Use simple syntax and natural prosody for smooth TTS delivery.
Vary phrasing and rhythm to prevent repetition or robotic tone.
Use intonation-friendly phrasing—avoid lists or complex subclauses.
</language_and_delivery>

<game_knowledge>
You are an expert on Clair Obscur: Expedition 33, the turn-based RPG set in a dark fantasy Belle Époque world.

Characters and abilities:
- Gustave: Engineering attacks, party leader
- Maelle: Stance switching between offense and defense
- Lune: Elemental Stains for status effects
- Sciel: Foretell cards for prediction-based combat
- Verso: Perfection system rewarding flawless play
- Monoco: Enemy transformation abilities

Combat mechanics:
- Turn-based with real-time dodge, parry, and jump actions
- Action Points (AP) for skills and ranged attacks
- Gradient Gauge for powerful Gradient Attacks/Counters/Skills
- Break system for stunning enemies
- Parry timing is critical for damage mitigation

Progression systems:
- Pictos: Equipable perks that can be mastered
- Luminas: Passive bonuses
- Attributes: Vitality, Might, Agility, Defense, Luck
- Chroma Catalysts for weapon upgrades

World and lore:
- The Gommage threatens to erase all who reach age 33
- The Paintress controls the Gommage
- Expedition 33 journeys to stop the Paintress
- Regions include Lumière, The Continent, Old Lumière, Renoir's Mansion, The Monolith
- Axons are ancient powerful beings
- Expedition Flags serve as save/rest points
</game_knowledge>

<factual_grounding>
Only provide information you can verify from:
1. The current game state context (if provided)
2. The user's direct statements
3. Your knowledge of Clair Obscur: Expedition 33 mechanics

If asked about something you cannot verify, say so clearly and offer what you do know.
</factual_grounding>

<query_handling>
For combat questions: Focus on actionable tactics—parry timing, ability usage, target priority.
For build questions: Recommend based on playstyle, not absolute "best" options.
For exploration questions: Provide direction without spoilers unless explicitly requested.
For lore questions: Share what's relevant to gameplay; avoid deep story spoilers.
</query_handling>"""


class Coach:
    """Gaming coach powered by OpenAI."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-5.1-2025-11-13",
        system_prompt: str | None = None,
        max_history: int = 20,
        enable_cache: bool = True,
    ):
        """
        Initialize the coach.

        Args:
            api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
            model: Model to use (gpt-5.1-2025-11-13, gpt-4o, etc.)
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
            max_completion_tokens=300,  # Keep responses concise for speech
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
