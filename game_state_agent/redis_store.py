"""Redis storage for game state persistence.

Usage - Query the latest game state from Redis:

    from game_state_agent.redis_store import GameStateStore

    store = GameStateStore()
    state = store.load()
    if state:
        print(f"Location: {state.player_location}")
        print(f"Inventory: {[item.name for item in state.inventory]}")
        print(f"Bosses defeated: {state.bosses_defeated}")
"""

import logging

import redis

from .config import GAME_STATE_KEY, REDIS_HOST, REDIS_PASSWORD, REDIS_PORT
from .models import GameState

logger = logging.getLogger(__name__)


class GameStateStore:
    """Stores and retrieves game state from Redis."""

    def __init__(self, client: redis.Redis | None = None):
        """Initialize the game state store.

        Args:
            client: Optional Redis client. If None, creates one from config.
        """
        self._client = client or redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD,
            decode_responses=True,
        )
        self._key = GAME_STATE_KEY

    def save(self, state: GameState) -> None:
        """Save the current game state to Redis.

        Args:
            state: The game state to persist.
        """
        json_data = state.model_dump_json()
        self._client.set(self._key, json_data)
        logger.debug(f"Saved game state to Redis: {self._key}")

    def load(self) -> GameState | None:
        """Load the latest game state from Redis.

        Returns:
            The game state if found, None otherwise.
        """
        json_data = self._client.get(self._key)
        if json_data is None:
            logger.debug("No game state found in Redis")
            return None

        state = GameState.model_validate_json(json_data)
        logger.debug(f"Loaded game state from Redis: {self._key}")
        return state

    def delete(self) -> bool:
        """Delete the stored game state.

        Returns:
            True if a key was deleted, False otherwise.
        """
        deleted = self._client.delete(self._key)
        return deleted > 0


def get_redis_client() -> redis.Redis:
    """Create a Redis client from configuration.

    Returns:
        Configured Redis client.
    """
    return redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD,
        decode_responses=True,
    )
