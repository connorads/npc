"""Context provider for enriching LLM queries with game state and NPC data."""

import os

import logfire
import numpy as np
import redis
from openai import OpenAI
from redis.commands.search.query import Query

from game_state_agent.models import GameState
from game_state_agent.redis_store import GameStateStore

# Redis and embedding configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
NPC_INDEX_NAME = "idx:npcs"
EMBEDDING_MODEL = "text-embedding-3-small"
SIMILARITY_THRESHOLD = 0.3  # Minimum similarity score (1 - distance) to include results


class ContextProvider:
    """Provides game context from Redis for LLM queries."""

    def __init__(
        self,
        openai_client: OpenAI | None = None,
        redis_client: redis.Redis | None = None,
    ):
        """Initialize the context provider.

        Args:
            openai_client: Optional OpenAI client for embeddings.
            redis_client: Optional Redis client. Creates one from env vars if not provided.
        """
        self._openai = openai_client or OpenAI()
        self._redis = redis_client or redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD,
            decode_responses=True,
        )
        self._game_store = GameStateStore(client=self._redis)

    @logfire.instrument("get_game_state")
    def get_game_state(self) -> GameState | None:
        """Fetch the current game state from Redis.

        Returns:
            The current GameState if available, None otherwise.
        """
        try:
            return self._game_store.load()
        except Exception as e:
            logfire.warn("Failed to load game state from Redis", error=str(e))
            return None

    @logfire.instrument("get_embedding")
    def _get_embedding(self, text: str) -> list[float]:
        """Get embedding for text using OpenAI API."""
        response = self._openai.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text,
        )
        return response.data[0].embedding

    @logfire.instrument("search_npcs")
    def search_npcs(self, query: str, top_k: int = 3) -> list[dict]:
        """Search for NPCs relevant to the query using semantic search.

        Args:
            query: Natural language query to search for.
            top_k: Maximum number of results to return.

        Returns:
            List of NPC dictionaries with relevant fields, filtered by similarity threshold.
        """
        try:
            query_embedding = np.array(self._get_embedding(query), dtype=np.float32)

            search_query = (
                Query(f"(*)=>[KNN {top_k} @embedding $query_vec AS score]")
                .sort_by("score")
                .return_fields(
                    "score",
                    "name",
                    "race",
                    "role",
                    "region",
                    "description",
                    "how_to_beat_tips",
                )
                .dialect(2)
            )

            results = self._redis.ft(NPC_INDEX_NAME).search(
                search_query, {"query_vec": query_embedding.tobytes()}
            )

            npcs = []
            for doc in results.docs:
                # Score is distance, so similarity = 1 - score
                similarity = 1 - float(doc.score)
                if similarity >= SIMILARITY_THRESHOLD:
                    npcs.append(
                        {
                            "name": doc.name,
                            "race": doc.race,
                            "role": doc.role,
                            "region": doc.region,
                            "description": getattr(doc, "description", ""),
                            "tips": getattr(doc, "how_to_beat_tips", ""),
                            "similarity": similarity,
                        }
                    )

            logfire.info(
                "NPC search completed",
                query_length=len(query),
                top_k=top_k,
                results_count=len(npcs),
            )
            return npcs

        except Exception as e:
            logfire.warn("Failed to search NPCs", error=str(e))
            return []

    def format_context(
        self,
        game_state: GameState | None,
        npc_results: list[dict],
    ) -> str | None:
        """Format game state and NPC results into a context string for the LLM.

        Args:
            game_state: Current game state from Redis.
            npc_results: List of relevant NPCs from semantic search.

        Returns:
            Formatted context string, or None if no context is available.
        """
        parts = []

        # Format game state
        if game_state:
            state_lines = [
                "<game_state>",
                f"Location: {game_state.player_location}",
                f"Active Party: {', '.join(game_state.active_party)}",
            ]

            if game_state.current_boss:
                state_lines.append(
                    f"Current Boss: {game_state.current_boss.name} "
                    f"(HP: {game_state.current_boss.hp_percentage:.0f}%)"
                )

            if game_state.last_flag:
                state_lines.append(f"Last Flag: {game_state.last_flag}")

            if game_state.gradient_gauge > 0:
                state_lines.append(f"Gradient Gauge: {game_state.gradient_gauge:.0f}%")

            if game_state.bosses_defeated:
                state_lines.append(
                    f"Bosses Defeated: {', '.join(game_state.bosses_defeated)}"
                )

            if game_state.at_camp:
                state_lines.append("Status: At camp")

            state_lines.append("</game_state>")
            parts.append("\n".join(state_lines))

        # Format NPC results
        if npc_results:
            npc_lines = ["<relevant_npcs>"]
            for i, npc in enumerate(npc_results, 1):
                npc_lines.append(
                    f"{i}. {npc['name']} ({npc['role']}) - {npc['region']}"
                )
                if npc["description"]:
                    # Truncate long descriptions
                    desc = npc["description"]
                    if len(desc) > 200:
                        desc = desc[:200] + "..."
                    npc_lines.append(f"   {desc}")
                if npc["tips"]:
                    tips = npc["tips"]
                    if len(tips) > 200:
                        tips = tips[:200] + "..."
                    npc_lines.append(f"   Tips: {tips}")
            npc_lines.append("</relevant_npcs>")
            parts.append("\n".join(npc_lines))

        if not parts:
            return None

        return "\n\n".join(parts)

    def get_context_for_query(self, query: str, top_k: int = 3) -> str | None:
        """Get formatted context for a user query.

        This is a convenience method that fetches game state and searches NPCs,
        then formats everything into a single context string.

        Args:
            query: The user's question.
            top_k: Maximum number of NPC results.

        Returns:
            Formatted context string, or None if no context is available.
        """
        game_state = self.get_game_state()
        npc_results = self.search_npcs(query, top_k=top_k)
        return self.format_context(game_state, npc_results)
