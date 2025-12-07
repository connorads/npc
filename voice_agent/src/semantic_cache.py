"""Semantic caching for LLM responses using LangCache."""

import os

import logfire


class SemanticCache:
    """Semantic cache for LLM responses using LangCache."""

    def __init__(
        self,
        server_url: str | None = None,
        cache_id: str | None = None,
        api_key: str | None = None,
        similarity_threshold: float = 0.9,
    ):
        """
        Initialize semantic cache.

        Args:
            server_url: LangCache server URL. Falls back to LANGCACHE_SERVER_URL env var.
            cache_id: Cache ID. Falls back to LANGCACHE_CACHE_ID env var.
            api_key: LangCache API key. Falls back to LANGCACHE_API_KEY env var.
            similarity_threshold: Minimum similarity score to consider a cache hit (0.0-1.0).
        """
        self._server_url = server_url or os.environ.get("LANGCACHE_SERVER_URL")
        self._cache_id = cache_id or os.environ.get("LANGCACHE_CACHE_ID")
        self._api_key = api_key or os.environ.get("LANGCACHE_API_KEY")
        self._similarity_threshold = similarity_threshold
        self._client = None
        self._enabled = False

        if not all([self._server_url, self._cache_id, self._api_key]):
            logfire.warn(
                "LangCache not configured. Set LANGCACHE_SERVER_URL, LANGCACHE_CACHE_ID, "
                "and LANGCACHE_API_KEY env vars to enable semantic caching."
            )
            return

        try:
            from langcache import LangCache

            self._client = LangCache(
                server_url=self._server_url,
                cache_id=self._cache_id,
                api_key=self._api_key,
            )
            self._enabled = True
            logfire.info("Semantic cache enabled with LangCache")
        except ImportError:
            logfire.warn("langcache package not installed. Semantic caching disabled.")
        except Exception as e:
            logfire.warn("Failed to initialize LangCache", error=str(e))

    @property
    def enabled(self) -> bool:
        """Check if semantic caching is enabled."""
        return self._enabled

    @logfire.instrument("langcache.search")
    def search(self, prompt: str) -> str | None:
        """
        Search for a cached response.

        Args:
            prompt: The user prompt to search for.

        Returns:
            Cached response if found above threshold, None otherwise.
        """
        if not self._enabled or not self._client:
            return None

        try:
            result = self._client.search(prompt=prompt)
            if result and result.get("score", 0) >= self._similarity_threshold:
                cached_response = result.get("response")
                logfire.info(
                    "Cache hit",
                    prompt_length=len(prompt),
                    similarity_score=result.get("score", 0),
                    cache_hit=True,
                )
                return cached_response
            logfire.debug(
                "Cache miss",
                prompt_length=len(prompt),
                cache_hit=False,
            )
            return None
        except Exception as e:
            logfire.warn("Cache search failed", error=str(e))
            return None

    @logfire.instrument("langcache.store")
    def store(self, prompt: str, response: str) -> bool:
        """
        Store a response in the cache.

        Args:
            prompt: The user prompt.
            response: The LLM response to cache.

        Returns:
            True if stored successfully, False otherwise.
        """
        if not self._enabled or not self._client:
            return False

        try:
            self._client.set(prompt=prompt, response=response)
            logfire.debug(
                "Cached response",
                prompt_length=len(prompt),
                response_length=len(response),
            )
            return True
        except Exception as e:
            logfire.warn("Cache store failed", error=str(e))
            return False

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup if needed."""
        pass
