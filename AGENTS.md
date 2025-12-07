# AGENTS.md

## Build & Run

```bash
uv sync                              # Install dependencies
uv run game-state                    # Run game state agent
uv run voice-agent                   # Run voice agent
uv run python redis_setup/setup_redis.py  # Setup Redis NPC database
uv run python -i query_npcs.py       # Interactive NPC queries
```

## Code Style

- **Python**: 3.13+, type hints required (use `X | None` not `Optional[X]`)
- **Imports**: stdlib → third-party → local, one blank line between groups
- **Models**: Use Pydantic `BaseModel` with `Field()` descriptions for structured data
- **Naming**: snake_case for functions/variables, PascalCase for classes, UPPER_CASE for constants
- **Private**: Prefix with underscore (`self._client`, `self._history`)
- **Docstrings**: Module-level required, Google-style for classes/functions
- **Error handling**: Raise `ValueError` for invalid inputs, let API errors propagate
- **Env vars**: Load via `python-dotenv`, check required vars at startup with clear error messages

## Project Structure

- `game_state_agent/` - Screenshot analysis, game state tracking (Pydantic models)
- `voice_agent/` - PTT voice assistant (imports as `voice_agent.src.*`)
- `redis_setup/` - NPC vector database setup (standalone script)
