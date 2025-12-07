# Tinkerers Hack - Gaming AI Agents

Two AI agents for gaming: a voice coach and a game state tracker, plus Redis-based game knowledge.

## Quick Start

```bash
# Install dependencies
uv sync

# Copy and configure environment variables
cp .env.example .env
# Edit .env with your API keys
```

## Running the Agents

### Voice Agent

Push-to-talk voice assistant. Hold Right Option, speak, release to get a spoken response.

```bash
uv run voice-agent
```

| Action | Key |
|--------|-----|
| Start recording | Hold **Right Option** |
| Get response | Release **Right Option** |
| Quit | Press **ESC** |

### Game State Agent

Tracks game state by analyzing screenshots periodically.

```bash
uv run game-state
```

Press `Ctrl+C` to stop.

### Running Both Simultaneously

Open two terminal windows and run each agent:

```bash
# Terminal 1
uv run game-state

# Terminal 2
uv run voice-agent
```

Or run both in one terminal (game-state in background):

```bash
uv run game-state & uv run voice-agent
```

## Redis Game Knowledge Database

### Prerequisites

**Redis Server** must be running locally:

```bash
# macOS
brew install redis
brew services start redis
```

### Setup

```bash
# Populate the Redis database
uv run python redis_setup/setup_redis.py
```

### Query Examples

```bash
# Run example queries
uv run python -i query_game.py
```

```python
from query_game import semantic_search, filter_search, get_entry

# Semantic search - natural language queries
semantic_search("boss with shields and flowers")
semantic_search("how to cross the sea")

# Filter by metadata
filter_search("@area:{Gestral Village}")
filter_search("@type:{Merchant}")
filter_search("@type:{Boss}")

# Combined: semantic + filter
semantic_search("secret items", filter_expr="@area:{Gestral Village}")

# Get specific entry by ID
get_entry("goblu")
```

### Game Knowledge Schema

Entries in `data/game_knowledge.json` have:
- `id`, `name`, `type`, `area`
- `location`, `description`, `tips`
- `rewards`, `weakness`, `resistance`, `requirements`

### Adding Game Knowledge

**Flat format**: Edit `data/game_knowledge.json` directly.

**Zone format**: Use the transformer to convert nested zone data:
```bash
# Preview transformed data
uv run python -m redis_setup.transform_zone zone_data.json

# Append to game_knowledge.json
uv run python -m redis_setup.transform_zone zone_data.json --append
```

Then re-run setup:
```bash
uv run python redis_setup/setup_redis.py
```

## Requirements

- Python 3.13+
- macOS (for Right Option key detection in voice agent)
- OpenAI API key
- ElevenLabs API key (for voice agent)
- Redis (for game knowledge database)

### macOS Permissions

Your terminal app needs:
- **Accessibility** permission (System Settings > Privacy & Security > Accessibility)
- **Microphone** permission (System Settings > Privacy & Security > Microphone)

## Environment Variables

Create a `.env` file in the project root:

| Variable | Required By | Description |
|----------|-------------|-------------|
| `OPENAI_API_KEY` | Both | OpenAI API key |
| `ELEVENLABS_API_KEY` | Voice Agent | ElevenLabs API key |
| `LOGFIRE_TOKEN` | Voice Agent | Logfire token (or run `uv run logfire auth`) |

## Observability

The voice agent is instrumented with [Pydantic Logfire](https://logfire.pydantic.dev) for tracing and observability.

### Setup

```bash
# Authenticate with Logfire (first time only)
uv run logfire auth
```

Or set `LOGFIRE_TOKEN` in your `.env` file.

### View Traces

After running the voice agent, view traces at https://logfire.pydantic.dev

Traces include:
- Full voice interaction flow timing
- OpenAI API calls (chat completions, embeddings)
- Redis operations
- ElevenLabs STT/TTS calls
- Semantic cache hits/misses
- Screenshot capture timing

## Project Structure

```
game_state_agent/     # Game state tracking agent
  main.py             # Entry point
  analyzer.py         # LLM screenshot analysis
  capture.py          # Screen capture
  state_manager.py    # State tracking

voice_agent/          # Voice coach agent
  main.py             # Entry point / orchestrator
  src/
    ptt.py            # Push-to-talk handler
    audio.py          # Recording & playback
    screenshot.py     # Screen capture
    stt.py            # Speech-to-text
    tts.py            # Text-to-speech
    coach.py          # LLM coach

redis_setup/          # Redis vector database setup
  setup_redis.py      # Populates Redis with embeddings
  transform_zone.py   # Convert zone JSON to flat format

data/
  game_knowledge.json # Game knowledge seed data

query_game.py         # Query utilities
```
