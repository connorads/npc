# Gaming Coach Voice Agent

A push-to-talk voice assistant for gaming. Hold Right Option, speak a question, release, and get a spoken response from an AI coach.

## Quick Start

```bash
# Install dependencies
uv sync

# Copy and configure environment variables
cp .env.example .env
# Edit .env with your API keys

# Run the agent
uv run python main.py
```

## Usage

| Action | Key |
|--------|-----|
| Start recording | Hold **Right Option** |
| Get response | Release **Right Option** |
| Quit | Press **ESC** |

## Requirements

- Python 3.13+
- macOS (for Right Option key detection)
- OpenAI API key
- ElevenLabs API key

### macOS Permissions

Your terminal app needs:
- **Accessibility** permission (System Settings → Privacy & Security → Accessibility)
- **Microphone** permission (System Settings → Privacy & Security → Microphone)

## Architecture

```
main.py              # Orchestrator - ties everything together
src/
├── ptt.py           # Push-to-talk keyboard handler (pynput)
├── audio.py         # Recording & playback (sounddevice)
├── screenshot.py    # Screen capture (mss + PIL)
├── stt.py           # Speech-to-text (ElevenLabs Scribe)
├── tts.py           # Text-to-speech (ElevenLabs)
└── coach.py         # LLM coach (OpenAI gpt-4.1)
```

## Configuration

Environment variables (in `.env`):

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key |
| `ELEVENLABS_API_KEY` | ElevenLabs API key |

## How It Works

1. **Press Right Option** → Starts recording audio + captures screenshot
2. **Release Right Option** → Stops recording and triggers pipeline:
   - Audio → ElevenLabs STT → transcript
   - Transcript + screenshot → OpenAI GPT-4.1 → response
   - Response → ElevenLabs TTS → audio playback

The screenshot is passed to the LLM for game-aware context (vision support).

## Tech Stack

- **PTT/Keyboard**: pynput
- **Audio**: sounddevice + numpy
- **Screenshot**: mss + Pillow  
- **STT**: ElevenLabs Scribe
- **LLM**: OpenAI GPT-4.1 (with vision)
- **TTS**: ElevenLabs Turbo v2.5
