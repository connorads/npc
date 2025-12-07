"""Gaming Coach Voice Agent - Main orchestrator."""

import sys
import os

from dotenv import load_dotenv

from src.ptt import PTTHandler
from src.audio import AudioRecorder, AudioPlayer
from src.screenshot import ScreenCapture
from src.stt import SpeechToText
from src.tts import TextToSpeech
from src.coach import Coach


class GamingCoach:
    """Main orchestrator for the gaming coach voice agent."""

    def __init__(self):
        """Initialize all components."""
        # Components
        self._ptt = PTTHandler()
        self._recorder = AudioRecorder(sample_rate=16000)
        self._player = AudioPlayer()
        self._screen = ScreenCapture(scale=0.5)
        self._stt = SpeechToText()
        self._tts = TextToSpeech()
        self._coach = Coach()

        # State
        self._screenshot: bytes | None = None
        self._running = False

        # Register callbacks
        self._ptt.on_press(self._on_ptt_press)
        self._ptt.on_release(self._on_ptt_release)
        self._ptt.on_quit(self._on_quit)

    def _on_ptt_press(self) -> None:
        """Called when PTT key is pressed."""
        print("\n[Recording...] ", end="", flush=True)
        # Capture screenshot for context (stored for later use)
        self._screenshot = self._screen.capture()
        # Start recording audio
        self._recorder.start()

    def _on_ptt_release(self) -> None:
        """Called when PTT key is released."""
        print("[Processing...]", flush=True)

        # Stop recording and get audio
        audio_bytes = self._recorder.stop()

        if not audio_bytes:
            print("No audio recorded.")
            return

        try:
            # Transcribe audio
            print("  Transcribing...", end=" ", flush=True)
            transcript = self._stt.transcribe(audio_bytes)
            print(f'"{transcript}"')

            if not transcript.strip():
                print("  No speech detected.")
                return

            # Get coach response (with screenshot for vision)
            print("  Thinking...", end=" ", flush=True)
            response = self._coach.get_response(
                user_message=transcript,
                screenshot=self._screenshot,
            )
            print(f'"{response}"')

            # Synthesize and play response
            print("  Speaking...", flush=True)
            audio_response = self._tts.synthesize(response)
            self._player.play(audio_response)

        except Exception as e:
            print(f"\n  Error: {e}")
            # Try to speak a fallback message
            try:
                fallback = self._tts.synthesize("Sorry, I couldn't process that.")
                self._player.play(fallback)
            except Exception:
                print("  (Failed to speak fallback message)")

        finally:
            self._screenshot = None

    def _on_quit(self) -> None:
        """Called when ESC is pressed."""
        print("\n\nShutting down...")
        self._running = False

    def run(self) -> None:
        """Main loop - start listening and wait for quit."""
        self._running = True

        print("=" * 50)
        print("  Gaming Coach Voice Agent")
        print("=" * 50)
        print()
        print("  Hold RIGHT OPTION to talk")
        print("  Release to get a response")
        print("  Press ESC to quit")
        print()
        print("=" * 50)
        print("\nReady! Waiting for input...")

        # Start the PTT listener
        self._ptt.start()

        # Wait for quit signal
        self._ptt.wait()

        print("Goodbye!")


def main():
    """Entry point."""
    # Load environment variables
    load_dotenv()

    # Check for required env vars
    missing = []
    if not os.environ.get("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if not os.environ.get("ELEVENLABS_API_KEY"):
        missing.append("ELEVENLABS_API_KEY")

    if missing:
        print("Error: Missing required environment variables:")
        for var in missing:
            print(f"  - {var}")
        print("\nCreate a .env file with these variables or export them.")
        sys.exit(1)

    # Create and run the coach
    try:
        coach = GamingCoach()
        coach.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\nFatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
