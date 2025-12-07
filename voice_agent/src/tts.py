"""Text-to-Speech using ElevenLabs."""

import os

import logfire
from elevenlabs import ElevenLabs
from elevenlabs.types import VoiceSettings


class TextToSpeech:
    """Generates speech using ElevenLabs."""

    def __init__(
        self,
        api_key: str | None = None,
        voice_id: str = "JBFqnCBsd6RMkjVDRZzb",
        model_id: str = "eleven_turbo_v2_5",
        speed: float = 1.2,
    ):
        """
        Initialize the TTS client.

        Args:
            api_key: ElevenLabs API key. Falls back to ELEVENLABS_API_KEY env var.
            voice_id: Voice ID to use. Default is "George".
            model_id: Model to use. Default is turbo v2.5 for low latency.
            speed: Speech speed multiplier (0.25 to 4.0). Default is 1.2 for slightly faster speech.
        """
        self._api_key = api_key or os.environ.get("ELEVENLABS_API_KEY")
        if not self._api_key:
            raise ValueError(
                "ElevenLabs API key required. Set ELEVENLABS_API_KEY env var or pass api_key."
            )
        self._client = ElevenLabs(api_key=self._api_key)
        self._voice_id = voice_id
        self._model_id = model_id
        self._speed = speed

    @logfire.instrument("elevenlabs.tts")
    def synthesize(self, text: str) -> bytes:
        """
        Convert text to speech.

        Args:
            text: Text to speak.

        Returns:
            Raw PCM audio bytes (24kHz, 16-bit mono).
        """
        if not text:
            return b""

        logfire.info(
            "Starting speech synthesis",
            text_length=len(text),
            voice_id=self._voice_id,
            model_id=self._model_id,
        )

        # Generate audio - returns a generator
        audio_generator = self._client.text_to_speech.convert(
            voice_id=self._voice_id,
            text=text,
            model_id=self._model_id,
            output_format="pcm_24000",  # 24kHz 16-bit mono PCM
            voice_settings=VoiceSettings(speed=self._speed),
        )

        # Collect all chunks from the generator
        audio_chunks = []
        for chunk in audio_generator:
            if isinstance(chunk, bytes):
                audio_chunks.append(chunk)

        audio_bytes = b"".join(audio_chunks)

        logfire.info(
            "Speech synthesis completed",
            audio_size_bytes=len(audio_bytes),
        )

        return audio_bytes
