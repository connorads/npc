"""Text-to-Speech using ElevenLabs."""

import os

from elevenlabs import ElevenLabs


class TextToSpeech:
    """Generates speech using ElevenLabs."""

    def __init__(
        self,
        api_key: str | None = None,
        voice_id: str = "JBFqnCBsd6RMkjVDRZzb",
        model_id: str = "eleven_turbo_v2_5",
    ):
        """
        Initialize the TTS client.

        Args:
            api_key: ElevenLabs API key. Falls back to ELEVENLABS_API_KEY env var.
            voice_id: Voice ID to use. Default is "George".
            model_id: Model to use. Default is turbo v2.5 for low latency.
        """
        self._api_key = api_key or os.environ.get("ELEVENLABS_API_KEY")
        if not self._api_key:
            raise ValueError(
                "ElevenLabs API key required. Set ELEVENLABS_API_KEY env var or pass api_key."
            )
        self._client = ElevenLabs(api_key=self._api_key)
        self._voice_id = voice_id
        self._model_id = model_id

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

        # Generate audio - returns a generator
        audio_generator = self._client.text_to_speech.convert(
            voice_id=self._voice_id,
            text=text,
            model_id=self._model_id,
            output_format="pcm_24000",  # 24kHz 16-bit mono PCM
        )

        # Collect all chunks from the generator
        audio_chunks = []
        for chunk in audio_generator:
            if isinstance(chunk, bytes):
                audio_chunks.append(chunk)

        return b"".join(audio_chunks)
