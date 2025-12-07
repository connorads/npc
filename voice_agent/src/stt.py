"""Speech-to-Text using ElevenLabs Scribe."""

import os
import io

import logfire
from elevenlabs import ElevenLabs


class SpeechToText:
    """Transcribes audio using ElevenLabs Scribe."""

    def __init__(self, api_key: str | None = None, model_id: str = "scribe_v1"):
        """
        Initialize the STT client.

        Args:
            api_key: ElevenLabs API key. Falls back to ELEVENLABS_API_KEY env var.
            model_id: The model to use for transcription.
        """
        self._api_key = api_key or os.environ.get("ELEVENLABS_API_KEY")
        if not self._api_key:
            raise ValueError(
                "ElevenLabs API key required. Set ELEVENLABS_API_KEY env var or pass api_key."
            )
        self._client = ElevenLabs(api_key=self._api_key)
        self._model_id = model_id

    @logfire.instrument("elevenlabs.stt")
    def transcribe(self, audio_bytes: bytes) -> str:
        """
        Transcribe audio to text.

        Args:
            audio_bytes: WAV format audio bytes.

        Returns:
            Transcribed text.
        """
        if not audio_bytes:
            return ""

        logfire.info(
            "Starting transcription",
            audio_size_bytes=len(audio_bytes),
            model_id=self._model_id,
        )

        # Create a file-like object from bytes
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "audio.wav"  # ElevenLabs needs a filename

        result = self._client.speech_to_text.convert(
            file=audio_file,
            model_id=self._model_id,
        )

        logfire.info(
            "Transcription completed",
            transcript_length=len(result.text),
        )

        return result.text
