"""Audio recording and playback using sounddevice."""

import io
import wave
import threading
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf


class AudioRecorder:
    """Records audio from the default microphone."""

    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        """
        Initialize the audio recorder.

        Args:
            sample_rate: Sample rate in Hz. Default 16000 for speech.
            channels: Number of audio channels. Default 1 (mono).
        """
        self._sample_rate = sample_rate
        self._channels = channels
        self._recording = False
        self._audio_chunks: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None
        self._lock = threading.Lock()

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: dict,
        status: sd.CallbackFlags,
    ) -> None:
        """Callback function for audio stream."""
        if status:
            print(f"Audio status: {status}")
        if self._recording:
            with self._lock:
                self._audio_chunks.append(indata.copy())

    def start(self) -> None:
        """Start recording (non-blocking, buffers in background)."""
        with self._lock:
            self._audio_chunks = []
        self._recording = True
        stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=self._channels,
            dtype=np.int16,
            callback=self._audio_callback,
        )
        stream.start()
        self._stream = stream

    def stop(self) -> bytes:
        """
        Stop recording and return WAV bytes.

        Returns:
            WAV format audio bytes.
        """
        self._recording = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        with self._lock:
            if not self._audio_chunks:
                return b""
            audio_data = np.concatenate(self._audio_chunks, axis=0)

        # Convert to WAV bytes
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(self._channels)
            wav_file.setsampwidth(2)  # 16-bit = 2 bytes
            wav_file.setframerate(self._sample_rate)
            wav_file.writeframes(audio_data.tobytes())

        buffer.seek(0)
        return buffer.read()

    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._recording


class AudioPlayer:
    """Plays audio through the default output device."""

    def play(self, audio_data: bytes, sample_rate: int = 24000) -> None:
        """
        Play raw PCM audio bytes. Blocks until complete.

        Args:
            audio_data: Raw PCM audio bytes (16-bit mono).
            sample_rate: Sample rate of the audio. Default 24000 for ElevenLabs.
        """
        if not audio_data:
            return

        # Convert bytes to numpy array (16-bit signed integers)
        audio_array = np.frombuffer(audio_data, dtype=np.int16)

        # Play and wait for completion
        sd.play(audio_array, sample_rate)
        sd.wait()

    def play_file(self, path: Path) -> None:
        """
        Play audio from a file.

        Args:
            path: Path to the audio file.
        """
        data, sample_rate = sf.read(path)
        sd.play(data, sample_rate)
        sd.wait()
