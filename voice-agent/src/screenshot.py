"""Screen capture using mss and PIL."""

import base64
import io

import mss
from PIL import Image


class ScreenCapture:
    """Captures screenshots of the primary display."""

    def __init__(self, scale: float = 0.5, quality: int = 85):
        """
        Initialize the screen capture.

        Args:
            scale: Scale factor to reduce image size (0.5 = half size).
            quality: JPEG quality (1-100). Lower = smaller file, lower quality.
        """
        self._scale = scale
        self._quality = quality

    def capture(self) -> bytes:
        """
        Capture screen and return as JPEG bytes.

        Returns:
            JPEG format image bytes.
        """
        with mss.mss() as sct:
            # Capture primary monitor (index 1, as 0 is "all monitors")
            monitor = sct.monitors[1]
            screenshot = sct.grab(monitor)

            # Convert to PIL Image
            img = Image.frombytes(
                "RGB", screenshot.size, screenshot.bgra, "raw", "BGRX"
            )

            # Scale down if needed
            if self._scale != 1.0:
                new_size = (
                    int(img.width * self._scale),
                    int(img.height * self._scale),
                )
                img = img.resize(new_size, Image.Resampling.LANCZOS)

            # Convert to JPEG bytes
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=self._quality)
            buffer.seek(0)
            return buffer.read()

    def capture_base64(self) -> str:
        """
        Capture screen and return as base64-encoded JPEG.

        Returns:
            Base64-encoded JPEG string.
        """
        jpeg_bytes = self.capture()
        return base64.b64encode(jpeg_bytes).decode("utf-8")
