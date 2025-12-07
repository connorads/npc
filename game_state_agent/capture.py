"""Screenshot capture functionality using mss."""

import base64
import io
from mss import mss
from mss.base import MSSBase


class ScreenCapture:
    """Handles screen capture and encoding for LLM input."""
    
    def __init__(self, monitor: int = 1):
        """Initialize screen capture.
        
        Args:
            monitor: Monitor number to capture (1 = primary monitor)
        """
        self.monitor = monitor
        self._sct: MSSBase | None = None
    
    def __enter__(self) -> "ScreenCapture":
        self._sct = mss()
        return self
    
    def __exit__(self, *args) -> None:
        if self._sct:
            self._sct.close()
            self._sct = None
    
    def capture_base64(self) -> str:
        """Capture screenshot and return as base64-encoded PNG.
        
        Returns:
            Base64-encoded PNG image string suitable for OpenAI API
        """
        if self._sct is None:
            raise RuntimeError("ScreenCapture must be used as context manager")
        
        # Capture the specified monitor
        screenshot = self._sct.grab(self._sct.monitors[self.monitor])
        
        # Convert to PNG bytes
        png_bytes = self._to_png(screenshot)
        
        # Encode to base64
        return base64.b64encode(png_bytes).decode("utf-8")
    
    def _to_png(self, screenshot) -> bytes:
        """Convert mss screenshot to PNG bytes."""
        from mss.tools import to_png
        return to_png(screenshot.rgb, screenshot.size)


def capture_screen_base64(monitor: int = 1) -> str:
    """Convenience function to capture screen as base64.
    
    Args:
        monitor: Monitor number to capture (1 = primary)
        
    Returns:
        Base64-encoded PNG image string
    """
    with ScreenCapture(monitor=monitor) as capture:
        return capture.capture_base64()

