"""Push-to-Talk Handler using pynput."""

from collections.abc import Callable
from pynput import keyboard
from pynput.keyboard import Key


class PTTHandler:
    """Handles push-to-talk keyboard events using pynput."""

    def __init__(self, key: Key = Key.alt_r):
        """
        Initialize the PTT handler.

        Args:
            key: The key to use for PTT. Default is Key.alt_r (right option on macOS).
        """
        self._ptt_key = key
        self._on_press_callback: Callable[[], None] | None = None
        self._on_release_callback: Callable[[], None] | None = None
        self._on_quit_callback: Callable[[], None] | None = None
        self._listener: keyboard.Listener | None = None
        self._is_pressed = False

    def on_press(self, callback: Callable[[], None]) -> None:
        """Register callback for PTT key press."""
        self._on_press_callback = callback

    def on_release(self, callback: Callable[[], None]) -> None:
        """Register callback for PTT key release."""
        self._on_release_callback = callback

    def on_quit(self, callback: Callable[[], None]) -> None:
        """Register callback for quit (ESC) key."""
        self._on_quit_callback = callback

    def _handle_press(self, key: Key | keyboard.KeyCode | None) -> None:
        """Internal handler for key press events."""
        if key == self._ptt_key and not self._is_pressed:
            self._is_pressed = True
            if self._on_press_callback:
                self._on_press_callback()

    def _handle_release(self, key: Key | keyboard.KeyCode | None) -> None:
        """Internal handler for key release events."""
        if key == self._ptt_key:
            self._is_pressed = False
            if self._on_release_callback:
                self._on_release_callback()
        elif key == Key.esc:
            if self._on_quit_callback:
                self._on_quit_callback()
            # Stop the listener by calling stop() instead of returning False
            self.stop()

    def start(self) -> None:
        """Start listening for key events (non-blocking)."""
        self._listener = keyboard.Listener(
            on_press=self._handle_press,
            on_release=self._handle_release,
        )
        self._listener.start()

    def stop(self) -> None:
        """Stop listening for key events."""
        if self._listener:
            self._listener.stop()
            self._listener = None

    def wait(self) -> None:
        """Wait for the listener to stop (blocks until ESC is pressed)."""
        if self._listener:
            self._listener.join()
