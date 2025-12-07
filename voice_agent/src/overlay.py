
import threading
from typing import Callable

from AppKit import (
    NSApplication,
    NSBackgroundStyleDark,
    NSColor,
    NSFloatingWindowLevel,
    NSFont,
    NSMakeRect,
    NSScreen,
    NSTextField,
    NSView,
    NSWindow,
    NSWindowStyleMaskBorderless,
)
from PyObjCTools import AppHelper


class TextOverlay:
    """Displays text in a subtle overlay in the top-right corner of the screen."""

    def __init__(
        self,
        max_width: int = 400,
        padding: int = 16,
        font_size: float = 14.0,
        background_alpha: float = 0.85,
        corner_radius: float = 10.0,
        margin: int = 20,
    ):
    """Text overlay for displaying spoken responses on screen."""
        """
        Initialize the text overlay.

        Args:
            max_width: Maximum width of the overlay in pixels.
            padding: Padding around the text in pixels.
            font_size: Font size for the text.
            background_alpha: Background transparency (0.0 to 1.0).
            corner_radius: Corner radius for rounded corners.
            margin: Margin from screen edges in pixels.
        """
        self._max_width = max_width
        self._padding = padding
        self._font_size = font_size
        self._background_alpha = background_alpha
        self._corner_radius = corner_radius
        self._margin = margin

        self._window: NSWindow | None = None
        self._text_field: NSTextField | None = None
        self._app_thread: threading.Thread | None = None
        self._app_started = threading.Event()

        # Start the AppKit run loop in a background thread
        self._start_app_thread()

    def _start_app_thread(self) -> None:
        """Start the AppKit application in a background thread."""
        def run_app():
            # Get or create the shared application
            app = NSApplication.sharedApplication()
            # Signal that the app is ready
            self._app_started.set()
            # Run the event loop (this blocks)
            AppHelper.runEventLoop()

        self._app_thread = threading.Thread(target=run_app, daemon=True)
        self._app_thread.start()
        # Wait for the app to be ready
        self._app_started.wait(timeout=5.0)

    def _run_on_main_thread(self, func: Callable[[], None]) -> None:
        """Run a function on the main AppKit thread."""
        AppHelper.callAfter(func)

    def _create_window(self, text: str) -> None:
        """Create and configure the overlay window."""
        # Get the main screen
        screen = NSScreen.mainScreen()
        if not screen:
            return

        screen_frame = screen.visibleFrame()

        # Create the text field to measure text size
        self._text_field = NSTextField.alloc().initWithFrame_(NSMakeRect(0, 0, 0, 0))
        self._text_field.setStringValue_(text)
        self._text_field.setFont_(NSFont.systemFontOfSize_(self._font_size))
        self._text_field.setTextColor_(NSColor.whiteColor())
        self._text_field.setBackgroundColor_(NSColor.clearColor())
        self._text_field.setBezeled_(False)
        self._text_field.setEditable_(False)
        self._text_field.setSelectable_(False)
        self._text_field.setDrawsBackground_(False)

        # Enable word wrapping
        self._text_field.cell().setWraps_(True)
        self._text_field.setPreferredMaxLayoutWidth_(self._max_width - 2 * self._padding)

        # Calculate the size needed for the text
        self._text_field.sizeToFit()
        text_size = self._text_field.frame().size

        # Constrain width and recalculate if needed
        if text_size.width > self._max_width - 2 * self._padding:
            self._text_field.setFrameSize_(
                (self._max_width - 2 * self._padding, text_size.height * 2)
            )
            self._text_field.sizeToFit()
            text_size = self._text_field.frame().size

        # Calculate window size
        window_width = min(text_size.width + 2 * self._padding, self._max_width)
        window_height = text_size.height + 2 * self._padding

        # Position in top-right corner
        window_x = screen_frame.origin.x + screen_frame.size.width - window_width - self._margin
        window_y = screen_frame.origin.y + screen_frame.size.height - window_height - self._margin

        window_rect = NSMakeRect(window_x, window_y, window_width, window_height)

        # Create borderless, transparent window
        self._window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            window_rect,
            NSWindowStyleMaskBorderless,
            2,  # NSBackingStoreBuffered
            False,
        )

        # Configure window properties
        self._window.setLevel_(NSFloatingWindowLevel)
        self._window.setOpaque_(False)
        self._window.setBackgroundColor_(
            NSColor.colorWithCalibratedRed_green_blue_alpha_(0.1, 0.1, 0.1, self._background_alpha)
        )
        self._window.setIgnoresMouseEvents_(True)
        self._window.setHasShadow_(True)

        # Create a content view with rounded corners
        content_view = self._window.contentView()
        content_view.setWantsLayer_(True)
        content_view.layer().setCornerRadius_(self._corner_radius)
        content_view.layer().setMasksToBounds_(True)

        # Position text field within the window
        self._text_field.setFrameOrigin_((self._padding, self._padding))
        content_view.addSubview_(self._text_field)

        # Show the window
        self._window.orderFrontRegardless()

    def _destroy_window(self) -> None:
        """Destroy the overlay window."""
        if self._window:
            self._window.orderOut_(None)
            self._window = None
            self._text_field = None

    def show(self, text: str) -> None:
        """
        Display text in the overlay.

        Args:
            text: The text to display.
        """
        if not text.strip():
            return

        def _show():
            # Destroy existing window first
            self._destroy_window()
            # Create new window with updated text
            self._create_window(text)

        self._run_on_main_thread(_show)

    def hide(self) -> None:
        """Hide the overlay."""
        self._run_on_main_thread(self._destroy_window)
