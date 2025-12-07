"""Main entry point for the game state agent."""

import logging
import signal
import time

from .analyzer import ScreenshotAnalyzer
from .capture import ScreenCapture
from .config import CAPTURE_INTERVAL
from .logging_config import setup_logging
from .state_manager import StateManager

logger = logging.getLogger(__name__)


class GameStateAgent:
    """Main agent that coordinates screenshot capture and analysis."""
    
    def __init__(self, capture_interval: float = CAPTURE_INTERVAL):
        """Initialize the game state agent.
        
        Args:
            capture_interval: Seconds between screenshot captures
        """
        self.capture_interval = capture_interval
        self.state_manager = StateManager()
        self.analyzer = ScreenshotAnalyzer()
        self._running = False
    
    def start(self) -> None:
        """Start the game state tracking loop."""
        self._running = True
        logger.info("=" * 50)
        logger.info("Game State Agent Started")
        logger.info(f"Capture interval: {self.capture_interval}s")
        logger.info("=" * 50)
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        with ScreenCapture() as capture:
            while self._running:
                try:
                    self._process_frame(capture)
                except Exception as e:
                    logger.error(f"Error processing frame: {e}", exc_info=True)
                
                # Wait for next capture interval
                if self._running:
                    time.sleep(self.capture_interval)
        
        logger.info("Game State Agent stopped")
    
    def stop(self) -> None:
        """Stop the game state tracking loop."""
        self._running = False
    
    def _signal_handler(self, signum, frame) -> None:
        """Handle shutdown signals gracefully."""
        logger.info("\nShutdown signal received...")
        self.stop()
    
    def _process_frame(self, capture: ScreenCapture) -> None:
        """Capture and analyze a single frame."""
        start_time = time.time()
        
        # Capture screenshot
        logger.debug("Capturing screenshot...")
        image_base64 = capture.capture_base64()
        capture_time = time.time() - start_time
        logger.debug(f"Screenshot captured in {capture_time:.3f}s")
        
        # Analyze with LLM
        logger.debug("Analyzing screenshot with LLM...")
        analysis_start = time.time()
        update = self.analyzer.analyze(image_base64)
        analysis_time = time.time() - analysis_start
        logger.debug(f"Analysis completed in {analysis_time:.3f}s")
        
        # Process update
        changed = self.state_manager.process_update(update)
        
        # Log timing and state
        total_time = time.time() - start_time
        logger.info(
            f"Frame processed in {total_time:.2f}s "
            f"(capture: {capture_time:.2f}s, analysis: {analysis_time:.2f}s) "
            f"- {'STATE CHANGED' if changed else 'no change'}"
        )


def main():
    """Entry point for the game state agent."""
    # Initialize logging first
    log_file = setup_logging()
    
    print(f"""
    ╔═══════════════════════════════════════════════════════╗
    ║         ELDEN RING GAME STATE AGENT                   ║
    ║                                                       ║
    ║  Tracking: Player Location, Inventory                 ║
    ║  Press Ctrl+C to stop                                 ║
    ║                                                       ║
    ║  Log file: {str(log_file):<43} ║
    ╚═══════════════════════════════════════════════════════╝
    """)
    
    logger.info("Starting Elden Ring Game State Agent")
    
    agent = GameStateAgent()
    agent.start()


if __name__ == "__main__":
    main()
