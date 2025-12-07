"""Game state management."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Callable

from .logging_config import log_game_state
from .models import GameState, StateUpdate, UpdateType

if TYPE_CHECKING:
    from .redis_store import GameStateStore

logger = logging.getLogger(__name__)


class StateManager:
    """Manages game state and coordinates updates."""

    def __init__(
        self,
        initial_state: GameState | None = None,
        redis_store: GameStateStore | None = None,
    ):
        """Initialize state manager.

        Args:
            initial_state: Starting game state. If None, creates default state.
            redis_store: Optional Redis store for persistence. If provided,
                state changes are automatically saved to Redis.
        """
        self.state = initial_state or GameState()
        self._listeners: list[Callable[[GameState, StateUpdate], None]] = []
        self._last_update: datetime | None = None
        self._redis_store = redis_store

        # Register Redis persistence listener if store provided
        if redis_store:
            self.add_listener(lambda state, _: redis_store.save(state))
            logger.info("Redis persistence enabled")

        # Log initial state
        logger.info("StateManager initialized")
        log_game_state(logger, self.state)
    
    @property
    def current_state(self) -> GameState:
        """Get current game state."""
        return self.state
    
    @property
    def last_update_time(self) -> datetime | None:
        """Get timestamp of last state change."""
        return self._last_update
    
    def add_listener(self, callback: Callable[[GameState, StateUpdate], None]) -> None:
        """Add a listener to be notified on state changes.
        
        Args:
            callback: Function called with (new_state, update) when state changes
        """
        self._listeners.append(callback)
    
    def remove_listener(self, callback: Callable[[GameState, StateUpdate], None]) -> None:
        """Remove a state change listener."""
        self._listeners.remove(callback)
    
    def process_update(self, update: StateUpdate) -> bool:
        """Process a state update from the analyzer.
        
        Args:
            update: The state update to apply
            
        Returns:
            True if state changed, False otherwise
        """
        if update.update_type == UpdateType.NOOP:
            logger.debug(f"Noop update: {update.reasoning}")
            # Still log the state even for noop (every update cycle)
            log_game_state(logger, self.state, update)
            return False
        
        changed = self.state.apply_update(update)
        
        if changed:
            self._last_update = datetime.now()
            logger.info(f"State updated: {update.update_type.value}")
            logger.info(f"  Reason: {update.reasoning}")
            
            if update.new_location:
                logger.info(f"  Location: {update.new_location}")
            if update.inventory_items:
                logger.info(f"  Inventory items: {len(update.inventory_items)}")
            
            # Notify listeners
            for listener in self._listeners:
                try:
                    listener(self.state, update)
                except Exception as e:
                    logger.error(f"Listener error: {e}")
        
        # Log game state after every update attempt
        log_game_state(logger, self.state, update)
        
        return changed
    
    def get_state_summary(self) -> dict:
        """Get a summary of current state for display/logging."""
        return {
            "location": self.state.player_location,
            "inventory_count": len(self.state.inventory),
            "inventory_items": [
                f"{item.name} x{item.quantity}" 
                for item in self.state.inventory
            ],
            "last_update": self._last_update.isoformat() if self._last_update else None
        }
