"""Pydantic models for game state and LLM structured outputs."""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class GameEvent(str, Enum):
    """Major game events detectable from screen."""
    NONE = "none"
    PLAYER_DIED = "player_died"           # "YOU DIED" screen
    BOSS_DEFEATED = "boss_defeated"       # "GREAT ENEMY FELLED" / "LEGEND FELLED"
    GRACE_DISCOVERED = "grace_discovered" # "Site of Grace Discovered"


class ScreenType(str, Enum):
    """Type of screen currently displayed."""
    GAMEPLAY = "gameplay"
    INVENTORY = "inventory"
    EQUIPMENT = "equipment"
    MAP = "map"
    STATUS = "status"
    GRACE_MENU = "grace_menu"
    LOADING = "loading"
    DEATH_SCREEN = "death_screen"
    CUTSCENE = "cutscene"


class TimeOfDay(str, Enum):
    """In-game time of day."""
    DAY = "day"
    NIGHT = "night"
    UNKNOWN = "unknown"


class UpdateType(str, Enum):
    """Type of game state update detected."""
    NOOP = "noop"
    LOCATION = "location"
    INVENTORY = "inventory"
    BOSS_ENCOUNTER = "boss_encounter"
    GAME_EVENT = "game_event"
    GRACE_REST = "grace_rest"
    STATS = "stats"
    MULTIPLE = "multiple"


class InventoryItem(BaseModel):
    """An item in the player's inventory."""
    name: str = Field(description="Name of the item")
    quantity: int = Field(default=1, description="Quantity of the item")


class BossState(BaseModel):
    """Boss encounter information."""
    name: str = Field(description="Boss name from health bar")
    hp_percentage: float = Field(description="Estimated HP remaining (0-100)")


class PlayerStats(BaseModel):
    """Player stats visible in HUD or menus."""
    hp_percentage: Optional[float] = Field(default=None, description="Estimated HP (0-100)")
    runes: Optional[int] = Field(default=None, description="Current rune count")
    rune_level: Optional[int] = Field(default=None, description="Character level if visible")


class StateUpdate(BaseModel):
    """Structured output from the LLM analyzing a screenshot.
    
    This model is used with OpenAI's structured outputs to ensure
    consistent parsing of the LLM's analysis.
    """
    update_type: UpdateType = Field(
        description="Type of update detected. Use 'noop' if no relevant game state information is visible."
    )
    screen_type: ScreenType = Field(
        default=ScreenType.GAMEPLAY,
        description="Type of screen currently displayed (gameplay, inventory, map, etc.)"
    )
    new_location: Optional[str] = Field(
        default=None,
        description="The player's current location/area name if visible (e.g., 'Limgrave', 'Stormveil Castle', 'Roundtable Hold'). Set when update_type is 'location' or 'multiple'."
    )
    inventory_items: Optional[list[InventoryItem]] = Field(
        default=None,
        description="List of items visible in the inventory screen. Only set if update_type is 'inventory' or 'multiple' and an inventory/equipment screen is clearly visible."
    )
    game_event: Optional[GameEvent] = Field(
        default=None,
        description="Major game event if visible (death screen, boss defeated, grace discovered). Set when update_type is 'game_event' or 'multiple'."
    )
    boss: Optional[BossState] = Field(
        default=None,
        description="Boss encounter info if a boss health bar is visible. Set when update_type is 'boss_encounter' or 'multiple'."
    )
    grace_name: Optional[str] = Field(
        default=None,
        description="Name of Site of Grace if player is resting at one. Set when update_type is 'grace_rest' or 'multiple'."
    )
    time_of_day: Optional[TimeOfDay] = Field(
        default=None,
        description="In-game time of day based on sky/lighting. Only set if clearly determinable from gameplay screenshot."
    )
    player_stats: Optional[PlayerStats] = Field(
        default=None,
        description="Player stats visible in HUD or menus (HP, runes, level)."
    )
    reasoning: str = Field(
        description="Brief explanation of what was detected in the screenshot and why this update type was chosen."
    )
    uncertainty_notes: Optional[str] = Field(
        default=None,
        description="Note any apparent discontinuities or uncertainty about state transitions (e.g., boss health bar disappeared unexpectedly)."
    )


class GameState(BaseModel):
    """Current state of the game being tracked."""
    player_location: str = Field(
        default="Unknown",
        description="Current area/region the player is in"
    )
    inventory: list[InventoryItem] = Field(
        default_factory=list,
        description="Items in the player's inventory"
    )
    current_boss: Optional[BossState] = Field(
        default=None,
        description="Current boss being fought, if any"
    )
    last_grace: Optional[str] = Field(
        default=None,
        description="Last Site of Grace the player rested at"
    )
    time_of_day: TimeOfDay = Field(
        default=TimeOfDay.UNKNOWN,
        description="Current in-game time of day"
    )
    deaths: int = Field(
        default=0,
        description="Total number of deaths tracked"
    )
    bosses_defeated: list[str] = Field(
        default_factory=list,
        description="List of boss names that have been defeated"
    )
    graces_discovered: list[str] = Field(
        default_factory=list,
        description="List of Sites of Grace discovered"
    )
    player_stats: PlayerStats = Field(
        default_factory=PlayerStats,
        description="Latest known player stats"
    )
    
    def apply_update(self, update: StateUpdate) -> bool:
        """Apply a state update and return True if state changed."""
        if update.update_type == UpdateType.NOOP:
            return False
        
        changed = False
        
        # Location updates
        if update.update_type in (UpdateType.LOCATION, UpdateType.MULTIPLE):
            if update.new_location and update.new_location != self.player_location:
                self.player_location = update.new_location
                changed = True
        
        # Inventory updates
        if update.update_type in (UpdateType.INVENTORY, UpdateType.MULTIPLE):
            if update.inventory_items is not None:
                self.inventory = update.inventory_items
                changed = True
        
        # Boss encounter updates
        if update.update_type in (UpdateType.BOSS_ENCOUNTER, UpdateType.MULTIPLE):
            if update.boss is not None:
                self.current_boss = update.boss
                changed = True
        
        # Game event updates
        if update.update_type in (UpdateType.GAME_EVENT, UpdateType.MULTIPLE):
            if update.game_event is not None and update.game_event != GameEvent.NONE:
                if update.game_event == GameEvent.PLAYER_DIED:
                    self.deaths += 1
                    self.current_boss = None  # Clear boss state on death
                    changed = True
                elif update.game_event == GameEvent.BOSS_DEFEATED:
                    if self.current_boss is not None:
                        if self.current_boss.name not in self.bosses_defeated:
                            self.bosses_defeated.append(self.current_boss.name)
                        self.current_boss = None
                    changed = True
                elif update.game_event == GameEvent.GRACE_DISCOVERED:
                    if update.grace_name and update.grace_name not in self.graces_discovered:
                        self.graces_discovered.append(update.grace_name)
                    changed = True
        
        # Grace rest updates
        if update.update_type in (UpdateType.GRACE_REST, UpdateType.MULTIPLE):
            if update.grace_name:
                self.last_grace = update.grace_name
                if update.grace_name not in self.graces_discovered:
                    self.graces_discovered.append(update.grace_name)
                self.current_boss = None  # Clear boss state when resting
                changed = True
        
        # Stats updates
        if update.update_type in (UpdateType.STATS, UpdateType.MULTIPLE):
            if update.player_stats is not None:
                # Update only non-None fields
                if update.player_stats.hp_percentage is not None:
                    self.player_stats.hp_percentage = update.player_stats.hp_percentage
                if update.player_stats.runes is not None:
                    self.player_stats.runes = update.player_stats.runes
                if update.player_stats.rune_level is not None:
                    self.player_stats.rune_level = update.player_stats.rune_level
                changed = True
        
        # Time of day (always update if provided, regardless of update_type)
        if update.time_of_day is not None and update.time_of_day != TimeOfDay.UNKNOWN:
            if update.time_of_day != self.time_of_day:
                self.time_of_day = update.time_of_day
                changed = True
        
        return changed

