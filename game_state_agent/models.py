"""Pydantic models for game state and LLM structured outputs."""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class GameEvent(str, Enum):
    """Major game events detectable from screen."""
    NONE = "none"
    PARTY_DEFEATED = "party_defeated"       # Party defeated/death screen
    BOSS_DEFEATED = "boss_defeated"         # Boss/enemy defeated notification
    FLAG_DISCOVERED = "flag_discovered"     # "Expedition Flag Discovered"


class ScreenType(str, Enum):
    """Type of screen currently displayed."""
    GAMEPLAY = "gameplay"
    COMBAT = "combat"
    INVENTORY = "inventory"
    EQUIPMENT = "equipment"
    MAP = "map"
    STATUS = "status"
    CAMP_MENU = "camp_menu"
    FLAG_MENU = "flag_menu"
    LOADING = "loading"
    DEATH_SCREEN = "death_screen"
    CUTSCENE = "cutscene"
    DIALOGUE = "dialogue"


class UpdateType(str, Enum):
    """Type of game state update detected."""
    NOOP = "noop"
    LOCATION = "location"
    INVENTORY = "inventory"
    BOSS_ENCOUNTER = "boss_encounter"
    GAME_EVENT = "game_event"
    FLAG_REST = "flag_rest"
    CAMP_REST = "camp_rest"
    STATS = "stats"
    MULTIPLE = "multiple"


class InventoryItem(BaseModel):
    """An item in the player's inventory."""
    name: str = Field(description="Name of the item")
    quantity: int = Field(default=1, description="Quantity of the item")


class Picto(BaseModel):
    """A Picto (equipable perk) in the game."""
    name: str = Field(description="Name of the Picto")
    character: Optional[str] = Field(default=None, description="Character the Picto is equipped to")
    mastered: bool = Field(default=False, description="Whether the Picto has been mastered")


class BossState(BaseModel):
    """Boss/enemy encounter information."""
    name: str = Field(description="Boss/enemy name from health bar")
    hp_percentage: float = Field(description="Estimated HP remaining (0-100)")
    is_axon: bool = Field(default=False, description="Whether this is an Axon (ancient powerful being)")


class CharacterStats(BaseModel):
    """Stats for a party member."""
    name: str = Field(description="Character name (Gustave, Maelle, Lune, Sciel, Verso, Monoco)")
    hp_percentage: Optional[float] = Field(default=None, description="Estimated HP (0-100)")
    level: Optional[int] = Field(default=None, description="Character level if visible")
    vitality: Optional[int] = Field(default=None, description="Vitality stat (max health)")
    might: Optional[int] = Field(default=None, description="Might stat (attack power)")
    agility: Optional[int] = Field(default=None, description="Agility stat (attack frequency)")
    defense: Optional[int] = Field(default=None, description="Defense stat (damage reduction)")
    luck: Optional[int] = Field(default=None, description="Luck stat (critical rate)")


class PartyStats(BaseModel):
    """Party-wide stats visible in HUD or menus."""
    active_characters: Optional[list[str]] = Field(default=None, description="Characters currently in the active party")
    gradient_gauge: Optional[float] = Field(default=None, description="Gradient gauge percentage (0-100) for powerful attacks")


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
        description="Type of screen currently displayed (gameplay, combat, inventory, map, etc.)"
    )
    new_location: Optional[str] = Field(
        default=None,
        description="The player's current location/area name if visible (e.g., 'Lumière', 'The Continent', 'Old Lumière', 'Renoir's Mansion'). Set when update_type is 'location' or 'multiple'."
    )
    inventory_items: Optional[list[InventoryItem]] = Field(
        default=None,
        description="List of items visible in the inventory screen. Only set if update_type is 'inventory' or 'multiple' and an inventory/equipment screen is clearly visible."
    )
    pictos: Optional[list[Picto]] = Field(
        default=None,
        description="List of Pictos (equipable perks) visible. Only set if viewing Picto menu."
    )
    game_event: Optional[GameEvent] = Field(
        default=None,
        description="Major game event if visible (death screen, boss defeated, flag discovered). Set when update_type is 'game_event' or 'multiple'."
    )
    boss: Optional[BossState] = Field(
        default=None,
        description="Boss/enemy encounter info if a boss health bar is visible in combat. Set when update_type is 'boss_encounter' or 'multiple'."
    )
    flag_name: Optional[str] = Field(
        default=None,
        description="Name of Expedition Flag if player is resting at one. Set when update_type is 'flag_rest' or 'multiple'."
    )
    at_camp: Optional[bool] = Field(
        default=None,
        description="Whether the party is resting at camp. Set when update_type is 'camp_rest' or 'multiple'."
    )
    character_stats: Optional[list[CharacterStats]] = Field(
        default=None,
        description="Stats for party members visible in HUD or menus."
    )
    party_stats: Optional[PartyStats] = Field(
        default=None,
        description="Party-wide stats like gradient gauge."
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
        description="Current area/region the party is in"
    )
    inventory: list[InventoryItem] = Field(
        default_factory=list,
        description="Items in the party's inventory"
    )
    pictos: list[Picto] = Field(
        default_factory=list,
        description="Pictos collected by the party"
    )
    current_boss: Optional[BossState] = Field(
        default=None,
        description="Current boss being fought, if any"
    )
    last_flag: Optional[str] = Field(
        default=None,
        description="Last Expedition Flag the party rested at"
    )
    at_camp: bool = Field(
        default=False,
        description="Whether the party is currently at camp"
    )
    party_defeats: int = Field(
        default=0,
        description="Total number of party defeats tracked"
    )
    bosses_defeated: list[str] = Field(
        default_factory=list,
        description="List of boss names that have been defeated"
    )
    axons_defeated: list[str] = Field(
        default_factory=list,
        description="List of Axon names that have been defeated"
    )
    flags_discovered: list[str] = Field(
        default_factory=list,
        description="List of Expedition Flags discovered"
    )
    active_party: list[str] = Field(
        default_factory=lambda: ["Gustave", "Maelle", "Lune"],
        description="Current active party members"
    )
    character_stats: dict[str, CharacterStats] = Field(
        default_factory=dict,
        description="Stats for each party member"
    )
    gradient_gauge: float = Field(
        default=0.0,
        description="Current gradient gauge percentage"
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
            if update.pictos is not None:
                self.pictos = update.pictos
                changed = True
        
        # Boss encounter updates
        if update.update_type in (UpdateType.BOSS_ENCOUNTER, UpdateType.MULTIPLE):
            if update.boss is not None:
                self.current_boss = update.boss
                changed = True
        
        # Game event updates
        if update.update_type in (UpdateType.GAME_EVENT, UpdateType.MULTIPLE):
            if update.game_event is not None and update.game_event != GameEvent.NONE:
                if update.game_event == GameEvent.PARTY_DEFEATED:
                    self.party_defeats += 1
                    self.current_boss = None  # Clear boss state on defeat
                    changed = True
                elif update.game_event == GameEvent.BOSS_DEFEATED:
                    if self.current_boss is not None:
                        if self.current_boss.name not in self.bosses_defeated:
                            self.bosses_defeated.append(self.current_boss.name)
                        if self.current_boss.is_axon and self.current_boss.name not in self.axons_defeated:
                            self.axons_defeated.append(self.current_boss.name)
                        self.current_boss = None
                    changed = True
                elif update.game_event == GameEvent.FLAG_DISCOVERED:
                    if update.flag_name and update.flag_name not in self.flags_discovered:
                        self.flags_discovered.append(update.flag_name)
                    changed = True
        
        # Flag rest updates
        if update.update_type in (UpdateType.FLAG_REST, UpdateType.MULTIPLE):
            if update.flag_name:
                self.last_flag = update.flag_name
                if update.flag_name not in self.flags_discovered:
                    self.flags_discovered.append(update.flag_name)
                self.current_boss = None  # Clear boss state when resting
                self.at_camp = False
                changed = True
        
        # Camp rest updates
        if update.update_type in (UpdateType.CAMP_REST, UpdateType.MULTIPLE):
            if update.at_camp is not None:
                self.at_camp = update.at_camp
                self.current_boss = None  # Clear boss state when at camp
                changed = True
        
        # Stats updates
        if update.update_type in (UpdateType.STATS, UpdateType.MULTIPLE):
            if update.character_stats is not None:
                for char_stat in update.character_stats:
                    self.character_stats[char_stat.name] = char_stat
                changed = True
            if update.party_stats is not None:
                if update.party_stats.active_characters:
                    self.active_party = update.party_stats.active_characters
                if update.party_stats.gradient_gauge is not None:
                    self.gradient_gauge = update.party_stats.gradient_gauge
                changed = True
        
        return changed
