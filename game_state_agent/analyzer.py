"""Screenshot analysis using OpenAI's vision model with structured outputs."""

import logging
from openai import OpenAI

from .config import MODEL_NAME, get_openai_client
from .logging_config import log_openai_request, log_openai_response
from .models import StateUpdate

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are an expert Clair Obscur: Expedition 33 player analyzing screenshots from the video game to track game state.

Your task is to examine each screenshot and identify relevant game state information.

## What to Detect

1. **Player Location**: Area names displayed when entering new areas, Expedition Flag names, or recognizable landmarks. The game is set in a dark fantasy Belle Époque world. Regions include "Lumière", "The Continent", "Old Lumière", "Renoir's Mansion", "The Monolith", and various areas accessible via the mythical creature Esquie.

2. **Inventory/Equipment**: Inventory screens showing items, Pictos (equipable perks), weapons, and character equipment. Look for Chroma Catalysts (weapon upgrade materials).

3. **Boss/Enemy Encounters**: Large health bar at the BOTTOM of the screen with enemy name above it. Boss names are displayed prominently. Report the boss name and estimate HP percentage from the bar fill. Note if it's an Axon (ancient powerful being).

4. **Game Events**:
   - Death screen - Player party defeated
   - Enemy/Boss defeated - Victory notification
   - "Expedition Flag Discovered" - Notification when finding a new flag

5. **Expedition Flags**: When resting at an Expedition Flag, a menu appears with options to heal the party, fast travel, restock items, and allocate attribute/skill points.

6. **Camp**: The party can rest at camp where Verso can converse with other Expedition members. Look for relationship/conversation options.

7. **Combat UI**:
   - Turn-based combat with real-time elements (dodge, parry, jump)
   - Action Points (AP) for skills and ranged attacks
   - Gradient Gauge for powerful Gradient Attacks/Skills
   - Break/Stamina system for stunning enemies
   - Character health bars for party members (Gustave, Maelle, Lune, Sciel, Verso, Monoco)

8. **Player Stats (HUD/Menus)**:
   - HP bars for party members
   - Ability Points (AP)
   - Character level and attributes: Vitality, Might, Agility, Defense, Luck
   - Lumina Points for passive bonuses

## Screen Types

Identify the current screen type:
- "gameplay" - Active gameplay, world visible, exploration
- "combat" - Turn-based battle screen
- "inventory" - Inventory/Pictos menu open
- "equipment" - Equipment/weapons menu
- "map" - Continent map open
- "status" - Character status/stats/skill tree screen
- "camp_menu" - Resting at camp
- "flag_menu" - At an Expedition Flag
- "loading" - Loading screen
- "death_screen" - Party defeated screen
- "cutscene" - Cinematic/cutscene playing
- "dialogue" - Character conversation/relationship scene

## Update Types

Use the appropriate update_type:
- "noop" - No relevant game state visible
- "location" - New location/area detected
- "inventory" - Inventory screen with items
- "boss_encounter" - Boss/enemy health bar visible in combat
- "game_event" - Death, boss defeat, or flag discovery
- "flag_rest" - Resting at Expedition Flag
- "camp_rest" - Resting at camp
- "stats" - Player stats visible in HUD/menu
- "multiple" - Multiple types of information detected

## IMPORTANT - Temporal Limitations

Screenshots are captured every ~5 seconds. You may miss events between frames.

Guidelines for handling gaps:
- Report ONLY what is directly observable in the current screenshot
- Do NOT infer what "must have happened" between screenshots
- If the current state seems inconsistent with what you'd expect, note this in uncertainty_notes
  (e.g., "Boss health bar no longer visible - fight may have ended or player fled")
- When you see result screens (death, victory), report the event but avoid assuming the cause
- If location changed unexpectedly, report the new location without inferring the path taken
- Prefer setting fields to null over guessing values you cannot observe

## General Guidelines

- Be conservative - if you're not sure, return "noop"
- Always set screen_type based on what's displayed
- Include brief reasoning for your decision
- Use uncertainty_notes when something seems discontinuous or unclear"""


class ScreenshotAnalyzer:
    """Analyzes game screenshots using GPT vision model."""
    
    def __init__(self, client: OpenAI | None = None):
        """Initialize analyzer with OpenAI client.
        
        Args:
            client: OpenAI client instance. If None, creates one from config.
        """
        self.client = client or get_openai_client()
        self.model = MODEL_NAME
    
    def analyze(self, image_base64: str) -> StateUpdate:
        """Analyze a screenshot and return state update.
        
        Args:
            image_base64: Base64-encoded PNG image
            
        Returns:
            StateUpdate with detected changes or noop
        """
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Analyze this Clair Obscur: Expedition 33 screenshot and determine if there's any game state to update."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ]
        
        # Log the request
        log_openai_request(logger, self.model, messages, StateUpdate)
        
        logger.debug(f"Sending request to OpenAI model: {self.model}")
        
        response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=messages,
            response_format=StateUpdate,
        )
        
        parsed_result = response.choices[0].message.parsed
        
        # Log the response
        log_openai_response(logger, response, parsed_result)
        
        return parsed_result
