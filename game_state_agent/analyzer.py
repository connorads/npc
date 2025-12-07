"""Screenshot analysis using OpenAI's vision model with structured outputs."""

import logging
from openai import OpenAI

from .config import MODEL_NAME, get_openai_client
from .logging_config import log_openai_request, log_openai_response
from .models import StateUpdate

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are an expert Elden Ring player analyzing screenshots from the video game Elden Ring to track game state.

Your task is to examine each screenshot and identify relevant game state information.

## What to Detect

1. **Player Location**: Area names displayed when entering new areas (bottom-center or top-left text), grace site names, or recognizable landmarks. Regions include "Limgrave", "Liurnia of the Lakes", "Caelid", "Altus Plateau", "Mountaintops of the Giants", and specific locations like "Stormveil Castle", "Raya Lucaria Academy", "Roundtable Hold", etc.

2. **Inventory/Equipment**: Inventory screens, equipment menus with dark backgrounds, item icons and descriptions.

3. **Boss Encounters**: Large health bar at the BOTTOM of the screen with boss name above it. Boss names are displayed prominently. Report the boss name and estimate HP percentage from the bar fill.

4. **Game Events**:
   - "YOU DIED" - Large red text on dark/black screen
   - "GREAT ENEMY FELLED" or "LEGEND FELLED" - Gold text indicating boss defeat
   - "Site of Grace Discovered" - Notification when finding a new grace

5. **Sites of Grace**: When resting at a grace, a menu appears with the grace name at the top and options like "Level Up", "Flasks", "Memorize Spells", etc.

6. **Time of Day**: Determine from sky/lighting in outdoor gameplay screenshots:
   - Day: Bright sky, sun visible, warm lighting
   - Night: Dark blue/black sky, stars, moon, darker environment

7. **Player Stats (HUD)**:
   - HP/FP bars: Top-left corner, red bar (HP) and blue bar (FP)
   - Runes: Bottom-right corner, displayed as a number
   - Estimate HP percentage from bar fill if visible

## Screen Types

Identify the current screen type:
- "gameplay" - Active gameplay, world visible
- "inventory" - Inventory menu open
- "equipment" - Equipment/armor menu
- "map" - World map open
- "status" - Character status/stats screen
- "grace_menu" - Resting at Site of Grace
- "loading" - Loading screen
- "death_screen" - "YOU DIED" screen
- "cutscene" - Cinematic/cutscene playing

## Update Types

Use the appropriate update_type:
- "noop" - No relevant game state visible
- "location" - New location/area detected
- "inventory" - Inventory screen with items
- "boss_encounter" - Boss health bar visible
- "game_event" - Death, boss defeat, or grace discovery
- "grace_rest" - Resting at Site of Grace
- "stats" - Player stats visible in HUD/menu
- "multiple" - Multiple types of information detected

## IMPORTANT - Temporal Limitations

Screenshots are captured every ~5 seconds. You may miss events between frames.

Guidelines for handling gaps:
- Report ONLY what is directly observable in the current screenshot
- Do NOT infer what "must have happened" between screenshots
- If the current state seems inconsistent with what you'd expect, note this in uncertainty_notes
  (e.g., "Boss health bar no longer visible - fight may have ended or player left the area")
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
                        "text": "Analyze this Elden Ring screenshot and determine if there's any game state to update."
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
