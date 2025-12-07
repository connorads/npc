"""
Zone Data Transformer
Converts nested zone-format JSON into flat game knowledge entries.

Usage:
    python -m redis_setup.transform_zone input.json [--output output.json] [--append]
"""

import argparse
import json
import re
import sys
from pathlib import Path


def slugify(text: str, suffix: str = "") -> str:
    """Convert text to a URL-friendly slug for use as an ID."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text)
    slug = text.strip("-")
    if suffix:
        suffix_slug = slugify(suffix)
        if suffix_slug and suffix_slug not in slug:
            slug = f"{slug}-{suffix_slug}"
    return slug


def extract_area_entry(zone: dict) -> dict:
    """Extract the area-level entry from zone data."""
    zone_name = zone.get("zone_name", "")
    location_details = zone.get("location_details", {})

    description_parts = []
    if zone.get("zone_type"):
        description_parts.append(f"Type: {zone['zone_type']}")
    if location_details.get("aesthetics"):
        description_parts.append(location_details["aesthetics"])
    if location_details.get("continent"):
        description_parts.append(f"Located in {location_details['continent']}")

    requirements = []
    if location_details.get("required_ability"):
        requirements.append(location_details["required_ability"])
    if location_details.get("act_accessible"):
        requirements.append(f"Accessible in {location_details['act_accessible']}")

    return {
        "id": slugify(zone_name),
        "name": zone_name,
        "type": "Area",
        "area": location_details.get("continent", ""),
        "location": location_details.get("access_location", ""),
        "description": " ".join(description_parts),
        "tips": "",
        "rewards": "",
        "weakness": "",
        "resistance": "",
        "requirements": ". ".join(requirements),
    }


def extract_boss_entries(zone: dict) -> list[dict]:
    """Extract boss entries from zone data."""
    zone_name = zone.get("zone_name", "")
    entries = []

    for boss in zone.get("bosses", []):
        rewards = boss.get("rewards", [])
        if isinstance(rewards, list):
            rewards = ", ".join(rewards)

        entries.append(
            {
                "id": slugify(boss.get("name", ""), zone_name),
                "name": boss.get("name", ""),
                "type": "Boss",
                "area": zone_name,
                "location": boss.get("location", ""),
                "description": f"{boss.get('type', 'Boss')}. {boss.get('difficulty', '')}".strip(
                    ". "
                ),
                "tips": "",
                "rewards": rewards,
                "weakness": "",
                "resistance": "",
                "requirements": "",
            }
        )

        # Add exit impact as a tip if present
        if boss.get("exit_impact"):
            entries[-1]["tips"] = boss["exit_impact"]

    return entries


def extract_enemy_entries(zone: dict) -> list[dict]:
    """Extract enemy entries from zone data."""
    zone_name = zone.get("zone_name", "")
    entries = []

    for enemy in zone.get("enemies", []):
        affinities = enemy.get("elemental_affinities", {})

        weakness = ""
        resistance = ""
        if isinstance(affinities, dict):
            weakness = affinities.get("weak", "")
            resist_parts = []
            if affinities.get("resist"):
                resist_parts.append(f"Resist: {affinities['resist']}")
            if affinities.get("absorb"):
                resist_parts.append(f"Absorb: {affinities['absorb']}")
            resistance = ", ".join(resist_parts)
        elif isinstance(affinities, str):
            # Handle string format like "N/A (Source does not specify...)"
            weakness = affinities

        entries.append(
            {
                "id": slugify(enemy.get("name", ""), zone_name),
                "name": enemy.get("name", ""),
                "type": "Enemy",
                "area": zone_name,
                "location": enemy.get("location_details", ""),
                "description": "",
                "tips": "",
                "rewards": enemy.get("drop_or_reward", ""),
                "weakness": weakness,
                "resistance": resistance,
                "requirements": "",
            }
        )

    return entries


def extract_npc_entries(zone: dict) -> list[dict]:
    """Extract NPC entries from zone data."""
    zone_name = zone.get("zone_name", "")
    entries = []

    for npc in zone.get("npcs", []):
        npc_type = npc.get("type", "NPC")
        entry_type = "Merchant" if "merchant" in npc_type.lower() else "NPC"

        wares = npc.get("wares", [])
        if isinstance(wares, list):
            wares = ", ".join(wares)

        description = npc.get("interaction", "")
        if wares:
            description = f"Sells: {wares}"

        entries.append(
            {
                "id": slugify(npc.get("name", ""), zone_name),
                "name": npc.get("name", ""),
                "type": entry_type,
                "area": zone_name,
                "location": npc.get("location", ""),
                "description": description,
                "tips": "",
                "rewards": "",
                "weakness": "",
                "resistance": "",
                "requirements": "",
            }
        )

    return entries


def extract_collectible_entries(zone: dict) -> list[dict]:
    """Extract collectible entries from zone data."""
    zone_name = zone.get("zone_name", "")
    entries = []
    collectibles = zone.get("collectibles", {})

    # Pictos
    for picto in collectibles.get("pictos", []):
        if isinstance(picto, dict):
            entries.append(
                {
                    "id": slugify(picto.get("name", ""), zone_name),
                    "name": picto.get("name", ""),
                    "type": "Picto",
                    "area": zone_name,
                    "location": picto.get("acquisition", ""),
                    "description": f"Picto found in {zone_name}",
                    "tips": picto.get("acquisition", ""),
                    "rewards": "",
                    "weakness": "",
                    "resistance": "",
                    "requirements": "",
                }
            )

    # Journals
    for journal in collectibles.get("journals", []):
        if isinstance(journal, dict):
            entries.append(
                {
                    "id": slugify(journal.get("name", ""), zone_name),
                    "name": journal.get("name", ""),
                    "type": "Journal",
                    "area": zone_name,
                    "location": journal.get("location", ""),
                    "description": f"Expedition journal found in {zone_name}",
                    "tips": "",
                    "rewards": "",
                    "weakness": "",
                    "resistance": "",
                    "requirements": "",
                }
            )

    return entries


def transform_zone(zone: dict) -> list[dict]:
    """Transform a zone-format JSON into flat game knowledge entries."""
    entries = []

    # Area-level entry
    entries.append(extract_area_entry(zone))

    # Bosses
    entries.extend(extract_boss_entries(zone))

    # Enemies
    entries.extend(extract_enemy_entries(zone))

    # NPCs
    entries.extend(extract_npc_entries(zone))

    # Collectibles
    entries.extend(extract_collectible_entries(zone))

    return entries


def main():
    parser = argparse.ArgumentParser(
        description="Transform zone JSON to game knowledge format"
    )
    parser.add_argument("input", help="Input zone JSON file")
    parser.add_argument("--output", "-o", help="Output file (default: stdout)")
    parser.add_argument(
        "--append",
        "-a",
        help="Append to existing game_knowledge.json instead of outputting",
        action="store_true",
    )
    args = parser.parse_args()

    # Load input
    with open(args.input, "r") as f:
        zone_data = json.load(f)

    # Transform
    entries = transform_zone(zone_data)

    if args.append:
        # Append to existing game_knowledge.json
        project_dir = Path(__file__).parent.parent
        knowledge_file = project_dir / "data" / "game_knowledge.json"

        existing = []
        if knowledge_file.exists():
            with open(knowledge_file, "r") as f:
                existing = json.load(f)

        # Check for duplicate IDs
        existing_ids = {e["id"] for e in existing}
        new_entries = [e for e in entries if e["id"] not in existing_ids]
        skipped = len(entries) - len(new_entries)

        if skipped:
            print(f"Skipped {skipped} entries with duplicate IDs", file=sys.stderr)

        combined = existing + new_entries

        with open(knowledge_file, "w") as f:
            json.dump(combined, f, indent=2)

        print(f"Added {len(new_entries)} entries to {knowledge_file}", file=sys.stderr)

    elif args.output:
        with open(args.output, "w") as f:
            json.dump(entries, f, indent=2)
        print(f"Wrote {len(entries)} entries to {args.output}", file=sys.stderr)

    else:
        print(json.dumps(entries, indent=2))


if __name__ == "__main__":
    main()
