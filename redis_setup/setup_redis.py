"""
Elden Ring NPC Vector Database Setup
Loads NPC data into Redis with vector embeddings for semantic search.
"""

import json
import os

import numpy as np
import redis
from dotenv import load_dotenv
from openai import OpenAI
from redis.commands.search.field import TagField, TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType

# --- Configuration ---
load_dotenv()
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
INDEX_NAME = "idx:npcs"
NPC_PREFIX = "npc:"
VECTOR_DIM = 1536  # OpenAI text-embedding-3-small output dimension
EMBEDDING_MODEL = "text-embedding-3-small"

# --- Connect to Redis ---
# Note: decode_responses=False for binary vector data
client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD,
    decode_responses=False
)

print("Connected to Redis")

# --- Load NPC Data ---
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(project_dir, "data/npcs.json"), "r") as f:
    npcs = json.load(f)

print(f"Loaded {len(npcs)} NPCs")

# --- Initialize OpenAI Client ---
openai_client = OpenAI()


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Get embeddings for a list of texts using OpenAI API."""
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]

# --- Create Embeddings ---
# Combine description + lore + dialogue + tips for richer embeddings
def create_embedding_text(npc):
    parts = [
        npc.get("description", ""),
        npc.get("lore", ""),
        npc.get("dialogue", ""),
        npc.get("how_to_beat_tips", "")
    ]
    return " ".join(parts)

print("Generating embeddings via OpenAI API...")
embedding_texts = [create_embedding_text(npc) for npc in npcs]
embeddings = np.array(get_embeddings(embedding_texts), dtype=np.float32)

# --- Store NPCs in Redis ---
print("Storing NPCs in Redis...")
pipeline = client.pipeline()

for npc, embedding in zip(npcs, embeddings):
    key = f"{NPC_PREFIX}{npc['id']}"

    # Prepare document - convert lists to comma-separated strings for TAG fields
    doc = {
        "id": npc["id"],
        "name": npc["name"],
        "race": npc["race"],
        "role": npc["role"],
        "locations": ",".join(npc["locations"]),
        "region": npc["region"],
        "affiliation": npc["affiliation"],
        "quest": npc["quest"],
        "is_hostile": str(npc["is_hostile"]).lower(),
        "becomes_hostile": str(npc["becomes_hostile"]).lower(),
        "drops": ",".join(npc["drops"]) if npc["drops"] else "",
        "description": npc["description"],
        "lore": npc["lore"],
        "dialogue": npc.get("dialogue", ""),
        "weakness": npc.get("weakness", ""),
        "resistance": npc.get("resistance", ""),
        "how_to_beat_tips": npc.get("how_to_beat_tips", ""),
        "embedding": embedding.tobytes()  # Store as raw bytes
    }

    pipeline.hset(key, mapping=doc)

pipeline.execute()
print(f"Stored {len(npcs)} NPCs")

# --- Create Search Index ---
print("Creating search index...")

# Drop existing index if it exists
try:
    client.ft(INDEX_NAME).dropindex(delete_documents=False)
    print("Dropped existing index")
except:
    pass

# Define schema
schema = (
    TextField("name", weight=2.0),
    TextField("description"),
    TextField("lore"),
    TextField("dialogue"),
    TagField("race"),
    TagField("role"),
    TagField("region"),
    TagField("affiliation"),
    TagField("quest"),
    TagField("is_hostile"),
    TagField("becomes_hostile"),
    TextField("weakness"),
    TextField("resistance"),
    TextField("how_to_beat_tips"),
    VectorField(
        "embedding",
        "FLAT",
        {
            "TYPE": "FLOAT32",
            "DIM": VECTOR_DIM,
            "DISTANCE_METRIC": "COSINE",
        },
    ),
)

# Create index
client.ft(INDEX_NAME).create_index(
    schema,
    definition=IndexDefinition(prefix=[NPC_PREFIX], index_type=IndexType.HASH),
)

print(f"Created index: {INDEX_NAME}")
print("\n--- Setup Complete ---")
print(f"NPCs stored: {len(npcs)}")
print(f"Index: {INDEX_NAME}")
print(f"Key prefix: {NPC_PREFIX}")
