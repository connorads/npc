"""
Game Knowledge Vector Database Setup
Loads game knowledge data into Redis with vector embeddings for semantic search.
"""

import json
import os
import re

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
INDEX_NAME = "idx:game"
KEY_PREFIX = "game:"
VECTOR_DIM = 1536  # OpenAI text-embedding-3-small output dimension
EMBEDDING_MODEL = "text-embedding-3-small"


def slugify(text: str) -> str:
    """Convert text to a URL-friendly slug for use as an ID."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-")


def create_embedding_text(entry: dict) -> str:
    """Combine relevant fields for rich semantic search embeddings."""
    parts = [
        entry.get("name", ""),
        entry.get("type", ""),
        entry.get("area", ""),
        entry.get("description", ""),
        entry.get("tips", ""),
        entry.get("location", ""),
        entry.get("rewards", ""),
        entry.get("requirements", ""),
    ]
    return " ".join(p for p in parts if p)


# --- Connect to Redis ---
client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD,
    decode_responses=False,
)

print("Connected to Redis")

# --- Load Game Knowledge Data ---
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(project_dir, "data/game_knowledge.json"), "r") as f:
    entries = json.load(f)

print(f"Loaded {len(entries)} entries")

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
print("Generating embeddings via OpenAI API...")
embedding_texts = [create_embedding_text(entry) for entry in entries]
embeddings = np.array(get_embeddings(embedding_texts), dtype=np.float32)

# --- Store Entries in Redis ---
print("Storing entries in Redis...")
pipeline = client.pipeline()

for entry, embedding in zip(entries, embeddings):
    # Generate ID if not present
    entry_id = entry.get("id") or slugify(entry["name"])
    key = f"{KEY_PREFIX}{entry_id}"

    doc = {
        "id": entry_id,
        "name": entry.get("name", ""),
        "type": entry.get("type", ""),
        "area": entry.get("area", ""),
        "location": entry.get("location", ""),
        "description": entry.get("description", ""),
        "tips": entry.get("tips", ""),
        "rewards": entry.get("rewards", ""),
        "weakness": entry.get("weakness", ""),
        "resistance": entry.get("resistance", ""),
        "requirements": entry.get("requirements", ""),
        "embedding": embedding.tobytes(),
    }

    pipeline.hset(key, mapping=doc)

pipeline.execute()
print(f"Stored {len(entries)} entries")

# --- Create Search Index ---
print("Creating search index...")

# Drop existing index if it exists
try:
    client.ft(INDEX_NAME).dropindex(delete_documents=False)
    print("Dropped existing index")
except Exception:
    pass

# Define schema
schema = [
    TextField("name", weight=2.0),
    TagField("type"),
    TagField("area"),
    TextField("location"),
    TextField("description"),
    TextField("tips"),
    TextField("rewards"),
    TextField("weakness"),
    TextField("resistance"),
    TextField("requirements"),
    VectorField(
        "embedding",
        "FLAT",
        {
            "TYPE": "FLOAT32",
            "DIM": VECTOR_DIM,
            "DISTANCE_METRIC": "COSINE",
        },
    ),
]

# Create index
client.ft(INDEX_NAME).create_index(
    schema,
    definition=IndexDefinition(prefix=[KEY_PREFIX], index_type=IndexType.HASH),
)

print(f"Created index: {INDEX_NAME}")
print("\n--- Setup Complete ---")
print(f"Entries stored: {len(entries)}")
print(f"Index: {INDEX_NAME}")
print(f"Key prefix: {KEY_PREFIX}")
