"""
Query Clair Obscur: Expedition 33 game knowledge from Redis Vector Database
"""

import os

import numpy as np
import redis
from dotenv import load_dotenv
from openai import OpenAI
from redis.commands.search.query import Query

# --- Configuration ---
load_dotenv()
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
INDEX_NAME = "idx:game"
KEY_PREFIX = "game:"
VECTOR_DIM = 1536  # OpenAI text-embedding-3-small output dimension
EMBEDDING_MODEL = "text-embedding-3-small"

# --- Connect ---
redis_client = redis.Redis(
    host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD, decode_responses=True
)
openai_client = OpenAI()


def get_embedding(text: str) -> list[float]:
    """Get embedding for a single text using OpenAI API."""
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding


def semantic_search(query_text: str, top_k: int = 3, filter_expr: str = "*") -> list:
    """
    Search by semantic similarity.

    Args:
        query_text: Natural language query
        top_k: Number of results to return
        filter_expr: Optional filter (e.g., "@area:{Gestral Village}")

    Returns:
        List of matching entries with scores
    """
    query_embedding = np.array(get_embedding(query_text), dtype=np.float32)

    query = (
        Query(f"({filter_expr})=>[KNN {top_k} @embedding $query_vec AS score]")
        .sort_by("score")
        .return_fields("score", "name", "type", "area", "description", "tips")
        .dialect(2)
    )

    results = redis_client.ft(INDEX_NAME).search(
        query, {"query_vec": query_embedding.tobytes()}
    )

    return results.docs


def filter_search(filter_expr: str) -> list:
    """
    Search by metadata filters.

    Examples:
        "@area:{Gestral Village}"
        "@type:{Merchant}"
        "@type:{Boss}"
        "@type:{Quest}"
        "@type:{Secret}"
    """
    query = Query(filter_expr).return_fields("name", "type", "area", "rewards")
    return redis_client.ft(INDEX_NAME).search(query).docs


def get_entry(entry_id: str) -> dict:
    """Get a specific entry by ID (excludes binary embedding field)."""
    fields = [
        "id",
        "name",
        "type",
        "area",
        "location",
        "description",
        "tips",
        "rewards",
        "weakness",
        "resistance",
        "requirements",
    ]
    values = redis_client.hmget(f"{KEY_PREFIX}{entry_id}", fields)
    return {k: v for k, v in zip(fields, values) if v}


def print_results(results: list, show_description: bool = True) -> None:
    """Pretty print search results."""
    for i, doc in enumerate(results, 1):
        score = getattr(doc, "score", None)
        score_str = f" (similarity: {1 - float(score):.2f})" if score else ""
        print(f"\n{i}. {doc.name}{score_str}")
        print(f"   Type: {doc.type} | Area: {doc.area}")
        if show_description and hasattr(doc, "description") and doc.description:
            desc = (
                doc.description[:150] + "..."
                if len(doc.description) > 150
                else doc.description
            )
            print(f"   {desc}")
        if hasattr(doc, "tips") and doc.tips:
            tips = doc.tips[:150] + "..." if len(doc.tips) > 150 else doc.tips
            print(f"   Tips: {tips}")


def main() -> None:
    """Run example queries to demonstrate the database."""
    print("=" * 60)
    print("CLAIR OBSCUR: EXPEDITION 33 - GAME KNOWLEDGE DATABASE")
    print("=" * 60)

    # Semantic search examples
    print("\n### Search: 'boss with shields and flowers' ###")
    results = semantic_search("boss with shields and flowers")
    print_results(results)

    print("\n### Search: 'how to get weapon for Maelle' ###")
    results = semantic_search("how to get weapon for Maelle")
    print_results(results)

    print("\n### Search: 'secret hidden items' ###")
    results = semantic_search("secret hidden items")
    print_results(results)

    print("\n### Search: 'how to cross the sea' ###")
    results = semantic_search("how to cross the sea")
    print_results(results)

    # Filter examples
    print("\n### Filter: All Bosses ###")
    results = filter_search("@type:{Boss}")
    for doc in results:
        print(f"  - {doc.name}")

    print("\n### Filter: All Merchants ###")
    results = filter_search("@type:{Merchant}")
    for doc in results:
        print(f"  - {doc.name} in {doc.area}")

    print("\n### Filter: All Quests ###")
    results = filter_search("@type:{Quest}")
    for doc in results:
        print(f"  - {doc.name}")

    print("\n### Filter: Secrets ###")
    results = filter_search("@type:{Secret}")
    for doc in results:
        print(f"  - {doc.name}")


if __name__ == "__main__":
    main()
