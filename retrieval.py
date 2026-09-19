from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

client = QdrantClient(path="storage/qdrant")
model = SentenceTransformer("BAAI/bge-small-en-v1.5")


def search(query,limit=5):
    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    results = client.query_points(
        collection_name="pakistan_tax_law",
        query=query_embedding,
        limit= limit

    )

    return results.points


if __name__ == "__main__":
    query = "How much tax to pay for a car?"
    results = search(query)
    for result in results:
        print("\n--- RESULT ---")
        print("Score:", result.score)
        print("Page:", result.payload["page"])
        print("Text:")
        print(result.payload["text"])