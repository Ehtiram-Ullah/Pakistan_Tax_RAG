from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


COLLECTION_NAME = "pakistan_tax_law"


def search(query, limit=5):
    client = QdrantClient(path="storage/qdrant")
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=limit
    )

    client.close()

    return results.points