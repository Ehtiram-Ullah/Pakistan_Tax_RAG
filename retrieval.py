"""Reuse retrieval resources; serialize access to the embedded Qdrant store."""
import os
from pathlib import Path
from threading import RLock

COLLECTION_NAME = "pakistan_tax_law"
_lock = RLock()
_client = None
_model = None


def search(query, limit=5):
    global _client, _model
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer

    with _lock:
        if _client is None:
            path = os.getenv("QDRANT_PATH", str(Path(__file__).resolve().parent / "storage" / "qdrant"))
            _client = QdrantClient(path=path)
        if _model is None:
            _model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        embedding = _model.encode(query, normalize_embeddings=True).tolist()
        return _client.query_points(collection_name=COLLECTION_NAME, query=embedding, limit=limit).points


def close_resources():
    global _client, _model
    with _lock:
        if _client is not None:
            _client.close()
        _client = None
        _model = None
