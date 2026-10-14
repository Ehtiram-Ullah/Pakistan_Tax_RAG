"""Reuse CPU PyTorch embeddings and either cloud or local Qdrant."""
import os
from threading import RLock
from settings import COLLECTION_NAME, MODEL_DIR, MODEL_ID, MODEL_REVISION, create_qdrant_client

_lock = RLock()
_client = None
_model = None


def search(query, limit=5):
    global _client, _model
    with _lock:
        if _client is None:
            _client = create_qdrant_client()
        if _model is None:
            _model = load_model()
        embedding = _model.encode(query, normalize_embeddings=True).tolist()
        return _client.query_points(collection_name=COLLECTION_NAME, query=embedding, limit=limit).points


def load_model():
    # Avoid excessive CPU threads on the one-vCPU deployment.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from sentence_transformers import SentenceTransformer
    import torch

    torch.set_num_threads(1)
    if MODEL_DIR.is_dir():
        return SentenceTransformer(str(MODEL_DIR), device="cpu", local_files_only=True)
    if os.getenv("VERCEL"):
        raise RuntimeError("Bundled embedding model is missing. Run the model preparation build step.")
    return SentenceTransformer(MODEL_ID, revision=MODEL_REVISION, device="cpu")


def close_resources():
    global _client, _model
    with _lock:
        if _client is not None:
            _client.close()
        _client = None
        _model = None
