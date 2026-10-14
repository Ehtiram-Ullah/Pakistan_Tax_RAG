"""Shared settings for local tools and the deployed API (never print secrets)."""
import os
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
MODEL_ID = "BAAI/bge-small-en-v1.5"
# Same snapshot used to embed the existing local corpus.
MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
MODEL_DIR = ROOT / "models" / "bge-small-en-v1.5"
COLLECTION_NAME = "pakistan_tax_law"


def local_path():
    return Path(os.getenv("QDRANT_PATH") or ROOT / "storage" / "qdrant").resolve()


def cloud_options():
    url = os.getenv("QDRANT_URL", "").strip().rstrip("/")
    key = os.getenv("QDRANT_API_KEY", "").strip()
    parsed = urlparse(url)
    if not key or parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path:
        raise ValueError("Set QDRANT_URL to the HTTPS cluster endpoint and QDRANT_API_KEY to its API key.")
    return {"url": url, "api_key": key, "timeout": 15, "prefer_grpc": False}


def create_qdrant_client():
    from qdrant_client import QdrantClient

    if os.getenv("QDRANT_URL") or os.getenv("QDRANT_API_KEY") or os.getenv("VERCEL"):
        return QdrantClient(**cloud_options())
    path = local_path()
    if not (path / "meta.json").is_file():
        raise ValueError("Local Qdrant index is missing. Configure Qdrant Cloud or restore the existing index.")
    return QdrantClient(path=str(path))
