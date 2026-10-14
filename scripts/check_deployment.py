"""Read-only cloud retrieval check. Does not call Gemini or modify the collection."""
from contextlib import closing
from settings import COLLECTION_NAME, cloud_options
from qdrant_client import QdrantClient


def main():
    with closing(QdrantClient(**cloud_options())) as client:
        count = client.count(COLLECTION_NAME, exact=True).count
        if count == 0:
            raise RuntimeError("Cloud collection has no passages.")
        print(f"Cloud collection is reachable: {count} passages.")
    from retrieval import search, close_resources
    try:
        hits = search("How much tax is payable on an annual salary of 1000000 rupees?", limit=3)
        if not hits or not all(hit.payload and hit.payload.get("text") for hit in hits):
            raise RuntimeError("Retrieval returned no usable passage text.")
        print(f"CPU PyTorch embedding + cloud search passed: {len(hits)} evidence passages.")
    finally:
        close_resources()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        raise SystemExit(f"Deployment check failed ({type(error).__name__}). Check cloud credentials, collection migration, and model build.") from None
