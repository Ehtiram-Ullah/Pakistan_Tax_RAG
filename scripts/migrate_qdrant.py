"""Copy the existing local collection to cloud without re-embedding or deleting data."""
import argparse
from contextlib import ExitStack, closing
from math import isclose

from qdrant_client import QdrantClient, models
from settings import COLLECTION_NAME, cloud_options, local_path


def migrate_collection(source, destination, batch_size=100):
    name = COLLECTION_NAME
    if destination.collection_exists(name):
        raise ValueError("Destination collection already exists; refusing to overwrite it.")
    config = source.get_collection(name).config.params
    vectors = config.vectors
    if not isinstance(vectors, models.VectorParams) or vectors.size != 384 or vectors.distance != models.Distance.COSINE:
        raise ValueError("Source must contain unnamed 384-dimensional cosine vectors.")
    expected = source.count(name, exact=True).count
    if expected == 0:
        raise ValueError("Source collection is empty; nothing to migrate.")
    destination.create_collection(collection_name=name, vectors_config=vectors)
    offset = None
    copied = 0
    while True:
        points, offset = source.scroll(name, limit=batch_size, offset=offset, with_payload=True, with_vectors=True)
        destination.upsert(name, points=[models.PointStruct(id=p.id, vector=p.vector, payload=p.payload) for p in points], wait=True)
        # Validate every copied ID, payload, and vector; cosine normalization can
        # introduce tiny float32 rounding differences on the remote server.
        saved = {str(p.id): p for p in destination.retrieve(name, ids=[p.id for p in points], with_payload=True, with_vectors=True)}
        for point in points:
            remote = saved.get(str(point.id))
            if remote is None or remote.payload != point.payload or len(remote.vector) != len(point.vector) or not all(isclose(a, b, rel_tol=1e-5, abs_tol=1e-6) for a, b in zip(remote.vector, point.vector)):
                raise RuntimeError("Copied point verification failed; inspect the destination before retrying.")
        copied += len(points)
        print(f"Copied and verified {copied}/{expected} passages.")
        if offset is None:
            break
    if copied != expected or destination.count(name, exact=True).count != expected:
        raise RuntimeError("Collection counts differ; keep the source stopped during migration.")
    return copied


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Create the cloud collection and copy points; otherwise only inspect.")
    args = parser.parse_args()
    path = local_path()
    if not (path / "meta.json").is_file():
        raise ValueError("Source meta.json is missing. QDRANT_PATH must point to the full local Qdrant directory.")
    with ExitStack() as stack:
        destination = stack.enter_context(closing(QdrantClient(**cloud_options())))
        source = stack.enter_context(closing(QdrantClient(path=str(path))))
        count = source.count(COLLECTION_NAME, exact=True).count
        exists = destination.collection_exists(COLLECTION_NAME)
        print(f"Source: {count} passages. Cloud collection already exists: {exists}.")
        if args.apply:
            migrate_collection(source, destination)
        else:
            print("Inspection only. Stop the local API, then use --apply to copy. Existing cloud collections are never overwritten.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Provider exceptions can include URLs or credentials; print only our own messages.
        safe = str(error) if type(error) in (ValueError, RuntimeError) else type(error).__name__
        raise SystemExit(f"Migration stopped: {safe}") from None
