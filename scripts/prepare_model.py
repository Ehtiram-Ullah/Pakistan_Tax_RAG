"""Bundle the existing PyTorch model at build time, not on a user's first request."""
from huggingface_hub import snapshot_download
from settings import MODEL_DIR, MODEL_ID, MODEL_REVISION


def main():
    snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=MODEL_DIR,
        allow_patterns=["*.json", "*.txt", "model.safetensors", "1_Pooling/config.json"],
    )
    # Also test a real embedding so a broken model fails the build.
    from retrieval import load_model
    model = load_model()
    vector = model.encode("Pakistan income tax", normalize_embeddings=True)
    if vector.shape != (384,):
        raise RuntimeError("Expected the corpus-compatible 384-dimensional embedding.")
    size = sum(p.stat().st_size for p in MODEL_DIR.rglob("*") if p.is_file())
    print(f"Bundled {MODEL_ID} at {MODEL_REVISION}: {size / 1024**2:.1f} MiB; embedding verified.")


if __name__ == "__main__":
    main()
