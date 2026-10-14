import os
from contextlib import closing
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from qdrant_client import QdrantClient, models
from settings import COLLECTION_NAME, cloud_options, create_qdrant_client
from scripts.migrate_qdrant import migrate_collection


class DeploymentTests(unittest.TestCase):
    def test_cloud_uses_https_key_and_bounded_timeout(self):
        with patch.dict(os.environ, {"QDRANT_URL": "https://example.cloud.qdrant.io:6333/", "QDRANT_API_KEY": "test-key"}, clear=True):
            with patch("qdrant_client.QdrantClient") as constructor:
                create_qdrant_client()
                constructor.assert_called_once_with(url="https://example.cloud.qdrant.io:6333", api_key="test-key", timeout=15, prefer_grpc=False)

    def test_vercel_never_silently_creates_local_database(self):
        with patch.dict(os.environ, {"VERCEL": "1"}, clear=True):
            with self.assertRaises(ValueError):
                create_qdrant_client()

    def test_partial_or_insecure_cloud_config_is_rejected(self):
        for config in ({"QDRANT_API_KEY": "secret"}, {"QDRANT_URL": "https://example.com"}, {"QDRANT_URL": "http://example.com", "QDRANT_API_KEY": "secret"}):
            with patch.dict(os.environ, config, clear=True), self.assertRaises(ValueError):
                cloud_options()

    def test_missing_local_index_is_not_created(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"QDRANT_PATH": directory}, clear=True), self.assertRaises(ValueError):
                create_qdrant_client()
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_migration_preserves_ids_vectors_payloads_and_refuses_overwrite(self):
        with closing(QdrantClient(":memory:")) as source, closing(QdrantClient(":memory:")) as target:
            source.create_collection(COLLECTION_NAME, vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE))
            source.upsert(COLLECTION_NAME, points=[models.PointStruct(id=i, vector=[1.0] + [0.0] * 383, payload={"text": f"Passage {i}", "source": "law.pdf", "page": i}) for i in range(3)])
            self.assertEqual(migrate_collection(source, target, batch_size=2), 3)
            self.assertEqual(target.retrieve(COLLECTION_NAME, ids=[2])[0].payload["text"], "Passage 2")
            with self.assertRaisesRegex(ValueError, "already exists"):
                migrate_collection(source, target)
            self.assertEqual(source.count(COLLECTION_NAME).count, 3)
            self.assertEqual(target.count(COLLECTION_NAME).count, 3)

    def test_missing_bundled_model_on_vercel_fails_without_download(self):
        from retrieval import load_model
        with patch.dict(os.environ, {"VERCEL": "1"}), patch("retrieval.MODEL_DIR") as directory:
            directory.is_dir.return_value = False
            with patch("sentence_transformers.SentenceTransformer") as model:
                with self.assertRaisesRegex(RuntimeError, "Bundled"):
                    load_model()
                model.assert_not_called()
