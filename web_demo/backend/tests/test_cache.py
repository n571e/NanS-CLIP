import json
import tempfile
import unittest
from pathlib import Path


class CacheTests(unittest.TestCase):
    def test_build_cache_key_changes_with_input_file_contents(self):
        from web_demo.backend.cache import build_cache_key

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            annotation_path = tmp / "annotations.json"
            annotation_path.write_text(json.dumps([{"filename": "a.png"}]), encoding="utf-8")

            key_a = build_cache_key(
                annotation_path=annotation_path,
                model_name="ViT-B-16",
                checkpoint_path="checkpoint-a.pt",
            )

            annotation_path.write_text(json.dumps([{"filename": "b.png"}]), encoding="utf-8")
            key_b = build_cache_key(
                annotation_path=annotation_path,
                model_name="ViT-B-16",
                checkpoint_path="checkpoint-a.pt",
            )

        self.assertNotEqual(key_a, key_b)

    def test_embedding_cache_round_trip(self):
        from web_demo.backend.cache import EmbeddingCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = EmbeddingCache(Path(tmpdir))
            payload = {"filenames": ["a.png"], "features": [[0.1, 0.2, 0.3]]}
            cache.write("demo-key", payload)
            restored = cache.read("demo-key")

        self.assertEqual(restored, payload)


if __name__ == "__main__":
    unittest.main()
