import hashlib
import pickle
from pathlib import Path


def build_cache_key(annotation_path: Path, model_name: str, checkpoint_path: str | None) -> str:
    annotation_digest = hashlib.sha256(annotation_path.read_bytes()).hexdigest()
    material = "::".join(
        [
            model_name,
            checkpoint_path or "__no_checkpoint__",
            annotation_digest,
        ]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


class EmbeddingCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path_for_key(self, key: str) -> Path:
        return self.cache_dir / f"{key}.pkl"

    def read(self, key: str):
        path = self._path_for_key(key)
        if not path.exists():
            return None
        with path.open("rb") as handle:
            return pickle.load(handle)

    def write(self, key: str, payload):
        path = self._path_for_key(key)
        with path.open("wb") as handle:
            pickle.dump(payload, handle)
