from __future__ import annotations

import base64
import hashlib
import json
from io import BytesIO
from pathlib import Path

from PIL import Image


def encode_image_for_dataset(image_path: Path, max_size: int = 512) -> str:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    if max(width, height) > max_size:
        ratio = max_size / max(width, height)
        image = image.resize((int(width * ratio), int(height * ratio)), Image.LANCZOS)

    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def load_validation_text_entries(valid_texts_path: Path) -> tuple[list[dict], dict[str, set[int]], dict[int, set[str]]]:
    ordered_texts: list[str] = []
    text_to_images: dict[str, set[int]] = {}
    image_to_texts: dict[int, set[str]] = {}

    for line in valid_texts_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        text = row["text"]
        if text not in text_to_images:
            ordered_texts.append(text)
            text_to_images[text] = set()

        for image_id in row.get("image_ids", []):
            image_id_int = int(image_id)
            text_to_images[text].add(image_id_int)
            image_to_texts.setdefault(image_id_int, set()).add(text)

    candidates = []
    for index, text in enumerate(ordered_texts):
        candidates.append(
            {
                "candidate_id": f"text:{index}",
                "text": text,
                "image_ids": sorted(text_to_images[text]),
                "text_type_label": "验证文本",
            }
        )

    return candidates, text_to_images, image_to_texts


def build_image_id_to_filename_map(
    *,
    valid_imgs_path: Path,
    image_dir: Path,
    filenames: list[str],
    cache_path: Path,
) -> dict[int, str | None]:
    if cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if payload.get("valid_imgs_path") == str(valid_imgs_path) and payload.get("image_dir") == str(image_dir):
            return {int(key): value for key, value in payload.get("mapping", {}).items()}

    digest_to_filename: dict[str, str] = {}
    for filename in filenames:
        digest = hashlib.sha1(encode_image_for_dataset(image_dir / filename).encode("utf-8")).hexdigest()
        digest_to_filename[digest] = filename

    mapping: dict[int, str | None] = {}
    with valid_imgs_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            image_id_raw, b64 = line.split("\t", 1)
            digest = hashlib.sha1(b64.encode("utf-8")).hexdigest()
            mapping[int(image_id_raw)] = digest_to_filename.get(digest)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "valid_imgs_path": str(valid_imgs_path),
                "image_dir": str(image_dir),
                "mapping": {str(key): value for key, value in mapping.items()},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return mapping
