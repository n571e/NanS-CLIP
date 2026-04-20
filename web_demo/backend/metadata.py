import json
from collections import defaultdict
from pathlib import Path


def build_metadata_index(augmented_annotations_path: Path, image_metadata_path: Path) -> dict[str, dict]:
    augmented_rows = json.loads(augmented_annotations_path.read_text(encoding="utf-8"))
    metadata_rows = {}
    for line in image_metadata_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        filename = row.get("filename")
        if filename:
            metadata_rows[filename] = row

    grouped = {}
    text_fields = ("modern_chinese", "ancient_style", "keywords")
    for row in augmented_rows:
        filename = row["filename"]
        item = grouped.setdefault(
            filename,
            {
                "filename": filename,
                "title": row.get("title", ""),
                "texts": {field: [] for field in text_fields},
            },
        )
        for field in text_fields:
            value = (row.get(field) or "").strip()
            if value:
                item["texts"][field].append(value)

    index = {}
    for filename, item in grouped.items():
        meta = metadata_rows.get(filename, {})
        representative_texts = {}
        for field, values in item["texts"].items():
            representative_texts[field] = values[0] if values else ""

        index[filename] = {
            "filename": filename,
            "title": item.get("title") or meta.get("title", ""),
            "source": meta.get("source", ""),
            "original_url": meta.get("original_url", ""),
            "description": meta.get("description", ""),
            "texts": item["texts"],
            "representative_texts": representative_texts,
        }

    return index
