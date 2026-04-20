from __future__ import annotations

import base64
import io
import json
import os
import time
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import lmdb
import torch
from PIL import Image

from web_demo.backend.benchmark_data import build_image_id_to_filename_map, load_validation_text_entries
from web_demo.backend.cache import EmbeddingCache, build_cache_key
from web_demo.backend.metadata import build_metadata_index


BENCHMARK_CASE_SPECS = [
    {
        "query": "德寿宫",
        "focus": "南宋宫殿遗址",
        "terms": ["德寿宫", "南宋德寿宫"],
        "explain": "用验证集标准文本查询，展示 DoRA 是否把标准答案排得更靠前。",
    },
    {
        "query": "马远",
        "focus": "南宋院体画家",
        "terms": ["马远", "踏歌图", "寒江独钓"],
        "explain": "用南宋画家相关验证文本，观察 DoRA 对领域画家语义的排序变化。",
    },
    {
        "query": "雷峰塔",
        "focus": "西湖塔类地标",
        "terms": ["雷峰塔", "雷峰夕照"],
        "explain": "用西湖地标类验证文本，对比模型是否把标准塔景图像排到前列。",
    },
    {
        "query": "临安",
        "focus": "南宋都城意象",
        "terms": ["临安", "临安城", "南宋皇城"],
        "explain": "用南宋都城相关文本，展示模型对都城遗址与皇城意象的检索能力。",
    },
    {
        "query": "保俶塔",
        "focus": "杭州古塔地标",
        "terms": ["保俶塔"],
        "explain": "用相近地标查询对比 DoRA 是否更稳定地区分塔类视觉语义。",
    },
]


def build_model_cards(dora_checkpoint_path: Path) -> list[dict]:
    dora_ready = dora_checkpoint_path.exists()
    return [
        {
            "key": "zero_shot",
            "label": "Zero-Shot",
            "ready": True,
            "status": "ready",
        },
        {
            "key": "dora",
            "label": "DoRA",
            "ready": dora_ready,
            "status": "ready" if dora_ready else "checkpoint_missing",
        },
    ]


def result_identity(item: dict) -> str:
    if item.get("candidate_id"):
        return item["candidate_id"]
    if item.get("image_id") is not None:
        return f"image:{item['image_id']}"
    return item["filename"]


def apply_rank_changes(zero_shot: list[dict], dora: list[dict]) -> dict[str, list[dict]]:
    zero_rank_map = {result_identity(item): item["rank"] for item in zero_shot}
    dora_rank_map = {result_identity(item): item["rank"] for item in dora}

    def decorate(items: list[dict]) -> list[dict]:
        enhanced = []
        for item in items:
            identity = result_identity(item)
            zero_rank = zero_rank_map.get(identity, item["rank"])
            dora_rank = dora_rank_map.get(identity, item["rank"])
            enhanced.append(item | {"rank_change_vs_other": zero_rank - dora_rank})
        return enhanced

    return {
        "zero_shot": decorate(zero_shot),
        "dora": decorate(dora),
    }


def resolve_source_root(repo_root: Path, explicit_source_root: Path | None = None) -> Path:
    candidates = [
        explicit_source_root,
        Path(os.getenv("NANS_CLIP_SOURCE_ROOT", "")) if os.getenv("NANS_CLIP_SOURCE_ROOT") else None,
        repo_root,
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        candidate = candidate.resolve()
        if (candidate / "data").exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate a usable source root with data/. Set NANS_CLIP_SOURCE_ROOT to your original NanS-CLIP checkout."
    )


def resolve_existing_path(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not locate any of the required files: {', '.join(str(candidate) for candidate in candidates)}")


def pick_featured_image_id(image_ids: list[int], image_id_to_filename: dict[int, str | None]) -> int | None:
    if not image_ids:
        return None
    return next((image_id for image_id in image_ids if image_id_to_filename.get(image_id)), image_ids[0])


def pick_mapped_image_id(image_ids: list[int], image_id_to_filename: dict[int, str | None]) -> int | None:
    return next((image_id for image_id in image_ids if image_id_to_filename.get(image_id)), None)


def build_benchmark_image_items(
    image_to_texts: dict[int, set[str]],
    image_id_to_filename: dict[int, str | None],
    metadata_index: dict[str, dict],
) -> list[dict]:
    items = []
    for image_id in sorted(image_to_texts):
        filename = image_id_to_filename.get(image_id)
        metadata = metadata_index.get(filename, {}) if filename else {}
        preview_text = min(image_to_texts.get(image_id, {""}), key=len)
        items.append(
            {
                "image_id": image_id,
                "filename": filename,
                "title": metadata.get("title") or preview_text or f"验证集图像 #{image_id}",
                "source": metadata.get("source", ""),
                "preview_text": preview_text,
            }
        )
    return items


def decorate_text_candidates(
    text_candidates: list[dict],
    image_id_to_filename: dict[int, str | None],
    metadata_index: dict[str, dict],
) -> list[dict]:
    decorated = []
    for candidate in text_candidates:
        featured_image_id = pick_featured_image_id(candidate["image_ids"], image_id_to_filename)
        filename = image_id_to_filename.get(featured_image_id) if featured_image_id is not None else None
        metadata = metadata_index.get(filename, {}) if filename else {}
        decorated.append(
            candidate
            | {
                "filename": filename,
                "image_id": featured_image_id,
                "title": metadata.get("title") or candidate["text"][:40],
                "source": metadata.get("source", ""),
            }
        )
    return decorated


def build_demo_cases(
    text_candidates: list[dict],
    image_id_to_filename: dict[int, str | None],
) -> list[dict]:
    cases = []
    for spec in BENCHMARK_CASE_SPECS:
        matches = [
            candidate
            for candidate in text_candidates
            if any(term in candidate["text"] for term in spec["terms"])
        ]
        if not matches:
            continue

        mapped_matches = [
            (candidate, pick_mapped_image_id(candidate["image_ids"], image_id_to_filename))
            for candidate in matches
        ]
        mapped_matches = [
            (candidate, mapped_image_id)
            for candidate, mapped_image_id in mapped_matches
            if mapped_image_id is not None
        ]
        if not mapped_matches:
            continue

        mapped_matches.sort(
            key=lambda candidate: (
                len(candidate[0]["text"]),
                candidate[0]["candidate_id"],
            )
        )
        chosen, featured_image_id = mapped_matches[0]
        cases.append(
            {
                "query": spec["query"],
                "query_text": chosen["text"],
                "focus": spec["focus"],
                "explain": spec["explain"],
                "image_id": featured_image_id,
                "filename": image_id_to_filename.get(featured_image_id),
                "ground_truth_image_ids": chosen["image_ids"],
            }
        )
    return cases


def build_examples(demo_cases: list[dict]) -> list[dict]:
    return [
        {
            "query": case["query"],
            "query_text": case["query_text"],
            "title": case["focus"],
            "narrative": case["explain"],
            "filename": case.get("filename"),
        }
        for case in demo_cases
    ]


def count_sources_for_mapping(image_id_to_filename: dict[int, str | None], metadata_index: dict[str, dict]) -> list[dict]:
    counter = Counter()
    for filename in image_id_to_filename.values():
        if not filename:
            continue
        counter[metadata_index.get(filename, {}).get("source", "Unknown")] += 1
    return [{"name": name, "count": count} for name, count in counter.most_common()]


def annotate_image_results(results: list[dict], ground_truth_image_ids: set[int]) -> tuple[list[dict], dict]:
    annotated = []
    hit_count = 0
    first_hit_rank = None
    for item in results:
        is_hit = item["image_id"] in ground_truth_image_ids
        if is_hit:
            hit_count += 1
            first_hit_rank = first_hit_rank or item["rank"]
        annotated.append(
            item
            | {
                "judgement": "标准答案" if is_hit else "非标准答案",
                "judgement_reason": (
                    "该图片是当前验证文本的标准配对图像。"
                    if is_hit
                    else "该图片不在当前验证文本的标准答案集合中。"
                ),
            }
        )
    return annotated, {"hit_count": hit_count, "first_hit_rank": first_hit_rank}


def annotate_text_results(results: list[dict], ground_truth_candidate_ids: set[str]) -> tuple[list[dict], dict]:
    annotated = []
    hit_count = 0
    first_hit_rank = None
    has_ground_truth = bool(ground_truth_candidate_ids)
    for item in results:
        is_hit = item["candidate_id"] in ground_truth_candidate_ids
        if is_hit:
            hit_count += 1
            first_hit_rank = first_hit_rank or item["rank"]
        if has_ground_truth:
            judgement = "标准答案" if is_hit else "非标准答案"
            reason = (
                "该文本是当前验证图像的标准配对文本。"
                if is_hit
                else "该文本不在当前验证图像的标准答案集合中。"
            )
        else:
            judgement = "无标准答案"
            reason = "当前图像不在验证集标准答案范围内，仅展示模型返回结果。"
        annotated.append(item | {"judgement": judgement, "judgement_reason": reason})
    return annotated, {"hit_count": hit_count, "first_hit_rank": first_hit_rank}


@dataclass
class SearchModelConfig:
    key: str
    label: str
    model_name: str
    checkpoint_path: Path | None
    rank: int = 8
    alpha: float = 16.0


class SearchModelRuntime:
    def __init__(
        self,
        config: SearchModelConfig,
        *,
        image_source_path: Path,
        text_source_path: Path,
        image_items: list[dict],
        text_candidates: list[dict],
        image_loader,
        cache: EmbeddingCache,
        pretrained_root: Path,
        device: str,
    ):
        self.config = config
        self.image_source_path = image_source_path
        self.text_source_path = text_source_path
        self.image_items = image_items
        self.text_candidates = text_candidates
        self.image_loader = image_loader
        self.cache = cache
        self.pretrained_root = pretrained_root
        self.device = device
        self._features = None
        self._text_features = None
        self._model = None
        self._preprocess = None

    def _autocast(self):
        if self.device == "cuda":
            return torch.amp.autocast("cuda")
        return nullcontext()

    def _cache_key(self, *, source_path: Path, kind: str) -> str:
        checkpoint_value = str(self.config.checkpoint_path) if self.config.checkpoint_path else None
        return f"{kind}-{build_cache_key(annotation_path=source_path, model_name=self.config.model_name, checkpoint_path=checkpoint_value)}"

    def _image_cache_key(self) -> str:
        return self._cache_key(source_path=self.image_source_path, kind="image")

    def _text_cache_key(self) -> str:
        return self._cache_key(source_path=self.text_source_path, kind="text")

    def _ensure_model(self):
        from cn_clip.clip import load_from_name
        from cn_clip.clip.lora import inject_lora, load_lora_state_dict

        if self._model is None or self._preprocess is None:
            model, preprocess = load_from_name(
                self.config.model_name,
                device="cpu",
                download_root=str(self.pretrained_root),
            )
            if self.config.checkpoint_path and self.config.checkpoint_path.exists():
                inject_lora(model, rank=self.config.rank, alpha=self.config.alpha)
                state_dict = torch.load(self.config.checkpoint_path, map_location="cpu")
                load_lora_state_dict(model, state_dict)
            self._model = model.to(self.device)
            self._model.eval()
            self._preprocess = preprocess

    def ensure_ready(self):
        self._ensure_model()
        if self._features is not None:
            return

        cached = self.cache.read(self._image_cache_key())
        if cached is not None:
            self._features = torch.tensor(cached["features"], dtype=torch.float32)
            return

        batched_features = []
        batch_images = []
        for index, item in enumerate(self.image_items, start=1):
            image = self._preprocess(self.image_loader(item))
            batch_images.append(image)
            if len(batch_images) == 32 or index == len(self.image_items):
                batch_tensor = torch.stack(batch_images).to(self.device)
                with torch.no_grad():
                    with self._autocast():
                        features = self._model.encode_image(batch_tensor)
                        features = features / features.norm(dim=-1, keepdim=True)
                batched_features.append(features.cpu())
                batch_images = []

        matrix = torch.cat(batched_features, dim=0)
        self._features = matrix
        self.cache.write(self._image_cache_key(), {"features": matrix.tolist()})

    def ensure_text_ready(self):
        from cn_clip.clip import tokenize

        self._ensure_model()
        if self._text_features is not None:
            return

        cached = self.cache.read(self._text_cache_key())
        if cached is not None:
            self._text_features = torch.tensor(cached["features"], dtype=torch.float32)
            return

        batched_features = []
        batch_texts = []
        for index, item in enumerate(self.text_candidates, start=1):
            batch_texts.append(item["text"])
            if len(batch_texts) == 128 or index == len(self.text_candidates):
                tokens = tokenize(batch_texts).to(self.device)
                with torch.no_grad():
                    with self._autocast():
                        features = self._model.encode_text(tokens)
                        features = features / features.norm(dim=-1, keepdim=True)
                batched_features.append(features.cpu())
                batch_texts = []

        matrix = torch.cat(batched_features, dim=0)
        self._text_features = matrix
        self.cache.write(self._text_cache_key(), {"features": matrix.tolist()})

    def search(self, query: str, top_k: int) -> list[dict]:
        from cn_clip.clip import tokenize

        self.ensure_ready()

        tokens = tokenize([query]).to(self.device)
        with torch.no_grad():
            with self._autocast():
                text_features = self._model.encode_text(tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        sims = (text_features.cpu() @ self._features.T).squeeze(0)
        values, indices = sims.topk(min(top_k, len(self.image_items)))

        results = []
        for rank, (score, idx) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
            item = self.image_items[idx]
            results.append(
                {
                    "image_id": item["image_id"],
                    "filename": item.get("filename"),
                    "rank": rank,
                    "score": round(float(score), 4),
                    "title": item["title"],
                    "source": item.get("source", ""),
                    "image_url": f"/api/eval-images/{item['image_id']}",
                    "preview_text": item.get("preview_text", ""),
                }
            )
        return results

    def search_image_to_text(self, image: Image.Image, top_k: int) -> list[dict]:
        self.ensure_text_ready()

        image_tensor = self._preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            with self._autocast():
                image_features = self._model.encode_image(image_tensor)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        sims = (image_features.cpu() @ self._text_features.T).squeeze(0)
        values, indices = sims.topk(min(top_k, len(self.text_candidates)))

        results = []
        for rank, (score, idx) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
            item = self.text_candidates[idx]
            results.append(
                {
                    "candidate_id": item["candidate_id"],
                    "filename": item.get("filename"),
                    "image_id": item.get("image_id"),
                    "rank": rank,
                    "score": round(float(score), 4),
                    "title": item["title"],
                    "source": item.get("source", ""),
                    "text": item["text"],
                    "text_type_label": item["text_type_label"],
                }
            )
        return results


class DemoBackend:
    def __init__(self, repo_root: Path, source_root: Path):
        self.repo_root = repo_root
        self.source_root = source_root

        self.augmented_annotations_path = source_root / "data" / "annotations_augmented.json"
        self.image_metadata_path = source_root / "data" / "image_metadata.jsonl"
        self.image_dir = source_root / "data" / "images"
        self.dataset_root = source_root.parent / "clip_data" / "datasets" / "SongDynasty"
        self.valid_texts_path = self.dataset_root / "valid_texts.jsonl"
        self.valid_imgs_path = self.dataset_root / "valid_imgs.tsv"
        self.valid_lmdb_dir = self.dataset_root / "lmdb" / "valid"
        self.pretrained_root = source_root.parent / "clip_data" / "pretrained_weights"
        self.dora_checkpoint_path = source_root.parent / "clip_data" / "experiments" / "dora_song" / "best_lora.pt"
        self.cache = EmbeddingCache(source_root.parent / "clip_data" / "cache" / "web_demo")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.metadata_index = build_metadata_index(self.augmented_annotations_path, self.image_metadata_path)
        self.image_id_to_filename = build_image_id_to_filename_map(
            valid_imgs_path=self.valid_imgs_path,
            image_dir=self.image_dir,
            filenames=sorted(self.metadata_index.keys()),
            cache_path=self.cache.cache_dir / "eval_image_id_to_filename.json",
        )
        self.eval_img_env = lmdb.open(
            str(self.valid_lmdb_dir / "imgs"),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.raw_text_candidates, self.text_to_image_ids, self.image_to_texts = load_validation_text_entries(self.valid_texts_path)
        self.text_candidates = decorate_text_candidates(self.raw_text_candidates, self.image_id_to_filename, self.metadata_index)
        self.image_items = build_benchmark_image_items(self.image_to_texts, self.image_id_to_filename, self.metadata_index)
        self.source_counts = count_sources_for_mapping(self.image_id_to_filename, self.metadata_index)
        self.models = build_model_cards(self.dora_checkpoint_path)
        self.demo_cases = build_demo_cases(self.text_candidates, self.image_id_to_filename)
        self.examples = build_examples(self.demo_cases)
        self.text_candidate_id_by_text = {candidate["text"]: candidate["candidate_id"] for candidate in self.text_candidates}
        self.image_to_candidate_ids: dict[int, set[str]] = {}
        for candidate in self.text_candidates:
            for image_id in candidate["image_ids"]:
                self.image_to_candidate_ids.setdefault(image_id, set()).add(candidate["candidate_id"])
        self.search_runtimes = self._build_search_runtimes()

    @classmethod
    def from_repo_root(cls, repo_root: Path | None = None, source_root: Path | None = None):
        repo_root = (repo_root or Path(__file__).resolve().parents[2]).resolve()
        source_root = resolve_source_root(repo_root, source_root)
        return cls(repo_root, source_root)

    def _load_benchmark_image(self, item: dict) -> Image.Image:
        image_bytes = self.get_benchmark_image_bytes(item["image_id"])
        if image_bytes is not None:
            return Image.open(io.BytesIO(image_bytes)).convert("RGB")
        filename = item.get("filename")
        if not filename:
            raise FileNotFoundError(f"Could not locate benchmark image bytes for image_id={item['image_id']}")
        return Image.open(self.image_dir / filename).convert("RGB")

    def _build_search_runtimes(self):
        configs = [
            SearchModelConfig(
                key="zero_shot",
                label="Zero-Shot",
                model_name="ViT-B-16",
                checkpoint_path=None,
            ),
            SearchModelConfig(
                key="dora",
                label="DoRA",
                model_name="ViT-B-16",
                checkpoint_path=self.dora_checkpoint_path if self.dora_checkpoint_path.exists() else None,
            ),
        ]
        return {
            config.key: SearchModelRuntime(
                config,
                image_source_path=self.valid_imgs_path,
                text_source_path=self.valid_texts_path,
                image_items=self.image_items,
                text_candidates=self.text_candidates,
                image_loader=self._load_benchmark_image,
                cache=self.cache,
                pretrained_root=self.pretrained_root,
                device=self.device,
            )
            for config in configs
        }

    def warm_up(self):
        for runtime in self.search_runtimes.values():
            runtime.ensure_ready()

    def get_image_directory(self):
        return self.image_dir

    def get_benchmark_image_bytes(self, image_id: int):
        with self.eval_img_env.begin() as txn:
            raw = txn.get(str(image_id).encode("utf-8"))
        if raw is None:
            return None
        if isinstance(raw, memoryview):
            raw = raw.tobytes()
        return base64.b64decode(raw.decode("utf-8"))

    def get_summary(self):
        mapped_image_count = sum(1 for filename in self.image_id_to_filename.values() if filename)
        return {
            "hero": {
                "title": "NanS-CLIP",
                "subtitle": "南宋文博标准检索 Demo",
                "description": "所有对比都基于 valid 验证集标准答案，而不是开放式 live 搜索。",
            },
            "benchmark_corpus": {
                "label": "验证集检索池",
                "image_count": len(self.image_items),
                "text_count": len(self.text_candidates),
                "mapped_image_count": mapped_image_count,
                "note": "本页所有命中与否均按 valid 验证集图文配对自动判定；能映射回原始 filename 的样本会继续展示公开来源。",
            },
            "sources": self.source_counts,
            "preset_queries": [case["query"] for case in self.demo_cases],
            "demo_cases": self.demo_cases,
            "models": self.models,
        }

    def get_examples(self):
        return self.examples

    def get_item(self, filename):
        return self.metadata_index.get(filename)

    def _resolve_demo_case(self, query: str) -> dict | None:
        return next((case for case in self.demo_cases if case["query"] == query), None)

    def compare_search(self, query: str, top_k: int):
        query = query.strip()
        if not query:
            raise ValueError("query is required")

        demo_case = self._resolve_demo_case(query)
        if demo_case is not None:
            query_text = demo_case["query_text"]
        elif query in self.text_to_image_ids:
            query_text = query
        else:
            raise ValueError("query must be one of the curated benchmark cases or an exact validation text")

        ground_truth_image_ids = set(self.text_to_image_ids[query_text])

        started = time.perf_counter()
        zero_results = self.search_runtimes["zero_shot"].search(query_text, top_k)
        zero_elapsed = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        dora_results = self.search_runtimes["dora"].search(query_text, top_k)
        dora_elapsed = (time.perf_counter() - started) * 1000

        ranked_results = apply_rank_changes(zero_results, dora_results)
        zero_annotated, zero_stats = annotate_image_results(ranked_results["zero_shot"], ground_truth_image_ids)
        dora_annotated, dora_stats = annotate_image_results(ranked_results["dora"], ground_truth_image_ids)

        return {
            "query": demo_case["query"] if demo_case else query_text,
            "query_text": query_text,
            "demo_case": demo_case,
            "pool_size": len(self.image_items),
            "ground_truth": {
                "image_ids": sorted(ground_truth_image_ids),
                "count": len(ground_truth_image_ids),
                "zero_shot_hit_count": zero_stats["hit_count"],
                "zero_shot_first_hit_rank": zero_stats["first_hit_rank"],
                "dora_hit_count": dora_stats["hit_count"],
                "dora_first_hit_rank": dora_stats["first_hit_rank"],
            },
            "timings_ms": {
                "zero_shot": round(zero_elapsed, 2),
                "dora": round(dora_elapsed, 2),
            },
            "results": {
                "zero_shot": zero_annotated,
                "dora": dora_annotated,
            },
        }

    def _resolve_image_query(self, *, image_id: int | None = None, filename: str | None = None, image_bytes: bytes | None = None):
        if image_id is not None:
            benchmark_item = next((item for item in self.image_items if item["image_id"] == int(image_id)), None)
            if benchmark_item is None:
                raise ValueError("image query is required")
            image = self._load_benchmark_image(benchmark_item)
            return image, {
                "type": "benchmark",
                "image_id": benchmark_item["image_id"],
                "filename": benchmark_item.get("filename"),
                "image_url": f"/api/eval-images/{benchmark_item['image_id']}",
                "label": benchmark_item["title"],
            }, self.image_to_candidate_ids.get(benchmark_item["image_id"], set())

        if filename:
            resolved_image_id = next((candidate_id for candidate_id, candidate_filename in self.image_id_to_filename.items() if candidate_filename == filename), None)
            if resolved_image_id is not None:
                return self._resolve_image_query(image_id=resolved_image_id)
            image_path = self.image_dir / filename
            if not image_path.exists():
                raise ValueError("image query is required")
            image = Image.open(image_path).convert("RGB")
            return image, {
                "type": "source",
                "image_id": None,
                "filename": filename,
                "image_url": f"/api/images/{filename}",
                "label": self.metadata_index.get(filename, {}).get("title", filename),
            }, set()

        if image_bytes:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            return image, {
                "type": "upload",
                "image_id": None,
                "filename": None,
                "image_url": None,
                "label": "上传图片",
            }, set()

        raise ValueError("image query is required")

    def compare_image_to_text(self, top_k: int, *, filename: str | None = None, image_id: int | None = None, image_bytes: bytes | None = None):
        image, query_source, ground_truth_candidate_ids = self._resolve_image_query(
            filename=filename,
            image_id=image_id,
            image_bytes=image_bytes,
        )

        started = time.perf_counter()
        zero_results = self.search_runtimes["zero_shot"].search_image_to_text(image, top_k)
        zero_elapsed = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        dora_results = self.search_runtimes["dora"].search_image_to_text(image, top_k)
        dora_elapsed = (time.perf_counter() - started) * 1000

        ranked_results = apply_rank_changes(zero_results, dora_results)
        zero_annotated, zero_stats = annotate_text_results(ranked_results["zero_shot"], ground_truth_candidate_ids)
        dora_annotated, dora_stats = annotate_text_results(ranked_results["dora"], ground_truth_candidate_ids)

        return {
            "query_source": query_source,
            "pool_size": len(self.text_candidates),
            "ground_truth": {
                "candidate_ids": sorted(ground_truth_candidate_ids),
                "count": len(ground_truth_candidate_ids),
                "zero_shot_hit_count": zero_stats["hit_count"],
                "zero_shot_first_hit_rank": zero_stats["first_hit_rank"],
                "dora_hit_count": dora_stats["hit_count"],
                "dora_first_hit_rank": dora_stats["first_hit_rank"],
            },
            "timings_ms": {
                "zero_shot": round(zero_elapsed, 2),
                "dora": round(dora_elapsed, 2),
            },
            "results": {
                "zero_shot": zero_annotated,
                "dora": dora_annotated,
            },
        }
