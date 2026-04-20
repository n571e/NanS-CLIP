import base64
import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from PIL import Image


def build_dataset_base64(image_path: Path, max_size: int = 512) -> str:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    if max(width, height) > max_size:
        ratio = max_size / max(width, height)
        image = image.resize((int(width * ratio), int(height * ratio)), Image.LANCZOS)

    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


class DemoBackendUtilityTests(unittest.TestCase):
    def test_build_model_cards_marks_missing_checkpoint(self):
        from web_demo.backend.demo_backend import build_model_cards

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            existing = tmp / "best_lora.pt"
            existing.write_bytes(b"demo")

            cards = build_model_cards(existing)

        self.assertEqual(cards[0]["label"], "Zero-Shot")
        self.assertTrue(cards[0]["ready"])
        self.assertEqual(cards[1]["label"], "DoRA")
        self.assertTrue(cards[1]["ready"])

        missing_cards = build_model_cards(Path("missing.pt"))
        self.assertFalse(missing_cards[1]["ready"])
        self.assertEqual(missing_cards[1]["status"], "checkpoint_missing")

    def test_apply_rank_changes_compares_two_result_lists(self):
        from web_demo.backend.demo_backend import apply_rank_changes

        zero_shot = [
            {"image_id": 10, "rank": 1},
            {"image_id": 20, "rank": 2},
            {"image_id": 30, "rank": 3},
        ]
        dora = [
            {"image_id": 20, "rank": 1},
            {"image_id": 10, "rank": 2},
            {"image_id": 30, "rank": 3},
        ]

        enhanced = apply_rank_changes(zero_shot, dora)

        self.assertEqual(enhanced["zero_shot"][0]["rank_change_vs_other"], -1)
        self.assertEqual(enhanced["dora"][0]["rank_change_vs_other"], 1)
        self.assertEqual(enhanced["dora"][2]["rank_change_vs_other"], 0)

    def test_apply_rank_changes_uses_candidate_id_when_present(self):
        from web_demo.backend.demo_backend import apply_rank_changes

        zero_shot = [
            {"candidate_id": "text:0", "image_id": 10, "rank": 1},
            {"candidate_id": "text:1", "image_id": 10, "rank": 2},
        ]
        dora = [
            {"candidate_id": "text:1", "image_id": 10, "rank": 1},
            {"candidate_id": "text:0", "image_id": 10, "rank": 2},
        ]

        enhanced = apply_rank_changes(zero_shot, dora)

        self.assertEqual(enhanced["zero_shot"][0]["rank_change_vs_other"], -1)
        self.assertEqual(enhanced["dora"][0]["rank_change_vs_other"], 1)

    def test_load_validation_text_entries_builds_ground_truth_maps(self):
        from web_demo.backend.benchmark_data import load_validation_text_entries

        with tempfile.TemporaryDirectory() as tmpdir:
            text_path = Path(tmpdir) / "valid_texts.jsonl"
            rows = [
                {"text_id": 0, "text": "马远 踏歌图", "image_ids": [0]},
                {"text_id": 1, "text": "雷峰塔 夕照", "image_ids": [1]},
                {"text_id": 2, "text": "马远 踏歌图", "image_ids": [2]},
            ]
            with text_path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")

            candidates, text_to_images, image_to_texts = load_validation_text_entries(text_path)

        self.assertEqual([candidate["text"] for candidate in candidates], ["马远 踏歌图", "雷峰塔 夕照"])
        self.assertEqual(candidates[0]["candidate_id"], "text:0")
        self.assertEqual(candidates[0]["image_ids"], [0, 2])
        self.assertEqual(text_to_images["马远 踏歌图"], {0, 2})
        self.assertEqual(image_to_texts[2], {"马远 踏歌图"})

    def test_build_image_id_to_filename_map_matches_dataset_encoding(self):
        from web_demo.backend.benchmark_data import build_image_id_to_filename_map

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            image_dir = tmp / "images"
            image_dir.mkdir()
            cache_path = tmp / "mapping_cache.json"
            valid_imgs_path = tmp / "valid_imgs.tsv"

            matched_image = image_dir / "matched.png"
            Image.new("RGB", (20, 10), color=(120, 50, 30)).save(matched_image)
            other_image = image_dir / "other.png"
            Image.new("RGB", (10, 20), color=(10, 130, 220)).save(other_image)

            matched_b64 = build_dataset_base64(matched_image)
            unknown_bytes = BytesIO()
            Image.new("RGB", (12, 12), color=(10, 10, 10)).save(unknown_bytes, format="JPEG", quality=85)
            unknown_b64 = base64.b64encode(unknown_bytes.getvalue()).decode("utf-8")
            valid_imgs_path.write_text(f"0\t{matched_b64}\n1\t{unknown_b64}\n", encoding="utf-8")

            mapping = build_image_id_to_filename_map(
                valid_imgs_path=valid_imgs_path,
                image_dir=image_dir,
                filenames=["matched.png", "other.png"],
                cache_path=cache_path,
            )

        self.assertEqual(mapping[0], "matched.png")
        self.assertIsNone(mapping[1])

    def test_resolve_existing_path_falls_back_to_source_root_file(self):
        from web_demo.backend.demo_backend import resolve_existing_path

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            worktree_path = tmp / "worktree" / "eval_results_zeroshot.json"
            source_path = tmp / "source" / "eval_results_zeroshot.json"
            source_path.parent.mkdir(parents=True, exist_ok=True)
            source_path.write_text("{}", encoding="utf-8")

            resolved = resolve_existing_path(worktree_path, source_path)

        self.assertEqual(resolved, source_path)

    def test_build_demo_cases_prefers_mapped_candidates_and_skips_unmapped_specs(self):
        from web_demo.backend.demo_backend import build_demo_cases

        custom_specs = [
            {
                "query": "德寿宫",
                "focus": "南宋宫殿遗址",
                "terms": ["德寿宫"],
                "explain": "优先选择可展示来源的标准案例。",
            },
            {
                "query": "苏堤春晓",
                "focus": "西湖十景之一",
                "terms": ["苏堤春晓"],
                "explain": "如果无法恢复来源，不应继续保留在精选案例中。",
            },
        ]
        text_candidates = [
            {
                "candidate_id": "text:0",
                "text": "德寿宫",
                "image_ids": [808],
                "text_type_label": "验证文本",
            },
            {
                "candidate_id": "text:1",
                "text": "南宋 德寿宫 建筑遗址 木格窗 歇山顶",
                "image_ids": [148],
                "text_type_label": "验证文本",
            },
            {
                "candidate_id": "text:2",
                "text": "苏堤春晓",
                "image_ids": [880],
                "text_type_label": "验证文本",
            },
        ]
        image_id_to_filename = {
            148: "deshou.png",
            808: None,
            880: None,
        }

        with patch("web_demo.backend.demo_backend.BENCHMARK_CASE_SPECS", custom_specs):
            cases = build_demo_cases(text_candidates, image_id_to_filename)

        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0]["query"], "德寿宫")
        self.assertEqual(cases[0]["query_text"], "南宋 德寿宫 建筑遗址 木格窗 歇山顶")
        self.assertEqual(cases[0]["image_id"], 148)
        self.assertEqual(cases[0]["filename"], "deshou.png")


if __name__ == "__main__":
    unittest.main()
