import tempfile
import unittest
from pathlib import Path


class FakeDemoBackend:
    def __init__(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.image_dir = Path(self.tempdir.name)
        (self.image_dir / "sample.png").write_bytes(b"fake-image")

    def get_summary(self):
        return {
            "hero": {
                "title": "NanS-CLIP",
                "subtitle": "南宋文博标准检索 Demo",
                "description": "所有对比都基于 valid 验证集标准答案，而不是开放式 live 搜索。",
            },
            "benchmark_corpus": {
                "label": "验证集检索池",
                "image_count": 1117,
                "text_count": 4868,
                "mapped_image_count": 1089,
                "note": "本页所有命中与否均按 valid 验证集图文配对自动判定。",
            },
            "sources": [
                {"name": "Baidu Images", "count": 1035},
                {"name": "Wikimedia Commons", "count": 54},
            ],
            "preset_queries": ["德寿宫", "马远", "雷峰塔", "临安", "保俶塔"],
            "demo_cases": [
                {
                    "query": "德寿宫",
                    "query_text": "南宋德寿宫遗址博物馆",
                    "focus": "南宋宫殿遗址",
                    "explain": "用验证集标准文本查询，展示 DoRA 是否把标准答案排得更靠前。",
                    "image_id": 42,
                    "filename": "sample.png",
                }
            ],
            "models": [
                {"key": "zero_shot", "label": "Zero-Shot", "ready": True},
                {"key": "dora", "label": "DoRA", "ready": True},
            ],
        }

    def get_examples(self):
        return [
            {
                "query": "德寿宫",
                "query_text": "南宋德寿宫遗址博物馆",
                "narrative": "先看标准文本，再看标准答案排位。",
            }
        ]

    def get_item(self, filename):
        if filename != "sample.png":
            return None
        return {
            "filename": "sample.png",
            "title": "南宋德寿宫遗址",
            "source": "Wikimedia Commons",
            "original_url": "https://example.com/sample.png",
            "description": "一张来自公开馆藏的南宋相关图像。",
            "representative_texts": {
                "modern_chinese": "南宋德寿宫遗址博物馆",
                "ancient_style": "宫阙旧影，南宋遗踪",
                "keywords": "德寿宫, 南宋, 宫殿遗址",
            },
        }

    def compare_search(self, query, top_k):
        if not query.strip():
            raise ValueError("query is required")
        return {
            "query": query,
            "query_text": "南宋德寿宫遗址博物馆",
            "top_k": top_k,
            "demo_case": {
                "query": "德寿宫",
                "query_text": "南宋德寿宫遗址博物馆",
                "focus": "南宋宫殿遗址",
                "image_id": 42,
                "filename": "sample.png",
            },
            "pool_size": 1117,
            "ground_truth": {"image_ids": [42], "hit_count": 1},
            "timings_ms": {"zero_shot": 12.5, "dora": 13.2},
            "results": {
                "zero_shot": [
                    {
                        "image_id": 99,
                        "filename": None,
                        "rank": 1,
                        "score": 0.82,
                        "title": "北京太庙",
                        "image_url": "/api/eval-images/99",
                        "rank_change_vs_other": -1,
                        "judgement": "非标准答案",
                        "judgement_reason": "该图片不在当前验证文本的标准答案集合中。",
                    }
                ],
                "dora": [
                    {
                        "image_id": 42,
                        "filename": "sample.png",
                        "rank": 1,
                        "score": 0.91,
                        "title": "南宋德寿宫遗址",
                        "image_url": "/api/eval-images/42",
                        "rank_change_vs_other": 1,
                        "judgement": "标准答案",
                        "judgement_reason": "该图片是当前验证文本的标准配对图像。",
                    }
                ],
            },
        }

    def compare_image_to_text(self, top_k, filename=None, image_id=None, image_bytes=None):
        if image_id is None and not filename and not image_bytes:
            raise ValueError("image query is required")
        return {
            "top_k": top_k,
            "pool_size": 4868,
            "query_source": {
                "type": "benchmark",
                "image_id": 42,
                "filename": "sample.png",
                "image_url": "/api/eval-images/42",
                "label": "南宋德寿宫遗址",
            },
            "ground_truth": {"candidate_ids": ["text:0"], "hit_count": 1},
            "timings_ms": {"zero_shot": 9.8, "dora": 10.4},
            "results": {
                "zero_shot": [
                    {
                        "candidate_id": "text:4",
                        "filename": None,
                        "rank": 1,
                        "score": 0.78,
                        "title": "其他文本",
                        "text": "北京皇家建筑",
                        "text_type_label": "验证文本",
                        "rank_change_vs_other": -1,
                        "judgement": "非标准答案",
                        "judgement_reason": "该文本不在当前验证图像的标准答案集合中。",
                    }
                ],
                "dora": [
                    {
                        "candidate_id": "text:0",
                        "filename": "sample.png",
                        "rank": 1,
                        "score": 0.88,
                        "title": "南宋德寿宫遗址",
                        "text": "南宋德寿宫遗址博物馆",
                        "text_type_label": "验证文本",
                        "rank_change_vs_other": 1,
                        "judgement": "标准答案",
                        "judgement_reason": "该文本是当前验证图像的标准配对文本。",
                    }
                ],
            },
        }

    def get_image_directory(self):
        return self.image_dir

    def get_benchmark_image_bytes(self, image_id):
        if image_id != 42:
            return None
        return b"benchmark-image"


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from web_demo.backend.app import create_app

        cls.fake_backend = FakeDemoBackend()
        cls.app = create_app(cls.fake_backend)

    def setUp(self):
        self.client = self.app.test_client()

    def test_summary_endpoint_returns_benchmark_sections(self):
        response = self.client.get("/api/summary")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["hero"]["title"], "NanS-CLIP")
        self.assertEqual(payload["benchmark_corpus"]["image_count"], 1117)
        self.assertEqual(payload["benchmark_corpus"]["text_count"], 4868)
        self.assertEqual(payload["models"][1]["label"], "DoRA")

    def test_summary_endpoint_returns_curated_demo_cases(self):
        response = self.client.get("/api/summary")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["demo_cases"][0]["query"], "德寿宫")
        self.assertEqual(payload["demo_cases"][0]["query_text"], "南宋德寿宫遗址博物馆")

    def test_examples_endpoint_returns_demo_narratives(self):
        response = self.client.get("/api/examples")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload[0]["query"], "德寿宫")

    def test_item_endpoint_returns_merged_sample_details(self):
        response = self.client.get("/api/item/sample.png")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["source"], "Wikimedia Commons")
        self.assertEqual(payload["representative_texts"]["keywords"], "德寿宫, 南宋, 宫殿遗址")

    def test_item_endpoint_returns_404_for_missing_item(self):
        response = self.client.get("/api/item/missing.png")

        self.assertEqual(response.status_code, 404)
        payload = response.get_json()
        self.assertEqual(payload["error"], "item_not_found")

    def test_compare_endpoint_rejects_blank_queries(self):
        response = self.client.post(
            "/api/search/compare",
            json={"query": "   ", "top_k": 5},
        )

        self.assertEqual(response.status_code, 400)
        payload = response.get_json()
        self.assertEqual(payload["error"], "invalid_query")

    def test_compare_endpoint_clamps_top_k_and_returns_benchmark_results(self):
        response = self.client.post(
            "/api/search/compare",
            json={"query": "德寿宫", "top_k": 100},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["top_k"], 20)
        self.assertEqual(payload["results"]["dora"][0]["rank_change_vs_other"], 1)
        self.assertEqual(payload["results"]["zero_shot"][0]["image_url"], "/api/eval-images/99")
        self.assertEqual(payload["ground_truth"]["image_ids"], [42])

    def test_compare_endpoint_returns_standard_answer_verdicts(self):
        response = self.client.post(
            "/api/search/compare",
            json={"query": "德寿宫", "top_k": 5},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["results"]["dora"][0]["judgement"], "标准答案")
        self.assertIn("标准配对图像", payload["results"]["dora"][0]["judgement_reason"])

    def test_image_to_text_endpoint_returns_ranked_text_results(self):
        response = self.client.post(
            "/api/search/image-to-text",
            json={"image_id": 42, "top_k": 6},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["top_k"], 6)
        self.assertEqual(payload["query_source"]["image_id"], 42)
        self.assertEqual(payload["results"]["dora"][0]["judgement"], "标准答案")

    def test_image_to_text_endpoint_rejects_missing_image_query(self):
        response = self.client.post(
            "/api/search/image-to-text",
            json={"top_k": 5},
        )

        self.assertEqual(response.status_code, 400)
        payload = response.get_json()
        self.assertEqual(payload["error"], "invalid_image_query")

    def test_benchmark_image_route_serves_eval_assets(self):
        response = self.client.get("/api/eval-images/42")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data, b"benchmark-image")
        response.close()


if __name__ == "__main__":
    unittest.main()
