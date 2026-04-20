import json
import tempfile
import unittest
from pathlib import Path


class MetadataIndexTests(unittest.TestCase):
    def test_build_metadata_index_merges_texts_and_source_records(self):
        from web_demo.backend.metadata import build_metadata_index

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            augmented_path = tmp / "annotations_augmented.json"
            metadata_path = tmp / "image_metadata.jsonl"

            augmented_path.write_text(
                json.dumps(
                    [
                        {
                            "filename": "sample.png",
                            "title": "西湖旧景",
                            "modern_chinese": "西湖风景图",
                            "ancient_style": "湖山清绝",
                            "keywords": "西湖, 山水",
                        },
                        {
                            "filename": "sample.png",
                            "title": "西湖旧景",
                            "modern_chinese": "另一条白话文",
                            "ancient_style": "烟水空濛",
                            "keywords": "西湖, 亭台",
                            "augmented": True,
                        },
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            metadata_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "filename": "sample.png",
                                "source": "Wikimedia Commons",
                                "original_url": "https://example.com/sample.png",
                                "description": "公开馆藏图像",
                            },
                            ensure_ascii=False,
                        )
                    ]
                ),
                encoding="utf-8",
            )

            index = build_metadata_index(augmented_path, metadata_path)

        item = index["sample.png"]
        self.assertEqual(item["source"], "Wikimedia Commons")
        self.assertEqual(item["original_url"], "https://example.com/sample.png")
        self.assertEqual(item["representative_texts"]["modern_chinese"], "西湖风景图")
        self.assertEqual(len(item["texts"]["modern_chinese"]), 2)
        self.assertEqual(item["texts"]["keywords"][1], "西湖, 亭台")


if __name__ == "__main__":
    unittest.main()
