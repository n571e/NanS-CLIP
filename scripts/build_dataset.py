# -*- coding: utf-8 -*-
"""
将 annotations.json 转换为 Chinese-CLIP 训练所需的格式：
  - {split}_imgs.tsv    (image_id \t base64)
  - {split}_texts.jsonl ({"text_id", "text", "image_ids"})
然后调用 build_lmdb_dataset.py 生成 LMDB。

使用方法:
    conda activate pytorch
    cd d:/Desktop/CV/CLIP/NanS-CLIP
    python scripts/build_dataset.py
"""

import json
import base64
import random
import argparse
from pathlib import Path
from io import BytesIO
from PIL import Image

# ============ 配置 ============
ANNOTATION_FILE = Path("data/annotations.json")
IMAGE_DIR = Path("data/images")
OUTPUT_DIR = Path("../clip_data/datasets/SongDynasty")  # Chinese-CLIP 约定
TRAIN_RATIO = 0.8


def image_to_base64(image_path: Path, max_size: int = 512) -> str:
    """将图片压缩并转为 base64（节省 LMDB 空间）"""
    img = Image.open(image_path).convert("RGB")

    # 等比缩放到 max_size
    w, h = img.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def build_texts_for_image(ann: dict, image_id: int) -> list:
    """
    从一条标注生成多条 text 记录。
    每条 text 都关联到同一个 image_id。
    """
    texts = []

    # 1. 现代中文描述
    modern = ann.get("modern_chinese", "").strip()
    if modern:
        texts.append(modern)

    # 2. 古文风格描述
    ancient = ann.get("ancient_style", "").strip()
    if ancient:
        texts.append(ancient)

    # 3. 关键词组合为一句话（更适合 CLIP 的短文本编码）
    keywords = ann.get("keywords", "").strip()
    if keywords:
        # "南宋, 山水画, 西湖" → "南宋 山水画 西湖"
        kw_text = keywords.replace(",", " ").replace("，", " ").strip()
        texts.append(kw_text)

    # 4. 标题本身也是一条文本
    title = ann.get("title", "").strip()
    if title and title not in texts:
        texts.append(title)

    return texts


def main():
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    parser = argparse.ArgumentParser(description="构建 Chinese-CLIP 数据集")
    parser.add_argument("--annotation", type=str, default=str(ANNOTATION_FILE))
    parser.add_argument("--image_dir", type=str, default=str(IMAGE_DIR))
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--train_ratio", type=float, default=TRAIN_RATIO)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    annotation_file = Path(args.annotation)
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)

    # 加载标注
    with open(annotation_file, "r", encoding="utf-8") as f:
        annotations = json.load(f)
    print(f"📋 加载标注: {len(annotations)} 条")

    # 过滤掉没有图片文件的记录
    valid = []
    for ann in annotations:
        img_path = image_dir / ann["filename"]
        if img_path.exists():
            valid.append(ann)
        else:
            print(f"  ⚠️ 图片不存在，跳过: {ann['filename']}")
    print(f"   有效记录: {len(valid)} 条")

    if len(valid) < 5:
        print("❌ 图片数量太少（<5），无法构建有效数据集")
        return

    # 打乱并划分 train/valid
    random.seed(args.seed)
    random.shuffle(valid)
    split_idx = int(len(valid) * args.train_ratio)
    splits = {
        "train": valid[:split_idx],
        "valid": valid[split_idx:],
    }
    print(f"   训练集: {len(splits['train'])} | 验证集: {len(splits['valid'])}")

    # 生成数据文件
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_data in splits.items():
        tsv_path = output_dir / f"{split_name}_imgs.tsv"
        jsonl_path = output_dir / f"{split_name}_texts.jsonl"

        text_id_counter = 0

        with open(tsv_path, "w", encoding="utf-8") as f_tsv, \
             open(jsonl_path, "w", encoding="utf-8") as f_jsonl:

            for image_id, ann in enumerate(split_data):
                img_path = image_dir / ann["filename"]

                # 写 TSV（图片 base64）
                try:
                    b64 = image_to_base64(img_path)
                    f_tsv.write(f"{image_id}\t{b64}\n")
                except Exception as e:
                    print(f"  ⚠️ 图片处理失败: {ann['filename']}: {e}")
                    continue

                # 写 JSONL（多条文本，都关联到同一个 image_id）
                texts = build_texts_for_image(ann, image_id)
                for text in texts:
                    entry = {
                        "text_id": text_id_counter,
                        "text": text,
                        "image_ids": [image_id]
                    }
                    f_jsonl.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    text_id_counter += 1

        print(f"   ✅ {split_name}: {len(split_data)} 图 | {text_id_counter} 文本对")
        print(f"      {tsv_path}")
        print(f"      {jsonl_path}")

    print(f"\n📁 数据文件输出: {output_dir.resolve()}")
    print(f"\n下一步：运行 LMDB 转换:")
    print(f"  python cn_clip/preprocess/build_lmdb_dataset.py \\")
    print(f"    --data_dir {output_dir} --splits train,valid")


if __name__ == "__main__":
    main()
