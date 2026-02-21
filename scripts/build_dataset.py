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


def build_texts_for_image(ann: dict) -> list:
    """
    从一条标注生成多条 text 记录。
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

    # 打乱并按 **图片** 划分 train/valid 避免泄露
    # 同一张图片的所有变体描述只会分到同一边
    unique_filenames = list(set([ann["filename"] for ann in valid]))
    random.seed(args.seed)
    random.shuffle(unique_filenames)
    
    split_img_idx = int(len(unique_filenames) * args.train_ratio)
    train_filenames = set(unique_filenames[:split_img_idx])
    
    splits = {
        "train": [ann for ann in valid if ann["filename"] in train_filenames],
        "valid": [ann for ann in valid if ann["filename"] not in train_filenames],
    }
    
    print(f"   按图片切分: 训练集包含 {len(train_filenames)} 张原图 | 验证集包含 {len(unique_filenames) - len(train_filenames)} 张原图")
    print(f"   最终记录数: 训练集 {len(splits['train'])} 条 | 验证集 {len(splits['valid'])} 条")

    # 生成数据文件
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_data in splits.items():
        tsv_path = output_dir / f"{split_name}_imgs.tsv"
        jsonl_path = output_dir / f"{split_name}_texts.jsonl"

        text_id_counter = 0

        # 由于可能有多条标注对应同一张图（虽然上面的处理保证了同一图都在同一个 split），
        # 我们需要在生成 TSV 时按去重的图片写入以防报错。
        # 建立去重映射
        split_unique_ann = []
        seen = set()
        for ann in split_data:
            if ann["filename"] not in seen:
                split_unique_ann.append(ann)
                seen.add(ann["filename"])

        with open(tsv_path, "w", encoding="utf-8") as f_tsv, \
             open(jsonl_path, "w", encoding="utf-8") as f_jsonl:

            # 遍历去重后的图片
            for image_id, ann_master in enumerate(split_unique_ann):
                img_path = image_dir / ann_master["filename"]

                # 写 TSV（图片 base64）
                try:
                    b64 = image_to_base64(img_path)
                    f_tsv.write(f"{image_id}\t{b64}\n")
                except Exception as e:
                    print(f"  ⚠️ 图片处理失败: {ann_master['filename']}: {e}")
                    continue

                # 收集所有此图片的标注
                ann_list = [a for a in split_data if a["filename"] == ann_master["filename"]]
                
                # 写 JSONL
                for ann in ann_list:
                    texts = build_texts_for_image(ann)
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
