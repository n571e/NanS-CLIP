# -*- coding: utf-8 -*-
"""
VLM 标注质量过滤器。

用 Chinese-CLIP 反向计算每条标注文本与对应图片之间的余弦相似度,
剔除相似度低于阈值的低质量标注，确保训练数据质量。

用法:
    python scripts/filter_annotations.py
    python scripts/filter_annotations.py --threshold 0.15 --dry-run
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cn_clip.clip as clip
from cn_clip.clip import load_from_name

ANNOTATION_FILE = PROJECT_ROOT / "data" / "annotations.json"
IMAGE_DIR = PROJECT_ROOT / "data" / "images"


def compute_similarity(model, preprocess, tokenizer_fn, image_path, text, device):
    """计算单张图片与一条文本之间的余弦相似度"""
    # 图像编码
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    # 文本编码
    text_tokens = tokenizer_fn([text]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_tokens)

        # L2 归一化
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # 余弦相似度
        similarity = (image_features @ text_features.T).item()

    return similarity


def main():
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

    parser = argparse.ArgumentParser(description="用 CLIP 过滤低质量 VLM 标注")
    parser.add_argument("--annotation", type=str, default=str(ANNOTATION_FILE))
    parser.add_argument("--threshold", type=float, default=0.15,
                        help="余弦相似度阈值，低于此值的标注将被标记为低质量 (默认 0.15)")
    parser.add_argument("--pretrained", type=str, default="../clip_data/pretrained_weights/")
    parser.add_argument("--dry-run", action="store_true",
                        help="只报告不实际删除，用于查看阈值效果")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 设备: {device}")

    # 加载 CLIP 模型（Zero-Shot，不加载 LoRA，用原始模型评估）
    print("📦 加载 Chinese-CLIP ViT-B-16 (Zero-Shot) 用于质量评估...")
    model, preprocess = load_from_name("ViT-B-16", device=device,
                                        download_root=args.pretrained)
    model.eval()

    # 加载标注
    annotation_file = Path(args.annotation)
    with open(annotation_file, "r", encoding="utf-8") as f:
        annotations = json.load(f)
    print(f"📋 加载 {len(annotations)} 条标注\n")

    kept = []
    removed = []

    for i, ann in enumerate(annotations):
        filename = ann.get("filename", "")
        img_path = IMAGE_DIR / filename

        if not img_path.exists():
            print(f"[{i+1}/{len(annotations)}] ⚠️ 图片不存在: {filename}，跳过")
            removed.append(ann)
            continue

        # 对 modern_chinese 计算相似度（它是最核心的描述文本）
        text = ann.get("modern_chinese", "")
        if not text:
            print(f"[{i+1}/{len(annotations)}] ⚠️ 无文本: {filename}，移除")
            removed.append(ann)
            continue

        sim = compute_similarity(model, preprocess, clip.tokenize, img_path, text, device)

        if sim < args.threshold:
            print(f"[{i+1}/{len(annotations)}] ❌ sim={sim:.4f} < {args.threshold} | {filename}")
            print(f"    文本: {text[:60]}...")
            removed.append(ann)
        else:
            if (i + 1) % 20 == 0 or i == 0:
                print(f"[{i+1}/{len(annotations)}] ✅ sim={sim:.4f} | {filename}")
            kept.append(ann)

    # 汇总
    print(f"\n{'='*50}")
    print(f"  总计: {len(annotations)} 条")
    print(f"  保留: {len(kept)} 条")
    print(f"  剔除: {len(removed)} 条 (相似度 < {args.threshold})")
    print(f"{'='*50}")

    if removed:
        print(f"\n被剔除的标注:")
        for r in removed:
            print(f"  - {r.get('filename', '?')}: {r.get('modern_chinese', '')[:50]}")

    if not args.dry_run and removed:
        # 备份原文件
        backup = annotation_file.with_suffix(".json.bak")
        annotation_file.rename(backup)
        print(f"\n💾 原文件已备份: {backup}")

        # 写入过滤后的文件
        with open(annotation_file, "w", encoding="utf-8") as f:
            json.dump(kept, f, ensure_ascii=False, indent=2)
        print(f"✅ 过滤后标注已保存: {annotation_file} ({len(kept)} 条)")
    elif args.dry_run:
        print(f"\n🔍 [Dry Run] 未实际修改文件。去掉 --dry-run 以执行过滤。")


if __name__ == "__main__":
    main()
