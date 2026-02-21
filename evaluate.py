# -*- coding: utf-8 -*-
"""
评估 Chinese-CLIP 的检索性能：Zero-Shot vs LoRA 微调 对比。
计算 Text→Image 和 Image→Text 的 Recall@K。

使用方法:
    conda activate pytorch
    cd d:/Desktop/CV/CLIP/NanS-CLIP

    # 评估 Zero-Shot 基线
    python evaluate.py --mode zeroshot

    # 评估 LoRA 微调后
    python evaluate.py --mode lora --lora_path ../clip_data/experiments/lora_song/best_lora.pt

    # Hard Negative 评测（混入干扰图片）
    python evaluate.py --distractor_dir data/distractors
"""

import os
import json
import argparse
import base64
from io import BytesIO
from pathlib import Path

import torch
import lmdb
import pickle
from tqdm import tqdm
from PIL import Image

from cn_clip.clip import load_from_name, tokenize
from cn_clip.clip.lora import inject_lora, load_lora_state_dict


def load_eval_data(lmdb_dir: str, preprocess, max_txt_length: int = 52):
    """从 LMDB 加载评估数据（不重复的图片和文本）"""
    # 读取所有 pairs
    env_pairs = lmdb.open(os.path.join(lmdb_dir, "pairs"), readonly=True, lock=False)
    env_imgs = lmdb.open(os.path.join(lmdb_dir, "imgs"), readonly=True, lock=False)

    with env_pairs.begin() as txn:
        num = int(txn.get(b"num_samples").decode("utf-8"))

    # 收集所有图文对
    pairs = []
    with env_pairs.begin() as txn:
        for i in range(num):
            data = pickle.loads(txn.get(str(i).encode("utf-8")))
            image_id, text_id, text = data
            pairs.append((image_id, text))

    # 提取唯一图片
    unique_image_ids = sorted(set(p[0] for p in pairs))
    images = []
    with env_imgs.begin() as txn:
        for img_id in unique_image_ids:
            b64 = txn.get(str(img_id).encode("utf-8"))
            if b64:
                img_bytes = base64.b64decode(b64.decode("utf-8"))
                img = Image.open(BytesIO(img_bytes)).convert("RGB")
                images.append((img_id, preprocess(img)))

    # 构建 image_id → 数组位置 的映射
    imgid_to_pos = {iid: pos for pos, (iid, _) in enumerate(images)}

    # 提取唯一文本
    unique_texts = list(set(p[1] for p in pairs))

    # 构建 ground truth 映射（用数组位置，不用 image_id）
    # text_idx -> set of 图片数组位置
    text_to_images = {}
    # img_pos -> set of 文本数组位置
    image_to_texts = {}
    for img_id, text in pairs:
        text_idx = unique_texts.index(text)
        img_pos = imgid_to_pos.get(img_id)
        if img_pos is None:
            continue

        if text_idx not in text_to_images:
            text_to_images[text_idx] = set()
        text_to_images[text_idx].add(img_pos)

        if img_pos not in image_to_texts:
            image_to_texts[img_pos] = set()
        image_to_texts[img_pos].add(text_idx)

    env_pairs.close()
    env_imgs.close()

    return images, unique_texts, text_to_images, image_to_texts


def load_distractors(distractor_dir: str, preprocess, start_id: int = 100000):
    """加载干扰图片（Hard Negative）。
    
    干扰图片使用与训练数据不重叠的 image_id（从 start_id 开始），
    仅混入图片检索池，不影响 ground truth 映射。
    """
    distractor_dir = Path(distractor_dir)
    if not distractor_dir.exists():
        print(f"   ⚠️ 干扰图片目录不存在: {distractor_dir}")
        return []

    exts = {".jpg", ".jpeg", ".png", ".webp"}
    distractors = []
    for i, p in enumerate(sorted(distractor_dir.iterdir())):
        if p.suffix.lower() not in exts or not p.is_file():
            continue
        try:
            img = Image.open(p).convert("RGB")
            distractors.append((start_id + i, preprocess(img)))
        except Exception:
            continue
    return distractors


def compute_features(model, images, texts, device, batch_size=32):
    """提取所有图片和文本的特征向量"""
    model.eval()

    # 图片特征
    all_img_feats = []
    for i in range(0, len(images), batch_size):
        batch = torch.stack([img for _, img in images[i:i+batch_size]]).to(device)
        with torch.no_grad():
            with torch.amp.autocast("cuda"):
                feats = model.encode_image(batch)
                feats = feats / feats.norm(dim=-1, keepdim=True)
        all_img_feats.append(feats.cpu())
    image_features = torch.cat(all_img_feats, dim=0)

    # 文本特征
    all_txt_feats = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        tokens = tokenize(batch_texts, context_length=52).to(device)
        with torch.no_grad():
            with torch.amp.autocast("cuda"):
                feats = model.encode_text(tokens)
                feats = feats / feats.norm(dim=-1, keepdim=True)
        all_txt_feats.append(feats.cpu())
    text_features = torch.cat(all_txt_feats, dim=0)

    return image_features, text_features


def recall_at_k(sim_matrix, ground_truth, k_list=[1, 5, 10]):
    """计算 Recall@K 和 Mean Recall"""
    results = {}
    for k in k_list:
        correct = 0
        total = 0
        for i in range(sim_matrix.shape[0]):
            if i not in ground_truth:
                continue
            topk = sim_matrix[i].topk(k).indices.tolist()
            gt = ground_truth[i]
            if any(t in gt for t in topk):
                correct += 1
            total += 1
        results[f"R@{k}"] = correct / max(total, 1) * 100
    results["MR"] = sum(results.values()) / len(results)
    return results


def main():
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    parser = argparse.ArgumentParser(description="Chinese-CLIP 检索评估")
    parser.add_argument("--mode", choices=["zeroshot", "lora", "both"], default="both")
    parser.add_argument("--lora_path", type=str, default="../clip_data/experiments/lora_song/best_lora.pt")
    parser.add_argument("--data_dir", type=str, default="../clip_data/datasets/SongDynasty/lmdb/valid")
    parser.add_argument("--pretrained", type=str, default="../clip_data/pretrained_weights/")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--distractor_dir", type=str, default="",
                        help="干扰图片目录（Hard Negative 评测），如 data/distractors")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载数据（只加载一次）
    print("📦 加载模型...")
    model, preprocess = load_from_name("ViT-B-16", device="cpu", download_root=args.pretrained)
    model.float()

    print("📂 加载评估数据...")
    images, texts, text_to_images, image_to_texts = load_eval_data(args.data_dir, preprocess)
    num_domain_images = len(images)
    print(f"   领域图片: {num_domain_images} | 文本: {len(texts)}")

    # 加载干扰图片（Hard Negative）
    if args.distractor_dir:
        distractors = load_distractors(args.distractor_dir, preprocess)
        if distractors:
            images = images + distractors  # 追加到末尾，不影响 ground truth 位置
            print(f"   🎯 干扰图片: {len(distractors)} | 总检索池: {len(images)}")

    modes = [args.mode] if args.mode != "both" else ["zeroshot", "lora"]
    all_results = {}

    for mode in modes:
        print(f"\n{'='*50}")
        print(f"  评估模式: {mode}")
        print(f"{'='*50}")

        if mode == "lora":
            # 重新加载干净模型再注入 LoRA
            model, preprocess = load_from_name("ViT-B-16", device="cpu", download_root=args.pretrained)
            model.float()
            inject_lora(model, rank=args.rank, alpha=args.alpha)
            if os.path.isfile(args.lora_path):
                state_dict = torch.load(args.lora_path, map_location="cpu", weights_only=False)
                load_lora_state_dict(model, state_dict)
                print(f"   LoRA 权重已加载: {args.lora_path}")
            else:
                print(f"   ⚠️ LoRA 权重不存在: {args.lora_path}，使用随机初始化")

        model = model.to(device)
        model.eval()

        print("🔧 提取特征...")
        image_features, text_features = compute_features(model, images, texts, device)

        sim_t2i = text_features @ image_features.T
        sim_i2t = image_features @ text_features.T

        t2i_recall = recall_at_k(sim_t2i, text_to_images)
        i2t_recall = recall_at_k(sim_i2t, image_to_texts)

        print(f"\n  Text → Image:")
        for k, v in t2i_recall.items():
            print(f"    {k}: {v:.1f}%")
        print(f"\n  Image → Text:")
        for k, v in i2t_recall.items():
            print(f"    {k}: {v:.1f}%")

        all_results[mode] = {
            "text_to_image": t2i_recall,
            "image_to_text": i2t_recall,
        }

        result = {"mode": mode, **all_results[mode],
                  "num_domain_images": num_domain_images,
                  "num_distractors": len(images) - num_domain_images,
                  "num_total_images": len(images),
                  "num_texts": len(texts)}
        with open(Path(f"eval_results_{mode}.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    # 如果跑了 both 模式，打印对比表格
    if len(all_results) == 2:
        pool_desc = f"（检索池: {len(images)} 张，其中干扰 {len(images) - num_domain_images}）" if len(images) > num_domain_images else ""
        print(f"\n{'='*60}")
        print(f"  📊 Zero-Shot vs LoRA 对比 {pool_desc}")
        print(f"{'='*60}")
        print(f"  {'指标':<16} {'Zero-Shot':>10} {'LoRA':>10} {'提升':>10}")
        print(f"  {'-'*50}")
        zs = all_results["zeroshot"]
        lo = all_results["lora"]
        for direction, label in [("text_to_image", "T→I"), ("image_to_text", "I→T")]:
            for k in ["R@1", "R@5", "R@10", "MR"]:
                z = zs[direction][k]
                l = lo[direction][k]
                delta = l - z
                sign = "+" if delta >= 0 else ""
                print(f"  {label} {k:<11} {z:>9.1f}% {l:>9.1f}% {sign}{delta:>8.1f}%")
        print(f"  {'='*50}")

    print(f"\n💾 结果已保存")


if __name__ == "__main__":
    main()
