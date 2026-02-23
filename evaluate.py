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
import base64
import argparse
from pathlib import Path
from io import BytesIO
import pickle
import math
import logging

import lmdb
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image

# ============ 日志配置 ============
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

from cn_clip.clip import load_from_name, tokenize
from cn_clip.clip.lora import inject_lora, load_lora_state_dict


def load_eval_data(lmdb_dir: str, preprocess):
    """从 LMDB 加载所有待评测数据和 Ground Truth"""
    lmdb_dir = Path(lmdb_dir)
    env_imgs = lmdb.open(str(lmdb_dir / "imgs"), readonly=True, lock=False)
    env_pairs = lmdb.open(str(lmdb_dir / "pairs"), readonly=True, lock=False)

    # 收集所有图文对
    pairs = []
    with env_pairs.begin() as txn:
        num_pairs = txn.stat()["entries"]
        for i in range(num_pairs):
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
    text_to_idx = {text: idx for idx, text in enumerate(unique_texts)}  # O(1) 查表优化

    # 构建 ground truth 映射（用数组位置，不用 image_id）
    text_to_images = {}
    image_to_texts = {}
    for img_id, text in pairs:
        text_idx = text_to_idx[text]
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
        logger.warning(f"干扰图片目录不存在: {distractor_dir}")
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


def metrics_at_k(sim_matrix, ground_truth, k_list=[1, 5, 10]):
    """计算 Recall@K, Mean Recall, mAP, NDCG@K"""
    results = {}
    
    # 存储指标的累加器
    recalls = {k: 0 for k in k_list}
    ndcgs = {k: 0 for k in k_list}
    map_sum = 0
    total = 0
    
    for i in range(sim_matrix.shape[0]):
        if i not in ground_truth:
            continue
            
        gt = ground_truth[i]
        num_gt = len(gt)
        if num_gt == 0: continue
            
        # 对该样本获取所有预测值的排序
        pred_indices = sim_matrix[i].argsort(descending=True).tolist()
        
        # 计算 Recall@K 和 NDCG@K
        for k in k_list:
            topk = pred_indices[:k]
            hits = [1 if idx in gt else 0 for idx in topk]
            
            # Recall@K: 是否命中
            if sum(hits) > 0:
                recalls[k] += 1
                
            # NDCG@K: 标准折扣累积收益
            dcg = sum([rel / math.log2(rank + 2) for rank, rel in enumerate(hits)])
            idcg = sum([1 / math.log2(rank + 2) for rank in range(min(num_gt, k))])
            ndcgs[k] += dcg / idcg if idcg > 0 else 0
            
        # 计算 Average Precision (AP) 用于 mAP
        ap = 0
        hits_so_far = 0
        for rank, idx in enumerate(pred_indices):
            if idx in gt:
                hits_so_far += 1
                ap += hits_so_far / (rank + 1)
        map_sum += ap / num_gt
        
        total += 1
        
    for k in k_list:
        results[f"R@{k}"] = recalls[k] / max(total, 1) * 100
        results[f"NDCG@{k}"] = ndcgs[k] / max(total, 1) * 100
    results["mAP"] = map_sum / max(total, 1) * 100
    results["MR"] = sum([results[f"R@{k}"] for k in k_list]) / len(k_list)
    
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
    parser.add_argument("--text_only", action="store_true", default=False, help="是否仅微调文本编码器")
    parser.add_argument("--distractor_dir", type=str, default="",
                        help="干扰图片目录（Hard Negative 评测），如 data/distractors")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载数据（只加载一次）
    logger.info("📦 加载模型...")
    model, preprocess = load_from_name("ViT-B-16", device="cpu", download_root=args.pretrained)
    model.float()

    logger.info("📂 加载评估数据...")
    images, texts, text_to_images, image_to_texts = load_eval_data(args.data_dir, preprocess)
    num_domain_images = len(images)
    logger.info(f"领域图片: {num_domain_images} | 文本: {len(texts)}")

    # 加载干扰图片（Hard Negative）
    if args.distractor_dir:
        distractors = load_distractors(args.distractor_dir, preprocess)
        if distractors:
            images = images + distractors  # 追加到末尾，不影响 ground truth 位置
            logger.info(f"🎯 干扰图片: {len(distractors)} | 总检索池: {len(images)}")

    modes = [args.mode] if args.mode != "both" else ["zeroshot", "lora"]
    all_results = {}

    for mode in modes:
        logger.info(f"{'='*50}")
        logger.info(f"评估模式: {mode}")
        logger.info(f"{'='*50}")

        if mode == "lora":
            # 重新加载干净模型再注入 LoRA
            model, preprocess = load_from_name("ViT-B-16", device="cpu", download_root=args.pretrained)
            model.float()
            inject_lora(model, rank=args.rank, alpha=args.alpha, text_only=args.text_only)
            if os.path.isfile(args.lora_path):
                state_dict = torch.load(args.lora_path, map_location="cpu", weights_only=False)
                load_lora_state_dict(model, state_dict)
                logger.info(f"LoRA 权重已加载: {args.lora_path}")
            else:
                logger.warning(f"LoRA 权重不存在: {args.lora_path}，使用随机初始化")

        model = model.to(device)
        model.eval()

        logger.info("🔧 提取特征...")
        image_features, text_features = compute_features(model, images, texts, device)

        sim_t2i = text_features @ image_features.T
        sim_i2t = image_features @ text_features.T

        t2i_metrics = metrics_at_k(sim_t2i, text_to_images)
        i2t_metrics = metrics_at_k(sim_i2t, image_to_texts)

        logger.info(f"\n  Text → Image:")
        for k, v in t2i_metrics.items():
            logger.info(f"    {k}: {v:.1f}%")
        logger.info(f"\n  Image → Text:")
        for k, v in i2t_metrics.items():
            logger.info(f"    {k}: {v:.1f}%")

        all_results[mode] = {
            "text_to_image": t2i_metrics,
            "image_to_text": i2t_metrics,
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
        print(f"\n{'='*75}")
        print(f"  📊 Zero-Shot vs LoRA 对比 {pool_desc}")
        print(f"{'='*75}")
        print(f"  {'指标':<16} {'Zero-Shot':>10} {'LoRA':>10} {'提升':>10}")
        print(f"  {'-'*70}")
        zs = all_results["zeroshot"]
        lo = all_results["lora"]
        for direction, label in [("text_to_image", "T→I"), ("image_to_text", "I→T")]:
            for k in ["R@1", "R@5", "R@10", "MR", "mAP", "NDCG@5"]:
                z = zs[direction].get(k, 0)
                l = lo[direction].get(k, 0)
                delta = l - z
                sign = "+" if delta >= 0 else ""
                print(f"  {label} {k:<11} {z:>9.1f}% {l:>9.1f}% {sign}{delta:>8.1f}%")
        print(f"  {'='*75}")

    logger.info("💾 结果已保存")


if __name__ == "__main__":
    main()
