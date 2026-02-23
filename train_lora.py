# -*- coding: utf-8 -*-
"""
Chinese-CLIP + LoRA 单卡训练脚本（不依赖 DDP 分布式）。

使用方法:
    conda activate pytorch
    cd d:/Desktop/CV/CLIP/NanS-CLIP

    # 确保已完成数据集构建（LMDB 格式）
    # python scripts/build_dataset.py
    # python cn_clip/preprocess/build_lmdb_dataset.py --data_dir ../clip_data/datasets/SongDynasty --splits train,valid

    # 开始训练
    python train_lora.py

    # 自定义参数
    python train_lora.py --epochs 50 --lr 2e-4 --rank 8 --batch_size 4
"""

import math
import os
import sys
import json
import argparse
import time
import logging
from pathlib import Path

# ============ 日志配置 ============
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import lmdb
import pickle

from cn_clip.clip import load_from_name, tokenize
from cn_clip.clip.lora import inject_lora, get_lora_state_dict


# ============ 数据集类 ============
class LMDBDataset(Dataset):
    """从 LMDB 读取图文对（兼容 Chinese-CLIP 的 LMDB 格式）"""

    def __init__(self, lmdb_dir: str, preprocess, max_txt_length: int = 52):
        self.lmdb_dir = lmdb_dir
        self.preprocess = preprocess
        self.max_txt_length = max_txt_length

        # 打开 pairs LMDB
        pairs_path = os.path.join(lmdb_dir, "pairs")
        self.env_pairs = lmdb.open(pairs_path, readonly=True, lock=False)
        with self.env_pairs.begin() as txn:
            self.num_samples = int(txn.get(b"num_samples").decode("utf-8"))

        # 打开 imgs LMDB
        imgs_path = os.path.join(lmdb_dir, "imgs")
        self.env_imgs = lmdb.open(imgs_path, readonly=True, lock=False)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        import base64
        from io import BytesIO
        from PIL import Image

        # 读取 (image_id, text_id, text) 对
        with self.env_pairs.begin() as txn:
            data = pickle.loads(txn.get(str(idx).encode("utf-8")))
        image_id, text_id, text = data

        # 读取图片 base64
        with self.env_imgs.begin() as txn:
            b64 = txn.get(str(image_id).encode("utf-8")).decode("utf-8")

        # 解码图片
        img_bytes = base64.b64decode(b64)
        image = Image.open(BytesIO(img_bytes)).convert("RGB")
        image = self.preprocess(image)

        # Tokenize 文本
        text_tokens = tokenize([text], context_length=self.max_txt_length)[0]

        return image, text_tokens


def contrastive_loss(image_features, text_features, logit_scale, label_smoothing=0.05):
    """InfoNCE 对比损失，增加 Label Smoothing 缓解小数据集过拟合"""
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)

    logits = logit_scale * image_features @ text_features.T
    batch_size = logits.shape[0]
    device = logits.device

    # 使用 F.cross_entropy 的 label_smoothing 参数 (要求 PyTorch >= 1.10)
    labels = torch.arange(batch_size, device=device)
    loss_i2t = F.cross_entropy(logits, labels, label_smoothing=label_smoothing)
    loss_t2i = F.cross_entropy(logits.T, labels, label_smoothing=label_smoothing)

    return (loss_i2t + loss_t2i) / 2


def main():
    parser = argparse.ArgumentParser(description="Chinese-CLIP + LoRA 训练")
    parser.add_argument("--data_dir", type=str, default="../clip_data/datasets/SongDynasty/lmdb/train")
    parser.add_argument("--val_dir", type=str, default="../clip_data/datasets/SongDynasty/lmdb/valid")
    parser.add_argument("--pretrained", type=str, default="../clip_data/pretrained_weights/")
    parser.add_argument("--output_dir", type=str, default="../clip_data/experiments/lora_song")
    parser.add_argument("--rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup 步数占总步数比例")
    parser.add_argument("--alpha", type=float, default=16.0, help="LoRA alpha")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--accum_freq", type=int, default=4, help="梯度累积步数，等效 batch = batch_size * accum_freq")
    parser.add_argument("--lr", type=float, default=5e-5)   
    parser.add_argument("--wd", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--text_only", action="store_true", default=False, help="是否仅微调文本编码器（防止灾难性遗忘）")
    parser.add_argument("--save_every", type=int, default=5, help="每 N 个 epoch 保存一次")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"🔧 设备: {device}")

    # 1. 加载模型
    logger.info("📦 加载 Chinese-CLIP ViT-B-16...")
    model, preprocess = load_from_name("ViT-B-16", device=device, download_root=args.pretrained)
    
    # 强制将模型从半精度（FP16）转换为单精度（FP32），兼容 PyTorch 原生 AMP
    model.float()

    # 2. 注入 LoRA
    logger.info(f"🔗 注入 LoRA (rank={args.rank}, alpha={args.alpha}, text_only={args.text_only})...")
    lora_params = inject_lora(model, rank=args.rank, alpha=args.alpha, text_only=args.text_only)

    # 3. 冻结非 LoRA 参数（并确保 logit_scale 不被更新避免极度自信过拟合）
    for name, param in model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False
    
    if hasattr(model, "logit_scale"):
        model.logit_scale.requires_grad = False

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"📊 总参数: {total:,} | 可训练(LoRA): {trainable:,} | 占比: {trainable/total*100:.2f}%")

    # 4. 数据集
    logger.info("📂 加载数据集...")
    train_dataset = LMDBDataset(args.data_dir, preprocess)
    logger.info(f"训练集: {len(train_dataset)} 对")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Windows 下建议设为 0
        drop_last=True,
        pin_memory=True,
    )

    val_dataset = None
    if os.path.isdir(args.val_dir):
        val_dataset = LMDBDataset(args.val_dir, preprocess)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        logger.info(f"验证集: {len(val_dataset)} 对")

    # 5. 优化器
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.wd,
    )

    # 6. 学习率调度：Linear Warmup + Cosine Annealing
    total_steps = args.epochs * (len(train_dataset) // args.batch_size + 1)
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    logger.info(f"📈 学习率调度: Warmup {warmup_steps} steps → Cosine decay, 总 {total_steps} steps")

    # 7. 混合精度
    scaler = torch.amp.GradScaler("cuda") if args.fp16 and device == "cuda" else None

    # 8. 输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 9. 训练循环
    # 关键：对比学习的负样本数量 = batch_size - 1
    # 必须在累积后的大 batch 上计算 contrastive loss，而不是每个 mini-batch 独立计算
    effective_batch = args.batch_size * args.accum_freq
    logger.info(f"🚀 开始训练（{args.epochs} epochs, 对比学习 batch = {effective_batch}）")
    best_val_loss = float("inf")

    # 初始化训练日志
    log_path = output_dir / "training_log.csv"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,val_loss,lr,is_best\n")
    logger.info(f"📝 训练日志: {log_path}")

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        num_updates = 0
        optimizer.zero_grad()

        # 累积特征的缓冲区
        accum_img_feats = []
        accum_txt_feats = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for step, (images, texts) in enumerate(pbar):
            images = images.to(device)
            texts = texts.to(device)

            # Forward: 提取特征并保留计算图
            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    img_f = model.encode_image(images)
                    txt_f = model.encode_text(texts)
            else:
                img_f = model.encode_image(images)
                txt_f = model.encode_text(texts)

            accum_img_feats.append(img_f)
            accum_txt_feats.append(txt_f)

            # 每累积 accum_freq 个 mini-batch，拼接成大 batch 计算 Loss
            if (step + 1) % args.accum_freq == 0:
                all_img = torch.cat(accum_img_feats, dim=0)
                all_txt = torch.cat(accum_txt_feats, dim=0)

                if scaler is not None:
                    with torch.amp.autocast("cuda"):
                        loss = contrastive_loss(all_img, all_txt, model.logit_scale.exp())
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss = contrastive_loss(all_img, all_txt, model.logit_scale.exp())
                    loss.backward()
                    optimizer.step()

                optimizer.zero_grad()
                scheduler.step()

                epoch_loss += loss.item()
                num_updates += 1

                # 清空累积缓冲
                accum_img_feats = []
                accum_txt_feats = []

            pbar.set_postfix(loss=f"{epoch_loss/max(num_updates,1):.4f}")

        avg_loss = epoch_loss / max(num_updates, 1)

        # 验证
        val_msg = ""
        if val_dataset is not None:
            model.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for images, texts in val_loader:
                    images, texts = images.to(device), texts.to(device)
                    if scaler is not None:
                        with torch.amp.autocast("cuda"):
                            img_feat = model.encode_image(images)
                            txt_feat = model.encode_text(texts)
                            loss = contrastive_loss(img_feat, txt_feat, model.logit_scale.exp())
                    else:
                        img_feat = model.encode_image(images)
                        txt_feat = model.encode_text(texts)
                        loss = contrastive_loss(img_feat, txt_feat, model.logit_scale.exp())
                    val_loss += loss.item()
                    val_batches += 1
            avg_val = val_loss / max(val_batches, 1)
            val_msg = f" | val_loss: {avg_val:.4f}"

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                save_path = output_dir / "best_lora.pt"
                torch.save(get_lora_state_dict(model), save_path)
                val_msg += " ⭐ best"

        logger.info(f"Epoch {epoch+1}: train_loss={avg_loss:.4f}{val_msg}")

        # 记录日志
        current_lr = optimizer.param_groups[0]["lr"]
        avg_val_str = f"{avg_val:.6f}" if val_dataset is not None else ""
        is_best = "⭐" if val_msg and "⭐" in val_msg else ""
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{epoch+1},{avg_loss:.6f},{avg_val_str},{current_lr:.8f},{is_best}\n")

        # 定期保存
        if (epoch + 1) % args.save_every == 0:
            save_path = output_dir / f"lora_epoch{epoch+1}.pt"
            torch.save(get_lora_state_dict(model), save_path)
            logger.info(f"💾 保存: {save_path}")

    # 保存最终模型
    final_path = output_dir / "lora_final.pt"
    torch.save(get_lora_state_dict(model), final_path)
    logger.info(f"🎉 训练完成！最终 LoRA 权重: {final_path}")
    logger.info(f"最佳验证 loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
