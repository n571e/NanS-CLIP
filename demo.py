# -*- coding: utf-8 -*-
"""
Gradio 南宋文化多模态检索 Demo。

使用方法:
    conda activate pytorch
    cd d:/Desktop/CV/CLIP/NanS-CLIP

    # 安装 gradio（如果尚未安装）
    pip install gradio

    # 运行 Demo
    python demo.py

    # 使用 LoRA 微调后的模型
    python demo.py --lora_path ../clip_data/experiments/lora_song/best_lora.pt
"""

import os
import json
import argparse
from pathlib import Path

import torch
from PIL import Image

from cn_clip.clip import load_from_name, tokenize
from cn_clip.clip.lora import inject_lora, load_lora_state_dict


def load_image_database(image_dir: str, annotations_path: str):
    """加载图片数据库"""
    images = []
    image_paths = []
    titles = []

    # 如果有标注文件，用标注文件中的图片
    if os.path.isfile(annotations_path):
        with open(annotations_path, "r", encoding="utf-8") as f:
            annotations = json.load(f)
        for ann in annotations:
            img_path = os.path.join(image_dir, ann["filename"])
            if os.path.isfile(img_path):
                image_paths.append(img_path)
                titles.append(ann.get("title", ann["filename"]))
    else:
        # 直接扫描目录
        exts = {".jpg", ".jpeg", ".png", ".webp"}
        for p in sorted(Path(image_dir).iterdir()):
            if p.suffix.lower() in exts:
                image_paths.append(str(p))
                titles.append(p.stem)

    return image_paths, titles


def build_image_features(model, preprocess, image_paths, device):
    """预计算所有图片的特征向量"""
    features = []
    for path in image_paths:
        img = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            with torch.amp.autocast("cuda"):
                feat = model.encode_image(img)
                feat = feat / feat.norm(dim=-1, keepdim=True)
        features.append(feat.cpu())
    return torch.cat(features, dim=0)


def main():
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
    parser = argparse.ArgumentParser(description="南宋文化多模态检索 Demo")
    parser.add_argument("--image_dir", type=str, default="data/images")
    parser.add_argument("--annotations", type=str, default="data/annotations.json")
    parser.add_argument("--pretrained", type=str, default="../clip_data/pretrained_weights/")
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    # 延迟导入 gradio（这样 --help 不需要安装 gradio）
    try:
        import gradio as gr
    except ImportError:
        print("❌ 缺少 gradio，正在安装...")
        os.system("pip install gradio")
        import gradio as gr

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载模型
    print("📦 加载模型...")
    model, preprocess = load_from_name("ViT-B-16", device="cpu", download_root=args.pretrained)

    mode_label = "Zero-Shot"
    if args.lora_path and os.path.isfile(args.lora_path):
        inject_lora(model, rank=args.rank, alpha=args.alpha)
        state_dict = torch.load(args.lora_path, map_location="cpu")
        load_lora_state_dict(model, state_dict)
        mode_label = "LoRA 微调"
        print(f"   LoRA 已加载: {args.lora_path}")

    model = model.to(device)
    model.eval()

    # 加载图片数据库
    print("🖼️ 加载图片数据库...")
    image_paths, titles = load_image_database(args.image_dir, args.annotations)
    print(f"   共 {len(image_paths)} 张图片")

    if len(image_paths) == 0:
        print("❌ 没有找到图片，请确认 data/images/ 目录")
        return

    # 预计算图片特征
    print("🔧 预计算图片特征...")
    image_features = build_image_features(model, preprocess, image_paths, device)
    print("   完成！")

    # ---- Gradio 界面 ----
    def text_to_image_search(query_text, top_k=5):
        """文本查询 → 检索最相似图片"""
        if not query_text.strip():
            return []

        tokens = tokenize([query_text]).to(device)
        with torch.no_grad():
            with torch.amp.autocast("cuda"):
                text_feat = model.encode_text(tokens)
                text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

        sims = (text_feat.cpu() @ image_features.T).squeeze()
        topk = sims.topk(min(top_k, len(image_paths)))

        results = []
        for score, idx in zip(topk.values, topk.indices):
            img = Image.open(image_paths[idx]).convert("RGB")
            label = f"{titles[idx]} (相似度: {score:.3f})"
            results.append((img, label))
        return results

    def image_to_text_search(query_image, candidate_texts):
        """图片 → 检索最匹配的文本"""
        if query_image is None or not candidate_texts.strip():
            return "请上传图片并输入候选文本"

        img = preprocess(Image.fromarray(query_image).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            with torch.amp.autocast("cuda"):
                img_feat = model.encode_image(img)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

        text_list = [t.strip() for t in candidate_texts.split("\n") if t.strip()]
        tokens = tokenize(text_list).to(device)
        with torch.no_grad():
            with torch.amp.autocast("cuda"):
                txt_feats = model.encode_text(tokens)
                txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)

        sims = (img_feat.cpu() @ txt_feats.cpu().T).squeeze()
        sorted_indices = sims.argsort(descending=True)

        results = []
        for idx in sorted_indices:
            results.append(f"  {sims[idx]:.3f}  |  {text_list[idx]}")
        return "\n".join(results)

    # 构建界面
    with gr.Blocks(title="南宋文化多模态检索") as demo:
        # 为了兼容 Gradio 6.0，如果想设置主题，可以直接在 launch 里改
        gr.Markdown(f"""
        # 🏯 南宋文化多模态检索系统
        **模型**: Chinese-CLIP ViT-B-16 ({mode_label}) | **图片库**: {len(image_paths)} 张
        """)

        with gr.Tab("📝 文字搜图"):
            gr.Markdown("输入中文描述或古文，检索最相关的图片")
            with gr.Row():
                text_input = gr.Textbox(
                    label="输入查询文本",
                    placeholder="例：西湖美景、南宋古画、青瓷碗",
                    lines=2,
                )
                top_k_slider = gr.Slider(1, 10, value=5, step=1, label="返回数量")
            text_btn = gr.Button("🔍 搜索", variant="primary")
            gallery = gr.Gallery(label="检索结果", columns=5, height=400)
            text_btn.click(text_to_image_search, [text_input, top_k_slider], gallery)

            gr.Markdown("### 💡 试试这些查询")
            gr.Examples(
                [["西湖美景"], ["南宋山水画"], ["青瓷"], ["宫殿建筑"], ["德寿宫"]],
                inputs=[text_input],
            )

        with gr.Tab("🖼️ 以图搜文"):
            gr.Markdown("上传图片，在候选文本中找最匹配的描述")
            with gr.Row():
                img_input = gr.Image(label="上传图片")
                txt_candidates = gr.Textbox(
                    label="候选文本（每行一条）",
                    placeholder="西湖美景\n南宋古画\n宋代青瓷\n山水画卷",
                    lines=6,
                )
            img_btn = gr.Button("🔍 匹配", variant="primary")
            match_result = gr.Textbox(label="匹配结果（按相似度排序）", lines=8)
            img_btn.click(image_to_text_search, [img_input, txt_candidates], match_result)

    print(f"\n🚀 启动 Demo: http://localhost:{args.port}")
    demo.launch(server_port=args.port, share=False)


if __name__ == "__main__":
    main()
