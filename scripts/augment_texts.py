# -*- coding: utf-8 -*-
"""
LLM 文本扩增脚本。

读取已有 annotations.json，对每条标注的 modern_chinese 描述
调用纯文本 LLM（qwen-turbo，更快更便宜）生成同义改写变体，
将训练文本从 ~340 条扩增至 ~1000+ 条。

用法:
    python scripts/augment_texts.py
    python scripts/augment_texts.py --limit 5     # 测试模式
    python scripts/augment_texts.py --variants 3   # 每条生成 3 个变体
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    print("❌ 缺少 openai 库，请先运行: pip install openai")
    sys.exit(1)

# ============ 配置 ============
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANNOTATION_FILE = PROJECT_ROOT / "data" / "annotations.json"
OUTPUT_FILE = PROJECT_ROOT / "data" / "annotations_augmented.json"

API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
API_DELAY = 0.8  # 秒，qwen-turbo 限流宽松

REWRITE_PROMPT = """你是一位中国历史文化领域的文本改写专家。请对以下描述进行同义改写：

原始描述：{text}
图片标题：{title}

请生成 {n} 条不同风格的改写版本：
1. 保留核心语义和关键信息（朝代、地点、人物、技法等），但变换句式和用词
2. 每条改写长度在 40-120 字之间
3. 可以适当调整叙述角度（如从"画面展示"改为"作品描绘"）

请严格以 JSON 数组格式输出，例如：
[
  "改写版本1...",
  "改写版本2..."
]"""


def init_client():
    return OpenAI(
        api_key=API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )


def augment_text(client, text, title, n_variants=2):
    """调用 LLM 对一条文本生成 n 个同义改写变体"""
    prompt = REWRITE_PROMPT.format(text=text, title=title, n=n_variants)

    try:
        response = client.chat.completions.create(
            model="qwen-turbo",
            messages=[
                {"role": "system", "content": "你是文本改写专家，只输出 JSON 数组。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,  # 稍高温度增加多样性
            max_tokens=500,
        )
        content = response.choices[0].message.content.strip()

        # 提取 JSON 数组
        if "[" in content:
            content = content[content.index("["):content.rindex("]") + 1]
        variants = json.loads(content)

        if isinstance(variants, list):
            return [v for v in variants if isinstance(v, str) and len(v) > 10]
        return []

    except json.JSONDecodeError:
        print(f"      ⚠️ JSON 解析失败")
        return []
    except Exception as e:
        print(f"      ❌ API 失败: {str(e)[:60]}")
        return []


def main():
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

    parser = argparse.ArgumentParser(description="LLM 文本扩增")
    parser.add_argument("--annotation", type=str, default=str(ANNOTATION_FILE))
    parser.add_argument("--output", type=str, default=str(OUTPUT_FILE))
    parser.add_argument("--variants", type=int, default=2, help="每条生成几个变体 (默认 2)")
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 条 (测试用)")
    args = parser.parse_args()

    # 检查 API Key
    if not API_KEY:
        print("❌ 请设置 DASHSCOPE_API_KEY 环境变量")
        print("   export DASHSCOPE_API_KEY=your_key  (Linux/Mac)")
        print("   $env:DASHSCOPE_API_KEY='your_key'  (PowerShell)")
        sys.exit(1)

    # 加载标注
    annotation_file = Path(args.annotation)
    with open(annotation_file, "r", encoding="utf-8") as f:
        annotations = json.load(f)
    print(f"📋 加载 {len(annotations)} 条标注")

    if args.limit:
        annotations = annotations[:args.limit]
        print(f"   [测试模式] 只处理前 {args.limit} 条")

    # 初始化 LLM 客户端
    print("🔧 初始化 qwen-turbo 客户端...")
    client = init_client()

    # 加载已有扩增结果（断点续做）
    output_file = Path(args.output)
    augmented = []
    processed_files = set()
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            augmented = json.load(f)
            processed_files = {a["filename"] for a in augmented}
        print(f"📂 已有 {len(augmented)} 条扩增结果，断点续做")

    original_count = len(augmented)
    new_count = 0

    for i, ann in enumerate(annotations):
        filename = ann.get("filename", "")

        # 跳过已处理的
        if filename in processed_files:
            continue

        text = ann.get("modern_chinese", "")
        title = ann.get("title", filename)

        if not text or len(text) < 10:
            # 无有效文本，原样保留
            augmented.append(ann)
            continue

        print(f"[{i+1}/{len(annotations)}] 扩增: {filename}")
        print(f"    原文: {text[:50]}...")

        # 生成变体
        variants = augment_text(client, text, title, n_variants=args.variants)

        # 原始条目保留
        augmented.append(ann)

        # 为每个变体创建新条目
        for vi, variant in enumerate(variants):
            new_ann = ann.copy()
            new_ann["modern_chinese"] = variant
            new_ann["augmented"] = True  # 标记为扩增数据
            new_ann["augment_source"] = filename
            new_ann["augment_variant"] = vi + 1
            augmented.append(new_ann)
            print(f"    ✅ 变体{vi+1}: {variant[:50]}...")

        new_count += 1

        # 每 10 条保存一次
        if new_count % 10 == 0:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(augmented, f, ensure_ascii=False, indent=2)
            print(f"    💾 已保存 ({len(augmented)} 条)")

        time.sleep(API_DELAY)

    # 最终保存
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(augmented, f, ensure_ascii=False, indent=2)

    # 统计
    orig_texts = sum(1 for a in augmented if not a.get("augmented"))
    aug_texts = sum(1 for a in augmented if a.get("augmented"))

    print(f"\n{'='*50}")
    print(f"🎉 文本扩增完成！")
    print(f"   原始标注: {orig_texts} 条")
    print(f"   新增变体: {aug_texts} 条")
    print(f"   总计: {len(augmented)} 条")
    print(f"   输出文件: {output_file}")
    print(f"{'='*50}")

    print(f"\n💡 下一步: 用扩增后的标注重建数据集")
    print(f"   python scripts/build_dataset.py --annotation {output_file}")


if __name__ == "__main__":
    main()
