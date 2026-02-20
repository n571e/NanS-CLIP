# -*- coding: utf-8 -*-
"""
使用通义千问 VL（视觉语言模型）自动为图片生成多种文本描述。

前置条件:
    1. 已运行 scrape_wikimedia.py 下载图片到 data/images/
    2. 已有 data/image_metadata.jsonl
    3. 已设置环境变量 DASHSCOPE_API_KEY 或直接修改下方 API_KEY

使用方法:
    # 设置 API Key（二选一）
    set DASHSCOPE_API_KEY=sk-xxx
    # 或直接修改脚本中的 API_KEY 变量

    conda activate pytorch
    cd d:/Desktop/CV/CLIP/NanS-CLIP

    # 测试模式（只处理前 3 张）
    python scripts/auto_annotate.py --limit 3

    # 正式运行（处理所有图片）
    python scripts/auto_annotate.py

输出:
    data/annotations.json   每张图的多种文本描述
"""

import os
import sys
import json
import time
import base64
import argparse
from pathlib import Path

# ============ 配置 ============
API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-67322cb9af1c4de493fcc371d3af9493")
API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
VLM_MODEL = "qwen-vl-plus"       # 视觉语言模型（看图生文）
LLM_MODEL = "qwen-turbo"         # 纯文本模型（改写古文）

IMAGE_DIR = Path("data/images")
METADATA_FILE = Path("data/image_metadata.jsonl")
OUTPUT_FILE = Path("data/annotations.json")

# API 调用间隔（秒），避免触发频率限制
API_DELAY = 1.5


def init_client():
    """初始化 OpenAI 兼容客户端"""
    try:
        from openai import OpenAI
    except ImportError:
        print("❌ 缺少 openai 库，正在安装...")
        os.system(f"{sys.executable} -m pip install openai")
        from openai import OpenAI

    return OpenAI(api_key=API_KEY, base_url=API_BASE)


def image_to_base64(image_path: Path) -> str:
    """将图片转为 base64 字符串"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_image_mime(image_path: Path) -> str:
    """根据后缀推断 MIME 类型"""
    suffix = image_path.suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".tiff": "image/tiff",
        ".tif": "image/tiff",
    }.get(suffix, "image/jpeg")


def build_vlm_prompt(meta: dict) -> str:
    """
    根据 metadata 构建 VLM prompt。
    metadata 提供背景信息，让 VLM 不只是"盲看"图片，
    而是在知道作品标题、朝代等信息的前提下做更精准的描述。
    """
    context_parts = []

    era = meta.get("era", "")
    category = meta.get("category", "")
    title = meta.get("title", "")
    artist = meta.get("artist", "")
    description = meta.get("description", "")

    # 从 Wikimedia 的 categories 推断时代和类型
    categories = meta.get("categories", [])
    categories_str = ", ".join(categories[:5]) if categories else ""

    if title:
        context_parts.append(f"这幅作品的标题/名称是：{title}")
    if era:
        context_parts.append(f"年代：{era}")
    if artist:
        context_parts.append(f"作者：{artist}")
    if category:
        context_parts.append(f"类别：{category}")
    if description:
        context_parts.append(f"已知信息：{description[:200]}")
    if categories_str:
        context_parts.append(f"相关标签：{categories_str}")

    context = "\n".join(context_parts) if context_parts else "（无已知背景信息）"

    return f"""你是一位南宋历史与艺术研究专家。以下是关于这幅图像的背景信息：

{context}

请根据图像内容和背景信息，生成以下 3 种文本描述：

1. **现代中文描述**（modern_chinese）：50-100字，客观描述画面内容，包含视觉元素（构图、色彩、物象）和文化意义。
2. **古文风格描述**（ancient_style）：30-80字，模仿宋代笔记体（如《梦梁录》风格），用文言文描述画面。
3. **检索关键词**（keywords）：5-8个用逗号分隔的关键词，涵盖朝代、题材、技法、地点等维度。

请严格以 JSON 格式输出，示例：
{{
  "modern_chinese": "这幅南宋山水画描绘了西湖烟波浩渺的景色...",
  "ancient_style": "湖山清远，烟波浩渺，舟楫往来如织...",
  "keywords": "南宋, 山水画, 西湖, 青绿山水, 绢本设色"
}}"""


def call_vlm(client, image_path: Path, prompt: str) -> dict:
    """调用通义千问 VL，输入图片+文字，返回结构化描述"""
    b64 = image_to_base64(image_path)
    mime = get_image_mime(image_path)

    try:
        response = client.chat.completions.create(
            model=VLM_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime};base64,{b64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            temperature=0.7,
            max_tokens=800,
        )

        text = response.choices[0].message.content.strip()

        # 尝试解析 JSON（处理可能的 markdown 代码块包裹）
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        return json.loads(text)

    except json.JSONDecodeError:
        print(f"    ⚠️ JSON 解析失败，使用原始文本作为描述")
        return {
            "modern_chinese": text[:200] if text else "描述生成失败",
            "ancient_style": "",
            "keywords": ""
        }
    except Exception as e:
        print(f"    ❌ VLM 调用失败: {e}")
        return None


def load_metadata() -> dict:
    """加载 image_metadata.jsonl，返回 filename -> metadata 的映射"""
    meta_map = {}
    if METADATA_FILE.exists():
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                fname = entry.get("filename", "")
                if fname:
                    meta_map[fname] = entry
    return meta_map


def get_all_images() -> list:
    """获取 data/images/ 下所有图片文件"""
    exts = {".jpg", ".jpeg", ".png", ".webp", ".tiff", ".tif"}
    images = []
    if IMAGE_DIR.exists():
        for p in sorted(IMAGE_DIR.iterdir()):
            if p.suffix.lower() in exts and p.is_file():
                images.append(p)
    return images


def main():
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    parser = argparse.ArgumentParser(description="VLM 自动标注南宋文化图片")
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 张图片（测试用）")
    parser.add_argument("--skip-existing", action="store_true", default=True, help="跳过已标注的图片")
    args = parser.parse_args()

    # 检查 API Key
    if not API_KEY or API_KEY == "YOUR_API_KEY_HERE":
        print("❌ 请设置 DASHSCOPE_API_KEY 环境变量或修改脚本中的 API_KEY")
        print("   获取地址: https://bailian.console.aliyun.com/")
        sys.exit(1)

    # 初始化客户端
    print("🔧 初始化 API 客户端...")
    client = init_client()

    # 加载 metadata
    meta_map = load_metadata()
    print(f"📋 加载 metadata: {len(meta_map)} 条记录")

    # 获取图片列表
    images = get_all_images()
    if not images:
        print(f"❌ data/images/ 目录下没有图片")
        print(f"   请先运行: python scripts/scrape_wikimedia.py")
        sys.exit(1)

    if args.limit:
        images = images[:args.limit]
    print(f"🖼️ 待处理图片: {len(images)} 张\n")

    # 加载已有标注（支持断点续做）
    existing = {}
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            existing_list = json.load(f)
            for entry in existing_list:
                existing[entry.get("filename", "")] = entry
        print(f"📂 已有 {len(existing)} 条标注，断点续做\n")

    results = list(existing.values())
    new_count = 0

    for i, img_path in enumerate(images):
        filename = img_path.name

        # 跳过已处理的
        if args.skip_existing and filename in existing:
            continue

        print(f"[{i+1}/{len(images)}] 标注: {filename}")

        # 获取 metadata（可能没有，则用默认值）
        meta = meta_map.get(filename, {"title": filename, "description": ""})

        # 构建 prompt 并调用 VLM
        prompt = build_vlm_prompt(meta)
        vlm_result = call_vlm(client, img_path, prompt)

        if vlm_result is None:
            print(f"    ⏭️ 跳过（调用失败）")
            continue

        # 组装标注结果
        annotation = {
            "filename": filename,
            "image_path": str(img_path),
            "title": meta.get("title", filename),
            "era": meta.get("era", ""),
            "category": meta.get("category", ""),
            "source": meta.get("source", ""),
            "modern_chinese": vlm_result.get("modern_chinese", ""),
            "ancient_style": vlm_result.get("ancient_style", ""),
            "keywords": vlm_result.get("keywords", ""),
        }

        results.append(annotation)
        new_count += 1
        print(f"    ✅ 现代: {annotation['modern_chinese'][:50]}...")
        print(f"    ✅ 古文: {annotation['ancient_style'][:50]}...")

        # 每处理 5 张保存一次（防止中断丢失）
        if new_count % 5 == 0:
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"    💾 进度保存（{len(results)} 条）")

        # API 调用间隔
        time.sleep(API_DELAY)

    # 最终保存
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n🎉 标注完成！")
    print(f"   新增标注: {new_count} 条")
    print(f"   总计标注: {len(results)} 条")
    print(f"   输出文件: {OUTPUT_FILE.resolve()}")

    # 打印样例
    if results:
        sample = results[-1]
        print(f"\n📝 最后一条样例:")
        print(f"   文件: {sample['filename']}")
        print(f"   现代: {sample['modern_chinese'][:80]}")
        print(f"   古文: {sample['ancient_style'][:80]}")
        print(f"   关键词: {sample['keywords']}")


if __name__ == "__main__":
    main()
