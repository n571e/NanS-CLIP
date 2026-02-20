# -*- coding: utf-8 -*-
"""
从 Wikimedia Commons 自动爬取南宋相关图片 + 元数据。

使用方法:
    conda activate pytorch
    cd d:/Desktop/CV/CLIP/NanS-CLIP
    python scripts/scrape_wikimedia.py

输出:
    data/images/          下载的图片
    data/image_metadata.jsonl   每张图的元数据（标题、描述、来源等）
"""

import os
import json
import time
import urllib.request
import urllib.parse
from pathlib import Path

# ============ 配置 ============
IMAGE_DIR = Path("data/images")
METADATA_FILE = Path("data/image_metadata.jsonl")
USER_AGENT = "NanS-CLIP/1.0 (Academic research; Song Dynasty culture retrieval)"

# 搜索关键词列表（中英双语，尽可能覆盖南宋文化主题）
SEARCH_QUERIES = [
    # ===== 绘画类 =====
    "Southern Song dynasty painting",
    "Song dynasty scroll painting",
    "Song dynasty landscape painting",
    "Song dynasty silk painting",
    "Chinese landscape painting Song",
    "Song dynasty court painting",
    "Song dynasty ink wash painting",
    "Ma Yuan painting",          # 马远（南宋四大家）
    "Xia Gui painting",         # 夏圭（南宋四大家）
    "Li Tang painting",         # 李唐
    "Liu Songnian painting",    # 刘松年
    "Song dynasty fan painting",
    "Song dynasty album leaf",
    # ===== 人物画 & 风俗 =====
    "Song dynasty figure painting",
    "Song dynasty genre painting",
    "Song dynasty Buddhist art",
    "Song dynasty portrait",
    "Song dynasty ladies painting",
    "Song dynasty market scene",
    "Song dynasty festival painting",
    # ===== 器物 & 工艺 =====
    "Song dynasty ceramics porcelain",
    "Song dynasty Longquan celadon",
    "Song dynasty Guan ware",
    "Song dynasty Ge ware",
    "Song dynasty Jun ware",
    "Song dynasty Ding ware",
    "Song dynasty lacquerware",
    "Song dynasty jade",
    "Song dynasty bronze mirror",
    "Song dynasty gold silver",
    # ===== 建筑 & 遗址 =====
    "Song dynasty architecture",
    "Hangzhou historical site",
    "West Lake Hangzhou",
    "Liuhe Pagoda Hangzhou",
    "Leifeng Pagoda",
    "Lin'an Southern Song",
    "Song dynasty temple pagoda",
    "Deshou Palace Hangzhou",
    # ===== 书法 & 碑帖 =====
    "Song dynasty calligraphy",
    "Song dynasty inscription rubbing",
    "Song dynasty stele",
    # ===== 地图 =====
    "Song dynasty map",
    "Lin'an ancient map",
    # ===== 中文搜索 =====
    "南宋 绘画",
    "西湖 古画",
    "宋代 青瓷",
    "临安 南宋",
    "六和塔",
    "南宋 人物画",
    "宋代 书法",
    "南宋 工艺美术",
    "德寿宫",
    "雷峰塔",
]

# 每个查询最多下载的图片数
MAX_PER_QUERY = 10
# 总共最多下载的图片数
MAX_TOTAL = 300
# 最小图片尺寸（跳过太小的缩略图）
MIN_IMAGE_SIZE = 50_000  # 50KB


def wiki_api_request(params: dict) -> dict:
    """调用 Wikimedia Commons API"""
    base_url = "https://commons.wikimedia.org/w/api.php"
    params["format"] = "json"
    url = base_url + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"  ⚠️ API 请求失败: {e}")
        return {}


def search_images(query: str, limit: int = 10) -> list:
    """通过搜索 API 查找文件"""
    data = wiki_api_request({
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srnamespace": "6",  # File namespace
        "srlimit": str(limit),
    })
    return data.get("query", {}).get("search", [])


def get_image_info(titles: list) -> dict:
    """获取图片的下载 URL、描述、分类等元数据"""
    if not titles:
        return {}
    data = wiki_api_request({
        "action": "query",
        "titles": "|".join(titles),
        "prop": "imageinfo|categories",
        "iiprop": "url|size|extmetadata|mime",
        "iiurlwidth": "800",  # 请求 800px 宽的缩略图
        "cllimit": "10",
    })
    return data.get("query", {}).get("pages", {})


def extract_metadata(page: dict) -> dict:
    """从 API 返回中提取结构化元数据"""
    info = page.get("imageinfo", [{}])[0]
    ext = info.get("extmetadata", {})

    # 提取描述（可能有多个语言版本）
    description = ext.get("ImageDescription", {}).get("value", "")
    # 清除 HTML 标签
    import re
    description = re.sub(r'<[^>]+>', '', description).strip()

    # 提取分类
    categories = [c["title"].replace("Category:", "") for c in page.get("categories", [])]

    return {
        "title": page.get("title", "").replace("File:", ""),
        "description": description[:500],  # 限制长度
        "categories": categories[:10],
        "url": info.get("thumburl") or info.get("url", ""),
        "original_url": info.get("url", ""),
        "width": info.get("width", 0),
        "height": info.get("height", 0),
        "size": info.get("size", 0),
        "mime": info.get("mime", ""),
        "source": "Wikimedia Commons",
    }


def download_image(url: str, save_path: Path) -> bool:
    """下载图片到本地"""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
            if len(data) < MIN_IMAGE_SIZE:
                print(f"  ⏭️ 跳过（太小 {len(data)//1024}KB）: {save_path.name}")
                return False
            with open(save_path, "wb") as f:
                f.write(data)
            return True
    except Exception as e:
        print(f"  ❌ 下载失败: {e}")
        return False


def sanitize_filename(name: str) -> str:
    """将文件名中的特殊字符替换为下划线"""
    import re
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    name = re.sub(r'\s+', '_', name)
    return name[:100]  # 限制长度


def main():
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)

    # 加载已有的 metadata（支持断点续爬）
    existing_urls = set()
    existing_entries = []
    if METADATA_FILE.exists():
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line.strip())
                existing_urls.add(entry.get("original_url", ""))
                existing_entries.append(entry)
        print(f"📂 已有 {len(existing_entries)} 条记录，断点续爬")

    downloaded_count = len(existing_entries)
    new_entries = []

    print(f"🔍 开始搜索，目标: {MAX_TOTAL} 张图片\n")

    for qi, query in enumerate(SEARCH_QUERIES):
        if downloaded_count >= MAX_TOTAL:
            break

        print(f"[{qi+1}/{len(SEARCH_QUERIES)}] 搜索: {query}")

        # Step 1: 搜索
        results = search_images(query, limit=MAX_PER_QUERY + 5)  # 多取一些，有些会被跳过
        if not results:
            print("  没有结果")
            continue

        # Step 2: 获取详细信息（批量 API 请求）
        titles = [r["title"] for r in results]
        pages = get_image_info(titles)
        if not pages:
            continue

        # Step 3: 逐个下载
        count_this_query = 0
        for page_id, page in pages.items():
            if page_id == "-1" or downloaded_count >= MAX_TOTAL or count_this_query >= MAX_PER_QUERY:
                continue

            meta = extract_metadata(page)

            # 跳过非图片（svg, pdf 等）
            if meta["mime"] not in ("image/jpeg", "image/png", "image/webp", "image/tiff"):
                continue

            # 跳过已下载
            if meta["original_url"] in existing_urls:
                continue

            # 确定文件名
            ext = meta["mime"].split("/")[-1]
            if ext == "jpeg":
                ext = "jpg"
            filename = f"wiki_{downloaded_count:03d}_{sanitize_filename(meta['title'][:50])}.{ext}"
            save_path = IMAGE_DIR / filename

            # 下载
            url = meta["url"] or meta["original_url"]
            if not url:
                continue

            print(f"  ⬇️ 下载: {filename}")
            if download_image(url, save_path):
                meta["filename"] = filename
                meta.pop("url", None)  # 不需要缩略图 url
                new_entries.append(meta)
                existing_urls.add(meta["original_url"])
                downloaded_count += 1
                count_this_query += 1

        print(f"  ✅ 本轮下载 {count_this_query} 张\n")

        # 礼貌爬取：每轮间隔 1 秒
        time.sleep(1)

    # 保存 metadata（追加模式）
    with open(METADATA_FILE, "a", encoding="utf-8") as f:
        for entry in new_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    total = len(existing_entries) + len(new_entries)
    print(f"🎉 完成！共 {total} 张图片")
    print(f"   图片目录: {IMAGE_DIR.resolve()}")
    print(f"   元数据文件: {METADATA_FILE.resolve()}")

    if total < 30:
        print(f"\n⚠️ 图片数量偏少（{total}<30），建议：")
        print("   1. 检查网络是否需要代理")
        print("   2. 手动从故宫数字文物库 (digicol.dpm.org.cn) 补充几张")
        print("   3. 手动在 image_metadata.jsonl 中添加对应记录")


if __name__ == "__main__":
    main()
