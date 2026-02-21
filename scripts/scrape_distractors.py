# -*- coding: utf-8 -*-
"""
通用干扰图片爬虫 —— 用于 Hard Negative 评测。

爬取与南宋文化无关的通用图片，存放在独立目录 data/distractors/，
不会污染南宋训练数据集。

用法:
    python scripts/scrape_distractors.py
"""

import json
import os
import re
import time
import random
from pathlib import Path
from hashlib import md5

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError:
    print("❌ 缺少 requests 库: pip install requests")
    import sys; sys.exit(1)

# ========== 配置 ==========
# 独立目录，与南宋数据集完全分开
IMAGE_DIR = Path("data/distractors")
METADATA_FILE = Path("data/distractors_metadata.jsonl")

MAX_PER_QUERY = 8
DOWNLOAD_DELAY = 0.8
MIN_IMAGE_SIZE = 20_000  # 20KB

# 通用关键词（和南宋文化完全无关的领域）
DISTRACTOR_QUERIES = [
    # 自然风景（现代摄影）
    "现代城市建筑 高清摄影",
    "北京天安门广场 照片",
    "上海陆家嘴 夜景",
    "高铁 中国 照片",
    "现代办公室 室内",
    "手机 数码产品 摄影",
    # 动物
    "猫咪 可爱 高清",
    "大熊猫 国宝",
    "金毛 犬 宠物",
    "热带鱼 水族箱",
    # 食物
    "中国美食 高清",
    "火锅 川菜 摄影",
    "西餐 牛排 美食",
    "咖啡 拿铁 拉花",
    # 自然
    "雪山 日出 风景",
    "沙漠 驼队 摄影",
    "大海 沙滩 夏天",
    "热带雨林 探险",
    # 体育
    "篮球 NBA 比赛",
    "足球 世界杯 精彩",
    # 科技
    "人工智能 机器人",
    "太空 宇航员 NASA",
    "芯片 半导体 微距",
    # 西方艺术（和中国古代艺术形成对照）
    "油画 梵高 星空",
    "莫奈 睡莲 印象派",
    "文艺复兴 雕塑 米开朗基罗",
    # 日常
    "书架 图书馆",
    "高速公路 汽车",
    "儿童 游乐场 欢乐",
    "毕业照 大学生",
]

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
]


def sanitize_fn(name):
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name)
    name = re.sub(r'\s+', '_', name)
    return name[:60]


def is_image_data(data):
    return (data[:3] == b'\xff\xd8\xff' or
            data[:8] == b'\x89PNG\r\n\x1a\n' or
            data[:4] == b'RIFF' or
            data[:4] == b'GIF8')


def main():
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("http://", HTTPAdapter(max_retries=retries))
    session.mount("https://", HTTPAdapter(max_retries=retries))
    import urllib3; urllib3.disable_warnings()

    # 断点续爬
    existing_urls = set()
    existing_count = 0
    if METADATA_FILE.exists():
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        entry = json.loads(line.strip())
                        existing_urls.add(entry.get("original_url", ""))
                        existing_count += 1
                    except: pass

    print(f"🎯 干扰图片爬虫（Hard Negative 评测用）")
    print(f"   存放目录: {IMAGE_DIR.resolve()}")
    print(f"   已有: {existing_count} 张\n")

    idx = existing_count
    new_entries = []

    for qi, q in enumerate(DISTRACTOR_QUERIES):
        print(f"[{qi+1}/{len(DISTRACTOR_QUERIES)}] {q}")

        params = {
            "tn": "resultjson_com", "ipn": "rj", "ct": "201326592",
            "fp": "result", "word": q, "queryWord": q, "cl": "2",
            "lm": "-1", "ie": "utf-8", "oe": "utf-8", "st": "-1",
            "ic": "0", "istype": "2", "pn": "0", "rn": "20",
        }

        try:
            url = "https://image.baidu.com/search/acjson?" + "&".join(f"{k}={v}" for k, v in params.items())
            hdrs = {
                "User-Agent": random.choice(USER_AGENTS),
                "Referer": "https://image.baidu.com/",
                "Accept": "application/json",
            }
            resp = session.get(url, headers=hdrs, verify=False, timeout=15)
            if resp.status_code != 200:
                print(f"  跳过 (HTTP {resp.status_code})"); continue
            text = resp.text.replace("\\'", "'")
            data = json.loads(text)
        except Exception as e:
            print(f"  API失败: {str(e)[:50]}"); continue

        items = data.get("data", [])
        cnt = 0
        for item in items:
            if not isinstance(item, dict) or cnt >= MAX_PER_QUERY:
                continue
            img_url = item.get("hoverURL") or item.get("middleURL") or item.get("thumbURL", "")
            if not img_url or img_url in existing_urls:
                continue

            h = md5(img_url.encode()).hexdigest()[:8]
            fn = f"dist_{idx:04d}_{sanitize_fn(q[:15])}_{h}.jpg"

            try:
                r = session.get(img_url, headers={"Referer": "https://image.baidu.com/",
                                                   "User-Agent": random.choice(USER_AGENTS)},
                                verify=False, timeout=15)
                content = r.content
                if len(content) < MIN_IMAGE_SIZE or not is_image_data(content):
                    continue
                with open(IMAGE_DIR / fn, "wb") as f:
                    f.write(content)

                new_entries.append({
                    "filename": fn,
                    "query": q,
                    "original_url": img_url,
                    "source": "distractor",
                })
                existing_urls.add(img_url)
                idx += 1
                cnt += 1
            except:
                continue
            time.sleep(DOWNLOAD_DELAY)

        print(f"  +{cnt}")

    # 保存
    if new_entries:
        with open(METADATA_FILE, "a", encoding="utf-8") as f:
            for entry in new_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    total = existing_count + len(new_entries)
    print(f"\n{'='*50}")
    print(f"🎉 完成! 新增 {len(new_entries)} 张，累计 {total} 张干扰图片")
    print(f"   📂 {IMAGE_DIR.resolve()}")
    print(f"   📝 {METADATA_FILE.resolve()}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
