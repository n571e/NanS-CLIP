# -*- coding: utf-8 -*-
"""
南宋文化图片增强版多渠道爬虫 (使用 requests)

依赖:
    pip install requests
    conda install requests
"""

import os
import re
import json
import time
import random
from pathlib import Path
from hashlib import md5
import argparse

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError:
    print("❌ 缺少 requests 库，请先运行: pip install requests")
    import sys
    sys.exit(1)

# ============ 配置 ============
IMAGE_DIR = Path("data/images")
METADATA_FILE = Path("data/image_metadata.jsonl")

# 随机 User-Agent 库，避免被识别为固定爬虫
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
]

MAX_PER_QUERY = 60         # 每个关键词下载数量 (目标: 3000+ 张)
DOWNLOAD_DELAY = 0.5       # 下载间隔
MIN_IMAGE_SIZE = 20_000    # 20KB


def get_session():
    """创建一个带有自动重试机制的 Requests Session"""
    session = requests.Session()
    # 设置重试策略：遇到 429(限流), 500, 502, 503, 504 会自动退避并重试
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    # 禁用 SSL 验证警告
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    return session


def make_request(session, url, headers=None, **kwargs):
    """通用的 GET 请求，带有随机 UA"""
    hdrs = {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    }
    if headers:
        hdrs.update(headers)
    
    # 禁用 SSL 验证以应对部分国内图床证书问题
    kwargs.setdefault("verify", False)
    kwargs.setdefault("timeout", 15)
    
    return session.get(url, headers=hdrs, **kwargs)


def sanitize_fn(name):
    """清理文件名中的非法字符"""
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name)
    name = re.sub(r'\s+', '_', name)
    return name[:80]


def is_image_data(data):
    """通过文件头签名判断是否真的是图片"""
    return (data[:3] == b'\xff\xd8\xff' or          # JPEG
            data[:8] == b'\x89PNG\r\n\x1a\n' or     # PNG
            data[:4] == b'RIFF' or                   # WEBP
            data[:4] == b'GIF8')                     # GIF


def download(session, url, save_path, referer=None):
    """下载图片并保存"""
    try:
        hdrs = {}
        if referer:
            hdrs["Referer"] = referer
        resp = make_request(session, url, headers=hdrs, stream=True)
        resp.raise_for_status()
        
        data = resp.content
        if len(data) < MIN_IMAGE_SIZE:
            print(f"      - 跳过 (太小: {len(data)//1024}KB)")
            return False
        if not is_image_data(data):
            print("      - 跳过 (非有效图片格式)")
            return False
            
        with open(save_path, "wb") as f:
            f.write(data)
        return True
    except Exception as e:
        # 只打印简短错误
        print(f"      x 失败: {str(e)[:50]}")
        return False


# ══════════════════════════════════════════════
#  数据源 1: 百度图片（最稳、中文搜索体验最好）
# ══════════════════════════════════════════════
BAIDU_QUERIES = [
    # --- 南宋绘画 (山水核心) ---
    "南宋 山水画 博物馆", "马远 踏歌图 高清", "马远 寒江独钓图", "马远 秋江渔隐图", "马远 梅石溪凫图",
    "夏圭 溪山清远图", "夏圭 梧竹幽居图", "夏圭 临流抚琴图", "夏圭 山水十二景",
    "李唐 万壑松风图", "李唐 采薇图", "李唐 江山小景图",
    "刘松年 四景山水图", "刘松年 罗汉图", "刘松年 醉圣图",
    "南宋 院体画 风格", "南宋 边角之景 构图", "大斧劈皴 技法", "水墨 晕染 技法 宋画",
    "西湖十景 古画 绢本", "断桥残雪 古画", "雷峰夕照 古画", "南屏晚钟 古画",

    # --- 南宋绘画 (其他名家) ---
    "梁楷 泼墨仙人图", "梁楷 太白行吟图", "梁楷 六祖斫竹图",
    "牧溪 潇湘八景图", "牧溪 渔村夕照图", "牧溪 柿图", "牧溪 猿图",
    "赵孟坚 墨兰图", "赵孟坚 水仙图", "陈容 九龙图", "陈容 云龙图",
    "林椿 果熟禽来图", "林椿 梅竹寒禽图", "吴炳 出水芙蓉图", "李迪 枫鹰雉鸡图",
    "南宋 佚名 宫廷画", "南宋 货郎图 李嵩", "南宋 骷髅幻戏图",

    # --- 陶瓷与器物 (深度细分) ---
    "南宋官窑 瓷器 典型器", "南宋 官窑 冰裂纹 细节", "南宋 官窑 粉青釉 弦纹瓶", 
    "南宋 官窑 鬲式炉", "南宋 官窑 贯耳瓶", "杭州老虎洞窑址 出土", "修内司官窑 瓷片",
    "龙泉青瓷 南宋 典型器", "龙泉窑 梅子青 鬲式炉", "龙泉窑 粉青 凤耳瓶", "龙泉窑 贴花 瓷器",
    "南宋 建盏 兔毫", "南宋 建盏 鹧鸪斑", "建窑 曜变天目 细节", "吉州窑 木叶纹 盏",
    "吉州窑 剪纸贴花 盏", "吉州窑 玳瑁釉", "南宋 哥窑 传世 瓷器", "哥窑 金丝铁线 细节",
    "宋代 定窑 刻花 白瓷", "宋代 景德镇 青白瓷 南宋", "南宋 湖田窑 枕",
    
    # --- 金银漆木与纺织 ---
    "南宋 金银器 坑出土", "南宋 鎏金 缠枝 簪", "南宋 银铤 铭文", "金华 兰溪 南宋 墓出土",
    "南宋 螺钿 漆器", "南宋 剔红 漆奁", "南宋 剔犀 漆盘", "南宋 漆器 博物馆",
    "南宋 缂丝 织物", "宋代 刺绣 针法 细节", "南宋 铜镜 湖州铭文", "南宋 玉器 饰品",

    # --- 建筑与考古遗址 ---
    "南宋 德寿宫 重建 建筑细节", "杭州 南宋皇城 遗址 公园", "临安城 遗址 夯土", 
    "南宋 太庙 遗址", "六和塔 砖石结构", "杭州 飞来峰 宋代造像", "灵隐寺 苏堤 春晓",
    "南宋 八卦田 航拍", "小河直街 宋文化", "桥西直街 古建筑",

    # --- 典籍书法与生活 ---
    "宋代 蝴蝶装 刻本", "南宋 临安 刊本", "南宋 宋高宗 书法", "岳飞 手迹 墨宝",
    "陆游 书法 拓片", "范成大 墨迹 诗帖", "张即之 书法 楷书", "南宋 抄本 文献",
    "宋代 点茶场景 画", "宋代 斗茶图", "南宋 焚香 挂画 点茶 插花", "清河坊 宋代风情",
    "南宋 铜钱 皇宋通宝", "南宋 铁钱 遗迹", "南宋 地图 舆图 杭州"
]
def scrape_baidu(session, existing_urls, start_idx):
    print("\n" + "=" * 55)
    print("  ⭐ [数据源 1] 百度图片极速版")
    print("=" * 55)
    entries, idx = [], start_idx

    for qi, q in enumerate(BAIDU_QUERIES):
        print(f"\n  [{qi+1}/{len(BAIDU_QUERIES)}] {q}")
        cnt = 0
        for page in range(3): # 尝试抓取 3 页
            if cnt >= MAX_PER_QUERY: break
            pn = page * 30
            
            params = {
                "tn": "resultjson_com", "ipn": "rj", "ct": "201326592",
                "fp": "result", "word": q, "queryWord": q, "cl": "2",
                "lm": "-1", "ie": "utf-8", "oe": "utf-8", "st": "-1",
                "ic": "0", "istype": "2", "pn": str(pn), "rn": "30",
            }
            
            try:
                url = "https://image.baidu.com/search/acjson?" + "&".join(f"{k}={v}" for k, v in params.items())
                hdrs = {"Referer": "https://image.baidu.com/", "Accept": "application/json"}
                resp = make_request(session, url, headers=hdrs)
                
                if resp.status_code != 200:
                    print(f"    接口请求失败 ({resp.status_code})"); continue
                data = json.loads(resp.text.replace("\\'", "'"))
            except Exception as e:
                print(f"    API 失败: {str(e)[:60]}"); continue

            items = data.get("data", [])
            for item in items:
                if not isinstance(item, dict): continue
                if cnt >= MAX_PER_QUERY: break
                
                img_url = item.get("hoverURL") or item.get("middleURL") or item.get("thumbURL") or item.get("objURL", "")
                if not img_url or img_url in existing_urls:
                    continue
                    
                from_title = item.get("fromPageTitleEnc", "") or item.get("fromPageTitle", "")
                from_title = re.sub(r'<[^>]+>', '', from_title)
                
                h = md5(img_url.encode()).hexdigest()[:8]
                ext = "jpg" if ".jpg" in img_url.lower() else "png"
                fn = f"baidu_{idx:03d}_{sanitize_fn(q[:20])}_{h}.{ext}"
                
                if download(session, img_url, IMAGE_DIR / fn, referer="https://image.baidu.com/"):
                    print(f"    [{cnt+1}/{MAX_PER_QUERY}] {fn}")
                    entries.append({
                        "filename": fn, "title": from_title or q,
                        "description": f"百度图片: {q}", "categories": ["百度图片"],
                        "original_url": img_url, "source": "Baidu Images",
                    })
                    existing_urls.add(img_url)
                    idx += 1
                    cnt += 1
                time.sleep(DOWNLOAD_DELAY * random.uniform(0.5, 1.5))
            
        print(f"    共计新增: {cnt}")
    return entries, idx


# ══════════════════════════════════════════════
#  数据源 2: Wikimedia Commons
# ══════════════════════════════════════════════
WIKI_QUERIES = [
    "Southern Song dynasty painting",
    "West Lake Hangzhou",
    "Song dynasty Longquan celadon",
    "Ma Yuan painting",
    "Xia Gui painting",
    "Southern Song Ceramics",
    "Southern Song Calligraphy"
]

def scrape_wikimedia(session, existing_urls, start_idx):
    print("\n" + "=" * 55)
    print("  [数据源 2] Wikimedia Commons")
    print("=" * 55)
    entries, idx = [], start_idx

    for qi, q in enumerate(WIKI_QUERIES):
        print(f"\n  [{qi+1}/{len(WIKI_QUERIES)}] {q}")

        params = {
            "action": "query", "list": "search", "srsearch": q, 
            "srnamespace": "6", "srlimit": "10", "format": "json"
        }
        try:
            url = "https://commons.wikimedia.org/w/api.php?" + "&".join(f"{k}={v}" for k, v in params.items())
            data = make_request(session, url).json()
            results = data.get("query", {}).get("search", [])
            if not results:
                print("    无结果"); continue
                
            titles = [r["title"] for r in results]
            
            url2 = "https://commons.wikimedia.org/w/api.php"
            params2 = {
                "action": "query", "titles": "|".join(titles),
                "prop": "imageinfo", "iiprop": "url|mime",
                "iiurlwidth": "800", "format": "json"
            }
            info = make_request(session, f"{url2}?{'&'.join(f'{k}={v}' for k, v in params2.items())}").json()
            pages = info.get("query", {}).get("pages", {})
            
            cnt = 0
            for pid, pg in pages.items():
                if pid == "-1" or cnt >= MAX_PER_QUERY: continue
                ii = pg.get("imageinfo", [{}])[0]
                mime = ii.get("mime", "")
                if mime not in ("image/jpeg", "image/png"): continue
                
                # 请求缩略图以避免 429
                url = ii.get("thumburl", "") or ii.get("url", "")
                if not url or url in existing_urls: continue
                
                title = pg.get("title", "").replace("File:", "")
                ext = "jpg" if "jpeg" in mime else "png"
                fn = f"wiki_{idx:03d}_{sanitize_fn(title[:30])}.{ext}"
                
                print(f"    -> {fn}")
                if download(session, url, IMAGE_DIR / fn, referer="https://commons.wikimedia.org/"):
                    entries.append({"filename": fn, "title": title, "description": "",
                                    "categories": ["Wiki"], "original_url": url,
                                    "source": "Wikimedia Commons"})
                    existing_urls.add(url); idx += 1; cnt += 1
                time.sleep(DOWNLOAD_DELAY * 2) # Wiki 容易 429，慢点爬
            print(f"    +{cnt}")
        except Exception as e:
            print(f"    获取失败: {str(e)[:50]}")
            
    return entries, idx


# ══════════════════════════════════════════════
#  数据源 3: 大都会艺术博物馆 Open Access API（免费）
#  https://metmuseum.github.io/
# ══════════════════════════════════════════════
MET_QUERIES = [
    "Song dynasty painting",
    "Song dynasty ceramics",
    "Song dynasty calligraphy",
    "Chinese landscape painting Song",
    "Song dynasty jade",
    "Song dynasty bronze",
    "Hangzhou",
]

def scrape_met_museum(session, existing_urls, start_idx):
    print("\n" + "=" * 55)
    print("  [数据源 3] 大都会艺术博物馆 Open Access")
    print("=" * 55)
    entries, idx = [], start_idx
    base = "https://collectionapi.metmuseum.org/public/collection/v1"

    for qi, q in enumerate(MET_QUERIES):
        print(f"\n  [{qi+1}/{len(MET_QUERIES)}] {q}")
        try:
            resp = make_request(session, f"{base}/search?q={q}&hasImages=true")
            data = resp.json()
            obj_ids = data.get("objectIDs", [])
            if not obj_ids:
                print("    无结果"); continue

            cnt = 0
            for oid in obj_ids[:15]:  # 每个查询最多检查 15 件
                if cnt >= MAX_PER_QUERY:
                    break
                try:
                    obj = make_request(session, f"{base}/objects/{oid}").json()
                except Exception:
                    continue

                img_url = obj.get("primaryImage", "")
                if not img_url or img_url in existing_urls:
                    continue
                # 仅下载公开领域的
                if not obj.get("isPublicDomain", False):
                    continue

                title = obj.get("title", "") or obj.get("objectName", "")
                period = obj.get("period", "") or obj.get("dynasty", "")
                artist = obj.get("artistDisplayName", "")

                h = md5(img_url.encode()).hexdigest()[:8]
                fn = f"met_{idx:03d}_{sanitize_fn((title or q)[:25])}_{h}.jpg"

                print(f"    -> {fn}")
                if download(session, img_url, IMAGE_DIR / fn,
                            referer="https://www.metmuseum.org/"):
                    entries.append({
                        "filename": fn,
                        "title": title,
                        "description": f"{period} {artist}".strip(),
                        "categories": [obj.get("department", ""), period],
                        "original_url": img_url,
                        "source": "The Metropolitan Museum of Art",
                        "era": period,
                    })
                    existing_urls.add(img_url)
                    idx += 1
                    cnt += 1

                time.sleep(DOWNLOAD_DELAY)
            print(f"    +{cnt}")
        except Exception as e:
            print(f"    获取失败: {str(e)[:60]}")

    return entries, idx


# ══════════════════════════════════════════════
#  数据源 4: 芝加哥艺术博物馆 Open Access API（免费）
#  https://api.artic.edu/docs/
# ══════════════════════════════════════════════
ARTIC_QUERIES = [
    "Song dynasty",
    "Southern Song",
    "Chinese landscape painting",
    "Chinese ceramics Song",
    "Chinese calligraphy",
]

def scrape_artic(session, existing_urls, start_idx):
    print("\n" + "=" * 55)
    print("  [数据源 4] 芝加哥艺术博物馆 Open Access")
    print("=" * 55)
    entries, idx = [], start_idx
    base = "https://api.artic.edu/api/v1"
    iiif_base = "https://www.artic.edu/iiif/2"

    for qi, q in enumerate(ARTIC_QUERIES):
        print(f"\n  [{qi+1}/{len(ARTIC_QUERIES)}] {q}")
        try:
            resp = make_request(session, f"{base}/artworks/search",
                                params={"q": q, "limit": 15,
                                        "fields": "id,title,date_display,artist_display,"
                                                   "image_id,department_title,is_public_domain"})
            data = resp.json().get("data", [])
            if not data:
                print("    无结果"); continue

            cnt = 0
            for item in data:
                if cnt >= MAX_PER_QUERY:
                    break
                image_id = item.get("image_id")
                if not image_id or not item.get("is_public_domain", False):
                    continue

                img_url = f"{iiif_base}/{image_id}/full/843,/0/default.jpg"
                if img_url in existing_urls:
                    continue

                title = item.get("title", "")
                period = item.get("date_display", "")
                artist = item.get("artist_display", "")

                h = md5(img_url.encode()).hexdigest()[:8]
                fn = f"artic_{idx:03d}_{sanitize_fn((title or q)[:25])}_{h}.jpg"

                print(f"    -> {fn}")
                if download(session, img_url, IMAGE_DIR / fn,
                            referer="https://www.artic.edu/"):
                    entries.append({
                        "filename": fn,
                        "title": title,
                        "description": f"{period} {artist}".strip(),
                        "categories": [item.get("department_title", ""), period],
                        "original_url": img_url,
                        "source": "Art Institute of Chicago",
                        "era": period,
                    })
                    existing_urls.add(img_url)
                    idx += 1
                    cnt += 1

                time.sleep(DOWNLOAD_DELAY)
            print(f"    +{cnt}")
        except Exception as e:
            print(f"    获取失败: {str(e)[:60]}")

    return entries, idx


def main():
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)

    # 确认 requests 库
    session = get_session()

    # 加载已有记录，断点续爬
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
    
    print(f"🎒 已有 {existing_count} 条记录，断点续爬\n")
    idx = existing_count
    all_new = []

    # 1. 百度图片（最稳、最快）
    new_entries, idx = scrape_baidu(session, existing_urls, idx)
    all_new.extend(new_entries)
    
    # 2. Wikimedia Commons
    new_entries, idx = scrape_wikimedia(session, existing_urls, idx)
    all_new.extend(new_entries)

    # 3. 大都会艺术博物馆（高质量公开领域藏品）
    new_entries, idx = scrape_met_museum(session, existing_urls, idx)
    all_new.extend(new_entries)

    # 4. 芝加哥艺术博物馆（高质量 IIIF 图像）
    new_entries, idx = scrape_artic(session, existing_urls, idx)
    all_new.extend(new_entries)

    # 保存
    if all_new:
        with open(METADATA_FILE, "a", encoding="utf-8") as f:
            for entry in all_new:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    total = existing_count + len(all_new)
    print(f"\n{'=' * 55}")
    print(f"🎉 完成! 本次新增 {len(all_new)} 张，累计 {total} 张")
    print(f"  📂 图片目录: {IMAGE_DIR.resolve()}")
    print(f"  📝 元数据: {METADATA_FILE.resolve()}")
    print(f"{'=' * 55}")

    if total >= 30:
        print("\n✅ 图片数量达标！现在你可以运行 VLM 标注脚本了：")
        print("   python scripts/auto_annotate.py")


if __name__ == "__main__":
    main()

