# -*- coding: utf-8 -*-
import json
import re
from pathlib import Path

def clean_text(text) -> str:
    """清洗 VLM 生成的长文本"""
    if not isinstance(text, str):
        if text is None:
            return ""
        text = str(text)
        
    # 1. 去除常见的 AI 前置废话
    prefixes_to_remove = [
        r"^这张图片展现了",
        r"^这张图片展示了",
        r"^这张图片显示了",
        r"^这幅图片展现了",
        r"^这幅图片展示了",
        r"^这幅图片显示了",
        r"^这张图展现了",
        r"^这张图展示了",
        r"^这幅图展示了",
        r"^图片展现了",
        r"^图片展示了",
        r"^图片显示了",
        r"^画面展现了",
        r"^画面展示了",
        r"^画面描绘了",
        r"^照片展示了",
        r"^照片展现了",
        r"^这是一幅",
        r"^这是一张",
        r"^图中是一",
        r"^图中是",
        r"^图中显示",
        r"^图中展现",
    ]
    
    # 将多个正则合并成一个以提高效率
    prefix_pattern = re.compile('|'.join(prefixes_to_remove) + r'(.*?)')
    
    # 多次替换直到开头没有违禁词（有些模型会叠杀，如 "这张图片展示了，这是一幅..."）
    old_text = ""
    while old_text != text:
        old_text = text
        text = re.sub(prefix_pattern, '', text).strip(' ，。,:\n\t')

    # 2. 移除 markdown 粗体/斜体符号
    text = text.replace('**', '').replace('*', '')

    # 3. 截断极长的冗余后缀段落（例如分段总结）
    # 通常第一段或者前两句已经足够描述核心内容。
    parts = text.split('\n')
    text = parts[0] if len(parts) > 0 else text
    
    # 进一步保证不在前 52 tokens (约 60-80 汉字) 后留下未尽之言
    # 如果实在太长，我们在 80 字处找个最近的标点砍掉
    # 这里简单处理，只要前缀去掉了，核心词自然就靠前了。
    
    return text.strip()

def main():
    input_file = Path("data/annotations.json")
    output_file = Path("data/annotations_cleaned.json")
    
    print(f"Reading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    print(f"Loaded {len(data)} records. Cleaning...")
    
    cleaned_count = 0
    valid_data = []
    keys_to_clean = ["modern_chinese", "ancient_style"]
    
    for item in data:
        # 只要存在任何一个描述字段，就进行清洗
        has_valid_text = False
        item_changed = False
        
        for k in keys_to_clean:
            original_text = item.get(k, None)
            if not original_text:
                continue
                
            has_valid_text = True
            cleaned = clean_text(original_text)
            
            if cleaned != original_text:
                item_changed = True
                
            item[k] = cleaned
            
        if has_valid_text:
            valid_data.append(item)
            if item_changed:
                cleaned_count += 1
        
    print(f"Total {cleaned_count}/{len(data)} records were cleaned. Valid records: {len(valid_data)}")
    
    print(f"Saving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(valid_data, f, ensure_ascii=False, indent=2)
        
    print("Done!")

if __name__ == "__main__":
    main()
