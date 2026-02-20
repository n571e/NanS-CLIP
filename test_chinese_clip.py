# test_chinese_clip.py
import torch
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models
from PIL import Image

print("可用模型:", available_models())
# ['ViT-B-16', 'ViT-L-14', 'ViT-L-14-336', 'ViT-H-14', 'RN50']

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

model, preprocess = load_from_name("ViT-B-16", device=device, download_root='../clip_data/pretrained_weights/')
model.eval()

# 测试中文编码！
text = clip.tokenize(["西湖美景", "德寿宫遗址", "南宋古画"]).to(device)
with torch.no_grad():
    text_features = model.encode_text(text)
    print(f"文本特征形状: {text_features.shape}")  # 期望: [3, 512]
    print("✅ 中文编码成功！")
