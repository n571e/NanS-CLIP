# -*- coding: utf-8 -*-
"""
LoRA (Low-Rank Adaptation) 模块实现。

注入位置：
  - ViT 视觉编码器: MultiheadAttention.out_proj
  - RoBERTa 文本编码器: BertSelfAttention.query + BertSelfAttention.value
"""

import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """在冻结的 Linear 层上叠加 LoRA 低秩更新。

    原理: W' = W + (alpha/r) * B @ A
    - W: 原始冻结权重
    - A: (rank, in_features), Kaiming 初始化
    - B: (out_features, rank), 零初始化
    - 初始 ΔW = B @ A = 0，不破坏预训练模型
    """

    def __init__(self, original_linear: nn.Linear, rank: int = 4, alpha: float = 16.0):
        super().__init__()
        in_features = original_linear.in_features
        out_features = original_linear.out_features

        # 冻结原始权重
        self.original_linear = original_linear
        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False

        # LoRA 低秩矩阵
        device_ = original_linear.weight.device
        dtype_ = original_linear.weight.dtype
        self.lora_A = nn.Parameter(torch.empty(rank, in_features, device=device_, dtype=dtype_))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, device=device_, dtype=dtype_))

        # A 用 Kaiming 初始化，B 用零初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.scaling = alpha / rank

    @property
    def weight(self):
        # 实时计算组合后的权重，保证梯度能回传到 lora_A 和 lora_B
        return self.original_linear.weight + (self.lora_B @ self.lora_A) * self.scaling

    @property
    def bias(self):
        return self.original_linear.bias

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight, self.bias)


def inject_lora(model, rank: int = 4, alpha: float = 16.0, text_only: bool = False) -> list:
    """将 LoRA 注入到 Chinese-CLIP 的视觉编码器和文本编码器。

    注入策略：
      - ViT: nn.MultiheadAttention.out_proj（C++ 底层算子，需 @property 代理）
      - RoBERTa: BertSelfAttention.query + .value（标准 nn.Linear，直接替换）

    Args:
        model: Chinese-CLIP 模型
        rank: LoRA 秩（推荐 4 或 8）
        alpha: 缩放因子
        text_only: 是否仅对文本编码器注入 LoRA（缓解灾难性遗忘）

    Returns:
        LoRA 参数列表（用于优化器）
    """
    lora_params = []

    for name, module in model.named_modules():
        # ---- ViT 视觉编码器: q_proj, v_proj, out_proj ----
        if not text_only and isinstance(module, nn.MultiheadAttention):
            # 注入 out_proj
            if getattr(module, "out_proj", None) is not None:
                old_out = module.out_proj
                new_out = LoRALinear(old_out, rank=rank, alpha=alpha)
                module.out_proj = new_out
                lora_params.extend([new_out.lora_A, new_out.lora_B])
                
            # PyTorch 的 MultiheadAttention 有时使用分离的 q_proj_weight, k_proj_weight, v_proj_weight
            # 如果存在分离的 q 和 v 偏置/权重，我们可以直接替换（这里为了简便直接对有独立 linear 的进行替换）
            if getattr(module, "q_proj", None) is not None:
                old_q = module.q_proj
                new_q = LoRALinear(old_q, rank=rank, alpha=alpha)
                module.q_proj = new_q
                lora_params.extend([new_q.lora_A, new_q.lora_B])
                
            if getattr(module, "v_proj", None) is not None:
                old_v = module.v_proj
                new_v = LoRALinear(old_v, rank=rank, alpha=alpha)
                module.v_proj = new_v
                lora_params.extend([new_v.lora_A, new_v.lora_B])
            
            if getattr(module, "k_proj", None) is not None:
                old_k = module.k_proj
                new_k = LoRALinear(old_k, rank=rank, alpha=alpha)
                module.k_proj = new_k
                lora_params.extend([new_k.lora_A, new_k.lora_B])

        # ---- RoBERTa 文本编码器: query + value ----
        # 用类名匹配避免循环导入
        elif type(module).__name__ == "BertSelfAttention":
            # 注入 query 投影
            old_q = module.query
            new_q = LoRALinear(old_q, rank=rank, alpha=alpha)
            module.query = new_q
            lora_params.extend([new_q.lora_A, new_q.lora_B])

            # 注入 value 投影
            old_v = module.value
            new_v = LoRALinear(old_v, rank=rank, alpha=alpha)
            module.value = new_v
            lora_params.extend([new_v.lora_A, new_v.lora_B])

    return lora_params


def get_lora_state_dict(model) -> dict:
    """提取仅 LoRA 相关的参数，用于保存轻量 checkpoint"""
    return {k: v for k, v in model.state_dict().items() if "lora_" in k}


def load_lora_state_dict(model, state_dict: dict):
    """加载 LoRA 参数"""
    model_dict = model.state_dict()
    lora_dict = {k: v for k, v in state_dict.items() if "lora_" in k}
    model_dict.update(lora_dict)
    model.load_state_dict(model_dict)
