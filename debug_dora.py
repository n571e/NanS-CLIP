import torch
import torch.nn as nn
import math
from cn_clip.clip.lora import inject_lora, LoRALinear

def test_injection():
    print("Creating dummy model...")
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.MultiheadAttention(embed_dim=128, num_heads=8)
            self.linear = nn.Linear(128, 128)
            
    model = DummyModel()
    print("Initial model state:")
    # print(model)
    
    print("\nInjecting DoRA...")
    try:
        lora_params = inject_lora(model, rank=4)
        print(f"Injection successful. LoRA params found: {len(lora_params)}")
    except Exception as e:
        print(f"Injection failed with error: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\nVerifying layers...")
    if isinstance(model.attn.out_proj, LoRALinear):
        print("Success: attn.out_proj is LoRALinear")
        if hasattr(model.attn.out_proj, 'm'):
            print(f"Success: attn.out_proj has magnitude parameter 'm', shape: {model.attn.out_proj.m.shape}")
        else:
            print("Error: attn.out_proj missing magnitude parameter 'm'")
    else:
        print(f"Error: attn.out_proj type is {type(model.attn.out_proj)}")

    print("\nTesting forward pass...")
    try:
        x = torch.randn(1, 10, 128)
        # MultiheadAttention input is (L, N, E) or (N, L, E) depending on batch_first
        # By default it's (L, N, E). L=10, N=1, E=128
        out, _ = model.attn(x, x, x)
        print(f"Forward pass successful. Output shape: {out.shape}")
    except Exception as e:
        print(f"Forward pass failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_injection()
