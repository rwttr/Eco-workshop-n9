"""Small vision models for the 241-353 AI Ecosystem workshops.

The from-scratch architectures used in both notebooks, packaged as an importable module so
the benchmark notebook can stay focused on benchmarking. See `docs/models.md` for the
architecture notes and measured comparison.

The workshops profile three models. Two come from torchvision and need no code here:

    ResNet-18            11.7 M params   torchvision.models.resnet18
    MobileNetV3-Small     2.5 M params   torchvision.models.mobilenet_v3_small

The third is defined below, and is deliberately tiny so it runs on a laptop CPU:

    TinyViT               2.9 M params   vision transformer

`TinyViT` accepts `fused_attn=True` (the default: one `F.scaled_dot_product_attention` call)
or `fused_attn=False` (textbook attention: matmul, softmax, matmul). They compute the same
values; only the operator granularity differs, which makes the pair the workshops' A/B
subject.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["Attention", "ViTBlock", "TinyViT", "nparams"]


def nparams(model):
    return sum(p.numel() for p in model.parameters())


# --------------------------------------------------------------------------- ViT
class Attention(nn.Module):
    def __init__(self, dim, heads, fused=True):
        super().__init__()
        self.h, self.dh, self.fused = heads, dim // heads, fused
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.fused:
            o = F.scaled_dot_product_attention(q, k, v)        # one fused kernel
        else:                                                  # textbook attention
            scores = (q @ k.transpose(-2, -1)) * self.dh ** -0.5
            o = scores.softmax(dim=-1) @ v
        return self.proj(o.transpose(1, 2).reshape(B, N, D))


class ViTBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0, fused=True):
        super().__init__()
        self.n1, self.attn = nn.LayerNorm(dim), Attention(dim, heads, fused)
        self.n2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.mlp(self.n2(x))


class TinyViT(nn.Module):
    """ViT-Tiny-ish: 224 px / patch 16 -> 196 tokens + CLS, dim 192, 6 blocks, 3 heads."""

    def __init__(self, img=224, patch=16, dim=192, depth=6, heads=3, n_cls=10, fused_attn=True):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, dim, patch, patch)
        n_tok = (img // patch) ** 2
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, n_tok + 1, dim))
        self.blocks = nn.ModuleList([ViTBlock(dim, heads, fused=fused_attn) for _ in range(depth)])
        self.norm, self.head = nn.LayerNorm(dim), nn.Linear(dim, n_cls)

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls.expand(x.size(0), -1, -1), x], dim=1) + self.pos
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x)[:, 0])

