"""Small vision models for the 241-353 AI Ecosystem workshops.

The from-scratch architectures used in both notebooks, packaged as an importable module so
the benchmark notebook can stay focused on benchmarking. See `docs/models.md` for the
architecture notes and measured comparison.

The workshops profile four models. Two come from torchvision and need no code here:

    ResNet-18            11.7 M params   torchvision.models.resnet18
    MobileNetV3-Small     2.5 M params   torchvision.models.mobilenet_v3_small

The other two are defined below, and are deliberately tiny so they run on a laptop CPU:

    TinyViT               2.9 M params   vision transformer
    TinyVisionMamba       0.16 M params  selective state-space model

`TinyVisionMamba` accepts `mixer_cls=S6Naive` (textbook, one Python step per timestep) or
`mixer_cls=S6Fast` (loop-invariant work hoisted out of the scan). They compute identical
values; only the operator granularity differs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["Attention", "ViTBlock", "TinyViT",
           "S6Naive", "S6Fast", "MambaBlock", "TinyVisionMamba", "nparams"]


def nparams(model):
    return sum(p.numel() for p in model.parameters())


# --------------------------------------------------------------------------- ViT
class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.h, self.dh = heads, dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        o = F.scaled_dot_product_attention(q, k, v)
        return self.proj(o.transpose(1, 2).reshape(B, N, D))


class ViTBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.n1, self.attn = nn.LayerNorm(dim), Attention(dim, heads)
        self.n2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.mlp(self.n2(x))


class TinyViT(nn.Module):
    """ViT-Tiny-ish: 224 px / patch 16 -> 196 tokens + CLS, dim 192, 6 blocks, 3 heads."""

    def __init__(self, img=224, patch=16, dim=192, depth=6, heads=3, n_cls=10):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, dim, patch, patch)
        n_tok = (img // patch) ** 2
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, n_tok + 1, dim))
        self.blocks = nn.ModuleList([ViTBlock(dim, heads) for _ in range(depth)])
        self.norm, self.head = nn.LayerNorm(dim), nn.Linear(dim, n_cls)

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls.expand(x.size(0), -1, -1), x], dim=1) + self.pos
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x)[:, 0])


# ------------------------------------------------------------------- Vision Mamba
class S6Naive(nn.Module):
    """Selective-scan mixer, textbook implementation: one Python step per timestep."""

    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d, self.n = d_model, d_state
        self.in_proj = nn.Linear(d_model, 2 * d_model)
        self.conv = nn.Conv1d(d_model, d_model, 3, padding=2, groups=d_model)
        self.x_proj = nn.Linear(d_model, 2 * d_state + 1)
        self.dt_proj = nn.Linear(1, d_model)
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1).float()).repeat(d_model, 1))
        self.D = nn.Parameter(torch.ones(d_model))
        self.out_proj = nn.Linear(d_model, d_model)

    def _pre(self, x):
        B, L, D = x.shape
        u, z = self.in_proj(x).chunk(2, dim=-1)
        u = F.silu(self.conv(u.transpose(1, 2))[..., :L].transpose(1, 2))
        proj = self.x_proj(u)
        dt, Bm, Cm = proj[..., :1], proj[..., 1:1 + self.n], proj[..., 1 + self.n:]
        dt = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log)
        return B, L, D, u, z, dt, Bm, Cm, A

    def forward(self, x):
        B, L, D, u, z, dt, Bm, Cm, A = self._pre(x)
        h, ys = x.new_zeros(B, D, self.n), []
        for t in range(L):                                   # sequential scan
            dA = torch.exp(dt[:, t].unsqueeze(-1) * A)
            dBu = dt[:, t].unsqueeze(-1) * Bm[:, t].unsqueeze(1) * u[:, t].unsqueeze(-1)
            h = dA * h + dBu
            ys.append((h @ Cm[:, t].unsqueeze(-1)).squeeze(-1))
        y = torch.stack(ys, dim=1) + u * self.D
        return self.out_proj(y * F.silu(z))


class S6Fast(S6Naive):
    """Identical maths to S6Naive; loop-invariant work hoisted out of the timestep loop."""

    def forward(self, x):
        B, L, D, u, z, dt, Bm, Cm, A = self._pre(x)
        dA = torch.exp(dt.unsqueeze(-1) * A)                      # (B, L, D, N) at once
        dBu = dt.unsqueeze(-1) * Bm.unsqueeze(2) * u.unsqueeze(-1)
        h, ys = x.new_zeros(B, D, self.n), []
        for t in range(L):                                        # only the recurrence
            h = dA[:, t] * h + dBu[:, t]
            ys.append(torch.einsum("bdn,bn->bd", h, Cm[:, t]))
        y = torch.stack(ys, dim=1) + u * self.D
        return self.out_proj(y * F.silu(z))


class MambaBlock(nn.Module):
    def __init__(self, dim, mixer_cls=S6Naive):
        super().__init__()
        self.norm, self.mixer = nn.LayerNorm(dim), mixer_cls(dim)

    def forward(self, x):
        return x + self.mixer(self.norm(x))


class TinyVisionMamba(nn.Module):
    """64 px / patch 8 -> 64 tokens, dim 96, 4 Mamba blocks."""

    def __init__(self, img=64, patch=8, dim=96, depth=4, n_cls=10, mixer_cls=S6Naive):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, dim, patch, patch)
        self.pos = nn.Parameter(torch.zeros(1, (img // patch) ** 2, dim))
        self.blocks = nn.ModuleList([MambaBlock(dim, mixer_cls) for _ in range(depth)])
        self.norm, self.head = nn.LayerNorm(dim), nn.Linear(dim, n_cls)

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2) + self.pos
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x).mean(dim=1))
