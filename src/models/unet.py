import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(timesteps, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=timesteps.device) / (half - 1))
    args = timesteps.float()[:, None] * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)


class CrossAttention(nn.Module):
    def __init__(self, d_model, context_dim, heads=8):
        super().__init__()
        assert d_model % heads == 0
        self.heads = heads
        self.head_dim = d_model // heads
        self.scale = self.head_dim ** -0.5

        self.norm = nn.GroupNorm(8, d_model)  # pre-norm on spatial features
        self.to_q = nn.Linear(d_model, d_model, bias=False)
        self.to_k = nn.Linear(context_dim, d_model, bias=False)
        self.to_v = nn.Linear(context_dim, d_model, bias=False)
        self.to_out = nn.Linear(d_model, d_model)

    def forward(self, x, context):
        b, c, h, w = x.shape
        # Pre-norm
        x_norm = self.norm(x).view(b, c, h * w).permute(0, 2, 1)  # (B, H*W, C)

        q = self.to_q(x_norm)
        k = self.to_k(context)
        v = self.to_v(context)

        q = q.view(b, -1, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(b, -1, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(b, -1, self.heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, h * w, c)
        out = self.to_out(out)
        out = out.permute(0, 2, 1).view(b, c, h, w)

        return x + out  


class UNetConditional(nn.Module):
    def __init__(self, latent_channels=4, context_dim=512, time_dim=256):
        super().__init__()

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )
        self.time_dim = time_dim
        self.time_proj1 = nn.Linear(time_dim, 128)
        self.time_proj2 = nn.Linear(time_dim, 256)

        # Encoder
        self.down1 = nn.Conv2d(latent_channels, 128, 3, padding=1)   # 16 → 16
        self.norm1 = nn.GroupNorm(8, 128)
        self.attn1 = CrossAttention(128, context_dim)

        self.down2 = nn.Conv2d(128, 256, 4, stride=2, padding=1)      # 16 → 8
        self.norm2 = nn.GroupNorm(8, 256)
        self.attn2 = CrossAttention(256, context_dim)

        # Decoder
        self.up1 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)  # 8 → 16
        self.norm_up = nn.GroupNorm(8, 128) 
        self.out = nn.Conv2d(128, latent_channels, 3, padding=1)

    def forward(self, x, t, context):
        t_emb = sinusoidal_embedding(t.long(), self.time_dim)   # (B, time_dim)
        t_emb = self.time_mlp(t_emb)
        t1 = self.time_proj1(t_emb).unsqueeze(-1).unsqueeze(-1)  # (B, 128, 1, 1)
        t2 = self.time_proj2(t_emb).unsqueeze(-1).unsqueeze(-1)  # (B, 256, 1, 1)

        # Encoder
        h1 = F.silu(self.norm1(self.down1(x) + t1))
        h1 = self.attn1(h1, context)                             # residual inside attn

        h2 = F.silu(self.norm2(self.down2(h1) + t2))
        h2 = self.attn2(h2, context)

        # Decoder + skip
        h_up = self.up1(h2)
        h_up = F.silu(self.norm_up(h_up + h1))                  # skip before activation

        return self.out(h_up)
