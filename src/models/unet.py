import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, d_model, context_dim, heads=8):
        super().__init__()
        assert d_model % heads == 0
        self.heads = heads
        self.head_dim = d_model // heads
        self.scale = self.head_dim ** -0.5

        self.to_q = nn.Linear(d_model, d_model, bias=False)
        self.to_k = nn.Linear(context_dim, d_model, bias=False)
        self.to_v = nn.Linear(context_dim, d_model, bias=False)
        self.to_out = nn.Linear(d_model, d_model)

    def forward(self, x, context):
        b, c, h, w = x.shape
        x_flat = x.view(b, c, h * w).permute(0, 2, 1)  # (B, H*W, C)

        q = self.to_q(x_flat)
        k = self.to_k(context)
        v = self.to_v(context)

        # Multi-head
        q = q.view(b, -1, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(b, -1, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(b, -1, self.heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, h * w, c)
        out = self.to_out(out)
        return out.permute(0, 2, 1).view(b, c, h, w)


class UNetConditional(nn.Module):
    def __init__(self, latent_channels=4, context_dim=512):
        super().__init__()
        # Timestep embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 256), nn.SiLU(), nn.Linear(256, 256)
        )
        self.time_proj1 = nn.Linear(256, 128)
        self.time_proj2 = nn.Linear(256, 256)

        # Down
        self.down1 = nn.Conv2d(latent_channels, 128, 3, padding=1)   # 16 -> 16
        self.attn1 = CrossAttention(128, context_dim)
        self.down2 = nn.Conv2d(128, 256, 4, stride=2, padding=1)      # 16 -> 8
        self.attn2 = CrossAttention(256, context_dim)

        # Up
        self.up1 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1) # 8 -> 16
        self.out = nn.Conv2d(128, latent_channels, 3, padding=1)

    def forward(self, x, t, context):
        # Timestep embeddings
        t_emb = self.time_mlp(t.float().unsqueeze(-1))          # (B, 256)
        t1 = self.time_proj1(t_emb).unsqueeze(-1).unsqueeze(-1) # (B,128,1,1)
        t2 = self.time_proj2(t_emb).unsqueeze(-1).unsqueeze(-1) # (B,256,1,1)

        # Encoder
        h1 = F.silu(self.down1(x) + t1)
        h1 = self.attn1(h1, context)

        h2 = F.silu(self.down2(h1) + t2)
        h2 = self.attn2(h2, context)

        # Decoder with skip connection
        h_up = F.silu(self.up1(h2))
        h_up = h_up + h1
        return self.out(h_up)