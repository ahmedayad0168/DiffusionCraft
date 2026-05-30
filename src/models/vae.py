import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)  # pre-norm residual — stable gradients


class VAE(nn.Module):
    def __init__(self, in_channels=3, latent_channels=4):
        super().__init__()

        # Encoder: 128 -> 64 -> 32 -> 16, outputs mean+logvar (latent_channels*2)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),   # 64
            nn.SiLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),            # 32
            nn.SiLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),           # 16
            nn.SiLU(),     
            ResBlock(256),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
            nn.Conv2d(256, latent_channels * 2, 3, padding=1),    # outputs mean & logvar
        )

        # Decoder: 16 -> 32 -> 64 -> 128
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, 256, 3, padding=1),
            nn.SiLU(),
            ResBlock(256),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 32
            nn.SiLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # 64
            nn.SiLU(),
            nn.ConvTranspose2d(64, in_channels, 4, stride=2, padding=1),  # 128
            nn.Tanh(),  # output in [-1,1], matches normalized input
        )

    def encode(self, x):
        moments = self.encoder(x)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        logvar = logvar.clamp(-30.0, 20.0)
        std = torch.exp(0.5 * logvar)
        z = mean + torch.randn_like(std) * std
        return z, mean, logvar

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z, mean, logvar = self.encode(x)
        return self.decode(z), mean, logvar
