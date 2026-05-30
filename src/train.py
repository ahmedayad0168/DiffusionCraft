"""
Train the conditional UNet (requires a trained VAE checkpoint).

Usage:
    python -m src.train --epochs 20 --batch-size 8
    python -m src.train --epochs 1 --batch-size 1 --max-samples 2   # smoke test
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.data.dataset import ImageCaptionDataset
from src.models.clip import TextEncoder
from src.models.unet import UNetConditional
from src.models.vae import VAE
from src.pipeline.scheduler import DDPMScheduler


def train_one_epoch(unet, vae, text_encoder, scheduler, dataloader, optimizer, device):
    unet.train()
    total_loss = 0.0

    for images, captions in tqdm(dataloader, desc="UNet"):
        images = images.to(device)

        with torch.no_grad():
            latents, _, _ = vae.encode(images)   # (B, 4, 16, 16)
            latents = latents * 0.18215           # scale to unit-ish variance

        with torch.no_grad():
            context = text_encoder(list(captions))  # (B, 77, 512)

        noise = torch.randn_like(latents)
        bs = latents.shape[0]
        timesteps = torch.randint(0, scheduler.num_timesteps, (bs,), device=device)

        noisy = scheduler.add_noise(latents, noise, timesteps)
        # BUG FIX: pass timesteps as long (sinusoidal_embedding expects long)
        noise_pred = unet(noisy, timesteps, context)

        loss = F.mse_loss(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(dataloader), 1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="src/data/laion-art_img")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-samples", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    checkpoint_dir = project_root / args.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    vae = VAE().to(device)
    vae_path = checkpoint_dir / "vae_celeba.pth"
    if vae_path.exists():
        try:
            vae.load_state_dict(torch.load(vae_path, map_location=device))
            print(f"Loaded VAE from {vae_path}")
        except RuntimeError:
            print("Warning: VAE checkpoint incompatible, using random weights.")
    else:
        print("Warning: no VAE checkpoint found, using random weights.")
    vae.eval()

    text_encoder = TextEncoder().to(device)
    text_encoder.eval()

    unet = UNetConditional().to(device)
    scheduler = DDPMScheduler()
    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.lr)

    dataset = ImageCaptionDataset(project_root / args.data_dir, args.image_size)
    if args.max_samples is not None:
        dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))
    print(f"Dataset size: {len(dataset)}")

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    for epoch in range(args.epochs):
        loss = train_one_epoch(unet, vae, text_encoder, scheduler, dataloader, optimizer, device)
        print(f"Epoch {epoch + 1}/{args.epochs}  loss={loss:.4f}")

    out = checkpoint_dir / "unet_final.pth"
    torch.save(unet.state_dict(), out)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
