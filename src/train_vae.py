"""
Train the VAE.

Usage:
    python -m src.train_vae --epochs 20 --batch-size 8
    python -m src.train_vae --epochs 1 --batch-size 1 --max-samples 2   # smoke test
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.data.dataset import ImageCaptionDataset
from src.models.vae import VAE


def vae_loss(recon, images, mean, logvar, kl_weight):
    recon_loss = F.mse_loss(recon, images)
    # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
    return recon_loss + kl_weight * kl_loss, recon_loss, kl_loss


def train_one_epoch(model, dataloader, optimizer, device, kl_weight):
    model.train()
    total_loss = 0.0
    for images, _captions in tqdm(dataloader, desc="VAE"):
        images = images.to(device)
        recon, mean, logvar = model(images)
        loss, recon_loss, kl_loss = vae_loss(recon, images, mean, logvar, kl_weight)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
    p.add_argument("--kl-weight", type=float, default=1e-4) 
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-samples", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    checkpoint_dir = project_root / args.checkpoint_dir
    checkpoint_dir.mkdir(parents= True, exist_ok= True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = VAE().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    dataset = ImageCaptionDataset(project_root / args.data_dir, args.image_size)
    if args.max_samples is not None:
        dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))
    print(f"Dataset size: {len(dataset)}")

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    for epoch in range(args.epochs):
        loss = train_one_epoch(model, dataloader, optimizer, device, args.kl_weight)
        print(f"Epoch {epoch + 1}/{args.epochs}  loss={loss:.4f}")

    out = checkpoint_dir / "vae_celeba.pth"
    torch.save(model.state_dict(), out)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
