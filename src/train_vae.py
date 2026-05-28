import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.data.dataset import ImageCaptionDataset
from src.models.vae import VAE


def vae_loss(reconstructed, images, mean, logvar, kl_weight):
    reconstruction_loss = F.mse_loss(reconstructed, images)
    kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
    return reconstruction_loss + kl_weight * kl_loss, reconstruction_loss, kl_loss


def train_one_epoch(model, dataloader, optimizer, device, kl_weight):
    model.train()
    total_loss = 0.0

    for images, _captions in tqdm(dataloader, desc="Training VAE"):
        images = images.to(device)

        reconstructed, mean, logvar = model(images)
        loss, reconstruction_loss, kl_loss = vae_loss(
            reconstructed,
            images,
            mean,
            logvar,
            kl_weight,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(dataloader), 1)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the DiffusionCraft VAE.")
    parser.add_argument("--data-dir", default="src/data/laion-art_img")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--kl-weight", type=float, default=1e-6)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / args.data_dir
    checkpoint_dir = project_root / args.checkpoint_dir
    checkpoint_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    dataset = ImageCaptionDataset(data_dir, args.image_size)
    if args.max_samples is not None:
        dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    for epoch in range(args.epochs):
        loss = train_one_epoch(model, dataloader, optimizer, device, args.kl_weight)
        print(f"Epoch {epoch + 1}, VAE loss: {loss:.4f}")
        # torch.save(model.state_dict(), checkpoint_dir / f"vae_epoch{epoch + 1}.pth")

    final_path = checkpoint_dir / "vae_celeba.pth"
    torch.save(model.state_dict(), final_path)
    print(f"Saved final VAE checkpoint to {final_path}")


if __name__ == "__main__":
    main()
