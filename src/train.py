import torch
import torch.nn.functional as F
import argparse
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

def train_one_epoch(model, vae, text_encoder, scheduler, dataloader, optimizer, device, scaler=None):
    model.train()
    total_loss = 0
    for images, captions in tqdm(dataloader, desc="Training"):
        images = images.to(device)

        # Encode images to latents (VAE frozen)
        with torch.no_grad():
            latents, _, _ = vae.encode(images)          # (B,4,16,16)
            latents = latents * 0.18215                  # scale

        with torch.no_grad():
            context = text_encoder(list(captions), device)
            
        noise = torch.randn_like(latents)
        bs = latents.shape[0]
        timesteps = torch.randint(0, scheduler.num_timesteps, (bs,), device=device)

        noisy = scheduler.add_noise(latents, noise, timesteps)
        noise_pred = model(noisy, timesteps.float(), context)

        loss = F.mse_loss(noise_pred, noise)

        optimizer.zero_grad()
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the DiffusionCraft conditional UNet.")
    parser.add_argument("--data-dir", default="src/data/laion-art_img")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from src.models.vae import VAE
    from src.models.clip import TextEncoder
    from src.models.unet import UNetConditional
    from src.pipeline.scheduler import DDPMScheduler
    from src.data.dataset import ImageCaptionDataset

    project_root = Path(__file__).resolve().parents[1]
    checkpoints_dir = project_root / "checkpoints_smoke"
    checkpoints_dir.mkdir(exist_ok= True)

    vae = VAE().to(device)
    vae_checkpoint = checkpoints_dir / "vae_celeba.pth"
    fallback_vae_checkpoint = checkpoints_dir / "vae.pth"
    if not vae_checkpoint.exists() and fallback_vae_checkpoint.exists():
        vae_checkpoint = fallback_vae_checkpoint
    if vae_checkpoint.exists():
        try:
            vae.load_state_dict(torch.load(vae_checkpoint, map_location=device))
        except RuntimeError:
            print(f"Warning: could not load incompatible VAE checkpoint {vae_checkpoint}.")
            print("Training will use randomly initialized VAE weights.")
    else:
        print("Warning: no VAE checkpoint found; training will use randomly initialized VAE weights.")
    vae.eval()

    text_encoder = TextEncoder().to(device)
    text_encoder.eval()

    unet = UNetConditional().to(device)
    scheduler = DDPMScheduler()
    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.lr)

    dataset = ImageCaptionDataset(project_root / args.data_dir, args.image_size)
    if args.max_samples is not None:
        dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))

    dataloader = DataLoader(dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)

    for epoch in range(args.epochs):
        loss = train_one_epoch(unet, vae, text_encoder, scheduler, dataloader, optimizer, device)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
        # torch.save(unet.state_dict(), checkpoints_dir / f"unet_epoch{epoch+1}.pth")

    final_path = checkpoints_dir / "unet_final.pth"
    torch.save(unet.state_dict(), final_path)
    print(f"Saved final UNet checkpoint to {final_path}")
