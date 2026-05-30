"""
DiffusionCraft API

Run:
    python -m uvicorn app.main:app --reload
"""

import io
import logging
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
from pydantic import BaseModel, Field

from src.models.clip import TextEncoder
from src.models.unet import UNetConditional
from src.models.vae import VAE
from src.pipeline.scheduler import DDPMScheduler

logging.basicConfig(level= logging.INFO)
logger = logging.getLogger("diffusioncraft")

app = FastAPI(title= "DiffusionCraft")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
project_root = Path(__file__).resolve().parents[1]


def _try_load(model, path):
    path = project_root / path
    if path.exists():
        try:
            model.load_state_dict(torch.load(path, map_location=device))
            logger.info("Loaded %s", path)
            return True
        except RuntimeError as e:
            logger.warning("Skipping incompatible checkpoint %s: %s", path, e)
    return False


vae = VAE().to(device).eval()
text_encoder = TextEncoder().to(device).eval()
unet = UNetConditional().to(device).eval()
scheduler = DDPMScheduler()

_try_load(vae, "checkpoints/vae_celeba.pth") or _try_load(vae, "checkpoints_smoke/vae.pth")
_try_load(unet, "checkpoints/unet_final.pth") or _try_load(unet, "checkpoints_smoke/unet_final.pth")

logger.info("Models ready on %s", device)


class GenRequest(BaseModel):
    prompt: str = Field(..., min_length= 1)
    steps: int = Field(default= 50, ge= 1, le= 200)
    cfg_scale: float = Field(default= 7.5, ge=1.0, le= 20.0)  


@app.post("/generate")
def generate(req: GenRequest):
    try:
        with torch.no_grad():
            context = text_encoder([req.prompt])         # (1, 77, 512)
            uncond = text_encoder([""]) if req.cfg_scale > 1.0 else None

            latents = torch.randn(1, 4, 16, 16, device=device)

            step_size = max(scheduler.num_timesteps // req.steps, 1)
            timesteps = list(range(0, scheduler.num_timesteps, step_size))[::-1][:req.steps]

            for t in timesteps:
                t_tensor = torch.tensor([t], device=device, dtype=torch.long)
                noise_pred = unet(latents, t_tensor, context)

                if uncond is not None:
                    uncond_pred = unet(latents, t_tensor, uncond)
                    noise_pred = uncond_pred + req.cfg_scale * (noise_pred - uncond_pred)

                latents = scheduler.step(noise_pred, t, latents)

            # Decode latents → image
            image = vae.decode(latents / 0.18215)           # (1, 3, 128, 128) in [-1,1]
            image = (image.clamp(-1, 1) + 1) / 2           # → [0,1]
            image = (image.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")

            buf = io.BytesIO()
            Image.fromarray(image).save(buf, format="PNG")
            buf.seek(0)
            return StreamingResponse(buf, media_type="image/png")

    except Exception as exc:
        logger.exception("Generation failed")
        raise HTTPException(500, str(exc)) from exc


@app.get("/health")
def health():
    return {"status": "ok", "device": str(device)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
