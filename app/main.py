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

app = FastAPI(title= "DiffusionCraft")
logger = logging.getLogger("diffusioncraft")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
project_root = Path(__file__).resolve().parents[1]


def load_first_available(model, checkpoint_paths):
    for checkpoint_path in checkpoint_paths:
        checkpoint_path = project_root / checkpoint_path
        if checkpoint_path.exists():
            try:
                model.load_state_dict(torch.load(checkpoint_path, map_location= device))
                logger.info("Loaded weights from %s", checkpoint_path)
                return checkpoint_path
            except RuntimeError:
                logger.warning("Skipping incompatible checkpoint %s", checkpoint_path)

    logger.warning("No checkpoint found for %s; using random weights.", model.__class__.__name__)
    return None


vae = VAE().to(device).eval()
text_encoder = TextEncoder().to(device).eval()
unet = UNetConditional().to(device).eval()
scheduler = DDPMScheduler()

load_first_available(vae, [Path("checkpoints/vae_celeba.pth"), Path("checkpoints_smoke/vae.pth")])
load_first_available(unet, [Path("checkpoints_smoke/unet_final.pth"), Path("checkpoints_smoke/unet_epoch10.pth")])


class GenRequest(BaseModel):
    prompt: str = Field(..., min_length= 1)
    steps: int = Field(default= 50, ge= 1, le= 1000)
    cfg_scale: float = Field(default= 1.0, ge= 1.0, le= 20.0)


@app.post("/generate")
def generate(req: GenRequest):
    try:
        with torch.no_grad():
            context = text_encoder([req.prompt], device)
            uncond_context = text_encoder([""], device) if req.cfg_scale > 1 else None

            latents = torch.randn(1, 4, 16, 16, device= device)

            step_size = max(scheduler.num_timesteps // req.steps, 1)
            timesteps = list(range(0, scheduler.num_timesteps, step_size))[::-1][:req.steps]

            for t in timesteps:
                t_tensor = torch.tensor([t], device= device).float()
                noise_pred = unet(latents, t_tensor, context)

                if uncond_context is not None:
                    uncond_pred = unet(latents, t_tensor, uncond_context)
                    noise_pred = uncond_pred + req.cfg_scale * (noise_pred - uncond_pred)

                latents = scheduler.step(noise_pred, t, latents)

            image = vae.decode(latents / 0.18215)
            image = (image.clamp(-1, 1) + 1) / 2
            image = image.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255
            img = Image.fromarray(image.astype("uint8"))

            buf = io.BytesIO()
            img.save(buf, format= "PNG")
            buf.seek(0)
            return StreamingResponse(buf, media_type= "image/png")
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host= "0.0.0.0", port= 8000)


# python -m uvicorn app.main:app --reload