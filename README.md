# DiffusionCraft

DiffusionCraft is a small text-to-image diffusion project built from core
PyTorch modules. It accepts a text prompt, converts that prompt into CLIP text
embeddings, denoises a latent image with a conditional UNet, decodes the latent
with a VAE, and returns a PNG image through a FastAPI endpoint.

The project is intentionally compact. It is useful for learning how the main
parts of a latent diffusion system fit together without hiding everything
behind a large framework.

## What The Project Does

- Trains a VAE that compresses images into a small latent space and reconstructs
  them back into RGB images.
- Uses a frozen CLIP text encoder to turn text captions into conditioning
  vectors.
- Trains a conditional UNet to predict diffusion noise from noisy latents,
  timesteps, and text embeddings.
- Runs reverse diffusion at inference time to generate an image from a prompt.
- Exposes generation through a FastAPI `/generate` endpoint.
- Provides a simple Streamlit UI for typing a prompt and viewing the generated
  image.

## Architecture

```text
Text prompt
  -> CLIP tokenizer and text encoder
  -> text context embeddings

Random latent noise
  -> DDPM denoising loop
  -> conditional UNet predicts noise at each timestep
  -> cleaner latent

Final latent
  -> VAE decoder
  -> RGB image
  -> PNG API response
```

During training, real images are encoded by the VAE, noise is added by the DDPM
scheduler, and the UNet learns to predict that noise while being conditioned on
the matching caption.

## Project Structure

```text
app/
  main.py          FastAPI app and /generate endpoint
  frontend.py      Streamlit UI

src/
  train.py         Conditional UNet training script
  train_vae.py     VAE training script

  data/
    dataset.py     Loads image/caption pairs
    laion-art_img/ Example .jpg/.txt training data

  models/
    clip.py        CLIP text encoder wrapper
    unet.py        Conditional UNet with cross-attention
    vae.py         VAE encoder and decoder

  pipeline/
    scheduler.py   DDPM noise scheduler
```

## Main Components

### VAE

File: `src/models/vae.py`

The VAE compresses `128x128` RGB images into latent tensors with shape
`[batch, 4, 16, 16]`. The decoder reconstructs those latents back into
`128x128` RGB images. The diffusion model works in this smaller latent space
instead of directly predicting pixels.

### Text Encoder

File: `src/models/clip.py`

The text encoder uses `openai/clip-vit-base-patch32` from Hugging Face
Transformers. CLIP is frozen, so training focuses on the VAE and UNet instead
of retraining the language/image representation.

### Conditional UNet

File: `src/models/unet.py`

The UNet predicts the noise that was added to an image latent. It receives:

- noisy latent tensor
- diffusion timestep
- CLIP text context

It uses cross-attention so the image generation process can be guided by text.

### Scheduler

File: `src/pipeline/scheduler.py`

The scheduler handles the DDPM noise schedule. It has two jobs:

- `add_noise(...)` for training
- `step(...)` for reverse diffusion during generation

### API

File: `app/main.py`

The FastAPI app loads the VAE, text encoder, UNet, and scheduler. The
`/generate` endpoint accepts JSON like this:

```json
{
  "prompt": "A watercolor painting of a cat",
  "steps": 20,
  "cfg_scale": 1.0
}
```

It returns an `image/png` response.

## Setup

Use Python 3.9 or newer.

```bash
pip install -r requirements.txt
```

If you do not want to use `requirements.txt`, install the packages manually:

```bash
pip install torch torchvision fastapi uvicorn pillow transformers streamlit requests tqdm
```

## Data Format

Training data should be image/caption pairs in the same folder:

```text
src/data/laion-art_img/
  1001.jpg
  1001.txt
  1002.jpg
  1002.txt
```

Each `.txt` file should contain the caption for the image with the same base
filename.

## Training

Train from the project root:

```bash
python -m src.train_vae --epochs 20 --batch-size 8
python -m src.train --epochs 20 --batch-size 8
```

The VAE should be trained first because the UNet training script uses the VAE
to encode training images into latents.

### Quick Training Smoke Test

Use this to make sure the training code works before starting a long run:

```bash
python -m src.train_vae --epochs 1 --batch-size 1 --max-samples 2
python -m src.train --epochs 1 --batch-size 1 --max-samples 2
```

### Training Options

Both training scripts support:

```text
--data-dir          Training data folder
--checkpoint-dir    Where model checkpoints are saved
--epochs            Number of training epochs
--batch-size        Batch size
--image-size        Image resize size
--lr                Learning rate
--num-workers       DataLoader worker count
--max-samples       Optional small subset for quick tests
```

The VAE trainer also supports:

```text
--kl-weight         Weight for the VAE KL loss term
```

## Saved Models

The VAE trainer saves:

```text
checkpoints/vae_epoch1.pth
checkpoints/vae_epoch2.pth
...
checkpoints/vae_celeba.pth
```

The UNet trainer saves:

```text
checkpoints/unet_epoch1.pth
checkpoints/unet_epoch2.pth
...
checkpoints/unet_final.pth
```

The app automatically looks for:

```text
checkpoints/vae_celeba.pth
checkpoints/unet_final.pth
checkpoints/unet_epoch10.pth
```

If compatible checkpoints are missing, the app still starts with random weights
so you can test the pipeline. Random weights prove the code path works, but
they will not produce high-quality prompt-matched images.

## Run The API

From the project root:

```bash
python -m uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

Generate an image with curl on Windows PowerShell:

```powershell
curl -X POST http://127.0.0.1:8000/generate `
  -H "Content-Type: application/json" `
  -d "{\"prompt\":\"A watercolor painting of a cat\",\"steps\":20}" `
  --output output.png
```

On macOS/Linux:

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"A watercolor painting of a cat","steps":20}' \
  --output output.png
```

## Run The Streamlit UI

Start the API first, then open a second terminal:

```bash
python -m streamlit run app/frontend.py
```

Streamlit usually opens:

```text
http://localhost:8501
```

## Verification

Useful checks:

```bash
python -m compileall app src
python -c "import app.main; print('api import ok')"
```

API smoke test:

```bash
python -c "from fastapi.testclient import TestClient; from app.main import app; r = TestClient(app).post('/generate', json={'prompt':'test','steps':1}); print(r.status_code, r.headers.get('content-type'))"
```

Expected output:

```text
200 image/png
```

## Notes And Limitations

This is an educational diffusion implementation, not a production Stable
Diffusion replacement. Quality depends heavily on data size, training time,
hardware, and checkpoint quality. For meaningful text-to-image results, train
with a larger captioned image dataset and use a CUDA GPU.
