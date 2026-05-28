import torch


class DDPMScheduler:
    def __init__(self, num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_train_timesteps = num_train_timesteps

        # Linear noise schedule: small noise early, more noise later
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    # ── Used during TRAINING: corrupt a clean latent at timestep t ──────────
    def add_noise(self, original_samples, noise, timesteps):
        alphas_cumprod = self.alphas_cumprod.to(original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5

        # Reshape to broadcast over (batch, channels, h, w)
        while sqrt_alpha_prod.dim() < original_samples.dim():
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        return sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise

    # ── Used during INFERENCE: one reverse-diffusion step ───────────────────
    def step(self, noise_pred, timestep, sample):
        """
        Given the UNet's noise prediction at timestep t,
        compute the cleaner sample at timestep t-1.
        """
        t = timestep
        device = sample.device
        alphas_cumprod = self.alphas_cumprod.to(device)

        alpha_prod_t = alphas_cumprod[t]
        alpha_prod_t_prev = alphas_cumprod[t - 1] if t > 0 else torch.tensor(1.0, device=device)
        beta_prod_t = 1 - alpha_prod_t

        # Predict the clean sample (x0) from the noisy sample and noise prediction
        pred_x0 = (sample - beta_prod_t ** 0.5 * noise_pred) / alpha_prod_t ** 0.5
        pred_x0 = pred_x0.clamp(-1, 1)

        # Compute the previous noisy sample
        coeff_x0    = (alpha_prod_t_prev ** 0.5 * self.betas[t].to(device)) / beta_prod_t
        coeff_xt    = (self.alphas[t].to(device) ** 0.5 * (1 - alpha_prod_t_prev)) / beta_prod_t
        prev_sample = coeff_x0 * pred_x0 + coeff_xt * sample

        # Add a small amount of noise (except at the very last step)
        if t > 0:
            variance    = ((1 - alpha_prod_t_prev) / (1 - alpha_prod_t)) * self.betas[t].to(device)
            prev_sample = prev_sample + variance ** 0.5 * torch.randn_like(sample)

        return prev_sample