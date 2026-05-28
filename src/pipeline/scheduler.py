import torch

class DDPMScheduler:
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_timesteps = num_timesteps
        self.num_train_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x0, noise, timesteps):
        alphas_cumprod = self.alphas_cumprod.to(x0.device)
        sqrt_alpha = alphas_cumprod[timesteps] ** 0.5
        sqrt_one_minus = (1 - alphas_cumprod[timesteps]) ** 0.5
        while sqrt_alpha.dim() < x0.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus = sqrt_one_minus.unsqueeze(-1)
        return sqrt_alpha * x0 + sqrt_one_minus * noise

    def step(self, noise_pred, t, x_t):
        alphas_cumprod = self.alphas_cumprod.to(x_t.device)
        alpha_prod_t = alphas_cumprod[t]
        alpha_prod_t_prev = alphas_cumprod[t-1] if t > 0 else torch.tensor(1.0, device=x_t.device)
        beta_prod_t = 1 - alpha_prod_t
        beta_t = self.betas[t].to(x_t.device)

        # Predict x0 from x_t and predicted noise
        pred_x0 = (x_t - (beta_prod_t ** 0.5) * noise_pred) / (alpha_prod_t ** 0.5)
        pred_x0 = pred_x0.clamp(-1, 1)

        # Compute posterior mean
        coeff1 = (alpha_prod_t_prev ** 0.5 * beta_t) / beta_prod_t
        coeff2 = (self.alphas[t].to(x_t.device) ** 0.5 * (1 - alpha_prod_t_prev)) / beta_prod_t
        x_t_prev = coeff1 * pred_x0 + coeff2 * x_t

        # Add variance
        if t > 0:
            variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * beta_t
            x_t_prev = x_t_prev + (variance ** 0.5) * torch.randn_like(x_t)
        return x_t_prev
