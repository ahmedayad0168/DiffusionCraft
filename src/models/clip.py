import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPTokenizer


class TextEncoder(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32", embed_dim=512):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        clip = CLIPModel.from_pretrained(model_name)
        self.transformer = clip.text_model
        self.projection = nn.Linear(512, embed_dim)

        for param in self.transformer.parameters():
            param.requires_grad = False

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, text_list):
        tokens = self.tokenizer(
            text_list,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        with torch.no_grad():
            outputs = self.transformer(**tokens)
            hidden = outputs.last_hidden_state  # (B, 77, 512)
        return self.projection(hidden)          # (B, 77, embed_dim)
