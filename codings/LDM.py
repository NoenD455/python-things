import tkinter as tk
from tkinter import filedialog, ttk
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import functional as F_vision
import torch.nn.functional as F_nn
from PIL import Image, ImageTk
import os
import threading
import queue
import numpy as np
import multiprocessing
import time
import random
import math
import csv
from typing import Optional, Tuple, List, Union

# ==================== Helper ====================

def load_image_as_rgb(path):
    """Load an image and convert to RGB (handle transparency by compositing on black)."""
    img = Image.open(path)
    if img.mode == 'RGBA':
        bg = Image.new('RGB', img.size, (0, 0, 0))
        bg.paste(img, mask=img.split()[3])
        return bg
    else:
        return img.convert('RGB')

def load_image_as_grayscale(path):
    """Load an image and convert to grayscale (L mode)."""
    img = Image.open(path).convert('L')
    return img

def text_to_indices(text, max_len=128):
    """Convert text to list of char indices (ASCII based). Unknown chars -> 0."""
    indices = []
    for ch in text[:max_len]:
        idx = ord(ch) if ord(ch) < 256 else 0
        indices.append(idx)
    # pad
    if len(indices) < max_len:
        indices += [0] * (max_len - len(indices))
    return indices

# ==================== Sinusoidal Embeddings ====================

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

# ==================== Text Encoders ====================

class TextEncoder(nn.Module):
    """Bidirectional GRU text encoder that outputs a conditioning vector."""
    def __init__(self, vocab_size=256, embed_dim=64, hidden_size=64, num_layers=2, cond_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_size, num_layers,
                          batch_first=True, bidirectional=True, dropout=0.1 if num_layers>1 else 0)
        self.fc = nn.Linear(hidden_size * 2, cond_dim)  # bidirectional -> *2

    def forward(self, x):
        # x: (B, T) long tensor of char indices
        emb = self.embedding(x)                       # (B, T, embed_dim)
        _, h = self.gru(emb)                           # h: (num_layers*2, B, hidden_size)
        # take last layer forward and backward
        h_fwd = h[-2, :, :]                            # (B, hidden_size)
        h_bwd = h[-1, :, :]                            # (B, hidden_size)
        h_cat = torch.cat([h_fwd, h_bwd], dim=1)       # (B, hidden_size*2)
        cond = self.fc(h_cat)                           # (B, cond_dim)
        return cond

class TransformerTextEncoder(nn.Module):
    """Transformer-based text encoder (encoder stack + mean pooling)."""
    def __init__(self, vocab_size=256, embed_dim=128, num_heads=4, num_layers=3,
                 ff_dim=256, cond_dim=512, max_len=128, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, embed_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                                   dim_feedforward=ff_dim, dropout=dropout,
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, cond_dim)

    def forward(self, x):
        # x: (B, T) token indices
        B, T = x.shape
        emb = self.embedding(x)                      # (B, T, embed_dim)
        pos = self.pos_embedding[:, :T, :]
        x = emb + pos
        x = self.transformer(x)                      # (B, T, embed_dim)
        x = x.mean(dim=1)                            # (B, embed_dim) mean pooling
        return self.fc(x)

# Preset definitions for text encoders
TEXT_ENCODER_PRESETS = {
    'BiGRU': {
        'tiny':   {'embed_dim': 32,  'hidden_size': 32,  'num_layers': 1, 'cond_dim': 128},
        'small':  {'embed_dim': 64,  'hidden_size': 64,  'num_layers': 2, 'cond_dim': 256},
        'medium': {'embed_dim': 128, 'hidden_size': 128, 'num_layers': 2, 'cond_dim': 512},
        'large':  {'embed_dim': 256, 'hidden_size': 256, 'num_layers': 3, 'cond_dim': 1024},
    },
    'BiTransformer': {
        'tiny':   {'embed_dim': 32,  'num_heads': 2, 'num_layers': 2, 'ff_dim': 128,  'cond_dim': 128},
        'small':  {'embed_dim': 64,  'num_heads': 4, 'num_layers': 3, 'ff_dim': 256,  'cond_dim': 256},
        'medium': {'embed_dim': 128, 'num_heads': 8, 'num_layers': 4, 'ff_dim': 512,  'cond_dim': 512},
        'large':  {'embed_dim': 256, 'num_heads': 8, 'num_layers': 6, 'ff_dim': 1024, 'cond_dim': 1024},
    }
}

def get_encoder_config(enc_type: str, size: str) -> dict:
    if enc_type not in TEXT_ENCODER_PRESETS:
        raise ValueError(f"Unknown encoder type '{enc_type}'. Choose 'BiGRU' or 'BiTransformer'")
    if size not in TEXT_ENCODER_PRESETS[enc_type]:
        raise ValueError(f"Unknown size '{size}' for {enc_type}. Choose {list(TEXT_ENCODER_PRESETS[enc_type].keys())}")
    return TEXT_ENCODER_PRESETS[enc_type][size]

# ==================== Improved UNet for Latent Space (with conditioning) ====================

class AttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(32, dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        residual = x
        x = self.norm(x)
        x = x.view(B, C, H * W).transpose(1, 2)  # (B, H*W, C)
        x, _ = self.attn(x, x, x)
        x = x.transpose(1, 2).view(B, C, H, W)
        return x + residual

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, cond_dim, dropout=0.1, has_attn=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        self.cond_proj = nn.Linear(cond_dim, out_channels)
        self.res_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.attn = AttentionBlock(out_channels) if has_attn else nn.Identity()

    def forward(self, x, t_emb, cond):
        # cond is always a tensor (never None) after fixes
        h = self.conv1(x)
        h = self.norm1(h)
        # add time embedding and conditioning
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = h + self.cond_proj(cond)[:, :, None, None]
        h = F_nn.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = F_nn.silu(h)
        res = self.res_conv(x)
        h = h + res
        h = self.attn(h)
        return h

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, cond_dim, dropout=0.1, has_attn=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        self.cond_proj = nn.Linear(cond_dim, out_channels)
        self.res_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.attn = AttentionBlock(out_channels) if has_attn else nn.Identity()

    def forward(self, x, t_emb, cond):
        h = self.conv1(x)
        h = self.norm1(h)
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = h + self.cond_proj(cond)[:, :, None, None]
        h = F_nn.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = F_nn.silu(h)
        res = self.res_conv(x)
        h = h + res
        h = self.attn(h)
        return h

class LatentDenoisingUNet(nn.Module):
    """
    UNet that works on latent maps of fixed spatial size (latent_h, latent_w).
    Supports conditioning via a conditioning vector (e.g., from text encoder).
    """
    def __init__(self, in_channels=8, base_channels=64, time_emb_dim=256, cond_dim=256,
                 channel_mult=(1, 2, 3, 4), dropout=0.1, attn_resolutions=(16,),
                 latent_h=4, latent_w=4):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.time_emb_dim = time_emb_dim
        self.cond_dim = cond_dim
        self.channel_mult = channel_mult
        self.attn_resolutions = set(attn_resolutions)
        self.latent_h = latent_h
        self.latent_w = latent_w

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Build the network based on latent dimensions
        self._build_network()

    def _build_network(self):
        """Build downsampling and upsampling paths based on latent size."""
        H, W = self.latent_h, self.latent_w

        # Determine number of downsampling levels possible
        num_down = 0
        cur_h, cur_w = H, W
        while cur_h >= 8 and cur_w >= 8:
            cur_h //= 2
            cur_w //= 2
            num_down += 1
        num_down = min(num_down, len(self.channel_mult))
        channel_mult_used = self.channel_mult[:num_down+1]  # include the last block without downsampling

        # Downsampling path
        self.downs = nn.ModuleList()
        cur_channels = self.base_channels
        for i, mult in enumerate(channel_mult_used):
            out_channels = self.base_channels * mult
            has_attn = (2 ** i) in self.attn_resolutions or (H // (2 ** i) in self.attn_resolutions)
            block = DownBlock(cur_channels, out_channels, self.time_emb_dim, self.cond_dim, dropout=0.1, has_attn=has_attn)
            self.downs.append(block)
            if i < len(channel_mult_used) - 1:
                self.downs.append(nn.Conv2d(out_channels, out_channels, 4, stride=2, padding=1))
            cur_channels = out_channels

        # Middle blocks
        self.mid_block1 = DownBlock(cur_channels, cur_channels, self.time_emb_dim, self.cond_dim, dropout=0.1, has_attn=True)
        self.mid_block2 = UpBlock(cur_channels, cur_channels, self.time_emb_dim, self.cond_dim, dropout=0.1, has_attn=True)

        # Upsampling path
        self.ups = nn.ModuleList()
        rev_blocks = list(reversed(channel_mult_used))
        for i, mult in enumerate(rev_blocks):
            out_channels = self.base_channels * mult
            has_attn = (2 ** (len(channel_mult_used)-1-i)) in self.attn_resolutions or (H // (2 ** (len(channel_mult_used)-1-i)) in self.attn_resolutions)
            block = UpBlock(cur_channels + out_channels, out_channels, self.time_emb_dim, self.cond_dim, dropout=0.1, has_attn=has_attn)
            self.ups.append(block)
            if i < len(rev_blocks) - 1:
                self.ups.append(nn.ConvTranspose2d(out_channels, out_channels, 4, stride=2, padding=1))
            cur_channels = out_channels

        self.final_conv = nn.Sequential(
            nn.GroupNorm(32, cur_channels),
            nn.SiLU(),
            nn.Conv2d(cur_channels, self.in_channels, 3, padding=1)
        )

    def forward(self, x, t, cond=None):
        t_emb = self.time_mlp(t)
        x = self.init_conv(x)

        # Store skip connections
        skips = []
        for layer in self.downs:
            if isinstance(layer, DownBlock):
                x = layer(x, t_emb, cond)
                skips.append(x)
            else:
                x = layer(x)

        x = self.mid_block1(x, t_emb, cond)
        x = self.mid_block2(x, t_emb, cond)

        for layer in self.ups:
            if isinstance(layer, UpBlock):
                # Take the corresponding skip from the end
                skip = skips.pop()
                x = torch.cat([x, skip], dim=1)
                x = layer(x, t_emb, cond)
            else:
                x = layer(x)

        return self.final_conv(x)

# ==================== VAE with Flexible Latent Shape and Configurable Size ====================

VAE_SIZE_CONFIGS = {
    'tiny':   (2, [0.25, 0.5]),
    'small':  (3, [0.25, 0.5, 1.0]),
    'medium': (3, [0.5, 1.0, 2.0]),
    'big':    (3, [1.0, 2.0, 4.0]),
    'large':  (3, [2.0, 4.0, 8.0]),
}

class FlexEncoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=32, latent_channels=8,
                 latent_h=4, latent_w=4, size='big'):
        super().__init__()
        self.latent_channels = latent_channels
        self.latent_h = latent_h
        self.latent_w = latent_w
        self.size = size

        if size not in VAE_SIZE_CONFIGS:
            raise ValueError(f"Unknown VAE size '{size}'. Choose from {list(VAE_SIZE_CONFIGS.keys())}")
        num_blocks, multipliers = VAE_SIZE_CONFIGS[size]

        self.down_blocks = nn.ModuleList()
        in_ch = in_channels
        for i, mult in enumerate(multipliers):
            out_ch = int(base_channels * mult)
            self.down_blocks.append(nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1))
            self.down_blocks.append(nn.LeakyReLU(0.2))
            in_ch = out_ch

        self.conv_mu = nn.Conv2d(in_ch, latent_channels, 3, padding=1)
        self.conv_logvar = nn.Conv2d(in_ch, latent_channels, 3, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((latent_h, latent_w))

    def forward(self, x):
        for layer in self.down_blocks:
            x = layer(x)
        mu = self.conv_mu(x)
        logvar = self.conv_logvar(x)
        mu = self.adaptive_pool(mu)
        logvar = self.adaptive_pool(logvar)
        return mu, logvar

class FlexDecoder(nn.Module):
    def __init__(self, latent_channels=8, base_channels=32, out_channels=3,
                 latent_h=4, latent_w=4, size='big'):
        super().__init__()
        self.latent_h = latent_h
        self.latent_w = latent_w
        self.size = size

        if size not in VAE_SIZE_CONFIGS:
            raise ValueError(f"Unknown VAE size '{size}'. Choose from {list(VAE_SIZE_CONFIGS.keys())}")
        num_blocks, multipliers = VAE_SIZE_CONFIGS[size]

        highest_mult = multipliers[-1]
        highest_ch = int(base_channels * highest_mult)
        self.init_conv = nn.Conv2d(latent_channels, highest_ch, 3, padding=1)
        self.act = nn.LeakyReLU(0.2)

        self.up_blocks = nn.ModuleList()
        in_ch = highest_ch
        for i in range(num_blocks):
            if i < num_blocks - 1:
                out_mult = multipliers[num_blocks - 2 - i]
                out_ch = int(base_channels * out_mult)
            else:
                out_ch = base_channels if multipliers[0] != 1.0 else base_channels
            self.up_blocks.append(nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1))
            self.up_blocks.append(nn.LeakyReLU(0.2))
            in_ch = out_ch

        self.conv_out = nn.Conv2d(in_ch, out_channels, 3, padding=1)

    def forward(self, z, target_size=None):
        if z.shape[2] != self.latent_h or z.shape[3] != self.latent_w:
            z = F_nn.interpolate(z, size=(self.latent_h, self.latent_w), mode='bilinear', align_corners=False)
        x = self.act(self.init_conv(z))
        for layer in self.up_blocks:
            x = layer(x)
        out = torch.tanh(self.conv_out(x))
        if target_size is not None and (out.shape[2] != target_size[0] or out.shape[3] != target_size[1]):
            out = F_nn.interpolate(out, size=target_size, mode='bilinear', align_corners=False)
        return out

class FlexVAE(nn.Module):
    def __init__(self, in_channels=3, base_channels=32, latent_channels=8,
                 latent_h=4, latent_w=4, size='big'):
        super().__init__()
        self.encoder = FlexEncoder(in_channels, base_channels, latent_channels, latent_h, latent_w, size)
        self.decoder = FlexDecoder(latent_channels, base_channels, in_channels, latent_h, latent_w, size)
        self.latent_channels = latent_channels
        self.latent_h = latent_h
        self.latent_w = latent_w
        self.size = size

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z, target_size=(x.shape[2], x.shape[3]))
        return recon, mu, logvar

    def encode(self, x):
        mu, _ = self.encoder(x)
        return mu

    def decode(self, z, target_size=None):
        return self.decoder(z, target_size=target_size)

# ==================== Latent DDPM (with conditioning and CFG) ====================

class LatentDDPM:
    def __init__(self, latent_channels=8, latent_h=4, latent_w=4, base_channels=64,
                 cond_dim=256, T=200, beta_start=1e-4, beta_end=0.02, schedule='linear', device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.T = T
        self.latent_h = latent_h
        self.latent_w = latent_w
        self.cond_dim = cond_dim

        if schedule == 'linear':
            beta = torch.linspace(beta_start, beta_end, T, device=self.device)
        elif schedule == 'cosine':
            t = torch.linspace(0, T-1, T, device=self.device) / T
            alpha_bar = torch.cos((t + 0.008) / (1 + 0.008) * math.pi / 2) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]
            alpha_bar = torch.clamp(alpha_bar, min=0.001)
            beta = 1 - alpha_bar / torch.cat([torch.tensor([1.0], device=self.device), alpha_bar[:-1]])
            beta = torch.clamp(beta, min=0.001, max=0.999)
        else:
            raise ValueError(f"Unknown schedule {schedule}")

        self.betas = beta
        self.alphas = 1 - beta
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        self.model = LatentDenoisingUNet(
            in_channels=latent_channels,
            base_channels=base_channels,
            time_emb_dim=base_channels*4,
            cond_dim=cond_dim,
            channel_mult=(1, 2, 3, 4),
            dropout=0.1,
            attn_resolutions=(16,),
            latent_h=latent_h,
            latent_w=latent_w
        ).to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=2e-4)

    def train_step(self, z0, cond=None, cfg_dropout_prob=0.0):
        batch_size = z0.size(0)
        z0 = z0.to(self.device)
        if cond is None:
            cond = torch.zeros(batch_size, self.cond_dim, device=self.device)
        else:
            cond = cond.to(self.device)
        t = torch.randint(0, self.T, (batch_size,), device=self.device).long()
        alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1, 1)
        noise = torch.randn_like(z0)
        zt = torch.sqrt(alpha_bar_t) * z0 + torch.sqrt(1 - alpha_bar_t) * noise
        if cfg_dropout_prob > 0:
            mask = torch.rand(batch_size, 1, device=self.device) > cfg_dropout_prob
            cond_dropped = cond * mask.float()
        else:
            cond_dropped = cond
        noise_pred = self.model(zt, t, cond=cond_dropped)
        loss = self.criterion(noise_pred, noise)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def sample(self, n_samples=16, cond=None, steps=None, method='ddpm', eta=0.0, cfg_scale=1.0, progress_callback=None):
        if method.lower() == 'ddpm':
            return self._sample_ddpm(n_samples, cond, steps, cfg_scale, progress_callback)
        elif method.lower() == 'ddim':
            return self._sample_ddim(n_samples, cond, steps, eta, cfg_scale, progress_callback)
        else:
            raise ValueError("method must be 'ddpm' or 'ddim'")

    def _sample_ddpm(self, n_samples=16, cond=None, n_steps=None, cfg_scale=1.0, progress_callback=None):
        if n_steps is None:
            n_steps = self.T
        img_shape = (self.model.in_channels, self.latent_h, self.latent_w)
        timesteps = torch.linspace(self.T-1, 0, n_steps, dtype=torch.long, device=self.device)
        x = torch.randn(n_samples, *img_shape, device=self.device)
        if cond is None:
            cond = torch.zeros(n_samples, self.cond_dim, device=self.device)
        else:
            cond = cond.to(self.device)
        null_cond = torch.zeros_like(cond)

        for i, t in enumerate(timesteps):
            t_tensor = torch.full((n_samples,), t, device=self.device, dtype=torch.long)
            if cfg_scale != 1.0:
                noise_pred_cond = self.model(x, t_tensor, cond=cond)
                noise_pred_uncond = self.model(x, t_tensor, cond=null_cond)
                noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = self.model(x, t_tensor, cond=cond)
            alpha_t = self.alphas[t]
            alpha_bar_t = self.alpha_bars[t]
            beta_t = self.betas[t]
            coeff_x = 1.0 / torch.sqrt(alpha_t)
            coeff_noise = beta_t / torch.sqrt(1.0 - alpha_bar_t)
            if t > 0:
                z = torch.randn_like(x)
                sigma_t = torch.sqrt(beta_t)
            else:
                z = 0.0
            x = coeff_x * (x - coeff_noise * noise_pred) + sigma_t * z
            if progress_callback is not None:
                x0_est = (x - torch.sqrt(1.0 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
                progress_callback(i+1, x, x0_est)
        return x

    def _sample_ddim(self, n_samples=16, cond=None, n_steps=50, eta=0.0, cfg_scale=1.0, progress_callback=None):
        img_shape = (self.model.in_channels, self.latent_h, self.latent_w)
        step_seq = torch.linspace(0, self.T - 1, n_steps, dtype=torch.long, device=self.device)
        step_seq = step_seq.flip(0)
        x = torch.randn(n_samples, *img_shape, device=self.device)
        if cond is None:
            cond = torch.zeros(n_samples, self.cond_dim, device=self.device)
        else:
            cond = cond.to(self.device)
        null_cond = torch.zeros_like(cond)

        for i in range(len(step_seq) - 1):
            t = step_seq[i]
            t_next = step_seq[i + 1]
            t_tensor = torch.full((n_samples,), t, device=self.device, dtype=torch.long)
            if cfg_scale != 1.0:
                noise_pred_cond = self.model(x, t_tensor, cond=cond)
                noise_pred_uncond = self.model(x, t_tensor, cond=null_cond)
                noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = self.model(x, t_tensor, cond=cond)
            alpha_bar_t = self.alpha_bars[t]
            alpha_bar_t_next = self.alpha_bars[t_next]
            x0_pred = (x - torch.sqrt(1.0 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
            sigma = eta * torch.sqrt((1.0 - alpha_bar_t_next) / (1.0 - alpha_bar_t) *
                                     (1.0 - alpha_bar_t / alpha_bar_t_next))
            dir_xt = torch.sqrt(1.0 - alpha_bar_t_next - sigma ** 2) * noise_pred
            x = torch.sqrt(alpha_bar_t_next) * x0_pred + dir_xt
            if eta > 0:
                x = x + sigma * torch.randn_like(x)
            if progress_callback is not None:
                progress_callback(i+1, x, x0_pred)
        return x

    @torch.no_grad()
    def sample_step_by_step(self, n_samples=16, cond=None, steps=None, method='ddpm', eta=0.0, cfg_scale=1.0):
        if method.lower() == 'ddpm':
            yield from self._sample_step_by_step_ddpm(n_samples, cond, steps, cfg_scale)
        elif method.lower() == 'ddim':
            yield from self._sample_step_by_step_ddim(n_samples, cond, steps, eta, cfg_scale)
        else:
            raise ValueError("method must be 'ddpm' or 'ddim'")

    def _sample_step_by_step_ddpm(self, n_samples=16, cond=None, n_steps=None, cfg_scale=1.0):
        if n_steps is None:
            n_steps = self.T
        img_shape = (self.model.in_channels, self.latent_h, self.latent_w)
        timesteps = torch.linspace(self.T-1, 0, n_steps, dtype=torch.long, device=self.device)
        x = torch.randn(n_samples, *img_shape, device=self.device)
        if cond is None:
            cond = torch.zeros(n_samples, self.cond_dim, device=self.device)
        else:
            cond = cond.to(self.device)
        null_cond = torch.zeros_like(cond)

        for i, t in enumerate(timesteps):
            t_tensor = torch.full((n_samples,), t, device=self.device, dtype=torch.long)
            if cfg_scale != 1.0:
                noise_pred_cond = self.model(x, t_tensor, cond=cond)
                noise_pred_uncond = self.model(x, t_tensor, cond=null_cond)
                noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = self.model(x, t_tensor, cond=cond)
            alpha_t = self.alphas[t]
            alpha_bar_t = self.alpha_bars[t]
            beta_t = self.betas[t]
            coeff_x = 1.0 / torch.sqrt(alpha_t)
            coeff_noise = beta_t / torch.sqrt(1.0 - alpha_bar_t)
            if t > 0:
                z = torch.randn_like(x)
                sigma_t = torch.sqrt(beta_t)
            else:
                z = 0.0
            x = coeff_x * (x - coeff_noise * noise_pred) + sigma_t * z
            yield t, x.clone()

    def _sample_step_by_step_ddim(self, n_samples=16, cond=None, n_steps=50, eta=0.0, cfg_scale=1.0):
        img_shape = (self.model.in_channels, self.latent_h, self.latent_w)
        step_seq = torch.linspace(0, self.T - 1, n_steps, dtype=torch.long, device=self.device)
        step_seq = step_seq.flip(0)
        x = torch.randn(n_samples, *img_shape, device=self.device)
        if cond is None:
            cond = torch.zeros(n_samples, self.cond_dim, device=self.device)
        else:
            cond = cond.to(self.device)
        null_cond = torch.zeros_like(cond)

        for i in range(len(step_seq) - 1):
            t = step_seq[i]
            t_next = step_seq[i + 1]
            t_tensor = torch.full((n_samples,), t, device=self.device, dtype=torch.long)
            if cfg_scale != 1.0:
                noise_pred_cond = self.model(x, t_tensor, cond=cond)
                noise_pred_uncond = self.model(x, t_tensor, cond=null_cond)
                noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = self.model(x, t_tensor, cond=cond)
            alpha_bar_t = self.alpha_bars[t]
            alpha_bar_t_next = self.alpha_bars[t_next]
            x0_pred = (x - torch.sqrt(1.0 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
            sigma = eta * torch.sqrt((1.0 - alpha_bar_t_next) / (1.0 - alpha_bar_t) *
                                     (1.0 - alpha_bar_t / alpha_bar_t_next))
            dir_xt = torch.sqrt(1.0 - alpha_bar_t_next - sigma ** 2) * noise_pred
            x = torch.sqrt(alpha_bar_t_next) * x0_pred + dir_xt
            if eta > 0:
                x = x + sigma * torch.randn_like(x)
            yield t, x.clone()

# ==================== Conditional Dataset (supports multiple captions per image) ====================

class ConditionalImageDataset(Dataset):
    def __init__(self, image_paths, labels_per_image, img_size=32, color_mode='rgb',
                 aug_settings=None, text_max_len=128):
        """
        labels_per_image: list of same length as image_paths.
        Each element can be a string or a list of strings (multiple captions).
        During __getitem__, a random caption is chosen if multiple.
        """
        self.image_paths = image_paths
        # Normalize: each element becomes a list of strings
        self.labels = []
        for lbl in labels_per_image:
            if isinstance(lbl, str):
                self.labels.append([lbl])
            elif isinstance(lbl, list):
                self.labels.append(lbl)
            else:
                self.labels.append([''])  # fallback
        self.img_size = img_size
        self.color_mode = color_mode.lower()
        self.aug_settings = aug_settings or {}
        self.text_max_len = text_max_len
        if self.color_mode == 'rgb':
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])

    def __len__(self):
        return len(self.image_paths)

    def apply_augmentations(self, pil_img):
        img = pil_img.copy()
        if self.aug_settings.get('flip_horizontal', False) and random.random() < 0.5:
            img = F_vision.hflip(img)
        if self.aug_settings.get('rotation', False) and random.random() < 0.5:
            angle = random.uniform(-30, 30)
            img = F_vision.rotate(img, angle, interpolation=Image.BICUBIC)
        return img

    def __getitem__(self, idx):
        try:
            if self.color_mode == 'rgb':
                pil_img = load_image_as_rgb(self.image_paths[idx])
            else:
                pil_img = load_image_as_grayscale(self.image_paths[idx])
            pil_img = self.apply_augmentations(pil_img)
            img_tensor = self.transform(pil_img)

            # Randomly pick one caption for this image
            captions = self.labels[idx]
            chosen_text = random.choice(captions).replace('_', ' ')
            text_indices = text_to_indices(chosen_text, self.text_max_len)
            text_tensor = torch.tensor(text_indices, dtype=torch.long)

            return img_tensor, text_tensor
        except Exception as e:
            print(f"Error loading {self.image_paths[idx]}: {e}")
            if self.color_mode == 'rgb':
                img_tensor = torch.zeros(3, self.img_size, self.img_size)
            else:
                img_tensor = torch.zeros(1, self.img_size, self.img_size)
            text_tensor = torch.zeros(self.text_max_len, dtype=torch.long)
            return img_tensor, text_tensor

# ==================== GUI Application ====================

class LatentDiffusionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Latent Diffusion - Conditional with Flexible Text Encoder")

        # --- Dynamic window sizing (80% of screen, min 1000x700) ---
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        win_width = max(1000, int(screen_width * 0.8))
        win_height = max(700, int(screen_height * 0.8))
        self.root.geometry(f"{win_width}x{win_height}")
        self.root.minsize(900, 650)

        # --- Basic DPI awareness (Windows) ---
        try:
            import ctypes
            awareness = ctypes.c_int()
            ctypes.windll.shcore.GetProcessDpiAwareness(0, ctypes.byref(awareness))
            if awareness.value == 0:
                ctypes.windll.shcore.SetProcessDpiAwareness(1)
            dpi = ctypes.windll.user32.GetDpiForWindow(root.winfo_id())
            scale = dpi / 72.0
            self.root.tk.call('tk', 'scaling', scale)
        except Exception:
            pass

        self.image_paths = []
        self.labels = []               # list of lists (multiple captions per image)
        self.csv_path = None
        self.training_vae = False
        self.training_latent = False
        self.vae_model = None
        self.latent_model = None
        self.text_encoder = None

        self.current_epoch_vae = 0
        self.current_epoch_latent = 0

        self.message_queue_vae = queue.Queue()
        self.message_queue_latent = queue.Queue()
        self.progressive_active = False

        # Settings variables
        self.settings = {
            # Image
            'img_size': tk.IntVar(value=32),
            'color_mode': tk.StringVar(value='rgb'),

            # VAE
            'vae_size': tk.StringVar(value='big'),
            'vae_base_channels': tk.IntVar(value=32),
            'vae_latent_channels': tk.IntVar(value=8),
            'vae_latent_h': tk.IntVar(value=4),
            'vae_latent_w': tk.IntVar(value=4),
            'vae_batch_size': tk.IntVar(value=16),
            'vae_lr': tk.DoubleVar(value=1e-3),
            'vae_num_workers': tk.IntVar(value=0),
            'vae_kl_weight': tk.DoubleVar(value=0.0001),

            # Latent DDPM
            'latent_base_channels': tk.IntVar(value=64),
            'latent_T': tk.IntVar(value=200),
            'latent_beta_start': tk.DoubleVar(value=1e-4),
            'latent_beta_end': tk.DoubleVar(value=0.02),
            'latent_batch_size': tk.IntVar(value=16),
            'latent_lr': tk.DoubleVar(value=2e-4),
            'beta_schedule': tk.StringVar(value='linear'),
            'cfg_dropout_prob': tk.DoubleVar(value=0.1),

            # Text conditioning
            'cond_enabled': tk.BooleanVar(value=False),
            'text_encoder_type': tk.StringVar(value='BiGRU'),
            'text_encoder_size': tk.StringVar(value='small'),
            'cond_embed_dim': tk.IntVar(value=64),
            'cond_hidden_size': tk.IntVar(value=64),
            'cond_num_layers': tk.IntVar(value=2),
            'cond_num_heads': tk.IntVar(value=4),     # for transformer
            'cond_ff_dim': tk.IntVar(value=256),      # for transformer
            'cond_dim': tk.IntVar(value=256),
            'cond_text_max_len': tk.IntVar(value=128),
            'cond_lr': tk.DoubleVar(value=5e-4),
        }

        # Augmentation settings
        self.aug_settings = {
            'flip_horizontal': tk.BooleanVar(value=True),
            'rotation': tk.BooleanVar(value=False),
        }

        # Generation sampler choice and CFG scale
        self.sampler_method = tk.StringVar(value='ddpm')
        self.gen_temperature = tk.DoubleVar(value=1.0)
        self.cfg_scale = tk.DoubleVar(value=2.0)

        self.thumbnail_size = 128

        self.setup_gui()
        self.root.after(100, self.process_messages_vae)
        self.root.after(100, self.process_messages_latent)

    def setup_gui(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Tab 1: VAE Training
        self.vae_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.vae_tab, text='VAE Training')
        self.setup_vae_tab()

        # Tab 2: Latent DDPM Training
        self.latent_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.latent_tab, text='Latent DDPM Training')
        self.setup_latent_tab()

        # Tab 3: Settings
        self.settings_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.settings_tab, text='Settings')
        self.setup_settings_tab()

        # Tab 4: Generation
        self.generation_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.generation_tab, text='Generation')
        self.setup_generation_tab()

        self.status_label = tk.Label(self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    # ------------------------------------------------------------------
    # VAE Training Tab
    # ------------------------------------------------------------------
    def setup_vae_tab(self):
        main_frame = tk.Frame(self.vae_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_frame = tk.Frame(main_frame, width=300)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0,10))
        left_frame.pack_propagate(False)

        tk.Label(left_frame, text="VAE Training", font=("Arial",12,"bold")).pack(pady=(0,10))

        img_frame = tk.LabelFrame(left_frame, text="Training Images", padx=5, pady=5)
        img_frame.pack(fill=tk.X, pady=(0,10))
        tk.Button(img_frame, text="Add Images", command=self.add_images, width=20).pack(pady=2)
        tk.Button(img_frame, text="Add Folder (recursive)", command=self.add_folder, width=20).pack(pady=2)
        tk.Button(img_frame, text="Clear All", command=self.clear_images, width=20).pack(pady=2)
        self.image_listbox = tk.Listbox(img_frame, height=5)
        self.image_listbox.pack(fill=tk.X, pady=2)

        tk.Button(left_frame, text="Initialize VAE", command=self.initialize_vae, width=20).pack(pady=5)
        epoch_frame = tk.Frame(left_frame)
        epoch_frame.pack(pady=5)
        tk.Label(epoch_frame, text="Epochs:").pack(side=tk.LEFT)
        self.vae_epoch_var = tk.StringVar(value="100")
        tk.Entry(epoch_frame, textvariable=self.vae_epoch_var, width=8).pack(side=tk.LEFT, padx=5)

        tk.Button(left_frame, text="Start VAE Training", command=self.start_vae_training,
                  width=20, bg="lightgreen").pack(pady=5)
        tk.Button(left_frame, text="Stop VAE Training", command=self.stop_vae_training,
                  width=20, bg="salmon").pack(pady=5)
        tk.Button(left_frame, text="Save VAE", command=self.save_vae, width=20).pack(pady=5)
        tk.Button(left_frame, text="Load VAE", command=self.load_vae, width=20).pack(pady=5)

        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        preview_frame = tk.LabelFrame(right_frame, text="Reconstructions (Top: Real, Bottom: Recon)", padx=5, pady=5)
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=(0,5))
        self.vae_preview_canvas = tk.Canvas(preview_frame, bg='gray', width=256, height=128)
        self.vae_preview_canvas.pack()

        log_frame = tk.LabelFrame(right_frame, text="Log", padx=5, pady=5)
        log_frame.pack(fill=tk.BOTH, expand=True)
        self.vae_log_text = tk.Text(log_frame, height=15, font=("Courier",9))
        self.vae_log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = tk.Scrollbar(log_frame, command=self.vae_log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.vae_log_text.config(yscrollcommand=scrollbar.set)

    # ------------------------------------------------------------------
    # Latent DDPM Training Tab
    # ------------------------------------------------------------------
    def setup_latent_tab(self):
        main_frame = tk.Frame(self.latent_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_frame = tk.Frame(main_frame, width=300)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0,10))
        left_frame.pack_propagate(False)

        tk.Label(left_frame, text="Latent DDPM Training", font=("Arial",12,"bold")).pack(pady=(0,10))
        tk.Label(left_frame, text="Requires a trained VAE", fg="blue").pack(pady=5)

        # Conditional training toggle
        cond_frame = tk.LabelFrame(left_frame, text="Conditional (text)", padx=5, pady=5)
        cond_frame.pack(fill=tk.X, pady=5)
        self.cond_enabled_cb = tk.Checkbutton(cond_frame, text="Enable text conditioning",
                                              variable=self.settings['cond_enabled'],
                                              command=self.toggle_cond_settings)
        self.cond_enabled_cb.pack(anchor='w')
        tk.Button(cond_frame, text="Load CSV (image,label)", command=self.load_csv, width=20).pack(pady=2)
        tk.Button(cond_frame, text="Use filenames as labels", command=self.use_filenames_as_labels, width=20).pack(pady=2)
        tk.Button(cond_frame, text="Use folder names as labels", command=self.use_folders_as_labels, width=20).pack(pady=2)
        self.csv_status = tk.Label(cond_frame, text="No CSV loaded", fg="red")
        self.csv_status.pack()

        cfg_frame = tk.LabelFrame(cond_frame, text="CFG Training", padx=5, pady=5)
        cfg_frame.pack(fill=tk.X, pady=5)
        f = tk.Frame(cfg_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="CFG dropout prob:", width=18, anchor='w').pack(side=tk.LEFT)
        cfg_dropout_scale = tk.Scale(f, from_=0.0, to=0.5, resolution=0.01,
                                     variable=self.settings['cfg_dropout_prob'],
                                     orient=tk.HORIZONTAL, length=150)
        cfg_dropout_scale.pack(side=tk.RIGHT)
        tk.Label(f, textvariable=self.settings['cfg_dropout_prob'], width=5).pack(side=tk.RIGHT)

        tk.Button(left_frame, text="Initialize Latent DDPM", command=self.initialize_latent_model, width=20).pack(pady=5)
        epoch_frame = tk.Frame(left_frame)
        epoch_frame.pack(pady=5)
        tk.Label(epoch_frame, text="Epochs:").pack(side=tk.LEFT)
        self.latent_epoch_var = tk.StringVar(value="200")
        tk.Entry(epoch_frame, textvariable=self.latent_epoch_var, width=8).pack(side=tk.LEFT, padx=5)

        tk.Button(left_frame, text="Start Latent Training", command=self.start_latent_training,
                  width=20, bg="lightgreen").pack(pady=5)
        tk.Button(left_frame, text="Stop Latent Training", command=self.stop_latent_training,
                  width=20, bg="salmon").pack(pady=5)
        tk.Button(left_frame, text="Save Latent Model", command=self.save_latent_model, width=20).pack(pady=5)
        tk.Button(left_frame, text="Load Latent Model", command=self.load_latent_model, width=20).pack(pady=5)

        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        preview_canvas_frame = tk.LabelFrame(right_frame, text="Generated Samples (decoded)", padx=5, pady=5)
        preview_canvas_frame.pack(fill=tk.BOTH, expand=True, pady=(0,5))
        self.latent_preview_canvas = tk.Canvas(preview_canvas_frame, bg='gray', width=256, height=256)
        self.latent_preview_canvas.pack()

        prompt_frame = tk.LabelFrame(right_frame, text="Test prompt (leave empty for unconditional)", padx=5, pady=5)
        prompt_frame.pack(fill=tk.X, pady=(0,5))
        self.test_prompt_entry = tk.Entry(prompt_frame)
        self.test_prompt_entry.insert(0, "a cute cat")
        self.test_prompt_entry.pack(fill=tk.X, pady=2)
        tk.Button(prompt_frame, text="Generate Preview", command=self.latent_preview_with_prompt, width=20).pack(pady=2)

        log_frame = tk.LabelFrame(right_frame, text="Log", padx=5, pady=5)
        log_frame.pack(fill=tk.BOTH, expand=True)
        self.latent_log_text = tk.Text(log_frame, height=15, font=("Courier",9))
        self.latent_log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = tk.Scrollbar(log_frame, command=self.latent_log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.latent_log_text.config(yscrollcommand=scrollbar.set)

    def toggle_cond_settings(self):
        enabled = self.settings['cond_enabled'].get()
        self.csv_status.config(state=tk.NORMAL if enabled else tk.DISABLED)

    # ------------------------------------------------------------------
    # Settings Tab
    # ------------------------------------------------------------------
    def setup_settings_tab(self):
        main_frame = tk.Frame(self.settings_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        tk.Label(main_frame, text="Model Settings", font=("Arial",14,"bold")).pack(pady=(0,20))

        canvas = tk.Canvas(main_frame)
        scrollbar = tk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0,0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # ----- Image Settings -----
        img_frame = tk.LabelFrame(scrollable_frame, text="Image", padx=10, pady=10)
        img_frame.pack(fill=tk.X, pady=5)

        f = tk.Frame(img_frame)
        f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Color mode:", width=20, anchor='w').pack(side=tk.LEFT)
        om = ttk.Combobox(f, textvariable=self.settings['color_mode'], values=['rgb', 'grayscale'], state='readonly', width=10)
        om.pack(side=tk.RIGHT)

        f = tk.Frame(img_frame)
        f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Image size:", width=20, anchor='w').pack(side=tk.LEFT)
        spin = ttk.Spinbox(f, from_=16, to=128, textvariable=self.settings['img_size'], width=8)
        spin.pack(side=tk.RIGHT)

        # ----- VAE Settings -----
        vae_frame = tk.LabelFrame(scrollable_frame, text="VAE", padx=10, pady=10)
        vae_frame.pack(fill=tk.X, pady=5)

        f_size = tk.Frame(vae_frame)
        f_size.pack(fill=tk.X, pady=2)
        tk.Label(f_size, text="VAE size:", width=20, anchor='w').pack(side=tk.LEFT)
        size_combo = ttk.Combobox(f_size, textvariable=self.settings['vae_size'],
                                  values=['tiny', 'small', 'medium', 'big', 'large'],
                                  state='readonly', width=10)
        size_combo.pack(side=tk.RIGHT)

        vae_params = [
            ("Base channels:", 'vae_base_channels', 16, 128, int),
            ("Latent channels:", 'vae_latent_channels', 1, 32, int),
            ("Latent height:", 'vae_latent_h', 1, 32, int),
            ("Latent width:", 'vae_latent_w', 1, 32, int),
            ("Batch size:", 'vae_batch_size', 1, 32, int),
            ("Learning rate:", 'vae_lr', 1e-5, 1e-2, float),
            ("KL weight:", 'vae_kl_weight', 1e-6, 1e-2, float),
            ("Workers:", 'vae_num_workers', 0, 4, int),
        ]
        for label, key, low, high, typ in vae_params:
            f = tk.Frame(vae_frame)
            f.pack(fill=tk.X, pady=2)
            tk.Label(f, text=label, width=20, anchor='w').pack(side=tk.LEFT)
            if typ == int:
                spin = ttk.Spinbox(f, from_=low, to=high, textvariable=self.settings[key], width=8)
            else:
                spin = ttk.Entry(f, textvariable=self.settings[key], width=8)
            spin.pack(side=tk.RIGHT)

        # ----- Latent DDPM Settings -----
        latent_frame = tk.LabelFrame(scrollable_frame, text="Latent DDPM", padx=10, pady=10)
        latent_frame.pack(fill=tk.X, pady=5)

        latent_params = [
            ("UNet base channels:", 'latent_base_channels', 32, 128, int),
            ("Timesteps (T):", 'latent_T', 50, 1000, int),
            ("Beta start:", 'latent_beta_start', 1e-5, 1e-2, float),
            ("Beta end:", 'latent_beta_end', 0.001, 0.1, float),
            ("Batch size:", 'latent_batch_size', 1, 32, int),
            ("Learning rate:", 'latent_lr', 1e-5, 1e-2, float),
        ]
        for label, key, low, high, typ in latent_params:
            f = tk.Frame(latent_frame)
            f.pack(fill=tk.X, pady=2)
            tk.Label(f, text=label, width=20, anchor='w').pack(side=tk.LEFT)
            if typ == int:
                spin = ttk.Spinbox(f, from_=low, to=high, textvariable=self.settings[key], width=8)
            else:
                spin = ttk.Entry(f, textvariable=self.settings[key], width=8)
            spin.pack(side=tk.RIGHT)

        sched_frame = tk.Frame(latent_frame)
        sched_frame.pack(fill=tk.X, pady=2)
        tk.Label(sched_frame, text="Beta schedule:", width=20, anchor='w').pack(side=tk.LEFT)
        sched_combo = ttk.Combobox(sched_frame, textvariable=self.settings['beta_schedule'],
                                   values=['linear', 'cosine'], state='readonly', width=10)
        sched_combo.pack(side=tk.RIGHT)

        # ----- Text Conditioning Settings -----
        cond_frame = tk.LabelFrame(scrollable_frame, text="Text Conditioning", padx=10, pady=10)
        cond_frame.pack(fill=tk.X, pady=5)

        # Encoder type and size preset
        f = tk.Frame(cond_frame)
        f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Encoder type:", width=20, anchor='w').pack(side=tk.LEFT)
        type_combo = ttk.Combobox(f, textvariable=self.settings['text_encoder_type'],
                                  values=['BiGRU', 'BiTransformer'], state='readonly', width=12)
        type_combo.pack(side=tk.RIGHT)

        f = tk.Frame(cond_frame)
        f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Encoder size preset:", width=20, anchor='w').pack(side=tk.LEFT)
        size_combo2 = ttk.Combobox(f, textvariable=self.settings['text_encoder_size'],
                                   values=['tiny', 'small', 'medium', 'large'], state='readonly', width=8)
        size_combo2.pack(side=tk.RIGHT)
        tk.Button(f, text="Apply preset to dims", command=self.apply_text_encoder_preset, width=18).pack(side=tk.RIGHT, padx=5)

        cond_params = [
            ("Embed dim:", 'cond_embed_dim', 32, 256, int),
            ("Hidden size:", 'cond_hidden_size', 32, 512, int),
            ("Num layers:", 'cond_num_layers', 1, 4, int),
            ("Num heads:", 'cond_num_heads', 1, 16, int),
            ("FF dim:", 'cond_ff_dim', 64, 1024, int),
            ("Conditioning dim:", 'cond_dim', 64, 512, int),
            ("Max text len:", 'cond_text_max_len', 32, 256, int),
            ("Learning rate:", 'cond_lr', 1e-5, 1e-2, float),
        ]
        for label, key, low, high, typ in cond_params:
            f = tk.Frame(cond_frame)
            f.pack(fill=tk.X, pady=2)
            tk.Label(f, text=label, width=20, anchor='w').pack(side=tk.LEFT)
            if typ == int:
                spin = ttk.Spinbox(f, from_=low, to=high, textvariable=self.settings[key], width=8)
            else:
                spin = ttk.Entry(f, textvariable=self.settings[key], width=8)
            spin.pack(side=tk.RIGHT)

        # Augmentations
        aug_frame = tk.LabelFrame(scrollable_frame, text="Augmentations", padx=10, pady=10)
        aug_frame.pack(fill=tk.X, pady=5)

        augs = [
            ("Horizontal Flip", 'flip_horizontal'),
            ("Rotation (±30°)", 'rotation'),
        ]
        for label, key in augs:
            cb = tk.Checkbutton(aug_frame, text=label, variable=self.aug_settings[key])
            cb.pack(anchor='w')

        # System info
        sys_frame = tk.LabelFrame(scrollable_frame, text="System", padx=10, pady=10)
        sys_frame.pack(fill=tk.X, pady=5)
        cpu = multiprocessing.cpu_count()
        tk.Label(sys_frame, text=f"CPU cores: {cpu}").pack(anchor='w')
        tk.Label(sys_frame, text=f"PyTorch threads: {torch.get_num_threads()}").pack(anchor='w')
        tk.Label(sys_frame, text=f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}").pack(anchor='w')

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def apply_text_encoder_preset(self):
        enc_type = self.settings['text_encoder_type'].get()
        size = self.settings['text_encoder_size'].get()
        config = get_encoder_config(enc_type, size)
        # Update the appropriate settings variables
        self.settings['cond_embed_dim'].set(config.get('embed_dim', self.settings['cond_embed_dim'].get()))
        if enc_type == 'BiGRU':
            self.settings['cond_hidden_size'].set(config.get('hidden_size', self.settings['cond_hidden_size'].get()))
            self.settings['cond_num_layers'].set(config.get('num_layers', self.settings['cond_num_layers'].get()))
        elif enc_type == 'BiTransformer':
            self.settings['cond_num_heads'].set(config.get('num_heads', self.settings['cond_num_heads'].get()))
            self.settings['cond_num_layers'].set(config.get('num_layers', self.settings['cond_num_layers'].get()))
            self.settings['cond_ff_dim'].set(config.get('ff_dim', self.settings['cond_ff_dim'].get()))
        self.settings['cond_dim'].set(config.get('cond_dim', self.settings['cond_dim'].get()))
        self.log_latent(f"Applied {enc_type} {size} preset dims.")

    # ------------------------------------------------------------------
    # Generation Tab
    # ------------------------------------------------------------------
    def setup_generation_tab(self):
        main_frame = tk.Frame(self.generation_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        tk.Label(main_frame, text="Generate Images (Latent DDPM + VAE Decoder)", font=("Arial",14,"bold")).pack(pady=(0,10))

        prompt_frame = tk.LabelFrame(main_frame, text="Text Prompt (leave empty for unconditional)", padx=5, pady=5)
        prompt_frame.pack(fill=tk.X, pady=(0,10))
        self.gen_prompt = tk.Entry(prompt_frame, width=60)
        self.gen_prompt.insert(0, "a cute cat")
        self.gen_prompt.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        ctrl_frame = tk.Frame(main_frame)
        ctrl_frame.pack(pady=5)

        tk.Label(ctrl_frame, text="Number:").pack(side=tk.LEFT)
        self.gen_count = tk.IntVar(value=16)
        tk.Spinbox(ctrl_frame, from_=1, to=64, textvariable=self.gen_count, width=5).pack(side=tk.LEFT, padx=5)

        tk.Label(ctrl_frame, text="Sampling steps:").pack(side=tk.LEFT, padx=(10,0))
        self.sampling_steps = tk.IntVar(value=self.settings['latent_T'].get())
        tk.Spinbox(ctrl_frame, from_=1, to=self.settings['latent_T'].get(), textvariable=self.sampling_steps, width=5).pack(side=tk.LEFT, padx=5)

        tk.Label(ctrl_frame, text="Sampler:").pack(side=tk.LEFT, padx=(10,0))
        sampler_combo = ttk.Combobox(ctrl_frame, textvariable=self.sampler_method,
                                      values=['ddpm', 'ddim'], state='readonly', width=6)
        sampler_combo.pack(side=tk.LEFT, padx=5)

        tk.Label(ctrl_frame, text="Temp:").pack(side=tk.LEFT, padx=(10,0))
        tk.Spinbox(ctrl_frame, from_=0.5, to=2.0, increment=0.1, textvariable=self.gen_temperature, width=5).pack(side=tk.LEFT, padx=5)

        tk.Label(ctrl_frame, text="CFG scale:").pack(side=tk.LEFT, padx=(10,0))
        self.cfg_entry = tk.Entry(ctrl_frame, width=6, textvariable=self.cfg_scale)
        self.cfg_entry.pack(side=tk.LEFT, padx=5)

        self.progressive_grid = tk.BooleanVar(value=False)
        prog_check = tk.Checkbutton(ctrl_frame, text="Progressive grid", variable=self.progressive_grid)
        prog_check.pack(side=tk.LEFT, padx=10)

        tk.Label(ctrl_frame, text="Update interval:").pack(side=tk.LEFT)
        self.prog_interval = tk.IntVar(value=10)
        tk.Spinbox(ctrl_frame, from_=1, to=50, textvariable=self.prog_interval, width=5).pack(side=tk.LEFT, padx=5)

        self.generate_btn = tk.Button(ctrl_frame, text="Generate", command=self.generate_samples, bg="lightgreen")
        self.generate_btn.pack(side=tk.LEFT, padx=5)
        self.stop_prog_btn = tk.Button(ctrl_frame, text="Stop", command=self.stop_progressive, state=tk.DISABLED, bg="salmon")
        self.stop_prog_btn.pack(side=tk.LEFT, padx=5)

        scroll_frame = tk.LabelFrame(main_frame, text="Results", padx=5, pady=5)
        scroll_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        canvas_frame = tk.Frame(scroll_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.gen_canvas = tk.Canvas(canvas_frame, bg='lightgray')
        self.v_scroll = tk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.gen_canvas.yview)
        self.h_scroll = tk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.gen_canvas.xview)
        self.gen_canvas.configure(yscrollcommand=self.v_scroll.set, xscrollcommand=self.h_scroll.set)

        self.v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.gen_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.inner_frame = tk.Frame(self.gen_canvas)
        self.canvas_window = self.gen_canvas.create_window((0,0), window=self.inner_frame, anchor='nw')
        self.inner_frame.bind('<Configure>', lambda e: self.gen_canvas.configure(scrollregion=self.gen_canvas.bbox('all')))

        self.gen_info = tk.Label(main_frame, text="", fg="blue")
        self.gen_info.pack()

    # ------------------------------------------------------------------
    # Core functions
    # ------------------------------------------------------------------
    def log_vae(self, msg):
        self.message_queue_vae.put(msg)

    def log_latent(self, msg):
        self.message_queue_latent.put(msg)

    def process_messages_vae(self):
        try:
            while True:
                msg = self.message_queue_vae.get_nowait()
                self.vae_log_text.insert(tk.END, f"{time.strftime('%H:%M:%S')} - {msg}\n")
                self.vae_log_text.see(tk.END)
                self.status_label.config(text=msg[:50])
                print(f"VAE: {msg}")
        except queue.Empty:
            pass
        self.root.after(100, self.process_messages_vae)

    def process_messages_latent(self):
        try:
            while True:
                msg = self.message_queue_latent.get_nowait()
                self.latent_log_text.insert(tk.END, f"{time.strftime('%H:%M:%S')} - {msg}\n")
                self.latent_log_text.see(tk.END)
                self.status_label.config(text=msg[:50])
                print(f"Latent: {msg}")
        except queue.Empty:
            pass
        self.root.after(100, self.process_messages_latent)

    def add_images(self):
        files = filedialog.askopenfilenames(filetypes=[("Images", "*.jpg *.jpeg *.png *.jfif *.webp *.bmp")])
        for f in files:
            if f not in self.image_paths:
                self.image_paths.append(f)
                self.image_listbox.insert(tk.END, os.path.basename(f))
        self.log_vae(f"Added {len(files)} images. Total: {len(self.image_paths)}")
        self.log_latent(f"Added {len(files)} images. Total: {len(self.image_paths)}")

    def add_folder(self):
        folder = filedialog.askdirectory()
        if not folder:
            return
        count = 0
        for root_dir, _, files in os.walk(folder):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.jfif', '.webp', '.bmp')):
                    full_path = os.path.join(root_dir, file)
                    if full_path not in self.image_paths:
                        self.image_paths.append(full_path)
                        self.image_listbox.insert(tk.END, os.path.basename(full_path))
                        count += 1
        self.log_vae(f"Added {count} images from folder (recursive). Total: {len(self.image_paths)}")
        self.log_latent(f"Added {count} images from folder. Total: {len(self.image_paths)}")

    def clear_images(self):
        self.image_paths = []
        self.labels = []
        self.image_listbox.delete(0, tk.END)
        self.csv_status.config(text="No CSV loaded", fg="red")
        self.log_vae("Cleared all images")
        self.log_latent("Cleared all images")

    # ----------------------------- CSV loading with robust matching & multi-captions -----------------------------
    def _match_csv_row(self, row_name: str) -> str:
        """Return the best matching full path from self.image_paths, or None."""
        # Try ends-with (case-insensitive) to handle relative paths like 'subfolder/image.png'
        norm = row_name.replace('\\', '/').lower()
        for full in self.image_paths:
            if full.replace('\\', '/').lower().endswith(norm):
                return full
        # Fallback: match only basename
        base = os.path.basename(row_name).lower()
        for full in self.image_paths:
            if os.path.basename(full).lower() == base:
                return full
        return None

    def load_csv(self):
        fname = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if not fname:
            return
        self.csv_path = fname
    
        # Build a fast lookup: basename -> full path
        path_by_basename = {}
        for full in self.image_paths:
            base = os.path.basename(full).lower()
            if base in path_by_basename:
                self.log_cond_seq(f"Warning: duplicate basename '{base}', CSV match may be ambiguous")
            else:
                path_by_basename[base] = full
    
        label_map = {}   # full_path -> list of labels
        try:
            # Fast reading – assume comma delimiter (most CSVs use it)
            with open(fname, 'r', encoding='utf-8-sig') as f:
                # Detect dialect only if needed (commented out for speed)
                # sample = f.read(1024)
                # f.seek(0)
                # dialect = csv.Sniffer().sniff(sample, delimiters=',\t|;')
                # reader = csv.reader(f, dialect)
                reader = csv.reader(f)   # use default comma
                for row in reader:
                    if not row or all(c.strip() == '' for c in row):
                        continue
                    if len(row) < 2:
                        continue
                    img_name = row[0].strip().strip('"').strip("'")
                    label = row[1].strip().strip('"').strip("'")
                    # Match by basename (fast lookup)
                    base = os.path.basename(img_name).lower()
                    matched = path_by_basename.get(base, None)
                    if matched:
                        label_map.setdefault(matched, []).append(label)
        except Exception as e:
            self.log_latent(f"Error reading CSV: {e}")
            return

        # Assign to labels list
        self.labels = []
        unknown = 0
        for path in self.image_paths:
            if path in label_map:
                self.labels.append(label_map[path])
            else:
                name_no_ext = os.path.splitext(os.path.basename(path))[0]
                self.labels.append([name_no_ext])
                unknown += 1

        self.csv_status.config(text=f"CSV loaded: {len(self.image_paths)-unknown} matched, {unknown} filenames used", fg="green")
        self.log_latent(f"CSV loaded. {len(self.image_paths)-unknown} images matched with labels.")

    def use_filenames_as_labels(self):
        self.labels = [[os.path.splitext(os.path.basename(p))[0]] for p in self.image_paths]
        self.csv_status.config(text="Using filenames as labels (single caption per image)", fg="blue")
        self.log_latent("Using filenames as labels.")

    def use_folders_as_labels(self):
        self.labels = []
        for p in self.image_paths:
            folder = os.path.basename(os.path.dirname(p))  # immediate parent folder
            if not folder:
                folder = 'unknown'
            self.labels.append([folder])
        self.csv_status.config(text="Using folder names as labels (single caption per image)", fg="blue")
        self.log_latent("Using folder names as labels.")

    # ========== VAE Methods ==========
    def initialize_vae(self):
        try:
            in_channels = 3 if self.settings['color_mode'].get() == 'rgb' else 1
            self.vae_model = FlexVAE(
                in_channels=in_channels,
                base_channels=self.settings['vae_base_channels'].get(),
                latent_channels=self.settings['vae_latent_channels'].get(),
                latent_h=self.settings['vae_latent_h'].get(),
                latent_w=self.settings['vae_latent_w'].get(),
                size=self.settings['vae_size'].get()
            )
            self.vae_model.to('cpu')
            self.vae_optimizer = optim.Adam(self.vae_model.parameters(), lr=self.settings['vae_lr'].get())
            self.log_vae(f"VAE initialized with size '{self.settings['vae_size'].get()}'.")
        except Exception as e:
            self.log_vae(f"Error initializing VAE: {e}")

    def start_vae_training(self):
        if not self.image_paths:
            self.log_vae("No images!")
            return
        if not self.vae_model:
            self.log_vae("VAE not initialized!")
            return
        if self.training_vae:
            self.log_vae("Already training VAE.")
            return

        try:
            epochs = int(self.vae_epoch_var.get())
        except:
            self.log_vae("Invalid epochs")
            return

        self.training_vae = True
        self.current_epoch_vae = 0
        self.vae_start_time = time.time()

        thread = threading.Thread(target=self.train_vae_loop, args=(epochs,), daemon=True)
        thread.start()
        self.log_vae(f"VAE training started for {epochs} epochs.")

    def train_vae_loop(self, epochs):
        try:
            batch_size = self.settings['vae_batch_size'].get()
            num_workers = self.settings['vae_num_workers'].get()
            img_size = self.settings['img_size'].get()
            kl_weight = self.settings['vae_kl_weight'].get()
            color_mode = self.settings['color_mode'].get()

            aug_dict = {k: v.get() for k, v in self.aug_settings.items()}
            # VAE uses labels only semantically (they are ignored, but dataset expects them)
            dataset = ConditionalImageDataset(self.image_paths, self.labels if self.labels else [['']]*len(self.image_paths),
                                              img_size, color_mode, aug_dict, self.settings['cond_text_max_len'].get())
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                num_workers=num_workers, pin_memory=False,
                                persistent_workers=False if num_workers==0 else True)

            for epoch in range(epochs):
                if not self.training_vae:
                    break
                self.current_epoch_vae = epoch
                epoch_loss = 0.0
                batches = 0

                for images, _ in loader:
                    if not self.training_vae:
                        break
                    images = images.to('cpu')
                    recon, mu, logvar = self.vae_model(images)
                    recon_loss = nn.functional.mse_loss(recon, images)
                    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                    kl_loss = kl_loss / images.size(0)
                    loss = recon_loss + kl_weight * kl_loss

                    self.vae_optimizer.zero_grad()
                    loss.backward()
                    self.vae_optimizer.step()

                    epoch_loss += loss.item()
                    batches += 1

                avg_loss = epoch_loss / batches if batches else 0
                elapsed = time.time() - self.vae_start_time
                self.log_vae(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f} | Time: {elapsed:.1f}s")

                if (epoch+1) % 5 == 0:
                    self.show_vae_preview()

            self.training_vae = False
            self.log_vae("VAE training finished.")
        except Exception as e:
            self.log_vae(f"VAE training error: {e}")
            import traceback
            traceback.print_exc()
            self.training_vae = False

    def show_vae_preview(self):
        if not self.vae_model:
            return
        try:
            total = len(self.image_paths)
            if total == 0:
                self.log_vae("No images for preview")
                return

            n_preview = min(16, total)
            indices = random.sample(range(total), n_preview)

            preview_paths = [self.image_paths[i] for i in indices]
            preview_labels = [self.labels[i] if self.labels and len(self.labels)==total else [''] for i in indices]

            color_mode = self.settings['color_mode'].get()
            dataset = ConditionalImageDataset(
                preview_paths,
                preview_labels,
                img_size=self.settings['img_size'].get(),
                color_mode=color_mode
            )
            loader = DataLoader(dataset, batch_size=n_preview, shuffle=False)
            images, _ = next(iter(loader))

            with torch.no_grad():
                recon, _, _ = self.vae_model(images)

            images = (images + 1) / 2
            recon = (recon + 1) / 2
            images = images.clamp(0, 1).cpu().numpy()
            recon = recon.clamp(0, 1).cpu().numpy()

            thumb = self.thumbnail_size // 2
            grid = Image.new('RGB', (8 * thumb, 4 * thumb))

            for i in range(n_preview):
                row = i // 4
                col = i % 4
                if images[i].shape[0] == 1:
                    img_arr = images[i][0] * 255
                    img_arr = np.stack([img_arr, img_arr, img_arr], axis=-1).astype(np.uint8)
                else:
                    img_arr = images[i].transpose(1, 2, 0) * 255
                    img_arr = img_arr.astype(np.uint8)
                pil_img = Image.fromarray(img_arr).resize((thumb, thumb), Image.NEAREST)
                grid.paste(pil_img, (col * thumb * 2, row * thumb))

                if recon[i].shape[0] == 1:
                    rec_arr = recon[i][0] * 255
                    rec_arr = np.stack([rec_arr, rec_arr, rec_arr], axis=-1).astype(np.uint8)
                else:
                    rec_arr = recon[i].transpose(1, 2, 0) * 255
                    rec_arr = rec_arr.astype(np.uint8)
                pil_rec = Image.fromarray(rec_arr).resize((thumb, thumb), Image.NEAREST)
                grid.paste(pil_rec, (col * thumb * 2 + thumb, row * thumb))

            grid = grid.resize((256, 128), Image.NEAREST)
            self.vae_preview_photo = ImageTk.PhotoImage(grid)
            self.vae_preview_canvas.delete("all")
            self.vae_preview_canvas.create_image(128, 64, image=self.vae_preview_photo)
        except Exception as e:
            self.log_vae(f"Preview error: {e}")

    def stop_vae_training(self):
        self.training_vae = False
        self.log_vae("VAE training stopped.")

    def save_vae(self):
        if not self.vae_model:
            self.log_vae("No VAE model.")
            return
        fname = filedialog.asksaveasfilename(defaultextension=".pth",
                                              filetypes=[("PyTorch","*.pth")])
        if fname:
            torch.save({
                'model_state': self.vae_model.state_dict(),
                'optimizer_state': self.vae_optimizer.state_dict(),
                'settings': {k:v.get() for k,v in self.settings.items() if k.startswith('vae') or k in ['img_size','color_mode']},
                'vae_size': self.settings['vae_size'].get(),
            }, fname)
            self.log_vae(f"VAE saved to {fname}")

    def load_vae(self):
        fname = filedialog.askopenfilename(filetypes=[("PyTorch","*.pth")])
        if not fname:
            return
        try:
            ckpt = torch.load(fname, map_location='cpu')
            if 'settings' in ckpt:
                s = ckpt['settings']
                for k, v in s.items():
                    if k in self.settings:
                        self.settings[k].set(v)
            if 'vae_size' in ckpt:
                self.settings['vae_size'].set(ckpt['vae_size'])
            self.initialize_vae()
            self.vae_model.load_state_dict(ckpt['model_state'])
            self.vae_optimizer.load_state_dict(ckpt['optimizer_state'])
            self.log_vae(f"VAE loaded from {fname}")
        except Exception as e:
            self.log_vae(f"Load error: {e}")

    # ========== Latent DDPM Methods ==========
    def initialize_latent_model(self):
        if not self.vae_model:
            self.log_latent("Please train/load a VAE first!")
            return
        try:
            cond_enabled = self.settings['cond_enabled'].get()
            cond_dim = self.settings['cond_dim'].get() if cond_enabled else 1

            self.latent_model = LatentDDPM(
                latent_channels=self.settings['vae_latent_channels'].get(),
                latent_h=self.settings['vae_latent_h'].get(),
                latent_w=self.settings['vae_latent_w'].get(),
                base_channels=self.settings['latent_base_channels'].get(),
                cond_dim=cond_dim,
                T=self.settings['latent_T'].get(),
                beta_start=self.settings['latent_beta_start'].get(),
                beta_end=self.settings['latent_beta_end'].get(),
                schedule=self.settings['beta_schedule'].get()
            )
            for pg in self.latent_model.optimizer.param_groups:
                pg['lr'] = self.settings['latent_lr'].get()

            if cond_enabled:
                enc_type = self.settings['text_encoder_type'].get()
                if enc_type == 'BiGRU':
                    self.text_encoder = TextEncoder(
                        vocab_size=256,
                        embed_dim=self.settings['cond_embed_dim'].get(),
                        hidden_size=self.settings['cond_hidden_size'].get(),
                        num_layers=self.settings['cond_num_layers'].get(),
                        cond_dim=self.settings['cond_dim'].get()
                    )
                else:  # BiTransformer
                    self.text_encoder = TransformerTextEncoder(
                        vocab_size=256,
                        embed_dim=self.settings['cond_embed_dim'].get(),
                        num_heads=self.settings['cond_num_heads'].get(),
                        num_layers=self.settings['cond_num_layers'].get(),
                        ff_dim=self.settings['cond_ff_dim'].get(),
                        cond_dim=self.settings['cond_dim'].get(),
                        max_len=self.settings['cond_text_max_len'].get()
                    )
                self.text_encoder.to(self.latent_model.device)
                self.latent_optimizer = optim.Adam(
                    list(self.latent_model.model.parameters()) + list(self.text_encoder.parameters()),
                    lr=self.settings['latent_lr'].get()
                )
                self.latent_model.optimizer = self.latent_optimizer
                self.log_latent(f"Conditional Latent DDPM with {enc_type} encoder initialized.")
            else:
                self.text_encoder = None
                self.log_latent("Unconditional Latent DDPM initialized.")
        except Exception as e:
            self.log_latent(f"Error initializing latent model: {e}")

    def start_latent_training(self):
        if not self.image_paths:
            self.log_latent("No images!")
            return
        if not self.latent_model:
            self.log_latent("Latent model not initialized!")
            return
        if not self.vae_model:
            self.log_latent("No VAE available to encode latents!")
            return
        if self.training_latent:
            self.log_latent("Already training latent DDPM.")
            return

        cond_enabled = self.settings['cond_enabled'].get()
        if cond_enabled and not self.labels:
            self.log_latent("Conditional training requires labels. Load CSV or use filenames/folders as labels.")
            return

        try:
            epochs = int(self.latent_epoch_var.get())
        except:
            self.log_latent("Invalid epochs")
            return

        self.training_latent = True
        self.current_epoch_latent = 0
        self.latent_start_time = time.time()

        thread = threading.Thread(target=self.train_latent_loop, args=(epochs,), daemon=True)
        thread.start()
        self.log_latent(f"Latent DDPM training started for {epochs} epochs.")

    def train_latent_loop(self, epochs):
        try:
            batch_size = self.settings['latent_batch_size'].get()
            num_workers = self.settings['vae_num_workers'].get()
            img_size = self.settings['img_size'].get()
            color_mode = self.settings['color_mode'].get()
            cond_enabled = self.settings['cond_enabled'].get()
            text_max_len = self.settings['cond_text_max_len'].get()
            cfg_dropout = self.settings['cfg_dropout_prob'].get()

            aug_dict = {k: v.get() for k, v in self.aug_settings.items()}
            # If unconditional, pass empty list of single empty strings
            labels_for_dataset = self.labels if self.labels else [['']]*len(self.image_paths)
            dataset = ConditionalImageDataset(self.image_paths, labels_for_dataset,
                                              img_size, color_mode, aug_dict, text_max_len)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                num_workers=num_workers, pin_memory=False,
                                persistent_workers=False if num_workers==0 else True)

            for epoch in range(epochs):
                if not self.training_latent:
                    break
                self.current_epoch_latent = epoch
                epoch_loss = 0.0
                batches = 0

                for images, text_tensors in loader:
                    if not self.training_latent:
                        break
                    with torch.no_grad():
                        latents = self.vae_model.encode(images)

                    if cond_enabled and self.text_encoder is not None:
                        cond = self.text_encoder(text_tensors)
                    else:
                        cond = None

                    loss = self.latent_model.train_step(latents, cond=cond, cfg_dropout_prob=cfg_dropout)
                    epoch_loss += loss
                    batches += 1

                avg_loss = epoch_loss / batches if batches else 0
                elapsed = time.time() - self.latent_start_time
                self.log_latent(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f} | Time: {elapsed:.1f}s")

                if (epoch+1) % 5 == 0:
                    self.latent_preview_with_prompt()

            self.training_latent = False
            self.log_latent("Latent DDPM training finished.")
        except Exception as e:
            self.log_latent(f"Latent training error: {e}")
            import traceback
            traceback.print_exc()
            self.training_latent = False

    def latent_preview_with_prompt(self):
        if not self.latent_model or not self.vae_model:
            self.log_latent("Models not loaded for preview")
            return
        try:
            prompt = self.test_prompt_entry.get().strip()
            unconditional = (prompt == "")
            cond = None
            cfg_scale = 1.0 if unconditional else self.cfg_scale.get()

            if not unconditional and self.settings['cond_enabled'].get() and self.text_encoder is not None:
                text_indices = text_to_indices(prompt, self.settings['cond_text_max_len'].get())
                text_tensor = torch.tensor([text_indices] * 16, dtype=torch.long, device=self.latent_model.device)
                with torch.no_grad():
                    cond = self.text_encoder(text_tensor)
                self.log_latent(f"Preview: conditional with prompt '{prompt}'")
            elif not unconditional:
                self.log_latent("Preview: conditioning disabled or text encoder missing, generating unconditional")
                unconditional = True
            else:
                self.log_latent("Preview: unconditional generation (empty prompt)")

            z = self.latent_model.sample(n_samples=16, steps=50, method='ddim', cond=cond, cfg_scale=cfg_scale)
            with torch.no_grad():
                samples = self.vae_model.decode(z, target_size=(self.settings['img_size'].get(), self.settings['img_size'].get()))
            samples = (samples + 1) / 2
            samples = samples.clamp(0,1).cpu().numpy()
            thumb = self.thumbnail_size
            grid = Image.new('RGB', (4*thumb, 4*thumb))
            for i in range(4):
                for j in range(4):
                    idx = i*4 + j
                    if idx < len(samples):
                        if samples[idx].shape[0] == 1:
                            img = samples[idx][0] * 255
                            img = np.stack([img, img, img], axis=-1).astype(np.uint8)
                        else:
                            img = samples[idx].transpose(1,2,0) * 255
                            img = img.astype(np.uint8)
                        pil_img = Image.fromarray(img).resize((thumb, thumb), Image.NEAREST)
                        grid.paste(pil_img, (j*thumb, i*thumb))
            grid = grid.resize((256,256), Image.NEAREST)
            self.latent_preview_photo = ImageTk.PhotoImage(grid)
            self.latent_preview_canvas.delete("all")
            self.latent_preview_canvas.create_image(128,128, image=self.latent_preview_photo)
        except Exception as e:
            self.log_latent(f"Preview error: {e}")

    def stop_latent_training(self):
        self.training_latent = False
        self.log_latent("Latent DDPM training stopped.")

    def save_latent_model(self):
        if not self.latent_model:
            self.log_latent("No latent model.")
            return
        fname = filedialog.asksaveasfilename(defaultextension=".pth",
                                              filetypes=[("PyTorch","*.pth")])
        if fname:
            save_dict = {
                'model_state': self.latent_model.model.state_dict(),
                'optimizer_state': self.latent_model.optimizer.state_dict(),
                'settings': {k:v.get() for k,v in self.settings.items() if k.startswith('latent') or k in ['beta_schedule','cfg_dropout_prob','text_encoder_type','text_encoder_size']},
                'T': self.latent_model.T,
                'betas': self.latent_model.betas.cpu(),
                'alphas': self.latent_model.alphas.cpu(),
                'alpha_bars': self.latent_model.alpha_bars.cpu()
            }
            if self.text_encoder is not None:
                save_dict['text_encoder_state'] = self.text_encoder.state_dict()
            torch.save(save_dict, fname)
            self.log_latent(f"Latent model saved to {fname}")

    def load_latent_model(self):
        fname = filedialog.askopenfilename(filetypes=[("PyTorch","*.pth")])
        if not fname:
            return
        try:
            ckpt = torch.load(fname, map_location='cpu')
            if 'settings' in ckpt:
                s = ckpt['settings']
                for k, v in s.items():
                    if k in self.settings:
                        self.settings[k].set(v)

            if not self.vae_model:
                self.log_latent("Please load VAE first, then try again.")
                return
            self.initialize_latent_model()
            self.latent_model.model.load_state_dict(ckpt['model_state'])
            self.latent_model.optimizer.load_state_dict(ckpt['optimizer_state'])
            self.latent_model.betas = ckpt['betas'].to(self.latent_model.device)
            self.latent_model.alphas = ckpt['alphas'].to(self.latent_model.device)
            self.latent_model.alpha_bars = ckpt['alpha_bars'].to(self.latent_model.device)
            self.latent_model.T = ckpt['T']
            if 'text_encoder_state' in ckpt and self.text_encoder is not None:
                self.text_encoder.load_state_dict(ckpt['text_encoder_state'])

            self.log_latent(f"Latent model loaded from {fname}")
        except Exception as e:
            self.log_latent(f"Load error: {e}")

    # ========== Generation Methods ==========
    def generate_samples(self):
        if not self.latent_model or not self.vae_model:
            self.gen_info.config(text="Models not loaded!")
            return
        if self.settings['cond_enabled'].get() and self.text_encoder is None:
            self.gen_info.config(text="Text encoder not initialized! Load conditional model.")
            return

        n = self.gen_count.get()
        steps = self.sampling_steps.get()
        method = self.sampler_method.get()
        prompt = self.gen_prompt.get().strip()
        cfg_scale_user = self.cfg_scale.get()

        unconditional = (prompt == "")
        cond = None
        effective_cfg_scale = 1.0 if unconditional else cfg_scale_user

        if not unconditional and self.settings['cond_enabled'].get() and self.text_encoder is not None:
            text_indices = text_to_indices(prompt, self.settings['cond_text_max_len'].get())
            text_tensor = torch.tensor([text_indices] * n, dtype=torch.long, device=self.latent_model.device)
            with torch.no_grad():
                cond = self.text_encoder(text_tensor)
        elif not unconditional:
            unconditional = True
            effective_cfg_scale = 1.0

        if self.progressive_grid.get():
            self.start_progressive(n, steps, method, cond, effective_cfg_scale)
        else:
            self.generate_btn.config(state=tk.DISABLED)
            if unconditional:
                self.gen_info.config(text="Generating unconditionally...")
            else:
                self.gen_info.config(text=f"Generating with prompt: {prompt}")
            self.root.update()
            thread = threading.Thread(target=self._generate_thread,
                                      args=(n, steps, method, cond, effective_cfg_scale), daemon=True)
            thread.start()

    def stop_progressive(self):
        self.progressive_active = False
        self.stop_prog_btn.config(state=tk.DISABLED)
        self.generate_btn.config(state=tk.NORMAL)
        self.gen_info.config(text="Progressive generation stopped.")

    def start_progressive(self, n, steps, method, cond, cfg_scale):
        self.progressive_active = True
        self.generate_btn.config(state=tk.DISABLED)
        self.stop_prog_btn.config(state=tk.NORMAL)
        self.gen_info.config(text="Progressive generation...")
        thread = threading.Thread(target=self._progressive_thread,
                                  args=(n, steps, method, cond, cfg_scale), daemon=True)
        thread.start()

    def _generate_thread(self, n, steps, method, cond, cfg_scale):
        try:
            z = self.latent_model.sample(n_samples=n, steps=steps, method=method, cond=cond, cfg_scale=cfg_scale)
            target_size = (self.settings['img_size'].get(), self.settings['img_size'].get())
            with torch.no_grad():
                samples = self.vae_model.decode(z, target_size=target_size)
            samples = (samples + 1) / 2
            samples = samples.clamp(0,1).cpu().numpy()
            thumb = self.thumbnail_size
            grid_size = int(math.ceil(math.sqrt(n)))
            total_width = grid_size * thumb
            total_height = grid_size * thumb
            grid_img = Image.new('RGB', (total_width, total_height), color=(128,128,128))
            for i in range(n):
                row = i // grid_size
                col = i % grid_size
                if samples[i].shape[0] == 1:
                    img = samples[i][0] * 255
                    img = np.stack([img, img, img], axis=-1).astype(np.uint8)
                else:
                    img = samples[i].transpose(1,2,0) * 255
                    img = img.astype(np.uint8)
                pil_img = Image.fromarray(img).resize((thumb, thumb), Image.NEAREST)
                grid_img.paste(pil_img, (col*thumb, row*thumb))
            self.root.after(0, lambda: self._display_generated(grid_img))
        except Exception as e:
            self.root.after(0, lambda: self.gen_info.config(text=f"Error: {e}"))
        finally:
            self.root.after(0, lambda: self.generate_btn.config(state=tk.NORMAL))

    def _progressive_thread(self, n, steps, method, cond, cfg_scale):
        try:
            thumb = self.thumbnail_size
            interval = self.prog_interval.get()
            generator = self.latent_model.sample_step_by_step(n_samples=n, steps=steps, method=method, cond=cond, cfg_scale=cfg_scale)
            step_count = 0
            target_size = (self.settings['img_size'].get(), self.settings['img_size'].get())
            for t, z in generator:
                if not self.progressive_active:
                    break
                step_count += 1
                if step_count % interval == 0 or t == 1:
                    with torch.no_grad():
                        samples = self.vae_model.decode(z, target_size=target_size)
                    samples = (samples + 1) / 2
                    samples = samples.clamp(0,1).cpu().numpy()
                    grid_size = int(math.ceil(math.sqrt(n)))
                    total_width = grid_size * thumb
                    total_height = grid_size * thumb
                    grid_img = Image.new('RGB', (total_width, total_height), color=(128,128,128))
                    for i in range(n):
                        row = i // grid_size
                        col = i % grid_size
                        if i < len(samples):
                            if samples[i].shape[0] == 1:
                                img = samples[i][0] * 255
                                img = np.stack([img, img, img], axis=-1).astype(np.uint8)
                            else:
                                img = samples[i].transpose(1,2,0) * 255
                                img = img.astype(np.uint8)
                            pil_img = Image.fromarray(img).resize((thumb, thumb), Image.NEAREST)
                            grid_img.paste(pil_img, (col*thumb, row*thumb))
                    self.root.after(0, lambda g=grid_img, t=t: self._update_progressive(g, t))
                    time.sleep(0.05)
            if not self.progressive_active:
                self.root.after(0, lambda: self.gen_info.config(text="Progressive stopped."))
            else:
                self.root.after(0, lambda: self.gen_info.config(text="Progressive finished."))
        except Exception as e:
            self.root.after(0, lambda: self.gen_info.config(text=f"Error: {e}"))
        finally:
            self.progressive_active = False
            self.root.after(0, lambda: self.generate_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.stop_prog_btn.config(state=tk.DISABLED))

    def _display_generated(self, grid_img):
        for widget in self.inner_frame.winfo_children():
            widget.destroy()
        self.gen_photo = ImageTk.PhotoImage(grid_img)
        label = tk.Label(self.inner_frame, image=self.gen_photo)
        label.image = self.gen_photo
        label.pack()
        self.inner_frame.update_idletasks()
        self.gen_canvas.configure(scrollregion=self.gen_canvas.bbox('all'))
        self.gen_info.config(text="Generation complete.")

    def _update_progressive(self, grid_img, t):
        for widget in self.inner_frame.winfo_children():
            widget.destroy()
        self.prog_photo = ImageTk.PhotoImage(grid_img)
        label = tk.Label(self.inner_frame, image=self.prog_photo)
        label.image = self.prog_photo
        label.pack()
        self.inner_frame.update_idletasks()
        self.gen_canvas.configure(scrollregion=self.gen_canvas.bbox('all'))
        self.gen_info.config(text=f"Step {t}/{self.latent_model.T}")

# ==================== Main ====================

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    root = tk.Tk()
    app = LatentDiffusionApp(root)
    root.mainloop()