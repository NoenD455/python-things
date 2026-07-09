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

# ==================== Helpers ====================

def load_image_as_rgb(path):
    img = Image.open(path)
    if img.mode == 'RGBA':
        bg = Image.new('RGB', img.size, (0, 0, 0))
        bg.paste(img, mask=img.split()[3])
        return bg
    else:
        return img.convert('RGB')

def load_image_as_grayscale(path):
    return Image.open(path).convert('L')

def text_to_indices(text, max_len=128):
    indices = []
    for ch in text[:max_len]:
        idx = ord(ch) if ord(ch) < 256 else 0
        indices.append(idx)
    if len(indices) < max_len:
        indices += [0] * (max_len - len(indices))
    return indices

# ==================== Sinusoidal Position Embedding ====================

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
    def __init__(self, vocab_size=256, embed_dim=64, hidden_size=64, num_layers=2, cond_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_size, num_layers,
                          batch_first=True, bidirectional=True, dropout=0.1 if num_layers>1 else 0)
        self.fc = nn.Linear(hidden_size * 2, cond_dim)

    def forward(self, x):
        emb = self.embedding(x)
        _, h = self.gru(emb)
        h_fwd = h[-2, :, :]
        h_bwd = h[-1, :, :]
        h_cat = torch.cat([h_fwd, h_bwd], dim=1)
        return self.fc(h_cat)

class TransformerTextEncoder(nn.Module):
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
        B, T = x.shape
        emb = self.embedding(x)
        pos = self.pos_embedding[:, :T, :]
        x = emb + pos
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc(x)

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

# ==================== VAE (unchanged) ====================

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

# ==================== DiT Components (Transformer / GRU) ====================

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256, cond_dim=None):
        super().__init__()
        self.cond_proj = nn.Linear(cond_dim, hidden_size) if cond_dim else None
        self.mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(frequency_embedding_size),
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, t, cond=None):
        emb = self.mlp(t)
        if cond is not None and self.cond_proj is not None:
            cond_emb = self.cond_proj(cond)
            emb = emb + cond_emb
        return emb

class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(hidden_size * mlp_ratio), hidden_size),
            nn.Dropout(dropout)
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        norm_x = self.norm1(x)
        mod_norm_x = norm_x * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_out, _ = self.attn(mod_norm_x, mod_norm_x, mod_norm_x)
        x = x + gate_msa.unsqueeze(1) * attn_out
        norm_x = self.norm2(x)
        mod_norm_x = norm_x * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        mlp_out = self.mlp(mod_norm_x)
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        return x

class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = self.norm_final(x)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.linear(x)
        return x

class PatchEmbed(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        B, D, Hp, Wp = x.shape
        x = x.flatten(2).transpose(1, 2)
        return x, Hp, Wp

class PatchUnembed(nn.Module):
    def __init__(self, in_channels, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels

    def forward(self, x, Hp, Wp):
        B, N, D = x.shape
        p = self.patch_size
        C = self.in_channels
        x = x.transpose(1, 2).view(B, C, p, p, Hp, Wp)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(B, C, Hp * p, Wp * p)
        return x

class DiT(nn.Module):
    def __init__(self, latent_channels, patch_size, embed_dim, num_heads, num_layers,
                 mlp_ratio=4.0, dropout=0.1, time_emb_dim=256, cond_dim=None):
        super().__init__()
        self.latent_channels = latent_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(latent_channels, patch_size, embed_dim)
        self.patch_unembed = PatchUnembed(latent_channels, patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, 1024, embed_dim) * 0.02)
        self.time_embedder = TimestepEmbedder(embed_dim, time_emb_dim, cond_dim=cond_dim)

        self.blocks = nn.ModuleList([
            DiTBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(num_layers)
        ])
        self.final_layer = FinalLayer(embed_dim, patch_size, latent_channels)

    def forward(self, z, t, cond=None):
        B, C, H, W = z.shape
        x, Hp, Wp = self.patch_embed(z)
        N = Hp * Wp
        pos = self.pos_embed[:, :N, :]
        x = x + pos
        c = self.time_embedder(t, cond=cond)
        for block in self.blocks:
            x = block(x, c)
        x = self.final_layer(x, c)
        out = self.patch_unembed(x, Hp, Wp)
        return out

class GRUModel(nn.Module):
    def __init__(self, latent_channels, patch_size, embed_dim, time_emb_dim,
                 gru_hidden_dim, gru_num_layers, dropout=0.1, cond_dim=None):
        super().__init__()
        self.latent_channels = latent_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(latent_channels, patch_size, embed_dim)
        self.patch_unembed = PatchUnembed(latent_channels, patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, 1024, embed_dim) * 0.02)
        self.time_embedder = TimestepEmbedder(embed_dim, time_emb_dim, cond_dim=cond_dim)

        self.gru = nn.GRU(embed_dim, gru_hidden_dim, gru_num_layers, batch_first=True,
                          dropout=dropout if gru_num_layers > 1 else 0)
        self.out_proj = nn.Linear(gru_hidden_dim, embed_dim)
        self.final_layer = FinalLayer(embed_dim, patch_size, latent_channels)

    def forward(self, z, t, cond=None):
        B, C, H, W = z.shape
        x, Hp, Wp = self.patch_embed(z)
        N = Hp * Wp
        pos = self.pos_embed[:, :N, :]
        x = x + pos
        c = self.time_embedder(t, cond=cond)
        x, _ = self.gru(x)
        x = self.out_proj(x)
        x = self.final_layer(x, c)
        out = self.patch_unembed(x, Hp, Wp)
        return out

# ==================== Rectified Flow Model Wrapper ====================

class RectifiedFlowModel:
    def __init__(self, model_type, latent_channels, latent_h, latent_w, patch_size,
                 embed_dim, num_heads, num_layers, mlp_ratio, dropout, time_emb_dim,
                 gru_hidden_dim, gru_num_layers, cond_dim, device):
        self.device = device
        self.latent_h = latent_h
        self.latent_w = latent_w
        self.latent_channels = latent_channels
        self.patch_size = patch_size
        self.model_type = model_type

        if model_type == 'transformer':
            self.model = DiT(
                latent_channels=latent_channels,
                patch_size=patch_size,
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                time_emb_dim=time_emb_dim,
                cond_dim=cond_dim if cond_dim > 0 else None
            ).to(device)
        else:
            self.model = GRUModel(
                latent_channels=latent_channels,
                patch_size=patch_size,
                embed_dim=embed_dim,
                time_emb_dim=time_emb_dim,
                gru_hidden_dim=gru_hidden_dim,
                gru_num_layers=gru_num_layers,
                dropout=dropout,
                cond_dim=cond_dim if cond_dim > 0 else None
            ).to(device)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=2e-4)

    def train_step(self, z0, z1, cond=None, cfg_dropout_prob=0.0):
        batch_size = z0.size(0)
        z0 = z0.to(self.device)
        z1 = z1.to(self.device)
        if cond is not None:
            cond = cond.to(self.device)

        t = torch.rand(batch_size, device=self.device)
        z_t = t.view(-1,1,1,1) * z1 + (1 - t.view(-1,1,1,1)) * z0
        target = z1 - z0

        if cond is not None and cfg_dropout_prob > 0:
            mask = torch.rand(batch_size, 1, device=self.device) > cfg_dropout_prob
            cond_dropped = cond * mask.float()
        else:
            cond_dropped = cond

        pred = self.model(z_t, t, cond=cond_dropped)
        loss = self.criterion(pred, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def sample(self, n_samples=16, cond=None, steps=50, method='euler', cfg_scale=1.0, progress_callback=None):
        img_shape = (self.latent_channels, self.latent_h, self.latent_w)
        z = torch.randn(n_samples, *img_shape, device=self.device)

        if cond is not None:
            cond = cond.to(self.device)
            null_cond = torch.zeros_like(cond)
        else:
            null_cond = None

        dt = 1.0 / steps
        times = torch.linspace(0, 1, steps+1, device=self.device)

        def v_fn(t, z):
            t_tensor = torch.full((n_samples,), t, device=self.device)
            if cond is not None and cfg_scale != 1.0:
                v_cond = self.model(z, t_tensor, cond=cond)
                v_uncond = self.model(z, t_tensor, cond=null_cond)
                return v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                return self.model(z, t_tensor, cond=cond)

        for i in range(steps):
            t = times[i]
            if method == 'euler':
                v = v_fn(t, z)
                z = z + v * dt
            elif method == 'heun':
                v1 = v_fn(t, z)
                z_pred = z + v1 * dt
                v2 = v_fn(t + dt, z_pred)
                z = z + (v1 + v2) * (dt / 2)
            elif method == 'rk3':
                v1 = v_fn(t, z)
                z2 = z + v1 * dt
                v2 = v_fn(t + dt, z2)
                z3 = z + (v1 + v2) * (dt / 2)
                v3 = v_fn(t + dt/2, z3)
                z = z + (v1 + 4*v2 + v3) * (dt / 6)
            elif method == 'rk4':
                v1 = v_fn(t, z)
                z2 = z + v1 * (dt/2)
                v2 = v_fn(t + dt/2, z2)
                z3 = z + v2 * (dt/2)
                v3 = v_fn(t + dt/2, z3)
                z4 = z + v3 * dt
                v4 = v_fn(t + dt, z4)
                z = z + (v1 + 2*v2 + 2*v3 + v4) * (dt / 6)
            elif method == 'midpoint':
                v1 = v_fn(t, z)
                t_mid = t + dt/2
                z_mid = z + v1 * (dt/2)
                v_mid = v_fn(t_mid, z_mid)
                z = z + v_mid * dt
            else:
                raise ValueError(f"Unknown method {method}")

            if progress_callback:
                progress_callback(i+1, z)

        return z

    @torch.no_grad()
    def sample_step_by_step(self, n_samples=16, cond=None, steps=50, method='euler', cfg_scale=1.0):
        img_shape = (self.latent_channels, self.latent_h, self.latent_w)
        z = torch.randn(n_samples, *img_shape, device=self.device)

        if cond is not None:
            cond = cond.to(self.device)
            null_cond = torch.zeros_like(cond)
        else:
            null_cond = None

        dt = 1.0 / steps
        times = torch.linspace(0, 1, steps+1, device=self.device)

        def v_fn(t, z):
            t_tensor = torch.full((n_samples,), t, device=self.device)
            if cond is not None and cfg_scale != 1.0:
                v_cond = self.model(z, t_tensor, cond=cond)
                v_uncond = self.model(z, t_tensor, cond=null_cond)
                return v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                return self.model(z, t_tensor, cond=cond)

        for i in range(steps):
            t = times[i]
            if method == 'euler':
                v = v_fn(t, z)
                z = z + v * dt
            elif method == 'heun':
                v1 = v_fn(t, z)
                z_pred = z + v1 * dt
                v2 = v_fn(t + dt, z_pred)
                z = z + (v1 + v2) * (dt / 2)
            elif method == 'rk3':
                v1 = v_fn(t, z)
                z2 = z + v1 * dt
                v2 = v_fn(t + dt, z2)
                z3 = z + (v1 + v2) * (dt / 2)
                v3 = v_fn(t + dt/2, z3)
                z = z + (v1 + 4*v2 + v3) * (dt / 6)
            elif method == 'rk4':
                v1 = v_fn(t, z)
                z2 = z + v1 * (dt/2)
                v2 = v_fn(t + dt/2, z2)
                z3 = z + v2 * (dt/2)
                v3 = v_fn(t + dt/2, z3)
                z4 = z + v3 * dt
                v4 = v_fn(t + dt, z4)
                z = z + (v1 + 2*v2 + 2*v3 + v4) * (dt / 6)
            elif method == 'midpoint':
                v1 = v_fn(t, z)
                t_mid = t + dt/2
                z_mid = z + v1 * (dt/2)
                v_mid = v_fn(t_mid, z_mid)
                z = z + v_mid * dt
            else:
                raise ValueError(f"Unknown method {method}")

            yield i+1, z.clone()

# ==================== Conditional Dataset ====================

class ConditionalImageDataset(Dataset):
    def __init__(self, image_paths, labels_per_image, img_size=32, color_mode='rgb',
                 aug_settings=None, text_max_len=128):
        self.valid_paths = []
        self.labels = []
        for i, path in enumerate(image_paths):
            if color_mode == 'rgb':
                tmp = load_image_as_rgb(path)
            else:
                tmp = load_image_as_grayscale(path)
            if tmp is not None:
                self.valid_paths.append(path)
                lbl = labels_per_image[i]
                if isinstance(lbl, str):
                    self.labels.append([lbl])
                elif isinstance(lbl, list):
                    self.labels.append(lbl)
                else:
                    self.labels.append([''])
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
        return len(self.valid_paths)

    def apply_augmentations(self, pil_img):
        img = pil_img.copy()
        if self.aug_settings.get('flip_horizontal', False) and random.random() < 0.5:
            img = F_vision.hflip(img)
        if self.aug_settings.get('rotation', False) and random.random() < 0.5:
            angle = random.uniform(-30, 30)
            img = F_vision.rotate(img, angle, interpolation=Image.BICUBIC)
        return img

    def __getitem__(self, idx):
        path = self.valid_paths[idx]
        if self.color_mode == 'rgb':
            pil_img = load_image_as_rgb(path)
        else:
            pil_img = load_image_as_grayscale(path)
        if pil_img is None:
            pil_img = Image.new('RGB' if self.color_mode == 'rgb' else 'L', (self.img_size, self.img_size), (0,0,0))
        pil_img = self.apply_augmentations(pil_img)
        img_tensor = self.transform(pil_img)

        captions = self.labels[idx]
        chosen_text = random.choice(captions).replace('_', ' ')
        text_indices = text_to_indices(chosen_text, self.text_max_len)
        text_tensor = torch.tensor(text_indices, dtype=torch.long)
        return img_tensor, text_tensor

# ==================== GUI Application ====================

# Presets for Transformer (DiT)
TRANSFORMER_PRESETS = {
    'tiny':   {'embed_dim': 128, 'num_heads': 4, 'num_layers': 4, 'mlp_ratio': 4.0},
    'small':  {'embed_dim': 256, 'num_heads': 4, 'num_layers': 6, 'mlp_ratio': 4.0},
    'medium': {'embed_dim': 384, 'num_heads': 6, 'num_layers': 8, 'mlp_ratio': 4.0},
    'large':  {'embed_dim': 512, 'num_heads': 8, 'num_layers': 10, 'mlp_ratio': 4.0},
}

# Presets for GRU
GRU_PRESETS = {
    'tiny':   {'embed_dim': 128, 'gru_hidden_dim': 256, 'gru_num_layers': 2},
    'small':  {'embed_dim': 256, 'gru_hidden_dim': 512, 'gru_num_layers': 2},
    'medium': {'embed_dim': 384, 'gru_hidden_dim': 768, 'gru_num_layers': 3},
    'large':  {'embed_dim': 512, 'gru_hidden_dim': 1024, 'gru_num_layers': 3},
}

class RectifiedFlowDiTApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Rectified Flow DiT (Transformer / GRU) - Configurable")

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        win_width = max(1000, int(screen_width * 0.8))
        win_height = max(700, int(screen_height * 0.8))
        self.root.geometry(f"{win_width}x{win_height}")
        self.root.minsize(900, 650)

        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
            dpi = ctypes.windll.user32.GetDpiForWindow(root.winfo_id())
            scale = dpi / 72.0
            self.root.tk.call('tk', 'scaling', scale)
        except:
            pass

        self.image_paths = []
        self.labels = []
        self.csv_path = None
        self.training_vae = False
        self.training_model = False
        self.vae_model = None
        self.rectified_model = None
        self.text_encoder = None

        self.current_epoch_vae = 0
        self.current_epoch_model = 0

        self.message_queue_vae = queue.Queue()
        self.message_queue_model = queue.Queue()
        self.progressive_active = False

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_var = tk.StringVar(value=str(self.device))

        # Settings variables
        self.settings = {
            'img_size': tk.IntVar(value=32),
            'color_mode': tk.StringVar(value='rgb'),

            'vae_size': tk.StringVar(value='big'),
            'vae_base_channels': tk.IntVar(value=32),
            'vae_latent_channels': tk.IntVar(value=8),
            'vae_latent_h': tk.IntVar(value=4),
            'vae_latent_w': tk.IntVar(value=4),
            'vae_batch_size': tk.IntVar(value=16),
            'vae_lr': tk.DoubleVar(value=1e-3),
            'vae_num_workers': tk.IntVar(value=0),
            'vae_kl_weight': tk.DoubleVar(value=0.0001),

            'model_type': tk.StringVar(value='transformer'),
            'patch_size': tk.IntVar(value=1),

            # Transformer params
            'trans_embed_dim': tk.IntVar(value=256),
            'trans_num_heads': tk.IntVar(value=4),
            'trans_num_layers': tk.IntVar(value=6),
            'trans_mlp_ratio': tk.DoubleVar(value=4.0),

            # GRU params
            'gru_embed_dim': tk.IntVar(value=256),
            'gru_hidden_dim': tk.IntVar(value=512),
            'gru_num_layers': tk.IntVar(value=2),

            'time_emb_dim': tk.IntVar(value=256),
            'dropout': tk.DoubleVar(value=0.1),
            'batch_size': tk.IntVar(value=16),
            'lr': tk.DoubleVar(value=2e-4),
            'cfg_dropout_prob': tk.DoubleVar(value=0.1),

            'cond_enabled': tk.BooleanVar(value=False),
            'text_encoder_type': tk.StringVar(value='BiGRU'),
            'text_encoder_size': tk.StringVar(value='small'),
            'cond_embed_dim': tk.IntVar(value=64),
            'cond_hidden_size': tk.IntVar(value=64),
            'cond_num_layers': tk.IntVar(value=2),
            'cond_num_heads': tk.IntVar(value=4),
            'cond_ff_dim': tk.IntVar(value=256),
            'cond_dim': tk.IntVar(value=256),
            'cond_text_max_len': tk.IntVar(value=128),
            'cond_lr': tk.DoubleVar(value=5e-4),
        }

        self.aug_settings = {
            'flip_horizontal': tk.BooleanVar(value=True),
            'rotation': tk.BooleanVar(value=False),
        }

        self.ode_method = tk.StringVar(value='euler')
        self.ode_steps = tk.IntVar(value=50)
        self.cfg_scale = tk.DoubleVar(value=2.0)

        self.thumbnail_size = 128

        self.setup_gui()
        self.root.after(100, self.process_messages_vae)
        self.root.after(100, self.process_messages_model)

    # ---------- GUI Setup ----------
    def setup_gui(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.vae_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.vae_tab, text='VAE Training')
        self.setup_vae_tab()

        self.model_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.model_tab, text='Model Training')
        self.setup_model_tab()

        self.settings_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.settings_tab, text='Settings')
        self.setup_settings_tab()

        self.generation_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.generation_tab, text='Generation')
        self.setup_generation_tab()

        self.status_label = tk.Label(self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

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

    def setup_model_tab(self):
        main_frame = tk.Frame(self.model_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        left_frame = tk.Frame(main_frame, width=300)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0,10))
        left_frame.pack_propagate(False)
        tk.Label(left_frame, text="Model Training (DiT / GRU)", font=("Arial",12,"bold")).pack(pady=(0,10))
        tk.Label(left_frame, text="Requires a trained VAE", fg="blue").pack(pady=5)

        cond_frame = tk.LabelFrame(left_frame, text="Conditional (text)", padx=5, pady=5)
        cond_frame.pack(fill=tk.X, pady=5)
        self.cond_enabled_cb = tk.Checkbutton(cond_frame, text="Enable text conditioning",
                                              variable=self.settings['cond_enabled'])
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
        cfg_dropout_spin = ttk.Spinbox(f, from_=0.0, to=0.5, increment=0.01,
                                       textvariable=self.settings['cfg_dropout_prob'], width=6)
        cfg_dropout_spin.pack(side=tk.RIGHT)

        tk.Button(left_frame, text="Initialize Model", command=self.initialize_model, width=20).pack(pady=5)
        epoch_frame = tk.Frame(left_frame)
        epoch_frame.pack(pady=5)
        tk.Label(epoch_frame, text="Epochs:").pack(side=tk.LEFT)
        self.model_epoch_var = tk.StringVar(value="200")
        tk.Entry(epoch_frame, textvariable=self.model_epoch_var, width=8).pack(side=tk.LEFT, padx=5)
        tk.Button(left_frame, text="Start Training", command=self.start_model_training,
                  width=20, bg="lightgreen").pack(pady=5)
        tk.Button(left_frame, text="Stop Training", command=self.stop_model_training,
                  width=20, bg="salmon").pack(pady=5)
        tk.Button(left_frame, text="Save Model", command=self.save_model, width=20).pack(pady=5)
        tk.Button(left_frame, text="Load Model", command=self.load_model, width=20).pack(pady=5)

        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        preview_canvas_frame = tk.LabelFrame(right_frame, text="Generated Samples (decoded)", padx=5, pady=5)
        preview_canvas_frame.pack(fill=tk.BOTH, expand=True, pady=(0,5))
        self.model_preview_canvas = tk.Canvas(preview_canvas_frame, bg='gray', width=256, height=256)
        self.model_preview_canvas.pack()

        prompt_frame = tk.LabelFrame(right_frame, text="Test prompt (leave empty for unconditional)", padx=5, pady=5)
        prompt_frame.pack(fill=tk.X, pady=(0,5))
        self.test_prompt_entry = tk.Entry(prompt_frame)
        self.test_prompt_entry.insert(0, "a cute cat")
        self.test_prompt_entry.pack(fill=tk.X, pady=2)
        tk.Button(prompt_frame, text="Generate Preview", command=self.model_preview_with_prompt, width=20).pack(pady=2)

        log_frame = tk.LabelFrame(right_frame, text="Log", padx=5, pady=5)
        log_frame.pack(fill=tk.BOTH, expand=True)
        self.model_log_text = tk.Text(log_frame, height=15, font=("Courier",9))
        self.model_log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = tk.Scrollbar(log_frame, command=self.model_log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.model_log_text.config(yscrollcommand=scrollbar.set)

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

        # Hardware
        dev_frame = tk.LabelFrame(scrollable_frame, text="Hardware", padx=10, pady=10)
        dev_frame.pack(fill=tk.X, pady=5)
        f = tk.Frame(dev_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Compute device:", width=20, anchor='w').pack(side=tk.LEFT)
        om = ttk.Combobox(f, textvariable=self.device_var, values=['cuda', 'cpu'], state='readonly', width=10)
        om.pack(side=tk.RIGHT)
        om.bind('<<ComboboxSelected>>', lambda e: self.set_device())

        # Image
        img_frame = tk.LabelFrame(scrollable_frame, text="Image", padx=10, pady=10)
        img_frame.pack(fill=tk.X, pady=5)
        f = tk.Frame(img_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Color mode:", width=20, anchor='w').pack(side=tk.LEFT)
        om = ttk.Combobox(f, textvariable=self.settings['color_mode'], values=['rgb', 'grayscale'], state='readonly', width=10)
        om.pack(side=tk.RIGHT)
        f = tk.Frame(img_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Image size:", width=20, anchor='w').pack(side=tk.LEFT)
        spin = ttk.Spinbox(f, from_=16, to=128, textvariable=self.settings['img_size'], width=8)
        spin.pack(side=tk.RIGHT)

        # VAE (simplified but complete)
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
            f = tk.Frame(vae_frame); f.pack(fill=tk.X, pady=2)
            tk.Label(f, text=label, width=20, anchor='w').pack(side=tk.LEFT)
            if typ == int:
                spin = ttk.Spinbox(f, from_=low, to=high, textvariable=self.settings[key], width=8)
            else:
                spin = ttk.Entry(f, textvariable=self.settings[key], width=8)
            spin.pack(side=tk.RIGHT)

        # Model type and common
        model_frame = tk.LabelFrame(scrollable_frame, text="Rectified Flow Model", padx=10, pady=10)
        model_frame.pack(fill=tk.X, pady=5)
        f = tk.Frame(model_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Model type:", width=20, anchor='w').pack(side=tk.LEFT)
        type_combo = ttk.Combobox(f, textvariable=self.settings['model_type'],
                                  values=['transformer', 'gru'], state='readonly', width=12)
        type_combo.pack(side=tk.RIGHT)
        type_combo.bind('<<ComboboxSelected>>', lambda e: self.update_model_params_visibility())

        f = tk.Frame(model_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Patch size:", width=20, anchor='w').pack(side=tk.LEFT)
        spin = ttk.Spinbox(f, from_=1, to=4, textvariable=self.settings['patch_size'], width=8)
        spin.pack(side=tk.RIGHT)

        # Presets
        preset_frame = tk.Frame(model_frame)
        preset_frame.pack(fill=tk.X, pady=5)
        tk.Label(preset_frame, text="Preset size:").pack(side=tk.LEFT)
        self.preset_var = tk.StringVar(value='small')
        preset_combo = ttk.Combobox(preset_frame, textvariable=self.preset_var,
                                    values=['tiny', 'small', 'medium', 'large'], state='readonly', width=8)
        preset_combo.pack(side=tk.LEFT, padx=5)
        tk.Button(preset_frame, text="Apply Preset", command=self.apply_preset).pack(side=tk.LEFT, padx=5)

        # Transformer params
        self.trans_frame = tk.LabelFrame(model_frame, text="Transformer Parameters", padx=5, pady=5)
        trans_params = [
            ("Embed dim:", 'trans_embed_dim', 32, 512),
            ("Num heads:", 'trans_num_heads', 1, 16),
            ("Num layers:", 'trans_num_layers', 2, 20),
            ("MLP ratio:", 'trans_mlp_ratio', 1.0, 8.0),
        ]
        for label, key, low, high in trans_params:
            f = tk.Frame(self.trans_frame); f.pack(fill=tk.X, pady=2)
            tk.Label(f, text=label, width=15, anchor='w').pack(side=tk.LEFT)
            if isinstance(low, int):
                spin = ttk.Spinbox(f, from_=low, to=high, textvariable=self.settings[key], width=8)
            else:
                spin = ttk.Entry(f, textvariable=self.settings[key], width=8)
            spin.pack(side=tk.RIGHT)

        # GRU params
        self.gru_frame = tk.LabelFrame(model_frame, text="GRU Parameters", padx=5, pady=5)
        gru_params = [
            ("Embed dim:", 'gru_embed_dim', 32, 512),
            ("GRU hidden dim:", 'gru_hidden_dim', 32, 1024),
            ("GRU num layers:", 'gru_num_layers', 1, 8),
        ]
        for label, key, low, high in gru_params:
            f = tk.Frame(self.gru_frame); f.pack(fill=tk.X, pady=2)
            tk.Label(f, text=label, width=15, anchor='w').pack(side=tk.LEFT)
            spin = ttk.Spinbox(f, from_=low, to=high, textvariable=self.settings[key], width=8)
            spin.pack(side=tk.RIGHT)

        # Common training params
        common_frame = tk.LabelFrame(model_frame, text="Common Training Params", padx=5, pady=5)
        common_frame.pack(fill=tk.X, pady=5)
        common_params = [
            ("Time emb dim:", 'time_emb_dim', 64, 512),
            ("Dropout:", 'dropout', 0.0, 0.5),
            ("Batch size:", 'batch_size', 1, 32),
            ("Learning rate:", 'lr', 1e-5, 1e-2),
        ]
        for label, key, low, high in common_params:
            f = tk.Frame(common_frame); f.pack(fill=tk.X, pady=2)
            tk.Label(f, text=label, width=15, anchor='w').pack(side=tk.LEFT)
            if isinstance(low, int):
                spin = ttk.Spinbox(f, from_=low, to=high, textvariable=self.settings[key], width=8)
            else:
                spin = ttk.Entry(f, textvariable=self.settings[key], width=8)
            spin.pack(side=tk.RIGHT)

        # Text Conditioning (simplified but complete)
        cond_frame = tk.LabelFrame(scrollable_frame, text="Text Conditioning", padx=10, pady=10)
        cond_frame.pack(fill=tk.X, pady=5)
        f = tk.Frame(cond_frame)
        f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Encoder type:", width=20, anchor='w').pack(side=tk.LEFT)
        type_combo2 = ttk.Combobox(f, textvariable=self.settings['text_encoder_type'],
                                   values=['BiGRU', 'BiTransformer'], state='readonly', width=12)
        type_combo2.pack(side=tk.RIGHT)
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
            f = tk.Frame(cond_frame); f.pack(fill=tk.X, pady=2)
            tk.Label(f, text=label, width=20, anchor='w').pack(side=tk.LEFT)
            if typ == int:
                spin = ttk.Spinbox(f, from_=low, to=high, textvariable=self.settings[key], width=8)
            else:
                spin = ttk.Entry(f, textvariable=self.settings[key], width=8)
            spin.pack(side=tk.RIGHT)

        # Augmentations
        aug_frame = tk.LabelFrame(scrollable_frame, text="Augmentations", padx=10, pady=10)
        aug_frame.pack(fill=tk.X, pady=5)
        for label, key in [("Horizontal Flip", 'flip_horizontal'), ("Rotation (±30°)", 'rotation')]:
            cb = tk.Checkbutton(aug_frame, text=label, variable=self.aug_settings[key])
            cb.pack(anchor='w')

        sys_frame = tk.LabelFrame(scrollable_frame, text="System", padx=10, pady=10)
        sys_frame.pack(fill=tk.X, pady=5)
        cpu = multiprocessing.cpu_count()
        tk.Label(sys_frame, text=f"CPU cores: {cpu}").pack(anchor='w')
        tk.Label(sys_frame, text=f"PyTorch threads: {torch.get_num_threads()}").pack(anchor='w')
        tk.Label(sys_frame, text=f"Device: {self.device}").pack(anchor='w')

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Show correct frame initially
        self.update_model_params_visibility()

    def update_model_params_visibility(self):
        if self.settings['model_type'].get() == 'transformer':
            self.trans_frame.pack(fill=tk.X, pady=5)
            self.gru_frame.pack_forget()
        else:
            self.trans_frame.pack_forget()
            self.gru_frame.pack(fill=tk.X, pady=5)

    def apply_preset(self):
        model_type = self.settings['model_type'].get()
        preset = self.preset_var.get()
        if model_type == 'transformer':
            if preset in TRANSFORMER_PRESETS:
                cfg = TRANSFORMER_PRESETS[preset]
                self.settings['trans_embed_dim'].set(cfg['embed_dim'])
                self.settings['trans_num_heads'].set(cfg['num_heads'])
                self.settings['trans_num_layers'].set(cfg['num_layers'])
                self.settings['trans_mlp_ratio'].set(cfg['mlp_ratio'])
                self.log_model(f"Applied Transformer preset '{preset}'")
            else:
                self.log_model(f"Unknown preset {preset}")
        else:
            if preset in GRU_PRESETS:
                cfg = GRU_PRESETS[preset]
                self.settings['gru_embed_dim'].set(cfg['embed_dim'])
                self.settings['gru_hidden_dim'].set(cfg['gru_hidden_dim'])
                self.settings['gru_num_layers'].set(cfg['gru_num_layers'])
                self.log_model(f"Applied GRU preset '{preset}'")
            else:
                self.log_model(f"Unknown preset {preset}")

    def apply_text_encoder_preset(self):
        enc_type = self.settings['text_encoder_type'].get()
        size = self.settings['text_encoder_size'].get()
        config = get_encoder_config(enc_type, size)
        self.settings['cond_embed_dim'].set(config.get('embed_dim', self.settings['cond_embed_dim'].get()))
        if enc_type == 'BiGRU':
            self.settings['cond_hidden_size'].set(config.get('hidden_size', self.settings['cond_hidden_size'].get()))
            self.settings['cond_num_layers'].set(config.get('num_layers', self.settings['cond_num_layers'].get()))
        elif enc_type == 'BiTransformer':
            self.settings['cond_num_heads'].set(config.get('num_heads', self.settings['cond_num_heads'].get()))
            self.settings['cond_num_layers'].set(config.get('num_layers', self.settings['cond_num_layers'].get()))
            self.settings['cond_ff_dim'].set(config.get('ff_dim', self.settings['cond_ff_dim'].get()))
        self.settings['cond_dim'].set(config.get('cond_dim', self.settings['cond_dim'].get()))
        self.log_model(f"Applied {enc_type} {size} preset dims.")

    def setup_generation_tab(self):
        main_frame = tk.Frame(self.generation_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        tk.Label(main_frame, text="Generate Images (Rectified Flow)", font=("Arial",14,"bold")).pack(pady=(0,10))

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

        tk.Label(ctrl_frame, text="ODE Steps:").pack(side=tk.LEFT, padx=(10,0))
        tk.Spinbox(ctrl_frame, from_=1, to=200, textvariable=self.ode_steps, width=5).pack(side=tk.LEFT, padx=5)

        tk.Label(ctrl_frame, text="Method:").pack(side=tk.LEFT, padx=(10,0))
        method_combo = ttk.Combobox(ctrl_frame, textvariable=self.ode_method,
                                    values=['euler', 'heun', 'rk3', 'rk4', 'midpoint'], state='readonly', width=8)
        method_combo.pack(side=tk.LEFT, padx=5)

        tk.Label(ctrl_frame, text="CFG scale:").pack(side=tk.LEFT, padx=(10,0))
        tk.Entry(ctrl_frame, width=6, textvariable=self.cfg_scale).pack(side=tk.LEFT, padx=5)

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

    # ---------- Core Methods ----------
    def set_device(self):
        dev_name = self.device_var.get()
        self.device = torch.device(dev_name)
        if self.vae_model:
            self.vae_model.to(self.device)
            self.log_vae(f"VAE moved to {dev_name}")
        if self.rectified_model:
            self.rectified_model.device = self.device
            self.rectified_model.model.to(self.device)
            self.log_model(f"Model moved to {dev_name}")
        if self.text_encoder:
            self.text_encoder.to(self.device)

    def log_vae(self, msg):
        self.message_queue_vae.put(msg)

    def log_model(self, msg):
        self.message_queue_model.put(msg)

    def process_messages_vae(self):
        try:
            while True:
                msg = self.message_queue_vae.get_nowait()
                self.vae_log_text.insert(tk.END, f"{time.strftime('%H:%M:%S')} - {msg}\n")
                self.vae_log_text.see(tk.END)
                self.status_label.config(text=msg[:50])
        except queue.Empty:
            pass
        self.root.after(100, self.process_messages_vae)

    def process_messages_model(self):
        try:
            while True:
                msg = self.message_queue_model.get_nowait()
                self.model_log_text.insert(tk.END, f"{time.strftime('%H:%M:%S')} - {msg}\n")
                self.model_log_text.see(tk.END)
                self.status_label.config(text=msg[:50])
        except queue.Empty:
            pass
        self.root.after(100, self.process_messages_model)

    def add_images(self):
        files = filedialog.askopenfilenames(filetypes=[("Images", "*.jpg *.jpeg *.png *.jfif *.webp *.bmp")])
        for f in files:
            if f not in self.image_paths:
                self.image_paths.append(f)
                self.image_listbox.insert(tk.END, os.path.basename(f))
        self.log_vae(f"Added {len(files)} images. Total: {len(self.image_paths)}")
        self.log_model(f"Added {len(files)} images. Total: {len(self.image_paths)}")

    def add_folder(self):
        folder = filedialog.askdirectory()
        if not folder:
            return
        count = 0
        for root_dir, _, files in os.walk(folder):
            for file in files:
                if file.lower().endswith(('.png','.jpg','.jpeg','.jfif','.webp','.bmp')):
                    full_path = os.path.join(root_dir, file)
                    if full_path not in self.image_paths:
                        self.image_paths.append(full_path)
                        self.image_listbox.insert(tk.END, os.path.basename(full_path))
                        count += 1
        self.log_vae(f"Added {count} images. Total: {len(self.image_paths)}")
        self.log_model(f"Added {count} images. Total: {len(self.image_paths)}")

    def clear_images(self):
        self.image_paths.clear()
        self.labels.clear()
        self.image_listbox.delete(0, tk.END)
        self.csv_status.config(text="No CSV loaded", fg="red")
        self.log_vae("Cleared all images")
        self.log_model("Cleared all images")

    def load_csv(self):
        fname = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if not fname:
            return
        self.csv_path = fname
        path_by_basename = {}
        for full in self.image_paths:
            base = os.path.basename(full).lower()
            if base in path_by_basename:
                self.log_model(f"Warning: duplicate basename '{base}', CSV match may be ambiguous")
            else:
                path_by_basename[base] = full
        label_map = {}
        try:
            with open(fname, 'r', encoding='utf-8-sig') as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row or all(c.strip() == '' for c in row):
                        continue
                    if len(row) < 2:
                        continue
                    img_name = row[0].strip().strip('"').strip("'")
                    label = row[1].strip().strip('"').strip("'")
                    base = os.path.basename(img_name).lower()
                    matched = path_by_basename.get(base, None)
                    if matched:
                        label_map.setdefault(matched, []).append(label)
        except Exception as e:
            self.log_model(f"Error reading CSV: {e}")
            return
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
        self.log_model(f"CSV loaded. {len(self.image_paths)-unknown} images matched with labels.")

    def use_filenames_as_labels(self):
        self.labels = [[os.path.splitext(os.path.basename(p))[0]] for p in self.image_paths]
        self.csv_status.config(text="Using filenames as labels", fg="blue")
        self.log_model("Using filenames as labels.")

    def use_folders_as_labels(self):
        self.labels = []
        for p in self.image_paths:
            folder = os.path.basename(os.path.dirname(p))
            if not folder:
                folder = 'unknown'
            self.labels.append([folder])
        self.csv_status.config(text="Using folder names as labels", fg="blue")
        self.log_model("Using folder names as labels.")

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
            self.vae_model.to(self.device)
            self.vae_optimizer = optim.Adam(self.vae_model.parameters(), lr=self.settings['vae_lr'].get())
            self.log_vae(f"VAE initialized with size '{self.settings['vae_size'].get()}' on {self.device}.")
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
            dummy_labels = [['']] * len(self.image_paths)
            dataset = ConditionalImageDataset(self.image_paths, dummy_labels, img_size, color_mode, aug_dict,
                                              self.settings['cond_text_max_len'].get())
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                num_workers=num_workers, pin_memory=(self.device.type=='cuda'),
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
                    images = images.to(self.device)
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
            if self.labels and len(self.labels) == total:
                preview_labels = [self.labels[i] for i in indices]
            else:
                preview_labels = [['']] * n_preview
            color_mode = self.settings['color_mode'].get()
            dataset = ConditionalImageDataset(preview_paths, preview_labels,
                                              img_size=self.settings['img_size'].get(),
                                              color_mode=color_mode)
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
        fname = filedialog.asksaveasfilename(defaultextension=".pth", filetypes=[("PyTorch","*.pth")])
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
                for k, v in ckpt['settings'].items():
                    if k in self.settings:
                        self.settings[k].set(v)
            if 'vae_size' in ckpt:
                self.settings['vae_size'].set(ckpt['vae_size'])
            self.initialize_vae()
            self.vae_model.load_state_dict(ckpt['model_state'])
            self.vae_optimizer.load_state_dict(ckpt['optimizer_state'])
            self.vae_model.to(self.device)
            self.log_vae(f"VAE loaded from {fname}")
        except Exception as e:
            self.log_vae(f"Load error: {e}")

    # ========== Model Methods ==========
    def initialize_model(self):
        if not self.vae_model:
            self.log_model("Please train/load a VAE first!")
            return
        try:
            model_type = self.settings['model_type'].get()
            latent_channels = self.settings['vae_latent_channels'].get()
            latent_h = self.settings['vae_latent_h'].get()
            latent_w = self.settings['vae_latent_w'].get()
            patch_size = self.settings['patch_size'].get()
            cond_enabled = self.settings['cond_enabled'].get()
            cond_dim = self.settings['cond_dim'].get() if cond_enabled else 0

            if latent_h % patch_size != 0 or latent_w % patch_size != 0:
                self.log_model(f"Latent dimensions ({latent_h}x{latent_w}) not divisible by patch size {patch_size}.")
                return

            if model_type == 'transformer':
                embed_dim = self.settings['trans_embed_dim'].get()
                num_heads = self.settings['trans_num_heads'].get()
                num_layers = self.settings['trans_num_layers'].get()
                mlp_ratio = self.settings['trans_mlp_ratio'].get()
                gru_hidden_dim = None
                gru_num_layers = None
            else:
                embed_dim = self.settings['gru_embed_dim'].get()
                gru_hidden_dim = self.settings['gru_hidden_dim'].get()
                gru_num_layers = self.settings['gru_num_layers'].get()
                num_heads = None
                num_layers = None
                mlp_ratio = None

            self.rectified_model = RectifiedFlowModel(
                model_type=model_type,
                latent_channels=latent_channels,
                latent_h=latent_h,
                latent_w=latent_w,
                patch_size=patch_size,
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                mlp_ratio=mlp_ratio,
                dropout=self.settings['dropout'].get(),
                time_emb_dim=self.settings['time_emb_dim'].get(),
                gru_hidden_dim=gru_hidden_dim,
                gru_num_layers=gru_num_layers,
                cond_dim=cond_dim,
                device=self.device
            )
            for pg in self.rectified_model.optimizer.param_groups:
                pg['lr'] = self.settings['lr'].get()

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
                else:
                    self.text_encoder = TransformerTextEncoder(
                        vocab_size=256,
                        embed_dim=self.settings['cond_embed_dim'].get(),
                        num_heads=self.settings['cond_num_heads'].get(),
                        num_layers=self.settings['cond_num_layers'].get(),
                        ff_dim=self.settings['cond_ff_dim'].get(),
                        cond_dim=self.settings['cond_dim'].get(),
                        max_len=self.settings['cond_text_max_len'].get()
                    )
                self.text_encoder.to(self.device)
                self.model_optimizer = optim.Adam(
                    list(self.rectified_model.model.parameters()) + list(self.text_encoder.parameters()),
                    lr=self.settings['lr'].get()
                )
                self.rectified_model.optimizer = self.model_optimizer
                self.log_model(f"Conditional {model_type.upper()} initialized, patch_size={patch_size}.")
            else:
                self.text_encoder = None
                self.log_model(f"Unconditional {model_type.upper()} initialized, patch_size={patch_size}.")
        except Exception as e:
            self.log_model(f"Init error: {e}")

    def start_model_training(self):
        if not self.image_paths:
            self.log_model("No images!")
            return
        if not self.rectified_model:
            self.log_model("Model not initialized!")
            return
        if not self.vae_model:
            self.log_model("No VAE!")
            return
        if self.training_model:
            self.log_model("Already training.")
            return
        cond_enabled = self.settings['cond_enabled'].get()
        if cond_enabled and not self.labels:
            self.log_model("Conditional training requires labels. Load CSV or use filenames/folders.")
            return
        try:
            epochs = int(self.model_epoch_var.get())
        except:
            self.log_model("Invalid epochs")
            return
        self.training_model = True
        self.current_epoch_model = 0
        self.model_start_time = time.time()
        thread = threading.Thread(target=self.train_model_loop, args=(epochs,), daemon=True)
        thread.start()
        self.log_model(f"Training started for {epochs} epochs.")

    def train_model_loop(self, epochs):
        try:
            batch_size = self.settings['batch_size'].get()
            num_workers = self.settings['vae_num_workers'].get()
            img_size = self.settings['img_size'].get()
            color_mode = self.settings['color_mode'].get()
            cond_enabled = self.settings['cond_enabled'].get()
            text_max_len = self.settings['cond_text_max_len'].get()
            cfg_dropout = self.settings['cfg_dropout_prob'].get()
            aug_dict = {k: v.get() for k, v in self.aug_settings.items()}
            dataset_labels = self.labels if self.labels else [['']]*len(self.image_paths)
            dataset = ConditionalImageDataset(self.image_paths, dataset_labels, img_size, color_mode, aug_dict, text_max_len)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                num_workers=num_workers, pin_memory=(self.device.type=='cuda'),
                                persistent_workers=False if num_workers==0 else True)

            latent_shape = (self.settings['vae_latent_channels'].get(),
                            self.settings['vae_latent_h'].get(),
                            self.settings['vae_latent_w'].get())

            for epoch in range(epochs):
                if not self.training_model:
                    break
                self.current_epoch_model = epoch
                epoch_loss = 0.0
                batches = 0
                for images, text_tensors in loader:
                    if not self.training_model:
                        break
                    images = images.to(self.device)
                    with torch.no_grad():
                        z1 = self.vae_model.encode(images)
                    z0 = torch.randn(z1.size(0), *latent_shape, device=self.device)
                    if cond_enabled and self.text_encoder is not None:
                        cond = self.text_encoder(text_tensors.to(self.device))
                    else:
                        cond = None
                    loss = self.rectified_model.train_step(z0, z1, cond=cond, cfg_dropout_prob=cfg_dropout)
                    epoch_loss += loss
                    batches += 1
                avg_loss = epoch_loss / batches if batches else 0
                elapsed = time.time() - self.model_start_time
                self.log_model(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f} | Time: {elapsed:.1f}s")
                if (epoch+1) % 5 == 0:
                    self.model_preview_with_prompt()
            self.training_model = False
            self.log_model("Training finished.")
        except Exception as e:
            self.log_model(f"Training error: {e}")
            import traceback
            traceback.print_exc()
            self.training_model = False

    def model_preview_with_prompt(self):
        if not self.rectified_model or not self.vae_model:
            self.log_model("Models not loaded for preview")
            return
        try:
            prompt = self.test_prompt_entry.get().strip()
            cond_enabled = self.settings['cond_enabled'].get()
            unconditional = (prompt == "") or not cond_enabled
            cond = None
            cfg_scale = 1.0 if unconditional else self.cfg_scale.get()

            if not unconditional and self.text_encoder is not None:
                text_indices = text_to_indices(prompt, self.settings['cond_text_max_len'].get())
                text_tensor = torch.tensor([text_indices] * 16, dtype=torch.long, device=self.device)
                with torch.no_grad():
                    cond = self.text_encoder(text_tensor)
                self.log_model(f"Preview: conditional with prompt '{prompt}'")
            elif not unconditional:
                self.log_model("Preview: conditioning disabled, generating unconditional")
                unconditional = True
            else:
                self.log_model("Preview: unconditional generation")

            z = self.rectified_model.sample(n_samples=16, steps=10, method='euler', cond=cond, cfg_scale=cfg_scale)
            target_size = (self.settings['img_size'].get(), self.settings['img_size'].get())
            with torch.no_grad():
                samples = self.vae_model.decode(z, target_size=target_size)
            samples = (samples + 1) / 2
            samples = samples.clamp(0,1).cpu().numpy()
            thumb = self.thumbnail_size
            grid = Image.new('RGB', (4*thumb, 4*thumb))
            for i in range(4):
                for j in range(4):
                    idx = i*4 + j
                    if idx < len(samples):
                        if samples[idx].shape[0] == 1:
                            img = np.stack([samples[idx][0]]*3, axis=-1)
                        else:
                            img = samples[idx].transpose(1,2,0)
                        img = (img * 255).astype(np.uint8)
                        pil_img = Image.fromarray(img).resize((thumb, thumb), Image.NEAREST)
                        grid.paste(pil_img, (j*thumb, i*thumb))
            grid = grid.resize((256,256), Image.NEAREST)
            self.model_preview_photo = ImageTk.PhotoImage(grid)
            self.model_preview_canvas.delete("all")
            self.model_preview_canvas.create_image(128,128, image=self.model_preview_photo)
        except Exception as e:
            self.log_model(f"Preview error: {e}")

    def stop_model_training(self):
        self.training_model = False
        self.log_model("Training stopped.")

    def save_model(self):
        if not self.rectified_model:
            self.log_model("No model.")
            return
        fname = filedialog.asksaveasfilename(defaultextension=".pth", filetypes=[("PyTorch","*.pth")])
        if fname:
            save_dict = {
                'model_state': self.rectified_model.model.state_dict(),
                'optimizer_state': self.rectified_model.optimizer.state_dict(),
                'settings': {k:v.get() for k,v in self.settings.items()},
                'model_type': self.settings['model_type'].get(),
            }
            if self.text_encoder is not None:
                save_dict['text_encoder_state'] = self.text_encoder.state_dict()
            torch.save(save_dict, fname)
            self.log_model(f"Model saved to {fname}")

    def load_model(self):
        fname = filedialog.askopenfilename(filetypes=[("PyTorch","*.pth")])
        if not fname:
            return
        try:
            ckpt = torch.load(fname, map_location='cpu')
            if 'settings' in ckpt:
                for k, v in ckpt['settings'].items():
                    if k in self.settings:
                        self.settings[k].set(v)
            if not self.vae_model:
                self.log_model("Please load VAE first.")
                return
            self.initialize_model()
            self.rectified_model.model.load_state_dict(ckpt['model_state'])
            self.rectified_model.optimizer.load_state_dict(ckpt['optimizer_state'])
            if 'text_encoder_state' in ckpt and self.text_encoder is not None:
                self.text_encoder.load_state_dict(ckpt['text_encoder_state'])
            self.log_model(f"Model loaded from {fname}")
        except Exception as e:
            self.log_model(f"Load error: {e}")

    # ========== Generation Methods ==========
    def generate_samples(self):
        if not self.rectified_model or not self.vae_model:
            self.gen_info.config(text="Models not loaded!")
            return
        cond_enabled = self.settings['cond_enabled'].get()
        if cond_enabled and self.text_encoder is None:
            self.gen_info.config(text="Text encoder not initialized!")
            return
        n = self.gen_count.get()
        steps = self.ode_steps.get()
        method = self.ode_method.get()
        prompt = self.gen_prompt.get().strip()
        cfg_scale_user = self.cfg_scale.get()
        cond = None
        unconditional = (prompt == "") or not cond_enabled
        effective_cfg_scale = 1.0 if unconditional else cfg_scale_user

        if not unconditional and cond_enabled and self.text_encoder is not None:
            text_indices = text_to_indices(prompt, self.settings['cond_text_max_len'].get())
            text_tensor = torch.tensor([text_indices] * n, dtype=torch.long, device=self.device)
            with torch.no_grad():
                cond = self.text_encoder(text_tensor)
        elif not unconditional:
            self.gen_info.config(text="No prompt provided.")
            return

        if self.progressive_grid.get():
            self.start_progressive(n, steps, method, cond, effective_cfg_scale)
        else:
            self.generate_btn.config(state=tk.DISABLED)
            if unconditional:
                self.gen_info.config(text="Generating unconditionally...")
            else:
                self.gen_info.config(text=f"Generating with prompt: {prompt}")
            self.root.update()
            thread = threading.Thread(target=self._generate_thread, args=(n, steps, method, cond, effective_cfg_scale), daemon=True)
            thread.start()

    def stop_progressive(self):
        self.progressive_active = False
        self.stop_prog_btn.config(state=tk.DISABLED)
        self.generate_btn.config(state=tk.NORMAL)
        self.gen_info.config(text="Progressive stopped.")

    def start_progressive(self, n, steps, method, cond, cfg_scale):
        self.progressive_active = True
        self.generate_btn.config(state=tk.DISABLED)
        self.stop_prog_btn.config(state=tk.NORMAL)
        self.gen_info.config(text="Progressive generation...")
        thread = threading.Thread(target=self._progressive_thread, args=(n, steps, method, cond, cfg_scale), daemon=True)
        thread.start()

    def _generate_thread(self, n, steps, method, cond, cfg_scale):
        try:
            z = self.rectified_model.sample(n_samples=n, steps=steps, method=method, cond=cond, cfg_scale=cfg_scale)
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
                    img = np.stack([samples[i][0]]*3, axis=-1)
                else:
                    img = samples[i].transpose(1,2,0)
                img = (img * 255).astype(np.uint8)
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
            generator = self.rectified_model.sample_step_by_step(n_samples=n, steps=steps, method=method, cond=cond, cfg_scale=cfg_scale)
            target_size = (self.settings['img_size'].get(), self.settings['img_size'].get())
            for step_idx, z in generator:
                if not self.progressive_active:
                    break
                if step_idx % interval == 0 or step_idx == steps:
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
                                img = np.stack([samples[i][0]]*3, axis=-1)
                            else:
                                img = samples[i].transpose(1,2,0)
                            img = (img * 255).astype(np.uint8)
                            pil_img = Image.fromarray(img).resize((thumb, thumb), Image.NEAREST)
                            grid_img.paste(pil_img, (col*thumb, row*thumb))
                    self.root.after(0, lambda g=grid_img, s=step_idx: self._update_progressive(g, s))
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

    def _update_progressive(self, grid_img, step_idx):
        for widget in self.inner_frame.winfo_children():
            widget.destroy()
        self.prog_photo = ImageTk.PhotoImage(grid_img)
        label = tk.Label(self.inner_frame, image=self.prog_photo)
        label.image = self.prog_photo
        label.pack()
        self.inner_frame.update_idletasks()
        self.gen_canvas.configure(scrollregion=self.gen_canvas.bbox('all'))
        self.gen_info.config(text=f"ODE step {step_idx}/{self.ode_steps.get()}")

# ==================== Main ====================

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    root = tk.Tk()
    app = RectifiedFlowDiTApp(root)
    root.mainloop()