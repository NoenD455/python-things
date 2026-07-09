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

# ==================== EMA ====================
class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name])

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name])

# ==================== Sinusoidal Position Embeddings ====================
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
        raise ValueError(f"Unknown encoder type '{enc_type}'")
    if size not in TEXT_ENCODER_PRESETS[enc_type]:
        raise ValueError(f"Unknown size '{size}'")
    return TEXT_ENCODER_PRESETS[enc_type][size]

# ==================== Pixel-Space UViT and U-GRU ====================
class PixelUViTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(hidden_size * mlp_ratio), hidden_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + attn_out
        norm_x = self.norm2(x)
        mlp_out = self.mlp(norm_x)
        x = x + mlp_out
        return x

class PixelTimestepEmbedder(nn.Module):
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

class PixelPatchEmbed(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        B, D, Hp, Wp = x.shape
        x = x.flatten(2).transpose(1, 2)
        return x, Hp, Wp

class PixelPatchUnembed(nn.Module):
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

class PixelFinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, eps=1e-6)
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

class PixelUViT(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim, num_heads, num_layers,
                 mlp_ratio=4.0, dropout=0.1, time_emb_dim=256, cond_dim=None, use_final_conv=True):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.use_final_conv = use_final_conv
        assert num_layers % 2 == 1, "U-ViT requires odd number of layers"

        self.patch_embed = PixelPatchEmbed(in_channels, patch_size, embed_dim)
        self.patch_unembed = PixelPatchUnembed(in_channels, patch_size)
        # Position embedding for up to 1024 patches (32x32 image with patch_size=1 gives 1024 patches)
        self.pos_embed = nn.Parameter(torch.randn(1, 1024, embed_dim) * 0.02)
        self.time_embedder = PixelTimestepEmbedder(embed_dim, time_emb_dim, cond_dim=cond_dim)

        self.blocks = nn.ModuleList([
            PixelUViTBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(num_layers)
        ])

        half = (num_layers - 1) // 2
        self.skip_projs = nn.ModuleList([nn.Identity() for _ in range(half)])
        self.combine_projs = nn.ModuleList([nn.Linear(embed_dim * 2, embed_dim) for _ in range(half)])

        self.final_layer = PixelFinalLayer(embed_dim, patch_size, in_channels)

        # Optional final 3x3 conv for better visual quality (as per U-ViT paper)
        if self.use_final_conv:
            self.final_conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        else:
            self.final_conv = nn.Identity()

    def forward(self, x, t, cond=None):
        B, C, H, W = x.shape
        x_patch, Hp, Wp = self.patch_embed(x)
        N = Hp * Wp
        pos = self.pos_embed[:, :N, :]
        x_patch = x_patch + pos
        c = self.time_embedder(t, cond=cond)

        shallow_outs = []
        half = (self.num_layers - 1) // 2

        for i, block in enumerate(self.blocks):
            if i < half:
                x_patch = block(x_patch)
                shallow_outs.append(x_patch)
            else:
                skip_idx = self.num_layers - 1 - i
                if skip_idx < half:
                    skip = shallow_outs[skip_idx]
                    skip = self.skip_projs[skip_idx](skip)
                    x_patch = torch.cat([x_patch, skip], dim=2)
                    x_patch = self.combine_projs[skip_idx](x_patch)
                x_patch = block(x_patch)

        out_patch = self.final_layer(x_patch, c)
        out = self.patch_unembed(out_patch, Hp, Wp)
        out = self.final_conv(out)
        return out

class PixelUGRU(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim, gru_hidden_dim, gru_num_layers,
                 dropout=0.1, time_emb_dim=256, cond_dim=None):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.gru_num_layers = gru_num_layers
        assert gru_num_layers % 2 == 1, "U-GRU requires odd number of layers"

        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, 1024, embed_dim) * 0.02)

        self.time_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.time_embedder = SinusoidalPositionEmbeddings(time_emb_dim)
        self.time_proj = nn.Linear(time_emb_dim, embed_dim)

        self.cond_proj = nn.Linear(cond_dim, embed_dim) if cond_dim else None

        self.gru_layers = nn.ModuleList()
        for l in range(gru_num_layers):
            input_dim = embed_dim if l == 0 else gru_hidden_dim
            layer_dropout = dropout if l+1 < gru_num_layers else 0.0
            self.gru_layers.append(nn.GRU(input_dim, gru_hidden_dim, batch_first=True, dropout=layer_dropout))

        half = (gru_num_layers - 1) // 2
        self.skip_projs = nn.ModuleList([nn.Identity() for _ in range(half)])
        self.combine_projs = nn.ModuleList([nn.Linear(gru_hidden_dim * 2, gru_hidden_dim) for _ in range(half)])

        self.out_proj = nn.Linear(gru_hidden_dim, embed_dim)
        self.final_norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.final_linear = nn.Linear(embed_dim, patch_size * patch_size * in_channels)
        self.final_conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)

    def forward(self, x, t, cond=None):
        B, C, H, W = x.shape
        p = self.patch_size
        Hp, Wp = H // p, W // p
        N = Hp * Wp

        x_patch = self.patch_embed(x).flatten(2).transpose(1, 2)
        t_emb = self.time_embedder(t)
        t_emb = self.time_proj(t_emb).unsqueeze(1)
        time_tokens = self.time_token + t_emb

        if cond is not None and self.cond_proj is not None:
            cond_tokens = self.cond_proj(cond).unsqueeze(1)
            tokens = torch.cat([time_tokens, cond_tokens, x_patch], dim=1)
            num_extra = 2
        else:
            tokens = torch.cat([time_tokens, x_patch], dim=1)
            num_extra = 1

        total_len = tokens.shape[1]
        pos = self.pos_embed[:, :total_len, :]
        tokens = tokens + pos

        shallow_outs = []
        half = (self.gru_num_layers - 1) // 2

        for i, gru in enumerate(self.gru_layers):
            if i < half:
                tokens, _ = gru(tokens)
                shallow_outs.append(tokens)
            else:
                skip_idx = self.gru_num_layers - 1 - i
                if skip_idx < half:
                    skip = shallow_outs[skip_idx]
                    skip = self.skip_projs[skip_idx](skip)
                    tokens = torch.cat([tokens, skip], dim=2)
                    tokens = self.combine_projs[skip_idx](tokens)
                tokens, _ = gru(tokens)

        tokens = self.out_proj(tokens)
        tokens = self.final_norm(tokens)
        tokens = self.final_linear(tokens)
        tokens = tokens[:, num_extra:, :]

        out = tokens.transpose(1, 2).reshape(B, C, p, p, Hp, Wp)
        out = out.permute(0, 1, 4, 2, 5, 3).contiguous()
        out = out.reshape(B, C, H, W)
        out = self.final_conv(out)
        return out

# Model presets for pixel space
PIXEL_UVIT_PRESETS = {
    'tiny':   {'embed_dim': 48,  'num_heads': 2, 'num_layers': 5, 'mlp_ratio': 4.0},
    'small':  {'embed_dim': 96,  'num_heads': 3, 'num_layers': 7, 'mlp_ratio': 4.0},
    'medium': {'embed_dim': 128, 'num_heads': 4, 'num_layers': 9, 'mlp_ratio': 4.0},
    'large':  {'embed_dim': 192, 'num_heads': 6, 'num_layers': 11, 'mlp_ratio': 4.0},
}

PIXEL_UGRU_PRESETS = {
    'tiny':   {'embed_dim': 48, 'gru_hidden_dim': 96, 'gru_num_layers': 3},
    'small':  {'embed_dim': 96, 'gru_hidden_dim': 192, 'gru_num_layers': 5},
    'medium': {'embed_dim': 128, 'gru_hidden_dim': 256, 'gru_num_layers': 7},
    'large':  {'embed_dim': 192, 'gru_hidden_dim': 384, 'gru_num_layers': 9},
}

# ==================== Rectified Flow Pixel Model with UViT/U-GRU ====================
class RectifiedFlowPixel:
    def __init__(self, model_type, in_channels=3, img_size=32, patch_size=1,
                 embed_dim=96, num_heads=3, num_layers=7, mlp_ratio=4.0,
                 gru_hidden_dim=192, gru_num_layers=5,
                 dropout=0.1, time_emb_dim=256, cond_dim=None, device=None,
                 use_final_conv=True):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.cond_dim = cond_dim if cond_dim else 0
        self.model_type = model_type

        if model_type == 'uvit':
            self.model = PixelUViT(
                in_channels=in_channels,
                patch_size=patch_size,
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                time_emb_dim=time_emb_dim,
                cond_dim=self.cond_dim if self.cond_dim > 0 else None,
                use_final_conv=use_final_conv
            ).to(self.device)
        else:  # ugru
            self.model = PixelUGRU(
                in_channels=in_channels,
                patch_size=patch_size,
                embed_dim=embed_dim,
                gru_hidden_dim=gru_hidden_dim,
                gru_num_layers=gru_num_layers,
                dropout=dropout,
                time_emb_dim=time_emb_dim,
                cond_dim=self.cond_dim if self.cond_dim > 0 else None
            ).to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=2e-4)
        self.ema = None

    def set_ema(self, decay=0.999):
        self.ema = EMA(self.model, decay)

    def train_step(self, x0, x1, cond=None, cfg_dropout_prob=0.0, update_ema=True, ema_decay=0.999):
        batch_size = x0.size(0)
        x0 = x0.to(self.device)
        x1 = x1.to(self.device)

        if cond is None:
            cond = torch.zeros(batch_size, self.cond_dim, device=self.device)
        else:
            cond = cond.to(self.device)

        t = torch.rand(batch_size, device=self.device)
        x_t = t.view(-1,1,1,1) * x1 + (1 - t.view(-1,1,1,1)) * x0
        target = x1 - x0

        if cfg_dropout_prob > 0:
            mask = torch.rand(batch_size, 1, device=self.device) > cfg_dropout_prob
            cond_dropped = cond * mask.float()
        else:
            cond_dropped = cond

        pred = self.model(x_t, t, cond=cond_dropped)
        loss = self.criterion(pred, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if update_ema and self.ema is not None:
            self.ema.update()

        return loss.item()

    @torch.no_grad()
    def sample(self, n_samples=16, cond=None, steps=50, method='euler', cfg_scale=1.0,
               use_ema=True, progress_callback=None):
        if use_ema and self.ema is not None:
            self.ema.apply_shadow()
        img_shape = (self.model.in_channels, self.img_size[0], self.img_size[1])
        x = torch.randn(n_samples, *img_shape, device=self.device)

        if cond is None:
            cond = torch.zeros(n_samples, self.cond_dim, device=self.device)
        else:
            cond = cond.to(self.device)
        null_cond = torch.zeros_like(cond)

        dt = 1.0 / steps
        times = torch.linspace(0, 1, steps+1, device=self.device)

        def v_fn(t, x):
            t_tensor = torch.full((n_samples,), t, device=self.device)
            if cfg_scale != 1.0:
                v_cond = self.model(x, t_tensor, cond=cond)
                v_uncond = self.model(x, t_tensor, cond=null_cond)
                return v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                return self.model(x, t_tensor, cond=cond)

        for i in range(steps):
            t = times[i]
            if method.lower() == 'euler':
                v = v_fn(t, x)
                x = x + v * dt
            elif method.lower() == 'heun':
                v1 = v_fn(t, x)
                x_pred = x + v1 * dt
                v2 = v_fn(t + dt, x_pred)
                x = x + (v1 + v2) * (dt / 2)
            elif method.lower() == 'rk3':
                v1 = v_fn(t, x)
                x2 = x + v1 * dt
                v2 = v_fn(t + dt, x2)
                x3 = x + (v1 + v2) * (dt / 2)
                v3 = v_fn(t + dt/2, x3)
                x = x + (v1 + 4*v2 + v3) * (dt / 6)
            elif method.lower() == 'rk4':
                v1 = v_fn(t, x)
                x2 = x + v1 * (dt/2)
                v2 = v_fn(t + dt/2, x2)
                x3 = x + v2 * (dt/2)
                v3 = v_fn(t + dt/2, x3)
                x4 = x + v3 * dt
                v4 = v_fn(t + dt, x4)
                x = x + (v1 + 2*v2 + 2*v3 + v4) * (dt / 6)
            else:
                raise ValueError(f"Unknown method {method}")

            if progress_callback is not None:
                progress_callback(i+1, x)

        if use_ema and self.ema is not None:
            self.ema.restore()
        return x

    @torch.no_grad()
    def sample_step_by_step(self, n_samples=16, cond=None, steps=50, method='euler',
                            cfg_scale=1.0, use_ema=True):
        if use_ema and self.ema is not None:
            self.ema.apply_shadow()
        img_shape = (self.model.in_channels, self.img_size[0], self.img_size[1])
        x = torch.randn(n_samples, *img_shape, device=self.device)

        if cond is None:
            cond = torch.zeros(n_samples, self.cond_dim, device=self.device)
        else:
            cond = cond.to(self.device)
        null_cond = torch.zeros_like(cond)

        dt = 1.0 / steps
        times = torch.linspace(0, 1, steps+1, device=self.device)

        def v_fn(t, x):
            t_tensor = torch.full((n_samples,), t, device=self.device)
            if cfg_scale != 1.0:
                v_cond = self.model(x, t_tensor, cond=cond)
                v_uncond = self.model(x, t_tensor, cond=null_cond)
                return v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                return self.model(x, t_tensor, cond=cond)

        for i in range(steps):
            t = times[i]
            if method.lower() == 'euler':
                v = v_fn(t, x)
                x = x + v * dt
            elif method.lower() == 'heun':
                v1 = v_fn(t, x)
                x_pred = x + v1 * dt
                v2 = v_fn(t + dt, x_pred)
                x = x + (v1 + v2) * (dt / 2)
            elif method.lower() == 'rk3':
                v1 = v_fn(t, x)
                x2 = x + v1 * dt
                v2 = v_fn(t + dt, x2)
                x3 = x + (v1 + v2) * (dt / 2)
                v3 = v_fn(t + dt/2, x3)
                x = x + (v1 + 4*v2 + v3) * (dt / 6)
            elif method.lower() == 'rk4':
                v1 = v_fn(t, x)
                x2 = x + v1 * (dt/2)
                v2 = v_fn(t + dt/2, x2)
                x3 = x + v2 * (dt/2)
                v3 = v_fn(t + dt/2, x3)
                x4 = x + v3 * dt
                v4 = v_fn(t + dt, x4)
                x = x + (v1 + 2*v2 + 2*v3 + v4) * (dt / 6)
            else:
                raise ValueError(f"Unknown method {method}")

            yield i+1, x.clone()

        if use_ema and self.ema is not None:
            self.ema.restore()

# ==================== Conditional Dataset ====================
class ConditionalImageDataset(Dataset):
    def __init__(self, image_paths, labels_per_image, img_size=32, color_mode='rgb',
                 aug_settings=None, text_max_len=128):
        self.image_paths = image_paths
        self.labels = []
        for lbl in labels_per_image:
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
class RectifiedFlowApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Rectified Flow Pixel Space - UViT / U-GRU (with final conv)")

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        win_width = max(1000, int(screen_width * 0.8))
        win_height = max(700, int(screen_height * 0.8))
        self.root.geometry(f"{win_width}x{win_height}")
        self.root.minsize(900, 650)

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
        self.labels = []
        self.csv_path = None
        self.training_rectified = False
        self.rectified_model = None
        self.text_encoder = None

        self.current_epoch_rectified = 0

        self.message_queue_dataset = queue.Queue()
        self.message_queue_rectified = queue.Queue()
        self.progressive_active = False

        # Settings variables
        self.settings = {
            # Image
            'img_size': tk.IntVar(value=32),
            'color_mode': tk.StringVar(value='rgb'),

            # Model type
            'model_type': tk.StringVar(value='uvit'),
            'patch_size': tk.IntVar(value=1),   # patch size for UViT/U-GRU

            # UViT params
            'uvit_embed_dim': tk.IntVar(value=96),
            'uvit_num_heads': tk.IntVar(value=3),
            'uvit_num_layers': tk.IntVar(value=7),
            'uvit_mlp_ratio': tk.DoubleVar(value=4.0),
            'uvit_final_conv': tk.BooleanVar(value=True),   # NEW: final 3x3 conv

            # U-GRU params
            'ugru_embed_dim': tk.IntVar(value=96),
            'ugru_gru_hidden_dim': tk.IntVar(value=192),
            'ugru_gru_num_layers': tk.IntVar(value=5),

            # Common training
            'rectified_batch_size': tk.IntVar(value=16),
            'rectified_lr': tk.DoubleVar(value=2e-4),
            'cfg_dropout_prob': tk.DoubleVar(value=0.1),
            'dropout': tk.DoubleVar(value=0.1),
            'time_emb_dim': tk.IntVar(value=256),

            # EMA
            'ema_enabled': tk.BooleanVar(value=False),
            'ema_decay': tk.DoubleVar(value=0.999),

            # Text conditioning
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

            # Preview during training
            'preview_enabled': tk.BooleanVar(value=True),
            'preview_steps': tk.IntVar(value=50),
            'preview_method': tk.StringVar(value='heun'),
            'preview_epoch_freq': tk.IntVar(value=5),
        }

        # Augmentation settings
        self.aug_settings = {
            'flip_horizontal': tk.BooleanVar(value=True),
            'rotation': tk.BooleanVar(value=False),
        }

        # Generation parameters
        self.ode_method = tk.StringVar(value='euler')
        self.ode_steps = tk.IntVar(value=50)
        self.cfg_scale = tk.DoubleVar(value=2.0)

        self.thumbnail_size = 128

        self.setup_gui()
        self.root.after(100, self.process_messages_dataset)
        self.root.after(100, self.process_messages_rectified)

    def setup_gui(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.dataset_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.dataset_tab, text='Dataset')
        self.setup_dataset_tab()

        self.rectified_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.rectified_tab, text='Rectified Flow Training')
        self.setup_rectified_tab()

        self.settings_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.settings_tab, text='Settings')
        self.setup_settings_tab()

        self.generation_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.generation_tab, text='Generation')
        self.setup_generation_tab()

        self.status_label = tk.Label(self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    def setup_dataset_tab(self):
        main_frame = tk.Frame(self.dataset_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_frame = tk.Frame(main_frame, width=300)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0,10))
        left_frame.pack_propagate(False)

        tk.Label(left_frame, text="Dataset Management", font=("Arial",12,"bold")).pack(pady=(0,10))

        img_frame = tk.LabelFrame(left_frame, text="Training Images", padx=5, pady=5)
        img_frame.pack(fill=tk.X, pady=(0,10))
        tk.Button(img_frame, text="Add Images", command=self.add_images, width=20).pack(pady=2)
        tk.Button(img_frame, text="Add Folder (recursive)", command=self.add_folder, width=20).pack(pady=2)
        tk.Button(img_frame, text="Clear All", command=self.clear_images, width=20).pack(pady=2)
        self.image_listbox = tk.Listbox(img_frame, height=10)
        self.image_listbox.pack(fill=tk.X, pady=2)

        cond_frame = tk.LabelFrame(left_frame, text="Labels / Captions", padx=5, pady=5)
        cond_frame.pack(fill=tk.X, pady=5)
        tk.Button(cond_frame, text="Load CSV (image,label)", command=self.load_csv, width=20).pack(pady=2)
        tk.Button(cond_frame, text="Use filenames as labels", command=self.use_filenames_as_labels, width=20).pack(pady=2)
        tk.Button(cond_frame, text="Use folder names as labels", command=self.use_folders_as_labels, width=20).pack(pady=2)
        self.csv_status = tk.Label(cond_frame, text="No CSV loaded", fg="red")
        self.csv_status.pack()

        log_frame = tk.LabelFrame(left_frame, text="Log", padx=5, pady=5)
        log_frame.pack(fill=tk.BOTH, expand=True)
        self.dataset_log_text = tk.Text(log_frame, height=15, font=("Courier",9))
        self.dataset_log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = tk.Scrollbar(log_frame, command=self.dataset_log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.dataset_log_text.config(yscrollcommand=scrollbar.set)

    def setup_rectified_tab(self):
        main_frame = tk.Frame(self.rectified_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_frame = tk.Frame(main_frame, width=300)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0,10))
        left_frame.pack_propagate(False)

        tk.Label(left_frame, text="Rectified Flow Training", font=("Arial",12,"bold")).pack(pady=(0,10))

        cond_toggle_frame = tk.LabelFrame(left_frame, text="Conditional Training", padx=5, pady=5)
        cond_toggle_frame.pack(fill=tk.X, pady=5)
        self.cond_cb = tk.Checkbutton(cond_toggle_frame, text="Enable text conditioning",
                                      variable=self.settings['cond_enabled'])
        self.cond_cb.pack(anchor='w')
        tk.Label(cond_toggle_frame, text="Labels must be loaded in Dataset tab", fg="gray").pack(anchor='w')

        cfg_frame = tk.LabelFrame(left_frame, text="CFG Training", padx=5, pady=5)
        cfg_frame.pack(fill=tk.X, pady=5)
        f = tk.Frame(cfg_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="CFG dropout prob:", width=18, anchor='w').pack(side=tk.LEFT)
        cfg_dropout_scale = tk.Scale(f, from_=0.0, to=0.5, resolution=0.01,
                                     variable=self.settings['cfg_dropout_prob'],
                                     orient=tk.HORIZONTAL, length=150)
        cfg_dropout_scale.pack(side=tk.RIGHT)
        tk.Label(f, textvariable=self.settings['cfg_dropout_prob'], width=5).pack(side=tk.RIGHT)

        tk.Button(left_frame, text="Initialize Rectified Flow", command=self.initialize_rectified_model, width=20).pack(pady=5)
        epoch_frame = tk.Frame(left_frame)
        epoch_frame.pack(pady=5)
        tk.Label(epoch_frame, text="Epochs:").pack(side=tk.LEFT)
        self.rectified_epoch_var = tk.StringVar(value="200")
        tk.Entry(epoch_frame, textvariable=self.rectified_epoch_var, width=8).pack(side=tk.LEFT, padx=5)

        tk.Button(left_frame, text="Start Training", command=self.start_rectified_training,
                  width=20, bg="lightgreen").pack(pady=5)
        tk.Button(left_frame, text="Stop Training", command=self.stop_rectified_training,
                  width=20, bg="salmon").pack(pady=5)
        tk.Button(left_frame, text="Save Model", command=self.save_rectified_model, width=20).pack(pady=5)
        tk.Button(left_frame, text="Load Model", command=self.load_rectified_model, width=20).pack(pady=5)

        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        preview_canvas_frame = tk.LabelFrame(right_frame, text="Generated Samples (preview)", padx=5, pady=5)
        preview_canvas_frame.pack(fill=tk.BOTH, expand=True, pady=(0,5))
        self.rectified_preview_canvas = tk.Canvas(preview_canvas_frame, bg='gray', width=256, height=256)
        self.rectified_preview_canvas.pack()

        prompt_frame = tk.LabelFrame(right_frame, text="Test prompt (for manual preview)", padx=5, pady=5)
        prompt_frame.pack(fill=tk.X, pady=(0,5))
        self.test_prompt_entry = tk.Entry(prompt_frame)
        self.test_prompt_entry.insert(0, "a cute cat")
        self.test_prompt_entry.pack(fill=tk.X, pady=2)
        tk.Button(prompt_frame, text="Generate Preview (manual)", command=self.rectified_preview_with_prompt, width=20).pack(pady=2)

        log_frame = tk.LabelFrame(right_frame, text="Log", padx=5, pady=5)
        log_frame.pack(fill=tk.BOTH, expand=True)
        self.rectified_log_text = tk.Text(log_frame, height=15, font=("Courier",9))
        self.rectified_log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = tk.Scrollbar(log_frame, command=self.rectified_log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.rectified_log_text.config(yscrollcommand=scrollbar.set)

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

        # Image settings
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

        # Model type and architecture
        arch_frame = tk.LabelFrame(scrollable_frame, text="Model Architecture", padx=10, pady=10)
        arch_frame.pack(fill=tk.X, pady=5)
        f = tk.Frame(arch_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Model type:", width=20, anchor='w').pack(side=tk.LEFT)
        type_combo = ttk.Combobox(f, textvariable=self.settings['model_type'],
                                  values=['uvit', 'ugru'], state='readonly', width=10)
        type_combo.pack(side=tk.RIGHT)
        type_combo.bind('<<ComboboxSelected>>', lambda e: self.update_model_params_visibility())

        f = tk.Frame(arch_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Patch size:", width=20, anchor='w').pack(side=tk.LEFT)
        spin = ttk.Spinbox(f, from_=1, to=4, textvariable=self.settings['patch_size'], width=8)
        spin.pack(side=tk.RIGHT)

        # Presets
        preset_frame = tk.Frame(arch_frame)
        preset_frame.pack(fill=tk.X, pady=5)
        tk.Label(preset_frame, text="Preset size:").pack(side=tk.LEFT)
        self.preset_var = tk.StringVar(value='small')
        preset_combo = ttk.Combobox(preset_frame, textvariable=self.preset_var,
                                    values=['tiny', 'small', 'medium', 'large'], state='readonly', width=8)
        preset_combo.pack(side=tk.LEFT, padx=5)
        tk.Button(preset_frame, text="Apply Preset", command=self.apply_preset).pack(side=tk.LEFT, padx=5)

        # UViT params
        self.uvit_frame = tk.LabelFrame(arch_frame, text="U-ViT Parameters", padx=5, pady=5)
        uvit_params = [
            ("Embed dim:", 'uvit_embed_dim', 32, 512),
            ("Num heads:", 'uvit_num_heads', 1, 16),
            ("Num layers:", 'uvit_num_layers', 3, 31),
            ("MLP ratio:", 'uvit_mlp_ratio', 1.0, 8.0),
        ]
        for label, key, low, high in uvit_params:
            f = tk.Frame(self.uvit_frame); f.pack(fill=tk.X, pady=2)
            tk.Label(f, text=label, width=15, anchor='w').pack(side=tk.LEFT)
            if isinstance(low, int):
                spin = ttk.Spinbox(f, from_=low, to=high, textvariable=self.settings[key], width=8)
            else:
                spin = ttk.Entry(f, textvariable=self.settings[key], width=8)
            spin.pack(side=tk.RIGHT)

        # Final 3x3 conv checkbox for UViT
        f = tk.Frame(self.uvit_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Final 3x3 Conv (paper)", width=15, anchor='w').pack(side=tk.LEFT)
        cb_final_conv = tk.Checkbutton(f, variable=self.settings['uvit_final_conv'])
        cb_final_conv.pack(side=tk.RIGHT)

        # U-GRU params
        self.ugru_frame = tk.LabelFrame(arch_frame, text="U-GRU Parameters", padx=5, pady=5)
        ugru_params = [
            ("Embed dim:", 'ugru_embed_dim', 32, 512),
            ("GRU hidden dim:", 'ugru_gru_hidden_dim', 32, 1024),
            ("GRU num layers:", 'ugru_gru_num_layers', 3, 15),
        ]
        for label, key, low, high in ugru_params:
            f = tk.Frame(self.ugru_frame); f.pack(fill=tk.X, pady=2)
            tk.Label(f, text=label, width=15, anchor='w').pack(side=tk.LEFT)
            spin = ttk.Spinbox(f, from_=low, to=high, textvariable=self.settings[key], width=8)
            spin.pack(side=tk.RIGHT)

        self.uvit_frame.pack(fill=tk.X, pady=5)
        self.ugru_frame.pack(fill=tk.X, pady=5)
        self.update_model_params_visibility()

        # Common training
        train_frame = tk.LabelFrame(scrollable_frame, text="Training Hyperparameters", padx=10, pady=10)
        train_frame.pack(fill=tk.X, pady=5)
        params = [
            ("Batch size:", 'rectified_batch_size', 1, 32, int),
            ("Learning rate:", 'rectified_lr', 1e-5, 1e-2, float),
            ("Dropout:", 'dropout', 0.0, 0.5, float),
            ("Time emb dim:", 'time_emb_dim', 64, 512, int),
        ]
        for label, key, low, high, typ in params:
            f = tk.Frame(train_frame); f.pack(fill=tk.X, pady=2)
            tk.Label(f, text=label, width=20, anchor='w').pack(side=tk.LEFT)
            if typ == int:
                spin = ttk.Spinbox(f, from_=low, to=high, textvariable=self.settings[key], width=8)
            else:
                spin = ttk.Entry(f, textvariable=self.settings[key], width=8)
            spin.pack(side=tk.RIGHT)

        # EMA
        ema_frame = tk.LabelFrame(scrollable_frame, text="Exponential Moving Average", padx=10, pady=10)
        ema_frame.pack(fill=tk.X, pady=5)
        f = tk.Frame(ema_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Enable EMA:", width=20, anchor='w').pack(side=tk.LEFT)
        cb_ema = tk.Checkbutton(f, variable=self.settings['ema_enabled'])
        cb_ema.pack(side=tk.RIGHT)
        f = tk.Frame(ema_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="EMA decay:", width=20, anchor='w').pack(side=tk.LEFT)
        spin_ema = ttk.Entry(f, textvariable=self.settings['ema_decay'], width=8)
        spin_ema.pack(side=tk.RIGHT)

        # Preview
        preview_frame = tk.LabelFrame(scrollable_frame, text="Preview During Training", padx=10, pady=10)
        preview_frame.pack(fill=tk.X, pady=5)
        f = tk.Frame(preview_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Enable preview:", width=20, anchor='w').pack(side=tk.LEFT)
        cb_preview = tk.Checkbutton(f, variable=self.settings['preview_enabled'])
        cb_preview.pack(side=tk.RIGHT)
        f = tk.Frame(preview_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Every N epochs:", width=20, anchor='w').pack(side=tk.LEFT)
        spin_freq = ttk.Spinbox(f, from_=1, to=50, textvariable=self.settings['preview_epoch_freq'], width=8)
        spin_freq.pack(side=tk.RIGHT)
        f = tk.Frame(preview_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Preview steps:", width=20, anchor='w').pack(side=tk.LEFT)
        spin_steps = ttk.Spinbox(f, from_=1, to=200, textvariable=self.settings['preview_steps'], width=8)
        spin_steps.pack(side=tk.RIGHT)
        f = tk.Frame(preview_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Preview ODE method:", width=20, anchor='w').pack(side=tk.LEFT)
        method_combo = ttk.Combobox(f, textvariable=self.settings['preview_method'],
                                    values=['euler', 'heun', 'rk3', 'rk4'], state='readonly', width=10)
        method_combo.pack(side=tk.RIGHT)

        # Text conditioning settings
        cond_frame = tk.LabelFrame(scrollable_frame, text="Text Conditioning", padx=10, pady=10)
        cond_frame.pack(fill=tk.X, pady=5)
        f = tk.Frame(cond_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Encoder type:", width=20, anchor='w').pack(side=tk.LEFT)
        type_combo2 = ttk.Combobox(f, textvariable=self.settings['text_encoder_type'],
                                   values=['BiGRU', 'BiTransformer'], state='readonly', width=12)
        type_combo2.pack(side=tk.RIGHT)
        f = tk.Frame(cond_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Encoder size preset:", width=20, anchor='w').pack(side=tk.LEFT)
        size_combo2 = ttk.Combobox(f, textvariable=self.settings['text_encoder_size'],
                                   values=['tiny', 'small', 'medium', 'large'], state='readonly', width=8)
        size_combo2.pack(side=tk.RIGHT)
        tk.Button(cond_frame, text="Apply preset to dims", command=self.apply_text_encoder_preset).pack(pady=2)
        cond_params = [
            ("Embed dim:", 'cond_embed_dim', 32, 256, int),
            ("Hidden size (GRU):", 'cond_hidden_size', 32, 512, int),
            ("Num layers:", 'cond_num_layers', 1, 4, int),
            ("Num heads (Transformer):", 'cond_num_heads', 1, 16, int),
            ("FF dim (Transformer):", 'cond_ff_dim', 64, 1024, int),
            ("Conditioning dim:", 'cond_dim', 64, 512, int),
            ("Max text len:", 'cond_text_max_len', 32, 256, int),
        ]
        for label, key, low, high, typ in cond_params:
            f = tk.Frame(cond_frame); f.pack(fill=tk.X, pady=2)
            tk.Label(f, text=label, width=22, anchor='w').pack(side=tk.LEFT)
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

        # System info
        sys_frame = tk.LabelFrame(scrollable_frame, text="System", padx=10, pady=10)
        sys_frame.pack(fill=tk.X, pady=5)
        cpu = multiprocessing.cpu_count()
        tk.Label(sys_frame, text=f"CPU cores: {cpu}").pack(anchor='w')
        tk.Label(sys_frame, text=f"PyTorch threads: {torch.get_num_threads()}").pack(anchor='w')
        tk.Label(sys_frame, text=f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}").pack(anchor='w')

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def update_model_params_visibility(self):
        if self.settings['model_type'].get() == 'uvit':
            self.uvit_frame.pack(fill=tk.X, pady=5)
            self.ugru_frame.pack_forget()
        else:
            self.uvit_frame.pack_forget()
            self.ugru_frame.pack(fill=tk.X, pady=5)

    def apply_preset(self):
        model_type = self.settings['model_type'].get()
        preset = self.preset_var.get()
        if model_type == 'uvit':
            if preset in PIXEL_UVIT_PRESETS:
                cfg = PIXEL_UVIT_PRESETS[preset]
                self.settings['uvit_embed_dim'].set(cfg['embed_dim'])
                self.settings['uvit_num_heads'].set(cfg['num_heads'])
                self.settings['uvit_num_layers'].set(cfg['num_layers'])
                self.settings['uvit_mlp_ratio'].set(cfg['mlp_ratio'])
                self.log_rectified(f"Applied U-ViT preset '{preset}'")
            else:
                self.log_rectified(f"Unknown preset {preset}")
        else:
            if preset in PIXEL_UGRU_PRESETS:
                cfg = PIXEL_UGRU_PRESETS[preset]
                self.settings['ugru_embed_dim'].set(cfg['embed_dim'])
                self.settings['ugru_gru_hidden_dim'].set(cfg['gru_hidden_dim'])
                self.settings['ugru_gru_num_layers'].set(cfg['gru_num_layers'])
                self.log_rectified(f"Applied U-GRU preset '{preset}'")
            else:
                self.log_rectified(f"Unknown preset {preset}")

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
        self.log_rectified(f"Applied {enc_type} {size} preset dims.")

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
                                    values=['euler', 'heun', 'rk3', 'rk4'], state='readonly', width=8)
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

    # ------------------------------------------------------------------
    # Core functions
    # ------------------------------------------------------------------
    def log_dataset(self, msg):
        self.message_queue_dataset.put(msg)

    def log_rectified(self, msg):
        self.message_queue_rectified.put(msg)

    def process_messages_dataset(self):
        try:
            while True:
                msg = self.message_queue_dataset.get_nowait()
                self.dataset_log_text.insert(tk.END, f"{time.strftime('%H:%M:%S')} - {msg}\n")
                self.dataset_log_text.see(tk.END)
                self.status_label.config(text=msg[:50])
                print(f"Dataset: {msg}")
        except queue.Empty:
            pass
        self.root.after(100, self.process_messages_dataset)

    def process_messages_rectified(self):
        try:
            while True:
                msg = self.message_queue_rectified.get_nowait()
                self.rectified_log_text.insert(tk.END, f"{time.strftime('%H:%M:%S')} - {msg}\n")
                self.rectified_log_text.see(tk.END)
                self.status_label.config(text=msg[:50])
                print(f"Rectified: {msg}")
        except queue.Empty:
            pass
        self.root.after(100, self.process_messages_rectified)

    def add_images(self):
        files = filedialog.askopenfilenames(filetypes=[("Images", "*.jpg *.jpeg *.png *.jfif *.webp *.bmp")])
        for f in files:
            if f not in self.image_paths:
                self.image_paths.append(f)
                self.image_listbox.insert(tk.END, os.path.basename(f))
        self.log_dataset(f"Added {len(files)} images. Total: {len(self.image_paths)}")
        self.log_rectified(f"Added {len(files)} images. Total: {len(self.image_paths)}")

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
        self.log_dataset(f"Added {count} images from folder. Total: {len(self.image_paths)}")
        self.log_rectified(f"Added {count} images from folder. Total: {len(self.image_paths)}")

    def clear_images(self):
        self.image_paths = []
        self.labels = []
        self.image_listbox.delete(0, tk.END)
        self.csv_status.config(text="No CSV loaded", fg="red")
        self.log_dataset("Cleared all images")
        self.log_rectified("Cleared all images")

    def _match_csv_row(self, row_name: str) -> str:
        norm = row_name.replace('\\', '/').lower()
        for full in self.image_paths:
            if full.replace('\\', '/').lower().endswith(norm):
                return full
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
        path_by_basename = {}
        for full in self.image_paths:
            base = os.path.basename(full).lower()
            if base in path_by_basename:
                self.log_rectified(f"Warning: duplicate basename '{base}'")
            else:
                path_by_basename[base] = full
        label_map = {}
        try:
            with open(fname, 'r', encoding='utf-8-sig') as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row or len(row) < 2:
                        continue
                    img_name = row[0].strip().strip('"').strip("'")
                    label = row[1].strip().strip('"').strip("'")
                    base = os.path.basename(img_name).lower()
                    matched = path_by_basename.get(base, None)
                    if matched:
                        label_map.setdefault(matched, []).append(label)
        except Exception as e:
            self.log_rectified(f"Error reading CSV: {e}")
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
        self.log_rectified(f"CSV loaded. {len(self.image_paths)-unknown} images matched with labels.")

    def use_filenames_as_labels(self):
        self.labels = [[os.path.splitext(os.path.basename(p))[0]] for p in self.image_paths]
        self.csv_status.config(text="Using filenames as labels", fg="blue")
        self.log_rectified("Using filenames as labels.")

    def use_folders_as_labels(self):
        self.labels = []
        for p in self.image_paths:
            folder = os.path.basename(os.path.dirname(p)) or 'unknown'
            self.labels.append([folder])
        self.csv_status.config(text="Using folder names as labels", fg="blue")
        self.log_rectified("Using folder names as labels.")

    # ========== Rectified Flow Methods ==========
    def initialize_rectified_model(self):
        try:
            model_type = self.settings['model_type'].get()
            cond_enabled = self.settings['cond_enabled'].get()
            cond_dim = self.settings['cond_dim'].get() if cond_enabled else 1
            in_channels = 3 if self.settings['color_mode'].get() == 'rgb' else 1
            img_size = self.settings['img_size'].get()
            patch_size = self.settings['patch_size'].get()
            use_final_conv = self.settings['uvit_final_conv'].get() if model_type == 'uvit' else True

            if img_size % patch_size != 0:
                self.log_rectified(f"Image size {img_size} not divisible by patch size {patch_size}.")
                return

            # Get UViT or U-GRU specific params
            if model_type == 'uvit':
                embed_dim = self.settings['uvit_embed_dim'].get()
                num_heads = self.settings['uvit_num_heads'].get()
                num_layers = self.settings['uvit_num_layers'].get()
                mlp_ratio = self.settings['uvit_mlp_ratio'].get()
                gru_hidden_dim = None
                gru_num_layers = None
                if num_layers % 2 == 0:
                    self.log_rectified("Warning: U-ViT requires odd number of layers. Adjusting to next odd.")
                    num_layers += 1
                    self.settings['uvit_num_layers'].set(num_layers)
            else:  # ugru
                embed_dim = self.settings['ugru_embed_dim'].get()
                gru_hidden_dim = self.settings['ugru_gru_hidden_dim'].get()
                gru_num_layers = self.settings['ugru_gru_num_layers'].get()
                num_heads = None
                num_layers = None
                mlp_ratio = None
                if gru_num_layers % 2 == 0:
                    self.log_rectified("Warning: U-GRU requires odd number of layers. Adjusting to next odd.")
                    gru_num_layers += 1
                    self.settings['ugru_gru_num_layers'].set(gru_num_layers)

            self.rectified_model = RectifiedFlowPixel(
                model_type=model_type,
                in_channels=in_channels,
                img_size=img_size,
                patch_size=patch_size,
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                mlp_ratio=mlp_ratio,
                gru_hidden_dim=gru_hidden_dim,
                gru_num_layers=gru_num_layers,
                dropout=self.settings['dropout'].get(),
                time_emb_dim=self.settings['time_emb_dim'].get(),
                cond_dim=cond_dim,
                use_final_conv=use_final_conv
            )
            for pg in self.rectified_model.optimizer.param_groups:
                pg['lr'] = self.settings['rectified_lr'].get()

            if self.settings['ema_enabled'].get():
                self.rectified_model.set_ema(decay=self.settings['ema_decay'].get())
                self.log_rectified(f"EMA enabled with decay {self.settings['ema_decay'].get()}")
            else:
                self.rectified_model.ema = None

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
                self.text_encoder.to(self.rectified_model.device)
                self.rectified_optimizer = optim.Adam(
                    list(self.rectified_model.model.parameters()) + list(self.text_encoder.parameters()),
                    lr=self.settings['rectified_lr'].get()
                )
                self.rectified_model.optimizer = self.rectified_optimizer
                self.log_rectified(f"Conditional {model_type.upper()} initialized with {enc_type} encoder.")
            else:
                self.text_encoder = None
                self.log_rectified(f"Unconditional {model_type.upper()} initialized.")
        except Exception as e:
            self.log_rectified(f"Error initializing rectified model: {e}")

    def start_rectified_training(self):
        if not self.image_paths:
            self.log_rectified("No images!")
            return
        if not self.rectified_model:
            self.log_rectified("Rectified model not initialized!")
            return
        if self.training_rectified:
            self.log_rectified("Already training.")
            return

        cond_enabled = self.settings['cond_enabled'].get()
        if cond_enabled and not self.labels:
            self.log_rectified("Conditional training requires labels. Load CSV or use filenames/folders as labels.")
            return

        try:
            epochs = int(self.rectified_epoch_var.get())
        except:
            self.log_rectified("Invalid epochs")
            return

        self.training_rectified = True
        self.current_epoch_rectified = 0
        self.rectified_start_time = time.time()

        thread = threading.Thread(target=self.train_rectified_loop, args=(epochs,), daemon=True)
        thread.start()
        self.log_rectified(f"Rectified Flow training started for {epochs} epochs.")

    def train_rectified_loop(self, epochs):
        try:
            batch_size = self.settings['rectified_batch_size'].get()
            num_workers = 0
            img_size = self.settings['img_size'].get()
            color_mode = self.settings['color_mode'].get()
            cond_enabled = self.settings['cond_enabled'].get()
            text_max_len = self.settings['cond_text_max_len'].get()
            cfg_dropout = self.settings['cfg_dropout_prob'].get()
            use_ema = self.settings['ema_enabled'].get()
            ema_decay = self.settings['ema_decay'].get()
            preview_enabled = self.settings['preview_enabled'].get()
            preview_freq = self.settings['preview_epoch_freq'].get()

            aug_dict = {k: v.get() for k, v in self.aug_settings.items()}
            labels_for_dataset = self.labels if self.labels else [['']]*len(self.image_paths)
            dataset = ConditionalImageDataset(self.image_paths, labels_for_dataset,
                                              img_size, color_mode, aug_dict, text_max_len)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                num_workers=num_workers, pin_memory=False)

            for epoch in range(epochs):
                if not self.training_rectified:
                    break
                self.current_epoch_rectified = epoch
                epoch_loss = 0.0
                batches = 0

                for images, text_tensors in loader:
                    if not self.training_rectified:
                        break
                    x1 = images.to(self.rectified_model.device)
                    x0 = torch.randn_like(x1)
                    if cond_enabled and self.text_encoder is not None:
                        cond = self.text_encoder(text_tensors.to(self.rectified_model.device))
                    else:
                        cond = None
                    loss = self.rectified_model.train_step(x0, x1, cond=cond,
                                                           cfg_dropout_prob=cfg_dropout,
                                                           update_ema=use_ema,
                                                           ema_decay=ema_decay)
                    epoch_loss += loss
                    batches += 1

                avg_loss = epoch_loss / batches if batches else 0
                elapsed = time.time() - self.rectified_start_time
                self.log_rectified(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f} | Time: {elapsed:.1f}s")

                if preview_enabled and (epoch+1) % preview_freq == 0:
                    self.rectified_preview_with_prompt(use_ema=use_ema)

            self.training_rectified = False
            self.log_rectified("Rectified Flow training finished.")
        except Exception as e:
            self.log_rectified(f"Training error: {e}")
            import traceback
            traceback.print_exc()
            self.training_rectified = False

    def rectified_preview_with_prompt(self, use_ema=None):
        if not self.rectified_model:
            self.log_rectified("Model not loaded for preview")
            return
        if use_ema is None:
            use_ema = self.settings['ema_enabled'].get()
        try:
            prompt = self.test_prompt_entry.get().strip()
            unconditional = (prompt == "")
            cond = None
            cfg_scale = 1.0 if unconditional else self.cfg_scale.get()
            steps = self.settings['preview_steps'].get()
            method = self.settings['preview_method'].get()

            if not unconditional and self.settings['cond_enabled'].get() and self.text_encoder is not None:
                text_indices = text_to_indices(prompt, self.settings['cond_text_max_len'].get())
                text_tensor = torch.tensor([text_indices] * 16, dtype=torch.long, device=self.rectified_model.device)
                with torch.no_grad():
                    cond = self.text_encoder(text_tensor)
                self.log_rectified(f"Preview: conditional with prompt '{prompt}'")
            elif not unconditional:
                self.log_rectified("Preview: conditioning disabled, generating unconditional")
                unconditional = True
            else:
                self.log_rectified("Preview: unconditional generation")

            samples = self.rectified_model.sample(n_samples=16, steps=steps, method=method,
                                                  cond=cond, cfg_scale=cfg_scale, use_ema=use_ema)
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
            self.rectified_preview_photo = ImageTk.PhotoImage(grid)
            self.rectified_preview_canvas.delete("all")
            self.rectified_preview_canvas.create_image(128,128, image=self.rectified_preview_photo)
        except Exception as e:
            self.log_rectified(f"Preview error: {e}")

    def stop_rectified_training(self):
        self.training_rectified = False
        self.log_rectified("Training stopped.")

    def save_rectified_model(self):
        if not self.rectified_model:
            self.log_rectified("No rectified model.")
            return
        fname = filedialog.asksaveasfilename(defaultextension=".pth", filetypes=[("PyTorch","*.pth")])
        if fname:
            save_dict = {
                'model_state': self.rectified_model.model.state_dict(),
                'optimizer_state': self.rectified_model.optimizer.state_dict(),
                'settings': {k:v.get() for k,v in self.settings.items()},
                'model_type': self.settings['model_type'].get(),
                'ema_shadow': self.rectified_model.ema.shadow if self.rectified_model.ema else None,
            }
            if self.text_encoder is not None:
                save_dict['text_encoder_state'] = self.text_encoder.state_dict()
            torch.save(save_dict, fname)
            self.log_rectified(f"Model saved to {fname}")

    def load_rectified_model(self):
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
            self.initialize_rectified_model()
            self.rectified_model.model.load_state_dict(ckpt['model_state'])
            self.rectified_model.optimizer.load_state_dict(ckpt['optimizer_state'])
            if 'text_encoder_state' in ckpt and self.text_encoder is not None:
                self.text_encoder.load_state_dict(ckpt['text_encoder_state'])
            if 'ema_shadow' in ckpt and self.rectified_model.ema is not None:
                self.rectified_model.ema.shadow = ckpt['ema_shadow']
                self.log_rectified("EMA shadow weights restored.")
            self.log_rectified(f"Model loaded from {fname}")
        except Exception as e:
            self.log_rectified(f"Load error: {e}")

    # ========== Generation Methods ==========
    def generate_samples(self):
        if not self.rectified_model:
            self.gen_info.config(text="Model not loaded!")
            return
        if self.settings['cond_enabled'].get() and self.text_encoder is None:
            self.gen_info.config(text="Text encoder not initialized! Load conditional model.")
            return

        n = self.gen_count.get()
        steps = self.ode_steps.get()
        method = self.ode_method.get()
        prompt = self.gen_prompt.get().strip()
        cfg_scale_user = self.cfg_scale.get()
        use_ema = self.settings['ema_enabled'].get()

        unconditional = (prompt == "")
        cond = None
        effective_cfg_scale = 1.0 if unconditional else cfg_scale_user

        if not unconditional and self.settings['cond_enabled'].get() and self.text_encoder is not None:
            text_indices = text_to_indices(prompt, self.settings['cond_text_max_len'].get())
            text_tensor = torch.tensor([text_indices] * n, dtype=torch.long, device=self.rectified_model.device)
            with torch.no_grad():
                cond = self.text_encoder(text_tensor)
        elif not unconditional:
            unconditional = True
            effective_cfg_scale = 1.0

        if self.progressive_grid.get():
            self.start_progressive(n, steps, method, cond, effective_cfg_scale, use_ema)
        else:
            self.generate_btn.config(state=tk.DISABLED)
            if unconditional:
                self.gen_info.config(text="Generating unconditionally...")
            else:
                self.gen_info.config(text=f"Generating with prompt: {prompt}")
            self.root.update()
            thread = threading.Thread(target=self._generate_thread,
                                      args=(n, steps, method, cond, effective_cfg_scale, use_ema), daemon=True)
            thread.start()

    def stop_progressive(self):
        self.progressive_active = False
        self.stop_prog_btn.config(state=tk.DISABLED)
        self.generate_btn.config(state=tk.NORMAL)
        self.gen_info.config(text="Progressive generation stopped.")

    def start_progressive(self, n, steps, method, cond, cfg_scale, use_ema):
        self.progressive_active = True
        self.generate_btn.config(state=tk.DISABLED)
        self.stop_prog_btn.config(state=tk.NORMAL)
        self.gen_info.config(text="Progressive generation...")
        thread = threading.Thread(target=self._progressive_thread,
                                  args=(n, steps, method, cond, cfg_scale, use_ema), daemon=True)
        thread.start()

    def _generate_thread(self, n, steps, method, cond, cfg_scale, use_ema):
        try:
            samples = self.rectified_model.sample(n_samples=n, steps=steps, method=method,
                                                  cond=cond, cfg_scale=cfg_scale, use_ema=use_ema)
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

    def _progressive_thread(self, n, steps, method, cond, cfg_scale, use_ema):
        try:
            thumb = self.thumbnail_size
            interval = self.prog_interval.get()
            generator = self.rectified_model.sample_step_by_step(n_samples=n, steps=steps, method=method,
                                                                 cond=cond, cfg_scale=cfg_scale, use_ema=use_ema)
            for step_idx, x in generator:
                if not self.progressive_active:
                    break
                if step_idx % interval == 0 or step_idx == steps:
                    samples = (x + 1) / 2
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
    app = RectifiedFlowApp(root)
    root.mainloop()