# rectified_flow.py
# Unified Rectified Flow with adaptive GroupNorm, extended augmentations, configurable padding mode,
# fixed EMA (target model), two EMA modes (Standard / Lookahead), live parameter tuning,
# Optimal Transport matching, source noise addition, pixelation rescale modes,
# and dataset‑aware preview (reconstruction on degraded dataset images).
# Compatible with flow_matching.py (legacy models load correctly).
import tkinter as tk
from tkinter import filedialog, ttk
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as F_vision
import torch.nn.functional as F_nn
from PIL import Image, ImageTk, ImageEnhance, ImageFilter
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
import copy
import io
from scipy.optimize import linear_sum_assignment   # for OT matching

# ==================== Helper ====================

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
    indices = [ord(ch) if ord(ch) < 256 else 0 for ch in text[:max_len]]
    if len(indices) < max_len:
        indices += [0] * (max_len - len(indices))
    return indices

def get_norm(channels):
    """GroupNorm with up to 32 groups, fallback to largest divisor if channels not divisible by 32."""
    if channels % 32 == 0:
        return nn.GroupNorm(32, channels)
    best = 1
    for g in range(32, 0, -1):
        if channels % g == 0:
            best = g
            break
    return nn.GroupNorm(best, channels)

def gaussian_blur_tensor(x, sigma):
    size = int(2 * round(3 * sigma) + 1)
    if size % 2 == 0:
        size += 1
    kernel_1d = torch.exp(-torch.arange(-size//2+1, size//2+1)**2 / (2*sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    kernel = kernel_2d[None, None, :, :].to(x.device).repeat(x.size(1), 1, 1, 1)
    pad = size // 2
    x_pad = F_nn.pad(x, (pad, pad, pad, pad), mode='reflect')
    return F_nn.conv2d(x_pad, kernel, groups=x.size(1))

# ==================== Augmentations (using replicate border) ====================

class RandomJPEG:
    """Simulate JPEG compression artifacts by quantizing DCT (simplified)."""
    def __init__(self, quality_low=50, quality_high=95, p=0.5):
        self.quality_low = quality_low
        self.quality_high = quality_high
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        quality = random.randint(self.quality_low, self.quality_high)
        if not isinstance(img, Image.Image):
            to_pil = transforms.ToPILImage()
            img_pil = to_pil(img)
        else:
            img_pil = img
        buffer = io.BytesIO()
        img_pil.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        compressed = Image.open(buffer).convert('RGB')
        if isinstance(img, torch.Tensor):
            return transforms.ToTensor()(compressed)
        return compressed

class ElasticTransform:
    """Elastic deformation with replicate padding to avoid dark borders."""
    def __init__(self, alpha=30, sigma=3, p=0.5):
        self.alpha = alpha
        self.sigma = sigma
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        if isinstance(img, torch.Tensor):
            img_pil = transforms.ToPILImage()(img)
        else:
            img_pil = img
        w, h = img_pil.size
        dx = torch.randn(1, h, w) * self.sigma
        dy = torch.randn(1, h, w) * self.sigma
        kernel = torch.ones(1, 1, 5, 5) / 25
        dx = F_nn.conv2d(dx.view(1,1,h,w), kernel, padding=2).view(h,w) * self.alpha
        dy = F_nn.conv2d(dy.view(1,1,h,w), kernel, padding=2).view(h,w) * self.alpha
        x, y = torch.meshgrid(torch.arange(w), torch.arange(h), indexing='xy')
        x = x.float() + dx
        y = y.float() + dy
        x = (x / (w-1)) * 2 - 1
        y = (y / (h-1)) * 2 - 1
        grid = torch.stack([x, y], dim=-1).unsqueeze(0)
        img_tensor = transforms.ToTensor()(img_pil).unsqueeze(0)
        deformed = F_nn.grid_sample(img_tensor, grid, mode='bilinear', padding_mode='border')
        return transforms.ToPILImage()(deformed.squeeze(0))

# ==================== Sinusoidal Embeddings ====================
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = time[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)

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
        return self.fc(torch.cat([h_fwd, h_bwd], dim=1))

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
        x = self.transformer(emb + pos)
        return self.fc(x.mean(dim=1))

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

def get_encoder_config(enc_type, size):
    return TEXT_ENCODER_PRESETS[enc_type][size]

# ==================== UNet with configurable padding mode ====================
class Conv2dWithPadding(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding_mode='constant', **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, **kwargs)
        self.padding_mode = padding_mode

    def forward(self, x):
        if self.kernel_size[0] % 2 == 1:
            pad_h = self.kernel_size[0] // 2
            pad_w = self.kernel_size[1] // 2
        else:
            pad_h = self.kernel_size[0] // 2
            pad_w = self.kernel_size[1] // 2
        x = F_nn.pad(x, (pad_w, pad_w, pad_h, pad_h), mode=self.padding_mode)
        return F_nn.conv2d(x, self.weight, self.bias, self.stride, 0, self.dilation, self.groups)

class AttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=4, padding_mode='constant'):
        super().__init__()
        self.norm = get_norm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        residual = x
        x = self.norm(x)
        x = x.view(B, C, H * W).transpose(1, 2)
        x, _ = self.attn(x, x, x)
        x = x.transpose(1, 2).view(B, C, H, W)
        return x + residual

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, cond_dim, dropout=0.1,
                 has_attn=False, padding_mode='constant'):
        super().__init__()
        self.conv1 = Conv2dWithPadding(in_channels, out_channels, 3, padding_mode=padding_mode)
        self.norm1 = get_norm(out_channels)
        self.conv2 = Conv2dWithPadding(out_channels, out_channels, 3, padding_mode=padding_mode)
        self.norm2 = get_norm(out_channels)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels))
        self.cond_proj = nn.Linear(cond_dim, out_channels)
        self.res_conv = Conv2dWithPadding(in_channels, out_channels, 1, padding_mode=padding_mode) if in_channels != out_channels else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.attn = AttentionBlock(out_channels, padding_mode=padding_mode) if has_attn else nn.Identity()

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
        return self.attn(h + self.res_conv(x))

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, cond_dim, dropout=0.1,
                 has_attn=False, padding_mode='constant'):
        super().__init__()
        self.conv1 = Conv2dWithPadding(in_channels, out_channels, 3, padding_mode=padding_mode)
        self.norm1 = get_norm(out_channels)
        self.conv2 = Conv2dWithPadding(out_channels, out_channels, 3, padding_mode=padding_mode)
        self.norm2 = get_norm(out_channels)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels))
        self.cond_proj = nn.Linear(cond_dim, out_channels)
        self.res_conv = Conv2dWithPadding(in_channels, out_channels, 1, padding_mode=padding_mode) if in_channels != out_channels else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.attn = AttentionBlock(out_channels, padding_mode=padding_mode) if has_attn else nn.Identity()

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
        return self.attn(h + self.res_conv(x))

class VelocityUNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, time_emb_dim=256, cond_dim=256,
                 img_size=32, channel_mult=(1, 2, 3, 4), dropout=0.1, use_attention=True,
                 padding_mode='constant'):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.time_emb_dim = time_emb_dim
        self.cond_dim = cond_dim
        self.channel_mult = channel_mult
        self.use_attention = use_attention
        self.padding_mode = padding_mode

        H, W = (img_size, img_size) if isinstance(img_size, int) else img_size
        num_down = 0
        cur_h, cur_w = H, W
        while cur_h >= 8 and cur_w >= 8:
            cur_h //= 2
            cur_w //= 2
            num_down += 1
        num_down = min(num_down, len(channel_mult))
        channel_mult_used = channel_mult[:num_down+1]

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        self.init_conv = Conv2dWithPadding(in_channels, base_channels, 3, padding_mode=padding_mode)

        self.downs = nn.ModuleList()
        cur_channels = base_channels
        for i, mult in enumerate(channel_mult_used):
            out_channels = base_channels * mult
            block = DownBlock(cur_channels, out_channels, time_emb_dim, cond_dim, dropout,
                              has_attn=use_attention, padding_mode=padding_mode)
            self.downs.append(block)
            if i < len(channel_mult_used) - 1:
                self.downs.append(nn.Conv2d(out_channels, out_channels, 4, stride=2, padding=1))
            cur_channels = out_channels

        self.mid_block1 = DownBlock(cur_channels, cur_channels, time_emb_dim, cond_dim, dropout,
                                    has_attn=use_attention, padding_mode=padding_mode)
        self.mid_block2 = UpBlock(cur_channels, cur_channels, time_emb_dim, cond_dim, dropout,
                                  has_attn=use_attention, padding_mode=padding_mode)

        self.ups = nn.ModuleList()
        rev_blocks = list(reversed(channel_mult_used))
        for i, mult in enumerate(rev_blocks):
            out_channels = base_channels * mult
            block = UpBlock(cur_channels + out_channels, out_channels, time_emb_dim, cond_dim, dropout,
                            has_attn=use_attention, padding_mode=padding_mode)
            self.ups.append(block)
            if i < len(rev_blocks) - 1:
                self.ups.append(nn.ConvTranspose2d(out_channels, out_channels, 4, stride=2, padding=1))
            cur_channels = out_channels

        self.final_conv = nn.Sequential(
            get_norm(cur_channels),
            nn.SiLU(),
            Conv2dWithPadding(cur_channels, in_channels, 3, padding_mode=padding_mode)
        )

    def forward(self, x, t, cond=None):
        t_emb = self.time_mlp(t)
        x = self.init_conv(x)
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
                skip = skips.pop()
                x = torch.cat([x, skip], dim=1)
                x = layer(x, t_emb, cond)
            else:
                x = layer(x)
        return self.final_conv(x)

# ==================== Rectified Flow Model (with fixed EMA and OT) ====================
@torch.no_grad()
def match_ot_batch(x0, x1):
    """
    x0: source batch (B, C, H, W)
    x1: target batch (B, C, H, W)
    Returns reordered x1 so that each x0 is paired with the closest x1.
    Also returns the column indices.
    """
    B = x0.shape[0]
    if B <= 1:
        return x0, x1, torch.arange(B)
    # Flatten
    x0_flat = x0.view(B, -1)
    x1_flat = x1.view(B, -1)
    # Pairwise squared distances: (B, B)
    dist = (torch.sum(x0_flat**2, dim=1, keepdim=True) +
            torch.sum(x1_flat**2, dim=1, keepdim=True).T -
            2.0 * torch.mm(x0_flat, x1_flat.T))
    cost = dist.cpu().numpy()
    _, col_ind = linear_sum_assignment(cost)
    col_ind = torch.from_numpy(col_ind).to(x1.device)
    return x0, x1[col_ind], col_ind

class RectifiedFlowPixel:
    def __init__(self, in_channels=3, img_size=32, base_channels=64,
                 cond_dim=256, device=None, use_attention=True,
                 task="noise", task_params=None, training_mode="dataset",
                 padding_mode='constant'):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.cond_dim = cond_dim
        self.task = task
        self.task_params = task_params or {}
        self.training_mode = training_mode
        self.in_channels = in_channels

        # OT matching flag
        self.use_ot = False

        self.model = VelocityUNet(
            in_channels=in_channels,
            base_channels=base_channels,
            time_emb_dim=base_channels*4,
            cond_dim=cond_dim,
            img_size=img_size,
            channel_mult=(1, 2, 3, 4),
            dropout=0.1,
            use_attention=use_attention,
            padding_mode=padding_mode
        ).to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=2e-4)

        # EMA specific attributes
        self.ema_model = None          # target model (EMA)
        self.ema_decay = 0.999
        self.ema_mode = 'standard'     # 'standard' or 'lookahead'
        self.use_ema = False           # whether EMA is enabled

    def set_ema(self, decay=0.999, mode='standard'):
        """Create EMA target model and set decay/mode."""
        self.ema_decay = decay
        self.ema_mode = mode
        self.use_ema = True
        # Create a copy of the model
        self.ema_model = VelocityUNet(
            in_channels=self.model.in_channels,
            base_channels=self.model.base_channels,
            time_emb_dim=self.model.time_emb_dim,
            cond_dim=self.model.cond_dim,
            img_size=self.img_size,
            channel_mult=self.model.channel_mult,
            dropout=0.1,
            use_attention=self.model.use_attention,
            padding_mode=self.model.padding_mode
        ).to(self.device)
        # Copy weights
        self.ema_model.load_state_dict(self.model.state_dict())
        # Freeze ema_model (no gradients)
        for param in self.ema_model.parameters():
            param.requires_grad = False

    def update_ema(self):
        """Update EMA target model from online model."""
        if self.ema_model is None:
            return
        with torch.no_grad():
            for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)
        if self.ema_mode == 'lookahead':
            # Copy EMA weights back to online model
            with torch.no_grad():
                for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
                    param.data.copy_(ema_param.data)

    def _add_noise_to_source(self, x0):
        """Add Gaussian noise to source if enabled."""
        noise_std = self.task_params.get('source_noise_std', 0.0)
        if noise_std <= 0.0:
            return x0
        # Range is 2 (since images are in [-1,1]), so std = noise_std * 2
        std = noise_std * 2.0
        noise = torch.randn_like(x0) * std
        return x0 + noise

    def _generate_source_from_target(self, x1):
        batch_size = x1.size(0)
        if self.task == "noise":
            x0 = torch.randn_like(x1)
        elif self.task == "grayscale_to_color":
            gray = 0.2989 * x1[:, 0:1] + 0.5870 * x1[:, 1:2] + 0.1140 * x1[:, 2:3]
            x0 = gray.repeat(1, 3, 1, 1)
        elif self.task == "blur_to_sharp":
            sigma = self.task_params.get('blur_sigma', 2.0)
            x0 = gaussian_blur_tensor(x1, sigma)
        elif self.task == "white_to_image":
            x0 = torch.full_like(x1, 1.0)
        elif self.task == "average_separation":
            indices = torch.randperm(batch_size, device=x1.device)
            x1_shuffled = x1[indices]
            x0 = (x1 + x1_shuffled) / 2.0
        elif self.task == "unquantization":
            bit_depth = self.task_params.get('bit_depth', 1)
            levels = 2 ** bit_depth
            x1_01 = (x1 + 1) / 2
            quantized = torch.floor(x1_01 * levels) / (levels - 1 + 1e-8)
            x0 = quantized * 2 - 1
        elif self.task == "pixelation":
            block = self.task_params.get('pixelation_size', 8)
            mode = self.task_params.get('pixelation_rescale_mode', 'nearest')
            H, W = x1.shape[2], x1.shape[3]
            H_new = (H // block) * block
            W_new = (W // block) * block
            x_cropped = x1[:, :, :H_new, :W_new]
            # Downsample
            down = F_nn.avg_pool2d(x_cropped, block, stride=block)
            # Upsample with chosen mode
            up = F_nn.interpolate(down, size=(H_new, W_new), mode=mode)
            if (H_new, W_new) != (H, W):
                up = F_nn.pad(up, (0, W - W_new, 0, H - H_new))
            x0 = up
        else:
            raise ValueError(f"Unknown task {self.task}")
        # Add noise if enabled (only if task is not "noise" to avoid double noise)
        if self.task != "noise":
            x0 = self._add_noise_to_source(x0)
        return x0

    def _generate_source_from_noise(self, x1_shape):
        noise = torch.randn(x1_shape, device=self.device)
        if self.task == "noise":
            x0 = noise
        elif self.task == "grayscale_to_color":
            gray = torch.randn(x1_shape[0], 1, x1_shape[2], x1_shape[3], device=self.device)
            x0 = gray.repeat(1, 3, 1, 1)
        elif self.task == "blur_to_sharp":
            sigma = self.task_params.get('blur_sigma', 2.0)
            x0 = gaussian_blur_tensor(noise, sigma)
        elif self.task == "white_to_image":
            x0 = torch.full(x1_shape, 1.0, device=self.device)
        elif self.task == "average_separation":
            x0 = noise
        elif self.task == "unquantization":
            bit_depth = self.task_params.get('bit_depth', 1)
            levels = 2 ** bit_depth
            noise_01 = (noise + 1) / 2
            quantized = torch.floor(noise_01 * levels) / (levels - 1 + 1e-8)
            x0 = quantized * 2 - 1
        elif self.task == "pixelation":
            block = self.task_params.get('pixelation_size', 8)
            mode = self.task_params.get('pixelation_rescale_mode', 'nearest')
            H, W = x1_shape[2], x1_shape[3]
            H_new = (H // block) * block
            W_new = (W // block) * block
            noise_cropped = noise[:, :, :H_new, :W_new]
            down = F_nn.avg_pool2d(noise_cropped, block, stride=block)
            up = F_nn.interpolate(down, size=(H_new, W_new), mode=mode)
            if (H_new, W_new) != (H, W):
                up = F_nn.pad(up, (0, W - W_new, 0, H - H_new))
            x0 = up
        else:
            x0 = noise
        # Add noise if enabled (but if we are in noise mode, it's already noise, skip)
        if self.task != "noise":
            x0 = self._add_noise_to_source(x0)
        return x0

    def train_step(self, x1, cond=None, cfg_dropout_prob=0.0):
        x1 = x1.to(self.device)
        batch_size = x1.size(0)

        # Generate source x0
        if self.training_mode == "dataset":
            x0 = self._generate_source_from_target(x1)
        elif self.training_mode == "noise":
            x0 = self._generate_source_from_noise(x1.shape)
        else:  # mixed
            if random.random() < 0.5:
                x0 = self._generate_source_from_target(x1)
            else:
                x0 = self._generate_source_from_noise(x1.shape)

        # Optional OT matching: reorder x1 (and cond) to match x0
        if self.use_ot and batch_size > 1:
            x0, x1, col_ind = match_ot_batch(x0, x1)
            if cond is not None:
                cond = cond[col_ind]

        if cond is None:
            cond = torch.zeros(batch_size, self.cond_dim, device=self.device)
        else:
            cond = cond.to(self.device)

        t = torch.rand(batch_size, device=self.device)
        x_t = t.view(-1,1,1,1) * x1 + (1 - t.view(-1,1,1,1)) * x0
        target = x1 - x0

        if cfg_dropout_prob > 0:
            mask = torch.rand(batch_size, 1, device=self.device) > cfg_dropout_prob
            cond = cond * mask.float()

        pred = self.model(x_t, t, cond=cond)
        loss = self.criterion(pred, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update EMA if enabled
        if self.use_ema:
            self.update_ema()

        return loss.item()

    @torch.no_grad()
    def sample(self, n_samples=16, cond=None, steps=50, method='euler', cfg_scale=1.0,
               use_ema=True, start_from=None):
        # Select model for inference
        if use_ema and self.use_ema and self.ema_model is not None:
            model = self.ema_model
        else:
            model = self.model

        img_shape = (self.in_channels, self.img_size[0], self.img_size[1])

        if start_from is not None:
            x = start_from.to(self.device)
        else:
            x = self._generate_source_from_noise((n_samples, *img_shape))

        if cond is None:
            cond = torch.zeros(n_samples, self.cond_dim, device=self.device)
        else:
            cond = cond.to(self.device)
        null_cond = torch.zeros_like(cond)

        dt = 1.0 / steps
        times = torch.linspace(0, 1, steps+1, device=self.device)

        def v_fn(t_val, x_val):
            t_tensor = torch.full((n_samples,), t_val, device=self.device)
            if cfg_scale != 1.0:
                v_cond = model(x_val, t_tensor, cond=cond)
                v_uncond = model(x_val, t_tensor, cond=null_cond)
                return v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                return model(x_val, t_tensor, cond=cond)

        for i in range(steps):
            t_cur = times[i]
            if method == 'euler':
                x = x + v_fn(t_cur, x) * dt
            elif method == 'heun':
                v1 = v_fn(t_cur, x)
                x_pred = x + v1 * dt
                v2 = v_fn(t_cur + dt, x_pred)
                x = x + (v1 + v2) * (dt / 2)
            elif method == 'rk4':
                v1 = v_fn(t_cur, x)
                x2 = x + v1 * dt/2
                v2 = v_fn(t_cur + dt/2, x2)
                x3 = x + v2 * dt/2
                v3 = v_fn(t_cur + dt/2, x3)
                x4 = x + v3 * dt
                v4 = v_fn(t_cur + dt, x4)
                x = x + (v1 + 2*v2 + 2*v3 + v4) * dt / 6
            elif method == 'midpoint':
                t_mid = t_cur + dt/2
                x_mid = x + v_fn(t_cur, x) * dt/2
                x = x + v_fn(t_mid, x_mid) * dt
            else:
                raise ValueError(f"Unknown method {method}")

        return x

    @torch.no_grad()
    def sample_step_by_step(self, n_samples=16, cond=None, steps=50, method='euler',
                            cfg_scale=1.0, use_ema=True, start_from=None):
        # Select model for inference
        if use_ema and self.use_ema and self.ema_model is not None:
            model = self.ema_model
        else:
            model = self.model

        img_shape = (self.in_channels, self.img_size[0], self.img_size[1])

        if start_from is not None:
            x = start_from.to(self.device)
        else:
            x = self._generate_source_from_noise((n_samples, *img_shape))

        if cond is None:
            cond = torch.zeros(n_samples, self.cond_dim, device=self.device)
        else:
            cond = cond.to(self.device)
        null_cond = torch.zeros_like(cond)

        dt = 1.0 / steps
        times = torch.linspace(0, 1, steps+1, device=self.device)

        def v_fn(t_val, x_val):
            t_tensor = torch.full((n_samples,), t_val, device=self.device)
            if cfg_scale != 1.0:
                v_cond = model(x_val, t_tensor, cond=cond)
                v_uncond = model(x_val, t_tensor, cond=null_cond)
                return v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                return model(x_val, t_tensor, cond=cond)

        for i in range(steps):
            t_cur = times[i]
            if method == 'euler':
                x = x + v_fn(t_cur, x) * dt
            elif method == 'heun':
                v1 = v_fn(t_cur, x)
                x_pred = x + v1 * dt
                v2 = v_fn(t_cur + dt, x_pred)
                x = x + (v1 + v2) * (dt / 2)
            elif method == 'rk4':
                v1 = v_fn(t_cur, x)
                x2 = x + v1 * dt/2
                v2 = v_fn(t_cur + dt/2, x2)
                x3 = x + v2 * dt/2
                v3 = v_fn(t_cur + dt/2, x3)
                x4 = x + v3 * dt
                v4 = v_fn(t_cur + dt, x4)
                x = x + (v1 + 2*v2 + 2*v3 + v4) * dt / 6
            elif method == 'midpoint':
                t_mid = t_cur + dt/2
                x_mid = x + v_fn(t_cur, x) * dt/2
                x = x + v_fn(t_mid, x_mid) * dt
            else:
                raise ValueError(f"Unknown method {method}")
            yield i+1, x.clone()

# ==================== Dataset with extended augmentations ====================
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
            self.base_transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            self.base_transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])

        self.aug_pipeline = []
        if self.aug_settings.get('flip_horizontal', False):
            self.aug_pipeline.append(transforms.RandomHorizontalFlip(p=0.5))
        if self.aug_settings.get('rotation', False):
            self.aug_pipeline.append(transforms.RandomRotation(degrees=30, interpolation=Image.BICUBIC, expand=False, fill=0))
        if self.aug_settings.get('random_crop', False):
            crop_size = self.aug_settings.get('crop_scale', 0.8)
            self.aug_pipeline.append(transforms.RandomResizedCrop(size=img_size, scale=(crop_size, 1.0), interpolation=Image.BICUBIC))
        if self.aug_settings.get('color_jitter', False):
            brightness = self.aug_settings.get('brightness', 0.2)
            contrast = self.aug_settings.get('contrast', 0.2)
            saturation = self.aug_settings.get('saturation', 0.2)
            hue = self.aug_settings.get('hue', 0.1)
            self.aug_pipeline.append(transforms.ColorJitter(brightness=brightness, contrast=contrast,
                                                           saturation=saturation, hue=hue))
        if self.aug_settings.get('random_perspective', False):
            distortion = self.aug_settings.get('perspective_distortion', 0.1)
            self.aug_pipeline.append(transforms.RandomPerspective(distortion_scale=distortion, p=0.5, interpolation=Image.BICUBIC, fill=0))
        if self.aug_settings.get('elastic_transform', False):
            alpha = self.aug_settings.get('elastic_alpha', 30)
            sigma = self.aug_settings.get('elastic_sigma', 3)
            self.aug_pipeline.append(ElasticTransform(alpha=alpha, sigma=sigma, p=0.5))
        if self.aug_settings.get('jpeg_compression', False):
            quality_low = self.aug_settings.get('jpeg_quality_low', 50)
            quality_high = self.aug_settings.get('jpeg_quality_high', 95)
            self.aug_pipeline.append(RandomJPEG(quality_low=quality_low, quality_high=quality_high, p=0.5))
        if self.aug_settings.get('stretch_vertical', False):
            self.aug_pipeline.append(transforms.RandomResizedCrop(size=img_size, scale=(0.8, 1.0),
                                                                 ratio=(0.5, 1.0), interpolation=Image.BICUBIC))
        if self.aug_settings.get('stretch_horizontal', False):
            self.aug_pipeline.append(transforms.RandomResizedCrop(size=img_size, scale=(0.8, 1.0),
                                                                 ratio=(1.0, 2.0), interpolation=Image.BICUBIC))

    def __len__(self):
        return len(self.image_paths)

    def apply_augmentations(self, pil_img):
        img = pil_img.copy()
        for aug in self.aug_pipeline:
            img = aug(img)
        return img

    def __getitem__(self, idx):
        try:
            if self.color_mode == 'rgb':
                pil_img = load_image_as_rgb(self.image_paths[idx])
            else:
                pil_img = load_image_as_grayscale(self.image_paths[idx])
            pil_img = self.apply_augmentations(pil_img)
            img_tensor = self.base_transform(pil_img)

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
        self.root.title("Rectified Flow - Unified (fixed EMA, two modes, live tuning, OT-FM)")

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
        self.flow_progressive_active = False

        # Settings variables
        self.settings = {
            'img_size': tk.IntVar(value=32),
            'color_mode': tk.StringVar(value='rgb'),
            'rectified_base_channels': tk.IntVar(value=64),
            'rectified_batch_size': tk.IntVar(value=16),
            'rectified_lr': tk.DoubleVar(value=2e-4),
            'cfg_dropout_prob': tk.DoubleVar(value=0.1),
            'use_attention': tk.BooleanVar(value=True),
            'ema_enabled': tk.BooleanVar(value=False),
            'ema_decay': tk.DoubleVar(value=0.999),
            'ema_mode': tk.StringVar(value='standard'),   # new
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
            'preview_enabled': tk.BooleanVar(value=True),
            'preview_steps': tk.IntVar(value=50),
            'preview_method': tk.StringVar(value='heun'),
            'preview_epoch_freq': tk.IntVar(value=5),
            'ot_task': tk.StringVar(value='noise'),
            'training_mode': tk.StringVar(value='dataset'),
            'blur_sigma': tk.DoubleVar(value=2.0),
            'bit_depth': tk.IntVar(value=1),
            'pixelation_size': tk.IntVar(value=8),
            'padding_mode': tk.StringVar(value='constant'),
            # New OT / noise / rescale options
            'use_ot': tk.BooleanVar(value=False),
            'source_noise_std': tk.DoubleVar(value=0.0),
            'pixelation_rescale_mode': tk.StringVar(value='nearest'),
        }

        self.aug_settings = {
            'flip_horizontal': tk.BooleanVar(value=True),
            'rotation': tk.BooleanVar(value=False),
            'random_crop': tk.BooleanVar(value=False),
            'crop_scale': tk.DoubleVar(value=0.8),
            'color_jitter': tk.BooleanVar(value=False),
            'brightness': tk.DoubleVar(value=0.2),
            'contrast': tk.DoubleVar(value=0.2),
            'saturation': tk.DoubleVar(value=0.2),
            'hue': tk.DoubleVar(value=0.1),
            'random_perspective': tk.BooleanVar(value=False),
            'perspective_distortion': tk.DoubleVar(value=0.1),
            'elastic_transform': tk.BooleanVar(value=False),
            'elastic_alpha': tk.IntVar(value=30),
            'elastic_sigma': tk.IntVar(value=3),
            'jpeg_compression': tk.BooleanVar(value=False),
            'jpeg_quality_low': tk.IntVar(value=50),
            'jpeg_quality_high': tk.IntVar(value=95),
            'stretch_vertical': tk.BooleanVar(value=False),
            'stretch_horizontal': tk.BooleanVar(value=False),
        }

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
        self.notebook.add(self.rectified_tab, text='Train Flow')
        self.setup_rectified_tab()

        self.settings_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.settings_tab, text='Settings')
        self.setup_settings_tab()

        self.generation_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.generation_tab, text='Generate')
        self.setup_generation_tab()

        self.flow_to_image_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.flow_to_image_tab, text='Flow to Image')
        self.setup_flow_to_image_tab()

        self.status_label = tk.Label(self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    # ------------------------------------------------------------------
    # Dataset Tab
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Training Tab
    # ------------------------------------------------------------------
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

        tk.Button(left_frame, text="Initialize Model", command=self.initialize_rectified_model, width=20).pack(pady=5)
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

    # ------------------------------------------------------------------
    # Settings Tab (with EMA mode dropdown, OT, noise, rescale)
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

        # Architecture
        arch_frame = tk.LabelFrame(scrollable_frame, text="Flow Architecture", padx=10, pady=10)
        arch_frame.pack(fill=tk.X, pady=5)
        f = tk.Frame(arch_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="UNet base channels:", width=20, anchor='w').pack(side=tk.LEFT)
        spin = ttk.Spinbox(f, from_=32, to=128, textvariable=self.settings['rectified_base_channels'], width=8)
        spin.pack(side=tk.RIGHT)
        f = tk.Frame(arch_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Use Self-Attention:", width=20, anchor='w').pack(side=tk.LEFT)
        cb = tk.Checkbutton(f, variable=self.settings['use_attention'])
        cb.pack(side=tk.RIGHT)
        f = tk.Frame(arch_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Padding mode:", width=20, anchor='w').pack(side=tk.LEFT)
        pad_combo = ttk.Combobox(f, textvariable=self.settings['padding_mode'],
                                 values=['constant', 'reflect', 'replicate', 'circular'],
                                 state='readonly', width=10)
        pad_combo.pack(side=tk.RIGHT)

        # EMA settings (with mode dropdown)
        ema_frame = tk.LabelFrame(scrollable_frame, text="Exponential Moving Average", padx=10, pady=10)
        ema_frame.pack(fill=tk.X, pady=5)
        f = tk.Frame(ema_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Enable EMA:", width=20, anchor='w').pack(side=tk.LEFT)
        cb_ema = tk.Checkbutton(f, variable=self.settings['ema_enabled'])
        cb_ema.pack(side=tk.RIGHT)
        f = tk.Frame(ema_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="EMA decay:", width=20, anchor='w').pack(side=tk.LEFT)
        spin_ema = tk.Entry(f, textvariable=self.settings['ema_decay'], width=8)
        spin_ema.pack(side=tk.RIGHT)
        f = tk.Frame(ema_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="EMA mode:", width=20, anchor='w').pack(side=tk.LEFT)
        mode_combo = ttk.Combobox(f, textvariable=self.settings['ema_mode'],
                                  values=['standard', 'lookahead'], state='readonly', width=10)
        mode_combo.pack(side=tk.RIGHT)
        tk.Label(ema_frame, text="standard: target updated from online; lookahead: online reset to target after each step", fg="gray", font=("Arial",8)).pack(anchor='w', padx=20)

        # Training params (live tune)
        train_frame = tk.LabelFrame(scrollable_frame, text="Training (live tune)", padx=10, pady=10)
        train_frame.pack(fill=tk.X, pady=5)
        for label, key, low, high, typ in [("Batch size:", 'rectified_batch_size', 1, 32, int),
                                            ("Learning rate:", 'rectified_lr', 1e-5, 1e-2, float)]:
            f = tk.Frame(train_frame); f.pack(fill=tk.X, pady=2)
            tk.Label(f, text=label, width=20, anchor='w').pack(side=tk.LEFT)
            if typ == int:
                spin = ttk.Spinbox(f, from_=low, to=high, textvariable=self.settings[key], width=8)
            else:
                spin = tk.Entry(f, textvariable=self.settings[key], width=8)
            spin.pack(side=tk.RIGHT)

        # Preview
        preview_frame = tk.LabelFrame(scrollable_frame, text="Preview During Training", padx=10, pady=10)
        preview_frame.pack(fill=tk.X, pady=5)
        for label, key in [("Enable preview:", 'preview_enabled'), ("Every N epochs:", 'preview_epoch_freq'),
                           ("Preview steps:", 'preview_steps')]:
            f = tk.Frame(preview_frame); f.pack(fill=tk.X, pady=2)
            tk.Label(f, text=label, width=20, anchor='w').pack(side=tk.LEFT)
            if key == 'preview_epoch_freq' or key == 'preview_steps':
                spin = ttk.Spinbox(f, from_=1, to=200, textvariable=self.settings[key], width=8)
                spin.pack(side=tk.RIGHT)
            else:
                cb = tk.Checkbutton(f, variable=self.settings[key])
                cb.pack(side=tk.RIGHT)
        f = tk.Frame(preview_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Preview ODE method:", width=20, anchor='w').pack(side=tk.LEFT)
        method_combo = ttk.Combobox(f, textvariable=self.settings['preview_method'],
                                    values=['euler', 'heun', 'rk4', 'midpoint'], state='readonly', width=10)
        method_combo.pack(side=tk.RIGHT)

        # OT Task + Training Mode + OT + Noise + Rescale
        ot_frame = tk.LabelFrame(scrollable_frame, text="Optimal Transport Task", padx=10, pady=10)
        ot_frame.pack(fill=tk.X, pady=5)
        f = tk.Frame(ot_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Task:", width=20, anchor='w').pack(side=tk.LEFT)
        task_combo = ttk.Combobox(f, textvariable=self.settings['ot_task'],
                                  values=['noise', 'grayscale_to_color', 'blur_to_sharp',
                                          'white_to_image', 'average_separation',
                                          'unquantization', 'pixelation'],
                                  state='readonly', width=20)
        task_combo.pack(side=tk.RIGHT)
        f = tk.Frame(ot_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Training mode:", width=20, anchor='w').pack(side=tk.LEFT)
        mode_combo = ttk.Combobox(f, textvariable=self.settings['training_mode'],
                                  values=['dataset', 'noise', 'mixed'], state='readonly', width=10)
        mode_combo.pack(side=tk.RIGHT)
        f = tk.Frame(ot_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Blur sigma:", width=20, anchor='w').pack(side=tk.LEFT)
        spin_sigma = ttk.Spinbox(f, from_=0.5, to=5.0, increment=0.1, textvariable=self.settings['blur_sigma'], width=8)
        spin_sigma.pack(side=tk.RIGHT)
        f = tk.Frame(ot_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Bit depth (unquant):", width=20, anchor='w').pack(side=tk.LEFT)
        spin_bit = ttk.Spinbox(f, from_=1, to=7, textvariable=self.settings['bit_depth'], width=8)
        spin_bit.pack(side=tk.RIGHT)
        f = tk.Frame(ot_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Pixelation block size:", width=20, anchor='w').pack(side=tk.LEFT)
        spin_pix = ttk.Spinbox(f, from_=2, to=32, textvariable=self.settings['pixelation_size'], width=8)
        spin_pix.pack(side=tk.RIGHT)

        # New: OT matching, source noise, pixelation rescale
        f = tk.Frame(ot_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Enable OT Matching:", width=20, anchor='w').pack(side=tk.LEFT)
        cb_ot = tk.Checkbutton(f, variable=self.settings['use_ot'])
        cb_ot.pack(side=tk.RIGHT)

        f = tk.Frame(ot_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Source noise std (0..1):", width=20, anchor='w').pack(side=tk.LEFT)
        spin_noise = ttk.Spinbox(f, from_=0.0, to=1.0, increment=0.01, textvariable=self.settings['source_noise_std'], width=8)
        spin_noise.pack(side=tk.RIGHT)

        f = tk.Frame(ot_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Pixelation rescale mode:", width=20, anchor='w').pack(side=tk.LEFT)
        rescale_combo = ttk.Combobox(f, textvariable=self.settings['pixelation_rescale_mode'],
                                     values=['nearest', 'bilinear', 'bicubic'], state='readonly', width=10)
        rescale_combo.pack(side=tk.RIGHT)

        # Text conditioning
        cond_frame = tk.LabelFrame(scrollable_frame, text="Text Conditioning", padx=10, pady=10)
        cond_frame.pack(fill=tk.X, pady=5)
        f = tk.Frame(cond_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Encoder type:", width=20, anchor='w').pack(side=tk.LEFT)
        type_combo = ttk.Combobox(f, textvariable=self.settings['text_encoder_type'],
                                  values=['BiGRU', 'BiTransformer'], state='readonly', width=12)
        type_combo.pack(side=tk.RIGHT)
        f = tk.Frame(cond_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Encoder size:", width=20, anchor='w').pack(side=tk.LEFT)
        size_combo = ttk.Combobox(f, textvariable=self.settings['text_encoder_size'],
                                  values=['tiny', 'small', 'medium', 'large'], state='readonly', width=8)
        size_combo.pack(side=tk.RIGHT)
        tk.Button(cond_frame, text="Apply preset", command=self.apply_text_encoder_preset).pack(pady=2)
        cond_params = [("Embed dim:", 'cond_embed_dim', 32, 256),
                       ("Hidden size (GRU):", 'cond_hidden_size', 32, 512),
                       ("Num layers:", 'cond_num_layers', 1, 4),
                       ("Num heads (TF):", 'cond_num_heads', 1, 16),
                       ("FF dim (TF):", 'cond_ff_dim', 64, 1024),
                       ("Cond dim:", 'cond_dim', 64, 512),
                       ("Max text len:", 'cond_text_max_len', 32, 256)]
        for label, key, low, high in cond_params:
            f = tk.Frame(cond_frame); f.pack(fill=tk.X, pady=2)
            tk.Label(f, text=label, width=22, anchor='w').pack(side=tk.LEFT)
            spin = ttk.Spinbox(f, from_=low, to=high, textvariable=self.settings[key], width=8)
            spin.pack(side=tk.RIGHT)

        # Augmentations
        aug_frame = tk.LabelFrame(scrollable_frame, text="Data Augmentations (all use replicate border)", padx=10, pady=10)
        aug_frame.pack(fill=tk.X, pady=5)
        f1 = tk.Frame(aug_frame); f1.pack(fill=tk.X, pady=2)
        tk.Checkbutton(f1, text="Horizontal Flip", variable=self.aug_settings['flip_horizontal']).pack(side=tk.LEFT, padx=5)
        tk.Checkbutton(f1, text="Rotation (±30°)", variable=self.aug_settings['rotation']).pack(side=tk.LEFT, padx=5)
        tk.Checkbutton(f1, text="Random Crop (scale)", variable=self.aug_settings['random_crop']).pack(side=tk.LEFT, padx=5)
        tk.Label(f1, text="crop scale:").pack(side=tk.LEFT, padx=(10,0))
        tk.Entry(f1, textvariable=self.aug_settings['crop_scale'], width=5).pack(side=tk.LEFT)
        f2 = tk.Frame(aug_frame); f2.pack(fill=tk.X, pady=2)
        tk.Checkbutton(f2, text="Color Jitter", variable=self.aug_settings['color_jitter']).pack(side=tk.LEFT, padx=5)
        tk.Label(f2, text="bri/con/sat/hue:").pack(side=tk.LEFT, padx=(10,0))
        tk.Entry(f2, textvariable=self.aug_settings['brightness'], width=4).pack(side=tk.LEFT)
        tk.Entry(f2, textvariable=self.aug_settings['contrast'], width=4).pack(side=tk.LEFT)
        tk.Entry(f2, textvariable=self.aug_settings['saturation'], width=4).pack(side=tk.LEFT)
        tk.Entry(f2, textvariable=self.aug_settings['hue'], width=4).pack(side=tk.LEFT)
        f3 = tk.Frame(aug_frame); f3.pack(fill=tk.X, pady=2)
        tk.Checkbutton(f3, text="Random Perspective", variable=self.aug_settings['random_perspective']).pack(side=tk.LEFT, padx=5)
        tk.Label(f3, text="distortion:").pack(side=tk.LEFT, padx=(10,0))
        tk.Entry(f3, textvariable=self.aug_settings['perspective_distortion'], width=5).pack(side=tk.LEFT)
        tk.Checkbutton(f3, text="Elastic Transform", variable=self.aug_settings['elastic_transform']).pack(side=tk.LEFT, padx=10)
        tk.Label(f3, text="alpha/sigma:").pack(side=tk.LEFT, padx=(10,0))
        tk.Entry(f3, textvariable=self.aug_settings['elastic_alpha'], width=4).pack(side=tk.LEFT)
        tk.Entry(f3, textvariable=self.aug_settings['elastic_sigma'], width=4).pack(side=tk.LEFT)
        f4 = tk.Frame(aug_frame); f4.pack(fill=tk.X, pady=2)
        tk.Checkbutton(f4, text="JPEG Compression", variable=self.aug_settings['jpeg_compression']).pack(side=tk.LEFT, padx=5)
        tk.Label(f4, text="quality low/high:").pack(side=tk.LEFT, padx=(10,0))
        tk.Entry(f4, textvariable=self.aug_settings['jpeg_quality_low'], width=4).pack(side=tk.LEFT)
        tk.Entry(f4, textvariable=self.aug_settings['jpeg_quality_high'], width=4).pack(side=tk.LEFT)
        tk.Checkbutton(f4, text="Stretch Vertical", variable=self.aug_settings['stretch_vertical']).pack(side=tk.LEFT, padx=10)
        tk.Checkbutton(f4, text="Stretch Horizontal", variable=self.aug_settings['stretch_horizontal']).pack(side=tk.LEFT, padx=5)

        # Reset button
        reset_frame = tk.Frame(scrollable_frame)
        reset_frame.pack(fill=tk.X, pady=10)
        tk.Button(reset_frame, text="Reset All Hyperparameters to Default", command=self.reset_all_settings,
                  bg="orange", fg="white", font=("Arial",10,"bold")).pack(pady=5)

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
        self.settings['cond_embed_dim'].set(config.get('embed_dim', self.settings['cond_embed_dim'].get()))
        if enc_type == 'BiGRU':
            self.settings['cond_hidden_size'].set(config.get('hidden_size', self.settings['cond_hidden_size'].get()))
            self.settings['cond_num_layers'].set(config.get('num_layers', self.settings['cond_num_layers'].get()))
        else:
            self.settings['cond_num_heads'].set(config.get('num_heads', self.settings['cond_num_heads'].get()))
            self.settings['cond_num_layers'].set(config.get('num_layers', self.settings['cond_num_layers'].get()))
            self.settings['cond_ff_dim'].set(config.get('ff_dim', self.settings['cond_ff_dim'].get()))
        self.settings['cond_dim'].set(config.get('cond_dim', self.settings['cond_dim'].get()))
        self.log_rectified(f"Applied {enc_type} {size} preset.")

    def reset_all_settings(self):
        self.settings['img_size'].set(32)
        self.settings['color_mode'].set('rgb')
        self.settings['rectified_base_channels'].set(64)
        self.settings['rectified_batch_size'].set(16)
        self.settings['rectified_lr'].set(2e-4)
        self.settings['cfg_dropout_prob'].set(0.1)
        self.settings['use_attention'].set(True)
        self.settings['ema_enabled'].set(False)
        self.settings['ema_decay'].set(0.999)
        self.settings['ema_mode'].set('standard')
        self.settings['cond_enabled'].set(False)
        self.settings['text_encoder_type'].set('BiGRU')
        self.settings['text_encoder_size'].set('small')
        self.settings['cond_embed_dim'].set(64)
        self.settings['cond_hidden_size'].set(64)
        self.settings['cond_num_layers'].set(2)
        self.settings['cond_num_heads'].set(4)
        self.settings['cond_ff_dim'].set(256)
        self.settings['cond_dim'].set(256)
        self.settings['cond_text_max_len'].set(128)
        self.settings['preview_enabled'].set(True)
        self.settings['preview_steps'].set(50)
        self.settings['preview_method'].set('heun')
        self.settings['preview_epoch_freq'].set(5)
        self.settings['ot_task'].set('noise')
        self.settings['training_mode'].set('dataset')
        self.settings['blur_sigma'].set(2.0)
        self.settings['bit_depth'].set(1)
        self.settings['pixelation_size'].set(8)
        self.settings['padding_mode'].set('constant')
        # new ones
        self.settings['use_ot'].set(False)
        self.settings['source_noise_std'].set(0.0)
        self.settings['pixelation_rescale_mode'].set('nearest')
        self.aug_settings['flip_horizontal'].set(True)
        self.aug_settings['rotation'].set(False)
        self.aug_settings['random_crop'].set(False)
        self.aug_settings['crop_scale'].set(0.8)
        self.aug_settings['color_jitter'].set(False)
        self.aug_settings['brightness'].set(0.2)
        self.aug_settings['contrast'].set(0.2)
        self.aug_settings['saturation'].set(0.2)
        self.aug_settings['hue'].set(0.1)
        self.aug_settings['random_perspective'].set(False)
        self.aug_settings['perspective_distortion'].set(0.1)
        self.aug_settings['elastic_transform'].set(False)
        self.aug_settings['elastic_alpha'].set(30)
        self.aug_settings['elastic_sigma'].set(3)
        self.aug_settings['jpeg_compression'].set(False)
        self.aug_settings['jpeg_quality_low'].set(50)
        self.aug_settings['jpeg_quality_high'].set(95)
        self.aug_settings['stretch_vertical'].set(False)
        self.aug_settings['stretch_horizontal'].set(False)
        self.log_rectified("All hyperparameters reset to default.")

    # ------------------------------------------------------------------
    # Generation Tab
    # ------------------------------------------------------------------
    def setup_generation_tab(self):
        main_frame = tk.Frame(self.generation_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        tk.Label(main_frame, text="Generate Images", font=("Arial",14,"bold")).pack(pady=(0,10))

        prompt_frame = tk.LabelFrame(main_frame, text="Text Prompt (optional)", padx=5, pady=5)
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
                                    values=['euler', 'heun', 'rk4', 'midpoint'], state='readonly', width=8)
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
    # Flow to Image Tab
    # ------------------------------------------------------------------
    def setup_flow_to_image_tab(self):
        main_frame = tk.Frame(self.flow_to_image_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        tk.Label(main_frame, text="Flow to Image (Transport selected images to dataset distribution)", font=("Arial",12,"bold")).pack(pady=(0,10))

        top_frame = tk.Frame(main_frame)
        top_frame.pack(fill=tk.X, pady=5)

        prompt_frame = tk.LabelFrame(top_frame, text="Text Prompt (optional)", padx=5, pady=5)
        prompt_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,10))
        self.flow_prompt = tk.Entry(prompt_frame, width=40)
        self.flow_prompt.insert(0, "a cute cat")
        self.flow_prompt.pack(fill=tk.X)

        img_buttons_frame = tk.Frame(top_frame)
        img_buttons_frame.pack(side=tk.RIGHT)
        tk.Button(img_buttons_frame, text="Select Images", command=self.add_flow_images, bg="lightblue").pack(side=tk.LEFT, padx=2)
        tk.Button(img_buttons_frame, text="Clear", command=self.clear_flow_images, bg="lightcoral").pack(side=tk.LEFT, padx=2)
        self.flow_image_count_label = tk.Label(img_buttons_frame, text="0 images")
        self.flow_image_count_label.pack(side=tk.LEFT, padx=5)

        param_frame = tk.Frame(main_frame)
        param_frame.pack(pady=5)
        tk.Label(param_frame, text="ODE Steps:").pack(side=tk.LEFT)
        self.flow_steps = tk.IntVar(value=50)
        tk.Spinbox(param_frame, from_=1, to=200, textvariable=self.flow_steps, width=5).pack(side=tk.LEFT, padx=5)
        tk.Label(param_frame, text="Method:").pack(side=tk.LEFT, padx=(10,0))
        self.flow_method = tk.StringVar(value='euler')
        method_combo = ttk.Combobox(param_frame, textvariable=self.flow_method,
                                    values=['euler', 'heun', 'rk4', 'midpoint'], state='readonly', width=8)
        method_combo.pack(side=tk.LEFT, padx=5)
        tk.Label(param_frame, text="CFG scale:").pack(side=tk.LEFT, padx=(10,0))
        self.flow_cfg = tk.DoubleVar(value=2.0)
        tk.Entry(param_frame, width=6, textvariable=self.flow_cfg).pack(side=tk.LEFT, padx=5)
        self.flow_progressive = tk.BooleanVar(value=False)
        tk.Checkbutton(param_frame, text="Progressive", variable=self.flow_progressive).pack(side=tk.LEFT, padx=10)
        self.flow_prog_interval = tk.IntVar(value=10)
        tk.Label(param_frame, text="Interval:").pack(side=tk.LEFT)
        tk.Spinbox(param_frame, from_=1, to=50, textvariable=self.flow_prog_interval, width=5).pack(side=tk.LEFT, padx=5)
        self.apply_degradation_var = tk.BooleanVar(value=True)
        tk.Checkbutton(param_frame, text="Apply degradation to input", variable=self.apply_degradation_var).pack(side=tk.LEFT, padx=10)

        self.flow_btn = tk.Button(param_frame, text="Run Flow", command=self.run_flow_to_image, bg="lightgreen")
        self.flow_btn.pack(side=tk.LEFT, padx=5)
        self.stop_flow_btn = tk.Button(param_frame, text="Stop", command=self.stop_flow, state=tk.DISABLED, bg="salmon")
        self.stop_flow_btn.pack(side=tk.LEFT, padx=5)

        result_frame = tk.LabelFrame(main_frame, text="Results", padx=5, pady=5)
        result_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        canvas_frame = tk.Frame(result_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        self.flow_canvas = tk.Canvas(canvas_frame, bg='lightgray')
        self.flow_v_scroll = tk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.flow_canvas.yview)
        self.flow_h_scroll = tk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.flow_canvas.xview)
        self.flow_canvas.configure(yscrollcommand=self.flow_v_scroll.set, xscrollcommand=self.flow_h_scroll.set)
        self.flow_v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.flow_h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.flow_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.flow_inner_frame = tk.Frame(self.flow_canvas)
        self.flow_canvas_window = self.flow_canvas.create_window((0,0), window=self.flow_inner_frame, anchor='nw')
        self.flow_inner_frame.bind('<Configure>', lambda e: self.flow_canvas.configure(scrollregion=self.flow_canvas.bbox('all')))
        self.flow_info = tk.Label(main_frame, text="", fg="blue")
        self.flow_info.pack()

        self.flow_image_paths = []

    def add_flow_images(self):
        files = filedialog.askopenfilenames(filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.jfif *.webp")])
        for f in files:
            if f not in self.flow_image_paths:
                self.flow_image_paths.append(f)
        self.flow_image_count_label.config(text=f"{len(self.flow_image_paths)} images")
        self.log_rectified(f"Flow to Image: added {len(files)} images, total {len(self.flow_image_paths)}")

    def clear_flow_images(self):
        self.flow_image_paths = []
        self.flow_image_count_label.config(text="0 images")
        for widget in self.flow_inner_frame.winfo_children():
            widget.destroy()
        self.flow_info.config(text="Cleared.")

    def run_flow_to_image(self):
        if not self.rectified_model:
            self.flow_info.config(text="Model not loaded!")
            return
        if not self.flow_image_paths:
            self.flow_info.config(text="No images selected. Use 'Select Images' first.")
            return

        for widget in self.flow_inner_frame.winfo_children():
            widget.destroy()

        prompt = self.flow_prompt.get().strip()
        cond = None
        if prompt and self.settings['cond_enabled'].get() and self.text_encoder is not None:
            text_indices = text_to_indices(prompt, self.settings['cond_text_max_len'].get())
            text_tensor = torch.tensor([text_indices] * len(self.flow_image_paths), dtype=torch.long, device=self.rectified_model.device)
            with torch.no_grad():
                cond = self.text_encoder(text_tensor)
        elif prompt:
            self.log_rectified("Warning: conditioning disabled or text encoder not available, ignoring prompt.")

        self.flow_btn.config(state=tk.DISABLED)
        self.stop_flow_btn.config(state=tk.NORMAL)
        steps = self.flow_steps.get()
        method = self.flow_method.get()
        cfg_scale = self.flow_cfg.get()
        use_ema = self.settings['ema_enabled'].get()
        degrade = self.apply_degradation_var.get()

        if self.flow_progressive.get():
            self.flow_progressive_active = True
        else:
            self.flow_progressive_active = False

        thread = threading.Thread(target=self._flow_images_thread,
                                  args=(self.flow_image_paths, cond, steps, method, cfg_scale, use_ema, degrade),
                                  daemon=True)
        thread.start()

    def _flow_images_thread(self, image_paths, cond, steps, method, cfg_scale, use_ema, degrade):
        try:
            color_mode = self.settings['color_mode'].get()
            img_size = self.settings['img_size'].get()
            in_channels = 3 if color_mode == 'rgb' else 1
            transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,)*in_channels, (0.5,)*in_channels)
            ])

            imgs_tensors = []
            for p in image_paths:
                try:
                    if color_mode == 'rgb':
                        pil = load_image_as_rgb(p)
                    else:
                        pil = load_image_as_grayscale(p)
                    tensor = transform(pil).unsqueeze(0).to(self.rectified_model.device)
                    imgs_tensors.append(tensor)
                except Exception as e:
                    self.log_rectified(f"Error loading {p}: {e}")
                    continue

            if not imgs_tensors:
                self.root.after(0, lambda: self.flow_info.config(text="No valid images loaded."))
                return

            n = len(imgs_tensors)
            if degrade:
                task = self.settings['ot_task'].get()
                if task == "average_separation":
                    self.root.after(0, lambda: self.flow_info.config(text="Average separation not supported for single images."))
                    return
                orig_task = self.rectified_model.task
                orig_params = self.rectified_model.task_params.copy()
                # Temporarily set task params to include noise and rescale settings
                self.rectified_model.task_params.update({
                    'blur_sigma': self.settings['blur_sigma'].get(),
                    'bit_depth': self.settings['bit_depth'].get(),
                    'pixelation_size': self.settings['pixelation_size'].get(),
                    'source_noise_std': self.settings['source_noise_std'].get(),
                    'pixelation_rescale_mode': self.settings['pixelation_rescale_mode'].get(),
                })
                with torch.no_grad():
                    start_tensors = [self.rectified_model._generate_source_from_target(img) for img in imgs_tensors]
                    start_batch = torch.cat(start_tensors, dim=0)
                # Restore original task_params
                self.rectified_model.task_params = orig_params
            else:
                start_batch = torch.cat(imgs_tensors, dim=0)

            if self.flow_progressive.get() and self.flow_progressive_active:
                interval = self.flow_prog_interval.get()
                generator = self.rectified_model.sample_step_by_step(n_samples=n, cond=cond, steps=steps, method=method,
                                                                     cfg_scale=cfg_scale, use_ema=use_ema,
                                                                     start_from=start_batch)
                for step_idx, x in generator:
                    if not self.flow_progressive_active:
                        break
                    if step_idx % interval == 0 or step_idx == steps:
                        self._display_flow_grid(x, step_idx, steps)
                        time.sleep(0.05)
                if self.flow_progressive_active:
                    self.root.after(0, lambda: self.flow_info.config(text="Flow completed."))
                self.flow_progressive_active = False
            else:
                result = self.rectified_model.sample(n_samples=n, cond=cond, steps=steps, method=method,
                                                     cfg_scale=cfg_scale, use_ema=use_ema, start_from=start_batch)
                self._display_flow_grid(result, steps, steps)
                self.root.after(0, lambda: self.flow_info.config(text="Flow completed."))

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda: self.flow_info.config(text=f"Error: {e}"))
        finally:
            self.root.after(0, lambda: self.flow_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.stop_flow_btn.config(state=tk.DISABLED))
            self.flow_progressive_active = False

    def _display_flow_grid(self, x, current_step, total_steps):
        samples = (x + 1) / 2
        samples = samples.clamp(0,1).cpu().numpy()
        n = len(samples)
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
        photo = ImageTk.PhotoImage(grid_img)
        self.root.after(0, lambda p=photo, s=current_step: self._update_flow_canvas(p, s, total_steps))

    def _update_flow_canvas(self, photo, step, total):
        for widget in self.flow_inner_frame.winfo_children():
            widget.destroy()
        label = tk.Label(self.flow_inner_frame, image=photo)
        label.image = photo
        label.pack()
        self.flow_inner_frame.update_idletasks()
        self.flow_canvas.configure(scrollregion=self.flow_canvas.bbox('all'))
        self.flow_info.config(text=f"Step {step}/{total}")

    def stop_flow(self):
        self.flow_progressive_active = False
        self.stop_flow_btn.config(state=tk.DISABLED)
        self.flow_btn.config(state=tk.NORMAL)
        self.flow_info.config(text="Stopped.")

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

    def clear_images(self):
        self.image_paths = []
        self.labels = []
        self.image_listbox.delete(0, tk.END)
        self.csv_status.config(text="No CSV loaded", fg="red")
        self.log_dataset("Cleared all images")

    def load_csv(self):
        fname = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if not fname:
            return
        self.csv_path = fname
        path_by_basename = {}
        for full in self.image_paths:
            base = os.path.basename(full).lower()
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
                    matched = path_by_basename.get(base)
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
        self.log_rectified(f"CSV loaded. {len(self.image_paths)-unknown} matched.")

    def use_filenames_as_labels(self):
        self.labels = [[os.path.splitext(os.path.basename(p))[0]] for p in self.image_paths]
        self.csv_status.config(text="Using filenames as labels", fg="blue")
        self.log_rectified("Using filenames as labels.")

    def use_folders_as_labels(self):
        self.labels = [[os.path.basename(os.path.dirname(p)) or 'unknown'] for p in self.image_paths]
        self.csv_status.config(text="Using folder names as labels", fg="blue")
        self.log_rectified("Using folder names as labels.")

    def initialize_rectified_model(self):
        try:
            cond_enabled = self.settings['cond_enabled'].get()
            cond_dim = self.settings['cond_dim'].get() if cond_enabled else 1
            color_mode = self.settings['color_mode'].get()
            in_channels = 3 if color_mode == 'rgb' else 1
            img_size = self.settings['img_size'].get()
            use_attention = self.settings['use_attention'].get()
            task = self.settings['ot_task'].get()
            training_mode = self.settings['training_mode'].get()
            padding_mode = self.settings['padding_mode'].get()
            task_params = {
                'blur_sigma': self.settings['blur_sigma'].get(),
                'bit_depth': self.settings['bit_depth'].get(),
                'pixelation_size': self.settings['pixelation_size'].get(),
                'source_noise_std': self.settings['source_noise_std'].get(),
                'pixelation_rescale_mode': self.settings['pixelation_rescale_mode'].get(),
            }
            self.rectified_model = RectifiedFlowPixel(
                in_channels=in_channels, img_size=img_size,
                base_channels=self.settings['rectified_base_channels'].get(),
                cond_dim=cond_dim, use_attention=use_attention,
                task=task, task_params=task_params, training_mode=training_mode,
                padding_mode=padding_mode
            )
            self.rectified_model.use_ot = self.settings['use_ot'].get()
            for pg in self.rectified_model.optimizer.param_groups:
                pg['lr'] = self.settings['rectified_lr'].get()

            # Configure EMA if enabled
            if self.settings['ema_enabled'].get():
                decay = self.settings['ema_decay'].get()
                mode = self.settings['ema_mode'].get()
                self.rectified_model.set_ema(decay=decay, mode=mode)
                self.log_rectified(f"EMA enabled: decay={decay}, mode={mode}")
            else:
                self.rectified_model.use_ema = False
                self.rectified_model.ema_model = None

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
                self.rectified_model.optimizer = optim.Adam(
                    list(self.rectified_model.model.parameters()) + list(self.text_encoder.parameters()),
                    lr=self.settings['rectified_lr'].get()
                )
                self.log_rectified(f"Conditional model with {enc_type} encoder, task={task}, mode={training_mode}, padding={padding_mode}, OT={self.rectified_model.use_ot}")
            else:
                self.text_encoder = None
                self.log_rectified(f"Unconditional model, task={task}, mode={training_mode}, padding={padding_mode}, OT={self.rectified_model.use_ot}")
        except Exception as e:
            self.log_rectified(f"Init error: {e}")

    def start_rectified_training(self):
        if not self.image_paths:
            self.log_rectified("No images!")
            return
        if not self.rectified_model:
            self.log_rectified("Model not initialized!")
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
        self.log_rectified(f"Training started for {epochs} epochs. Task: {self.settings['ot_task'].get()}, Mode: {self.settings['training_mode'].get()}, OT: {self.settings['use_ot'].get()}")

    def train_rectified_loop(self, epochs):
        try:
            # Store current values for live tuning
            current_batch_size = self.settings['rectified_batch_size'].get()
            current_lr = self.settings['rectified_lr'].get()
            current_ema_decay = self.settings['ema_decay'].get()
            current_ema_mode = self.settings['ema_mode'].get()
            current_ema_enabled = self.settings['ema_enabled'].get()
            current_use_ot = self.settings['use_ot'].get()
            current_source_noise_std = self.settings['source_noise_std'].get()
            current_pixelation_rescale_mode = self.settings['pixelation_rescale_mode'].get()

            img_size = self.settings['img_size'].get()
            color_mode = self.settings['color_mode'].get()
            cond_enabled = self.settings['cond_enabled'].get()
            text_max_len = self.settings['cond_text_max_len'].get()
            cfg_dropout = self.settings['cfg_dropout_prob'].get()
            preview_enabled = self.settings['preview_enabled'].get()
            preview_freq = self.settings['preview_epoch_freq'].get()
            aug_dict = {k: v.get() for k, v in self.aug_settings.items()}
            labels_for_dataset = self.labels if self.labels else [['']]*len(self.image_paths)

            # Dataset and loader (will be recreated if batch size changes)
            dataset = ConditionalImageDataset(self.image_paths, labels_for_dataset,
                                              img_size, color_mode, aug_dict, text_max_len)
            loader = DataLoader(dataset, batch_size=current_batch_size, shuffle=True, num_workers=0, pin_memory=False)

            for epoch in range(epochs):
                if not self.training_rectified:
                    break
                self.current_epoch_rectified = epoch

                # --- Live parameter updates at epoch start ---
                new_batch_size = self.settings['rectified_batch_size'].get()
                new_lr = self.settings['rectified_lr'].get()
                new_ema_decay = self.settings['ema_decay'].get()
                new_ema_mode = self.settings['ema_mode'].get()
                new_ema_enabled = self.settings['ema_enabled'].get()
                new_use_ot = self.settings['use_ot'].get()
                new_source_noise_std = self.settings['source_noise_std'].get()
                new_pixelation_rescale_mode = self.settings['pixelation_rescale_mode'].get()

                # Update model task params (for noise and rescale)
                if (new_source_noise_std != current_source_noise_std or
                    new_pixelation_rescale_mode != current_pixelation_rescale_mode):
                    self.rectified_model.task_params['source_noise_std'] = new_source_noise_std
                    self.rectified_model.task_params['pixelation_rescale_mode'] = new_pixelation_rescale_mode
                    current_source_noise_std = new_source_noise_std
                    current_pixelation_rescale_mode = new_pixelation_rescale_mode
                    self.log_rectified(f"Task params updated: noise_std={new_source_noise_std}, rescale={new_pixelation_rescale_mode}")

                # Update OT flag
                if new_use_ot != current_use_ot:
                    self.rectified_model.use_ot = new_use_ot
                    current_use_ot = new_use_ot
                    self.log_rectified(f"OT matching {'enabled' if new_use_ot else 'disabled'}")

                # Update learning rate
                if new_lr != current_lr:
                    for pg in self.rectified_model.optimizer.param_groups:
                        pg['lr'] = new_lr
                    current_lr = new_lr
                    self.log_rectified(f"LR updated to {new_lr}")

                # Update batch size if changed (recreate loader)
                if new_batch_size != current_batch_size:
                    current_batch_size = new_batch_size
                    loader = DataLoader(dataset, batch_size=current_batch_size, shuffle=True, num_workers=0, pin_memory=False)
                    self.log_rectified(f"Batch size updated to {current_batch_size}")

                # Update EMA settings
                if new_ema_enabled != current_ema_enabled:
                    current_ema_enabled = new_ema_enabled
                    if current_ema_enabled:
                        self.rectified_model.set_ema(decay=new_ema_decay, mode=new_ema_mode)
                        self.log_rectified(f"EMA enabled: decay={new_ema_decay}, mode={new_ema_mode}")
                    else:
                        self.rectified_model.use_ema = False
                        self.rectified_model.ema_model = None
                        self.log_rectified("EMA disabled")
                elif current_ema_enabled:
                    if new_ema_decay != current_ema_decay:
                        current_ema_decay = new_ema_decay
                        self.rectified_model.ema_decay = new_ema_decay
                        self.log_rectified(f"EMA decay updated to {new_ema_decay}")
                    if new_ema_mode != current_ema_mode:
                        current_ema_mode = new_ema_mode
                        self.rectified_model.ema_mode = new_ema_mode
                        self.log_rectified(f"EMA mode updated to {new_ema_mode}")

                epoch_loss = 0.0
                batches = 0
                for images, text_tensors in loader:
                    if not self.training_rectified:
                        break
                    x1 = images.to(self.rectified_model.device)
                    if cond_enabled and self.text_encoder is not None:
                        cond = self.text_encoder(text_tensors.to(self.rectified_model.device))
                    else:
                        cond = None
                    loss = self.rectified_model.train_step(x1, cond=cond, cfg_dropout_prob=cfg_dropout)
                    epoch_loss += loss
                    batches += 1
                avg_loss = epoch_loss / batches if batches else 0
                elapsed = time.time() - self.rectified_start_time
                self.log_rectified(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f} | Time: {elapsed:.1f}s")
                if preview_enabled and (epoch+1) % preview_freq == 0:
                    self.rectified_preview_with_prompt(use_ema=current_ema_enabled)

            self.training_rectified = False
            self.log_rectified("Training finished.")
        except Exception as e:
            self.log_rectified(f"Training error: {e}")
            import traceback
            traceback.print_exc()
            self.training_rectified = False

    # ===================== MODIFIED PREVIEW =====================
    def rectified_preview_with_prompt(self, use_ema=None):
        """Show reconstruction on dataset images if training_mode is dataset/mixed, else generate from noise."""
        if not self.rectified_model:
            self.log_rectified("Model not loaded for preview")
            return
        use_ema = use_ema if use_ema is not None else self.settings['ema_enabled'].get()
        steps = self.settings['preview_steps'].get()
        method = self.settings['preview_method'].get()
        n = 16  # number of images to show

        training_mode = self.settings['training_mode'].get()
        # If we have a dataset and the model is trained to map degraded -> clean, show reconstruction
        if self.image_paths and training_mode in ['dataset', 'mixed']:
            self.log_rectified("Preview: reconstructing from degraded dataset images")
            try:
                # Build a temporary dataset
                color_mode = self.settings['color_mode'].get()
                img_size = self.settings['img_size'].get()
                text_max_len = self.settings['cond_text_max_len'].get()
                labels = self.labels if self.labels else [[''] for _ in self.image_paths]
                dataset = ConditionalImageDataset(self.image_paths, labels, img_size, color_mode, {}, text_max_len)
                indices = np.random.choice(len(dataset), min(n, len(dataset)), replace=False)
                imgs = []
                texts = []
                for idx in indices:
                    img, text = dataset[idx]
                    imgs.append(img)
                    texts.append(text)
                x1 = torch.stack(imgs).to(self.rectified_model.device)
                # Degrade them using the current task
                with torch.no_grad():
                    x0 = self.rectified_model._generate_source_from_target(x1)
                # Prepare conditioning if enabled
                cond = None
                if self.settings['cond_enabled'].get() and self.text_encoder is not None:
                    text_tensor = torch.stack(texts).to(self.rectified_model.device)
                    with torch.no_grad():
                        cond = self.text_encoder(text_tensor)
                # Run the ODE from degraded to clean
                samples = self.rectified_model.sample(n_samples=len(x1), cond=cond, steps=steps, method=method,
                                                      cfg_scale=1.0, use_ema=use_ema, start_from=x0)
                self._display_preview_grid(samples)
                self.log_rectified(f"Preview: reconstruction on {len(x1)} dataset images")
            except Exception as e:
                self.log_rectified(f"Preview reconstruction error: {e}")
                # Fallback to noise generation
                self._preview_from_noise(use_ema, steps, method)
        else:
            # No dataset or training_mode is 'noise' – generate from noise
            self._preview_from_noise(use_ema, steps, method)

    def _preview_from_noise(self, use_ema, steps, method):
        """Original generation-from-noise preview."""
        prompt = self.test_prompt_entry.get().strip()
        unconditional = (prompt == "")
        cond = None
        cfg_scale = 1.0 if unconditional else self.cfg_scale.get()
        n = 16
        if not unconditional and self.settings['cond_enabled'].get() and self.text_encoder is not None:
            text_indices = text_to_indices(prompt, self.settings['cond_text_max_len'].get())
            text_tensor = torch.tensor([text_indices]*n, dtype=torch.long, device=self.rectified_model.device)
            with torch.no_grad():
                cond = self.text_encoder(text_tensor)
            self.log_rectified(f"Preview generation from noise with prompt: '{prompt}'")
        else:
            self.log_rectified("Preview generation from noise (unconditional)")
        samples = self.rectified_model.sample(n_samples=n, steps=steps, method=method,
                                              cond=cond, cfg_scale=cfg_scale, use_ema=use_ema)
        self._display_preview_grid(samples)

    def _display_preview_grid(self, samples):
        """Display a grid of samples (already in [-1,1] on GPU) on the preview canvas."""
        samples = (samples + 1) / 2
        samples = samples.clamp(0,1).cpu().numpy()
        n = len(samples)
        thumb = self.thumbnail_size
        grid_size = int(math.ceil(math.sqrt(n)))
        total_width = grid_size * thumb
        total_height = grid_size * thumb
        grid = Image.new('RGB', (total_width, total_height), color=(128,128,128))
        for i in range(n):
            row = i // grid_size
            col = i % grid_size
            if samples[i].shape[0] == 1:
                img = samples[i][0]*255
                img = np.stack([img,img,img], axis=-1).astype(np.uint8)
            else:
                img = samples[i].transpose(1,2,0)*255
                img = img.astype(np.uint8)
            pil_img = Image.fromarray(img).resize((thumb, thumb), Image.NEAREST)
            grid.paste(pil_img, (col*thumb, row*thumb))
        grid = grid.resize((256,256), Image.NEAREST)
        self.rectified_preview_photo = ImageTk.PhotoImage(grid)
        self.rectified_preview_canvas.delete("all")
        self.rectified_preview_canvas.create_image(128,128, image=self.rectified_preview_photo)

    def stop_rectified_training(self):
        self.training_rectified = False
        self.log_rectified("Training stopped.")

    def save_rectified_model(self):
        if not self.rectified_model:
            self.log_rectified("No model.")
            return
        fname = filedialog.asksaveasfilename(defaultextension=".pth", filetypes=[("PyTorch","*.pth")])
        if fname:
            # Save the target (EMA) model if EMA enabled, else the online model
            if self.rectified_model.use_ema and self.rectified_model.ema_model is not None:
                model_state = self.rectified_model.ema_model.state_dict()
                self.log_rectified("Saving EMA target model.")
            else:
                model_state = self.rectified_model.model.state_dict()
                self.log_rectified("Saving online model (EMA not enabled).")
            save_dict = {
                'model_state': model_state,
                'optimizer_state': self.rectified_model.optimizer.state_dict(),
                'settings': {k:v.get() for k,v in self.settings.items() if k.startswith('rectified') or k in ['cfg_dropout_prob','text_encoder_type','text_encoder_size','ot_task','training_mode','blur_sigma','bit_depth','pixelation_size','padding_mode','ema_enabled','ema_decay','ema_mode','use_ot','source_noise_std','pixelation_rescale_mode']},
                'task': self.rectified_model.task,
                'task_params': self.rectified_model.task_params,
                'training_mode': self.rectified_model.training_mode,
                'padding_mode': self.rectified_model.model.padding_mode,
                'ema_enabled': self.rectified_model.use_ema,
                'ema_decay': self.rectified_model.ema_decay if self.rectified_model.use_ema else None,
                'ema_mode': self.rectified_model.ema_mode if self.rectified_model.use_ema else None,
                'use_ot': self.rectified_model.use_ot,
            }
            if self.text_encoder:
                save_dict['text_encoder_state'] = self.text_encoder.state_dict()
            torch.save(save_dict, fname)
            self.log_rectified(f"Model saved to {fname}")

    def load_rectified_model(self):
        fname = filedialog.askopenfilename(filetypes=[("PyTorch","*.pth")])
        if not fname:
            return
        try:
            ckpt = torch.load(fname, map_location='cpu')
            is_legacy = 'task' not in ckpt
            if 'settings' in ckpt:
                for k, v in ckpt['settings'].items():
                    if k in self.settings:
                        self.settings[k].set(v)
            if is_legacy:
                self.settings['ot_task'].set('noise')
                self.settings['training_mode'].set('noise')
                self.log_rectified("Loaded legacy model from flow_matching.py -> set task=noise, mode=noise")
            else:
                task = ckpt.get('task', 'noise')
                task_params = ckpt.get('task_params', {})
                training_mode = ckpt.get('training_mode', 'dataset')
                padding_mode = ckpt.get('padding_mode', 'constant')
                self.settings['ot_task'].set(task)
                self.settings['training_mode'].set(training_mode)
                self.settings['padding_mode'].set(padding_mode)
                if 'blur_sigma' in task_params:
                    self.settings['blur_sigma'].set(task_params['blur_sigma'])
                if 'bit_depth' in task_params:
                    self.settings['bit_depth'].set(task_params['bit_depth'])
                if 'pixelation_size' in task_params:
                    self.settings['pixelation_size'].set(task_params['pixelation_size'])
                if 'source_noise_std' in task_params:
                    self.settings['source_noise_std'].set(task_params['source_noise_std'])
                if 'pixelation_rescale_mode' in task_params:
                    self.settings['pixelation_rescale_mode'].set(task_params['pixelation_rescale_mode'])
                # Load EMA settings if present
                if 'ema_enabled' in ckpt:
                    self.settings['ema_enabled'].set(ckpt['ema_enabled'])
                if 'ema_decay' in ckpt:
                    self.settings['ema_decay'].set(ckpt['ema_decay'])
                if 'ema_mode' in ckpt:
                    self.settings['ema_mode'].set(ckpt['ema_mode'])
                if 'use_ot' in ckpt:
                    self.settings['use_ot'].set(ckpt['use_ot'])

            self.initialize_rectified_model()
            # Load model state (which is the target/EMA state if saved that way)
            self.rectified_model.model.load_state_dict(ckpt['model_state'])
            self.rectified_model.optimizer.load_state_dict(ckpt['optimizer_state'])
            # If EMA is enabled after loading, sync ema_model with loaded model
            if self.rectified_model.use_ema and self.rectified_model.ema_model is not None:
                self.rectified_model.ema_model.load_state_dict(self.rectified_model.model.state_dict())
                self.log_rectified("EMA target initialized from loaded model.")
            if 'text_encoder_state' in ckpt and self.text_encoder:
                self.text_encoder.load_state_dict(ckpt['text_encoder_state'])
            self.log_rectified(f"Model loaded from {fname} (task: {self.settings['ot_task'].get()}, mode: {self.settings['training_mode'].get()}, padding: {self.settings['padding_mode'].get()}, OT: {self.settings['use_ot'].get()})")
        except Exception as e:
            self.log_rectified(f"Load error: {e}")

    # ========== Generation Methods (unchanged) ==========
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
        if not unconditional and self.settings['cond_enabled'].get() and self.text_encoder:
            text_indices = text_to_indices(prompt, self.settings['cond_text_max_len'].get())
            text_tensor = torch.tensor([text_indices]*n, dtype=torch.long, device=self.rectified_model.device)
            with torch.no_grad():
                cond = self.text_encoder(text_tensor)
        elif not unconditional:
            unconditional = True
            effective_cfg_scale = 1.0

        if self.progressive_grid.get():
            self.start_progressive(n, steps, method, cond, effective_cfg_scale, use_ema)
        else:
            self.generate_btn.config(state=tk.DISABLED)
            self.gen_info.config(text="Generating..." + (" with prompt" if not unconditional else ""))
            self.root.update()
            thread = threading.Thread(target=self._generate_thread,
                                      args=(n, steps, method, cond, effective_cfg_scale, use_ema), daemon=True)
            thread.start()

    def stop_progressive(self):
        self.progressive_active = False
        self.stop_prog_btn.config(state=tk.DISABLED)
        self.generate_btn.config(state=tk.NORMAL)
        self.gen_info.config(text="Stopped.")

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
                    img = samples[i][0]*255
                    img = np.stack([img,img,img], axis=-1).astype(np.uint8)
                else:
                    img = samples[i].transpose(1,2,0)*255
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
                                img = samples[i][0]*255
                                img = np.stack([img,img,img], axis=-1).astype(np.uint8)
                            else:
                                img = samples[i].transpose(1,2,0)*255
                                img = img.astype(np.uint8)
                            pil_img = Image.fromarray(img).resize((thumb, thumb), Image.NEAREST)
                            grid_img.paste(pil_img, (col*thumb, row*thumb))
                    self.root.after(0, lambda g=grid_img, s=step_idx: self._update_progressive(g, s))
                    time.sleep(0.05)
            if not self.progressive_active:
                self.root.after(0, lambda: self.gen_info.config(text="Stopped."))
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