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
    """Load an image and convert to RGB (handle transparency)."""
    img = Image.open(path)
    if img.mode == 'RGBA':
        bg = Image.new('RGB', img.size, (0, 0, 0))
        bg.paste(img, mask=img.split()[3])
        return bg
    else:
        return img.convert('RGB')

def load_image_as_grayscale(path):
    img = Image.open(path).convert('L')
    return img

def text_to_indices(text, max_len=128):
    indices = []
    for ch in text[:max_len]:
        idx = ord(ch) if ord(ch) < 256 else 0
        indices.append(idx)
    if len(indices) < max_len:
        indices += [0] * (max_len - len(indices))
    return indices

# ==================== EMA (single shadow, memory efficient) ====================
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

# ==================== Text Encoders (same as before) ====================
class TextEncoder(nn.Module):
    """Bidirectional GRU text encoder."""
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
    """Transformer text encoder."""
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

# ==================== UNet for Consistency Model (predicts clean image) ====================
class AttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(32, dim)
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
    def __init__(self, in_channels, out_channels, time_emb_dim, cond_dim, dropout=0.1, has_attn=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels))
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

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, cond_dim, dropout=0.1, has_attn=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels))
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

class ConsistencyUNet(nn.Module):
    """UNet that predicts clean image (x0) given noisy image x_t and time t."""
    def __init__(self, in_channels=3, base_channels=64, time_emb_dim=256, cond_dim=256,
                 img_size=32, channel_mult=(1, 2, 3, 4), dropout=0.1, use_attention=True):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.time_emb_dim = time_emb_dim
        self.cond_dim = cond_dim
        self.channel_mult = channel_mult
        self.use_attention = use_attention

        H, W = (img_size, img_size) if isinstance(img_size, int) else img_size
        num_down = 0
        cur_h, cur_w = H, W
        while cur_h >= 8 and cur_w >= 8:
            cur_h //= 2
            cur_w //= 2
            num_down += 1
        num_down = min(num_down, len(channel_mult))
        self.num_down = num_down
        channel_mult_used = channel_mult[:num_down+1]

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.downs = nn.ModuleList()
        cur_channels = base_channels
        for i, mult in enumerate(channel_mult_used):
            out_channels = base_channels * mult
            has_attn = use_attention
            block = DownBlock(cur_channels, out_channels, time_emb_dim, cond_dim, dropout, has_attn=has_attn)
            self.downs.append(block)
            if i < len(channel_mult_used) - 1:
                self.downs.append(nn.Conv2d(out_channels, out_channels, 4, stride=2, padding=1))
            cur_channels = out_channels

        self.mid_block1 = DownBlock(cur_channels, cur_channels, time_emb_dim, cond_dim, dropout, has_attn=use_attention)
        self.mid_block2 = UpBlock(cur_channels, cur_channels, time_emb_dim, cond_dim, dropout, has_attn=use_attention)

        self.ups = nn.ModuleList()
        rev_blocks = list(reversed(channel_mult_used))
        for i, mult in enumerate(rev_blocks):
            out_channels = base_channels * mult
            has_attn = use_attention
            block = UpBlock(cur_channels + out_channels, out_channels, time_emb_dim, cond_dim, dropout, has_attn=has_attn)
            self.ups.append(block)
            if i < len(rev_blocks) - 1:
                self.ups.append(nn.ConvTranspose2d(out_channels, out_channels, 4, stride=2, padding=1))
            cur_channels = out_channels

        self.final_conv = nn.Sequential(
            nn.GroupNorm(32, cur_channels),
            nn.SiLU(),
            nn.Conv2d(cur_channels, in_channels, 3, padding=1)
        )

    def forward(self, x, t, cond=None):
        """Returns raw clean image prediction (without skip connection)."""
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

# ==================== Consistency Model ====================
class ConsistencyModel:
    def __init__(self, in_channels=3, img_size=32, base_channels=64,
                 cond_dim=256, device=None, use_attention=True,
                 sigma_data=0.5, epsilon=0.002, T=80.0):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.cond_dim = cond_dim
        self.sigma_data = sigma_data
        self.epsilon = epsilon
        self.T = T

        self.model = ConsistencyUNet(
            in_channels=in_channels,
            base_channels=base_channels,
            time_emb_dim=base_channels*4,
            cond_dim=cond_dim,
            img_size=img_size,
            channel_mult=(1, 2, 3, 4),
            dropout=0.1,
            use_attention=use_attention
        ).to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=2e-4)
        self.ema = None

    def set_ema(self, decay=0.999):
        self.ema = EMA(self.model, decay)

    def c_skip(self, t):
        """c_skip(t) = sigma_data^2 / ((t - epsilon)^2 + sigma_data^2)"""
        t_adj = t - self.epsilon
        return self.sigma_data**2 / (t_adj**2 + self.sigma_data**2)

    def c_out(self, t):
        """c_out(t) = sigma_data * (t - epsilon) / sqrt(sigma_data^2 + t^2)"""
        t_adj = t - self.epsilon
        return self.sigma_data * t_adj / torch.sqrt(self.sigma_data**2 + t**2)

    def forward(self, x, t, cond=None):
        """Applies skip connection to produce f_theta(x_t, t)"""
        raw = self.model(x, t, cond)
        skip = self.c_skip(t).view(-1, 1, 1, 1)
        out = self.c_out(t).view(-1, 1, 1, 1)
        return skip * x + out * raw

    def train_step(self, x0, cond=None, iter_num=None, total_iters=None,
                   q=4, k=8, b=1, update_ema=True, ema_decay=0.999):
        """
        Consistency training step with shrinking delta t.
        x0: clean images [B, C, H, W] in [-1,1] range.
        cond: text condition [B, cond_dim] or None.
        iter_num, total_iters: for scheduling r/t.
        """
        batch_size = x0.size(0)
        x0 = x0.to(self.device)
        if cond is not None:
            cond = cond.to(self.device)
        else:
            cond = torch.zeros(batch_size, self.cond_dim, device=self.device)

        # Sample t from lognormal distribution (mean=-1.1, std=2.0)
        # but clamp to [epsilon, T]
        log_t = torch.randn(batch_size, device=self.device) * 2.0 - 1.1
        t = torch.exp(log_t).clamp(self.epsilon, self.T)

        # Sample noise direction
        eps = torch.randn_like(x0)

        # Compute r based on mapping function
        # r/t = 1 - (1/q^a) * n(t), a = floor(iter / d), d = total_iters/4
        if iter_num is not None and total_iters is not None:
            a = iter_num // (total_iters // 4)
            a = max(0, min(a, 10))  # limit
            q_pow = q ** a
            # n(t) = 1 + k * sigmoid(-b * t)
            n_t = 1.0 + k * torch.sigmoid(-b * t)
            r_div_t = 1.0 - (1.0 / q_pow) * n_t
            # Clamp to [0, 0.999] to avoid r negative or too close
            r_div_t = r_div_t.clamp(0.0, 0.999)
            r = t * r_div_t
            # Ensure r >= epsilon
            r = r.clamp(self.epsilon, self.T)
        else:
            # Fallback: fixed small delta (like distillation mode)
            r = t * 0.99

        # Compute x_t and x_r
        x_t = x0 + t.view(-1,1,1,1) * eps
        x_r = x0 + r.view(-1,1,1,1) * eps

        # Predictions
        pred_t = self.forward(x_t, t, cond)
        with torch.no_grad():
            target_r = self.forward(x_r, r, cond) if self.ema is None else self.ema_forward(x_r, r, cond)

        loss = self.criterion(pred_t, target_r)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if update_ema and self.ema is not None:
            self.ema.update()

        return loss.item()

    def ema_forward(self, x, t, cond):
        """Forward pass using EMA model if available, else normal model with stopgrad."""
        if self.ema is not None:
            # Temporarily apply EMA shadow
            self.ema.apply_shadow()
            out = self.forward(x, t, cond)
            self.ema.restore()
            return out
        else:
            return self.forward(x, t, cond).detach()

    @torch.no_grad()
    def sample_one_step(self, n_samples=16, cond=None, use_ema=True):
        """Single-step generation: sample noise -> consistency model."""
        if use_ema and self.ema is not None:
            self.ema.apply_shadow()
        img_shape = (self.model.in_channels, self.img_size[0], self.img_size[1])
        x_T = torch.randn(n_samples, *img_shape, device=self.device) * self.T
        if cond is None:
            cond = torch.zeros(n_samples, self.cond_dim, device=self.device)
        else:
            cond = cond.to(self.device)
        samples = self.forward(x_T, torch.full((n_samples,), self.T, device=self.device), cond)
        if use_ema and self.ema is not None:
            self.ema.restore()
        return samples

    @torch.no_grad()
    def sample_multistep(self, n_samples=16, cond=None, steps=10, method='euler',
                         use_ema=True, progress_callback=None):
        """
        Multistep sampling following consistency model algorithm.
        Steps: from noise, iteratively add noise and denoise.
        """
        if use_ema and self.ema is not None:
            self.ema.apply_shadow()
        img_shape = (self.model.in_channels, self.img_size[0], self.img_size[1])
        # Time points from T down to epsilon, log spaced
        rho = 7.0
        t_points = []
        for i in range(steps+1):
            u = i / steps
            t = (self.epsilon**(1/rho) + u * (self.T**(1/rho) - self.epsilon**(1/rho)))**rho
            t_points.append(t)
        t_points = list(reversed(t_points))  # now from T to epsilon
        # Start from noise
        x = torch.randn(n_samples, *img_shape, device=self.device) * self.T
        if cond is None:
            cond = torch.zeros(n_samples, self.cond_dim, device=self.device)
        else:
            cond = cond.to(self.device)

        for i in range(steps):
            t_curr = t_points[i]
            t_next = t_points[i+1]
            # Denoise current sample
            x = self.forward(x, torch.full((n_samples,), t_curr, device=self.device), cond)
            if i < steps - 1:
                # Add noise back: x -> x + sqrt(t_next^2 - epsilon^2) * noise
                noise = torch.randn_like(x)
                sigma_next = math.sqrt(t_next**2 - self.epsilon**2)
                x = x + sigma_next * noise
            if progress_callback:
                progress_callback(i+1, x)
        if use_ema and self.ema is not None:
            self.ema.restore()
        return x

    @torch.no_grad()
    def sample_step_by_step(self, n_samples=16, cond=None, steps=10, method='euler',
                            use_ema=True):
        """Generator that yields intermediate samples for progressive display."""
        if use_ema and self.ema is not None:
            self.ema.apply_shadow()
        img_shape = (self.model.in_channels, self.img_size[0], self.img_size[1])
        rho = 7.0
        t_points = []
        for i in range(steps+1):
            u = i / steps
            t = (self.epsilon**(1/rho) + u * (self.T**(1/rho) - self.epsilon**(1/rho)))**rho
            t_points.append(t)
        t_points = list(reversed(t_points))
        x = torch.randn(n_samples, *img_shape, device=self.device) * self.T
        if cond is None:
            cond = torch.zeros(n_samples, self.cond_dim, device=self.device)
        else:
            cond = cond.to(self.device)

        for i in range(steps):
            t_curr = t_points[i]
            t_next = t_points[i+1]
            x = self.forward(x, torch.full((n_samples,), t_curr, device=self.device), cond)
            yield i+1, x.clone()
            if i < steps - 1:
                noise = torch.randn_like(x)
                sigma_next = math.sqrt(t_next**2 - self.epsilon**2)
                x = x + sigma_next * noise
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
class ConsistencyApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Consistency Model - Pixel Space")

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
        self.training = False
        self.model = None
        self.text_encoder = None
        self.current_epoch = 0
        self.message_queue_dataset = queue.Queue()
        self.message_queue_train = queue.Queue()
        self.progressive_active = False

        # Settings variables
        self.settings = {
            'img_size': tk.IntVar(value=32),
            'color_mode': tk.StringVar(value='rgb'),
            'base_channels': tk.IntVar(value=64),
            'batch_size': tk.IntVar(value=16),
            'lr': tk.DoubleVar(value=2e-4),
            'use_attention': tk.BooleanVar(value=True),
            'ema_enabled': tk.BooleanVar(value=True),
            'ema_decay': tk.DoubleVar(value=0.999),
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
            'preview_steps': tk.IntVar(value=10),
            'preview_epoch_freq': tk.IntVar(value=5),
        }

        self.aug_settings = {
            'flip_horizontal': tk.BooleanVar(value=True),
            'rotation': tk.BooleanVar(value=False),
        }

        self.ode_steps = tk.IntVar(value=10)      # default 10 steps
        self.thumbnail_size = 128

        self.setup_gui()
        self.root.after(100, self.process_messages_dataset)
        self.root.after(100, self.process_messages_train)

    def setup_gui(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.dataset_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.dataset_tab, text='Dataset')
        self.setup_dataset_tab()

        self.train_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.train_tab, text='Consistency Training')
        self.setup_train_tab()

        self.settings_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.settings_tab, text='Settings')
        self.setup_settings_tab()

        self.generation_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.generation_tab, text='Generation')
        self.setup_generation_tab()

        self.status_label = tk.Label(self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    # Dataset Tab (identical to before)
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

    # Training Tab (Consistency)
    def setup_train_tab(self):
        main_frame = tk.Frame(self.train_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_frame = tk.Frame(main_frame, width=300)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0,10))
        left_frame.pack_propagate(False)

        tk.Label(left_frame, text="Consistency Model Training", font=("Arial",12,"bold")).pack(pady=(0,10))

        cond_toggle_frame = tk.LabelFrame(left_frame, text="Conditional Training", padx=5, pady=5)
        cond_toggle_frame.pack(fill=tk.X, pady=5)
        self.cond_cb = tk.Checkbutton(cond_toggle_frame, text="Enable text conditioning",
                                      variable=self.settings['cond_enabled'])
        self.cond_cb.pack(anchor='w')
        tk.Label(cond_toggle_frame, text="Labels must be loaded in Dataset tab", fg="gray").pack(anchor='w')

        tk.Button(left_frame, text="Initialize Consistency Model", command=self.initialize_model, width=20).pack(pady=5)
        epoch_frame = tk.Frame(left_frame)
        epoch_frame.pack(pady=5)
        tk.Label(epoch_frame, text="Epochs:").pack(side=tk.LEFT)
        self.epoch_var = tk.StringVar(value="200")
        tk.Entry(epoch_frame, textvariable=self.epoch_var, width=8).pack(side=tk.LEFT, padx=5)

        tk.Button(left_frame, text="Start Training", command=self.start_training,
                  width=20, bg="lightgreen").pack(pady=5)
        tk.Button(left_frame, text="Stop Training", command=self.stop_training,
                  width=20, bg="salmon").pack(pady=5)
        tk.Button(left_frame, text="Save Model", command=self.save_model, width=20).pack(pady=5)
        tk.Button(left_frame, text="Load Model", command=self.load_model, width=20).pack(pady=5)

        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        preview_canvas_frame = tk.LabelFrame(right_frame, text="Generated Samples (preview)", padx=5, pady=5)
        preview_canvas_frame.pack(fill=tk.BOTH, expand=True, pady=(0,5))
        self.preview_canvas = tk.Canvas(preview_canvas_frame, bg='gray', width=256, height=256)
        self.preview_canvas.pack()

        prompt_frame = tk.LabelFrame(right_frame, text="Test prompt (for manual preview)", padx=5, pady=5)
        prompt_frame.pack(fill=tk.X, pady=(0,5))
        self.test_prompt_entry = tk.Entry(prompt_frame)
        self.test_prompt_entry.insert(0, "a cute cat")
        self.test_prompt_entry.pack(fill=tk.X, pady=2)
        tk.Button(prompt_frame, text="Generate Preview (manual)", command=self.preview_with_prompt, width=20).pack(pady=2)

        log_frame = tk.LabelFrame(right_frame, text="Log", padx=5, pady=5)
        log_frame.pack(fill=tk.BOTH, expand=True)
        self.train_log_text = tk.Text(log_frame, height=15, font=("Courier",9))
        self.train_log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = tk.Scrollbar(log_frame, command=self.train_log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.train_log_text.config(yscrollcommand=scrollbar.set)

    # Settings Tab
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
        arch_frame = tk.LabelFrame(scrollable_frame, text="Consistency Model - Architecture", padx=10, pady=10)
        arch_frame.pack(fill=tk.X, pady=5)
        f = tk.Frame(arch_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="UNet base channels:", width=20, anchor='w').pack(side=tk.LEFT)
        spin = ttk.Spinbox(f, from_=32, to=128, textvariable=self.settings['base_channels'], width=8)
        spin.pack(side=tk.RIGHT)
        f = tk.Frame(arch_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Use Self-Attention:", width=20, anchor='w').pack(side=tk.LEFT)
        cb = tk.Checkbutton(f, variable=self.settings['use_attention'])
        cb.pack(side=tk.RIGHT)
        f = tk.Frame(arch_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="EMA enabled:", width=20, anchor='w').pack(side=tk.LEFT)
        cb_ema = tk.Checkbutton(f, variable=self.settings['ema_enabled'])
        cb_ema.pack(side=tk.RIGHT)
        f = tk.Frame(arch_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="EMA decay:", width=20, anchor='w').pack(side=tk.LEFT)
        spin_ema = ttk.Entry(f, textvariable=self.settings['ema_decay'], width=8)
        spin_ema.pack(side=tk.RIGHT)

        # Training hyperparams
        train_frame = tk.LabelFrame(scrollable_frame, text="Training", padx=10, pady=10)
        train_frame.pack(fill=tk.X, pady=5)
        params = [
            ("Batch size:", 'batch_size', 1, 32, int),
            ("Learning rate:", 'lr', 1e-5, 1e-2, float),
        ]
        for label, key, low, high, typ in params:
            f = tk.Frame(train_frame); f.pack(fill=tk.X, pady=2)
            tk.Label(f, text=label, width=20, anchor='w').pack(side=tk.LEFT)
            if typ == int:
                spin = ttk.Spinbox(f, from_=low, to=high, textvariable=self.settings[key], width=8)
            else:
                spin = ttk.Entry(f, textvariable=self.settings[key], width=8)
            spin.pack(side=tk.RIGHT)

        # Preview during training
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
        tk.Label(f, text="Preview steps (multistep):", width=20, anchor='w').pack(side=tk.LEFT)
        spin_steps = ttk.Spinbox(f, from_=1, to=200, textvariable=self.settings['preview_steps'], width=8)
        spin_steps.pack(side=tk.RIGHT)

        # Text conditioning
        cond_frame = tk.LabelFrame(scrollable_frame, text="Text Conditioning", padx=10, pady=10)
        cond_frame.pack(fill=tk.X, pady=5)
        f = tk.Frame(cond_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Encoder type:", width=20, anchor='w').pack(side=tk.LEFT)
        type_combo = ttk.Combobox(f, textvariable=self.settings['text_encoder_type'],
                                  values=['BiGRU', 'BiTransformer'], state='readonly', width=12)
        type_combo.pack(side=tk.RIGHT)
        f = tk.Frame(cond_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Encoder size preset:", width=20, anchor='w').pack(side=tk.LEFT)
        size_combo = ttk.Combobox(f, textvariable=self.settings['text_encoder_size'],
                                  values=['tiny', 'small', 'medium', 'large'], state='readonly', width=8)
        size_combo.pack(side=tk.RIGHT)
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
        self.log_train(f"Applied {enc_type} {size} preset dims.")

    # Generation Tab
    def setup_generation_tab(self):
        main_frame = tk.Frame(self.generation_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        tk.Label(main_frame, text="Generate Images (Consistency Model)", font=("Arial",14,"bold")).pack(pady=(0,10))

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

        tk.Label(ctrl_frame, text="Steps:").pack(side=tk.LEFT, padx=(10,0))
        tk.Spinbox(ctrl_frame, from_=1, to=200, textvariable=self.ode_steps, width=5).pack(side=tk.LEFT, padx=5)

        self.progressive_grid = tk.BooleanVar(value=False)
        prog_check = tk.Checkbutton(ctrl_frame, text="Progressive grid", variable=self.progressive_grid)
        prog_check.pack(side=tk.LEFT, padx=10)

        tk.Label(ctrl_frame, text="Update interval:").pack(side=tk.LEFT)
        self.prog_interval = tk.IntVar(value=1)
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

    # Core functions
    def log_dataset(self, msg):
        self.message_queue_dataset.put(msg)

    def log_train(self, msg):
        self.message_queue_train.put(msg)

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

    def process_messages_train(self):
        try:
            while True:
                msg = self.message_queue_train.get_nowait()
                self.train_log_text.insert(tk.END, f"{time.strftime('%H:%M:%S')} - {msg}\n")
                self.train_log_text.see(tk.END)
                self.status_label.config(text=msg[:50])
                print(f"Train: {msg}")
        except queue.Empty:
            pass
        self.root.after(100, self.process_messages_train)

    def add_images(self):
        files = filedialog.askopenfilenames(filetypes=[("Images", "*.jpg *.jpeg *.png *.jfif *.webp *.bmp")])
        for f in files:
            if f not in self.image_paths:
                self.image_paths.append(f)
                self.image_listbox.insert(tk.END, os.path.basename(f))
        self.log_dataset(f"Added {len(files)} images. Total: {len(self.image_paths)}")
        self.log_train(f"Added {len(files)} images. Total: {len(self.image_paths)}")

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
        self.log_train(f"Added {count} images from folder. Total: {len(self.image_paths)}")

    def clear_images(self):
        self.image_paths = []
        self.labels = []
        self.image_listbox.delete(0, tk.END)
        self.csv_status.config(text="No CSV loaded", fg="red")
        self.log_dataset("Cleared all images")
        self.log_train("Cleared all images")

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
                self.log_train(f"Warning: duplicate basename '{base}'")
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
            self.log_train(f"Error reading CSV: {e}")
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
        self.log_train(f"CSV loaded. {len(self.image_paths)-unknown} images matched with labels.")

    def use_filenames_as_labels(self):
        self.labels = [[os.path.splitext(os.path.basename(p))[0]] for p in self.image_paths]
        self.csv_status.config(text="Using filenames as labels", fg="blue")
        self.log_train("Using filenames as labels.")

    def use_folders_as_labels(self):
        self.labels = []
        for p in self.image_paths:
            folder = os.path.basename(os.path.dirname(p)) or 'unknown'
            self.labels.append([folder])
        self.csv_status.config(text="Using folder names as labels", fg="blue")
        self.log_train("Using folder names as labels.")

    # Model initialization
    def initialize_model(self):
        try:
            cond_enabled = self.settings['cond_enabled'].get()
            cond_dim = self.settings['cond_dim'].get() if cond_enabled else 1
            in_channels = 3 if self.settings['color_mode'].get() == 'rgb' else 1
            img_size = self.settings['img_size'].get()
            use_attention = self.settings['use_attention'].get()

            self.model = ConsistencyModel(
                in_channels=in_channels,
                img_size=img_size,
                base_channels=self.settings['base_channels'].get(),
                cond_dim=cond_dim,
                use_attention=use_attention
            )
            for pg in self.model.optimizer.param_groups:
                pg['lr'] = self.settings['lr'].get()

            if self.settings['ema_enabled'].get():
                self.model.set_ema(decay=self.settings['ema_decay'].get())
                self.log_train(f"EMA enabled with decay {self.settings['ema_decay'].get()}")
            else:
                self.model.ema = None

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
                self.text_encoder.to(self.model.device)
                self.optimizer = optim.Adam(
                    list(self.model.model.parameters()) + list(self.text_encoder.parameters()),
                    lr=self.settings['lr'].get()
                )
                self.model.optimizer = self.optimizer
                self.log_train(f"Conditional Consistency Model with {enc_type} encoder initialized.")
            else:
                self.text_encoder = None
                self.log_train("Unconditional Consistency Model initialized.")
        except Exception as e:
            self.log_train(f"Error initializing model: {e}")

    def start_training(self):
        if not self.image_paths:
            self.log_train("No images!")
            return
        if not self.model:
            self.log_train("Model not initialized!")
            return
        if self.training:
            self.log_train("Already training.")
            return

        cond_enabled = self.settings['cond_enabled'].get()
        if cond_enabled and not self.labels:
            self.log_train("Conditional training requires labels. Load CSV or use filenames/folders as labels.")
            return

        try:
            epochs = int(self.epoch_var.get())
        except:
            self.log_train("Invalid epochs")
            return

        self.training = True
        self.current_epoch = 0
        self.start_time = time.time()

        thread = threading.Thread(target=self.train_loop, args=(epochs,), daemon=True)
        thread.start()
        self.log_train(f"Consistency training started for {epochs} epochs.")

    def train_loop(self, epochs):
        try:
            batch_size = self.settings['batch_size'].get()
            num_workers = 0
            img_size = self.settings['img_size'].get()
            color_mode = self.settings['color_mode'].get()
            cond_enabled = self.settings['cond_enabled'].get()
            text_max_len = self.settings['cond_text_max_len'].get()
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

            total_iters = epochs * len(loader)

            for epoch in range(epochs):
                if not self.training:
                    break
                self.current_epoch = epoch
                epoch_loss = 0.0
                batches = 0

                for images, text_tensors in loader:
                    if not self.training:
                        break
                    x0 = images.to(self.model.device)
                    if cond_enabled and self.text_encoder is not None:
                        cond = self.text_encoder(text_tensors.to(self.model.device))
                    else:
                        cond = None
                    # compute iteration number
                    iter_num = epoch * len(loader) + batches
                    loss = self.model.train_step(x0, cond=cond,
                                                 iter_num=iter_num,
                                                 total_iters=total_iters,
                                                 update_ema=use_ema,
                                                 ema_decay=ema_decay)
                    epoch_loss += loss
                    batches += 1

                avg_loss = epoch_loss / batches if batches else 0
                elapsed = time.time() - self.start_time
                self.log_train(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f} | Time: {elapsed:.1f}s")

                if preview_enabled and (epoch+1) % preview_freq == 0:
                    self.preview_with_prompt(use_ema=use_ema)

            self.training = False
            self.log_train("Consistency training finished.")
        except Exception as e:
            self.log_train(f"Training error: {e}")
            import traceback
            traceback.print_exc()
            self.training = False

    def preview_with_prompt(self, use_ema=None):
        if not self.model:
            self.log_train("Model not initialized for preview")
            return
        if use_ema is None:
            use_ema = self.settings['ema_enabled'].get()
        try:
            prompt = self.test_prompt_entry.get().strip()
            unconditional = (prompt == "")
            cond = None
            steps = self.settings['preview_steps'].get()

            if not unconditional and self.settings['cond_enabled'].get() and self.text_encoder is not None:
                text_indices = text_to_indices(prompt, self.settings['cond_text_max_len'].get())
                text_tensor = torch.tensor([text_indices] * 16, dtype=torch.long, device=self.model.device)
                with torch.no_grad():
                    cond = self.text_encoder(text_tensor)
                self.log_train(f"Preview: conditional with prompt '{prompt}'")
            elif not unconditional:
                self.log_train("Preview: conditioning disabled, generating unconditional")
                unconditional = True
            else:
                self.log_train("Preview: unconditional generation")

            samples = self.model.sample_multistep(n_samples=16, steps=steps, cond=cond,
                                                  use_ema=use_ema, method='euler')
            samples = (samples + 1) / 2
            samples = samples.clamp(0,1).cpu().numpy()
            thumb = 64
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
            self.preview_photo = ImageTk.PhotoImage(grid)
            self.preview_canvas.delete("all")
            self.preview_canvas.create_image(128,128, image=self.preview_photo)
        except Exception as e:
            self.log_train(f"Preview error: {e}")

    def stop_training(self):
        self.training = False
        self.log_train("Training stopped.")

    def save_model(self):
        if not self.model:
            self.log_train("No model.")
            return
        fname = filedialog.asksaveasfilename(defaultextension=".pth", filetypes=[("PyTorch","*.pth")])
        if fname:
            save_dict = {
                'model_state': self.model.model.state_dict(),
                'optimizer_state': self.model.optimizer.state_dict(),
                'settings': {k:v.get() for k,v in self.settings.items() if k.startswith('base') or k in ['cond_enabled','text_encoder_type','text_encoder_size']},
                'ema_shadow': self.model.ema.shadow if self.model.ema else None,
            }
            if self.text_encoder is not None:
                save_dict['text_encoder_state'] = self.text_encoder.state_dict()
            torch.save(save_dict, fname)
            self.log_train(f"Model saved to {fname}")

    def load_model(self):
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
            self.initialize_model()
            self.model.model.load_state_dict(ckpt['model_state'])
            self.model.optimizer.load_state_dict(ckpt['optimizer_state'])
            if 'text_encoder_state' in ckpt and self.text_encoder is not None:
                self.text_encoder.load_state_dict(ckpt['text_encoder_state'])
            if 'ema_shadow' in ckpt and self.model.ema is not None:
                self.model.ema.shadow = ckpt['ema_shadow']
                self.log_train("EMA shadow weights restored.")
            self.log_train(f"Model loaded from {fname}")
        except Exception as e:
            self.log_train(f"Load error: {e}")

    # Generation methods
    def generate_samples(self):
        if not self.model:
            self.gen_info.config(text="Model not loaded!")
            return
        if self.settings['cond_enabled'].get() and self.text_encoder is None:
            self.gen_info.config(text="Text encoder not initialized! Load conditional model.")
            return

        n = self.gen_count.get()
        steps = self.ode_steps.get()
        prompt = self.gen_prompt.get().strip()
        use_ema = self.settings['ema_enabled'].get()

        unconditional = (prompt == "")
        cond = None

        if not unconditional and self.settings['cond_enabled'].get() and self.text_encoder is not None:
            text_indices = text_to_indices(prompt, self.settings['cond_text_max_len'].get())
            text_tensor = torch.tensor([text_indices] * n, dtype=torch.long, device=self.model.device)
            with torch.no_grad():
                cond = self.text_encoder(text_tensor)
        elif not unconditional:
            unconditional = True

        if self.progressive_grid.get():
            self.start_progressive(n, steps, cond, use_ema)
        else:
            self.generate_btn.config(state=tk.DISABLED)
            if unconditional:
                self.gen_info.config(text="Generating unconditionally...")
            else:
                self.gen_info.config(text=f"Generating with prompt: {prompt}")
            self.root.update()
            thread = threading.Thread(target=self._generate_thread,
                                      args=(n, steps, cond, use_ema), daemon=True)
            thread.start()

    def stop_progressive(self):
        self.progressive_active = False
        self.stop_prog_btn.config(state=tk.DISABLED)
        self.generate_btn.config(state=tk.NORMAL)
        self.gen_info.config(text="Progressive generation stopped.")

    def start_progressive(self, n, steps, cond, use_ema):
        self.progressive_active = True
        self.generate_btn.config(state=tk.DISABLED)
        self.stop_prog_btn.config(state=tk.NORMAL)
        self.gen_info.config(text="Progressive generation...")
        thread = threading.Thread(target=self._progressive_thread,
                                  args=(n, steps, cond, use_ema), daemon=True)
        thread.start()

    def _generate_thread(self, n, steps, cond, use_ema):
        try:
            samples = self.model.sample_multistep(n_samples=n, steps=steps, cond=cond,
                                                  use_ema=use_ema, method='euler')
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

    def _progressive_thread(self, n, steps, cond, use_ema):
        try:
            thumb = self.thumbnail_size
            interval = self.prog_interval.get()
            generator = self.model.sample_step_by_step(n_samples=n, steps=steps, cond=cond, use_ema=use_ema)
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
        self.gen_info.config(text=f"Step {step_idx}/{self.ode_steps.get()}")

# ==================== Main ====================
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    root = tk.Tk()
    app = ConsistencyApp(root)
    root.mainloop()