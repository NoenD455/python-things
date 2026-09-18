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
import io

# ==================== Helpers ====================

def load_image_as_rgb(path):
    img = Image.open(path)
    if img.mode == 'RGBA':
        bg = Image.new('RGB', img.size, (0, 0, 0))
        bg.paste(img, mask=img.split()[3])
        return bg
    return img.convert('RGB')

def load_image_as_grayscale(path):
    return Image.open(path).convert('L')

def text_to_indices(text, max_len=128):
    idx = [ord(c) if ord(c) < 256 else 0 for c in text[:max_len]]
    if len(idx) < max_len:
        idx += [0] * (max_len - len(idx))
    return idx

# ==================== Positional Embeddings ====================

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

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, device=pos.device, dtype=pos.dtype)
    omega = 1.0 / (10000 ** (2 * omega / embed_dim))
    out = pos[:, None] * omega[None, :]
    return torch.cat([torch.sin(out), torch.cos(out)], dim=-1)

def get_2d_sincos_pos_embed(embed_dim, grid_h, grid_w):
    half_dim = embed_dim // 2
    dim_h = half_dim
    dim_w = embed_dim - half_dim

    pos_h = torch.arange(grid_h, dtype=torch.float32)
    pos_w = torch.arange(grid_w, dtype=torch.float32)

    emb_h = get_1d_sincos_pos_embed_from_grid(dim_h, pos_h)
    emb_w = get_1d_sincos_pos_embed_from_grid(dim_w, pos_w)

    emb_h = emb_h.unsqueeze(1).expand(grid_h, grid_w, -1)
    emb_w = emb_w.unsqueeze(0).expand(grid_h, grid_w, -1)

    pos_embed = torch.cat([emb_h, emb_w], dim=-1).flatten(0, 1).unsqueeze(0)
    return pos_embed

# ==================== RoPE (2D) ====================
# NOTE: internal helper is named '_rotate' to avoid clashing with
# nn.Module._apply (which PyTorch uses for .to(device)/.cuda()/etc.).

class RoPE2D(nn.Module):
    """Rotary Position Embedding for 2D grid tokens.
    head_dim must be divisible by 4."""
    def __init__(self, head_dim, base=10000.0):
        super().__init__()
        self.head_dim = head_dim
        half = head_dim // 2
        inv_freq = 1.0 / (base ** (torch.arange(0, half, 2).float() / half))
        self.register_buffer('inv_freq', inv_freq)

    def _rotate(self, x, positions):
        # x: (B, H, N, half)  positions: (N,) float
        angles = positions[:, None] * self.inv_freq[None, :].to(positions.device)
        cos = angles.cos()[None, None, :, :]
        sin = angles.sin()[None, None, :, :]
        x_r = x.view(*x.shape[:-1], -1, 2)
        x1, x2 = x_r.unbind(-1)
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
        return torch.stack([out1, out2], dim=-1).flatten(-2)

    def forward(self, q, k, positions_h, positions_w):
        half = self.head_dim // 2
        qh, qw = q[..., :half], q[..., half:]
        kh, kw = k[..., :half], k[..., half:]
        qh = self._rotate(qh, positions_h)
        qw = self._rotate(qw, positions_w)
        kh = self._rotate(kh, positions_h)
        kw = self._rotate(kw, positions_w)
        return torch.cat([qh, qw], dim=-1), torch.cat([kh, kw], dim=-1)

# ==================== Patchifier / Unpatchifier ====================

class Patchifier(nn.Module):
    def __init__(self, in_channels, patch_size, token_dim, ptype='conv'):
        super().__init__()
        self.patch_size = patch_size
        self.ptype = ptype
        if ptype == 'conv':
            self.proj = nn.Conv2d(in_channels, token_dim,
                                  kernel_size=patch_size, stride=patch_size)
        else:
            self.proj = nn.Linear(in_channels * patch_size * patch_size, token_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        Hp = H // self.patch_size
        Wp = W // self.patch_size
        if self.ptype == 'conv':
            out = self.proj(x)                            # (B, D, Hp, Wp)
            tokens = out.flatten(2).transpose(1, 2)       # (B, N, D)
        else:
            x = x.view(B, C, Hp, self.patch_size, Wp, self.patch_size)
            x = x.permute(0, 2, 4, 1, 3, 5).contiguous()  # (B, Hp, Wp, C, p, p)
            x = x.view(B, Hp * Wp, -1)                    # (B, N, C*p*p)
            tokens = self.proj(x)
        return tokens, Hp, Wp

class Unpatchifier(nn.Module):
    def __init__(self, out_channels, patch_size, token_dim, ptype='conv'):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.ptype = ptype
        if ptype == 'conv':
            self.proj = nn.ConvTranspose2d(token_dim, out_channels,
                                           kernel_size=patch_size, stride=patch_size)
        else:
            self.proj = nn.Linear(token_dim, out_channels * patch_size * patch_size)

    def forward(self, x, Hp, Wp):
        B, N, D = x.shape
        if self.ptype == 'conv':
            x = x.transpose(1, 2).view(B, D, Hp, Wp)
            out = self.proj(x)
        else:
            x = self.proj(x)                              # (B, N, C*p*p)
            x = x.view(B, Hp, Wp, self.out_channels, self.patch_size, self.patch_size)
            x = x.permute(0, 3, 1, 4, 2, 5).contiguous()  # (B, C, Hp, p, Wp, p)
            out = x.view(B, self.out_channels, Hp * self.patch_size, Wp * self.patch_size)
        return out

# ==================== Text Encoders ====================

class TextEncoder(nn.Module):
    def __init__(self, vocab_size=256, embed_dim=64, hidden_size=64, num_layers=2, cond_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_size, num_layers,
                          batch_first=True, bidirectional=True,
                          dropout=0.1 if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size * 2, cond_dim)

    def forward(self, x):
        emb = self.embedding(x)
        _, h = self.gru(emb)
        return self.fc(torch.cat([h[-2], h[-1]], dim=1))

class TransformerTextEncoder(nn.Module):
    def __init__(self, vocab_size=256, embed_dim=128, num_heads=4, num_layers=3,
                 ff_dim=256, cond_dim=512, max_len=128, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, embed_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                           dim_feedforward=ff_dim, dropout=dropout,
                                           batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, cond_dim)

    def forward(self, x):
        B, T = x.shape
        x = self.transformer(self.embedding(x) + self.pos_embedding[:, :T, :])
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

def get_encoder_config(t, s):
    return TEXT_ENCODER_PRESETS[t][s]

# ==================== VAE ====================

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
        num_blocks, mults = VAE_SIZE_CONFIGS[size]
        self.down = nn.ModuleList()
        c = in_channels
        for m in mults:
            oc = max(1, int(base_channels * m))
            self.down.append(nn.Conv2d(c, oc, 3, stride=2, padding=1))
            self.down.append(nn.LeakyReLU(0.2))
            c = oc
        self.conv_mu = nn.Conv2d(c, latent_channels, 3, padding=1)
        self.conv_logvar = nn.Conv2d(c, latent_channels, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((latent_h, latent_w))

    def forward(self, x):
        for l in self.down:
            x = l(x)
        return self.pool(self.conv_mu(x)), self.pool(self.conv_logvar(x))

class FlexDecoder(nn.Module):
    def __init__(self, latent_channels=8, base_channels=32, out_channels=3,
                 latent_h=4, latent_w=4, size='big'):
        super().__init__()
        self.latent_h, self.latent_w = latent_h, latent_w
        num_blocks, mults = VAE_SIZE_CONFIGS[size]
        high = max(1, int(base_channels * mults[-1]))
        self.init_conv = nn.Conv2d(latent_channels, high, 3, padding=1)
        self.act = nn.LeakyReLU(0.2)
        self.up = nn.ModuleList()
        c = high
        for i in range(num_blocks):
            oc = (max(1, int(base_channels * mults[num_blocks - 2 - i]))
                  if i < num_blocks - 1 else base_channels)
            self.up.append(nn.ConvTranspose2d(c, oc, 4, stride=2, padding=1))
            self.up.append(nn.LeakyReLU(0.2))
            c = oc
        self.conv_out = nn.Conv2d(c, out_channels, 3, padding=1)

    def forward(self, z, target_size=None):
        if z.shape[2] != self.latent_h or z.shape[3] != self.latent_w:
            z = F_nn.interpolate(z, size=(self.latent_h, self.latent_w),
                                 mode='bilinear', align_corners=False)
        x = self.act(self.init_conv(z))
        for l in self.up:
            x = l(x)
        out = torch.tanh(self.conv_out(x))
        if target_size and (out.shape[2] != target_size[0] or out.shape[3] != target_size[1]):
            out = F_nn.interpolate(out, size=target_size,
                                   mode='bilinear', align_corners=False)
        return out

class FlexVAE(nn.Module):
    def __init__(self, in_channels=3, base_channels=32, latent_channels=8,
                 latent_h=4, latent_w=4, size='big'):
        super().__init__()
        self.encoder = FlexEncoder(in_channels, base_channels, latent_channels,
                                    latent_h, latent_w, size)
        self.decoder = FlexDecoder(latent_channels, base_channels, in_channels,
                                    latent_h, latent_w, size)
        self.latent_channels = latent_channels
        self.latent_h, self.latent_w = latent_h, latent_w
        self.size = size

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z, (x.shape[2], x.shape[3])), mu, logvar

    def encode(self, x):
        return self.encoder(x)[0]

    def decode(self, z, target_size=None):
        return self.decoder(z, target_size)

# ==================== DiT Components ====================

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
            emb = emb + self.cond_proj(cond)
        return emb

class DiTBlock(nn.Module):
    """Standard MHA block (used for sinusoidal / learned pos embed)."""
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads,
                                           dropout=dropout, batch_first=True)
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

    def forward(self, x, c, positions_h=None, positions_w=None):
        s_msa, sc_msa, g_msa, s_mlp, sc_mlp, g_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)
        nx = self.norm1(x) * (1 + sc_msa.unsqueeze(1)) + s_msa.unsqueeze(1)
        a, _ = self.attn(nx, nx, nx)
        x = x + g_msa.unsqueeze(1) * a
        nx = self.norm2(x) * (1 + sc_mlp.unsqueeze(1)) + s_mlp.unsqueeze(1)
        x = x + g_mlp.unsqueeze(1) * self.mlp(nx)
        return x

class DiTBlockRoPE(nn.Module):
    """Attention block using 2D RoPE (no additive pos embed on tokens)."""
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        if self.head_dim % 4 != 0:
            raise ValueError(
                f"RoPE requires head_dim divisible by 4. "
                f"Got token_dim={hidden_size}, num_heads={num_heads} "
                f"-> head_dim={self.head_dim}."
            )
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.attn_proj = nn.Linear(hidden_size, hidden_size)
        self.rope = RoPE2D(self.head_dim)
        self.dropout = nn.Dropout(dropout)

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

    def forward(self, x, c, positions_h=None, positions_w=None):
        s_msa, sc_msa, g_msa, s_mlp, sc_mlp, g_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)

        nx = self.norm1(x) * (1 + sc_msa.unsqueeze(1)) + s_msa.unsqueeze(1)
        B, N, C = nx.shape
        qkv = self.qkv(nx).view(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                 # (B, H, N, head_dim)
        if positions_h is not None:
            q, k = self.rope(q, k, positions_h, positions_w)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.attn_proj(out)
        x = x + g_msa.unsqueeze(1) * out

        nx = self.norm2(x) * (1 + sc_mlp.unsqueeze(1)) + s_mlp.unsqueeze(1)
        x = x + g_mlp.unsqueeze(1) * self.mlp(nx)
        return x

class FinalLayer(nn.Module):
    def __init__(self, hidden_size, output_dim):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, output_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = self.norm_final(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return self.linear(x)

class DiT(nn.Module):
    def __init__(self, latent_channels, patch_size, token_dim, num_heads, num_layers,
                 mlp_ratio=4.0, dropout=0.1, time_emb_dim=256, cond_dim=None,
                 patchifier_type='conv', unpatchifier_type='conv',
                 pos_embed_type='sinusoidal', latent_h=None, latent_w=None):
        super().__init__()
        self.latent_channels = latent_channels
        self.patch_size = patch_size
        self.token_dim = token_dim
        self.pos_embed_type = pos_embed_type

        self.patchifier = Patchifier(latent_channels, patch_size, token_dim, patchifier_type)
        self.unpatchifier = Unpatchifier(latent_channels, patch_size, token_dim, unpatchifier_type)
        self.time_embedder = TimestepEmbedder(token_dim, time_emb_dim, cond_dim=cond_dim)

        self.learned_pos = None
        if pos_embed_type == 'learned':
            if latent_h is None or latent_w is None:
                raise ValueError("Learned pos embed requires latent_h / latent_w")
            Hp = latent_h // patch_size
            Wp = latent_w // patch_size
            self.learned_pos = nn.Parameter(torch.randn(1, Hp * Wp, token_dim) * 0.02)

        if pos_embed_type == 'rope':
            self.blocks = nn.ModuleList([
                DiTBlockRoPE(token_dim, num_heads, mlp_ratio, dropout)
                for _ in range(num_layers)
            ])
        else:
            self.blocks = nn.ModuleList([
                DiTBlock(token_dim, num_heads, mlp_ratio, dropout)
                for _ in range(num_layers)
            ])

        self.final_layer = FinalLayer(token_dim, token_dim)

    def forward(self, z, t, cond=None):
        x, Hp, Wp = self.patchifier(z)     # (B, N, token_dim)
        N = Hp * Wp

        if self.pos_embed_type == 'sinusoidal':
            x = x + get_2d_sincos_pos_embed(self.token_dim, Hp, Wp).to(x.device)
        elif self.pos_embed_type == 'learned':
            x = x + self.learned_pos

        pos_h = pos_w = None
        if self.pos_embed_type == 'rope':
            pos_h = torch.arange(Hp, device=x.device).repeat_interleave(Wp).float()
            pos_w = torch.arange(Wp, device=x.device).repeat(Hp).float()

        c = self.time_embedder(t, cond=cond)
        for block in self.blocks:
            x = block(x, c, pos_h, pos_w)
        x = self.final_layer(x, c)
        return self.unpatchifier(x, Hp, Wp)

class GRUModel(nn.Module):
    def __init__(self, latent_channels, patch_size, token_dim, time_emb_dim,
                 gru_hidden_dim, gru_num_layers, dropout=0.1, cond_dim=None,
                 patchifier_type='conv', unpatchifier_type='conv',
                 pos_embed_type='sinusoidal', latent_h=None, latent_w=None):
        super().__init__()
        self.latent_channels = latent_channels
        self.patch_size = patch_size
        self.token_dim = token_dim
        self.pos_embed_type = pos_embed_type

        self.patchifier = Patchifier(latent_channels, patch_size, token_dim, patchifier_type)
        self.unpatchifier = Unpatchifier(latent_channels, patch_size, token_dim, unpatchifier_type)
        self.time_embedder = TimestepEmbedder(token_dim, time_emb_dim, cond_dim=cond_dim)

        self.learned_pos = None
        if pos_embed_type == 'learned':
            if latent_h is None or latent_w is None:
                raise ValueError("Learned pos embed requires latent_h / latent_w")
            Hp = latent_h // patch_size
            Wp = latent_w // patch_size
            self.learned_pos = nn.Parameter(torch.randn(1, Hp * Wp, token_dim) * 0.02)

        self.gru = nn.GRU(token_dim, gru_hidden_dim, gru_num_layers,
                          batch_first=True,
                          dropout=dropout if gru_num_layers > 1 else 0)
        self.out_proj = nn.Linear(gru_hidden_dim, token_dim)
        self.final_layer = FinalLayer(token_dim, token_dim)

    def forward(self, z, t, cond=None):
        x, Hp, Wp = self.patchifier(z)

        if self.pos_embed_type == 'sinusoidal':
            x = x + get_2d_sincos_pos_embed(self.token_dim, Hp, Wp).to(x.device)
        elif self.pos_embed_type == 'learned':
            x = x + self.learned_pos
        # 'rope' has no effect for GRU - silently ignore

        c = self.time_embedder(t, cond=cond)
        x, _ = self.gru(x)
        x = self.out_proj(x)
        x = self.final_layer(x, c)
        return self.unpatchifier(x, Hp, Wp)

# ==================== Rectified Flow Model Wrapper ====================

class RectifiedFlowModel:
    def __init__(self, model_type, latent_channels, latent_h, latent_w, patch_size,
                 token_dim, num_heads, num_layers, mlp_ratio, dropout, time_emb_dim,
                 gru_hidden_dim, gru_num_layers, cond_dim, device,
                 patchifier_type='conv', unpatchifier_type='conv', pos_embed_type='sinusoidal'):
        self.device = device
        self.latent_h, self.latent_w = latent_h, latent_w
        self.latent_channels = latent_channels
        self.patch_size = patch_size
        self.model_type = model_type

        if model_type == 'transformer':
            self.model = DiT(
                latent_channels=latent_channels,
                patch_size=patch_size,
                token_dim=token_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                time_emb_dim=time_emb_dim,
                cond_dim=cond_dim if cond_dim > 0 else None,
                patchifier_type=patchifier_type,
                unpatchifier_type=unpatchifier_type,
                pos_embed_type=pos_embed_type,
                latent_h=latent_h,
                latent_w=latent_w,
            ).to(device)
        else:
            self.model = GRUModel(
                latent_channels=latent_channels,
                patch_size=patch_size,
                token_dim=token_dim,
                time_emb_dim=time_emb_dim,
                gru_hidden_dim=gru_hidden_dim,
                gru_num_layers=gru_num_layers,
                dropout=dropout,
                cond_dim=cond_dim if cond_dim > 0 else None,
                patchifier_type=patchifier_type,
                unpatchifier_type=unpatchifier_type,
                pos_embed_type=pos_embed_type,
                latent_h=latent_h,
                latent_w=latent_w,
            ).to(device)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=2e-4)

    def train_step(self, z0, z1, cond=None, cfg_dropout_prob=0.0):
        B = z0.size(0)
        z0 = z0.to(self.device)
        z1 = z1.to(self.device)
        if cond is not None:
            cond = cond.to(self.device)

        t = torch.rand(B, device=self.device)
        zt = t.view(-1, 1, 1, 1) * z1 + (1 - t.view(-1, 1, 1, 1)) * z0
        target = z1 - z0

        if cond is not None and cfg_dropout_prob > 0:
            mask = torch.rand(B, 1, device=self.device) > cfg_dropout_prob
            cond_dropped = cond * mask.float()
        else:
            cond_dropped = cond

        pred = self.model(zt, t, cond=cond_dropped)
        loss = self.criterion(pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def sample(self, n_samples=16, cond=None, steps=50, method='euler', cfg_scale=1.0,
               progress_callback=None, start_from=None):
        img_shape = (self.latent_channels, self.latent_h, self.latent_w)
        z = start_from.to(self.device) if start_from is not None \
            else torch.randn(n_samples, *img_shape, device=self.device)

        if cond is not None:
            cond = cond.to(self.device)
            null_cond = torch.zeros_like(cond)
        else:
            null_cond = None

        dt = 1.0 / steps
        times = torch.linspace(0, 1, steps + 1, device=self.device)

        def v_fn(t_val, z_val):
            t_tensor = torch.full((n_samples,), t_val, device=self.device)
            if cond is not None and cfg_scale != 1.0:
                vc = self.model(z_val, t_tensor, cond=cond)
                vu = self.model(z_val, t_tensor, cond=null_cond)
                return vu + cfg_scale * (vc - vu)
            return self.model(z_val, t_tensor, cond=cond)

        for i in range(steps):
            t_cur = times[i]
            if method == 'euler':
                z = z + v_fn(t_cur, z) * dt
            elif method == 'heun':
                v1 = v_fn(t_cur, z); zp = z + v1 * dt
                v2 = v_fn(t_cur + dt, zp)
                z = z + (v1 + v2) * (dt / 2)
            elif method == 'midpoint':
                v1 = v_fn(t_cur, z)
                z = z + v_fn(t_cur + dt / 2, z + v1 * (dt / 2)) * dt
            elif method == 'rk3':
                v1 = v_fn(t_cur, z)
                v2 = v_fn(t_cur + dt, z + v1 * dt)
                v3 = v_fn(t_cur + dt / 2, z + (v1 + v2) * (dt / 2))
                z = z + (v1 + 4 * v2 + v3) * (dt / 6)
            elif method == 'rk4':
                v1 = v_fn(t_cur, z)
                v2 = v_fn(t_cur + dt / 2, z + v1 * dt / 2)
                v3 = v_fn(t_cur + dt / 2, z + v2 * dt / 2)
                v4 = v_fn(t_cur + dt, z + v3 * dt)
                z = z + (v1 + 2 * v2 + 2 * v3 + v4) * (dt / 6)
            else:
                raise ValueError(f"Unknown method {method}")

            if progress_callback:
                progress_callback(i + 1, z)
        return z

    @torch.no_grad()
    def sample_step_by_step(self, n_samples=16, cond=None, steps=50, method='euler',
                            cfg_scale=1.0, start_from=None):
        img_shape = (self.latent_channels, self.latent_h, self.latent_w)
        z = start_from.to(self.device) if start_from is not None \
            else torch.randn(n_samples, *img_shape, device=self.device)

        if cond is not None:
            cond = cond.to(self.device)
            null_cond = torch.zeros_like(cond)
        else:
            null_cond = None

        dt = 1.0 / steps
        times = torch.linspace(0, 1, steps + 1, device=self.device)

        def v_fn(t_val, z_val):
            t_tensor = torch.full((n_samples,), t_val, device=self.device)
            if cond is not None and cfg_scale != 1.0:
                vc = self.model(z_val, t_tensor, cond=cond)
                vu = self.model(z_val, t_tensor, cond=null_cond)
                return vu + cfg_scale * (vc - vu)
            return self.model(z_val, t_tensor, cond=cond)

        for i in range(steps):
            t_cur = times[i]
            if method == 'euler':
                z = z + v_fn(t_cur, z) * dt
            elif method == 'heun':
                v1 = v_fn(t_cur, z); zp = z + v1 * dt
                v2 = v_fn(t_cur + dt, zp)
                z = z + (v1 + v2) * (dt / 2)
            elif method == 'midpoint':
                v1 = v_fn(t_cur, z)
                z = z + v_fn(t_cur + dt / 2, z + v1 * (dt / 2)) * dt
            elif method == 'rk3':
                v1 = v_fn(t_cur, z)
                v2 = v_fn(t_cur + dt, z + v1 * dt)
                v3 = v_fn(t_cur + dt / 2, z + (v1 + v2) * (dt / 2))
                z = z + (v1 + 4 * v2 + v3) * (dt / 6)
            elif method == 'rk4':
                v1 = v_fn(t_cur, z)
                v2 = v_fn(t_cur + dt / 2, z + v1 * dt / 2)
                v3 = v_fn(t_cur + dt / 2, z + v2 * dt / 2)
                v4 = v_fn(t_cur + dt, z + v3 * dt)
                z = z + (v1 + 2 * v2 + 2 * v3 + v4) * (dt / 6)
            else:
                raise ValueError(f"Unknown method {method}")
            yield i + 1, z.clone()

# ==================== Extended Augmentations ====================

class RandomJPEG:
    def __init__(self, quality_low=50, quality_high=95, p=0.5):
        self.low, self.high, self.p = quality_low, quality_high, p
    def __call__(self, img):
        if random.random() > self.p:
            return img
        q = random.randint(self.low, self.high)
        pil = img if isinstance(img, Image.Image) else transforms.ToPILImage()(img)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        out = Image.open(buf).convert('RGB')
        return transforms.ToTensor()(out) if isinstance(img, torch.Tensor) else out

class ElasticTransform:
    def __init__(self, alpha=30, sigma=3, p=0.5):
        self.alpha, self.sigma, self.p = alpha, sigma, p
    def __call__(self, img):
        if random.random() > self.p:
            return img
        pil = transforms.ToPILImage()(img) if isinstance(img, torch.Tensor) else img
        w, h = pil.size
        dx = torch.randn(1, h, w) * self.sigma
        dy = torch.randn(1, h, w) * self.sigma
        k = torch.ones(1, 1, 5, 5) / 25
        dx = F_nn.conv2d(dx.view(1, 1, h, w), k, padding=2).view(h, w) * self.alpha
        dy = F_nn.conv2d(dy.view(1, 1, h, w), k, padding=2).view(h, w) * self.alpha
        x, y = torch.meshgrid(torch.arange(w), torch.arange(h), indexing='xy')
        x = (x.float() + dx) / (w - 1) * 2 - 1
        y = (y.float() + dy) / (h - 1) * 2 - 1
        grid = torch.stack([x, y], dim=-1).unsqueeze(0)
        t = transforms.ToTensor()(pil).unsqueeze(0)
        out = F_nn.grid_sample(t, grid, mode='bilinear', padding_mode='border')
        return transforms.ToPILImage()(out.squeeze(0))

# ==================== Dataset ====================

class ConditionalImageDataset(Dataset):
    def __init__(self, image_paths, labels_per_image, img_size=32, color_mode='rgb',
                 aug_settings=None, text_max_len=128):
        self.image_paths = image_paths
        self.labels = []
        for lbl in labels_per_image:
            if isinstance(lbl, str): self.labels.append([lbl])
            elif isinstance(lbl, list): self.labels.append(lbl)
            else: self.labels.append([''])
        self.img_size = img_size
        self.color_mode = color_mode.lower()
        self.aug_settings = aug_settings or {}
        self.text_max_len = text_max_len

        if color_mode == 'rgb':
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

        a = self.aug_settings
        self.aug_pipeline = []
        if a.get('flip_horizontal', False):
            self.aug_pipeline.append(transforms.RandomHorizontalFlip(p=0.5))
        if a.get('rotation', False):
            self.aug_pipeline.append(transforms.RandomRotation(
                30, interpolation=Image.BICUBIC, expand=False, fill=0))
        if a.get('random_crop', False):
            sc = a.get('crop_scale', 0.8)
            self.aug_pipeline.append(transforms.RandomResizedCrop(
                size=img_size, scale=(sc, 1.0), interpolation=Image.BICUBIC))
        if a.get('color_jitter', False):
            self.aug_pipeline.append(transforms.ColorJitter(
                brightness=a.get('brightness', 0.2), contrast=a.get('contrast', 0.2),
                saturation=a.get('saturation', 0.2), hue=a.get('hue', 0.1)))
        if a.get('random_perspective', False):
            self.aug_pipeline.append(transforms.RandomPerspective(
                distortion_scale=a.get('perspective_distortion', 0.1),
                p=0.5, interpolation=Image.BICUBIC, fill=0))
        if a.get('elastic_transform', False):
            self.aug_pipeline.append(ElasticTransform(
                alpha=a.get('elastic_alpha', 30), sigma=a.get('elastic_sigma', 3), p=0.5))
        if a.get('jpeg_compression', False):
            self.aug_pipeline.append(RandomJPEG(
                quality_low=a.get('jpeg_quality_low', 50),
                quality_high=a.get('jpeg_quality_high', 95), p=0.5))
        if a.get('stretch_vertical', False):
            self.aug_pipeline.append(transforms.RandomResizedCrop(
                size=img_size, scale=(0.8, 1.0), ratio=(0.5, 1.0),
                interpolation=Image.BICUBIC))
        if a.get('stretch_horizontal', False):
            self.aug_pipeline.append(transforms.RandomResizedCrop(
                size=img_size, scale=(0.8, 1.0), ratio=(1.0, 2.0),
                interpolation=Image.BICUBIC))

    def __len__(self):
        return len(self.image_paths)

    def apply_aug(self, pil):
        img = pil.copy()
        for a in self.aug_pipeline:
            img = a(img)
        return img

    def __getitem__(self, idx):
        try:
            pil = (load_image_as_rgb(self.image_paths[idx])
                   if self.color_mode == 'rgb'
                   else load_image_as_grayscale(self.image_paths[idx]))
            pil = self.apply_aug(pil)
            t = self.base_transform(pil)
            caps = self.labels[idx]
            txt = random.choice(caps).replace('_', ' ')
            ti = text_to_indices(txt, self.text_max_len)
            return t, torch.tensor(ti, dtype=torch.long)
        except Exception as e:
            print(f"Error {self.image_paths[idx]}: {e}")
            ch = 3 if self.color_mode == 'rgb' else 1
            return (torch.zeros(ch, self.img_size, self.img_size),
                    torch.zeros(self.text_max_len, dtype=torch.long))

# ==================== Presets ====================

TRANSFORMER_PRESETS = {
    'tiny':   {'token_dim': 128, 'num_heads': 4, 'num_layers': 4, 'mlp_ratio': 4.0},
    'small':  {'token_dim': 256, 'num_heads': 4, 'num_layers': 6, 'mlp_ratio': 4.0},
    'medium': {'token_dim': 384, 'num_heads': 6, 'num_layers': 8, 'mlp_ratio': 4.0},
    'large':  {'token_dim': 512, 'num_heads': 8, 'num_layers': 10, 'mlp_ratio': 4.0},
}

GRU_PRESETS = {
    'tiny':   {'token_dim': 128, 'gru_hidden_dim': 256, 'gru_num_layers': 2},
    'small':  {'token_dim': 256, 'gru_hidden_dim': 512, 'gru_num_layers': 2},
    'medium': {'token_dim': 384, 'gru_hidden_dim': 768, 'gru_num_layers': 3},
    'large':  {'token_dim': 512, 'gru_hidden_dim': 1024, 'gru_num_layers': 3},
}

# ==================== GUI ====================

def default_aug_dict():
    return {
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

AUG_DEFAULTS = {
    'flip_horizontal': True, 'rotation': False, 'random_crop': False, 'crop_scale': 0.8,
    'color_jitter': False, 'brightness': 0.2, 'contrast': 0.2, 'saturation': 0.2, 'hue': 0.1,
    'random_perspective': False, 'perspective_distortion': 0.1,
    'elastic_transform': False, 'elastic_alpha': 30, 'elastic_sigma': 3,
    'jpeg_compression': False, 'jpeg_quality_low': 50, 'jpeg_quality_high': 95,
    'stretch_vertical': False, 'stretch_horizontal': False,
}

class RectifiedFlowDiTApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Rectified Flow DiT / GRU")

        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.root.geometry(f"{max(1000, int(sw*0.8))}x{max(700, int(sh*0.8))}")
        self.root.minsize(900, 650)

        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
            dpi = ctypes.windll.user32.GetDpiForWindow(root.winfo_id())
            self.root.tk.call('tk', 'scaling', dpi / 72.0)
        except Exception:
            pass

        self.image_paths = []
        self.labels = []
        self.csv_path = None

        self.training_vae = False
        self.training_model = False
        self.vae_model = None
        self.rectified_model = None
        self.text_encoder = None

        self.message_queue_vae = queue.Queue()
        self.message_queue_model = queue.Queue()
        self.progressive_active = False

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_var = tk.StringVar(value=str(self.device))

        # ======= Settings groups =======
        self.global_settings = {
            'img_size': tk.IntVar(value=32),
            'color_mode': tk.StringVar(value='rgb'),
        }

        self.vae_settings = {
            'vae_size': tk.StringVar(value='big'),
            'vae_base_channels': tk.IntVar(value=32),
            'vae_latent_channels': tk.IntVar(value=8),
            'vae_latent_h': tk.IntVar(value=4),
            'vae_latent_w': tk.IntVar(value=4),
            'vae_batch_size': tk.IntVar(value=16),
            'vae_lr': tk.DoubleVar(value=1e-3),
            'vae_num_workers': tk.IntVar(value=0),
            'vae_kl_weight': tk.DoubleVar(value=0.0001),
        }

        self.dit_settings = {
            'model_type': tk.StringVar(value='transformer'),
            'patch_size': tk.IntVar(value=1),
            'patchifier_type': tk.StringVar(value='conv'),
            'unpatchifier_type': tk.StringVar(value='conv'),
            'pos_embed_type': tk.StringVar(value='sinusoidal'),
            'token_dim': tk.IntVar(value=256),

            'trans_num_heads': tk.IntVar(value=4),
            'trans_num_layers': tk.IntVar(value=6),
            'trans_mlp_ratio': tk.DoubleVar(value=4.0),

            'gru_hidden_dim': tk.IntVar(value=512),
            'gru_num_layers': tk.IntVar(value=2),

            'time_emb_dim': tk.IntVar(value=256),
            'dropout': tk.DoubleVar(value=0.1),
            'batch_size': tk.IntVar(value=16),
            'lr': tk.DoubleVar(value=2e-4),
            'cfg_dropout_prob': tk.DoubleVar(value=0.1),

            # Preview during training
            'preview_enabled': tk.BooleanVar(value=True),
            'preview_epoch_freq': tk.IntVar(value=5),
            'preview_steps': tk.IntVar(value=10),
            'preview_method': tk.StringVar(value='euler'),

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
        }

        self.vae_aug_settings = default_aug_dict()
        self.dit_aug_settings = default_aug_dict()

        self.ode_method = tk.StringVar(value='euler')
        self.ode_steps = tk.IntVar(value=50)
        self.cfg_scale = tk.DoubleVar(value=2.0)
        self.thumbnail_size = 128

        self.setup_gui()
        self.root.after(100, self.process_messages_vae)
        self.root.after(100, self.process_messages_model)

    # ---------------- GUI ----------------
    def setup_gui(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        tabs = [
            ('Dataset',               self.setup_dataset_tab),
            ('VAE Settings',          self.setup_vae_settings_tab),
            ('VAE Training',          self.setup_vae_training_tab),
            ('DiT Settings',          self.setup_dit_settings_tab),
            ('DiT Training',          self.setup_dit_training_tab),
            ('Augmentation Settings', self.setup_augmentation_tab),
            ('Generation',            self.setup_generation_tab),
        ]
        for name, builder in tabs:
            frame = ttk.Frame(self.notebook)
            self.notebook.add(frame, text=name)
            builder(frame)

        self.status_label = tk.Label(self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    # ---------------- Widget helpers ----------------
    def _spin(self, parent, label, var, low, high):
        f = tk.Frame(parent); f.pack(fill=tk.X, pady=1)
        tk.Label(f, text=label, width=22, anchor='w').pack(side=tk.LEFT)
        ttk.Spinbox(f, from_=low, to=high, textvariable=var, width=10).pack(side=tk.RIGHT)

    def _entry(self, parent, label, var):
        f = tk.Frame(parent); f.pack(fill=tk.X, pady=1)
        tk.Label(f, text=label, width=22, anchor='w').pack(side=tk.LEFT)
        tk.Entry(f, textvariable=var, width=12).pack(side=tk.RIGHT)

    def _combo(self, parent, label, var, values, width=12):
        f = tk.Frame(parent); f.pack(fill=tk.X, pady=1)
        tk.Label(f, text=label, width=22, anchor='w').pack(side=tk.LEFT)
        ttk.Combobox(f, textvariable=var, values=values, state='readonly',
                     width=width).pack(side=tk.RIGHT)

    def _make_scrollable(self, parent):
        canvas = tk.Canvas(parent)
        sb = tk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        inner = tk.Frame(canvas)
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        return inner

    # ---------------- Dataset tab ----------------
    def setup_dataset_tab(self, tab):
        main = tk.Frame(tab); main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        left = tk.Frame(main, width=300); left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left.pack_propagate(False)

        tk.Label(left, text="Dataset Management", font=("Arial", 12, "bold")).pack(pady=(0, 10))

        imgf = tk.LabelFrame(left, text="Training Images", padx=5, pady=5)
        imgf.pack(fill=tk.X, pady=(0, 5))
        tk.Button(imgf, text="Add Images", command=self.add_images, width=22).pack(pady=1)
        tk.Button(imgf, text="Add Folder (recursive)", command=self.add_folder, width=22).pack(pady=1)
        tk.Button(imgf, text="Clear All", command=self.clear_images, width=22).pack(pady=1)
        self.image_listbox = tk.Listbox(imgf, height=12)
        self.image_listbox.pack(fill=tk.X, pady=2)

        cf = tk.LabelFrame(left, text="Labels / Captions", padx=5, pady=5)
        cf.pack(fill=tk.X, pady=5)
        tk.Button(cf, text="Load CSV (image,label)", command=self.load_csv, width=22).pack(pady=1)
        tk.Button(cf, text="Use filenames as labels",
                  command=self.use_filenames_as_labels, width=22).pack(pady=1)
        tk.Button(cf, text="Use folder names as labels",
                  command=self.use_folders_as_labels, width=22).pack(pady=1)
        self.csv_status = tk.Label(cf, text="No CSV loaded", fg="red")
        self.csv_status.pack()

        right = tk.Frame(main); right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        lf = tk.LabelFrame(right, text="Log", padx=5, pady=5); lf.pack(fill=tk.BOTH, expand=True)
        self.dataset_log_text = tk.Text(lf, height=20, font=("Courier", 9))
        self.dataset_log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = tk.Scrollbar(lf, command=self.dataset_log_text.yview); sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.dataset_log_text.config(yscrollcommand=sb.set)

    # ---------------- VAE Settings tab ----------------
    def setup_vae_settings_tab(self, tab):
        inner = self._make_scrollable(tab)
        wrap = tk.Frame(inner); wrap.pack(fill=tk.X, padx=20, pady=20)

        tk.Label(wrap, text="VAE Settings", font=("Arial", 14, "bold")).pack(pady=(0, 15))

        mf = tk.LabelFrame(wrap, text="VAE Model", padx=10, pady=10); mf.pack(fill=tk.X, pady=5)
        self._combo(mf, "VAE size:", self.vae_settings['vae_size'],
                    ['tiny', 'small', 'medium', 'big', 'large'])
        self._spin(mf, "Base channels:", self.vae_settings['vae_base_channels'], 4, 256)
        self._spin(mf, "Latent channels:", self.vae_settings['vae_latent_channels'], 1, 64)
        self._spin(mf, "Latent height:", self.vae_settings['vae_latent_h'], 1, 64)
        self._spin(mf, "Latent width:", self.vae_settings['vae_latent_w'], 1, 64)

        tf = tk.LabelFrame(wrap, text="VAE Training", padx=10, pady=10); tf.pack(fill=tk.X, pady=5)
        self._spin(tf, "Batch size:", self.vae_settings['vae_batch_size'], 1, 128)
        self._entry(tf, "Learning rate:", self.vae_settings['vae_lr'])
        self._entry(tf, "KL weight:", self.vae_settings['vae_kl_weight'])
        self._spin(tf, "DataLoader workers:", self.vae_settings['vae_num_workers'], 0, 8)

        gf = tk.LabelFrame(wrap, text="Global Image Settings", padx=10, pady=10)
        gf.pack(fill=tk.X, pady=5)
        self._combo(gf, "Color mode:", self.global_settings['color_mode'], ['rgb', 'grayscale'])
        self._spin(gf, "Image size:", self.global_settings['img_size'], 16, 128)

        hw = tk.LabelFrame(wrap, text="Hardware", padx=10, pady=10); hw.pack(fill=tk.X, pady=5)
        f = tk.Frame(hw); f.pack(fill=tk.X, pady=1)
        tk.Label(f, text="Compute device:", width=22, anchor='w').pack(side=tk.LEFT)
        om = ttk.Combobox(f, textvariable=self.device_var, values=['cuda', 'cpu'],
                          state='readonly', width=10)
        om.pack(side=tk.RIGHT)
        om.bind('<<ComboboxSelected>>', lambda e: self.set_device())

    # ---------------- VAE Training tab ----------------
    def setup_vae_training_tab(self, tab):
        main = tk.Frame(tab); main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        left = tk.Frame(main, width=300); left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left.pack_propagate(False)

        tk.Label(left, text="VAE Training", font=("Arial", 12, "bold")).pack(pady=(0, 10))

        ep = tk.Frame(left); ep.pack(pady=5)
        tk.Label(ep, text="Epochs:").pack(side=tk.LEFT)
        self.vae_epoch_var = tk.StringVar(value="100")
        tk.Entry(ep, textvariable=self.vae_epoch_var, width=8).pack(side=tk.LEFT, padx=5)

        tk.Button(left, text="Initialize VAE", command=self.initialize_vae, width=24).pack(pady=2)
        tk.Button(left, text="Start VAE Training", command=self.start_vae_training,
                  width=24, bg="lightgreen").pack(pady=2)
        tk.Button(left, text="Stop VAE Training", command=self.stop_vae_training,
                  width=24, bg="salmon").pack(pady=2)
        tk.Button(left, text="Save VAE", command=self.save_vae, width=24).pack(pady=2)
        tk.Button(left, text="Load VAE", command=self.load_vae, width=24).pack(pady=2)
        tk.Button(left, text="Show Preview Now",
                  command=self.show_vae_preview, width=24).pack(pady=8)

        right = tk.Frame(main); right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        pf = tk.LabelFrame(right, text="Top: Augmented Input | Bottom: Reconstruction",
                           padx=5, pady=5)
        pf.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        self.vae_preview_canvas = tk.Canvas(pf, bg='gray', width=256, height=128)
        self.vae_preview_canvas.pack()

        lf = tk.LabelFrame(right, text="Log", padx=5, pady=5); lf.pack(fill=tk.BOTH, expand=True)
        self.vae_log_text = tk.Text(lf, height=15, font=("Courier", 9))
        self.vae_log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = tk.Scrollbar(lf, command=self.vae_log_text.yview); sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.vae_log_text.config(yscrollcommand=sb.set)

    # ---------------- DiT Settings tab ----------------
    def setup_dit_settings_tab(self, tab):
        inner = self._make_scrollable(tab)
        wrap = tk.Frame(inner); wrap.pack(fill=tk.X, padx=20, pady=20)

        tk.Label(wrap, text="DiT Settings", font=("Arial", 14, "bold")).pack(pady=(0, 15))

        # Architecture
        arch = tk.LabelFrame(wrap, text="Model Architecture", padx=10, pady=10)
        arch.pack(fill=tk.X, pady=5)
        self._combo(arch, "Model type:", self.dit_settings['model_type'],
                    ['transformer', 'gru'])
        self._spin(arch, "Patch size:", self.dit_settings['patch_size'], 1, 8)
        self._combo(arch, "Patchifier type:", self.dit_settings['patchifier_type'],
                    ['conv', 'linear'])
        self._combo(arch, "Unpatchifier type:", self.dit_settings['unpatchifier_type'],
                    ['conv', 'linear'])
        self._combo(arch, "Positional embed:", self.dit_settings['pos_embed_type'],
                    ['sinusoidal', 'learned', 'rope'])
        self._spin(arch, "Token dim:", self.dit_settings['token_dim'], 32, 2048)

        preset_frame = tk.Frame(arch); preset_frame.pack(fill=tk.X, pady=5)
        tk.Label(preset_frame, text="Preset size:").pack(side=tk.LEFT)
        self.preset_var = tk.StringVar(value='small')
        ttk.Combobox(preset_frame, textvariable=self.preset_var,
                     values=['tiny', 'small', 'medium', 'large'],
                     state='readonly', width=8).pack(side=tk.LEFT, padx=5)
        tk.Button(preset_frame, text="Apply Preset",
                  command=self.apply_preset).pack(side=tk.LEFT, padx=5)

        # Transformer-specific
        tf = tk.LabelFrame(wrap, text="Transformer", padx=10, pady=10); tf.pack(fill=tk.X, pady=5)
        self._spin(tf, "Num heads:", self.dit_settings['trans_num_heads'], 1, 32)
        self._spin(tf, "Num layers:", self.dit_settings['trans_num_layers'], 2, 32)
        self._entry(tf, "MLP ratio:", self.dit_settings['trans_mlp_ratio'])

        # GRU-specific
        gf = tk.LabelFrame(wrap, text="GRU", padx=10, pady=10); gf.pack(fill=tk.X, pady=5)
        self._spin(gf, "GRU hidden dim:", self.dit_settings['gru_hidden_dim'], 32, 4096)
        self._spin(gf, "GRU num layers:", self.dit_settings['gru_num_layers'], 1, 8)

        # Training
        trf = tk.LabelFrame(wrap, text="Training", padx=10, pady=10); trf.pack(fill=tk.X, pady=5)
        self._spin(trf, "Time emb dim:", self.dit_settings['time_emb_dim'], 32, 1024)
        self._entry(trf, "Dropout:", self.dit_settings['dropout'])
        self._spin(trf, "Batch size:", self.dit_settings['batch_size'], 1, 128)
        self._entry(trf, "Learning rate:", self.dit_settings['lr'])
        self._entry(trf, "CFG dropout:", self.dit_settings['cfg_dropout_prob'])

        # Preview during training
        pvf = tk.LabelFrame(wrap, text="Preview During Training", padx=10, pady=10)
        pvf.pack(fill=tk.X, pady=5)
        tk.Checkbutton(pvf, text="Enable preview",
                       variable=self.dit_settings['preview_enabled']).pack(anchor='w')
        self._spin(pvf, "Every N epochs:", self.dit_settings['preview_epoch_freq'], 1, 200)
        self._spin(pvf, "Preview steps:", self.dit_settings['preview_steps'], 1, 200)
        self._combo(pvf, "ODE solver:", self.dit_settings['preview_method'],
                    ['euler', 'heun', 'midpoint', 'rk3', 'rk4'])

        # Text conditioning
        cf = tk.LabelFrame(wrap, text="Text Conditioning", padx=10, pady=10)
        cf.pack(fill=tk.X, pady=5)
        tk.Checkbutton(cf, text="Enable text conditioning",
                       variable=self.dit_settings['cond_enabled']).pack(anchor='w')
        self._combo(cf, "Encoder type:", self.dit_settings['text_encoder_type'],
                    ['BiGRU', 'BiTransformer'])
        self._combo(cf, "Encoder size:", self.dit_settings['text_encoder_size'],
                    ['tiny', 'small', 'medium', 'large'])
        tk.Button(cf, text="Apply preset to dims",
                  command=self.apply_text_encoder_preset).pack(pady=3)
        self._spin(cf, "Embed dim:", self.dit_settings['cond_embed_dim'], 8, 512)
        self._spin(cf, "Hidden size (GRU):", self.dit_settings['cond_hidden_size'], 8, 1024)
        self._spin(cf, "Num layers:", self.dit_settings['cond_num_layers'], 1, 6)
        self._spin(cf, "Num heads (TF):", self.dit_settings['cond_num_heads'], 1, 16)
        self._spin(cf, "FF dim (TF):", self.dit_settings['cond_ff_dim'], 32, 2048)
        self._spin(cf, "Cond dim:", self.dit_settings['cond_dim'], 32, 1024)
        self._spin(cf, "Max text len:", self.dit_settings['cond_text_max_len'], 32, 512)

        r = tk.Frame(wrap); r.pack(fill=tk.X, pady=10)
        tk.Button(r, text="Reset VAE + DiT Settings to Defaults",
                  command=self.reset_all_settings,
                  bg="orange", fg="white", font=("Arial", 10, "bold")).pack(pady=5)

    # ---------------- DiT Training tab ----------------
    def setup_dit_training_tab(self, tab):
        main = tk.Frame(tab); main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        left = tk.Frame(main, width=300); left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left.pack_propagate(False)

        tk.Label(left, text="DiT / GRU Training", font=("Arial", 12, "bold")).pack(pady=(0, 10))

        ep = tk.Frame(left); ep.pack(pady=5)
        tk.Label(ep, text="Epochs:").pack(side=tk.LEFT)
        self.model_epoch_var = tk.StringVar(value="200")
        tk.Entry(ep, textvariable=self.model_epoch_var, width=8).pack(side=tk.LEFT, padx=5)

        tk.Button(left, text="Initialize Model", command=self.initialize_model,
                  width=24).pack(pady=2)
        tk.Button(left, text="Start Training", command=self.start_model_training,
                  width=24, bg="lightgreen").pack(pady=2)
        tk.Button(left, text="Stop Training", command=self.stop_model_training,
                  width=24, bg="salmon").pack(pady=2)
        tk.Button(left, text="Save Model", command=self.save_model, width=24).pack(pady=2)
        tk.Button(left, text="Load Model", command=self.load_model, width=24).pack(pady=2)

        right = tk.Frame(main); right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        pf = tk.LabelFrame(right, text="Generated Samples (decoded)", padx=5, pady=5)
        pf.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        self.model_preview_canvas = tk.Canvas(pf, bg='gray', width=256, height=256)
        self.model_preview_canvas.pack()

        prf = tk.LabelFrame(right, text="Test prompt (empty = unconditional)", padx=5, pady=5)
        prf.pack(fill=tk.X, pady=(0, 5))
        self.test_prompt_entry = tk.Entry(prf)
        self.test_prompt_entry.insert(0, "a cute cat")
        self.test_prompt_entry.pack(fill=tk.X, pady=2)
        tk.Button(prf, text="Generate Preview",
                  command=self.model_preview_with_prompt).pack(pady=2)

        lf = tk.LabelFrame(right, text="Log", padx=5, pady=5); lf.pack(fill=tk.BOTH, expand=True)
        self.model_log_text = tk.Text(lf, height=15, font=("Courier", 9))
        self.model_log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = tk.Scrollbar(lf, command=self.model_log_text.yview); sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.model_log_text.config(yscrollcommand=sb.set)

    # ---------------- Augmentation Settings tab ----------------
    def setup_augmentation_tab(self, tab):
        tk.Label(tab, text="Augmentation Settings",
                 font=("Arial", 14, "bold")).pack(pady=(15, 10))

        cols = tk.Frame(tab); cols.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        vae_col = tk.LabelFrame(cols, text="VAE Augmentations", padx=10, pady=10)
        vae_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        self._build_aug_controls(vae_col, self.vae_aug_settings)

        dit_col = tk.LabelFrame(cols, text="DiT Augmentations", padx=10, pady=10)
        dit_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))
        self._build_aug_controls(dit_col, self.dit_aug_settings)

        r = tk.Frame(tab); r.pack(fill=tk.X, pady=10)
        tk.Button(r, text="Reset VAE Augs", command=lambda: self._reset_aug(self.vae_aug_settings),
                  bg="orange").pack(side=tk.LEFT, padx=10, expand=True)
        tk.Button(r, text="Reset DiT Augs", command=lambda: self._reset_aug(self.dit_aug_settings),
                  bg="orange").pack(side=tk.RIGHT, padx=10, expand=True)

    def _build_aug_controls(self, parent, aug):
        def row():
            f = tk.Frame(parent); f.pack(fill=tk.X, pady=2); return f

        r = row()
        tk.Checkbutton(r, text="Horizontal Flip",
                       variable=aug['flip_horizontal']).pack(anchor='w')
        r = row()
        tk.Checkbutton(r, text="Rotation (±30°)",
                       variable=aug['rotation']).pack(anchor='w')
        r = row()
        tk.Checkbutton(r, text="Random Crop",
                       variable=aug['random_crop']).pack(side=tk.LEFT)
        tk.Label(r, text="scale:").pack(side=tk.LEFT, padx=(10, 2))
        tk.Entry(r, textvariable=aug['crop_scale'], width=6).pack(side=tk.LEFT)
        r = row()
        tk.Checkbutton(r, text="Color Jitter",
                       variable=aug['color_jitter']).pack(anchor='w')
        r = row()
        tk.Label(r, text="  bri / con / sat / hue:").pack(side=tk.LEFT)
        for k in ['brightness', 'contrast', 'saturation', 'hue']:
            tk.Entry(r, textvariable=aug[k], width=5).pack(side=tk.LEFT, padx=1)
        r = row()
        tk.Checkbutton(r, text="Random Perspective",
                       variable=aug['random_perspective']).pack(side=tk.LEFT)
        tk.Label(r, text="distortion:").pack(side=tk.LEFT, padx=(8, 2))
        tk.Entry(r, textvariable=aug['perspective_distortion'], width=6).pack(side=tk.LEFT)
        r = row()
        tk.Checkbutton(r, text="Elastic Transform",
                       variable=aug['elastic_transform']).pack(side=tk.LEFT)
        tk.Label(r, text="alpha:").pack(side=tk.LEFT, padx=(8, 2))
        tk.Entry(r, textvariable=aug['elastic_alpha'], width=4).pack(side=tk.LEFT)
        tk.Label(r, text="sigma:").pack(side=tk.LEFT, padx=(4, 2))
        tk.Entry(r, textvariable=aug['elastic_sigma'], width=4).pack(side=tk.LEFT)
        r = row()
        tk.Checkbutton(r, text="JPEG Compression",
                       variable=aug['jpeg_compression']).pack(side=tk.LEFT)
        tk.Label(r, text="quality:").pack(side=tk.LEFT, padx=(8, 2))
        tk.Entry(r, textvariable=aug['jpeg_quality_low'], width=5).pack(side=tk.LEFT)
        tk.Entry(r, textvariable=aug['jpeg_quality_high'], width=5).pack(side=tk.LEFT)
        r = row()
        tk.Checkbutton(r, text="Stretch Vertical",
                       variable=aug['stretch_vertical']).pack(side=tk.LEFT, padx=(0, 10))
        tk.Checkbutton(r, text="Stretch Horizontal",
                       variable=aug['stretch_horizontal']).pack(side=tk.LEFT)

    def _reset_aug(self, aug):
        for k, v in AUG_DEFAULTS.items():
            aug[k].set(v)

    # ---------------- Generation tab ----------------
    def setup_generation_tab(self, tab):
        main = tk.Frame(tab); main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        tk.Label(main, text="Generate Images (Flow → VAE Decode)",
                 font=("Arial", 14, "bold")).pack(pady=(0, 10))

        pf = tk.LabelFrame(main, text="Text Prompt (empty = unconditional)", padx=5, pady=5)
        pf.pack(fill=tk.X, pady=(0, 10))
        self.gen_prompt = tk.Entry(pf, width=60)
        self.gen_prompt.insert(0, "a cute cat")
        self.gen_prompt.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        cf = tk.Frame(main); cf.pack(pady=5)
        tk.Label(cf, text="Number:").pack(side=tk.LEFT)
        self.gen_count = tk.IntVar(value=16)
        tk.Spinbox(cf, from_=1, to=64, textvariable=self.gen_count, width=5).pack(side=tk.LEFT, padx=5)
        tk.Label(cf, text="ODE Steps:").pack(side=tk.LEFT, padx=(10, 0))
        tk.Spinbox(cf, from_=1, to=200, textvariable=self.ode_steps, width=5).pack(side=tk.LEFT, padx=5)
        tk.Label(cf, text="Method:").pack(side=tk.LEFT, padx=(10, 0))
        ttk.Combobox(cf, textvariable=self.ode_method,
                     values=['euler', 'heun', 'midpoint', 'rk3', 'rk4'],
                     state='readonly', width=10).pack(side=tk.LEFT, padx=5)
        tk.Label(cf, text="CFG scale:").pack(side=tk.LEFT, padx=(10, 0))
        tk.Entry(cf, width=6, textvariable=self.cfg_scale).pack(side=tk.LEFT, padx=5)
        self.progressive_grid = tk.BooleanVar(value=False)
        tk.Checkbutton(cf, text="Progressive", variable=self.progressive_grid).pack(side=tk.LEFT, padx=10)
        tk.Label(cf, text="Interval:").pack(side=tk.LEFT)
        self.prog_interval = tk.IntVar(value=10)
        tk.Spinbox(cf, from_=1, to=50, textvariable=self.prog_interval, width=5).pack(side=tk.LEFT, padx=5)
        self.generate_btn = tk.Button(cf, text="Generate",
                                       command=self.generate_samples, bg="lightgreen")
        self.generate_btn.pack(side=tk.LEFT, padx=5)
        self.stop_prog_btn = tk.Button(cf, text="Stop", command=self.stop_progressive,
                                       state=tk.DISABLED, bg="salmon")
        self.stop_prog_btn.pack(side=tk.LEFT, padx=5)

        rf = tk.LabelFrame(main, text="Results", padx=5, pady=5); rf.pack(fill=tk.BOTH, expand=True, pady=5)
        cfr = tk.Frame(rf); cfr.pack(fill=tk.BOTH, expand=True)
        self.gen_canvas = tk.Canvas(cfr, bg='lightgray')
        vs = tk.Scrollbar(cfr, orient=tk.VERTICAL, command=self.gen_canvas.yview)
        hs = tk.Scrollbar(cfr, orient=tk.HORIZONTAL, command=self.gen_canvas.xview)
        self.gen_canvas.configure(yscrollcommand=vs.set, xscrollcommand=hs.set)
        vs.pack(side=tk.RIGHT, fill=tk.Y); hs.pack(side=tk.BOTTOM, fill=tk.X)
        self.gen_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.inner_frame = tk.Frame(self.gen_canvas)
        self.gen_canvas.create_window((0, 0), window=self.inner_frame, anchor='nw')
        self.inner_frame.bind("<Configure>",
                              lambda e: self.gen_canvas.configure(scrollregion=self.gen_canvas.bbox('all')))
        self.gen_info = tk.Label(main, text="", fg="blue"); self.gen_info.pack()

    # ---------------- Core ----------------
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
                self.status_label.config(text=msg[:80])
        except queue.Empty:
            pass
        self.root.after(100, self.process_messages_vae)

    def process_messages_model(self):
        try:
            while True:
                msg = self.message_queue_model.get_nowait()
                self.model_log_text.insert(tk.END, f"{time.strftime('%H:%M:%S')} - {msg}\n")
                self.model_log_text.see(tk.END)
                self.status_label.config(text=msg[:80])
        except queue.Empty:
            pass
        self.root.after(100, self.process_messages_model)

    # ---------------- Dataset ops ----------------
    def add_images(self):
        files = filedialog.askopenfilenames(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.jfif *.webp *.bmp")])
        for f in files:
            if f not in self.image_paths:
                self.image_paths.append(f)
                self.image_listbox.insert(tk.END, os.path.basename(f))
        self.log_model(f"Added {len(files)} images. Total: {len(self.image_paths)}")

    def add_folder(self):
        folder = filedialog.askdirectory()
        if not folder:
            return
        n = 0
        for r, _, fs in os.walk(folder):
            for f in fs:
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.jfif', '.webp', '.bmp')):
                    p = os.path.join(r, f)
                    if p not in self.image_paths:
                        self.image_paths.append(p)
                        self.image_listbox.insert(tk.END, os.path.basename(p))
                        n += 1
        self.log_model(f"Added {n} images. Total: {len(self.image_paths)}")

    def clear_images(self):
        self.image_paths = []
        self.labels = []
        self.image_listbox.delete(0, tk.END)
        self.csv_status.config(text="No CSV loaded", fg="red")
        self.log_model("Cleared all images")

    def load_csv(self):
        fname = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if not fname:
            return
        self.csv_path = fname
        base_map = {os.path.basename(p).lower(): p for p in self.image_paths}
        label_map = {}
        try:
            with open(fname, 'r', encoding='utf-8-sig') as f:
                for row in csv.reader(f):
                    if not row or len(row) < 2:
                        continue
                    nm = row[0].strip().strip('"').strip("'")
                    lb = row[1].strip().strip('"').strip("'")
                    matched = base_map.get(os.path.basename(nm).lower())
                    if matched:
                        label_map.setdefault(matched, []).append(lb)
        except Exception as e:
            self.log_model(f"CSV error: {e}")
            return
        self.labels = []
        unknown = 0
        for p in self.image_paths:
            if p in label_map:
                self.labels.append(label_map[p])
            else:
                self.labels.append([os.path.splitext(os.path.basename(p))[0]])
                unknown += 1
        self.csv_status.config(
            text=f"CSV: {len(self.image_paths) - unknown} matched, {unknown} fallback", fg="green")
        self.log_model(f"CSV loaded. {len(self.image_paths) - unknown} matched.")

    def use_filenames_as_labels(self):
        self.labels = [[os.path.splitext(os.path.basename(p))[0]] for p in self.image_paths]
        self.csv_status.config(text="Using filenames as labels", fg="blue")
        self.log_model("Using filenames as labels.")

    def use_folders_as_labels(self):
        self.labels = [[os.path.basename(os.path.dirname(p)) or 'unknown']
                       for p in self.image_paths]
        self.csv_status.config(text="Using folder names as labels", fg="blue")
        self.log_model("Using folder names as labels.")

    def apply_preset(self):
        model_type = self.dit_settings['model_type'].get()
        preset = self.preset_var.get()
        if model_type == 'transformer' and preset in TRANSFORMER_PRESETS:
            cfg = TRANSFORMER_PRESETS[preset]
            self.dit_settings['token_dim'].set(cfg['token_dim'])
            self.dit_settings['trans_num_heads'].set(cfg['num_heads'])
            self.dit_settings['trans_num_layers'].set(cfg['num_layers'])
            self.dit_settings['trans_mlp_ratio'].set(cfg['mlp_ratio'])
            self.log_model(f"Applied Transformer preset '{preset}'")
        elif model_type == 'gru' and preset in GRU_PRESETS:
            cfg = GRU_PRESETS[preset]
            self.dit_settings['token_dim'].set(cfg['token_dim'])
            self.dit_settings['gru_hidden_dim'].set(cfg['gru_hidden_dim'])
            self.dit_settings['gru_num_layers'].set(cfg['gru_num_layers'])
            self.log_model(f"Applied GRU preset '{preset}'")
        else:
            self.log_model(f"Unknown preset '{preset}'")

    def apply_text_encoder_preset(self):
        enc = self.dit_settings['text_encoder_type'].get()
        size = self.dit_settings['text_encoder_size'].get()
        cfg = get_encoder_config(enc, size)
        self.dit_settings['cond_embed_dim'].set(
            cfg.get('embed_dim', self.dit_settings['cond_embed_dim'].get()))
        if enc == 'BiGRU':
            self.dit_settings['cond_hidden_size'].set(
                cfg.get('hidden_size', self.dit_settings['cond_hidden_size'].get()))
            self.dit_settings['cond_num_layers'].set(
                cfg.get('num_layers', self.dit_settings['cond_num_layers'].get()))
        else:
            self.dit_settings['cond_num_heads'].set(
                cfg.get('num_heads', self.dit_settings['cond_num_heads'].get()))
            self.dit_settings['cond_num_layers'].set(
                cfg.get('num_layers', self.dit_settings['cond_num_layers'].get()))
            self.dit_settings['cond_ff_dim'].set(
                cfg.get('ff_dim', self.dit_settings['cond_ff_dim'].get()))
        self.dit_settings['cond_dim'].set(
            cfg.get('cond_dim', self.dit_settings['cond_dim'].get()))
        self.log_model(f"Applied {enc} {size} preset.")

    def reset_all_settings(self):
        for k, v in {
            'vae_size': 'big', 'vae_base_channels': 32, 'vae_latent_channels': 8,
            'vae_latent_h': 4, 'vae_latent_w': 4, 'vae_batch_size': 16,
            'vae_lr': 1e-3, 'vae_num_workers': 0, 'vae_kl_weight': 1e-4,
        }.items():
            self.vae_settings[k].set(v)
        for k, v in {
            'model_type': 'transformer', 'patch_size': 1,
            'patchifier_type': 'conv', 'unpatchifier_type': 'conv',
            'pos_embed_type': 'sinusoidal', 'token_dim': 256,
            'trans_num_heads': 4, 'trans_num_layers': 6, 'trans_mlp_ratio': 4.0,
            'gru_hidden_dim': 512, 'gru_num_layers': 2,
            'time_emb_dim': 256, 'dropout': 0.1,
            'batch_size': 16, 'lr': 2e-4, 'cfg_dropout_prob': 0.1,
            'preview_enabled': True,
            'preview_epoch_freq': 5,
            'preview_steps': 10,
            'preview_method': 'euler',
            'cond_enabled': False, 'text_encoder_type': 'BiGRU',
            'text_encoder_size': 'small', 'cond_embed_dim': 64,
            'cond_hidden_size': 64, 'cond_num_layers': 2, 'cond_num_heads': 4,
            'cond_ff_dim': 256, 'cond_dim': 256, 'cond_text_max_len': 128,
        }.items():
            self.dit_settings[k].set(v)
        self.global_settings['img_size'].set(32)
        self.global_settings['color_mode'].set('rgb')
        self._reset_aug(self.vae_aug_settings)
        self._reset_aug(self.dit_aug_settings)
        self.log_model("All settings reset to defaults.")

    # ---------------- VAE ----------------
    def initialize_vae(self):
        try:
            in_ch = 3 if self.global_settings['color_mode'].get() == 'rgb' else 1
            self.vae_model = FlexVAE(
                in_channels=in_ch,
                base_channels=self.vae_settings['vae_base_channels'].get(),
                latent_channels=self.vae_settings['vae_latent_channels'].get(),
                latent_h=self.vae_settings['vae_latent_h'].get(),
                latent_w=self.vae_settings['vae_latent_w'].get(),
                size=self.vae_settings['vae_size'].get(),
            ).to(self.device)
            self.vae_optimizer = optim.Adam(self.vae_model.parameters(),
                                            lr=self.vae_settings['vae_lr'].get())
            self.log_vae(f"VAE initialized (size={self.vae_settings['vae_size'].get()}) "
                         f"on {self.device}.")
        except Exception as e:
            self.log_vae(f"Init error: {e}")

    def start_vae_training(self):
        if not self.image_paths:
            self.log_vae("No images!"); return
        if not self.vae_model:
            self.log_vae("Initialize VAE first!"); return
        if self.training_vae:
            self.log_vae("Already training."); return
        try:
            epochs = int(self.vae_epoch_var.get())
        except Exception:
            self.log_vae("Invalid epochs"); return
        self.training_vae = True
        self.vae_start_time = time.time()
        threading.Thread(target=self.train_vae_loop, args=(epochs,), daemon=True).start()
        self.log_vae(f"VAE training started for {epochs} epochs.")

    def train_vae_loop(self, epochs):
        try:
            bs = self.vae_settings['vae_batch_size'].get()
            nw = self.vae_settings['vae_num_workers'].get()
            img_size = self.global_settings['img_size'].get()
            kl_w = self.vae_settings['vae_kl_weight'].get()
            cm = self.global_settings['color_mode'].get()

            aug_dict = {k: v.get() for k, v in self.vae_aug_settings.items()}
            labels = self.labels if self.labels else [['']] * len(self.image_paths)
            ds = ConditionalImageDataset(self.image_paths, labels, img_size, cm, aug_dict,
                                         self.dit_settings['cond_text_max_len'].get())
            dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=nw,
                            pin_memory=(self.device.type == 'cuda'),
                            persistent_workers=(nw > 0))

            for ep in range(epochs):
                if not self.training_vae:
                    break
                tot, n = 0.0, 0
                for imgs, _ in dl:
                    if not self.training_vae:
                        break
                    imgs = imgs.to(self.device)
                    recon, mu, logvar = self.vae_model(imgs)
                    rl = F_nn.mse_loss(recon, imgs)
                    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / imgs.size(0)
                    loss = rl + kl_w * kl
                    self.vae_optimizer.zero_grad()
                    loss.backward()
                    self.vae_optimizer.step()
                    tot += loss.item(); n += 1
                avg = tot / max(1, n)
                el = time.time() - self.vae_start_time
                self.log_vae(f"Epoch {ep+1}/{epochs} | Loss: {avg:.6f} | Time: {el:.1f}s")
                if (ep + 1) % 5 == 0:
                    self.show_vae_preview()
            self.training_vae = False
            self.log_vae("VAE training finished.")
        except Exception as e:
            import traceback; traceback.print_exc()
            self.log_vae(f"VAE training error: {e}")
            self.training_vae = False

    def stop_vae_training(self):
        self.training_vae = False
        self.log_vae("VAE training stopped.")

    def show_vae_preview(self):
        if not self.vae_model or not self.image_paths:
            return
        try:
            n = min(16, len(self.image_paths))
            idx = random.sample(range(len(self.image_paths)), n)
            paths = [self.image_paths[i] for i in idx]
            labs = ([self.labels[i] for i in idx]
                    if (self.labels and len(self.labels) == len(self.image_paths))
                    else [['']] * n)
            cm = self.global_settings['color_mode'].get()
            aug_dict = {k: v.get() for k, v in self.vae_aug_settings.items()}
            ds = ConditionalImageDataset(paths, labs, self.global_settings['img_size'].get(),
                                          cm, aug_dict)
            dl = DataLoader(ds, batch_size=n, shuffle=False)
            imgs, _ = next(iter(dl))
            with torch.no_grad():
                recon, _, _ = self.vae_model(imgs.to(self.device))
            imgs = ((imgs + 1) / 2).clamp(0, 1).cpu().numpy()
            recon = ((recon + 1) / 2).clamp(0, 1).cpu().numpy()
            thumb = self.thumbnail_size // 2
            grid = Image.new('RGB', (8 * thumb, 4 * thumb))
            for i in range(n):
                row, col = i // 4, i % 4
                for arr, xoff in [(imgs[i], 0), (recon[i], thumb)]:
                    if arr.shape[0] == 1:
                        a = np.stack([arr[0] * 255] * 3, axis=-1).astype(np.uint8)
                    else:
                        a = (arr.transpose(1, 2, 0) * 255).astype(np.uint8)
                    pil = Image.fromarray(a).resize((thumb, thumb), Image.NEAREST)
                    grid.paste(pil, (col * thumb * 2 + xoff, row * thumb))
            grid = grid.resize((256, 128), Image.NEAREST)
            self.vae_preview_photo = ImageTk.PhotoImage(grid)
            self.vae_preview_canvas.delete("all")
            self.vae_preview_canvas.create_image(128, 64, image=self.vae_preview_photo)
        except Exception as e:
            self.log_vae(f"Preview error: {e}")

    def save_vae(self):
        if not self.vae_model:
            self.log_vae("No VAE."); return
        fname = filedialog.asksaveasfilename(defaultextension=".pth",
                                             filetypes=[("PyTorch", "*.pth")])
        if fname:
            torch.save({
                'model_state': self.vae_model.state_dict(),
                'optimizer_state': self.vae_optimizer.state_dict(),
                'vae_settings': {k: v.get() for k, v in self.vae_settings.items()},
                'global_settings': {k: v.get() for k, v in self.global_settings.items()},
            }, fname)
            self.log_vae(f"VAE saved to {fname}")

    def load_vae(self):
        fname = filedialog.askopenfilename(filetypes=[("PyTorch", "*.pth")])
        if not fname:
            return
        try:
            ckpt = torch.load(fname, map_location='cpu')
            for k, v in ckpt.get('vae_settings', {}).items():
                if k in self.vae_settings:
                    self.vae_settings[k].set(v)
            for k, v in ckpt.get('global_settings', {}).items():
                if k in self.global_settings:
                    self.global_settings[k].set(v)
            self.initialize_vae()
            self.vae_model.load_state_dict(ckpt['model_state'])
            self.vae_optimizer.load_state_dict(ckpt['optimizer_state'])
            self.vae_model.to(self.device)
            self.log_vae(f"VAE loaded from {fname}")
        except Exception as e:
            self.log_vae(f"Load error: {e}")

    # ---------------- DiT ----------------
    def initialize_model(self):
        if not self.vae_model:
            self.log_model("Train/load a VAE first!"); return
        try:
            model_type = self.dit_settings['model_type'].get()
            latent_channels = self.vae_settings['vae_latent_channels'].get()
            latent_h = self.vae_settings['vae_latent_h'].get()
            latent_w = self.vae_settings['vae_latent_w'].get()
            patch_size = self.dit_settings['patch_size'].get()
            token_dim = self.dit_settings['token_dim'].get()
            cond_enabled = self.dit_settings['cond_enabled'].get()
            cond_dim = self.dit_settings['cond_dim'].get() if cond_enabled else 0

            patchifier_type = self.dit_settings['patchifier_type'].get()
            unpatchifier_type = self.dit_settings['unpatchifier_type'].get()
            pos_embed_type = self.dit_settings['pos_embed_type'].get()

            if latent_h % patch_size != 0 or latent_w % patch_size != 0:
                self.log_model(f"Latent size ({latent_h}x{latent_w}) not divisible "
                               f"by patch size {patch_size}.")
                return

            if model_type == 'transformer':
                num_heads = self.dit_settings['trans_num_heads'].get()
                num_layers = self.dit_settings['trans_num_layers'].get()
                mlp_ratio = self.dit_settings['trans_mlp_ratio'].get()
                gru_hidden_dim = None
                gru_num_layers = None
                if token_dim % num_heads != 0:
                    self.log_model(f"Token dim ({token_dim}) must be divisible by "
                                   f"num heads ({num_heads}).")
                    return
                if pos_embed_type == 'rope':
                    head_dim = token_dim // num_heads
                    if head_dim % 4 != 0:
                        self.log_model(f"RoPE requires head_dim divisible by 4. "
                                       f"token_dim={token_dim} / heads={num_heads} "
                                       f"-> head_dim={head_dim}.")
                        return
            else:
                num_heads = None
                num_layers = None
                mlp_ratio = None
                gru_hidden_dim = self.dit_settings['gru_hidden_dim'].get()
                gru_num_layers = self.dit_settings['gru_num_layers'].get()

            self.rectified_model = RectifiedFlowModel(
                model_type=model_type,
                latent_channels=latent_channels,
                latent_h=latent_h,
                latent_w=latent_w,
                patch_size=patch_size,
                token_dim=token_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                mlp_ratio=mlp_ratio,
                dropout=self.dit_settings['dropout'].get(),
                time_emb_dim=self.dit_settings['time_emb_dim'].get(),
                gru_hidden_dim=gru_hidden_dim,
                gru_num_layers=gru_num_layers,
                cond_dim=cond_dim,
                device=self.device,
                patchifier_type=patchifier_type,
                unpatchifier_type=unpatchifier_type,
                pos_embed_type=pos_embed_type,
            )
            for pg in self.rectified_model.optimizer.param_groups:
                pg['lr'] = self.dit_settings['lr'].get()

            if cond_enabled:
                enc = self.dit_settings['text_encoder_type'].get()
                if enc == 'BiGRU':
                    self.text_encoder = TextEncoder(
                        vocab_size=256,
                        embed_dim=self.dit_settings['cond_embed_dim'].get(),
                        hidden_size=self.dit_settings['cond_hidden_size'].get(),
                        num_layers=self.dit_settings['cond_num_layers'].get(),
                        cond_dim=self.dit_settings['cond_dim'].get(),
                    )
                else:
                    self.text_encoder = TransformerTextEncoder(
                        vocab_size=256,
                        embed_dim=self.dit_settings['cond_embed_dim'].get(),
                        num_heads=self.dit_settings['cond_num_heads'].get(),
                        num_layers=self.dit_settings['cond_num_layers'].get(),
                        ff_dim=self.dit_settings['cond_ff_dim'].get(),
                        cond_dim=self.dit_settings['cond_dim'].get(),
                        max_len=self.dit_settings['cond_text_max_len'].get(),
                    )
                self.text_encoder.to(self.device)
                self.rectified_model.optimizer = optim.Adam(
                    list(self.rectified_model.model.parameters())
                    + list(self.text_encoder.parameters()),
                    lr=self.dit_settings['lr'].get(),
                )
                self.log_model(f"Conditional {model_type.upper()} initialized "
                               f"(token_dim={token_dim}, patchifier={patchifier_type}, "
                               f"unpatchifier={unpatchifier_type}, pos={pos_embed_type}, "
                               f"patch={patch_size}).")
            else:
                self.text_encoder = None
                self.log_model(f"Unconditional {model_type.upper()} initialized "
                               f"(token_dim={token_dim}, patchifier={patchifier_type}, "
                               f"unpatchifier={unpatchifier_type}, pos={pos_embed_type}, "
                               f"patch={patch_size}).")
        except Exception as e:
            import traceback; traceback.print_exc()
            self.log_model(f"Init error: {e}")

    def start_model_training(self):
        if not self.image_paths:
            self.log_model("No images!"); return
        if not self.rectified_model:
            self.log_model("Initialize model first!"); return
        if not self.vae_model:
            self.log_model("No VAE!"); return
        if self.training_model:
            self.log_model("Already training."); return
        if self.dit_settings['cond_enabled'].get() and not self.labels:
            self.log_model("Conditional training requires labels."); return
        try:
            epochs = int(self.model_epoch_var.get())
        except Exception:
            self.log_model("Invalid epochs"); return
        self.training_model = True
        self.model_start_time = time.time()
        threading.Thread(target=self.train_model_loop, args=(epochs,), daemon=True).start()
        self.log_model(f"Training started for {epochs} epochs.")

    def train_model_loop(self, epochs):
        try:
            bs = self.dit_settings['batch_size'].get()
            nw = self.vae_settings['vae_num_workers'].get()
            img_size = self.global_settings['img_size'].get()
            cm = self.global_settings['color_mode'].get()
            cond_enabled = self.dit_settings['cond_enabled'].get()
            text_max = self.dit_settings['cond_text_max_len'].get()
            cfg_drop = self.dit_settings['cfg_dropout_prob'].get()

            aug_dict = {k: v.get() for k, v in self.dit_aug_settings.items()}
            labels = self.labels if self.labels else [['']] * len(self.image_paths)
            ds = ConditionalImageDataset(self.image_paths, labels, img_size, cm, aug_dict, text_max)
            dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=nw,
                            pin_memory=(self.device.type == 'cuda'),
                            persistent_workers=(nw > 0))

            latent_shape = (self.vae_settings['vae_latent_channels'].get(),
                            self.vae_settings['vae_latent_h'].get(),
                            self.vae_settings['vae_latent_w'].get())

            for ep in range(epochs):
                if not self.training_model:
                    break

                # Live: LR and preview settings (read every epoch)
                for pg in self.rectified_model.optimizer.param_groups:
                    pg['lr'] = self.dit_settings['lr'].get()
                preview_enabled = self.dit_settings['preview_enabled'].get()
                preview_freq = self.dit_settings['preview_epoch_freq'].get()

                tot, n = 0.0, 0
                for imgs, txts in dl:
                    if not self.training_model:
                        break
                    imgs = imgs.to(self.device)
                    with torch.no_grad():
                        z1 = self.vae_model.encode(imgs)
                    z0 = torch.randn(z1.size(0), *latent_shape, device=self.device)
                    if cond_enabled and self.text_encoder is not None:
                        cond = self.text_encoder(txts.to(self.device))
                    else:
                        cond = None
                    loss = self.rectified_model.train_step(z0, z1, cond=cond,
                                                           cfg_dropout_prob=cfg_drop)
                    tot += loss; n += 1
                avg = tot / max(1, n)
                el = time.time() - self.model_start_time
                self.log_model(f"Epoch {ep+1}/{epochs} | Loss: {avg:.6f} | Time: {el:.1f}s")
                if preview_enabled and preview_freq > 0 and (ep + 1) % preview_freq == 0:
                    self.model_preview_with_prompt()
            self.training_model = False
            self.log_model("Training finished.")
        except Exception as e:
            import traceback; traceback.print_exc()
            self.log_model(f"Training error: {e}")
            self.training_model = False

    def stop_model_training(self):
        self.training_model = False
        self.log_model("Training stopped.")

    def model_preview_with_prompt(self):
        if not self.rectified_model or not self.vae_model:
            self.log_model("Models not loaded for preview"); return
        try:
            prompt = self.test_prompt_entry.get().strip()
            cond_enabled = self.dit_settings['cond_enabled'].get()
            unconditional = (prompt == "") or not cond_enabled
            cond = None
            cfg_scale = 1.0 if unconditional else self.cfg_scale.get()

            if not unconditional and self.text_encoder is not None:
                ti = text_to_indices(prompt, self.dit_settings['cond_text_max_len'].get())
                tt = torch.tensor([ti] * 16, dtype=torch.long, device=self.device)
                with torch.no_grad():
                    cond = self.text_encoder(tt)
                self.log_model(f"Preview: conditional on '{prompt}'")
            elif not unconditional:
                self.log_model("Preview: unconditional")
                unconditional = True; cfg_scale = 1.0
            else:
                self.log_model("Preview: unconditional")

            z = self.rectified_model.sample(
                n_samples=16,
                steps=self.dit_settings['preview_steps'].get(),
                method=self.dit_settings['preview_method'].get(),
                cond=cond,
                cfg_scale=cfg_scale,
            )
            with torch.no_grad():
                samples = self.vae_model.decode(
                    z, target_size=(self.global_settings['img_size'].get(),
                                    self.global_settings['img_size'].get()))
            samples = ((samples + 1) / 2).clamp(0, 1).cpu().numpy()
            thumb = self.thumbnail_size
            grid = Image.new('RGB', (4 * thumb, 4 * thumb))
            for i in range(4):
                for j in range(4):
                    idx = i * 4 + j
                    if idx < len(samples):
                        if samples[idx].shape[0] == 1:
                            im = np.stack([samples[idx][0] * 255] * 3, axis=-1).astype(np.uint8)
                        else:
                            im = (samples[idx].transpose(1, 2, 0) * 255).astype(np.uint8)
                        pil = Image.fromarray(im).resize((thumb, thumb), Image.NEAREST)
                        grid.paste(pil, (j * thumb, i * thumb))
            grid = grid.resize((256, 256), Image.NEAREST)
            self.model_preview_photo = ImageTk.PhotoImage(grid)
            self.model_preview_canvas.delete("all")
            self.model_preview_canvas.create_image(128, 128, image=self.model_preview_photo)
        except Exception as e:
            self.log_model(f"Preview error: {e}")

    def save_model(self):
        if not self.rectified_model:
            self.log_model("No model."); return
        fname = filedialog.asksaveasfilename(defaultextension=".pth",
                                             filetypes=[("PyTorch", "*.pth")])
        if fname:
            d = {
                'model_state': self.rectified_model.model.state_dict(),
                'optimizer_state': self.rectified_model.optimizer.state_dict(),
                'dit_settings': {k: v.get() for k, v in self.dit_settings.items()},
                'vae_latent': {
                    'latent_channels': self.vae_settings['vae_latent_channels'].get(),
                    'latent_h': self.vae_settings['vae_latent_h'].get(),
                    'latent_w': self.vae_settings['vae_latent_w'].get(),
                },
                'model_type': self.dit_settings['model_type'].get(),
            }
            if self.text_encoder is not None:
                d['text_encoder_state'] = self.text_encoder.state_dict()
            torch.save(d, fname)
            self.log_model(f"Model saved to {fname}")

    def load_model(self):
        fname = filedialog.askopenfilename(filetypes=[("PyTorch", "*.pth")])
        if not fname:
            return
        try:
            ckpt = torch.load(fname, map_location='cpu')
            for k, v in ckpt.get('dit_settings', {}).items():
                if k in self.dit_settings:
                    self.dit_settings[k].set(v)
            for k, v in ckpt.get('vae_latent', {}).items():
                map_to = {'latent_channels': 'vae_latent_channels',
                          'latent_h': 'vae_latent_h', 'latent_w': 'vae_latent_w'}[k]
                self.vae_settings[map_to].set(v)
            if not self.vae_model:
                self.log_model("Load VAE first, then retry."); return
            self.initialize_model()
            self.rectified_model.model.load_state_dict(ckpt['model_state'])
            try:
                self.rectified_model.optimizer.load_state_dict(ckpt['optimizer_state'])
            except Exception:
                pass
            if 'text_encoder_state' in ckpt and self.text_encoder is not None:
                self.text_encoder.load_state_dict(ckpt['text_encoder_state'])
            self.log_model(f"Model loaded from {fname}")
        except Exception as e:
            self.log_model(f"Load error: {e}")

    # ---------------- Generation ----------------
    def generate_samples(self):
        if not self.rectified_model or not self.vae_model:
            self.gen_info.config(text="Models not loaded!"); return
        cond_enabled = self.dit_settings['cond_enabled'].get()
        if cond_enabled and self.text_encoder is None:
            self.gen_info.config(text="Text encoder missing!"); return
        n = self.gen_count.get()
        steps = self.ode_steps.get()
        method = self.ode_method.get()
        prompt = self.gen_prompt.get().strip()
        cfg_scale = self.cfg_scale.get()
        unconditional = (prompt == "") or not cond_enabled
        cond = None
        eff_cfg = 1.0 if unconditional else cfg_scale
        if not unconditional and self.text_encoder is not None:
            ti = text_to_indices(prompt, self.dit_settings['cond_text_max_len'].get())
            tt = torch.tensor([ti] * n, dtype=torch.long, device=self.device)
            with torch.no_grad():
                cond = self.text_encoder(tt)
        elif not unconditional:
            unconditional = True; eff_cfg = 1.0

        if self.progressive_grid.get():
            self.start_progressive(n, steps, method, cond, eff_cfg)
        else:
            self.generate_btn.config(state=tk.DISABLED)
            self.gen_info.config(text="Generating...")
            self.root.update()
            threading.Thread(target=self._generate_thread,
                             args=(n, steps, method, cond, eff_cfg),
                             daemon=True).start()

    def stop_progressive(self):
        self.progressive_active = False
        self.stop_prog_btn.config(state=tk.DISABLED)
        self.generate_btn.config(state=tk.NORMAL)
        self.gen_info.config(text="Stopped.")

    def start_progressive(self, n, steps, method, cond, cfg_scale):
        self.progressive_active = True
        self.generate_btn.config(state=tk.DISABLED)
        self.stop_prog_btn.config(state=tk.NORMAL)
        self.gen_info.config(text="Progressive generation...")
        threading.Thread(target=self._progressive_thread,
                         args=(n, steps, method, cond, cfg_scale),
                         daemon=True).start()

    def _grid_from_latents(self, z, n):
        with torch.no_grad():
            samples = self.vae_model.decode(
                z, target_size=(self.global_settings['img_size'].get(),
                                self.global_settings['img_size'].get()))
        samples = ((samples + 1) / 2).clamp(0, 1).cpu().numpy()
        thumb = self.thumbnail_size
        gs = int(math.ceil(math.sqrt(n)))
        grid = Image.new('RGB', (gs * thumb, gs * thumb), color=(128, 128, 128))
        for i in range(n):
            if i >= len(samples):
                break
            r, c = i // gs, i % gs
            if samples[i].shape[0] == 1:
                im = np.stack([samples[i][0] * 255] * 3, axis=-1).astype(np.uint8)
            else:
                im = (samples[i].transpose(1, 2, 0) * 255).astype(np.uint8)
            pil = Image.fromarray(im).resize((thumb, thumb), Image.NEAREST)
            grid.paste(pil, (c * thumb, r * thumb))
        return grid

    def _generate_thread(self, n, steps, method, cond, cfg_scale):
        try:
            z = self.rectified_model.sample(n_samples=n, steps=steps, method=method,
                                            cond=cond, cfg_scale=cfg_scale)
            grid = self._grid_from_latents(z, n)
            self.root.after(0, lambda: self._display_generated(grid))
        except Exception as e:
            self.root.after(0, lambda: self.gen_info.config(text=f"Error: {e}"))
        finally:
            self.root.after(0, lambda: self.generate_btn.config(state=tk.NORMAL))

    def _progressive_thread(self, n, steps, method, cond, cfg_scale):
        try:
            interval = self.prog_interval.get()
            gen = self.rectified_model.sample_step_by_step(
                n_samples=n, steps=steps, method=method, cond=cond, cfg_scale=cfg_scale)
            for step, z in gen:
                if not self.progressive_active:
                    break
                if step % interval == 0 or step == steps:
                    grid = self._grid_from_latents(z, n)
                    self.root.after(0, lambda g=grid, s=step: self._update_progressive(g, s))
                    time.sleep(0.05)
            self.root.after(0, lambda: self.gen_info.config(
                text="Progressive stopped." if not self.progressive_active
                else "Progressive finished."))
        except Exception as e:
            self.root.after(0, lambda: self.gen_info.config(text=f"Error: {e}"))
        finally:
            self.progressive_active = False
            self.root.after(0, lambda: self.generate_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.stop_prog_btn.config(state=tk.DISABLED))

    def _display_generated(self, grid):
        for w in self.inner_frame.winfo_children():
            w.destroy()
        self.gen_photo = ImageTk.PhotoImage(grid)
        lbl = tk.Label(self.inner_frame, image=self.gen_photo)
        lbl.image = self.gen_photo
        lbl.pack()
        self.inner_frame.update_idletasks()
        self.gen_canvas.configure(scrollregion=self.gen_canvas.bbox('all'))
        self.gen_info.config(text="Generation complete.")

    def _update_progressive(self, grid, step):
        for w in self.inner_frame.winfo_children():
            w.destroy()
        self.prog_photo = ImageTk.PhotoImage(grid)
        lbl = tk.Label(self.inner_frame, image=self.prog_photo)
        lbl.image = self.prog_photo
        lbl.pack()
        self.inner_frame.update_idletasks()
        self.gen_canvas.configure(scrollregion=self.gen_canvas.bbox('all'))
        self.gen_info.config(text=f"ODE step {step}/{self.ode_steps.get()}")

# ==================== Main ====================
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    root = tk.Tk()
    app = RectifiedFlowDiTApp(root)
    root.mainloop()