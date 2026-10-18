# flow_api.py
"""
Rectified Flow — API / backend.

Contains:
  - UNet with per-resolution attention toggle (num_heads, head_dim, attn dropout)
  - Configurable augmentation padding (edge / reflect / symmetric / constant),
    with emulated padding for transforms that lack native support
  - Sinkhorn minibatch OT coupling (log-domain, stable) + Hungarian fallback
  - OTFM prior (low-frequency projection + noise interpolation; Malnick et al.
    2025, "Designing Optimal Transport Flows")
  - Inpainting with per-step generator, noise σ, composite updates
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as F_vision
import torch.nn.functional as F_nn
from PIL import Image
import os
import numpy as np
import math
import random
import io
import inspect
from typing import List, Tuple, Optional
from scipy.optimize import linear_sum_assignment


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
    indices = [ord(ch) if ord(ch) < 256 else 0 for ch in text[:max_len]]
    if len(indices) < max_len:
        indices += [0] * (max_len - len(indices))
    return indices


def get_norm(channels):
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
    kernel_1d = torch.exp(-torch.arange(-size // 2 + 1, size // 2 + 1).float() ** 2 / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    kernel = kernel_2d[None, None, :, :].to(x.device).repeat(x.size(1), 1, 1, 1)
    pad = size // 2
    x_pad = F_nn.pad(x, (pad, pad, pad, pad), mode='reflect')
    return F_nn.conv2d(x_pad, kernel, groups=x.size(1))


# Map torchvision-style padding names to grid_sample names (ElasticTransform).
_GRID_SAMPLE_PAD = {
    "constant":   "zeros",
    "zeros":      "zeros",
    "edge":       "border",
    "border":     "border",
    "reflect":    "reflection",
    "reflection": "reflection",
    "symmetric":  "reflection",
}


def _grid_sample_pad(mode: str) -> str:
    return _GRID_SAMPLE_PAD.get(mode, "border")


# Probe once at import: does this torchvision have native padding_mode on
# RandomRotation / RandomPerspective?
def _cls_has_padding_mode(cls):
    try:
        return 'padding_mode' in inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):
        return False


TV_ROT_HAS_PAD   = _cls_has_padding_mode(transforms.RandomRotation)
TV_PERSP_HAS_PAD = _cls_has_padding_mode(transforms.RandomPerspective)


# ==================== Augmentations ====================

class RandomJPEG:
    def __init__(self, quality_low=50, quality_high=95, p=0.5):
        self.quality_low = quality_low
        self.quality_high = quality_high
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        quality = random.randint(self.quality_low, self.quality_high)
        img_pil = transforms.ToPILImage()(img) if isinstance(img, torch.Tensor) else img
        buffer = io.BytesIO()
        img_pil.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        compressed = Image.open(buffer).convert('RGB')
        if isinstance(img, torch.Tensor):
            return transforms.ToTensor()(compressed)
        return compressed


class ElasticTransform:
    def __init__(self, alpha=30, sigma=3, p=0.5, padding_mode="edge"):
        self.alpha = alpha
        self.sigma = sigma
        self.p = p
        self.padding_mode = _grid_sample_pad(padding_mode)

    def __call__(self, img):
        if random.random() > self.p:
            return img
        img_pil = transforms.ToPILImage()(img) if isinstance(img, torch.Tensor) else img
        w, h = img_pil.size
        dx = torch.randn(1, h, w) * self.sigma
        dy = torch.randn(1, h, w) * self.sigma
        kernel = torch.ones(1, 1, 5, 5) / 25
        dx = F_nn.conv2d(dx.view(1, 1, h, w), kernel, padding=2).view(h, w) * self.alpha
        dy = F_nn.conv2d(dy.view(1, 1, h, w), kernel, padding=2).view(h, w) * self.alpha
        x, y = torch.meshgrid(torch.arange(w), torch.arange(h), indexing='xy')
        x = x.float() + dx
        y = y.float() + dy
        x = (x / (w - 1)) * 2 - 1
        y = (y / (h - 1)) * 2 - 1
        grid = torch.stack([x, y], dim=-1).unsqueeze(0)
        img_tensor = transforms.ToTensor()(img_pil).unsqueeze(0)
        deformed = F_nn.grid_sample(img_tensor, grid, mode='bilinear',
                                    padding_mode=self.padding_mode)
        return transforms.ToPILImage()(deformed.squeeze(0))


class PaddedTransform:
    """
    Emulate `padding_mode` for PIL-level transforms that lack native support.

    Strategy:
      1. Pad the PIL image with the requested mode.
      2. Run the wrapped transform on the padded image.
      3. Crop the result back, centered, to the original size.

    The padded border content survives the warp, so after the crop the
    "empty corners" that would normally be black are gone.

    pad_fraction controls how much room we make. It must exceed the
    transform's max displacement:
      - RandomRotation(30°)    → ~0.20 is enough
      - RandomPerspective(0.5) → ~0.25–0.35 is safer
    """
    def __init__(self, transform, pad_fraction=0.2, padding_mode='edge'):
        self.transform = transform
        self.pad_fraction = pad_fraction
        mode = padding_mode if padding_mode in (
            'constant', 'edge', 'reflect', 'symmetric') else 'edge'
        self.padding_mode = mode

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)
        w, h = img.size
        pad = max(1, int(max(w, h) * self.pad_fraction))
        if self.padding_mode == 'constant':
            padded = F_vision.pad(img, (pad, pad, pad, pad),
                                  padding_mode='constant', fill=0)
        else:
            padded = F_vision.pad(img, (pad, pad, pad, pad),
                                  padding_mode=self.padding_mode)
        out = self.transform(padded)
        nw, nh = out.size
        left = max(0, (nw - w) // 2)
        top = max(0, (nh - h) // 2)
        return out.crop((left, top, left + w, top + h))


# ==================== Sinusoidal Time Embeddings ====================

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
                          batch_first=True, bidirectional=True,
                          dropout=0.1 if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size * 2, cond_dim)

    def forward(self, x):
        emb = self.embedding(x)
        _, h = self.gru(emb)
        return self.fc(torch.cat([h[-2, :, :], h[-1, :, :]], dim=1))


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
        emb = self.embedding(x)
        x = self.transformer(emb + self.pos_embedding[:, :T, :])
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


# ==================== Conv / Attention / Blocks ====================

class Conv2dWithPadding(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding_mode='constant', **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, **kwargs)
        self.padding_mode = padding_mode

    def forward(self, x):
        pad_h = self.kernel_size[0] // 2
        pad_w = self.kernel_size[1] // 2
        x = F_nn.pad(x, (pad_w, pad_w, pad_h, pad_h), mode=self.padding_mode)
        return F_nn.conv2d(x, self.weight, self.bias, self.stride, 0, self.dilation, self.groups)


class AttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.0):
        super().__init__()
        if num_heads < 1:
            num_heads = 1
        if dim % num_heads != 0:
            for h in range(min(num_heads, dim), 0, -1):
                if dim % h == 0:
                    num_heads = h
                    break
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.norm = get_norm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

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
                 use_attention=False, attn_num_heads=4, attn_dropout=0.0,
                 padding_mode='constant'):
        super().__init__()
        self.conv1 = Conv2dWithPadding(in_channels, out_channels, 3, padding_mode=padding_mode)
        self.norm1 = get_norm(out_channels)
        self.conv2 = Conv2dWithPadding(out_channels, out_channels, 3, padding_mode=padding_mode)
        self.norm2 = get_norm(out_channels)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels))
        self.cond_proj = nn.Linear(cond_dim, out_channels)
        self.res_conv = (Conv2dWithPadding(in_channels, out_channels, 1, padding_mode=padding_mode)
                         if in_channels != out_channels else nn.Identity())
        self.dropout = nn.Dropout(dropout)
        self.attn = (AttentionBlock(out_channels, attn_num_heads, attn_dropout)
                     if use_attention else nn.Identity())

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
                 use_attention=False, attn_num_heads=4, attn_dropout=0.0,
                 padding_mode='constant'):
        super().__init__()
        self.conv1 = Conv2dWithPadding(in_channels, out_channels, 3, padding_mode=padding_mode)
        self.norm1 = get_norm(out_channels)
        self.conv2 = Conv2dWithPadding(out_channels, out_channels, 3, padding_mode=padding_mode)
        self.norm2 = get_norm(out_channels)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels))
        self.cond_proj = nn.Linear(cond_dim, out_channels)
        self.res_conv = (Conv2dWithPadding(in_channels, out_channels, 1, padding_mode=padding_mode)
                         if in_channels != out_channels else nn.Identity())
        self.dropout = nn.Dropout(dropout)
        self.attn = (AttentionBlock(out_channels, attn_num_heads, attn_dropout)
                     if use_attention else nn.Identity())

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
                 img_size=32, channel_mult=(1, 2, 4), dropout=0.1,
                 attention_levels=None, attn_num_heads=4, attn_dropout=0.0,
                 padding_mode='constant'):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.time_emb_dim = time_emb_dim
        self.cond_dim = cond_dim
        self.channel_mult = channel_mult
        self.padding_mode = padding_mode
        self.attn_num_heads = attn_num_heads
        self.attn_dropout = attn_dropout

        H, W = (img_size, img_size) if isinstance(img_size, int) else img_size
        num_down = 0
        cur_h, cur_w = H, W
        while cur_h >= 8 and cur_w >= 8:
            cur_h //= 2
            cur_w //= 2
            num_down += 1
        num_down = min(num_down, len(channel_mult))
        cm = channel_mult[:num_down + 1]

        if attention_levels is None:
            attention_levels = [True] * len(cm)
        attn_flags = list(attention_levels)
        if len(attn_flags) < len(cm):
            attn_flags += [False] * (len(cm) - len(attn_flags))
        attn_flags = attn_flags[:len(cm)]
        self.attention_levels = attn_flags

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        self.init_conv = Conv2dWithPadding(in_channels, base_channels, 3, padding_mode=padding_mode)

        self.downs = nn.ModuleList()
        cur_channels = base_channels
        for i, mult in enumerate(cm):
            out_channels = base_channels * mult
            self.downs.append(DownBlock(
                cur_channels, out_channels, time_emb_dim, cond_dim, dropout,
                use_attention=attn_flags[i],
                attn_num_heads=attn_num_heads,
                attn_dropout=attn_dropout,
                padding_mode=padding_mode))
            if i < len(cm) - 1:
                self.downs.append(nn.Conv2d(out_channels, out_channels, 4, stride=2, padding=1))
            cur_channels = out_channels

        mid_attn = attn_flags[-1] if attn_flags else True
        self.mid_block1 = DownBlock(cur_channels, cur_channels, time_emb_dim, cond_dim, dropout,
                                    use_attention=mid_attn,
                                    attn_num_heads=attn_num_heads,
                                    attn_dropout=attn_dropout,
                                    padding_mode=padding_mode)
        self.mid_block2 = UpBlock(cur_channels, cur_channels, time_emb_dim, cond_dim, dropout,
                                  use_attention=mid_attn,
                                  attn_num_heads=attn_num_heads,
                                  attn_dropout=attn_dropout,
                                  padding_mode=padding_mode)

        self.ups = nn.ModuleList()
        rev_blocks = list(reversed(cm))
        rev_attn = list(reversed(attn_flags))
        for i, mult in enumerate(rev_blocks):
            out_channels = base_channels * mult
            self.ups.append(UpBlock(
                cur_channels + out_channels, out_channels, time_emb_dim, cond_dim, dropout,
                use_attention=rev_attn[i],
                attn_num_heads=attn_num_heads,
                attn_dropout=attn_dropout,
                padding_mode=padding_mode))
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


# ==================== OT coupling ====================

@torch.no_grad()
def match_ot_batch_hungarian(x0, x1):
    """Exact minibatch OT via Hungarian (O(B^3))."""
    B = x0.shape[0]
    if B <= 1:
        return x0, x1, torch.arange(B)
    x0_flat = x0.reshape(B, -1)
    x1_flat = x1.reshape(B, -1)
    dist = (torch.sum(x0_flat ** 2, dim=1, keepdim=True) +
            torch.sum(x1_flat ** 2, dim=1, keepdim=True).T -
            2.0 * torch.mm(x0_flat, x1_flat.T))
    cost = dist.cpu().numpy()
    _, col_ind = linear_sum_assignment(cost)
    col_ind = torch.from_numpy(col_ind).to(x1.device)
    return x0, x1[col_ind], col_ind


@torch.no_grad()
def match_ot_batch_sinkhorn(x0, x1, epsilon=0.05, n_iters=50):
    """
    Entropic (Sinkhorn) minibatch OT coupling in log domain, then extract
    a hard permutation by maximizing sum P[i, σ(i)] via Hungarian on -P.
    """
    B = x0.shape[0]
    if B <= 1:
        return x0, x1, torch.arange(B)
    x0_flat = x0.reshape(B, -1)
    x1_flat = x1.reshape(B, -1)
    cost = torch.cdist(x0_flat, x1_flat, p=2).pow(2)

    log_K = -cost / epsilon
    log_u = torch.zeros(B, device=cost.device)
    log_v = torch.zeros(B, device=cost.device)
    log_B = math.log(B)
    for _ in range(n_iters):
        log_u = -torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1) - log_B
        log_v = -torch.logsumexp(log_K + log_u.unsqueeze(1), dim=0) - log_B
    log_P = log_u.unsqueeze(1) + log_K + log_v.unsqueeze(0)
    P = log_P.exp()

    P_np = P.detach().cpu().numpy()
    _, col_ind = linear_sum_assignment(-P_np)
    col_ind = torch.from_numpy(col_ind).to(x1.device)
    return x0, x1[col_ind], col_ind


def match_ot_batch(x0, x1, method='sinkhorn', sinkhorn_eps=0.05, sinkhorn_iters=50):
    if method == 'sinkhorn':
        return match_ot_batch_sinkhorn(x0, x1, epsilon=sinkhorn_eps, n_iters=sinkhorn_iters)
    return match_ot_batch_hungarian(x0, x1)


# ==================== Rectified Flow Model ====================

class RectifiedFlowPixel:
    def __init__(self, in_channels=3, img_size=32, base_channels=64,
                 cond_dim=256, device=None, attention_levels=None,
                 attn_num_heads=4, attn_dropout=0.0,
                 task="noise", task_params=None, training_mode="dataset",
                 padding_mode='constant'):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.cond_dim = cond_dim
        self.task = task
        self.task_params = task_params or {}
        self.training_mode = training_mode
        self.in_channels = in_channels

        self.use_ot = False
        self.ot_method = 'sinkhorn'
        self.ot_sinkhorn_eps = 0.05
        self.ot_sinkhorn_iters = 50

        self.model = VelocityUNet(
            in_channels=in_channels,
            base_channels=base_channels,
            time_emb_dim=base_channels * 4,
            cond_dim=cond_dim,
            img_size=img_size,
            channel_mult=(1, 2, 4),
            dropout=0.1,
            attention_levels=attention_levels,
            attn_num_heads=attn_num_heads,
            attn_dropout=attn_dropout,
            padding_mode=padding_mode
        ).to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=2e-4)

        self.ema_model = None
        self.ema_decay = 0.999
        self.ema_mode = 'standard'
        self.use_ema = False

    # ---------- EMA ----------
    def set_ema(self, decay=0.999, mode='standard'):
        self.ema_decay = decay
        self.ema_mode = mode
        self.use_ema = True
        self.ema_model = VelocityUNet(
            in_channels=self.model.in_channels,
            base_channels=self.model.base_channels,
            time_emb_dim=self.model.time_emb_dim,
            cond_dim=self.model.cond_dim,
            img_size=self.img_size,
            channel_mult=self.model.channel_mult,
            dropout=0.1,
            attention_levels=self.model.attention_levels,
            attn_num_heads=self.model.attn_num_heads,
            attn_dropout=self.model.attn_dropout,
            padding_mode=self.model.padding_mode
        ).to(self.device)
        self.ema_model.load_state_dict(self.model.state_dict())
        for p in self.ema_model.parameters():
            p.requires_grad = False

    def update_ema(self):
        if self.ema_model is None:
            return
        with torch.no_grad():
            for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)
        if self.ema_mode == 'lookahead':
            with torch.no_grad():
                for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
                    param.data.copy_(ema_param.data)

    # ---------- source generation ----------
    def _add_noise_to_source(self, x0):
        noise_std = self.task_params.get('source_noise_std', 0.0)
        if noise_std <= 0.0:
            return x0
        return x0 + torch.randn_like(x0) * (noise_std * 2.0)

    def _random_inpaint_mask(self, shape, mask_ratio=0.3):
        B, _, H, W = shape
        noise = torch.rand(B, 1, H, W, device=self.device)
        mask = (noise < mask_ratio).float()
        mask = F_nn.max_pool2d(mask, 3, stride=1, padding=1)
        return mask

    def _generate_source_from_target(self, x1):
        batch_size = x1.size(0)
        if self.task == "noise":
            x0 = torch.randn_like(x1)
        elif self.task == "grayscale_to_color":
            gray = 0.2989 * x1[:, 0:1] + 0.5870 * x1[:, 1:2] + 0.1140 * x1[:, 2:3]
            x0 = gray.repeat(1, 3, 1, 1)
        elif self.task == "blur_to_sharp":
            x0 = gaussian_blur_tensor(x1, self.task_params.get('blur_sigma', 2.0))
        elif self.task == "white_to_image":
            x0 = torch.full_like(x1, 1.0)
        elif self.task == "average_separation":
            idx = torch.randperm(batch_size, device=x1.device)
            x0 = (x1 + x1[idx]) / 2.0
        elif self.task == "unquantization":
            levels = 2 ** self.task_params.get('bit_depth', 1)
            x1_01 = (x1 + 1) / 2
            q = torch.floor(x1_01 * levels) / (levels - 1 + 1e-8)
            x0 = q * 2 - 1
        elif self.task == "pixelation":
            block = self.task_params.get('pixelation_size', 8)
            mode = self.task_params.get('pixelation_rescale_mode', 'nearest')
            H, W = x1.shape[2], x1.shape[3]
            H_new = (H // block) * block
            W_new = (W // block) * block
            cropped = x1[:, :, :H_new, :W_new]
            down = F_nn.avg_pool2d(cropped, block, stride=block)
            up = F_nn.interpolate(down, size=(H_new, W_new), mode=mode)
            if (H_new, W_new) != (H, W):
                up = F_nn.pad(up, (0, W - W_new, 0, H - H_new))
            x0 = up
        elif self.task == "inpainting":
            mask = self._random_inpaint_mask(x1.shape,
                                             self.task_params.get('inpaint_mask_ratio', 0.3))
            noise = torch.randn_like(x1)
            x0 = x1 * (1 - mask) + noise * mask
        elif self.task == "otfm":
            # OTFM prior (Malnick et al.): x0 = (1 - α)·U(D(x1)) + α·ε
            alpha = float(self.task_params.get('otfm_alpha', 0.5))
            factor = max(1, int(self.task_params.get('otfm_downsample', 4)))
            H, W = x1.shape[2], x1.shape[3]
            H_new = max(1, H // factor)
            W_new = max(1, W // factor)
            down = F_nn.interpolate(x1, size=(H_new, W_new), mode='area')
            up = F_nn.interpolate(down, size=(H, W), mode='bicubic', align_corners=False)
            eps = torch.randn_like(x1)
            x0 = (1.0 - alpha) * up + alpha * eps
        else:
            raise ValueError(f"Unknown task {self.task}")

        if self.task not in ("noise", "inpainting", "otfm"):
            x0 = self._add_noise_to_source(x0)
        return x0

    def _generate_source_from_noise(self, shape):
        noise = torch.randn(shape, device=self.device)
        if self.task in ("noise", "inpainting", "average_separation", "otfm"):
            x0 = noise
        elif self.task == "grayscale_to_color":
            gray = torch.randn(shape[0], 1, shape[2], shape[3], device=self.device)
            x0 = gray.repeat(1, 3, 1, 1)
        elif self.task == "blur_to_sharp":
            x0 = gaussian_blur_tensor(noise, self.task_params.get('blur_sigma', 2.0))
        elif self.task == "white_to_image":
            x0 = torch.full(shape, 1.0, device=self.device)
        elif self.task == "unquantization":
            levels = 2 ** self.task_params.get('bit_depth', 1)
            n01 = (noise + 1) / 2
            q = torch.floor(n01 * levels) / (levels - 1 + 1e-8)
            x0 = q * 2 - 1
        elif self.task == "pixelation":
            block = self.task_params.get('pixelation_size', 8)
            mode = self.task_params.get('pixelation_rescale_mode', 'nearest')
            H, W = shape[2], shape[3]
            H_new = (H // block) * block
            W_new = (W // block) * block
            cropped = noise[:, :, :H_new, :W_new]
            down = F_nn.avg_pool2d(cropped, block, stride=block)
            up = F_nn.interpolate(down, size=(H_new, W_new), mode=mode)
            if (H_new, W_new) != (H, W):
                up = F_nn.pad(up, (0, W - W_new, 0, H - H_new))
            x0 = up
        else:
            x0 = noise
        if self.task not in ("noise", "inpainting", "otfm"):
            x0 = self._add_noise_to_source(x0)
        return x0

    # ---------- train step ----------
    def train_step(self, x1, cond=None, cfg_dropout_prob=0.0):
        self.model.train()
        x1 = x1.to(self.device)
        B = x1.size(0)

        if self.training_mode == "dataset":
            x0 = self._generate_source_from_target(x1)
        elif self.training_mode == "noise":
            x0 = self._generate_source_from_noise(x1.shape)
        else:
            x0 = (self._generate_source_from_target(x1) if random.random() < 0.5
                  else self._generate_source_from_noise(x1.shape))

        if self.use_ot and B > 1:
            x0, x1, col_ind = match_ot_batch(
                x0, x1,
                method=self.ot_method,
                sinkhorn_eps=self.ot_sinkhorn_eps,
                sinkhorn_iters=self.ot_sinkhorn_iters)
            if cond is not None:
                cond = cond[col_ind]

        if cond is None:
            cond = torch.zeros(B, self.cond_dim, device=self.device)
        else:
            cond = cond.to(self.device)

        t = torch.rand(B, device=self.device)
        x_t = t.view(-1, 1, 1, 1) * x1 + (1 - t.view(-1, 1, 1, 1)) * x0
        target = x1 - x0

        if cfg_dropout_prob > 0:
            mask = torch.rand(B, 1, device=self.device) > cfg_dropout_prob
            cond = cond * mask.float()

        pred = self.model(x_t, t, cond=cond)
        loss = self.criterion(pred, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.use_ema:
            self.update_ema()

        return loss.item()

    # ---------- sampling ----------
    @torch.no_grad()
    def sample(self, n_samples=16, cond=None, steps=50, method='euler', cfg_scale=1.0,
               use_ema=True, start_from=None):
        model = self.ema_model if (use_ema and self.use_ema and self.ema_model is not None) else self.model
        model.eval()
        img_shape = (self.in_channels, self.img_size[0], self.img_size[1])

        x = start_from.to(self.device) if start_from is not None \
            else self._generate_source_from_noise((n_samples, *img_shape))

        if cond is None:
            cond = torch.zeros(x.size(0), self.cond_dim, device=self.device)
        else:
            cond = cond.to(self.device)
        null_cond = torch.zeros_like(cond)

        dt = 1.0 / steps
        times = torch.linspace(0, 1, steps + 1, device=self.device)

        def v_fn(t_val, x_val):
            t_tensor = torch.full((x_val.size(0),), t_val, device=self.device)
            if cfg_scale != 1.0:
                v_cond = model(x_val, t_tensor, cond=cond)
                v_uncond = model(x_val, t_tensor, cond=null_cond)
                return v_uncond + cfg_scale * (v_cond - v_uncond)
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
                x2 = x + v1 * dt / 2
                v2 = v_fn(t_cur + dt / 2, x2)
                x3 = x + v2 * dt / 2
                v3 = v_fn(t_cur + dt / 2, x3)
                x4 = x + v3 * dt
                v4 = v_fn(t_cur + dt, x4)
                x = x + (v1 + 2 * v2 + 2 * v3 + v4) * dt / 6
            elif method == 'midpoint':
                t_mid = t_cur + dt / 2
                x_mid = x + v_fn(t_cur, x) * dt / 2
                x = x + v_fn(t_mid, x_mid) * dt
            else:
                raise ValueError(f"Unknown method {method}")
        return x

    @torch.no_grad()
    def inpaint_step_by_step(self, image, mask, steps=50, method='euler', cfg_scale=1.0,
                             use_ema=True, cond=None, noise_sigma=1.0,
                             composite_updates=True, full_edit_from=0.7):
        model = self.ema_model if (use_ema and self.use_ema and self.ema_model is not None) else self.model
        image = image.to(self.device)
        mask = mask.to(self.device)
        noise = torch.randn_like(image) * float(noise_sigma)
        x = image * (1 - mask) + noise * mask

        n = image.size(0)
        if cond is None:
            cond = torch.zeros(n, self.cond_dim, device=self.device)
        else:
            cond = cond.to(self.device)
        null_cond = torch.zeros_like(cond)

        dt = 1.0 / steps
        times = torch.linspace(0, 1, steps + 1, device=self.device)

        def v_fn(t_val, x_val):
            t_tensor = torch.full((x_val.size(0),), t_val, device=self.device)
            if cfg_scale != 1.0:
                v_cond = model(x_val, t_tensor, cond=cond)
                v_uncond = model(x_val, t_tensor, cond=null_cond)
                return v_uncond + cfg_scale * (v_cond - v_uncond)
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
                x2 = x + v1 * dt / 2
                v2 = v_fn(t_cur + dt / 2, x2)
                x3 = x + v2 * dt / 2
                v3 = v_fn(t_cur + dt / 2, x3)
                x4 = x + v3 * dt
                v4 = v_fn(t_cur + dt, x4)
                x = x + (v1 + 2 * v2 + 2 * v3 + v4) * dt / 6
            elif method == 'midpoint':
                t_mid = t_cur + dt / 2
                x_mid = x + v_fn(t_cur, x) * dt / 2
                x = x + v_fn(t_mid, x_mid) * dt
            else:
                raise ValueError(f"Unknown method {method}")

            if composite_updates:
                progress = (i + 1) / steps
                if progress < full_edit_from:
                    x = x * mask + image * (1 - mask)

            yield i + 1, x.clone()

    @torch.no_grad()
    def inpaint(self, image, mask, steps=50, method='euler', cfg_scale=1.0,
                use_ema=True, cond=None, noise_sigma=1.0,
                composite_updates=True, full_edit_from=0.7):
        last = None
        for _, x in self.inpaint_step_by_step(
                image, mask, steps=steps, method=method, cfg_scale=cfg_scale,
                use_ema=use_ema, cond=cond, noise_sigma=noise_sigma,
                composite_updates=composite_updates, full_edit_from=full_edit_from):
            last = x
        return last

    @torch.no_grad()
    def sample_step_by_step(self, n_samples=16, cond=None, steps=50, method='euler',
                            cfg_scale=1.0, use_ema=True, start_from=None):
        model = self.ema_model if (use_ema and self.use_ema and self.ema_model is not None) else self.model
        img_shape = (self.in_channels, self.img_size[0], self.img_size[1])
        x = start_from.to(self.device) if start_from is not None \
            else self._generate_source_from_noise((n_samples, *img_shape))

        if cond is None:
            cond = torch.zeros(x.size(0), self.cond_dim, device=self.device)
        else:
            cond = cond.to(self.device)
        null_cond = torch.zeros_like(cond)

        dt = 1.0 / steps
        times = torch.linspace(0, 1, steps + 1, device=self.device)

        def v_fn(t_val, x_val):
            t_tensor = torch.full((x_val.size(0),), t_val, device=self.device)
            if cfg_scale != 1.0:
                v_cond = model(x_val, t_tensor, cond=cond)
                v_uncond = model(x_val, t_tensor, cond=null_cond)
                return v_uncond + cfg_scale * (v_cond - v_uncond)
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
                x2 = x + v1 * dt / 2
                v2 = v_fn(t_cur + dt / 2, x2)
                x3 = x + v2 * dt / 2
                v3 = v_fn(t_cur + dt / 2, x3)
                x4 = x + v3 * dt
                v4 = v_fn(t_cur + dt, x4)
                x = x + (v1 + 2 * v2 + 2 * v3 + v4) * dt / 6
            elif method == 'midpoint':
                t_mid = t_cur + dt / 2
                x_mid = x + v_fn(t_cur, x) * dt / 2
                x = x + v_fn(t_mid, x_mid) * dt
            else:
                raise ValueError(f"Unknown method {method}")
            yield i + 1, x.clone()


# ==================== Dataset ====================

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

        aug_pad = self.aug_settings.get('aug_padding_mode', 'edge')
        if aug_pad not in ('constant', 'edge', 'reflect', 'symmetric'):
            aug_pad = 'edge'

        self.aug_pipeline = []
        a = self.aug_settings

        if a.get('flip_horizontal', False):
            self.aug_pipeline.append(transforms.RandomHorizontalFlip(p=0.5))

        # --- Rotation ---
        if a.get('rotation', False):
            if TV_ROT_HAS_PAD:
                self.aug_pipeline.append(transforms.RandomRotation(
                    degrees=30, interpolation=Image.BICUBIC, fill=0,
                    padding_mode=aug_pad))
            else:
                base = transforms.RandomRotation(
                    degrees=30, interpolation=Image.BICUBIC, fill=0)
                # 30° rotation on a square needs ≈0.20/side padding; add margin.
                self.aug_pipeline.append(PaddedTransform(
                    base, pad_fraction=0.22, padding_mode=aug_pad))

        if a.get('random_crop', False):
            self.aug_pipeline.append(transforms.RandomResizedCrop(
                size=img_size, scale=(a.get('crop_scale', 0.8), 1.0),
                interpolation=Image.BICUBIC))

        if a.get('color_jitter', False):
            self.aug_pipeline.append(transforms.ColorJitter(
                brightness=a.get('brightness', 0.2),
                contrast=a.get('contrast', 0.2),
                saturation=a.get('saturation', 0.2),
                hue=a.get('hue', 0.1)))

        # --- Perspective ---
        if a.get('random_perspective', False):
            if TV_PERSP_HAS_PAD:
                self.aug_pipeline.append(transforms.RandomPerspective(
                    distortion_scale=a.get('perspective_distortion', 0.1),
                    p=0.5, interpolation=Image.BICUBIC, fill=0,
                    padding_mode=aug_pad))
            else:
                base = transforms.RandomPerspective(
                    distortion_scale=a.get('perspective_distortion', 0.1),
                    p=0.5, interpolation=Image.BICUBIC, fill=0)
                dist = float(a.get('perspective_distortion', 0.1))
                frac = min(0.35, max(0.20, dist * 0.5 + 0.15))
                self.aug_pipeline.append(PaddedTransform(
                    base, pad_fraction=frac, padding_mode=aug_pad))

        # --- Elastic: native grid_sample padding already supported ---
        if a.get('elastic_transform', False):
            self.aug_pipeline.append(ElasticTransform(
                alpha=a.get('elastic_alpha', 30),
                sigma=a.get('elastic_sigma', 3),
                p=0.5, padding_mode=aug_pad))

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


# ==================== Utility ====================

def compute_head_dims(base_channels, channel_mult=(1, 2, 4), num_heads=4):
    result = []
    for mult in channel_mult:
        ch = base_channels * mult
        nh = num_heads if num_heads > 0 else 1
        if ch % nh != 0:
            for h in range(min(nh, ch), 0, -1):
                if ch % h == 0:
                    nh = h
                    break
        result.append((ch, nh, ch // nh))
    return result