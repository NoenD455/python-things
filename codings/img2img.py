# pix2pix_paired.py
# Paired Pix2Pix with bidirectional support, cycle consistency (ABA/BAB),
# color drawing, flexible pairing, and per-direction training preview.
import tkinter as tk
from tkinter import filedialog, ttk, colorchooser
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as F_vision
import torch.nn.functional as F_nn
from PIL import Image, ImageTk, ImageDraw
import os
import threading
import queue
import numpy as np
import multiprocessing
import time
import random
import math
import io
import csv

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

def get_norm(channels):
    if channels % 32 == 0:
        return nn.GroupNorm(32, channels)
    best = 1
    for g in range(32, 0, -1):
        if channels % g == 0:
            best = g
            break
    return nn.GroupNorm(best, channels)

def is_image_file(name):
    return name.lower().endswith(('.png', '.jpg', '.jpeg', '.jfif', '.webp', '.bmp'))

# ==================== SSIM Loss ====================

def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2 / float(2*sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    return _2D_window.expand(channel, 1, window_size, window_size).contiguous()

def ssim_loss(img1, img2, window_size=11, size_average=True):
    _, channel, _, _ = img2.size()
    window = create_window(window_size, channel).to(img1.device)
    mu1 = F_nn.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F_nn.conv2d(img2, window, padding=window_size//2, groups=channel)
    mu1_sq = mu1.pow(2); mu2_sq = mu2.pow(2); mu1_mu2 = mu1 * mu2
    sigma1_sq = F_nn.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F_nn.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F_nn.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2
    C1 = 0.01 ** 2; C2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if size_average:
        return ssim_map.mean()
    return ssim_map.mean(1).mean(1).mean(1)

# ==================== Augmentations ====================

class RandomJPEG:
    def __init__(self, quality_low=50, quality_high=95, p=0.5):
        self.quality_low = quality_low; self.quality_high = quality_high; self.p = p
    def __call__(self, img):
        if random.random() > self.p:
            return img
        quality = random.randint(self.quality_low, self.quality_high)
        img_pil = transforms.ToPILImage()(img) if isinstance(img, torch.Tensor) else img
        buffer = io.BytesIO()
        img_pil.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        compressed = Image.open(buffer).convert(img_pil.mode)
        if isinstance(img, torch.Tensor):
            return transforms.ToTensor()(compressed)
        return compressed

class ElasticTransform:
    def __init__(self, alpha=30, sigma=3, p=0.5):
        self.alpha = alpha; self.sigma = sigma; self.p = p
    def __call__(self, img):
        if random.random() > self.p:
            return img
        img_pil = transforms.ToPILImage()(img) if isinstance(img, torch.Tensor) else img
        w, h = img_pil.size
        dx = torch.randn(1, h, w) * self.sigma
        dy = torch.randn(1, h, w) * self.sigma
        kernel = torch.ones(1, 1, 5, 5) / 25
        dx = F_nn.conv2d(dx.view(1,1,h,w), kernel, padding=2).view(h,w) * self.alpha
        dy = F_nn.conv2d(dy.view(1,1,h,w), kernel, padding=2).view(h,w) * self.alpha
        x, y = torch.meshgrid(torch.arange(w), torch.arange(h), indexing='xy')
        x = x.float() + dx; y = y.float() + dy
        x = (x / (w-1)) * 2 - 1; y = (y / (h-1)) * 2 - 1
        grid = torch.stack([x, y], dim=-1).unsqueeze(0)
        img_tensor = transforms.ToTensor()(img_pil).unsqueeze(0)
        deformed = F_nn.grid_sample(img_tensor, grid, mode='bilinear', padding_mode='border')
        return transforms.ToPILImage()(deformed.squeeze(0))

# ==================== Pix2Pix UNet ====================

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
    def __init__(self, dim, num_heads=4):
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
    def __init__(self, in_channels, out_channels, dropout=0.1,
                 has_attn=False, padding_mode='constant'):
        super().__init__()
        self.conv1 = Conv2dWithPadding(in_channels, out_channels, 3, padding_mode=padding_mode)
        self.norm1 = get_norm(out_channels)
        self.conv2 = Conv2dWithPadding(out_channels, out_channels, 3, padding_mode=padding_mode)
        self.norm2 = get_norm(out_channels)
        self.res_conv = Conv2dWithPadding(in_channels, out_channels, 1, padding_mode=padding_mode) if in_channels != out_channels else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.attn = AttentionBlock(out_channels) if has_attn else nn.Identity()
    def forward(self, x):
        h = self.conv1(x); h = self.norm1(h); h = F_nn.silu(h); h = self.dropout(h)
        h = self.conv2(h); h = self.norm2(h); h = F_nn.silu(h)
        return self.attn(h + self.res_conv(x))

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1,
                 has_attn=False, padding_mode='constant'):
        super().__init__()
        self.conv1 = Conv2dWithPadding(in_channels, out_channels, 3, padding_mode=padding_mode)
        self.norm1 = get_norm(out_channels)
        self.conv2 = Conv2dWithPadding(out_channels, out_channels, 3, padding_mode=padding_mode)
        self.norm2 = get_norm(out_channels)
        self.res_conv = Conv2dWithPadding(in_channels, out_channels, 1, padding_mode=padding_mode) if in_channels != out_channels else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.attn = AttentionBlock(out_channels) if has_attn else nn.Identity()
    def forward(self, x):
        h = self.conv1(x); h = self.norm1(h); h = F_nn.silu(h); h = self.dropout(h)
        h = self.conv2(h); h = self.norm2(h); h = F_nn.silu(h)
        return self.attn(h + self.res_conv(x))

class Pix2PixUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=None, base_channels=64,
                 img_size=32, channel_mult=(1, 2, 3, 4), dropout=0.1,
                 use_attention=True, padding_mode='constant'):
        super().__init__()
        out_channels = out_channels if out_channels is not None else in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        H, W = (img_size, img_size) if isinstance(img_size, int) else img_size
        num_down = 0; cur_h, cur_w = H, W
        while cur_h >= 8 and cur_w >= 8:
            cur_h //= 2; cur_w //= 2; num_down += 1
        num_down = min(num_down, len(channel_mult))
        cm = channel_mult[:num_down+1]
        self.init_conv = Conv2dWithPadding(in_channels, base_channels, 3, padding_mode=padding_mode)
        self.downs = nn.ModuleList()
        cur_channels = base_channels
        for i, mult in enumerate(cm):
            out_ch = base_channels * mult
            self.downs.append(DownBlock(cur_channels, out_ch, dropout,
                                        has_attn=use_attention, padding_mode=padding_mode))
            if i < len(cm) - 1:
                self.downs.append(nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1))
            cur_channels = out_ch
        self.mid_block1 = DownBlock(cur_channels, cur_channels, dropout,
                                    has_attn=use_attention, padding_mode=padding_mode)
        self.mid_block2 = UpBlock(cur_channels, cur_channels, dropout,
                                  has_attn=use_attention, padding_mode=padding_mode)
        self.ups = nn.ModuleList()
        rev = list(reversed(cm))
        for i, mult in enumerate(rev):
            out_ch = base_channels * mult
            self.ups.append(UpBlock(cur_channels + out_ch, out_ch, dropout,
                                    has_attn=use_attention, padding_mode=padding_mode))
            if i < len(rev) - 1:
                self.ups.append(nn.ConvTranspose2d(out_ch, out_ch, 4, stride=2, padding=1))
            cur_channels = out_ch
        self.final_conv = nn.Sequential(
            get_norm(cur_channels),
            nn.SiLU(),
            Conv2dWithPadding(cur_channels, out_channels, 3, padding_mode=padding_mode)
        )
    def forward(self, x):
        x = self.init_conv(x)
        skips = []
        for layer in self.downs:
            if isinstance(layer, DownBlock):
                x = layer(x); skips.append(x)
            else:
                x = layer(x)
        x = self.mid_block1(x)
        x = self.mid_block2(x)
        for layer in self.ups:
            if isinstance(layer, UpBlock):
                skip = skips.pop()
                x = torch.cat([x, skip], dim=1)
                x = layer(x)
            else:
                x = layer(x)
        return self.final_conv(x)

# ==================== Pairing Utilities ====================

def scan_folder(folder):
    paths = []
    for root, _, files in os.walk(folder):
        for f in files:
            if is_image_file(f):
                paths.append(os.path.join(root, f))
    return paths

def pair_by_name(paths_A, paths_B):
    map_A = {}; map_B = {}
    for p in paths_A:
        stem = os.path.splitext(os.path.basename(p))[0].lower()
        map_A.setdefault(stem, []).append(p)
    for p in paths_B:
        stem = os.path.splitext(os.path.basename(p))[0].lower()
        map_B.setdefault(stem, []).append(p)
    pairs = []
    for stem, a_list in map_A.items():
        if stem in map_B:
            pairs.append((a_list[0], map_B[stem][0]))
    return pairs

def pair_by_csv(csv_path, paths_A, paths_B):
    def build_lookup(paths):
        lookup = {}
        for p in paths:
            base = os.path.basename(p)
            stem = os.path.splitext(base)[0]
            lookup[base.lower()] = p
            lookup[stem.lower()] = p
            lookup[p.lower()] = p
        return lookup
    lookup_A = build_lookup(paths_A); lookup_B = build_lookup(paths_B)
    pairs = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2: continue
            a_name, b_name = row[0].strip(), row[1].strip()
            if not a_name or not b_name: continue
            a_path = lookup_A.get(a_name.lower()) or lookup_A.get(os.path.splitext(a_name)[0].lower())
            b_path = lookup_B.get(b_name.lower()) or lookup_B.get(os.path.splitext(b_name)[0].lower())
            if a_path and b_path:
                pairs.append((a_path, b_path))
    return pairs

def pair_by_similarity(paths_A, paths_B, thumb_size=32, log_fn=None):
    def load_thumb(p):
        try:
            img = load_image_as_rgb(p).resize((thumb_size, thumb_size))
            return np.asarray(img, dtype=np.float32) / 255.0
        except Exception:
            return None
    if log_fn:
        log_fn(f"Loading thumbnails ({len(paths_A)} x {len(paths_B)})...")
    thumbs_A = [load_thumb(p) for p in paths_A]
    thumbs_B = [load_thumb(p) for p in paths_B]
    valid_A = [(i, t) for i, t in enumerate(thumbs_A) if t is not None]
    valid_B = [(j, t) for j, t in enumerate(thumbs_B) if t is not None]
    if log_fn:
        log_fn(f"Computing similarity matrix ({len(valid_A)} x {len(valid_B)})...")
    cost = np.zeros((len(valid_A), len(valid_B)), dtype=np.float32)
    for i, (_, ta) in enumerate(valid_A):
        for j, (_, tb) in enumerate(valid_B):
            cost[i, j] = np.mean((ta - tb) ** 2)
    pairs = []
    used_A = set(); used_B = set()
    flat = np.dstack(np.unravel_index(np.argsort(cost, axis=None), cost.shape))[0]
    for i, j in flat:
        if i in used_A or j in used_B: continue
        used_A.add(int(i)); used_B.add(int(j))
        pairs.append((paths_A[valid_A[i][0]], paths_B[valid_B[j][0]]))
        if len(pairs) == min(len(valid_A), len(valid_B)):
            break
    return pairs

# ==================== Paired Dataset ====================

class PairedPix2PixDataset(Dataset):
    def __init__(self, pairs, img_size=64, color_mode='rgb',
                 out_color_mode=None, aug_settings=None):
        self.pairs = pairs
        self.img_size = img_size
        self.color_mode = color_mode.lower()
        self.out_color_mode = (out_color_mode or color_mode).lower()
        self.aug_settings = aug_settings or {}
        self.transform_A = self._make_transform(self.color_mode)
        self.transform_B = self._make_transform(self.out_color_mode)
        self.aug_pipeline = []
        if self.aug_settings.get('flip_horizontal', False):
            self.aug_pipeline.append(('flip_h', None))
        if self.aug_settings.get('flip_vertical', False):
            self.aug_pipeline.append(('flip_v', None))
        if self.aug_settings.get('rotation', False):
            self.aug_pipeline.append(('rot', 30))
        if self.aug_settings.get('random_crop', False):
            self.aug_pipeline.append(('crop', self.aug_settings.get('crop_scale', 0.8)))
        if self.aug_settings.get('random_perspective', False):
            self.aug_pipeline.append(('persp', self.aug_settings.get('perspective_distortion', 0.1)))

    def _make_transform(self, mode):
        if mode == 'rgb':
            return transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        return transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.pairs)

    def _load(self, path, mode):
        return load_image_as_rgb(path) if mode == 'rgb' else load_image_as_grayscale(path)

    def __getitem__(self, idx):
        path_a, path_b = self.pairs[idx]
        try:
            pil_A = self._load(path_a, self.color_mode)
            pil_B = self._load(path_b, self.out_color_mode)
        except Exception as e:
            print(f"Error loading pair {path_a},{path_b}: {e}")
            C_A = 3 if self.color_mode == 'rgb' else 1
            C_B = 3 if self.out_color_mode == 'rgb' else 1
            return (torch.zeros(C_A, self.img_size, self.img_size),
                    torch.zeros(C_B, self.img_size, self.img_size))
        for name, param in self.aug_pipeline:
            seed = random.randint(0, 2**31)
            random.seed(seed); torch.manual_seed(seed)
            pil_A = self._apply_aug(pil_A, name, param)
            random.seed(seed); torch.manual_seed(seed)
            pil_B = self._apply_aug(pil_B, name, param)
        return self.transform_A(pil_A), self.transform_B(pil_B)

    def _apply_aug(self, pil, name, param):
        if name == 'flip_h':
            return F_vision.hflip(pil) if random.random() < 0.5 else pil
        if name == 'flip_v':
            return F_vision.vflip(pil) if random.random() < 0.5 else pil
        if name == 'rot':
            angle = random.uniform(-param, param)
            return F_vision.rotate(pil, angle, interpolation=Image.BICUBIC, fill=0)
        if name == 'crop':
            w, h = pil.size
            scale = random.uniform(param, 1.0)
            new_w, new_h = int(w * scale), int(h * scale)
            x = random.randint(0, w - new_w); y = random.randint(0, h - new_h)
            cropped = pil.crop((x, y, x + new_w, y + new_h))
            return cropped.resize((w, h), Image.BICUBIC)
        if name == 'persp':
            dist = param; w, h = pil.size
            start = np.float32([[0,0],[w,0],[w,h],[0,h]])
            d = dist * min(w, h)
            end = start + np.random.uniform(-d, d, size=(4, 2)).astype(np.float32)
            import cv2 as _cv2
            M = _cv2.getPerspectiveTransform(start, end)
            warped = _cv2.warpPerspective(np.array(pil), M, (w, h),
                                          borderMode=_cv2.BORDER_CONSTANT, borderValue=0)
            return Image.fromarray(warped)
        return pil

# ==================== Trainer (with cycle consistency) ====================

class Pix2PixTrainer:
    def __init__(self, model_ab, device, lr=2e-4, model_ba=None):
        self.device = device
        self.model_ab = model_ab.to(device)
        self.optimizer_ab = optim.Adam(self.model_ab.parameters(), lr=lr)
        self.model_ba = model_ba.to(device) if model_ba is not None else None
        self.optimizer_ba = optim.Adam(self.model_ba.parameters(), lr=lr) if self.model_ba is not None else None
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    @property
    def bidirectional(self):
        return self.model_ba is not None

    def _dir_loss(self, output, target, ssim_w, l1_w, l2_w):
        if output.shape[1] != target.shape[1]:
            if output.shape[1] == 1 and target.shape[1] == 3:
                output = output.repeat(1, 3, 1, 1)
            elif output.shape[1] == 3 and target.shape[1] == 1:
                output = output.mean(dim=1, keepdim=True)
        l1 = self.l1_loss(output, target)
        l2 = self.mse_loss(output, target)
        ssim_val = ssim_loss(output, target)
        loss = ssim_w * (1 - ssim_val) + l1_w * l1 + l2_w * l2
        return loss, l1.item(), ssim_val.item(), l2.item()

    def train_step(self, a, b, ssim_w=0.5, l1_w=0.5, l2_w=0.0,
                   direct_weight=1.0, cycle_weight=0.0):
        """
        a: input for A->B direction
        b: input for B->A direction (also target for A->B)
        direct_weight: weight for A->B and B->A losses
        cycle_weight: weight for ABA and BAB cycle consistency losses
        """
        a = a.to(self.device); b = b.to(self.device)

        # Zero grads up front
        self.optimizer_ab.zero_grad()
        if self.optimizer_ba is not None:
            self.optimizer_ba.zero_grad()

        # --- Direct pass ---
        out_ab = self.model_ab(a)
        loss_ab, l1_ab, ssim_ab, l2_ab = self._dir_loss(out_ab, b, ssim_w, l1_w, l2_w)

        if self.model_ba is not None:
            out_ba = self.model_ba(b)
            loss_ba, l1_ba, ssim_ba, l2_ba = self._dir_loss(out_ba, a, ssim_w, l1_w, l2_w)

            direct_loss = loss_ab + loss_ba

            # --- Cycle consistency ---
            cycle_loss = torch.zeros((), device=self.device)
            l1_cyc = ssim_cyc = l2_cyc = 0.0
            if cycle_weight > 0:
                # ABA: model_ba(model_ab(a)) ≈ a
                aba = self.model_ba(out_ab)
                loss_aba, l1_aba, ssim_aba, l2_aba = self._dir_loss(aba, a, ssim_w, l1_w, l2_w)
                # BAB: model_ab(model_ba(b)) ≈ b
                bab = self.model_ab(out_ba)
                loss_bab, l1_bab, ssim_bab, l2_bab = self._dir_loss(bab, b, ssim_w, l1_w, l2_w)
                cycle_loss = loss_aba + loss_bab
                l1_cyc = (l1_aba + l1_bab) / 2
                ssim_cyc = (ssim_aba + ssim_bab) / 2
                l2_cyc = (l2_aba + l2_bab) / 2

            total = direct_weight * direct_loss + cycle_weight * cycle_loss

            # Backprop once through the combined graph
            total.backward()

            self.optimizer_ab.step()
            self.optimizer_ba.step()

            # Report: average of direct metrics, and total loss
            avg_loss = total.item()
            return (avg_loss,
                    (l1_ab + l1_ba) / 2,
                    (ssim_ab + ssim_ba) / 2,
                    (l2_ab + l2_ba) / 2,
                    l1_cyc, ssim_cyc, l2_cyc)
        else:
            total = direct_weight * loss_ab
            total.backward()
            self.optimizer_ab.step()
            return (total.item(), l1_ab, ssim_ab, l2_ab, 0.0, 0.0, 0.0)

    @torch.no_grad()
    def generate(self, x, direction='ab'):
        x = x.to(self.device)
        if direction == 'ba':
            if self.model_ba is None:
                raise RuntimeError("Backward model not available (bidirectional disabled).")
            return self.model_ba(x)
        return self.model_ab(x)

    def save(self, path, in_channels_a, in_channels_b):
        state = {
            'model_ab': self.model_ab.state_dict(),
            'optimizer_ab': self.optimizer_ab.state_dict(),
            'in_channels_a': in_channels_a,
            'in_channels_b': in_channels_b,
            'bidirectional': self.bidirectional,
        }
        if self.bidirectional:
            state['model_ba'] = self.model_ba.state_dict()
            state['optimizer_ba'] = self.optimizer_ba.state_dict()
        torch.save(state, path)

    def load(self, path, map_location='cpu'):
        ckpt = torch.load(path, map_location=map_location)
        self.model_ab.load_state_dict(ckpt['model_ab'])
        self.optimizer_ab.load_state_dict(ckpt['optimizer_ab'])
        if ckpt.get('bidirectional', False) and self.model_ba is not None:
            if 'model_ba' in ckpt:
                self.model_ba.load_state_dict(ckpt['model_ba'])
                self.optimizer_ba.load_state_dict(ckpt['optimizer_ba'])
        return ckpt.get('in_channels_a', 3), ckpt.get('in_channels_b', 3), ckpt.get('bidirectional', False)

# ==================== Color Picker Popup ====================

class ColorPicker(tk.Toplevel):
    def __init__(self, parent, initial=(255, 255, 255), mode='rgb', on_ok=None):
        super().__init__(parent)
        self.title("Pick Color")
        self.mode = mode; self.on_ok = on_ok
        self.resizable(False, False)
        self.transient(parent); self.grab_set()
        if mode == 'rgb':
            self.initial = tuple(initial) if len(initial) == 3 else (255, 255, 255)
            init_r, init_g, init_b = self.initial
        else:
            self.initial = int(initial[0]) if isinstance(initial, (tuple, list)) else int(initial)
            init_r = init_g = init_b = self.initial
        self.var_r = tk.IntVar(value=init_r)
        self.var_g = tk.IntVar(value=init_g)
        self.var_b = tk.IntVar(value=init_b)
        self.swatch = tk.Canvas(self, width=120, height=60, highlightthickness=1, highlightbackground='gray')
        self.swatch.pack(padx=10, pady=10)
        if mode == 'rgb':
            self._make_slider("R", self.var_r)
            self._make_slider("G", self.var_g)
            self._make_slider("B", self.var_b)
        else:
            self._make_slider("Value", self.var_r)
        btn_frame = tk.Frame(self); btn_frame.pack(pady=8)
        tk.Button(btn_frame, text="OK", command=self._ok, width=8).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_frame, text="Cancel", command=self.destroy, width=8).pack(side=tk.LEFT, padx=4)
        self._update_swatch()

    def _make_slider(self, label, var):
        f = tk.Frame(self); f.pack(fill=tk.X, padx=10, pady=2)
        tk.Label(f, text=label, width=6, anchor='w').pack(side=tk.LEFT)
        tk.Scale(f, from_=0, to=255, orient=tk.HORIZONTAL, variable=var,
                 length=200, command=lambda _v: self._update_swatch()).pack(side=tk.LEFT)

    def _update_swatch(self):
        if self.mode == 'rgb':
            color = f'#{self.var_r.get():02x}{self.var_g.get():02x}{self.var_b.get():02x}'
        else:
            v = self.var_r.get(); color = f'#{v:02x}{v:02x}{v:02x}'
        self.swatch.delete("all")
        self.swatch.create_rectangle(0, 0, 120, 60, fill=color, outline='')

    def _ok(self):
        result = (self.var_r.get(), self.var_g.get(), self.var_b.get()) if self.mode == 'rgb' else self.var_r.get()
        if self.on_ok:
            self.on_ok(result)
        self.destroy()

# ==================== GUI Application ====================

class Pix2PixApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Paired Pix2Pix - Bidirectional + Cycle Consistency")
        sw = self.root.winfo_screenwidth(); sh = self.root.winfo_screenheight()
        self.root.geometry(f"{max(1100, int(sw*0.85))}x{max(750, int(sh*0.85))}")
        self.root.minsize(1000, 700)
        self.paths_A = []; self.paths_B = []; self.pairs = []
        self.training = False; self.trainer = None; self.current_epoch = 0
        self.message_queue_dataset = queue.Queue(); self.message_queue_train = queue.Queue()

        self.settings = {
            'img_size': tk.IntVar(value=64),
            'color_mode_A': tk.StringVar(value='rgb'),
            'color_mode_B': tk.StringVar(value='rgb'),
            'base_channels': tk.IntVar(value=64),
            'batch_size': tk.IntVar(value=8),
            'lr': tk.DoubleVar(value=2e-4),
            'use_attention': tk.BooleanVar(value=True),
            'padding_mode': tk.StringVar(value='constant'),
            'preview_enabled': tk.BooleanVar(value=True),
            'preview_epoch_freq': tk.IntVar(value=5),
            'ssim_weight': tk.DoubleVar(value=0.5),
            'l1_weight': tk.DoubleVar(value=0.5),
            'l2_weight': tk.DoubleVar(value=0.0),
            'ab_ba_weight': tk.DoubleVar(value=1.0),      # direct AB/BA loss weight
            'aba_bab_weight': tk.DoubleVar(value=0.0),    # cycle ABA/BAB loss weight
            'bidirectional': tk.BooleanVar(value=False),
            'pair_method': tk.StringVar(value='name'),
            'csv_path': tk.StringVar(value=''),
        }
        self.aug_settings = {
            'flip_horizontal': tk.BooleanVar(value=True),
            'flip_vertical': tk.BooleanVar(value=False),
            'rotation': tk.BooleanVar(value=False),
            'random_crop': tk.BooleanVar(value=False),
            'crop_scale': tk.DoubleVar(value=0.8),
            'random_perspective': tk.BooleanVar(value=False),
            'perspective_distortion': tk.DoubleVar(value=0.1),
        }
        self.thumbnail_size = 128
        self.display_size = 400
        self.draw_img = None; self.draw = None; self.photo_display = None
        self.last_x = None; self.last_y = None; self.is_erasing = False
        self.current_color = (255, 255, 255); self.current_gray = 255
        self.auto_update_id = None; self.auto_update_seconds = 0.0
        self.setup_gui()
        self.root.after(100, self.process_messages_dataset)
        self.root.after(100, self.process_messages_train)

    # ---------- GUI Setup ----------
    def setup_gui(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.dataset_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.dataset_tab, text='Dataset'); self.setup_dataset_tab()
        self.train_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.train_tab, text='Train'); self.setup_train_tab()
        self.settings_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.settings_tab, text='Settings'); self.setup_settings_tab()
        self.gen_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.gen_tab, text='Image → Image'); self.setup_gen_tab()
        self.status_label = tk.Label(self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    def setup_dataset_tab(self):
        main = tk.Frame(self.dataset_tab); main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        panels = tk.Frame(main); panels.pack(fill=tk.BOTH, expand=True)
        a_frame = tk.LabelFrame(panels, text="Dataset A (input)", padx=5, pady=5)
        a_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0,5))
        tk.Button(a_frame, text="Add Images", command=lambda: self.add_images('A')).pack(fill=tk.X, pady=2)
        tk.Button(a_frame, text="Add Folder", command=lambda: self.add_folder('A')).pack(fill=tk.X, pady=2)
        tk.Button(a_frame, text="Clear A", command=lambda: self.clear_side('A')).pack(fill=tk.X, pady=2)
        self.listbox_A = tk.Listbox(a_frame, height=8); self.listbox_A.pack(fill=tk.BOTH, expand=True, pady=2)
        b_frame = tk.LabelFrame(panels, text="Dataset B (target)", padx=5, pady=5)
        b_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5,0))
        tk.Button(b_frame, text="Add Images", command=lambda: self.add_images('B')).pack(fill=tk.X, pady=2)
        tk.Button(b_frame, text="Add Folder", command=lambda: self.add_folder('B')).pack(fill=tk.X, pady=2)
        tk.Button(b_frame, text="Clear B", command=lambda: self.clear_side('B')).pack(fill=tk.X, pady=2)
        self.listbox_B = tk.Listbox(b_frame, height=8); self.listbox_B.pack(fill=tk.BOTH, expand=True, pady=2)
        pair_frame = tk.LabelFrame(main, text="Pairing", padx=5, pady=5)
        pair_frame.pack(fill=tk.X, pady=(10,5))
        row1 = tk.Frame(pair_frame); row1.pack(fill=tk.X, pady=2)
        tk.Label(row1, text="Method:").pack(side=tk.LEFT)
        ttk.Combobox(row1, textvariable=self.settings['pair_method'],
                     values=['name', 'similarity', 'csv'], state='readonly', width=12).pack(side=tk.LEFT, padx=5)
        tk.Label(row1, text="CSV:").pack(side=tk.LEFT, padx=(15,0))
        tk.Entry(row1, textvariable=self.settings['csv_path'], width=40).pack(side=tk.LEFT, padx=2)
        tk.Button(row1, text="Browse", command=self.browse_csv).pack(side=tk.LEFT)
        row2 = tk.Frame(pair_frame); row2.pack(fill=tk.X, pady=2)
        tk.Button(row2, text="Build Pairs", command=self.build_pairs, bg='lightblue').pack(side=tk.LEFT, padx=2)
        self.pairs_label = tk.Label(row2, text="No pairs built", fg='blue')
        self.pairs_label.pack(side=tk.LEFT, padx=10)
        log_frame = tk.LabelFrame(main, text="Log", padx=5, pady=5)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.dataset_log_text = tk.Text(log_frame, height=8, font=("Courier", 9))
        self.dataset_log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = tk.Scrollbar(log_frame, command=self.dataset_log_text.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y); self.dataset_log_text.config(yscrollcommand=sb.set)

    def setup_train_tab(self):
        main = tk.Frame(self.train_tab); main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        left = tk.Frame(main, width=300); left.pack(side=tk.LEFT, fill=tk.Y, padx=(0,10)); left.pack_propagate(False)
        tk.Label(left, text="Paired Pix2Pix Training", font=("Arial",12,"bold")).pack(pady=(0,10))
        tk.Button(left, text="Initialize Model", command=self.init_model, width=20).pack(pady=5)
        ef = tk.Frame(left); ef.pack(pady=5)
        tk.Label(ef, text="Epochs:").pack(side=tk.LEFT)
        self.train_epoch_var = tk.StringVar(value="100")
        tk.Entry(ef, textvariable=self.train_epoch_var, width=8).pack(side=tk.LEFT, padx=5)
        tk.Button(left, text="Start Training", command=self.start_training, width=20, bg="lightgreen").pack(pady=5)
        tk.Button(left, text="Stop Training", command=self.stop_training, width=20, bg="salmon").pack(pady=5)
        tk.Button(left, text="Save Model", command=self.save_model, width=20).pack(pady=5)
        tk.Button(left, text="Load Model", command=self.load_model, width=20).pack(pady=5)
        right = tk.Frame(main); right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        prev_frame = tk.LabelFrame(right, text="Preview", padx=5, pady=5)
        prev_frame.pack(fill=tk.BOTH, expand=True, pady=(0,5))
        self.train_preview_canvas = tk.Canvas(prev_frame, bg='gray', width=512, height=384)
        self.train_preview_canvas.pack()
        log_frame = tk.LabelFrame(right, text="Log", padx=5, pady=5)
        log_frame.pack(fill=tk.BOTH, expand=True)
        self.train_log_text = tk.Text(log_frame, height=12, font=("Courier",9))
        self.train_log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = tk.Scrollbar(log_frame, command=self.train_log_text.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y); self.train_log_text.config(yscrollcommand=sb.set)

    def setup_settings_tab(self):
        main = tk.Frame(self.settings_tab); main.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        tk.Label(main, text="Model & Training Settings", font=("Arial",14,"bold")).pack(pady=(0,10))
        canvas = tk.Canvas(main)
        sb = tk.Scrollbar(main, orient="vertical", command=canvas.yview)
        scrollable = tk.Frame(canvas)
        scrollable.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0,0), window=scrollable, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)

        img_frame = tk.LabelFrame(scrollable, text="Image", padx=10, pady=10)
        img_frame.pack(fill=tk.X, pady=5)
        f = tk.Frame(img_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Color mode (A):", width=22, anchor='w').pack(side=tk.LEFT)
        ttk.Combobox(f, textvariable=self.settings['color_mode_A'],
                     values=['rgb','grayscale'], state='readonly', width=10).pack(side=tk.RIGHT)
        f = tk.Frame(img_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Color mode (B):", width=22, anchor='w').pack(side=tk.LEFT)
        ttk.Combobox(f, textvariable=self.settings['color_mode_B'],
                     values=['rgb','grayscale'], state='readonly', width=10).pack(side=tk.RIGHT)
        f = tk.Frame(img_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Image size:", width=22, anchor='w').pack(side=tk.LEFT)
        ttk.Spinbox(f, from_=16, to=512, textvariable=self.settings['img_size'], width=8).pack(side=tk.RIGHT)

        arch = tk.LabelFrame(scrollable, text="Architecture", padx=10, pady=10)
        arch.pack(fill=tk.X, pady=5)
        f = tk.Frame(arch); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="UNet base channels:", width=22, anchor='w').pack(side=tk.LEFT)
        ttk.Spinbox(f, from_=32, to=128, textvariable=self.settings['base_channels'], width=8).pack(side=tk.RIGHT)
        f = tk.Frame(arch); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Use Self-Attention:", width=22, anchor='w').pack(side=tk.LEFT)
        tk.Checkbutton(f, variable=self.settings['use_attention']).pack(side=tk.RIGHT)
        f = tk.Frame(arch); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Padding mode:", width=22, anchor='w').pack(side=tk.LEFT)
        ttk.Combobox(f, textvariable=self.settings['padding_mode'],
                     values=['constant','reflect','replicate','circular'],
                     state='readonly', width=10).pack(side=tk.RIGHT)
        f = tk.Frame(arch); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Bidirectional:", width=22, anchor='w').pack(side=tk.LEFT)
        tk.Checkbutton(f, variable=self.settings['bidirectional']).pack(side=tk.RIGHT)

        # Loss weights (expanded with AB/BA and ABA/BAB)
        loss = tk.LabelFrame(scrollable, text="Loss Weights", padx=10, pady=10)
        loss.pack(fill=tk.X, pady=5)
        for label, key in [("SSIM weight:", 'ssim_weight'),
                           ("L1 weight:", 'l1_weight'),
                           ("L2 weight:", 'l2_weight'),
                           ("AB/BA direct weight:", 'ab_ba_weight'),
                           ("ABA/BAB cycle weight:", 'aba_bab_weight')]:
            f = tk.Frame(loss); f.pack(fill=tk.X, pady=2)
            tk.Label(f, text=label, width=22, anchor='w').pack(side=tk.LEFT)
            tk.Entry(f, textvariable=self.settings[key], width=8).pack(side=tk.RIGHT)
        tk.Label(loss, text="(Cycle loss requires Bidirectional enabled.)",
                 fg='gray', font=("Arial", 8, "italic")).pack(anchor='w', pady=(5,0))

        tr = tk.LabelFrame(scrollable, text="Training", padx=10, pady=10)
        tr.pack(fill=tk.X, pady=5)
        f = tk.Frame(tr); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Batch size:", width=22, anchor='w').pack(side=tk.LEFT)
        ttk.Spinbox(f, from_=1, to=64, textvariable=self.settings['batch_size'], width=8).pack(side=tk.RIGHT)
        f = tk.Frame(tr); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Learning rate:", width=22, anchor='w').pack(side=tk.LEFT)
        tk.Entry(f, textvariable=self.settings['lr'], width=8).pack(side=tk.RIGHT)

        pv = tk.LabelFrame(scrollable, text="Preview", padx=10, pady=10)
        pv.pack(fill=tk.X, pady=5)
        f = tk.Frame(pv); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Enable preview:", width=22, anchor='w').pack(side=tk.LEFT)
        tk.Checkbutton(f, variable=self.settings['preview_enabled']).pack(side=tk.RIGHT)
        f = tk.Frame(pv); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Every N epochs:", width=22, anchor='w').pack(side=tk.LEFT)
        ttk.Spinbox(f, from_=1, to=100, textvariable=self.settings['preview_epoch_freq'], width=8).pack(side=tk.RIGHT)

        aug = tk.LabelFrame(scrollable, text="Paired Augmentations", padx=10, pady=10)
        aug.pack(fill=tk.X, pady=5)
        f1 = tk.Frame(aug); f1.pack(fill=tk.X, pady=2)
        tk.Checkbutton(f1, text="Horizontal Flip", variable=self.aug_settings['flip_horizontal']).pack(side=tk.LEFT, padx=5)
        tk.Checkbutton(f1, text="Vertical Flip", variable=self.aug_settings['flip_vertical']).pack(side=tk.LEFT, padx=5)
        tk.Checkbutton(f1, text="Rotation (±30°)", variable=self.aug_settings['rotation']).pack(side=tk.LEFT, padx=5)
        f2 = tk.Frame(aug); f2.pack(fill=tk.X, pady=2)
        tk.Checkbutton(f2, text="Random Crop", variable=self.aug_settings['random_crop']).pack(side=tk.LEFT, padx=5)
        tk.Label(f2, text="scale:").pack(side=tk.LEFT)
        tk.Entry(f2, textvariable=self.aug_settings['crop_scale'], width=5).pack(side=tk.LEFT)
        tk.Checkbutton(f2, text="Random Perspective", variable=self.aug_settings['random_perspective']).pack(side=tk.LEFT, padx=15)
        tk.Label(f2, text="distortion:").pack(side=tk.LEFT)
        tk.Entry(f2, textvariable=self.aug_settings['perspective_distortion'], width=5).pack(side=tk.LEFT)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

    def setup_gen_tab(self):
        main = tk.Frame(self.gen_tab); main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        top = tk.Frame(main); top.pack(fill=tk.X, pady=5)
        draw_frame = tk.Frame(top); draw_frame.pack(side=tk.LEFT, padx=(0,10))
        self.draw_canvas = tk.Canvas(draw_frame, width=self.display_size,
                                     height=self.display_size, bg='black')
        self.draw_canvas.pack()
        self._init_draw_image()
        controls = tk.Frame(draw_frame); controls.pack(pady=5, fill=tk.X)
        row1 = tk.Frame(controls); row1.pack(fill=tk.X, pady=2)
        tk.Button(row1, text="Clear", command=self.clear_draw).pack(side=tk.LEFT, padx=2)
        tk.Button(row1, text="Load Image as Input", command=self.load_image_as_input).pack(side=tk.LEFT, padx=2)
        self.color_btn = tk.Button(row1, text="Color...", command=self.open_color_picker,
                                    bg='white', fg='black', width=10)
        self.color_btn.pack(side=tk.LEFT, padx=2)
        tk.Button(row1, text="Generate", command=self.manual_generate, bg="lightgreen").pack(side=tk.LEFT, padx=2)
        row2 = tk.Frame(controls); row2.pack(fill=tk.X, pady=2)
        tk.Label(row2, text="Brush:").pack(side=tk.LEFT)
        self.brush_size_var = tk.IntVar(value=4)
        tk.Spinbox(row2, from_=1, to=40, textvariable=self.brush_size_var, width=4).pack(side=tk.LEFT, padx=2)
        tk.Label(row2, text="Direction:").pack(side=tk.LEFT, padx=(10,0))
        self.direction_var = tk.StringVar(value='ab')
        self.direction_combo = ttk.Combobox(row2, textvariable=self.direction_var,
                                             values=['ab', 'ba'], state='readonly', width=5)
        self.direction_combo.pack(side=tk.LEFT)
        tk.Label(row2, text="Auto-update (s):").pack(side=tk.LEFT, padx=(10,0))
        self.auto_update_var = tk.DoubleVar(value=0.0)
        tk.Entry(row2, textvariable=self.auto_update_var, width=5).pack(side=tk.LEFT, padx=2)
        tk.Button(row2, text="Start", command=self.start_auto_update).pack(side=tk.LEFT, padx=1)
        tk.Button(row2, text="Stop", command=self.stop_auto_update).pack(side=tk.LEFT, padx=1)
        self.output_label = tk.Label(top, text="Output", bg='lightgray', width=30, height=15)
        self.output_label.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.draw_canvas.bind("<Button-1>", self.start_paint)
        self.draw_canvas.bind("<B1-Motion>", self.paint)
        self.draw_canvas.bind("<ButtonRelease-1>", self.stop_paint)
        self.draw_canvas.bind("<Button-3>", self.start_erase)
        self.draw_canvas.bind("<B3-Motion>", self.erase)
        self.draw_canvas.bind("<ButtonRelease-3>", self.stop_erase)
        self.gen_info = tk.Label(main, text="", fg="blue"); self.gen_info.pack()
        self.update_display()

    # ---------- Drawing ----------
    def _init_draw_image(self):
        img_size = self.settings['img_size'].get()
        mode = 'RGB' if self.settings['color_mode_A'].get() == 'rgb' else 'L'
        bg = (0,0,0) if mode == 'RGB' else 0
        self.draw_img = Image.new(mode, (img_size, img_size), bg)
        self.draw = ImageDraw.Draw(self.draw_img)
        self.update_display()

    def _current_fill(self):
        if self.settings['color_mode_A'].get() == 'rgb':
            return tuple(self.current_color)
        return int(self.current_gray)

    def _current_bg(self):
        return (0,0,0) if self.settings['color_mode_A'].get() == 'rgb' else 0

    def update_display(self):
        if self.draw_img is None: return
        display = self.draw_img.resize((self.display_size, self.display_size), Image.NEAREST)
        if display.mode != 'RGB': display = display.convert('RGB')
        self.photo_display = ImageTk.PhotoImage(display)
        self.draw_canvas.delete("all")
        self.draw_canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_display)
        self.draw_canvas.image = self.photo_display

    def _map_coords(self, event):
        img_size = self.settings['img_size'].get()
        x = int(event.x * img_size / self.display_size); y = int(event.y * img_size / self.display_size)
        return max(0, min(img_size-1, x)), max(0, min(img_size-1, y))

    def start_paint(self, event):
        self.last_x, self.last_y = self._map_coords(event); self.is_erasing = False
    def paint(self, event):
        if self.last_x is None or self.last_y is None: return
        x, y = self._map_coords(event); size = self.brush_size_var.get()
        self.draw.line([self.last_x, self.last_y, x, y], fill=self._current_fill(), width=size)
        self.last_x, self.last_y = x, y; self.update_display(); self.auto_generate()
    def stop_paint(self, event):
        self.last_x = None; self.last_y = None
    def start_erase(self, event):
        self.last_x, self.last_y = self._map_coords(event); self.is_erasing = True
    def erase(self, event):
        if not self.is_erasing or self.last_x is None or self.last_y is None: return
        x, y = self._map_coords(event); size = self.brush_size_var.get()
        self.draw.line([self.last_x, self.last_y, x, y], fill=self._current_bg(), width=size)
        self.last_x, self.last_y = x, y; self.update_display(); self.auto_generate()
    def stop_erase(self, event):
        self.last_x = None; self.last_y = None; self.is_erasing = False
    def clear_draw(self):
        self._init_draw_image(); self.auto_generate()

    def open_color_picker(self):
        mode = self.settings['color_mode_A'].get()
        initial = tuple(self.current_color) if mode == 'rgb' else int(self.current_gray)
        def on_ok(color):
            if mode == 'rgb':
                self.current_color = tuple(color)
                self.color_btn.config(bg=f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}',
                                       fg='black' if sum(color) > 384 else 'white')
            else:
                self.current_gray = int(color)
                self.color_btn.config(bg=f'#{color:02x}{color:02x}{color:02x}',
                                       fg='black' if color > 128 else 'white')
        ColorPicker(self.root, initial=initial, mode=mode, on_ok=on_ok)

    def load_image_as_input(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp *.jfif")])
        if not path: return
        try:
            img_size = self.settings['img_size'].get()
            # Load using current direction's input color mode
            direction = self.direction_var.get() if self.settings['bidirectional'].get() else 'ab'
            in_mode = self.settings['color_mode_A'].get() if direction == 'ab' else self.settings['color_mode_B'].get()
            if in_mode == 'rgb':
                pil = load_image_as_rgb(path).resize((img_size, img_size), Image.BICUBIC)
            else:
                pil = load_image_as_grayscale(path).resize((img_size, img_size), Image.BICUBIC)
            self.draw_img = pil
            self.draw = ImageDraw.Draw(self.draw_img)
            self.update_display()
            self.gen_info.config(text=f"Loaded input: {os.path.basename(path)}")
            self.manual_generate()
        except Exception as e:
            self.gen_info.config(text=f"Error: {e}")

    def auto_generate(self):
        if self.auto_update_seconds > 0:
            if self.auto_update_id is not None:
                self.root.after_cancel(self.auto_update_id)
            self.auto_update_id = self.root.after(int(self.auto_update_seconds*1000), self.manual_generate)
    def start_auto_update(self):
        try:
            sec = float(self.auto_update_var.get())
            if sec <= 0:
                self.gen_info.config(text="Interval must be > 0"); return
            self.auto_update_seconds = sec
            self.gen_info.config(text=f"Auto-update every {sec}s")
            self.manual_generate()
            if self.auto_update_id is not None:
                self.root.after_cancel(self.auto_update_id)
            self.auto_update_id = self.root.after(int(sec*1000), self._auto_update_loop)
        except ValueError:
            self.gen_info.config(text="Invalid interval")
    def _auto_update_loop(self):
        if self.auto_update_seconds > 0:
            self.manual_generate()
            self.auto_update_id = self.root.after(int(self.auto_update_seconds*1000), self._auto_update_loop)
        else:
            self.auto_update_id = None
    def stop_auto_update(self):
        self.auto_update_seconds = 0
        if self.auto_update_id is not None:
            self.root.after_cancel(self.auto_update_id); self.auto_update_id = None
        self.gen_info.config(text="Auto-update stopped")

    def manual_generate(self):
        if self.trainer is None:
            self.gen_info.config(text="Model not loaded"); return
        img_size = self.settings['img_size'].get()
        if self.draw_img.size != (img_size, img_size):
            self.draw_img = self.draw_img.resize((img_size, img_size), Image.BICUBIC)
            self.draw = ImageDraw.Draw(self.draw_img)
        direction = self.direction_var.get() if self.settings['bidirectional'].get() else 'ab'
        in_mode = self.settings['color_mode_A'].get() if direction == 'ab' else self.settings['color_mode_B'].get()
        if in_mode == 'rgb':
            work = self.draw_img.convert('RGB') if self.draw_img.mode != 'RGB' else self.draw_img
            t = transforms.ToTensor()(work).unsqueeze(0) * 2 - 1
        else:
            work = self.draw_img.convert('L') if self.draw_img.mode != 'L' else self.draw_img
            t = transforms.ToTensor()(work).unsqueeze(0) * 2 - 1
        try:
            with torch.no_grad():
                output = self.trainer.generate(t, direction=direction)
        except Exception as e:
            self.gen_info.config(text=f"Generate error: {e}"); return
        output = ((output + 1) / 2).clamp(0, 1).cpu()
        if output.shape[1] == 3:
            out_img = transforms.ToPILImage()(output.squeeze(0))
        else:
            out_img = transforms.ToPILImage()(output.squeeze(0).repeat(3,1,1))
        out_img = out_img.resize((self.display_size, self.display_size), Image.NEAREST)
        self.output_photo = ImageTk.PhotoImage(out_img)
        self.output_label.config(image=self.output_photo, text="")
        self.output_label.image = self.output_photo
        self.gen_info.config(text=f"Generated ({direction.upper()})")

    # ---------- Logging ----------
    def log_dataset(self, msg): self.message_queue_dataset.put(msg)
    def log_train(self, msg): self.message_queue_train.put(msg)
    def process_messages_dataset(self):
        try:
            while True:
                msg = self.message_queue_dataset.get_nowait()
                self.dataset_log_text.insert(tk.END, f"{time.strftime('%H:%M:%S')} - {msg}\n")
                self.dataset_log_text.see(tk.END); self.status_label.config(text=msg[:60])
        except queue.Empty: pass
        self.root.after(100, self.process_messages_dataset)
    def process_messages_train(self):
        try:
            while True:
                msg = self.message_queue_train.get_nowait()
                self.train_log_text.insert(tk.END, f"{time.strftime('%H:%M:%S')} - {msg}\n")
                self.train_log_text.see(tk.END); self.status_label.config(text=msg[:60])
        except queue.Empty: pass
        self.root.after(100, self.process_messages_train)

    # ---------- Dataset actions ----------
    def add_images(self, side):
        files = filedialog.askopenfilenames(filetypes=[("Images", "*.jpg *.jpeg *.png *.jfif *.webp *.bmp")])
        target_paths = self.paths_A if side == 'A' else self.paths_B
        target_list = self.listbox_A if side == 'A' else self.listbox_B
        for f in files:
            if f not in target_paths:
                target_paths.append(f); target_list.insert(tk.END, os.path.basename(f))
        self.log_dataset(f"Added {len(files)} images to {side}. Total {side}: {len(target_paths)}")
    def add_folder(self, side):
        folder = filedialog.askdirectory()
        if not folder: return
        paths = scan_folder(folder)
        target_paths = self.paths_A if side == 'A' else self.paths_B
        target_list = self.listbox_A if side == 'A' else self.listbox_B
        count = 0
        for p in paths:
            if p not in target_paths:
                target_paths.append(p); target_list.insert(tk.END, os.path.basename(p)); count += 1
        self.log_dataset(f"Added {count} images to {side} from folder. Total {side}: {len(target_paths)}")
    def clear_side(self, side):
        if side == 'A':
            self.paths_A = []; self.listbox_A.delete(0, tk.END)
        else:
            self.paths_B = []; self.listbox_B.delete(0, tk.END)
        self.log_dataset(f"Cleared {side}")
    def browse_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if path:
            self.settings['csv_path'].set(path); self.settings['pair_method'].set('csv')

    def build_pairs(self):
        if not self.paths_A or not self.paths_B:
            self.log_dataset("Both datasets A and B must be loaded."); return
        method = self.settings['pair_method'].get()
        self.log_dataset(f"Building pairs using method: {method}...")
        def worker():
            try:
                if method == 'name':
                    pairs = pair_by_name(self.paths_A, self.paths_B)
                elif method == 'csv':
                    csv_path = self.settings['csv_path'].get()
                    if not csv_path or not os.path.exists(csv_path):
                        self.log_dataset("CSV path invalid."); return
                    pairs = pair_by_csv(csv_path, self.paths_A, self.paths_B)
                elif method == 'similarity':
                    pairs = pair_by_similarity(self.paths_A, self.paths_B, log_fn=self.log_dataset)
                else:
                    self.log_dataset("Unknown pairing method."); return
                self.pairs = pairs
                self.root.after(0, lambda: self.pairs_label.config(
                    text=f"{len(pairs)} pairs built", fg='green'))
                self.log_dataset(f"Pairing done. {len(pairs)} pairs.")
                for i, (a, b) in enumerate(pairs[:5]):
                    self.log_dataset(f"  {i+1}. {os.path.basename(a)} ↔ {os.path.basename(b)}")
            except Exception as e:
                self.log_dataset(f"Pairing error: {e}")
        threading.Thread(target=worker, daemon=True).start()

    # ---------- Model / Training ----------
    def init_model(self):
        try:
            in_ch_a = 3 if self.settings['color_mode_A'].get() == 'rgb' else 1
            in_ch_b = 3 if self.settings['color_mode_B'].get() == 'rgb' else 1
            img_size = self.settings['img_size'].get()
            use_attn = self.settings['use_attention'].get()
            pad = self.settings['padding_mode'].get()
            base = self.settings['base_channels'].get()
            bidirectional = self.settings['bidirectional'].get()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_ab = Pix2PixUNet(in_channels=in_ch_a, out_channels=in_ch_b,
                                   base_channels=base, img_size=img_size,
                                   channel_mult=(1,2,3,4), dropout=0.1,
                                   use_attention=use_attn, padding_mode=pad)
            model_ba = None
            if bidirectional:
                model_ba = Pix2PixUNet(in_channels=in_ch_b, out_channels=in_ch_a,
                                       base_channels=base, img_size=img_size,
                                       channel_mult=(1,2,3,4), dropout=0.1,
                                       use_attention=use_attn, padding_mode=pad)
            self.trainer = Pix2PixTrainer(model_ab, device,
                                          lr=self.settings['lr'].get(),
                                          model_ba=model_ba)
            n_ab = sum(p.numel() for p in model_ab.parameters())
            msg = f"Model initialized. A→B: {n_ab:,} params"
            if bidirectional:
                n_ba = sum(p.numel() for p in model_ba.parameters())
                msg += f" | B→A: {n_ba:,} params"
            self.log_train(msg)
            self._init_draw_image()
        except Exception as e:
            self.log_train(f"Init error: {e}")

    def start_training(self):
        if not self.pairs:
            self.log_train("No pairs built. Build pairs first."); return
        if self.trainer is None:
            self.log_train("Model not initialized!"); return
        if self.training:
            self.log_train("Already training."); return
        try:
            epochs = int(self.train_epoch_var.get())
        except Exception:
            self.log_train("Invalid epochs"); return
        self.training = True
        self.current_epoch = 0
        self.train_start_time = time.time()
        threading.Thread(target=self.train_loop, args=(epochs,), daemon=True).start()
        self.log_train(f"Training started for {epochs} epochs.")

    def train_loop(self, epochs):
        try:
            img_size = self.settings['img_size'].get()
            color_mode_A = self.settings['color_mode_A'].get()
            color_mode_B = self.settings['color_mode_B'].get()
            batch_size = self.settings['batch_size'].get()
            lr = self.settings['lr'].get()
            preview_enabled = self.settings['preview_enabled'].get()
            preview_freq = self.settings['preview_epoch_freq'].get()
            aug_dict = {k: v.get() for k, v in self.aug_settings.items()}
            ssim_w = self.settings['ssim_weight'].get()
            l1_w = self.settings['l1_weight'].get()
            l2_w = self.settings['l2_weight'].get()
            ab_ba_w = self.settings['ab_ba_weight'].get()
            aba_bab_w = self.settings['aba_bab_weight'].get()

            if aba_bab_w > 0 and not self.trainer.bidirectional:
                self.log_train("WARNING: ABA/BAB cycle weight > 0 but Bidirectional is OFF. Cycle loss will be ignored.")

            dataset = PairedPix2PixDataset(self.pairs, img_size=img_size,
                                           color_mode=color_mode_A,
                                           out_color_mode=color_mode_B,
                                           aug_settings=aug_dict)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                num_workers=0, pin_memory=False)

            for epoch in range(epochs):
                if not self.training: break
                self.current_epoch = epoch

                new_lr = self.settings['lr'].get()
                if new_lr != lr:
                    for pg in self.trainer.optimizer_ab.param_groups: pg['lr'] = new_lr
                    if self.trainer.optimizer_ba is not None:
                        for pg in self.trainer.optimizer_ba.param_groups: pg['lr'] = new_lr
                    lr = new_lr
                    self.log_train(f"LR updated to {lr}")

                new_ssim = self.settings['ssim_weight'].get()
                new_l1 = self.settings['l1_weight'].get()
                new_l2 = self.settings['l2_weight'].get()
                new_ab_ba = self.settings['ab_ba_weight'].get()
                new_aba_bab = self.settings['aba_bab_weight'].get()
                if (new_ssim, new_l1, new_l2, new_ab_ba, new_aba_bab) != \
                   (ssim_w, l1_w, l2_w, ab_ba_w, aba_bab_w):
                    ssim_w, l1_w, l2_w = new_ssim, new_l1, new_l2
                    ab_ba_w, aba_bab_w = new_ab_ba, new_aba_bab
                    self.log_train(f"Loss weights updated: SSIM={ssim_w} L1={l1_w} L2={l2_w} "
                                   f"AB/BA={ab_ba_w} ABA/BAB={aba_bab_w}")

                epoch_loss = epoch_l1 = epoch_ssim = epoch_l2 = 0.0
                epoch_l1c = epoch_ssimc = epoch_l2c = 0.0
                batches = 0
                for a, b in loader:
                    if not self.training: break
                    loss, l1, ssim, l2, l1c, ssimc, l2c = self.trainer.train_step(
                        a, b, ssim_w, l1_w, l2_w,
                        direct_weight=ab_ba_w, cycle_weight=aba_bab_w)
                    epoch_loss += loss; epoch_l1 += l1; epoch_ssim += ssim; epoch_l2 += l2
                    epoch_l1c += l1c; epoch_ssimc += ssimc; epoch_l2c += l2c
                    batches += 1
                if batches == 0: continue
                avg_loss = epoch_loss/batches; avg_l1 = epoch_l1/batches
                avg_ssim = epoch_ssim/batches; avg_l2 = epoch_l2/batches
                avg_l1c = epoch_l1c/batches; avg_ssimc = epoch_ssimc/batches
                avg_l2c = epoch_l2c/batches
                elapsed = time.time() - self.train_start_time
                msg = (f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | "
                       f"L1: {avg_l1:.4f} | SSIM: {avg_ssim:.4f} | L2: {avg_l2:.4f}")
                if self.trainer.bidirectional and aba_bab_w > 0:
                    msg += (f" | CycleL1: {avg_l1c:.4f} | CycleSSIM: {avg_ssimc:.4f} | "
                            f"CycleL2: {avg_l2c:.4f}")
                msg += f" | Time: {elapsed:.1f}s"
                self.log_train(msg)

                if preview_enabled and (epoch+1) % preview_freq == 0:
                    self.show_train_preview(loader)

            self.training = False
            self.log_train("Training finished.")
        except Exception as e:
            self.log_train(f"Training error: {e}")
            import traceback; traceback.print_exc()
            self.training = False

    def show_train_preview(self, loader):
        """
        Layout:
          - Bidirectional OFF: 4 columns.
              Row 0: A inputs (cols 0-3)
              Row 1: A->B outputs
              Row 2: B targets
          - Bidirectional ON: 4 columns split into two halves.
              Cols 0-1 = A->B direction:  Row 0 = A,  Row 1 = out_ab,  Row 2 = B target
              Cols 2-3 = B->A direction:  Row 0 = B,  Row 1 = out_ba,  Row 2 = A target
        """
        try:
            self.trainer.model_ab.eval()
            bidir = self.trainer.model_ba is not None
            if bidir:
                self.trainer.model_ba.eval()
            with torch.no_grad():
                a, b = next(iter(loader))
                if bidir:
                    n = min(2, len(a))
                    a = a[:n].to(self.trainer.device)
                    b = b[:n].to(self.trainer.device)
                    out_ab = self.trainer.generate(a, direction='ab')
                    out_ba = self.trainer.generate(b, direction='ba')
                else:
                    n = min(4, len(a))
                    a = a[:n].to(self.trainer.device)
                    b = b[:n].to(self.trainer.device)
                    out_ab = self.trainer.generate(a, direction='ab')
                    out_ba = None
            self.trainer.model_ab.train()
            if bidir:
                self.trainer.model_ba.train()

            thumb = 96
            cols = 4
            rows = 3
            W = cols * thumb
            H = rows * thumb
            grid = Image.new('RGB', (W, H), color=(128,128,128))

            def to_pil(t):
                t = ((t.cpu() + 1) / 2).clamp(0, 1)
                if t.shape[0] == 1:
                    t = t.repeat(3, 1, 1)
                return transforms.ToPILImage()(t).resize((thumb, thumb), Image.NEAREST)

            if bidir:
                # Cols 0-1: A -> B, Cols 2-3: B -> A
                for i in range(n):
                    # Row 0 — inputs
                    grid.paste(to_pil(a[i]), (i*thumb, 0))
                    grid.paste(to_pil(b[i]), ((i+2)*thumb, 0))
                    # Row 1 — outputs
                    grid.paste(to_pil(out_ab[i]), (i*thumb, thumb))
                    grid.paste(to_pil(out_ba[i]), ((i+2)*thumb, thumb))
                    # Row 2 — targets
                    grid.paste(to_pil(b[i]), (i*thumb, 2*thumb))          # target for A->B is B
                    grid.paste(to_pil(a[i]), ((i+2)*thumb, 2*thumb))      # target for B->A is A
                # Red separator between the two direction halves
                sep = ImageDraw.Draw(grid)
                sep.line([(2*thumb, 0), (2*thumb, H)], fill=(255, 0, 0), width=2)
            else:
                for i in range(n):
                    grid.paste(to_pil(a[i]), (i*thumb, 0))
                    grid.paste(to_pil(out_ab[i]), (i*thumb, thumb))
                    grid.paste(to_pil(b[i]), (i*thumb, 2*thumb))

            disp_w = 512
            disp_h = int(disp_w * (rows/cols))
            grid = grid.resize((disp_w, disp_h), Image.NEAREST)
            self.train_preview_photo = ImageTk.PhotoImage(grid)
            self.train_preview_canvas.delete("all")
            self.train_preview_canvas.config(width=disp_w, height=disp_h)
            self.train_preview_canvas.create_image(disp_w//2, disp_h//2, image=self.train_preview_photo)
            self.train_preview_canvas.image = self.train_preview_photo
        except Exception as e:
            self.log_train(f"Preview error: {e}")

    def stop_training(self):
        self.training = False
        self.log_train("Training stopped.")

    def save_model(self):
        if self.trainer is None:
            self.log_train("No model."); return
        fname = filedialog.asksaveasfilename(defaultextension=".pth",
                                             filetypes=[("PyTorch","*.pth")])
        if fname:
            in_a = 3 if self.settings['color_mode_A'].get() == 'rgb' else 1
            in_b = 3 if self.settings['color_mode_B'].get() == 'rgb' else 1
            self.trainer.save(fname, in_a, in_b)
            self.log_train(f"Model saved to {fname}")

    def load_model(self):
        fname = filedialog.askopenfilename(filetypes=[("PyTorch","*.pth")])
        if not fname: return
        try:
            ckpt = torch.load(fname, map_location='cpu')
            in_a = ckpt.get('in_channels_a', 3)
            in_b = ckpt.get('in_channels_b', 3)
            bidir = ckpt.get('bidirectional', False)
            self.settings['color_mode_A'].set('rgb' if in_a == 3 else 'grayscale')
            self.settings['color_mode_B'].set('rgb' if in_b == 3 else 'grayscale')
            self.settings['bidirectional'].set(bool(bidir))
            self.init_model()
            loaded_a, loaded_b, _ = self.trainer.load(fname, map_location='cpu')
            self.trainer.model_ab.to(self.trainer.device)
            if self.trainer.model_ba is not None:
                self.trainer.model_ba.to(self.trainer.device)
            self.log_train(f"Model loaded from {fname} (A→B:{loaded_a}ch, B→A:{loaded_b}ch, bidir={bidir})")
        except Exception as e:
            self.log_train(f"Load error: {e}")


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    root = tk.Tk()
    app = Pix2PixApp(root)
    root.mainloop()