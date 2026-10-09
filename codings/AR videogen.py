"""
Video Autoregressive Generator (v5)
- Autoregressive conv preview (feeds predictions back in)
- Red border only on source frame
- Dataset preview auto-fits tiles
- AE preview fills 4x8 grid
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageTk
import os
import threading
import queue
import numpy as np
import math
import random
import time
import io
import traceback

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

print(f"cv2: {HAS_CV2}  imageio: {HAS_IMAGEIO}")


def load_image_as_rgb(path):
    img = Image.open(path)
    if img.mode == 'RGBA':
        bg = Image.new('RGB', img.size, (0, 0, 0))
        bg.paste(img, mask=img.split()[3])
        return bg
    return img.convert('RGB')


def compute_layers_for_rf(target_rf, kernel_size):
    if kernel_size < 2:
        raise ValueError("kernel_size must be >= 2")
    if target_rf < 2:
        target_rf = 2
    inner = (target_rf - 1) / (kernel_size - 1) + 1
    n = max(1, math.ceil(math.log2(inner)))
    actual_rf = 1 + (kernel_size - 1) * (2 ** n - 1)
    return n, actual_rf


def snap_to_odd(n):
    n = int(n)
    if n < 1:
        return 1
    if n % 2 == 0:
        return n - 1 if n > 1 else 1
    return n


def extract_frames_from_video(path, target_size=32, stride=1, max_frames=None):
    frames = []
    if HAS_CV2:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open {path}")
        idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if idx % stride == 0:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil = Image.fromarray(frame).resize((target_size, target_size),
                                                        Image.LANCZOS)
                    frames.append(pil)
                    if max_frames and len(frames) >= max_frames:
                        break
                idx += 1
        finally:
            cap.release()
        return frames
    elif HAS_IMAGEIO:
        reader = imageio.get_reader(path)
        try:
            for idx, frame in enumerate(reader):
                if idx % stride == 0:
                    pil = Image.fromarray(frame).resize((target_size, target_size),
                                                        Image.LANCZOS)
                    frames.append(pil)
                    if max_frames and len(frames) >= max_frames:
                        break
        finally:
            reader.close()
        return frames
    else:
        raise RuntimeError("Neither cv2 nor imageio is available.")


def save_frames_as_gif(frames, path, fps=10):
    duration = max(1, int(round(1000 / max(1, fps))))
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=duration, loop=0, optimize=False)


def save_frames_as_mp4(frames, path, fps=10):
    if not HAS_IMAGEIO:
        raise RuntimeError("imageio not available for mp4 export")
    arrs = [np.array(f.convert('RGB')) for f in frames]
    imageio.mimsave(path, arrs, fps=fps)


def save_frames_as_png_sequence(frames, folder, prefix="frame"):
    os.makedirs(folder, exist_ok=True)
    for i, f in enumerate(frames):
        f.save(os.path.join(folder, f"{prefix}_{i:05d}.png"))


def fit_to_cell(pil, cell):
    w, h = pil.size
    if w == cell and h == cell:
        return pil
    scale = min(cell / max(1, w), cell / max(1, h))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = pil.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new('RGB', (cell, cell), (20, 20, 20))
    canvas.paste(resized, ((cell - nw) // 2, (cell - nh) // 2))
    return canvas


VAE_SIZE_CONFIGS = {
    'tiny':   [0.25, 0.5],
    'small':  [0.25, 0.5, 1.0],
    'medium': [0.5, 1.0, 2.0],
    'big':    [1.0, 2.0, 4.0],
    'large':  [2.0, 4.0, 8.0],
}


def _vae_num_blocks(img_size, latent_h, latent_w):
    if img_size % latent_h or img_size % latent_w:
        raise ValueError(f"img_size={img_size} not divisible by latent {latent_h}x{latent_w}")
    rh, rw = img_size // latent_h, img_size // latent_w
    if rh != rw:
        raise ValueError(f"non-square downsample ratio {rh}:{rw}")
    n = int(round(math.log2(rh)))
    if 2 ** n != rh:
        raise ValueError(f"ratio {rh} not a power of 2")
    if n < 1:
        raise ValueError(f"need >=1 down block, got ratio {rh}")
    return n


def _fit_mults(mults, n):
    if len(mults) == n:
        return list(mults)
    if len(mults) > n:
        return list(mults[:n])
    ext = list(mults)
    while len(ext) < n:
        ext.append(ext[-1] * 2)
    return ext


class FlexEncoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=16, latent_channels=32,
                 latent_h=4, latent_w=4, size='medium', img_size=32):
        super().__init__()
        mults = VAE_SIZE_CONFIGS[size]
        n = _vae_num_blocks(img_size, latent_h, latent_w)
        mults = _fit_mults(mults, n)

        self.down = nn.ModuleList()
        c = in_channels
        for m in mults:
            oc = max(1, int(base_channels * m))
            self.down.append(nn.Conv2d(c, oc, 3, stride=2, padding=1))
            self.down.append(nn.LeakyReLU(0.2))
            c = oc
        self.conv_out = nn.Conv2d(c, latent_channels, 3, padding=1)

    def forward(self, x):
        for l in self.down:
            x = l(x)
        return self.conv_out(x)


class FlexDecoder(nn.Module):
    def __init__(self, latent_channels=32, base_channels=16, out_channels=3,
                 latent_h=4, latent_w=4, size='medium', img_size=32):
        super().__init__()
        mults = VAE_SIZE_CONFIGS[size]
        n = _vae_num_blocks(img_size, latent_h, latent_w)
        mults = _fit_mults(mults, n)
        widths = [max(1, int(base_channels * m)) for m in mults]

        high = widths[-1]
        self.init_conv = nn.Conv2d(latent_channels, high, 3, padding=1)
        self.act = nn.LeakyReLU(0.2)
        self.up = nn.ModuleList()
        c = high
        for i in range(n):
            oc = widths[n - 2 - i] if i < n - 1 else widths[0]
            self.up.append(nn.ConvTranspose2d(c, oc, 4, stride=2, padding=1))
            self.up.append(nn.LeakyReLU(0.2))
            c = oc
        self.conv_out = nn.Conv2d(c, out_channels, 3, padding=1)

    def forward(self, z, target_size=None):
        x = self.act(self.init_conv(z))
        for l in self.up:
            x = l(x)
        out = torch.tanh(self.conv_out(x))
        if target_size and (out.shape[2] != target_size[0] or out.shape[3] != target_size[1]):
            out = F.interpolate(out, size=target_size, mode='bilinear',
                                align_corners=False)
        return out


class PureAE(nn.Module):
    def __init__(self, in_channels=3, base_channels=16, latent_channels=32,
                 latent_h=4, latent_w=4, size='medium', img_size=32):
        super().__init__()
        self.encoder = FlexEncoder(in_channels, base_channels, latent_channels,
                                   latent_h, latent_w, size, img_size)
        self.decoder = FlexDecoder(latent_channels, base_channels, in_channels,
                                   latent_h, latent_w, size, img_size)
        self.latent_channels = latent_channels
        self.latent_h, self.latent_w = latent_h, latent_w
        self.latent_dim = latent_channels * latent_h * latent_w

    def encode(self, x, flatten=False):
        z = self.encoder(x)
        if flatten:
            return z.view(z.size(0), -1)
        return z

    def decode(self, z, target_size=None):
        if z.dim() == 2:
            z = z.view(z.size(0), self.latent_channels, self.latent_h, self.latent_w)
        return self.decoder(z, target_size)

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z, (x.shape[2], x.shape[3]))
        return recon, z


class DilatedConvBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.norm = nn.LayerNorm(channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        pad = (self.kernel_size - 1) * self.dilation
        x_pad = F.pad(x, (pad, 0))
        out = self.conv(x_pad)
        out = out.transpose(1, 2)
        out = self.norm(out)
        out = self.act(out)
        out = self.dropout(out)
        out = out.transpose(1, 2)
        return x + out


class LatentConvPredictor(nn.Module):
    def __init__(self, latent_dim, hidden_size=256, n_layers=8, kernel_size=3,
                 context_size=64, dropout=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.context_size = context_size
        self.kernel_size = kernel_size
        self.n_layers = n_layers

        self.input_proj = nn.Linear(latent_dim, hidden_size)
        self.blocks = nn.ModuleList([
            DilatedConvBlock(hidden_size, kernel_size, 2 ** i, dropout)
            for i in range(n_layers)
        ])
        self.norm_out = nn.LayerNorm(hidden_size)
        self.output_proj = nn.Linear(hidden_size, latent_dim)

        self.receptive_field = 1 + (kernel_size - 1) * (2 ** n_layers - 1)

    def forward(self, x):
        h = self.input_proj(x)
        h = h.transpose(1, 2)
        for block in self.blocks:
            h = block(h)
        h = h.transpose(1, 2)
        h = self.norm_out(h)
        return self.output_proj(h)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

    def save(self, path, meta=None):
        torch.save({
            'state_dict': self.state_dict(),
            'latent_dim': self.latent_dim,
            'hidden_size': self.hidden_size,
            'n_layers': self.n_layers,
            'kernel_size': self.kernel_size,
            'context_size': self.context_size,
            'receptive_field': self.receptive_field,
            'meta': meta or {},
        }, path)

    @classmethod
    def load(cls, path, map_location='cpu'):
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        m = cls(
            latent_dim=ckpt['latent_dim'],
            hidden_size=ckpt['hidden_size'],
            n_layers=ckpt['n_layers'],
            kernel_size=ckpt.get('kernel_size', 3),
            context_size=ckpt.get('context_size', 64),
        )
        m.load_state_dict(ckpt['state_dict'])
        m.eval()
        return m, ckpt.get('meta', {})


def corrupt_latents(z, prob=0.2, noise_std=0.3, mask_prob=0.1, replace_prob=0.05,
                    multiply=False, mult_min=1.1, mult_max=2.0, mult_prob=0.5):
    B = z.shape[0]
    device = z.device

    if prob > 0:
        apply = (torch.rand(B, 1, 1, device=device) < prob).float()
        noise_mask = (torch.rand_like(z) < mask_prob).float()
        noise = torch.randn_like(z) * noise_std
        z = z + apply * noise_mask * noise
        zero_mask = (torch.rand_like(z) < replace_prob).float()
        z = z * (1 - apply * zero_mask)

    if multiply:
        factor = torch.empty(B, 1, 1, device=device).uniform_(mult_min, mult_max)
        apply_m = (torch.rand(B, 1, 1, device=device) < mult_prob).float()
        z = z * (1 + apply_m * (factor - 1))

    return z


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
        return out if isinstance(img, Image.Image) else transforms.ToTensor()(out)


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
        dx = F.conv2d(dx.view(1, 1, h, w), k, padding=2).view(h, w) * self.alpha
        dy = F.conv2d(dy.view(1, 1, h, w), k, padding=2).view(h, w) * self.alpha
        x, y = torch.meshgrid(torch.arange(w), torch.arange(h), indexing='xy')
        x = (x.float() + dx) / (w - 1) * 2 - 1
        y = (y.float() + dy) / (h - 1) * 2 - 1
        grid = torch.stack([x, y], dim=-1).unsqueeze(0)
        t = transforms.ToTensor()(pil).unsqueeze(0)
        out = F.grid_sample(t, grid, mode='bilinear', padding_mode='border')
        return transforms.ToPILImage()(out.squeeze(0))


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
        'latent_corruption': tk.BooleanVar(value=False),
        'latent_corrupt_prob': tk.DoubleVar(value=0.3),
        'latent_noise_std': tk.DoubleVar(value=0.3),
        'latent_mask_prob': tk.DoubleVar(value=0.1),
        'latent_zero_prob': tk.DoubleVar(value=0.05),
        'latent_multiply': tk.BooleanVar(value=False),
        'latent_mult_min': tk.DoubleVar(value=1.1),
        'latent_mult_max': tk.DoubleVar(value=2.0),
        'latent_mult_prob': tk.DoubleVar(value=0.5),
    }


AUG_DEFAULTS = {
    'flip_horizontal': True, 'rotation': False, 'random_crop': False, 'crop_scale': 0.8,
    'color_jitter': False, 'brightness': 0.2, 'contrast': 0.2, 'saturation': 0.2, 'hue': 0.1,
    'random_perspective': False, 'perspective_distortion': 0.1,
    'elastic_transform': False, 'elastic_alpha': 30, 'elastic_sigma': 3,
    'jpeg_compression': False, 'jpeg_quality_low': 50, 'jpeg_quality_high': 95,
    'stretch_vertical': False, 'stretch_horizontal': False,
    'latent_corruption': False, 'latent_corrupt_prob': 0.3,
    'latent_noise_std': 0.3, 'latent_mask_prob': 0.1, 'latent_zero_prob': 0.05,
    'latent_multiply': False, 'latent_mult_min': 1.1, 'latent_mult_max': 2.0,
    'latent_mult_prob': 0.5,
}


def build_image_aug(aug_dict, img_size=32):
    pipeline = []
    if aug_dict.get('flip_horizontal'):
        pipeline.append(transforms.RandomHorizontalFlip(p=0.5))
    if aug_dict.get('rotation'):
        pipeline.append(transforms.RandomRotation(30, interpolation=Image.BICUBIC,
                                                  expand=False, fill=0))
    if aug_dict.get('random_crop'):
        sc = float(aug_dict.get('crop_scale', 0.8))
        pipeline.append(transforms.RandomResizedCrop(
            size=img_size, scale=(sc, 1.0), interpolation=Image.BICUBIC))
    if aug_dict.get('color_jitter'):
        pipeline.append(transforms.ColorJitter(
            brightness=float(aug_dict.get('brightness', 0.2)),
            contrast=float(aug_dict.get('contrast', 0.2)),
            saturation=float(aug_dict.get('saturation', 0.2)),
            hue=float(aug_dict.get('hue', 0.1))))
    if aug_dict.get('random_perspective'):
        pipeline.append(transforms.RandomPerspective(
            distortion_scale=float(aug_dict.get('perspective_distortion', 0.1)),
            p=0.5, interpolation=Image.BICUBIC, fill=0))
    if aug_dict.get('elastic_transform'):
        pipeline.append(ElasticTransform(
            alpha=int(aug_dict.get('elastic_alpha', 30)),
            sigma=int(aug_dict.get('elastic_sigma', 3)), p=0.5))
    if aug_dict.get('jpeg_compression'):
        pipeline.append(RandomJPEG(
            quality_low=int(aug_dict.get('jpeg_quality_low', 50)),
            quality_high=int(aug_dict.get('jpeg_quality_high', 95)), p=0.5))
    if aug_dict.get('stretch_vertical'):
        pipeline.append(transforms.RandomResizedCrop(
            size=img_size, scale=(0.8, 1.0), ratio=(0.5, 1.0),
            interpolation=Image.BICUBIC))
    if aug_dict.get('stretch_horizontal'):
        pipeline.append(transforms.RandomResizedCrop(
            size=img_size, scale=(0.8, 1.0), ratio=(1.0, 2.0),
            interpolation=Image.BICUBIC))
    return pipeline


class FrameDataset(Dataset):
    def __init__(self, videos, img_size=32, aug_pipeline=None):
        self.videos = videos
        self.img_size = img_size
        self.aug_pipeline = aug_pipeline or []
        self.base = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self._index = []
        for vi, frames in enumerate(videos):
            for fi in range(len(frames)):
                self._index.append((vi, fi))

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        vi, fi = self._index[idx]
        pil = self.videos[vi][fi].copy()
        for a in self.aug_pipeline:
            pil = a(pil)
        return self.base(pil)


class VideoConvApp:
    DISPLAY_SIZE = 256

    def __init__(self, root):
        self.root = root
        self.root.title("Video AR Conv - AE + Dilated Conv Latent Predictor")

        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.root.geometry(f"{max(1100, int(sw*0.85))}x{max(750, int(sh*0.85))}")
        self.root.minsize(1000, 700)

        self.videos = []
        self.video_names = []
        self.ae_model = None
        self.ae_optimizer = None
        self.conv_model = None
        self.conv_optimizer = None
        self.latent_cache = []
        self.latent_cache_ae_sig = None
        self.latent_mean = None
        self.latent_std = None
        self.training_ae = False
        self.training_conv = False
        self.generating = False
        self.last_generated_frames = []
        self.preview_playing = False
        self.preview_after_id = None
        self._preview_was_playing = False
        self.raw_sources = []
        self.gen_tab_index = -1

        self.msg_q_ae = queue.Queue()
        self.msg_q_conv = queue.Queue()
        self.gen_q = queue.Queue()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._build_settings()
        self.setup_gui()
        self.root.after(100, self._poll_ae)
        self.root.after(100, self._poll_conv)
        self.root.after(50, self._poll_gen)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        print(f"Device: {self.device}")

    def _build_settings(self):
        self.global_settings = {
            'img_size': tk.IntVar(value=32),
            'frame_stride': tk.IntVar(value=4),
            'max_frames_per_video': tk.IntVar(value=4000),
        }
        self.ae_settings = {
            'ae_size': tk.StringVar(value='medium'),
            'ae_base_channels': tk.IntVar(value=16),
            'ae_latent_channels': tk.IntVar(value=32),
            'ae_latent_h': tk.IntVar(value=4),
            'ae_latent_w': tk.IntVar(value=4),
            'ae_batch_size': tk.IntVar(value=16),
            'ae_lr': tk.DoubleVar(value=1e-3),
            'ae_num_workers': tk.IntVar(value=0),
        }
        self.conv_settings = {
            'conv_hidden': tk.IntVar(value=256),
            'conv_n_layers': tk.IntVar(value=8),
            'conv_kernel': tk.IntVar(value=3),
            'conv_target_rf': tk.IntVar(value=127),
            'conv_auto_layers': tk.BooleanVar(value=True),
            'conv_context': tk.IntVar(value=64),
            'conv_dropout': tk.DoubleVar(value=0.1),
            'conv_batch': tk.IntVar(value=16),
            'conv_lr': tk.DoubleVar(value=1e-3),
            'conv_steps_per_epoch': tk.IntVar(value=200),
            'conv_residual': tk.BooleanVar(value=True),
        }
        self.gen_settings = {
            'n_frames': tk.IntVar(value=64),
            'fps': tk.IntVar(value=10),
            'temp_noise': tk.DoubleVar(value=0.0),
            'use_latent_dist': tk.BooleanVar(value=True),
        }
        self.ae_aug = default_aug_dict()
        self.conv_aug = default_aug_dict()

    def _ae_signature(self):
        if self.ae_model is None:
            return None
        return (
            self.ae_settings['ae_size'].get(),
            int(self.ae_settings['ae_base_channels'].get()),
            int(self.ae_settings['ae_latent_channels'].get()),
            int(self.ae_settings['ae_latent_h'].get()),
            int(self.ae_settings['ae_latent_w'].get()),
            int(self.global_settings['img_size'].get()),
        )

    def setup_gui(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        builders = [
            ('Dataset', self.setup_dataset_tab),
            ('AE Settings', self.setup_ae_settings_tab),
            ('AE Training', self.setup_ae_train_tab),
            ('Conv Settings', self.setup_conv_settings_tab),
            ('Conv Training', self.setup_conv_train_tab),
            ('Augmentation', self.setup_aug_tab),
            ('Generation', self.setup_generation_tab),
        ]
        for name, builder in builders:
            f = ttk.Frame(self.notebook)
            self.notebook.add(f, text=name)
            builder(f)
        self.gen_tab_index = len(builders) - 1

        self.status = tk.Label(self.root, text=f"Ready | device={self.device}",
                               relief=tk.SUNKEN, anchor=tk.W)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

        self.notebook.bind('<<NotebookTabChanged>>', self._on_tab_changed)

    def _on_tab_changed(self, event=None):
        try:
            current = self.notebook.index(self.notebook.select())
        except Exception:
            return
        on_gen = (current == self.gen_tab_index)
        if not on_gen:
            if self.preview_playing:
                self._preview_was_playing = True
                self.pause_preview()
        else:
            if getattr(self, '_preview_was_playing', False) and self.last_generated_frames:
                self._preview_was_playing = False
                self.play_preview()

    def _spin(self, parent, label, var, low, high):
        f = tk.Frame(parent); f.pack(fill=tk.X, pady=1)
        tk.Label(f, text=label, width=22, anchor='w').pack(side=tk.LEFT)
        ttk.Spinbox(f, from_=low, to=high, textvariable=var, width=10).pack(side=tk.RIGHT)

    def _entry(self, parent, label, var):
        f = tk.Frame(parent); f.pack(fill=tk.X, pady=1)
        tk.Label(f, text=label, width=22, anchor='w').pack(side=tk.LEFT)
        tk.Entry(f, textvariable=var, width=12).pack(side=tk.RIGHT)

    def _combo(self, parent, label, var, values):
        f = tk.Frame(parent); f.pack(fill=tk.X, pady=1)
        tk.Label(f, text=label, width=22, anchor='w').pack(side=tk.LEFT)
        ttk.Combobox(f, textvariable=var, values=values, state='readonly',
                     width=12).pack(side=tk.RIGHT)

    def _scroll(self, parent):
        canvas = tk.Canvas(parent)
        sb = tk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        inner = tk.Frame(canvas)
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        return inner

    def setup_dataset_tab(self, tab):
        main = tk.Frame(tab); main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        left = tk.Frame(main, width=320); left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left.pack_propagate(False)

        tk.Label(left, text="Video Dataset", font=("Arial", 12, "bold")).pack(pady=(0, 8))
        f = tk.LabelFrame(left, text="Sources", padx=6, pady=6); f.pack(fill=tk.X)
        tk.Button(f, text="Add Video(s)", command=self.add_videos, width=24).pack(pady=2)
        tk.Button(f, text="Add Image Folder (as sequence)",
                  command=self.add_image_folder, width=24).pack(pady=2)
        tk.Button(f, text="Clear All", command=self.clear_videos, width=24).pack(pady=2)

        tk.Label(left, text="Videos loaded:", anchor='w').pack(pady=(10, 0), fill=tk.X)
        self.video_listbox = tk.Listbox(left, height=10)
        self.video_listbox.pack(fill=tk.X)

        f2 = tk.LabelFrame(left, text="Extraction", padx=6, pady=6); f2.pack(fill=tk.X, pady=8)
        self._spin(f2, "Frame stride:", self.global_settings['frame_stride'], 1, 30)
        self._spin(f2, "Max frames/video:", self.global_settings['max_frames_per_video'],
                   100, 100000)

        tk.Button(left, text="Rebuild from sources",
                  command=self.rebuild_videos_from_sources).pack(pady=4)
        self.dataset_info = tk.Label(left, text="", fg='blue', wraplength=300, justify='left')
        self.dataset_info.pack(fill=tk.X, pady=6)

        right = tk.Frame(main); right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        pf = tk.LabelFrame(right, text="Frame Grid Preview (first 64 frames)",
                           padx=5, pady=5)
        pf.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        self.frame_preview_canvas = tk.Canvas(pf, bg='#1e1e1e', width=512, height=128)
        self.frame_preview_canvas.pack()

    def add_videos(self):
        paths = filedialog.askopenfilenames(
            filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv *.webm *.gif"),
                       ("All files", "*.*")])
        for p in paths:
            self.raw_sources.append(('video', p))
            self.video_listbox.insert(tk.END, f"[VID] {os.path.basename(p)}")
        if paths:
            self.rebuild_videos_from_sources()

    def add_image_folder(self):
        folder = filedialog.askdirectory()
        if not folder:
            return
        files = sorted([os.path.join(folder, f) for f in os.listdir(folder)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))])
        if not files:
            messagebox.showwarning("Empty", "No images found in folder.")
            return
        self.raw_sources.append(('images', files))
        self.video_listbox.insert(tk.END,
                                  f"[IMG] {os.path.basename(folder)} ({len(files)} files)")
        self.rebuild_videos_from_sources()

    def clear_videos(self):
        self.raw_sources = []
        self.videos = []
        self.video_names = []
        self.latent_cache = []
        self.latent_cache_ae_sig = None
        self.video_listbox.delete(0, tk.END)
        self.dataset_info.config(text="")
        self.frame_preview_canvas.delete("all")

    def rebuild_videos_from_sources(self):
        try:
            img_size = self.global_settings['img_size'].get()
            stride = self.global_settings['frame_stride'].get()
            maxf = self.global_settings['max_frames_per_video'].get()
            new_videos = []
            new_names = []
            for kind, path in self.raw_sources:
                if kind == 'video':
                    frames = extract_frames_from_video(path, img_size, stride, maxf)
                    new_videos.append(frames)
                    new_names.append(os.path.basename(path))
                else:
                    frames = []
                    for i, p in enumerate(path):
                        if i % stride != 0:
                            continue
                        try:
                            im = load_image_as_rgb(p).resize((img_size, img_size),
                                                             Image.LANCZOS)
                            frames.append(im)
                        except Exception:
                            pass
                        if len(frames) >= maxf:
                            break
                    new_videos.append(frames)
                    new_names.append(os.path.basename(os.path.dirname(path[0])))
            self.videos = new_videos
            self.video_names = new_names
            self.latent_cache = []
            self.latent_cache_ae_sig = None
            self._update_dataset_info()
            self._preview_frames()
            self._refresh_rf()
            self.status.config(text=f"Extracted frames from {len(self.videos)} sources.")
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to extract frames:\n{e}")

    def _update_dataset_info(self):
        total = sum(len(v) for v in self.videos)
        parts = [f"{name}: {len(v)} frames"
                 for name, v in zip(self.video_names, self.videos)]
        self.dataset_info.config(text=f"Total frames: {total}\n" + "\n".join(parts))

    def _preview_frames(self):
        if not self.videos:
            self.frame_preview_canvas.delete("all")
            return
        cell = 32
        cols = 16
        rows = 4
        grid = Image.new('RGB', (cell * cols, cell * rows), (20, 20, 20))
        flat = []
        for v in self.videos:
            flat.extend(v)
            if len(flat) >= cols * rows:
                break
        for i, f in enumerate(flat[:cols * rows]):
            r, c = divmod(i, cols)
            tile = fit_to_cell(f, cell)
            grid.paste(tile, (c * cell, r * cell))
        self._frame_preview_photo = ImageTk.PhotoImage(grid)
        self.frame_preview_canvas.delete("all")
        self.frame_preview_canvas.create_image(0, 0, anchor='nw',
                                               image=self._frame_preview_photo)

    def setup_ae_settings_tab(self, tab):
        inner = self._scroll(tab)
        wrap = tk.Frame(inner); wrap.pack(fill=tk.X, padx=20, pady=20)

        tk.Label(wrap, text="Autoencoder Settings",
                 font=("Arial", 14, "bold")).pack(pady=(0, 10))

        mf = tk.LabelFrame(wrap, text="AE Architecture", padx=10, pady=10)
        mf.pack(fill=tk.X, pady=5)
        self._combo(mf, "Size preset:", self.ae_settings['ae_size'],
                    ['tiny', 'small', 'medium', 'big', 'large'])
        self._spin(mf, "Base channels:", self.ae_settings['ae_base_channels'], 4, 256)
        self._spin(mf, "Latent channels:", self.ae_settings['ae_latent_channels'], 1, 128)
        self._spin(mf, "Latent height:", self.ae_settings['ae_latent_h'], 1, 32)
        self._spin(mf, "Latent width:", self.ae_settings['ae_latent_w'], 1, 32)

        tf = tk.LabelFrame(wrap, text="AE Training", padx=10, pady=10)
        tf.pack(fill=tk.X, pady=5)
        self._spin(tf, "Batch size:", self.ae_settings['ae_batch_size'], 1, 128)
        self._entry(tf, "Learning rate:", self.ae_settings['ae_lr'])
        self._spin(tf, "DataLoader workers:", self.ae_settings['ae_num_workers'], 0, 8)

        gf = tk.LabelFrame(wrap, text="Global", padx=10, pady=10)
        gf.pack(fill=tk.X, pady=5)
        self._spin(gf, "Image size:", self.global_settings['img_size'], 16, 128)

        self.latent_dim_label = tk.Label(wrap, text="", fg='#2266aa',
                                         font=('Arial', 10, 'italic'))
        self.latent_dim_label.pack(pady=8)
        self._refresh_latent_dim()
        for v in self.ae_settings.values():
            try:
                v.trace_add('write', lambda *a: self._refresh_latent_dim())
            except Exception:
                pass
        try:
            self.global_settings['img_size'].trace_add(
                'write', lambda *a: self._refresh_latent_dim())
        except Exception:
            pass

    def _refresh_latent_dim(self):
        try:
            d = (self.ae_settings['ae_latent_channels'].get()
                 * self.ae_settings['ae_latent_h'].get()
                 * self.ae_settings['ae_latent_w'].get())
            self.latent_dim_label.config(text=f"Latent dimension: {d}")
        except Exception:
            pass

    def setup_ae_train_tab(self, tab):
        main = tk.Frame(tab); main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        left = tk.Frame(main, width=320); left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left.pack_propagate(False)

        tk.Label(left, text="AE Training", font=("Arial", 12, "bold")).pack(pady=(0, 10))

        ep = tk.Frame(left); ep.pack(pady=5)
        tk.Label(ep, text="Epochs:").pack(side=tk.LEFT)
        self.ae_epoch_var = tk.StringVar(value="100")
        tk.Entry(ep, textvariable=self.ae_epoch_var, width=8).pack(side=tk.LEFT, padx=5)

        tk.Button(left, text="Initialize AE", command=self.init_ae, width=24).pack(pady=2)
        tk.Button(left, text="Start AE Training", command=self.start_ae_training,
                  width=24, bg="lightgreen").pack(pady=2)
        tk.Button(left, text="Stop", command=self.stop_ae_training,
                  width=24, bg="salmon").pack(pady=2)
        tk.Button(left, text="Save AE", command=self.save_ae, width=24).pack(pady=2)
        tk.Button(left, text="Load AE", command=self.load_ae, width=24).pack(pady=2)
        tk.Button(left, text="Show Preview", command=self.show_ae_preview,
                  width=24).pack(pady=8)

        right = tk.Frame(main); right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        pf = tk.LabelFrame(right, text="Top: inputs | Bottom: reconstructions",
                           padx=5, pady=5)
        pf.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        self.ae_preview_canvas = tk.Canvas(pf, bg='#1e1e1e', width=512, height=256)
        self.ae_preview_canvas.pack()

        lf = tk.LabelFrame(right, text="Log", padx=5, pady=5); lf.pack(fill=tk.BOTH, expand=True)
        self.ae_log = tk.Text(lf, height=15, font=("Courier", 9))
        self.ae_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = tk.Scrollbar(lf, command=self.ae_log.yview); sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.ae_log.config(yscrollcommand=sb.set)

    def setup_conv_settings_tab(self, tab):
        inner = self._scroll(tab)
        wrap = tk.Frame(inner); wrap.pack(fill=tk.X, padx=20, pady=20)

        tk.Label(wrap, text="Conv Predictor Settings",
                 font=("Arial", 14, "bold")).pack(pady=(0, 10))

        af = tk.LabelFrame(wrap, text="Architecture", padx=10, pady=10)
        af.pack(fill=tk.X, pady=5)
        self._spin(af, "Hidden size:", self.conv_settings['conv_hidden'], 16, 2048)
        self._spin(af, "Kernel size:", self.conv_settings['conv_kernel'], 2, 9)

        rf_row = tk.Frame(af); rf_row.pack(fill=tk.X, pady=1)
        tk.Label(rf_row, text="Target RF:", width=22, anchor='w').pack(side=tk.LEFT)
        self.rf_entry = ttk.Entry(rf_row,
                                  textvariable=self.conv_settings['conv_target_rf'],
                                  width=10)
        self.rf_entry.pack(side=tk.RIGHT)
        self.rf_entry.bind('<FocusOut>', self._snap_rf)
        self.rf_entry.bind('<Return>', self._snap_rf)

        tk.Checkbutton(af, text="Auto-compute layers from Target RF",
                       variable=self.conv_settings['conv_auto_layers'],
                       command=self._refresh_rf).pack(anchor='w', pady=(4, 0))
        self._spin(af, "Num layers (manual):", self.conv_settings['conv_n_layers'], 1, 20)
        self._spin(af, "Context (window):", self.conv_settings['conv_context'], 2, 2048)
        self._entry(af, "Dropout:", self.conv_settings['conv_dropout'])
        tk.Checkbutton(af, text="Predict residual (z_{t+1} - z_t)",
                       variable=self.conv_settings['conv_residual']).pack(anchor='w')

        self.conv_diag = tk.Label(wrap, text="", fg='#2266aa',
                                  font=('Courier', 10), justify='left', anchor='w')
        self.conv_diag.pack(fill=tk.X, pady=8)

        tf = tk.LabelFrame(wrap, text="Training", padx=10, pady=10)
        tf.pack(fill=tk.X, pady=5)
        self._spin(tf, "Batch size:", self.conv_settings['conv_batch'], 1, 256)
        self._entry(tf, "Learning rate:", self.conv_settings['conv_lr'])
        self._spin(tf, "Steps per epoch:", self.conv_settings['conv_steps_per_epoch'],
                   10, 5000)

        for v in self.conv_settings.values():
            try:
                v.trace_add('write', lambda *a: self._refresh_rf())
            except Exception:
                pass

        self._refresh_rf()

    def _snap_rf(self, event=None):
        try:
            v = int(float(self.conv_settings['conv_target_rf'].get()))
        except Exception:
            v = 127
        self.conv_settings['conv_target_rf'].set(snap_to_odd(v))
        self._refresh_rf()

    def _refresh_rf(self):
        try:
            k = int(self.conv_settings['conv_kernel'].get())
            auto = bool(self.conv_settings['conv_auto_layers'].get())
            if auto:
                target_rf = snap_to_odd(int(self.conv_settings['conv_target_rf'].get()))
                self.conv_settings['conv_target_rf'].set(target_rf)
                n, actual_rf = compute_layers_for_rf(target_rf, k)
                self.conv_settings['conv_n_layers'].set(n)
            else:
                n = int(self.conv_settings['conv_n_layers'].get())
                actual_rf = 1 + (k - 1) * (2 ** n - 1)
                self.conv_settings['conv_target_rf'].set(actual_rf)

            params = ""
            if self.ae_model is not None:
                D = self.ae_model.latent_dim
                H = int(self.conv_settings['conv_hidden'].get())
                inp = D * H + H
                blk = n * (H * H * k + 3 * H)
                out = H * D + D
                norm_out = 2 * H
                total = inp + blk + norm_out + out
                params = (f"Params: ~{total:,}\n"
                          f"  input_proj    {inp:>10,}\n"
                          f"  conv blocks   {blk:>10,}  ({n} × H×H×k)\n"
                          f"  norm_out      {norm_out:>10,}\n"
                          f"  output_proj   {out:>10,}\n")

            cache_line = ""
            if self.latent_cache:
                lens = [t.shape[0] for t in self.latent_cache]
                shortest = min(lens)
                ctx = int(self.conv_settings['conv_context'].get())
                cache_line = (f"Cache: {len(lens)} video(s), "
                              f"frames min={shortest} max={max(lens)} total={sum(lens)}\n")
                if self.latent_cache_ae_sig != self._ae_signature():
                    cache_line += ("  cache was built with a different AE or img_size; "
                                   "rebuild before training.\n")
                if ctx + 1 > shortest:
                    cache_line += (f"  context+1 ({ctx + 1}) > shortest video "
                                   f"({shortest}); training will produce no batches.\n")
                if actual_rf > shortest:
                    cache_line += (f"  RF ({actual_rf}) > shortest video "
                                   f"({shortest}); model can't see full RF.\n")

            rf_line = f"Kernel={k}, layers={n}, actual RF={actual_rf}\n"
            latent_line = ""
            if self.ae_model is not None:
                latent_line = (f"Latent: dim={self.ae_model.latent_dim} "
                               f"({self.ae_model.latent_channels}×"
                               f"{self.ae_model.latent_h}×"
                               f"{self.ae_model.latent_w})\n")

            self.conv_diag.config(text=rf_line + latent_line + params + cache_line)
        except Exception as e:
            self.conv_diag.config(text=f"(diag error: {e})")

    def setup_conv_train_tab(self, tab):
        main = tk.Frame(tab); main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        left = tk.Frame(main, width=320); left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left.pack_propagate(False)

        tk.Label(left, text="Conv Training", font=("Arial", 12, "bold")).pack(pady=(0, 10))

        ep = tk.Frame(left); ep.pack(pady=5)
        tk.Label(ep, text="Epochs:").pack(side=tk.LEFT)
        self.conv_epoch_var = tk.StringVar(value="100")
        tk.Entry(ep, textvariable=self.conv_epoch_var, width=8).pack(side=tk.LEFT, padx=5)

        tk.Button(left, text="Rebuild Latent Cache",
                  command=self.build_latent_cache, width=24).pack(pady=2)
        tk.Button(left, text="Initialize Conv", command=self.init_conv,
                  width=24).pack(pady=2)
        tk.Button(left, text="Start Conv Training", command=self.start_conv_training,
                  width=24, bg="lightgreen").pack(pady=2)
        tk.Button(left, text="Stop", command=self.stop_conv_training,
                  width=24, bg="salmon").pack(pady=2)
        tk.Button(left, text="Save Conv", command=self.save_conv, width=24).pack(pady=2)
        tk.Button(left, text="Load Conv", command=self.load_conv, width=24).pack(pady=2)
        tk.Button(left, text="Autoregressive Preview",
                  command=self.preview_conv_rollout, width=24).pack(pady=8)

        right = tk.Frame(main); right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        pf = tk.Frame(right)
        pf.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        self.conv_preview_canvas = tk.Canvas(pf, bg='#1e1e1e', width=384, height=384,
                                             highlightthickness=0)
        self.conv_preview_canvas.pack()

        lf = tk.LabelFrame(right, text="Log", padx=5, pady=5); lf.pack(fill=tk.BOTH, expand=True)
        self.conv_log = tk.Text(lf, height=15, font=("Courier", 9))
        self.conv_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = tk.Scrollbar(lf, command=self.conv_log.yview); sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.conv_log.config(yscrollcommand=sb.set)

    def setup_aug_tab(self, tab):
        tk.Label(tab, text="Augmentation", font=("Arial", 14, "bold")).pack(pady=(15, 10))
        cols = tk.Frame(tab); cols.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        ae_col = tk.LabelFrame(cols, text="AE Image Augmentations", padx=8, pady=8)
        ae_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        self._build_aug_controls(ae_col, self.ae_aug)

        conv_col = tk.LabelFrame(cols, text="Conv Augmentations (latent-space)",
                                 padx=8, pady=8)
        conv_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))
        self._build_latent_corruption_controls(conv_col, self.conv_aug)

        r = tk.Frame(tab); r.pack(fill=tk.X, pady=10)
        tk.Button(r, text="Reset AE augs",
                  command=lambda: self._reset_aug(self.ae_aug)).pack(
            side=tk.LEFT, padx=10, expand=True)
        tk.Button(r, text="Reset Conv augs",
                  command=lambda: self._reset_aug(self.conv_aug)).pack(
            side=tk.RIGHT, padx=10, expand=True)

    def _build_aug_controls(self, parent, aug):
        def row():
            f = tk.Frame(parent); f.pack(fill=tk.X, pady=2); return f

        row(); tk.Checkbutton(parent, text="Horizontal Flip",
                              variable=aug['flip_horizontal']).pack(anchor='w')
        tk.Checkbutton(parent, text="Rotation (±30°)",
                       variable=aug['rotation']).pack(anchor='w')
        r = row()
        tk.Checkbutton(r, text="Random Crop",
                       variable=aug['random_crop']).pack(side=tk.LEFT)
        tk.Label(r, text="scale:").pack(side=tk.LEFT, padx=(8, 2))
        tk.Entry(r, textvariable=aug['crop_scale'], width=6).pack(side=tk.LEFT)
        tk.Checkbutton(parent, text="Color Jitter",
                       variable=aug['color_jitter']).pack(anchor='w')
        r = row()
        tk.Label(r, text="  bri/con/sat/hue:").pack(side=tk.LEFT)
        for k in ['brightness', 'contrast', 'saturation', 'hue']:
            tk.Entry(r, textvariable=aug[k], width=5).pack(side=tk.LEFT, padx=1)
        r = row()
        tk.Checkbutton(r, text="Random Perspective",
                       variable=aug['random_perspective']).pack(side=tk.LEFT)
        tk.Label(r, text="dist:").pack(side=tk.LEFT, padx=(8, 2))
        tk.Entry(r, textvariable=aug['perspective_distortion'], width=6).pack(side=tk.LEFT)
        r = row()
        tk.Checkbutton(r, text="Elastic",
                       variable=aug['elastic_transform']).pack(side=tk.LEFT)
        tk.Label(r, text="alpha:").pack(side=tk.LEFT, padx=(8, 2))
        tk.Entry(r, textvariable=aug['elastic_alpha'], width=4).pack(side=tk.LEFT)
        tk.Label(r, text="sigma:").pack(side=tk.LEFT, padx=(4, 2))
        tk.Entry(r, textvariable=aug['elastic_sigma'], width=4).pack(side=tk.LEFT)
        r = row()
        tk.Checkbutton(r, text="JPEG Compression",
                       variable=aug['jpeg_compression']).pack(side=tk.LEFT)
        tk.Label(r, text="q:").pack(side=tk.LEFT, padx=(8, 2))
        tk.Entry(r, textvariable=aug['jpeg_quality_low'], width=5).pack(side=tk.LEFT)
        tk.Entry(r, textvariable=aug['jpeg_quality_high'], width=5).pack(side=tk.LEFT)
        r = row()
        tk.Checkbutton(r, text="Stretch V",
                       variable=aug['stretch_vertical']).pack(side=tk.LEFT, padx=(0, 10))
        tk.Checkbutton(r, text="Stretch H",
                       variable=aug['stretch_horizontal']).pack(side=tk.LEFT)

    def _build_latent_corruption_controls(self, parent, aug):
        tk.Label(parent, text=("Randomly corrupt input latents so the conv learns to\n"
                               "ignore noisy / amplified values. Target stays clean."),
                 justify='left', fg='#335', font=('Arial', 9)).pack(anchor='w', pady=(0, 8))

        def row(label, var, hint=""):
            f = tk.Frame(parent); f.pack(fill=tk.X, pady=2)
            tk.Label(f, text=label, width=24, anchor='w').pack(side=tk.LEFT)
            tk.Entry(f, textvariable=var, width=8).pack(side=tk.LEFT)
            if hint:
                tk.Label(f, text=hint, fg='gray', font=('Arial', 8)).pack(side=tk.LEFT, padx=6)

        tk.Label(parent, text="— Gaussian noise + zeros —",
                 fg='#555', font=('Arial', 9, 'italic')).pack(anchor='w', pady=(4, 2))
        tk.Checkbutton(parent, text="Enable corruption",
                       variable=aug['latent_corruption']).pack(anchor='w', pady=2)
        row("Corrupt probability:", aug['latent_corrupt_prob'], "(per batch)")
        row("Noise std:", aug['latent_noise_std'])
        row("Dim mask probability:", aug['latent_mask_prob'])
        row("Dim zero probability:", aug['latent_zero_prob'])

        tk.Label(parent, text="— Multiplication —",
                 fg='#555', font=('Arial', 9, 'italic')).pack(anchor='w', pady=(10, 2))
        tk.Checkbutton(parent, text="Enable multiplication",
                       variable=aug['latent_multiply']).pack(anchor='w', pady=2)
        row("Multiply min:", aug['latent_mult_min'])
        row("Multiply max:", aug['latent_mult_max'])
        row("Multiply probability:", aug['latent_mult_prob'], "(per batch item)")

    def _reset_aug(self, aug):
        for k, v in AUG_DEFAULTS.items():
            if k in aug:
                aug[k].set(v)

    def setup_generation_tab(self, tab):
        main = tk.Frame(tab); main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        top = tk.Frame(main); top.pack(fill=tk.X, pady=5)

        tk.Label(top, text="Frames:").pack(side=tk.LEFT)
        tk.Spinbox(top, from_=2, to=10000, textvariable=self.gen_settings['n_frames'],
                   width=8).pack(side=tk.LEFT, padx=4)
        tk.Label(top, text="FPS:").pack(side=tk.LEFT, padx=(12, 0))
        tk.Spinbox(top, from_=1, to=60, textvariable=self.gen_settings['fps'],
                   width=5).pack(side=tk.LEFT, padx=4)
        tk.Label(top, text="Sampling noise:").pack(side=tk.LEFT, padx=(12, 0))
        tk.Entry(top, textvariable=self.gen_settings['temp_noise'],
                 width=6).pack(side=tk.LEFT, padx=4)
        tk.Checkbutton(top, text="Start from N(μ, σ²) of training latents",
                       variable=self.gen_settings['use_latent_dist']).pack(
            side=tk.LEFT, padx=8)

        btnrow = tk.Frame(main); btnrow.pack(fill=tk.X, pady=6)
        self.gen_btn = tk.Button(btnrow, text="Generate", bg="lightgreen",
                                 command=self.start_generation)
        self.gen_btn.pack(side=tk.LEFT, padx=4)
        self.stop_gen_btn = tk.Button(btnrow, text="Stop", bg="salmon",
                                      command=self.stop_generation, state=tk.DISABLED)
        self.stop_gen_btn.pack(side=tk.LEFT, padx=4)
        tk.Button(btnrow, text="Play Preview",
                  command=self.play_preview).pack(side=tk.LEFT, padx=4)
        tk.Button(btnrow, text="Pause Preview",
                  command=self.pause_preview).pack(side=tk.LEFT, padx=4)
        tk.Button(btnrow, text="Save GIF", command=self.save_gif).pack(side=tk.LEFT, padx=4)
        tk.Button(btnrow, text="Save MP4",
                  command=self.save_mp4).pack(side=tk.LEFT, padx=4)
        tk.Button(btnrow, text="Save PNG sequence",
                  command=self.save_png_seq).pack(side=tk.LEFT, padx=4)

        cf = tk.LabelFrame(main, text="Video preview", padx=5, pady=5)
        cf.pack(fill=tk.BOTH, expand=True, pady=5)
        self.gen_preview_canvas = tk.Canvas(cf, bg='black', width=320, height=320)
        self.gen_preview_canvas.pack()

        self.gen_info = tk.Label(main, text="", fg='blue')
        self.gen_info.pack()

        lf = tk.LabelFrame(main, text="Log", padx=5, pady=5)
        lf.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        self.gen_log = tk.Text(lf, height=10, font=("Courier", 9))
        self.gen_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = tk.Scrollbar(lf, command=self.gen_log.yview); sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.gen_log.config(yscrollcommand=sb.set)

    def log_ae(self, msg):
        self.msg_q_ae.put(msg)

    def log_conv(self, msg):
        self.msg_q_conv.put(msg)

    def log_gen(self, msg):
        self.gen_log.insert(tk.END, f"{time.strftime('%H:%M:%S')} - {msg}\n")
        self.gen_log.see(tk.END)
        self.status.config(text=msg[:100])

    def _poll_ae(self):
        try:
            while True:
                m = self.msg_q_ae.get_nowait()
                self.ae_log.insert(tk.END, f"{time.strftime('%H:%M:%S')} - {m}\n")
                self.ae_log.see(tk.END)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_ae)

    def _poll_conv(self):
        try:
            while True:
                m = self.msg_q_conv.get_nowait()
                self.conv_log.insert(tk.END, f"{time.strftime('%H:%M:%S')} - {m}\n")
                self.conv_log.see(tk.END)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_conv)

    def _poll_gen(self):
        try:
            while True:
                kind, payload = self.gen_q.get_nowait()
                if kind == 'frame':
                    try:
                        if self.notebook.index(self.notebook.select()) == self.gen_tab_index:
                            self._show_gen_frame(payload)
                    except Exception:
                        pass
                elif kind == 'done':
                    self._on_gen_done(payload)
                elif kind == 'error':
                    self.log_gen(f"Generation error: {payload}")
                    self.gen_btn.config(state=tk.NORMAL)
                    self.stop_gen_btn.config(state=tk.DISABLED)
                    self.generating = False
        except queue.Empty:
            pass
        self.root.after(50, self._poll_gen)

    def init_ae(self):
        try:
            img_size = self.global_settings['img_size'].get()
            lh = int(self.ae_settings['ae_latent_h'].get())
            lw = int(self.ae_settings['ae_latent_w'].get())
            if img_size % lh or img_size % lw:
                raise ValueError(
                    f"img_size={img_size} must be divisible by latent {lh}×{lw}")
            if (img_size // lh) != (img_size // lw):
                raise ValueError(
                    f"non-square downsampling ratio "
                    f"{(img_size // lh)}:{(img_size // lw)} not supported")
            self.ae_model = PureAE(
                in_channels=3,
                base_channels=self.ae_settings['ae_base_channels'].get(),
                latent_channels=self.ae_settings['ae_latent_channels'].get(),
                latent_h=lh, latent_w=lw,
                size=self.ae_settings['ae_size'].get(),
                img_size=img_size,
            ).to(self.device)
            self.ae_optimizer = optim.Adam(self.ae_model.parameters(),
                                           lr=self.ae_settings['ae_lr'].get())
            n = sum(p.numel() for p in self.ae_model.parameters())
            self.log_ae(f"AE initialized. Params: {n:,}. "
                        f"Latent dim: {self.ae_model.latent_dim}.")
            self.latent_cache = []
            self.latent_cache_ae_sig = None
            self._refresh_rf()
        except Exception as e:
            traceback.print_exc()
            self.ae_model = None
            self.log_ae(f"AE init error: {e}")

    def start_ae_training(self):
        if not self.videos:
            self.log_ae("No videos loaded."); return
        if self.ae_model is None:
            self.log_ae("Init AE first."); return
        if self.training_ae:
            return
        try:
            epochs = int(self.ae_epoch_var.get())
            if epochs < 1:
                raise ValueError
        except Exception:
            self.log_ae("Invalid epochs"); return
        self.training_ae = True
        threading.Thread(target=self._train_ae, args=(epochs,), daemon=True).start()
        self.log_ae(f"AE training started for {epochs} epochs.")

    def stop_ae_training(self):
        self.training_ae = False
        self.log_ae("AE training stopped.")

    def _train_ae(self, epochs):
        try:
            bs = self.ae_settings['ae_batch_size'].get()
            nw = self.ae_settings['ae_num_workers'].get()
            img_size = self.global_settings['img_size'].get()
            aug_cfg = {k: v.get() for k, v in self.ae_aug.items()}
            aug_pipe = build_image_aug(aug_cfg, img_size)
            ds = FrameDataset(self.videos, img_size, aug_pipe)
            if len(ds) == 0:
                self.log_ae("No frames to train on.")
                self.training_ae = False
                return
            dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=nw,
                            pin_memory=(self.device.type == 'cuda'),
                            persistent_workers=(nw > 0))

            t0 = time.time()
            for ep in range(epochs):
                if not self.training_ae:
                    break
                total, n = 0.0, 0
                for imgs in dl:
                    if not self.training_ae:
                        break
                    imgs = imgs.to(self.device)
                    recon, _ = self.ae_model(imgs)
                    loss = F.mse_loss(recon, imgs)
                    if not torch.isfinite(loss):
                        self.ae_optimizer.zero_grad()
                        continue
                    self.ae_optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.ae_model.parameters(), 1.0)
                    self.ae_optimizer.step()
                    total += loss.item(); n += 1
                if n == 0:
                    self.log_ae(f"Epoch {ep+1}: no steps ran")
                    self.training_ae = False
                    break
                avg = total / n
                self.log_ae(f"Epoch {ep+1}/{epochs} | MSE: {avg:.6f} | "
                            f"t={time.time()-t0:.1f}s")
                if (ep + 1) % 5 == 0:
                    self.root.after(0, self.show_ae_preview)

            self.training_ae = False
            self.latent_cache = []
            self.latent_cache_ae_sig = None
            self.log_ae("AE training finished.")
            self.root.after(0, self.build_latent_cache)
        except Exception as e:
            traceback.print_exc()
            self.log_ae(f"AE train error: {e}")
            self.training_ae = False

    def show_ae_preview(self):
        if self.ae_model is None or not self.videos:
            return
        try:
            cols = 8
            rows = 4
            cell = 64
            n_total = cols * rows
            n_inputs = n_total // 2
            flat = [f for v in self.videos for f in v]
            if not flat:
                return

            if len(flat) >= n_inputs:
                picks = random.sample(flat, n_inputs)
            else:
                picks = [random.choice(flat) for _ in range(n_inputs)]

            base = transforms.Compose([
                transforms.Resize((self.global_settings['img_size'].get(),
                                   self.global_settings['img_size'].get())),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            batch = torch.stack([base(p) for p in picks]).to(self.device)
            with torch.no_grad():
                recon, _ = self.ae_model(batch)
            inp = ((batch + 1) / 2).clamp(0, 1).cpu().numpy()
            rec = ((recon + 1) / 2).clamp(0, 1).cpu().numpy()

            grid = Image.new('RGB', (cell * cols, cell * rows), (20, 20, 20))
            for i in range(n_inputs):
                r = (i // cols) * 2
                c = i % cols
                im_in = (inp[i].transpose(1, 2, 0) * 255).astype(np.uint8)
                im_rc = (rec[i].transpose(1, 2, 0) * 255).astype(np.uint8)
                tile_in = Image.fromarray(im_in).resize((cell, cell), Image.NEAREST)
                tile_rc = Image.fromarray(im_rc).resize((cell, cell), Image.NEAREST)
                grid.paste(tile_in, (c * cell, r * cell))
                grid.paste(tile_rc, (c * cell, (r + 1) * cell))

            self._ae_preview_photo = ImageTk.PhotoImage(grid)
            self.ae_preview_canvas.delete("all")
            self.ae_preview_canvas.create_image(0, 0, anchor='nw',
                                                image=self._ae_preview_photo)
        except Exception as e:
            traceback.print_exc()
            self.log_ae(f"AE preview error: {e}")

    def save_ae(self):
        if self.ae_model is None:
            self.log_ae("No AE to save.")
            return
        fname = filedialog.asksaveasfilename(defaultextension=".pth",
                                             filetypes=[("PyTorch", "*.pth")])
        if not fname:
            return
        torch.save({
            'state': self.ae_model.state_dict(),
            'settings': {
                'base_channels': self.ae_settings['ae_base_channels'].get(),
                'latent_channels': self.ae_settings['ae_latent_channels'].get(),
                'latent_h': self.ae_settings['ae_latent_h'].get(),
                'latent_w': self.ae_settings['ae_latent_w'].get(),
                'size': self.ae_settings['ae_size'].get(),
                'img_size': self.global_settings['img_size'].get(),
            },
        }, fname)
        self.log_ae(f"Saved AE: {fname}")

    def load_ae(self):
        fname = filedialog.askopenfilename(filetypes=[("PyTorch", "*.pth")])
        if not fname:
            return
        try:
            ckpt = torch.load(fname, map_location='cpu', weights_only=False)
            s = ckpt['settings']
            self.ae_settings['ae_base_channels'].set(s['base_channels'])
            self.ae_settings['ae_latent_channels'].set(s['latent_channels'])
            self.ae_settings['ae_latent_h'].set(s['latent_h'])
            self.ae_settings['ae_latent_w'].set(s['latent_w'])
            self.ae_settings['ae_size'].set(s['size'])
            self.global_settings['img_size'].set(s['img_size'])
            self.init_ae()
            if self.ae_model is not None:
                self.ae_model.load_state_dict(ckpt['state'])
                self.log_ae(f"Loaded AE: {fname}")
        except Exception as e:
            traceback.print_exc()
            self.log_ae(f"AE load error: {e}")

    def build_latent_cache(self):
        if self.ae_model is None or not self.videos:
            self.log_conv("Cannot build cache: need AE + videos.")
            return
        try:
            self.ae_model.eval()
            img_size = self.global_settings['img_size'].get()
            base = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            new_cache = []
            with torch.no_grad():
                for v in self.videos:
                    if not v:
                        continue
                    tensors = [base(f) for f in v]
                    lats = []
                    for i in range(0, len(tensors), 64):
                        chunk = torch.stack(tensors[i:i+64]).to(self.device)
                        z = self.ae_model.encode(chunk, flatten=True)
                        lats.append(z.cpu())
                    new_cache.append(torch.cat(lats, dim=0))
            self.latent_cache = new_cache
            self.latent_cache_ae_sig = self._ae_signature()
            if self.latent_cache:
                allz = torch.cat(self.latent_cache, dim=0)
                self.latent_mean = allz.mean(0)
                self.latent_std = allz.std(0).clamp(min=1e-6)
                lens = [t.shape[0] for t in self.latent_cache]
                self.log_conv(
                    f"Latent cache: {len(lens)} video(s), "
                    f"frames min={min(lens)} max={max(lens)} total={sum(lens)}, "
                    f"latent_dim={allz.shape[1]}"
                )
                if min(lens) < 2:
                    self.log_conv("  some videos have < 2 frames; skipped in training.")
            else:
                self.log_conv("Latent cache: EMPTY.")
            self.root.after(0, self._refresh_rf)
        except Exception as e:
            traceback.print_exc()
            self.log_conv(f"Cache error: {e}")

    def _latent_cache_valid(self):
        if not self.latent_cache:
            return False
        if self.latent_cache_ae_sig != self._ae_signature():
            return False
        return True

    def init_conv(self):
        if self.ae_model is None:
            self.log_conv("Init/load AE first."); return
        try:
            latent_dim = self.ae_model.latent_dim
            k = int(self.conv_settings['conv_kernel'].get())
            auto = bool(self.conv_settings['conv_auto_layers'].get())
            if auto:
                target_rf = snap_to_odd(int(self.conv_settings['conv_target_rf'].get()))
                n_layers, actual_rf = compute_layers_for_rf(target_rf, k)
                self.conv_settings['conv_n_layers'].set(n_layers)
            else:
                n_layers = int(self.conv_settings['conv_n_layers'].get())
                actual_rf = 1 + (k - 1) * (2 ** n_layers - 1)

            self.conv_model = LatentConvPredictor(
                latent_dim=latent_dim,
                hidden_size=self.conv_settings['conv_hidden'].get(),
                n_layers=n_layers,
                kernel_size=k,
                context_size=self.conv_settings['conv_context'].get(),
                dropout=self.conv_settings['conv_dropout'].get(),
            ).to(self.device)
            self.conv_optimizer = optim.Adam(self.conv_model.parameters(),
                                             lr=self.conv_settings['conv_lr'].get())
            n = self.conv_model.count_params()
            self.log_conv(f"Conv initialized. Latent dim={latent_dim}, "
                          f"params={n:,}, layers={n_layers}, RF={actual_rf}")
            self._refresh_rf()
        except Exception as e:
            traceback.print_exc()
            self.conv_model = None
            self.log_conv(f"Conv init error: {e}")

    def start_conv_training(self):
        if self.conv_model is None:
            self.log_conv("Init conv first."); return
        if not self._latent_cache_valid():
            self.log_conv("Latent cache empty or stale. Rebuilding...")
            self.build_latent_cache()
        if not self._latent_cache_valid():
            self.log_conv("Cannot train: no valid latent cache.")
            return
        if self.training_conv:
            return
        try:
            epochs = int(self.conv_epoch_var.get())
            if epochs < 1:
                raise ValueError
        except Exception:
            self.log_conv("Invalid epochs"); return
        self.training_conv = True
        threading.Thread(target=self._train_conv, args=(epochs,), daemon=True).start()
        self.log_conv(f"Conv training started for {epochs} epochs.")

    def stop_conv_training(self):
        self.training_conv = False
        self.log_conv("Conv training stopped.")

    def _sample_sequences(self, batch_size, seq_len):
        usable = [v for v in self.latent_cache if v.shape[0] >= seq_len + 1]
        if not usable:
            return None
        seqs = []
        max_tries = batch_size * 20
        tries = 0
        while len(seqs) < batch_size and tries < max_tries:
            tries += 1
            v = random.choice(usable)
            T = v.shape[0]
            start = random.randint(0, T - seq_len - 1)
            seqs.append(v[start:start + seq_len + 1])
        if not seqs:
            return None
        while len(seqs) < batch_size:
            seqs.append(seqs[random.randrange(len(seqs))])
        return torch.stack(seqs)

    def _train_conv(self, epochs):
        try:
            bs = self.conv_settings['conv_batch'].get()
            seq_len = self.conv_settings['conv_context'].get()
            steps_per_epoch = self.conv_settings['conv_steps_per_epoch'].get()
            use_residual = self.conv_settings['conv_residual'].get()
            aug_cfg = {k: v.get() for k, v in self.conv_aug.items()}

            lengths = [t.shape[0] for t in self.latent_cache]
            if not lengths:
                self.log_conv("ABORT: empty latent cache.")
                self.training_conv = False
                return
            shortest = min(lengths)
            usable_count = sum(1 for L in lengths if L >= seq_len + 1)
            if usable_count == 0:
                self.log_conv(
                    f"ABORT: shortest video has {shortest} frames but "
                    f"context={seq_len} requires >= {seq_len + 1}. "
                    f"Lower context or load longer video.")
                self.training_conv = False
                return

            t0 = time.time()
            for ep in range(epochs):
                if not self.training_conv:
                    break
                self.conv_model.train()
                total, n = 0.0, 0
                skipped = 0
                for step in range(steps_per_epoch):
                    if not self.training_conv:
                        break
                    batch = self._sample_sequences(bs, seq_len)
                    if batch is None:
                        skipped += 1
                        continue
                    batch = batch.to(self.device)
                    inp = batch[:, :-1]
                    tgt = batch[:, 1:]

                    if (aug_cfg.get('latent_corruption')
                            or aug_cfg.get('latent_multiply')):
                        inp = corrupt_latents(
                            inp,
                            prob=(float(aug_cfg.get('latent_corrupt_prob', 0.0))
                                  if aug_cfg.get('latent_corruption') else 0.0),
                            noise_std=float(aug_cfg.get('latent_noise_std', 0.3)),
                            mask_prob=float(aug_cfg.get('latent_mask_prob', 0.1)),
                            replace_prob=float(aug_cfg.get('latent_zero_prob', 0.05)),
                            multiply=bool(aug_cfg.get('latent_multiply', False)),
                            mult_min=float(aug_cfg.get('latent_mult_min', 1.1)),
                            mult_max=float(aug_cfg.get('latent_mult_max', 2.0)),
                            mult_prob=float(aug_cfg.get('latent_mult_prob', 0.5)),
                        )

                    target = (tgt - inp) if use_residual else tgt
                    pred = self.conv_model(inp)
                    loss = F.mse_loss(pred, target)

                    if not torch.isfinite(loss):
                        skipped += 1
                        self.conv_optimizer.zero_grad()
                        continue

                    self.conv_optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.conv_model.parameters(), 1.0)
                    self.conv_optimizer.step()
                    total += loss.item(); n += 1

                if n == 0:
                    self.log_conv(
                        f"Epoch {ep+1}/{epochs} | no training happened "
                        f"(skipped {skipped}). Check context vs. video length.")
                    self.training_conv = False
                    break

                avg = total / n
                self.log_conv(f"Epoch {ep+1}/{epochs} | MSE: {avg:.6f} | "
                              f"t={time.time()-t0:.1f}s (skipped {skipped})")

            self.training_conv = False
            self.log_conv("Conv training finished.")
        except Exception as e:
            traceback.print_exc()
            self.log_conv(f"Conv train error: {e}")
            self.training_conv = False

    def preview_conv_rollout(self):
        """Autoregressive rollout. Top row: source + 7 GT next frames (real).
        Bottom row: source + 7 model predictions, each fed back in as input."""
        if not self.conv_model or not self.latent_cache or self.ae_model is None:
            self.log_conv("Need conv + latent cache + AE.")
            return
        try:
            use_res = self.conv_settings['conv_residual'].get()

            usable = [v for v in self.latent_cache if v.shape[0] >= 8]
            if not usable:
                self.log_conv("No video with >= 8 frames for preview.")
                return
            v = random.choice(usable)
            total = v.shape[0]

            N_PRED = 7
            start = random.randint(0, total - (N_PRED + 1))
            window = v[start:start + N_PRED + 1].to(self.device)

            self.conv_model.eval()
            self.ae_model.eval()
            with torch.no_grad():
                rollout = [window[0]]
                for i in range(N_PRED):
                    seq = torch.stack(rollout).unsqueeze(0)
                    delta = self.conv_model(seq)[0, -1]
                    z_next = rollout[-1] + delta if use_res else delta
                    rollout.append(z_next)
                pred_latents = torch.stack(rollout[1:])

                source_img = self.ae_model.decode(window[0:1])[0]
                gt_imgs = self.ae_model.decode(window[1:])
                pr_imgs = self.ae_model.decode(pred_latents)

            for i in range(N_PRED):
                mse = F.mse_loss(rollout[i + 1], window[i + 1]).item()
                self.log_conv(f"  rollout step {i+1}: latent MSE={mse:.4f}")

            def to_pil(t):
                arr = ((t + 1) / 2).clamp(0, 1).cpu().numpy()
                arr = (arr.transpose(1, 2, 0) * 255).astype(np.uint8)
                return Image.fromarray(arr)

            source_pil = to_pil(source_img)
            gt_pils = [to_pil(gt_imgs[i]) for i in range(N_PRED)]
            pr_pils = [to_pil(pr_imgs[i]) for i in range(N_PRED)]

            cell = 96
            cols = 4
            rows = 4
            RED = (220, 40, 40)

            grid = Image.new('RGB', (cell * cols, cell * rows), (25, 25, 25))
            draw = ImageDraw.Draw(grid)

            def paste_tile(pil, col, row, border_color=None):
                x = col * cell
                y = row * cell
                tile = pil.resize((cell, cell), Image.NEAREST)
                grid.paste(tile, (x, y))
                if border_color is not None:
                    draw.rectangle([x, y, x + cell - 1, y + cell - 1],
                                   outline=border_color, width=3)

            paste_tile(source_pil, 0, 0, RED)
            for i in range(N_PRED):
                col = (i + 1) % 4
                row = (i + 1) // 4
                paste_tile(gt_pils[i], col, row)

            paste_tile(source_pil, 0, 2, RED)
            for i in range(N_PRED):
                col = (i + 1) % 4
                row = 2 + (i + 1) // 4
                paste_tile(pr_pils[i], col, row)

            self.conv_preview_canvas.config(width=cell * cols, height=cell * rows)
            self._conv_preview_photo = ImageTk.PhotoImage(grid)
            self.conv_preview_canvas.delete("all")
            self.conv_preview_canvas.create_image(0, 0, anchor='nw',
                                                  image=self._conv_preview_photo)
        except Exception as e:
            traceback.print_exc()
            self.log_conv(f"Preview error: {e}")

    def save_conv(self):
        if not self.conv_model:
            return
        fname = filedialog.asksaveasfilename(defaultextension=".pth",
                                             filetypes=[("PyTorch", "*.pth")])
        if not fname:
            return
        meta = {
            'latent_channels': self.ae_settings['ae_latent_channels'].get(),
            'latent_h': self.ae_settings['ae_latent_h'].get(),
            'latent_w': self.ae_settings['ae_latent_w'].get(),
            'img_size': self.global_settings['img_size'].get(),
            'residual': self.conv_settings['conv_residual'].get(),
        }
        self.conv_model.save(fname, meta=meta)
        self.log_conv(f"Saved conv: {fname}")

    def load_conv(self):
        fname = filedialog.askopenfilename(filetypes=[("PyTorch", "*.pth")])
        if not fname:
            return
        try:
            model, meta = LatentConvPredictor.load(fname, map_location='cpu')
            if self.ae_model is not None and model.latent_dim != self.ae_model.latent_dim:
                if not messagebox.askyesno(
                    "Latent dimension mismatch",
                    f"Checkpoint latent_dim={model.latent_dim} but current AE "
                    f"latent_dim={self.ae_model.latent_dim}. Continue anyway?"):
                    return
            self.conv_model = model.to(self.device)
            self.conv_optimizer = optim.Adam(self.conv_model.parameters(),
                                             lr=self.conv_settings['conv_lr'].get())
            if meta:
                self.conv_settings['conv_hidden'].set(model.hidden_size)
                self.conv_settings['conv_n_layers'].set(model.n_layers)
                self.conv_settings['conv_kernel'].set(model.kernel_size)
                self.conv_settings['conv_context'].set(model.context_size)
                if 'residual' in meta:
                    self.conv_settings['conv_residual'].set(meta['residual'])
            self.log_conv(f"Loaded conv: {fname}")
            self._refresh_rf()
        except Exception as e:
            traceback.print_exc()
            self.log_conv(f"Conv load error: {e}")

    def start_generation(self):
        if not self.conv_model or self.ae_model is None:
            messagebox.showerror("Error", "Need AE + conv.")
            return
        if self.generating:
            return
        try:
            n = int(self.gen_settings['n_frames'].get())
            if n < 1:
                raise ValueError
        except Exception:
            messagebox.showerror("Error", "Invalid frame count.")
            return

        self.pause_preview()
        self.last_generated_frames = []

        self.generating = True
        self.gen_btn.config(state=tk.DISABLED)
        self.stop_gen_btn.config(state=tk.NORMAL)
        self.gen_info.config(text="Generating...")
        threading.Thread(target=self._gen_thread, daemon=True).start()

    def stop_generation(self):
        self.generating = False

    def _gen_thread(self):
        try:
            n_frames = int(self.gen_settings['n_frames'].get())
            noise_std = float(self.gen_settings['temp_noise'].get())
            use_dist = self.gen_settings['use_latent_dist'].get()
            use_res = self.conv_settings['conv_residual'].get()
            ctx = self.conv_settings['conv_context'].get()
            latent_dim = self.ae_model.latent_dim

            if use_dist and self.latent_mean is not None:
                z0 = self.latent_mean + torch.randn(latent_dim) * self.latent_std
            else:
                z0 = torch.randn(latent_dim)
            z0 = z0.to(self.device).unsqueeze(0)

            self.conv_model.eval()
            self.ae_model.eval()

            with torch.no_grad():
                frame0 = self.ae_model.decode(z0)
            frame0 = ((frame0[0] + 1) / 2).clamp(0, 1).cpu()
            frame0_pil = Image.fromarray(
                (frame0.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            )

            self.last_generated_frames = [frame0_pil]
            self.gen_q.put(('frame', (frame0_pil, 1, n_frames)))

            history = [z0[0]]
            for t in range(1, n_frames):
                if not self.generating:
                    break
                with torch.no_grad():
                    seq = torch.stack(history[-ctx:]).unsqueeze(0)
                    delta = self.conv_model(seq)[0, -1]
                    z_next = history[-1] + delta if use_res else delta
                    if noise_std > 0:
                        z_next = z_next + noise_std * torch.randn_like(z_next)
                    history.append(z_next)
                    frame = self.ae_model.decode(z_next.unsqueeze(0))
                f = ((frame[0] + 1) / 2).clamp(0, 1).cpu()
                pil = Image.fromarray(
                    (f.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                )
                self.last_generated_frames.append(pil)
                if t % 2 == 0 or t == n_frames - 1:
                    self.gen_q.put(('frame', (pil, t + 1, n_frames)))

            self.generating = False
            self.gen_q.put(('done', len(self.last_generated_frames)))
        except Exception as e:
            traceback.print_exc()
            self.generating = False
            self.gen_q.put(('error', str(e)))

    def _show_gen_frame(self, payload):
        pil, idx, total = payload
        display = pil.resize((self.DISPLAY_SIZE, self.DISPLAY_SIZE), Image.NEAREST)
        self._gen_photo = ImageTk.PhotoImage(display)
        self.gen_preview_canvas.delete("all")
        self.gen_preview_canvas.create_image(160, 160, image=self._gen_photo)
        self.gen_info.config(text=f"Frame {idx}/{total}")

    def _on_gen_done(self, n):
        self.gen_btn.config(state=tk.NORMAL)
        self.stop_gen_btn.config(state=tk.DISABLED)
        self.gen_info.config(text=f"Generated {n} frames.")
        self.log_gen(f"Generation complete: {n} frames.")
        try:
            if self.notebook.index(self.notebook.select()) == self.gen_tab_index:
                self.play_preview()
        except Exception:
            pass

    def play_preview(self):
        if not self.last_generated_frames:
            return
        try:
            if self.notebook.index(self.notebook.select()) != self.gen_tab_index:
                self._preview_was_playing = True
                return
        except Exception:
            pass

        self.preview_playing = True
        if self.preview_after_id is not None:
            try:
                self.root.after_cancel(self.preview_after_id)
            except Exception:
                pass
            self.preview_after_id = None
        self._preview_idx = 0
        self._step_preview()

    def pause_preview(self):
        self.preview_playing = False
        if self.preview_after_id is not None:
            try:
                self.root.after_cancel(self.preview_after_id)
            except Exception:
                pass
            self.preview_after_id = None

    def _step_preview(self):
        if not self.preview_playing or not self.last_generated_frames:
            return
        try:
            if self.notebook.index(self.notebook.select()) != self.gen_tab_index:
                self.preview_playing = False
                self._preview_was_playing = True
                self.preview_after_id = None
                return
        except Exception:
            pass

        idx = self._preview_idx % len(self.last_generated_frames)
        pil = self.last_generated_frames[idx]
        display = pil.resize((self.DISPLAY_SIZE, self.DISPLAY_SIZE), Image.NEAREST)
        self._preview_photo = ImageTk.PhotoImage(display)
        self.gen_preview_canvas.delete("all")
        self.gen_preview_canvas.create_image(160, 160, image=self._preview_photo)
        self._preview_idx += 1
        fps = max(1, int(self.gen_settings['fps'].get()))
        delay = max(1, int(1000 / fps))
        self.preview_after_id = self.root.after(delay, self._step_preview)

    def save_gif(self):
        if not self.last_generated_frames:
            messagebox.showwarning("No frames", "Generate first.")
            return
        fname = filedialog.asksaveasfilename(defaultextension=".gif",
                                             filetypes=[("GIF", "*.gif")])
        if not fname:
            return
        try:
            fps = max(1, int(self.gen_settings['fps'].get()))
            save_frames_as_gif(self.last_generated_frames, fname, fps=fps)
            self.log_gen(f"Saved GIF: {fname} ({fps} fps)")
        except Exception as e:
            traceback.print_exc()
            self.log_gen(f"GIF save error: {e}")

    def save_mp4(self):
        if not self.last_generated_frames:
            messagebox.showwarning("No frames", "Generate first.")
            return
        if not HAS_IMAGEIO:
            messagebox.showerror("Missing",
                                 "imageio not installed. Try: pip install imageio[ffmpeg]")
            return
        fname = filedialog.asksaveasfilename(defaultextension=".mp4",
                                             filetypes=[("MP4", "*.mp4")])
        if not fname:
            return
        try:
            fps = max(1, int(self.gen_settings['fps'].get()))
            save_frames_as_mp4(self.last_generated_frames, fname, fps=fps)
            self.log_gen(f"Saved MP4: {fname} ({fps} fps)")
        except Exception as e:
            traceback.print_exc()
            self.log_gen(f"MP4 save error: {e}")

    def save_png_seq(self):
        if not self.last_generated_frames:
            messagebox.showwarning("No frames", "Generate first.")
            return
        folder = filedialog.askdirectory()
        if not folder:
            return
        try:
            save_frames_as_png_sequence(self.last_generated_frames, folder)
            self.log_gen(f"Saved PNG sequence to: {folder}")
        except Exception as e:
            traceback.print_exc()
            self.log_gen(f"PNG sequence error: {e}")

    def _on_close(self):
        self.pause_preview()
        self.training_ae = False
        self.training_conv = False
        self.generating = False
        try:
            self.root.destroy()
        except Exception:
            pass


if __name__ == "__main__":
    root = tk.Tk()
    app = VideoConvApp(root)
    root.mainloop()