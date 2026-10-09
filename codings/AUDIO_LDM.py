import tkinter as tk
from tkinter import filedialog, ttk
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as F_vision
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
import librosa
import soundfile as sf
import tempfile
import warnings
from playsound3 import playsound

warnings.filterwarnings('ignore')


# ==================== Helpers ====================

def text_to_indices(text, max_len=128):
    idx = [ord(c) if ord(c) < 256 else 0 for c in text[:max_len]]
    if len(idx) < max_len:
        idx += [0] * (max_len - len(idx))
    return idx


def get_norm(channels):
    """Adaptive GroupNorm - allows base_channels < 32."""
    if channels % 32 == 0:
        return nn.GroupNorm(32, channels)
    best = 1
    for g in range(32, 0, -1):
        if channels % g == 0:
            best = g
            break
    return nn.GroupNorm(best, channels)


# ==================== Audio Processor ====================

class AudioProcessor:
    """
    Converts audio <-> mel-spectrogram images.
    All spectrograms are grayscale PIL images of size (spec_size, spec_size).
    """
    def __init__(self, sr=22050, n_fft=2048, n_mels=128,
                 segment_duration=1.472, spec_size=64, fmax=8000):
        self.sr = sr
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.segment_duration = segment_duration
        self.segment_samples = int(sr * segment_duration)
        self.spec_size = spec_size
        self.fmax = fmax
        # Choose hop_length so we get ≈ spec_size frames before resize
        self.hop_length = max(256, int(self.segment_samples / spec_size))
        self.actual_width = int(self.segment_samples / self.hop_length) + 1

    def audio_to_segments(self, audio_path, max_segments=None):
        """Returns a list of grayscale PIL images (mel-spectrograms)."""
        try:
            audio, sr = librosa.load(audio_path, sr=self.sr)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return []
        if len(audio) == 0:
            return []
        num_segments = max(1, len(audio) // self.segment_samples)
        if max_segments:
            num_segments = min(num_segments, max_segments)

        segs = []
        for i in range(num_segments):
            start = i * self.segment_samples
            end = start + self.segment_samples
            seg = audio[start:end]
            if len(seg) < self.segment_samples:
                seg = np.pad(seg, (0, self.segment_samples - len(seg)), mode='constant')
            try:
                mel = librosa.feature.melspectrogram(
                    y=seg, sr=sr, n_fft=self.n_fft,
                    hop_length=self.hop_length, n_mels=self.n_mels,
                    fmax=self.fmax,
                )
                mel_db = librosa.power_to_db(mel, ref=np.max)
                mn, mx = mel_db.min(), mel_db.max()
                if mx - mn < 1e-8:
                    mel_norm = np.zeros_like(mel_db)
                else:
                    mel_norm = (mel_db - mn) / (mx - mn)
                img = Image.fromarray((mel_norm * 255).astype(np.uint8))
                img = img.resize((self.spec_size, self.spec_size), Image.Resampling.LANCZOS)
                segs.append(img)
            except Exception as e:
                print(f"Spectrogram error: {e}")
                continue
        return segs

    def mel_to_audio(self, mel_image, n_iter=32):
        """Convert a grayscale PIL spectrogram back to audio."""
        try:
            mel_img = np.array(mel_image.convert('L')).astype(float) / 255.0
            target_shape = (self.n_mels, self.actual_width)
            if mel_img.shape != target_shape:
                img = Image.fromarray((mel_img * 255).astype(np.uint8))
                img = img.resize((target_shape[1], target_shape[0]),
                                 Image.Resampling.LANCZOS)
                mel_img = np.array(img).astype(float) / 255.0
            mel_img = np.clip(mel_img, 0, 1)
            mel_linear = librosa.db_to_power(mel_img * 80 - 80)
            audio = librosa.feature.inverse.mel_to_audio(
                mel_linear, sr=self.sr, n_fft=self.n_fft,
                hop_length=self.hop_length, n_iter=n_iter,
            )
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio)) * 0.9
            return audio[:self.segment_samples]
        except Exception as e:
            print(f"mel_to_audio error: {e}")
            return np.zeros(self.segment_samples)

    def save_audio(self, audio, path):
        sf.write(path, audio, self.sr)

    def play_audio(self, audio):
        """Non-blocking audio playback."""
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                temp_path = f.name
            self.save_audio(audio, temp_path)

            def _play():
                try:
                    playsound(temp_path, block=True)
                except Exception as e:
                    print(f"Playback error: {e}")
                finally:
                    try:
                        os.unlink(temp_path)
                    except Exception:
                        pass

            threading.Thread(target=_play, daemon=True).start()
        except Exception as e:
            print(f"play_audio error: {e}")


# ==================== Augmentations ====================

class SpecAugment:
    """SpecAugment-style time/frequency masking."""
    def __init__(self, time_mask=8, freq_mask=8, num_masks=2, p=0.5):
        self.time_mask = time_mask
        self.freq_mask = freq_mask
        self.num_masks = num_masks
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        arr = np.array(img).astype(np.float32)
        H, W = arr.shape[:2]
        for _ in range(self.num_masks):
            t = random.randint(1, max(1, self.time_mask))
            if W - t > 0:
                t0 = random.randint(0, W - t)
                arr[:, t0:t0 + t] = 0
            f = random.randint(1, max(1, self.freq_mask))
            if H - f > 0:
                f0 = random.randint(0, H - f)
                arr[f0:f0 + f, :] = 0
        return Image.fromarray(arr.astype(np.uint8))


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
        out = Image.open(buf).convert(pil.mode)
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


# ==================== Positional Embeddings ====================

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=time.device) * -emb)
        emb = time[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


# ==================== Text Encoders ====================

class TextEncoder(nn.Module):
    def __init__(self, vocab_size=256, embed_dim=64, hidden_size=64,
                 num_layers=2, cond_dim=256):
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


# ==================== Flow UNet ====================

class Conv2dWithPadding(nn.Conv2d):
    def __init__(self, in_c, out_c, k, stride=1, padding_mode='constant', **kw):
        super().__init__(in_c, out_c, k, stride, 0, **kw)
        self.padding_mode = padding_mode

    def forward(self, x):
        ph, pw = self.kernel_size[0] // 2, self.kernel_size[1] // 2
        if ph == 0 and pw == 0:
            return F_nn.conv2d(x, self.weight, self.bias, self.stride, 0,
                               self.dilation, self.groups)
        x = F_nn.pad(x, (pw, pw, ph, ph), mode=self.padding_mode)
        return F_nn.conv2d(x, self.weight, self.bias, self.stride, 0,
                           self.dilation, self.groups)


class AttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.norm = get_norm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        r = x
        x = self.norm(x).view(B, C, H * W).transpose(1, 2)
        x, _ = self.attn(x, x, x)
        return x.transpose(1, 2).view(B, C, H, W) + r


class DownBlock(nn.Module):
    def __init__(self, in_c, out_c, tdim, cdim, drop=0.1, has_attn=False,
                 padding_mode='constant'):
        super().__init__()
        self.conv1 = Conv2dWithPadding(in_c, out_c, 3, padding_mode=padding_mode)
        self.norm1 = get_norm(out_c)
        self.conv2 = Conv2dWithPadding(out_c, out_c, 3, padding_mode=padding_mode)
        self.norm2 = get_norm(out_c)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(tdim, out_c))
        self.cond_proj = nn.Linear(cdim, out_c)
        self.res_conv = (Conv2dWithPadding(in_c, out_c, 1, padding_mode=padding_mode)
                         if in_c != out_c else nn.Identity())
        self.dropout = nn.Dropout(drop)
        self.attn = AttentionBlock(out_c) if has_attn else nn.Identity()

    def forward(self, x, t, c):
        h = self.conv1(x)
        h = self.norm1(h)
        h = h + self.time_mlp(t)[:, :, None, None]
        h = h + self.cond_proj(c)[:, :, None, None]
        h = F_nn.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = F_nn.silu(h)
        return self.attn(h + self.res_conv(x))


class UpBlock(nn.Module):
    def __init__(self, in_c, out_c, tdim, cdim, drop=0.1, has_attn=False,
                 padding_mode='constant'):
        super().__init__()
        self.conv1 = Conv2dWithPadding(in_c, out_c, 3, padding_mode=padding_mode)
        self.norm1 = get_norm(out_c)
        self.conv2 = Conv2dWithPadding(out_c, out_c, 3, padding_mode=padding_mode)
        self.norm2 = get_norm(out_c)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(tdim, out_c))
        self.cond_proj = nn.Linear(cdim, out_c)
        self.res_conv = (Conv2dWithPadding(in_c, out_c, 1, padding_mode=padding_mode)
                         if in_c != out_c else nn.Identity())
        self.dropout = nn.Dropout(drop)
        self.attn = AttentionBlock(out_c) if has_attn else nn.Identity()

    def forward(self, x, t, c):
        h = self.conv1(x)
        h = self.norm1(h)
        h = h + self.time_mlp(t)[:, :, None, None]
        h = h + self.cond_proj(c)[:, :, None, None]
        h = F_nn.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = F_nn.silu(h)
        return self.attn(h + self.res_conv(x))


class VelocityUNet(nn.Module):
    def __init__(self, in_channels=8, base_channels=32, time_emb_dim=128,
                 cond_dim=256, latent_h=4, latent_w=4, channel_mult=(1, 2, 4),
                 dropout=0.1, use_attention=True, padding_mode='constant'):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.time_emb_dim = time_emb_dim
        self.cond_dim = cond_dim
        self.channel_mult = channel_mult
        self.use_attention = use_attention
        self.padding_mode = padding_mode
        self.latent_h = latent_h
        self.latent_w = latent_w

        num_down = 0
        ch, cw = latent_h, latent_w
        while ch >= 2 and cw >= 2 and num_down < len(channel_mult) - 1:
            ch //= 2
            cw //= 2
            num_down += 1
        mult_used = channel_mult[:num_down + 1]

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        self.init_conv = Conv2dWithPadding(in_channels, base_channels, 3,
                                           padding_mode=padding_mode)
        self.downs = nn.ModuleList()
        cur = base_channels
        for i, m in enumerate(mult_used):
            out = base_channels * m
            self.downs.append(DownBlock(cur, out, time_emb_dim, cond_dim, dropout,
                                        has_attn=use_attention,
                                        padding_mode=padding_mode))
            if i < len(mult_used) - 1:
                self.downs.append(nn.Conv2d(out, out, 4, stride=2, padding=1))
            cur = out

        self.mid_block1 = DownBlock(cur, cur, time_emb_dim, cond_dim, dropout,
                                    has_attn=use_attention, padding_mode=padding_mode)
        self.mid_block2 = UpBlock(cur, cur, time_emb_dim, cond_dim, dropout,
                                  has_attn=use_attention, padding_mode=padding_mode)

        self.ups = nn.ModuleList()
        rev = list(reversed(mult_used))
        for i, m in enumerate(rev):
            out = base_channels * m
            self.ups.append(UpBlock(cur + out, out, time_emb_dim, cond_dim, dropout,
                                    has_attn=use_attention,
                                    padding_mode=padding_mode))
            if i < len(rev) - 1:
                self.ups.append(nn.ConvTranspose2d(out, out, 4, stride=2, padding=1))
            cur = out

        self.final_conv = nn.Sequential(
            get_norm(cur), nn.SiLU(),
            Conv2dWithPadding(cur, in_channels, 3, padding_mode=padding_mode),
        )

    def forward(self, x, t, cond=None):
        te = self.time_mlp(t)
        x = self.init_conv(x)
        skips = []
        for layer in self.downs:
            if isinstance(layer, DownBlock):
                x = layer(x, te, cond)
                skips.append(x)
            else:
                x = layer(x)
        x = self.mid_block1(x, te, cond)
        x = self.mid_block2(x, te, cond)
        for layer in self.ups:
            if isinstance(layer, UpBlock):
                x = torch.cat([x, skips.pop()], dim=1)
                x = layer(x, te, cond)
            else:
                x = layer(x)
        return self.final_conv(x)


# ==================== VAE (with AE / VAE switch) ====================

VAE_SIZE_CONFIGS = {
    'tiny':   (2, [0.25, 0.5]),
    'small':  (3, [0.25, 0.5, 1.0]),
    'medium': (3, [0.5, 1.0, 2.0]),
    'big':    (3, [1.0, 2.0, 4.0]),
    'large':  (3, [2.0, 4.0, 8.0]),
}


def _vae_num_blocks(img_size, latent_h, latent_w):
    if img_size % latent_h or img_size % latent_w:
        raise ValueError(
            f"img_size={img_size} must be divisible by latent {latent_h}x{latent_w}")
    rh, rw = img_size // latent_h, img_size // latent_w
    if rh != rw:
        raise ValueError(f"non-square downsampling ratio {rh}:{rw} not supported")
    n = int(round(math.log2(rh)))
    if 2 ** n != rh:
        raise ValueError(f"ratio {rh} is not a power of 2")
    if n < 1:
        raise ValueError(f"need at least 1 downsampling block, got ratio {rh}")
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
    def __init__(self, in_channels=1, base_channels=32, latent_channels=8,
                 latent_h=8, latent_w=8, size='big', img_size=64):
        super().__init__()
        _, mults = VAE_SIZE_CONFIGS[size]
        n = _vae_num_blocks(img_size, latent_h, latent_w)
        mults = _fit_mults(mults, n)
        self.latent_h, self.latent_w = latent_h, latent_w

        self.down = nn.ModuleList()
        c = in_channels
        for m in mults:
            oc = max(1, int(base_channels * m))
            self.down.append(nn.Conv2d(c, oc, 3, stride=2, padding=1))
            self.down.append(nn.LeakyReLU(0.2))
            c = oc
        self.conv_mu = nn.Conv2d(c, latent_channels, 3, padding=1)
        self.conv_logvar = nn.Conv2d(c, latent_channels, 3, padding=1)

    def forward(self, x):
        for l in self.down:
            x = l(x)
        return self.conv_mu(x), self.conv_logvar(x)


class FlexDecoder(nn.Module):
    def __init__(self, latent_channels=8, base_channels=32, out_channels=1,
                 latent_h=8, latent_w=8, size='big', img_size=64):
        super().__init__()
        _, mults = VAE_SIZE_CONFIGS[size]
        n = _vae_num_blocks(img_size, latent_h, latent_w)
        mults = _fit_mults(mults, n)
        widths = [max(1, int(base_channels * m)) for m in mults]
        self.latent_h, self.latent_w = latent_h, latent_w

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
            out = F_nn.interpolate(out, size=target_size, mode='bilinear',
                                   align_corners=False)
        return out


class FlexVAE(nn.Module):
    """
    Flexible autoencoder for spectrograms.
    mode='AE'  -> deterministic autoencoder (encode returns mu, no sampling)
    mode='VAE' -> probabilistic VAE (reparameterization, KL loss)
    """
    def __init__(self, in_channels=1, base_channels=32, latent_channels=8,
                 latent_h=8, latent_w=8, size='big', img_size=64, mode='AE'):
        super().__init__()
        self.encoder = FlexEncoder(in_channels, base_channels, latent_channels,
                                   latent_h, latent_w, size, img_size)
        self.decoder = FlexDecoder(latent_channels, base_channels, in_channels,
                                   latent_h, latent_w, size, img_size)
        self.latent_channels = latent_channels
        self.latent_h = latent_h
        self.latent_w = latent_w
        self.size = size
        self.img_size = img_size
        self.mode = mode.upper()
        assert self.mode in ('AE', 'VAE'), "mode must be 'AE' or 'VAE'"

    def reparameterize(self, mu, logvar):
        if self.mode == 'AE':
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z, (x.shape[2], x.shape[3])), mu, logvar

    def encode(self, x, deterministic=True):
        """
        For AE mode, always returns mu.
        For VAE mode, returns mu if deterministic, otherwise a sample.
        """
        mu, logvar = self.encoder(x)
        if self.mode == 'AE' or deterministic:
            return mu
        return self.reparameterize(mu, logvar)

    def decode(self, z, target_size=None):
        return self.decoder(z, target_size)


# ==================== Rectified Flow Latent ====================

@torch.no_grad()
def match_ot_batch(z0, z1):
    B = z0.shape[0]
    if B <= 1:
        return z0, z1, torch.arange(B)
    a = z0.view(B, -1)
    b = z1.view(B, -1)
    dist = (a.pow(2).sum(1, keepdim=True) + b.pow(2).sum(1, keepdim=True).T
            - 2 * a @ b.T)
    try:
        from scipy.optimize import linear_sum_assignment
        _, col = linear_sum_assignment(dist.cpu().numpy())
        col = torch.from_numpy(col).to(z1.device)
        return z0, z1[col], col
    except Exception:
        return z0, z1, torch.arange(B, device=z1.device)


class RectifiedFlowLatent:
    def __init__(self, latent_channels=8, latent_h=8, latent_w=8, base_channels=32,
                 cond_dim=256, device=None, use_attention=True,
                 padding_mode='constant', channel_mult=(1, 2, 4)):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.latent_h, self.latent_w = latent_h, latent_w
        self.cond_dim = cond_dim
        self.use_ot = False
        self.model = VelocityUNet(
            in_channels=latent_channels, base_channels=base_channels,
            time_emb_dim=max(64, base_channels * 4), cond_dim=cond_dim,
            latent_h=latent_h, latent_w=latent_w, channel_mult=channel_mult,
            dropout=0.1, use_attention=use_attention, padding_mode=padding_mode,
        ).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=2e-4)
        self.ema_model = None
        self.ema_decay = 0.999
        self.ema_mode = 'standard'
        self.use_ema = False

    def set_ema(self, decay=0.999, mode='standard'):
        self.ema_decay, self.ema_mode, self.use_ema = decay, mode, True
        self.ema_model = VelocityUNet(
            in_channels=self.model.in_channels,
            base_channels=self.model.base_channels,
            time_emb_dim=self.model.time_emb_dim,
            cond_dim=self.model.cond_dim,
            latent_h=self.latent_h, latent_w=self.latent_w,
            channel_mult=self.model.channel_mult, dropout=0.1,
            use_attention=self.model.use_attention,
            padding_mode=self.model.padding_mode,
        ).to(self.device)
        self.ema_model.load_state_dict(self.model.state_dict())
        for p in self.ema_model.parameters():
            p.requires_grad = False

    def update_ema(self):
        if self.ema_model is None:
            return
        with torch.no_grad():
            for p, ep in zip(self.model.parameters(), self.ema_model.parameters()):
                ep.data.mul_(self.ema_decay).add_(p.data, alpha=1 - self.ema_decay)
        if self.ema_mode == 'lookahead':
            with torch.no_grad():
                for p, ep in zip(self.model.parameters(), self.ema_model.parameters()):
                    p.data.copy_(ep.data)

    def train_step(self, z0, z1, cond=None, cfg_dropout_prob=0.0):
        B = z0.size(0)
        z0 = z0.to(self.device)
        z1 = z1.to(self.device)
        if self.use_ot and B > 1:
            z0, z1, col = match_ot_batch(z0, z1)
            if cond is not None:
                cond = cond[col]
        if cond is None:
            cond = torch.zeros(B, self.cond_dim, device=self.device)
        else:
            cond = cond.to(self.device)
        t = torch.rand(B, device=self.device)
        zt = t.view(-1, 1, 1, 1) * z1 + (1 - t.view(-1, 1, 1, 1)) * z0
        target = z1 - z0
        if cfg_dropout_prob > 0:
            mask = (torch.rand(B, 1, device=self.device) > cfg_dropout_prob).float()
            cond = cond * mask
        pred = self.model(zt, t, cond=cond)
        loss = self.criterion(pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.use_ema:
            self.update_ema()
        return loss.item()

    def _v_fn_factory(self, model, n, cond, cfg_scale):
        null = torch.zeros_like(cond)

        def v_fn(t_val, z_val):
            tt = torch.full((n,), t_val, device=self.device)
            if cfg_scale != 1.0:
                vc = model(z_val, tt, cond=cond)
                vu = model(z_val, tt, cond=null)
                return vu + cfg_scale * (vc - vu)
            return model(z_val, tt, cond=cond)
        return v_fn

    def _step(self, method, t_cur, dt, z, v_fn):
        if method == 'euler':
            z = z + v_fn(t_cur, z) * dt
        elif method == 'heun':
            v1 = v_fn(t_cur, z)
            zp = z + v1 * dt
            v2 = v_fn(t_cur + dt, zp)
            z = z + (v1 + v2) * (dt / 2)
        elif method == 'midpoint':
            zm = z + v_fn(t_cur, z) * (dt / 2)
            z = z + v_fn(t_cur + dt / 2, zm) * dt
        elif method == 'rk4':
            v1 = v_fn(t_cur, z)
            v2 = v_fn(t_cur + dt / 2, z + v1 * dt / 2)
            v3 = v_fn(t_cur + dt / 2, z + v2 * dt / 2)
            v4 = v_fn(t_cur + dt, z + v3 * dt)
            z = z + (v1 + 2 * v2 + 2 * v3 + v4) * dt / 6
        else:
            raise ValueError(f"Unknown method {method}")
        return z

    @torch.no_grad()
    def sample(self, n_samples=16, cond=None, steps=50, method='euler',
               cfg_scale=1.0, use_ema=True, start_from=None):
        model = (self.ema_model
                 if (use_ema and self.use_ema and self.ema_model is not None)
                 else self.model)
        shape = (n_samples, self.model.in_channels, self.latent_h, self.latent_w)
        z = (start_from.to(self.device) if start_from is not None
             else torch.randn(*shape, device=self.device))
        if cond is None:
            cond = torch.zeros(n_samples, self.cond_dim, device=self.device)
        else:
            cond = cond.to(self.device)
        dt = 1.0 / steps
        times = torch.linspace(0, 1, steps + 1, device=self.device)
        v_fn = self._v_fn_factory(model, n_samples, cond, cfg_scale)
        for i in range(steps):
            z = self._step(method, times[i], dt, z, v_fn)
        return z

    @torch.no_grad()
    def sample_step_by_step(self, n_samples=16, cond=None, steps=50,
                            method='euler', cfg_scale=1.0, use_ema=True,
                            start_from=None):
        model = (self.ema_model
                 if (use_ema and self.use_ema and self.ema_model is not None)
                 else self.model)
        shape = (n_samples, self.model.in_channels, self.latent_h, self.latent_w)
        z = (start_from.to(self.device) if start_from is not None
             else torch.randn(*shape, device=self.device))
        if cond is None:
            cond = torch.zeros(n_samples, self.cond_dim, device=self.device)
        else:
            cond = cond.to(self.device)
        dt = 1.0 / steps
        times = torch.linspace(0, 1, steps + 1, device=self.device)
        v_fn = self._v_fn_factory(model, n_samples, cond, cfg_scale)
        for i in range(steps):
            z = self._step(method, times[i], dt, z, v_fn)
            yield i + 1, z.clone()


# ==================== Audio Dataset ====================

class AudioSpectrogramDataset(Dataset):
    """
    Loads audio files, splits into fixed-length segments, converts each
    segment to a grayscale mel-spectrogram image, and returns
    (spectrogram_tensor, text_indices).
    """
    def __init__(self, audio_paths, labels_per_file, spec_size=64,
                 segment_duration=1.472, aug_settings=None,
                 text_max_len=128, audio_processor=None, verbose=False):
        self.spec_size = spec_size
        self.text_max_len = text_max_len
        self.aug_settings = aug_settings or {}
        self.ap = audio_processor or AudioProcessor(
            segment_duration=segment_duration, spec_size=spec_size)

        self.spectrograms = []
        self.labels = []
        for i, path in enumerate(audio_paths):
            segs = self.ap.audio_to_segments(path)
            lbls = labels_per_file[i] if i < len(labels_per_file) else ['']
            if isinstance(lbls, str):
                lbls = [lbls]
            for s in segs:
                self.spectrograms.append(s)
                self.labels.append(lbls)
        if verbose:
            print(f"Loaded {len(self.spectrograms)} spectrograms from "
                  f"{len(audio_paths)} audio files.")

        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

        a = self.aug_settings
        self.aug_pipeline = []
        if a.get('flip_horizontal', False):
            self.aug_pipeline.append(transforms.RandomHorizontalFlip(p=0.5))
        if a.get('rotation', False):
            self.aug_pipeline.append(transforms.RandomRotation(
                15, interpolation=Image.BICUBIC, expand=False, fill=0))
        if a.get('random_crop', False):
            sc = a.get('crop_scale', 0.8)
            self.aug_pipeline.append(transforms.RandomResizedCrop(
                size=spec_size, scale=(sc, 1.0), interpolation=Image.BICUBIC))
        if a.get('color_jitter', False):
            self.aug_pipeline.append(transforms.ColorJitter(
                brightness=a.get('brightness', 0.2),
                contrast=a.get('contrast', 0.2)))
        if a.get('elastic_transform', False):
            self.aug_pipeline.append(ElasticTransform(
                alpha=a.get('elastic_alpha', 30),
                sigma=a.get('elastic_sigma', 3), p=0.5))
        if a.get('jpeg_compression', False):
            self.aug_pipeline.append(RandomJPEG(
                quality_low=a.get('jpeg_quality_low', 50),
                quality_high=a.get('jpeg_quality_high', 95), p=0.5))
        if a.get('spec_augment', False):
            self.aug_pipeline.append(SpecAugment(
                time_mask=a.get('spec_time_mask', 8),
                freq_mask=a.get('spec_freq_mask', 8),
                num_masks=a.get('spec_num_masks', 2), p=0.5))

    def __len__(self):
        return len(self.spectrograms)

    def apply_aug(self, pil):
        img = pil.copy()
        for a in self.aug_pipeline:
            img = a(img)
        return img

    def __getitem__(self, idx):
        try:
            pil = self.apply_aug(self.spectrograms[idx])
            t = self.base_transform(pil)
            caps = self.labels[idx]
            txt = random.choice(caps).replace('_', ' ')
            ti = text_to_indices(txt, self.text_max_len)
            return t, torch.tensor(ti, dtype=torch.long)
        except Exception as e:
            print(f"Dataset error at {idx}: {e}")
            return (torch.zeros(1, self.spec_size, self.spec_size),
                    torch.zeros(self.text_max_len, dtype=torch.long))


# ==================== GUI ====================

def default_aug_dict():
    return {
        'flip_horizontal': tk.BooleanVar(value=False),
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
        'spec_augment': tk.BooleanVar(value=False),
        'spec_time_mask': tk.IntVar(value=8),
        'spec_freq_mask': tk.IntVar(value=8),
        'spec_num_masks': tk.IntVar(value=2),
    }


AUG_DEFAULTS = {
    'flip_horizontal': False, 'rotation': False, 'random_crop': False,
    'crop_scale': 0.8,
    'color_jitter': False, 'brightness': 0.2, 'contrast': 0.2,
    'saturation': 0.2, 'hue': 0.1,
    'random_perspective': False, 'perspective_distortion': 0.1,
    'elastic_transform': False, 'elastic_alpha': 30, 'elastic_sigma': 3,
    'jpeg_compression': False, 'jpeg_quality_low': 50, 'jpeg_quality_high': 95,
    'stretch_vertical': False, 'stretch_horizontal': False,
    'spec_augment': False, 'spec_time_mask': 8, 'spec_freq_mask': 8,
    'spec_num_masks': 2,
}


class RectifiedFlowApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Rectified Flow Audio LDM")

        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.root.geometry(f"{max(1000, int(sw * 0.8))}x{max(700, int(sh * 0.8))}")
        self.root.minsize(900, 650)

        try:
            import ctypes
            a = ctypes.c_int()
            ctypes.windll.shcore.GetProcessDpiAwareness(0, ctypes.byref(a))
            if a.value == 0:
                ctypes.windll.shcore.SetProcessDpiAwareness(1)
            dpi = ctypes.windll.user32.GetDpiForWindow(root.winfo_id())
            self.root.tk.call('tk', 'scaling', dpi / 72.0)
        except Exception:
            pass

        self.audio_paths = []
        self.labels = []
        self.csv_path = None

        self.vae_model = None
        self.vae_optimizer = None
        self.rectified_model = None
        self.text_encoder = None

        self.training_vae = False
        self.training_rectified = False
        self.progressive_active = False

        self.message_queue_vae = queue.Queue()
        self.message_queue_rectified = queue.Queue()

        # Global settings
        self.global_settings = {
            'img_size': tk.IntVar(value=64),          # spectrogram size
            'segment_duration': tk.DoubleVar(value=1.472),
            'sample_rate': tk.IntVar(value=22050),
            'n_mels': tk.IntVar(value=128),
        }

        self.vae_settings = {
            'vae_size': tk.StringVar(value='big'),
            'vae_mode': tk.StringVar(value='AE'),     # <-- NEW: AE or VAE
            'vae_base_channels': tk.IntVar(value=32),
            'vae_latent_channels': tk.IntVar(value=8),
            'vae_latent_h': tk.IntVar(value=8),
            'vae_latent_w': tk.IntVar(value=8),
            'vae_batch_size': tk.IntVar(value=16),
            'vae_lr': tk.DoubleVar(value=1e-3),
            'vae_num_workers': tk.IntVar(value=0),
            'vae_kl_weight': tk.DoubleVar(value=0.0001),
        }

        self.flow_settings = {
            'flow_base_channels': tk.IntVar(value=32),
            'flow_use_attention': tk.BooleanVar(value=True),
            'flow_padding_mode': tk.StringVar(value='constant'),
            'flow_batch_size': tk.IntVar(value=16),
            'flow_lr': tk.DoubleVar(value=2e-4),
            'flow_cfg_dropout': tk.DoubleVar(value=0.1),
            'flow_use_ot': tk.BooleanVar(value=False),
            'ema_enabled': tk.BooleanVar(value=False),
            'ema_decay': tk.DoubleVar(value=0.999),
            'ema_mode': tk.StringVar(value='standard'),
            'preview_enabled': tk.BooleanVar(value=True),
            'preview_epoch_freq': tk.IntVar(value=5),
            'preview_steps': tk.IntVar(value=20),
            'preview_method': tk.StringVar(value='heun'),
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
        self.flow_aug_settings = default_aug_dict()

        self.ode_method = tk.StringVar(value='euler')
        self.ode_steps = tk.IntVar(value=50)
        self.cfg_scale = tk.DoubleVar(value=2.0)
        self.thumbnail_size = 128

        self.setup_gui()
        self.root.after(100, self.process_messages_vae)
        self.root.after(100, self.process_messages_rectified)

    # ---------------- GUI ----------------
    def setup_gui(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        tabs = [
            ('Dataset',              self.setup_dataset_tab),
            ('VAE Settings',         self.setup_vae_settings_tab),
            ('VAE Training',         self.setup_vae_training_tab),
            ('Flow Settings',        self.setup_flow_settings_tab),
            ('Flow Training',        self.setup_flow_training_tab),
            ('Augmentation Settings', self.setup_augmentation_tab),
            ('Generation',           self.setup_generation_tab),
        ]
        for name, builder in tabs:
            frame = ttk.Frame(self.notebook)
            self.notebook.add(frame, text=name)
            builder(frame)

        self.status_label = tk.Label(self.root, text="Ready", relief=tk.SUNKEN,
                                     anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    def _spin(self, parent, label, var, low, high):
        f = tk.Frame(parent)
        f.pack(fill=tk.X, pady=1)
        tk.Label(f, text=label, width=20, anchor='w').pack(side=tk.LEFT)
        ttk.Spinbox(f, from_=low, to=high, textvariable=var,
                    width=10).pack(side=tk.RIGHT)

    def _entry(self, parent, label, var):
        f = tk.Frame(parent)
        f.pack(fill=tk.X, pady=1)
        tk.Label(f, text=label, width=20, anchor='w').pack(side=tk.LEFT)
        tk.Entry(f, textvariable=var, width=12).pack(side=tk.RIGHT)

    def _combo(self, parent, label, var, values):
        f = tk.Frame(parent)
        f.pack(fill=tk.X, pady=1)
        tk.Label(f, text=label, width=20, anchor='w').pack(side=tk.LEFT)
        ttk.Combobox(f, textvariable=var, values=values, state='readonly',
                     width=12).pack(side=tk.RIGHT)

    def _make_scrollable(self, parent):
        canvas = tk.Canvas(parent)
        sb = tk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        inner = tk.Frame(canvas)
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        return inner

    # ---------------- Dataset tab ----------------
    def setup_dataset_tab(self, tab):
        main = tk.Frame(tab)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        left = tk.Frame(main, width=300)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left.pack_propagate(False)

        tk.Label(left, text="Audio Dataset",
                 font=("Arial", 12, "bold")).pack(pady=(0, 10))

        imgf = tk.LabelFrame(left, text="Training Audio", padx=5, pady=5)
        imgf.pack(fill=tk.X, pady=(0, 5))
        tk.Button(imgf, text="Add Audio Files", command=self.add_audio_files,
                  width=22).pack(pady=1)
        tk.Button(imgf, text="Add Folder (recursive)",
                  command=self.add_audio_folder, width=22).pack(pady=1)
        tk.Button(imgf, text="Clear All", command=self.clear_audio,
                  width=22).pack(pady=1)
        self.image_listbox = tk.Listbox(imgf, height=12)
        self.image_listbox.pack(fill=tk.X, pady=2)

        cf = tk.LabelFrame(left, text="Labels / Captions", padx=5, pady=5)
        cf.pack(fill=tk.X, pady=5)
        tk.Button(cf, text="Load CSV (filename,label)",
                  command=self.load_csv, width=22).pack(pady=1)
        tk.Button(cf, text="Use filenames as labels",
                  command=self.use_filenames_as_labels, width=22).pack(pady=1)
        tk.Button(cf, text="Use folder names as labels",
                  command=self.use_folders_as_labels, width=22).pack(pady=1)
        self.csv_status = tk.Label(cf, text="No CSV loaded", fg="red")
        self.csv_status.pack()

        right = tk.Frame(main)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        lf = tk.LabelFrame(right, text="Log", padx=5, pady=5)
        lf.pack(fill=tk.BOTH, expand=True)
        self.dataset_log_text = tk.Text(lf, height=20, font=("Courier", 9))
        self.dataset_log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = tk.Scrollbar(lf, command=self.dataset_log_text.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.dataset_log_text.config(yscrollcommand=sb.set)

    # ---------------- VAE Settings tab ----------------
    def setup_vae_settings_tab(self, tab):
        inner = self._make_scrollable(tab)
        wrap = tk.Frame(inner)
        wrap.pack(fill=tk.X, padx=20, pady=20)

        tk.Label(wrap, text="VAE / AE Settings",
                 font=("Arial", 14, "bold")).pack(pady=(0, 15))

        mf = tk.LabelFrame(wrap, text="Model", padx=10, pady=10)
        mf.pack(fill=tk.X, pady=5)
        self._combo(mf, "VAE size:", self.vae_settings['vae_size'],
                    ['tiny', 'small', 'medium', 'big', 'large'])
        self._combo(mf, "Mode (AE/VAE):", self.vae_settings['vae_mode'],
                    ['AE', 'VAE'])                                # <-- NEW
        self._spin(mf, "Base channels:", self.vae_settings['vae_base_channels'], 4, 256)
        self._spin(mf, "Latent channels:", self.vae_settings['vae_latent_channels'], 1, 64)
        self._spin(mf, "Latent height:", self.vae_settings['vae_latent_h'], 1, 64)
        self._spin(mf, "Latent width:", self.vae_settings['vae_latent_w'], 1, 64)
        tk.Label(mf, text="Tip: AE is deterministic & recommended for audio.\n"
                          "VAE adds stochasticity via reparameterization + KL.",
                 fg="gray", justify=tk.LEFT).pack(anchor='w', pady=(5, 0))

        tf = tk.LabelFrame(wrap, text="Training", padx=10, pady=10)
        tf.pack(fill=tk.X, pady=5)
        self._spin(tf, "Batch size:", self.vae_settings['vae_batch_size'], 1, 128)
        self._entry(tf, "Learning rate:", self.vae_settings['vae_lr'])
        self._entry(tf, "KL weight (VAE only):", self.vae_settings['vae_kl_weight'])
        self._spin(tf, "DataLoader workers:", self.vae_settings['vae_num_workers'], 0, 8)

        gf = tk.LabelFrame(wrap, text="Audio / Spectrogram Settings", padx=10, pady=10)
        gf.pack(fill=tk.X, pady=5)
        self._spin(gf, "Spectrogram size:", self.global_settings['img_size'], 16, 256)
        self._entry(gf, "Segment duration (s):", self.global_settings['segment_duration'])
        self._spin(gf, "Sample rate:", self.global_settings['sample_rate'], 8000, 48000)
        self._spin(gf, "N mel bins:", self.global_settings['n_mels'], 32, 256)

    # ---------------- VAE Training tab ----------------
    def setup_vae_training_tab(self, tab):
        main = tk.Frame(tab)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        left = tk.Frame(main, width=300)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left.pack_propagate(False)

        tk.Label(left, text="VAE Training",
                 font=("Arial", 12, "bold")).pack(pady=(0, 10))

        ep = tk.Frame(left)
        ep.pack(pady=5)
        tk.Label(ep, text="Epochs:").pack(side=tk.LEFT)
        self.vae_epoch_var = tk.StringVar(value="100")
        tk.Entry(ep, textvariable=self.vae_epoch_var, width=8).pack(side=tk.LEFT, padx=5)

        tk.Button(left, text="Initialize Model",
                  command=self.initialize_vae, width=24).pack(pady=2)
        tk.Button(left, text="Start Training", command=self.start_vae_training,
                  width=24, bg="lightgreen").pack(pady=2)
        tk.Button(left, text="Stop Training", command=self.stop_vae_training,
                  width=24, bg="salmon").pack(pady=2)
        tk.Button(left, text="Save Model", command=self.save_vae,
                  width=24).pack(pady=2)
        tk.Button(left, text="Load Model", command=self.load_vae,
                  width=24).pack(pady=2)
        tk.Button(left, text="Show Preview Now",
                  command=self.show_vae_preview, width=24).pack(pady=8)

        right = tk.Frame(main)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        pf = tk.LabelFrame(right, text="Left: Input Spec | Right: Reconstruction",
                           padx=5, pady=5)
        pf.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        self.vae_preview_canvas = tk.Canvas(pf, bg='gray', width=256, height=128)
        self.vae_preview_canvas.pack()

        lf = tk.LabelFrame(right, text="Log", padx=5, pady=5)
        lf.pack(fill=tk.BOTH, expand=True)
        self.vae_log_text = tk.Text(lf, height=15, font=("Courier", 9))
        self.vae_log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = tk.Scrollbar(lf, command=self.vae_log_text.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.vae_log_text.config(yscrollcommand=sb.set)

    # ---------------- Flow Settings tab ----------------
    def setup_flow_settings_tab(self, tab):
        inner = self._make_scrollable(tab)
        wrap = tk.Frame(inner)
        wrap.pack(fill=tk.X, padx=20, pady=20)

        tk.Label(wrap, text="Flow Settings",
                 font=("Arial", 14, "bold")).pack(pady=(0, 15))

        uf = tk.LabelFrame(wrap, text="Flow UNet", padx=10, pady=10)
        uf.pack(fill=tk.X, pady=5)
        self._spin(uf, "Base channels:", self.flow_settings['flow_base_channels'], 4, 256)
        tk.Checkbutton(uf, text="Use Self-Attention",
                       variable=self.flow_settings['flow_use_attention']).pack(anchor='w')
        self._combo(uf, "Padding mode:", self.flow_settings['flow_padding_mode'],
                    ['constant', 'reflect', 'replicate', 'circular'])

        tf = tk.LabelFrame(wrap, text="Flow Training", padx=10, pady=10)
        tf.pack(fill=tk.X, pady=5)
        self._spin(tf, "Batch size:", self.flow_settings['flow_batch_size'], 1, 128)
        self._entry(tf, "Learning rate:", self.flow_settings['flow_lr'])
        self._entry(tf, "CFG dropout:", self.flow_settings['flow_cfg_dropout'])
        tk.Checkbutton(tf, text="Use OT matching",
                       variable=self.flow_settings['flow_use_ot']).pack(anchor='w')

        ef = tk.LabelFrame(wrap, text="EMA", padx=10, pady=10)
        ef.pack(fill=tk.X, pady=5)
        tk.Checkbutton(ef, text="Enable EMA",
                       variable=self.flow_settings['ema_enabled']).pack(anchor='w')
        self._entry(ef, "Decay:", self.flow_settings['ema_decay'])
        self._combo(ef, "Mode:", self.flow_settings['ema_mode'],
                    ['standard', 'lookahead'])

        pvf = tk.LabelFrame(wrap, text="Preview During Training", padx=10, pady=10)
        pvf.pack(fill=tk.X, pady=5)
        tk.Checkbutton(pvf, text="Enable preview",
                       variable=self.flow_settings['preview_enabled']).pack(anchor='w')
        self._spin(pvf, "Every N epochs:", self.flow_settings['preview_epoch_freq'], 1, 100)
        self._spin(pvf, "Preview steps:", self.flow_settings['preview_steps'], 1, 200)
        self._combo(pvf, "ODE method:", self.flow_settings['preview_method'],
                    ['euler', 'heun', 'midpoint', 'rk4'])

        cf = tk.LabelFrame(wrap, text="Text Conditioning", padx=10, pady=10)
        cf.pack(fill=tk.X, pady=5)
        tk.Checkbutton(cf, text="Enable text conditioning",
                       variable=self.flow_settings['cond_enabled']).pack(anchor='w')
        self._combo(cf, "Encoder type:", self.flow_settings['text_encoder_type'],
                    ['BiGRU', 'BiTransformer'])
        self._combo(cf, "Encoder size:", self.flow_settings['text_encoder_size'],
                    ['tiny', 'small', 'medium', 'large'])
        tk.Button(cf, text="Apply preset to dims",
                  command=self.apply_text_encoder_preset).pack(pady=3)
        self._spin(cf, "Embed dim:", self.flow_settings['cond_embed_dim'], 8, 512)
        self._spin(cf, "Hidden size (GRU):", self.flow_settings['cond_hidden_size'], 8, 1024)
        self._spin(cf, "Num layers:", self.flow_settings['cond_num_layers'], 1, 6)
        self._spin(cf, "Num heads (TF):", self.flow_settings['cond_num_heads'], 1, 16)
        self._spin(cf, "FF dim (TF):", self.flow_settings['cond_ff_dim'], 32, 2048)
        self._spin(cf, "Cond dim:", self.flow_settings['cond_dim'], 32, 1024)
        self._spin(cf, "Max text len:", self.flow_settings['cond_text_max_len'], 32, 512)

        r = tk.Frame(wrap)
        r.pack(fill=tk.X, pady=10)
        tk.Button(r, text="Reset All Settings to Defaults",
                  command=self.reset_all_settings,
                  bg="orange", fg="white",
                  font=("Arial", 10, "bold")).pack(pady=5)

    # ---------------- Flow Training tab ----------------
    def setup_flow_training_tab(self, tab):
        main = tk.Frame(tab)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        left = tk.Frame(main, width=300)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left.pack_propagate(False)

        tk.Label(left, text="Flow Training",
                 font=("Arial", 12, "bold")).pack(pady=(0, 10))

        ep = tk.Frame(left)
        ep.pack(pady=5)
        tk.Label(ep, text="Epochs:").pack(side=tk.LEFT)
        self.flow_epoch_var = tk.StringVar(value="200")
        tk.Entry(ep, textvariable=self.flow_epoch_var, width=8).pack(side=tk.LEFT, padx=5)

        tk.Button(left, text="Initialize Flow Model",
                  command=self.initialize_rectified_model,
                  width=24).pack(pady=2)
        tk.Button(left, text="Start Flow Training",
                  command=self.start_rectified_training,
                  width=24, bg="lightgreen").pack(pady=2)
        tk.Button(left, text="Stop Flow Training",
                  command=self.stop_rectified_training,
                  width=24, bg="salmon").pack(pady=2)
        tk.Button(left, text="Save Flow Model",
                  command=self.save_rectified_model, width=24).pack(pady=2)
        tk.Button(left, text="Load Flow Model",
                  command=self.load_rectified_model, width=24).pack(pady=2)

        right = tk.Frame(main)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        pf = tk.LabelFrame(right, text="Generated Samples (decoded)",
                           padx=5, pady=5)
        pf.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        self.rectified_preview_canvas = tk.Canvas(pf, bg='gray',
                                                  width=256, height=256)
        self.rectified_preview_canvas.pack()

        prf = tk.LabelFrame(right, text="Test prompt (empty = unconditional)",
                            padx=5, pady=5)
        prf.pack(fill=tk.X, pady=(0, 5))
        self.test_prompt_entry = tk.Entry(prf)
        self.test_prompt_entry.insert(0, "a soft piano melody")
        self.test_prompt_entry.pack(fill=tk.X, pady=2)
        tk.Button(prf, text="Generate Preview",
                  command=self.rectified_preview_with_prompt).pack(pady=2)

        lf = tk.LabelFrame(right, text="Log", padx=5, pady=5)
        lf.pack(fill=tk.BOTH, expand=True)
        self.rectified_log_text = tk.Text(lf, height=15, font=("Courier", 9))
        self.rectified_log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = tk.Scrollbar(lf, command=self.rectified_log_text.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.rectified_log_text.config(yscrollcommand=sb.set)

    # ---------------- Augmentation Settings tab ----------------
    def setup_augmentation_tab(self, tab):
        tk.Label(tab, text="Augmentation Settings",
                 font=("Arial", 14, "bold")).pack(pady=(15, 10))

        cols = tk.Frame(tab)
        cols.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        vae_col = tk.LabelFrame(cols, text="VAE Augmentations", padx=10, pady=10)
        vae_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        self._build_aug_controls(vae_col, self.vae_aug_settings)

        flow_col = tk.LabelFrame(cols, text="Flow Augmentations", padx=10, pady=10)
        flow_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))
        self._build_aug_controls(flow_col, self.flow_aug_settings)

        r = tk.Frame(tab)
        r.pack(fill=tk.X, pady=10)
        tk.Button(r, text="Reset VAE Augs",
                  command=lambda: self._reset_aug(self.vae_aug_settings),
                  bg="orange").pack(side=tk.LEFT, padx=10, expand=True)
        tk.Button(r, text="Reset Flow Augs",
                  command=lambda: self._reset_aug(self.flow_aug_settings),
                  bg="orange").pack(side=tk.RIGHT, padx=10, expand=True)

    def _build_aug_controls(self, parent, aug):
        def row():
            f = tk.Frame(parent)
            f.pack(fill=tk.X, pady=2)
            return f

        r = row()
        tk.Checkbutton(r, text="Horizontal Flip (time reversal)",
                       variable=aug['flip_horizontal']).pack(anchor='w')
        r = row()
        tk.Checkbutton(r, text="Rotation (±15°)",
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
        tk.Label(r, text="  bri / con:").pack(side=tk.LEFT)
        for k in ['brightness', 'contrast']:
            tk.Entry(r, textvariable=aug[k], width=5).pack(side=tk.LEFT, padx=1)
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
        tk.Checkbutton(r, text="SpecAugment (time/freq masking)",
                       variable=aug['spec_augment']).pack(side=tk.LEFT)
        r = row()
        tk.Label(r, text="  time mask:").pack(side=tk.LEFT)
        tk.Entry(r, textvariable=aug['spec_time_mask'], width=4).pack(side=tk.LEFT, padx=2)
        tk.Label(r, text="freq mask:").pack(side=tk.LEFT)
        tk.Entry(r, textvariable=aug['spec_freq_mask'], width=4).pack(side=tk.LEFT, padx=2)
        tk.Label(r, text="num masks:").pack(side=tk.LEFT)
        tk.Entry(r, textvariable=aug['spec_num_masks'], width=4).pack(side=tk.LEFT, padx=2)

    def _reset_aug(self, aug):
        for k, v in AUG_DEFAULTS.items():
            aug[k].set(v)

    # ---------------- Generation tab ----------------
    def setup_generation_tab(self, tab):
        main = tk.Frame(tab)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        tk.Label(main, text="Generate Audio (Flow → VAE Decode → Griffin-Lim)",
                 font=("Arial", 14, "bold")).pack(pady=(0, 10))

        pf = tk.LabelFrame(main, text="Text Prompt (empty = unconditional)",
                           padx=5, pady=5)
        pf.pack(fill=tk.X, pady=(0, 10))
        self.gen_prompt = tk.Entry(pf, width=60)
        self.gen_prompt.insert(0, "a soft piano melody")
        self.gen_prompt.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        cf = tk.Frame(main)
        cf.pack(pady=5)
        tk.Label(cf, text="Number:").pack(side=tk.LEFT)
        self.gen_count = tk.IntVar(value=4)
        tk.Spinbox(cf, from_=1, to=16, textvariable=self.gen_count,
                   width=5).pack(side=tk.LEFT, padx=5)
        tk.Label(cf, text="ODE Steps:").pack(side=tk.LEFT, padx=(10, 0))
        tk.Spinbox(cf, from_=1, to=200, textvariable=self.ode_steps,
                   width=5).pack(side=tk.LEFT, padx=5)
        tk.Label(cf, text="Method:").pack(side=tk.LEFT, padx=(10, 0))
        ttk.Combobox(cf, textvariable=self.ode_method,
                     values=['euler', 'heun', 'midpoint', 'rk4'],
                     state='readonly', width=10).pack(side=tk.LEFT, padx=5)
        tk.Label(cf, text="CFG scale:").pack(side=tk.LEFT, padx=(10, 0))
        tk.Entry(cf, width=6, textvariable=self.cfg_scale).pack(side=tk.LEFT, padx=5)
        self.progressive_grid = tk.BooleanVar(value=False)
        tk.Checkbutton(cf, text="Progressive",
                       variable=self.progressive_grid).pack(side=tk.LEFT, padx=10)
        tk.Label(cf, text="Interval:").pack(side=tk.LEFT)
        self.prog_interval = tk.IntVar(value=10)
        tk.Spinbox(cf, from_=1, to=50, textvariable=self.prog_interval,
                   width=5).pack(side=tk.LEFT, padx=5)
        self.generate_btn = tk.Button(cf, text="Generate",
                                      command=self.generate_samples,
                                      bg="lightgreen")
        self.generate_btn.pack(side=tk.LEFT, padx=5)
        self.stop_prog_btn = tk.Button(cf, text="Stop",
                                       command=self.stop_progressive,
                                       state=tk.DISABLED, bg="salmon")
        self.stop_prog_btn.pack(side=tk.LEFT, padx=5)

        rf = tk.LabelFrame(main, text="Results (spectrogram + audio)",
                           padx=5, pady=5)
        rf.pack(fill=tk.BOTH, expand=True, pady=5)
        cfr = tk.Frame(rf)
        cfr.pack(fill=tk.BOTH, expand=True)
        self.gen_canvas = tk.Canvas(cfr, bg='lightgray')
        vs = tk.Scrollbar(cfr, orient=tk.VERTICAL, command=self.gen_canvas.yview)
        hs = tk.Scrollbar(cfr, orient=tk.HORIZONTAL, command=self.gen_canvas.xview)
        self.gen_canvas.configure(yscrollcommand=vs.set, xscrollcommand=hs.set)
        vs.pack(side=tk.RIGHT, fill=tk.Y)
        hs.pack(side=tk.BOTTOM, fill=tk.X)
        self.gen_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.inner_frame = tk.Frame(self.gen_canvas)
        self.gen_canvas.create_window((0, 0), window=self.inner_frame, anchor='nw')
        self.inner_frame.bind(
            "<Configure>",
            lambda e: self.gen_canvas.configure(scrollregion=self.gen_canvas.bbox('all')))
        self.gen_info = tk.Label(main, text="", fg="blue")
        self.gen_info.pack()

    # ---------------- Logging ----------------
    def log_vae(self, msg):
        self.message_queue_vae.put(msg)

    def log_rectified(self, msg):
        self.message_queue_rectified.put(msg)

    def process_messages_vae(self):
        try:
            while True:
                msg = self.message_queue_vae.get_nowait()
                self.vae_log_text.insert(tk.END,
                                         f"{time.strftime('%H:%M:%S')} - {msg}\n")
                self.vae_log_text.see(tk.END)
                self.status_label.config(text=msg[:80])
        except queue.Empty:
            pass
        self.root.after(100, self.process_messages_vae)

    def process_messages_rectified(self):
        try:
            while True:
                msg = self.message_queue_rectified.get_nowait()
                self.rectified_log_text.insert(tk.END,
                                               f"{time.strftime('%H:%M:%S')} - {msg}\n")
                self.rectified_log_text.see(tk.END)
                self.status_label.config(text=msg[:80])
        except queue.Empty:
            pass
        self.root.after(100, self.process_messages_rectified)

    # ---------------- Dataset ops ----------------
    def add_audio_files(self):
        files = filedialog.askopenfilenames(
            filetypes=[("Audio", "*.wav *.mp3 *.ogg *.flac *.m4a")])
        for f in files:
            if f not in self.audio_paths:
                self.audio_paths.append(f)
                self.image_listbox.insert(tk.END, os.path.basename(f))
        self.log_rectified(f"Added {len(files)} audio files. "
                           f"Total: {len(self.audio_paths)}")

    def add_audio_folder(self):
        folder = filedialog.askdirectory()
        if not folder:
            return
        n = 0
        for r, _, fs in os.walk(folder):
            for f in fs:
                if f.lower().endswith(('.wav', '.mp3', '.ogg', '.flac', '.m4a')):
                    p = os.path.join(r, f)
                    if p not in self.audio_paths:
                        self.audio_paths.append(p)
                        self.image_listbox.insert(tk.END, os.path.basename(p))
                        n += 1
        self.log_rectified(f"Added {n} audio files. Total: {len(self.audio_paths)}")

    def clear_audio(self):
        self.audio_paths = []
        self.labels = []
        self.image_listbox.delete(0, tk.END)
        self.csv_status.config(text="No CSV loaded", fg="red")
        self.log_rectified("Cleared all audio files")

    def load_csv(self):
        fname = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if not fname:
            return
        self.csv_path = fname
        base_map = {os.path.basename(p).lower(): p for p in self.audio_paths}
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
            self.log_rectified(f"CSV error: {e}")
            return
        self.labels = []
        unknown = 0
        for p in self.audio_paths:
            if p in label_map:
                self.labels.append(label_map[p])
            else:
                self.labels.append([os.path.splitext(os.path.basename(p))[0]])
                unknown += 1
        self.csv_status.config(
            text=f"CSV: {len(self.audio_paths) - unknown} matched, "
                 f"{unknown} fallback", fg="green")
        self.log_rectified(f"CSV loaded. "
                           f"{len(self.audio_paths) - unknown} matched.")

    def use_filenames_as_labels(self):
        self.labels = [[os.path.splitext(os.path.basename(p))[0]]
                       for p in self.audio_paths]
        self.csv_status.config(text="Using filenames as labels", fg="blue")
        self.log_rectified("Using filenames as labels.")

    def use_folders_as_labels(self):
        self.labels = [[os.path.basename(os.path.dirname(p)) or 'unknown']
                       for p in self.audio_paths]
        self.csv_status.config(text="Using folder names as labels", fg="blue")
        self.log_rectified("Using folder names as labels.")

    def apply_text_encoder_preset(self):
        enc = self.flow_settings['text_encoder_type'].get()
        size = self.flow_settings['text_encoder_size'].get()
        cfg = get_encoder_config(enc, size)
        self.flow_settings['cond_embed_dim'].set(
            cfg.get('embed_dim', self.flow_settings['cond_embed_dim'].get()))
        if enc == 'BiGRU':
            self.flow_settings['cond_hidden_size'].set(
                cfg.get('hidden_size', self.flow_settings['cond_hidden_size'].get()))
            self.flow_settings['cond_num_layers'].set(
                cfg.get('num_layers', self.flow_settings['cond_num_layers'].get()))
        else:
            self.flow_settings['cond_num_heads'].set(
                cfg.get('num_heads', self.flow_settings['cond_num_heads'].get()))
            self.flow_settings['cond_num_layers'].set(
                cfg.get('num_layers', self.flow_settings['cond_num_layers'].get()))
            self.flow_settings['cond_ff_dim'].set(
                cfg.get('ff_dim', self.flow_settings['cond_ff_dim'].get()))
        self.flow_settings['cond_dim'].set(
            cfg.get('cond_dim', self.flow_settings['cond_dim'].get()))
        self.log_rectified(f"Applied {enc} {size} preset.")

    def reset_all_settings(self):
        for k, v in {
            'vae_size': 'big', 'vae_mode': 'AE', 'vae_base_channels': 32,
            'vae_latent_channels': 8, 'vae_latent_h': 8, 'vae_latent_w': 8,
            'vae_batch_size': 16, 'vae_lr': 1e-3, 'vae_num_workers': 0,
            'vae_kl_weight': 1e-4,
        }.items():
            self.vae_settings[k].set(v)
        for k, v in {
            'flow_base_channels': 32, 'flow_use_attention': True,
            'flow_padding_mode': 'constant', 'flow_batch_size': 16,
            'flow_lr': 2e-4, 'flow_cfg_dropout': 0.1, 'flow_use_ot': False,
            'ema_enabled': False, 'ema_decay': 0.999, 'ema_mode': 'standard',
            'preview_enabled': True, 'preview_epoch_freq': 5,
            'preview_steps': 20, 'preview_method': 'heun',
            'cond_enabled': False, 'text_encoder_type': 'BiGRU',
            'text_encoder_size': 'small', 'cond_embed_dim': 64,
            'cond_hidden_size': 64, 'cond_num_layers': 2, 'cond_num_heads': 4,
            'cond_ff_dim': 256, 'cond_dim': 256, 'cond_text_max_len': 128,
        }.items():
            self.flow_settings[k].set(v)
        self.global_settings['img_size'].set(64)
        self.global_settings['segment_duration'].set(1.472)
        self.global_settings['sample_rate'].set(22050)
        self.global_settings['n_mels'].set(128)
        self._reset_aug(self.vae_aug_settings)
        self._reset_aug(self.flow_aug_settings)
        self.log_rectified("All settings reset to defaults.")

    # ---------------- Audio processor helper ----------------
    def _make_audio_processor(self):
        return AudioProcessor(
            sr=self.global_settings['sample_rate'].get(),
            n_mels=self.global_settings['n_mels'].get(),
            segment_duration=self.global_settings['segment_duration'].get(),
            spec_size=self.global_settings['img_size'].get(),
        )

    # ---------------- VAE methods ----------------
    def initialize_vae(self):
        try:
            self.vae_model = FlexVAE(
                in_channels=1,  # grayscale spectrogram
                base_channels=self.vae_settings['vae_base_channels'].get(),
                latent_channels=self.vae_settings['vae_latent_channels'].get(),
                latent_h=self.vae_settings['vae_latent_h'].get(),
                latent_w=self.vae_settings['vae_latent_w'].get(),
                size=self.vae_settings['vae_size'].get(),
                img_size=self.global_settings['img_size'].get(),
                mode=self.vae_settings['vae_mode'].get(),
            ).to('cpu')
            self.vae_optimizer = optim.Adam(
                self.vae_model.parameters(),
                lr=self.vae_settings['vae_lr'].get())
            self.log_vae(f"Model initialized: size={self.vae_settings['vae_size'].get()}, "
                         f"mode={self.vae_settings['vae_mode'].get()}")
        except Exception as e:
            self.log_vae(f"Init error: {e}")

    def start_vae_training(self):
        if not self.audio_paths:
            self.log_vae("No audio files!")
            return
        if not self.vae_model:
            self.log_vae("Initialize model first!")
            return
        if self.training_vae:
            self.log_vae("Already training.")
            return
        try:
            epochs = int(self.vae_epoch_var.get())
        except Exception:
            self.log_vae("Invalid epochs")
            return
        self.training_vae = True
        self.vae_start_time = time.time()
        threading.Thread(target=self.train_vae_loop, args=(epochs,),
                         daemon=True).start()
        self.log_vae(f"VAE training started for {epochs} epochs.")

    def train_vae_loop(self, epochs):
        try:
            bs = self.vae_settings['vae_batch_size'].get()
            nw = self.vae_settings['vae_num_workers'].get()
            spec_size = self.global_settings['img_size'].get()
            seg_dur = self.global_settings['segment_duration'].get()
            kl_w = self.vae_settings['vae_kl_weight'].get()
            mode = self.vae_settings['vae_mode'].get()
            self.vae_model.mode = mode
            aug_dict = {k: v.get() for k, v in self.vae_aug_settings.items()}
            labels = self.labels if self.labels else [['']] * len(self.audio_paths)
            ap = self._make_audio_processor()
            ds = AudioSpectrogramDataset(
                self.audio_paths, labels, spec_size, seg_dur, aug_dict,
                self.flow_settings['cond_text_max_len'].get(),
                audio_processor=ap, verbose=True,
            )
            if len(ds) == 0:
                self.log_vae("No spectrograms produced — check audio files.")
                self.training_vae = False
                return
            dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=nw,
                            pin_memory=False, persistent_workers=(nw > 0))

            for ep in range(epochs):
                if not self.training_vae:
                    break
                tot, n = 0.0, 0
                for imgs, _ in dl:
                    if not self.training_vae:
                        break
                    imgs = imgs.to('cpu')
                    recon, mu, logvar = self.vae_model(imgs)
                    rl = F_nn.mse_loss(recon, imgs)
                    if mode == 'VAE':
                        kl = -0.5 * torch.sum(
                            1 + logvar - mu.pow(2) - logvar.exp()) / imgs.size(0)
                        loss = rl + kl_w * kl
                    else:
                        loss = rl
                    self.vae_optimizer.zero_grad()
                    loss.backward()
                    self.vae_optimizer.step()
                    tot += loss.item()
                    n += 1
                avg = tot / max(1, n)
                el = time.time() - self.vae_start_time
                self.log_vae(f"Epoch {ep + 1}/{epochs} | Loss: {avg:.6f} | "
                             f"Time: {el:.1f}s")
                if (ep + 1) % 5 == 0:
                    self.show_vae_preview()
            self.training_vae = False
            self.log_vae("VAE training finished.")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.log_vae(f"VAE training error: {e}")
            self.training_vae = False

    def stop_vae_training(self):
        self.training_vae = False
        self.log_vae("VAE training stopped.")

    def show_vae_preview(self):
        if not self.vae_model or not self.audio_paths:
            return
        try:
            n = min(16, len(self.audio_paths))
            idx = random.sample(range(len(self.audio_paths)), n)
            paths = [self.audio_paths[i] for i in idx]
            labs = ([self.labels[i] for i in idx]
                    if (self.labels and len(self.labels) == len(self.audio_paths))
                    else [['']] * n)
            spec_size = self.global_settings['img_size'].get()
            seg_dur = self.global_settings['segment_duration'].get()
            aug_dict = {k: v.get() for k, v in self.vae_aug_settings.items()}
            ap = self._make_audio_processor()
            ds = AudioSpectrogramDataset(
                paths, labs, spec_size, seg_dur, aug_dict,
                audio_processor=ap, verbose=False,
            )
            if len(ds) == 0:
                self.log_vae("No spectrograms for preview.")
                return
            dl = DataLoader(ds, batch_size=min(n, len(ds)), shuffle=False)
            imgs, _ = next(iter(dl))
            with torch.no_grad():
                recon, _, _ = self.vae_model(imgs)
            imgs = ((imgs + 1) / 2).clamp(0, 1).cpu().numpy()
            recon = ((recon + 1) / 2).clamp(0, 1).cpu().numpy()
            thumb = self.thumbnail_size // 2
            grid = Image.new('RGB', (8 * thumb, 4 * thumb))
            for i in range(min(16, imgs.shape[0])):
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
            self.log_vae("No model.")
            return
        fname = filedialog.asksaveasfilename(
            defaultextension=".pth", filetypes=[("PyTorch", "*.pth")])
        if fname:
            torch.save({
                'model_state': self.vae_model.state_dict(),
                'optimizer_state': self.vae_optimizer.state_dict(),
                'vae_settings': {k: v.get() for k, v in self.vae_settings.items()},
                'global_settings': {k: v.get() for k, v in self.global_settings.items()},
            }, fname)
            self.log_vae(f"Model saved to {fname}")

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
            self.log_vae(f"Model loaded from {fname}")
        except Exception as e:
            self.log_vae(f"Load error: {e}")

    # ---------------- Flow methods ----------------
    def initialize_rectified_model(self):
        if not self.vae_model:
            self.log_rectified("Train/load a VAE first!")
            return
        try:
            cond_enabled = self.flow_settings['cond_enabled'].get()
            cond_dim = self.flow_settings['cond_dim'].get() if cond_enabled else 1

            self.rectified_model = RectifiedFlowLatent(
                latent_channels=self.vae_settings['vae_latent_channels'].get(),
                latent_h=self.vae_settings['vae_latent_h'].get(),
                latent_w=self.vae_settings['vae_latent_w'].get(),
                base_channels=self.flow_settings['flow_base_channels'].get(),
                cond_dim=cond_dim,
                use_attention=self.flow_settings['flow_use_attention'].get(),
                padding_mode=self.flow_settings['flow_padding_mode'].get(),
                channel_mult=(1, 2, 4),
            )
            self.rectified_model.use_ot = self.flow_settings['flow_use_ot'].get()
            for pg in self.rectified_model.optimizer.param_groups:
                pg['lr'] = self.flow_settings['flow_lr'].get()

            if self.flow_settings['ema_enabled'].get():
                self.rectified_model.set_ema(
                    decay=self.flow_settings['ema_decay'].get(),
                    mode=self.flow_settings['ema_mode'].get())
                self.log_rectified(
                    f"EMA enabled: decay={self.flow_settings['ema_decay'].get()}, "
                    f"mode={self.flow_settings['ema_mode'].get()}")
            else:
                self.rectified_model.use_ema = False
                self.rectified_model.ema_model = None

            if cond_enabled:
                enc = self.flow_settings['text_encoder_type'].get()
                if enc == 'BiGRU':
                    self.text_encoder = TextEncoder(
                        vocab_size=256,
                        embed_dim=self.flow_settings['cond_embed_dim'].get(),
                        hidden_size=self.flow_settings['cond_hidden_size'].get(),
                        num_layers=self.flow_settings['cond_num_layers'].get(),
                        cond_dim=self.flow_settings['cond_dim'].get(),
                    )
                else:
                    self.text_encoder = TransformerTextEncoder(
                        vocab_size=256,
                        embed_dim=self.flow_settings['cond_embed_dim'].get(),
                        num_heads=self.flow_settings['cond_num_heads'].get(),
                        num_layers=self.flow_settings['cond_num_layers'].get(),
                        ff_dim=self.flow_settings['cond_ff_dim'].get(),
                        cond_dim=self.flow_settings['cond_dim'].get(),
                        max_len=self.flow_settings['cond_text_max_len'].get(),
                    )
                self.text_encoder.to(self.rectified_model.device)
                self.rectified_model.optimizer = optim.Adam(
                    list(self.rectified_model.model.parameters())
                    + list(self.text_encoder.parameters()),
                    lr=self.flow_settings['flow_lr'].get(),
                )
                self.log_rectified(f"Conditional Flow model initialized ({enc}).")
            else:
                self.text_encoder = None
                self.log_rectified("Unconditional Flow model initialized.")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.log_rectified(f"Init error: {e}")

    def start_rectified_training(self):
        if not self.audio_paths:
            self.log_rectified("No audio files!")
            return
        if not self.rectified_model:
            self.log_rectified("Initialize Flow model first!")
            return
        if not self.vae_model:
            self.log_rectified("No VAE to encode latents!")
            return
        if self.training_rectified:
            self.log_rectified("Already training.")
            return
        if self.flow_settings['cond_enabled'].get() and not self.labels:
            self.log_rectified("Load labels first.")
            return
        try:
            epochs = int(self.flow_epoch_var.get())
        except Exception:
            self.log_rectified("Invalid epochs")
            return
        self.training_rectified = True
        self.flow_start_time = time.time()
        threading.Thread(target=self.train_rectified_loop, args=(epochs,),
                         daemon=True).start()
        self.log_rectified(f"Flow training started for {epochs} epochs.")

    def train_rectified_loop(self, epochs):
        try:
            bs = self.flow_settings['flow_batch_size'].get()
            nw = self.vae_settings['vae_num_workers'].get()
            spec_size = self.global_settings['img_size'].get()
            seg_dur = self.global_settings['segment_duration'].get()
            cond_enabled = self.flow_settings['cond_enabled'].get()
            text_max = self.flow_settings['cond_text_max_len'].get()
            cfg_drop = self.flow_settings['flow_cfg_dropout'].get()
            preview_enabled = self.flow_settings['preview_enabled'].get()
            preview_freq = self.flow_settings['preview_epoch_freq'].get()

            aug_dict = {k: v.get() for k, v in self.flow_aug_settings.items()}
            labels = self.labels if self.labels else [['']] * len(self.audio_paths)
            ap = self._make_audio_processor()
            ds = AudioSpectrogramDataset(
                self.audio_paths, labels, spec_size, seg_dur, aug_dict,
                text_max, audio_processor=ap, verbose=True,
            )
            if len(ds) == 0:
                self.log_rectified("No spectrograms produced.")
                self.training_rectified = False
                return
            dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=nw,
                            pin_memory=False, persistent_workers=(nw > 0))

            latent_shape = (self.vae_settings['vae_latent_channels'].get(),
                            self.vae_settings['vae_latent_h'].get(),
                            self.vae_settings['vae_latent_w'].get())

            for ep in range(epochs):
                if not self.training_rectified:
                    break

                for pg in self.rectified_model.optimizer.param_groups:
                    pg['lr'] = self.flow_settings['flow_lr'].get()
                self.rectified_model.use_ot = self.flow_settings['flow_use_ot'].get()

                tot, n = 0.0, 0
                for imgs, txts in dl:
                    if not self.training_rectified:
                        break
                    with torch.no_grad():
                        z1 = self.vae_model.encode(
                            imgs.to(self.rectified_model.device),
                            deterministic=True)
                    z0 = torch.randn(z1.size(0), *latent_shape,
                                    device=self.rectified_model.device)
                    if cond_enabled and self.text_encoder is not None:
                        cond = self.text_encoder(
                            txts.to(self.rectified_model.device))
                    else:
                        cond = None
                    loss = self.rectified_model.train_step(
                        z0, z1, cond=cond, cfg_dropout_prob=cfg_drop)
                    tot += loss
                    n += 1

                avg = tot / max(1, n)
                el = time.time() - self.flow_start_time
                self.log_rectified(f"Epoch {ep + 1}/{epochs} | Loss: {avg:.6f} | "
                                   f"Time: {el:.1f}s")
                if preview_enabled and (ep + 1) % preview_freq == 0:
                    self.rectified_preview_with_prompt()

            self.training_rectified = False
            self.log_rectified("Flow training finished.")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.log_rectified(f"Training error: {e}")
            self.training_rectified = False

    def stop_rectified_training(self):
        self.training_rectified = False
        self.log_rectified("Flow training stopped.")

    def rectified_preview_with_prompt(self):
        if not self.rectified_model or not self.vae_model:
            self.log_rectified("Models not loaded for preview")
            return
        try:
            prompt = self.test_prompt_entry.get().strip()
            unconditional = (prompt == "")
            cond = None
            cfg_scale = 1.0 if unconditional else self.cfg_scale.get()
            n = 16
            if (not unconditional and self.flow_settings['cond_enabled'].get()
                    and self.text_encoder is not None):
                ti = text_to_indices(
                    prompt, self.flow_settings['cond_text_max_len'].get())
                tt = torch.tensor([ti] * n, dtype=torch.long,
                                  device=self.rectified_model.device)
                with torch.no_grad():
                    cond = self.text_encoder(tt)
                self.log_rectified(f"Preview: conditional on '{prompt}'")
            else:
                if not unconditional:
                    self.log_rectified("Preview: unconditional (no text encoder)")
                unconditional = True
                cfg_scale = 1.0

            use_ema = self.flow_settings['ema_enabled'].get()
            z = self.rectified_model.sample(
                n_samples=n,
                steps=self.flow_settings['preview_steps'].get(),
                method=self.flow_settings['preview_method'].get(),
                cond=cond, cfg_scale=cfg_scale, use_ema=use_ema)
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
                            im = np.stack([samples[idx][0] * 255] * 3,
                                          axis=-1).astype(np.uint8)
                        else:
                            im = (samples[idx].transpose(1, 2, 0) * 255).astype(np.uint8)
                        pil = Image.fromarray(im).resize((thumb, thumb), Image.NEAREST)
                        grid.paste(pil, (j * thumb, i * thumb))
            grid = grid.resize((256, 256), Image.NEAREST)
            self.rectified_preview_photo = ImageTk.PhotoImage(grid)
            self.rectified_preview_canvas.delete("all")
            self.rectified_preview_canvas.create_image(
                128, 128, image=self.rectified_preview_photo)
        except Exception as e:
            self.log_rectified(f"Preview error: {e}")

    def save_rectified_model(self):
        if not self.rectified_model:
            self.log_rectified("No model.")
            return
        fname = filedialog.asksaveasfilename(
            defaultextension=".pth", filetypes=[("PyTorch", "*.pth")])
        if fname:
            if (self.rectified_model.use_ema
                    and self.rectified_model.ema_model is not None):
                state = self.rectified_model.ema_model.state_dict()
                self.log_rectified("Saving EMA target model.")
            else:
                state = self.rectified_model.model.state_dict()
            d = {
                'model_state': state,
                'optimizer_state': self.rectified_model.optimizer.state_dict(),
                'flow_settings': {k: v.get() for k, v in self.flow_settings.items()},
                'vae_latent': {
                    'latent_channels': self.vae_settings['vae_latent_channels'].get(),
                    'latent_h': self.vae_settings['vae_latent_h'].get(),
                    'latent_w': self.vae_settings['vae_latent_w'].get(),
                },
                'use_ema': self.rectified_model.use_ema,
                'ema_decay': self.rectified_model.ema_decay,
                'ema_mode': self.rectified_model.ema_mode,
                'use_ot': self.rectified_model.use_ot,
                'channel_mult': (1, 2, 4),
            }
            if self.text_encoder is not None:
                d['text_encoder_state'] = self.text_encoder.state_dict()
            torch.save(d, fname)
            self.log_rectified(f"Flow model saved to {fname}")

    def load_rectified_model(self):
        fname = filedialog.askopenfilename(filetypes=[("PyTorch", "*.pth")])
        if not fname:
            return
        try:
            ckpt = torch.load(fname, map_location='cpu')
            for k, v in ckpt.get('flow_settings', {}).items():
                if k in self.flow_settings:
                    self.flow_settings[k].set(v)
            for k, v in ckpt.get('vae_latent', {}).items():
                map_to = {'latent_channels': 'vae_latent_channels',
                          'latent_h': 'vae_latent_h',
                          'latent_w': 'vae_latent_w'}[k]
                self.vae_settings[map_to].set(v)
            if not self.vae_model:
                self.log_rectified("Load VAE first, then retry.")
                return
            self.initialize_rectified_model()
            self.rectified_model.model.load_state_dict(ckpt['model_state'])
            try:
                self.rectified_model.optimizer.load_state_dict(ckpt['optimizer_state'])
            except Exception:
                pass
            if 'text_encoder_state' in ckpt and self.text_encoder is not None:
                self.text_encoder.load_state_dict(ckpt['text_encoder_state'])
            self.log_rectified(f"Flow model loaded from {fname}")
        except Exception as e:
            self.log_rectified(f"Load error: {e}")

    # ---------------- Generation ----------------
    def generate_samples(self):
        if not self.rectified_model or not self.vae_model:
            self.gen_info.config(text="Models not loaded!")
            return
        if (self.flow_settings['cond_enabled'].get()
                and self.text_encoder is None):
            self.gen_info.config(text="Text encoder missing!")
            return
        n = self.gen_count.get()
        steps = self.ode_steps.get()
        method = self.ode_method.get()
        prompt = self.gen_prompt.get().strip()
        cfg_scale = self.cfg_scale.get()
        use_ema = self.flow_settings['ema_enabled'].get()
        unconditional = (prompt == "")
        cond = None
        eff_cfg = 1.0 if unconditional else cfg_scale
        if (not unconditional and self.flow_settings['cond_enabled'].get()
                and self.text_encoder is not None):
            ti = text_to_indices(
                prompt, self.flow_settings['cond_text_max_len'].get())
            tt = torch.tensor([ti] * n, dtype=torch.long,
                              device=self.rectified_model.device)
            with torch.no_grad():
                cond = self.text_encoder(tt)
        elif not unconditional:
            unconditional = True
            eff_cfg = 1.0
        if self.progressive_grid.get():
            self.start_progressive(n, steps, method, cond, eff_cfg, use_ema)
        else:
            self.generate_btn.config(state=tk.DISABLED)
            self.gen_info.config(text="Generating...")
            self.root.update()
            threading.Thread(
                target=self._generate_thread,
                args=(n, steps, method, cond, eff_cfg, use_ema),
                daemon=True).start()

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
        threading.Thread(
            target=self._progressive_thread,
            args=(n, steps, method, cond, cfg_scale, use_ema),
            daemon=True).start()

    def _decode_latents(self, z):
        with torch.no_grad():
            samples = self.vae_model.decode(
                z, target_size=(self.global_settings['img_size'].get(),
                                self.global_settings['img_size'].get()))
        return ((samples + 1) / 2).clamp(0, 1).cpu().numpy()

    def _specs_to_audio_list(self, specs_np, ap):
        audios = []
        for i in range(specs_np.shape[0]):
            img = Image.fromarray((specs_np[i][0] * 255).astype(np.uint8))
            audios.append(ap.mel_to_audio(img))
        return audios

    def _grid_from_latents(self, z, n):
        samples = self._decode_latents(z)
        thumb = self.thumbnail_size
        gs = int(math.ceil(math.sqrt(n)))
        grid = Image.new('RGB', (gs * thumb, gs * thumb), color=(128, 128, 128))
        for i in range(min(n, len(samples))):
            r, c = i // gs, i % gs
            if samples[i].shape[0] == 1:
                im = np.stack([samples[i][0] * 255] * 3, axis=-1).astype(np.uint8)
            else:
                im = (samples[i].transpose(1, 2, 0) * 255).astype(np.uint8)
            pil = Image.fromarray(im).resize((thumb, thumb), Image.NEAREST)
            grid.paste(pil, (c * thumb, r * thumb))
        return grid

    def _generate_thread(self, n, steps, method, cond, cfg_scale, use_ema):
        try:
            z = self.rectified_model.sample(
                n_samples=n, steps=steps, method=method,
                cond=cond, cfg_scale=cfg_scale, use_ema=use_ema)
            samples = self._decode_latents(z)
            ap = self._make_audio_processor()
            audios = self._specs_to_audio_list(samples, ap)
            self.root.after(0, lambda: self._display_generated(samples, audios, ap))
        except Exception as e:
            self.root.after(0, lambda err=str(e):
                            self.gen_info.config(text=f"Error: {err}"))
        finally:
            self.root.after(0,
                            lambda: self.generate_btn.config(state=tk.NORMAL))

    def _progressive_thread(self, n, steps, method, cond, cfg_scale, use_ema):
        try:
            interval = self.prog_interval.get()
            gen = self.rectified_model.sample_step_by_step(
                n_samples=n, steps=steps, method=method, cond=cond,
                cfg_scale=cfg_scale, use_ema=use_ema)
            for step, z in gen:
                if not self.progressive_active:
                    break
                if step % interval == 0 or step == steps:
                    grid = self._grid_from_latents(z, n)
                    self.root.after(0, lambda g=grid, s=step:
                                    self._update_progressive(g, s))
                    time.sleep(0.05)
            self.root.after(0, lambda: self.gen_info.config(
                text="Progressive stopped." if not self.progressive_active
                else "Progressive finished."))
        except Exception as e:
            self.root.after(0, lambda err=str(e):
                            self.gen_info.config(text=f"Error: {err}"))
        finally:
            self.progressive_active = False
            self.root.after(0,
                            lambda: self.generate_btn.config(state=tk.NORMAL))
            self.root.after(0,
                            lambda: self.stop_prog_btn.config(state=tk.DISABLED))

    def _clear_gen_frame(self):
        for w in self.inner_frame.winfo_children():
            w.destroy()

    def _display_generated(self, spec_tensors, audios, ap):
        """Display generated spectrograms with individual play buttons."""
        self._clear_gen_frame()
        thumb = self.thumbnail_size
        cols = 2
        n = len(spec_tensors)
        rows = (n + cols - 1) // cols
        # Keep references to PhotoImage objects
        self._gen_photos = []
        for i in range(rows):
            row_frame = tk.Frame(self.inner_frame)
            row_frame.pack(pady=8)
            for j in range(cols):
                idx = i * cols + j
                if idx >= n:
                    continue
                frame = tk.LabelFrame(row_frame, text=f"Sample {idx + 1}",
                                      padx=5, pady=5)
                frame.pack(side=tk.LEFT, padx=8)
                img = spec_tensors[idx][0]
                pil_img = Image.fromarray(
                    (img * 255).astype(np.uint8)).resize(
                    (thumb, thumb), Image.NEAREST)
                photo = ImageTk.PhotoImage(pil_img)
                self._gen_photos.append(photo)
                lbl = tk.Label(frame, image=photo)
                lbl.image = photo
                lbl.pack()

                def make_play(a=audios[idx]):
                    return lambda: ap.play_audio(a)

                tk.Button(frame, text="▶ Play", command=make_play(),
                          width=8).pack(pady=3)
                tk.Button(frame, text="Save WAV",
                          command=lambda a=audios[idx]:
                          self._save_audio_dialog(a, ap),
                          width=8).pack(pady=1)

        self.inner_frame.update_idletasks()
        self.gen_canvas.configure(scrollregion=self.gen_canvas.bbox('all'))
        self.gen_info.config(text="Generation complete.")

    def _save_audio_dialog(self, audio, ap):
        fname = filedialog.asksaveasfilename(
            defaultextension=".wav", filetypes=[("WAV", "*.wav")])
        if fname:
            ap.save_audio(audio, fname)
            self.gen_info.config(text=f"Saved: {os.path.basename(fname)}")

    def _update_progressive(self, grid, step):
        self._clear_gen_frame()
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
    app = RectifiedFlowApp(root)
    root.mainloop()