import tkinter as tk
from tkinter import filedialog, ttk
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import functional as F_vision
import torch.nn.functional as F
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

# ==================== Dynamic VQVAE with compression ratio ====================

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)

    def forward(self, z):
        z_flattened = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z_flattened.view(-1, self.embedding_dim)
        distances = (z_flattened.pow(2).sum(1, keepdim=True)
                     - 2 * z_flattened @ self.embedding.weight.t()
                     + self.embedding.weight.pow(2).sum(1, keepdim=True).t())
        indices = distances.argmin(dim=1)
        z_q = self.embedding(indices).view(z.shape[0], z.shape[2], z.shape[3], self.embedding_dim)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        codebook_loss = F.mse_loss(z_q.detach(), z)
        commitment_loss = F.mse_loss(z_q, z.detach())
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss
        z_q_st = z + (z_q - z).detach()
        return z_q_st, vq_loss, indices

def get_downsample_layers(compression_ratio):
    """Number of stride-2 convs needed."""
    return int(math.log2(compression_ratio))

class VQEncoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=32, embed_dim=64, compression_ratio=4):
        super().__init__()
        self.compression_ratio = compression_ratio
        num_down = get_downsample_layers(compression_ratio)
        layers = []
        ch = in_channels
        for i in range(num_down):
            out_ch = base_channels * (2 ** i) if i < 2 else base_channels * (2 ** 2)  # cap growth
            layers.append(nn.Conv2d(ch, out_ch, 4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.2))
            ch = out_ch
        layers.append(nn.Conv2d(ch, embed_dim, 3, padding=1))
        layers.append(nn.LeakyReLU(0.2))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)

class VQDecoder(nn.Module):
    def __init__(self, embed_dim=64, base_channels=32, out_channels=3, compression_ratio=4):
        super().__init__()
        self.compression_ratio = compression_ratio
        num_up = get_downsample_layers(compression_ratio)
        layers = []
        ch = embed_dim
        for i in reversed(range(num_up)):
            out_ch = base_channels * (2 ** i) if i < 2 else base_channels * (2 ** 2)
            layers.append(nn.ConvTranspose2d(ch, out_ch, 4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.2))
            ch = out_ch
        layers.append(nn.Conv2d(ch, out_channels, 3, padding=1))
        layers.append(nn.Tanh())
        self.conv = nn.Sequential(*layers)

    def forward(self, z):
        return self.conv(z)

class VQVAE(nn.Module):
    def __init__(self, in_channels=3, base_channels=32, embed_dim=64,
                 num_embeddings=512, commitment_cost=0.25, compression_ratio=4):
        super().__init__()
        self.encoder = VQEncoder(in_channels, base_channels, embed_dim, compression_ratio)
        self.quantizer = VectorQuantizer(num_embeddings, embed_dim, commitment_cost)
        self.decoder = VQDecoder(embed_dim, base_channels, in_channels, compression_ratio)
        self.embed_dim = embed_dim
        self.num_embeddings = num_embeddings
        self.compression_ratio = compression_ratio
        # Dummy values – will be set later when image size is known
        self.latent_h = 8
        self.latent_w = 8

    def forward(self, x):
        z = self.encoder(x)
        z_q, vq_loss, indices = self.quantizer(z)
        recon = self.decoder(z_q)
        return recon, vq_loss, indices

    def encode_indices(self, x):
        z = self.encoder(x)
        _, _, indices = self.quantizer(z)
        return indices.view(x.size(0), -1)

    def decode_from_indices(self, indices, latent_shape=None):
        if indices.dim() == 1:
            indices = indices.unsqueeze(0)
        if latent_shape is None:
            latent_h, latent_w = self.latent_h, self.latent_w
        else:
            latent_h, latent_w = latent_shape
        z_q = self.quantizer.embedding(indices)
        z_q = z_q.view(-1, latent_h, latent_w, self.embed_dim).permute(0, 3, 1, 2)
        return self.decoder(z_q)

    def set_latent_shape(self, h, w):
        self.latent_h = h
        self.latent_w = w

# ==================== Text Encoders ====================

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
        B, T = x.shape
        emb = self.embedding(x)
        pos = self.pos_embedding[:, :T, :]
        x = emb + pos
        x = self.transformer(x)
        x = x.mean(dim=1)
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

# ==================== Conditional Sequence Models ====================

class ConditionalCausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        y = (attn @ v).transpose(1, 2).reshape(B, T, C)
        y = self.proj_drop(self.proj(y))
        return y

class ConditionalTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = ConditionalCausalSelfAttention(embed_dim, num_heads, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x

class ConditionalGPT(nn.Module):
    def __init__(self, vocab_size, max_seq_len, cond_dim=256, embed_dim=256, num_layers=6,
                 num_heads=4, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.cond_proj = nn.Linear(cond_dim, embed_dim)
        self.blocks = nn.ModuleList([
            ConditionalTransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.register_buffer('causal_mask',
                             torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1,1,max_seq_len,max_seq_len))

    def forward(self, x, cond):
        B, T = x.shape
        tok_emb = self.token_embedding(x)
        pos_emb = self.pos_embedding(torch.arange(T, device=x.device))
        cond_emb = self.cond_proj(cond).unsqueeze(1)
        x = tok_emb + pos_emb.unsqueeze(0) + cond_emb
        mask = self.causal_mask[:, :, :T, :T]
        for block in self.blocks:
            x = block(x, mask)
        x = self.ln_f(x)
        return self.head(x)

class ConditionalGRUModel(nn.Module):
    def __init__(self, vocab_size, max_seq_len, cond_dim=256, embed_dim=128, hidden_size=256,
                 num_layers=2, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, embed_dim))
        self.cond_proj = nn.Linear(cond_dim, num_layers * hidden_size)
        self.gru = nn.GRU(embed_dim, hidden_size, num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, cond):
        B, T = x.shape
        emb = self.token_embedding(x) + self.pos_embedding[:, :T, :]
        h0 = self.cond_proj(cond).view(B, -1, self.hidden_size).permute(1, 0, 2).contiguous()
        out, _ = self.gru(emb, h0)
        return self.fc(out)

class ConditionalMaskedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, mask_type='A', padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.register_buffer('mask', self._create_mask(mask_type, in_channels, out_channels, kernel_size))

    def _create_mask(self, mask_type, in_c, out_c, k):
        mask = torch.ones(out_c, in_c, k, k)
        center = k // 2
        for i in range(k):
            for j in range(k):
                if i > center or (i == center and j > center):
                    mask[:, :, i, j] = 0
        if mask_type == 'A':
            mask[:, :, center, center] = 0
        return mask

    def forward(self, x):
        self.conv.weight.data *= self.mask
        return self.conv(x)

class ConditionalPixelCNN(nn.Module):
    def __init__(self, vocab_size, cond_dim=256, embed_dim=64, num_layers=6, hidden_dim=128, kernel_size=3, height=8, width=8):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.height = height
        self.width = width

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.cond_proj = nn.Linear(cond_dim, embed_dim)

        self.conv_in = ConditionalMaskedConv2d(embed_dim * 2, hidden_dim, kernel_size, mask_type='A', padding=self.padding)
        self.act_in = nn.ReLU()

        self.blocks = nn.ModuleList()
        for _ in range(num_layers - 1):
            block = nn.Sequential(
                ConditionalMaskedConv2d(hidden_dim, hidden_dim, kernel_size, mask_type='B', padding=self.padding),
                nn.ReLU(),
                ConditionalMaskedConv2d(hidden_dim, hidden_dim, kernel_size, mask_type='B', padding=self.padding),
                nn.ReLU()
            )
            self.blocks.append(block)

        self.conv_out = nn.Conv2d(hidden_dim, vocab_size - 1, kernel_size=1)

    def forward(self, x, cond):
        B, H, W = x.shape
        x_emb = self.token_embedding(x).permute(0, 3, 1, 2)
        cond_emb = self.cond_proj(cond).view(B, -1, 1, 1).expand(-1, -1, H, W)
        h = torch.cat([x_emb, cond_emb], dim=1)
        h = self.act_in(self.conv_in(h))
        for block in self.blocks:
            h = h + block(h)
        return self.conv_out(h)

# ==================== Datasets ====================

class UnconditionalImageDataset(Dataset):
    def __init__(self, image_paths, img_size=32, color_mode='rgb', aug_settings=None):
        self.image_paths = image_paths
        self.img_size = img_size
        self.color_mode = color_mode.lower()
        self.aug_settings = aug_settings or {}
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
            return img_tensor
        except Exception as e:
            print(f"Error loading {self.image_paths[idx]}: {e}")
            if self.color_mode == 'rgb':
                return torch.zeros(3, self.img_size, self.img_size)
            else:
                return torch.zeros(1, self.img_size, self.img_size)

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

TRANSFORMER_CONFIGS = {
    'GPT-tiny': {'embed_dim': 128, 'num_layers': 4, 'num_heads': 4},
    'GPT-small': {'embed_dim': 256, 'num_layers': 6, 'num_heads': 4},
    'GPT-medium': {'embed_dim': 384, 'num_layers': 8, 'num_heads': 6},
    'GPT-large': {'embed_dim': 512, 'num_layers': 10, 'num_heads': 8},
}

GRU_CONFIGS = {
    'GRU-tiny': {'embed_dim': 64, 'hidden_size': 128, 'num_layers': 2},
    'GRU-small': {'embed_dim': 128, 'hidden_size': 256, 'num_layers': 2},
    'GRU-medium': {'embed_dim': 256, 'hidden_size': 512, 'num_layers': 3},
    'GRU-large': {'embed_dim': 384, 'hidden_size': 768, 'num_layers': 3},
}

PIXELCNN_CONFIGS = {
    'pixelcnn-tiny': {'embed_dim': 32, 'num_layers': 4, 'hidden_dim': 64},
    'pixelcnn-small': {'embed_dim': 64, 'num_layers': 6, 'hidden_dim': 128},
    'pixelcnn-medium': {'embed_dim': 128, 'num_layers': 8, 'hidden_dim': 256},
}

class VQGANTransformerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("VQGAN + Conditional Models - Compression Ratio & Conditioning Toggle")
        self.root.geometry("1200x800")

        self.image_paths = []
        self.labels = []
        self.csv_path = None
        self.training_vqvae = False
        self.training_cond_seq = False
        self.vqvae_model = None
        self.cond_seq_model = None
        self.cond_seq_model_type = None
        self.text_encoder = None

        self.current_epoch_vqvae = 0
        self.current_epoch_cond_seq = 0

        self.message_queue_vqvae = queue.Queue()
        self.message_queue_cond_seq = queue.Queue()

        self.settings = {
            'img_size': tk.IntVar(value=32),
            'color_mode': tk.StringVar(value='rgb'),
            'compression_ratio': tk.IntVar(value=4),
            'vq_base_channels': tk.IntVar(value=32),
            'vq_embed_dim': tk.IntVar(value=64),
            'vq_num_embeddings': tk.IntVar(value=512),
            'vq_commitment_cost': tk.DoubleVar(value=0.25),
            'vq_batch_size': tk.IntVar(value=16),
            'vq_lr': tk.DoubleVar(value=1e-3),
            'vq_num_workers': tk.IntVar(value=0),

            'cond_enabled': tk.BooleanVar(value=True),
            'seq_model_type': tk.StringVar(value='transformer'),
            'transformer_model_size': tk.StringVar(value='GPT-small'),
            'gru_model_size': tk.StringVar(value='GRU-small'),
            'pixelcnn_model_size': tk.StringVar(value='pixelcnn-small'),
            'seq_dropout': tk.DoubleVar(value=0.1),

            'text_encoder_type': tk.StringVar(value='BiGRU'),
            'text_encoder_size': tk.StringVar(value='small'),
            'cond_embed_dim': tk.IntVar(value=64),
            'cond_hidden_size': tk.IntVar(value=64),
            'cond_num_layers': tk.IntVar(value=2),
            'cond_num_heads': tk.IntVar(value=4),
            'cond_ff_dim': tk.IntVar(value=256),
            'cond_dim': tk.IntVar(value=256),
            'cond_text_max_len': tk.IntVar(value=128),
            'cond_batch_size': tk.IntVar(value=16),
            'cond_lr': tk.DoubleVar(value=5e-4),
            'cond_temperature': tk.DoubleVar(value=0.8),
            'cond_dropout_prob': tk.DoubleVar(value=0.1),
        }

        self.aug_settings = {
            'flip_horizontal': tk.BooleanVar(value=True),
            'rotation': tk.BooleanVar(value=False),
        }

        self.real_time_preview = tk.BooleanVar(value=False)
        self.preview_interval = tk.IntVar(value=5)
        self.cfg_scale = tk.DoubleVar(value=2.0)

        self.thumbnail_size = 128
        self.setup_gui()
        self.root.after(100, self.process_messages_vqvae)
        self.root.after(100, self.process_messages_cond_seq)

    def setup_gui(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.vqvae_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.vqvae_tab, text='VQVAE Training')
        self.setup_vqvae_tab()

        self.cond_seq_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.cond_seq_tab, text='Conditional Seq Training')
        self.setup_cond_seq_tab()

        self.settings_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.settings_tab, text='Settings')
        self.setup_settings_tab()

        self.generation_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.generation_tab, text='Generation')
        self.setup_generation_tab()

        self.status_label = tk.Label(self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    # ------------------------------------------------------------------
    # VQVAE Training Tab
    # ------------------------------------------------------------------
    def setup_vqvae_tab(self):
        main_frame = tk.Frame(self.vqvae_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_frame = tk.Frame(main_frame, width=300)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0,10))
        left_frame.pack_propagate(False)

        tk.Label(left_frame, text="VQVAE Training", font=("Arial",12,"bold")).pack(pady=(0,10))

        img_frame = tk.LabelFrame(left_frame, text="Training Images", padx=5, pady=5)
        img_frame.pack(fill=tk.X, pady=(0,10))
        tk.Button(img_frame, text="Add Images", command=self.add_images, width=20).pack(pady=2)
        tk.Button(img_frame, text="Add Folder", command=self.add_folder, width=20).pack(pady=2)
        tk.Button(img_frame, text="Clear All", command=self.clear_images, width=20).pack(pady=2)
        self.image_listbox = tk.Listbox(img_frame, height=5)
        self.image_listbox.pack(fill=tk.X, pady=2)

        tk.Button(left_frame, text="Initialize VQVAE", command=self.initialize_vqvae, width=20).pack(pady=5)
        epoch_frame = tk.Frame(left_frame)
        epoch_frame.pack(pady=5)
        tk.Label(epoch_frame, text="Epochs:").pack(side=tk.LEFT)
        self.vqvae_epoch_var = tk.StringVar(value="50")
        tk.Entry(epoch_frame, textvariable=self.vqvae_epoch_var, width=8).pack(side=tk.LEFT, padx=5)

        tk.Button(left_frame, text="Start VQVAE Training", command=self.start_vqvae_training,
                  width=20, bg="lightgreen").pack(pady=5)
        tk.Button(left_frame, text="Stop VQVAE Training", command=self.stop_vqvae_training,
                  width=20, bg="salmon").pack(pady=5)
        tk.Button(left_frame, text="Save VQVAE", command=self.save_vqvae, width=20).pack(pady=5)
        tk.Button(left_frame, text="Load VQVAE", command=self.load_vqvae, width=20).pack(pady=5)

        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        preview_frame = tk.LabelFrame(right_frame, text="Reconstructions (random images)", padx=5, pady=5)
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=(0,5))
        self.vqvae_preview_canvas = tk.Canvas(preview_frame, bg='gray', width=256, height=128)
        self.vqvae_preview_canvas.pack()

        log_frame = tk.LabelFrame(right_frame, text="Log", padx=5, pady=5)
        log_frame.pack(fill=tk.BOTH, expand=True)
        self.vqvae_log_text = tk.Text(log_frame, height=15, font=("Courier",9))
        self.vqvae_log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = tk.Scrollbar(log_frame, command=self.vqvae_log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.vqvae_log_text.config(yscrollcommand=scrollbar.set)

    # ------------------------------------------------------------------
    # Conditional Sequence Model Training Tab
    # ------------------------------------------------------------------
    def setup_cond_seq_tab(self):
        main_frame = tk.Frame(self.cond_seq_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_frame = tk.Frame(main_frame, width=300)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0,10))
        left_frame.pack_propagate(False)

        tk.Label(left_frame, text="Conditional Sequence Model", font=("Arial",12,"bold")).pack(pady=(0,10))
        tk.Label(left_frame, text="Requires a trained VQVAE", fg="blue").pack(pady=5)

        csv_frame = tk.LabelFrame(left_frame, text="Text Labels (ignored if conditioning disabled)", padx=5, pady=5)
        csv_frame.pack(fill=tk.X, pady=5)
        tk.Button(csv_frame, text="Load CSV (image,label)", command=self.load_csv, width=20).pack(pady=2)
        tk.Button(csv_frame, text="Use Filenames as Labels", command=self.use_filenames_as_labels, width=20).pack(pady=2)
        tk.Button(csv_frame, text="Use Folder Names as Labels", command=self.use_folders_as_labels, width=20).pack(pady=2)
        self.csv_status = tk.Label(csv_frame, text="No CSV loaded", fg="red")
        self.csv_status.pack()

        tk.Button(left_frame, text="Initialize Models", command=self.initialize_cond_models, width=20).pack(pady=5)
        epoch_frame = tk.Frame(left_frame)
        epoch_frame.pack(pady=5)
        tk.Label(epoch_frame, text="Epochs:").pack(side=tk.LEFT)
        self.cond_epoch_var = tk.StringVar(value="100")
        tk.Entry(epoch_frame, textvariable=self.cond_epoch_var, width=8).pack(side=tk.LEFT, padx=5)

        tk.Button(left_frame, text="Start Training", command=self.start_cond_training,
                  width=20, bg="lightgreen").pack(pady=5)
        tk.Button(left_frame, text="Stop Training", command=self.stop_cond_training,
                  width=20, bg="salmon").pack(pady=5)
        tk.Button(left_frame, text="Save Model", command=self.save_cond_model, width=20).pack(pady=5)
        tk.Button(left_frame, text="Load Model", command=self.load_cond_model, width=20).pack(pady=5)

        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        preview_frame = tk.LabelFrame(right_frame, text="Conditional Samples (test prompt)", padx=5, pady=5)
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=(0,5))
        self.cond_preview_canvas = tk.Canvas(preview_frame, bg='gray', width=256, height=256)
        self.cond_preview_canvas.pack()
        tk.Label(preview_frame, text="Test prompt (leave empty for unconditional):").pack()
        self.cond_test_prompt = tk.Entry(preview_frame)
        self.cond_test_prompt.insert(0, "a cute cat")
        self.cond_test_prompt.pack(fill=tk.X)
        tk.Button(preview_frame, text="Generate Preview", command=self.cond_preview).pack(pady=2)

        log_frame = tk.LabelFrame(right_frame, text="Log", padx=5, pady=5)
        log_frame.pack(fill=tk.BOTH, expand=True)
        self.cond_log_text = tk.Text(log_frame, height=15, font=("Courier",9))
        self.cond_log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = tk.Scrollbar(log_frame, command=self.cond_log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.cond_log_text.config(yscrollcommand=scrollbar.set)

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

        # Image Settings
        img_frame = tk.LabelFrame(scrollable_frame, text="Image", padx=10, pady=10)
        img_frame.pack(fill=tk.X, pady=5)
        f = tk.Frame(img_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Color mode:", width=20, anchor='w').pack(side=tk.LEFT)
        om = ttk.Combobox(f, textvariable=self.settings['color_mode'], values=['rgb', 'grayscale'], state='readonly', width=10)
        om.pack(side=tk.RIGHT)
        f = tk.Frame(img_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Image size:", width=20, anchor='w').pack(side=tk.LEFT)
        tk.Scale(f, from_=16, to=128, variable=self.settings['img_size'],
                 orient=tk.HORIZONTAL, length=200, resolution=1).pack(side=tk.RIGHT)
        tk.Label(f, textvariable=self.settings['img_size'], width=5).pack(side=tk.RIGHT)

        # Compression Ratio
        f = tk.Frame(img_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Compression ratio:", width=20, anchor='w').pack(side=tk.LEFT)
        ratio_combo = ttk.Combobox(f, textvariable=self.settings['compression_ratio'],
                                   values=[2,4,8], state='readonly', width=10)
        ratio_combo.pack(side=tk.RIGHT)

        # VQVAE Settings
        vqvae_frame = tk.LabelFrame(scrollable_frame, text="VQVAE", padx=10, pady=10)
        vqvae_frame.pack(fill=tk.X, pady=5)
        vqvae_params = [
            ("Base channels:", 'vq_base_channels', 16, 128),
            ("Embedding dim:", 'vq_embed_dim', 32, 256),
            ("Codebook size:", 'vq_num_embeddings', 128, 1024),
            ("Commitment cost:", 'vq_commitment_cost', 0.1, 1.0),
            ("Batch size:", 'vq_batch_size', 1, 32),
            ("Learning rate:", 'vq_lr', 1e-5, 1e-2),
            ("Workers:", 'vq_num_workers', 0, 4),
        ]
        for label, key, low, high in vqvae_params:
            f = tk.Frame(vqvae_frame); f.pack(fill=tk.X, pady=2)
            tk.Label(f, text=label, width=20, anchor='w').pack(side=tk.LEFT)
            res = 0.01 if 'cost' in key else (0.00001 if 'lr' in key else 1)
            tk.Scale(f, from_=low, to=high, variable=self.settings[key],
                     orient=tk.HORIZONTAL, length=200, resolution=res).pack(side=tk.RIGHT)
            tk.Label(f, textvariable=self.settings[key], width=5).pack(side=tk.RIGHT)

        # Conditional Model Settings
        cond_frame = tk.LabelFrame(scrollable_frame, text="Conditional Model", padx=10, pady=10)
        cond_frame.pack(fill=tk.X, pady=5)

        # Conditioning toggle
        cb_cond = tk.Checkbutton(cond_frame, text="Enable text conditioning",
                                 variable=self.settings['cond_enabled'])
        cb_cond.pack(anchor='w', pady=2)

        f = tk.Frame(cond_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Model type:", width=20, anchor='w').pack(side=tk.LEFT)
        type_combo = ttk.Combobox(f, textvariable=self.settings['seq_model_type'],
                                   values=['transformer', 'gru', 'pixelcnn'], state='readonly', width=10)
        type_combo.pack(side=tk.RIGHT)

        f = tk.Frame(cond_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Transformer size:", width=20, anchor='w').pack(side=tk.LEFT)
        trans_combo = ttk.Combobox(f, textvariable=self.settings['transformer_model_size'],
                                    values=list(TRANSFORMER_CONFIGS.keys()), state='readonly', width=10)
        trans_combo.pack(side=tk.RIGHT)

        f = tk.Frame(cond_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="GRU size:", width=20, anchor='w').pack(side=tk.LEFT)
        gru_combo = ttk.Combobox(f, textvariable=self.settings['gru_model_size'],
                                  values=list(GRU_CONFIGS.keys()), state='readonly', width=10)
        gru_combo.pack(side=tk.RIGHT)

        f = tk.Frame(cond_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="PixelCNN size:", width=20, anchor='w').pack(side=tk.LEFT)
        pixel_combo = ttk.Combobox(f, textvariable=self.settings['pixelcnn_model_size'],
                                    values=list(PIXELCNN_CONFIGS.keys()), state='readonly', width=10)
        pixel_combo.pack(side=tk.RIGHT)

        seq_params = [
            ("Dropout:", 'seq_dropout', 0.0, 0.5),
        ]
        for label, key, low, high in seq_params:
            f = tk.Frame(cond_frame); f.pack(fill=tk.X, pady=2)
            tk.Label(f, text=label, width=20, anchor='w').pack(side=tk.LEFT)
            res = 0.01
            tk.Scale(f, from_=low, to=high, variable=self.settings[key],
                     orient=tk.HORIZONTAL, length=200, resolution=res).pack(side=tk.RIGHT)
            tk.Label(f, textvariable=self.settings[key], width=5).pack(side=tk.RIGHT)

        # Text Encoder Settings
        text_frame = tk.LabelFrame(cond_frame, text="Text Encoder (used only if conditioning enabled)", padx=10, pady=10)
        text_frame.pack(fill=tk.X, pady=5)

        f = tk.Frame(text_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Encoder type:", width=20, anchor='w').pack(side=tk.LEFT)
        type_enc = ttk.Combobox(f, textvariable=self.settings['text_encoder_type'],
                                values=['BiGRU', 'BiTransformer'], state='readonly', width=12)
        type_enc.pack(side=tk.RIGHT)

        f = tk.Frame(text_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Encoder size preset:", width=20, anchor='w').pack(side=tk.LEFT)
        size_enc = ttk.Combobox(f, textvariable=self.settings['text_encoder_size'],
                                values=['tiny', 'small', 'medium', 'large'], state='readonly', width=8)
        size_enc.pack(side=tk.RIGHT)
        tk.Button(f, text="Apply preset to dims", command=self.apply_text_encoder_preset, width=18).pack(side=tk.RIGHT, padx=5)

        cond_text_params = [
            ("Text embed dim:", 'cond_embed_dim', 32, 256),
            ("Text hidden size:", 'cond_hidden_size', 32, 512),
            ("Text num layers:", 'cond_num_layers', 1, 4),
            ("Num heads:", 'cond_num_heads', 1, 16),
            ("FF dim:", 'cond_ff_dim', 64, 1024),
            ("Conditioning dim:", 'cond_dim', 64, 512),
            ("Max text length:", 'cond_text_max_len', 32, 256),
            ("Batch size:", 'cond_batch_size', 1, 32),
            ("Learning rate:", 'cond_lr', 1e-5, 1e-2),
            ("Sampling temp:", 'cond_temperature', 0.1, 2.0),
            ("CFG dropout prob:", 'cond_dropout_prob', 0.0, 0.5),
        ]
        for label, key, low, high in cond_text_params:
            f = tk.Frame(text_frame); f.pack(fill=tk.X, pady=2)
            tk.Label(f, text=label, width=20, anchor='w').pack(side=tk.LEFT)
            res = 0.01 if 'temp' in key or 'dropout' in key else (0.00001 if 'lr' in key else 1)
            tk.Scale(f, from_=low, to=high, variable=self.settings[key],
                     orient=tk.HORIZONTAL, length=200, resolution=res).pack(side=tk.RIGHT)
            tk.Label(f, textvariable=self.settings[key], width=5).pack(side=tk.RIGHT)

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
        self.log_cond_seq(f"Applied {enc_type} {size} preset dims.")

    # ------------------------------------------------------------------
    # Generation Tab
    # ------------------------------------------------------------------
    def setup_generation_tab(self):
        main_frame = tk.Frame(self.generation_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        tk.Label(main_frame, text="Generate Images", font=("Arial",14,"bold")).pack(pady=(0,10))

        prompt_frame = tk.LabelFrame(main_frame, text="Text Prompt (leave empty for unconditional generation)", padx=5, pady=5)
        prompt_frame.pack(fill=tk.X, pady=5)
        self.gen_prompt = tk.Entry(prompt_frame, width=50)
        self.gen_prompt.insert(0, "a cute cat")
        self.gen_prompt.pack(side=tk.LEFT, padx=5)

        ctrl_frame = tk.Frame(main_frame)
        ctrl_frame.pack(pady=5)

        tk.Label(ctrl_frame, text="Number:").pack(side=tk.LEFT)
        self.gen_count = tk.IntVar(value=16)
        tk.Spinbox(ctrl_frame, from_=1, to=64, textvariable=self.gen_count, width=5).pack(side=tk.LEFT, padx=5)

        tk.Label(ctrl_frame, text="Temperature:").pack(side=tk.LEFT, padx=(10,0))
        self.gen_temp = tk.DoubleVar(value=self.settings['cond_temperature'].get())
        tk.Spinbox(ctrl_frame, from_=0.1, to=2.0, increment=0.1, textvariable=self.gen_temp, width=5).pack(side=tk.LEFT, padx=5)

        tk.Label(ctrl_frame, text="CFG Scale:").pack(side=tk.LEFT, padx=(10,0))
        self.cfg_scale_entry = tk.Entry(ctrl_frame, width=5, textvariable=self.cfg_scale)
        self.cfg_scale_entry.pack(side=tk.LEFT, padx=5)

        self.realtime_cb = tk.Checkbutton(ctrl_frame, text="Real-time preview",
                                          variable=self.real_time_preview)
        self.realtime_cb.pack(side=tk.LEFT, padx=10)

        tk.Label(ctrl_frame, text="Update interval:").pack(side=tk.LEFT)
        tk.Spinbox(ctrl_frame, from_=1, to=20, textvariable=self.preview_interval, width=3).pack(side=tk.LEFT, padx=5)

        self.generate_btn = tk.Button(ctrl_frame, text="Generate", command=self.generate_samples, bg="lightgreen")
        self.generate_btn.pack(side=tk.LEFT, padx=5)

        self.stop_gen_btn = tk.Button(ctrl_frame, text="Stop", command=self.stop_generation,
                                      state=tk.DISABLED, bg="salmon")
        self.stop_gen_btn.pack(side=tk.LEFT, padx=5)

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

        self.generation_active = False

    # ------------------------------------------------------------------
    # Core helper methods
    # ------------------------------------------------------------------
    def log_vqvae(self, msg):
        self.message_queue_vqvae.put(msg)

    def log_cond_seq(self, msg):
        self.message_queue_cond_seq.put(msg)

    def process_messages_vqvae(self):
        try:
            while True:
                msg = self.message_queue_vqvae.get_nowait()
                self.vqvae_log_text.insert(tk.END, f"{time.strftime('%H:%M:%S')} - {msg}\n")
                self.vqvae_log_text.see(tk.END)
                self.status_label.config(text=msg[:50])
        except queue.Empty:
            pass
        self.root.after(100, self.process_messages_vqvae)

    def process_messages_cond_seq(self):
        try:
            while True:
                msg = self.message_queue_cond_seq.get_nowait()
                self.cond_log_text.insert(tk.END, f"{time.strftime('%H:%M:%S')} - {msg}\n")
                self.cond_log_text.see(tk.END)
                self.status_label.config(text=msg[:50])
        except queue.Empty:
            pass
        self.root.after(100, self.process_messages_cond_seq)

    def add_images(self):
        files = filedialog.askopenfilenames(filetypes=[("Images", "*.jpg *.jpeg *.png *.jfif *.webp *.bmp *.ico *.tiff")])
        for f in files:
            if f not in self.image_paths:
                self.image_paths.append(f)
                self.image_listbox.insert(tk.END, os.path.basename(f))
        self.log_vqvae(f"Added {len(files)} images. Total: {len(self.image_paths)}")
        self.log_cond_seq(f"Added {len(files)} images. Total: {len(self.image_paths)}")

    def add_folder(self):
        folder = filedialog.askdirectory()
        if not folder:
            return
        count = 0
        for root_dir, _, files in os.walk(folder):
            for file in files:
                if file.lower().endswith(('.png','.jpg','.jpeg','.jfif','.webp','.bmp', '.ico', '.tiff')):
                    full = os.path.join(root_dir, file)
                    if full not in self.image_paths:
                        self.image_paths.append(full)
                        self.image_listbox.insert(tk.END, os.path.basename(full))
                        count += 1
        self.log_vqvae(f"Added {count} images. Total: {len(self.image_paths)}")
        self.log_cond_seq(f"Added {count} images. Total: {len(self.image_paths)}")

    def clear_images(self):
        self.image_paths.clear()
        self.labels.clear()
        self.image_listbox.delete(0, tk.END)
        self.csv_status.config(text="No CSV loaded", fg="red")
        self.log_vqvae("Cleared all images")
        self.log_cond_seq("Cleared all images")

    # ----- Robust CSV loading -----
    def load_csv(self):
        fname = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if not fname:
            return
        self.csv_path = fname
        path_by_basename = {}
        for full in self.image_paths:
            base = os.path.basename(full).lower()
            if base in path_by_basename:
                self.log_cond_seq(f"Warning: duplicate basename '{base}'")
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
            self.log_cond_seq(f"Error reading CSV: {e}")
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
        self.log_cond_seq(f"CSV loaded. {len(self.image_paths)-unknown} images matched with labels.")

    def use_filenames_as_labels(self):
        self.labels = [[os.path.splitext(os.path.basename(p))[0]] for p in self.image_paths]
        self.csv_status.config(text="Using filenames as labels", fg="blue")
        self.log_cond_seq("Using filenames as labels.")

    def use_folders_as_labels(self):
        self.labels = []
        for p in self.image_paths:
            folder = os.path.basename(os.path.dirname(p))
            if not folder:
                folder = 'unknown'
            self.labels.append([folder])
        self.csv_status.config(text="Using folder names as labels", fg="blue")
        self.log_cond_seq("Using folder names as labels.")

    # ========== VQVAE Methods ==========
    def compute_latent_size(self):
        img_size = self.settings['img_size'].get()
        ratio = self.settings['compression_ratio'].get()
        h = img_size // ratio
        w = img_size // ratio
        return max(1, h), max(1, w)

    def initialize_vqvae(self):
        try:
            in_channels = 3 if self.settings['color_mode'].get() == 'rgb' else 1
            ratio = self.settings['compression_ratio'].get()
            self.vqvae_model = VQVAE(
                in_channels=in_channels,
                base_channels=self.settings['vq_base_channels'].get(),
                embed_dim=self.settings['vq_embed_dim'].get(),
                num_embeddings=self.settings['vq_num_embeddings'].get(),
                commitment_cost=self.settings['vq_commitment_cost'].get(),
                compression_ratio=ratio
            )
            self.vqvae_model.to('cpu')
            latent_h, latent_w = self.compute_latent_size()
            self.vqvae_model.set_latent_shape(latent_h, latent_w)
            self.vqvae_optimizer = optim.Adam(self.vqvae_model.parameters(), lr=self.settings['vq_lr'].get())
            self.log_vqvae(f"VQVAE initialized (compression {ratio}, latent {latent_h}x{latent_w})")
        except Exception as e:
            self.log_vqvae(f"Error: {e}")

    def start_vqvae_training(self):
        if not self.image_paths:
            self.log_vqvae("No images!"); return
        if not self.vqvae_model:
            self.log_vqvae("VQVAE not initialized!"); return
        if self.training_vqvae:
            self.log_vqvae("Already training."); return
        try:
            epochs = int(self.vqvae_epoch_var.get())
        except:
            self.log_vqvae("Invalid epochs"); return
        self.training_vqvae = True
        self.current_epoch_vqvae = 0
        self.vqvae_start_time = time.time()
        thread = threading.Thread(target=self.train_vqvae_loop, args=(epochs,), daemon=True)
        thread.start()
        self.log_vqvae(f"VQVAE training started for {epochs} epochs.")

    def train_vqvae_loop(self, epochs):
        try:
            batch_size = self.settings['vq_batch_size'].get()
            num_workers = self.settings['vq_num_workers'].get()
            img_size = self.settings['img_size'].get()
            color_mode = self.settings['color_mode'].get()
            aug_dict = {k: v.get() for k, v in self.aug_settings.items()}
            dataset = UnconditionalImageDataset(self.image_paths, img_size, color_mode, aug_dict)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                num_workers=num_workers, pin_memory=False)
            for epoch in range(epochs):
                if not self.training_vqvae: break
                self.current_epoch_vqvae = epoch
                epoch_loss = 0.0
                batches = 0
                for images in loader:
                    if not self.training_vqvae: break
                    images = images
                    recon, vq_loss, _ = self.vqvae_model(images)
                    recon_loss = F.mse_loss(recon, images)
                    loss = recon_loss + vq_loss
                    self.vqvae_optimizer.zero_grad()
                    loss.backward()
                    self.vqvae_optimizer.step()
                    epoch_loss += loss.item()
                    batches += 1
                avg_loss = epoch_loss / batches if batches else 0
                elapsed = time.time() - self.vqvae_start_time
                self.log_vqvae(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f} | Time: {elapsed:.1f}s")
                if (epoch+1) % 5 == 0:
                    self.show_vqvae_preview()
            self.training_vqvae = False
            self.log_vqvae("VQVAE training finished.")
        except Exception as e:
            self.log_vqvae(f"Training error: {e}")
            import traceback; traceback.print_exc()
            self.training_vqvae = False

    def show_vqvae_preview(self):
        if not self.vqvae_model or len(self.image_paths) == 0:
            return
        try:
            n_samples = min(16, len(self.image_paths))
            indices = random.sample(range(len(self.image_paths)), n_samples)
            color_mode = self.settings['color_mode'].get()
            dataset = UnconditionalImageDataset([self.image_paths[i] for i in indices],
                                                img_size=self.settings['img_size'].get(),
                                                color_mode=color_mode)
            loader = DataLoader(dataset, batch_size=n_samples, shuffle=False)
            images = next(iter(loader))

            with torch.no_grad():
                recon, _, _ = self.vqvae_model(images)

            images = (images + 1) / 2
            recon = (recon + 1) / 2
            images = images.clamp(0,1).cpu().numpy()
            recon = recon.clamp(0,1).cpu().numpy()

            thumb = self.thumbnail_size // 2
            grid = Image.new('RGB', (8*thumb, 4*thumb))
            for i in range(4):
                for j in range(4):
                    idx = i*4 + j
                    if idx < len(images):
                        if images[idx].shape[0] == 1:
                            img = np.stack([images[idx][0]]*3, axis=-1)
                        else:
                            img = images[idx].transpose(1,2,0)
                        img = (img * 255).astype(np.uint8)
                        pil_img = Image.fromarray(img).resize((thumb, thumb), Image.NEAREST)
                        grid.paste(pil_img, (j*thumb*2, i*thumb))
                        if recon[idx].shape[0] == 1:
                            img_r = np.stack([recon[idx][0]]*3, axis=-1)
                        else:
                            img_r = recon[idx].transpose(1,2,0)
                        img_r = (img_r * 255).astype(np.uint8)
                        pil_img_r = Image.fromarray(img_r).resize((thumb, thumb), Image.NEAREST)
                        grid.paste(pil_img_r, (j*thumb*2 + thumb, i*thumb))
            grid = grid.resize((256,128), Image.NEAREST)
            self.vqvae_preview_photo = ImageTk.PhotoImage(grid)
            self.vqvae_preview_canvas.delete("all")
            self.vqvae_preview_canvas.create_image(128,64, image=self.vqvae_preview_photo)
        except Exception as e:
            self.log_vqvae(f"Preview error: {e}")

    def stop_vqvae_training(self):
        self.training_vqvae = False
        self.log_vqvae("VQVAE training stopped.")

    def save_vqvae(self):
        if not self.vqvae_model:
            self.log_vqvae("No VQVAE."); return
        fname = filedialog.asksaveasfilename(defaultextension=".pth", filetypes=[("PyTorch","*.pth")])
        if fname:
            torch.save({
                'model_state': self.vqvae_model.state_dict(),
                'optimizer_state': self.vqvae_optimizer.state_dict(),
                'settings': {k:v.get() for k,v in self.settings.items() if k.startswith('vq') or k in ['img_size','color_mode','compression_ratio']},
            }, fname)
            self.log_vqvae(f"VQVAE saved to {fname}")

    def load_vqvae(self):
        fname = filedialog.askopenfilename(filetypes=[("PyTorch","*.pth")])
        if not fname: return
        try:
            ckpt = torch.load(fname, map_location='cpu')
            if 'settings' in ckpt:
                for k,v in ckpt['settings'].items():
                    if k in self.settings:
                        self.settings[k].set(v)
            self.initialize_vqvae()
            self.vqvae_model.load_state_dict(ckpt['model_state'])
            self.vqvae_optimizer.load_state_dict(ckpt['optimizer_state'])
            self.log_vqvae(f"VQVAE loaded from {fname}")
        except Exception as e:
            self.log_vqvae(f"Load error: {e}")

    # ========== Conditional Sequence Model Methods ==========
    def initialize_cond_models(self):
        if not self.vqvae_model:
            self.log_cond_seq("Please train/load a VQVAE first!")
            return
        try:
            cond_enabled = self.settings['cond_enabled'].get()
            vocab_size = self.settings['vq_num_embeddings'].get() + 1
            latent_h, latent_w = self.compute_latent_size()
            max_seq_len = latent_h * latent_w + 1
            self.settings['seq_max_len'] = tk.IntVar(value=max_seq_len)  # store for later

            if cond_enabled:
                cond_dim = self.settings['cond_dim'].get()
                enc_type = self.settings['text_encoder_type'].get()
                if enc_type == 'BiGRU':
                    self.text_encoder = TextEncoder(
                        vocab_size=256,
                        embed_dim=self.settings['cond_embed_dim'].get(),
                        hidden_size=self.settings['cond_hidden_size'].get(),
                        num_layers=self.settings['cond_num_layers'].get(),
                        cond_dim=cond_dim
                    )
                else:
                    self.text_encoder = TransformerTextEncoder(
                        vocab_size=256,
                        embed_dim=self.settings['cond_embed_dim'].get(),
                        num_heads=self.settings['cond_num_heads'].get(),
                        num_layers=self.settings['cond_num_layers'].get(),
                        ff_dim=self.settings['cond_ff_dim'].get(),
                        cond_dim=cond_dim,
                        max_len=self.settings['cond_text_max_len'].get()
                    )
                self.text_encoder.to('cpu')
            else:
                self.text_encoder = None
                cond_dim = 1  # dummy

            model_type = self.settings['seq_model_type'].get()
            dropout = self.settings['seq_dropout'].get()

            if model_type == 'transformer':
                size_name = self.settings['transformer_model_size'].get()
                config = TRANSFORMER_CONFIGS[size_name]
                self.cond_seq_model = ConditionalGPT(
                    vocab_size=vocab_size,
                    max_seq_len=max_seq_len,
                    cond_dim=cond_dim,
                    embed_dim=config['embed_dim'],
                    num_layers=config['num_layers'],
                    num_heads=config['num_heads'],
                    mlp_ratio=4.0,
                    dropout=dropout
                )
            elif model_type == 'gru':
                size_name = self.settings['gru_model_size'].get()
                config = GRU_CONFIGS[size_name]
                self.cond_seq_model = ConditionalGRUModel(
                    vocab_size=vocab_size,
                    max_seq_len=max_seq_len,
                    cond_dim=cond_dim,
                    embed_dim=config['embed_dim'],
                    hidden_size=config['hidden_size'],
                    num_layers=config['num_layers'],
                    dropout=dropout
                )
            elif model_type == 'pixelcnn':
                size_name = self.settings['pixelcnn_model_size'].get()
                config = PIXELCNN_CONFIGS[size_name]
                self.cond_seq_model = ConditionalPixelCNN(
                    vocab_size=vocab_size,
                    cond_dim=cond_dim,
                    embed_dim=config['embed_dim'],
                    num_layers=config['num_layers'],
                    hidden_dim=config['hidden_dim'],
                    kernel_size=3,
                    height=latent_h,
                    width=latent_w
                )
            else:
                self.log_cond_seq("Unknown model type.")
                return

            self.cond_seq_model.to('cpu')
            if cond_enabled:
                self.cond_optimizer = optim.Adam(
                    list(self.text_encoder.parameters()) + list(self.cond_seq_model.parameters()),
                    lr=self.settings['cond_lr'].get()
                )
            else:
                self.cond_optimizer = optim.Adam(
                    self.cond_seq_model.parameters(),
                    lr=self.settings['cond_lr'].get()
                )
            self.cond_seq_model_type = model_type
            self.log_cond_seq(f"Models initialized. cond_enabled={cond_enabled}, latent={latent_h}x{latent_w}, seq_len={max_seq_len}")
        except Exception as e:
            self.log_cond_seq(f"Error: {e}")

    def start_cond_training(self):
        if not self.image_paths:
            self.log_cond_seq("No images!"); return
        cond_enabled = self.settings['cond_enabled'].get()
        if cond_enabled and not self.labels:
            self.log_cond_seq("Conditional training requires labels. Load CSV or use filenames/folders."); return
        if not self.cond_seq_model:
            self.log_cond_seq("Models not initialized!"); return
        if not self.vqvae_model:
            self.log_cond_seq("No VQVAE!"); return
        if self.training_cond_seq:
            self.log_cond_seq("Already training."); return
        try:
            epochs = int(self.cond_epoch_var.get())
        except:
            self.log_cond_seq("Invalid epochs"); return
        self.training_cond_seq = True
        self.current_epoch_cond_seq = 0
        self.cond_start_time = time.time()
        thread = threading.Thread(target=self.train_cond_loop, args=(epochs,), daemon=True)
        thread.start()
        self.log_cond_seq(f"Conditional training started for {epochs} epochs.")

    def train_cond_loop(self, epochs):
        try:
            batch_size = self.settings['cond_batch_size'].get()
            num_workers = self.settings['vq_num_workers'].get()
            img_size = self.settings['img_size'].get()
            color_mode = self.settings['color_mode'].get()
            aug_dict = {k: v.get() for k, v in self.aug_settings.items()}
            text_max_len = self.settings['cond_text_max_len'].get()
            cond_enabled = self.settings['cond_enabled'].get()
            cond_dropout_prob = self.settings['cond_dropout_prob'].get() if cond_enabled else 0.0

            if cond_enabled:
                dataset = ConditionalImageDataset(self.image_paths, self.labels, img_size, color_mode, aug_dict, text_max_len)
            else:
                dummy_labels = [[''] for _ in self.image_paths]
                dataset = ConditionalImageDataset(self.image_paths, dummy_labels, img_size, color_mode, aug_dict, text_max_len)

            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                num_workers=num_workers, pin_memory=False)
            criterion = nn.CrossEntropyLoss()
            self.vqvae_model.eval()
            if cond_enabled:
                self.text_encoder.train()
            self.cond_seq_model.train()

            start_token = self.settings['vq_num_embeddings'].get()
            model_type = self.cond_seq_model_type

            for epoch in range(epochs):
                if not self.training_cond_seq: break
                self.current_epoch_cond_seq = epoch
                epoch_loss = 0.0
                batches = 0
                for images, text_tensors in loader:
                    if not self.training_cond_seq: break
                    with torch.no_grad():
                        indices = self.vqvae_model.encode_indices(images)

                    if cond_enabled:
                        cond = self.text_encoder(text_tensors)
                        if cond_dropout_prob > 0:
                            mask = torch.rand(cond.size(0), 1, device=cond.device) > cond_dropout_prob
                            cond = cond * mask.float()
                    else:
                        cond = torch.zeros(images.size(0), 1, device=indices.device)

                    if model_type in ['transformer', 'gru']:
                        B, T = indices.shape
                        start = torch.full((B, 1), start_token, dtype=torch.long, device=indices.device)
                        inp = torch.cat([start, indices[:, :-1]], dim=1)
                        target = indices
                        logits = self.cond_seq_model(inp, cond)
                        loss = criterion(logits.view(-1, logits.size(-1)), target.view(-1))
                    else:  # pixelcnn
                        grid = indices.view(-1, self.vqvae_model.latent_h, self.vqvae_model.latent_w)
                        logits = self.cond_seq_model(grid, cond)
                        loss = criterion(logits.permute(0,2,3,1).reshape(-1, logits.size(1)),
                                         grid.reshape(-1))

                    self.cond_optimizer.zero_grad()
                    loss.backward()
                    self.cond_optimizer.step()
                    epoch_loss += loss.item()
                    batches += 1

                avg_loss = epoch_loss / batches if batches else 0
                elapsed = time.time() - self.cond_start_time
                self.log_cond_seq(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f} | Time: {elapsed:.1f}s")
                if (epoch+1) % 5 == 0:
                    self.cond_preview()
            self.training_cond_seq = False
            self.log_cond_seq("Conditional training finished.")
        except Exception as e:
            self.log_cond_seq(f"Training error: {e}")
            import traceback; traceback.print_exc()
            self.training_cond_seq = False

    def cond_preview(self):
        if not self.cond_seq_model or not self.vqvae_model:
            self.log_cond_seq("Models not loaded")
            return
        cond_enabled = self.settings['cond_enabled'].get()
        prompt = self.cond_test_prompt.get().strip()
        unconditional = (not cond_enabled) or (prompt == "")
        try:
            n = 16
            device = 'cpu'
            model_type = self.cond_seq_model_type
            latent_h, latent_w = self.compute_latent_size()
            text_max_len = self.settings['cond_text_max_len'].get()
            if unconditional:
                cond_tensor = torch.zeros(n, 1, device=device)
                self.log_cond_seq("Preview: unconditional generation")
                cfg_scale = 1.0
            else:
                text_indices = text_to_indices(prompt, text_max_len)
                text_tensor = torch.tensor([text_indices], dtype=torch.long, device=device).repeat(n, 1)
                with torch.no_grad():
                    cond_tensor = self.text_encoder(text_tensor)
                cfg_scale = self.cfg_scale.get()
            null_cond = torch.zeros_like(cond_tensor)

            with torch.no_grad():
                if model_type in ['transformer', 'gru']:
                    start_token = self.settings['vq_num_embeddings'].get()
                    max_len = latent_h * latent_w + 1
                    self.cond_seq_model.eval()
                    x = torch.full((n, 1), start_token, dtype=torch.long, device=device)
                    for step in range(max_len - 1):
                        logits_cond = self.cond_seq_model(x, cond_tensor)[:, -1, :]
                        logits_uncond = self.cond_seq_model(x, null_cond)[:, -1, :]
                        logits = logits_uncond + cfg_scale * (logits_cond - logits_uncond)
                        if self.settings['cond_temperature'].get() > 0:
                            probs = F.softmax(logits / self.settings['cond_temperature'].get(), dim=-1)
                            next_token = torch.multinomial(probs, 1)
                        else:
                            next_token = logits.argmax(dim=-1, keepdim=True)
                        x = torch.cat([x, next_token], dim=1)
                    indices = x[:, 1:]
                else:  # pixelcnn
                    self.cond_seq_model.eval()
                    H, W = latent_h, latent_w
                    grid = torch.full((n, H, W), self.settings['vq_num_embeddings'].get(), dtype=torch.long, device=device)
                    for i in range(H):
                        for j in range(W):
                            logits_cond = self.cond_seq_model(grid, cond_tensor)
                            logits_uncond = self.cond_seq_model(grid, null_cond)
                            logits = logits_uncond + cfg_scale * (logits_cond - logits_uncond)
                            logits_at_pos = logits[:, :, i, j]
                            if self.settings['cond_temperature'].get() > 0:
                                probs = F.softmax(logits_at_pos / self.settings['cond_temperature'].get(), dim=-1)
                                next_token = torch.multinomial(probs, 1).squeeze(-1)
                            else:
                                next_token = logits_at_pos.argmax(dim=-1)
                            grid[:, i, j] = next_token
                    indices = grid.view(n, -1)
                samples = self.vqvae_model.decode_from_indices(indices, latent_shape=(latent_h, latent_w))
            samples = (samples + 1) / 2
            samples = samples.clamp(0,1).cpu().numpy()
            thumb = self.thumbnail_size
            grid_img = Image.new('RGB', (4*thumb, 4*thumb))
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
                        grid_img.paste(pil_img, (j*thumb, i*thumb))
            grid_img = grid_img.resize((256,256), Image.NEAREST)
            self.cond_preview_photo = ImageTk.PhotoImage(grid_img)
            self.cond_preview_canvas.delete("all")
            self.cond_preview_canvas.create_image(128,128, image=self.cond_preview_photo)
        except Exception as e:
            self.log_cond_seq(f"Preview error: {e}")

    def stop_cond_training(self):
        self.training_cond_seq = False
        self.log_cond_seq("Conditional training stopped.")

    def save_cond_model(self):
        if not self.cond_seq_model:
            self.log_cond_seq("No model."); return
        fname = filedialog.asksaveasfilename(defaultextension=".pth", filetypes=[("PyTorch","*.pth")])
        if fname:
            save_dict = {
                'cond_seq_model_state': self.cond_seq_model.state_dict(),
                'optimizer_state': self.cond_optimizer.state_dict(),
                'settings': {k:v.get() for k,v in self.settings.items() if k.startswith('cond') or k in ['seq_model_type','transformer_model_size','gru_model_size','pixelcnn_model_size','seq_dropout','text_encoder_type','text_encoder_size','cond_enabled']},
                'model_type': self.cond_seq_model_type,
            }
            if self.text_encoder is not None:
                save_dict['text_encoder_state'] = self.text_encoder.state_dict()
            torch.save(save_dict, fname)
            self.log_cond_seq(f"Conditional model saved to {fname}")

    def load_cond_model(self):
        fname = filedialog.askopenfilename(filetypes=[("PyTorch","*.pth")])
        if not fname: return
        try:
            ckpt = torch.load(fname, map_location='cpu')
            if 'settings' in ckpt:
                for k,v in ckpt['settings'].items():
                    if k in self.settings:
                        self.settings[k].set(v)
            if not self.vqvae_model:
                self.log_cond_seq("Please load VQVAE first.")
                return
            self.cond_seq_model_type = ckpt.get('model_type', 'transformer')
            self.initialize_cond_models()
            self.cond_seq_model.load_state_dict(ckpt['cond_seq_model_state'])
            self.cond_optimizer.load_state_dict(ckpt['optimizer_state'])
            if 'text_encoder_state' in ckpt and self.text_encoder is not None:
                self.text_encoder.load_state_dict(ckpt['text_encoder_state'])
            self.log_cond_seq(f"Conditional model loaded from {fname}")
        except Exception as e:
            self.log_cond_seq(f"Load error: {e}")

    # ========== Generation with Real-time Preview ==========
    def generate_samples(self):
        if not self.cond_seq_model or not self.vqvae_model:
            self.gen_info.config(text="Conditional models not loaded!")
            return
        n = self.gen_count.get()
        temp = self.gen_temp.get()
        prompt = self.gen_prompt.get().strip()
        self.generation_active = True
        self.generate_btn.config(state=tk.DISABLED)
        self.stop_gen_btn.config(state=tk.NORMAL)
        if prompt == "" or not self.settings['cond_enabled'].get():
            self.gen_info.config(text="Generating unconditionally...")
        else:
            self.gen_info.config(text=f"Generating with prompt: {prompt}")
        self.root.update()
        thread = threading.Thread(target=self._generate_thread,
                                   args=(n, temp, prompt),
                                   daemon=True)
        thread.start()

    def stop_generation(self):
        self.generation_active = False
        self.stop_gen_btn.config(state=tk.DISABLED)
        self.gen_info.config(text="Generation stopped.")

    def _generate_thread(self, n, temp, prompt):
        try:
            device = 'cpu'
            realtime = self.real_time_preview.get()
            interval = self.preview_interval.get()
            model = self.cond_seq_model
            model_type = self.cond_seq_model_type
            cond_enabled = self.settings['cond_enabled'].get()
            cfg_scale = self.cfg_scale.get() if cond_enabled else 1.0

            latent_h, latent_w = self.compute_latent_size()
            text_max_len = self.settings['cond_text_max_len'].get()
            unconditional = (not cond_enabled) or (prompt == "")
            if unconditional:
                cond_tensor = torch.zeros(n, 1, device=device)
                effective_cfg_scale = 1.0
            else:
                text_indices = text_to_indices(prompt, text_max_len)
                text_tensor = torch.tensor([text_indices], dtype=torch.long, device=device).repeat(n, 1)
                with torch.no_grad():
                    cond_tensor = self.text_encoder(text_tensor)
                effective_cfg_scale = cfg_scale
            null_cond = torch.zeros_like(cond_tensor)

            if model_type in ['transformer', 'gru']:
                start_token = self.settings['vq_num_embeddings'].get()
                max_len = latent_h * latent_w + 1
                if realtime:
                    x = torch.full((n, 1), start_token, dtype=torch.long, device=device)
                    model.eval()
                    total_steps = max_len - 1
                    for step in range(total_steps):
                        if not self.generation_active: break
                        logits_cond = model(x, cond_tensor)[:, -1, :]
                        logits_uncond = model(x, null_cond)[:, -1, :]
                        logits = logits_uncond + effective_cfg_scale * (logits_cond - logits_uncond)
                        if temp > 0:
                            probs = F.softmax(logits / temp, dim=-1)
                            next_token = torch.multinomial(probs, 1)
                        else:
                            next_token = logits.argmax(dim=-1, keepdim=True)
                        x = torch.cat([x, next_token], dim=1)
                        if (step+1) % interval == 0 or step+1 == total_steps:
                            current_tokens = x[:, 1:]
                            pad_len = max_len - 1 - current_tokens.size(1)
                            if pad_len > 0:
                                padded = torch.cat([current_tokens, torch.zeros(n, pad_len, dtype=torch.long, device=device)], dim=1)
                            else:
                                padded = current_tokens
                            with torch.no_grad():
                                samples = self.vqvae_model.decode_from_indices(padded, latent_shape=(latent_h, latent_w))
                            self.root.after(0, lambda s=samples.clone(), st=step+1: self._update_preview(s, st, is_pixelcnn=False))
                    if self.generation_active:
                        indices = x[:, 1:]
                        with torch.no_grad():
                            samples = self.vqvae_model.decode_from_indices(indices, latent_shape=(latent_h, latent_w))
                        self.root.after(0, lambda: self._display_final(samples, n))
                else:
                    x = torch.full((n, 1), start_token, dtype=torch.long, device=device)
                    model.eval()
                    for step in range(max_len - 1):
                        if not self.generation_active: break
                        logits_cond = model(x, cond_tensor)[:, -1, :]
                        logits_uncond = model(x, null_cond)[:, -1, :]
                        logits = logits_uncond + effective_cfg_scale * (logits_cond - logits_uncond)
                        if temp > 0:
                            probs = F.softmax(logits / temp, dim=-1)
                            next_token = torch.multinomial(probs, 1)
                        else:
                            next_token = logits.argmax(dim=-1, keepdim=True)
                        x = torch.cat([x, next_token], dim=1)
                    if self.generation_active:
                        indices = x[:, 1:]
                        with torch.no_grad():
                            samples = self.vqvae_model.decode_from_indices(indices, latent_shape=(latent_h, latent_w))
                        self.root.after(0, lambda: self._display_final(samples, n))
            else:  # pixelcnn
                H, W = latent_h, latent_w
                total_tokens = H * W
                grid = torch.full((n, H, W), self.settings['vq_num_embeddings'].get(), dtype=torch.long, device=device)
                model.eval()
                if realtime:
                    for i in range(H):
                        for j in range(W):
                            if not self.generation_active: break
                            logits_cond = model(grid, cond_tensor)
                            logits_uncond = model(grid, null_cond)
                            logits = logits_uncond + effective_cfg_scale * (logits_cond - logits_uncond)
                            logits_at_pos = logits[:, :, i, j]
                            if temp > 0:
                                probs = F.softmax(logits_at_pos / temp, dim=-1)
                                next_token = torch.multinomial(probs, 1).squeeze(-1)
                            else:
                                next_token = logits_at_pos.argmax(dim=-1)
                            grid[:, i, j] = next_token
                            step = i * W + j + 1
                            if step % interval == 0 or step == total_tokens:
                                flat = grid.view(n, -1)
                                with torch.no_grad():
                                    samples = self.vqvae_model.decode_from_indices(flat, latent_shape=(latent_h, latent_w))
                                self.root.after(0, lambda s=samples.clone(), st=step: self._update_preview(s, st, is_pixelcnn=True))
                        if not self.generation_active: break
                    if self.generation_active:
                        indices = grid.view(n, -1)
                        with torch.no_grad():
                            samples = self.vqvae_model.decode_from_indices(indices, latent_shape=(latent_h, latent_w))
                        self.root.after(0, lambda: self._display_final(samples, n))
                else:
                    for i in range(H):
                        for j in range(W):
                            if not self.generation_active: break
                            logits_cond = model(grid, cond_tensor)
                            logits_uncond = model(grid, null_cond)
                            logits = logits_uncond + effective_cfg_scale * (logits_cond - logits_uncond)
                            logits_at_pos = logits[:, :, i, j]
                            if temp > 0:
                                probs = F.softmax(logits_at_pos / temp, dim=-1)
                                next_token = torch.multinomial(probs, 1).squeeze(-1)
                            else:
                                next_token = logits_at_pos.argmax(dim=-1)
                            grid[:, i, j] = next_token
                    if self.generation_active:
                        indices = grid.view(n, -1)
                        with torch.no_grad():
                            samples = self.vqvae_model.decode_from_indices(indices, latent_shape=(latent_h, latent_w))
                        self.root.after(0, lambda: self._display_final(samples, n))
        except Exception as e:
            self.root.after(0, lambda e=e: self.gen_info.config(text=f"Error: {e}"))
            import traceback; traceback.print_exc()
        finally:
            self.generation_active = False
            self.root.after(0, lambda: self.generate_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.stop_gen_btn.config(state=tk.DISABLED))

    def _update_preview(self, samples, step, is_pixelcnn=False):
        samples = (samples + 1) / 2
        samples = samples.clamp(0,1).cpu().numpy()
        n = samples.shape[0]
        thumb = self.thumbnail_size
        grid_size = int(math.ceil(math.sqrt(n)))
        total = grid_size * thumb
        grid_img = Image.new('RGB', (total, total), color=(128,128,128))
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
        for w in self.inner_frame.winfo_children():
            w.destroy()
        self.prog_photo = ImageTk.PhotoImage(grid_img)
        label = tk.Label(self.inner_frame, image=self.prog_photo)
        label.image = self.prog_photo
        label.pack()
        self.inner_frame.update_idletasks()
        self.gen_canvas.configure(scrollregion=self.gen_canvas.bbox('all'))
        if is_pixelcnn:
            self.gen_info.config(text=f"Token {step}/{self.vqvae_model.latent_h * self.vqvae_model.latent_w}")
        else:
            max_len = self.vqvae_model.latent_h * self.vqvae_model.latent_w
            self.gen_info.config(text=f"Token {step}/{max_len}")

    def _display_final(self, samples, n):
        samples = (samples + 1) / 2
        samples = samples.clamp(0,1).cpu().numpy()
        thumb = self.thumbnail_size
        grid_size = int(math.ceil(math.sqrt(n)))
        total = grid_size * thumb
        grid_img = Image.new('RGB', (total, total), color=(128,128,128))
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
        for w in self.inner_frame.winfo_children():
            w.destroy()
        self.gen_photo = ImageTk.PhotoImage(grid_img)
        label = tk.Label(self.inner_frame, image=self.gen_photo)
        label.image = self.gen_photo
        label.pack()
        self.inner_frame.update_idletasks()
        self.gen_canvas.configure(scrollregion=self.gen_canvas.bbox('all'))
        self.gen_info.config(text="Generation complete.")

# ==================== Main ====================

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    root = tk.Tk()
    app = VQGANTransformerApp(root)
    root.mainloop()