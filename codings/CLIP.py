import tkinter as tk
from tkinter import filedialog, ttk
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
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
    img = Image.open(path)
    if img.mode == 'RGBA':
        bg = Image.new('RGB', img.size, (0, 0, 0))
        bg.paste(img, mask=img.split()[3])
        return bg
    return img.convert('RGB')

def load_image_as_grayscale(path):
    return Image.open(path).convert('L')

VOCAB_SIZE = 259
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
ASCII_OFFSET = 3

def tokenize(text, max_len, add_special=True):
    indices = []
    for ch in text[:max_len - (2 if add_special else 0)]:
        idx = ord(ch) if ord(ch) < 256 else 0
        indices.append(idx + ASCII_OFFSET)
    if add_special:
        indices = [SOS_IDX] + indices + [EOS_IDX]
    if len(indices) < max_len:
        indices += [PAD_IDX] * (max_len - len(indices))
    return indices[:max_len]

def detokenize(indices, remove_special=True):
    chars = []
    for idx in indices:
        if remove_special and idx in (PAD_IDX, SOS_IDX, EOS_IDX):
            continue
        if idx >= ASCII_OFFSET:
            chars.append(chr(idx - ASCII_OFFSET))
    return ''.join(chars)

def text_to_indices_ascii(text, max_len=128):
    indices = []
    for ch in text[:max_len]:
        idx = ord(ch) if ord(ch) < 256 else 0
        indices.append(idx)
    if len(indices) < max_len:
        indices += [0] * (max_len - len(indices))
    return indices

# ==================== Image Encoders ====================

class ResidualBlock(nn.Module):
    """Basic residual block for ResNet encoder."""
    def __init__(self, in_channels, out_channels, stride=1, use_batchnorm=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=not use_batchnorm)
        self.bn1 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=not use_batchnorm)
        self.bn2 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=not use_batchnorm),
                nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)

class ResNetImageEncoder(nn.Module):
    """
    ResNet‑style image encoder with optional attention pooling (as per CLIP paper).
    Lightweight version for CPU.
    """
    def __init__(self, in_channels=3, base_channels=64, num_blocks=3, use_batchnorm=True,
                 embedding_dim=256, img_size=32, use_attention_pool=True):
        super().__init__()
        self.use_attention_pool = use_attention_pool
        self.conv1 = nn.Conv2d(in_channels, base_channels, 3, stride=1, padding=1, bias=not use_batchnorm)
        self.bn1 = nn.BatchNorm2d(base_channels) if use_batchnorm else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

        layers = []
        in_ch = base_channels
        for i in range(num_blocks):
            out_ch = base_channels * (2 ** i)
            stride = 2 if i > 0 else 1
            layers.append(ResidualBlock(in_ch, out_ch, stride=stride, use_batchnorm=use_batchnorm))
            in_ch = out_ch
        self.stages = nn.Sequential(*layers)

        if use_attention_pool:
            self.attn_pool = nn.MultiheadAttention(embed_dim=in_ch, num_heads=4, batch_first=True)
            self.attn_query = nn.Parameter(torch.randn(1, 1, in_ch) * 0.02)
        else:
            self.attn_pool = None
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.projection = nn.Linear(in_ch, embedding_dim)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.stages(x)
        if self.use_attention_pool:
            B, C, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)
            query = self.attn_query.expand(B, -1, -1)
            x, _ = self.attn_pool(query, x, x)
            x = x.squeeze(1)
        else:
            x = self.global_pool(x).flatten(1)
        return self.projection(x)

class ViTImageEncoder(nn.Module):
    """Vision Transformer (ViT) image encoder."""
    def __init__(self, in_channels=3, img_size=32, patch_size=4, embed_dim=256,
                 num_heads=8, num_layers=6, mlp_ratio=4, dropout=0.1):
        super().__init__()
        assert img_size % patch_size == 0
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                                   dim_feedforward=int(embed_dim * mlp_ratio),
                                                   dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.transformer(x)
        x = self.norm(x[:, 0, :])
        return x

# ==================== Text Encoders (BiGRU and Transformer) ====================

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

def get_encoder_config(enc_type, size):
    return TEXT_ENCODER_PRESETS[enc_type][size]

# ==================== Joint Embedding Model ====================

class JointEmbeddingModel(nn.Module):
    def __init__(self, image_encoder, text_encoder, temperature=0.07):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))

    def forward(self, images, text_ascii):
        img_emb = self.image_encoder(images)
        txt_emb = self.text_encoder(text_ascii)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp().clamp(max=100)  # Prevent extreme logits
        logits_per_image = logit_scale * img_emb @ txt_emb.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text

def clip_loss(logits_per_image, logits_per_text):
    batch_size = logits_per_image.shape[0]
    labels = torch.arange(batch_size, device=logits_per_image.device)
    loss_i = F_nn.cross_entropy(logits_per_image, labels)
    loss_t = F_nn.cross_entropy(logits_per_text, labels)
    return (loss_i + loss_t) / 2

# ==================== Caption Decoders ====================

class GRUDecoder(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=256, hidden_dim=512, img_embed_dim=256, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers>1 else 0)
        self.img_to_hidden = nn.Linear(img_embed_dim, hidden_dim * num_layers)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, img_emb, captions, teacher_forcing_ratio=1.0):
        B, seq_len = captions.shape
        hidden = self.img_to_hidden(img_emb).view(self.rnn.num_layers, B, self.rnn.hidden_size)
        outputs = []
        input_token = captions[:, 0].unsqueeze(1)
        for t in range(1, seq_len):
            emb = self.embedding(input_token)
            out, hidden = self.rnn(emb, hidden)
            logits = self.fc(out.squeeze(1))
            outputs.append(logits)
            if teacher_forcing_ratio >= random.random():
                input_token = captions[:, t].unsqueeze(1)
            else:
                input_token = logits.argmax(-1).unsqueeze(1)
        return torch.stack(outputs, dim=1)

    def generate(self, img_emb, max_len, cfg_scale=1.0, null_emb=None):
        B = img_emb.size(0)
        hidden = self.img_to_hidden(img_emb).view(self.rnn.num_layers, B, self.rnn.hidden_size)
        hidden_uncond = None
        if null_emb is not None and cfg_scale != 1.0:
            hidden_uncond = self.img_to_hidden(null_emb).view(self.rnn.num_layers, B, self.rnn.hidden_size)
        input_token = torch.full((B, 1), SOS_IDX, dtype=torch.long, device=img_emb.device)
        generated = []
        for _ in range(max_len):
            emb = self.embedding(input_token)
            out, hidden = self.rnn(emb, hidden)
            logits = self.fc(out.squeeze(1))
            if hidden_uncond is not None:
                _, hidden_uncond = self.rnn(emb, hidden_uncond)
                logits_uncond = self.fc(out.squeeze(1))
                logits = logits_uncond + cfg_scale * (logits - logits_uncond)
            token = logits.argmax(-1, keepdim=True)
            generated.append(token)
            if (token == EOS_IDX).all():
                break
            input_token = token
        return torch.cat(generated, dim=1)

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=256, num_heads=8, num_layers=3,
                 ff_dim=512, img_embed_dim=256, max_len=128, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, embed_dim) * 0.02)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads,
                                                   dim_feedforward=ff_dim, dropout=dropout,
                                                   batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.img_proj = nn.Linear(img_embed_dim, embed_dim)
        self.fc = nn.Linear(embed_dim, vocab_size)
        self.max_len = max_len

    def forward(self, img_emb, captions, teacher_forcing_ratio=1.0):
        B, seq_len = captions.shape
        tgt = captions[:, :-1]
        tgt_emb = self.embedding(tgt) + self.pos_embedding[:, :tgt.size(1), :]
        memory = self.img_proj(img_emb).unsqueeze(1)  # (B, 1, embed_dim)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        out = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        return self.fc(out)

    def generate(self, img_emb, max_len, cfg_scale=1.0, null_emb=None):
        B = img_emb.size(0)
        memory = self.img_proj(img_emb).unsqueeze(1)
        memory_uncond = None
        if null_emb is not None and cfg_scale != 1.0:
            memory_uncond = self.img_proj(null_emb).unsqueeze(1)
        generated = torch.full((B, 1), SOS_IDX, dtype=torch.long, device=img_emb.device)
        for _ in range(max_len - 1):
            seq_len = generated.size(1)
            tgt_emb = self.embedding(generated) + self.pos_embedding[:, :seq_len, :]
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(img_emb.device)
            out = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
            logits = self.fc(out[:, -1, :])
            if memory_uncond is not None:
                out_uncond = self.decoder(tgt_emb, memory_uncond, tgt_mask=tgt_mask)
                logits_uncond = self.fc(out_uncond[:, -1, :])
                logits = logits_uncond + cfg_scale * (logits - logits_uncond)
            next_token = logits.argmax(-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if (next_token == EOS_IDX).all():
                break
        return generated[:, 1:]

# ==================== Dataset ====================

class CaptionDataset(Dataset):
    def __init__(self, image_paths, labels_per_image, img_size=32, color_mode='rgb',
                 aug_settings=None, text_max_len=128, decoder_max_len=128):
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
        self.decoder_max_len = decoder_max_len

        # Paper: only random resized crop (no other augmentations by default)
        transform_list = [
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.ToTensor()
        ]
        if color_mode == 'rgb':
            transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        else:
            transform_list.append(transforms.Normalize((0.5,), (0.5,)))
        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.image_paths)

    def apply_augmentations(self, pil_img):
        # Additional augmentations (optional, not in paper)
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
            ascii_text = text_to_indices_ascii(chosen_text, self.text_max_len)
            decoder_text = tokenize(chosen_text, self.decoder_max_len, add_special=True)
            return img_tensor, torch.tensor(ascii_text, dtype=torch.long), torch.tensor(decoder_text, dtype=torch.long)
        except Exception as e:
            print(f"Error loading {self.image_paths[idx]}: {e}")
            if self.color_mode == 'rgb':
                img_tensor = torch.zeros(3, self.img_size, self.img_size)
            else:
                img_tensor = torch.zeros(1, self.img_size, self.img_size)
            return img_tensor, torch.zeros(self.text_max_len, dtype=torch.long), torch.zeros(self.decoder_max_len, dtype=torch.long)

# ==================== GUI Application ====================

class ImageCaptioningApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Two‑Stage Image Captioning (CLIP-based)")

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        win_width = max(1050, int(screen_width * 0.85))
        win_height = max(750, int(screen_height * 0.85))
        self.root.geometry(f"{win_width}x{win_height}")
        self.root.minsize(950, 700)

        # Data
        self.image_paths = []
        self.labels = []
        self.csv_path = None

        # Training flags
        self.training_joint = False
        self.training_decoder = False

        # Models
        self.joint_model = None
        self.decoder_model = None
        self.optimizer_joint = None
        self.optimizer_decoder = None
        self.scheduler_joint = None

        self.current_epoch_joint = 0
        self.current_epoch_decoder = 0

        self.message_queue_joint = queue.Queue()
        self.message_queue_decoder = queue.Queue()

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.log_joint(f"Using device: {self.device}")

        # Settings
        self.settings = {}
        self.aug_settings = {}
        self._setup_settings_vars()
        self.setup_gui()
        self.root.after(100, self.process_messages_joint)
        self.root.after(100, self.process_messages_decoder)

    def _setup_settings_vars(self):
        # General
        self.settings['device'] = tk.StringVar(value='cuda' if torch.cuda.is_available() else 'cpu')
        self.settings['img_size'] = tk.IntVar(value=32)
        self.settings['color_mode'] = tk.StringVar(value='rgb')
        self.settings['max_text_len'] = tk.IntVar(value=128)
        self.settings['max_decoder_len'] = tk.IntVar(value=80)
        self.settings['num_workers'] = tk.IntVar(value=0)

        # Joint model
        self.settings['joint_embed_dim'] = tk.IntVar(value=256)
        self.settings['joint_temperature'] = tk.DoubleVar(value=0.07)
        self.settings['image_backbone'] = tk.StringVar(value='CNN')
        self.settings['text_encoder_type'] = tk.StringVar(value='BiGRU')
        self.settings['text_encoder_size'] = tk.StringVar(value='small')
        self.settings['joint_batch_size'] = tk.IntVar(value=32)
        self.settings['joint_lr'] = tk.DoubleVar(value=1e-4)
        self.settings['joint_epochs'] = tk.StringVar(value="50")
        self.settings['joint_grad_clip'] = tk.DoubleVar(value=1.0)

        # CNN backbone settings
        self.settings['cnn_base_channels'] = tk.IntVar(value=64)
        self.settings['cnn_num_blocks'] = tk.IntVar(value=3)
        self.settings['cnn_batchnorm'] = tk.BooleanVar(value=True)
        self.settings['cnn_attention_pool'] = tk.BooleanVar(value=True)

        # ViT backbone settings
        self.settings['vit_patch_size'] = tk.IntVar(value=4)
        self.settings['vit_num_heads'] = tk.IntVar(value=8)
        self.settings['vit_num_layers'] = tk.IntVar(value=6)
        self.settings['vit_mlp_ratio'] = tk.DoubleVar(value=4.0)
        self.settings['vit_dropout'] = tk.DoubleVar(value=0.1)

        # Decoder
        self.settings['decoder_type'] = tk.StringVar(value='GRU')
        self.settings['decoder_hidden_dim'] = tk.IntVar(value=512)
        self.settings['decoder_layers'] = tk.IntVar(value=2)
        self.settings['decoder_batch_size'] = tk.IntVar(value=32)
        self.settings['decoder_lr'] = tk.DoubleVar(value=1e-4)
        self.settings['decoder_epochs'] = tk.StringVar(value="50")
        self.settings['teacher_forcing_ratio'] = tk.DoubleVar(value=1.0)

        # Augmentations (optional)
        self.aug_settings = {
            'flip_horizontal': tk.BooleanVar(value=False),
            'rotation': tk.BooleanVar(value=False),
        }

    def setup_gui(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Joint Training tab
        self.joint_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.joint_tab, text='Joint Training')
        self._build_joint_tab()

        # Decoder Training tab
        self.decoder_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.decoder_tab, text='Decoder Training')
        self._build_decoder_tab()

        # Caption tab
        self.caption_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.caption_tab, text='Caption')
        self._build_caption_tab()

        # Settings tab
        self.settings_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.settings_tab, text='Settings')
        self._build_settings_tab()

        self.status_label = tk.Label(self.root, text="Ready   Add images/folder/CSV first.", relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    # ---------- Data management ----------
    def _add_images(self):
        files = filedialog.askopenfilenames(filetypes=[("Images", "*.jpg *.jpeg *.png *.jfif *.webp *.bmp")])
        for f in files:
            if f not in self.image_paths:
                self.image_paths.append(f)
        self.log_joint(f"Added {len(files)} images. Total: {len(self.image_paths)}")
        self.log_decoder(f"Added {len(files)} images. Total: {len(self.image_paths)}")

    def _add_folder(self):
        folder = filedialog.askdirectory()
        if not folder: return
        count = 0
        for root_dir, _, files in os.walk(folder):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.jfif', '.webp', '.bmp')):
                    full = os.path.join(root_dir, file)
                    if full not in self.image_paths:
                        self.image_paths.append(full)
                        count += 1
        self.log_joint(f"Added {count} images from folder. Total: {len(self.image_paths)}")
        self.log_decoder(f"Added {count} images from folder. Total: {len(self.image_paths)}")

    def _clear_images(self):
        self.image_paths = []
        self.labels = []
        self.log_joint("Cleared all images.")
        self.log_decoder("Cleared all images.")

    def _load_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if not path: return
        self.csv_path = path
        label_map = {}
        with open(path, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    name = row[0].strip()
                    cap = row[1].strip()
                    label_map.setdefault(name, []).append(cap)
        self.labels = []
        for full in self.image_paths:
            base = os.path.basename(full)
            if base in label_map:
                self.labels.append(label_map[base])
            else:
                self.labels.append([os.path.splitext(base)[0]])
        self.log_joint(f"CSV loaded: {len(label_map)} matched.")
        self.log_decoder(f"CSV loaded: {len(label_map)} matched.")

    def _use_filenames(self):
        self.labels = [[os.path.splitext(os.path.basename(p))[0]] for p in self.image_paths]
        self.log_joint("Using filenames as labels.")
        self.log_decoder("Using filenames as labels.")

    def _use_folders(self):
        self.labels = []
        for p in self.image_paths:
            folder = os.path.basename(os.path.dirname(p)) or 'unknown'
            self.labels.append([folder])
        self.log_joint("Using folder names as labels.")
        self.log_decoder("Using folder names as labels.")

    # ---------- Tab builders ----------
    def _build_joint_tab(self):
        main = tk.Frame(self.joint_tab)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        data_frame = tk.LabelFrame(main, text="Training Data", padx=5, pady=5)
        data_frame.pack(fill=tk.X, pady=(0,10))
        btn_data = tk.Frame(data_frame)
        btn_data.pack()
        tk.Button(btn_data, text="Add Images", command=self._add_images, width=12).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_data, text="Add Folder", command=self._add_folder, width=12).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_data, text="Clear All", command=self._clear_images, width=12).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_data, text="Load CSV", command=self._load_csv, width=12).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_data, text="Filenames as Labels", command=self._use_filenames, width=16).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_data, text="Folders as Labels", command=self._use_folders, width=16).pack(side=tk.LEFT, padx=2)

        train_frame = tk.Frame(main)
        train_frame.pack(fill=tk.X)
        tk.Button(train_frame, text="Initialize Joint Model", command=self._init_joint, width=18).pack(side=tk.LEFT, padx=5)
        tk.Button(train_frame, text="Start Training", command=self._start_joint_training, width=14, bg="lightgreen").pack(side=tk.LEFT, padx=5)
        tk.Button(train_frame, text="Stop Training", command=lambda: setattr(self, 'training_joint', False), width=14, bg="salmon").pack(side=tk.LEFT, padx=5)
        tk.Button(train_frame, text="Save Model", command=self._save_joint, width=12).pack(side=tk.LEFT, padx=5)
        tk.Button(train_frame, text="Load Model", command=self._load_joint, width=12).pack(side=tk.LEFT, padx=5)
        tk.Label(train_frame, text="Epochs:").pack(side=tk.LEFT, padx=(20,0))
        tk.Entry(train_frame, textvariable=self.settings['joint_epochs'], width=6).pack(side=tk.LEFT)

        self.joint_log = tk.Text(main, height=12, font=("Courier",9))
        self.joint_log.pack(fill=tk.BOTH, expand=True, pady=(10,0))
        self.joint_log.insert(tk.END, "Load data and initialize a model to begin.\n")

    def _build_decoder_tab(self):
        main = tk.Frame(self.decoder_tab)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        data_frame = tk.LabelFrame(main, text="Training Data (shared with Joint)", padx=5, pady=5)
        data_frame.pack(fill=tk.X, pady=(0,10))
        tk.Label(data_frame, text="Data must be loaded in the Joint Training tab.", fg="grey").pack()

        train_frame = tk.Frame(main)
        train_frame.pack(fill=tk.X)
        tk.Button(train_frame, text="Initialize Decoder", command=self._init_decoder, width=18).pack(side=tk.LEFT, padx=5)
        tk.Button(train_frame, text="Start Training", command=self._start_decoder_training, width=14, bg="lightgreen").pack(side=tk.LEFT, padx=5)
        tk.Button(train_frame, text="Stop Training", command=lambda: setattr(self, 'training_decoder', False), width=14, bg="salmon").pack(side=tk.LEFT, padx=5)
        tk.Button(train_frame, text="Save Decoder", command=self._save_decoder, width=12).pack(side=tk.LEFT, padx=5)
        tk.Button(train_frame, text="Load Decoder", command=self._load_decoder, width=12).pack(side=tk.LEFT, padx=5)
        tk.Label(train_frame, text="Epochs:").pack(side=tk.LEFT, padx=(20,0))
        tk.Entry(train_frame, textvariable=self.settings['decoder_epochs'], width=6).pack(side=tk.LEFT)

        self.decoder_log = tk.Text(main, height=12, font=("Courier",9))
        self.decoder_log.pack(fill=tk.BOTH, expand=True, pady=(10,0))
        self.decoder_log.insert(tk.END, "Load data and initialise joint model first, then decoder.\n")

    def _build_caption_tab(self):
        main = tk.Frame(self.caption_tab)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        single_frame = tk.LabelFrame(main, text="Single Image", padx=5, pady=5)
        single_frame.pack(fill=tk.X, pady=5)
        tk.Button(single_frame, text="Load Image & Predict", command=self._predict_single, width=18).pack(side=tk.LEFT, padx=5)
        self.single_label = tk.Label(single_frame, text="No image selected")
        self.single_label.pack(side=tk.LEFT, padx=5)
        tk.Label(single_frame, text="CFG scale:").pack(side=tk.LEFT, padx=(20,0))
        self.cfg_scale = tk.DoubleVar(value=2.0)
        tk.Entry(single_frame, textvariable=self.cfg_scale, width=5).pack(side=tk.LEFT, padx=5)
        self.single_result = tk.Text(single_frame, height=3)
        self.single_result.pack(fill=tk.X, pady=5)

        batch_frame = tk.LabelFrame(main, text="Batch Folder → CSV", padx=5, pady=5)
        batch_frame.pack(fill=tk.X, pady=10)
        tk.Button(batch_frame, text="Select Folder & Predict", command=self._batch_predict, width=20).pack(side=tk.LEFT, padx=5)
        self.batch_multi = tk.BooleanVar(value=False)
        tk.Checkbutton(batch_frame, text="Multiple captions (beam=3, pipe-separated)", variable=self.batch_multi).pack(side=tk.LEFT, padx=10)
        self.batch_status = tk.Label(batch_frame, text="")
        self.batch_status.pack(side=tk.LEFT, padx=10)

    def _build_settings_tab(self):
        main_frame = tk.Frame(self.settings_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        canvas = tk.Canvas(main_frame)
        scrollbar = tk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0,0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # System & device
        sys_frame = tk.LabelFrame(scrollable_frame, text="System", padx=10, pady=10)
        sys_frame.pack(fill=tk.X, pady=5)
        self._combo_row(sys_frame, "Device:", 'device', ['cuda', 'cpu'] if torch.cuda.is_available() else ['cpu'])
        self._spin_row(sys_frame, "DataLoader workers:", 'num_workers', 0, 4)

        # Image
        img_frame = tk.LabelFrame(scrollable_frame, text="Image", padx=10, pady=10)
        img_frame.pack(fill=tk.X, pady=5)
        self._spin_row(img_frame, "Image size:", 'img_size', 16, 256)
        self._combo_row(img_frame, "Color mode:", 'color_mode', ['rgb', 'grayscale'])
        self._spin_row(img_frame, "Max text len (ASCII):", 'max_text_len', 16, 256)
        self._spin_row(img_frame, "Max decoder len:", 'max_decoder_len', 10, 200)

        # Augmentations (optional)
        aug_frame = tk.LabelFrame(scrollable_frame, text="Augmentations (optional, not in CLIP paper)", padx=10, pady=10)
        aug_frame.pack(fill=tk.X, pady=5)
        tk.Checkbutton(aug_frame, text="Horizontal flip", variable=self.aug_settings['flip_horizontal']).pack(anchor='w')
        tk.Checkbutton(aug_frame, text="Rotation ±30°", variable=self.aug_settings['rotation']).pack(anchor='w')

        # Joint model
        joint_frame = tk.LabelFrame(scrollable_frame, text="Joint Model", padx=10, pady=10)
        joint_frame.pack(fill=tk.X, pady=5)
        self._combo_row(joint_frame, "Image backbone:", 'image_backbone', ['CNN', 'ViT'])
        self._entry_row(joint_frame, "Embedding dim:", 'joint_embed_dim', 32, 512)
        self._entry_row(joint_frame, "Temperature:", 'joint_temperature', 0.01, 1.0)
        self._combo_row(joint_frame, "Text encoder:", 'text_encoder_type', ['BiGRU', 'BiTransformer'])
        self._combo_row(joint_frame, "Encoder size:", 'text_encoder_size', ['tiny', 'small', 'medium', 'large'])
        self._spin_row(joint_frame, "Batch size:", 'joint_batch_size', 1, 128)
        self._entry_row(joint_frame, "Learning rate:", 'joint_lr', 1e-5, 1e-2)
        self._entry_row(joint_frame, "Grad clip:", 'joint_grad_clip', 0.1, 5.0)

        # CNN specific
        cnn_frame = tk.LabelFrame(scrollable_frame, text="CNN Backbone (ResNet style)", padx=10, pady=10)
        cnn_frame.pack(fill=tk.X, pady=5)
        self._spin_row(cnn_frame, "Base channels:", 'cnn_base_channels', 16, 256)
        self._spin_row(cnn_frame, "Number of blocks:", 'cnn_num_blocks', 1, 5)
        tk.Checkbutton(cnn_frame, text="Use BatchNorm", variable=self.settings['cnn_batchnorm']).pack(anchor='w')
        tk.Checkbutton(cnn_frame, text="Use Attention Pooling (CLIP paper)", variable=self.settings['cnn_attention_pool']).pack(anchor='w')

        # ViT specific
        vit_frame = tk.LabelFrame(scrollable_frame, text="ViT Backbone", padx=10, pady=10)
        vit_frame.pack(fill=tk.X, pady=5)
        self._spin_row(vit_frame, "Patch size:", 'vit_patch_size', 2, 16)
        self._spin_row(vit_frame, "Number of heads:", 'vit_num_heads', 2, 16)
        self._spin_row(vit_frame, "Number of layers:", 'vit_num_layers', 2, 12)
        self._entry_row(vit_frame, "MLP ratio:", 'vit_mlp_ratio', 1.0, 8.0)
        self._entry_row(vit_frame, "Dropout:", 'vit_dropout', 0.0, 0.5)

        # Decoder
        dec_frame = tk.LabelFrame(scrollable_frame, text="Decoder", padx=10, pady=10)
        dec_frame.pack(fill=tk.X, pady=5)
        self._combo_row(dec_frame, "Decoder type:", 'decoder_type', ['GRU', 'Transformer'])
        self._spin_row(dec_frame, "Hidden dim (GRU) / embed dim (Trans):", 'decoder_hidden_dim', 64, 1024)
        self._spin_row(dec_frame, "Num layers:", 'decoder_layers', 1, 6)
        self._spin_row(dec_frame, "Batch size:", 'decoder_batch_size', 1, 128)
        self._entry_row(dec_frame, "Learning rate:", 'decoder_lr', 1e-5, 1e-2)
        self._entry_row(dec_frame, "Teacher forcing ratio:", 'teacher_forcing_ratio', 0.0, 1.0)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def _spin_row(self, parent, label, key, fr, to):
        f = tk.Frame(parent); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text=label, width=30, anchor='w').pack(side=tk.LEFT)
        ttk.Spinbox(f, from_=fr, to=to, textvariable=self.settings[key], width=6).pack(side=tk.RIGHT)

    def _entry_row(self, parent, label, key, fr, to):
        f = tk.Frame(parent); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text=label, width=30, anchor='w').pack(side=tk.LEFT)
        tk.Entry(f, textvariable=self.settings[key], width=8).pack(side=tk.RIGHT)

    def _combo_row(self, parent, label, key, values):
        f = tk.Frame(parent); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text=label, width=30, anchor='w').pack(side=tk.LEFT)
        ttk.Combobox(f, textvariable=self.settings[key], values=values, state='readonly', width=10).pack(side=tk.RIGHT)

    # ---------- Logging ----------
    def log_joint(self, msg):
        self.message_queue_joint.put(msg)
    def log_decoder(self, msg):
        self.message_queue_decoder.put(msg)

    def process_messages_joint(self):
        try:
            while True:
                msg = self.message_queue_joint.get_nowait()
                self.joint_log.insert(tk.END, f"{time.strftime('%H:%M:%S')} - {msg}\n")
                self.joint_log.see(tk.END)
                self.status_label.config(text=msg[:80])
        except queue.Empty:
            pass
        self.root.after(100, self.process_messages_joint)

    def process_messages_decoder(self):
        try:
            while True:
                msg = self.message_queue_decoder.get_nowait()
                self.decoder_log.insert(tk.END, f"{time.strftime('%H:%M:%S')} - {msg}\n")
                self.decoder_log.see(tk.END)
                self.status_label.config(text=msg[:80])
        except queue.Empty:
            pass
        self.root.after(100, self.process_messages_decoder)

    # ---------- Model init ----------
    def _init_joint(self):
        if not self.image_paths:
            self.log_joint("No images loaded! Add images first.")
            return
        try:
            device = torch.device(self.settings['device'].get())
            in_ch = 3 if self.settings['color_mode'].get() == 'rgb' else 1
            img_size = self.settings['img_size'].get()
            embed_dim = self.settings['joint_embed_dim'].get()

            backbone = self.settings['image_backbone'].get()
            if backbone == 'CNN':
                img_enc = ResNetImageEncoder(
                    in_channels=in_ch,
                    base_channels=self.settings['cnn_base_channels'].get(),
                    num_blocks=self.settings['cnn_num_blocks'].get(),
                    use_batchnorm=self.settings['cnn_batchnorm'].get(),
                    embedding_dim=embed_dim,
                    img_size=img_size,
                    use_attention_pool=self.settings['cnn_attention_pool'].get()
                )
            else:  # ViT
                img_enc = ViTImageEncoder(
                    in_channels=in_ch,
                    img_size=img_size,
                    patch_size=self.settings['vit_patch_size'].get(),
                    embed_dim=embed_dim,
                    num_heads=self.settings['vit_num_heads'].get(),
                    num_layers=self.settings['vit_num_layers'].get(),
                    mlp_ratio=self.settings['vit_mlp_ratio'].get(),
                    dropout=self.settings['vit_dropout'].get()
                )

            enc_type = self.settings['text_encoder_type'].get()
            size = self.settings['text_encoder_size'].get()
            cfg = get_encoder_config(enc_type, size)
            if enc_type == 'BiGRU':
                txt_enc = TextEncoder(vocab_size=256, embed_dim=cfg['embed_dim'],
                                      hidden_size=cfg['hidden_size'], num_layers=cfg['num_layers'],
                                      cond_dim=embed_dim)
            else:
                txt_enc = TransformerTextEncoder(vocab_size=256, embed_dim=cfg['embed_dim'],
                                                  num_heads=cfg['num_heads'], num_layers=cfg['num_layers'],
                                                  ff_dim=cfg['ff_dim'], cond_dim=embed_dim,
                                                  max_len=self.settings['max_text_len'].get())
            self.joint_model = JointEmbeddingModel(img_enc, txt_enc, self.settings['joint_temperature'].get())
            self.joint_model.to(device)
            self.optimizer_joint = optim.Adam(self.joint_model.parameters(), lr=self.settings['joint_lr'].get())
            self.log_joint(f"Joint model initialized ({backbone} image encoder, {enc_type} {size} text encoder) on {device}.")
        except Exception as e:
            self.log_joint(f"Error: {e}")

    def _init_decoder(self):
        if not self.joint_model:
            self.log_decoder("Joint model must be loaded/trained first.")
            return
        try:
            img_embed_dim = self.settings['joint_embed_dim'].get()
            dec_type = self.settings['decoder_type'].get()
            if dec_type == 'GRU':
                self.decoder_model = GRUDecoder(
                    embed_dim=256,
                    hidden_dim=self.settings['decoder_hidden_dim'].get(),
                    img_embed_dim=img_embed_dim,
                    num_layers=self.settings['decoder_layers'].get()
                )
            else:
                self.decoder_model = TransformerDecoder(
                    embed_dim=self.settings['decoder_hidden_dim'].get(),
                    num_heads=8,
                    num_layers=self.settings['decoder_layers'].get(),
                    ff_dim=512,
                    img_embed_dim=img_embed_dim,
                    max_len=self.settings['max_decoder_len'].get()
                )
            device = torch.device(self.settings['device'].get())
            self.decoder_model.to(device)
            self.optimizer_decoder = optim.Adam(self.decoder_model.parameters(), lr=self.settings['decoder_lr'].get())
            self.log_decoder(f"Decoder ({dec_type}) initialized on {device}.")
        except Exception as e:
            self.log_decoder(f"Error: {e}")

    # ---------- Training ----------
    def _start_joint_training(self):
        if not self.image_paths:
            self.log_joint("No images loaded.")
            return
        if not self.joint_model:
            self.log_joint("Joint model not initialized.")
            return
        if self.training_joint:
            self.log_joint("Training already in progress.")
            return
        try:
            epochs = int(self.settings['joint_epochs'].get())
        except:
            self.log_joint("Invalid epoch number.")
            return
        self.training_joint = True
        threading.Thread(target=self._run_joint_training, args=(epochs,), daemon=True).start()

    def _run_joint_training(self, epochs):
        try:
            device = torch.device(self.settings['device'].get())
            self.joint_model.to(device)
            dataset = CaptionDataset(self.image_paths, self.labels,
                                     self.settings['img_size'].get(),
                                     self.settings['color_mode'].get(),
                                     self.aug_settings,
                                     self.settings['max_text_len'].get(),
                                     self.settings['max_decoder_len'].get())
            loader = DataLoader(dataset, batch_size=self.settings['joint_batch_size'].get(),
                                shuffle=True, num_workers=self.settings['num_workers'].get(),
                                pin_memory=(device.type=='cuda'))
            # Cosine annealing scheduler (paper)
            self.scheduler_joint = CosineAnnealingLR(self.optimizer_joint, T_max=epochs * len(loader))
            for epoch in range(epochs):
                if not self.training_joint: break
                self.current_epoch_joint = epoch
                epoch_loss = 0.0
                for imgs, ascii_txt, _ in loader:
                    if not self.training_joint: break
                    imgs = imgs.to(device)
                    ascii_txt = ascii_txt.to(device)
                    self.optimizer_joint.zero_grad()
                    logits_i, logits_t = self.joint_model(imgs, ascii_txt)
                    loss = clip_loss(logits_i, logits_t)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.joint_model.parameters(), self.settings['joint_grad_clip'].get())
                    self.optimizer_joint.step()
                    self.scheduler_joint.step()
                    epoch_loss += loss.item()
                avg_loss = epoch_loss / len(loader)
                self.log_joint(f"Epoch {epoch+1}/{epochs} loss: {avg_loss:.4f}")
            self.training_joint = False
            self.log_joint("Joint training finished.")
        except Exception as e:
            self.log_joint(f"Training error: {e}")
            self.training_joint = False

    def _start_decoder_training(self):
        if not self.image_paths:
            self.log_decoder("No images loaded.")
            return
        if not self.joint_model:
            self.log_decoder("Joint model required for image embeddings.")
            return
        if not self.decoder_model:
            self.log_decoder("Decoder not initialized.")
            return
        if self.training_decoder:
            self.log_decoder("Training already in progress.")
            return
        try:
            epochs = int(self.settings['decoder_epochs'].get())
        except:
            self.log_decoder("Invalid epoch number.")
            return
        self.training_decoder = True
        threading.Thread(target=self._run_decoder_training, args=(epochs,), daemon=True).start()

    def _run_decoder_training(self, epochs):
        try:
            device = torch.device(self.settings['device'].get())
            self.joint_model.to(device)
            self.decoder_model.to(device)
            dataset = CaptionDataset(self.image_paths, self.labels,
                                     self.settings['img_size'].get(),
                                     self.settings['color_mode'].get(),
                                     self.aug_settings,
                                     self.settings['max_text_len'].get(),
                                     self.settings['max_decoder_len'].get())
            loader = DataLoader(dataset, batch_size=self.settings['decoder_batch_size'].get(),
                                shuffle=True, num_workers=self.settings['num_workers'].get(),
                                pin_memory=(device.type=='cuda'))
            teacher_forcing = self.settings['teacher_forcing_ratio'].get()
            for epoch in range(epochs):
                if not self.training_decoder: break
                self.current_epoch_decoder = epoch
                epoch_loss = 0.0
                for imgs, _, decoder_txt in loader:
                    if not self.training_decoder: break
                    imgs = imgs.to(device)
                    decoder_txt = decoder_txt.to(device)
                    with torch.no_grad():
                        img_emb = self.joint_model.image_encoder(imgs)
                    self.optimizer_decoder.zero_grad()
                    logits = self.decoder_model(img_emb, decoder_txt, teacher_forcing_ratio=teacher_forcing)
                    loss = F_nn.cross_entropy(logits.reshape(-1, VOCAB_SIZE),
                                              decoder_txt[:, 1:].reshape(-1), ignore_index=PAD_IDX)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.decoder_model.parameters(), 1.0)
                    self.optimizer_decoder.step()
                    epoch_loss += loss.item()
                avg_loss = epoch_loss / len(loader)
                self.log_decoder(f"Epoch {epoch+1}/{epochs} loss: {avg_loss:.4f}")
            self.training_decoder = False
            self.log_decoder("Decoder training finished.")
        except Exception as e:
            self.log_decoder(f"Training error: {e}")
            self.training_decoder = False

    # ---------- Save/Load ----------
    def _save_joint(self):
        if not self.joint_model: return
        path = filedialog.asksaveasfilename(defaultextension=".pth")
        if path:
            torch.save({'model_state': self.joint_model.state_dict(),
                        'optimizer': self.optimizer_joint.state_dict()}, path)
            self.log_joint(f"Saved to {path}")

    def _load_joint(self):
        path = filedialog.askopenfilename(filetypes=[("PyTorch", "*.pth")])
        if not path: return
        try:
            device = torch.device(self.settings['device'].get())
            if not self.joint_model:
                self._init_joint()
            ckpt = torch.load(path, map_location=device)
            self.joint_model.load_state_dict(ckpt['model_state'])
            self.optimizer_joint.load_state_dict(ckpt['optimizer'])
            self.log_joint(f"Loaded from {path}")
        except Exception as e:
            self.log_joint(f"Load error: {e}")

    def _save_decoder(self):
        if not self.decoder_model: return
        path = filedialog.asksaveasfilename(defaultextension=".pth")
        if path:
            torch.save({'model_state': self.decoder_model.state_dict(),
                        'optimizer': self.optimizer_decoder.state_dict()}, path)
            self.log_decoder(f"Saved to {path}")

    def _load_decoder(self):
        path = filedialog.askopenfilename(filetypes=[("PyTorch", "*.pth")])
        if not path: return
        try:
            device = torch.device(self.settings['device'].get())
            if not self.decoder_model:
                self._init_decoder()
            ckpt = torch.load(path, map_location=device)
            self.decoder_model.load_state_dict(ckpt['model_state'])
            self.optimizer_decoder.load_state_dict(ckpt['optimizer'])
            self.log_decoder(f"Loaded from {path}")
        except Exception as e:
            self.log_decoder(f"Load error: {e}")

    # ---------- Caption generation ----------
    def _preprocess_image(self, pil_img):
        color_mode = self.settings['color_mode'].get()
        if color_mode == 'rgb':
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
        else:
            if pil_img.mode != 'L':
                pil_img = pil_img.convert('L')
        img_size = self.settings['img_size'].get()
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5) if color_mode == 'rgb' else (0.5,),
                                 (0.5, 0.5, 0.5) if color_mode == 'rgb' else (0.5,))
        ])
        return transform(pil_img).unsqueeze(0)

    def _predict_single(self):
        if not self.joint_model or not self.decoder_model:
            self.single_result.delete('1.0', tk.END)
            self.single_result.insert(tk.END, "Models not loaded.\n")
            return
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png *.jfif *.webp *.bmp")])
        if not path: return
        try:
            device = torch.device(self.settings['device'].get())
            self.joint_model.to(device)
            self.decoder_model.to(device)
            img = Image.open(path)
            if img.mode == 'RGBA':
                bg = Image.new('RGB', img.size, (0,0,0))
                bg.paste(img, mask=img.split()[3])
                img = bg
            img_tensor = self._preprocess_image(img).to(device)
            with torch.no_grad():
                img_emb = self.joint_model.image_encoder(img_tensor)
                null_emb = torch.zeros_like(img_emb) if self.cfg_scale.get() != 1.0 else None
                indices = self.decoder_model.generate(img_emb, self.settings['max_decoder_len'].get(),
                                                      cfg_scale=self.cfg_scale.get(), null_emb=null_emb)
                caption = detokenize(indices[0].cpu().tolist())
            self.single_label.config(text=os.path.basename(path))
            self.single_result.delete('1.0', tk.END)
            self.single_result.insert(tk.END, caption)
        except Exception as e:
            self.single_result.delete('1.0', tk.END)
            self.single_result.insert(tk.END, f"Error: {e}")

    def _batch_predict(self):
        if not self.joint_model or not self.decoder_model:
            self.batch_status.config(text="Models not loaded.")
            return
        folder = filedialog.askdirectory()
        if not folder: return
        out_csv = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not out_csv: return

        img_exts = ('.png', '.jpg', '.jpeg', '.jfif', '.webp', '.bmp')
        files = [f for f in os.listdir(folder) if f.lower().endswith(img_exts)]
        if not files:
            self.batch_status.config(text="No images found.")
            return

        device = torch.device(self.settings['device'].get())
        self.joint_model.to(device)
        self.decoder_model.to(device)
        cfg = self.cfg_scale.get()
        max_len = self.settings['max_decoder_len'].get()
        with open(out_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['image', 'caption'])
            for fname in files:
                full = os.path.join(folder, fname)
                try:
                    img = Image.open(full)
                    if img.mode == 'RGBA':
                        bg = Image.new('RGB', img.size, (0,0,0))
                        bg.paste(img, mask=img.split()[3])
                        img = bg
                    img_tensor = self._preprocess_image(img).to(device)
                    with torch.no_grad():
                        img_emb = self.joint_model.image_encoder(img_tensor)
                        null_emb = torch.zeros_like(img_emb) if cfg != 1.0 else None
                        indices = self.decoder_model.generate(img_emb, max_len,
                                                              cfg_scale=cfg, null_emb=null_emb)
                        caption = detokenize(indices[0].cpu().tolist())
                    writer.writerow([full, caption])
                except Exception as e:
                    writer.writerow([full, f"ERROR: {e}"])
        self.batch_status.config(text=f"CSV saved: {out_csv}")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    root = tk.Tk()
    app = ImageCaptioningApp(root)
    root.mainloop()