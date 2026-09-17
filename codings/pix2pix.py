# pix2pix_canny.py
# Pix2Pix with Canny edges, adjustable loss weights, auto-update drawing, right-click erase, integrated image loading.
import tkinter as tk
from tkinter import filedialog, ttk
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
import cv2
import io

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

def get_norm(channels):
    if channels % 32 == 0:
        return nn.GroupNorm(32, channels)
    best = 1
    for g in range(32, 0, -1):
        if channels % g == 0:
            best = g
            break
    return nn.GroupNorm(best, channels)

# ==================== SSIM Loss (channels from target) ====================

def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2 / float(2*sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim_loss(img1, img2, window_size=11, size_average=True):
    _, channel, _, _ = img2.size()  # use target's channels
    window = create_window(window_size, channel).to(img1.device)
    mu1 = F_nn.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F_nn.conv2d(img2, window, padding=window_size//2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F_nn.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F_nn.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F_nn.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

# ==================== Canny Edge Detection ====================

def canny_edge_detection(tensor_img, low_threshold=50, high_threshold=150):
    if tensor_img.shape[1] == 3:
        gray = 0.2989 * tensor_img[:, 0:1] + 0.5870 * tensor_img[:, 1:2] + 0.1140 * tensor_img[:, 2:3]
    else:
        gray = tensor_img
    gray_01 = (gray + 1) / 2
    gray_01 = torch.clamp(gray_01, 0, 1)
    gray_np = (gray_01 * 255).byte().cpu().numpy()
    edges = []
    for i in range(gray_np.shape[0]):
        img = gray_np[i, 0]
        edge = cv2.Canny(img, low_threshold, high_threshold)
        edge_float = edge.astype(np.float32) / 255.0
        edges.append(edge_float)
    edge_tensor = torch.tensor(np.array(edges), dtype=torch.float32, device=tensor_img.device).unsqueeze(1)
    return edge_tensor * 2 - 1

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

# ==================== Pix2Pix UNet ====================

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
    def __init__(self, in_channels, out_channels, dropout=0.1,
                 has_attn=False, padding_mode='constant'):
        super().__init__()
        self.conv1 = Conv2dWithPadding(in_channels, out_channels, 3, padding_mode=padding_mode)
        self.norm1 = get_norm(out_channels)
        self.conv2 = Conv2dWithPadding(out_channels, out_channels, 3, padding_mode=padding_mode)
        self.norm2 = get_norm(out_channels)
        self.res_conv = Conv2dWithPadding(in_channels, out_channels, 1, padding_mode=padding_mode) if in_channels != out_channels else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.attn = AttentionBlock(out_channels, padding_mode=padding_mode) if has_attn else nn.Identity()

    def forward(self, x):
        h = self.conv1(x)
        h = self.norm1(h)
        h = F_nn.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = F_nn.silu(h)
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
        self.attn = AttentionBlock(out_channels, padding_mode=padding_mode) if has_attn else nn.Identity()

    def forward(self, x):
        h = self.conv1(x)
        h = self.norm1(h)
        h = F_nn.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = F_nn.silu(h)
        return self.attn(h + self.res_conv(x))

class Pix2PixUNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=64,
                 img_size=32, channel_mult=(1, 2, 3, 4), dropout=0.1,
                 use_attention=True, padding_mode='constant'):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
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

        self.init_conv = Conv2dWithPadding(in_channels, base_channels, 3, padding_mode=padding_mode)

        self.downs = nn.ModuleList()
        cur_channels = base_channels
        for i, mult in enumerate(channel_mult_used):
            out_channels = base_channels * mult
            block = DownBlock(cur_channels, out_channels, dropout,
                              has_attn=use_attention, padding_mode=padding_mode)
            self.downs.append(block)
            if i < len(channel_mult_used) - 1:
                self.downs.append(nn.Conv2d(out_channels, out_channels, 4, stride=2, padding=1))
            cur_channels = out_channels

        self.mid_block1 = DownBlock(cur_channels, cur_channels, dropout,
                                    has_attn=use_attention, padding_mode=padding_mode)
        self.mid_block2 = UpBlock(cur_channels, cur_channels, dropout,
                                  has_attn=use_attention, padding_mode=padding_mode)

        self.ups = nn.ModuleList()
        rev_blocks = list(reversed(channel_mult_used))
        for i, mult in enumerate(rev_blocks):
            out_channels = base_channels * mult
            block = UpBlock(cur_channels + out_channels, out_channels, dropout,
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

    def forward(self, x):
        x = self.init_conv(x)
        skips = []
        for layer in self.downs:
            if isinstance(layer, DownBlock):
                x = layer(x)
                skips.append(x)
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

# ==================== Dataset ====================

class Pix2PixDataset(Dataset):
    def __init__(self, image_paths, img_size=32, color_mode='rgb',
                 aug_settings=None, canny_low=50, canny_high=150):
        self.image_paths = image_paths
        self.img_size = img_size
        self.color_mode = color_mode.lower()
        self.aug_settings = aug_settings or {}
        self.canny_low = canny_low
        self.canny_high = canny_high

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
            img_tensor = self.base_transform(pil_img)   # [-1,1]

            edge_tensor = canny_edge_detection(img_tensor.unsqueeze(0),
                                               low_threshold=self.canny_low,
                                               high_threshold=self.canny_high).squeeze(0)  # (1,H,W)

            target_channels = img_tensor.shape[0]
            if edge_tensor.shape[0] != target_channels:
                edge_tensor = edge_tensor.repeat(target_channels, 1, 1)

            return edge_tensor, img_tensor
        except Exception as e:
            print(f"Error loading {self.image_paths[idx]}: {e}")
            C = 3 if self.color_mode == 'rgb' else 1
            img_tensor = torch.zeros(C, self.img_size, self.img_size)
            edge_tensor = torch.zeros(C, self.img_size, self.img_size)
            return edge_tensor, img_tensor

# ==================== Trainer ====================

class Pix2PixTrainer:
    def __init__(self, model, device, lr=2e-4):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def train_step(self, edge, target, ssim_w=0.5, l1_w=0.5, l2_w=0.0):
        edge = edge.to(self.device)
        target = target.to(self.device)
        output = self.model(edge)
        # Adapt channels if needed
        if output.shape[1] != target.shape[1]:
            if output.shape[1] == 1 and target.shape[1] == 3:
                output = output.repeat(1, 3, 1, 1)
            elif output.shape[1] == 3 and target.shape[1] == 1:
                output = output.mean(dim=1, keepdim=True)
        l1 = self.l1_loss(output, target)
        l2 = self.mse_loss(output, target)
        ssim_val = ssim_loss(output, target)
        ssim_loss_val = 1 - ssim_val
        loss = ssim_w * ssim_loss_val + l1_w * l1 + l2_w * l2
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item(), l1.item(), ssim_val.item(), l2.item()

    @torch.no_grad()
    def generate(self, edge):
        edge = edge.to(self.device)
        output = self.model(edge)
        return output

    def save(self, path, in_channels):
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'in_channels': in_channels,
        }, path)

    def load(self, path, map_location='cpu'):
        ckpt = torch.load(path, map_location=map_location)
        self.model.load_state_dict(ckpt['model_state'])
        self.optimizer.load_state_dict(ckpt['optimizer_state'])
        return ckpt.get('in_channels', 3)

# ==================== GUI Application ====================

class Pix2PixApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pix2Pix - Canny Edge Translation ")

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
        self.training = False
        self.trainer = None
        self.current_epoch = 0

        self.message_queue_dataset = queue.Queue()
        self.message_queue_train = queue.Queue()

        # Settings
        self.settings = {
            'img_size': tk.IntVar(value=64),
            'color_mode': tk.StringVar(value='rgb'),
            'base_channels': tk.IntVar(value=64),
            'batch_size': tk.IntVar(value=8),
            'lr': tk.DoubleVar(value=2e-4),
            'use_attention': tk.BooleanVar(value=True),
            'padding_mode': tk.StringVar(value='constant'),
            'preview_enabled': tk.BooleanVar(value=True),
            'preview_epoch_freq': tk.IntVar(value=5),
            'canny_low': tk.IntVar(value=50),
            'canny_high': tk.IntVar(value=150),
            'ssim_weight': tk.DoubleVar(value=0.5),
            'l1_weight': tk.DoubleVar(value=0.5),
            'l2_weight': tk.DoubleVar(value=0.0),
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

        self.thumbnail_size = 128
        self.last_x = None
        self.last_y = None
        self.auto_update_id = None
        self.auto_update_seconds = 0

        self.setup_gui()
        self.root.after(100, self.process_messages_dataset)
        self.root.after(100, self.process_messages_train)

    # ---------- GUI Setup ----------
    def setup_gui(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.dataset_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.dataset_tab, text='Dataset')
        self.setup_dataset_tab()

        self.train_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.train_tab, text='Train')
        self.setup_train_tab()

        self.settings_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.settings_tab, text='Settings')
        self.setup_settings_tab()

        self.edge_to_image_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.edge_to_image_tab, text='Edge → Image')
        self.setup_edge_to_image_tab()

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

        log_frame = tk.LabelFrame(left_frame, text="Log", padx=5, pady=5)
        log_frame.pack(fill=tk.BOTH, expand=True)
        self.dataset_log_text = tk.Text(log_frame, height=15, font=("Courier",9))
        self.dataset_log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = tk.Scrollbar(log_frame, command=self.dataset_log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.dataset_log_text.config(yscrollcommand=scrollbar.set)

    def setup_train_tab(self):
        main_frame = tk.Frame(self.train_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_frame = tk.Frame(main_frame, width=300)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0,10))
        left_frame.pack_propagate(False)

        tk.Label(left_frame, text="Pix2Pix Training", font=("Arial",12,"bold")).pack(pady=(0,10))

        tk.Button(left_frame, text="Initialize Model", command=self.init_model, width=20).pack(pady=5)
        epoch_frame = tk.Frame(left_frame)
        epoch_frame.pack(pady=5)
        tk.Label(epoch_frame, text="Epochs:").pack(side=tk.LEFT)
        self.train_epoch_var = tk.StringVar(value="100")
        tk.Entry(epoch_frame, textvariable=self.train_epoch_var, width=8).pack(side=tk.LEFT, padx=5)

        tk.Button(left_frame, text="Start Training", command=self.start_training,
                  width=20, bg="lightgreen").pack(pady=5)
        tk.Button(left_frame, text="Stop Training", command=self.stop_training,
                  width=20, bg="salmon").pack(pady=5)
        tk.Button(left_frame, text="Save Model", command=self.save_model, width=20).pack(pady=5)
        tk.Button(left_frame, text="Load Model", command=self.load_model, width=20).pack(pady=5)

        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        preview_frame = tk.LabelFrame(right_frame, text="Preview (4x4: edges top, outputs bottom)", padx=5, pady=5)
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=(0,5))
        self.train_preview_canvas = tk.Canvas(preview_frame, bg='gray', width=512, height=256)
        self.train_preview_canvas.pack()

        log_frame = tk.LabelFrame(right_frame, text="Log", padx=5, pady=5)
        log_frame.pack(fill=tk.BOTH, expand=True)
        self.train_log_text = tk.Text(log_frame, height=15, font=("Courier",9))
        self.train_log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = tk.Scrollbar(log_frame, command=self.train_log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.train_log_text.config(yscrollcommand=scrollbar.set)

    def setup_settings_tab(self):
        main_frame = tk.Frame(self.settings_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        tk.Label(main_frame, text="Model & Training Settings", font=("Arial",14,"bold")).pack(pady=(0,20))

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
        spin = ttk.Spinbox(f, from_=16, to=256, textvariable=self.settings['img_size'], width=8)
        spin.pack(side=tk.RIGHT)

        # Canny settings
        canny_frame = tk.LabelFrame(scrollable_frame, text="Canny Edge Detection", padx=10, pady=10)
        canny_frame.pack(fill=tk.X, pady=5)
        f = tk.Frame(canny_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Low threshold:", width=20, anchor='w').pack(side=tk.LEFT)
        spin = ttk.Spinbox(f, from_=10, to=200, textvariable=self.settings['canny_low'], width=8)
        spin.pack(side=tk.RIGHT)
        f = tk.Frame(canny_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="High threshold:", width=20, anchor='w').pack(side=tk.LEFT)
        spin = ttk.Spinbox(f, from_=50, to=300, textvariable=self.settings['canny_high'], width=8)
        spin.pack(side=tk.RIGHT)

        # Architecture
        arch_frame = tk.LabelFrame(scrollable_frame, text="Architecture", padx=10, pady=10)
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
        tk.Label(f, text="Padding mode:", width=20, anchor='w').pack(side=tk.LEFT)
        pad_combo = ttk.Combobox(f, textvariable=self.settings['padding_mode'],
                                 values=['constant', 'reflect', 'replicate', 'circular'],
                                 state='readonly', width=10)
        pad_combo.pack(side=tk.RIGHT)

        # Loss weights
        loss_frame = tk.LabelFrame(scrollable_frame, text="Loss Weights", padx=10, pady=10)
        loss_frame.pack(fill=tk.X, pady=5)
        for label, key in [("SSIM weight:", 'ssim_weight'),
                           ("L1 weight:", 'l1_weight'),
                           ("L2 weight:", 'l2_weight')]:
            f = tk.Frame(loss_frame); f.pack(fill=tk.X, pady=2)
            tk.Label(f, text=label, width=20, anchor='w').pack(side=tk.LEFT)
            entry = tk.Entry(f, textvariable=self.settings[key], width=8)
            entry.pack(side=tk.RIGHT)

        # Training
        train_frame = tk.LabelFrame(scrollable_frame, text="Training", padx=10, pady=10)
        train_frame.pack(fill=tk.X, pady=5)
        f = tk.Frame(train_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Batch size:", width=20, anchor='w').pack(side=tk.LEFT)
        spin = ttk.Spinbox(f, from_=1, to=32, textvariable=self.settings['batch_size'], width=8)
        spin.pack(side=tk.RIGHT)
        f = tk.Frame(train_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Learning rate:", width=20, anchor='w').pack(side=tk.LEFT)
        spin = tk.Entry(f, textvariable=self.settings['lr'], width=8)
        spin.pack(side=tk.RIGHT)

        # Preview
        preview_frame = tk.LabelFrame(scrollable_frame, text="Preview", padx=10, pady=10)
        preview_frame.pack(fill=tk.X, pady=5)
        f = tk.Frame(preview_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Enable preview:", width=20, anchor='w').pack(side=tk.LEFT)
        cb = tk.Checkbutton(f, variable=self.settings['preview_enabled'])
        cb.pack(side=tk.RIGHT)
        f = tk.Frame(preview_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Every N epochs:", width=20, anchor='w').pack(side=tk.LEFT)
        spin = ttk.Spinbox(f, from_=1, to=50, textvariable=self.settings['preview_epoch_freq'], width=8)
        spin.pack(side=tk.RIGHT)

        # Augmentations
        aug_frame = tk.LabelFrame(scrollable_frame, text="Data Augmentations", padx=10, pady=10)
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

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # ---------- Edge → Image Tab (integrated with auto-update, right-click erase, load image) ----------
    def setup_edge_to_image_tab(self):
        main_frame = tk.Frame(self.edge_to_image_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        top_frame = tk.Frame(main_frame)
        top_frame.pack(fill=tk.X, pady=5)

        # Drawing area
        draw_frame = tk.Frame(top_frame)
        draw_frame.pack(side=tk.LEFT, padx=(0,10))

        # We'll use a display canvas that is scaled (e.g., 400x400) but the actual drawing is at model resolution.
        self.display_size = 400  # fixed display size
        self.draw_canvas = tk.Canvas(draw_frame, width=self.display_size, height=self.display_size, bg='black')
        self.draw_canvas.pack()
        # Internal image at model resolution (will be resized when model size changes)
        self.draw_img = None
        self.draw = None
        self.photo_display = None  # for display

        # Create initial drawing image at current img_size
        self._init_draw_image()

        # Controls
        controls_frame = tk.Frame(draw_frame)
        controls_frame.pack(pady=5)
        tk.Button(controls_frame, text="Clear", command=self.clear_draw).pack(side=tk.LEFT, padx=5)
        tk.Button(controls_frame, text="Load Image", command=self.load_image_to_draw).pack(side=tk.LEFT, padx=5)
        tk.Button(controls_frame, text="Generate", command=self.manual_generate, bg="lightgreen").pack(side=tk.LEFT, padx=5)

        # Brush size
        tk.Label(controls_frame, text="Brush:").pack(side=tk.LEFT, padx=(10,0))
        self.brush_size_var = tk.IntVar(value=4)
        tk.Spinbox(controls_frame, from_=1, to=20, textvariable=self.brush_size_var, width=4).pack(side=tk.LEFT, padx=2)

        # Auto-update interval
        tk.Label(controls_frame, text="Auto-update (sec):").pack(side=tk.LEFT, padx=(10,0))
        self.auto_update_var = tk.DoubleVar(value=0.0)
        tk.Entry(controls_frame, textvariable=self.auto_update_var, width=5).pack(side=tk.LEFT, padx=2)
        tk.Button(controls_frame, text="Start Auto", command=self.start_auto_update).pack(side=tk.LEFT, padx=2)
        tk.Button(controls_frame, text="Stop Auto", command=self.stop_auto_update).pack(side=tk.LEFT, padx=2)

        # Output display
        self.output_label = tk.Label(top_frame, text="Output", bg='lightgray', width=30, height=15)
        self.output_label.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Bind mouse events: left button for drawing, right button for erase
        self.draw_canvas.bind("<Button-1>", self.start_paint)
        self.draw_canvas.bind("<B1-Motion>", self.paint)
        self.draw_canvas.bind("<ButtonRelease-1>", self.stop_paint)
        self.draw_canvas.bind("<Button-3>", self.start_erase)
        self.draw_canvas.bind("<B3-Motion>", self.erase)
        self.draw_canvas.bind("<ButtonRelease-3>", self.stop_erase)

        self.last_x = None
        self.last_y = None
        self.is_erasing = False

        # Status info
        self.gen_info = tk.Label(main_frame, text="", fg="blue")
        self.gen_info.pack()

        # Initialize display
        self.update_display()

    def _init_draw_image(self):
        """Create or resize internal drawing image to match current img_size."""
        img_size = self.settings['img_size'].get()
        self.draw_img = Image.new('L', (img_size, img_size), 'black')
        self.draw = ImageDraw.Draw(self.draw_img)
        self.update_display()

    def update_display(self):
        """Scale the internal image to display size and show on canvas."""
        if self.draw_img is None:
            return
        display_img = self.draw_img.resize((self.display_size, self.display_size), Image.NEAREST)
        self.photo_display = ImageTk.PhotoImage(display_img)
        self.draw_canvas.delete("all")
        self.draw_canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_display)
        self.draw_canvas.image = self.photo_display  # keep reference

    def _map_coords(self, event):
        """Map canvas coordinates to internal image coordinates."""
        img_size = self.settings['img_size'].get()
        x = int(event.x * img_size / self.display_size)
        y = int(event.y * img_size / self.display_size)
        # Clamp
        x = max(0, min(img_size-1, x))
        y = max(0, min(img_size-1, y))
        return x, y

    def start_paint(self, event):
        self.last_x, self.last_y = self._map_coords(event)
        self.is_erasing = False

    def paint(self, event):
        if self.last_x is None or self.last_y is None:
            return
        x, y = self._map_coords(event)
        size = self.brush_size_var.get()
        # Draw on internal image
        self.draw.line([self.last_x, self.last_y, x, y], fill='white', width=size)
        self.last_x, self.last_y = x, y
        # Update display
        self.update_display()
        # Trigger auto-update if applicable
        self.auto_generate()

    def stop_paint(self, event):
        self.last_x = None
        self.last_y = None

    def start_erase(self, event):
        self.last_x, self.last_y = self._map_coords(event)
        self.is_erasing = True

    def erase(self, event):
        if not self.is_erasing or self.last_x is None or self.last_y is None:
            return
        x, y = self._map_coords(event)
        size = self.brush_size_var.get()
        self.draw.line([self.last_x, self.last_y, x, y], fill='black', width=size)
        self.last_x, self.last_y = x, y
        self.update_display()
        self.auto_generate()

    def stop_erase(self, event):
        self.last_x = None
        self.last_y = None
        self.is_erasing = False

    def clear_draw(self):
        img_size = self.settings['img_size'].get()
        self.draw_img = Image.new('L', (img_size, img_size), 'black')
        self.draw = ImageDraw.Draw(self.draw_img)
        self.update_display()
        self.auto_generate()

    def load_image_to_draw(self):
        """Load an image, compute its Canny edge, and set it as the drawing."""
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")])
        if not path:
            return
        try:
            pil = load_image_as_rgb(path)
            img_size = self.settings['img_size'].get()
            transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
            ])
            tensor = transform(pil).unsqueeze(0)
            edge = canny_edge_detection(tensor,
                                        low_threshold=self.settings['canny_low'].get(),
                                        high_threshold=self.settings['canny_high'].get())
            # Convert to PIL (single channel, values 0..255)
            edge_img = (edge.squeeze(0).cpu() + 1) / 2
            edge_img = edge_img.clamp(0,1)
            edge_pil = transforms.ToPILImage()(edge_img)  # 'L' mode
            self.draw_img = edge_pil
            self.draw = ImageDraw.Draw(self.draw_img)
            self.update_display()
            self.gen_info.config(text="Loaded edge image")
            self.auto_generate()
        except Exception as e:
            self.gen_info.config(text=f"Error loading image: {e}")

    def auto_generate(self):
        """If auto-update is active, schedule generation after the interval."""
        if self.auto_update_seconds > 0:
            if self.auto_update_id is not None:
                self.root.after_cancel(self.auto_update_id)
                self.auto_update_id = None
            self.auto_update_id = self.root.after(int(self.auto_update_seconds * 1000), self.manual_generate)

    def start_auto_update(self):
        try:
            sec = float(self.auto_update_var.get())
            if sec <= 0:
                self.gen_info.config(text="Auto-update interval must be > 0")
                return
            self.auto_update_seconds = sec
            self.gen_info.config(text=f"Auto-update every {sec}s")
            # Trigger immediate generation
            self.manual_generate()
            # Schedule next
            if self.auto_update_id is not None:
                self.root.after_cancel(self.auto_update_id)
            self.auto_update_id = self.root.after(int(sec * 1000), self._auto_update_loop)
        except ValueError:
            self.gen_info.config(text="Invalid interval")

    def _auto_update_loop(self):
        if self.auto_update_seconds > 0:
            self.manual_generate()
            self.auto_update_id = self.root.after(int(self.auto_update_seconds * 1000), self._auto_update_loop)
        else:
            self.auto_update_id = None

    def stop_auto_update(self):
        self.auto_update_seconds = 0
        if self.auto_update_id is not None:
            self.root.after_cancel(self.auto_update_id)
            self.auto_update_id = None
        self.gen_info.config(text="Auto-update stopped")

    def manual_generate(self):
        if self.trainer is None:
            self.gen_info.config(text="Model not loaded")
            return
        # Convert drawing to tensor
        img_size = self.settings['img_size'].get()
        # Ensure drawing is at correct size
        if self.draw_img.size != (img_size, img_size):
            self.draw_img = self.draw_img.resize((img_size, img_size), Image.NEAREST)
            self.draw = ImageDraw.Draw(self.draw_img)
        img_tensor = transforms.ToTensor()(self.draw_img).unsqueeze(0) * 2 - 1
        if self.settings['color_mode'].get() == 'rgb':
            edge = img_tensor.repeat(1, 3, 1, 1)
        else:
            edge = img_tensor
        edge = edge.to(self.trainer.device)
        with torch.no_grad():
            output = self.trainer.generate(edge)
        output = (output + 1) / 2
        output = output.clamp(0,1).cpu()
        if output.shape[1] == 3:
            out_img = transforms.ToPILImage()(output.squeeze(0))
        else:
            out_img = transforms.ToPILImage()(output.squeeze(0).repeat(3,1,1))
        out_img = out_img.resize((self.display_size, self.display_size), Image.NEAREST)
        self.output_photo = ImageTk.PhotoImage(out_img)
        self.output_label.config(image=self.output_photo, text="")
        self.output_label.image = self.output_photo
        self.gen_info.config(text="Generated")

    # ---------- Core functions (unchanged from previous) ----------
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
        self.image_listbox.delete(0, tk.END)
        self.log_dataset("Cleared all images")

    def init_model(self):
        try:
            color_mode = self.settings['color_mode'].get()
            in_channels = 3 if color_mode == 'rgb' else 1
            img_size = self.settings['img_size'].get()
            use_attention = self.settings['use_attention'].get()
            padding_mode = self.settings['padding_mode'].get()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = Pix2PixUNet(
                in_channels=in_channels,
                base_channels=self.settings['base_channels'].get(),
                img_size=img_size,
                channel_mult=(1,2,3,4),
                dropout=0.1,
                use_attention=use_attention,
                padding_mode=padding_mode
            )
            self.trainer = Pix2PixTrainer(model, device, lr=self.settings['lr'].get())
            self.log_train(f"Model initialized: {sum(p.numel() for p in model.parameters()):,} params, in_channels={in_channels}")
            # Re-initialize drawing image to match new size
            self._init_draw_image()
        except Exception as e:
            self.log_train(f"Init error: {e}")

    def start_training(self):
        if not self.image_paths:
            self.log_train("No images!")
            return
        if self.trainer is None:
            self.log_train("Model not initialized!")
            return
        if self.training:
            self.log_train("Already training.")
            return
        try:
            epochs = int(self.train_epoch_var.get())
        except:
            self.log_train("Invalid epochs")
            return
        self.training = True
        self.current_epoch = 0
        self.train_start_time = time.time()
        thread = threading.Thread(target=self.train_loop, args=(epochs,), daemon=True)
        thread.start()
        self.log_train(f"Training started for {epochs} epochs.")

    def train_loop(self, epochs):
        try:
            img_size = self.settings['img_size'].get()
            color_mode = self.settings['color_mode'].get()
            batch_size = self.settings['batch_size'].get()
            lr = self.settings['lr'].get()
            preview_enabled = self.settings['preview_enabled'].get()
            preview_freq = self.settings['preview_epoch_freq'].get()
            aug_dict = {k: v.get() for k, v in self.aug_settings.items()}
            canny_low = self.settings['canny_low'].get()
            canny_high = self.settings['canny_high'].get()
            ssim_w = self.settings['ssim_weight'].get()
            l1_w = self.settings['l1_weight'].get()
            l2_w = self.settings['l2_weight'].get()

            dataset = Pix2PixDataset(self.image_paths, img_size, color_mode, aug_dict,
                                     canny_low=canny_low, canny_high=canny_high)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)

            for epoch in range(epochs):
                if not self.training:
                    break
                self.current_epoch = epoch

                new_lr = self.settings['lr'].get()
                if new_lr != lr:
                    for pg in self.trainer.optimizer.param_groups:
                        pg['lr'] = new_lr
                    lr = new_lr
                    self.log_train(f"LR updated to {lr}")

                new_ssim = self.settings['ssim_weight'].get()
                new_l1 = self.settings['l1_weight'].get()
                new_l2 = self.settings['l2_weight'].get()
                if (new_ssim != ssim_w) or (new_l1 != l1_w) or (new_l2 != l2_w):
                    ssim_w, l1_w, l2_w = new_ssim, new_l1, new_l2
                    self.log_train(f"Loss weights updated: SSIM={ssim_w}, L1={l1_w}, L2={l2_w}")

                epoch_loss = 0.0
                epoch_l1 = 0.0
                epoch_ssim = 0.0
                epoch_l2 = 0.0
                batches = 0
                for edge, target in loader:
                    if not self.training:
                        break
                    loss, l1, ssim, l2 = self.trainer.train_step(edge, target, ssim_w, l1_w, l2_w)
                    epoch_loss += loss
                    epoch_l1 += l1
                    epoch_ssim += ssim
                    epoch_l2 += l2
                    batches += 1
                avg_loss = epoch_loss / batches if batches else 0
                avg_l1 = epoch_l1 / batches if batches else 0
                avg_ssim = epoch_ssim / batches if batches else 0
                avg_l2 = epoch_l2 / batches if batches else 0
                elapsed = time.time() - self.train_start_time
                self.log_train(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | L1: {avg_l1:.4f} | SSIM: {avg_ssim:.4f} | L2: {avg_l2:.4f} | Time: {elapsed:.1f}s")

                if preview_enabled and (epoch+1) % preview_freq == 0:
                    self.show_train_preview(loader)

            self.training = False
            self.log_train("Training finished.")
        except Exception as e:
            self.log_train(f"Training error: {e}")
            import traceback
            traceback.print_exc()
            self.training = False

    def show_train_preview(self, loader):
        try:
            self.trainer.model.eval()
            with torch.no_grad():
                edge, target = next(iter(loader))
                n = min(8, len(edge))
                edge = edge[:n].to(self.trainer.device)
                target = target[:n].to(self.trainer.device)
                output = self.trainer.generate(edge)
            self.trainer.model.train()

            thumb = 64
            cols = 4
            rows = 4
            total_width = cols * thumb
            total_height = rows * thumb
            grid = Image.new('RGB', (total_width, total_height), color=(128,128,128))

            for i in range(min(n, 8)):
                row = i // cols
                col = i % cols
                e_img = (edge[i, 0:1].cpu() + 1) / 2
                e_img = e_img.clamp(0,1)
                e_pil = transforms.ToPILImage()(e_img)
                e_pil = e_pil.resize((thumb, thumb), Image.NEAREST)
                grid.paste(e_pil, (col*thumb, row*thumb))

            for i in range(min(n, 8)):
                row = 2 + i // cols
                col = i % cols
                o_img = (output[i].cpu() + 1) / 2
                o_img = o_img.clamp(0,1)
                if o_img.shape[0] == 1:
                    o_pil = transforms.ToPILImage()(o_img.repeat(3,1,1))
                else:
                    o_pil = transforms.ToPILImage()(o_img)
                o_pil = o_pil.resize((thumb, thumb), Image.NEAREST)
                grid.paste(o_pil, (col*thumb, row*thumb))

            display_width = 512
            display_height = int(display_width * (rows/cols))
            grid = grid.resize((display_width, display_height), Image.NEAREST)
            self.train_preview_photo = ImageTk.PhotoImage(grid)
            self.train_preview_canvas.delete("all")
            self.train_preview_canvas.config(width=display_width, height=display_height)
            self.train_preview_canvas.create_image(display_width//2, display_height//2, image=self.train_preview_photo)
            self.train_preview_canvas.image = self.train_preview_photo
        except Exception as e:
            self.log_train(f"Preview error: {e}")

    def stop_training(self):
        self.training = False
        self.log_train("Training stopped.")

    def save_model(self):
        if self.trainer is None:
            self.log_train("No model.")
            return
        fname = filedialog.asksaveasfilename(defaultextension=".pth", filetypes=[("PyTorch","*.pth")])
        if fname:
            in_channels = 3 if self.settings['color_mode'].get() == 'rgb' else 1
            self.trainer.save(fname, in_channels)
            self.log_train(f"Model saved to {fname}")

    def load_model(self):
        fname = filedialog.askopenfilename(filetypes=[("PyTorch","*.pth")])
        if not fname:
            return
        try:
            if self.trainer is None:
                tmp = torch.load(fname, map_location='cpu')
                in_channels = tmp.get('in_channels', 3)
                if in_channels == 3:
                    self.settings['color_mode'].set('rgb')
                else:
                    self.settings['color_mode'].set('grayscale')
                self.init_model()
            loaded_in = self.trainer.load(fname, map_location='cpu')
            self.trainer.model.to(self.trainer.device)
            if loaded_in != (3 if self.settings['color_mode'].get() == 'rgb' else 1):
                self.log_train(f"Warning: loaded model has in_channels={loaded_in}, but current color_mode is {self.settings['color_mode'].get()}. Consider re-initializing.")
            self.log_train(f"Model loaded from {fname}")
        except Exception as e:
            self.log_train(f"Load error: {e}")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    root = tk.Tk()
    app = Pix2PixApp(root)
    root.mainloop()