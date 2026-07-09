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
import traceback

# ========================= Helper functions =========================
def load_image_as_rgb(path):
    img = Image.open(path)
    if img.mode == 'RGBA':
        bg = Image.new('RGB', img.size, (0, 0, 0))
        bg.paste(img, mask=img.split()[3])
        return bg
    return img.convert('RGB')

def load_image_as_grayscale(path):
    img = Image.open(path)
    if img.mode == 'RGBA':
        bg = Image.new('L', img.size, 0)
        bg.paste(img.convert('L'), mask=img.split()[3])
        return bg
    return img.convert('L')

def text_to_indices(text, max_len=128):
    indices = [ord(ch) if ord(ch) < 256 else 0 for ch in text[:max_len]]
    if len(indices) < max_len:
        indices += [0] * (max_len - len(indices))
    return indices

# ========================= EMA =========================
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
                self.shadow[name] = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name])
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name])

# ========================= Text Encoders =========================
class BidirectionalGRUEncoder(nn.Module):
    def __init__(self, vocab_size=256, embed_dim=64, hidden_size=64, num_layers=2, cond_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=0.1 if num_layers>1 else 0)
        self.fc = nn.Linear(hidden_size * 2, cond_dim)
    def forward(self, x):
        emb = self.embedding(x)
        _, h = self.gru(emb)
        h_fwd = h[-2, :, :]
        h_bwd = h[-1, :, :]
        return self.fc(torch.cat([h_fwd, h_bwd], dim=1))
    def get_word_features(self, x):
        emb = self.embedding(x)
        out, _ = self.gru(emb)
        return out

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size=256, embed_dim=128, num_heads=4, num_layers=3, ff_dim=256, cond_dim=512, max_len=128, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, embed_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, cond_dim)
    def forward(self, x):
        B, T = x.shape
        emb = self.embedding(x) + self.pos_embedding[:, :T, :]
        out = self.transformer(emb)
        return self.fc(out.mean(dim=1))
    def get_word_features(self, x):
        B, T = x.shape
        emb = self.embedding(x) + self.pos_embedding[:, :T, :]
        return self.transformer(emb)

# ========================= Spatial Attention (stabilized) =========================
class SpatialAttention:
    def __init__(self, N, A, B):
        self.N, self.A, self.B = N, A, B
        self.patch_y = torch.arange(N).float()
        self.patch_x = torch.arange(N).float()
    def parameters_from_hidden(self, h, device):
        B = h.size(0)
        params = torch.tanh(h[:, :5])
        gY_raw, gX_raw, log_sigma2, log_delta, log_gamma = params.split(1, dim=1)
        gY = (self.A - 1) * (gY_raw + 1) / 2
        gX = (self.B - 1) * (gX_raw + 1) / 2
        sigma2 = torch.exp(torch.clamp(log_sigma2, min=-10, max=10))
        sigma2 = torch.clamp(sigma2, min=1e-6, max=1e4)
        delta = (max(self.A, self.B) - 1) / (self.N - 1) * torch.exp(torch.clamp(log_delta, min=-10, max=10))
        delta = torch.clamp(delta, min=1e-6, max=100.0)
        gamma = torch.exp(torch.clamp(log_gamma, min=-2, max=2))
        gamma = torch.clamp(gamma, min=0.1, max=10.0)
        return gX.squeeze(1), gY.squeeze(1), sigma2.squeeze(1), delta.squeeze(1), gamma.squeeze(1)
    def filterbank(self, gX, gY, sigma2, delta, device):
        i = self.patch_x.to(device)
        mu_X = gX.unsqueeze(1) + (i - self.N/2 - 0.5) * delta.unsqueeze(1)
        j = self.patch_y.to(device)
        mu_Y = gY.unsqueeze(1) + (j - self.N/2 - 0.5) * delta.unsqueeze(1)
        a = torch.arange(self.A, device=device).float()
        diff_X = a.unsqueeze(0).unsqueeze(0) - mu_X.unsqueeze(2)
        F_X = torch.exp(-diff_X**2 / (2 * sigma2.unsqueeze(1).unsqueeze(2)))
        F_X = F_X / (F_X.sum(dim=2, keepdim=True) + 1e-8)
        b = torch.arange(self.B, device=device).float()
        diff_Y = b.unsqueeze(0).unsqueeze(0) - mu_Y.unsqueeze(2)
        F_Y = torch.exp(-diff_Y**2 / (2 * sigma2.unsqueeze(1).unsqueeze(2)))
        F_Y = F_Y / (F_Y.sum(dim=2, keepdim=True) + 1e-8)
        return F_X, F_Y
    def read(self, image, error_image, h_dec, device):
        B, C, _, _ = image.shape
        gX, gY, sigma2, delta, gamma = self.parameters_from_hidden(h_dec, device)
        F_X, F_Y = self.filterbank(gX, gY, sigma2, delta, device)
        patch_img = torch.matmul(F_Y.unsqueeze(1), image)
        patch_img = torch.matmul(patch_img, F_X.unsqueeze(1).transpose(-1,-2)).squeeze(1)
        patch_err = torch.matmul(F_Y.unsqueeze(1), error_image)
        patch_err = torch.matmul(patch_err, F_X.unsqueeze(1).transpose(-1,-2)).squeeze(1)
        patch_img = gamma.view(B,1,1,1) * patch_img
        patch_err = gamma.view(B,1,1,1) * patch_err
        return torch.cat([patch_img, patch_err], dim=1)
    def write(self, patch, h_dec, device):
        B, C, _, _ = patch.shape
        gX, gY, sigma2, delta, gamma = self.parameters_from_hidden(h_dec, device)
        F_X, F_Y = self.filterbank(gX, gY, sigma2, delta, device)
        inv_gamma = 1.0 / gamma
        patch_scaled = inv_gamma.view(B,1,1,1) * patch
        delta_canvas = torch.matmul(F_Y.transpose(1,2).unsqueeze(1), patch_scaled)
        delta_canvas = torch.matmul(delta_canvas, F_X.unsqueeze(1)).squeeze(1)
        return delta_canvas

# ========================= Transformer Autoregressive Decoder =========================
class TransformerImageDecoder(nn.Module):
    def __init__(self, hidden_dim=256, latent_dim=64, context_dim=256, num_layers=4, num_heads=8, ff_dim=1024):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(latent_dim + context_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, 1000, hidden_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, z, context, past_states=None):
        B = z.size(0)
        inp = torch.cat([z, context], dim=1)
        inp = self.input_proj(inp).unsqueeze(1)

        if past_states is None:
            tgt = inp
        else:
            past = torch.stack(past_states, dim=1)
            tgt = torch.cat([past, inp], dim=1)

        seq_len = tgt.size(1)
        tgt = tgt + self.pos_embed[:, :seq_len, :]

        mask = torch.triu(torch.ones(seq_len, seq_len, device=z.device) * float('-inf'), diagonal=1)
        out = self.transformer(tgt, mask=mask)
        return out[:, -1, :]

# ========================= AlignDRAW Model =========================
class AlignDRAW(nn.Module):
    def __init__(self, in_channels=3, img_size=32, text_encoder=None, latent_dim=64, hidden_dim=256, num_steps=16, patch_size=12, decoder_type='LSTM', device='cpu'):
        super().__init__()
        self.in_channels = in_channels
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.text_encoder = text_encoder
        self.use_conditioning = text_encoder is not None
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps
        self.patch_size = patch_size
        self.decoder_type = decoder_type
        self.device = device

        if self.use_conditioning:
            with torch.no_grad():
                dummy = torch.zeros(1, 1, dtype=torch.long, device=device)
                self.word_feat_dim = text_encoder.get_word_features(dummy).size(-1)
        else:
            self.word_feat_dim = hidden_dim

        # Compute encoder input size correctly
        self.encoder_input_size = 2 * in_channels * patch_size * patch_size + hidden_dim
        if decoder_type == 'LSTM':
            self.decoder_cell = nn.LSTMCell(input_size=latent_dim + hidden_dim, hidden_size=hidden_dim)
        else:
            self.decoder_transformer = TransformerImageDecoder(hidden_dim, latent_dim, hidden_dim)

        self.write_patch_fc = nn.Linear(hidden_dim, patch_size * patch_size * in_channels)
        self.prior_mu_fc = nn.Linear(hidden_dim, latent_dim)
        self.prior_logvar_fc = nn.Linear(hidden_dim, latent_dim)
        if self.use_conditioning:
            self.align_attn = nn.Linear(hidden_dim + self.word_feat_dim, 1)
        else:
            self.align_attn = None

        self.attention = SpatialAttention(patch_size, self.img_size[0], self.img_size[1])
        self.encoder_lstm = nn.LSTMCell(input_size=self.encoder_input_size, hidden_size=hidden_dim)
        self.posterior_mu_fc = nn.Linear(hidden_dim, latent_dim)
        self.posterior_logvar_fc = nn.Linear(hidden_dim, latent_dim)

        self.init_h = nn.Parameter(torch.randn(1, hidden_dim))
        self.init_c = nn.Parameter(torch.randn(1, hidden_dim))
        self.canvas_bias = nn.Parameter(torch.zeros(1, in_channels, self.img_size[0], self.img_size[1]))
        self.to(device)

    def reparameterize(self, mu, logvar):
        logvar = torch.clamp(logvar, min=-10, max=10)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, images, text_tensors, teacher_forcing=True):
        B = images.size(0)
        device = images.device
        images = torch.clamp(images, 0.0, 1.0)

        if self.use_conditioning and text_tensors is not None:
            word_feats = self.text_encoder.get_word_features(text_tensors)
            T = word_feats.size(1)
        else:
            word_feats = torch.zeros(B, 1, self.word_feat_dim, device=device)
            T = 1

        if self.decoder_type == 'LSTM':
            h_dec = self.init_h.expand(B, -1)
            c_dec = self.init_c.expand(B, -1)
        else:
            h_dec = self.init_h.expand(B, -1)
            past_states = []

        h_enc = self.init_h.expand(B, -1)
        c_enc = self.init_c.expand(B, -1)
        canvas = self.canvas_bias.expand(B, -1, -1, -1).clone()
        kl_sum = 0.0

        for step in range(self.num_steps):
            if self.use_conditioning and text_tensors is not None:
                h_dec_exp = h_dec.unsqueeze(1).expand(-1, T, -1)
                attn_input = torch.cat([h_dec_exp, word_feats], dim=2)
                scores = self.align_attn(attn_input).squeeze(2)
                alpha = F.softmax(scores, dim=1)
                context = (alpha.unsqueeze(2) * word_feats).sum(dim=1)
            else:
                context = torch.zeros(B, self.hidden_dim, device=device)

            prior_mu = self.prior_mu_fc(h_dec)
            prior_logvar = self.prior_logvar_fc(h_dec)

            if teacher_forcing:
                error = images - torch.sigmoid(canvas)
                read_patch = self.attention.read(images, error, h_dec, device)
                read_patch_flat = read_patch.view(B, -1)
                # Verify dimension
                expected = 2 * self.in_channels * self.patch_size * self.patch_size
                if read_patch_flat.size(1) != expected:
                    raise RuntimeError(f"read_patch_flat size {read_patch_flat.size(1)} != expected {expected}")
                enc_input = torch.cat([read_patch_flat, h_dec], dim=1)
                if enc_input.size(1) != self.encoder_input_size:
                    raise RuntimeError(f"enc_input size {enc_input.size(1)} != encoder_input_size {self.encoder_input_size}")
                h_enc, c_enc = self.encoder_lstm(enc_input, (h_enc, c_enc))
                post_mu = self.posterior_mu_fc(h_enc)
                post_logvar = self.posterior_logvar_fc(h_enc)
                z = self.reparameterize(post_mu, post_logvar)
                kl = -0.5 * torch.sum(1 + post_logvar - prior_logvar - (post_mu - prior_mu).pow(2) / prior_logvar.exp() - (post_logvar.exp())/prior_logvar.exp(), dim=1)
                kl_sum += kl.mean()
            else:
                z = self.reparameterize(prior_mu, prior_logvar)

            dec_input = torch.cat([z, context], dim=1)
            if self.decoder_type == 'LSTM':
                h_dec, c_dec = self.decoder_cell(dec_input, (h_dec, c_dec))
            else:
                h_dec = self.decoder_transformer(z, context, past_states if step>0 else None)
                past_states.append(h_dec)

            patch_flat = self.write_patch_fc(h_dec)
            patch = patch_flat.view(B, self.in_channels, self.patch_size, self.patch_size)
            canvas = canvas + self.attention.write(patch, h_dec, device)

        recon = torch.sigmoid(canvas)
        eps = 1e-7
        recon = torch.clamp(recon, eps, 1.0 - eps)
        recon_loss = F.binary_cross_entropy(recon, images, reduction='sum') / B
        total_loss = recon_loss + kl_sum
        return total_loss, recon, kl_sum

    @torch.no_grad()
    def generate_stepwise(self, text_tensors, num_steps=None):
        if num_steps is None:
            num_steps = self.num_steps
        B = text_tensors.size(0)
        device = text_tensors.device

        if self.use_conditioning and text_tensors is not None:
            word_feats = self.text_encoder.get_word_features(text_tensors)
            T = word_feats.size(1)
        else:
            word_feats = torch.zeros(B, 1, self.word_feat_dim, device=device)
            T = 1

        if self.decoder_type == 'LSTM':
            h_dec = self.init_h.expand(B, -1)
            c_dec = self.init_c.expand(B, -1)
        else:
            h_dec = self.init_h.expand(B, -1)
            past_states = []

        canvas = self.canvas_bias.expand(B, -1, -1, -1).clone()

        for step in range(num_steps):
            if self.use_conditioning and text_tensors is not None:
                h_dec_exp = h_dec.unsqueeze(1).expand(-1, T, -1)
                attn_input = torch.cat([h_dec_exp, word_feats], dim=2)
                scores = self.align_attn(attn_input).squeeze(2)
                alpha = F.softmax(scores, dim=1)
                context = (alpha.unsqueeze(2) * word_feats).sum(dim=1)
            else:
                context = torch.zeros(B, self.hidden_dim, device=device)

            prior_mu = self.prior_mu_fc(h_dec)
            prior_logvar = self.prior_logvar_fc(h_dec)
            z = self.reparameterize(prior_mu, prior_logvar)

            dec_input = torch.cat([z, context], dim=1)
            if self.decoder_type == 'LSTM':
                h_dec, c_dec = self.decoder_cell(dec_input, (h_dec, c_dec))
            else:
                h_dec = self.decoder_transformer(z, context, past_states if step>0 else None)
                past_states.append(h_dec)

            patch_flat = self.write_patch_fc(h_dec)
            patch = patch_flat.view(B, self.in_channels, self.patch_size, self.patch_size)
            canvas = canvas + self.attention.write(patch, h_dec, device)

            yield step+1, torch.sigmoid(canvas)

    @torch.no_grad()
    def generate(self, text_tensors, num_steps=None):
        for _, canvas in self.generate_stepwise(text_tensors, num_steps):
            pass
        return canvas

# ========================= Refinement UNet (optional) =========================
class RefinementUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, cond_dim=256, base_channels=64):
        super().__init__()
        self.cond_proj = nn.Sequential(nn.Linear(cond_dim, base_channels), nn.ReLU(), nn.Linear(base_channels, base_channels))
        self.enc1 = self.conv_block(in_channels, base_channels)
        self.enc2 = self.conv_block(base_channels, base_channels*2)
        self.enc3 = self.conv_block(base_channels*2, base_channels*4)
        self.enc4 = self.conv_block(base_channels*4, base_channels*8)
        self.up4 = self.up_block(base_channels*8, base_channels*4)
        self.up3 = self.up_block(base_channels*4, base_channels*2)
        self.up2 = self.up_block(base_channels*2, base_channels)
        self.up1 = self.up_block(base_channels, base_channels)
        self.final = nn.Sequential(nn.Conv2d(base_channels, out_channels, 3, padding=1), nn.Sigmoid())
    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True))
    def up_block(self, in_c, out_c):
        return nn.Sequential(nn.ConvTranspose2d(in_c, out_c, 2, stride=2), self.conv_block(out_c*2, out_c))
    def forward(self, x, cond):
        B, C, H, W = x.shape
        cond_emb = self.cond_proj(cond).view(B, -1, 1, 1)
        e1 = self.enc1(x) + cond_emb
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        d4 = self.up4(e4)
        d4 = torch.cat([d4, e3], dim=1)
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e2], dim=1)
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e1], dim=1)
        d1 = self.up1(d2)
        return self.final(d1)

# ========================= Dataset =========================
class ConditionalImageDataset(Dataset):
    def __init__(self, image_paths, labels_per_image, img_size=32, color_mode='rgb', aug_settings=None, text_max_len=128):
        self.image_paths = image_paths
        self.labels = []
        for lbl in labels_per_image:
            if isinstance(lbl, str):
                self.labels.append([lbl])
            else:
                self.labels.append(lbl if isinstance(lbl, list) else [''])
        self.img_size = img_size
        self.color_mode = color_mode.lower()
        self.aug_settings = aug_settings or {}
        self.text_max_len = text_max_len
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])
    def __len__(self):
        return len(self.image_paths)
    def apply_augmentations(self, pil_img):
        img = pil_img.copy()
        if self.aug_settings.get('flip_horizontal', False) and random.random() < 0.5:
            img = F_vision.hflip(img)
        if self.aug_settings.get('rotation', False) and random.random() < 0.5:
            img = F_vision.rotate(img, random.uniform(-30, 30), interpolation=Image.BICUBIC)
        return img
    def __getitem__(self, idx):
        try:
            if self.color_mode == 'rgb':
                pil_img = load_image_as_rgb(self.image_paths[idx])
            else:
                pil_img = load_image_as_grayscale(self.image_paths[idx])
            pil_img = self.apply_augmentations(pil_img)
            img_tensor = self.transform(pil_img)
            img_tensor = torch.clamp(img_tensor, 0.0, 1.0)
            captions = self.labels[idx]
            chosen_text = random.choice(captions).replace('_', ' ')
            text_indices = text_to_indices(chosen_text, self.text_max_len)
            text_tensor = torch.tensor(text_indices, dtype=torch.long)
            return img_tensor, text_tensor
        except Exception as e:
            print(f"Error loading {self.image_paths[idx]}: {e}")
            channels = 3 if self.color_mode == 'rgb' else 1
            dummy_img = torch.zeros(channels, self.img_size, self.img_size)
            dummy_txt = torch.zeros(self.text_max_len, dtype=torch.long)
            return dummy_img, dummy_txt

# ========================= GUI Application =========================
class AlignDRAWApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AlignDRAW - Text-to-Image Generation")
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        win_width = max(1000, int(screen_width * 0.8))
        win_height = max(700, int(screen_height * 0.8))
        self.root.geometry(f"{win_width}x{win_height}")
        self.root.minsize(900, 650)

        self.image_paths = []
        self.labels = []
        self.training = False
        self.model = None
        self.text_encoder = None
        self.refinement_unet = None
        self.message_queue_dataset = queue.Queue()
        self.message_queue_training = queue.Queue()
        self.progressive_stop = False

        self.settings = {
            'img_size': tk.IntVar(value=32), 'color_mode': tk.StringVar(value='rgb'),
            'cond_enabled': tk.BooleanVar(value=False),
            'text_encoder_type': tk.StringVar(value='BiGRU'), 'text_encoder_size': tk.StringVar(value='small'),
            'cond_embed_dim': tk.IntVar(value=64), 'cond_hidden_size': tk.IntVar(value=64),
            'cond_num_layers': tk.IntVar(value=2), 'cond_num_heads': tk.IntVar(value=4),
            'cond_ff_dim': tk.IntVar(value=256), 'cond_dim': tk.IntVar(value=256),
            'cond_text_max_len': tk.IntVar(value=128), 'latent_dim': tk.IntVar(value=64),
            'hidden_dim': tk.IntVar(value=256), 'num_steps': tk.IntVar(value=8),
            'patch_size': tk.IntVar(value=8), 'decoder_type': tk.StringVar(value='LSTM'),
            'batch_size': tk.IntVar(value=16), 'lr': tk.DoubleVar(value=1e-4),
            'use_refinement': tk.BooleanVar(value=False), 'refinement_lr': tk.DoubleVar(value=1e-4),
            'ema_enabled': tk.BooleanVar(value=False), 'ema_decay': tk.DoubleVar(value=0.999),
            'preview_enabled': tk.BooleanVar(value=True), 'preview_epoch_freq': tk.IntVar(value=5),
            'grad_clip': tk.DoubleVar(value=1.0),
        }
        self.aug_settings = {'flip_horizontal': tk.BooleanVar(value=True), 'rotation': tk.BooleanVar(value=False)}
        self.gen_count = tk.IntVar(value=16)
        self.progressive_check = tk.BooleanVar(value=False)
        self.prog_interval = tk.IntVar(value=5)

        self.setup_gui()
        self.root.after(100, self.process_messages)

    def setup_gui(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        # Dataset tab
        self.dataset_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.dataset_tab, text='Dataset')
        self.setup_dataset_tab()
        # Training tab
        self.train_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.train_tab, text='AlignDRAW Training')
        self.setup_train_tab()
        # Settings tab
        self.settings_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.settings_tab, text='Settings')
        self.setup_settings_tab()
        # Generation tab
        self.gen_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.gen_tab, text='Generation')
        self.setup_generation_tab()
        # Status
        self.status_label = tk.Label(self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    def setup_dataset_tab(self):
        frame = tk.Frame(self.dataset_tab)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        tk.Label(frame, text="Dataset Management", font=("Arial",12,"bold")).pack(pady=(0,10))
        tk.Button(frame, text="Add Images", command=self.add_images).pack(pady=2)
        tk.Button(frame, text="Add Folder (recursive)", command=self.add_folder).pack(pady=2)
        tk.Button(frame, text="Clear All", command=self.clear_images).pack(pady=2)
        self.image_listbox = tk.Listbox(frame, height=10)
        self.image_listbox.pack(fill=tk.X, pady=2)
        tk.Button(frame, text="Load CSV (image,label)", command=self.load_csv).pack(pady=2)
        tk.Button(frame, text="Use filenames as labels", command=self.use_filenames_as_labels).pack(pady=2)
        tk.Button(frame, text="Use folder names as labels", command=self.use_folders_as_labels).pack(pady=2)
        self.csv_status = tk.Label(frame, text="No CSV loaded", fg="red")
        self.csv_status.pack()
        self.log_text = tk.Text(frame, height=15, font=("Courier",9))
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def setup_train_tab(self):
        frame = tk.Frame(self.train_tab)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        left = tk.Frame(frame, width=300)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0,10))
        left.pack_propagate(False)
        tk.Label(left, text="AlignDRAW Training", font=("Arial",12,"bold")).pack(pady=(0,10))
        self.cond_cb = tk.Checkbutton(left, text="Enable text conditioning", variable=self.settings['cond_enabled'])
        self.cond_cb.pack(anchor='w')
        tk.Button(left, text="Initialize Model", command=self.init_model, width=20).pack(pady=5)
        epoch_frame = tk.Frame(left)
        epoch_frame.pack(pady=5)
        tk.Label(epoch_frame, text="Epochs:").pack(side=tk.LEFT)
        self.epoch_var = tk.StringVar(value="100")
        tk.Entry(epoch_frame, textvariable=self.epoch_var, width=8).pack(side=tk.LEFT, padx=5)
        tk.Button(left, text="Start Training", command=self.start_training, bg="lightgreen").pack(pady=5)
        tk.Button(left, text="Stop Training", command=self.stop_training, bg="salmon").pack(pady=5)
        tk.Button(left, text="Save Model", command=self.save_model).pack(pady=5)
        tk.Button(left, text="Load Model", command=self.load_model).pack(pady=5)
        right = tk.Frame(frame)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        preview_frame = tk.LabelFrame(right, text="Preview (generated samples)", padx=5, pady=5)
        preview_frame.pack(fill=tk.BOTH, expand=True)
        self.preview_canvas = tk.Canvas(preview_frame, bg='gray', width=256, height=256)
        self.preview_canvas.pack()
        self.train_log = tk.Text(right, height=15, font=("Courier",9))
        self.train_log.pack(fill=tk.BOTH, expand=True, pady=(5,0))

    def setup_settings_tab(self):
        frame = tk.Frame(self.settings_tab)
        frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        tk.Label(frame, text="Model & Training Settings", font=("Arial",14,"bold")).pack(pady=(0,20))
        canvas = tk.Canvas(frame)
        scrollbar = tk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable = tk.Frame(canvas)
        scrollable.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0,0), window=scrollable, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        row = 0
        for key, var in self.settings.items():
            tk.Label(scrollable, text=key+":", anchor='w').grid(row=row, column=0, sticky='w', pady=2)
            if isinstance(var, tk.BooleanVar):
                cb = tk.Checkbutton(scrollable, variable=var)
                cb.grid(row=row, column=1, sticky='w')
            elif isinstance(var, (tk.IntVar, tk.DoubleVar)):
                entry = tk.Entry(scrollable, textvariable=var, width=10)
                entry.grid(row=row, column=1, sticky='w')
            elif isinstance(var, tk.StringVar):
                if key == 'decoder_type':
                    combo = ttk.Combobox(scrollable, textvariable=var, values=['LSTM','Transformer'], state='readonly')
                    combo.grid(row=row, column=1, sticky='w')
                elif key in ('color_mode', 'text_encoder_type', 'text_encoder_size'):
                    values = {'color_mode': ['rgb','grayscale'], 'text_encoder_type': ['BiGRU','BiTransformer'],
                              'text_encoder_size': ['tiny','small','medium','large']}[key]
                    combo = ttk.Combobox(scrollable, textvariable=var, values=values, state='readonly')
                    combo.grid(row=row, column=1, sticky='w')
                else:
                    entry = tk.Entry(scrollable, textvariable=var, width=10)
                    entry.grid(row=row, column=1, sticky='w')
            row += 1
        tk.Label(scrollable, text="Augmentations", font=("Arial",10,"bold")).grid(row=row, column=0, sticky='w', pady=(10,0))
        row += 1
        for aug, var in self.aug_settings.items():
            cb = tk.Checkbutton(scrollable, text=aug, variable=var)
            cb.grid(row=row, column=0, columnspan=2, sticky='w')
            row += 1

    def setup_generation_tab(self):
        frame = tk.Frame(self.gen_tab)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        tk.Label(frame, text="Generate Images", font=("Arial",14,"bold")).pack(pady=(0,10))
        prompt_frame = tk.LabelFrame(frame, text="Text Prompt (leave empty for unconditional)", padx=5, pady=5)
        prompt_frame.pack(fill=tk.X, pady=(0,10))
        self.gen_prompt = tk.Entry(prompt_frame, width=60)
        self.gen_prompt.insert(0, "a cute cat")
        self.gen_prompt.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ctrl_frame = tk.Frame(frame)
        ctrl_frame.pack(pady=5)
        tk.Label(ctrl_frame, text="Number:").pack(side=tk.LEFT)
        tk.Spinbox(ctrl_frame, from_=1, to=64, textvariable=self.gen_count, width=5).pack(side=tk.LEFT, padx=5)
        self.prog_cb = tk.Checkbutton(ctrl_frame, text="Progressive (real-time)", variable=self.progressive_check)
        self.prog_cb.pack(side=tk.LEFT, padx=10)
        tk.Label(ctrl_frame, text="Update every N steps:").pack(side=tk.LEFT)
        tk.Spinbox(ctrl_frame, from_=1, to=50, textvariable=self.prog_interval, width=5).pack(side=tk.LEFT, padx=5)
        self.generate_btn = tk.Button(ctrl_frame, text="Generate", command=self.generate_samples, bg="lightgreen")
        self.generate_btn.pack(side=tk.LEFT, padx=5)
        self.stop_prog_btn = tk.Button(ctrl_frame, text="Stop", command=self.stop_progressive, state=tk.DISABLED, bg="salmon")
        self.stop_prog_btn.pack(side=tk.LEFT, padx=5)
        self.results_canvas = tk.Canvas(frame, bg='lightgray')
        self.results_canvas.pack(fill=tk.BOTH, expand=True)
        self.inner_frame = tk.Frame(self.results_canvas)
        self.results_canvas.create_window((0,0), window=self.inner_frame, anchor='nw')
        self.inner_frame.bind('<Configure>', lambda e: self.results_canvas.configure(scrollregion=self.results_canvas.bbox('all')))
        self.prog_label = tk.Label(frame, text="", fg="blue")
        self.prog_label.pack()

    # ----- Dataset helpers -----
    def add_images(self):
        files = filedialog.askopenfilenames(filetypes=[("Images", "*.jpg *.jpeg *.png *.jfif *.webp *.bmp")])
        for f in files:
            if f not in self.image_paths:
                self.image_paths.append(f)
                self.image_listbox.insert(tk.END, os.path.basename(f))
        self.log_msg(f"Added {len(files)} images. Total: {len(self.image_paths)}")
    def add_folder(self):
        folder = filedialog.askdirectory()
        if not folder: return
        count = 0
        for root_dir, _, files in os.walk(folder):
            for file in files:
                if file.lower().endswith(('.png','.jpg','.jpeg','.jfif','.webp','.bmp')):
                    full = os.path.join(root_dir, file)
                    if full not in self.image_paths:
                        self.image_paths.append(full)
                        self.image_listbox.insert(tk.END, os.path.basename(full))
                        count += 1
        self.log_msg(f"Added {count} images from folder.")
    def clear_images(self):
        self.image_paths = []
        self.labels = []
        self.image_listbox.delete(0, tk.END)
        self.csv_status.config(text="No CSV loaded", fg="red")
        self.log_msg("Cleared all images")
    def load_csv(self):
        fname = filedialog.askopenfilename(filetypes=[("CSV","*.csv")])
        if not fname: return
        label_map = {}
        try:
            with open(fname, 'r', encoding='utf-8-sig') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row)>=2:
                        label_map[row[0].strip().lower()] = row[1].strip()
        except Exception as e:
            self.log_msg(f"Error reading CSV: {e}")
            return
        self.labels = []
        for path in self.image_paths:
            base = os.path.basename(path).lower()
            if base in label_map:
                self.labels.append([label_map[base]])
            else:
                self.labels.append([''])
        self.csv_status.config(text=f"CSV loaded: matched {sum(1 for l in self.labels if l[0])} images", fg="green")
        self.log_msg("CSV loaded.")
    def use_filenames_as_labels(self):
        self.labels = [[os.path.splitext(os.path.basename(p))[0]] for p in self.image_paths]
        self.csv_status.config(text="Using filenames as labels", fg="blue")
        self.log_msg("Using filenames as labels.")
    def use_folders_as_labels(self):
        self.labels = [[os.path.basename(os.path.dirname(p)) or 'unknown'] for p in self.image_paths]
        self.csv_status.config(text="Using folder names as labels", fg="blue")
        self.log_msg("Using folder names as labels.")

    # ----- Model initialization -----
    def init_model(self):
        try:
            cond_enabled = self.settings['cond_enabled'].get()
            in_channels = 3 if self.settings['color_mode'].get() == 'rgb' else 1
            img_size = self.settings['img_size'].get()
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if cond_enabled:
                enc_type = self.settings['text_encoder_type'].get()
                if enc_type == 'BiGRU':
                    self.text_encoder = BidirectionalGRUEncoder(
                        vocab_size=256,
                        embed_dim=self.settings['cond_embed_dim'].get(),
                        hidden_size=self.settings['cond_hidden_size'].get(),
                        num_layers=self.settings['cond_num_layers'].get(),
                        cond_dim=self.settings['cond_dim'].get()
                    )
                else:
                    self.text_encoder = TransformerEncoder(
                        vocab_size=256,
                        embed_dim=self.settings['cond_embed_dim'].get(),
                        num_heads=self.settings['cond_num_heads'].get(),
                        num_layers=self.settings['cond_num_layers'].get(),
                        ff_dim=self.settings['cond_ff_dim'].get(),
                        cond_dim=self.settings['cond_dim'].get(),
                        max_len=self.settings['cond_text_max_len'].get()
                    )
                self.text_encoder.to(device)
            else:
                self.text_encoder = None
            self.model = AlignDRAW(
                in_channels=in_channels, img_size=img_size, text_encoder=self.text_encoder,
                latent_dim=self.settings['latent_dim'].get(), hidden_dim=self.settings['hidden_dim'].get(),
                num_steps=self.settings['num_steps'].get(), patch_size=self.settings['patch_size'].get(),
                decoder_type=self.settings['decoder_type'].get(), device=device
            )
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.settings['lr'].get())
            if self.settings['ema_enabled'].get():
                self.ema = EMA(self.model, decay=self.settings['ema_decay'].get())
            else:
                self.ema = None
            self.log_train(f"Model initialized on {device} with decoder={self.settings['decoder_type'].get()}, in_channels={in_channels}")
        except Exception as e:
            self.log_train(f"Init error: {e}\n{traceback.format_exc()}")

    # ----- Training -----
    def start_training(self):
        if not self.image_paths:
            self.log_train("No images loaded.")
            return
        if not self.model:
            self.log_train("Model not initialized.")
            return
        if self.training:
            self.log_train("Already training.")
            return
        cond_enabled = self.settings['cond_enabled'].get()
        if cond_enabled and not any(lbl for lbl in self.labels if lbl):
            self.log_train("Conditional training requires labels. Load CSV or use filename/folder labels.")
            return
        try:
            epochs = int(self.epoch_var.get())
        except:
            self.log_train("Invalid epochs")
            return
        self.training = True
        threading.Thread(target=self.train_loop, args=(epochs,), daemon=True).start()
        self.log_train(f"Training started for {epochs} epochs.")

    def train_loop(self, epochs):
        try:
            batch_size = self.settings['batch_size'].get()
            img_size = self.settings['img_size'].get()
            color_mode = self.settings['color_mode'].get()
            cond_enabled = self.settings['cond_enabled'].get()
            text_max_len = self.settings['cond_text_max_len'].get()
            use_ema = self.settings['ema_enabled'].get()
            preview_freq = self.settings['preview_epoch_freq'].get()
            preview_enabled = self.settings['preview_enabled'].get()
            grad_clip = self.settings['grad_clip'].get()

            aug_dict = {k: v.get() for k,v in self.aug_settings.items()}
            labels_for_dataset = self.labels if self.labels else [['']]*len(self.image_paths)
            dataset = ConditionalImageDataset(self.image_paths, labels_for_dataset, img_size, color_mode, aug_dict, text_max_len)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

            for epoch in range(epochs):
                if not self.training:
                    break
                epoch_loss = 0.0
                batches = 0
                for images, text_tensors in loader:
                    if not self.training:
                        break
                    images = images.to(self.model.device)
                    # Verify channel consistency
                    if images.size(1) != self.model.in_channels:
                        raise RuntimeError(f"Image channels {images.size(1)} != model in_channels {self.model.in_channels}")
                    text_tensors = text_tensors.to(self.model.device) if cond_enabled else None
                    self.optimizer.zero_grad()
                    loss, _, _ = self.model(images, text_tensors, teacher_forcing=True)
                    if torch.isnan(loss):
                        self.log_train("NaN loss detected, skipping batch")
                        continue
                    loss.backward()
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                    self.optimizer.step()
                    if use_ema and self.ema:
                        self.ema.update()
                    epoch_loss += loss.item()
                    batches += 1
                if batches == 0:
                    self.log_train(f"Epoch {epoch+1}: no valid batches")
                    continue
                avg_loss = epoch_loss / batches
                self.log_train(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
                if preview_enabled and (epoch+1) % preview_freq == 0:
                    self.preview_generation()
            self.training = False
            self.log_train("Training finished.")
        except Exception as e:
            self.log_train(f"Training error: {e}\n{traceback.format_exc()}")
            self.training = False

    def preview_generation(self):
        if not self.model:
            return
        self.model.eval()
        try:
            device = self.model.device
            cond_enabled = self.settings['cond_enabled'].get()
            if cond_enabled and self.text_encoder and len(self.labels)>0 and self.labels[0]:
                sample_text = self.labels[0][0]
                text_indices = text_to_indices(sample_text, self.settings['cond_text_max_len'].get())
                text_tensor = torch.tensor([text_indices]*4, dtype=torch.long, device=device)
                generated = self.model.generate(text_tensor, num_steps=self.settings['num_steps'].get())
            else:
                text_tensor = torch.zeros(4, 1, dtype=torch.long, device=device)
                generated = self.model.generate(text_tensor, num_steps=self.settings['num_steps'].get())
            images = (generated.cpu().numpy() * 255).astype(np.uint8)
            thumb = 64
            grid = Image.new('RGB', (2*thumb, 2*thumb))
            for i in range(4):
                row, col = divmod(i, 2)
                img = images[i]
                if img.shape[0]==1:
                    img = np.stack([img[0], img[0], img[0]], axis=-1)
                else:
                    img = img.transpose(1,2,0)
                pil = Image.fromarray(img).resize((thumb, thumb), Image.NEAREST)
                grid.paste(pil, (col*thumb, row*thumb))
            photo = ImageTk.PhotoImage(grid.resize((256,256), Image.NEAREST))
            self.preview_canvas.delete("all")
            self.preview_canvas.create_image(128,128, image=photo)
            self.preview_canvas.image = photo
        except Exception as e:
            self.log_train(f"Preview error: {e}\n{traceback.format_exc()}")
        self.model.train()

    def stop_training(self):
        self.training = False
        self.log_train("Training stopped.")
    def save_model(self):
        if not self.model:
            self.log_train("No model to save.")
            return
        fname = filedialog.asksaveasfilename(defaultextension=".pth", filetypes=[("PyTorch","*.pth")])
        if fname:
            save_dict = {'model_state': self.model.state_dict(), 'optimizer_state': self.optimizer.state_dict(),
                         'settings': {k:v.get() for k,v in self.settings.items()},
                         'ema_shadow': self.ema.shadow if self.ema else None}
            if self.text_encoder:
                save_dict['text_encoder_state'] = self.text_encoder.state_dict()
            if self.refinement_unet:
                save_dict['refinement_state'] = self.refinement_unet.state_dict()
            torch.save(save_dict, fname)
            self.log_train(f"Model saved to {fname}")
    def load_model(self):
        fname = filedialog.askopenfilename(filetypes=[("PyTorch","*.pth")])
        if not fname: return
        try:
            ckpt = torch.load(fname, map_location='cpu')
            if 'settings' in ckpt:
                for k,v in ckpt['settings'].items():
                    if k in self.settings:
                        self.settings[k].set(v)
            self.init_model()
            self.model.load_state_dict(ckpt['model_state'])
            self.optimizer.load_state_dict(ckpt['optimizer_state'])
            if self.text_encoder and 'text_encoder_state' in ckpt:
                self.text_encoder.load_state_dict(ckpt['text_encoder_state'])
            if self.refinement_unet and 'refinement_state' in ckpt:
                self.refinement_unet.load_state_dict(ckpt['refinement_state'])
            if self.ema and 'ema_shadow' in ckpt:
                self.ema.shadow = ckpt['ema_shadow']
            self.log_train(f"Loaded model from {fname}")
        except Exception as e:
            self.log_train(f"Load error: {e}\n{traceback.format_exc()}")

    # ----- Generation -----
    def generate_samples(self):
        if not self.model:
            self.log_generation("Model not loaded.")
            return
        n = self.gen_count.get()
        prompt = self.gen_prompt.get().strip()
        cond_enabled = self.settings['cond_enabled'].get()
        progressive = self.progressive_check.get()
        self.generate_btn.config(state=tk.DISABLED)
        if progressive:
            self.stop_prog_btn.config(state=tk.NORMAL)
            self.progressive_stop = False
            threading.Thread(target=self._progressive_generate_thread, args=(n, prompt, cond_enabled), daemon=True).start()
        else:
            threading.Thread(target=self._generate_thread, args=(n, prompt, cond_enabled), daemon=True).start()

    def _generate_thread(self, n, prompt, cond_enabled):
        try:
            device = self.model.device
            if cond_enabled and self.text_encoder and prompt:
                text_indices = text_to_indices(prompt, self.settings['cond_text_max_len'].get())
                text_tensor = torch.tensor([text_indices]*n, dtype=torch.long, device=device)
                generated = self.model.generate(text_tensor, num_steps=self.settings['num_steps'].get())
            else:
                text_tensor = torch.zeros(n, 1, dtype=torch.long, device=device)
                generated = self.model.generate(text_tensor, num_steps=self.settings['num_steps'].get())
            images = (generated.cpu().numpy() * 255).astype(np.uint8)
            thumb = 128
            grid_size = int(math.ceil(math.sqrt(n)))
            total = grid_size * thumb
            grid_img = Image.new('RGB', (total, total), color=(128,128,128))
            for i in range(n):
                row, col = divmod(i, grid_size)
                img = images[i]
                if img.shape[0]==1:
                    img = np.stack([img[0], img[0], img[0]], axis=-1)
                else:
                    img = img.transpose(1,2,0)
                pil = Image.fromarray(img).resize((thumb, thumb), Image.NEAREST)
                grid_img.paste(pil, (col*thumb, row*thumb))
            self.root.after(0, lambda: self.display_generated(grid_img, final=True))
        except Exception as e:
            self.root.after(0, lambda: self.log_generation(f"Generation error: {e}\n{traceback.format_exc()}"))
        finally:
            self.root.after(0, lambda: self.generate_btn.config(state=tk.NORMAL))

    def _progressive_generate_thread(self, n, prompt, cond_enabled):
        try:
            device = self.model.device
            if cond_enabled and self.text_encoder and prompt:
                text_indices = text_to_indices(prompt, self.settings['cond_text_max_len'].get())
                text_tensor = torch.tensor([text_indices]*n, dtype=torch.long, device=device)
                generator = self.model.generate_stepwise(text_tensor, num_steps=self.settings['num_steps'].get())
            else:
                text_tensor = torch.zeros(n, 1, dtype=torch.long, device=device)
                generator = self.model.generate_stepwise(text_tensor, num_steps=self.settings['num_steps'].get())
            thumb = 128
            grid_size = int(math.ceil(math.sqrt(n)))
            total = grid_size * thumb
            interval = self.prog_interval.get()
            for step, canvas in generator:
                if self.progressive_stop:
                    break
                if step % interval == 0 or step == self.settings['num_steps'].get():
                    images = (canvas.cpu().numpy() * 255).astype(np.uint8)
                    grid_img = Image.new('RGB', (total, total), color=(128,128,128))
                    for i in range(n):
                        row, col = divmod(i, grid_size)
                        img = images[i]
                        if img.shape[0]==1:
                            img = np.stack([img[0], img[0], img[0]], axis=-1)
                        else:
                            img = img.transpose(1,2,0)
                        pil = Image.fromarray(img).resize((thumb, thumb), Image.NEAREST)
                        grid_img.paste(pil, (col*thumb, row*thumb))
                    self.root.after(0, lambda g=grid_img, s=step: self.display_progressive(g, s))
                    time.sleep(0.05)
            if not self.progressive_stop:
                self.root.after(0, lambda: self.log_generation("Progressive generation finished."))
            else:
                self.root.after(0, lambda: self.log_generation("Stopped."))
        except Exception as e:
            self.root.after(0, lambda: self.log_generation(f"Progressive error: {e}\n{traceback.format_exc()}"))
        finally:
            self.progressive_stop = False
            self.root.after(0, lambda: self.generate_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.stop_prog_btn.config(state=tk.DISABLED))

    def display_progressive(self, grid_img, step):
        for w in self.inner_frame.winfo_children():
            w.destroy()
        photo = ImageTk.PhotoImage(grid_img)
        label = tk.Label(self.inner_frame, image=photo)
        label.image = photo
        label.pack()
        self.results_canvas.configure(scrollregion=self.results_canvas.bbox('all'))
        self.prog_label.config(text=f"Step {step}/{self.settings['num_steps'].get()}")
    def display_generated(self, grid_img, final=False):
        for w in self.inner_frame.winfo_children():
            w.destroy()
        photo = ImageTk.PhotoImage(grid_img)
        label = tk.Label(self.inner_frame, image=photo)
        label.image = photo
        label.pack()
        self.results_canvas.configure(scrollregion=self.results_canvas.bbox('all'))
        if final:
            self.prog_label.config(text="Generation complete.")
        else:
            self.prog_label.config(text="")
    def stop_progressive(self):
        self.progressive_stop = True
        self.log_generation("Stopping progressive generation...")

    # ----- Logging -----
    def log_msg(self, msg):
        self.message_queue_dataset.put(msg)
    def log_train(self, msg):
        self.message_queue_training.put(msg)
    def log_generation(self, msg):
        print(msg)
        self.status_label.config(text=msg[:100])
    def process_messages(self):
        try:
            while True:
                msg = self.message_queue_dataset.get_nowait()
                self.log_text.insert(tk.END, f"{time.strftime('%H:%M:%S')} - {msg}\n")
                self.log_text.see(tk.END)
        except queue.Empty:
            pass
        try:
            while True:
                msg = self.message_queue_training.get_nowait()
                self.train_log.insert(tk.END, f"{time.strftime('%H:%M:%S')} - {msg}\n")
                self.train_log.see(tk.END)
                self.status_label.config(text=msg[:50])
        except queue.Empty:
            pass
        self.root.after(100, self.process_messages)

# ========================= Main =========================
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    root = tk.Tk()
    app = AlignDRAWApp(root)
    root.mainloop()