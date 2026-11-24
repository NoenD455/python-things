"""
Dream - Spatial VAE + Flow Matching World Model (drift-resistant)

World model: rectified flow (noise -> next latent) conditioned on K past
latents + K past actions, trained with context-noise augmentation (noise level
is a model input) and genuine self-rollout training, EMA weights, normalized
latents, and a decoder trained to tolerate slightly-off latents.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageTk
import threading
import queue
import random
import math
import copy

# ========== Import everything from the original file ==========
from world_lstm_2 import (
    GAMES, Maze2D, Maze3D, DVDDodge, SquareJump, LightWorld,
    CarRacingGame, CartPoleGame, AtariGame,
    to_vae_size, ensure_rgb, gym_reset, gym_step, gym_render,
    generate_maze, try_move, astar_path, Raycaster,
    BaseGame, MazeBase,
)

# Normalize GAMES keys (strip whitespace)
GAMES = {k.strip(): v for k, v in GAMES.items()}

# Games where action[1] is the left/right turn axis (safe for mirror aug)
MIRRORABLE_GAMES = ("2D Maze", "3D Maze", "Light World")


# ========== Spatial VAE (outputs C x H x W latents) ==========
class SpatialVAE(nn.Module):
    def __init__(self, in_channels=3, latent_channels=4, spatial_size=8,
                 img_size=64, base_channels=32):
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.spatial_size = spatial_size
        self.img_size = img_size
        self.base_channels = base_channels

        # Number of 2x downsamples needed to go from img_size -> spatial_size
        assert img_size >= spatial_size and (img_size % spatial_size) == 0
        n_down = int(round(math.log2(img_size // spatial_size)))
        assert 2 ** n_down == img_size // spatial_size, \
            "img_size / spatial_size must be a power of 2"
        self.n_down = n_down

        # Encoder
        enc = []
        c_in = in_channels
        c = base_channels
        for i in range(n_down):
            c_out = min(c * (2 ** i), 256)
            enc += [nn.Conv2d(c_in, c_out, 4, 2, 1), nn.SiLU()]
            c_in = c_out
        enc += [nn.Conv2d(c_in, latent_channels * 2, 3, 1, 1)]
        self.encoder = nn.Sequential(*enc)

        # Decoder
        dec = []
        c_in = latent_channels
        for i in range(n_down - 1, -1, -1):
            c_out = min(base_channels * (2 ** i), 256)
            dec += [nn.ConvTranspose2d(c_in, c_out, 4, 2, 1), nn.SiLU()]
            c_in = c_out
        dec += [nn.Conv2d(c_in, in_channels, 3, 1, 1), nn.Sigmoid()]
        self.decoder = nn.Sequential(*dec)

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def get_config(self):
        return {
            'in_channels': self.in_channels,
            'latent_channels': self.latent_channels,
            'spatial_size': self.spatial_size,
            'img_size': self.img_size,
            'base_channels': self.base_channels,
        }


# ========== Flow Matching UNet ==========
class SinusoidalPosEmb(nn.Module):
    """Embeds a scalar in [0, ~1] (flow time or noise level)."""
    def __init__(self, dim, scale=1000.0):
        super().__init__()
        self.dim = dim
        self.scale = scale

    def forward(self, t):
        half = max(self.dim // 2, 1)
        freqs = torch.exp(-math.log(10000.0) *
                          torch.arange(half, device=t.device) / max(half - 1, 1))
        emb = (t.float() * self.scale)[:, None] * freqs[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class FlowResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.cond_proj = nn.Sequential(
            nn.SiLU(), nn.Linear(cond_dim, out_channels * 2)
        )
        self.skip = (nn.Conv2d(in_channels, out_channels, 1)
                     if in_channels != out_channels else nn.Identity())

    def forward(self, x, cond):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.cond_proj(cond)[:, :, None, None].chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift      # FiLM
        h = self.conv2(F.silu(h))
        return h + self.skip(x)


class SelfAttn2d(nn.Module):
    """Global mixing at the bottleneck (a camera turn shifts the whole view,
    which 3x3 convs on a tiny latent grid can't see)."""
    def __init__(self, ch):
        super().__init__()
        self.norm = nn.GroupNorm(min(8, ch), ch)
        self.attn = nn.MultiheadAttention(ch, 4 if ch % 4 == 0 else 1,
                                          batch_first=True)
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x).flatten(2).transpose(1, 2)
        h, _ = self.attn(h, h, h, need_weights=False)
        return x + h.transpose(1, 2).reshape(B, C, H, W)


class FlowUNet(nn.Module):
    """
    Rectified-flow next-frame model in (normalized) VAE latent space.

        x_t = (1-t) * noise + t * z_next,   target v = z_next - noise

    Conditioned on:
      ctx       (B, K, C, H, W)  last K latents (possibly noise-augmented)
      acts      (B, K, A)        action taken at each of those K frames
                                 (acts[:, -1] is the action producing z_next)
      ctx_sigma (B,)             how much noise was added to ctx; the model is
                                 told, so it learns to clean up its own errors
    """
    def __init__(self, latent_channels=4, spatial_size=8, action_dim=2,
                 hidden_dim=64, cond_dim=64, context_frames=4):
        super().__init__()
        self.latent_channels = latent_channels
        self.spatial_size = spatial_size
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim
        self.context_frames = context_frames

        # Per-channel latent statistics (filled in at WM training time and
        # saved with the checkpoint). The WM works in normalized units.
        self.register_buffer('lat_mean', torch.zeros(1, latent_channels, 1, 1))
        self.register_buffer('lat_std', torch.ones(1, latent_channels, 1, 1))

        def scalar_mlp():
            return nn.Sequential(
                SinusoidalPosEmb(cond_dim),
                nn.Linear(cond_dim, cond_dim * 2), nn.SiLU(),
                nn.Linear(cond_dim * 2, cond_dim))
        self.time_mlp = scalar_mlp()
        self.noise_mlp = scalar_mlp()
        self.action_mlp = nn.Sequential(
            nn.Linear(action_dim * context_frames, cond_dim * 2), nn.SiLU(),
            nn.Linear(cond_dim * 2, cond_dim))

        self.history_encoder = nn.Sequential(
            nn.Conv2d(latent_channels * context_frames, hidden_dim, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1))
        self.in_conv = nn.Conv2d(latent_channels + hidden_dim, hidden_dim, 3,
                                 padding=1)

        self.enc1 = FlowResBlock(hidden_dim, hidden_dim, cond_dim)
        self.enc1b = FlowResBlock(hidden_dim, hidden_dim, cond_dim)
        self.down1 = nn.Conv2d(hidden_dim, hidden_dim, 3, 2, 1)
        self.enc2 = FlowResBlock(hidden_dim, hidden_dim, cond_dim)
        self.mid1 = FlowResBlock(hidden_dim, hidden_dim, cond_dim)
        self.attn = SelfAttn2d(hidden_dim)
        self.mid2 = FlowResBlock(hidden_dim, hidden_dim, cond_dim)
        self.up2 = nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, 2, 1)
        self.dec2 = FlowResBlock(hidden_dim * 2, hidden_dim, cond_dim)
        self.up1 = nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, 2, 1)
        self.dec1 = FlowResBlock(hidden_dim * 2, hidden_dim, cond_dim)

        self.out_norm = nn.GroupNorm(min(8, hidden_dim), hidden_dim)
        self.out_conv = nn.Conv2d(hidden_dim, latent_channels, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def normalize(self, z):
        return (z - self.lat_mean) / self.lat_std

    def denormalize(self, z):
        return z * self.lat_std + self.lat_mean

    def forward(self, x_t, t, ctx, acts, ctx_sigma):
        cond = (self.time_mlp(t) + self.noise_mlp(ctx_sigma)
                + self.action_mlp(acts.flatten(1)))

        hist = self.history_encoder(ctx.flatten(1, 2))
        x = self.in_conv(torch.cat([x_t, hist], dim=1))

        e1 = self.enc1b(self.enc1(x, cond), cond)
        e2 = self.enc2(self.down1(e1), cond)
        h = self.mid2(self.attn(self.mid1(e2, cond)), cond)

        u2 = self.up2(h)
        if u2.shape[-2:] != e2.shape[-2:]:
            u2 = F.interpolate(u2, size=e2.shape[-2:], mode='nearest')
        d2 = self.dec2(torch.cat([u2, e2], dim=1), cond)

        u1 = self.up1(d2)
        if u1.shape[-2:] != e1.shape[-2:]:
            u1 = F.interpolate(u1, size=e1.shape[-2:], mode='nearest')
        d1 = self.dec1(torch.cat([u1, e1], dim=1), cond)

        return self.out_conv(F.silu(self.out_norm(d1)))

    def get_config(self):
        return {
            'latent_channels': self.latent_channels,
            'spatial_size': self.spatial_size,
            'action_dim': self.action_dim,
            'hidden_dim': self.hidden_dim,
            'cond_dim': self.cond_dim,
            'context_frames': self.context_frames,
        }


def add_ctx_noise(ctx, sigma):
    """ctx (B,K,C,H,W) + per-sample gaussian noise of std sigma (B,)."""
    return ctx + sigma.view(-1, 1, 1, 1, 1) * torch.randn_like(ctx)


@torch.no_grad()
def flow_sample(wm, ctx, acts, ctx_sigma, n_steps):
    """Euler-integrate noise -> next latent (normalized units)."""
    B, _, C, H, W = ctx.shape
    x = torch.randn(B, C, H, W, device=ctx.device)
    n_steps = max(1, int(n_steps))
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = torch.full((B,), i * dt, device=ctx.device)
        x = x + dt * wm(x, t, ctx, acts, ctx_sigma)
    return x.clamp(-6.0, 6.0)   # keep a bad sample from blowing up the context


# ========== Flow Dataset ==========
class FlowDataset(torch.utils.data.Dataset):
    """
    Returns a window of `window` consecutive normalized latents
    (window, C, H, W) and the `window-1` actions between them, where
    acts[j] is the action taken at latents[j] (producing latents[j+1]).
    The training loop slices context / target out of the window so that it
    can roll the model forward on its own predictions.
    """
    def __init__(self, latents, actions, window,
                 latents_flip=None, actions_flip=None):
        self.latents = latents
        self.actions = actions
        self.window = window
        self.latents_flip = latents_flip
        self.actions_flip = actions_flip
        self.n = max(0, len(latents) - window + 1)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        W = self.window
        if self.latents_flip is not None and random.random() < 0.5:
            return self.latents_flip[i:i + W], self.actions_flip[i:i + W - 1]
        return self.latents[i:i + W], self.actions[i:i + W - 1]


# ========== App ==========
class App:
    def __init__(self, root):
        self.root = root
        root.title("Dream - Spatial VAE + Flow Matching World Model")
        root.geometry("1100x1000")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.img_size = 64
        self.game = GAMES["3D Maze"](self.img_size)
        self.frames = []; self.actions = []
        self.keys = {}
        self.recording = False
        self.last_frame = None

        self.vae = None; self.wm = None
        self.z_ctx = None; self.a_ctx = None
        self.inference_running = False
        self.advance_one_frame = False

        self.vae_stop_event = None; self.wm_stop_event = None
        self.vae_queue = None; self.wm_queue = None

        self.auto_gen_running = False
        self.auto_gen_step = 0; self.auto_gen_target = 0

        self.wm_last_roll = None
        self.record_photo = None; self.inference_photo = None
        self.vae_photo = None; self.wm_photo = None

        self.setup_ui()
        root.bind('<KeyPress>', self.on_key_press)
        root.bind('<KeyRelease>', self.on_key_release)
        self.render_record_view()
        self._update_game_info()

    # ========== UI ==========
    def setup_ui(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True)
        for name, builder in [("Record", self.setup_record_tab),
                              ("Train VAE", self.setup_vae_tab),
                              ("Train World Model", self.setup_wm_tab),
                              ("Settings", self.setup_settings_tab),
                              ("Inference", self.setup_inference_tab)]:
            tab = ttk.Frame(notebook)
            notebook.add(tab, text=name)
            builder(tab)

    def setup_record_tab(self, parent):
        frame = ttk.Frame(parent); frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.record_canvas = tk.Canvas(frame, width=512, height=512, bg='black')
        self.record_canvas.pack(side=tk.LEFT, padx=10)
        panel = ttk.Frame(frame); panel.pack(side=tk.LEFT, fill=tk.Y, padx=10)
        ttk.Label(panel, text="Game:").pack(anchor=tk.W)
        self.game_name_var = tk.StringVar(value=self.game.name)
        self.game_combo = ttk.Combobox(panel, textvariable=self.game_name_var,
                                       values=list(GAMES.keys()), state='readonly')
        self.game_combo.pack(fill=tk.X, pady=(0, 8))
        self.game_combo.bind('<<ComboboxSelected>>', self.on_game_change)
        self.record_status = ttk.Label(panel, text="Frames: 0")
        self.record_status.pack(anchor=tk.W, pady=2)
        self.game_info_label = ttk.Label(panel, text="", foreground="gray")
        self.game_info_label.pack(anchor=tk.W, pady=(0, 6))
        for text, cmd in [
            ("Start Recording", self.start_recording),
            ("Stop Recording", self.stop_recording),
            ("—", None),
            ("Auto-Generate Dataset", self.auto_generate),
            ("Stop Auto-Generate", self.stop_auto_generate),
            ("—", None),
            ("Clear Data", self.clear_data),
            ("New Level / Reset Env", self.new_level),
            ("Save Data", self.save_data),
            ("Load Data", self.load_data),
        ]:
            if text == "—":
                ttk.Separator(panel, orient='horizontal').pack(fill=tk.X, pady=6)
            else:
                ttk.Button(panel, text=text, command=cmd).pack(fill=tk.X, pady=2)
        self.record_canvas.bind('<Button-1>', lambda e: self.record_canvas.focus_set())

    def setup_vae_tab(self, parent):
        frame = ttk.Frame(parent); frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.vae_canvas = tk.Canvas(frame, width=512, height=256, bg='black')
        self.vae_canvas.pack(pady=10)
        self.vae_progress = ttk.Progressbar(frame, orient=tk.HORIZONTAL,
                                            length=400, mode='determinate')
        self.vae_progress.pack(pady=5)
        self.vae_status = ttk.Label(frame, text="Not initialized")
        self.vae_status.pack(pady=5)
        self.vae_config_label = ttk.Label(frame, text="", foreground="gray")
        self.vae_config_label.pack(pady=2)
        btns = ttk.Frame(frame); btns.pack(pady=5)
        ttk.Button(btns, text="Initialize Weights",
                   command=self.init_vae).pack(side=tk.LEFT, padx=3)
        ttk.Button(btns, text="Start Training",
                   command=self.start_vae_training).pack(side=tk.LEFT, padx=3)
        ttk.Button(btns, text="Stop Training",
                   command=self.stop_vae_training).pack(side=tk.LEFT, padx=3)
        btns2 = ttk.Frame(frame); btns2.pack(pady=2)
        ttk.Button(btns2, text="Save VAE",
                   command=self.save_vae).pack(side=tk.LEFT, padx=3)
        ttk.Button(btns2, text="Load VAE",
                   command=self.load_vae).pack(side=tk.LEFT, padx=3)

    def setup_wm_tab(self, parent):
        frame = ttk.Frame(parent); frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.wm_canvas = tk.Canvas(frame, width=512, height=256, bg='black')
        self.wm_canvas.pack(pady=10)
        self.wm_progress = ttk.Progressbar(frame, orient=tk.HORIZONTAL,
                                           length=400, mode='determinate')
        self.wm_progress.pack(pady=5)
        self.wm_status = ttk.Label(frame, text="Not initialized")
        self.wm_status.pack(pady=5)
        self.wm_config_label = ttk.Label(frame, text="", foreground="gray")
        self.wm_config_label.pack(pady=2)
        btns = ttk.Frame(frame); btns.pack(pady=5)
        ttk.Button(btns, text="Initialize Weights",
                   command=self.init_wm).pack(side=tk.LEFT, padx=3)
        ttk.Button(btns, text="Start Training",
                   command=self.start_wm_training).pack(side=tk.LEFT, padx=3)
        ttk.Button(btns, text="Stop Training",
                   command=self.stop_wm_training).pack(side=tk.LEFT, padx=3)
        btns2 = ttk.Frame(frame); btns2.pack(pady=2)
        ttk.Button(btns2, text="Save WM",
                   command=self.save_wm).pack(side=tk.LEFT, padx=3)
        ttk.Button(btns2, text="Load WM",
                   command=self.load_wm).pack(side=tk.LEFT, padx=3)

    def setup_settings_tab(self, parent):
        frame = ttk.Frame(parent); frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        vae_frame = ttk.LabelFrame(frame, text="Spatial VAE Settings")
        vae_frame.pack(fill=tk.X, pady=5)

        ttk.Label(vae_frame, text="Image size:").grid(row=0, column=0,
                                                      sticky=tk.W, padx=5, pady=2)
        self.img_size_var = tk.IntVar(value=self.img_size)
        ttk.Combobox(vae_frame, textvariable=self.img_size_var,
                     values=[32, 64, 128]).grid(row=0, column=1, padx=5, pady=2)
        ttk.Label(vae_frame, text="(applies on game reload / VAE init)",
                  foreground="gray").grid(row=0, column=2, sticky=tk.W, padx=5)

        ttk.Label(vae_frame, text="Base channels:").grid(row=1, column=0,
                                                         sticky=tk.W, padx=5, pady=2)
        self.vae_base_var = tk.IntVar(value=32)
        ttk.Combobox(vae_frame, textvariable=self.vae_base_var,
                     values=[16, 32, 64]).grid(row=1, column=1, padx=5, pady=2)

        ttk.Label(vae_frame, text="Latent channels:").grid(row=2, column=0,
                                                           sticky=tk.W, padx=5, pady=2)
        self.vae_latent_ch_var = tk.IntVar(value=4)
        ttk.Combobox(vae_frame, textvariable=self.vae_latent_ch_var,
                     values=[2, 4, 8, 16]).grid(row=2, column=1, padx=5, pady=2)

        ttk.Label(vae_frame, text="Spatial size:").grid(row=3, column=0,
                                                        sticky=tk.W, padx=5, pady=2)
        self.vae_spatial_var = tk.IntVar(value=8)
        ttk.Combobox(vae_frame, textvariable=self.vae_spatial_var,
                     values=[4, 8, 16]).grid(row=3, column=1, padx=5, pady=2)

        ttk.Label(vae_frame, text="Epochs:").grid(row=4, column=0,
                                                  sticky=tk.W, padx=5, pady=2)
        self.vae_epochs_var = tk.IntVar(value=50)
        ttk.Entry(vae_frame, textvariable=self.vae_epochs_var).grid(row=4, column=1,
                                                                    padx=5, pady=2)

        ttk.Label(vae_frame, text="Learning rate:").grid(row=5, column=0,
                                                         sticky=tk.W, padx=5, pady=2)
        self.vae_lr_var = tk.DoubleVar(value=1e-3)
        ttk.Entry(vae_frame, textvariable=self.vae_lr_var).grid(row=5, column=1,
                                                                padx=5, pady=2)

        ttk.Label(vae_frame, text="KL weight (β):").grid(row=6, column=0,
                                                         sticky=tk.W, padx=5, pady=2)
        self.vae_kl_var = tk.DoubleVar(value=1e-4)
        ttk.Entry(vae_frame, textvariable=self.vae_kl_var).grid(row=6, column=1,
                                                                padx=5, pady=2)

        ttk.Label(vae_frame, text="Latent noise:").grid(row=7, column=0,
                                                        sticky=tk.W, padx=5, pady=2)
        self.vae_latnoise_var = tk.DoubleVar(value=0.1)
        ttk.Entry(vae_frame, textvariable=self.vae_latnoise_var).grid(
            row=7, column=1, padx=5, pady=2)
        ttk.Label(vae_frame,
                  text="(x latent std; decoder tolerates small WM errors)",
                  foreground="gray").grid(row=7, column=2, sticky=tk.W, padx=5)

        wm_frame = ttk.LabelFrame(frame, text="Flow Matching World Model")
        wm_frame.pack(fill=tk.X, pady=5)

        ttk.Label(wm_frame, text="Hidden dim:").grid(row=0, column=0,
                                                     sticky=tk.W, padx=5, pady=2)
        self.wm_hidden_var = tk.IntVar(value=64)
        ttk.Combobox(wm_frame, textvariable=self.wm_hidden_var,
                     values=[32, 64, 128, 256]).grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(wm_frame, text="Epochs:").grid(row=1, column=0,
                                                 sticky=tk.W, padx=5, pady=2)
        self.wm_epochs_var = tk.IntVar(value=100)
        ttk.Entry(wm_frame, textvariable=self.wm_epochs_var).grid(row=1, column=1,
                                                                  padx=5, pady=2)

        ttk.Label(wm_frame, text="Learning rate:").grid(row=2, column=0,
                                                        sticky=tk.W, padx=5, pady=2)
        self.wm_lr_var = tk.DoubleVar(value=2e-4)
        ttk.Entry(wm_frame, textvariable=self.wm_lr_var).grid(row=2, column=1,
                                                              padx=5, pady=2)

        ttk.Label(wm_frame, text="Sampling steps:").grid(row=5, column=0,
                                                          sticky=tk.W, padx=5, pady=2)
        self.wm_steps_var = tk.IntVar(value=8)
        ttk.Combobox(wm_frame, textvariable=self.wm_steps_var,
                     values=[1, 2, 4, 8, 16, 32]).grid(row=5, column=1, padx=5, pady=2)

        ttk.Label(wm_frame, text="Context frames (K):").grid(row=6, column=0,
                                                             sticky=tk.W, padx=5, pady=2)
        self.wm_ctx_var = tk.IntVar(value=4)
        ttk.Combobox(wm_frame, textvariable=self.wm_ctx_var,
                     values=[1, 2, 4, 8]).grid(row=6, column=1, padx=5, pady=2)
        ttk.Label(wm_frame, text="(applies on Initialize WM)",
                  foreground="gray").grid(row=6, column=2, sticky=tk.W, padx=5)

        # Drift prevention augmentation settings
        drift_frame = ttk.LabelFrame(frame, text="Drift Prevention (WM Training)")
        drift_frame.pack(fill=tk.X, pady=5)

        ttk.Label(drift_frame, text="Noise P:").grid(row=0, column=0,
                                                      sticky=tk.W, padx=5, pady=2)
        self.noise_p_var = tk.DoubleVar(value=0.5)
        ttk.Entry(drift_frame, textvariable=self.noise_p_var, width=8).grid(row=0, column=1,
                                                                              padx=5, pady=2,
                                                                              sticky=tk.W)
        ttk.Label(drift_frame, text="(prob. of noising the context frames)",
                  foreground="gray").grid(row=0, column=2, sticky=tk.W, padx=5)

        ttk.Label(drift_frame, text="Max noise δ:").grid(row=1, column=0,
                                                          sticky=tk.W, padx=5, pady=2)
        self.max_noise_var = tk.DoubleVar(value=0.3)
        ttk.Entry(drift_frame, textvariable=self.max_noise_var, width=8).grid(row=1, column=1,
                                                                                padx=5, pady=2,
                                                                                sticky=tk.W)
        ttk.Label(drift_frame, text="(max context noise, in latent-std units; fed to model)",
                  foreground="gray").grid(row=1, column=2, sticky=tk.W, padx=5)

        ttk.Label(drift_frame, text="Self P:").grid(row=2, column=0,
                                                     sticky=tk.W, padx=5, pady=2)
        self.self_p_var = tk.DoubleVar(value=0.5)
        ttk.Entry(drift_frame, textvariable=self.self_p_var, width=8).grid(row=2, column=1,
                                                                             padx=5, pady=2,
                                                                             sticky=tk.W)
        ttk.Label(drift_frame, text="(prob. of building context from the model's own rollout)",
                  foreground="gray").grid(row=2, column=2, sticky=tk.W, padx=5)

        ttk.Label(drift_frame, text="Max self frames:").grid(row=3, column=0,
                                                              sticky=tk.W, padx=5, pady=2)
        self.max_self_frames_var = tk.IntVar(value=4)
        ttk.Combobox(drift_frame, textvariable=self.max_self_frames_var,
                     values=[1, 2, 4, 8]).grid(row=3, column=1, padx=5, pady=2, sticky=tk.W)
        ttk.Label(drift_frame, text="(max rollout length used to build that context)",
                  foreground="gray").grid(row=3, column=2, sticky=tk.W, padx=5)

        aug_frame = ttk.LabelFrame(frame, text="Augmentation")
        aug_frame.pack(fill=tk.X, pady=5)
        ttk.Label(aug_frame, text="Mirror augmentation:").grid(row=0, column=0,
                                                               sticky=tk.W, padx=5, pady=2)
        self.mirror_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(aug_frame, variable=self.mirror_var).grid(row=0, column=1,
                                                                  padx=5, pady=2)
        ttk.Label(aug_frame,
                  text="(mazes / light world only: flips frames, negates turn axis)",
                  foreground="gray").grid(row=1, column=0, columnspan=2,
                                          sticky=tk.W, padx=5, pady=2)

    def setup_inference_tab(self, parent):
        frame = ttk.Frame(parent); frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.inference_canvas = tk.Canvas(frame, width=512, height=512, bg='black')
        self.inference_canvas.pack(side=tk.LEFT, padx=10)
        panel = ttk.Frame(frame); panel.pack(side=tk.LEFT, fill=tk.Y, padx=10)
        self.inference_status = ttk.Label(panel, text="Ready")
        self.inference_status.pack(anchor=tk.W, pady=5)

        wait_frame = ttk.LabelFrame(panel, text="Frame Control")
        wait_frame.pack(fill=tk.X, pady=5)
        self.frame_wait_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(wait_frame, text="Frame wait (step per input)",
                        variable=self.frame_wait_var).pack(anchor=tk.W, padx=5, pady=2)
        ttk.Button(wait_frame, text="Step (0 action)",
                   command=self.step_one_frame).pack(fill=tk.X, padx=5, pady=2)

        noise_frame = ttk.LabelFrame(panel, text="Context Noise")
        noise_frame.pack(fill=tk.X, pady=5)
        self.infer_noise_var = tk.DoubleVar(value=0.1)
        noise_label = ttk.Label(noise_frame, text="σ = 0.100")
        noise_label.pack(anchor=tk.W, padx=5)
        def on_noise_change(val):
            noise_label.config(text=f"σ = {float(val):.3f}")
        ttk.Scale(noise_frame, from_=0.0, to=0.5, orient=tk.HORIZONTAL,
                  variable=self.infer_noise_var,
                  command=on_noise_change).pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(noise_frame,
                  text="Noise added to past frames each step.\n~0.1 keeps long dreams stable; 0 drifts.",
                  foreground="gray").pack(anchor=tk.W, padx=5, pady=2)

        ttk.Button(panel, text="Start Dream",
                   command=self.start_inference).pack(fill=tk.X, pady=2)
        ttk.Button(panel, text="Stop Dream",
                   command=self.stop_inference).pack(fill=tk.X, pady=2)
        ttk.Button(panel, text="Reset Dream",
                   command=self.reset_inference).pack(fill=tk.X, pady=2)
        ttk.Label(panel,
                  text=("Arrow keys control the dream.\n"
                        "Space = fire (Atari).\n"
                        "Reset re-encodes current view."),
                  justify=tk.LEFT).pack(anchor=tk.W, pady=10)
        self.inference_canvas.bind('<Button-1>',
                                   lambda e: self.inference_canvas.focus_set())
        self.display_frame(self.game.render(), self.inference_canvas,
                           'inference_photo')

    def _update_game_info(self):
        name = self.game.name
        if name in ("2D Maze", "3D Maze", "Light World"):
            info = "Up/Down: forward/back\nLeft/Right: turn"
        elif name == "DVD Dodge":
            info = "Arrows: move in 2D"
        elif name == "Square Jump":
            info = "Left/Right: move"
        elif "Car Racing" in name:
            info = "Up: accelerate\nDown: brake\nLeft/Right: steer"
        elif "Cart Pole" in name:
            info = "Left/Right: push cart"
        elif any(x in name for x in ("Pong", "Breakout", "SpaceInvaders")):
            info = ("Arrows: direction\nSpace: fire\n"
                    "Combos → 18 Atari actions")
        else:
            info = ""
        self.game_info_label.config(text=info)

    def display_frame(self, frame, canvas, photo_attr, size=512):
        img = Image.fromarray(frame).resize((size, size), Image.NEAREST)
        photo = ImageTk.PhotoImage(img)
        setattr(self, photo_attr, photo)
        canvas.delete("frame")
        canvas.create_image(0, 0, anchor=tk.NW, image=photo, tags="frame")

    def render_record_view(self):
        self.display_frame(self.game.render(), self.record_canvas, 'record_photo')

    def on_key_press(self, event):
        if isinstance(self.root.focus_get(), (tk.Entry, ttk.Entry)):
            return
        if event.keysym in ['Up', 'Down', 'Left', 'Right', 'space']:
            self.keys[event.keysym] = True
            if event.keysym != 'space':
                self.advance_one_frame = True
            return "break"

    def on_key_release(self, event):
        if isinstance(self.root.focus_get(), (tk.Entry, ttk.Entry)):
            return
        if event.keysym in ['Up', 'Down', 'Left', 'Right', 'space']:
            self.keys[event.keysym] = False
            return "break"

    def read_action(self):
        return self.game.keys_to_action(self.keys)

    # ========== Game switching ==========
    def on_game_change(self, event=None):
        if self.recording or self.auto_gen_running or self.inference_running:
            self.game_name_var.set(self.game.name)
            messagebox.showwarning("Warning",
                "Stop recording / auto-gen / dreaming before switching games.")
            return
        new_name = self.game_name_var.get()
        if new_name == self.game.name:
            return
        if self.frames and not messagebox.askyesno(
                "Switch game",
                "Switching games clears recorded data and trained models. Continue?"):
            self.game_name_var.set(self.game.name)
            return
        try: self.game.close()
        except Exception: pass
        self.frames = []; self.actions = []
        self.last_frame = None
        self.vae = None; self.wm = None
        self.z_ctx = None; self.a_ctx = None
        try:
            self.img_size = int(self.img_size_var.get())
            self.game = GAMES[new_name](self.img_size)
        except Exception as e:
            messagebox.showerror("Game unavailable", str(e))
            self.game = GAMES["3D Maze"](self.img_size)
            self.game_name_var.set(self.game.name)
            return
        self.game.reset()
        self.render_record_view()
        self._update_game_info()
        self.record_status.config(
            text=f"Frames: 0  Game: {new_name}  action_dim={self.game.action_dim}")
        self.vae_status.config(text="Not initialized")
        self.wm_status.config(text="Not initialized")
        self.wm_config_label.config(text="")
        self.wm_progress['value'] = 0
        self.vae_progress['value'] = 0

    def new_level(self):
        if self.recording: self.stop_recording()
        if self.auto_gen_running: self.stop_auto_generate()
        if self.inference_running: self.stop_inference()
        new_size = int(self.img_size_var.get())
        if new_size != self.img_size:
            self.img_size = new_size
            old_name = self.game.name
            try: self.game.close()
            except Exception: pass
            for key, factory in GAMES.items():
                try:
                    g = factory(new_size)
                    if g.name == old_name:
                        self.game = g
                        break
                except Exception:
                    continue
        self.game.new_level()
        self.frames = []; self.actions = []
        self.last_frame = None
        self.z_ctx = None; self.a_ctx = None
        self.render_record_view()
        self.record_status.config(text=f"Frames: 0  New level ({self.game.name})")

    # ========== Record ==========
    def start_recording(self):
        if self.auto_gen_running: self.stop_auto_generate()
        self.frames = []; self.actions = []
        self.last_frame = None
        self.recording = True
        self.record_tick()

    def stop_recording(self):
        self.recording = False

    def record_tick(self):
        if not self.recording: return
        if self.last_frame is None:
            self.frames.append(self.game.render())
        action = self.read_action()
        self.game.step(action)
        new_frame = self.game.render()
        self.frames.append(new_frame); self.actions.append(action)
        self.last_frame = new_frame
        self.display_frame(new_frame, self.record_canvas, 'record_photo')
        status = self.game.status_text()
        suffix = f"  {status}" if status else ""
        self.record_status.config(text=f"Frames: {len(self.frames)}{suffix}")
        self.root.after(40, self.record_tick)

    # ========== Auto-generate ==========
    def auto_generate(self):
        if self.auto_gen_running: return
        if self.recording: self.stop_recording()
        n_steps = simpledialog.askinteger(
            "Auto-Generate Dataset", "Number of steps to generate:",
            initialvalue=5000, minvalue=100, maxvalue=200000, parent=self.root)
        if n_steps is None: return
        self.frames = []; self.actions = []
        self.game.reset()
        self.auto_gen_running = True
        self.auto_gen_step = 0; self.auto_gen_target = n_steps
        frame = self.game.render()
        self.frames.append(frame); self.last_frame = frame
        self.auto_gen_tick()

    def stop_auto_generate(self):
        self.auto_gen_running = False
        self.record_status.config(text=f"Auto-gen stopped. Frames: {len(self.frames)}")

    def auto_gen_tick(self):
        steps_per_tick = 1 if self.game.is_gym else 5
        for _ in range(steps_per_tick):
            if not self.auto_gen_running or self.auto_gen_step >= self.auto_gen_target:
                self.auto_gen_running = False
                self.record_status.config(text=f"Auto-gen done. Frames: {len(self.frames)}")
                if self.last_frame is not None:
                    self.display_frame(self.last_frame, self.record_canvas, 'record_photo')
                return
            self.auto_gen_step_once()
        if self.last_frame is not None:
            self.display_frame(self.last_frame, self.record_canvas, 'record_photo')
        self.record_status.config(
            text=f"Auto-gen: {self.auto_gen_step}/{self.auto_gen_target}  "
                 f"Frames: {len(self.frames)}")
        self.root.after(1, self.auto_gen_tick)

    def auto_gen_step_once(self):
        old_px = getattr(self.game, 'px', None)
        old_py = getattr(self.game, 'py', None)
        action = self.game.auto_action()
        new_frame, _ = self.game.step(action)
        if old_px is not None and hasattr(self.game, '_auto_stuck_counter') and action[0] > 0:
            moved = (abs(self.game.px - old_px) > 1e-4 or abs(self.game.py - old_py) > 1e-4)
            if not moved:
                self.game._auto_stuck_counter += 1
            else:
                self.game._auto_stuck_counter = 0
            if self.game._auto_stuck_counter > 10:
                self.game._auto_last_cell = None
                self.game._auto_stuck_counter = 0
                self.game.theta = random.uniform(0, 2 * math.pi)
        self.frames.append(new_frame); self.actions.append(action)
        self.last_frame = new_frame
        self.auto_gen_step += 1

    # ========== Data I/O ==========
    def clear_data(self):
        if self.auto_gen_running: self.stop_auto_generate()
        self.frames = []; self.actions = []; self.last_frame = None
        self.record_status.config(text="Frames: 0")

    def save_data(self):
        if len(self.frames) == 0:
            messagebox.showinfo("Info", "No data to save."); return
        path = filedialog.asksaveasfilename(defaultextension=".npz",
                                            filetypes=[("NPZ", "*.npz")])
        if path:
            np.savez_compressed(path,
                                frames=np.stack(self.frames),
                                actions=np.stack(self.actions))
            messagebox.showinfo("Info", f"Saved {len(self.frames)} frames to {path}")

    def load_data(self):
        path = filedialog.askopenfilename(filetypes=[("NPZ", "*.npz")])
        if path:
            data = np.load(path)
            self.frames = list(data['frames'])
            self.actions = list(data['actions'])
            self.last_frame = self.frames[-1] if self.frames else None
            self.recording = False
            self.record_status.config(text=f"Loaded {len(self.frames)} frames")

    # ========== Preview helpers ==========
    def _make_compare_grid(self, left_t, right_t, size=128):
        left = left_t.detach().cpu().permute(0, 2, 3, 1).numpy()
        right = right_t.detach().cpu().permute(0, 2, 3, 1).numpy()
        n = left.shape[0]
        grid = np.zeros((n * size, 2 * size, 3), dtype=np.uint8)
        for i in range(n):
            l = (left[i] * 255).clip(0, 255).astype(np.uint8)
            r = (right[i] * 255).clip(0, 255).astype(np.uint8)
            grid[i * size:(i + 1) * size, :size] = np.array(
                Image.fromarray(l).resize((size, size), Image.NEAREST))
            grid[i * size:(i + 1) * size, size:] = np.array(
                Image.fromarray(r).resize((size, size), Image.NEAREST))
        return Image.fromarray(grid)

    def _preview_schedule(self, epochs):
        freq = 5 if epochs >= 5 else 1
        def should(epoch):
            return epoch == 0 or (epoch + 1) % freq == 0 or epoch == epochs - 1
        return should

    # ========== VAE init/train/save/load ==========
    def init_vae(self):
        base_ch = self.vae_base_var.get()
        latent_ch = self.vae_latent_ch_var.get()
        spatial_size = self.vae_spatial_var.get()
        img_size = int(self.img_size_var.get())

        ratio = img_size // spatial_size
        if ratio < 1 or (ratio & (ratio - 1)) != 0:
            messagebox.showerror(
                "Error",
                f"img_size ({img_size}) / spatial_size ({spatial_size}) "
                f"must be a power of 2.")
            return

        self.img_size = img_size
        self.vae = SpatialVAE(
            in_channels=3, latent_channels=latent_ch,
            spatial_size=spatial_size, img_size=img_size,
            base_channels=base_ch,
        ).to(self.device)

        # Sanity check shapes
        dummy = torch.zeros(1, 3, img_size, img_size, device=self.device)
        with torch.no_grad():
            recon, mu, logvar = self.vae(dummy)
        assert mu.shape[1:] == (latent_ch, spatial_size, spatial_size), \
            f"Unexpected latent shape {mu.shape}"
        assert recon.shape[-2:] == (img_size, img_size), \
            f"Unexpected recon shape {recon.shape}"

        n_params = sum(p.numel() for p in self.vae.parameters())
        self.vae_status.config(
            text=f"VAE initialized (latent={latent_ch}x{spatial_size}x{spatial_size}, "
                 f"size={img_size})")
        self.vae_config_label.config(text=f"params={n_params:,}")

        if self.wm is not None and self.wm.latent_channels != latent_ch:
            self.wm = None
            self.wm_status.config(text="Not initialized (reinit WM)")

    def start_vae_training(self):
        if self.vae is None:
            messagebox.showerror("Error", "Initialize VAE weights first."); return
        if len(self.frames) < 20:
            messagebox.showerror("Error", "Need at least 20 frames."); return
        if self.vae_stop_event is not None: self.vae_stop_event.set()

        epochs = self.vae_epochs_var.get()
        lr = self.vae_lr_var.get()
        kl_weight = float(self.vae_kl_var.get())
        lat_noise = float(self.vae_latnoise_var.get())
        mirror = self.mirror_var.get()
        size = self.vae.img_size

        frames_np = np.stack([to_vae_size(f, size) for f in self.frames]
                             ).astype(np.float32) / 255.0
        frames_t = torch.from_numpy(frames_np).permute(0, 3, 1, 2).contiguous()
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(frames_t), batch_size=32, shuffle=True)

        preview_rng = np.random.RandomState(42)
        preview_idx = preview_rng.choice(len(frames_t), min(4, len(frames_t)),
                                         replace=False)
        preview_frames_cpu = frames_t[preview_idx]

        optimizer = torch.optim.Adam(self.vae.parameters(), lr=lr)
        stop_event = threading.Event()
        self.vae_stop_event = stop_event
        q = queue.Queue(); self.vae_queue = q
        vae = self.vae
        should_preview = self._preview_schedule(epochs)

        def train():
            vae.train()
            for epoch in range(epochs):
                if stop_event.is_set():
                    q.put(('done', True)); return
                total_loss = 0
                for batch in loader:
                    x = batch[0].to(self.device)
                    if mirror and random.random() < 0.5:
                        x = torch.flip(x, dims=[3])
                    mu, logvar = vae.encode(x)
                    z = vae.reparameterize(mu, logvar)
                    if lat_noise > 0:
                        # decoder must survive slightly-off latents (WM error)
                        zstd = z.detach().std(dim=(0, 2, 3), keepdim=True)
                        z = z + lat_noise * zstd * torch.randn_like(z)
                    recon = vae.decode(z)
                    recon_loss = F.mse_loss(recon, x)
                    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                    loss = recon_loss + kl_weight * kl_loss
                    optimizer.zero_grad(); loss.backward(); optimizer.step()
                    total_loss += loss.item()
                avg_loss = total_loss / len(loader)
                q.put(('progress', epoch + 1, epochs, avg_loss))
                if should_preview(epoch):
                    vae.eval()
                    pv = preview_frames_cpu.to(self.device)
                    with torch.no_grad():
                        recon_pv, _, _ = vae(pv)
                    img = self._make_compare_grid(pv, recon_pv)
                    q.put(('preview', img))
                    vae.train()
            vae.eval(); q.put(('done', False))

        threading.Thread(target=train, daemon=True).start()
        self.poll_vae_queue()

    def stop_vae_training(self):
        if self.vae_stop_event is not None: self.vae_stop_event.set()

    def poll_vae_queue(self):
        q = self.vae_queue
        if q is None: return
        try:
            while True:
                msg = q.get_nowait()
                if msg[0] == 'progress':
                    _, epoch, total, loss = msg
                    self.vae_progress['value'] = epoch / total * 100
                    self.vae_status.config(text=f"Epoch {epoch}/{total}  Loss: {loss:.4f}")
                elif msg[0] == 'preview':
                    img = msg[1]
                    photo = ImageTk.PhotoImage(img)
                    self.vae_photo = photo
                    self.vae_canvas.delete("all")
                    self.vae_canvas.config(width=img.width, height=img.height)
                    self.vae_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
                elif msg[0] == 'done':
                    if msg[1]: self.vae_status.config(text="Stopped")
                    else:
                        self.vae_progress['value'] = 100
                        self.vae_status.config(text="Training complete!")
                    return
        except queue.Empty:
            pass
        self.root.after(100, self.poll_vae_queue)

    def save_vae(self):
        if self.vae is None:
            messagebox.showinfo("Info", "No VAE to save. Initialize first."); return
        path = filedialog.asksaveasfilename(defaultextension=".pt",
                                            filetypes=[("PyTorch", "*.pt")])
        if not path: return
        torch.save({'config': self.vae.get_config(),
                    'state_dict': self.vae.state_dict()}, path)
        messagebox.showinfo("Saved", f"VAE saved to {path}")

    def load_vae(self):
        path = filedialog.askopenfilename(filetypes=[("PyTorch", "*.pt")])
        if not path: return
        try:
            ck = torch.load(path, map_location=self.device, weights_only=False)
            self.vae = SpatialVAE(**ck['config']).to(self.device)
            self.vae.load_state_dict(ck['state_dict'])
            self.vae.eval()
            cfg = ck['config']
            self.vae_status.config(
                text=f"VAE loaded (latent={cfg['latent_channels']}x"
                     f"{cfg['spatial_size']}x{cfg['spatial_size']}, "
                     f"size={cfg['img_size']})")
            self.vae_config_label.config(
                text=f"params={sum(p.numel() for p in self.vae.parameters()):,}")
            if self.wm is not None and self.wm.latent_channels != cfg['latent_channels']:
                self.wm = None
                self.wm_status.config(text="Not initialized (reinit WM)")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load VAE: {e}")

    # ========== WM init/train/save/load ==========
    def init_wm(self):
        if self.vae is None:
            messagebox.showerror("Error", "Initialize VAE first."); return
        hidden_dim = self.wm_hidden_var.get()
        self.wm = FlowUNet(
            latent_channels=self.vae.latent_channels,
            spatial_size=self.vae.spatial_size,
            action_dim=self.game.action_dim,
            hidden_dim=hidden_dim,
            cond_dim=hidden_dim,
            context_frames=int(self.wm_ctx_var.get()),
        ).to(self.device)
        n_params = sum(p.numel() for p in self.wm.parameters() if p.requires_grad)
        self.wm_status.config(
            text=f"WM initialized (latent={self.vae.latent_channels}x"
                 f"{self.vae.spatial_size}x{self.vae.spatial_size}, "
                 f"act={self.game.action_dim}, K={self.wm.context_frames})")
        self.wm_config_label.config(
            text=f"hidden={hidden_dim}  act_dim={self.game.action_dim}  "
                 f"K={self.wm.context_frames}  "
                 f"params={n_params:,}")

    def start_wm_training(self):
        if self.vae is None:
            messagebox.showerror("Error", "Initialize VAE first."); return
        if self.wm is None:
            messagebox.showerror("Error", "Initialize WM weights first."); return
        if len(self.actions) < 20:
            messagebox.showerror("Error", "Need at least 20 transitions."); return
        if self.wm.action_dim != self.game.action_dim:
            messagebox.showerror("Error",
                f"WM action_dim={self.wm.action_dim} ≠ "
                f"game action_dim={self.game.action_dim}. Reinitialize WM.")
            return
        if self.wm_stop_event is not None: self.wm_stop_event.set()

        epochs = self.wm_epochs_var.get()
        lr = self.wm_lr_var.get()
        wm = self.wm; vae = self.vae; dev = self.device

        # Drift-prevention settings
        noise_p = float(self.noise_p_var.get())
        max_noise = float(self.max_noise_var.get())
        self_p = float(self.self_p_var.get())
        R = int(self.max_self_frames_var.get()) if self_p > 0 else 0
        K = wm.context_frames
        window = K + R + 1
        n_steps = int(self.wm_steps_var.get())
        n_roll = max(1, min(n_steps, 4))          # cheaper sampler for rollouts
        preview_sigma = float(self.infer_noise_var.get())

        # Mirroring is only valid where action[1] is the turn axis.
        mirror = (self.mirror_var.get() and self.game.continuous_actions
                  and self.game.name.strip() in MIRRORABLE_GAMES)

        self.vae.eval()
        size = vae.img_size
        N = min(len(self.frames), len(self.actions) + 1)
        if N < window + 2:
            messagebox.showerror("Error",
                f"Need at least {window + 2} frames for context={K} + "
                f"rollout={R}."); return
        frames_np = np.stack([to_vae_size(f, size) for f in self.frames[:N]]
                             ).astype(np.float32) / 255.0
        frames_t = torch.from_numpy(frames_np).permute(0, 3, 1, 2).contiguous()

        def encode_all(x):
            out = []
            with torch.no_grad():
                for i in range(0, len(x), 64):
                    mu, _ = vae.encode(x[i:i + 64].to(dev))
                    out.append(mu.cpu())
            return torch.cat(out)

        latents = encode_all(frames_t)

        # Per-channel normalization so the flow works in ~unit-variance space
        # (noise levels below are then meaningful relative to the data).
        mean = latents.mean(dim=(0, 2, 3), keepdim=True)
        std = latents.std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-3)
        with torch.no_grad():
            wm.lat_mean.copy_(mean.to(dev)); wm.lat_std.copy_(std.to(dev))
        latents = (latents - mean) / std

        actions_t = torch.from_numpy(
            np.stack(self.actions[:N - 1]).astype(np.float32))

        latents_flip = actions_flip = None
        if mirror:
            latents_flip = (encode_all(torch.flip(frames_t, dims=[3])) - mean) / std
            actions_flip = actions_t.clone()
            actions_flip[..., 1] = -actions_flip[..., 1]   # mirrored ONCE here

        dataset = FlowDataset(latents, actions_t, window,
                              latents_flip=latents_flip,
                              actions_flip=actions_flip)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=64, shuffle=True, drop_last=len(dataset) > 64)

        optimizer = torch.optim.AdamW(wm.parameters(), lr=lr, weight_decay=1e-4)
        stop_event = threading.Event()
        self.wm_stop_event = stop_event
        q = queue.Queue(); self.wm_queue = q

        # Preview = real P-step autoregressive rollout from true context with
        # the true actions, compared to the true frame P steps later.
        P = min(10, N - K)
        n_valid = N - (K + P) + 1
        pidx = torch.from_numpy(np.random.RandomState(42).choice(
            n_valid, min(4, n_valid), replace=False)).long()
        prev_lat = latents[pidx[:, None] + torch.arange(K + P)[None, :]]
        prev_act = actions_t[pidx[:, None] + torch.arange(K + P - 1)[None, :]]
        prev_gt = frames_t[pidx + K + P - 1]
        should_preview = self._preview_schedule(epochs)

        def sample_sigma(B):
            sig = torch.rand(B, device=dev) * max_noise
            return sig * (torch.rand(B, device=dev) < noise_p).float()

        def run():
            ema = copy.deepcopy(wm).eval()
            for p in ema.parameters(): p.requires_grad_(False)
            step = 0

            def finish(stopped):
                wm.load_state_dict(ema.state_dict()); wm.eval()
                q.put(('done', stopped))

            wm.train()
            for epoch in range(epochs):
                if stop_event.is_set():
                    finish(True); return
                total_loss = 0.0
                for lat, act in loader:
                    lat = lat.to(dev); act = act.to(dev)
                    B = lat.shape[0]

                    # --- self-rollout: build the context from the model's own
                    # predictions (r steps), then train on the NEXT true frame.
                    r = (random.randint(1, R)
                         if (R > 0 and random.random() < self_p) else 0)
                    ctx = lat[:, :K]
                    if r > 0:
                        with torch.no_grad():
                            for m in range(r):
                                sig = sample_sigma(B)
                                z = flow_sample(wm, add_ctx_noise(ctx, sig),
                                                act[:, m:m + K], sig, n_roll)
                                ctx = torch.cat([ctx[:, 1:], z.unsqueeze(1)], 1)
                    target = lat[:, r + K]
                    acts_in = act[:, r:r + K]

                    # --- noise augmentation of the context (level is an input)
                    sig = sample_sigma(B)
                    ctx_in = add_ctx_noise(ctx, sig)

                    # --- rectified-flow loss
                    t = torch.rand(B, device=dev)
                    x0 = torch.randn_like(target)
                    tt = t.view(-1, 1, 1, 1)
                    x_t = (1 - tt) * x0 + tt * target
                    v_pred = wm(x_t, t, ctx_in, acts_in, sig)
                    loss = F.mse_loss(v_pred, target - x0)

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
                    optimizer.step()
                    total_loss += loss.item()

                    step += 1
                    d = min(0.999, (1 + step) / (10 + step))
                    with torch.no_grad():
                        for pe, p in zip(ema.parameters(), wm.parameters()):
                            pe.mul_(d).add_(p.detach(), alpha=1 - d)

                q.put(('progress', epoch + 1, epochs,
                       total_loss / max(1, len(loader))))

                if should_preview(epoch):
                    with torch.no_grad():
                        ctx = prev_lat[:, :K].to(dev)
                        pa = prev_act.to(dev)
                        psig = torch.full((ctx.shape[0],), preview_sigma,
                                          device=dev)
                        for m in range(P):
                            z = flow_sample(ema, add_ctx_noise(ctx, psig),
                                            pa[:, m:m + K], psig, n_steps)
                            ctx = torch.cat([ctx[:, 1:], z.unsqueeze(1)], 1)
                        pred = ctx[:, -1]
                        mse = F.mse_loss(pred, prev_lat[:, -1].to(dev)).item()
                        frames = vae.decode(ema.denormalize(pred))
                    q.put(('preview', self._make_compare_grid(prev_gt, frames),
                           mse, P))
            finish(False)

        def train():
            try:
                run()
            except Exception as e:
                import traceback; traceback.print_exc()
                q.put(('error', str(e)))

        threading.Thread(target=train, daemon=True).start()
        self.poll_wm_queue()

    def stop_wm_training(self):
        if self.wm_stop_event is not None: self.wm_stop_event.set()

    def poll_wm_queue(self):
        q = self.wm_queue
        if q is None: return
        try:
            while True:
                msg = q.get_nowait()
                if msg[0] == 'progress':
                    _, epoch, total, loss = msg
                    self.wm_progress['value'] = epoch / total * 100
                    extra = ""
                    roll = getattr(self, 'wm_last_roll', None)
                    if roll is not None:
                        extra = f"   {roll[1]}-step rollout latent MSE: {roll[0]:.3f}"
                    self.wm_status.config(
                        text=f"Epoch {epoch}/{total}  Loss: {loss:.4f}{extra}")
                elif msg[0] == 'preview':
                    img = msg[1]
                    self.wm_last_roll = (msg[2], msg[3])
                    photo = ImageTk.PhotoImage(img)
                    self.wm_photo = photo
                    self.wm_canvas.delete("all")
                    self.wm_canvas.config(width=img.width, height=img.height)
                    self.wm_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
                elif msg[0] == 'error':
                    self.wm_status.config(text="Training failed")
                    messagebox.showerror("WM training error", msg[1])
                    return
                elif msg[0] == 'done':
                    if msg[1]: self.wm_status.config(text="Stopped")
                    else:
                        self.wm_progress['value'] = 100
                        self.wm_status.config(text="Training complete!")
                    return
        except queue.Empty:
            pass
        self.root.after(100, self.poll_wm_queue)

    def save_wm(self):
        if self.wm is None:
            messagebox.showinfo("Info", "No WM to save. Initialize first."); return
        path = filedialog.asksaveasfilename(defaultextension=".pt",
                                            filetypes=[("PyTorch", "*.pt")])
        if not path: return
        torch.save({'config': self.wm.get_config(),
                    'state_dict': self.wm.state_dict()}, path)
        messagebox.showinfo("Saved", f"WM saved to {path}")

    def load_wm(self):
        path = filedialog.askopenfilename(filetypes=[("PyTorch", "*.pt")])
        if not path: return
        try:
            ck = torch.load(path, map_location=self.device, weights_only=False)
            cfg = ck['config']
            if cfg['action_dim'] != self.game.action_dim:
                messagebox.showerror("Action mismatch",
                    f"WM trained with action_dim={cfg['action_dim']} "
                    f"but current game needs action_dim={self.game.action_dim}.")
                return
            if self.vae is not None:
                if cfg['latent_channels'] != self.vae.latent_channels:
                    messagebox.showerror("Latent mismatch",
                        f"WM latent_channels={cfg['latent_channels']} "
                        f"but VAE has {self.vae.latent_channels}.")
                    return
            self.wm = FlowUNet(**cfg).to(self.device)
            self.wm.load_state_dict(ck['state_dict'])
            self.wm.eval()
            self.wm_status.config(text=f"WM loaded (act={cfg['action_dim']})")
            self.wm_config_label.config(
                text=f"hidden={cfg['hidden_dim']}  act_dim={cfg['action_dim']}  "
                     f"params={sum(p.numel() for p in self.wm.parameters()):,}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load WM: {e}")

    # ========== Inference ==========
    def _encode_frame_to_z(self):
        """Seed the dream: K copies of the current real frame, idle actions."""
        size = self.vae.img_size
        frame = to_vae_size(self.game.render(), size)
        frame_t = (torch.from_numpy(frame.astype(np.float32) / 255.0)
                   .permute(2, 0, 1).contiguous().unsqueeze(0).to(self.device))
        K = self.wm.context_frames
        with torch.no_grad():
            mu, _ = self.vae.encode(frame_t)
            z = self.wm.normalize(mu)
        self.z_ctx = z.unsqueeze(1).repeat(1, K, 1, 1, 1)
        self.a_ctx = torch.zeros(1, K, self.wm.action_dim, device=self.device)

    def start_inference(self):
        if self.vae is None or self.wm is None:
            messagebox.showerror("Error", "Load/init VAE and WM first."); return
        if self.inference_running: return
        if self.recording: self.stop_recording()
        if self.auto_gen_running: self.stop_auto_generate()
        if self.wm.action_dim != self.game.action_dim:
            messagebox.showerror("Error",
                f"WM action_dim={self.wm.action_dim} ≠ "
                f"game action_dim={self.game.action_dim}.")
            return
        self.vae.eval(); self.wm.eval()
        self.advance_one_frame = False
        self._encode_frame_to_z()
        self.inference_running = True
        wait = "frame-wait" if self.frame_wait_var.get() else "continuous"
        steps = self.wm_steps_var.get()
        self.inference_status.config(
            text=f"Dreaming (flow matching, {wait}, steps={steps})")
        self.inference_tick()

    def stop_inference(self):
        self.inference_running = False
        self.inference_status.config(text="Stopped")

    def reset_inference(self):
        if self.vae is None or self.wm is None: return
        self.vae.eval(); self.wm.eval()
        self._encode_frame_to_z()
        self.inference_status.config(text="Dream reset")

    def step_one_frame(self):
        self.advance_one_frame = True

    def inference_tick(self):
        if not self.inference_running: return
        if self.frame_wait_var.get():
            any_key_held = any(self.keys.values())
            if not any_key_held and not self.advance_one_frame:
                self.root.after(30, self.inference_tick); return
            if self.advance_one_frame:
                action = self.game.zero_action()
                self.advance_one_frame = False
            else:
                action = self.read_action()
        else:
            action = self.read_action()

        a_t = (torch.from_numpy(np.asarray(action, dtype=np.float32))
               .view(1, 1, -1).to(self.device))
        steps = int(self.wm_steps_var.get())
        sigma = float(self.infer_noise_var.get())

        # a_ctx[j] is the action taken at frame z_ctx[j]; the newest is "now".
        self.a_ctx = torch.cat([self.a_ctx[:, 1:], a_t], dim=1)
        with torch.no_grad():
            sig = torch.full((1,), sigma, device=self.device)
            # Noise the context exactly like training did -> the model cleans
            # up its own accumulated error instead of amplifying it.
            z_next = flow_sample(self.wm, add_ctx_noise(self.z_ctx, sig),
                                 self.a_ctx, sig, steps)
            self.z_ctx = torch.cat([self.z_ctx[:, 1:], z_next.unsqueeze(1)], dim=1)
            recon = self.vae.decode(self.wm.denormalize(z_next))

        frame = (recon.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255
                 ).clip(0, 255).astype(np.uint8)
        self.display_frame(frame, self.inference_canvas, 'inference_photo')
        self.root.after(50, self.inference_tick)


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
