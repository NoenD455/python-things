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
import heapq


# ---------- Maze generation ----------
def generate_maze(w=15, h=15, seed=None):
    if seed is not None:
        random.seed(seed)
    maze = np.ones((h, w), dtype=np.uint8)
    stack = [(1, 1)]
    maze[1, 1] = 0
    while stack:
        x, y = stack[-1]
        neighbors = []
        for dx, dy in [(2, 0), (-2, 0), (0, 2), (0, -2)]:
            nx, ny = x + dx, y + dy
            if 1 <= nx < w - 1 and 1 <= ny < h - 1 and maze[ny, nx] == 1:
                neighbors.append((nx, ny, dx, dy))
        if neighbors:
            nx, ny, dx, dy = random.choice(neighbors)
            maze[y + dy // 2, x + dx // 2] = 0
            maze[ny, nx] = 0
            stack.append((nx, ny))
        else:
            stack.pop()
    return maze


def try_move(maze, px, py, dx, dy):
    h, w = maze.shape
    new_x = px + dx
    if 0 <= new_x < w and maze[int(py), int(new_x)] == 0:
        px = new_x
    new_y = py + dy
    if 0 <= new_y < h and maze[int(new_y), int(px)] == 0:
        py = new_y
    return px, py


def astar_path(maze, start, goal):
    h, w = maze.shape
    if start == goal:
        return [start]
    if maze[goal[1], goal[0]] != 0:
        return None
    frontier = [(0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0}
    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal:
            break
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = current[0] + dx, current[1] + dy
            if 0 <= nx < w and 0 <= ny < h and maze[ny, nx] == 0:
                new_cost = cost_so_far[current] + 1
                if (nx, ny) not in cost_so_far or new_cost < cost_so_far[(nx, ny)]:
                    cost_so_far[(nx, ny)] = new_cost
                    priority = new_cost + abs(nx - goal[0]) + abs(ny - goal[1])
                    heapq.heappush(frontier, (priority, (nx, ny)))
                    came_from[(nx, ny)] = current
    if goal not in came_from:
        return None
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = came_from.get(current)
    path.reverse()
    return path


# ---------- Raycaster ----------
class Raycaster:
    def __init__(self, maze, resolution=32, fov=math.pi / 3, supersample=2):
        self.maze = maze
        self.res = resolution
        self.fov = fov
        self.supersample = supersample
        self.h, self.w = maze.shape

    def cast(self, px, py, theta):
        ss = self.supersample
        res = self.res * ss
        half = res // 2

        img = np.zeros((res, res, 3), dtype=np.float32)
        img[:half, :] = np.array([50, 50, 70], dtype=np.float32)
        img[half:, :] = np.array([30, 30, 40], dtype=np.float32)

        dir_x = math.cos(theta)
        dir_y = math.sin(theta)
        plane_len = math.tan(self.fov / 2)
        plane_x = -dir_y * plane_len
        plane_y = dir_x * plane_len

        for x in range(res):
            camera_x = 2 * x / res - 1
            ray_dir_x = dir_x + plane_x * camera_x
            ray_dir_y = dir_y + plane_y * camera_x

            map_x = int(px)
            map_y = int(py)
            delta_dist_x = abs(1 / ray_dir_x) if ray_dir_x != 0 else 1e30
            delta_dist_y = abs(1 / ray_dir_y) if ray_dir_y != 0 else 1e30

            if ray_dir_x < 0:
                step_x = -1
                side_dist_x = (px - map_x) * delta_dist_x
            else:
                step_x = 1
                side_dist_x = (map_x + 1 - px) * delta_dist_x
            if ray_dir_y < 0:
                step_y = -1
                side_dist_y = (py - map_y) * delta_dist_y
            else:
                step_y = 1
                side_dist_y = (map_y + 1 - py) * delta_dist_y

            side = 0
            perp_dist = 1e30
            hit_map_x, hit_map_y = map_x, map_y
            for _ in range(4 * (self.w + self.h)):
                if side_dist_x < side_dist_y:
                    side_dist_x += delta_dist_x
                    map_x += step_x
                    side = 0
                else:
                    side_dist_y += delta_dist_y
                    map_y += step_y
                    side = 1
                if map_x < 0 or map_x >= self.w or map_y < 0 or map_y >= self.h:
                    perp_dist = 1e30
                    break
                elif self.maze[map_y, map_x] == 1:
                    if side == 0:
                        perp_dist = side_dist_x - delta_dist_x
                    else:
                        perp_dist = side_dist_y - delta_dist_y
                    hit_map_x, hit_map_y = map_x, map_y
                    break

            perp_dist = max(perp_dist, 0.01)
            line_height = int(res / perp_dist)
            draw_start = max(0, half - line_height // 2)
            draw_end = min(res - 1, half + line_height // 2)

            checker = (hit_map_x + hit_map_y) & 1
            if side == 0:
                base = (np.array([205, 165, 120], dtype=np.float32) if checker
                        else np.array([145, 105, 70], dtype=np.float32))
            else:
                base = (np.array([175, 185, 215], dtype=np.float32) if checker
                        else np.array([115, 125, 160], dtype=np.float32))
            shade = max(0.25, 1.0 - perp_dist / 18.0)
            img[draw_start:draw_end + 1, x] = base * shade

        if ss > 1:
            img = img.reshape(self.res, ss, self.res, ss, 3).mean(axis=(1, 3))
        return img.clip(0, 255).astype(np.uint8)


# ---------- Quantization ----------
def quantize(z, z_min, z_max, num_bins):
    range_ = (z_max - z_min).clamp(min=1e-6)
    normalized = (z - z_min) / range_
    return (normalized * num_bins).floor().long().clamp(0, num_bins - 1)


def dequantize(bins, z_min, z_max, num_bins):
    return z_min + (bins.float() + 0.5) / num_bins * (z_max - z_min)


# ---------- VAE ----------
class VAE(nn.Module):
    def __init__(self, in_channels=3, base_channels=16, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        c = base_channels
        self.c4 = c * 4
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, c, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(c, c * 2, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(c * 2, c * 4, 4, 2, 1), nn.ReLU(),
        )
        self.enc_out_size = c * 4 * 4 * 4
        self.fc_mu = nn.Linear(self.enc_out_size, latent_dim)
        self.fc_logvar = nn.Linear(self.enc_out_size, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, self.enc_out_size)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(c * 4, c * 2, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(c * 2, c, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(c, in_channels, 4, 2, 1), nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.enc(x)
        h = h.reshape(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z):
        h = self.fc_dec(z)
        h = h.reshape(h.size(0), self.c4, 4, 4)
        return self.dec(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# ---------- Looped Transformer Block ----------
class LoopedTransformerBlock(nn.Module):
    """Pre-norm transformer block with no internal residuals.

    The caller is responsible for the residual connection across loop
    iterations (x = x + block(x)), which is what makes the same block
    reusable across many passes.
    """
    def __init__(self, dim, num_heads, ff_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, dim),
        )

    def forward(self, x):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        h = self.norm2(h)
        h = self.mlp(h)
        return h


# ---------- Looped Transformer World Model ----------
class LoopedTransformerWM(nn.Module):
    """Looped transformer world model, faithful to tinyworldformer's design.

    Sequence layout: just the latent chunks [z_1, ..., z_n]. The action
    embedding is broadcast-added to every token (matching tinyworldformer's
    "every token also gets a position vector and the action's vector").
    There are no separate START/END tokens: because every latent chunk is
    predicted in parallel from every input chunk (like tinyworldformer
    predicts every cell from every cell), each position reads out its own
    next chunk and the sequence length stays constant through the loop.

    Loop: x = x + block(x) for each block, num_loops times. Output after
    every pass is returned so the training loop can supervise all depths.
    """
    def __init__(self, latent_dim, action_dim, token_dim=4, model_dim=64,
                 num_heads=4, ff_dim=128, num_blocks=1, num_loops=2, num_bins=0):
        super().__init__()
        assert latent_dim % token_dim == 0, \
            f"latent_dim ({latent_dim}) must be divisible by token_dim ({token_dim})"
        assert model_dim % num_heads == 0, \
            f"model_dim ({model_dim}) must be divisible by num_heads ({num_heads})"

        self.latent_dim = latent_dim
        self.token_dim = token_dim
        self.num_tokens = latent_dim // token_dim
        self.model_dim = model_dim
        self.num_bins = num_bins
        self.z_min = None
        self.z_max = None

        # Per-token projections
        self.latent_proj = nn.Linear(token_dim, model_dim)
        self.action_proj = nn.Linear(action_dim, model_dim)

        # Positional embeddings: one per latent chunk
        self.pos_emb = nn.Parameter(torch.randn(1, self.num_tokens, model_dim) * 0.02)

        # Loopable blocks. num_blocks > 1 is an extension beyond the single
        # block used by tinyworldformer; num_blocks=1 is the faithful config.
        self.blocks = nn.ModuleList([
            LoopedTransformerBlock(model_dim, num_heads, ff_dim)
            for _ in range(num_blocks)
        ])
        self.num_loops = num_loops

        # Per-position readout: each position predicts its own next chunk.
        out_per_token = token_dim if num_bins == 0 else token_dim * num_bins
        self.readout = nn.Linear(model_dim, out_per_token)

    def forward(self, z, a, return_all_loops=False):
        B = z.shape[0]
        z_chunks = z.reshape(B, self.num_tokens, self.token_dim)
        z_emb = self.latent_proj(z_chunks)                   # (B, n, D)
        a_emb = self.action_proj(a).unsqueeze(1)             # (B, 1, D)

        # Action added to every token; no start/end tokens needed since
        # every position predicts its own next chunk in parallel.
        x = z_emb + self.pos_emb + a_emb                     # (B, n, D)

        preds = []
        for _ in range(self.num_loops):
            for block in self.blocks:
                x = x + block(x)
            out = self.readout(x)                            # (B, n, token_dim*K or token_dim)
            if self.num_bins > 0:
                out = out.view(B, self.latent_dim, self.num_bins)
            else:
                out = out.view(B, self.latent_dim)
            preds.append(out)

        if return_all_loops:
            return preds
        return preds[-1]


# ---------- Transition dataset with mirror augmentation ----------
class TransitionDataset(torch.utils.data.Dataset):
    def __init__(self, z_t, actions, z_next, z_t_flip=None, z_next_flip=None):
        self.z_t = z_t
        self.actions = actions
        self.z_next = z_next
        self.z_t_flip = z_t_flip
        self.z_next_flip = z_next_flip

    def __len__(self):
        return self.z_t.shape[0]

    def __getitem__(self, i):
        if self.z_t_flip is not None and random.random() < 0.5:
            a = self.actions[i].clone()
            a[1] = -a[1]
            return self.z_t_flip[i], a, self.z_next_flip[i]
        return self.z_t[i], self.actions[i], self.z_next[i]


# ---------- App ----------
class App:
    def __init__(self, root):
        self.root = root
        root.title("Dream Maze - VAE + Looped Transformer World Model")
        root.geometry("1000x820")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.maze = generate_maze(15, 15, seed=42)
        self.raycaster = Raycaster(self.maze, resolution=32, supersample=2)
        self.px, self.py, self.theta = 1.5, 1.5, 0.0
        self.frames = []
        self.actions = []
        self.keys = {}
        self.recording = False
        self.last_frame = None
        self.vae = None
        self.wm = None
        self.z = None
        self.inference_running = False
        self.vae_stop_event = None
        self.wm_stop_event = None
        self.vae_queue = None
        self.wm_queue = None

        self.speed = 0.08
        self.rot_speed = 0.25

        self.auto_gen_running = False
        self.auto_gen_step = 0
        self.auto_gen_target = 0
        self.auto_gen_target_cell = (1, 1)
        self.auto_gen_path = []
        self.auto_gen_last_cell = None
        self.auto_gen_aim_x = self.px
        self.auto_gen_aim_y = self.py
        self.auto_gen_wiggle = 0
        self.auto_gen_stuck_counter = 0

        self.record_photo = None
        self.inference_photo = None
        self.vae_photo = None
        self.wm_photo = None

        self.setup_ui()
        root.bind('<KeyPress>', self.on_key_press)
        root.bind('<KeyRelease>', self.on_key_release)
        self.render_record_view()

    # ========== UI ==========
    def setup_ui(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True)

        self.record_tab = ttk.Frame(notebook)
        notebook.add(self.record_tab, text="Record")
        self.setup_record_tab()

        self.vae_tab = ttk.Frame(notebook)
        notebook.add(self.vae_tab, text="Train VAE")
        self.setup_vae_tab()

        self.wm_tab = ttk.Frame(notebook)
        notebook.add(self.wm_tab, text="Train World Model")
        self.setup_wm_tab()

        self.settings_tab = ttk.Frame(notebook)
        notebook.add(self.settings_tab, text="Settings")
        self.setup_settings_tab()

        self.inference_tab = ttk.Frame(notebook)
        notebook.add(self.inference_tab, text="Inference")
        self.setup_inference_tab()

    def setup_record_tab(self):
        frame = ttk.Frame(self.record_tab)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.record_canvas = tk.Canvas(frame, width=512, height=512, bg='black')
        self.record_canvas.pack(side=tk.LEFT, padx=10)

        panel = ttk.Frame(frame)
        panel.pack(side=tk.LEFT, fill=tk.Y, padx=10)

        self.record_status = ttk.Label(panel, text="Frames: 0")
        self.record_status.pack(anchor=tk.W, pady=5)

        ttk.Button(panel, text="Start Recording", command=self.start_recording).pack(fill=tk.X, pady=2)
        ttk.Button(panel, text="Stop Recording", command=self.stop_recording).pack(fill=tk.X, pady=2)
        ttk.Separator(panel, orient='horizontal').pack(fill=tk.X, pady=6)
        ttk.Button(panel, text="Auto-Generate Dataset (A*)", command=self.auto_generate).pack(fill=tk.X, pady=2)
        ttk.Button(panel, text="Stop Auto-Generate", command=self.stop_auto_generate).pack(fill=tk.X, pady=2)
        ttk.Separator(panel, orient='horizontal').pack(fill=tk.X, pady=6)
        ttk.Button(panel, text="Clear Data", command=self.clear_data).pack(fill=tk.X, pady=2)
        ttk.Button(panel, text="New Maze", command=self.new_maze).pack(fill=tk.X, pady=2)
        ttk.Button(panel, text="Save Data", command=self.save_data).pack(fill=tk.X, pady=2)
        ttk.Button(panel, text="Load Data", command=self.load_data).pack(fill=tk.X, pady=2)

        ttk.Label(
            panel,
            text=("Manual: arrow keys to move/turn.\n\n"
                  "Auto-gen walks A* but aims at an\n"
                  "offset point (±15°) with occasional\n"
                  "wiggles, producing varied non-\n"
                  "spinning data."),
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=10)

        self.record_canvas.bind('<Button-1>', lambda e: self.record_canvas.focus_set())

    def setup_vae_tab(self):
        frame = ttk.Frame(self.vae_tab)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.vae_canvas = tk.Canvas(frame, width=512, height=256, bg='black')
        self.vae_canvas.pack(pady=10)
        self.vae_progress = ttk.Progressbar(frame, orient=tk.HORIZONTAL, length=400, mode='determinate')
        self.vae_progress.pack(pady=5)
        self.vae_status = ttk.Label(frame, text="Not trained")
        self.vae_status.pack(pady=5)
        btn = ttk.Frame(frame)
        btn.pack(pady=5)
        ttk.Button(btn, text="Start Training", command=self.start_vae_training).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn, text="Stop Training", command=self.stop_vae_training).pack(side=tk.LEFT, padx=5)

    def setup_wm_tab(self):
        frame = ttk.Frame(self.wm_tab)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.wm_canvas = tk.Canvas(frame, width=512, height=256, bg='black')
        self.wm_canvas.pack(pady=10)
        self.wm_progress = ttk.Progressbar(frame, orient=tk.HORIZONTAL, length=400, mode='determinate')
        self.wm_progress.pack(pady=5)
        self.wm_status = ttk.Label(frame, text="Not trained")
        self.wm_status.pack(pady=5)
        self.wm_config_label = ttk.Label(frame, text="", foreground="gray")
        self.wm_config_label.pack(pady=2)
        btn = ttk.Frame(frame)
        btn.pack(pady=5)
        ttk.Button(btn, text="Start Training", command=self.start_wm_training).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn, text="Stop Training", command=self.stop_wm_training).pack(side=tk.LEFT, padx=5)

    def setup_settings_tab(self):
        frame = ttk.Frame(self.settings_tab)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        vae_frame = ttk.LabelFrame(frame, text="VAE Settings")
        vae_frame.pack(fill=tk.X, pady=5)
        ttk.Label(vae_frame, text="Base channels:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.vae_base_var = tk.IntVar(value=16)
        ttk.Combobox(vae_frame, textvariable=self.vae_base_var,
                     values=[8, 16, 32, 64]).grid(row=0, column=1, padx=5, pady=2)
        ttk.Label(vae_frame, text="Latent dim:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.vae_latent_var = tk.IntVar(value=32)
        ttk.Combobox(vae_frame, textvariable=self.vae_latent_var,
                     values=[8, 16, 32, 64, 128]).grid(row=1, column=1, padx=5, pady=2)
        ttk.Label(vae_frame, text="Epochs:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.vae_epochs_var = tk.IntVar(value=50)
        ttk.Entry(vae_frame, textvariable=self.vae_epochs_var).grid(row=2, column=1, padx=5, pady=2)
        ttk.Label(vae_frame, text="Learning rate:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.vae_lr_var = tk.DoubleVar(value=1e-3)
        ttk.Entry(vae_frame, textvariable=self.vae_lr_var).grid(row=3, column=1, padx=5, pady=2)

        wm_frame = ttk.LabelFrame(frame, text="Looped Transformer World Model")
        wm_frame.pack(fill=tk.X, pady=5)

        ttk.Label(wm_frame, text="Token dim (chunks per latent):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.wm_token_dim_var = tk.IntVar(value=4)
        ttk.Combobox(wm_frame, textvariable=self.wm_token_dim_var,
                     values=[1, 2, 4, 8]).grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(wm_frame, text="Model dim:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.wm_model_dim_var = tk.IntVar(value=64)
        ttk.Combobox(wm_frame, textvariable=self.wm_model_dim_var,
                     values=[32, 64, 128, 256]).grid(row=1, column=1, padx=5, pady=2)

        ttk.Label(wm_frame, text="Attention heads:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.wm_num_heads_var = tk.IntVar(value=4)
        ttk.Combobox(wm_frame, textvariable=self.wm_num_heads_var,
                     values=[2, 4, 8]).grid(row=2, column=1, padx=5, pady=2)

        ttk.Label(wm_frame, text="Feedforward dim:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.wm_ff_dim_var = tk.IntVar(value=128)
        ttk.Combobox(wm_frame, textvariable=self.wm_ff_dim_var,
                     values=[64, 128, 256, 512]).grid(row=3, column=1, padx=5, pady=2)

        ttk.Label(wm_frame, text="Num blocks:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=2)
        self.wm_num_blocks_var = tk.IntVar(value=1)
        ttk.Combobox(wm_frame, textvariable=self.wm_num_blocks_var,
                     values=[1, 2, 3]).grid(row=4, column=1, padx=5, pady=2)

        ttk.Label(wm_frame, text="Num loops (per block):").grid(row=5, column=0, sticky=tk.W, padx=5, pady=2)
        self.wm_num_loops_var = tk.IntVar(value=2)
        ttk.Combobox(wm_frame, textvariable=self.wm_num_loops_var,
                     values=[1, 2, 3, 4]).grid(row=5, column=1, padx=5, pady=2)

        ttk.Label(wm_frame, text="Epochs:").grid(row=6, column=0, sticky=tk.W, padx=5, pady=2)
        self.wm_epochs_var = tk.IntVar(value=50)
        ttk.Entry(wm_frame, textvariable=self.wm_epochs_var).grid(row=6, column=1, padx=5, pady=2)

        ttk.Label(wm_frame, text="Learning rate:").grid(row=7, column=0, sticky=tk.W, padx=5, pady=2)
        self.wm_lr_var = tk.DoubleVar(value=1e-3)
        ttk.Entry(wm_frame, textvariable=self.wm_lr_var).grid(row=7, column=1, padx=5, pady=2)

        aug_frame = ttk.LabelFrame(frame, text="Augmentation & Regularization")
        aug_frame.pack(fill=tk.X, pady=5)
        ttk.Label(aug_frame, text="Mirror augmentation:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.mirror_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(aug_frame, variable=self.mirror_var).grid(row=0, column=1, padx=5, pady=2)
        ttk.Label(aug_frame, text="Stillness loss weight:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.stillness_var = tk.DoubleVar(value=0.1)
        ttk.Entry(aug_frame, textvariable=self.stillness_var).grid(row=1, column=1, padx=5, pady=2)

        lat_frame = ttk.LabelFrame(frame, text="Latent Representation")
        lat_frame.pack(fill=tk.X, pady=5)
        ttk.Label(lat_frame, text="Mode:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.latent_mode_var = tk.StringVar(value="Continuous (MSE)")
        ttk.Combobox(lat_frame, textvariable=self.latent_mode_var,
                     values=["Continuous (MSE)", "Quantized (cross-entropy)"],
                     state='readonly').grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
        ttk.Label(lat_frame, text="Bins per dim:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.num_bins_var = tk.IntVar(value=64)
        ttk.Combobox(lat_frame, textvariable=self.num_bins_var,
                     values=[8, 16, 32, 64, 128]).grid(row=1, column=1, padx=5, pady=2)

    def setup_inference_tab(self):
        frame = ttk.Frame(self.inference_tab)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.inference_canvas = tk.Canvas(frame, width=512, height=512, bg='black')
        self.inference_canvas.pack(side=tk.LEFT, padx=10)
        panel = ttk.Frame(frame)
        panel.pack(side=tk.LEFT, fill=tk.Y, padx=10)
        self.inference_status = ttk.Label(panel, text="Ready")
        self.inference_status.pack(anchor=tk.W, pady=5)
        ttk.Button(panel, text="Start Dream", command=self.start_inference).pack(fill=tk.X, pady=2)
        ttk.Button(panel, text="Stop Dream", command=self.stop_inference).pack(fill=tk.X, pady=2)
        ttk.Button(panel, text="Reset Dream", command=self.reset_inference).pack(fill=tk.X, pady=2)
        ttk.Label(panel, text=("Arrow keys control the dream.\n"
                               "Releasing keys should freeze\n"
                               "the view (stillness loss).\n\n"
                               "Drift after a while; Reset\n"
                               "Dream re-grounds."),
                  justify=tk.LEFT).pack(anchor=tk.W, pady=10)
        self.inference_canvas.bind('<Button-1>', lambda e: self.inference_canvas.focus_set())
        self.display_frame(self.raycaster.cast(self.px, self.py, self.theta),
                           self.inference_canvas, 'inference_photo')

    def display_frame(self, frame, canvas, photo_attr, size=512):
        img = Image.fromarray(frame).resize((size, size), Image.NEAREST)
        photo = ImageTk.PhotoImage(img)
        setattr(self, photo_attr, photo)
        canvas.delete("frame")
        canvas.create_image(0, 0, anchor=tk.NW, image=photo, tags="frame")

    def render_record_view(self):
        self.display_frame(self.raycaster.cast(self.px, self.py, self.theta),
                           self.record_canvas, 'record_photo')

    def on_key_press(self, event):
        if isinstance(self.root.focus_get(), (tk.Entry, ttk.Entry)):
            return
        if event.keysym in ['Up', 'Down', 'Left', 'Right']:
            self.keys[event.keysym] = True
            return "break"

    def on_key_release(self, event):
        if isinstance(self.root.focus_get(), (tk.Entry, ttk.Entry)):
            return
        if event.keysym in ['Up', 'Down', 'Left', 'Right']:
            self.keys[event.keysym] = False
            return "break"

    # ========== Physics ==========
    def apply_action(self, action):
        dx = self.speed * math.cos(self.theta) * action[0]
        dy = self.speed * math.sin(self.theta) * action[0]
        self.px, self.py = try_move(self.maze, self.px, self.py, dx, dy)
        self.theta = (self.theta + self.rot_speed * action[1]) % (2 * math.pi)

    def read_keys(self):
        a = np.zeros(2, dtype=np.float32)
        if self.keys.get('Up'): a[0] = 1.0
        if self.keys.get('Down'): a[0] = -1.0
        if self.keys.get('Left'): a[1] = -1.0
        if self.keys.get('Right'): a[1] = 1.0
        return a

    # ========== Record ==========
    def start_recording(self):
        if self.auto_gen_running:
            self.stop_auto_generate()
        self.frames = []
        self.actions = []
        self.last_frame = None
        self.recording = True
        self.record_tick()

    def stop_recording(self):
        self.recording = False

    def record_tick(self):
        if not self.recording:
            return
        frame = self.raycaster.cast(self.px, self.py, self.theta)
        if self.last_frame is None:
            self.frames.append(frame)
        action = self.read_keys()
        self.apply_action(action)
        new_frame = self.raycaster.cast(self.px, self.py, self.theta)
        self.frames.append(new_frame)
        self.actions.append(action)
        self.last_frame = new_frame
        self.display_frame(new_frame, self.record_canvas, 'record_photo')
        self.record_status.config(
            text=f"Frames: {len(self.frames)}  Pos: ({self.px:.2f}, {self.py:.2f})")
        self.root.after(40, self.record_tick)

    # ========== Auto-generate ==========
    def pick_random_cell(self, min_dist=0, from_cell=None):
        h, w = self.maze.shape
        for _ in range(200):
            x = random.randint(1, w - 2)
            y = random.randint(1, h - 2)
            if self.maze[y, x] != 0:
                continue
            if from_cell is not None:
                d = abs(x - from_cell[0]) + abs(y - from_cell[1])
                if d < min_dist:
                    continue
            return (x, y)
        return (1, 1)

    def auto_generate(self):
        if self.auto_gen_running:
            return
        if self.recording:
            self.stop_recording()

        n_steps = simpledialog.askinteger(
            "Auto-Generate Dataset",
            "Number of steps to generate:",
            initialvalue=5000, minvalue=100, maxvalue=200000,
            parent=self.root)
        if n_steps is None:
            return

        self.frames = []
        self.actions = []
        self.px, self.py, self.theta = 1.5, 1.5, 0.0

        self.auto_gen_running = True
        self.auto_gen_step = 0
        self.auto_gen_target = n_steps
        self.auto_gen_target_cell = (1, 1)
        self.auto_gen_path = []
        self.auto_gen_last_cell = None
        self.auto_gen_aim_x = self.px
        self.auto_gen_aim_y = self.py
        self.auto_gen_wiggle = 0
        self.auto_gen_stuck_counter = 0

        frame = self.raycaster.cast(self.px, self.py, self.theta)
        self.frames.append(frame)
        self.last_frame = frame
        self.auto_gen_tick()

    def stop_auto_generate(self):
        self.auto_gen_running = False
        self.record_status.config(text=f"Auto-gen stopped. Frames: {len(self.frames)}")

    def auto_gen_tick(self):
        for _ in range(5):
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
        old_px, old_py = self.px, self.py
        cx, cy = int(self.px), int(self.py)

        if (cx, cy) != self.auto_gen_last_cell:
            self.auto_gen_last_cell = (cx, cy)

            need_replan = (not self.auto_gen_path
                           or (cx, cy) == self.auto_gen_target_cell
                           or self.auto_gen_path[0] != (cx, cy))
            if need_replan:
                if (cx, cy) == self.auto_gen_target_cell or not self.auto_gen_path:
                    for _ in range(50):
                        t = self.pick_random_cell(min_dist=4, from_cell=(cx, cy))
                        if astar_path(self.maze, (cx, cy), t) is not None:
                            self.auto_gen_target_cell = t
                            break
                new_path = astar_path(self.maze, (cx, cy), self.auto_gen_target_cell)
                self.auto_gen_path = new_path if new_path else [(cx, cy)]

            if len(self.auto_gen_path) >= 2:
                nxt = self.auto_gen_path[1]
                aim_x = nxt[0] + 0.5
                aim_y = nxt[1] + 0.5
                dx = aim_x - self.px
                dy = aim_y - self.py
                dist = math.hypot(dx, dy)
                if dist > 1e-3:
                    max_off = math.tan(math.radians(15)) * dist
                    off = random.uniform(-max_off, max_off)
                    perp_x = -dy / dist
                    perp_y = dx / dist
                    aim_x += perp_x * off
                    aim_y += perp_y * off
                self.auto_gen_aim_x = aim_x
                self.auto_gen_aim_y = aim_y
            else:
                self.auto_gen_aim_x = self.px + 0.5 * math.cos(self.theta)
                self.auto_gen_aim_y = self.py + 0.5 * math.sin(self.theta)

            if random.random() < 0.3:
                self.auto_gen_wiggle = random.choice([-1, 1]) * random.randint(1, 3)

        desired = math.atan2(self.auto_gen_aim_y - self.py,
                             self.auto_gen_aim_x - self.px)
        error = (desired - self.theta + math.pi) % (2 * math.pi) - math.pi

        action = np.zeros(2, dtype=np.float32)
        if self.auto_gen_wiggle != 0:
            action[1] = 1.0 if self.auto_gen_wiggle > 0 else -1.0
            self.auto_gen_wiggle -= 1
            action[0] = 1.0
        elif abs(error) > 0.5:
            action[1] = 1.0 if error > 0 else -1.0
            action[0] = 0.0
        else:
            action[1] = max(-1.0, min(1.0, error * 2.0))
            action[0] = 1.0

        self.apply_action(action)

        moved = abs(self.px - old_px) > 1e-4 or abs(self.py - old_py) > 1e-4
        if action[0] > 0 and not moved:
            self.auto_gen_stuck_counter += 1
        else:
            self.auto_gen_stuck_counter = 0
        if self.auto_gen_stuck_counter > 10:
            self.auto_gen_last_cell = None
            self.auto_gen_stuck_counter = 0
            self.theta = random.uniform(0, 2 * math.pi)

        new_frame = self.raycaster.cast(self.px, self.py, self.theta)
        self.frames.append(new_frame)
        self.actions.append(action)
        self.last_frame = new_frame
        self.auto_gen_step += 1

    # ========== Data I/O ==========
    def clear_data(self):
        if self.auto_gen_running:
            self.stop_auto_generate()
        self.frames = []
        self.actions = []
        self.last_frame = None
        self.record_status.config(text="Frames: 0")

    def new_maze(self):
        if self.auto_gen_running:
            self.stop_auto_generate()
        self.recording = False
        seed = random.randint(0, 100000)
        self.maze = generate_maze(15, 15, seed=seed)
        self.raycaster = Raycaster(self.maze, resolution=32, supersample=2)
        self.px, self.py, self.theta = 1.5, 1.5, 0.0
        self.frames = []
        self.actions = []
        self.last_frame = None
        self.render_record_view()
        self.record_status.config(text="Frames: 0  New maze generated")

    def save_data(self):
        if len(self.frames) == 0:
            messagebox.showinfo("Info", "No data to save.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".npz", filetypes=[("NPZ", "*.npz")])
        if path:
            np.savez_compressed(path, frames=np.stack(self.frames), actions=np.stack(self.actions))
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

    # ========== Preview helper ==========
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

    # ========== VAE training ==========
    def start_vae_training(self):
        if len(self.frames) < 20:
            messagebox.showerror("Error", "Need at least 20 frames. Record or auto-generate first.")
            return
        if self.vae_stop_event is not None:
            self.vae_stop_event.set()

        base_ch = self.vae_base_var.get()
        latent_dim = self.vae_latent_var.get()
        epochs = self.vae_epochs_var.get()
        lr = self.vae_lr_var.get()
        mirror = self.mirror_var.get()

        frames = np.stack(self.frames).astype(np.float32) / 255.0
        frames_t = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(frames_t), batch_size=32, shuffle=True)

        preview_rng = np.random.RandomState(42)
        preview_idx = preview_rng.choice(len(frames), min(4, len(frames)), replace=False)
        preview_frames_cpu = frames_t[preview_idx]

        self.vae = VAE(in_channels=3, base_channels=base_ch, latent_dim=latent_dim).to(self.device)
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=lr)
        stop_event = threading.Event()
        self.vae_stop_event = stop_event
        q = queue.Queue()
        self.vae_queue = q
        vae = self.vae
        should_preview = self._preview_schedule(epochs)

        def train():
            for epoch in range(epochs):
                if stop_event.is_set():
                    q.put(('done', True))
                    return
                total_loss = 0
                for batch in loader:
                    x = batch[0].to(self.device)
                    if mirror and random.random() < 0.5:
                        x = torch.flip(x, dims=[3])
                    recon, mu, logvar = vae(x)
                    recon_loss = F.mse_loss(recon, x)
                    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                    loss = recon_loss + 0.001 * kl_loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                avg_loss = total_loss / len(loader)
                q.put(('progress', epoch + 1, epochs, avg_loss))

                if should_preview(epoch):
                    pv = preview_frames_cpu.to(self.device)
                    with torch.no_grad():
                        recon_pv, _, _ = vae(pv)
                    img = self._make_compare_grid(pv, recon_pv)
                    q.put(('preview', img))
            q.put(('done', False))

        threading.Thread(target=train, daemon=True).start()
        self.poll_vae_queue()

    def stop_vae_training(self):
        if self.vae_stop_event is not None:
            self.vae_stop_event.set()

    def poll_vae_queue(self):
        q = self.vae_queue
        if q is None:
            return
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
                    if msg[1]:
                        self.vae_status.config(text="Stopped")
                    else:
                        self.vae_progress['value'] = 100
                        self.vae_status.config(text="Training complete!")
                    return
        except queue.Empty:
            pass
        self.root.after(100, self.poll_vae_queue)

    # ========== World model training ==========
    def start_wm_training(self):
        if self.vae is None:
            messagebox.showerror("Error", "Train VAE first.")
            return
        if len(self.actions) < 20:
            messagebox.showerror("Error", "Need at least 20 transitions.")
            return
        if self.wm_stop_event is not None:
            self.wm_stop_event.set()

        token_dim = self.wm_token_dim_var.get()
        model_dim = self.wm_model_dim_var.get()
        num_heads = self.wm_num_heads_var.get()
        ff_dim = self.wm_ff_dim_var.get()
        num_blocks = self.wm_num_blocks_var.get()
        num_loops = self.wm_num_loops_var.get()
        epochs = self.wm_epochs_var.get()
        lr = self.wm_lr_var.get()
        mirror = self.mirror_var.get()
        stillness_w = float(self.stillness_var.get())
        quantized = self.latent_mode_var.get().startswith("Quantized")
        num_bins = int(self.num_bins_var.get()) if quantized else 0

        if self.vae.latent_dim % token_dim != 0:
            messagebox.showerror("Error",
                f"latent_dim ({self.vae.latent_dim}) must be divisible by token_dim ({token_dim}).")
            return
        if model_dim % num_heads != 0:
            messagebox.showerror("Error",
                f"model_dim ({model_dim}) must be divisible by num_heads ({num_heads}).")
            return

        self.vae.eval()
        frames = np.stack(self.frames).astype(np.float32) / 255.0
        frames_t = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()

        latents = []
        with torch.no_grad():
            for i in range(0, len(frames_t), 64):
                mu, _ = self.vae.encode(frames_t[i:i + 64].to(self.device))
                latents.append(mu.cpu())
        latents = torch.cat(latents)

        if mirror:
            frames_flip = torch.flip(frames_t, dims=[3])
            latents_flip = []
            with torch.no_grad():
                for i in range(0, len(frames_flip), 64):
                    mu, _ = self.vae.encode(frames_flip[i:i + 64].to(self.device))
                    latents_flip.append(mu.cpu())
            latents_flip = torch.cat(latents_flip)
        else:
            latents_flip = None

        actions_t = torch.from_numpy(np.stack(self.actions).astype(np.float32))
        z_t = latents[:-1]
        z_next = latents[1:]
        z_t_flip = latents_flip[:-1] if latents_flip is not None else None
        z_next_flip = latents_flip[1:] if latents_flip is not None else None

        dataset = TransitionDataset(z_t, actions_t, z_next, z_t_flip, z_next_flip)
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

        self.wm = LoopedTransformerWM(
            latent_dim=self.vae.latent_dim,
            action_dim=2,
            token_dim=token_dim,
            model_dim=model_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_blocks=num_blocks,
            num_loops=num_loops,
            num_bins=num_bins,
        ).to(self.device)

        num_params = sum(p.numel() for p in self.wm.parameters() if p.requires_grad)
        num_tokens = self.vae.latent_dim // token_dim
        self.wm_config_label.config(
            text=(f"latent={self.vae.latent_dim}  token_dim={token_dim} ({num_tokens} tokens)  "
                  f"model={model_dim}  heads={num_heads}  ff={ff_dim}  "
                  f"blocks={num_blocks}  loops={num_loops}  "
                  f"bins={num_bins}  params={num_params:,}  "
                  f"(supervising all {num_loops * num_blocks} passes)"))

        if num_bins > 0:
            all_lat = torch.cat([latents, latents_flip], 0) if latents_flip is not None else latents
            z_min = all_lat.min(dim=0).values
            z_max = all_lat.max(dim=0).values
            margin = (z_max - z_min) * 0.05 + 1e-4
            self.wm.z_min = (z_min - margin).to(self.device)
            self.wm.z_max = (z_max + margin).to(self.device)

        optimizer = torch.optim.Adam(self.wm.parameters(), lr=lr)
        stop_event = threading.Event()
        self.wm_stop_event = stop_event
        q = queue.Queue()
        self.wm_queue = q
        wm = self.wm
        vae = self.vae

        preview_rng = np.random.RandomState(42)
        preview_idx = preview_rng.choice(len(actions_t), min(4, len(actions_t)), replace=False)
        preview_idx_list = preview_idx.tolist()
        preview_z_cpu = latents[preview_idx]
        preview_a_cpu = actions_t[preview_idx]
        preview_next_raw_cpu = frames_t[[i + 1 for i in preview_idx_list]]
        should_preview = self._preview_schedule(epochs)

        def train():
            for epoch in range(epochs):
                if stop_event.is_set():
                    q.put(('done', True))
                    return
                total_loss = 0
                for batch in loader:
                    z, a, zn = [b.to(self.device) for b in batch]
                    zero_a = torch.zeros_like(a)

                    if num_bins > 0:
                        target_bins = quantize(zn, wm.z_min, wm.z_max, num_bins)
                        current_bins = quantize(z, wm.z_min, wm.z_max, num_bins)
                        z_in = dequantize(current_bins, wm.z_min, wm.z_max, num_bins)

                        # Per-loop supervision: every pass contributes to the loss.
                        logits_list = wm(z_in, a, return_all_loops=True)
                        recon_loss = sum(
                            F.cross_entropy(lg.reshape(-1, num_bins),
                                            target_bins.reshape(-1))
                            for lg in logits_list
                        ) / len(logits_list)

                        still_logits_list = wm(z_in, zero_a, return_all_loops=True)
                        still_loss = sum(
                            F.cross_entropy(lg.reshape(-1, num_bins),
                                            current_bins.reshape(-1))
                            for lg in still_logits_list
                        ) / len(still_logits_list)
                    else:
                        # Per-loop supervision: every pass contributes to the loss.
                        preds_list = wm(z, a, return_all_loops=True)
                        recon_loss = sum(
                            F.mse_loss(p, zn) for p in preds_list
                        ) / len(preds_list)

                        still_preds_list = wm(z, zero_a, return_all_loops=True)
                        still_loss = sum(
                            F.mse_loss(p, z) for p in still_preds_list
                        ) / len(still_preds_list)

                    loss = recon_loss + stillness_w * still_loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                avg_loss = total_loss / len(loader)
                q.put(('progress', epoch + 1, epochs, avg_loss))

                if should_preview(epoch):
                    pz = preview_z_cpu.to(self.device)
                    pa = preview_a_cpu.to(self.device)
                    pnext = preview_next_raw_cpu.to(self.device)
                    with torch.no_grad():
                        if num_bins > 0:
                            bins = quantize(pz, wm.z_min, wm.z_max, num_bins)
                            z_in = dequantize(bins, wm.z_min, wm.z_max, num_bins)
                            # Use the final pass's prediction for the preview.
                            logits = wm(z_in, pa)  # return_all_loops=False
                            pred_bins = logits.argmax(dim=-1)
                            z_next_pred = dequantize(pred_bins, wm.z_min, wm.z_max, num_bins)
                        else:
                            z_next_pred = wm(pz, pa)  # final pass
                        pred_frames = vae.decode(z_next_pred)
                    img = self._make_compare_grid(pnext, pred_frames)
                    q.put(('preview', img))
            q.put(('done', False))

        threading.Thread(target=train, daemon=True).start()
        self.poll_wm_queue()

    def stop_wm_training(self):
        if self.wm_stop_event is not None:
            self.wm_stop_event.set()

    def poll_wm_queue(self):
        q = self.wm_queue
        if q is None:
            return
        try:
            while True:
                msg = q.get_nowait()
                if msg[0] == 'progress':
                    _, epoch, total, loss = msg
                    self.wm_progress['value'] = epoch / total * 100
                    self.wm_status.config(text=f"Epoch {epoch}/{total}  Loss: {loss:.4f}")
                elif msg[0] == 'preview':
                    img = msg[1]
                    photo = ImageTk.PhotoImage(img)
                    self.wm_photo = photo
                    self.wm_canvas.delete("all")
                    self.wm_canvas.config(width=img.width, height=img.height)
                    self.wm_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
                elif msg[0] == 'done':
                    if msg[1]:
                        self.wm_status.config(text="Stopped")
                    else:
                        self.wm_progress['value'] = 100
                        self.wm_status.config(text="Training complete!")
                    return
        except queue.Empty:
            pass
        self.root.after(100, self.poll_wm_queue)

    # ========== Inference ==========
    def _encode_frame_to_z(self):
        frame = self.raycaster.cast(self.px, self.py, self.theta)
        frame_t = (torch.from_numpy(frame.astype(np.float32) / 255.0)
                   .permute(2, 0, 1).contiguous().unsqueeze(0).to(self.device))
        with torch.no_grad():
            mu, _ = self.vae.encode(frame_t)
            z = mu
            if self.wm.num_bins > 0:
                bins = quantize(z, self.wm.z_min, self.wm.z_max, self.wm.num_bins)
                z = dequantize(bins, self.wm.z_min, self.wm.z_max, self.wm.num_bins)
            self.z = z

    def start_inference(self):
        if self.vae is None or self.wm is None:
            messagebox.showerror("Error", "Train VAE and world model first.")
            return
        if self.inference_running:
            return
        if self.recording:
            self.stop_recording()
        if self.auto_gen_running:
            self.stop_auto_generate()
        self.vae.eval()
        self.wm.eval()
        self._encode_frame_to_z()
        self.inference_running = True
        self.inference_status.config(text="Dreaming...")
        self.inference_tick()

    def stop_inference(self):
        self.inference_running = False
        self.inference_status.config(text="Stopped")

    def reset_inference(self):
        if self.vae is None or self.wm is None:
            return
        self.vae.eval()
        self.wm.eval()
        self._encode_frame_to_z()
        self.inference_status.config(text="Dream reset")

    def inference_tick(self):
        if not self.inference_running:
            return
        action = self.read_keys()
        a_t = torch.from_numpy(action).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if self.wm.num_bins > 0:
                bins = quantize(self.z, self.wm.z_min, self.wm.z_max, self.wm.num_bins)
                z_in = dequantize(bins, self.wm.z_min, self.wm.z_max, self.wm.num_bins)
                logits = self.wm(z_in, a_t)  # final pass
                pred_bins = logits.argmax(dim=-1)
                z_next = dequantize(pred_bins, self.wm.z_min, self.wm.z_max, self.wm.num_bins)
            else:
                z_next = self.wm(self.z, a_t)  # final pass
            recon = self.vae.decode(z_next)
        self.z = z_next
        frame = (recon.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255
                 ).clip(0, 255).astype(np.uint8)
        self.display_frame(frame, self.inference_canvas, 'inference_photo')
        self.root.after(50, self.inference_tick)


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()