# LDM_gui.py
"""
Latent Diffusion — Tkinter GUI.

Tabs: Dataset | VAE Settings | VAE Training | Flow Settings | Flow Training |
      Augmentation | Infill | Generate

New in this version:
  * Flow UNet settings mirror flow_api: per-level attention toggles,
    attn_num_heads, attn_dropout, cond_mode (shift/film/adagn/cross_attn)
  * Aug-padding mode (edge/reflect/symmetric/constant)
  * Infill tab: draw mask at image resolution, encode → latent flow → decode,
    with live preview, composite updates, noise σ, full-edit fraction
  * Model evals are correct: dropout is OFF at inference, ON during training
"""
import tkinter as tk
from tkinter import filedialog, ttk
import torch
import torch.nn.functional as F_nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image, ImageTk, ImageDraw
import os
import threading
import queue
import numpy as np
import multiprocessing
import time
import random
import math
import csv

from flow_api import (
    TextEncoder, TransformerTextEncoder, get_encoder_config,
    load_image_as_rgb, load_image_as_grayscale, text_to_indices,
    ConditionalImageDataset, compute_head_dims,
)
from LDM_api import FlexVAE, RectifiedFlowLatent, VAE_SIZE_CONFIGS


# ==================== Augmentation defaults ====================

AUG_DEFAULTS = {
    'flip_horizontal': True, 'rotation': False, 'random_crop': False, 'crop_scale': 0.8,
    'color_jitter': False, 'brightness': 0.2, 'contrast': 0.2, 'saturation': 0.2, 'hue': 0.1,
    'random_perspective': False, 'perspective_distortion': 0.1,
    'elastic_transform': False, 'elastic_alpha': 30, 'elastic_sigma': 3,
    'jpeg_compression': False, 'jpeg_quality_low': 50, 'jpeg_quality_high': 95,
    'stretch_vertical': False, 'stretch_horizontal': False,
    'aug_padding_mode': 'edge',
}


def default_aug_dict():
    d = {}
    for k, v in AUG_DEFAULTS.items():
        if isinstance(v, bool):
            d[k] = tk.BooleanVar(value=v)
        elif isinstance(v, int):
            d[k] = tk.IntVar(value=v)
        elif isinstance(v, float):
            d[k] = tk.DoubleVar(value=v)
        else:
            d[k] = tk.StringVar(value=v)
    return d


class RectifiedFlowApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Latent Rectified Flow — VAE + Latent Flow")

        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.root.geometry(f"{max(1000, int(sw*0.85))}x{max(720, int(sh*0.85))}")
        self.root.minsize(950, 700)

        self.image_paths = []
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

        self.global_settings = {
            'img_size': tk.IntVar(value=32),
            'color_mode': tk.StringVar(value='rgb'),
        }

        self.vae_settings = {
            'vae_size': tk.StringVar(value='big'),
            'vae_base_channels': tk.IntVar(value=32),
            'vae_latent_channels': tk.IntVar(value=8),
            'vae_latent_h': tk.IntVar(value=4),
            'vae_latent_w': tk.IntVar(value=4),
            'vae_batch_size': tk.IntVar(value=16),
            'vae_lr': tk.DoubleVar(value=1e-3),
            'vae_num_workers': tk.IntVar(value=0),
            'vae_kl_weight': tk.DoubleVar(value=0.0001),
        }

        self.flow_settings = {
            'flow_base_channels': tk.IntVar(value=32),
            'flow_padding_mode': tk.StringVar(value='constant'),

            # NEW: full attention control (mirrors flow_api UNet)
            'attn_level_0': tk.BooleanVar(value=True),
            'attn_level_1': tk.BooleanVar(value=True),
            'attn_level_2': tk.BooleanVar(value=True),
            'attn_num_heads': tk.IntVar(value=4),
            'attn_dropout': tk.DoubleVar(value=0.0),

            # NEW: conditioning mode
            'cond_mode': tk.StringVar(value='shift'),

            'flow_batch_size': tk.IntVar(value=16),
            'flow_lr': tk.DoubleVar(value=2e-4),
            'flow_cfg_dropout': tk.DoubleVar(value=0.1),
            'flow_use_ot': tk.BooleanVar(value=False),
            'ot_method': tk.StringVar(value='sinkhorn'),
            'ot_sinkhorn_eps': tk.DoubleVar(value=0.05),
            'ot_sinkhorn_iters': tk.IntVar(value=50),

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

        # Generation
        self.progressive_grid = tk.BooleanVar(value=False)
        self.prog_interval = tk.IntVar(value=10)

        # Infill tab
        self.infill_original_img = None
        self.infill_mask = None
        self.infill_photo = None
        self.infill_output_photo = None
        self.infill_last_xy = None
        self.infill_erase_mode = False
        self.infill_display_size = 360
        self.infill_in_progress = False
        self.infill_stop_flag = False
        self.infill_brush_size = tk.IntVar(value=8)
        self.infill_steps = tk.IntVar(value=50)
        self.infill_method = tk.StringVar(value='euler')
        self.infill_cfg = tk.DoubleVar(value=1.0)
        self.infill_noise_sigma = tk.DoubleVar(value=1.0)
        self.infill_composite_updates = tk.BooleanVar(value=True)
        self.infill_full_edit_from = tk.DoubleVar(value=0.7)
        self.infill_live_preview = tk.BooleanVar(value=True)
        self.infill_prompt = tk.StringVar(value="")

        # Small live head-dim label
        self.flow_head_dim_var = tk.StringVar(value="—")

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
            ('Augmentation',         self.setup_augmentation_tab),
            ('Infill',               self.setup_infill_tab),
            ('Generate',             self.setup_generation_tab),
        ]
        for name, builder in tabs:
            frame = ttk.Frame(self.notebook)
            self.notebook.add(frame, text=name)
            builder(frame)

        self.status_label = tk.Label(self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    # ---------------- helpers ----------------
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

    def _make_scrollable(self, parent):
        canvas = tk.Canvas(parent)
        sb = tk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        inner = tk.Frame(canvas)
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        return inner

    # ---------------- Dataset ----------------
    def setup_dataset_tab(self, tab):
        main = tk.Frame(tab); main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        left = tk.Frame(main, width=300); left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left.pack_propagate(False)
        tk.Label(left, text="Dataset Management",
                 font=("Arial", 12, "bold")).pack(pady=(0, 10))

        imgf = tk.LabelFrame(left, text="Training Images", padx=5, pady=5)
        imgf.pack(fill=tk.X, pady=(0, 5))
        tk.Button(imgf, text="Add Images", command=self.add_images, width=22).pack(pady=1)
        tk.Button(imgf, text="Add Folder (recursive)", command=self.add_folder, width=22).pack(pady=1)
        tk.Button(imgf, text="Clear All", command=self.clear_images, width=22).pack(pady=1)
        self.image_listbox = tk.Listbox(imgf, height=12)
        self.image_listbox.pack(fill=tk.X, pady=2)

        cf = tk.LabelFrame(left, text="Labels / Captions", padx=5, pady=5)
        cf.pack(fill=tk.X, pady=5)
        tk.Button(cf, text="Load CSV (image,label)", command=self.load_csv, width=22).pack(pady=1)
        tk.Button(cf, text="Use filenames as labels",
                  command=self.use_filenames_as_labels, width=22).pack(pady=1)
        tk.Button(cf, text="Use folder names as labels",
                  command=self.use_folders_as_labels, width=22).pack(pady=1)
        self.csv_status = tk.Label(cf, text="No CSV loaded", fg="red")
        self.csv_status.pack()

        right = tk.Frame(main); right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        lf = tk.LabelFrame(right, text="Log", padx=5, pady=5); lf.pack(fill=tk.BOTH, expand=True)
        self.dataset_log_text = tk.Text(lf, height=20, font=("Courier", 9))
        self.dataset_log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = tk.Scrollbar(lf, command=self.dataset_log_text.yview); sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.dataset_log_text.config(yscrollcommand=sb.set)

    # ---------------- VAE Settings ----------------
    def setup_vae_settings_tab(self, tab):
        inner = self._make_scrollable(tab)
        wrap = tk.Frame(inner); wrap.pack(fill=tk.X, padx=20, pady=20)
        tk.Label(wrap, text="VAE Settings",
                 font=("Arial", 14, "bold")).pack(pady=(0, 15))

        mf = tk.LabelFrame(wrap, text="VAE Model", padx=10, pady=10)
        mf.pack(fill=tk.X, pady=5)
        self._combo(mf, "VAE size:", self.vae_settings['vae_size'],
                    ['tiny', 'small', 'medium', 'big', 'large'])
        self._spin(mf, "Base channels:", self.vae_settings['vae_base_channels'], 4, 256)
        self._spin(mf, "Latent channels:", self.vae_settings['vae_latent_channels'], 1, 64)
        self._spin(mf, "Latent height:", self.vae_settings['vae_latent_h'], 1, 64)
        self._spin(mf, "Latent width:", self.vae_settings['vae_latent_w'], 1, 64)

        tf = tk.LabelFrame(wrap, text="VAE Training", padx=10, pady=10)
        tf.pack(fill=tk.X, pady=5)
        self._spin(tf, "Batch size:", self.vae_settings['vae_batch_size'], 1, 128)
        self._entry(tf, "Learning rate:", self.vae_settings['vae_lr'])
        self._entry(tf, "KL weight:", self.vae_settings['vae_kl_weight'])
        self._spin(tf, "DataLoader workers:", self.vae_settings['vae_num_workers'], 0, 8)

        gf = tk.LabelFrame(wrap, text="Global Image Settings", padx=10, pady=10)
        gf.pack(fill=tk.X, pady=5)
        self._combo(gf, "Color mode:", self.global_settings['color_mode'], ['rgb', 'grayscale'])
        self._spin(gf, "Image size:", self.global_settings['img_size'], 16, 128)

    # ---------------- VAE Training ----------------
    def setup_vae_training_tab(self, tab):
        main = tk.Frame(tab); main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        left = tk.Frame(main, width=300); left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left.pack_propagate(False)
        tk.Label(left, text="VAE Training", font=("Arial", 12, "bold")).pack(pady=(0, 10))
        ep = tk.Frame(left); ep.pack(pady=5)
        tk.Label(ep, text="Epochs:").pack(side=tk.LEFT)
        self.vae_epoch_var = tk.StringVar(value="100")
        tk.Entry(ep, textvariable=self.vae_epoch_var, width=8).pack(side=tk.LEFT, padx=5)
        tk.Button(left, text="Initialize VAE", command=self.initialize_vae, width=24).pack(pady=2)
        tk.Button(left, text="Start VAE Training", command=self.start_vae_training,
                  width=24, bg="lightgreen").pack(pady=2)
        tk.Button(left, text="Stop VAE Training", command=self.stop_vae_training,
                  width=24, bg="salmon").pack(pady=2)
        tk.Button(left, text="Save VAE", command=self.save_vae, width=24).pack(pady=2)
        tk.Button(left, text="Load VAE", command=self.load_vae, width=24).pack(pady=2)
        tk.Button(left, text="Show Preview Now",
                  command=self.show_vae_preview, width=24).pack(pady=8)

        right = tk.Frame(main); right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        pf = tk.LabelFrame(right, text="Top: Augmented Input | Bottom: Reconstruction",
                           padx=5, pady=5)
        pf.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        self.vae_preview_canvas = tk.Canvas(pf, bg='gray', width=256, height=128)
        self.vae_preview_canvas.pack()

        lf = tk.LabelFrame(right, text="Log", padx=5, pady=5); lf.pack(fill=tk.BOTH, expand=True)
        self.vae_log_text = tk.Text(lf, height=15, font=("Courier", 9))
        self.vae_log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = tk.Scrollbar(lf, command=self.vae_log_text.yview); sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.vae_log_text.config(yscrollcommand=sb.set)

    # ---------------- Flow Settings ----------------
    def setup_flow_settings_tab(self, tab):
        inner = self._make_scrollable(tab)
        wrap = tk.Frame(inner); wrap.pack(fill=tk.X, padx=20, pady=20)
        tk.Label(wrap, text="Flow Settings",
                 font=("Arial", 14, "bold")).pack(pady=(0, 15))

        uf = tk.LabelFrame(wrap, text="Flow UNet", padx=10, pady=10); uf.pack(fill=tk.X, pady=5)
        self._spin(uf, "Base channels:", self.flow_settings['flow_base_channels'], 4, 256)
        self._combo(uf, "Padding mode:", self.flow_settings['flow_padding_mode'],
                    ['constant', 'reflect', 'replicate', 'circular'])

        af = tk.LabelFrame(uf, text="Attention (per resolution)", padx=6, pady=6)
        af.pack(fill=tk.X, pady=(8, 2))
        f = tk.Frame(af); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Enable attention:", width=22, anchor='w').pack(side=tk.LEFT)
        tk.Checkbutton(f, text="L0", variable=self.flow_settings['attn_level_0']).pack(side=tk.LEFT, padx=3)
        tk.Checkbutton(f, text="L1", variable=self.flow_settings['attn_level_1']).pack(side=tk.LEFT, padx=3)
        tk.Checkbutton(f, text="L2", variable=self.flow_settings['attn_level_2']).pack(side=tk.LEFT, padx=3)
        self._spin(af, "Number of heads:", self.flow_settings['attn_num_heads'], 1, 32)
        self._entry(af, "Attention dropout:", self.flow_settings['attn_dropout'])
        tk.Label(af, textvariable=self.flow_head_dim_var,
                 font=("Courier", 9), anchor='w').pack(fill=tk.X, pady=(4, 0))

        tk.Button(af, text="Refresh head dims",
                  command=self._refresh_flow_head_dims).pack(pady=2)
        self._refresh_flow_head_dims()

        tf = tk.LabelFrame(wrap, text="Flow Training", padx=10, pady=10); tf.pack(fill=tk.X, pady=5)
        self._spin(tf, "Batch size:", self.flow_settings['flow_batch_size'], 1, 128)
        self._entry(tf, "Learning rate:", self.flow_settings['flow_lr'])
        self._entry(tf, "CFG dropout:", self.flow_settings['flow_cfg_dropout'])
        tk.Checkbutton(tf, text="Use OT matching (Sinkhorn)",
                       variable=self.flow_settings['flow_use_ot']).pack(anchor='w')
        self._entry(tf, "Sinkhorn epsilon:", self.flow_settings['ot_sinkhorn_eps'])
        self._spin(tf, "Sinkhorn iters:", self.flow_settings['ot_sinkhorn_iters'], 1, 500)

        ef = tk.LabelFrame(wrap, text="EMA", padx=10, pady=10); ef.pack(fill=tk.X, pady=5)
        tk.Checkbutton(ef, text="Enable EMA",
                       variable=self.flow_settings['ema_enabled']).pack(anchor='w')
        self._entry(ef, "Decay:", self.flow_settings['ema_decay'])
        self._combo(ef, "Mode:", self.flow_settings['ema_mode'], ['standard', 'lookahead'])

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
        self._combo(cf, "Cond mode:", self.flow_settings['cond_mode'],
                    ['shift', 'film', 'adagn', 'cross_attn'])
        tk.Label(cf, text="shift = additive · film = scale+shift · "
                          "adagn = adaptive GroupNorm · cross_attn = cross-attention",
                 fg='gray', font=("Arial", 8, "italic")).pack(anchor='w')
        tk.Button(cf, text="Apply preset",
                  command=self.apply_text_encoder_preset).pack(pady=2)
        self._spin(cf, "Embed dim:", self.flow_settings['cond_embed_dim'], 8, 512)
        self._spin(cf, "Hidden size (GRU):", self.flow_settings['cond_hidden_size'], 8, 1024)
        self._spin(cf, "Num layers:", self.flow_settings['cond_num_layers'], 1, 6)
        self._spin(cf, "Num heads (TF):", self.flow_settings['cond_num_heads'], 1, 16)
        self._spin(cf, "FF dim (TF):", self.flow_settings['cond_ff_dim'], 32, 2048)
        self._spin(cf, "Cond dim:", self.flow_settings['cond_dim'], 32, 1024)
        self._spin(cf, "Max text len:", self.flow_settings['cond_text_max_len'], 32, 512)

        r = tk.Frame(wrap); r.pack(fill=tk.X, pady=10)
        tk.Button(r, text="Reset VAE + Flow Settings to Defaults",
                  command=self.reset_all_settings,
                  bg="orange", fg="white", font=("Arial", 10, "bold")).pack(pady=5)

    def _refresh_flow_head_dims(self):
        try:
            base = int(self.flow_settings['flow_base_channels'].get())
            nh = int(self.flow_settings['attn_num_heads'].get())
        except Exception:
            return
        dims = compute_head_dims(base, (1, 2, 4), nh)
        flags = [self.flow_settings['attn_level_0'].get(),
                 self.flow_settings['attn_level_1'].get(),
                 self.flow_settings['attn_level_2'].get()]
        parts = []
        for i, (ch, eff_nh, hd) in enumerate(dims):
            tag = "ON " if flags[i] else "off"
            parts.append(f"L{i}[{tag}] ch={ch} h={eff_nh} hd={hd}")
        self.flow_head_dim_var.set("  ".join(parts))

    # ---------------- Flow Training ----------------
    def setup_flow_training_tab(self, tab):
        main = tk.Frame(tab); main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        left = tk.Frame(main, width=300); left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left.pack_propagate(False)
        tk.Label(left, text="Flow Training", font=("Arial", 12, "bold")).pack(pady=(0, 10))
        ep = tk.Frame(left); ep.pack(pady=5)
        tk.Label(ep, text="Epochs:").pack(side=tk.LEFT)
        self.flow_epoch_var = tk.StringVar(value="200")
        tk.Entry(ep, textvariable=self.flow_epoch_var, width=8).pack(side=tk.LEFT, padx=5)
        tk.Button(left, text="Initialize Flow Model",
                  command=self.initialize_rectified_model, width=24).pack(pady=2)
        tk.Button(left, text="Start Flow Training", command=self.start_rectified_training,
                  width=24, bg="lightgreen").pack(pady=2)
        tk.Button(left, text="Stop Flow Training", command=self.stop_rectified_training,
                  width=24, bg="salmon").pack(pady=2)
        tk.Button(left, text="Save Flow Model",
                  command=self.save_rectified_model, width=24).pack(pady=2)
        tk.Button(left, text="Load Flow Model",
                  command=self.load_rectified_model, width=24).pack(pady=2)

        right = tk.Frame(main); right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        pf = tk.LabelFrame(right, text="Generated Samples (decoded)",
                           padx=5, pady=5)
        pf.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        self.rectified_preview_canvas = tk.Canvas(pf, bg='gray', width=256, height=256)
        self.rectified_preview_canvas.pack()

        prf = tk.LabelFrame(right, text="Test prompt (empty = unconditional)",
                            padx=5, pady=5)
        prf.pack(fill=tk.X, pady=(0, 5))
        self.test_prompt_entry = tk.Entry(prf)
        self.test_prompt_entry.insert(0, "a cute cat")
        self.test_prompt_entry.pack(fill=tk.X, pady=2)
        tk.Button(prf, text="Generate Preview",
                  command=self.rectified_preview_with_prompt).pack(pady=2)

        lf = tk.LabelFrame(right, text="Log", padx=5, pady=5); lf.pack(fill=tk.BOTH, expand=True)
        self.rectified_log_text = tk.Text(lf, height=15, font=("Courier", 9))
        self.rectified_log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = tk.Scrollbar(lf, command=self.rectified_log_text.yview); sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.rectified_log_text.config(yscrollcommand=sb.set)

    # ---------------- Augmentation ----------------
    def setup_augmentation_tab(self, tab):
        tk.Label(tab, text="Augmentation Settings",
                 font=("Arial", 14, "bold")).pack(pady=(15, 10))
        cols = tk.Frame(tab); cols.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        vae_col = tk.LabelFrame(cols, text="VAE Augmentations", padx=10, pady=10)
        vae_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        self._build_aug_controls(vae_col, self.vae_aug_settings)
        flow_col = tk.LabelFrame(cols, text="Flow Augmentations", padx=10, pady=10)
        flow_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))
        self._build_aug_controls(flow_col, self.flow_aug_settings)
        r = tk.Frame(tab); r.pack(fill=tk.X, pady=10)
        tk.Button(r, text="Reset VAE Augs", command=lambda: self._reset_aug(self.vae_aug_settings),
                  bg="orange").pack(side=tk.LEFT, padx=10, expand=True)
        tk.Button(r, text="Reset Flow Augs", command=lambda: self._reset_aug(self.flow_aug_settings),
                  bg="orange").pack(side=tk.RIGHT, padx=10, expand=True)

    def _build_aug_controls(self, parent, aug):
        def row():
            f = tk.Frame(parent); f.pack(fill=tk.X, pady=2); return f

        r = row()
        tk.Label(r, text="Aug padding mode:").pack(side=tk.LEFT)
        ttk.Combobox(r, textvariable=aug['aug_padding_mode'],
                     values=['edge', 'reflect', 'symmetric', 'constant'],
                     state='readonly', width=10).pack(side=tk.RIGHT)

        tk.Checkbutton(parent, text="Horizontal Flip",
                       variable=aug['flip_horizontal']).pack(anchor='w')
        tk.Checkbutton(parent, text="Rotation (±30°)",
                       variable=aug['rotation']).pack(anchor='w')
        r = row()
        tk.Checkbutton(r, text="Random Crop", variable=aug['random_crop']).pack(side=tk.LEFT)
        tk.Label(r, text="scale:").pack(side=tk.LEFT, padx=(10, 2))
        tk.Entry(r, textvariable=aug['crop_scale'], width=6).pack(side=tk.LEFT)
        tk.Checkbutton(parent, text="Color Jitter",
                       variable=aug['color_jitter']).pack(anchor='w')
        r = row()
        tk.Label(r, text="  bri / con / sat / hue:").pack(side=tk.LEFT)
        for k in ['brightness', 'contrast', 'saturation', 'hue']:
            tk.Entry(r, textvariable=aug[k], width=5).pack(side=tk.LEFT, padx=1)
        r = row()
        tk.Checkbutton(r, text="Random Perspective",
                       variable=aug['random_perspective']).pack(side=tk.LEFT)
        tk.Label(r, text="dist:").pack(side=tk.LEFT, padx=(8, 2))
        tk.Entry(r, textvariable=aug['perspective_distortion'], width=6).pack(side=tk.LEFT)
        r = row()
        tk.Checkbutton(r, text="Elastic", variable=aug['elastic_transform']).pack(side=tk.LEFT)
        tk.Label(r, text="α:").pack(side=tk.LEFT, padx=(8, 2))
        tk.Entry(r, textvariable=aug['elastic_alpha'], width=4).pack(side=tk.LEFT)
        tk.Label(r, text="σ:").pack(side=tk.LEFT, padx=(4, 2))
        tk.Entry(r, textvariable=aug['elastic_sigma'], width=4).pack(side=tk.LEFT)
        r = row()
        tk.Checkbutton(r, text="JPEG", variable=aug['jpeg_compression']).pack(side=tk.LEFT)
        tk.Label(r, text="quality:").pack(side=tk.LEFT, padx=(8, 2))
        tk.Entry(r, textvariable=aug['jpeg_quality_low'], width=5).pack(side=tk.LEFT)
        tk.Entry(r, textvariable=aug['jpeg_quality_high'], width=5).pack(side=tk.LEFT)
        r = row()
        tk.Checkbutton(r, text="Stretch V",
                       variable=aug['stretch_vertical']).pack(side=tk.LEFT, padx=(0, 10))
        tk.Checkbutton(r, text="Stretch H",
                       variable=aug['stretch_horizontal']).pack(side=tk.LEFT)

    def _reset_aug(self, aug):
        for k, v in AUG_DEFAULTS.items():
            aug[k].set(v)

    # ---------------- Infill tab ----------------
    def setup_infill_tab(self, tab):
        main = tk.Frame(tab); main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        tk.Label(main, text="Infilling — left-drag paints mask, right-drag erases",
                 font=("Arial", 12, "bold")).pack(pady=(0, 6))

        row0 = tk.Frame(main); row0.pack(fill=tk.X, pady=4)
        tk.Label(row0, text="Prompt:").pack(side=tk.LEFT, padx=(2, 4))
        tk.Entry(row0, textvariable=self.infill_prompt, width=60).pack(
            side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        tk.Label(row0, text="(used only if conditioning is enabled)",
                 fg='gray', font=("Arial", 8, "italic")).pack(side=tk.LEFT, padx=6)

        top = tk.Frame(main); top.pack(fill=tk.X, pady=4)
        tk.Button(top, text="Load Image", command=self.infill_load_image).pack(side=tk.LEFT, padx=2)
        tk.Button(top, text="Clear Mask", command=self.infill_clear_mask).pack(side=tk.LEFT, padx=2)
        tk.Button(top, text="Random Mask", command=self.infill_random_mask).pack(side=tk.LEFT, padx=2)
        tk.Button(top, text="Fill Entire", command=self.infill_fill_all).pack(side=tk.LEFT, padx=2)
        tk.Label(top, text="Brush:").pack(side=tk.LEFT, padx=(10, 0))
        tk.Spinbox(top, from_=1, to=64, textvariable=self.infill_brush_size,
                   width=4).pack(side=tk.LEFT, padx=2)

        top2 = tk.Frame(main); top2.pack(fill=tk.X, pady=4)
        tk.Label(top2, text="Steps:").pack(side=tk.LEFT, padx=(2, 0))
        tk.Spinbox(top2, from_=1, to=200, textvariable=self.infill_steps,
                   width=4).pack(side=tk.LEFT, padx=2)
        tk.Label(top2, text="Method:").pack(side=tk.LEFT, padx=(10, 0))
        ttk.Combobox(top2, textvariable=self.infill_method,
                     values=['euler', 'heun', 'rk4', 'midpoint'],
                     state='readonly', width=8).pack(side=tk.LEFT, padx=2)
        tk.Label(top2, text="CFG:").pack(side=tk.LEFT, padx=(10, 0))
        tk.Entry(top2, textvariable=self.infill_cfg, width=5).pack(side=tk.LEFT, padx=2)
        tk.Label(top2, text="Noise σ:").pack(side=tk.LEFT, padx=(10, 0))
        tk.Scale(top2, from_=0.0, to=1.5, resolution=0.05,
                 variable=self.infill_noise_sigma,
                 orient=tk.HORIZONTAL, length=120).pack(side=tk.LEFT)
        tk.Label(top2, textvariable=self.infill_noise_sigma,
                 width=5, font=("Courier", 9)).pack(side=tk.LEFT)

        top3 = tk.Frame(main); top3.pack(fill=tk.X, pady=4)
        tk.Checkbutton(top3, text="Composite updates",
                       variable=self.infill_composite_updates).pack(side=tk.LEFT, padx=4)
        tk.Label(top3, text="Full-edit from:").pack(side=tk.LEFT, padx=(10, 0))
        tk.Scale(top3, from_=0.0, to=1.0, resolution=0.05,
                 variable=self.infill_full_edit_from,
                 orient=tk.HORIZONTAL, length=140).pack(side=tk.LEFT)
        tk.Label(top3, textvariable=self.infill_full_edit_from,
                 width=5, font=("Courier", 9)).pack(side=tk.LEFT)
        tk.Checkbutton(top3, text="Live preview",
                       variable=self.infill_live_preview).pack(side=tk.LEFT, padx=(10, 4))
        self.infill_btn = tk.Button(top3, text="Inpaint", command=self.infill_run,
                                    bg="lightgreen")
        self.infill_btn.pack(side=tk.LEFT, padx=8)
        self.infill_stop_btn = tk.Button(top3, text="Stop", command=self.infill_stop,
                                         bg="salmon", state=tk.DISABLED)
        self.infill_stop_btn.pack(side=tk.LEFT, padx=2)

        cf = tk.Frame(main); cf.pack(fill=tk.BOTH, expand=True, pady=6)
        left = tk.Frame(cf); left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
        tk.Label(left, text="Draw mask (L=paint, R=erase)").pack()
        self.infill_canvas = tk.Canvas(left, width=self.infill_display_size,
                                       height=self.infill_display_size, bg='black')
        self.infill_canvas.pack()

        right = tk.Frame(cf); right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=4)
        tk.Label(right, text="Result (live)").pack()
        self.infill_output_canvas = tk.Canvas(right, width=self.infill_display_size,
                                              height=self.infill_display_size, bg='gray')
        self.infill_output_canvas.pack()

        self.infill_info = tk.Label(main, text="Load an image to begin.", fg="blue")
        self.infill_info.pack()

        self.infill_canvas.bind("<Button-1>", self.infill_start_paint)
        self.infill_canvas.bind("<B1-Motion>", self.infill_paint)
        self.infill_canvas.bind("<ButtonRelease-1>", self.infill_stop_paint)
        self.infill_canvas.bind("<Button-3>", self.infill_start_erase)
        self.infill_canvas.bind("<B3-Motion>", self.infill_erase)
        self.infill_canvas.bind("<ButtonRelease-3>", self.infill_stop_erase)

    # ---- infill helpers ----
    def infill_load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.jfif *.webp")])
        if not path:
            return
        img_size = self.global_settings['img_size'].get()
        color_mode = self.global_settings['color_mode'].get()
        try:
            if color_mode == 'rgb':
                pil = load_image_as_rgb(path).resize((img_size, img_size), Image.BICUBIC)
            else:
                pil = load_image_as_grayscale(path).resize((img_size, img_size), Image.BICUBIC)
        except Exception as e:
            self.infill_info.config(text=f"Load error: {e}")
            return
        self.infill_original_img = pil
        self.infill_mask = np.zeros((img_size, img_size), dtype=np.uint8)
        self.infill_redraw()
        self.infill_info.config(text=f"Loaded: {os.path.basename(path)}")

    def infill_clear_mask(self):
        if self.infill_original_img is None:
            return
        img_size = self.global_settings['img_size'].get()
        self.infill_mask = np.zeros((img_size, img_size), dtype=np.uint8)
        self.infill_redraw()
        self.infill_info.config(text="Mask cleared.")

    def infill_random_mask(self):
        if self.infill_original_img is None:
            return
        img_size = self.global_settings['img_size'].get()
        ratio = 0.3
        m = (np.random.rand(img_size, img_size) < ratio).astype(np.uint8) * 255
        self.infill_mask = m
        self.infill_redraw()
        self.infill_info.config(text="Random mask generated.")

    def infill_fill_all(self):
        if self.infill_original_img is None:
            return
        img_size = self.global_settings['img_size'].get()
        self.infill_mask = np.full((img_size, img_size), 255, dtype=np.uint8)
        self.infill_redraw()
        self.infill_info.config(text="Mask = entire image.")

    def infill_redraw(self):
        if self.infill_original_img is None:
            return
        img_size = self.global_settings['img_size'].get()
        orig = self.infill_original_img.resize((img_size, img_size), Image.NEAREST).convert('RGBA')
        mask_img = Image.fromarray(self.infill_mask, mode='L')
        overlay = Image.new('RGBA', (img_size, img_size), (255, 0, 0, 128))
        composite = Image.composite(overlay, orig, mask_img).convert('RGB')
        composite = composite.resize(
            (self.infill_display_size, self.infill_display_size), Image.NEAREST)
        self.infill_photo = ImageTk.PhotoImage(composite)
        self.infill_canvas.delete("all")
        self.infill_canvas.create_image(0, 0, anchor=tk.NW, image=self.infill_photo)
        self.infill_canvas.image = self.infill_photo

    def _infill_canvas_to_img_coords(self, event):
        img_size = self.global_settings['img_size'].get()
        x = int(event.x * img_size / self.infill_display_size)
        y = int(event.y * img_size / self.infill_display_size)
        return max(0, min(img_size - 1, x)), max(0, min(img_size - 1, y))

    def infill_start_paint(self, event):
        if self.infill_original_img is None:
            return
        self.infill_erase_mode = False
        self.infill_last_xy = self._infill_canvas_to_img_coords(event)

    def infill_paint(self, event):
        self._infill_stroke(event, erase=False)

    def infill_stop_paint(self, event):
        self.infill_last_xy = None

    def infill_start_erase(self, event):
        if self.infill_original_img is None:
            return
        self.infill_erase_mode = True
        self.infill_last_xy = self._infill_canvas_to_img_coords(event)

    def infill_erase(self, event):
        self._infill_stroke(event, erase=True)

    def infill_stop_erase(self, event):
        self.infill_last_xy = None
        self.infill_erase_mode = False

    def _infill_stroke(self, event, erase):
        if self.infill_original_img is None or self.infill_last_xy is None:
            return
        x1, y1 = self.infill_last_xy
        x2, y2 = self._infill_canvas_to_img_coords(event)
        mask_img = Image.fromarray(self.infill_mask, mode='L')
        draw = ImageDraw.Draw(mask_img)
        value = 0 if erase else 255
        draw.line([x1, y1, x2, y2], fill=value,
                  width=int(self.infill_brush_size.get()))
        self.infill_mask = np.array(mask_img, dtype=np.uint8)
        self.infill_last_xy = (x2, y2)
        self.infill_redraw()

    def infill_stop(self):
        self.infill_stop_flag = True
        self.infill_info.config(text="Stopping...")

    def infill_run(self):
        if self.vae_model is None or self.rectified_model is None:
            self.infill_info.config(text="Load VAE + Flow model first!")
            return
        if self.infill_original_img is None:
            self.infill_info.config(text="Load an image first.")
            return
        if self.infill_mask is None or self.infill_mask.max() == 0:
            self.infill_info.config(text="Draw a mask first.")
            return
        if self.infill_in_progress:
            return

        self.infill_in_progress = True
        self.infill_stop_flag = False
        self.infill_btn.config(state=tk.DISABLED)
        self.infill_stop_btn.config(state=tk.NORMAL)
        self.infill_info.config(text="Inpainting...")
        self.root.update()

        threading.Thread(
            target=self._infill_thread,
            args=(self.infill_steps.get(), self.infill_method.get(),
                  self.infill_cfg.get(), float(self.infill_noise_sigma.get()),
                  bool(self.infill_composite_updates.get()),
                  float(self.infill_full_edit_from.get()),
                  bool(self.infill_live_preview.get()),
                  self.infill_prompt.get().strip()),
            daemon=True).start()

    def _infill_thread(self, steps, method, cfg, noise_sigma, composite,
                       full_edit_from, live, prompt):
        try:
            img_size = self.global_settings['img_size'].get()
            color_mode = self.global_settings['color_mode'].get()
            in_channels = 3 if color_mode == 'rgb' else 1
            latent_h = self.vae_settings['vae_latent_h'].get()
            latent_w = self.vae_settings['vae_latent_w'].get()

            # Prepare pixel image
            img = self.infill_original_img
            img = img.convert('RGB') if color_mode == 'rgb' else img.convert('L')
            tf = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,) * in_channels, (0.5,) * in_channels),
            ])
            img_t = tf(img).unsqueeze(0)  # stay CPU for VAE

            # Encode (VAE on CPU)
            with torch.no_grad():
                z1 = self.vae_model.encode(img_t).to(self.rectified_model.device)

            # Mask at image resolution
            mask_np = self.infill_mask.astype(np.float32) / 255.0
            mask_t = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)  # CPU

            # Downsample mask to latent res (max-pool: any masked pixel → masked)
            kh = img_size // latent_h
            kw = img_size // latent_w
            mask_latent = F_nn.max_pool2d(mask_t, kernel_size=(kh, kw),
                                          stride=(kh, kw)).to(self.rectified_model.device)

            # Conditioning
            cond = None
            if (prompt and self.flow_settings['cond_enabled'].get()
                    and self.text_encoder is not None):
                ti = text_to_indices(prompt, self.flow_settings['cond_text_max_len'].get())
                tt = torch.tensor([ti], dtype=torch.long,
                                  device=self.rectified_model.device)
                with torch.no_grad():
                    cond = self.text_encoder(tt)

            use_ema = self.flow_settings['ema_enabled'].get()
            update_every = max(1, steps // 20)

            def show_z(z, step_i):
                with torch.no_grad():
                    decoded = self.vae_model.decode(
                        z.cpu(), target_size=(img_size, img_size))
                
                composited = decoded * mask_t + img_t * (1 - mask_t)
                preview = ((decoded + 1) / 2).clamp(0, 1)
                if preview.shape[1] == 1:
                    pil = transforms.ToPILImage()(preview.squeeze(0).repeat(3, 1, 1))
                else:
                    pil = transforms.ToPILImage()(preview.squeeze(0))
                pil = pil.resize((self.infill_display_size, self.infill_display_size),
                                 Image.NEAREST)
                self.root.after(0, self._infill_show_result, pil, step_i, steps)

            if live:
                with torch.no_grad():
                    for step_i, z in self.rectified_model.inpaint_step_by_step(
                            z1, mask_latent, steps=steps, method=method,
                            cfg_scale=cfg, use_ema=use_ema,
                            cond=cond, noise_sigma=noise_sigma,
                            composite_updates=composite,
                            full_edit_from=full_edit_from):
                        if self.infill_stop_flag:
                            break
                        if step_i % update_every == 0 or step_i == steps:
                            show_z(z, step_i)
            else:
                with torch.no_grad():
                    final_z = self.rectified_model.inpaint(
                        z1, mask_latent, steps=steps, method=method,
                        cfg_scale=cfg, use_ema=use_ema,
                        cond=cond, noise_sigma=noise_sigma,
                        composite_updates=composite,
                        full_edit_from=full_edit_from)
                show_z(final_z, steps)

            self.root.after(0, lambda: self.infill_info.config(
                text="Inpainting complete." if not self.infill_stop_flag
                else "Stopped."))
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda: self.infill_info.config(text=f"Error: {e}"))
        finally:
            self.infill_in_progress = False
            self.root.after(0, lambda: self.infill_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.infill_stop_btn.config(state=tk.DISABLED))

    def _infill_show_result(self, pil, step_i, total_steps):
        self.infill_output_photo = ImageTk.PhotoImage(pil)
        self.infill_output_canvas.delete("all")
        self.infill_output_canvas.create_image(
            self.infill_display_size // 2, self.infill_display_size // 2,
            image=self.infill_output_photo)
        self.infill_output_canvas.image = self.infill_output_photo
        self.infill_info.config(text=f"Step {step_i}/{total_steps}")

    # ---------------- Generation ----------------
    def setup_generation_tab(self, tab):
        main = tk.Frame(tab); main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        tk.Label(main, text="Generate Images (Flow → VAE Decode)",
                 font=("Arial", 14, "bold")).pack(pady=(0, 10))
        pf = tk.LabelFrame(main, text="Text Prompt (empty = unconditional)", padx=5, pady=5)
        pf.pack(fill=tk.X, pady=(0, 10))
        self.gen_prompt = tk.Entry(pf, width=60)
        self.gen_prompt.insert(0, "a cute cat")
        self.gen_prompt.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        cf = tk.Frame(main); cf.pack(pady=5)
        tk.Label(cf, text="Number:").pack(side=tk.LEFT)
        self.gen_count = tk.IntVar(value=16)
        tk.Spinbox(cf, from_=1, to=64, textvariable=self.gen_count, width=5).pack(side=tk.LEFT, padx=5)
        tk.Label(cf, text="ODE Steps:").pack(side=tk.LEFT, padx=(10, 0))
        tk.Spinbox(cf, from_=1, to=200, textvariable=self.ode_steps, width=5).pack(side=tk.LEFT, padx=5)
        tk.Label(cf, text="Method:").pack(side=tk.LEFT, padx=(10, 0))
        ttk.Combobox(cf, textvariable=self.ode_method,
                     values=['euler', 'heun', 'midpoint', 'rk4'],
                     state='readonly', width=10).pack(side=tk.LEFT, padx=5)
        tk.Label(cf, text="CFG scale:").pack(side=tk.LEFT, padx=(10, 0))
        tk.Entry(cf, width=6, textvariable=self.cfg_scale).pack(side=tk.LEFT, padx=5)
        self.progressive_grid = tk.BooleanVar(value=False)
        tk.Checkbutton(cf, text="Progressive", variable=self.progressive_grid).pack(side=tk.LEFT, padx=10)
        tk.Label(cf, text="Interval:").pack(side=tk.LEFT)
        self.prog_interval = tk.IntVar(value=10)
        tk.Spinbox(cf, from_=1, to=50, textvariable=self.prog_interval, width=5).pack(side=tk.LEFT, padx=5)
        self.generate_btn = tk.Button(cf, text="Generate", command=self.generate_samples, bg="lightgreen")
        self.generate_btn.pack(side=tk.LEFT, padx=5)
        self.stop_prog_btn = tk.Button(cf, text="Stop", command=self.stop_progressive,
                                       state=tk.DISABLED, bg="salmon")
        self.stop_prog_btn.pack(side=tk.LEFT, padx=5)

        rf = tk.LabelFrame(main, text="Results", padx=5, pady=5); rf.pack(fill=tk.BOTH, expand=True, pady=5)
        cfr = tk.Frame(rf); cfr.pack(fill=tk.BOTH, expand=True)
        self.gen_canvas = tk.Canvas(cfr, bg='lightgray')
        vs = tk.Scrollbar(cfr, orient=tk.VERTICAL, command=self.gen_canvas.yview)
        hs = tk.Scrollbar(cfr, orient=tk.HORIZONTAL, command=self.gen_canvas.xview)
        self.gen_canvas.configure(yscrollcommand=vs.set, xscrollcommand=hs.set)
        vs.pack(side=tk.RIGHT, fill=tk.Y); hs.pack(side=tk.BOTTOM, fill=tk.X)
        self.gen_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.inner_frame = tk.Frame(self.gen_canvas)
        self.gen_canvas.create_window((0, 0), window=self.inner_frame, anchor='nw')
        self.inner_frame.bind("<Configure>",
                              lambda e: self.gen_canvas.configure(scrollregion=self.gen_canvas.bbox('all')))
        self.gen_info = tk.Label(main, text="", fg="blue"); self.gen_info.pack()

    # ---------------- Logging ----------------
    def log_vae(self, msg): self.message_queue_vae.put(msg)
    def log_rectified(self, msg): self.message_queue_rectified.put(msg)

    def process_messages_vae(self):
        try:
            while True:
                msg = self.message_queue_vae.get_nowait()
                self.vae_log_text.insert(tk.END, f"{time.strftime('%H:%M:%S')} - {msg}\n")
                self.vae_log_text.see(tk.END)
                self.status_label.config(text=msg[:80])
        except queue.Empty:
            pass
        self.root.after(100, self.process_messages_vae)

    def process_messages_rectified(self):
        try:
            while True:
                msg = self.message_queue_rectified.get_nowait()
                self.rectified_log_text.insert(tk.END, f"{time.strftime('%H:%M:%S')} - {msg}\n")
                self.rectified_log_text.see(tk.END)
                self.status_label.config(text=msg[:80])
        except queue.Empty:
            pass
        self.root.after(100, self.process_messages_rectified)

    # ---------------- Dataset ops ----------------
    def add_images(self):
        files = filedialog.askopenfilenames(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.jfif *.webp *.bmp")])
        for f in files:
            if f not in self.image_paths:
                self.image_paths.append(f)
                self.image_listbox.insert(tk.END, os.path.basename(f))
        self.log_rectified(f"Added {len(files)} images. Total: {len(self.image_paths)}")

    def add_folder(self):
        folder = filedialog.askdirectory()
        if not folder:
            return
        n = 0
        for r, _, fs in os.walk(folder):
            for f in fs:
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.jfif', '.webp', '.bmp')):
                    p = os.path.join(r, f)
                    if p not in self.image_paths:
                        self.image_paths.append(p)
                        self.image_listbox.insert(tk.END, os.path.basename(p))
                        n += 1
        self.log_rectified(f"Added {n} images. Total: {len(self.image_paths)}")

    def clear_images(self):
        self.image_paths = []
        self.labels = []
        self.image_listbox.delete(0, tk.END)
        self.csv_status.config(text="No CSV loaded", fg="red")
        self.log_rectified("Cleared all images")

    def load_csv(self):
        fname = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if not fname:
            return
        self.csv_path = fname
        base_map = {os.path.basename(p).lower(): p for p in self.image_paths}
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
        for p in self.image_paths:
            if p in label_map:
                self.labels.append(label_map[p])
            else:
                self.labels.append([os.path.splitext(os.path.basename(p))[0]])
                unknown += 1
        self.csv_status.config(
            text=f"CSV: {len(self.image_paths)-unknown} matched, {unknown} fallback", fg="green")
        self.log_rectified(f"CSV loaded. {len(self.image_paths)-unknown} matched.")

    def use_filenames_as_labels(self):
        self.labels = [[os.path.splitext(os.path.basename(p))[0]] for p in self.image_paths]
        self.csv_status.config(text="Using filenames as labels", fg="blue")
        self.log_rectified("Using filenames as labels.")

    def use_folders_as_labels(self):
        self.labels = [[os.path.basename(os.path.dirname(p)) or 'unknown']
                       for p in self.image_paths]
        self.csv_status.config(text="Using folder names as labels", fg="blue")
        self.log_rectified("Using folder names as labels.")

    def apply_text_encoder_preset(self):
        enc = self.flow_settings['text_encoder_type'].get()
        size = self.flow_settings['text_encoder_size'].get()
        cfg = get_encoder_config(enc, size)
        self.flow_settings['cond_embed_dim'].set(cfg.get('embed_dim', self.flow_settings['cond_embed_dim'].get()))
        if enc == 'BiGRU':
            self.flow_settings['cond_hidden_size'].set(cfg.get('hidden_size', self.flow_settings['cond_hidden_size'].get()))
            self.flow_settings['cond_num_layers'].set(cfg.get('num_layers', self.flow_settings['cond_num_layers'].get()))
        else:
            self.flow_settings['cond_num_heads'].set(cfg.get('num_heads', self.flow_settings['cond_num_heads'].get()))
            self.flow_settings['cond_num_layers'].set(cfg.get('num_layers', self.flow_settings['cond_num_layers'].get()))
            self.flow_settings['cond_ff_dim'].set(cfg.get('ff_dim', self.flow_settings['cond_ff_dim'].get()))
        self.flow_settings['cond_dim'].set(cfg.get('cond_dim', self.flow_settings['cond_dim'].get()))
        self.log_rectified(f"Applied {enc} {size} preset.")

    def reset_all_settings(self):
        for k, v in {
            'vae_size': 'big', 'vae_base_channels': 32, 'vae_latent_channels': 8,
            'vae_latent_h': 4, 'vae_latent_w': 4, 'vae_batch_size': 16,
            'vae_lr': 1e-3, 'vae_num_workers': 0, 'vae_kl_weight': 1e-4,
        }.items():
            self.vae_settings[k].set(v)
        for k, v in {
            'flow_base_channels': 32, 'flow_padding_mode': 'constant',
            'attn_level_0': True, 'attn_level_1': True, 'attn_level_2': True,
            'attn_num_heads': 4, 'attn_dropout': 0.0, 'cond_mode': 'shift',
            'flow_batch_size': 16, 'flow_lr': 2e-4, 'flow_cfg_dropout': 0.1,
            'flow_use_ot': False, 'ot_method': 'sinkhorn',
            'ot_sinkhorn_eps': 0.05, 'ot_sinkhorn_iters': 50,
            'ema_enabled': False, 'ema_decay': 0.999, 'ema_mode': 'standard',
            'preview_enabled': True, 'preview_epoch_freq': 5, 'preview_steps': 20,
            'preview_method': 'heun',
            'cond_enabled': False, 'text_encoder_type': 'BiGRU',
            'text_encoder_size': 'small', 'cond_embed_dim': 64, 'cond_hidden_size': 64,
            'cond_num_layers': 2, 'cond_num_heads': 4, 'cond_ff_dim': 256,
            'cond_dim': 256, 'cond_text_max_len': 128,
        }.items():
            self.flow_settings[k].set(v)
        self.global_settings['img_size'].set(32)
        self.global_settings['color_mode'].set('rgb')
        self._reset_aug(self.vae_aug_settings)
        self._reset_aug(self.flow_aug_settings)
        self._refresh_flow_head_dims()
        self.log_rectified("All settings reset to defaults.")

    # ---------------- VAE methods ----------------
    def initialize_vae(self):
        try:
            in_ch = 3 if self.global_settings['color_mode'].get() == 'rgb' else 1
            self.vae_model = FlexVAE(
                in_channels=in_ch,
                base_channels=self.vae_settings['vae_base_channels'].get(),
                latent_channels=self.vae_settings['vae_latent_channels'].get(),
                latent_h=self.vae_settings['vae_latent_h'].get(),
                latent_w=self.vae_settings['vae_latent_w'].get(),
                size=self.vae_settings['vae_size'].get(),
                img_size=self.global_settings['img_size'].get(),
            ).to('cpu')
            self.vae_optimizer = optim.Adam(self.vae_model.parameters(),
                                            lr=self.vae_settings['vae_lr'].get())
            self.log_vae(f"VAE initialized (size={self.vae_settings['vae_size'].get()}, "
                         f"latent={self.vae_settings['vae_latent_h'].get()}x"
                         f"{self.vae_settings['vae_latent_w'].get()}).")
        except Exception as e:
            self.log_vae(f"Init error: {e}")

    def start_vae_training(self):
        if not self.image_paths:
            self.log_vae("No images!"); return
        if not self.vae_model:
            self.log_vae("Initialize VAE first!"); return
        if self.training_vae:
            self.log_vae("Already training."); return
        try:
            epochs = int(self.vae_epoch_var.get())
        except Exception:
            self.log_vae("Invalid epochs"); return
        self.training_vae = True
        self.vae_start_time = time.time()
        threading.Thread(target=self.train_vae_loop, args=(epochs,), daemon=True).start()
        self.log_vae(f"VAE training started for {epochs} epochs.")

    def train_vae_loop(self, epochs):
        try:
            bs = self.vae_settings['vae_batch_size'].get()
            nw = self.vae_settings['vae_num_workers'].get()
            img_size = self.global_settings['img_size'].get()
            kl_w = self.vae_settings['vae_kl_weight'].get()
            cm = self.global_settings['color_mode'].get()

            aug_dict = {k: v.get() for k, v in self.vae_aug_settings.items()}
            labels = self.labels if self.labels else [['']] * len(self.image_paths)
            ds = ConditionalImageDataset(self.image_paths, labels, img_size, cm, aug_dict,
                                         self.flow_settings['cond_text_max_len'].get())
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
                    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / imgs.size(0)
                    loss = rl + kl_w * kl
                    self.vae_optimizer.zero_grad()
                    loss.backward()
                    self.vae_optimizer.step()
                    tot += loss.item(); n += 1
                avg = tot / max(1, n)
                el = time.time() - self.vae_start_time
                self.log_vae(f"Epoch {ep+1}/{epochs} | Loss: {avg:.6f} | Time: {el:.1f}s")
                if (ep + 1) % 5 == 0:
                    self.show_vae_preview()
            self.training_vae = False
            self.log_vae("VAE training finished.")
        except Exception as e:
            import traceback; traceback.print_exc()
            self.log_vae(f"VAE training error: {e}")
            self.training_vae = False

    def stop_vae_training(self):
        self.training_vae = False
        self.log_vae("VAE training stopped.")

    def show_vae_preview(self):
        if not self.vae_model or not self.image_paths:
            return
        try:
            n = min(16, len(self.image_paths))
            idx = random.sample(range(len(self.image_paths)), n)
            paths = [self.image_paths[i] for i in idx]
            labs = ([self.labels[i] for i in idx]
                    if (self.labels and len(self.labels) == len(self.image_paths))
                    else [['']] * n)
            cm = self.global_settings['color_mode'].get()
            aug_dict = {k: v.get() for k, v in self.vae_aug_settings.items()}
            ds = ConditionalImageDataset(paths, labs,
                                         self.global_settings['img_size'].get(),
                                         cm, aug_dict)
            dl = DataLoader(ds, batch_size=n, shuffle=False)
            imgs, _ = next(iter(dl))
            with torch.no_grad():
                recon, _, _ = self.vae_model(imgs.to('cpu'))
            imgs = ((imgs + 1) / 2).clamp(0, 1).cpu().numpy()
            recon = ((recon + 1) / 2).clamp(0, 1).cpu().numpy()
            thumb = self.thumbnail_size // 2
            grid = Image.new('RGB', (8 * thumb, 4 * thumb))
            for i in range(n):
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
            self.log_vae("No VAE."); return
        fname = filedialog.asksaveasfilename(defaultextension=".pth",
                                             filetypes=[("PyTorch", "*.pth")])
        if fname:
            torch.save({
                'model_state': self.vae_model.state_dict(),
                'optimizer_state': self.vae_optimizer.state_dict(),
                'vae_settings': {k: v.get() for k, v in self.vae_settings.items()},
                'global_settings': {k: v.get() for k, v in self.global_settings.items()},
            }, fname)
            self.log_vae(f"VAE saved to {fname}")

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
            self.log_vae(f"VAE loaded from {fname}")
        except Exception as e:
            self.log_vae(f"Load error: {e}")

    # ---------------- Flow methods ----------------
    def initialize_rectified_model(self):
        if not self.vae_model:
            self.log_rectified("Train/load a VAE first!"); return
        try:
            cond_enabled = self.flow_settings['cond_enabled'].get()
            cond_dim = self.flow_settings['cond_dim'].get() if cond_enabled else 1

            attn_levels = [self.flow_settings['attn_level_0'].get(),
                           self.flow_settings['attn_level_1'].get(),
                           self.flow_settings['attn_level_2'].get()]
            attn_num_heads = int(self.flow_settings['attn_num_heads'].get())
            attn_dropout = float(self.flow_settings['attn_dropout'].get())
            cond_mode = self.flow_settings['cond_mode'].get()
            if cond_mode not in ('shift', 'film', 'adagn', 'cross_attn'):
                cond_mode = 'shift'
            padding_mode = self.flow_settings['flow_padding_mode'].get()

            self.rectified_model = RectifiedFlowLatent(
                latent_channels=self.vae_settings['vae_latent_channels'].get(),
                latent_h=self.vae_settings['vae_latent_h'].get(),
                latent_w=self.vae_settings['vae_latent_w'].get(),
                base_channels=self.flow_settings['flow_base_channels'].get(),
                cond_dim=cond_dim,
                attention_levels=attn_levels,
                attn_num_heads=attn_num_heads,
                attn_dropout=attn_dropout,
                padding_mode=padding_mode,
                cond_mode=cond_mode,
                channel_mult=(1, 2, 4),
            )
            self.rectified_model.use_ot = self.flow_settings['flow_use_ot'].get()
            self.rectified_model.ot_method = self.flow_settings['ot_method'].get()
            self.rectified_model.ot_sinkhorn_eps = float(self.flow_settings['ot_sinkhorn_eps'].get())
            self.rectified_model.ot_sinkhorn_iters = int(self.flow_settings['ot_sinkhorn_iters'].get())
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
                    lr=self.flow_settings['flow_lr'].get())
                self.log_rectified(f"Conditional Flow model initialized ({enc}, "
                                   f"cond_mode={cond_mode}).")
            else:
                self.text_encoder = None
                self.log_rectified(f"Unconditional Flow model initialized "
                                   f"(cond_mode={cond_mode}).")
        except Exception as e:
            import traceback; traceback.print_exc()
            self.log_rectified(f"Init error: {e}")

    def start_rectified_training(self):
        if not self.image_paths:
            self.log_rectified("No images!"); return
        if not self.rectified_model:
            self.log_rectified("Initialize Flow model first!"); return
        if not self.vae_model:
            self.log_rectified("No VAE to encode latents!"); return
        if self.training_rectified:
            self.log_rectified("Already training."); return
        if self.flow_settings['cond_enabled'].get() and not self.labels:
            self.log_rectified("Load labels first."); return
        try:
            epochs = int(self.flow_epoch_var.get())
        except Exception:
            self.log_rectified("Invalid epochs"); return
        self.training_rectified = True
        self.flow_start_time = time.time()
        threading.Thread(target=self.train_rectified_loop, args=(epochs,), daemon=True).start()
        self.log_rectified(f"Flow training started for {epochs} epochs.")

    def train_rectified_loop(self, epochs):
        try:
            bs = self.flow_settings['flow_batch_size'].get()
            nw = self.vae_settings['vae_num_workers'].get()
            img_size = self.global_settings['img_size'].get()
            cm = self.global_settings['color_mode'].get()
            cond_enabled = self.flow_settings['cond_enabled'].get()
            text_max = self.flow_settings['cond_text_max_len'].get()
            cfg_drop = self.flow_settings['flow_cfg_dropout'].get()
            preview_enabled = self.flow_settings['preview_enabled'].get()
            preview_freq = self.flow_settings['preview_epoch_freq'].get()

            aug_dict = {k: v.get() for k, v in self.flow_aug_settings.items()}
            labels = self.labels if self.labels else [['']] * len(self.image_paths)
            ds = ConditionalImageDataset(self.image_paths, labels, img_size, cm, aug_dict, text_max)
            dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=nw,
                            pin_memory=False, persistent_workers=(nw > 0))

            for ep in range(epochs):
                if not self.training_rectified:
                    break
                for pg in self.rectified_model.optimizer.param_groups:
                    pg['lr'] = self.flow_settings['flow_lr'].get()
                self.rectified_model.use_ot = self.flow_settings['flow_use_ot'].get()
                self.rectified_model.ot_sinkhorn_eps = float(
                    self.flow_settings['ot_sinkhorn_eps'].get())
                self.rectified_model.ot_sinkhorn_iters = int(
                    self.flow_settings['ot_sinkhorn_iters'].get())

                tot, n = 0.0, 0
                for imgs, txts in dl:
                    if not self.training_rectified:
                        break
                    # VAE lives on CPU → encode there, then move to flow device
                    with torch.no_grad():
                        z1 = self.vae_model.encode(imgs.to('cpu')).to(
                            self.rectified_model.device)
                    z0 = torch.randn_like(z1)
                    if cond_enabled and self.text_encoder is not None:
                        cond = self.text_encoder(txts.to(self.rectified_model.device))
                    else:
                        cond = None
                    loss = self.rectified_model.train_step(
                        z0, z1, cond=cond, cfg_dropout_prob=cfg_drop)
                    tot += loss; n += 1

                avg = tot / max(1, n)
                el = time.time() - self.flow_start_time
                self.log_rectified(f"Epoch {ep+1}/{epochs} | Loss: {avg:.6f} | Time: {el:.1f}s")
                if preview_enabled and (ep + 1) % preview_freq == 0:
                    self.rectified_preview_with_prompt()

            self.training_rectified = False
            self.log_rectified("Flow training finished.")
        except Exception as e:
            import traceback; traceback.print_exc()
            self.log_rectified(f"Training error: {e}")
            self.training_rectified = False

    def stop_rectified_training(self):
        self.training_rectified = False
        self.log_rectified("Flow training stopped.")

    def rectified_preview_with_prompt(self):
        if not self.rectified_model or not self.vae_model:
            self.log_rectified("Models not loaded for preview"); return
        try:
            prompt = self.test_prompt_entry.get().strip()
            unconditional = (prompt == "")
            cond = None
            cfg_scale = 1.0 if unconditional else self.cfg_scale.get()
            n = 16
            if (not unconditional
                    and self.flow_settings['cond_enabled'].get()
                    and self.text_encoder is not None):
                ti = text_to_indices(prompt, self.flow_settings['cond_text_max_len'].get())
                tt = torch.tensor([ti] * n, dtype=torch.long,
                                  device=self.rectified_model.device)
                with torch.no_grad():
                    cond = self.text_encoder(tt)
                self.log_rectified(f"Preview: conditional on '{prompt}'")
            elif not unconditional:
                self.log_rectified("Preview: unconditional")
                unconditional = True
                cfg_scale = 1.0
            else:
                self.log_rectified("Preview: unconditional")

            use_ema = self.flow_settings['ema_enabled'].get()
            z = self.rectified_model.sample(
                n_samples=n,
                steps=self.flow_settings['preview_steps'].get(),
                method=self.flow_settings['preview_method'].get(),
                cond=cond, cfg_scale=cfg_scale, use_ema=use_ema)
            with torch.no_grad():
                samples = self.vae_model.decode(
                    z.cpu(),
                    target_size=(self.global_settings['img_size'].get(),
                                 self.global_settings['img_size'].get()))
            samples = ((samples + 1) / 2).clamp(0, 1).cpu().numpy()
            thumb = self.thumbnail_size
            grid = Image.new('RGB', (4 * thumb, 4 * thumb))
            for i in range(4):
                for j in range(4):
                    idx = i * 4 + j
                    if idx < len(samples):
                        if samples[idx].shape[0] == 1:
                            im = np.stack([samples[idx][0] * 255] * 3, axis=-1).astype(np.uint8)
                        else:
                            im = (samples[idx].transpose(1, 2, 0) * 255).astype(np.uint8)
                        pil = Image.fromarray(im).resize((thumb, thumb), Image.NEAREST)
                        grid.paste(pil, (j * thumb, i * thumb))
            grid = grid.resize((256, 256), Image.NEAREST)
            self.rectified_preview_photo = ImageTk.PhotoImage(grid)
            self.rectified_preview_canvas.delete("all")
            self.rectified_preview_canvas.create_image(128, 128,
                                                       image=self.rectified_preview_photo)
        except Exception as e:
            self.log_rectified(f"Preview error: {e}")

    def save_rectified_model(self):
        if not self.rectified_model:
            self.log_rectified("No model."); return
        fname = filedialog.asksaveasfilename(defaultextension=".pth",
                                             filetypes=[("PyTorch", "*.pth")])
        if fname:
            if self.rectified_model.use_ema and self.rectified_model.ema_model is not None:
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
                # Top-level copies for robustness
                'attn_levels': self.rectified_model.model.attention_levels,
                'attn_num_heads': self.rectified_model.model.attn_num_heads,
                'attn_dropout': self.rectified_model.model.attn_dropout,
                'cond_mode': self.rectified_model.cond_mode,
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

            # --- migrate old flow_settings (flow_use_attention → attn_level_*) ---
            fs = ckpt.get('flow_settings', {})
            if 'flow_use_attention' in fs and 'attn_level_0' not in fs:
                v = bool(fs['flow_use_attention'])
                fs['attn_level_0'] = v
                fs['attn_level_1'] = v
                fs['attn_level_2'] = v

            for k, v in fs.items():
                if k in self.flow_settings:
                    try:
                        self.flow_settings[k].set(v)
                    except Exception:
                        pass
            for k, v in ckpt.get('vae_latent', {}).items():
                map_to = {'latent_channels': 'vae_latent_channels',
                          'latent_h': 'vae_latent_h',
                          'latent_w': 'vae_latent_w'}[k]
                self.vae_settings[map_to].set(v)

            if 'attn_levels' in ckpt and len(ckpt['attn_levels']) >= 3:
                self.flow_settings['attn_level_0'].set(bool(ckpt['attn_levels'][0]))
                self.flow_settings['attn_level_1'].set(bool(ckpt['attn_levels'][1]))
                self.flow_settings['attn_level_2'].set(bool(ckpt['attn_levels'][2]))
            if 'attn_num_heads' in ckpt:
                self.flow_settings['attn_num_heads'].set(ckpt['attn_num_heads'])
            if 'attn_dropout' in ckpt:
                self.flow_settings['attn_dropout'].set(ckpt['attn_dropout'])
            if 'cond_mode' in ckpt:
                self.flow_settings['cond_mode'].set(ckpt['cond_mode'])

            if not self.vae_model:
                self.log_rectified("Load VAE first, then retry."); return
            self.initialize_rectified_model()
            self.rectified_model.model.load_state_dict(ckpt['model_state'])
            try:
                self.rectified_model.optimizer.load_state_dict(ckpt['optimizer_state'])
            except Exception:
                pass
            if self.rectified_model.use_ema and self.rectified_model.ema_model is not None:
                self.rectified_model.ema_model.load_state_dict(
                    self.rectified_model.model.state_dict())
            if 'text_encoder_state' in ckpt and self.text_encoder is not None:
                self.text_encoder.load_state_dict(ckpt['text_encoder_state'])
            self.log_rectified(f"Flow model loaded from {fname}")
        except Exception as e:
            self.log_rectified(f"Load error: {e}")

    # ---------------- Generation ----------------
    def generate_samples(self):
        if not self.rectified_model or not self.vae_model:
            self.gen_info.config(text="Models not loaded!"); return
        if self.flow_settings['cond_enabled'].get() and self.text_encoder is None:
            self.gen_info.config(text="Text encoder missing!"); return
        n = self.gen_count.get()
        steps = self.ode_steps.get()
        method = self.ode_method.get()
        prompt = self.gen_prompt.get().strip()
        cfg_scale = self.cfg_scale.get()
        use_ema = self.flow_settings['ema_enabled'].get()
        unconditional = (prompt == "")
        cond = None
        eff_cfg = 1.0 if unconditional else cfg_scale
        if (not unconditional
                and self.flow_settings['cond_enabled'].get()
                and self.text_encoder is not None):
            ti = text_to_indices(prompt, self.flow_settings['cond_text_max_len'].get())
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
            threading.Thread(target=self._generate_thread,
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
        threading.Thread(target=self._progressive_thread,
                         args=(n, steps, method, cond, cfg_scale, use_ema),
                         daemon=True).start()

    def _grid_from_latents(self, z, n):
        with torch.no_grad():
            samples = self.vae_model.decode(
                z.cpu(),
                target_size=(self.global_settings['img_size'].get(),
                             self.global_settings['img_size'].get()))
        samples = ((samples + 1) / 2).clamp(0, 1).cpu().numpy()
        thumb = self.thumbnail_size
        gs = int(math.ceil(math.sqrt(n)))
        grid = Image.new('RGB', (gs * thumb, gs * thumb), color=(128, 128, 128))
        for i in range(n):
            if i >= len(samples):
                break
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
            z = self.rectified_model.sample(n_samples=n, steps=steps, method=method,
                                            cond=cond, cfg_scale=cfg_scale,
                                            use_ema=use_ema)
            grid = self._grid_from_latents(z, n)
            self.root.after(0, lambda: self._display_generated(grid))
        except Exception as e:
            self.root.after(0, lambda: self.gen_info.config(text=f"Error: {e}"))
        finally:
            self.root.after(0, lambda: self.generate_btn.config(state=tk.NORMAL))

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
            self.root.after(0, lambda: self.gen_info.config(text=f"Error: {e}"))
        finally:
            self.progressive_active = False
            self.root.after(0, lambda: self.generate_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.stop_prog_btn.config(state=tk.DISABLED))

    def _display_generated(self, grid):
        for w in self.inner_frame.winfo_children():
            w.destroy()
        self.gen_photo = ImageTk.PhotoImage(grid)
        lbl = tk.Label(self.inner_frame, image=self.gen_photo)
        lbl.image = self.gen_photo
        lbl.pack()
        self.inner_frame.update_idletasks()
        self.gen_canvas.configure(scrollregion=self.gen_canvas.bbox('all'))
        self.gen_info.config(text="Generation complete.")

    def _update_progressive(self, grid, step):
        for w in self.inner_frame.winfo_children():
            w.destroy()
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