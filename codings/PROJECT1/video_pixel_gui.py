# video_pixel_gui.py
"""
Pixel-space Video Rectified Flow — Tkinter GUI.

Tab structure follows flow_gui.py (pixel flow):
    Dataset | Train Flow | Settings | Generate | Continue

Augmentations live inside the Settings tab (alongside the model, training,
EMA, preview, and text-conditioning controls). No VAE — the flow model
operates directly on pixel tensors.
"""
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
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

from video_pixel_api import (
    RectifiedFlowPixelVideo, VideoFrameDataset, load_video_frames,
    load_image_as_rgb, load_image_as_grayscale, text_to_indices,
    TextEncoder, TransformerTextEncoder, get_encoder_config,
    VIDEO_EXTS, IMAGE_EXTS, _HAS_CV2, _HAS_IMAGEIO, VALID_COND_MODES,
)
from flow_api import compute_head_dims


# ==================== Augmentation defaults ====================

AUG_DEFAULTS = {
    'flip_horizontal': True, 'rotation': False, 'random_crop': False, 'crop_scale': 0.8,
    'color_jitter': False, 'brightness': 0.2, 'contrast': 0.2, 'saturation': 0.2, 'hue': 0.1,
    'random_perspective': False, 'perspective_distortion': 0.1,
    'elastic_transform': False, 'elastic_alpha': 30, 'elastic_sigma': 3,
    'jpeg_compression': False, 'jpeg_quality_low': 50, 'jpeg_quality_high': 95,
    'stretch_vertical': False, 'stretch_horizontal': False,
    'aug_padding_mode': 'edge',
    # Flow-only context-noise augmentation
    'noise_ctx_enabled': False, 'noise_ctx_prob': 0.1, 'noise_ctx_scale': 0.5,
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


# ==================== Main App ====================

class RectifiedFlowApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pixel Video Rectified Flow")

        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.root.geometry(f"{max(1100, int(sw*0.85))}x{max(750, int(sh*0.85))}")
        self.root.minsize(1000, 700)

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

        self.sources = []
        self.source_labels = []
        self.csv_path = None

        self.rectified_model = None
        self.text_encoder = None
        self.training_rectified = False

        self.message_queue_dataset = queue.Queue()
        self.message_queue_rectified = queue.Queue()

        # ---------------- settings ----------------
        self.settings = {
            # Image
            'img_size': tk.IntVar(value=32),
            'color_mode': tk.StringVar(value='rgb'),
            # Architecture
            'base_channels': tk.IntVar(value=32),
            'padding_mode': tk.StringVar(value='constant'),
            'history': tk.IntVar(value=1),
            # Attention
            'attn_level_0': tk.BooleanVar(value=True),
            'attn_level_1': tk.BooleanVar(value=True),
            'attn_level_2': tk.BooleanVar(value=True),
            'attn_num_heads': tk.IntVar(value=4),
            'attn_dropout': tk.DoubleVar(value=0.0),
            # EMA
            'ema_enabled': tk.BooleanVar(value=False),
            'ema_decay': tk.DoubleVar(value=0.999),
            'ema_mode': tk.StringVar(value='standard'),
            # Training
            'batch_size': tk.IntVar(value=8),
            'lr': tk.DoubleVar(value=2e-4),
            'cfg_dropout_prob': tk.DoubleVar(value=0.1),
            'use_ot': tk.BooleanVar(value=False),
            # Preview during training
            'preview_enabled': tk.BooleanVar(value=True),
            'preview_epoch_freq': tk.IntVar(value=5),
            'preview_steps': tk.IntVar(value=20),
            'preview_num_frames': tk.IntVar(value=8),
            'preview_method': tk.StringVar(value='heun'),
            # Text conditioning
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
            'cond_mode': tk.StringVar(value='shift'),
        }

        self.aug_settings = default_aug_dict()

        self.ode_method = tk.StringVar(value='euler')
        self.ode_steps = tk.IntVar(value=50)
        self.cfg_scale = tk.DoubleVar(value=2.0)
        self.thumbnail_size = 96
        self.frame_stride = tk.IntVar(value=1)
        self.head_dim_label_vars = []

        # ---------------- Generation tab state ----------------
        self.playback_fps = tk.IntVar(value=10)
        self.playback_active = False
        self.playback_idx = 0
        self.playback_after_id = None
        self.gen_frames_pil = []
        self.gen_n_videos = 1
        self.gen_stop_flag = False
        self.progressive_preview = tk.BooleanVar(value=False)
        self.prog_interval = tk.IntVar(value=10)
        self._anim_photo = None

        # ---------------- Continue tab state ----------------
        self.continue_mode = tk.StringVar(value='image')
        self.continue_image_path = None
        self.continue_video_path = None
        self.continue_video_frames = []
        self.continue_frame_index = tk.IntVar(value=-1)
        self.continue_num_frames = tk.IntVar(value=8)
        self.continue_steps = tk.IntVar(value=50)
        self.continue_method = tk.StringVar(value='euler')
        self.continue_cfg = tk.DoubleVar(value=2.0)
        self.continue_prompt = tk.StringVar(value="")
        self.continue_progressive = tk.BooleanVar(value=True)
        self.continue_prog_interval = tk.IntVar(value=10)
        self.continue_stop_flag = False
        self.continue_in_progress = False
        self.continue_context_photo = None
        self.continue_gen_photo = None
        self.continue_playback_active = False
        self.continue_playback_idx = 0
        self.continue_playback_after_id = None
        self.continue_gen_frames_pil = []

        self.setup_gui()
        self.root.after(100, self.process_messages_dataset)
        self.root.after(100, self.process_messages_rectified)

    # ---------------- GUI skeleton ----------------
    def setup_gui(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        tabs = [
            ('Dataset',    self.setup_dataset_tab),
            ('Train Flow', self.setup_train_tab),
            ('Settings',   self.setup_settings_tab),
            ('Generate',   self.setup_generation_tab),
            ('Continue',   self.setup_continue_tab),
        ]
        for name, builder in tabs:
            frame = ttk.Frame(self.notebook)
            self.notebook.add(frame, text=name)
            builder(frame)
        self.status_label = tk.Label(self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

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

    # ---------------- Dataset tab ----------------
    def setup_dataset_tab(self, tab):
        main = tk.Frame(tab); main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        left = tk.Frame(main, width=360); left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left.pack_propagate(False)

        tk.Label(left, text="Dataset Management",
                 font=("Arial", 12, "bold")).pack(pady=(0, 10))

        sf = tk.LabelFrame(left, text="Sources (images, videos, folders-as-video)",
                           padx=5, pady=5)
        sf.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        bf = tk.Frame(sf); bf.pack(fill=tk.X, pady=2)
        tk.Button(bf, text="Add Images", command=self.add_images, width=15).pack(side=tk.LEFT, padx=1)
        tk.Button(bf, text="Add Videos", command=self.add_videos, width=15).pack(side=tk.LEFT, padx=1)
        bf2 = tk.Frame(sf); bf2.pack(fill=tk.X, pady=2)
        tk.Button(bf2, text="Add Folder (as video)", command=self.add_video_folder, width=15).pack(side=tk.LEFT, padx=1)
        tk.Button(bf2, text="Add Folder (recursive)", command=self.add_folder, width=15).pack(side=tk.LEFT, padx=1)
        bf3 = tk.Frame(sf); bf3.pack(fill=tk.X, pady=2)
        tk.Button(bf3, text="Clear All", command=self.clear_all, width=31).pack(side=tk.LEFT)

        stride_f = tk.Frame(sf); stride_f.pack(fill=tk.X, pady=4)
        tk.Label(stride_f, text="Frame stride when adding:").pack(side=tk.LEFT)
        ttk.Spinbox(stride_f, from_=1, to=64, textvariable=self.frame_stride, width=5).pack(side=tk.LEFT, padx=5)

        self.source_listbox = tk.Listbox(sf, height=14)
        self.source_listbox.pack(fill=tk.BOTH, expand=True, pady=2)

        if not (_HAS_CV2 or _HAS_IMAGEIO):
            tk.Label(sf, text="Video support: install 'opencv-python' or 'imageio[ffmpeg]'",
                     fg="orange", wraplength=330, justify=tk.LEFT).pack(fill=tk.X, pady=2)

        cf = tk.LabelFrame(left, text="Labels / Captions", padx=5, pady=5)
        cf.pack(fill=tk.X, pady=5)
        tk.Button(cf, text="Load CSV (name,label)", command=self.load_csv, width=31).pack(pady=1)
        tk.Button(cf, text="Use filenames as labels",
                  command=self.use_filenames_as_labels, width=31).pack(pady=1)
        tk.Button(cf, text="Use folder names as labels",
                  command=self.use_folders_as_labels, width=31).pack(pady=1)
        self.csv_status = tk.Label(cf, text="No CSV loaded", fg="red")
        self.csv_status.pack()

        right = tk.Frame(main); right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        lf = tk.LabelFrame(right, text="Log", padx=5, pady=5); lf.pack(fill=tk.BOTH, expand=True)
        self.dataset_log_text = tk.Text(lf, height=20, font=("Courier", 9))
        self.dataset_log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = tk.Scrollbar(lf, command=self.dataset_log_text.yview); sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.dataset_log_text.config(yscrollcommand=sb.set)

    # ---------------- Train Flow tab ----------------
    def setup_train_tab(self, tab):
        main = tk.Frame(tab); main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        left = tk.Frame(main, width=300); left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left.pack_propagate(False)
        tk.Label(left, text="Rectified Flow Training",
                 font=("Arial", 12, "bold")).pack(pady=(0, 10))

        cond_toggle = tk.LabelFrame(left, text="Conditional Training", padx=5, pady=5)
        cond_toggle.pack(fill=tk.X, pady=5)
        tk.Checkbutton(cond_toggle, text="Enable text conditioning",
                       variable=self.settings['cond_enabled']).pack(anchor='w')

        cfg_frame = tk.LabelFrame(left, text="CFG Training", padx=5, pady=5)
        cfg_frame.pack(fill=tk.X, pady=5)
        f = tk.Frame(cfg_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="CFG dropout:", width=18, anchor='w').pack(side=tk.LEFT)
        tk.Scale(f, from_=0.0, to=0.5, resolution=0.01,
                 variable=self.settings['cfg_dropout_prob'],
                 orient=tk.HORIZONTAL, length=140).pack(side=tk.RIGHT)

        tk.Button(left, text="Initialize Model", command=self.initialize_rectified_model,
                  width=20).pack(pady=5)
        ef = tk.Frame(left); ef.pack(pady=5)
        tk.Label(ef, text="Epochs:").pack(side=tk.LEFT)
        self.epoch_var = tk.StringVar(value="200")
        tk.Entry(ef, textvariable=self.epoch_var, width=8).pack(side=tk.LEFT, padx=5)

        tk.Button(left, text="Start Training", command=self.start_training,
                  width=20, bg="lightgreen").pack(pady=5)
        tk.Button(left, text="Stop Training", command=self.stop_training,
                  width=20, bg="salmon").pack(pady=5)
        tk.Button(left, text="Save Model", command=self.save_model, width=20).pack(pady=5)
        tk.Button(left, text="Load Model", command=self.load_model, width=20).pack(pady=5)

        right = tk.Frame(main); right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        pf = tk.LabelFrame(right, text="Preview (decoded frames)", padx=5, pady=5)
        pf.pack(fill=tk.X, pady=(0, 5))
        self.rectified_preview_canvas = tk.Canvas(pf, bg='gray', width=512, height=192)
        self.rectified_preview_canvas.pack()

        prf = tk.LabelFrame(right, text="Test prompt (empty = unconditional)",
                            padx=5, pady=5)
        prf.pack(fill=tk.X, pady=(0, 5))
        self.test_prompt_entry = tk.Entry(prf)
        self.test_prompt_entry.insert(0, "a cute cat")
        self.test_prompt_entry.pack(fill=tk.X, pady=2)
        tk.Button(prf, text="Generate Preview",
                  command=self.preview_with_prompt).pack(pady=2)

        lf = tk.LabelFrame(right, text="Log", padx=5, pady=5); lf.pack(fill=tk.BOTH, expand=True)
        self.rectified_log_text = tk.Text(lf, height=15, font=("Courier", 9))
        self.rectified_log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = tk.Scrollbar(lf, command=self.rectified_log_text.yview); sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.rectified_log_text.config(yscrollcommand=sb.set)

    # ---------------- Settings tab ----------------
    def setup_settings_tab(self, tab):
        inner = self._make_scrollable(tab)
        wrap = tk.Frame(inner); wrap.pack(fill=tk.X, padx=20, pady=20)
        tk.Label(wrap, text="Model & Training Settings",
                 font=("Arial", 14, "bold")).pack(pady=(0, 15))

        # --- Image ---
        img_f = tk.LabelFrame(wrap, text="Image", padx=10, pady=10)
        img_f.pack(fill=tk.X, pady=5)
        f = tk.Frame(img_f); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Color mode:", width=22, anchor='w').pack(side=tk.LEFT)
        ttk.Combobox(f, textvariable=self.settings['color_mode'],
                     values=['rgb', 'grayscale'], state='readonly', width=12).pack(side=tk.RIGHT)
        f = tk.Frame(img_f); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Image size:", width=22, anchor='w').pack(side=tk.LEFT)
        ttk.Spinbox(f, from_=16, to=128, textvariable=self.settings['img_size'],
                    width=8).pack(side=tk.RIGHT)

        # --- Architecture ---
        arch_f = tk.LabelFrame(wrap, text="Architecture", padx=10, pady=10)
        arch_f.pack(fill=tk.X, pady=5)
        f = tk.Frame(arch_f); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="UNet base channels:", width=22, anchor='w').pack(side=tk.LEFT)
        ttk.Spinbox(f, from_=4, to=256, textvariable=self.settings['base_channels'],
                    width=8, command=self._refresh_head_dims).pack(side=tk.RIGHT)
        self.settings['base_channels'].trace_add('write', lambda *_: self._refresh_head_dims())
        f = tk.Frame(arch_f); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Conv padding mode:", width=22, anchor='w').pack(side=tk.LEFT)
        ttk.Combobox(f, textvariable=self.settings['padding_mode'],
                     values=['constant', 'reflect', 'replicate', 'circular'],
                     state='readonly', width=12).pack(side=tk.RIGHT)
        f = tk.Frame(arch_f); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="History frames (1–10):", width=22, anchor='w').pack(side=tk.LEFT)
        ttk.Spinbox(f, from_=1, to=10, textvariable=self.settings['history'],
                    width=8).pack(side=tk.RIGHT)
        tk.Label(arch_f,
                 text="Each history frame adds C=3 channels to the UNet input.",
                 fg='gray', font=("Arial", 8, "italic")).pack(anchor='w')

        # --- Attention ---
        attn_f = tk.LabelFrame(wrap, text="Attention (per resolution)", padx=10, pady=10)
        attn_f.pack(fill=tk.X, pady=5)
        f = tk.Frame(attn_f); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Enable attention:", width=22, anchor='w').pack(side=tk.LEFT)
        tk.Checkbutton(f, text="L0", variable=self.settings['attn_level_0'],
                       command=self._refresh_head_dims).pack(side=tk.LEFT, padx=3)
        tk.Checkbutton(f, text="L1", variable=self.settings['attn_level_1'],
                       command=self._refresh_head_dims).pack(side=tk.LEFT, padx=3)
        tk.Checkbutton(f, text="L2", variable=self.settings['attn_level_2'],
                       command=self._refresh_head_dims).pack(side=tk.LEFT, padx=3)
        f = tk.Frame(attn_f); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Number of heads:", width=22, anchor='w').pack(side=tk.LEFT)
        ttk.Spinbox(f, from_=1, to=32, textvariable=self.settings['attn_num_heads'],
                    width=8, command=self._refresh_head_dims).pack(side=tk.RIGHT)
        self.settings['attn_num_heads'].trace_add('write', lambda *_: self._refresh_head_dims())
        f = tk.Frame(attn_f); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Attention dropout:", width=22, anchor='w').pack(side=tk.LEFT)
        tk.Entry(f, textvariable=self.settings['attn_dropout'], width=8).pack(side=tk.RIGHT)
        hd_f = tk.LabelFrame(attn_f, text="Auto head dim", padx=6, pady=6)
        hd_f.pack(fill=tk.X, pady=(8, 2))
        self.head_dim_label_vars = [tk.StringVar(value="L0: —"),
                                    tk.StringVar(value="L1: —"),
                                    tk.StringVar(value="L2: —")]
        for v in self.head_dim_label_vars:
            tk.Label(hd_f, textvariable=v, anchor='w',
                     font=("Courier", 9)).pack(fill=tk.X)

        # --- EMA ---
        ema_f = tk.LabelFrame(wrap, text="EMA", padx=10, pady=10)
        ema_f.pack(fill=tk.X, pady=5)
        f = tk.Frame(ema_f); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Enable EMA:", width=22, anchor='w').pack(side=tk.LEFT)
        tk.Checkbutton(f, variable=self.settings['ema_enabled']).pack(side=tk.RIGHT)
        f = tk.Frame(ema_f); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="EMA decay:", width=22, anchor='w').pack(side=tk.LEFT)
        tk.Entry(f, textvariable=self.settings['ema_decay'], width=8).pack(side=tk.RIGHT)
        f = tk.Frame(ema_f); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="EMA mode:", width=22, anchor='w').pack(side=tk.LEFT)
        ttk.Combobox(f, textvariable=self.settings['ema_mode'],
                     values=['standard', 'lookahead'], state='readonly',
                     width=12).pack(side=tk.RIGHT)

        # --- Training ---
        tr_f = tk.LabelFrame(wrap, text="Training", padx=10, pady=10)
        tr_f.pack(fill=tk.X, pady=5)
        for label, key in [("Batch size:", 'batch_size'),
                           ("Learning rate:", 'lr')]:
            f = tk.Frame(tr_f); f.pack(fill=tk.X, pady=2)
            tk.Label(f, text=label, width=22, anchor='w').pack(side=tk.LEFT)
            tk.Entry(f, textvariable=self.settings[key], width=8).pack(side=tk.RIGHT)
        f = tk.Frame(tr_f); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Use minibatch OT matching:", width=22, anchor='w').pack(side=tk.LEFT)
        tk.Checkbutton(f, variable=self.settings['use_ot']).pack(side=tk.RIGHT)

        # --- Preview During Training ---
        pv_f = tk.LabelFrame(wrap, text="Preview During Training", padx=10, pady=10)
        pv_f.pack(fill=tk.X, pady=5)
        for label, key, kind in [("Enable preview:", 'preview_enabled', 'bool'),
                                 ("Every N epochs:", 'preview_epoch_freq', 'int'),
                                 ("Preview steps:", 'preview_steps', 'int'),
                                 ("Preview frames:", 'preview_num_frames', 'int')]:
            f = tk.Frame(pv_f); f.pack(fill=tk.X, pady=2)
            tk.Label(f, text=label, width=22, anchor='w').pack(side=tk.LEFT)
            if kind == 'bool':
                tk.Checkbutton(f, variable=self.settings[key]).pack(side=tk.RIGHT)
            else:
                ttk.Spinbox(f, from_=1, to=200, textvariable=self.settings[key],
                            width=8).pack(side=tk.RIGHT)
        f = tk.Frame(pv_f); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Preview ODE method:", width=22, anchor='w').pack(side=tk.LEFT)
        ttk.Combobox(f, textvariable=self.settings['preview_method'],
                     values=['euler', 'heun', 'midpoint', 'rk4'],
                     state='readonly', width=12).pack(side=tk.RIGHT)

        # --- Text Conditioning ---
        cond_f = tk.LabelFrame(wrap, text="Text Conditioning", padx=10, pady=10)
        cond_f.pack(fill=tk.X, pady=5)
        f = tk.Frame(cond_f); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Encoder type:", width=22, anchor='w').pack(side=tk.LEFT)
        ttk.Combobox(f, textvariable=self.settings['text_encoder_type'],
                     values=['BiGRU', 'BiTransformer'], state='readonly',
                     width=12).pack(side=tk.RIGHT)
        f = tk.Frame(cond_f); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Encoder size:", width=22, anchor='w').pack(side=tk.LEFT)
        ttk.Combobox(f, textvariable=self.settings['text_encoder_size'],
                     values=['tiny', 'small', 'medium', 'large'],
                     state='readonly', width=12).pack(side=tk.RIGHT)
        f = tk.Frame(cond_f); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Cond mode:", width=22, anchor='w').pack(side=tk.LEFT)
        ttk.Combobox(f, textvariable=self.settings['cond_mode'],
                     values=list(VALID_COND_MODES), state='readonly',
                     width=14).pack(side=tk.RIGHT)
        tk.Label(cond_f,
                 text="shift = additive · film = scale+shift · "
                      "adagn = adaptive GroupNorm · cross_attn = cross-attention",
                 fg='gray', font=("Arial", 8, "italic")).pack(anchor='w')
        tk.Button(cond_f, text="Apply preset to dims",
                  command=self.apply_text_encoder_preset).pack(pady=3)
        for label, key, low, high in [
            ("Embed dim:", 'cond_embed_dim', 8, 512),
            ("Hidden size (GRU):", 'cond_hidden_size', 8, 1024),
            ("Num layers:", 'cond_num_layers', 1, 6),
            ("Num heads (TF):", 'cond_num_heads', 1, 16),
            ("FF dim (TF):", 'cond_ff_dim', 32, 2048),
            ("Cond dim:", 'cond_dim', 32, 1024),
            ("Max text len:", 'cond_text_max_len', 32, 512),
        ]:
            f = tk.Frame(cond_f); f.pack(fill=tk.X, pady=2)
            tk.Label(f, text=label, width=22, anchor='w').pack(side=tk.LEFT)
            ttk.Spinbox(f, from_=low, to=high, textvariable=self.settings[key],
                        width=8).pack(side=tk.RIGHT)

        # --- Augmentations (with padding mode) ---
        aug_f = tk.LabelFrame(wrap, text="Data Augmentations", padx=10, pady=10)
        aug_f.pack(fill=tk.X, pady=5)
        f = tk.Frame(aug_f); f.pack(fill=tk.X, pady=(0, 6))
        tk.Label(f, text="Aug padding mode:", width=22, anchor='w').pack(side=tk.LEFT)
        ttk.Combobox(f, textvariable=self.aug_settings['aug_padding_mode'],
                     values=['edge', 'reflect', 'symmetric', 'constant'],
                     state='readonly', width=12).pack(side=tk.RIGHT)
        tk.Label(aug_f,
                 text="'edge' = replicate border pixel (recommended for video).",
                 fg='gray', font=("Arial", 8, "italic")).pack(anchor='w')

        f1 = tk.Frame(aug_f); f1.pack(fill=tk.X, pady=2)
        tk.Checkbutton(f1, text="Horizontal Flip",
                       variable=self.aug_settings['flip_horizontal']).pack(side=tk.LEFT, padx=5)
        tk.Checkbutton(f1, text="Rotation (±30°)",
                       variable=self.aug_settings['rotation']).pack(side=tk.LEFT, padx=5)
        tk.Checkbutton(f1, text="Random Crop",
                       variable=self.aug_settings['random_crop']).pack(side=tk.LEFT, padx=5)
        tk.Label(f1, text="scale:").pack(side=tk.LEFT, padx=(5, 0))
        tk.Entry(f1, textvariable=self.aug_settings['crop_scale'], width=5).pack(side=tk.LEFT)
        f2 = tk.Frame(aug_f); f2.pack(fill=tk.X, pady=2)
        tk.Checkbutton(f2, text="Color Jitter",
                       variable=self.aug_settings['color_jitter']).pack(side=tk.LEFT, padx=5)
        tk.Label(f2, text="b/c/s/h:").pack(side=tk.LEFT)
        for k in ['brightness', 'contrast', 'saturation', 'hue']:
            tk.Entry(f2, textvariable=self.aug_settings[k], width=4).pack(side=tk.LEFT)
        f3 = tk.Frame(aug_f); f3.pack(fill=tk.X, pady=2)
        tk.Checkbutton(f3, text="Random Perspective",
                       variable=self.aug_settings['random_perspective']).pack(side=tk.LEFT, padx=5)
        tk.Label(f3, text="dist:").pack(side=tk.LEFT)
        tk.Entry(f3, textvariable=self.aug_settings['perspective_distortion'], width=5).pack(side=tk.LEFT)
        tk.Checkbutton(f3, text="Elastic",
                       variable=self.aug_settings['elastic_transform']).pack(side=tk.LEFT, padx=10)
        tk.Label(f3, text="α/σ:").pack(side=tk.LEFT)
        tk.Entry(f3, textvariable=self.aug_settings['elastic_alpha'], width=4).pack(side=tk.LEFT)
        tk.Entry(f3, textvariable=self.aug_settings['elastic_sigma'], width=4).pack(side=tk.LEFT)
        f4 = tk.Frame(aug_f); f4.pack(fill=tk.X, pady=2)
        tk.Checkbutton(f4, text="JPEG Compression",
                       variable=self.aug_settings['jpeg_compression']).pack(side=tk.LEFT, padx=5)
        tk.Label(f4, text="q low/high:").pack(side=tk.LEFT)
        tk.Entry(f4, textvariable=self.aug_settings['jpeg_quality_low'], width=4).pack(side=tk.LEFT)
        tk.Entry(f4, textvariable=self.aug_settings['jpeg_quality_high'], width=4).pack(side=tk.LEFT)
        tk.Checkbutton(f4, text="Stretch V",
                       variable=self.aug_settings['stretch_vertical']).pack(side=tk.LEFT, padx=10)
        tk.Checkbutton(f4, text="Stretch H",
                       variable=self.aug_settings['stretch_horizontal']).pack(side=tk.LEFT, padx=5)

        f5 = tk.Frame(aug_f); f5.pack(fill=tk.X, pady=(8, 2))
        tk.Label(f5, text="— Context-noise augmentation (latent-space style) —",
                 fg="blue").pack(anchor='w')
        f6 = tk.Frame(aug_f); f6.pack(fill=tk.X, pady=2)
        tk.Checkbutton(f6, text="Noise-Ctx",
                       variable=self.aug_settings['noise_ctx_enabled']).pack(side=tk.LEFT, padx=5)
        tk.Label(f6, text="prob:").pack(side=tk.LEFT, padx=(8, 2))
        tk.Entry(f6, textvariable=self.aug_settings['noise_ctx_prob'], width=5).pack(side=tk.LEFT)
        tk.Label(f6, text="scale:").pack(side=tk.LEFT, padx=(6, 2))
        tk.Entry(f6, textvariable=self.aug_settings['noise_ctx_scale'], width=5).pack(side=tk.LEFT)

        r = tk.Frame(wrap); r.pack(fill=tk.X, pady=10)
        tk.Button(r, text="Reset All Hyperparameters to Default",
                  command=self.reset_all_settings,
                  bg="orange", fg="white", font=("Arial", 10, "bold")).pack(pady=5)

        sys_f = tk.LabelFrame(wrap, text="System", padx=10, pady=10)
        sys_f.pack(fill=tk.X, pady=5)
        tk.Label(sys_f, text=f"CPU cores: {multiprocessing.cpu_count()}").pack(anchor='w')
        tk.Label(sys_f, text=f"PyTorch threads: {torch.get_num_threads()}").pack(anchor='w')
        tk.Label(sys_f, text=f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}").pack(anchor='w')

        self._refresh_head_dims()

    def _refresh_head_dims(self):
        try:
            base = int(self.settings['base_channels'].get())
            nh = int(self.settings['attn_num_heads'].get())
        except Exception:
            return
        dims = compute_head_dims(base, (1, 2, 4), nh)
        flags = [self.settings['attn_level_0'].get(),
                 self.settings['attn_level_1'].get(),
                 self.settings['attn_level_2'].get()]
        for i, (ch, eff_nh, hd) in enumerate(dims):
            if i >= len(self.head_dim_label_vars):
                break
            tag = "ON " if flags[i] else "off"
            self.head_dim_label_vars[i].set(
                f"L{i} [{tag}]  channels={ch:4d}  heads={eff_nh:2d}  head_dim={hd:4d}")

    def apply_text_encoder_preset(self):
        enc = self.settings['text_encoder_type'].get()
        size = self.settings['text_encoder_size'].get()
        cfg = get_encoder_config(enc, size)
        self.settings['cond_embed_dim'].set(cfg.get('embed_dim', self.settings['cond_embed_dim'].get()))
        if enc == 'BiGRU':
            self.settings['cond_hidden_size'].set(cfg.get('hidden_size', self.settings['cond_hidden_size'].get()))
            self.settings['cond_num_layers'].set(cfg.get('num_layers', self.settings['cond_num_layers'].get()))
        else:
            self.settings['cond_num_heads'].set(cfg.get('num_heads', self.settings['cond_num_heads'].get()))
            self.settings['cond_num_layers'].set(cfg.get('num_layers', self.settings['cond_num_layers'].get()))
            self.settings['cond_ff_dim'].set(cfg.get('ff_dim', self.settings['cond_ff_dim'].get()))
        self.settings['cond_dim'].set(cfg.get('cond_dim', self.settings['cond_dim'].get()))
        self.log_rectified(f"Applied {enc} {size} preset.")

    def reset_all_settings(self):
        defaults = {
            'img_size': 32, 'color_mode': 'rgb', 'base_channels': 32,
            'padding_mode': 'constant', 'history': 1,
            'attn_level_0': True, 'attn_level_1': True, 'attn_level_2': True,
            'attn_num_heads': 4, 'attn_dropout': 0.0,
            'ema_enabled': False, 'ema_decay': 0.999, 'ema_mode': 'standard',
            'batch_size': 8, 'lr': 2e-4, 'cfg_dropout_prob': 0.1, 'use_ot': False,
            'preview_enabled': True, 'preview_epoch_freq': 5, 'preview_steps': 20,
            'preview_num_frames': 8, 'preview_method': 'heun',
            'cond_enabled': False, 'text_encoder_type': 'BiGRU',
            'text_encoder_size': 'small', 'cond_embed_dim': 64,
            'cond_hidden_size': 64, 'cond_num_layers': 2, 'cond_num_heads': 4,
            'cond_ff_dim': 256, 'cond_dim': 256, 'cond_text_max_len': 128,
            'cond_mode': 'shift',
        }
        for k, v in defaults.items():
            self.settings[k].set(v)
        for k, v in AUG_DEFAULTS.items():
            self.aug_settings[k].set(v)
        self._refresh_head_dims()
        self.log_rectified("All settings reset to defaults.")

    # ---------------- Generate tab ----------------
    def setup_generation_tab(self, tab):
        main = tk.Frame(tab); main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        tk.Label(main, text="Generate Video (pixel flow → frames)",
                 font=("Arial", 14, "bold")).pack(pady=(0, 10))

        pf = tk.LabelFrame(main, text="Text Prompt (empty = unconditional)",
                           padx=5, pady=5)
        pf.pack(fill=tk.X, pady=(0, 10))
        self.gen_prompt = tk.Entry(pf, width=60)
        self.gen_prompt.insert(0, "a cute cat")
        self.gen_prompt.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        cf = tk.Frame(main); cf.pack(pady=5)
        tk.Label(cf, text="# videos:").pack(side=tk.LEFT)
        self.gen_count = tk.IntVar(value=1)
        tk.Spinbox(cf, from_=1, to=16, textvariable=self.gen_count, width=4).pack(side=tk.LEFT, padx=5)
        tk.Label(cf, text="Frames/video:").pack(side=tk.LEFT, padx=(8, 0))
        self.gen_frames = tk.IntVar(value=16)
        tk.Spinbox(cf, from_=1, to=128, textvariable=self.gen_frames, width=4).pack(side=tk.LEFT, padx=5)
        tk.Label(cf, text="ODE Steps:").pack(side=tk.LEFT, padx=(8, 0))
        tk.Spinbox(cf, from_=1, to=200, textvariable=self.ode_steps, width=4).pack(side=tk.LEFT, padx=5)
        tk.Label(cf, text="Method:").pack(side=tk.LEFT, padx=(8, 0))
        ttk.Combobox(cf, textvariable=self.ode_method,
                     values=['euler', 'heun', 'midpoint', 'rk4'],
                     state='readonly', width=8).pack(side=tk.LEFT, padx=5)
        tk.Label(cf, text="CFG:").pack(side=tk.LEFT, padx=(8, 0))
        tk.Entry(cf, width=5, textvariable=self.cfg_scale).pack(side=tk.LEFT, padx=5)

        cf2 = tk.Frame(main); cf2.pack(pady=5)
        self.generate_btn = tk.Button(cf2, text="Generate",
                                      command=self.generate_samples, bg="lightgreen")
        self.generate_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn = tk.Button(cf2, text="Stop", command=self._stop_generation,
                                  state=tk.DISABLED, bg="salmon")
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        self.play_btn = tk.Button(cf2, text="▶ Play", command=self._start_playback,
                                  state=tk.DISABLED)
        self.play_btn.pack(side=tk.LEFT, padx=5)
        self.pause_btn = tk.Button(cf2, text="⏸ Pause", command=self._pause_playback,
                                   state=tk.DISABLED)
        self.pause_btn.pack(side=tk.LEFT, padx=5)
        tk.Label(cf2, text="FPS:").pack(side=tk.LEFT, padx=(10, 2))
        tk.Spinbox(cf2, from_=1, to=60, textvariable=self.playback_fps, width=4).pack(side=tk.LEFT)

        cf3 = tk.Frame(main); cf3.pack(pady=5)
        tk.Checkbutton(cf3, text="Progressive preview (live)",
                       variable=self.progressive_preview).pack(side=tk.LEFT)
        tk.Label(cf3, text="Update every N ODE steps:").pack(side=tk.LEFT, padx=(10, 0))
        tk.Spinbox(cf3, from_=1, to=100, textvariable=self.prog_interval,
                   width=5).pack(side=tk.LEFT, padx=5)

        rf = tk.LabelFrame(main, text="Results", padx=5, pady=5)
        rf.pack(fill=tk.BOTH, expand=True, pady=5)
        cfr = tk.Frame(rf); cfr.pack(fill=tk.BOTH, expand=True)
        self.gen_canvas = tk.Canvas(cfr, bg='lightgray')
        vs = tk.Scrollbar(cfr, orient=tk.VERTICAL, command=self.gen_canvas.yview)
        hs = tk.Scrollbar(cfr, orient=tk.HORIZONTAL, command=self.gen_canvas.xview)
        self.gen_canvas.configure(yscrollcommand=vs.set, xscrollcommand=hs.set)
        vs.pack(side=tk.RIGHT, fill=tk.Y); hs.pack(side=tk.BOTTOM, fill=tk.X)
        self.gen_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.gen_info = tk.Label(main, text="", fg="blue"); self.gen_info.pack()

    # ---------------- Continue tab ----------------
    def setup_continue_tab(self, tab):
        main = tk.Frame(tab); main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        tk.Label(main, text="Continue Generation — from Image or Video frame (pixel)",
                 font=("Arial", 12, "bold")).pack(pady=(0, 6))

        src = tk.LabelFrame(main, text="Source", padx=8, pady=6)
        src.pack(fill=tk.X, pady=4)

        mode_row = tk.Frame(src); mode_row.pack(fill=tk.X, pady=2)
        tk.Label(mode_row, text="Mode:", width=10, anchor='w').pack(side=tk.LEFT)
        tk.Radiobutton(mode_row, text="From Image",
                       variable=self.continue_mode, value='image',
                       command=self._on_continue_mode_change).pack(side=tk.LEFT)
        tk.Radiobutton(mode_row, text="From Video",
                       variable=self.continue_mode, value='video',
                       command=self._on_continue_mode_change).pack(side=tk.LEFT, padx=(10, 0))

        load_row = tk.Frame(src); load_row.pack(fill=tk.X, pady=2)
        tk.Button(load_row, text="Load Source",
                  command=self.continue_load_source).pack(side=tk.LEFT)
        tk.Button(load_row, text="Clear",
                  command=self.continue_clear_source).pack(side=tk.LEFT, padx=4)
        self.continue_path_label = tk.Label(load_row, text="(no source loaded)",
                                            fg="gray", anchor='w')
        self.continue_path_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)

        idx_row = tk.Frame(src); idx_row.pack(fill=tk.X, pady=2)
        tk.Label(idx_row, text="Start frame index:", width=16, anchor='w').pack(side=tk.LEFT)
        self.continue_idx_spin = tk.Spinbox(
            idx_row, from_=-9999, to=9999,
            textvariable=self.continue_frame_index, width=8,
            command=self._continue_refresh_context_preview)
        self.continue_idx_spin.pack(side=tk.LEFT)
        tk.Label(idx_row, text=" (-1 = last, negative = from end)",
                 fg='gray', font=("Arial", 8, "italic")).pack(side=tk.LEFT, padx=6)
        tk.Button(idx_row, text="Refresh Preview",
                  command=self._continue_refresh_context_preview).pack(side=tk.RIGHT)

        prev = tk.LabelFrame(main, text="Preview", padx=8, pady=6)
        prev.pack(fill=tk.X, pady=4)
        prev_row = tk.Frame(prev); prev_row.pack(fill=tk.X)

        left = tk.Frame(prev_row); left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        tk.Label(left, text="Context (history — direct pixel input)").pack(anchor='w')
        self.continue_context_canvas = tk.Canvas(left, bg='gray', width=384, height=100)
        self.continue_context_canvas.pack(anchor='w')

        right = tk.Frame(prev_row); right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(4, 0))
        tk.Label(right, text="Generated (continuation)").pack(anchor='w')
        self.continue_gen_canvas = tk.Canvas(right, bg='gray', width=768, height=100)
        self.continue_gen_canvas.pack(anchor='w')

        self.continue_info = tk.Label(main,
                                      text="Load a source image or video to begin.",
                                      fg="blue")
        self.continue_info.pack(pady=2)

        gen = tk.LabelFrame(main, text="Generation", padx=8, pady=6)
        gen.pack(fill=tk.X, pady=4)

        prompt_row = tk.Frame(gen); prompt_row.pack(fill=tk.X, pady=2)
        tk.Label(prompt_row, text="Prompt:").pack(side=tk.LEFT, padx=(2, 4))
        tk.Entry(prompt_row, textvariable=self.continue_prompt, width=50).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        tk.Label(prompt_row, text="(optional; used only if conditioning enabled)",
                 fg='gray', font=("Arial", 8, "italic")).pack(side=tk.LEFT, padx=6)

        ctrl_row = tk.Frame(gen); ctrl_row.pack(fill=tk.X, pady=2)
        tk.Label(ctrl_row, text="Frames:").pack(side=tk.LEFT, padx=(2, 0))
        tk.Spinbox(ctrl_row, from_=1, to=64, textvariable=self.continue_num_frames,
                   width=4).pack(side=tk.LEFT, padx=2)
        tk.Label(ctrl_row, text="Steps:").pack(side=tk.LEFT, padx=(8, 0))
        tk.Spinbox(ctrl_row, from_=1, to=200, textvariable=self.continue_steps,
                   width=4).pack(side=tk.LEFT, padx=2)
        tk.Label(ctrl_row, text="Method:").pack(side=tk.LEFT, padx=(8, 0))
        ttk.Combobox(ctrl_row, textvariable=self.continue_method,
                     values=['euler', 'heun', 'midpoint', 'rk4'],
                     state='readonly', width=8).pack(side=tk.LEFT, padx=2)
        tk.Label(ctrl_row, text="CFG:").pack(side=tk.LEFT, padx=(8, 0))
        tk.Entry(ctrl_row, width=5, textvariable=self.continue_cfg).pack(side=tk.LEFT, padx=2)

        prog_row = tk.Frame(gen); prog_row.pack(fill=tk.X, pady=2)
        tk.Checkbutton(prog_row, text="Progressive preview",
                       variable=self.continue_progressive).pack(side=tk.LEFT)
        tk.Label(prog_row, text="Update every N steps:").pack(side=tk.LEFT, padx=(8, 0))
        tk.Spinbox(prog_row, from_=1, to=100, textvariable=self.continue_prog_interval,
                   width=4).pack(side=tk.LEFT, padx=2)
        self.continue_generate_btn = tk.Button(prog_row, text="Generate",
                                                command=self.continue_generate,
                                                bg="lightgreen")
        self.continue_generate_btn.pack(side=tk.LEFT, padx=(15, 4))
        self.continue_stop_btn = tk.Button(prog_row, text="Stop",
                                            command=self.continue_stop,
                                            bg="salmon", state=tk.DISABLED)
        self.continue_stop_btn.pack(side=tk.LEFT, padx=2)

        play_row = tk.Frame(gen); play_row.pack(fill=tk.X, pady=2)
        tk.Label(play_row, text="Playback:", fg="gray").pack(side=tk.LEFT)
        self.continue_play_btn = tk.Button(
            play_row, text="▶ Play",
            command=self._continue_start_playback, state=tk.DISABLED)
        self.continue_play_btn.pack(side=tk.LEFT, padx=(4, 2))
        self.continue_pause_btn = tk.Button(
            play_row, text="⏸ Pause",
            command=self._continue_pause_playback, state=tk.DISABLED)
        self.continue_pause_btn.pack(side=tk.LEFT, padx=2)
        self.continue_show_all_btn = tk.Button(
            play_row, text="Show All Frames",
            command=self._continue_show_strip, state=tk.DISABLED)
        self.continue_show_all_btn.pack(side=tk.LEFT, padx=2)
        tk.Label(play_row, text="FPS:").pack(side=tk.LEFT, padx=(10, 2))
        tk.Spinbox(play_row, from_=1, to=60, textvariable=self.playback_fps,
                   width=4).pack(side=tk.LEFT)

        self._on_continue_mode_change()

    def _on_continue_mode_change(self):
        if self.continue_mode.get() == 'video':
            self.continue_idx_spin.config(state='normal')
        else:
            self.continue_idx_spin.config(state='disabled')

    def continue_clear_source(self):
        self.continue_image_path = None
        self.continue_video_path = None
        self.continue_video_frames = []
        self.continue_path_label.config(text="(no source loaded)", fg="gray")
        self.continue_context_canvas.delete("all")
        self.continue_gen_canvas.delete("all")
        self.continue_info.config(text="Source cleared.")

    def continue_load_source(self):
        if self.continue_mode.get() == 'image':
            path = filedialog.askopenfilename(
                filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.jfif *.webp")])
            if not path:
                return
            self.continue_image_path = path
            self.continue_video_path = None
            self.continue_video_frames = []
            self.continue_path_label.config(text=os.path.basename(path), fg="black")
        else:
            path = filedialog.askopenfilename(
                filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv *.webm *.gif *.m4v *.wmv *.flv"),
                           ("All files", "*.*")])
            if not path:
                return
            if not (_HAS_CV2 or _HAS_IMAGEIO):
                messagebox.showerror("Video support missing",
                                     "Install 'opencv-python' or 'imageio[ffmpeg]'.")
                return
            try:
                frames = load_video_frames(path, stride=1)
            except Exception as e:
                self.continue_info.config(text=f"Video load error: {e}")
                return
            if not frames:
                self.continue_info.config(text="No frames loaded from video.")
                return
            self.continue_video_path = path
            self.continue_video_frames = frames
            self.continue_image_path = None
            self.continue_path_label.config(
                text=f"{os.path.basename(path)} ({len(frames)} frames)", fg="black")

        self._continue_refresh_context_preview()

    def _continue_get_context_pils(self):
        """Returns (list_of_pils_or_None, description). Length = history."""
        history = self.settings['history'].get()

        if self.continue_mode.get() == 'image':
            if not self.continue_image_path:
                return None, "No image loaded"
            try:
                color_mode = self.settings['color_mode'].get()
                if color_mode == 'rgb':
                    img = load_image_as_rgb(self.continue_image_path)
                else:
                    img = load_image_as_grayscale(self.continue_image_path)
            except Exception as e:
                return None, f"Image load error: {e}"
            ctx = [None] * (history - 1) + [img]
            return ctx, (f"Image as last of {history} context slot(s) "
                         f"({history - 1} padding)")
        else:
            if not self.continue_video_frames:
                return None, "No video loaded"
            T = len(self.continue_video_frames)
            K = self.continue_frame_index.get()
            if K < 0:
                K = T + K
            K = max(0, min(T - 1, K))
            start = max(0, K - history + 1)
            real_frames = self.continue_video_frames[start:K + 1]
            pad_count = history - len(real_frames)
            ctx = [None] * pad_count + list(real_frames)
            desc = (f"Frames {start}..{K} "
                    f"({len(real_frames)} real, {pad_count} padding)")
            return ctx, desc

    def _continue_refresh_context_preview(self):
        ctx, desc = self._continue_get_context_pils()
        if ctx is None:
            self.continue_context_canvas.delete("all")
            self.continue_context_canvas.create_text(
                192, 50, text=desc, fill="white")
            self.continue_info.config(text=desc)
            return
        # Pixel space: the source pixels ARE the model input. Show them directly.
        self._display_continue_context(ctx, desc + "  [pixel input]",
                                        padding_flags=[p is None for p in ctx])

    def _display_continue_context(self, context_pils, desc, padding_flags=None):
        thumb = 96
        n = max(1, len(context_pils))
        grid = Image.new('RGB', (n * thumb, thumb), color=(50, 50, 50))
        for i, pil in enumerate(context_pils):
            is_pad = (pil is None) or (padding_flags[i] if padding_flags else False)
            if pil is None:
                img = Image.new('RGB', (thumb, thumb), (0, 0, 0))
            else:
                img = pil.convert('RGB').resize((thumb, thumb), Image.NEAREST)
            if is_pad:
                draw = ImageDraw.Draw(img)
                draw.rectangle([0, 0, thumb - 1, thumb - 1],
                               outline=(255, 100, 100), width=2)
                draw.text((4, 4), "PAD", fill=(255, 220, 220))
            grid.paste(img, (i * thumb, 0))
        photo = ImageTk.PhotoImage(grid)
        self.continue_context_canvas.delete("all")
        self.continue_context_canvas.config(width=grid.width, height=grid.height)
        self.continue_context_canvas.create_image(0, 0, image=photo, anchor='nw')
        self.continue_context_photo = photo
        self.continue_info.config(text=desc)

    def _build_continue_context_tensor(self, context_pils):
        """context_pils: list of PIL or None. Returns [1, history, C, H, W]."""
        img_size = self.settings['img_size'].get()
        color_mode = self.settings['color_mode'].get()
        in_ch = 3 if color_mode == 'rgb' else 1

        tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * in_ch, (0.5,) * in_ch),
        ])
        tensors = []
        for pil in context_pils:
            if pil is None:
                tensors.append(torch.zeros(in_ch, img_size, img_size))
            else:
                pil_norm = pil.convert('RGB') if color_mode == 'rgb' else pil.convert('L')
                tensors.append(tf(pil_norm))
        return torch.stack(tensors, dim=0).unsqueeze(0).to(self.rectified_model.device)

    # ---------------- Continue playback ----------------
    def _continue_start_playback(self):
        if not self.continue_gen_frames_pil:
            return
        self._continue_pause_playback()
        self.continue_playback_active = True
        self.continue_playback_idx = 0
        self.continue_play_btn.config(state=tk.DISABLED)
        self.continue_pause_btn.config(state=tk.NORMAL)
        self._continue_playback_tick()

    def _continue_pause_playback(self):
        self.continue_playback_active = False
        if self.continue_playback_after_id is not None:
            try:
                self.root.after_cancel(self.continue_playback_after_id)
            except Exception:
                pass
            self.continue_playback_after_id = None
        if self.continue_gen_frames_pil:
            self.continue_play_btn.config(state=tk.NORMAL)
        self.continue_pause_btn.config(state=tk.DISABLED)

    def _continue_playback_tick(self):
        if not self.continue_playback_active or not self.continue_gen_frames_pil:
            return
        fi = self.continue_playback_idx % len(self.continue_gen_frames_pil)
        self._show_continue_frame(fi)
        self.continue_playback_idx += 1
        fps = max(1, int(self.playback_fps.get()))
        self.continue_playback_after_id = self.root.after(
            int(1000 / fps), self._continue_playback_tick)

    def _show_continue_frame(self, fi):
        if not self.continue_gen_frames_pil:
            return
        fi = fi % len(self.continue_gen_frames_pil)
        pil = self.continue_gen_frames_pil[fi]
        display = 256
        pil_disp = pil.convert('RGB').resize((display, display), Image.NEAREST)
        photo = ImageTk.PhotoImage(pil_disp)
        self.continue_gen_canvas.delete("all")
        self.continue_gen_canvas.config(width=display, height=display)
        self.continue_gen_canvas.create_image(0, 0, image=photo, anchor='nw')
        self.continue_gen_photo = photo
        self.continue_info.config(
            text=f"Playing frame {fi + 1}/{len(self.continue_gen_frames_pil)}")

    def _continue_show_strip(self):
        self._continue_pause_playback()
        if not self.continue_gen_frames_pil:
            return
        grid = self._render_strip_from_pils(self.continue_gen_frames_pil)
        photo = ImageTk.PhotoImage(grid)
        self.continue_gen_canvas.delete("all")
        self.continue_gen_canvas.config(width=grid.width, height=grid.height)
        self.continue_gen_canvas.create_image(0, 0, image=photo, anchor='nw')
        self.continue_gen_photo = photo
        self.continue_info.config(
            text=f"All {len(self.continue_gen_frames_pil)} generated frames")

    def _render_strip_from_pils(self, pils):
        thumb = 96
        T = len(pils)
        grid = Image.new('RGB', (max(1, T) * thumb, thumb), color=(80, 80, 80))
        for fi, pil in enumerate(pils):
            grid.paste(pil.convert('RGB').resize((thumb, thumb), Image.NEAREST),
                       (fi * thumb, 0))
        return grid

    def _pixels_to_pils(self, tensor_batch):
        """tensor_batch: [N, C, H, W] in [-1, 1]. Returns list[PIL]."""
        arr = ((tensor_batch + 1) / 2).clamp(0, 1).cpu().numpy()
        pils = []
        for i in range(arr.shape[0]):
            im = arr[i]
            if im.shape[0] == 1:
                im = np.stack([im[0]] * 3, axis=-1)
            else:
                im = im.transpose(1, 2, 0)
            pils.append(Image.fromarray((im * 255).astype(np.uint8)))
        return pils

    def _continue_finalize(self, decoded_tensor):
        pils = self._pixels_to_pils(decoded_tensor)
        self.continue_gen_frames_pil = pils
        self._continue_show_strip()
        self.continue_play_btn.config(state=tk.NORMAL)
        self.continue_pause_btn.config(state=tk.NORMAL)
        self.continue_show_all_btn.config(state=tk.NORMAL)
        self._continue_start_playback()

    # ---------------- Continue generation ----------------
    def continue_stop(self):
        self.continue_stop_flag = True
        self.continue_stop_btn.config(state=tk.DISABLED)
        self.continue_info.config(text="Stopping...")

    def continue_generate(self):
        if not self.rectified_model:
            self.continue_info.config(text="Load/init the flow model first!")
            return
        if self.continue_in_progress:
            return
        context_pils, desc = self._continue_get_context_pils()
        if context_pils is None:
            self.continue_info.config(text=desc)
            return

        self._continue_pause_playback()
        self.continue_gen_frames_pil = []
        self.continue_play_btn.config(state=tk.DISABLED)
        self.continue_pause_btn.config(state=tk.DISABLED)
        self.continue_show_all_btn.config(state=tk.DISABLED)

        num_frames = int(self.continue_num_frames.get())
        steps = int(self.continue_steps.get())
        method = self.continue_method.get()
        cfg_scale = float(self.continue_cfg.get())
        use_ema = self.settings['ema_enabled'].get()
        prompt = self.continue_prompt.get().strip()

        cond = None
        if (prompt and self.settings['cond_enabled'].get()
                and self.text_encoder is not None):
            ti = text_to_indices(prompt, self.settings['cond_text_max_len'].get())
            tt = torch.tensor([ti], dtype=torch.long,
                              device=self.rectified_model.device)
            with torch.no_grad():
                cond = self.text_encoder(tt)
        else:
            cfg_scale = 1.0

        try:
            context_latent = self._build_continue_context_tensor(context_pils)
        except Exception as e:
            self.continue_info.config(text=f"Context encode error: {e}")
            return

        self.continue_stop_flag = False
        self.continue_in_progress = True
        self.continue_generate_btn.config(state=tk.DISABLED)
        self.continue_stop_btn.config(state=tk.NORMAL)
        self.continue_info.config(text="Generating...")

        progressive = bool(self.continue_progressive.get())
        interval = max(1, int(self.continue_prog_interval.get()))

        threading.Thread(
            target=self._continue_generate_thread,
            args=(context_latent, cond, num_frames, steps, method, cfg_scale,
                  use_ema, progressive, interval),
            daemon=True).start()

    def _continue_generate_thread(self, context_latent, cond, num_frames, steps,
                                    method, cfg_scale, use_ema, progressive, interval):
        final = None
        try:
            if progressive:
                for fi, si, frames_z in self.rectified_model.sample_video_step_by_step(
                        n_samples=1, num_frames=num_frames, cond=cond,
                        steps=steps, method=method, cfg_scale=cfg_scale,
                        use_ema=use_ema, init_context=context_latent):
                    if self.continue_stop_flag:
                        break
                    if si % interval != 0 and si != steps:
                        continue
                    all_z = torch.cat(frames_z, dim=0)  # [T_so_far, C, H, W]
                    final = all_z
                    self.root.after(
                        0, lambda d=all_z, f=fi, s=si:
                        self._continue_show_progressive(d, f, s, num_frames, steps))
            else:
                z_vid = self.rectified_model.sample_video(
                    n_samples=1, num_frames=num_frames, cond=cond,
                    steps=steps, method=method, cfg_scale=cfg_scale,
                    use_ema=use_ema, init_context=context_latent,
                    stop_check=lambda: self.continue_stop_flag)
                if z_vid is not None:
                    T, C, H, W = z_vid.shape[1:]
                    final = z_vid.reshape(T, C, H, W)

            if (not self.continue_stop_flag) and final is not None:
                self.root.after(0, lambda d=final: self._continue_finalize(d))
                self.root.after(0, lambda: self.continue_info.config(text="Complete."))
            else:
                self.root.after(0, lambda: self.continue_info.config(text="Stopped."))
        except Exception as e:
            import traceback; traceback.print_exc()
            self.root.after(0, lambda: self.continue_info.config(text=f"Error: {e}"))
        finally:
            self.continue_in_progress = False
            self.root.after(0, lambda: self.continue_generate_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.continue_stop_btn.config(state=tk.DISABLED))

    def _continue_show_progressive(self, decoded, fi, si, num_frames, total_steps):
        pils = self._pixels_to_pils(decoded)
        grid = self._render_strip_from_pils(pils)
        photo = ImageTk.PhotoImage(grid)
        self.continue_gen_canvas.delete("all")
        self.continue_gen_canvas.config(width=grid.width, height=grid.height)
        self.continue_gen_canvas.create_image(0, 0, image=photo, anchor='nw')
        self.continue_gen_photo = photo
        self.continue_info.config(
            text=f"Frame {fi + 1}/{num_frames}, step {si}/{total_steps}")

    # ---------------- Logging ----------------
    def log_dataset(self, msg):
        self.message_queue_dataset.put(msg)

    def log_rectified(self, msg):
        self.message_queue_rectified.put(msg)

    def process_messages_dataset(self):
        try:
            while True:
                msg = self.message_queue_dataset.get_nowait()
                self.dataset_log_text.insert(tk.END, f"{time.strftime('%H:%M:%S')} - {msg}\n")
                self.dataset_log_text.see(tk.END)
                self.status_label.config(text=msg[:80])
        except queue.Empty:
            pass
        self.root.after(100, self.process_messages_dataset)

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

    # ---------------- Source list ----------------
    def _refresh_source_listbox(self):
        self.source_listbox.delete(0, tk.END)
        for s in self.sources:
            if s['kind'] == 'image':
                self.source_listbox.insert(tk.END, f"[img] {os.path.basename(s['path'])}")
            elif s['kind'] == 'video':
                self.source_listbox.insert(
                    tk.END, f"[vid] {os.path.basename(s['path'])} (stride={s.get('stride',1)})")
            else:
                self.source_listbox.insert(
                    tk.END, f"[dir] {os.path.basename(s['path'])} ({len(s['paths'])} files, stride={s.get('stride',1)})")

    def add_images(self):
        files = filedialog.askopenfilenames(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.jfif *.webp *.bmp")])
        added = 0
        for f in files:
            if any(s.get('path') == f and s['kind'] == 'image' for s in self.sources):
                continue
            self.sources.append({'kind': 'image', 'path': f})
            self.source_labels.append([os.path.splitext(os.path.basename(f))[0]])
            added += 1
        self._refresh_source_listbox()
        self.log_dataset(f"Added {added} images. Total sources: {len(self.sources)}")

    def add_videos(self):
        files = filedialog.askopenfilenames(
            filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv *.webm *.gif *.m4v *.wmv *.flv"),
                       ("All files", "*.*")])
        if not files:
            return
        if not (_HAS_CV2 or _HAS_IMAGEIO):
            messagebox.showerror("Video support missing",
                                 "Install 'opencv-python' or 'imageio[ffmpeg]'.")
            return
        stride = max(1, int(self.frame_stride.get()))
        added = 0
        for f in files:
            if any(s.get('path') == f and s['kind'] == 'video' for s in self.sources):
                continue
            self.sources.append({'kind': 'video', 'path': f, 'stride': stride})
            self.source_labels.append([os.path.splitext(os.path.basename(f))[0]])
            added += 1
        self._refresh_source_listbox()
        self.log_dataset(f"Added {added} videos (stride={stride}).")

    def add_video_folder(self):
        folder = filedialog.askdirectory(title="Pick folder of sequential images (sorted by name)")
        if not folder:
            return
        paths = []
        for f in sorted(os.listdir(folder)):
            if f.lower().endswith(IMAGE_EXTS):
                paths.append(os.path.join(folder, f))
        if not paths:
            self.log_dataset("No images found in folder.")
            return
        stride = max(1, int(self.frame_stride.get()))
        self.sources.append({'kind': 'folder', 'path': folder, 'paths': paths, 'stride': stride})
        self.source_labels.append([os.path.basename(folder) or 'video'])
        self._refresh_source_listbox()
        self.log_dataset(
            f"Added folder '{os.path.basename(folder)}' as one video "
            f"({len(paths)} frames, stride={stride}).")

    def add_folder(self):
        folder = filedialog.askdirectory(title="Add all images under this folder (recursive)")
        if not folder:
            return
        n = 0
        for r, _, fs in os.walk(folder):
            for f in fs:
                if f.lower().endswith(IMAGE_EXTS):
                    p = os.path.join(r, f)
                    if any(s.get('path') == p and s['kind'] == 'image' for s in self.sources):
                        continue
                    self.sources.append({'kind': 'image', 'path': p})
                    self.source_labels.append([os.path.splitext(f)[0]])
                    n += 1
        self._refresh_source_listbox()
        self.log_dataset(f"Added {n} images from folder.")

    def clear_all(self):
        self.sources = []
        self.source_labels = []
        self._refresh_source_listbox()
        self.csv_status.config(text="No CSV loaded", fg="red")
        self.log_dataset("Cleared all sources")

    def _source_basename(self, src):
        if src['kind'] in ('image', 'video'):
            return os.path.basename(src['path'])
        return os.path.basename(src['path']) if src.get('path') else ''

    def load_csv(self):
        fname = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if not fname:
            return
        self.csv_path = fname
        label_map = {}
        try:
            with open(fname, 'r', encoding='utf-8-sig') as f:
                for row in csv.reader(f):
                    if not row or len(row) < 2:
                        continue
                    nm = row[0].strip().strip('"').strip("'")
                    lb = row[1].strip().strip('"').strip("'")
                    label_map.setdefault(os.path.basename(nm).lower(), []).append(lb)
        except Exception as e:
            self.log_dataset(f"CSV error: {e}")
            return
        matched = 0
        for i, src in enumerate(self.sources):
            base = self._source_basename(src).lower()
            if base in label_map:
                self.source_labels[i] = label_map[base]
                matched += 1
            else:
                self.source_labels[i] = [os.path.splitext(base)[0]] if base else ['']
        self.csv_status.config(text=f"CSV: {matched}/{len(self.sources)} matched", fg="green")
        self.log_dataset(f"CSV loaded. {matched}/{len(self.sources)} sources matched.")

    def use_filenames_as_labels(self):
        for i, src in enumerate(self.sources):
            self.source_labels[i] = [os.path.splitext(self._source_basename(src))[0] or 'unknown']
        self.csv_status.config(text="Using filenames as labels", fg="blue")
        self.log_dataset("Using filenames as labels.")

    def use_folders_as_labels(self):
        for i, src in enumerate(self.sources):
            if src['kind'] in ('image', 'video'):
                p = src['path']
            else:
                p = src.get('path', '')
            self.source_labels[i] = [os.path.basename(os.path.dirname(p)) or 'unknown']
        self.csv_status.config(text="Using folder names as labels", fg="blue")
        self.log_dataset("Using folder names as labels.")

    # ---------------- Model init / Training ----------------
    def _build_dataset(self):
        aug_dict = {k: v.get() for k, v in self.aug_settings.items()}
        return VideoFrameDataset(
            sources=self.sources,
            labels=self.source_labels,
            img_size=self.settings['img_size'].get(),
            color_mode=self.settings['color_mode'].get(),
            history=max(1, min(10, self.settings['history'].get())),
            aug_settings=aug_dict,
            text_max_len=self.settings['cond_text_max_len'].get(),
        )

    def initialize_rectified_model(self):
        try:
            cond_enabled = self.settings['cond_enabled'].get()
            cond_dim = self.settings['cond_dim'].get() if cond_enabled else 1
            color_mode = self.settings['color_mode'].get()
            in_channels = 3 if color_mode == 'rgb' else 1
            img_size = self.settings['img_size'].get()
            history = max(1, min(10, self.settings['history'].get()))

            attn_levels = [self.settings['attn_level_0'].get(),
                           self.settings['attn_level_1'].get(),
                           self.settings['attn_level_2'].get()]
            attn_num_heads = int(self.settings['attn_num_heads'].get())
            attn_dropout = float(self.settings['attn_dropout'].get())

            cond_mode = self.settings['cond_mode'].get()
            if cond_mode not in VALID_COND_MODES:
                cond_mode = 'shift'

            self.rectified_model = RectifiedFlowPixelVideo(
                in_channels=in_channels, img_size=img_size,
                history=history,
                base_channels=self.settings['base_channels'].get(),
                cond_dim=cond_dim,
                attention_levels=attn_levels,
                attn_num_heads=attn_num_heads,
                attn_dropout=attn_dropout,
                padding_mode=self.settings['padding_mode'].get(),
                cond_mode=cond_mode,
                channel_mult=(1, 2, 4),
            )
            self.rectified_model.use_ot = self.settings['use_ot'].get()
            for pg in self.rectified_model.optimizer.param_groups:
                pg['lr'] = self.settings['lr'].get()

            if self.settings['ema_enabled'].get():
                self.rectified_model.set_ema(
                    decay=self.settings['ema_decay'].get(),
                    mode=self.settings['ema_mode'].get())

            if cond_enabled:
                enc = self.settings['text_encoder_type'].get()
                if enc == 'BiGRU':
                    self.text_encoder = TextEncoder(
                        vocab_size=256,
                        embed_dim=self.settings['cond_embed_dim'].get(),
                        hidden_size=self.settings['cond_hidden_size'].get(),
                        num_layers=self.settings['cond_num_layers'].get(),
                        cond_dim=self.settings['cond_dim'].get(),
                    )
                else:
                    self.text_encoder = TransformerTextEncoder(
                        vocab_size=256,
                        embed_dim=self.settings['cond_embed_dim'].get(),
                        num_heads=self.settings['cond_num_heads'].get(),
                        num_layers=self.settings['cond_num_layers'].get(),
                        ff_dim=self.settings['cond_ff_dim'].get(),
                        cond_dim=self.settings['cond_dim'].get(),
                        max_len=self.settings['cond_text_max_len'].get(),
                    )
                self.text_encoder.to(self.rectified_model.device)
                self.rectified_model.optimizer = optim.Adam(
                    list(self.rectified_model.model.parameters())
                    + list(self.text_encoder.parameters()),
                    lr=self.settings['lr'].get(),
                )
            else:
                self.text_encoder = None

            self.log_rectified(
                f"Init OK. pixel={in_channels}ch {img_size}px, history={history}, "
                f"base_ch={self.settings['base_channels'].get()}, "
                f"pad={self.settings['padding_mode'].get()}, "
                f"attn={attn_levels}, heads={attn_num_heads}, cond_mode={cond_mode}, "
                f"aug_pad={self.aug_settings['aug_padding_mode'].get()}")
        except Exception as e:
            import traceback; traceback.print_exc()
            self.log_rectified(f"Init error: {e}")

    def start_training(self):
        if not self.sources:
            self.log_rectified("No sources!"); return
        if not self.rectified_model:
            self.log_rectified("Initialize model first!"); return
        if self.training_rectified:
            self.log_rectified("Already training."); return
        if self.settings['cond_enabled'].get() and not self.source_labels:
            self.log_rectified("Load labels first."); return
        try:
            epochs = int(self.epoch_var.get())
        except Exception:
            self.log_rectified("Invalid epochs"); return
        self.training_rectified = True
        self.train_start_time = time.time()
        threading.Thread(target=self.train_loop, args=(epochs,), daemon=True).start()
        self.log_rectified(f"Training started for {epochs} epochs.")

    def train_loop(self, epochs):
        try:
            bs = self.settings['batch_size'].get()
            cond_enabled = self.settings['cond_enabled'].get()
            cfg_drop = self.settings['cfg_dropout_prob'].get()
            preview_enabled = self.settings['preview_enabled'].get()
            preview_freq = self.settings['preview_epoch_freq'].get()

            noise_ctx_on = self.aug_settings['noise_ctx_enabled'].get()
            noise_ctx_prob = self.aug_settings['noise_ctx_prob'].get() if noise_ctx_on else 0.0
            noise_ctx_scale = self.aug_settings['noise_ctx_scale'].get()

            ds = self._build_dataset()
            dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=0,
                            pin_memory=False)

            for ep in range(epochs):
                if not self.training_rectified:
                    break
                for pg in self.rectified_model.optimizer.param_groups:
                    pg['lr'] = self.settings['lr'].get()
                self.rectified_model.use_ot = self.settings['use_ot'].get()

                tot, n = 0.0, 0
                for seqs, txts in dl:
                    if not self.training_rectified:
                        break
                    B, S, C, H, W = seqs.shape
                    seqs = seqs.to(self.rectified_model.device)
                    context = seqs[:, :-1]        # [B, history, C, H, W]
                    x1 = seqs[:, -1]              # [B, C, H, W]
                    x0 = torch.randn_like(x1)
                    cond = None
                    if cond_enabled and self.text_encoder is not None:
                        cond = self.text_encoder(txts.to(self.rectified_model.device))
                    loss = self.rectified_model.train_step(
                        x0, x1, cond=cond, context=context,
                        cfg_dropout_prob=cfg_drop,
                        noise_ctx_prob=noise_ctx_prob,
                        noise_ctx_scale=noise_ctx_scale)
                    tot += loss; n += 1

                avg = tot / max(1, n)
                el = time.time() - self.train_start_time
                self.log_rectified(f"Epoch {ep+1}/{epochs} | Loss: {avg:.6f} | Time: {el:.1f}s")
                if preview_enabled and (ep + 1) % preview_freq == 0:
                    self.preview_with_prompt()

            self.training_rectified = False
            self.log_rectified("Training finished.")
        except Exception as e:
            import traceback; traceback.print_exc()
            self.log_rectified(f"Training error: {e}")
            self.training_rectified = False

    def stop_training(self):
        self.training_rectified = False
        self.log_rectified("Training stopped.")

    def preview_with_prompt(self):
        if not self.rectified_model:
            return
        try:
            prompt = self.test_prompt_entry.get().strip()
            unconditional = (prompt == "")
            cond = None
            cfg_scale = 1.0 if unconditional else self.cfg_scale.get()
            n = 1
            num_frames = self.settings['preview_num_frames'].get()
            if (not unconditional
                    and self.settings['cond_enabled'].get()
                    and self.text_encoder is not None):
                ti = text_to_indices(prompt, self.settings['cond_text_max_len'].get())
                tt = torch.tensor([ti] * n, dtype=torch.long,
                                  device=self.rectified_model.device)
                with torch.no_grad():
                    cond = self.text_encoder(tt)
            use_ema = self.settings['ema_enabled'].get()
            z_vid = self.rectified_model.sample_video(
                n_samples=n, num_frames=num_frames, cond=cond,
                steps=self.settings['preview_steps'].get(),
                method=self.settings['preview_method'].get(),
                cfg_scale=cfg_scale, use_ema=use_ema)
            if z_vid is None:
                self.log_rectified("Preview: stopped before first frame.")
                return
            _, T, C, H, W = z_vid.shape
            pils = self._pixels_to_pils(z_vid.reshape(T, C, H, W))
            cols = min(4, T)
            rows = int(math.ceil(T / cols))
            thumb = self.thumbnail_size
            grid = Image.new('RGB', (cols * thumb, rows * thumb))
            for i, pil in enumerate(pils):
                r, c = i // cols, i % cols
                grid.paste(pil.resize((thumb, thumb), Image.NEAREST),
                           (c * thumb, r * thumb))
            self.rectified_preview_photo = ImageTk.PhotoImage(grid)
            self.rectified_preview_canvas.delete("all")
            self.rectified_preview_canvas.config(width=grid.width, height=grid.height)
            self.rectified_preview_canvas.create_image(
                max(1, grid.width // 2), max(1, grid.height // 2),
                image=self.rectified_preview_photo)
            self.log_rectified(
                f"Preview: {num_frames} frames"
                + (f" (cond '{prompt}')" if not unconditional else " (uncond)"))
        except Exception as e:
            self.log_rectified(f"Preview error: {e}")

    # ---------------- Save / Load ----------------
    def save_model(self):
        if not self.rectified_model:
            self.log_rectified("No model."); return
        fname = filedialog.asksaveasfilename(defaultextension=".pth",
                                             filetypes=[("PyTorch", "*.pth")])
        if not fname:
            return
        if self.rectified_model.use_ema and self.rectified_model.ema_model is not None:
            model_state = self.rectified_model.ema_model.state_dict()
        else:
            model_state = self.rectified_model.model.state_dict()

        d = {
            'model_state': model_state,
            'optimizer_state': self.rectified_model.optimizer.state_dict(),
            'settings': {k: v.get() for k, v in self.settings.items()},
            'aug_settings': {k: v.get() for k, v in self.aug_settings.items()},
            'history': self.rectified_model.history,
            'use_ema': self.rectified_model.use_ema,
            'ema_decay': self.rectified_model.ema_decay,
            'ema_mode': self.rectified_model.ema_mode,
            'use_ot': self.rectified_model.use_ot,
            'channel_mult': (1, 2, 4),
            'attn_levels': self.rectified_model.model.attention_levels,
            'attn_num_heads': self.rectified_model.model.attn_num_heads,
            'attn_dropout': self.rectified_model.model.attn_dropout,
            'cond_mode': self.rectified_model.cond_mode,
            # top-level pixel-specific
            'in_channels': self.rectified_model.in_channels,
            'img_size': self.rectified_model.img_size,
        }
        if self.text_encoder is not None:
            d['text_encoder_state'] = self.text_encoder.state_dict()
        torch.save(d, fname)
        self.log_rectified(f"Model saved to {fname}")

    def load_model(self):
        fname = filedialog.askopenfilename(filetypes=[("PyTorch", "*.pth")])
        if not fname:
            return
        try:
            ckpt = torch.load(fname, map_location='cpu')

            # ---- settings (with backward-compat) ----
            if 'settings' in ckpt:
                if 'cond_mode' not in ckpt['settings']:
                    self.settings['cond_mode'].set('shift')
                for k, v in ckpt['settings'].items():
                    if k in self.settings:
                        try:
                            self.settings[k].set(v)
                        except Exception:
                            pass

            # Top-level overrides
            if 'img_size' in ckpt and not isinstance(ckpt['img_size'], dict):
                try:
                    sz = ckpt['img_size']
                    if isinstance(sz, (list, tuple)):
                        sz = sz[0]
                    self.settings['img_size'].set(int(sz))
                except Exception:
                    pass
            if 'history' in ckpt:
                try:
                    self.settings['history'].set(int(ckpt['history']))
                except Exception:
                    pass

            if 'cond_mode' in ckpt:
                try:
                    self.settings['cond_mode'].set(ckpt['cond_mode'])
                except Exception:
                    pass
            if 'attn_levels' in ckpt and len(ckpt['attn_levels']) >= 3:
                self.settings['attn_level_0'].set(bool(ckpt['attn_levels'][0]))
                self.settings['attn_level_1'].set(bool(ckpt['attn_levels'][1]))
                self.settings['attn_level_2'].set(bool(ckpt['attn_levels'][2]))
            if 'attn_num_heads' in ckpt:
                self.settings['attn_num_heads'].set(ckpt['attn_num_heads'])
            if 'attn_dropout' in ckpt:
                self.settings['attn_dropout'].set(ckpt['attn_dropout'])
            if 'use_ot' in ckpt:
                self.settings['use_ot'].set(ckpt['use_ot'])
            if 'ema_enabled' in ckpt:
                self.settings['ema_enabled'].set(ckpt['ema_enabled'])
            if 'ema_decay' in ckpt and ckpt['ema_decay'] is not None:
                self.settings['ema_decay'].set(ckpt['ema_decay'])
            if 'ema_mode' in ckpt and ckpt['ema_mode'] is not None:
                self.settings['ema_mode'].set(ckpt['ema_mode'])

            # Infer color_mode / history from in_channels if missing
            if 'in_channels' in ckpt:
                in_ch = ckpt['in_channels']
                h = int(ckpt.get('history', self.settings['history'].get()))
                per_frame = in_ch // max(1, (1 + h))
                self.settings['color_mode'].set('rgb' if per_frame >= 3 else 'grayscale')

            # Aug settings
            if 'aug_settings' in ckpt:
                for k, v in ckpt['aug_settings'].items():
                    if k in self.aug_settings:
                        try:
                            self.aug_settings[k].set(v)
                        except Exception:
                            pass

            self._refresh_head_dims()
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
            self.log_rectified(
                f"Loaded from {fname} "
                f"(img_size={self.settings['img_size'].get()}, "
                f"color_mode={self.settings['color_mode'].get()}, "
                f"history={self.settings['history'].get()})")
        except Exception as e:
            import traceback; traceback.print_exc()
            self.log_rectified(f"Load error: {e}")

    # ---------------- Generation ----------------
    def _stop_generation(self):
        self.gen_stop_flag = True
        self.stop_btn.config(state=tk.DISABLED)
        self.gen_info.config(text="Stopping...")

    def generate_samples(self):
        if not self.rectified_model:
            self.gen_info.config(text="Model not loaded!"); return
        if self.settings['cond_enabled'].get() and self.text_encoder is None:
            self.gen_info.config(text="Text encoder missing!"); return
        n_videos = int(self.gen_count.get())
        num_frames = int(self.gen_frames.get())
        steps = self.ode_steps.get()
        method = self.ode_method.get()
        prompt = self.gen_prompt.get().strip()
        cfg_scale = self.cfg_scale.get()
        use_ema = self.settings['ema_enabled'].get()
        unconditional = (prompt == "")
        cond = None
        eff_cfg = 1.0 if unconditional else cfg_scale
        if (not unconditional
                and self.settings['cond_enabled'].get()
                and self.text_encoder is not None):
            ti = text_to_indices(prompt, self.settings['cond_text_max_len'].get())
            tt = torch.tensor([ti] * n_videos, dtype=torch.long,
                              device=self.rectified_model.device)
            with torch.no_grad():
                cond = self.text_encoder(tt)
        elif not unconditional:
            unconditional = True; eff_cfg = 1.0

        progressive = bool(self.progressive_preview.get())
        interval = max(1, int(self.prog_interval.get()))

        self._pause_playback()
        self.play_btn.config(state=tk.DISABLED)
        self.pause_btn.config(state=tk.DISABLED)
        self.generate_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.gen_stop_flag = False
        self.gen_info.config(text="Generating...")

        threading.Thread(
            target=self._generate_thread,
            args=(n_videos, num_frames, steps, method, cond, eff_cfg, use_ema,
                  progressive, interval),
            daemon=True).start()

    def _generate_thread(self, n_videos, num_frames, steps, method, cond,
                         cfg_scale, use_ema, progressive, interval):
        try:
            if progressive:
                self._generate_progressive(n_videos, num_frames, steps, method,
                                            cond, cfg_scale, use_ema, interval)
            else:
                self._generate_batch(n_videos, num_frames, steps, method,
                                      cond, cfg_scale, use_ema)
        except Exception as e:
            import traceback; traceback.print_exc()
            self.root.after(0, lambda: self.gen_info.config(text=f"Error: {e}"))
        finally:
            self.root.after(0, lambda: self.generate_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.stop_btn.config(state=tk.DISABLED))

    def _generate_batch(self, n_videos, num_frames, steps, method, cond,
                         cfg_scale, use_ema):
        def _check():
            return self.gen_stop_flag

        def _prog(cur, total):
            self.root.after(0, lambda: self.gen_info.config(
                text=f"Generating frame {cur}/{total}..."))

        z_vid = self.rectified_model.sample_video(
            n_samples=n_videos, num_frames=num_frames, cond=cond,
            steps=steps, method=method, cfg_scale=cfg_scale, use_ema=use_ema,
            stop_check=_check, progress_cb=_prog)

        if z_vid is None:
            self.root.after(0, lambda: self.gen_info.config(text="Stopped."))
            return

        frames_pil = self._decode_video_frames(z_vid, n_videos)
        self.gen_frames_pil = frames_pil
        self.gen_n_videos = n_videos
        self.root.after(0, self._on_generation_done)

    def _generate_progressive(self, n_videos, num_frames, steps, method, cond,
                                cfg_scale, use_ema, interval):
        decoded_per_frame = {}
        try:
            gen = self.rectified_model.sample_video_step_by_step(
                n_samples=n_videos, num_frames=num_frames, cond=cond,
                steps=steps, method=method, cfg_scale=cfg_scale, use_ema=use_ema)

            for fi, si, frames_z in gen:
                if self.gen_stop_flag:
                    break
                if si % interval != 0 and si != steps:
                    continue
                decoded_per_frame[fi] = frames_z[-1].clone()

                grid = self._build_progressive_grid(decoded_per_frame, n_videos, fi)
                self.root.after(
                    0, lambda g=grid, f=fi, s=si:
                    self._update_progressive_preview(g, f, s, steps, num_frames))
        finally:
            if decoded_per_frame:
                frames_pil = []
                n_done = max(decoded_per_frame.keys()) + 1
                for fi in range(n_done):
                    if fi not in decoded_per_frame:
                        continue
                    pils_row = self._pixels_to_pils(decoded_per_frame[fi])
                    frames_pil.append(pils_row)
                self.gen_frames_pil = frames_pil
                self.gen_n_videos = n_videos
                if self.gen_stop_flag:
                    self.root.after(0, lambda: self.gen_info.config(text="Stopped."))
                else:
                    self.root.after(0, self._on_generation_done)
            else:
                self.root.after(0, lambda: self.gen_info.config(text="Stopped."))

    def _decode_video_frames(self, z_vid, n_videos):
        """z_vid: [N, T, C, H, W] in [-1,1]. Returns list[list[PIL]]."""
        _, T, C, H, W = z_vid.shape
        flat = z_vid.reshape(n_videos * T, C, H, W)
        pils = self._pixels_to_pils(flat)
        frames_pil = []
        idx = 0
        for fi in range(T):
            row = []
            for vi in range(n_videos):
                row.append(pils[idx]); idx += 1
            frames_pil.append(row)
        return frames_pil

    def _build_progressive_grid(self, decoded_per_frame, n_videos, current_fi):
        thumb = self.thumbnail_size
        n_cols = current_fi + 1
        grid = Image.new('RGB', (n_cols * thumb, n_videos * thumb),
                         color=(100, 100, 100))
        for fi in range(n_cols):
            if fi not in decoded_per_frame:
                continue
            row_pils = self._pixels_to_pils(decoded_per_frame[fi])
            for vi, pil in enumerate(row_pils):
                grid.paste(pil.resize((thumb, thumb), Image.NEAREST),
                           (fi * thumb, vi * thumb))
        return grid

    def _update_progressive_preview(self, grid, fi, si, total_steps, total_frames):
        self._anim_photo = ImageTk.PhotoImage(grid)
        self.gen_canvas.delete("all")
        self.gen_canvas.create_image(0, 0, image=self._anim_photo, anchor='nw')
        self.gen_canvas.configure(scrollregion=(0, 0, grid.width, grid.height))
        self.gen_info.config(
            text=f"Frame {fi+1}/{total_frames}, step {si}/{total_steps}")

    def _on_generation_done(self):
        self.generate_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.play_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.NORMAL)
        self.gen_info.config(
            text=f"Done. {len(self.gen_frames_pil)} frames × {self.gen_n_videos} videos.")
        self._start_playback()

    def _start_playback(self):
        if not self.gen_frames_pil:
            return
        self.playback_active = True
        self.playback_idx = 0
        self._playback_tick()

    def _pause_playback(self):
        self.playback_active = False
        if self.playback_after_id is not None:
            try:
                self.root.after_cancel(self.playback_after_id)
            except Exception:
                pass
            self.playback_after_id = None

    def _playback_tick(self):
        if not self.playback_active or not self.gen_frames_pil:
            return
        fi = self.playback_idx % len(self.gen_frames_pil)
        self._show_video_frame(fi)
        self.playback_idx += 1
        fps = max(1, int(self.playback_fps.get()))
        self.playback_after_id = self.root.after(int(1000 / fps), self._playback_tick)

    def _show_video_frame(self, fi):
        row = self.gen_frames_pil[fi]
        n = len(row)
        thumb = 128
        gs = int(math.ceil(math.sqrt(n)))
        grid = Image.new('RGB', (gs * thumb, gs * thumb), color=(128, 128, 128))
        for i, pil in enumerate(row):
            r, c = i // gs, i % gs
            grid.paste(pil.resize((thumb, thumb), Image.NEAREST),
                       (c * thumb, r * thumb))
        self._anim_photo = ImageTk.PhotoImage(grid)
        self.gen_canvas.delete("all")
        self.gen_canvas.create_image(0, 0, image=self._anim_photo, anchor='nw')
        self.gen_canvas.configure(scrollregion=(0, 0, grid.width, grid.height))
        self.gen_info.config(text=f"Frame {fi+1}/{len(self.gen_frames_pil)}")


# ==================== Main ====================

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    root = tk.Tk()
    app = RectifiedFlowApp(root)
    root.mainloop()