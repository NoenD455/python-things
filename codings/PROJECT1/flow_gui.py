# flow_gui.py
"""
Rectified Flow — Tkinter GUI.

Features:
  - Per-resolution attention toggles, num_heads, head_dim, attn dropout
  - Conditioning mode selector: shift / film / adagn / cross_attn
  - Augmentation padding mode in Settings (edge/reflect/symmetric/constant)
  - Sinkhorn / Hungarian minibatch OT selector
  - OTFM prior task (low-freq projection + noise interpolation)
  - Inpainting tab with mask drawing (L=paint, R=erase), noise σ,
    composite updates, full-edit fraction, live preview, text conditioning
  - Generation tab with progressive (live) preview + Stop
  - Flow-to-Image tab with progressive (live) preview + Stop
  - Checkpoints persist img_size / color_mode / cond_mode and load
    older-version checkpoints transparently.
"""
import tkinter as tk
from tkinter import filedialog, ttk
import torch
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
    RectifiedFlowPixel, ConditionalImageDataset,
    TextEncoder, TransformerTextEncoder, get_encoder_config,
    load_image_as_rgb, load_image_as_grayscale,
    text_to_indices, compute_head_dims,
)


class RectifiedFlowApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Rectified Flow — Sinkhorn / OTFM / Inpainting")

        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self.root.geometry(f"{max(1100, int(sw*0.85))}x{max(750, int(sh*0.85))}")
        self.root.minsize(1000, 700)

        self.image_paths = []
        self.labels = []
        self.csv_path = None
        self.training_rectified = False
        self.rectified_model = None
        self.text_encoder = None
        self.current_epoch_rectified = 0

        self.message_queue_dataset = queue.Queue()
        self.message_queue_rectified = queue.Queue()
        self.progressive_active = False

        # ---------------- settings ----------------
        self.settings = {
            'img_size': tk.IntVar(value=32),
            'color_mode': tk.StringVar(value='rgb'),
            'rectified_base_channels': tk.IntVar(value=64),
            'rectified_batch_size': tk.IntVar(value=16),
            'rectified_lr': tk.DoubleVar(value=2e-4),
            'cfg_dropout_prob': tk.DoubleVar(value=0.1),

            'attn_level_0': tk.BooleanVar(value=True),
            'attn_level_1': tk.BooleanVar(value=True),
            'attn_level_2': tk.BooleanVar(value=True),
            'attn_num_heads': tk.IntVar(value=4),
            'attn_dropout': tk.DoubleVar(value=0.0),

            'ema_enabled': tk.BooleanVar(value=False),
            'ema_decay': tk.DoubleVar(value=0.999),
            'ema_mode': tk.StringVar(value='standard'),

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

            'preview_enabled': tk.BooleanVar(value=True),
            'preview_steps': tk.IntVar(value=50),
            'preview_method': tk.StringVar(value='heun'),
            'preview_epoch_freq': tk.IntVar(value=5),

            'ot_task': tk.StringVar(value='noise'),
            'training_mode': tk.StringVar(value='dataset'),
            'blur_sigma': tk.DoubleVar(value=2.0),
            'bit_depth': tk.IntVar(value=1),
            'pixelation_size': tk.IntVar(value=8),
            'padding_mode': tk.StringVar(value='constant'),

            'use_ot': tk.BooleanVar(value=False),
            'ot_method': tk.StringVar(value='sinkhorn'),
            'ot_sinkhorn_eps': tk.DoubleVar(value=0.05),
            'ot_sinkhorn_iters': tk.IntVar(value=50),

            'source_noise_std': tk.DoubleVar(value=0.0),
            'pixelation_rescale_mode': tk.StringVar(value='nearest'),
            'inpaint_mask_ratio': tk.DoubleVar(value=0.3),

            'otfm_alpha': tk.DoubleVar(value=0.5),
            'otfm_downsample': tk.IntVar(value=4),
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
            'aug_padding_mode': tk.StringVar(value='edge'),
        }

        self.ode_method = tk.StringVar(value='euler')
        self.ode_steps = tk.IntVar(value=50)
        self.cfg_scale = tk.DoubleVar(value=2.0)
        self.thumbnail_size = 128

        # Generation progressive state
        self.progressive_grid = tk.BooleanVar(value=False)
        self.prog_interval = tk.IntVar(value=10)

        # Flow-to-Image progressive state
        self.flow_live_preview = tk.BooleanVar(value=True)
        self.flow_prog_interval = tk.IntVar(value=10)
        self.flow_progressive_active = False
        self.flow_stop_flag = False

        # Inpaint state
        self.inpaint_original_img = None
        self.inpaint_mask = None
        self.inpaint_photo = None
        self.inpaint_output_photo = None
        self.inpaint_last_xy = None
        self.inpaint_erase_mode = False
        self.inpaint_display_size = 400
        self.inpaint_in_progress = False
        self.inpaint_stop_flag = False

        self.inpaint_brush_size = tk.IntVar(value=8)
        self.inpaint_steps = tk.IntVar(value=50)
        self.inpaint_method = tk.StringVar(value='euler')
        self.inpaint_cfg = tk.DoubleVar(value=1.0)
        self.inpaint_noise_sigma = tk.DoubleVar(value=1.0)
        self.inpaint_composite_updates = tk.BooleanVar(value=True)
        self.inpaint_full_edit_from = tk.DoubleVar(value=0.7)
        self.inpaint_live_preview = tk.BooleanVar(value=True)
        self.inpaint_prompt = tk.StringVar(value="")

        self.head_dim_label_vars = []

        self.setup_gui()
        self.root.after(100, self.process_messages_dataset)
        self.root.after(100, self.process_messages_rectified)

    # ==================== GUI layout ====================
    def setup_gui(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.dataset_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.dataset_tab, text='Dataset')
        self.setup_dataset_tab()

        self.rectified_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.rectified_tab, text='Train Flow')
        self.setup_rectified_tab()

        self.settings_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.settings_tab, text='Settings')
        self.setup_settings_tab()

        self.generation_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.generation_tab, text='Generate')
        self.setup_generation_tab()

        self.inpaint_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.inpaint_tab, text='Inpaint')
        self.setup_inpaint_tab()

        self.flow_to_image_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.flow_to_image_tab, text='Flow to Image')
        self.setup_flow_to_image_tab()

        self.status_label = tk.Label(self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    # ---------------- Dataset tab ----------------
    def setup_dataset_tab(self):
        main = tk.Frame(self.dataset_tab)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left = tk.Frame(main, width=300)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left.pack_propagate(False)
        tk.Label(left, text="Dataset Management", font=("Arial", 12, "bold")).pack(pady=(0, 10))

        img_frame = tk.LabelFrame(left, text="Training Images", padx=5, pady=5)
        img_frame.pack(fill=tk.X, pady=(0, 10))
        tk.Button(img_frame, text="Add Images", command=self.add_images, width=20).pack(pady=2)
        tk.Button(img_frame, text="Add Folder (recursive)", command=self.add_folder, width=20).pack(pady=2)
        tk.Button(img_frame, text="Clear All", command=self.clear_images, width=20).pack(pady=2)
        self.image_listbox = tk.Listbox(img_frame, height=10)
        self.image_listbox.pack(fill=tk.X, pady=2)

        cond_frame = tk.LabelFrame(left, text="Labels / Captions", padx=5, pady=5)
        cond_frame.pack(fill=tk.X, pady=5)
        tk.Button(cond_frame, text="Load CSV (image,label)", command=self.load_csv, width=20).pack(pady=2)
        tk.Button(cond_frame, text="Use filenames as labels", command=self.use_filenames_as_labels, width=20).pack(pady=2)
        tk.Button(cond_frame, text="Use folder names as labels", command=self.use_folders_as_labels, width=20).pack(pady=2)
        self.csv_status = tk.Label(cond_frame, text="No CSV loaded", fg="red")
        self.csv_status.pack()

        log_frame = tk.LabelFrame(left, text="Log", padx=5, pady=5)
        log_frame.pack(fill=tk.BOTH, expand=True)
        self.dataset_log_text = tk.Text(log_frame, height=15, font=("Courier", 9))
        self.dataset_log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = tk.Scrollbar(log_frame, command=self.dataset_log_text.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.dataset_log_text.config(yscrollcommand=sb.set)

    # ---------------- Train tab ----------------
    def setup_rectified_tab(self):
        main = tk.Frame(self.rectified_tab)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left = tk.Frame(main, width=300)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left.pack_propagate(False)
        tk.Label(left, text="Rectified Flow Training", font=("Arial", 12, "bold")).pack(pady=(0, 10))

        cond_toggle = tk.LabelFrame(left, text="Conditional Training", padx=5, pady=5)
        cond_toggle.pack(fill=tk.X, pady=5)
        tk.Checkbutton(cond_toggle, text="Enable text conditioning",
                       variable=self.settings['cond_enabled']).pack(anchor='w')

        cfg_frame = tk.LabelFrame(left, text="CFG Training", padx=5, pady=5)
        cfg_frame.pack(fill=tk.X, pady=5)
        f = tk.Frame(cfg_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="CFG dropout prob:", width=18, anchor='w').pack(side=tk.LEFT)
        tk.Scale(f, from_=0.0, to=0.5, resolution=0.01,
                 variable=self.settings['cfg_dropout_prob'],
                 orient=tk.HORIZONTAL, length=150).pack(side=tk.RIGHT)

        tk.Button(left, text="Initialize Model", command=self.initialize_rectified_model,
                  width=20).pack(pady=5)
        ef = tk.Frame(left); ef.pack(pady=5)
        tk.Label(ef, text="Epochs:").pack(side=tk.LEFT)
        self.rectified_epoch_var = tk.StringVar(value="200")
        tk.Entry(ef, textvariable=self.rectified_epoch_var, width=8).pack(side=tk.LEFT, padx=5)

        tk.Button(left, text="Start Training", command=self.start_rectified_training,
                  width=20, bg="lightgreen").pack(pady=5)
        tk.Button(left, text="Stop Training", command=self.stop_rectified_training,
                  width=20, bg="salmon").pack(pady=5)
        tk.Button(left, text="Save Model", command=self.save_rectified_model, width=20).pack(pady=5)
        tk.Button(left, text="Load Model", command=self.load_rectified_model, width=20).pack(pady=5)

        right = tk.Frame(main)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        prev = tk.LabelFrame(right, text="Generated Samples (preview)", padx=5, pady=5)
        prev.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        self.rectified_preview_canvas = tk.Canvas(prev, bg='gray', width=256, height=256)
        self.rectified_preview_canvas.pack()

        prompt_frame = tk.LabelFrame(right, text="Test prompt (for manual preview)", padx=5, pady=5)
        prompt_frame.pack(fill=tk.X, pady=(0, 5))
        self.test_prompt_entry = tk.Entry(prompt_frame)
        self.test_prompt_entry.insert(0, "a cute cat")
        self.test_prompt_entry.pack(fill=tk.X, pady=2)
        tk.Button(prompt_frame, text="Generate Preview (manual)",
                  command=self.rectified_preview_with_prompt, width=20).pack(pady=2)

        log_frame = tk.LabelFrame(right, text="Log", padx=5, pady=5)
        log_frame.pack(fill=tk.BOTH, expand=True)
        self.rectified_log_text = tk.Text(log_frame, height=15, font=("Courier", 9))
        self.rectified_log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = tk.Scrollbar(log_frame, command=self.rectified_log_text.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.rectified_log_text.config(yscrollcommand=sb.set)

    # ---------------- Settings tab ----------------
    def setup_settings_tab(self):
        main = tk.Frame(self.settings_tab)
        main.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        tk.Label(main, text="Model Settings", font=("Arial", 14, "bold")).pack(pady=(0, 20))

        canvas = tk.Canvas(main)
        sb = tk.Scrollbar(main, orient="vertical", command=canvas.yview)
        scrollable = tk.Frame(canvas)
        scrollable.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)

        # Image
        img_frame = tk.LabelFrame(scrollable, text="Image", padx=10, pady=10)
        img_frame.pack(fill=tk.X, pady=5)
        f = tk.Frame(img_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Color mode:", width=22, anchor='w').pack(side=tk.LEFT)
        ttk.Combobox(f, textvariable=self.settings['color_mode'],
                     values=['rgb', 'grayscale'], state='readonly', width=10).pack(side=tk.RIGHT)
        f = tk.Frame(img_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Image size:", width=22, anchor='w').pack(side=tk.LEFT)
        ttk.Spinbox(f, from_=16, to=128, textvariable=self.settings['img_size'],
                    width=8).pack(side=tk.RIGHT)

        # Architecture
        arch_frame = tk.LabelFrame(scrollable, text="Flow Architecture", padx=10, pady=10)
        arch_frame.pack(fill=tk.X, pady=5)
        f = tk.Frame(arch_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="UNet base channels:", width=22, anchor='w').pack(side=tk.LEFT)
        ttk.Spinbox(f, from_=32, to=128, textvariable=self.settings['rectified_base_channels'],
                    width=8, command=self._refresh_head_dims).pack(side=tk.RIGHT)
        f = tk.Frame(arch_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Conv padding mode:", width=22, anchor='w').pack(side=tk.LEFT)
        ttk.Combobox(f, textvariable=self.settings['padding_mode'],
                     values=['constant', 'reflect', 'replicate', 'circular'],
                     state='readonly', width=12).pack(side=tk.RIGHT)

        # Attention
        attn_frame = tk.LabelFrame(scrollable, text="Attention (per resolution)", padx=10, pady=10)
        attn_frame.pack(fill=tk.X, pady=5)
        f = tk.Frame(attn_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Enable attn at level:", width=22, anchor='w').pack(side=tk.LEFT)
        tk.Checkbutton(f, text="L0", variable=self.settings['attn_level_0'],
                       command=self._refresh_head_dims).pack(side=tk.LEFT, padx=3)
        tk.Checkbutton(f, text="L1", variable=self.settings['attn_level_1'],
                       command=self._refresh_head_dims).pack(side=tk.LEFT, padx=3)
        tk.Checkbutton(f, text="L2", variable=self.settings['attn_level_2'],
                       command=self._refresh_head_dims).pack(side=tk.LEFT, padx=3)
        f = tk.Frame(attn_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Number of heads:", width=22, anchor='w').pack(side=tk.LEFT)
        ttk.Spinbox(f, from_=1, to=32, textvariable=self.settings['attn_num_heads'],
                    width=8, command=self._refresh_head_dims).pack(side=tk.RIGHT)
        self.settings['attn_num_heads'].trace_add('write', lambda *_: self._refresh_head_dims())
        self.settings['rectified_base_channels'].trace_add('write', lambda *_: self._refresh_head_dims())
        f = tk.Frame(attn_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Attention dropout:", width=22, anchor='w').pack(side=tk.LEFT)
        tk.Entry(f, textvariable=self.settings['attn_dropout'], width=8).pack(side=tk.RIGHT)

        hd_frame = tk.LabelFrame(attn_frame, text="Auto head dim", padx=6, pady=6)
        hd_frame.pack(fill=tk.X, pady=(8, 2))
        self.head_dim_label_vars = [tk.StringVar(value="L0: —"),
                                    tk.StringVar(value="L1: —"),
                                    tk.StringVar(value="L2: —")]
        for v in self.head_dim_label_vars:
            tk.Label(hd_frame, textvariable=v, anchor='w',
                     font=("Courier", 9)).pack(fill=tk.X)

        # EMA
        ema_frame = tk.LabelFrame(scrollable, text="EMA", padx=10, pady=10)
        ema_frame.pack(fill=tk.X, pady=5)
        f = tk.Frame(ema_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Enable EMA:", width=22, anchor='w').pack(side=tk.LEFT)
        tk.Checkbutton(f, variable=self.settings['ema_enabled']).pack(side=tk.RIGHT)
        f = tk.Frame(ema_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="EMA decay:", width=22, anchor='w').pack(side=tk.LEFT)
        tk.Entry(f, textvariable=self.settings['ema_decay'], width=8).pack(side=tk.RIGHT)
        f = tk.Frame(ema_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="EMA mode:", width=22, anchor='w').pack(side=tk.LEFT)
        ttk.Combobox(f, textvariable=self.settings['ema_mode'],
                     values=['standard', 'lookahead'], state='readonly', width=12).pack(side=tk.RIGHT)

        # Training
        train_frame = tk.LabelFrame(scrollable, text="Training", padx=10, pady=10)
        train_frame.pack(fill=tk.X, pady=5)
        for label, key in [("Batch size:", 'rectified_batch_size'),
                           ("Learning rate:", 'rectified_lr')]:
            f = tk.Frame(train_frame); f.pack(fill=tk.X, pady=2)
            tk.Label(f, text=label, width=22, anchor='w').pack(side=tk.LEFT)
            tk.Entry(f, textvariable=self.settings[key], width=8).pack(side=tk.RIGHT)

        # Preview
        preview_frame = tk.LabelFrame(scrollable, text="Preview During Training", padx=10, pady=10)
        preview_frame.pack(fill=tk.X, pady=5)
        for label, key, kind in [("Enable preview:", 'preview_enabled', 'bool'),
                                 ("Every N epochs:", 'preview_epoch_freq', 'int'),
                                 ("Preview steps:", 'preview_steps', 'int')]:
            f = tk.Frame(preview_frame); f.pack(fill=tk.X, pady=2)
            tk.Label(f, text=label, width=22, anchor='w').pack(side=tk.LEFT)
            if kind == 'bool':
                tk.Checkbutton(f, variable=self.settings[key]).pack(side=tk.RIGHT)
            else:
                ttk.Spinbox(f, from_=1, to=200, textvariable=self.settings[key], width=8).pack(side=tk.RIGHT)
        f = tk.Frame(preview_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Preview ODE method:", width=22, anchor='w').pack(side=tk.LEFT)
        ttk.Combobox(f, textvariable=self.settings['preview_method'],
                     values=['euler', 'heun', 'rk4', 'midpoint'],
                     state='readonly', width=12).pack(side=tk.RIGHT)

        # OT / Prior
        ot_frame = tk.LabelFrame(scrollable, text="Optimal Transport / Prior Task", padx=10, pady=10)
        ot_frame.pack(fill=tk.X, pady=5)
        f = tk.Frame(ot_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Task:", width=22, anchor='w').pack(side=tk.LEFT)
        ttk.Combobox(f, textvariable=self.settings['ot_task'],
                     values=['noise', 'grayscale_to_color', 'blur_to_sharp',
                             'white_to_image', 'average_separation',
                             'unquantization', 'pixelation', 'inpainting',
                             'otfm'],
                     state='readonly', width=20).pack(side=tk.RIGHT)
        f = tk.Frame(ot_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Training mode:", width=22, anchor='w').pack(side=tk.LEFT)
        ttk.Combobox(f, textvariable=self.settings['training_mode'],
                     values=['dataset', 'noise', 'mixed'], state='readonly',
                     width=12).pack(side=tk.RIGHT)
        for label, key, low, high, inc in [
            ("Blur sigma:", 'blur_sigma', 0.5, 5.0, 0.1),
            ("Bit depth (unquant):", 'bit_depth', 1, 7, 1),
            ("Pixelation block size:", 'pixelation_size', 2, 32, 1),
            ("Inpaint mask ratio:", 'inpaint_mask_ratio', 0.05, 0.9, 0.05),
            ("OTFM α:", 'otfm_alpha', 0.0, 1.0, 0.05),
            ("OTFM downsample factor:", 'otfm_downsample', 2, 16, 1),
        ]:
            f = tk.Frame(ot_frame); f.pack(fill=tk.X, pady=2)
            tk.Label(f, text=label, width=22, anchor='w').pack(side=tk.LEFT)
            ttk.Spinbox(f, from_=low, to=high, increment=inc,
                        textvariable=self.settings[key], width=8).pack(side=tk.RIGHT)

        # Minibatch OT coupling
        otsink = tk.LabelFrame(ot_frame, text="Minibatch OT coupling", padx=6, pady=6)
        otsink.pack(fill=tk.X, pady=(8, 2))
        f = tk.Frame(otsink); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Enable OT Matching:", width=22, anchor='w').pack(side=tk.LEFT)
        tk.Checkbutton(f, variable=self.settings['use_ot']).pack(side=tk.RIGHT)
        f = tk.Frame(otsink); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="OT method:", width=22, anchor='w').pack(side=tk.LEFT)
        ttk.Combobox(f, textvariable=self.settings['ot_method'],
                     values=['sinkhorn', 'hungarian'], state='readonly',
                     width=12).pack(side=tk.RIGHT)
        f = tk.Frame(otsink); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Sinkhorn epsilon:", width=22, anchor='w').pack(side=tk.LEFT)
        ttk.Spinbox(f, from_=0.001, to=1.0, increment=0.005,
                    textvariable=self.settings['ot_sinkhorn_eps'], width=8).pack(side=tk.RIGHT)
        f = tk.Frame(otsink); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Sinkhorn iters:", width=22, anchor='w').pack(side=tk.LEFT)
        ttk.Spinbox(f, from_=1, to=500, textvariable=self.settings['ot_sinkhorn_iters'],
                    width=8).pack(side=tk.RIGHT)
        tk.Label(otsink, text="Lower ε → sharper coupling; 0.01–0.05 works well.",
                 fg='gray', font=("Arial", 8, "italic")).pack(anchor='w')

        f = tk.Frame(ot_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Source noise std (0..1):", width=22, anchor='w').pack(side=tk.LEFT)
        ttk.Spinbox(f, from_=0.0, to=1.0, increment=0.01,
                    textvariable=self.settings['source_noise_std'], width=8).pack(side=tk.RIGHT)
        f = tk.Frame(ot_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Pixelation rescale mode:", width=22, anchor='w').pack(side=tk.LEFT)
        ttk.Combobox(f, textvariable=self.settings['pixelation_rescale_mode'],
                     values=['nearest', 'bilinear', 'bicubic'],
                     state='readonly', width=12).pack(side=tk.RIGHT)

        # Text conditioning
        cond_frame = tk.LabelFrame(scrollable, text="Text Conditioning", padx=10, pady=10)
        cond_frame.pack(fill=tk.X, pady=5)
        f = tk.Frame(cond_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Encoder type:", width=22, anchor='w').pack(side=tk.LEFT)
        ttk.Combobox(f, textvariable=self.settings['text_encoder_type'],
                     values=['BiGRU', 'BiTransformer'],
                     state='readonly', width=12).pack(side=tk.RIGHT)
        f = tk.Frame(cond_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Encoder size:", width=22, anchor='w').pack(side=tk.LEFT)
        ttk.Combobox(f, textvariable=self.settings['text_encoder_size'],
                     values=['tiny', 'small', 'medium', 'large'],
                     state='readonly', width=10).pack(side=tk.RIGHT)
        f = tk.Frame(cond_frame); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Cond mode:", width=22, anchor='w').pack(side=tk.LEFT)
        ttk.Combobox(f, textvariable=self.settings['cond_mode'],
                     values=['shift', 'film', 'adagn', 'cross_attn'],
                     state='readonly', width=14).pack(side=tk.RIGHT)
        tk.Label(cond_frame,
                 text="shift = additive (legacy) · film = scale+shift · "
                      "adagn = adaptive GroupNorm · cross_attn = cross-attention",
                 fg='gray', font=("Arial", 8, "italic")).pack(anchor='w')
        tk.Button(cond_frame, text="Apply preset",
                  command=self.apply_text_encoder_preset).pack(pady=2)
        for label, key, low, high in [
            ("Embed dim:", 'cond_embed_dim', 32, 256),
            ("Hidden size (GRU):", 'cond_hidden_size', 32, 512),
            ("Num layers:", 'cond_num_layers', 1, 4),
            ("Num heads (TF):", 'cond_num_heads', 1, 16),
            ("FF dim (TF):", 'cond_ff_dim', 64, 1024),
            ("Cond dim:", 'cond_dim', 64, 512),
            ("Max text len:", 'cond_text_max_len', 32, 256),
        ]:
            f = tk.Frame(cond_frame); f.pack(fill=tk.X, pady=2)
            tk.Label(f, text=label, width=22, anchor='w').pack(side=tk.LEFT)
            ttk.Spinbox(f, from_=low, to=high, textvariable=self.settings[key],
                        width=8).pack(side=tk.RIGHT)

        # Augmentations
        aug_frame = tk.LabelFrame(scrollable, text="Data Augmentations", padx=10, pady=10)
        aug_frame.pack(fill=tk.X, pady=5)
        fpad = tk.Frame(aug_frame); fpad.pack(fill=tk.X, pady=(0, 6))
        tk.Label(fpad, text="Aug padding mode:", width=22, anchor='w').pack(side=tk.LEFT)
        ttk.Combobox(fpad, textvariable=self.aug_settings['aug_padding_mode'],
                     values=['edge', 'reflect', 'symmetric', 'constant'],
                     state='readonly', width=14).pack(side=tk.RIGHT)
        tk.Label(aug_frame,
                 text="'edge' = replicate the last border pixel value (recommended).",
                 fg='gray', font=("Arial", 8, "italic")).pack(anchor='w')

        f1 = tk.Frame(aug_frame); f1.pack(fill=tk.X, pady=2)
        tk.Checkbutton(f1, text="Horizontal Flip",
                       variable=self.aug_settings['flip_horizontal']).pack(side=tk.LEFT, padx=5)
        tk.Checkbutton(f1, text="Rotation (±30°)",
                       variable=self.aug_settings['rotation']).pack(side=tk.LEFT, padx=5)
        tk.Checkbutton(f1, text="Random Crop",
                       variable=self.aug_settings['random_crop']).pack(side=tk.LEFT, padx=5)
        tk.Label(f1, text="scale:").pack(side=tk.LEFT, padx=(5, 0))
        tk.Entry(f1, textvariable=self.aug_settings['crop_scale'], width=5).pack(side=tk.LEFT)
        f2 = tk.Frame(aug_frame); f2.pack(fill=tk.X, pady=2)
        tk.Checkbutton(f2, text="Color Jitter",
                       variable=self.aug_settings['color_jitter']).pack(side=tk.LEFT, padx=5)
        tk.Label(f2, text="b/c/s/h:").pack(side=tk.LEFT)
        for k in ['brightness', 'contrast', 'saturation', 'hue']:
            tk.Entry(f2, textvariable=self.aug_settings[k], width=4).pack(side=tk.LEFT)
        f3 = tk.Frame(aug_frame); f3.pack(fill=tk.X, pady=2)
        tk.Checkbutton(f3, text="Random Perspective",
                       variable=self.aug_settings['random_perspective']).pack(side=tk.LEFT, padx=5)
        tk.Label(f3, text="dist:").pack(side=tk.LEFT)
        tk.Entry(f3, textvariable=self.aug_settings['perspective_distortion'], width=5).pack(side=tk.LEFT)
        tk.Checkbutton(f3, text="Elastic",
                       variable=self.aug_settings['elastic_transform']).pack(side=tk.LEFT, padx=10)
        tk.Label(f3, text="α/σ:").pack(side=tk.LEFT)
        tk.Entry(f3, textvariable=self.aug_settings['elastic_alpha'], width=4).pack(side=tk.LEFT)
        tk.Entry(f3, textvariable=self.aug_settings['elastic_sigma'], width=4).pack(side=tk.LEFT)
        f4 = tk.Frame(aug_frame); f4.pack(fill=tk.X, pady=2)
        tk.Checkbutton(f4, text="JPEG Compression",
                       variable=self.aug_settings['jpeg_compression']).pack(side=tk.LEFT, padx=5)
        tk.Label(f4, text="q low/high:").pack(side=tk.LEFT)
        tk.Entry(f4, textvariable=self.aug_settings['jpeg_quality_low'], width=4).pack(side=tk.LEFT)
        tk.Entry(f4, textvariable=self.aug_settings['jpeg_quality_high'], width=4).pack(side=tk.LEFT)
        tk.Checkbutton(f4, text="Stretch V",
                       variable=self.aug_settings['stretch_vertical']).pack(side=tk.LEFT, padx=10)
        tk.Checkbutton(f4, text="Stretch H",
                       variable=self.aug_settings['stretch_horizontal']).pack(side=tk.LEFT, padx=5)

        reset_frame = tk.Frame(scrollable)
        reset_frame.pack(fill=tk.X, pady=10)
        tk.Button(reset_frame, text="Reset All Hyperparameters to Default",
                  command=self.reset_all_settings,
                  bg="orange", fg="white", font=("Arial", 10, "bold")).pack(pady=5)

        sys_frame = tk.LabelFrame(scrollable, text="System", padx=10, pady=10)
        sys_frame.pack(fill=tk.X, pady=5)
        tk.Label(sys_frame, text=f"CPU cores: {multiprocessing.cpu_count()}").pack(anchor='w')
        tk.Label(sys_frame, text=f"PyTorch threads: {torch.get_num_threads()}").pack(anchor='w')
        tk.Label(sys_frame, text=f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}").pack(anchor='w')

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._refresh_head_dims()

    def _refresh_head_dims(self):
        try:
            base = int(self.settings['rectified_base_channels'].get())
            nh = int(self.settings['attn_num_heads'].get())
        except Exception:
            return
        dims = compute_head_dims(base, (1, 2, 4), nh)
        flags = [self.settings['attn_level_0'].get(),
                 self.settings['attn_level_1'].get(),
                 self.settings['attn_level_2'].get()]
        for i, (ch, eff_nh, hd) in enumerate(dims):
            tag = "ON " if flags[i] else "off"
            self.head_dim_label_vars[i].set(
                f"L{i} [{tag}]  channels={ch:4d}  heads={eff_nh:2d}  head_dim={hd:4d}")

    def apply_text_encoder_preset(self):
        enc_type = self.settings['text_encoder_type'].get()
        size = self.settings['text_encoder_size'].get()
        config = get_encoder_config(enc_type, size)
        self.settings['cond_embed_dim'].set(config.get('embed_dim', self.settings['cond_embed_dim'].get()))
        if enc_type == 'BiGRU':
            self.settings['cond_hidden_size'].set(config.get('hidden_size', self.settings['cond_hidden_size'].get()))
            self.settings['cond_num_layers'].set(config.get('num_layers', self.settings['cond_num_layers'].get()))
        else:
            self.settings['cond_num_heads'].set(config.get('num_heads', self.settings['cond_num_heads'].get()))
            self.settings['cond_num_layers'].set(config.get('num_layers', self.settings['cond_num_layers'].get()))
            self.settings['cond_ff_dim'].set(config.get('ff_dim', self.settings['cond_ff_dim'].get()))
        self.settings['cond_dim'].set(config.get('cond_dim', self.settings['cond_dim'].get()))
        self.log_rectified(f"Applied {enc_type} {size} preset.")

    def reset_all_settings(self):
        defaults = {
            'img_size': 32, 'color_mode': 'rgb', 'rectified_base_channels': 64,
            'rectified_batch_size': 16, 'rectified_lr': 2e-4, 'cfg_dropout_prob': 0.1,
            'attn_level_0': True, 'attn_level_1': True, 'attn_level_2': True,
            'attn_num_heads': 4, 'attn_dropout': 0.0,
            'ema_enabled': False, 'ema_decay': 0.999, 'ema_mode': 'standard',
            'cond_enabled': False, 'text_encoder_type': 'BiGRU', 'text_encoder_size': 'small',
            'cond_embed_dim': 64, 'cond_hidden_size': 64, 'cond_num_layers': 2,
            'cond_num_heads': 4, 'cond_ff_dim': 256, 'cond_dim': 256, 'cond_text_max_len': 128,
            'cond_mode': 'shift',
            'preview_enabled': True, 'preview_steps': 50, 'preview_method': 'heun',
            'preview_epoch_freq': 5, 'ot_task': 'noise', 'training_mode': 'dataset',
            'blur_sigma': 2.0, 'bit_depth': 1, 'pixelation_size': 8,
            'padding_mode': 'constant', 'use_ot': False,
            'ot_method': 'sinkhorn', 'ot_sinkhorn_eps': 0.05, 'ot_sinkhorn_iters': 50,
            'source_noise_std': 0.0, 'pixelation_rescale_mode': 'nearest',
            'inpaint_mask_ratio': 0.3, 'otfm_alpha': 0.5, 'otfm_downsample': 4,
        }
        for k, v in defaults.items():
            self.settings[k].set(v)
        self.aug_settings['aug_padding_mode'].set('edge')
        self._refresh_head_dims()
        self.log_rectified("Settings reset to defaults.")

    # ---------------- Generate tab ----------------
    def setup_generation_tab(self):
        main = tk.Frame(self.generation_tab)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        tk.Label(main, text="Generate Images", font=("Arial", 14, "bold")).pack(pady=(0, 10))

        prompt_frame = tk.LabelFrame(main, text="Text Prompt (optional)", padx=5, pady=5)
        prompt_frame.pack(fill=tk.X, pady=(0, 10))
        self.gen_prompt = tk.Entry(prompt_frame, width=60)
        self.gen_prompt.insert(0, "a cute cat")
        self.gen_prompt.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        ctrl = tk.Frame(main)
        ctrl.pack(pady=5)
        tk.Label(ctrl, text="Number:").pack(side=tk.LEFT)
        self.gen_count = tk.IntVar(value=16)
        tk.Spinbox(ctrl, from_=1, to=64, textvariable=self.gen_count, width=5).pack(side=tk.LEFT, padx=5)
        tk.Label(ctrl, text="ODE Steps:").pack(side=tk.LEFT, padx=(10, 0))
        tk.Spinbox(ctrl, from_=1, to=200, textvariable=self.ode_steps, width=5).pack(side=tk.LEFT, padx=5)
        tk.Label(ctrl, text="Method:").pack(side=tk.LEFT, padx=(10, 0))
        ttk.Combobox(ctrl, textvariable=self.ode_method,
                     values=['euler', 'heun', 'rk4', 'midpoint'],
                     state='readonly', width=8).pack(side=tk.LEFT, padx=5)
        tk.Label(ctrl, text="CFG scale:").pack(side=tk.LEFT, padx=(10, 0))
        tk.Entry(ctrl, width=6, textvariable=self.cfg_scale).pack(side=tk.LEFT, padx=5)

        ctrl2 = tk.Frame(main)
        ctrl2.pack(pady=5)
        tk.Checkbutton(ctrl2, text="Live preview (progressive grid)",
                       variable=self.progressive_grid).pack(side=tk.LEFT, padx=5)
        tk.Label(ctrl2, text="Update every N steps:").pack(side=tk.LEFT, padx=(10, 0))
        tk.Spinbox(ctrl2, from_=1, to=50, textvariable=self.prog_interval,
                   width=5).pack(side=tk.LEFT, padx=5)
        self.generate_btn = tk.Button(ctrl2, text="Generate",
                                      command=self.generate_samples, bg="lightgreen")
        self.generate_btn.pack(side=tk.LEFT, padx=5)
        self.stop_prog_btn = tk.Button(ctrl2, text="Stop", command=self.stop_progressive,
                                       state=tk.DISABLED, bg="salmon")
        self.stop_prog_btn.pack(side=tk.LEFT, padx=5)

        scroll_frame = tk.LabelFrame(main, text="Results", padx=5, pady=5)
        scroll_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        canvas_frame = tk.Frame(scroll_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        self.gen_canvas = tk.Canvas(canvas_frame, bg='lightgray')
        vs = tk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.gen_canvas.yview)
        hs = tk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.gen_canvas.xview)
        self.gen_canvas.configure(yscrollcommand=vs.set, xscrollcommand=hs.set)
        vs.pack(side=tk.RIGHT, fill=tk.Y)
        hs.pack(side=tk.BOTTOM, fill=tk.X)
        self.gen_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.inner_frame = tk.Frame(self.gen_canvas)
        self.gen_canvas.create_window((0, 0), window=self.inner_frame, anchor='nw')
        self.inner_frame.bind('<Configure>',
                              lambda e: self.gen_canvas.configure(scrollregion=self.gen_canvas.bbox('all')))
        self.gen_info = tk.Label(main, text="", fg="blue")
        self.gen_info.pack()

    # ---------------- Inpaint tab ----------------
    def setup_inpaint_tab(self):
        main = tk.Frame(self.inpaint_tab)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        tk.Label(main, text="Inpainting — left-drag to paint mask, right-drag to erase",
                 font=("Arial", 12, "bold")).pack(pady=(0, 6))

        row0 = tk.Frame(main)
        row0.pack(fill=tk.X, pady=4)
        tk.Label(row0, text="Prompt:").pack(side=tk.LEFT, padx=(2, 4))
        tk.Entry(row0, textvariable=self.inpaint_prompt, width=60).pack(
            side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        tk.Label(row0, text="(used only if conditioning is enabled)",
                 fg='gray', font=("Arial", 8, "italic")).pack(side=tk.LEFT, padx=6)

        top = tk.Frame(main)
        top.pack(fill=tk.X, pady=4)
        tk.Button(top, text="Load Image", command=self.inpaint_load_image).pack(side=tk.LEFT, padx=2)
        tk.Button(top, text="Clear Mask", command=self.inpaint_clear_mask).pack(side=tk.LEFT, padx=2)
        tk.Button(top, text="Random Mask", command=self.inpaint_random_mask).pack(side=tk.LEFT, padx=2)
        tk.Button(top, text="Fill Entire", command=self.inpaint_fill_all).pack(side=tk.LEFT, padx=2)
        tk.Label(top, text="Brush:").pack(side=tk.LEFT, padx=(10, 0))
        tk.Spinbox(top, from_=1, to=64, textvariable=self.inpaint_brush_size,
                   width=4).pack(side=tk.LEFT, padx=2)

        top2 = tk.Frame(main)
        top2.pack(fill=tk.X, pady=4)
        tk.Label(top2, text="Steps:").pack(side=tk.LEFT, padx=(2, 0))
        tk.Spinbox(top2, from_=1, to=200, textvariable=self.inpaint_steps,
                   width=4).pack(side=tk.LEFT, padx=2)
        tk.Label(top2, text="Method:").pack(side=tk.LEFT, padx=(10, 0))
        ttk.Combobox(top2, textvariable=self.inpaint_method,
                     values=['euler', 'heun', 'rk4', 'midpoint'],
                     state='readonly', width=8).pack(side=tk.LEFT, padx=2)
        tk.Label(top2, text="CFG:").pack(side=tk.LEFT, padx=(10, 0))
        tk.Entry(top2, textvariable=self.inpaint_cfg, width=5).pack(side=tk.LEFT, padx=2)
        tk.Label(top2, text="Noise σ:").pack(side=tk.LEFT, padx=(10, 0))
        tk.Scale(top2, from_=0.0, to=1.5, resolution=0.05,
                 variable=self.inpaint_noise_sigma,
                 orient=tk.HORIZONTAL, length=120).pack(side=tk.LEFT)
        tk.Label(top2, textvariable=self.inpaint_noise_sigma,
                 width=5, font=("Courier", 9)).pack(side=tk.LEFT)

        top3 = tk.Frame(main)
        top3.pack(fill=tk.X, pady=4)
        tk.Checkbutton(top3, text="Composite updates",
                       variable=self.inpaint_composite_updates).pack(side=tk.LEFT, padx=4)
        tk.Label(top3, text="Full-edit from:").pack(side=tk.LEFT, padx=(10, 0))
        tk.Scale(top3, from_=0.0, to=1.0, resolution=0.05,
                 variable=self.inpaint_full_edit_from,
                 orient=tk.HORIZONTAL, length=140).pack(side=tk.LEFT)
        tk.Label(top3, textvariable=self.inpaint_full_edit_from,
                 width=5, font=("Courier", 9)).pack(side=tk.LEFT)
        tk.Checkbutton(top3, text="Live preview",
                       variable=self.inpaint_live_preview).pack(side=tk.LEFT, padx=(10, 4))
        tk.Button(top3, text="Inpaint", command=self.inpaint_run,
                  bg="lightgreen").pack(side=tk.LEFT, padx=8)
        tk.Button(top3, text="Stop", command=self.inpaint_stop,
                  bg="salmon").pack(side=tk.LEFT, padx=2)

        cf = tk.Frame(main)
        cf.pack(fill=tk.BOTH, expand=True, pady=6)
        left = tk.Frame(cf)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
        tk.Label(left, text="Draw Mask (L=paint, R=erase)").pack()
        self.inpaint_canvas = tk.Canvas(left, width=self.inpaint_display_size,
                                        height=self.inpaint_display_size, bg='black')
        self.inpaint_canvas.pack()

        right = tk.Frame(cf)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=4)
        tk.Label(right, text="Result (live)").pack()
        self.inpaint_output_canvas = tk.Canvas(right, width=self.inpaint_display_size,
                                               height=self.inpaint_display_size, bg='gray')
        self.inpaint_output_canvas.pack()

        self.inpaint_info = tk.Label(main, text="Load an image to begin.", fg="blue")
        self.inpaint_info.pack()

        self.inpaint_canvas.bind("<Button-1>", self.inpaint_start_paint)
        self.inpaint_canvas.bind("<B1-Motion>", self.inpaint_paint)
        self.inpaint_canvas.bind("<ButtonRelease-1>", self.inpaint_stop_paint)
        self.inpaint_canvas.bind("<Button-3>", self.inpaint_start_erase)
        self.inpaint_canvas.bind("<B3-Motion>", self.inpaint_erase)
        self.inpaint_canvas.bind("<ButtonRelease-3>", self.inpaint_stop_erase)

    # ---- inpaint helpers ----
    def inpaint_load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.jfif *.webp")])
        if not path:
            return
        img_size = self.settings['img_size'].get()
        color_mode = self.settings['color_mode'].get()
        try:
            if color_mode == 'rgb':
                pil = load_image_as_rgb(path).resize((img_size, img_size), Image.BICUBIC)
            else:
                pil = load_image_as_grayscale(path).resize((img_size, img_size), Image.BICUBIC)
        except Exception as e:
            self.inpaint_info.config(text=f"Load error: {e}")
            return
        self.inpaint_original_img = pil
        self.inpaint_mask = np.zeros((img_size, img_size), dtype=np.uint8)
        self.inpaint_redraw()
        self.inpaint_info.config(text=f"Loaded: {os.path.basename(path)}")

    def inpaint_clear_mask(self):
        if self.inpaint_original_img is None:
            return
        img_size = self.settings['img_size'].get()
        self.inpaint_mask = np.zeros((img_size, img_size), dtype=np.uint8)
        self.inpaint_redraw()
        self.inpaint_info.config(text="Mask cleared.")

    def inpaint_random_mask(self):
        if self.inpaint_original_img is None:
            return
        img_size = self.settings['img_size'].get()
        ratio = float(self.settings['inpaint_mask_ratio'].get())
        m = (np.random.rand(img_size, img_size) < ratio).astype(np.uint8) * 255
        self.inpaint_mask = m
        self.inpaint_redraw()
        self.inpaint_info.config(text="Random mask generated.")

    def inpaint_fill_all(self):
        if self.inpaint_original_img is None:
            return
        img_size = self.settings['img_size'].get()
        self.inpaint_mask = np.full((img_size, img_size), 255, dtype=np.uint8)
        self.inpaint_redraw()
        self.inpaint_info.config(text="Mask = entire image.")

    def inpaint_redraw(self):
        if self.inpaint_original_img is None:
            return
        img_size = self.settings['img_size'].get()
        orig = self.inpaint_original_img.resize((img_size, img_size), Image.NEAREST).convert('RGBA')
        mask_img = Image.fromarray(self.inpaint_mask, mode='L')
        overlay = Image.new('RGBA', (img_size, img_size), (255, 0, 0, 128))
        composite = Image.composite(overlay, orig, mask_img).convert('RGB')
        composite = composite.resize(
            (self.inpaint_display_size, self.inpaint_display_size), Image.NEAREST)
        self.inpaint_photo = ImageTk.PhotoImage(composite)
        self.inpaint_canvas.delete("all")
        self.inpaint_canvas.create_image(0, 0, anchor=tk.NW, image=self.inpaint_photo)
        self.inpaint_canvas.image = self.inpaint_photo

    def _canvas_to_img_coords(self, event):
        img_size = self.settings['img_size'].get()
        x = int(event.x * img_size / self.inpaint_display_size)
        y = int(event.y * img_size / self.inpaint_display_size)
        return max(0, min(img_size - 1, x)), max(0, min(img_size - 1, y))

    def inpaint_start_paint(self, event):
        if self.inpaint_original_img is None:
            return
        self.inpaint_erase_mode = False
        self.inpaint_last_xy = self._canvas_to_img_coords(event)

    def inpaint_paint(self, event):
        self._inpaint_stroke(event, erase=False)

    def inpaint_stop_paint(self, event):
        self.inpaint_last_xy = None

    def inpaint_start_erase(self, event):
        if self.inpaint_original_img is None:
            return
        self.inpaint_erase_mode = True
        self.inpaint_last_xy = self._canvas_to_img_coords(event)

    def inpaint_erase(self, event):
        self._inpaint_stroke(event, erase=True)

    def inpaint_stop_erase(self, event):
        self.inpaint_last_xy = None
        self.inpaint_erase_mode = False

    def _inpaint_stroke(self, event, erase):
        if self.inpaint_original_img is None or self.inpaint_last_xy is None:
            return
        x1, y1 = self.inpaint_last_xy
        x2, y2 = self._canvas_to_img_coords(event)
        mask_img = Image.fromarray(self.inpaint_mask, mode='L')
        draw = ImageDraw.Draw(mask_img)
        value = 0 if erase else 255
        draw.line([x1, y1, x2, y2], fill=value,
                  width=int(self.inpaint_brush_size.get()))
        self.inpaint_mask = np.array(mask_img, dtype=np.uint8)
        self.inpaint_last_xy = (x2, y2)
        self.inpaint_redraw()

    def inpaint_stop(self):
        self.inpaint_stop_flag = True
        self.inpaint_info.config(text="Stopping...")

    def inpaint_run(self):
        if self.rectified_model is None:
            self.inpaint_info.config(text="Model not loaded!")
            return
        if self.inpaint_original_img is None:
            self.inpaint_info.config(text="Load an image first.")
            return
        if self.inpaint_mask is None or self.inpaint_mask.max() == 0:
            self.inpaint_info.config(text="Draw a mask first.")
            return
        if self.inpaint_in_progress:
            return

        self.inpaint_in_progress = True
        self.inpaint_stop_flag = False
        self.inpaint_info.config(text="Inpainting...")
        self.root.update()

        steps = self.inpaint_steps.get()
        method = self.inpaint_method.get()
        cfg = self.inpaint_cfg.get()
        noise_sigma = float(self.inpaint_noise_sigma.get())
        composite = bool(self.inpaint_composite_updates.get())
        full_edit_from = float(self.inpaint_full_edit_from.get())
        live = bool(self.inpaint_live_preview.get())
        prompt = self.inpaint_prompt.get().strip()

        thread = threading.Thread(
            target=self._inpaint_thread,
            args=(steps, method, cfg, noise_sigma, composite, full_edit_from, live, prompt),
            daemon=True)
        thread.start()

    def _inpaint_thread(self, steps, method, cfg, noise_sigma,
                        composite, full_edit_from, live, prompt):
        try:
            img_size = self.settings['img_size'].get()
            color_mode = self.settings['color_mode'].get()
            in_channels = 3 if color_mode == 'rgb' else 1

            img = self.inpaint_original_img
            img = img.convert('RGB') if color_mode == 'rgb' else img.convert('L')

            tf = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,) * in_channels, (0.5,) * in_channels)
            ])
            img_t = tf(img).unsqueeze(0).to(self.rectified_model.device)
            mask_np = self.inpaint_mask.astype(np.float32) / 255.0
            mask_t = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(
                self.rectified_model.device)

            cond = None
            if (prompt
                    and self.settings['cond_enabled'].get()
                    and self.text_encoder is not None):
                text_indices = text_to_indices(
                    prompt, self.settings['cond_text_max_len'].get())
                text_tensor = torch.tensor([text_indices], dtype=torch.long,
                                           device=self.rectified_model.device)
                with torch.no_grad():
                    cond = self.text_encoder(text_tensor)
                self.log_rectified(f"Inpaint cond prompt: '{prompt}'")

            use_ema = self.settings['ema_enabled'].get()
            update_every = max(1, steps // 20)

            with torch.no_grad():
                for step_i, x in self.rectified_model.inpaint_step_by_step(
                        img_t, mask_t, steps=steps, method=method,
                        cfg_scale=cfg, use_ema=use_ema,
                        cond=cond, noise_sigma=noise_sigma,
                        composite_updates=composite,
                        full_edit_from=full_edit_from):
                    if self.inpaint_stop_flag:
                        break
                    if live and (step_i % update_every == 0 or step_i == steps):
                        preview = ((x + 1) / 2).clamp(0, 1).cpu()
                        if preview.shape[1] == 1:
                            pil = transforms.ToPILImage()(preview.squeeze(0).repeat(3, 1, 1))
                        else:
                            pil = transforms.ToPILImage()(preview.squeeze(0))
                        pil = pil.resize((self.inpaint_display_size, self.inpaint_display_size),
                                         Image.NEAREST)
                        self.root.after(0, self._inpaint_show_result, pil, step_i, steps)

            if not live:
                with torch.no_grad():
                    final = self.rectified_model.inpaint(
                        img_t, mask_t, steps=steps, method=method,
                        cfg_scale=cfg, use_ema=use_ema,
                        cond=cond, noise_sigma=noise_sigma,
                        composite_updates=composite,
                        full_edit_from=full_edit_from)
                preview = ((final + 1) / 2).clamp(0, 1).cpu()
                if preview.shape[1] == 1:
                    pil = transforms.ToPILImage()(preview.squeeze(0).repeat(3, 1, 1))
                else:
                    pil = transforms.ToPILImage()(preview.squeeze(0))
                pil = pil.resize((self.inpaint_display_size, self.inpaint_display_size),
                                 Image.NEAREST)
                self.root.after(0, self._inpaint_show_result, pil, steps, steps)

            self.root.after(0, lambda: self.inpaint_info.config(
                text="Inpainting complete." if not self.inpaint_stop_flag else "Stopped."))
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda: self.inpaint_info.config(text=f"Error: {e}"))
        finally:
            self.inpaint_in_progress = False

    def _inpaint_show_result(self, pil, step_i, total_steps):
        self.inpaint_output_photo = ImageTk.PhotoImage(pil)
        self.inpaint_output_canvas.delete("all")
        self.inpaint_output_canvas.create_image(
            self.inpaint_display_size // 2, self.inpaint_display_size // 2,
            image=self.inpaint_output_photo)
        self.inpaint_output_canvas.image = self.inpaint_output_photo
        self.inpaint_info.config(text=f"Step {step_i}/{total_steps}")

    # ---------------- Flow to Image tab ----------------
    def setup_flow_to_image_tab(self):
        main = tk.Frame(self.flow_to_image_tab)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        tk.Label(main, text="Flow to Image",
                 font=("Arial", 12, "bold")).pack(pady=(0, 10))

        top = tk.Frame(main); top.pack(fill=tk.X, pady=5)
        pf = tk.LabelFrame(top, text="Text Prompt (optional)", padx=5, pady=5)
        pf.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.flow_prompt = tk.Entry(pf, width=40)
        self.flow_prompt.insert(0, "a cute cat")
        self.flow_prompt.pack(fill=tk.X)
        bf = tk.Frame(top); bf.pack(side=tk.RIGHT)
        tk.Button(bf, text="Select Images", command=self.add_flow_images,
                  bg="lightblue").pack(side=tk.LEFT, padx=2)
        tk.Button(bf, text="Clear", command=self.clear_flow_images,
                  bg="lightcoral").pack(side=tk.LEFT, padx=2)
        self.flow_image_count_label = tk.Label(bf, text="0 images")
        self.flow_image_count_label.pack(side=tk.LEFT, padx=5)

        pf2 = tk.Frame(main); pf2.pack(pady=5)
        tk.Label(pf2, text="ODE Steps:").pack(side=tk.LEFT)
        self.flow_steps = tk.IntVar(value=50)
        tk.Spinbox(pf2, from_=1, to=200, textvariable=self.flow_steps, width=5).pack(side=tk.LEFT, padx=5)
        tk.Label(pf2, text="Method:").pack(side=tk.LEFT, padx=(10, 0))
        self.flow_method = tk.StringVar(value='euler')
        ttk.Combobox(pf2, textvariable=self.flow_method,
                     values=['euler', 'heun', 'rk4', 'midpoint'],
                     state='readonly', width=8).pack(side=tk.LEFT, padx=5)
        tk.Label(pf2, text="CFG scale:").pack(side=tk.LEFT, padx=(10, 0))
        self.flow_cfg = tk.DoubleVar(value=2.0)
        tk.Entry(pf2, width=6, textvariable=self.flow_cfg).pack(side=tk.LEFT, padx=5)
        self.apply_degradation_var = tk.BooleanVar(value=True)
        tk.Checkbutton(pf2, text="Apply degradation to input",
                       variable=self.apply_degradation_var).pack(side=tk.LEFT, padx=10)
        self.flow_btn = tk.Button(pf2, text="Run Flow",
                                  command=self.run_flow_to_image, bg="lightgreen")
        self.flow_btn.pack(side=tk.LEFT, padx=5)
        self.flow_stop_btn = tk.Button(pf2, text="Stop",
                                       command=self.stop_flow,
                                       state=tk.DISABLED, bg="salmon")
        self.flow_stop_btn.pack(side=tk.LEFT, padx=2)

        pf3 = tk.Frame(main); pf3.pack(pady=5)
        tk.Checkbutton(pf3, text="Live preview (progressive)",
                       variable=self.flow_live_preview).pack(side=tk.LEFT, padx=5)
        tk.Label(pf3, text="Update every N steps:").pack(side=tk.LEFT, padx=(10, 0))
        tk.Spinbox(pf3, from_=1, to=50, textvariable=self.flow_prog_interval,
                   width=5).pack(side=tk.LEFT, padx=5)

        result_frame = tk.LabelFrame(main, text="Results", padx=5, pady=5)
        result_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        canvas_frame = tk.Frame(result_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        self.flow_canvas = tk.Canvas(canvas_frame, bg='lightgray')
        vs = tk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.flow_canvas.yview)
        hs = tk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.flow_canvas.xview)
        self.flow_canvas.configure(yscrollcommand=vs.set, xscrollcommand=hs.set)
        vs.pack(side=tk.RIGHT, fill=tk.Y)
        hs.pack(side=tk.BOTTOM, fill=tk.X)
        self.flow_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.flow_inner_frame = tk.Frame(self.flow_canvas)
        self.flow_canvas.create_window((0, 0), window=self.flow_inner_frame, anchor='nw')
        self.flow_inner_frame.bind('<Configure>',
                                   lambda e: self.flow_canvas.configure(
                                       scrollregion=self.flow_canvas.bbox('all')))
        self.flow_info = tk.Label(main, text="", fg="blue")
        self.flow_info.pack()
        self.flow_image_paths = []

    def add_flow_images(self):
        files = filedialog.askopenfilenames(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.jfif *.webp")])
        for f in files:
            if f not in self.flow_image_paths:
                self.flow_image_paths.append(f)
        self.flow_image_count_label.config(text=f"{len(self.flow_image_paths)} images")

    def clear_flow_images(self):
        self.flow_image_paths = []
        self.flow_image_count_label.config(text="0 images")
        for w in self.flow_inner_frame.winfo_children():
            w.destroy()
        self.flow_info.config(text="Cleared.")

    def stop_flow(self):
        self.flow_stop_flag = True
        self.flow_progressive_active = False
        self.flow_stop_btn.config(state=tk.DISABLED)
        self.flow_info.config(text="Stopping...")

    def run_flow_to_image(self):
        if not self.rectified_model:
            self.flow_info.config(text="Model not loaded!")
            return
        if not self.flow_image_paths:
            self.flow_info.config(text="No images selected.")
            return
        for w in self.flow_inner_frame.winfo_children():
            w.destroy()

        prompt = self.flow_prompt.get().strip()
        cond = None
        if prompt and self.settings['cond_enabled'].get() and self.text_encoder is not None:
            text_indices = text_to_indices(prompt, self.settings['cond_text_max_len'].get())
            text_tensor = torch.tensor([text_indices] * len(self.flow_image_paths),
                                       dtype=torch.long, device=self.rectified_model.device)
            with torch.no_grad():
                cond = self.text_encoder(text_tensor)

        steps = self.flow_steps.get()
        method = self.flow_method.get()
        cfg_scale = self.flow_cfg.get()
        use_ema = self.settings['ema_enabled'].get()
        degrade = self.apply_degradation_var.get()

        self.flow_stop_flag = False
        self.flow_btn.config(state=tk.DISABLED)
        self.flow_stop_btn.config(state=tk.NORMAL)

        if self.flow_live_preview.get():
            self.flow_progressive_active = True
            threading.Thread(
                target=self._flow_progressive_thread,
                args=(self.flow_image_paths, cond, steps, method,
                      cfg_scale, use_ema, degrade),
                daemon=True).start()
        else:
            threading.Thread(
                target=self._flow_images_thread,
                args=(self.flow_image_paths, cond, steps, method,
                      cfg_scale, use_ema, degrade),
                daemon=True).start()

    def _prepare_flow_start_batch(self, image_paths, degrade):
        color_mode = self.settings['color_mode'].get()
        img_size = self.settings['img_size'].get()
        in_channels = 3 if color_mode == 'rgb' else 1
        tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * in_channels, (0.5,) * in_channels)
        ])
        imgs = []
        for p in image_paths:
            try:
                pil = load_image_as_rgb(p) if color_mode == 'rgb' \
                    else load_image_as_grayscale(p)
                imgs.append(tf(pil).unsqueeze(0).to(self.rectified_model.device))
            except Exception as e:
                self.log_rectified(f"Error loading {p}: {e}")
        if not imgs:
            return None
        start_batch = torch.cat(imgs, dim=0)
        if degrade:
            with torch.no_grad():
                start_batch = self.rectified_model._generate_source_from_target(start_batch)
        return start_batch

    def _flow_images_thread(self, image_paths, cond, steps, method,
                            cfg_scale, use_ema, degrade):
        try:
            start_batch = self._prepare_flow_start_batch(image_paths, degrade)
            if start_batch is None:
                self.root.after(0, lambda: self.flow_info.config(
                    text="No valid images loaded."))
                return
            n = start_batch.size(0)
            result = self.rectified_model.sample(
                n_samples=n, cond=cond, steps=steps, method=method,
                cfg_scale=cfg_scale, use_ema=use_ema, start_from=start_batch)
            self._display_flow_grid(result, steps, steps)
            self.root.after(0, lambda: self.flow_info.config(text="Flow completed."))
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda: self.flow_info.config(text=f"Error: {e}"))
        finally:
            self.root.after(0, lambda: self.flow_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.flow_stop_btn.config(state=tk.DISABLED))

    def _flow_progressive_thread(self, image_paths, cond, steps, method,
                                 cfg_scale, use_ema, degrade):
        try:
            start_batch = self._prepare_flow_start_batch(image_paths, degrade)
            if start_batch is None:
                self.root.after(0, lambda: self.flow_info.config(
                    text="No valid images loaded."))
                return
            n = start_batch.size(0)
            interval = max(1, int(self.flow_prog_interval.get()))

            gen = self.rectified_model.sample_step_by_step(
                n_samples=n, cond=cond, steps=steps, method=method,
                cfg_scale=cfg_scale, use_ema=use_ema, start_from=start_batch)

            for step_i, x in gen:
                if self.flow_stop_flag or not self.flow_progressive_active:
                    break
                if step_i % interval == 0 or step_i == steps:
                    self._display_flow_grid(x, step_i, steps)
                    time.sleep(0.02)

            if self.flow_stop_flag:
                self.root.after(0, lambda: self.flow_info.config(text="Stopped."))
            else:
                self.root.after(0, lambda: self.flow_info.config(text="Flow completed."))
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda: self.flow_info.config(text=f"Error: {e}"))
        finally:
            self.flow_progressive_active = False
            self.root.after(0, lambda: self.flow_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.flow_stop_btn.config(state=tk.DISABLED))

    def _display_flow_grid(self, x, current_step, total_steps):
        samples = ((x + 1) / 2).clamp(0, 1).cpu().numpy()
        n = len(samples)
        thumb = self.thumbnail_size
        gs = int(math.ceil(math.sqrt(n)))
        W = gs * thumb
        H = gs * thumb
        grid = Image.new('RGB', (W, H), color=(128, 128, 128))
        for i in range(n):
            row = i // gs
            col = i % gs
            if samples[i].shape[0] == 1:
                im = samples[i][0] * 255
                im = np.stack([im, im, im], axis=-1).astype(np.uint8)
            else:
                im = (samples[i].transpose(1, 2, 0) * 255).astype(np.uint8)
            pil = Image.fromarray(im).resize((thumb, thumb), Image.NEAREST)
            grid.paste(pil, (col * thumb, row * thumb))
        photo = ImageTk.PhotoImage(grid)
        self.root.after(0, lambda p=photo: self._update_flow_canvas(
            p, current_step, total_steps))

    def _update_flow_canvas(self, photo, step, total):
        for w in self.flow_inner_frame.winfo_children():
            w.destroy()
        label = tk.Label(self.flow_inner_frame, image=photo)
        label.image = photo
        label.pack()
        self.flow_inner_frame.update_idletasks()
        self.flow_canvas.configure(scrollregion=self.flow_canvas.bbox('all'))
        self.flow_info.config(text=f"Step {step}/{total}")

    # ==================== Logging ====================
    def log_dataset(self, msg): self.message_queue_dataset.put(msg)
    def log_rectified(self, msg): self.message_queue_rectified.put(msg)

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

    def process_messages_rectified(self):
        try:
            while True:
                msg = self.message_queue_rectified.get_nowait()
                self.rectified_log_text.insert(tk.END, f"{time.strftime('%H:%M:%S')} - {msg}\n")
                self.rectified_log_text.see(tk.END)
                self.status_label.config(text=msg[:50])
        except queue.Empty:
            pass
        self.root.after(100, self.process_messages_rectified)

    # ==================== Dataset actions ====================
    def add_images(self):
        files = filedialog.askopenfilenames(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.jfif *.webp *.bmp")])
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
                    full = os.path.join(root_dir, file)
                    if full not in self.image_paths:
                        self.image_paths.append(full)
                        self.image_listbox.insert(tk.END, os.path.basename(full))
                        count += 1
        self.log_dataset(f"Added {count} images from folder. Total: {len(self.image_paths)}")

    def clear_images(self):
        self.image_paths = []
        self.labels = []
        self.image_listbox.delete(0, tk.END)
        self.csv_status.config(text="No CSV loaded", fg="red")
        self.log_dataset("Cleared all images")

    def load_csv(self):
        fname = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if not fname:
            return
        self.csv_path = fname
        path_by_basename = {os.path.basename(p).lower(): p for p in self.image_paths}
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
                    if base in path_by_basename:
                        label_map.setdefault(path_by_basename[base], []).append(label)
        except Exception as e:
            self.log_rectified(f"CSV read error: {e}")
            return
        self.labels = []
        unknown = 0
        for path in self.image_paths:
            if path in label_map:
                self.labels.append(label_map[path])
            else:
                self.labels.append([os.path.splitext(os.path.basename(path))[0]])
                unknown += 1
        self.csv_status.config(
            text=f"CSV loaded: {len(self.image_paths)-unknown} matched, {unknown} filenames used",
            fg="green")
        self.log_rectified(f"CSV loaded: {len(self.image_paths)-unknown} matched.")

    def use_filenames_as_labels(self):
        self.labels = [[os.path.splitext(os.path.basename(p))[0]] for p in self.image_paths]
        self.csv_status.config(text="Using filenames as labels", fg="blue")

    def use_folders_as_labels(self):
        self.labels = [[os.path.basename(os.path.dirname(p)) or 'unknown']
                       for p in self.image_paths]
        self.csv_status.config(text="Using folder names as labels", fg="blue")

    # ==================== Model init / Training ====================
    def initialize_rectified_model(self):
        try:
            cond_enabled = self.settings['cond_enabled'].get()
            cond_dim = self.settings['cond_dim'].get() if cond_enabled else 1
            color_mode = self.settings['color_mode'].get()
            in_channels = 3 if color_mode == 'rgb' else 1
            img_size = self.settings['img_size'].get()

            attn_levels = [self.settings['attn_level_0'].get(),
                           self.settings['attn_level_1'].get(),
                           self.settings['attn_level_2'].get()]
            attn_num_heads = int(self.settings['attn_num_heads'].get())
            attn_dropout = float(self.settings['attn_dropout'].get())

            task = self.settings['ot_task'].get()
            training_mode = self.settings['training_mode'].get()
            padding_mode = self.settings['padding_mode'].get()

            cond_mode = self.settings['cond_mode'].get()
            if cond_mode not in ('shift', 'film', 'adagn', 'cross_attn'):
                cond_mode = 'shift'
            if cond_mode == 'cross_attn' and not cond_enabled:
                self.log_rectified(
                    "Note: cond_mode='cross_attn' with cond_enabled=False — "
                    "the condition is all-zeros; enable text conditioning for effect.")

            task_params = {
                'blur_sigma': self.settings['blur_sigma'].get(),
                'bit_depth': self.settings['bit_depth'].get(),
                'pixelation_size': self.settings['pixelation_size'].get(),
                'source_noise_std': self.settings['source_noise_std'].get(),
                'pixelation_rescale_mode': self.settings['pixelation_rescale_mode'].get(),
                'inpaint_mask_ratio': self.settings['inpaint_mask_ratio'].get(),
                'otfm_alpha': self.settings['otfm_alpha'].get(),
                'otfm_downsample': self.settings['otfm_downsample'].get(),
            }

            self.rectified_model = RectifiedFlowPixel(
                in_channels=in_channels, img_size=img_size,
                base_channels=self.settings['rectified_base_channels'].get(),
                cond_dim=cond_dim,
                attention_levels=attn_levels,
                attn_num_heads=attn_num_heads,
                attn_dropout=attn_dropout,
                task=task, task_params=task_params, training_mode=training_mode,
                padding_mode=padding_mode,
                cond_mode=cond_mode
            )
            self.rectified_model.use_ot = self.settings['use_ot'].get()
            self.rectified_model.ot_method = self.settings['ot_method'].get()
            self.rectified_model.ot_sinkhorn_eps = float(self.settings['ot_sinkhorn_eps'].get())
            self.rectified_model.ot_sinkhorn_iters = int(self.settings['ot_sinkhorn_iters'].get())
            for pg in self.rectified_model.optimizer.param_groups:
                pg['lr'] = self.settings['rectified_lr'].get()

            if self.settings['ema_enabled'].get():
                self.rectified_model.set_ema(
                    decay=self.settings['ema_decay'].get(),
                    mode=self.settings['ema_mode'].get())

            if cond_enabled:
                enc_type = self.settings['text_encoder_type'].get()
                if enc_type == 'BiGRU':
                    self.text_encoder = TextEncoder(
                        vocab_size=256,
                        embed_dim=self.settings['cond_embed_dim'].get(),
                        hidden_size=self.settings['cond_hidden_size'].get(),
                        num_layers=self.settings['cond_num_layers'].get(),
                        cond_dim=self.settings['cond_dim'].get())
                else:
                    self.text_encoder = TransformerTextEncoder(
                        vocab_size=256,
                        embed_dim=self.settings['cond_embed_dim'].get(),
                        num_heads=self.settings['cond_num_heads'].get(),
                        num_layers=self.settings['cond_num_layers'].get(),
                        ff_dim=self.settings['cond_ff_dim'].get(),
                        cond_dim=self.settings['cond_dim'].get(),
                        max_len=self.settings['cond_text_max_len'].get())
                self.text_encoder.to(self.rectified_model.device)
                self.rectified_model.optimizer = optim.Adam(
                    list(self.rectified_model.model.parameters()) +
                    list(self.text_encoder.parameters()),
                    lr=self.settings['rectified_lr'].get())

            self.log_rectified(
                f"Init OK. task={task}, mode={training_mode}, pad={padding_mode}, "
                f"attn={attn_levels}, heads={attn_num_heads}, cond_mode={cond_mode}, "
                f"OT={self.rectified_model.use_ot} ({self.rectified_model.ot_method}), "
                f"aug_pad={self.aug_settings['aug_padding_mode'].get()}")
            if task == 'otfm':
                f = int(self.settings['otfm_downsample'].get())
                s = int(self.settings['img_size'].get())
                prior = max(1, s // f)
                self.log_rectified(
                    f"OTFM prior: {s}×{s} → {prior}×{prior} "
                    f"(factor {f}, α={self.settings['otfm_alpha'].get():.2f})")
                if prior < 4:
                    self.log_rectified(
                        "Warning: OTFM prior < 4×4 — consider reducing the downsample factor.")
        except Exception as e:
            self.log_rectified(f"Init error: {e}")

    def start_rectified_training(self):
        if not self.image_paths:
            self.log_rectified("No images!")
            return
        if not self.rectified_model:
            self.log_rectified("Model not initialized!")
            return
        if self.training_rectified:
            self.log_rectified("Already training.")
            return
        if self.settings['cond_enabled'].get() and not self.labels:
            self.log_rectified("Conditional training requires labels.")
            return
        try:
            epochs = int(self.rectified_epoch_var.get())
        except Exception:
            self.log_rectified("Invalid epochs")
            return
        self.training_rectified = True
        self.current_epoch_rectified = 0
        self.rectified_start_time = time.time()
        threading.Thread(target=self.train_rectified_loop, args=(epochs,), daemon=True).start()
        self.log_rectified(f"Training started. Epochs={epochs}")

    def train_rectified_loop(self, epochs):
        try:
            current_batch_size = self.settings['rectified_batch_size'].get()
            current_lr = self.settings['rectified_lr'].get()
            current_ema_decay = self.settings['ema_decay'].get()
            current_ema_mode = self.settings['ema_mode'].get()
            current_ema_enabled = self.settings['ema_enabled'].get()
            current_use_ot = self.settings['use_ot'].get()
            current_ot_method = self.settings['ot_method'].get()
            current_source_noise_std = self.settings['source_noise_std'].get()
            current_pixelation_rescale_mode = self.settings['pixelation_rescale_mode'].get()

            img_size = self.settings['img_size'].get()
            color_mode = self.settings['color_mode'].get()
            cond_enabled = self.settings['cond_enabled'].get()
            text_max_len = self.settings['cond_text_max_len'].get()
            cfg_dropout = self.settings['cfg_dropout_prob'].get()
            preview_enabled = self.settings['preview_enabled'].get()
            preview_freq = self.settings['preview_epoch_freq'].get()
            aug_dict = {k: v.get() for k, v in self.aug_settings.items()}
            labels = self.labels if self.labels else [['']] * len(self.image_paths)

            dataset = ConditionalImageDataset(self.image_paths, labels, img_size,
                                              color_mode, aug_dict, text_max_len)
            loader = DataLoader(dataset, batch_size=current_batch_size, shuffle=True,
                                num_workers=0, pin_memory=False)

            for epoch in range(epochs):
                if not self.training_rectified:
                    break
                self.current_epoch_rectified = epoch

                nb = self.settings['rectified_batch_size'].get()
                nl = self.settings['rectified_lr'].get()
                nd = self.settings['ema_decay'].get()
                nm = self.settings['ema_mode'].get()
                ne = self.settings['ema_enabled'].get()
                no = self.settings['use_ot'].get()
                nom = self.settings['ot_method'].get()
                ns = self.settings['source_noise_std'].get()
                npr = self.settings['pixelation_rescale_mode'].get()

                if ns != current_source_noise_std or npr != current_pixelation_rescale_mode:
                    self.rectified_model.task_params['source_noise_std'] = ns
                    self.rectified_model.task_params['pixelation_rescale_mode'] = npr
                    current_source_noise_std = ns
                    current_pixelation_rescale_mode = npr

                if no != current_use_ot:
                    self.rectified_model.use_ot = no
                    current_use_ot = no
                    self.log_rectified(f"OT matching {'ON' if no else 'OFF'}")

                if nom != current_ot_method:
                    self.rectified_model.ot_method = nom
                    current_ot_method = nom
                    self.log_rectified(f"OT method updated to {nom}")

                self.rectified_model.ot_sinkhorn_eps = float(
                    self.settings['ot_sinkhorn_eps'].get())
                self.rectified_model.ot_sinkhorn_iters = int(
                    self.settings['ot_sinkhorn_iters'].get())

                self.rectified_model.task_params['otfm_alpha'] = float(
                    self.settings['otfm_alpha'].get())
                self.rectified_model.task_params['otfm_downsample'] = int(
                    self.settings['otfm_downsample'].get())

                if nl != current_lr:
                    for pg in self.rectified_model.optimizer.param_groups:
                        pg['lr'] = nl
                    current_lr = nl
                    self.log_rectified(f"LR updated to {nl}")

                if nb != current_batch_size:
                    current_batch_size = nb
                    loader = DataLoader(dataset, batch_size=current_batch_size,
                                        shuffle=True, num_workers=0, pin_memory=False)
                    self.log_rectified(f"Batch size updated to {current_batch_size}")

                if ne != current_ema_enabled:
                    current_ema_enabled = ne
                    if ne:
                        self.rectified_model.set_ema(decay=nd, mode=nm)
                    else:
                        self.rectified_model.use_ema = False
                        self.rectified_model.ema_model = None
                elif current_ema_enabled:
                    if nd != current_ema_decay:
                        current_ema_decay = nd
                        self.rectified_model.ema_decay = nd
                    if nm != current_ema_mode:
                        current_ema_mode = nm
                        self.rectified_model.ema_mode = nm

                epoch_loss = 0.0
                batches = 0
                for images, text_tensors in loader:
                    if not self.training_rectified:
                        break
                    x1 = images.to(self.rectified_model.device)
                    if cond_enabled and self.text_encoder is not None:
                        cond = self.text_encoder(
                            text_tensors.to(self.rectified_model.device))
                    else:
                        cond = None
                    loss = self.rectified_model.train_step(
                        x1, cond=cond, cfg_dropout_prob=cfg_dropout)
                    epoch_loss += loss
                    batches += 1
                avg = epoch_loss / batches if batches else 0.0
                elapsed = time.time() - self.rectified_start_time
                self.log_rectified(
                    f"Epoch {epoch+1}/{epochs} | Loss: {avg:.6f} | Time: {elapsed:.1f}s")
                if preview_enabled and (epoch + 1) % preview_freq == 0:
                    self.rectified_preview_with_prompt(use_ema=current_ema_enabled)

            self.training_rectified = False
            self.log_rectified("Training finished.")
        except Exception as e:
            self.log_rectified(f"Training error: {e}")
            import traceback
            traceback.print_exc()
            self.training_rectified = False

    def stop_rectified_training(self):
        self.training_rectified = False
        self.log_rectified("Training stopped.")

    # ---------------- Preview ----------------
    def rectified_preview_with_prompt(self, use_ema=None):
        if not self.rectified_model:
            return
        use_ema = use_ema if use_ema is not None else self.settings['ema_enabled'].get()
        steps = self.settings['preview_steps'].get()
        method = self.settings['preview_method'].get()
        n = 16
        training_mode = self.settings['training_mode'].get()

        if self.image_paths and training_mode in ['dataset', 'mixed']:
            try:
                color_mode = self.settings['color_mode'].get()
                img_size = self.settings['img_size'].get()
                text_max_len = self.settings['cond_text_max_len'].get()
                labels = self.labels if self.labels else [[''] for _ in self.image_paths]
                ds = ConditionalImageDataset(self.image_paths, labels, img_size,
                                             color_mode, {}, text_max_len)
                indices = np.random.choice(len(ds), min(n, len(ds)), replace=False)
                imgs = []
                texts = []
                for idx in indices:
                    img, text = ds[idx]
                    imgs.append(img)
                    texts.append(text)
                x1 = torch.stack(imgs).to(self.rectified_model.device)
                with torch.no_grad():
                    x0 = self.rectified_model._generate_source_from_target(x1)
                cond = None
                if self.settings['cond_enabled'].get() and self.text_encoder is not None:
                    text_tensor = torch.stack(texts).to(self.rectified_model.device)
                    with torch.no_grad():
                        cond = self.text_encoder(text_tensor)
                samples = self.rectified_model.sample(
                    n_samples=len(x1), cond=cond, steps=steps, method=method,
                    cfg_scale=1.0, use_ema=use_ema, start_from=x0)
                self._display_preview_grid(samples)
                self.log_rectified(f"Preview: reconstruction of {len(x1)} dataset images")
            except Exception as e:
                self.log_rectified(f"Preview error: {e}")
                self._preview_from_noise(use_ema, steps, method)
        else:
            self._preview_from_noise(use_ema, steps, method)

    def _preview_from_noise(self, use_ema, steps, method):
        prompt = self.test_prompt_entry.get().strip()
        n = 16
        cond = None
        cfg = 1.0
        if prompt and self.settings['cond_enabled'].get() and self.text_encoder is not None:
            text_indices = text_to_indices(prompt, self.settings['cond_text_max_len'].get())
            text_tensor = torch.tensor([text_indices] * n, dtype=torch.long,
                                       device=self.rectified_model.device)
            with torch.no_grad():
                cond = self.text_encoder(text_tensor)
            cfg = self.cfg_scale.get()
        samples = self.rectified_model.sample(n_samples=n, steps=steps, method=method,
                                              cond=cond, cfg_scale=cfg, use_ema=use_ema)
        self._display_preview_grid(samples)

    def _display_preview_grid(self, samples):
        samples = ((samples + 1) / 2).clamp(0, 1).cpu().numpy()
        n = len(samples)
        thumb = self.thumbnail_size
        gs = int(math.ceil(math.sqrt(n)))
        W = gs * thumb
        H = gs * thumb
        grid = Image.new('RGB', (W, H), color=(128, 128, 128))
        for i in range(n):
            row = i // gs
            col = i % gs
            if samples[i].shape[0] == 1:
                im = samples[i][0] * 255
                im = np.stack([im, im, im], axis=-1).astype(np.uint8)
            else:
                im = (samples[i].transpose(1, 2, 0) * 255).astype(np.uint8)
            pil = Image.fromarray(im).resize((thumb, thumb), Image.NEAREST)
            grid.paste(pil, (col * thumb, row * thumb))
        grid = grid.resize((256, 256), Image.NEAREST)
        self.rectified_preview_photo = ImageTk.PhotoImage(grid)
        self.rectified_preview_canvas.delete("all")
        self.rectified_preview_canvas.create_image(128, 128,
                                                   image=self.rectified_preview_photo)

    # ---------------- Save / Load ----------------
    def save_rectified_model(self):
        if not self.rectified_model:
            return
        fname = filedialog.asksaveasfilename(defaultextension=".pth",
                                             filetypes=[("PyTorch", "*.pth")])
        if not fname:
            return
        if self.rectified_model.use_ema and self.rectified_model.ema_model is not None:
            model_state = self.rectified_model.ema_model.state_dict()
        else:
            model_state = self.rectified_model.model.state_dict()

        save_dict = {
            'model_state': model_state,
            'optimizer_state': self.rectified_model.optimizer.state_dict(),
            'settings': {k: v.get() for k, v in self.settings.items()
                         if k.startswith('rectified') or k in
                         ['cfg_dropout_prob', 'text_encoder_type', 'text_encoder_size',
                          'ot_task', 'training_mode', 'blur_sigma', 'bit_depth',
                          'pixelation_size', 'padding_mode', 'ema_enabled', 'ema_decay',
                          'ema_mode', 'use_ot', 'ot_method', 'ot_sinkhorn_eps',
                          'ot_sinkhorn_iters', 'source_noise_std',
                          'pixelation_rescale_mode', 'inpaint_mask_ratio',
                          'otfm_alpha', 'otfm_downsample',
                          'attn_level_0', 'attn_level_1', 'attn_level_2',
                          'attn_num_heads', 'attn_dropout',
                          'img_size', 'color_mode', 'cond_mode']},
            'aug_settings': {k: v.get() for k, v in self.aug_settings.items()},
            'task': self.rectified_model.task,
            'task_params': self.rectified_model.task_params,
            'training_mode': self.rectified_model.training_mode,
            'padding_mode': self.rectified_model.model.padding_mode,
            'attn_levels': self.rectified_model.model.attention_levels,
            'attn_num_heads': self.rectified_model.model.attn_num_heads,
            'attn_dropout': self.rectified_model.model.attn_dropout,
            'ot_method': self.rectified_model.ot_method,
            'ot_sinkhorn_eps': self.rectified_model.ot_sinkhorn_eps,
            'ot_sinkhorn_iters': self.rectified_model.ot_sinkhorn_iters,
            'ema_enabled': self.rectified_model.use_ema,
            'ema_decay': self.rectified_model.ema_decay if self.rectified_model.use_ema else None,
            'ema_mode': self.rectified_model.ema_mode if self.rectified_model.use_ema else None,
            'use_ot': self.rectified_model.use_ot,
            # --- top-level duplicates for robustness / future loaders ---
            'img_size': self.settings['img_size'].get(),
            'color_mode': self.settings['color_mode'].get(),
            'cond_mode': self.rectified_model.cond_mode,
        }
        if self.text_encoder:
            save_dict['text_encoder_state'] = self.text_encoder.state_dict()
        torch.save(save_dict, fname)
        self.log_rectified(f"Saved to {fname}")

    def load_rectified_model(self):
        fname = filedialog.askopenfilename(filetypes=[("PyTorch", "*.pth")])
        if not fname:
            return
        try:
            ckpt = torch.load(fname, map_location='cpu')

            # ---------- restore settings (with backward-compat) ----------
            if 'settings' in ckpt:
                # Old checkpoints predate cond_mode; default to legacy 'shift'.
                if 'cond_mode' not in ckpt['settings']:
                    self.settings['cond_mode'].set('shift')
                for k, v in ckpt['settings'].items():
                    if k in self.settings:
                        try:
                            self.settings[k].set(v)
                        except Exception:
                            pass

            # Explicit top-level keys (newer checkpoints); honoured if present.
            if 'img_size' in ckpt:
                sz = ckpt['img_size']
                if isinstance(sz, (list, tuple)):
                    sz = sz[0]
                try:
                    self.settings['img_size'].set(int(sz))
                except Exception:
                    pass
            if 'color_mode' in ckpt:
                try:
                    self.settings['color_mode'].set(ckpt['color_mode'])
                except Exception:
                    pass
            if 'cond_mode' in ckpt:
                try:
                    self.settings['cond_mode'].set(ckpt['cond_mode'])
                except Exception:
                    pass

            # Infer color_mode from init_conv weights if absent (very old checkpoints).
            if ('color_mode' not in ckpt.get('settings', {})
                    and 'color_mode' not in ckpt
                    and 'model_state' in ckpt):
                for k, v in ckpt['model_state'].items():
                    if 'init_conv.weight' in k:
                        in_ch = v.shape[1]
                        if in_ch == 3:
                            self.settings['color_mode'].set('rgb')
                        elif in_ch == 1:
                            self.settings['color_mode'].set('grayscale')
                        break

            # Infer base channels if absent (very old checkpoints).
            if ('rectified_base_channels' not in ckpt.get('settings', {})
                    and 'model_state' in ckpt):
                for k, v in ckpt['model_state'].items():
                    if k.endswith('init_conv.weight') and k.startswith('model.'):
                        self.settings['rectified_base_channels'].set(int(v.shape[0]))
                        break

            task = ckpt.get('task', 'noise')
            task_params = ckpt.get('task_params', {})
            training_mode = ckpt.get('training_mode', 'dataset')
            padding_mode = ckpt.get('padding_mode', 'constant')
            self.settings['ot_task'].set(task)
            self.settings['training_mode'].set(training_mode)
            self.settings['padding_mode'].set(padding_mode)

            for key in ['blur_sigma', 'bit_depth', 'pixelation_size', 'source_noise_std',
                        'pixelation_rescale_mode', 'inpaint_mask_ratio',
                        'otfm_alpha', 'otfm_downsample']:
                if key in task_params:
                    try:
                        self.settings[key].set(task_params[key])
                    except Exception:
                        pass

            if 'ot_method' in ckpt:
                self.settings['ot_method'].set(ckpt['ot_method'])
            if 'ot_sinkhorn_eps' in ckpt:
                self.settings['ot_sinkhorn_eps'].set(ckpt['ot_sinkhorn_eps'])
            if 'ot_sinkhorn_iters' in ckpt:
                self.settings['ot_sinkhorn_iters'].set(ckpt['ot_sinkhorn_iters'])
            if 'ema_enabled' in ckpt:
                self.settings['ema_enabled'].set(ckpt['ema_enabled'])
            if 'ema_decay' in ckpt and ckpt['ema_decay'] is not None:
                self.settings['ema_decay'].set(ckpt['ema_decay'])
            if 'ema_mode' in ckpt and ckpt['ema_mode'] is not None:
                self.settings['ema_mode'].set(ckpt['ema_mode'])
            if 'use_ot' in ckpt:
                self.settings['use_ot'].set(ckpt['use_ot'])
            if 'attn_levels' in ckpt and len(ckpt['attn_levels']) >= 3:
                self.settings['attn_level_0'].set(bool(ckpt['attn_levels'][0]))
                self.settings['attn_level_1'].set(bool(ckpt['attn_levels'][1]))
                self.settings['attn_level_2'].set(bool(ckpt['attn_levels'][2]))
            if 'attn_num_heads' in ckpt:
                self.settings['attn_num_heads'].set(ckpt['attn_num_heads'])
            if 'attn_dropout' in ckpt:
                self.settings['attn_dropout'].set(ckpt['attn_dropout'])

            self.initialize_rectified_model()
            self.rectified_model.model.load_state_dict(ckpt['model_state'])
            try:
                self.rectified_model.optimizer.load_state_dict(ckpt['optimizer_state'])
            except Exception:
                pass
            if self.rectified_model.use_ema and self.rectified_model.ema_model is not None:
                self.rectified_model.ema_model.load_state_dict(
                    self.rectified_model.model.state_dict())
            if 'text_encoder_state' in ckpt and self.text_encoder:
                self.text_encoder.load_state_dict(ckpt['text_encoder_state'])
            self.log_rectified(
                f"Loaded from {fname}  "
                f"(img_size={self.settings['img_size'].get()}, "
                f"color_mode={self.settings['color_mode'].get()}, "
                f"cond_mode={self.settings['cond_mode'].get()})")
        except Exception as e:
            self.log_rectified(f"Load error: {e}")

    # ==================== Generation ====================
    def generate_samples(self):
        if not self.rectified_model:
            self.gen_info.config(text="Model not loaded!")
            return
        if self.settings['cond_enabled'].get() and self.text_encoder is None:
            self.gen_info.config(text="Text encoder not initialized!")
            return
        n = self.gen_count.get()
        steps = self.ode_steps.get()
        method = self.ode_method.get()
        prompt = self.gen_prompt.get().strip()
        cfg_scale = self.cfg_scale.get()
        use_ema = self.settings['ema_enabled'].get()
        cond = None
        if prompt and self.settings['cond_enabled'].get() and self.text_encoder:
            text_indices = text_to_indices(prompt, self.settings['cond_text_max_len'].get())
            text_tensor = torch.tensor([text_indices] * n, dtype=torch.long,
                                       device=self.rectified_model.device)
            with torch.no_grad():
                cond = self.text_encoder(text_tensor)
        else:
            cfg_scale = 1.0

        if self.progressive_grid.get():
            self.start_progressive(n, steps, method, cond, cfg_scale, use_ema)
        else:
            self.generate_btn.config(state=tk.DISABLED)
            self.gen_info.config(text="Generating...")
            self.root.update()
            threading.Thread(target=self._generate_thread,
                             args=(n, steps, method, cond, cfg_scale, use_ema),
                             daemon=True).start()

    def start_progressive(self, n, steps, method, cond, cfg_scale, use_ema):
        self.progressive_active = True
        self.generate_btn.config(state=tk.DISABLED)
        self.stop_prog_btn.config(state=tk.NORMAL)
        self.gen_info.config(text="Progressive generation...")
        threading.Thread(target=self._progressive_thread,
                         args=(n, steps, method, cond, cfg_scale, use_ema),
                         daemon=True).start()

    def stop_progressive(self):
        self.progressive_active = False
        self.stop_prog_btn.config(state=tk.DISABLED)
        self.generate_btn.config(state=tk.NORMAL)
        self.gen_info.config(text="Stopped.")

    def _generate_thread(self, n, steps, method, cond, cfg_scale, use_ema):
        try:
            samples = self.rectified_model.sample(n_samples=n, steps=steps, method=method,
                                                  cond=cond, cfg_scale=cfg_scale,
                                                  use_ema=use_ema)
            samples = ((samples + 1) / 2).clamp(0, 1).cpu().numpy()
            grid = self._numpy_to_grid(samples, n)
            self.root.after(0, lambda: self._display_generated(grid))
        except Exception as e:
            self.root.after(0, lambda: self.gen_info.config(text=f"Error: {e}"))
        finally:
            self.root.after(0, lambda: self.generate_btn.config(state=tk.NORMAL))

    def _progressive_thread(self, n, steps, method, cond, cfg_scale, use_ema):
        try:
            interval = self.prog_interval.get()
            generator = self.rectified_model.sample_step_by_step(
                n_samples=n, steps=steps, method=method, cond=cond,
                cfg_scale=cfg_scale, use_ema=use_ema)
            for step_idx, x in generator:
                if not self.progressive_active:
                    break
                if step_idx % interval == 0 or step_idx == steps:
                    samples = ((x + 1) / 2).clamp(0, 1).cpu().numpy()
                    grid = self._numpy_to_grid(samples, n)
                    self.root.after(0, lambda g=grid, s=step_idx: self._update_progressive(g, s))
                    time.sleep(0.05)
            if not self.progressive_active:
                self.root.after(0, lambda: self.gen_info.config(text="Stopped."))
            else:
                self.root.after(0, lambda: self.gen_info.config(text="Progressive finished."))
        except Exception as e:
            self.root.after(0, lambda: self.gen_info.config(text=f"Error: {e}"))
        finally:
            self.progressive_active = False
            self.root.after(0, lambda: self.generate_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.stop_prog_btn.config(state=tk.DISABLED))

    def _numpy_to_grid(self, samples, n):
        thumb = self.thumbnail_size
        gs = int(math.ceil(math.sqrt(n)))
        W = gs * thumb
        H = gs * thumb
        grid = Image.new('RGB', (W, H), color=(128, 128, 128))
        for i in range(min(n, len(samples))):
            row = i // gs
            col = i % gs
            if samples[i].shape[0] == 1:
                im = samples[i][0] * 255
                im = np.stack([im, im, im], axis=-1).astype(np.uint8)
            else:
                im = (samples[i].transpose(1, 2, 0) * 255).astype(np.uint8)
            pil = Image.fromarray(im).resize((thumb, thumb), Image.NEAREST)
            grid.paste(pil, (col * thumb, row * thumb))
        return grid

    def _display_generated(self, grid_img):
        for w in self.inner_frame.winfo_children():
            w.destroy()
        self.gen_photo = ImageTk.PhotoImage(grid_img)
        label = tk.Label(self.inner_frame, image=self.gen_photo)
        label.image = self.gen_photo
        label.pack()
        self.inner_frame.update_idletasks()
        self.gen_canvas.configure(scrollregion=self.gen_canvas.bbox('all'))
        self.gen_info.config(text="Generation complete.")

    def _update_progressive(self, grid_img, step_idx):
        for w in self.inner_frame.winfo_children():
            w.destroy()
        self.prog_photo = ImageTk.PhotoImage(grid_img)
        label = tk.Label(self.inner_frame, image=self.prog_photo)
        label.image = self.prog_photo
        label.pack()
        self.inner_frame.update_idletasks()
        self.gen_canvas.configure(scrollregion=self.gen_canvas.bbox('all'))
        self.gen_info.config(text=f"ODE step {step_idx}/{self.ode_steps.get()}")


# ==================== Main ====================

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    root = tk.Tk()
    app = RectifiedFlowApp(root)
    root.mainloop()