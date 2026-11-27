# dream_pixel_gui.py
"""
Dream Pixel LDM — World Model GUI (pixel space, no VAE).

Tabs: Dataset | Flow Settings | Flow Training | Augmentation | Settings |
      Inference | Play

Changes vs dream_ldm_gui.py:
  * No VAE tabs/settings/training/preview — flow operates directly on pixels.
  * Configurable number of down/up blocks via ``channel_mult`` (flow settings).
  * Flow head-dim display shows how many levels are actually used + bottleneck.
"""
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
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

from dream_pixel_api import (
    RectifiedFlowPixelVideo, VideoFrameDataset,
    ButtonEncoder, load_video_frames, load_video_fps,
    load_button_csv, buttons_at_time, collect_button_names,
    load_image_as_rgb, load_image_as_grayscale,
    VIDEO_EXTS, IMAGE_EXTS, _HAS_CV2, _HAS_IMAGEIO,
)
from flow_api import compute_head_dims


AUG_DEFAULTS = {
    'flip_horizontal': True, 'rotation': False, 'random_crop': False, 'crop_scale': 0.8,
    'color_jitter': False, 'brightness': 0.2, 'contrast': 0.2, 'saturation': 0.2, 'hue': 0.1,
    'random_perspective': False, 'perspective_distortion': 0.1,
    'elastic_transform': False, 'elastic_alpha': 30, 'elastic_sigma': 3,
    'jpeg_compression': False, 'jpeg_quality_low': 50, 'jpeg_quality_high': 95,
    'stretch_vertical': False, 'stretch_horizontal': False,
    'aug_padding_mode': 'edge',
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


_KEY_CHOICES = [
    ('', '(none)'),
    ('up', 'Up'), ('down', 'Down'), ('left', 'Left'), ('right', 'Right'),
    ('space', 'Space'), ('return', 'Return'), ('tab', 'Tab'),
    ('escape', 'Escape'), ('backspace', 'Backspace'),
    ('insert', 'Insert'), ('delete', 'Delete'),
    ('home', 'Home'), ('end', 'End'),
    ('pageup', 'PageUp'), ('pagedown', 'PageDown'),
]
for _c in 'abcdefghijklmnopqrstuvwxyz':
    _KEY_CHOICES.append((_c, _c.upper()))
for _c in '0123456789':
    _KEY_CHOICES.append((_c, _c))
for _i in range(1, 13):
    _KEY_CHOICES.append((f'f{_i}', f'F{_i}'))

_KEY_TO_DISPLAY = {k: d for k, d in _KEY_CHOICES}
_DISPLAY_TO_KEY = {d: k for k, d in _KEY_CHOICES}
_KEY_DISPLAY_VALUES = [d for _, d in _KEY_CHOICES]

_PURE_MOD_KEYS = {
    'shift_l', 'shift_r', 'control_l', 'control_r',
    'alt_l', 'alt_r', 'super_l', 'super_r', 'caps_lock',
}


class DreamPixelLDMApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dream Pixel LDM — Button-Conditioned World Model (pixel)")
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.root.geometry(f"{max(1100, int(sw*0.85))}x{max(750, int(sh*0.85))}")
        self.root.minsize(1000, 700)

        self.sources = []
        self.button_names = []
        self.button_to_idx = {}
        self.rectified_model = None
        self.button_encoder = None
        self.training_rectified = False
        self.message_queue_rectified = queue.Queue()

        self.global_settings = {
            'img_size': tk.IntVar(value=32),
            'color_mode': tk.StringVar(value='rgb'),
        }
        self.flow_settings = {
            'flow_base_channels': tk.IntVar(value=32),
            'flow_padding_mode': tk.StringVar(value='constant'),
            'flow_history': tk.IntVar(value=1),
            'flow_batch_size': tk.IntVar(value=8),
            'flow_lr': tk.DoubleVar(value=2e-4),
            'flow_num_workers': tk.IntVar(value=0),
            'flow_cfg_dropout': tk.DoubleVar(value=0.1),
            'flow_use_ot': tk.BooleanVar(value=False),
            'channel_mult': tk.StringVar(value='1,2,4'),
            'attn_level_0': tk.BooleanVar(value=True),
            'attn_level_1': tk.BooleanVar(value=True),
            'attn_level_2': tk.BooleanVar(value=True),
            'attn_num_heads': tk.IntVar(value=4),
            'attn_dropout': tk.DoubleVar(value=0.0),
            'ema_enabled': tk.BooleanVar(value=False),
            'ema_decay': tk.DoubleVar(value=0.999),
            'ema_mode': tk.StringVar(value='standard'),
            'preview_enabled': tk.BooleanVar(value=True),
            'preview_epoch_freq': tk.IntVar(value=5),
            'preview_steps': tk.IntVar(value=20),
            'preview_num_frames': tk.IntVar(value=8),
            'preview_method': tk.StringVar(value='heun'),
            'cond_mode': tk.StringVar(value='shift'),
            'button_embed_dim': tk.IntVar(value=128),
        }

        self.flow_aug_settings = default_aug_dict()

        self.ode_steps = tk.IntVar(value=50)
        self.ode_method = tk.StringVar(value='euler')
        self.cfg_scale = tk.DoubleVar(value=1.0)
        self.thumbnail_size = 96

        self.flow_head_dim_var = tk.StringVar(value="—")

        # Inference tab state
        self.inf_mode = tk.StringVar(value='image')
        self.inf_image_path = None
        self.inf_video_path = None
        self.inf_video_frames = []
        self.inf_video_fps = 30.0
        self.inf_frame_index = tk.IntVar(value=-1)
        self.inf_button_source = tk.StringVar(value='csv')
        self.inf_button_csv_path = None
        self.inf_button_rows = []
        self.inf_manual_sequence = tk.StringVar(value="A,,A,,A")
        self.inf_hold_buttons = tk.StringVar(value="")
        self.inf_num_frames = tk.IntVar(value=16)
        self.inf_steps = tk.IntVar(value=50)
        self.inf_method = tk.StringVar(value='euler')
        self.inf_cfg = tk.DoubleVar(value=1.0)
        self.inf_live = tk.BooleanVar(value=True)
        self.inf_stop_flag = False
        self.inf_in_progress = False
        self.inf_context_photo = None
        self.inf_gen_photo = None
        self.inf_result_pils = []
        self.inf_playback_active = False
        self.inf_playback_idx = 0
        self.inf_playback_after = None
        self.playback_fps = tk.IntVar(value=10)

        self.button_to_key = {}

        # Play tab state
        self.play_ctx_mode = tk.StringVar(value='image')
        self.play_ctx_image_path = None
        self.play_ctx_video_path = None
        self.play_ctx_video_frames = []
        self.play_frame_index = tk.IntVar(value=-1)
        self.play_steps = tk.IntVar(value=20)
        self.play_method = tk.StringVar(value='euler')
        self.play_cfg = tk.DoubleVar(value=1.0)
        self.play_advance_mode = tk.BooleanVar(value=True)
        self.play_live_preview = tk.BooleanVar(value=True)
        self.play_running = False
        self.play_stop_flag = False
        self.play_advance_trigger = False
        self.play_context = None
        self.play_pressed_keys = {}
        self.play_frame_count = 0
        self.play_queue = None
        self.play_photo = None
        self.key_to_button = {}
        self._kb_press_id = None
        self._kb_release_id = None

        self.setup_gui()
        self.root.after(100, self.process_messages_rectified)

    # ---------- Helpers ----------
    def _parse_channel_mult(self):
        try:
            s = self.flow_settings['channel_mult'].get().strip()
            if not s:
                return (1, 2, 4)
            vals = [int(x.strip()) for x in s.split(',') if x.strip()]
            if not vals or any(v <= 0 for v in vals):
                return (1, 2, 4)
            return tuple(vals)
        except Exception:
            return (1, 2, 4)

    def _levels_used(self):
        """Mirror the VelocityUNet num_down computation."""
        channel_mult = self._parse_channel_mult()
        img_size = self.global_settings['img_size'].get()
        num_down = 0
        sz = img_size
        while sz >= 2 and num_down < len(channel_mult) - 1:
            sz //= 2
            num_down += 1
        return num_down + 1, sz

    # ---------- GUI ----------
    def setup_gui(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        tabs = [
            ('Dataset',              self.setup_dataset_tab),
            ('Flow Settings',        self.setup_flow_settings_tab),
            ('Flow Training',        self.setup_flow_training_tab),
            ('Augmentation',         self.setup_augmentation_tab),
            ('Settings',             self.setup_settings_tab),
            ('Inference',            self.setup_inference_tab),
            ('Play',                 self.setup_play_tab),
        ]
        for name, builder in tabs:
            frame = ttk.Frame(self.notebook)
            self.notebook.add(frame, text=name)
            builder(frame)
        self.status_label = tk.Label(self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

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

    # ---------- Settings tab ----------
    def setup_settings_tab(self, tab):
        inner = self._make_scrollable(tab)
        wrap = tk.Frame(inner); wrap.pack(fill=tk.X, padx=20, pady=20)
        tk.Label(wrap, text="Settings", font=("Arial", 14, "bold")).pack(pady=(0, 15))

        bf = tk.LabelFrame(wrap, text="Button Bindings — keyboard key → button name",
                           padx=10, pady=10)
        bf.pack(fill=tk.X, pady=5)
        tk.Label(bf,
                 text=("While Play is running, hold a bound key to keep its button "
                       "active. Bindings survive vocabulary reloads; unbound buttons "
                       "simply stay off."),
                 fg='gray', font=("Arial", 8, "italic"),
                 wraplength=700, justify=tk.LEFT).pack(anchor='w', pady=(0, 6))

        self.bindings_frame = tk.Frame(bf)
        self.bindings_frame.pack(fill=tk.X, pady=4)

        btn_row = tk.Frame(bf); btn_row.pack(fill=tk.X, pady=(6, 0))
        tk.Button(btn_row, text="Refresh from Vocabulary",
                  command=self._refresh_bindings_ui).pack(side=tk.LEFT, padx=(0, 6))
        tk.Button(btn_row, text="Clear All Bindings",
                  command=self._clear_all_bindings).pack(side=tk.LEFT)

        self._refresh_bindings_ui()

    def _refresh_bindings_ui(self):
        if not hasattr(self, 'bindings_frame'):
            return
        for w in self.bindings_frame.winfo_children():
            w.destroy()
        if not self.button_names:
            tk.Label(self.bindings_frame,
                     text="(no buttons — load videos with button CSVs first)",
                     fg="gray").pack(anchor='w')
            return

        hdr = tk.Frame(self.bindings_frame); hdr.pack(fill=tk.X, pady=(0, 2))
        tk.Label(hdr, text="Button name", width=24, anchor='w',
                 font=('', 9, 'bold')).pack(side=tk.LEFT)
        tk.Label(hdr, text="Key", width=14, anchor='w',
                 font=('', 9, 'bold')).pack(side=tk.LEFT)

        for name in self.button_names:
            row = tk.Frame(self.bindings_frame); row.pack(fill=tk.X, pady=1)
            tk.Label(row, text=name, width=24, anchor='w').pack(side=tk.LEFT)
            current_key = self.button_to_key.get(name, '')
            current_disp = _KEY_TO_DISPLAY.get(current_key, '(none)')
            var = tk.StringVar(value=current_disp)
            combo = ttk.Combobox(row, textvariable=var,
                                 values=_KEY_DISPLAY_VALUES,
                                 state='readonly', width=14)
            combo.pack(side=tk.LEFT)

            def _commit(n=name, v=var):
                key = _DISPLAY_TO_KEY.get(v.get(), '')
                if not key:
                    self.button_to_key.pop(n, None)
                else:
                    for other, ok in list(self.button_to_key.items()):
                        if ok == key and other != n:
                            del self.button_to_key[other]
                    self.button_to_key[n] = key
                self.root.after(10, self._refresh_bindings_ui)

            combo.bind('<<ComboboxSelected>>', lambda e, c=_commit: c())

    def _clear_all_bindings(self):
        self.button_to_key = {}
        self._refresh_bindings_ui()

    # ---------- Dataset tab ----------
    def setup_dataset_tab(self, tab):
        main = tk.Frame(tab); main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        left = tk.Frame(main, width=400); left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left.pack_propagate(False)
        tk.Label(left, text="Sources", font=("Arial", 12, "bold")).pack(pady=(0, 6))

        sf = tk.LabelFrame(left, text="Sources (videos + button CSVs)", padx=5, pady=5)
        sf.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        bf = tk.Frame(sf); bf.pack(fill=tk.X, pady=2)
        tk.Button(bf, text="Add Videos (+ CSV)", command=self.add_videos,
                  width=22, bg="lightgreen").pack(side=tk.LEFT, padx=1)
        tk.Button(bf, text="Add Images", command=self.add_images, width=15).pack(side=tk.LEFT, padx=1)
        bf2 = tk.Frame(sf); bf2.pack(fill=tk.X, pady=2)
        tk.Button(bf2, text="Attach CSV to Selected", command=self.attach_csv_to_selected,
                  width=22).pack(side=tk.LEFT, padx=1)
        tk.Button(bf2, text="Clear All", command=self.clear_all, width=15).pack(side=tk.LEFT, padx=1)

        stride_f = tk.Frame(sf); stride_f.pack(fill=tk.X, pady=4)
        tk.Label(stride_f, text="Frame stride:").pack(side=tk.LEFT)
        self.frame_stride = tk.IntVar(value=1)
        ttk.Spinbox(stride_f, from_=1, to=64, textvariable=self.frame_stride,
                    width=5).pack(side=tk.LEFT, padx=5)

        self.source_listbox = tk.Listbox(sf, height=14)
        self.source_listbox.pack(fill=tk.BOTH, expand=True, pady=2)

        if not (_HAS_CV2 or _HAS_IMAGEIO):
            tk.Label(sf, text="Install 'opencv-python' or 'imageio[ffmpeg]'",
                     fg="orange").pack(fill=tk.X, pady=2)

        vf = tk.LabelFrame(left, text="Button Vocabulary", padx=5, pady=5)
        vf.pack(fill=tk.X, pady=5)
        self.vocab_label = tk.Label(vf, text="(empty — load a CSV first)",
                                    fg="gray", anchor='w', justify=tk.LEFT, wraplength=380)
        self.vocab_label.pack(fill=tk.X)
        tk.Button(vf, text="Rebuild Vocabulary from Sources",
                  command=self.rebuild_vocab).pack(fill=tk.X, pady=(4, 0))

        right = tk.Frame(main); right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        lf = tk.LabelFrame(right, text="Log", padx=5, pady=5)
        lf.pack(fill=tk.BOTH, expand=True)
        self.dataset_log_text = tk.Text(lf, height=20, font=("Courier", 9))
        self.dataset_log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = tk.Scrollbar(lf, command=self.dataset_log_text.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.dataset_log_text.config(yscrollcommand=sb.set)

    # ---------- Flow Settings ----------
    def setup_flow_settings_tab(self, tab):
        inner = self._make_scrollable(tab)
        wrap = tk.Frame(inner); wrap.pack(fill=tk.X, padx=20, pady=20)
        tk.Label(wrap, text="Flow Settings (pixel)",
                 font=("Arial", 14, "bold")).pack(pady=(0, 15))

        gf = tk.LabelFrame(wrap, text="Global Image Settings", padx=10, pady=10)
        gf.pack(fill=tk.X, pady=5)
        self._combo(gf, "Color mode:", self.global_settings['color_mode'], ['rgb', 'grayscale'])
        self._spin(gf, "Image size:", self.global_settings['img_size'], 16, 128)
        self.global_settings['img_size'].trace_add('write',
            lambda *_: self._refresh_flow_head_dims())

        uf = tk.LabelFrame(wrap, text="Flow UNet (pixel)", padx=10, pady=10)
        uf.pack(fill=tk.X, pady=5)
        self._spin(uf, "Base channels:", self.flow_settings['flow_base_channels'], 4, 256)
        self._spin(uf, "History frames:", self.flow_settings['flow_history'], 1, 10)
        self._combo(uf, "UNet padding mode:", self.flow_settings['flow_padding_mode'],
                    ['constant', 'reflect', 'replicate', 'circular'])

        # ---------- Channel mult (down/up blocks) ----------
        cmf = tk.Frame(uf); cmf.pack(fill=tk.X, pady=2)
        tk.Label(cmf, text="Channel mult:", width=22, anchor='w').pack(side=tk.LEFT)
        tk.Entry(cmf, textvariable=self.flow_settings['channel_mult'],
                 width=14).pack(side=tk.RIGHT)
        tk.Label(uf,
                 text="Comma-separated multipliers per level, e.g. '1,2,4' or "
                      "'1,2,4,8'. Actual down/up blocks = min(len, log2(img_size)-1).",
                 fg='gray', font=("Arial", 8, "italic"),
                 wraplength=560, justify=tk.LEFT).pack(anchor='w')
        self.flow_settings['channel_mult'].trace_add('write',
            lambda *_: self._refresh_flow_head_dims())

        af = tk.LabelFrame(uf, text="Attention (first 3 levels; extra levels default off)",
                           padx=6, pady=6)
        af.pack(fill=tk.X, pady=(8, 2))
        f = tk.Frame(af); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Enable attention:", width=22, anchor='w').pack(side=tk.LEFT)
        tk.Checkbutton(f, text="L0", variable=self.flow_settings['attn_level_0']).pack(side=tk.LEFT, padx=3)
        tk.Checkbutton(f, text="L1", variable=self.flow_settings['attn_level_1']).pack(side=tk.LEFT, padx=3)
        tk.Checkbutton(f, text="L2", variable=self.flow_settings['attn_level_2']).pack(side=tk.LEFT, padx=3)
        self._spin(af, "Number of heads:", self.flow_settings['attn_num_heads'], 1, 32)
        self._entry(af, "Attention dropout:", self.flow_settings['attn_dropout'])
        tk.Label(af, textvariable=self.flow_head_dim_var, font=("Courier", 9)).pack(fill=tk.X, pady=4)
        self.flow_settings['attn_num_heads'].trace_add('write',
            lambda *_: self._refresh_flow_head_dims())
        self.flow_settings['flow_base_channels'].trace_add('write',
            lambda *_: self._refresh_flow_head_dims())
        self._refresh_flow_head_dims()

        tf = tk.LabelFrame(wrap, text="Flow Training", padx=10, pady=10)
        tf.pack(fill=tk.X, pady=5)
        self._spin(tf, "Batch size:", self.flow_settings['flow_batch_size'], 1, 64)
        self._entry(tf, "Learning rate:", self.flow_settings['flow_lr'])
        self._spin(tf, "Num workers:", self.flow_settings['flow_num_workers'], 0, 16)
        self._entry(tf, "CFG dropout:", self.flow_settings['flow_cfg_dropout'])
        tk.Checkbutton(tf, text="Use OT matching",
                       variable=self.flow_settings['flow_use_ot']).pack(anchor='w')

        bf = tk.LabelFrame(wrap, text="Button Conditioning", padx=10, pady=10)
        bf.pack(fill=tk.X, pady=5)
        self._spin(bf, "Button embed dim:", self.flow_settings['button_embed_dim'], 32, 512)
        self._combo(bf, "Cond mode:", self.flow_settings['cond_mode'],
                    ['shift', 'film', 'adagn', 'cross_attn'])
        tk.Label(bf, text="shift = additive · film = scale+shift · "
                          "adagn = adaptive GroupNorm · cross_attn = cross-attention",
                 fg='gray', font=("Arial", 8, "italic")).pack(anchor='w')
        self.vocab_info_label = tk.Label(bf, text="", fg="blue", anchor='w',
                                         justify=tk.LEFT, wraplength=560)
        self.vocab_info_label.pack(fill=tk.X, pady=(6, 0))
        tk.Button(bf, text="Refresh vocab from dataset",
                  command=self.rebuild_vocab).pack(pady=4)

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
        self._spin(pvf, "Preview frames:", self.flow_settings['preview_num_frames'], 1, 32)
        self._combo(pvf, "ODE method:", self.flow_settings['preview_method'],
                    ['euler', 'heun', 'midpoint', 'rk4'])

    def _refresh_flow_head_dims(self):
        try:
            base = int(self.flow_settings['flow_base_channels'].get())
            nh = int(self.flow_settings['attn_num_heads'].get())
        except Exception:
            return
        channel_mult = self._parse_channel_mult()
        dims = compute_head_dims(base, channel_mult, nh)
        flags = [self.flow_settings['attn_level_0'].get(),
                 self.flow_settings['attn_level_1'].get(),
                 self.flow_settings['attn_level_2'].get()]
        parts = []
        for i, (ch, eff, hd) in enumerate(dims):
            flag_str = 'ON' if (i < len(flags) and flags[i]) else 'off'
            parts.append(f"L{i}[{flag_str}] ch={ch} h={eff} hd={hd}")
        levels_used, bottleneck = self._levels_used()
        summary = (f"| used={levels_used}/{len(channel_mult)} "
                   f"bn={bottleneck}×{bottleneck}")
        self.flow_head_dim_var.set("  ".join(parts) + "  " + summary)

    # ---------- Flow Training ----------
    def setup_flow_training_tab(self, tab):
        main = tk.Frame(tab); main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        left = tk.Frame(main, width=300); left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left.pack_propagate(False)
        tk.Label(left, text="Flow Training (pixel)",
                 font=("Arial", 12, "bold")).pack(pady=(0, 10))
        ep = tk.Frame(left); ep.pack(pady=5)
        tk.Label(ep, text="Epochs:").pack(side=tk.LEFT)
        self.flow_epoch_var = tk.StringVar(value="200")
        tk.Entry(ep, textvariable=self.flow_epoch_var, width=8).pack(side=tk.LEFT, padx=5)
        tk.Button(left, text="Initialize Flow Model",
                  command=self.initialize_rectified_model, width=24).pack(pady=2)
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
        right = tk.Frame(main); right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        pf = tk.LabelFrame(right, text="Preview (last generated frames)",
                           padx=5, pady=5)
        pf.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        self.rectified_preview_canvas = tk.Canvas(pf, bg='gray', width=512, height=256)
        self.rectified_preview_canvas.pack()
        lf = tk.LabelFrame(right, text="Log", padx=5, pady=5)
        lf.pack(fill=tk.BOTH, expand=True)
        self.rectified_log_text = tk.Text(lf, height=15, font=("Courier", 9))
        self.rectified_log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = tk.Scrollbar(lf, command=self.rectified_log_text.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.rectified_log_text.config(yscrollcommand=sb.set)

    # ---------- Augmentation ----------
    def setup_augmentation_tab(self, tab):
        tk.Label(tab, text="Augmentation Settings (flow)",
                 font=("Arial", 14, "bold")).pack(pady=(15, 10))
        cols = tk.Frame(tab); cols.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        flow_col = tk.LabelFrame(cols, text="Flow Augmentations", padx=10, pady=10)
        flow_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))
        self._build_aug_controls(flow_col, self.flow_aug_settings, show_noise_ctx=True)
        r = tk.Frame(tab); r.pack(fill=tk.X, pady=10)
        tk.Button(r, text="Reset Flow Augs",
                  command=lambda: self._reset_aug(self.flow_aug_settings)).pack(padx=10)

    def _build_aug_controls(self, parent, aug, show_noise_ctx=False):
        def row():
            f = tk.Frame(parent); f.pack(fill=tk.X, pady=2); return f
        r = row()
        tk.Label(r, text="Aug padding:").pack(side=tk.LEFT)
        ttk.Combobox(r, textvariable=aug['aug_padding_mode'],
                     values=['edge', 'reflect', 'symmetric', 'constant'],
                     state='readonly', width=10).pack(side=tk.RIGHT)
        row(); tk.Checkbutton(parent, text="Horizontal Flip",
                              variable=aug['flip_horizontal']).pack(anchor='w')
        row(); tk.Checkbutton(parent, text="Rotation (±30°)",
                              variable=aug['rotation']).pack(anchor='w')
        r = row(); tk.Checkbutton(r, text="Random Crop",
                                  variable=aug['random_crop']).pack(side=tk.LEFT)
        tk.Label(r, text="scale:").pack(side=tk.LEFT)
        tk.Entry(r, textvariable=aug['crop_scale'], width=6).pack(side=tk.LEFT)
        r = row(); tk.Checkbutton(r, text="Color Jitter",
                                  variable=aug['color_jitter']).pack(side=tk.LEFT)
        r = row(); tk.Label(r, text="bri/con/sat/hue:").pack(side=tk.LEFT)
        for k in ['brightness', 'contrast', 'saturation', 'hue']:
            tk.Entry(r, textvariable=aug[k], width=5).pack(side=tk.LEFT, padx=1)
        r = row(); tk.Checkbutton(r, text="Perspective",
                                  variable=aug['random_perspective']).pack(side=tk.LEFT)
        tk.Label(r, text="dist:").pack(side=tk.LEFT)
        tk.Entry(r, textvariable=aug['perspective_distortion'], width=5).pack(side=tk.LEFT)
        r = row(); tk.Checkbutton(r, text="Elastic",
                                  variable=aug['elastic_transform']).pack(side=tk.LEFT)
        tk.Label(r, text="α/σ:").pack(side=tk.LEFT)
        tk.Entry(r, textvariable=aug['elastic_alpha'], width=4).pack(side=tk.LEFT)
        tk.Entry(r, textvariable=aug['elastic_sigma'], width=4).pack(side=tk.LEFT)
        r = row(); tk.Checkbutton(r, text="JPEG",
                                  variable=aug['jpeg_compression']).pack(side=tk.LEFT)
        tk.Label(r, text="q:").pack(side=tk.LEFT)
        tk.Entry(r, textvariable=aug['jpeg_quality_low'], width=5).pack(side=tk.LEFT)
        tk.Entry(r, textvariable=aug['jpeg_quality_high'], width=5).pack(side=tk.LEFT)
        r = row()
        tk.Checkbutton(r, text="Stretch V",
                       variable=aug['stretch_vertical']).pack(side=tk.LEFT)
        tk.Checkbutton(r, text="Stretch H",
                       variable=aug['stretch_horizontal']).pack(side=tk.LEFT, padx=6)
        if show_noise_ctx:
            r = row(); tk.Checkbutton(r, text="Noise-Ctx aug",
                                      variable=aug['noise_ctx_enabled']).pack(side=tk.LEFT)
            tk.Label(r, text="p:").pack(side=tk.LEFT)
            tk.Entry(r, textvariable=aug['noise_ctx_prob'], width=4).pack(side=tk.LEFT)
            tk.Label(r, text="σ:").pack(side=tk.LEFT)
            tk.Entry(r, textvariable=aug['noise_ctx_scale'], width=4).pack(side=tk.LEFT)

    def _reset_aug(self, aug):
        for k, v in AUG_DEFAULTS.items():
            if k in aug:
                aug[k].set(v)

    # ---------- Inference ----------
    def setup_inference_tab(self, tab):
        main = tk.Frame(tab); main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        tk.Label(main, text="World-Model Inference — buttons drive the future",
                 font=("Arial", 12, "bold")).pack(pady=(0, 6))

        src = tk.LabelFrame(main, text="Context source", padx=8, pady=6)
        src.pack(fill=tk.X, pady=4)
        mode_row = tk.Frame(src); mode_row.pack(fill=tk.X, pady=2)
        tk.Label(mode_row, text="Mode:", width=10).pack(side=tk.LEFT)
        tk.Radiobutton(mode_row, text="Image", variable=self.inf_mode,
                       value='image', command=self._on_inf_mode_change).pack(side=tk.LEFT)
        tk.Radiobutton(mode_row, text="Video frame", variable=self.inf_mode,
                       value='video', command=self._on_inf_mode_change).pack(side=tk.LEFT, padx=(10, 0))

        load_row = tk.Frame(src); load_row.pack(fill=tk.X, pady=2)
        tk.Button(load_row, text="Load Context", command=self.inf_load_context).pack(side=tk.LEFT)
        tk.Button(load_row, text="Clear", command=self.inf_clear_context).pack(side=tk.LEFT, padx=4)
        self.inf_path_label = tk.Label(load_row, text="(no context)", fg="gray")
        self.inf_path_label.pack(side=tk.LEFT, padx=6, fill=tk.X, expand=True)

        idx_row = tk.Frame(src); idx_row.pack(fill=tk.X, pady=2)
        tk.Label(idx_row, text="Start frame index:", width=16).pack(side=tk.LEFT)
        self.inf_idx_spin = tk.Spinbox(idx_row, from_=-9999, to=9999,
                                       textvariable=self.inf_frame_index, width=8)
        self.inf_idx_spin.pack(side=tk.LEFT)
        tk.Label(idx_row, text=" (-1 = last)", fg='gray').pack(side=tk.LEFT, padx=6)

        bsrc = tk.LabelFrame(main, text="Button sequence to play", padx=8, pady=6)
        bsrc.pack(fill=tk.X, pady=4)
        bmode = tk.Frame(bsrc); bmode.pack(fill=tk.X, pady=2)
        tk.Label(bmode, text="Source:", width=10).pack(side=tk.LEFT)
        tk.Radiobutton(bmode, text="From CSV", variable=self.inf_button_source,
                       value='csv').pack(side=tk.LEFT)
        tk.Radiobutton(bmode, text="Manual", variable=self.inf_button_source,
                       value='manual').pack(side=tk.LEFT, padx=(10, 0))
        tk.Radiobutton(bmode, text="Hold", variable=self.inf_button_source,
                       value='hold').pack(side=tk.LEFT, padx=(10, 0))

        csvrow = tk.Frame(bsrc); csvrow.pack(fill=tk.X, pady=2)
        tk.Button(csvrow, text="Load CSV", command=self.inf_load_button_csv).pack(side=tk.LEFT)
        self.inf_csv_label = tk.Label(csvrow, text="(none)", fg="gray")
        self.inf_csv_label.pack(side=tk.LEFT, padx=6, fill=tk.X, expand=True)

        manrow = tk.Frame(bsrc); manrow.pack(fill=tk.X, pady=2)
        tk.Label(manrow, text="Manual (csv of 'A|B,,A'):").pack(side=tk.LEFT)
        tk.Entry(manrow, textvariable=self.inf_manual_sequence).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

        holdrow = tk.Frame(bsrc); holdrow.pack(fill=tk.X, pady=2)
        tk.Label(holdrow, text="Hold set (e.g. A|B):").pack(side=tk.LEFT)
        tk.Entry(holdrow, textvariable=self.inf_hold_buttons).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

        gen = tk.LabelFrame(main, text="Generation", padx=8, pady=6)
        gen.pack(fill=tk.X, pady=4)
        row = tk.Frame(gen); row.pack(fill=tk.X, pady=2)
        tk.Label(row, text="Frames:").pack(side=tk.LEFT, padx=(2, 0))
        tk.Spinbox(row, from_=1, to=128, textvariable=self.inf_num_frames,
                   width=4).pack(side=tk.LEFT, padx=2)
        tk.Label(row, text="Steps / frame:").pack(side=tk.LEFT, padx=(8, 0))
        tk.Spinbox(row, from_=1, to=200, textvariable=self.inf_steps,
                   width=4).pack(side=tk.LEFT, padx=2)
        tk.Label(row, text="Method:").pack(side=tk.LEFT, padx=(8, 0))
        ttk.Combobox(row, textvariable=self.inf_method,
                     values=['euler', 'heun', 'midpoint', 'rk4'],
                     state='readonly', width=8).pack(side=tk.LEFT, padx=2)
        tk.Label(row, text="CFG:").pack(side=tk.LEFT, padx=(8, 0))
        tk.Entry(row, textvariable=self.inf_cfg, width=5).pack(side=tk.LEFT, padx=2)

        row2 = tk.Frame(gen); row2.pack(fill=tk.X, pady=2)
        tk.Checkbutton(row2, text="Live preview (show frames as they generate)",
                       variable=self.inf_live).pack(side=tk.LEFT)
        self.inf_generate_btn = tk.Button(row2, text="Generate",
                                          command=self.inf_generate, bg="lightgreen")
        self.inf_generate_btn.pack(side=tk.LEFT, padx=(15, 4))
        self.inf_stop_btn = tk.Button(row2, text="Stop", command=self.inf_stop,
                                      bg="salmon", state=tk.DISABLED)
        self.inf_stop_btn.pack(side=tk.LEFT, padx=2)

        row3 = tk.Frame(gen); row3.pack(fill=tk.X, pady=2)
        tk.Label(row3, text="Playback:", fg="gray").pack(side=tk.LEFT)
        self.inf_play_btn = tk.Button(row3, text="▶ Play", command=self._inf_start_playback,
                                      state=tk.DISABLED)
        self.inf_play_btn.pack(side=tk.LEFT, padx=4)
        self.inf_pause_btn = tk.Button(row3, text="⏸ Pause", command=self._inf_pause_playback,
                                       state=tk.DISABLED)
        self.inf_pause_btn.pack(side=tk.LEFT, padx=2)
        self.inf_strip_btn = tk.Button(row3, text="Show Strip",
                                       command=self._inf_show_strip, state=tk.DISABLED)
        self.inf_strip_btn.pack(side=tk.LEFT, padx=2)
        tk.Label(row3, text="FPS:").pack(side=tk.LEFT, padx=(10, 2))
        tk.Spinbox(row3, from_=1, to=60, textvariable=self.playback_fps,
                   width=4).pack(side=tk.LEFT)

        prev = tk.LabelFrame(main, text="Preview", padx=8, pady=6)
        prev.pack(fill=tk.BOTH, expand=True, pady=4)
        prow = tk.Frame(prev); prow.pack(fill=tk.X)
        left = tk.Frame(prow); left.pack(side=tk.LEFT, padx=(0, 4))
        tk.Label(left, text="Context (direct pixel input)").pack(anchor='w')
        self.inf_context_canvas = tk.Canvas(left, bg='gray', width=320, height=100)
        self.inf_context_canvas.pack(anchor='w')
        right = tk.Frame(prow); right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(4, 0))
        tk.Label(right, text="Generated").pack(anchor='w')
        self.inf_gen_canvas = tk.Canvas(right, bg='gray', height=100)
        self.inf_gen_canvas.pack(fill=tk.X, expand=True)

        self.inf_info = tk.Label(main, text="Load a context source and a button sequence.",
                                 fg="blue")
        self.inf_info.pack(pady=2)

        self._on_inf_mode_change()

    def _on_inf_mode_change(self):
        if self.inf_mode.get() == 'video':
            self.inf_idx_spin.config(state='normal')
        else:
            self.inf_idx_spin.config(state='disabled')

    # ---------- Inference helpers ----------
    def inf_load_context(self):
        if self.inf_mode.get() == 'image':
            path = filedialog.askopenfilename(
                filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.jfif *.webp")])
            if not path:
                return
            self.inf_image_path = path
            self.inf_video_path = None
            self.inf_video_frames = []
            self.inf_path_label.config(text=os.path.basename(path), fg="black")
        else:
            path = filedialog.askopenfilename(
                filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv *.webm *.gif *.m4v *.wmv *.flv")])
            if not path:
                return
            if not (_HAS_CV2 or _HAS_IMAGEIO):
                messagebox.showerror("Missing video lib",
                                     "Install 'opencv-python' or 'imageio[ffmpeg]'.")
                return
            try:
                frames = load_video_frames(path, stride=1)
                fps = load_video_fps(path)
            except Exception as e:
                self.inf_info.config(text=f"Video load error: {e}")
                return
            if not frames:
                self.inf_info.config(text="No frames loaded.")
                return
            self.inf_video_path = path
            self.inf_video_frames = frames
            self.inf_video_fps = fps
            self.inf_image_path = None
            self.inf_path_label.config(
                text=f"{os.path.basename(path)} ({len(frames)} frames @ {fps:.1f} fps)",
                fg="black")
        self._inf_refresh_context_preview()

    def inf_clear_context(self):
        self.inf_image_path = None
        self.inf_video_path = None
        self.inf_video_frames = []
        self.inf_path_label.config(text="(no context)", fg="gray")
        self.inf_context_canvas.delete("all")
        self.inf_gen_canvas.delete("all")
        self.inf_info.config(text="Context cleared.")

    def inf_load_button_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if not path:
            return
        try:
            rows = load_button_csv(path)
        except Exception as e:
            self.inf_info.config(text=f"CSV parse error: {e}")
            return
        self.inf_button_csv_path = path
        self.inf_button_rows = rows
        self.inf_button_source.set('csv')
        self.inf_csv_label.config(
            text=f"{os.path.basename(path)} ({len(rows)} rows)", fg="black")

    def _inf_get_context_pils(self):
        history = max(1, int(self.flow_settings['flow_history'].get()))
        if self.inf_mode.get() == 'image':
            if not self.inf_image_path:
                return None, "No image loaded"
            try:
                cm = self.global_settings['color_mode'].get()
                img = load_image_as_rgb(self.inf_image_path) if cm == 'rgb' \
                    else load_image_as_grayscale(self.inf_image_path)
            except Exception as e:
                return None, f"Image load error: {e}"
            ctx = [None] * (history - 1) + [img]
            return ctx, f"Image as last of {history} context slot(s)"
        else:
            if not self.inf_video_frames:
                return None, "No video loaded"
            T = len(self.inf_video_frames)
            K = self.inf_frame_index.get()
            if K < 0:
                K = T + K
            K = max(0, min(T - 1, K))
            start = max(0, K - history + 1)
            real = self.inf_video_frames[start:K + 1]
            pad = history - len(real)
            return [None] * pad + list(real), \
                   f"Frames {start}..{K} ({len(real)} real, {pad} padding)"

    def _inf_refresh_context_preview(self):
        ctx, desc = self._inf_get_context_pils()
        if ctx is None:
            self.inf_context_canvas.delete("all")
            self.inf_context_canvas.create_text(160, 50, text=desc, fill="white")
            self.inf_info.config(text=desc)
            return
        self._inf_display_context(ctx, "source", pad_flags=[p is None for p in ctx])

    def _inf_display_context(self, pils, tag, pad_flags=None):
        thumb = 96
        n = max(1, len(pils))
        grid = Image.new('RGB', (n * thumb, thumb), (50, 50, 50))
        for i, p in enumerate(pils):
            if p is None:
                im = Image.new('RGB', (thumb, thumb), (0, 0, 0))
                is_pad = True
            else:
                im = p.convert('RGB').resize((thumb, thumb), Image.NEAREST)
                is_pad = pad_flags[i] if pad_flags else False
            if is_pad:
                from PIL import ImageDraw
                d = ImageDraw.Draw(im)
                d.rectangle([0, 0, thumb - 1, thumb - 1], outline=(255, 100, 100), width=2)
                d.text((4, 4), "PAD", fill=(255, 220, 220))
            grid.paste(im, (i * thumb, 0))
        photo = ImageTk.PhotoImage(grid)
        self.inf_context_canvas.delete("all")
        self.inf_context_canvas.config(width=grid.width, height=grid.height)
        self.inf_context_canvas.create_image(0, 0, image=photo, anchor='nw')
        self.inf_context_photo = photo
        self.inf_info.config(text=f"Context ({tag})")

    def _build_context_tensor(self, ctx_pils):
        img_size = self.global_settings['img_size'].get()
        cm = self.global_settings['color_mode'].get()
        in_ch = 3 if cm == 'rgb' else 1
        tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * in_ch, (0.5,) * in_ch),
        ])
        tensors = []
        for pil in ctx_pils:
            if pil is None:
                tensors.append(torch.zeros(in_ch, img_size, img_size))
            else:
                pil_n = pil.convert('RGB') if cm == 'rgb' else pil.convert('L')
                tensors.append(tf(pil_n))
        stacked = torch.stack(tensors, dim=0)
        return stacked.unsqueeze(0).to(self.rectified_model.device)

    def _parse_manual_sequence(self, text, n):
        parts = [p.strip() for p in text.split(',')]
        out = []
        for i in range(n):
            if i < len(parts):
                names = set(x.strip() for x in parts[i].split('|') if x.strip())
            else:
                names = set()
            out.append(names)
        return out

    def _parse_hold(self, text, n):
        names = set(x.strip() for x in text.split('|') if x.strip())
        return [set(names) for _ in range(n)]

    def _parse_button_csv_sequence(self, rows, fps, start_frame, n):
        out = []
        for i in range(n):
            t = (start_frame + i) / fps
            out.append(buttons_at_time(rows, t))
        return out

    def _inf_get_button_sequence(self, n_frames):
        mode = self.inf_button_source.get()
        if mode == 'csv':
            if not self.inf_button_rows:
                return None, "No CSV loaded"
            fps = self.inf_video_fps if self.inf_video_path else 30.0
            seq = self._parse_button_csv_sequence(
                self.inf_button_rows, fps, 0, n_frames)
            return seq, f"CSV ({len(seq)} frames)"
        elif mode == 'manual':
            seq = self._parse_manual_sequence(self.inf_manual_sequence.get(), n_frames)
            return seq, "Manual"
        elif mode == 'hold':
            seq = self._parse_hold(self.inf_hold_buttons.get(), n_frames)
            return seq, "Hold"
        return None, "unknown"

    def _sets_to_button_tensors(self, sets_seq):
        N = len(self.button_names)
        out = []
        unknown = set()
        for names in sets_seq:
            vec = torch.zeros(1, max(1, N))
            if N > 0:
                for name in names:
                    if name in self.button_to_idx:
                        vec[0, self.button_to_idx[name]] = 1.0
                    else:
                        unknown.add(name)
            out.append(vec.to(self.rectified_model.device))
        if unknown:
            self.log_rectified(f"Warning: unknown buttons ignored: {sorted(unknown)}")
        return out

    def _sets_to_conds(self, sets_seq):
        tensors = self._sets_to_button_tensors(sets_seq)
        with torch.no_grad():
            return [self.button_encoder(t) for t in tensors]

    def inf_stop(self):
        self.inf_stop_flag = True
        self.inf_stop_btn.config(state=tk.DISABLED)
        self.inf_info.config(text="Stopping...")

    def inf_generate(self):
        if not self.rectified_model:
            self.inf_info.config(text="Load Flow model first!")
            return
        if self.inf_in_progress:
            return
        ctx, desc = self._inf_get_context_pils()
        if ctx is None:
            self.inf_info.config(text=desc)
            return
        n_frames = int(self.inf_num_frames.get())
        seq, seq_desc = self._inf_get_button_sequence(n_frames)
        if seq is None:
            self.inf_info.config(text=f"Button source error: {seq_desc}")
            return
        if self.button_encoder is None:
            self.inf_info.config(text="Button encoder not initialized!")
            return
        self._inf_pause_playback()
        self.inf_result_pils = []
        self.inf_play_btn.config(state=tk.DISABLED)
        self.inf_pause_btn.config(state=tk.DISABLED)
        self.inf_strip_btn.config(state=tk.DISABLED)
        try:
            context_tensor = self._build_context_tensor(ctx)
        except Exception as e:
            self.inf_info.config(text=f"Context encode error: {e}")
            return
        cond_seq = self._sets_to_conds(seq)
        self.inf_stop_flag = False
        self.inf_in_progress = True
        self.inf_generate_btn.config(state=tk.DISABLED)
        self.inf_stop_btn.config(state=tk.NORMAL)
        self.inf_info.config(text=f"Generating {n_frames} frames ({seq_desc})...")
        live = bool(self.inf_live.get())
        threading.Thread(
            target=self._inf_thread,
            args=(context_tensor, cond_seq, n_frames,
                  int(self.inf_steps.get()),
                  self.inf_method.get(),
                  float(self.inf_cfg.get()),
                  bool(self.flow_settings['ema_enabled'].get()),
                  live),
            daemon=True).start()

    def _inf_thread(self, context_tensor, cond_seq, num_frames, steps,
                    method, cfg, use_ema, live):
        try:
            decoded_pils = []
            def on_frame(frame_idx, z):
                arr = ((z + 1) / 2).clamp(0, 1).cpu().numpy()[0]
                if arr.shape[0] == 1:
                    im = np.stack([arr[0] * 255] * 3, axis=-1).astype(np.uint8)
                else:
                    im = (arr.transpose(1, 2, 0) * 255).astype(np.uint8)
                pil = Image.fromarray(im)
                decoded_pils.append(pil)
                if live:
                    self.root.after(0, self._inf_show_current,
                                    list(decoded_pils), frame_idx, num_frames)
            z_vid = self.rectified_model.sample_video(
                n_samples=1, num_frames=num_frames, cond_sequence=cond_seq,
                steps=steps, method=method, cfg_scale=cfg, use_ema=use_ema,
                init_context=context_tensor,
                stop_check=lambda: self.inf_stop_flag,
                frame_cb=on_frame if live else None)
            if z_vid is None:
                self.root.after(0, lambda: self.inf_info.config(text="Stopped."))
                return
            if not live:
                _, T, Cz, Hl, Wl = z_vid.shape
                flat = z_vid.reshape(T, Cz, Hl, Wl)
                arr = ((flat + 1) / 2).clamp(0, 1).cpu().numpy()
                decoded_pils = []
                for i in range(arr.shape[0]):
                    a = arr[i]
                    if a.shape[0] == 1:
                        im = np.stack([a[0] * 255] * 3, axis=-1).astype(np.uint8)
                    else:
                        im = (a.transpose(1, 2, 0) * 255).astype(np.uint8)
                    decoded_pils.append(Image.fromarray(im))
            self.root.after(0, self._inf_finalize, decoded_pils)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda: self.inf_info.config(text=f"Error: {e}"))
        finally:
            self.inf_in_progress = False
            self.root.after(0, lambda: self.inf_generate_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.inf_stop_btn.config(state=tk.DISABLED))

    def _inf_show_current(self, pils, frame_idx, total):
        if not pils:
            return
        self.inf_result_pils = pils
        grid = self._render_strip(pils)
        photo = ImageTk.PhotoImage(grid)
        self.inf_gen_canvas.delete("all")
        self.inf_gen_canvas.config(width=grid.width, height=grid.height)
        self.inf_gen_canvas.create_image(0, 0, image=photo, anchor='nw')
        self.inf_gen_photo = photo
        self.inf_info.config(text=f"Frame {frame_idx + 1}/{total}")

    def _inf_finalize(self, pils):
        self.inf_result_pils = pils
        self._inf_show_strip()
        self.inf_play_btn.config(state=tk.NORMAL)
        self.inf_pause_btn.config(state=tk.NORMAL)
        self.inf_strip_btn.config(state=tk.NORMAL)
        self.inf_info.config(text=f"Done. {len(pils)} frames.")
        self._inf_start_playback()

    def _render_strip(self, pils):
        thumb = 96
        n = max(1, len(pils))
        grid = Image.new('RGB', (n * thumb, thumb), (80, 80, 80))
        for i, pil in enumerate(pils):
            grid.paste(pil.convert('RGB').resize((thumb, thumb), Image.NEAREST),
                       (i * thumb, 0))
        return grid

    def _inf_show_strip(self):
        self._inf_pause_playback()
        if not self.inf_result_pils:
            return
        grid = self._render_strip(self.inf_result_pils)
        photo = ImageTk.PhotoImage(grid)
        self.inf_gen_canvas.delete("all")
        self.inf_gen_canvas.config(width=grid.width, height=grid.height)
        self.inf_gen_canvas.create_image(0, 0, image=photo, anchor='nw')
        self.inf_gen_photo = photo
        self.inf_info.config(text=f"All {len(self.inf_result_pils)} frames")

    def _inf_start_playback(self):
        if not self.inf_result_pils:
            return
        self._inf_pause_playback()
        self.inf_playback_active = True
        self.inf_playback_idx = 0
        self.inf_play_btn.config(state=tk.DISABLED)
        self.inf_pause_btn.config(state=tk.NORMAL)
        self._inf_playback_tick()

    def _inf_pause_playback(self):
        self.inf_playback_active = False
        if self.inf_playback_after is not None:
            try:
                self.root.after_cancel(self.inf_playback_after)
            except Exception:
                pass
            self.inf_playback_after = None
        if self.inf_result_pils:
            self.inf_play_btn.config(state=tk.NORMAL)
        self.inf_pause_btn.config(state=tk.DISABLED)

    def _inf_playback_tick(self):
        if not self.inf_playback_active or not self.inf_result_pils:
            return
        i = self.inf_playback_idx % len(self.inf_result_pils)
        pil = self.inf_result_pils[i]
        thumb = 256
        disp = pil.convert('RGB').resize((thumb, thumb), Image.NEAREST)
        photo = ImageTk.PhotoImage(disp)
        self.inf_gen_canvas.delete("all")
        self.inf_gen_canvas.config(width=thumb, height=thumb)
        self.inf_gen_canvas.create_image(0, 0, image=photo, anchor='nw')
        self.inf_gen_photo = photo
        self.inf_info.config(text=f"Frame {i + 1}/{len(self.inf_result_pils)}")
        self.inf_playback_idx += 1
        fps = max(1, int(self.playback_fps.get()))
        self.inf_playback_after = self.root.after(int(1000 / fps), self._inf_playback_tick)

    # ---------- Play tab ----------
    def setup_play_tab(self, tab):
        frame = tk.Frame(tab); frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left = tk.Frame(frame)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        tk.Label(left, text="Live Dream (pixel)",
                 font=("Arial", 12, "bold")).pack(anchor='w', pady=(0, 6))
        self.play_canvas = tk.Canvas(left, width=512, height=512, bg='black')
        self.play_canvas.pack(padx=0, pady=0)
        self.play_frame_count_label = tk.Label(left, text="Frames: 0",
                                               font=("Courier", 10))
        self.play_frame_count_label.pack(anchor='w', pady=(6, 0))

        panel = tk.Frame(frame)
        panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))

        self.play_info = tk.Label(panel, text="Load a context and press Start.",
                                  fg="blue", wraplength=280, justify=tk.LEFT)
        self.play_info.pack(anchor='w', pady=(0, 6))

        ctx_frame = tk.LabelFrame(panel, text="Starting Context", padx=6, pady=4)
        ctx_frame.pack(fill=tk.X, pady=4)

        ctx_mode_row = tk.Frame(ctx_frame); ctx_mode_row.pack(fill=tk.X, pady=2)
        tk.Label(ctx_mode_row, text="Mode:").pack(side=tk.LEFT)
        tk.Radiobutton(ctx_mode_row, text="Image",
                       variable=self.play_ctx_mode, value='image',
                       command=self._on_play_ctx_mode_change).pack(side=tk.LEFT)
        tk.Radiobutton(ctx_mode_row, text="Video",
                       variable=self.play_ctx_mode, value='video',
                       command=self._on_play_ctx_mode_change).pack(side=tk.LEFT, padx=(10, 0))

        ctx_load_row = tk.Frame(ctx_frame); ctx_load_row.pack(fill=tk.X, pady=2)
        tk.Button(ctx_load_row, text="Load Context",
                  command=self.play_load_context).pack(side=tk.LEFT)
        self.play_ctx_label = tk.Label(ctx_load_row, text="(no context)",
                                       fg="gray", anchor='w')
        self.play_ctx_label.pack(side=tk.LEFT, padx=6, fill=tk.X, expand=True)

        ctx_idx_row = tk.Frame(ctx_frame); ctx_idx_row.pack(fill=tk.X, pady=2)
        tk.Label(ctx_idx_row, text="Frame index:", width=12).pack(side=tk.LEFT)
        self.play_idx_spin = tk.Spinbox(
            ctx_idx_row, from_=-9999, to=9999,
            textvariable=self.play_frame_index, width=8)
        self.play_idx_spin.pack(side=tk.LEFT, padx=2)
        tk.Label(ctx_idx_row, text="(-1=last)", fg="gray",
                 font=("Arial", 8)).pack(side=tk.LEFT, padx=4)

        gen_frame = tk.LabelFrame(panel, text="Generation", padx=6, pady=4)
        gen_frame.pack(fill=tk.X, pady=4)

        row = tk.Frame(gen_frame); row.pack(fill=tk.X, pady=2)
        tk.Label(row, text="Steps per frame:", width=16, anchor='w').pack(side=tk.LEFT)
        tk.Spinbox(row, from_=1, to=200, textvariable=self.play_steps,
                   width=6).pack(side=tk.RIGHT)

        row = tk.Frame(gen_frame); row.pack(fill=tk.X, pady=2)
        tk.Label(row, text="Solver:", width=16, anchor='w').pack(side=tk.LEFT)
        ttk.Combobox(row, textvariable=self.play_method,
                     values=['euler', 'heun', 'midpoint', 'rk4'],
                     state='readonly', width=10).pack(side=tk.RIGHT)

        row = tk.Frame(gen_frame); row.pack(fill=tk.X, pady=2)
        tk.Label(row, text="CFG scale:", width=16, anchor='w').pack(side=tk.LEFT)
        tk.Entry(row, textvariable=self.play_cfg, width=8).pack(side=tk.RIGHT)

        beh_frame = tk.LabelFrame(panel, text="Behavior", padx=6, pady=4)
        beh_frame.pack(fill=tk.X, pady=4)

        tk.Checkbutton(beh_frame,
                       text="Only advance on key press (1/0)",
                       variable=self.play_advance_mode).pack(anchor='w', pady=2)
        tk.Checkbutton(beh_frame,
                       text="Live preview (show frame generating)",
                       variable=self.play_live_preview).pack(anchor='w', pady=2)

        ctrl_frame = tk.Frame(panel)
        ctrl_frame.pack(fill=tk.X, pady=6)

        self.play_start_btn = tk.Button(ctrl_frame, text="▶ Start Play",
                                        command=self.start_play, bg="lightgreen")
        self.play_start_btn.pack(fill=tk.X, pady=2)
        self.play_stop_btn = tk.Button(ctrl_frame, text="⏸ Stop",
                                       command=self.stop_play,
                                       bg="salmon", state=tk.DISABLED)
        self.play_stop_btn.pack(fill=tk.X, pady=2)
        self.play_step_btn = tk.Button(ctrl_frame, text="Step (advance once)",
                                       command=self.play_step_once, state=tk.DISABLED)
        self.play_step_btn.pack(fill=tk.X, pady=2)

        info = tk.Label(panel,
                        text=("Arrow keys / bound keys activate buttons.\n"
                              "Bindings are configured in the Settings tab.\n"
                              "In advance mode, press any key (or Step) to advance."),
                        justify=tk.LEFT, fg="gray", font=("Arial", 8, "italic"))
        info.pack(anchor='w', pady=(8, 0))

        self._on_play_ctx_mode_change()

    def _on_play_ctx_mode_change(self):
        if self.play_ctx_mode.get() == 'video':
            self.play_idx_spin.config(state='normal')
        else:
            self.play_idx_spin.config(state='disabled')

    def play_load_context(self):
        if self.play_ctx_mode.get() == 'image':
            path = filedialog.askopenfilename(
                filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.jfif *.webp")])
            if not path:
                return
            self.play_ctx_image_path = path
            self.play_ctx_video_path = None
            self.play_ctx_video_frames = []
            self.play_ctx_label.config(text=os.path.basename(path), fg="black")
        else:
            path = filedialog.askopenfilename(
                filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv *.webm *.gif *.m4v *.wmv *.flv")])
            if not path:
                return
            if not (_HAS_CV2 or _HAS_IMAGEIO):
                messagebox.showerror("Missing video lib",
                                     "Install 'opencv-python' or 'imageio[ffmpeg]'.")
                return
            try:
                frames = load_video_frames(path, stride=1)
            except Exception as e:
                messagebox.showerror("Video load error", str(e))
                return
            if not frames:
                messagebox.showerror("Error", "No frames loaded.")
                return
            self.play_ctx_video_path = path
            self.play_ctx_video_frames = frames
            self.play_ctx_image_path = None
            self.play_ctx_label.config(
                text=f"{os.path.basename(path)} ({len(frames)} frames)",
                fg="black")

    def _play_get_context_pils(self):
        history = max(1, int(self.flow_settings['flow_history'].get()))
        if self.play_ctx_mode.get() == 'image':
            if not self.play_ctx_image_path:
                return None
            try:
                cm = self.global_settings['color_mode'].get()
                img = load_image_as_rgb(self.play_ctx_image_path) if cm == 'rgb' \
                    else load_image_as_grayscale(self.play_ctx_image_path)
            except Exception as e:
                self.play_info.config(text=f"Image load error: {e}")
                return None
            return [None] * (history - 1) + [img]
        else:
            if not self.play_ctx_video_frames:
                return None
            T = len(self.play_ctx_video_frames)
            K = self.play_frame_index.get()
            if K < 0:
                K = T + K
            K = max(0, min(T - 1, K))
            start = max(0, K - history + 1)
            real = self.play_ctx_video_frames[start:K + 1]
            pad = history - len(real)
            return [None] * pad + list(real)

    def _is_text_input_focused(self):
        try:
            focus = self.root.focus_get()
            if focus is None:
                return False
            cls = focus.winfo_class()
            return cls in ('Entry', 'TEntry', 'TSpinbox', 'TCombobox', 'Text')
        except Exception:
            return False

    def _key_identifiers(self, event):
        ids = set()
        if event.char:
            ids.add(event.char.lower())
        if event.keysym:
            ids.add(event.keysym.lower())
        return ids

    def _on_play_key_press(self, event):
        if self._is_text_input_focused():
            return
        if not self.play_running:
            return
        ids = self._key_identifiers(event)
        if not ids:
            return
        already = any(self.play_pressed_keys.get(k, False) for k in ids)
        if self.play_advance_mode.get() and not already:
            if not ids.issubset(_PURE_MOD_KEYS):
                self.play_advance_trigger = True
        for k in ids:
            self.play_pressed_keys[k] = True

    def _on_play_key_release(self, event):
        ids = self._key_identifiers(event)
        for k in ids:
            self.play_pressed_keys[k] = False

    def _active_button_names(self):
        active = set()
        for k, pressed in self.play_pressed_keys.items():
            if pressed and k in self.key_to_button:
                active.add(self.key_to_button[k])
        return active

    def _buttons_to_cond(self, active_names):
        N = len(self.button_names)
        if N == 0 or self.button_encoder is None:
            return torch.zeros(1, self.rectified_model.cond_dim,
                               device=self.rectified_model.device)
        vec = torch.zeros(1, N)
        for name in active_names:
            if name in self.button_to_idx:
                vec[0, self.button_to_idx[name]] = 1.0
        vec = vec.to(self.rectified_model.device)
        with torch.no_grad():
            return self.button_encoder(vec)

    def _pixels_to_pil(self, z):
        arr = ((z + 1) / 2).clamp(0, 1).cpu().numpy()[0]
        if arr.shape[0] == 1:
            im = np.stack([arr[0] * 255] * 3, axis=-1).astype(np.uint8)
        else:
            im = (arr.transpose(1, 2, 0) * 255).astype(np.uint8)
        return Image.fromarray(im)

    def _display_play_frame(self, pil):
        display = 512
        pil_disp = pil.convert('RGB').resize((display, display), Image.NEAREST)
        photo = ImageTk.PhotoImage(pil_disp)
        self.play_canvas.delete("all")
        self.play_canvas.create_image(0, 0, image=photo, anchor='nw')
        self.play_photo = photo

    def play_step_once(self):
        self.play_advance_trigger = True

    def start_play(self):
        if not self.rectified_model:
            messagebox.showerror("Missing model", "Load/initialize Flow model first.")
            return
        if self.button_encoder is None:
            messagebox.showerror("Missing encoder",
                                 "Initialize the Flow model to build the button encoder.")
            return
        if self.play_running:
            return

        ctx_pils = self._play_get_context_pils()
        if ctx_pils is None:
            messagebox.showerror("No context",
                                 "Load a context image/video before starting Play.")
            return
        try:
            self.play_context = self._build_context_tensor(ctx_pils)
        except Exception as e:
            messagebox.showerror("Context encode error", str(e))
            return

        self.key_to_button = {}
        for name, key in self.button_to_key.items():
            if name in self.button_to_idx and key:
                self.key_to_button[key.lower()] = name

        self.play_pressed_keys = {}
        self.play_advance_trigger = False
        self.play_stop_flag = False
        self.play_running = True
        self.play_frame_count = 0
        self.play_frame_count_label.config(text="Frames: 0")

        try:
            ctx_last = self.play_context[:, -1]
            self._display_play_frame(self._pixels_to_pil(ctx_last))
        except Exception:
            pass

        self._kb_press_id = self.root.bind('<KeyPress>', self._on_play_key_press, add='+')
        self._kb_release_id = self.root.bind('<KeyRelease>', self._on_play_key_release, add='+')

        self.play_start_btn.config(state=tk.DISABLED)
        self.play_stop_btn.config(state=tk.NORMAL)
        self.play_step_btn.config(state=tk.NORMAL)

        mode = "manual (advance on key)" if self.play_advance_mode.get() else "continuous"
        self.play_info.config(
            text=f"Playing — {mode}. Bound buttons: "
                 f"{', '.join(f'{k}={v}' for k, v in self.key_to_button.items()) or '(none)'}")

        self.play_queue = queue.Queue()
        threading.Thread(target=self._play_worker, daemon=True).start()
        self.root.after(50, self._poll_play_queue)

    def stop_play(self):
        self.play_stop_flag = True
        if self.play_stop_btn:
            self.play_stop_btn.config(state=tk.DISABLED)
        self.play_info.config(text="Stopping...")

    def _on_play_end(self):
        self.play_running = False
        self.play_start_btn.config(state=tk.NORMAL)
        self.play_stop_btn.config(state=tk.DISABLED)
        self.play_step_btn.config(state=tk.DISABLED)
        try:
            if self._kb_press_id:
                self.root.unbind('<KeyPress>', self._kb_press_id)
            if self._kb_release_id:
                self.root.unbind('<KeyRelease>', self._kb_release_id)
        except Exception:
            pass
        self._kb_press_id = None
        self._kb_release_id = None
        self.play_pressed_keys = {}
        self.play_info.config(text="Play stopped.")

    def _play_worker(self):
        try:
            while not self.play_stop_flag:
                if self.play_advance_mode.get():
                    while not self.play_advance_trigger and not self.play_stop_flag:
                        time.sleep(0.02)
                    if self.play_stop_flag:
                        break
                    self.play_advance_trigger = False

                active = self._active_button_names()
                cond = self._buttons_to_cond(active)

                n_steps = int(self.play_steps.get())
                method = self.play_method.get()
                cfg = float(self.play_cfg.get())
                use_ema = self.flow_settings['ema_enabled'].get()
                live = self.play_live_preview.get()

                final_z = None
                if live:
                    gen = self.rectified_model.sample_video_step_by_step(
                        n_samples=1, num_frames=1, cond_sequence=[cond],
                        steps=n_steps, method=method, cfg_scale=cfg,
                        use_ema=use_ema, init_context=self.play_context)
                    update_every = max(1, n_steps // 10)
                    for fi, si, frames_z in gen:
                        if self.play_stop_flag:
                            break
                        z = frames_z[-1]
                        if si % update_every == 0 or si == n_steps:
                            pil = self._pixels_to_pil(z)
                            self.play_queue.put(('preview', pil, si, n_steps))
                        if si == n_steps:
                            final_z = z
                else:
                    z_vid = self.rectified_model.sample_video(
                        n_samples=1, num_frames=1, cond_sequence=[cond],
                        steps=n_steps, method=method, cfg_scale=cfg,
                        use_ema=use_ema, init_context=self.play_context,
                        stop_check=lambda: self.play_stop_flag)
                    if z_vid is None:
                        break
                    final_z = z_vid[:, 0]
                    pil = self._pixels_to_pil(final_z)
                    self.play_queue.put(('final', pil, n_steps, n_steps))

                if self.play_stop_flag or final_z is None:
                    break

                self.play_context = torch.cat(
                    [self.play_context[:, 1:], final_z.unsqueeze(1)], dim=1)
                self.play_frame_count += 1
                self.play_queue.put(('frame_done', None, self.play_frame_count, 0))
        except Exception as e:
            import traceback
            traceback.print_exc()
            try:
                self.play_queue.put(('error', str(e), 0, 0))
            except Exception:
                pass
        finally:
            try:
                self.play_queue.put(('end', None, 0, 0))
            except Exception:
                pass

    def _poll_play_queue(self):
        if self.play_queue is None:
            return
        try:
            while True:
                msg = self.play_queue.get_nowait()
                kind = msg[0]
                if kind == 'preview':
                    _, pil, si, total = msg
                    self._display_play_frame(pil)
                    self.play_info.config(text=f"Generating: step {si}/{total}")
                elif kind == 'final':
                    _, pil, si, total = msg
                    self._display_play_frame(pil)
                elif kind == 'frame_done':
                    _, _, count, _ = msg
                    self.play_frame_count_label.config(text=f"Frames: {count}")
                elif kind == 'error':
                    _, err, _, _ = msg
                    self.play_info.config(text=f"Error: {err}")
                elif kind == 'end':
                    self._on_play_end()
                    return
        except queue.Empty:
            pass
        if self.play_running:
            self.root.after(50, self._poll_play_queue)

    # ---------- Logging ----------
    def log_rectified(self, msg):
        self.message_queue_rectified.put(msg)

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

    # ---------- Source management ----------
    def _refresh_source_listbox(self):
        self.source_listbox.delete(0, tk.END)
        for s in self.sources:
            if s['kind'] == 'image':
                self.source_listbox.insert(tk.END, f"[img] {os.path.basename(s['path'])}")
            elif s['kind'] == 'video':
                csv_tag = ""
                if s.get('button_csv_path'):
                    csv_tag = f"  + CSV:{os.path.basename(s['button_csv_path'])}"
                else:
                    csv_tag = "  (no CSV)"
                self.source_listbox.insert(
                    tk.END, f"[vid] {os.path.basename(s['path'])}{csv_tag}")

    def add_videos(self):
        files = filedialog.askopenfilenames(
            filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv *.webm *.gif *.m4v *.wmv *.flv"),
                       ("All files", "*.*")])
        if not files:
            return
        if not (_HAS_CV2 or _HAS_IMAGEIO):
            messagebox.showerror("Missing video lib",
                                 "Install 'opencv-python' or 'imageio[ffmpeg]'.")
            return
        stride = max(1, int(self.frame_stride.get()))
        added = 0
        for f in files:
            if any(s.get('path') == f and s['kind'] == 'video' for s in self.sources):
                continue
            csv_path = filedialog.askopenfilename(
                title=f"Button CSV for '{os.path.basename(f)}'  (Cancel = skip)",
                filetypes=[("CSV", "*.csv"), ("All files", "*.*")])
            button_rows = []
            button_csv_path = None
            if csv_path:
                try:
                    button_rows = load_button_csv(csv_path)
                    button_csv_path = csv_path
                except Exception as e:
                    messagebox.showerror("CSV error", f"{csv_path}\n\n{e}")
            fps = load_video_fps(f) or 30.0
            self.sources.append({
                'kind': 'video',
                'path': f,
                'stride': stride,
                'button_csv_path': button_csv_path,
                'button_rows': button_rows,
                'fps': fps,
            })
            added += 1
        self._refresh_source_listbox()
        self.rebuild_vocab()
        self.log_rectified(f"Added {added} video(s).")

    def add_images(self):
        files = filedialog.askopenfilenames(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.jfif *.webp *.bmp")])
        added = 0
        for f in files:
            if any(s.get('path') == f and s['kind'] == 'image' for s in self.sources):
                continue
            self.sources.append({'kind': 'image', 'path': f})
            added += 1
        self._refresh_source_listbox()
        self.log_rectified(f"Added {added} image(s).")

    def attach_csv_to_selected(self):
        sel = self.source_listbox.curselection()
        if not sel:
            messagebox.showinfo("No selection", "Select a video in the list first.")
            return
        idx = sel[0]
        if idx >= len(self.sources):
            return
        src = self.sources[idx]
        if src['kind'] != 'video':
            messagebox.showinfo("Not a video", "Only videos can have button CSVs.")
            return
        csv_path = filedialog.askopenfilename(
            title=f"Button CSV for '{os.path.basename(src['path'])}'",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")])
        if not csv_path:
            return
        try:
            src['button_rows'] = load_button_csv(csv_path)
            src['button_csv_path'] = csv_path
        except Exception as e:
            messagebox.showerror("CSV error", str(e))
            return
        self._refresh_source_listbox()
        self.rebuild_vocab()

    def clear_all(self):
        self.sources = []
        self.button_names = []
        self.button_to_idx = {}
        self._refresh_source_listbox()
        self._update_vocab_label()
        self._refresh_bindings_ui()
        self.log_rectified("Cleared all sources.")

    def rebuild_vocab(self):
        row_lists = [s.get('button_rows', []) for s in self.sources]
        self.button_names = collect_button_names(row_lists)
        self.button_to_idx = {n: i for i, n in enumerate(self.button_names)}
        self._update_vocab_label()
        self.button_to_key = {n: k for n, k in self.button_to_key.items()
                              if n in self.button_to_idx}
        try:
            self._refresh_bindings_ui()
        except Exception:
            pass
        self.log_rectified(
            f"Vocabulary rebuilt: {len(self.button_names)} buttons: "
            f"{', '.join(self.button_names) if self.button_names else '(empty)'}")

    def _update_vocab_label(self):
        n = len(self.button_names)
        if n == 0:
            txt = "(empty — no CSV loaded)"
            col = "gray"
        else:
            txt = f"{n} buttons: " + ", ".join(
                f"{i}:{name}" for i, name in enumerate(self.button_names))
            col = "black"
        self.vocab_label.config(text=txt, fg=col)
        try:
            self.vocab_info_label.config(text=txt)
        except Exception:
            pass

    # ---------- Dataset ----------
    def _build_dataset(self, aug_settings, include_history=True):
        aug_dict = {k: v.get() for k, v in aug_settings.items()
                    if k not in ('noise_ctx_enabled', 'noise_ctx_prob', 'noise_ctx_scale')}
        hist = self.flow_settings['flow_history'].get() if include_history else 1
        ds = VideoFrameDataset(
            sources=self.sources,
            img_size=self.global_settings['img_size'].get(),
            color_mode=self.global_settings['color_mode'].get(),
            history=hist,
            aug_settings=aug_dict,
            button_to_idx=self.button_to_idx,
            num_buttons=len(self.button_names),
        )
        ds._source_strides = [s.get('stride', 1) for s in self.sources]
        return ds

    # ---------- Flow model ----------
    def initialize_rectified_model(self):
        try:
            cond_dim = 256
            history = max(1, min(10, self.flow_settings['flow_history'].get()))
            attn_levels = [self.flow_settings['attn_level_0'].get(),
                           self.flow_settings['attn_level_1'].get(),
                           self.flow_settings['attn_level_2'].get()]
            attn_num_heads = int(self.flow_settings['attn_num_heads'].get())
            attn_dropout = float(self.flow_settings['attn_dropout'].get())
            cond_mode = self.flow_settings['cond_mode'].get()
            if cond_mode not in ('shift', 'film', 'adagn', 'cross_attn'):
                cond_mode = 'shift'

            in_ch = 3 if self.global_settings['color_mode'].get() == 'rgb' else 1
            img_size = self.global_settings['img_size'].get()
            channel_mult = self._parse_channel_mult()
            levels_used, bottleneck = self._levels_used()

            self.rectified_model = RectifiedFlowPixelVideo(
                in_channels=in_ch,
                img_size=img_size,
                history=history,
                base_channels=self.flow_settings['flow_base_channels'].get(),
                cond_dim=cond_dim,
                attention_levels=attn_levels,
                attn_num_heads=attn_num_heads,
                attn_dropout=attn_dropout,
                padding_mode=self.flow_settings['flow_padding_mode'].get(),
                cond_mode=cond_mode,
                channel_mult=channel_mult,
            )
            self.rectified_model.use_ot = self.flow_settings['flow_use_ot'].get()
            for pg in self.rectified_model.optimizer.param_groups:
                pg['lr'] = self.flow_settings['flow_lr'].get()

            if self.flow_settings['ema_enabled'].get():
                self.rectified_model.set_ema(
                    decay=self.flow_settings['ema_decay'].get(),
                    mode=self.flow_settings['ema_mode'].get())

            N = max(1, len(self.button_names))
            self.button_encoder = ButtonEncoder(
                num_buttons=N, cond_dim=cond_dim,
                hidden_dim=int(self.flow_settings['button_embed_dim'].get()),
            ).to(self.rectified_model.device)
            self.rectified_model.optimizer = optim.Adam(
                list(self.rectified_model.model.parameters())
                + list(self.button_encoder.parameters()),
                lr=self.flow_settings['flow_lr'].get())

            self.log_rectified(
                f"Flow model initialized. pixel={in_ch}ch {img_size}px, "
                f"history={history}, channel_mult={channel_mult} "
                f"(levels used={levels_used}, bottleneck {bottleneck}×{bottleneck}), "
                f"buttons={len(self.button_names)}, cond_dim={cond_dim}, "
                f"cond_mode={cond_mode}.")
            self._update_vocab_label()
        except Exception as e:
            import traceback; traceback.print_exc()
            self.log_rectified(f"Init error: {e}")

    def start_rectified_training(self):
        if not self.sources: self.log_rectified("No sources!"); return
        if not self.rectified_model: self.log_rectified("Initialize flow model!"); return
        if not self.button_encoder: self.log_rectified("Button encoder missing!"); return
        if self.training_rectified: self.log_rectified("Already training."); return
        if len(self.button_names) == 0:
            self.log_rectified("No buttons in vocabulary — load videos with button CSVs.")
            return
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
            nw = self.flow_settings['flow_num_workers'].get()
            cfg_drop = self.flow_settings['flow_cfg_dropout'].get()
            preview_enabled = self.flow_settings['preview_enabled'].get()
            preview_freq = self.flow_settings['preview_epoch_freq'].get()
            noise_ctx_on = self.flow_aug_settings['noise_ctx_enabled'].get()
            noise_ctx_prob = self.flow_aug_settings['noise_ctx_prob'].get() if noise_ctx_on else 0.0
            noise_ctx_scale = self.flow_aug_settings['noise_ctx_scale'].get()

            ds = self._build_dataset(self.flow_aug_settings, include_history=True)
            dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=nw,
                            pin_memory=False, persistent_workers=(nw > 0))

            for ep in range(epochs):
                if not self.training_rectified:
                    break
                for pg in self.rectified_model.optimizer.param_groups:
                    pg['lr'] = self.flow_settings['flow_lr'].get()
                self.rectified_model.use_ot = self.flow_settings['flow_use_ot'].get()

                tot, n = 0.0, 0
                for seqs, btns in dl:
                    if not self.training_rectified:
                        break
                    seqs = seqs.to(self.rectified_model.device)  # [B, S, C, H, W]
                    context = seqs[:, :-1]                       # [B, history, C, H, W]
                    z1 = seqs[:, -1]                             # [B, C, H, W]
                    z0 = torch.randn_like(z1)
                    target_btns = btns[:, -1].to(self.rectified_model.device)
                    cond = self.button_encoder(target_btns)
                    loss = self.rectified_model.train_step(
                        z0, z1, cond=cond, context=context,
                        cfg_dropout_prob=cfg_drop,
                        noise_ctx_prob=noise_ctx_prob,
                        noise_ctx_scale=noise_ctx_scale)
                    tot += loss; n += 1

                avg = tot / max(1, n)
                el = time.time() - self.flow_start_time
                self.log_rectified(
                    f"Epoch {ep+1}/{epochs} | Loss: {avg:.6f} | Time: {el:.1f}s")
                if preview_enabled and (ep + 1) % preview_freq == 0:
                    self._flow_preview()

            self.training_rectified = False
            self.log_rectified("Flow training finished.")
        except Exception as e:
            import traceback; traceback.print_exc()
            self.log_rectified(f"Training error: {e}")
            self.training_rectified = False

    def stop_rectified_training(self):
        self.training_rectified = False
        self.log_rectified("Flow training stopped.")

    def _flow_preview(self):
        if not self.rectified_model:
            return
        try:
            n = 1
            num_frames = int(self.flow_settings['preview_num_frames'].get())
            steps = int(self.flow_settings['preview_steps'].get())
            method = self.flow_settings['preview_method'].get()
            cond_seq = []
            N = max(1, len(self.button_names))
            for _ in range(num_frames):
                vec = torch.zeros(1, N)
                if N > 0 and random.random() < 0.6:
                    idx = random.randrange(N)
                    vec[0, idx] = 1.0
                with torch.no_grad():
                    cond_seq.append(self.button_encoder(vec.to(
                        self.rectified_model.device)))
            z_vid = self.rectified_model.sample_video(
                n_samples=n, num_frames=num_frames, cond_sequence=cond_seq,
                steps=steps, method=method, cfg_scale=1.0,
                use_ema=self.flow_settings['ema_enabled'].get())
            if z_vid is None:
                return
            _, T, Cz, Hl, Wl = z_vid.shape
            flat = z_vid.reshape(T, Cz, Hl, Wl)
            arr = ((flat + 1) / 2).clamp(0, 1).cpu().numpy()
            cols = min(8, T)
            rows = int(math.ceil(T / cols))
            thumb = 96
            grid = Image.new('RGB', (cols * thumb, rows * thumb))
            for i in range(T):
                a = arr[i]
                if a.shape[0] == 1:
                    im = np.stack([a[0] * 255] * 3, axis=-1).astype(np.uint8)
                else:
                    im = (a.transpose(1, 2, 0) * 255).astype(np.uint8)
                pil = Image.fromarray(im).resize((thumb, thumb), Image.NEAREST)
                r, c = i // cols, i % cols
                grid.paste(pil, (c * thumb, r * thumb))
            self.rectified_preview_photo = ImageTk.PhotoImage(grid)
            self.rectified_preview_canvas.delete("all")
            self.rectified_preview_canvas.create_image(
                grid.width // 2, grid.height // 2,
                image=self.rectified_preview_photo)
            self.rectified_preview_canvas.config(width=grid.width, height=grid.height)
            self.log_rectified(f"Preview: {num_frames} frames generated")
        except Exception as e:
            self.log_rectified(f"Preview error: {e}")

    def save_rectified_model(self):
        if not self.rectified_model: return
        fname = filedialog.asksaveasfilename(defaultextension=".pth",
                                             filetypes=[("PyTorch", "*.pth")])
        if not fname: return
        if self.rectified_model.use_ema and self.rectified_model.ema_model is not None:
            state = self.rectified_model.ema_model.state_dict()
        else:
            state = self.rectified_model.model.state_dict()
        d = {
            'model_state': state,
            'optimizer_state': self.rectified_model.optimizer.state_dict(),
            'flow_settings': {k: v.get() for k, v in self.flow_settings.items()},
            'global_settings': {k: v.get() for k, v in self.global_settings.items()},
            'use_ema': self.rectified_model.use_ema,
            'ema_decay': self.rectified_model.ema_decay,
            'ema_mode': self.rectified_model.ema_mode,
            'use_ot': self.rectified_model.use_ot,
            'channel_mult': self._parse_channel_mult(),
            'history': self.rectified_model.history,
            'attn_levels': self.rectified_model.model.attention_levels,
            'attn_num_heads': self.rectified_model.model.attn_num_heads,
            'attn_dropout': self.rectified_model.model.attn_dropout,
            'cond_mode': self.rectified_model.cond_mode,
            'button_names': list(self.button_names),
            'button_embed_dim': int(self.flow_settings['button_embed_dim'].get()),
            'button_to_key': dict(self.button_to_key),
            'in_channels': self.rectified_model.in_channels,
            'img_size': self.rectified_model.img_size,
        }
        if self.button_encoder is not None:
            d['button_encoder_state'] = self.button_encoder.state_dict()
        torch.save(d, fname)
        self.log_rectified(f"Flow model saved to {fname}")

    def load_rectified_model(self):
        fname = filedialog.askopenfilename(filetypes=[("PyTorch", "*.pth")])
        if not fname: return
        try:
            ckpt = torch.load(fname, map_location='cpu')
            fs = ckpt.get('flow_settings', {})
            for k, v in fs.items():
                if k in self.flow_settings:
                    try:
                        self.flow_settings[k].set(v)
                    except Exception:
                        pass
            for k, v in ckpt.get('global_settings', {}).items():
                if k in self.global_settings:
                    try:
                        self.global_settings[k].set(v)
                    except Exception:
                        pass
            if 'history' in ckpt:
                self.flow_settings['flow_history'].set(int(ckpt['history']))
            if 'channel_mult' in ckpt:
                cm = ckpt['channel_mult']
                if isinstance(cm, (list, tuple)):
                    self.flow_settings['channel_mult'].set(
                        ','.join(str(int(v)) for v in cm))
                else:
                    self.flow_settings['channel_mult'].set(str(cm))
            if 'attn_levels' in ckpt and len(ckpt['attn_levels']) >= 3:
                self.flow_settings['attn_level_0'].set(bool(ckpt['attn_levels'][0]))
                self.flow_settings['attn_level_1'].set(bool(ckpt['attn_levels'][1]))
                self.flow_settings['attn_level_2'].set(bool(ckpt['attn_levels'][2]))
            if 'attn_num_heads' in ckpt:
                self.flow_settings['attn_num_heads'].set(ckpt['attn_num_heads'])
            if 'attn_dropout' in ckpt:
                self.flow_settings['attn_dropout'].set(ckpt['attn_dropout'])
            if 'cond_mode' in ckpt:
                try:
                    self.flow_settings['cond_mode'].set(ckpt['cond_mode'])
                except Exception:
                    pass
            if 'button_names' in ckpt:
                self.button_names = list(ckpt['button_names'])
                self.button_to_idx = {n: i for i, n in enumerate(self.button_names)}
                self._update_vocab_label()
            if 'button_embed_dim' in ckpt:
                self.flow_settings['button_embed_dim'].set(int(ckpt['button_embed_dim']))
            if 'button_to_key' in ckpt:
                self.button_to_key = dict(ckpt['button_to_key'])
                self._refresh_bindings_ui()
            self._refresh_flow_head_dims()
            self.initialize_rectified_model()
            self.rectified_model.model.load_state_dict(ckpt['model_state'])
            try:
                self.rectified_model.optimizer.load_state_dict(ckpt['optimizer_state'])
            except Exception:
                pass
            if 'button_encoder_state' in ckpt and self.button_encoder is not None:
                try:
                    self.button_encoder.load_state_dict(ckpt['button_encoder_state'])
                except Exception as e:
                    self.log_rectified(f"Button encoder load error: {e}")
            self.log_rectified(f"Flow model loaded from {fname}")
        except Exception as e:
            import traceback; traceback.print_exc()
            self.log_rectified(f"Load error: {e}")


def main():
    multiprocessing.set_start_method('spawn', force=True)
    root = tk.Tk()
    app = DreamPixelLDMApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()