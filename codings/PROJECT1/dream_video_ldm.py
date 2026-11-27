# dream_video_ldm.py
"""
Dream — Video Latent Rectified Flow GUI (button-conditioned).

Reuses:
  - flow_api       (augmentation helpers, image loaders)
  - LDM_api        (FlexVAE)
  - video_ldm_api  (RectifiedFlowLatent, load_video_frames, VALID_COND_MODES)
"""
import os, csv as _csv, math, random, threading, queue
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image, ImageTk
from torch.utils.data import Dataset, DataLoader

try:
    import cv2
except ImportError:
    cv2 = None

from flow_api import (
    RandomJPEG, ElasticTransform, PaddedTransform,
    TV_ROT_HAS_PAD, TV_PERSP_HAS_PAD,
)
from LDM_api import FlexVAE, VAE_SIZE_CONFIGS
from video_ldm_api import (
    RectifiedFlowLatent, load_video_frames,
    _HAS_IMAGEIO, VALID_COND_MODES,
)


# ---------- Button → key choices ----------
KEY_CHOICES = (
    [""] + list("abcdefghijklmnopqrstuvwxyz") + list("0123456789") +
    ['Up', 'Down', 'Left', 'Right', 'space', 'Return', 'Tab', 'Escape',
     'Shift_L', 'Shift_R', 'Control_L', 'Control_R', 'Alt_L', 'Alt_R',
     'comma', 'period', 'slash', 'semicolon', 'apostrophe',
     'bracketleft', 'bracketright', 'minus', 'equal', 'backslash', 'grave',
     'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12']
)


# ---------- Button encoder (replaces TextEncoder) ----------
class ButtonEncoder(nn.Module):
    def __init__(self, n_buttons, cond_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(max(1, int(n_buttons)), cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

    def forward(self, buttons):
        return self.net(buttons.float())


# ---------- Loaded video ----------
class LoadedVideo:
    def __init__(self, video_path, csv_path, img_size=64):
        self.video_path = video_path
        self.csv_path = csv_path
        self.img_size = img_size
        self.name = os.path.basename(video_path)
        self.fps = 30.0
        self.raw_frames = []
        self.resized_frames = []
        self.raw_per_frame_buttons = []
        self.button_names = []
        self.start_frame = 0
        self.end_frame = None
        self.stride = 1
        self._frames_data = []
        self._actions_data = []
        self.current_idx = 0
        self._load_raw()
        self._rebuild()

    @property
    def frames_data(self): return self._frames_data
    @property
    def actions_data(self): return self._actions_data
    @property
    def total_raw_frames(self): return len(self.raw_frames)
    @property
    def end_frame_effective(self):
        return min(self.end_frame, len(self.raw_frames)) \
            if self.end_frame is not None else len(self.raw_frames)

    def _load_raw(self):
        if cv2 is not None:
            cap = cv2.VideoCapture(self.video_path)
            if cap.isOpened():
                self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                if self.fps <= 0: self.fps = 30.0
                while True:
                    ok, bgr = cap.read()
                    if not ok: break
                    self.raw_frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                cap.release()
        if not self.raw_frames:
            try:
                pil = Image.open(self.video_path)
                while True:
                    self.raw_frames.append(np.array(pil.convert('RGB')))
                    pil.seek(pil.tell() + 1)
            except EOFError:
                pass
            except Exception:
                pass
        if not self.raw_frames:
            raise RuntimeError("Video contained no readable frames.")

        rows = []
        with open(self.csv_path, newline='', encoding='utf-8') as f:
            reader = _csv.DictReader(f)
            fields = reader.fieldnames or []
            if 'timestamp' not in fields:
                raise RuntimeError("CSV missing 'timestamp' column.")
            for row in reader:
                try:
                    ts = float(row['timestamp'])
                except (KeyError, ValueError, TypeError):
                    continue
                names_str = (row.get('buttonspressed') or '').strip()
                names = set()
                if names_str:
                    for n in names_str.split('|'):
                        n = n.strip()
                        if n: names.add(n)
                rows.append((ts, names))

        all_names = set()
        for _, names in rows:
            all_names |= names
        self.button_names = sorted(all_names)

        n = len(self.raw_frames)
        self.raw_per_frame_buttons = [set() for _ in range(n)]
        for ts, names in rows:
            if not names: continue
            fi = int(round(ts * self.fps))
            if 0 <= fi < n:
                self.raw_per_frame_buttons[fi] |= names

        self.resized_frames = []
        for rgb in self.raw_frames:
            h, w = rgb.shape[:2]
            if h == self.img_size and w == self.img_size:
                self.resized_frames.append(rgb); continue
            if cv2 is not None:
                interp = cv2.INTER_AREA if h > self.img_size else cv2.INTER_LINEAR
                self.resized_frames.append(
                    cv2.resize(rgb, (self.img_size, self.img_size), interpolation=interp))
            else:
                resample = Image.LANCZOS if h > self.img_size else Image.BILINEAR
                self.resized_frames.append(
                    np.array(Image.fromarray(rgb).resize((self.img_size, self.img_size), resample)))

    def _rebuild(self):
        n = len(self.raw_frames)
        if n == 0:
            self._frames_data = []; self._actions_data = []; return
        start = max(0, min(int(self.start_frame), n - 1))
        end = max(start + 1, min(self.end_frame_effective, n))
        stride = max(1, int(self.stride))
        self._frames_data = []; self._actions_data = []
        for i in range(start, end, stride):
            combined = set()
            for j in range(i, min(i + stride, end)):
                combined |= self.raw_per_frame_buttons[j]
            action = np.zeros(len(self.button_names), dtype=np.float32)
            for k, name in enumerate(self.button_names):
                if name in combined: action[k] = 1.0
            self._frames_data.append(self.resized_frames[i])
            self._actions_data.append(action)
        self.current_idx = min(self.current_idx, max(0, len(self._frames_data) - 1))

    def set_range(self, start_frame, end_frame, stride):
        self.start_frame = max(0, int(start_frame))
        self.end_frame = None if end_frame is None else int(end_frame)
        self.stride = max(1, int(stride))
        self._rebuild()

    def reset(self):
        self.current_idx = 0
        return self.render()

    def step(self, action=None):
        if self._frames_data:
            self.current_idx = min(self.current_idx + 1, len(self._frames_data) - 1)
        return self.render(), 0.0

    def render(self):
        if not self._frames_data:
            return np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        return self._frames_data[self.current_idx]

    def status_text(self):
        if not self._frames_data: return "(no frames)"
        idx = self.current_idx; n = len(self._frames_data)
        t = (self.start_frame + idx * self.stride) / self.fps if self.fps > 0 else 0.0
        pressed = [self.button_names[i] for i, v in enumerate(self._actions_data[idx]) if v > 0.5]
        return f"Frame {idx}/{n - 1}   t={t:.2f}s   [{'|'.join(pressed) if pressed else '-'}]"


# ---------- Edit video dialog ----------
class EditVideoDialog(tk.Toplevel):
    def __init__(self, parent, video, on_save):
        super().__init__(parent)
        self.title(f"Edit — {video.name}")
        self.video = video; self.on_save = on_save
        self.transient(parent); self.grab_set(); self.resizable(False, False)
        n = video.total_raw_frames
        f = ttk.Frame(self, padding=12); f.pack(fill=tk.BOTH, expand=True)
        ttk.Label(f, text=f"{video.name}", font=("", 10, "bold")
                  ).grid(row=0, column=0, columnspan=2, sticky=tk.W)
        ttk.Label(f, text=f"Raw frames: {n}   fps: {video.fps:.2f}", foreground="gray"
                  ).grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(0, 8))
        ttk.Label(f, text="Start frame:").grid(row=2, column=0, sticky=tk.W, pady=3)
        self.start_var = tk.IntVar(value=int(video.start_frame))
        ttk.Spinbox(f, from_=0, to=max(0, n - 1), increment=1, width=12,
                    textvariable=self.start_var).grid(row=2, column=1, sticky=tk.W)
        ttk.Label(f, text="End frame (exclusive):").grid(row=3, column=0, sticky=tk.W, pady=3)
        self.end_var = tk.StringVar(value="" if video.end_frame is None else str(video.end_frame))
        ttk.Entry(f, textvariable=self.end_var, width=14).grid(row=3, column=1, sticky=tk.W)
        ttk.Label(f, text="(blank = end)", foreground="gray"
                  ).grid(row=3, column=2, padx=6, sticky=tk.W)
        ttk.Label(f, text="Frame stride:").grid(row=4, column=0, sticky=tk.W, pady=3)
        self.stride_var = tk.IntVar(value=int(video.stride))
        ttk.Spinbox(f, from_=1, to=100, increment=1, width=12,
                    textvariable=self.stride_var).grid(row=4, column=1, sticky=tk.W)
        self.info_var = tk.StringVar()
        ttk.Label(f, textvariable=self.info_var, foreground="#0055aa"
                  ).grid(row=5, column=0, columnspan=3, sticky=tk.W, pady=(8, 0))
        self._update_info()
        for v in (self.start_var, self.stride_var): v.trace_add('write', lambda *_: self._update_info())
        self.end_var.trace_add('write', lambda *_: self._update_info())
        btns = ttk.Frame(f); btns.grid(row=6, column=0, columnspan=3, pady=(12, 0), sticky=tk.E)
        ttk.Button(btns, text="Cancel", command=self.destroy).pack(side=tk.RIGHT, padx=4)
        ttk.Button(btns, text="Apply", command=self.apply).pack(side=tk.RIGHT, padx=4)

    def _update_info(self):
        n = self.video.total_raw_frames
        try: start = int(self.start_var.get())
        except Exception: start = 0
        try: stride = max(1, int(self.stride_var.get()))
        except Exception: stride = 1
        end_str = self.end_var.get().strip()
        try: end = int(end_str) if end_str else n
        except ValueError: end = n
        start = max(0, min(start, max(0, n - 1)))
        end = max(start + 1, min(end, n))
        count = max(0, (end - start + stride - 1) // stride)
        self.info_var.set(f"→ {count} processed frames")

    def apply(self):
        try: start = int(self.start_var.get())
        except Exception: start = 0
        end_str = self.end_var.get().strip()
        try: end = int(end_str) if end_str else None
        except ValueError:
            messagebox.showerror("Invalid", "End frame must be a number or blank.", parent=self)
            return
        try: stride = int(self.stride_var.get())
        except Exception: stride = 1
        try: self.video.set_range(start, end, stride)
        except Exception as e:
            messagebox.showerror("Error", str(e), parent=self); return
        self.on_save(); self.destroy()


# ---------- Datasets ----------
class ButtonVideoDataset(Dataset):
    def __init__(self, videos, global_buttons, history, img_size, color_mode, aug_settings):
        self.history = max(1, int(history))
        self.img_size = img_size
        self.color_mode = color_mode.lower()
        self.n_buttons = max(1, len(global_buttons))
        name_to_g = {n: i for i, n in enumerate(global_buttons)}

        self.remapped_actions = []
        for v in videos:
            N = len(v.frames_data)
            vec = np.zeros((N, self.n_buttons), dtype=np.float32)
            if N > 0 and len(v.actions_data) == N and v.button_names:
                arr = np.stack(v.actions_data)
                for local_i, name in enumerate(v.button_names):
                    if name in name_to_g:
                        vec[:, name_to_g[name]] = arr[:, local_i]
            self.remapped_actions.append(vec)

        self.samples = []
        self.videos = videos
        for vi, v in enumerate(videos):
            N = len(v.frames_data)
            for start in range(-self.history, N - self.history):
                self.samples.append((vi, start))

        self.aug_settings = aug_settings or {}
        pad = self.aug_settings.get('aug_padding_mode', 'edge')
        self.aug_padding_mode = pad if pad in ('constant', 'edge', 'reflect', 'symmetric') else 'edge'
        self.aug_pipeline = self._build_aug_pipeline()

        self.ch = 3 if self.color_mode == 'rgb' else 1
        norm = (0.5,) * self.ch
        self.base_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(norm, norm),
        ])

    def _build_aug_pipeline(self):
        a = self.aug_settings; pad = self.aug_padding_mode; pipe = []
        if a.get('flip_horizontal', False):
            pipe.append(transforms.RandomHorizontalFlip(p=0.5))
        if a.get('rotation', False):
            if TV_ROT_HAS_PAD:
                pipe.append(transforms.RandomRotation(30, interpolation=Image.BICUBIC,
                                                     expand=False, fill=0, padding_mode=pad))
            else:
                base = transforms.RandomRotation(30, interpolation=Image.BICUBIC,
                                                 expand=False, fill=0)
                pipe.append(PaddedTransform(base, pad_fraction=0.22, padding_mode=pad))
        if a.get('random_crop', False):
            sc = a.get('crop_scale', 0.8)
            pipe.append(transforms.RandomResizedCrop(size=self.img_size,
                                                     scale=(sc, 1.0),
                                                     interpolation=Image.BICUBIC))
        if a.get('color_jitter', False):
            pipe.append(transforms.ColorJitter(
                brightness=a.get('brightness', 0.2), contrast=a.get('contrast', 0.2),
                saturation=a.get('saturation', 0.2), hue=a.get('hue', 0.1)))
        if a.get('random_perspective', False):
            dist = float(a.get('perspective_distortion', 0.1))
            if TV_PERSP_HAS_PAD:
                pipe.append(transforms.RandomPerspective(distortion_scale=dist, p=0.5,
                                                         interpolation=Image.BICUBIC,
                                                         fill=0, padding_mode=pad))
            else:
                base = transforms.RandomPerspective(distortion_scale=dist, p=0.5,
                                                    interpolation=Image.BICUBIC, fill=0)
                frac = min(0.35, max(0.20, dist * 0.5 + 0.15))
                pipe.append(PaddedTransform(base, pad_fraction=frac, padding_mode=pad))
        if a.get('elastic_transform', False):
            pipe.append(ElasticTransform(alpha=a.get('elastic_alpha', 30),
                                         sigma=a.get('elastic_sigma', 3),
                                         p=0.5, padding_mode=pad))
        if a.get('jpeg_compression', False):
            pipe.append(RandomJPEG(quality_low=a.get('jpeg_quality_low', 50),
                                   quality_high=a.get('jpeg_quality_high', 95), p=0.5))
        if a.get('stretch_vertical', False):
            pipe.append(transforms.RandomResizedCrop(
                size=self.img_size, scale=(0.8, 1.0), ratio=(0.5, 1.0),
                interpolation=Image.BICUBIC))
        if a.get('stretch_horizontal', False):
            pipe.append(transforms.RandomResizedCrop(
                size=self.img_size, scale=(0.8, 1.0), ratio=(1.0, 2.0),
                interpolation=Image.BICUBIC))
        return pipe

    def _apply_aug(self, pil):
        for aug in self.aug_pipeline:
            pil = aug(pil)
        return pil

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vi, start = self.samples[idx]
        v = self.videos[vi]
        frames = v.frames_data
        actions = self.remapped_actions[vi]
        seed = random.randint(0, 2**31 - 1)
        seq = []
        for i in range(self.history + 1):
            fi = start + i
            if 0 <= fi < len(frames):
                try:
                    random.seed(seed); torch.manual_seed(seed)
                    pil = Image.fromarray(frames[fi])
                    pil = self._apply_aug(pil)
                    t = self.base_transform(pil)
                except Exception:
                    t = torch.zeros(self.ch, self.img_size, self.img_size)
            else:
                t = torch.zeros(self.ch, self.img_size, self.img_size)
            seq.append(t)
        seq_t = torch.stack(seq, dim=0)
        target_fi = start + self.history
        if 0 <= target_fi < len(actions):
            button = torch.from_numpy(actions[target_fi].copy())
        else:
            button = torch.zeros(self.n_buttons, dtype=torch.float32)
        return seq_t, button


class FrameOnlyDataset(Dataset):
    def __init__(self, frames_list, img_size, color_mode, aug_settings):
        self.frames = frames_list
        self.img_size = img_size
        self.color_mode = color_mode.lower()
        self.aug_settings = aug_settings or {}
        pad = self.aug_settings.get('aug_padding_mode', 'edge')
        self.aug_padding_mode = pad if pad in ('constant', 'edge', 'reflect', 'symmetric') else 'edge'
        self.aug_pipeline = self._build_aug_pipeline()
        self.ch = 3 if self.color_mode == 'rgb' else 1
        norm = (0.5,) * self.ch
        self.base_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(norm, norm),
        ])

    def _build_aug_pipeline(self):
        a = self.aug_settings; pad = self.aug_padding_mode; pipe = []
        if a.get('flip_horizontal', False):
            pipe.append(transforms.RandomHorizontalFlip(p=0.5))
        if a.get('rotation', False):
            if TV_ROT_HAS_PAD:
                pipe.append(transforms.RandomRotation(30, interpolation=Image.BICUBIC,
                                                     expand=False, fill=0, padding_mode=pad))
            else:
                base = transforms.RandomRotation(30, interpolation=Image.BICUBIC,
                                                 expand=False, fill=0)
                pipe.append(PaddedTransform(base, pad_fraction=0.22, padding_mode=pad))
        if a.get('color_jitter', False):
            pipe.append(transforms.ColorJitter(
                brightness=a.get('brightness', 0.2), contrast=a.get('contrast', 0.2),
                saturation=a.get('saturation', 0.2), hue=a.get('hue', 0.1)))
        if a.get('jpeg_compression', False):
            pipe.append(RandomJPEG(quality_low=a.get('jpeg_quality_low', 50),
                                   quality_high=a.get('jpeg_quality_high', 95), p=0.5))
        return pipe

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        try:
            pil = Image.fromarray(self.frames[i])
            for aug in self.aug_pipeline:
                pil = aug(pil)
            return self.base_transform(pil),
        except Exception:
            return torch.zeros(self.ch, self.img_size, self.img_size),


def _default_aug_dict():
    def V(v): return tk.BooleanVar(value=v)
    def I(v): return tk.IntVar(value=v)
    def D(v): return tk.DoubleVar(value=v)
    def S(v): return tk.StringVar(value=v)
    return {
        'flip_horizontal': V(True), 'rotation': V(False),
        'random_crop': V(False), 'crop_scale': D(0.8),
        'color_jitter': V(False), 'brightness': D(0.2), 'contrast': D(0.2),
        'saturation': D(0.2), 'hue': D(0.1),
        'random_perspective': V(False), 'perspective_distortion': D(0.1),
        'elastic_transform': V(False), 'elastic_alpha': I(30), 'elastic_sigma': I(3),
        'jpeg_compression': V(False), 'jpeg_quality_low': I(50), 'jpeg_quality_high': I(95),
        'stretch_vertical': V(False), 'stretch_horizontal': V(False),
        'aug_padding_mode': S('edge'),
    }


# ==================== Main App ====================
class App:
    def __init__(self, root):
        self.root = root
        root.title("Dream — Video Latent Rectified Flow (button-conditioned)")
        root.geometry("1280x1000")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.videos = []
        self.current_video_idx = -1
        self.game = None
        self.button_to_key = {}
        self.keys = {}

        self.vae = None
        self.vae_optimizer = None
        self.flow = None
        self.button_encoder = None

        self.inference_running = False
        self.inference_stop = False
        self._playback_running = False
        self.advance_one_frame = False

        self.vae_stop = False
        self.flow_stop = False
        self.vae_queue = queue.Queue()
        self.flow_queue = queue.Queue()

        self.record_photo = None
        self.inference_photo = None

        self._build_settings()
        self.setup_ui()
        root.bind('<KeyPress>', self._on_key_press)
        root.bind('<KeyRelease>', self._on_key_release)
        self._render_record_view()

    # ---------- Settings vars ----------
    def _build_settings(self):
        def V(v): return tk.IntVar(value=v)
        def D(v): return tk.DoubleVar(value=v)
        def S(v): return tk.StringVar(value=v)
        def B(v): return tk.BooleanVar(value=v)

        self.global_s = {'img_size': V(32), 'color_mode': S('rgb')}
        self.vae_s = {
            'vae_size': S('big'),
            'vae_base_channels': V(32),
            'vae_latent_channels': V(8),
            'vae_latent_h': V(4), 'vae_latent_w': V(4),
            'vae_batch_size': V(16), 'vae_lr': D(1e-3),
            'vae_kl_weight': D(1e-4), 'vae_num_workers': V(0),
            'vae_epochs': V(100),
        }
        self.flow_s = {
            'flow_base_channels': V(32),
            'flow_padding_mode': S('constant'),
            'flow_history': V(1),
            'flow_batch_size': V(16),
            'flow_lr': D(2e-4),
            'flow_cfg_dropout': D(0.1),
            'flow_epochs': V(200),
            'flow_cond_dim': V(64),
            'flow_cond_mode': S('shift'),
            'flow_use_ot': B(False),
            'flow_ot_method': S('sinkhorn'),
            'flow_ot_eps': D(0.05),
            'flow_ot_iters': V(50),
            'flow_attn_level_0': B(True),
            'flow_attn_level_1': B(True),
            'flow_attn_level_2': B(True),
            'flow_attn_num_heads': V(4),
            'flow_attn_dropout': D(0.0),
            'flow_ema_enabled': B(False),
            'flow_ema_decay': D(0.999),
            'flow_ema_mode': S('standard'),
            'flow_preview_enabled': B(True),
            'flow_preview_freq': V(5),
        }
        self.vae_aug = _default_aug_dict()
        self.flow_aug = _default_aug_dict()
        self.inf_num_frames = V(16)
        self.inf_steps = V(50)
        self.inf_method = S('euler')
        self.inf_cfg = D(1.0)
        self.inf_progressive = B(True)
        self.inf_prog_interval = V(10)
        self.inf_start_frame = V(0)
        self.inf_fps = V(10)
        self.inf_frame_wait = B(False)

    # ---------- UI ----------
    def setup_ui(self):
        nb = ttk.Notebook(self.root)
        nb.pack(fill=tk.BOTH, expand=True)
        for name, builder in [("Videos", self.tab_videos),
                              ("Train VAE", self.tab_vae),
                              ("Train Flow", self.tab_flow),
                              ("Settings", self.tab_settings),
                              ("Inference", self.tab_inference)]:
            f = ttk.Frame(nb); nb.add(f, text=name); builder(f)
        self.status = tk.Label(self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)
        self.root.after(100, self._poll_vae_q)
        self.root.after(100, self._poll_flow_q)

    # ----- Videos tab -----
    def tab_videos(self, parent):
        f = ttk.Frame(parent); f.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        left = tk.Canvas(f, width=512, height=512, bg='black')
        left.pack(side=tk.LEFT, padx=10)
        left.bind('<Button-1>', lambda e: left.focus_set())
        self.record_canvas = left

        right = ttk.Frame(f); right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        ttk.Label(right, text="Loaded videos", font=("", 10, "bold")).pack(anchor=tk.W)

        tree_f = ttk.Frame(right); tree_f.pack(fill=tk.BOTH, expand=True, pady=(4, 6))
        cols = ("name", "range", "stride", "frames", "buttons")
        heads = (("name", "Name", 180), ("range", "Range", 150), ("stride", "Stride", 55),
                 ("frames", "Frames", 60), ("buttons", "Buttons", 180))
        self.video_tree = ttk.Treeview(tree_f, columns=cols, show="headings",
                                        height=12, selectmode="browse")
        for c, t, w in heads:
            self.video_tree.heading(c, text=t)
            self.video_tree.column(c, width=w, anchor="w" if c in ("name", "range", "buttons") else "center",
                                    stretch=False)
        vs = ttk.Scrollbar(tree_f, orient="vertical", command=self.video_tree.yview)
        self.video_tree.configure(yscrollcommand=vs.set)
        self.video_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vs.pack(side=tk.RIGHT, fill=tk.Y)
        self.video_tree.bind("<<TreeviewSelect>>", self._on_video_select)
        self.video_tree.bind("<Double-1>", lambda e: self.edit_video())

        row = ttk.Frame(right); row.pack(fill=tk.X, pady=2)
        ttk.Button(row, text="Add Video + CSV…", command=self.add_video).pack(side=tk.LEFT)
        ttk.Button(row, text="Edit", command=self.edit_video).pack(side=tk.LEFT, padx=6)
        ttk.Button(row, text="Remove", command=self.remove_video).pack(side=tk.LEFT)
        ttk.Button(row, text="Clear All", command=self.clear_videos).pack(side=tk.LEFT, padx=6)

        ttk.Separator(right, orient='horizontal').pack(fill=tk.X, pady=6)
        pb = ttk.LabelFrame(right, text="Playback"); pb.pack(fill=tk.X, pady=2)
        prow = ttk.Frame(pb); prow.pack(fill=tk.X, padx=6, pady=4)
        ttk.Button(prow, text="Play / Pause", command=self.toggle_play).pack(side=tk.LEFT)
        ttk.Button(prow, text="Reset", command=self.reset_play).pack(side=tk.LEFT, padx=6)
        self.playback_lbl = ttk.Label(pb, text="(no video)", foreground="#333", wraplength=500)
        self.playback_lbl.pack(anchor=tk.W, padx=6, pady=(0, 4))

        ttk.Separator(right, orient='horizontal').pack(fill=tk.X, pady=6)
        info = ttk.LabelFrame(right, text="Dataset"); info.pack(fill=tk.X)
        self.dataset_lbl = ttk.Label(info, text="(no videos)", wraplength=500, justify=tk.LEFT)
        self.dataset_lbl.pack(anchor=tk.W, padx=6, pady=6)

    # ----- VAE tab -----
    def tab_vae(self, parent):
        f = ttk.Frame(parent); f.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        ctrl = ttk.Frame(f); ctrl.pack(fill=tk.X, pady=4)
        ttk.Button(ctrl, text="Init VAE", command=self.init_vae).pack(side=tk.LEFT)
        ttk.Button(ctrl, text="Start Training", command=self.start_vae_train).pack(side=tk.LEFT, padx=4)
        ttk.Button(ctrl, text="Stop", command=self.stop_vae_train).pack(side=tk.LEFT)
        ttk.Button(ctrl, text="Save VAE", command=self.save_vae).pack(side=tk.LEFT, padx=8)
        ttk.Button(ctrl, text="Load VAE", command=self.load_vae).pack(side=tk.LEFT)
        self.vae_prog = ttk.Progressbar(f, orient=tk.HORIZONTAL, mode='determinate')
        self.vae_prog.pack(fill=tk.X, pady=4)
        self.vae_status = ttk.Label(f, text="Not initialized"); self.vae_status.pack(anchor=tk.W)
        self.vae_canvas = tk.Canvas(f, width=512, height=256, bg='black')
        self.vae_canvas.pack(pady=6)
        log_f = ttk.LabelFrame(f, text="Log"); log_f.pack(fill=tk.BOTH, expand=True)
        self.vae_log = tk.Text(log_f, height=12, font=("Courier", 9))
        self.vae_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = ttk.Scrollbar(log_f, command=self.vae_log.yview); sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.vae_log.config(yscrollcommand=sb.set)

    # ----- Flow tab -----
    def tab_flow(self, parent):
        f = ttk.Frame(parent); f.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        ctrl = ttk.Frame(f); ctrl.pack(fill=tk.X, pady=4)
        ttk.Button(ctrl, text="Init Flow", command=self.init_flow).pack(side=tk.LEFT)
        ttk.Button(ctrl, text="Start Training", command=self.start_flow_train).pack(side=tk.LEFT, padx=4)
        ttk.Button(ctrl, text="Stop", command=self.stop_flow_train).pack(side=tk.LEFT)
        ttk.Button(ctrl, text="Save Flow", command=self.save_flow).pack(side=tk.LEFT, padx=8)
        ttk.Button(ctrl, text="Load Flow", command=self.load_flow).pack(side=tk.LEFT)
        ttk.Button(ctrl, text="Preview", command=self.preview_flow).pack(side=tk.LEFT, padx=8)
        self.flow_prog = ttk.Progressbar(f, orient=tk.HORIZONTAL, mode='determinate')
        self.flow_prog.pack(fill=tk.X, pady=4)
        self.flow_status = ttk.Label(f, text="Not initialized"); self.flow_status.pack(anchor=tk.W)
        self.flow_canvas = tk.Canvas(f, width=512, height=256, bg='black')
        self.flow_canvas.pack(pady=6)
        log_f = ttk.LabelFrame(f, text="Log"); log_f.pack(fill=tk.BOTH, expand=True)
        self.flow_log = tk.Text(log_f, height=12, font=("Courier", 9))
        self.flow_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = ttk.Scrollbar(log_f, command=self.flow_log.yview); sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.flow_log.config(yscrollcommand=sb.set)

    # ----- Settings tab -----
    def tab_settings(self, parent):
        canvas = tk.Canvas(parent); sb = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        # ---- These helpers correctly take `parent` as first arg ----
        def spin(parent, lbl, var, lo, hi):
            r = ttk.Frame(parent); r.pack(fill=tk.X, pady=1)
            ttk.Label(r, text=lbl, width=24, anchor='w').pack(side=tk.LEFT)
            ttk.Spinbox(r, from_=lo, to=hi, textvariable=var, width=10).pack(side=tk.RIGHT)

        def entry(parent, lbl, var):
            r = ttk.Frame(parent); r.pack(fill=tk.X, pady=1)
            ttk.Label(r, text=lbl, width=24, anchor='w').pack(side=tk.LEFT)
            ttk.Entry(r, textvariable=var, width=12).pack(side=tk.RIGHT)

        def combo(parent, lbl, var, vals):
            r = ttk.Frame(parent); r.pack(fill=tk.X, pady=1)
            ttk.Label(r, text=lbl, width=24, anchor='w').pack(side=tk.LEFT)
            ttk.Combobox(r, textvariable=var, values=vals, state='readonly',
                         width=12).pack(side=tk.RIGHT)

        gf = ttk.LabelFrame(inner, text="Global"); gf.pack(fill=tk.X, pady=6)
        combo(gf, "Color mode:", self.global_s['color_mode'], ['rgb', 'grayscale'])
        spin(gf, "Image size:", self.global_s['img_size'], 16, 128)
        ttk.Label(gf, text="(reload videos after changing img_size)",
                  foreground="gray").pack(anchor=tk.W)

        vf = ttk.LabelFrame(inner, text="VAE"); vf.pack(fill=tk.X, pady=6)
        combo(vf, "VAE size:", self.vae_s['vae_size'], list(VAE_SIZE_CONFIGS.keys()))
        spin(vf, "Base channels:", self.vae_s['vae_base_channels'], 4, 256)
        spin(vf, "Latent channels:", self.vae_s['vae_latent_channels'], 1, 64)
        spin(vf, "Latent H:", self.vae_s['vae_latent_h'], 1, 64)
        spin(vf, "Latent W:", self.vae_s['vae_latent_w'], 1, 64)
        spin(vf, "Batch size:", self.vae_s['vae_batch_size'], 1, 128)
        entry(vf, "LR:", self.vae_s['vae_lr'])
        entry(vf, "KL weight:", self.vae_s['vae_kl_weight'])
        spin(vf, "Workers:", self.vae_s['vae_num_workers'], 0, 8)
        spin(vf, "Epochs:", self.vae_s['vae_epochs'], 1, 2000)

        ff = ttk.LabelFrame(inner, text="Flow"); ff.pack(fill=tk.X, pady=6)
        spin(ff, "Base channels:", self.flow_s['flow_base_channels'], 4, 256)
        combo(ff, "Conv padding:", self.flow_s['flow_padding_mode'],
              ['constant', 'reflect', 'replicate', 'circular'])
        spin(ff, "History:", self.flow_s['flow_history'], 1, 10)
        spin(ff, "Cond dim:", self.flow_s['flow_cond_dim'], 8, 512)
        combo(ff, "Cond mode:", self.flow_s['flow_cond_mode'], list(VALID_COND_MODES))
        spin(ff, "Batch size:", self.flow_s['flow_batch_size'], 1, 64)
        entry(ff, "LR:", self.flow_s['flow_lr'])
        entry(ff, "CFG dropout:", self.flow_s['flow_cfg_dropout'])
        spin(ff, "Epochs:", self.flow_s['flow_epochs'], 1, 5000)
        ttk.Label(ff, text="Attention (per level):").pack(anchor='w', pady=(6, 0))
        ar = ttk.Frame(ff); ar.pack(fill=tk.X)
        ttk.Checkbutton(ar, text="L0", variable=self.flow_s['flow_attn_level_0']).pack(side=tk.LEFT)
        ttk.Checkbutton(ar, text="L1", variable=self.flow_s['flow_attn_level_1']).pack(side=tk.LEFT)
        ttk.Checkbutton(ar, text="L2", variable=self.flow_s['flow_attn_level_2']).pack(side=tk.LEFT)
        spin(ff, "Attn heads:", self.flow_s['flow_attn_num_heads'], 1, 32)
        entry(ff, "Attn dropout:", self.flow_s['flow_attn_dropout'])
        ttk.Checkbutton(ff, text="Use Sinkhorn OT", variable=self.flow_s['flow_use_ot']
                        ).pack(anchor='w')
        combo(ff, "OT method:", self.flow_s['flow_ot_method'], ['sinkhorn', 'hungarian'])
        entry(ff, "OT epsilon:", self.flow_s['flow_ot_eps'])
        spin(ff, "OT iters:", self.flow_s['flow_ot_iters'], 1, 500)

        ef = ttk.LabelFrame(inner, text="EMA"); ef.pack(fill=tk.X, pady=6)
        ttk.Checkbutton(ef, text="Enable EMA",
                        variable=self.flow_s['flow_ema_enabled']).pack(anchor='w')
        entry(ef, "Decay:", self.flow_s['flow_ema_decay'])
        combo(ef, "Mode:", self.flow_s['flow_ema_mode'], ['standard', 'lookahead'])

        pf = ttk.LabelFrame(inner, text="Preview"); pf.pack(fill=tk.X, pady=6)
        ttk.Checkbutton(pf, text="Enable",
                        variable=self.flow_s['flow_preview_enabled']).pack(anchor='w')
        spin(pf, "Every N epochs:", self.flow_s['flow_preview_freq'], 1, 100)

        augf = ttk.LabelFrame(inner, text="Augmentation"); augf.pack(fill=tk.X, pady=6)
        ttk.Label(augf, text="Aug padding mode:").pack(anchor='w')
        ttk.Combobox(augf, textvariable=self.vae_aug['aug_padding_mode'],
                     values=['edge', 'reflect', 'symmetric', 'constant'],
                     state='readonly', width=12).pack(anchor='w')
        ttk.Checkbutton(augf, text="Horizontal Flip",
                        variable=self.vae_aug['flip_horizontal']).pack(anchor='w')
        ttk.Checkbutton(augf, text="Rotation",
                        variable=self.vae_aug['rotation']).pack(anchor='w')
        ttk.Checkbutton(augf, text="Color Jitter",
                        variable=self.vae_aug['color_jitter']).pack(anchor='w')
        ttk.Checkbutton(augf, text="JPEG",
                        variable=self.vae_aug['jpeg_compression']).pack(anchor='w')

        kf = ttk.LabelFrame(inner, text="Button → Key"); kf.pack(fill=tk.X, pady=6)
        self.key_frame = ttk.Frame(kf); self.key_frame.pack(fill=tk.X, padx=4, pady=4)
        self._rebuild_key_map_ui()

    # ----- Inference tab -----
    def tab_inference(self, parent):
        f = ttk.Frame(parent); f.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        left = tk.Canvas(f, width=512, height=512, bg='black')
        left.pack(side=tk.LEFT, padx=10)
        left.bind('<Button-1>', lambda e: left.focus_set())
        self.inference_canvas = left

        right = ttk.Frame(f); right.pack(side=tk.LEFT, fill=tk.Y, padx=10)
        ttk.Label(right, text="Inference", font=("", 10, "bold")).pack(anchor=tk.W)

        src = ttk.LabelFrame(right, text="Seed source"); src.pack(fill=tk.X, pady=4)
        r1 = ttk.Frame(src); r1.pack(fill=tk.X, pady=2)
        ttk.Label(r1, text="Video:").pack(side=tk.LEFT)
        self.inf_video_var = tk.StringVar()
        self.inf_video_combo = ttk.Combobox(r1, textvariable=self.inf_video_var,
                                            state='readonly', width=24)
        self.inf_video_combo.pack(side=tk.LEFT, padx=4)
        r2 = ttk.Frame(src); r2.pack(fill=tk.X, pady=2)
        ttk.Label(r2, text="Start frame:").pack(side=tk.LEFT)
        ttk.Spinbox(r2, from_=0, to=99999,
                    textvariable=self.inf_start_frame, width=8).pack(side=tk.LEFT, padx=4)

        ctl = ttk.LabelFrame(right, text="Generation"); ctl.pack(fill=tk.X, pady=4)
        r3 = ttk.Frame(ctl); r3.pack(fill=tk.X, pady=2)
        ttk.Label(r3, text="Frames:").pack(side=tk.LEFT)
        ttk.Spinbox(r3, from_=1, to=200,
                    textvariable=self.inf_num_frames, width=6).pack(side=tk.LEFT, padx=4)
        ttk.Label(r3, text="Steps/frame:").pack(side=tk.LEFT, padx=(10, 0))
        ttk.Spinbox(r3, from_=1, to=500,
                    textvariable=self.inf_steps, width=6).pack(side=tk.LEFT, padx=4)
        r4 = ttk.Frame(ctl); r4.pack(fill=tk.X, pady=2)
        ttk.Label(r4, text="Sampler:").pack(side=tk.LEFT)
        ttk.Combobox(r4, textvariable=self.inf_method,
                     values=['euler', 'heun', 'midpoint', 'rk4'],
                     state='readonly', width=10).pack(side=tk.LEFT, padx=4)
        ttk.Label(r4, text="CFG:").pack(side=tk.LEFT, padx=(10, 0))
        ttk.Entry(r4, textvariable=self.inf_cfg, width=6).pack(side=tk.LEFT, padx=4)

        r5 = ttk.Frame(ctl); r5.pack(fill=tk.X, pady=2)
        ttk.Checkbutton(r5, text="Progressive preview",
                        variable=self.inf_progressive).pack(side=tk.LEFT)
        ttk.Label(r5, text="Update every N steps:").pack(side=tk.LEFT, padx=(8, 0))
        ttk.Spinbox(r5, from_=1, to=100,
                    textvariable=self.inf_prog_interval, width=4).pack(side=tk.LEFT, padx=4)

        r6 = ttk.Frame(ctl); r6.pack(fill=tk.X, pady=2)
        ttk.Label(r6, text="Playback FPS:").pack(side=tk.LEFT)
        ttk.Spinbox(r6, from_=1, to=60,
                    textvariable=self.inf_fps, width=4).pack(side=tk.LEFT, padx=4)
        ttk.Checkbutton(r6, text="Frame wait (step per key press release)",
                        variable=self.inf_frame_wait).pack(side=tk.LEFT, padx=8)

        r7 = ttk.Frame(right); r7.pack(fill=tk.X, pady=4)
        self.btn_start = ttk.Button(r7, text="Start Dream", command=self.start_inference)
        self.btn_start.pack(side=tk.LEFT)
        self.btn_stop = ttk.Button(r7, text="Stop", command=self.stop_inference, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=4)
        ttk.Button(r7, text="Reset", command=self.reset_inference).pack(side=tk.LEFT)

        self.inf_status = ttk.Label(right, text="Ready", foreground="#0055aa",
                                    wraplength=320, justify=tk.LEFT)
        self.inf_status.pack(anchor=tk.W, pady=4)

        ttk.Label(right,
                  text=("Controls:\n"
                        "• Buttons via assigned keys (Settings tab)\n"
                        "• Frame-wait: one frame per key press release\n"
                        "• Uses the selected video as seed"),
                  justify=tk.LEFT, foreground="gray").pack(anchor=tk.W, pady=6)

        self._render_inference_blank()

    # ---------- Display helpers ----------
    def _render_record_view(self):
        if self.game is not None:
            frame = self.game.render()
        else:
            sz = int(self.global_s['img_size'].get())
            frame = np.zeros((sz, sz, 3), dtype=np.uint8)
        self._show(frame, self.record_canvas, 'record_photo')

    def _render_inference_blank(self):
        sz = int(self.global_s['img_size'].get())
        self._show(np.zeros((sz, sz, 3), dtype=np.uint8),
                   self.inference_canvas, 'inference_photo')

    def _show(self, arr, canvas, photo_attr, size=512):
        img = Image.fromarray(arr).resize((size, size), Image.NEAREST)
        photo = ImageTk.PhotoImage(img)
        setattr(self, photo_attr, photo)
        canvas.delete("all")
        canvas.create_image(0, 0, anchor=tk.NW, image=photo)

    # ---------- Key events ----------
    def _on_key_press(self, event):
        if isinstance(self.root.focus_get(), (tk.Entry, ttk.Entry, ttk.Combobox)):
            return
        ks = event.keysym
        if len(ks) == 1 and ks.isalpha(): ks = ks.lower()
        self.keys[ks] = True
        if ks in ('Up', 'Down', 'Left', 'Right', 'space'):
            if ks != 'space': self.advance_one_frame = True
            return "break"

    def _on_key_release(self, event):
        if isinstance(self.root.focus_get(), (tk.Entry, ttk.Entry, ttk.Combobox)):
            return
        ks = event.keysym
        if len(ks) == 1 and ks.isalpha(): ks = ks.lower()
        self.keys[ks] = False
        if ks in ('Up', 'Down', 'Left', 'Right', 'space'):
            return "break"

    # ---------- Video management ----------
    def _global_buttons(self):
        names = set()
        for v in self.videos:
            names |= set(v.button_names)
        return sorted(names)

    def _sync_button_keys(self):
        g = set(self._global_buttons())
        self.button_to_key = {n: self.button_to_key.get(n, "") for n in g}
        self._rebuild_key_map_ui()
        self._refresh_video_combo()

    def _rebuild_key_map_ui(self):
        for w in self.key_frame.winfo_children(): w.destroy()
        if not self.button_to_key:
            ttk.Label(self.key_frame, text="(load a video to configure)",
                      foreground="gray").pack(anchor=tk.W); return
        for name in sorted(self.button_to_key.keys()):
            row = ttk.Frame(self.key_frame); row.pack(fill=tk.X, pady=1)
            ttk.Label(row, text=name, width=20).pack(side=tk.LEFT)
            var = tk.StringVar(value=self.button_to_key[name])
            cb = ttk.Combobox(row, textvariable=var, values=KEY_CHOICES,
                              state='readonly', width=14)
            cb.pack(side=tk.LEFT, padx=4)
            cb.bind('<<ComboboxSelected>>',
                    lambda e, n=name, v=var: self.button_to_key.__setitem__(n, v.get()))

    def _refresh_video_combo(self):
        names = [v.name for v in self.videos]
        self.inf_video_combo['values'] = names
        if names and self.inf_video_var.get() not in names:
            self.inf_video_var.set(names[0])

    def _refresh_video_tree(self):
        self.video_tree.delete(*self.video_tree.get_children())
        for i, v in enumerate(self.videos):
            end = "end" if v.end_frame is None else str(v.end_frame)
            self.video_tree.insert("", "end", iid=str(i), values=(
                v.name,
                f"{v.start_frame}–{end}  (raw {v.total_raw_frames})",
                f"{v.stride}",
                f"{len(v.frames_data)}",
                ", ".join(v.button_names) if v.button_names else "(none)",
            ))
        if 0 <= self.current_video_idx < len(self.videos):
            iid = str(self.current_video_idx)
            if self.video_tree.exists(iid):
                self.video_tree.selection_set(iid)
                self.video_tree.focus(iid)

    def _on_video_select(self, _=None):
        sel = self.video_tree.selection()
        if not sel: return
        idx = int(sel[0])
        if 0 <= idx < len(self.videos):
            self.current_video_idx = idx
            self.game = self.videos[idx]
            self._playback_running = False
            self._render_record_view()
            self.playback_lbl.config(text=self.game.status_text())
            self.inf_video_var.set(self.game.name)

    def add_video(self):
        if cv2 is None and not _HAS_IMAGEIO:
            messagebox.showerror("Missing deps", "Install opencv-python or imageio[ffmpeg].")
            return
        vp = filedialog.askopenfilename(
            title="Select video",
            filetypes=[("Video / GIF",
                        "*.mp4 *.avi *.mov *.mkv *.webm *.gif *.m4v *.mpg *.mpeg"),
                       ("All files", "*.*")])
        if not vp: return
        cp = filedialog.askopenfilename(title="Select button-press CSV",
                                        filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if not cp: return
        img_sz = int(self.global_s['img_size'].get())
        try:
            v = LoadedVideo(vp, cp, img_sz)
        except Exception as e:
            messagebox.showerror("Load failed", str(e)); return
        self.videos.append(v)
        self._sync_button_keys()
        self._refresh_video_tree()
        self._update_dataset_lbl()
        self.current_video_idx = len(self.videos) - 1
        self._on_video_select()
        EditVideoDialog(self.root, v, on_save=self._after_video_edit)

    def edit_video(self):
        sel = self.video_tree.selection()
        if not sel: return
        idx = int(sel[0])
        if 0 <= idx < len(self.videos):
            EditVideoDialog(self.root, self.videos[idx], on_save=self._after_video_edit)

    def _after_video_edit(self):
        self._refresh_video_tree()
        self._update_dataset_lbl()
        if 0 <= self.current_video_idx < len(self.videos):
            self._render_record_view()
            self.playback_lbl.config(text=self.videos[self.current_video_idx].status_text())

    def remove_video(self):
        sel = self.video_tree.selection()
        if not sel: return
        idx = int(sel[0])
        del self.videos[idx]
        if self.current_video_idx >= len(self.videos):
            self.current_video_idx = len(self.videos) - 1
        self._sync_button_keys()
        self._refresh_video_tree()
        self._update_dataset_lbl()
        if self.videos:
            self._on_video_select()
        else:
            self.game = None
            self.playback_lbl.config(text="(no video)")
            self._render_record_view()

    def clear_videos(self):
        if not self.videos: return
        if not messagebox.askyesno("Clear", "Remove all videos?"): return
        self.videos = []
        self.current_video_idx = -1
        self.game = None
        self.button_to_key = {}
        self._rebuild_key_map_ui()
        self._refresh_video_tree()
        self._refresh_video_combo()
        self._update_dataset_lbl()
        self.playback_lbl.config(text="(no video)")
        self._render_record_view()

    def _update_dataset_lbl(self):
        if not self.videos:
            self.dataset_lbl.config(text="(no videos)"); return
        total = sum(len(v.frames_data) for v in self.videos)
        buttons = self._global_buttons()
        self.dataset_lbl.config(
            text=f"Total frames: {total}\n"
                 f"Buttons: {', '.join(buttons) if buttons else '(none)'}\n"
                 f"Action dim: {max(1, len(buttons))}")

    # ---------- Playback ----------
    def toggle_play(self):
        if self.game is None or not self.game.frames_data: return
        self._playback_running = not self._playback_running
        if self._playback_running: self._play_tick()

    def reset_play(self):
        if self.game is None: return
        self._playback_running = False
        self.game.reset()
        self._render_record_view()
        self.playback_lbl.config(text=self.game.status_text())

    def _play_tick(self):
        if not self._playback_running or self.game is None: return
        if not self.game.frames_data:
            self._playback_running = False; return
        self._show(self.game.render(), self.record_canvas, 'record_photo')
        self.playback_lbl.config(text=self.game.status_text())
        self.game.current_idx = (self.game.current_idx + 1) % len(self.game.frames_data)
        interval = 40
        try:
            if self.game.fps > 0:
                interval = max(15, int(1000 * self.game.stride / self.game.fps))
        except Exception:
            pass
        self.root.after(interval, self._play_tick)

    # ---------- VAE ----------
    def _vae_aug_dict(self):
        return {k: v.get() for k, v in self.vae_aug.items()}

    def _flow_aug_dict(self):
        return {k: v.get() for k, v in self.flow_aug.items()}

    def _all_frames_concat(self):
        out = []
        for v in self.videos: out.extend(v.frames_data)
        return out

    def _log(self, widget, msg):
        widget.insert(tk.END, f"{msg}\n")
        widget.see(tk.END)
        self.status.config(text=str(msg)[:100])

    def init_vae(self):
        try:
            in_ch = 3 if self.global_s['color_mode'].get() == 'rgb' else 1
            self.vae = FlexVAE(
                in_channels=in_ch,
                base_channels=self.vae_s['vae_base_channels'].get(),
                latent_channels=self.vae_s['vae_latent_channels'].get(),
                latent_h=self.vae_s['vae_latent_h'].get(),
                latent_w=self.vae_s['vae_latent_w'].get(),
                size=self.vae_s['vae_size'].get(),
                img_size=self.global_s['img_size'].get(),
            )
            self.vae_optimizer = optim.Adam(self.vae.parameters(),
                                            lr=self.vae_s['vae_lr'].get())
            n = sum(p.numel() for p in self.vae.parameters())
            self.vae_status.config(text=f"VAE ready ({n:,} params)")
            self._log(self.vae_log, f"VAE initialized: {n:,} params")
        except Exception as e:
            self.vae_status.config(text=f"Init failed: {e}")
            self._log(self.vae_log, f"Init error: {e}")

    def start_vae_train(self):
        if self.vae is None:
            messagebox.showerror("", "Init VAE first."); return
        frames = self._all_frames_concat()
        if len(frames) < 20:
            messagebox.showerror("", "Need at least 20 frames."); return
        self.vae_stop = False
        t = threading.Thread(target=self._vae_train_loop, args=(frames,), daemon=True)
        t.start()

    def stop_vae_train(self):
        self.vae_stop = True

    def _vae_train_loop(self, frames):
        try:
            img_sz = int(self.global_s['img_size'].get())
            cm = self.global_s['color_mode'].get()
            bs = int(self.vae_s['vae_batch_size'].get())
            nw = int(self.vae_s['vae_num_workers'].get())
            kl_w = float(self.vae_s['vae_kl_weight'].get())
            epochs = int(self.vae_s['vae_epochs'].get())

            ds = FrameOnlyDataset(frames, img_sz, cm, self._vae_aug_dict())
            dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=nw,
                            pin_memory=False, persistent_workers=(nw > 0))

            for ep in range(epochs):
                if self.vae_stop: break
                total, n = 0.0, 0
                for (imgs,) in dl:
                    if self.vae_stop: break
                    recon, mu, logvar = self.vae(imgs)
                    rl = F_nn.mse_loss(recon, imgs)
                    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / imgs.size(0)
                    loss = rl + kl_w * kl
                    self.vae_optimizer.zero_grad(); loss.backward(); self.vae_optimizer.step()
                    total += loss.item(); n += 1
                avg = total / max(1, n)
                self.vae_queue.put(('progress', ep + 1, epochs, avg))
                if (ep + 1) % 5 == 0:
                    self._vae_preview()
            self.vae_queue.put(('done', self.vae_stop))
        except Exception as e:
            import traceback; traceback.print_exc()
            self.vae_queue.put(('error', str(e)))

    def _vae_preview(self):
        try:
            frames = self._all_frames_concat()
            if not frames or self.vae is None: return
            img_sz = int(self.global_s['img_size'].get())
            cm = self.global_s['color_mode'].get()
            rng = np.random.RandomState(0)
            idx = rng.choice(len(frames), min(4, len(frames)), replace=False)
            imgs = []
            for i in idx:
                pil = Image.fromarray(frames[i])
                pil = pil.convert('RGB') if cm == 'rgb' else pil.convert('L')
                t = transforms.Compose([
                    transforms.Resize((img_sz, img_sz)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,)*(3 if cm == 'rgb' else 1),
                                         (0.5,)*(3 if cm == 'rgb' else 1)),
                ])(pil)
                imgs.append(t)
            batch = torch.stack(imgs)
            with torch.no_grad():
                recon, _, _ = self.vae(batch)
            b = ((batch + 1) / 2).clamp(0, 1).cpu().numpy()
            r = ((recon + 1) / 2).clamp(0, 1).cpu().numpy()
            thumb = 96
            grid = np.zeros((2 * thumb, len(idx) * thumb, 3), dtype=np.uint8)
            for i in range(len(idx)):
                for row, arr in enumerate([b[i], r[i]]):
                    a = np.stack([arr[0]]*3, -1) if arr.shape[0] == 1 else arr.transpose(1, 2, 0)
                    a = (a * 255).clip(0, 255).astype(np.uint8)
                    a = np.array(Image.fromarray(a).resize((thumb, thumb), Image.NEAREST))
                    grid[row*thumb:(row+1)*thumb, i*thumb:(i+1)*thumb] = a
            self.vae_queue.put(('preview', Image.fromarray(grid)))
        except Exception as e:
            self.vae_queue.put(('log', f"Preview error: {e}"))

    def _poll_vae_q(self):
        try:
            while True:
                msg = self.vae_queue.get_nowait()
                if msg[0] == 'progress':
                    _, ep, total, loss = msg
                    self.vae_prog['value'] = ep / total * 100
                    self.vae_status.config(text=f"Epoch {ep}/{total}  loss={loss:.5f}")
                elif msg[0] == 'preview':
                    img = msg[1]
                    photo = ImageTk.PhotoImage(img)
                    self._vae_preview_photo = photo
                    self.vae_canvas.delete("all")
                    self.vae_canvas.config(width=img.width, height=img.height)
                    self.vae_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
                elif msg[0] == 'done':
                    self.vae_prog['value'] = 100 if not msg[1] else 0
                    self.vae_status.config(text="Stopped" if msg[1] else "Complete!")
                elif msg[0] == 'error':
                    self.vae_status.config(text=f"Error: {msg[1]}")
                elif msg[0] == 'log':
                    self._log(self.vae_log, msg[1])
        except queue.Empty:
            pass
        self.root.after(100, self._poll_vae_q)

    def save_vae(self):
        if self.vae is None: return
        path = filedialog.asksaveasfilename(defaultextension=".pth",
                                            filetypes=[("PyTorch", "*.pth")])
        if not path: return
        torch.save({
            'model_state': self.vae.state_dict(),
            'optimizer_state': self.vae_optimizer.state_dict() if self.vae_optimizer else None,
            'vae_settings': {k: v.get() for k, v in self.vae_s.items()},
            'global_settings': {k: v.get() for k, v in self.global_s.items()},
        }, path)
        self._log(self.vae_log, f"Saved → {path}")

    def load_vae(self):
        path = filedialog.askopenfilename(filetypes=[("PyTorch", "*.pth")])
        if not path: return
        try:
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            for k, v in ckpt.get('vae_settings', {}).items():
                if k in self.vae_s: self.vae_s[k].set(v)
            for k, v in ckpt.get('global_settings', {}).items():
                if k in self.global_s: self.global_s[k].set(v)
            self.init_vae()
            self.vae.load_state_dict(ckpt['model_state'])
            if ckpt.get('optimizer_state') and self.vae_optimizer:
                self.vae_optimizer.load_state_dict(ckpt['optimizer_state'])
            self._log(self.vae_log, f"Loaded ← {path}")
        except Exception as e:
            self._log(self.vae_log, f"Load error: {e}")

    # ---------- Flow ----------
    def init_flow(self):
        if self.vae is None:
            messagebox.showerror("", "Init/load VAE first."); return
        try:
            cond_mode = self.flow_s['flow_cond_mode'].get()
            if cond_mode not in VALID_COND_MODES: cond_mode = 'shift'
            attn_levels = [self.flow_s['flow_attn_level_0'].get(),
                           self.flow_s['flow_attn_level_1'].get(),
                           self.flow_s['flow_attn_level_2'].get()]
            cond_dim = int(self.flow_s['flow_cond_dim'].get())

            self.flow = RectifiedFlowLatent(
                latent_channels=self.vae_s['vae_latent_channels'].get(),
                latent_h=self.vae_s['vae_latent_h'].get(),
                latent_w=self.vae_s['vae_latent_w'].get(),
                history=int(self.flow_s['flow_history'].get()),
                base_channels=int(self.flow_s['flow_base_channels'].get()),
                cond_dim=cond_dim,
                attention_levels=attn_levels,
                attn_num_heads=int(self.flow_s['flow_attn_num_heads'].get()),
                attn_dropout=float(self.flow_s['flow_attn_dropout'].get()),
                padding_mode=self.flow_s['flow_padding_mode'].get(),
                cond_mode=cond_mode,
                channel_mult=(1, 2, 4),
                device=self.device,
            )
            self.flow.use_ot = bool(self.flow_s['flow_use_ot'].get())
            self.flow.ot_method = self.flow_s['flow_ot_method'].get()
            self.flow.ot_sinkhorn_eps = float(self.flow_s['flow_ot_eps'].get())
            self.flow.ot_sinkhorn_iters = int(self.flow_s['flow_ot_iters'].get())
            for pg in self.flow.optimizer.param_groups:
                pg['lr'] = float(self.flow_s['flow_lr'].get())

            if self.flow_s['flow_ema_enabled'].get():
                self.flow.set_ema(decay=float(self.flow_s['flow_ema_decay'].get()),
                                  mode=self.flow_s['flow_ema_mode'].get())

            n_buttons = max(1, len(self._global_buttons()))
            self.button_encoder = ButtonEncoder(n_buttons, cond_dim).to(self.device)
            self.flow.optimizer = optim.Adam(
                list(self.flow.model.parameters()) + list(self.button_encoder.parameters()),
                lr=float(self.flow_s['flow_lr'].get()))

            n = sum(p.numel() for p in self.flow.model.parameters())
            self.flow_status.config(text=f"Flow ready: {n:,} params, cond_mode={cond_mode}")
            self._log(self.flow_log,
                      f"Flow initialized. cond_mode={cond_mode}, cond_dim={cond_dim}, "
                      f"n_buttons={n_buttons}, history={int(self.flow_s['flow_history'].get())}")
        except Exception as e:
            self.flow_status.config(text=f"Init failed: {e}")
            self._log(self.flow_log, f"Init error: {e}")

    def start_flow_train(self):
        if self.flow is None or self.vae is None or self.button_encoder is None:
            messagebox.showerror("", "Init VAE + Flow first."); return
        if not self.videos:
            messagebox.showerror("", "Load videos first."); return
        self.flow_stop = False
        t = threading.Thread(target=self._flow_train_loop, daemon=True)
        t.start()

    def stop_flow_train(self):
        self.flow_stop = True

    def _flow_train_loop(self):
        try:
            img_sz = int(self.global_s['img_size'].get())
            cm = self.global_s['color_mode'].get()
            history = int(self.flow_s['flow_history'].get())
            bs = int(self.flow_s['flow_batch_size'].get())
            epochs = int(self.flow_s['flow_epochs'].get())
            cfg_drop = float(self.flow_s['flow_cfg_dropout'].get())
            preview_on = bool(self.flow_s['flow_preview_enabled'].get())
            preview_freq = int(self.flow_s['flow_preview_freq'].get())
            global_buttons = self._global_buttons()

            ds = ButtonVideoDataset(self.videos, global_buttons, history,
                                    img_sz, cm, self._flow_aug_dict())
            if len(ds) == 0:
                self.flow_queue.put(('error', "No valid training windows")); return
            dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=0)

            vae = self.vae.eval()
            for ep in range(epochs):
                if self.flow_stop: break
                for pg in self.flow.optimizer.param_groups:
                    pg['lr'] = float(self.flow_s['flow_lr'].get())
                self.flow.use_ot = bool(self.flow_s['flow_use_ot'].get())
                self.flow.ot_sinkhorn_eps = float(self.flow_s['flow_ot_eps'].get())
                self.flow.ot_sinkhorn_iters = int(self.flow_s['flow_ot_iters'].get())
                if self.flow_s['flow_ema_enabled'].get() and not self.flow.use_ema:
                    self.flow.set_ema(decay=float(self.flow_s['flow_ema_decay'].get()),
                                      mode=self.flow_s['flow_ema_mode'].get())
                elif not self.flow_s['flow_ema_enabled'].get() and self.flow.use_ema:
                    self.flow.use_ema = False; self.flow.ema_model = None

                total, n = 0.0, 0
                for seqs, buttons in dl:
                    if self.flow_stop: break
                    B, S, C, H, W = seqs.shape
                    flat = seqs.reshape(B * S, C, H, W)
                    with torch.no_grad():
                        z = vae.encode(flat)
                    z = z.reshape(B, S, *z.shape[1:]).to(self.device)
                    context = z[:, :-1]
                    z1 = z[:, -1]
                    z0 = torch.randn_like(z1)
                    cond = self.button_encoder(buttons.to(self.device))
                    loss = self.flow.train_step(z0, z1, cond=cond, context=context,
                                                cfg_dropout_prob=cfg_drop)
                    total += loss; n += 1
                avg = total / max(1, n)
                self.flow_queue.put(('progress', ep + 1, epochs, avg))
                if preview_on and (ep + 1) % preview_freq == 0:
                    self._flow_preview_async()
            self.flow_queue.put(('done', self.flow_stop))
        except Exception as e:
            import traceback; traceback.print_exc()
            self.flow_queue.put(('error', str(e)))

    def _flow_preview_async(self):
        try:
            if not self.videos: return
            v = self.videos[0]
            if len(v.frames_data) < 2: return
            history = int(self.flow_s['flow_history'].get())
            img_sz = int(self.global_s['img_size'].get())
            cm = self.global_s['color_mode'].get()
            start = random.randint(0, len(v.frames_data) - 1)
            ctx_frames = []
            for k in range(history):
                fi = start - history + 1 + k
                if 0 <= fi < len(v.frames_data):
                    ctx_frames.append(v.frames_data[fi])
                else:
                    ctx_frames.append(np.zeros((img_sz, img_sz, 3), dtype=np.uint8))
            tf = transforms.Compose([
                transforms.Resize((img_sz, img_sz)), transforms.ToTensor(),
                transforms.Normalize((0.5,)*(3 if cm == 'rgb' else 1),
                                     (0.5,)*(3 if cm == 'rgb' else 1)),
            ])
            ctx_t = []
            for f in ctx_frames:
                pil = Image.fromarray(f)
                pil = pil.convert('RGB') if cm == 'rgb' else pil.convert('L')
                ctx_t.append(tf(pil))
            ctx_batch = torch.stack(ctx_t).unsqueeze(0).to(self.device)
            with torch.no_grad():
                ctx_z = self.vae.encode(ctx_batch.reshape(-1, *ctx_batch.shape[2:]))
                ctx_z = ctx_z.reshape(1, history, *ctx_z.shape[1:])
            n_b = max(1, len(self._global_buttons()))
            btns = np.zeros(n_b, dtype=np.float32)
            target_fi = start
            if 0 <= target_fi < len(v.actions_data):
                g_list = self._global_buttons()
                for local_i, name in enumerate(v.button_names):
                    if name in g_list:
                        g = g_list.index(name)
                        btns[g] = v.actions_data[target_fi][local_i]
            bt = torch.from_numpy(btns).unsqueeze(0).to(self.device)
            cond = self.button_encoder(bt)
            z1 = self.flow.sample_frame(
                n_samples=1, cond=cond, steps=20, method='euler',
                cfg_scale=1.0, use_ema=True, context=ctx_z)
            with torch.no_grad():
                decoded = self.vae.decode(z1)
            arr = ((decoded + 1) / 2).clamp(0, 1).cpu().numpy()[0]
            img = np.stack([arr[0]]*3, -1) if arr.shape[0] == 1 else arr.transpose(1, 2, 0)
            img = (img * 255).clip(0, 255).astype(np.uint8)
            self.flow_queue.put(('preview', Image.fromarray(img)))
        except Exception as e:
            self.flow_queue.put(('log', f"Preview error: {e}"))

    def _poll_flow_q(self):
        try:
            while True:
                msg = self.flow_queue.get_nowait()
                if msg[0] == 'progress':
                    _, ep, total, loss = msg
                    self.flow_prog['value'] = ep / total * 100
                    self.flow_status.config(text=f"Epoch {ep}/{total}  loss={loss:.5f}")
                elif msg[0] == 'preview':
                    img = msg[1].resize((256, 256), Image.NEAREST)
                    photo = ImageTk.PhotoImage(img)
                    self._flow_preview_photo = photo
                    self.flow_canvas.delete("all")
                    self.flow_canvas.config(width=256, height=256)
                    self.flow_canvas.create_image(128, 128, image=photo)
                elif msg[0] == 'done':
                    self.flow_prog['value'] = 100 if not msg[1] else 0
                    self.flow_status.config(text="Stopped" if msg[1] else "Complete!")
                elif msg[0] == 'error':
                    self.flow_status.config(text=f"Error: {msg[1]}")
                elif msg[0] == 'log':
                    self._log(self.flow_log, msg[1])
        except queue.Empty:
            pass
        self.root.after(100, self._poll_flow_q)

    def preview_flow(self):
        if self.flow is None or self.vae is None: return
        self._flow_preview_async()

    def save_flow(self):
        if self.flow is None: return
        path = filedialog.asksaveasfilename(defaultextension=".pth",
                                            filetypes=[("PyTorch", "*.pth")])
        if not path: return
        state = (self.flow.ema_model.state_dict()
                 if (self.flow.use_ema and self.flow.ema_model is not None)
                 else self.flow.model.state_dict())
        torch.save({
            'model_state': state,
            'optimizer_state': self.flow.optimizer.state_dict(),
            'button_encoder_state': self.button_encoder.state_dict(),
            'flow_settings': {k: v.get() for k, v in self.flow_s.items()},
            'vae_latent': {
                'latent_channels': self.vae_s['vae_latent_channels'].get(),
                'latent_h': self.vae_s['vae_latent_h'].get(),
                'latent_w': self.vae_s['vae_latent_w'].get(),
            },
            'history': int(self.flow_s['flow_history'].get()),
            'cond_mode': self.flow_s['flow_cond_mode'].get(),
            'attn_levels': self.flow.model.attention_levels,
            'attn_num_heads': self.flow.model.attn_num_heads,
            'attn_dropout': self.flow.model.attn_dropout,
            'n_buttons': max(1, len(self._global_buttons())),
            'global_buttons': self._global_buttons(),
        }, path)
        self._log(self.flow_log, f"Saved → {path}")

    def load_flow(self):
        path = filedialog.askopenfilename(filetypes=[("PyTorch", "*.pth")])
        if not path: return
        try:
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            for k, v in ckpt.get('flow_settings', {}).items():
                if k in self.flow_s:
                    try: self.flow_s[k].set(v)
                    except Exception: pass
            if 'history' in ckpt:
                self.flow_s['flow_history'].set(ckpt['history'])
            if 'cond_mode' in ckpt:
                self.flow_s['flow_cond_mode'].set(ckpt['cond_mode'])
            for k, v in ckpt.get('vae_latent', {}).items():
                mp = {'latent_channels': 'vae_latent_channels',
                      'latent_h': 'vae_latent_h', 'latent_w': 'vae_latent_w'}[k]
                self.vae_s[mp].set(v)
            if 'attn_levels' in ckpt:
                for i, f in enumerate(['flow_attn_level_0',
                                       'flow_attn_level_1',
                                       'flow_attn_level_2']):
                    if i < len(ckpt['attn_levels']):
                        self.flow_s[f].set(bool(ckpt['attn_levels'][i]))
            if 'attn_num_heads' in ckpt:
                self.flow_s['flow_attn_num_heads'].set(ckpt['attn_num_heads'])
            if 'attn_dropout' in ckpt:
                self.flow_s['flow_attn_dropout'].set(ckpt['attn_dropout'])
            self.init_flow()
            self.flow.model.load_state_dict(ckpt['model_state'])
            try: self.flow.optimizer.load_state_dict(ckpt['optimizer_state'])
            except Exception: pass
            if self.flow.use_ema and self.flow.ema_model is not None:
                self.flow.ema_model.load_state_dict(self.flow.model.state_dict())
            if 'button_encoder_state' in ckpt and self.button_encoder is not None:
                self.button_encoder.load_state_dict(ckpt['button_encoder_state'])
            self._log(self.flow_log, f"Loaded ← {path}")
        except Exception as e:
            self._log(self.flow_log, f"Load error: {e}")

    # ---------- Inference ----------
    def _read_buttons(self):
        g = self._global_buttons()
        b = np.zeros(max(1, len(g)), dtype=np.float32)
        for i, name in enumerate(g):
            key = self.button_to_key.get(name)
            if key and self.keys.get(key):
                b[i] = 1.0
        return b

    def _build_seed_context(self):
        if self.game is None: return None
        history = int(self.flow_s['flow_history'].get())
        start = int(self.inf_start_frame.get())
        n = len(self.game.frames_data)
        if n == 0: return None
        start = max(0, min(n - 1, start))
        cm = self.global_s['color_mode'].get()
        img_sz = int(self.global_s['img_size'].get())
        tf = transforms.Compose([
            transforms.Resize((img_sz, img_sz)), transforms.ToTensor(),
            transforms.Normalize((0.5,)*(3 if cm == 'rgb' else 1),
                                 (0.5,)*(3 if cm == 'rgb' else 1)),
        ])
        ctx = []
        for k in range(history):
            fi = start - history + 1 + k
            if 0 <= fi < n:
                pil = Image.fromarray(self.game.frames_data[fi])
                pil = pil.convert('RGB') if cm == 'rgb' else pil.convert('L')
                ctx.append(tf(pil))
            else:
                ctx.append(torch.zeros(3 if cm == 'rgb' else 1, img_sz, img_sz))
        ctx_batch = torch.stack(ctx).unsqueeze(0).to(self.device)
        with torch.no_grad():
            z = self.vae.encode(ctx_batch.reshape(-1, *ctx_batch.shape[2:]))
            z = z.reshape(1, history, *z.shape[1:])
        return z

    def start_inference(self):
        if self.vae is None or self.flow is None or self.button_encoder is None:
            messagebox.showerror("", "Load VAE + Flow first."); return
        name = self.inf_video_var.get()
        found = None
        for v in self.videos:
            if v.name == name: found = v; break
        if found is None:
            messagebox.showerror("", "Select a source video."); return
        self.game = found
        try:
            ctx = self._build_seed_context()
        except Exception as e:
            messagebox.showerror("", f"Context encode error: {e}"); return
        if ctx is None:
            messagebox.showerror("", "No context frames."); return

        self.inference_running = True
        self.inference_stop = False
        self.advance_one_frame = False
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.inf_status.config(text="Dreaming...")
        t = threading.Thread(target=self._inference_thread, args=(ctx,), daemon=True)
        t.start()

    def stop_inference(self):
        self.inference_stop = True
        self.btn_stop.config(state=tk.DISABLED)

    def reset_inference(self):
        self.inference_stop = True
        self.inference_running = False
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self._render_inference_blank()
        self.inf_status.config(text="Reset.")

    def _inference_thread(self, ctx):
        try:
            num = int(self.inf_num_frames.get())
            steps = int(self.inf_steps.get())
            method = self.inf_method.get()
            cfg = float(self.inf_cfg.get())
            prog = bool(self.inf_progressive.get())
            interval = max(1, int(self.inf_prog_interval.get()))
            frame_wait = bool(self.inf_frame_wait.get())
            fps = max(1, int(self.inf_fps.get()))

            history = int(self.flow_s['flow_history'].get())
            context = ctx

            for fi in range(num):
                if self.inference_stop: break

                # Frame-wait: block until at least one key is pressed or step_one advanced
                if frame_wait:
                    while (not any(self.keys.values())) and (not self.advance_one_frame) \
                            and (not self.inference_stop):
                        import time; time.sleep(0.02)
                    if self.inference_stop: break
                    self.advance_one_frame = False

                btns = self._read_buttons()
                bt = torch.from_numpy(btns).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    cond = self.button_encoder(bt)

                z = torch.randn(1, self.flow.latent_channels,
                                self.flow.latent_h, self.flow.latent_w, device=self.device)
                dt = 1.0 / steps
                times = torch.linspace(0, 1, steps + 1, device=self.device)
                ctx_flat = context.reshape(1, -1, self.flow.latent_h, self.flow.latent_w)
                model = (self.flow.ema_model
                         if (self.flow.use_ema and self.flow.ema_model is not None)
                         else self.flow.model)
                model.eval()
                null = torch.zeros_like(cond)
                stopped = False
                for si in range(steps):
                    if self.inference_stop:
                        stopped = True; break
                    t_val = times[si]
                    tt = torch.full((1,), t_val, device=self.device)
                    inp = torch.cat([z, ctx_flat], dim=1)
                    if cfg != 1.0:
                        vc = model(inp, tt, cond=cond)
                        vu = model(inp, tt, cond=null)
                        v = vu + cfg * (vc - vu)
                    else:
                        v = model(inp, tt, cond=cond)
                    z = z + v * dt
                    if prog and (si + 1) % interval == 0:
                        try:
                            with torch.no_grad():
                                dec = self.vae.decode(z)
                            arr = ((dec + 1) / 2).clamp(0, 1).cpu().numpy()[0]
                            img = np.stack([arr[0]]*3, -1) if arr.shape[0] == 1 \
                                else arr.transpose(1, 2, 0)
                            img = (img * 255).clip(0, 255).astype(np.uint8)
                            self.root.after(
                                0, lambda a=img, f=fi, s=si+1:
                                self._update_inference_preview(a, f, num, s, steps))
                        except Exception:
                            pass
                if stopped: break

                with torch.no_grad():
                    dec = self.vae.decode(z)
                arr = ((dec + 1) / 2).clamp(0, 1).cpu().numpy()[0]
                img = np.stack([arr[0]]*3, -1) if arr.shape[0] == 1 \
                    else arr.transpose(1, 2, 0)
                img = (img * 255).clip(0, 255).astype(np.uint8)
                self.root.after(0, lambda a=img, f=fi:
                                self._update_inference_preview(a, f, num, steps, steps))

                context = torch.cat([context[:, 1:], z.unsqueeze(1)], dim=1)

                # Pace playback if not frame_wait
                if not frame_wait:
                    import time; time.sleep(max(0.0, 1.0 / fps))

            self.root.after(0, lambda: self.inf_status.config(
                text="Stopped" if self.inference_stop else "Complete."))
        except Exception as e:
            import traceback; traceback.print_exc()
            self.root.after(0, lambda: self.inf_status.config(text=f"Error: {e}"))
        finally:
            self.inference_running = False
            self.root.after(0, lambda: self.btn_start.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.btn_stop.config(state=tk.DISABLED))

    def _update_inference_preview(self, arr, frame_i, total_frames, step_i, total_steps):
        img = Image.fromarray(arr).resize((512, 512), Image.NEAREST)
        photo = ImageTk.PhotoImage(img)
        self.inference_photo = photo
        self.inference_canvas.delete("all")
        self.inference_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.inf_status.config(
            text=f"Frame {frame_i + 1}/{total_frames}  step {step_i}/{total_steps}")


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()