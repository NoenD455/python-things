import os
import sys
import random
import threading
import queue
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ============================================================================
# VAE  (inlined from LDM_flow_retry.py, with dilation support)
# ============================================================================

VAE_SIZE_CONFIGS = {
    'tiny':   (2, [0.25, 0.5]),
    'small':  (3, [0.25, 0.5, 1.0]),
    'medium': (3, [0.5, 1.0, 2.0]),
    'big':    (3, [1.0, 2.0, 4.0]),
    'large':  (3, [2.0, 4.0, 8.0]),
}

DILATION_PRESETS = {
    'none':   dict(pattern='',        init=1, out=1, refine=False, ref_dil=1),
    'mild':   dict(pattern='1,1,2',   init=2, out=1, refine=False, ref_dil=1),
    'medium': dict(pattern='1,2,4',   init=4, out=2, refine=False, ref_dil=1),
    'strong': dict(pattern='1,3,6',   init=6, out=3, refine=False, ref_dil=1),
    'wide':   dict(pattern='1,2,4,8', init=8, out=4, refine=True,  ref_dil=2),
}


def _vae_num_blocks(img_size, latent_h, latent_w):
    if img_size % latent_h or img_size % latent_w:
        raise ValueError(
            f"img_size={img_size} must be divisible by latent {latent_h}x{latent_w}")
    rh, rw = img_size // latent_h, img_size // latent_w
    if rh != rw:
        raise ValueError(f"non-square downsampling ratio {rh}:{rw} not supported")
    n = int(round(math.log2(rh)))
    if 2 ** n != rh:
        raise ValueError(f"ratio {rh} is not a power of 2")
    if n < 1:
        raise ValueError(f"need at least 1 downsampling block, got ratio {rh}")
    return n


def _fit_mults(mults, n):
    if len(mults) == n:
        return list(mults)
    if len(mults) > n:
        return list(mults[:n])
    ext = list(mults)
    while len(ext) < n:
        ext.append(ext[-1] * 2)
    return ext


def _extend_dilation(pattern, n):
    """Extend/truncate a dilation pattern to exactly n entries.

    Patterns are extended by doubling the last entry (so deeper levels get
    larger dilations by default). If pattern is None or empty, all-ones.
    """
    if not pattern:
        return [1] * n
    vals = [max(1, int(v)) for v in pattern]
    if len(vals) >= n:
        return vals[:n]
    while len(vals) < n:
        vals.append(vals[-1] * 2)
    return vals


# ---------- Encoder ----------

class FlexEncoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=32, latent_channels=8,
                 latent_h=4, latent_w=4, size='big', img_size=32,
                 dilation_pattern=None):
        super().__init__()
        _, mults = VAE_SIZE_CONFIGS[size]
        n = _vae_num_blocks(img_size, latent_h, latent_w)
        mults = _fit_mults(mults, n)
        self.dilations = _extend_dilation(dilation_pattern, n)
        self.latent_h, self.latent_w = latent_h, latent_w

        self.down = nn.ModuleList()
        c = in_channels
        for i, m in enumerate(mults):
            oc = max(1, int(base_channels * m))
            d = self.dilations[i]
            # stride=2 with padding=dilation keeps the halving exact
            self.down.append(nn.Conv2d(c, oc, 3, stride=2, padding=d, dilation=d))
            self.down.append(nn.LeakyReLU(0.2))
            c = oc
        self.conv_mu = nn.Conv2d(c, latent_channels, 3, padding=1)
        self.conv_logvar = nn.Conv2d(c, latent_channels, 3, padding=1)

    def forward(self, x):
        for l in self.down:
            x = l(x)
        return self.conv_mu(x), self.conv_logvar(x)


# ---------- Decoder ----------

class FlexDecoder(nn.Module):
    def __init__(self, latent_channels=8, base_channels=32, out_channels=3,
                 latent_h=4, latent_w=4, size='big', img_size=32,
                 dilation_pattern=None, dilate_init=1, dilate_out=1,
                 add_refinement=False, refinement_dilation=1):
        super().__init__()
        _, mults = VAE_SIZE_CONFIGS[size]
        n = _vae_num_blocks(img_size, latent_h, latent_w)
        mults = _fit_mults(mults, n)
        widths = [max(1, int(base_channels * m)) for m in mults]

        enc_dil = _extend_dilation(dilation_pattern, n)
        # Reverse: decoder goes low-res → high-res, so lowest-res (first ops)
        # should get the widest dilation.
        up_dil = list(reversed(enc_dil))
        self.dilations = up_dil
        self.latent_h, self.latent_w = latent_h, latent_w

        high = widths[-1]
        self.init_conv = nn.Conv2d(
            latent_channels, high, 3,
            padding=max(1, int(dilate_init)),
            dilation=max(1, int(dilate_init)))
        self.act = nn.LeakyReLU(0.2)
        self.up = nn.ModuleList()
        c = high
        for i in range(n):
            oc = widths[n - 2 - i] if i < n - 1 else widths[0]
            self.up.append(nn.ConvTranspose2d(c, oc, 4, stride=2, padding=1))
            self.up.append(nn.LeakyReLU(0.2))
            if add_refinement:
                # Refinement conv is at the *upsampled* resolution, so use a
                # smaller dilation than the low-res stages.
                rd = max(1, int(refinement_dilation))
                self.up.append(nn.Conv2d(oc, oc, 3, padding=rd, dilation=rd))
                self.up.append(nn.LeakyReLU(0.2))
            c = oc
        self.conv_out = nn.Conv2d(
            c, out_channels, 3,
            padding=max(1, int(dilate_out)),
            dilation=max(1, int(dilate_out)))

    def forward(self, z, target_size=None):
        x = self.act(self.init_conv(z))
        for l in self.up:
            x = l(x)
        out = torch.tanh(self.conv_out(x))
        if target_size and (out.shape[2] != target_size[0] or out.shape[3] != target_size[1]):
            out = F.interpolate(out, size=target_size, mode='bilinear',
                                align_corners=False)
        return out


# ---------- VAE wrapper ----------

class FlexVAE(nn.Module):
    def __init__(self, in_channels=3, base_channels=32, latent_channels=8,
                 latent_h=4, latent_w=4, size='big', img_size=32,
                 dilation_pattern=None, dilate_init=1, dilate_out=1,
                 add_refinement=False, refinement_dilation=1):
        super().__init__()
        self.encoder = FlexEncoder(
            in_channels, base_channels, latent_channels,
            latent_h, latent_w, size, img_size,
            dilation_pattern=dilation_pattern)
        self.decoder = FlexDecoder(
            latent_channels, base_channels, in_channels,
            latent_h, latent_w, size, img_size,
            dilation_pattern=dilation_pattern,
            dilate_init=dilate_init, dilate_out=dilate_out,
            add_refinement=add_refinement,
            refinement_dilation=refinement_dilation)
        self.latent_channels = latent_channels
        self.latent_h, self.latent_w = latent_h, latent_w
        self.size = size
        self.img_size = img_size

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z, (x.shape[2], x.shape[3])), mu, logvar

    def encode(self, x):
        return self.encoder(x)[0]

    def decode(self, z, target_size=None):
        return self.decoder(z, target_size)


# ============================================================================
# Masked-latent wrapper
# ============================================================================

class MaskedLatentVAE(nn.Module):
    def __init__(self, **vae_kwargs):
        super().__init__()
        self.vae = FlexVAE(**vae_kwargs)
        self.latent_h = vae_kwargs['latent_h']
        self.latent_w = vae_kwargs['latent_w']
        self.latent_channels = vae_kwargs['latent_channels']
        self.img_size = vae_kwargs['img_size']

    def forward(self, x, mask=None, mask_ratio=0.5, sample_latent=False):
        mu, logvar = self.vae.encoder(x)
        if sample_latent:
            std = torch.exp(0.5 * logvar)
            z = mu + torch.randn_like(std) * std
        else:
            z = mu
        if mask is None and mask_ratio > 0:
            mask = (torch.rand(z.shape[0], 1, self.latent_h, self.latent_w,
                              device=z.device) < mask_ratio).float()
        z_in = z * (1.0 - mask) if mask is not None else z
        recon = self.vae.decoder(z_in, (x.shape[2], x.shape[3]))
        return recon, z, mu, logvar, mask

    def encode(self, x):
        return self.vae.encoder(x)[0]

    def decode(self, z, target_size=None):
        return self.vae.decoder(z, target_size)


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def get_loss(name):
    if name.startswith('MSE'):
        return F.mse_loss
    if name.startswith('MAE'):
        return F.l1_loss
    return F.smooth_l1_loss


def make_mask(kind, ratio, latent_h, latent_w, device):
    m = torch.zeros(1, 1, latent_h, latent_w, device=device)
    if kind == 'random':
        m = (torch.rand(1, 1, latent_h, latent_w, device=device) < ratio).float()
    elif kind == 'half':
        m[:, :, :, :latent_w // 2] = 1.0
    elif kind == 'middle':
        qh, qw = max(1, latent_h // 4), max(1, latent_w // 4)
        m[:, :, qh:latent_h - qh, qw:latent_w - qw] = 1.0
    return m


def tensor_to_pil(t):
    t = t.detach().cpu().clamp(-1, 1)
    if t.dim() == 4:
        t = t[0]
    t = (t + 1) / 2
    if t.shape[0] == 1:
        arr = (t[0].numpy() * 255).astype(np.uint8)
        arr = np.stack([arr, arr, arr], axis=-1)
    else:
        arr = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


# ============================================================================
# Dataset
# ============================================================================

IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tif', '.tiff')


class FolderImageDataset(Dataset):
    def __init__(self, folder, img_size=64, color_mode='rgb'):
        self.files = []
        for root, _, files in os.walk(folder):
            for f in files:
                if f.lower().endswith(IMG_EXTS):
                    self.files.append(os.path.join(root, f))
        self.files.sort()
        self.img_size = img_size
        self.color_mode = color_mode.lower()
        if self.color_mode == 'rgb':
            norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        else:
            norm = transforms.Normalize((0.5,), (0.5,))
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size), Image.BICUBIC),
            transforms.ToTensor(),
            norm,
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = Image.open(self.files[i])
        img = img.convert('RGB' if self.color_mode == 'rgb' else 'L')
        return self.transform(img)


# ============================================================================
# App
# ============================================================================

class App:
    def __init__(self, root):
        self.root = root
        root.title('Masked Latent VAE (MAE-style, dilated)')
        root.geometry('1150x860')
        root.minsize(1000, 740)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = None
        self.dataset = None
        self.preview_batch = None
        self.train_thread = None
        self.stop_event = threading.Event()
        self.msg_queue = queue.Queue()

        self.vae_settings = {
            'size':            tk.StringVar(value='large'),
            'base_channels':   tk.IntVar(value=48),
            'latent_channels': tk.IntVar(value=4),
            'latent_h':        tk.IntVar(value=16),
            'latent_w':        tk.IntVar(value=16),
            'img_size':        tk.IntVar(value=64),
            'color_mode':      tk.StringVar(value='rgb'),
            # Dilation
            'dilation_preset':      tk.StringVar(value='medium'),
            'dilation_pattern':     tk.StringVar(value='1,2,4'),
            'dilate_init':          tk.IntVar(value=4),
            'dilate_out':           tk.IntVar(value=2),
            'add_refinement':       tk.BooleanVar(value=False),
            'refinement_dilation':  tk.IntVar(value=2),
        }
        self.train_settings = {
            'batch_size':     tk.IntVar(value=32),
            'lr':             tk.DoubleVar(value=1e-3),
            'loss':           tk.StringVar(value='MSE (L2)'),
            'mask_ratio':     tk.DoubleVar(value=0.5),
            'kl_weight':      tk.DoubleVar(value=1e-4),
            'sample_latent':  tk.BooleanVar(value=False),
            'preview_every':  tk.IntVar(value=5),
        }

        self.img_size = 64
        self.color_mode = 'rgb'
        self.latent_h = 16
        self.latent_w = 16
        self.latent_channels = 4

        # Inference canvas state
        self._inference_tensor = None
        self._inference_pil = None
        self.infer_mask = np.zeros((16, 16), dtype=np.uint8)
        self._canvas_photo = None
        self._canvas_size = 320
        self._paint_value = None
        self._last_cell = None
        self._brush_size = tk.IntVar(value=1)

        self._build_ui()

        self.status = tk.StringVar(
            value=f'Device: {self.device} | Model: not initialized')
        ttk.Label(root, textvariable=self.status, anchor='w',
                  relief='sunken').pack(fill='x')

        root.after(100, self._poll_queue)

    # ---------------------------------------------------------------- UI
    def _build_ui(self):
        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill='both', expand=True, padx=6, pady=6)
        self._build_dataset_tab()
        self._build_settings_tab()
        self._build_training_tab()
        self._build_inference_tab()

    def _build_dataset_tab(self):
        f = ttk.Frame(self.nb); self.nb.add(f, text='Dataset')
        row = ttk.Frame(f); row.pack(fill='x', padx=8, pady=8)
        ttk.Label(row, text='Image folder:').pack(side='left')
        self.dataset_path = tk.StringVar()
        ttk.Entry(row, textvariable=self.dataset_path).pack(
            side='left', fill='x', expand=True, padx=4)
        ttk.Button(row, text='Browse...', command=self._browse_dataset).pack(side='left')
        ttk.Button(row, text='Load', command=self._load_dataset).pack(side='left', padx=4)

        self.dataset_info = tk.StringVar(value='No dataset loaded.')
        ttk.Label(f, textvariable=self.dataset_info).pack(anchor='w', padx=8)

        self.dataset_preview_label = ttk.Label(f)
        self.dataset_preview_label.pack(padx=8, pady=8, anchor='w')

        hint = ("Images are resized to img_size×img_size and normalized to [-1,1] "
                "(FlexVAE uses tanh outputs). Color mode is taken from Settings.")
        ttk.Label(f, text=hint, foreground='#555').pack(anchor='w', padx=8, pady=4)

    def _build_settings_tab(self):
        outer = ttk.Frame(self.nb); self.nb.add(outer, text='Settings')

        canvas = tk.Canvas(outer)
        sb = ttk.Scrollbar(outer, orient='vertical', command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.create_window((0, 0), window=inner, anchor='nw')
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side='left', fill='both', expand=True)
        sb.pack(side='right', fill='y')

        wrap = ttk.Frame(inner); wrap.pack(padx=20, pady=16, anchor='nw')

        # --- VAE architecture ---
        vf = ttk.LabelFrame(wrap, text='FlexVAE Architecture', padding=10)
        vf.pack(fill='x', pady=6)
        self._combo(vf, 'Size preset:', self.vae_settings['size'],
                    ['tiny', 'small', 'medium', 'big', 'large'])
        self._spin(vf, 'Base channels:', self.vae_settings['base_channels'], 4, 256)
        self._spin(vf, 'Latent channels:', self.vae_settings['latent_channels'], 1, 64)
        self._spin(vf, 'Latent height:', self.vae_settings['latent_h'], 1, 64)
        self._spin(vf, 'Latent width:', self.vae_settings['latent_w'], 1, 64)
        self._spin(vf, 'Image size:', self.vae_settings['img_size'], 16, 256)
        self._combo(vf, 'Color mode:', self.vae_settings['color_mode'],
                    ['rgb', 'grayscale'])

        # --- Dilation ---
        df = ttk.LabelFrame(wrap, text='Dilated Convolutions', padding=10)
        df.pack(fill='x', pady=6)
        self._combo(df, 'Dilation preset:', self.vae_settings['dilation_preset'],
                    ['none', 'mild', 'medium', 'strong', 'wide', 'custom'])
        self.vae_settings['dilation_preset'].trace_add(
            'write', lambda *_: self._apply_dilation_preset(quiet=True))
        ttk.Button(df, text='Apply preset', command=self._apply_dilation_preset
                   ).pack(anchor='w', pady=(4, 6))

        self._entry(df, 'Encoder pattern:', self.vae_settings['dilation_pattern'])
        ttk.Label(df, text='(comma-separated dilations per stride-2 block, e.g. 1,2,4)',
                  foreground='#555').pack(anchor='w', padx=(24, 0))
        self._spin(df, 'Decoder init dilation:', self.vae_settings['dilate_init'], 1, 16)
        self._spin(df, 'Decoder out dilation:', self.vae_settings['dilate_out'], 1, 16)
        ttk.Checkbutton(df, text='Post-upsample refinement conv',
                        variable=self.vae_settings['add_refinement']).pack(
                            anchor='w', pady=(4, 0))
        self._spin(df, 'Refinement dilation:', self.vae_settings['refinement_dilation'], 1, 16)

        ttk.Label(df,
                  text='Effective kernel = 2·dilation + 1. Decoder init at latent '
                       'resolution is the biggest lever for MAE.',
                  foreground='#555', wraplength=420).pack(anchor='w', pady=(6, 0))

        # --- Training ---
        tf = ttk.LabelFrame(wrap, text='Training', padding=10)
        tf.pack(fill='x', pady=6)
        self._spin(tf, 'Batch size:', self.train_settings['batch_size'], 1, 512)
        self._entry(tf, 'Learning rate:', self.train_settings['lr'])
        self._combo(tf, 'Loss function:', self.train_settings['loss'],
                    ['MSE (L2)', 'MAE (L1)', 'SmoothL1'])
        self._entry(tf, 'Mask ratio:', self.train_settings['mask_ratio'])
        self._entry(tf, 'KL weight:', self.train_settings['kl_weight'])
        ttk.Checkbutton(tf, text='Sample latent (reparameterize) — otherwise use μ',
                        variable=self.train_settings['sample_latent']).pack(
                            anchor='w', pady=2)
        self._spin(tf, 'Preview every N epochs:', self.train_settings['preview_every'], 1, 500)

        ttk.Button(wrap, text='Apply Settings',
                   command=self._apply_settings).pack(pady=10, anchor='w')

    def _build_training_tab(self):
        f = ttk.Frame(self.nb); self.nb.add(f, text='Training')
        ctrl = ttk.Frame(f); ctrl.pack(fill='x', padx=8, pady=8)
        ttk.Button(ctrl, text='Init Model', command=self._init_model).pack(side='left')
        ttk.Label(ctrl, text='  Epochs:').pack(side='left')
        self.var_epochs = tk.IntVar(value=20)
        ttk.Spinbox(ctrl, from_=1, to=100000, textvariable=self.var_epochs,
                    width=8).pack(side='left')
        ttk.Button(ctrl, text='Start Training', command=self._start_training
                   ).pack(side='left', padx=6)
        ttk.Button(ctrl, text='Stop', command=self._stop_training).pack(side='left')

        self.train_status = tk.StringVar(value='Idle.')
        ttk.Label(f, textvariable=self.train_status).pack(anchor='w', padx=8)

        self.preview_label = ttk.Label(f)
        self.preview_label.pack(padx=8, pady=8)

    def _build_inference_tab(self):
        f = ttk.Frame(self.nb); self.nb.add(f, text='Inference / Draw Mask')
        top = ttk.Frame(f); top.pack(fill='x', padx=8, pady=8)
        ttk.Button(top, text='Load Image...',
                   command=self._load_infer_image).pack(side='left')
        ttk.Button(top, text='Clear Mask',
                   command=self._clear_mask).pack(side='left', padx=4)
        ttk.Button(top, text='Random',
                   command=lambda: self._preset_mask('random')).pack(side='left', padx=4)
        ttk.Button(top, text='Half',
                   command=lambda: self._preset_mask('half')).pack(side='left', padx=4)
        ttk.Button(top, text='Middle',
                   command=lambda: self._preset_mask('middle')).pack(side='left', padx=4)
        ttk.Button(top, text='Encode / Decode',
                   command=self._run_inference).pack(side='left', padx=10)

        brush_row = ttk.Frame(f); brush_row.pack(fill='x', padx=8, pady=(0, 4))
        ttk.Label(brush_row, text='Brush size (cells):').pack(side='left')
        ttk.Spinbox(brush_row, from_=1, to=8, textvariable=self._brush_size,
                    width=5).pack(side='left', padx=4)
        ttk.Label(brush_row, text='(left-drag paint · right-drag erase)',
                  foreground='#555').pack(side='left', padx=8)

        body = ttk.Frame(f); body.pack(fill='both', expand=True, padx=8, pady=8)
        left = ttk.Frame(body); left.pack(side='left', anchor='n')
        self.infer_canvas_hint = tk.StringVar(
            value='Left-drag to paint mask · Right-drag to erase')
        ttk.Label(left, textvariable=self.infer_canvas_hint).pack()
        self.infer_canvas = tk.Canvas(left, width=self._canvas_size,
                                      height=self._canvas_size, bg='gray30')
        self.infer_canvas.pack()
        self.infer_canvas.bind('<Button-1>', self._on_paint_start)
        self.infer_canvas.bind('<B1-Motion>', self._on_paint_drag)
        self.infer_canvas.bind('<ButtonRelease-1>', self._on_paint_end)
        self.infer_canvas.bind('<Button-3>', self._on_erase_start)
        self.infer_canvas.bind('<B3-Motion>', self._on_erase_drag)
        self.infer_canvas.bind('<ButtonRelease-3>', self._on_paint_end)
        self.infer_canvas.bind('<Leave>', self._on_canvas_leave)

        right = ttk.Frame(body); right.pack(side='left', padx=14, anchor='n')
        self.infer_output_label = ttk.Label(right)
        self.infer_output_label.pack(anchor='nw')
        self._redraw_canvas()

    # ---- widget helpers ----
    def _spin(self, parent, label, var, lo, hi):
        row = ttk.Frame(parent); row.pack(fill='x', pady=1)
        ttk.Label(row, text=label, width=24, anchor='w').pack(side='left')
        ttk.Spinbox(row, from_=lo, to=hi, textvariable=var,
                    width=10).pack(side='right')

    def _entry(self, parent, label, var):
        row = ttk.Frame(parent); row.pack(fill='x', pady=1)
        ttk.Label(row, text=label, width=24, anchor='w').pack(side='left')
        ttk.Entry(row, textvariable=var, width=20).pack(side='right')

    def _combo(self, parent, label, var, values):
        row = ttk.Frame(parent); row.pack(fill='x', pady=1)
        ttk.Label(row, text=label, width=24, anchor='w').pack(side='left')
        ttk.Combobox(row, textvariable=var, values=values, state='readonly',
                     width=12).pack(side='right')

    # ---- dilation preset ----
    def _apply_dilation_preset(self, quiet=False):
        preset = self.vae_settings['dilation_preset'].get()
        if preset == 'custom' or preset not in DILATION_PRESETS:
            return
        p = DILATION_PRESETS[preset]
        # Update the fields (this fires the trace again, but the guard above
        # prevents infinite recursion since we only update on 'custom' return).
        self.vae_settings['dilation_pattern'].set(p['pattern'])
        self.vae_settings['dilate_init'].set(p['init'])
        self.vae_settings['dilate_out'].set(p['out'])
        self.vae_settings['add_refinement'].set(p['refine'])
        self.vae_settings['refinement_dilation'].set(p['ref_dil'])
        if not quiet:
            self.status.set(f'Applied dilation preset: {preset}')

    # ---------------------------------------------------------------- dataset
    def _browse_dataset(self):
        p = filedialog.askdirectory()
        if p:
            self.dataset_path.set(p)

    def _load_dataset(self):
        p = self.dataset_path.get().strip()
        if not p or not os.path.isdir(p):
            messagebox.showerror('Error', 'Invalid folder'); return
        try:
            self._apply_settings(quiet=True)
            ds = FolderImageDataset(p, self.img_size, self.color_mode)
        except Exception as e:
            messagebox.showerror('Error', f'Failed to load dataset: {e}'); return
        if len(ds) == 0:
            messagebox.showerror('Error', 'No images found'); return
        self.dataset = ds
        self.dataset_info.set(f'Loaded {len(ds)} images from {p} '
                              f'(img_size={self.img_size}, mode={self.color_mode})')

        n = min(8, len(ds))
        idxs = random.sample(range(len(ds)), n)
        strip = Image.new('RGB', (64 * n, 64), 'white')
        for i, idx in enumerate(idxs):
            strip.paste(tensor_to_pil(ds[idx]).resize((64, 64), Image.NEAREST),
                        (i * 64, 0))
        photo = ImageTk.PhotoImage(strip)
        self.dataset_preview_label.configure(image=photo)
        self.dataset_preview_label.image = photo

        k = min(4, len(ds))
        self.preview_batch = torch.stack(
            [ds[i] for i in random.sample(range(len(ds)), k)]).to(self.device)

    # --------------------------------------------------------------- settings
    def _apply_settings(self, quiet=False):
        try:
            self.img_size = int(self.vae_settings['img_size'].get())
            self.color_mode = self.vae_settings['color_mode'].get()
            self.latent_h = int(self.vae_settings['latent_h'].get())
            self.latent_w = int(self.vae_settings['latent_w'].get())
            self.latent_channels = int(self.vae_settings['latent_channels'].get())

            _vae_num_blocks(self.img_size, self.latent_h, self.latent_w)
            if self.vae_settings['size'].get() not in VAE_SIZE_CONFIGS:
                raise ValueError('unknown size preset')
        except Exception as e:
            if not quiet:
                messagebox.showerror('Settings error', str(e))
            return False

        if self.infer_mask.shape != (self.latent_h, self.latent_w):
            self.infer_mask = np.zeros((self.latent_h, self.latent_w), dtype=np.uint8)
        self._redraw_canvas()

        if not quiet:
            self.status.set(
                f'Device: {self.device} | '
                f'VAE size={self.vae_settings["size"].get()} '
                f'base={self.vae_settings["base_channels"].get()} '
                f'latent={self.latent_channels}x{self.latent_h}x{self.latent_w} '
                f'img={self.img_size}')
        return True

    # ------------------------------------------------------------------ model
    def _read_dilation_config(self):
        """Returns a dict of dilation kwargs to pass to FlexVAE/MaskedLatentVAE."""
        pattern_str = self.vae_settings['dilation_pattern'].get().strip()
        pattern = None
        if pattern_str:
            try:
                pattern = [max(1, int(x.strip()))
                           for x in pattern_str.split(',') if x.strip()]
            except ValueError:
                pattern = None
        return dict(
            dilation_pattern=pattern,
            dilate_init=max(1, int(self.vae_settings['dilate_init'].get())),
            dilate_out=max(1, int(self.vae_settings['dilate_out'].get())),
            add_refinement=bool(self.vae_settings['add_refinement'].get()),
            refinement_dilation=max(1, int(self.vae_settings['refinement_dilation'].get())),
        )

    def _init_model(self):
        if self.train_thread is not None and self.train_thread.is_alive():
            messagebox.showinfo('Info', 'Stop training first.'); return
        if not self._apply_settings():
            return
        try:
            in_ch = 3 if self.color_mode == 'rgb' else 1
            dil_kwargs = self._read_dilation_config()
            self.model = MaskedLatentVAE(
                in_channels=in_ch,
                base_channels=int(self.vae_settings['base_channels'].get()),
                latent_channels=self.latent_channels,
                latent_h=self.latent_h,
                latent_w=self.latent_w,
                size=self.vae_settings['size'].get(),
                img_size=self.img_size,
                **dil_kwargs,
            ).to(self.device)
        except Exception as e:
            messagebox.showerror('Init error', str(e)); return

        n_params = count_params(self.model)
        enc_dil = self.model.vae.encoder.dilations
        dec_dil = self.model.vae.decoder.dilations
        self.status.set(
            f'Device: {self.device} | Params: {n_params:,} | '
            f'latent {self.latent_channels}×{self.latent_h}×{self.latent_w} | '
            f'img {self.img_size} ({self.color_mode}) | '
            f'enc dil {enc_dil}, dec dil {dec_dil}')
        self.train_status.set(
            f'Model initialized. Params: {n_params:,}. '
            f'Encoder dilations={enc_dil}, decoder dilations={dec_dil}.')

        if self.preview_batch is not None:
            try:
                ratio = float(self.train_settings['mask_ratio'].get())
            except Exception:
                ratio = 0.5
            pil = self._compose_preview(ratio)
            if pil is not None:
                photo = ImageTk.PhotoImage(pil)
                self.preview_label.configure(image=photo)
                self.preview_label.image = photo

    def _start_training(self):
        if self.dataset is None:
            messagebox.showerror('Error', 'Load a dataset first.'); return
        if self.model is None:
            messagebox.showerror('Error', 'Init model first.'); return
        if self.train_thread is not None and self.train_thread.is_alive():
            messagebox.showinfo('Info', 'Training already running.'); return
        if not self._apply_settings():
            return

        try:
            epochs = int(self.var_epochs.get())
            bs = int(self.train_settings['batch_size'].get())
            lr = float(self.train_settings['lr'].get())
            loss = self.train_settings['loss'].get()
            mr = float(self.train_settings['mask_ratio'].get())
            kl = float(self.train_settings['kl_weight'].get())
            samp = bool(self.train_settings['sample_latent'].get())
            prev = int(self.train_settings['preview_every'].get())
        except Exception as e:
            messagebox.showerror('Error', f'Invalid setting: {e}'); return

        if self.preview_batch is None:
            k = min(4, len(self.dataset))
            self.preview_batch = torch.stack(
                [self.dataset[i] for i in random.sample(range(len(self.dataset)), k)]
            ).to(self.device)

        self.stop_event.clear()
        self.train_thread = threading.Thread(
            target=self._train_worker,
            args=(epochs, bs, lr, loss, mr, kl, samp, prev),
            daemon=True)
        self.train_thread.start()
        self.train_status.set('Training...')

    def _stop_training(self):
        self.stop_event.set()
        self.train_status.set('Stopping...')

    def _train_worker(self, epochs, bs, lr, loss_name, mask_ratio,
                      kl_weight, sample_latent, preview_every):
        try:
            loader = DataLoader(self.dataset, batch_size=bs, shuffle=True,
                                num_workers=0)
            self.model.train()
            opt = optim.Adam(self.model.parameters(), lr=lr)
            loss_fn = get_loss(loss_name)

            for epoch in range(epochs):
                if self.stop_event.is_set():
                    break
                total, n = 0.0, 0
                for x in loader:
                    if self.stop_event.is_set():
                        break
                    x = x.to(self.device)
                    recon, z, mu, logvar, m = self.model(
                        x, mask_ratio=mask_ratio, sample_latent=sample_latent)
                    rl = loss_fn(recon, x)
                    kl = -0.5 * torch.sum(
                        1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
                    loss = rl + kl_weight * kl
                    opt.zero_grad(); loss.backward(); opt.step()
                    total += loss.item() * x.size(0); n += x.size(0)
                avg = total / max(n, 1)
                self.msg_queue.put(('epoch', (epoch + 1, avg, epochs, kl_weight)))
                if (epoch + 1) % preview_every == 0 or epoch == epochs - 1:
                    self.model.eval()
                    with torch.no_grad():
                        pil = self._compose_preview(mask_ratio)
                    self.model.train()
                    if pil is not None:
                        self.msg_queue.put(('preview', pil))
            self.msg_queue.put(('done', None))
        except Exception as e:
            import traceback; traceback.print_exc()
            self.msg_queue.put(('error', str(e)))

    def _compose_preview(self, mask_ratio):
        if self.preview_batch is None or self.model is None:
            return None
        imgs = self.preview_batch[: min(4, self.preview_batch.shape[0])]
        try:
            z = self.model.encode(imgs)
        except Exception:
            return None
        rows = []
        for i in range(imgs.shape[0]):
            img = imgs[i:i + 1]
            zi = z[i:i + 1]
            row = [tensor_to_pil(img[0])]
            for kind in ('random', 'half', 'middle'):
                m = make_mask(kind, mask_ratio, self.latent_h, self.latent_w,
                              img.device)
                rec = self.model.decode(zi * (1 - m))
                row.append(tensor_to_pil(rec[0]))
            rows.append(row)

        w, h, pad = 96, 96, 22
        total_w = 44 + w * 4
        total_h = pad + h * len(rows)
        comp = Image.new('RGB', (total_w, total_h), 'white')
        d = ImageDraw.Draw(comp)
        for j, name in enumerate(['Original', 'Random', 'Half', 'Middle']):
            d.text((44 + j * w + 4, 4), name, fill='black')
        for i, row in enumerate(rows):
            y = pad + i * h
            d.text((4, y + h // 2), f'#{i}', fill='black')
            for j, img in enumerate(row):
                comp.paste(img.resize((w, h), Image.NEAREST), (44 + j * w, y))
        return comp

    # ------------------------------------------------------------- inference
    def _load_infer_image(self):
        p = filedialog.askopenfilename(
            filetypes=[('Images', '*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff')])
        if not p:
            return
        if not self._apply_settings():
            return
        img = Image.open(p)
        img = img.convert('RGB' if self.color_mode == 'rgb' else 'L')
        img = img.resize((self.img_size, self.img_size), Image.BICUBIC)
        self._inference_pil = img

        if self.color_mode == 'rgb':
            arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
        else:
            arr = np.asarray(img, dtype=np.float32)[..., None] / 127.5 - 1.0
        self._inference_tensor = (torch.from_numpy(arr)
                                  .permute(2, 0, 1)
                                  .unsqueeze(0).to(self.device))
        if self.infer_mask.shape != (self.latent_h, self.latent_w):
            self.infer_mask = np.zeros((self.latent_h, self.latent_w), dtype=np.uint8)
        self._redraw_canvas()

    def _redraw_canvas(self):
        c = self.infer_canvas
        c.delete('all')
        self.infer_canvas_hint.set(
            f'Left-drag paint · Right-drag erase  '
            f'({self.latent_w}×{self.latent_h} latent grid)')
        if self._inference_pil is None:
            c.create_text(self._canvas_size // 2, self._canvas_size // 2,
                          text='Load an image', fill='white')
            self._canvas_photo = None
            return

        big = self._inference_pil.convert('RGB').resize(
            (self._canvas_size, self._canvas_size), Image.NEAREST)
        self._canvas_photo = ImageTk.PhotoImage(big)
        c.create_image(0, 0, anchor='nw', image=self._canvas_photo)

        cw = self._canvas_size / self.latent_w
        ch = self._canvas_size / self.latent_h
        for i in range(self.latent_w + 1):
            x = i * cw
            c.create_line(x, 0, x, self._canvas_size, fill='#d2d2d2')
        for i in range(self.latent_h + 1):
            y = i * ch
            c.create_line(0, y, self._canvas_size, y, fill='#d2d2d2')

        for r in range(self.latent_h):
            for cc in range(self.latent_w):
                if self.infer_mask[r, cc]:
                    self._draw_mask_rect(r, cc)

    def _draw_mask_rect(self, r, cc):
        try:
            cw = self._canvas_size / self.latent_w
            ch = self._canvas_size / self.latent_h
            if not (math.isfinite(cw) and math.isfinite(ch)) or cw <= 0 or ch <= 0:
                return
            self.infer_canvas.create_rectangle(
                cc * cw, r * ch, (cc + 1) * cw, (r + 1) * ch,
                fill='#dc1e1e', stipple='gray50', outline='',
                tags=f'mask_{r}_{cc}')
        except tk.TclError:
            pass

    def _cell_at(self, x, y):
        if self._inference_pil is None:
            return None
        try:
            if not (math.isfinite(x) and math.isfinite(y)):
                return None
            if self.latent_w <= 0 or self.latent_h <= 0:
                return None
            cw = self._canvas_size / self.latent_w
            ch = self._canvas_size / self.latent_h
            if not (math.isfinite(cw) and math.isfinite(ch)) or cw <= 0 or ch <= 0:
                return None
            cc = int(math.floor(x / cw))
            r = int(math.floor(y / ch))
            if 0 <= r < self.latent_h and 0 <= cc < self.latent_w:
                return r, cc
        except (OverflowError, ValueError, ZeroDivisionError):
            pass
        return None

    def _stroke_segment(self, r0, c0, r1, c1, value):
        """PIL-based stroke. Mask stays 0/1 internally; scale to 0/255 for
        PIL 'L' mode so existing cells aren't wiped on each segment."""
        try:
            r0 = int(r0); c0 = int(c0); r1 = int(r1); c1 = int(c1)
            value = int(value)
        except (TypeError, ValueError, OverflowError):
            return

        if self.infer_mask is None or self.infer_mask.shape != (self.latent_h, self.latent_w):
            return

        try:
            old_mask = self.infer_mask.copy()
            mask_img = Image.fromarray((self.infer_mask * 255).astype(np.uint8),
                                       mode='L')
            draw = ImageDraw.Draw(mask_img)
            brush = max(1, int(self._brush_size.get()))
            fill = 255 if value else 0
            # PIL uses (x, y) = (col, row)
            draw.line([c0, r0, c1, r1], fill=fill, width=brush)
            new_mask = np.array(mask_img, dtype=np.uint8)
            self.infer_mask = (new_mask > 127).astype(np.uint8)
        except Exception:
            import traceback; traceback.print_exc()
            return

        changed = np.argwhere(old_mask != self.infer_mask)
        for r, c in changed:
            r = int(r); c = int(c)
            tag = f'mask_{r}_{c}'
            try:
                self.infer_canvas.delete(tag)
            except tk.TclError:
                pass
            if self.infer_mask[r, c]:
                self._draw_mask_rect(r, c)

    def _on_paint_start(self, event):
        if self._inference_pil is None:
            return
        try:
            self._paint_value = 1
            cell = self._cell_at(event.x, event.y)
            if cell is None:
                self._last_cell = None
                return
            self._stroke_segment(cell[0], cell[1], cell[0], cell[1], 1)
            self._last_cell = cell
        except Exception:
            import traceback; traceback.print_exc()
            self._paint_value = None
            self._last_cell = None

    def _on_erase_start(self, event):
        if self._inference_pil is None:
            return
        try:
            self._paint_value = 0
            cell = self._cell_at(event.x, event.y)
            if cell is None:
                self._last_cell = None
                return
            self._stroke_segment(cell[0], cell[1], cell[0], cell[1], 0)
            self._last_cell = cell
        except Exception:
            import traceback; traceback.print_exc()
            self._paint_value = None
            self._last_cell = None

    def _on_paint_drag(self, event):
        if self._paint_value is None or self._inference_pil is None:
            return
        try:
            cell = self._cell_at(event.x, event.y)
            if cell is None:
                return
            if self._last_cell is None:
                self._stroke_segment(cell[0], cell[1], cell[0], cell[1],
                                     self._paint_value)
            else:
                self._stroke_segment(self._last_cell[0], self._last_cell[1],
                                     cell[0], cell[1], self._paint_value)
            self._last_cell = cell
        except Exception:
            import traceback; traceback.print_exc()
            self._paint_value = None
            self._last_cell = None

    def _on_erase_drag(self, event):
        if self._inference_pil is None:
            return
        try:
            if self._paint_value != 0:
                self._paint_value = 0
                self._last_cell = None
            cell = self._cell_at(event.x, event.y)
            if cell is None:
                return
            if self._last_cell is None:
                self._stroke_segment(cell[0], cell[1], cell[0], cell[1], 0)
            else:
                self._stroke_segment(self._last_cell[0], self._last_cell[1],
                                     cell[0], cell[1], 0)
            self._last_cell = cell
        except Exception:
            import traceback; traceback.print_exc()
            self._paint_value = None
            self._last_cell = None

    def _on_paint_end(self, _event=None):
        self._paint_value = None
        self._last_cell = None

    def _on_canvas_leave(self, _event=None):
        if self._paint_value is not None:
            self._last_cell = None

    def _clear_mask(self):
        self.infer_mask[:] = 0
        self._redraw_canvas()

    def _preset_mask(self, kind):
        self.infer_mask[:] = 0
        if kind == 'random':
            try:
                ratio = float(self.train_settings['mask_ratio'].get())
            except Exception:
                ratio = 0.5
            self.infer_mask[:] = (np.random.rand(self.latent_h, self.latent_w)
                                  < ratio).astype(np.uint8)
        elif kind == 'half':
            self.infer_mask[:, :self.latent_w // 2] = 1
        elif kind == 'middle':
            qh, qw = max(1, self.latent_h // 4), max(1, self.latent_w // 4)
            self.infer_mask[qh:self.latent_h - qh, qw:self.latent_w - qw] = 1
        self._redraw_canvas()

    def _run_inference(self):
        if self._inference_tensor is None:
            messagebox.showerror('Error', 'Load an image first.'); return
        if self.model is None:
            messagebox.showerror('Error', 'Initialize the model first.'); return
        try:
            self.model.eval()
            with torch.no_grad():
                z = self.model.encode(self._inference_tensor)
                if self.infer_mask.shape != (self.latent_h, self.latent_w):
                    self.infer_mask = np.zeros((self.latent_h, self.latent_w),
                                               dtype=np.uint8)
                m = torch.from_numpy(self.infer_mask.astype(np.float32)) \
                           .to(self.device).unsqueeze(0).unsqueeze(0)
                if m.shape[-2:] != z.shape[-2:]:
                    m = F.interpolate(m, size=z.shape[-2:], mode='nearest')
                rec = self.model.decode(z * (1 - m))

            orig = tensor_to_pil(self._inference_tensor)
            recon = tensor_to_pil(rec)
            w = 128
            comp = Image.new('RGB', (w * 3, w + 20), 'white')
            d = ImageDraw.Draw(comp)
            comp.paste(orig.resize((w, w), Image.NEAREST), (0, 20))
            d.text((4, 3), 'Original', fill='black')

            ov = Image.new('RGBA', (w, w), (0, 0, 0, 0))
            od = ImageDraw.Draw(ov)
            cw = w / max(1, self.latent_w)
            ch = w / max(1, self.latent_h)
            for r in range(self.latent_h):
                for cc in range(self.latent_w):
                    if self.infer_mask[r, cc]:
                        od.rectangle([cc * cw, r * ch,
                                      (cc + 1) * cw, (r + 1) * ch],
                                     fill=(220, 30, 30, 160))
            masked_view = Image.alpha_composite(
                orig.resize((w, w), Image.NEAREST).convert('RGBA'),
                ov).convert('RGB')
            comp.paste(masked_view, (w, 20))
            d.text((w + 4, 3), 'Mask', fill='black')

            comp.paste(recon.resize((w, w), Image.NEAREST), (2 * w, 20))
            d.text((2 * w + 4, 3), 'Decoded', fill='black')

            photo = ImageTk.PhotoImage(comp)
            self.infer_output_label.configure(image=photo)
            self.infer_output_label.image = photo
        except Exception as e:
            import traceback; traceback.print_exc()
            messagebox.showerror('Inference error', str(e))

    def _poll_queue(self):
        try:
            while True:
                kind, data = self.msg_queue.get_nowait()
                if kind == 'epoch':
                    e, loss, total, kl = data
                    self.train_status.set(
                        f'Epoch {e}/{total}  loss={loss:.5f}  kl_w={kl}')
                elif kind == 'preview':
                    photo = ImageTk.PhotoImage(data)
                    self.preview_label.configure(image=photo)
                    self.preview_label.image = photo
                elif kind == 'done':
                    self.train_status.set('Training finished.')
                elif kind == 'error':
                    messagebox.showerror('Training error', data)
                    self.train_status.set('Error.')
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)


def main():
    root = tk.Tk()

    def _report_callback_exception(exc, val, tb):
        import traceback
        sys.stderr.write('--- Tkinter callback error (recovered) ---\n')
        traceback.print_exception(exc, val, tb)
        sys.stderr.flush()
    root.report_callback_exception = _report_callback_exception

    App(root)
    root.mainloop()


if __name__ == '__main__':
    main()