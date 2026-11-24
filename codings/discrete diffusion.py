"""
Discrete Diffusion App — PyTorch + Tkinter
Tab 1 Dataset: color-palette images OR tileset/tilemaps; draw, load, extract tiles
Tab 2 Settings: UNet config + augmentations
Tab 3 Training: init / train / stop, epoch & batch size
Tab 4 Inference: steps, scheduler (linear/cosine/confidence/halton), live preview
"""
import os, math, threading, queue
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser
from PIL import Image, ImageTk, ImageDraw

# ------------------------- Halton helper -------------------------
def _halton(index, base):
    result, f, i = 0.0, 1.0, index
    while i > 0:
        f /= base
        result += f * (i % base)
        i //= base
    return result

# ------------------------- Model -------------------------
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim * 4)
        )
    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None] * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, temb_dim):
        super().__init__()
        g1, g2 = max(1, min(8, in_ch)), max(1, min(8, out_ch))
        self.norm1 = nn.GroupNorm(g1, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.temb = nn.Linear(temb_dim, out_ch)
        self.norm2 = nn.GroupNorm(g2, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.temb(temb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)

class UNet(nn.Module):
    def __init__(self, vocab_size, base_ch=64, mults=(1, 2, 4), tdim=128):
        super().__init__()
        self.vocab_size = vocab_size
        temb_dim = tdim * 4
        self.time_mlp = TimeEmbedding(tdim)
        self.input_conv = nn.Conv2d(vocab_size, base_ch, 3, padding=1)
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        skip_chs = []
        ch = base_ch
        for i, m in enumerate(mults):
            out_ch = base_ch * m
            self.down_blocks.append(ResBlock(ch, out_ch, temb_dim))
            ch = out_ch
            skip_chs.append(ch)
            if i < len(mults) - 1:
                self.down_samples.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
            else:
                self.down_samples.append(nn.Identity())
        self.mid1 = ResBlock(ch, ch, temb_dim)
        self.mid2 = ResBlock(ch, ch, temb_dim)
        self.up_samples = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for i, m in enumerate(reversed(mults)):
            skip_ch = skip_chs[-(i + 1)]
            if i > 0:
                self.up_samples.append(
                    nn.ConvTranspose2d(ch, ch, 3, stride=2, padding=1, output_padding=1)
                )
            else:
                self.up_samples.append(nn.Identity())
            self.up_blocks.append(ResBlock(ch + skip_ch, skip_ch, temb_dim))
            ch = skip_ch
        self.out_norm = nn.GroupNorm(max(1, min(8, ch)), ch)
        self.out_conv = nn.Conv2d(ch, vocab_size, 3, padding=1)
    def forward(self, x, t):
        temb = self.time_mlp(t)
        h = F.one_hot(x.long(), self.vocab_size).permute(0, 3, 1, 2).float()
        h = self.input_conv(h)
        skips = []
        for block, sample in zip(self.down_blocks, self.down_samples):
            h = block(h, temb)
            skips.append(h)
            h = sample(h)
        h = self.mid1(h, temb)
        h = self.mid2(h, temb)
        for i, (sample, block) in enumerate(zip(self.up_samples, self.up_blocks)):
            h = sample(h)
            skip = skips[-(i + 1)]
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = torch.cat([h, skip], dim=1)
            h = block(h, temb)
        return self.out_conv(F.silu(self.out_norm(h)))

def alpha_bar(t):
    return torch.cos(t * math.pi / 2.0) ** 2

class DiscreteDiffusion:
    def __init__(self, vocab_size, base_ch, mults, tdim, device="cpu"):
        self.vocab_size = vocab_size
        self.device = device
        self.model = UNet(vocab_size, base_ch, mults, tdim).to(device)
        self.optim = None
    def make_optimizer(self, lr=1e-3):
        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)
    def train_step(self, x_0):
        B = x_0.shape[0]
        t = torch.rand(B, device=self.device)
        ab = alpha_bar(t)[:, None, None]
        mask = torch.rand(B, *x_0.shape[1:], device=self.device) < ab
        rand = torch.randint(0, self.vocab_size, x_0.shape, device=self.device)
        x_t = torch.where(mask, x_0, rand)
        logits = self.model(x_t, t)
        loss = F.cross_entropy(logits, x_0)
        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optim.step()
        return loss.item()
    def _make_timesteps(self, steps, scheduler):
        device = self.device
        if scheduler == "cosine":
            s = torch.linspace(0, math.pi / 2, steps, device=device)
            return torch.cos(s)
        elif scheduler == "halton":
            vals = torch.tensor([_halton(i + 1, 2) for i in range(steps)],
                                device=device, dtype=torch.float32)
            vals = vals / vals.max()
            vals = vals.sort(descending=True).values
            vals[0] = 1.0
            vals[-1] = 0.0
            return vals
        else:  # linear
            return torch.linspace(1.0, 0.0, steps, device=device)
    @torch.no_grad()
    def sample(self, shape, steps=50, scheduler="cosine", on_step=None):
        if scheduler == "confidence":
            return self._sample_confidence(shape, steps, on_step)
        B, H, W = shape
        device = self.device
        ts = self._make_timesteps(steps, scheduler)
        x = torch.randint(0, self.vocab_size, (B, H, W), device=device)
        for i, tv in enumerate(ts):
            t = tv.expand(B)
            logits = self.model(x, t)
            probs = F.softmax(logits, dim=1)
            flat = probs.permute(0, 2, 3, 1).reshape(-1, self.vocab_size)
            x_0 = torch.multinomial(flat, 1).reshape(B, H, W)
            if i + 1 < len(ts):
                t_next = ts[i + 1].item()
                ab = math.cos(t_next * math.pi / 2.0) ** 2
                keep = torch.rand(B, H, W, device=device) < ab
                rand = torch.randint(0, self.vocab_size, (B, H, W), device=device)
                x = torch.where(keep, x_0, rand)
                mask = keep
            else:
                x = x_0
                mask = torch.ones(B, H, W, dtype=torch.bool, device=device)
            if on_step:
                on_step(i, x_0, mask)
        return x.cpu()
    @torch.no_grad()
    def _sample_confidence(self, shape, steps, on_step):
        B, H, W = shape
        device = self.device
        x = torch.randint(0, self.vocab_size, (B, H, W), device=device)
        known = torch.zeros(B, H, W, dtype=torch.bool, device=device)
        for i in range(steps):
            t_val = max(0.0, 1.0 - (i + 1) / steps)
            t = torch.full((B,), t_val, device=device)
            logits = self.model(x, t)
            probs = F.softmax(logits, dim=1)
            conf, x_0 = probs.max(dim=1)
            target = int(round((i + 1) / steps * H * W))
            for b in range(B):
                cur = int(known[b].sum().item())
                need = target - cur
                if need <= 0: continue
                unknown = ~known[b].flatten()
                n_unk = int(unknown.sum().item())
                if n_unk == 0: continue
                c = conf[b].flatten().clone()
                c[~unknown] = -1.0
                k = min(need, n_unk)
                _, top = c.topk(k)
                known[b].view(-1)[top] = True
            rand = torch.randint(0, self.vocab_size, (B, H, W), device=device)
            x = torch.where(known, x_0, rand)
            if on_step:
                on_step(i, x_0, known)
        return x.cpu()

# ------------------------- Dataset -------------------------
class DataSet:
    def __init__(self):
        self.mode = "image"
        self.palette = [(255, 255, 255), (0, 0, 0), (220, 60, 60), (60, 120, 220)]
        self.tileset = []
        self.samples = []
    @property
    def vocab_size(self):
        return len(self.palette) if self.mode == "image" else len(self.tileset)

# ------------------------- App -------------------------
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Discrete Diffusion — PyTorch + Tkinter")
        self.data = DataSet()
        self.diffusion = None
        self.training_thread = None
        self.stop_flag = False
        self.current_sample = None
        self.current_idx = 0
        self.log_queue = queue.Queue()
        self._tile_cell = 36
        self._build_ui()
        self._poll_log()

    def _build_ui(self):
        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True)
        self.tab_data = ttk.Frame(nb)
        self.tab_set = ttk.Frame(nb)
        self.tab_train = ttk.Frame(nb)
        self.tab_inf = ttk.Frame(nb)
        nb.add(self.tab_data, text="1  Dataset")
        nb.add(self.tab_set, text="2  Settings")
        nb.add(self.tab_train, text="3  Training")
        nb.add(self.tab_inf, text="4  Inference")
        self._build_dataset_tab()
        self._build_settings_tab()
        self._build_training_tab()
        self._build_inference_tab()
        self.new_canvas()
        self._rebuild_palette_panel()

    # ---- Dataset tab ----
    def _build_dataset_tab(self):
        f = self.tab_data
        top = ttk.Frame(f); top.pack(fill="x", padx=6, pady=6)
        self.mode_var = tk.StringVar(value="image")
        ttk.Label(top, text="Mode:").pack(side="left")
        ttk.Radiobutton(top, text="Color palette", variable=self.mode_var,
                        value="image", command=self.on_mode_change).pack(side="left", padx=3)
        ttk.Radiobutton(top, text="Tileset", variable=self.mode_var,
                        value="tiles", command=self.on_mode_change).pack(side="left", padx=3)
        ttk.Label(top, text="   Canvas W:").pack(side="left")
        self.w_var = tk.IntVar(value=16)
        ttk.Entry(top, textvariable=self.w_var, width=4).pack(side="left")
        ttk.Label(top, text="H:").pack(side="left")
        self.h_var = tk.IntVar(value=16)
        ttk.Entry(top, textvariable=self.h_var, width=4).pack(side="left")
        ttk.Button(top, text="New Canvas", command=self.new_canvas).pack(side="left", padx=4)
        
        main = ttk.Frame(f); main.pack(fill="both", expand=True, padx=6, pady=4)
        left = ttk.Frame(main, width=220); left.pack(side="left", fill="y")
        left.pack_propagate(False)
        self.palette_panel = ttk.Frame(left); self.palette_panel.pack(fill="both", expand=True)
        
        center = ttk.Frame(main); center.pack(side="left", fill="both", expand=True, padx=6)
        self.canvas = tk.Canvas(center, bg="#222")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Button-1>", self.on_paint)
        self.canvas.bind("<B1-Motion>", self.on_paint)
        self.canvas.bind("<Button-3>", self.on_erase)
        
        right = ttk.Frame(main, width=220); right.pack(side="right", fill="y")
        right.pack_propagate(False)
        ttk.Label(right, text="Samples").pack()
        self.listbox = tk.Listbox(right, height=20)
        self.listbox.pack(fill="both", expand=True)
        self.listbox.bind("<<ListboxSelect>>", self.on_pick_sample)
        bf = ttk.Frame(right); bf.pack(fill="x")
        ttk.Button(bf, text="Add current", command=self.add_current).pack(fill="x")
        ttk.Button(bf, text="Load selected", command=self.load_selected).pack(fill="x")
        ttk.Button(bf, text="Delete selected", command=self.delete_selected).pack(fill="x")
        ttk.Button(bf, text="Clear all", command=self.clear_samples).pack(fill="x")
        
        bottom = ttk.Frame(f); bottom.pack(fill="x", padx=6, pady=6)
        ttk.Button(bottom, text="Load Images as Dataset", command=self.load_images).pack(side="left")
        ttk.Button(bottom, text="Load Tileset (sliced)", command=self.load_tileset).pack(side="left", padx=6)
        ttk.Button(bottom, text="Load Individual Tiles", command=self.load_individual_tiles).pack(side="left")
        ttk.Button(bottom, text="Extract Tiles From Image", command=self.extract_tiles).pack(side="left", padx=6)
        
        ttk.Label(bottom, text="  Quantization (Levels/Ch, 0=off):").pack(side="left")
        self.quant_var = tk.IntVar(value=0)
        ttk.Entry(bottom, textvariable=self.quant_var, width=4).pack(side="left")
        ttk.Label(bottom, text="(e.g. 2=1bit, 4=2bit, 8=3bit)").pack(side="left", padx=4)

    def _rebuild_palette_panel(self):
        for w in self.palette_panel.winfo_children():
            w.destroy()
        if self.mode_var.get() == "image":
            ttk.Label(self.palette_panel, text="Palette").pack(anchor="w")
            self.pal_list = tk.Listbox(self.palette_panel, height=12)
            self.pal_list.pack(fill="both", expand=True)
            self.pal_list.bind("<<ListboxSelect>>", self.on_pick_color)
            self._refresh_pal_list()
            bf = ttk.Frame(self.palette_panel); bf.pack(fill="x")
            ttk.Button(bf, text="+", width=3, command=self.add_color).pack(side="left")
            ttk.Button(bf, text="−", width=3, command=self.remove_color).pack(side="left")
            ttk.Button(bf, text="Edit", command=self.edit_color).pack(side="left")
            ttk.Button(bf, text="Reset", command=self.reset_palette).pack(side="left", padx=5)
        else:
            ttk.Label(self.palette_panel, text="Tiles").pack(anchor="w")
            self.tiles_canvas = tk.Canvas(self.palette_panel, bg="#222", height=340)
            self.tiles_canvas.pack(fill="both", expand=True)
            self.tiles_canvas.bind("<Button-1>", self.on_pick_tile)
            self._refresh_tiles_panel()

    def _refresh_pal_list(self):
        self.pal_list.delete(0, "end")
        for i, c in enumerate(self.data.palette):
            self.pal_list.insert("end", f"{i}: #{c[0]:02x}{c[1]:02x}{c[2]:02x}")
        if self.data.palette:
            self.pal_list.selection_set(min(self.current_idx, len(self.data.palette) - 1))

    def _refresh_tiles_panel(self):
        if not hasattr(self, "tiles_canvas"): return
        self.tiles_canvas.delete("all")
        self._tile_photos = []
        if not self.data.tileset: return
        maxdim = max(max(t.width, t.height) for t in self.data.tileset)
        cell = min(48, max(24, maxdim))
        cols = 4; pad = 4
        self._tile_cell = cell + pad
        for i, t in enumerate(self.data.tileset):
            r, c = divmod(i, cols)
            x, y = c * (cell + pad), r * (cell + pad)
            im = t.copy()
            im.thumbnail((cell, cell), Image.NEAREST)
            bg = Image.new("RGBA", (cell, cell), (60, 60, 60, 255))
            if im.mode == "RGBA":
                bg.paste(im, ((cell - im.width) // 2, (cell - im.height) // 2), im)
            else:
                bg.paste(im, ((cell - im.width) // 2, (cell - im.height) // 2))
            ph = ImageTk.PhotoImage(bg)
            self._tile_photos.append(ph)
            self.tiles_canvas.create_image(x, y, image=ph, anchor="nw")
            color = "yellow" if i == self.current_idx else "gray"
            self.tiles_canvas.create_rectangle(x, y, x + cell, y + cell, outline=color)
            self.tiles_canvas.create_text(x + 2, y + 2, text=str(i), anchor="nw", fill="cyan")

    # ---- Settings tab ----
    def _build_settings_tab(self):
        f = self.tab_set
        frame = ttk.LabelFrame(f, text="UNet")
        frame.pack(fill="x", padx=10, pady=10)
        self.base_ch_var = tk.IntVar(value=64)
        self.mults_var = tk.StringVar(value="1,2,4")
        self.tdim_var = tk.IntVar(value=128)
        self.lr_var = tk.DoubleVar(value=1e-3)
        rows = [("Base channels", self.base_ch_var),
                ("Multipliers (comma separated)", self.mults_var),
                ("Time-embedding dim", self.tdim_var),
                ("Learning rate", self.lr_var)]
        for i, (lbl, var) in enumerate(rows):
            ttk.Label(frame, text=lbl).grid(row=i, column=0, sticky="w", padx=5, pady=3)
            ttk.Entry(frame, textvariable=var).grid(row=i, column=1, sticky="we", padx=5, pady=3)
        frame.columnconfigure(1, weight=1)
        
        aug = ttk.LabelFrame(f, text="Augmentations (applied per-batch during training)")
        aug.pack(fill="x", padx=10, pady=10)
        self.aug_on_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(aug, text="Enable augmentations", variable=self.aug_on_var)\
            .grid(row=0, column=0, columnspan=8, sticky="w", padx=5, pady=2)
            
        # Probabilities & Settings Grid
        self.aug_p_hflip = tk.DoubleVar(value=0.5)
        self.aug_p_vflip = tk.DoubleVar(value=0.5)
        self.aug_p_rot = tk.DoubleVar(value=0.5)
        
        ttk.Label(aug, text="H-Flip P:").grid(row=1, column=0, sticky="w", padx=25, pady=2)
        ttk.Entry(aug, textvariable=self.aug_p_hflip, width=5).grid(row=1, column=1, sticky="w")
        ttk.Label(aug, text="V-Flip P:").grid(row=1, column=2, sticky="w", padx=10, pady=2)
        ttk.Entry(aug, textvariable=self.aug_p_vflip, width=5).grid(row=1, column=3, sticky="w")
        ttk.Label(aug, text="Rot90 P:").grid(row=1, column=4, sticky="w", padx=10, pady=2)
        ttk.Entry(aug, textvariable=self.aug_p_rot, width=5).grid(row=1, column=5, sticky="w")
        
        # Shear
        self.aug_shear_var = tk.BooleanVar(value=False)
        self.aug_p_shear = tk.DoubleVar(value=0.5)
        self.aug_max_shear_h = tk.DoubleVar(value=15.0)
        self.aug_max_shear_v = tk.DoubleVar(value=15.0)
        ttk.Checkbutton(aug, text="Shear", variable=self.aug_shear_var).grid(row=2, column=0, sticky="w", padx=5, pady=2)
        ttk.Label(aug, text="P:").grid(row=2, column=1, sticky="w", padx=5, pady=2)
        ttk.Entry(aug, textvariable=self.aug_p_shear, width=5).grid(row=2, column=2, sticky="w")
        ttk.Label(aug, text="Max H°:").grid(row=2, column=3, sticky="w", padx=10, pady=2)
        ttk.Entry(aug, textvariable=self.aug_max_shear_h, width=5).grid(row=2, column=4, sticky="w")
        ttk.Label(aug, text="Max V°:").grid(row=2, column=5, sticky="w", padx=10, pady=2)
        ttk.Entry(aug, textvariable=self.aug_max_shear_v, width=5).grid(row=2, column=6, sticky="w")
        
        # Paeth Rotation
        self.aug_paeth_var = tk.BooleanVar(value=False)
        self.aug_p_paeth = tk.DoubleVar(value=0.5)
        self.aug_max_paeth_deg = tk.DoubleVar(value=30.0)
        ttk.Checkbutton(aug, text="Paeth Rot", variable=self.aug_paeth_var).grid(row=3, column=0, sticky="w", padx=5, pady=2)
        ttk.Label(aug, text="P:").grid(row=3, column=1, sticky="w", padx=5, pady=2)
        ttk.Entry(aug, textvariable=self.aug_p_paeth, width=5).grid(row=3, column=2, sticky="w")
        ttk.Label(aug, text="Max Deg:").grid(row=3, column=3, sticky="w", padx=10, pady=2)
        ttk.Entry(aug, textvariable=self.aug_max_paeth_deg, width=5).grid(row=3, column=4, sticky="w")
        
        # Offset
        self.aug_offset_var = tk.BooleanVar(value=False)
        self.aug_p_offset = tk.DoubleVar(value=0.5)
        self.aug_max_off_h = tk.IntVar(value=4)
        self.aug_max_off_v = tk.IntVar(value=4)
        ttk.Checkbutton(aug, text="Offset", variable=self.aug_offset_var).grid(row=4, column=0, sticky="w", padx=5, pady=2)
        ttk.Label(aug, text="P:").grid(row=4, column=1, sticky="w", padx=5, pady=2)
        ttk.Entry(aug, textvariable=self.aug_p_offset, width=5).grid(row=4, column=2, sticky="w")
        ttk.Label(aug, text="Max H-px:").grid(row=4, column=3, sticky="w", padx=10, pady=2)
        ttk.Entry(aug, textvariable=self.aug_max_off_h, width=5).grid(row=4, column=4, sticky="w")
        ttk.Label(aug, text="Max V-px:").grid(row=4, column=5, sticky="w", padx=10, pady=2)
        ttk.Entry(aug, textvariable=self.aug_max_off_v, width=5).grid(row=4, column=6, sticky="w")

    # ---- Training tab ----
    def _build_training_tab(self):
        f = self.tab_train
        frame = ttk.Frame(f); frame.pack(fill="x", padx=10, pady=8)
        self.epochs_var = tk.IntVar(value=10)
        self.bs_var = tk.IntVar(value=16)
        ttk.Label(frame, text="Epochs").grid(row=0, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.epochs_var, width=8).grid(row=0, column=1)
        ttk.Label(frame, text="Batch size").grid(row=0, column=2, sticky="w", padx=(20, 0))
        ttk.Entry(frame, textvariable=self.bs_var, width=8).grid(row=0, column=3)
        bf = ttk.Frame(f); bf.pack(fill="x", padx=10, pady=6)
        ttk.Button(bf, text="Initialize Model", command=self.init_model).pack(side="left")
        ttk.Button(bf, text="Train", command=self.start_training).pack(side="left", padx=6)
        ttk.Button(bf, text="Stop", command=self.stop_training).pack(side="left")
        self.train_log = tk.Text(f, height=18)
        self.train_log.pack(fill="both", expand=True, padx=10, pady=6)

    # ---- Inference tab ----
    def _build_inference_tab(self):
        f = self.tab_inf
        frame = ttk.Frame(f); frame.pack(fill="x", padx=10, pady=8)
        self.steps_var = tk.IntVar(value=50)
        self.nsamp_var = tk.IntVar(value=4)
        self.scheduler_var = tk.StringVar(value="cosine")
        ttk.Label(frame, text="Steps").grid(row=0, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.steps_var, width=8).grid(row=0, column=1)
        ttk.Label(frame, text="Samples").grid(row=0, column=2, sticky="w", padx=(20, 0))
        ttk.Entry(frame, textvariable=self.nsamp_var, width=8).grid(row=0, column=3)
        ttk.Label(frame, text="Scheduler").grid(row=0, column=4, sticky="w", padx=(20, 0))
        ttk.Combobox(frame, textvariable=self.scheduler_var,
                     values=["linear", "cosine", "confidence", "halton"],
                     state="readonly", width=12).grid(row=0, column=5)
        ttk.Button(frame, text="Generate", command=self.generate).grid(row=0, column=6, padx=10)
        ttk.Button(frame, text="Preview Augmentations", command=self.preview_aug).grid(row=1, column=0, columnspan=7, pady=5)
        
        # Scrollable Canvas Frame
        self.gen_frame = ttk.Frame(f)
        self.gen_frame.pack(fill="both", expand=True, padx=10, pady=6)
        self.gen_canvas = tk.Canvas(self.gen_frame, bg="#222")
        self.gen_canvas.pack(side="left", fill="both", expand=True)
        self.gen_scroll_y = ttk.Scrollbar(self.gen_frame, orient="vertical", command=self.gen_canvas.yview)
        self.gen_scroll_y.pack(side="right", fill="y")
        self.gen_canvas.configure(yscrollcommand=self.gen_scroll_y.set)
        
        # Mousewheel binding
        self.gen_canvas.bind("<MouseWheel>", lambda e: self.gen_canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
        self.gen_canvas.bind("<Button-4>", lambda e: self.gen_canvas.yview_scroll(-1, "units"))
        self.gen_canvas.bind("<Button-5>", lambda e: self.gen_canvas.yview_scroll(1, "units"))

    # ---------------- Dataset interaction ----------------
    def on_mode_change(self):
        self.data.mode = self.mode_var.get()
        self.current_idx = 0
        self._rebuild_palette_panel()
        self.refresh_canvas()

    def new_canvas(self):
        w = max(2, int(self.w_var.get())); h = max(2, int(self.h_var.get()))
        self.current_sample = np.zeros((h, w), dtype=np.int32)
        self.refresh_canvas()

    def _render(self, sample, cell=16, mask=None):
        h, w = sample.shape
        if self.data.mode == "image":
            bg_col = (128, 128, 128) if mask is not None else (0, 0, 0)
            img = Image.new("RGB", (w, h), bg_col)
            px = img.load()
            for y in range(h):
                for x in range(w):
                    if mask is not None and not mask[y, x]: continue
                    i = int(sample[y, x])
                    if i >= len(self.data.palette): i = 0
                    if self.data.palette: px[x, y] = self.data.palette[i]
            return img.resize((w * cell, h * cell), Image.NEAREST)
        else:
            bg_col = (128, 128, 128, 255) if mask is not None else (40, 40, 40, 255)
            img = Image.new("RGBA", (w * cell, h * cell), bg_col)
            for y in range(h):
                for x in range(w):
                    if mask is not None and not mask[y, x]: continue
                    i = int(sample[y, x])
                    if i < len(self.data.tileset):
                        tile = self.data.tileset[i].resize((cell, cell), Image.NEAREST)
                        img.paste(tile, (x * cell, y * cell), tile if tile.mode == "RGBA" else None)
            return img.convert("RGB")

    def render_sample_image(self, sample, cell=16): return self._render(sample, cell, mask=None)
    def render_sample_image_masked(self, sample, mask, cell=16): return self._render(sample, cell, mask=mask)

    def refresh_canvas(self):
        self.canvas.delete("all")
        if self.current_sample is None: return
        img = self.render_sample_image(self.current_sample, cell=16)
        self._canvas_photo = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, image=self._canvas_photo, anchor="nw")

    def on_paint(self, event):
        if self.current_sample is None: return
        cell = 16
        x, y = event.x // cell, event.y // cell
        h, w = self.current_sample.shape
        if 0 <= x < w and 0 <= y < h:
            if self.current_idx < self.data.vocab_size:
                self.current_sample[y, x] = self.current_idx
                self.refresh_canvas()

    def on_erase(self, event):
        if self.current_sample is None: return
        cell = 16
        x, y = event.x // cell, event.y // cell
        h, w = self.current_sample.shape
        if 0 <= x < w and 0 <= y < h:
            self.current_sample[y, x] = 0
            self.refresh_canvas()

    def on_pick_color(self, _evt):
        sel = self.pal_list.curselection()
        if sel: self.current_idx = sel[0]

    def on_pick_tile(self, event):
        cell = self._tile_cell or 36
        c = event.x // cell
        r = event.y // cell
        idx = r * 4 + c
        if 0 <= idx < len(self.data.tileset):
            self.current_idx = idx
            self._refresh_tiles_panel()

    def add_color(self):
        c = colorchooser.askcolor()
        if c and c[0]:
            self.data.palette.append(tuple(int(v) for v in c[0]))
            self._refresh_pal_list()
            self.refresh_canvas()

    def remove_color(self):
        sel = self.pal_list.curselection()
        if sel and len(self.data.palette) > 1 and sel[0] > 0:
            del self.data.palette[sel[0]]
            self.current_idx = 0
            self._refresh_pal_list()
            self.refresh_canvas()

    def edit_color(self):
        sel = self.pal_list.curselection()
        if not sel: return
        cur = self.data.palette[sel[0]]
        c = colorchooser.askcolor(color=f"#{cur[0]:02x}{cur[1]:02x}{cur[2]:02x}")
        if c and c[0]:
            self.data.palette[sel[0]] = tuple(int(v) for v in c[0])
            self._refresh_pal_list()
            self.refresh_canvas()

    def reset_palette(self):
        self.data.palette = [(255, 255, 255), (0, 0, 0), (220, 60, 60), (60, 120, 220)]
        self.current_idx = 0
        self._refresh_pal_list()
        self.refresh_canvas()

    def add_current(self):
        if self.current_sample is None: return
        self.data.samples.append(self.current_sample.copy())
        self._refresh_sample_list()

    def _refresh_sample_list(self):
        self.listbox.delete(0, "end")
        for i, s in enumerate(self.data.samples):
            self.listbox.insert("end", f"Sample {i}   {s.shape[1]}×{s.shape[0]}")

    def on_pick_sample(self, _evt):
        sel = self.listbox.curselection()
        if not sel: return
        self.current_sample = self.data.samples[sel[0]].copy()
        h, w = self.current_sample.shape
        self.h_var.set(int(h)); self.w_var.set(int(w))
        self.refresh_canvas()

    def load_selected(self): self.on_pick_sample(None)
    def delete_selected(self):
        sel = self.listbox.curselection()
        if sel:
            del self.data.samples[sel[0]]
            self._refresh_sample_list()
    def clear_samples(self):
        self.data.samples = []
        self._refresh_sample_list()

    # --- loading ---
    def _quantize_rgb(self, arr, levels):
        if levels <= 1: return arr
        factor = 255.0 / (levels - 1)
        return np.round(arr / factor) * factor

    def load_images(self):
        files = filedialog.askopenfilenames(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.gif")])
        if not files: return
        self.mode_var.set("image"); self.data.mode = "image"
        quant = int(self.quant_var.get())
        
        for path in files:
            img = Image.open(path).convert("RGB")
            arr = np.asarray(img)
            if quant > 1:
                arr = self._quantize_rgb(arr, quant).astype(np.uint8)
                
            colors = np.unique(arr.reshape(-1, 3), axis=0)
            if len(colors) > 512: 
                messagebox.showerror("Too many colors", f"{path}: {len(colors)} colors. Try increasing quantization.")
                continue
                
            for c in colors:
                tup = tuple(int(v) for v in c)
                if tup not in self.data.palette:
                    self.data.palette.append(tup)
            idx = np.zeros(arr.shape[:2], dtype=np.int32)
            for i, c in enumerate(self.data.palette):
                idx[np.all(arr == np.array(c), axis=-1)] = i
            self.data.samples.append(idx)
        self._rebuild_palette_panel()
        self._refresh_sample_list()

    def load_tileset(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.bmp")])
        if not path: return
        self.mode_var.set("tiles"); self.data.mode = "tiles"
        self.data.tileset = []
        img = Image.open(path).convert("RGBA")
        tw = max(1, int(self.w_var.get())); th = max(1, int(self.h_var.get()))
        for y in range(0, img.height - th + 1, th):
            for x in range(0, img.width - tw + 1, tw):
                self.data.tileset.append(img.crop((x, y, x + tw, y + th)))
        self._rebuild_palette_panel()

    def load_individual_tiles(self):
        files = filedialog.askopenfilenames(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.gif")])
        if not files: return
        self.mode_var.set("tiles"); self.data.mode = "tiles"
        for path in files:
            img = Image.open(path).convert("RGBA")
            self.data.tileset.append(img)
        self.current_idx = 0
        self._rebuild_palette_panel()

    def extract_tiles(self):
        if not self.data.tileset:
            messagebox.showerror("Error", "Load a tileset first.")
            return
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.bmp")])
        if not path: return
        img = Image.open(path).convert("RGBA")
        tw = max(1, int(self.w_var.get())); th = max(1, int(self.h_var.get()))
        hcells = img.height // th; wcells = img.width // tw
        idx = np.zeros((hcells, wcells), dtype=np.int32)
        tile_arrs = [np.array(t) for t in self.data.tileset]
        
        for cy in range(hcells):
            for cx in range(wcells):
                tile = img.crop((cx * tw, cy * th, (cx + 1) * tw, (cy + 1) * th))
                arr = np.array(tile)
                best_match = -1
                min_diff = float('inf')
                for i, ta in enumerate(tile_arrs):
                    if ta.shape == arr.shape:
                        diff = np.sum(np.abs(ta.astype(int) - arr.astype(int)))
                        if diff < min_diff:
                            min_diff = diff
                            best_match = i
                if best_match == -1 or min_diff > 0:
                    self.data.tileset.append(tile)
                    tile_arrs.append(np.array(tile))
                    best_match = len(self.data.tileset) - 1
                idx[cy, cx] = best_match
                
        self.data.samples.append(idx)
        self._rebuild_palette_panel()
        self._refresh_sample_list()
        messagebox.showinfo("Extracted", f"{wcells}×{hcells} tilemap extracted.")

    # ---------------- Training ----------------
    def _log(self, msg): self.log_queue.put(msg)
    def _poll_log(self):
        while True:
            try: msg = self.log_queue.get_nowait()
            except queue.Empty: break
            self.train_log.insert("end", msg + "\n")
            self.train_log.see("end")
        self.root.after(120, self._poll_log)

    def init_model(self):
        if not self.data.samples:
            messagebox.showerror("Error", "Add samples first.")
            return
        K = self.data.vocab_size
        if K < 2:
            messagebox.showerror("Error", "Need at least 2 palette entries or tiles.")
            return
        try:
            base = int(self.base_ch_var.get())
            mults = tuple(int(x.strip()) for x in self.mults_var.get().split(",") if x.strip())
            tdim = int(self.tdim_var.get())
            lr = float(self.lr_var.get())
        except Exception as e:
            messagebox.showerror("Settings error", str(e)); return
        self.diffusion = DiscreteDiffusion(K, base, mults, tdim, device="cpu")
        self.diffusion.make_optimizer(lr)
        self._log(f"Model initialized: K={K}, base={base}, mults={mults}, tdim={tdim}, lr={lr}")

    def _apply_shear(self, arr, max_deg_h, max_deg_v):
        rad_h = math.radians(np.random.uniform(-max_deg_h, max_deg_h))
        rad_v = math.radians(np.random.uniform(-max_deg_v, max_deg_v))
        h, w = arr.shape[:2]
        # Horizontal shear (shift X based on Y)
        shifts_x = np.round(np.tan(rad_h) * np.arange(h)).astype(int)
        for y in range(h):
            arr[y] = np.roll(arr[y], shift=shifts_x[y], axis=0)
        # Vertical shear (shift Y based on X)
        shifts_y = np.round(np.tan(rad_v) * np.arange(w)).astype(int)
        for x in range(w):
            arr[:, x] = np.roll(arr[:, x], shift=shifts_y[x], axis=0)
        return arr

    def _apply_paeth_rotation(self, arr, max_deg):
        angle = np.random.uniform(-max_deg, max_deg)
        rad = math.radians(angle)
        h, w = arr.shape[:2]
        
        tan_half = math.tan(rad / 2)
        sin_theta = math.sin(rad)
        
        # 1. Horizontal shear by tan(theta/2)
        shifts_x1 = np.round(tan_half * np.arange(h)).astype(int)
        for y in range(h):
            arr[y] = np.roll(arr[y], shift=shifts_x1[y], axis=0)
            
        # 2. Vertical shear by -sin(theta)
        shifts_y = np.round(-sin_theta * np.arange(w)).astype(int)
        for x in range(w):
            arr[:, x] = np.roll(arr[:, x], shift=shifts_y[x], axis=0)
            
        # 3. Horizontal shear by tan(theta/2)
        for y in range(h):
            arr[y] = np.roll(arr[y], shift=shifts_x1[y], axis=0)
            
        return arr

    def _get_aug_cfg(self):
        return {
            "on": bool(self.aug_on_var.get()),
            "hflip": True, "vflip": True, "rot": True,
            "shear": bool(self.aug_shear_var.get()),
            "paeth": bool(self.aug_paeth_var.get()),
            "offset": bool(self.aug_offset_var.get()),
            "p_hflip": float(self.aug_p_hflip.get()),
            "p_vflip": float(self.aug_p_vflip.get()),
            "p_rot": float(self.aug_p_rot.get()),
            "p_shear": float(self.aug_p_shear.get()),
            "p_paeth": float(self.aug_p_paeth.get()),
            "p_offset": float(self.aug_p_offset.get()),
            "max_shear_h": float(self.aug_max_shear_h.get()),
            "max_shear_v": float(self.aug_max_shear_v.get()),
            "max_paeth_deg": float(self.aug_max_paeth_deg.get()),
            "max_off_h": int(self.aug_max_off_h.get()),
            "max_off_v": int(self.aug_max_off_v.get()),
        }

    def _apply_aug(self, arr, cfg):
        if cfg["hflip"] and np.random.random() < cfg["p_hflip"]:
            arr = np.fliplr(arr)
        if cfg["vflip"] and np.random.random() < cfg["p_vflip"]:
            arr = np.flipud(arr)
        if cfg["rot"] and np.random.random() < cfg["p_rot"]:
            k = np.random.randint(1, 4)
            arr = np.rot90(arr, k)
        if cfg["shear"] and np.random.random() < cfg["p_shear"]:
            arr = self._apply_shear(arr.copy(), cfg["max_shear_h"], cfg["max_shear_v"])
        if cfg["paeth"] and np.random.random() < cfg["p_paeth"]:
            arr = self._apply_paeth_rotation(arr.copy(), cfg["max_paeth_deg"])
        if cfg["offset"] and np.random.random() < cfg["p_offset"]:
            dy = np.random.randint(-cfg["max_off_v"], cfg["max_off_v"] + 1)
            dx = np.random.randint(-cfg["max_off_h"], cfg["max_off_h"] + 1)
            arr = np.roll(arr, shift=(dy, dx), axis=(0, 1))
        return np.ascontiguousarray(arr)

    def start_training(self):
        if self.diffusion is None:
            messagebox.showerror("Error", "Initialize the model first.")
            return
        if self.training_thread and self.training_thread.is_alive(): return
        samples = [s.copy() for s in self.data.samples]
        shapes = {s.shape for s in samples}
        if len(shapes) > 1:
            shape = next(iter(shapes))
            samples = [s for s in samples if s.shape == shape]
            self._log(f"Mixed sizes — using shape {shape}.")
        if not samples: return
        epochs = int(self.epochs_var.get())
        bs = max(1, int(self.bs_var.get()))
        aug_cfg = self._get_aug_cfg()
        
        self.stop_flag = False
        self.training_thread = threading.Thread(
            target=self._train_loop, args=(samples, epochs, bs, aug_cfg), daemon=True)
        self.training_thread.start()

    def _train_loop(self, samples, epochs, bs, aug_cfg):
        try:
            self._log("Training started." + (" (augmentations ON)" if aug_cfg["on"] else ""))
            for epoch in range(epochs):
                if self.stop_flag:
                    self._log("Stopped by user.")
                    return
                order = torch.randperm(len(samples))
                total, nb = 0.0, 0
                for i in range(0, len(samples), bs):
                    if self.stop_flag:
                        self._log("Stopped by user.")
                        return
                    idx = order[i:i + bs].tolist()
                    batch_list = []
                    for j in idx:
                        arr = samples[j]
                        if aug_cfg["on"]:
                            arr = self._apply_aug(arr, aug_cfg)
                        batch_list.append(torch.from_numpy(arr.astype(np.int64)))
                    batch = torch.stack(batch_list)
                    loss = self.diffusion.train_step(batch)
                    total += loss; nb += 1
                self._log(f"Epoch {epoch + 1}/{epochs}  loss={total / max(1, nb):.4f}")
            self._log("Training finished.")
        except Exception as e:
            import traceback
            self._log(f"Error: {e}\n{traceback.format_exc()}")

    def stop_training(self): self.stop_flag = True

    # ---------------- Inference ----------------
    def preview_aug(self):
        if not self.data.samples:
            messagebox.showerror("Error", "No samples to preview.")
            return
        idx = np.random.randint(0, len(self.data.samples))
        arr = self.data.samples[idx].copy()
        aug_cfg = self._get_aug_cfg()
        
        if aug_cfg["on"]:
            arr = self._apply_aug(arr, aug_cfg)
            
        img = self.render_sample_image(arr, cell=16)
        self._gen_photos = [ImageTk.PhotoImage(img)]
        self.gen_canvas.delete("all")
        self.gen_canvas.create_image(0, 0, image=self._gen_photos[0], anchor="nw")
        self.gen_canvas.create_text(5, 5, text="Augmentation Preview", fill="#0f0", anchor="nw")
        self.gen_canvas.configure(scrollregion=self.gen_canvas.bbox("all"))

    def generate(self):
        if self.diffusion is None:
            messagebox.showerror("Error", "Initialize the model first.")
            return
        if not self.data.samples:
            messagebox.showerror("Error", "No samples to infer shape from.")
            return
        h, w = self.data.samples[0].shape
        n = max(1, int(self.nsamp_var.get()))
        steps = max(1, int(self.steps_var.get()))
        scheduler = self.scheduler_var.get()
        def on_step(step, x_0, mask):
            self.root.after(0, lambda: self._update_preview(
                x_0.detach().cpu().numpy(),
                mask.detach().cpu().numpy(), step, steps))
        def run():
            self.diffusion.model.eval()
            with torch.no_grad():
                result = self.diffusion.sample(
                    (n, h, w), steps=steps, scheduler=scheduler, on_step=on_step)
            self.diffusion.model.train()
            self.root.after(0, lambda: self._show_generated(result.numpy()))
        threading.Thread(target=run, daemon=True).start()

    def _update_preview(self, samples_np, mask_np, step, total):
        self.gen_canvas.delete("all")
        self._gen_photos = []
        n = len(samples_np)
        cols = max(1, int(math.ceil(math.sqrt(n))))
        h, w = samples_np[0].shape
        cell = 12; pad = 8
        for i, s in enumerate(samples_np):
            img = self.render_sample_image_masked(s, mask_np[i], cell=cell)
            ph = ImageTk.PhotoImage(img)
            self._gen_photos.append(ph)
            r, c = divmod(i, cols)
            x = c * (w * cell + pad)
            y = r * (h * cell + pad)
            self.gen_canvas.create_image(x, y, image=ph, anchor="nw")
        self.gen_canvas.create_text(5, 5, text=f"Step {step + 1}/{total}", fill="#0f0", anchor="nw")
        self.gen_canvas.configure(scrollregion=self.gen_canvas.bbox("all"))

    def _show_generated(self, samples):
        self.gen_canvas.delete("all")
        self._gen_photos = []
        n = len(samples)
        cols = max(1, int(math.ceil(math.sqrt(n))))
        h, w = samples[0].shape
        cell = 12; pad = 8
        for i, s in enumerate(samples):
            img = self.render_sample_image(s, cell=cell)
            ph = ImageTk.PhotoImage(img)
            self._gen_photos.append(ph)
            r, c = divmod(i, cols)
            x = c * (w * cell + pad)
            y = r * (h * cell + pad)
            self.gen_canvas.create_image(x, y, image=ph, anchor="nw")
        self.gen_canvas.configure(scrollregion=self.gen_canvas.bbox("all"))

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1280x860")
    App(root)
    root.mainloop()