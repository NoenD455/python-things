import queue
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
from PIL import Image, ImageTk

BOARD = 128
CELL = 4
N_COLORS = 16
WINDOW = 7
MAX_PATCHES = 100_000

DEFAULT_PALETTE = np.array([
    [0, 0, 0], [255, 255, 255], [255, 0, 0], [0, 255, 0],
    [0, 0, 255], [255, 255, 0], [0, 255, 255], [255, 0, 255],
    [128, 0, 0], [0, 128, 0], [0, 0, 128], [128, 128, 0],
    [0, 128, 128], [128, 0, 128], [128, 128, 128], [192, 192, 192],
], dtype=np.uint8)


# --------------------------------------------------------------------------- #
#  Model
# --------------------------------------------------------------------------- #
class ImageNCA(nn.Module):
    def __init__(self, n_colors=N_COLORS, window=WINDOW, h1=128, h2=64):
        super().__init__()
        n_in = window * window * n_colors
        self.fc1 = nn.Linear(n_in, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, n_colors)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# --------------------------------------------------------------------------- #
#  Patch helpers
# --------------------------------------------------------------------------- #
def extract_patches_torch(img_idx, window=WINDOW, n_colors=N_COLORS):
    H, W = img_idx.shape
    half = window // 2
    oh = F.one_hot(img_idx, n_colors).float()
    oh = oh.permute(2, 0, 1).unsqueeze(0)
    oh = F.pad(oh, (half,) * 4, mode="circular")
    p = oh.unfold(2, window, 1).unfold(3, window, 1)
    p = p.permute(0, 2, 3, 4, 5, 1).reshape(-1, window, window, n_colors)
    c = window // 2
    p[:, c, c, :] = 0.0
    return p.reshape(-1, window * window * n_colors)


def apply_context_noise(patches_flat, n_colors, window, prob):
    if prob <= 0.0:
        return patches_flat
    N = patches_flat.shape[0]
    p = patches_flat.view(N, window * window, n_colors)
    center = (window // 2) * window + (window // 2)
    mask = torch.rand(N, window * window, device=patches_flat.device) < prob
    mask[:, center] = False
    rand_colors = torch.randint(0, n_colors, (N, window * window),
                                device=patches_flat.device)
    rand_oh = F.one_hot(rand_colors, n_colors).float()
    p = torch.where(mask.unsqueeze(-1), rand_oh, p)
    return p.reshape(N, -1)


# --------------------------------------------------------------------------- #
#  GUI
# --------------------------------------------------------------------------- #
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Neural CA — Image Texture Learner")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ImageNCA().to(self.device)
        self.stop_event = threading.Event()
        self.q = queue.Queue()
        self.worker = None

        self.palette = DEFAULT_PALETTE.copy()
        self.board = np.zeros((BOARD, BOARD), dtype=np.uint8)
        self.selected_color = tk.IntVar(value=1)
        self.photo = None

        # dataset
        self.dataset_rgb = []       # list of (H,W,3) uint8
        self.dataset_info = []      # list of dicts: name, size, orig
        self.indexed_images = None  # list of (H,W) int64 (set by prepare)

        # training tensors
        self.train_X = None
        self.train_y = None

        nb = ttk.Notebook(self)
        self.tab_dataset = ttk.Frame(nb)
        self.tab_train = ttk.Frame(nb)
        self.tab_infer = ttk.Frame(nb)
        nb.add(self.tab_dataset, text="  Dataset  ")
        nb.add(self.tab_train, text="  Training  ")
        nb.add(self.tab_infer, text="  Inference  ")
        nb.pack(fill="both", expand=True)

        self.build_dataset_tab()
        self.build_train_tab()
        self.build_infer_tab()

        self.log(f"Device: {self.device}")
        self.log(f"Model parameters: {self.count_params():,}")
        self.log(f"Window {WINDOW}x{WINDOW} | {N_COLORS} colors | board {BOARD}x{BOARD}")
        self.redraw()
        self.after(100, self.poll_queue)

    # ------------------------------------------------------------------ misc
    def count_params(self):
        return sum(p.numel() for p in self.model.parameters())

    def log(self, msg):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def poll_queue(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "log":
                    self.log(payload)
                elif kind == "status":
                    self.status_var.set(payload)
                elif kind == "done":
                    self.start_btn.config(state="normal")
                    self.stop_btn.config(state="disabled")
                    self.init_btn.config(state="normal")
        except queue.Empty:
            pass
        self.after(100, self.poll_queue)

    # ------------------------------------------------------------ dataset tab
    def build_dataset_tab(self):
        ctrl = ttk.Frame(self.tab_dataset, padding=8)
        ctrl.pack(fill="x")
        ttk.Button(ctrl, text="Add image(s)…",
                   command=self.add_images).pack(side="left")
        ttk.Button(ctrl, text="Remove selected",
                   command=self.remove_selected).pack(side="left", padx=4)
        ttk.Button(ctrl, text="Remove all",
                   command=self.remove_all).pack(side="left", padx=4)

        self.dataset_info_var = tk.StringVar(value="0 image(s)")
        ttk.Label(ctrl, textvariable=self.dataset_info_var).pack(side="left", padx=14)

        cols = ("name", "size", "orig", "colors")
        self.dataset_tree = ttk.Treeview(
            self.tab_dataset, columns=cols, show="headings", height=18)
        for c, w, h in (("name", 380, "Filename"),
                        ("size", 100, "Size"),
                        ("orig", 110, "Original"),
                        ("colors", 80, "Colors")):
            self.dataset_tree.heading(c, text=h)
            self.dataset_tree.column(c, width=w)
        self.dataset_tree.pack(fill="both", expand=True, padx=10, pady=8)

        self.log_text = ScrolledText(self.tab_dataset, height=8, width=90,
                                     state="disabled", font=("Courier New", 9))
        self.log_text.pack(fill="x", padx=10, pady=(0, 10))

    def add_images(self):
        paths = filedialog.askopenfilenames(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.gif *.webp"),
                       ("All files", "*.*")])
        if not paths:
            return
        added = 0
        for path in paths:
            try:
                img = Image.open(path).convert("RGB")
            except Exception as e:
                self.log(f"!! Failed to open {Path(path).name}: {e}")
                continue
            orig_w, orig_h = img.size
            w, h = orig_w, orig_h
            if max(w, h) > BOARD:
                scale = BOARD / max(w, h)
                w, h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
                img = img.resize((w, h), Image.LANCZOS)
            if min(w, h) < WINDOW:
                self.log(f"!! {Path(path).name}: too small ({w}x{h}), skipped")
                continue

            arr = np.array(img, dtype=np.uint8)
            # count distinct colors after quantization
            q = Image.fromarray(arr, "RGB").quantize(
                colors=N_COLORS, dither=Image.Dither.NONE)
            n_used = len(np.unique(np.array(q)))

            info = {"name": Path(path).name, "size": f"{w}x{h}",
                    "orig": f"{orig_w}x{orig_h}", "n_used": n_used}
            self.dataset_rgb.append(arr)
            self.dataset_info.append(info)
            self.dataset_tree.insert(
                "", "end",
                values=(info["name"], info["size"], info["orig"], info["n_used"]))
            added += 1
            self.log(f"Added: {Path(path).name}  ({w}x{h}, {n_used} colors)")

        if added:
            self.log(f"Total: {len(self.dataset_rgb)} image(s)")
            self.update_dataset_info()
            self.update_image_combo()
            self.indexed_images = None  # invalidate prepared data

    def remove_selected(self):
        sel = self.dataset_tree.selection()
        if not sel:
            return
        all_items = list(self.dataset_tree.get_children())
        indices = sorted([all_items.index(item) for item in sel], reverse=True)
        for i in indices:
            self.dataset_tree.delete(all_items[i])
            del self.dataset_rgb[i]
            del self.dataset_info[i]
        self.log(f"Removed {len(indices)} image(s). Total: {len(self.dataset_rgb)}")
        self.update_dataset_info()
        self.update_image_combo()
        self.indexed_images = None

    def remove_all(self):
        if not self.dataset_rgb:
            return
        self.dataset_tree.delete(*self.dataset_tree.get_children())
        self.dataset_rgb.clear()
        self.dataset_info.clear()
        self.indexed_images = None
        self.log("Dataset cleared.")
        self.update_dataset_info()
        self.update_image_combo()

    def update_dataset_info(self):
        n = len(self.dataset_rgb)
        px = sum(a.shape[0] * a.shape[1] for a in self.dataset_rgb)
        self.dataset_info_var.set(f"{n} image(s) | {px:,} pixels")

    # ----------------------------------------------------------- training tab
    def build_train_tab(self):
        pad = {"padx": 6, "pady": 4}

        box = ttk.LabelFrame(self.tab_train, text="Hyper-parameters", padding=8)
        box.pack(fill="x", padx=10, pady=(10, 4))

        ttk.Label(box, text="Iterations:").grid(row=0, column=0, sticky="w", **pad)
        self.iters_var = tk.StringVar(value="2000")
        ttk.Entry(box, textvariable=self.iters_var, width=12).grid(row=0, column=1, **pad)

        ttk.Label(box, text="Batch size:").grid(row=0, column=2, sticky="w", **pad)
        self.batch_var = tk.StringVar(value="2048")
        ttk.Entry(box, textvariable=self.batch_var, width=12).grid(row=0, column=3, **pad)

        ttk.Label(box, text="Learning rate:").grid(row=0, column=4, sticky="w", **pad)
        self.lr_var = tk.StringVar(value="0.001")
        ttk.Entry(box, textvariable=self.lr_var, width=12).grid(row=0, column=5, **pad)

        ttk.Label(box, text="Context noise (%):").grid(row=1, column=0, sticky="w", **pad)
        self.noise_var = tk.StringVar(value="0")
        ttk.Entry(box, textvariable=self.noise_var, width=12).grid(row=1, column=1, **pad)

        btn = ttk.Frame(self.tab_train, padding=(10, 4))
        btn.pack(fill="x")
        self.init_btn = ttk.Button(btn, text="Init model", command=self.init_model)
        self.init_btn.pack(side="left", padx=4)
        self.start_btn = ttk.Button(btn, text="Start training",
                                    command=self.start_training)
        self.start_btn.pack(side="left", padx=4)
        self.stop_btn = ttk.Button(btn, text="Stop training",
                                   command=self.stop_training, state="disabled")
        self.stop_btn.pack(side="left", padx=4)

        self.status_var = tk.StringVar(value="Idle.")
        ttk.Label(self.tab_train, textvariable=self.status_var).pack(
            anchor="w", padx=14)

        self.log_text2 = ScrolledText(self.tab_train, height=18, width=90,
                                      state="disabled", font=("Courier New", 9))
        self.log_text2.pack(fill="both", expand=True, padx=10, pady=(4, 10))
        # route log() output to this text box too
        self.log_text = self.log_text2

    def init_model(self):
        for m in self.model.children():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
        self.log(f"Model re-initialised — {self.count_params():,} parameters.")

    def prepare_dataset(self):
        """Quantize all images to a shared 16-color palette and extract patches."""
        if not self.dataset_rgb:
            self.log("!! No images in dataset.")
            return False

        t0 = time.time()
        self.log("Preparing dataset…")

        # 1) build per-image used-color palettes
        per_img_palettes = []
        for img in self.dataset_rgb:
            pil = Image.fromarray(img, "RGB")
            q = pil.quantize(colors=N_COLORS, dither=Image.Dither.NONE)
            idx = np.array(q, dtype=np.int64)
            pal_flat = np.array(q.getpalette() or [], dtype=np.uint8)
            if len(pal_flat) < N_COLORS * 3:
                pal_flat = np.concatenate(
                    [pal_flat, np.zeros(N_COLORS * 3 - len(pal_flat),
                                        dtype=np.uint8)])
            pal = pal_flat[:N_COLORS * 3].reshape(-1, 3)
            used = np.unique(idx)
            per_img_palettes.append(pal[used])

        # 2) merge + quantize to a single 16-color global palette
        combined = np.concatenate(per_img_palettes, axis=0)
        if combined.shape[0] <= N_COLORS:
            unique_colors = np.unique(combined, axis=0)
            gpal = np.zeros((N_COLORS, 3), dtype=np.uint8)
            gpal[:len(unique_colors)] = unique_colors
        else:
            combined_img = Image.fromarray(
                combined.reshape(1, -1, 3), "RGB")
            gq = combined_img.quantize(colors=N_COLORS,
                                       dither=Image.Dither.NONE)
            pal_flat = np.array(gq.getpalette() or [], dtype=np.uint8)
            if len(pal_flat) < N_COLORS * 3:
                pal_flat = np.concatenate(
                    [pal_flat, np.zeros(N_COLORS * 3 - len(pal_flat),
                                        dtype=np.uint8)])
            gpal = pal_flat[:N_COLORS * 3].reshape(-1, 3)

        # 3) map each image to global palette indices (nearest color)
        indexed_images = []
        for img in self.dataset_rgb:
            flat = img.reshape(-1, 3).astype(np.int32)
            diff = flat[:, None, :] - gpal[None, :, :].astype(np.int32)
            dist = (diff * diff).sum(axis=2)
            idx = dist.argmin(axis=1).astype(np.int64).reshape(img.shape[:2])
            indexed_images.append(idx)

        # 4) reorder palette so index 0 = most common globally
        all_flat = np.concatenate([i.flatten() for i in indexed_images])
        counts = np.bincount(all_flat, minlength=N_COLORS)
        order = np.argsort(-counts)
        remap = np.zeros(N_COLORS, dtype=np.uint8)
        remap[order] = np.arange(N_COLORS, dtype=np.uint8)
        gpal = gpal[order]
        indexed_images = [remap[i].astype(np.int64) for i in indexed_images]

        # 5) extract patches
        all_X, all_y = [], []
        for idx in indexed_images:
            idx_t = torch.from_numpy(idx)
            all_X.append(extract_patches_torch(idx_t))
            all_y.append(idx_t.flatten())

        train_X = torch.cat(all_X, dim=0)
        train_y = torch.cat(all_y, dim=0)
        total = train_X.shape[0]
        if total > MAX_PATCHES:
            sel = torch.randperm(total)[:MAX_PATCHES]
            train_X = train_X[sel]
            train_y = train_y[sel]
            self.log(f"  subsampled {MAX_PATCHES:,} / {total:,} patches")

        self.train_X = train_X.to(torch.uint8)
        self.train_y = train_y.to(torch.long)
        self.palette = gpal
        self.indexed_images = indexed_images

        self.update_swatches()
        self.board[:] = 0
        self.redraw()

        self.log(f"  {len(self.dataset_rgb)} image(s), {self.train_X.shape[0]:,} "
                 f"patches, palette set  ({time.time() - t0:.2f}s)")
        return True

    def start_training(self):
        if self.worker is not None and self.worker.is_alive():
            return
        if not self.prepare_dataset():
            return
        try:
            iters = int(self.iters_var.get())
            batch = int(self.batch_var.get())
            lr = float(self.lr_var.get())
            noise_pct = float(self.noise_var.get())
        except ValueError:
            self.log("!! Invalid hyper-parameter value.")
            return
        if iters <= 0 or batch <= 0 or lr <= 0:
            self.log("!! Values must be positive.")
            return
        if not (0.0 <= noise_pct <= 100.0):
            self.log("!! Noise % must be in [0, 100].")
            return
        noise_prob = noise_pct / 100.0

        self.stop_event.clear()
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.init_btn.config(state="disabled")
        self.worker = threading.Thread(target=self.train_worker,
                                       args=(iters, batch, lr, noise_prob),
                                       daemon=True)
        self.worker.start()

    def stop_training(self):
        self.stop_event.set()

    def train_worker(self, iters, batch, lr, noise_prob):
        model = self.model
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        X, y = self.train_X, self.train_y
        N = X.shape[0]
        bs = batch

        self.q.put(("log",
                    f"--- Training: iters={iters}, batch={bs}, lr={lr}, "
                    f"noise={noise_prob * 100:.1f}%, patches={N:,} ---"))
        t0 = time.time()
        for it in range(iters):
            if self.stop_event.is_set():
                self.q.put(("log", "--- Stopped by user. ---"))
                break

            idx = torch.randint(0, N, (bs,), device=self.device)
            xb = X[idx].to(self.device, dtype=torch.float32)
            yb = y[idx].to(self.device)
            if noise_prob > 0.0:
                xb = apply_context_noise(xb, N_COLORS, WINDOW, noise_prob)

            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

            if it % 10 == 0 or it == iters - 1:
                with torch.no_grad():
                    acc = (logits.argmax(1) == yb).float().mean().item()
                self.q.put(("log",
                            f"iter {it:6d}   loss {loss.item():.5f}   "
                            f"acc {acc * 100:6.2f}%"))
                self.q.put(("status",
                            f"Training… iter {it}/{iters}  "
                            f"loss {loss.item():.5f}  acc {acc * 100:.2f}%"))

        model.eval()
        self.q.put(("log", f"--- Done in {time.time() - t0:.2f}s "
                           f"({self.count_params():,} params) ---"))
        self.q.put(("status", "Finished."))
        self.q.put(("done", None))

    # ---------------------------------------------------------- inference tab
    def build_infer_tab(self):
        ctrl = ttk.Frame(self.tab_infer, padding=8)
        ctrl.pack(fill="x")
        ttk.Button(ctrl, text="Step ▶", command=self.step_model).pack(side="left")
        ttk.Button(ctrl, text="Step ×5",
                   command=lambda: self.step_model(5)).pack(side="left", padx=4)
        ttk.Button(ctrl, text="Clear",
                   command=self.clear_board).pack(side="left", padx=4)
        ttk.Button(ctrl, text="Random",
                   command=self.random_board).pack(side="left", padx=4)
        ttk.Button(ctrl, text="Load from dataset",
                   command=self.load_selected_image).pack(side="left", padx=4)
        self.load_img_var = tk.StringVar()
        self.load_img_combo = ttk.Combobox(
            ctrl, textvariable=self.load_img_var, values=[],
            state="readonly", width=26)
        self.load_img_combo.pack(side="left", padx=4)

        pal_frame = ttk.Frame(self.tab_infer, padding=8)
        pal_frame.pack(fill="x")
        ttk.Label(pal_frame, text="Paint color:").pack(side="left")
        self.swatches = []
        for i in range(N_COLORS):
            btn = tk.Button(pal_frame, width=2, relief="raised",
                            command=lambda idx=i: self.selected_color.set(idx))
            btn.pack(side="left", padx=1)
            self.swatches.append(btn)
        self.update_swatches()

        self.image_label = ttk.Label(self.tab_infer)
        self.image_label.pack(pady=8)
        self.image_label.bind("<Button-1>", lambda e: self.paint(e, "draw"))
        self.image_label.bind("<B1-Motion>", lambda e: self.paint(e, "draw"))
        self.image_label.bind("<Button-3>", lambda e: self.paint(e, "erase"))
        self.image_label.bind("<B3-Motion>", lambda e: self.paint(e, "erase"))

        ttk.Label(self.tab_infer,
                  text="Left click / drag = paint     "
                       "Right click / drag = erase").pack(pady=(0, 8))

    def update_image_combo(self):
        names = [info["name"] for info in self.dataset_info]
        self.load_img_combo["values"] = names
        if names and self.load_img_var.get() not in names:
            self.load_img_var.set(names[0])
        elif not names:
            self.load_img_var.set("")

    def update_swatches(self):
        for i, btn in enumerate(self.swatches):
            r, g, b = self.palette[i]
            btn.config(bg=f"#{r:02x}{g:02x}{b:02x}")

    def redraw(self):
        rgb = self.palette[self.board]
        img = Image.fromarray(rgb, "RGB").resize(
            (BOARD * CELL, BOARD * CELL), Image.NEAREST)
        self.photo = ImageTk.PhotoImage(img)
        self.image_label.config(image=self.photo)

    def paint(self, event, mode):
        r, c = event.y // CELL, event.x // CELL
        if 0 <= r < BOARD and 0 <= c < BOARD:
            new_val = self.selected_color.get() if mode == "draw" else 0
            if self.board[r, c] != new_val:
                self.board[r, c] = new_val
                self.redraw()

    def clear_board(self):
        self.board[:] = 0
        self.redraw()

    def random_board(self):
        self.board = np.random.randint(0, N_COLORS,
                                       (BOARD, BOARD), dtype=np.uint8)
        self.redraw()

    def load_selected_image(self):
        if self.indexed_images is None:
            if not self.prepare_dataset():
                return
        name = self.load_img_var.get()
        if not name:
            return
        for i, info in enumerate(self.dataset_info):
            if info["name"] == name:
                idx = self.indexed_images[i]
                h, w = idx.shape
                full = np.zeros((BOARD, BOARD), dtype=np.uint8)
                r0 = max(0, (BOARD - h) // 2)
                c0 = max(0, (BOARD - w) // 2)
                rh, rw = min(h, BOARD), min(w, BOARD)
                full[r0:r0 + rh, c0:c0 + rw] = idx[:rh, :rw].astype(np.uint8)
                self.board = full
                self.redraw()
                self.log(f"Loaded image #{i}: {name}")
                return

    def step_model(self, n=1):
        self.model.eval()
        for _ in range(n):
            img_t = torch.from_numpy(self.board.astype(np.int64)).to(self.device)
            p = extract_patches_torch(img_t)
            with torch.no_grad():
                preds = self.model(p).argmax(1).cpu().numpy().astype(np.uint8)
            self.board = preds.reshape(BOARD, BOARD)
        self.redraw()


if __name__ == "__main__":
    App().mainloop()