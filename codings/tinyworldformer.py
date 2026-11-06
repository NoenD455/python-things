import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import threading
import tkinter as tk
from tkinter import ttk, messagebox


# ============================================================
# Game: 16x16 torus, 3x3 player, TWO 3x3 blocks, push mechanic
# ============================================================
GRID = 16
SIZE = 3
VOCAB = 4            # 0=empty, 1=player, 2=block1 (orange), 3=block2 (red)
N_ACTIONS = 4
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]   # up, down, left, right


def rects_overlap(p, b, size=SIZE, n=GRID):
    """Two `size`-squares on a torus overlap?"""
    dy = (b[0] - p[0]) % n
    dx = (b[1] - p[1]) % n
    return (dy < size or dy > n - size) and (dx < size or dx > n - size)


def step_game(player, block1, block2, action_idx):
    """Deterministic transition with two blocks and chain pushing.

    If the player's new position overlaps a block, that block is pushed
    in the same direction. If a pushed block then overlaps the other
    (stationary) block, that one is pushed too. Torus wraps.
    """
    dy, dx = ACTIONS[action_idx]
    new_player = ((player[0] + dy) % GRID, (player[1] + dx) % GRID)

    hit1 = rects_overlap(new_player, block1)
    hit2 = rects_overlap(new_player, block2)

    if hit1 and hit2:
        new_b1 = ((block1[0] + dy) % GRID, (block1[1] + dx) % GRID)
        new_b2 = ((block2[0] + dy) % GRID, (block2[1] + dx) % GRID)
    elif hit1:
        new_b1 = ((block1[0] + dy) % GRID, (block1[1] + dx) % GRID)
        if rects_overlap(new_b1, block2):
            new_b2 = ((block2[0] + dy) % GRID, (block2[1] + dx) % GRID)
        else:
            new_b2 = block2
    elif hit2:
        new_b2 = ((block2[0] + dy) % GRID, (block2[1] + dx) % GRID)
        if rects_overlap(new_b2, block1):
            new_b1 = ((block1[0] + dy) % GRID, (block1[1] + dx) % GRID)
        else:
            new_b1 = block1
    else:
        new_b1, new_b2 = block1, block2

    return new_player, new_b1, new_b2


def render(player, block1, block2):
    f = np.zeros((GRID, GRID), dtype=np.uint8)
    for i in range(SIZE):
        for j in range(SIZE):
            f[(player[0] + i) % GRID, (player[1] + j) % GRID] = 1
    for i in range(SIZE):
        for j in range(SIZE):
            f[(block1[0] + i) % GRID, (block1[1] + j) % GRID] = 2
    for i in range(SIZE):
        for j in range(SIZE):
            f[(block2[0] + i) % GRID, (block2[1] + j) % GRID] = 3
    return f


def random_non_overlapping(rng):
    while True:
        p = (rng.randrange(GRID), rng.randrange(GRID))
        b1 = (rng.randrange(GRID), rng.randrange(GRID))
        b2 = (rng.randrange(GRID), rng.randrange(GRID))
        if (not rects_overlap(p, b1) and
            not rects_overlap(p, b2) and
            not rects_overlap(b1, b2)):
            return p, b1, b2


def gen_random_data(n_sequences, seed=0, n_step=2):
    """Generate n_sequences trajectories of n_step steps each.
    Returns frames (N, n_step+1, GRID, GRID) and actions (N, n_step)."""
    rng = random.Random(seed)
    frames = np.empty((n_sequences, n_step + 1, GRID, GRID), dtype=np.uint8)
    actions = np.empty((n_sequences, n_step), dtype=np.int64)
    for k in range(n_sequences):
        p, b1, b2 = random_non_overlapping(rng)
        frames[k, 0] = render(p, b1, b2)
        for t in range(n_step):
            a = rng.randrange(N_ACTIONS)
            actions[k, t] = a
            p, b1, b2 = step_game(p, b1, b2, a)
            frames[k, t + 1] = render(p, b1, b2)
    return (torch.from_numpy(frames),
            torch.from_numpy(actions))


# ============================================================
# Singleton decode (used only at inference)
# ============================================================
def decode_frame(logits_np):
    """Greedy singleton decode: place player, block1, block2 at best
    non-overlapping positions. logits_np: (VOCAB, GRID, GRID) numpy.
    Returns (GRID, GRID) uint8 frame."""
    probs = logits_np - logits_np.max(axis=0, keepdims=True)
    probs = np.exp(probs)
    probs /= probs.sum(axis=0, keepdims=True)

    out = np.zeros((GRID, GRID), dtype=np.uint8)
    occupied = np.zeros((GRID, GRID), dtype=bool)

    for cls in (1, 2, 3):        # player, block1, block2
        scores = np.zeros((GRID, GRID), dtype=np.float64)
        for di in range(SIZE):
            for dj in range(SIZE):
                scores += np.roll(np.roll(probs[cls], -di, axis=0), -dj, axis=1)

        valid = np.ones((GRID, GRID), dtype=bool)
        for di in range(SIZE):
            for dj in range(SIZE):
                valid &= ~np.roll(np.roll(occupied, -di, axis=0), -dj, axis=1)

        scores_masked = np.where(valid, scores, -np.inf)
        if not np.isfinite(scores_masked).any():
            scores_masked = scores
        idx = int(scores_masked.argmax())
        i, j = idx // GRID, idx % GRID

        for di in range(SIZE):
            for dj in range(SIZE):
                out[(i + di) % GRID, (j + dj) % GRID] = cls
                occupied[(i + di) % GRID, (j + dj) % GRID] = True

    return out


# ============================================================
# Model
# ============================================================
PATCH = 2
D = 64
HEADS = 4
FF = 128
LOOPS = 3
NP = GRID // PATCH
K = PATCH * PATCH


class PatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.cell_emb = nn.Embedding(K * VOCAB, D)
        self.pos_emb = nn.Embedding(NP * NP, D)
        self.act_emb = nn.Embedding(N_ACTIONS, D)

    def forward(self, frames, actions):
        B = frames.shape[0]
        x = frames.view(B, NP, PATCH, NP, PATCH).permute(0, 1, 3, 2, 4).reshape(B, NP * NP, K)
        offsets = torch.arange(K, device=frames.device) * VOCAB
        idx = x + offsets[None, None, :]
        emb = self.cell_emb(idx).sum(dim=2)
        emb = emb + self.pos_emb(torch.arange(NP * NP, device=frames.device))[None]
        emb = emb + self.act_emb(actions)[:, None, :]
        return emb


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(D)
        self.attn = nn.MultiheadAttention(D, HEADS, batch_first=True)
        self.norm2 = nn.LayerNorm(D)
        self.mlp = nn.Sequential(nn.Linear(D, FF), nn.ReLU(), nn.Linear(FF, D))

    def forward(self, x):
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x


class WorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = PatchEmbed()
        self.block = Block()
        self.head = nn.Linear(D, K * VOCAB)

    def forward(self, frames, actions):
        """Returns a list of per-loop logits (for intermediate supervision)."""
        x = self.embed(frames, actions)
        inp = x
        B = frames.shape[0]
        logits_list = []
        for _ in range(LOOPS):
            x = self.block(x + inp)
            lg = self.head(x)
            lg = lg.view(B, NP, NP, PATCH, PATCH, VOCAB)
            lg = lg.permute(0, 5, 1, 3, 2, 4).reshape(B, VOCAB, GRID, GRID)
            logits_list.append(lg)
        return logits_list


# ============================================================
# Training: weighted CE, per-loop supervision, input noise, 2-step rollout
# ============================================================
def train_model(model, data, steps, batch_size=128, lr=3e-4,
                progress_cb=None, stop_flag=None, lock=None,
                noise_p=0.03, rollout_weight=1.0):
    frames, actions = data
    N, n_step_plus1, _, _ = frames.shape
    n_step = n_step_plus1 - 1
    device = next(model.parameters()).device

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1))
    class_w = torch.tensor([1.0, 20.0, 20.0, 20.0], device=device)

    def weighted_ce(logits, target, cell_w):
        loss = F.cross_entropy(logits, target, reduction='none')
        return (loss * cell_w).sum() / cell_w.sum()

    for step in range(steps):
        if stop_flag is not None and stop_flag.is_set():
            break
        idx = torch.randint(0, N, (batch_size,))

        f_clean = frames[idx, 0].long().to(device)
        a0 = actions[idx, 0].to(device)
        f1 = frames[idx, 1].long().to(device)

        # Input noise -> robustness to own prediction errors
        if noise_p > 0:
            noise_mask = torch.rand(f_clean.shape, device=device) < noise_p
            noise_vals = torch.randint(0, VOCAB, f_clean.shape, device=device)
            f_in = torch.where(noise_mask, noise_vals, f_clean)
        else:
            f_in = f_clean

        # Weight: per-class + "cells that changed" boost
        change1 = (f1 != f_clean).float()
        cell_w1 = class_w[f1] * (1.0 + 4.0 * change1)

        # Step 1: supervise every loop output
        logits_list_1 = model(f_in, a0)
        loss = 0.0
        for lg in logits_list_1:
            loss = loss + weighted_ce(lg, f1, cell_w1)
        loss = loss / len(logits_list_1)

        # Step 2: rollout with scheduled sampling (half GT, half own argmax)
        if n_step >= 2:
            f2 = frames[idx, 2].long().to(device)
            a1 = actions[idx, 1].to(device)
            f1_pred = logits_list_1[-1].argmax(dim=1)
            use_gt = (torch.rand(batch_size, device=device) < 0.5).view(-1, 1, 1)
            f1_in = torch.where(use_gt, f1, f1_pred)
            change2 = (f2 != f1).float()
            cell_w2 = class_w[f2] * (1.0 + 4.0 * change2)
            logits_list_2 = model(f1_in, a1)
            loss2 = 0.0
            for lg in logits_list_2:
                loss2 = loss2 + weighted_ce(lg, f2, cell_w2)
            loss2 = loss2 / len(logits_list_2)
            loss = loss + rollout_weight * loss2

        if lock is not None:
            lock.acquire()
        try:
            model.train()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            loss_val = loss.item()
        finally:
            if lock is not None:
                lock.release()

        if progress_cb is not None and (step % 5 == 0 or step == steps - 1):
            progress_cb(step + 1, steps, loss_val)


# ============================================================
# GUI
# ============================================================
COLORS = ['#1a1a1a', '#4a90e2', '#e2a04a', '#e24a4a']  # empty, player, block1, block2


class App:
    def __init__(self, root):
        self.root = root
        root.title("Looped-Transformer World Model — 16x16 Block Pusher (2 blocks)")

        self.model = WorldModel()
        self.model_lock = threading.Lock()
        self.data = None
        self.train_thread = None
        self.stop_flag = threading.Event()
        self.rng = random.Random()
        self.auto_after_id = None

        # Recording state
        self.recording = False
        self.recorded_episodes = []    # list of {'frames': [arr, ...], 'actions': [int, ...]}
        self.current_episode = None

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True)

        self.train_tab = ttk.Frame(self.notebook)
        self.record_tab = ttk.Frame(self.notebook)
        self.infer_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.train_tab, text="Training")
        self.notebook.add(self.record_tab, text="Record")
        self.notebook.add(self.infer_tab, text="Inference")

        self.build_train_tab()
        self.build_infer_tab()   # creates real_canvas, dream_canvas
        self.build_record_tab()  # creates rec_canvas
        self.reset_inference()   # depends on all three canvases

        n_params = sum(p.numel() for p in self.model.parameters())
        self.log(f"Model: looped transformer | patch {PATCH}x{PATCH} -> {NP*NP} tokens | "
                 f"d={D}, heads={HEADS}, ff={FF}, loops={LOOPS}")
        self.log(f"Parameters: {n_params:,} | vocab={VOCAB} "
                 f"(empty, player, orange block, red block)")

    # ---------------- Training tab ----------------
    def build_train_tab(self):
        frame = self.train_tab

        ctrl = ttk.Frame(frame)
        ctrl.pack(fill='x', padx=10, pady=10)

        ttk.Label(ctrl, text="Random sequences (2-step):").pack(side='left')
        self.n_trans_var = tk.IntVar(value=100000)
        ttk.Entry(ctrl, textvariable=self.n_trans_var, width=9).pack(side='left', padx=5)

        ttk.Button(ctrl, text="Generate random data",
                   command=self.on_generate).pack(side='left', padx=5)

        ttk.Label(ctrl, text="Steps:").pack(side='left', padx=(20, 0))
        self.n_steps_var = tk.IntVar(value=5000)
        ttk.Entry(ctrl, textvariable=self.n_steps_var, width=8).pack(side='left', padx=5)

        ttk.Button(ctrl, text="Train", command=self.on_train).pack(side='left', padx=5)
        ttk.Button(ctrl, text="Stop", command=self.on_stop).pack(side='left', padx=5)

        self.loss_label = ttk.Label(frame, text="Loss: --",
                                    font=('TkDefaultFont', 11))
        self.loss_label.pack(pady=(10, 0))

        self.progress = ttk.Progressbar(frame, length=520, maximum=100)
        self.progress.pack(pady=5)

        prev_lf = ttk.LabelFrame(frame, text="Sample frame from generated data")
        prev_lf.pack(pady=10)
        self.preview = tk.Canvas(prev_lf, width=GRID * 16, height=GRID * 16,
                                 bg='black', highlightthickness=0)
        self.preview.pack()

        log_lf = ttk.LabelFrame(frame, text="Log")
        log_lf.pack(fill='both', expand=True, padx=10, pady=10)
        self.log_text = tk.Text(log_lf, height=8)
        self.log_text.pack(fill='both', expand=True)

        self.log("Welcome. 1) Generate random data or record some episodes. "
                 "2) Train. 3) Open the Inference tab.")

    def log(self, s):
        self.log_text.insert('end', s + '\n')
        self.log_text.see('end')

    def show_preview(self, frame):
        self.preview.delete('all')
        cs = 16
        for i in range(GRID):
            for j in range(GRID):
                self.preview.create_rectangle(
                    j * cs, i * cs, (j + 1) * cs, (i + 1) * cs,
                    fill=COLORS[int(frame[i, j])], outline='')

    def on_generate(self):
        n = int(self.n_trans_var.get())
        self.log(f"Generating {n:,} 2-step sequences...")
        self.data = gen_random_data(n)
        self.log(f"Done: {self.data[0].shape[0]:,} sequences, "
                 f"{self.data[0].shape[1]} frames each.")
        self.show_preview(self.data[0][0, 0].numpy())

    def _recorded_to_2step(self):
        """Convert banked episodes into 2-step training windows."""
        if not self.recorded_episodes:
            return None
        frames_list = []
        actions_list = []
        for ep in self.recorded_episodes:
            ef = ep['frames']
            ea = ep['actions']
            for t in range(len(ea) - 1):
                frames_list.append(np.stack([ef[t], ef[t + 1], ef[t + 2]]))
                actions_list.append([ea[t], ea[t + 1]])
        if not frames_list:
            return None
        return (torch.from_numpy(np.stack(frames_list)),
                torch.from_numpy(np.array(actions_list, dtype=np.int64)))

    def on_train(self):
        if self.train_thread is not None and self.train_thread.is_alive():
            self.log("Training already running.")
            return

        data = self.data
        rec = self._recorded_to_2step()
        if rec is not None:
            if data is None:
                data = rec
                self.log(f"Training on {rec[0].shape[0]:,} recorded 2-step samples only.")
            else:
                n_rand = data[0].shape[0]
                n_rec = rec[0].shape[0]
                data = (torch.cat([data[0], rec[0]], dim=0),
                        torch.cat([data[1], rec[1]], dim=0))
                self.log(f"Combining random ({n_rand:,}) + recorded ({n_rec:,}) "
                         f"= {data[0].shape[0]:,} samples.")
        if data is None:
            self.log("Generate data or record some sequences first.")
            return

        self.stop_flag.clear()
        steps = int(self.n_steps_var.get())
        self.log(f"Training for {steps} steps (2-step rollout + input noise)...")

        def run():
            def cb(step, total, loss):
                self.root.after(0, self.update_progress, step, total, loss)
            train_model(self.model, data, steps,
                        progress_cb=cb, stop_flag=self.stop_flag,
                        lock=self.model_lock)
            self.root.after(0, lambda: self.log("Training finished."))

        self.train_thread = threading.Thread(target=run, daemon=True)
        self.train_thread.start()

    def update_progress(self, step, total, loss):
        self.loss_label.config(text=f"Loss: {loss:.4f}   (step {step}/{total})")
        self.progress['value'] = 100.0 * step / total

    def on_stop(self):
        self.stop_flag.set()
        self.log("Stop requested.")

    # ---------------- Inference tab ----------------
    def build_infer_tab(self):
        frame = self.infer_tab

        canvases = ttk.Frame(frame)
        canvases.pack(pady=10)

        real_lf = ttk.LabelFrame(canvases, text="Real game")
        real_lf.grid(row=0, column=0, padx=10)
        self.real_canvas = tk.Canvas(real_lf, width=GRID * 24, height=GRID * 24,
                                     bg='black', highlightthickness=0)
        self.real_canvas.pack()

        dream_lf = ttk.LabelFrame(canvases, text="Dream (world model)")
        dream_lf.grid(row=0, column=1, padx=10)
        self.dream_canvas = tk.Canvas(dream_lf, width=GRID * 24, height=GRID * 24,
                                      bg='black', highlightthickness=0)
        self.dream_canvas.pack()

        ctrl = ttk.Frame(frame)
        ctrl.pack(pady=10)

        ttk.Button(ctrl, text="↑", width=4,
                   command=lambda: self.do_action(0)).grid(row=0, column=1, padx=2, pady=2)
        ttk.Button(ctrl, text="←", width=4,
                   command=lambda: self.do_action(2)).grid(row=1, column=0, padx=2, pady=2)
        ttk.Button(ctrl, text="↓", width=4,
                   command=lambda: self.do_action(1)).grid(row=1, column=1, padx=2, pady=2)
        ttk.Button(ctrl, text="→", width=4,
                   command=lambda: self.do_action(3)).grid(row=1, column=2, padx=2, pady=2)

        ttk.Button(ctrl, text="Random step",
                   command=self.random_step).grid(row=2, column=0, columnspan=3,
                                                  pady=5, sticky='ew')
        ttk.Button(ctrl, text="Reset (new random start)",
                   command=self.reset_inference).grid(row=3, column=0, columnspan=3,
                                                      pady=2, sticky='ew')

        self.auto_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(ctrl, text="Auto random step (300 ms)",
                        variable=self.auto_var,
                        command=self.schedule_auto).grid(row=4, column=0, columnspan=3,
                                                         pady=5)

        self.status_label = ttk.Label(frame, text="", font=('TkDefaultFont', 11))
        self.status_label.pack(pady=5)

    # ---------------- Record tab ----------------
    def build_record_tab(self):
        frame = self.record_tab

        canvas_frame = ttk.Frame(frame)
        canvas_frame.pack(pady=10)
        rec_lf = ttk.LabelFrame(canvas_frame, text="Real game — play to record")
        rec_lf.pack()
        self.rec_canvas = tk.Canvas(rec_lf, width=GRID * 24, height=GRID * 24,
                                    bg='black', highlightthickness=0)
        self.rec_canvas.pack()

        ctrl = ttk.Frame(frame)
        ctrl.pack(pady=10)

        ttk.Button(ctrl, text="↑", width=4,
                   command=lambda: self._record_action(0)
                   ).grid(row=0, column=1, padx=2, pady=2)
        ttk.Button(ctrl, text="←", width=4,
                   command=lambda: self._record_action(2)
                   ).grid(row=1, column=0, padx=2, pady=2)
        ttk.Button(ctrl, text="↓", width=4,
                   command=lambda: self._record_action(1)
                   ).grid(row=1, column=1, padx=2, pady=2)
        ttk.Button(ctrl, text="→", width=4,
                   command=lambda: self._record_action(3)
                   ).grid(row=1, column=2, padx=2, pady=2)

        ttk.Button(ctrl, text="Random step",
                   command=lambda: self._record_action(self.rng.randrange(N_ACTIONS))
                   ).grid(row=2, column=0, columnspan=3, pady=5, sticky='ew')
        ttk.Button(ctrl, text="Reset (new random start)",
                   command=self.reset_inference
                   ).grid(row=3, column=0, columnspan=3, pady=2, sticky='ew')

        rec_ctrl = ttk.Frame(frame)
        rec_ctrl.pack(pady=10)
        self.rec_toggle_btn = ttk.Button(rec_ctrl, text="Start recording",
                                         command=self.on_rec_toggle)
        self.rec_toggle_btn.pack(side='left', padx=5)
        ttk.Button(rec_ctrl, text="Discard recordings",
                   command=self.on_rec_discard).pack(side='left', padx=5)

        self.rec_status_label = ttk.Label(frame, text="", font=('TkDefaultFont', 11))
        self.rec_status_label.pack(pady=5)

        hint = ttk.Label(
            frame,
            text=("Press 'Start recording', then play. Reset at any time to start a new episode.\n"
                  "Stop recording to bank the current episode. Recorded data is merged into "
                  "training."),
            justify='center', foreground='#666')
        hint.pack(pady=(5, 5))

        self._update_rec_status()

    def _update_rec_status(self):
        total = sum(len(ep['actions']) for ep in self.recorded_episodes)
        cur = 0
        if self.recording and self.current_episode is not None:
            cur = len(self.current_episode['actions'])
        text = (f"Banked episodes: {len(self.recorded_episodes)}  |  "
                f"banked transitions: {total}")
        if self.recording:
            text += f"  |  RECORDING (current episode: {cur} steps)"
        self.rec_status_label.config(text=text)

    def on_rec_toggle(self):
        if self.recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        self.recording = True
        self.current_episode = {'frames': [self.real_frame.copy()], 'actions': []}
        self.rec_toggle_btn.config(text="Stop recording")
        self._update_rec_status()
        self.log("Recording started.")

    def _stop_recording(self):
        self.recording = False
        if self.current_episode is not None and len(self.current_episode['actions']) > 0:
            self.recorded_episodes.append(self.current_episode)
            self.log(f"Banked episode ({len(self.current_episode['actions'])} steps). "
                     f"Total episodes: {len(self.recorded_episodes)}.")
        else:
            self.log("Stopped recording (no steps captured).")
        self.current_episode = None
        self.rec_toggle_btn.config(text="Start recording")
        self._update_rec_status()

    def on_rec_discard(self):
        if not self.recorded_episodes and not (self.recording and self.current_episode
                                                and self.current_episode['actions']):
            return
        if not messagebox.askyesno(
                "Discard recordings",
                f"Discard {len(self.recorded_episodes)} banked episode(s)? "
                f"This cannot be undone."):
            return
        self.recorded_episodes = []
        if self.recording:
            self.current_episode = {'frames': [self.real_frame.copy()], 'actions': []}
        self._update_rec_status()
        self.log("Recordings discarded.")

    def _record_action(self, action_idx):
        self._advance_real(action_idx)

    # ---------------- Shared real-game step ----------------
    def _advance_real(self, action_idx):
        """Update real game state; capture the transition if recording."""
        np_, nb1, nb2 = step_game(self.real_player, self.real_block1,
                                  self.real_block2, action_idx)
        self.real_player, self.real_block1, self.real_block2 = np_, nb1, nb2
        self.real_frame = render(np_, nb1, nb2)

        if self.recording and self.current_episode is not None:
            self.current_episode['frames'].append(self.real_frame.copy())
            self.current_episode['actions'].append(action_idx)
            self._update_rec_status()

        self.draw(self.real_canvas, self.real_frame)
        self.draw(self.rec_canvas, self.real_frame)

    # ---------------- Reset (used by both Record and Inference) ----------------
    def reset_inference(self):
        # If recording, bank the current episode and start a new one.
        if (self.recording and self.current_episode is not None
                and len(self.current_episode['actions']) > 0):
            self.recorded_episodes.append(self.current_episode)
            self.log(f"Banked episode ({len(self.current_episode['actions'])} steps) on reset. "
                     f"Total episodes: {len(self.recorded_episodes)}.")

        p, b1, b2 = random_non_overlapping(self.rng)
        self.real_player, self.real_block1, self.real_block2 = p, b1, b2
        self.real_frame = render(p, b1, b2)

        if self.recording:
            self.current_episode = {'frames': [self.real_frame.copy()], 'actions': []}

        self.dream_frame = self.real_frame.copy()
        self.dream_frame_t = torch.tensor(self.dream_frame[None], dtype=torch.long)

        self.draw(self.real_canvas, self.real_frame)
        self.draw(self.rec_canvas, self.real_frame)
        self.draw(self.dream_canvas, self.dream_frame)
        self.status_label.config(text="Reset: real and dream start from the same frame.")
        self._update_rec_status()

    def draw(self, canvas, frame):
        canvas.delete('all')
        cs = 24
        for i in range(GRID):
            for j in range(GRID):
                canvas.create_rectangle(
                    j * cs, i * cs, (j + 1) * cs, (i + 1) * cs,
                    fill=COLORS[int(frame[i, j])], outline='')

    # ---------------- Inference action ----------------
    def do_action(self, action_idx):
        # Advances the real game (also captures a transition if recording).
        self._advance_real(action_idx)

        # Dream: feed model its own previous (decoded) frame.
        with self.model_lock:
            self.model.eval()
            with torch.no_grad():
                logits_list = self.model(
                    self.dream_frame_t,
                    torch.tensor([action_idx], dtype=torch.long))
                logits_np = logits_list[-1][0].cpu().numpy()
        pred = decode_frame(logits_np)
        self.dream_frame = pred
        self.dream_frame_t = torch.tensor(pred[None], dtype=torch.long)
        self.draw(self.dream_canvas, self.dream_frame)

        match = float((self.real_frame == self.dream_frame).mean()) * 100.0
        self.status_label.config(
            text=f"Cell match: {match:5.1f}%   |   action: "
                 f"{['up','down','left','right'][action_idx]}")

    def random_step(self):
        self.do_action(self.rng.randrange(N_ACTIONS))

    def schedule_auto(self):
        if self.auto_after_id is not None:
            try:
                self.root.after_cancel(self.auto_after_id)
            except Exception:
                pass
            self.auto_after_id = None
        if self.auto_var.get():
            self.random_step()
            self.auto_after_id = self.root.after(300, self.schedule_auto)


def main():
    root = tk.Tk()
    root.geometry("1080x860")
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()