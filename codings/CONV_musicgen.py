import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os
import math
import time
import threading
import queue
from collections import Counter, defaultdict
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import traceback

try:
    import mido
except ImportError:
    raise SystemExit("Needs 'mido'.  Install with:  pip install mido")

# ==================== CPU OPTIMIZATION ====================
torch.set_num_threads(2)
torch.set_num_interop_threads(2)
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'

print(f"PyTorch threads: {torch.get_num_threads()}")


# ==================== MIDO LENIENCY ====================
def _patch_mido_lenient():
    """Tolerate malformed meta messages in MIDI files."""
    try:
        from mido.midifiles import meta as _meta
    except Exception:
        return

    def _wrap(name, default):
        orig = getattr(_meta, name, None)
        if orig is None:
            return
        def safe(*args, **kwargs):
            try:
                return orig(*args, **kwargs)
            except Exception:
                return default
        setattr(_meta, name, safe)

    _wrap("decode_key", ("C", 0))
    _wrap("decode_time_signature", (4, 4))
    _wrap("decode_smpte_offset", (0, 0, 0, 0, 0))
    _wrap("decode_smpte_frame_rate", 24)


_patch_mido_lenient()


# ==================== MIDI CONSTANTS / CONVERSION ====================
TPB = 480
QUANT = TPB // 4
MAX_DELTA = 32
MAX_DUR = 16
MAX_PITCH = 108
MIN_PITCH = 24
FILE_SEP = "\n"

MAX_GEN_LENGTH = 1_000_000   # chars

NOTE_PC = ['c', '^c', 'd', '^d', 'e', 'f', '^f', 'g', '^g', 'a', '^a', 'b']


def pitch_to_text(p):
    if not (0 <= p <= 127):
        return ""
    name = NOTE_PC[p % 12]
    if p >= 60:
        return name + "'" * ((p - 60) // 12)
    upper = ("^" + name[1].upper()) if name.startswith("^") else name.upper()
    return upper + "," * ((59 - p) // 12)


def extract_programs(path):
    progs = {}
    try:
        mid = mido.MidiFile(path, clip=True)
    except Exception:
        return progs
    for track in mid.tracks:
        for msg in track:
            if msg.type == "program_change":
                progs[msg.channel] = msg.program
    return progs


def midi_to_text(path, multi_track=True):
    try:
        mid = mido.MidiFile(path, clip=True)
    except Exception as exc:
        raise RuntimeError(f"could not read MIDI: {exc}") from exc

    scale = TPB / float(mid.ticks_per_beat)

    bpm = 120.0
    for track in mid.tracks:
        for msg in track:
            if msg.type == "set_tempo":
                bpm = mido.tempo2bpm(msg.tempo)
                break
        else:
            continue
        break

    notes = []   # (start, pitch, dur, vel, channel)
    for track in mid.tracks:
        t = 0
        pending = defaultdict(list)
        for msg in track:
            t += msg.time
            ch = getattr(msg, "channel", 0)
            if msg.type == "note_on" and msg.velocity > 0:
                if not multi_track and ch == 9:
                    continue
                pending[msg.note].append((t, msg.velocity, ch))
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                if not multi_track and ch == 9:
                    continue
                if pending[msg.note]:
                    start, vel, _ch = pending[msg.note].pop(0)
                    notes.append((start, msg.note, max(1, t - start), vel, ch))

    if not notes:
        return "", bpm

    quant = []
    for start, pitch, dur, vel, ch in notes:
        s = int(round(start * scale / QUANT))
        d = max(1, min(MAX_DUR, int(round(dur * scale / QUANT))))
        pitch = max(0, min(127, int(pitch)))
        vel = max(1, min(127, int(vel)))
        quant.append((s, pitch, d, vel, ch))
    quant.sort(key=lambda n: (n[0], n[1]))

    events = []
    i, n = 0, len(quant)
    while i < n:
        s = quant[i][0]
        ch = quant[i][4]
        group = []
        while i < n and quant[i][0] == s and quant[i][4] == ch:
            group.append(quant[i])
            i += 1
        pitches = sorted({g[1] for g in group})
        dur = Counter(g[2] for g in group).most_common(1)[0][0]
        vel = sum(g[3] for g in group) / len(group)
        events.append((s, pitches, dur, vel, ch))

    events.sort(key=lambda e: (e[0], e[4]))

    parts = []
    prev_onset = 0
    for onset, pitches, dur, vel, ch in events:
        delta = max(0, min(MAX_DELTA, onset - prev_onset))
        if delta > 0:
            parts.append("z" if delta == 1 else f"z{delta}")

        vel_mark = ">" if vel >= 100 else ("<" if vel < 50 else "")

        if len(pitches) == 1:
            note_str = pitch_to_text(pitches[0])
        else:
            note_str = "[" + "".join(pitch_to_text(p) for p in pitches) + "]"
        dur_str = "" if dur == 1 else str(dur)

        if multi_track:
            parts.append(f"{vel_mark}@{ch}{note_str}{dur_str}")
        else:
            parts.append(f"{vel_mark}{note_str}{dur_str}")

        prev_onset = onset

    return " ".join(parts), bpm


def _read_number(text, i):
    j = i
    limit = i + 3
    while j < len(text) and j < limit and text[j].isdigit():
        j += 1
    if j > i:
        return int(text[i:j]), j
    return 0, i


def _read_note(text, i):
    n = len(text)
    accidental = 0
    while i < n and text[i] in "^_":
        accidental += 1 if text[i] == "^" else -1
        i += 1
    if i >= n:
        return None, i
    letter = text[i]
    if letter.lower() not in "abcdefg":
        return None, i
    upper = letter.isupper()
    pc = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}[letter.lower()] + accidental
    pc %= 12
    i += 1
    if upper:
        commas = 0
        while i < n and text[i] == ",":
            commas += 1
            i += 1
        base = 48 - 12 * commas
    else:
        apostrophes = 0
        while i < n and text[i] == "'":
            apostrophes += 1
            i += 1
        base = 60 + 12 * apostrophes
    pitch = base + pc
    while pitch < MIN_PITCH:
        pitch += 12
    while pitch > MAX_PITCH:
        pitch -= 12
    return pitch, i


def _read_chord(text, i):
    pitches = []
    n = len(text)
    end = min(n, i + 40)
    while i < end and text[i] != "]":
        if text[i].lower() in "abcdefg" or text[i] in "^_":
            p, j = _read_note(text, i)
            if p is not None:
                pitches.append(p)
                i = j
                continue
        i += 1
    if i < n and text[i] == "]":
        i += 1
    return pitches, i


def text_to_events(text):
    multi = "@" in text
    events = []
    i, n = 0, len(text)
    t = 0
    pending_vel = 0
    pending_track = 0

    while i < n:
        c = text[i]
        if c in " \t\n|":
            i += 1
            continue
        if c == ">":
            pending_vel = 1
            i += 1
            continue
        if c == "<":
            pending_vel = -1
            i += 1
            continue
        if c == "@":
            j = i + 1
            num = 0
            digits = 0
            while j < n and text[j].isdigit() and digits < 2:
                num = num * 10 + int(text[j])
                j += 1
                digits += 1
            pending_track = num % 16
            i = j
            continue

        if c.lower() == "z":
            dur, i = _read_number(text, i + 1)
            t += min(MAX_DELTA, dur if dur > 0 else 1)
            pending_vel = 0
            pending_track = 0
            continue

        if c == "[":
            pitches, j = _read_chord(text, i + 1)
            if pitches:
                dur, k = _read_number(text, j)
                dur = min(MAX_DUR, dur if dur > 0 else 1)
                events.append((t, tuple(pitches), dur, pending_vel, pending_track))
                if not multi:
                    t += dur
                i = k
                pending_vel = 0
                pending_track = 0
            else:
                i += 1
            continue

        if c.lower() in "abcdefg" or c in "^_":
            pitch, j = _read_note(text, i)
            if pitch is not None:
                dur, k = _read_number(text, j)
                dur = min(MAX_DUR, dur if dur > 0 else 1)
                events.append((t, (pitch,), dur, pending_vel, pending_track))
                if not multi:
                    t += dur
                i = k
                pending_vel = 0
                pending_track = 0
            else:
                i += 1
            continue

        i += 1

    return events


def events_to_midi(events, path, tempo=120.0, programs=None, tpb=TPB):
    mid = mido.MidiFile(ticks_per_beat=tpb)

    by_track = defaultdict(list)
    for onset, pitches, dur, vlevel, ch in events:
        by_track[ch % 16].append((onset, pitches, dur, vlevel))

    if not by_track:
        empty = mido.MidiTrack()
        mid.tracks.append(empty)
        empty.append(mido.MetaMessage("set_tempo",
                                      tempo=mido.bpm2tempo(tempo), time=0))
        mid.save(path)
        return

    first = True
    for track_id in sorted(by_track.keys()):
        track = mido.MidiTrack()
        mid.tracks.append(track)

        if first:
            track.append(mido.MetaMessage("set_tempo",
                                          tempo=mido.bpm2tempo(tempo), time=0))
            first = False

        if track_id != 9:
            prog = 0
            if programs and track_id in programs:
                prog = programs[track_id]
            track.append(mido.Message("program_change", channel=track_id,
                                      program=max(0, min(127, int(prog))),
                                      time=0))

        abs_events = []
        for onset, pitches, dur, vlevel in by_track[track_id]:
            tick = onset * QUANT
            vel = 100 if vlevel > 0 else (45 if vlevel < 0 else 80)
            vel = max(1, min(127, int(vel)))
            for p in pitches:
                p = max(0, min(127, int(p)))
                abs_events.append((tick, 1, p, vel))
                abs_events.append((tick + dur * QUANT, 0, p, 0))
        abs_events.sort(key=lambda e: (e[0], e[1]))

        last = 0
        for tick, prio, pitch, vel in abs_events:
            d = tick - last
            last = tick
            if prio:
                track.append(mido.Message("note_on", note=pitch, velocity=vel,
                                          channel=track_id, time=d))
            else:
                track.append(mido.Message("note_off", note=pitch, velocity=0,
                                          channel=track_id, time=d))

    mid.save(path)


# ==================== RF / LAYER MATH ====================
def compute_layers_for_rf(target_rf, kernel_size):
    if kernel_size < 2:
        raise ValueError("kernel_size must be >= 2")
    if target_rf < 2:
        target_rf = 2
    inner = (target_rf - 1) / (kernel_size - 1) + 1
    n = max(1, math.ceil(math.log2(inner)))
    actual_rf = 1 + (kernel_size - 1) * (2 ** n - 1)
    return n, actual_rf


def snap_to_odd(n):
    n = int(n)
    if n < 1:
        return 1
    if n % 2 == 0:
        return n - 1 if n > 1 else 1
    return n


# ==================== PROGRESSIVE UI UPDATE SCHEDULE ====================
def update_interval(n_chars):
    """How many generated characters between UI refresh events.

    - n < 2000: every 10 chars
    - n >= 2000: doubles each time we cross the next power-of-2 boundary:
        [2000, 4000) -> 20
        [4000, 8000) -> 40
        [8000,16000) -> 80
        ...
    """
    if n_chars < 2000:
        return 10
    interval = 20
    threshold = 2000
    while n_chars >= threshold * 2:
        threshold *= 2
        interval *= 2
    return interval


# ==================== DILATED CONV BLOCK ====================
class DilatedConvBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.norm = nn.LayerNorm(channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        pad = (self.kernel_size - 1) * self.dilation
        x_pad = F.pad(x, (pad, 0))
        out = self.conv(x_pad)
        out = out.transpose(1, 2)
        out = self.norm(out)
        out = self.act(out)
        out = self.dropout(out)
        out = out.transpose(1, 2)
        return x + out


# ==================== DILATED CONV LM ====================
class DilatedConvLM(nn.Module):
    def __init__(self, vocab_size, embed_size=128, hidden_size=256,
                 n_layers=6, kernel_size=3, context_size=100, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.context_size = context_size

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.input_proj = nn.Linear(embed_size, hidden_size)

        self.blocks = nn.ModuleList([
            DilatedConvBlock(hidden_size, kernel_size, dilation=2 ** i, dropout=dropout)
            for i in range(n_layers)
        ])

        self.norm_out = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)

        self.receptive_field = 1 + (kernel_size - 1) * (2 ** n_layers - 1)

        self.init_weights()
        print(f"DilatedConvLM: {sum(p.numel() for p in self.parameters()):,} params, "
              f"k={kernel_size}, layers={n_layers}, RF={self.receptive_field}, "
              f"context={context_size}")

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, x):
        emb = self.embedding(x)
        h = self.input_proj(emb)
        h = h.transpose(1, 2)

        for block in self.blocks:
            h = block(h)

        h = h.transpose(1, 2)
        h = self.norm_out(h)
        logits = self.fc(h)
        return logits

    def count_params(self):
        breakdown = {
            'embedding':   self.embedding.weight.numel(),
            'input_proj':  sum(p.numel() for p in self.input_proj.parameters()),
            'conv_blocks': sum(p.numel() for b in self.blocks for p in b.parameters()),
            'norm_out':    sum(p.numel() for p in self.norm_out.parameters()),
            'fc':          sum(p.numel() for p in self.fc.parameters()),
        }
        breakdown['TOTAL'] = sum(breakdown.values())
        return breakdown

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'vocab_size': self.vocab_size,
            'embed_size': self.embed_size,
            'hidden_size': self.hidden_size,
            'n_layers': self.n_layers,
            'kernel_size': self.kernel_size,
            'context_size': self.context_size,
            'receptive_field': self.receptive_field,
        }, path)
        print(f"Model saved to {path}")

    @classmethod
    def load_model(cls, path):
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        model = cls(
            vocab_size=checkpoint['vocab_size'],
            embed_size=checkpoint['embed_size'],
            hidden_size=checkpoint['hidden_size'],
            n_layers=checkpoint['n_layers'],
            kernel_size=checkpoint.get('kernel_size', 3),
            context_size=checkpoint.get('context_size', 100),
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"Model loaded from {path}")
        return model


# ==================== FIXED VOCABULARY (MUSIC-TEXT ONLY) ====================
class TextProcessor:
    """Character-level processor with a small, fixed, music-text-only vocab."""

    def __init__(self, seq_len=100):
        self.seq_len = seq_len
        self.char2idx, self.idx2char = self._build_fixed_vocab()
        self.tokens = []

    def _build_fixed_vocab(self):
        specials = ['<PAD>', '<UNK>']

        # Only characters the music-text format actually uses.
        #   cdefgab / CDEFGAB : note letters (lowercase = octave 4+,
        #                       uppercase = octave 3 and below)
        #   z                 : rest
        #   ^ _               : accidentals
        #   ' ,               : octave shifts
        #   [ ]               : chord delimiters
        #   0-9               : durations / rest lengths / channel numbers
        #   > <               : accent / soft (velocity markers)
        #   @                 : channel prefix
        #   space \t \n \r |  : separators / bar lines
        allowed_chars = (
            "abcdefgz"
            "ABCDEFG"
            "^_"
            "',"
            "[]"
            "0123456789"
            "><@"
            " \t\n\r|"
        )
        allowed_chars = sorted(set(allowed_chars))

        char2idx, idx2char = {}, {}
        for idx, token in enumerate(specials):
            char2idx[token] = idx
            idx2char[idx] = token
        for idx, ch in enumerate(allowed_chars, start=len(specials)):
            char2idx[ch] = idx
            idx2char[idx] = ch

        return char2idx, idx2char

    def text_to_tokens(self, text):
        return [self.char2idx.get(c, 1) for c in text]

    def tokens_to_text(self, tokens):
        result = []
        for idx in tokens:
            ch = self.idx2char.get(idx, '')
            if ch and ch not in ('<PAD>', '<UNK>'):
                result.append(ch)
        return ''.join(result)

    def get_batch(self, batch_size):
        if len(self.tokens) <= self.seq_len + 1:
            seq = self.tokens + [0] * (self.seq_len + 1 - len(self.tokens))
            batch_x = [seq[:self.seq_len]] * batch_size
            batch_y = [seq[1:self.seq_len + 1]] * batch_size
        else:
            batch_x, batch_y = [], []
            for _ in range(batch_size):
                start = random.randint(0, len(self.tokens) - self.seq_len - 1)
                batch_x.append(self.tokens[start:start + self.seq_len])
                batch_y.append(self.tokens[start + 1:start + self.seq_len + 1])

        return torch.tensor(batch_x), torch.tensor(batch_y)


# ==================== GUI ====================
class MultiFileMusicGenerator:
    def __init__(self, root):
        self.root = root
        self.root.title("Music Pattern Learner - Dilated Conv (Fixed Window)")
        self.root.geometry("1000x850")

        self.model = None
        self.processor = TextProcessor()
        self.training = False
        self.generating = False
        self.device = torch.device('cpu')
        self.loaded_files = []       # MIDI paths
        self.loaded_texts = []       # corresponding music-text representations
        self.programs = {}           # channel -> program (from MIDI files)
        self.tempos = []             # bpm per loaded file

        self.generation_queue = queue.Queue()
        self.update_queue = queue.Queue()

        self.setup_ui()
        threading.Thread(target=self.update_ui_thread, daemon=True).start()

        print("Ready! Load multiple MIDI files to start.")
        print(f"Fixed vocabulary size: {len(self.processor.char2idx)}")

    def setup_ui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)

        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill='both', expand=True)

        self.setup_file_tab(notebook)
        self.setup_train_tab(notebook)
        self.setup_generate_tab(notebook)

        self.status = tk.StringVar(value="Ready - Load MIDI files to begin")
        status_bar = ttk.Label(main_frame, textvariable=self.status, relief='sunken')
        status_bar.pack(fill='x', pady=(5, 0))

    def setup_file_tab(self, notebook):
        frame = ttk.Frame(notebook)
        notebook.add(frame, text='Files')

        ttk.Label(frame, text="MIDI File Management", font='Arial 12 bold').pack(pady=(10, 5))

        list_frame = ttk.LabelFrame(frame, text="Loaded Files", padding=10)
        list_frame.pack(fill='both', expand=True, padx=10, pady=10)

        list_container = ttk.Frame(list_frame)
        list_container.pack(fill='both', expand=True)

        columns = ('#', 'Filename', 'Size', 'Chars', 'Status')
        self.file_tree = ttk.Treeview(list_container, columns=columns, show='headings', height=10)

        for col, txt, w, anchor in [
            ('#', '#', 40, 'center'),
            ('Filename', 'Filename', 300, 'w'),
            ('Size', 'Size', 80, 'center'),
            ('Chars', 'Chars', 80, 'center'),
            ('Status', 'Status', 100, 'center'),
        ]:
            self.file_tree.heading(col, text=txt)
            self.file_tree.column(col, width=w, anchor=anchor)

        scrollbar = ttk.Scrollbar(list_container, orient='vertical', command=self.file_tree.yview)
        self.file_tree.configure(yscrollcommand=scrollbar.set)
        self.file_tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        control_frame = ttk.Frame(list_frame)
        control_frame.pack(fill='x', pady=(10, 0))
        ttk.Button(control_frame, text="Add Files", command=self.add_files).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Add Folder", command=self.add_folder).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Remove Selected", command=self.remove_file).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Clear All", command=self.clear_files).pack(side='left', padx=5)

        preview_frame = ttk.LabelFrame(frame, text="Music-Text Preview", padding=10)
        preview_frame.pack(fill='both', expand=True, padx=10, pady=10)

        preview_top = ttk.Frame(preview_frame)
        preview_top.pack(fill='x', pady=(0, 5))
        ttk.Label(preview_top, text="Preview file:").pack(side='left', padx=5)
        self.preview_var = tk.StringVar()
        self.preview_combo = ttk.Combobox(preview_top, textvariable=self.preview_var, state='readonly', width=50)
        self.preview_combo.pack(side='left', padx=5)
        self.preview_combo.bind('<<ComboboxSelected>>', self.update_preview)

        self.preview_text = scrolledtext.ScrolledText(preview_frame, height=12, width=80, wrap='word')
        self.preview_text.pack(fill='both', expand=True)

        stats_frame = ttk.LabelFrame(frame, text="Combined Statistics", padding=10)
        stats_frame.pack(fill='x', padx=10, pady=10)
        self.stats_label = ttk.Label(stats_frame, text="No files loaded", font='Arial 10')
        self.stats_label.pack(anchor='w')

        ttk.Button(frame, text="Process All Files & Build Token List",
                   command=self.process_files, style='Accent.TButton').pack(pady=10)

    def setup_train_tab(self, notebook):
        frame = ttk.Frame(notebook)
        notebook.add(frame, text='Train')

        ttk.Label(frame, text="Training Settings:", font='Arial 10 bold').pack(anchor='w', pady=(10, 5))

        settings_frame = ttk.Frame(frame)
        settings_frame.pack(fill='x', pady=5)

        ttk.Label(settings_frame, text="Window (Context):").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.seq_len = tk.IntVar(value=100)
        ttk.Entry(settings_frame, textvariable=self.seq_len, width=8).grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(settings_frame, text="Batch Size:").grid(row=0, column=2, sticky='w', padx=5, pady=2)
        self.batch_size = tk.IntVar(value=16)
        ttk.Entry(settings_frame, textvariable=self.batch_size, width=8).grid(row=0, column=3, padx=5, pady=2)

        ttk.Label(settings_frame, text="Epochs:").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        self.epochs = tk.IntVar(value=50)
        ttk.Entry(settings_frame, textvariable=self.epochs, width=8).grid(row=1, column=1, padx=5, pady=2)

        ttk.Label(settings_frame, text="Learning Rate:").grid(row=1, column=2, sticky='w', padx=5, pady=2)
        self.lr = tk.StringVar(value="0.001")
        ttk.Entry(settings_frame, textvariable=self.lr, width=8).grid(row=1, column=3, padx=5, pady=2)

        ttk.Label(settings_frame, text="Model Size:").grid(row=2, column=0, sticky='w', padx=5, pady=2)
        self.model_size = tk.StringVar(value="Medium")
        size_combo = ttk.Combobox(settings_frame, textvariable=self.model_size,
                                  values=["Small", "Medium", "Large"], width=8)
        size_combo.grid(row=2, column=1, padx=5, pady=2)
        size_combo.bind('<<ComboboxSelected>>', self._on_model_size_change)

        ttk.Label(settings_frame, text="Kernel Size:").grid(row=2, column=2, sticky='w', padx=5, pady=2)
        self.kernel_size = tk.StringVar(value="3")
        kernel_combo = ttk.Combobox(settings_frame, textvariable=self.kernel_size,
                                    values=["3", "5", "7"], width=8)
        kernel_combo.grid(row=2, column=3, padx=5, pady=2)
        kernel_combo.bind('<<ComboboxSelected>>', self._on_kernel_change)

        ttk.Label(settings_frame, text="Receptive Field:").grid(row=3, column=0, sticky='w', padx=5, pady=2)
        self.rf = tk.StringVar(value="127")
        self.rf_entry = ttk.Entry(settings_frame, textvariable=self.rf, width=8)
        self.rf_entry.grid(row=3, column=1, padx=5, pady=2)
        self.rf_entry.bind('<FocusOut>', self._snap_rf)
        self.rf_entry.bind('<Return>', self._snap_rf)

        self.rf_hint = ttk.Label(settings_frame, text="(odd; snapped on blur)",
                                 foreground='gray', font=('Arial', 8))
        self.rf_hint.grid(row=3, column=2, columnspan=2, sticky='w', padx=5, pady=2)

        self.rf_preview = tk.StringVar(value="")
        ttk.Label(frame, textvariable=self.rf_preview,
                  foreground='#2266aa', font=('Arial', 9, 'italic')).pack(anchor='w', padx=10, pady=(5, 0))

        save_load_frame = ttk.Frame(frame)
        save_load_frame.pack(fill='x', pady=10)
        ttk.Button(save_load_frame, text="Save Model", command=self.save_model).pack(side='left', padx=5)
        ttk.Button(save_load_frame, text="Load Model", command=self.load_model).pack(side='left', padx=5)

        init_train_frame = ttk.Frame(frame)
        init_train_frame.pack(pady=10)
        self.init_btn = ttk.Button(init_train_frame, text="1. Initialize Model",
                                   command=self.init_model, style='Accent.TButton', state='disabled')
        self.init_btn.pack(side='left', padx=5)
        self.train_btn = ttk.Button(init_train_frame, text="2. Start Training",
                                    command=self.start_training, state='disabled')
        self.train_btn.pack(side='left', padx=5)
        self.stop_btn = ttk.Button(init_train_frame, text="Stop",
                                   command=self.stop_training, state='disabled')
        self.stop_btn.pack(side='left', padx=5)

        self.progress = ttk.Progressbar(frame, mode='indeterminate', length=400)
        self.progress.pack(pady=5)

        loss_frame = ttk.Frame(frame)
        loss_frame.pack(pady=5)
        ttk.Label(loss_frame, text="Current Loss:").pack(side='left')
        self.loss_label = ttk.Label(loss_frame, text="0.0000", font='Arial 10 bold', foreground='blue')
        self.loss_label.pack(side='left', padx=5)

        ttk.Label(frame, text="Training Log:", font='Arial 10 bold').pack(anchor='w', pady=(10, 5))
        self.log = scrolledtext.ScrolledText(frame, height=12, width=80, state='disabled')
        self.log.pack(fill='both', expand=True, pady=5)

        self._on_model_size_change()
        self._refresh_rf_preview()

    def _on_model_size_change(self, event=None):
        presets = {"Small": 127, "Medium": 255, "Large": 511}
        size = self.model_size.get()
        if size in presets:
            self.rf.set(str(presets[size]))
        self._refresh_rf_preview()

    def _on_kernel_change(self, event=None):
        self._refresh_rf_preview()

    def _snap_rf(self, event=None):
        try:
            value = int(float(self.rf.get()))
        except (ValueError, TypeError):
            value = 127
        snapped = snap_to_odd(value)
        self.rf.set(str(snapped))
        self._refresh_rf_preview()

    def _refresh_rf_preview(self):
        try:
            target_rf = snap_to_odd(int(float(self.rf.get())))
            k = int(self.kernel_size.get())
        except (ValueError, TypeError):
            self.rf_preview.set("")
            return
        try:
            n, actual = compute_layers_for_rf(target_rf, k)
        except Exception as e:
            self.rf_preview.set(f"error: {e}")
            return
        self.rf_preview.set(
            f"-> kernel={k}, layers={n}, actual RF={actual} "
            f"(target {target_rf})"
        )

    def setup_generate_tab(self, notebook):
        frame = ttk.Frame(notebook)
        notebook.add(frame, text='Generate')

        ttk.Label(frame, text="Generation Settings:", font='Arial 10 bold').pack(anchor='w', pady=(10, 5))

        gen_frame = ttk.Frame(frame)
        gen_frame.pack(fill='x', pady=5)

        # Seed text
        ttk.Label(gen_frame, text="Seed Text:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.seed_text = tk.StringVar(value="c4 ")
        ttk.Entry(gen_frame, textvariable=self.seed_text, width=40).grid(
            row=0, column=1, columnspan=3, padx=5, pady=2, sticky='ew')

        # Length — integer, up to MAX_GEN_LENGTH.  Text entry instead of slider.
        ttk.Label(gen_frame, text="Length (chars):").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        self.gen_length = tk.StringVar(value="1000")
        self.gen_length_entry = ttk.Entry(gen_frame, textvariable=self.gen_length, width=12)
        self.gen_length_entry.grid(row=1, column=1, padx=5, pady=2, sticky='w')
        ttk.Label(gen_frame, text=f"(1 – {MAX_GEN_LENGTH:,}; longer = slower)",
                  foreground='gray', font=('Arial', 8)).grid(
            row=1, column=2, columnspan=2, sticky='w', padx=5, pady=2)

        # Temperature — float, step 0.01
        ttk.Label(gen_frame, text="Temperature:").grid(row=2, column=0, sticky='w', padx=5, pady=2)
        self.temp = tk.DoubleVar(value=0.80)
        self.temp_display = tk.StringVar(value="0.80")
        tk.Scale(gen_frame, from_=0.10, to=2.00, variable=self.temp,
                 orient='horizontal', length=200, resolution=0.01,
                 command=lambda v: self.temp_display.set(f"{float(v):.2f}")
                 ).grid(row=2, column=1, padx=5, pady=2)
        ttk.Label(gen_frame, textvariable=self.temp_display, width=6).grid(
            row=2, column=2, padx=5, pady=2, sticky='w')

        # Top-K — integer, step 1, 0 = off
        ttk.Label(gen_frame, text="Top-K (0=off):").grid(row=3, column=0, sticky='w', padx=5, pady=2)
        self.top_k = tk.IntVar(value=0)
        self.top_k_display = tk.StringVar(value="0")
        tk.Scale(gen_frame, from_=0, to=200, variable=self.top_k,
                 orient='horizontal', length=200, resolution=1,
                 command=lambda v: self.top_k_display.set(str(int(float(v))))
                 ).grid(row=3, column=1, padx=5, pady=2)
        ttk.Label(gen_frame, textvariable=self.top_k_display, width=6).grid(
            row=3, column=2, padx=5, pady=2, sticky='w')

        # Top-P — float, step 0.01, 1.0 = off
        ttk.Label(gen_frame, text="Top-P (1.0=off):").grid(row=4, column=0, sticky='w', padx=5, pady=2)
        self.top_p = tk.DoubleVar(value=1.00)
        self.top_p_display = tk.StringVar(value="1.00")
        tk.Scale(gen_frame, from_=0.00, to=1.00, variable=self.top_p,
                 orient='horizontal', length=200, resolution=0.01,
                 command=lambda v: self.top_p_display.set(f"{float(v):.2f}")
                 ).grid(row=4, column=1, padx=5, pady=2)
        ttk.Label(gen_frame, textvariable=self.top_p_display, width=6).grid(
            row=4, column=2, padx=5, pady=2, sticky='w')

        # Buttons
        btn_frame = ttk.Frame(gen_frame)
        btn_frame.grid(row=5, column=0, columnspan=4, pady=10)

        self.gen_btn = ttk.Button(btn_frame, text="Generate Music", command=self.start_generation)
        self.gen_btn.pack(side='left', padx=5)

        self.stop_gen_btn = ttk.Button(btn_frame, text="Stop",
                                       command=self.stop_generation, state='disabled')
        self.stop_gen_btn.pack(side='left', padx=5)

        self.gen_progress_label = ttk.Label(gen_frame, text="", foreground='green')
        self.gen_progress_label.grid(row=6, column=0, columnspan=4, pady=5)

        # Generated text display
        text_frame = ttk.LabelFrame(frame, text="Generated Music-Text", padding=10)
        text_frame.pack(fill='both', expand=True, padx=10, pady=10)

        self.output = scrolledtext.ScrolledText(text_frame, height=20, width=80,
                                                wrap='word', font=('Courier', 10))
        self.output.pack(fill='both', expand=True)

        action_frame = ttk.Frame(frame)
        action_frame.pack(fill='x', pady=5)
        ttk.Button(action_frame, text="Clear", command=self.clear_output).pack(side='left', padx=5)
        ttk.Button(action_frame, text="Copy", command=self.copy_output).pack(side='left', padx=5)
        ttk.Button(action_frame, text="Save Text", command=self.save_output).pack(side='left', padx=5)
        ttk.Button(action_frame, text="Save MIDI",
                   command=self.save_midi, style='Accent.TButton').pack(side='left', padx=5)

    # ==================== FILE MANAGEMENT ====================
    def add_files(self):
        files = filedialog.askopenfilenames(
            title="Select MIDI files",
            filetypes=[("MIDI files", "*.mid *.midi *.MID"), ("All files", "*.*")]
        )
        if files:
            for file in files:
                if file not in self.loaded_files:
                    self.load_file(file)

    def add_folder(self):
        folder = filedialog.askdirectory(title="Select folder with MIDI files")
        if folder:
            for root, dirs, files in os.walk(folder):
                for file in files:
                    if file.lower().endswith(('.mid', '.midi')):
                        file_path = os.path.join(root, file)
                        if file_path not in self.loaded_files:
                            self.load_file(file_path)

    def load_file(self, file_path):
        try:
            text, bpm = midi_to_text(file_path, multi_track=True)
            if not text.strip():
                messagebox.showwarning("Skipped", f"{os.path.basename(file_path)}: no notes found")
                return

            self.loaded_files.append(file_path)
            self.loaded_texts.append(text)
            self.programs.update(extract_programs(file_path))
            self.tempos.append(bpm)

            filename = os.path.basename(file_path)
            file_size = os.path.getsize(file_path)
            char_count = len(text)

            item_id = self.file_tree.insert('', 'end', values=(
                len(self.loaded_files), filename, f"{file_size:,}", f"{char_count:,}", "✓ Loaded"))
            self.file_tree.set(item_id, column='#', value=len(self.loaded_files))

            self.preview_combo['values'] = list(self.loaded_files)
            if len(self.loaded_files) == 1:
                self.preview_var.set(file_path)
                self.update_preview()

            self.update_stats()
            self.status.set(f"Loaded: {filename} ({char_count:,} chars, {bpm:.0f} BPM)")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load {file_path}: {str(e)}")

    def remove_file(self):
        selection = self.file_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a file to remove")
            return
        for item in selection:
            index = int(self.file_tree.item(item, 'values')[0]) - 1
            if 0 <= index < len(self.loaded_files):
                removed_file = self.loaded_files.pop(index)
                self.loaded_texts.pop(index)
                if index < len(self.tempos):
                    self.tempos.pop(index)
                self.file_tree.delete(item)
                for i, item_id in enumerate(self.file_tree.get_children()):
                    values = list(self.file_tree.item(item_id, 'values'))
                    values[0] = str(i + 1)
                    self.file_tree.item(item_id, values=values)
                self.status.set(f"Removed: {os.path.basename(removed_file)}")

        self.preview_combo['values'] = list(self.loaded_files)
        if self.loaded_files:
            self.preview_var.set(self.loaded_files[0])
            self.update_preview()
        else:
            self.preview_text.delete('1.0', 'end')
        self.update_stats()

    def clear_files(self):
        if not self.loaded_files:
            return
        if messagebox.askyesno("Confirm", "Clear all loaded files?"):
            self.loaded_files.clear()
            self.loaded_texts.clear()
            self.programs.clear()
            self.tempos.clear()
            self.file_tree.delete(*self.file_tree.get_children())
            self.preview_text.delete('1.0', 'end')
            self.preview_combo['values'] = []
            self.preview_var.set('')
            self.stats_label.config(text="No files loaded")
            self.status.set("All files cleared")

    def update_preview(self, event=None):
        selected = self.preview_var.get()
        if selected in self.loaded_files:
            index = self.loaded_files.index(selected)
            text = self.loaded_texts[index]
            self.preview_text.delete('1.0', 'end')
            self.preview_text.insert('1.0', text[:5000])
            if len(text) > 5000:
                self.preview_text.insert('end', '\n\n[... truncated]')

    def update_stats(self):
        if not self.loaded_files:
            self.stats_label.config(text="No files loaded")
            return
        total_chars = sum(len(t) for t in self.loaded_texts)
        total_size = sum(os.path.getsize(f) for f in self.loaded_files)
        avg_tempo = (sum(self.tempos) / len(self.tempos)) if self.tempos else 120.0
        self.stats_label.config(text=(f"Files: {len(self.loaded_files)} | "
                                      f"Total music-text characters: {total_chars:,} | "
                                      f"Total size: {total_size:,} bytes | "
                                      f"Avg tempo: {avg_tempo:.0f} BPM"))

    def process_files(self):
        if not self.loaded_texts:
            messagebox.showerror("Error", "No files loaded!")
            return

        self.processor = TextProcessor(seq_len=self.seq_len.get())
        combined_text = FILE_SEP.join(self.loaded_texts)
        self.processor.tokens = self.processor.text_to_tokens(combined_text)

        stats = {'chars': len(combined_text),
                 'vocab': len(self.processor.char2idx),
                 'files': len(self.loaded_texts)}

        self.log_message(f"Processed {stats['files']} MIDI files")
        self.log_message(f"Combined: {stats['chars']:,} music-text characters")
        self.log_message(f"Fixed vocabulary size: {stats['vocab']} tokens")

        unk_count = self.processor.tokens.count(1)
        if unk_count > 0:
            percent = (unk_count / max(1, stats['chars'])) * 100
            self.log_message(f"Warning: {unk_count} unknown characters ({percent:.2f}%) mapped to <UNK>")

        self.init_btn.config(state='normal')
        self.status.set(f"Ready to initialize model on {stats['chars']:,} characters")

    # ==================== MODEL INITIALIZATION ====================
    def init_model(self):
        if not hasattr(self.processor, 'tokens') or len(self.processor.tokens) == 0:
            messagebox.showerror("Error", "Please process files first!")
            return

        vocab_size = len(self.processor.char2idx)
        context_size = self.seq_len.get()
        model_size = self.model_size.get()
        kernel = int(self.kernel_size.get())

        try:
            target_rf = snap_to_odd(int(float(self.rf.get())))
        except (ValueError, TypeError):
            target_rf = 127
        self.rf.set(str(target_rf))

        if model_size == "Small":
            embed, hidden = 64, 128
        elif model_size == "Medium":
            embed, hidden = 128, 256
        else:
            embed, hidden = 192, 384

        n_layers, actual_rf = compute_layers_for_rf(target_rf, kernel)

        self.model = DilatedConvLM(
            vocab_size=vocab_size,
            embed_size=embed,
            hidden_size=hidden,
            n_layers=n_layers,
            kernel_size=kernel,
            context_size=context_size,
        )

        breakdown = self.model.count_params()

        self.log_message(f"Model initialized ({model_size} config)")
        self.log_message(f"  Embed: {embed}, Hidden: {hidden}")
        self.log_message(f"  Kernel: {kernel}, Layers: {n_layers}")
        self.log_message(f"  Requested RF: {target_rf}  ->  Actual RF: {actual_rf}")
        self.log_message(f"  Training window: {context_size} tokens")
        if actual_rf < context_size:
            self.log_message(
                f"  WARNING: RF ({actual_rf}) < window ({context_size}). "
                f"Increase RF to at least {context_size}."
            )
        self.log_message(f"  Vocabulary size: {vocab_size}")
        self.log_message(f"  --- Parameter breakdown ---")
        for name, n in breakdown.items():
            self.log_message(f"    {name:<14} {n:>12,}")

        self.train_btn.config(state='normal')
        self.status.set(f"Model initialized (RF={actual_rf}) - Ready to train")

    # ==================== TRAINING ====================
    def start_training(self):
        if self.model is None:
            messagebox.showerror("Error", "Please initialize model first!")
            return
        if self.training:
            return

        self.training = True
        self.init_btn.config(state='disabled')
        self.train_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.progress.start()

        threading.Thread(target=self.train_loop, daemon=True).start()

        self.log_message(f"Training started on {len(self.processor.tokens):,} tokens")
        self.status.set("Training started...")

    def train_loop(self):
        try:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.lr.get()))
            criterion = nn.CrossEntropyLoss()

            epochs = self.epochs.get()
            batch_size = self.batch_size.get()

            for epoch in range(epochs):
                if not self.training:
                    break

                self.model.train()
                total_loss = 0
                steps = 0

                for _ in range(100):
                    if not self.training:
                        break
                    batch_x, batch_y = self.processor.get_batch(batch_size)
                    optimizer.zero_grad()
                    logits = self.model(batch_x)
                    logits = logits.reshape(-1, self.model.vocab_size)
                    targets = batch_y.reshape(-1)
                    loss = criterion(logits, targets)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()

                    total_loss += loss.item()
                    steps += 1

                avg_loss = total_loss / max(steps, 1)
                self.update_queue.put(('loss', (epoch + 1, epochs, avg_loss)))

                if (epoch + 1) % 10 == 0:
                    threading.Thread(target=self.generate_sample_for_training,
                                     args=(epoch + 1,), daemon=True).start()

                time.sleep(0.1)

            self.update_queue.put(('training_complete', None))
        except Exception as e:
            self.update_queue.put(('error', f"Training error: {str(e)}"))
            traceback.print_exc()

    @staticmethod
    def _sample_token(logits, temperature, top_k, top_p):
        t = max(float(temperature), 1e-6)
        logits = logits / t

        if top_k and top_k > 0:
            k = min(int(top_k), logits.size(-1))
            topk_vals, _ = torch.topk(logits, k)
            threshold = topk_vals[-1]
            logits = torch.where(
                logits < threshold,
                torch.full_like(logits, float('-inf')),
                logits,
            )

        if top_p and top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            cumprobs = torch.cumsum(probs, dim=-1)

            to_remove = cumprobs > top_p
            to_remove[1:] = to_remove[:-1].clone()
            to_remove[0] = False

            indices_to_remove = torch.zeros_like(to_remove)
            indices_to_remove.scatter_(0, sorted_idx, to_remove)
            logits = logits.masked_fill(indices_to_remove, float('-inf'))

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1).item()

    def generate_sample_for_training(self, epoch):
        try:
            if self.model is None:
                return
            self.model.eval()
            seed_tokens = self.processor.text_to_tokens("c4 ")
            if not seed_tokens:
                return

            window = self.model.context_size
            generated = list(seed_tokens)

            with torch.no_grad():
                for _ in range(100):
                    ctx = generated[-window:]
                    x = torch.tensor([ctx], dtype=torch.long)
                    logits = self.model(x)
                    nxt = self._sample_token(logits[0, -1, :], 0.8, 0, 1.0)
                    generated.append(nxt)

            sample = self.processor.tokens_to_text(generated)
            self.update_queue.put(('sample', (epoch, sample[:100])))
        except Exception as e:
            print(f"Sample generation error: {e}")

    def stop_training(self):
        self.training = False
        self.log_message("Training stopped")

    def log_message(self, message):
        self.log.config(state='normal')
        self.log.insert('end', f"{time.strftime('%H:%M:%S')} - {message}\n")
        self.log.see('end')
        self.log.config(state='disabled')

    # ==================== SAVE / LOAD MODEL ====================
    def save_model(self):
        if self.model is None:
            messagebox.showerror("Error", "No model to save!")
            return
        filename = filedialog.asksaveasfilename(
            defaultextension=".pth",
            filetypes=[("PyTorch model", "*.pth"), ("All files", "*.*")]
        )
        if filename:
            try:
                self.model.save_model(filename)
                self.log_message(f"Model saved to {filename}")
                self.status.set(f"Model saved: {os.path.basename(filename)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save model: {str(e)}")

    def load_model(self):
        filename = filedialog.askopenfilename(
            title="Select model file",
            filetypes=[("PyTorch model", "*.pth"), ("All files", "*.*")]
        )
        if filename:
            try:
                self.model = DilatedConvLM.load_model(filename)
                self.processor = TextProcessor(seq_len=self.model.context_size)

                fixed_vocab_size = len(self.processor.char2idx)
                if fixed_vocab_size != self.model.vocab_size:
                    self.log_message(f"WARNING: Fixed vocabulary size ({fixed_vocab_size}) does "
                                     f"not match model's vocab size ({self.model.vocab_size}).")

                self.log_message(f"Model loaded from {os.path.basename(filename)}")
                self.log_message(f"  kernel={self.model.kernel_size}, "
                                 f"layers={self.model.n_layers}, "
                                 f"RF={self.model.receptive_field}, "
                                 f"context={self.model.context_size}")
                self.log_message(f"Fixed vocabulary size: {fixed_vocab_size}")
                self.status.set("Model loaded successfully")

                self.init_btn.config(state='normal')
                self.train_btn.config(state='normal')
                self.update_model_settings_from_checkpoint()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {str(e)}")
                traceback.print_exc()

    def update_model_settings_from_checkpoint(self):
        if not self.model:
            return
        embed = self.model.embed_size
        if embed <= 64:
            self.model_size.set('Small')
        elif embed <= 128:
            self.model_size.set('Medium')
        else:
            self.model_size.set('Large')

        self.kernel_size.set(str(self.model.kernel_size))
        self.seq_len.set(self.model.context_size)
        self.rf.set(str(self.model.receptive_field))
        self._refresh_rf_preview()

    # ==================== GENERATION ====================
    def start_generation(self):
        if self.model is None:
            messagebox.showerror("Error", "Please train or load a model first")
            return
        if self.generating:
            return

        seed = self.seed_text.get()
        if not seed:
            messagebox.showerror("Error", "Please enter seed text")
            return

        # Parse length from the text entry
        try:
            length = int(self.gen_length.get())
        except (ValueError, TypeError):
            messagebox.showerror("Error", "Length must be an integer.")
            return
        if length < 1:
            length = 1
        elif length > MAX_GEN_LENGTH:
            if not messagebox.askyesno(
                "Very long generation",
                f"Length of {length:,} exceeds the {MAX_GEN_LENGTH:,} cap.\n"
                f"Clamp to {MAX_GEN_LENGTH:,} and continue?"
            ):
                return
            length = MAX_GEN_LENGTH
            self.gen_length.set(str(length))

        temp = float(self.temp.get())
        top_k = int(self.top_k.get())
        top_p = float(self.top_p.get())

        # Clear and start with the seed (so the visible text is the
        # exact text that Save MIDI / Save Text operate on).
        self.output.delete('1.0', 'end')
        self.output.insert('1.0', seed)

        self.status.set("Generating music...")
        self.gen_progress_label.config(text=f"Generating 0 / {length:,}...", foreground='green')

        self.generating = True
        self.gen_btn.config(state='disabled')
        self.stop_gen_btn.config(state='normal')

        threading.Thread(target=self.generate_thread,
                         args=(seed, length, temp, top_k, top_p),
                         daemon=True).start()

    def stop_generation(self):
        if not self.generating:
            return
        self.generating = False
        self.gen_progress_label.config(text="Stopping...", foreground='orange')
        self.status.set("Stopping generation...")

    def generate_thread(self, seed, length, temp, top_k, top_p):
        try:
            if self.model is None:
                return
            self.model.eval()
            seed_tokens = self.processor.text_to_tokens(seed)
            if not seed_tokens:
                return

            window = self.model.context_size
            generated = list(seed_tokens)
            last_sent = len(generated)             # index of next char to stream out
            next_update_at = len(generated) + update_interval(len(generated))

            with torch.no_grad():
                for i in range(length):
                    if not self.generating:
                        break

                    ctx = generated[-window:]
                    x = torch.tensor([ctx], dtype=torch.long)
                    logits = self.model(x)
                    nxt = self._sample_token(logits[0, -1, :], temp, top_k, top_p)
                    generated.append(nxt)

                    n = len(generated)
                    if n >= next_update_at:
                        new_text = self.processor.tokens_to_text(generated[last_sent:])
                        self.update_queue.put(('gen_append', (new_text, n - len(seed_tokens))))
                        last_sent = n
                        next_update_at = n + update_interval(n)

            # Flush any remaining characters
            if last_sent < len(generated):
                new_text = self.processor.tokens_to_text(generated[last_sent:])
                self.update_queue.put(('gen_append',
                                       (new_text, len(generated) - len(seed_tokens))))

            was_stopped = not self.generating
            self.generating = False
            self.update_queue.put(('gen_complete',
                                   (len(generated) - len(seed_tokens), was_stopped)))

        except Exception as e:
            self.generating = False
            self.update_queue.put(('error', f"Generation error: {str(e)}"))
            traceback.print_exc()

    def update_ui_thread(self):
        while True:
            try:
                msg_type, data = self.update_queue.get(timeout=0.1)

                if msg_type == 'loss':
                    epoch, total_epochs, loss = data
                    self.loss_label.config(text=f"{loss:.4f}")
                    self.log_message(f"Epoch {epoch}/{total_epochs} - Loss: {loss:.4f}")
                    self.status.set(f"Training: Epoch {epoch}/{total_epochs}, Loss: {loss:.4f}")

                elif msg_type == 'sample':
                    epoch, sample = data
                    self.log_message(f"Epoch {epoch} sample: {sample}...")

                elif msg_type == 'training_complete':
                    self.training = False
                    self.init_btn.config(state='normal')
                    self.train_btn.config(state='normal')
                    self.stop_btn.config(state='disabled')
                    self.progress.stop()
                    self.log_message("Training completed!")
                    self.status.set("Training completed")

                elif msg_type == 'gen_append':
                    new_text, produced = data
                    self.output.insert('end', new_text)
                    self.output.see('end')
                    self.gen_progress_label.config(
                        text=f"Generating {produced:,}...", foreground='green')
                    self.status.set(f"Generating... {produced:,} chars so far")

                elif msg_type == 'gen_complete':
                    produced, was_stopped = data
                    self.gen_btn.config(state='normal')
                    self.stop_gen_btn.config(state='disabled')

                    if was_stopped:
                        self.status.set(f"Generation stopped ({produced:,} chars)")
                        self.gen_progress_label.config(text="⏹ Stopped", foreground='orange')
                    else:
                        self.status.set(f"Generated {produced:,} characters")
                        self.gen_progress_label.config(text="✓ Generation complete!",
                                                       foreground='green')

                    self.log_message(
                        f"Generated from seed: '{self.seed_text.get()[:30]}...' "
                        f"(temp={self.temp.get():.2f}, top_k={self.top_k.get()}, "
                        f"top_p={self.top_p.get():.2f}) -> {produced:,} chars"
                    )
                    self.generating = False

                elif msg_type == 'error':
                    self.log_message(data)
                    self.gen_progress_label.config(text="✗ Failed", foreground='red')
                    self.gen_btn.config(state='normal')
                    self.stop_gen_btn.config(state='disabled')
                    self.generating = False

            except queue.Empty:
                continue
            except Exception as e:
                print(f"UI update error: {e}")
                self.generating = False

    def clear_output(self):
        self.output.delete('1.0', 'end')
        self.gen_progress_label.config(text="")

    def copy_output(self):
        text = self.output.get('1.0', 'end-1c')
        if text:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.status.set("Copied to clipboard")

    def save_output(self):
        text = self.output.get('1.0', 'end-1c')
        if not text:
            return
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filename:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(text)
            self.status.set(f"Saved to {os.path.basename(filename)}")

    # ==================== SAVE MIDI ====================
    def save_midi(self):
        text = self.output.get('1.0', 'end-1c')
        if not text.strip():
            messagebox.showwarning("No output", "Generate something first.")
            return

        events = text_to_events(text)
        if not events:
            messagebox.showerror(
                "Cannot parse",
                "No musical events could be parsed from the generated text.\n"
                "Try training more, lowering temperature, or using a musical seed "
                "like 'c4 e4 g4 '."
            )
            return

        filename = filedialog.asksaveasfilename(
            title="Save generated MIDI",
            defaultextension=".mid",
            filetypes=[("MIDI files", "*.mid"), ("All files", "*.*")]
        )
        if not filename:
            return

        try:
            tempo = (sum(self.tempos) / len(self.tempos)) if self.tempos else 120.0
            events_to_midi(events, filename, tempo=tempo, programs=self.programs)
            self.status.set(f"Saved MIDI: {os.path.basename(filename)} "
                            f"({len(events):,} events, {tempo:.0f} BPM)")
            self.log_message(f"Saved MIDI: {filename} "
                             f"({len(events):,} events, tempo={tempo:.0f})")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save MIDI: {str(e)}")
            traceback.print_exc()


# ==================== MAIN ====================
def main():
    try:
        root = tk.Tk()
        style = ttk.Style()
        style.configure('Accent.TButton', font=('Arial', 10, 'bold'))
        app = MultiFileMusicGenerator(root)

        root.update_idletasks()
        width = root.winfo_width()
        height = root.winfo_height()
        x = (root.winfo_screenwidth() // 2) - (width // 2)
        y = (root.winfo_screenheight() // 2) - (height // 2)
        root.geometry(f'1000x850+{x}+{y}')
        root.mainloop()
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        input("Press Enter to exit...")


if __name__ == "__main__":
    main()