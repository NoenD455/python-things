#!/usr/bin/env python3
"""
Tiny RNN Text Generator  —  PyTorch + Tkinter

Features
--------
* Unified file list for `.txt` and `.csv` files; remove them individually,
  mix text files and CSVs freely.
* **CSV loading**: pick a column per CSV; each row becomes a `<start>…<end>`
  sample.  The special tokens are added to the tokenizer vocabulary and the
  training corpus.  Double-click a CSV in the file list to change its column.
* Char / BPE tokenizers (both with special-token support; BPE has a
  character-level fallback plus `<unk>`).
* RNN / LSTM / GRU model with configurable embed, hidden, layers, dropout
  and BPTT unroll length.
* Separate "Initialize Model" from "Start / Stop Training".
* Save / Load full checkpoints (model + tokenizer + data + file list +
  CSV column choices).
* Real-time generation preview; generate while training (snapshot model).
* Multi-sample generation in CSV mode: every sample is seeded with
  `<start>` + prompt, stops at `<end>`, and `<start>` is masked out so it
  can never re-appear mid-sample.  The Samples spinbox is only shown when
  the tokenizer actually has `<start>`/`<end>` tokens.
* "Prime (inspect)" shows token count, perplexity and next-token guesses.
* Neuron activation visualizer (Karpathy-style).

Requirements:  pip install torch
"""

import os
import re
import csv
import copy
import time
import queue
import threading
import traceback
from collections import Counter

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
#  TOKENIZERS
# =============================================================================
EOW = "</w>"
UNK_TOKEN = "<unk>"
START_TOKEN = "<start>"
END_TOKEN = "<end>"


def _split_on_special(text, special_tokens):
    if not special_tokens:
        return [(text, False)]
    pattern = "|".join(re.escape(t) for t in special_tokens)
    parts = re.split(f"({pattern})", text)
    return [(p, p in special_tokens) for p in parts if p]


class CharTokenizer:
    """Plain character level tokenizer (supports special tokens)."""
    name = "Character"

    def __init__(self):
        self.chars = []
        self.stoi = {}
        self.itos = {}
        self.special_tokens = []

    def train(self, text, special_tokens=None):
        if special_tokens is not None:
            self.special_tokens = list(special_tokens)
        stripped = text
        for st in self.special_tokens:
            stripped = stripped.replace(st, "")
        char_set = set(stripped) | set(self.special_tokens)
        self.chars = sorted(char_set)
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for i, c in enumerate(self.chars)}

    def encode(self, text):
        ids = []
        for piece, is_special in _split_on_special(text, self.special_tokens):
            if is_special:
                ids.append(self.stoi[piece])
            else:
                ids.extend(self.stoi[c] for c in piece if c in self.stoi)
        return ids

    def decode(self, ids):
        return "".join(self.itos.get(i, "") for i in ids)

    @property
    def vocab_size(self):
        return len(self.chars)

    def state_dict(self):
        return {"chars": list(self.chars),
                "special_tokens": list(self.special_tokens)}

    def load_state_dict(self, state):
        self.chars = list(state["chars"])
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for i, c in enumerate(self.chars)}
        self.special_tokens = list(state.get("special_tokens", []))


class BPETokenizer:
    """Dependency-free BPE tokenizer with special-token support and a
    character-level fallback for characters not seen during training."""

    name = "BPE"

    def __init__(self, target_vocab=2000):
        self.target_vocab = target_vocab
        self.merges = []
        self.ranks = {}
        self.vocab = []
        self.stoi = {}
        self.itos = {}
        self.unk_id = 0
        self.special_tokens = []

    @staticmethod
    def _chunks(text):
        parts = re.findall(r"\s+|\S+", text)
        return [p if p.isspace() else p + EOW for p in parts]

    def train(self, text, target_vocab=None, progress_cb=None,
              special_tokens=None):
        if target_vocab:
            self.target_vocab = target_vocab
        if special_tokens is not None:
            self.special_tokens = list(special_tokens)

        stripped = text
        for st in self.special_tokens:
            stripped = stripped.replace(st, " ")

        chunks = self._chunks(stripped)
        freq = Counter(chunks)
        words = list(freq.keys())
        freqs = [freq[w] for w in words]
        splits = [list(w) for w in words]

        vocab = {UNK_TOKEN}
        vocab.update(self.special_tokens)
        for w in words:
            vocab.update(w)

        stats = Counter()
        where = {}
        for idx, s in enumerate(splits):
            f = freqs[idx]
            for i in range(len(s) - 1):
                p = (s[i], s[i + 1])
                stats[p] += f
                where.setdefault(p, set()).add(idx)

        merges = []
        while len(vocab) < self.target_vocab and stats:
            pair, cnt = max(stats.items(), key=lambda kv: (kv[1], kv[0]))
            if cnt < 2:
                break
            a, b = pair
            new_tok = a + b
            if new_tok in self.special_tokens:
                stats.pop(pair, None)
                continue
            vocab.add(new_tok)
            merges.append(pair)

            for idx in list(where.get(pair, ())):
                s = splits[idx]
                f = freqs[idx]
                for i in range(len(s) - 1):
                    p = (s[i], s[i + 1])
                    stats[p] -= f
                    if stats[p] <= 0:
                        del stats[p]
                    ws = where.get(p)
                    if ws:
                        ws.discard(idx)
                ns, i = [], 0
                while i < len(s):
                    if i < len(s) - 1 and s[i] == a and s[i + 1] == b:
                        ns.append(new_tok)
                        i += 2
                    else:
                        ns.append(s[i])
                        i += 1
                splits[idx] = ns
                for i in range(len(ns) - 1):
                    p = (ns[i], ns[i + 1])
                    stats[p] += f
                    where.setdefault(p, set()).add(idx)
            where.pop(pair, None)
            if progress_cb and len(merges) % 100 == 0:
                progress_cb(len(merges))

        self.merges = merges
        self.ranks = {p: i for i, p in enumerate(merges)}
        others = sorted(t for t in vocab if t != UNK_TOKEN)
        self.vocab = [UNK_TOKEN] + others
        self.stoi = {t: i for i, t in enumerate(self.vocab)}
        self.itos = {i: t for i, t in enumerate(self.vocab)}
        self.unk_id = self.stoi[UNK_TOKEN]

    def _encode_no_special(self, text):
        ids = []
        for chunk in self._chunks(text):
            if any(c not in self.stoi for c in chunk):
                for c in chunk:
                    ids.append(self.stoi.get(c, self.unk_id))
                continue
            s = list(chunk)
            while len(s) > 1:
                best_rank, best_i = None, -1
                for i in range(len(s) - 1):
                    r = self.ranks.get((s[i], s[i + 1]))
                    if r is not None and (best_rank is None or r < best_rank):
                        best_rank, best_i = r, i
                if best_i < 0:
                    break
                s = s[:best_i] + [s[best_i] + s[best_i + 1]] + s[best_i + 2:]
            ids.extend(self.stoi.get(t, self.unk_id) for t in s)
        return ids

    def encode(self, text):
        ids = []
        for piece, is_special in _split_on_special(text, self.special_tokens):
            if is_special:
                ids.append(self.stoi[piece])
            else:
                ids.extend(self._encode_no_special(piece))
        return ids

    def decode(self, ids):
        return ("".join(self.itos.get(i, "") for i in ids)
                .replace(EOW, "")
                .replace(UNK_TOKEN, ""))

    @property
    def vocab_size(self):
        return len(self.vocab)

    def state_dict(self):
        return {
            "merges": list(self.merges),
            "vocab": list(self.vocab),
            "target_vocab": self.target_vocab,
            "special_tokens": list(self.special_tokens),
        }

    def load_state_dict(self, state):
        self.merges = [tuple(m) for m in state["merges"]]
        self.ranks = {p: i for i, p in enumerate(self.merges)}
        self.vocab = list(state["vocab"])
        self.stoi = {t: i for i, t in enumerate(self.vocab)}
        self.itos = {i: t for i, t in enumerate(self.vocab)}
        self.target_vocab = state["target_vocab"]
        self.unk_id = self.stoi.get(UNK_TOKEN, 0)
        self.special_tokens = list(state.get("special_tokens", []))


# =============================================================================
#  MODEL
# =============================================================================
class RNNLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers,
                 rnn_type="LSTM", dropout=0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.dropout = dropout

        self.embed = nn.Embedding(vocab_size, embed_dim)
        cls = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[rnn_type]
        self.rnn = cls(embed_dim, hidden_size, num_layers,
                       batch_first=True,
                       dropout=dropout if num_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        e = self.drop(self.embed(x))
        out, hidden = self.rnn(e) if hidden is None else self.rnn(e, hidden)
        return self.fc(self.drop(out)), hidden

    def init_hidden(self, batch_size, device=None):
        if device is None:
            device = next(self.parameters()).device
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        if self.rnn_type == "LSTM":
            return (h, torch.zeros_like(h))
        return h

    def detach_hidden(self, hidden):
        if isinstance(hidden, tuple):
            return tuple(h.detach() for h in hidden)
        return hidden.detach()

    def config(self):
        return {
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "rnn_type": self.rnn_type,
            "dropout": self.dropout,
        }


def batchify(data, batch_size):
    n = len(data) // batch_size
    data = data[: n * batch_size]
    return data.view(batch_size, n).t().contiguous()


def get_batch(source, i, seq_len):
    seq_len = min(seq_len, len(source) - 1 - i)
    x = source[i: i + seq_len]
    y = source[i + 1: i + 1 + seq_len]
    return x.t(), y.t()


# =============================================================================
#  GUI
# =============================================================================
class App(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Tiny RNN Text Generator  —  PyTorch + Tkinter")
        self.geometry("1100x880")
        self.minsize(950, 720)

        # ---- state ----------------------------------------------------------
        self.files = []               # unified list of .txt and .csv paths
        self.csv_columns = {}         # path -> column name (CSVs only)

        self.tokenizer = None
        self.data_tensor = None
        self.prepared_sig = None

        self.model = None
        self.inference_model = None
        self.inference_lock = threading.Lock()
        self.optimizer = None

        self.q = queue.Queue()
        self.stop_flag = threading.Event()
        self.worker = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._is_generating = False

        self._build_ui()
        self.after(100, self._poll_queue)

    # =========================================================== UI BUILDING
    def _build_ui(self):
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=6, pady=6)

        self.tab_data = ttk.Frame(nb)
        self.tab_set = ttk.Frame(nb)
        self.tab_train = ttk.Frame(nb)
        self.tab_gen = ttk.Frame(nb)
        self.tab_act = ttk.Frame(nb)

        nb.add(self.tab_data, text="  1 · Data  ")
        nb.add(self.tab_set, text="  2 · Settings  ")
        nb.add(self.tab_train, text="  3 · Train  ")
        nb.add(self.tab_gen, text="  4 · Generate  ")
        nb.add(self.tab_act, text="  5 · Neurons  ")

        self._build_data_tab()
        self._build_settings_tab()
        self._build_train_tab()
        self._build_generate_tab()
        self._build_activation_tab()

        self.status = tk.StringVar(value=f"Device: {self.device}  |  no model loaded")
        ttk.Label(self, textvariable=self.status, anchor="w",
                  relief="sunken").pack(fill="x", side="bottom")

    # ------------------------------------------------------------ DATA TAB
    def _build_data_tab(self):
        f = self.tab_data

        bar = ttk.Frame(f)
        bar.pack(fill="x", padx=8, pady=8)
        ttk.Button(bar, text="Add Files…", command=self.add_files).pack(side="left")
        ttk.Button(bar, text="Add CSV…", command=self.add_csv).pack(side="left", padx=4)
        ttk.Button(bar, text="Add Folder…", command=self.add_folder).pack(side="left", padx=4)
        ttk.Button(bar, text="Remove Selected", command=self.remove_selected).pack(side="left", padx=4)
        ttk.Button(bar, text="Clear All", command=self.clear_all).pack(side="left", padx=4)
        ttk.Button(bar, text="Clear All CSVs", command=self.clear_all_csvs).pack(side="left", padx=4)

        mid = ttk.Frame(f)
        mid.pack(fill="both", expand=True, padx=8)
        sb = ttk.Scrollbar(mid, orient="vertical")
        self.file_list = tk.Listbox(mid, selectmode="extended", yscrollcommand=sb.set)
        sb.config(command=self.file_list.yview)
        self.file_list.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.file_list.bind("<Double-Button-1>", self._on_file_double_click)

        info = ttk.Frame(f)
        info.pack(fill="x", padx=8, pady=(8, 4))
        ttk.Button(info, text="Prepare Dataset  (read + tokenize)",
                   command=self.prepare_data).pack(side="left")
        self.data_info = tk.StringVar(value="No data prepared.")
        ttk.Label(info, textvariable=self.data_info).pack(side="left", padx=12)

        ttk.Label(f, foreground="#666",
                  text=("Add .txt files and/or .csv files (pick a column per CSV).  "
                        "Each CSV row becomes <start>text<end>.  Double-click a CSV "
                        "to change its column.\nPrepare Dataset reads everything, "
                        "trains the tokenizer and encodes the corpus into a tensor.  "
                        "Re-run whenever the file list or tokenizer settings change.")
                  ).pack(anchor="w", padx=10, pady=(0, 8))

    # -------------------------------------------------------- SETTINGS TAB
    def _build_settings_tab(self):
        f = self.tab_set

        self.var_tok = tk.StringVar(value="Character")
        self.var_bpe_vocab = tk.IntVar(value=2000)
        self.var_rnn = tk.StringVar(value="LSTM")
        self.var_embed = tk.IntVar(value=128)
        self.var_hidden = tk.IntVar(value=256)
        self.var_layers = tk.IntVar(value=2)
        self.var_dropout = tk.DoubleVar(value=0.2)
        self.var_seq = tk.IntVar(value=64)
        self.var_unroll = tk.IntVar(value=1)
        self.var_batch = tk.IntVar(value=32)
        self.var_lr = tk.DoubleVar(value=0.002)
        self.var_epochs = tk.IntVar(value=5)

        tok = ttk.LabelFrame(f, text="Tokenizer")
        tok.pack(fill="x", padx=10, pady=8)
        ttk.Label(tok, text="Vocabulary type").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        ttk.Combobox(tok, textvariable=self.var_tok, state="readonly",
                     values=["Character", "BPE"], width=14
                     ).grid(row=0, column=1, sticky="w", padx=6, pady=4)
        ttk.Label(tok, text="BPE target vocab size").grid(row=1, column=0, sticky="w", padx=6, pady=4)
        ttk.Spinbox(tok, from_=200, to=20000, increment=100,
                    textvariable=self.var_bpe_vocab, width=14
                    ).grid(row=1, column=1, sticky="w", padx=6, pady=4)

        mdl = ttk.LabelFrame(f, text="Recurrent network")
        mdl.pack(fill="x", padx=10, pady=8)
        ttk.Label(mdl, text="Cell type").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        ttk.Combobox(mdl, textvariable=self.var_rnn, state="readonly",
                     values=["RNN", "LSTM", "GRU"], width=14
                     ).grid(row=0, column=1, sticky="w", padx=6, pady=4)

        for i, (label, var, lo, hi, inc) in enumerate([
            ("Embedding size", self.var_embed, 8, 2048, 8),
            ("Hidden size (neurons)", self.var_hidden, 8, 4096, 8),
            ("Number of layers", self.var_layers, 1, 8, 1),
            ("Dropout", self.var_dropout, 0.0, 0.9, 0.05),
        ], start=1):
            ttk.Label(mdl, text=label).grid(row=i, column=0, sticky="w", padx=6, pady=4)
            ttk.Spinbox(mdl, from_=lo, to=hi, increment=inc,
                        textvariable=var, width=14
                        ).grid(row=i, column=1, sticky="w", padx=6, pady=4)

        tr = ttk.LabelFrame(f, text="Training")
        tr.pack(fill="x", padx=10, pady=8)
        for i, (label, var, lo, hi, inc) in enumerate([
            ("Sequence length", self.var_seq, 8, 1024, 8),
            ("Unroll length (BPTT)", self.var_unroll, 1, 64, 1),
            ("Batch size", self.var_batch, 1, 512, 1),
            ("Learning rate", self.var_lr, 0.00001, 1.0, 0.0005),
            ("Epochs", self.var_epochs, 1, 500, 1),
        ]):
            ttk.Label(tr, text=label).grid(row=i, column=0, sticky="w", padx=6, pady=4)
            ttk.Spinbox(tr, from_=lo, to=hi, increment=inc,
                        textvariable=var, width=14
                        ).grid(row=i, column=1, sticky="w", padx=6, pady=4)

        ttk.Label(tr, foreground="#666",
                  text="Unroll length = how many batches of BPTT before the hidden\n"
                       "state is detached.  1 = vanilla step-by-step training."
                  ).grid(row=5, column=0, columnspan=2, sticky="w", padx=6, pady=(4, 6))

    # ----------------------------------------------------------- TRAIN TAB
    def _build_train_tab(self):
        f = self.tab_train

        mdl = ttk.LabelFrame(f, text="Model")
        mdl.pack(fill="x", padx=8, pady=8)
        self.btn_init = ttk.Button(mdl, text="Initialize Model", command=self.init_model)
        self.btn_init.pack(side="left", padx=(6, 4), pady=6)
        self.btn_save = ttk.Button(mdl, text="Save Model…", command=self.save_model)
        self.btn_save.pack(side="left", padx=4, pady=6)
        self.btn_load = ttk.Button(mdl, text="Load Model…", command=self.load_model)
        self.btn_load.pack(side="left", padx=4, pady=6)
        self.model_status = tk.StringVar(value="No model initialised.")
        ttk.Label(mdl, textvariable=self.model_status).pack(side="left", padx=12)

        bar = ttk.Frame(f)
        bar.pack(fill="x", padx=8, pady=(0, 8))
        self.btn_start = ttk.Button(bar, text="Start Training", command=self.start_training)
        self.btn_start.pack(side="left")
        self.btn_stop = ttk.Button(bar, text="Stop", command=self.stop_training, state="disabled")
        self.btn_stop.pack(side="left", padx=6)

        self.progress = ttk.Progressbar(f, orient="horizontal", mode="determinate", maximum=100)
        self.progress.pack(fill="x", padx=8, pady=(0, 8))

        box = ttk.Frame(f)
        box.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        sb = ttk.Scrollbar(box, orient="vertical")
        self.log_text = tk.Text(box, height=20, wrap="word", yscrollcommand=sb.set,
                                state="disabled", background="#101418", foreground="#d6e2ea",
                                insertbackground="#d6e2ea")
        sb.config(command=self.log_text.yview)
        self.log_text.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

    # -------------------------------------------------------- GENERATE TAB
    def _build_generate_tab(self):
        f = self.tab_gen

        top = ttk.LabelFrame(f, text="Priming prompt")
        top.pack(fill="x", padx=10, pady=8)
        self.prompt_text = tk.Text(top, height=4, wrap="word")
        self.prompt_text.pack(fill="x", padx=6, pady=6)
        self.prompt_text.insert("1.0", "Super")

        prompt_bar = ttk.Frame(top)
        prompt_bar.pack(fill="x", padx=6, pady=(0, 6))
        ttk.Button(prompt_bar, text="Prime (inspect)",
                   command=self.prime_prompt).pack(side="left")
        ttk.Label(prompt_bar, foreground="#666",
                  text="Feeds prompt through model: token count, perplexity, next-token guesses."
                  ).pack(side="left", padx=8)

        smp = ttk.LabelFrame(f, text="Sampling")
        smp.pack(fill="x", padx=10, pady=8)

        self.var_mode = tk.StringVar(value="top-k")
        self.var_k = tk.IntVar(value=40)
        self.var_p = tk.DoubleVar(value=0.9)
        self.var_temp = tk.DoubleVar(value=0.8)
        self.var_maxnew = tk.IntVar(value=300)
        self.var_samples = tk.IntVar(value=5)

        ttk.Label(smp, text="Mode").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        ttk.Combobox(smp, textvariable=self.var_mode, state="readonly",
                     values=["top-k", "top-p"], width=10
                     ).grid(row=0, column=1, sticky="w", padx=6, pady=4)
        ttk.Label(smp, text="k").grid(row=0, column=2, sticky="w", padx=6, pady=4)
        ttk.Spinbox(smp, from_=1, to=10000, increment=1,
                    textvariable=self.var_k, width=8
                    ).grid(row=0, column=3, sticky="w", padx=6, pady=4)

        ttk.Label(smp, text="p").grid(row=1, column=0, sticky="w", padx=6, pady=4)
        ttk.Spinbox(smp, from_=0.05, to=1.0, increment=0.05,
                    textvariable=self.var_p, width=10
                    ).grid(row=1, column=1, sticky="w", padx=6, pady=4)
        ttk.Label(smp, text="Temperature").grid(row=1, column=2, sticky="w", padx=6, pady=4)
        ttk.Spinbox(smp, from_=0.01, to=5.0, increment=0.05,
                    textvariable=self.var_temp, width=8
                    ).grid(row=1, column=3, sticky="w", padx=6, pady=4)

        ttk.Label(smp, text="New tokens").grid(row=2, column=0, sticky="w", padx=6, pady=4)
        ttk.Spinbox(smp, from_=1, to=20000, increment=50,
                    textvariable=self.var_maxnew, width=10
                    ).grid(row=2, column=1, sticky="w", padx=6, pady=4)

        # ---- samples row (only shown when CSV mode is active) ------------
        self.samples_label = ttk.Label(smp, text="Samples")
        self.samples_label.grid(row=2, column=2, sticky="w", padx=6, pady=4)
        self.samples_spin = ttk.Spinbox(smp, from_=1, to=1000, increment=1,
                                        textvariable=self.var_samples, width=8)
        self.samples_spin.grid(row=2, column=3, sticky="w", padx=6, pady=4)

        self.samples_note = ttk.Label(
            smp, foreground="#666",
            text="Samples = number of independent generations.  Each sample is "
                 "seeded with <start> + prompt\nand stops at <end>;  <start> is "
                 "masked so it can never re-appear mid-sample.")
        self.samples_note.grid(row=3, column=0, columnspan=4, sticky="w",
                               padx=6, pady=(2, 4))

        self.btn_gen = ttk.Button(smp, text="Generate", command=self.generate_text)
        self.btn_gen.grid(row=4, column=3, sticky="e", padx=6, pady=4)

        # hide samples UI by default (until csv_mode is confirmed)
        self._hide_samples_widgets()

        out = ttk.LabelFrame(f, text="Output (live)")
        out.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        sb = ttk.Scrollbar(out, orient="vertical")
        self.output_text = tk.Text(out, wrap="word", yscrollcommand=sb.set, height=14)
        sb.config(command=self.output_text.yview)
        self.output_text.pack(side="left", fill="both", expand=True, padx=6, pady=6)
        sb.pack(side="right", fill="y")

    def _hide_samples_widgets(self):
        self.samples_label.grid_remove()
        self.samples_spin.grid_remove()
        self.samples_note.grid_remove()

    def _show_samples_widgets(self):
        self.samples_label.grid()
        self.samples_spin.grid()
        self.samples_note.grid()

    def _update_samples_visibility(self):
        csv_mode = (self.tokenizer is not None
                    and START_TOKEN in self.tokenizer.stoi
                    and END_TOKEN in self.tokenizer.stoi)
        if csv_mode:
            self._show_samples_widgets()
        else:
            self._hide_samples_widgets()

    # ------------------------------------------------------- NEURON TAB
    def _build_activation_tab(self):
        f = self.tab_act

        top = ttk.LabelFrame(f, text="Input text")
        top.pack(fill="x", padx=10, pady=8)
        self.act_input = tk.Text(top, height=5, wrap="word")
        self.act_input.pack(fill="x", padx=6, pady=6)
        self.act_input.insert("1.0",
            "The Roman Empire was one of the most powerful empires in history.")

        ctl = ttk.Frame(top)
        ctl.pack(fill="x", padx=6, pady=(0, 6))
        ttk.Label(ctl, text="Neuron index").pack(side="left")
        self.var_neuron = tk.IntVar(value=0)
        ttk.Spinbox(ctl, from_=0, to=100000, increment=1,
                    textvariable=self.var_neuron, width=8
                    ).pack(side="left", padx=6)
        ttk.Button(ctl, text="Visualize",
                   command=self.visualize_activation).pack(side="left", padx=6)
        self.act_info = tk.StringVar(value="")
        ttk.Label(ctl, textvariable=self.act_info, foreground="#444"
                  ).pack(side="left", padx=12)

        out = ttk.LabelFrame(f,
                             text="Activation map  (blue = low, white = mid, green = high)")
        out.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        sb = ttk.Scrollbar(out, orient="vertical")
        self.act_output = tk.Text(out, wrap="word", yscrollcommand=sb.set, height=18,
                                  state="disabled", font=("Courier", 12))
        sb.config(command=self.act_output.yview)
        self.act_output.pack(side="left", fill="both", expand=True, padx=6, pady=6)
        sb.pack(side="right", fill="y")

        ttk.Label(f, foreground="#666",
                  text=("Feed the text character-by-character through the model and "
                        "colour each character by the activation of the chosen neuron "
                        "in the last hidden layer.  Like the visualisations in "
                        "Karpathy's 'Unreasonable Effectiveness of RNNs'.")
                  ).pack(anchor="w", padx=10, pady=(0, 8))

    # ================================================================ FILES
    def add_files(self):
        paths = filedialog.askopenfilenames(
            title="Select text or CSV files",
            filetypes=[("Text & CSV", "*.txt *.csv"),
                       ("Text files", "*.txt"),
                       ("CSV files", "*.csv"),
                       ("All files", "*.*")])
        if paths:
            self._add_paths(paths)

    def add_csv(self):
        paths = filedialog.askopenfilenames(
            title="Select CSV files",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if paths:
            self._add_paths(paths)

    def add_folder(self):
        folder = filedialog.askdirectory(title="Select folder (recursive)")
        if not folder:
            return
        found = []
        for root, _dirs, names in os.walk(folder):
            for n in names:
                found.append(os.path.join(root, n))
        self._add_paths(found)

    def _add_paths(self, paths):
        added = 0
        skipped = 0
        for p in paths:
            if p in self.files:
                continue
            if p.lower().endswith(".csv"):
                if self._add_csv_path(p):
                    added += 1
                else:
                    skipped += 1
            else:
                self.files.append(p)
                added += 1
        self._refresh_list()
        if added or skipped:
            msg = f"Added {added} file(s). Total: {len(self.files)}"
            if skipped:
                msg += f"  ({skipped} CSV(s) skipped)"
            self.log(msg)
        if added:
            self.prepared_sig = None

    def _add_csv_path(self, path):
        try:
            with open(path, "r", encoding="utf-8",
                      errors="ignore", newline="") as fh:
                header = next(csv.reader(fh), None)
        except Exception as e:
            self.log(f"  ! could not read CSV {path}: {e}")
            return False
        if not header:
            self.log(f"  ! CSV {path} has no header row")
            return False
        col = self._pick_csv_column(path, header)
        if col is None:
            return False
        self.files.append(path)
        self.csv_columns[path] = col
        return True

    def remove_selected(self):
        sel = list(self.file_list.curselection())
        removed = 0
        for i in reversed(sel):
            path = self.files[i]
            self.csv_columns.pop(path, None)
            self.file_list.delete(i)
            del self.files[i]
            removed += 1
        self.prepared_sig = None
        self.log(f"Removed {removed} file(s). Remaining: {len(self.files)}")

    def clear_all(self):
        self.files.clear()
        self.csv_columns.clear()
        self.file_list.delete(0, "end")
        self.prepared_sig = None
        self.data_info.set("No data prepared.")
        self.log("Cleared all files.")

    def clear_all_csvs(self):
        keep = [p for p in self.files if p not in self.csv_columns]
        removed = len(self.files) - len(keep)
        self.files = keep
        self.csv_columns.clear()
        self._refresh_list()
        self.prepared_sig = None
        self.log(f"Removed {removed} CSV file(s). Remaining: {len(self.files)}")

    def _refresh_list(self):
        self.file_list.delete(0, "end")
        for p in self.files:
            if p in self.csv_columns:
                self.file_list.insert(
                    "end", f"[CSV · column: {self.csv_columns[p]}]  {p}")
            else:
                self.file_list.insert("end", p)

    def _on_file_double_click(self, _event):
        sel = self.file_list.curselection()
        if not sel:
            return
        idx = sel[0]
        path = self.files[idx]
        if path not in self.csv_columns:
            return
        try:
            with open(path, "r", encoding="utf-8",
                      errors="ignore", newline="") as fh:
                header = next(csv.reader(fh), None)
        except Exception:
            return
        if not header:
            return
        col = self._pick_csv_column(path, header,
                                    default=self.csv_columns.get(path))
        if col is not None and col != self.csv_columns[path]:
            self.csv_columns[path] = col
            self._refresh_list()
            self.prepared_sig = None
            self.log(f"Changed column for {os.path.basename(path)} to '{col}'.")

    def _pick_csv_column(self, path, header, default=None):
        result = [None]
        dlg = tk.Toplevel(self)
        dlg.title("Pick CSV column")
        dlg.geometry("440x200")
        dlg.transient(self)
        dlg.grab_set()

        ttk.Label(dlg, text=f"File: {os.path.basename(path)}"
                  ).pack(padx=10, pady=(10, 2), anchor="w")
        ttk.Label(dlg, text="Column to use as text sample:"
                  ).pack(padx=10, pady=2, anchor="w")
        var = tk.StringVar(value=default if default in header else header[0])
        ttk.Combobox(dlg, textvariable=var, values=header,
                     state="readonly", width=44
                     ).pack(padx=10, pady=4, fill="x")

        ttk.Label(dlg, foreground="#666",
                  text="Each row's value becomes <start>text<end>."
                  ).pack(padx=10, anchor="w")

        def ok():
            result[0] = var.get()
            dlg.destroy()

        btn = ttk.Frame(dlg)
        btn.pack(pady=10)
        ttk.Button(btn, text="OK", command=ok).pack(side="left", padx=4)
        ttk.Button(btn, text="Cancel", command=dlg.destroy
                   ).pack(side="left", padx=4)
        dlg.wait_window()
        return result[0]

    # ============================================================== PREPARE
    def _data_signature(self):
        return (self.var_tok.get(),
                self._safe_int(self.var_bpe_vocab, 2000),
                tuple(self.files),
                tuple(sorted(self.csv_columns.items())))

    def prepare_data(self):
        if not self.files:
            messagebox.showwarning("No data",
                                   "Add some .txt or .csv files first.")
            return
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("Busy", "A background task is still running.")
            return

        tok_type = self.var_tok.get()
        bpe_vocab = self._safe_int(self.var_bpe_vocab, 2000)
        files = list(self.files)
        csv_columns = dict(self.csv_columns)
        self.prepared_sig = self._data_signature()

        self.data_info.set("Preparing…")
        self.worker = threading.Thread(
            target=self._prepare_worker,
            args=(files, csv_columns, tok_type, bpe_vocab),
            daemon=True)
        self.worker.start()

    def _prepare_worker(self, files, csv_columns, tok_type, bpe_vocab):
        try:
            parts = []
            csv_sample_count = 0
            for p in files:
                try:
                    if p in csv_columns:
                        col = csv_columns[p]
                        with open(p, "r", encoding="utf-8",
                                  errors="ignore", newline="") as fh:
                            reader = csv.reader(fh)
                            header = next(reader, None)
                            if not header:
                                self._q("log", f"  ! {p}: no header")
                                continue
                            if col not in header:
                                self._q("log",
                                        f"  ! {p}: column '{col}' not found")
                                continue
                            col_idx = header.index(col)
                            local = 0
                            for row in reader:
                                if len(row) <= col_idx:
                                    continue
                                text = row[col_idx].strip()
                                text = text.replace(START_TOKEN, "") \
                                           .replace(END_TOKEN, "")
                                if text:
                                    parts.append(
                                        f"{START_TOKEN}{text}{END_TOKEN}")
                                    local += 1
                            csv_sample_count += local
                            self._q("log",
                                    f"  CSV {os.path.basename(p)} "
                                    f"['{col}']: {local} samples")
                    else:
                        with open(p, "r", encoding="utf-8",
                                  errors="ignore") as fh:
                            parts.append(fh.read())
                except Exception as e:
                    self._q("log", f"  ! skipped {p}: {e}")

            if csv_sample_count:
                self._q("log",
                        f"Total CSV samples appended: {csv_sample_count}")

            text = "\n".join(parts)
            if not text.strip():
                self._q("error",
                        "The selected inputs contain no readable text.")
                return

            self._q("log", f"Corpus: {len(text):,} characters.")

            special = [START_TOKEN, END_TOKEN] if csv_sample_count else []

            if tok_type == "BPE":
                tok = BPETokenizer(bpe_vocab)
                tok.train(text, bpe_vocab, special_tokens=special,
                          progress_cb=lambda n: self._q("log",
                                                        f"  … {n} merges"))
            else:
                tok = CharTokenizer()
                tok.train(text, special_tokens=special)

            self._q("log",
                    f"Tokenizer built: {tok.name}, vocab = {tok.vocab_size}")

            ids = tok.encode(text)
            self._q("log", f"Encoded corpus: {len(ids):,} tokens.")

            self.tokenizer = tok
            self.data_tensor = torch.tensor(ids, dtype=torch.long)
            self._q("data_ready",
                    (tok.name, tok.vocab_size, len(ids), len(text),
                     csv_sample_count))
        except Exception:
            self._q("error", traceback.format_exc())

    # ======================================================== MODEL MANAGEMENT
    def init_model(self):
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("Busy", "Wait for the current task to finish.")
            return
        if self.tokenizer is None:
            messagebox.showwarning("No tokenizer", "Prepare the dataset first.")
            return

        cfg = dict(
            rnn=self.var_rnn.get(),
            embed=max(1, self._safe_int(self.var_embed, 128)),
            hidden=max(1, self._safe_int(self.var_hidden, 256)),
            layers=max(1, self._safe_int(self.var_layers, 2)),
            dropout=max(0.0, min(0.9, self._safe_float(self.var_dropout, 0.2))),
        )
        model = RNNLanguageModel(
            self.tokenizer.vocab_size,
            cfg["embed"], cfg["hidden"], cfg["layers"],
            cfg["rnn"], cfg["dropout"]
        ).to(self.device)

        self.model = model
        with self.inference_lock:
            self.inference_model = copy.deepcopy(model)
        self.optimizer = None

        n_params = sum(p.numel() for p in model.parameters())
        self.model_status.set(
            f"{cfg['rnn']} | vocab {self.tokenizer.vocab_size} | "
            f"embed {cfg['embed']} | hidden {cfg['hidden']} | "
            f"layers {cfg['layers']} | params {n_params:,}")
        self.status.set(f"Device: {self.device}  |  model initialised")
        self.log(f"Model initialised: {cfg['rnn']}  |  params {n_params:,}")

        # tokenizer already exists; make sure samples visibility is correct
        self._update_samples_visibility()

    def save_model(self):
        if self.inference_model is None:
            messagebox.showwarning("No model",
                                   "Initialise or load a model first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save model checkpoint", defaultextension=".pt",
            filetypes=[("PyTorch checkpoint", "*.pt"), ("All files", "*.*")])
        if not path:
            return
        try:
            with self.inference_lock:
                model_state = {k: v.detach().clone()
                               for k, v in self.inference_model.state_dict().items()}
            ckpt = {
                "format": 4,
                "model_cfg": self.inference_model.config(),
                "model_state": model_state,
                "tokenizer_type": self.tokenizer.name if self.tokenizer else None,
                "tokenizer_state": (self.tokenizer.state_dict()
                                    if self.tokenizer else None),
                "data_tensor": (self.data_tensor.clone()
                                if self.data_tensor is not None else None),
                "files": list(self.files),
                "csv_columns": dict(self.csv_columns),
            }
            torch.save(ckpt, path)
            self.log(f"Model saved to {path}")
            self.status.set(f"Saved checkpoint → {os.path.basename(path)}")
        except Exception:
            messagebox.showerror("Save failed", traceback.format_exc())

    def load_model(self):
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("Busy", "Cannot load while a task is running.")
            return
        path = filedialog.askopenfilename(
            title="Load model checkpoint",
            filetypes=[("PyTorch checkpoint", "*.pt"), ("All files", "*.*")])
        if not path:
            return
        try:
            ckpt = torch.load(path, map_location=self.device)
            if "model_state" not in ckpt or "model_cfg" not in ckpt:
                messagebox.showerror("Load failed", "Not a valid checkpoint.")
                return

            tok_type = ckpt.get("tokenizer_type")
            tok_state = ckpt.get("tokenizer_state")
            if tok_type == "Character" and tok_state:
                tok = CharTokenizer()
                tok.load_state_dict(tok_state)
            elif tok_type == "BPE" and tok_state:
                tok = BPETokenizer(tok_state.get("target_vocab", 2000))
                tok.load_state_dict(tok_state)
            else:
                messagebox.showwarning("No tokenizer",
                    "This checkpoint doesn't include a tokenizer.")
                return

            mcfg = ckpt["model_cfg"]
            model = RNNLanguageModel(
                mcfg["vocab_size"], mcfg["embed_dim"], mcfg["hidden_size"],
                mcfg["num_layers"], mcfg["rnn_type"], mcfg["dropout"]
            ).to(self.device)
            model.load_state_dict(ckpt["model_state"])
            model.eval()

            self.tokenizer = tok
            self.model = model
            with self.inference_lock:
                self.inference_model = copy.deepcopy(model)
            self.optimizer = None

            self.data_tensor = ckpt.get("data_tensor")
            if self.data_tensor is not None:
                self.data_info.set(
                    f"Loaded · vocab {tok.vocab_size} · "
                    f"{len(self.data_tensor):,} tokens")
            else:
                self.data_info.set("Loaded tokenizer — re-prepare data to train further.")

            self.files = list(ckpt.get("files") or [])
            self.csv_columns = dict(ckpt.get("csv_columns") or {})
            self._refresh_list()

            self.prepared_sig = (
                tok.name,
                getattr(tok, "target_vocab", 0),
                tuple(self.files),
                tuple(sorted(self.csv_columns.items())),
            )

            self.var_rnn.set(mcfg["rnn_type"])
            self.var_embed.set(mcfg["embed_dim"])
            self.var_hidden.set(mcfg["hidden_size"])
            self.var_layers.set(mcfg["num_layers"])
            self.var_dropout.set(mcfg["dropout"])
            self.var_tok.set(tok.name)
            if isinstance(tok, BPETokenizer):
                self.var_bpe_vocab.set(tok.target_vocab)

            n_params = sum(p.numel() for p in model.parameters())
            self.model_status.set(
                f"{mcfg['rnn_type']} | vocab {mcfg['vocab_size']} | "
                f"embed {mcfg['embed_dim']} | hidden {mcfg['hidden_size']} | "
                f"layers {mcfg['num_layers']} | params {n_params:,}  [loaded]")
            self.status.set(f"Device: {self.device}  |  "
                            f"loaded {os.path.basename(path)}")
            self.log(f"Loaded checkpoint from {path}")

            self._update_samples_visibility()
        except Exception:
            messagebox.showerror("Load failed", traceback.format_exc())

    # ============================================================== TRAINING
    def start_training(self):
        if self.model is None:
            messagebox.showinfo(
                "No model",
                "Click “Initialize Model” (or load a checkpoint) first.")
            return
        if self.tokenizer is None or self.data_tensor is None:
            messagebox.showwarning("No data", "Press “Prepare Dataset” first.")
            return
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("Busy", "Training is already running.")
            return

        if self.prepared_sig is not None and \
                self._data_signature() != self.prepared_sig:
            messagebox.showwarning(
                "Settings changed",
                "The tokenizer / file list / CSV settings changed after the "
                "data was prepared.\nPlease press “Prepare Dataset” again.")
            return

        cfg = dict(
            seq=max(2, self._safe_int(self.var_seq, 64)),
            unroll=max(1, self._safe_int(self.var_unroll, 1)),
            batch=max(1, self._safe_int(self.var_batch, 32)),
            lr=max(1e-6, self._safe_float(self.var_lr, 0.002)),
            epochs=max(1, self._safe_int(self.var_epochs, 5)),
        )

        self.stop_flag.clear()
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.btn_init.configure(state="disabled")
        self.btn_load.configure(state="disabled")
        self.progress["value"] = 0
        self.log("=" * 60)
        self.log(f"Starting training: {cfg}")

        self.worker = threading.Thread(target=self._train_worker,
                                       args=(cfg,), daemon=True)
        self.worker.start()

    def stop_training(self):
        self.stop_flag.set()
        self.btn_stop.configure(state="disabled")
        self.log("Stop requested — finishing current step…")

    def _train_worker(self, cfg):
        try:
            model = self.model
            vocab_size = self.tokenizer.vocab_size
            data = self.data_tensor.clone()

            if len(data) < 64:
                self._q("error",
                        "Not enough data to train on (need ~64+ tokens).")
                return

            batch = min(cfg["batch"], max(1, len(data) // (cfg["seq"] * 2)))
            batch = max(1, batch)

            train_data = batchify(data, batch)
            if train_data.size(0) < cfg["seq"] + 1:
                self._q("error",
                        "Corpus too small for the chosen batch/sequence size.")
                return

            criterion = nn.CrossEntropyLoss()
            if self.optimizer is None:
                self.optimizer = torch.optim.Adam(model.parameters(),
                                                  lr=cfg["lr"])
            else:
                for g in self.optimizer.param_groups:
                    g["lr"] = cfg["lr"]
            optimizer = self.optimizer

            model.train()
            hidden = model.init_hidden(batch, self.device)
            seq_len = cfg["seq"]
            unroll = cfg["unroll"]

            batches_per_epoch = max(1, (train_data.size(0) - 1) // seq_len)
            total_batches = batches_per_epoch * cfg["epochs"]
            done_batches = 0
            sync_counter = 0

            for epoch in range(1, cfg["epochs"] + 1):
                if self.stop_flag.is_set():
                    break
                model.train()
                hidden = model.detach_hidden(hidden)
                epoch_loss, n_batches = 0.0, 0
                i = 0

                while i < train_data.size(0) - 1:
                    if self.stop_flag.is_set():
                        break
                    optimizer.zero_grad(set_to_none=True)
                    accum_loss = 0.0
                    actual = 0

                    for _ in range(unroll):
                        if i >= train_data.size(0) - 1:
                            break
                        x, y = get_batch(train_data, i, seq_len)
                        if x.numel() == 0:
                            break
                        x = x.to(self.device)
                        y = y.to(self.device)
                        logits, hidden = model(x, hidden)
                        loss = criterion(
                            logits.reshape(-1, vocab_size), y.reshape(-1))
                        accum_loss = accum_loss + loss
                        actual += 1
                        i += seq_len

                    if actual == 0:
                        break
                    avg_loss = accum_loss / actual
                    avg_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    hidden = model.detach_hidden(hidden)

                    epoch_loss += avg_loss.item() * actual
                    n_batches += actual
                    done_batches += actual

                    sync_counter += 1
                    if sync_counter >= 10:
                        with self.inference_lock:
                            self.inference_model.load_state_dict(
                                model.state_dict())
                        sync_counter = 0

                    self._q("progress",
                            100.0 * done_batches / total_batches)
                    if done_batches % 10 < actual:
                        self._q("log",
                                f"epoch {epoch}/{cfg['epochs']}  "
                                f"batch {done_batches}/{total_batches}  "
                                f"loss {avg_loss.item():.4f}")

                avg = epoch_loss / max(1, n_batches)
                self._q("log", f"── epoch {epoch} done — avg loss {avg:.4f}")
                self._q("progress",
                        100.0 * done_batches / total_batches)

                with self.inference_lock:
                    self.inference_model.load_state_dict(model.state_dict())

            with self.inference_lock:
                self.inference_model.load_state_dict(model.state_dict())

            self._q("progress", 100.0)
            self._q("train_done", None)
        except Exception:
            self._q("error", traceback.format_exc())
            self._q("train_done", None)

    # ============================================================ GENERATION
    def _snapshot_model(self):
        with self.inference_lock:
            if self.inference_model is None:
                return None
            model = copy.deepcopy(self.inference_model)
        model.eval()
        model.to(self.device)
        return model

    @staticmethod
    def _sample_token(logits, mode, k, p, temp, banned_ids=None):
        if banned_ids:
            logits = logits.clone()
            for bid in banned_ids:
                logits[..., bid] = float("-inf")

        if temp < 0.01:
            return logits.argmax(dim=-1, keepdim=True)

        scaled = logits / temp
        if mode == "top-k":
            kk = min(k, scaled.size(-1))
            v, _ = torch.topk(scaled, kk, dim=-1)
            thresh = v[:, [-1]]
            scaled = torch.where(scaled < thresh,
                                 torch.full_like(scaled, float("-inf")),
                                 scaled)
            probs = F.softmax(scaled, dim=-1)
            return torch.multinomial(probs, 1)
        else:
            probs = F.softmax(scaled, dim=-1)
            sp, si = torch.sort(probs, descending=True, dim=-1)
            cum = torch.cumsum(sp, dim=-1)
            drop = (cum - sp) > p
            sp = sp.masked_fill(drop, 0.0)
            sp = sp / sp.sum(dim=-1, keepdim=True)
            probs = torch.zeros_like(probs).scatter_(1, si, sp)
            return torch.multinomial(probs, 1)

    def _clean_text(self, text):
        # <start>/<end> are technical markers — never show them, even if the
        # model learned to spell them out as ordinary characters.
        text = text.replace(START_TOKEN, "").replace(END_TOKEN, "")
        for st in getattr(self.tokenizer, "special_tokens", []):
            if st not in (START_TOKEN, END_TOKEN):
                text = text.replace(st, "")
        return text

    def _render_generation(self, sample_texts, current_text, num_samples):
        if num_samples <= 1:
            display = (current_text if current_text is not None
                       else (sample_texts[0] if sample_texts else ""))
        else:
            parts = [f"--- Sample {i + 1} ---\n{t}"
                     for i, t in enumerate(sample_texts)]
            if current_text is not None:
                parts.append(
                    f"--- Sample {len(sample_texts) + 1} ---\n{current_text}")
            display = "\n\n".join(parts)
        self.output_text.delete("1.0", "end")
        self.output_text.insert("1.0", display)
        self.output_text.see("end")
        self.update()

    def prime_prompt(self):
        if self.inference_model is None or self.tokenizer is None:
            messagebox.showwarning("No model",
                                   "Initialise or load a model first.")
            return
        prompt = self.prompt_text.get("1.0", "end").rstrip("\n")
        if not prompt:
            messagebox.showinfo("Empty prompt", "Type a prompt first.")
            return

        ids = self.tokenizer.encode(prompt)
        if not ids:
            messagebox.showinfo("Empty encoding",
                                "The prompt encodes to 0 tokens.")
            return

        model = self._snapshot_model()
        vocab = self.tokenizer.vocab_size

        try:
            with torch.no_grad():
                x = torch.tensor([ids], dtype=torch.long, device=self.device)
                logits, _ = model(x)

                ppl = float("nan")
                if len(ids) > 1:
                    target = x[:, 1:]
                    pred = logits[:, :-1, :].reshape(-1, vocab)
                    loss = F.cross_entropy(pred, target.reshape(-1))
                    ppl = torch.exp(loss).item()

                top_p, top_i = torch.topk(
                    F.softmax(logits[0, -1], dim=-1), 8)
                guesses = [(self.tokenizer.decode([i.item()]), p.item())
                           for p, i in zip(top_p, top_i)]

            lines = [f"Prompt encoded to {len(ids)} token(s).",
                     f"Prompt perplexity: {ppl:.2f}", "",
                     "Next-token guesses:"]
            for tok, prob in guesses:
                disp = tok.replace("\n", "\\n") if tok else "∅"
                lines.append(f"  {disp!r:24s} {prob * 100:5.1f}%")
            self.output_text.delete("1.0", "end")
            self.output_text.insert("1.0", "\n".join(lines))
            self.status.set(
                f"Primed prompt · {len(ids)} tokens · ppl {ppl:.2f}")
        except Exception:
            messagebox.showerror("Priming failed", traceback.format_exc())

    def generate_text(self):
        if self._is_generating:
            return
        if self.inference_model is None or self.tokenizer is None:
            messagebox.showwarning("No model",
                                   "Initialise or load a model first.")
            return

        prompt = self.prompt_text.get("1.0", "end").rstrip("\n")
        mode = self.var_mode.get()
        k = max(1, self._safe_int(self.var_k, 40))
        p = min(1.0, max(0.01, self._safe_float(self.var_p, 0.9)))
        temp = max(0.0, self._safe_float(self.var_temp, 0.8))
        max_new = max(1, self._safe_int(self.var_maxnew, 300))

        start_id = self.tokenizer.stoi.get(START_TOKEN)
        end_id = self.tokenizer.stoi.get(END_TOKEN)
        csv_mode = start_id is not None and end_id is not None

        # In non-CSV mode samples are meaningless, so force a single stream.
        num_samples = (max(1, self._safe_int(self.var_samples, 1))
                       if csv_mode else 1)

        csv_in_files = any(fp in self.csv_columns for fp in self.files)
        if csv_in_files and not csv_mode:
            self.log(
                "WARNING: CSVs are loaded but the tokenizer has no "
                "<start>/<end> special tokens (probably an old checkpoint).  "
                "Generation will still stop at the '<end>' string, but "
                "re-preparing the data will give cleaner results.")

        prompt_ids = self.tokenizer.encode(prompt) if prompt else []

        if csv_mode:
            seed_ids = [start_id] + prompt_ids
            stop_ids = {end_id}
            banned_ids = {start_id}
        else:
            seed_ids = prompt_ids
            stop_ids = set()
            banned_ids = set()

        vocab = self.tokenizer.vocab_size

        model = self._snapshot_model()
        if model is None:
            messagebox.showwarning("No model", "No inference model available.")
            return

        self._is_generating = True
        self.btn_gen.configure(state="disabled")
        self.output_text.delete("1.0", "end")
        self._render_generation([], "", num_samples)

        def make_display(gen_ids):
            raw = self.tokenizer.decode(gen_ids)
            if END_TOKEN in raw:
                raw = raw[:raw.index(END_TOKEN)]
            if START_TOKEN in raw:
                raw = raw[:raw.index(START_TOKEN)]
            body = self._clean_text(raw)
            return (prompt + body) if prompt else body

        try:
            completed_texts = []
            for s_idx in range(num_samples):
                with torch.no_grad():
                    if seed_ids:
                        x = torch.tensor([seed_ids], dtype=torch.long,
                                         device=self.device)
                        out, hidden = model(x)
                        logits = out[:, -1, :]
                    else:
                        hidden = model.init_hidden(1, self.device)
                        first = torch.randint(0, vocab, (1, 1),
                                              device=self.device)
                        out, hidden = model(first, hidden)
                        logits = out[:, -1, :]

                    generated = []
                    last_refresh = time.time()

                    for step in range(max_new):
                        nxt = self._sample_token(logits, mode, k, p, temp,
                                                 banned_ids)
                        tok_id = nxt.item()

                        if stop_ids and tok_id in stop_ids:
                            break

                        generated.append(tok_id)
                        out, hidden = model(nxt, hidden)
                        logits = out[:, -1, :]

                        # Fallback: model spelled out "<end>" char-wise.
                        if (step + 1) % 10 == 0:
                            if END_TOKEN in self.tokenizer.decode(generated):
                                break

                        now = time.time()
                        if ((step + 1) % 5 == 0
                                or (now - last_refresh) > 0.15
                                or step == max_new - 1):
                            self._render_generation(
                                completed_texts, make_display(generated),
                                num_samples)
                            last_refresh = now
                            self.status.set(
                                f"Generating… sample {s_idx + 1}/{num_samples}, "
                                f"{step + 1}/{max_new} tokens  |  "
                                f"device={self.device}")

                completed_texts.append(make_display(generated))
                self._render_generation(completed_texts, None, num_samples)

            if num_samples > 1:
                summary = f"Generated {num_samples} samples"
            else:
                summary = (
                    f"Generated "
                    f"{len(completed_texts[0]) if completed_texts else 0} "
                    f"chars")
            self.status.set(
                f"{summary}  |  mode={mode}  k={k} p={p} temp={temp}  |  "
                f"device={self.device}")
        except Exception:
            messagebox.showerror("Generation failed", traceback.format_exc())
        finally:
            self._is_generating = False
            self.btn_gen.configure(state="normal")

    # ============================================================ ACTIVATION
    def visualize_activation(self):
        if self.inference_model is None or self.tokenizer is None:
            messagebox.showwarning("No model",
                                   "Initialise or load a model first.")
            return
        text = self.act_input.get("1.0", "end").rstrip("\n")
        if not text:
            messagebox.showinfo("Empty", "Enter some text first.")
            return

        neuron_idx = max(0, self._safe_int(self.var_neuron, 0))

        ids = self.tokenizer.encode(text)
        if not ids:
            messagebox.showinfo("Empty encoding",
                                "Text encodes to 0 tokens.")
            return

        model = self._snapshot_model()
        hidden_size = model.hidden_size
        if neuron_idx >= hidden_size:
            messagebox.showwarning(
                "Neuron out of range",
                f"Neuron index {neuron_idx} is out of range "
                f"(hidden size = {hidden_size}).")
            return

        try:
            with torch.no_grad():
                x = torch.tensor([ids], dtype=torch.long, device=self.device)
                e = model.drop(model.embed(x))
                rnn_out, _ = model.rnn(e)
                acts = rnn_out[0, :, neuron_idx].tolist()

            pieces = []
            for tok_id in ids:
                piece = self.tokenizer.decode([tok_id])
                pieces.append(piece if piece else "·")

            vmin, vmax = min(acts), max(acts)
            self.act_output.configure(state="normal")
            self.act_output.delete("1.0", "end")
            for i, (piece, act) in enumerate(zip(pieces, acts)):
                tag = f"a{i}"
                color = self._activation_color(act, vmin, vmax)
                self.act_output.tag_configure(tag, background=color)
                self.act_output.insert("end", piece, tag)
            self.act_output.configure(state="disabled")

            top_idx = sorted(range(len(acts)),
                             key=lambda i: acts[i])[-3:][::-1]
            top_parts = []
            for i in top_idx:
                disp = pieces[i].replace("\n", "\\n")
                top_parts.append(f"'{disp}' ({acts[i]:+.3f})")
            top_str = ", ".join(top_parts)

            self.act_info.set(
                f"Neuron {neuron_idx} (last layer): "
                f"[{vmin:+.3f}, {vmax:+.3f}]  top: {top_str}")
        except Exception:
            messagebox.showerror("Visualization failed",
                                 traceback.format_exc())

    @staticmethod
    def _activation_color(value, vmin, vmax):
        if vmax - vmin < 1e-9:
            return "#ffffff"
        t = (value - vmin) / (vmax - vmin)
        if t < 0.5:
            s = t * 2.0
            r = int(180 + 75 * s)
            g = int(200 + 55 * s)
            b = 255
        else:
            s = (t - 0.5) * 2.0
            r = int(255 - 75 * s)
            g = 255
            b = int(255 - 75 * s)
        return f"#{r:02x}{g:02x}{b:02x}"

    # ================================================================ UTILS
    @staticmethod
    def _safe_int(var, default):
        try:
            return int(var.get())
        except Exception:
            return default

    @staticmethod
    def _safe_float(var, default):
        try:
            return float(var.get())
        except Exception:
            return default

    def _q(self, kind, payload):
        self.q.put((kind, payload))

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "log":
                    self.log(payload)
                elif kind == "progress":
                    self.progress["value"] = payload
                elif kind == "error":
                    self.log("ERROR:\n" + payload)
                    last = (payload.strip().splitlines()[-1]
                            if payload.strip() else "Unknown error")
                    messagebox.showerror("Error", last)
                elif kind == "data_ready":
                    name, vocab, ntok, nchar, n_csv = payload
                    suffix = (f" · {n_csv} CSV samples" if n_csv else "")
                    self.data_info.set(
                        f"{name} tokenizer · vocab {vocab} · {ntok:,} tokens "
                        f"from {nchar:,} chars{suffix}")
                    self.status.set("Data ready.")
                    self._update_samples_visibility()
                elif kind == "train_done":
                    self.btn_start.configure(state="normal")
                    self.btn_stop.configure(state="disabled")
                    self.btn_init.configure(state="normal")
                    self.btn_load.configure(state="normal")
                    self.status.set("Training finished."
                                    if not self.stop_flag.is_set()
                                    else "Training stopped.")
                    self.log("Training finished.")
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    def log(self, msg):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", str(msg) + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def destroy(self):
        self.stop_flag.set()
        super().destroy()


# =============================================================================
if __name__ == "__main__":
    App().mainloop()