import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import math
import threading
import queue
import traceback
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path
import time


# ==================== FIXED VOCABULARY ====================
class FixedVocab:
    """Fixed vocabulary of ~500 characters + special tokens."""
    def __init__(self):
        self.special_tokens = ['<PAD>', '<UNK>', '<SOS>', '<EOS>', '<USER>', '<BOT>']
        self.allowed_chars = self._build_allowed_chars()
        self.char_to_idx, self.idx_to_char = self._build_mapping()
        self.pad_idx = self.char_to_idx['<PAD>']
        self.unk_idx = self.char_to_idx['<UNK>']
        self.sos_idx = self.char_to_idx['<SOS>']
        self.eos_idx = self.char_to_idx['<EOS>']
        self.user_idx = self.char_to_idx['<USER>']
        self.bot_idx = self.char_to_idx['<BOT>']
        print(f"Fixed vocabulary size: {len(self.char_to_idx)}")

    def _build_allowed_chars(self):
        chars = set()
        for code in range(32, 127):
            chars.add(chr(code))
        for code in range(0x0400, 0x0500):
            try:
                ch = chr(code)
                if ch.isprintable():
                    chars.add(ch)
            except Exception:
                pass
        for code in range(0x0370, 0x0400):
            try:
                ch = chr(code)
                if ch.isprintable():
                    chars.add(ch)
            except Exception:
                pass
        chars.update(['\n', '\r', '\t'])
        chars.update(['—', '–', '“', '”', '‘', '’', '…', '•'])
        return sorted(chars)

    def _build_mapping(self):
        char_to_idx = {}
        idx_to_char = {}
        for i, tok in enumerate(self.special_tokens):
            char_to_idx[tok] = i
            idx_to_char[i] = tok
        for i, ch in enumerate(self.allowed_chars, start=len(self.special_tokens)):
            char_to_idx[ch] = i
            idx_to_char[i] = ch
        return char_to_idx, idx_to_char

    def encode(self, text):
        return [self.char_to_idx.get(c, self.unk_idx) for c in text]

    def decode(self, ids, skip_special=True):
        chars = []
        for i in ids:
            ch = self.idx_to_char.get(i, '')
            if skip_special and ch in self.special_tokens:
                continue
            chars.append(ch)
        return ''.join(chars)


# ==================== POSITIONAL ENCODING ====================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# ==================== GENERIC RNN SEQ2SEQ (RNN / LSTM / GRU) ====================
class RNNSeq2Seq(nn.Module):
    """Plain (no attention) seq2seq with a stacked RNN/LSTM/GRU encoder & decoder."""
    def __init__(self, vocab_size, rnn_type='GRU', max_len=256, embed_size=128,
                 hidden_size=128, num_encoder_layers=2, num_decoder_layers=2,
                 dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.rnn_type = rnn_type.upper()
        if self.rnn_type not in ('RNN', 'LSTM', 'GRU'):
            raise ValueError(f"Unknown RNN type: {rnn_type}")
        self.max_len = max_len
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)

        rnn_cls = {'RNN': nn.RNN, 'LSTM': nn.LSTM, 'GRU': nn.GRU}[self.rnn_type]
        enc_dropout = dropout if num_encoder_layers > 1 else 0.0
        dec_dropout = dropout if num_decoder_layers > 1 else 0.0

        self.encoder = rnn_cls(embed_size, hidden_size, num_layers=num_encoder_layers,
                               batch_first=True, dropout=enc_dropout)
        self.decoder = rnn_cls(embed_size, hidden_size, num_layers=num_decoder_layers,
                               batch_first=True, dropout=dec_dropout)
        self.fc_out = nn.Linear(hidden_size, vocab_size)

        self._init_weights()
        print(f"{self.rnn_type} Seq2Seq parameters: "
              f"{sum(p.numel() for p in self.parameters()):,}")

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _transfer_single(self, h):
        n_enc = h.size(0)
        n_dec = self.num_decoder_layers
        if n_enc == n_dec:
            return h.contiguous()
        if n_enc > n_dec:
            return h[-n_dec:].contiguous()
        extra = h[-1:].expand(n_dec - n_enc, -1, -1)
        return torch.cat([h, extra], dim=0).contiguous()

    def _transfer_hidden(self, hidden):
        if isinstance(hidden, tuple):  # LSTM
            h, c = hidden
            return (self._transfer_single(h), self._transfer_single(c))
        return self._transfer_single(hidden)

    def forward(self, src, tgt):
        src_emb = self.embedding(src)
        _, enc_hidden = self.encoder(src_emb)
        dec_hidden = self._transfer_hidden(enc_hidden)

        decoder_input = tgt[:, :-1]
        dec_emb = self.embedding(decoder_input)
        output, _ = self.decoder(dec_emb, dec_hidden)
        logits = self.fc_out(output)
        return logits

    @torch.no_grad()
    def generate(self, src, sos_idx, eos_idx, max_len=None, temperature=1.0, callback=None):
        self.eval()
        if max_len is None:
            max_len = self.max_len
        batch_size = src.size(0)

        src_emb = self.embedding(src)
        _, enc_hidden = self.encoder(src_emb)
        dec_hidden = self._transfer_hidden(enc_hidden)

        decoder_input = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=src.device)
        generated = [sos_idx]

        for _ in range(max_len):
            dec_emb = self.embedding(decoder_input)
            output, dec_hidden = self.decoder(dec_emb, dec_hidden)
            logits = self.fc_out(output.squeeze(1)) / max(temperature, 1e-6)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()

            if callback:
                callback(next_token)

            if next_token == eos_idx:
                break
            generated.append(next_token)
            decoder_input = torch.tensor([[next_token]], device=src.device)

        return generated

    def get_config(self):
        return {
            'arch': 'RNN',
            'rnn_type': self.rnn_type,
            'vocab_size': self.vocab_size,
            'max_len': self.max_len,
            'embed_size': self.embed_size,
            'hidden_size': self.hidden_size,
            'num_encoder_layers': self.num_encoder_layers,
            'num_decoder_layers': self.num_decoder_layers,
            'dropout': self.dropout,
        }

    def save(self, path):
        torch.save({'config': self.get_config(), 'state_dict': self.state_dict()}, path)

    @classmethod
    def load(cls, path):
        checkpoint = torch.load(path, map_location='cpu')
        config = checkpoint['config']
        model = cls(**{k: v for k, v in config.items() if k != 'arch'})
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        return model


# ==================== FEED-FORWARD (FFN) SEQ2SEQ ====================
class FNNSeq2Seq(nn.Module):
    """
    Purely feed-forward encoder-decoder.
      Encoder: masked mean of input embeddings -> MLP -> context vector
      Decoder: per-position MLP over (context, previous token embedding)
    """
    def __init__(self, vocab_size, max_len=256, embed_size=128,
                 hidden_layers=None, dropout=0.1):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = []
        hidden_layers = list(hidden_layers)

        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embed_size = embed_size
        self.hidden_layers = hidden_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)

        enc_layers = []
        in_dim = embed_size
        for h in hidden_layers:
            enc_layers.append(nn.Linear(in_dim, h))
            enc_layers.append(nn.ReLU())
            enc_layers.append(nn.Dropout(dropout))
            in_dim = h
        if not hidden_layers:
            enc_layers.append(nn.Identity())
            context_size = embed_size
        else:
            context_size = hidden_layers[-1]
        self.encoder = nn.Sequential(*enc_layers)
        self.context_size = context_size

        dec_layers = []
        in_dim = embed_size + context_size
        for h in hidden_layers:
            dec_layers.append(nn.Linear(in_dim, h))
            dec_layers.append(nn.ReLU())
            dec_layers.append(nn.Dropout(dropout))
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, vocab_size))
        self.decoder = nn.Sequential(*dec_layers)

        self._init_weights()
        print(f"FFN Seq2Seq parameters: {sum(p.numel() for p in self.parameters()):,}")

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _encode(self, src):
        src_mask = (src != 0).unsqueeze(-1).float()
        src_emb = self.embedding(src)
        summed = (src_emb * src_mask).sum(dim=1)
        counts = src_mask.sum(dim=1).clamp(min=1.0)
        pooled = summed / counts
        return self.encoder(pooled)

    def forward(self, src, tgt):
        context = self._encode(src)
        decoder_input = tgt[:, :-1]
        dec_emb = self.embedding(decoder_input)
        tlen = decoder_input.size(1)
        context_exp = context.unsqueeze(1).expand(-1, tlen, -1)
        dec_in = torch.cat([dec_emb, context_exp], dim=-1)
        logits = self.decoder(dec_in)
        return logits

    @torch.no_grad()
    def generate(self, src, sos_idx, eos_idx, max_len=None, temperature=1.0, callback=None):
        self.eval()
        if max_len is None:
            max_len = self.max_len
        batch_size = src.size(0)
        context = self._encode(src)

        prev_token = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=src.device)
        generated = [sos_idx]

        for _ in range(max_len):
            prev_emb = self.embedding(prev_token)
            dec_in = torch.cat([prev_emb, context.unsqueeze(1)], dim=-1)
            logits = self.decoder(dec_in).squeeze(1) / max(temperature, 1e-6)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()

            if callback:
                callback(next_token)

            if next_token == eos_idx:
                break
            generated.append(next_token)
            prev_token = torch.tensor([[next_token]], device=src.device)

        return generated

    def get_config(self):
        return {
            'arch': 'FFN',
            'vocab_size': self.vocab_size,
            'max_len': self.max_len,
            'embed_size': self.embed_size,
            'hidden_layers': list(self.hidden_layers),
            'dropout': self.dropout,
        }

    def save(self, path):
        torch.save({'config': self.get_config(), 'state_dict': self.state_dict()}, path)

    @classmethod
    def load(cls, path):
        checkpoint = torch.load(path, map_location='cpu')
        config = checkpoint['config']
        model = cls(**{k: v for k, v in config.items() if k != 'arch'})
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        return model


# ==================== TRANSFORMER SEQ2SEQ ====================
class TransformerSeq2Seq(nn.Module):
    def __init__(self, vocab_size, max_len=256, embed_size=128, nhead=8,
                 num_encoder_layers=4, num_decoder_layers=4, dim_feedforward=512,
                 dropout=0.1, activation='gelu'):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embed_size = embed_size
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation

        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.pos_encoder = PositionalEncoding(embed_size, max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation=activation, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation=activation, batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.fc_out = nn.Linear(embed_size, vocab_size)
        self._init_weights()
        print(f"Transformer Seq2Seq parameters: "
              f"{sum(p.numel() for p in self.parameters()):,}")

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_padding_mask(self, x):
        return (x == 0)

    def generate_causal_mask(self, sz):
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

    def forward(self, src, tgt):
        decoder_input = tgt[:, :-1]

        src_pad_mask = self.generate_padding_mask(src)
        tgt_pad_mask = self.generate_padding_mask(decoder_input)
        tgt_causal_mask = self.generate_causal_mask(decoder_input.size(1)).to(src.device)

        src_emb = self.pos_encoder(self.embedding(src) * math.sqrt(self.embed_size))
        tgt_emb = self.pos_encoder(self.embedding(decoder_input) * math.sqrt(self.embed_size))

        memory = self.encoder(src_emb, src_key_padding_mask=src_pad_mask)
        output = self.decoder(
            tgt_emb, memory,
            tgt_mask=tgt_causal_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask,
        )
        return self.fc_out(output)

    @torch.no_grad()
    def generate(self, src, sos_idx, eos_idx, max_len=None, temperature=1.0, callback=None):
        self.eval()
        if max_len is None:
            max_len = self.max_len
        batch_size = src.size(0)

        src_pad_mask = self.generate_padding_mask(src)
        src_emb = self.pos_encoder(self.embedding(src) * math.sqrt(self.embed_size))
        memory = self.encoder(src_emb, src_key_padding_mask=src_pad_mask)

        tgt = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=src.device)
        generated = [sos_idx]

        for _ in range(max_len):
            tgt_len = tgt.size(1)
            tgt_causal_mask = self.generate_causal_mask(tgt_len).to(src.device)
            tgt_pad_mask = (tgt == 0)

            tgt_emb = self.pos_encoder(self.embedding(tgt) * math.sqrt(self.embed_size))
            output = self.decoder(
                tgt_emb, memory,
                tgt_mask=tgt_causal_mask,
                tgt_key_padding_mask=tgt_pad_mask,
                memory_key_padding_mask=src_pad_mask,
            )
            logits = self.fc_out(output[:, -1:, :]) / max(temperature, 1e-6)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs.squeeze(1), 1).item()

            if callback:
                callback(next_token)

            if next_token == eos_idx:
                break
            generated.append(next_token)
            tgt = torch.cat([tgt, torch.tensor([[next_token]], device=src.device)], dim=1)

        return generated

    def get_config(self):
        return {
            'arch': 'Transformer',
            'vocab_size': self.vocab_size,
            'max_len': self.max_len,
            'embed_size': self.embed_size,
            'nhead': self.nhead,
            'num_encoder_layers': self.num_encoder_layers,
            'num_decoder_layers': self.num_decoder_layers,
            'dim_feedforward': self.dim_feedforward,
            'dropout': self.dropout,
            'activation': self.activation,
        }

    def save(self, path):
        torch.save({'config': self.get_config(), 'state_dict': self.state_dict()}, path)

    @classmethod
    def load(cls, path):
        checkpoint = torch.load(path, map_location='cpu')
        config = checkpoint['config']
        model = cls(**{k: v for k, v in config.items() if k != 'arch'})
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        return model


# ==================== DATASET ====================
class WordPairDataset(Dataset):
    def __init__(self, pairs, vocab, max_len=256):
        self.pairs = pairs
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        input_word, output_word = self.pairs[idx]

        input_ids = self.vocab.encode(input_word)
        if len(input_ids) > self.max_len:
            input_ids = input_ids[:self.max_len]
        input_ids += [self.vocab.pad_idx] * (self.max_len - len(input_ids))

        output_ids = [self.vocab.sos_idx] + self.vocab.encode(output_word) + [self.vocab.eos_idx]
        if len(output_ids) > self.max_len:
            output_ids = output_ids[:self.max_len - 1] + [self.vocab.eos_idx]
        output_ids += [self.vocab.pad_idx] * (self.max_len - len(output_ids))

        return (torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(output_ids, dtype=torch.long))


# ==================== DATA LOADING HELPERS ====================
def parse_text_pairs(text):
    pairs = []
    for line in text.strip().split('\n'):
        parts = line.strip().split()
        if len(parts) >= 2:
            pairs.append((parts[0], parts[1]))
    return pairs


def load_pairs_from_csv(file_path, input_col, output_col, has_header=True, delimiter=','):
    if file_path.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file_path, header=0 if has_header else None, dtype=str)
    else:
        df = pd.read_csv(file_path, delimiter=delimiter, header=0 if has_header else None, dtype=str)
    pairs = []
    for _, row in df.iterrows():
        inp = str(row[input_col]) if input_col in df.columns else str(row.iloc[int(input_col)])
        out = str(row[output_col]) if output_col in df.columns else str(row.iloc[int(output_col)])
        pairs.append((inp.strip(), out.strip()))
    return pairs


def load_pairs_from_xml(file_path, input_tag, output_tag):
    tree = ET.parse(file_path)
    root = tree.getroot()
    pairs = []
    for elem in root.findall('.//' + input_tag):
        parent = elem.getparent()
        out_elem = parent.find(output_tag)
        if out_elem is not None:
            pairs.append((elem.text.strip() if elem.text else '',
                          out_elem.text.strip() if out_elem.text else ''))
    return pairs


# ==================== PRESETS ====================
def get_preset(size, arch):
    """Return a dict of settings for a given (size, arch) combination."""
    if arch in ('RNN', 'LSTM', 'GRU'):
        family = 'RNN'
    else:
        family = arch

    presets = {
        ('Small', 'Transformer'): dict(embed_size=64,  nhead=4,  dim_ff=256, enc=3, dec=3),
        ('Medium', 'Transformer'): dict(embed_size=128, nhead=8,  dim_ff=512, enc=4, dec=4),
        ('Large', 'Transformer'): dict(embed_size=192, nhead=12, dim_ff=768, enc=6, dec=6),
        ('Small', 'RNN'):          dict(embed_size=64,  hidden=64,  enc=2, dec=2),
        ('Medium', 'RNN'):         dict(embed_size=128, hidden=128, enc=3, dec=3),
        ('Large', 'RNN'):          dict(embed_size=192, hidden=192, enc=4, dec=4),
        ('Small', 'FFN'):          dict(embed_size=64,  hidden_layers='256,256'),
        ('Medium', 'FFN'):         dict(embed_size=128, hidden_layers='512,512'),
        ('Large', 'FFN'):          dict(embed_size=192, hidden_layers='768,768'),
    }
    return presets.get((size, family), {})


# ==================== TRAINING ====================
def train_model(pairs, vocab, model, epochs=100, batch_size=16, lr=0.0003, device='cpu'):
    max_len = model.max_len
    dataset = WordPairDataset(pairs, vocab, max_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            output = model(src, tgt)
            targets = tgt[:, 1:].contiguous()
            loss = criterion(output.reshape(-1, len(vocab.char_to_idx)), targets.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(dataloader), 1)

        if epoch % 10 == 0:
            model.eval()
            test_results = []
            for inp_word, exp_word in pairs[:min(3, len(pairs))]:
                src_ids = vocab.encode(inp_word)
                if len(src_ids) > max_len:
                    src_ids = src_ids[:max_len]
                src_tensor = torch.tensor(
                    [src_ids + [vocab.pad_idx] * (max_len - len(src_ids))],
                    dtype=torch.long, device=device)
                gen_ids = model.generate(src_tensor, vocab.sos_idx, vocab.eos_idx, max_len)
                result = vocab.decode(gen_ids[1:])
                test_results.append(f"{inp_word}->{result}")
            yield epoch + 1, avg_loss, f" | Test: {' '.join(test_results)}"
        else:
            yield epoch + 1, avg_loss, ""

    yield None, None, ""


# ==================== CSV IMPORT DIALOG ====================
class CSVImportDialog(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.title("Import from CSV/Excel")
        self.geometry("650x450")
        self.result = None

        self.file_path = tk.StringVar()
        self.delimiter = tk.StringVar(value=",")
        self.has_header = tk.BooleanVar(value=True)
        self.sheet_name = tk.StringVar()
        self.input_col = tk.StringVar()
        self.output_col = tk.StringVar()
        self.df = None

        self.setup_ui()
        self.transient(parent)
        self.grab_set()

    def setup_ui(self):
        main = ttk.Frame(self, padding="10")
        main.pack(fill='both', expand=True)

        ttk.Label(main, text="File:").grid(row=0, column=0, sticky='w', pady=5)
        ttk.Entry(main, textvariable=self.file_path, width=50).grid(row=0, column=1, padx=5, sticky='ew')
        ttk.Button(main, text="Browse", command=self.browse).grid(row=0, column=2, padx=5)
        main.columnconfigure(1, weight=1)

        ttk.Label(main, text="Delimiter (CSV):").grid(row=1, column=0, sticky='w', pady=5)
        self.delimiter_combo = ttk.Combobox(main, textvariable=self.delimiter,
                                            values=[",", ";", "\t", "|"], width=10)
        self.delimiter_combo.grid(row=1, column=1, sticky='w', padx=5)

        ttk.Checkbutton(main, text="File has header row", variable=self.has_header,
                        command=self.load_preview).grid(row=2, column=1, sticky='w', pady=5)

        ttk.Label(main, text="Excel sheet:").grid(row=3, column=0, sticky='w', pady=5)
        self.sheet_combo = ttk.Combobox(main, textvariable=self.sheet_name, width=30)
        self.sheet_combo.grid(row=3, column=1, sticky='w', padx=5)
        ttk.Button(main, text="Load sheets", command=self.load_sheets).grid(row=3, column=2)

        ttk.Button(main, text="Load Preview", command=self.load_preview).grid(row=4, column=1, pady=10)

        col_frame = ttk.LabelFrame(main, text="Select Columns", padding="10")
        col_frame.grid(row=5, column=0, columnspan=3, sticky='ew', pady=10)
        col_frame.columnconfigure(1, weight=1)
        col_frame.columnconfigure(3, weight=1)

        ttk.Label(col_frame, text="Input column (context):").grid(row=0, column=0, sticky='w')
        self.input_combo = ttk.Combobox(col_frame, textvariable=self.input_col, width=20)
        self.input_combo.grid(row=0, column=1, padx=5, sticky='ew')

        ttk.Label(col_frame, text="Output column (response):").grid(row=0, column=2, sticky='w', padx=(20, 0))
        self.output_combo = ttk.Combobox(col_frame, textvariable=self.output_col, width=20)
        self.output_combo.grid(row=0, column=3, padx=5, sticky='ew')

        preview_frame = ttk.LabelFrame(main, text="Preview", padding="5")
        preview_frame.grid(row=6, column=0, columnspan=3, sticky='nsew', pady=10)
        main.rowconfigure(6, weight=1)

        self.preview_text = scrolledtext.ScrolledText(preview_frame, height=8, font=('Courier', 9))
        self.preview_text.pack(fill='both', expand=True)

        btn_frame = ttk.Frame(main)
        btn_frame.grid(row=7, column=0, columnspan=3, pady=10)
        ttk.Button(btn_frame, text="Import", command=self.import_data).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Cancel", command=self.destroy).pack(side='left')

    def browse(self):
        filename = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx *.xls"), ("All files", "*.*")])
        if filename:
            self.file_path.set(filename)
            self.load_preview()

    def load_sheets(self):
        try:
            xl = pd.ExcelFile(self.file_path.get())
            self.sheet_combo['values'] = xl.sheet_names
            if xl.sheet_names:
                self.sheet_name.set(xl.sheet_names[0])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read sheets: {e}")

    def load_preview(self):
        path = self.file_path.get()
        if not path:
            return
        try:
            if path.endswith(('.xlsx', '.xls')):
                sheet = self.sheet_name.get() if self.sheet_name.get() else 0
                self.df = pd.read_excel(path, sheet_name=sheet,
                                        header=0 if self.has_header.get() else None, dtype=str)
            else:
                delim = self.delimiter.get()
                self.df = pd.read_csv(path, delimiter=delim,
                                      header=0 if self.has_header.get() else None, dtype=str)

            cols = list(self.df.columns) if self.has_header.get() \
                else [str(i) for i in range(len(self.df.columns))]
            self.input_combo['values'] = cols
            self.output_combo['values'] = cols
            if cols:
                self.input_col.set(cols[0])
                self.output_col.set(cols[1] if len(cols) > 1 else cols[0])

            self.preview_text.delete('1.0', 'end')
            self.preview_text.insert('1.0', self.df.head(10).to_string(index=False))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {e}")

    def import_data(self):
        if self.df is None or not self.input_col.get() or not self.output_col.get():
            messagebox.showerror("Error", "Please load a file and select columns.")
            return
        try:
            input_idx = self.input_col.get()
            output_idx = self.output_col.get()
            if not self.has_header.get():
                input_idx = int(input_idx)
                output_idx = int(output_idx)
            pairs = []
            for _, row in self.df.iterrows():
                inp = str(row[input_idx]) if isinstance(input_idx, str) else str(row.iloc[input_idx])
                out = str(row[output_idx]) if isinstance(output_idx, str) else str(row.iloc[output_idx])
                pairs.append((inp.strip(), out.strip()))
            self.result = pairs
            self.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"Import failed: {e}")


# ==================== XML IMPORT DIALOG ====================
class XMLImportDialog(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.title("Import from XML")
        self.geometry("500x300")
        self.result = None

        self.file_path = tk.StringVar()
        self.input_tag = tk.StringVar(value="input")
        self.output_tag = tk.StringVar(value="output")

        self.setup_ui()
        self.transient(parent)
        self.grab_set()

    def setup_ui(self):
        main = ttk.Frame(self, padding="10")
        main.pack(fill='both', expand=True)

        ttk.Label(main, text="XML File:").grid(row=0, column=0, sticky='w', pady=5)
        ttk.Entry(main, textvariable=self.file_path, width=40).grid(row=0, column=1, padx=5)
        ttk.Button(main, text="Browse", command=self.browse).grid(row=0, column=2)

        ttk.Label(main, text="Input tag name:").grid(row=1, column=0, sticky='w', pady=5)
        ttk.Entry(main, textvariable=self.input_tag, width=30).grid(row=1, column=1, padx=5, sticky='w')

        ttk.Label(main, text="Output tag name:").grid(row=2, column=0, sticky='w', pady=5)
        ttk.Entry(main, textvariable=self.output_tag, width=30).grid(row=2, column=1, padx=5, sticky='w')

        self.preview_text = scrolledtext.ScrolledText(main, height=10, font=('Courier', 9))
        self.preview_text.grid(row=3, column=0, columnspan=3, sticky='nsew', pady=10)
        main.rowconfigure(3, weight=1)
        main.columnconfigure(1, weight=1)

        ttk.Button(main, text="Load Preview", command=self.load_preview).grid(row=4, column=1, pady=5)

        btn_frame = ttk.Frame(main)
        btn_frame.grid(row=5, column=0, columnspan=3, pady=10)
        ttk.Button(btn_frame, text="Import", command=self.import_data).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Cancel", command=self.destroy).pack(side='left')

    def browse(self):
        filename = filedialog.askopenfilename(filetypes=[("XML files", "*.xml"), ("All files", "*.*")])
        if filename:
            self.file_path.set(filename)

    def _parse(self):
        tree = ET.parse(self.file_path.get())
        root = tree.getroot()
        pairs = []
        for elem in root.findall('.//' + self.input_tag.get()):
            parent = elem.getparent()
            out_elem = parent.find(self.output_tag.get())
            if out_elem is not None:
                pairs.append((elem.text.strip() if elem.text else '',
                              out_elem.text.strip() if out_elem.text else ''))
        return pairs

    def load_preview(self):
        if not self.file_path.get():
            return
        try:
            pairs = self._parse()
            preview = "\n".join([f"{i}: {p[0]} -> {p[1]}" for i, p in enumerate(pairs[:10])])
            self.preview_text.delete('1.0', 'end')
            self.preview_text.insert('1.0', preview)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to parse XML: {e}")

    def import_data(self):
        if not self.file_path.get():
            messagebox.showerror("Error", "Please select a file.")
            return
        try:
            self.result = self._parse()
            self.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"Import failed: {e}")


# ==================== MAIN UI ====================
class TransformerPredictorUI:
    ARCHS = ["Transformer", "GRU", "LSTM", "RNN", "FFN"]

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Talk to a Neural Network")
        self.root.geometry("1100x920")

        self.vocab = FixedVocab()

        self.model = None
        self.pairs = []
        self.training_thread = None
        self.stop_training = False
        self.queue = queue.Queue()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.chat_history = []

        self.setup_ui()
        self.root.after(100, self.process_queue)

    # ------------------------------------------------------------------ UI
    def setup_ui(self):
        main = ttk.Frame(self.root, padding="10")
        main.pack(fill='both', expand=True)
        main.columnconfigure(0, weight=1)
        main.rowconfigure(0, weight=1)

        notebook = ttk.Notebook(main)
        notebook.grid(row=0, column=0, sticky='nsew')

        self.data_tab = ttk.Frame(notebook, padding="10")
        notebook.add(self.data_tab, text="Data")
        self.setup_data_tab()

        self.train_tab = ttk.Frame(notebook, padding="10")
        notebook.add(self.train_tab, text="Train")
        self.setup_train_tab()

        self.test_tab = ttk.Frame(notebook, padding="10")
        notebook.add(self.test_tab, text="Test (word)")
        self.setup_test_tab()

        self.chat_tab = ttk.Frame(notebook, padding="10")
        notebook.add(self.chat_tab, text="Chat")
        self.setup_chat_tab()

        self.status_var = tk.StringVar(value=f"Ready (device: {self.device})")
        status = ttk.Label(main, textvariable=self.status_var, relief='sunken', anchor='w')
        status.grid(row=1, column=0, sticky='ew', pady=(5, 0))

    # ------------------------------------------------------------------ DATA TAB
    def setup_data_tab(self):
        list_frame = ttk.LabelFrame(self.data_tab, text="Word/Conversation Pairs", padding="5")
        list_frame.pack(fill='both', expand=True, pady=5)

        columns = ('#', 'Input (context)', 'Output (response)')
        self.pair_tree = ttk.Treeview(list_frame, columns=columns, show='headings', height=12)
        self.pair_tree.heading('#', text='#')
        self.pair_tree.heading('Input (context)', text='Input (context)')
        self.pair_tree.heading('Output (response)', text='Output (response)')
        self.pair_tree.column('#', width=40, anchor='center')
        self.pair_tree.column('Input (context)', width=300)
        self.pair_tree.column('Output (response)', width=300)

        scrollbar = ttk.Scrollbar(list_frame, orient='vertical', command=self.pair_tree.yview)
        self.pair_tree.configure(yscrollcommand=scrollbar.set)
        self.pair_tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        btn_frame = ttk.Frame(self.data_tab)
        btn_frame.pack(fill='x', pady=5)

        ttk.Button(btn_frame, text="Add Pair", command=self.add_pair_dialog).pack(side='left', padx=2)
        ttk.Button(btn_frame, text="Remove Selected", command=self.remove_selected_pair).pack(side='left', padx=2)
        ttk.Button(btn_frame, text="Clear All", command=self.clear_all_pairs).pack(side='left', padx=2)

        ttk.Separator(btn_frame, orient='vertical').pack(side='left', padx=10, fill='y')

        ttk.Button(btn_frame, text="Load Text File", command=self.load_text_file).pack(side='left', padx=2)
        ttk.Button(btn_frame, text="Load CSV/Excel", command=self.load_csv_dialog).pack(side='left', padx=2)
        ttk.Button(btn_frame, text="Load XML", command=self.load_xml_dialog).pack(side='left', padx=2)

        self.pair_count_label = ttk.Label(self.data_tab, text="Total pairs: 0")
        self.pair_count_label.pack(anchor='w', pady=5)

    # ------------------------------------------------------------------ TRAIN TAB
    def setup_train_tab(self):
        # ============ Model settings ============
        settings = ttk.LabelFrame(self.train_tab, text="Model Settings", padding="10")
        settings.pack(fill='x', pady=5)
        for c in (1, 3, 5):
            settings.columnconfigure(c, weight=1)

        # Row 0: architecture + preset
        ttk.Label(settings, text="Architecture:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.arch = tk.StringVar(value="Transformer")
        self.arch_combo = ttk.Combobox(settings, textvariable=self.arch,
                                       values=self.ARCHS, width=20, state='readonly')
        self.arch_combo.grid(row=0, column=1, padx=5, pady=2, sticky='w')
        self.arch_combo.bind('<<ComboboxSelected>>', self.on_arch_changed)

        ttk.Label(settings, text="Preset:").grid(row=0, column=2, sticky='w', padx=(20, 5), pady=2)
        self.preset_size = tk.StringVar(value="Medium")
        ttk.Combobox(settings, textvariable=self.preset_size,
                     values=["Small", "Medium", "Large"], width=10,
                     state='readonly').grid(row=0, column=3, padx=5, pady=2, sticky='w')
        ttk.Button(settings, text="Apply Preset", command=self.apply_preset).grid(row=0, column=4, padx=5)

        # Row 1: common settings
        ttk.Label(settings, text="Embed size:").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        self.embed_size_var = tk.StringVar(value="128")
        ttk.Entry(settings, textvariable=self.embed_size_var, width=8).grid(
            row=1, column=1, padx=5, pady=2, sticky='w')

        ttk.Label(settings, text="Dropout:").grid(row=1, column=2, sticky='w', padx=(20, 5), pady=2)
        self.dropout_var = tk.StringVar(value="0.1")
        ttk.Entry(settings, textvariable=self.dropout_var, width=8).grid(
            row=1, column=3, padx=5, pady=2, sticky='w')

        ttk.Label(settings, text="Max len:").grid(row=1, column=4, sticky='w', padx=(20, 5), pady=2)
        self.max_len_var = tk.StringVar(value="256")
        ttk.Entry(settings, textvariable=self.max_len_var, width=8).grid(
            row=1, column=5, padx=5, pady=2, sticky='w')

        # Row 2: RNN settings
        self.rnn_frame = ttk.LabelFrame(settings, text="RNN settings (RNN / LSTM / GRU)", padding="5")
        self.rnn_frame.grid(row=2, column=0, columnspan=6, sticky='ew', padx=5, pady=5)

        ttk.Label(self.rnn_frame, text="Hidden size:").grid(row=0, column=0, sticky='w', padx=5)
        self.rnn_hidden_var = tk.StringVar(value="128")
        ttk.Entry(self.rnn_frame, textvariable=self.rnn_hidden_var, width=8).grid(row=0, column=1, padx=5)

        ttk.Label(self.rnn_frame, text="Encoder layers:").grid(row=0, column=2, sticky='w', padx=(20, 5))
        self.rnn_enc_layers_var = tk.StringVar(value="2")
        ttk.Entry(self.rnn_frame, textvariable=self.rnn_enc_layers_var, width=8).grid(row=0, column=3, padx=5)

        ttk.Label(self.rnn_frame, text="Decoder layers:").grid(row=0, column=4, sticky='w', padx=(20, 5))
        self.rnn_dec_layers_var = tk.StringVar(value="2")
        ttk.Entry(self.rnn_frame, textvariable=self.rnn_dec_layers_var, width=8).grid(row=0, column=5, padx=5)

        # Row 3: FFN settings
        self.ffn_frame = ttk.LabelFrame(settings, text="FFN settings", padding="5")
        self.ffn_frame.grid(row=3, column=0, columnspan=6, sticky='ew', padx=5, pady=5)

        ttk.Label(self.ffn_frame,
                  text="Hidden layer sizes (comma-separated, e.g. 512,512,256):").grid(
            row=0, column=0, sticky='w', padx=5)
        self.ffn_hidden_var = tk.StringVar(value="512,512")
        ttk.Entry(self.ffn_frame, textvariable=self.ffn_hidden_var, width=40).grid(
            row=0, column=1, padx=5, sticky='w')

        # Row 4: Transformer settings
        self.tf_frame = ttk.LabelFrame(settings, text="Transformer settings", padding="5")
        self.tf_frame.grid(row=4, column=0, columnspan=6, sticky='ew', padx=5, pady=5)

        ttk.Label(self.tf_frame, text="Heads:").grid(row=0, column=0, sticky='w', padx=5)
        self.tf_nhead_var = tk.StringVar(value="8")
        ttk.Entry(self.tf_frame, textvariable=self.tf_nhead_var, width=8).grid(row=0, column=1, padx=5)

        ttk.Label(self.tf_frame, text="Dim feedforward:").grid(row=0, column=2, sticky='w', padx=(20, 5))
        self.tf_dim_ff_var = tk.StringVar(value="512")
        ttk.Entry(self.tf_frame, textvariable=self.tf_dim_ff_var, width=8).grid(row=0, column=3, padx=5)

        ttk.Label(self.tf_frame, text="Encoder layers:").grid(row=0, column=4, sticky='w', padx=(20, 5))
        self.tf_enc_layers_var = tk.StringVar(value="4")
        ttk.Entry(self.tf_frame, textvariable=self.tf_enc_layers_var, width=8).grid(row=0, column=5, padx=5)

        ttk.Label(self.tf_frame, text="Decoder layers:").grid(row=0, column=6, sticky='w', padx=(20, 5))
        self.tf_dec_layers_var = tk.StringVar(value="4")
        ttk.Entry(self.tf_frame, textvariable=self.tf_dec_layers_var, width=8).grid(row=0, column=7, padx=5)

        # ============ Training settings ============
        train_settings = ttk.LabelFrame(self.train_tab, text="Training Settings", padding="10")
        train_settings.pack(fill='x', pady=5)

        ttk.Label(train_settings, text="Epochs:").grid(row=0, column=0, sticky='w', padx=5)
        self.epochs_var = tk.StringVar(value="100")
        ttk.Entry(train_settings, textvariable=self.epochs_var, width=8).grid(row=0, column=1, padx=5)

        ttk.Label(train_settings, text="Batch size:").grid(row=0, column=2, sticky='w', padx=(20, 5))
        self.batch_var = tk.StringVar(value="16")
        ttk.Entry(train_settings, textvariable=self.batch_var, width=8).grid(row=0, column=3, padx=5)

        ttk.Label(train_settings, text="Learning rate:").grid(row=0, column=4, sticky='w', padx=(20, 5))
        self.lr_var = tk.StringVar(value="0.0003")
        ttk.Entry(train_settings, textvariable=self.lr_var, width=8).grid(row=0, column=5, padx=5)

        # ============ Buttons ============
        btn_frame1 = ttk.Frame(self.train_tab)
        btn_frame1.pack(pady=5)
        self.init_btn = ttk.Button(btn_frame1, text="Init Model", command=self.init_model)
        self.init_btn.pack(side='left', padx=5)
        self.save_btn = ttk.Button(btn_frame1, text="Save Model",
                                   command=self.save_model, state='disabled')
        self.save_btn.pack(side='left', padx=5)
        self.load_btn = ttk.Button(btn_frame1, text="Load Model", command=self.load_model)
        self.load_btn.pack(side='left', padx=5)

        btn_frame2 = ttk.Frame(self.train_tab)
        btn_frame2.pack(pady=5)
        self.train_btn = ttk.Button(btn_frame2, text="Start Training",
                                    command=self.start_training, state='disabled')
        self.train_btn.pack(side='left', padx=5)
        self.stop_btn = ttk.Button(btn_frame2, text="Stop",
                                   command=self.stop_training_cmd, state='disabled')
        self.stop_btn.pack(side='left', padx=5)

        self.progress = ttk.Progressbar(self.train_tab, mode='indeterminate', length=400)
        self.progress.pack(pady=5)

        loss_frame = ttk.Frame(self.train_tab)
        loss_frame.pack(pady=5)
        ttk.Label(loss_frame, text="Current Loss:").pack(side='left')
        self.loss_label = ttk.Label(loss_frame, text="0.0000",
                                    font='Arial 10 bold', foreground='blue')
        self.loss_label.pack(side='left', padx=5)

        ttk.Label(self.train_tab, text="Training Log:").pack(anchor='w', pady=(10, 5))
        self.log = scrolledtext.ScrolledText(self.train_tab, height=10, state='disabled')
        self.log.pack(fill='both', expand=True)

        # init visibility
        self.on_arch_changed()

    def _set_widget_state(self, widget, state):
        try:
            widget.configure(state=state)
        except Exception:
            pass
        for child in widget.winfo_children():
            self._set_widget_state(child, state)

    def on_arch_changed(self, event=None):
        arch = self.arch.get()
        rnn_state = 'normal' if arch in ('RNN', 'LSTM', 'GRU') else 'disabled'
        ffn_state = 'normal' if arch == 'FFN' else 'disabled'
        tf_state = 'normal' if arch == 'Transformer' else 'disabled'

        self._set_widget_state(self.rnn_frame, rnn_state)
        self._set_widget_state(self.ffn_frame, ffn_state)
        self._set_widget_state(self.tf_frame, tf_state)

        try:
            self.rnn_frame.configure(state='normal')
            self.ffn_frame.configure(state='normal')
            self.tf_frame.configure(state='normal')
        except Exception:
            pass

    def apply_preset(self):
        arch = self.arch.get()
        size = self.preset_size.get()
        p = get_preset(size, arch)

        if 'embed_size' in p:
            self.embed_size_var.set(str(p['embed_size']))
        if 'hidden' in p:
            self.rnn_hidden_var.set(str(p['hidden']))
        if 'enc' in p:
            if arch in ('RNN', 'LSTM', 'GRU'):
                self.rnn_enc_layers_var.set(str(p['enc']))
            else:
                self.tf_enc_layers_var.set(str(p['enc']))
        if 'dec' in p:
            if arch in ('RNN', 'LSTM', 'GRU'):
                self.rnn_dec_layers_var.set(str(p['dec']))
            else:
                self.tf_dec_layers_var.set(str(p['dec']))
        if 'nhead' in p:
            self.tf_nhead_var.set(str(p['nhead']))
        if 'dim_ff' in p:
            self.tf_dim_ff_var.set(str(p['dim_ff']))
        if 'hidden_layers' in p:
            self.ffn_hidden_var.set(p['hidden_layers'])

        self.status_var.set(f"Applied preset: {size} / {arch}")

    # ------------------------------------------------------------------ TEST TAB
    def setup_test_tab(self):
        single_frame = ttk.LabelFrame(self.test_tab, text="Single Word", padding="10")
        single_frame.pack(fill='x', pady=5)

        ttk.Label(single_frame, text="Input:").grid(row=0, column=0, sticky='w')
        self.input_var = tk.StringVar()
        ttk.Entry(single_frame, textvariable=self.input_var, width=40).grid(
            row=0, column=1, padx=5, sticky='w')
        ttk.Button(single_frame, text="Transform",
                   command=self.process_input).grid(row=0, column=2, padx=5)

        ttk.Label(single_frame, text="Output:").grid(row=1, column=0, sticky='nw', pady=(10, 0))
        self.output_text = scrolledtext.ScrolledText(single_frame, height=5, wrap='word', width=60)
        self.output_text.grid(row=2, column=0, columnspan=3, pady=5, sticky='ew')

        batch_frame = ttk.LabelFrame(self.test_tab, text="Batch Test", padding="10")
        batch_frame.pack(fill='both', expand=True, pady=5)
        ttk.Button(batch_frame, text="Test All Pairs",
                   command=self.test_all_pairs).pack(anchor='w', pady=5)
        self.test_output = scrolledtext.ScrolledText(batch_frame, height=15, font=('Courier', 10))
        self.test_output.pack(fill='both', expand=True)

    # ------------------------------------------------------------------ CHAT TAB
    def setup_chat_tab(self):
        chat_frame = ttk.LabelFrame(self.chat_tab, text="Conversation", padding="5")
        chat_frame.pack(fill='both', expand=True, pady=5)

        self.chat_display = scrolledtext.ScrolledText(chat_frame, height=20, wrap='word',
                                                      state='disabled', font=('Arial', 11))
        self.chat_display.pack(fill='both', expand=True)

        input_frame = ttk.Frame(self.chat_tab)
        input_frame.pack(fill='x', pady=5)

        ttk.Label(input_frame, text="You:").pack(side='left', padx=5)
        self.chat_entry = ttk.Entry(input_frame, width=70)
        self.chat_entry.pack(side='left', padx=5, fill='x', expand=True)
        self.chat_entry.bind('<Return>', lambda e: self.send_message())

        self.send_btn = ttk.Button(input_frame, text="Send", command=self.send_message)
        self.send_btn.pack(side='left', padx=5)

        ctrl_frame = ttk.Frame(self.chat_tab)
        ctrl_frame.pack(fill='x', pady=5)
        ttk.Button(ctrl_frame, text="Clear Chat", command=self.clear_chat).pack(side='left', padx=5)
        ttk.Button(ctrl_frame, text="Save Conversation", command=self.save_chat).pack(side='left', padx=5)
        ttk.Button(ctrl_frame, text="Load Conversation", command=self.load_chat).pack(side='left', padx=5)

        temp_frame = ttk.Frame(self.chat_tab)
        temp_frame.pack(fill='x', pady=5)
        ttk.Label(temp_frame, text="Temperature:").pack(side='left', padx=5)
        self.chat_temp = tk.DoubleVar(value=0.8)
        ttk.Scale(temp_frame, from_=0.1, to=2.0, variable=self.chat_temp,
                  orient='horizontal', length=200).pack(side='left', padx=5)
        ttk.Label(temp_frame, textvariable=self.chat_temp).pack(side='left', padx=5)

        self.chat_queue = queue.Queue()

    # ------------------------------------------------------------------ CHAT logic
    def send_message(self):
        if self.model is None:
            messagebox.showwarning("Warning", "Please load or train a model first.")
            return

        user_msg = self.chat_entry.get().strip()
        if not user_msg:
            return

        self.chat_history.append(('user', user_msg))
        self.update_chat_display()

        source_parts = []
        for speaker, text in self.chat_history:
            source_parts.append(f"<USER> {text}" if speaker == 'user' else f"<BOT> {text}")
        source_str = " ".join(source_parts)

        max_len = self.model.max_len
        src_ids = self.vocab.encode(source_str)
        if len(src_ids) > max_len:
            src_ids = src_ids[-max_len:]
        src_tensor = torch.tensor(
            [src_ids + [self.vocab.pad_idx] * (max_len - len(src_ids))],
            dtype=torch.long, device=self.device)

        self.chat_history.append(('bot', ''))
        self.update_chat_display()

        self.send_btn.config(state='disabled')
        self.chat_entry.config(state='disabled')

        threading.Thread(target=self.generate_response_thread,
                         args=(src_tensor, self.chat_temp.get()),
                         daemon=True).start()

        self.chat_entry.delete(0, 'end')

    def generate_response_thread(self, src_tensor, temperature):
        try:
            def token_callback(token_idx):
                self.chat_queue.put(('token', token_idx))

            self.model.generate(src_tensor, self.vocab.sos_idx, self.vocab.eos_idx,
                                max_len=None, temperature=temperature, callback=token_callback)
            self.chat_queue.put(('done', None))
        except Exception as e:
            self.chat_queue.put(('error', str(e)))

    def process_chat_queue(self):
        try:
            while True:
                msg_type, data = self.chat_queue.get_nowait()
                if msg_type == 'token':
                    token = data
                    if self.chat_history and self.chat_history[-1][0] == 'bot':
                        current_text = self.chat_history[-1][1]
                        ch = self.vocab.idx_to_char.get(token, '')
                        if ch and ch not in self.vocab.special_tokens:
                            current_text += ch
                            self.chat_history[-1] = ('bot', current_text)
                            self.update_chat_display()
                elif msg_type == 'done':
                    self.send_btn.config(state='normal')
                    self.chat_entry.config(state='normal')
                    self.status_var.set("Bot replied")
                elif msg_type == 'error':
                    messagebox.showerror("Generation Error", data)
                    self.send_btn.config(state='normal')
                    self.chat_entry.config(state='normal')
                    self.status_var.set("Error during generation")
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.process_chat_queue)

    def update_chat_display(self):
        self.chat_display.config(state='normal')
        self.chat_display.delete('1.0', 'end')
        for speaker, text in self.chat_history:
            if speaker == 'user':
                self.chat_display.insert('end', f"You: {text}\n\n")
            else:
                self.chat_display.insert('end', f"Bot: {text}\n\n")
        self.chat_display.see('end')
        self.chat_display.config(state='disabled')

    def clear_chat(self):
        self.chat_history.clear()
        self.update_chat_display()

    def save_chat(self):
        path = filedialog.asksaveasfilename(defaultextension=".txt",
                                            filetypes=[("Text files", "*.txt")])
        if path:
            with open(path, 'w', encoding='utf-8') as f:
                for speaker, text in self.chat_history:
                    f.write(f"{speaker.upper()}: {text}\n")
            self.status_var.set(f"Chat saved to {os.path.basename(path)}")

    def load_chat(self):
        path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if path:
            self.chat_history.clear()
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("USER:"):
                        self.chat_history.append(('user', line[5:].strip()))
                    elif line.startswith("BOT:"):
                        self.chat_history.append(('bot', line[4:].strip()))
            self.update_chat_display()
            self.status_var.set(f"Chat loaded from {os.path.basename(path)}")

    # ------------------------------------------------------------------ DATA methods
    def refresh_pair_list(self):
        for item in self.pair_tree.get_children():
            self.pair_tree.delete(item)
        for i, (inp, out) in enumerate(self.pairs):
            self.pair_tree.insert('', 'end', values=(i + 1, inp, out))
        self.pair_count_label.config(text=f"Total pairs: {len(self.pairs)}")

    def add_pair_dialog(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Add Pair")
        dialog.geometry("400x150")
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(dialog, text="Input (context):").grid(row=0, column=0, padx=10, pady=10, sticky='w')
        input_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=input_var, width=30).grid(row=0, column=1, padx=10, pady=10)

        ttk.Label(dialog, text="Output (response):").grid(row=1, column=0, padx=10, pady=10, sticky='w')
        output_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=output_var, width=30).grid(row=1, column=1, padx=10, pady=10)

        def add():
            inp = input_var.get().strip()
            out = output_var.get().strip()
            if inp and out:
                self.pairs.append((inp, out))
                self.refresh_pair_list()
                dialog.destroy()

        ttk.Button(dialog, text="Add", command=add).grid(row=2, column=0, columnspan=2, pady=10)

    def remove_selected_pair(self):
        selected = self.pair_tree.selection()
        if not selected:
            return
        idxs = sorted([int(self.pair_tree.item(item, 'values')[0]) - 1 for item in selected],
                      reverse=True)
        for idx in idxs:
            if 0 <= idx < len(self.pairs):
                self.pairs.pop(idx)
        self.refresh_pair_list()

    def clear_all_pairs(self):
        if messagebox.askyesno("Confirm", "Clear all pairs?"):
            self.pairs.clear()
            self.refresh_pair_list()

    def load_text_file(self):
        path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if path:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    text = f.read()
                new_pairs = parse_text_pairs(text)
                self.pairs.extend(new_pairs)
                self.refresh_pair_list()
                self.status_var.set(f"Loaded {len(new_pairs)} pairs from {os.path.basename(path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {e}")

    def load_csv_dialog(self):
        dlg = CSVImportDialog(self.root)
        self.root.wait_window(dlg)
        if dlg.result:
            self.pairs.extend(dlg.result)
            self.refresh_pair_list()
            self.status_var.set(f"Imported {len(dlg.result)} pairs from CSV")

    def load_xml_dialog(self):
        dlg = XMLImportDialog(self.root)
        self.root.wait_window(dlg)
        if dlg.result:
            self.pairs.extend(dlg.result)
            self.refresh_pair_list()
            self.status_var.set(f"Imported {len(dlg.result)} pairs from XML")

    # ------------------------------------------------------------------ TRAIN methods
    def init_model(self):
        arch = self.arch.get()
        try:
            embed_size = int(self.embed_size_var.get())
            dropout = float(self.dropout_var.get())
            max_len = int(self.max_len_var.get())
        except Exception as e:
            messagebox.showerror("Error", f"Invalid common setting: {e}")
            return

        common = dict(vocab_size=len(self.vocab.char_to_idx),
                      max_len=max_len,
                      embed_size=embed_size,
                      dropout=dropout)

        try:
            if arch in ('RNN', 'LSTM', 'GRU'):
                self.model = RNNSeq2Seq(
                    rnn_type=arch,
                    hidden_size=int(self.rnn_hidden_var.get()),
                    num_encoder_layers=int(self.rnn_enc_layers_var.get()),
                    num_decoder_layers=int(self.rnn_dec_layers_var.get()),
                    **common,
                )
            elif arch == 'FFN':
                raw = self.ffn_hidden_var.get().strip()
                hidden_layers = [int(x.strip()) for x in raw.split(',') if x.strip()] if raw else []
                self.model = FNNSeq2Seq(hidden_layers=hidden_layers, **common)
            else:  # Transformer
                self.model = TransformerSeq2Seq(
                    nhead=int(self.tf_nhead_var.get()),
                    dim_feedforward=int(self.tf_dim_ff_var.get()),
                    num_encoder_layers=int(self.tf_enc_layers_var.get()),
                    num_decoder_layers=int(self.tf_dec_layers_var.get()),
                    activation='gelu',
                    **common,
                )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize model: {e}")
            traceback.print_exc()
            return

        self.model.to(self.device)
        self.log_message(f"Model initialized: {arch}")
        self.save_btn.config(state='normal')
        self.train_btn.config(state='normal')
        self.status_var.set("Model initialized")

    def save_model(self):
        if self.model is None:
            messagebox.showerror("Error", "No model to save.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".pth",
                                            filetypes=[("PyTorch model", "*.pth")])
        if path:
            try:
                self.model.save(path)
                self.log_message(f"Model saved to {os.path.basename(path)}")
                self.status_var.set("Model saved")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save model: {e}")

    def load_model(self):
        path = filedialog.askopenfilename(filetypes=[("PyTorch model", "*.pth")])
        if not path:
            return
        try:
            checkpoint = torch.load(path, map_location='cpu')
            config = checkpoint['config']
            arch = config.get('arch', 'Transformer')

            if arch in ('RNN', 'LSTM', 'GRU'):
                self.model = RNNSeq2Seq.load(path)
                self.arch.set(config.get('rnn_type', arch))
                self.rnn_hidden_var.set(str(config.get('hidden_size', 128)))
                self.rnn_enc_layers_var.set(str(config.get('num_encoder_layers', 2)))
                self.rnn_dec_layers_var.set(str(config.get('num_decoder_layers', 2)))
            elif arch == 'FFN':
                self.model = FNNSeq2Seq.load(path)
                self.arch.set('FFN')
                hl = config.get('hidden_layers', [])
                self.ffn_hidden_var.set(','.join(str(x) for x in hl))
            else:
                self.model = TransformerSeq2Seq.load(path)
                self.arch.set('Transformer')
                self.tf_nhead_var.set(str(config.get('nhead', 8)))
                self.tf_dim_ff_var.set(str(config.get('dim_feedforward', 512)))
                self.tf_enc_layers_var.set(str(config.get('num_encoder_layers', 4)))
                self.tf_dec_layers_var.set(str(config.get('num_decoder_layers', 4)))

            self.embed_size_var.set(str(config.get('embed_size', 128)))
            self.dropout_var.set(str(config.get('dropout', 0.1)))
            self.max_len_var.set(str(config.get('max_len', 256)))

            self.model.to(self.device)
            self.on_arch_changed()

            self.log_message(f"Model loaded from {os.path.basename(path)}")
            self.log_message(f"  Architecture: {arch}, embed_size={config.get('embed_size')}")
            self.save_btn.config(state='normal')
            self.train_btn.config(state='normal')
            self.status_var.set("Model loaded")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")
            traceback.print_exc()

    def start_training(self):
        if self.model is None:
            messagebox.showwarning("Warning", "Please initialize or load a model first.")
            return
        if self.training_thread and self.training_thread.is_alive():
            return
        if len(self.pairs) < 2:
            messagebox.showwarning("Warning", "Need at least 2 pairs to train.")
            return

        try:
            epochs = int(self.epochs_var.get())
            batch_size = int(self.batch_var.get())
            lr = float(self.lr_var.get())
        except Exception as e:
            messagebox.showerror("Error", f"Invalid training setting: {e}")
            return

        self.log_message("Training started...")
        self.progress.start()
        self.train_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.init_btn.config(state='disabled')
        self.save_btn.config(state='disabled')
        self.load_btn.config(state='disabled')
        self.status_var.set("Training...")

        self.stop_training = False
        self.training_thread = threading.Thread(
            target=self.training_worker,
            args=(self.pairs.copy(), epochs, batch_size, lr),
            daemon=True)
        self.training_thread.start()

    def training_worker(self, pairs, epochs, batch_size, lr):
        try:
            for epoch, loss, info in train_model(pairs, self.vocab, self.model,
                                                  epochs, batch_size, lr, self.device):
                if self.stop_training:
                    self.queue.put(("status", "Training stopped"))
                    break
                if epoch is not None:
                    self.queue.put(("progress", f"Epoch {epoch}, Loss: {loss:.4f}{info}"))
                else:
                    self.queue.put(("progress", "Training complete"))
                    self.queue.put(("status", "Training complete"))
            self.queue.put(("training_done", None))
        except Exception as e:
            self.queue.put(("error", str(e)))
            traceback.print_exc()

    def stop_training_cmd(self):
        self.stop_training = True
        self.stop_btn.config(state='disabled')
        self.status_var.set("Stopping...")

    def log_message(self, msg):
        self.log.config(state='normal')
        self.log.insert('end', msg + '\n')
        self.log.see('end')
        self.log.config(state='disabled')

    # ------------------------------------------------------------------ TEST methods
    def process_input(self):
        if self.model is None:
            messagebox.showwarning("Warning", "Please train or load a model first.")
            return
        word = self.input_var.get().strip()
        if not word:
            return
        result = self.transform_word(word)
        self.output_text.delete('1.0', 'end')
        self.output_text.insert('1.0', result)
        self.status_var.set(f"{word} → {result}")

    def transform_word(self, word):
        self.model.eval()
        max_len = self.model.max_len
        ids = self.vocab.encode(word)
        if len(ids) > max_len:
            ids = ids[:max_len]
        src = torch.tensor([ids + [self.vocab.pad_idx] * (max_len - len(ids))],
                           dtype=torch.long, device=self.device)
        gen = self.model.generate(src, self.vocab.sos_idx, self.vocab.eos_idx, max_len)
        return self.vocab.decode(gen[1:])

    def test_all_pairs(self):
        if self.model is None:
            messagebox.showwarning("Warning", "Please train or load a model first.")
            return
        self.test_output.delete('1.0', 'end')
        correct = 0
        total = len(self.pairs)
        for i, (inp, exp) in enumerate(self.pairs):
            res = self.transform_word(inp)
            ok = (res == exp)
            if ok:
                correct += 1
            mark = "✓" if ok else "✗"
            self.test_output.insert('end', f"{inp:20} → {res:20}  ({exp}) {mark}\n")
            if i % 10 == 0:
                self.test_output.see('end')
                self.root.update()
        acc = correct / max(total, 1) * 100
        self.test_output.insert('end', f"\nAccuracy: {correct}/{total} ({acc:.1f}%)")
        self.status_var.set(f"Tested {total} pairs, accuracy {acc:.1f}%")

    # ------------------------------------------------------------------ QUEUE processing
    def process_queue(self):
        try:
            while True:
                msg, data = self.queue.get_nowait()
                if msg == "progress":
                    self.log_message(data)
                    if "Loss:" in data:
                        try:
                            loss_part = data.split("Loss:")[1].split()[0]
                            self.loss_label.config(text=loss_part)
                        except Exception:
                            pass
                elif msg == "status":
                    self.status_var.set(data)
                elif msg == "error":
                    messagebox.showerror("Error", data)
                    self.train_btn.config(state='normal')
                    self.stop_btn.config(state='disabled')
                    self.init_btn.config(state='normal')
                    self.save_btn.config(state='normal')
                    self.load_btn.config(state='normal')
                    self.progress.stop()
                elif msg == "training_done":
                    self.train_btn.config(state='normal')
                    self.stop_btn.config(state='disabled')
                    self.init_btn.config(state='normal')
                    self.save_btn.config(state='normal')
                    self.load_btn.config(state='normal')
                    self.progress.stop()
        except queue.Empty:
            pass
        self.root.after(100, self.process_queue)

    def run(self):
        self.root.after(100, self.process_chat_queue)
        self.root.mainloop()


# ==================== MAIN ====================
if __name__ == "__main__":
    app = TransformerPredictorUI()
    app.run()