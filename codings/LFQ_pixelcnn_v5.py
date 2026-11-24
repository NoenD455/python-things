# -*- coding: utf-8 -*-
"""
LFQ-VAE + autoregressive prior (LSTM / PixelCNN / Transformer), Tkinter GUI.

v5 changes (vs v4)
  * New causal Transformer prior (pre-norm, sin-cos/learned/none positions).
  * Every prior has its own settings page (LSTM / PixelCNN / Transformer), including
    its own positional-embedding choice. Settings are split into sub-pages.
  * Preview rate: for both the VQ-VAE and the prior, preview every N steps or N epochs
    (0 = off), plus one final preview. Number of preview images is configurable.
  * GUI: status header, resizable log, per-tab training/arch frames, ETA,
    save/load buttons next to the model they belong to.
  * PixelCNN input-mask kernel size is configurable.
"""

import os, sys, time, random, threading, queue, traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import tkinter.font as tkfont

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset

import torchvision.transforms as T
import torchvision.utils as vutils
from PIL import Image, ImageTk


IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.ppm', '.tif', '.tiff')
MAX_D = 12          # K = 2**D codes; the prior uses a K-way softmax, so keep it sane


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
class ImageListDataset(Dataset):
    """Resize+center-crop once (cached in RAM), optional PIL-level augmentation."""

    def __init__(self, paths, size=64, aug=None, store=None):
        self.paths = list(paths)
        self.size = int(size)
        self.aug = aug
        self.store = store if store is not None else [None] * len(self.paths)
        self.base = T.Compose([T.Resize(self.size), T.CenterCrop(self.size)])
        self.to_tensor = T.ToTensor()

    def with_aug(self, aug_tf):
        """Same images/cache, different augmentation pipeline."""
        return ImageListDataset(self.paths, self.size, aug_tf, self.store)

    def __len__(self):
        return len(self.paths)

    def _get_pil(self, i):
        im = self.store[i]
        if im is None:
            try:
                im = self.base(Image.open(self.paths[i]).convert('RGB'))
            except Exception:
                im = False                      # remembered as unreadable
            self.store[i] = im
        return im

    def __getitem__(self, i):
        n = len(self.paths)
        for k in range(n):                      # skip unreadable files, never fake data
            im = self._get_pil((i + k) % n)
            if im is not False:
                break
        else:
            raise RuntimeError('No readable images in dataset.')
        if self.aug is not None:
            im = self.aug(im)
        return self.to_tensor(im)


def build_aug(a, size):
    """a: dict(hflip, vflip, rot, zoom, jitter). Returns a PIL->PIL transform or None."""
    ops = []
    if a.get('zoom', 1.0) < 0.999:
        ops.append(T.RandomResizedCrop(size, scale=(a['zoom'], 1.0), ratio=(1.0, 1.0)))
    if a.get('rot', 0.0) > 0:
        ops.append(T.RandomRotation(a['rot'], interpolation=T.InterpolationMode.BILINEAR))
    if a.get('hflip'):
        ops.append(T.RandomHorizontalFlip(0.5))
    if a.get('vflip'):
        ops.append(T.RandomVerticalFlip(0.5))
    if a.get('jitter', 0.0) > 0:
        j = a['jitter']
        ops.append(T.ColorJitter(brightness=j, contrast=j, saturation=j))
    return T.Compose(ops) if ops else None


# ----------------------------------------------------------------------------
# LFQ quantizer + tokenizer
# ----------------------------------------------------------------------------
class LookupFreeQuantizer(nn.Module):
    def __init__(self, D, beta=0.1, entropy_weight=0.1, tau=2.0):
        super().__init__()
        self.D = D
        self.K = 2 ** D
        self.beta = beta
        self.entropy_weight = entropy_weight
        self.tau = tau
        self.register_buffer('powers', (2 ** torch.arange(D)).long())         # [1,2,4,...]
        self.register_buffer('bit_positions', torch.arange(D).long())         # [0,1,2,...]
        self._self_test()

    def _self_test(self):
        """Verify codes_to_latent is the exact inverse of forward()."""
        z = torch.randn(2, self.D, 4, 4)
        z_q, _, codes = self.forward(z)
        z_rec = self.codes_to_latent(codes)
        err = (z_q - z_rec).abs().max().item()
        if err > 1e-4:
            raise RuntimeError('codes_to_latent is not the inverse of forward(): '
                               'max err %.5f' % err)

    def entropy_loss(self, z):
        """Differentiable LFQ entropy regulariser (MAGVIT-v2 style, bit-factorised).
        minimise  E[H(bits | z)]  -  H(mean bits)
        = confident per-sample bits, balanced bit usage across the batch."""
        p = torch.sigmoid(z * (self.tau * self.D ** 0.5))             # (B,D,H,W)
        Hb = lambda q: -(q * torch.log(q + 1e-6) + (1 - q) * torch.log(1 - q + 1e-6))
        per_sample = Hb(p).sum(1).mean()
        pm = p.mean(dim=(0, 2, 3))
        return per_sample - Hb(pm).sum()

    def forward(self, z):
        B, D, H, W = z.shape
        bits = (z >= 0).long()
        z_q = (2.0 * bits.float() - 1.0) * (D ** -0.5)
        z_q_st = z + (z_q - z).detach()
        commit = F.mse_loss(z, z_q.detach())
        codes = (bits * self.powers.view(1, D, 1, 1)).sum(1)                    # (B,H,W)
        loss = self.beta * commit + self.entropy_weight * self.entropy_loss(z)
        return z_q_st, loss, codes

    def codes_to_latent(self, idx):
        bits = ((idx.unsqueeze(1) >> self.bit_positions.view(1, -1, 1, 1)) & 1).long()
        return (2.0 * bits.float() - 1.0) * (self.D ** -0.5)


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.f = nn.Sequential(nn.ReLU(), nn.Conv2d(ch, ch, 3, 1, 1),
                               nn.ReLU(), nn.Conv2d(ch, ch, 1))

    def forward(self, x):
        return x + self.f(x)


def n_downs(img_size, latent_size):
    if latent_size < 1 or img_size < 1:
        raise ValueError('Sizes must be positive.')
    if latent_size > img_size:
        raise ValueError('Latent size (%d) cannot exceed image size (%d).' % (latent_size, img_size))
    if img_size % latent_size:
        raise ValueError('Image size %d must be divisible by latent size %d.' % (img_size, latent_size))
    r = img_size // latent_size
    if r & (r - 1):
        raise ValueError('Image/latent ratio must be a power of 2 (got %d).' % r)
    return r.bit_length() - 1


class VQVAE(nn.Module):
    def __init__(self, D=8, hidden=128, in_ch=3, beta=0.1, entropy_weight=0.1,
                 img_size=64, latent_size=8):
        super().__init__()
        self.D = D
        self.K = 2 ** D
        self.img_size, self.latent_size = img_size, latent_size
        n = n_downs(img_size, latent_size)

        enc, c = [], in_ch
        if n == 0:
            enc.append(nn.Conv2d(in_ch, hidden, 3, 1, 1))
        for _ in range(n):
            enc += [nn.Conv2d(c, hidden, 4, 2, 1), nn.ReLU()]
            c = hidden
        enc += [ResBlock(hidden), ResBlock(hidden), nn.ReLU(), nn.Conv2d(hidden, D, 3, 1, 1)]
        self.encoder = nn.Sequential(*enc)

        self.vq = LookupFreeQuantizer(D, beta, entropy_weight)

        dec = [nn.Conv2d(D, hidden, 3, 1, 1), ResBlock(hidden), ResBlock(hidden)]
        for k in range(n):
            dec += [nn.ReLU(), nn.ConvTranspose2d(hidden, hidden, 4, 2, 1)]
            if k < n - 1:                       # no res-block at full resolution (CPU cost)
                dec.append(ResBlock(hidden))
        dec += [nn.ReLU(), nn.Conv2d(hidden, in_ch, 3, 1, 1)]
        self.decoder = nn.Sequential(*dec)

    def encode(self, x):
        return self.vq(self.encoder(x))

    def decode(self, z_q):
        return torch.sigmoid(self.decoder(z_q))

    def decode_indices(self, idx):
        return self.decode(self.vq.codes_to_latent(idx))

    def forward(self, x):
        z = self.encoder(x)
        z_q_st, vq_loss, codes = self.vq(z)
        x_rec = self.decode(z_q_st)
        rec = F.l1_loss(x_rec, x) + F.mse_loss(x_rec, x)
        return x_rec, rec + vq_loss, rec, vq_loss, codes


# ----------------------------------------------------------------------------
# Positional embeddings
# ----------------------------------------------------------------------------
def sincos_2d(H, W, dim, base=1000.0):
    """Fixed 2D sin-cos embedding, returns (H*W, dim) in raster order.
    First half of the channels encodes the row, second half the column."""
    q = max(dim // 4, 1)
    omega = 1.0 / (base ** (torch.arange(q, dtype=torch.float32) / q))
    ys = torch.arange(H, dtype=torch.float32)[:, None] * omega[None]
    xs = torch.arange(W, dtype=torch.float32)[:, None] * omega[None]
    ey = torch.cat([ys.sin(), ys.cos()], 1)                     # (H,2q)
    ex = torch.cat([xs.sin(), xs.cos()], 1)                     # (W,2q)
    grid = torch.cat([ey[:, None, :].expand(H, W, 2 * q),
                      ex[None, :, :].expand(H, W, 2 * q)], 2)   # (H,W,4q)
    if grid.shape[2] < dim:
        grid = F.pad(grid, (0, dim - grid.shape[2]))
    return grid[:, :, :dim].reshape(H * W, dim).contiguous()


# ----------------------------------------------------------------------------
# Priors
# ----------------------------------------------------------------------------
class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _, _, kH, kW = self.weight.shape
        mask = torch.ones_like(self.weight)
        off = 1 if mask_type == 'B' else 0
        mask[:, :, kH // 2, kW // 2 + off:] = 0
        mask[:, :, kH // 2 + 1:, :] = 0
        self.register_buffer('mask', mask)

    def forward(self, x):
        return F.conv2d(x, self.weight * self.mask, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)


class GatedBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = MaskedConv2d('B', ch, 2 * ch, 3, padding=1)

    def forward(self, x):
        a, b = self.conv(x).chunk(2, dim=1)
        return x + torch.tanh(a) * torch.sigmoid(b)


class PixelCNN(nn.Module):
    def __init__(self, K=256, ch=64, n_layers=6, H=8, W=8, pos='sincos', in_kernel=7):
        super().__init__()
        if in_kernel < 3 or in_kernel % 2 == 0:
            raise ValueError('Input kernel must be odd and >= 3.')
        self.K, self.H, self.W, self.pos_mode = K, H, W, pos
        self.emb = nn.Embedding(K, ch)
        if pos == 'sincos':
            grid = sincos_2d(H, W, ch).t().reshape(1, ch, H, W).contiguous()
            self.register_buffer('pos_grid', grid, persistent=False)
        elif pos == 'learned':
            self.pos_grid = nn.Parameter(torch.randn(1, ch, H, W) * 0.02)
        if pos != 'none':
            self.pos_proj = nn.Conv2d(ch, ch, 1)
        self.inp = MaskedConv2d('A', ch, ch, in_kernel, padding=in_kernel // 2)
        self.blocks = nn.ModuleList([GatedBlock(ch) for _ in range(n_layers)])
        self.head = nn.Sequential(nn.ReLU(), nn.Conv2d(ch, ch, 1),
                                  nn.ReLU(), nn.Conv2d(ch, K, 1))

    def forward(self, idx):
        x = self.emb(idx).permute(0, 3, 1, 2).contiguous()
        if self.pos_mode != 'none':
            x = x + self.pos_grid                       # neighbours know where they are
        x = self.inp(x)                                 # mask A hides the current pixel...
        if self.pos_mode != 'none':
            x = x + self.pos_proj(self.pos_grid)        # ...so inject its own position here
        for b in self.blocks:
            x = b(x)
        return self.head(x)                             # (B,K,H,W)


class _SeqPrior(nn.Module):
    """Shared token + position embedding for raster-order sequence priors.
    Index K is the start token. The position added at step t is the position
    of the token being PREDICTED (t), the input token is the previous one."""

    def __init__(self, K, H, W, dim, pos):
        super().__init__()
        self.K, self.H, self.W, self.N = K, H, W, H * W
        self.pos_mode = pos
        self.emb = nn.Embedding(K + 1, dim)
        if pos == 'sincos':
            self.register_buffer('pos_table', sincos_2d(H, W, dim), persistent=False)
        elif pos == 'learned':
            self.pos_emb = nn.Embedding(self.N, dim)

    def embed(self, tokens, t0=0):
        L = tokens.shape[1]
        x = self.emb(tokens)
        if self.pos_mode == 'sincos':
            x = x + self.pos_table[t0:t0 + L].unsqueeze(0)
        elif self.pos_mode == 'learned':
            x = x + self.pos_emb(torch.arange(t0, t0 + L, device=tokens.device)).unsqueeze(0)
        return x

    def shift(self, idx):
        """(B,H,W) or (B,N) targets -> (B,N) inputs: [start, t0, t1, ..., t_{N-2}]."""
        B = idx.shape[0]
        seq = idx.reshape(B, -1)
        start = torch.full((B, 1), self.K, dtype=torch.long, device=idx.device)
        return torch.cat([start, seq[:, :-1]], dim=1)

    def to_grid(self, logits, idx):
        if idx.dim() == 3:
            B = idx.shape[0]
            return logits.view(B, self.H, self.W, self.K).permute(0, 3, 1, 2)
        return logits


class LSTMPrior(_SeqPrior):
    def __init__(self, K, H, W, hidden=256, n_layers=2, dropout=0.1, pos='sincos'):
        super().__init__(K, H, W, hidden, pos)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(hidden, hidden, n_layers, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0.0)
        self.out = nn.Linear(hidden, K)

    def forward(self, idx):
        h, _ = self.lstm(self.dropout(self.embed(self.shift(idx))))
        return self.to_grid(self.out(h), idx)


class TransformerPrior(_SeqPrior):
    """Causal (decoder-style) Transformer over the token sequence, pre-norm."""

    def __init__(self, K, H, W, dim=128, heads=4, n_layers=3, ff_mult=4, dropout=0.1, pos='sincos'):
        if dim % heads:
            raise ValueError('d_model (%d) must be divisible by heads (%d).' % (dim, heads))
        super().__init__(K, H, W, dim, pos)
        self.dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * ff_mult, dropout, activation='gelu',
                                           batch_first=True, norm_first=True)
        try:
            self.encoder = nn.TransformerEncoder(layer, n_layers, norm=nn.LayerNorm(dim),
                                                 enable_nested_tensor=False)
        except TypeError:                                   # older torch
            self.encoder = nn.TransformerEncoder(layer, n_layers, norm=nn.LayerNorm(dim))
        self.out = nn.Linear(dim, K)
        causal = torch.triu(torch.full((self.N, self.N), float('-inf')), diagonal=1)
        self.register_buffer('causal', causal, persistent=False)

    def hidden(self, inp):
        L = inp.shape[1]
        x = self.dropout(self.embed(inp))
        return self.encoder(x, mask=self.causal[:L, :L])

    def forward(self, idx):
        return self.to_grid(self.out(self.hidden(self.shift(idx))), idx)

    def logits_last(self, inp):
        """inp = [start, tokens so far]; returns logits for the next token."""
        return self.out(self.hidden(inp)[:, -1])


def build_prior(spec, K, H, W):
    t = spec['type']
    if t == 'lstm':
        return LSTMPrior(K, H, W, hidden=spec['hidden'], n_layers=spec['layers'],
                         dropout=spec['dropout'], pos=spec['pos'])
    if t == 'pixelcnn':
        return PixelCNN(K, spec['ch'], spec['layers'], H, W, pos=spec['pos'], in_kernel=spec['kernel'])
    if t == 'transformer':
        return TransformerPrior(K, H, W, dim=spec['dim'], heads=spec['heads'], n_layers=spec['layers'],
                                ff_mult=spec['ff'], dropout=spec['dropout'], pos=spec['pos'])
    raise ValueError('Unknown prior type: %s' % t)


def describe_spec(spec):
    t = spec['type']
    if t == 'lstm':
        return 'LSTM  hidden=%d layers=%d dropout=%.2f pos=%s' % (
            spec['hidden'], spec['layers'], spec['dropout'], spec['pos'])
    if t == 'pixelcnn':
        return 'PixelCNN  ch=%d layers=%d in-kernel=%d pos=%s' % (
            spec['ch'], spec['layers'], spec['kernel'], spec['pos'])
    return 'Transformer  d=%d heads=%d layers=%d ff x%d dropout=%.2f pos=%s' % (
        spec['dim'], spec['heads'], spec['layers'], spec['ff'], spec['dropout'], spec['pos'])


def spec_from_legacy(cfg):
    """Checkpoints saved by v4 stored flat keys instead of a spec dict."""
    pos = cfg.get('pos', 'sincos')
    if cfg.get('type') == 'lstm':
        return dict(type='lstm', hidden=cfg.get('lstm_hidden', 256), layers=cfg.get('lstm_layers', 2),
                    dropout=0.1, pos=pos)
    return dict(type='pixelcnn', ch=cfg.get('ch', 64), layers=cfg.get('layers', 6), kernel=7, pos=pos)


# ----------------------------------------------------------------------------
# Sampling (all support a raster-order prefix of known tokens)
# ----------------------------------------------------------------------------
def _pick(logits, temperature, used_mask, greedy, top_k, top_p):
    l = logits / max(temperature, 1e-6)
    if used_mask is not None:
        l = l.masked_fill(~used_mask.unsqueeze(0), float('-inf'))
    if greedy:
        return l.argmax(1)
    if top_k and 0 < top_k < l.size(1):
        kth = torch.topk(l, top_k, dim=1)[0][:, -1:]
        l = l.masked_fill(l < kth, float('-inf'))
    if top_p and 0 < top_p < 1:
        sl, si = torch.sort(l, descending=True, dim=1)
        pr = F.softmax(sl, 1)
        remove = (pr.cumsum(1) - pr) > top_p
        sl = sl.masked_fill(remove, float('-inf'))
        l = torch.full_like(l, float('-inf')).scatter(1, si, sl)
    return torch.multinomial(F.softmax(l, 1), 1).squeeze(1)


@torch.no_grad()
def sample_pixelcnn(model, n, H, W, device, temperature, used_mask, greedy,
                    top_k, top_p, prefix, n_keep):
    model.eval()
    idx = torch.zeros(n, H, W, dtype=torch.long, device=device)
    if prefix is not None and n_keep > 0:
        idx.view(n, -1)[:, :n_keep] = prefix.reshape(prefix.shape[0], -1)[:, :n_keep].to(device)
    for i in range(H):
        for j in range(W):
            if i * W + j < n_keep:
                continue
            logits = model(idx)[:, :, i, j]
            idx[:, i, j] = _pick(logits, temperature, used_mask, greedy, top_k, top_p)
    return idx


@torch.no_grad()
def sample_lstm(model, n, H, W, device, temperature, used_mask, greedy,
                top_k, top_p, prefix, n_keep):
    model.eval()
    N = H * W
    seq = torch.zeros(n, N, dtype=torch.long, device=device)
    if prefix is not None and n_keep > 0:
        seq[:, :n_keep] = prefix.reshape(prefix.shape[0], -1)[:, :n_keep].to(device)
    hidden = None
    for t in range(N):
        prev = (torch.full((n,), model.K, dtype=torch.long, device=device)
                if t == 0 else seq[:, t - 1])
        out, hidden = model.lstm(model.embed(prev.unsqueeze(1), t), hidden)
        if t < n_keep:
            continue                                   # known token: just advance the state
        logits = model.out(out[:, 0])
        seq[:, t] = _pick(logits, temperature, used_mask, greedy, top_k, top_p)
    return seq.view(n, H, W)


@torch.no_grad()
def sample_transformer(model, n, H, W, device, temperature, used_mask, greedy,
                       top_k, top_p, prefix, n_keep):
    """No KV cache: re-runs the causal model on the growing prefix (fine for small grids)."""
    model.eval()
    N = H * W
    seq = torch.zeros(n, N, dtype=torch.long, device=device)
    if prefix is not None and n_keep > 0:
        seq[:, :n_keep] = prefix.reshape(prefix.shape[0], -1)[:, :n_keep].to(device)
    start = torch.full((n, 1), model.K, dtype=torch.long, device=device)
    for t in range(n_keep, N):
        inp = torch.cat([start, seq[:, :t]], dim=1)
        seq[:, t] = _pick(model.logits_last(inp), temperature, used_mask, greedy, top_k, top_p)
    return seq.view(n, H, W)


@torch.no_grad()
def sample_prior(model, n, H, W, device, temperature=1.0, used_mask=None, greedy=False,
                 top_k=0, top_p=1.0, prefix=None, n_keep=0):
    if used_mask is not None:
        used_mask = used_mask.to(device)
    if isinstance(model, TransformerPrior):
        fn = sample_transformer
    elif isinstance(model, LSTMPrior):
        fn = sample_lstm
    else:
        fn = sample_pixelcnn
    return fn(model, n, H, W, device, temperature, used_mask, greedy,
              top_k, top_p, prefix, n_keep)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def tensor_to_pil(t, ncols=8):
    t = t.detach().cpu().clamp(0, 1)
    if t.dim() == 3:
        t = t.unsqueeze(0)
    grid = vutils.make_grid(t, nrow=ncols, padding=2, pad_value=1.0)
    arr = (grid.permute(1, 2, 0).numpy() * 255.0).astype('uint8')
    return Image.fromarray(arr)


def stack_pil(imgs, pad=8):
    w = max(i.width for i in imgs)
    h = sum(i.height for i in imgs) + pad * (len(imgs) - 1)
    canvas = Image.new('RGB', (w, h), (255, 255, 255))
    y = 0
    for im in imgs:
        canvas.paste(im, (0, y))
        y += im.height + pad
    return canvas


def codes_to_pil(codes):
    c = codes.detach().cpu().float()
    if c.dim() == 3:
        c = c.unsqueeze(1)
    c = (c / max(1.0, float(c.max().item()))).clamp(0, 1).repeat(1, 3, 1, 1)
    up = F.interpolate(c, size=(64, 64), mode='nearest')
    return tensor_to_pil(up, ncols=min(8, up.shape[0]))


def fit_img(img, maxw, maxh):
    s = min(maxw / img.width, maxh / img.height, 1.0)
    if s < 1.0:
        img = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))), Image.BICUBIC)
    return img


def safe_torch_load(path, map_location='cpu'):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


@torch.no_grad()
def random_preview(model, ds, n, dev):
    """Random dataset images -> encode -> decode. Returns (PIL grid, roundtrip err).
    Grid: first n = originals, next n = reconstructions."""
    model.eval()
    idxs = random.sample(range(len(ds)), min(n, len(ds)))
    x = torch.stack([ds[i] for i in idxs]).to(dev)
    rec, _, _, _, codes = model(x)
    rt_err = (model.decode_indices(codes) - rec).abs().max().item()
    return tensor_to_pil(torch.cat([x.cpu(), rec.cpu()]), ncols=min(8, len(idxs))), rt_err


@torch.no_grad()
def encode_codes(vqvae, ds, dev, stop_event, bs=64):
    """Dataset -> int32 code tensor (N,H,W); None if the user pressed Stop."""
    out = []
    for x in DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0):
        if stop_event.is_set():
            return None
        out.append(vqvae.encode(x.to(dev))[2].cpu().to(torch.int32))
    return torch.cat(out) if out else None


@torch.no_grad()
def eval_nll(model, codes, K, dev, bs=128):
    """Mean NLL in nats per token, no label smoothing, eval mode."""
    model.eval()
    tot, cnt = 0.0, 0
    for i in range(0, len(codes), bs):
        idx = codes[i:i + bs].long().to(dev)
        logits = model(idx)
        tot += F.cross_entropy(logits.permute(0, 2, 3, 1).reshape(-1, K),
                               idx.reshape(-1), reduction='sum').item()
        cnt += idx.numel()
    return tot / max(cnt, 1)


# ----------------------------------------------------------------------------
# Worker threads  (never touch Tk variables in here; everything arrives via cfg)
# ----------------------------------------------------------------------------
def train_vqvae(app, cfg, model, ds_train, ds_plain):
    """cfg['preview_every'] = 0 disables previews; otherwise a preview is produced every
    N cfg['preview_unit'] ('steps' or 'epochs'), plus one at the very end."""
    try:
        dev = torch.device(cfg['device'])
        torch.manual_seed(cfg['seed']); random.seed(cfg['seed'])
        K = 2 ** cfg['D']
        every, unit = cfg['preview_every'], cfg['preview_unit']
        model = model.to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
        loader = DataLoader(ds_train, batch_size=cfg['batch'], shuffle=True, num_workers=0)
        steps = len(loader)
        if steps == 0:
            app.q.put(('error', 'Dataset is empty.')); app.q.put(('busy', False)); return
        app.q.put(('log', '[VQ-VAE] params={:,} steps/epoch={} D={} K={} img={} latent={}'.format(
            sum(p.numel() for p in model.parameters()), steps, cfg['D'], K,
            cfg['img_size'], cfg['latent_size'])))

        def preview():
            img, rt_err = random_preview(model, ds_plain, cfg['preview_n'], dev)
            model.train()
            if rt_err > 1e-3:
                app.q.put(('log', '[VQ-VAE] WARNING: decode_indices mismatch err=%.4f' % rt_err))
            app.q.put(('vqvae_preview', img))

        gstep, stopped = 0, False
        for epoch in range(1, cfg['epochs'] + 1):
            model.train()
            tot, n = 0.0, 0
            seen = torch.zeros(K, dtype=torch.bool)
            for step, x in enumerate(loader, 1):
                if app.stop_event.is_set():
                    stopped = True
                    break
                x = x.to(dev)
                _, loss, rec, vql, codes = model(x)
                opt.zero_grad(set_to_none=True)
                loss.backward(); opt.step()
                tot += loss.item(); n += 1; gstep += 1
                seen[codes.detach().flatten().cpu()] = True
                if step % 10 == 0 or step == steps:
                    app.q.put(('vqvae_step', dict(epoch=epoch, epochs=cfg['epochs'],
                        step=step, steps=steps, loss=loss.item())))
                if every > 0 and unit == 'steps' and gstep % every == 0:
                    preview()
            if stopped:
                app.q.put(('log', '[VQ-VAE] stopped by user'))
                break
            app.q.put(('vqvae_epoch', dict(epoch=epoch, epochs=cfg['epochs'], loss=tot / max(n, 1),
                                           used=int(seen.sum()), K=K)))
            if every > 0 and unit == 'epochs' and epoch % every == 0:
                preview()

        if every > 0:
            preview()                                   # always show the final state
        model.eval().cpu()
        app.q.put(('vqvae_done', dict(model=model, cfg=cfg)))
    except Exception as e:
        traceback.print_exc()
        app.q.put(('error', 'VQ-VAE training failed: %s' % e)); app.q.put(('busy', False))


def train_prior(app, cfg, model, ds_plain, ds_aug):
    try:
        vqvae, vqcfg = app.vqvae, app.vqvae_cfg
        dev = torch.device(cfg['device'])
        torch.manual_seed(cfg['seed']); random.seed(cfg['seed'])
        vqvae = vqvae.to(dev).eval()
        K = 2 ** vqcfg['D']
        every, unit = cfg['preview_every'], cfg['preview_unit']

        # --- split by IMAGE so augmented copies never leak into validation ---
        N = len(ds_plain)
        perm = torch.randperm(N, generator=torch.Generator().manual_seed(cfg['seed'])).tolist()
        n_val = int(round(N * cfg['val_frac'])) if N >= 20 else 0
        val_idx, tr_idx = perm[:n_val], perm[n_val:]

        app.q.put(('log', '[Prior] encoding dataset (%d train / %d val images) ...'
                   % (len(tr_idx), len(val_idx))))
        parts = [encode_codes(vqvae, Subset(ds_plain, tr_idx), dev, app.stop_event)]
        if ds_aug is not None:
            for _ in range(cfg['aug_copies'] - 1):
                parts.append(encode_codes(vqvae, Subset(ds_aug, tr_idx), dev, app.stop_event))
        codes_val = (encode_codes(vqvae, Subset(ds_plain, val_idx), dev, app.stop_event)
                     if n_val > 0 else None)
        if any(p is None for p in parts) or (n_val > 0 and codes_val is None):
            app.q.put(('log', '[Prior] stopped during encoding.')); app.q.put(('busy', False)); return
        codes_tr = torch.cat(parts)
        H, W = codes_tr.shape[1], codes_tr.shape[2]

        allc = codes_tr if codes_val is None else torch.cat([codes_tr, codes_val])
        freq = torch.bincount(allc.long().flatten(), minlength=K).float()
        used_mask = freq > 0
        app.used_codes_mask = used_mask
        p = freq / freq.sum()
        unigram = -(p[p > 0] * p[p > 0].log()).sum().item()
        app.q.put(('log', '[Prior] train codes={} unique={}/{} | context-free baseline NLL={:.3f} '
                          'nats/token | params={:,}'.format(
            tuple(codes_tr.shape), int(used_mask.sum()), K, unigram,
            sum(q.numel() for q in model.parameters()))))

        model = model.to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
        dl = DataLoader(TensorDataset(codes_tr), batch_size=cfg['batch'], shuffle=True, num_workers=0)
        steps = len(dl)
        ls = cfg.get('label_smoothing', 0.0)
        use_mask = cfg.get('use_code_mask', True)
        best_val, best_state = float('inf'), None

        def preview():
            idx = sample_prior(model, cfg['preview_n'], H, W, dev, 1.0,
                               used_mask if use_mask else None, False)
            with torch.no_grad():
                imgs = vqvae.decode_indices(idx)
            model.train()
            app.q.put(('pcnn_preview', tensor_to_pil(imgs, ncols=min(8, cfg['preview_n']))))

        gstep, stopped = 0, False
        for epoch in range(1, cfg['epochs'] + 1):
            model.train()
            tot, n = 0.0, 0
            for step, (idx,) in enumerate(dl, 1):
                if app.stop_event.is_set():
                    stopped = True
                    break
                idx = idx.long().to(dev)
                logits = model(idx)
                loss = F.cross_entropy(logits.permute(0, 2, 3, 1).reshape(-1, K),
                                       idx.reshape(-1), label_smoothing=ls)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tot += loss.item(); n += 1; gstep += 1
                if step % 10 == 0 or step == steps:
                    app.q.put(('pcnn_step', dict(epoch=epoch, epochs=cfg['epochs'],
                        step=step, steps=steps, loss=loss.item())))
                if every > 0 and unit == 'steps' and gstep % every == 0:
                    preview()
            if stopped:
                app.q.put(('log', '[Prior] stopped by user'))
                break

            val = None
            if codes_val is not None:
                val = eval_nll(model, codes_val, K, dev)
                if val < best_val:
                    best_val = val
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            app.q.put(('pcnn_epoch', dict(epoch=epoch, epochs=cfg['epochs'],
                                          loss=tot / max(n, 1), val=val)))
            if every > 0 and unit == 'epochs' and epoch % every == 0:
                preview()

        if cfg.get('restore_best') and best_state is not None:
            model.load_state_dict(best_state)
            app.q.put(('log', '[Prior] restored best-validation weights (val NLL %.4f)' % best_val))
        if every > 0:
            preview()                                   # always show the final state
        vqvae.cpu()
        model.eval().cpu()
        app.q.put(('pcnn_done', dict(model=model, cfg=dict(spec=cfg['spec'], K=K, H=H, W=W))))
    except Exception as e:
        traceback.print_exc()
        app.q.put(('error', 'Prior training failed: %s' % e)); app.q.put(('busy', False))


def run_sample(app, n, sa):
    try:
        vqvae, prior = app.vqvae, app.pixelcnn
        dev = sa['dev']
        vqvae = vqvae.to(dev).eval(); prior = prior.to(dev).eval()
        Hl = app.vqvae_cfg['latent_size']
        with torch.no_grad():
            idx = sample_prior(prior, n, Hl, Hl, dev, sa['temp'], sa['used_mask'],
                               sa['greedy'], sa['top_k'], sa['top_p'])
            imgs = vqvae.decode_indices(idx)
        app.q.put(('infer_img', tensor_to_pil(imgs, ncols=min(8, n))))
        app.q.put(('infer_codes', codes_to_pil(idx[:min(8, n)])))
        app.q.put(('log', '[Inference] sampled %d (T=%.2f top_k=%d top_p=%.2f greedy=%s mask=%s)' %
                   (n, sa['temp'], sa['top_k'], sa['top_p'], sa['greedy'], sa['used_mask'] is not None)))
        vqvae.cpu(); prior.cpu()
    except Exception as e:
        traceback.print_exc(); app.q.put(('error', str(e)))
    finally:
        app.q.put(('busy', False))


def run_reconstruct(app, path, dev):
    try:
        vqvae = app.vqvae.to(dev).eval()
        S = app.vqvae_cfg['img_size']
        tf = T.Compose([T.Resize(S), T.CenterCrop(S), T.ToTensor()])
        x = tf(Image.open(path).convert('RGB')).unsqueeze(0).to(dev)
        with torch.no_grad():
            rec, _, _, _, codes = vqvae(x)
            rec_rt = vqvae.decode_indices(codes)
            rt_err = (rec_rt - rec).abs().max().item()
            comp = torch.cat([x[0].cpu(), rec[0].cpu(), rec_rt[0].cpu()], dim=2)
        app.q.put(('infer_img', tensor_to_pil(comp.unsqueeze(0), ncols=1)))
        app.q.put(('infer_codes', codes_to_pil(codes)))
        app.q.put(('log', '[Inference] original | forward-recon | decode_indices-recon (err=%.6f)' % rt_err))
        vqvae.cpu()
    except Exception as e:
        traceback.print_exc(); app.q.put(('error', str(e)))
    finally:
        app.q.put(('busy', False))


def run_infill(app, a):
    """Occlude the bottom a['cut']% of an image, keep the fully-visible token rows
    as a prefix, let the prior generate the remaining tokens in raster order."""
    try:
        vqvae, prior = app.vqvae, app.pixelcnn
        cfg = app.vqvae_cfg
        S, Hl = cfg['img_size'], cfg['latent_size']
        if getattr(prior, 'H', Hl) != Hl or getattr(prior, 'K', 2 ** cfg['D']) != 2 ** cfg['D']:
            raise RuntimeError('Prior does not match the VQ-VAE latent grid. Re-initialise the prior.')
        dev, n = a['dev'], a['n']
        vqvae = vqvae.to(dev).eval(); prior = prior.to(dev).eval()

        tf = T.Compose([T.Resize(S), T.CenterCrop(S), T.ToTensor()])
        x = tf(Image.open(a['path']).convert('RGB')).unsqueeze(0).to(dev)
        y_cut = min(S - 1, max(0, int(round(S * (1.0 - a['cut'] / 100.0)))))
        stride = S // Hl
        keep_rows = y_cut // stride                   # only rows entirely above the cut
        n_keep = keep_rows * Hl
        masked = x.clone()
        masked[:, :, y_cut:, :] = 0.5                 # the model never sees the hidden pixels

        with torch.no_grad():
            codes_m = vqvae.encode(masked)[2]
            rec_full = vqvae.decode_indices(vqvae.encode(x)[2])     # tokenizer ceiling
            idx = sample_prior(prior, n, Hl, Hl, dev, a['temp'], a['used_mask'], a['greedy'],
                               a['top_k'], a['top_p'], prefix=codes_m, n_keep=n_keep)
            imgs = vqvae.decode_indices(idx)
        if a['paste']:
            imgs = imgs.clone()
            imgs[:, :, :y_cut, :] = x[:, :, :y_cut, :]

        top = tensor_to_pil(torch.cat([x, masked, rec_full]).cpu(), ncols=3)
        bot = tensor_to_pil(imgs.cpu(), ncols=min(8, n))
        app.q.put(('infill_img', stack_pil([top, bot])))
        app.q.put(('log', '[Infill] cut %d%% (pixel row >= %d): kept %d/%d token rows, generated %d/%d tokens'
                   % (a['cut'], y_cut, keep_rows, Hl, Hl * Hl - n_keep, Hl * Hl)))
        vqvae.cpu(); prior.cpu()
    except Exception as e:
        traceback.print_exc(); app.q.put(('error', str(e)))
    finally:
        app.q.put(('busy', False))


# ----------------------------------------------------------------------------
# GUI
# ----------------------------------------------------------------------------
OK, WARN, ERR, MUTED = '#1a7f37', '#b35900', '#c62828', '#666666'
PRIOR_TYPES = ['lstm', 'pixelcnn', 'transformer']
POS_MODES = ['sincos', 'learned', 'none']


def fmt_time(s):
    s = int(max(0, s))
    return '%d:%02d:%02d' % (s // 3600, (s // 60) % 60, s % 60) if s >= 3600 else '%d:%02d' % (s // 60, s % 60)


class App:
    def __init__(self, root):
        self.root = root
        root.title('LFQ-VAE + Prior Studio')
        root.geometry('1240x960')
        root.minsize(980, 700)
        self.q = queue.Queue()
        self.stop_event = threading.Event()
        self.busy = False
        self.paths = []
        self._ds_cache = {}
        self.vqvae = None
        self.vqvae_cfg = None
        self.pixelcnn = None            # the prior (any type); name kept for the worker functions
        self.pcnn_cfg = None            # dict(spec=..., K=..., H=..., W=...)
        self.used_codes_mask = None
        self.infill_path = None
        self._photos = {}
        self._t0 = None
        self._f0 = 0.0
        self._fonts = []
        self._setup_style()
        self._make_vars()
        self._build_ui()
        self.set_device.trace_add('write', lambda *_: self._refresh_status())
        root.report_callback_exception = self._on_tk_error
        self._refresh_status()
        self._update_arch_label()
        root.after(80, self._poll)

    # ---------------- style ----------------
    def _setup_style(self):
        s = ttk.Style()
        try:
            names = s.theme_names()
            if sys.platform == 'win32' and 'vista' in names:
                s.theme_use('vista')
            elif sys.platform.startswith('linux') and 'clam' in names:
                s.theme_use('clam')
        except tk.TclError:
            pass
        base = tkfont.nametofont('TkDefaultFont')
        bold = tkfont.Font(family=base.cget('family'), size=base.cget('size'), weight='bold')
        title = tkfont.Font(family=base.cget('family'), size=base.cget('size') + 6, weight='bold')
        self._fonts += [bold, title]
        s.configure('TNotebook.Tab', padding=(14, 6))
        s.configure('TButton', padding=(10, 4))
        s.configure('Accent.TButton', font=bold)
        s.configure('TLabelframe.Label', font=bold)
        s.configure('Title.TLabel', font=title)
        s.configure('Hint.TLabel', foreground=MUTED)
        s.configure('Preview.TLabel', background='#e9e9e9')

    # ---------------- variables ----------------
    def _make_vars(self):
        V = lambda cls, v: cls(value=v)
        # VQ-VAE training tab
        self.vq_epochs = V(tk.IntVar, 20); self.vq_batch = V(tk.IntVar, 32); self.vq_lr = V(tk.DoubleVar, 2e-4)
        self.vq_prev_every = V(tk.IntVar, 1); self.vq_prev_unit = V(tk.StringVar, 'epochs')
        self.vq_prev_n = V(tk.IntVar, 8)
        # prior training tab
        self.pr_type = V(tk.StringVar, 'lstm')
        self.pr_epochs = V(tk.IntVar, 30); self.pr_batch = V(tk.IntVar, 32); self.pr_lr = V(tk.DoubleVar, 1e-3)
        self.pr_ls = V(tk.DoubleVar, 0.05); self.pr_val = V(tk.DoubleVar, 0.05)
        self.pr_copies = V(tk.IntVar, 4); self.pr_restore = V(tk.BooleanVar, True)
        self.pr_prev_every = V(tk.IntVar, 1); self.pr_prev_unit = V(tk.StringVar, 'epochs')
        self.pr_prev_n = V(tk.IntVar, 8)
        # settings: general / VQ-VAE
        self.set_device = V(tk.StringVar, 'cpu'); self.set_seed = V(tk.IntVar, 1234)
        self.set_img_size = V(tk.IntVar, 64); self.set_latent = V(tk.IntVar, 8)
        self.set_D = V(tk.IntVar, 8); self.set_hidden = V(tk.IntVar, 128)
        self.set_beta = V(tk.DoubleVar, 0.1); self.set_entropy = V(tk.DoubleVar, 0.1)
        # settings: augmentation
        self.set_aug_h = V(tk.BooleanVar, True); self.set_aug_v = V(tk.BooleanVar, False)
        self.set_aug_rot = V(tk.DoubleVar, 0.0); self.set_aug_zoom = V(tk.DoubleVar, 1.0)
        self.set_aug_jit = V(tk.DoubleVar, 0.0)
        # settings: one block per prior type
        self.lstm_hidden = V(tk.IntVar, 256); self.lstm_layers = V(tk.IntVar, 2)
        self.lstm_dropout = V(tk.DoubleVar, 0.1); self.lstm_pos = V(tk.StringVar, 'sincos')
        self.pc_ch = V(tk.IntVar, 64); self.pc_layers = V(tk.IntVar, 6)
        self.pc_kernel = V(tk.IntVar, 7); self.pc_pos = V(tk.StringVar, 'sincos')
        self.tf_dim = V(tk.IntVar, 128); self.tf_heads = V(tk.IntVar, 4); self.tf_layers = V(tk.IntVar, 3)
        self.tf_ff = V(tk.IntVar, 4); self.tf_dropout = V(tk.DoubleVar, 0.1); self.tf_pos = V(tk.StringVar, 'sincos')
        # sampling (shared by Inference + Infill)
        self.inf_n = V(tk.IntVar, 16); self.inf_temp = V(tk.DoubleVar, 1.0)
        self.inf_topk = V(tk.IntVar, 0); self.inf_topp = V(tk.DoubleVar, 1.0)
        self.inf_greedy = V(tk.BooleanVar, False); self.set_use_code_mask = V(tk.BooleanVar, True)
        # infill
        self.fill_cut = V(tk.IntVar, 40); self.fill_n = V(tk.IntVar, 8); self.fill_paste = V(tk.BooleanVar, True)

    # ---------------- small widget helpers ----------------
    def _spin(self, parent, var, lo, hi, inc, width=8):
        return ttk.Spinbox(parent, from_=lo, to=hi, increment=inc, width=width, textvariable=var)

    def _combo(self, parent, var, values, width=10, state='readonly'):
        return ttk.Combobox(parent, textvariable=var, values=values, width=width, state=state)

    def _inline(self, parent, label, widget, padx=(0, 14)):
        ttk.Label(parent, text=label).pack(side='left', padx=(0, 4))
        widget.pack(side='left', padx=padx)

    def _row(self, parent, r, label, widget, hint=''):
        ttk.Label(parent, text=label).grid(row=r, column=0, sticky='w', padx=(0, 14), pady=4)
        widget.grid(row=r, column=1, sticky='w', pady=4)
        if hint:
            ttk.Label(parent, text=hint, style='Hint.TLabel').grid(row=r, column=2, sticky='w', padx=14)

    def _preview_box(self, parent, title='Preview'):
        box = ttk.LabelFrame(parent, text=title, padding=4)
        box.pack(fill='both', expand=True, pady=(8, 0))
        lbl = ttk.Label(box, anchor='center', style='Preview.TLabel')
        lbl.pack(fill='both', expand=True)
        return lbl

    def _progress_block(self, parent):
        prog = ttk.Progressbar(parent, maximum=100)
        prog.pack(fill='x', pady=(8, 2))
        status = ttk.Label(parent, text='Idle.', style='Hint.TLabel')
        status.pack(anchor='w')
        return prog, status

    def _preview_rate(self, parent, every_var, unit_var, n_var):
        ttk.Label(parent, text='Preview every').pack(side='left', padx=(0, 4))
        self._spin(parent, every_var, 0, 10 ** 9, 1, 7).pack(side='left')
        self._combo(parent, unit_var, ['steps', 'epochs'], 7).pack(side='left', padx=(4, 4))
        ttk.Label(parent, text='(0 = off)', style='Hint.TLabel').pack(side='left', padx=(0, 14))
        self._inline(parent, 'Images', self._spin(parent, n_var, 1, 64, 1, 5))

    # ---------------- layout ----------------
    def _build_ui(self):
        hdr = ttk.Frame(self.root, padding=(10, 8, 10, 2))
        hdr.pack(fill='x')
        ttk.Label(hdr, text='LFQ-VAE + Prior Studio', style='Title.TLabel').pack(side='left')
        self.st_dev = ttk.Label(hdr, style='Hint.TLabel'); self.st_dev.pack(side='right', padx=(14, 0))
        self.st_prior = ttk.Label(hdr); self.st_prior.pack(side='right', padx=(14, 0))
        self.st_vq = ttk.Label(hdr); self.st_vq.pack(side='right', padx=(14, 0))
        self.st_data = ttk.Label(hdr); self.st_data.pack(side='right', padx=(14, 0))
        ttk.Separator(self.root).pack(fill='x', padx=8, pady=(4, 6))

        paned = ttk.PanedWindow(self.root, orient='vertical')
        paned.pack(fill='both', expand=True, padx=8, pady=(0, 8))
        top = ttk.Frame(paned)
        self.nb = ttk.Notebook(top)
        self.nb.pack(fill='both', expand=True)
        paned.add(top, weight=6)

        for name, attr in [('Dataset', 'tab_dataset'), ('VQ-VAE', 'tab_vqvae'), ('Prior', 'tab_prior'),
                           ('Inference', 'tab_inference'), ('Infill', 'tab_infill'),
                           ('Settings', 'tab_settings')]:
            frame = ttk.Frame(self.nb, padding=10)
            self.nb.add(frame, text=name)
            setattr(self, attr, frame)
        self._build_ds(self.tab_dataset)
        self._build_vq(self.tab_vqvae)
        self._build_pc(self.tab_prior)
        self._build_inf(self.tab_inference)
        self._build_fill(self.tab_infill)
        self._build_set(self.tab_settings)

        logf = ttk.LabelFrame(paned, text='Log', padding=4)
        paned.add(logf, weight=1)
        bar = ttk.Frame(logf); bar.pack(side='right', fill='y', padx=(4, 0))
        ttk.Button(bar, text='Clear', command=lambda: self.log_text.delete('1.0', 'end')).pack()
        self.log_text = tk.Text(logf, height=7, wrap='word', font=tkfont.nametofont('TkFixedFont'))
        self.log_text.tag_configure('err', foreground=ERR)
        self.log_text.tag_configure('warn', foreground=WARN)
        self.log_text.tag_configure('ok', foreground=OK)
        sb = ttk.Scrollbar(logf, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=sb.set)
        sb.pack(side='right', fill='y')
        self.log_text.pack(side='left', fill='both', expand=True)

    # ---- dataset tab ----
    def _build_ds(self, f):
        row = ttk.Frame(f); row.pack(fill='x')
        ttk.Button(row, text='Select images...', style='Accent.TButton',
                   command=self.on_choose_files).pack(side='left')
        ttk.Button(row, text='Shuffle preview', command=self.on_shuffle_ds).pack(side='left', padx=6)
        ttk.Button(row, text='Preview augmentation', command=self.on_preview_aug).pack(side='left')
        self.ds_count = ttk.Label(row, text='0 images', style='Hint.TLabel')
        self.ds_count.pack(side='right')
        self.ds_preview = self._preview_box(f, 'Samples (augmentation preview: top row plain, bottom row augmented)')

    # ---- VQ-VAE tab ----
    def _build_vq(self, f):
        tr = ttk.LabelFrame(f, text='Training', padding=8)
        tr.pack(fill='x')
        r1 = ttk.Frame(tr); r1.pack(fill='x')
        self._inline(r1, 'Epochs', self._spin(r1, self.vq_epochs, 1, 10 ** 6, 1, 6))
        self._inline(r1, 'Batch', self._spin(r1, self.vq_batch, 1, 10 ** 6, 16, 6))
        self._inline(r1, 'LR', self._spin(r1, self.vq_lr, 0.0, 1.0, 0.0001, 9))
        r2 = ttk.Frame(tr); r2.pack(fill='x', pady=(6, 0))
        self._preview_rate(r2, self.vq_prev_every, self.vq_prev_unit, self.vq_prev_n)
        ttk.Label(tr, text='Architecture, image/latent size and loss weights: Settings > VQ-VAE.',
                  style='Hint.TLabel').pack(anchor='w', pady=(6, 0))

        row = ttk.Frame(f); row.pack(fill='x', pady=(10, 0))
        self.btn_vq_init = ttk.Button(row, text='Initialize model', command=self.on_init_vqvae)
        self.btn_vq_init.pack(side='left')
        self.btn_vq_start = ttk.Button(row, text='Start training', style='Accent.TButton',
                                       command=self.on_start_vqvae, state='disabled')
        self.btn_vq_start.pack(side='left', padx=6)
        self.btn_vq_stop = ttk.Button(row, text='Stop', command=self.on_stop, state='disabled')
        self.btn_vq_stop.pack(side='left')
        self.btn_vq_prev = ttk.Button(row, text='New random preview', command=self.on_vq_preview)
        self.btn_vq_prev.pack(side='left', padx=(18, 0))
        self.btn_vq_load = ttk.Button(row, text='Load...', command=self.on_load_vqvae)
        self.btn_vq_load.pack(side='right')
        self.btn_vq_save = ttk.Button(row, text='Save...', command=self.on_save_vqvae)
        self.btn_vq_save.pack(side='right', padx=6)
        self.vq_model_label = ttk.Label(f, text='Model: not initialized', foreground=WARN)
        self.vq_model_label.pack(anchor='w', pady=(8, 0))
        self.vq_prog, self.vq_status = self._progress_block(f)
        self.vq_preview = self._preview_box(f, 'Random images: first half originals, second half encode -> decode')

    # ---- prior tab ----
    def _build_pc(self, f):
        a = ttk.LabelFrame(f, text='Architecture', padding=8)
        a.pack(fill='x')
        r = ttk.Frame(a); r.pack(fill='x')
        ttk.Label(r, text='Prior type').pack(side='left', padx=(0, 4))
        cb = self._combo(r, self.pr_type, PRIOR_TYPES, 13)
        cb.pack(side='left', padx=(0, 12))
        cb.bind('<<ComboboxSelected>>', lambda e: self._update_arch_label())
        ttk.Button(r, text='Edit architecture settings...', command=self.on_edit_arch).pack(side='left')
        self.pr_arch_label = ttk.Label(a, style='Hint.TLabel')
        self.pr_arch_label.pack(anchor='w', pady=(6, 0))

        tr = ttk.LabelFrame(f, text='Training', padding=8)
        tr.pack(fill='x', pady=(8, 0))
        r1 = ttk.Frame(tr); r1.pack(fill='x')
        self._inline(r1, 'Epochs', self._spin(r1, self.pr_epochs, 1, 10 ** 6, 1, 6))
        self._inline(r1, 'Batch', self._spin(r1, self.pr_batch, 1, 10 ** 6, 16, 6))
        self._inline(r1, 'LR', self._spin(r1, self.pr_lr, 0.0, 1.0, 0.0005, 9))
        self._inline(r1, 'Label smoothing', self._spin(r1, self.pr_ls, 0.0, 0.5, 0.01, 6))
        r2 = ttk.Frame(tr); r2.pack(fill='x', pady=(6, 0))
        self._inline(r2, 'Validation fraction', self._spin(r2, self.pr_val, 0.0, 0.5, 0.05, 6))
        self._inline(r2, 'Encoded copies per image', self._spin(r2, self.pr_copies, 1, 64, 1, 5))
        ttk.Checkbutton(r2, text='Restore best-validation weights',
                        variable=self.pr_restore).pack(side='left')
        r3 = ttk.Frame(tr); r3.pack(fill='x', pady=(6, 0))
        self._preview_rate(r3, self.pr_prev_every, self.pr_prev_unit, self.pr_prev_n)
        ttk.Label(tr, text='Copies > 1 only matters if augmentation is enabled (Settings > Augmentation). '
                           'Sampling previews cost one full autoregressive pass each: raise the interval '
                           'if previews slow training down.', style='Hint.TLabel',
                  wraplength=1100).pack(anchor='w', pady=(6, 0))

        row = ttk.Frame(f); row.pack(fill='x', pady=(10, 0))
        self.btn_pc_init = ttk.Button(row, text='Initialize prior', command=self.on_init_pcnn)
        self.btn_pc_init.pack(side='left')
        self.btn_pc_start = ttk.Button(row, text='Start training', style='Accent.TButton',
                                       command=self.on_start_pcnn, state='disabled')
        self.btn_pc_start.pack(side='left', padx=6)
        self.btn_pc_stop = ttk.Button(row, text='Stop', command=self.on_stop, state='disabled')
        self.btn_pc_stop.pack(side='left')
        self.btn_pc_load = ttk.Button(row, text='Load...', command=self.on_load_pcnn)
        self.btn_pc_load.pack(side='right')
        self.btn_pc_save = ttk.Button(row, text='Save...', command=self.on_save_pcnn)
        self.btn_pc_save.pack(side='right', padx=6)
        self.pc_model_label = ttk.Label(f, text='Prior: not initialized', foreground=WARN)
        self.pc_model_label.pack(anchor='w', pady=(8, 0))
        self.pc_prog, self.pc_status = self._progress_block(f)
        self.pc_preview = self._preview_box(f, 'Samples from the prior')

    # ---- sampling controls shared by Inference + Infill ----
    def _sampling_controls(self, row):
        self._inline(row, 'Temp', self._spin(row, self.inf_temp, 0.05, 3.0, 0.05, 5), (0, 10))
        self._inline(row, 'Top-k (0=off)', self._spin(row, self.inf_topk, 0, 4096, 8, 5), (0, 10))
        self._inline(row, 'Top-p (1=off)', self._spin(row, self.inf_topp, 0.05, 1.0, 0.05, 5), (0, 10))
        ttk.Checkbutton(row, text='Greedy', variable=self.inf_greedy).pack(side='left', padx=(0, 10))
        ttk.Checkbutton(row, text='Mask unused codes', variable=self.set_use_code_mask).pack(side='left')

    def _build_inf(self, f):
        box = ttk.LabelFrame(f, text='Sampling', padding=8)
        box.pack(fill='x')
        r1 = ttk.Frame(box); r1.pack(fill='x')
        self._inline(r1, 'Samples', self._spin(r1, self.inf_n, 1, 64, 1, 5), (0, 14))
        self._sampling_controls(r1)
        row = ttk.Frame(f); row.pack(fill='x', pady=(10, 0))
        self.btn_sample = ttk.Button(row, text='Sample from prior', style='Accent.TButton', command=self.on_sample)
        self.btn_sample.pack(side='left')
        self.btn_recon = ttk.Button(row, text='Test roundtrip on image...', command=self.on_reconstruct)
        self.btn_recon.pack(side='left', padx=6)
        self.inf_preview = self._preview_box(f, 'Output')
        cb = ttk.LabelFrame(f, text='Codes (brightness = code index)', padding=4)
        cb.pack(fill='x', pady=(8, 0))
        self.inf_codes = ttk.Label(cb, anchor='center', style='Preview.TLabel')
        self.inf_codes.pack(fill='x')

    def _build_fill(self, f):
        box = ttk.LabelFrame(f, text='Occlusion infill', padding=8)
        box.pack(fill='x')
        r0 = ttk.Frame(box); r0.pack(fill='x')
        self.btn_fill_pick = ttk.Button(r0, text='Select image...', command=self.on_pick_infill)
        self.btn_fill_pick.pack(side='left')
        self.fill_label = ttk.Label(r0, text='(no image)', style='Hint.TLabel')
        self.fill_label.pack(side='left', padx=10)
        r1 = ttk.Frame(box); r1.pack(fill='x', pady=(8, 0))
        self._inline(r1, 'Cut from bottom %', self._spin(r1, self.fill_cut, 1, 99, 5, 5))
        self._inline(r1, 'Samples', self._spin(r1, self.fill_n, 1, 32, 1, 5))
        ttk.Checkbutton(r1, text='Paste original visible pixels', variable=self.fill_paste).pack(side='left')
        r2 = ttk.Frame(box); r2.pack(fill='x', pady=(8, 0))
        self._sampling_controls(r2)
        self.btn_infill = ttk.Button(r2, text='Run infill', style='Accent.TButton', command=self.on_infill)
        self.btn_infill.pack(side='right')
        self.fill_preview = self._preview_box(
            f, 'Top: original | occluded input | tokenizer reconstruction (best case). Below: prior completions')

    # ---- settings tab (sub-pages) ----
    def _page(self, title, hint=''):
        page = ttk.Frame(self.set_nb, padding=14)
        self.set_nb.add(page, text=title)
        if hint:
            ttk.Label(page, text=hint, style='Hint.TLabel', wraplength=900).grid(
                row=99, column=0, columnspan=3, sticky='w', pady=(14, 0))
        return page

    def _build_set(self, f):
        self.set_nb = ttk.Notebook(f)
        self.set_nb.pack(fill='both', expand=True)
        self.arch_pages = {}

        p = self._page('General')
        self._row(p, 0, 'Device', self._combo(p, self.set_device, ['cpu', 'cuda'], 10),
                  'cuda available: %s' % ('yes' if torch.cuda.is_available() else 'no'))
        self._row(p, 1, 'Seed', self._spin(p, self.set_seed, 0, 10 ** 9, 1, 10))

        p = self._page('VQ-VAE', 'Changing size/bits/hidden requires re-initializing the VQ-VAE (and the prior).')
        self._row(p, 0, 'Image size (px)',
                  self._combo(p, self.set_img_size, [16, 32, 48, 64, 96, 128, 192, 256], 10, 'normal'),
                  'input images are resized + center-cropped to this')
        self._row(p, 1, 'Latent grid size',
                  self._combo(p, self.set_latent, [1, 2, 4, 8, 16, 32, 64], 10, 'normal'),
                  'tokens per side; image/latent must be a power of 2')
        self._row(p, 2, 'Latent bits D (<=%d)' % MAX_D, self._spin(p, self.set_D, 1, MAX_D, 1, 10),
                  'codebook size K = 2^D')
        self._row(p, 3, 'Hidden channels', self._spin(p, self.set_hidden, 16, 1024, 16, 10))
        self._row(p, 4, 'Commit beta', self._spin(p, self.set_beta, 0.0, 5.0, 0.05, 10))
        self._row(p, 5, 'Entropy weight', self._spin(p, self.set_entropy, 0.0, 5.0, 0.01, 10),
                  'encourages balanced, confident bits (codebook usage)')
        self.k_label = ttk.Label(p, text='')
        self.k_label.grid(row=6, column=0, columnspan=3, sticky='w', pady=(10, 0))
        for v in (self.set_D, self.set_img_size, self.set_latent):
            v.trace_add('write', lambda *_: self._update_k_label())
        self._update_k_label()

        p = self._page('Augmentation', 'Rotation leaves black corners; combine with zoom < 1 to hide them. '
                                       'The VQ-VAE sees fresh augmentations every epoch; the prior sees a fixed '
                                       'number of encoded copies (Prior tab).')
        r = 0
        ttk.Checkbutton(p, text='Horizontal flip', variable=self.set_aug_h).grid(row=r, column=1, sticky='w', pady=4); r += 1
        ttk.Checkbutton(p, text='Vertical flip', variable=self.set_aug_v).grid(row=r, column=1, sticky='w', pady=4); r += 1
        self._row(p, r, 'Rotation +/- degrees', self._spin(p, self.set_aug_rot, 0, 90, 5, 8), '0 = off'); r += 1
        self._row(p, r, 'Zoom min scale', self._spin(p, self.set_aug_zoom, 0.3, 1.0, 0.05, 8),
                  'random crop covering at least this fraction of the side; 1 = off'); r += 1
        self._row(p, r, 'Colour jitter', self._spin(p, self.set_aug_jit, 0.0, 0.8, 0.05, 8),
                  'brightness / contrast / saturation strength; 0 = off'); r += 1
        ttk.Button(p, text='Preview augmentation', command=self.on_preview_aug_tab).grid(
            row=r, column=1, sticky='w', pady=(10, 0))

        p = self._page('LSTM', 'Recurrent prior. Small and fast; carries the full raster history in its state.')
        self.arch_pages['lstm'] = p
        self._row(p, 0, 'Hidden size', self._spin(p, self.lstm_hidden, 32, 2048, 32, 10))
        self._row(p, 1, 'Layers', self._spin(p, self.lstm_layers, 1, 8, 1, 10))
        self._row(p, 2, 'Dropout', self._spin(p, self.lstm_dropout, 0.0, 0.8, 0.05, 10))
        self._row(p, 3, 'Positional embedding', self._combo(p, self.lstm_pos, POS_MODES, 10),
                  '2D sin-cos / learned / none')

        p = self._page('PixelCNN', 'Masked-convolution prior. Sampling needs one forward pass per token.')
        self.arch_pages['pixelcnn'] = p
        self._row(p, 0, 'Channels', self._spin(p, self.pc_ch, 8, 512, 8, 10))
        self._row(p, 1, 'Gated layers', self._spin(p, self.pc_layers, 1, 32, 1, 10))
        self._row(p, 2, 'Input mask kernel', self._spin(p, self.pc_kernel, 3, 15, 2, 10), 'odd number, mask type A')
        self._row(p, 3, 'Positional embedding', self._combo(p, self.pc_pos, POS_MODES, 10),
                  'added before and after the masked input conv')

        p = self._page('Transformer', 'Causal pre-norm Transformer over the raster token sequence. Cost grows with '
                                      'tokens^2; on weak hardware keep the latent grid small (8x8 or 16x16) and '
                                      'the model narrow. Sampling re-runs the model on the growing prefix.')
        self.arch_pages['transformer'] = p
        self._row(p, 0, 'Model width (d_model)', self._spin(p, self.tf_dim, 16, 1024, 16, 10),
                  'must be divisible by heads')
        self._row(p, 1, 'Attention heads', self._spin(p, self.tf_heads, 1, 32, 1, 10))
        self._row(p, 2, 'Layers', self._spin(p, self.tf_layers, 1, 24, 1, 10))
        self._row(p, 3, 'FFN multiplier', self._spin(p, self.tf_ff, 1, 8, 1, 10), 'feed-forward width = d_model x this')
        self._row(p, 4, 'Dropout', self._spin(p, self.tf_dropout, 0.0, 0.8, 0.05, 10))
        self._row(p, 5, 'Positional embedding', self._combo(p, self.tf_pos, POS_MODES, 10),
                  '2D sin-cos / learned / none')

        for v in (self.lstm_hidden, self.lstm_layers, self.lstm_dropout, self.lstm_pos, self.pc_ch,
                  self.pc_layers, self.pc_kernel, self.pc_pos, self.tf_dim, self.tf_heads, self.tf_layers,
                  self.tf_ff, self.tf_dropout, self.tf_pos, self.pr_type):
            v.trace_add('write', lambda *_: self._update_arch_label())

    # ---------------- derived labels / status ----------------
    def _update_k_label(self):
        try:
            d, s, l = int(self.set_D.get()), int(self.set_img_size.get()), int(self.set_latent.get())
            n = n_downs(s, l)
            self.k_label.config(
                text='K = %d codes | latent %dx%d = %d tokens/image | downsample x%d (%d stage%s)'
                     % (2 ** d, l, l, l * l, s // l, n, '' if n == 1 else 's'), foreground=MUTED)
        except ValueError as e:
            self.k_label.config(text=str(e), foreground=ERR)
        except Exception:
            pass

    def current_spec(self):
        t = self.pr_type.get()
        if t == 'lstm':
            return dict(type='lstm', hidden=int(self.lstm_hidden.get()), layers=int(self.lstm_layers.get()),
                        dropout=float(self.lstm_dropout.get()), pos=self.lstm_pos.get())
        if t == 'pixelcnn':
            return dict(type='pixelcnn', ch=int(self.pc_ch.get()), layers=int(self.pc_layers.get()),
                        kernel=int(self.pc_kernel.get()), pos=self.pc_pos.get())
        return dict(type='transformer', dim=int(self.tf_dim.get()), heads=int(self.tf_heads.get()),
                    layers=int(self.tf_layers.get()), ff=int(self.tf_ff.get()),
                    dropout=float(self.tf_dropout.get()), pos=self.tf_pos.get())

    def _apply_spec_to_vars(self, s):
        self.pr_type.set(s['type'])
        if s['type'] == 'lstm':
            self.lstm_hidden.set(s['hidden']); self.lstm_layers.set(s['layers'])
            self.lstm_dropout.set(s['dropout']); self.lstm_pos.set(s['pos'])
        elif s['type'] == 'pixelcnn':
            self.pc_ch.set(s['ch']); self.pc_layers.set(s['layers'])
            self.pc_kernel.set(s['kernel']); self.pc_pos.set(s['pos'])
        else:
            self.tf_dim.set(s['dim']); self.tf_heads.set(s['heads']); self.tf_layers.set(s['layers'])
            self.tf_ff.set(s['ff']); self.tf_dropout.set(s['dropout']); self.tf_pos.set(s['pos'])

    def _update_arch_label(self):
        try:
            txt = 'Settings: ' + describe_spec(self.current_spec())
            if self.pcnn_cfg is not None and self.pcnn_cfg['spec'] != self.current_spec():
                txt += '   [differs from the initialized prior: click Initialize]'
            self.pr_arch_label.config(text=txt)
        except Exception:
            pass

    def _refresh_status(self):
        n = len(self.paths)
        self.st_data.config(text='Data: %s' % ('%d images' % n if n else 'none'), foreground=OK if n else WARN)
        if self.vqvae_cfg:
            c = self.vqvae_cfg
            self.st_vq.config(text='VQ-VAE: %d>%d  D=%d' % (c['img_size'], c['latent_size'], c['D']), foreground=OK)
        else:
            self.st_vq.config(text='VQ-VAE: not ready', foreground=WARN)
        if self.pcnn_cfg:
            self.st_prior.config(text='Prior: %s' % self.pcnn_cfg['spec']['type'], foreground=OK)
        else:
            self.st_prior.config(text='Prior: not ready', foreground=WARN)
        self.st_dev.config(text='Device: %s' % self.device_obj().type)

    def on_edit_arch(self):
        self.nb.select(self.tab_settings)
        self.set_nb.select(self.arch_pages[self.pr_type.get()])

    # ---------------- utilities ----------------
    def log(self, t):
        tag = 'err' if ('[ERROR]' in t or 'failed' in t) else 'warn' if 'WARNING' in t else \
              'ok' if t.endswith('done.') else None
        self.log_text.insert('end', t + '\n', (tag,) if tag else ())
        self.log_text.see('end')

    def _on_tk_error(self, exc, val, tb):
        traceback.print_exception(exc, val, tb)
        self.log('[ERROR] %s: %s' % (exc.__name__, val))
        messagebox.showerror('Error', '%s: %s' % (exc.__name__, val))

    def show_image(self, label, img, maxw=1150, maxh=430):
        if img is None:
            return
        photo = ImageTk.PhotoImage(fit_img(img, maxw, maxh))
        self._photos[str(label)] = photo
        label.configure(image=photo)

    def device_obj(self):
        d = self.set_device.get()
        return torch.device('cpu') if (d == 'cuda' and not torch.cuda.is_available()) else torch.device(d)

    def get_dataset(self, size):
        if size not in self._ds_cache:
            self._ds_cache[size] = ImageListDataset(self.paths, size)
        return self._ds_cache[size]

    def aug_settings(self):
        return dict(hflip=bool(self.set_aug_h.get()), vflip=bool(self.set_aug_v.get()),
                    rot=float(self.set_aug_rot.get()), zoom=float(self.set_aug_zoom.get()),
                    jitter=float(self.set_aug_jit.get()))

    def sampling_args(self):
        return dict(temp=max(0.05, float(self.inf_temp.get())),
                    top_k=max(0, int(self.inf_topk.get())),
                    top_p=min(1.0, max(0.0, float(self.inf_topp.get()))),
                    greedy=bool(self.inf_greedy.get()), dev=self.device_obj(),
                    used_mask=self.used_codes_mask if self.set_use_code_mask.get() else None)

    def model_settings(self):
        return dict(device=self.device_obj().type, D=int(self.set_D.get()),
                    hidden=int(self.set_hidden.get()), beta=float(self.set_beta.get()),
                    entropy=float(self.set_entropy.get()), seed=int(self.set_seed.get()),
                    img_size=int(self.set_img_size.get()), latent_size=int(self.set_latent.get()))

    def prior_train_settings(self):
        return dict(device=self.device_obj().type, seed=int(self.set_seed.get()),
                    epochs=max(1, int(self.pr_epochs.get())), batch=max(1, int(self.pr_batch.get())),
                    lr=float(self.pr_lr.get()), label_smoothing=float(self.pr_ls.get()),
                    val_frac=min(0.5, max(0.0, float(self.pr_val.get()))),
                    aug_copies=max(1, int(self.pr_copies.get())), restore_best=bool(self.pr_restore.get()),
                    use_code_mask=bool(self.set_use_code_mask.get()),
                    preview_every=max(0, int(self.pr_prev_every.get())), preview_unit=self.pr_prev_unit.get(),
                    preview_n=max(1, int(self.pr_prev_n.get())))

    def _set_busy(self, busy):
        self.busy = busy
        st = 'disabled' if busy else 'normal'
        for b in (self.btn_vq_init, self.btn_pc_init, self.btn_sample, self.btn_recon, self.btn_infill,
                  self.btn_fill_pick, self.btn_vq_prev, self.btn_vq_load, self.btn_pc_load):
            b.configure(state=st)
        self.btn_vq_start.configure(state='disabled' if (busy or self.vqvae is None) else 'normal')
        self.btn_pc_start.configure(state='disabled' if (busy or self.pixelcnn is None) else 'normal')
        self.btn_vq_stop.configure(state='normal' if busy else 'disabled')
        self.btn_pc_stop.configure(state='normal' if busy else 'disabled')

    def _eta_text(self, frac):
        now = time.time()
        if self._t0 is None:
            self._t0, self._f0 = now, frac
            return ''
        el = now - self._t0
        done = frac - self._f0
        eta = el * (1 - frac) / done if done > 1e-4 else 0
        return ' | elapsed %s | ETA %s' % (fmt_time(el), fmt_time(eta) if done > 1e-4 else '...')

    # ---------------- dataset ----------------
    def on_choose_files(self):
        paths = filedialog.askopenfilenames(
            title='Select images', filetypes=[('Images', '*.png *.jpg *.jpeg *.bmp *.webp'), ('All', '*.*')])
        if not paths:
            return
        paths = [p for p in paths if p.lower().endswith(IMG_EXTS)]
        if not paths:
            messagebox.showwarning('No images', 'None selected.'); return
        self.paths = paths
        self._ds_cache = {}
        self.used_codes_mask = None
        self.ds_count.config(text='%d images' % len(paths))
        self.log('[Dataset] %d images' % len(paths))
        self._refresh_status()
        self.on_shuffle_ds()

    def _ds_sample(self, k=16):
        ds = self.get_dataset(int(self.set_img_size.get()))
        return ds, random.sample(range(len(ds)), min(k, len(ds)))

    def on_shuffle_ds(self):
        if not self.paths:
            return
        try:
            ds, idxs = self._ds_sample(16)
            self.show_image(self.ds_preview, tensor_to_pil(torch.stack([ds[i] for i in idxs]), ncols=8))
        except Exception as e:
            messagebox.showerror('Preview failed', str(e))

    def on_preview_aug(self):
        if not self.paths:
            messagebox.showinfo('No data', 'Select images first.'); return
        try:
            ds, idxs = self._ds_sample(8)
            aug = build_aug(self.aug_settings(), ds.size)
            ds_a = ds.with_aug(aug)
            imgs = [ds[i] for i in idxs] + [ds_a[i] for i in idxs]
            self.show_image(self.ds_preview, tensor_to_pil(torch.stack(imgs), ncols=len(idxs)))
            if aug is None:
                self.log('[Dataset] no augmentation is enabled (Settings > Augmentation).')
        except Exception as e:
            messagebox.showerror('Preview failed', str(e))

    def on_preview_aug_tab(self):
        self.nb.select(self.tab_dataset)
        self.on_preview_aug()

    # ---------------- VQ-VAE ----------------
    def on_init_vqvae(self):
        if self.busy:
            return
        cfg = self.model_settings()
        try:
            model = VQVAE(D=cfg['D'], hidden=cfg['hidden'], beta=cfg['beta'], entropy_weight=cfg['entropy'],
                          img_size=cfg['img_size'], latent_size=cfg['latent_size'])
            model.eval()
        except Exception as e:
            traceback.print_exc(); messagebox.showerror('Init failed', str(e)); return
        self.vqvae = model; self.vqvae_cfg = cfg; self.used_codes_mask = None
        self._vq_ready_label(model, cfg)
        self.btn_vq_start.configure(state='normal')
        self.log('[VQ-VAE] initialized img=%d latent=%d D=%d' % (cfg['img_size'], cfg['latent_size'], cfg['D']))
        self._refresh_status()

    def _vq_ready_label(self, model, cfg):
        n = sum(p.numel() for p in model.parameters())
        self.vq_model_label.config(
            text='VQ-VAE: %dpx -> %dx%d tokens, D=%d (K=%d), %.2fM params'
                 % (cfg['img_size'], cfg['latent_size'], cfg['latent_size'], cfg['D'], 2 ** cfg['D'], n / 1e6),
            foreground=OK)

    def on_start_vqvae(self):
        if self.busy or self.vqvae is None:
            return
        if not self.paths:
            messagebox.showwarning('No data', 'Select images first.'); return
        cur, old = self.model_settings(), self.vqvae_cfg
        if any(cur[k] != old.get(k) for k in ('D', 'hidden', 'img_size', 'latent_size')):
            if not messagebox.askyesno('Re-init?', 'Architecture settings changed. Re-initialize?'):
                return
            self.on_init_vqvae()
            if self.vqvae is None:
                return
        cfg = dict(self.vqvae_cfg)
        cfg.update(epochs=max(1, int(self.vq_epochs.get())), batch=max(1, int(self.vq_batch.get())),
                   lr=float(self.vq_lr.get()), device=self.device_obj().type, seed=int(self.set_seed.get()),
                   preview_every=max(0, int(self.vq_prev_every.get())), preview_unit=self.vq_prev_unit.get(),
                   preview_n=max(1, int(self.vq_prev_n.get())), aug=self.aug_settings())
        ds_plain = self.get_dataset(cfg['img_size'])
        ds_train = ds_plain.with_aug(build_aug(cfg['aug'], cfg['img_size']))
        self.vqvae_cfg = cfg
        self.stop_event.clear(); self._set_busy(True); self.vq_prog['value'] = 0
        self._t0 = None
        self.log('[VQ-VAE] starting | dev=%s D=%d aug=%s' % (cfg['device'], cfg['D'], cfg['aug']))
        threading.Thread(target=train_vqvae, args=(self, cfg, self.vqvae, ds_train, ds_plain),
                         daemon=True).start()

    def on_vq_preview(self):
        if self.busy or self.vqvae is None or not self.paths:
            return
        try:
            ds = self.get_dataset(self.vqvae_cfg['img_size'])
            img, _ = random_preview(self.vqvae.cpu().eval(), ds, max(1, int(self.vq_prev_n.get())),
                                    torch.device('cpu'))
            self.show_image(self.vq_preview, img)
        except Exception as e:
            traceback.print_exc(); messagebox.showerror('Preview failed', str(e))

    # ---------------- Prior ----------------
    def on_init_pcnn(self):
        if self.busy:
            return
        if self.vqvae is None:
            messagebox.showwarning('No VQ-VAE', 'Train or load a VQ-VAE first.'); return
        K = 2 ** self.vqvae_cfg['D']
        H = W = self.vqvae_cfg['latent_size']
        try:
            spec = self.current_spec()
            model = build_prior(spec, K, H, W)
            model.eval()
        except Exception as e:
            traceback.print_exc(); messagebox.showerror('Init failed', str(e)); return
        self.pixelcnn = model; self.pcnn_cfg = dict(spec=spec, K=K, H=H, W=W)
        self._prior_ready_label(model)
        self.btn_pc_start.configure(state='normal')
        self.log('[Prior] initialized: %s (K=%d, %dx%d tokens)' % (describe_spec(spec), K, H, W))
        self._refresh_status(); self._update_arch_label()

    def _prior_ready_label(self, model):
        c = self.pcnn_cfg
        n = sum(p.numel() for p in model.parameters())
        self.pc_model_label.config(text='%s | K=%d, %dx%d tokens | %.2fM params'
                                        % (describe_spec(c['spec']), c['K'], c['H'], c['W'], n / 1e6),
                                   foreground=OK)

    def on_start_pcnn(self):
        if self.busy or self.vqvae is None or self.pixelcnn is None:
            return
        if not self.paths:
            messagebox.showwarning('No data', 'Select images first.'); return
        vq = self.vqvae_cfg
        try:
            spec = self.current_spec()
        except Exception as e:
            messagebox.showerror('Bad settings', str(e)); return
        old = self.pcnn_cfg
        if spec != old['spec'] or old['K'] != 2 ** vq['D'] or old['H'] != vq['latent_size']:
            if not messagebox.askyesno('Re-init?', 'Prior settings (or the VQ-VAE grid) changed. Re-initialize?'):
                return
            self.on_init_pcnn()
            if self.pixelcnn is None:
                return
        cfg = self.prior_train_settings()
        cfg['spec'] = self.pcnn_cfg['spec']
        size = vq['img_size']
        ds_plain = self.get_dataset(size)
        aug_tf = build_aug(self.aug_settings(), size)
        ds_aug = ds_plain.with_aug(aug_tf) if (aug_tf is not None and cfg['aug_copies'] > 1) else None
        self.stop_event.clear(); self._set_busy(True); self.pc_prog['value'] = 0
        self._t0 = None
        self.log('[Prior] starting: %s | aug copies=%d' % (describe_spec(cfg['spec']),
                                                          cfg['aug_copies'] if ds_aug else 1))
        threading.Thread(target=train_prior, args=(self, cfg, self.pixelcnn, ds_plain, ds_aug),
                         daemon=True).start()

    def on_stop(self):
        self.stop_event.set(); self.log('Stop requested...')

    # ---------------- Inference / infill ----------------
    def _ready(self):
        if self.vqvae is None or self.pixelcnn is None:
            messagebox.showwarning('Not ready', 'Need a VQ-VAE and a prior.'); return False
        return True

    def on_sample(self):
        if self.busy or not self._ready():
            return
        self._set_busy(True)
        threading.Thread(target=run_sample, args=(self, max(1, int(self.inf_n.get())), self.sampling_args()),
                         daemon=True).start()

    def on_reconstruct(self):
        if self.busy or self.vqvae is None:
            return
        path = filedialog.askopenfilename(title='Image', filetypes=[('Images', '*.png *.jpg *.jpeg'), ('All', '*.*')])
        if not path:
            return
        self._set_busy(True)
        threading.Thread(target=run_reconstruct, args=(self, path, self.device_obj()), daemon=True).start()

    def on_pick_infill(self):
        path = filedialog.askopenfilename(title='Image to infill',
                                          filetypes=[('Images', '*.png *.jpg *.jpeg *.bmp *.webp'), ('All', '*.*')])
        if not path:
            return
        self.infill_path = path
        self.fill_label.config(text=os.path.basename(path), foreground=OK)
        if not self.busy and self.vqvae is not None and self.pixelcnn is not None:
            self.on_infill()

    def on_infill(self):
        if self.busy or not self._ready():
            return
        if not self.infill_path:
            self.on_pick_infill(); return
        a = self.sampling_args()
        a.update(path=self.infill_path, cut=int(self.fill_cut.get()), n=max(1, int(self.fill_n.get())),
                 paste=bool(self.fill_paste.get()))
        self._set_busy(True)
        threading.Thread(target=run_infill, args=(self, a), daemon=True).start()

    # ---------------- Checkpoints ----------------
    def on_save_vqvae(self):
        if self.vqvae is None:
            return
        p = filedialog.asksaveasfilename(defaultextension='.pt', initialfile='vqvae_lfq.pt',
                                         filetypes=[('PyTorch', '*.pt')])
        if not p:
            return
        torch.save({'state_dict': self.vqvae.state_dict(), 'cfg': self.vqvae_cfg}, p)
        self.log('[Save] VQ-VAE -> %s' % p)

    def on_load_vqvae(self):
        p = filedialog.askopenfilename(filetypes=[('PyTorch', '*.pt'), ('All', '*.*')])
        if not p:
            return
        try:
            ck = safe_torch_load(p)
            cfg = ck['cfg']
            model = VQVAE(D=cfg['D'], hidden=cfg['hidden'], beta=cfg.get('beta', 0.1),
                          entropy_weight=cfg.get('entropy', 0.1),
                          img_size=cfg.get('img_size', 64), latent_size=cfg.get('latent_size', 8))
            model.load_state_dict(ck['state_dict']); model.eval()
            cfg = dict(cfg, img_size=model.img_size, latent_size=model.latent_size)
            self.vqvae = model; self.vqvae_cfg = cfg; self.used_codes_mask = None
            self.set_D.set(cfg['D']); self.set_hidden.set(cfg['hidden'])
            self.set_img_size.set(cfg['img_size']); self.set_latent.set(cfg['latent_size'])
            self.set_beta.set(cfg.get('beta', 0.1)); self.set_entropy.set(cfg.get('entropy', 0.1))
            self._vq_ready_label(model, cfg)
            self.btn_vq_start.configure(state='normal')
            self.log('[Load] VQ-VAE <- %s' % p)
            self._refresh_status()
        except Exception as e:
            traceback.print_exc(); messagebox.showerror('Load failed', str(e))

    def on_save_pcnn(self):
        if self.pixelcnn is None:
            return
        p = filedialog.asksaveasfilename(defaultextension='.pt', initialfile='prior.pt',
                                         filetypes=[('PyTorch', '*.pt')])
        if not p:
            return
        torch.save({'state_dict': self.pixelcnn.state_dict(), 'cfg': self.pcnn_cfg}, p)
        self.log('[Save] Prior -> %s' % p)

    def on_load_pcnn(self):
        p = filedialog.askopenfilename(filetypes=[('PyTorch', '*.pt'), ('All', '*.*')])
        if not p:
            return
        try:
            ck = safe_torch_load(p)
            cfg = ck['cfg']
            if 'type' in ck and 'type' not in cfg:
                cfg = dict(cfg, type=ck['type'])
            spec = cfg.get('spec') or spec_from_legacy(cfg)              # v4 checkpoints
            lat = self.vqvae_cfg['latent_size'] if self.vqvae_cfg else 8
            K = cfg.get('K', 2 ** self.vqvae_cfg['D'] if self.vqvae_cfg else 256)
            H, W = cfg.get('H', lat), cfg.get('W', lat)
            model = build_prior(spec, K, H, W)
            model.load_state_dict(ck['state_dict']); model.eval()
            self.pixelcnn = model
            self.pcnn_cfg = dict(spec=spec, K=K, H=H, W=W)
            self._apply_spec_to_vars(spec)
            self._prior_ready_label(model)
            self.btn_pc_start.configure(state='normal')
            self.log('[Load] Prior <- %s (%s)' % (p, spec['type']))
            self._refresh_status(); self._update_arch_label()
        except Exception as e:
            traceback.print_exc(); messagebox.showerror('Load failed', str(e))

    # ---------------- Message pump ----------------
    def _poll(self):
        try:
            while True:
                try:
                    self._handle(self.q.get_nowait())
                except queue.Empty:
                    break
                except Exception as e:                       # never let one bad message kill the pump
                    traceback.print_exc()
                    self.log('[ERROR] UI update failed: %s' % e)
        finally:
            self.root.after(80, self._poll)

    def _handle(self, msg):
        kind = msg[0]
        if kind == 'log':
            self.log(msg[1])
        elif kind == 'error':
            self.log('[ERROR] ' + msg[1]); self._set_busy(False)
            messagebox.showerror('Error', msg[1])
        elif kind == 'busy':
            self._set_busy(bool(msg[1]))
        elif kind in ('vqvae_step', 'pcnn_step'):
            d = msg[1]
            frac = ((d['epoch'] - 1) + d['step'] / d['steps']) / d['epochs']
            prog, status = (self.vq_prog, self.vq_status) if kind == 'vqvae_step' else (self.pc_prog, self.pc_status)
            prog['value'] = 100.0 * frac
            status.config(text='Epoch %d/%d | step %d/%d | loss %.4f%s' % (
                d['epoch'], d['epochs'], d['step'], d['steps'], d['loss'], self._eta_text(frac)))
        elif kind == 'vqvae_epoch':
            d = msg[1]
            self.log('[VQ-VAE] epoch %d/%d loss=%.4f | codes used %d/%d' %
                     (d['epoch'], d['epochs'], d['loss'], d['used'], d['K']))
        elif kind == 'pcnn_epoch':
            d = msg[1]
            v = '' if d['val'] is None else ' | val NLL=%.4f' % d['val']
            self.log('[Prior] epoch %d/%d train loss=%.4f%s' % (d['epoch'], d['epochs'], d['loss'], v))
        elif kind == 'vqvae_preview':
            self.show_image(self.vq_preview, msg[1])
        elif kind == 'pcnn_preview':
            self.show_image(self.pc_preview, msg[1])
        elif kind == 'vqvae_done':
            self.vqvae = msg[1]['model']; self.vqvae_cfg = msg[1]['cfg']
            self.used_codes_mask = None
            self._set_busy(False); self.vq_prog['value'] = 100
            self.vq_status.config(text='Done.'); self.log('[VQ-VAE] done.')
            self._refresh_status()
        elif kind == 'pcnn_done':
            self.pixelcnn = msg[1]['model']; self.pcnn_cfg = msg[1]['cfg']
            self._set_busy(False); self.pc_prog['value'] = 100
            self.pc_status.config(text='Done.'); self.log('[Prior] done.')
            self._refresh_status(); self._update_arch_label()
        elif kind == 'infer_img':
            self.show_image(self.inf_preview, msg[1])
        elif kind == 'infer_codes':
            self.show_image(self.inf_codes, msg[1], 1150, 100)
        elif kind == 'infill_img':
            self.show_image(self.fill_preview, msg[1], 1150, 560)


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == '__main__':
    main()
