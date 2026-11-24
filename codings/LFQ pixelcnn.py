# -*- coding: utf-8 -*-
"""
VQ-VAE (LFQ) + Prior (PixelCNN or LSTM), Tkinter GUI.

v3: fixed codes_to_latent bit-shift bug (root cause of "prior outputs garbage"),
    fixed Stop button, added roundtrip debug + greedy sampling.
"""

import os, random, threading, queue, traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

import torchvision.transforms as T
import torchvision.utils as vutils
from PIL import Image, ImageTk


IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.ppm', '.tif', '.tiff')


class ImageListDataset(Dataset):
    def __init__(self, paths, size=64):
        self.paths = list(paths)
        self.tf = T.Compose([T.Resize(size), T.CenterCrop(size), T.ToTensor()])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            return self.tf(Image.open(self.paths[i]).convert('RGB'))
        except Exception:
            return torch.zeros(3, 64, 64)


# ----------------------------------------------------------------------------
# LFQ Quantizer  (v3: fixed codes_to_latent)
# ----------------------------------------------------------------------------
class LookupFreeQuantizer(nn.Module):
    def __init__(self, D, beta=0.1, entropy_weight=0.1):
        super().__init__()
        self.D = D
        self.K = 2 ** D
        self.beta = beta
        self.entropy_weight = entropy_weight
        self.register_buffer('powers', (2 ** torch.arange(D)).long())         # [1,2,4,...]
        self.register_buffer('bit_positions', torch.arange(D).long())         # [0,1,2,...]
        self._self_test()

    def _self_test(self):
        """Verify codes_to_latent is the exact inverse of forward()."""
        dev = self.powers.device
        z = torch.randn(2, self.D, 4, 4, device=dev)
        z_q, _, codes = self.forward(z)
        z_rec = self.codes_to_latent(codes)
        err = (z_q - z_rec).abs().max().item()
        if err > 1e-4:
            raise RuntimeError(
                "codes_to_latent is not the inverse of forward(): max err %.5f. "
                "Prior-generated images would be garbage." % err
            )

    def forward(self, z):
        B, D, H, W = z.shape
        inv_sqrt_d = D ** -0.5
        bits = (z >= 0).long()
        z_q = (2.0 * bits.float() - 1.0) * inv_sqrt_d
        z_q_st = z + (z_q - z).detach()
        commit_loss = F.mse_loss(z, z_q.detach())
        codes = (bits * self.powers.view(1, D, 1, 1)).sum(1)                   # (B,H,W)
        p = F.one_hot(codes.view(-1), num_classes=self.K).float().mean(0)
        entropy_loss = (p * torch.log(p + 1e-8)).sum()
        return z_q_st, self.beta * commit_loss + self.entropy_weight * entropy_loss, codes

    def codes_to_latent(self, idx):
        # >>> THE FIX <<<
        # Shift by bit POSITION [0,1,2,...] not by POWERS [1,2,4,...].
        bits = ((idx.unsqueeze(1) >> self.bit_positions.view(1, -1, 1, 1)) & 1).long()
        return (2.0 * bits.float() - 1.0) * (self.D ** -0.5)


class VQVAE(nn.Module):
    def __init__(self, D=8, hidden=128, in_ch=3, beta=0.1, entropy_weight=0.1):
        super().__init__()
        self.D = D
        self.K = 2 ** D
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(hidden, hidden, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(hidden, hidden, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(hidden, D, 3, 1, 1),
        )
        self.vq = LookupFreeQuantizer(D, beta, entropy_weight)
        self.decoder = nn.Sequential(
            nn.Conv2d(D, hidden, 3, 1, 1), nn.ReLU(True),
            nn.ConvTranspose2d(hidden, hidden, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(hidden, hidden, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(hidden, in_ch, 4, 2, 1),
        )

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
        rec = F.mse_loss(x_rec, x)
        return x_rec, rec + vq_loss, rec, vq_loss, codes


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
        h = self.conv(x)
        a, b = h.chunk(2, dim=1)
        return x + torch.tanh(a) * torch.sigmoid(b)


class PixelCNN(nn.Module):
    def __init__(self, K=256, ch=64, n_layers=6):
        super().__init__()
        self.K = K
        self.emb = nn.Embedding(K, ch)
        self.inp = MaskedConv2d('A', ch, ch, 7, padding=3)
        self.blocks = nn.ModuleList([GatedBlock(ch) for _ in range(n_layers)])
        self.out = nn.Conv2d(ch, K, 1)

    def forward(self, idx):
        x = self.emb(idx).permute(0, 3, 1, 2).contiguous()
        x = self.inp(x)
        for b in self.blocks:
            x = b(x)
        return self.out(x)                       # (B,K,H,W)


class LSTMPrior(nn.Module):
    def __init__(self, K, H, W, hidden=256, n_layers=2, dropout=0.1):
        super().__init__()
        self.K = K
        self.H, self.W = H, W
        self.N = H * W
        self.emb = nn.Embedding(K + 1, hidden)   # K == start token
        self.pos_emb = nn.Embedding(self.N, hidden)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(hidden, hidden, n_layers, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0.0)
        self.out = nn.Linear(hidden, K)

    def forward(self, idx):
        B = idx.shape[0]
        seq = idx.reshape(B, -1) if idx.dim() == 3 else idx
        start = torch.full((B, 1), self.K, dtype=torch.long, device=idx.device)
        inp = torch.cat([start, seq[:, :-1]], dim=1)
        pos = torch.arange(self.N, device=idx.device).unsqueeze(0).expand(B, -1)
        x = self.dropout(self.emb(inp) + self.pos_emb(pos))
        h, _ = self.lstm(x)
        logits = self.out(h)
        if idx.dim() == 3:
            logits = logits.view(B, self.H, self.W, self.K).permute(0, 3, 1, 2)
        return logits


# ----------------------------------------------------------------------------
# Sampling
# ----------------------------------------------------------------------------
@torch.no_grad()
def sample_pixelcnn(model, n, H, W, K, device, temperature=1.0, used_mask=None, greedy=False):
    model.eval()
    idx = torch.zeros(n, H, W, dtype=torch.long, device=device)
    for i in range(H):
        for j in range(W):
            logits = model(idx)[:, :, i, j] / max(temperature, 1e-6)
            if used_mask is not None:
                logits = logits.masked_fill(~used_mask.unsqueeze(0), float('-inf'))
            if greedy:
                idx[:, i, j] = logits.argmax(1)
            else:
                idx[:, i, j] = torch.multinomial(F.softmax(logits, 1), 1).squeeze(1)
    return idx


@torch.no_grad()
def sample_lstm(model, n, H, W, K, device, temperature=1.0, used_mask=None, greedy=False):
    model.eval()
    B, N = n, H * W
    seq = torch.zeros(B, N, dtype=torch.long, device=device)
    hidden = None
    for t in range(N):
        prev = torch.full((B,), model.K, dtype=torch.long, device=device) if t == 0 else seq[:, t-1]
        x = model.emb(prev.unsqueeze(1)) + model.pos_emb(torch.tensor([t], device=device))
        out, hidden = model.lstm(x, hidden)
        logits = model.out(out[:, 0]) / max(temperature, 1e-6)
        if used_mask is not None:
            logits = logits.masked_fill(~used_mask.unsqueeze(0), float('-inf'))
        if greedy:
            seq[:, t] = logits.argmax(1)
        else:
            seq[:, t] = torch.multinomial(F.softmax(logits, 1), 1).squeeze(1)
    return seq.view(B, H, W)


@torch.no_grad()
def sample_prior(model, n, H, W, K, device, temperature=1.0, used_mask=None, greedy=False):
    if isinstance(model, LSTMPrior):
        return sample_lstm(model, n, H, W, K, device, temperature, used_mask, greedy)
    return sample_pixelcnn(model, n, H, W, K, device, temperature, used_mask, greedy)


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


def codes_to_pil(codes, scale=255):
    """Render integer codes as an image (single-channel heatmap)."""
    c = codes.detach().cpu().float()
    if c.dim() == 3:
        c = c.unsqueeze(1)            # (B,1,H,W)
    K_max = max(1.0, float(c.max().item()))
    c = (c / K_max).clamp(0, 1)
    c = c.repeat(1, 3, 1, 1)          # 3 channels for display
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


# ----------------------------------------------------------------------------
# Training threads
# ----------------------------------------------------------------------------
def train_vqvae(app, cfg, model):
    try:
        dev = torch.device(cfg['device'])
        torch.manual_seed(cfg['seed']); random.seed(cfg['seed'])
        model = model.to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
        loader = DataLoader(app.dataset, batch_size=cfg['batch'], shuffle=True,
                            num_workers=0, drop_last=False)
        steps = len(loader)
        if steps == 0:
            app.q.put(('error', 'Dataset is empty.')); app.q.put(('busy', False)); return
        app.q.put(('log', '[VQ-VAE] params={:,} steps/epoch={} D={} K={}'.format(
            sum(p.numel() for p in model.parameters()), steps, cfg['D'], 2**cfg['D'])))

        try:
            _first = next(iter(loader))
            if isinstance(_first, (list, tuple)): _first = _first[0]
            preview = _first[:8].to(dev)
        except StopIteration:
            preview = None

        stopped = False
        for epoch in range(1, cfg['epochs'] + 1):
            model.train()
            tot, n = 0.0, 0
            for step, x in enumerate(loader, 1):
                if app.stop_event.is_set():
                    stopped = True
                    break
                x = x.to(dev)
                _, loss, rec, vql, _ = model(x)
                opt.zero_grad(set_to_none=True)
                loss.backward(); opt.step()
                tot += loss.item(); n += 1
                if step % 10 == 0 or step == steps:
                    app.q.put(('vqvae_step', dict(epoch=epoch, epochs=cfg['epochs'],
                        step=step, steps=steps, loss=loss.item(),
                        recon=rec.item(), vq=vql.item())))
            if stopped:
                app.q.put(('log', '[VQ-VAE] stopped by user'))
                break

            model.eval()
            if preview is not None:
                with torch.no_grad():
                    rec, _, _, _, codes = model(preview)
                    rec_rt = model.decode_indices(codes)    # <- roundtrip
                    used = len(torch.unique(codes))
                    rt_err = (rec_rt - rec).abs().max().item()
                if rt_err > 1e-3:
                    app.q.put(('log', '[VQ-VAE] WARNING: decode_indices mismatch '
                                      'err=%.4f (codes_to_latent bug?)' % rt_err))
                # Top: originals | Middle: forward recon | Bottom: roundtrip recon
                img = tensor_to_pil(torch.cat([preview, rec, rec_rt], 0), ncols=8)
                app.q.put(('vqvae_epoch', dict(epoch=epoch, epochs=cfg['epochs'],
                    loss=tot / max(n, 1), img=img, used=used, K=2**cfg['D'])))

        model.eval().cpu()
        app.q.put(('vqvae_done', dict(model=model, cfg=cfg)))
    except Exception as e:
        traceback.print_exc()
        app.q.put(('error', 'VQ-VAE training failed: %s' % e)); app.q.put(('busy', False))


def train_pixelcnn(app, cfg, model):
    try:
        vqvae, vqcfg = app.vqvae, app.vqvae_cfg
        if vqvae is None:
            app.q.put(('error', 'No VQ-VAE.')); app.q.put(('busy', False)); return

        dev = torch.device(cfg['device'])
        torch.manual_seed(cfg['seed']); random.seed(cfg['seed'])
        vqvae = vqvae.to(dev).eval()

        app.q.put(('log', '[Prior] encoding dataset ...'))
        codes = []
        loader = DataLoader(app.dataset, batch_size=64, shuffle=False,
                            num_workers=0, drop_last=False)
        with torch.no_grad():
            for x in loader:
                if app.stop_event.is_set(): break
                _, _, idx = vqvae.encode(x.to(dev))
                codes.append(idx.cpu())
        if not codes:
            app.q.put(('error', 'No codes.')); app.q.put(('busy', False)); return
        codes = torch.cat(codes)
        H, W = codes.shape[1], codes.shape[2]
        K = 2 ** vqcfg['D']

        freq = torch.bincount(codes.flatten(), minlength=K).float()
        used_mask = (freq > 0)
        app.used_codes_mask = used_mask
        coverage = used_mask.sum().item() / K
        app.q.put(('log', '[Prior] codes={} unique={}/{} ({:.1f}%) | params={:,}'.format(
            tuple(codes.shape), int(used_mask.sum().item()), K, coverage * 100,
            sum(p.numel() for p in model.parameters()))))

        model = model.to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
        dl = DataLoader(TensorDataset(codes), batch_size=cfg['batch'],
                        shuffle=True, num_workers=0, drop_last=False)
        steps = len(dl)
        ls = cfg.get('label_smoothing', 0.0)
        use_mask = cfg.get('use_code_mask', True)

        stopped = False
        for epoch in range(1, cfg['epochs'] + 1):
            model.train()
            tot, n = 0.0, 0
            for step, (idx,) in enumerate(dl, 1):
                if app.stop_event.is_set():
                    stopped = True
                    break
                idx = idx.to(dev)
                logits = model(idx)
                loss = F.cross_entropy(
                    logits.permute(0, 2, 3, 1).reshape(-1, K),
                    idx.reshape(-1), label_smoothing=ls)
                opt.zero_grad(set_to_none=True)
                loss.backward(); opt.step()
                tot += loss.item(); n += 1
                if step % 10 == 0 or step == steps:
                    app.q.put(('pcnn_step', dict(epoch=epoch, epochs=cfg['epochs'],
                        step=step, steps=steps, loss=loss.item())))
            if stopped:
                app.q.put(('log', '[Prior] stopped by user'))
                break

            with torch.no_grad():
                sidx = sample_prior(model, 8, H, W, K, dev, temperature=1.0,
                                    used_mask=used_mask if use_mask else None,
                                    greedy=False)
                imgs = vqvae.decode_indices(sidx)
            app.q.put(('pcnn_epoch', dict(epoch=epoch, epochs=cfg['epochs'],
                                          loss=tot / max(n, 1),
                                          img=tensor_to_pil(imgs, ncols=8))))

        model.eval().cpu()
        app.q.put(('pcnn_done', dict(model=model, cfg=dict(cfg, K=K, H=H, W=W))))
    except Exception as e:
        traceback.print_exc()
        app.q.put(('error', 'Prior training failed: %s' % e)); app.q.put(('busy', False))


def run_sample(app, n, temperature, greedy):
    try:
        vqvae, prior = app.vqvae, app.pixelcnn
        if vqvae is None or prior is None:
            app.q.put(('log', '[Inference] need both VQ-VAE and prior.')); return
        dev = app.device_obj()
        vqvae = vqvae.to(dev).eval()
        prior = prior.to(dev).eval()
        H = W = app.latent_size
        K = 2 ** app.vqvae_cfg['D']
        used_mask = getattr(app, 'used_codes_mask', None) if app.settings_use_code_mask() else None
        with torch.no_grad():
            idx = sample_prior(prior, n, H, W, K, dev, temperature, used_mask, greedy)
            imgs = vqvae.decode_indices(idx)
        app.q.put(('infer_img', tensor_to_pil(imgs, ncols=min(8, n))))
        app.q.put(('infer_codes', codes_to_pil(idx[:min(8, n)])))
        app.q.put(('log', '[Inference] sampled %d (T=%.1f greedy=%s mask=%s)' %
                   (n, temperature, greedy, used_mask is not None)))
        vqvae.cpu(); prior.cpu()
    except Exception as e:
        traceback.print_exc(); app.q.put(('error', str(e)))
    finally:
        app.q.put(('busy', False))


def run_reconstruct(app, path):
    try:
        vqvae = app.vqvae
        if vqvae is None:
            app.q.put(('log', '[Inference] need a VQ-VAE.')); return
        dev = app.device_obj()
        vqvae = vqvae.to(dev).eval()
        tf = T.Compose([T.Resize(app.img_size), T.CenterCrop(app.img_size), T.ToTensor()])
        x = tf(Image.open(path).convert('RGB')).unsqueeze(0).to(dev)
        with torch.no_grad():
            rec, _, _, _, codes = vqvae(x)
            rec_rt = vqvae.decode_indices(codes)              # roundtrip
            rt_err = (rec_rt - rec).abs().max().item()
            # 3-panel: original | forward recon | decode_indices recon
            comp = torch.cat([x[0].cpu(), rec[0].cpu(), rec_rt[0].cpu()], dim=2)
        app.q.put(('infer_img', tensor_to_pil(comp.unsqueeze(0), ncols=1)))
        app.q.put(('infer_codes', codes_to_pil(codes)))
        app.q.put(('log', '[Inference] original | forward-recon | decode_indices-recon '
                          '(err=%.6f)' % rt_err))
        vqvae.cpu()
    except Exception as e:
        traceback.print_exc(); app.q.put(('error', str(e)))
    finally:
        app.q.put(('busy', False))


# ----------------------------------------------------------------------------
# GUI
# ----------------------------------------------------------------------------
class App:
    def __init__(self, root):
        self.root = root
        root.title('LFQ-VAE + Prior (PixelCNN/LSTM)')
        root.geometry('1180x950')
        self.q = queue.Queue()
        self.stop_event = threading.Event()
        self.busy = False
        self.img_size = 64
        self.latent_size = 8
        self.dataset = None
        self.vqvae = None
        self.vqvae_cfg = None
        self.pixelcnn = None
        self.pcnn_cfg = None
        self.used_codes_mask = None
        self._photos = {}
        self._build_ui()
        root.after(80, self._poll)

    def _build_ui(self):
        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill='both', expand=True, padx=6, pady=6)
        tab_defs = [
            ('Dataset',   'tab_dataset'),
            ('VQ-VAE',    'tab_vqvae'),
            ('Prior',     'tab_prior'),
            ('Inference', 'tab_inference'),
            ('Settings',  'tab_settings'),
        ]
        for name, attr in tab_defs:
            frame = ttk.Frame(self.nb, padding=8)
            self.nb.add(frame, text=name)
            setattr(self, attr, frame)
        self._build_ds(self.tab_dataset)
        self._build_vq(self.tab_vqvae)
        self._build_pc(self.tab_prior)
        self._build_inf(self.tab_inference)
        self._build_set(self.tab_settings)

        logf = ttk.LabelFrame(self.root, text='Log', padding=4)
        logf.pack(fill='x', padx=6, pady=(0, 6))
        self.log_text = tk.Text(logf, height=8, wrap='word')
        sb = ttk.Scrollbar(logf, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=sb.set)
        self.log_text.pack(side='left', fill='both', expand=True)
        sb.pack(side='right', fill='y')

    def _build_ds(self, f):
        row = ttk.Frame(f); row.pack(fill='x')
        ttk.Button(row, text='Select images...', command=self.on_choose_files).pack(side='left')
        self.ds_count = tk.StringVar(value='0 images')
        ttk.Label(row, textvariable=self.ds_count, foreground='#0a5').pack(side='right')
        self.ds_preview = ttk.Label(f, relief='sunken', anchor='center')
        self.ds_preview.pack(fill='both', expand=True, pady=8)

    def _build_vq(self, f):
        row = ttk.Frame(f); row.pack(fill='x')
        self.vq_epochs = tk.IntVar(value=20)
        self.vq_batch = tk.IntVar(value=32)
        self.vq_lr = tk.DoubleVar(value=2e-4)
        for lbl, var, w, inc in (('Epochs', self.vq_epochs, 6, 1),
                                  ('Batch', self.vq_batch, 6, 16),
                                  ('LR', self.vq_lr, 9, 0.0001)):
            ttk.Label(row, text=lbl).pack(side='left')
            ttk.Spinbox(row, from_=0, to=1e9, increment=inc, width=w,
                        textvariable=var).pack(side='left', padx=(2, 12))
        row2 = ttk.Frame(f); row2.pack(fill='x', pady=(6, 0))
        self.btn_vq_init = ttk.Button(row2, text='Initialize model', command=self.on_init_vqvae)
        self.btn_vq_init.pack(side='left', padx=4)
        self.btn_vq_start = ttk.Button(row2, text='Start training', command=self.on_start_vqvae, state='disabled')
        self.btn_vq_start.pack(side='left', padx=4)
        self.btn_vq_stop = ttk.Button(row2, text='Stop', command=self.on_stop, state='disabled')
        self.btn_vq_stop.pack(side='left')
        self.vq_model_label = ttk.Label(f, text='Model: not initialized', foreground='#a60')
        self.vq_model_label.pack(anchor='w', pady=(6, 0))
        self.vq_prog = ttk.Progressbar(f, maximum=100); self.vq_prog.pack(fill='x', pady=8)
        self.vq_status = ttk.Label(f, text='Idle.'); self.vq_status.pack(anchor='w')
        self.vq_preview = ttk.Label(f, relief='sunken', anchor='center')
        self.vq_preview.pack(fill='both', expand=True, pady=8)
        ttk.Label(f, text='Preview: top=originals, middle=forward recon, bottom=decode_indices recon '
                          '(should match middle).', foreground='#666').pack(anchor='w')

    def _build_pc(self, f):
        row = ttk.Frame(f); row.pack(fill='x')
        self.btn_pc_init = ttk.Button(row, text='Initialize Prior', command=self.on_init_pcnn)
        self.btn_pc_init.pack(side='left', padx=4)
        self.btn_pc_start = ttk.Button(row, text='Start training', command=self.on_start_pcnn, state='disabled')
        self.btn_pc_start.pack(side='left', padx=4)
        self.btn_pc_stop = ttk.Button(row, text='Stop', command=self.on_stop, state='disabled')
        self.btn_pc_stop.pack(side='left')
        self.pc_model_label = ttk.Label(f, text='Prior: not initialized', foreground='#a60')
        self.pc_model_label.pack(anchor='w', pady=(6, 0))
        ttk.Label(f, text='(Prior hyper-parameters are on the Settings tab.)',
                  foreground='#666').pack(anchor='w', pady=(6, 0))
        self.pc_prog = ttk.Progressbar(f, maximum=100); self.pc_prog.pack(fill='x', pady=8)
        self.pc_status = ttk.Label(f, text='Idle.'); self.pc_status.pack(anchor='w')
        self.pc_preview = ttk.Label(f, relief='sunken', anchor='center')
        self.pc_preview.pack(fill='both', expand=True, pady=8)

    def _build_inf(self, f):
        row = ttk.Frame(f); row.pack(fill='x')
        ttk.Label(row, text='Samples').pack(side='left')
        self.inf_n = tk.IntVar(value=16)
        ttk.Spinbox(row, from_=1, to=64, increment=1, width=6,
                    textvariable=self.inf_n).pack(side='left', padx=(2, 12))
        ttk.Label(row, text='Temp').pack(side='left', padx=(0, 2))
        self.inf_temp = tk.DoubleVar(value=1.0)
        ttk.Spinbox(row, from_=0.1, to=3.0, increment=0.1, width=6,
                    textvariable=self.inf_temp).pack(side='left', padx=(0, 12))
        self.inf_greedy = tk.BooleanVar(value=False)
        ttk.Checkbutton(row, text='Greedy', variable=self.inf_greedy).pack(side='left', padx=(0, 12))
        self.btn_sample = ttk.Button(row, text='Sample from Prior', command=self.on_sample)
        self.btn_sample.pack(side='left', padx=4)
        self.btn_recon = ttk.Button(row, text='Test roundtrip on image...', command=self.on_reconstruct)
        self.btn_recon.pack(side='left', padx=4)
        self.inf_preview = ttk.Label(f, relief='sunken', anchor='center')
        self.inf_preview.pack(fill='both', expand=True, pady=(8, 4))
        ttk.Label(f, text='Codes:', foreground='#666').pack(anchor='w')
        self.inf_codes = ttk.Label(f, relief='sunken', anchor='center')
        self.inf_codes.pack(fill='x', pady=(0, 4))

    def _build_set(self, f):
        self.set_device = tk.StringVar(value='cpu')
        self.set_D = tk.IntVar(value=8)
        self.set_hidden = tk.IntVar(value=128)
        self.set_beta = tk.DoubleVar(value=0.1)
        self.set_entropy = tk.DoubleVar(value=0.1)
        self.set_seed = tk.IntVar(value=1234)
        self.set_prior_type = tk.StringVar(value='lstm')
        self.set_pc_epochs = tk.IntVar(value=30)
        self.set_pc_batch = tk.IntVar(value=32)
        self.set_pc_lr = tk.DoubleVar(value=1e-3)
        self.set_pc_layers = tk.IntVar(value=6)
        self.set_pc_ch = tk.IntVar(value=64)
        self.set_lstm_hidden = tk.IntVar(value=256)
        self.set_lstm_layers = tk.IntVar(value=2)
        self.set_label_smoothing = tk.DoubleVar(value=0.05)
        self.set_use_code_mask = tk.BooleanVar(value=True)

        m = ttk.LabelFrame(f, text='VQ-VAE (LFQ)', padding=8)
        m.pack(anchor='nw', fill='x', pady=(0, 8))
        g = ttk.Frame(m); g.pack(anchor='nw')
        rows = [
            ('Device', ttk.Combobox(g, textvariable=self.set_device, values=['cpu', 'cuda'], width=10, state='readonly')),
            ('Latent dim D', ttk.Spinbox(g, from_=1, to=16, increment=1, width=10, textvariable=self.set_D)),
            ('Hidden', ttk.Spinbox(g, from_=16, to=1024, increment=16, width=10, textvariable=self.set_hidden)),
            ('Commit β', ttk.Spinbox(g, from_=0.0, to=5.0, increment=0.05, width=10, textvariable=self.set_beta)),
            ('Entropy weight', ttk.Spinbox(g, from_=0.0, to=5.0, increment=0.01, width=10, textvariable=self.set_entropy)),
            ('Seed', ttk.Spinbox(g, from_=0, to=1e9, increment=1, width=10, textvariable=self.set_seed)),
        ]
        for i, (lbl, w) in enumerate(rows):
            ttk.Label(g, text=lbl).grid(row=i // 2, column=(i % 2) * 2, sticky='w', pady=2, padx=(0, 6))
            w.grid(row=i // 2, column=(i % 2) * 2 + 1, sticky='w', pady=2, padx=(0, 14))
        self.k_label = ttk.Label(g, text='', foreground='#666')
        self.k_label.grid(row=3, column=0, columnspan=4, sticky='w', pady=(6, 0))
        self.set_D.trace_add('write', lambda *_: self._update_k_label())
        self._update_k_label()

        p = ttk.LabelFrame(f, text='Prior', padding=8)
        p.pack(anchor='nw', fill='x', pady=(0, 8))
        g2 = ttk.Frame(p); g2.pack(anchor='nw')
        ttk.Label(g2, text='Type').grid(row=0, column=0, sticky='w', pady=2, padx=(0, 6))
        ttk.Combobox(g2, textvariable=self.set_prior_type, values=['lstm', 'pixelcnn'],
                     width=10, state='readonly').grid(row=0, column=1, sticky='w', padx=(0, 14))
        ttk.Label(g2, text='Epochs').grid(row=0, column=2, sticky='w', pady=2, padx=(0, 6))
        ttk.Spinbox(g2, from_=1, to=1e6, increment=1, width=8,
                    textvariable=self.set_pc_epochs).grid(row=0, column=3, sticky='w', padx=(0, 14))
        ttk.Label(g2, text='Batch').grid(row=0, column=4, sticky='w', pady=2, padx=(0, 6))
        ttk.Spinbox(g2, from_=1, to=1e6, increment=16, width=8,
                    textvariable=self.set_pc_batch).grid(row=0, column=5, sticky='w', padx=(0, 14))
        ttk.Label(g2, text='LR').grid(row=1, column=0, sticky='w', pady=2, padx=(0, 6))
        ttk.Spinbox(g2, from_=0.0, to=1.0, increment=0.0005, width=10,
                    textvariable=self.set_pc_lr).grid(row=1, column=1, sticky='w', padx=(0, 14))
        ttk.Label(g2, text='Label smooth').grid(row=1, column=2, sticky='w', pady=2, padx=(0, 6))
        ttk.Spinbox(g2, from_=0.0, to=0.5, increment=0.01, width=8,
                    textvariable=self.set_label_smoothing).grid(row=1, column=3, sticky='w', padx=(0, 14))

        pf = ttk.Frame(p); pf.pack(anchor='nw', pady=(6, 0))
        ttk.Label(pf, text='PixelCNN:').pack(side='left')
        ttk.Label(pf, text='Layers').pack(side='left', padx=(8, 2))
        ttk.Spinbox(pf, from_=1, to=32, increment=1, width=5,
                    textvariable=self.set_pc_layers).pack(side='left', padx=(0, 8))
        ttk.Label(pf, text='Channels').pack(side='left', padx=(0, 2))
        ttk.Spinbox(pf, from_=8, to=512, increment=8, width=5,
                    textvariable=self.set_pc_ch).pack(side='left')

        lf = ttk.Frame(p); lf.pack(anchor='nw', pady=(4, 0))
        ttk.Label(lf, text='LSTM:').pack(side='left')
        ttk.Label(lf, text='Hidden').pack(side='left', padx=(8, 2))
        ttk.Spinbox(lf, from_=32, to=1024, increment=32, width=5,
                    textvariable=self.set_lstm_hidden).pack(side='left', padx=(0, 8))
        ttk.Label(lf, text='Layers').pack(side='left', padx=(0, 2))
        ttk.Spinbox(lf, from_=1, to=6, increment=1, width=5,
                    textvariable=self.set_lstm_layers).pack(side='left', padx=(0, 8))

        ttk.Checkbutton(p, text='Mask unused codes during sampling',
                        variable=self.set_use_code_mask).pack(anchor='w', pady=(6, 0))

        btns = ttk.LabelFrame(f, text='Checkpoints', padding=6)
        btns.pack(anchor='nw', fill='x', pady=8)
        ttk.Button(btns, text='Save VQ-VAE...', command=self.on_save_vqvae).pack(side='left', padx=4)
        ttk.Button(btns, text='Load VQ-VAE...', command=self.on_load_vqvae).pack(side='left', padx=4)
        ttk.Button(btns, text='Save Prior...',  command=self.on_save_pcnn).pack(side='left', padx=4)
        ttk.Button(btns, text='Load Prior...',  command=self.on_load_pcnn).pack(side='left', padx=4)

    def _update_k_label(self):
        try:
            d = int(self.set_D.get())
        except Exception:
            d = 8
        self.k_label.config(text='Number of codes: K = 2^D = %d' % (2 ** d))

    def log(self, t):
        self.log_text.insert('end', t + '\n'); self.log_text.see('end')

    def show_image(self, label, img, maxw=1000, maxh=420):
        if img is None: return
        img = fit_img(img, maxw, maxh)
        photo = ImageTk.PhotoImage(img)
        self._photos[str(label)] = photo
        label.configure(image=photo)

    def device_obj(self):
        d = self.set_device.get()
        if d == 'cuda' and not torch.cuda.is_available():
            return torch.device('cpu')
        return torch.device(d)

    def settings_use_code_mask(self):
        return bool(self.set_use_code_mask.get())

    def model_settings(self):
        return dict(device=self.device_obj().type,
                    D=int(self.set_D.get()), hidden=int(self.set_hidden.get()),
                    beta=float(self.set_beta.get()), entropy=float(self.set_entropy.get()),
                    seed=int(self.set_seed.get()))

    def pcnn_settings(self):
        return dict(device=self.device_obj().type, seed=int(self.set_seed.get()),
                    type=str(self.set_prior_type.get()),
                    epochs=max(1, int(self.set_pc_epochs.get())),
                    batch=max(1, int(self.set_pc_batch.get())),
                    lr=float(self.set_pc_lr.get()),
                    layers=max(1, int(self.set_pc_layers.get())),
                    ch=max(8, int(self.set_pc_ch.get())),
                    lstm_hidden=max(32, int(self.set_lstm_hidden.get())),
                    lstm_layers=max(1, int(self.set_lstm_layers.get())),
                    label_smoothing=float(self.set_label_smoothing.get()),
                    use_code_mask=bool(self.set_use_code_mask.get()))

    def _set_busy(self, busy):
        self.busy = busy
        st = 'disabled' if busy else 'normal'
        self.btn_vq_init.configure(state=st)
        self.btn_pc_init.configure(state=st)
        self.btn_sample.configure(state=st)
        self.btn_recon.configure(state=st)
        self.btn_vq_start.configure(state='disabled' if (busy or self.vqvae is None) else 'normal')
        self.btn_pc_start.configure(state='disabled' if (busy or self.pixelcnn is None) else 'normal')
        self.btn_vq_stop.configure(state='normal' if busy else 'disabled')
        self.btn_pc_stop.configure(state='normal' if busy else 'disabled')

    def on_choose_files(self):
        paths = filedialog.askopenfilenames(
            title='Select images',
            filetypes=[('Images', '*.png *.jpg *.jpeg *.bmp *.webp'), ('All', '*.*')])
        if not paths: return
        paths = [p for p in paths if p.lower().endswith(IMG_EXTS)]
        if not paths:
            messagebox.showwarning('No images', 'None selected.'); return
        self.dataset = ImageListDataset(paths, size=self.img_size)
        self.used_codes_mask = None
        self.ds_count.set('%d images' % len(paths))
        self.log('[Dataset] %d images' % len(paths))
        k = min(16, len(paths))
        idxs = random.sample(range(len(paths)), k)
        imgs = torch.stack([self.dataset[i] for i in idxs])
        self.show_image(self.ds_preview, tensor_to_pil(imgs, ncols=8), 1000, 300)

    # ---------------- VQ-VAE ----------------
    def on_init_vqvae(self):
        if self.busy: return
        cfg = self.model_settings()
        try:
            model = VQVAE(D=cfg['D'], hidden=cfg['hidden'],
                          beta=cfg['beta'], entropy_weight=cfg['entropy'])
            model.eval()
        except Exception as e:
            traceback.print_exc(); messagebox.showerror('Init failed', str(e)); return
        self.vqvae = model; self.vqvae_cfg = cfg; self.used_codes_mask = None
        n = sum(p.numel() for p in model.parameters())
        self.vq_model_label.config(
            text='VQ-VAE: D=%d K=%d %.2fM params (roundtrip self-test passed)'
                 % (cfg['D'], 2**cfg['D'], n / 1e6),
            foreground='#060')
        self.btn_vq_start.configure(state='normal')
        self.log('[VQ-VAE] initialized D=%d K=%d params=%.2fM'
                 % (cfg['D'], 2**cfg['D'], n / 1e6))

    def on_start_vqvae(self):
        if self.busy or self.dataset is None or self.vqvae is None:
            return
        cur = self.model_settings()
        if cur['D'] != self.vqvae_cfg['D'] or cur['hidden'] != self.vqvae_cfg['hidden']:
            if not messagebox.askyesno('Re-init?', 'Architecture changed. Re-initialize?'):
                return
            self.on_init_vqvae()
            if self.vqvae is None: return
        cfg = dict(self.vqvae_cfg)
        cfg.update(epochs=max(1, int(self.vq_epochs.get())),
                   batch=max(1, int(self.vq_batch.get())),
                   lr=float(self.vq_lr.get()))
        self.stop_event.clear(); self._set_busy(True); self.vq_prog['value'] = 0
        self.log('[VQ-VAE] starting | dev=%s D=%d' % (cfg['device'], cfg['D']))
        threading.Thread(target=train_vqvae, args=(self, cfg, self.vqvae), daemon=True).start()

    # ---------------- Prior ----------------
    def on_init_pcnn(self):
        if self.busy: return
        if self.vqvae is None:
            messagebox.showwarning('No VQ-VAE', 'Train or load a VQ-VAE first.'); return
        cfg = self.pcnn_settings()
        K = 2 ** self.vqvae_cfg['D']
        H = W = self.latent_size
        try:
            if cfg['type'] == 'lstm':
                model = LSTMPrior(K, H, W, hidden=cfg['lstm_hidden'],
                                  n_layers=cfg['lstm_layers'], dropout=0.1)
            else:
                model = PixelCNN(K, cfg['ch'], cfg['layers'])
            model.eval()
        except Exception as e:
            traceback.print_exc(); messagebox.showerror('Init failed', str(e)); return
        self.pixelcnn = model; self.pcnn_cfg = dict(cfg, K=K, H=H, W=W)
        n = sum(p.numel() for p in model.parameters())
        self.pc_model_label.config(
            text='%s: K=%d %.2fM params' % (cfg['type'].upper(), K, n / 1e6),
            foreground='#060')
        self.btn_pc_start.configure(state='normal')
        self.log('[Prior] initialized %s (K=%d params=%.2fM)' % (cfg['type'], K, n / 1e6))

    def on_start_pcnn(self):
        if self.busy or self.dataset is None or self.vqvae is None or self.pixelcnn is None:
            return
        cur = self.pcnn_settings()
        old = self.pcnn_cfg
        arch_changed = (
            cur['type'] != old.get('type', 'pixelcnn') or
            (cur['type'] == 'lstm' and (
                cur['lstm_hidden'] != old.get('lstm_hidden', 256) or
                cur['lstm_layers'] != old.get('lstm_layers', 2))) or
            (cur['type'] == 'pixelcnn' and (
                cur['ch'] != old.get('ch', 64) or
                cur['layers'] != old.get('layers', 6)))
        )
        if arch_changed:
            if not messagebox.askyesno('Re-init?', 'Prior settings changed. Re-initialize?'):
                return
            self.on_init_pcnn()
            if self.pixelcnn is None: return
        cfg = dict(self.pcnn_cfg)
        cfg.update(epochs=max(1, int(self.set_pc_epochs.get())),
                   batch=max(1, int(self.set_pc_batch.get())),
                   lr=float(self.set_pc_lr.get()),
                   label_smoothing=float(self.set_label_smoothing.get()),
                   use_code_mask=bool(self.set_use_code_mask.get()))
        self.stop_event.clear(); self._set_busy(True); self.pc_prog['value'] = 0
        self.log('[Prior] starting %s' % cfg['type'])
        threading.Thread(target=train_pixelcnn, args=(self, cfg, self.pixelcnn),
                         daemon=True).start()

    def on_stop(self):
        self.stop_event.set(); self.log('Stop requested...')

    def on_sample(self):
        if self.busy: return
        if self.vqvae is None or self.pixelcnn is None:
            messagebox.showwarning('Not ready', 'Need VQ-VAE and prior.'); return
        self._set_busy(True)
        n = max(1, int(self.inf_n.get()))
        t = max(0.1, float(self.inf_temp.get()))
        g = bool(self.inf_greedy.get())
        threading.Thread(target=run_sample, args=(self, n, t, g), daemon=True).start()

    def on_reconstruct(self):
        if self.busy or self.vqvae is None: return
        path = filedialog.askopenfilename(
            title='Image', filetypes=[('Images', '*.png *.jpg *.jpeg'), ('All', '*.*')])
        if not path: return
        self._set_busy(True)
        threading.Thread(target=run_reconstruct, args=(self, path), daemon=True).start()

    # ---------------- Checkpoints ----------------
    def on_save_vqvae(self):
        if self.vqvae is None: return
        p = filedialog.asksaveasfilename(defaultextension='.pt', initialfile='vqvae_lfq.pt',
                                         filetypes=[('PyTorch', '*.pt')])
        if not p: return
        torch.save({'state_dict': self.vqvae.state_dict(), 'cfg': self.vqvae_cfg}, p)
        self.log('[Save] VQ-VAE -> %s' % p)

    def on_load_vqvae(self):
        p = filedialog.askopenfilename(filetypes=[('PyTorch', '*.pt'), ('All', '*.*')])
        if not p: return
        try:
            ck = safe_torch_load(p)
            cfg = ck['cfg']
            model = VQVAE(D=cfg['D'], hidden=cfg['hidden'],
                          beta=cfg.get('beta', 0.1),
                          entropy_weight=cfg.get('entropy', 0.1))
            model.load_state_dict(ck['state_dict']); model.eval()
            self.vqvae = model; self.vqvae_cfg = cfg; self.used_codes_mask = None
            self.set_D.set(cfg['D']); self.set_hidden.set(cfg['hidden'])
            self.set_beta.set(cfg.get('beta', 0.1)); self.set_entropy.set(cfg.get('entropy', 0.1))
            n = sum(p.numel() for p in model.parameters())
            self.vq_model_label.config(
                text='VQ-VAE: D=%d K=%d %.2fM' % (cfg['D'], 2**cfg['D'], n / 1e6),
                foreground='#060')
            self.btn_vq_start.configure(state='normal')
            self.log('[Load] VQ-VAE <- %s' % p)
        except Exception as e:
            traceback.print_exc(); messagebox.showerror('Load failed', str(e))

    def on_save_pcnn(self):
        if self.pixelcnn is None: return
        p = filedialog.asksaveasfilename(defaultextension='.pt', initialfile='prior.pt',
                                         filetypes=[('PyTorch', '*.pt')])
        if not p: return
        torch.save({'state_dict': self.pixelcnn.state_dict(), 'cfg': self.pcnn_cfg,
                    'type': self.pcnn_cfg.get('type', 'pixelcnn')}, p)
        self.log('[Save] Prior -> %s' % p)

    def on_load_pcnn(self):
        p = filedialog.askopenfilename(filetypes=[('PyTorch', '*.pt'), ('All', '*.*')])
        if not p: return
        try:
            ck = safe_torch_load(p)
            cfg = ck['cfg']
            ptype = ck.get('type', cfg.get('type', 'pixelcnn'))
            K = cfg.get('K', 2 ** self.vqvae_cfg['D'] if self.vqvae_cfg else 256)
            H = cfg.get('H', self.latent_size); W = cfg.get('W', self.latent_size)
            if ptype == 'lstm':
                model = LSTMPrior(K, H, W, hidden=cfg.get('lstm_hidden', 256),
                                  n_layers=cfg.get('lstm_layers', 2))
            else:
                model = PixelCNN(K, cfg.get('ch', 64), cfg.get('layers', 6))
            model.load_state_dict(ck['state_dict']); model.eval()
            self.pixelcnn = model
            self.pcnn_cfg = dict(cfg, type=ptype, K=K, H=H, W=W)
            n = sum(p.numel() for p in model.parameters())
            self.pc_model_label.config(
                text='%s: K=%d %.2fM' % (ptype.upper(), K, n / 1e6),
                foreground='#060')
            self.btn_pc_start.configure(state='normal')
            self.log('[Load] Prior <- %s (%s)' % (p, ptype))
        except Exception as e:
            traceback.print_exc(); messagebox.showerror('Load failed', str(e))

    # ---------------- Message pump ----------------
    def _poll(self):
        try:
            while True:
                self._handle(self.q.get_nowait())
        except queue.Empty:
            pass
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
        elif kind == 'vqvae_step':
            d = msg[1]; frac = ((d['epoch'] - 1) + d['step'] / d['steps']) / d['epochs']
            self.vq_prog['value'] = 100.0 * frac
            self.vq_status.config(
                text='Epoch %d/%d step %d/%d | loss %.4f' %
                     (d['epoch'], d['epochs'], d['step'], d['steps'], d['loss']))
        elif kind == 'vqvae_epoch':
            d = msg[1]
            if d['img'] is not None:
                self.show_image(self.vq_preview, d['img'])
            self.log('[VQ-VAE] epoch %d/%d loss=%.4f used=%d/%d' %
                     (d['epoch'], d['epochs'], d['loss'], d['used'], d['K']))
        elif kind == 'vqvae_done':
            self.vqvae = msg[1]['model']; self.vqvae_cfg = msg[1]['cfg']
            self.used_codes_mask = None
            self._set_busy(False); self.vq_prog['value'] = 100
            self.vq_status.config(text='Done.')
            self.log('[VQ-VAE] done.')
        elif kind == 'pcnn_step':
            d = msg[1]; frac = ((d['epoch'] - 1) + d['step'] / d['steps']) / d['epochs']
            self.pc_prog['value'] = 100.0 * frac
            self.pc_status.config(
                text='Epoch %d/%d step %d/%d | NLL %.4f' %
                     (d['epoch'], d['epochs'], d['step'], d['steps'], d['loss']))
        elif kind == 'pcnn_epoch':
            d = msg[1]
            self.show_image(self.pc_preview, d['img'])
            self.log('[Prior] epoch %d/%d NLL=%.4f' %
                     (d['epoch'], d['epochs'], d['loss']))
        elif kind == 'pcnn_done':
            self.pixelcnn = msg[1]['model']; self.pcnn_cfg = msg[1]['cfg']
            self._set_busy(False); self.pc_prog['value'] = 100
            self.pc_status.config(text='Done.')
            self.log('[Prior] done.')
        elif kind == 'infer_img':
            self.show_image(self.inf_preview, msg[1], 1050, 420)
        elif kind == 'infer_codes':
            self.show_image(self.inf_codes, msg[1], 1050, 100)


def main():
    root = tk.Tk()
    try:
        s = ttk.Style()
        if 'vista' in s.theme_names():
            s.theme_use('vista')
    except Exception:
        pass
    App(root)
    root.mainloop()


if __name__ == '__main__':
    main()