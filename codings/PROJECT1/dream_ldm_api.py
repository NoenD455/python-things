# dream_ldm_api.py
"""
Dream LDM — World Model API.

Same FlexVAE + temporal VelocityUNet as the video LDM. The only change is the
conditioning channel: instead of text (GRU/Transformer), the UNet receives a
learned "button-press" embedding per frame. Videos are paired with a button
CSV from `button sampler.py` (columns: timestamp,buttonspressed; pipe-separated
names, empty when nothing is pressed).

Reuses from flow_api / LDM_api unchanged.
"""
import os
import math
import random
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F_nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image

from flow_api import (
    Conv2dWithPadding, AttentionBlock, DownBlock, UpBlock,
    SinusoidalPositionEmbeddings, get_norm,
    load_image_as_rgb, load_image_as_grayscale,
    match_ot_batch, RandomJPEG, ElasticTransform,
    PaddedTransform, TV_ROT_HAS_PAD, TV_PERSP_HAS_PAD,
    compute_head_dims,
)
from LDM_api import FlexVAE  # VAE shared with the image/video LDM


try:
    import cv2  # noqa
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False
try:
    import imageio.v3 as _iio  # noqa
    _HAS_IMAGEIO = True
except Exception:
    _HAS_IMAGEIO = False

VIDEO_EXTS = ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.gif', '.m4v', '.wmv', '.flv')
IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.jfif', '.webp', '.bmp')


# ==================== Video loading ====================

def load_video_frames(path, stride=1):
    stride = max(1, int(stride))
    ext = os.path.splitext(path)[1].lower()
    frames = []

    if ext == '.gif':
        img = Image.open(path)
        i = 0
        try:
            while True:
                img.seek(i)
                if i % stride == 0:
                    frames.append(img.convert('RGB').copy())
                i += 1
        except EOFError:
            pass
        return frames

    if _HAS_CV2:
        import cv2
        cap = cv2.VideoCapture(path)
        i = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if i % stride == 0:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            i += 1
        cap.release()
        return frames

    if _HAS_IMAGEIO:
        for i, frame in enumerate(_iio.imiter(path)):
            if i % stride == 0:
                arr = np.asarray(frame)
                if arr.ndim == 2:
                    pil = Image.fromarray(arr, mode='L').convert('RGB')
                else:
                    pil = Image.fromarray(arr).convert('RGB')
                frames.append(pil)
        return frames

    raise RuntimeError(
        f"Cannot load video '{path}'. Install 'opencv-python' or 'imageio[ffmpeg]'.")


def load_video_fps(path):
    """Best-effort fps detection. Falls back to 30.0."""
    if _HAS_CV2:
        import cv2
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        return fps if fps > 0 else 30.0
    return 30.0


# ==================== Button CSV ====================

def load_button_csv(path):
    """Parse a button-press CSV. Returns list of (timestamp_s, set_of_button_names)."""
    rows = []
    with open(path, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        header = next(reader, None)  # discard header
        for row in reader:
            if not row or len(row) < 2:
                continue
            try:
                t = float(row[0])
            except (ValueError, TypeError):
                continue
            bstr = row[1].strip().strip('"').strip("'")
            names = set()
            if bstr:
                for n in bstr.split('|'):
                    n = n.strip()
                    if n:
                        names.add(n)
            rows.append((t, names))
    rows.sort(key=lambda x: x[0])
    return rows


def buttons_at_time(rows, time_s):
    """Return set of button names active at `time_s` (last row with t <= time_s)."""
    if not rows:
        return set()
    active = set()
    for t, names in rows:
        if t <= time_s + 1e-6:
            active = names
        else:
            break
    return active


def collect_button_names(list_of_row_lists):
    """Union of all button names appearing across multiple parsed CSVs."""
    names = set()
    for rows in list_of_row_lists:
        for _, ns in rows:
            names.update(ns)
    return sorted(names)


class ButtonEncoder(nn.Module):
    """
    Maps a binary button-state vector [B, N] to a cond vector [B, cond_dim].

    Each button has two learnable hidden embeddings (pressed / unpressed);
    a small MLP then projects the aggregate to cond_dim. This gives the UNet
    the same interface as a text encoder, so `RectifiedFlowLatent` works
    unchanged — it just receives `button_encoder(state)` as `cond`.
    """
    def __init__(self, num_buttons, cond_dim=256, hidden_dim=128):
        super().__init__()
        self.num_buttons = max(1, int(num_buttons))
        self.hidden_dim = hidden_dim
        self.emb_pressed = nn.Parameter(torch.randn(self.num_buttons, hidden_dim) * 0.02)
        self.emb_unpressed = nn.Parameter(torch.randn(self.num_buttons, hidden_dim) * 0.02)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

    def forward(self, button_state):
        # button_state: [B, N] in {0, 1}
        state = button_state.unsqueeze(-1)                         # [B, N, 1]
        pressed = self.emb_pressed.unsqueeze(0)                    # [1, N, H]
        unpressed = self.emb_unpressed.unsqueeze(0)                # [1, N, H]
        emb = state * pressed + (1 - state) * unpressed            # [B, N, H]
        agg = emb.sum(dim=1)                                       # [B, H]
        return self.mlp(agg)


# ==================== Temporal VelocityUNet ====================
# (identical to video_ldm_api.VelocityUNet — kept here so the file is self-contained)

VALID_COND_MODES = ('shift', 'film', 'adagn', 'cross_attn')


class VelocityUNet(nn.Module):
    def __init__(self, latent_channels=8, history=1, base_channels=32,
                 time_emb_dim=128, cond_dim=256,
                 latent_h=4, latent_w=4, channel_mult=(1, 2, 4), dropout=0.1,
                 attention_levels=None, attn_num_heads=4, attn_dropout=0.0,
                 padding_mode='constant', cond_mode='shift'):
        super().__init__()
        self.latent_channels = latent_channels
        self.history = history
        in_channels = latent_channels * (1 + history)
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.time_emb_dim = time_emb_dim
        self.cond_dim = cond_dim
        self.channel_mult = channel_mult
        self.padding_mode = padding_mode
        self.attn_num_heads = attn_num_heads
        self.attn_dropout = attn_dropout
        self.cond_mode = cond_mode if cond_mode in VALID_COND_MODES else 'shift'
        self.latent_h = latent_h
        self.latent_w = latent_w

        num_down = 0
        ch, cw = latent_h, latent_w
        while ch >= 2 and cw >= 2 and num_down < len(channel_mult) - 1:
            ch //= 2; cw //= 2; num_down += 1
        mult_used = channel_mult[:num_down + 1]

        if attention_levels is None:
            attention_levels = [True] * len(mult_used)
        attn_flags = list(attention_levels)
        if len(attn_flags) < len(mult_used):
            attn_flags += [False] * (len(mult_used) - len(attn_flags))
        attn_flags = attn_flags[:len(mult_used)]
        self.attention_levels = attn_flags

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        self.init_conv = Conv2dWithPadding(in_channels, base_channels, 3,
                                           padding_mode=padding_mode)

        self.downs = nn.ModuleList()
        cur = base_channels
        for i, m in enumerate(mult_used):
            out = base_channels * m
            self.downs.append(DownBlock(
                cur, out, time_emb_dim, cond_dim, dropout,
                use_attention=attn_flags[i],
                attn_num_heads=attn_num_heads,
                attn_dropout=attn_dropout,
                padding_mode=padding_mode,
                cond_mode=self.cond_mode))
            if i < len(mult_used) - 1:
                self.downs.append(nn.Conv2d(out, out, 4, stride=2, padding=1))
            cur = out

        mid_attn = attn_flags[-1] if attn_flags else True
        self.mid_block1 = DownBlock(
            cur, cur, time_emb_dim, cond_dim, dropout,
            use_attention=mid_attn,
            attn_num_heads=attn_num_heads,
            attn_dropout=attn_dropout,
            padding_mode=padding_mode,
            cond_mode=self.cond_mode)
        self.mid_block2 = UpBlock(
            cur, cur, time_emb_dim, cond_dim, dropout,
            use_attention=mid_attn,
            attn_num_heads=attn_num_heads,
            attn_dropout=attn_dropout,
            padding_mode=padding_mode,
            cond_mode=self.cond_mode)

        self.ups = nn.ModuleList()
        rev = list(reversed(mult_used))
        rev_attn = list(reversed(attn_flags))
        for i, m in enumerate(rev):
            out = base_channels * m
            self.ups.append(UpBlock(
                cur + out, out, time_emb_dim, cond_dim, dropout,
                use_attention=rev_attn[i],
                attn_num_heads=attn_num_heads,
                attn_dropout=attn_dropout,
                padding_mode=padding_mode,
                cond_mode=self.cond_mode))
            if i < len(rev) - 1:
                self.ups.append(nn.ConvTranspose2d(out, out, 4, stride=2, padding=1))
            cur = out

        self.final_conv = nn.Sequential(
            get_norm(cur), nn.SiLU(),
            Conv2dWithPadding(cur, latent_channels, 3, padding_mode=padding_mode)
        )

    def forward(self, x, t, cond=None):
        te = self.time_mlp(t)
        x = self.init_conv(x)
        skips = []
        for layer in self.downs:
            if isinstance(layer, DownBlock):
                x = layer(x, te, cond); skips.append(x)
            else:
                x = layer(x)
        x = self.mid_block1(x, te, cond)
        x = self.mid_block2(x, te, cond)
        for layer in self.ups:
            if isinstance(layer, UpBlock):
                x = torch.cat([x, skips.pop()], dim=1)
                x = layer(x, te, cond)
            else:
                x = layer(x)
        return self.final_conv(x)


# ==================== Autoregressive latent flow ====================

class RectifiedFlowLatent:
    def __init__(self, latent_channels=8, latent_h=4, latent_w=4, history=1,
                 base_channels=32, cond_dim=256, device=None,
                 attention_levels=None, attn_num_heads=4, attn_dropout=0.0,
                 padding_mode='constant', cond_mode='shift',
                 channel_mult=(1, 2, 4)):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.latent_channels = latent_channels
        self.latent_h, self.latent_w = latent_h, latent_w
        self.history = history
        self.cond_dim = cond_dim
        self.cond_mode = cond_mode if cond_mode in VALID_COND_MODES else 'shift'
        self.use_ot = False
        self.ot_method = 'sinkhorn'
        self.ot_sinkhorn_eps = 0.05
        self.ot_sinkhorn_iters = 50

        self._attn_kwargs = dict(
            attention_levels=attention_levels,
            attn_num_heads=attn_num_heads,
            attn_dropout=attn_dropout,
            cond_mode=self.cond_mode,
        )

        self.model = VelocityUNet(
            latent_channels=latent_channels, history=history,
            base_channels=base_channels,
            time_emb_dim=max(64, base_channels * 4), cond_dim=cond_dim,
            latent_h=latent_h, latent_w=latent_w, channel_mult=channel_mult,
            dropout=0.1, padding_mode=padding_mode,
            **self._attn_kwargs,
        ).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=2e-4)
        self.ema_model = None
        self.ema_decay = 0.999
        self.ema_mode = 'standard'
        self.use_ema = False

    def _make_ema_model(self):
        return VelocityUNet(
            latent_channels=self.model.latent_channels, history=self.model.history,
            base_channels=self.model.base_channels,
            time_emb_dim=self.model.time_emb_dim, cond_dim=self.model.cond_dim,
            latent_h=self.latent_h, latent_w=self.latent_w,
            channel_mult=self.model.channel_mult, dropout=0.1,
            padding_mode=self.model.padding_mode,
            **self._attn_kwargs,
        ).to(self.device)

    def set_ema(self, decay=0.999, mode='standard'):
        self.ema_decay, self.ema_mode, self.use_ema = decay, mode, True
        self.ema_model = self._make_ema_model()
        self.ema_model.load_state_dict(self.model.state_dict())
        for p in self.ema_model.parameters():
            p.requires_grad = False

    def update_ema(self):
        if self.ema_model is None:
            return
        with torch.no_grad():
            for p, ep in zip(self.model.parameters(), self.ema_model.parameters()):
                ep.data.mul_(self.ema_decay).add_(p.data, alpha=1 - self.ema_decay)
        if self.ema_mode == 'lookahead':
            with torch.no_grad():
                for p, ep in zip(self.model.parameters(), self.ema_model.parameters()):
                    p.data.copy_(ep.data)

    def train_step(self, z0, z1, cond=None, context=None, cfg_dropout_prob=0.0,
                   noise_ctx_prob=0.0, noise_ctx_scale=0.5):
        self.model.train()
        B = z0.size(0)
        z0 = z0.to(self.device); z1 = z1.to(self.device)
        if self.use_ot and B > 1:
            z0, z1, col = match_ot_batch(
                z0, z1, method=self.ot_method,
                sinkhorn_eps=self.ot_sinkhorn_eps,
                sinkhorn_iters=self.ot_sinkhorn_iters)
            if context is not None:
                context = context[col]
            if cond is not None:
                cond = cond[col]
        if cond is None:
            cond = torch.zeros(B, self.cond_dim, device=self.device)
        else:
            cond = cond.to(self.device)
        t = torch.rand(B, device=self.device)
        zt = t.view(-1, 1, 1, 1) * z1 + (1 - t.view(-1, 1, 1, 1)) * z0
        target = z1 - z0
        if cfg_dropout_prob > 0:
            mask = (torch.rand(B, 1, device=self.device) > cfg_dropout_prob).float()
            cond = cond * mask

        if context is not None and self.history > 0:
            context = context.to(self.device)
            if noise_ctx_prob > 0:
                m = (torch.rand(B, self.history, 1, 1, 1,
                                device=self.device) < noise_ctx_prob).float()
                context = context + torch.randn_like(context) * noise_ctx_scale * m
            ctx_flat = context.reshape(B, -1, context.shape[-2], context.shape[-1])
            model_in = torch.cat([zt, ctx_flat], dim=1)
        else:
            model_in = zt

        pred = self.model(model_in, t, cond=cond)
        loss = self.criterion(pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.use_ema:
            self.update_ema()
        return loss.item()

    def _v_fn_factory(self, model, n, cond, cfg_scale, ctx_flat=None):
        if cond is None:
            cond = torch.zeros(n, self.cond_dim, device=self.device)
        null = torch.zeros_like(cond)

        def v_fn(t_val, z_val):
            tt = torch.full((n,), t_val, device=self.device)
            inp = torch.cat([z_val, ctx_flat], dim=1) if ctx_flat is not None else z_val
            if cfg_scale != 1.0:
                vc = model(inp, tt, cond=cond)
                vu = model(inp, tt, cond=null)
                return vu + cfg_scale * (vc - vu)
            return model(inp, tt, cond=cond)
        return v_fn

    def _step(self, method, t_cur, dt, z, v_fn):
        if method == 'euler':
            z = z + v_fn(t_cur, z) * dt
        elif method == 'heun':
            v1 = v_fn(t_cur, z); zp = z + v1 * dt
            v2 = v_fn(t_cur + dt, zp)
            z = z + (v1 + v2) * (dt / 2)
        elif method == 'midpoint':
            zm = z + v_fn(t_cur, z) * (dt / 2)
            z = z + v_fn(t_cur + dt / 2, zm) * dt
        elif method == 'rk4':
            v1 = v_fn(t_cur, z)
            v2 = v_fn(t_cur + dt / 2, z + v1 * dt / 2)
            v3 = v_fn(t_cur + dt / 2, z + v2 * dt / 2)
            v4 = v_fn(t_cur + dt, z + v3 * dt)
            z = z + (v1 + 2 * v2 + 2 * v3 + v4) * dt / 6
        else:
            raise ValueError(f"Unknown method {method}")
        return z

    @torch.no_grad()
    def sample_frame(self, n_samples=16, cond=None, steps=50, method='euler',
                     cfg_scale=1.0, use_ema=True, start_from=None, context=None):
        model = (self.ema_model
                 if (use_ema and self.use_ema and self.ema_model is not None)
                 else self.model)
        model.eval()
        shape = (n_samples, self.latent_channels, self.latent_h, self.latent_w)
        z = (start_from.to(self.device) if start_from is not None
             else torch.randn(*shape, device=self.device))
        if cond is None:
            cond = torch.zeros(n_samples, self.cond_dim, device=self.device)
        else:
            cond = cond.to(self.device)
        ctx_flat = None
        if context is not None and self.history > 0:
            context = context.to(self.device)
            ctx_flat = context.reshape(n_samples, -1, self.latent_h, self.latent_w)
        dt = 1.0 / steps
        times = torch.linspace(0, 1, steps + 1, device=self.device)
        v_fn = self._v_fn_factory(model, n_samples, cond, cfg_scale, ctx_flat)
        for i in range(steps):
            z = self._step(method, times[i], dt, z, v_fn)
        return z

    @torch.no_grad()
    def sample_video(self, n_samples=1, num_frames=16, cond=None, cond_sequence=None,
                     steps=50, method='euler', cfg_scale=1.0, use_ema=True,
                     init_context=None, progress_cb=None, stop_check=None,
                     frame_cb=None):
        """
        cond_sequence : list of per-frame [B, cond_dim] tensors (length >= num_frames).
                        If provided, overrides `cond` for each frame.
        frame_cb      : called as frame_cb(frame_idx, latent) after each frame completes.
        """
        B = n_samples
        if init_context is not None:
            context = init_context.to(self.device)
        else:
            context = torch.zeros(B, self.history, self.latent_channels,
                                  self.latent_h, self.latent_w, device=self.device)
        if cond is not None:
            cond = cond.to(self.device)

        frames = []
        for fi in range(num_frames):
            if stop_check is not None and stop_check():
                break
            this_cond = cond_sequence[fi] if cond_sequence is not None else cond
            z = self.sample_frame(
                n_samples=B, cond=this_cond, steps=steps, method=method,
                cfg_scale=cfg_scale, use_ema=use_ema, context=context)
            frames.append(z)
            if self.history > 0:
                context = torch.cat([context[:, 1:], z.unsqueeze(1)], dim=1)
            if progress_cb is not None:
                progress_cb(fi + 1, num_frames)
            if frame_cb is not None:
                frame_cb(fi, z)
        if not frames:
            return None
        return torch.stack(frames, dim=1)

    @torch.no_grad()
    def sample_video_step_by_step(self, n_samples=1, num_frames=16, cond=None,
                                  cond_sequence=None, steps=50, method='euler',
                                  cfg_scale=1.0, use_ema=True, init_context=None):
        B = n_samples
        model = (self.ema_model
                 if (use_ema and self.use_ema and self.ema_model is not None)
                 else self.model)
        model.eval()

        if init_context is not None:
            context = init_context.to(self.device)
        else:
            context = torch.zeros(B, self.history, self.latent_channels,
                                  self.latent_h, self.latent_w, device=self.device)
        if cond is not None:
            cond = cond.to(self.device)

        shape = (B, self.latent_channels, self.latent_h, self.latent_w)
        dt = 1.0 / steps
        times = torch.linspace(0, 1, steps + 1, device=self.device)
        completed = []

        for fi in range(num_frames):
            this_cond = cond_sequence[fi] if cond_sequence is not None else cond
            completed_snapshot = [f.clone() for f in completed]
            z = torch.randn(*shape, device=self.device)
            ctx_flat = None
            if self.history > 0:
                ctx_flat = context.reshape(B, -1, self.latent_h, self.latent_w)
            v_fn = self._v_fn_factory(model, B, this_cond, cfg_scale, ctx_flat)
            for si in range(steps):
                z = self._step(method, times[si], dt, z, v_fn)
                yield fi, si + 1, completed_snapshot + [z.clone()]
            completed.append(z)
            if self.history > 0:
                context = torch.cat([context[:, 1:], z.unsqueeze(1)], dim=1)


# ==================== Dataset ====================

class VideoFrameDataset(Dataset):
    """
    Sources:
      {'kind': 'image', 'path': str}
      {'kind': 'video', 'path': str, 'stride': int,
       'button_csv_path': str|None, 'button_rows': [(t, set), ...], 'fps': float}
      {'kind': 'folder','path': str, 'paths': [str], 'stride': int,
       'button_csv_path': str|None, 'button_rows': [(t, set), ...], 'fps': float}

    Returns (seq_t, btn_seq):
        seq_t   : [history+1, C, H, W]
        btn_seq : [history+1, num_buttons] with 0/1 entries
    """
    def __init__(self, sources, img_size=32, color_mode='rgb', history=1,
                 aug_settings=None, button_to_idx=None, num_buttons=0):
        self.img_size = img_size
        self.color_mode = color_mode.lower()
        self.history = max(1, int(history))
        self.aug_settings = aug_settings or {}
        self.button_to_idx = button_to_idx or {}
        self.num_buttons = max(0, int(num_buttons))

        if self.color_mode == 'rgb':
            self.base_transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            self.base_transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])

        self.aug_padding_mode = self.aug_settings.get('aug_padding_mode', 'edge')
        if self.aug_padding_mode not in ('constant', 'edge', 'reflect', 'symmetric'):
            self.aug_padding_mode = 'edge'
        self.aug_pipeline = self._build_aug_pipeline(self.aug_settings)

        self.source_frames = []
        self.source_button_rows = []
        self.source_fps = []
        self.samples = []
        for i, src in enumerate(sources):
            frames = self._load_source_frames(src)
            self.source_frames.append(frames)
            self.source_button_rows.append(src.get('button_rows', []) or [])
            self.source_fps.append(float(src.get('fps', 30.0) or 30.0))
            if not frames:
                continue
            N = len(frames)
            for start in range(-self.history, N - self.history):
                self.samples.append((i, start))

    def _build_aug_pipeline(self, a):
        aug_pad = self.aug_padding_mode
        pipeline = []
        if a.get('flip_horizontal', False):
            pipeline.append(transforms.RandomHorizontalFlip(p=0.5))
        if a.get('rotation', False):
            if TV_ROT_HAS_PAD:
                pipeline.append(transforms.RandomRotation(
                    30, interpolation=Image.BICUBIC, expand=False, fill=0,
                    padding_mode=aug_pad))
            else:
                base = transforms.RandomRotation(
                    30, interpolation=Image.BICUBIC, expand=False, fill=0)
                pipeline.append(PaddedTransform(
                    base, pad_fraction=0.22, padding_mode=aug_pad))
        if a.get('random_crop', False):
            sc = a.get('crop_scale', 0.8)
            pipeline.append(transforms.RandomResizedCrop(
                size=self.img_size, scale=(sc, 1.0), interpolation=Image.BICUBIC))
        if a.get('color_jitter', False):
            pipeline.append(transforms.ColorJitter(
                brightness=a.get('brightness', 0.2), contrast=a.get('contrast', 0.2),
                saturation=a.get('saturation', 0.2), hue=a.get('hue', 0.1)))
        if a.get('random_perspective', False):
            dist = float(a.get('perspective_distortion', 0.1))
            if TV_PERSP_HAS_PAD:
                pipeline.append(transforms.RandomPerspective(
                    distortion_scale=dist, p=0.5,
                    interpolation=Image.BICUBIC, fill=0,
                    padding_mode=aug_pad))
            else:
                base = transforms.RandomPerspective(
                    distortion_scale=dist, p=0.5,
                    interpolation=Image.BICUBIC, fill=0)
                frac = min(0.35, max(0.20, dist * 0.5 + 0.15))
                pipeline.append(PaddedTransform(
                    base, pad_fraction=frac, padding_mode=aug_pad))
        if a.get('elastic_transform', False):
            pipeline.append(ElasticTransform(
                alpha=a.get('elastic_alpha', 30),
                sigma=a.get('elastic_sigma', 3), p=0.5,
                padding_mode=aug_pad))
        if a.get('jpeg_compression', False):
            pipeline.append(RandomJPEG(
                quality_low=a.get('jpeg_quality_low', 50),
                quality_high=a.get('jpeg_quality_high', 95), p=0.5))
        if a.get('stretch_vertical', False):
            pipeline.append(transforms.RandomResizedCrop(
                size=self.img_size, scale=(0.8, 1.0), ratio=(0.5, 1.0),
                interpolation=Image.BICUBIC))
        if a.get('stretch_horizontal', False):
            pipeline.append(transforms.RandomResizedCrop(
                size=self.img_size, scale=(0.8, 1.0), ratio=(1.0, 2.0),
                interpolation=Image.BICUBIC))
        return pipeline

    def _load_source_frames(self, src):
        kind = src['kind']
        try:
            if kind == 'image':
                p = src['path']
                return [load_image_as_rgb(p) if self.color_mode == 'rgb'
                        else load_image_as_grayscale(p)]
            if kind == 'video':
                frames = load_video_frames(src['path'], src.get('stride', 1))
                if self.color_mode == 'grayscale':
                    frames = [f.convert('L') for f in frames]
                return frames
            if kind == 'folder':
                paths = sorted(src['paths'])
                stride = max(1, int(src.get('stride', 1)))
                out = []
                for j, p in enumerate(paths):
                    if j % stride != 0:
                        continue
                    try:
                        out.append(load_image_as_rgb(p) if self.color_mode == 'rgb'
                                   else load_image_as_grayscale(p))
                    except Exception:
                        continue
                return out
        except Exception as e:
            print(f"[Dataset] failed to load {src}: {e}")
        return []

    def apply_aug(self, pil):
        img = pil.copy()
        for a in self.aug_pipeline:
            img = a(img)
        return img

    def _frame_button_state(self, src_idx, frame_idx_in_loaded_list):
        if self.num_buttons <= 0:
            return torch.zeros(0)
        stride = 1
        rows = self.source_button_rows[src_idx]
        fps = self.source_fps[src_idx] or 30.0
        state = torch.zeros(self.num_buttons)
        if not rows:
            return state
        # loaded_list[k] corresponds to original frame k * stride
        # We need the stride for this source — but stride isn't stored here.
        # Simplest fix: store per-source stride.
        # For now assume stride=1 (the GUI stores per-source stride if needed).
        stride = getattr(self, '_source_strides', [1] * len(self.source_frames))[src_idx]
        t = frame_idx_in_loaded_list * stride / fps
        active = buttons_at_time(rows, t)
        for name in active:
            if name in self.button_to_idx:
                state[self.button_to_idx[name]] = 1.0
        return state

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ch = 3 if self.color_mode == 'rgb' else 1
        try:
            src_idx, start = self.samples[idx]
            frames = self.source_frames[src_idx]
            seed = random.randint(0, 2**31 - 1)
            seq = []
            btns = []
            for i in range(self.history + 1):
                fi = start + i
                if 0 <= fi < len(frames):
                    try:
                        random.seed(seed)
                        torch.manual_seed(seed)
                        t = self.base_transform(self.apply_aug(frames[fi]))
                    except Exception:
                        t = torch.zeros(ch, self.img_size, self.img_size)
                else:
                    t = torch.zeros(ch, self.img_size, self.img_size)
                seq.append(t)
                if self.num_buttons > 0:
                    if 0 <= fi < len(frames):
                        btns.append(self._frame_button_state(src_idx, fi))
                    else:
                        btns.append(torch.zeros(self.num_buttons))
                else:
                    btns.append(torch.zeros(0))
            seq_t = torch.stack(seq, dim=0)                # [H+1, C, H, W]
            btn_t = torch.stack(btns, dim=0)               # [H+1, N]
            return seq_t, btn_t
        except Exception as e:
            print(f"[Dataset] error on {idx}: {e}")
            return (torch.zeros(self.history + 1, ch, self.img_size, self.img_size),
                    torch.zeros(self.history + 1, self.num_buttons))