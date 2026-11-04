# LDM_api.py
"""
Latent Rectified Flow — API / backend.

The VAE and latent UNet are LDM-specific; everything else (UNet blocks,
attention, conditioners, text encoders, dataset, OT matcher) is imported
from flow_api so the two UIs share one implementation.
"""
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F_nn

from flow_api import (
    Conv2dWithPadding, AttentionBlock, DownBlock, UpBlock,
    SinusoidalPositionEmbeddings, get_norm,
    TextEncoder, TransformerTextEncoder, get_encoder_config,
    ConditionalImageDataset,
    load_image_as_rgb, load_image_as_grayscale, text_to_indices,
    match_ot_batch, match_ot_batch_hungarian, match_ot_batch_sinkhorn,
    compute_head_dims,
)


# ==================== VAE ====================

VAE_SIZE_CONFIGS = {
    'tiny':   (2, [0.25, 0.5]),
    'small':  (3, [0.25, 0.5, 1.0]),
    'medium': (3, [0.5, 1.0, 2.0]),
    'big':    (3, [1.0, 2.0, 4.0]),
    'large':  (3, [2.0, 4.0, 8.0]),
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


class FlexEncoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=32, latent_channels=8,
                 latent_h=4, latent_w=4, size='big', img_size=32):
        super().__init__()
        _, mults = VAE_SIZE_CONFIGS[size]
        n = _vae_num_blocks(img_size, latent_h, latent_w)
        mults = _fit_mults(mults, n)
        self.latent_h, self.latent_w = latent_h, latent_w

        self.down = nn.ModuleList()
        c = in_channels
        for m in mults:
            oc = max(1, int(base_channels * m))
            self.down.append(nn.Conv2d(c, oc, 3, stride=2, padding=1))
            self.down.append(nn.LeakyReLU(0.2))
            c = oc
        self.conv_mu = nn.Conv2d(c, latent_channels, 3, padding=1)
        self.conv_logvar = nn.Conv2d(c, latent_channels, 3, padding=1)

    def forward(self, x):
        for l in self.down:
            x = l(x)
        return self.conv_mu(x), self.conv_logvar(x)


class FlexDecoder(nn.Module):
    def __init__(self, latent_channels=8, base_channels=32, out_channels=3,
                 latent_h=4, latent_w=4, size='big', img_size=32):
        super().__init__()
        _, mults = VAE_SIZE_CONFIGS[size]
        n = _vae_num_blocks(img_size, latent_h, latent_w)
        mults = _fit_mults(mults, n)
        widths = [max(1, int(base_channels * m)) for m in mults]
        self.latent_h, self.latent_w = latent_h, latent_w

        high = widths[-1]
        self.init_conv = nn.Conv2d(latent_channels, high, 3, padding=1)
        self.act = nn.LeakyReLU(0.2)
        self.up = nn.ModuleList()
        c = high
        for i in range(n):
            oc = widths[n - 2 - i] if i < n - 1 else widths[0]
            self.up.append(nn.ConvTranspose2d(c, oc, 4, stride=2, padding=1))
            self.up.append(nn.LeakyReLU(0.2))
            c = oc
        self.conv_out = nn.Conv2d(c, out_channels, 3, padding=1)

    def forward(self, z, target_size=None):
        x = self.act(self.init_conv(z))
        for l in self.up:
            x = l(x)
        out = torch.tanh(self.conv_out(x))
        if target_size and (out.shape[2] != target_size[0] or out.shape[3] != target_size[1]):
            out = F_nn.interpolate(out, size=target_size, mode='bilinear',
                                   align_corners=False)
        return out


class FlexVAE(nn.Module):
    def __init__(self, in_channels=3, base_channels=32, latent_channels=8,
                 latent_h=4, latent_w=4, size='big', img_size=32):
        super().__init__()
        self.encoder = FlexEncoder(in_channels, base_channels, latent_channels,
                                   latent_h, latent_w, size, img_size)
        self.decoder = FlexDecoder(latent_channels, base_channels, in_channels,
                                   latent_h, latent_w, size, img_size)
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


# ==================== Latent UNet ====================
# Same building blocks as flow_api.VelocityUNet, but the "img_size" concept
# is replaced by the latent spatial size. This lets the LDM use the full
# flow_api attention / cond_mode feature set while staying in latent space.

class VelocityUNet(nn.Module):
    def __init__(self, in_channels=8, base_channels=32, time_emb_dim=128, cond_dim=256,
                 latent_h=4, latent_w=4, channel_mult=(1, 2, 4), dropout=0.1,
                 attention_levels=None, attn_num_heads=4, attn_dropout=0.0,
                 padding_mode='constant', cond_mode='shift'):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.time_emb_dim = time_emb_dim
        self.cond_dim = cond_dim
        self.channel_mult = channel_mult
        self.padding_mode = padding_mode
        self.attn_num_heads = attn_num_heads
        self.attn_dropout = attn_dropout
        self.cond_mode = cond_mode
        self.latent_h = latent_h
        self.latent_w = latent_w

        # In latent space we can down to 1x1; keep halving while we still can.
        num_down = 0
        ch, cw = latent_h, latent_w
        while ch >= 2 and cw >= 2 and num_down < len(channel_mult) - 1:
            ch //= 2
            cw //= 2
            num_down += 1
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
            nn.Linear(time_emb_dim, time_emb_dim),
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
                cond_mode=cond_mode))
            if i < len(mult_used) - 1:
                self.downs.append(nn.Conv2d(out, out, 4, stride=2, padding=1))
            cur = out

        mid_attn = attn_flags[-1] if attn_flags else True
        self.mid_block1 = DownBlock(
            cur, cur, time_emb_dim, cond_dim, dropout,
            use_attention=mid_attn, attn_num_heads=attn_num_heads,
            attn_dropout=attn_dropout, padding_mode=padding_mode, cond_mode=cond_mode)
        self.mid_block2 = UpBlock(
            cur, cur, time_emb_dim, cond_dim, dropout,
            use_attention=mid_attn, attn_num_heads=attn_num_heads,
            attn_dropout=attn_dropout, padding_mode=padding_mode, cond_mode=cond_mode)

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
                cond_mode=cond_mode))
            if i < len(rev) - 1:
                self.ups.append(nn.ConvTranspose2d(out, out, 4, stride=2, padding=1))
            cur = out

        self.final_conv = nn.Sequential(
            get_norm(cur), nn.SiLU(),
            Conv2dWithPadding(cur, in_channels, 3, padding_mode=padding_mode),
        )

    def forward(self, x, t, cond=None):
        te = self.time_mlp(t)
        x = self.init_conv(x)
        skips = []
        for layer in self.downs:
            if isinstance(layer, DownBlock):
                x = layer(x, te, cond)
                skips.append(x)
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


# ==================== Rectified Flow (latent) ====================

class RectifiedFlowLatent:
    def __init__(self, latent_channels=8, latent_h=4, latent_w=4, base_channels=32,
                 cond_dim=256, device=None,
                 attention_levels=None, attn_num_heads=4, attn_dropout=0.0,
                 padding_mode='constant', cond_mode='shift',
                 channel_mult=(1, 2, 4)):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.latent_h, self.latent_w = latent_h, latent_w
        self.cond_dim = cond_dim
        self.cond_mode = cond_mode
        self.use_ot = False
        self.ot_method = 'sinkhorn'
        self.ot_sinkhorn_eps = 0.05
        self.ot_sinkhorn_iters = 50

        self.model = VelocityUNet(
            in_channels=latent_channels,
            base_channels=base_channels,
            time_emb_dim=max(64, base_channels * 4),
            cond_dim=cond_dim,
            latent_h=latent_h, latent_w=latent_w,
            channel_mult=channel_mult,
            dropout=0.1,
            attention_levels=attention_levels,
            attn_num_heads=attn_num_heads,
            attn_dropout=attn_dropout,
            padding_mode=padding_mode,
            cond_mode=cond_mode,
        ).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=2e-4)
        self.ema_model = None
        self.ema_decay = 0.999
        self.ema_mode = 'standard'
        self.use_ema = False

    # ---------- EMA ----------
    def set_ema(self, decay=0.999, mode='standard'):
        self.ema_decay, self.ema_mode, self.use_ema = decay, mode, True
        self.ema_model = VelocityUNet(
            in_channels=self.model.in_channels,
            base_channels=self.model.base_channels,
            time_emb_dim=self.model.time_emb_dim,
            cond_dim=self.model.cond_dim,
            latent_h=self.latent_h, latent_w=self.latent_w,
            channel_mult=self.model.channel_mult,
            dropout=0.1,
            attention_levels=self.model.attention_levels,
            attn_num_heads=self.model.attn_num_heads,
            attn_dropout=self.model.attn_dropout,
            padding_mode=self.model.padding_mode,
            cond_mode=self.cond_mode,
        ).to(self.device)
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

    # ---------- train ----------
    def train_step(self, z0, z1, cond=None, cfg_dropout_prob=0.0):
        self.model.train()  # FIX: make sure dropout is active during training
        B = z0.size(0)
        z0 = z0.to(self.device)
        z1 = z1.to(self.device)
        if self.use_ot and B > 1:
            z0, z1, col = match_ot_batch(
                z0, z1, method=self.ot_method,
                sinkhorn_eps=self.ot_sinkhorn_eps,
                sinkhorn_iters=self.ot_sinkhorn_iters)
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
        pred = self.model(zt, t, cond=cond)
        loss = self.criterion(pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.use_ema:
            self.update_ema()
        return loss.item()

    # ---------- sampling helpers ----------
    def _v_fn_factory(self, model, cond, cfg_scale):
        null = torch.zeros_like(cond)

        def v_fn(t_val, z_val):
            tt = torch.full((z_val.size(0),), t_val, device=self.device)
            if cfg_scale != 1.0:
                vc = model(z_val, tt, cond=cond)
                vu = model(z_val, tt, cond=null)
                return vu + cfg_scale * (vc - vu)
            return model(z_val, tt, cond=cond)

        return v_fn

    def _step(self, method, t_cur, dt, z, v_fn):
        if method == 'euler':
            z = z + v_fn(t_cur, z) * dt
        elif method == 'heun':
            v1 = v_fn(t_cur, z)
            zp = z + v1 * dt
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

    def _pick_model(self, use_ema):
        return self.ema_model if (use_ema and self.use_ema
                                  and self.ema_model is not None) else self.model

    @torch.no_grad()
    def sample(self, n_samples=16, cond=None, steps=50, method='euler', cfg_scale=1.0,
               use_ema=True, start_from=None):
        model = self._pick_model(use_ema)
        model.eval()  # FIX: disable dropout during inference
        shape = (n_samples, self.model.in_channels, self.latent_h, self.latent_w)
        z = (start_from.to(self.device) if start_from is not None
             else torch.randn(*shape, device=self.device))
        if cond is None:
            cond = torch.zeros(n_samples, self.cond_dim, device=self.device)
        else:
            cond = cond.to(self.device)
        dt = 1.0 / steps
        times = torch.linspace(0, 1, steps + 1, device=self.device)
        v_fn = self._v_fn_factory(model, cond, cfg_scale)
        for i in range(steps):
            z = self._step(method, times[i], dt, z, v_fn)
        return z

    @torch.no_grad()
    def sample_step_by_step(self, n_samples=16, cond=None, steps=50, method='euler',
                            cfg_scale=1.0, use_ema=True, start_from=None):
        model = self._pick_model(use_ema)
        model.eval()  # FIX
        shape = (n_samples, self.model.in_channels, self.latent_h, self.latent_w)
        z = (start_from.to(self.device) if start_from is not None
             else torch.randn(*shape, device=self.device))
        if cond is None:
            cond = torch.zeros(n_samples, self.cond_dim, device=self.device)
        else:
            cond = cond.to(self.device)
        dt = 1.0 / steps
        times = torch.linspace(0, 1, steps + 1, device=self.device)
        v_fn = self._v_fn_factory(model, cond, cfg_scale)
        for i in range(steps):
            z = self._step(method, times[i], dt, z, v_fn)
            yield i + 1, z.clone()

    # ---------- latent inpainting ----------
    @torch.no_grad()
    def inpaint_step_by_step(self, z_image, mask, steps=50, method='euler',
                             cfg_scale=1.0, use_ema=True, cond=None,
                             noise_sigma=1.0, composite_updates=True,
                             full_edit_from=0.7):
        """
        z_image : (B, C, h, w) latent of the original image
        mask    : (B, 1, h, w) at latent resolution; 1 = regenerate, 0 = keep
        """
        model = self._pick_model(use_ema)
        model.eval()  # FIX

        z_image = z_image.to(self.device)
        mask = mask.to(self.device)
        noise = torch.randn_like(z_image) * float(noise_sigma)
        z = z_image * (1 - mask) + noise * mask

        n = z_image.size(0)
        if cond is None:
            cond = torch.zeros(n, self.cond_dim, device=self.device)
        else:
            cond = cond.to(self.device)
        dt = 1.0 / steps
        times = torch.linspace(0, 1, steps + 1, device=self.device)
        v_fn = self._v_fn_factory(model, cond, cfg_scale)

        for i in range(steps):
            z = self._step(method, times[i], dt, z, v_fn)
            if composite_updates:
                progress = (i + 1) / steps
                if progress < full_edit_from:
                    z = z * mask + z_image * (1 - mask)
            yield i + 1, z.clone()

    @torch.no_grad()
    def inpaint(self, z_image, mask, steps=50, method='euler', cfg_scale=1.0,
                use_ema=True, cond=None, noise_sigma=1.0,
                composite_updates=True, full_edit_from=0.7):
        last = None
        for _, z in self.inpaint_step_by_step(
                z_image, mask, steps=steps, method=method, cfg_scale=cfg_scale,
                use_ema=use_ema, cond=cond, noise_sigma=noise_sigma,
                composite_updates=composite_updates,
                full_edit_from=full_edit_from):
            last = z
        return last