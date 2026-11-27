# dream_pixel_api.py
"""
Dream Pixel LDM — Pixel-space World Model API.

Same button-conditioned video flow as dream_ldm_api.py, but operating directly
on pixels (no VAE). Reuses the temporal VelocityUNet and autoregressive
RectifiedFlowLatent machinery.
"""
from dream_ldm_api import (
    VideoFrameDataset, ButtonEncoder,
    load_video_frames, load_video_fps,
    load_button_csv, buttons_at_time, collect_button_names,
    load_image_as_rgb, load_image_as_grayscale,
    VIDEO_EXTS, IMAGE_EXTS, _HAS_CV2, _HAS_IMAGEIO,
    VelocityUNet, RectifiedFlowLatent, VALID_COND_MODES,
)


class RectifiedFlowPixelVideo(RectifiedFlowLatent):
    """Pixel-space button-conditioned rectified flow for video.

    Identical machinery to ``RectifiedFlowLatent``, but parameterized by
    pixel dimensions. Operates on ``in_channels × img_size × img_size``
    tensors in ``[-1, 1]``; no VAE encode/decode step.
    """
    def __init__(self, in_channels=3, img_size=32, history=1,
                 base_channels=32, cond_dim=256, device=None,
                 attention_levels=None, attn_num_heads=4, attn_dropout=0.0,
                 padding_mode='constant', cond_mode='shift',
                 channel_mult=(1, 2, 4)):
        super().__init__(
            latent_channels=in_channels,
            latent_h=img_size,
            latent_w=img_size,
            history=history,
            base_channels=base_channels,
            cond_dim=cond_dim,
            device=device,
            attention_levels=attention_levels,
            attn_num_heads=attn_num_heads,
            attn_dropout=attn_dropout,
            padding_mode=padding_mode,
            cond_mode=cond_mode,
            channel_mult=channel_mult,
        )
        self.in_channels = in_channels
        self.img_size = img_size

    @property
    def pixel_h(self):
        return self.latent_h

    @property
    def pixel_w(self):
        return self.latent_w