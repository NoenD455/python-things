# video_pixel_api.py
"""
Video Rectified Flow in PIXEL space — API / backend.

Unlike video_ldm_api.py (which operates in a VAE latent space), this module
trains and samples the flow directly on pixels. No VAE is needed.

Reuses from video_ldm_api:
  - VideoFrameDataset, load_video_frames  (pixel data already)
  - VelocityUNet (temporal UNet; spatial size is a plain parameter)
  - RectifiedFlowLatent (autoregressive sampling machinery)

Reuses from flow_api (transitively through video_ldm_api):
  - blocks / attention / OT matcher / augmentations / text encoders.
"""
from video_ldm_api import (
    VideoFrameDataset, load_video_frames,
    VelocityUNet, RectifiedFlowLatent,
    load_image_as_rgb, load_image_as_grayscale, text_to_indices,
    TextEncoder, TransformerTextEncoder, get_encoder_config,
    VIDEO_EXTS, IMAGE_EXTS, _HAS_CV2, _HAS_IMAGEIO, VALID_COND_MODES,
)


class RectifiedFlowPixelVideo(RectifiedFlowLatent):
    """Pixel-space rectified flow for video.

    Identical machinery to ``RectifiedFlowLatent`` but with pixel-appropriate
    parameter names. The flow operates directly on
    ``in_channels × img_size × img_size`` tensors in ``[-1, 1]``; there is no
    VAE encode/decode step.
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
        # Pixel-friendly aliases (parent keeps latent_* for internal use)
        self.in_channels = in_channels
        self.img_size = img_size

    @property
    def pixel_h(self):
        return self.latent_h

    @property
    def pixel_w(self):
        return self.latent_w