"""
ae_flow_launcher.py

Drop-in launcher for LDM_flow_retry.py that swaps the VAE for a
deterministic autoencoder:

  - Encoder uses mu directly (no reparameterization, no sampling noise)
  - Training loss = MSE + L1_WEIGHT * L1   (no KL penalty)

Everything else (flow matching, text conditioning, GUI, augmentations,
ODEs, EMA, OT) is inherited unchanged from the original app.

Place this file next to LDM_flow_retry.py and run:
    python ae_flow_launcher.py
"""

import importlib.util
import sys
import os
import time
import multiprocessing
import tkinter as tk

import torch
import torch.optim
import torch.nn.functional as F
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# 1. Load the original module from disk
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ORIG = os.path.join(_HERE, "LDM_flow_retry.py")

if not os.path.exists(_ORIG):
    raise FileNotFoundError(
        f"Could not find LDM_flow_retry.py next to this launcher.\n"
        f"Looked at: {_ORIG}"
    )

_spec = importlib.util.spec_from_file_location("ldm_flow_retry", _ORIG)
ldm = importlib.util.module_from_spec(_spec)
sys.modules["ldm_flow_retry"] = ldm   # so pickling / isinstance checks work
_spec.loader.exec_module(ldm)


# ---------------------------------------------------------------------------
# 2. Deterministic AE — same architecture as FlexVAE, no sampling, no KL
# ---------------------------------------------------------------------------
class FlexAE(ldm.FlexVAE):
    """FlexVAE with reparameterize() returning mu directly.

    The encoder still has a logvar head, but since logvar is never used
    it simply receives no gradient and stays at its init values. Harmless.
    FlexVAE.encode() (returns mu) and FlexVAE.decode() work as-is.
    """
    def reparameterize(self, mu, logvar):
        return mu


# ---------------------------------------------------------------------------
# 3. App subclass — swaps VAE for AE + changes the reconstruction loss
# ---------------------------------------------------------------------------
class AEFlowApp(ldm.RectifiedFlowApp):
    # Tune this. 0.0 = pure MSE (blurrier). 0.5 = balanced. 1.0 = strong edge emphasis.
    L1_WEIGHT = 0.5

    def initialize_vae(self):
        try:
            in_ch = 3 if self.global_settings['color_mode'].get() == 'rgb' else 1
            self.vae_model = FlexAE(
                in_channels=in_ch,
                base_channels=self.vae_settings['vae_base_channels'].get(),
                latent_channels=self.vae_settings['vae_latent_channels'].get(),
                latent_h=self.vae_settings['vae_latent_h'].get(),
                latent_w=self.vae_settings['vae_latent_w'].get(),
                size=self.vae_settings['vae_size'].get(),
            ).to('cpu')
            self.vae_optimizer = torch.optim.Adam(
                self.vae_model.parameters(),
                lr=self.vae_settings['vae_lr'].get(),
            )
            self.log_vae(
                f"AE initialized (size={self.vae_settings['vae_size'].get()}, "
                f"loss = MSE + {self.L1_WEIGHT}*L1, no KL)."
            )
        except Exception as e:
            self.log_vae(f"Init error: {e}")

    def train_vae_loop(self, epochs):
        try:
            bs = self.vae_settings['vae_batch_size'].get()
            nw = self.vae_settings['vae_num_workers'].get()
            img_size = self.global_settings['img_size'].get()
            cm = self.global_settings['color_mode'].get()

            aug_dict = {k: v.get() for k, v in self.vae_aug_settings.items()}
            labels = self.labels if self.labels else [['']] * len(self.image_paths)
            ds = ldm.ConditionalImageDataset(
                self.image_paths, labels, img_size, cm, aug_dict,
                self.flow_settings['cond_text_max_len'].get(),
            )
            dl = DataLoader(
                ds, batch_size=bs, shuffle=True, num_workers=nw,
                pin_memory=False, persistent_workers=(nw > 0),
            )

            for ep in range(epochs):
                if not self.training_vae:
                    break
                tot, n = 0.0, 0
                for imgs, _ in dl:
                    if not self.training_vae:
                        break
                    imgs = imgs.to('cpu')
                    # FlexAE.forward uses mu directly (deterministic)
                    recon, mu, _ = self.vae_model(imgs)
                    mse = F.mse_loss(recon, imgs)
                    l1 = F.l1_loss(recon, imgs)
                    loss = mse + self.L1_WEIGHT * l1
                    self.vae_optimizer.zero_grad()
                    loss.backward()
                    self.vae_optimizer.step()
                    tot += loss.item()
                    n += 1
                avg = tot / max(1, n)
                el = time.time() - self.vae_start_time
                self.log_vae(
                    f"Epoch {ep+1}/{epochs} | Loss: {avg:.6f} | Time: {el:.1f}s"
                )
                if (ep + 1) % 5 == 0:
                    self.show_vae_preview()
            self.training_vae = False
            self.log_vae("AE training finished.")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.log_vae(f"AE training error: {e}")
            self.training_vae = False


# ---------------------------------------------------------------------------
# 4. Launch
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # already set

    root = tk.Tk()
    app = AEFlowApp(root)

    def _announce():
        app.log_vae(
            f"[AE MODE] Deterministic autoencoder active. "
            f"Loss = MSE + {AEFlowApp.L1_WEIGHT}*L1, no KL. "
            f"The 'KL weight' slider in the GUI is ignored."
        )

    root.after(500, _announce)
    root.mainloop()