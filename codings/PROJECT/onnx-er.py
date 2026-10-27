#!/usr/bin/env python3
"""
onnx_converter_gui.py

Tkinter GUI to export a Rectified Flow model (from flow_api.py) to ONNX
for inference only.

What gets exported
------------------
The velocity UNet:
    forward(x: [B,C,H,W], t: [B], cond: [B,cond_dim]) -> velocity: [B,C,H,W]

The ODE stepping (Euler / Heun / RK4 / Midpoint, CFG, EMA handling, inpainting
composites) all live in Python and are NOT part of the ONNX graph — you just
run the UNet graph from your own sampling loop.

Run:
    python onnx_converter_gui.py

Requires: torch, onnx, onnxruntime, and flow_api.py in the same directory.
"""
import os
import threading
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

import torch

from flow_api import VelocityUNet, RectifiedFlowPixel


# ----------------------------------------------------------------------
# GUI
# ----------------------------------------------------------------------
class ONNXConverterApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Flow Matching → ONNX (inference)")
        root.geometry("860x760")

        # --- Vars ---
        self.pth_path = tk.StringVar()
        self.onnx_path = tk.StringVar()

        self.in_channels = tk.StringVar(value="3")
        self.base_channels = tk.StringVar(value="64")
        self.cond_dim = tk.StringVar(value="256")
        self.img_size = tk.StringVar(value="32")
        self.channel_mult = tk.StringVar(value="1,2,4")
        self.attention_levels = tk.StringVar(value="True,True,True")
        self.attn_num_heads = tk.StringVar(value="4")
        self.attn_dropout = tk.StringVar(value="0.0")
        self.padding_mode = tk.StringVar(value="constant")

        self.opset = tk.StringVar(value="14")
        self.use_ema = tk.BooleanVar(value=True)
        self.dynamic_batch = tk.BooleanVar(value=True)
        self.dynamic_hw = tk.BooleanVar(value=False)

        self.load_mode = tk.StringVar(value="auto")

        self._build_ui()

    # ---------- UI ----------
    def _build_ui(self):
        root = self.root
        main = ttk.Frame(root, padding=10)
        main.pack(fill="both", expand=True)

        # Files
        f_files = ttk.LabelFrame(main, text="Files", padding=8)
        f_files.pack(fill="x", pady=4)
        ttk.Label(f_files, text="Input .pth:").grid(row=0, column=0, sticky="w")
        ttk.Entry(f_files, textvariable=self.pth_path).grid(
            row=0, column=1, padx=4, sticky="ew")
        ttk.Button(f_files, text="Browse…", command=self._pick_pth).grid(
            row=0, column=2)
        ttk.Label(f_files, text="Output .onnx:").grid(row=1, column=0, sticky="w")
        ttk.Entry(f_files, textvariable=self.onnx_path).grid(
            row=1, column=1, padx=4, sticky="ew")
        ttk.Button(f_files, text="Browse…", command=self._pick_onnx).grid(
            row=1, column=2)
        f_files.columnconfigure(1, weight=1)

        # Load mode
        f_mode = ttk.LabelFrame(main, text="Checkpoint format", padding=8)
        f_mode.pack(fill="x", pady=4)
        for txt, val in [
            ("Auto-detect", "auto"),
            ("Full saved model (RectifiedFlowPixel or VelocityUNet)",
             "full"),
            ("state_dict (weights only)", "state_dict"),
        ]:
            ttk.Radiobutton(f_mode, text=txt, variable=self.load_mode,
                            value=val).pack(anchor="w")

        # Architecture
        f_arch = ttk.LabelFrame(
            main, text="Architecture (used for state_dict / to fill defaults)",
            padding=8)
        f_arch.pack(fill="x", pady=4)

        rows = [
            ("in_channels:", self.in_channels),
            ("base_channels:", self.base_channels),
            ("cond_dim:", self.cond_dim),
            ("img_size (H=W):", self.img_size),
            ("channel_mult (comma sep):", self.channel_mult),
            ("attention_levels (True/False, comma sep):", self.attention_levels),
            ("attn_num_heads:", self.attn_num_heads),
            ("attn_dropout:", self.attn_dropout),
        ]
        for i, (lbl, var) in enumerate(rows):
            ttk.Label(f_arch, text=lbl).grid(row=i, column=0, sticky="w", pady=1)
            ttk.Entry(f_arch, textvariable=var, width=40).grid(
                row=i, column=1, sticky="w", padx=4)
        ttk.Label(f_arch, text="padding_mode:").grid(row=len(rows), column=0,
                                                      sticky="w", pady=1)
        ttk.Combobox(f_arch, textvariable=self.padding_mode,
                     values=["constant", "reflect", "replicate", "circular"],
                     state="readonly", width=37).grid(
            row=len(rows), column=1, sticky="w", padx=4)

        # Export options
        f_exp = ttk.LabelFrame(main, text="Export options", padding=8)
        f_exp.pack(fill="x", pady=4)
        ttk.Checkbutton(f_exp, text="Use EMA weights if available",
                        variable=self.use_ema).pack(anchor="w")
        ttk.Checkbutton(f_exp, text="Dynamic batch dimension",
                        variable=self.dynamic_batch).pack(anchor="w")
        ttk.Checkbutton(f_exp, text="Dynamic H/W dimension (rarely needed)",
                        variable=self.dynamic_hw).pack(anchor="w")
        row = ttk.Frame(f_exp); row.pack(anchor="w", pady=2)
        ttk.Label(row, text="ONNX opset:").pack(side="left")
        ttk.Entry(row, textvariable=self.opset, width=6).pack(side="left", padx=4)

        # Buttons
        f_btn = ttk.Frame(main)
        f_btn.pack(fill="x", pady=8)
        self.btn_convert = ttk.Button(
            f_btn, text="Convert to ONNX", command=self._start_convert)
        self.btn_convert.pack(side="left")
        self.btn_validate = ttk.Button(
            f_btn, text="Validate ONNX", command=self._start_validate)
        self.btn_validate.pack(side="left", padx=4)
        ttk.Button(f_btn, text="Clear log",
                   command=self._clear_log).pack(side="right")

        # Log
        f_log = ttk.LabelFrame(main, text="Log", padding=4)
        f_log.pack(fill="both", expand=True, pady=4)
        self.log = scrolledtext.ScrolledText(f_log, height=18, state="disabled",
                                             wrap="word")
        self.log.pack(fill="both", expand=True)

    # ---------- helpers ----------
    def _log(self, msg):
        self.log.configure(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")
        self.root.update_idletasks()

    def _clear_log(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def _pick_pth(self):
        p = filedialog.askopenfilename(
            title="Select checkpoint",
            filetypes=[("PyTorch", "*.pth *.pt *.ckpt"), ("All files", "*.*")])
        if p:
            self.pth_path.set(p)
            if not self.onnx_path.get():
                self.onnx_path.set(os.path.splitext(p)[0] + ".onnx")

    def _pick_onnx(self):
        p = filedialog.asksaveasfilename(
            defaultextension=".onnx",
            filetypes=[("ONNX", "*.onnx"), ("All files", "*.*")])
        if p:
            self.onnx_path.set(p)

    # ---------- arch parsing ----------
    def _parse_int_tuple(self, s):
        return tuple(int(x.strip()) for x in s.split(",") if x.strip())

    def _parse_bool_list(self, s):
        return [p.strip().lower() in ("true", "1", "yes", "y")
                for p in s.split(",") if p.strip()]

    # ---------- checkpoint inspection ----------
    def _looks_like_state_dict(self, obj):
        return (isinstance(obj, dict) and obj
                and all(isinstance(v, torch.Tensor) for v in obj.values()))

    def _tensor_only(self, obj):
        """Extract tensor-valued entries whose keys look like model params."""
        if not isinstance(obj, dict):
            return None
        t = {k: v for k, v in obj.items() if isinstance(v, torch.Tensor)}
        if not t:
            return None
        # Require at least one ".weight" / ".bias" key to avoid
        # accidentally grabbing optimizer hyperparameters.
        if not any((".weight" in k or ".bias" in k) for k in t):
            return None
        return t

    def _strip_prefix(self, sd):
        """Strip a common module prefix if present."""
        prefixes = ("model.", "module.", "_orig_mod.", "ema_model.", "ema.",
                    "net.", "flow.", "network.")
        # Uniform prefix -> clean strip
        for pfx in prefixes:
            if sd and all(k.startswith(pfx) for k in sd.keys()):
                return {k[len(pfx):]: v for k, v in sd.items()}
        # Mixed prefix -> strip only from keys that have it
        for pfx in prefixes:
            if any(k.startswith(pfx) for k in sd.keys()):
                stripped = {k[len(pfx):] if k.startswith(pfx) else k: v
                            for k, v in sd.items()}
                if all(isinstance(v, torch.Tensor) for v in stripped.values()):
                    return stripped
        return sd

    def _unwrap_state_dict(self, obj):
        if not isinstance(obj, dict):
            return None

        # 1) Direct state dict
        if self._looks_like_state_dict(obj):
            return self._strip_prefix(obj)

        # 2) Nested under a known key
        known_keys = (
            "state_dict", "model_state_dict", "model", "unet", "network",
            "net", "flow", "flow_model", "ema_model", "ema_model_state_dict",
            "ema", "model_ema", "net_state_dict", "weights", "params",
            "model_state", "model_weights", "state", "sd",
        )
        for key in known_keys:
            if key in obj and isinstance(obj[key], dict):
                inner = obj[key]
                if self._looks_like_state_dict(inner):
                    self._log(f"(found state dict under key '{key}')")
                    return self._strip_prefix(inner)

        # 3) Fallback: mixed dict — keep only tensor entries that look like params
        t = self._tensor_only(obj)
        if t is not None:
            self._log(f"(extracted {len(t)} tensor entries from mixed dict)")
            return self._strip_prefix(t)

        return None

    def _infer_arch_from_sd(self, sd):
        """Best-effort fill of the architecture fields from a state_dict."""
        try:
            if "init_conv.weight" in sd:
                w = sd["init_conv.weight"]                       # [base, C, 3, 3]
                self.base_channels.set(str(int(w.shape[0])))
                self.in_channels.set(str(int(w.shape[1])))

            base = int(self.base_channels.get())

            if "downs.0.cond_proj.weight" in sd:
                self.cond_dim.set(str(int(sd["downs.0.cond_proj.weight"].shape[1])))

            # channel_mult & attention_levels from downs.0, downs.2, downs.4, ...
            mults, attn = [], []
            for idx in range(0, 40, 2):
                wkey = f"downs.{idx}.conv1.weight"
                if wkey not in sd:
                    break
                mults.append(int(sd[wkey].shape[0]) // max(base, 1))
                attn.append("True" if f"downs.{idx}.attn.attn.in_proj_weight"
                            in sd else "False")
            if mults:
                self.channel_mult.set(",".join(str(m) for m in mults))
                self.attention_levels.set(",".join(attn))

            self._log("Architecture inferred from state_dict:")
            self._log(f"  in_channels      = {self.in_channels.get()}")
            self._log(f"  base_channels    = {self.base_channels.get()}")
            self._log(f"  cond_dim         = {self.cond_dim.get()}")
            self._log(f"  channel_mult     = {self.channel_mult.get()}")
            self._log(f"  attention_levels = {self.attention_levels.get()}")
            self._log("  img_size / attn_num_heads could not be inferred — "
                      "set them manually if wrong.")
        except Exception as e:
            self._log(f"(arch inference partial: {e})")

    # ---------- checkpoint loading ----------
    def _load(self):
        """Returns (model_for_export, img_size_H, img_size_W, note)."""
        path = self.pth_path.get().strip()
        if not path or not os.path.isfile(path):
            raise RuntimeError("Please choose a valid .pth file.")

        self._log(f"Loading checkpoint: {path}")
        try:
            obj = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:
            raise RuntimeError(f"torch.load failed: {e}")

        mode = self.load_mode.get()

        # ---- Auto-detect ----
        if mode == "auto":
            if isinstance(obj, RectifiedFlowPixel):
                mode = "full_rf"
            elif isinstance(obj, VelocityUNet):
                mode = "full_unet"
            elif self._unwrap_state_dict(obj) is not None:
                mode = "state_dict"
            else:
                # Dump dict structure to help the user
                if isinstance(obj, dict):
                    self._log("Checkpoint is a dict. Top-level entries:")
                    for k, v in list(obj.items())[:40]:
                        if isinstance(v, torch.Tensor):
                            self._log(f"  {k!r}: Tensor shape={tuple(v.shape)}")
                        elif isinstance(v, dict):
                            self._log(f"  {k!r}: dict with {len(v)} keys")
                        else:
                            self._log(f"  {k!r}: {type(v).__name__}")
                raise RuntimeError(
                    "Auto-detect failed. See the dict dump in the log above; "
                    "select a format explicitly, or the dict doesn't contain a "
                    "recognisable state_dict.")
            self._log(f"Auto-detected: {mode}")

        # ---- Full RectifiedFlowPixel ----
        if mode in ("full", "full_rf"):
            if isinstance(obj, RectifiedFlowPixel):
                rf = obj
                model = rf.model
                if self.use_ema.get() and rf.use_ema and rf.ema_model is not None:
                    self._log("Using EMA weights from the loaded RectifiedFlowPixel.")
                    model = rf.ema_model
                H, W = rf.img_size if isinstance(rf.img_size, tuple) \
                    else (rf.img_size, rf.img_size)
                self._populate_from_rf(rf)
                return model, H, W, "RectifiedFlowPixel"
            if mode == "full_rf":
                raise RuntimeError(
                    "Selected full RectifiedFlowPixel, but the file is not one.")

        # ---- Full VelocityUNet ----
        if mode in ("full", "full_unet"):
            if isinstance(obj, VelocityUNet):
                return (obj,
                        int(self.img_size.get()),
                        int(self.img_size.get()),
                        "VelocityUNet")
            if mode == "full_unet":
                raise RuntimeError(
                    "Selected full VelocityUNet, but the file is not one.")

        # ---- state_dict ----
        if mode in ("state_dict", "full", "auto"):
            sd = self._unwrap_state_dict(obj)
            if sd is not None:
                # Auto-populate arch fields from the state dict itself
                self._infer_arch_from_sd(sd)

                cm = self._parse_int_tuple(self.channel_mult.get())
                al = self._parse_bool_list(self.attention_levels.get())
                base = int(self.base_channels.get())
                model = VelocityUNet(
                    in_channels=int(self.in_channels.get()),
                    base_channels=base,
                    time_emb_dim=base * 4,
                    cond_dim=int(self.cond_dim.get()),
                    img_size=int(self.img_size.get()),
                    channel_mult=cm,
                    dropout=0.1,
                    attention_levels=al,
                    attn_num_heads=int(self.attn_num_heads.get()),
                    attn_dropout=float(self.attn_dropout.get()),
                    padding_mode=self.padding_mode.get(),
                )

                mk = set(model.state_dict().keys())
                sk = set(sd.keys())
                if not mk.issubset(sk) or not sk.issubset(mk):
                    missing = sorted(mk - sk)[:8]
                    extra   = sorted(sk - mk)[:8]
                    self._log("WARNING: state_dict keys do not fully match the "
                              "constructed model.")
                    if missing: self._log(f"  missing: {missing}")
                    if extra:   self._log(f"  extra:   {extra}")

                self._log("Loading state_dict…")
                try:
                    model.load_state_dict(sd, strict=True)
                except RuntimeError as e:
                    self._log(f"strict=True failed ({e}); retrying strict=False")
                    model.load_state_dict(sd, strict=False)

                return (model,
                        int(self.img_size.get()),
                        int(self.img_size.get()),
                        "state_dict")

        raise RuntimeError("Could not interpret the checkpoint with the "
                           "selected format.")

    def _populate_from_rf(self, rf: "RectifiedFlowPixel"):
        try:
            m = rf.model
            self.in_channels.set(str(m.in_channels))
            self.base_channels.set(str(m.base_channels))
            self.cond_dim.set(str(m.cond_dim))
            H, W = rf.img_size if isinstance(rf.img_size, tuple) \
                else (rf.img_size, rf.img_size)
            self.img_size.set(str(H) if H == W else f"{H}")
            self.channel_mult.set(",".join(str(x) for x in m.channel_mult))
            self.attention_levels.set(
                ",".join("True" if a else "False" for a in m.attention_levels))
            self.attn_num_heads.set(str(m.attn_num_heads))
            self.attn_dropout.set(str(m.attn_dropout))
            self.padding_mode.set(m.padding_mode)
        except Exception as e:
            self._log(f"(could not fully populate arch fields: {e})")

    # ---------- conversion ----------
    def _start_convert(self):
        self.btn_convert.configure(state="disabled")
        threading.Thread(target=self._do_convert, daemon=True).start()

    def _do_convert(self):
        try:
            self._do_convert_impl()
        except Exception as e:
            self._log("ERROR:")
            self._log(str(e))
            self._log(traceback.format_exc())
        finally:
            self.btn_convert.configure(state="normal")

    def _do_convert_impl(self):
        out_path = self.onnx_path.get().strip()
        if not out_path:
            raise RuntimeError("Please choose an output .onnx path.")

        model, H, W, note = self._load()
        model.eval().to("cpu")

        C = model.in_channels
        cond_dim = model.cond_dim

        self._log(f"Model source: {note}")
        self._log(f"Exported input: x=[B,{C},{H},{W}], t=[B], cond=[B,{cond_dim}]")

        dummy_x = torch.randn(1, C, H, W, dtype=torch.float32)
        dummy_t = torch.rand(1, dtype=torch.float32)
        dummy_cond = torch.randn(1, cond_dim, dtype=torch.float32)

        dynamic_axes = None
        if self.dynamic_batch.get() or self.dynamic_hw.get():
            dynamic_axes = {}
            if self.dynamic_batch.get():
                dynamic_axes["x"] = {0: "batch"}
                dynamic_axes["t"] = {0: "batch"}
                dynamic_axes["cond"] = {0: "batch"}
                dynamic_axes["velocity"] = {0: "batch"}
            if self.dynamic_hw.get():
                dynamic_axes.setdefault("x", {})[2] = "height"
                dynamic_axes.setdefault("x", {})[3] = "width"
                dynamic_axes.setdefault("velocity", {})[2] = "height"
                dynamic_axes.setdefault("velocity", {})[3] = "width"

        self._log(f"Exporting to: {out_path}")
        torch.onnx.export(
            model,
            (dummy_x, dummy_t, dummy_cond),
            out_path,
            input_names=["x", "t", "cond"],
            output_names=["velocity"],
            dynamic_axes=dynamic_axes,
            opset_version=int(self.opset.get()),
            do_constant_folding=True,
            export_params=True,
            training=torch.onnx.TrainingMode.EVAL,
        )
        self._log("Export completed.")
        self._validate_torch_vs_onnx(out_path, model, dummy_x, dummy_t, dummy_cond)

    # ---------- validation ----------
    def _start_validate(self):
        path = self.onnx_path.get().strip()
        if not path or not os.path.isfile(path):
            messagebox.showerror("Error", "Please select an existing .onnx file.")
            return
        threading.Thread(target=self._do_validate_only,
                         args=(path,), daemon=True).start()

    def _do_validate_only(self, path):
        try:
            import onnx
            import onnxruntime as ort
            import numpy as np

            onnx_model = onnx.load(path)
            onnx.checker.check_model(onnx_model)
            self._log("ONNX checker: model structure is valid.")

            sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            self._log("Inputs:")
            for i in sess.get_inputs():
                self._log(f"  {i.name}: shape={i.shape}, dtype={i.type}")
            self._log("Outputs:")
            for o in sess.get_outputs():
                self._log(f"  {o.name}: shape={o.shape}, dtype={o.type}")

            feed = {}
            for i in sess.get_inputs():
                shape = [d if isinstance(d, int) else 1 for d in i.shape]
                feed[i.name] = np.random.randn(*shape).astype(np.float32)
            out = sess.run(None, feed)
            self._log(f"Runtime inference OK. Outputs: "
                      f"{[o.shape for o in out]}")
        except Exception as e:
            self._log(f"Validation error: {e}")
            self._log(traceback.format_exc())

    def _validate_torch_vs_onnx(self, path, torch_model,
                                dummy_x, dummy_t, dummy_cond):
        try:
            import onnx
            import onnxruntime as ort
            import numpy as np

            onnx_model = onnx.load(path)
            onnx.checker.check_model(onnx_model)
            self._log("ONNX checker: OK")

            sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

            with torch.no_grad():
                torch_out = torch_model(dummy_x, dummy_t, dummy_cond).numpy()

            ort_out = sess.run(
                ["velocity"],
                {"x": dummy_x.numpy(),
                 "t": dummy_t.numpy(),
                 "cond": dummy_cond.numpy()},
            )[0]

            diff = float(np.abs(torch_out - ort_out).max())
            self._log(f"Max |torch - onnx| on random input: {diff:.3e}")
            if diff < 1e-3:
                self._log("Numerical validation: PASS")
            else:
                self._log("Numerical validation: WARNING (large difference)")

            # Also try a batch of 2 to confirm dynamic batch (if enabled)
            if self.dynamic_batch.get():
                x2 = dummy_x.repeat(2, 1, 1, 1)
                t2 = dummy_t.repeat(2)
                c2 = dummy_cond.repeat(2, 1)
                out2 = sess.run(
                    ["velocity"],
                    {"x": x2.numpy(), "t": t2.numpy(), "cond": c2.numpy()},
                )[0]
                self._log(f"Dynamic batch test OK: output shape = {out2.shape}")
        except Exception as e:
            self._log(f"Post-export validation error: {e}")
            self._log(traceback.format_exc())


# ----------------------------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    try:
        style = ttk.Style(root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        elif "clam" in style.theme_names():
            style.theme_use("clam")
    except Exception:
        pass
    ONNXConverterApp(root)
    root.mainloop()