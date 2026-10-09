#!/usr/bin/env python3
"""
rgba_fixer.py

Batch-fix RGBA images whose fully transparent pixels still carry colour data.
Classic symptom: Doom-extracted PNGs where alpha == 0 pixels are (0,255,255,0)
cyan instead of (0,0,0,0).  That happens because palette images (P mode) store
RGB in the palette even for the "transparent" index, and converting P -> RGBA
just copies that palette colour into the RGB channels.

This tool finds every pixel with alpha <= threshold (default 0 = fully
transparent only) and forces its R, G, B to 0.  Visible pixels are never
touched.

Requires:  Pillow
Optional:  numpy  (much faster on large batches; falls back to pure PIL)
"""

from __future__ import annotations

import os
import queue
import shutil
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

from PIL import Image, ImageChops

try:
    import numpy as np
    HAVE_NUMPY = True
except Exception:
    HAVE_NUMPY = False


# ----------------------------------------------------------------------------
# Core image operation
# ----------------------------------------------------------------------------

def fix_transparent_pixels(img: Image.Image, threshold: int = 0):
    """Return (new_image, changed_pixel_count).

    Every pixel whose alpha is <= `threshold` gets RGB = (0, 0, 0).
    Alpha values themselves are left untouched.
    """
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    if HAVE_NUMPY:
        arr = np.array(img, dtype=np.uint8)
        if arr.size == 0:
            return img, 0

        alpha = arr[:, :, 3]
        mask = (alpha <= threshold) if threshold > 0 else (alpha == 0)
        if not mask.any():
            return img, 0

        sub = arr[mask]                     # (N, 4)
        changed = int(np.count_nonzero(
            (sub[:, 0] != 0) | (sub[:, 1] != 0) | (sub[:, 2] != 0)
        ))
        if changed:
            arr[mask, 0] = 0
            arr[mask, 1] = 0
            arr[mask, 2] = 0
            return Image.fromarray(arr, "RGBA"), changed
        return img, 0

    # --- pure-PIL fallback ---------------------------------------------------
    r, g, b, a = img.split()
    if threshold > 0:
        mask = a.point(lambda v: 255 if v <= threshold else 0)
    else:
        mask = a.point(lambda v: 255 if v == 0 else 0)

    if mask.getbbox() is None:              # nothing transparent at all
        return img, 0

    black = Image.new("L", img.size, 0)
    nr = Image.composite(black, r, mask)
    ng = Image.composite(black, g, mask)
    nb = Image.composite(black, b, mask)

    old_rgb = Image.merge("RGB", (r, g, b))
    new_rgb = Image.merge("RGB", (nr, ng, nb))
    diff = ImageChops.difference(old_rgb, new_rgb)
    any_ch = ImageChops.lighter(
        ImageChops.lighter(diff.getchannel(0), diff.getchannel(1)),
        diff.getchannel(2),
    )
    changed = sum(any_ch.histogram()[1:])
    if changed == 0:
        return img, 0
    return Image.merge("RGBA", (nr, ng, nb, a)), changed


def collect_files(folder: str, exts: set, recursive: bool):
    out = []
    if recursive:
        for root, _dirs, names in os.walk(folder):
            for n in names:
                if os.path.splitext(n)[1].lower() in exts:
                    out.append(os.path.join(root, n))
    else:
        for n in os.listdir(folder):
            p = os.path.join(folder, n)
            if os.path.isfile(p) and os.path.splitext(n)[1].lower() in exts:
                out.append(p)
    return sorted(out)


# ----------------------------------------------------------------------------
# GUI
# ----------------------------------------------------------------------------

class FixerApp(tk.Tk):

    POLL_MS = 80

    def __init__(self):
        super().__init__()
        self.title("RGBA Transparent-Pixel Fixer")
        self.minsize(760, 560)

        self.msg_queue: queue.Queue = queue.Queue()
        self.worker: threading.Thread | None = None
        self.stop_event = threading.Event()

        self._build_ui()
        self._sync_dest()
        self.after(self.POLL_MS, self._poll)

    # -- UI construction ------------------------------------------------------

    def _build_ui(self):
        pad = dict(padx=8, pady=4)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(5, weight=1)

        # --- source ----------------------------------------------------------
        src = ttk.LabelFrame(self, text="Source")
        src.grid(row=0, column=0, sticky="ew", **pad)
        src.columnconfigure(1, weight=1)

        self.in_dir = tk.StringVar()
        ttk.Label(src, text="Input folder:").grid(
            row=0, column=0, sticky="w", padx=(6, 4), pady=6)
        ttk.Entry(src, textvariable=self.in_dir).grid(
            row=0, column=1, sticky="ew", padx=4, pady=6)
        ttk.Button(src, text="Browse…", command=self._pick_in).grid(
            row=0, column=2, padx=6, pady=6)

        # --- destination -----------------------------------------------------
        dst = ttk.LabelFrame(self, text="Destination")
        dst.grid(row=1, column=0, sticky="ew", **pad)
        dst.columnconfigure(1, weight=1)

        self.overwrite = tk.BooleanVar(value=False)
        ttk.Radiobutton(
            dst, text="Overwrite originals in place",
            variable=self.overwrite, value=True, command=self._sync_dest,
        ).grid(row=0, column=0, columnspan=3, sticky="w", padx=6, pady=(6, 0))

        ttk.Radiobutton(
            dst, text="Write copies to:",
            variable=self.overwrite, value=False, command=self._sync_dest,
        ).grid(row=1, column=0, sticky="w", padx=6, pady=4)

        self.out_dir = tk.StringVar()
        self.out_entry = ttk.Entry(dst, textvariable=self.out_dir)
        self.out_entry.grid(row=1, column=1, sticky="ew", padx=4, pady=4)
        self.out_btn = ttk.Button(dst, text="Browse…", command=self._pick_out)
        self.out_btn.grid(row=1, column=2, padx=6, pady=4)

        self.backup = tk.BooleanVar(value=True)
        self.backup_chk = ttk.Checkbutton(
            dst, text="Keep a “.bak” copy of each file before overwriting it",
            variable=self.backup)
        self.backup_chk.grid(row=2, column=0, columnspan=3,
                             sticky="w", padx=6, pady=(0, 6))

        # --- options ---------------------------------------------------------
        opt = ttk.LabelFrame(self, text="Options")
        opt.grid(row=2, column=0, sticky="ew", **pad)

        self.recursive = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt, text="Include subfolders",
                        variable=self.recursive).grid(
            row=0, column=0, sticky="w", padx=6, pady=6)

        ttk.Label(opt, text="File types:").grid(
            row=0, column=1, sticky="e", padx=(12, 4), pady=6)
        self.exts = tk.StringVar(value=".png")
        ttk.Entry(opt, textvariable=self.exts, width=30).grid(
            row=0, column=2, columnspan=2, sticky="w", padx=4, pady=6)

        ttk.Label(opt, text="Alpha threshold:").grid(
            row=1, column=1, sticky="e", padx=(12, 4), pady=(0, 6))
        self.threshold = tk.IntVar(value=0)
        ttk.Spinbox(opt, from_=0, to=255, textvariable=self.threshold,
                    width=6).grid(row=1, column=2, sticky="w", padx=4,
                                  pady=(0, 6))
        ttk.Label(opt, text="(0 = only fully transparent pixels)").grid(
            row=1, column=3, sticky="w", padx=4, pady=(0, 6))

        # --- buttons ---------------------------------------------------------
        btns = ttk.Frame(self)
        btns.grid(row=3, column=0, sticky="ew", padx=8, pady=4)

        self.start_btn = ttk.Button(btns, text="Start", command=self._start)
        self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(btns, text="Stop", command=self._stop,
                                   state="disabled")
        self.stop_btn.pack(side="left", padx=6)
        ttk.Button(btns, text="Clear log",
                   command=self._clear_log).pack(side="left", padx=6)

        self.status = tk.StringVar(value="Idle.")
        ttk.Label(btns, textvariable=self.status).pack(side="right")

        # --- progress + log --------------------------------------------------
        self.progress = ttk.Progressbar(self, mode="determinate")
        self.progress.grid(row=4, column=0, sticky="ew", padx=8, pady=(0, 4))

        self.log_box = ScrolledText(self, height=14, wrap="none",
                                    state="disabled")
        self.log_box.grid(row=5, column=0, sticky="nsew", padx=8, pady=(0, 8))

        self._append_log(
            "Ready.  Numpy acceleration: " + ("on" if HAVE_NUMPY else "off")
        )

    # -- small helpers --------------------------------------------------------

    def _sync_dest(self):
        ow = self.overwrite.get()
        state = "disabled" if ow else "normal"
        self.out_entry.configure(state=state)
        self.out_btn.configure(state=state)
        self.backup_chk.configure(state="normal" if ow else "disabled")

    def _pick_in(self):
        d = filedialog.askdirectory(title="Choose the folder with images")
        if d:
            self.in_dir.set(d)

    def _pick_out(self):
        d = filedialog.askdirectory(title="Choose the output folder")
        if d:
            self.out_dir.set(d)

    def _clear_log(self):
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")

    def _append_log(self, text: str):
        box = self.log_box
        box.configure(state="normal")
        box.insert("end", text + "\n")
        try:
            if int(box.index("end-1c").split(".")[0]) > 4000:
                box.delete("1.0", "1000.0")
        except Exception:
            pass
        box.see("end")
        box.configure(state="disabled")

    def _q(self, kind, payload):
        self.msg_queue.put((kind, payload))

    # -- queue pump -----------------------------------------------------------

    def _poll(self):
        try:
            while True:
                kind, payload = self.msg_queue.get_nowait()
                if kind == "log":
                    self._append_log(payload)
                elif kind == "progress":
                    done, total = payload
                    self.progress.configure(maximum=total, value=done)
                    self.status.set(f"{done} / {total}")
                elif kind == "done":
                    self._finish(*payload)
        except queue.Empty:
            pass
        self.after(self.POLL_MS, self._poll)

    # -- job control ----------------------------------------------------------

    def _start(self):
        if self.worker and self.worker.is_alive():
            return

        in_dir = self.in_dir.get().strip()
        if not os.path.isdir(in_dir):
            messagebox.showerror("Bad input folder",
                                 "Please choose a valid input folder.")
            return

        overwrite = self.overwrite.get()
        out_dir = self.out_dir.get().strip()

        if not overwrite:
            if not out_dir:
                messagebox.showerror(
                    "No output folder",
                    "Choose an output folder or switch to overwrite mode.")
                return
            if not os.path.isdir(out_dir):
                try:
                    os.makedirs(out_dir, exist_ok=True)
                except OSError as exc:
                    messagebox.showerror("Bad output folder", str(exc))
                    return
            if os.path.abspath(out_dir) == os.path.abspath(in_dir):
                if not messagebox.askyesno(
                        "Same folder",
                        "The output folder is the same as the input folder.\n"
                        "Files will be overwritten in place. Continue?"):
                    return
                overwrite = True

        # parse extensions
        exts = set()
        for tok in self.exts.get().replace(",", " ").replace(";", " ").split():
            tok = tok.strip().lower()
            if not tok:
                continue
            if not tok.startswith("."):
                tok = "." + tok
            exts.add(tok)
        if not exts:
            messagebox.showerror("No file types",
                                 "List at least one extension, e.g. .png")
            return

        files = collect_files(in_dir, exts, self.recursive.get())
        if not files:
            messagebox.showinfo("Nothing to do",
                                "No matching files were found.")
            return

        out_map = {}
        for p in files:
            if overwrite:
                out_map[p] = p
            else:
                rel = os.path.relpath(p, in_dir)
                out_map[p] = os.path.join(out_dir, rel)

        self.stop_event.clear()
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.progress.configure(maximum=len(files), value=0)
        self.status.set(f"0 / {len(files)}")
        self._append_log(f"--- queued {len(files)} file(s) ---")

        self.worker = threading.Thread(
            target=self._work,
            args=(files, out_map, in_dir, self.threshold.get(),
                  overwrite, self.backup.get()),
            daemon=True,
        )
        self.worker.start()

    def _stop(self):
        self.stop_event.set()
        self.status.set("Stopping…")

    def _finish(self, fixed, clean, errors):
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.status.set(f"Done — {fixed} fixed, {clean} clean, {errors} errors")
        self._append_log(
            f"--- done: {fixed} fixed, {clean} clean/unchanged, "
            f"{errors} error(s) ---")

    # -- worker ---------------------------------------------------------------

    def _work(self, files, out_map, base, threshold, overwrite, backup):
        fixed = clean = errors = 0
        total = len(files)

        for i, src in enumerate(files, 1):
            if self.stop_event.is_set():
                self._q("log", "*** stopped by user ***")
                break

            try:
                tag, msg = self._process_one(src, out_map[src], base,
                                             threshold, backup)
            except Exception as exc:
                errors += 1
                self._q("log", f"[ERROR] {os.path.relpath(src, base)}: {exc}")
            else:
                if tag == "fixed":
                    fixed += 1
                    self._q("log", f"[fixed] {msg}")
                elif tag == "copied":
                    clean += 1
                    self._q("log", f"[copy ] {msg}")
                elif tag == "clean":
                    clean += 1
                    self._q("log", f"[clean] {msg}")
                else:
                    self._q("log", f"[skip ] {msg}")

            self._q("progress", (i, total))

        self._q("done", (fixed, clean, errors))

    def _process_one(self, src, dst, base, threshold, backup):
        """Return (tag, human_message).  tag in {fixed, copied, clean, skip}."""
        name = os.path.relpath(src, base)
        same_file = os.path.abspath(src) == os.path.abspath(dst)

        with Image.open(src) as im:
            im.load()
            has_alpha = ("A" in im.getbands()) or ("transparency" in im.info)
            if not has_alpha:
                return "skip", f"{name}: no alpha channel"
            fixed, changed = fix_transparent_pixels(im, threshold)

        if same_file:
            if changed == 0:
                return "clean", f"{name}: already clean"
            if backup:
                bak = src + ".bak"
                if not os.path.exists(bak):
                    shutil.copy2(src, bak)
            fixed.save(dst)
            return "fixed", f"{name}: zeroed {changed} px"

        # writing to a mirror folder
        parent = os.path.dirname(dst)
        if parent:
            os.makedirs(parent, exist_ok=True)
        fixed.save(dst)
        if changed:
            return "fixed", f"{name}: zeroed {changed} px"
        return "copied", f"{name}: already clean (copied)"


# ----------------------------------------------------------------------------

def main():
    app = FixerApp()
    app.mainloop()


if __name__ == "__main__":
    main()