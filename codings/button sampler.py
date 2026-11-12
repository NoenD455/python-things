"""
Video Button Press Extractor
----------------------------
- Load a video
- Add named buttons, click on the video to set their position
- Scrub to a frame where the button is PRESSED   -> "Sample Pressed Color"
- Scrub to a frame where the button is RELEASED  -> "Sample Unpressed Color"
- Set a tolerance (% of the max possible RGB distance)
- Process & export CSV:  timestamp,buttonspressed
     e.g.  1.233,"A|B"     (empty string when nothing is pressed)

Requires:  pip install opencv-python pillow numpy
"""

import os
import csv
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog

import cv2
import numpy as np
from PIL import Image, ImageTk

# Maximum possible Euclidean distance in RGB space (0,0,0) <-> (255,255,255)
MAX_RGB_DIST = (255.0 ** 2 * 3) ** 0.5  # ~441.67


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def color_dist(c1, c2):
    return float(np.sqrt((int(c1[0]) - int(c2[0])) ** 2 +
                         (int(c1[1]) - int(c2[1])) ** 2 +
                         (int(c1[2]) - int(c2[2])) ** 2))


def sample_patch(frame, x, y, radius):
    """Average BGR color of a (2r+1)x(2r+1) patch centred on (x, y)."""
    if frame is None:
        return None
    h, w = frame.shape[:2]
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return None
    patch = frame[y0:y1, x0:x1].reshape(-1, 3).astype(np.float32)
    m = patch.mean(axis=0)
    return (int(round(m[0])), int(round(m[1])), int(round(m[2])))


class ButtonDef:
    __slots__ = ("name", "x", "y", "pressed_color", "unpressed_color", "tolerance")

    def __init__(self, name):
        self.name = name
        self.x = None
        self.y = None
        self.pressed_color = None      # (B, G, R)
        self.unpressed_color = None    # (B, G, R)
        self.tolerance = 10.0          # percent

    def ready(self):
        return (self.x is not None and
                self.pressed_color is not None and
                self.unpressed_color is not None)


def detect_button(b, frame, radius):
    """True if the button looks pressed on `frame`."""
    if not b.ready() or frame is None:
        return False
    px = sample_patch(frame, b.x, b.y, radius)
    if px is None:
        return False
    d_pressed = color_dist(px, b.pressed_color)
    d_unpressed = color_dist(px, b.unpressed_color)
    tol = (b.tolerance / 100.0) * MAX_RGB_DIST
    # Confident match to the pressed colour and closer to it than to the
    # released colour -> pressed.
    return d_pressed <= tol and d_pressed <= d_unpressed


# --------------------------------------------------------------------------- #
#  Application
# --------------------------------------------------------------------------- #
class App:
    def __init__(self, root):
        self.root = root
        root.title("Video Button Press Extractor")
        root.geometry("1250x780")

        self.cap = None
        self.video_path = None
        self.fps = 30.0
        self.frame_count = 0
        self.current_idx = 0
        self.current_frame = None

        self.buttons = []
        self.mode = None            # 'position'
        self.pending = None         # ButtonDef awaiting a click
        self.sample_radius = 1

        self.display_scale = 1.0
        self.display_off = (0, 0)
        self._photo = None
        self._render_job = None
        self._pending_idx = 0

        self._build_ui()
        self._bind_keys()

    # ------------------------------------------------------------------ UI --
    def _build_ui(self):
        self.root.rowconfigure(1, weight=1)
        self.root.columnconfigure(0, weight=1)

        # ---- top bar
        top = ttk.Frame(self.root, padding=(8, 8, 8, 0))
        top.grid(row=0, column=0, sticky="ew")
        ttk.Button(top, text="Load Video…", command=self.load_video).pack(side="left")
        self.info = ttk.Label(top, text="No video loaded")
        self.info.pack(side="left", padx=12)

        # ---- main area
        main = ttk.Frame(self.root, padding=8)
        main.grid(row=1, column=0, sticky="nsew")
        main.rowconfigure(0, weight=1)
        main.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(main, bg="#101010", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Configure>", lambda e: self.redraw())

        panel = ttk.Frame(main, padding=(10, 0, 0, 0))
        panel.grid(row=0, column=1, sticky="ns")

        ttk.Label(panel, text="Buttons", font=("", 10, "bold")).pack(anchor="w")

        tree_frame = ttk.Frame(panel)
        tree_frame.pack(fill="both", expand=True, pady=(2, 0))

        cols = ("name", "pos", "pressed", "unpressed", "tol")
        heads = (("name", "Name", 110), ("pos", "Position", 78),
                 ("pressed", "Pressed", 78), ("unpressed", "Unpressed", 84),
                 ("tol", "Tol %", 52))
        self.tree = ttk.Treeview(tree_frame, columns=cols, show="headings",
                                 height=15, selectmode="browse")
        for c, t, w in heads:
            self.tree.heading(c, text=t)
            self.tree.column(c, width=w, anchor="center", stretch=False)
        vs = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vs.set)
        self.tree.pack(side="left", fill="both", expand=True)
        vs.pack(side="right", fill="y")
        self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)

        row = ttk.Frame(panel); row.pack(fill="x", pady=(8, 0))
        ttk.Button(row, text="Add Button", command=self.add_button).pack(side="left")
        ttk.Button(row, text="Remove", command=self.remove_button).pack(side="left", padx=6)

        ttk.Button(panel, text="Set Position  (then click video)",
                   command=self.set_position).pack(fill="x", pady=(6, 0))

        row = ttk.Frame(panel); row.pack(fill="x", pady=(6, 0))
        ttk.Button(row, text="Sample Pressed Color",
                   command=lambda: self.sample_color(True)).pack(side="left")
        ttk.Button(row, text="Sample Unpressed",
                   command=lambda: self.sample_color(False)).pack(side="left", padx=6)

        row = ttk.Frame(panel); row.pack(fill="x", pady=(8, 0))
        ttk.Label(row, text="Tolerance %:").pack(side="left")
        self.tol_var = tk.DoubleVar(value=10.0)
        ttk.Spinbox(row, from_=0, to=100, increment=1, width=6,
                    textvariable=self.tol_var).pack(side="left", padx=6)
        ttk.Button(row, text="Apply", command=self.apply_tolerance).pack(side="left")

        row = ttk.Frame(panel); row.pack(fill="x", pady=(6, 0))
        ttk.Label(row, text="Sample radius (px):").pack(side="left")
        self.radius_var = tk.IntVar(value=1)
        sp = ttk.Spinbox(row, from_=0, to=25, increment=1, width=6,
                         textvariable=self.radius_var, command=self.on_radius)
        sp.pack(side="left", padx=6)
        sp.bind("<KeyRelease>", lambda e: self.on_radius())

        self.status = ttk.Label(panel, text="", foreground="#008800", wraplength=300)
        self.status.pack(fill="x", pady=(10, 0))

        # ---- bottom bar
        bottom = ttk.Frame(self.root, padding=8)
        bottom.grid(row=2, column=0, sticky="ew")
        bottom.columnconfigure(1, weight=1)

        self.frame_lbl = ttk.Label(bottom, text="Frame - / -", width=34)
        self.frame_lbl.grid(row=0, column=0, sticky="w")

        self.slider = ttk.Scale(bottom, from_=0, to=100, orient="horizontal",
                                command=self.on_slider)
        self.slider.grid(row=0, column=1, sticky="ew", padx=10)

        self.process_btn = ttk.Button(bottom, text="Process && Export CSV",
                                      command=self.process_video)
        self.process_btn.grid(row=0, column=2, sticky="e")

        self.progress = ttk.Progressbar(bottom, mode="determinate")
        self.progress.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(6, 0))

    def _bind_keys(self):
        self.root.bind("<Left>", lambda e: self.step(-1))
        self.root.bind("<Right>", lambda e: self.step(1))
        self.root.bind("<Shift-Left>", lambda e: self.step(-10))
        self.root.bind("<Shift-Right>", lambda e: self.step(10))

    # -------------------------------------------------------------- video --
    def load_video(self):
        path = filedialog.askopenfilename(
            title="Select a video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.webm *.m4v"),
                       ("All files", "*.*")])
        if not path:
            return
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open the video file.")
            return
        if self.cap is not None:
            self.cap.release()

        self.cap = cap
        self.video_path = path
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if self.fps <= 0:
            self.fps = 30.0
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        dur = self.frame_count / self.fps if self.fps else 0

        self.info.configure(
            text=f"{os.path.basename(path)}  —  {w}×{h} @ {self.fps:.2f} fps, "
                 f"{self.frame_count} frames ({dur:.2f}s)")
        self.slider.configure(to=max(0, self.frame_count - 1))
        self._pending_idx = 0
        self.show_frame(0)

    def show_frame(self, idx):
        if self.cap is None or self.frame_count <= 0:
            return
        idx = int(max(0, min(idx, self.frame_count - 1)))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        if not ok:
            return
        self.current_frame = frame
        self.current_idx = idx
        self.frame_lbl.configure(
            text=f"Frame {idx} / {self.frame_count - 1}    t = {idx / self.fps:.3f}s")
        self.redraw()

    # ------------------------------------------------------------ drawing --
    def redraw(self):
        if self.current_frame is None:
            return
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            return

        disp = self.current_frame.copy()
        r = max(1, self.sample_radius)
        for b in self.buttons:
            if b.x is None:
                continue
            if not b.ready():
                col = (150, 150, 150)
            else:
                col = (0, 200, 0) if detect_button(b, self.current_frame,
                                                   self.sample_radius) else (0, 0, 220)
            cv2.rectangle(disp, (b.x - r, b.y - r), (b.x + r, b.y + r), col, 2)
            cv2.drawMarker(disp, (b.x, b.y), col, cv2.MARKER_CROSS, 7, 1)
            cv2.putText(disp, b.name, (b.x + r + 4, b.y - r - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)

        h, w = disp.shape[:2]
        scale = min(cw / w, ch / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        out = cv2.resize(disp, (nw, nh), interpolation=interp)
        rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.delete("all")
        ox, oy = (cw - nw) // 2, (ch - nh) // 2
        self.canvas.create_image(ox, oy, anchor="nw", image=self._photo)

        self.display_scale = scale
        self.display_off = (ox, oy)

    def canvas_to_frame(self, cx, cy):
        if self.current_frame is None or self.display_scale <= 0:
            return None
        ox, oy = self.display_off
        fx = (cx - ox) / self.display_scale
        fy = (cy - oy) / self.display_scale
        h, w = self.current_frame.shape[:2]
        if 0 <= fx < w and 0 <= fy < h:
            return int(fx), int(fy)
        return None

    # ------------------------------------------------------------- slider --
    def on_slider(self, val):
        self._pending_idx = int(float(val))
        if self._render_job is None:
            self._render_job = self.root.after(30, self._flush_slider)

    def _flush_slider(self):
        self._render_job = None
        self.show_frame(self._pending_idx)

    def step(self, d):
        if self.cap is None:
            return
        base = self._pending_idx if self._render_job is not None else self.current_idx
        new = max(0, min(base + d, self.frame_count - 1))
        self._pending_idx = new
        self.slider.set(new)          # triggers debounced show_frame

    # -------------------------------------------------------------- table --
    def selected_index(self):
        sel = self.tree.selection()
        return int(sel[0]) if sel else None

    def selected_button(self):
        i = self.selected_index()
        if i is None or i >= len(self.buttons):
            return None
        return self.buttons[i]

    @staticmethod
    def _col_str(c):
        if c is None:
            return "—"
        return "#{:02X}{:02X}{:02X}".format(c[2], c[1], c[0])   # BGR -> RGB

    def refresh_tree(self, keep="__current__"):
        if keep == "__current__":
            keep = self.selected_index()
        self.tree.delete(*self.tree.get_children())
        for i, b in enumerate(self.buttons):
            self.tree.insert("", "end", iid=str(i), values=(
                b.name,
                f"{b.x},{b.y}" if b.x is not None else "—",
                self._col_str(b.pressed_color),
                self._col_str(b.unpressed_color),
                f"{b.tolerance:g}",
            ))
        if keep is not None and 0 <= keep < len(self.buttons):
            self.tree.selection_set(str(keep))
            self.tree.focus(str(keep))

    def on_tree_select(self, _evt=None):
        b = self.selected_button()
        if b is not None:
            self.tol_var.set(b.tolerance)

    # ------------------------------------------------------------ actions --
    def add_button(self):
        name = simpledialog.askstring("New Button", "Name for the new button:",
                                      parent=self.root)
        if not name:
            return
        name = name.strip().replace("|", "_")
        if not name:
            return
        if any(b.name == name for b in self.buttons):
            messagebox.showwarning("Duplicate",
                                   f"A button named '{name}' already exists.")
            return
        self.buttons.append(ButtonDef(name))
        self.refresh_tree(keep=len(self.buttons) - 1)
        self.redraw()

    def remove_button(self):
        i = self.selected_index()
        if i is None:
            return
        del self.buttons[i]
        self.refresh_tree(keep=None)
        self.redraw()

    def set_position(self):
        b = self.selected_button()
        if b is None:
            messagebox.showinfo("Select a button",
                                "Select a button in the list first.")
            return
        if self.current_frame is None:
            messagebox.showinfo("No video", "Load a video first.")
            return
        self.mode = "position"
        self.pending = b
        self.canvas.configure(cursor="crosshair")
        self.status.configure(
            text=f"Click on the video to place the sample point for '{b.name}'.")

    def on_canvas_click(self, ev):
        if self.mode != "position" or self.pending is None:
            return
        pos = self.canvas_to_frame(ev.x, ev.y)
        if pos is None:
            return
        self.pending.x, self.pending.y = pos
        self.mode = None
        self.pending = None
        self.canvas.configure(cursor="")
        self.status.configure(text="")
        self.refresh_tree()
        self.redraw()

    def sample_color(self, pressed):
        b = self.selected_button()
        if b is None:
            messagebox.showinfo("Select a button",
                                "Select a button in the list first.")
            return
        if b.x is None:
            messagebox.showinfo("No position", "Set the button position first.")
            return
        if self.current_frame is None:
            return
        col = sample_patch(self.current_frame, b.x, b.y, self.sample_radius)
        if col is None:
            return
        if pressed:
            b.pressed_color = col
        else:
            b.unpressed_color = col
        self.refresh_tree()
        self.redraw()

    def apply_tolerance(self):
        b = self.selected_button()
        if b is None:
            return
        try:
            v = float(self.tol_var.get())
        except Exception:
            return
        b.tolerance = max(0.0, min(100.0, v))
        self.refresh_tree()

    def on_radius(self):
        try:
            self.sample_radius = max(0, int(self.radius_var.get()))
        except Exception:
            return
        self.redraw()

    # ------------------------------------------------------------ process --
    def process_video(self):
        if self.video_path is None:
            messagebox.showinfo("No video", "Load a video first.")
            return

        ready = [b for b in self.buttons if b.ready()]
        if not ready:
            messagebox.showinfo(
                "No buttons",
                "Define at least one button (position + pressed & unpressed colours).")
            return

        skipped = [b.name for b in self.buttons if not b.ready()]
        if skipped:
            if not messagebox.askyesno(
                    "Incomplete buttons",
                    "These buttons are incomplete and will be ignored:\n  " +
                    ", ".join(skipped) + "\n\nContinue?"):
                return

        out = filedialog.asksaveasfilename(
            title="Save CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile=os.path.splitext(os.path.basename(self.video_path))[0]
                        + "_buttons.csv")
        if not out:
            return

        self.process_btn.configure(state="disabled")
        self.status.configure(text="")
        total = max(1, self.frame_count)
        self.progress.configure(maximum=total, value=0)
        self.root.update_idletasks()

        rows = []
        idx = 0
        cap = cv2.VideoCapture(self.video_path)
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                names = [b.name for b in ready
                         if detect_button(b, frame, self.sample_radius)]
                rows.append(("{:.3f}".format(idx / self.fps), "|".join(names)))
                idx += 1
                if idx % 15 == 0:
                    self.progress.configure(value=idx)
                    self.root.update_idletasks()
        finally:
            cap.release()
            self.process_btn.configure(state="normal")

        try:
            with open(out, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "buttonspressed"])
                writer.writerows(rows)
        except Exception as e:
            messagebox.showerror("Write error", str(e))
            return

        self.progress.configure(maximum=max(1, idx), value=idx)
        messagebox.showinfo("Done", f"Wrote {len(rows)} rows to:\n{out}")


def main():
    root = tk.Tk()
    try:
        style = ttk.Style()
        for theme in ("vista", "winnative", "aqua", "clam"):
            if theme in style.theme_names():
                style.theme_use(theme)
                break
    except Exception:
        pass
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()