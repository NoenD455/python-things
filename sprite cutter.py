import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser
from PIL import Image, ImageTk


class SpriteSheetCutter(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Spritesheet Cutter")
        self.geometry("1200x780")
        self.minsize(900, 600)

        # ---- state ----
        self.image = None            # PIL RGBA image
        self.image_path = None
        self.tk_img = None
        self.scale = 1.0
        self.ox = 0
        self.oy = 0
        self.bg_color = (255, 0, 255)
        self.region_boxes = []       # preview boxes for bg mode

        # ---- variables ----
        self.out_dir = tk.StringVar(value=os.path.join(os.getcwd(), "output"))
        self.mode = tk.StringVar(value="bg")
        self.tolerance = tk.IntVar(value=0)
        self.min_pixels = tk.IntVar(value=1)
        self.conn8 = tk.BooleanVar(value=False)

        self.gw = tk.IntVar(value=16)
        self.gh = tk.IntVar(value=16)
        self.gxs = tk.IntVar(value=0)
        self.gys = tk.IntVar(value=0)
        self.gx0 = tk.IntVar(value=0)
        self.gy0 = tk.IntVar(value=0)

        self.tw = tk.IntVar(value=16)
        self.th = tk.IntVar(value=16)

        self._build_ui()
        self._bind_traces()
        self.on_mode_change()

    # ================================================================== UI
    def _build_ui(self):
        top = ttk.Frame(self, padding=(8, 6))
        top.pack(fill="x")

        ttk.Button(top, text="Load Spritesheet…", command=self.load_image).pack(side="left")
        self.lbl_file = ttk.Label(top, text="(no image loaded)")
        self.lbl_file.pack(side="left", padx=10)

        ttk.Button(top, text="Browse…", command=self.choose_out).pack(side="right")
        ttk.Entry(top, textvariable=self.out_dir).pack(side="right", fill="x", expand=True, padx=6)
        ttk.Label(top, text="Output folder:").pack(side="right")

        main = ttk.Frame(self, padding=(8, 0, 8, 0))
        main.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(main, bg="#1e1e1e", highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Configure>", lambda e: self.redraw())

        side = ttk.Frame(main, padding=(10, 0, 0, 0), width=340)
        side.pack(side="right", fill="y")
        side.pack_propagate(False)

        mode_box = ttk.LabelFrame(side, text="Mode", padding=8)
        mode_box.pack(fill="x")
        for text, val in (("By background color", "bg"),
                          ("By sprite size + spacing", "grid"),
                          ("By tile grid", "tile")):
            ttk.Radiobutton(mode_box, text=text, value=val, variable=self.mode,
                            command=self.on_mode_change).pack(anchor="w")

        self.opt_holder = ttk.Frame(side)
        self.opt_holder.pack(fill="both", expand=True, pady=(8, 0))

        self.frame_bg = self._build_bg_opts()
        self.frame_grid = self._build_grid_opts()
        self.frame_tile = self._build_tile_opts()

        ttk.Button(side, text="Extract Sprites", command=self.extract).pack(
            side="bottom", fill="x", pady=8)

        self.status = ttk.Label(self, text="Ready", anchor="w", padding=(10, 4))
        self.status.pack(fill="x", side="bottom")

    def _build_bg_opts(self):
        f = ttk.LabelFrame(self.opt_holder, text="Background color", padding=8)

        row = ttk.Frame(f)
        row.pack(fill="x")
        self.swatch = tk.Canvas(row, width=44, height=44, highlightthickness=1,
                                highlightbackground="#888")
        self.swatch.pack(side="left")
        self.lbl_color = ttk.Label(row, text="#FF00FF")
        self.lbl_color.pack(side="left", padx=8)
        ttk.Button(row, text="Choose…", command=self.pick_color).pack(side="right")

        ttk.Label(f, text="Tip: click the preview to sample a pixel.",
                  foreground="#777").pack(anchor="w", pady=(6, 0))

        row2 = ttk.Frame(f)
        row2.pack(fill="x", pady=(8, 0))
        ttk.Label(row2, text="Tolerance").pack(side="left")
        ttk.Spinbox(row2, from_=0, to=255, textvariable=self.tolerance, width=6).pack(side="left", padx=6)

        row3 = ttk.Frame(f)
        row3.pack(fill="x", pady=(4, 0))
        ttk.Label(row3, text="Min pixels").pack(side="left")
        ttk.Spinbox(row3, from_=1, to=100000, textvariable=self.min_pixels, width=6).pack(side="left", padx=6)

        ttk.Checkbutton(f, text="8-connected regions", variable=self.conn8).pack(anchor="w", pady=(6, 0))
        ttk.Button(f, text="Preview regions", command=self.preview_regions).pack(fill="x", pady=(8, 0))

        self.update_swatch()
        return f

    def _build_grid_opts(self):
        f = ttk.LabelFrame(self.opt_holder, text="Sprite size + spacing", padding=8)

        def add_row(r, label, var):
            ttk.Label(f, text=label).grid(row=r, column=0, sticky="w", pady=2)
            ttk.Spinbox(f, from_=0, to=100000, textvariable=var, width=8).grid(
                row=r, column=1, sticky="w", padx=6)

        add_row(0, "Width (px)", self.gw)
        add_row(1, "Height (px)", self.gh)
        add_row(2, "X spacing", self.gxs)
        add_row(3, "Y spacing", self.gys)
        add_row(4, "Start X", self.gx0)
        add_row(5, "Start Y", self.gy0)

        ttk.Label(f, text="Tip: click the preview to set Start X / Y.",
                  foreground="#777", wraplength=280).grid(
            row=6, column=0, columnspan=2, sticky="w", pady=(6, 0))
        return f

    def _build_tile_opts(self):
        f = ttk.LabelFrame(self.opt_holder, text="Tile grid", padding=8)

        def add_row(r, label, var):
            ttk.Label(f, text=label).grid(row=r, column=0, sticky="w", pady=2)
            ttk.Spinbox(f, from_=1, to=100000, textvariable=var, width=8).grid(
                row=r, column=1, sticky="w", padx=6)

        add_row(0, "Tile width (px)", self.tw)
        add_row(1, "Tile height (px)", self.th)

        ttk.Label(f, text="Every full tile is cut (leftover edges are ignored).",
                  foreground="#777", wraplength=280).grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(6, 0))
        return f

    def _bind_traces(self):
        for v in (self.gw, self.gh, self.gxs, self.gys, self.gx0, self.gy0,
                  self.tw, self.th):
            v.trace_add("write", self._on_var_change)
        for v in (self.tolerance, self.conn8):
            v.trace_add("write", self._on_region_var_change)

    def _on_var_change(self, *a):
        self.redraw()

    def _on_region_var_change(self, *a):
        self.region_boxes = []
        self.redraw()

    def on_mode_change(self):
        for f in (self.frame_bg, self.frame_grid, self.frame_tile):
            f.pack_forget()
        {"bg": self.frame_bg,
         "grid": self.frame_grid,
         "tile": self.frame_tile}[self.mode.get()].pack(fill="x")
        self.redraw()

    def update_status(self, text):
        self.status.config(text=text)

    # ============================================================ rendering
    def redraw(self):
        self.canvas.delete("all")
        if self.image is None:
            self.update_status("Ready")
            return
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw < 20 or ch < 20:
            return
        iw, ih = self.image.size
        self.scale = min(cw / iw, ch / ih)
        sw = max(1, int(iw * self.scale))
        sh = max(1, int(ih * self.scale))
        self.ox = (cw - sw) // 2
        self.oy = (ch - sh) // 2

        self.tk_img = ImageTk.PhotoImage(self.image.resize((sw, sh), Image.NEAREST))
        self.canvas.create_image(self.ox, self.oy, anchor="nw", image=self.tk_img)

        mode = self.mode.get()
        if mode == "bg":
            self._draw_bg_overlay()
        elif mode == "grid":
            self._draw_grid_overlay()
        else:
            self._draw_tile_overlay()

    def _to_canvas(self, x, y):
        return self.ox + x * self.scale, self.oy + y * self.scale

    def _draw_bg_overlay(self):
        for (x0, y0, x1, y1, area) in self.region_boxes:
            cx0, cy0 = self._to_canvas(x0, y0)
            cx1, cy1 = self._to_canvas(x1, y1)
            self.canvas.create_rectangle(cx0, cy0, cx1, cy1, outline="#00e0ff")
        if self.region_boxes:
            self.update_status(f"{len(self.region_boxes)} region(s) detected "
                               f"(click preview to re-sample background)")

    def _draw_grid_overlay(self):
        iw, ih = self.image.size
        w, h = self.gw.get(), self.gh.get()
        xs, ys = self.gxs.get(), self.gys.get()
        x0, y0 = self.gx0.get(), self.gy0.get()
        if w <= 0 or h <= 0 or (w + xs) <= 0 or (h + ys) <= 0:
            self.update_status("Invalid sprite size / spacing")
            return
        n = 0
        for y in range(y0, ih - h + 1, h + ys):
            for x in range(x0, iw - w + 1, w + xs):
                cx0, cy0 = self._to_canvas(x, y)
                cx1, cy1 = self._to_canvas(x + w, y + h)
                self.canvas.create_rectangle(cx0, cy0, cx1, cy1, outline="#ff3b3b")
                n += 1
        self.update_status(f"{n} sprite(s) will be cut")

    def _draw_tile_overlay(self):
        iw, ih = self.image.size
        tw, th = self.tw.get(), self.th.get()
        if tw <= 0 or th <= 0:
            self.update_status("Invalid tile size")
            return
        cols, rows = iw // tw, ih // th
        x_end, y_end = self._to_canvas(cols * tw, rows * th)
        for i in range(cols + 1):
            cx, _ = self._to_canvas(i * tw, 0)
            self.canvas.create_line(cx, self.oy, cx, y_end, fill="#39ff6a")
        for j in range(rows + 1):
            _, cy = self._to_canvas(0, j * th)
            self.canvas.create_line(self.ox, cy, x_end, cy, fill="#39ff6a")
        self.update_status(f"{cols * rows} tile(s) ({cols} x {rows})")

    # ============================================================== events
    def on_canvas_click(self, event):
        if self.image is None:
            return
        ix = int((event.x - self.ox) / self.scale)
        iy = int((event.y - self.oy) / self.scale)
        iw, ih = self.image.size
        if not (0 <= ix < iw and 0 <= iy < ih):
            return
        mode = self.mode.get()
        if mode == "bg":
            r, g, b, a = self.image.getpixel((ix, iy))
            if a == 0:
                r, g, b = self.bg_color
            self.bg_color = (r, g, b)
            self.region_boxes = []
            self.update_swatch()
            self.redraw()
        elif mode == "grid":
            self.gx0.set(ix)
            self.gy0.set(iy)

    def load_image(self):
        path = filedialog.askopenfilename(
            title="Select spritesheet",
            filetypes=[("Images", "*.png *.gif *.bmp *.jpg *.jpeg *.webp *.tga"),
                       ("All files", "*.*")])
        if not path:
            return
        try:
            img = Image.open(path).convert("RGBA")
        except Exception as e:
            messagebox.showerror("Error", f"Could not open image:\n{e}")
            return
        self.image = img
        self.image_path = path
        self.region_boxes = []
        self.lbl_file.config(text=f"{os.path.basename(path)}  ({img.width}x{img.height})")
        self.redraw()

    def choose_out(self):
        d = filedialog.askdirectory(title="Choose output folder")
        if d:
            self.out_dir.set(d)

    # ---------------------------------------------------------- colors
    def hex_color(self):
        return "#%02x%02x%02x" % self.bg_color

    def update_swatch(self):
        self.swatch.configure(bg=self.hex_color())
        self.lbl_color.config(text=self.hex_color().upper())

    def pick_color(self):
        rgb, _ = colorchooser.askcolor(color=self.hex_color(),
                                       title="Pick background color")
        if rgb:
            self.bg_color = tuple(int(c) for c in rgb)
            self.region_boxes = []
            self.update_swatch()
            self.redraw()

    # =========================================================== detection
    def _build_mask(self):
        """1 = background pixel (matches bg color within tolerance, or alpha 0)."""
        img = self.image
        w, h = img.size
        data = img.tobytes()          # RGBA bytes
        tol = max(0, self.tolerance.get())
        br, bg_, bb = self.bg_color
        mask = bytearray(w * h)
        for i in range(w * h):
            o = i * 4
            if data[o + 3] == 0:
                mask[i] = 1
            elif (abs(data[o] - br) <= tol and
                  abs(data[o + 1] - bg_) <= tol and
                  abs(data[o + 2] - bb) <= tol):
                mask[i] = 1
        return mask

    def find_regions(self, mask, w, h):
        if self.conn8.get():
            nbrs = ((-1, -1), (0, -1), (1, -1), (-1, 0),
                    (1, 0), (-1, 1), (0, 1), (1, 1))
        else:
            nbrs = ((0, -1), (-1, 0), (1, 0), (0, 1))

        seen = bytearray(w * h)
        regions = []
        for y in range(h):
            base = y * w
            for x in range(w):
                i = base + x
                if seen[i] or mask[i]:
                    continue
                seen[i] = 1
                stack = [i]
                minx = maxx = x
                miny = maxy = y
                area = 0
                while stack:
                    ci = stack.pop()
                    area += 1
                    cy, cx = divmod(ci, w)
                    if cx < minx: minx = cx
                    if cx > maxx: maxx = cx
                    if cy < miny: miny = cy
                    if cy > maxy: maxy = cy
                    for dx, dy in nbrs:
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < w and 0 <= ny < h:
                            ni = ny * w + nx
                            if not seen[ni] and not mask[ni]:
                                seen[ni] = 1
                                stack.append(ni)
                regions.append((minx, miny, maxx + 1, maxy + 1, area))
        return regions

    def preview_regions(self):
        if self.image is None:
            return
        w, h = self.image.size
        mask = self._build_mask()
        regions = self.find_regions(mask, w, h)
        minp = max(1, self.min_pixels.get())
        self.region_boxes = [r for r in regions if r[4] >= minp]
        self.redraw()

    # =========================================================== extraction
    def extract(self):
        if self.image is None:
            messagebox.showwarning("No image", "Load a spritesheet first.")
            return
        out = self.out_dir.get().strip()
        if not out:
            messagebox.showwarning("No output", "Choose an output folder.")
            return
        try:
            os.makedirs(out, exist_ok=True)
        except Exception as e:
            messagebox.showerror("Error", f"Cannot create output folder:\n{e}")
            return

        mode = self.mode.get()
        try:
            if mode == "bg":
                n = self.extract_bg(out)
            elif mode == "grid":
                n = self.extract_grid(out)
            else:
                n = self.extract_tile(out)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        self.update_status(f"Extracted {n} image(s) → {out}")
        messagebox.showinfo("Done", f"Extracted {n} image(s) to:\n{out}")

    def extract_bg(self, out):
        img = self.image
        w, h = img.size
        mask = self._build_mask()
        minp = max(1, self.min_pixels.get())
        regions = [r for r in self.find_regions(mask, w, h) if r[4] >= minp]
        if not regions:
            raise ValueError("No sprites found.\nTry a different background color or a higher tolerance.")
        regions.sort(key=lambda r: (r[1], r[0]))   # raster order

        # Clean image: every background pixel becomes fully transparent
        data = bytearray(img.tobytes())
        for i in range(w * h):
            if mask[i]:
                o = i * 4
                data[o] = data[o + 1] = data[o + 2] = data[o + 3] = 0
        clean = Image.frombytes("RGBA", (w, h), bytes(data))

        for n, (x0, y0, x1, y1, _area) in enumerate(regions):
            clean.crop((x0, y0, x1, y1)).save(
                os.path.join(out, f"sprite_{n:03d}.png"))

        self.region_boxes = regions
        self.redraw()
        return len(regions)

    def extract_grid(self, out):
        img = self.image
        w, h = img.size
        sw, sh = self.gw.get(), self.gh.get()
        xs, ys = self.gxs.get(), self.gys.get()
        x0, y0 = self.gx0.get(), self.gy0.get()
        if sw <= 0 or sh <= 0 or (sw + xs) <= 0 or (sh + ys) <= 0:
            raise ValueError("Invalid sprite size / spacing.")

        n = 0
        for y in range(y0, h - sh + 1, sh + ys):
            for x in range(x0, w - sw + 1, sw + xs):
                img.crop((x, y, x + sw, y + sh)).save(
                    os.path.join(out, f"sprite_{n:03d}.png"))
                n += 1
        if n == 0:
            raise ValueError("No sprites matched the given size / start position.")
        return n

    def extract_tile(self, out):
        img = self.image
        w, h = img.size
        tw, th = self.tw.get(), self.th.get()
        if tw <= 0 or th <= 0:
            raise ValueError("Invalid tile size.")

        n = 0
        for y in range(0, h - th + 1, th):
            for x in range(0, w - tw + 1, tw):
                img.crop((x, y, x + tw, y + th)).save(
                    os.path.join(out, f"tile_{n:03d}.png"))
                n += 1
        if n == 0:
            raise ValueError("Image is smaller than one tile.")
        return n


if __name__ == "__main__":
    SpriteSheetCutter().mainloop()