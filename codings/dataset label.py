#!/usr/bin/env python3
"""
Dataset Labeling Tool
---------------------
- Pick a folder of images
- Define classes, each with its own color
- Draw bounding boxes over the images (color-coded per class)
- Export to YOLO format (.txt), rendered images, color masks,
  or a single-folder bundle of masks + masks.csv

Requires: pillow   ->   pip install pillow
Run:      python labeler.py
"""

import os
import csv
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser
from PIL import Image, ImageTk, ImageDraw

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff')

DEFAULT_COLORS = [
    '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4',
    '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff',
    '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1',
]

PROJECT_FILE = '.labeler_project.json'


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def text_color_for(hex_color):
    """Return black or white depending on the brightness of the background."""
    try:
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
    except Exception:
        return 'white'
    return 'black' if (0.299 * r + 0.587 * g + 0.114 * b) > 150 else 'white'


class ScrollFrame(tk.Frame):
    """A vertically scrollable frame."""

    def __init__(self, parent, height=200, width=240):
        super().__init__(parent)
        self.canvas = tk.Canvas(self, highlightthickness=0, height=height, width=width)
        self.sb = ttk.Scrollbar(self, orient='vertical', command=self.canvas.yview)
        self.inner = tk.Frame(self.canvas)
        self._win = self.canvas.create_window((0, 0), window=self.inner, anchor='nw')

        self.inner.bind('<Configure>', self._on_inner)
        self.canvas.bind('<Configure>', self._on_canvas)
        self.canvas.configure(yscrollcommand=self.sb.set)

        self.canvas.pack(side='left', fill='both', expand=True)
        self.sb.pack(side='right', fill='y')

        self.canvas.bind('<Enter>', lambda e: self.canvas.bind_all('<MouseWheel>', self._wheel))
        self.canvas.bind('<Leave>', lambda e: self.canvas.unbind_all('<MouseWheel>'))

    def _on_inner(self, _e):
        self.canvas.configure(scrollregion=self.canvas.bbox('all'))

    def _on_canvas(self, e):
        self.canvas.itemconfig(self._win, width=e.width)

    def _wheel(self, e):
        self.canvas.yview_scroll(-1 * (e.delta // 120), 'units')


# --------------------------------------------------------------------------- #
#  Main application
# --------------------------------------------------------------------------- #
class Labeler(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title('Dataset Labeling Tool')
        self.geometry('1280x820')
        self.minsize(940, 620)

        # ---- state -------------------------------------------------------- #
        self.folder = None
        self.project_file = None
        self.image_paths = []
        self.index = -1
        self.img = None
        self.tkimg = None

        self.classes = []                 # [{'name': str, 'color': '#rrggbb'}]
        self.active_var = tk.IntVar(value=0)
        self.annotations = {}             # basename -> [{'cls','x1','y1','x2','y2'}]

        self.scale = 1.0
        self.ox = self.oy = 0
        self.selected = None
        self.drag_start = None
        self.drag_id = None

        self._build_ui()
        self.bind('<KeyPress>', self.on_key)
        self.protocol('WM_DELETE_WINDOW', self.on_close)

    # ------------------------------------------------------------------ UI #
    def _build_ui(self):
        # ---------- toolbar ----------
        tb = ttk.Frame(self, padding=(6, 5))
        tb.pack(side='top', fill='x')

        ttk.Button(tb, text='📂  Open Folder', command=self.open_folder).pack(side='left')
        ttk.Separator(tb, orient='vertical').pack(side='left', fill='y', padx=8)
        ttk.Button(tb, text='◀  Prev', command=self.prev_image).pack(side='left')
        ttk.Button(tb, text='Next  ▶', command=self.next_image).pack(side='left', padx=(4, 0))
        self.lbl_pos = ttk.Label(tb, text='No folder open')
        self.lbl_pos.pack(side='left', padx=12)
        ttk.Separator(tb, orient='vertical').pack(side='left', fill='y', padx=8)
        ttk.Button(tb, text='💾  Save Project', command=self.save_project).pack(side='left')
        ttk.Separator(tb, orient='vertical').pack(side='left', fill='y', padx=8)
        ttk.Button(tb, text='Export YOLO', command=self.export_yolo).pack(side='left')
        ttk.Button(tb, text='Export Images', command=self.export_images).pack(side='left', padx=4)
        ttk.Button(tb, text='Export Masks', command=self.export_masks).pack(side='left')
        ttk.Button(tb, text='Export CSV+Masks', command=self.export_csv_masks).pack(side='left', padx=4)

        # ---------- body ----------
        body = ttk.Frame(self)
        body.pack(fill='both', expand=True)

        left = ttk.Frame(body, width=280)
        left.pack(side='left', fill='y')
        left.pack_propagate(False)

        ttk.Label(left, text='CLASSES', font=('TkDefaultFont', 9, 'bold')).pack(
            anchor='w', padx=8, pady=(8, 2))
        self.class_scroll = ScrollFrame(left, height=230, width=250)
        self.class_scroll.pack(fill='x', padx=6)

        cbtn = ttk.Frame(left)
        cbtn.pack(fill='x', padx=6, pady=6)
        ttk.Button(cbtn, text='+ Add Class', command=self.add_class).pack(side='left')
        ttk.Label(cbtn, text='  keys 1-9', foreground='#888').pack(side='left')

        ttk.Separator(left).pack(fill='x', pady=4)

        ttk.Label(left, text='IMAGES', font=('TkDefaultFont', 9, 'bold')).pack(
            anchor='w', padx=8, pady=(4, 2))
        self.img_list = tk.Listbox(left, exportselection=False, activestyle='dotbox')
        self.img_list.pack(fill='both', expand=True, padx=6, pady=(0, 6))
        self.img_list.bind('<<ListboxSelect>>', self.on_list_select)

        # ---------- canvas ----------
        self.canvas = tk.Canvas(body, bg='#2b2b2b', highlightthickness=0,
                                cursor='crosshair')
        self.canvas.pack(side='left', fill='both', expand=True)
        self.canvas.bind('<ButtonPress-1>', self.on_press)
        self.canvas.bind('<B1-Motion>', self.on_motion)
        self.canvas.bind('<ButtonRelease-1>', self.on_release)
        self.canvas.bind('<ButtonPress-3>', self.on_right_click)
        self.canvas.bind('<Configure>', lambda e: self.render())

        # ---------- status ----------
        self.status_var = tk.StringVar(
            value='Drag = new box   |   Right-click = delete box   |   '
                  'Shift+Drag = force new box   |   ←/→ = navigate   |   Del = remove selected')
        ttk.Label(self, textvariable=self.status_var, anchor='w',
                  relief='sunken', padding=(6, 3)).pack(side='bottom', fill='x')

    # -------------------------------------------------------------- classes #
    def refresh_classes(self):
        inner = self.class_scroll.inner
        for w in inner.winfo_children():
            w.destroy()

        if not self.classes:
            tk.Label(inner, text='(no classes yet — click Add Class)',
                     fg='#888').pack(anchor='w', padx=6, pady=4)
            return

        for i, c in enumerate(self.classes):
            row = tk.Frame(inner)
            row.pack(fill='x', pady=1)

            tk.Radiobutton(row, variable=self.active_var, value=i,
                           width=1, command=lambda: None).pack(side='left')

            sw = tk.Button(row, bg=c['color'], width=2, relief='raised', bd=1,
                           activebackground=c['color'],
                           command=lambda i=i: self.pick_color(i))
            sw.pack(side='left', padx=(0, 4))

            var = tk.StringVar(value=c['name'])
            var.trace_add('write', lambda *_a, i=i, v=var: self.set_class_name(i, v.get()))
            ent = tk.Entry(row, textvariable=var, width=10)
            ent.pack(side='left', fill='x', expand=True)

            tk.Button(row, text='✕', width=2, bd=0, fg='#a00',
                      command=lambda i=i: self.remove_class(i)).pack(side='left')

    def set_class_name(self, i, name):
        if 0 <= i < len(self.classes):
            self.classes[i]['name'] = name

    def add_class(self):
        n = len(self.classes)
        self.classes.append({
            'name': f'class_{n + 1}',
            'color': DEFAULT_COLORS[n % len(DEFAULT_COLORS)],
        })
        self.active_var.set(n)
        self.refresh_classes()

    def remove_class(self, i):
        if not (0 <= i < len(self.classes)):
            return
        name = self.classes[i]['name']
        if not messagebox.askyesno(
                'Remove class',
                f'Remove "{name}"?\nAll boxes of this class will be deleted.'):
            return
        del self.classes[i]

        # drop boxes of that class and shift class indices
        for key, boxes in list(self.annotations.items()):
            new = []
            for b in boxes:
                if b['cls'] == i:
                    continue
                if b['cls'] > i:
                    b['cls'] -= 1
                new.append(b)
            self.annotations[key] = new

        if self.active_var.get() >= len(self.classes):
            self.active_var.set(max(0, len(self.classes) - 1))
        self.refresh_classes()
        self.draw_boxes()

    def pick_color(self, i):
        if not (0 <= i < len(self.classes)):
            return
        _, hexcol = colorchooser.askcolor(color=self.classes[i]['color'],
                                          title='Pick a color for this class')
        if hexcol:
            self.classes[i]['color'] = hexcol
            self.active_var.set(i)
            self.refresh_classes()
            self.draw_boxes()

    def active_class(self):
        i = self.active_var.get()
        if 0 <= i < len(self.classes):
            return i
        return None

    # -------------------------------------------------------------- project #
    def open_folder(self):
        d = filedialog.askdirectory(title='Select the folder containing your images')
        if not d:
            return
        names = sorted(f for f in os.listdir(d) if f.lower().endswith(IMAGE_EXTS))
        if not names:
            messagebox.showwarning('No images',
                                   'No supported image files were found in that folder.')
            return

        self.folder = d
        self.project_file = os.path.join(d, PROJECT_FILE)
        self.image_paths = [os.path.join(d, f) for f in names]
        self.annotations = {}
        self.classes = []

        if os.path.exists(self.project_file):
            try:
                with open(self.project_file, 'r', encoding='utf-8') as fh:
                    data = json.load(fh)
                self.classes = data.get('classes', []) or []
                self.annotations = data.get('annotations', {}) or {}
            except Exception as e:
                messagebox.showerror('Could not read project', str(e))

        if not self.classes:
            self.classes = [
                {'name': 'class_1', 'color': DEFAULT_COLORS[0]},
                {'name': 'class_2', 'color': DEFAULT_COLORS[1]},
                {'name': 'class_3', 'color': DEFAULT_COLORS[2]},
            ]

        self.active_var.set(0)
        self.refresh_classes()

        self.img_list.delete(0, 'end')
        for f in names:
            self.img_list.insert('end', f)

        self.title(f'Dataset Labeling Tool — {os.path.basename(d)}')
        self.load_image(0)

    def save_project(self, silent=False):
        if not self.folder:
            return
        data = {
            'classes': self.classes,
            'annotations': {k: v for k, v in self.annotations.items() if v},
        }
        try:
            with open(self.project_file, 'w', encoding='utf-8') as fh:
                json.dump(data, fh, indent=1)
            if not silent:
                self.status(f'Project saved to {PROJECT_FILE}')
        except Exception as e:
            if not silent:
                messagebox.showerror('Save failed', str(e))

    def on_close(self):
        self.save_project(silent=True)
        self.destroy()

    # -------------------------------------------------------------- images #
    def load_image(self, i):
        if not self.image_paths:
            return
        i = max(0, min(i, len(self.image_paths) - 1))
        path = self.image_paths[i]
        try:
            im = Image.open(path)
            im.load()
        except Exception as e:
            messagebox.showerror('Cannot open image', f'{path}\n\n{e}')
            return

        if im.mode not in ('RGB', 'RGBA'):
            im = im.convert('RGB')
        self.img = im
        self.index = i
        self.selected = None
        self.drag_id = None
        self.drag_start = None

        self.img_list.selection_clear(0, 'end')
        self.img_list.selection_set(i)
        self.img_list.see(i)
        self.lbl_pos.config(
            text=f'{i + 1} / {len(self.image_paths)}   ·   {os.path.basename(path)}')
        self.render()

    def prev_image(self):
        if self.index > 0:
            self.save_project(silent=True)
            self.load_image(self.index - 1)

    def next_image(self):
        if self.index < len(self.image_paths) - 1:
            self.save_project(silent=True)
            self.load_image(self.index + 1)

    def on_list_select(self, _e):
        sel = self.img_list.curselection()
        if not sel:
            return
        if sel[0] != self.index:
            self.save_project(silent=True)
            self.load_image(sel[0])

    # ----------------------------------------------------------- rendering #
    def boxes_for(self, path, create=False):
        key = os.path.basename(path)
        if create:
            return self.annotations.setdefault(key, [])
        return self.annotations.get(key, [])

    def current_boxes(self, create=True):
        if self.index < 0 or not self.image_paths:
            return []
        return self.boxes_for(self.image_paths[self.index], create=create)

    def i2c(self, x, y):
        return x * self.scale + self.ox, y * self.scale + self.oy

    def c2i(self, x, y):
        return (x - self.ox) / self.scale, (y - self.oy) / self.scale

    def render(self):
        self.canvas.delete('all')
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()

        if self.img is None:
            self.canvas.create_text(cw // 2, ch // 2,
                                    text='Open a folder to start labeling',
                                    fill='#777', font=('TkDefaultFont', 14))
            return
        if cw < 10 or ch < 10:
            return

        iw, ih = self.img.size
        self.scale = min(cw / iw, ch / ih)
        w = max(1, int(iw * self.scale))
        h = max(1, int(ih * self.scale))
        self.ox = (cw - w) // 2
        self.oy = (ch - h) // 2

        resized = self.img.resize((w, h), Image.LANCZOS)
        self.tkimg = ImageTk.PhotoImage(resized)
        self.canvas.create_image(self.ox, self.oy, anchor='nw', image=self.tkimg)
        self.draw_boxes()

    def draw_boxes(self):
        self.canvas.delete('box')
        if self.img is None:
            return
        boxes = self.current_boxes(create=False)

        for i, b in enumerate(boxes):
            if not (0 <= b['cls'] < len(self.classes)):
                continue
            col = self.classes[b['cls']]['color']
            x1, y1 = self.i2c(b['x1'], b['y1'])
            x2, y2 = self.i2c(b['x2'], b['y2'])
            width = 3 if i == self.selected else 2
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=col,
                                         width=width, tags='box')
            if i == self.selected:
                self.canvas.create_rectangle(x1 - 3, y1 - 3, x2 + 3, y2 + 3,
                                             outline='white', dash=(3, 3),
                                             tags='box')

            label = self.classes[b['cls']]['name']
            tid = self.canvas.create_text(x1 + 5, y1 + 5, anchor='nw', text=label,
                                          fill=text_color_for(col),
                                          font=('TkDefaultFont', 9, 'bold'),
                                          tags='box')
            bb = self.canvas.bbox(tid)
            if bb:
                rid = self.canvas.create_rectangle(bb[0] - 3, bb[1] - 1,
                                                   bb[2] + 3, bb[3] + 1,
                                                   fill=col, outline=col, tags='box')
                self.canvas.tag_raise(tid)

    # ------------------------------------------------------------- mouse   #
    def hit_test(self, cx, cy):
        boxes = self.current_boxes(create=False)
        for i in range(len(boxes) - 1, -1, -1):
            b = boxes[i]
            x1, y1 = self.i2c(b['x1'], b['y1'])
            x2, y2 = self.i2c(b['x2'], b['y2'])
            if x1 - 3 <= cx <= x2 + 3 and y1 - 3 <= cy <= y2 + 3:
                return i
        return None

    def on_press(self, e):
        self.canvas.focus_set()
        if self.img is None:
            return

        shift = bool(e.state & 0x0001)
        hit = self.hit_test(e.x, e.y)

        if hit is not None and not shift:
            self.selected = hit
            self.draw_boxes()
            return

        if self.active_class() is None:
            messagebox.showinfo('No class selected',
                                'Add a class and select it before drawing boxes.')
            return

        self.selected = None
        self.drag_start = (e.x, e.y)
        col = self.classes[self.active_class()]['color']
        self.drag_id = self.canvas.create_rectangle(e.x, e.y, e.x, e.y,
                                                    outline=col, width=2,
                                                    dash=(5, 3))

    def on_motion(self, e):
        if self.drag_id is None or self.drag_start is None:
            return
        x0, y0 = self.drag_start
        self.canvas.coords(self.drag_id, x0, y0, e.x, e.y)

    def on_release(self, e):
        if self.drag_id is None:
            return
        self.canvas.delete(self.drag_id)
        self.drag_id = None
        x0, y0 = self.drag_start
        self.drag_start = None

        x1, y1 = self.c2i(x0, y0)
        x2, y2 = self.c2i(e.x, e.y)
        iw, ih = self.img.size
        x1 = max(0, min(iw, x1))
        x2 = max(0, min(iw, x2))
        y1 = max(0, min(ih, y1))
        y2 = max(0, min(ih, y2))

        if abs(x2 - x1) < 4 or abs(y2 - y1) < 4:
            self.draw_boxes()
            return

        boxes = self.current_boxes(create=True)
        boxes.append({
            'cls': self.active_class(),
            'x1': min(x1, x2), 'y1': min(y1, y2),
            'x2': max(x1, x2), 'y2': max(y1, y2),
        })
        self.selected = len(boxes) - 1
        self.draw_boxes()
        self.save_project(silent=True)

    def on_right_click(self, e):
        if self.img is None:
            return
        hit = self.hit_test(e.x, e.y)
        if hit is not None:
            boxes = self.current_boxes(create=False)
            del boxes[hit]
            self.selected = None
            self.draw_boxes()
            self.save_project(silent=True)

    # ---------------------------------------------------------- keyboard   #
    def on_key(self, e):
        if isinstance(self.focus_get(), (tk.Entry, ttk.Entry)):
            return

        k = e.keysym
        if k in ('Left', 'a', 'A'):
            self.prev_image()
        elif k in ('Right', 'd', 'D'):
            self.next_image()
        elif k in ('Delete', 'BackSpace'):
            self.delete_selected()
        elif k == 'Escape':
            self.selected = None
            self.draw_boxes()
        elif k in '123456789':
            i = int(k) - 1
            if i < len(self.classes):
                self.active_var.set(i)
                self.refresh_classes()

    def delete_selected(self):
        if self.selected is None:
            return
        boxes = self.current_boxes(create=False)
        if 0 <= self.selected < len(boxes):
            del boxes[self.selected]
        self.selected = None
        self.draw_boxes()
        self.save_project(silent=True)

    # ------------------------------------------------------------ exports  #
    def export_yolo(self):
        if not self.folder:
            return
        out = filedialog.askdirectory(title='Choose a folder for YOLO labels')
        if not out:
            return

        labels_dir = os.path.join(out, 'labels')
        os.makedirs(labels_dir, exist_ok=True)

        with open(os.path.join(out, 'classes.txt'), 'w', encoding='utf-8') as fh:
            for c in self.classes:
                fh.write(c['name'] + '\n')

        written = 0
        for path in self.image_paths:
            try:
                with Image.open(path) as im:
                    iw, ih = im.size
            except Exception:
                continue

            lines = []
            for b in self.annotations.get(os.path.basename(path), []):
                if not (0 <= b['cls'] < len(self.classes)):
                    continue
                xc = (b['x1'] + b['x2']) / 2.0 / iw
                yc = (b['y1'] + b['y2']) / 2.0 / ih
                bw = (b['x2'] - b['x1']) / iw
                bh = (b['y2'] - b['y1']) / ih
                lines.append(f"{b['cls']} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

            stem = os.path.splitext(os.path.basename(path))[0]
            with open(os.path.join(labels_dir, stem + '.txt'), 'w', encoding='utf-8') as fh:
                fh.write('\n'.join(lines))
            written += 1

        self.status(f'YOLO export complete — {written} label files in {labels_dir}')
        messagebox.showinfo(
            'Export complete',
            f'{written} label files written to:\n{labels_dir}\n\n'
            f'classes.txt is in:\n{out}')

    def _render_boxes(self, im, boxes, with_labels=True, width_scale=1.0):
        draw = ImageDraw.Draw(im)
        lw = max(2, int(min(im.size) / 300 * width_scale))
        for b in boxes:
            if not (0 <= b['cls'] < len(self.classes)):
                continue
            col = self.classes[b['cls']]['color']
            draw.rectangle([b['x1'], b['y1'], b['x2'], b['y2']],
                           outline=col, width=lw)
            if not with_labels:
                continue
            label = self.classes[b['cls']]['name']
            tx, ty = b['x1'] + lw + 2, b['y1'] + lw + 2
            try:
                bb = draw.textbbox((tx, ty), label)
            except Exception:
                bb = (tx, ty, tx + 6 * len(label), ty + 11)
            draw.rectangle([bb[0] - 2, bb[1] - 1, bb[2] + 2, bb[3] + 1], fill=col)
            draw.text((tx, ty), label, fill=text_color_for(col))
        return im

    def export_images(self):
        if not self.folder:
            return
        out = filedialog.askdirectory(title='Choose an output folder for rendered images')
        if not out:
            return

        n = 0
        for path in self.image_paths:
            try:
                im = Image.open(path).convert('RGB')
            except Exception:
                continue
            boxes = self.annotations.get(os.path.basename(path), [])
            self._render_boxes(im, boxes)
            name = os.path.basename(path)
            im.save(os.path.join(out, name))
            n += 1

        self.status(f'Rendered {n} images to {out}')
        messagebox.showinfo('Export complete', f'{n} images written to:\n{out}')

    def export_masks(self):
        if not self.folder:
            return
        out = filedialog.askdirectory(title='Choose an output folder for color masks')
        if not out:
            return

        n = 0
        for path in self.image_paths:
            try:
                with Image.open(path) as src:
                    size = src.size
            except Exception:
                continue
            mask = Image.new('RGB', size, (0, 0, 0))
            draw = ImageDraw.Draw(mask)
            for b in self.annotations.get(os.path.basename(path), []):
                if not (0 <= b['cls'] < len(self.classes)):
                    continue
                draw.rectangle([b['x1'], b['y1'], b['x2'], b['y2']],
                               fill=self.classes[b['cls']]['color'])
            stem = os.path.splitext(os.path.basename(path))[0]
            mask.save(os.path.join(out, stem + '_mask.png'))
            n += 1

        # save a legend
        with open(os.path.join(out, 'legend.txt'), 'w', encoding='utf-8') as fh:
            for i, c in enumerate(self.classes):
                fh.write(f'{i}\t{c["name"]}\t{c["color"]}\n')

        self.status(f'Wrote {n} masks to {out}')
        messagebox.showinfo('Export complete', f'{n} masks written to:\n{out}')

    # ------------------------------------------------------- CSV + MASKS  #
    def export_csv_masks(self):
        """
        Export color masks and a CSV that maps each mask back to its source
        image, all into a single chosen folder.

        Layout of the chosen folder:
            <chosen>/
                <image_stem>_mask.png      (one per source image)
                masks.csv                  (columns: mask_filename,image_filename)
                legend.txt                 (class index -> name -> color)
        """
        if not self.folder:
            return
        out = filedialog.askdirectory(
            title='Choose a folder to receive masks + masks.csv')
        if not out:
            return

        rows = []       # (mask_filename, image_filename)
        n = 0

        for path in self.image_paths:
            try:
                with Image.open(path) as src:
                    size = src.size
            except Exception:
                continue

            mask = Image.new('RGB', size, (0, 0, 0))
            draw = ImageDraw.Draw(mask)
            for b in self.annotations.get(os.path.basename(path), []):
                if not (0 <= b['cls'] < len(self.classes)):
                    continue
                draw.rectangle([b['x1'], b['y1'], b['x2'], b['y2']],
                               fill=self.classes[b['cls']]['color'])

            image_name = os.path.basename(path)
            stem = os.path.splitext(image_name)[0]
            mask_name = stem + '_mask.png'

            mask.save(os.path.join(out, mask_name))
            rows.append((mask_name, image_name))
            n += 1

        # Write the CSV in the same folder.
        csv_path = os.path.join(out, 'masks.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as fh:
            writer = csv.writer(fh)
            writer.writerow(['mask_filename', 'image_filename'])
            writer.writerows(rows)

        # Write a small class legend too, so the mask colors are decodable.
        with open(os.path.join(out, 'legend.txt'), 'w', encoding='utf-8') as fh:
            for i, c in enumerate(self.classes):
                fh.write(f'{i}\t{c["name"]}\t{c["color"]}\n')

        self.status(f'Wrote {n} masks + masks.csv to {out}')
        messagebox.showinfo(
            'Export complete',
            f'{n} masks + masks.csv written to:\n{out}\n\n'
            f'CSV columns: mask_filename, image_filename')

    # ------------------------------------------------------------- status  #
    def status(self, text):
        self.status_var.set(text)


if __name__ == '__main__':
    Labeler().mainloop()