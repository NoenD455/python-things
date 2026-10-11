import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import mido
import os

# ----------------------------------------------------------------------
# Scale definitions (semitone offsets from the root)
# ----------------------------------------------------------------------
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

SCALES = {
    "Major (Ionian)":          [0, 2, 4, 5, 7, 9, 11],
    "Natural Minor (Aeolian)": [0, 2, 3, 5, 7, 8, 10],
    "Harmonic Minor":          [0, 2, 3, 5, 7, 8, 11],
    "Melodic Minor":           [0, 2, 3, 5, 7, 9, 11],
    "Dorian":                  [0, 2, 3, 5, 7, 9, 10],
    "Phrygian":                [0, 1, 3, 5, 7, 8, 10],
    "Lydian":                  [0, 2, 4, 6, 7, 9, 11],
    "Mixolydian":              [0, 2, 4, 5, 7, 9, 10],
    "Locrian":                 [0, 1, 3, 5, 6, 8, 10],
    "Major Pentatonic":        [0, 2, 4, 7, 9],
    "Minor Pentatonic":        [0, 3, 5, 7, 10],
    "Blues":                   [0, 3, 5, 6, 7, 10],
    "Whole Tone":              [0, 2, 4, 6, 8, 10],
    "Chromatic":               list(range(12)),
}

# ----------------------------------------------------------------------
# Build a 128-entry lookup table mapping every MIDI note -> new note
# ----------------------------------------------------------------------
def build_note_map(source_root, source_scale,
                   target_root, target_scale):
    """
    For each MIDI note (0-127):
      - compute its scale-degree in the source scale (snapping
        chromatic notes to the nearest degree)
      - map that degree proportionally onto the target scale,
        preserving the octave position relative to the source root.
    """
    n_src = len(source_scale)
    n_tgt = len(target_scale)
    note_map = {}

    for n in range(128):
        rel = n - source_root
        octave = rel // 12           # which scale-octave we're in
        pc = rel % 12                # pitch class relative to source root

        # Find the closest degree in the source scale
        best_deg = 0
        best_dist = 99
        for i, s in enumerate(source_scale):
            d = min(abs(pc - s), 12 - abs(pc - s))   # circular distance
            if d < best_dist:
                best_dist = d
                best_deg = i

        # Map the degree proportionally onto the target scale,
        # e.g. degree 6 of 7 -> index 5 of 6 (Blues)
        tgt_index = int(round(best_deg * n_tgt / n_src))
        if tgt_index >= n_tgt:       # wrapped past the top -> next octave
            tgt_index -= n_tgt
            octave += 1

        new_note = target_root + target_scale[tgt_index] + octave * 12

        note_map[n] = new_note if 0 <= new_note <= 127 else n

    return note_map

# ----------------------------------------------------------------------
# Apply the map to a MIDI file
# ----------------------------------------------------------------------
def convert_midi(input_path, output_path,
                 source_root, source_scale,
                 target_root, target_scale):
    note_map = build_note_map(source_root, source_scale,
                              target_root, target_scale)

    mid = mido.MidiFile(input_path)
    for track in mid.tracks:
        for msg in track:
            if msg.type in ('note_on', 'note_off'):
                msg.note = note_map.get(msg.note, msg.note)
    mid.save(output_path)

# ----------------------------------------------------------------------
# GUI
# ----------------------------------------------------------------------
class ScaleChangerApp:
    def __init__(self, root):
        self.root = root
        root.title("MIDI Scale Changer")
        root.geometry("560x420")
        root.resizable(False, False)

        pad = {"padx": 10, "pady": 4}

        # ---------- Input file ----------
        tk.Label(root, text="Input MIDI file:").grid(row=0, column=0,
                                                     sticky="w", **pad)
        self.input_entry = tk.Entry(root, width=55)
        self.input_entry.grid(row=1, column=0, columnspan=2,
                              padx=10, sticky="we")
        tk.Button(root, text="Browse…", command=self.browse_input)\
            .grid(row=1, column=2, padx=(0, 10))

        # ---------- Source scale ----------
        src = tk.LabelFrame(root, text="Source scale (what the song is in)")
        src.grid(row=2, column=0, columnspan=3, padx=10, pady=10, sticky="we")

        tk.Label(src, text="Root:").grid(row=0, column=0, padx=6, pady=6)
        self.src_root = ttk.Combobox(src, values=NOTE_NAMES,
                                     width=4, state="readonly")
        self.src_root.set("C")
        self.src_root.grid(row=0, column=1, padx=6)

        tk.Label(src, text="Scale:").grid(row=0, column=2, padx=6)
        self.src_scale = ttk.Combobox(src, values=list(SCALES.keys()),
                                      width=24, state="readonly")
        self.src_scale.set("Major (Ionian)")
        self.src_scale.grid(row=0, column=3, padx=6)

        # ---------- Target scale ----------
        tgt = tk.LabelFrame(root, text="Target scale (what you want)")
        tgt.grid(row=3, column=0, columnspan=3, padx=10, pady=5, sticky="we")

        tk.Label(tgt, text="Root:").grid(row=0, column=0, padx=6, pady=6)
        self.tgt_root = ttk.Combobox(tgt, values=NOTE_NAMES,
                                     width=4, state="readonly")
        self.tgt_root.set("C")
        self.tgt_root.grid(row=0, column=1, padx=6)

        tk.Label(tgt, text="Scale:").grid(row=0, column=2, padx=6)
        self.tgt_scale = ttk.Combobox(tgt, values=list(SCALES.keys()),
                                      width=24, state="readonly")
        self.tgt_scale.set("Melodic Minor")
        self.tgt_scale.grid(row=0, column=3, padx=6)

        # ---------- Output file ----------
        tk.Label(root, text="Save output as:").grid(row=4, column=0,
                                                    sticky="w", **pad)
        self.output_entry = tk.Entry(root, width=55)
        self.output_entry.grid(row=5, column=0, columnspan=2,
                               padx=10, sticky="we")
        tk.Button(root, text="Save As…", command=self.browse_output)\
            .grid(row=5, column=2, padx=(0, 10))

        # ---------- Action ----------
        tk.Button(root, text="Convert & Save", command=self.run,
                  bg="#4CAF50", fg="white",
                  font=("Arial", 11, "bold"))\
            .grid(row=6, column=0, columnspan=3, pady=18, ipadx=25)

        # ---------- Status ----------
        self.status = tk.Label(root, text="", fg="gray")
        self.status.grid(row=7, column=0, columnspan=3)

    # ------------------------------------------------------------------
    def browse_input(self):
        path = filedialog.askopenfilename(
            title="Select MIDI file",
            filetypes=[("MIDI files", "*.mid *.midi"), ("All files", "*.*")])
        if path:
            self.input_entry.delete(0, tk.END)
            self.input_entry.insert(0, path)
            base, ext = os.path.splitext(path)
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, f"{base}_converted{ext}")

    def browse_output(self):
        path = filedialog.asksaveasfilename(
            title="Save converted MIDI as",
            defaultextension=".mid",
            filetypes=[("MIDI files", "*.mid *.midi"), ("All files", "*.*")])
        if path:
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, path)

    # ------------------------------------------------------------------
    def run(self):
        inp = self.input_entry.get().strip()
        out = self.output_entry.get().strip()

        if not inp or not os.path.isfile(inp):
            messagebox.showerror("Error", "Please select a valid input file.")
            return
        if not out:
            messagebox.showerror("Error", "Please specify an output file.")
            return

        src_root  = NOTE_NAMES.index(self.src_root.get())
        tgt_root  = NOTE_NAMES.index(self.tgt_root.get())
        src_scale = SCALES[self.src_scale.get()]
        tgt_scale = SCALES[self.tgt_scale.get()]

        try:
            self.status.config(text="Working…")
            self.root.update_idletasks()
            convert_midi(inp, out, src_root, src_scale, tgt_root, tgt_scale)
            self.status.config(text="")
            messagebox.showinfo(
                "Done",
                f"Converted from {self.src_root.get()} "
                f"{self.src_scale.get()}\n"
                f"to {self.tgt_root.get()} {self.tgt_scale.get()}\n\n"
                f"Saved to:\n{out}")
        except Exception as e:
            self.status.config(text="")
            messagebox.showerror("Error", f"Something went wrong:\n{e}")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = ScaleChangerApp(root)
    root.mainloop()