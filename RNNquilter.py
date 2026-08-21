import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext
from PIL import Image, ImageTk
import threading
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.cluster import KMeans
import os

# ----------------------------
#  Default configuration
# ----------------------------
DEFAULT_CONFIG = {
    'image_size': 28,
    'patch_size': 4,
    'channels': 1,
    'vocab_size': 128,
    'rnn_type': 'LSTM',
    'num_layers': 1,
    'hidden_dim': 128,
    'dropout': 0.0,
    'batch_size': 128,
    'lr': 1e-3,
    'beta1': 0.9,
    'epochs': 10,
}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ----------------------------
#  Model definition
# ----------------------------
class AutoregressiveRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=1, rnn_type='LSTM', dropout=0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim)
        rnn_class = {'RNN': nn.RNN, 'GRU': nn.GRU, 'LSTM': nn.LSTM}[rnn_type]
        self.rnn = rnn_class(embed_dim, hidden_dim, num_layers,
                             batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.rnn(emb)
        logits = self.fc(out)
        return logits


# ----------------------------
#  Helper functions
# ----------------------------
def extract_patches_from_images(images, config):
    img_size = config['image_size']
    patch_size = config['patch_size']
    channels = config['channels']
    N = images.shape[0]
    patches = []
    for i in range(N):
        img = images[i]
        for r in range(0, img_size, patch_size):
            for c in range(0, img_size, patch_size):
                if channels == 1:
                    patch = img[r:r+patch_size, c:c+patch_size].flatten()
                else:
                    patch = img[r:r+patch_size, c:c+patch_size].transpose(2,0,1).flatten()
                patches.append(patch)
    return np.array(patches, dtype=np.float32)


def encode_image(img_tensor, codebook, config):
    img_size = config['image_size']
    patch_size = config['patch_size']
    channels = config['channels']
    if img_tensor.dim() == 2:
        img_tensor = img_tensor.unsqueeze(0)
    elif img_tensor.dim() == 3 and img_tensor.shape[-1] == channels:
        img_tensor = img_tensor.permute(2,0,1)
    idxs = []
    for r in range(0, img_size, patch_size):
        for c in range(0, img_size, patch_size):
            patch = img_tensor[:, r:r+patch_size, c:c+patch_size].flatten()
            dist = torch.cdist(patch.unsqueeze(0), codebook)
            idx = torch.argmin(dist).item()
            idxs.append(idx)
    return idxs


def reconstruct_from_indices(indices, codebook_np, config):
    img_size = config['image_size']
    patch_size = config['patch_size']
    channels = config['channels']
    patches = codebook_np[indices]
    if channels == 1:
        img = np.zeros((img_size, img_size), dtype=np.float32)
        for i in range(len(indices)):
            r = (i // (img_size // patch_size)) * patch_size
            c = (i % (img_size // patch_size)) * patch_size
            patch = patches[i].reshape(patch_size, patch_size)
            img[r:r+patch_size, c:c+patch_size] = patch
        img = (img + 1) / 2
        return np.clip(img, 0, 1)
    else:
        img = np.zeros((img_size, img_size, channels), dtype=np.float32)
        for i in range(len(indices)):
            r = (i // (img_size // patch_size)) * patch_size
            c = (i % (img_size // patch_size)) * patch_size
            patch = patches[i].reshape(channels, patch_size, patch_size).transpose(1,2,0)
            img[r:r+patch_size, c:c+patch_size] = patch
        img = (img + 1) / 2
        return np.clip(img, 0, 1)


# ----------------------------
#  Dataset
# ----------------------------
class SeqDataset(Dataset):
    def __init__(self, sequences, start_token):
        self.sequences = sequences
        self.start_token = start_token
        self.seq_len = len(sequences[0])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        inp = [self.start_token] + seq[:-1]
        target = seq[:]
        return torch.tensor(inp, dtype=torch.long), torch.tensor(target, dtype=torch.long)


# ----------------------------
#  Main Application
# ----------------------------
class App:
    def __init__(self, root):
        self.root = root
        root.title("Autoregressive Image Generator")
        root.geometry("1100x750")

        self.config = DEFAULT_CONFIG.copy()
        self.codebook = None
        self.codebook_np = None
        self.model = None
        self.train_loader = None
        self.sequences = None
        self.loaded_images_np = None
        self.is_trained = False
        self.training_thread = None
        self.stop_training = False
        self.images_loaded = False

        # Completion state
        self.test_image = None       # numpy array (H,W) or (H,W,C) in [-1,1]
        self.test_indices = None     # list of patch indices
        self.completion_image = None # numpy array for display

        self.create_widgets()
        self.status_var = tk.StringVar()
        self.status_var.set("Ready. Load images, then click 'Initialize Model'.")
        status_label = tk.Label(root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        status_label.pack(side=tk.BOTTOM, fill=tk.X)

    def create_widgets(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        train_tab = ttk.Frame(self.notebook)
        self.notebook.add(train_tab, text="Train / Generate")
        self.build_train_tab(train_tab)

        settings_tab = ttk.Frame(self.notebook)
        self.notebook.add(settings_tab, text="Settings")
        self.build_settings_tab(settings_tab)

        # ---- NEW Completion Tab ----
        completion_tab = ttk.Frame(self.notebook)
        self.notebook.add(completion_tab, text="Completion")
        self.build_completion_tab(completion_tab)

    # ---------- Train / Generate Tab ----------
    def build_train_tab(self, parent):
        control_frame = tk.Frame(parent, width=280)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0,10))
        control_frame.pack_propagate(False)

        tk.Label(control_frame, text="Controls", font=('Arial', 14, 'bold')).pack(pady=5)
        tk.Label(control_frame, text="Load Images", font=('Arial', 10, 'bold')).pack(pady=(5,0))
        tk.Button(control_frame, text="From Folder", command=self.load_from_folder, width=22).pack(pady=2)
        tk.Button(control_frame, text="From Files (multiple)", command=self.load_from_files, width=22).pack(pady=2)

        self.init_btn = tk.Button(control_frame, text="Initialize Model", command=self.init_model,
                                  width=22, state=tk.DISABLED)
        self.init_btn.pack(pady=5)

        self.train_btn = tk.Button(control_frame, text="Train RNN", command=self.train_rnn,
                                   width=22, state=tk.DISABLED)
        self.train_btn.pack(pady=5)

        tk.Label(control_frame, text="Temperature:").pack(pady=(10,0))
        self.temp_var = tk.DoubleVar(value=0.8)
        tk.Scale(control_frame, from_=0.1, to=2.0, resolution=0.1,
                 orient=tk.HORIZONTAL, variable=self.temp_var, length=180).pack(pady=5)

        self.gen_btn = tk.Button(control_frame, text="Generate", command=self.generate,
                                 width=22, state=tk.DISABLED)
        self.gen_btn.pack(pady=10)

        self.stop_btn = tk.Button(control_frame, text="Stop Training", command=self.stop_training_cmd,
                                  width=22, state=tk.DISABLED)
        self.stop_btn.pack(pady=5)

        tk.Label(control_frame, text="Training Log:", font=('Arial', 10, 'bold')).pack(pady=(15,0))
        self.log_text = scrolledtext.ScrolledText(control_frame, height=10, width=30, state=tk.NORMAL)
        self.log_text.pack(pady=5, fill=tk.BOTH, expand=True)
        self.log_text.config(state=tk.DISABLED)
        tk.Button(control_frame, text="Clear Log", command=self.clear_log, width=22).pack(pady=2)

        display_frame = tk.Frame(parent)
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.image_labels = []
        for title in ["Original", "Reconstruction", "Generated"]:
            subframe = tk.Frame(display_frame, relief=tk.RIDGE, bd=2)
            subframe.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
            tk.Label(subframe, text=title, font=('Arial', 10, 'bold')).pack()
            lbl = tk.Label(subframe, bg='white', width=150, height=150)
            lbl.pack(padx=5, pady=5, expand=True)
            self.image_labels.append(lbl)

        for lbl in self.image_labels:
            self.display_placeholder(lbl)

    # ---------- Settings Tab ----------
    def build_settings_tab(self, parent):
        arch_frame = ttk.LabelFrame(parent, text="Model Architecture", padding=10)
        arch_frame.grid(row=0, column=0, sticky='w', padx=10, pady=10)

        row = 0
        ttk.Label(arch_frame, text="RNN Type:").grid(row=row, column=0, sticky='w', pady=2)
        self.rnn_type_var = tk.StringVar(value=self.config['rnn_type'])
        ttk.Combobox(arch_frame, textvariable=self.rnn_type_var, values=['RNN', 'GRU', 'LSTM'], width=10).grid(row=row, column=1, sticky='w', padx=10)
        row += 1

        ttk.Label(arch_frame, text="Num Layers:").grid(row=row, column=0, sticky='w', pady=2)
        self.num_layers_var = tk.IntVar(value=self.config['num_layers'])
        ttk.Spinbox(arch_frame, from_=1, to=5, textvariable=self.num_layers_var, width=10).grid(row=row, column=1, sticky='w', padx=10)
        row += 1

        ttk.Label(arch_frame, text="Hidden Units:").grid(row=row, column=0, sticky='w', pady=2)
        self.hidden_dim_var = tk.IntVar(value=self.config['hidden_dim'])
        ttk.Spinbox(arch_frame, from_=16, to=512, increment=16, textvariable=self.hidden_dim_var, width=10).grid(row=row, column=1, sticky='w', padx=10)
        row += 1

        ttk.Label(arch_frame, text="Dropout:").grid(row=row, column=0, sticky='w', pady=2)
        self.dropout_var = tk.DoubleVar(value=self.config['dropout'])
        tk.Scale(arch_frame, from_=0.0, to=1.0, resolution=0.01,
                 variable=self.dropout_var, orient=tk.HORIZONTAL, length=150).grid(row=row, column=1, sticky='w', padx=10)
        ttk.Label(arch_frame, textvariable=self.dropout_var).grid(row=row, column=2, padx=5)
        row += 1

        data_frame = ttk.LabelFrame(parent, text="Data (applied on Init)", padding=10)
        data_frame.grid(row=1, column=0, sticky='w', padx=10, pady=10)

        row = 0
        ttk.Label(data_frame, text="Image Size (px):").grid(row=row, column=0, sticky='w', pady=2)
        self.img_size_var = tk.IntVar(value=self.config['image_size'])
        ttk.Spinbox(data_frame, from_=8, to=256, increment=4, textvariable=self.img_size_var, width=10).grid(row=row, column=1, sticky='w', padx=10)
        row += 1

        ttk.Label(data_frame, text="Patch Size:").grid(row=row, column=0, sticky='w', pady=2)
        self.patch_size_var = tk.IntVar(value=self.config['patch_size'])
        ttk.Spinbox(data_frame, from_=2, to=64, increment=2, textvariable=self.patch_size_var, width=10).grid(row=row, column=1, sticky='w', padx=10)
        row += 1

        ttk.Label(data_frame, text="Channels:").grid(row=row, column=0, sticky='w', pady=2)
        self.channels_var = tk.IntVar(value=self.config['channels'])
        ttk.Combobox(data_frame, textvariable=self.channels_var, values=[1, 3], width=5).grid(row=row, column=1, sticky='w', padx=10)
        row += 1

        ttk.Label(data_frame, text="Vocab Size:").grid(row=row, column=0, sticky='w', pady=2)
        self.vocab_size_var = tk.IntVar(value=self.config['vocab_size'])
        ttk.Spinbox(data_frame, from_=16, to=512, increment=16, textvariable=self.vocab_size_var, width=10).grid(row=row, column=1, sticky='w', padx=10)
        row += 1

        train_frame = ttk.LabelFrame(parent, text="Training (applied on Train)", padding=10)
        train_frame.grid(row=0, column=1, sticky='n', padx=10, pady=10)

        row = 0
        ttk.Label(train_frame, text="Batch Size:").grid(row=row, column=0, sticky='w', pady=2)
        self.batch_size_var = tk.IntVar(value=self.config['batch_size'])
        ttk.Spinbox(train_frame, from_=8, to=512, increment=8, textvariable=self.batch_size_var, width=10).grid(row=row, column=1, sticky='w', padx=10)
        row += 1

        ttk.Label(train_frame, text="Learning Rate:").grid(row=row, column=0, sticky='w', pady=2)
        self.lr_var = tk.StringVar(value=str(self.config['lr']))
        ttk.Entry(train_frame, textvariable=self.lr_var, width=10).grid(row=row, column=1, sticky='w', padx=10)
        row += 1

        ttk.Label(train_frame, text="Beta1 (Adam):").grid(row=row, column=0, sticky='w', pady=2)
        self.beta1_var = tk.StringVar(value=str(self.config['beta1']))
        ttk.Entry(train_frame, textvariable=self.beta1_var, width=10).grid(row=row, column=1, sticky='w', padx=10)
        row += 1

        ttk.Label(train_frame, text="Epochs:").grid(row=row, column=0, sticky='w', pady=2)
        self.epochs_var = tk.IntVar(value=self.config['epochs'])
        ttk.Spinbox(train_frame, from_=1, to=1000, textvariable=self.epochs_var, width=10).grid(row=row, column=1, sticky='w', padx=10)
        row += 1

    # ---------- NEW Completion Tab ----------
    def build_completion_tab(self, parent):
        control_frame = tk.Frame(parent, width=280)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0,10))
        control_frame.pack_propagate(False)

        tk.Label(control_frame, text="Completion", font=('Arial', 14, 'bold')).pack(pady=5)

        tk.Button(control_frame, text="Load Test Image", command=self.load_test_image, width=22).pack(pady=5)

        tk.Label(control_frame, text="Completion % (bottom):").pack(pady=(10,0))
        self.completion_percent_var = tk.IntVar(value=50)
        comp_scale = tk.Scale(control_frame, from_=10, to=90, orient=tk.HORIZONTAL,
                              variable=self.completion_percent_var, length=180)
        comp_scale.pack(pady=5)

        tk.Label(control_frame, text="Temperature:").pack(pady=(10,0))
        self.completion_temp_var = tk.DoubleVar(value=0.8)
        temp_scale = tk.Scale(control_frame, from_=0.1, to=2.0, resolution=0.1,
                              orient=tk.HORIZONTAL, variable=self.completion_temp_var, length=180)
        temp_scale.pack(pady=5)

        self.complete_btn = tk.Button(control_frame, text="Complete", command=self.complete_image,
                                      width=22, state=tk.DISABLED)
        self.complete_btn.pack(pady=10)

        # Display area for completion
        display_frame = tk.Frame(parent)
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.completion_labels = []
        for title in ["Original", "Partial (masked)", "Completed"]:
            subframe = tk.Frame(display_frame, relief=tk.RIDGE, bd=2)
            subframe.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
            tk.Label(subframe, text=title, font=('Arial', 10, 'bold')).pack()
            lbl = tk.Label(subframe, bg='white', width=150, height=150)
            lbl.pack(padx=5, pady=5, expand=True)
            self.completion_labels.append(lbl)

        for lbl in self.completion_labels:
            self.display_placeholder(lbl)

    # ---------- Common UI Helpers ----------
    def read_settings_from_ui(self):
        self.config['rnn_type'] = self.rnn_type_var.get()
        self.config['num_layers'] = self.num_layers_var.get()
        self.config['hidden_dim'] = self.hidden_dim_var.get()
        self.config['dropout'] = self.dropout_var.get()
        self.config['image_size'] = self.img_size_var.get()
        self.config['patch_size'] = self.patch_size_var.get()
        self.config['channels'] = self.channels_var.get()
        self.config['vocab_size'] = self.vocab_size_var.get()
        self.config['batch_size'] = self.batch_size_var.get()
        try:
            self.config['lr'] = float(self.lr_var.get())
        except:
            self.config['lr'] = 1e-3
        try:
            self.config['beta1'] = float(self.beta1_var.get())
        except:
            self.config['beta1'] = 0.9
        self.config['epochs'] = self.epochs_var.get()

    def display_placeholder(self, label):
        arr = np.zeros((28, 28), dtype=np.uint8)
        img = Image.fromarray(arr, mode='L')
        img = img.resize((150, 150), Image.NEAREST)
        photo = ImageTk.PhotoImage(img)
        label.config(image=photo)
        label.image = photo

    def show_image_on_label(self, label, img_np):
        if img_np is None:
            self.display_placeholder(label)
            return
        if img_np.ndim == 2:
            arr = (img_np * 255).astype(np.uint8)
            img = Image.fromarray(arr, mode='L')
        else:
            arr = (img_np * 255).astype(np.uint8)
            img = Image.fromarray(arr, mode='RGB')
        img = img.resize((150, 150), Image.NEAREST)
        photo = ImageTk.PhotoImage(img)
        label.config(image=photo)
        label.image = photo

    def append_log(self, text):
        self.root.after(0, lambda: self._append_log_impl(text))

    def _append_log_impl(self, text):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, text + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def clear_log(self):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)

    # ---------- Image Loading (training set) ----------
    def load_images_from_paths(self, paths):
        if not paths:
            return

        self.read_settings_from_ui()
        img_size = self.config['image_size']
        channels = self.config['channels']

        self.status_var.set(f"Loading {len(paths)} images...")
        self.root.update()

        images = []
        for p in paths:
            try:
                img = Image.open(p).convert('RGB')
                img = img.resize((img_size, img_size), Image.Resampling.LANCZOS)
                arr = np.array(img, dtype=np.float32) / 255.0 * 2 - 1
                if channels == 1:
                    arr = 0.299 * arr[:,:,0] + 0.587 * arr[:,:,1] + 0.114 * arr[:,:,2]
                images.append(arr)
            except Exception as e:
                print(f"Error loading {p}: {e}")

        if not images:
            messagebox.showerror("Error", "No valid images could be loaded.")
            return

        self.loaded_images_np = images
        self.images_loaded = True
        self.append_log(f"Loaded {len(images)} images.")
        self.status_var.set(f"Loaded {len(images)} images. Click 'Initialize Model'.")
        self.init_btn.config(state=tk.NORMAL)
        self.train_btn.config(state=tk.DISABLED)
        self.gen_btn.config(state=tk.DISABLED)

        first = images[0]
        if first.ndim == 2:
            self.original_img_np = (first + 1) / 2
            self.show_image_on_label(self.image_labels[0], self.original_img_np)
        else:
            self.original_img_np = (first + 1) / 2
            self.show_image_on_label(self.image_labels[0], self.original_img_np)
        self.display_placeholder(self.image_labels[1])
        self.display_placeholder(self.image_labels[2])

    def load_from_folder(self):
        folder = filedialog.askdirectory(title="Select folder with images")
        if not folder:
            return
        paths = [os.path.join(folder, f) for f in os.listdir(folder)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not paths:
            messagebox.showerror("Error", "No image files found.")
            return
        self.load_images_from_paths(paths)

    def load_from_files(self):
        paths = filedialog.askopenfilenames(
            title="Select one or more images",
            filetypes=[("Image files", "*.png *.jpg *.jpeg"), ("All files", "*.*")]
        )
        if not paths:
            return
        self.load_images_from_paths(list(paths))

    # ---------- Initialize Model ----------
    def init_model(self):
        if not self.images_loaded or self.loaded_images_np is None:
            messagebox.showerror("Error", "Load images first.")
            return

        if self.training_thread and self.training_thread.is_alive():
            if not messagebox.askyesno("Training in progress",
                                       "Training is running. Stop and re-initialize?"):
                return
            self.stop_training = True
            self.training_thread.join(timeout=1.0)

        self.read_settings_from_ui()
        img_size = self.config['image_size']
        patch_size = self.config['patch_size']
        channels = self.config['channels']
        vocab_size = self.config['vocab_size']

        if img_size % patch_size != 0:
            messagebox.showerror("Error", f"Image size ({img_size}) must be divisible by patch size ({patch_size}).")
            return

        self.status_var.set("Re-initializing model...")
        self.root.update()

        # Convert stored images to current config
        converted = []
        for arr in self.loaded_images_np:
            if arr.ndim == 2:
                mode = 'L'
                pil_img = Image.fromarray(((arr+1)/2*255).astype(np.uint8), mode='L')
            else:
                mode = 'RGB'
                pil_img = Image.fromarray(((arr+1)/2*255).astype(np.uint8), mode='RGB')
            if pil_img.size != (img_size, img_size):
                pil_img = pil_img.resize((img_size, img_size), Image.Resampling.LANCZOS)
            if channels == 1:
                if mode == 'RGB':
                    pil_img = pil_img.convert('L')
                arr_new = np.array(pil_img, dtype=np.float32) / 255.0 * 2 - 1
            else:
                if mode == 'L':
                    pil_img = pil_img.convert('RGB')
                arr_new = np.array(pil_img, dtype=np.float32) / 255.0 * 2 - 1
            converted.append(arr_new)

        self.loaded_images_np = converted
        images_np = np.stack(converted, axis=0)

        self.append_log(f"Re-encoding {len(images_np)} images with settings: "
                        f"size={img_size}, patch={patch_size}, channels={channels}, vocab={vocab_size}")

        patches = extract_patches_from_images(images_np, self.config)
        kmeans = KMeans(n_clusters=vocab_size, random_state=42, n_init=10)
        kmeans.fit(patches)
        centroids = kmeans.cluster_centers_.astype(np.float32)
        self.codebook_np = centroids
        self.codebook = torch.from_numpy(centroids).to(device)

        sequences = []
        for i in range(len(images_np)):
            if channels == 1:
                t = torch.from_numpy(images_np[i]).float()
            else:
                t = torch.from_numpy(images_np[i]).permute(2,0,1).float()
            seq = encode_image(t, self.codebook, self.config)
            sequences.append(seq)

        self.sequences = sequences
        dataset = SeqDataset(sequences, vocab_size)
        self.train_loader = DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True)

        self.model = AutoregressiveRNN(
            vocab_size=vocab_size,
            embed_dim=64,
            hidden_dim=self.config['hidden_dim'],
            num_layers=self.config['num_layers'],
            rnn_type=self.config['rnn_type'],
            dropout=self.config['dropout']
        ).to(device)

        self.is_trained = False
        self.train_btn.config(state=tk.NORMAL)
        self.gen_btn.config(state=tk.NORMAL)
        self.init_btn.config(state=tk.NORMAL)
        self.complete_btn.config(state=tk.NORMAL)   # Enable completion button
        self.append_log(f"Model initialized: {self.config['rnn_type']}, "
                        f"layers={self.config['num_layers']}, "
                        f"hidden={self.config['hidden_dim']}, "
                        f"dropout={self.config['dropout']:.2f}")

        first = images_np[0]
        if channels == 1:
            self.original_img_np = (first + 1) / 2
            self.show_image_on_label(self.image_labels[0], self.original_img_np)
            t = torch.from_numpy(first).float()
        else:
            self.original_img_np = (first + 1) / 2
            self.show_image_on_label(self.image_labels[0], self.original_img_np)
            t = torch.from_numpy(first).permute(2,0,1).float()

        rec_idx = encode_image(t, self.codebook, self.config)
        rec_img = reconstruct_from_indices(rec_idx, self.codebook_np, self.config)
        self.reconstructed_img_np = rec_img
        self.show_image_on_label(self.image_labels[1], rec_img)
        self.display_placeholder(self.image_labels[2])
        self.status_var.set("Model initialized. You can train, generate, or test completion.")

    # ---------- Training ----------
    def train_rnn(self):
        if not self.images_loaded or self.model is None:
            messagebox.showerror("Error", "Load images and initialize model first.")
            return

        self.read_settings_from_ui()
        vocab_size = self.config['vocab_size']
        batch_size = self.config['batch_size']
        lr = self.config['lr']
        beta1 = self.config['beta1']
        epochs = self.config['epochs']

        if self.sequences is None:
            messagebox.showerror("Error", "No sequences found. Re-initialize model.")
            return
        dataset = SeqDataset(self.sequences, vocab_size)
        self.train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.to(device)

        self.train_btn.config(state=tk.DISABLED)
        self.gen_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.stop_training = False
        self.append_log(f"--- Training started (epochs={epochs}, lr={lr}, beta1={beta1}) ---")

        self.training_thread = threading.Thread(
            target=self.training_worker, args=(epochs, lr, beta1), daemon=True
        )
        self.training_thread.start()

    def training_worker(self, epochs, lr, beta1):
        model = self.model
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1, 0.999))
        criterion = nn.CrossEntropyLoss()
        loader = self.train_loader

        model.train()
        for epoch in range(1, epochs + 1):
            if self.stop_training:
                self.append_log("*** Training stopped by user ***")
                break
            total_loss = 0
            for batch_inp, batch_target in loader:
                if self.stop_training:
                    break
                batch_inp = batch_inp.to(device)
                batch_target = batch_target.to(device)
                optimizer.zero_grad()
                logits = model(batch_inp)
                loss = criterion(logits.permute(0,2,1), batch_target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(loader)
            self.append_log(f"Epoch {epoch}/{epochs}  Loss: {avg_loss:.4f}")
            self.root.after(0, lambda e=epoch, l=avg_loss: self.status_var.set(f"Epoch {e}/{epochs} - Loss: {l:.4f}"))

        if not self.stop_training:
            self.append_log("--- Training completed ---")
        self.is_trained = True
        self.root.after(0, self.training_done)

    def training_done(self):
        self.train_btn.config(state=tk.NORMAL)
        self.gen_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_var.set("Training finished.")

    def stop_training_cmd(self):
        self.stop_training = True
        self.append_log("Stopping training...")
        self.status_var.set("Stopping training...")

    # ---------- Generation (from Train tab) ----------
    def generate(self):
        if not self.images_loaded or self.model is None:
            messagebox.showerror("Error", "Load images and initialize model first.")
            return
        if not self.is_trained:
            if not messagebox.askyesno("Not Trained", "Model is not trained. Generate with random weights?"):
                return

        temp = self.temp_var.get()
        gen_img = self.generate_image(temp)
        self.show_image_on_label(self.image_labels[2], gen_img)

    def generate_image(self, temperature=0.8):
        model = self.model
        model.eval()
        vocab_size = self.config['vocab_size']
        num_patches = (self.config['image_size'] // self.config['patch_size']) ** 2
        start_token = vocab_size

        with torch.no_grad():
            tokens = [start_token]
            for _ in range(num_patches):
                inp = torch.tensor([tokens], dtype=torch.long).to(device)
                logits = model(inp)
                last_logits = logits[0, -1, :] / temperature
                probs = torch.softmax(last_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                tokens.append(next_token)
            indices = tokens[1:]
            return reconstruct_from_indices(indices, self.codebook_np, self.config)

    # ---------- Completion Tab ----------
    def load_test_image(self):
        """Load a single test image for completion."""
        if self.model is None or self.codebook is None:
            messagebox.showerror("Error", "Initialize the model first (Train/Generate tab).")
            return

        path = filedialog.askopenfilename(
            title="Select a test image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg"), ("All files", "*.*")]
        )
        if not path:
            return

        self.status_var.set("Loading test image...")
        self.root.update()

        try:
            img_size = self.config['image_size']
            channels = self.config['channels']
            # Load as RGB then convert if needed
            pil_img = Image.open(path).convert('RGB')
            pil_img = pil_img.resize((img_size, img_size), Image.Resampling.LANCZOS)
            arr = np.array(pil_img, dtype=np.float32) / 255.0 * 2 - 1
            if channels == 1:
                arr = 0.299 * arr[:,:,0] + 0.587 * arr[:,:,1] + 0.114 * arr[:,:,2]
            self.test_image = arr  # (H,W) or (H,W,3)
            # Encode the test image
            if arr.ndim == 2:
                t = torch.from_numpy(arr).float()
            else:
                t = torch.from_numpy(arr).permute(2,0,1).float()
            self.test_indices = encode_image(t, self.codebook, self.config)
            # Show original in completion tab
            if arr.ndim == 2:
                img_disp = (arr + 1) / 2
            else:
                img_disp = (arr + 1) / 2
            self.show_image_on_label(self.completion_labels[0], img_disp)
            # Clear partial and completed
            self.display_placeholder(self.completion_labels[1])
            self.display_placeholder(self.completion_labels[2])
            self.status_var.set(f"Test image loaded. Set completion % and click 'Complete'.")
            self.complete_btn.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{e}")
            self.status_var.set("Error loading test image.")

    def complete_image(self):
        """Run completion on the loaded test image."""
        if self.test_indices is None or self.model is None:
            messagebox.showerror("Error", "Load a test image and initialize the model first.")
            return

        # Read completion percentage
        percent = self.completion_percent_var.get()
        if percent < 1 or percent > 99:
            messagebox.showerror("Error", "Completion percentage must be between 1 and 99.")
            return
        temp = self.completion_temp_var.get()

        total_patches = len(self.test_indices)
        # Number of patches to keep (top part)
        keep = int(total_patches * (100 - percent) / 100)
        # Ensure at least 1 patch is kept
        keep = max(1, min(keep, total_patches - 1))
        hidden_start = keep  # index where hidden patches begin

        self.status_var.set(f"Completing {percent}% of image...")
        self.root.update()

        # Prepare sequence: start with start_token, then known patches
        vocab_size = self.config['vocab_size']
        start_token = vocab_size
        seq = [start_token] + self.test_indices[:keep]  # known top patches

        # Generate missing patches autoregressively
        model = self.model
        model.eval()
        with torch.no_grad():
            for _ in range(total_patches - keep):
                inp = torch.tensor([seq], dtype=torch.long).to(device)
                logits = model(inp)
                last_logits = logits[0, -1, :] / temp
                probs = torch.softmax(last_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                seq.append(next_token)

        # seq now contains start_token + full sequence of indices (all patches)
        full_indices = seq[1:]  # remove start token

        # Reconstruct the completed image
        completed_img = reconstruct_from_indices(full_indices, self.codebook_np, self.config)

        # Create partial image (visible top patches, bottom blacked out)
        partial_indices = self.test_indices[:]
        # Replace hidden indices with a placeholder (e.g., 0) for visualization
        # We'll create a mask: black out the bottom region
        if self.test_image.ndim == 2:
            partial_img = (self.test_image + 1) / 2  # [0,1]
            mask = np.ones_like(partial_img)
            # Determine rows where hidden patches start
            patches_per_row = self.config['image_size'] // self.config['patch_size']
            start_row = (hidden_start // patches_per_row) * self.config['patch_size']
            mask[start_row:, :] = 0
            partial_img = partial_img * mask
        else:
            partial_img = (self.test_image + 1) / 2
            mask = np.ones_like(partial_img[:,:,0])
            patches_per_row = self.config['image_size'] // self.config['patch_size']
            start_row = (hidden_start // patches_per_row) * self.config['patch_size']
            mask[start_row:, :] = 0
            partial_img = partial_img * mask[:,:,np.newaxis]

        # Display results
        self.show_image_on_label(self.completion_labels[1], partial_img)
        self.show_image_on_label(self.completion_labels[2], completed_img)

        self.status_var.set("Completion finished.")


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()