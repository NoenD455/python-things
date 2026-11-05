import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import colors as mcolors
import os
import threading
import traceback

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_OK = True
except ImportError:
    TORCH_OK = False


if TORCH_OK:
    class ConvAutoencoder(nn.Module):
        def __init__(self, latent_dim, in_channels=3, img_size=64):
            super().__init__()
            self.spatial = img_size // 16
            self.encoder_conv = nn.Sequential(
                nn.Conv2d(in_channels, 32, 4, 2, 1), nn.ReLU(),
                nn.Conv2d(32, 64, 4, 2, 1),          nn.ReLU(),
                nn.Conv2d(64, 128, 4, 2, 1),         nn.ReLU(),
                nn.Conv2d(128, 256, 4, 2, 1),        nn.ReLU(),
            )
            self.feature_size = 256 * self.spatial * self.spatial
            self.fc_enc = nn.Linear(self.feature_size, latent_dim)
            self.fc_dec = nn.Linear(latent_dim, self.feature_size)
            self.decoder_conv = nn.Sequential(
                nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),
                nn.ConvTranspose2d(128, 64, 4, 2, 1),  nn.ReLU(),
                nn.ConvTranspose2d(64, 32, 4, 2, 1),   nn.ReLU(),
                nn.ConvTranspose2d(32, in_channels, 4, 2, 1), nn.Sigmoid(),
            )

        def encode(self, x):
            h = self.encoder_conv(x)
            h = h.reshape(h.size(0), -1) 
            return self.fc_enc(h)

        def decode(self, z):
            h = self.fc_dec(z)
            h = h.reshape(h.size(0), 256, self.spatial, self.spatial)
            return self.decoder_conv(h)

        def forward(self, x):
            z = self.encode(x)
            return self.decode(z), z

    class VAE(nn.Module):
        def __init__(self, latent_dim, in_channels=3, img_size=64):
            super().__init__()
            self.spatial = img_size // 16
            self.encoder_conv = nn.Sequential(
                nn.Conv2d(in_channels, 32, 4, 2, 1), nn.ReLU(),
                nn.Conv2d(32, 64, 4, 2, 1),          nn.ReLU(),
                nn.Conv2d(64, 128, 4, 2, 1),         nn.ReLU(),
                nn.Conv2d(128, 256, 4, 2, 1),        nn.ReLU(),
            )
            self.feature_size = 256 * self.spatial * self.spatial
            self.fc_mu = nn.Linear(self.feature_size, latent_dim)
            self.fc_logvar = nn.Linear(self.feature_size, latent_dim)
            
            self.fc_dec = nn.Linear(latent_dim, self.feature_size)
            self.decoder_conv = nn.Sequential(
                nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),
                nn.ConvTranspose2d(128, 64, 4, 2, 1),  nn.ReLU(),
                nn.ConvTranspose2d(64, 32, 4, 2, 1),   nn.ReLU(),
                nn.ConvTranspose2d(32, in_channels, 4, 2, 1), nn.Sigmoid(),
            )

        def encode(self, x):
            h = self.encoder_conv(x)
            h = h.reshape(h.size(0), -1)
            return self.fc_mu(h), self.fc_logvar(h)

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def decode(self, z):
            h = self.fc_dec(z)
            h = h.reshape(h.size(0), 256, self.spatial, self.spatial)
            return self.decoder_conv(h)

        def forward(self, x):
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            return self.decode(z), mu, logvar


class ImageExplorer:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Exploration Tool (Autoencoder / VAE)")

        if not TORCH_OK:
            messagebox.showerror("Error", "PyTorch is required.\nInstall with: pip install torch")
            self.root.destroy()
            return

        # Data
        self.image_paths = []
        self.images_pil = []
        self.image_tensors = None
        self.image_size = (64, 64)
        self.display_size = (200, 200)

        # Model / latent
        self.model = None
        self.device = None
        self.is_vae = False
        self.latent_dim = 8
        self.latents = None
        self.current_latent = None
        self.training = False
        self.stop_flag = False
        self.loss_history = []
        
        # Importance tracking
        self.imp_dims = [0, 1]
        self.imp_scores = []
        self.ordered_dims = []
        
        # Image properties for sorting
        self.img_props = {}

        # Interpolation state
        self.interp_latent_a = None
        self.interp_latent_b = None
        self.interp_name_a = ""
        self.interp_name_b = ""

        # UI refs
        self.sliders = []
        self.plot_canvas = None
        self.plot_ax = None
        self.current_point_handle = None
        self.reconstructed_image_label = None
        self.slider_canvas = None
        self.slider_frame = None
        self.loss_fig = None
        self.loss_ax = None
        self.loss_canvas = None
        self.recon_fig = None
        self.recon_canvas = None
        self.interp_fig = None
        self.interp_canvas = None
        self.fig = None
        self.ax = None

        # Vars
        self.epochs_var = tk.StringVar(value="100")
        self.batch_var = tk.StringVar(value="16")
        self.lr_var = tk.StringVar(value="0.001")
        self.dims_var = tk.StringVar(value="8")
        self.model_type_var = tk.StringVar(value="Autoencoder")
        self.beta_var = tk.StringVar(value="0.0001") # Updated default
        self.status_var = tk.StringVar(value="Ready. Load images to begin.")
        self.interp_steps_var = tk.StringVar(value="10")
        self.sort_var = tk.StringVar(value="None")

        self.build_ui()

    def build_ui(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.tab_dataset = ttk.Frame(self.notebook)
        self.tab_train = ttk.Frame(self.notebook)
        self.tab_latent = ttk.Frame(self.notebook)
        self.tab_interp = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_dataset, text="Dataset")
        self.notebook.add(self.tab_train, text="Training")
        self.notebook.add(self.tab_latent, text="Latent Space")
        self.notebook.add(self.tab_interp, text="Interpolation")

        self.build_dataset_tab()
        self.build_training_tab()
        self.build_latent_tab()
        self.build_interpolation_tab()

    # ================== Dataset Tab ==================
    def build_dataset_tab(self):
        top = ttk.Frame(self.tab_dataset, padding="5")
        top.pack(fill=tk.X)
        
        ttk.Button(top, text="Add Folder", command=self.add_folder).pack(side=tk.LEFT, padx=5)
        ttk.Button(top, text="Add Images", command=self.add_images).pack(side=tk.LEFT, padx=5)
        ttk.Button(top, text="Remove Selected", command=self.remove_selected).pack(side=tk.LEFT, padx=15)
        ttk.Button(top, text="Clear All", command=self.clear_all_images).pack(side=tk.LEFT, padx=5)

        list_frame = ttk.Frame(self.tab_dataset)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.dataset_listbox = tk.Listbox(list_frame, selectmode=tk.EXTENDED)
        self.dataset_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.dataset_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.dataset_listbox.config(yscrollcommand=scrollbar.set)

    def add_folder(self):
        folder = filedialog.askdirectory(title="Select Image Folder")
        if not folder:
            return
        paths = [os.path.join(folder, f) for f in os.listdir(folder)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        self._add_paths_to_dataset(paths)

    def add_images(self):
        paths = filedialog.askopenfilenames(
            title="Select Images",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")])
        if not paths:
            return
        self._add_paths_to_dataset(list(paths))

    def _add_paths_to_dataset(self, new_paths):
        # Filter duplicates
        added = 0
        for p in new_paths:
            if p not in self.image_paths:
                self.image_paths.append(p)
                added += 1
        if added > 0:
            self._update_dataset()

    def remove_selected(self):
        selected = self.dataset_listbox.curselection()
        if not selected:
            return
        # Delete from highest index to lowest to avoid shifting issues
        for index in sorted(selected, reverse=True):
            del self.image_paths[index]
        self._update_dataset()

    def clear_all_images(self):
        self.image_paths = []
        self._update_dataset()

    def _update_dataset(self):
        self.dataset_listbox.delete(0, tk.END)
        self.images_pil = []
        tensors = []
        
        for p in self.image_paths:
            try:
                img = Image.open(p).convert('RGB').resize(self.image_size, Image.Resampling.LANCZOS)
                self.images_pil.append(img)
                arr = np.array(img).astype(np.float32) / 255.0
                arr = np.transpose(arr, (2, 0, 1))
                tensors.append(arr)
                self.dataset_listbox.insert(tk.END, os.path.basename(p))
            except Exception as e:
                print(f"Error loading {p}: {e}")
                # Remove corrupted image from paths silently to keep things in sync
                self.image_paths.remove(p)

        if tensors:
            self.image_tensors = torch.from_numpy(np.stack(tensors, axis=0))
            self.status_var.set(f"Dataset ready: {len(self.image_paths)} images.")
        else:
            self.image_tensors = None
            self.status_var.set("Dataset empty.")

        # Clear downstream states since data changed
        self.latents = None
        self.clear_latent_tab()


    # ================== Training Tab ==================
    def build_training_tab(self):
        top = ttk.Frame(self.tab_train, padding="5")
        top.pack(fill=tk.X)
        ttk.Button(top, text="Initialize Model", command=self.initialize_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(top, text="Start Training", command=self.start_training).pack(side=tk.LEFT, padx=5)
        self.stop_btn = ttk.Button(top, text="Stop Training", command=self.stop_training_fn, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        ttk.Label(top, textvariable=self.status_var).pack(side=tk.LEFT, padx=20)

        settings = ttk.Frame(self.tab_train, padding="5")
        settings.pack(fill=tk.X)
        
        ttk.Label(settings, text="Model Type:").pack(side=tk.LEFT, padx=5)
        type_combo = ttk.Combobox(settings, textvariable=self.model_type_var, 
                                  values=["Autoencoder", "VAE"], state="readonly", width=12)
        type_combo.pack(side=tk.LEFT, padx=5)

        ttk.Label(settings, text="Latent Dims:").pack(side=tk.LEFT, padx=5)
        ttk.Spinbox(settings, from_=2, to=256, textvariable=self.dims_var, width=5).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(settings, text="Epochs:").pack(side=tk.LEFT, padx=5)
        ttk.Spinbox(settings, from_=1, to=5000, textvariable=self.epochs_var, width=5).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(settings, text="Batch Size:").pack(side=tk.LEFT, padx=5)
        ttk.Spinbox(settings, from_=1, to=512, textvariable=self.batch_var, width=5).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(settings, text="Learning Rate:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(settings, textvariable=self.lr_var, width=8).pack(side=tk.LEFT, padx=5)

        # VAE specific
        self.beta_frame = ttk.Frame(settings)
        self.beta_frame.pack(side=tk.LEFT, padx=5)
        ttk.Label(self.beta_frame, text="Beta (KL):").pack(side=tk.LEFT)
        ttk.Entry(self.beta_frame, textvariable=self.beta_var, width=6).pack(side=tk.LEFT, padx=2)
        
        def on_type_change(e):
            if self.model_type_var.get() == "VAE":
                self.beta_frame.pack(side=tk.LEFT, padx=5)
            else:
                self.beta_frame.pack_forget()
        type_combo.bind("<<ComboboxSelected>>", on_type_change)
        if self.model_type_var.get() != "VAE":
            self.beta_frame.pack_forget()

        body = ttk.Frame(self.tab_train)
        body.pack(fill=tk.BOTH, expand=True)

        loss_frame = ttk.LabelFrame(body, text="Training Loss")
        loss_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.loss_fig, self.loss_ax = plt.subplots(figsize=(6, 3))
        self.loss_canvas = FigureCanvasTkAgg(self.loss_fig, master=loss_frame)
        self.loss_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.loss_fig.tight_layout()

        recon_frame = ttk.LabelFrame(body, text="Reconstruction Preview")
        recon_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.recon_fig = plt.Figure(figsize=(8, 2))
        self.recon_canvas = FigureCanvasTkAgg(self.recon_fig, master=recon_frame)
        self.recon_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def initialize_model(self):
        if self.image_tensors is None:
            messagebox.showerror("Error", "Load a dataset first.")
            return
        if self.training:
            messagebox.showerror("Error", "Cannot initialize while training.")
            return
            
        try:
            latent_dim = int(self.dims_var.get())
            self.is_vae = (self.model_type_var.get() == "VAE")
            self.latent_dim = latent_dim
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            if self.is_vae:
                self.model = VAE(latent_dim).to(self.device)
            else:
                self.model = ConvAutoencoder(latent_dim).to(self.device)
            
            self.latents = None
            self.loss_history = []
            self.loss_ax.clear()
            self.loss_canvas.draw()
            self.recon_fig.clear()
            self.recon_canvas.draw()
            self.clear_latent_tab()
            
            self.status_var.set(f"{self.model_type_var.get()} Initialized ({self.latent_dim} dims) on {self.device.type.upper()}.")
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Error", f"Initialization failed: {e}")
            self.model = None

    def start_training(self):
        if self.image_tensors is None:
            messagebox.showerror("Error", "Load a dataset first.")
            return
        if self.model is None:
            messagebox.showerror("Error", "Please click 'Initialize Model' first.")
            return
        if self.training:
            return
            
        try:
            epochs = int(self.epochs_var.get())
            batch_size = int(self.batch_var.get())
            lr = float(self.lr_var.get())
            beta = float(self.beta_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid training parameters.")
            return

        self.loss_history = []
        self.stop_flag = False
        self.training = True
        self.stop_btn.config(state=tk.NORMAL)
        self.status_var.set("Training...")

        def run():
            try:
                optimizer = optim.Adam(self.model.parameters(), lr=lr)
                criterion = nn.MSELoss()
                tensors = self.image_tensors
                loader = DataLoader(TensorDataset(tensors),
                                    batch_size=min(batch_size, len(tensors)),
                                    shuffle=True)

                for epoch in range(epochs):
                    if self.stop_flag:
                        break
                    self.model.train()
                    total, nb = 0.0, 0
                    for (batch,) in loader:
                        batch = batch.to(self.device)
                        optimizer.zero_grad()
                        
                        if self.is_vae:
                            recon, mu, logvar = self.model(batch)
                            recon_loss = criterion(recon, batch)
                            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                            kld = kld / batch.size(0)
                            loss = recon_loss + beta * kld
                        else:
                            recon, _ = self.model(batch)
                            loss = criterion(recon, batch)
                            
                        loss.backward()
                        optimizer.step()
                        total += loss.item(); nb += 1
                        
                    self.loss_history.append(total / max(1, nb))
                    self.root.after(0, lambda e=epoch: self._update_loss_plot(e))

                    if (epoch + 1) % 10 == 0 or epoch == 0:
                        with torch.no_grad():
                            n_show = min(8, len(tensors))
                            sample = tensors[:n_show].to(self.device)
                            if self.is_vae:
                                recons, _, _ = self.model(sample)
                            else:
                                recons, _ = self.model(sample)
                            recons_np = recons.cpu().numpy()
                            sample_np = tensors[:n_show].numpy()
                        self.root.after(0, lambda r=recons_np, s=sample_np:
                                        self._update_recon_preview(r, s))

                self.model.eval()
                with torch.no_grad():
                    if self.is_vae:
                        mu, _ = self.model.encode(tensors.to(self.device))
                        latents = mu.cpu().numpy()
                    else:
                        latents = self.model.encode(tensors.to(self.device)).cpu().numpy()
                self.latents = latents
                self.current_latent = latents.mean(axis=0)
                self.root.after(0, self._on_training_done)
            except Exception as e:
                self.training = False
                traceback.print_exc()
                self.root.after(0, lambda err=e: messagebox.showerror("Error", f"Training failed: {err}"))
                self.root.after(0, lambda: self.status_var.set("Training failed"))
                self.root.after(0, lambda: self.stop_btn.config(state=tk.DISABLED))

        threading.Thread(target=run, daemon=True).start()

    def stop_training_fn(self):
        self.stop_flag = True

    def _update_loss_plot(self, epoch):
        self.loss_ax.clear()
        self.loss_ax.plot(self.loss_history, 'b-')
        self.loss_ax.set_xlabel('Epoch')
        self.loss_ax.set_ylabel('Loss')
        self.loss_ax.set_title(f'Loss (epoch {epoch + 1}, last={self.loss_history[-1]:.5f})')
        try:
            self.loss_ax.set_yscale('log')
        except Exception:
            pass
        self.loss_fig.tight_layout()
        self.loss_canvas.draw_idle()

    def _update_recon_preview(self, recons, samples):
        self.recon_fig.clear()
        n = len(recons)
        for i in range(n):
            ax = self.recon_fig.add_subplot(2, n, i + 1)
            ax.imshow(np.transpose(samples[i], (1, 2, 0)))
            ax.axis('off')
            if i == 0:
                ax.set_title('Original', fontsize=8)
            ax = self.recon_fig.add_subplot(2, n, n + i + 1)
            ax.imshow(np.clip(np.transpose(recons[i], (1, 2, 0)), 0, 1))
            ax.axis('off')
            if i == 0:
                ax.set_title('Reconstructed', fontsize=8)
        self.recon_fig.tight_layout()
        self.recon_canvas.draw_idle()

    def _on_training_done(self):
        self.training = False
        self.stop_btn.config(state=tk.DISABLED)
        self.status_var.set(f"Training done. Latent dim = {self.latent_dim}")
        
        self._compute_feature_importance()
        self._compute_image_properties()
        self.populate_latent_tab()

    def _compute_feature_importance(self):
        if self.model is None or self.latents is None:
            self.imp_dims = [0, 1]
            self.imp_scores = [1.0] * self.latent_dim
            return

        self.model.eval()
        base_z = torch.from_numpy(self.latents.mean(axis=0)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            base_img = self.model.decode(base_z)

        scores = []
        stds = self.latents.std(axis=0)
        stds[stds == 0] = 1.0 

        for i in range(self.latent_dim):
            z_perturbed = base_z.clone()
            z_perturbed[0, i] += stds[i] 
            with torch.no_grad():
                perturbed_img = self.model.decode(z_perturbed)
            mse = torch.mean((base_img - perturbed_img) ** 2).item()
            scores.append(mse)

        self.imp_scores = np.array(scores)
        sorted_indices = np.argsort(self.imp_scores)[::-1]
        self.imp_dims = [sorted_indices[0], sorted_indices[1]]

    def _compute_image_properties(self):
        self.img_props = {
            "Total Pixel Change": [], "Hue": [], "Brightness": [], 
            "Saturation": [], "Red": [], "Green": [], "Blue": []
        }
        if self.image_tensors is None:
            return
            
        np_imgs = self.image_tensors.numpy()
        for i in range(len(np_imgs)):
            img_hwc = np.transpose(np_imgs[i], (1, 2, 0))
            self.img_props["Total Pixel Change"].append(np.var(img_hwc))
            
            hsv = mcolors.rgb_to_hsv(img_hwc)
            self.img_props["Hue"].append(np.mean(hsv[..., 0]))
            self.img_props["Saturation"].append(np.mean(hsv[..., 1]))
            self.img_props["Brightness"].append(np.mean(hsv[..., 2]))
            
            self.img_props["Red"].append(np.mean(img_hwc[..., 0]))
            self.img_props["Green"].append(np.mean(img_hwc[..., 1]))
            self.img_props["Blue"].append(np.mean(img_hwc[..., 2]))

    # ================== Latent Tab ==================
    def build_latent_tab(self):
        top = ttk.Frame(self.tab_latent, padding="5")
        top.pack(fill=tk.X)
        
        ttk.Button(top, text="Load Image to Latent", command=self.load_image_to_latent).pack(side=tk.LEFT, padx=5)
        ttk.Button(top, text="Reset Sliders", command=self.reset_sliders).pack(side=tk.LEFT, padx=5)
        ttk.Button(top, text="Invert Sliders", command=self.invert_sliders).pack(side=tk.LEFT, padx=5)
        ttk.Button(top, text="Sample Gaussian", command=self.sample_gaussian).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(top, text="Sort / Color by:").pack(side=tk.LEFT, padx=10)
        self.sort_combo = ttk.Combobox(top, textvariable=self.sort_var, 
                                       values=["None", "Total Pixel Change", "Hue", "Brightness", 
                                               "Saturation", "Red", "Green", "Blue"],
                                       state="readonly", width=18)
        self.sort_combo.pack(side=tk.LEFT, padx=5)
        self.sort_combo.bind("<<ComboboxSelected>>", lambda e: self.populate_latent_tab())

        main = ttk.Frame(self.tab_latent)
        main.pack(fill=tk.BOTH, expand=True)

        left_frame = ttk.Frame(main, width=400)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        left_frame.pack_propagate(False)

        self.slider_canvas = tk.Canvas(left_frame, highlightthickness=0)
        self.slider_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(left_frame, orient=tk.VERTICAL, command=self.slider_canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.slider_canvas.configure(yscrollcommand=scrollbar.set)

        self.slider_frame = ttk.Frame(self.slider_canvas)
        self.slider_canvas.create_window((0, 0), window=self.slider_frame, anchor="nw")
        self.slider_frame.bind("<Configure>",
                               lambda e: self.slider_canvas.configure(
                                   scrollregion=self.slider_canvas.bbox("all")))

        right_frame = ttk.Frame(main)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        plot_frame = ttk.Frame(right_frame)
        plot_frame.pack(fill=tk.BOTH, expand=True)

        image_frame = ttk.Frame(right_frame)
        image_frame.pack(fill=tk.X, pady=5)
        ttk.Label(image_frame, text="Decoded Image:").pack()
        self.reconstructed_image_label = ttk.Label(image_frame)
        self.reconstructed_image_label.pack()

        self.fig, self.ax = plt.subplots(figsize=(5, 4))
        self.plot_canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.plot_ax = self.ax
        self.fig.tight_layout()
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)

    def clear_latent_tab(self):
        for w in self.slider_frame.winfo_children():
            w.destroy()
        self.sliders = []
        
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        self.plot_canvas.draw()
        
        self.reconstructed_image_label.config(image='')

    def populate_latent_tab(self):
        if self.latents is None:
            return
        for w in self.slider_frame.winfo_children():
            w.destroy()
        self.sliders = []

        # Determine slider order
        sort_by = self.sort_var.get()
        if sort_by != "None" and sort_by in self.img_props:
            target = np.array(self.img_props[sort_by])
            correlations = []
            for d in range(self.latent_dim):
                dim_vals = self.latents[:, d]
                if np.std(dim_vals) == 0 or np.std(target) == 0:
                    corr = 0
                else:
                    corr = np.corrcoef(dim_vals, target)[0, 1]
                correlations.append(np.abs(corr))
            self.ordered_dims = np.argsort(correlations)[::-1]
        else:
            self.ordered_dims = np.argsort(self.imp_scores)[::-1]

        mins, maxs = self.latents.min(0), self.latents.max(0)
        if self.is_vae:
            mins = np.minimum(mins, -3.0)
            maxs = np.maximum(maxs, 3.0)
            
        rng = maxs - mins
        margin = 0.1 * rng
        margin[margin == 0] = 0.1
        lows, highs = mins - margin, maxs + margin

        for i in range(self.latent_dim):
            real_dim = self.ordered_dims[i]
            frame = ttk.Frame(self.slider_frame)
            frame.pack(fill=tk.X, pady=2)
            
            is_important = real_dim in self.imp_dims
            label_text = f"★ Dim {real_dim + 1}" if is_important else f"Dim {real_dim + 1}"
            label_fg = "green" if is_important else "black"
            
            lbl = ttk.Label(frame, text=label_text, width=8, foreground=label_fg)
            lbl.pack(side=tk.LEFT)
            
            slider = tk.Scale(frame, from_=float(lows[real_dim]), to=float(highs[real_dim]),
                              resolution=0.01, orient=tk.HORIZONTAL,
                              length=250, showvalue=1)
            slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
            slider.set(float(self.latents[:, real_dim].mean()))
            self.sliders.append(slider)
            slider.bind("<Motion>", self.on_slider_move)
            slider.bind("<ButtonRelease-1>", self.on_slider_move)

        self.slider_frame.update_idletasks()
        self.slider_canvas.configure(scrollregion=self.slider_canvas.bbox("all"))

        self.current_latent = self.latents.mean(axis=0)
        self.update_plot()
        self.update_reconstruction()

    def update_plot(self):
        if self.latents is None:
            return
        
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
            
        d1, d2 = self.imp_dims
        x = self.latents[:, d1]
        y = self.latents[:, d2]
        
        sort_by = self.sort_var.get()
        if sort_by != "None" and sort_by in self.img_props and len(self.img_props[sort_by]) == len(x):
            c_values = self.img_props[sort_by]
            sc = self.ax.scatter(x, y, c=c_values, cmap='viridis', alpha=0.8, s=40, picker=True)
            self.fig.colorbar(sc, ax=self.ax, label=sort_by)
        else:
            self.ax.scatter(x, y, c='blue', alpha=0.6, s=30, picker=True)
        
        score1 = self.imp_scores[d1]
        score2 = self.imp_scores[d2]
        self.ax.set_xlabel(f'Dim {d1 + 1} (Importance: {score1:.4f})')
        self.ax.set_ylabel(f'Dim {d2 + 1} (Importance: {score2:.4f})')
        self.ax.set_title('Latent Space (Top 2 Important Dimensions)')

        if self.current_latent is not None:
            cx = self.current_latent[d1]
            cy = self.current_latent[d2]
            self.current_point_handle = self.ax.scatter(cx, cy, c='red', s=100,
                                                        marker='X', edgecolors='white', zorder=5)
        self.fig.tight_layout()
        self.plot_canvas.draw()

    def invert_sliders(self):
        if not self.sliders:
            return
        for i, real_dim in enumerate(self.ordered_dims):
            slider = self.sliders[i]
            low = float(slider.cget('from'))
            high = float(slider.cget('to'))
            val = slider.get()
            
            # Invert across the midpoint of the bounds
            inverted_val = (low + high) - val
            slider.set(float(np.clip(inverted_val, low, high)))
            
        self.current_latent = self.get_latent_from_sliders()
        self.on_slider_move()

    def sample_gaussian(self):
        if self.model is None or self.latents is None:
            return
        
        # Sample from empirical distribution
        if len(self.latents) > 1:
            mu = self.latents.mean(axis=0)
            cov = np.cov(self.latents, rowvar=False)
            cov += np.eye(self.latent_dim) * 1e-6
            z = np.random.multivariate_normal(mu, cov)
        else:
            z = np.random.randn(self.latent_dim)
            
        for i, real_dim in enumerate(self.ordered_dims):
            slider = self.sliders[i]
            low = float(slider.cget('from'))
            high = float(slider.cget('to'))
            slider.set(float(np.clip(z[real_dim], low, high)))
        
        self.current_latent = self.get_latent_from_sliders()
        self.on_slider_move()

    def get_latent_from_sliders(self):
        latent = np.zeros(self.latent_dim)
        for i, real_dim in enumerate(self.ordered_dims):
            latent[real_dim] = self.sliders[i].get()
        return latent

    def set_sliders_from_latent(self, latent):
        for i, real_dim in enumerate(self.ordered_dims):
            slider = self.sliders[i]
            low = float(slider.cget('from'))
            high = float(slider.cget('to'))
            slider.set(float(np.clip(latent[real_dim], low, high)))

    def load_image_to_latent(self):
        if self.model is None:
            messagebox.showerror("Error", "Initialize and train a model first.")
            return
            
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")])
        if not file_path:
            return

        try:
            img = Image.open(file_path).convert('RGB').resize(self.image_size, Image.Resampling.LANCZOS)
            arr = np.array(img).astype(np.float32) / 255.0
            arr = np.transpose(arr, (2, 0, 1))
            tensor = torch.from_numpy(arr).unsqueeze(0).to(self.device)
            
            self.model.eval()
            with torch.no_grad():
                if self.is_vae:
                    mu, _ = self.model.encode(tensor)
                    latent = mu.cpu().numpy()[0]
                else:
                    latent = self.model.encode(tensor).cpu().numpy()[0]
                    
            self.set_sliders_from_latent(latent)
            self.current_latent = self.get_latent_from_sliders()
            self.on_slider_move()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to encode image: {e}")

    def on_slider_move(self, event=None):
        if not self.sliders or self.model is None:
            return
        latent = self.get_latent_from_sliders()
        self.current_latent = latent

        if self.current_point_handle is not None and self.latents is not None:
            d1, d2 = self.imp_dims
            cx = latent[d1]
            cy = latent[d2]
            self.current_point_handle.set_offsets([[cx, cy]])
            self.plot_canvas.draw_idle()

        self.update_reconstruction()

    def update_reconstruction(self):
        if self.model is None or self.current_latent is None:
            return
        try:
            with torch.no_grad():
                z = torch.from_numpy(self.current_latent).float().unsqueeze(0).to(self.device)
                out = self.model.decode(z).squeeze(0).cpu().numpy()
            out = np.clip(np.transpose(out, (1, 2, 0)), 0, 1)
            img = Image.fromarray((out * 255).astype(np.uint8)).resize(
                self.display_size, Image.Resampling.LANCZOS)
            imgtk = ImageTk.PhotoImage(img)
            self.reconstructed_image_label.config(image=imgtk)
            self.reconstructed_image_label.image = imgtk
        except Exception as e:
            pass

    def reset_sliders(self):
        if self.latents is None:
            return
        mean_vals = self.latents.mean(axis=0)
        self.set_sliders_from_latent(mean_vals)
        self.current_latent = self.get_latent_from_sliders()
        self.on_slider_move()

    def on_pick(self, event):
        if self.latents is None:
            return
        ind = event.ind[0]
        latent = self.latents[ind]
        self.set_sliders_from_latent(latent)
        self.current_latent = self.get_latent_from_sliders()
        self.on_slider_move()

    # ================== Interpolation Tab ==================
    def build_interpolation_tab(self):
        top = ttk.Frame(self.tab_interp, padding="5")
        top.pack(fill=tk.X)
        
        ttk.Button(top, text="Load Image A", command=lambda: self.load_interp_image('A')).pack(side=tk.LEFT, padx=5)
        self.interp_a_label = ttk.Label(top, text="None", width=20)
        self.interp_a_label.pack(side=tk.LEFT, padx=5)

        ttk.Button(top, text="Load Image B", command=lambda: self.load_interp_image('B')).pack(side=tk.LEFT, padx=15)
        self.interp_b_label = ttk.Label(top, text="None", width=20)
        self.interp_b_label.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(top, text="Steps:").pack(side=tk.LEFT, padx=5)
        ttk.Spinbox(top, from_=2, to=50, textvariable=self.interp_steps_var, width=5).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(top, text="Generate Interpolation", command=self.generate_interpolation).pack(side=tk.LEFT, padx=15)

        self.interp_fig, _ = plt.subplots(figsize=(10, 2))
        self.interp_canvas = FigureCanvasTkAgg(self.interp_fig, master=self.tab_interp)
        self.interp_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.interp_fig.tight_layout()

    def load_interp_image(self, slot):
        if self.model is None:
            messagebox.showerror("Error", "Train a model first.")
            return
            
        file_path = filedialog.askopenfilename(
            title=f"Select Image {slot}",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")])
            
        if not file_path:
            return

        try:
            img = Image.open(file_path).convert('RGB').resize(self.image_size, Image.Resampling.LANCZOS)
            arr = np.array(img).astype(np.float32) / 255.0
            arr = np.transpose(arr, (2, 0, 1))
            tensor = torch.from_numpy(arr).unsqueeze(0).to(self.device)
            
            self.model.eval()
            with torch.no_grad():
                if self.is_vae:
                    mu, _ = self.model.encode(tensor)
                    z = mu.cpu()
                else:
                    z = self.model.encode(tensor).cpu()

            if slot == 'A':
                self.interp_latent_a = z
                self.interp_name_a = os.path.basename(file_path)
                self.interp_a_label.config(text=self.interp_name_a)
            else:
                self.interp_latent_b = z
                self.interp_name_b = os.path.basename(file_path)
                self.interp_b_label.config(text=self.interp_name_b)
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")

    def generate_interpolation(self):
        if self.model is None or self.interp_latent_a is None or self.interp_latent_b is None:
            messagebox.showerror("Error", "Please load both Image A and Image B.")
            return

        try:
            steps = int(self.interp_steps_var.get())
            if steps < 2:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Steps must be >= 2.")
            return

        try:
            self.model.eval()
            z1 = self.interp_latent_a.to(self.device)
            z2 = self.interp_latent_b.to(self.device)
            
            alphas = np.linspace(0, 1, steps)
            imgs = []
            
            with torch.no_grad():
                for a in alphas:
                    z = (1 - a) * z1 + a * z2
                    out = self.model.decode(z).squeeze(0).cpu().numpy()
                    imgs.append(np.clip(np.transpose(out, (1, 2, 0)), 0, 1))

            self.interp_fig.clear()
            for i, img in enumerate(imgs):
                ax = self.interp_fig.add_subplot(1, steps, i + 1)
                ax.imshow(img)
                ax.axis('off')
                ax.set_title(f"{alphas[i]:.2f}", fontsize=8)
            self.interp_fig.tight_layout()
            self.interp_canvas.draw()
        except Exception as e:
            messagebox.showerror("Error", f"Interpolation failed: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageExplorer(root)
    root.geometry("1250x850")
    root.mainloop()