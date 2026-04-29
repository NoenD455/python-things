import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os
import time
import threading
import queue
from collections import Counter
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import traceback

# ==================== CPU OPTIMIZATION ====================
torch.set_num_threads(2)
torch.set_num_interop_threads(2)
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'

print(f"PyTorch threads: {torch.get_num_threads()}")

# ==================== GRU MODEL ====================
class TextGRU(nn.Module):
    def __init__(self, vocab_size, embed_size=128, hidden_size=256, n_layers=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(
            embed_size, 
            hidden_size, 
            n_layers,
            batch_first=True,
            dropout=0.1 if n_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        self.init_weights()
        print(f"GRU Model: {sum(p.numel() for p in self.parameters()):,} params")
    
    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
    
    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        
        if hidden is None:
            hidden = self.init_hidden(batch_size)
        
        embedded = self.embedding(x)
        output, hidden = self.gru(embedded, hidden)
        output = output.reshape(-1, self.hidden_size)
        output = self.fc(output)
        output = output.view(batch_size, -1, self.vocab_size)
        
        return output, hidden
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return weight.new_zeros(self.n_layers, batch_size, self.hidden_size)
    
    def generate_step(self, input_token, hidden=None, temperature=0.8):
        input_tensor = torch.tensor([[input_token]], dtype=torch.long)
        
        with torch.no_grad():
            embedded = self.embedding(input_tensor)
            if hidden is None:
                hidden = self.init_hidden(1)
            
            output, hidden = self.gru(embedded, hidden)
            output = output.reshape(-1, self.hidden_size)
            logits = self.fc(output)[0] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
        
        return next_token, hidden

    def save_model(self, path):
        """Save model state only (processor is fixed, so not saved separately)"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'vocab_size': self.vocab_size,
            'embed_size': self.embed_size,
            'hidden_size': self.hidden_size,
            'n_layers': self.n_layers
        }, path)
        print(f"Model saved to {path}")

    @classmethod
    def load_model(cls, path):
        """Load model from file (processor not included)"""
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        
        model = cls(
            vocab_size=checkpoint['vocab_size'],
            embed_size=checkpoint['embed_size'],
            hidden_size=checkpoint['hidden_size'],
            n_layers=checkpoint['n_layers']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"Model loaded from {path}")
        return model

# ==================== FIXED VOCABULARY TEXT PROCESSOR ====================
class TextProcessor:
    """Character-level processor with a fixed, predefined vocabulary."""
    def __init__(self, seq_len=100):
        self.seq_len = seq_len
        self.char2idx, self.idx2char = self._build_fixed_vocab()
        self.tokens = []  # will be filled later from loaded texts

    def _build_fixed_vocab(self):
        """Create a fixed set of allowed characters:
           Latin, Cyrillic, Greek, punctuation, digits,
           plus important whitespace: newline, carriage return, tab,
           and common typographic extras.
        """
        # Special tokens (must be first)
        specials = ['<PAD>', '<UNK>']
        
        # Collect all printable characters from desired Unicode ranges
        allowed_chars = []
        
        # 1. Basic Latin (space to ~) – includes letters, digits, punctuation
        for code in range(32, 127):
            allowed_chars.append(chr(code))
        
        # 2. Cyrillic (U+0400 to U+04FF)
        for code in range(0x0400, 0x0500):
            try:
                ch = chr(code)
                if ch.isprintable():
                    allowed_chars.append(ch)
            except:
                pass
        
        # 3. Greek (U+0370 to U+03FF)
        for code in range(0x0370, 0x0400):
            try:
                ch = chr(code)
                if ch.isprintable():
                    allowed_chars.append(ch)
            except:
                pass
        
        # 4. Essential whitespace characters (not in printable ASCII range)
        whitespace = ['\n', '\r', '\t']
        for ch in whitespace:
            if ch not in allowed_chars:
                allowed_chars.append(ch)
        
        # 5. Additional punctuation/symbols often found in text
        extra = '—–“”‘’…•'
        for ch in extra:
            if ch not in allowed_chars:
                allowed_chars.append(ch)
        
        # Remove duplicates and sort for consistency
        allowed_chars = sorted(set(allowed_chars))
        
        # Build mapping: specials first, then all other chars
        char2idx = {}
        idx2char = {}
        for idx, token in enumerate(specials):
            char2idx[token] = idx
            idx2char[idx] = token
        for idx, ch in enumerate(allowed_chars, start=len(specials)):
            char2idx[ch] = idx
            idx2char[idx] = ch
        
        return char2idx, idx2char

    def text_to_tokens(self, text):
        """Convert a string to a list of token IDs (unknown chars become <UNK>=1)."""
        return [self.char2idx.get(c, 1) for c in text]

    def tokens_to_text(self, tokens):
        """Convert a list of token IDs back to a string, skipping special tokens."""
        result = []
        for idx in tokens:
            ch = self.idx2char.get(idx, '')
            if ch and ch not in ('<PAD>', '<UNK>'):
                result.append(ch)
        return ''.join(result)

    def get_batch(self, batch_size):
        """Return a random batch of (input, target) sequences from self.tokens."""
        if len(self.tokens) <= self.seq_len + 1:
            # Not enough data – pad with zeros
            seq = self.tokens + [0] * (self.seq_len + 1 - len(self.tokens))
            batch_x = [seq[:self.seq_len]] * batch_size
            batch_y = [seq[1:self.seq_len + 1]] * batch_size
        else:
            batch_x = []
            batch_y = []
            for _ in range(batch_size):
                start = random.randint(0, len(self.tokens) - self.seq_len - 1)
                batch_x.append(self.tokens[start:start + self.seq_len])
                batch_y.append(self.tokens[start + 1:start + self.seq_len + 1])
        
        return torch.tensor(batch_x), torch.tensor(batch_y)

# ==================== GUI WITH MULTIPLE FILE SELECTION ====================
class MultiFileTextGenerator:
    def __init__(self, root):
        self.root = root
        self.root.title("Text Pattern Learner - RNN (Fixed Vocabulary)")
        self.root.geometry("1000x800")
        
        # Model and data
        self.model = None
        self.processor = TextProcessor()  # will be recreated after loading files
        self.training = False
        self.generating = False
        self.device = torch.device('cpu')
        self.loaded_files = []
        self.loaded_texts = []
        
        # Queues
        self.generation_queue = queue.Queue()
        self.update_queue = queue.Queue()
        
        # Build UI
        self.setup_ui()
        
        # Start update thread
        threading.Thread(target=self.update_ui_thread, daemon=True).start()
        
        print("Ready! Load multiple text files to start.")
        print(f"Fixed vocabulary size: {len(self.processor.char2idx)}")
    
    def setup_ui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill='both', expand=True)
        
        self.setup_file_tab(notebook)
        self.setup_train_tab(notebook)
        self.setup_generate_tab(notebook)
        
        self.status = tk.StringVar(value="Ready - Load text files to begin")
        status_bar = ttk.Label(main_frame, textvariable=self.status, relief='sunken')
        status_bar.pack(fill='x', pady=(5, 0))
    
    def setup_file_tab(self, notebook):
        frame = ttk.Frame(notebook)
        notebook.add(frame, text='Files')
        
        ttk.Label(frame, text="Text File Management", font='Arial 12 bold').pack(pady=(10,5))
        
        list_frame = ttk.LabelFrame(frame, text="Loaded Files", padding=10)
        list_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        list_container = ttk.Frame(list_frame)
        list_container.pack(fill='both', expand=True)
        
        columns = ('#', 'Filename', 'Size', 'Chars', 'Status')
        self.file_tree = ttk.Treeview(list_container, columns=columns, show='headings', height=10)
        
        self.file_tree.heading('#', text='#')
        self.file_tree.heading('Filename', text='Filename')
        self.file_tree.heading('Size', text='Size')
        self.file_tree.heading('Chars', text='Chars')
        self.file_tree.heading('Status', text='Status')
        
        self.file_tree.column('#', width=40, anchor='center')
        self.file_tree.column('Filename', width=300, anchor='w')
        self.file_tree.column('Size', width=80, anchor='center')
        self.file_tree.column('Chars', width=80, anchor='center')
        self.file_tree.column('Status', width=100, anchor='center')
        
        scrollbar = ttk.Scrollbar(list_container, orient='vertical', command=self.file_tree.yview)
        self.file_tree.configure(yscrollcommand=scrollbar.set)
        
        self.file_tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        control_frame = ttk.Frame(list_frame)
        control_frame.pack(fill='x', pady=(10,0))
        
        ttk.Button(control_frame, text="Add Files", command=self.add_files).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Add Folder", command=self.add_folder).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Remove Selected", command=self.remove_file).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Clear All", command=self.clear_files).pack(side='left', padx=5)
        
        preview_frame = ttk.LabelFrame(frame, text="File Preview", padding=10)
        preview_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        preview_top = ttk.Frame(preview_frame)
        preview_top.pack(fill='x', pady=(0,5))
        
        ttk.Label(preview_top, text="Preview file:").pack(side='left', padx=5)
        self.preview_var = tk.StringVar()
        self.preview_combo = ttk.Combobox(preview_top, textvariable=self.preview_var, state='readonly', width=50)
        self.preview_combo.pack(side='left', padx=5)
        self.preview_combo.bind('<<ComboboxSelected>>', self.update_preview)
        
        self.preview_text = scrolledtext.ScrolledText(preview_frame, height=12, width=80, wrap='word')
        self.preview_text.pack(fill='both', expand=True)
        
        stats_frame = ttk.LabelFrame(frame, text="Combined Statistics", padding=10)
        stats_frame.pack(fill='x', padx=10, pady=10)
        
        self.stats_label = ttk.Label(stats_frame, text="No files loaded", font='Arial 10')
        self.stats_label.pack(anchor='w')
        
        ttk.Button(frame, text="Process All Files & Build Token List", 
                  command=self.process_files, style='Accent.TButton').pack(pady=10)
    
    def setup_train_tab(self, notebook):
        frame = ttk.Frame(notebook)
        notebook.add(frame, text='Train')
        
        ttk.Label(frame, text="Training Settings:", font='Arial 10 bold').pack(anchor='w', pady=(10,5))
        
        settings_frame = ttk.Frame(frame)
        settings_frame.pack(fill='x', pady=5)
        
        # Sequence length
        ttk.Label(settings_frame, text="Seq Length:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.seq_len = tk.IntVar(value=100)
        ttk.Entry(settings_frame, textvariable=self.seq_len, width=8).grid(row=0, column=1, padx=5, pady=2)
        
        # Batch size
        ttk.Label(settings_frame, text="Batch Size:").grid(row=0, column=2, sticky='w', padx=5, pady=2)
        self.batch_size = tk.IntVar(value=8)
        ttk.Entry(settings_frame, textvariable=self.batch_size, width=8).grid(row=0, column=3, padx=5, pady=2)
        
        # Epochs
        ttk.Label(settings_frame, text="Epochs:").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        self.epochs = tk.IntVar(value=50)
        ttk.Entry(settings_frame, textvariable=self.epochs, width=8).grid(row=1, column=1, padx=5, pady=2)
        
        # Learning rate
        ttk.Label(settings_frame, text="Learning Rate:").grid(row=1, column=2, sticky='w', padx=5, pady=2)
        self.lr = tk.StringVar(value="0.001")
        ttk.Entry(settings_frame, textvariable=self.lr, width=8).grid(row=1, column=3, padx=5, pady=2)
        
        # Model size
        ttk.Label(settings_frame, text="Model Size:").grid(row=2, column=0, sticky='w', padx=5, pady=2)
        self.model_size = tk.StringVar(value="Medium")
        size_combo = ttk.Combobox(settings_frame, textvariable=self.model_size, 
                                 values=["Small", "Medium", "Large"], width=8)
        size_combo.grid(row=2, column=1, padx=5, pady=2)
        
        # Save/Load buttons
        save_load_frame = ttk.Frame(frame)
        save_load_frame.pack(fill='x', pady=10)
        
        ttk.Button(save_load_frame, text="Save Model", command=self.save_model).pack(side='left', padx=5)
        ttk.Button(save_load_frame, text="Load Model", command=self.load_model).pack(side='left', padx=5)
        
        # ===== SEPARATE INIT AND TRAIN BUTTONS =====
        init_train_frame = ttk.Frame(frame)
        init_train_frame.pack(pady=10)
        
        self.init_btn = ttk.Button(init_train_frame, text="1. Initialize Model", 
                                  command=self.init_model, style='Accent.TButton', state='disabled')
        self.init_btn.pack(side='left', padx=5)
        
        self.train_btn = ttk.Button(init_train_frame, text="2. Start Training", 
                                   command=self.start_training, state='disabled')
        self.train_btn.pack(side='left', padx=5)
        
        self.stop_btn = ttk.Button(init_train_frame, text="Stop", 
                                  command=self.stop_training, state='disabled')
        self.stop_btn.pack(side='left', padx=5)
        
        # Progress
        self.progress = ttk.Progressbar(frame, mode='indeterminate', length=400)
        self.progress.pack(pady=5)
        
        # Loss display
        loss_frame = ttk.Frame(frame)
        loss_frame.pack(pady=5)
        
        ttk.Label(loss_frame, text="Current Loss:").pack(side='left')
        self.loss_label = ttk.Label(loss_frame, text="0.0000", font='Arial 10 bold', foreground='blue')
        self.loss_label.pack(side='left', padx=5)
        
        # Log
        ttk.Label(frame, text="Training Log:", font='Arial 10 bold').pack(anchor='w', pady=(10,5))
        
        self.log = scrolledtext.ScrolledText(frame, height=12, width=80, state='disabled')
        self.log.pack(fill='both', expand=True, pady=5)
    
    def setup_generate_tab(self, notebook):
        frame = ttk.Frame(notebook)
        notebook.add(frame, text='Generate')
        
        ttk.Label(frame, text="Generation Settings:", font='Arial 10 bold').pack(anchor='w', pady=(10,5))
        
        gen_frame = ttk.Frame(frame)
        gen_frame.pack(fill='x', pady=5)
        
        # Seed text
        ttk.Label(gen_frame, text="Seed Text:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.seed_text = tk.StringVar(value="The ")
        seed_entry = ttk.Entry(gen_frame, textvariable=self.seed_text, width=40)
        seed_entry.grid(row=0, column=1, columnspan=3, padx=5, pady=2, sticky='ew')
        
        # Length
        ttk.Label(gen_frame, text="Length:").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        self.gen_length = tk.IntVar(value=500)
        ttk.Scale(gen_frame, from_=50, to=2000, variable=self.gen_length, 
                 orient='horizontal', length=200).grid(row=1, column=1, padx=5, pady=2)
        ttk.Label(gen_frame, textvariable=self.gen_length).grid(row=1, column=2, padx=5, pady=2)
        
        # Temperature
        ttk.Label(gen_frame, text="Temperature:").grid(row=2, column=0, sticky='w', padx=5, pady=2)
        self.temp = tk.DoubleVar(value=0.8)
        ttk.Scale(gen_frame, from_=0.1, to=2.0, variable=self.temp, 
                 orient='horizontal', length=200).grid(row=2, column=1, padx=5, pady=2)
        ttk.Label(gen_frame, textvariable=self.temp).grid(row=2, column=2, padx=5, pady=2)
        
        # Generate button
        ttk.Button(gen_frame, text="Generate Text", command=self.start_generation).grid(row=3, column=0, columnspan=4, pady=10)
        
        # Generation progress label (for future use, not currently updated in RNN generation)
        self.gen_progress_label = ttk.Label(gen_frame, text="", foreground='green')
        self.gen_progress_label.grid(row=4, column=0, columnspan=4, pady=5)
        
        # Generated text display
        text_frame = ttk.LabelFrame(frame, text="Generated Text", padding=10)
        text_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.output = scrolledtext.ScrolledText(text_frame, height=20, width=80, wrap='word', font=('Courier', 10))
        self.output.pack(fill='both', expand=True)
        
        # Action buttons
        action_frame = ttk.Frame(frame)
        action_frame.pack(fill='x', pady=5)
        
        ttk.Button(action_frame, text="Clear", command=self.clear_output).pack(side='left', padx=5)
        ttk.Button(action_frame, text="Copy", command=self.copy_output).pack(side='left', padx=5)
        ttk.Button(action_frame, text="Save", command=self.save_output).pack(side='left', padx=5)
    
    # ==================== FILE MANAGEMENT ====================
    def add_files(self):
        files = filedialog.askopenfilenames(
            title="Select text files",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if files:
            for file in files:
                if file not in self.loaded_files:
                    self.load_file(file)
    
    def add_folder(self):
        folder = filedialog.askdirectory(title="Select folder with text files")
        if folder:
            for root, dirs, files in os.walk(folder):
                for file in files:
                    if file.lower().endswith('.txt'):
                        file_path = os.path.join(root, file)
                        if file_path not in self.loaded_files:
                            self.load_file(file_path)
    
    def load_file(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            
            self.loaded_files.append(file_path)
            self.loaded_texts.append(text)
            
            filename = os.path.basename(file_path)
            file_size = os.path.getsize(file_path)
            char_count = len(text)
            
            item_id = self.file_tree.insert('', 'end', values=(
                len(self.loaded_files),
                filename,
                f"{file_size:,}",
                f"{char_count:,}",
                "✓ Loaded"
            ))
            
            self.file_tree.set(item_id, column='#', value=len(self.loaded_files))
            
            self.preview_combo['values'] = list(self.loaded_files)
            if len(self.loaded_files) == 1:
                self.preview_var.set(file_path)
                self.update_preview()
            
            self.update_stats()
            
            self.status.set(f"Loaded: {filename} ({char_count:,} chars)")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load {file_path}: {str(e)}")
    
    def remove_file(self):
        selection = self.file_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a file to remove")
            return
        
        for item in selection:
            index = int(self.file_tree.item(item, 'values')[0]) - 1
            if 0 <= index < len(self.loaded_files):
                removed_file = self.loaded_files.pop(index)
                self.loaded_texts.pop(index)
                self.file_tree.delete(item)
                
                for i, item_id in enumerate(self.file_tree.get_children()):
                    values = list(self.file_tree.item(item_id, 'values'))
                    values[0] = str(i + 1)
                    self.file_tree.item(item_id, values=values)
                
                self.status.set(f"Removed: {os.path.basename(removed_file)}")
        
        self.preview_combo['values'] = list(self.loaded_files)
        if self.loaded_files:
            self.preview_var.set(self.loaded_files[0])
            self.update_preview()
        else:
            self.preview_text.delete('1.0', 'end')
        
        self.update_stats()
    
    def clear_files(self):
        if not self.loaded_files:
            return
        
        if messagebox.askyesno("Confirm", "Clear all loaded files?"):
            self.loaded_files.clear()
            self.loaded_texts.clear()
            self.file_tree.delete(*self.file_tree.get_children())
            self.preview_text.delete('1.0', 'end')
            self.preview_combo['values'] = []
            self.preview_var.set('')
            self.stats_label.config(text="No files loaded")
            self.status.set("All files cleared")
    
    def update_preview(self, event=None):
        selected = self.preview_var.get()
        if selected in self.loaded_files:
            index = self.loaded_files.index(selected)
            text = self.loaded_texts[index]
            
            self.preview_text.delete('1.0', 'end')
            preview = text[:5000]
            self.preview_text.insert('1.0', preview)
            
            if len(text) > 5000:
                self.preview_text.insert('end', '\n\n[... truncated]')
    
    def update_stats(self):
        if not self.loaded_files:
            self.stats_label.config(text="No files loaded")
            return
        
        total_chars = sum(len(text) for text in self.loaded_texts)
        total_size = sum(os.path.getsize(f) for f in self.loaded_files)
        
        stats = (f"Files: {len(self.loaded_files)} | "
                f"Total characters: {total_chars:,} | "
                f"Total size: {total_size:,} bytes")
        
        self.stats_label.config(text=stats)
    
    def process_files(self):
        """Convert all loaded texts to tokens using the fixed vocabulary."""
        if not self.loaded_texts:
            messagebox.showerror("Error", "No files loaded!")
            return
        
        # Create a new processor with current seq_len
        self.processor = TextProcessor(seq_len=self.seq_len.get())
        
        # Concatenate all texts and convert to tokens
        combined_text = "".join(self.loaded_texts)
        self.processor.tokens = self.processor.text_to_tokens(combined_text)
        
        stats = {
            'chars': len(combined_text),
            'vocab': len(self.processor.char2idx),
            'files': len(self.loaded_texts)
        }
        
        self.log_message(f"Processed {stats['files']} files")
        self.log_message(f"Combined: {stats['chars']:,} characters")
        self.log_message(f"Fixed vocabulary size: {stats['vocab']} tokens")
        
        # Count unknown characters
        unk_count = self.processor.tokens.count(1)  # <UNK> is index 1
        if unk_count > 0:
            percent = (unk_count / stats['chars']) * 100
            self.log_message(f"Warning: {unk_count} unknown characters ({percent:.2f}%) mapped to <UNK>")
        
        self.init_btn.config(state='normal')  # Enable init button
        self.status.set(f"Ready to initialize model on {stats['chars']:,} characters")
    
    # ==================== MODEL INITIALIZATION ====================
    def init_model(self):
        """Initialize the RNN model using the processor's vocab size."""
        if not hasattr(self.processor, 'tokens') or len(self.processor.tokens) == 0:
            messagebox.showerror("Error", "Please process files first!")
            return
        
        vocab_size = len(self.processor.char2idx)
        
        # Get model configuration
        model_size = self.model_size.get()
        if model_size == "Small":
            embed, hidden, layers = 64, 128, 1
        elif model_size == "Medium":
            embed, hidden, layers = 128, 256, 2
        else:  # Large
            embed, hidden, layers = 192, 384, 2
        
        # Create model
        self.model = TextGRU(
            vocab_size=vocab_size,
            embed_size=embed,
            hidden_size=hidden,
            n_layers=layers
        )
        
        self.log_message(f"Model initialized with {model_size} config")
        self.log_message(f"  Embed: {embed}, Hidden: {hidden}, Layers: {layers}")
        self.log_message(f"  Vocabulary size: {vocab_size}")
        
        self.train_btn.config(state='normal')  # Enable training button
        self.status.set(f"Model initialized - Ready to train")
    
    # ==================== TRAINING ====================
    def start_training(self):
        if self.model is None:
            messagebox.showerror("Error", "Please initialize model first!")
            return
        
        if self.training:
            return
        
        self.training = True
        self.init_btn.config(state='disabled')
        self.train_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.progress.start()
        
        threading.Thread(target=self.train_loop, daemon=True).start()
        
        self.log_message(f"Training started on {len(self.processor.tokens):,} tokens")
        self.status.set("Training started...")
    
    def train_loop(self):
        try:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.lr.get()))
            criterion = nn.CrossEntropyLoss()
            
            epochs = self.epochs.get()
            batch_size = self.batch_size.get()
            
            for epoch in range(epochs):
                if not self.training:
                    break
                
                self.model.train()
                total_loss = 0
                steps = 0
                
                for _ in range(100):
                    if not self.training:
                        break
                    
                    batch_x, batch_y = self.processor.get_batch(batch_size)
                    optimizer.zero_grad()
                    output, _ = self.model(batch_x)
                    output = output.reshape(-1, self.model.vocab_size)
                    targets = batch_y.reshape(-1)
                    loss = criterion(output, targets)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    
                    total_loss += loss.item()
                    steps += 1
                
                avg_loss = total_loss / max(steps, 1)
                self.update_queue.put(('loss', (epoch+1, epochs, avg_loss)))
                
                if (epoch + 1) % 10 == 0:
                    threading.Thread(target=self.generate_sample_for_training, 
                                   args=(epoch+1,), daemon=True).start()
                
                time.sleep(0.1)
            
            self.update_queue.put(('training_complete', None))
            
        except Exception as e:
            self.update_queue.put(('error', f"Training error: {str(e)}"))
            traceback.print_exc()
    
    def generate_sample_for_training(self, epoch):
        try:
            if self.model is None:
                return
            
            self.model.eval()
            seed_tokens = self.processor.text_to_tokens("Sample: ")
            if not seed_tokens:
                return
            
            generated = seed_tokens.copy()
            hidden = None
            
            with torch.no_grad():
                for _ in range(100):
                    last_token = generated[-1]
                    next_token, hidden = self.model.generate_step(last_token, hidden, 0.8)
                    generated.append(next_token)
            
            sample = self.processor.tokens_to_text(generated)
            self.update_queue.put(('sample', (epoch, sample[:100])))
            
        except Exception as e:
            print(f"Sample generation error: {e}")
    
    def stop_training(self):
        self.training = False
        self.log_message("Training stopped")
    
    def log_message(self, message):
        self.log.config(state='normal')
        self.log.insert('end', f"{time.strftime('%H:%M:%S')} - {message}\n")
        self.log.see('end')
        self.log.config(state='disabled')
    
    # ==================== MODEL SAVE/LOAD ====================
    def save_model(self):
        if self.model is None:
            messagebox.showerror("Error", "No model to save!")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".pth",
            filetypes=[("PyTorch model", "*.pth"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                self.model.save_model(filename)
                self.log_message(f"Model saved to {filename}")
                self.status.set(f"Model saved: {os.path.basename(filename)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save model: {str(e)}")
    
    def load_model(self):
        filename = filedialog.askopenfilename(
            title="Select model file",
            filetypes=[("PyTorch model", "*.pth"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                self.model = TextGRU.load_model(filename)
                
                # Create a new processor with the fixed vocabulary
                # Use the current seq_len setting (could also use model's max_seq_len if stored)
                self.processor = TextProcessor(seq_len=self.seq_len.get())
                
                # Check if vocab sizes match
                fixed_vocab_size = len(self.processor.char2idx)
                if fixed_vocab_size != self.model.vocab_size:
                    self.log_message(f"WARNING: Fixed vocabulary size ({fixed_vocab_size}) does not match model's vocab size ({self.model.vocab_size}).")
                    self.log_message("This may cause errors during generation/training. The model was likely trained with a different character set.")
                
                self.log_message(f"Model loaded from {os.path.basename(filename)}")
                self.log_message(f"Fixed vocabulary size: {fixed_vocab_size}")
                self.status.set(f"Model loaded successfully")
                
                # Enable training and generation
                self.init_btn.config(state='normal')
                self.train_btn.config(state='normal')
                
                # Update UI to show loaded model settings (approx)
                self.update_model_settings_from_checkpoint()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {str(e)}")
                traceback.print_exc()
    
    def update_model_settings_from_checkpoint(self):
        """Update UI settings based on loaded model"""
        if self.model:
            # Update model size display based on parameters
            embed = self.model.embed_size
            if embed == 64:
                self.model_size.set('Small')
            elif embed == 128:
                self.model_size.set('Medium')
            elif embed == 192:
                self.model_size.set('Large')
            # (hidden_size and layers could also be used for more precise matching)
    
    # ==================== GENERATION ====================
    def start_generation(self):
        if self.model is None:
            messagebox.showerror("Error", "Please train or load a model first")
            return
        
        if self.generating:
            return
        
        seed = self.seed_text.get()
        if not seed:
            messagebox.showerror("Error", "Please enter seed text")
            return
        
        length = self.gen_length.get()
        temp = self.temp.get()
        
        # Clear and start
        self.output.delete('1.0', 'end')
        self.output.insert('1.0', "Generating...\n" + "="*50 + "\n\n" + seed)
        self.status.set("Generating text...")
        self.gen_progress_label.config(text="Generating...")
        
        # Generate in thread
        self.generating = True
        threading.Thread(target=self.generate_thread, 
                        args=(seed, length, temp), daemon=True).start()
    
    def generate_thread(self, seed, length, temp):
        try:
            if self.model is None:
                return
            
            self.model.eval()
            seed_tokens = self.processor.text_to_tokens(seed)
            if not seed_tokens:
                return
            
            generated = seed_tokens.copy()
            hidden = None
            
            with torch.no_grad():
                for i in range(length):
                    last_token = generated[-1]
                    next_token, hidden = self.model.generate_step(last_token, hidden, temp)
                    generated.append(next_token)
                    
                    # Update every 10 characters for responsiveness
                    if i % 10 == 0:
                        current_text = self.processor.tokens_to_text(generated)
                        self.update_queue.put(('gen_update', (i, current_text)))
            
            final_text = self.processor.tokens_to_text(generated)
            self.update_queue.put(('gen_complete', final_text))
            
        except Exception as e:
            self.update_queue.put(('error', f"Generation error: {str(e)}"))
            traceback.print_exc()
        finally:
            self.generating = False
    
    def update_ui_thread(self):
        while True:
            try:
                msg_type, data = self.update_queue.get(timeout=0.1)
                
                if msg_type == 'loss':
                    epoch, total_epochs, loss = data
                    self.loss_label.config(text=f"{loss:.4f}")
                    self.log_message(f"Epoch {epoch}/{total_epochs} - Loss: {loss:.4f}")
                    self.status.set(f"Training: Epoch {epoch}/{total_epochs}, Loss: {loss:.4f}")
                
                elif msg_type == 'sample':
                    epoch, sample = data
                    self.log_message(f"Epoch {epoch} sample: {sample}...")
                
                elif msg_type == 'training_complete':
                    self.training = False
                    self.init_btn.config(state='normal')
                    self.train_btn.config(state='normal')
                    self.stop_btn.config(state='disabled')
                    self.progress.stop()
                    self.log_message("Training completed!")
                    self.status.set("Training completed")
                
                elif msg_type == 'gen_update':
                    i, current_text = data
                    self.output.delete('1.0', 'end')
                    self.output.insert('1.0', current_text)
                    self.output.see('end')
                
                elif msg_type == 'gen_complete':
                    final_text = data
                    self.output.delete('1.0', 'end')
                    self.output.insert('1.0', final_text)
                    self.output.see('end')
                    self.status.set(f"Generated {len(final_text)} characters")
                    self.log_message(f"Generated text from seed: '{self.seed_text.get()[:30]}...'")
                    self.gen_progress_label.config(text="✓ Generation complete!")
                    self.generating = False
                
                elif msg_type == 'error':
                    self.log_message(data)
                    self.gen_progress_label.config(text="✗ Generation failed", foreground='red')
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"UI update error: {e}")
                self.generating = False
    
    def clear_output(self):
        self.output.delete('1.0', 'end')
        self.gen_progress_label.config(text="")
    
    def copy_output(self):
        text = self.output.get('1.0', 'end-1c')
        if text:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.status.set("Copied to clipboard")
    
    def save_output(self):
        text = self.output.get('1.0', 'end-1c')
        if not text:
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filename:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(text)
            self.status.set(f"Saved to {os.path.basename(filename)}")

# ==================== MAIN ====================
def main():
    try:
        root = tk.Tk()
        
        style = ttk.Style()
        style.configure('Accent.TButton', font=('Arial', 10, 'bold'))
        
        app = MultiFileTextGenerator(root)
        
        root.update_idletasks()
        width = root.winfo_width()
        height = root.winfo_height()
        x = (root.winfo_screenwidth() // 2) - (width // 2)
        y = (root.winfo_screenheight() // 2) - (height // 2)
        root.geometry(f'1000x800+{x}+{y}')
        
        root.mainloop()
        
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()