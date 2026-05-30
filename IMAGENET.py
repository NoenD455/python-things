import os
import re
from datasets import load_dataset
from PIL import Image


def sanitize(name):
    name = name.replace(' ', '_')
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    name = re.sub(r'_+', '_', name)
    return name.strip('_')

def save_split(dataset, split_name, out_base, class_names=None):
    split_dir = os.path.join(out_base, split_name)
    os.makedirs(split_dir, exist_ok=True)
    total = len(dataset)
    for idx, sample in enumerate(dataset):
        label = sample["label"]
        if class_names is not None and label != -1:
            subdir = os.path.join(split_dir, sanitize(class_names[label]))
            os.makedirs(subdir, exist_ok=True)
            fname = f"image_{idx:08d}.png"
            path = os.path.join(subdir, fname)
        else:
            fname = f"test_image_{idx:08d}.png"
            path = os.path.join(split_dir, fname)
        sample["image"].save(path, "PNG")
        if (idx+1) % 5000 == 0:
            print(f"{split_name}: {idx+1}/{total}")
    print(f"{split_name} done ({total} images)")

# ========== CONFIG =========================================
SIZE = 128                     # 32, 64, 128, or 256
OUTPUT_DIR = "D:/IMAGENET128"  # where PNG folders will go
# ===========================================================

print(f"Downloading ImageNet-1k {SIZE}x{SIZE}")
print(f"Cache will use: {os.environ.get('HF_DATASETS_CACHE', 'default')}")
print(f"Output: {OUTPUT_DIR}")

ds_train = load_dataset(f"benjamin-paine/imagenet-1k-{SIZE}x{SIZE}", split="train")
ds_val   = load_dataset(f"benjamin-paine/imagenet-1k-{SIZE}x{SIZE}", split="validation")
ds_test  = load_dataset(f"benjamin-paine/imagenet-1k-{SIZE}x{SIZE}", split="test")

class_names = {i: name for i, name in enumerate(ds_train.features["label"].names)}

os.makedirs(OUTPUT_DIR, exist_ok=True)
save_split(ds_train, "train", OUTPUT_DIR, class_names)
save_split(ds_val,   "val",   OUTPUT_DIR, class_names)
save_split(ds_test,  "test",  OUTPUT_DIR, class_names=None)

print("\nComplete!")
print(f"Train: {OUTPUT_DIR}/train")
print(f"Val:   {OUTPUT_DIR}/val")
print(f"Test:  {OUTPUT_DIR}/test")