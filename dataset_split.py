import os
import shutil
import random
from pathlib import Path

# === CONFIGURATION ===
source_folders = {
    "broken_original": "dataset_original/break/train",
    "whole_original": "dataset_original/whole/train",
    "broken_augmented": "dataset_break_aug/train",
    "whole_augmented": "dataset_whole_aug/train"
}

output_base = "dataset"  # Where train/val will be placed

train_ratio = 0.8  # 80% train, 20% val

# === FUNCTION TO SPLIT DATA ===
def split_dataset(src_dir, dst_base, train_ratio=0.8):
    image_dir = Path(src_dir) / "images"
    label_dir = Path(src_dir) / "labels"

    image_files = list(image_dir.glob("*.jpg"))
    random.shuffle(image_files)

    split_index = int(len(image_files) * train_ratio)
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]

    for split_name, split_files in [("train", train_files), ("val", val_files)]:
        out_img_dir = Path(dst_base) / split_name / "images"
        out_lbl_dir = Path(dst_base) / split_name / "labels"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path in split_files:
            label_path = label_dir / (img_path.stem + ".txt")
            shutil.copy2(img_path, out_img_dir / img_path.name)
            if label_path.exists():
                shutil.copy2(label_path, out_lbl_dir / label_path.name)
            else:
                print(f"⚠️ Label not found for: {img_path.name}")

# === MAIN EXECUTION ===
if __name__ == "__main__":
    random.seed(42)  # for reproducibility

    for name, folder in source_folders.items():
        print(f"🔄 Splitting {name}...")
        split_dataset(folder, output_base)
    
    print("✅ Done. Dataset split into train and val.")
