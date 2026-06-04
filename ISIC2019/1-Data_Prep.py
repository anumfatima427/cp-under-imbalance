#!/usr/bin/env python3
"""
ISIC 2019 Dataset Preparation - Uses Existing Train/Test Split
(Modified: Recursively finds images inside class subfolders and drops UNK)

Usage:
    python 1-Data_Prep.py --data_dir /path/to/isic2019 --output_dir ./output
"""

import os
import shutil
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# ISIC 2019 Official Classes (UNK has been removed!)
ALL_CLASS_COLS  = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
ALL_CLASS_NAMES = ['Melanoma', 'Nevus', 'Basal Cell Carc.', 'Actinic Keratosis', 
                   'Benign Keratosis', 'Dermatofibroma', 'Vascular Lesion', 
                   'Squamous Cell Carc.']

def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare ISIC 2019 dataset using existing train/test split (8 classes)'
    )
    parser.add_argument('--data_dir',   type=str, required=True,
                        help='Path to ISIC dataset folder (must contain train/ and optionally test/ subfolders)')
    parser.add_argument('--output_dir', type=str, default='./dataset_split',
                        help='Where to save output files (default: ./dataset_split)')
    parser.add_argument('--seed',       type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--symlink',    action='store_true',
                        help='Create symlinks instead of copying images (saves disk space)')
    return parser.parse_args()

def load_split(data_dir, split_name, enforce_cols=None):
    """Load a single split (train or test) from its subfolder."""
    csv_path  = os.path.join(data_dir, split_name, f'{split_name}.csv')
    image_dir = os.path.join(data_dir, split_name, 'images')

    if not os.path.exists(csv_path):
        return None, None, None

    print(f"  Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    # Determine which ISIC columns are actually in this CSV
    if enforce_cols:
        existing_cols = enforce_cols
        existing_names = [ALL_CLASS_NAMES[ALL_CLASS_COLS.index(c)] for c in existing_cols]
    else:
        existing_cols = [c for c in ALL_CLASS_COLS if c in df.columns]
        existing_names = [ALL_CLASS_NAMES[ALL_CLASS_COLS.index(c)] for c in existing_cols]

    if not existing_cols:
        raise ValueError(f"None of the expected label columns found in {csv_path}")

    before = len(df)
    df = df.dropna(subset=existing_cols).copy()
    
    # SAFETY CHECK: Only keep rows where exactly ONE of our 8 classes is active. 
    df[existing_cols] = df[existing_cols].astype(int)
    valid_rows = df[existing_cols].sum(axis=1) == 1
    df = df[valid_rows].copy()
    
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped:,} invalid/unlabeled rows (e.g., UNK or missing labels)")

    # Convert one-hot -> integer label (0 to 7) and map the label name
    df['label'] = df[existing_cols].values.argmax(axis=1)
    
    idx_to_name = {i: name for i, name in enumerate(existing_names)}
    df['label_name'] = df['label'].map(idx_to_name)

    # --- NEW: Recursive Image Search ---
    print(f"  Scanning for images in {image_dir} (including subfolders)...")
    image_lookup = {}
    for root, _, files in os.walk(image_dir):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Store by exact filename AND by name without extension for robust matching
                image_lookup[f] = os.path.join(root, f)
                name_no_ext = os.path.splitext(f)[0]
                image_lookup[name_no_ext] = os.path.join(root, f)

    def resolve_path(img_name):
        img_name = str(img_name)
        if img_name in image_lookup:
            return image_lookup[img_name]
        name_no_ext = os.path.splitext(img_name)[0]
        if name_no_ext in image_lookup:
            return image_lookup[name_no_ext]
        # Fallback if totally missing
        ext = '' if img_name.lower().endswith(('.png', '.jpg', '.jpeg')) else '.jpg'
        return os.path.join(image_dir, img_name + ext)

    df['image_path'] = df['image'].apply(resolve_path)

    # Standardize the final output image names to have .jpg
    def format_image_name(x):
        x = str(x)
        if not x.lower().endswith(('.png', '.jpg', '.jpeg')):
            return x + '.jpg'
        return x

    df['image'] = df['image'].apply(format_image_name)

    # Validate images exist
    print(f"  Validating image paths...")
    valid = [os.path.exists(p) for p in tqdm(df['image_path'], desc="  Checking", unit="img")]
    missing = len(valid) - sum(valid)
    if missing:
        print(f"  Warning: {missing:,} images not found on disk, skipping them")
        df = df[[v for v in valid]]

    return df, existing_cols, existing_names

def print_distribution(df, title, class_names):
    """Print class distribution table."""
    print(f"\n{title}")
    print("-" * 55)
    print(f"  {'#':<3} {'Name':<22} {'Count':>7}  {'Percent':>7}")
    print("-" * 55)
    for i, name in enumerate(class_names):
        count = (df['label'] == i).sum()
        pct   = count / len(df) * 100 if len(df) > 0 else 0
        print(f"  {i:<3} {name:<22} {count:>7,}  ({pct:>5.1f}%)")
    print("-" * 55)
    print(f"  {'Total':<26} {len(df):>7,}")

def plot_distribution(df, title, save_path, class_names):
    """Bar chart of class distribution for a single split."""
    counts = [int((df['label'] == i).sum()) for i in range(len(class_names))]
    
    # 8 Distinct colors for ISIC classes
    base_colors = ['#e74c3c', '#3498db', '#2ecc71', '#f1c40f', '#9b59b6', '#34495e', '#e67e22', '#1abc9c']
    colors = base_colors[:len(class_names)]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(class_names, counts, color=colors, edgecolor='black', linewidth=1.2, alpha=0.85)

    ax.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max(counts) * 1.18 if counts else 10)

    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=25, ha='right', fontsize=10)

    for bar, count in zip(bars, counts):
        if count == 0: continue
        pct = count / len(df) * 100
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                f'{count:,}\n({pct:.1f}%)', ha='center', va='bottom',
                fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.basename(save_path)}")

def plot_splits(train_df, test_df, save_path, class_names):
    """Side-by-side bar chart comparing train and test distributions."""
    base_colors = ['#e74c3c', '#3498db', '#2ecc71', '#f1c40f', '#9b59b6', '#34495e', '#e67e22', '#1abc9c']
    colors = base_colors[:len(class_names)]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for ax, (name, df) in zip(axes, [('Train', train_df), ('Test', test_df)]):
        counts = [int((df['label'] == i).sum()) for i in range(len(class_names))]
        bars = ax.bar(class_names, counts, color=colors, edgecolor='black',
                      linewidth=1.2, alpha=0.85)
        
        ax.set_title(f'{name} Set  (n={len(df):,})', fontsize=13, fontweight='bold')
        ax.set_ylabel('Number of Samples', fontsize=11)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(0, max(counts) * 1.2 if counts else 10)
        
        ax.set_xticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=25, ha='right', fontsize=9)
        
        for bar, count in zip(bars, counts):
            if count == 0: continue
            pct = count / len(df) * 100
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                    f'{count:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=8)

    plt.suptitle('ISIC 2019 - Train vs Test Class Distribution (8 Classes)', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.basename(save_path)}")

def save_split(df, split_name, output_dir, class_cols, copy_images=True):
    """Save CSV and copy/symlink images for a split."""
    split_dir  = os.path.join(output_dir, split_name)
    images_dir = os.path.join(split_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    csv_path = os.path.join(split_dir, f'{split_name}.csv')
    save_cols = ['image', 'label', 'label_name', 'image_path'] + class_cols
    df[save_cols].to_csv(csv_path, index=False)

    action = "Copying" if copy_images else "Linking"
    for _, row in tqdm(df.iterrows(), total=len(df),
                       desc=f"  {action} {split_name}", unit="img"):
        src = row['image_path']
        dst = os.path.join(images_dir, row['image'])
        if not os.path.exists(dst):
            if copy_images:
                shutil.copy2(src, dst)
            else:
                try:
                    os.symlink(os.path.abspath(src), dst)
                except OSError:
                    shutil.copy2(src, dst)

    return csv_path

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("  ISIC 2019 Dataset Preparation (8 Classes)")
    print("=" * 60)

    # 1. Load train split
    print("\n[1/4] Loading train split...")
    train_df, class_cols, class_names = load_split(args.data_dir, 'train')
    if train_df is None:
        print("  ERROR: Could not find train.csv in train/ subfolder.")
        return
    print(f"  Loaded {len(train_df):,} labeled train samples")

    # 2. Load test split (Optional)
    print("\n[2/4] Loading test split...")
    test_df, _, _ = load_split(args.data_dir, 'test', enforce_cols=class_cols)
    if test_df is not None:
        print(f"  Loaded {len(test_df):,} labeled test samples")
    else:
        print("  No test split found. Proceeding with train only.")

    # 3. Analyze
    print("\n[3/4] Analyzing dataset...")
    print_distribution(train_df, "Train Distribution", class_names)
    if test_df is not None:
        print_distribution(test_df, "Test Distribution", class_names)

    # Compute Train IR
    counts = [int((train_df['label'] == i).sum()) for i in range(len(class_names))]
    non_zero_counts = [c for c in counts if c > 0]
    if non_zero_counts:
        ir = max(counts) / min(non_zero_counts)
        print(f"\n  Train Imbalance Ratio: {ir:.2f}x")

    # 4. Save
    print("\n[4/5] Saving splits...")
    save_split(train_df, 'train', args.output_dir, class_cols, copy_images=not args.symlink)
    print(f"\n  Saved: train/train.csv ({len(train_df):,} samples)")

    if test_df is not None:
        save_split(test_df, 'test', args.output_dir, class_cols, copy_images=not args.symlink)
        print(f"  Saved: test/test.csv   ({len(test_df):,} samples)")

    # 5. Plots
    print("\n[5/5] Saving visualizations...")
    plot_distribution(train_df, 'ISIC 2019 - Train Class Distribution',
                      os.path.join(args.output_dir, 'train_distribution.png'), class_names)
    if test_df is not None:
        plot_distribution(test_df, 'ISIC 2019 - Test Class Distribution',
                          os.path.join(args.output_dir, 'test_distribution.png'), class_names)
        plot_splits(train_df, test_df,
                    os.path.join(args.output_dir, 'split_distribution.png'), class_names)

    print("\n" + "=" * 60)
    print("  Done! Dataset is formatted and ready for Conformal Training.")
    print("=" * 60)

if __name__ == '__main__':
    main()