#!/usr/bin/env python3
"""
DFUC2021 Dataset Preparation - Uses Existing Train/Test Split

The CSV has one-hot encoded labels across 4 columns:
    none, infection, ischaemia, both

Rows where all label columns are NaN are unlabeled (test images) and are excluded.

Usage:
    python 1-Data_Prep.py --data_dir /path/to/dfuc2021 --output_dir ./output

Output:
    output/
    ├── train/
    │   ├── train.csv          # image, label, label_name, image_path
    │   └── images/
    ├── test/
    │   ├── test.csv
    │   └── images/
"""

import os
import shutil
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# DFUC2021 class definitions
CLASS_COLS  = ['none', 'infection', 'ischaemia', 'both']
CLASS_NAMES = ['None', 'Infection', 'Ischaemia', 'Both']
NUM_CLASSES = 4


def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare DFUC2021 dataset using existing train/test split'
    )
    parser.add_argument('--data_dir',   type=str, required=True,
                        help='Path to DFUC2021 dataset folder (must contain train/ and test/ subfolders, each with a CSV and images/)')
    parser.add_argument('--output_dir', type=str, default='./dataset_split',
                        help='Where to save output files (default: ./dataset_split)')
    parser.add_argument('--seed',       type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--symlink',    action='store_true',
                        help='Create symlinks instead of copying images')
    return parser.parse_args()


def load_split(data_dir, split_name):
    """Load a single split (train or test) from its subfolder."""
    csv_path  = os.path.join(data_dir, split_name, f'{split_name}.csv')
    image_dir = os.path.join(data_dir, split_name, 'images')

    print(f"  Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    # Drop unlabeled rows (all label cols are NaN)
    before = len(df)
    df = df.dropna(subset=CLASS_COLS).copy()
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped:,} unlabeled rows (NaN labels)")

    # Convert one-hot -> integer label (0-3) and label name
    df[CLASS_COLS] = df[CLASS_COLS].astype(int)
    df['label']      = df[CLASS_COLS].values.argmax(axis=1)
    df['label_name'] = df['label'].apply(lambda x: CLASS_NAMES[x])

    # Full image path
    df['image_path'] = df['image'].apply(lambda x: os.path.join(image_dir, x))

    # Validate images exist
    print(f"  Validating image paths...")
    valid = [os.path.exists(p) for p in tqdm(df['image_path'], desc="  Checking", unit="img")]
    missing = len(valid) - sum(valid)
    if missing:
        print(f"  Warning: {missing:,} images not found, skipping them")
        df = df[[v for v in valid]]

    return df


def print_distribution(df, title):
    """Print class distribution table."""
    print(f"\n{title}")
    print("-" * 50)
    print(f"  {'#':<3} {'Name':<15} {'Count':>7}  {'Percent':>7}")
    print("-" * 50)
    for i, name in enumerate(CLASS_NAMES):
        count = (df['label'] == i).sum()
        pct   = count / len(df) * 100
        print(f"  {i:<3} {name:<15} {count:>7,}  ({pct:>5.1f}%)")
    print("-" * 50)
    print(f"  {'Total':<19} {len(df):>7,}")


def plot_distribution(df, title, save_path):
    """Bar chart of class distribution for a single split."""
    counts = [int((df['label'] == i).sum()) for i in range(NUM_CLASSES)]
    colors = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(CLASS_NAMES, counts, color=colors, edgecolor='black', linewidth=1.2, alpha=0.85)

    ax.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max(counts) * 1.18)

    for bar, count in zip(bars, counts):
        pct = count / len(df) * 100
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                f'{count:,}\n({pct:.1f}%)', ha='center', va='bottom',
                fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.basename(save_path)}")


def plot_splits(train_df, test_df, save_path):
    """Side-by-side bar chart comparing train and test distributions."""
    colors = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (name, df) in zip(axes, [('Train', train_df), ('Test', test_df)]):
        counts = [int((df['label'] == i).sum()) for i in range(NUM_CLASSES)]
        bars = ax.bar(CLASS_NAMES, counts, color=colors, edgecolor='black',
                      linewidth=1.2, alpha=0.85)
        ax.set_title(f'{name} Set  (n={len(df):,})', fontsize=13, fontweight='bold')
        ax.set_ylabel('Number of Samples', fontsize=11)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(0, max(counts) * 1.2)
        for bar, count in zip(bars, counts):
            pct = count / len(df) * 100
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                    f'{count:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9)

    plt.suptitle('DFUC2021 - Train vs Test Class Distribution',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.basename(save_path)}")


def save_split(df, split_name, output_dir, copy_images=True):
    """Save CSV and copy/symlink images for a split."""
    split_dir  = os.path.join(output_dir, split_name)
    images_dir = os.path.join(split_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    # Save CSV: image, label, label_name, image_path + one-hot cols
    csv_path = os.path.join(split_dir, f'{split_name}.csv')
    save_cols = ['image', 'label', 'label_name', 'image_path'] + CLASS_COLS
    df[save_cols].to_csv(csv_path, index=False)

    # Copy or symlink images
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
    print("  DFUC2021 Dataset Preparation - Using Existing Split")
    print("=" * 60)
    print(f"  Data directory  : {args.data_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Random seed     : {args.seed}")
    print(f"  Copy images     : {not args.symlink}")

    # 1. Load both splits
    print("\n[1/4] Loading train split...")
    train_df = load_split(args.data_dir, 'train')
    print(f"  Loaded {len(train_df):,} labeled train samples")

    print("\n[2/4] Loading test split...")
    test_df = load_split(args.data_dir, 'test')
    print(f"  Loaded {len(test_df):,} labeled test samples")

    # 2. Analyze
    print("\n[3/4] Analyzing dataset...")
    print_distribution(train_df, "Train Distribution")
    print_distribution(test_df,  "Test Distribution")

    counts = train_df['label'].value_counts()
    ir = counts.max() / counts.min()
    print(f"\n  Train Imbalance Ratio: {ir:.2f}x")

    # 3. Save
    print("\n[4/5] Saving splits...")
    save_split(train_df, 'train', args.output_dir, copy_images=not args.symlink)
    save_split(test_df,  'test',  args.output_dir, copy_images=not args.symlink)

    print(f"\n  Saved: train/train.csv ({len(train_df):,} samples)")
    print(f"  Saved: test/test.csv   ({len(test_df):,} samples)")

    # 4. Plots
    print("\n[5/5] Saving visualizations...")
    plot_distribution(train_df, 'DFUC2021 - Train Class Distribution',
                      os.path.join(args.output_dir, 'train_distribution.png'))
    plot_distribution(test_df, 'DFUC2021 - Test Class Distribution',
                      os.path.join(args.output_dir, 'test_distribution.png'))
    plot_splits(train_df, test_df,
                os.path.join(args.output_dir, 'split_distribution.png'))

    print("\n" + "=" * 60)
    print("  Done!")
    print("=" * 60)
    print(f"""
Output structure:
  {args.output_dir}/
  |-- train/
  |   |-- train.csv        ({len(train_df):,} samples)
  |   `-- images/
  |-- test/
  |   |-- test.csv         ({len(test_df):,} samples)
  |   `-- images/
  |-- train_distribution.png
  |-- test_distribution.png
  `-- split_distribution.png

CSV columns: image, label, label_name, image_path, none, infection, ischaemia, both
""")


if __name__ == '__main__':
    main()