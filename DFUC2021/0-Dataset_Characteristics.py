#!/usr/bin/env python3
"""
DFUC2021 Dataset Imbalance Analysis
Diabetic Foot Ulcer Challenge 2021 - Multi-label classification dataset
Classes: none, infection, ischaemia, both
"""

import numpy as np
import pandas as pd
from scipy.stats import entropy
from collections import Counter

# DFUC2021 class names
DFUC_CLASSES = ['None', 'Infection', 'Ischaemia', 'Both']
DFUC_COLS    = ['none', 'infection', 'ischaemia', 'both']

# Severity colors (green → orange → red → dark red)
CLASS_COLORS = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']


def load_labeled(csv_path):
    """Load CSV and drop rows where all label columns are NaN (unlabeled test images)."""
    df = pd.read_csv(csv_path)
    labeled = df.dropna(subset=DFUC_COLS).copy()
    labeled[DFUC_COLS] = labeled[DFUC_COLS].astype(int)
    return labeled


def analyze_imbalance(csv_path, dataset_name='Dataset'):
    """Analyze class imbalance for DFUC2021 dataset (multi-label, one-hot encoded)."""

    df = load_labeled(csv_path)
    total = len(df)

    # Per-class positive counts (each image can belong to exactly one class here)
    counts = [int(df[col].sum()) for col in DFUC_COLS]

    # Sanity check: confirm each row sums to 1 (mutually exclusive labels)
    row_sums = df[DFUC_COLS].sum(axis=1)
    is_exclusive = (row_sums == 1).all()

    min_count = min(counts)
    max_count = max(counts)
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')

    non_zero = [c for c in counts if c > 0]
    shannon = entropy(non_zero) / np.log(len(non_zero))

    percentages = [(c / total) * 100 for c in counts]

    # ── Print results ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  {dataset_name}")
    print(f"{'='*70}")
    print(f"Total labeled samples : {total:,}")
    print(f"Labels are mutually exclusive: {is_exclusive}")
    print(f"\nClass Distribution:")
    print(f"{'#':<4} {'Name':<15} {'Count':>8} {'Percent':>10}  Bar")
    print(f"{'-'*70}")

    for i, (name, count, pct) in enumerate(zip(DFUC_CLASSES, counts, percentages)):
        bar = '█' * int(pct / 2)
        print(f"{i:<4} {name:<15} {count:>8,} {pct:>9.2f}%  {bar}")

    print(f"{'-'*70}")
    print(f"{'Total':<20} {total:>8,}")

    print(f"\n{'='*70}")
    print(f"Imbalance Metrics:")
    print(f"{'='*70}")
    print(f"  Imbalance Ratio (IR):         {imbalance_ratio:.2f}x")
    print(f"    └─ Largest / Smallest:      {max_count:,} / {min_count:,}")
    print(f"\n  Shannon Entropy (normalized):  {shannon:.4f}")
    print(f"    └─ 1.0 = perfectly balanced, 0.0 = totally imbalanced")
    print(f"\n  Majority Class Dominance:     {max(percentages):.1f}%")
    print(f"  Minority Class Coverage:      {min(percentages):.1f}%")

    if imbalance_ratio < 2:
        severity = "LOW — Nearly balanced"
    elif imbalance_ratio < 10:
        severity = "MODERATE — Some imbalance"
    elif imbalance_ratio < 50:
        severity = "HIGH — Significant imbalance"
    else:
        severity = "EXTREME — Severe imbalance"

    print(f"\n  Imbalance Severity:           {severity}")
    print(f"{'='*70}\n")

    return {
        'df': df,
        'counts': counts,
        'percentages': percentages,
        'total': total,
        'imbalance_ratio': imbalance_ratio,
        'shannon': shannon,
    }



def create_comparison_table(train_stats, test_stats):
    """Print a train vs test comparison table."""
    print(f"\n{'='*90}")
    print(f"{'TRAIN vs TEST COMPARISON':^90}")
    print(f"{'='*90}")
    print(f"{'#':<4} {'Name':<15} {'Train Count':>12} {'Train %':>10} "
          f"{'Test Count':>12} {'Test %':>10}")
    print(f"{'-'*90}")

    for i, name in enumerate(DFUC_CLASSES):
        print(f"{i:<4} {name:<15} {train_stats['counts'][i]:>12,} "
              f"{train_stats['percentages'][i]:>9.2f}% "
              f"{test_stats['counts'][i]:>12,} "
              f"{test_stats['percentages'][i]:>9.2f}%")

    print(f"{'-'*90}")
    print(f"{'Total':<20} {train_stats['total']:>12,} {'100.00%':>10} "
          f"{test_stats['total']:>12,} {'100.00%':>10}")
    print(f"{'='*90}\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='DFUC2021 Dataset Imbalance Analysis')
    parser.add_argument('--train_csv', default='train.csv',
                        help='Path to train CSV (one-hot encoded labels)')
    parser.add_argument('--test_csv', default=None,
                        help='Path to test CSV (optional)')
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  DFUC2021 DATASET IMBALANCE ANALYSIS")
    print("=" * 70)

    train_stats = analyze_imbalance(args.train_csv, 'TRAIN SET')
    stats_dict = {'train': train_stats}

    if args.test_csv:
        test_stats = analyze_imbalance(args.test_csv, 'TEST SET')
        stats_dict['test'] = test_stats
        create_comparison_table(train_stats, test_stats)

    print("\n✔ Analysis complete!\n")


if __name__ == '__main__':
    main()