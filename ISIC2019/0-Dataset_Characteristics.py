#!/usr/bin/env python3
"""
ISIC 2019 Dataset Imbalance Analysis
Skin Lesion Classification dataset
Classes: MEL, NV, BCC, AK, BKL, DF, VASC, SCC, UNK
"""

import numpy as np
import pandas as pd
from scipy.stats import entropy
import argparse

# ISIC 2019 Official Classes and corresponding CSV Columns
ISIC_COLS = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']
ISIC_CLASSES = [
    'Melanoma', 
    'Nevus', 
    'Basal Cell Carc.', 
    'Actinic Keratosis', 
    'Benign Keratosis', 
    'Dermatofibroma', 
    'Vascular Lesion', 
    'Squamous Cell Carc.', 
    'Unknown (OOD)'
]

def load_labeled(csv_path):
    """Load CSV and gracefully handle missing label columns."""
    df = pd.read_csv(csv_path)
    
    # Check which ISIC columns actually exist in the provided CSV
    existing_cols = [col for col in ISIC_COLS if col in df.columns]
    
    if not existing_cols:
        raise ValueError(
            f"ERROR: None of the expected label columns {ISIC_COLS} were found in {csv_path}.\n"
            f"Are you sure you provided the Ground Truth file and not the Metadata file?"
        )
        
    labeled = df.dropna(subset=existing_cols).copy()
    labeled[existing_cols] = labeled[existing_cols].astype(int)
    
    return labeled, existing_cols

def analyze_imbalance(csv_path, dataset_name='Dataset'):
    """Analyze class imbalance for ISIC dataset."""
    df, existing_cols = load_labeled(csv_path)
    total = len(df)

    existing_classes = [ISIC_CLASSES[ISIC_COLS.index(col)] for col in existing_cols]

    # Per-class positive counts
    counts = [int(df[col].sum()) for col in existing_cols]

    # Sanity check: confirm each row sums to 1 (mutually exclusive labels)
    row_sums = df[existing_cols].sum(axis=1)
    is_exclusive = (row_sums == 1).all()

    # Filter out classes with 0 counts to prevent ZeroDivisionError/Infinity
    non_zero_counts = [c for c in counts if c > 0]
    min_count = min(non_zero_counts) if non_zero_counts else 0
    max_count = max(counts) if counts else 0
    
    imbalance_ratio = max_count / min_count if min_count > 0 else 0
    shannon = entropy(non_zero_counts) / np.log(len(non_zero_counts)) if len(non_zero_counts) > 1 else 0.0
    percentages = [(c / total) * 100 for c in counts]

    # ── Print results ──────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  {dataset_name}")
    print(f"{'='*80}")
    print(f"Total labeled samples : {total:,}")
    print(f"Labels are mutually exclusive: {is_exclusive}")
    print(f"\nClass Distribution:")
    print(f"{'Code':<6} {'Name':<22} {'Count':>8} {'Percent':>10}  Bar")
    print(f"{'-'*80}")

    for code, name, count, pct in zip(existing_cols, existing_classes, counts, percentages):
        bar = '█' * int(pct / 2) 
        print(f"{code:<6} {name:<22} {count:>8,} {pct:>9.2f}%  {bar}")

    print(f"{'-'*80}")
    print(f"{'Total':<29} {total:>8,}")

    print(f"\n{'='*80}")
    print(f"Imbalance Metrics:")
    print(f"{'='*80}")
    print(f"  Imbalance Ratio (IR):         {imbalance_ratio:.2f}x")
    print(f"    └─ Largest / Smallest:      {max_count:,} / {min_count:,} (excluding 0-count classes)")
    print(f"\n  Shannon Entropy (normalized):  {shannon:.4f}")
    print(f"    └─ 1.0 = perfectly balanced, 0.0 = totally imbalanced")
    print(f"\n  Majority Class Dominance:     {max(percentages):.1f}%")
    
    min_pct = min([pct for pct in percentages if pct > 0]) if any(pct > 0 for pct in percentages) else 0.0
    print(f"  Minority Class Coverage:      {min_pct:.1f}% (excluding 0-count classes)")

    if imbalance_ratio < 2:
        severity = "LOW — Nearly balanced"
    elif imbalance_ratio < 10:
        severity = "MODERATE — Some imbalance"
    elif imbalance_ratio < 50:
        severity = "HIGH — Significant imbalance"
    else:
        severity = "EXTREME — Severe imbalance"

    print(f"\n  Imbalance Severity:           {severity}")
    print(f"{'='*80}\n")

    return {
        'df': df,
        'cols': existing_cols,
        'classes': existing_classes,
        'counts': counts,
        'percentages': percentages,
        'total': total,
    }

def create_comparison_table(train_stats, test_stats):
    """Print a train vs test comparison table."""
    print(f"\n{'='*95}")
    print(f"{'TRAIN vs TEST COMPARISON':^95}")
    print(f"{'='*95}")
    print(f"{'Code':<6} {'Name':<22} {'Train Count':>12} {'Train %':>10} "
          f"{'Test Count':>12} {'Test %':>10}")
    print(f"{'-'*95}")

    cols = train_stats['cols']
    classes = train_stats['classes']

    for i, (code, name) in enumerate(zip(cols, classes)):
        test_idx = test_stats['cols'].index(code) if code in test_stats['cols'] else -1
        test_count = test_stats['counts'][test_idx] if test_idx >= 0 else 0
        test_pct = test_stats['percentages'][test_idx] if test_idx >= 0 else 0.0

        print(f"{code:<6} {name:<22} {train_stats['counts'][i]:>12,} "
              f"{train_stats['percentages'][i]:>9.2f}% "
              f"{test_count:>12,} "
              f"{test_pct:>9.2f}%")

    print(f"{'-'*95}")
    print(f"{'Total':<29} {train_stats['total']:>12,} {'100.00%':>10} "
          f"{test_stats['total']:>12,} {'100.00%':>10}")
    print(f"{'='*95}\n")

def main():
    parser = argparse.ArgumentParser(description='ISIC 2019 Dataset Imbalance Analysis')
    parser.add_argument('--train_csv', default='train.csv',
                        help='Path to train CSV (one-hot encoded labels)')
    parser.add_argument('--test_csv', default=None,
                        help='Path to test CSV (optional)')
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("  ISIC 2019 DATASET IMBALANCE ANALYSIS")
    print("=" * 80)

    try:
        train_stats = analyze_imbalance(args.train_csv, 'TRAIN SET')
    except ValueError as e:
        print(e)
        return

    if args.test_csv:
        try:
            test_stats = analyze_imbalance(args.test_csv, 'TEST SET')
            create_comparison_table(train_stats, test_stats)
        except ValueError as e:
            print(f"Error processing test CSV: {e}")

if __name__ == '__main__':
    main()