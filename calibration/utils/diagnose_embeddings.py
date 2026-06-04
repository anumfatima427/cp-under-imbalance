"""
Diagnostic script: verify the improved embeddings work on your DDR data.

Run this FIRST before the full experiments to confirm:
1. Embeddings are computed correctly for all classes
2. Clustering produces sensible groupings 
3. The improved method doesn't skip clustering (unlike the original)

Usage:
    python diagnose_embeddings.py

This loads ddr.npz and runs a single configuration to show diagnostics.
"""

import numpy as np
import sys
import os
from collections import Counter

# Add parent directory to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --- Load DDR data ---
# Adjust this path to match your setup
DATA_PATHS = ['data/ddr.npz', '../data/ddr.npz', 'ddr.npz']
data = None
for p in DATA_PATHS:
    if os.path.exists(p):
        data = np.load(p)
        print(f'Loaded data from {p}')
        break

if data is None:
    print('ERROR: Could not find ddr.npz. Place it in ./data/ or current directory.')
    sys.exit(1)

softmax_scores = data['softmax_scores']
labels = data['labels']
num_classes = softmax_scores.shape[1]
N = len(labels)

print(f'\nDataset: DDR')
print(f'  N = {N}, K = {num_classes}')
class_cts = np.array([np.sum(labels == k) for k in range(num_classes)])
print(f'  Class counts: {class_cts}')
print(f'  Imbalance ratio: {max(class_cts)/min(class_cts):.1f}:1')

# --- Simulate a calibration split at n_avg=100 ---
n_avg = 100
n_totalcal = n_avg * num_classes
alpha = 0.1

np.random.seed(42)
cal_idx = np.random.choice(N, size=min(n_totalcal, N), replace=False)
val_idx = np.setdiff1d(np.arange(N), cal_idx)

cal_softmax = softmax_scores[cal_idx]
cal_labels = labels[cal_idx]
val_softmax = softmax_scores[val_idx]
val_labels = labels[val_idx]

# Compute conformal scores (using softmax/THR score for simplicity)
cal_scores_all = 1 - cal_softmax
val_scores_all = 1 - val_softmax

cal_cts = Counter(cal_labels)
print(f'\nCalibration set (n_avg={n_avg}):')
print(f'  Total: {len(cal_labels)}')
print(f'  Per-class: {[cal_cts.get(k, 0) for k in range(num_classes)]}')

# --- Import embedding functions ---
# Try importing from utils/ first, fall back to current directory
try:
    from utils.improved_clustering_utils import (
        confusion_embedding,
        cross_class_score_embedding,
        shrinkage_embedding,
        embed_all_classes_improved,
        get_improved_clustering_parameters,
        select_num_clusters_silhouette,
    )
except ImportError:
    from improved_clustering_utils import (
        confusion_embedding,
        cross_class_score_embedding,
        shrinkage_embedding,
        embed_all_classes_improved,
        get_improved_clustering_parameters,
        select_num_clusters_silhouette,
    )

# Also import original for comparison
try:
    from utils.clustering_utils import embed_all_classes
except ImportError:
    try:
        from clustering_utils import embed_all_classes
    except ImportError:
        embed_all_classes = None


# ============================================================
# 1. COMPARE EMBEDDING STABILITY
# ============================================================
print(f'\n{"="*70}')
print(f' EMBEDDING STABILITY ANALYSIS')
print(f'{"="*70}')

n_trials = 20
embedding_methods = {
    'quantile (original)': lambda s, l, sm: embed_all_classes(s, l, q=[0.5, 0.6, 0.7, 0.8, 0.9], return_cts=True) if embed_all_classes else (None, None),
    'confusion': lambda s, l, sm: confusion_embedding(sm, l, num_classes),
    'cross_class': lambda s, l, sm: cross_class_score_embedding(s, l, num_classes=num_classes),
    'shrinkage': lambda s, l, sm: shrinkage_embedding(s, l, kappa=20, num_classes=num_classes),
}

for method_name, embed_fn in embedding_methods.items():
    print(f'\n--- {method_name} ---')
    
    all_embeddings = []
    for trial in range(n_trials):
        np.random.seed(trial * 100)
        # Resample calibration set
        idx = np.random.choice(N, size=min(n_totalcal, N), replace=False)
        trial_scores = 1 - softmax_scores[idx]
        trial_labels = labels[idx]
        trial_softmax = softmax_scores[idx]
        
        emb, cts = embed_fn(trial_scores, trial_labels, trial_softmax)
        if emb is None:
            print('  (original embedding not available)')
            break
        all_embeddings.append(emb)
    
    if not all_embeddings:
        continue
    
    all_embeddings = np.array(all_embeddings)  # (n_trials, K, D)
    
    # Compute per-class embedding variance
    per_class_std = all_embeddings.std(axis=0).mean(axis=1)  # avg std across dimensions
    
    print(f'  Embedding shape: {all_embeddings[0].shape}')
    print(f'  Per-class avg std across {n_trials} resamples:')
    for y in range(num_classes):
        n_y_avg = int(np.mean([np.sum(labels[np.random.choice(N, n_totalcal, replace=False)] == y) 
                               for _ in range(5)]))
        print(f'    Class {y} (avg n≈{n_y_avg}): std = {per_class_std[y]:.4f}')
    print(f'  Mean stability (lower = more stable): {per_class_std.mean():.4f}')


# ============================================================
# 2. CLUSTERING DIAGNOSTICS
# ============================================================
print(f'\n{"="*70}')
print(f' CLUSTERING DIAGNOSTICS')
print(f'{"="*70}')

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

for emb_method in ['confusion', 'cross_class', 'shrinkage']:
    print(f'\n--- Embedding: {emb_method} ---')
    
    emb, cts = embed_all_classes_improved(
        cal_scores_all, cal_labels,
        method=emb_method,
        softmax_scores=cal_softmax,
        num_classes=num_classes,
        return_cts=True,
    )
    
    print(f'  Embedding shape: {emb.shape}')
    print(f'  Class counts: {cts.astype(int)}')
    
    # Try different numbers of clusters
    print(f'  Silhouette scores:')
    for k in range(2, num_classes):
        km = KMeans(n_clusters=k, random_state=0, n_init=10)
        cluster_labels = km.fit_predict(emb, sample_weight=np.sqrt(cts))
        sil = silhouette_score(emb, cluster_labels)
        print(f'    k={k}: silhouette={sil:.3f}, assignments={cluster_labels}')
    
    # Auto-select
    best_k = select_num_clusters_silhouette(emb, cts, max_clusters=num_classes-1)
    print(f'  Auto-selected k={best_k}')
    
    # Show the actual clustering
    km = KMeans(n_clusters=best_k, random_state=0, n_init=10)
    assignments = km.fit_predict(emb, sample_weight=np.sqrt(cts))
    
    print(f'  Final assignments:')
    for cluster_id in range(best_k):
        classes_in_cluster = np.where(assignments == cluster_id)[0]
        samples_in_cluster = sum(cts[c] for c in classes_in_cluster)
        print(f'    Cluster {cluster_id}: classes {classes_in_cluster} '
              f'(total samples: {int(samples_in_cluster)})')


# ============================================================
# 3. COMPARE WITH ORIGINAL CLUSTERING  
# ============================================================
print(f'\n{"="*70}')
print(f' ORIGINAL vs IMPROVED: DOES CLUSTERING HAPPEN?')
print(f'{"="*70}')

for n_avg_test in [50, 100, 200, 300, 400]:
    n_test = n_avg_test * num_classes
    np.random.seed(0)
    idx = np.random.choice(N, size=min(n_test, N), replace=False)
    test_labels = labels[idx]
    test_cts = Counter(test_labels)
    n_min = min(test_cts.get(k, 0) for k in range(num_classes))
    
    # Original heuristic
    n_thresh = int(np.ceil(1/alpha - 1))
    n_min_eff = max(n_min, n_thresh)
    num_remaining = sum(1 for k in range(num_classes) if test_cts.get(k, 0) >= n_thresh)
    K = num_remaining
    orig_n_clustering = int(n_min_eff * K / (75 + K))
    orig_num_clusters = int(np.floor(orig_n_clustering / 2))
    orig_skip = orig_num_clusters <= 1
    
    # Improved: we always get >= 2 clusters if num_non_rare > 2
    imp_num_clusters = get_improved_clustering_parameters(num_remaining, n_min_eff, alpha)
    imp_skip = imp_num_clusters < 2
    
    print(f'  n_avg={n_avg_test}: n_min={n_min}, non_rare={num_remaining} | '
          f'Original: k={orig_num_clusters} {"(SKIP)" if orig_skip else ""} | '
          f'Improved: k={imp_num_clusters} {"(SKIP)" if imp_skip else ""}')


# ============================================================
# 4. QUICK SANITY CHECK: RUN ONE CONFIGURATION
# ============================================================
print(f'\n{"="*70}')
print(f' SANITY CHECK: Run improved_clustered_conformal once')
print(f'{"="*70}')

try:
    try:
        from utils.improved_conformal import improved_clustered_conformal
    except ImportError:
        from improved_conformal import improved_clustered_conformal
    
    for emb in ['confusion', 'cross_class', 'shrinkage']:
        print(f'\n--- {emb} embedding ---')
        result = improved_clustered_conformal(
            cal_scores_all, cal_labels, alpha,
            val_scores_all=val_scores_all,
            val_labels=val_labels,
            embedding_method=emb,
            softmax_scores_cal=cal_softmax,
            num_clusters='auto',
            split='doubledip',
            seed=42,
            verbose=True,
        )
        
        if len(result) == 4:
            qhats, preds, cov_metrics, size_metrics = result
            print(f'  => CovGap = {cov_metrics["mean_class_cov_gap"]:.4f}')
            print(f'  => AvgSize = {size_metrics["mean"]:.3f}')
            print(f'  => MargCov = {cov_metrics["marginal_cov"]:.3f}')
            raw = cov_metrics.get('raw_class_coverages', [])
            if len(raw) > 0:
                print(f'  => Per-class coverage: {np.round(raw, 3)}')
        else:
            print(f'  => (fell back to standard, result has {len(result)} items)')

except Exception as e:
    print(f'  ERROR running improved_clustered_conformal: {e}')
    import traceback
    traceback.print_exc()

print(f'\n{"="*70}')
print(f' DIAGNOSTICS COMPLETE')
print(f'{"="*70}')
print(f'\nIf the above looks reasonable:')
print(f'  1. Confusion/cross-class embeddings should be more stable than quantile')
print(f'  2. Improved method should NOT skip clustering')
print(f'  3. CovGap should be lower than standard conformal')
print(f'\nNext: run the full experiments with run_improved.py')
