#!/usr/bin/env python3
"""
DDR Training Script — Baseline (CrossEntropyLoss) & CP Readiness
- Train/Val split from train.csv (Stratified)
- Data summary post-split
- Comprehensive Train & Val metric logging per epoch
- Best model tracking in CSV
- Aspect-ratio preserving transforms for morphological integrity
- Layer-wise learning rates for stable transfer learning
- Early stopping & Best model selection
- Conformal Prediction (CP) extraction (.npz files)
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report


# --------------------------------------------------
# Configuration
# --------------------------------------------------
CLASS_NAMES = ['None', 'Infection', 'Ischaemia', 'Both']
NUM_CLASSES = len(CLASS_NAMES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline ResNet50 for DFUC2021 with CP setup")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to dataset root")
    parser.add_argument('--output', type=str, default='./model_output/baseline_ce', help="Output directory")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--patience', type=int, default=10, help="Early stopping patience")
    parser.add_argument('--monitor', type=str, default='val_auc', 
                        choices=['val_loss', 'val_acc', 'val_f1', 'val_auc'])
    return parser.parse_args()


# --------------------------------------------------
# Data Pipeline
# --------------------------------------------------
class DDRDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_dir: Path, transform: transforms.Compose = None):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_dir / self.df.loc[idx, 'image']
        img = Image.open(img_path).convert('RGB')
        label = self.df.loc[idx, 'label']

        if self.transform:
            img = self.transform(img)
        return img, label


def get_transforms(img_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    """Preserves aspect ratio before cropping to maintain lesion morphology."""
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    train_t = transforms.Compose([
        transforms.Resize(256), 
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        normalize
    ])

    val_t = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize
    ])
    return train_t, val_t


def make_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    counts = np.bincount(labels)
    weights = 1.0 / counts
    sample_weights = weights[labels]
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


# --------------------------------------------------
# Modeling & Metrics
# --------------------------------------------------
def create_model(device: torch.device) -> nn.Module:
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model.to(device)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    metrics = {
        'acc': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, average='macro', zero_division=0)
    }
    try:
        metrics['auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    except ValueError:
        metrics['auc'] = np.nan
    return metrics


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, 
                criterion: nn.Module, device: torch.device) -> float:
    model.train()
    losses = []

    for x, y in tqdm(loader, desc='  Train', leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())

    return np.mean(losses)


@torch.no_grad()
def eval_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, 
               device: torch.device) -> Tuple[float, Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    losses, preds, probs, labels = [], [], [], []

    for x, y in tqdm(loader, desc='  Eval', leave=False):
        x, y = x.to(device), y.to(device)
        out = model(x)

        if criterion:
            losses.append(criterion(out, y).item())

        prob = torch.softmax(out, dim=1)
        preds.extend(prob.argmax(1).cpu().numpy())
        probs.extend(prob.cpu().numpy())
        labels.extend(y.cpu().numpy())

    probs_arr, labels_arr, preds_arr = np.array(probs), np.array(labels), np.array(preds)
    metrics = compute_metrics(labels_arr, preds_arr, probs_arr)
    avg_loss = np.mean(losses) if losses else 0.0

    return avg_loss, metrics, probs_arr, labels_arr, preds_arr


# --------------------------------------------------
# Utilities
# --------------------------------------------------
def plot_curves(df: pd.DataFrame, out_dir: Path) -> None:
    for metric in ['loss', 'acc', 'f1', 'auc']:
        plt.figure()
        plt.plot(df['epoch'], df[f'train_{metric}'], label='Train')
        plt.plot(df['epoch'], df[f'val_{metric}'], label='Val')
        
        # Highlight best epoch
        if 'is_best' in df.columns and df['is_best'].any():
            best_epoch = df[df['is_best']]['epoch'].values[0]
            best_val = df[df['is_best']][f'val_{metric}'].values[0]
            plt.scatter(best_epoch, best_val, color='red', zorder=5, label=f'Best Epoch ({best_epoch})')

        plt.title(metric.upper())
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(out_dir / f'{metric}_curve.png', dpi=300)
        plt.close()


# --------------------------------------------------
# Main Pipeline
# --------------------------------------------------
def main() -> None:
    args = parse_args()
    
    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Setup paths
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load and split data
    train_csv = data_dir / 'train' / 'train.csv'
    train_imgs = data_dir / 'train' / 'images'
    test_csv = data_dir / 'test' / 'test.csv'
    test_imgs = data_dir / 'test' / 'images'

    full_train_df = pd.read_csv(train_csv)
    train_df, val_df = train_test_split(
        full_train_df, test_size=0.2, stratify=full_train_df['label'], random_state=args.seed
    )

    # --- Print Data Summary After Split ---
    print("\n" + "="*45)
    print(" DATA SUMMARY AFTER STRATIFIED SPLIT")
    print("="*45)
    
    train_counts = train_df['label'].value_counts().reindex(range(NUM_CLASSES), fill_value=0)
    val_counts = val_df['label'].value_counts().reindex(range(NUM_CLASSES), fill_value=0)
    
    summary_df = pd.DataFrame({
        'Class Name': CLASS_NAMES,
        'Label Index': list(range(NUM_CLASSES)),
        'Train Count': train_counts.values,
        'Val Count': val_counts.values
    })
    print(summary_df.to_string(index=False))
    print("="*45 + "\n")

    # Datasets
    train_t, val_t = get_transforms()
    train_ds = DDRDataset(train_df, train_imgs, train_t)
    val_ds = DDRDataset(val_df, train_imgs, val_t)
    test_ds = DDRDataset(pd.read_csv(test_csv), test_imgs, val_t)

    # Dataloaders
    sampler = make_sampler(train_df['label'].values)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=4)
    train_eval_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=4) 
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model, Optimizer, Scheduler
    model = create_model(device)
    
    # Layer-wise learning rates to protect pre-trained weights
    optimizer = torch.optim.Adam([
        {'params': model.conv1.parameters(), 'lr': args.lr * 0.1},
        {'params': model.layer1.parameters(), 'lr': args.lr * 0.1},
        {'params': model.layer2.parameters(), 'lr': args.lr * 0.1},
        {'params': model.layer3.parameters(), 'lr': args.lr},
        {'params': model.layer4.parameters(), 'lr': args.lr},
        {'params': model.fc.parameters(), 'lr': args.lr * 10}
    ], lr=args.lr)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    # Tracking
    best_score = -np.inf if args.monitor != 'val_loss' else np.inf
    patience_counter = 0
    history = []

    # ---------------- Training Loop ----------------
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()

        # Evaluate on unaltered training set and validation set
        _, train_metrics, _, _, _ = eval_epoch(model, train_eval_loader, criterion, device)
        val_loss, val_metrics, val_probs, val_labels, _ = eval_epoch(model, val_loader, criterion, device)

        # Print all metrics for Train and Val
        print(f"Train | Loss: {train_loss:.4f} | Acc: {train_metrics['acc']:.4f} | F1: {train_metrics['f1']:.4f} | AUC: {train_metrics['auc']:.4f}")
        print(f"Val   | Loss: {val_loss:.4f} | Acc: {val_metrics['acc']:.4f} | F1: {val_metrics['f1']:.4f} | AUC: {val_metrics['auc']:.4f}")

        # Check for improvement
        score = val_loss if args.monitor == 'val_loss' else val_metrics[args.monitor.replace('val_', '')]
        improved = (score < best_score) if args.monitor == 'val_loss' else (score > best_score)

        row = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
            'is_best': False # Default initialization
        }

        if improved:
            best_score = score
            patience_counter = 0
            
            # Update history logic for best model
            for h in history:
                h['is_best'] = False
            row['is_best'] = True
            
            # Save artifacts
            torch.save(model.state_dict(), out_dir / 'best_model.pth')
            np.savez(out_dir / 'cp_calibration_data.npz', probs=val_probs, labels=val_labels)
            
            print("  ✔ New best model & CP calibration data saved.")
        else:
            patience_counter += 1
            print(f"  EarlyStopping: {patience_counter}/{args.patience}")

        history.append(row)

        if patience_counter >= args.patience:
            print("\n⏹ Early stopping triggered.")
            break

    # ---------------- Final Testing & Logging ----------------
    # Save CSV with the 'is_best' boolean flag
    pd.DataFrame(history).to_csv(out_dir / 'training_log.csv', index=False)
    plot_curves(pd.DataFrame(history), out_dir)

    print("\nLoading best model for testing...")
    model.load_state_dict(torch.load(out_dir / 'best_model.pth'))
    
    _, test_metrics, test_probs, test_labels, test_preds = eval_epoch(model, test_loader, criterion, device)

    print("\nTest Metrics (Best Model):")
    for k, v in test_metrics.items():
        print(f"  {k.upper()}: {v:.4f}")

    with open(out_dir / 'classification_report.txt', 'w') as f:
        f.write(classification_report(test_labels, test_preds, target_names=CLASS_NAMES))

    np.savez(out_dir / 'cp_test_data.npz', probs=test_probs, labels=test_labels)
    print(f"\n✔ Pipeline complete. Check `{out_dir.name}` for outputs and CP data.")


if __name__ == "__main__":
    main()