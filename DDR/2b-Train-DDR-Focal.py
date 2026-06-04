#!/usr/bin/env python3
"""
DDR Training Script — Baseline (FocalLoss)
- Train/Val split from train.csv
- Early stopping
- Best model saving
- Metrics: Acc, Macro-F1, Macro-AUC

This is the focal loss baseline for comparison against CE and CP-aware scripts.
Everything is identical except the loss function.

Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
  - Downweights easy/well-classified examples
  - Focuses training on hard, misclassified samples
  - Particularly useful for class imbalance
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report
)


# --------------------------------------------------
# Config
# --------------------------------------------------
CLASS_NAMES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
NUM_CLASSES = 5


# --------------------------------------------------
# Args
# --------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, required=True)
    p.add_argument('--output', type=str, default='./model_output_baseline_focal')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=42)

    # Focal loss hyperparameters
    p.add_argument('--focal_gamma', type=float, default=2.0,
                   help='Focusing parameter — higher = more focus on hard examples')
    p.add_argument('--focal_alpha', type=float, nargs='+', default=None,
                   help='Per-class weights (5 values). If not set, computed from class frequencies.')

    # Early stopping / model selection
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--monitor', type=str, default='val_auc',
                   choices=['val_loss', 'val_acc', 'val_f1', 'val_auc'])
    return p.parse_args()


# --------------------------------------------------
# Focal Loss
# --------------------------------------------------
class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma (float):  Focusing parameter. gamma=0 recovers standard CE.
                        gamma=2 is the original paper default.
        alpha (Tensor): Per-class weights of shape (num_classes,).
                        If None, all classes weighted equally.
        reduction (str): 'mean' or 'none'.
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        if alpha is not None:
            self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, logits, targets):
        # Compute softmax probabilities
        probs = F.softmax(logits, dim=1)                        # (B, C)
        batch_size = logits.shape[0]

        # Probability of the true class
        p_t = probs[torch.arange(batch_size), targets]          # (B,)

        # Standard CE component: -log(p_t)
        ce_loss = -torch.log(p_t + 1e-8)

        # Focal modulating factor: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Per-class alpha weighting
        if self.alpha is not None:
            alpha_t = self.alpha[targets]                       # (B,)
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss


# --------------------------------------------------
# Dataset
# --------------------------------------------------
class DDRDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img = Image.open(
            os.path.join(self.image_dir, self.df.loc[idx, 'image_name'])
        ).convert('RGB')
        label = self.df.loc[idx, 'label']

        if self.transform:
            img = self.transform(img)
        return img, label


# --------------------------------------------------
# Transforms
# --------------------------------------------------
def get_transforms():
    train_t = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    val_t = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return train_t, val_t


# --------------------------------------------------
# Sampler
# --------------------------------------------------
def make_sampler(labels):
    counts = np.bincount(labels)
    weights = 1.0 / counts
    return WeightedRandomSampler(weights[labels], len(labels), replacement=True)


# --------------------------------------------------
# Model
# --------------------------------------------------
def create_model(device):
    model = models.resnet50(weights='DEFAULT')
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model.to(device)


# --------------------------------------------------
# Metrics
# --------------------------------------------------
def compute_metrics(y_true, y_pred, y_prob):
    metrics = {}
    metrics['acc'] = accuracy_score(y_true, y_pred)
    metrics['f1'] = f1_score(y_true, y_pred, average='macro')

    try:
        metrics['auc'] = roc_auc_score(
            y_true, y_prob, multi_class='ovr', average='macro'
        )
    except ValueError:
        metrics['auc'] = np.nan

    return metrics


# --------------------------------------------------
# Train / Eval
# --------------------------------------------------
def train_epoch(model, loader, optimizer, criterion, device):
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


def eval_epoch(model, loader, criterion, device):
    model.eval()
    losses, preds, probs, labels = [], [], [], []

    with torch.no_grad():
        for x, y in tqdm(loader, desc='  Val', leave=False):
            x, y = x.to(device), y.to(device)
            out = model(x)

            loss = criterion(out, y)
            losses.append(loss.item())

            prob = torch.softmax(out, dim=1)
            preds.extend(prob.argmax(1).cpu().numpy())
            probs.extend(prob.cpu().numpy())
            labels.extend(y.cpu().numpy())

    metrics = compute_metrics(labels, preds, probs)
    return np.mean(losses), metrics


# --------------------------------------------------
# Plotting
# --------------------------------------------------
def plot_curves(df, out_dir):
    for metric in ['loss', 'acc', 'f1', 'auc']:
        plt.figure()
        plt.plot(df['epoch'], df[f'train_{metric}'], label='Train')
        plt.plot(df['epoch'], df[f'val_{metric}'], label='Val')
        plt.title(metric.upper())
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(out_dir, f'{metric}_curve.png'), dpi=300)
        plt.close()


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---------------- Load data ----------------
    train_csv = os.path.join(args.data_dir, 'train', 'train.csv')
    train_imgs = os.path.join(args.data_dir, 'train', 'images')
    test_csv = os.path.join(args.data_dir, 'test', 'test.csv')
    test_imgs = os.path.join(args.data_dir, 'test', 'images')

    df = pd.read_csv(train_csv)
    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df['label'], random_state=args.seed
    )

    train_t, val_t = get_transforms()
    train_ds = DDRDataset(train_df, train_imgs, train_t)
    val_ds = DDRDataset(val_df, train_imgs, val_t)
    test_ds = DDRDataset(pd.read_csv(test_csv), test_imgs, val_t)

    sampler = make_sampler(train_df['label'].values)

    train_loader = DataLoader(train_ds, args.batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, args.batch_size, shuffle=False)

    # ---------------- Model ----------------
    model = create_model(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # ---------------- Focal Loss ----------------
    # Compute inverse-frequency alpha weights from training set if not provided
    if args.focal_alpha is not None:
        alpha = args.focal_alpha
        assert len(alpha) == NUM_CLASSES, \
            f"Expected {NUM_CLASSES} alpha values, got {len(alpha)}"
    else:
        counts = np.bincount(train_df['label'].values, minlength=NUM_CLASSES)
        alpha = (1.0 / counts)
        alpha = alpha / alpha.sum() * NUM_CLASSES   # normalise so they sum to NUM_CLASSES

    print(f"Focal Loss | gamma={args.focal_gamma}  alpha={[f'{a:.3f}' for a in alpha]}")

    criterion = FocalLoss(
        gamma=args.focal_gamma,
        alpha=alpha,
    ).to(device)

    # ---------------- Training loop ----------------
    best_score = -np.inf if args.monitor != 'val_loss' else np.inf
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        train_metrics = eval_epoch(model, train_loader, criterion, device)[1]

        row = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()}
        }
        history.append(row)

        score = row[args.monitor]
        improved = (score < best_score) if args.monitor == 'val_loss' else (score > best_score)

        print(f"Val | Acc {row['val_acc']:.4f} | F1 {row['val_f1']:.4f} | AUC {row['val_auc']:.4f}")

        if improved:
            best_score = score
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(args.output, 'best_model.pth'))
            print("  ✔ Best model saved")
        else:
            patience_counter += 1
            print(f"  EarlyStopping: {patience_counter}/{args.patience}")

        if patience_counter >= args.patience:
            print("\n⏹ Early stopping triggered")
            break

    # ---------------- Save & Test ----------------
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(os.path.join(args.output, 'training_log.csv'), index=False)
    plot_curves(hist_df, args.output)

    model.load_state_dict(torch.load(os.path.join(args.output, 'best_model.pth')))
    _, test_metrics = eval_epoch(model, test_loader, criterion, device)

    print("\nTest Metrics (Best Model)")
    print(test_metrics)

    with open(os.path.join(args.output, 'classification_report.txt'), 'w') as f:
        preds, labels = [], []
        for x, y in test_loader:
            x = x.to(device)
            out = model(x)
            preds.extend(out.argmax(1).cpu().numpy())
            labels.extend(y.numpy())
        f.write(classification_report(labels, preds, target_names=CLASS_NAMES))


if __name__ == "__main__":
    main()
