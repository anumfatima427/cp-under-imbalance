#!/usr/bin/env python3
"""
Generalized Conformal Training Script
- Dynamically infers classes and names from CSV files.
- Additive Lagrangian Constraints & Differentiable Set Size.
- JSON logging per epoch.
- Beautified Terminal Dashboard.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
from torch.nn.functional import softmax

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report
)

# --------------------------------------------------
# Args & Config
# --------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Generalized Conformal Image Classifier")
    
    # Dataset paths
    p.add_argument('--train_csv', type=str, required=True, help="Path to training CSV")
    p.add_argument('--test_csv', type=str, required=True, help="Path to test CSV")
    p.add_argument('--train_dir', type=str, required=True, help="Path to training images dir")
    p.add_argument('--test_dir', type=str, required=True, help="Path to test images dir")
    p.add_argument('--output', type=str, default='./model_output/', help="Output directory")
    
    # CSV Column configurations
    p.add_argument('--img_col', type=str, default='image_name', help="CSV column with image filenames")
    p.add_argument('--label_col', type=str, default='label', help="CSV column with integer labels (0 to N-1)")
    p.add_argument('--name_col', type=str, default='label_name', help="CSV column with string class names (optional)")

    # Hyperparameters
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--monitor', type=str, default='val_loss', choices=['val_loss', 'val_acc', 'val_f1', 'val_auc'])
    
    return p.parse_args()


def infer_dataset_metadata(df, label_col, name_col):
    """Dynamically calculates number of classes and their string names."""
    num_classes = df[label_col].nunique()
    
    # Ensure labels are continuous 0 to N-1
    assert set(df[label_col].unique()) == set(range(num_classes)), \
        f"Labels in {label_col} must be continuous integers from 0 to {num_classes-1}."

    if name_col in df.columns:
        mapping = df.drop_duplicates(subset=[label_col]).set_index(label_col)[name_col].to_dict()
        class_names = [mapping[i] for i in range(num_classes)]
    else:
        class_names = [f"Class_{i}" for i in range(num_classes)]
        
    return num_classes, class_names


# --------------------------------------------------
# Conformal Logic
# --------------------------------------------------
class ConformalThresholdUpdater:
    def __init__(self, val_loader, num_classes, epsilon=0.1, temperature=1.1, device="cuda"):
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.temperature = temperature
        self.device = device
        self.thresholds = np.ones(self.num_classes)

    @torch.no_grad()
    def update(self, model):
        model.eval()
        nonconformity = {c: [] for c in range(self.num_classes)}
        all_nc = []

        for images, labels in self.val_loader:
            images = images.to(self.device)
            labels = labels.numpy()

            logits = model(images)
            smoothed_logits = logits / self.temperature
            probs = softmax(smoothed_logits, dim=1).cpu().numpy()

            for p, y in zip(probs, labels):
                nc = 1.0 - p[y]
                nonconformity[y].append(nc)
                all_nc.append(nc)

        n_total = len(all_nc)
        if n_total > 0:
            q_global = np.clip(np.ceil((n_total + 1) * (1 - self.epsilon)) / n_total, 0.0, 1.0)
            global_thresh = np.quantile(all_nc, q_global, method='higher')
        else:
            global_thresh = 0.5

        new_thresholds = np.zeros(self.num_classes)
        for c in range(self.num_classes):
            arr = np.array(nonconformity[c])
            n_c = len(arr)
            
            if n_c < 10:
                new_thresholds[c] = global_thresh
            else:
                q_c = np.clip(np.ceil((n_c + 1) * (1 - self.epsilon)) / n_c, 0.0, 1.0)
                class_thresh = np.quantile(arr, q_c, method='higher')
                alpha_blend = min(1.0, n_c / 200.0) 
                new_thresholds[c] = alpha_blend * class_thresh + (1 - alpha_blend) * global_thresh

        self.thresholds = new_thresholds
        return self.thresholds


class ConformalAwareLoss(nn.Module):
    def __init__(self, num_classes, alpha=1.5, gamma=0.15, temperature=1.5, get_thresholds=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(reduction="none")
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.temperature = temperature
        self.get_thresholds = get_thresholds

    def forward(self, logits, targets, epoch):
        ce_loss = self.ce(logits, targets)  

        smoothed_logits = logits / self.temperature
        probs = softmax(smoothed_logits, dim=1)
        batch_size = logits.shape[0]

        thresholds = self.get_thresholds()
        thresholds = torch.tensor(thresholds, dtype=torch.float32, device=logits.device)

        true_probs = probs[torch.arange(batch_size), targets]
        non_conformity = 1.0 - true_probs
        diffs = torch.clamp(non_conformity - thresholds[targets], min=0.0)
        penalty_margin = self.alpha * diffs 

        k = 50.0 
        margin = thresholds.view(1, -1) - (1.0 - probs)
        soft_cp_sets = torch.sigmoid(k * margin) 
        
        cp_sizes = soft_cp_sets.sum(dim=1) 
        half_classes = self.num_classes / 2.0
        overfull = torch.clamp(cp_sizes - half_classes, min=0.0)
        penalty_size = self.gamma * overfull 

        warmup_factor = np.clip((epoch - 3) / 7.0, 0.0, 1.0) 
        final_loss = ce_loss + warmup_factor * (penalty_margin + penalty_size)

        return final_loss.mean()


# --------------------------------------------------
# Dataset & Utils
# --------------------------------------------------
class GenericImageDataset(Dataset):
    def __init__(self, df, image_dir, img_col, label_col, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.img_col = img_col
        self.label_col = label_col
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = str(self.df.loc[idx, self.img_col])
        img_path = os.path.join(self.image_dir, img_name)
        
        img = Image.open(img_path).convert('RGB')
        label = self.df.loc[idx, self.label_col]

        if self.transform:
            img = self.transform(img)
        return img, label

def get_transforms():
    train_t = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_t = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return train_t, val_t

def make_sampler(labels):
    counts = np.bincount(labels)
    weights = 1.0 / (counts + 1e-8) # added epsilon for safety
    return WeightedRandomSampler(weights[labels], len(labels), replacement=True)

def create_model(num_classes, device):
    model = models.resnet50(weights='DEFAULT')
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)

def compute_metrics(y_true, y_pred, y_prob):
    metrics = {}
    metrics['acc'] = accuracy_score(y_true, y_pred)
    metrics['f1'] = f1_score(y_true, y_pred, average='macro')
    try:
        metrics['auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    except ValueError:
        metrics['auc'] = np.nan
    return metrics


# --------------------------------------------------
# Training Loops
# --------------------------------------------------
def train_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    losses = []
    for x, y in tqdm(loader, desc='  Train', leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y, epoch) 
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)

def eval_epoch(model, loader, criterion, device, epoch):
    model.eval()
    losses, preds, probs, labels = [], [], [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc='  Val', leave=False):
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y, epoch)
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
        if f'train_{metric}' in df.columns and f'val_{metric}' in df.columns:
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
# Main Pipeline
# --------------------------------------------------
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Data & Metadata
    print(f"Loading training data from {args.train_csv}...")
    df = pd.read_csv(args.train_csv)
    
    num_classes, class_names = infer_dataset_metadata(df, args.label_col, args.name_col)
    print(f"Detected {num_classes} classes: {class_names}")

    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df[args.label_col], random_state=args.seed
    )
    test_df = pd.read_csv(args.test_csv)

    # Datasets & Loaders
    train_t, val_t = get_transforms()
    train_ds = GenericImageDataset(train_df, args.train_dir, args.img_col, args.label_col, train_t)
    val_ds = GenericImageDataset(val_df, args.train_dir, args.img_col, args.label_col, val_t)
    test_ds = GenericImageDataset(test_df, args.test_dir, args.img_col, args.label_col, val_t)

    sampler = make_sampler(train_df[args.label_col].values)
    train_loader = DataLoader(train_ds, args.batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, args.batch_size, shuffle=False)

    # Model Initialization
    model = create_model(num_classes, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    
    threshold_updater = ConformalThresholdUpdater(
        val_loader, num_classes=num_classes, epsilon=0.1, temperature=1.5, device=device
    )

    loss_fn = ConformalAwareLoss(
        num_classes=num_classes, 
        alpha=1.5, 
        gamma=0.15, 
        temperature=1.5,
        get_thresholds=lambda: threshold_updater.thresholds
    )

    # Training logic
    best_score = -np.inf if args.monitor != 'val_loss' else np.inf
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device, epoch)
        val_loss, val_metrics = eval_epoch(model, val_loader, loss_fn, device, epoch)
        scheduler.step()

        train_metrics = eval_epoch(model, train_loader, loss_fn, device, epoch)[1]
        
        # Update thresholds and map them to class names for readable JSON output
        new_t = threshold_updater.update(model)
        threshold_dict = {class_names[i]: float(new_t[i]) for i in range(num_classes)}

        row = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
            'thresholds': threshold_dict
        }
        history.append(row)
        
        # Save JSON 
        with open(os.path.join(args.output, 'training_log.json'), 'w') as f:
            json.dump(history, f, indent=4)

        score = row[args.monitor]
        improved = (score < best_score) if args.monitor == 'val_loss' else (score > best_score)
        
        # --- BEAUTIFIED TERMINAL LOGGING ---
        print(f"  Train | Loss: {train_loss:.4f} | Acc: {train_metrics['acc']:.4f} | F1: {train_metrics['f1']:.4f} | AUC: {train_metrics['auc']:.4f}")
        print(f"  Val   | Loss: {val_loss:.4f} | Acc: {val_metrics['acc']:.4f} | F1: {val_metrics['f1']:.4f} | AUC: {val_metrics['auc']:.4f}")
        
        thresh_str = " | ".join([f"{k}: {v:.3f}" for k, v in threshold_dict.items()])
        print(f"  Thresh| {thresh_str}")
        # -----------------------------------

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

    # Save & Test
    hist_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'thresholds'} for r in history])
    hist_df.to_csv(os.path.join(args.output, 'training_log.csv'), index=False)
    
    plot_curves(hist_df, args.output)

    model.load_state_dict(torch.load(os.path.join(args.output, 'best_model.pth')))
    _, test_metrics = eval_epoch(model, test_loader, loss_fn, device, epoch=999)

    print("\nTest Metrics (Best Model)")
    print(test_metrics)

    with open(os.path.join(args.output, 'classification_report.txt'), 'w') as f:
        preds, labels = [], []
        for x, y in test_loader:
            x = x.to(device)
            out = model(x)
            preds.extend(out.argmax(1).cpu().numpy())
            labels.extend(y.numpy())
        f.write(classification_report(labels, preds, target_names=class_names))

if __name__ == "__main__":
    main()