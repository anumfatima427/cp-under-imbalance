#!/usr/bin/env python3
"""
Generate .npz Predictions for Conformal Evaluation
Loads a trained PyTorch model, runs inference on a test dataset, 
and saves the softmax probabilities and true labels to an .npz file.
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

# --------------------------------------------------
# Args
# --------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Generate .npz predictions for conformal evaluation")
    
    # Dataset and Model paths
    p.add_argument('--test_csv', type=str, required=True, help="Path to test CSV")
    p.add_argument('--test_dir', type=str, required=True, help="Path to test images dir")
    p.add_argument('--model_path', type=str, required=True, help="Path to best_model.pth")
    p.add_argument('--output_npz', type=str, default='./isic-cp-aware.npz', help="Output .npz file path")
    
    # Dataset config
    p.add_argument('--img_col', type=str, default='image', help="Column with image filenames")
    p.add_argument('--label_col', type=str, default='label', help="Column with integer labels")
    p.add_argument('--batch_size', type=int, default=32)
    
    # IMPORTANT: Temperature matching
    p.add_argument('--temperature', type=float, default=1.1, 
                   help="MUST match the temperature used during training! (e.g., 1.1 for ISIC)")

    return p.parse_args()

# --------------------------------------------------
# Dataset & Model Setup
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

def get_test_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def create_model(num_classes, device):
    # UPDATED for DenseNet121
    model = models.densenet121(weights=None) 
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model.to(device)

def infer_num_classes(df, label_col):
    return df[label_col].nunique()

# --------------------------------------------------
# Main Generation Logic
# --------------------------------------------------
def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Data
    print(f"Loading test data from {args.test_csv}...")
    df = pd.read_csv(args.test_csv)
    num_classes = infer_num_classes(df, args.label_col)
    print(f"Detected {num_classes} classes.")

    test_t = get_test_transform()
    test_ds = GenericImageDataset(df, args.test_dir, args.img_col, args.label_col, transform=test_t)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # Load Model
    print(f"Loading model weights from {args.model_path}...")
    model = create_model(num_classes, device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # Collect predictions
    all_probs = []
    all_labels = []

    print(f"Extracting probabilities (Temperature Scaling T={args.temperature})...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            
            logits = model(images)
            
            # Apply the SAME temperature scaling used during training
            smoothed_logits = logits / args.temperature 
            probs = torch.softmax(smoothed_logits, dim=1)
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Convert to Numpy Arrays
    all_probs = np.array(all_probs, dtype=np.float32)
    all_labels = np.array(all_labels, dtype=np.int64)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output_npz)), exist_ok=True)

    # Save to .npz format expected by experiment_utils.py
    print(f"Saving results to {args.output_npz}...")
    
    # FIXED: Replaced 'probs=' with 'softmax='
    np.savez(
        args.output_npz, 
        softmax=all_probs, 
        labels=all_labels
    )
    print("Done! You can now run the conformal evaluation scripts on this .npz file.")

if __name__ == "__main__":
    main()