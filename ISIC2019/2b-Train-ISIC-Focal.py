import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from tqdm import tqdm  # Added for progress bars

# --------------------------------------------------
# 1. SETUP & CONFIGURATION
# --------------------------------------------------
DATA_DIR = "./isic_train_val_split"
OUTPUT_DIR = "./model_output/isic_focal"
os.makedirs(OUTPUT_DIR, exist_ok=True)

EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
GAMMA = 2.0  # Fixed the hidden syntax character here

CLASS_NAMES = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
NUM_CLASSES = len(CLASS_NAMES)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --------------------------------------------------
# 2. DATASET & TRANSFORMS
# --------------------------------------------------
class ISICDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['image']
        img_path = os.path.join(self.image_dir, img_name)
        
        image = Image.open(img_path).convert('RGB')
        label = int(self.df.iloc[idx]['label'])

        if self.transform:
            image = self.transform(image)
            
        return image, label

# Simple, standard image transformations
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create datasets and data loaders
train_dataset = ISICDataset(f"{DATA_DIR}/train/train.csv", f"{DATA_DIR}/train/images", train_transform)
val_dataset = ISICDataset(f"{DATA_DIR}/val/val.csv", f"{DATA_DIR}/val/images", val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# Calculate class weights for Focal Loss based on training frequencies
train_df = pd.read_csv(f"{DATA_DIR}/train/train.csv")
counts = train_df['label'].value_counts().reindex(range(NUM_CLASSES), fill_value=0).values
reciprocal_counts = 1.0 / np.maximum(counts, 1)
alpha_weights = torch.tensor(reciprocal_counts / np.sum(reciprocal_counts), dtype=torch.float32).to(DEVICE)

# --------------------------------------------------
# 3. FOCAL LOSS & MODEL MODEL
# --------------------------------------------------
def focal_loss(inputs, targets, alpha, gamma):
    """Simple vectorized multi-class Focal Loss."""
    log_p = F.log_softmax(inputs, dim=1)
    log_pt = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)
    pt = log_pt.exp()
    
    focal_weight = (1 - pt) ** gamma
    loss = -focal_weight * log_pt
    
    if alpha is not None:
        alpha_t = alpha.gather(0, targets)
        loss = alpha_t * loss
        
    return loss.mean()

# Load a standard pretrained ResNet50
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)

# Simple Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --------------------------------------------------
# 4. TRAINING & EVALUATION FUNCTIONS
# --------------------------------------------------
def train_one_epoch():
    model.train()
    total_loss = 0
    # Added tqdm progress bar for the training loop
    for images, labels in tqdm(train_loader, desc="  Training", leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = focal_loss(outputs, labels, alpha_weights, GAMMA)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate(loader, desc="Evaluating"):
    model.eval()
    total_loss = 0
    all_labels = []
    all_preds = []
    all_probs = []
    
    # Added tqdm progress bar for the evaluation loops
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"  {desc}", leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            
            loss = focal_loss(outputs, labels, alpha_weights, GAMMA)
            total_loss += loss.item()
            
            probs = F.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    avg_loss = total_loss / len(loader)
    
    # Calculate simple metrics
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    except:
        auc = 0.0
        
    return avg_loss, acc, f1, auc, all_labels, all_preds

# --------------------------------------------------
# 5. MAIN LOOP
# --------------------------------------------------
best_val_auc = 0.0
history = []

print(f"Starting training on {DEVICE}...")
for epoch in range(1, EPOCHS + 1):
    print(f"\n--- Epoch {epoch}/{EPOCHS} ---")
    
    train_loss = train_one_epoch()
    
    # Passing descriptor string to show distinct progress bars
    val_loss, val_acc, val_f1, val_auc, _, _ = evaluate(val_loader, desc="Val Eval")
    _, train_acc, train_f1, train_auc, _, _ = evaluate(train_loader, desc="Train Eval")
    
    # Explicit breakdown of Threshold strategy and both sets of metrics per epoch
    print(f"  Decision Threshold Rule : argmax")
    print(f"  Train Metrics           -> Loss: {train_loss:.4f} | AUC: {train_auc:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
    print(f"  Val Metrics             -> Loss: {val_loss:.4f}  | AUC: {val_auc:.4f}  | Acc: {val_acc:.4f}  | F1: {val_f1:.4f}")
    
    # Save statistics to dictionary
    row = {
        'epoch': epoch,
        'threshold': 'argmax',
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'train_f1': train_f1,
        'val_f1': val_f1,
        'train_auc': train_auc,
        'val_auc': val_auc
    }
    history.append(row)
    
    # Overwrite the CSV after every single epoch
    pd.DataFrame(history).to_csv(f"{OUTPUT_DIR}/training_log.csv", index=False)
    
    # Save the best model based on validation AUC score
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        torch.save(model.state_dict(), f"{OUTPUT_DIR}/best_model.pth")
        print("  ✔ New best model saved based on Val AUC!")

# --------------------------------------------------
# 6. FINAL PLOTS & REPORT
# --------------------------------------------------
# Plot training metrics curves
df = pd.DataFrame(history)
for metric in ['loss', 'acc', 'f1', 'auc']:
    plt.figure()
    plt.plot(df['epoch'], df[f'train_{metric}'], label='Train')
    plt.plot(df['epoch'], df[f'val_{metric}'], label='Val')
    plt.title(metric.upper())
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{OUTPUT_DIR}/{metric}_curve.png")
    plt.close()

# Load the best model to generate final report
print("\nLoading best model for final report...")
model.load_state_dict(torch.load(f"{OUTPUT_DIR}/best_model.pth"))
_, _, _, _, final_labels, final_preds = evaluate(val_loader, desc="Final Eval")

# Save text report
with open(f"{OUTPUT_DIR}/classification_report.txt", "w") as f:
    f.write(classification_report(final_labels, final_preds, target_names=CLASS_NAMES))

print("Pipeline finished successfully!")
