import os
import copy
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import models

from dataset import create_data_loaders
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from tqdm import tqdm  

# ==================== Configuration ====================

DATA_DIR = r"E:\master3\Improvment\kidney_stone_detection\Dataset"

SAVE_DIR = "./checkpoints"

LOG_DIR = "./logs"

BATCH_SIZE = 8
EPOCHS = 20
LR = 1e-3
VAL_SPLIT = 0.2
NUM_WORKERS = 4
SEED = 42
# =======================================================

def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    """
    انجام یک epoch آموزش با نمایش نوار پیشرفت و بروزرسانی loss/accuracy
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
  
    pbar = tqdm(loader, desc=f"Train Epoch {epoch}/{EPOCHS}", unit="batch")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    
        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += batch_size

   
        epoch_loss = running_loss / total
        epoch_acc = correct / total
     
        pbar.set_postfix(loss=f"{epoch_loss:.4f}", acc=f"{epoch_acc:.4f}")

    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
   
    model.eval()
    running_loss = 0.0
    preds_all = []
    labels_all = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            _, preds = torch.max(outputs, 1)
            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())

    total = len(labels_all)
    epoch_loss = running_loss / total
    epoch_acc = np.sum(np.array(preds_all) == np.array(labels_all)) / total
    report = classification_report(labels_all, preds_all, target_names=['Normal','Stone'], digits=4)
    cm = confusion_matrix(labels_all, preds_all)
    return epoch_loss, epoch_acc, report, cm


def main():
    
    torch.manual_seed(SEED)
    np.random.seed(SEED)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  
    train_loader, val_loader = create_data_loaders(
        DATA_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        val_split=VAL_SPLIT,
        random_seed=SEED
    )


    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    model = model.to(device)

 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(LOG_DIR, f"run_{timestamp}")
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(1, EPOCHS+1):
        
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
     
        val_loss, val_acc, report, cm = validate(model, val_loader, criterion, device)

        
        writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
        writer.add_scalars('Accuracy', {'train': train_acc, 'val': val_acc}, epoch)

       
        print(f"\nEpoch {epoch}/{EPOCHS} Summary:")
        print(f"  Train   - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Validate- Loss: {val_loss:.4f}, Acc: {val_acc:.4f}\n")

        print("Classification Report:")
        print(report)
        print("Confusion Matrix:")
        print(cm)

   
        scheduler.step(val_acc)
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, os.path.join(SAVE_DIR, 'best_model.pth'))
            print(f"Saved best model (Acc: {best_acc:.4f})\n")

    writer.close()
    print(f"Training complete. Best Val Acc: {best_acc:.4f}")


if __name__ == '__main__':
    main()

