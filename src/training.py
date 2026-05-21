# -*- coding: utf-8 -*-
"""
Facial Expression Recognition — RTX 4050 Optimised
====================================================
Optimisations applied:
  • Mixed-precision FP16 (autocast + GradScaler)   → ~2x throughput, half VRAM
  • torch.compile                                   → 10–30% free speedup
  • TF32 matmul/conv (Ampere GPU)                  → ~20% faster
  • Fused Adam / SGD kernels                        → faster parameter updates
  • non_blocking transfers + persistent_workers     → GPU always fed
  • Atomic checkpoint writes                        → safe overnight saves
  • Per-epoch ETA + VRAM display                    → know where you are in the morning
  • Plots saved to files (no plt.show blocking)     → safe for overnight terminal runs
"""

# ── 1. Imports ────────────────────────────────────────────────────────────────
import os
import json
import time
import pickle
from collections import Counter
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')          # non-interactive backend — never blocks terminal
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

# ── 2. RTX 4050 one-time GPU setup ───────────────────────────────────────────
torch.backends.cudnn.benchmark        = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🖥️  Device : {DEVICE}")
if DEVICE == 'cuda':
    print(f"   GPU    : {torch.cuda.get_device_name(0)}")
    print(f"   VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ── 3. Constants ──────────────────────────────────────────────────────────────
CLASSES = [
    'angry', 'contempt', 'disgust', 'fear',
    'happy', 'natural', 'sad', 'sleepy', 'surprised'
]
NUM_CLASSES = len(CLASSES)
IMG_SIZE    = 224


# ── 4. Dataset ────────────────────────────────────────────────────────────────
class FacialExpressionDataset(Dataset):
    def __init__(self, base, split, transform=None):
        self.img_dir   = f"{base}/{split}/images"
        self.label_dir = f"{base}/{split}/labels"
        self.transform = transform
        self.samples   = []

        for lf in sorted(os.listdir(self.label_dir)):
            stem       = os.path.splitext(lf)[0]
            label_path = f"{self.label_dir}/{lf}"

            img_path = None
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                candidate = f"{self.img_dir}/{stem}{ext}"
                if os.path.exists(candidate):
                    img_path = candidate
                    break
            if img_path is None:
                continue

            with open(label_path) as f:
                lines = f.readlines()
            if not lines:
                continue

            cls_id = int(lines[0].strip().split()[0])
            self.samples.append((img_path, cls_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, cls_id = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, cls_id


# ── 5. Transforms ─────────────────────────────────────────────────────────────
train_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# ── 6. Class weights & sampler ────────────────────────────────────────────────
def get_class_weights(dataset):
    counts  = Counter(cls_id for _, cls_id in dataset.samples)
    total   = sum(counts.values())
    weights = torch.tensor(
        [total / counts[i] for i in range(NUM_CLASSES)], dtype=torch.float
    )
    return weights


def get_sampler(dataset, class_weights):
    sample_weights = torch.tensor(
        [class_weights[cls_id].item() for _, cls_id in dataset.samples],
        dtype=torch.float
    )
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )


# ── 7. DataLoaders ────────────────────────────────────────────────────────────
def get_dataloaders(base, batch_size=64, num_workers=4):
    train_dataset = FacialExpressionDataset(base, "train", transform=train_transforms)
    val_dataset   = FacialExpressionDataset(base, "valid", transform=val_transforms)
    test_dataset  = FacialExpressionDataset(base, "test",  transform=val_transforms)

    class_weights = get_class_weights(train_dataset)
    sampler       = get_sampler(train_dataset, class_weights)

    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
        ),
    }
    return loaders, class_weights


# ── 8. Model ──────────────────────────────────────────────────────────────────
class FacialExpressionCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, dropout_rate=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(3,   32,  kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32,  64,  kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64,  128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4   = nn.BatchNorm2d(256)

        self.pool          = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

        self.fc1      = nn.Linear(256 * 7 * 7, 512)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2      = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3      = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout2(F.relu(self.fc2(x)))
        return self.fc3(x)


# ── 9. Train ──────────────────────────────────────────────────────────────────
def train_model(model, loaders, class_weights, config, device=DEVICE, checkpoint_freq=1):
    print(f"\n{'='*60}")
    print(f"  Config : {config['name']}")
    print(f"  Device : {device}")
    print(f"{'='*60}")

    model         = model.to(device)
    class_weights = class_weights.to(device)

    print("⚡ Compiling model (first epoch will be slower) …")
    model = torch.compile(model)

    criterion  = nn.CrossEntropyLoss(weight=class_weights)
    pt_version = tuple(int(x) for x in torch.__version__.split('.')[:2])
    use_fused  = (device == 'cuda')

    if config['optimizer_type'] == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            fused=use_fused,
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config['learning_rate'],
            momentum=0.9,
            weight_decay=config['weight_decay'],
            fused=use_fused and pt_version >= (2, 1),
        )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=config['scheduler_patience'], factor=0.5
    )

    use_amp = (device == 'cuda')
    scaler  = GradScaler('cuda', enabled=use_amp)
    if use_amp:
        print("🔥 Mixed-precision FP16 enabled")

    checkpoint_path = f'checkpoint_{config["name"]}.pth'
    history_path    = f'history_{config["name"]}.pkl'
    best_model_path = f'best_model_{config["name"]}.pth'

    start_epoch  = 0
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [],
               'val_loss':   [], 'val_acc':   [], 'epoch_time': []}

    if os.path.exists(checkpoint_path):
        print("🔄 Checkpoint found — resuming …")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        if 'scaler_state' in ckpt and use_amp:
            scaler.load_state_dict(ckpt['scaler_state'])
        start_epoch  = ckpt['epoch'] + 1
        best_val_acc = ckpt['best_val_acc']
        history      = ckpt['history']
        print(f"   Epoch {start_epoch}/{config['epochs']}  |  "
              f"Best val acc: {best_val_acc:.2f}%  |  "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        if start_epoch >= config['epochs']:
            print("   ✓ Already completed!")
            return history, best_val_acc
    else:
        print("📝 Fresh start")

    if os.path.exists(history_path) and not history['train_loss']:
        with open(history_path, 'rb') as f:
            saved = pickle.load(f)
        if len(saved['train_loss']) > len(history['train_loss']):
            history = saved

    epochs_left = config['epochs'] - start_epoch
    if history.get('epoch_time'):
        avg_s = np.mean(history['epoch_time'][-5:])
        print(f"⏱️  ~{avg_s/60:.1f} min/epoch  →  ETA {avg_s * epochs_left / 3600:.1f} h")
    print()

    for epoch in range(start_epoch, config['epochs']):
        t0 = time.time()

        # Train
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        bar = tqdm(loaders['train'],
                   desc=f'Epoch {epoch+1}/{config["epochs"]} [Train]',
                   dynamic_ncols=True)
        for inputs, labels in bar:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda', enabled=use_amp):
                outputs = model(inputs)
                loss    = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss    += loss.item()
            _, pred        = outputs.max(1)
            train_total   += labels.size(0)
            train_correct += (pred == labels).sum().item()
            bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = train_loss / len(loaders['train'])
        train_acc      = 100.0 * train_correct / train_total

        # Validate
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            bar = tqdm(loaders['val'],
                       desc=f'Epoch {epoch+1}/{config["epochs"]} [Val]',
                       dynamic_ncols=True)
            for inputs, labels in bar:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with autocast('cuda', enabled=use_amp):
                    outputs = model(inputs)
                    loss    = criterion(outputs, labels)
                val_loss    += loss.item()
                _, pred      = outputs.max(1)
                val_total   += labels.size(0)
                val_correct += (pred == labels).sum().item()

        avg_val_loss = val_loss / len(loaders['val'])
        val_acc      = 100.0 * val_correct / val_total

        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < old_lr:
            print(f"   📉 LR: {old_lr:.2e} → {new_lr:.2e}")

        epoch_time = time.time() - t0
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        history.setdefault('epoch_time', []).append(epoch_time)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"   ✨ New best! Val acc: {val_acc:.2f}%  → {best_model_path}")

        # Atomic checkpoint
        if (epoch + 1) % checkpoint_freq == 0 or epoch == config['epochs'] - 1:
            ckpt = {
                'epoch':           epoch,
                'model_state':     model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'scaler_state':    scaler.state_dict(),
                'best_val_acc':    best_val_acc,
                'history':         history,
                'config':          config,
            }
            tmp = checkpoint_path + '.tmp'
            torch.save(ckpt, tmp)
            os.replace(tmp, checkpoint_path)
            with open(history_path, 'wb') as f:
                pickle.dump(history, f)
            print(f"   💾 Checkpoint saved (epoch {epoch+1})")

        if device == 'cuda':
            used  = torch.cuda.memory_reserved(0) / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            vram  = f"  VRAM {used:.1f}/{total:.1f} GB"
        else:
            vram = ""
        eta = epoch_time * (config['epochs'] - epoch - 1)
        print(
            f'Epoch {epoch+1:>4}/{config["epochs"]}  '
            f'TrLoss {avg_train_loss:.4f}  TrAcc {train_acc:.2f}%  '
            f'VaLoss {avg_val_loss:.4f}  VaAcc {val_acc:.2f}%  '
            f'LR {new_lr:.2e}  {epoch_time/60:.1f}m  ETA {eta/3600:.1f}h{vram}'
        )

    print(f"\n🎉 Done — {config['name']}  |  Best val acc: {best_val_acc:.2f}%")
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    return history, best_val_acc


# ── 10. Evaluation ────────────────────────────────────────────────────────────
def evaluate_model(model, loaders, config_name, device=DEVICE):
    model.eval()
    model.to(device)
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in tqdm(loaders['test'], desc='Testing'):
            inputs  = inputs.to(device, non_blocking=True)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=CLASSES))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar()
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center')
    plt.xticks(range(len(CLASSES)), CLASSES, rotation=45)
    plt.yticks(range(len(CLASSES)), CLASSES)
    plt.tight_layout()
    # Save to file instead of showing — won't block overnight terminal
    plt.savefig(f'confusion_matrix_{config_name}.png', dpi=150)
    plt.close()
    print(f"   📊 Confusion matrix saved → confusion_matrix_{config_name}.png")

    test_acc = 100.0 * (np.array(all_preds) == np.array(all_labels)).sum() / len(all_labels)
    return test_acc


def plot_training_history(history, config_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'],   label='Val Loss')
    ax1.set(xlabel='Epoch', ylabel='Loss', title=f'{config_name} — Loss')
    ax1.legend(); ax1.grid(True)

    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'],   label='Val Acc')
    ax2.set(xlabel='Epoch', ylabel='Accuracy (%)', title=f'{config_name} — Accuracy')
    ax2.legend(); ax2.grid(True)

    plt.tight_layout()
    # Save to file instead of showing — won't block overnight terminal
    plt.savefig(f'training_history_{config_name}.png', dpi=150)
    plt.close()
    print(f"   📈 Training history saved → training_history_{config_name}.png")


# ── 11. Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    BASE       = "/home/jtshi/.cache/kagglehub/datasets/aklimarimi/8-facial-expressions-for-yolo/versions/4/9 Facial Expressions you need"
    BATCH_SIZE = 128   # reduce to 32 if VRAM > 5.5 GB in first epoch

    loaders, class_weights = get_dataloaders(BASE, batch_size=BATCH_SIZE, num_workers=4)
    print(f"\nDataset sizes:")
    print(f"  Train : {len(loaders['train'].dataset)}")
    print(f"  Val   : {len(loaders['val'].dataset)}")
    print(f"  Test  : {len(loaders['test'].dataset)}")

    # ── Single config — baseline Adam, 50 epochs ──────────────────────────────
    config = {
        'name':               'baseline_adam',
        'learning_rate':      0.001,
        'weight_decay':       0.0001,
        'epochs':             50,
        'dropout_rate':       0.5,
        'optimizer_type':     'adam',
        'scheduler_patience': 5,      # bumped from 3 — gives LR more room to breathe
    }

    results = {}

    try:
        model = FacialExpressionCNN(dropout_rate=config['dropout_rate'])
        print(f"\nParameters: {sum(p.numel() for p in model.parameters()):,}")

        history, best_val_acc = train_model(model, loaders, class_weights, config)
        test_acc = evaluate_model(model, loaders, config['name'])
        plot_training_history(history, config['name'])

        results[config['name']] = {
            'history':      history,
            'best_val_acc': best_val_acc,
            'test_acc':     test_acc,
            'config':       config,
        }
        print(f"\n✅ Done  |  Val: {best_val_acc:.2f}%  |  Test: {test_acc:.2f}%")

    except Exception as e:
        import traceback; traceback.print_exc()

    # ── Save results ──────────────────────────────────────────────────────────
    if results:
        summary = {
            'timestamp':          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'best_test_accuracy': float(results[config['name']]['test_acc']),
            'best_val_accuracy':  float(results[config['name']]['best_val_acc']),
            'dataset_info': {
                'train_size':  len(loaders['train'].dataset),
                'val_size':    len(loaders['val'].dataset),
                'test_size':   len(loaders['test'].dataset),
                'num_classes': NUM_CLASSES,
                'classes':     CLASSES,
            },
            'config': {k: v for k, v in config.items()},
        }
        with open('results.json', 'w') as f:
            json.dump(summary, f, indent=2)
        print("\n📊 Results saved → results.json")