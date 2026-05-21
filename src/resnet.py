# -*- coding: utf-8 -*-
"""
Facial Expression Recognition — ResNet50 Transfer Learning
===========================================================
RTX 4050 + Ryzen 7 optimised.

Transfer learning strategy:
  Phase 1 (epochs 1–5)   : Freeze ResNet backbone, train classifier head only.
                            Fast convergence, low risk of destroying pretrained weights.
  Phase 2 (epochs 6–end) : Unfreeze all layers, fine-tune end-to-end at low LR.
                            Lets the backbone adapt to facial expression features.
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
matplotlib.use('Agg')
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
from torchvision import transforms, models

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
            # Skip any labels outside valid range
            if cls_id >= NUM_CLASSES:
                continue
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
# ResNet50 was trained on ImageNet with these exact mean/std values
train_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
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
    return torch.tensor(
        [total / counts[i] for i in range(NUM_CLASSES)], dtype=torch.float
    )


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

    print(f"  Labels in range: train={len(train_dataset)}, "
          f"val={len(val_dataset)}, test={len(test_dataset)}")

    class_weights = get_class_weights(train_dataset)
    sampler       = get_sampler(train_dataset, class_weights)

    loaders = {
        "train": DataLoader(train_dataset, batch_size=batch_size, sampler=sampler,
                            num_workers=num_workers, pin_memory=True,
                            persistent_workers=True),
        "val":   DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True,
                            persistent_workers=True),
        "test":  DataLoader(test_dataset,  batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True,
                            persistent_workers=True),
    }
    return loaders, class_weights


# ── 8. ResNet50 model ─────────────────────────────────────────────────────────
def build_resnet50(num_classes=NUM_CLASSES, dropout_rate=0.5):
    """
    ResNet50 pretrained on ImageNet.
    Replace the final FC layer with a new head for our 9 expression classes.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # Freeze all backbone layers initially (Phase 1)
    for param in model.parameters():
        param.requires_grad = False

    # Replace final layer — this is the only trainable part in Phase 1
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(dropout_rate / 2),
        nn.Linear(512, num_classes)
    )

    return model


def unfreeze_backbone(model, unfreeze_from_layer='layer3'):
    """
    Unfreeze ResNet layers from a given layer onwards for fine-tuning (Phase 2).
    Unfreezing from layer3 gives a good balance of adaptation vs stability.
    """
    unfreeze = False
    for name, param in model.named_parameters():
        if unfreeze_from_layer in name:
            unfreeze = True
        if unfreeze:
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   🔓 Backbone unfrozen from {unfreeze_from_layer}  "
          f"({trainable:,} trainable parameters)")


# ── 9. Train ──────────────────────────────────────────────────────────────────
def train_model(model, loaders, class_weights, config, device=DEVICE, checkpoint_freq=1):
    """
    Two-phase transfer learning:
      Phase 1: Train head only (fast, safe)
      Phase 2: Fine-tune full network at lower LR
    """
    print(f"\n{'='*60}")
    print(f"  Config : {config['name']}")
    print(f"  Device : {device}")
    print(f"  Epochs : {config['epochs']}  "
          f"(Phase 1: 1–{config['warmup_epochs']}, "
          f"Phase 2: {config['warmup_epochs']+1}–{config['epochs']})")
    print(f"{'='*60}")

    model         = model.to(device)
    class_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Phase 1 optimizer — only trains the new head
    def make_optimizer(lr):
        pt_version = tuple(int(x) for x in torch.__version__.split('.')[:2])
        use_fused  = (device == 'cuda')
        return torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=config['weight_decay'],
            fused=use_fused,
        )

    optimizer = make_optimizer(config['learning_rate'])
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

    start_epoch    = 0
    best_val_acc   = 0.0
    phase2_started = False
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
        start_epoch    = ckpt['epoch'] + 1
        best_val_acc   = ckpt['best_val_acc']
        history        = ckpt['history']
        phase2_started = ckpt.get('phase2_started', False)
        print(f"   Epoch {start_epoch}/{config['epochs']}  |  "
              f"Best val acc: {best_val_acc:.2f}%  |  "
              f"Phase 2 started: {phase2_started}")
        if start_epoch >= config['epochs']:
            print("   ✓ Already completed!")
            return history, best_val_acc
    else:
        print("📝 Fresh start — Phase 1: training head only")

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

        # ── Phase 2: unfreeze backbone after warmup epochs ────────────────────
        if epoch == config['warmup_epochs'] and not phase2_started:
            print(f"\n🔓 Phase 2 — unfreezing backbone for fine-tuning …")
            unfreeze_backbone(model, unfreeze_from_layer='layer3')
            # Rebuild optimizer with lower LR for fine-tuning
            optimizer = make_optimizer(config['finetune_lr'])
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=config['scheduler_patience'], factor=0.5
            )
            phase2_started = True
            print(f"   Fine-tune LR: {config['finetune_lr']:.2e}\n")

        phase = "P2-finetune" if phase2_started else "P1-headonly"

        # Train
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        bar = tqdm(loaders['train'],
                   desc=f'Epoch {epoch+1}/{config["epochs"]} [{phase}]',
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
                'phase2_started':  phase2_started,
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
            f'Epoch {epoch+1:>4}/{config["epochs"]} [{phase}]  '
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
    plt.savefig(f'training_history_{config_name}.png', dpi=150)
    plt.close()
    print(f"   📈 Training history saved → training_history_{config_name}.png")


# ── 11. Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    BASE       = "/home/jtshi/.cache/kagglehub/datasets/aklimarimi/8-facial-expressions-for-yolo/versions/4/9 Facial Expressions you need"
    BATCH_SIZE = 64

    loaders, class_weights = get_dataloaders(BASE, batch_size=BATCH_SIZE, num_workers=4)
    print(f"\nDataset sizes:")
    print(f"  Train : {len(loaders['train'].dataset)}")
    print(f"  Val   : {len(loaders['val'].dataset)}")
    print(f"  Test  : {len(loaders['test'].dataset)}")

    config = {
        'name':               'resnet50_transfer',
        'learning_rate':      0.001,    # Phase 1 LR — head only
        'finetune_lr':        0.0001,   # Phase 2 LR — full network (10x lower)
        'weight_decay':       0.0001,
        'epochs':             50,
        'warmup_epochs':      5,        # epochs before unfreezing backbone
        'dropout_rate':       0.5,
        'optimizer_type':     'adam',
        'scheduler_patience': 5,
    }

    try:
        model = build_resnet50(dropout_rate=config['dropout_rate'])
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in model.parameters())
        print(f"\nResNet50: {total:,} total params, {trainable:,} trainable (Phase 1)")

        history, best_val_acc = train_model(model, loaders, class_weights, config)
        test_acc = evaluate_model(model, loaders, config['name'])
        plot_training_history(history, config['name'])

        print(f"\n✅ Done  |  Val: {best_val_acc:.2f}%  |  Test: {test_acc:.2f}%")

        summary = {
            'timestamp':          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'best_test_accuracy': float(test_acc),
            'best_val_accuracy':  float(best_val_acc),
            'config':             config,
            'dataset_info': {
                'train_size':  len(loaders['train'].dataset),
                'val_size':    len(loaders['val'].dataset),
                'test_size':   len(loaders['test'].dataset),
                'num_classes': NUM_CLASSES,
                'classes':     CLASSES,
            },
        }
        with open('results_resnet50.json', 'w') as f:
            json.dump(summary, f, indent=2)
        print("📊 Results saved → results_resnet50.json")

    except Exception as e:
        import traceback; traceback.print_exc()