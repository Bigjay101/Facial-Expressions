import os
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import kagglehub


# ── Constants ─────────────────────────────────────────────────────────────────

CLASSES = [
    'angry', 'contempt', 'disgust', 'fear',
    'happy', 'natural', 'sad', 'sleepy', 'surprised'
]
NUM_CLASSES = len(CLASSES)
IMG_SIZE    = 224


# ── Dataset ───────────────────────────────────────────────────────────────────

class FacialExpressionDataset(Dataset):
    def __init__(self, base, split, transform=None):
        self.img_dir   = f"{base}/{split}/images"
        self.label_dir = f"{base}/{split}/labels"
        self.transform = transform
        self.samples   = []  # list of (img_path, class_id)

        for lf in sorted(os.listdir(self.label_dir)):
            stem       = os.path.splitext(lf)[0]
            label_path = f"{self.label_dir}/{lf}"

            # find matching image file
            img_path = None
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                candidate = f"{self.img_dir}/{stem}{ext}"
                if os.path.exists(candidate):
                    img_path = candidate
                    break

            if img_path is None:
                continue

            # read class id from first annotation line
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


# ── Transforms ────────────────────────────────────────────────────────────────

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


# ── Class weights ─────────────────────────────────────────────────────────────

def get_class_weights(dataset):
    """Compute inverse frequency weights for each class."""
    counts = Counter(cls_id for _, cls_id in dataset.samples)
    total  = sum(counts.values())
    weights = torch.tensor(
        [total / counts[i] for i in range(NUM_CLASSES)],
        dtype=torch.float
    )
    return weights


# ── Sampler ───────────────────────────────────────────────────────────────────

def get_sampler(dataset, class_weights):
    """Build a WeightedRandomSampler from per-image class weights."""
    sample_weights = torch.tensor(
        [class_weights[cls_id].item() for _, cls_id in dataset.samples],
        dtype=torch.float
    )
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )


# ── DataLoaders ───────────────────────────────────────────────────────────────

def get_dataloaders(base, batch_size=32, num_workers=4):
    """
    Returns train, val, test DataLoaders and class weights for the loss function.

    Args:
        base        : path to dataset root (contains train/, valid/, test/)
        batch_size  : number of samples per batch
        num_workers : parallel workers for data loading

    Returns:
        loaders      : dict with keys 'train', 'val', 'test'
        class_weights: tensor of inverse frequency weights for CrossEntropyLoss
    """
    # datasets
    train_dataset = FacialExpressionDataset(base, "train", transform=train_transforms)
    val_dataset   = FacialExpressionDataset(base, "valid", transform=val_transforms)
    test_dataset  = FacialExpressionDataset(base, "test",  transform=val_transforms)

    # class weights + sampler (train only)
    class_weights = get_class_weights(train_dataset)
    sampler       = get_sampler(train_dataset, class_weights)

    # dataloaders
    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,        # replaces shuffle=True
            num_workers=num_workers,
            pin_memory=True         # faster GPU transfer
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    }

    return loaders, class_weights


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    path = kagglehub.dataset_download("aklimarimi/8-facial-expressions-for-yolo")
    base = f"{path}/9 Facial Expressions you need"

    loaders, class_weights = get_dataloaders(base, batch_size=32)

    print("Class weights:")
    for cls, w in zip(CLASSES, class_weights):
        print(f"  {cls:12s}: {w:.4f}")

    # sanity check — one batch
    imgs, labels = next(iter(loaders["train"]))
    print(f"\nBatch shape : {imgs.shape}")       # [32, 3, 224, 224]
    print(f"Labels      : {labels}")
    print(f"Pixel range : {imgs.min():.2f} to {imgs.max():.2f}")