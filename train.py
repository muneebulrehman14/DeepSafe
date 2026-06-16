"""
train.py
--------
Trains EfficientNet-B0 (efficientnet-pytorch) for deepfake detection.
Uses BCEWithLogitsLoss for binary classification (1 output neuron).

Key features:
- Real GAN-generated fake faces (StyleGAN), not synthetic noise
- Calibration step tunes fake threshold on validation set
- Default fake threshold = 0.90

Usage:
  python setup_data.py --source E:\SOFT2\archive
  python train.py --epochs 10
"""

import os
import time
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from efficientnet_pytorch import EfficientNet
from tqdm import tqdm

Path("checkpoints").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def get_transforms(split):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if split == "train":
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


def create_model():
    model = EfficientNet.from_pretrained("efficientnet-b0")
    num_features = model._fc.in_features
    model._fc = nn.Linear(num_features, 1)
    return model


def build_loaders(data_dir, batch_size, num_workers, subset=None):
    from torch.utils.data import Subset

    def balanced_subset(ds, n):
        if n is None or len(ds) <= n:
            return ds
        rng = np.random.default_rng(SEED)
        n_per_class = n // 2
        fake_extra = [i for i, (p, t) in enumerate(ds.samples) if t == 0 and os.path.basename(p).startswith("extra_")]
        fake_reg = [i for i, (p, t) in enumerate(ds.samples) if t == 0 and i not in fake_extra]
        real_idx = [i for i, (p, t) in enumerate(ds.samples) if t == 1]
        n_fake = min(n_per_class, len(fake_extra) + len(fake_reg))
        n_real = min(n_per_class, len(real_idx))
        chosen = fake_extra[:]
        if len(chosen) < n_fake:
            chosen += rng.choice(fake_reg, n_fake - len(chosen), replace=False).tolist()
        chosen += rng.choice(real_idx, n_real, replace=False).tolist()
        return Subset(ds, chosen)

    classes = ["fake", "real"]
    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=get_transforms("train"))
    val_ds = datasets.ImageFolder(os.path.join(data_dir, "valid"), transform=get_transforms("val"))
    test_ds = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=get_transforms("test"))

    if subset:
        train_ds = balanced_subset(train_ds, subset)
        val_ds = balanced_subset(val_ds, subset)
        test_ds = balanced_subset(test_ds, subset)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())

    print(f"  Train: {len(train_ds)} images | Classes: {classes}")
    print(f"  Val:   {len(val_ds)} images | Classes: {classes}")
    print(f"  Test:  {len(test_ds)} images | Classes: {classes}")
    return train_loader, val_loader, test_loader


def run_epoch(model, loader, criterion, optimizer, device, phase):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"  {phase:5s}", leave=False, unit="batch")

    with torch.set_grad_enabled(is_train):
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device).float().view(-1, 1)
            labels = 1 - labels  # flip: ImageFolder 0=fake -> 1, real -> 0 for sigmoid

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).int()
            correct += (preds == labels.int()).sum().item()
            total += labels.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100*correct/total:.1f}%")

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def calibrate_threshold(model, val_loader, device, target_fpr=0.05):
    """Find threshold keeping false positive rate (real->fake) below target_fpr."""
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="  Calibrating", unit="batch"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    thresholds = np.linspace(0.5, 0.99, 50)
    best_threshold = 0.90
    best_fpr = 1.0

    for thresh in thresholds:
        preds = (all_probs >= thresh).astype(int)
        real_mask = all_labels == 1
        fpr = preds[real_mask].mean()
        if fpr <= target_fpr and fpr < best_fpr:
            best_fpr = fpr
            best_threshold = thresh

    print(f"  Calibrated threshold: {best_threshold:.3f} (FPR={best_fpr*100:.2f}%)")
    return best_threshold


def test_5_real_photos(model, val_loader, device, threshold):
    """Test on 5 random real photos to confirm they classify as REAL."""
    model.eval()
    real_images = []

    with torch.no_grad():
        for images, labels in val_loader:
            for i in range(len(labels)):
                if labels[i].item() == 1:
                    real_images.append(images[i].unsqueeze(0))
            if len(real_images) >= 5:
                break

    real_images = real_images[:5]
    print(f"\n  Testing {len(real_images)} real photos (threshold={threshold:.3f}):")
    all_real = True
    for i, img in enumerate(real_images):
        img = img.to(device)
        with torch.no_grad():
            output = model(img)
            prob = torch.sigmoid(output).item()
        label = "FAKE" if prob >= threshold else "REAL"
        status = "PASS" if label == "REAL" else "FAIL"
        if label != "REAL":
            all_real = False
        print(f"    Photo {i+1}: {label} (fake_prob={prob:.4f}) [{status}]")

    if all_real:
        print("  All 5 real photos correctly classified as REAL!")
    else:
        print("  WARNING: Some real photos misclassified as FAKE")
    return all_real


def plot_curves(history):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training Curves - Deepfake Detector", fontsize=14, fontweight="bold")
    ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss", markersize=4)
    ax1.plot(epochs, history["val_loss"], "r-o", label="Val Loss", markersize=4)
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.plot(epochs, history["train_acc"], "b-o", label="Train Acc", markersize=4)
    ax2.plot(epochs, history["val_acc"], "r-o", label="Val Acc", markersize=4)
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_ylim([0, 105])
    plt.tight_layout()
    plt.savefig("results/training_curves.png", dpi=150)
    plt.close()
    print(f"  Training curves saved -> results/training_curves.png")


def train(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"\nDevice: {device}")

    print("\n[Data]")
    train_loader, val_loader, test_loader = build_loaders(args.data_dir, args.batch_size, args.num_workers, args.subset)

    print("\n[Model]")
    model = create_model().to(device)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total:,}  Trainable: {trainable:,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    print(f"\n[Training] epochs={args.epochs}  batch={args.batch_size}  lr={args.lr}")
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    threshold = 0.90

    if args.resume:
        ckpt = "checkpoints/best_model.pth"
        if os.path.isfile(ckpt):
            model.load_state_dict(torch.load(ckpt, map_location=device))
            print(f"  Resumed from {ckpt}")
        else:
            print(f"  No checkpoint found at {ckpt}, starting fresh")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        print(f"\nEpoch {epoch}/{args.epochs}  (lr={scheduler.get_last_lr()[0]:.2e})")

        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, "Train")
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device, "Val")
        scheduler.step()

        elapsed = time.time() - t0
        print(f"  Train  loss={train_loss:.4f}  acc={train_acc*100:.2f}%")
        print(f"  Val    loss={val_loss:.4f}  acc={val_acc*100:.2f}%  [{elapsed:.1f}s]")

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc * 100)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc * 100)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "checkpoints/best_model.pth")
            print(f"  -> Best saved (val={val_acc*100:.2f}%)")

    print("\n[Final calibration & test]")
    if args.calibrate:
        threshold = calibrate_threshold(model, val_loader, device)
    with open("results/threshold.json", "w") as f:
        json.dump({"threshold": threshold}, f, indent=2)
    print(f"  Threshold: {threshold:.3f}")

    test_loss, test_acc = run_epoch(model, test_loader, criterion, None, device, "Test")
    print(f"  Test loss={test_loss:.4f}  acc={test_acc*100:.2f}%")

    test_5_real_photos(model, val_loader, device, threshold)
    plot_curves(history)

    print(f"\n[Done] Best val: {best_val_acc*100:.2f}%  Test: {test_acc*100:.2f}%  Threshold: {threshold:.3f}")


def parse_args():
    p = argparse.ArgumentParser(description="Train deepfake detector")
    p.add_argument("--data_dir", default="data_140k")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--subset", type=int, default=None, help="Total images per split (e.g. 10000 = 5K real + 5K fake per split)")
    p.add_argument("--resume", action="store_true", help="Resume from best_model.pth checkpoint")
    p.add_argument("--calibrate", action="store_true", default=True)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("=" * 60)
    print("  Deepfake Image Analyzer - Training (140K Dataset)")
    print("=" * 60)
    train(args)
    print("\n  ** Training complete! Run evaluate.py -> app.py next.")
