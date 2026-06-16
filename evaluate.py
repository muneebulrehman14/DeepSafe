import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, confusion_matrix,
                             classification_report)
from efficientnet_pytorch import EfficientNet

DATA_DIR = "data_140k"
RESULTS_DIR = "results"
BEST_MODEL_PATH = os.path.join("checkpoints", "best_model.pth")
THRESHOLD_PATH = os.path.join(RESULTS_DIR, "threshold.json")

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def create_model():
    model = EfficientNet.from_pretrained("efficientnet-b0")
    num_features = model._fc.in_features
    model._fc = nn.Linear(num_features, 1)
    return model


def main():
    device = get_device()
    print(f"Device: {device}")

    if not os.path.isfile(BEST_MODEL_PATH):
        print(f"ERROR: no model found at {BEST_MODEL_PATH}")
        return

    threshold = 0.90
    if os.path.isfile(THRESHOLD_PATH):
        with open(THRESHOLD_PATH) as f:
            threshold = json.load(f).get("threshold", 0.90)
    print(f"Using threshold: {threshold:.3f}")

    test_dir = os.path.join(DATA_DIR, "test")
    test_dataset = datasets.ImageFolder(test_dir, transform=EVAL_TRANSFORM)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    print(f"Test samples: {len(test_dataset)} ({test_dataset.classes})")

    model = create_model().to(device)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    model.eval()

    criterion = nn.BCEWithLogitsLoss()

    all_probs = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels_dev = labels.to(device).float().view(-1, 1)
            labels_dev = 1 - labels_dev
            outputs = model(images)
            loss = criterion(outputs, labels_dev)
            total_loss += loss.item() * images.size(0)
            probs = torch.sigmoid(outputs)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels_dev.cpu().numpy())

    all_probs = np.array(all_probs).flatten()
    all_labels = np.array(all_labels).flatten()
    avg_loss = total_loss / len(test_dataset)

    all_preds = (all_probs >= threshold).astype(int)
    preds_05 = (all_probs >= 0.5).astype(int)

    acc = accuracy_score(all_labels, all_preds)
    acc_05 = accuracy_score(all_labels, preds_05)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\nTest Results (threshold={threshold:.3f}):")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  AUC-ROC:   {auc:.4f}")
    print(f"  Loss:      {avg_loss:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"              Predicted")
    print(f"              real  fake")
    print(f"  Actual real {cm[0,0]:5d} {cm[0,1]:5d}")
    print(f"         fake {cm[1,0]:5d} {cm[1,1]:5d}")
    print(f"\n  Real misclassified as fake: {cm[0,1]} / {cm[0,0]+cm[0,1]} ({cm[0,1]/(cm[0,0]+cm[0,1])*100:.1f}%)")
    print(f"  Fake misclassified as real: {cm[1,0]} / {cm[1,0]+cm[1,1]} ({cm[1,0]/(cm[1,0]+cm[1,1])*100:.1f}%)")
    print(f"\nAccuracy @ threshold=0.50: {acc_05:.4f}")
    print(f"Accuracy @ threshold={threshold:.2f}: {acc:.4f}")

    report = classification_report(all_labels, all_preds, target_names=["Real", "Fake"])
    print(f"\n{report}")

    results = {
        "threshold": threshold,
        "accuracy": acc,
        "accuracy_at_05": acc_05,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc_roc": auc,
        "loss": avg_loss,
        "confusion_matrix": cm.tolist(),
        "real_misclassified_as_fake": int(cm[0,1]),
        "fake_misclassified_as_real": int(cm[1,0]),
    }
    with open(os.path.join(RESULTS_DIR, "test_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {RESULTS_DIR}/test_results.json")


if __name__ == "__main__":
    main()
