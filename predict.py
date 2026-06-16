import os
import json
import sys
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
from efficientnet_pytorch import EfficientNet

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


class DeepfakeDetector:
    def __init__(self, model_path=BEST_MODEL_PATH, threshold=0.80):
        self.device = get_device()
        self.threshold = threshold

        if os.path.isfile(THRESHOLD_PATH):
            with open(THRESHOLD_PATH) as f:
                self.threshold = json.load(f).get("threshold", threshold)

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        self.model = create_model().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.transform = EVAL_TRANSFORM

    def predict(self, image_path):
        img = Image.open(image_path).convert("RGB")
        input_tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            prob = torch.sigmoid(output).item()

        is_fake = prob >= self.threshold
        label = "FAKE" if is_fake else "REAL"
        return label, prob

    def predict_batch(self, image_paths):
        results = []
        for path in image_paths:
            label, prob = self.predict(path)
            results.append({"path": path, "label": label, "probability": round(prob, 4)})
        return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path> [image_path2 ...]")
        return

    detector = DeepfakeDetector()
    paths = sys.argv[1:]
    results = detector.predict_batch(paths)
    for r in results:
        print(f"{r['path']}: {r['label']} (prob={r['probability']:.4f})")


if __name__ == "__main__":
    main()
