import os
import json
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from efficientnet_pytorch import EfficientNet

import gradio as gr

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


class DeepfakeDetector:
    def __init__(self):
        self.device = get_device()
        self.threshold = 0.90
        if os.path.isfile(THRESHOLD_PATH):
            with open(THRESHOLD_PATH) as f:
                self.threshold = json.load(f).get("threshold", 0.90)
        self.model = EfficientNet.from_pretrained("efficientnet-b0")
        num_features = self.model._fc.in_features
        self.model._fc = nn.Linear(num_features, 1)
        if os.path.isfile(BEST_MODEL_PATH):
            self.model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.transform = EVAL_TRANSFORM
        self._activations = None
        self._gradients = None

    def _forward_hook(self, module, input, output):
        self._activations = output
        output.register_hook(self._save_gradients)

    def _save_gradients(self, grad):
        self._gradients = grad

    def gradcam(self, img):
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img).convert("RGB")

        input_tensor = self.transform(img).unsqueeze(0).to(self.device)
        input_tensor.requires_grad = True

        self._activations = None
        self._gradients = None
        handle = self.model._conv_head.register_forward_hook(self._forward_hook)

        output = self.model(input_tensor)
        prob = torch.sigmoid(output).item()

        self.model.zero_grad()
        output.backward()
        handle.remove()

        if self._activations is not None and self._gradients is not None:
            activations = self._activations
            gradients = self._gradients
            weights = gradients.mean(dim=[2, 3], keepdim=True)
            cam = (weights * activations).sum(dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = cam.squeeze().cpu().data.numpy()
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            heatmap = np.uint8(255 * cam)
        else:
            heatmap = np.zeros((224, 224), dtype=np.uint8)

        orig_w, orig_h = img.size
        heatmap_resized = np.array(
            Image.fromarray(heatmap).resize((orig_w, orig_h), Image.Resampling.LANCZOS)
        )

        overlay = np.array(img).astype(np.float32)
        heatmap_colored = np.stack([heatmap_resized] * 3, axis=-1).astype(np.float32)
        overlay = overlay * 0.5 + heatmap_colored * 0.5
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        is_fake = prob >= self.threshold
        label = "FAKE" if is_fake else "REAL"
        return overlay, label, prob

    def predict_proba(self, img):
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img).convert("RGB")
        input_tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor)
            return torch.sigmoid(output).item()


def build_ui():
    detector = DeepfakeDetector()

    with gr.Blocks(theme=gr.themes.Soft(), title="DeepSafe — AI Face Detector") as demo:
        gr.Markdown("""
        # DeepSafe — AI Face Detector
        Upload a face image to detect if it's REAL or AI-generated FAKE.
        """)

        with gr.Tab("Single Image"):
            with gr.Row():
                with gr.Column():
                    img_input = gr.Image(type="numpy", label="Upload Face Image", height=400)
                    submit_btn = gr.Button("Analyze", variant="primary")

                with gr.Column():
                    result_output = gr.Label(label="Result")
                    prob_output = gr.Number(label="Fake Probability (%)")

            with gr.Accordion("Grad-CAM Heatmap", open=True):
                with gr.Row():
                    original_display = gr.Image(type="numpy", label="Original", height=300)
                    heatmap_output = gr.Image(type="numpy", label="Grad-CAM Overlay", height=300)

            def analyze(img):
                if img is None:
                    return None, None, None
                overlay, label, prob = detector.gradcam(img)
                pct = round(prob * 100, 2)
                return img, overlay, {label: pct}, pct

            submit_btn.click(
                fn=analyze,
                inputs=img_input,
                outputs=[original_display, heatmap_output, result_output, prob_output],
            )

        with gr.Tab("Batch Analysis"):
            batch_input = gr.File(file_count="multiple", label="Upload multiple images", file_types=["image"])
            batch_btn = gr.Button("Analyze All", variant="primary")
            batch_output = gr.Dataframe(
                headers=["Filename", "Label", "Fake Probability", "Confidence"],
                label="Results",
                interactive=False,
            )

            def analyze_batch(files):
                if not files:
                    return []
                results = []
                for f in files:
                    path = f.name if hasattr(f, 'name') else f
                    prob = detector.predict_proba(path)
                    pct = prob * 100
                    is_fake = prob >= detector.threshold
                    label = "FAKE" if is_fake else "REAL"
                    confidence = pct if is_fake else 100 - pct
                    results.append([os.path.basename(path), label, f"{pct:.2f}%", f"{confidence:.1f}%"])
                return results

            batch_btn.click(fn=analyze_batch, inputs=batch_input, outputs=batch_output)

        with gr.Tab("About"):
            gr.Markdown(f"""
            ## DeepSafe

            **Model**: EfficientNet-B0 trained on 140K face images (70K real FFHQ + 70K fake StyleGAN2)

            **Performance**: 92.6% accuracy | 97.4% precision | 98.7% AUC-ROC

            **Threshold**: {detector.threshold*100:.0f}% — images above this are classified as FAKE
            """)

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(share=True)
