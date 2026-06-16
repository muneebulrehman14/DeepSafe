# DeepSafe — AI Face Detector

Detects AI-generated fake faces using EfficientNet-B0. Trained on a combined dataset of **140K face images** (FFHQ real + StyleGAN2 fake) + diverse generator samples for better generalization.

## Performance

| Metric | Value |
|--------|-------|
| Accuracy | 88.0% |
| Precision | 94.0% |
| AUC-ROC | 96.5% |
| Fake Detection Rate | 81.6% |
| Real→Fake False Positives | 5.3% |

## Dataset

- **140K Real and Fake Faces** ([Kaggle: `xhlulu/140k-real-and-fake-faces`](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces))
  - 70,000 real FFHQ faces + 70,000 StyleGAN2 fake faces
  - Pre-split: 50K/10K/10K train/valid/test per class
- **Deepfake-vs-Real** ([Hugging Face: `prithivMLmods/Deepfake-vs-Real`](https://huggingface.co/datasets/prithivMLmods/Deepfake-vs-Real))
  - Diverse AI-generated fake images from multiple GAN/diffusion models
  - 1,200+ fake images from various generators for improved generalization
- **Real human reference photos** for threshold calibration (real-world test images)

## Files to Upload to GitHub

```
app.py              # Gradio web interface with Grad-CAM
train.py            # Training script with balanced subset support
evaluate.py         # Test evaluation with metrics
predict.py          # Single/batch image prediction
requirements.txt    # Python dependencies
README.md           # This file
```

Do **NOT** upload:
- `checkpoints/` — model weights (16MB, regenerate via train.py)
- `data_140k/` — dataset (download separately)
- `data_extra/`, `data_extra2/`, `data/` — extracted temp files
- `results/` — evaluation outputs
- `__pycache__/`, `.gradio/` — caches

## Setup

```bash
pip install -r requirements.txt
```

Download the dataset and extract to `data_140k/`:
```
data_140k/
├── train/{real,fake}/
├── valid/{real,fake}/
└── test/{real,fake}/
```

## Training

```bash
python train.py --epochs 10 --subset 10000
```

Use `--subset N` for N total images (N/2 per class). All extra files are prioritized to ensure diverse generator coverage.

## Evaluation

```bash
python evaluate.py
```

## Prediction

```bash
python predict.py path/to/image.jpg
python predict.py img1.jpg img2.jpg img3.jpg
```

## Interactive App

```bash
python app.py
```

Opens a Gradio web UI with:
- Image upload + Grad-CAM heatmap overlay
- Batch analysis mode
- Conservative threshold (default 0.99) to minimize false positives on real photos

## Threshold

Default threshold: **0.99** — only images with ≥99% fake probability are classified as FAKE. This prioritizes minimizing false positives on real human photos.

To adjust:
```bash
# Edit results/threshold.json
{"threshold": 0.95}
```
