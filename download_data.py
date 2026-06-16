from datasets import load_dataset
import os
import random

dataset_name = "prithivMLmods/Deepfake-vs-Real"

print(f"Downloading {dataset_name}...")
ds = load_dataset(dataset_name, split="train")

data = list(ds)
print(f"Total samples: {len(data)}")
random.shuffle(data)

n = len(data)
n_train = int(n * 0.8)
n_val = int(n * 0.1)
splits = {
    "train": data[:n_train],
    "valid": data[n_train:n_train+n_val],
    "test": data[n_train+n_val:],
}

for split_name, items in splits.items():
    for i, item in enumerate(items):
        label = "real" if item["label"] == 1 else "fake"
        folder = f"data_extra/{split_name}/{label}"
        os.makedirs(folder, exist_ok=True)
        item["image"].save(f"{folder}/{i}.jpg")
        if i % 100 == 0:
            print(f"  {split_name} - saved {i}/{len(items)} images...")
    print(f"  {split_name} done: {len(items)} images")

print("Done!")
