import os, shutil, random
from pathlib import Path

random.seed(42)

source = Path(r"E:\SOFT2\archive") / "real_vs_fake" / "real-vs-fake"
extra = Path("data_extra")
dest = Path("data_140k")

n_per_split = 5000  # 5K real + 5K fake per split

for split in ["train", "valid", "test"]:
    for cls in ["real", "fake"]:
        dest_dir = dest / split / cls
        dest_dir.mkdir(parents=True, exist_ok=True)

        src_dir = source / split / cls
        all_files = list(src_dir.glob("*.jpg"))
        random.shuffle(all_files)
        selected = all_files[:n_per_split]
        for f in selected:
            shutil.copy2(f, dest_dir / f.name)
        print(f"140K {split}/{cls}: {len(selected)}")

        extra_dir = extra / split / cls
        if extra_dir.exists():
            for f in extra_dir.iterdir():
                if f.is_file():
                    shutil.copy2(f, dest_dir / f"extra_{f.name}")
            print(f"  + extra {split}/{cls}: {len(list(extra_dir.iterdir()))}")

for s in ["train","valid","test"]:
    for c in ["real","fake"]:
        p = dest / s / c
        print(f"Total {s}/{c}: {len(list(p.iterdir()))}")
