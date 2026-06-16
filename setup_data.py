"""setup_data.py"""
import os
import shutil
from pathlib import Path


def setup_data(source_dir, output_dir="data_140k"):
    source = Path(source_dir)
    output = Path(output_dir)

    train_src = source / "real_vs_fake" / "real-vs-fake" / "train"
    val_src = source / "real_vs_fake" / "real-vs-fake" / "valid"
    test_src = source / "real_vs_fake" / "real-vs-fake" / "test"

    for split in [train_src, val_src, test_src]:
        if not split.exists():
            print(f"Source not found: {split}")
            return

    for split in ["train", "valid", "test"]:
        for cls in ["real", "fake"]:
            (output / split / cls).mkdir(parents=True, exist_ok=True)

    copy_files(train_src / "real", output / "train" / "real")
    copy_files(train_src / "fake", output / "train" / "fake")
    copy_files(val_src / "real", output / "valid" / "real")
    copy_files(val_src / "fake", output / "valid" / "fake")
    copy_files(test_src / "real", output / "test" / "real")
    copy_files(test_src / "fake", output / "test" / "fake")

    for split in ["train", "valid", "test"]:
        for cls in ["real", "fake"]:
            count = len(list((output / split / cls).iterdir()))
            print(f"  {split}/{cls}: {count} images")


def copy_files(src, dst):
    files = list(src.glob("*.jpg"))
    for i, f in enumerate(files):
        dst_file = dst / f.name
        if not dst_file.exists():
            for attempt in range(3):
                try:
                    shutil.copy2(f, dst_file)
                    break
                except PermissionError:
                    import time
                    time.sleep(0.1)
        if (i + 1) % 1000 == 0:
            print(f"    copied {i+1}/{len(files)}")


if __name__ == "__main__":
    import sys
    source = sys.argv[1] if len(sys.argv) > 1 else r"E:\SOFT2\archive"
    setup_data(source)
