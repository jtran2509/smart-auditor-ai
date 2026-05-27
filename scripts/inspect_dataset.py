import json
from pathlib import Path
import random

def inspect_dataset():
    train_img_dir = Path(r"D:\smart-auditor-ai\data\processed\sroie_clean\train\images")
    train_ann_dir = Path(r"D:\smart-auditor-ai\data\processed\sroie_clean\train\annotations")

    # Get a random sample
    sample_images = list(train_img_dir.glob("*.jpg"))[:5]

    for img_path in sample_images:
        ann_path = train_ann_dir / f"{img_path.stem}.txt"

        print(f"\n Image: {img_path.name}")
        print(f"    Size: {img_path.stat().st_size / 1024:.1f} KB")

        if ann_path.exists():
            with open(ann_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()[:15]

            print(f"    Annotation preview (first 15 lines):")
            for line in lines[:5]:
                print(f"    {line.strip()[:100]}...")

if __name__ == "__main__":
    inspect_dataset()