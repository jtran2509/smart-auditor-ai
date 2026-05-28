# scripts/debug_annotation.py
from pathlib import Path

# Lấy 1 file annotation bất kỳ
ann_path = Path(r"D:\smart-auditor-ai\data\processed\sroie_clean\train\annotations").glob("*.txt").__next__()

print(f"File: {ann_path.name}")
print("=" * 50)

with open(ann_path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f.readlines()[:20]):  # Chỉ in 20 dòng đầu
        parts = line.strip().split()
        print(f"Line {i}: {parts}")
        print(f"   Number of parts: {len(parts)}")
        print(f"   Last 5 parts: {parts[-5:]}")
        print()