# scripts/clean_sroie_dataset.py
import os
import shutil
from pathlib import Path
import random
from collections import defaultdict

def organize_sroie_correctly():
    """
    Tổ chức dataset SROIE một cách CHÍNH XÁC:
    - Giữ nguyên cặp ảnh-annotation dựa trên tên file trùng nhau
    - Chia train/test theo tỷ lệ 80/20 (no official file list chính thức)
    """
    
    # Đường dẫn - ĐIỀU CHỈNH THEO MÁY BẠN
    RAW_DIR = Path(r"D:\smart-auditor-ai\data\raw\sroie_original")  # Thư mục chứa 5 folder bạn tải về
    OUTPUT_DIR = Path(r"D:\smart-auditor-ai\data\processed\sroie_clean")
    
    # Tạo thư mục output
    train_img_dir = OUTPUT_DIR / "train" / "images"
    train_ann_dir = OUTPUT_DIR / "train" / "annotations"
    test_img_dir = OUTPUT_DIR / "test" / "images"
    test_ann_dir = OUTPUT_DIR / "test" / "annotations"
    
    for d in [train_img_dir, train_ann_dir, test_img_dir, test_ann_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Bước 1: Gom tất cả ảnh và annotation từ các folder
    all_pairs = {}  # filename -> {'image': path, 'annotation': path}
    
    # Tìm tất cả folder con trong raw
    for folder in RAW_DIR.iterdir():
        if not folder.is_dir():
            continue
            
        print(f"Scanning {folder.name}...")
        
        # Tìm tất cả ảnh trong folder này
        for img_file in folder.glob("*.jpg"):
            base_name = img_file.stem  # Ví dụ: X51016469619
            
            # Tìm file txt tương ứng (cùng tên)
            # Có thể nằm trong folder hiện tại hoặc folder khác
            txt_file = folder / f"{base_name}.txt"
            
            # Nếu không tìm thấy txt trong cùng folder, tìm trong các folder khác
            if not txt_file.exists():
                for other_folder in RAW_DIR.iterdir():
                    if other_folder != folder and other_folder.is_dir():
                        candidate = other_folder / f"{base_name}.txt"
                        if candidate.exists():
                            txt_file = candidate
                            break
            
            if txt_file.exists():
                if base_name not in all_pairs:
                    all_pairs[base_name] = {}
                all_pairs[base_name]['image'] = img_file
                all_pairs[base_name]['annotation'] = txt_file
                all_pairs[base_name]['has_both'] = True
    
    # Bước 2: Lọc chỉ giữ các cặp có cả ảnh và annotation
    valid_pairs = {
        name: data for name, data in all_pairs.items() 
        if 'image' in data and 'annotation' in data
    }
    
    print(f"\n📊 Found {len(valid_pairs)} valid image-annotation pairs")
    
    # Bước 3: Random split train/test (80/20)
    pair_names = list(valid_pairs.keys())
    random.seed(42)  # Fixed seed for reproducibility
    random.shuffle(pair_names)
    
    split_idx = int(len(pair_names) * 0.8)
    train_names = pair_names[:split_idx]
    test_names = pair_names[split_idx:]
    
    print(f"   Train: {len(train_names)} pairs")
    print(f"   Test: {len(test_names)} pairs")
    
    # Bước 4: Copy files
    print("\n📁 Copying files...")
    
    for name in train_names:
        pair = valid_pairs[name]
        shutil.copy2(pair['image'], train_img_dir / f"{name}.jpg")
        shutil.copy2(pair['annotation'], train_ann_dir / f"{name}.txt")
    
    for name in test_names:
        pair = valid_pairs[name]
        shutil.copy2(pair['image'], test_img_dir / f"{name}.jpg")
        shutil.copy2(pair['annotation'], test_ann_dir / f"{name}.txt")
    
    print("\n✅ Done!")
    print(f"   Train: {len(list(train_img_dir.glob('*.jpg')))} images, {len(list(train_ann_dir.glob('*.txt')))} annotations")
    print(f"   Test: {len(list(test_img_dir.glob('*.jpg')))} images, {len(list(test_ann_dir.glob('*.txt')))} annotations")
    
    # Bước 5: Lưu metadata
    import json
    metadata = {
        "total_pairs": len(valid_pairs),
        "train_pairs": len(train_names),
        "test_pairs": len(test_names),
        "train_test_split_ratio": "80/20",
        "random_seed": 42
    }
    
    with open(OUTPUT_DIR / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n📄 Metadata saved to {OUTPUT_DIR / 'metadata.json'}")

if __name__ == "__main__":
    organize_sroie_correctly()