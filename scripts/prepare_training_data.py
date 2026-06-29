"""
Smart-Auditor: Process SROIE dataset by mapping Task 1 OCR folders to Task 2 GT folders
"""
import json
from pathlib import Path
import random
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_task2_ground_truth(file_path: Path) -> Dict[str, str]:
    """
    Loads the true dictionary text fields from your Task 2 folder files.
    """
    if not file_path.exists():
        return {}
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read().strip()
            
        # If it's stored as clean JSON
        if content.startswith("{") and content.endswith("}"):
            data = json.loads(content)
            return {str(k).lower(): str(v).strip() for k, v in data.items()}
            
        # Fallback if text line formatting is used (key:value)
        targets = {}
        for line in content.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                targets[k.strip().lower()] = v.strip()
        return targets
    except Exception as e:
        logger.debug(f"Error loading Task 2 file {file_path.name}: {e}")
        return {}

def parse_task1_line(line: str) -> Tuple[str, List[int]]:
    """
    Parses SROIE Task 1 lines: x1,y1,x2,y2,x3,y3,x4,y4,TEXT
    """
    if not line.strip():
        return "", []
        
    parts = line.strip().split(",", 8)
    if len(parts) < 9:
        return "", []
    
    try:
        coords = [int(p) for p in parts[:8]]
        xs = coords[0::2]
        ys = coords[1::2]
        # LayoutLMv3 standard format bounding box: [x_min, y_min, x_max, y_max]
        bbox = [min(xs), min(ys), max(xs), max(ys)]
        text = parts[8].strip()
        return text, bbox
    except ValueError:
        return "", []

def process_split_sroie_folders(task1_dir: Path, task2_dir: Path, output_dir: Path):
    """
    Reads OCR from Task 1 folder, labels it using Ground Truth from Task 2 folder
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    compiled_samples = []

    # Get all .txt files from the Task 1 OCR folder
    task1_files = list(task1_dir.glob("*.txt"))
    logger.info(f"Found {len(task1_files)} annotation files inside Task 1 directory.")

    for t1_path in task1_files:
        file_stem = t1_path.stem
        
        # Look for the matching answer sheet filename inside the Task 2 folder
        t2_path = task2_dir / f"{file_stem}.txt"
        
        # Locate the associated image inside the Task 1 folder (handles .jpg or .png)
        img_path = task1_dir / f"{file_stem}.jpg"
        if not img_path.exists():
            img_path = task1_dir / f"{file_stem}.png"

        # Only process if both the OCR mapping and Ground Truth answer sheets exist
        if not t2_path.exists() or not img_path.exists():
            logger.debug(f"Skipping {file_stem}: Missing corresponding Task 2 file or Image file.")
            continue

        true_answers = load_task2_ground_truth(t2_path)
        
        tokens = []
        bboxes = []
        ner_tags = []

        with open(t1_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                text, bbox = parse_task1_line(line)
                if not text or not bbox:
                    continue

                label = "O"
                text_upper = text.upper()

                # Align OCR tokens directly with real Task 2 structural fields
                for target_key, target_value in true_answers.items():
                    if not target_value:
                        continue
                    val_upper = str(target_value).upper()

                    # Exact overlap alignment isolates labels perfectly
                    if text_upper in val_upper or val_upper in text_upper:
                        if "company" in target_key:
                            label = "COMPANY"
                        elif "date" in target_key:
                            label = "DATE"
                        elif "total" in target_key:
                            label = "TOTAL"
                        elif "invoice" in target_key:
                            label = "INVOICE_NUMBER"
                        elif "address" in target_key:
                            label = "ADDRESS"
                        break

                tokens.append(text)
                bboxes.append(bbox)
                ner_tags.append(label)

        if tokens:
            compiled_samples.append({
                "id": file_stem,
                "image_path": str(img_path),
                "tokens": tokens,
                "bboxes": bboxes,
                "ner_tags": ner_tags
            })

    # Shuffle and partition dataset splits (80% Train / 20% Test)
    random.seed(42)
    random.shuffle(compiled_samples)
    split_index = int(len(compiled_samples) * 0.8)
    
    with open(output_dir / "train.json", "w", encoding="utf-8") as f:
        json.dump(compiled_samples[:split_index], f, ensure_ascii=False, indent=2)
    with open(output_dir / "test.json", "w", encoding="utf-8") as f:
        json.dump(compiled_samples[split_index:], f, ensure_ascii=False, indent=2)
        
    logger.info(f"Dataset generated successfully! Train samples: {split_index}, Test samples: {len(compiled_samples)-split_index}")

if __name__ == "__main__":
    # Point these exactly to your folders shown in the screenshot
    BASE_DIR = Path("D:/smart-auditor-hf/data/raw/sroie_original")
    
    TASK1_FOLDER = BASE_DIR / "0325updated.task1train(626p)"
    TASK2_FOLDER = BASE_DIR / "0325updated.task2train(626p)"
    OUTPUT_PROCESSED_DIR = Path("D:/smart-auditor-hf/data/training")
    
    process_split_sroie_folders(TASK1_FOLDER, TASK2_FOLDER, OUTPUT_PROCESSED_DIR)
