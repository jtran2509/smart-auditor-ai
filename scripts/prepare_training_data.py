"""
Prepare data from SROIE format into suitable format for fine-tuning LayoutLMV3
"""
import json
from pathlib import Path
import random
from typing import Dict, List, Tuple
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def detect_encoding(file_path: Path):
    """
    Detect file encoding by trying common encodings
    """
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    for encoding in encodings:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                f.read()
            return encoding
        except UnicodeDecodeError:
            continue
    return "latin-1" # Fallback

def parse_sroie_line(line: str):
    """
    Parse SROIE annotation lines.

    Format: x1,y1,x2,y2,x3,y3,x4,y4 TEXT1 TEXT2 ... TEXTN
    where (x1,y1)...(x4,y4) are 4 corners of the bounding box
    Returns: (text, [x_min, y_min, x_max, y_max], label)
    """
    if not line.strip():
        return None, None, None
    
    parts = line.strip().split()
    if len(parts) < 2:
        return None, None, None
    
    # First part contains the 8 coordinates separated by commas
    coord_part = parts[0]

    # Remove trailing special characters like '>'
    coord_part = coord_part.rstrip('>')

    # Split by comma
    coords = coord_part.split(",")

    # Filter out empty strings and non-numeric
    numeric_coords = []
    for c in coords:
        try:
            # Try to convert to int, if fails, skip
            numeric_coords.append(int(c))
        except ValueError:
            # Check if it's a number with trailing numbers
            match = re.search(r"(\d+)", c)
            if match:
                numeric_coords.append(int(match.group(1)))
            continue
    # Need at least 4 coordinates (x1, y1, x2, y2) for bounding box
    if len(numeric_coords) < 4:
        return None, None, None
    
    # Take the first 4 or 8 coordinates
    if len(numeric_coords) >= 8:
        x1, y1, x2, y2, x3, y3, x4, y4 = numeric_coords[:8]

        # Convert to x_min, y_min, x_max, y_max (LayoutLMv3 format)
        x_min = min(x1, x2, x3, x4)
        x_max = max(x1, x2, x3, x4)
        y_min = min(y1, y2, y3, y4)
        y_max = max(y1, y2, y3, y4)
    else:
        # Only 4 coordinates (x1, y1, x2, y2)
        x1, y1, x2, y2 = numeric_coords[:4]
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)

    # The rest parts are the text (maybe multiple words)
    text = " ".join(parts[1:])
    if not text:
        return None, None, None

    # Determine label based on keywords in text (simplified)
    text_upper = text.upper()
    label = "O"

    if any(keyword in text_upper for keyword in ["COMPANY", "SDN BHD", "INC", "LTD"]):
        label= "COMPANY"
    elif any(keyword in text_upper for keyword in ["DATE", "JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]):
        label = "DATE"
    elif any(keyword in text_upper for keyword in ["TOTAL", "NETT", "AMOUNT", "RM", "$"]):
        label = "TOTAL"
    elif any(keyword in text_upper for keyword in ["INVOICE", "#", "NO."]):
        label= "INVOICE_NUMBER"
    elif any(keyword in text_upper for keyword in ["ADDRESS", "JALAN", "STREET", "ROAD"]):
        label = "ADDRESS"

    bbox = [x_min, y_min, x_max, y_max]

    return text, bbox, label

def convert_sroie_to_layoutlmv3_format(
    image_dir: Path,
    annotations_dir: Path,
    output_dir: Path
):
    """
    Convert SROIE annotations to LayoutLMV3 training format
    LayoutLMV3 needs: tokens, boxes, ner_tags
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    train_data=[]

    # Get all of the image pair annotations
    image_files = list(image_dir.glob("*.jpg"))
    logger.info(f"Found {len(image_files)} images.")

    for img_idx, img_path in enumerate(image_files):
        ann_path = annotations_dir / f"{img_path.stem}.txt"

        if not ann_path.exists():
            logger.debug(f"No annotation for {img_path.name}, skipping!")
            continue

        # Detect encoding
        encoding = detect_encoding(ann_path)

        # READ ANNOTATIONS
        tokens = []
        boxes = []
        ner_tags = []

        try:
            with open(ann_path, "r", encoding=encoding) as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue

                    text, bbox, label = parse_sroie_line(line)

                    if text and bbox:
                        tokens.append(text)
                        boxes.append(bbox)
                        ner_tags.append(label)

        except Exception as e:
            logger.warning(f"Error reading {ann_path.name}: {e}")
            continue

        if tokens:
            train_data.append({
                "id": img_path.stem,
                "image_path": str(img_path),
                "tokens": tokens,
                "bboxes": boxes,
                "ner_tags": ner_tags
            })
        # Progress log every 100 images
        if (img_idx + 1) % 100 == 0:
            logger.info(f"Processed {img_idx +1}/{len(image_files)} images, test: {len(train_data)}")

    logger.info(f"Total valid samples: {len(train_data)}")

    if len(train_data) == 0:
        logger.error("NO valid training data found!")
        return train_data
    
    # Save to JSON
    output_path = output_dir / "training_data.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(train_data)} samples to {output_path}")

    # Create train/test split (80/20)
    random.seed(42)
    random.shuffle(train_data)
    split_idx = int(len(train_data) * 0.8)

    train_split = train_data[:split_idx]
    test_split= train_data[split_idx:]

    with open(output_dir / "train.json", "w") as f:
        json.dump(train_split, f, indent=2, ensure_ascii=False)
    with open(output_dir / "test.json", "w") as f:
        json.dump(test_split, f, indent=2, ensure_ascii=False)

    logger.info(f" Train: {len(train_split)}, Test: {len(test_split)}")

    # Print statistics
    if train_split:
        sample = train_split[0]
        all_labels =set()
        for item in train_split:
            all_labels.update(item['ner_tags'])
        logger.info(f"📊 Statistics:")
        logger.info(f"  Total unique labels: {sorted(all_labels)}")
        logger.info(f"  Sample: {sample['id']}")
        logger.info(f"  Tokens: {len(sample['tokens'])}")
        logger.info(f"  First 10 tokens: {sample['tokens'][:10]}")
        logger.info(f"  First 10 labels: {sample['ner_tags'][:10]}")

    return train_data

if __name__ == "__main__":
    IMAGES_DIR = Path(r"D:\smart-auditor-ai\data\processed\sroie_clean\train\images")
    ANNOTATIONS_DIR = Path(r"D:\smart-auditor-ai\data\processed\sroie_clean\train\annotations")
    OUTPUT_DIR = Path(r"D:\smart-auditor-ai\data\training")
    
    convert_sroie_to_layoutlmv3_format(IMAGES_DIR, ANNOTATIONS_DIR, OUTPUT_DIR)