import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
from PIL import Image
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Đường dẫn
image_path = Path(r"D:\smart-auditor-ai\data\processed\sroie_clean\train\images\X51007846303.jpg")

# Load model và processor
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=True)
model = LayoutLMv3ForTokenClassification.from_pretrained("Theivaprakasham/layoutlmv3-finetuned-sroie")

# Load ảnh
image = Image.open(image_path).convert("RGB")

# Process
encoding = processor(image, return_tensors="pt", truncation=True)

# In ra những gì model nhìn thấy
print("=" * 60)
print("WHAT THE MODEL SEES:")
print("=" * 60)

# In ra tokens (words) mà model nhận được từ OCR
tokens = processor.tokenizer.convert_ids_to_tokens(encoding["input_ids"].squeeze().tolist())
print(f"\n📝 Number of tokens: {len(tokens)}")
print(f"First 50 tokens:")
for i, token in enumerate(tokens[:50]):
    print(f"  {i}: {token}")

# In ra bounding boxes (vị trí của từng token trên ảnh)
boxes = encoding["bbox"].squeeze().tolist()
print(f"\n📐 Sample bounding boxes (first 10):")
for i, box in enumerate(boxes[:10]):
    print(f"  Token {i} ('{tokens[i]}'): {box}")

# Thử inference và in ra predictions chi tiết
print("\n" + "=" * 60)
print("MODEL PREDICTIONS:")
print("=" * 60)

with torch.no_grad():
    outputs = model(**encoding)

predictions = outputs.logits.argmax(-1).squeeze().tolist()
id2label = model.config.id2label

# In ra các token có prediction không phải O
print("\n🔍 Tokens with non-O predictions:")
found_any = False
for i, (token, pred_id) in enumerate(zip(tokens, predictions)):
    label = id2label.get(str(pred_id), "O")
    if label != "O":
        found_any = True
        print(f"  Token {i}: '{token}' -> {label}")

if not found_any:
    print("  ❌ NO NON-O PREDICTIONS FOUND!")
    print("  Model predicted 'O' (Outside) for every token.")
    
    # In phân phối predictions
    from collections import Counter
    pred_counts = Counter([id2label.get(str(p), "O") for p in predictions])
    print(f"\n📊 Prediction distribution:")
    for label, count in pred_counts.most_common(10):
        print(f"  {label}: {count} times")