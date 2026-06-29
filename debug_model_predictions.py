"""
Debug: Xem model predict được gì
"""
import torch
from PIL import Image
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from pathlib import Path
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Đường dẫn model
model_path = r"D:\smart-auditor-ai\app\models\layoutlmv3-finetuned"
processor_path = r"D:\smart-auditor-ai\app\models\layoutlmv3-processor"

print("Loading processor and model...")
processor = LayoutLMv3Processor.from_pretrained(processor_path, apply_ocr=True)
model = LayoutLMv3ForTokenClassification.from_pretrained(model_path)
model.eval()

# Test image
test_image = Path(r"D:\smart-auditor-ai\data\processed\sroie_clean\train\images\X51007846303.jpg")
print(f"\nTesting with: {test_image.name}")

# Process
image = Image.open(test_image).convert("RGB")
encoding = processor(image, return_tensors="pt", truncation=True)

with torch.no_grad():
    outputs = model(**encoding)

# Get predictions
predictions = outputs.logits.argmax(-1).squeeze().tolist()
tokens = processor.tokenizer.convert_ids_to_tokens(encoding["input_ids"].squeeze().tolist())

id2label = model.config.id2label

# In ra các token có prediction KHÁC "O"
print("\n" + "=" * 60)
print("TOKENS WITH NON-O PREDICTIONS:")
print("=" * 60)

found_any = False
for i, (token, pred_id) in enumerate(zip(tokens, predictions)):
    label = id2label.get(str(pred_id), "O")
    if label != "O":
        found_any = True
        print(f"  Token {i}: '{token}' -> {label}")

if not found_any:
    print("  ❌ NO NON-O PREDICTIONS FOUND!")
    print("  Model predicted 'O' for every token.")
    
    # In distribution of predictions
    from collections import Counter
    pred_counts = Counter([id2label.get(str(p), "O") for p in predictions])
    print(f"\n📊 Prediction distribution:")
    for label, count in pred_counts.most_common(10):
        print(f"  {label}: {count} times")
else:
    print(f"\n✅ Found {sum(1 for l in predictions if id2label.get(str(l), 'O') != 'O')} non-O predictions")

# In một vài tokens đầu để xem OCR có đọc được gì không
print("\n" + "=" * 60)
print("FIRST 30 TOKENS (OCR OUTPUT):")
print("=" * 60)
for i, token in enumerate(tokens[:30]):
    print(f"  {i}: '{token}'")