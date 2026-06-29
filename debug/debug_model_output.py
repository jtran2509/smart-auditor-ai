"""
Debug model predictions - standalone test
"""
import torch
from PIL import Image
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from pathlib import Path
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Đường dẫn
model_path = r"D:\smart-auditor-ai\app\models\layoutlmv3-finetuned"
processor_path = r"D:\smart-auditor-ai\app\models\layoutlmv3-processor"

print("=" * 60)
print("LOADING MODEL AND PROCESSOR")
print("=" * 60)

# Load processor và model
processor = LayoutLMv3Processor.from_pretrained(processor_path, apply_ocr=True)
model = LayoutLMv3ForTokenClassification.from_pretrained(model_path)
model.eval()

# In labels
print("\n📋 Model labels:")
id2label = model.config.id2label
for k, v in id2label.items():
    print(f"   {k}: {v}")

# Test image
test_image = Path(r"D:\smart-auditor-ai\data\processed\sroie_clean\train\images\X51007846303.jpg")
print(f"\n🖼️ Testing with: {test_image.name}")

# Process
image = Image.open(test_image).convert("RGB")
encoding = processor(image, return_tensors="pt", truncation=True)

# Inference
with torch.no_grad():
    outputs = model(**encoding)

predictions = outputs.logits.argmax(-1).squeeze().tolist()
tokens = processor.tokenizer.convert_ids_to_tokens(encoding["input_ids"].squeeze().tolist())

# Debug: In các token có prediction khác O
print("\n🔍 TOKENS WITH NON-O PREDICTIONS:")
found = False
for i, (token, pred_id) in enumerate(zip(tokens, predictions)):
    label = id2label.get(str(pred_id), "O")
    if label != "O":
        found = True
        print(f"   Token {i}: '{token}' -> {label}")

if not found:
    print("   ❌ NO NON-O PREDICTIONS!")
    
    # In distribution
    from collections import Counter
    pred_counts = Counter([id2label.get(str(p), "O") for p in predictions])
    print(f"\n📊 Prediction distribution (first 20):")
    for label, count in list(pred_counts.items())[:20]:
        print(f"   {label}: {count}")

# In first 30 tokens để xem OCR có đọc được không
print("\n📝 FIRST 30 TOKENS (OCR OUTPUT):")
for i, token in enumerate(tokens[:30]):
    print(f"   {i}: '{token}'")