import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from app.models.invoice_processor import get_invoice_processor
from pathlib import Path
from PIL import Image
import pytesseract
import logging
from huggingface_hub import login
import torch

# Set up loggin to see what's happening
logging.basicConfig(level=logging.INFO)

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
login(token="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")

# def debug_model_labels():
#     """Debug: In ra tất cả labels của model"""
#     processor = get_invoice_processor()
#     print(f"\n📋 Model labels:")
#     for idx, (key, label) in enumerate(processor.id2label.items()):
#         print(f"   {key}: {label}")
    
#     # Kiểm tra xem có label nào chứa 'COMPANY', 'DATE', 'TOTAL' không
#     print(f"\n🔍 Searching for relevant labels:")
#     for key, label in processor.id2label.items():
#         if any(x in label.upper() for x in ['COMPANY', 'DATE', 'TOTAL', 'ADDRESS', 'INVOICE']):
#             print(f"   Found: {key} -> {label}")

def test_processor():
    # Test with multiple images
    train_images_dir = Path(r"D:\smart-auditor-ai\data\processed\sroie_clean\train\images")
    # Get 1 image from train set
    test_images = list(train_images_dir.glob("*.jpg"))[:3]
    
    processor = get_invoice_processor()
    
    for test_image in test_images:
        print(f"\n{'='*60}")
        print(f"Testing with image: {test_image.name}")
        print("=" * 60)

        result = processor.extract_fields(test_image)

        print(f"\n✅ Extracted fields:")
        for key, value in result.items():
            if value:
                print(f"   {key}: {value}")
            else:
                print(f"    ❌ {key}: NOT FOUND")

        print()
    

if __name__ == "__main__":
    test_processor()