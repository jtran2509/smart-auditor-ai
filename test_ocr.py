import pytesseract
from PIL import Image
from pathlib import Path

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# test_image = Path(r"D:\smart-auditor-ai\data\processed\sroie_clean\train\images\X51007846303.jpg")
# test_image = Path(r"D:\smart-auditor-ai\data\processed\sroie_clean\train\images\X00016469670.jpg")
test_image = Path(r"D:\smart-auditor-ai\data\processed\sroie_clean\train\images\X51005303661(4).jpg")

# Read image
image = Image.open(test_image)

# Run OCR
text = pytesseract.image_to_string(image)

print(f" Image: {test_image.name}")
print(f"Image size: {image.size}")
print(f"\n== OCR output==")
print(text if text.strip() else "Cannot READ ANYTHING!")

# Try with another config
text_psm6 = pytesseract.image_to_string(image, config="--psm 6")
print("\n === OCR Result ===")
print(text_psm6 if text_psm6.strip() else "CANNOT READ!")