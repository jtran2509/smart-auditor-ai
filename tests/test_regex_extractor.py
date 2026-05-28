import pytest
from pathlib import Path
from app.models.invoice_processor import InvoiceProcessor

def test_regex_extraction():
    processor = InvoiceProcessor()
    test_image = Path(r"D:\smart-auditor-ai\data\processed\sroie_clean\train\images\X51005200938.jpg")

    if test_image.exists():
        result = processor.extract_fields_regex(test_image)
        assert "company name" in result
        print(f"Extracted: {result}")