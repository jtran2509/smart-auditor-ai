"""
Invoice Processor using LayoutLMv3 for document understanding
Extracts: Company Name, Date, Total Amount, Tax Code, Invoice Number
"""

import torch
from PIL import Image
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from pathlib import Path
from typing import Dict, List, Tuple
import logging
import easyocr
import re
import pytesseract

logger = logging.getLogger(__name__)

class InvoiceProcessor:
    """
    Process Invoice images and extract key information
    """
    def __init__(self, model_name: str = "microsoft/layoutlmv3-base", use_finetuned: bool = True):
        """
        Initalize LayoutLMv3 model

        Args:
            model_name: Pretrained model from Hugging Face
            use_finetuned: If True, use your own fine-tuned model
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Load processor and model
        self.processor = LayoutLMv3Processor.from_pretrained(model_name, apply_ocr=True)

        if use_finetuned:
            # Use community fine-tuned model (F1 0.94 on SROIE)
            finetune_path = "Theivaprakasham/layoutlmv3-finetuned-sroie"
            logger.info(f"Loading fine-tuned model from {finetune_path}")
            self.model = LayoutLMv3ForTokenClassification.from_pretrained(finetune_path)

        else:
            self.model = LayoutLMv3ForTokenClassification.from_pretrained(model_name)

        self.model.to(self.device)
        self.model.eval()

        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id

        # Create mapping from labels names to our simplified keys
        self.label_to_key = {}

        for label in self.id2label.values():
            label_upper = label.upper()
            if "COMPANY" in label_upper or "NAME" in label_upper:
                self.label_to_key[label] = "company_name"
            elif "DATE" in label_upper:
                self.label_to_key[label] = "date"
            elif "TOTAL" in label_upper or "AMOUNT" in label_upper:
                self.label_to_key[label] = "total_amount"
            elif "INVOICE" in label_upper or "INV_NUMBER" in label_upper:
                self.label_to_key[label] = "invoice_number"
            elif "ADDRESS" in label_upper:
                self.label_to_key[label] ="address"
            else:
                self.label_to_key[label] = None
        
        logger.info(f"Model loaded with {len(self.id2label)} labels...")
        logger.info("InvoiceProcessor initialize successfully")

    def extract_entity_name(self, label: str) -> str:
        """
        Extract entity name from label (e.g. "B-COMPANY" -> "company")
        """
        if label.startswith("B-") or label.startswith("I-"):
            return label[2:].lower()
        elif label.startswith("B_") or label.startswith("I_"):
            return label[2:].lower()
        return label.lower()
    
    def extract_with_model(self, image_path: Path) -> Dict[str, str]:
        """
        Extract fields using LayoutLMv3 model
        """
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")

            # Process image (LayoutLMv3 handles OCR internally)
            encoding = self.processor(image, return_tensors="pt", truncation=True)

            # Move to device
            encoding = {k: v.to(self.device) for k, v in encoding.items()}

            # Inference
            with torch.no_grad():
                outputs= self.model(**encoding)

            # Get prediction
            predictions = outputs.logits.argmax(-1).squeeze().tolist()

            # Decode tokens and predictions
            tokens = self.processor.tokenizer.convert_ids_to_tokens(encoding["input_ids"].squeeze().tolist())

            # Extract entities
            extracted = {
                "company_name": "",
                "date": "",
                "total_amount": "",
                "invoice_number": "",
                "address": ""
            }

            current_entity= None
            current_text = []

            for token, pred_id in zip(tokens, predictions):
                label = self.id2label.get(str(pred_id), "O")

                if label.startswith("B-") or label.startswith("B_"):
                    # Save previous entity
                    if current_entity and current_text:

                        entity_name = self.extract_entity_name(current_entity)
                        if entity_name in extracted:
                            extracted[entity_name] = " ".join(current_text).replace("##", "")

                    # Start new entity
                    current_entity = label
                    current_text = [token]

                elif (label.startswith("I-") or label.startswith("I_")) and current_entity:
                    current_text.append(token)
                else:
                    # outside token - save if we have an entity
                    if current_entity and current_text:
                        entity_name = self.extract_entity_name(current_entity)
                        if entity_name in extracted:
                            extracted[entity_name] = " ".join(current_text).replace("##", "")
                    
                    current_entity = None
                    current_text = []

            # Save last entity
            if current_entity and current_text:
                entity_name = self.extract_entity_name(current_entity)
                if entity_name in extracted:
                    extracted[entity_name] = " ".join(current_text).replace("##", "")
            
            # CLean up extracted values
            extracted = {k: v.strip() for k, v in extracted.items() if v}

            logger.info(f"Extracted fields from {image_path.name}: {extracted}")
            return extracted
        
        except Exception as e:
            logger.error(f"Error processing {image_path}: {str(e)}")
            return {}

    def extract_fields_regex(self, image_path: Path) -> Dict[str, str]:
        """
        Extract fields using Regex on OCR text (fall back when ML fails)
        """
        # Set Tesseract path
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

        # Load image and run OCR
        image = Image.open(image_path).convert("RGB")
        text = pytesseract.image_to_string(image)

        extracted = {
        "company_name": "",
        "date": "",
        "total_amount": "",
        "invoice_number": "",
        "address": ""
        }

        # Clean text
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        full_text = " ".join(lines)

        # # Debug: print first 500 chars
        # print(f"\n OCR text (first 500 chars): \n{full_text[:500]}\n")

        # 1. Company name (usually the first row or rows containing  SDN BHD, INC, LTD)
        company_patterns = [
            r'^([A-Z][A-Z\s&]+(?:SDN BHD|INC|LTD|LLC|CORP|CO\.?))',
            r'([A-Z][A-Z\s&]{5,50}(?:SDN BHD|INC|LTD))',
            r'([A-Z][A-Z\s&]{5,50})'
        ]
        for pattern in company_patterns:
            match= re.search(pattern, text, re.MULTILINE)
            if match:
                extracted['company_name'] = match.group(1).strip()
                break

        # 2. Date
        date_patterns = [
            r'(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4})',
            r'(\d{1,2}/\d{1,2}/\d{4})',
            r'(\d{4}-\d{2}-\d{2})'
        ]
        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                extracted['date'] = match.group(1)
                break

        # 3. Total amount
        amount_patterns = [
            # Pattern 1: "Total Amount: RM6.69" or "Nett Total: RM6.70"
            r'(?:Total Amount:|Nett Total:|Amount Due:|TOTAL)\s*:?\s*([A-Z]{2,3})?\s*([\d,]+\.\d{2})',
            # Pattern 2: "RM6.69" at end of line
            r'([A-Z]{2,3})\s*([\d,]+\.\d{2})\s*$',
            # Pattern 3: Just the number with possible currency
            r'(\d+\.\d{2})\s*(?:MYR|RM|USD|EUR|SGD)?'
        ]
        amount = None
        for pattern in amount_patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                amount = match.group(1)
                if amount:
                    break

        # If not found, try to find any decimal number that looks like money
        if amount:
            # Remove commas and clean
            extracted['total_amount'] = amount.replace(',', '').strip()
            # FALLBACK
            money_matches=re.findall(r'[\d,]+\.\d{2}', full_text)
            if money_matches:
                amounts = [float(m.replace(',', '')) for m in money_matches]
                if amounts:
                    max_amount = max(amounts)
                    extracted['total_amount'] = f"{max_amount:.2f}"

        # 4. Invoice number
        inv_patterns = [
            r'(?:Invoice #:|Invoice No:|INV-|OR)(\w{10,})',
            r'(?:#|No\.?)[:\s]*([A-Z0-9]{8,})'
        ]
        for pattern in inv_patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                extracted['invoice_number'] = match.group(1)
                break

        # 5. Address
        address_lines = []
        for i, line in enumerate(lines):
            if re.search(r'\d+[,.]?\s+[\w\s]+(?:Street|St|Road|Rd|Lane|Ln|Drive|Dr|Jalan|Kawasan)', line, re.IGNORECASE):
                address_lines.append(line)
                # Get the next lines if present
                if i +1 < len(lines) and not re.search(r'(Invoice|Total|Tax|GST)', lines[i+1], re.IGNORECASE):
                    address_lines.append(lines[i+1])
                break
        extracted['address'] = " ".join(address_lines)

        # Debug
        print(f"\n OCR text (first 500 chars): \n{full_text[:500]}\n")
        return extracted
 
        
      
        
    def extract_fields(self, image_path: Path) -> Dict[str, str]:
        """
        Extract key fields from invoice image (with fallback)
        First try LayoutMV3 model, falls back to regex if model returns nothing
        """
        
        result = self.extract_with_model(image_path)

        # If model returned meaningful results, use them
        if result and any(result.values()):
            logger.info(f"Model extraction successful for {image_path.name}")
            return result
        
        # Otherwise fall back to regex
        logger.info(f"Model returned empty, falling back to regex for {image_path.name}")
        return self.extract_fields_regex(image_path)
    
# Singleton instance
_invoice_processor = None

def get_invoice_processor() -> InvoiceProcessor:
    global _invoice_processor
    if _invoice_processor is None:
        _invoice_processor = InvoiceProcessor()
    return _invoice_processor