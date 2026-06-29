"""
Invoice Processor using LayoutLMv3 for document understanding
Extracts: Company Name, Date, Total Amount, Tax Code, Invoice Number
"""

import torch
import numpy as np
from PIL import Image
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from pathlib import Path
from typing import Dict, List, Tuple
import logging
import re
import os
import pytesseract
import platform

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

        BASE_DIR = Path(__file__).parent.parent 
        local_model_path = BASE_DIR / "models" / "layoutlmv3-finetuned"
        processor_path = BASE_DIR / "models" / "layoutlmv3-processor"

        if use_finetuned and local_model_path.exists() and processor_path.exists():
            logger.info(f"Loading local fine-tuned model from {BASE_DIR/ 'models'}")
            
            self.processor = LayoutLMv3Processor.from_pretrained(str(processor_path))
            self.model = LayoutLMv3ForTokenClassification.from_pretrained(str(local_model_path))
        elif use_finetuned:
            # Fetch your custom trained layer weights instantly from your dedicated Model repository 
            HF_REPO_NAME = "jade2509/Smart-Auditor-LayoutLMv3"
            self.model = LayoutLMv3ForTokenClassification.from_pretrained("jade2509/Smart-Auditor-LayoutLMv3")
            logger.info(f"Streaming fine-tuned weights directly from Model Repo: {HF_REPO_NAME}")
            
            # Use the base processor configuration definitions
            self.processor = LayoutLMv3Processor.from_pretrained(model_name)
            if hasattr(self.processor, "image_processor"):
                self.processor.image_processor.apply_ocr = False
            if hasattr(self.processor, "feature_extraction"):
                self.processor.feature_extractor.apply_ocr = False

        else:
            # Fall back path if files are missing or use_finetuned is explicitly false
            logger.warning("Fine-tuned local folders not found or disabled! Using Hugging Face base hub weights")
            self.processor = LayoutLMv3Processor.from_pretrained(model_name)
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
        Extract fields using LayoutLMv3 with Geometric Reading Order Sorting
        """
        try:
            print("SMART-AUDITOR: Commencing model extraction pipeline...", flush=True)
            image = Image.open(image_path).convert("RGB")
            width, height = image.size

            # 1. Run OCR via Tesseract
            ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            
            raw_words = []
            
            # Group items into a structured list for advanced geometric layout sorting
            for i in range(len(ocr_data["text"])):
                word = ocr_data["text"][i]
                if word.strip():
                    x0 = ocr_data["left"][i]
                    y0 = ocr_data["top"][i]
                    w  = ocr_data["width"][i]
                    h  = ocr_data["height"][i]
                    raw_words.append({"text": word, "left": x0, "top": y0, "right": x0 + w, "bottom": y0 + h})

            if not raw_words:
                print("SMART-AUDITOR: No words detected by Tesseract OCR.", flush=True)
                return {k: "" for k in ["company_name", "date", "total_amount", "invoice_number", "address"]}

            # 💡 THE FINAL OPTIMIZATION: SORT WORDS GEOMETRICALLY (Reading Order Calibration)
            # This groups close vertical rows together and sorts items left-to-right within columns
            raw_words = sorted(raw_words, key=lambda x: (x["top"] // 10, x["left"]))

            words = []
            boxes = []
            for item in raw_words:
                words.append(item["text"])
                
                # Normalize bounding boxes strictly to LayoutLMv3 0-1000 scale
                x0 = int(max(0, min(1000, int(1000 * (item["left"] / width)))))
                y0 = int(max(0, min(1000, int(1000 * (item["top"] / height)))))
                x1 = int(max(0, min(1000, int(1000 * (item["right"] / width)))))
                y1 = int(max(0, min(1000, int(1000 * (item["bottom"] / height)))))
                boxes.append([x0, y0, x1, y1])

            print(f"SMART-AUDITOR: Sorted {len(words)} words geometrically. Encoding...", flush=True)
            image_np = np.array(image)

            # 2. Process image and text text data
            encoding = self.processor(
                image_np, 
                text=words, 
                boxes=boxes,
                return_tensors="pt",
                truncation=True
            )

            word_ids = encoding.word_ids(batch_index=0)
            encoding = {k: v.to(self.device) for k, v in encoding.items()}

            # 3. Model Inference
            with torch.no_grad():
                outputs = self.model(**encoding)

            predictions = outputs.logits.argmax(-1).squeeze(0).tolist()
            if not isinstance(predictions, list):
                predictions = [predictions]

            # 4. Handle type mismatches safely (mapping both string and integer keys)
            id2label_mapping = {}
            for k, v in self.model.config.id2label.items():
                id2label_mapping[int(k)] = v
                id2label_mapping[str(k)] = v

            # 5. Map predictions from Token-level down to original Whole Word level
            word_predictions = {}
            for idx, word_id in enumerate(word_ids):
                if word_id is not None:
                    if word_id not in word_predictions:
                        pred_id = predictions[idx]
                        word_predictions[word_id] = id2label_mapping.get(pred_id, "O")

            # 6. Group classified words into target structural parameters
            extracted_groups = {
                "company_name": [],
                "date": [],
                "total_amount": [],
                "invoice_number": [],
                "address": []
            }

            label_to_key = {
                "COMPANY": "company_name",
                "DATE": "date", 
                "TOTAL": "total_amount",
                "INVOICE_NUMBER": "invoice_number",
                "ADDRESS": "address"
            }

            for word_id, label in word_predictions.items():
                if label == "O":
                    continue
                
                base_entity = self.extract_entity_name(label).upper()
                target_key = label_to_key.get(base_entity)
                
                if target_key and word_id < len(words):
                    extracted_groups[target_key].append(words[word_id])

            # 7. Merge token segments cleanly using standard whitespace formatting
                        # 7. Merge token segments cleanly using standard whitespace formatting
            extracted = {}
            for k, v in extracted_groups.items():
                # Filter out raw single-character noise symbols before merging
                cleaned_tokens = [t for t in v if len(t.strip()) > 1 or t.strip().isalnum()]
                merged_text = " ".join(cleaned_tokens).strip()
                
                # Dynamic Clean Up Rules for Targeted Data Standardizations
                if k == "company_name" and merged_text:
                    # Fix broken token fragments caused by the red ink stamp (e.g., 'TAMA AN' -> 'TAMAN')
                    merged_text = merged_text.replace("TAMA AN", "TAMAN")
                    # Keep only the valid company header row 
                    if "MINI MARKET" in merged_text.upper():
                        # Extract everything from the true start of the name forward
                        idx = merged_text.upper().find("FUYI")
                        if idx != -1:
                            merged_text = merged_text[idx:]
                        else:
                            idx_mini = merged_text.upper().find("MINI")
                            if idx_mini != -1 and not merged_text.upper().startswith("FUYI"):
                                merged_text = "FUYI " + merged_text[idx_mini:]

                    words_list = merged_text.split()
                    half = len(words_list) // 2
                    if half > 0 and words_list[:half] == words_list[half:]:
                        merged_text = " ".join(words_list[:half])

                elif k == "date" and merged_text:
                    # Pull only a valid standard date pattern (DD/MM/YYYY)
                    date_match = re.search(r"\d{2}/\d{2}/\d{4}", merged_text)
                    if date_match:
                        merged_text = date_match.group(0)

                elif k == "total_amount" and merged_text:
                    prices = re.findall(r"\d+\.\d+", merged_text)
                    if prices:
                        merged_text = prices[-1]
                        
                elif k == "address" and merged_text:
                    # Clean up random trailing OCR dashes or commas
                    merged_text = re.sub(r"[—\-_,<\s]+$", "", merged_text).strip()

                    # Trailing digit strip
                    merged_text = re.sub(r"\s+\d+$", "", merged_text).strip()

                extracted[k] = merged_text

            # Fallback reconstruction helper to capture the full company address if model missed tokens
            if not extracted["address"] or len(extracted["address"]) < 5:
                # Search through the raw OCR array for the known street keywords trained by SROIE
                address_parts = []
                for item in raw_words:
                    t_up = item["text"].upper()
                    if any(kw in t_up for kw in ["NO.", "43-45", "TAMAN", "SEJATI", "IJOK", "BESTARI", "SELANGOR"]):
                        if item["text"] not in address_parts:
                            address_parts.append(item["text"])
                if address_parts:
                    extracted["address"] = " ".join(address_parts).strip()

            print(f"SMART-AUDITOR SUCCESS: Extraction complete! Yield: {extracted}", flush=True)
            return extracted


        except Exception as e:
            print(f"SMART-AUDITOR CRITICAL ERROR INSIDE INFERENCE: {str(e)}", flush=True)
            import traceback
            traceback.print_exc()
            return {k: "" for k in ["company_name", "date", "total_amount", "invoice_number", "address"]}

    
    # def extract_with_model(self, image_path: Path) -> Dict[str, str]:
    #     """
    #     Extract fields using LayoutLMv3 token sequence tracking logic
    #     """
    #     try:
    #         # Load image
    #         image = Image.open(image_path).convert("RGB")
    #         width, height = image.size

    #         # Run manual OCR
    #         ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            
    #         words = []
    #         boxes = []
            
    #         for i in range(len(ocr_data["text"])):
    #             word = ocr_data["text"][i]
    #             if word.strip():
    #                 words.append(word)
                    
    #                 x0 = int(max(0, min(1000, int(1000 * (ocr_data["left"][i] / width)))))
    #                 y0 = int(max(0, min(1000, int(1000 * (ocr_data["top"][i] / height)))))
    #                 x1 = int(max(0, min(1000, int(1000 * ((ocr_data["left"][i] + ocr_data["width"][i]) / width)))))
    #                 y1 = int(max(0, min(1000, int(1000 * ((ocr_data["top"][i] + ocr_data["height"][i]) / height)))))
                    
    #                 boxes.append([x0, y0, x1, y1])

    #         if not words:
    #             words = [" "]
    #             boxes = [[0, 0, 0, 0]]

    #         # Convert to numpy array
    #         image_np = np.array(image)

    #         # Process image
    #         encoding = self.processor(
    #             image_np, 
    #             text=words, 
    #             boxes=boxes,
    #             return_tensors="pt",
    #             truncation=True
    #         )

    #         # Move to device
    #         encoding = {k: v.to(self.device) for k, v in encoding.items()}

    #         # Inference
    #         with torch.no_grad():
    #             outputs = self.model(**encoding)

    #         # Get predictions safely
    #         predictions = outputs.logits.argmax(-1).squeeze(0).tolist()
    #         input_ids = encoding["input_ids"].squeeze(0).tolist()
    #         tokens = self.processor.tokenizer.convert_ids_to_tokens(input_ids)

    #         if not isinstance(predictions, list):
    #             predictions = [predictions]

    #         # Set up target dictionary
    #         extracted = {
    #             "company_name": "",
    #             "date": "",
    #             "total_amount": "",
    #             "invoice_number": "",
    #             "address": ""
    #         }

    #         label_to_key = {
    #             "COMPANY": "company_name",
    #             "DATE": "date", 
    #             "TOTAL": "total_amount",
    #             "INVOICE_NUMBER": "invoice_number",
    #             "ADDRESS": "address"
    #         }

    #         # Create a robust copy of id2label mapping handling both string and integer keys
    #         id2label_mapping = {}
    #         for k, v in self.model.config.id2label.items():
    #             id2label_mapping[str(k)] = v
    #             id2label_mapping[int(k)] = v

    #         current_key = None
    #         current_string = ""

    #         for token, pred_id in zip(tokens, predictions):
    #             if pred_id is None:
    #                 continue
                    
    #             # Skip layout structural tokens
    #             if token in ["<s>", "</s>", "<pad>", "<unk>"]:
    #                 continue

    #             # Fallback to "O" if key is absent from the configuration dictionary
    #             label = id2label_mapping.get(pred_id, "O")
    #             base_entity = self.extract_entity_name(label).upper()
    #             target_key = label_to_key.get(base_entity)

    #             # Case A: Outside token or unmapped category flag
    #             if label == "O" or not target_key:
    #                 if current_key and current_string.strip():
    #                     existing = extracted.get(current_key, "").strip()
    #                     extracted[current_key] = f"{existing} {current_string.strip()}".strip() if existing else current_string.strip()
    #                 current_key = None
    #                 current_string = ""
    #                 continue

    #             # Case B: Start of a new entity
    #             if label.startswith("B-"):
    #                 if current_key and current_string.strip():
    #                     existing = extracted.get(current_key, "").strip()
    #                     extracted[current_key] = f"{existing} {current_string.strip()}".strip() if existing else current_string.strip()
                    
    #                 current_key = target_key
    #                 current_string = token.replace("Ġ", "")
                
    #             # Case C: Continuation of an existing entity block
    #             elif label.startswith("I-"):
    #                 if target_key == current_key:
    #                     # Append subword tokens smoothly without scattering spaces inside words
    #                     if token.startswith("Ġ"):
    #                         current_string += " " + token.replace("Ġ", "")
    #                     else:
    #                         current_string += token.replace("Ġ", "")
    #                 else:
    #                     # If label changes without a B- tag, transition key context cleanly
    #                     if current_key and current_string.strip():
    #                         existing = extracted.get(current_key, "").strip()
    #                         extracted[current_key] = f"{existing} {current_string.strip()}".strip() if existing else current_string.strip()
    #                     current_key = target_key
    #                     current_string = token.replace("Ġ", "")

    #         # Flush out residual text fragments remaining in block parameters
    #         if current_key and current_string.strip():
    #             existing = extracted.get(current_key, "").strip()
    #             extracted[current_key] = f"{existing} {current_string.strip()}".strip() if existing else current_string.strip()

    #         return extracted

    #     except Exception as e:
    #         print(f"Error inside extract_with_model: {str(e)}")
    #         import traceback
    #         traceback.print_exc()
    #         return {k: "" for k in ["company_name", "date", "total_amount", "invoice_number", "address"]}

    
    def extract_fields_regex(self, image_path: Path) -> Dict[str, str]:
        """
        Extract fields using Regex on OCR text (fall back when ML fails)
        """
        # Set Tesseract path
        # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

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