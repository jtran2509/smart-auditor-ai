"""
Test the fine-tuned LayoutLMv3 model on sample invoice
"""
import torch
from PIL import Image
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from pathlib import Path

def load_finetuned_model(model_path: str = "D:/smart-auditor-ai/app/models"):
    """
    Load fine-tuned model
    """
    processor = LayoutLMv3Processor.from_pretrained(model_path, apply_ocr=True)
    model = LayoutLMv3ForTokenClassification.from_pretrained(model_path)

    return processor, model

def extract_fields(image_path: Path, processor, model):
    """
    Extract fields from invoice using fine-tuned model
    """
    image = Image.open(image_path).convert("RGB")
    encoding = processor(image, return_tensors ="pt", truncation=True)

    with torch.no_grad():
        outputs = model(**encoding)
    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    tokens = processor.tokenizer.convert_ids_to_tokens(encoding['input_ids'].squeeze().tolist())

    id2label= model.config.id2label

    # Extract entities
    extracted = {}
    current_entity = None
    current_text = []

    for token, pred_id in zip(tokens, predictions):
        label = id2label.get(str(pred_id), "O")

        if label.startswith("B-"):
            if current_entity and current_text:
                extracted[current_entity] = " ".join(current_text)
            current_entity = label[2:].lower()
            current_text = [token]

        elif label.startswith("I-") and current_entity:
            current_text.append(token)
        else:
            if current_entity and current_text:
                extracted[current_entity] = " ".join(current_text)
            current_entity = None
            current_text = []
    return extracted

if __name__ == "__main__":
    # Load model
    processor, model = load_finetuned_model()

    # Test on a sample
    test_image = Path(r"D:\smart-auditor-ai\data\processed\sroie_clean\train\images").glob("*.jpg").__next__()

    result = extract_fields(test_image, processor, model)

    print(f"Image: {test_image.name}")
    print(f"Extracted: {result}")