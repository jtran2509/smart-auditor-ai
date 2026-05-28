import zipfile
from pathlib import Path 

def extract_model():
    print("Extracting...")
    model_zip = Path("models/layoutlmv3-finetuned.zip")
    processor_zip = Path("models/layoutlmv3-processor.zip")
    with zipfile.ZipFile(model_zip, 'r') as zip_ref:
        zip_ref.extractall("D:\smart-auditor-ai\app\models\layoutlmv3-finetuned.zip")
        
    with zipfile.ZipFile(processor_zip, 'r') as zip_ref:
        zip_ref.extractall("D:\smart-auditor-ai\app\models\layoutlmv3-processor.zip")

    print("✅ Model extracted successfully")
if __name__ == "__main__":
    extract_model()