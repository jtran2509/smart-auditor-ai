"""
Smart-Auditor: Production Unseen A/B Testing Suite
Aligned to match 'task3-test' images with 'text.task1&2-test' answer sheets.
"""
import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from app.models.invoice_processor import InvoiceProcessor
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def load_test_ground_truth(file_path: Path) -> Dict[str, str]:
    """
    Parses SROIE Task 3 Test files containing definitive target answer sheets.
    Handles both clean JSON objects and legacy colon-separated strings smoothly.
    """
    if not file_path.exists():
        return {}
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read().strip()
        
        # Read standard JSON configuration data sheets
        if content.startswith("{") and content.endswith("}"):
            data = json.loads(content)
            return {str(k).lower().strip(): str(v).strip() for k, v in data.items()}
        
        # Fallback processing loop for text key:value formats
        targets = {}
        for line in content.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                targets[k.strip().lower()] = v.strip()
        return targets
    except Exception as e:
        print(f"⚠️ Warning error parsing test ground truth sheet {file_path.name}: {e}")
        return {}

def compare_fields(predicted: Dict[str, str], expected: Dict[str, str]) -> Tuple[int, int, Dict[str, str]]:
    """
    Evaluates extraction precision parameters against expected target values.
    Standardizes on keys: company_name, date, total_amount, invoice_number, address
    """
    key_mapping = {
        "company": "company_name",
        "date": "date",
        "total": "total_amount",
        "invoice": "invoice_number",
        "invoice_number": "invoice_number",
        "address": "address"
    }

    correct = 0
    total = 0
    details = {}

    for gt_key, gt_val in expected.items():
        target_field = key_mapping.get(gt_key)
        if not target_field or not gt_val.strip():
            continue  # Skip unmapped or empty target parameters safely

        total += 1
        pred_val = str(predicted.get(target_field, "")).strip().lower()
        exp_val = str(gt_val).strip().lower()

        # Fuzzy matching check handles varying OCR row text block concatenations
        if pred_val and exp_val and (pred_val == exp_val or pred_val in exp_val or exp_val in pred_val):
            correct += 1
            details[target_field] = "✅ MATCH"
        else:
            details[target_field] = f"❌ MISMATCH (Expected: '{exp_val}', Got: '{pred_val}')"

    return correct, total, details

class ABTestOnTestSet:
    def __init__(self, base_data_dir: Path):
        self.results = {
            "regex": {"correct": 0, "total": 0, "time": 0.0, "details": []},
            "model": {"correct": 0, "total": 0, "time": 0.0, "details": []}
        }

        print("\n🔍 --- SMART-AUDITOR DIRECTORY DEBUGGER ---")
        print(f"Checking Base Directory: {base_data_dir.resolve()}")
        
        if base_data_dir.exists():
            print("📁 Real folders found inside this directory by Python:")
            for item in base_data_dir.iterdir():
                if item.is_dir():
                    # repr() forces Python to display hidden spaces like 'folder ' as 'folder '
                    print(f"   ↳ {repr(item.name)}")
        else:
            print("❌ CRITICAL: The base directory path itself does not exist!")
        print("-------------------------------------------\n")

        # Explicit folder mappings
        self.images_dir = base_data_dir / "task3-test 347p"
        self.gt_dir = base_data_dir / "text.task1&2-test(361p)"


    def load_test_dataset(self) -> List[Dict]:
        """
        Scans test folders, pairing raw images with target ground truth dictionaries.
        """
        if not self.images_dir.exists() or not self.gt_dir.exists():
            print(f"❌ Error: Verification source directories missing layout paths!\nPath 1: {self.images_dir}\nPath 2: {self.gt_dir}")
            return []

        # Find all images inside the task3-test directory
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"]
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(self.images_dir.glob(ext)))

        print(f"📂 Discovered {len(image_files)} raw test invoice images inside images directory.")

        compiled_dataset = []
        for img_path in image_files:
            file_stem = img_path.stem
            
            # Find the matching target answer file inside your text folder
            gt_path = self.gt_dir / f"{file_stem}.txt"
            if not gt_path.exists():
                gt_path = self.gt_dir / f"{file_stem}.json"
            
            if gt_path.exists():
                ground_truth = load_test_ground_truth(gt_path)
                if ground_truth:
                    compiled_dataset.append({
                        "image_path": img_path,
                        "ground_truth": ground_truth
                    })

        print(f"✅ Successfully paired {len(compiled_dataset)} document profiles with valid ground-truth answers.")
        return compiled_dataset

    def run_benchmark(self, dataset: List[Dict], limit: int = None):
        """
        Executes sequence comparison benchmarking across both processing types.
        """
        if limit:
            dataset = dataset[:limit]

        print("\n" + "="*50)
        print("🧪 STARTING SMART-AUDITOR BENCHMARK SUITE")
        print("   Evaluating Performance: Regex Pipeline vs Fine-Tuned LayoutLMv3 AI")
        print("="*50)
        print(f"Target Evaluation Payload Volume: {len(dataset)} items\n")

        processor = InvoiceProcessor()

        for idx, sample in enumerate(dataset):
            img_path = sample["image_path"]
            expected = sample["ground_truth"]

            print(f"[{idx+1}/{len(dataset)}] Auditing Document Profile: {img_path.name}")

            # --- REGEX RUN PIPELINE ---
            if hasattr(processor, "extract_fields_regex"):
                start_time = time.time()
                regex_res = processor.extract_fields_regex(img_path)
                regex_duration = time.time() - start_time
            else:
                regex_res = {k: "" for k in ["company_name", "date", "total_amount", "invoice_number", "address"]}
                regex_duration = 0.001

            reg_correct, reg_total, reg_details = compare_fields(regex_res, expected)
            self.results["regex"]["correct"] += reg_correct
            self.results["regex"]["total"] += reg_total
            self.results["regex"]["time"] += regex_duration
            self.results["regex"]["details"].append({
                "image": img_path.name, "correct": reg_correct, "total": reg_total, "details": reg_details
            })
            print(f"    ↳ Pipeline [Regex]:     Accuracy Score: {reg_correct}/{reg_total} fields matched in {regex_duration:.3f}s")

            # --- MODEL RUN PIPELINE ---
            start_time = time.time()
            model_res = processor.extract_with_model(img_path)
            model_duration = time.time() - start_time

            mod_correct, mod_total, mod_details = compare_fields(model_res, expected)
            self.results["model"]["correct"] += mod_correct
            self.results["model"]["total"] += mod_total
            self.results["model"]["time"] += model_duration
            self.results["model"]["details"].append({
                "image": img_path.name, "correct": mod_correct, "total": mod_total, "details": mod_details
            })
            print(f"    ↳ Pipeline [LayoutLMv3]: Accuracy Score: {mod_correct}/{mod_total} fields matched in {model_duration:.3f}s")

    def print_final_report(self):
        """
        Outputs clean, actionable statistical performance analytics metrics.
        """
        print("\n" + "="*50)
        print("📊 METRIC ANALYSIS PERFORMANCE SCORECARD")
        print("="*50)

        metrics = {}
        for method in ["regex", "model"]:
            correct = self.results[method]["correct"]
            total = self.results[method]["total"]
            total_time = self.results[method]["time"]
            samples_count = len(self.results[method]["details"])

            accuracy = (correct / total * 100) if total > 0 else 0.0
            avg_latency = (total_time / samples_count) if samples_count > 0 else 0.0
            metrics[method] = accuracy

            label_title = "Deterministic Regular Expressions (Regex)" if method == "regex" else "Fine-Tuned Multi-Modal Transformer (LayoutLMv3)"
            print(f"\n🔹 Engine variant: {label_title}")
            print(f"   📊 Accuracy Precision: {accuracy:.2f}% ({correct}/{total} fields verified successfully)")
            print(f"   ⏱️  Mean Processing Latency: {avg_latency:.3f} seconds per invoice scan file")

        print("\n" + "="*50)
        print("🚀 DEPLOYMENT VALUE REVELATION ANALYSIS")
        print("="*50)
        improvement = metrics["model"] - metrics["regex"]
        if improvement > 0:
            print(f"✅ The fine-tuned LayoutLMv3 Visual AI outpaces Regex by +{improvement:.2f}% accuracy points!")
            if improvement >= 15.0:
                print("🎯 Strategic Milestone Cleared: AI model yields a major reduction in manual audit human workloads.")
        elif improvement < 0:
            print(f"⚠️ Warning: Regex variation remains {-improvement:.2f}% ahead. Review model training convergence loops.")
        else:
            print("⚖️ Both extraction configurations yield identical statistical layout target match performance parity.")
        print("="*50 + "\n")

if __name__ == "__main__":
    # Point this strictly to your main local SROIE directory footprint
    SROIE_BASE_PATH = Path(r"D:/smart-auditor-hf/data/raw/sroie_original")
    
    tester = ABTestOnTestSet(SROIE_BASE_PATH)
    test_suite_data = tester.load_test_dataset()

    if not test_suite_data:
        print("❌ Evaluation halted: Zero valid document pairs found. Check system path declarations.")
        sys.exit(1)

    # Run the test on all paired images (use limit=10 for a quick 10-image trial run)
    tester.run_benchmark(test_suite_data, limit=None)
    tester.print_final_report()
