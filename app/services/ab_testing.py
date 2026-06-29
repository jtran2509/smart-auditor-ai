"""
A/B Testing Service for comparing Regex vs Fine-tuned model
"""
import random
import time
from datetime import datetime
from typing import Dict, Any
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ABTestingService():
    def __init__(self):
        self.result_file = Path(f"data/ab_test_results.json")
        self.result_file.parent.mkdir(parents=True, exist_ok=True)
        self.load_results()

        # Traffic split : 50% model_a, 50% model_b
        self.traffic_split = 0.5

    def load_results(self):
        """
        Load existing test results
        """
        if self.result_file.exists():
            with open(self.result_file, 'r') as f:
                self.results = json.load(f)
        else:
            self.results = {
                "model_a": {"name": "Regex", "total": 0, "success": 0, "total_time": 0},
                "model_b": {"name": "Fine-tuned LayoutLMv3", "total": 0, "success": 0, "total_time": 0}
            }

    def save_results(self):
        """
        Save results to file
        """
        with open(self.result_file, 'w') as f:
            json.dump(self.results, f, indent=2)

    def get_model_version(self, user_id: str = None) -> str:
        """
        Determine which model to used based on traffic split
        """
        if user_id:
            # Deterministic based on user_id (same user always get same model)
            hash_value = hash(user_id) % 100
            return "model_a" if hash_value < self.traffic_split * 100 else "model_b"
        else:
            # Random split
            return "model_a" if random.random() < self.traffic_split else "model_b"
        
    def log_result(self, model_version: str, success: bool, processing_time: float, extracted_fields: Dict):
        """
        Log the result of a model inference
        """
        self.results[model_version]['total'] +=1
        if success:
            self.results[model_version]['success'] +=1
        self.results[model_version]['total_time'] += processing_time

        self.save_results()
        logger.info(f"A/B Test - {model_version}: success={success}, time={processing_time:.3f}")

    def get_statistics(self) -> Dict:
        """
        Get comparison statistics between models
        """
        stats = {}
        for key, data in self.results.items():
            if data['total'] > 0:
                accuracy = data['success'] / data['total']
                avg_time = data['total_time'] / data['total']
                stats[key] = {
                    "name": data['name'],
                    "total": data['total'],
                    "accuracy": round(accuracy * 100, 1),
                    "avg_time_ms": round(avg_time * 1000, 2)
                }
        
        # Calculate uplift
        if stats.get("model_a", {}).get("total", 0) > 0:
            acc_a= stats["model_a"]['accuracy']
            acc_b = stats['model_b']['accuracy']
            uplift = ((acc_b - acc_a) / acc_a * 100) if acc_a > 0 else 0
            stats['uplift_percent'] = round(uplift, 2)

# Singleton
_ab_testing = None

def get_ab_testing_service() -> ABTestingService:
    global _ab_testing
    if _ab_testing is None:
        _ab_testing = ABTestingService()
    return _ab_testing