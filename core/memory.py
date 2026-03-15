import json
import os
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional, List
from .config import PATIENT_MEMORY_PATH, STORAGE_DIR

def generate_patient_id(name: str, email: str) -> str:
    key = f"{name.strip().lower()}|{email.strip().lower()}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]

def load_memory() -> Dict[str, Any]:
    os.makedirs(STORAGE_DIR, exist_ok=True)
    if not os.path.exists(PATIENT_MEMORY_PATH):
        return {}
    with open(PATIENT_MEMORY_PATH, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}

def save_memory(data: Dict[str, Any]) -> None:
    os.makedirs(STORAGE_DIR, exist_ok=True)
    with open(PATIENT_MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_patient_history(patient_id: str) -> Optional[Dict[str, Any]]:
    memory = load_memory()
    return memory.get(patient_id)

def update_patient_memory(
    patient_id: str,
    demographics: Dict[str, Any],
    symptoms: str,
    diagnosis_result: Dict[str, Any],
    insurance_price: Optional[float],
    doctors: List[Dict[str, Any]],
) -> None:
    memory = load_memory()
    if patient_id not in memory:
        memory[patient_id] = {
            "demographics": demographics,
            "history": []
        }
    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "symptoms": symptoms,
        "diagnosis": diagnosis_result,
        "insurance_price": insurance_price,
        "doctors": doctors,
    }
    memory[patient_id]["history"].append(entry)
    save_memory(memory)
