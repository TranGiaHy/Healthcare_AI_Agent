from typing import Dict, Any, Optional, List
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from .config import (
    MEDICAL_KB_PATH,
    SYMPTOM_INDEX_PATH,
    EMBEDDING_MODEL_NAME,
    TOP_K_DIAGNOSES,
    SAFETY_DISCLAIMER,
)
from .memory import get_patient_history

class SymptomDiagnosticEngine:
    def __init__(self):
        self._df = pd.read_csv(MEDICAL_KB_PATH)
        self._model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self._index = faiss.read_index(SYMPTOM_INDEX_PATH)

    def diagnose(self, symptoms: str, patient_id: Optional[str] = None) -> Dict[str, Any]:
        query_vec = self._model.encode([symptoms]).astype("float32")
        distances, indices = self._index.search(query_vec, TOP_K_DIAGNOSES)
        
        distances = distances[0]
        indices = indices[0]
        matches: List[Dict[str, Any]] = []

        for dist, idx in zip(distances, indices):
            if idx < 0 or idx >= len(self._df):
                continue
            row = self._df.iloc[idx]
            confidence = float(1.0 / (1.0 + dist))
            matches.append({
                "id": int(row["id"]),
                "symptom_pattern": str(row["symptom_pattern"]),
                "diagnosis": str(row["diagnosis"]),
                "specialty": str(row["specialty"]),
                "triage_level": str(row["triage_level"]),
                "red_flags": str(row["red_flags"]),
                "recommended_tests": str(row["recommended_tests"]),
                "advice": str(row["advice"]),
                "icd_code": str(row.get("icd_code", "")),
                "distance": float(dist),
                "confidence": confidence,
            })

        matches_sorted = sorted(matches, key=lambda m: m["confidence"], reverse=True)
        top_diagnosis = matches_sorted[0] if matches_sorted else None
        
        history_summary = ""
        if patient_id:
            history = get_patient_history(patient_id)
            if history and history.get("history"):
                visits = history["history"]
                history_summary = f"This patient has {len(visits)} previous recorded visit(s)."

        return {
            "top_diagnosis": top_diagnosis,
            "matches": matches_sorted,
            "history_summary": history_summary,
            "safety_disclaimer": SAFETY_DISCLAIMER,
        }
