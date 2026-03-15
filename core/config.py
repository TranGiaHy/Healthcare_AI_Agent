import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
MEDICAL_KB_PATH = os.path.join(DATA_DIR, "medical_knowledge.csv")
DOCTORS_PATH = os.path.join(DATA_DIR, "doctors.csv")
INSURANCE_RAW_PATH = os.path.join(DATA_DIR, "insurance.csv")
INSURANCE_MODEL_PATH = os.path.join(MODELS_DIR, "insurance_model.pkl")
SYMPTOM_INDEX_PATH = os.path.join(MODELS_DIR, "symptom_index.faiss")
SYMPTOM_EMBEDDINGS_PATH = os.path.join(MODELS_DIR, "symptom_embeddings.npy")
PATIENT_MEMORY_PATH = os.path.join(STORAGE_DIR, "patient_memory.json")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K_DIAGNOSES = 5
SAFETY_DISCLAIMER = (
    "This system is for informational purposes only and does not provide "
    "a medical diagnosis. Always consult a licensed healthcare professional."
)
SECRETS_PATH = os.path.join(BASE_DIR, "secrets.json")
