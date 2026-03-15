import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import os
from core.config import (
    MEDICAL_KB_PATH,
    SYMPTOM_INDEX_PATH,
    SYMPTOM_EMBEDDINGS_PATH,
    EMBEDDING_MODEL_NAME,
)

def build_index():
    if not os.path.exists(MEDICAL_KB_PATH):
        raise FileNotFoundError(f"Medical knowledge file not found at {MEDICAL_KB_PATH}")
    df = pd.read_csv(MEDICAL_KB_PATH)
    texts = df["symptom_pattern"].astype(str).tolist()
    print(f"Loaded {len(texts)} symptom patterns")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("Encoding symptom patterns...")
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    os.makedirs(os.path.dirname(SYMPTOM_INDEX_PATH), exist_ok=True)
    faiss.write_index(index, SYMPTOM_INDEX_PATH)
    np.save(SYMPTOM_EMBEDDINGS_PATH, embeddings)
    print("Knowledge index built successfully.")

if __name__ == "__main__":
    build_index()
