import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from core.config import INSURANCE_RAW_PATH, INSURANCE_MODEL_PATH, MODELS_DIR

def train():
    if not os.path.exists(INSURANCE_RAW_PATH):
        raise FileNotFoundError(f"Insurance CSV not found at {INSURANCE_RAW_PATH}")
    df = pd.read_csv(INSURANCE_RAW_PATH)
    required_cols = {"age", "sex", "bmi", "children", "smoker", "region", "charges"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in insurance.csv: {missing}")
    
    X = df[["age", "sex", "bmi", "children", "smoker", "region"]].copy()
    y = df["charges"].values
    categorical_cols = ["sex", "smoker", "region"]
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    print(f"R^2 on test set: {model.score(X_test, y_test):.4f}")
    
    bundle = {"model": model, "encoders": encoders, "categorical_cols": categorical_cols, "numeric_cols": ["age", "bmi", "children"], "feature_order": ["age", "sex", "bmi", "children", "smoker", "region"]}
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(INSURANCE_MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)
    print("Insurance model saved successfully.")

if __name__ == "__main__":
    train()
