from typing import Tuple
import numpy as np
import pandas as pd
import pickle
import os
from .config import INSURANCE_MODEL_PATH
from .memory import get_patient_history

class InsuranceEstimator:
    def __init__(self):
        if not os.path.exists(INSURANCE_MODEL_PATH):
            raise FileNotFoundError(
                f"Insurance model not found at {INSURANCE_MODEL_PATH}. "
                f"Run python -m scripts.train_insurance_model first."
            )

        with open(INSURANCE_MODEL_PATH, "rb") as f:
            bundle = pickle.load(f)

        self.model = bundle["model"]
        self.encoders = bundle["encoders"]
        self.categorical_cols = bundle["categorical_cols"]
        self.numeric_cols = bundle["numeric_cols"]
        self.feature_order = bundle["feature_order"]

    def _build_feature_row(
        self,
        age: int,
        sex: str,
        bmi: float,
        children: int,
        smoker: str,
        region: str,
    ) -> pd.DataFrame:
        data = {
            "age": [age],
            "sex": [sex],
            "bmi": [bmi],
            "children": [children],
            "smoker": [smoker],
            "region": [region],
        }

        df = pd.DataFrame(data)

        for col in self.categorical_cols:
            le = self.encoders[col]
            df[col] = df[col].apply(
                lambda v: le.transform([v])[0] if v in le.classes_ else 0
            )

        return df[self.feature_order]

    def estimate(
        self,
        patient_id: str,
        age: int,
        sex: str,
        bmi: float,
        children: int,
        smoker: str,
        region: str,
    ) -> Tuple[float, str]:
        features = self._build_feature_row(age, sex, bmi, children, smoker, region)
        base_pred = float(self.model.predict(features)[0])

        history = get_patient_history(patient_id)
        history_prices = []
        if history and history.get("history"):
            for entry in history["history"]:
                price = entry.get("insurance_price")
                if price is not None:
                    history_prices.append(float(price))

        if history_prices:
            hist_avg = float(np.mean(history_prices))
            blended = 0.7 * base_pred + 0.3 * hist_avg
            explanation = (
                f"Estimated insurance cost: {blended:,.0f} "
                f"(70% current estimate + 30% historical average)."
            )
            return blended, explanation
        else:
            return (
                base_pred,
                f"Estimated insurance cost: {base_pred:,.0f} (no prior history)."
            )
