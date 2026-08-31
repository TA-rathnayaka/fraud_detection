import json
import os

import numpy as np
import shap
import xgboost as xgb


class FraudModel:
    """Loads the artifact notebook 04/06 saved to models/ -- no MLflow server dependency
    at serving time, so the API doesn't take a hard runtime dependency on the tracking
    server staying up."""

    def __init__(self, model_dir: str):
        self.model = xgb.XGBClassifier()
        self.model.load_model(os.path.join(model_dir, "fraud_xgboost.json"))

        with open(os.path.join(model_dir, "feature_columns.json")) as f:
            cfg = json.load(f)
        self.features: list[str] = cfg["features"]
        self.model_version = str(cfg.get("mlflow_model_version", "unknown"))

        with open(os.path.join(model_dir, "threshold.json")) as f:
            threshold_cfg = json.load(f)
        self.threshold: float = threshold_cfg["decision_threshold"]
        self.category_thresholds: dict[str, float] = threshold_cfg.get("category_thresholds", {})

        self.explainer = shap.TreeExplainer(self.model)

    def score(self, feature_dict: dict, category: str | None = None, top_k: int = 5):
        x = np.array([[feature_dict[f] for f in self.features]], dtype=float)
        proba = float(self.model.predict_proba(x)[0, 1])
        threshold = self.category_thresholds.get(category, self.threshold)
        decision = "decline_review" if proba >= threshold else "approve"

        shap_values = self.explainer(x)
        contributions = shap_values.values[0]
        order = np.argsort(-np.abs(contributions))[:top_k]
        reasons = [
            {
                "feature": self.features[i],
                "value": float(x[0, i]),
                "shap_contribution": float(contributions[i]),
            }
            for i in order
        ]
        return proba, decision, threshold, reasons
