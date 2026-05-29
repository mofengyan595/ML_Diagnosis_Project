from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
FIGURES_DIR = BASE_DIR / "figures"
MODELS_DIR = BASE_DIR / "models"

MODEL_RESULTS_PATH = BASE_DIR / "model_results.csv"
B_MODEL_COMPARISON_RESULT_PATH = BASE_DIR / "model_comparison_result.csv"
TUNING_RESULTS_PATH = BASE_DIR / "tuning_results.csv"
BEST_PARAMS_PATH = BASE_DIR / "best_params.json"
FINAL_METRICS_PATH = BASE_DIR / "final_metrics.csv"
FINAL_PREDICTION_DETAILS_PATH = BASE_DIR / "final_prediction_details.csv"
ERROR_ANALYSIS_PATH = BASE_DIR / "error_analysis.csv"
FEATURE_IMPORTANCE_PATH = BASE_DIR / "feature_importance.csv"

BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"

CONFUSION_MATRIX_PATH = FIGURES_DIR / "confusion_matrix.png"
ROC_CURVE_PATH = FIGURES_DIR / "roc_curve.png"
FEATURE_IMPORTANCE_FIGURE_PATH = FIGURES_DIR / "feature_importance.png"

FEATURE_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
TARGET_COLUMN = "Outcome"
RANDOM_STATE = 42
DEFAULT_CLASSIFICATION_THRESHOLD = 0.5

RISK_THRESHOLDS = {
    "low": 0.35,
    "high": 0.65,
}

FEATURE_LABELS = {
    "Pregnancies": "怀孕次数",
    "Glucose": "血糖值",
    "BloodPressure": "血压",
    "SkinThickness": "皮肤厚度",
    "Insulin": "胰岛素",
    "BMI": "BMI",
    "DiabetesPedigreeFunction": "糖尿病家族遗传指数",
    "Age": "年龄",
}


def ensure_output_dirs() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_split(filename: str) -> tuple[pd.DataFrame, pd.Series]:
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"未找到 {path}，请先运行 src/data_process.py")

    df = pd.read_csv(path, encoding="utf-8-sig")
    missing_columns = [column for column in FEATURE_COLUMNS + [TARGET_COLUMN] if column not in df.columns]
    if missing_columns:
        raise ValueError(f"{path} 缺少必要字段: {missing_columns}")

    return df[FEATURE_COLUMNS].copy(), df[TARGET_COLUMN].astype(int)


def predict_probability(model: Any, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(x)[:, 1], dtype=float)

    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(x), dtype=float)
        min_score = float(np.min(scores))
        max_score = float(np.max(scores))
        if max_score == min_score:
            return np.full_like(scores, 0.5, dtype=float)
        return (scores - min_score) / (max_score - min_score)

    predictions = np.asarray(model.predict(x), dtype=float)
    return np.clip(predictions, 0.0, 1.0)


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


def evaluate_model(model: Any, x: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    y_pred = np.asarray(model.predict(x), dtype=int)
    y_prob = predict_probability(model, x)
    return evaluate_predictions(y, y_pred, y_prob)


class ThresholdClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        estimator: Any,
        threshold: float = DEFAULT_CLASSIFICATION_THRESHOLD,
        model_key: str | None = None,
        display_name: str | None = None,
    ) -> None:
        self.estimator = estimator
        self.threshold = threshold
        self.model_key = model_key
        self.display_name = display_name
        self.classes_ = getattr(estimator, "classes_", np.array([0, 1]))

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "ThresholdClassifier":
        self.estimator.fit(x, y)
        self.classes_ = getattr(self.estimator, "classes_", np.array([0, 1]))
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        if hasattr(self.estimator, "predict_proba"):
            return self.estimator.predict_proba(x)

        probability = predict_probability(self.estimator, x)
        return np.column_stack([1 - probability, probability])

    def decision_function(self, x: pd.DataFrame) -> np.ndarray:
        if hasattr(self.estimator, "decision_function"):
            return self.estimator.decision_function(x)
        return self.predict_proba(x)[:, 1]

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        probabilities = self.predict_proba(x)[:, 1]
        return (probabilities >= self.threshold).astype(int)


def risk_level(probability: float) -> str:
    if probability < RISK_THRESHOLDS["low"]:
        return "低风险"
    if probability < RISK_THRESHOLDS["high"]:
        return "中风险"
    return "高风险"


def round_metrics(metrics: dict[str, float], digits: int = 4) -> dict[str, float]:
    return {key: round(float(value), digits) for key, value in metrics.items()}


def build_candidate_models() -> dict[str, dict[str, Any]]:
    return {
        "logistic_regression": {
            "display_name": "Logistic Regression",
            "estimator": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            "param_grid": {
                "C": [0.01, 0.1, 1.0, 10.0, 100.0],
                "penalty": ["l2"],
                "solver": ["liblinear", "lbfgs"],
                "class_weight": [None, "balanced"],
            },
        },
        "decision_tree": {
            "display_name": "Decision Tree",
            "estimator": DecisionTreeClassifier(random_state=RANDOM_STATE),
            "param_grid": {
                "max_depth": [3, 5, 7, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "class_weight": [None, "balanced"],
            },
        },
        "random_forest": {
            "display_name": "Random Forest",
            "estimator": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            "param_grid": {
                "n_estimators": [100, 200, 300],
                "max_depth": [None, 3, 5, 7, 10],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "class_weight": [None, "balanced"],
            },
        },
        "svm": {
            "display_name": "SVM",
            "estimator": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
            "param_grid": {
                "C": [0.1, 1.0, 10.0, 100.0],
                "gamma": ["scale", 0.01, 0.1, 1.0],
                "class_weight": [None, "balanced"],
            },
        },
        "knn": {
            "display_name": "KNN",
            "estimator": KNeighborsClassifier(),
            "param_grid": {
                "n_neighbors": [3, 5, 7, 9, 11, 15],
                "weights": ["uniform", "distance"],
                "metric": ["euclidean", "manhattan"],
            },
        },
    }


MODEL_ALIASES = {
    "logisticregression": "logistic_regression",
    "logisticregressionclassifier": "logistic_regression",
    "logistic_regression": "logistic_regression",
    "logistic regression": "logistic_regression",
    "lr": "logistic_regression",
    "baseline": "logistic_regression",
    "decisiontree": "decision_tree",
    "decisiontreeclassifier": "decision_tree",
    "decision_tree": "decision_tree",
    "decision tree": "decision_tree",
    "dt": "decision_tree",
    "randomforest": "random_forest",
    "randomforestclassifier": "random_forest",
    "random_forest": "random_forest",
    "random forest": "random_forest",
    "rf": "random_forest",
    "svm": "svm",
    "svc": "svm",
    "supportvectormachine": "svm",
    "support vector machine": "svm",
    "knn": "knn",
    "knearestneighbors": "knn",
    "kneighborsclassifier": "knn",
    "k_nearest_neighbors": "knn",
    "k-nearest neighbors": "knn",
}


def normalize_model_name(name: Any) -> str | None:
    normalized = str(name).strip().lower()
    compact = normalized.replace("-", "").replace("_", "").replace(" ", "")
    return MODEL_ALIASES.get(normalized) or MODEL_ALIASES.get(compact)
