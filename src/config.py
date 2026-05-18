from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"


FEATURES = [
    "pregnancies",
    "glucose",
    "blood_pressure",
    "skin_thickness",
    "insulin",
    "bmi",
    "diabetes_pedigree",
    "age",
]


FEATURE_LABELS = {
    "pregnancies": "怀孕次数",
    "glucose": "血糖值",
    "blood_pressure": "血压",
    "skin_thickness": "皮肤厚度",
    "insulin": "胰岛素",
    "bmi": "BMI",
    "diabetes_pedigree": "糖尿病家族遗传指数",
    "age": "年龄",
}


RISK_THRESHOLDS = {
    "low": 0.35,
    "high": 0.65,
}
