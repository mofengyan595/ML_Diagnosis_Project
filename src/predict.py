from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from config import FEATURE_LABELS, FEATURES, MODEL_FEATURES, MODEL_PATH, RISK_THRESHOLDS, SCALER_PATH


def _load_artifact(path: Path) -> Any | None:
    if not path.exists():
        return None
    return joblib.load(path)


def _build_input_frame(values: dict[str, float]) -> pd.DataFrame:
    row = {model_name: values[local_name] for local_name, model_name in zip(FEATURES, MODEL_FEATURES)}
    return pd.DataFrame([row], columns=MODEL_FEATURES)


def _predict_with_model(values: dict[str, float]) -> tuple[float, str] | None:
    model = _load_artifact(MODEL_PATH)
    if model is None:
        return None

    input_frame = _build_input_frame(values)
    scaler = _load_artifact(SCALER_PATH)
    model_input = scaler.transform(input_frame) if scaler is not None else input_frame

    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(model_input)[0][1])
    else:
        prediction = float(model.predict(model_input)[0])
        probability = prediction

    return max(0.0, min(1.0, probability)), "已加载 models/best_model.pkl 进行预测。"


def _predict_with_rule(values: dict[str, float]) -> tuple[float, str]:
    # Temporary fallback for the UI demo before teammates provide the final model.
    score = -4.2
    score += 0.055 * (values["glucose"] - 100)
    score += 0.075 * (values["bmi"] - 24)
    score += 0.025 * (values["age"] - 35)
    score += 0.012 * (values["blood_pressure"] - 80)
    score += 0.65 * values["diabetes_pedigree"]
    score += 0.08 * values["pregnancies"]
    score += 0.003 * max(values["insulin"] - 80, 0)
    score += 0.01 * max(values["skin_thickness"] - 20, 0)

    probability = 1 / (1 + math.exp(-score))
    return probability, "当前未检测到正式模型，结果由演示用规则函数生成。"


def _risk_level(probability: float) -> str:
    if probability < RISK_THRESHOLDS["low"]:
        return "低风险"
    if probability < RISK_THRESHOLDS["high"]:
        return "中风险"
    return "高风险"


def _key_factors(values: dict[str, float]) -> list[str]:
    factors = []
    if values["glucose"] >= 140:
        factors.append(f'{FEATURE_LABELS["glucose"]}偏高')
    if values["bmi"] >= 28:
        factors.append("BMI 偏高")
    if values["age"] >= 50:
        factors.append("年龄较高")
    if values["blood_pressure"] >= 90:
        factors.append(f'{FEATURE_LABELS["blood_pressure"]}偏高')
    if values["diabetes_pedigree"] >= 0.8:
        factors.append(f'{FEATURE_LABELS["diabetes_pedigree"]}较高')
    return factors or ["当前输入中没有特别突出的高风险指标"]


def predict_diabetes_risk(
    pregnancies: float,
    glucose: float,
    blood_pressure: float,
    skin_thickness: float,
    insulin: float,
    bmi: float,
    diabetes_pedigree: float,
    age: float,
) -> tuple[str, str, str]:
    values = {
        "pregnancies": pregnancies,
        "glucose": glucose,
        "blood_pressure": blood_pressure,
        "skin_thickness": skin_thickness,
        "insulin": insulin,
        "bmi": bmi,
        "diabetes_pedigree": diabetes_pedigree,
        "age": age,
    }

    model_result = _predict_with_model(values)
    probability, source_note = model_result or _predict_with_rule(values)
    probability = max(0.0, min(1.0, probability))
    level = _risk_level(probability)
    factors = "、".join(_key_factors(values))

    probability_text = f"{probability * 100:.1f}%"
    explanation = (
        f"风险等级：{level}\n\n"
        f"主要参考因素：{factors}\n\n"
        f"{source_note}\n\n"
        "说明：本系统仅用于机器学习课程项目演示，不能替代医生诊断。"
    )
    return level, probability_text, explanation
