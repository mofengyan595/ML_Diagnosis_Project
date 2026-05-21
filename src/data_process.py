from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "diabetes.csv"
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

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
EXPECTED_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]

INVALID_ZERO_COLUMNS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
RANDOM_STATE = 42
TEST_SIZE = 0.20
VAL_SIZE_IN_TRAIN_VAL = 0.20


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_raw_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, encoding="utf-8")
    missing_columns = [column for column in EXPECTED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"原始数据缺少必要字段: {missing_columns}")
    return df[EXPECTED_COLUMNS].copy()


def mark_invalid_zero_as_missing(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    # 医学上 Glucose/BloodPressure/SkinThickness/Insulin/BMI 不应为 0，按缺失值处理。
    cleaned[INVALID_ZERO_COLUMNS] = cleaned[INVALID_ZERO_COLUMNS].replace(0, np.nan)
    return cleaned


def stratified_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_val_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[TARGET_COLUMN],
    )
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=VAL_SIZE_IN_TRAIN_VAL,
        random_state=RANDOM_STATE,
        stratify=train_val_df[TARGET_COLUMN],
    )
    return train_df, val_df, test_df


def fit_iqr_bounds(train_features: pd.DataFrame) -> dict[str, tuple[float, float]]:
    bounds: dict[str, tuple[float, float]] = {}
    for column in FEATURE_COLUMNS:
        q1 = train_features[column].quantile(0.25)
        q3 = train_features[column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        # 这些字段天然不应小于 0，裁剪下界不能低于 0。
        if column in FEATURE_COLUMNS:
            lower = max(lower, 0.0)
        bounds[column] = (float(lower), float(upper))
    return bounds


def apply_iqr_clipping(features: pd.DataFrame, bounds: dict[str, tuple[float, float]]) -> pd.DataFrame:
    clipped = features.copy()
    for column, (lower, upper) in bounds.items():
        clipped[column] = clipped[column].clip(lower=lower, upper=upper)
    return clipped


def transform_split(
    df: pd.DataFrame,
    imputer: SimpleImputer,
    bounds: dict[str, tuple[float, float]],
    scaler: StandardScaler | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = df[FEATURE_COLUMNS]
    target = df[TARGET_COLUMN].astype(int).reset_index(drop=True)

    imputed_array = imputer.transform(features)
    imputed_features = pd.DataFrame(imputed_array, columns=FEATURE_COLUMNS)
    clipped_features = apply_iqr_clipping(imputed_features, bounds)

    cleaned_df = clipped_features.copy()
    cleaned_df[TARGET_COLUMN] = target

    if scaler is None:
        processed_features = clipped_features.copy()
    else:
        # scaler 使用数组拟合/转换，避免系统预测时因为列名不同触发 sklearn 特征名错误。
        processed_array = scaler.transform(clipped_features.to_numpy())
        processed_features = pd.DataFrame(processed_array, columns=FEATURE_COLUMNS)

    processed_df = processed_features.copy()
    processed_df[TARGET_COLUMN] = target
    return cleaned_df, processed_df


def main() -> None:
    ensure_dirs()
    raw_df = load_raw_data()
    missing_marked_df = mark_invalid_zero_as_missing(raw_df)

    train_df, val_df, test_df = stratified_split(missing_marked_df)

    imputer = SimpleImputer(strategy="median")
    imputer.fit(train_df[FEATURE_COLUMNS])

    train_imputed = pd.DataFrame(imputer.transform(train_df[FEATURE_COLUMNS]), columns=FEATURE_COLUMNS)
    bounds = fit_iqr_bounds(train_imputed)
    train_clipped = apply_iqr_clipping(train_imputed, bounds)

    scaler = StandardScaler()
    scaler.fit(train_clipped.to_numpy())

    cleaned_train, processed_train = transform_split(train_df, imputer, bounds, scaler)
    cleaned_val, processed_val = transform_split(val_df, imputer, bounds, scaler)
    cleaned_test, processed_test = transform_split(test_df, imputer, bounds, scaler)

    cleaned_all = pd.concat([cleaned_train, cleaned_val, cleaned_test], ignore_index=True)
    processed_all = pd.concat([processed_train, processed_val, processed_test], ignore_index=True)

    cleaned_all.to_csv(DATA_DIR / "cleaned_data.csv", index=False, encoding="utf-8-sig")
    processed_all.to_csv(DATA_DIR / "processed_data.csv", index=False, encoding="utf-8-sig")
    processed_train.to_csv(DATA_DIR / "train_processed.csv", index=False, encoding="utf-8-sig")
    processed_val.to_csv(DATA_DIR / "val_processed.csv", index=False, encoding="utf-8-sig")
    processed_test.to_csv(DATA_DIR / "test_processed.csv", index=False, encoding="utf-8-sig")

    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    joblib.dump(imputer, MODELS_DIR / "imputer.pkl")
    joblib.dump(bounds, MODELS_DIR / "iqr_bounds.pkl")

    print("数据预处理完成")
    print(f"训练集: {processed_train.shape}")
    print(f"验证集: {processed_val.shape}")
    print(f"测试集: {processed_test.shape}")
    print(f"处理后数据缺失值数量: {int(processed_all.isna().sum().sum())}")
    print(f"scaler 已保存到: {MODELS_DIR / 'scaler.pkl'}")


if __name__ == "__main__":
    main()
