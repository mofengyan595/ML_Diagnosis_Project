from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
FIGURES_DIR = BASE_DIR / "figures"
MODELS_DIR = BASE_DIR / "models"
RESULT_PATH = BASE_DIR / "baseline_result.csv"
PREDICTION_PATH = BASE_DIR / "baseline_prediction_details.csv"
FEATURE_IMPORTANCE_PATH = BASE_DIR / "baseline_feature_importance.csv"

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


def load_split(filename: str) -> tuple[pd.DataFrame, pd.Series]:
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"未找到 {path}，请先运行 src/data_process.py")
    df = pd.read_csv(path, encoding="utf-8-sig")
    return df[FEATURE_COLUMNS], df[TARGET_COLUMN].astype(int)


def evaluate_model(model: LogisticRegression, x: pd.DataFrame, y: pd.Series, dataset_name: str) -> dict[str, float | str]:
    y_pred = model.predict(x)
    y_prob = model.predict_proba(x)[:, 1]

    return {
        "dataset": dataset_name,
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y, y_prob),
    }


def risk_level(probability: float) -> str:
    if probability < RISK_THRESHOLDS["low"]:
        return "低风险"
    if probability < RISK_THRESHOLDS["high"]:
        return "中风险"
    return "高风险"


def build_prediction_details(
    model: LogisticRegression,
    x: pd.DataFrame,
    y: pd.Series,
    dataset_name: str,
) -> pd.DataFrame:
    probabilities = model.predict_proba(x)[:, 1]
    predictions = model.predict(x)
    details = pd.DataFrame(
        {
            "dataset": dataset_name,
            "y_true": y.to_numpy(),
            "y_pred": predictions,
            "predicted_probability": probabilities,
        }
    )
    details["risk_level"] = details["predicted_probability"].map(risk_level)
    details["predicted_probability"] = details["predicted_probability"].round(4)
    return details


def save_feature_importance(model: LogisticRegression) -> None:
    coefficients = model.coef_[0]
    importance_df = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "feature_cn": [FEATURE_LABELS[feature] for feature in FEATURE_COLUMNS],
            "coefficient": coefficients,
        }
    )
    importance_df["abs_coefficient"] = importance_df["coefficient"].abs()
    importance_df["direction"] = importance_df["coefficient"].map(lambda value: "风险升高" if value > 0 else "风险降低")
    importance_df = importance_df.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)
    importance_df["rank"] = importance_df.index + 1
    importance_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False, encoding="utf-8-sig")

    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    sns.set_theme(style="whitegrid", font="Microsoft YaHei")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    plot_df = importance_df.sort_values("abs_coefficient", ascending=True)
    sns.barplot(data=plot_df, x="abs_coefficient", y="feature_cn", hue="feature_cn", palette="Blues_r", legend=False, ax=ax)
    ax.set_title("Logistic Regression 基准模型特征影响排序", fontsize=14, weight="bold")
    ax.set_xlabel("标准化特征系数绝对值")
    ax.set_ylabel("特征")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "baseline_feature_importance.png", dpi=200)
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    x_train, y_train = load_split("train_processed.csv")
    x_val, y_val = load_split("val_processed.csv")
    x_test, y_test = load_split("test_processed.csv")

    # class_weight="balanced" 用于缓解糖尿病阳性样本少于阴性样本的问题。
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    model.fit(x_train, y_train)

    results = [
        evaluate_model(model, x_val, y_val, "validation"),
        evaluate_model(model, x_test, y_test, "test"),
    ]
    result_df = pd.DataFrame(results)
    metric_columns = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    result_df[metric_columns] = result_df[metric_columns].round(4)

    result_df.to_csv(RESULT_PATH, index=False, encoding="utf-8-sig")
    prediction_details = pd.concat(
        [
            build_prediction_details(model, x_val, y_val, "validation"),
            build_prediction_details(model, x_test, y_test, "test"),
        ],
        ignore_index=True,
    )
    prediction_details.to_csv(PREDICTION_PATH, index=False, encoding="utf-8-sig")
    save_feature_importance(model)
    joblib.dump(model, MODELS_DIR / "baseline_logistic_regression.pkl")

    print("Logistic Regression 基准模型训练完成")
    print(result_df.to_string(index=False))
    print(f"基准模型已保存到: {MODELS_DIR / 'baseline_logistic_regression.pkl'}")
    print(f"结果表已保存到: {RESULT_PATH}")
    print(f"预测概率与风险等级明细已保存到: {PREDICTION_PATH}")
    print(f"特征影响排序已保存到: {FEATURE_IMPORTANCE_PATH}")


if __name__ == "__main__":
    main()
