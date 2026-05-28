from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


TARGET_COLUMN = "Outcome"
RANDOM_STATE = 42
BASELINE_RESULT_FILENAME = "baseline_result.csv"
RESULT_FILENAME = "model_comparison_result.csv"
TRAINED_MODELS_FILENAME = "trained_models.pkl"
RESULT_COLUMNS = ["model", "split", "accuracy", "precision", "recall", "f1", "roc_auc"]
METRIC_COLUMNS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def get_project_root() -> Path:
    """Return project root when this script is run from the repository root."""
    return Path(__file__).resolve().parents[1]


def ensure_output_dirs() -> None:
    root = get_project_root()
    for dirname in ["models", "figures", "report"]:
        (root / dirname).mkdir(parents=True, exist_ok=True)


def load_split_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = get_project_root()
    data_dir = root / "data"

    train_df = pd.read_csv(data_dir / "train_processed.csv", encoding="utf-8-sig")
    val_df = pd.read_csv(data_dir / "val_processed.csv", encoding="utf-8-sig")
    test_df = pd.read_csv(data_dir / "test_processed.csv", encoding="utf-8-sig")

    for dataset_name, df in [
        ("train", train_df),
        ("validation", val_df),
        ("test", test_df),
    ]:
        print(f"{dataset_name} shape: {df.shape}")
        print(f"{dataset_name} columns: {list(df.columns)}")

    return train_df, val_df, test_df


def split_features_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"数据集中缺少标签列: {TARGET_COLUMN}")

    x = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)
    return x, y


def load_baseline_results() -> pd.DataFrame:
    baseline_path = get_project_root() / BASELINE_RESULT_FILENAME
    required_columns = ["dataset", *METRIC_COLUMNS]

    try:
        baseline_df = pd.read_csv(baseline_path, encoding="utf-8-sig")
        missing_columns = [column for column in required_columns if column not in baseline_df.columns]
        if missing_columns:
            raise ValueError(f"missing columns: {missing_columns}")

        baseline_df = baseline_df[required_columns].rename(columns={"dataset": "split"})
        baseline_df["model"] = "Logistic Regression"
        baseline_df["split"] = baseline_df["split"].replace({"val": "validation"})
        baseline_df[METRIC_COLUMNS] = baseline_df[METRIC_COLUMNS].round(4)
        return baseline_df[RESULT_COLUMNS]
    except Exception as exc:
        warnings.warn(
            f"无法合并 baseline_result.csv，原因: {exc}。将只保存成员 B 的四个模型结果。",
            RuntimeWarning,
            stacklevel=2,
        )
        return pd.DataFrame(columns=RESULT_COLUMNS)


def build_models() -> dict[str, object]:
    return {
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_estimators=100,
        ),
        "SVM": SVC(probability=True, random_state=RANDOM_STATE, class_weight="balanced"),
    }


def evaluate_model(model: object, x: pd.DataFrame, y: pd.Series, dataset_name: str) -> dict[str, object]:
    y_pred = model.predict(x)

    roc_auc = np.nan
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(x)[:, 1]
        roc_auc = roc_auc_score(y, y_score)
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(x)
        if getattr(y_score, "ndim", 1) > 1:
            y_score = y_score[:, 1]
        roc_auc = roc_auc_score(y, y_score)
    else:
        warnings.warn(
            f"{model.__class__.__name__} does not provide predict_proba or decision_function; "
            f"roc_auc is set to NaN for {dataset_name}.",
            RuntimeWarning,
            stacklevel=2,
        )

    return {
        "split": dataset_name,
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "roc_auc": roc_auc,
    }


def train_and_evaluate_models(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[dict[str, object], pd.DataFrame]:
    models = build_models()
    results = []
    trained_models: dict[str, object] = {}

    for model_name, model in models.items():
        print(f"Training {model_name}...")
        model.fit(x_train, y_train)
        trained_models[model_name] = model

        for split_name, x_split, y_split in [
            ("validation", x_val, y_val),
            ("test", x_test, y_test),
        ]:
            metrics = evaluate_model(model, x_split, y_split, split_name)
            results.append({"model": model_name, **metrics})

    results_df = pd.DataFrame(results, columns=RESULT_COLUMNS)
    results_df[METRIC_COLUMNS] = results_df[METRIC_COLUMNS].round(4)
    return trained_models, results_df


def save_trained_models(models: dict[str, object]) -> None:
    model_path = get_project_root() / "models" / TRAINED_MODELS_FILENAME
    joblib.dump(models, model_path)
    print(f"候选模型字典已保存到: {model_path}")


def print_split_summary(dataset_name: str, x: pd.DataFrame, y: pd.Series) -> None:
    print(f"{dataset_name} samples: {len(y)}")
    print(f"{dataset_name} features: {x.shape[1]}")
    print(f"{dataset_name} class distribution:")
    print(y.value_counts().sort_index().to_string())


def main() -> None:
    ensure_output_dirs()

    train_df, val_df, test_df = load_split_data()

    x_train, y_train = split_features_labels(train_df)
    x_val, y_val = split_features_labels(val_df)
    x_test, y_test = split_features_labels(test_df)

    print_split_summary("train", x_train, y_train)
    print_split_summary("validation", x_val, y_val)
    print_split_summary("test", x_test, y_test)

    trained_models, results_df = train_and_evaluate_models(
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
    )
    save_trained_models(trained_models)

    baseline_results_df = load_baseline_results()
    if not baseline_results_df.empty:
        results_df = pd.concat([baseline_results_df, results_df], ignore_index=True)

    print("模型对比结果:")
    print(results_df.to_string(index=False))

    result_path = get_project_root() / RESULT_FILENAME
    results_df.to_csv(result_path, index=False, encoding="utf-8-sig")
    print(f"模型对比结果已保存到: {result_path}")
    print(f"已训练并保存候选模型数量: {len(trained_models)}")


if __name__ == "__main__":
    main()
