from __future__ import annotations

from pathlib import Path

import pandas as pd


TARGET_COLUMN = "Outcome"


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


def build_models() -> dict[str, object]:
    # TODO: Add KNN, Decision Tree, Random Forest, SVM and other comparison models.
    return {}


def evaluate_model(model: object, x: pd.DataFrame, y: pd.Series, dataset_name: str) -> dict[str, object]:
    # TODO: Evaluate a trained model with accuracy, precision, recall, F1 and ROC AUC.
    raise NotImplementedError("模型评估逻辑将在下一步实现")


def train_and_evaluate_models(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[dict[str, object], pd.DataFrame]:
    # TODO: Train candidate models and collect validation/test metrics.
    models = build_models()
    results_df = pd.DataFrame()
    return models, results_df


def save_trained_models(models: dict[str, object]) -> None:
    # TODO: Save candidate model artifacts after model training is implemented.
    return None


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

    print("多模型训练模块骨架已就绪；本阶段未训练模型，未生成任何输出文件。")


if __name__ == "__main__":
    main()
