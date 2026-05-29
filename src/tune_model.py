from __future__ import annotations

import json
from typing import Any

import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from evaluation_utils import (
    BASE_DIR,
    B_MODEL_COMPARISON_RESULT_PATH,
    BEST_MODEL_PATH,
    BEST_PARAMS_PATH,
    DEFAULT_CLASSIFICATION_THRESHOLD,
    MODEL_RESULTS_PATH,
    RANDOM_STATE,
    ThresholdClassifier,
    TUNING_RESULTS_PATH,
    build_candidate_models,
    ensure_output_dirs,
    evaluate_predictions,
    evaluate_model,
    load_split,
    normalize_model_name,
    predict_probability,
    round_metrics,
)


DEFAULT_TUNING_MODELS = ["svm", "random_forest", "logistic_regression", "knn"]
TOP_MODELS_FROM_B = 2
SCORING = "roc_auc"
CV_SPLITS = 5
MIN_VALIDATION_RECALL = 0.70
MAX_F1_GENERALIZATION_GAP = 0.12


def _find_b_result_path() -> tuple[Any | None, str | None]:
    for path in [B_MODEL_COMPARISON_RESULT_PATH, MODEL_RESULTS_PATH]:
        if path.exists():
            return path, path.name
    return None, None


def _select_models_from_b_results(candidates: dict[str, dict[str, Any]]) -> list[str]:
    result_path, result_name = _find_b_result_path()
    if result_path is None:
        return []

    try:
        result_df = pd.read_csv(result_path, encoding="utf-8-sig")
    except Exception as exc:
        print(f"读取成员 B 结果失败，使用默认候选模型。原因: {exc}")
        return []

    if "model" not in result_df.columns:
        print("成员 B 结果表缺少 model 列，使用默认候选模型。")
        return []

    metric_column = None
    for candidate_metric in ["roc_auc", "f1", "recall", "accuracy"]:
        if candidate_metric in result_df.columns:
            metric_column = candidate_metric
            break

    if metric_column is None:
        print("成员 B 结果表缺少可排序指标列，使用默认候选模型。")
        return []

    split_column = None
    for candidate_column in ["split", "dataset"]:
        if candidate_column in result_df.columns:
            split_column = candidate_column
            break

    if split_column is not None:
        dataset_values = result_df[split_column].astype(str).str.lower()
        validation_df = result_df[dataset_values.isin(["validation", "val"])]
        if not validation_df.empty:
            result_df = validation_df

    rows = result_df.copy()
    rows["_model_key"] = rows["model"].map(normalize_model_name)
    rows = rows[rows["_model_key"].isin(candidates.keys())]
    if rows.empty:
        print("成员 B 结果表没有匹配到当前脚本支持的模型，使用默认候选模型。")
        return []

    selected = (
        rows.sort_values(metric_column, ascending=False)["_model_key"]
        .drop_duplicates()
        .head(TOP_MODELS_FROM_B)
        .tolist()
    )
    print(f"检测到成员 B 的 {result_name}，将优先调参: {selected}")
    return selected


def select_tuning_models(candidates: dict[str, dict[str, Any]]) -> list[str]:
    selected_from_b = _select_models_from_b_results(candidates)
    if selected_from_b:
        return selected_from_b

    return [model_name for model_name in DEFAULT_TUNING_MODELS if model_name in candidates]


def tune_single_model(
    model_key: str,
    model_info: dict[str, Any],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
) -> tuple[Any, dict[str, Any]]:
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        estimator=model_info["estimator"],
        param_grid=model_info["param_grid"],
        scoring=SCORING,
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=True,
    )
    search.fit(x_train, y_train)

    best_model = search.best_estimator_
    train_metrics = evaluate_model(best_model, x_train, y_train)
    validation_metrics = evaluate_model(best_model, x_val, y_val)
    best_threshold, threshold_validation_metrics = find_best_threshold(best_model, x_val, y_val)
    train_probabilities = predict_probability(best_model, x_train)
    threshold_train_metrics = evaluate_predictions(
        y_train,
        (train_probabilities >= best_threshold).astype(int),
        train_probabilities,
    )
    f1_generalization_gap = max(
        0.0,
        threshold_train_metrics["f1"] - threshold_validation_metrics["f1"],
    )

    row = {
        "model_key": model_key,
        "model": model_info["display_name"],
        "best_cv_roc_auc": float(search.best_score_),
        "classification_threshold": best_threshold,
        "f1_generalization_gap": f1_generalization_gap,
        "best_params": json.dumps(search.best_params_, ensure_ascii=False),
    }
    row.update({f"train_{key}": value for key, value in train_metrics.items()})
    row.update({f"validation_{key}": value for key, value in validation_metrics.items()})
    row.update({f"threshold_train_{key}": value for key, value in threshold_train_metrics.items()})
    row.update({f"threshold_validation_{key}": value for key, value in threshold_validation_metrics.items()})
    return best_model, row


def choose_best_tuned_model(results: list[dict[str, Any]]) -> dict[str, Any]:
    # 风险预测不只追求概率排序能力，也要控制漏判；用验证集阈值调整后的 F1/Recall/AUC 选模型。
    stable_results = [
        row for row in results if row["f1_generalization_gap"] <= MAX_F1_GENERALIZATION_GAP
    ]
    selection_pool = stable_results or results
    return max(
        selection_pool,
        key=lambda row: (
            row["threshold_validation_f1"],
            row["threshold_validation_recall"],
            row["threshold_validation_roc_auc"],
        ),
    )


def find_best_threshold(
    model: Any,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    min_recall: float = MIN_VALIDATION_RECALL,
) -> tuple[float, dict[str, float]]:
    probabilities = predict_probability(model, x_val)
    best_threshold = DEFAULT_CLASSIFICATION_THRESHOLD
    best_metrics = evaluate_predictions(y_val, (probabilities >= best_threshold).astype(int), probabilities)
    best_key = (
        best_metrics["recall"] >= min_recall,
        best_metrics["f1"],
        best_metrics["recall"],
        best_metrics["precision"],
    )

    for threshold in [round(value / 100, 2) for value in range(25, 76)]:
        predictions = (probabilities >= threshold).astype(int)
        metrics = evaluate_predictions(y_val, predictions, probabilities)
        key = (
            metrics["recall"] >= min_recall,
            metrics["f1"],
            metrics["recall"],
            metrics["precision"],
        )
        if key > best_key:
            best_threshold = threshold
            best_metrics = metrics
            best_key = key

    return best_threshold, best_metrics


def main() -> None:
    ensure_output_dirs()
    x_train, y_train = load_split("train_processed.csv")
    x_val, y_val = load_split("val_processed.csv")
    x_test, y_test = load_split("test_processed.csv")

    candidates = build_candidate_models()
    selected_model_keys = select_tuning_models(candidates)
    print(f"本次调参候选模型: {selected_model_keys}")

    tuned_models: dict[str, Any] = {}
    tuning_rows: list[dict[str, Any]] = []

    for model_key in selected_model_keys:
        print(f"开始调参: {candidates[model_key]['display_name']}")
        tuned_model, result_row = tune_single_model(
            model_key,
            candidates[model_key],
            x_train,
            y_train,
            x_val,
            y_val,
        )
        tuned_models[model_key] = tuned_model
        tuning_rows.append(result_row)
        print(
            f"{result_row['model']} 验证集 ROC AUC={result_row['validation_roc_auc']:.4f}, "
            f"F1={result_row['validation_f1']:.4f}"
        )

    best_row = choose_best_tuned_model(tuning_rows)
    selected_estimator = tuned_models[best_row["model_key"]]
    best_threshold = float(best_row["classification_threshold"])
    best_model = ThresholdClassifier(
        selected_estimator,
        threshold=best_threshold,
        model_key=best_row["model_key"],
        display_name=best_row["model"],
    )
    test_metrics = evaluate_model(best_model, x_test, y_test)
    best_row.update({f"test_{key}": value for key, value in test_metrics.items()})

    tuning_df = pd.DataFrame(tuning_rows)
    metric_columns = [column for column in tuning_df.columns if column.endswith(("accuracy", "precision", "recall", "f1", "roc_auc"))]
    tuning_df[metric_columns] = tuning_df[metric_columns].round(4)
    tuning_df.to_csv(TUNING_RESULTS_PATH, index=False, encoding="utf-8-sig")

    metadata = {
        "selected_model_key": best_row["model_key"],
        "selected_model": best_row["model"],
        "selection_rule": "threshold-adjusted validation f1, then recall, then roc_auc",
        "scoring": SCORING,
        "cv": f"StratifiedKFold(n_splits={CV_SPLITS}, shuffle=True, random_state={RANDOM_STATE})",
        "classification_threshold": best_threshold,
        "threshold_selection_rule": (
            f"choose threshold on validation set by f1, requiring recall >= {MIN_VALIDATION_RECALL} when possible"
        ),
        "generalization_guard": (
            f"prefer models with threshold train-validation f1 gap <= {MAX_F1_GENERALIZATION_GAP}"
        ),
        "f1_generalization_gap": round(float(best_row["f1_generalization_gap"]), 4),
        "used_b_model_results": _find_b_result_path()[0] is not None,
        "b_model_results_file": _find_b_result_path()[1],
        "candidate_models": selected_model_keys,
        "best_cv_roc_auc": round(float(best_row["best_cv_roc_auc"]), 4),
        "best_params": json.loads(best_row["best_params"]),
        "validation_metrics": round_metrics(
            {
                "accuracy": best_row["validation_accuracy"],
                "precision": best_row["validation_precision"],
                "recall": best_row["validation_recall"],
                "f1": best_row["validation_f1"],
                "roc_auc": best_row["validation_roc_auc"],
            }
        ),
        "threshold_validation_metrics": round_metrics(
            {
                "accuracy": best_row["threshold_validation_accuracy"],
                "precision": best_row["threshold_validation_precision"],
                "recall": best_row["threshold_validation_recall"],
                "f1": best_row["threshold_validation_f1"],
                "roc_auc": best_row["threshold_validation_roc_auc"],
            }
        ),
        "test_metrics": round_metrics(test_metrics),
        "data_files": {
            "train": str((BASE_DIR / "data" / "train_processed.csv").relative_to(BASE_DIR)),
            "validation": str((BASE_DIR / "data" / "val_processed.csv").relative_to(BASE_DIR)),
            "test": str((BASE_DIR / "data" / "test_processed.csv").relative_to(BASE_DIR)),
        },
    }

    joblib.dump(best_model, BEST_MODEL_PATH)
    BEST_PARAMS_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print("调参完成")
    print(f"最终模型: {metadata['selected_model']}")
    print(f"最佳参数: {metadata['best_params']}")
    print(f"验证集选择阈值: {best_threshold}")
    print(f"阈值调整后验证集指标: {metadata['threshold_validation_metrics']}")
    print(f"测试集指标: {metadata['test_metrics']}")
    print(f"调参结果已保存到: {TUNING_RESULTS_PATH}")
    print(f"最佳参数已保存到: {BEST_PARAMS_PATH}")
    print(f"最终模型已保存到: {BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()
