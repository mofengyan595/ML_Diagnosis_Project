from __future__ import annotations

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix, roc_curve

from evaluation_utils import (
    BEST_MODEL_PATH,
    CONFUSION_MATRIX_PATH,
    ERROR_ANALYSIS_PATH,
    FEATURE_COLUMNS,
    FINAL_METRICS_PATH,
    FINAL_PREDICTION_DETAILS_PATH,
    ROC_CURVE_PATH,
    ensure_output_dirs,
    evaluate_model,
    load_split,
    predict_probability,
    risk_level,
)


def configure_plots() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    sns.set_theme(style="whitegrid", font="Microsoft YaHei")


def load_best_model():
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"未找到 {BEST_MODEL_PATH}，请先运行 src/tune_model.py")
    return joblib.load(BEST_MODEL_PATH)


def build_metrics_table(model) -> pd.DataFrame:
    rows = []
    for filename, dataset_name in [
        ("val_processed.csv", "validation"),
        ("test_processed.csv", "test"),
    ]:
        x, y = load_split(filename)
        metrics = evaluate_model(model, x, y)
        row = {"dataset": dataset_name}
        row.update(metrics)
        rows.append(row)

    metrics_df = pd.DataFrame(rows)
    metric_columns = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    metrics_df[metric_columns] = metrics_df[metric_columns].round(4)
    return metrics_df


def build_prediction_details(model) -> pd.DataFrame:
    detail_frames = []
    for filename, dataset_name in [
        ("val_processed.csv", "validation"),
        ("test_processed.csv", "test"),
    ]:
        x, y = load_split(filename)
        y_pred = model.predict(x).astype(int)
        y_prob = predict_probability(model, x)
        details = x.copy()
        details.insert(0, "dataset", dataset_name)
        details.insert(1, "sample_index", x.index)
        details["y_true"] = y.to_numpy()
        details["y_pred"] = y_pred
        details["predicted_probability"] = y_prob
        details["risk_level"] = details["predicted_probability"].map(risk_level)
        details["is_correct"] = details["y_true"] == details["y_pred"]
        detail_frames.append(details)

    prediction_details = pd.concat(detail_frames, ignore_index=True)
    prediction_details["predicted_probability"] = prediction_details["predicted_probability"].round(4)
    return prediction_details


def save_error_analysis(prediction_details: pd.DataFrame) -> pd.DataFrame:
    test_errors = prediction_details[
        (prediction_details["dataset"] == "test") & (~prediction_details["is_correct"])
    ].copy()
    test_errors["error_type"] = test_errors.apply(
        lambda row: "FP" if row["y_true"] == 0 and row["y_pred"] == 1 else "FN",
        axis=1,
    )
    ordered_columns = [
        "dataset",
        "sample_index",
        "error_type",
        "y_true",
        "y_pred",
        "predicted_probability",
        "risk_level",
    ] + FEATURE_COLUMNS
    test_errors = test_errors[ordered_columns].sort_values(
        ["error_type", "predicted_probability"],
        ascending=[True, False],
    )
    test_errors.to_csv(ERROR_ANALYSIS_PATH, index=False, encoding="utf-8-sig")
    return test_errors


def save_confusion_matrix(model) -> None:
    x_test, y_test = load_split("test_processed.csv")
    y_pred = model.predict(x_test).astype(int)
    matrix = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["低风险/未患病", "高风险/患病"],
    )
    display.plot(cmap="Blues", values_format="d", colorbar=False, ax=ax)
    ax.set_title("最终模型测试集混淆矩阵", fontsize=14, weight="bold")
    ax.set_xlabel("预测标签")
    ax.set_ylabel("真实标签")
    fig.tight_layout()
    fig.savefig(CONFUSION_MATRIX_PATH, dpi=220)
    plt.close(fig)


def save_roc_curve(model) -> None:
    x_test, y_test = load_split("test_processed.csv")
    y_prob = predict_probability(model, x_test)
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    RocCurveDisplay(fpr=fpr, tpr=tpr).plot(ax=ax, color="#2563eb", linewidth=2.5)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#94a3b8", linewidth=1.5)
    ax.set_title("最终模型测试集 ROC 曲线", fontsize=14, weight="bold")
    ax.set_xlabel("假阳性率 FPR")
    ax.set_ylabel("真阳性率 TPR")
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(ROC_CURVE_PATH, dpi=220)
    plt.close(fig)


def main() -> None:
    ensure_output_dirs()
    configure_plots()
    model = load_best_model()

    metrics_df = build_metrics_table(model)
    prediction_details = build_prediction_details(model)
    error_df = save_error_analysis(prediction_details)
    save_confusion_matrix(model)
    save_roc_curve(model)

    metrics_df.to_csv(FINAL_METRICS_PATH, index=False, encoding="utf-8-sig")
    prediction_details.to_csv(FINAL_PREDICTION_DETAILS_PATH, index=False, encoding="utf-8-sig")

    print("最终模型评估完成")
    print(metrics_df.to_string(index=False))
    print(f"预测明细已保存到: {FINAL_PREDICTION_DETAILS_PATH}")
    print(f"误判样本已保存到: {ERROR_ANALYSIS_PATH}，共 {len(error_df)} 条")
    print(f"混淆矩阵已保存到: {CONFUSION_MATRIX_PATH}")
    print(f"ROC 曲线已保存到: {ROC_CURVE_PATH}")


if __name__ == "__main__":
    main()
