from __future__ import annotations

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance

from evaluation_utils import (
    BEST_MODEL_PATH,
    FEATURE_COLUMNS,
    FEATURE_IMPORTANCE_FIGURE_PATH,
    FEATURE_IMPORTANCE_PATH,
    FEATURE_LABELS,
    RANDOM_STATE,
    ensure_output_dirs,
    load_split,
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


def compute_permutation_importance(model) -> pd.DataFrame:
    x_test, y_test = load_split("test_processed.csv")
    importance_model = getattr(model, "estimator", model)
    result = permutation_importance(
        importance_model,
        x_test,
        y_test,
        scoring="roc_auc",
        n_repeats=30,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    importance_df = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "feature_cn": [FEATURE_LABELS[feature] for feature in FEATURE_COLUMNS],
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    importance_df = importance_df.sort_values("importance_mean", ascending=False).reset_index(drop=True)
    importance_df["rank"] = importance_df.index + 1
    return importance_df


def save_feature_importance_plot(importance_df: pd.DataFrame) -> None:
    plot_df = importance_df.sort_values("importance_mean", ascending=True)

    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    sns.barplot(
        data=plot_df,
        x="importance_mean",
        y="feature_cn",
        hue="feature_cn",
        palette="Greens_r",
        legend=False,
        ax=ax,
    )
    ax.errorbar(
        x=plot_df["importance_mean"],
        y=range(len(plot_df)),
        xerr=plot_df["importance_std"],
        fmt="none",
        ecolor="#475569",
        elinewidth=1,
        capsize=3,
    )
    ax.axvline(0, color="#94a3b8", linewidth=1)
    ax.set_title("最终模型特征重要性（Permutation Importance）", fontsize=14, weight="bold")
    ax.set_xlabel("打乱该特征后 ROC AUC 的平均下降幅度")
    ax.set_ylabel("特征")
    fig.tight_layout()
    fig.savefig(FEATURE_IMPORTANCE_FIGURE_PATH, dpi=220)
    plt.close(fig)


def main() -> None:
    ensure_output_dirs()
    configure_plots()
    model = load_best_model()

    importance_df = compute_permutation_importance(model)
    rounded_df = importance_df.copy()
    rounded_df[["importance_mean", "importance_std"]] = rounded_df[["importance_mean", "importance_std"]].round(6)
    rounded_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False, encoding="utf-8-sig")
    save_feature_importance_plot(importance_df)

    print("最终模型解释分析完成")
    print(rounded_df.to_string(index=False))
    print(f"特征重要性表已保存到: {FEATURE_IMPORTANCE_PATH}")
    print(f"特征重要性图已保存到: {FEATURE_IMPORTANCE_FIGURE_PATH}")


if __name__ == "__main__":
    main()
