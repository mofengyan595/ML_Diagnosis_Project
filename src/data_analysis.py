from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "diabetes.csv"
FIGURES_DIR = BASE_DIR / "figures"

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

FIELD_DESCRIPTIONS = {
    "Pregnancies": "怀孕次数，表示患者既往怀孕的次数。0 是合理取值。",
    "Glucose": "口服葡萄糖耐量测试中的血糖浓度，是糖尿病风险预测的重要指标。0 在医学上不合理。",
    "BloodPressure": "舒张压，单位通常为 mmHg。0 在医学上不合理。",
    "SkinThickness": "三头肌皮褶厚度，反映皮下脂肪情况。0 通常代表缺失测量。",
    "Insulin": "2 小时血清胰岛素水平。0 通常代表缺失测量。",
    "BMI": "身体质量指数，计算公式为体重/身高平方。0 在医学上不合理。",
    "DiabetesPedigreeFunction": "糖尿病家族遗传指数，数值越高表示家族遗传风险越高。",
    "Age": "年龄，单位为岁。",
    "Outcome": "分类标签，0 表示未患病或低风险，1 表示患病或高风险。",
}

INVALID_ZERO_COLUMNS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def ensure_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def set_plot_style() -> None:
    # 设置中文字体，保证图表标题和坐标轴标签在 Windows/常见中文环境下可读。
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    sns.set_theme(style="whitegrid", font="Microsoft YaHei")


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, encoding="utf-8")
    missing_columns = [column for column in EXPECTED_COLUMNS if column not in df.columns]
    extra_columns = [column for column in df.columns if column not in EXPECTED_COLUMNS]
    if missing_columns or extra_columns:
        raise ValueError(
            "数据字段与预期不一致。"
            f"缺少字段: {missing_columns or '无'}; "
            f"额外字段: {extra_columns or '无'}"
        )
    return df[EXPECTED_COLUMNS].copy()


def save_class_distribution(df: pd.DataFrame) -> None:
    counts = df[TARGET_COLUMN].value_counts().sort_index()
    labels = ["未患病/低风险", "患病/高风险"]

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(x=labels, y=counts.values, hue=labels, palette=["#3b82f6", "#ef4444"], legend=False, ax=ax)
    ax.set_title("糖尿病风险类别分布", fontsize=15, weight="bold")
    ax.set_xlabel("类别")
    ax.set_ylabel("样本数量")

    total = len(df)
    for index, value in enumerate(counts.values):
        ax.text(index, value + 6, f"{value} ({value / total:.1%})", ha="center", fontsize=11)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "class_distribution.png", dpi=200)
    plt.close(fig)


def save_feature_distribution(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()

    for ax, column in zip(axes, FEATURE_COLUMNS):
        sns.histplot(data=df, x=column, hue=TARGET_COLUMN, bins=30, kde=True, element="step", ax=ax)
        ax.set_title(column, fontsize=12, weight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("样本数")

        # 对医学上不合理的 0 值做提示，便于报告说明后续清洗逻辑。
        if column in INVALID_ZERO_COLUMNS:
            zero_count = int((df[column] == 0).sum())
            ax.axvline(0, color="#dc2626", linestyle="--", linewidth=1)
            ax.text(0.02, 0.95, f"0 值: {zero_count}", transform=ax.transAxes, va="top", color="#b91c1c")

    fig.suptitle("各特征分布及类别对比", fontsize=16, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIGURES_DIR / "feature_distribution.png", dpi=200)
    plt.close(fig)


def save_correlation_heatmap(df: pd.DataFrame) -> None:
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, square=True, linewidths=0.5, ax=ax)
    ax.set_title("特征与标签相关性热力图", fontsize=15, weight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=200)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    set_plot_style()
    df = load_data()

    print("数据读取成功")
    print(f"数据规模: {df.shape[0]} 行, {df.shape[1]} 列")
    print("\n字段含义:")
    for column in EXPECTED_COLUMNS:
        print(f"- {column}: {FIELD_DESCRIPTIONS[column]}")
    print("\n缺失值统计:")
    print(df.isna().sum())
    print("\n0 值统计:")
    print((df == 0).sum())
    print("\n类别分布:")
    print(df[TARGET_COLUMN].value_counts().sort_index())

    save_class_distribution(df)
    save_feature_distribution(df)
    save_correlation_heatmap(df)

    print(f"\n图表已保存到: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
