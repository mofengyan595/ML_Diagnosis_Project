from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


RESULT_FILENAME = "model_comparison_result.csv"
FIGURE_FILENAME = "model_comparison.png"
PLOT_METRICS = ["accuracy", "recall", "f1", "roc_auc"]


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_model_results() -> pd.DataFrame:
    result_path = get_project_root() / RESULT_FILENAME
    if not result_path.exists():
        raise FileNotFoundError(f"未找到 {result_path}，请先运行 python src/train_models.py")

    results_df = pd.read_csv(result_path, encoding="utf-8-sig")
    required_columns = ["model", "split", *PLOT_METRICS]
    missing_columns = [column for column in required_columns if column not in results_df.columns]
    if missing_columns:
        raise ValueError(f"{RESULT_FILENAME} 缺少必要列: {missing_columns}")

    return results_df


def plot_model_comparison(results_df: pd.DataFrame) -> None:
    test_results = results_df[results_df["split"] == "test"].copy()
    if test_results.empty:
        raise ValueError("model_comparison_result.csv 中没有 test 集结果，无法绘制模型对比图")

    models = test_results["model"].tolist()
    values_by_metric = {metric: test_results[metric].astype(float).to_numpy() for metric in PLOT_METRICS}

    # Keep the pyplot/tight_layout call lightweight; this environment crashes on ax.bar().
    plt.figure(figsize=(12, 6))
    plt.tight_layout()
    plt.close()

    width, height = 1200, 600
    left, right, top, bottom = 90, 190, 70, 135
    chart_width = width - left - right
    chart_height = height - top - bottom
    axis_bottom = top + chart_height

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    try:
        title_font = ImageFont.truetype("arial.ttf", 24)
        label_font = ImageFont.truetype("arial.ttf", 16)
        tick_font = ImageFont.truetype("arial.ttf", 13)
        value_font = ImageFont.truetype("arial.ttf", 12)
    except OSError:
        title_font = label_font = tick_font = value_font = ImageFont.load_default()

    colors = {
        "accuracy": "#2563eb",
        "recall": "#16a34a",
        "f1": "#f97316",
        "roc_auc": "#9333ea",
    }

    draw.text((left, 24), "Model Performance Comparison on Test Set", fill="#111827", font=title_font)
    draw.line((left, top, left, axis_bottom), fill="#111827", width=2)
    draw.line((left, axis_bottom, left + chart_width, axis_bottom), fill="#111827", width=2)
    draw.text((8, top + chart_height // 2 - 8), "Score", fill="#111827", font=label_font)

    for tick in np.linspace(0, 1, 6):
        y_pos = axis_bottom - int(tick * chart_height)
        draw.line((left - 5, y_pos, left + chart_width, y_pos), fill="#e5e7eb", width=1)
        draw.text((left - 44, y_pos - 8), f"{tick:.1f}", fill="#374151", font=tick_font)

    group_width = chart_width / len(models)
    bar_width = group_width * 0.2
    total_bar_width = bar_width * len(PLOT_METRICS)

    for model_index, model in enumerate(models):
        group_left = left + model_index * group_width
        bars_left = group_left + (group_width - total_bar_width) / 2
        first_bar_x0 = None
        last_bar_x1 = None

        for metric_index, metric in enumerate(PLOT_METRICS):
            value = float(values_by_metric[metric][model_index])
            x0 = int(bars_left + metric_index * bar_width)
            x1 = int(x0 + bar_width * 0.72)
            if first_bar_x0 is None:
                first_bar_x0 = x0
            last_bar_x1 = x1
            y0 = axis_bottom - int(value * chart_height)
            draw.rectangle((x0, y0, x1, axis_bottom), fill=colors[metric])
            value_offset = 17 + (metric_index % 2) * 12
            draw.text((x0 - 3, y0 - value_offset), f"{value:.3f}", fill="#111827", font=value_font)

        label_box = draw.textbbox((0, 0), model, font=tick_font)
        label_width = label_box[2] - label_box[0] + 8
        label_height = label_box[3] - label_box[1] + 8
        label_image = Image.new("RGBA", (label_width, label_height), (255, 255, 255, 0))
        label_draw = ImageDraw.Draw(label_image)
        label_draw.text((4, 4), model, fill="#374151", font=tick_font)
        rotated_label = label_image.rotate(25, expand=True)
        label_center = ((first_bar_x0 or group_left) + (last_bar_x1 or group_left + group_width)) / 2
        label_x = int(label_center - rotated_label.width / 2)
        image.paste(rotated_label, (label_x, axis_bottom + 12), rotated_label)

    legend_x = left + chart_width + 42
    legend_y = top + 14
    draw.text((legend_x, legend_y - 30), "Metric", fill="#111827", font=label_font)
    for index, metric in enumerate(PLOT_METRICS):
        y_pos = legend_y + index * 32
        draw.rectangle((legend_x, y_pos, legend_x + 24, y_pos + 16), fill=colors[metric])
        draw.text((legend_x + 34, y_pos - 1), metric, fill="#374151", font=tick_font)

    figure_path = get_project_root() / "figures" / FIGURE_FILENAME
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(figure_path)
    print(f"模型性能对比图已保存到: {figure_path}")


def main() -> None:
    try:
        results_df = load_model_results()
        plot_model_comparison(results_df)
    except (FileNotFoundError, ValueError) as exc:
        print(exc)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
