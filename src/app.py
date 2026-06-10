from __future__ import annotations

from html import escape
import json
import math

import gradio as gr

from config import BASE_DIR, BASELINE_MODEL_PATH, MODEL_PATH, RISK_THRESHOLDS
from predict import predict_diabetes_risk


DEFAULT_SAMPLE = (1, 120, 72, 20, 80, 25.0, 0.5, 35)
LOW_RISK_SAMPLE = (1, 85, 66, 29, 80, 26.6, 0.351, 31)
HIGH_RISK_SAMPLE = (6, 148, 72, 35, 80, 33.6, 0.627, 50)
BEST_PARAMS_PATH = BASE_DIR / "best_params.json"

INPUT_SPECS = [
    ("pregnancies", "怀孕次数", "次", 0, 20, 0, 10),
    ("glucose", "血糖", "mg/dL", 40, 300, 70, 140),
    ("blood_pressure", "血压", "mmHg", 40, 220, 60, 90),
    ("skin_thickness", "皮肤厚度", "mm", 1, 100, 10, 50),
    ("insulin", "胰岛素", "uU/mL", 1, 900, 16, 166),
    ("bmi", "BMI", "kg/m²", 10, 70, 18.5, 24.9),
    ("diabetes_pedigree", "遗传指数", "", 0.01, 3, 0.01, 0.8),
    ("age", "年龄", "岁", 18, 100, 18, 65),
]

CUSTOM_CSS = """
:root {
    --ink: #172033;
    --muted: #5f6f85;
    --line: #d7e0ea;
    --panel: #ffffff;
    --page: #f5f8fb;
    --primary: #1f5fbf;
    --primary-dark: #174a98;
    --teal: #0f766e;
    --low: #15803d;
    --low-bg: #ecfdf3;
    --medium: #b45309;
    --medium-bg: #fff7df;
    --high: #b91c1c;
    --high-bg: #fff1f2;
    --invalid: #7f1d1d;
    --invalid-bg: #fff1f2;
}

.gradio-container {
    max-width: 1240px !important;
    margin: 0 auto !important;
    background: var(--page);
    color: var(--ink);
}

#hero {
    border-bottom: 1px solid var(--line);
    padding: 24px 4px 18px;
    margin-bottom: 16px;
}

#hero h1 {
    margin: 0 0 8px;
    color: var(--ink);
    font-size: 28px;
    line-height: 1.2;
    letter-spacing: 0;
}

#hero p {
    margin: 0;
    max-width: 920px;
    color: var(--muted);
    font-size: 14px;
    line-height: 1.7;
}

.model-strip {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 10px;
    margin: 4px 0 16px;
}

.status-pill {
    border: 1px solid var(--line);
    border-left: 4px solid var(--teal);
    border-radius: 8px;
    background: #ffffff;
    padding: 11px 13px;
    min-height: 58px;
}

.status-label {
    color: var(--muted);
    font-size: 12px;
    font-weight: 700;
    margin-bottom: 4px;
}

.status-value {
    color: var(--ink);
    font-size: 14px;
    font-weight: 800;
}

.panel {
    border: 1px solid var(--line);
    border-radius: 8px;
    background: var(--panel);
    padding: 18px;
    box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
}

.section-title {
    margin: 0 0 6px;
    color: var(--ink);
    font-size: 15px;
    font-weight: 850;
}

.hint {
    margin: 0 0 12px;
    color: var(--muted);
    font-size: 12.5px;
    line-height: 1.65;
}

#sample-panel button,
#sample-actions button,
#action-row button {
    min-height: 38px;
    border-radius: 8px;
    font-weight: 750;
}

#dashboard-row {
    align-items: flex-start;
}

#input-column,
#result-column {
    gap: 14px;
}

#analysis-grid {
    align-items: stretch;
    gap: 14px;
}

#sample-actions {
    gap: 8px;
    margin: 2px 0 8px;
}

#input-panel {
    padding-bottom: 16px;
}

#predict-button button {
    min-height: 48px;
    border-radius: 8px;
    background: var(--primary);
    border-color: var(--primary);
    font-weight: 850;
    box-shadow: 0 12px 24px rgba(31, 95, 191, 0.20);
}

#predict-button button:hover {
    background: var(--primary-dark);
    border-color: var(--primary-dark);
}

.result-placeholder {
    border: 1px dashed #b9c7d8;
    border-radius: 8px;
    padding: 22px;
    background: #f8fafc;
    color: var(--muted);
    text-align: center;
    line-height: 1.7;
}

.risk-card {
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 20px;
    background: #ffffff;
}

.risk-card.low {
    border-color: #86efac;
    background: linear-gradient(180deg, #ffffff 0%, var(--low-bg) 100%);
}

.risk-card.medium {
    border-color: #facc15;
    background: linear-gradient(180deg, #ffffff 0%, var(--medium-bg) 100%);
}

.risk-card.high {
    border-color: #fca5a5;
    background: linear-gradient(180deg, #ffffff 0%, var(--high-bg) 100%);
}

.risk-card.invalid {
    border-color: #fda4af;
    background: linear-gradient(180deg, #ffffff 0%, var(--invalid-bg) 100%);
}

.result-label {
    color: var(--muted);
    font-size: 12px;
    font-weight: 800;
    margin-bottom: 9px;
}

.result-main {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 16px;
    margin-bottom: 14px;
}

.risk-name {
    color: var(--ink);
    font-size: 30px;
    line-height: 1.15;
    font-weight: 900;
}

.risk-card.low .risk-name {
    color: var(--low);
}

.risk-card.medium .risk-name {
    color: var(--medium);
}

.risk-card.high .risk-name {
    color: var(--high);
}

.risk-card.invalid .risk-name {
    color: var(--invalid);
}

.probability {
    color: var(--ink);
    font-size: 28px;
    font-weight: 900;
}

.meter {
    height: 10px;
    border-radius: 999px;
    background: #dbe4ee;
    overflow: hidden;
}

.meter span {
    display: block;
    height: 100%;
    border-radius: 999px;
}

.risk-card.low .meter span {
    background: var(--low);
}

.risk-card.medium .meter span {
    background: var(--medium);
}

.risk-card.high .meter span {
    background: var(--high);
}

.risk-card.invalid .meter span {
    background: var(--invalid);
}

.explanation-box {
    border: 1px solid var(--line);
    border-radius: 8px;
    background: #ffffff;
    padding: 14px 16px;
    min-height: 188px;
}

.explanation-box p,
.explanation-box li {
    color: var(--ink);
    font-size: 13px;
    line-height: 1.72;
}

.explanation-box strong {
    color: var(--primary-dark);
}

.footer-note {
    margin-top: 14px;
    color: var(--muted);
    font-size: 12px;
    text-align: center;
}

.compact-row {
    gap: 10px;
}

.compact-row .info,
.compact-row [data-testid="block-info"] {
    white-space: nowrap;
    word-break: keep-all;
}

.gauge-box {
    border: 1px solid var(--line);
    border-radius: 8px;
    background: #ffffff;
    padding: 14px 16px 12px;
    margin-top: 12px;
}

.gauge-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    color: var(--ink);
    font-weight: 850;
    font-size: 13px;
}

.gauge-head strong {
    font-size: 18px;
}

.gauge-wrap {
    position: relative;
    width: min(100%, 268px);
    height: 132px;
    margin: 10px auto 0;
    overflow: hidden;
}

.gauge-arc {
    position: absolute;
    left: 50%;
    top: 6px;
    width: 254px;
    height: 254px;
    transform: translateX(-50%);
    border-radius: 50%;
    background: conic-gradient(from 270deg, var(--low) 0deg 63deg, var(--medium) 63deg 117deg, var(--high) 117deg 180deg, #e6edf5 180deg 360deg);
}

.gauge-arc::after {
    content: "";
    position: absolute;
    left: 50%;
    top: 31px;
    width: 190px;
    height: 190px;
    transform: translateX(-50%);
    border-radius: 50%;
    background: #ffffff;
}

.gauge-needle {
    position: absolute;
    left: 50%;
    bottom: 20px;
    width: 4px;
    height: 90px;
    border-radius: 999px;
    background: var(--ink);
    transform-origin: bottom center;
    box-shadow: 0 4px 12px rgba(23, 32, 51, 0.22);
}

.gauge-knob {
    position: absolute;
    left: 50%;
    bottom: 12px;
    width: 18px;
    height: 18px;
    transform: translateX(-50%);
    border: 4px solid var(--ink);
    border-radius: 50%;
    background: #ffffff;
}

.gauge-labels,
.risk-legend {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 8px;
}

.gauge-labels {
    color: var(--muted);
    font-size: 12px;
    margin-top: -2px;
    text-align: center;
}

.risk-legend {
    border: 1px solid var(--line);
    border-radius: 8px;
    background: #ffffff;
    padding: 10px;
    margin: 12px 0;
}

.legend-item {
    border-radius: 7px;
    padding: 8px 9px;
    font-size: 12px;
    line-height: 1.45;
}

.legend-item strong {
    display: block;
    color: var(--ink);
    font-size: 12.5px;
}

.legend-item span {
    display: inline-block;
    width: 9px;
    height: 9px;
    border-radius: 50%;
    margin-right: 5px;
}

.legend-low {
    background: var(--low-bg);
}

.legend-medium {
    background: var(--medium-bg);
}

.legend-high {
    background: var(--high-bg);
}

.legend-low span {
    background: var(--low);
}

.legend-medium span {
    background: var(--medium);
}

.legend-high span {
    background: var(--high);
}

.visual-panel {
    border: 1px solid var(--line);
    border-radius: 8px;
    background: var(--panel);
    padding: 14px;
    box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
    min-height: 336px;
    box-sizing: border-box;
}

.visual-card {
    min-height: 304px;
}

.visual-card h3 {
    margin: 0 0 10px;
    color: var(--ink);
    font-size: 14px;
    font-weight: 850;
}

.visual-card.empty {
    display: flex;
    min-height: 304px;
    flex-direction: column;
    justify-content: center;
    text-align: center;
}

.chart-note {
    margin-top: 10px;
    color: var(--muted);
    font-size: 11.5px;
    line-height: 1.5;
}

.radar-svg {
    display: block;
    width: 100%;
    max-width: 260px;
    margin: 0 auto;
}

#radar-panel .visual-card {
    display: flex;
    flex-direction: column;
    justify-content: center;
}

#radar-panel .chart-note {
    text-align: center;
}

#reference-panel .visual-card {
    min-height: 304px;
}

#reference-panel .reference-row {
    grid-template-columns: 104px minmax(0, 1fr) 68px;
    margin: 13px 0;
}

#reference-panel .reference-track {
    height: 16px;
}

#reference-panel .reference-normal {
    top: 4px;
}

.factor-row,
.reference-row {
    display: grid;
    align-items: center;
    gap: 9px;
    margin: 10px 0;
}

.factor-row {
    grid-template-columns: 90px minmax(0, 1fr) 32px;
}

.reference-row {
    grid-template-columns: 82px minmax(0, 1fr) 58px;
}

.factor-label,
.reference-label {
    color: var(--ink);
    font-size: 12px;
    font-weight: 760;
    line-height: 1.25;
}

.factor-label small,
.reference-label small {
    display: block;
    margin-top: 2px;
    color: var(--muted);
    font-size: 10.5px;
    font-weight: 600;
}

.factor-track,
.reference-track {
    position: relative;
    border-radius: 999px;
    background: #dbe4ee;
}

.factor-track {
    height: 12px;
    overflow: hidden;
}

.factor-fill {
    display: block;
    height: 100%;
    border-radius: 999px;
}

.factor-score,
.reference-value {
    color: var(--muted);
    font-size: 11.5px;
    font-weight: 800;
    text-align: right;
}

.reference-track {
    height: 14px;
    overflow: visible;
}

.reference-normal {
    position: absolute;
    top: 3px;
    height: 8px;
    border-radius: 999px;
    background: rgba(21, 128, 61, 0.75);
}

.reference-dot {
    position: absolute;
    top: 50%;
    width: 13px;
    height: 13px;
    transform: translate(-50%, -50%);
    border: 2px solid #ffffff;
    border-radius: 50%;
    box-shadow: 0 2px 7px rgba(23, 32, 51, 0.20);
}

@media (max-width: 820px) {
    #hero h1 {
        font-size: 23px;
    }

    .model-strip {
        grid-template-columns: 1fr;
    }

    .panel {
        padding: 15px;
    }

    .result-main {
        display: block;
    }

    .probability {
        margin-top: 6px;
    }

    .risk-legend {
        grid-template-columns: 1fr;
    }

    .factor-row,
    .reference-row {
        grid-template-columns: 1fr;
        gap: 4px;
    }

    .factor-score,
    .reference-value {
        text-align: left;
    }
}
"""


def _sample_to_outputs(sample: tuple[float, ...]) -> tuple[float, ...]:
    return sample


def _risk_class(level: str) -> str:
    if level == "输入异常":
        return "invalid"
    if level == "低风险":
        return "low"
    if level == "中风险":
        return "medium"
    return "high"


def _format_result_card(level: str, probability_text: str) -> str:
    probability_value = 0.0 if probability_text == "无法评估" else float(probability_text.rstrip("%"))
    risk_class = _risk_class(level)
    return f"""
    <div class="risk-card {risk_class}">
        <div class="result-label">筛查评估结果</div>
        <div class="result-main">
            <div class="risk-name">{level}</div>
            <div class="probability">{probability_text}</div>
        </div>
        <div class="meter">
            <span style="width: {probability_value}%;"></span>
        </div>
    </div>
    """


def _format_explanation(explanation: str) -> str:
    escaped = explanation.strip()
    if not escaped:
        return '<div class="explanation-box">暂无结果说明。</div>'

    sections = escaped.split("\n\n")
    html_parts = ['<div class="explanation-box">']
    for section in sections:
        lines = [line.strip() for line in section.splitlines() if line.strip()]
        if not lines:
            continue
        if len(lines) == 1:
            line = lines[0]
            if "：" in line:
                title, body = line.split("：", 1)
                html_parts.append(f"<p><strong>{title}：</strong>{body}</p>")
            else:
                html_parts.append(f"<p>{line}</p>")
            continue

        title = lines[0]
        if title.endswith("："):
            html_parts.append(f"<p><strong>{title}</strong></p>")
            html_parts.append("<ul>")
            for item in lines[1:]:
                html_parts.append(f"<li>{item.lstrip('- ')}</li>")
            html_parts.append("</ul>")
        else:
            html_parts.append("<ul>")
            for item in lines:
                html_parts.append(f"<li>{item.lstrip('- ')}</li>")
            html_parts.append("</ul>")
    html_parts.append("</div>")
    return "\n".join(html_parts)


def _format_metric_value(value: float, unit: str = "") -> str:
    value_text = f"{value:.0f}" if math.isclose(value, round(value), abs_tol=0.05) else f"{value:.1f}"
    return f"{value_text} {unit}".strip()


def _parse_probability(probability_text: str) -> float | None:
    if probability_text == "无法评估":
        return None
    try:
        return max(0.0, min(100.0, float(probability_text.rstrip("%"))))
    except ValueError:
        return None


def _patient_values(
    pregnancies: float,
    glucose: float,
    blood_pressure: float,
    skin_thickness: float,
    insulin: float,
    bmi: float,
    diabetes_pedigree: float,
    age: float,
) -> dict[str, float]:
    raw_values = (
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree,
        age,
    )
    return {name: float(value) for (name, *_), value in zip(INPUT_SPECS, raw_values)}


def _normalize(value: float, lower: float, upper: float) -> float:
    if math.isclose(lower, upper):
        return 0.0
    return max(0.0, min(1.0, (value - lower) / (upper - lower)))


def _feature_scores(values: dict[str, float]) -> list[dict[str, object]]:
    scores = []
    for name, label, unit, valid_min, valid_max, ref_low, ref_high in INPUT_SPECS:
        value = values[name]
        ref_width = max(ref_high - ref_low, 1e-6)
        if value < ref_low:
            score = (ref_low - value) / ref_width * 100
            direction = "低于参考范围"
        elif value > ref_high:
            score = (value - ref_high) / ref_width * 100
            direction = "高于参考范围"
        else:
            score = 0.0
            direction = "参考范围内"

        scores.append(
            {
                "name": name,
                "label": label,
                "unit": unit,
                "value": value,
                "valid_min": valid_min,
                "valid_max": valid_max,
                "ref_low": ref_low,
                "ref_high": ref_high,
                "score": max(0.0, min(100.0, score)),
                "direction": direction,
            }
        )
    return scores


def _score_color(score: float) -> str:
    if score >= 65:
        return "#b91c1c"
    if score >= 25:
        return "#b45309"
    return "#15803d"


def _empty_visual_html(title: str, message: str) -> str:
    return f"""
    <div class="visual-card empty">
        <h3>{escape(title)}</h3>
        <div class="chart-note">{escape(message)}</div>
    </div>
    """


def _format_probability_gauge(probability_text: str, level: str) -> str:
    probability = _parse_probability(probability_text)
    pointer = 0.0 if probability is None else probability
    needle_angle = -90 + pointer * 1.8
    risk_class = _risk_class(level)
    return f"""
    <div class="gauge-box {risk_class}">
        <div class="gauge-head">
            <span>风险概率仪表盘</span>
            <strong>{probability_text}</strong>
        </div>
        <div class="gauge-wrap">
            <div class="gauge-arc"></div>
            <div class="gauge-needle" style="transform: translateX(-50%) rotate({needle_angle:.1f}deg);"></div>
            <div class="gauge-knob"></div>
        </div>
        <div class="gauge-labels">
            <span>低风险</span>
            <span>中风险</span>
            <span>高风险</span>
        </div>
    </div>
    """


def _risk_legend_html() -> str:
    low_threshold = RISK_THRESHOLDS["low"] * 100
    high_threshold = RISK_THRESHOLDS["high"] * 100
    return f"""
    <div class="risk-legend">
        <div class="legend-item legend-low">
            <strong><span></span>低风险</strong>
            &lt; {low_threshold:.0f}%
        </div>
        <div class="legend-item legend-medium">
            <strong><span></span>中风险</strong>
            {low_threshold:.0f}% - {high_threshold:.0f}%
        </div>
        <div class="legend-item legend-high">
            <strong><span></span>高风险</strong>
            ≥ {high_threshold:.0f}%
        </div>
    </div>
    """


def _radar_points(series: list[float], radius: float = 74.0, center: float = 120.0) -> str:
    points = []
    for idx, value in enumerate(series):
        angle = -math.pi / 2 + 2 * math.pi * idx / len(series)
        x_position = center + radius * value * math.cos(angle)
        y_position = center + radius * value * math.sin(angle)
        points.append(f"{x_position:.1f},{y_position:.1f}")
    return " ".join(points)


def _build_radar_plot(values: dict[str, float]) -> str:
    labels = [spec[1] for spec in INPUT_SPECS]
    patient_series = [
        _normalize(values[name], valid_min, valid_max)
        for name, _, _, valid_min, valid_max, _, _ in INPUT_SPECS
    ]
    reference_series = [
        _normalize(ref_high, valid_min, valid_max)
        for _, _, _, valid_min, valid_max, _, ref_high in INPUT_SPECS
    ]

    grid_shapes = []
    for ratio in (0.25, 0.5, 0.75, 1.0):
        grid_shapes.append(
            f'<polygon points="{_radar_points([ratio] * len(labels))}" fill="none" '
            'stroke="#d7e0ea" stroke-width="0.8" />'
        )

    axis_lines = []
    label_nodes = []
    for idx, label in enumerate(labels):
        angle = -math.pi / 2 + 2 * math.pi * idx / len(labels)
        axis_x = 120 + 74 * math.cos(angle)
        axis_y = 120 + 74 * math.sin(angle)
        label_x = 120 + 99 * math.cos(angle)
        label_y = 120 + 99 * math.sin(angle)
        anchor = "middle"
        if label_x < 105:
            anchor = "end"
        elif label_x > 135:
            anchor = "start"
        axis_lines.append(f'<line x1="120" y1="120" x2="{axis_x:.1f}" y2="{axis_y:.1f}" stroke="#d7e0ea" stroke-width="0.8" />')
        label_nodes.append(
            f'<text x="{label_x:.1f}" y="{label_y:.1f}" text-anchor="{anchor}" '
            f'font-size="8.6" fill="#172033">{escape(label)}</text>'
        )

    return f"""
    <div class="visual-card">
        <h3>个体输入指标雷达图</h3>
        <svg class="radar-svg" viewBox="0 0 240 240" role="img" aria-label="个体输入指标雷达图">
            {"".join(grid_shapes)}
            {"".join(axis_lines)}
            <polygon points="{_radar_points(reference_series)}" fill="none" stroke="#15803d" stroke-width="2" stroke-dasharray="4 4" />
            <polygon points="{_radar_points(patient_series)}" fill="#1f5fbf" fill-opacity="0.16" stroke="#1f5fbf" stroke-width="2.4" />
            <circle cx="120" cy="120" r="2.8" fill="#172033" />
            {"".join(label_nodes)}
        </svg>
        <div class="chart-note">蓝色区域为当前输入在有效范围中的相对位置，绿色虚线为各指标参考上限。</div>
    </div>
    """


def _build_factor_bar_plot(values: dict[str, float]) -> str:
    ranked = sorted(_feature_scores(values), key=lambda item: float(item["score"]), reverse=True)[:5]
    if not ranked or float(ranked[0]["score"]) == 0:
        return _empty_visual_html("关键风险因素条形图", "当前输入未明显偏离参考范围")

    rows = []
    for item in ranked:
        score = float(item["score"])
        label = escape(str(item["label"]))
        value_text = escape(_format_metric_value(float(item["value"]), str(item["unit"])))
        direction = escape(str(item["direction"]))
        rows.append(
            f"""
            <div class="factor-row">
                <div class="factor-label">{label}<small>{value_text} · {direction}</small></div>
                <div class="factor-track">
                    <span class="factor-fill" style="width: {score:.1f}%; background: {_score_color(score)};"></span>
                </div>
                <div class="factor-score">{score:.0f}</div>
            </div>
            """
        )

    return f"""
    <div class="visual-card">
        <h3>关键风险因素条形图</h3>
        {"".join(rows)}
        <div class="chart-note">按当前输入相对参考范围的偏离程度排序，分数越高表示越需要关注。</div>
    </div>
    """


def _build_reference_range_plot(values: dict[str, float]) -> str:
    rows = []
    for item in _feature_scores(values):
        valid_min = float(item["valid_min"])
        valid_max = float(item["valid_max"])
        ref_low = float(item["ref_low"])
        ref_high = float(item["ref_high"])
        value = float(item["value"])
        score = float(item["score"])
        unit = str(item["unit"])

        ref_start = _normalize(ref_low, valid_min, valid_max) * 100
        ref_end = _normalize(ref_high, valid_min, valid_max) * 100
        ref_width = max(2.0, ref_end - ref_start)
        value_position = _normalize(value, valid_min, valid_max) * 100
        label = escape(str(item["label"]))
        ref_text = escape(f"{ref_low:g}-{ref_high:g} {unit}".strip())
        value_text = escape(_format_metric_value(value, unit))
        rows.append(
            f"""
            <div class="reference-row">
                <div class="reference-label">{label}<small>参考 {ref_text}</small></div>
                <div class="reference-track">
                    <span class="reference-normal" style="left: {ref_start:.1f}%; width: {ref_width:.1f}%;"></span>
                    <span class="reference-dot" style="left: {value_position:.1f}%; background: {_score_color(score)};"></span>
                </div>
                <div class="reference-value">{value_text}</div>
            </div>
            """
        )

    return f"""
    <div class="visual-card">
        <h3>与正常参考范围对比图</h3>
        {"".join(rows)}
        <div class="chart-note">灰色条表示有效输入范围，绿色段表示参考范围，圆点表示当前个体输入值。</div>
    </div>
    """


def _load_model_metadata() -> dict[str, object]:
    if not BEST_PARAMS_PATH.exists():
        return {}
    try:
        return json.loads(BEST_PARAMS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _model_status_html() -> str:
    metadata = _load_model_metadata()
    if MODEL_PATH.exists():
        model_name = metadata.get("selected_model", "最终模型")
        threshold = metadata.get("classification_threshold", "-")
        metrics = metadata.get("test_metrics", {})
        auc = metrics.get("roc_auc", "-") if isinstance(metrics, dict) else "-"
        f1 = metrics.get("f1", "-") if isinstance(metrics, dict) else "-"
        return f"""
        <div class="model-strip">
            <div class="status-pill">
                <div class="status-label">当前模型</div>
                <div class="status-value">{model_name}</div>
            </div>
            <div class="status-pill">
                <div class="status-label">分类阈值</div>
                <div class="status-value">{threshold}</div>
            </div>
            <div class="status-pill">
                <div class="status-label">测试集指标</div>
                <div class="status-value">AUC {auc} / F1 {f1}</div>
            </div>
        </div>
        """

    if BASELINE_MODEL_PATH.exists():
        return """
        <div class="model-strip">
            <div class="status-pill">
                <div class="status-label">当前模型</div>
                <div class="status-value">Logistic Regression baseline</div>
            </div>
            <div class="status-pill">
                <div class="status-label">模型状态</div>
                <div class="status-value">等待最终模型</div>
            </div>
            <div class="status-pill">
                <div class="status-label">推理模式</div>
                <div class="status-value">已接入预处理对象</div>
            </div>
        </div>
        """

    return """
    <div class="model-strip">
        <div class="status-pill">
            <div class="status-label">当前模型</div>
            <div class="status-value">演示规则函数</div>
        </div>
        <div class="status-pill">
            <div class="status-label">模型状态</div>
            <div class="status-value">未检测到模型文件</div>
        </div>
        <div class="status-pill">
            <div class="status-label">推理模式</div>
            <div class="status-value">仅用于页面兜底</div>
        </div>
    </div>
    """


def run_prediction(
    pregnancies: float,
    glucose: float,
    blood_pressure: float,
    skin_thickness: float,
    insulin: float,
    bmi: float,
    diabetes_pedigree: float,
    age: float,
) -> tuple[object, ...]:
    values = _patient_values(
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree,
        age,
    )
    level, probability_text, explanation = predict_diabetes_risk(
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree,
        age,
    )
    return (
        _format_result_card(level, probability_text),
        _format_explanation(explanation),
        _format_probability_gauge(probability_text, level),
        _build_radar_plot(values),
        _build_factor_bar_plot(values),
        _build_reference_range_plot(values),
    )


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="糖尿病风险预测问诊系统") as demo:
        gr.Markdown(
            """
            # 糖尿病风险预测问诊系统
            输入基础体征和代谢相关指标后，系统会完成输入校验、统一预处理、模型推理和风险解释。
            """,
            elem_id="hero",
        )
        gr.HTML(_model_status_html())

        with gr.Row(elem_id="dashboard-row"):
            with gr.Column(scale=7, elem_id="input-column"):
                with gr.Column(elem_classes="panel", elem_id="input-panel"):
                    gr.Markdown("问诊输入", elem_classes="section-title")
                    gr.Markdown(
                        "输入真实测量值后生成风险评估；若关键指标为 0 或超出合理范围，系统会提示修正。",
                        elem_classes="hint",
                    )
                    with gr.Row(elem_id="sample-actions"):
                        low_sample_button = gr.Button("低风险示例", variant="secondary")
                        high_sample_button = gr.Button("高风险示例", variant="secondary")
                        reset_button = gr.Button("默认输入", variant="secondary")

                    gr.Markdown("基础信息", elem_classes="section-title")
                    with gr.Row(elem_classes="compact-row"):
                        pregnancies = gr.Slider(
                            label="怀孕次数",
                            minimum=0,
                            maximum=20,
                            value=1,
                            step=1,
                            info="范围 0~20 次",
                        )
                        age = gr.Slider(
                            label="年龄",
                            minimum=18,
                            maximum=100,
                            value=35,
                            step=1,
                            info="范围 18~100 岁",
                        )

                    gr.Markdown("检查指标", elem_classes="section-title")
                    with gr.Row(elem_classes="compact-row"):
                        glucose = gr.Slider(
                            label="血糖值 (mg/dL)",
                            minimum=0,
                            maximum=300,
                            value=120,
                            step=1,
                            info="范围 40~300",
                        )
                        blood_pressure = gr.Slider(
                            label="血压 (mmHg)",
                            minimum=0,
                            maximum=220,
                            value=72,
                            step=1,
                            info="范围 40~220",
                        )
                    with gr.Row(elem_classes="compact-row"):
                        skin_thickness = gr.Slider(
                            label="皮肤厚度 (mm)",
                            minimum=0,
                            maximum=100,
                            value=20,
                            step=1,
                            info="范围 1~100",
                        )
                        insulin = gr.Slider(
                            label="胰岛素 (uU/mL)",
                            minimum=0,
                            maximum=900,
                            value=80,
                            step=1,
                            info="范围 1~900",
                        )
                    with gr.Row(elem_classes="compact-row"):
                        bmi = gr.Slider(
                            label="BMI (kg/m^2)",
                            minimum=0,
                            maximum=70,
                            value=25.0,
                            step=0.1,
                            info="范围 10~70",
                        )
                        diabetes_pedigree = gr.Slider(
                            label="糖尿病家族遗传指数",
                            minimum=0,
                            maximum=3,
                            value=0.5,
                            step=0.01,
                            info="范围 0.01~3",
                        )

                predict_button = gr.Button("生成风险评估", variant="primary", elem_id="predict-button")

                with gr.Row(elem_id="analysis-grid"):
                    with gr.Column(scale=2, elem_classes="visual-panel", elem_id="radar-panel") as radar_panel:
                        radar_plot = gr.HTML(_empty_visual_html("个体输入指标雷达图", "点击生成风险评估后显示"))
                    with gr.Column(scale=5, elem_classes="visual-panel", elem_id="factor-panel") as factor_panel:
                        factor_plot = gr.HTML(_empty_visual_html("关键风险因素条形图", "点击生成风险评估后显示"))

            with gr.Column(scale=5, elem_id="result-column"):
                with gr.Column(elem_classes="panel"):
                    gr.Markdown("评估结果", elem_classes="section-title")
                    result_card = gr.HTML(
                        """
                        <div class="result-placeholder">
                            完成问诊输入后点击“生成风险评估”，这里会显示风险等级和预测概率。
                        </div>
                        """
                    )
                    risk_gauge = gr.HTML(_format_probability_gauge("无法评估", "输入异常"))
                    gr.HTML(_risk_legend_html())

                explanation = gr.HTML(
                    """
                    <div class="explanation-box">
                        <p><strong>结果说明：</strong>尚未生成评估结果。</p>
                    </div>
                    """
                )

                with gr.Column(elem_classes="visual-panel", elem_id="reference-panel") as reference_panel:
                    reference_plot = gr.HTML(_empty_visual_html("与正常参考范围对比图", "点击生成风险评估后显示"))

        input_components = [
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            diabetes_pedigree,
            age,
        ]

        low_sample_button.click(
            fn=lambda: _sample_to_outputs(LOW_RISK_SAMPLE),
            outputs=input_components,
        )
        high_sample_button.click(
            fn=lambda: _sample_to_outputs(HIGH_RISK_SAMPLE),
            outputs=input_components,
        )
        reset_button.click(
            fn=lambda: _sample_to_outputs(DEFAULT_SAMPLE),
            outputs=input_components,
        )

        predict_button.click(
            fn=run_prediction,
            inputs=input_components,
            outputs=[
                result_card,
                explanation,
                risk_gauge,
                radar_plot,
                factor_plot,
                reference_plot,
            ],
            show_progress="hidden",
        )

        gr.Markdown(
            "本系统仅用于机器学习课程项目演示，不能替代医生诊断或真实临床决策。",
            elem_classes="footer-note",
        )

    return demo


if __name__ == "__main__":
    build_demo().launch(
        theme=gr.themes.Soft(
            primary_hue="blue",
            neutral_hue="slate",
            radius_size="sm",
            font=[gr.themes.GoogleFont("Noto Sans SC"), "Arial", "sans-serif"],
        ),
        css=CUSTOM_CSS,
    )
