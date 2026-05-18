import gradio as gr

from predict import predict_diabetes_risk


DEFAULT_SAMPLE = (1, 120, 72, 20, 80, 25.0, 0.5, 35)
LOW_RISK_SAMPLE = (1, 85, 66, 29, 0, 26.6, 0.351, 31)
HIGH_RISK_SAMPLE = (6, 148, 72, 35, 0, 33.6, 0.627, 50)


CUSTOM_CSS = """
:root {
    --primary: #2563eb;
    --primary-dark: #1d4ed8;
    --ink: #172033;
    --muted: #64748b;
    --panel: #ffffff;
    --line: #d9e2ef;
    --soft: #f4f7fb;
    --low: #15803d;
    --low-bg: #dcfce7;
    --medium: #b45309;
    --medium-bg: #fef3c7;
    --high: #b91c1c;
    --high-bg: #fee2e2;
}

.gradio-container {
    max-width: 1180px !important;
    margin: 0 auto !important;
    background: linear-gradient(180deg, #f7fbff 0%, #edf4fb 100%);
    color: var(--ink);
}

#hero {
    padding: 30px 34px 24px;
    border: 1px solid var(--line);
    border-radius: 8px;
    background:
        linear-gradient(135deg, rgba(37, 99, 235, 0.13), rgba(20, 184, 166, 0.1)),
        #ffffff;
    box-shadow: 0 18px 42px rgba(15, 23, 42, 0.08);
    margin-bottom: 18px;
}

#hero h1 {
    font-size: 30px;
    line-height: 1.2;
    margin: 0 0 10px;
    letter-spacing: 0;
    color: var(--ink);
}

#hero p {
    max-width: 820px;
    margin: 0;
    color: var(--muted);
    font-size: 15px;
    line-height: 1.7;
}

.panel {
    border: 1px solid var(--line);
    border-radius: 8px;
    background: var(--panel);
    padding: 20px;
    box-shadow: 0 10px 26px rgba(15, 23, 42, 0.06);
}

.section-title {
    font-size: 16px;
    font-weight: 800;
    color: var(--ink);
    margin: 0 0 8px;
}

.hint {
    color: var(--muted);
    font-size: 13px;
    line-height: 1.6;
    margin-bottom: 14px;
}

#sample-panel button {
    min-height: 40px;
    border-radius: 8px;
    font-weight: 700;
}

#predict-button {
    margin-top: 18px;
}

#predict-button button {
    min-height: 48px;
    border-radius: 8px;
    font-weight: 800;
    background: var(--primary);
    border-color: var(--primary);
    box-shadow: 0 12px 24px rgba(37, 99, 235, 0.22);
}

#predict-button button:hover {
    background: var(--primary-dark);
    border-color: var(--primary-dark);
}

.result-placeholder {
    border: 1px dashed #cbd5e1;
    border-radius: 8px;
    padding: 22px;
    background: #f8fafc;
    color: var(--muted);
    text-align: center;
}

.risk-card {
    border-radius: 8px;
    border: 1px solid var(--line);
    padding: 22px;
    background: #ffffff;
}

.risk-card.low {
    border-color: #86efac;
    background: linear-gradient(180deg, #ffffff 0%, var(--low-bg) 100%);
}

.risk-card.medium {
    border-color: #fcd34d;
    background: linear-gradient(180deg, #ffffff 0%, var(--medium-bg) 100%);
}

.risk-card.high {
    border-color: #fca5a5;
    background: linear-gradient(180deg, #ffffff 0%, var(--high-bg) 100%);
}

.result-label {
    color: var(--muted);
    font-size: 13px;
    font-weight: 700;
    margin-bottom: 10px;
}

.result-main {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 16px;
    margin-bottom: 16px;
}

.risk-name {
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

.probability {
    font-size: 28px;
    font-weight: 900;
    color: var(--ink);
}

.meter {
    height: 10px;
    border-radius: 999px;
    background: #e2e8f0;
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

#explanation textarea {
    line-height: 1.7;
}

.footer-note {
    color: var(--muted);
    font-size: 12px;
    text-align: center;
    margin-top: 12px;
}

@media (max-width: 760px) {
    #hero {
        padding: 22px 18px;
    }

    #hero h1 {
        font-size: 24px;
    }

    .panel {
        padding: 16px;
    }

    .result-main {
        display: block;
    }

    .probability {
        margin-top: 8px;
    }
}
"""


def _sample_to_outputs(sample: tuple[float, ...]) -> tuple[float, ...]:
    return sample


def _risk_class(level: str) -> str:
    if level == "低风险":
        return "low"
    if level == "中风险":
        return "medium"
    return "high"


def _format_result_card(level: str, probability_text: str) -> str:
    probability_value = float(probability_text.rstrip("%"))
    risk_class = _risk_class(level)
    return f"""
    <div class="risk-card {risk_class}">
        <div class="result-label">预测结果</div>
        <div class="result-main">
            <div class="risk-name">{level}</div>
            <div class="probability">{probability_text}</div>
        </div>
        <div class="meter">
            <span style="width: {probability_value}%;"></span>
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
) -> tuple[str, str]:
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
    return _format_result_card(level, probability_text), explanation


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="糖尿病风险预测问诊系统") as demo:
        gr.Markdown(
            """
            # 基于机器学习的糖尿病风险预测问诊系统
            面向课程项目展示的交互式问诊 Demo。输入身体指标后，系统会输出风险等级、预测概率和简要解释。
            """,
            elem_id="hero",
        )

        with gr.Row():
            with gr.Column(scale=3):
                with gr.Column(elem_classes="panel", elem_id="sample-panel"):
                    gr.Markdown("载入示例病例", elem_classes="section-title")
                    gr.Markdown(
                        "答辩展示时可快速载入数据集中的典型样本；真实使用时仍可手动调整指标。",
                        elem_classes="hint",
                    )
                    low_sample_button = gr.Button("低风险示例", variant="secondary")
                    high_sample_button = gr.Button("高风险示例", variant="secondary")
                    reset_button = gr.Button("恢复默认输入", variant="secondary")

                with gr.Column(elem_classes="panel"):
                    gr.Markdown("系统说明", elem_classes="section-title")
                    gr.Markdown(
                        """
                        - 输入项来自糖尿病预测数据集字段。
                        - 页面会优先加载 `models/best_model.pkl`。
                        - 未接入正式模型时，系统使用演示规则函数兜底。
                        """,
                        elem_classes="hint",
                    )

            with gr.Column(scale=7):
                with gr.Row():
                    with gr.Column(elem_classes="panel"):
                        gr.Markdown("基础身体指标", elem_classes="section-title")
                        gr.Markdown("请根据用户情况调整各项数值。", elem_classes="hint")
                        pregnancies = gr.Slider(label="怀孕次数", minimum=0, maximum=20, value=1, step=1)
                        glucose = gr.Slider(label="血糖值", minimum=0, maximum=250, value=120, step=1)
                        blood_pressure = gr.Slider(label="血压", minimum=0, maximum=150, value=72, step=1)
                        skin_thickness = gr.Slider(label="皮肤厚度", minimum=0, maximum=100, value=20, step=1)

                    with gr.Column(elem_classes="panel"):
                        gr.Markdown("代谢与风险相关指标", elem_classes="section-title")
                        gr.Markdown("系统会综合这些特征生成风险概率和解释。", elem_classes="hint")
                        insulin = gr.Slider(label="胰岛素", minimum=0, maximum=900, value=80, step=1)
                        bmi = gr.Slider(label="BMI", minimum=0, maximum=70, value=25.0, step=0.1)
                        diabetes_pedigree = gr.Slider(
                            label="糖尿病家族遗传指数",
                            minimum=0,
                            maximum=3,
                            value=0.5,
                            step=0.01,
                        )
                        age = gr.Slider(label="年龄", minimum=1, maximum=100, value=35, step=1)

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

        predict_button = gr.Button("开始预测", variant="primary", elem_id="predict-button")

        with gr.Column(elem_classes="panel"):
            gr.Markdown("预测结果", elem_classes="section-title")
            result_card = gr.HTML(
                """
                <div class="result-placeholder">
                    调整输入指标后点击“开始预测”，这里会展示风险等级和预测概率。
                </div>
                """
            )
            explanation = gr.Textbox(label="结果解释", lines=8, elem_id="explanation")

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
            outputs=[result_card, explanation],
        )

        gr.Markdown(
            "本系统仅用于机器学习课程项目演示，不能替代医生诊断。",
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
