import gradio as gr

from predict import predict_diabetes_risk


CUSTOM_CSS = """
:root {
    --primary: #2563eb;
    --primary-dark: #1d4ed8;
    --ink: #172033;
    --muted: #64748b;
    --panel: #ffffff;
    --line: #d9e2ef;
    --soft: #f4f7fb;
    --accent: #0f766e;
    --warn: #b45309;
}

.gradio-container {
    max-width: 1180px !important;
    margin: 0 auto !important;
    background: linear-gradient(180deg, #f6f9fc 0%, #eef4fb 100%);
    color: var(--ink);
}

#hero {
    padding: 28px 32px 22px;
    border: 1px solid var(--line);
    border-radius: 8px;
    background:
        linear-gradient(135deg, rgba(37, 99, 235, 0.12), rgba(15, 118, 110, 0.08)),
        #ffffff;
    box-shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
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
    max-width: 780px;
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
    font-weight: 700;
    color: var(--ink);
    margin: 0 0 14px;
}

.hint {
    color: var(--muted);
    font-size: 13px;
    line-height: 1.6;
    margin-bottom: 12px;
}

#predict-button {
    margin-top: 18px;
}

#predict-button button {
    min-height: 46px;
    border-radius: 8px;
    font-weight: 700;
    background: var(--primary);
    border-color: var(--primary);
    box-shadow: 0 12px 24px rgba(37, 99, 235, 0.22);
}

#predict-button button:hover {
    background: var(--primary-dark);
    border-color: var(--primary-dark);
}

#risk-level textarea,
#probability textarea {
    font-size: 22px;
    font-weight: 800;
    text-align: center;
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
}
"""


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="糖尿病风险预测问诊系统") as demo:
        gr.Markdown(
            """
            # 基于机器学习的糖尿病风险预测问诊系统
            面向课程项目展示的交互式风险评估 Demo，整合身体指标输入、模型预测结果和风险解释。
            """,
            elem_id="hero",
        )

        with gr.Row():
            with gr.Column(scale=7, elem_classes="panel"):
                gr.Markdown("基础身体指标", elem_classes="section-title")
                gr.Markdown("请根据样本或用户情况调整各项数值。", elem_classes="hint")
                pregnancies = gr.Slider(label="怀孕次数", minimum=0, maximum=20, value=1, step=1)
                glucose = gr.Slider(label="血糖值", minimum=0, maximum=250, value=120, step=1)
                blood_pressure = gr.Slider(label="血压", minimum=0, maximum=150, value=72, step=1)
                skin_thickness = gr.Slider(label="皮肤厚度", minimum=0, maximum=100, value=20, step=1)

            with gr.Column(scale=7, elem_classes="panel"):
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

        predict_button = gr.Button("开始预测", variant="primary", elem_id="predict-button")

        with gr.Column(elem_classes="panel"):
            gr.Markdown("预测结果", elem_classes="section-title")
            with gr.Row():
                risk_level = gr.Textbox(label="风险等级", elem_id="risk-level")
                probability = gr.Textbox(label="预测概率", elem_id="probability")

            explanation = gr.Textbox(label="结果解释", lines=8, elem_id="explanation")

        gr.Markdown(
            "本系统仅用于机器学习课程项目演示，不能替代医生诊断。",
            elem_classes="footer-note",
        )

        predict_button.click(
            fn=predict_diabetes_risk,
            inputs=[
                pregnancies,
                glucose,
                blood_pressure,
                skin_thickness,
                insulin,
                bmi,
                diabetes_pedigree,
                age,
            ],
            outputs=[risk_level, probability, explanation],
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
