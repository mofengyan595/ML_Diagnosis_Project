# ML_Diagnosis_Project

本项目为机器学习课程期末项目，主题为 **基于机器学习的糖尿病风险预测问诊系统**。

系统基于用户输入的身体指标进行风险预测，并通过 Gradio 构建交互式网页 Demo，展示糖尿病风险等级、预测概率和结果解释。本项目仅用于课程实验与演示，不能替代医生诊断。

## 项目结构

```text
ML_Diagnosis_Project/
|-- data/                  # 数据集文件
|   |-- diabetes.csv
|-- figures/               # 图表、系统截图
|-- models/                # 训练完成的模型文件
|   |-- best_model.pkl     # 最终模型，后续由模型训练同学提供
|   |-- scaler.pkl         # 标准化器，如模型需要则提供
|-- report/                # 报告、PPT、展示说明材料
|-- src/                   # 源代码
|   |-- app.py             # Gradio 系统 Demo 入口
|   |-- config.py          # 特征名、模型路径等配置
|   |-- predict.py         # 模型加载与预测函数
|-- environment.yml        # Conda 环境配置
|-- requirements.txt       # pip 依赖列表
|-- README.md              # 项目说明文档
```

## 环境配置

推荐使用 Python 3.10。

进入项目目录：

```bash
cd ML_Diagnosis_Project
```

使用 Conda 创建环境：

```bash
conda env create -f environment.yml
conda activate ml-diagnosis
```

如果已经创建过环境，只需要激活：

```bash
conda activate ml-diagnosis
```

也可以使用 pip 安装依赖：

```bash
pip install -r requirements.txt
```

检查环境是否可用：

```bash
python -c "import gradio, pandas, sklearn, joblib; print('ok')"
```

## 协作开发流程

开始新任务前，先同步主分支：

```bash
git checkout main
git pull origin main
```

从最新 `main` 新建自己的功能分支：

```bash
git checkout -b feature/your-task-name
```

完成修改后提交：

```bash
git status
git add .
git commit -m "简要说明本次修改"
git push -u origin feature/your-task-name
```

然后在 GitHub 上创建 Pull Request，请求合并到 `main`。

建议分支命名：

```text
feature/data-process       # 数据理解、预处理、基准模型
feature/model-training     # 多模型训练与对比
feature/evaluation         # 调参、评估、解释与图表
feature/app                # 系统 Demo 与工程整合
```

## 数据集说明

当前数据集路径：

```text
data/diabetes.csv
```

字段包括：

```text
Pregnancies
Glucose
BloodPressure
SkinThickness
Insulin
BMI
DiabetesPedigreeFunction
Age
Outcome
```

其中 `Outcome` 为标签列：

```text
0 = 低风险 / 未患病
1 = 高风险 / 患病
```

系统 Demo 的输入字段与数据集特征字段保持一致，预测时不使用 `Outcome`。

## 各模块开始方式

### 数据处理与基准模型

建议负责内容：

- 读取 `data/diabetes.csv`
- 检查缺失值、异常值和类别分布
- 完成训练集 / 测试集划分
- 完成特征标准化
- 训练 Logistic Regression 作为基准模型
- 输出数据字段说明、缺失值统计图、类别分布图、特征分布图

建议新增或维护的文件：

```text
src/data_preprocess.py
src/train_baseline.py
figures/class_distribution.png
figures/feature_distribution.png
```

### 多模型训练与模型对比

建议负责内容：

- 训练 KNN、Decision Tree、Random Forest、SVM 等模型
- 如环境允许，可加入 XGBoost 或 LightGBM
- 使用统一训练 / 测试集进行比较
- 输出模型性能对比表和柱状图

建议新增或维护的文件：

```text
src/train_models.py
figures/model_comparison.png
report/model_comparison.md
```

### 超参数调优、评估与解释

建议负责内容：

- 使用 GridSearchCV 或 RandomizedSearchCV 调参
- 计算 Accuracy、Precision、Recall、F1、AUC
- 绘制混淆矩阵、ROC 曲线、特征重要性图
- 给出最终模型选择理由

建议新增或维护的文件：

```text
src/evaluate.py
figures/confusion_matrix.png
figures/roc_curve.png
figures/feature_importance.png
report/evaluation_summary.md
```

## 模型输出约定

最终用于系统 Demo 的模型文件请保存为：

```text
models/best_model.pkl
```

如果模型训练时使用了标准化器，请保存为：

```text
models/scaler.pkl
```

模型输入特征顺序必须与数据集保持一致：

```text
Pregnancies
Glucose
BloodPressure
SkinThickness
Insulin
BMI
DiabetesPedigreeFunction
Age
```

如果使用 scikit-learn 模型，建议通过 `joblib.dump()` 保存：

```python
import joblib

joblib.dump(model, "models/best_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
```

## 运行系统 Demo

在项目根目录下运行：

```bash
python src/app.py
```

终端出现本地地址后，在浏览器打开：

```text
http://127.0.0.1:7860
```

页面支持：

- 手动输入身体指标
- 一键载入低风险示例病例
- 一键载入高风险示例病例
- 输出风险等级
- 输出预测概率
- 输出结果解释

## 模型接入方式

系统会优先尝试加载以下文件：

```text
models/best_model.pkl
models/scaler.pkl
```

如果 `models/best_model.pkl` 存在，系统会使用该模型进行预测。

如果 `models/scaler.pkl` 存在，系统会先对输入特征进行标准化，再传入模型。

如果暂时没有正式模型，系统会使用 `predict.py` 中的演示规则函数生成预测结果，以保证 Demo 可以正常运行和展示。

当前 Demo 已实现从用户输入到风险预测结果展示的完整交互流程。后续只需将训练好的模型文件放入 `models/` 目录，即可完成正式模型接入。

## 提交前检查

提交代码前建议运行：

```bash
python -c "import pandas as pd; df = pd.read_csv('data/diabetes.csv'); print(df.shape)"
python src/app.py
```

确认系统 Demo 能正常打开，并且没有提交临时缓存文件，例如：

```text
__pycache__/
*.pyc
```

## 注意事项

本系统仅用于机器学习课程项目演示，预测结果不具备医学诊断效力。如需真实医疗判断，请咨询专业医生。
