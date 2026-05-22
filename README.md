# ML_Diagnosis_Project

本项目为机器学习课程期末项目，主题为 **基于机器学习的糖尿病风险预测问诊系统**。

系统基于用户输入的身体指标进行糖尿病风险预测，并通过 Gradio 网页 Demo 展示风险等级、预测概率和结果解释。本项目仅用于课程实验与展示，不能替代医生诊断。

## 快速开始

推荐使用 Python 3.10。

```bash
cd ML_Diagnosis_Project
conda activate ml-diagnosis
python src/app.py
```

启动后在浏览器打开：

```text
http://127.0.0.1:7860
```

如果尚未创建环境：

```bash
conda env create -f environment.yml
conda activate ml-diagnosis
```

也可以使用 pip 安装依赖：

```bash
pip install -r requirements.txt
```

检查环境：

```bash
python -c "import gradio, pandas, sklearn, joblib; print('ok')"
```

## 当前进度

| 模块 | 状态 | 当前产物 | 下一步 |
| --- | --- | --- | --- |
| 数据理解与预处理 | 已完成 baseline 阶段 | 清洗数据、标准化数据、训练/验证/测试集、数据图表 | 后续成员沿用统一 split |
| 基准模型 | 已完成 | Logistic Regression baseline、baseline 指标、特征影响排序 | 作为后续模型对比基准 |
| 多模型训练与对比 | 待完成 | 暂无 | 训练 KNN、Decision Tree、Random Forest、SVM 等 |
| 调参与最终评估 | 待完成 | 暂无 | 调参、绘制 ROC/混淆矩阵、保存最终模型 |
| 系统 Demo 与工程整合 | 已接入 baseline | Gradio 页面、预测接口、baseline 模型加载 | 等待 `models/best_model.pkl` 后自动切换最终模型 |

## 项目结构

```text
ML_Diagnosis_Project/
|-- data/                         # 数据集与处理后数据
|   |-- diabetes.csv              # 原始数据
|   |-- cleaned_data.csv          # 清洗后但未标准化的数据
|   |-- processed_data.csv        # 清洗并标准化后的完整数据
|   |-- train_processed.csv       # 训练集
|   |-- val_processed.csv         # 验证集
|   |-- test_processed.csv        # 测试集
|-- figures/                      # 图表与系统截图
|-- models/                       # 预处理工具与模型文件
|   |-- imputer.pkl
|   |-- iqr_bounds.pkl
|   |-- scaler.pkl
|   |-- baseline_logistic_regression.pkl
|   |-- best_model.pkl            # 最终模型，后续由调参评估阶段产出
|-- report/                       # 报告、PPT、展示说明材料
|-- src/                          # 源代码
|   |-- data_analysis.py          # 数据探索分析与图表
|   |-- data_process.py           # 数据清洗、划分、标准化
|   |-- baseline_model.py         # 基准模型训练与结果输出
|   |-- app.py                    # Gradio 系统 Demo 入口
|   |-- config.py                 # 路径、字段、模型文件配置
|   |-- predict.py                # 预测流程与模型加载
|-- baseline_result.csv           # 基准模型指标
|-- baseline_prediction_details.csv
|-- baseline_feature_importance.csv
|-- environment.yml
|-- requirements.txt
|-- README.md
```

## 数据说明

原始数据：

```text
data/diabetes.csv
```

字段：

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

`Outcome` 是标签列：

```text
0 = 低风险 / 未患病
1 = 高风险 / 患病
```

后续模型训练建议直接使用统一划分后的数据，避免不同成员重复划分导致结果不可比：

```text
data/train_processed.csv
data/val_processed.csv
data/test_processed.csv
```

## 已有产物

数据产物：

```text
data/cleaned_data.csv
data/processed_data.csv
data/train_processed.csv
data/val_processed.csv
data/test_processed.csv
```

图表产物：

```text
figures/class_distribution.png
figures/feature_distribution.png
figures/correlation_heatmap.png
figures/baseline_feature_importance.png
```

模型与预处理工具：

```text
models/imputer.pkl
models/iqr_bounds.pkl
models/scaler.pkl
models/baseline_logistic_regression.pkl
```

结果表：

```text
baseline_result.csv
baseline_prediction_details.csv
baseline_feature_importance.csv
```

当前 baseline 指标：

| 数据集 | Accuracy | Precision | Recall | F1 | ROC AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| validation | 0.7967 | 0.6800 | 0.7907 | 0.7312 | 0.8756 |
| test | 0.6948 | 0.5538 | 0.6667 | 0.6050 | 0.8087 |

## 各成员下一步

### 多模型训练与对比

建议直接读取：

```text
data/train_processed.csv
data/val_processed.csv
data/test_processed.csv
baseline_result.csv
```

建议完成：

- 训练 KNN、Decision Tree、Random Forest、SVM 等模型
- 使用相同 train/val/test 划分与 baseline 对比
- 输出模型对比结果表
- 绘制模型性能对比图

建议新增或维护：

```text
src/train_models.py
figures/model_comparison.png
report/model_comparison.md
```

### 调参与最终评估

建议继续沿用：

```text
data/train_processed.csv
data/val_processed.csv
data/test_processed.csv
```

注意：调参阶段不要提前使用测试集，测试集建议只在最终模型确定后使用。

建议完成：

- 使用 GridSearchCV 或 RandomizedSearchCV 调参
- 计算 Accuracy、Precision、Recall、F1、ROC AUC
- 绘制混淆矩阵、ROC 曲线、特征重要性图
- 给出最终模型选择理由
- 保存最终模型

最终模型请保存为：

```text
models/best_model.pkl
```

建议新增或维护：

```text
src/evaluate.py
figures/confusion_matrix.png
figures/roc_curve.png
figures/feature_importance.png
report/evaluation_summary.md
```

### 系统整合与展示

系统当前已能加载 baseline 模型。后续只要最终模型保存为：

```text
models/best_model.pkl
```

页面预测会自动优先使用最终模型。

建议继续补充：

```text
figures/demo_low_risk.png
figures/demo_high_risk.png
report/system_integration_notes.md
```

## 模型接入约定

系统预测按以下优先级加载模型：

```text
1. models/best_model.pkl
2. models/baseline_logistic_regression.pkl
3. predict.py 中的演示规则函数
```

系统预测时会配合使用：

```text
models/imputer.pkl
models/iqr_bounds.pkl
models/scaler.pkl
```

预测流程：

```text
用户输入原始指标
-> 处理不合理 0 值
-> 中位数填充
-> IQR 异常值裁剪
-> 标准化
-> 模型预测
```

模型输入特征顺序必须保持一致：

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

保存模型示例：

```python
import joblib

joblib.dump(model, "models/best_model.pkl")
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
feature/data-process
feature/model-training
feature/evaluation
feature/app
```

## 提交前检查

提交代码前建议运行：

```bash
python -c "import pandas as pd; df = pd.read_csv('data/diabetes.csv'); print(df.shape)"
python -c "import gradio, pandas, sklearn, joblib; print('ok')"
python src/app.py
```

确认：

- Demo 能正常打开
- 低风险 / 高风险示例能正常预测
- 不提交临时缓存文件

不要提交：

```text
__pycache__/
*.pyc
```

## 注意事项

本系统仅用于机器学习课程项目演示，预测结果不具备医学诊断效力。如需真实医疗判断，请咨询专业医生。
