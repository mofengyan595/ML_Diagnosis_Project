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
python -c "import gradio, pandas, sklearn, joblib, xgboost, lightgbm; print('ok')"
```

运行成员 B 多模型训练与对比：

```bash
python src/train_models.py
python src/compare_models.py
```

运行成员 C 调参与最终评估：

```bash
python src/tune_model.py
python src/evaluate_model.py
python src/explain_model.py
```

## 当前进度

| 模块 | 状态 | 当前产物 | 下一步 |
| --- | --- | --- | --- |
| 数据理解与预处理 | 已完成 baseline 阶段 | 清洗数据、标准化数据、训练/验证/测试集、数据图表 | 后续成员沿用统一 split |
| 基准模型 | 已完成 | Logistic Regression baseline、baseline 指标、特征影响排序 | 作为后续模型对比基准 |
| 多模型训练与对比 | 已完成 | 多模型训练脚本、候选模型字典、模型对比结果表、test 集对比图 | 作为最终调参候选来源 |
| 调参与最终评估 | 已完成 | GridSearchCV 调参、最终模型、最佳参数、最终评估表、误判分析、ROC/混淆矩阵/特征重要性图 | 后续补充报告实验分析 |
| 系统 Demo 与工程整合 | 已接入最终模型 | Gradio 页面、预测接口、最终模型加载、异常输入校验、风险建议 | 后续可补充 Demo 截图 |

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
|   |-- model_comparison.png       # 成员 B 输出的 test 集模型性能对比图
|-- models/                       # 预处理工具与模型文件
|   |-- imputer.pkl
|   |-- iqr_bounds.pkl
|   |-- scaler.pkl
|   |-- baseline_logistic_regression.pkl
|   |-- trained_models.pkl        # 成员 B 训练得到的候选模型字典
|   |-- best_model.pkl            # 成员 C 调参后输出的最终模型
|-- report/                       # 报告、PPT、展示说明材料
|-- src/                          # 源代码
|   |-- data_analysis.py          # 数据探索分析与图表
|   |-- data_process.py           # 数据清洗、划分、标准化
|   |-- baseline_model.py         # 基准模型训练与结果输出
|   |-- train_models.py           # 成员 B 多模型训练与结果输出
|   |-- compare_models.py         # 成员 B 模型对比图生成
|   |-- evaluation_utils.py       # 成员 C 评估、调参、路径与模型工具函数
|   |-- tune_model.py             # 成员 C 参数调优与最终模型保存
|   |-- evaluate_model.py         # 成员 C 最终指标、预测明细、混淆矩阵与 ROC 曲线
|   |-- explain_model.py          # 成员 C 特征重要性分析
|   |-- app.py                    # Gradio 系统 Demo 入口
|   |-- config.py                 # 路径、字段、模型文件配置
|   |-- predict.py                # 预测流程与模型加载
|-- baseline_result.csv           # 基准模型指标
|-- baseline_prediction_details.csv
|-- baseline_feature_importance.csv
|-- model_comparison_result.csv   # 成员 B 模型对比结果表
|-- tuning_results.csv            # 成员 C 调参结果
|-- best_params.json              # 成员 C 最终模型参数与选择依据
|-- final_metrics.csv             # 成员 C 最终模型验证集/测试集指标
|-- final_prediction_details.csv  # 成员 C 最终模型预测明细
|-- error_analysis.csv            # 成员 C 测试集误判样本分析
|-- feature_importance.csv        # 成员 C 最终模型特征重要性
|-- environment.yml
|-- requirements.txt
|-- README.md
```

## 数据说明

原始数据：

```text
data/diabetes.csv
```

数据集来源：Kaggle Pima Indians Diabetes Database  
https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

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
figures/model_comparison.png
figures/confusion_matrix.png
figures/roc_curve.png
figures/feature_importance.png
```

模型与预处理工具：

```text
models/imputer.pkl
models/iqr_bounds.pkl
models/scaler.pkl
models/baseline_logistic_regression.pkl
models/trained_models.pkl
models/best_model.pkl
```

结果表：

```text
baseline_result.csv
baseline_prediction_details.csv
baseline_feature_importance.csv
model_comparison_result.csv
tuning_results.csv
best_params.json
final_metrics.csv
final_prediction_details.csv
error_analysis.csv
feature_importance.csv
```

成员 B 产物：

```text
src/train_models.py
src/compare_models.py
model_comparison_result.csv
models/trained_models.pkl
figures/model_comparison.png
```

当前 baseline 指标：

| 数据集 | Accuracy | Precision | Recall | F1 | ROC AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| validation | 0.7642 | 0.6346 | 0.7674 | 0.6947 | 0.8709 |
| test | 0.7273 | 0.5938 | 0.7037 | 0.6441 | 0.8154 |

当前最终模型为阈值优化后的 SVM，使用验证集选择分类阈值，最终保存为 `models/best_model.pkl`。

| 数据集 | Accuracy | Precision | Recall | F1 | ROC AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| validation | 0.7886 | 0.6545 | 0.8372 | 0.7347 | 0.8762 |
| test | 0.7403 | 0.5972 | 0.7963 | 0.6825 | 0.8215 |

## 各成员下一步

### 多模型训练与对比

该模块已完成。成员 B 直接读取：

```text
data/train_processed.csv
data/val_processed.csv
data/test_processed.csv
baseline_result.csv
```

已完成：

- 使用相同 train/val/test 划分与 baseline 对比
- 训练 Logistic Regression baseline、KNN、Decision Tree、Random Forest、SVM、Gradient Boosting、XGBoost、LightGBM
- 输出模型对比结果表 `model_comparison_result.csv`
- 保存候选模型字典 `models/trained_models.pkl`
- 绘制模型性能对比图 `figures/model_comparison.png`

当前结果显示，SVM 在 test 集 recall 和 F1 上最好，Random Forest 在 test 集 ROC AUC 上最好。成员 C 已基于验证集结果进行最终调参，避免使用测试集参与模型选择。

### 调参与最终评估

该模块已完成，沿用：

```text
data/train_processed.csv
data/val_processed.csv
data/test_processed.csv
```

调参阶段未使用测试集进行模型选择，测试集只用于最终评估。

已完成：

- 使用 GridSearchCV 调参
- 使用 5 折 StratifiedKFold 交叉验证
- 计算 Accuracy、Precision、Recall、F1、ROC AUC
- 绘制混淆矩阵、ROC 曲线、特征重要性图
- 基于验证集 F1、Recall 和 ROC AUC 选择最终模型
- 保存最终模型与最佳参数

最终模型已保存为：

```text
models/best_model.pkl
```

成员 C 产物：

```text
src/evaluation_utils.py
src/tune_model.py
src/evaluate_model.py
src/explain_model.py
best_params.json
tuning_results.csv
final_metrics.csv
final_prediction_details.csv
error_analysis.csv
feature_importance.csv
figures/confusion_matrix.png
figures/roc_curve.png
figures/feature_importance.png
```

### 系统整合与展示

系统当前已能加载最终模型：

```text
models/best_model.pkl
```

页面预测会自动优先使用最终模型，并在预测前检查输入是否明显异常。若血糖、血压、BMI、皮肤厚度、胰岛素等关键医学指标为 0 或超出合理范围，系统会提示修正输入，不会直接输出风险概率。

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
-> 检查输入范围，明显异常时提示修正
-> 处理不合理 0 值
-> 中位数填充
-> IQR 异常值裁剪（边界基于训练集有效观测值）
-> 标准化
-> 模型预测
```

当前 Demo 使用的输入校验范围：

| 字段 | 合理范围 | 单位 |
| --- | ---: | --- |
| Pregnancies | 0-20 | 次 |
| Glucose | 40-300 | mg/dL |
| BloodPressure | 40-220 | mmHg |
| SkinThickness | 1-100 | mm |
| Insulin | 1-900 | uU/mL |
| BMI | 10-70 | kg/m^2 |
| DiabetesPedigreeFunction | 0.01-3 | - |
| Age | 18-100 | 岁 |

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
python -c "import gradio, pandas, sklearn, joblib, xgboost, lightgbm; print('ok')"
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
