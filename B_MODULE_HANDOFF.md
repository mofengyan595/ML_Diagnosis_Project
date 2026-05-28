# 成员 B 模块交付说明：多模型训练与模型对比

## 1. 本模块完成的任务

本模块完成了成员 B 负责的多模型训练与模型对比工作：

- 读取 A 已处理好的 train/validation/test 数据；
- 训练 KNN、Decision Tree、Random Forest、SVM 四个候选模型；
- 合并 A 的 Logistic Regression baseline；
- 在 validation 和 test 集上统一计算 accuracy、precision、recall、f1、roc_auc；
- 保存模型对比结果；
- 保存候选模型字典；
- 生成 test 集模型性能对比图。

## 2. 使用的数据

本模块直接读取：

- `data/train_processed.csv`
- `data/val_processed.csv`
- `data/test_processed.csv`

说明：

- 标签列为 `Outcome`；
- 其余 8 列为输入特征；
- 本模块没有重新划分数据集；
- 这样保证 B 的模型结果可以和 A 的 Logistic Regression baseline 公平比较。

## 3. 主要代码文件

- `src/train_models.py`

  用于训练多个模型、计算 validation/test 指标、合并 `baseline_result.csv`、保存 `model_comparison_result.csv` 和 `models/trained_models.pkl`。

- `src/compare_models.py`

  用于读取 `model_comparison_result.csv`，并生成 `figures/model_comparison.png`。

## 4. 输出文件说明

- `model_comparison_result.csv`

  保存 Logistic Regression、KNN、Decision Tree、Random Forest、SVM 在 validation/test 上的 accuracy、precision、recall、f1、roc_auc。

- `models/trained_models.pkl`

  保存 B 训练得到的候选模型字典，包含 KNN、Decision Tree、Random Forest、SVM。

- `figures/model_comparison.png`

  test 集模型性能对比图，展示 accuracy、recall、f1、roc_auc。

注意：

- 本模块没有生成 `models/best_model.pkl`。
- `models/best_model.pkl` 应由成员 C 在调参和最终模型选择后生成。

## 5. 运行方式

运行顺序如下：

```bash
python src/train_models.py
python src/compare_models.py
```

说明：

- 先运行 `train_models.py`，生成模型结果和候选模型；
- 再运行 `compare_models.py`，生成模型对比图。

## 6. 当前模型结果摘要

根据当前 `model_comparison_result.csv`：

- test 集 accuracy 最高的模型：SVM，accuracy = 0.7468；
- test 集 recall 最高的模型：SVM，recall = 0.7778；
- test 集 f1 最高的模型：SVM，f1 = 0.6829；
- test 集 roc_auc 最高的模型：Random Forest，roc_auc = 0.8275。

完整结果表如下：

| model | split | accuracy | precision | recall | f1 | roc_auc |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | validation | 0.7642 | 0.6346 | 0.7674 | 0.6947 | 0.8709 |
| Logistic Regression | test | 0.7273 | 0.5938 | 0.7037 | 0.6441 | 0.8154 |
| KNN | validation | 0.7805 | 0.7105 | 0.6279 | 0.6667 | 0.8581 |
| KNN | test | 0.7273 | 0.6154 | 0.5926 | 0.6038 | 0.7675 |
| Decision Tree | validation | 0.7886 | 0.7429 | 0.6047 | 0.6667 | 0.7461 |
| Decision Tree | test | 0.6948 | 0.5714 | 0.5185 | 0.5437 | 0.6543 |
| Random Forest | validation | 0.7967 | 0.7368 | 0.6512 | 0.6914 | 0.8365 |
| Random Forest | test | 0.7338 | 0.6444 | 0.5370 | 0.5859 | 0.8275 |
| SVM | validation | 0.7967 | 0.6667 | 0.8372 | 0.7423 | 0.8759 |
| SVM | test | 0.7468 | 0.6087 | 0.7778 | 0.6829 | 0.8157 |

## 7. 结果分析与给成员 C 的建议

从当前结果看，SVM 在 test 集上的 recall 和 f1 表现最好，适合作为后续重点调参候选模型。Random Forest 的 roc_auc 最高，说明其整体区分正负样本能力较好，也适合后续调参。Logistic Regression baseline 表现稳定，可以作为重要基准。Decision Tree 默认参数下表现相对较弱，不建议作为主要优化对象。

给成员 C 的建议：

- 重点调参 SVM 和 Random Forest；
- SVM 可调参数包括 `C`、`kernel`、`gamma`、`class_weight`；
- Random Forest 可调参数包括 `n_estimators`、`max_depth`、`min_samples_split`、`min_samples_leaf`、`max_features`、`class_weight`；
- 糖尿病风险预测任务更关注 recall 和 f1，因为漏判高风险样本的代价更高；
- 后续可以进一步结合 GridSearchCV / RandomizedSearchCV 和阈值调整。

## 8. 注意事项

- 不要重新划分数据集；
- 不要覆盖 A 的数据处理产物；
- 不要修改 `baseline_result.csv`；
- 不要直接使用 test 集做调参；
- C 最终调参后再保存 `models/best_model.pkl`；
- D 的系统会优先加载 `models/best_model.pkl`，所以在 C 完成前不要随意生成该文件。
