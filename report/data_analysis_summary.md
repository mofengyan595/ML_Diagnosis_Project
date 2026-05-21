# 成员 A 数据分析摘要

## 数据集基本信息

- 数据文件：`data/diabetes.csv`
- 样本数量：768
- 字段数量：9
- 问题类型：二分类问题，预测是否存在糖尿病风险。

## 字段含义

- `Pregnancies`：怀孕次数，表示患者既往怀孕的次数。0 是合理取值。
- `Glucose`：口服葡萄糖耐量测试中的血糖浓度，是糖尿病风险预测的重要指标。0 在医学上不合理。
- `BloodPressure`：舒张压，单位通常为 mmHg。0 在医学上不合理。
- `SkinThickness`：三头肌皮褶厚度，反映皮下脂肪情况。0 通常代表缺失测量。
- `Insulin`：2 小时血清胰岛素水平。0 通常代表缺失测量。
- `BMI`：身体质量指数，计算公式为体重/身高平方。0 在医学上不合理。
- `DiabetesPedigreeFunction`：糖尿病家族遗传指数，数值越高表示家族遗传风险越高。
- `Age`：年龄，单位为岁。
- `Outcome`：分类标签，0 表示未患病或低风险，1 表示患病或高风险。

## 缺失值检查

| 字段 | 缺失值数量 |
| --- | ---: |
| Pregnancies | 0 |
| Glucose | 0 |
| BloodPressure | 0 |
| SkinThickness | 0 |
| Insulin | 0 |
| BMI | 0 |
| DiabetesPedigreeFunction | 0 |
| Age | 0 |
| Outcome | 0 |

原始 CSV 中没有显式空缺失值，但部分医学指标存在不合理的 0 值，应在预处理阶段按缺失值处理。

## 0 值与异常说明

| 字段 | 0 值数量 |
| --- | ---: |
| Pregnancies | 111 |
| Glucose | 5 |
| BloodPressure | 35 |
| SkinThickness | 227 |
| Insulin | 374 |
| BMI | 11 |
| DiabetesPedigreeFunction | 0 |
| Age | 0 |
| Outcome | 500 |

医学上不合理的 0 值字段如下：

| 字段 | 不合理 0 值数量 |
| --- | ---: |
| Glucose | 5 |
| BloodPressure | 35 |
| SkinThickness | 227 |
| Insulin | 374 |
| BMI | 11 |

`Pregnancies=0` 表示未怀孕过，`Outcome=0` 表示未患病或低风险，二者不是异常值。

## 类别分布

| 原始标签 | 风险含义 | 样本数量 |
| --- | --- | ---: |
| 0 | 未患病/低风险 | 500 |
| 1 | 患病/高风险 | 268 |

- 低风险/未患病比例：65.10%
- 高风险/患病比例：34.90%
- 说明：模型训练仍使用原始二分类标签 0/1；低/中/高风险等级由预测概率在系统层进一步划分。

## 已生成图表

- `figures/class_distribution.png`：类别分布图
- `figures/feature_distribution.png`：特征分布图
- `figures/correlation_heatmap.png`：相关性热力图
