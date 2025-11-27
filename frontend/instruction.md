# 前端 Demo 任务说明（v2）

## 1. 准备可下载的样例 CSV
1. 从 `assets/raw` 中挑选 **两个不同化学体系** 的电池（例如：`LCO/Capacity_25C/charge.csv` 与 `LFP/lfp_cell_1/charge.csv`）。
2. 使用下方脚本将原始 CSV 规整为 100 行左右的示例文件（按照 `sample_index` 取等距行，保留主要列）：
   ```bash
   .venv\Scripts\python.exe -c "import pandas as pd; import numpy as np; \
from pathlib import Path; src = Path('assets/raw/LCO/Capacity_25C/charge.csv'); \
df = pd.read_csv(src); idx = np.linspace(0, len(df)-1, 100, dtype=int); \
df.iloc[idx].to_csv('frontend/battery-best/public/datasets/LCO_sample.csv', index=False)"
   ```
   - 对第二个化学体系重复上述步骤，命名为 `XXX_sample.csv`。
3. 在 React App 的 “Need a dataset to try?” 区域提供这两个示例文件的下载链接（引用 `public/datasets/*.csv`）。

## 2. 上传与解析
1. 用户点击 “上传 CSV”，前端用 `PapaParse`（或 `FileReader + d3-dsv`）读取原始 100 点文件。
2. 在解析阶段校验必需字段：`cycle_index, sample_index, normalized_time, voltage_v, c_rate, temperature_k`。

## 3. 特征转换（与 Python 版本保持一致）
1. 将上传的 CSV 拆成 charge / discharge（或根据文件名判断，如包含 `charge` 则走充电分支）。
2. 对以下字段计算统计量：`mean / std / min / max`。
   - Charge: `voltage_v`, `c_rate`, `temperature_k`
   - Discharge: `voltage_v`, `c_rate`
3. 生成 11 维特征向量（顺序必须与 `logreg_model.json` 的 `feature_names` 一致）：
   ```
   charge_voltage_v_mean
   charge_voltage_v_std
   charge_voltage_v_min
   charge_voltage_v_max
   charge_c_rate_mean
   charge_temperature_k_mean
   discharge_voltage_v_mean
   discharge_voltage_v_std
   discharge_voltage_v_min
   discharge_voltage_v_max
   discharge_c_rate_mean
   ```

## 4. 预测（前端 JavaScript）
1. React 在初始化时 `import logregPredict from "./utils/logregPredict"`.
2. 将上一步得到的特征对象传入：
   ```ts
   const result = predictChemistry(featureMap); // 返回 { label, probability, probabilities }
   ```
3. 在 UI 中展示：
   - 预测化学体系（`result.label`）
   - 对应置信度（`(result.probability*100).toFixed(2)%`）
   - 可选：列出四个类别的概率条形图。

## 5. 交互流程
1. 用户下载示例 CSV，或上传自己的原始 CSV。
2. 上传后立即解析 → 特征转换 → 调用 `predictChemistry`。
3. 显示预测结果；允许用户再次上传以比较不同电池。
4. 在日志区/弹窗提示如果必需列缺失或无法解析。

> **备注**：如需在前端直接处理原始上千行 CSV，可复用 `src/prepare_logreg_dataset.py` 的逻辑在后端提供 API；当前方案假设上传文件已经是 100 行左右的规整数据，便于在浏览器内完成全部计算。***