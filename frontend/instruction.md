# 前端 Demo 任务说明（v2）

## 1. 准备可下载的样例 CSV
1. 从 `assets/raw` 中挑选 **任意两个不同化学体系** 的电池原始 CSV（charge/discharge 均可，不需要额外 API）。
2. 使用下方脚本将原始 CSV 规整为 100 行左右的示例文件（按照 `sample_index` 取等距行，保留主要列）：
   ```bash
   .venv\Scripts\python.exe -c "import pandas as pd; import numpy as np; \
from pathlib import Path; src = Path('assets/raw/LCO/Capacity_25C/charge.csv'); \
df = pd.read_csv(src); idx = np.linspace(0, len(df)-1, 100, dtype=int); \
df.iloc[idx].to_csv('frontend/battery-best/public/datasets/LCO_sample.csv', index=False)"
   ```
   - 对第二个化学体系重复上述步骤，命名为 `XXX_sample.csv`。
3. 在 React App 的 “Need a dataset to try?” 区域提供这两个示例文件的下载链接（引用 `public/datasets/*.csv`）。

## 2. 上传与解析（纯前端）
1. 用户点击 “上传 CSV”，直接从本地选择 `assets/raw` 下任意文件；前端使用 `PapaParse`（或 `FileReader + d3-dsv`）读取原始数据。
2. 在浏览器中校验必需字段：`cycle_index, sample_index, normalized_time, voltage_v, c_rate, temperature_k`。整个过程无需服务器或额外 API。

## 3. 特征转换（完全在前端）
1. 解析到的原始 CSV 在前端拆分 charge / discharge（可根据文件名或 `current_a` 符号判断）。
2. 在浏览器里计算以下字段的 `mean / std / min / max`：
   - Charge: `voltage_v`, `c_rate`, `temperature_k`
   - Discharge: `voltage_v`, `c_rate`
3. 在前端生成 11 维特征向量（顺序必须与 `logreg_model.json` 的 `feature_names` 一致）：
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

## 4. 预测（前端 JavaScript，无需后端）
1. React 初始化时 `import logregPredict from "./utils/logregPredict"`。
2. 将上一步的特征对象传入 `predictChemistry`。模型参数来自 `logreg_model.json`，直接随前端打包。
3. 在 UI 中展示预测标签与概率（同前）。

## 5. 交互流程
1. 用户下载示例 CSV，或上传自己的原始 CSV。
2. 上传后立即解析 → 特征转换 → 调用 `predictChemistry`。
3. 显示预测结果；允许用户再次上传以比较不同电池。
4. 在日志区/弹窗提示如果必需列缺失或无法解析。

> **备注**：当前方案假设所有操作都在前端完成：用户直接从 `assets/raw` 复制 CSV 后上传，浏览器完成采样、特征提取、模型预测，无需额外后端服务。*** 