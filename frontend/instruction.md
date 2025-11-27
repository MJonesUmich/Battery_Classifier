# 前端 Demo 任务说明（v2）

## 1. 准备可下载的样例 CSV
1. 使用脚本 `src/create_demo_datasets.py` 可直接生成 demo 数据。脚本会遍历 `assets/processed/<chemistry>/<battery>` 下所有拥有 charge/discharge 成对 CSV 的目录，再随机挑选 4 个样本生成。
   示例：
   ```bash
   # 生成默认的四个化学体系（LCO / LFP / NCA / NMC）
   .venv\Scripts\python.exe src\create_demo_datasets.py

   # 或随机从候选池里挑 count 个样本
   .venv\Scripts\python.exe src\create_demo_datasets.py --random --count 4
   ```
2. 每个输出文件（如 `LCO_sample.csv`）包含两段数据：前 100 行为 charge，后 100 行为 discharge，同时保留 `phase` 字段（`charge` / `discharge`）以及模型必需列：
   ```
   phase,battery_id,chemistry,cycle_index,sample_index,normalized_time,elapsed_time_s,
   voltage_v,current_a,c_rate,temperature_k
   ```
3. 将生成的 CSV 放在 `frontend/battery-best/public/datasets/`；脚本同时会写入 `datasets.json`，React 页面会自动读取该清单来渲染下载列表。

## 2. 上传与解析（纯前端）
1. 用户点击 “上传 CSV”，可上传上述 demo 文件或任意自备的原始 CSV（建议同一个循环内包含 charge + discharge）。
2. 浏览器端使用 `PapaParse` 读取数据，并校验必需字段：`phase`（或 `current_a` 符号用于推断）、`cycle_index`, `sample_index`, `normalized_time`, `voltage_v`, `c_rate`, `temperature_k` 等，无需后端。

## 3. 特征转换（完全在前端）
1. 根据 `phase` 列（若缺失则根据 `current_a` 正负）在前端拆分 charge / discharge。
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