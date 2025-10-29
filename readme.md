using python 3.11

important
regard to down sample: no down sample yet
NCA,LFP low sample size
export_pl_data.m: update the data source

## Matplotlib Figure Standards

All parsers must follow these standards when generating battery cycle plots:

### Standard Figure Format

```python
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for multi-threading
import matplotlib.pyplot as plt

# Charge Plot
plt.figure(figsize=(10, 6))
plt.plot(
    charge_df["Charge_Time(s)"],
    charge_df["Voltage(V)"],
    "b-",
    linewidth=2,
)
plt.xlabel("Charge Time (s)", fontsize=12)
plt.ylabel("Voltage (V)", fontsize=12)
plt.title(f"Cycle {cycle} Charge Profile", fontsize=14)
plt.grid(True, alpha=0.3)
plt.savefig(save_path, dpi=100, bbox_inches="tight")
plt.close()

# Discharge Plot
plt.figure(figsize=(10, 6))
plt.plot(
    discharge_df["Discharge_Time(s)"],
    discharge_df["Voltage(V)"],
    "r-",
    linewidth=2,
)
plt.xlabel("Discharge Time (s)", fontsize=12)
plt.ylabel("Voltage (V)", fontsize=12)
plt.title(f"Cycle {cycle} Discharge Profile", fontsize=14)
plt.grid(True, alpha=0.3)
plt.savefig(save_path, dpi=100, bbox_inches="tight")
plt.close()
```

### Required Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `figsize` | `(10, 6)` | Figure size in inches |
| Line style (charge) | `"b-"` | Blue solid line |
| Line style (discharge) | `"r-"` | Red solid line |
| `linewidth` | `2` | Line width |
| X-axis label fontsize | `12` | Font size for axis labels |
| Y-axis label fontsize | `12` | Font size for axis labels |
| Title fontsize | `14` | Font size for title |
| Grid | `True, alpha=0.3` | Semi-transparent grid |
| DPI | `100` | Resolution |
| `bbox_inches` | `"tight"` | Tight bounding box |

### Axis Labels

- **Charge plots**: 
  - X-axis: `"Charge Time (s)"`
  - Y-axis: `"Voltage (V)"`
  
- **Discharge plots**:
  - X-axis: `"Discharge Time (s)"`
  - Y-axis: `"Voltage (V)"`

### File Naming Convention

```
Cycle_{N}_{charge|discharge}_Crate_{rate}_tempK_{temp}_batteryID_{id}.png
```

Example: `Cycle_1_charge_Crate_0.5_tempK_298.15_batteryID_CS2_3.png`

### Compliance

All parsers (CS2, CX2, PL, Stanford, Oxford, INR) follow this standard.