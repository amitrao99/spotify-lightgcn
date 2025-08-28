import json
import numpy as np
import matplotlib.pyplot as plt
import os

# ---------- Paths (adjust if needed) ----------
FILES = {
    "Baseline": "baseline_metrics_20250822_162100.json",
    "Optimized": "perf_optimized_metrics_20250822_161614.json",
    "Feat-Aware Init": "feature_init_metrics_20250822_164926.json",
    "Feat-Aware MP": "feature_aware_metrics_20250822_165413.json",
}

# If True, and a model is missing test recall, fall back to best validation recall
FALLBACK_TO_VAL = True

def safe_max(arr):
    return max(arr) if arr else None

def load_params_and_test(path):
    with open(path, "r") as f:
        d = json.load(f)

    # Parameter count
    params = None
    if "model_stats" in d:
        params = d["model_stats"].get("total_params") \
                 or d["model_stats"].get("total_trainable_params")
    if params is None:
        raise ValueError(f"Param count not found in {path}")

    # ---- Test recall (primary location you specified) ----
    test = None
    if "test_results" in d and isinstance(d["test_results"], dict):
        test = d["test_results"].get("final_test_recall", None)

    # ---- Fallbacks (just in case a file doesn't have test_results) ----
    if test is None and "final_results" in d and isinstance(d["final_results"], dict):
        test = d["final_results"].get("test_recall", None)
    if test is None:
        test = d.get("test_recall", None)
    if test is None:
        th = d.get("training_history", {})
        test_hist = th.get("test_recall", [])
        if test_hist:
            test = test_hist[-1]

    # Best validation recall (only used as optional fallback)
    best_val = None
    th = d.get("training_history", {})
    if "val_recall" in th:
        best_val = safe_max(th["val_recall"])

    return params, test, best_val

labels, params_list, test_vals, best_vals = [], [], [], []
for name, path in FILES.items():
    p, t, v = load_params_and_test(path)
    labels.append(name)
    params_list.append(p)
    test_vals.append(t)
    best_vals.append(v)

# Optional fallback to keep the line continuous
used_fallback = []
final_test_vals = []
for t, v in zip(test_vals, best_vals):
    if t is None and FALLBACK_TO_VAL and v is not None:
        final_test_vals.append(v)
        used_fallback.append(True)
    else:
        final_test_vals.append(t)
        used_fallback.append(False)

# ---------- Plot ----------
x = np.arange(len(labels))
params_arr = np.array(params_list, dtype=float)
test_arr = np.array([np.nan if tv is None else float(tv) for tv in final_test_vals], dtype=float)

fig, ax1 = plt.subplots(figsize=(11, 6))

# Bars: parameter counts
bars = ax1.bar(x, params_arr, width=0.6, color="lightsteelblue", alpha=0.9, label="Parameter Count")
ax1.set_ylabel("Parameter Count", color="navy", fontsize=12)
ax1.tick_params(axis="y", labelcolor="navy")

# Tighten y-limits and add headroom
ymin = np.nanmin(params_arr) * 0.985
ymax = np.nanmax(params_arr) * 1.015
ax1.set_ylim(ymin, ymax)

# Annotate param counts
for bar, val in zip(bars, params_arr):
    ax1.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + (ymax - ymin) * 0.004,
        f"{int(val):,}",
        ha="center", va="bottom", fontsize=9, color="navy"
    )

# Right axis: test recall line + dots
ax2 = ax1.twinx()
ax2.set_ylabel("Test Recall", fontsize=12)
ax2.set_ylim(0, 1.0)

# Use "o-" to force markers *and* a connecting line
ax2.plot(x, test_arr, "o-", color="seagreen", linewidth=2, markersize=8, label="Test Recall")

# Annotate test values; add '*' if it’s a fallback from val_recall
for xi, (val, fb) in enumerate(zip(test_arr, used_fallback)):
    if np.isfinite(val):
        lbl = f"{val:.3f}" + ("*" if fb else "")
        ax2.text(xi, val + 0.02, lbl, ha="center", va="bottom", fontsize=9, color="seagreen")

# X labels, title, legends
ax1.set_xticks(x)
ax1.set_xticklabels(labels, rotation=15)
plt.title("Model Comparison: Parameter Count vs Test Recall", fontsize=14, fontweight="bold")

ax1.legend(loc="upper left")
ax2.legend(loc="upper right")

plt.tight_layout()
out_path = "params_vs_test_recall_fixed.png"
plt.savefig(out_path, dpi=150)
print("Saved to:", out_path)

