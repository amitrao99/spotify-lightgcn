# Validation recall vs time (smoothed) for 4 models, truncated at the best model's best val recall time.
# Also plot test recalls as single markers.


import json
from itertools import accumulate
import numpy as np
import matplotlib.pyplot as plt
import os

# ------------------ Config ------------------
FILES = {
    "Baseline": "baseline_metrics_20250822_162100.json",
    "Optimized": "perf_optimized_metrics_20250822_223528.json",
    "Feat-Aware Init": "feature_init_metrics_20250822_164926.json",
    "Feat-Aware MP": "feature_aware_metrics_20250822_165413.json",
}

SMOOTH_WIN = 7  # odd window for centered moving average; set 1 to disable

# ------------------ Helpers ------------------
def load_history(path):
    with open(path, "r") as f:
        d = json.load(f)
    h = d["training_history"]
    epoch_time = np.asarray(h["epoch_time"], dtype=float)
    cum_t = np.asarray(list(accumulate(epoch_time)), dtype=float)
    val_rec = np.asarray(h["val_recall"], dtype=float)
    # test recall: primary location per user -> test_results.final_test_recall
    test = None
    if "test_results" in d and isinstance(d["test_results"], dict):
        test = d["test_results"].get("final_test_recall", None)
    # fallback
    if test is None and "final_results" in d and isinstance(d["final_results"], dict):
        test = d["final_results"].get("test_recall", None)
    if test is None:
        test = d.get("test_recall", None)
    if test is None:
        test_hist = d.get("training_history", {}).get("test_recall", [])
        test = test_hist[-1] if test_hist else None
    return cum_t, val_rec, test

def build_step_series(cum_t, values):
    """Step series that holds previous value until epoch end."""
    t = [0.0]
    v = [values[0]]
    for ct, val in zip(cum_t, values):
        t.append(ct); v.append(v[-1])
        t.append(ct); v.append(float(val))
    return np.array(t), np.array(v)

def truncate_series(t, v, t_max):
    mask = t <= t_max
    t_out = t[mask].tolist()
    v_out = v[mask].tolist()
    if not t_out:
        t_out, v_out = [0.0], [v[0]]
    if t_out[-1] < t_max:
        t_out.append(t_max)
        v_out.append(v_out[-1])
    return np.array(t_out), np.array(v_out)

def centered_moving_average(t, v, win):
    if win <= 1 or win % 2 == 0 or len(v) < win:
        return t, v
    k = win // 2
    v_s = np.convolve(v, np.ones(win)/win, mode="valid")
    t_s = np.array([t[i-k:i+k+1].mean() for i in range(k, len(t)-k)])
    return t_s, v_s

# ------------------ Load all ------------------
series = {}
best_model = None
best_val_value = -np.inf
best_val_time = None

for name, path in FILES.items():
    cum_t, val_rec, test_rec = load_history(path)
    # find this model's best val recall and its time
    idx_best = int(np.argmax(val_rec))
    t_best = float(cum_t[idx_best])
    v_best = float(val_rec[idx_best])
    if v_best > best_val_value:
        best_val_value = v_best
        best_val_time = t_best
        best_model = name
    series[name] = {
        "cum_t": cum_t,
        "val": val_rec,
        "test": test_rec,
        "t_best": t_best,
        "v_best": v_best,
        "t_total": float(cum_t[-1]),
    }

T_CUTOFF = best_val_time

# ------------------ Prepare plot data ------------------
colors = {
    "Baseline": "tab:blue",
    "Optimized": "tab:orange",
    "Feat-Aware Init": "tab:green",
    "Feat-Aware MP": "tab:red",
}
markers = {
    "Baseline": "o",
    "Optimized": "s",
    "Feat-Aware Init": "^",
    "Feat-Aware MP": "D",
}

plt.figure(figsize=(11, 6))

for name in FILES.keys():
    cum_t = series[name]["cum_t"]
    val = series[name]["val"]
    t_step, v_step = build_step_series(cum_t, val)
    t_tr, v_tr = truncate_series(t_step, v_step, T_CUTOFF)
    t_sm, v_sm = centered_moving_average(t_tr, v_tr, SMOOTH_WIN)
    plt.plot(t_sm, v_sm, linewidth=2, label=f"{name} — Val Recall", color=colors[name])

# Plot test recall markers
for name in FILES.keys():
    test = series[name]["test"]
    if test is None:
        continue
    t_mark = series[name]["t_total"]
    clipped = False
    if t_mark > T_CUTOFF:
        t_mark = T_CUTOFF
        clipped = True
    plt.plot([t_mark], [test], marker=markers[name], linestyle="None",
             markersize=8, color=colors[name], label=f"{name} — Test Recall" + (" (at end, clipped)" if clipped else ""))
    # annotate value
    plt.text(t_mark, test + 0.015, f"{test:.3f}", ha="center", va="bottom", fontsize=9, color=colors[name])

# Vertical line at cutoff with annotation
plt.axvline(T_CUTOFF, linestyle=":", linewidth=1)
plt.annotate(f"Cutoff @ best val (best: {best_model}, {best_val_value:.3f})",
             xy=(T_CUTOFF, 0.02), xytext=(10, 10), textcoords="offset points", fontsize=10)

plt.xlabel("Cumulative training time (seconds)")
plt.ylabel("Validation recall")
plt.title("Validation Recall vs Time (Smoothed), with Test Recall Markers\nTruncated at time of the best model's best validation recall")
plt.ylim(0, 1.0)
plt.grid(True, alpha=0.3)
plt.legend(ncol=2, fontsize=9)
plt.tight_layout()

out_path = "val_vs_time_with_test_markers.png"
plt.savefig(out_path, dpi=150)
out_path

