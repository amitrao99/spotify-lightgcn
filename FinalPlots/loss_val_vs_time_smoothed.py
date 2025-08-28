import json
import os
from itertools import accumulate
import numpy as np
import matplotlib.pyplot as plt

# ---------- Paths (adjust if needed) ----------
OPT_PATH = "perf_optimized_metrics_20250822_161614.json"
BASE_PATH = "baseline_metrics_20250822_162100.json"

# Smoothing window (must be odd). Set to 1 to disable smoothing.
SMOOTH_WIN = 5  # e.g., 5-point centered moving average

# ---------- Helpers ----------
def load_history(path):
    with open(path, "r") as f:
        d = json.load(f)
    h = d["training_history"]
    # Use epoch-end values vs cumulative time at epoch end
    epoch_time = np.asarray(h["epoch_time"], dtype=float)
    cum_t = np.asarray(list(accumulate(epoch_time)), dtype=float)
    train_loss = np.asarray(h["train_loss"], dtype=float)
    val_rec = np.asarray(h["val_recall"], dtype=float)  # using recall as requested earlier
    return cum_t, train_loss, val_rec

def truncate_to_time(cum_t, values, t_max):
    """Truncate (cum_t, values) at t_max; append a final point at t_max (held last value)."""
    mask = cum_t <= t_max
    t = cum_t[mask]
    v = values[mask]
    if t.size == 0:
        # If nothing fits, start from t=0 with first value
        t = np.array([0.0])
        v = np.array([values[0]])
    if t[-1] < t_max:
        t = np.append(t, t_max)
        v = np.append(v, v[-1])
    return t, v

def centered_moving_average(t, v, win):
    """Centered moving average with odd window; keeps alignment by trimming ends."""
    if win <= 1 or win % 2 == 0 or v.size < win:
        return t, v  # no smoothing or not enough points
    k = win // 2
    # For time, average window times too (keeps x monotone and aligned to center)
    v_smooth = np.convolve(v, np.ones(win)/win, mode="valid")
    # Rolling mean of time using the same window
    # (simple average of window's times places the smoothed point at the center)
    t_smooth = np.array([t[i-k:i+k+1].mean() for i in range(k, len(t)-k)])
    return t_smooth, v_smooth

# ---------- Load data ----------
opt_t, opt_loss, opt_val = load_history(OPT_PATH)
base_t, base_loss, base_val = load_history(BASE_PATH)

# Optimized 100-epoch time budget
T_BUDGET = float(opt_t[-1])

# Truncate all series at the same budget
opt_tL, opt_loss_t = truncate_to_time(opt_t,   opt_loss, T_BUDGET)
opt_tV, opt_val_t  = truncate_to_time(opt_t,   opt_val,  T_BUDGET)
bas_tL, bas_loss_t = truncate_to_time(base_t,  base_loss, T_BUDGET)
bas_tV, bas_val_t  = truncate_to_time(base_t,  base_val,  T_BUDGET)

# ---------- Smooth (centered MA) ----------
opt_tL_s, opt_loss_s = centered_moving_average(opt_tL, opt_loss_t, SMOOTH_WIN)
opt_tV_s, opt_val_s  = centered_moving_average(opt_tV, opt_val_t,  SMOOTH_WIN)
bas_tL_s, bas_loss_s = centered_moving_average(bas_tL, bas_loss_t, SMOOTH_WIN)
bas_tV_s, bas_val_s  = centered_moving_average(bas_tV, bas_val_t,  SMOOTH_WIN)

# ---------- Plot ----------
plt.figure(figsize=(10, 6))

# Four distinct colors: blue/orange/green/red
# Solid = optimized, dashed = baseline
plt.plot(opt_tL_s, opt_loss_s,  linewidth=2, label="Optimized — Train Loss", color="tab:blue")
plt.plot(bas_tL_s, bas_loss_s,  linewidth=2, linestyle="--", label="Baseline — Train Loss", color="tab:orange")
plt.plot(opt_tV_s, opt_val_s,   linewidth=2, label="Optimized — Val Recall", color="tab:green")
plt.plot(bas_tV_s, bas_val_s,   linewidth=2, linestyle="--", label="Baseline — Val Recall", color="tab:red")

# Budget cutoff indicator
plt.axvline(T_BUDGET, linestyle=":", linewidth=1)

plt.xlabel("Cumulative training time (seconds)")
plt.ylabel("Value")
plt.title("Training Loss & Validation Recall vs Time (Smoothed)\nTruncated at Optimized 100-Epoch Total Time")
plt.grid(True, alpha=0.3)
plt.legend(loc="best")
plt.tight_layout()

out_path = "loss_val_vs_time_smoothed.png"
plt.savefig(out_path, dpi=150)
print(f"Saved: {os.path.abspath(out_path)}")

# Optional: also show a quick numeric snapshot at the budget time
def snapshot_at_budget(t, v, t_star):
    # value at/just before t_star
    idx = np.searchsorted(t, t_star, side="right") - 1
    idx = max(0, min(idx, len(v)-1))
    return v[idx]

print("Snapshot @ budget:")
print(f"  Optimized  Train Loss: {snapshot_at_budget(opt_tL, opt_loss_t, T_BUDGET):.4f}")
print(f"  Baseline   Train Loss: {snapshot_at_budget(bas_tL, bas_loss_t, T_BUDGET):.4f}")
print(f"  Optimized  Val Recall: {snapshot_at_budget(opt_tV, opt_val_t, T_BUDGET):.4f}")
print(f"  Baseline   Val Recall: {snapshot_at_budget(bas_tV, bas_val_t, T_BUDGET):.4f}")

