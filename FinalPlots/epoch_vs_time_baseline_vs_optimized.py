import json
import os
from itertools import accumulate
import numpy as np
import matplotlib.pyplot as plt

# --- Paths (update if needed) ---
OPT_PATH = "perf_optimized_metrics_20250822_161614.json"
BASE_PATH = "baseline_metrics_20250822_162100.json"

# --- Load JSONs ---
def load_epoch_times(path):
    with open(path, "r") as f:
        data = json.load(f)
    epoch_times = data["training_history"]["epoch_time"]
    epochs = data["training_history"]["epoch"]
    # Some logs index epochs starting at 0; we want 1..N for display
    epoch_numbers = [e + 1 for e in epochs]
    return np.array(epoch_numbers, dtype=float), np.array(epoch_times, dtype=float), data

opt_epochs, opt_epoch_times, opt_raw = load_epoch_times(OPT_PATH)
base_epochs, base_epoch_times, base_raw = load_epoch_times(BASE_PATH)

# --- Cumulative time (seconds) ---
opt_cum_time = np.array(list(accumulate(opt_epoch_times)))
base_cum_time = np.array(list(accumulate(base_epoch_times)))

# Total time budget = optimized 100-epoch time
T_budget = float(opt_cum_time[-1])

# --- Find where baseline hits the optimized time budget ---
# If baseline never reaches (shouldn't happen here), clamp to last
hit_idx = np.searchsorted(base_cum_time, T_budget)

if hit_idx == 0:
    # Budget is within first epoch; interpolate within epoch 1
    prev_cum = 0.0
    this_epoch_time = base_epoch_times[0]
    frac_within = (T_budget - prev_cum) / this_epoch_time
    base_hit_epoch_fractional = 1.0 * frac_within  # between 0 and 1
elif hit_idx >= len(base_cum_time):
    # Budget exceeds all baseline epochs; we just end at last epoch
    base_hit_epoch_fractional = float(base_epochs[-1])
else:
    # Interpolate between epoch hit_idx and hit_idx+1 in 1-based epoch space
    prev_cum = base_cum_time[hit_idx - 1]
    this_epoch_time = base_epoch_times[hit_idx]
    frac_within = (T_budget - prev_cum) / this_epoch_time
    # epoch numbers are 1-based
    base_hit_epoch_fractional = (hit_idx) + frac_within  # since hit_idx is 0-based index

# Integer epoch at/just before budget
base_hit_epoch_integer = int(np.searchsorted(base_cum_time, T_budget, side="right"))

# --- Build "epoch reached vs time" curves ---
# For each run, we want a staircase mapping: at time t, what's the highest epoch fully completed?

def build_staircase(cum_times, epoch_numbers):
    """
    Returns time_points, epoch_points suitable for step plotting.
    Staircase steps up after each cumulative time point.
    """
    # Start at time 0 with epoch 0
    time_points = [0.0]
    epoch_points = [0.0]
    for t, e in zip(cum_times, epoch_numbers):
        # stay at previous epoch until time t
        time_points.append(t)
        epoch_points.append(epoch_points[-1])
        # then step up to epoch e at time t (vertical step)
        time_points.append(t)
        epoch_points.append(float(e))
    return np.array(time_points), np.array(epoch_points)

opt_t, opt_e = build_staircase(opt_cum_time, opt_epochs)

# For baseline, we truncate at T_budget and include the fractional epoch point
# Build staircase up to the last fully completed epoch <= budget
base_t, base_e = build_staircase(base_cum_time, base_epochs)

# Truncate baseline staircase at T_budget
def truncate_staircase(t, e, t_max):
    mask = t <= t_max
    t_trunc = t[mask].tolist()
    e_trunc = e[mask].tolist()
    # Ensure we end exactly at t_max with interpolated epoch value
    # Interpolate on the vertical/horizontal segments:
    # Find the segment baseline is on at t_max: epoch increases at exact cum_times.
    # Between cum_times, e is flat; so e_at_tmax = number of fully completed epochs.
    # But for the final display, we can show a marker at fractional epoch.
    if t_trunc[-1] < t_max:
        # We are in a flat segment; append (t_max, same epoch)
        t_trunc.append(t_max)
        e_trunc.append(e_trunc[-1])
    return np.array(t_trunc), np.array(e_trunc)

base_t_trunc, base_e_trunc = truncate_staircase(base_t, base_e, T_budget)

# --- Plot ---
plt.figure(figsize=(9, 6))
plt.step(opt_t, opt_e, where="post", linewidth=2, label="Optimized (epochs completed)")
plt.step(base_t_trunc, base_e_trunc, where="post", linewidth=2, linestyle="--", label="Baseline (epochs completed, truncated)")

# Mark the exact fractional epoch point for baseline at the budget time
plt.plot([T_budget], [base_hit_epoch_fractional], marker="o")
plt.annotate(
    f"Baseline @ budget ≈ {base_hit_epoch_fractional:.2f} epochs",
    xy=(T_budget, base_hit_epoch_fractional),
    xytext=(10, 10),
    textcoords="offset points",
    fontsize=9
)

# Cosmetic axes/labels
plt.xlabel("Cumulative training time (seconds)")
plt.ylabel("Epochs completed")
plt.title("Epoch progress vs cumulative time\nBaseline vs Optimized (baseline truncated at optimized total time)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# --- Print summary numbers ---
speedup = (base_cum_time[-1] / T_budget) if T_budget > 0 else float("nan")
print(f"Optimized total time for 100 epochs: {T_budget:.3f} s")
print(f"Baseline epochs completed within that time: {base_hit_epoch_integer} (exact ~{base_hit_epoch_fractional:.2f})")
print(f"Baseline total time for 100 epochs: {base_cum_time[-1]:.3f} s")
print(f"Approx speedup (baseline time / optimized time): {speedup:.2f}×")

# --- Save figure ---
out_path = "epoch_vs_time_baseline_vs_optimized.png"
plt.savefig(out_path, dpi=150)
print(f"Saved plot to: {os.path.abspath(out_path)}")

