import json
from itertools import accumulate
import numpy as np
import matplotlib.pyplot as plt
import os

# ------------------ Paths ------------------
BASELINE_JSON = "baseline_metrics_20250822_162100.json"
FEATURE_JSON  = "feature_init_metrics_20250822_164926.json"

# Smoothing window (MUST be odd). Set to 1 to disable smoothing.
SMOOTH_WIN = 5

# ------------------ Helpers ------------------
def load_history(path):
    with open(path, "r") as f:
        data = json.load(f)
    h = data["training_history"]
    epoch_time = np.asarray(h["epoch_time"], dtype=float)
    cum_t = np.asarray(list(accumulate(epoch_time)), dtype=float)
    train_loss = np.asarray(h["train_loss"], dtype=float)
    val_rec = np.asarray(h["val_recall"], dtype=float)
    return cum_t, train_loss, val_rec, data

def centered_moving_average(t, v, win):
    """Centered moving average; returns (t_smooth, v_smooth)."""
    if win <= 1 or win % 2 == 0 or v.size < win:
        return t, v
    k = win // 2
    v_s = np.convolve(v, np.ones(win)/win, mode="valid")
    t_s = np.array([t[i-k:i+k+1].mean() for i in range(k, len(t)-k)])
    return t_s, v_s

def build_step_series(cum_t, values):
    """
    Build stepwise series (hold previous value until epoch end).
    Returns arrays t, v appropriate for step plotting (where='post').
    """
    t = [0.0]
    v = [values[0]]
    for ct, val in zip(cum_t, values):
        t.append(ct); v.append(v[-1])   # horizontal segment
        t.append(ct); v.append(float(val))  # vertical jump
    return np.array(t), np.array(v)

def truncate_to_time(t, v, t_max):
    """
    Truncate step series (t, v) at t_max; extend horizontally to t_max if needed.
    Assumes t is non-decreasing.
    """
    mask = t <= t_max
    t_out = t[mask].tolist()
    v_out = v[mask].tolist()
    if len(t_out) == 0:
        # start at 0 with first value
        t_out = [0.0]
        v_out = [v[0]]
    if t_out[-1] < t_max:
        t_out.append(t_max)
        v_out.append(v_out[-1])
    return np.array(t_out), np.array(v_out)

# ------------------ Load both runs ------------------
base_t, base_loss, base_val, _ = load_history(BASELINE_JSON)
feat_t, feat_loss, feat_val, feat_raw = load_history(FEATURE_JSON)

# ------------------ Find the cutoff time ------------------
# Use Feature-Aware Initialization's BEST validation recall
best_idx = int(np.argmax(feat_val))
T_CUTOFF = float(feat_t[best_idx])

print(f"Feature-Aware Init best val_recall at epoch {best_idx+1} (0-based {best_idx}) "
      f"occurs at cumulative time T = {T_CUTOFF:.3f} s, value = {feat_val[best_idx]:.6f}")

# ------------------ Build step series and truncate ------------------
# Feature run
feat_t_loss_step, feat_loss_step = build_step_series(feat_t, feat_loss)
feat_t_val_step,  feat_val_step  = build_step_series(feat_t, feat_val)
feat_t_loss_tr, feat_loss_tr = truncate_to_time(feat_t_loss_step, feat_loss_step, T_CUTOFF)
feat_t_val_tr,  feat_val_tr  = truncate_to_time(feat_t_val_step,  feat_val_step,  T_CUTOFF)

# Baseline run
base_t_loss_step, base_loss_step = build_step_series(base_t, base_loss)
base_t_val_step,  base_val_step  = build_step_series(base_t, base_val)
base_t_loss_tr, base_loss_tr = truncate_to_time(base_t_loss_step, base_loss_step, T_CUTOFF)
base_t_val_tr,  base_val_tr  = truncate_to_time(base_t_val_step,  base_val_step,  T_CUTOFF)

# ------------------ Optional smoothing ------------------
feat_t_loss_sm, feat_loss_sm = centered_moving_average(feat_t_loss_tr, feat_loss_tr, SMOOTH_WIN)
feat_t_val_sm,  feat_val_sm  = centered_moving_average(feat_t_val_tr,  feat_val_tr,  SMOOTH_WIN)

base_t_loss_sm, base_loss_sm = centered_moving_average(base_t_loss_tr, base_loss_tr, SMOOTH_WIN)
base_t_val_sm,  base_val_sm  = centered_moving_average(base_t_val_tr,  base_val_tr,  SMOOTH_WIN)

# ------------------ Plot 1: Combined (loss + val recall) ------------------
plt.figure(figsize=(10, 6))
ax1 = plt.gca()
ax2 = ax1.twinx()

# Four distinct colors; solid = Feature-Aware Init, dashed = Baseline
# Loss (left y)
ax1.plot(feat_t_loss_sm, feat_loss_sm,  label="Feature-Aware Init — Train Loss", linewidth=2, color="tab:blue")
ax1.plot(base_t_loss_sm, base_loss_sm,  label="Baseline — Train Loss",         linewidth=2, linestyle="--", color="tab:orange")
# Val recall (right y)
ax2.plot(feat_t_val_sm,  feat_val_sm,   label="Feature-Aware Init — Val Recall", linewidth=2, color="tab:green")
ax2.plot(base_t_val_sm,  base_val_sm,   label="Baseline — Val Recall",           linewidth=2, linestyle="--", color="tab:red")

ax1.axvline(T_CUTOFF, linestyle=":", linewidth=1)

ax1.set_xlabel("Cumulative training time (seconds)")
ax1.set_ylabel("Training loss")
ax2.set_ylabel("Validation recall")
plt.title("Baseline vs Feature-Aware Initialization — Loss & Val Recall vs Time\nTruncated at Feature-Aware Init best val_recall time")

# merged legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("combined_loss_and_val_vs_time.png", dpi=150)

# ------------------ Plot 2: Validation recall vs time ------------------
plt.figure(figsize=(10, 6))
plt.plot(feat_t_val_sm, feat_val_sm,  linewidth=2, label="Feature-Aware Init — Val Recall", color="tab:green")
plt.plot(base_t_val_sm, base_val_sm,  linewidth=2, linestyle="--", label="Baseline — Val Recall", color="tab:red")
plt.axvline(T_CUTOFF, linestyle=":", linewidth=1)
plt.xlabel("Cumulative training time (seconds)")
plt.ylabel("Validation recall")
plt.title("Validation Recall vs Time (Truncated at Feature-Aware Init best val_recall time)")
plt.grid(True, alpha=0.3)
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("val_recall_vs_time.png", dpi=150)

# ------------------ Plot 3: Training loss vs time ------------------
plt.figure(figsize=(10, 6))
plt.plot(feat_t_loss_sm, feat_loss_sm, linewidth=2, label="Feature-Aware Init — Train Loss", color="tab:blue")
plt.plot(base_t_loss_sm, base_loss_sm, linewidth=2, linestyle="--", label="Baseline — Train Loss", color="tab:orange")
plt.axvline(T_CUTOFF, linestyle=":", linewidth=1)
plt.xlabel("Cumulative training time (seconds)")
plt.ylabel("Training loss")
plt.title("Training Loss vs Time (Truncated at Feature-Aware Init best val_recall time)")
plt.grid(True, alpha=0.3)
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("train_loss_vs_time.png", dpi=150)

print("Saved figures:")
for f in ["combined_loss_and_val_vs_time.png", "val_recall_vs_time.png", "train_loss_vs_time.png"]:
    print("  ", os.path.abspath(f))

