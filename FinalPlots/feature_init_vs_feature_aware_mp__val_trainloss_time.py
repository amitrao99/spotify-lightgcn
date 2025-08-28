import json
from itertools import accumulate
import numpy as np
import matplotlib.pyplot as plt
import os

# ------------------ Paths ------------------
FEATURE_INIT_JSON = "feature_init_metrics_20250822_164926.json"
FEATURE_AWARE_MP_JSON = "feature_aware_metrics_20250822_165413.json"

# Smoothing window (must be odd). Set to 1 to disable smoothing.
SMOOTH_WIN = 5

# Optional: if you *insist* the second file's best accuracy is ~57%,
# set this to 0.57 to stop at the first time it reaches/exceeds this.
# Otherwise, leave as None to use its absolute best val_recall.
TARGET_VAL = None  # e.g., 0.57

# ------------------ Helpers ------------------
def load_history(path):
    with open(path, "r") as f:
        data = json.load(f)
    h = data["training_history"]
    et = np.asarray(h["epoch_time"], dtype=float)
    cum_t = np.asarray(list(accumulate(et)), dtype=float)
    train_loss = np.asarray(h["train_loss"], dtype=float)
    val_rec = np.asarray(h["val_recall"], dtype=float)
    return cum_t, train_loss, val_rec

def centered_moving_average(t, v, win):
    """Centered moving average; returns (t_smooth, v_smooth)."""
    if win <= 1 or win % 2 == 0 or v.size < win:
        return t, v
    k = win // 2
    v_s = np.convolve(v, np.ones(win)/win, mode="valid")
    t_s = np.array([t[i-k:i+k+1].mean() for i in range(k, len(t)-k)])
    return t_s, v_s

def build_step_series(cum_t, values):
    """Hold value until epoch end; step up at each cum_t."""
    t = [0.0]
    v = [values[0]]
    for ct, val in zip(cum_t, values):
        t.append(ct); v.append(v[-1])      # flat
        t.append(ct); v.append(float(val)) # jump
    return np.array(t), np.array(v)

def truncate_to_time(t, v, t_max):
    """Truncate step series (t,v) at t_max, extend flat if needed."""
    mask = t <= t_max
    t_out = t[mask].tolist()
    v_out = v[mask].tolist()
    if not t_out:
        t_out, v_out = [0.0], [v[0]]
    if t_out[-1] < t_max:
        t_out.append(t_max)
        v_out.append(v_out[-1])
    return np.array(t_out), np.array(v_out)

# ------------------ Load both runs ------------------
fi_t, fi_loss, fi_val = load_history(FEATURE_INIT_JSON)      # Feature-Aware Initialization
mp_t, mp_loss, mp_val = load_history(FEATURE_AWARE_MP_JSON)  # Feature-Aware Message Passing (the "second" run)

# ------------------ Find cutoff: when the SECOND run hits best accuracy ------------------
if TARGET_VAL is not None:
    # First time MP reaches or exceeds TARGET_VAL
    idxs = np.where(mp_val >= TARGET_VAL)[0]
    if len(idxs) == 0:
        best_idx = int(np.argmax(mp_val))  # fallback to its best
    else:
        best_idx = int(idxs[0])
else:
    best_idx = int(np.argmax(mp_val))

T_CUTOFF = float(mp_t[best_idx])
best_val_pct = float(mp_val[best_idx]) * 100.0
print(f"Feature-Aware MP best val_recall at epoch {best_idx+1} occurs at T = {T_CUTOFF:.3f}s, value ≈ {best_val_pct:.2f}%.")

# ------------------ Build step series & truncate at T_CUTOFF ------------------
# Feature-Aware Init
fi_tL_step, fi_loss_step = build_step_series(fi_t, fi_loss)
fi_tV_step,  fi_val_step  = build_step_series(fi_t, fi_val)
fi_tL_tr, fi_loss_tr = truncate_to_time(fi_tL_step, fi_loss_step, T_CUTOFF)
fi_tV_tr,  fi_val_tr  = truncate_to_time(fi_tV_step,  fi_val_step,  T_CUTOFF)

# Feature-Aware MP
mp_tL_step, mp_loss_step = build_step_series(mp_t, mp_loss)
mp_tV_step,  mp_val_step  = build_step_series(mp_t, mp_val)
mp_tL_tr, mp_loss_tr = truncate_to_time(mp_tL_step, mp_loss_step, T_CUTOFF)
mp_tV_tr,  mp_val_tr  = truncate_to_time(mp_tV_step,  mp_val_step,  T_CUTOFF)

# ------------------ Smooth (optional) ------------------
fi_tL_sm, fi_loss_sm = centered_moving_average(fi_tL_tr, fi_loss_tr, SMOOTH_WIN)
fi_tV_sm, fi_val_sm  = centered_moving_average(fi_tV_tr,  fi_val_tr,  SMOOTH_WIN)
mp_tL_sm, mp_loss_sm = centered_moving_average(mp_tL_tr, mp_loss_tr, SMOOTH_WIN)
mp_tV_sm, mp_val_sm  = centered_moving_average(mp_tV_tr,  mp_val_tr,  SMOOTH_WIN)

# ------------------ Plot 1: Combined (loss + val recall; 4 colors) ------------------
plt.figure(figsize=(10, 6))
ax1 = plt.gca()
ax2 = ax1.twinx()

# Colors: blue/green for Feature-Aware Init; orange/red for Feature-Aware MP
ax1.plot(fi_tL_sm, fi_loss_sm,  linewidth=2, label="Feature-Aware Init — Train Loss", color="tab:blue")
ax1.plot(mp_tL_sm, mp_loss_sm,  linewidth=2, linestyle="--", label="Feature-Aware MP — Train Loss", color="tab:orange")
ax2.plot(fi_tV_sm, fi_val_sm,   linewidth=2, label="Feature-Aware Init — Val Recall", color="tab:green")
ax2.plot(mp_tV_sm, mp_val_sm,   linewidth=2, linestyle="--", label="Feature-Aware MP — Val Recall", color="tab:red")

ax1.axvline(T_CUTOFF, linestyle=":", linewidth=1)

ax1.set_xlabel("Cumulative training time (seconds)")
ax1.set_ylabel("Training loss")
ax2.set_ylabel("Validation recall")
plt.title("Feature-Aware Init vs Feature-Aware MP — Loss & Val Recall vs Time\nTruncated at MP best val_recall time")

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("combined_fi_vs_mp_loss_and_val_vs_time.png", dpi=150)

# ------------------ Plot 2: Validation recall vs time ------------------
plt.figure(figsize=(10, 6))
plt.plot(fi_tV_sm, fi_val_sm, linewidth=2, label="Feature-Aware Init — Val Recall", color="tab:green")
plt.plot(mp_tV_sm, mp_val_sm, linewidth=2, linestyle="--", label="Feature-Aware MP — Val Recall", color="tab:red")
plt.axvline(T_CUTOFF, linestyle=":", linewidth=1)
plt.xlabel("Cumulative training time (seconds)")
plt.ylabel("Validation recall")
plt.title("Validation Recall vs Time (Truncated at MP best val_recall time)")
plt.grid(True, alpha=0.3)
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("val_recall_fi_vs_mp_vs_time.png", dpi=150)

# ------------------ Plot 3: Training loss vs time ------------------
plt.figure(figsize=(10, 6))
plt.plot(fi_tL_sm, fi_loss_sm, linewidth=2, label="Feature-Aware Init — Train Loss", color="tab:blue")
plt.plot(mp_tL_sm, mp_loss_sm, linewidth=2, linestyle="--", label="Feature-Aware MP — Train Loss", color="tab:orange")
plt.axvline(T_CUTOFF, linestyle=":", linewidth=1)
plt.xlabel("Cumulative training time (seconds)")
plt.ylabel("Training loss")
plt.title("Training Loss vs Time (Truncated at MP best val_recall time)")
plt.grid(True, alpha=0.3)
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("train_loss_fi_vs_mp_vs_time.png", dpi=150)

print("Saved figures:")
for f in [
    "combined_fi_vs_mp_loss_and_val_vs_time.png",
    "val_recall_fi_vs_mp_vs_time.png",
    "train_loss_fi_vs_mp_vs_time.png",
]:
    print("  ", os.path.abspath(f))

