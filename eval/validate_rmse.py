"""
EC7207 HPC Project - Group 12
validate_rmse.py  –  Verify correctness of all parallel implementations
                     by comparing their output flags against the serial
                     ground truth using RMSE.

Expected RMSE = 0.0 for deterministic implementations.
Your proposal target: RMSE < 0.01%

Run from project root:
    python eval/validate_rmse.py
"""

import numpy as np
import os
import struct

DATA_DIR = "./data"

# ── Load flags from binary file ────────────────────────────────────────────────
def load_flags(path):
    if not os.path.exists(path):
        return None
    flags = np.fromfile(path, dtype=np.int32)
    return flags

# ── Load dataset to get ground truth labels ────────────────────────────────────
def load_labels(bin_path):
    """Read y labels from a dataset binary produced by prepare_data.py."""
    with open(bin_path, "rb") as f:
        n_rows, n_feat = struct.unpack("ii", f.read(8))
        f.seek(n_rows * n_feat * 4, 1)   # skip X
        y = np.frombuffer(f.read(n_rows * 4), dtype=np.int32)
    return y

# ── RMSE ───────────────────────────────────────────────────────────────────────
def rmse(pred, truth):
    pred_f  = pred.astype(np.float64)
    truth_f = truth.astype(np.float64)
    return float(np.sqrt(np.mean((pred_f - truth_f) ** 2)))

# ── Confusion matrix ───────────────────────────────────────────────────────────
def confusion(pred, truth):
    tp = int(np.sum((pred == 1) & (truth == 1)))
    tn = int(np.sum((pred == 0) & (truth == 0)))
    fp = int(np.sum((pred == 1) & (truth == 0)))
    fn = int(np.sum((pred == 0) & (truth == 1)))
    return tp, tn, fp, fn

def metrics(pred, truth):
    tp, tn, fp, fn = confusion(pred, truth)
    n = len(pred)
    acc  = (tp + tn) / n
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return acc, prec, rec, f1

# ── Load reference data ────────────────────────────────────────────────────────
print("=" * 70)
print("EC7207 HPC – RMSE Validation Report")
print("=" * 70)

serial_path = os.path.join(DATA_DIR, "serial_flags.bin")
if not os.path.exists(serial_path):
    print(f"\nERROR: Serial ground truth not found at {serial_path}")
    print("Run the serial baseline first: ./bin/serial ./data/train_data.bin")
    exit(1)

serial_flags = load_flags(serial_path)
n = len(serial_flags)
print(f"\nSerial ground truth: {n} flags loaded")
print(f"  Anomalies: {serial_flags.sum()} ({100.0*serial_flags.sum()/n:.2f}%)")

# Load true labels if dataset is available
train_path = os.path.join(DATA_DIR, "train_data.bin")
y_true = None
if os.path.exists(train_path):
    try:
        y_true = load_labels(train_path)
        print(f"  True labels: {y_true.sum()} attacks ({100.0*y_true.sum()/n:.2f}%)")
    except Exception as e:
        print(f"  Could not load true labels: {e}")

# ── Validate each implementation ───────────────────────────────────────────────
implementations = [
    ("OpenMP",          "omp_flags.bin"),
    ("pthreads",        "pthreads_flags.bin"),
    ("MPI",             "mpi_flags.bin"),
    ("CUDA",            "cuda_flags.bin"),
    ("Hybrid MPI+OMP",  "hybrid_flags.bin"),
]

print("\n" + "-" * 70)
print(f"{'Implementation':<20} {'RMSE vs Serial':<18} {'Match %':<12} {'Status'}")
print("-" * 70)

all_passed = True
for name, fname in implementations:
    path = os.path.join(DATA_DIR, fname)
    flags = load_flags(path)

    if flags is None:
        print(f"{name:<20} {'NOT FOUND':<18} {'—':<12} ⚠ SKIP")
        continue

    if len(flags) != n:
        print(f"{name:<20} {'SIZE MISMATCH':<18} {'—':<12} ✗ FAIL")
        all_passed = False
        continue

    r = rmse(flags, serial_flags)
    match_pct = 100.0 * np.mean(flags == serial_flags)
    passed = r < 1e-6

    status = "✓ PASS" if passed else "✗ MISMATCH"
    if not passed: all_passed = False

    print(f"{name:<20} {r:<18.8f} {match_pct:<12.4f} {status}")

    # Diff breakdown
    diff_idx = np.where(flags != serial_flags)[0]
    if len(diff_idx) > 0:
        print(f"  → {len(diff_idx)} differing rows (first 5: {diff_idx[:5].tolist()})")

print("-" * 70)
print(f"\nOverall: {'ALL PASS ✓' if all_passed else 'SOME FAILURES ✗'}")

# ── Validate against true labels (detection quality) ──────────────────────────
if y_true is not None:
    print("\n" + "=" * 70)
    print("Detection Quality vs True Labels")
    print("=" * 70)
    print(f"{'Implementation':<20} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8} {'RMSE'}")
    print("-" * 70)

    # Serial vs true
    if y_true is not None and len(serial_flags) == len(y_true):
        acc, prec, rec, f1 = metrics(serial_flags, y_true)
        r = rmse(serial_flags, y_true)
        print(f"{'Serial (baseline)':<20} {acc:.4f}  {prec:.4f}  {rec:.4f}  {f1:.4f}  {r:.6f}")

    for name, fname in implementations:
        path = os.path.join(DATA_DIR, fname)
        flags = load_flags(path)
        if flags is None or len(flags) != len(y_true):
            continue
        acc, prec, rec, f1 = metrics(flags, y_true)
        r = rmse(flags, y_true)
        target = "✓" if r < 0.0001 else "✗"
        print(f"{name:<20} {acc:.4f}  {prec:.4f}  {rec:.4f}  {f1:.4f}  {r:.6f} {target}")

    print("\nTarget RMSE < 0.0001 (0.01%) as per project proposal.")

print("\nDone.")
