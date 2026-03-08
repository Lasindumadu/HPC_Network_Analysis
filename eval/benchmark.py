"""
EC7207 HPC Project - Group 12
benchmark.py  –  Automates all runs, collects timings, computes speedup,
                 and generates publication-quality plots.

Run from the project root after building all targets:
    cd EC7207_HPC_G12
    python eval/benchmark.py
"""

import subprocess
import re
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime

# ── Configuration ──────────────────────────────────────────────────────────────
BIN_DIR    = "./bin"
DATA_TRAIN = "./data/train_data.bin"
DATA_TEST  = "./data/test_data.bin"
OUT_DIR    = "./results"
os.makedirs(OUT_DIR, exist_ok=True)

THREAD_COUNTS = [1, 2, 4, 8]
PROC_COUNTS   = [1, 2, 4, 8]
N_REPEATS     = 3          # repeat each run and take minimum (more stable)

# ── Helper: run a shell command and extract time in seconds ───────────────────
def run_timed(cmd, label=""):
    """Returns the minimum time (seconds) across N_REPEATS runs."""
    times = []
    for rep in range(N_REPEATS):
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=300
            )
            output = result.stdout + result.stderr
            # Try to match "time: X.XXXX s" or "time=X.XXXX s"
            match = re.search(r'[Tt]ime\s*[=:]\s*([\d.]+)\s*s', output)
            if match:
                t = float(match.group(1))
                times.append(t)
                print(f"    [{label} rep {rep+1}] {t:.4f}s")
            else:
                print(f"    [{label}] Could not parse time from output:")
                print(f"    {output[:200]}")
        except subprocess.TimeoutExpired:
            print(f"    [{label}] TIMEOUT after 300s")
        except Exception as e:
            print(f"    [{label}] ERROR: {e}")

    if not times:
        return None
    return min(times)   # Use minimum to reduce noise

# ── Run all benchmarks ─────────────────────────────────────────────────────────
results = {}

print("=" * 60)
print("EC7207 HPC Benchmark Suite")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

# Serial baseline
print("\n[1/6] Serial baseline...")
t = run_timed(f"{BIN_DIR}/serial {DATA_TRAIN}", "serial")
results["serial"] = t
print(f"  => Serial time: {t:.4f}s" if t else "  => FAILED")

if not results["serial"]:
    print("Serial baseline failed — cannot compute speedup. Exiting.")
    exit(1)

T_serial = results["serial"]

# OpenMP scaling
print("\n[2/6] OpenMP scaling...")
results["omp"] = {}
for nt in THREAD_COUNTS:
    t = run_timed(f"{BIN_DIR}/omp {DATA_TRAIN} {nt}", f"omp-{nt}T")
    results["omp"][nt] = t
    if t:
        print(f"  => {nt} threads: {t:.4f}s  speedup={T_serial/t:.2f}x")

# pthreads scaling
print("\n[3/6] POSIX Threads scaling...")
results["pthreads"] = {}
for nt in THREAD_COUNTS:
    t = run_timed(f"{BIN_DIR}/pthreads {DATA_TRAIN} {nt}", f"pthreads-{nt}T")
    results["pthreads"][nt] = t
    if t:
        print(f"  => {nt} threads: {t:.4f}s  speedup={T_serial/t:.2f}x")

# MPI scaling
print("\n[4/6] MPI scaling...")
results["mpi"] = {}
for np_ in PROC_COUNTS:
    t = run_timed(f"mpirun -np {np_} {BIN_DIR}/mpi {DATA_TRAIN}", f"mpi-{np_}P")
    results["mpi"][np_] = t
    if t:
        print(f"  => {np_} ranks: {t:.4f}s  speedup={T_serial/t:.2f}x")

# CUDA
print("\n[5/6] CUDA GPU...")
t = run_timed(f"{BIN_DIR}/cuda {DATA_TRAIN}", "cuda")
results["cuda"] = t
if t:
    print(f"  => CUDA: {t:.4f}s  speedup={T_serial/t:.2f}x")

# Hybrid MPI+OpenMP
print("\n[6/6] Hybrid MPI+OpenMP...")
results["hybrid"] = {}
configs = [(1, 4), (2, 4), (4, 4), (2, 8)]   # (ranks, threads)
for np_, nt in configs:
    key = f"{np_}rx{nt}t"
    cmd = f"OMP_NUM_THREADS={nt} mpirun -np {np_} {BIN_DIR}/hybrid {DATA_TRAIN}"
    t = run_timed(cmd, f"hybrid-{key}")
    results["hybrid"][key] = {"time": t, "ranks": np_, "threads": nt, "total": np_*nt}
    if t:
        print(f"  => {np_} ranks x {nt} threads ({np_*nt} total): "
              f"{t:.4f}s  speedup={T_serial/t:.2f}x")

# ── Save raw results ───────────────────────────────────────────────────────────
results_path = os.path.join(OUT_DIR, "benchmark_results.json")
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nRaw results saved to {results_path}")

# ── Plotting ───────────────────────────────────────────────────────────────────
print("\nGenerating plots...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("EC7207 HPC – Network Traffic Anomaly Detection Speedup",
             fontsize=14, fontweight="bold")

# ── Plot 1: Shared-memory speedup (OpenMP vs pthreads) ──────────────────────
ax = axes[0]
x = THREAD_COUNTS

omp_speedup = [T_serial / results["omp"][t] for t in x
               if results["omp"].get(t)]
pt_speedup  = [T_serial / results["pthreads"][t] for t in x
               if results["pthreads"].get(t)]

ax.plot(x[:len(omp_speedup)], omp_speedup, "b-o", linewidth=2,
        markersize=8, label="OpenMP", zorder=3)
ax.plot(x[:len(pt_speedup)],  pt_speedup,  "r-s", linewidth=2,
        markersize=8, label="pthreads", zorder=3)
ax.plot(x, x, "k--", linewidth=1, alpha=0.5, label="Ideal linear")

ax.set_xlabel("Number of Threads", fontsize=11)
ax.set_ylabel("Speedup vs Serial", fontsize=11)
ax.set_title("Shared-Memory Parallelism", fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xticks(x)

# ── Plot 2: Distributed (MPI) speedup ───────────────────────────────────────
ax = axes[1]
x2 = PROC_COUNTS

mpi_speedup = [T_serial / results["mpi"][p] for p in x2
               if results["mpi"].get(p)]

ax.plot(x2[:len(mpi_speedup)], mpi_speedup, "g-^", linewidth=2,
        markersize=8, label="MPI", zorder=3)
ax.plot(x2, x2, "k--", linewidth=1, alpha=0.5, label="Ideal linear")

# CUDA as horizontal line
if results.get("cuda"):
    cuda_su = T_serial / results["cuda"]
    ax.axhline(y=cuda_su, color="purple", linestyle="-.", linewidth=2,
               label=f"CUDA ({cuda_su:.1f}x)")

ax.set_xlabel("Number of MPI Processes", fontsize=11)
ax.set_ylabel("Speedup vs Serial", fontsize=11)
ax.set_title("Distributed & GPU Parallelism", fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xticks(x2)

# ── Plot 3: Execution time comparison (bar chart) ───────────────────────────
ax = axes[2]

labels, times, colors = [], [], []

labels.append("Serial");         times.append(T_serial);           colors.append("#e74c3c")
if results["omp"].get(4):
    labels.append("OMP 4T");     times.append(results["omp"][4]);  colors.append("#3498db")
if results["omp"].get(8):
    labels.append("OMP 8T");     times.append(results["omp"][8]);  colors.append("#2980b9")
if results["pthreads"].get(4):
    labels.append("pthreads 4T"); times.append(results["pthreads"][4]); colors.append("#e67e22")
if results["mpi"].get(4):
    labels.append("MPI 4P");     times.append(results["mpi"][4]);  colors.append("#27ae60")
if results["cuda"]:
    labels.append("CUDA");       times.append(results["cuda"]);    colors.append("#8e44ad")
h_vals = results.get("hybrid", {})
for k, v in h_vals.items():
    if v["time"]:
        labels.append(f"Hybrid\n{v['ranks']}Rx{v['threads']}T")
        times.append(v["time"])
        colors.append("#1abc9c")
        break   # just show one hybrid config

bars = ax.bar(labels, times, color=colors, edgecolor="white", linewidth=0.8)
ax.set_ylabel("Execution Time (seconds)", fontsize=11)
ax.set_title("Execution Time Comparison", fontsize=12)
ax.tick_params(axis="x", labelsize=8.5)
ax.grid(True, axis="y", alpha=0.3)

# Add speedup labels on bars
for bar, t in zip(bars[1:], times[1:]):
    su = T_serial / t
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{su:.1f}x", ha="center", va="bottom", fontsize=9, fontweight="bold")

plt.tight_layout()
plot_path = os.path.join(OUT_DIR, "speedup_curves.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"Plot saved to {plot_path}")
plt.show()

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY TABLE")
print("=" * 60)
print(f"{'Implementation':<25} {'Time (s)':<12} {'Speedup':<10} {'Efficiency'}")
print("-" * 60)

def row(name, t, base, n_units=1):
    if t is None:
        print(f"{name:<25} {'N/A':<12} {'N/A':<10} N/A")
        return
    su = base / t
    eff = su / n_units
    print(f"{name:<25} {t:<12.4f} {su:<10.2f} {eff:.2f}")

row("Serial",          T_serial,                   T_serial, 1)
for nt in THREAD_COUNTS:
    row(f"OpenMP {nt}T",    results["omp"].get(nt),     T_serial, nt)
for nt in THREAD_COUNTS:
    row(f"pthreads {nt}T",  results["pthreads"].get(nt),T_serial, nt)
for np_ in PROC_COUNTS:
    row(f"MPI {np_}P",       results["mpi"].get(np_),    T_serial, np_)
row("CUDA",            results["cuda"],             T_serial, 1)
for k, v in results.get("hybrid", {}).items():
    row(f"Hybrid {k}",   v["time"],                  T_serial, v["total"])

print("=" * 60)
print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
