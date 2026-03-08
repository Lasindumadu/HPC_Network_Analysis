"""
generate_charts.py
EC7207 — HPC Network Traffic Analysis
Reads saved output logs from the last execution and generates charts.

FIRST RUN  : runs all programs, saves output to results/logs/, generates charts
LATER RUNS : reads saved logs, generates charts instantly (no re-running)

Usage:
    python3 generate_charts.py          # use saved logs (or run if no logs)
    python3 generate_charts.py --rerun  # force re-run all programs

Requirements:
    pip install matplotlib numpy
"""

import os
import re
import sys
import subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ── Config ────────────────────────────────────────────────────
DATASET  = "data/UNSW-NB15_1.csv/UNSW-NB15_1_with_header.csv"
SERIAL   = "./results/serial"
OPENMP   = "./results/openmp"
MPI      = "./results/mpi"
WORKERS  = [1, 2, 4, 8]
LOG_DIR  = "results/logs"
os.makedirs("charts",  exist_ok=True)
os.makedirs(LOG_DIR,   exist_ok=True)

# Log file paths — one per run
LOG_SERIAL  = f"{LOG_DIR}/serial.log"
LOG_OMP     = [f"{LOG_DIR}/openmp_{t}t.log"  for t in WORKERS]
LOG_MPI     = [f"{LOG_DIR}/mpi_{p}p.log"     for p in WORKERS]

# ── Style ─────────────────────────────────────────────────────
BG       = "#0b0f14"
CARD     = "#111820"
BORDER   = "#1e2a38"
TEXT     = "#e0e6f0"
DIMTEXT  = "#5a6a7a"
SERIAL_C = "#8b949e"
OMP_C    = "#00d4ff"
MPI_C    = "#ff6b35"
IDEAL_C  = "#ffffff"
GRID_C   = "#1e2a38"
GREEN_C  = "#3fb950"

# ══════════════════════════════════════════════════════════════
# HELPERS — run & parse
# ══════════════════════════════════════════════════════════════

def run_and_save(cmd, log_path, env=None):
    """Run command, save stdout to log file, return stdout."""
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr.strip()}")
    with open(log_path, "w") as f:
        f.write(result.stdout)
    return result.stdout

def read_log(log_path):
    """Read saved log file and return contents."""
    with open(log_path, "r") as f:
        return f.read()

def logs_exist():
    """Check if all log files from a previous run are present."""
    all_logs = [LOG_SERIAL] + LOG_OMP + LOG_MPI
    return all(os.path.exists(p) for p in all_logs)

def parse_single_time(output):
    m = re.search(r"Single-pass time:\s+([\d.]+)s", output)
    return float(m.group(1)) if m else None

def parse_throughput(output):
    m = re.search(r"Throughput:\s+([\d]+)\s+rec/s", output)
    return int(m.group(1)) if m else None

def parse_speedup(output):
    m = re.search(r"Speedup:\s+([\d.]+)x", output)
    return float(m.group(1)) if m else None

def parse_efficiency(output):
    m = re.search(r"Efficiency:\s+([\d.]+)%", output)
    return float(m.group(1)) if m else None

def parse_accuracy_metrics(output):
    metrics = {}
    for key, pattern in [
        ("accuracy",  r"Accuracy:\s+([\d.]+)%"),
        ("precision", r"Precision:\s+([\d.]+)%"),
        ("recall",    r"Recall:\s+([\d.]+)%"),
        ("f1",        r"F1 Score:\s+([\d.]+)%"),
        ("rmse",      r"RMSE:\s+([\d.]+)\s"),
    ]:
        m = re.search(pattern, output)
        metrics[key] = float(m.group(1)) if m else None
    return metrics

# ══════════════════════════════════════════════════════════════
# STEP 1 — Load or run
# ══════════════════════════════════════════════════════════════

force_rerun = "--rerun" in sys.argv

if force_rerun or not logs_exist():
    if force_rerun:
        print("Force re-run requested.\n")
    else:
        print("No saved logs found — running all programs now.\n")

    base_env = os.environ.copy()

    print("[1/3] Running Serial...")
    serial_out = run_and_save([SERIAL, DATASET], LOG_SERIAL)

    print("\n[2/3] Running OpenMP (1, 2, 4, 8 threads)...")
    omp_outs = []
    for i, t in enumerate(WORKERS):
        env = base_env.copy()
        env["OMP_NUM_THREADS"] = str(t)
        omp_outs.append(run_and_save([OPENMP, DATASET], LOG_OMP[i], env=env))

    print("\n[3/3] Running MPI (1, 2, 4, 8 processes)...")
    mpi_outs = []
    for i, p in enumerate(WORKERS):
        mpi_outs.append(run_and_save(
            ["mpirun", "--allow-run-as-root", "--oversubscribe",
             "-np", str(p), MPI, DATASET],
            LOG_MPI[i]
        ))
    print(f"\nLogs saved to {LOG_DIR}/")

else:
    print(f"Reading saved logs from {LOG_DIR}/ (use --rerun to re-execute)\n")
    serial_out = read_log(LOG_SERIAL)
    omp_outs   = [read_log(p) for p in LOG_OMP]
    mpi_outs   = [read_log(p) for p in LOG_MPI]

# ══════════════════════════════════════════════════════════════
# STEP 2 — Parse all results
# ══════════════════════════════════════════════════════════════

serial_time    = parse_single_time(serial_out)
serial_thr     = parse_throughput(serial_out)
serial_metrics = parse_accuracy_metrics(serial_out)

omp_time       = [parse_single_time(o)  for o in omp_outs]
omp_speedup    = [parse_speedup(o)      for o in omp_outs]
omp_efficiency = [parse_efficiency(o)   for o in omp_outs]
omp_throughput = [parse_throughput(o)   for o in omp_outs]

mpi_time       = [parse_single_time(o)  for o in mpi_outs]
mpi_speedup    = [parse_speedup(o)      for o in mpi_outs]
mpi_efficiency = [parse_efficiency(o)   for o in mpi_outs]
mpi_throughput = [parse_throughput(o)   for o in mpi_outs]

# ── Summary table ─────────────────────────────────────────────
print("="*65)
print(f"{'Implementation':<18} {'W/P':<6} {'Time(s)':<10} {'Speedup':<10} {'Efficiency'}")
print("="*65)
print(f"{'Serial':<18} {'—':<6} {serial_time:<10.4f} {'1.00x':<10} {'100.0%'}")
for i, t in enumerate(WORKERS):
    print(f"{'OpenMP':<18} {t:<6} {omp_time[i]:<10.4f} {str(omp_speedup[i])+'x':<10} {str(omp_efficiency[i])+'%'}")
for i, p in enumerate(WORKERS):
    print(f"{'MPI':<18} {p:<6} {mpi_time[i]:<10.4f} {str(mpi_speedup[i])+'x':<10} {str(mpi_efficiency[i])+'%'}")
print("="*65)
print(f"\nAccuracy Metrics (Serial):")
for k, v in serial_metrics.items():
    print(f"  {k.capitalize():<12}: {v}")

# ══════════════════════════════════════════════════════════════
# STEP 3 — Generate Charts
# ══════════════════════════════════════════════════════════════

def apply_style(ax, title, xlabel, ylabel):
    ax.set_facecolor(CARD)
    ax.set_title(title, color=TEXT, fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel(xlabel, color=DIMTEXT, fontsize=10)
    ax.set_ylabel(ylabel, color=DIMTEXT, fontsize=10)
    ax.tick_params(colors=DIMTEXT, labelsize=9)
    ax.spines['bottom'].set_color(BORDER)
    ax.spines['left'].set_color(BORDER)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks(WORKERS)
    ax.grid(axis='y', color=GRID_C, linewidth=0.6, linestyle='--', alpha=0.8)
    ax.set_xlim(0.5, 8.5)

def add_labels(ax, x, y, color, fmt="{:.2f}"):
    for xi, yi in zip(x, y):
        ax.annotate(fmt.format(yi), xy=(xi, yi), xytext=(0, 8),
                    textcoords='offset points', ha='center', va='bottom',
                    color=color, fontsize=8, fontweight='bold')

x            = np.array(WORKERS)
width        = 0.35
ideal        = [1, 2, 4, 8]
omp_thr_m    = [t/1e6 for t in omp_throughput]
mpi_thr_m    = [t/1e6 for t in mpi_throughput]
serial_thr_m = serial_thr / 1e6

print("\nGenerating charts...")

# Chart 1: Speedup
fig, ax = plt.subplots(figsize=(8, 5)); fig.patch.set_facecolor(BG)
ax.plot(WORKERS, ideal, color=IDEAL_C, linewidth=1.2, linestyle='--', alpha=0.3, label='Ideal')
ax.plot(WORKERS, omp_speedup, color=OMP_C, linewidth=2.2, marker='o', markersize=7, label='OpenMP')
ax.plot(WORKERS, mpi_speedup, color=MPI_C, linewidth=2.2, marker='s', markersize=7, label='MPI')
ax.axhline(y=1.0, color=SERIAL_C, linewidth=1, linestyle=':', alpha=0.5)
ax.annotate('Serial (1.00x)', xy=(0.6, 1.08), color=SERIAL_C, fontsize=8)
add_labels(ax, WORKERS, omp_speedup, OMP_C, fmt="{:.2f}x")
add_labels(ax, WORKERS, mpi_speedup, MPI_C, fmt="{:.2f}x")
apply_style(ax, "Speedup vs Threads / Processes", "Threads / Processes", "Speedup (x)")
ax.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT, fontsize=9); ax.set_ylim(0, 9)
fig.tight_layout(); fig.savefig("charts/speedup.png", dpi=150, bbox_inches='tight', facecolor=BG); plt.close(fig)
print("  ✓ charts/speedup.png")

# Chart 2: Efficiency
fig, ax = plt.subplots(figsize=(8, 5)); fig.patch.set_facecolor(BG)
ax.axhline(y=100, color=IDEAL_C, linewidth=1.2, linestyle='--', alpha=0.3, label='Ideal (100%)')
ax.plot(WORKERS, omp_efficiency, color=OMP_C, linewidth=2.2, marker='o', markersize=7, label='OpenMP')
ax.plot(WORKERS, mpi_efficiency, color=MPI_C, linewidth=2.2, marker='s', markersize=7, label='MPI')
ax.axhspan(100, 125, alpha=0.04, color=GREEN_C)
ax.annotate('Super-linear region', xy=(0.6, 118), color=GREEN_C, fontsize=8, alpha=0.7)
add_labels(ax, WORKERS, omp_efficiency, OMP_C, fmt="{:.1f}%")
add_labels(ax, WORKERS, mpi_efficiency, MPI_C, fmt="{:.1f}%")
apply_style(ax, "Parallel Efficiency (%)", "Threads / Processes", "Efficiency (%)")
ax.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT, fontsize=9); ax.set_ylim(0, 130)
fig.tight_layout(); fig.savefig("charts/efficiency.png", dpi=150, bbox_inches='tight', facecolor=BG); plt.close(fig)
print("  ✓ charts/efficiency.png")

# Chart 3: Execution Time
fig, ax = plt.subplots(figsize=(8, 5)); fig.patch.set_facecolor(BG)
b1 = ax.bar(x-width/2, omp_time, width, color=OMP_C, alpha=0.75, label='OpenMP', zorder=3)
b2 = ax.bar(x+width/2, mpi_time, width, color=MPI_C, alpha=0.75, label='MPI',    zorder=3)
ax.axhline(y=serial_time, color=SERIAL_C, linewidth=1.5, linestyle='--', label=f'Serial ({serial_time}s)', zorder=4)
for b in b1: ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.003, f'{b.get_height():.4f}s', ha='center', va='bottom', color=OMP_C, fontsize=7.5, fontweight='bold')
for b in b2: ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.003, f'{b.get_height():.4f}s', ha='center', va='bottom', color=MPI_C, fontsize=7.5, fontweight='bold')
apply_style(ax, "Execution Time per Pass (seconds)", "Threads / Processes", "Time (seconds)")
ax.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT, fontsize=9)
fig.tight_layout(); fig.savefig("charts/execution_time.png", dpi=150, bbox_inches='tight', facecolor=BG); plt.close(fig)
print("  ✓ charts/execution_time.png")

# Chart 4: Throughput
fig, ax = plt.subplots(figsize=(8, 5)); fig.patch.set_facecolor(BG)
b1 = ax.bar(x-width/2, omp_thr_m, width, color=OMP_C, alpha=0.75, label='OpenMP', zorder=3)
b2 = ax.bar(x+width/2, mpi_thr_m, width, color=MPI_C, alpha=0.75, label='MPI',    zorder=3)
ax.axhline(y=serial_thr_m, color=SERIAL_C, linewidth=1.5, linestyle='--', label=f'Serial ({serial_thr_m:.2f}M rec/s)', zorder=4)
for b in b1: ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.05, f'{b.get_height():.2f}M', ha='center', va='bottom', color=OMP_C, fontsize=7.5, fontweight='bold')
for b in b2: ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.05, f'{b.get_height():.2f}M', ha='center', va='bottom', color=MPI_C, fontsize=7.5, fontweight='bold')
apply_style(ax, "Throughput (Million records / second)", "Threads / Processes", "Throughput (M rec/s)")
ax.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT, fontsize=9)
fig.tight_layout(); fig.savefig("charts/throughput.png", dpi=150, bbox_inches='tight', facecolor=BG); plt.close(fig)
print("  ✓ charts/throughput.png")

# Chart 5: Combined 2x2
fig, axes = plt.subplots(2, 2, figsize=(14, 9)); fig.patch.set_facecolor(BG)
fig.suptitle(
    "EC7207 — HPC Network Traffic Analysis: Performance Results\n"
    "Dataset: UNSW-NB15 · 700,001 records · Repeat x50 · "
    "EG/2021/4426 · EG/2021/4432 · EG/2021/4433",
    color=TEXT, fontsize=11, y=1.01
)
ax = axes[0][0]
ax.plot(WORKERS, ideal, color=IDEAL_C, linewidth=1, linestyle='--', alpha=0.3, label='Ideal')
ax.plot(WORKERS, omp_speedup, color=OMP_C, linewidth=2, marker='o', markersize=6, label='OpenMP')
ax.plot(WORKERS, mpi_speedup, color=MPI_C, linewidth=2, marker='s', markersize=6, label='MPI')
ax.axhline(y=1.0, color=SERIAL_C, linewidth=1, linestyle=':', alpha=0.5)
add_labels(ax, WORKERS, omp_speedup, OMP_C, fmt="{:.2f}x")
add_labels(ax, WORKERS, mpi_speedup, MPI_C, fmt="{:.2f}x")
apply_style(ax, "Speedup", "Threads / Processes", "Speedup (x)")
ax.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT, fontsize=8); ax.set_ylim(0, 9)

ax = axes[0][1]
ax.axhline(y=100, color=IDEAL_C, linewidth=1, linestyle='--', alpha=0.3, label='Ideal (100%)')
ax.plot(WORKERS, omp_efficiency, color=OMP_C, linewidth=2, marker='o', markersize=6, label='OpenMP')
ax.plot(WORKERS, mpi_efficiency, color=MPI_C, linewidth=2, marker='s', markersize=6, label='MPI')
ax.axhspan(100, 125, alpha=0.04, color=GREEN_C)
add_labels(ax, WORKERS, omp_efficiency, OMP_C, fmt="{:.1f}%")
add_labels(ax, WORKERS, mpi_efficiency, MPI_C, fmt="{:.1f}%")
apply_style(ax, "Parallel Efficiency (%)", "Threads / Processes", "Efficiency (%)")
ax.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT, fontsize=8); ax.set_ylim(0, 130)

ax = axes[1][0]
ax.bar(x-width/2, omp_time, width, color=OMP_C, alpha=0.75, label='OpenMP', zorder=3)
ax.bar(x+width/2, mpi_time, width, color=MPI_C, alpha=0.75, label='MPI',    zorder=3)
ax.axhline(y=serial_time, color=SERIAL_C, linewidth=1.5, linestyle='--', label=f'Serial ({serial_time}s)', zorder=4)
apply_style(ax, "Execution Time per Pass (s)", "Threads / Processes", "Time (seconds)")
ax.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)

ax = axes[1][1]
ax.bar(x-width/2, omp_thr_m, width, color=OMP_C, alpha=0.75, label='OpenMP', zorder=3)
ax.bar(x+width/2, mpi_thr_m, width, color=MPI_C, alpha=0.75, label='MPI',    zorder=3)
ax.axhline(y=serial_thr_m, color=SERIAL_C, linewidth=1.5, linestyle='--', label=f'Serial ({serial_thr_m:.2f}M)', zorder=4)
apply_style(ax, "Throughput (M rec/s)", "Threads / Processes", "Throughput (M rec/s)")
ax.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)

fig.tight_layout(); fig.savefig("charts/all_charts.png", dpi=150, bbox_inches='tight', facecolor=BG); plt.close(fig)
print("  ✓ charts/all_charts.png")

print("\nDone. All charts saved to charts/")