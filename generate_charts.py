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
PTHREADS = "./results/pthreads"
MPI      = "./results/mpi"
HYBRID   = "./results/hybrid"
WORKERS  = [1, 2, 4, 8, 16]
LOG_DIR  = "results/logs"
os.makedirs("charts",  exist_ok=True)
os.makedirs(LOG_DIR,   exist_ok=True)

# Log file paths — one per run
LOG_SERIAL   = f"{LOG_DIR}/serial.log"
LOG_OMP      = [f"{LOG_DIR}/openmp_{t}t.log"  for t in WORKERS]
LOG_PTHREAD  = [f"{LOG_DIR}/pthreads_{t}t.log" for t in WORKERS]
LOG_MPI      = [f"{LOG_DIR}/mpi_{p}p.log"     for p in WORKERS]

# Hybrid configs (np x nt) and their total parallelism
HYBRID_CONFIGS = ["2x2", "2x4", "2x8", "4x2", "4x4", "8x2", "1x16", "2x16", "4x8", "8x4", "16x1"]
LOG_HYBRID = {cfg: f"{LOG_DIR}/hybrid_{cfg.split('x')[0]}p_{cfg.split('x')[1]}t.log" for cfg in HYBRID_CONFIGS}

# ── Style ─────────────────────────────────────────────────────
BG       = "#0b0f14"
CARD     = "#111820"
BORDER   = "#1e2a38"
TEXT     = "#e0e6f0"
DIMTEXT  = "#5a6a7a"
SERIAL_C = "#8b949e"
OMP_C    = "#00d4ff"
PTHREAD_C= "#a371f7"   # purple
MPI_C    = "#ff6b35"
HYBRID_C = "#3fb950"   # green
IDEAL_C  = "#ffffff"
GRID_C   = "#1e2a38"

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
    all_logs = [LOG_SERIAL] + LOG_OMP + LOG_PTHREAD + LOG_MPI + list(LOG_HYBRID.values())
    return all(os.path.exists(p) for p in all_logs)

def parse_single_time(output):
    m = re.search(r"Single-pass time:\s+([\d.]+)s", output)
    return float(m.group(1)) if m else None

def parse_throughput(output):
    m = re.search(r"Throughput:\s+([\d,]+)\s+rec/s", output)
    if m:
        return int(m.group(1).replace(',', ''))
    # fallback: try without comma
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
        ("rmse",      r"RMSE:\s+([\d.]+)"),
    ]:
        m = re.search(pattern, output)
        metrics[key] = float(m.group(1)) if m else None
    return metrics

def parse_hybrid_config(cfg):
    """Parse '2x4' into (np, nt, total)."""
    np_val = int(cfg.split('x')[0])
    nt_val = int(cfg.split('x')[1])
    return np_val, nt_val, np_val * nt_val

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

    print("[1/5] Running Serial...")
    serial_out = run_and_save([SERIAL, DATASET], LOG_SERIAL)

    print("\n[2/5] Running OpenMP (1, 2, 4, 8, 16 threads)...")
    omp_outs = []
    for i, t in enumerate(WORKERS):
        env = base_env.copy()
        env["OMP_NUM_THREADS"] = str(t)
        omp_outs.append(run_and_save([OPENMP, DATASET], LOG_OMP[i], env=env))

    print("\n[3/5] Running Pthreads (1, 2, 4, 8, 16 threads)...")
    pthread_outs = []
    for i, t in enumerate(WORKERS):
        pthread_outs.append(run_and_save([PTHREADS, DATASET, str(t)], LOG_PTHREAD[i]))

    print("\n[4/5] Running MPI (1, 2, 4, 8, 16 processes)...")
    mpi_outs = []
    for i, p in enumerate(WORKERS):
        mpi_outs.append(run_and_save(
            ["mpirun", "--allow-run-as-root", "--oversubscribe",
             "-np", str(p), MPI, DATASET],
            LOG_MPI[i]
        ))

    print("\n[5/5] Running Hybrid MPI + OpenMP...")
    hybrid_outs = {}
    for cfg in HYBRID_CONFIGS:
        np_val, nt_val, total = parse_hybrid_config(cfg)
        env = base_env.copy()
        env["OMP_NUM_THREADS"] = str(nt_val)
        hybrid_outs[cfg] = run_and_save(
            ["mpirun", "--allow-run-as-root", "--oversubscribe",
             "-np", str(np_val), HYBRID, DATASET],
            LOG_HYBRID[cfg], env=env
        )
    print(f"\nLogs saved to {LOG_DIR}/")

else:
    print(f"Reading saved logs from {LOG_DIR}/ (use --rerun to re-execute)\n")
    serial_out = read_log(LOG_SERIAL)
    omp_outs   = [read_log(p) for p in LOG_OMP]
    pthread_outs = [read_log(p) for p in LOG_PTHREAD]
    mpi_outs   = [read_log(p) for p in LOG_MPI]
    hybrid_outs = {cfg: read_log(LOG_HYBRID[cfg]) for cfg in HYBRID_CONFIGS}

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

pthread_time       = [parse_single_time(o)  for o in pthread_outs]
pthread_speedup    = [parse_speedup(o)      for o in pthread_outs]
pthread_efficiency = [parse_efficiency(o)   for o in pthread_outs]
pthread_throughput = [parse_throughput(o)   for o in pthread_outs]

mpi_time       = [parse_single_time(o)  for o in mpi_outs]
mpi_speedup    = [parse_speedup(o)      for o in mpi_outs]
mpi_efficiency = [parse_efficiency(o)   for o in mpi_outs]
mpi_throughput = [parse_throughput(o)   for o in mpi_outs]

# Parse hybrid — group by total parallelism
hybrid_by_total = {}
for cfg in HYBRID_CONFIGS:
    out = hybrid_outs.get(cfg, "")
    t = parse_single_time(out)
    sp = parse_speedup(out)
    ef = parse_efficiency(out)
    th = parse_throughput(out)
    np_val, nt_val, total = parse_hybrid_config(cfg)
    if total not in hybrid_by_total or (sp and sp > hybrid_by_total[total].get('speedup', 0)):
        hybrid_by_total[total] = {
            'time': t, 'speedup': sp, 'efficiency': ef,
            'throughput': th, 'config': cfg
        }

# Extract hybrid data for plotting (at totals 4, 8, 16)
hybrid_totals = sorted([t for t in hybrid_by_total.keys() if t > 1])
hybrid_time_plot      = [hybrid_by_total[t]['time']      for t in hybrid_totals]
hybrid_speedup_plot   = [hybrid_by_total[t]['speedup']   for t in hybrid_totals]
hybrid_efficiency_plot= [hybrid_by_total[t]['efficiency']for t in hybrid_totals]
hybrid_throughput_plot= [hybrid_by_total[t]['throughput']for t in hybrid_totals]

# ── Summary table ─────────────────────────────────────────────
print("="*75)
print(f"{'Implementation':<18} {'W/P':<8} {'Time(s)':<10} {'Speedup':<10} {'Efficiency':<12} {'Throughput'}")
print("="*75)
print(f"{'Serial':<18} {'—':<8} {serial_time:<10.4f} {'1.00x':<10} {'100.0%':<12} {serial_thr:,} rec/s")
for i, t in enumerate(WORKERS):
    thr = omp_throughput[i] if omp_throughput[i] else 0
    print(f"{'OpenMP':<18} {t:<8} {omp_time[i]:<10.4f} {str(omp_speedup[i])+'x':<10} {str(omp_efficiency[i])+'%':<12} {thr:,} rec/s")
for i, t in enumerate(WORKERS):
    thr = pthread_throughput[i] if pthread_throughput[i] else 0
    print(f"{'Pthreads':<18} {t:<8} {pthread_time[i]:<10.4f} {str(pthread_speedup[i])+'x':<10} {str(pthread_efficiency[i])+'%':<12} {thr:,} rec/s")
for i, p in enumerate(WORKERS):
    thr = mpi_throughput[i] if mpi_throughput[i] else 0
    print(f"{'MPI':<18} {p:<8} {mpi_time[i]:<10.4f} {str(mpi_speedup[i])+'x':<10} {str(mpi_efficiency[i])+'%':<12} {thr:,} rec/s")
for total in hybrid_totals:
    h = hybrid_by_total[total]
    cfg = h['config']
    thr = h['throughput'] if h['throughput'] else 0
    print(f"{'Hybrid('+cfg+')':<18} {total:<8} {h['time']:<10.4f} {str(h['speedup'])+'x':<10} {str(h['efficiency'])+'%':<12} {thr:,} rec/s")
print("="*75)
print(f"\nAccuracy Metrics (Serial):")
for k, v in serial_metrics.items():
    if v is not None:
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

def add_labels(ax, x, y, color, fmt="{:.2f}"):
    for xi, yi in zip(x, y):
        if yi is not None:
            ax.annotate(fmt.format(yi), xy=(xi, yi), xytext=(0, 8),
                        textcoords='offset points', ha='center', va='bottom',
                        color=color, fontsize=8, fontweight='bold')

x            = np.array(WORKERS)
width        = 0.25
ideal        = WORKERS
omp_thr_m    = [t/1e6 for t in omp_throughput]
pthread_thr_m= [t/1e6 for t in pthread_throughput]
mpi_thr_m    = [t/1e6 for t in mpi_throughput]
serial_thr_m = serial_thr / 1e6 if serial_thr else 0

print("\nGenerating charts...")

# Chart 1: Speedup
fig, ax = plt.subplots(figsize=(10, 6)); fig.patch.set_facecolor(BG)
ax.plot(WORKERS, ideal, color=IDEAL_C, linewidth=1.2, linestyle='--', alpha=0.3, label='Ideal')
ax.plot(WORKERS, omp_speedup, color=OMP_C, linewidth=2.2, marker='o', markersize=7, label='OpenMP')
ax.plot(WORKERS, pthread_speedup, color=PTHREAD_C, linewidth=2.2, marker='^', markersize=7, label='Pthreads')
ax.plot(WORKERS, mpi_speedup, color=MPI_C, linewidth=2.2, marker='s', markersize=7, label='MPI')
if hybrid_speedup_plot:
    ax.plot(hybrid_totals, hybrid_speedup_plot, color=HYBRID_C, linewidth=2.2, marker='D', markersize=7, label='Hybrid', linestyle='-.')
ax.axhline(y=1.0, color=SERIAL_C, linewidth=1, linestyle=':', alpha=0.5)
ax.annotate('Serial (1.00x)', xy=(0.8, 1.08), color=SERIAL_C, fontsize=8)
add_labels(ax, WORKERS, omp_speedup, OMP_C, fmt="{:.2f}x")
add_labels(ax, WORKERS, pthread_speedup, PTHREAD_C, fmt="{:.2f}x")
add_labels(ax, WORKERS, mpi_speedup, MPI_C, fmt="{:.2f}x")
apply_style(ax, "Speedup vs Threads / Processes", "Threads / Processes", "Speedup (x)")
ax.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT, fontsize=9); ax.set_ylim(0, 18)
fig.tight_layout(); fig.savefig("charts/speedup.png", dpi=150, bbox_inches='tight', facecolor=BG); plt.close(fig)
print("  ✓ charts/speedup.png")

# Chart 2: Efficiency
fig, ax = plt.subplots(figsize=(10, 6)); fig.patch.set_facecolor(BG)
ax.axhline(y=100, color=IDEAL_C, linewidth=1.2, linestyle='--', alpha=0.3, label='Ideal (100%)')
ax.plot(WORKERS, omp_efficiency, color=OMP_C, linewidth=2.2, marker='o', markersize=7, label='OpenMP')
ax.plot(WORKERS, pthread_efficiency, color=PTHREAD_C, linewidth=2.2, marker='^', markersize=7, label='Pthreads')
ax.plot(WORKERS, mpi_efficiency, color=MPI_C, linewidth=2.2, marker='s', markersize=7, label='MPI')
if hybrid_efficiency_plot:
    ax.plot(hybrid_totals, hybrid_efficiency_plot, color=HYBRID_C, linewidth=2.2, marker='D', markersize=7, label='Hybrid', linestyle='-.')
ax.axhspan(100, 125, alpha=0.04, color=HYBRID_C)
ax.annotate('Super-linear region', xy=(0.8, 118), color=HYBRID_C, fontsize=8, alpha=0.7)
add_labels(ax, WORKERS, omp_efficiency, OMP_C, fmt="{:.1f}%")
add_labels(ax, WORKERS, pthread_efficiency, PTHREAD_C, fmt="{:.1f}%")
add_labels(ax, WORKERS, mpi_efficiency, MPI_C, fmt="{:.1f}%")
apply_style(ax, "Parallel Efficiency (%)", "Threads / Processes", "Efficiency (%)")
ax.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT, fontsize=9); ax.set_ylim(0, 130)
fig.tight_layout(); fig.savefig("charts/efficiency.png", dpi=150, bbox_inches='tight', facecolor=BG); plt.close(fig)
print("  ✓ charts/efficiency.png")

# Chart 3: Execution Time
fig, ax = plt.subplots(figsize=(10, 6)); fig.patch.set_facecolor(BG)
b1 = ax.bar(x-width, omp_time, width, color=OMP_C, alpha=0.75, label='OpenMP', zorder=3)
b2 = ax.bar(x, pthread_time, width, color=PTHREAD_C, alpha=0.75, label='Pthreads', zorder=3)
b3 = ax.bar(x+width, mpi_time, width, color=MPI_C, alpha=0.75, label='MPI', zorder=3)
ax.axhline(y=serial_time, color=SERIAL_C, linewidth=1.5, linestyle='--', label=f'Serial ({serial_time:.4f}s)', zorder=4)
for b in b1: ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.003, f'{b.get_height():.4f}s', ha='center', va='bottom', color=OMP_C, fontsize=7, fontweight='bold')
for b in b2: ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.003, f'{b.get_height():.4f}s', ha='center', va='bottom', color=PTHREAD_C, fontsize=7, fontweight='bold')
for b in b3: ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.003, f'{b.get_height():.4f}s', ha='center', va='bottom', color=MPI_C, fontsize=7, fontweight='bold')
apply_style(ax, "Execution Time per Pass (seconds)", "Threads / Processes", "Time (seconds)")
ax.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT, fontsize=9)
fig.tight_layout(); fig.savefig("charts/execution_time.png", dpi=150, bbox_inches='tight', facecolor=BG); plt.close(fig)
print("  ✓ charts/execution_time.png")

# Chart 4: Throughput
fig, ax = plt.subplots(figsize=(10, 6)); fig.patch.set_facecolor(BG)
b1 = ax.bar(x-width, omp_thr_m, width, color=OMP_C, alpha=0.75, label='OpenMP', zorder=3)
b2 = ax.bar(x, pthread_thr_m, width, color=PTHREAD_C, alpha=0.75, label='Pthreads', zorder=3)
b3 = ax.bar(x+width, mpi_thr_m, width, color=MPI_C, alpha=0.75, label='MPI', zorder=3)
ax.axhline(y=serial_thr_m, color=SERIAL_C, linewidth=1.5, linestyle='--', label=f'Serial ({serial_thr_m:.2f}M rec/s)', zorder=4)
for b in b1: ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.05, f'{b.get_height():.2f}M', ha='center', va='bottom', color=OMP_C, fontsize=7, fontweight='bold')
for b in b2: ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.05, f'{b.get_height():.2f}M', ha='center', va='bottom', color=PTHREAD_C, fontsize=7, fontweight='bold')
for b in b3: ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.05, f'{b.get_height():.2f}M', ha='center', va='bottom', color=MPI_C, fontsize=7, fontweight='bold')
apply_style(ax, "Throughput (Million records / second)", "Threads / Processes", "Throughput (M rec/s)")
ax.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT, fontsize=9)
fig.tight_layout(); fig.savefig("charts/throughput.png", dpi=150, bbox_inches='tight', facecolor=BG); plt.close(fig)
print("  ✓ charts/throughput.png")

# Chart 5: Combined 2x2
fig, axes = plt.subplots(2, 2, figsize=(14, 10)); fig.patch.set_facecolor(BG)
fig.suptitle(
    "EC7207 — HPC Network Traffic Analysis: Performance Results\n"
    "Dataset: UNSW-NB15 · 700,001 records · Repeat x50 · "
    "EG/2021/4426 · EG/2021/4432 · EG/2021/4433",
    color=TEXT, fontsize=11, y=1.01
)
ax = axes[0][0]
ax.plot(WORKERS, ideal, color=IDEAL_C, linewidth=1, linestyle='--', alpha=0.3, label='Ideal')
ax.plot(WORKERS, omp_speedup, color=OMP_C, linewidth=2, marker='o', markersize=6, label='OpenMP')
ax.plot(WORKERS, pthread_speedup, color=PTHREAD_C, linewidth=2, marker='^', markersize=6, label='Pthreads')
ax.plot(WORKERS, mpi_speedup, color=MPI_C, linewidth=2, marker='s', markersize=6, label='MPI')
if hybrid_speedup_plot:
    ax.plot(hybrid_totals, hybrid_speedup_plot, color=HYBRID_C, linewidth=2, marker='D', markersize=6, label='Hybrid', linestyle='-.')
ax.axhline(y=1.0, color=SERIAL_C, linewidth=1, linestyle=':', alpha=0.5)
add_labels(ax, WORKERS, omp_speedup, OMP_C, fmt="{:.2f}x")
add_labels(ax, WORKERS, pthread_speedup, PTHREAD_C, fmt="{:.2f}x")
add_labels(ax, WORKERS, mpi_speedup, MPI_C, fmt="{:.2f}x")
apply_style(ax, "Speedup", "Threads / Processes", "Speedup (x)")
ax.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT, fontsize=8); ax.set_ylim(0, 18)

ax = axes[0][1]
ax.axhline(y=100, color=IDEAL_C, linewidth=1, linestyle='--', alpha=0.3, label='Ideal (100%)')
ax.plot(WORKERS, omp_efficiency, color=OMP_C, linewidth=2, marker='o', markersize=6, label='OpenMP')
ax.plot(WORKERS, pthread_efficiency, color=PTHREAD_C, linewidth=2, marker='^', markersize=6, label='Pthreads')
ax.plot(WORKERS, mpi_efficiency, color=MPI_C, linewidth=2, marker='s', markersize=6, label='MPI')
if hybrid_efficiency_plot:
    ax.plot(hybrid_totals, hybrid_efficiency_plot, color=HYBRID_C, linewidth=2, marker='D', markersize=6, label='Hybrid', linestyle='-.')
ax.axhspan(100, 125, alpha=0.04, color=HYBRID_C)
add_labels(ax, WORKERS, omp_efficiency, OMP_C, fmt="{:.1f}%")
add_labels(ax, WORKERS, pthread_efficiency, PTHREAD_C, fmt="{:.1f}%")
add_labels(ax, WORKERS, mpi_efficiency, MPI_C, fmt="{:.1f}%")
apply_style(ax, "Parallel Efficiency (%)", "Threads / Processes", "Efficiency (%)")
ax.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT, fontsize=8); ax.set_ylim(0, 130)

ax = axes[1][0]
ax.bar(x-width, omp_time, width, color=OMP_C, alpha=0.75, label='OpenMP', zorder=3)
ax.bar(x, pthread_time, width, color=PTHREAD_C, alpha=0.75, label='Pthreads', zorder=3)
ax.bar(x+width, mpi_time, width, color=MPI_C, alpha=0.75, label='MPI', zorder=3)
ax.axhline(y=serial_time, color=SERIAL_C, linewidth=1.5, linestyle='--', label=f'Serial ({serial_time:.4f}s)', zorder=4)
apply_style(ax, "Execution Time per Pass (s)", "Threads / Processes", "Time (seconds)")
ax.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)

ax = axes[1][1]
ax.bar(x-width, omp_thr_m, width, color=OMP_C, alpha=0.75, label='OpenMP', zorder=3)
ax.bar(x, pthread_thr_m, width, color=PTHREAD_C, alpha=0.75, label='Pthreads', zorder=3)
ax.bar(x+width, mpi_thr_m, width, color=MPI_C, alpha=0.75, label='MPI', zorder=3)
ax.axhline(y=serial_thr_m, color=SERIAL_C, linewidth=1.5, linestyle='--', label=f'Serial ({serial_thr_m:.2f}M)', zorder=4)
apply_style(ax, "Throughput (M rec/s)", "Threads / Processes", "Throughput (M rec/s)")
ax.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)

fig.tight_layout(); fig.savefig("charts/all_charts.png", dpi=150, bbox_inches='tight', facecolor=BG); plt.close(fig)
print("  ✓ charts/all_charts.png")

print("\nDone. All charts saved to charts/")
