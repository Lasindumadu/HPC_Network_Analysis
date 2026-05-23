#!/usr/bin/env python3
"""
generate_charts3.py
EC7207 — HPC Network Traffic Analysis

Generates CPU-only performance charts (no CUDA).
  - Data source : results/logs/*.log  (saved during Run All in the UI)
  - CUDA        : intentionally excluded

Output: charts/speedup_v3.png  efficiency_v3.png  execution_time_v3.png
        charts/throughput_v3.png  all_charts_v3.png

Usage:
    python3 generate_charts3.py
"""

import os, re, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
BASE    = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE, 'results', 'logs')
CHARTS  = os.path.join(BASE, 'charts')
os.makedirs(CHARTS, exist_ok=True)

WORKERS        = [1, 2, 4, 8, 16]
HYBRID_CONFIGS = ["2x2","2x4","2x8","4x2","4x4","8x2","1x16","2x16","4x8","8x4","16x1"]

LOG_SERIAL = os.path.join(LOG_DIR, 'serial.log')
LOG_OMP    = {w: os.path.join(LOG_DIR, f'openmp_{w}t.log')   for w in WORKERS}
LOG_PTH    = {w: os.path.join(LOG_DIR, f'pthreads_{w}t.log') for w in WORKERS}
LOG_MPI    = {w: os.path.join(LOG_DIR, f'mpi_{w}p.log')      for w in WORKERS}
LOG_HYB    = {c: os.path.join(LOG_DIR,
              f"hybrid_{c.split('x')[0]}p_{c.split('x')[1]}t.log") for c in HYBRID_CONFIGS}

# ─────────────────────────────────────────────────────────────
# Colors — exact match to UI CSS / JS variables
# ─────────────────────────────────────────────────────────────
BG     = '#0a0e17'
CARD   = '#131a26'
CARD2  = '#192032'
BORDER = '#1e2d42'
GRID_C = '#21262d'
TEXT   = '#d4e0f0'
TEXT2  = '#7a94b0'
TEXT3  = '#3d526a'

SER_C  = '#6b7c93'
OMP_C  = '#58a6ff'
PTH_C  = '#bc8cff'
MPI_C  = '#db6d28'
HYB_C  = '#3fb950'

try:
    from matplotlib import font_manager as _fm
    _avail = {fp.name for fp in _fm.fontManager.ttflist}
    FONT = next((f for f in ('JetBrains Mono','Fira Code','DejaVu Sans Mono','Courier New')
                 if f in _avail), 'monospace')
except Exception:
    FONT = 'monospace'

# ─────────────────────────────────────────────────────────────
# Log helpers
# ─────────────────────────────────────────────────────────────

def read_log(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return ''

def parse_log(text):
    def get(pat):
        m = re.search(pat, text or '')
        return float(m.group(1)) if m else None
    thr_m = re.search(r'Throughput:\s+([\d,]+)\s+rec/s', text or '')
    return {
        'time':       get(r'Single-pass time:\s+([\d.]+)s'),
        'throughput': int(thr_m.group(1).replace(',','')) if thr_m else None,
        'speedup':    get(r'Speedup:\s+([\d.]+)x'),
        'efficiency': get(r'Efficiency:\s+([\d.]+)%'),
        'accuracy':   get(r'Accuracy:\s+([\d.]+)%'),
        'f1':         get(r'F1 Score:\s+([\d.]+)%'),
        'rmse':       get(r'RMSE:\s+([\d.]+)'),
    }

# ─────────────────────────────────────────────────────────────
# Load CPU results from log files
# ─────────────────────────────────────────────────────────────

serial_d = parse_log(read_log(LOG_SERIAL))
omp_d    = {w: parse_log(read_log(LOG_OMP[w])) for w in WORKERS}
pth_d    = {w: parse_log(read_log(LOG_PTH[w])) for w in WORKERS}
mpi_d    = {w: parse_log(read_log(LOG_MPI[w])) for w in WORKERS}

hyb_by_total = {}
for cfg in HYBRID_CONFIGS:
    np_v, nt_v = int(cfg.split('x')[0]), int(cfg.split('x')[1])
    total = np_v * nt_v
    d = parse_log(read_log(LOG_HYB[cfg]))
    if d['time'] is not None:
        prev = hyb_by_total.get(total)
        if prev is None or (d['speedup'] or 0) > (prev.get('speedup') or 0):
            hyb_by_total[total] = d

# ─────────────────────────────────────────────────────────────
# Style helpers
# ─────────────────────────────────────────────────────────────

def _style(ax, title, xlabel, ylabel, xticks, xlabels):
    ax.set_facecolor(CARD)
    ax.set_title(title, color=TEXT, fontsize=11, fontweight='bold', pad=10, fontfamily=FONT)
    ax.set_xlabel(xlabel, color=TEXT2, fontsize=9, fontfamily=FONT, labelpad=6)
    ax.set_ylabel(ylabel, color=TEXT2, fontsize=9, fontfamily=FONT, labelpad=6)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('bottom', 'left'):
        ax.spines[sp].set_color(BORDER)
        ax.spines[sp].set_linewidth(0.6)
    ax.yaxis.grid(True, color=GRID_C, linewidth=0.5, linestyle='--', alpha=0.9, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=TEXT3, labelsize=8, length=0)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, color=TEXT3, fontsize=8, fontfamily=FONT)
    ax.yaxis.set_tick_params(colors=TEXT3, labelsize=8)

def _legend(ax, **kw):
    ax.legend(facecolor=CARD2, edgecolor=BORDER, labelcolor=TEXT2,
              fontsize=8, framealpha=0.9, handlelength=1.6, **kw)

def _nan(lst):
    return [v if (v is not None and not (isinstance(v, float) and np.isnan(v))) else np.nan
            for v in lst]

def _vals(d_by_w, field, workers=WORKERS):
    return _nan([d_by_w.get(w, {}).get(field) for w in workers])

# ─────────────────────────────────────────────────────────────
# Chart drawing functions
# ─────────────────────────────────────────────────────────────

def draw_speedup(ax, workers, omp_d, pth_d, mpi_d, hyb_by_total):
    x = np.array(workers, dtype=float)

    # Ideal y=x line
    ax.plot(x, x, color='#ffffff', linewidth=1, linestyle='--',
            alpha=0.12, label='Ideal', zorder=1)

    # Serial constant 1.0
    ax.plot(x, np.ones_like(x), color=SER_C, linewidth=1.4, linestyle='--',
            dashes=(3, 3), marker='o', markersize=3, markeredgewidth=0,
            alpha=0.85, label='Serial (1×)', zorder=2)

    # CPU lines
    for d_dict, c, mk, lbl in [(omp_d, OMP_C,'o','OpenMP'),
                                 (pth_d, PTH_C,'^','Pthreads'),
                                 (mpi_d, MPI_C,'s','MPI')]:
        y = _vals(d_dict, 'speedup', workers)
        ax.plot(x, y, color=c, linewidth=2, marker=mk, markersize=5,
                markeredgewidth=0, label=lbl, zorder=4)

    # Hybrid (only where total workers ∈ workers list)
    hx = sorted(w for w in hyb_by_total if w in workers)
    if hx:
        hy = _nan([hyb_by_total[w].get('speedup') for w in hx])
        ax.plot(hx, hy, color=HYB_C, linewidth=2, marker='D', markersize=5,
                markeredgewidth=0, label='Hybrid', zorder=4)

    _style(ax, 'Speedup vs Workers (CPU only)', 'Workers (Threads / Processes)', 'Speedup (×)',
           x, [str(w) for w in workers])
    ax.set_ylim(bottom=0)
    _legend(ax)


def draw_efficiency(ax, workers, omp_d, pth_d, mpi_d, hyb_by_total):
    x = np.array(workers, dtype=float)

    # Serial constant 100%
    ax.plot(x, np.full_like(x, 100.0), color=SER_C, linewidth=1.4, linestyle='--',
            dashes=(3, 3), marker='o', markersize=3, markeredgewidth=0,
            alpha=0.85, label='Serial (100%)', zorder=2)

    for d_dict, c, mk, lbl in [(omp_d, OMP_C,'o','OpenMP'),
                                 (pth_d, PTH_C,'^','Pthreads'),
                                 (mpi_d, MPI_C,'s','MPI')]:
        y = _vals(d_dict, 'efficiency', workers)
        ax.plot(x, y, color=c, linewidth=2, marker=mk, markersize=5,
                markeredgewidth=0, label=lbl, zorder=4)

    hx = sorted(w for w in hyb_by_total if w in workers)
    if hx:
        hy = _nan([hyb_by_total[w].get('efficiency') for w in hx])
        ax.plot(hx, hy, color=HYB_C, linewidth=2, marker='D', markersize=5,
                markeredgewidth=0, label='Hybrid', zorder=4)

    _style(ax, 'Parallel Efficiency (%) — CPU only', 'Workers (Threads / Processes)', 'Efficiency (%)',
           x, [str(w) for w in workers])
    ax.set_ylim(bottom=0)
    _legend(ax)


def draw_bar_chart(ax, workers, serial_d, omp_d, pth_d, mpi_d,
                   hyb_by_total, field, title, ylabel):
    x_labels = [str(w) for w in workers]
    xi       = np.arange(len(x_labels))
    bw       = 0.14
    # 5 implementations: Serial, OpenMP, Pthreads, MPI, Hybrid
    cpu_off  = np.array([-2, -1, 0, 1, 2], dtype=float) * bw

    # Serial — only at workers[0] (index 0, slot -2)
    sv = serial_d.get(field)
    if sv:
        ax.bar(xi[0] + cpu_off[0], sv, bw, color=SER_C, alpha=0.85,
               label='Serial', zorder=3)

    # OpenMP, Pthreads, MPI
    for slot, (d_by_w, c, lbl) in enumerate(
            [(omp_d, OMP_C, 'OpenMP'),
             (pth_d, PTH_C, 'Pthreads'),
             (mpi_d, MPI_C, 'MPI')], start=1):
        first = True
        for i, w in enumerate(workers):
            v = d_by_w.get(w, {}).get(field)
            if v:
                ax.bar(xi[i] + cpu_off[slot], v, bw, color=c, alpha=0.85,
                       label=lbl if first else '', zorder=3)
                first = False

    # Hybrid
    first = True
    for i, w in enumerate(workers):
        if w in hyb_by_total:
            v = hyb_by_total[w].get(field)
            if v:
                ax.bar(xi[i] + cpu_off[4], v, bw, color=HYB_C, alpha=0.85,
                       label='Hybrid' if first else '', zorder=3)
                first = False

    _style(ax, title, 'Workers (Threads / Processes)', ylabel, xi, x_labels)
    ax.set_ylim(bottom=0)
    _legend(ax)


# ─────────────────────────────────────────────────────────────
# Generate all charts
# ─────────────────────────────────────────────────────────────

print('Generating CPU-only charts (no CUDA)...')

SUPTITLE = (
    "EC7207 — HPC Network Traffic Analysis: CPU Performance Results\n"
    "Dataset: UNSW-NB15 · 700,001 records · Repeat ×50 · "
    "EG/2021/4426 · EG/2021/4432 · EG/2021/4433"
)

def _speedup(ax):
    draw_speedup(ax, WORKERS, omp_d, pth_d, mpi_d, hyb_by_total)
def _efficiency(ax):
    draw_efficiency(ax, WORKERS, omp_d, pth_d, mpi_d, hyb_by_total)
def _time(ax):
    draw_bar_chart(ax, WORKERS, serial_d, omp_d, pth_d, mpi_d, hyb_by_total,
                   'time', 'Execution Time (seconds / pass) — CPU only', 'Time (seconds)')
def _throughput(ax):
    draw_bar_chart(ax, WORKERS, serial_d, omp_d, pth_d, mpi_d, hyb_by_total,
                   'throughput', 'Throughput (records / sec) — CPU only', 'Throughput (rec/s)')

# Individual PNG files
for fn, name in [(_speedup,'speedup_v3'), (_efficiency,'efficiency_v3'),
                 (_time,'execution_time_v3'), (_throughput,'throughput_v3')]:
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=BG)
    fn(ax)
    fig.tight_layout(pad=1.5)
    out = os.path.join(CHARTS, f'{name}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f'  ✓ charts/{name}.png')

# Combined 2×2 dashboard
fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor=BG)
fig.patch.set_facecolor(BG)
fig.suptitle(SUPTITLE, color=TEXT2, fontsize=10, fontfamily=FONT, y=1.015)
_speedup(axes[0][0])
_efficiency(axes[0][1])
_time(axes[1][0])
_throughput(axes[1][1])
fig.tight_layout(pad=2.2)
out = os.path.join(CHARTS, 'all_charts_v3.png')
fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
plt.close(fig)
print('  ✓ charts/all_charts_v3.png')

print('\nDone. CPU-only charts saved to charts/')
