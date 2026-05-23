#!/usr/bin/env python3
"""
generate_charts2.py
EC7207 — HPC Network Traffic Analysis

Generates report-quality charts using:
  - CPU results  : results/logs/*.log  (saved during Run All in the UI)
  - CUDA results : hardcoded Tesla T4 values from Google Colab
                   (VMware cannot run CUDA — real GPU results captured separately)

CUDA benchmarks (Tesla T4, Google Colab):
    Single-pass time : 0.0022 s
    Speedup          : 532.40× vs serial (serial baseline on same machine: 1.1580 s)
    Throughput       : 321,824,052 rec/s
    Block size       : 256  |  Grid: 2735 × 256 = 700,160 threads

Output: charts/speedup_v2.png  efficiency_v2.png  execution_time_v2.png
        charts/throughput_v2.png  all_charts_v2.png

Usage:
    python3 generate_charts2.py
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
# Hardcoded CUDA values — Tesla T4, Google Colab
# ─────────────────────────────────────────────────────────────
CUDA_D = {
    'time':       0.0022,
    'throughput': 321_824_052,
    'speedup':    532.40,
    'efficiency': None,       # not applicable for GPU
    'accuracy':   97.072,
    'f1':         58.518,
    'rmse':       0.171126,
}
CUDA_LABEL = 'CUDA (Tesla T4)'

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
CUDA_C = '#e3b341'

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
# Style helpers (identical to generate_charts.py)
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

    # Collect all CPU speedup values for y-axis limit
    all_cpu = [v for d in [omp_d.get(w,{}) for w in workers] + [pth_d.get(w,{}) for w in workers] +
               [mpi_d.get(w,{}) for w in workers] + list(hyb_by_total.values())
               for v in [d.get('speedup')] if v]
    cpu_max = max(all_cpu, default=10)

    # Ideal line (capped to CPU range for readability)
    ax.plot(x, np.minimum(x, cpu_max * 1.5), color='#ffffff', linewidth=1,
            linestyle='--', alpha=0.12, label='Ideal', zorder=1)

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

    hx = sorted(w for w in hyb_by_total if w in workers)
    if hx:
        hy = _nan([hyb_by_total[w].get('speedup') for w in hx])
        ax.plot(hx, hy, color=HYB_C, linewidth=2, marker='D', markersize=5,
                markeredgewidth=0, label='Hybrid', zorder=4)

    # CUDA — hardcoded 532.40× — annotated at top of visible area
    cuda_sp = CUDA_D['speedup']
    y_top   = cpu_max * 1.45
    ax.set_ylim(0, y_top)

    # Draw star at the top edge and annotate with a label box
    star_y = y_top * 0.90
    ax.scatter([x[-1]], [star_y], marker='*', color=CUDA_C, s=220, zorder=7,
               label=f'{CUDA_LABEL}: {cuda_sp:.2f}×')

    # Upward arrow indicating CUDA is off-scale
    ax.annotate(
        f'★ {cuda_sp:.2f}×\n(Tesla T4)',
        xy=(x[-1], star_y),
        xytext=(x[-1] - 1.4, star_y * 0.88),
        color=CUDA_C, fontsize=8, fontweight='bold', fontfamily=FONT,
        arrowprops=dict(arrowstyle='->', color=CUDA_C, lw=1.2),
        bbox=dict(boxstyle='round,pad=0.3', facecolor=CARD, edgecolor=CUDA_C,
                  alpha=0.9, linewidth=0.8),
    )

    _style(ax, 'Speedup vs Workers', 'Workers (Threads / Processes)', 'Speedup (×)',
           x, [str(w) for w in workers])
    _legend(ax)


def draw_efficiency(ax, workers, omp_d, pth_d, mpi_d, hyb_by_total):
    x = np.array(workers, dtype=float)

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

    _style(ax, 'Parallel Efficiency (%)', 'Workers (Threads / Processes)', 'Efficiency (%)',
           x, [str(w) for w in workers])
    ax.set_ylim(bottom=0)
    _legend(ax)


def draw_bar_chart(ax, workers, serial_d, omp_d, pth_d, mpi_d,
                   hyb_by_total, cuda_d, field, title, ylabel):
    x_labels = [str(w) for w in workers] + ['GPU']
    xi       = np.arange(len(x_labels))
    bw       = 0.14
    cpu_off  = np.array([-2, -1, 0, 1, 2], dtype=float) * bw

    # Serial
    sv = serial_d.get(field)
    if sv:
        ax.bar(xi[0] + cpu_off[0], sv, bw, color=SER_C, alpha=0.85,
               label='Serial', zorder=3)

    for slot, (d_by_w, c, lbl) in enumerate(
            [(omp_d, OMP_C,'OpenMP'), (pth_d, PTH_C,'Pthreads'), (mpi_d, MPI_C,'MPI')], 1):
        first = True
        for i, w in enumerate(workers):
            v = d_by_w.get(w, {}).get(field)
            if v:
                ax.bar(xi[i] + cpu_off[slot], v, bw, color=c, alpha=0.85,
                       label=lbl if first else '', zorder=3)
                first = False

    first = True
    for i, w in enumerate(workers):
        if w in hyb_by_total:
            v = hyb_by_total[w].get(field)
            if v:
                ax.bar(xi[i] + cpu_off[4], v, bw, color=HYB_C, alpha=0.85,
                       label='Hybrid' if first else '', zorder=3)
                first = False

    cv = cuda_d.get(field)
    if cv:
        ax.bar(xi[-1], cv, bw * 2.2, color=CUDA_C, alpha=0.85,
               label=CUDA_LABEL, zorder=3)

    _style(ax, title, 'Workers (Threads / Processes)', ylabel, xi, x_labels)
    ax.set_ylim(bottom=0)
    _legend(ax)


# ─────────────────────────────────────────────────────────────
# Generate all charts
# ─────────────────────────────────────────────────────────────

print('Generating report charts (with Tesla T4 CUDA data)...')

SUPTITLE = (
    "EC7207 — HPC Network Traffic Analysis: Performance Results\n"
    "Dataset: UNSW-NB15 · 700,001 records · Repeat ×50 · "
    "EG/2021/4426 · EG/2021/4432 · EG/2021/4433 · CUDA: Tesla T4 (Google Colab)"
)

def _speedup(ax):
    draw_speedup(ax, WORKERS, omp_d, pth_d, mpi_d, hyb_by_total)
def _efficiency(ax):
    draw_efficiency(ax, WORKERS, omp_d, pth_d, mpi_d, hyb_by_total)
def _time(ax):
    draw_bar_chart(ax, WORKERS, serial_d, omp_d, pth_d, mpi_d, hyb_by_total, CUDA_D,
                   'time', 'Execution Time (seconds / pass)', 'Time (seconds)')
def _throughput(ax):
    draw_bar_chart(ax, WORKERS, serial_d, omp_d, pth_d, mpi_d, hyb_by_total, CUDA_D,
                   'throughput', 'Throughput (records / sec)', 'Throughput (rec/s)')

for fn, name in [(_speedup,'speedup_v2'), (_efficiency,'efficiency_v2'),
                 (_time,'execution_time_v2'), (_throughput,'throughput_v2')]:
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
out = os.path.join(CHARTS, 'all_charts_v2.png')
fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
plt.close(fig)
print('  ✓ charts/all_charts_v2.png')

print('\nDone. Report charts saved to charts/')
print(f'  CUDA: {CUDA_LABEL} — {CUDA_D["speedup"]:.2f}× speedup, '
      f'{CUDA_D["throughput"]/1e6:.2f}M rec/s, {CUDA_D["time"]*1000:.2f} ms/pass')
